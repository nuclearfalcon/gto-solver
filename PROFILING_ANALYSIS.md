# Phase 6 Profiling Analysis

**Date:** 2025-11-02
**Goal:** Identify bottlenecks preventing Hold'em scaling

---

## Baseline Performance

| Game | Nodes | Speed | Time/Iteration | Time/1000 Iterations |
|------|-------|-------|----------------|----------------------|
| **Kuhn (sparse)** | 58 | 6.13 it/s | 0.163s | 2.7 minutes |
| **Leduc (sparse)** | 9,457 | 0.14 it/s | 7.38s | **123 minutes (2.0 hours)** |

**Target:** 1.0 it/s on Leduc (7.4x speedup needed)

---

## Critical Bottlenecks (cProfile Analysis)

### Top Time Consumers (10 Leduc iterations, 96.45s total)

| Function | Total Time | Per Call | % of Total | Impact |
|----------|------------|----------|------------|--------|
| **Scatter operations** | 51.6s | 2.58s | **53%** | 🔥🔥🔥 CRITICAL |
| **Batch bottom-up** | 41.7s | 2.08s | **43%** | 🔥🔥 HIGH |
| **Array indexing** | 44.7s | - | **46%** | 🔥🔥 HIGH |
| **Update cumulative strategy** | 48.0s | 2.40s | **50%** | 🔥🔥🔥 CRITICAL |
| **Regret updates** | 51.6s | 2.58s | **53%** | 🔥🔥🔥 CRITICAL |

**Note:** These overlap significantly - scatter and indexing are called FROM the update functions.

### Detailed Breakdown

#### 1. Scatter Operations (53% of time) 🔥🔥🔥

```
_update_regrets_and_strategy:     51.62s (2.58s × 20 calls)
  └─ _scatter_update:             51.58s
      └─ _scatter_impl:           45.04s (JAX scatter on GPU)
```

**Problem:** Using `array.at[indices].set(values)` for incremental updates
- Each infoset-action update is a separate scatter operation
- JAX scatter on GPU has high overhead for small batches
- Called 22,988 times across all iterations

**Root Cause:** Lines in `matrix_cfr_solver.py`:
- `_update_regrets_and_strategy()`: Line 1715
- `_update_cumulative_strategy()`: Line 1783

Both use similar scatter patterns.

#### 2. Array Indexing (46% of time) 🔥🔥

```
Array.__getitem__:                44.71s (73,750 calls)
  └─ rewriting_take:              43.69s
      └─ _attempt_rewriting_take: 33.71s
          └─ index_to_gather:     33.17s
```

**Problem:** Too many small array access operations
- 73,750 __getitem__ calls in 96s = 764 calls/second
- Each access has JAX dispatch overhead
- Should batch or vectorize

#### 3. Batch Bottom-Up Utilities (43% of time) 🔥🔥

```
_batch_bottom_up_utilities_sparse: 41.66s (8.46s self-time, 20 calls)
  └─ Python for-loops over configs and levels
  └─ _bottom_up_scan_sparse (not showing separately - inlined)
```

**Problem:** Python for-loops iterating over:
1. Multiple strategy configs (different player overrides)
2. Tree levels (varying BCOO sparsity prevents jax.lax.scan)

**Current structure:**
```python
for config in configs:
    for level in levels:
        result = sparse_matmul(...)  # GPU op, but loop is in Python
```

#### 4. Child Cache Building (ONE-TIME, 23.8s) ⚠️

```
_build_action_child_cache:        23.80s (during __init__)
  └─ _find_child_for_action:      23.76s (2,184 calls)
```

**NOT a per-iteration bottleneck** - happens once during initialization.
- Can be optimized later or cached to disk
- Acceptable one-time cost

---

## Optimization Strategy

### Priority 1: Optimize Scatter Operations (47-53% speedup potential)

**Current approach (SLOW):**
```python
# Incremental scatter - one at a time
for infoset in infosets:
    for action in actions:
        cumulative_strategy = cumulative_strategy.at[idx].add(value)
```

**Optimized approach (FAST):**
```python
# Batch scatter - all at once
all_indices = jnp.array([...])  # Pre-computed
all_values = jnp.array([...])   # Computed in batch
cumulative_strategy = cumulative_strategy.at[all_indices].add(all_values)
```

**Expected gain:** 2x (scatter operations go from 51s to ~25s)

### Priority 2: JIT Inner Sparse Operations (2-3x speedup potential)

**Current (Python loops):**
```python
def _batch_bottom_up_utilities_sparse(configs):
    results = []
    for config in configs:  # Python loop
        for level in levels:  # Python loop
            result = L_bcoo @ vector  # GPU op
        results.append(result)
    return results
```

**Optimized (JIT inner):**
```python
@jax.jit
def _sparse_level_step(L_bcoo, vector, strategy):
    weighted = L_bcoo * strategy
    return weighted @ vector

def _batch_bottom_up_utilities_sparse(configs):
    results = []
    for config in configs:  # Python loop (necessary)
        for level in levels:  # Python loop (necessary)
            result = _sparse_level_step(L, vec, strat)  # JIT-compiled
        results.append(result)
    return results
```

**Expected gain:** 2-3x on bottom-up (41.7s → ~15s)

### Priority 3: Reduce Array Indexing (1.5-2x speedup potential)

**Current:**
```python
# Many small accesses
for i in indices:
    value = array[i]  # Each is a JAX operation
    process(value)
```

**Optimized:**
```python
# Batch access
values = array[indices]  # Single JAX operation
process(values)          # Vectorized
```

**Expected gain:** 1.5-2x on indexing-heavy operations

---

## Combined Expected Speedup

**Conservative estimate:**
- Scatter optimization: 2x on 50% = 1.5x overall
- JIT inner loops: 2x on remaining 40% = 1.4x overall
- Reduce indexing: 1.3x on remaining = 1.3x overall
- **Total: 1.5 × 1.4 × 1.3 = 2.73x speedup**

**Optimistic estimate:**
- Scatter optimization: 2.5x on 50% = 1.75x overall
- JIT inner loops: 3x on remaining 40% = 1.6x overall
- Reduce indexing: 1.5x on remaining = 1.5x overall
- **Total: 1.75 × 1.6 × 1.5 = 4.2x speedup**

**Target achieved?**
- Baseline: 0.14 it/s
- Conservative (2.73x): **0.38 it/s** ⚠️ (not enough)
- Optimistic (4.2x): **0.59 it/s** ⚠️ (still not quite 1.0)
- **Need additional optimizations or accept slower speed**

---

## Implementation Plan

### Phase 6.1: Scatter Optimization (HIGHEST PRIORITY)

**Files to modify:**
- `matrix_cfr/matrix_cfr_solver.py:1715` (`_update_regrets_and_strategy`)
- `matrix_cfr/matrix_cfr_solver.py:1783` (`_update_cumulative_strategy`)

**Changes:**
1. Pre-compute all indices for batch scatter
2. Compute all values in parallel
3. Single scatter operation instead of loop

**Estimated time:** 2-3 hours
**Expected gain:** 2x (51s → ~25s per 10 iterations)

### Phase 6.2: JIT Inner Loops

**Files to modify:**
- `matrix_cfr/matrix_cfr_solver.py:1018-1061` (`_bottom_up_scan_sparse`)
- `matrix_cfr/matrix_cfr_solver.py:1146-1183` (`_full_reach_scan_sparse`)
- `matrix_cfr/matrix_cfr_solver.py:1283-1320` (`_counterfactual_reach_scan_sparse`)

**Changes:**
1. Extract inner operations to separate JIT functions
2. Keep Python loops but JIT the computation inside

**Estimated time:** 2-3 hours
**Expected gain:** 2-3x on utility/reach computation

### Phase 6.3: Reduce Indexing

**Files to modify:**
- Various locations using array[single_index]

**Changes:**
1. Identify repeated indexing patterns
2. Replace with batch indexing
3. Vectorize processing

**Estimated time:** 1-2 hours
**Expected gain:** 1.5x

---

## Next Steps

1. ✅ Profiling complete
2. ⏭️ Implement Phase 6.1 (scatter optimization)
3. ⏭️ Benchmark on Leduc (target: 0.3-0.4 it/s)
4. ⏭️ Implement Phase 6.2 (JIT inner loops)
5. ⏭️ Benchmark on Leduc (target: 0.5-0.8 it/s)
6. ⏭️ Implement Phase 6.3 (reduce indexing)
7. ⏭️ Final benchmark (target: 0.7-1.0 it/s)
8. ⏭️ Test on tiny Hold'em

**Realistic target:** 3-5x speedup (0.14 → 0.42-0.70 it/s)
**Stretch target:** 7x speedup (0.14 → 1.0 it/s)

---

## Conclusion

The profiling clearly shows that **scatter operations** and **Python loop overhead** are the primary bottlenecks. By optimizing these three areas, we can achieve 3-5x speedup, which should make small Hold'em games (2p_1bb_fc) viable.

For larger games, we'll need additional strategies:
- More aggressive abstractions
- Chunking by betting round
- Custom CUDA kernels
- Hybrid CPU/GPU approaches
