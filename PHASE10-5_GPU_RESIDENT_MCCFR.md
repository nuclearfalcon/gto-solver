# Phase 10.5: GPU-Resident Bucketed MCCFR - The Ultimate Speedup

**Date**: 2025-11-04
**Status**: 🚀 **PLANNING → IMPLEMENTATION**
**Goal**: Achieve 100-1000× speedup by combining MCCFR sampling with Matrix CFR's GPU tensor operations

---

## Executive Summary

**The Pivot**: Phase 10.4 revealed that CPU sequential processing is the fundamental bottleneck, even with optimized state storage. The solution is to **stay on GPU entirely** using numeric bucket abstractions instead of string infosets.

### Performance Evolution

| Phase | Approach | Speed | Bottleneck | Status |
|-------|----------|-------|------------|--------|
| **Baseline** | Sequential MCCFR | 0.22 it/s | Everything | ✅ Complete |
| **10.2** | Batched sampling (standalone) | 814× trajectories | N/A | ✅ Complete |
| **10.3** | Infrastructure integration | 0.22 it/s | Sequential Python | ✅ Complete |
| **10.4 Hybrid** | GPU sample + CPU replay | 0.10 it/s | CPU replay (500s) | ❌ Failed |
| **10.4 Option 1** | GPU sample + store states | 0.10 it/s | CPU unflatten + regrets | ❌ Failed |
| **10.5** | **GPU-resident buckets** | **Expected: 22-220 it/s** | **None (all GPU)** | 🚀 **NOW** |

---

## The Core Insight

### Why Phase 10.4 Failed

**Time breakdown per iteration (batch_size=100):**
- GPU sampling: ~0.5-1s (fast, parallelized)
- **CPU unflattening: ~4-5s** (sequential Python loops)
- **CPU regret updates: ~4-5s** (dict lookups, sequential)
- **Total: ~10s per iteration → 0.10 it/s**

**Root cause**: Transferring to CPU and using Python dicts/loops negates GPU speedup.

### The Matrix CFR Lesson

Matrix CFR is fast because:
1. **Everything stays on GPU** - no CPU transfers
2. **Pure tensor operations** - no Python loops
3. **Numeric indices** - no string conversions
4. **Vectorized updates** - all infosets updated in parallel

**We need to apply these principles to MCCFR!**

---

## Phase 10.5 Architecture

### The Breakthrough: Numeric Bucket Abstraction

Instead of string infosets like:
```
"Ah Kh | Qd Jd 9h | pot:500 bets:200,0"
```

Use numeric bucket indices:
```
bucket_index = 742  # Maps to bucket 742 out of 10,000 total buckets
```

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GPU-RESIDENT MCCFR                       │
│                  (Everything on GPU!)                        │
└─────────────────────────────────────────────────────────────┘

1. Sample Trajectories (GPU - Batched)
   ↓
   states_batch: (100, 50, state_size)
   actions_batch: (100, 50)
   players_batch: (100, 50)
   payoffs_batch: (100, num_players)

2. Convert to Bucket Indices (GPU - Vectorized)
   ↓
   bucket_indices: (100, 50)  [values: 0 to num_buckets-1]

3. Compute CFVs (GPU - Vectorized)
   ↓
   cfvs: (100, 50)

4. Compute Regret Deltas (GPU - Vectorized)
   ↓
   regret_deltas: (100, 50, num_actions)

5. Scatter to Regret Tensor (GPU - Parallel)
   ↓
   regret_tensor: (num_buckets, num_actions)
   regret_tensor.at[bucket_indices].add(regret_deltas)

NO CPU INVOLVEMENT! NO PYTHON LOOPS!
```

---

## Numeric Bucket Design

### Card Abstraction

**Earth Mover's Distance (EMD) Buckets** - Standard in poker research:

```python
@jax.jit
def state_to_bucket_index(state: HoldemState, num_buckets: int = 10000) -> int:
    """
    Convert poker state to numeric bucket index (GPU-compatible).

    Uses hierarchical bucketing:
    1. Hand strength bucket (based on hole cards + board)
    2. Pot size bucket (coarse discretization)
    3. Round multiplier

    Total buckets: 200 hand × 10 pot × 4 rounds = 8,000 buckets
    (Can increase to 10,000 for finer granularity)
    """
    # Preflop: Bucket by hand strength (52 choose 2 = 1,326 → 200 buckets)
    # Flop: Bucket by equity vs random (infinite → 200 buckets)
    # Turn: Bucket by equity + potential (infinite → 200 buckets)
    # River: Bucket by exact hand rank (7,462 → 200 buckets)

    hand_bucket = compute_hand_bucket(state.hole_cards, state.board, state.round)
    pot_bucket = compute_pot_bucket(state.pot, state.stacks)

    bucket_index = (
        hand_bucket +
        pot_bucket * 200 +
        state.round * 2000
    )

    return bucket_index % num_buckets
```

**Benefits**:
- ✅ Fixed memory: 10,000 buckets × 4 actions = 40,000 float32 values (160 KB)
- ✅ GPU-compatible: Pure JAX operations
- ✅ Information preserving: Captures key strategic features
- ✅ Parallelizable: Can bucket entire batch at once

### Simplified Version (For Quick Implementation)

**Hash-based bucketing** - Faster to implement:

```python
@jax.jit
def state_to_bucket_index_simple(state: HoldemState, num_buckets: int = 10000) -> int:
    """Simple hash-based bucketing for quick implementation."""
    # Combine state features into hash
    h = jnp.int32(0)
    h = h * 31 + jnp.sum(state.hole_cards[0])  # Player 0 cards
    h = h * 31 + jnp.sum(state.board[state.board >= 0])  # Board cards
    h = h * 31 + jnp.int32(state.pot / 100)  # Pot size (discretized)
    h = h * 31 + state.round  # Round

    return h % num_buckets
```

---

## Implementation Plan

### Phase 10.5.1: Bucket Infrastructure

**Tasks**:
1. Implement `state_to_bucket_index()` function (JAX-compatible)
2. Create GPU-resident regret tensor: `jnp.zeros((num_buckets, num_actions))`
3. Test bucketing on sample trajectories

**Expected time**: 1-2 hours

### Phase 10.5.2: Vectorized CFV Computation

**Tasks**:
1. Rewrite `compute_counterfactual_values()` as pure JAX function
2. Vectorize over batch dimension with `jax.vmap`
3. Handle terminal values correctly

**Expected time**: 2-3 hours

### Phase 10.5.3: GPU Scatter Updates

**Tasks**:
1. Implement batched regret scatter: `regret_tensor.at[indices].add(updates)`
2. Handle conflicts (multiple updates to same bucket)
3. Strategy sum updates (also on GPU)

**Expected time**: 1-2 hours

### Phase 10.5.4: Integration & Testing

**Tasks**:
1. Integrate into `run_iteration()`
2. Create validation tests
3. Benchmark against baseline

**Expected time**: 2-3 hours

**Total estimated time**: 6-10 hours

---

## Expected Performance

### Resource Usage

**GPU Memory**:
- Regret tensor: 10,000 buckets × 4 actions × 4 bytes = 160 KB
- Strategy sum: 10,000 × 4 × 4 bytes = 160 KB
- Trajectory batch: 100 × 50 × 73 × 4 bytes = 1.5 MB
- **Total: ~2 MB** (trivial on modern GPUs)

**Performance Targets**:
| Batch Size | GPU Time | Total Time | Speed | Speedup |
|------------|----------|------------|-------|---------|
| 100 | ~0.5s | ~0.5s | 2 it/s | 9× |
| 500 | ~1s | ~1s | 1 it/s | 4.5× |
| 1000 | ~1.5s | ~1.5s | 0.67 it/s | 3× |

Wait, these are per-iteration times. Let me recalculate:

**Per iteration (batch_size=100)**:
- GPU sampling: 0.3s
- GPU bucketing: 0.01s
- GPU CFV computation: 0.05s
- GPU regret scatter: 0.01s
- **Total: ~0.4s**

**Speed: 2.5 it/s**
**Speedup vs baseline (0.22 it/s): 11×**

But wait - with batch_size=100, we're processing **100 trajectories per iteration**, not 1!

**Correct calculation**:
- Baseline: 0.22 it/s = 0.22 trajectories/s
- GPU-resident (batch=100): 2.5 it/s × 100 traj/it = **250 trajectories/s**
- **Speedup: 250 / 0.22 = 1136× faster!**

### Conservative vs Optimistic

**Conservative** (accounting for overhead):
- Speed: 1.5 it/s
- Throughput: 150 trajectories/s
- **Speedup: 682×**

**Optimistic** (minimal overhead):
- Speed: 3 it/s
- Throughput: 300 trajectories/s
- **Speedup: 1364×**

---

## Key Innovations

1. **Numeric Bucketing**: Replaces string infosets with GPU-friendly indices
2. **GPU-Resident Regrets**: Tensor storage instead of Python dicts
3. **Vectorized CFVs**: Parallel computation instead of sequential loops
4. **Scatter Updates**: GPU-parallel tensor scatter instead of dict updates
5. **No CPU Transfers**: Everything stays on GPU

---

## Comparison to Previous Approaches

| Feature | Phase 10.4 | Phase 10.5 |
|---------|------------|------------|
| **Sampling** | GPU (batched) | GPU (batched) ✅ |
| **State Storage** | Flattened arrays | Not needed ✅ |
| **Bucket Conversion** | N/A | GPU (vectorized) ✅ |
| **CFV Computation** | CPU (sequential) | GPU (vectorized) ✅ |
| **Regret Updates** | CPU (dict) | GPU (tensor scatter) ✅ |
| **CPU/GPU Transfers** | 4 MB per iteration | **0 bytes** ✅ |
| **Python Loops** | 100+ per iteration | **0** ✅ |
| **Expected Speed** | 0.10 it/s | 1.5-3 it/s |
| **Expected Throughput** | 10 traj/s | 150-300 traj/s |
| **Speedup vs Baseline** | 0.45× (slower!) | **682-1364×** ✅ |

---

## Success Criteria

**Minimum Success** (Conservative):
- Speed: ≥1 it/s
- Throughput: ≥100 traj/s
- Speedup: ≥454×

**Target Success**:
- Speed: ≥2 it/s
- Throughput: ≥200 traj/s
- Speedup: ≥900×

**Stretch Goal**:
- Speed: ≥3 it/s
- Throughput: ≥300 traj/s
- Speedup: ≥1364×

---

## Risks & Mitigations

### Risk 1: Bucket Collisions

**Issue**: Multiple distinct game states map to same bucket.

**Mitigation**:
- Use 10,000 buckets (enough for most poker research)
- Can increase to 50,000 or 100,000 if needed (still only 2 MB memory)
- Monitor collision rates during training

### Risk 2: CFV Computation Complexity

**Issue**: Counterfactual values are complex to compute in pure JAX.

**Mitigation**:
- Start with simplified CFV (terminal payoffs only)
- Incrementally add complexity
- Can always fall back to CPU for complex cases

### Risk 3: Convergence Quality

**Issue**: Bucketing may reduce solution quality.

**Mitigation**:
- This is standard in poker research (all practical solvers use abstraction)
- Can validate against unbucketed version on small games
- Finer buckets if needed (100K buckets still trivial memory)

---

## Next Steps

1. ✅ Design numeric bucket abstraction (THIS DOCUMENT)
2. ⏳ Implement `state_to_bucket_index()` function
3. ⏳ Create GPU-resident regret tensor
4. ⏳ Implement vectorized CFV computation
5. ⏳ Implement GPU scatter updates
6. ⏳ Integration & testing
7. ⏳ Benchmark and validate

**Status**: Ready to implement!

---

**Prepared by**: Claude (Sonnet 4.5)
**Date**: 2025-11-04
**Phase**: 10.5 - GPU-Resident Bucketed MCCFR (The Ultimate Speedup)
