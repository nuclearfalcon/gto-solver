# Phase 4 Optimization Plan

**Goal:** Eliminate remaining Python loops by refactoring to pure array-based operations.

**Current Status:** Phase 3 achieved 5.6-7.8x speedup (0.43 → 2.4-3.4 it/s on Kuhn poker). However, ~45-50% of iteration time is still spent in Python loops.

**Target:** Achieve 50-100 it/s on 3-player Hold'em games (100-200x baseline).

---

## Why Phase 3.4 Was Reverted

Phase 3.4 attempted to "vectorize" regret updates but had critical flaws:

```python
# Phase 3.4 code (REVERTED)
instant_regrets_full = jnp.zeros_like(self.cumulative_regrets)  # ❌ Memory allocation every iteration

for infoset, action_values in cf_values.items():  # ❌ Still has Python loop!
    instant_regrets_full = instant_regrets_full.at[action_indices].set(instant_regrets)

self.cumulative_regrets = self.cumulative_regrets + instant_regrets_full
```

**Problems:**
1. **Still has Python loop** - Dictionary structure forces iteration over infosets
2. **Memory allocation overhead** - Allocates 1-2 MB every iteration (bad for 3-player Hold'em)
3. **Not truly vectorized** - Uses `.set()` in loop, then one addition
4. **No performance benefit** - Benchmarks showed neutral-to-negative impact

**Root cause:** Cannot vectorize operations on dictionary-based data structures.

---

## Phase 4 Solution: Array-Based Refactoring

### Core Idea

Replace dictionary-based `cf_values` with 2D array representation:

```python
# CURRENT (Phase 3):
cf_values = {
    'infoset_0': jnp.array([v1, v2, v3]),     # Dictionary lookup required
    'infoset_1': jnp.array([v4, v5]),
    'infoset_5': jnp.array([v6, v7, v8, v9]),
    ...
}

# PHASE 4 TARGET:
cf_values_2d = jnp.array([
    [v1, v2, v3, 0,  0,  0],   # Row 0 = infoset 0 (padded to max_actions)
    [v4, v5, 0,  0,  0,  0],   # Row 1 = infoset 1
    [0,  0,  0,  0,  0,  0],   # Row 2 = infoset 2 (not active this iteration)
    ...
    [v6, v7, v8, v9, 0,  0],   # Row 5 = infoset 5
])
current_strategy_2d = jnp.array([...])  # Same shape
```

With arrays, ALL operations become single vectorized JAX calls with **no Python loops**.

---

## Phase 4 Optimizations

### OPT-4.1: Array-Based Counterfactual Values (CRITICAL)

**Current bottleneck:**
```python
# matrix_cfr_solver.py:1140-1157
def _extract_cf_values_from_utilities(self, all_utilities, metadata):
    cf_values = {}  # ❌ Dictionary

    for idx, (infoset, action, action_idx, child_node_id) in enumerate(metadata):  # ❌ Python loop
        if infoset not in cf_values:
            num_actions = len(self.matrix_repr.infoset_to_actions[infoset])
            cf_values[infoset] = jnp.zeros(num_actions, dtype=jnp.float32)

        child_utility = all_utilities[idx, child_node.depth, child_node_id]
        cf_values[infoset] = cf_values[infoset].at[action_idx].set(child_utility)

    return cf_values  # Dictionary
```

**Phase 4 refactor:**
```python
def _extract_cf_values_from_utilities_vectorized(self, all_utilities, metadata_array):
    """
    Extract CF values using pure array operations.

    Args:
        all_utilities: (num_actions, num_depths, num_nodes) utilities
        metadata_array: Pre-built array of (infoset_idx, action_idx, child_depth, child_id)

    Returns:
        cf_values_2d: (num_infosets, max_actions) padded array
    """
    # Pre-built indices from metadata
    infoset_indices = metadata_array[:, 0]  # Which infoset
    action_indices = metadata_array[:, 1]   # Which action within infoset
    child_depths = metadata_array[:, 2]     # Which depth
    child_ids = metadata_array[:, 3]        # Which node

    # Single vectorized gather (no loop!)
    batch_indices = jnp.arange(len(metadata_array))
    child_utilities = all_utilities[batch_indices, child_depths, child_ids]

    # Single vectorized scatter (no loop!)
    cf_values_2d = jnp.zeros((num_infosets, max_actions), dtype=jnp.float32)
    cf_values_2d = cf_values_2d.at[infoset_indices, action_indices].set(child_utilities)

    return cf_values_2d
```

**Impact:**
- Eliminates Python loop over actions (~12-1000+ iterations)
- Zero overhead from dict operations
- Fully JIT-compilable

**Complexity:** Medium - requires pre-building metadata arrays at initialization

---

### OPT-4.2: Vectorized Regret Updates (CRITICAL)

**Current bottleneck:**
```python
# matrix_cfr_solver.py:1364-1379
def _update_regrets_and_strategy(self, player, cf_values):
    for infoset, action_values in cf_values.items():  # ❌ Python loop over infosets
        action_indices = self.infoset_action_indices[infoset]
        current_probs = self.current_strategy[action_indices]
        strategy_value = jnp.sum(current_probs * action_values)
        instant_regrets = action_values - strategy_value

        for i, action_idx in enumerate(action_indices):  # ❌ Nested Python loop
            self.cumulative_regrets = self.cumulative_regrets.at[action_idx].add(
                instant_regrets[i]
            )
```

**Phase 4 refactor:**
```python
def _update_regrets_and_strategy_vectorized(self, player, cf_values_2d):
    """
    Update regrets using pure array operations.

    Args:
        player: Player index
        cf_values_2d: (num_infosets, max_actions) counterfactual values (padded)
    """
    # Get current strategy as 2D array
    current_strategy_2d = self._convert_1d_to_2d(self.current_strategy)

    # Vectorized strategy values (one op for ALL infosets)
    strategy_values = jnp.sum(current_strategy_2d * cf_values_2d, axis=1, keepdims=True)

    # Vectorized instant regrets (one op for ALL infosets)
    instant_regrets_2d = cf_values_2d - strategy_values  # Broadcasting

    # Convert back to 1D and update (one op)
    instant_regrets_1d = self._convert_2d_to_1d(instant_regrets_2d)
    self.cumulative_regrets = self.cumulative_regrets + instant_regrets_1d

    # Rest of method unchanged
    self.current_strategy = self._regret_matching()
    ...
```

**Impact:**
- Eliminates BOTH Python loops (outer over infosets, inner over actions)
- Three vectorized JAX ops instead of 12-1000+ sequential ops
- Fully JIT-compilable
- No memory allocation overhead

**Complexity:** Low - once OPT-4.1 provides array-based cf_values

---

### OPT-4.3: Pre-build Override Application (MEDIUM)

**Current bottleneck:**
```python
# matrix_cfr_solver.py:1194-1199
for i in range(num_overrides):  # ❌ Python loop (12-1000+ iterations)
    all_overrides = all_overrides.at[i, zero_indices[i]].set(0.0)
    all_overrides = all_overrides.at[i, one_indices[i]].set(1.0)
```

**Phase 4 refactor:**
```python
def _build_all_action_overrides_vectorized(self, player):
    """Apply override templates using advanced indexing (no loop)."""
    # Pre-built 2D index arrays
    batch_indices = self.override_batch_indices[player]  # (num_overrides,)
    zero_row_indices = self.override_zero_rows[player]   # (total_zeros,)
    zero_col_indices = self.override_zero_cols[player]   # (total_zeros,)
    one_row_indices = self.override_one_rows[player]     # (num_overrides,)
    one_col_indices = self.override_one_cols[player]     # (num_overrides,)

    num_overrides = len(batch_indices)
    all_overrides = jnp.tile(self.current_strategy, (num_overrides, 1))

    # Single vectorized zero-out
    all_overrides = all_overrides.at[zero_row_indices, zero_col_indices].set(0.0)

    # Single vectorized one-set
    all_overrides = all_overrides.at[one_row_indices, one_col_indices].set(1.0)

    return all_overrides, self.override_metadata[player]
```

**Impact:**
- Eliminates loop building override matrices
- Two vectorized ops instead of 12-1000+ sequential ops

**Complexity:** Medium - requires flattening override indices at initialization

---

### OPT-4.4: Batched Reach Probability Computation (MEDIUM)

**Current bottleneck:**
```python
# matrix_cfr_solver.py:1250-1268
def _full_reach_probabilities(self, node_strategy):
    reach = jnp.ones(num_nodes, dtype=jnp.float32)

    for level in range(num_levels):  # ❌ Python loop over depths
        level_matrix = self.level_matrices_jax[level]
        reach = level_matrix @ (reach * node_strategy)  # Works, but loop overhead

    return reach
```

**Phase 4 refactor:**
```python
def _full_reach_probabilities_vectorized(self, node_strategy):
    """Compute reach using JAX scan (no Python loop)."""

    def step_fn(reach, level_matrix):
        return level_matrix @ (reach * node_strategy), None

    # Single JIT-compiled scan (no Python loop!)
    final_reach, _ = jax.lax.scan(
        step_fn,
        jnp.ones(num_nodes, dtype=jnp.float32),
        self.level_matrices_jax_stacked
    )

    return final_reach
```

**Impact:**
- Eliminates loop over tree depths
- `jax.lax.scan` compiles to fused kernel

**Complexity:** Low - `jax.lax.scan` is straightforward

---

## Phase 4 Pre-building Requirements

To support vectorized operations, we need to pre-build index arrays at initialization:

```python
# In _init_cfr_state():

# OPT-4.1: Metadata for CF value extraction
self.cf_extraction_metadata = self._prebuild_cf_extraction_indices()
# Returns: (num_total_actions, 4) array of [infoset_idx, action_idx, child_depth, child_id]

# OPT-4.3: Flattened override indices
self.override_batch_indices = {}   # player → (num_overrides,)
self.override_zero_rows = {}       # player → (total_zeros,)
self.override_zero_cols = {}       # player → (total_zeros,)
self.override_one_rows = {}        # player → (num_overrides,)
self.override_one_cols = {}        # player → (num_overrides,)
self._prebuild_override_application_indices()
```

**Memory overhead:** Minimal - these are small integer arrays (few KB even for Hold'em).

---

## Expected Performance Impact

### Kuhn Poker (2 players, 12 infosets, 24 actions)

**Current bottlenecks after Phase 3:**
- 30%: Building override matrices (OPT-4.3)
- 25%: Extracting CF values (OPT-4.1)
- 20%: Updating regrets (OPT-4.2)
- 15%: Reach probability computation (OPT-4.4)
- 10%: Other (JIT overhead, 1D/2D conversions)

**Projected speedup:**
- Phase 4.1 + 4.2: **2-3x** (eliminates 45% of runtime)
- Phase 4.3: **1.5x** (eliminates 30% of runtime)
- Phase 4.4: **1.2x** (eliminates 15% of runtime)
- **Combined: 3.6-5.4x on top of Phase 3 = 20-40x total from baseline**

**Kuhn projection:** 0.43 → 9-17 it/s

### 3-Player Hold'em (50k infosets, 500k actions)

**Why Phase 4 is critical for Hold'em:**

1. **Batch size scales massively:**
   - Kuhn: 24 actions → small batches, GPU underutilized
   - Hold'em: 500,000 actions → huge batches, GPU saturated
   - GPU parallelism is ~1000-10,000x more effective

2. **Python loop overhead scales poorly:**
   - Kuhn: 12 infosets → 12 loop iterations (small)
   - Hold'em: 50,000 infosets → 50,000 loop iterations (massive!)
   - Eliminating loops saves seconds per iteration

3. **Memory allocation scales poorly:**
   - Kuhn: 24 actions × 4 bytes = 96 bytes per allocation
   - Hold'em: 500,000 actions × 4 bytes = 2 MB per allocation
   - Phase 3.4 would allocate 2 MB every iteration = 100 MB/s garbage!

**Hold'em projection:**
- Phase 3: ~5-10 it/s (extrapolating from Kuhn)
- Phase 4: **50-150 it/s** (GPU fully saturated)
- **100-300x speedup from baseline!**

---

## Implementation Priority

| Priority | Optimization | Impact | Complexity | Estimate |
|----------|-------------|---------|------------|----------|
| 🔴 **P0** | OPT-4.1: Array-based CF values | Critical | Medium | 4-6 hours |
| 🔴 **P0** | OPT-4.2: Vectorized regret updates | Critical | Low | 2-3 hours |
| 🟡 **P1** | OPT-4.3: Vectorized override application | High | Medium | 3-4 hours |
| 🟢 **P2** | OPT-4.4: Batched reach probabilities | Medium | Low | 1-2 hours |

**Total estimated effort:** 10-15 hours of implementation + 5-10 hours testing/debugging

---

## Phase 4 Implementation Roadmap

### Step 1: Pre-build Index Arrays (Foundation)

1. Implement `_prebuild_cf_extraction_indices()`
   - Map each (infoset, action) to (infoset_idx, action_idx, child_depth, child_id)
   - Store as (N, 4) int32 array

2. Implement `_prebuild_override_application_indices()`
   - Flatten override zero/one patterns into row/col arrays
   - Store per-player

**Validation:** Print shapes, verify indices match current behavior

### Step 2: Array-Based CF Values (OPT-4.1)

1. Implement `_extract_cf_values_from_utilities_vectorized()`
2. Update `_cfr_iteration_both_players()` to use vectorized version
3. Run tests to verify correctness

**Validation:** Compare results to Phase 3 version (should be identical)

### Step 3: Vectorized Regret Updates (OPT-4.2)

1. Implement `_update_regrets_and_strategy_vectorized()`
2. Update main loop to use vectorized version
3. Run benchmarks

**Validation:** Test on Kuhn, verify learning still works

### Step 4: Vectorized Override Application (OPT-4.3)

1. Implement `_build_all_action_overrides_vectorized()`
2. Update `_cfr_iteration_both_players()` to use it
3. Run benchmarks

**Validation:** Compare override matrices to Phase 3

### Step 5: Batched Reach Probabilities (OPT-4.4)

1. Implement `_full_reach_probabilities_vectorized()` with `jax.lax.scan`
2. Update `_update_regrets_and_strategy_vectorized()` to use it
3. Final benchmarks

**Validation:** Compare reach probabilities to Phase 3

### Step 6: Comprehensive Testing

1. Kuhn poker: Verify 20-40x total speedup (9-17 it/s)
2. Leduc poker: Test on medium game (~200 infosets)
3. Small Hold'em: Test on abstracted 2-player Hold'em
4. Learning verification: Run 10k iterations, check exploitability convergence

---

## Risks and Mitigations

### Risk 1: JAX Advanced Indexing Limitations

**Problem:** JAX has restrictions on scatter operations (`.at[indices].set()`).

**Mitigation:**
- Use `jax.numpy` advanced indexing carefully
- May need `jax.ops.segment_sum` for some operations
- Test incrementally, keep Phase 3 as fallback

### Risk 2: Memory Usage on Large Games

**Problem:** 2D padded arrays use more memory than sparse dicts.

**Analysis:**
- Hold'em: 50k infosets × 10 max_actions × 4 bytes = 2 MB (negligible)
- Trade-off: 2 MB constant memory vs 2 MB/iteration allocations
- **Net win:** Eliminates garbage collection overhead

**Mitigation:**
- Monitor memory usage during Hold'em tests
- If needed, use sparse representations (but likely unnecessary)

### Risk 3: Complexity Increase

**Problem:** Array-based code is less readable than dict-based.

**Mitigation:**
- Extensive documentation with examples
- Keep helper methods small and focused
- Add assertions to verify index correctness
- Maintain Phase 3 as reference implementation

---

## Success Metrics

**Phase 4 is successful if:**

1. ✅ Kuhn poker: 9-17 it/s (20-40x baseline)
2. ✅ Leduc poker: 50-100 it/s (100-200x baseline)
3. ✅ 3-player Hold'em: 50-150 it/s (100-300x baseline)
4. ✅ Learning preserved: Exploitability converges correctly
5. ✅ Code correctness: All tests pass, results match Phase 3

---

## Beyond Phase 4: Future Optimizations

After Phase 4, the codebase will be fully vectorized. Remaining opportunities:

### OPT-5.1: Multi-GPU Batching
- Split batch across multiple GPUs
- Potential 2-4x speedup with 4 GPUs

### OPT-5.2: Mixed Precision (FP16)
- Use half-precision floats for intermediate calculations
- Potential 1.5-2x speedup, but may hurt learning

### OPT-5.3: Approximate CFR Variants
- Use External Sampling MCCFR (process subset of actions)
- Potential 5-10x speedup, but different algorithm

### OPT-5.4: Custom CUDA Kernels
- Hand-write GPU kernels for critical paths
- Extreme effort, potential 2-3x speedup

**Recommendation:** Only pursue these if Phase 4 doesn't hit 50-100 it/s target on Hold'em.

---

## Summary

**Phase 3 achieved:** 5.6-7.8x speedup via JIT compilation and batching

**Phase 4 goal:** 100-300x total speedup via eliminating Python loops

**Key insight:** Cannot vectorize dictionary-based data structures. Must refactor to arrays.

**Implementation plan:** 4 optimizations, 10-15 hours estimated effort

**Target performance:**
- Kuhn: 9-17 it/s
- Leduc: 50-100 it/s
- Hold'em: 50-150 it/s ✅ **Achieves project goal!**
