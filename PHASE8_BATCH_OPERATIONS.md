# Phase 8.1: Batch Array Operations Optimization

**Goal**: Eliminate 73,750 individual `__getitem__` calls (44.7s / 34.6% of runtime)

## Profiling Analysis

### Current Bottleneck (Leduc, 10 iterations, 129s total)

| Function | Calls | Time | % | Issue |
|----------|-------|------|---|-------|
| `_batch_bottom_up_utilities_sparse:193` | 43,680 | 21.1s | 16.3% | Array indexing in scan |
| `_update_cumulative_strategy:1783` | 18,720 | 8.7s | 6.7% | 1D/2D conversion |
| `_find_child_for_action:782` | 4,368 | 9.5s | 7.4% | **BIGGEST WIN - sparse lookup loop** |
| JAX BCOO internals | 4,368 | 2.7s | 2.1% | Sparse matrix overhead |
| `_prebuild_override_templates:495` | 2,184 | 1.0s | 0.8% | Template building |
| **TOTAL** | **73,750** | **44.7s** | **34.6%** | |

## Optimization Strategy

### OPT-8.1.1: Vectorize `_find_child_for_action` ⭐⭐⭐ (HIGHEST IMPACT)

**Current** (4,368 calls, 9.5s):
```python
def _find_child_for_action(self, parent_node_id, action, parent_depth):
    L_l = self.level_matrices_jax[child_depth]
    row_mask = L_l.indices[:, 0] == parent_node_id  # 🐌 Called 4,368 times!
    child_col_indices = L_l.indices[row_mask, 1]
    child_list = [int(idx) for idx in child_col_indices]
    return child_list[action]
```

**After** (pre-build lookup once):
```python
# Build once during initialization:
def _build_child_lookup_table(self):
    """
    Pre-build (infoset, action) -> child_node mapping.

    Returns:
        child_lookup: Dict[(infoset, action)] -> child_node_id
    """
    # Use existing action_index_to_node dict (already has most of mapping!)
    # Just need to map to CHILDREN instead of parents
    pass

# Use O(1) lookup:
def _find_child_for_action_fast(self, parent_node_id, action, parent_depth):
    return self.child_lookup_table[(parent_node_id, action)]
```

**Expected gain**: 9.5s → ~0.1s (95x faster, eliminates 4,368 sparse lookups)

---

### OPT-8.1.2: Cache Sparse Matrix Row Access

**Current** (43,680 calls in `_batch_bottom_up_utilities_sparse`):
```python
def single_config_utilities(node_strategy):
    def scan_fn(carry_utils, L_l):
        weighted_L = L_l * node_strategy[jnp.newaxis, :]  # 🐌 Indexing overhead
        propagated = weighted_L @ carry_utils
        ...
```

**After**:
```python
# Pre-compute node strategy with newaxis ONCE
node_strategy_2d = node_strategy[jnp.newaxis, :]  # Do outside loop

def scan_fn(carry_utils, L_l):
    weighted_L = L_l * node_strategy_2d  # Reuse pre-computed
    ...
```

**Expected gain**: Modest (maybe 5-10% on this function)

---

### OPT-8.1.3: Optimize 1D/2D Conversions

**Current** (`_update_cumulative_strategy` called 18,720 times):
```python
def _convert_2d_to_1d(self, padded_array):
    flat = jnp.zeros(num_ia, dtype=padded_array.dtype)
    flat = flat.at[self.flat_to_2d_indices].set(
        padded_array[self.flat_to_2d_rows, self.flat_to_2d_cols]  # 🐌 Indexing
    )
    return flat
```

**Observation**: Already optimized with pre-built indices (Phase 3.3). The 18,720 calls are inherent to algorithm (936 infosets × 20 iterations). Not much to optimize here without changing algorithm.

**Expected gain**: Minimal (already near-optimal)

---

## Implementation Plan

### Priority 1: Child Lookup Table (OPT-8.1.1)

1. **Add to `__init__`**:
   ```python
   self.child_lookup_table = self._build_child_lookup_table()
   ```

2. **Implement `_build_child_lookup_table()`**:
   - Iterate through level matrices once
   - Build dict mapping `(parent_node, action) -> child_node`
   - Use sparse indices for efficiency

3. **Replace `_find_child_for_action()`**:
   - Change from O(num_nonzero_in_row) to O(1)
   - Eliminate 4,368 sparse matrix scans

4. **Validation**:
   - Test on Kuhn/Leduc - verify identical results
   - Measure memory overhead (should be ~2 MB for Hold'em)

### Priority 2: Scan Function Optimization (OPT-8.1.2)

1. **Modify `_batch_bottom_up_utilities_sparse()`**:
   - Move `node_strategy[jnp.newaxis, :]` outside scan
   - Pass as closure variable

2. **Validation**:
   - Ensure numerical equivalence
   - Measure impact

## Success Criteria

**Minimum**:
- ✅ Child lookup table reduces `_find_child_for_action` time by 80%+ (9.5s → <2s)
- ✅ No correctness regressions (all tests pass)

**Target**:
- ✅ Total reduction: 44.7s → ~25-30s (30-40% speedup on indexing overhead)
- ✅ Leduc speed: 0.10 it/s → 0.13-0.15 it/s

**Stretch**:
- ✅ Combined with other optimizations: 0.10 → 0.20 it/s (2x)

## Memory Trade-offs

**Child Lookup Table**:
- Kuhn: 24 entries × 16 bytes = 384 bytes
- Leduc: 2,184 entries × 16 bytes = 34 KB
- Hold'em (74K nodes): ~150K entries × 16 bytes = 2.4 MB

**Verdict**: Acceptable - 2.4 MB is tiny compared to 16 GB VRAM

## Testing Plan

1. **Unit test**: `test_phase8_child_lookup.py`
   - Verify lookup table correctness
   - Compare against old `_find_child_for_action`

2. **Integration test**: Run Leduc convergence
   - Ensure identical policy learned
   - Measure speed improvement

3. **Scaling test**: Profile on Tiny Hold'em
   - Verify optimization scales to larger games

---

## Next Steps After This

Once batch operations are done:
- **Option A**: FP16 mixed precision (2x memory → bigger chunks)
- **Option B**: Chunking architecture design (direct path to full Hold'em)

My recommendation: **Option B** (chunking) since it's the critical path and we have enough memory headroom now.
