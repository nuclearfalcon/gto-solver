# Phase 5: Sparse Matrix Support

**Status:** ✅ COMPLETE
**Date:** 2025-11-02
**Goal:** Enable Leduc poker and larger games through sparse matrix support

## Executive Summary

Phase 5 successfully implemented sparse matrix support using JAX's BCOO format, achieving **185x memory compression** on Leduc poker and enabling games that previously caused OOM errors.

### Key Results

| Metric | Dense | Sparse | Improvement |
|--------|-------|--------|-------------|
| **Leduc Memory** | 2.67 GB (OOM) | 14.77 MB | **185x compression** |
| **Leduc Status** | ❌ Fails | ✅ Works | **ENABLED** |
| **Correctness** | Baseline | Max diff = 0.0 | **Identical** |

## Problem Statement

Phase 4 achieved 9.8x speedup but **couldn't scale to realistic games**:
- Kuhn poker (58 nodes): Works fine with dense matrices
- Leduc poker (9,457 nodes): **OOM at 2.67 GB** with dense matrices
- Hold'em (millions of nodes): Impossible with dense matrices

**Root cause:** Dense N×N matrices waste memory on tree-sparse game trees (99.9%+ zeros).

## Solution Architecture

### BCOO Sparse Format

JAX's experimental BCOO (Batched COO) format stores only non-zero elements:

```python
# Dense: N×N array (N² memory)
dense_matrix[i, j] = value

# BCOO: (indices, values, shape)
bcoo_matrix.data       # Non-zero values only
bcoo_matrix.indices    # (row, col) pairs
bcoo_matrix.nse        # Number of non-zeros
```

**Memory:** O(non-zeros) instead of O(N²)

### Implementation Strategy

**Phase 5.1 - Infrastructure:**
- Added JAX sparse imports (`jax.experimental.sparse`)
- Modified `_convert_to_jax()` to create BCOO matrices
- Added `use_sparse` feature flag (default: `True`)

**Phase 5.2 - Sparse JIT Functions:**
- `_bottom_up_scan_sparse()` - Utility propagation with BCOO
- `_full_reach_scan_sparse()` - Reach probability forward pass
- `_counterfactual_reach_scan_sparse()` - Counterfactual reach

**Phase 5.3 - Integration:**
- Conditional sparse/dense paths in core functions
- Sparse batch utilities: `_batch_bottom_up_utilities_sparse()`
- Removed dense stacked matrices from sparse mode

## Key Technical Decisions

### 1. Python for-loops Instead of jax.lax.scan

**Problem:** BCOO matrices have **varying sparsity** (different `nse` per level).

```python
# This FAILS with BCOO:
jax.lax.scan(fn, init, bcoo_list)  # Error: varying shapes
```

**Solution:** Use Python for-loops (JAX still JIT-compiles individual ops).

```python
# This works:
for L_bcoo in level_matrices:
    result = L_bcoo @ vector  # GPU-accelerated sparse op
```

**Trade-off:**
- ❌ Slower than dense scan on tiny games (7.9x slower on Kuhn)
- ✅ Enables large games that were impossible (Leduc works!)

### 2. Remove Dense Stacked Matrices

**Initially:** Kept both sparse BCOO and dense stacked for backward compatibility.
- Result: 8.2 GB memory usage (worse than dense!)

**Fix:** Implement sparse batch code, remove dense stacked.
- Result: 14.77 MB (185x compression)

### 3. JIT Single-Config Function

**Problem:** JIT-compiling entire batch with for-loops caused 20+ minute compile times.

**Solution:** JIT the single-configuration function, loop in Python.

```python
@jax.jit
def _single_sparse_bottom_up(matrices, utils, strategy):
    # JIT this function only
    return sparse_scan(matrices, utils, strategy)

def _batch_bottom_up_utilities_sparse(...):
    # Loop over configs in Python
    for config in configs:
        result = _single_sparse_bottom_up(...)  # JIT-compiled call
```

## Performance Analysis

### Memory Compression

**Kuhn Poker (58 nodes):**
- Dense: ~0.5 MB (acceptable)
- Sparse: ~0.5 MB (similar)
- Compression: ~1x (overhead cancels out)

**Leduc Poker (9,457 nodes):**
- Dense: 2.67 GB → **OOM** ❌
- Sparse: 14.77 MB ✅
- **Compression: 185x**

**Expected Hold'em (millions of nodes):**
- Dense: Terabytes (impossible)
- Sparse: Gigabytes (feasible)
- **Enables 3+ player Hold'em research**

### Speed Trade-offs

| Game | Dense (it/s) | Sparse (it/s) | Ratio |
|------|--------------|---------------|-------|
| Kuhn | 6 | 1 | 7.9x slower |
| Leduc | N/A (OOM) | 0.12 | **Enabled!** |

**Analysis:**
- Sparse is slower on tiny games (Python for-loop overhead)
- **But enables large games that were impossible**
- Trade-off is acceptable (memory > speed for research)

## Validation

### Correctness Tests

✅ **test_phase5_bcoo_conversion.py**
- Validates BCOO matches dense matrices
- Confirms sparsity (3 non-zeros per Kuhn level)

✅ **test_phase5_sparse_kuhn_quick.py**
- Compares sparse vs dense policies
- Max difference: **0.0** (identical)

✅ **test_phase5_leduc_memory.py**
- **THE CRITICAL TEST**
- Leduc initialization: ✅ Success (no OOM)
- 10 iterations: ✅ Complete
- Memory: 14.77 MB (185x compression)

## Files Modified

### Core Implementation
- `matrix_cfr/matrix_cfr_solver.py`
  - Lines 178-219: Sparse batch utilities
  - Lines 290-314: BCOO conversion in `_convert_to_jax()`
  - Lines 1018-1061: `_bottom_up_scan_sparse()`
  - Lines 1146-1183: `_full_reach_scan_sparse()`
  - Lines 1283-1320: `_counterfactual_reach_scan_sparse()`
  - Lines 975-991, 1118-1134, 1254-1270: Sparse/dense conditional paths

### Test Suite
- `test_phase5_bcoo_conversion.py` - BCOO format validation
- `test_phase5_sparse_kuhn_quick.py` - Correctness test
- `test_phase5_leduc_memory.py` - THE CRITICAL TEST

### Documentation
- `phase5_benchmark_results.txt` - Performance summary
- `PHASE5_SPARSE_MATRICES.md` - This file

## Usage

### Enable Sparse Mode (Default)

```python
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver
import pyspiel

game = pyspiel.load_game("leduc_poker")
solver = MatrixCFRSolver(game, use_sparse=True)  # Default
solver.solve(iterations=1000)
```

### Force Dense Mode (Small Games Only)

```python
solver = MatrixCFRSolver(game, use_sparse=False)  # Faster for Kuhn
```

## Success Criteria

✅ **Enable Leduc poker** - Was OOM with dense, now works
✅ **Achieve >100x compression** - Achieved 185x on Leduc
✅ **Maintain correctness** - Max difference = 0.0 vs dense
✅ **Unlock path to Hold'em** - Memory scales to larger games

## Future Work

### Phase 6 Optimization Opportunities

1. **Hybrid sparse/dense**
   - Use dense for small levels, sparse for large levels
   - Could recover some speed on intermediate games

2. **BCSR format**
   - JAX also supports BCSR (Batched CSR)
   - Might be faster for row-oriented operations

3. **Sparse vmap**
   - Research JAX sparse vmap capabilities
   - Could potentially vectorize some batch operations

4. **Custom CUDA kernels**
   - Write specialized sparse operations for poker trees
   - Could be faster than general BCOO operations

### Hold'em Testing

With sparse matrices working, we can now attempt:
- 2-player limit Hold'em
- 3-player Leduc
- Eventually: 3+ player NLHE (ultimate goal)

## Lessons Learned

1. **Varying sparsity requires Python for-loops**
   - JAX scan doesn't support varying shapes
   - Trade-off is acceptable for memory savings

2. **Don't mix sparse and dense representations**
   - Keeping both causes memory bloat
   - Commit to one representation per mode

3. **JIT strategy matters for sparse code**
   - JIT entire batch: 20+ min compile
   - JIT single config: Reasonable compile time

4. **Memory >> Speed for research**
   - 7.9x slowdown on Kuhn is acceptable
   - Enabling Leduc/Hold'em is the real win

## Conclusion

Phase 5 successfully implemented sparse matrix support, achieving the primary goal of **enabling Leduc poker and beyond**. The 185x memory compression demonstrates that sparse matrices are essential for scaling Matrix CFR to realistic poker games.

**Phase 5: COMPLETE ✅**

Next steps: Document in git commit and proceed to Hold'em testing or Phase 6 optimizations.
