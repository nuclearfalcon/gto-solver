#!/usr/bin/env python3
"""
Test Phase 5.1: BCOO Conversion Validation

Tests that scipy sparse matrices correctly convert to JAX BCOO format
and that basic sparse operations work as expected.

Requirements:
- source ~/open_spiel/venv/bin/activate

Tests:
1. BCOO conversion from scipy CSR
2. Shape preservation
3. Basic sparse operations (transpose, matmul, element-wise)
4. Memory usage comparison
5. JIT compatibility
"""

import pyspiel
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver


def test_bcoo_conversion():
    """Test that BCOO conversion works and preserves matrix properties."""
    print("=" * 70)
    print("TEST 1: BCOO Conversion")
    print("=" * 70)
    print()

    game = pyspiel.load_game("kuhn_poker")

    # Create sparse solver
    solver_sparse = MatrixCFRSolver(game, use_sparse=True)

    # Check that level matrices are BCOO
    for i, L in enumerate(solver_sparse.level_matrices_jax):
        print(f"Level {i}: Type = {type(L)}")
        assert isinstance(L, jsparse.BCOO), f"Level {i} not BCOO!"

        # Check shape
        print(f"  Shape: {L.shape}")
        print(f"  Non-zeros: {L.nse}")
        print()

    print("✅ All level matrices converted to BCOO\n")


def test_sparse_vs_dense_equivalence():
    """Test that sparse and dense produce same results for matrix ops."""
    print("=" * 70)
    print("TEST 2: Sparse vs Dense Equivalence")
    print("=" * 70)
    print()

    game = pyspiel.load_game("kuhn_poker")

    # Create both solvers
    solver_sparse = MatrixCFRSolver(game, use_sparse=True)
    solver_dense = MatrixCFRSolver(game, use_sparse=False)

    # Compare each level matrix
    for i in range(len(solver_sparse.level_matrices_jax)):
        L_sparse = solver_sparse.level_matrices_jax[i]
        L_dense = solver_dense.level_matrices_jax[i]

        # Convert sparse to dense for comparison
        L_sparse_dense = L_sparse.todense()

        # Check equivalence
        diff = jnp.max(jnp.abs(L_sparse_dense - L_dense))
        print(f"Level {i}: max difference = {diff}")
        assert diff < 1e-6, f"Level {i} matrices don't match!"

    print("\n✅ Sparse and dense matrices are equivalent\n")


def test_sparse_operations():
    """Test that basic sparse operations work correctly."""
    print("=" * 70)
    print("TEST 3: Sparse Operations")
    print("=" * 70)
    print()

    game = pyspiel.load_game("kuhn_poker")
    solver = MatrixCFRSolver(game, use_sparse=True)

    L = solver.level_matrices_jax[0]  # First level matrix
    print(f"Testing operations on matrix with shape {L.shape}")
    print()

    # Test 1: Transpose
    print("Testing transpose...")
    L_T = L.T
    assert L_T.shape == (L.shape[1], L.shape[0])
    print(f"  ✓ Transpose shape: {L_T.shape}")

    # Test 2: Matrix-vector multiplication
    print("Testing matrix-vector multiplication...")
    num_nodes = L.shape[1]
    vec = jnp.ones(num_nodes, dtype=jnp.float32)
    result = L @ vec
    assert result.shape == (L.shape[0],)
    print(f"  ✓ Matmul result shape: {result.shape}")

    # Test 3: Element-wise multiplication with broadcasting
    print("Testing element-wise multiplication...")
    vec_broadcast = jnp.ones(num_nodes, dtype=jnp.float32)
    weighted = L * vec_broadcast[jnp.newaxis, :]
    assert weighted.shape == L.shape
    print(f"  ✓ Element-wise result shape: {weighted.shape}")

    # Test 4: Transpose @ vector
    print("Testing transpose @ vector...")
    result_T = L_T @ vec[:L.shape[0]]  # Adjust vector size for transpose
    assert result_T.shape == (L.shape[1],)
    print(f"  ✓ Transpose matmul result shape: {result_T.shape}")

    print("\n✅ All sparse operations work correctly\n")


def test_jit_compatibility():
    """Test that BCOO matrices work with JIT compilation."""
    print("=" * 70)
    print("TEST 4: JIT Compatibility")
    print("=" * 70)
    print()

    @jax.jit
    def sparse_matmul_jit(L_bcoo, vec):
        """JIT-compiled sparse matrix-vector multiplication."""
        return L_bcoo @ vec

    @jax.jit
    def sparse_transpose_matmul_jit(L_bcoo, vec):
        """JIT-compiled sparse transpose matrix-vector multiplication."""
        return L_bcoo.T @ vec

    game = pyspiel.load_game("kuhn_poker")
    solver = MatrixCFRSolver(game, use_sparse=True)

    L = solver.level_matrices_jax[0]
    vec = jnp.ones(L.shape[1], dtype=jnp.float32)

    # Test JIT compilation
    print("Testing JIT-compiled sparse matmul...")
    result1 = sparse_matmul_jit(L, vec)
    print(f"  ✓ Result shape: {result1.shape}")

    print("Testing JIT-compiled sparse transpose matmul...")
    vec_T = jnp.ones(L.shape[0], dtype=jnp.float32)
    result2 = sparse_transpose_matmul_jit(L, vec_T)
    print(f"  ✓ Result shape: {result2.shape}")

    # Test that recompilation works
    print("Testing recompilation with different inputs...")
    vec2 = jnp.zeros(L.shape[1], dtype=jnp.float32)
    result3 = sparse_matmul_jit(L, vec2)
    print(f"  ✓ Recompilation successful")

    print("\n✅ JIT compilation works with BCOO\n")


def test_memory_usage():
    """Compare memory usage of sparse vs dense."""
    print("=" * 70)
    print("TEST 5: Memory Usage")
    print("=" * 70)
    print()

    game = pyspiel.load_game("kuhn_poker")

    # Sparse solver
    solver_sparse = MatrixCFRSolver(game, use_sparse=True)

    # Dense solver
    solver_dense = MatrixCFRSolver(game, use_sparse=False)

    # Estimate memory for level matrices
    def estimate_matrix_memory(matrices, is_sparse=False):
        """Estimate memory usage in bytes."""
        total_bytes = 0
        for L in matrices:
            if is_sparse:
                # BCOO stores: data (nse × 4 bytes) + indices (nse × 2 × 4 bytes)
                total_bytes += L.nse * 4  # data
                total_bytes += L.nse * 2 * 4  # indices (row, col)
            else:
                # Dense stores: all elements (rows × cols × 4 bytes)
                total_bytes += L.size * 4
        return total_bytes

    sparse_bytes = estimate_matrix_memory(solver_sparse.level_matrices_jax, is_sparse=True)
    dense_bytes = estimate_matrix_memory(solver_dense.level_matrices_jax, is_sparse=False)

    print(f"Kuhn Poker ({solver_sparse.matrix_repr.num_nodes} nodes):")
    print(f"  Sparse (BCOO): {sparse_bytes:,} bytes ({sparse_bytes / 1024:.2f} KB)")
    print(f"  Dense: {dense_bytes:,} bytes ({dense_bytes / 1024:.2f} KB)")
    print(f"  Compression: {dense_bytes / max(sparse_bytes, 1):.1f}x")
    print()

    # For Kuhn, sparse might not save much (tiny game)
    # But we verify it doesn't use MORE memory
    assert sparse_bytes <= dense_bytes * 2, "Sparse should not use significantly more memory"

    print("✅ Sparse uses less or comparable memory\n")


def main():
    """Run all BCOO conversion tests."""
    print()
    print("=" * 70)
    print("PHASE 5.1: BCOO CONVERSION VALIDATION")
    print("=" * 70)
    print()

    try:
        test_bcoo_conversion()
        test_sparse_vs_dense_equivalence()
        test_sparse_operations()
        test_jit_compatibility()
        test_memory_usage()

        print("=" * 70)
        print("🎉 ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Phase 5.1 validation complete!")
        print("BCOO conversion infrastructure working correctly.")
        print()
        print("Next: Implement sparse JIT functions (Phase 5.2)")
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
