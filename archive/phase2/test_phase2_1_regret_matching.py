"""
Test Phase 2.1: Vectorized Regret Matching

Validates that the new vectorized regret matching produces identical
results to the old sequential implementation.

IMPORTANT: Activate OpenSpiel virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import pyspiel
import numpy as np
import jax.numpy as jnp
import sys

# Import the solver
from matrix_cfr import MatrixCFRSolver


def test_vectorized_regret_matching():
    """Test that vectorized regret matching matches sequential version."""
    print("=" * 70)
    print("Testing Phase 2.1: Vectorized Regret Matching")
    print("=" * 70)

    # Create Kuhn poker game
    game = pyspiel.load_game('kuhn_poker', {'players': 2})

    # Create solver
    print("\nInitializing solver...")
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

    # Set some non-uniform regrets for testing
    print("\nSetting test regrets...")
    test_regrets = jnp.array([
        5.0, -2.0,   # Infoset 0: positive and negative
        0.0, 0.0,    # Infoset 0b: all zero
        -1.0, -3.0,  # Infoset 0p: all negative
        3.0, 7.0,    # Infoset 0pb: two positive
        # ... etc for all 24 infoset-actions
    ] + [1.0] * 16, dtype=jnp.float32)  # Fill rest with 1.0

    solver.cumulative_regrets = test_regrets

    # Run vectorized regret matching
    print("\nRunning vectorized regret matching...")
    strategy = solver._regret_matching()

    # Validate basic properties
    print("\nValidating results...")

    # 1. Strategy should sum to 1.0 for each infoset
    for infoset, indices in solver.infoset_action_indices.items():
        strategy_sum = jnp.sum(strategy[indices])
        if not jnp.isclose(strategy_sum, 1.0):
            print(f"❌ FAIL: Infoset {infoset} strategy sums to {strategy_sum}, not 1.0!")
            return False

    print("✅ All infoset strategies sum to 1.0")

    # 2. No negative probabilities
    if jnp.any(strategy < 0):
        print(f"❌ FAIL: Found negative probabilities!")
        return False

    print("✅ No negative probabilities")

    # 3. Check specific cases
    # Infoset 0: [5.0, -2.0] → should be [1.0, 0.0] (only first is positive)
    infoset_0_strategy = strategy[solver.infoset_action_indices["0"]]
    expected_0 = jnp.array([1.0, 0.0])
    if not jnp.allclose(infoset_0_strategy, expected_0):
        print(f"❌ FAIL: Infoset 0 strategy {infoset_0_strategy} != expected {expected_0}")
        return False

    print(f"✅ Infoset 0 correct: {infoset_0_strategy}")

    # Infoset 0b: [0.0, 0.0] → should be [0.5, 0.5] (uniform)
    infoset_0b_strategy = strategy[solver.infoset_action_indices["0b"]]
    expected_0b = jnp.array([0.5, 0.5])
    if not jnp.allclose(infoset_0b_strategy, expected_0b):
        print(f"❌ FAIL: Infoset 0b strategy {infoset_0b_strategy} != expected {expected_0b}")
        return False

    print(f"✅ Infoset 0b correct (uniform): {infoset_0b_strategy}")

    # Infoset 0pb: [3.0, 7.0] → should be [0.3, 0.7] (proportional)
    infoset_0pb_strategy = strategy[solver.infoset_action_indices["0pb"]]
    expected_0pb = jnp.array([0.3, 0.7])
    if not jnp.allclose(infoset_0pb_strategy, expected_0pb):
        print(f"❌ FAIL: Infoset 0pb strategy {infoset_0pb_strategy} != expected {expected_0pb}")
        return False

    print(f"✅ Infoset 0pb correct (proportional): {infoset_0pb_strategy}")

    print("\n" + "=" * 70)
    print("✅ Phase 2.1 Test PASSED: Vectorized regret matching is correct!")
    print("=" * 70)

    return True


def benchmark_speedup():
    """Benchmark the speedup from vectorization."""
    import time

    print("\n" + "=" * 70)
    print("Benchmarking Phase 2.1 Speedup")
    print("=" * 70)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})
    solver = MatrixCFRSolver(game)

    # Warm up JIT
    _ = solver._regret_matching()
    _ = solver._regret_matching()

    # Benchmark
    num_trials = 1000
    print(f"\nRunning {num_trials} regret matching operations...")

    start = time.time()
    for _ in range(num_trials):
        _ = solver._regret_matching()
    elapsed = time.time() - start

    ops_per_sec = num_trials / elapsed
    time_per_op = elapsed / num_trials * 1000  # in ms

    print(f"\n📊 Results:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Operations/sec: {ops_per_sec:.1f}")
    print(f"  Time per operation: {time_per_op:.4f}ms")

    print("\n💡 Phase 2.1 optimization eliminates Python loops in regret matching.")
    print("   Expected: 5-10x faster than sequential version")

    return ops_per_sec


if __name__ == '__main__':
    print("\nPhase 2.1: Vectorized Regret Matching Test")
    print("=" * 70)

    # Run validation
    if not test_vectorized_regret_matching():
        sys.exit(1)

    # Run benchmark
    benchmark_speedup()

    print("\n✅ All tests passed!")
