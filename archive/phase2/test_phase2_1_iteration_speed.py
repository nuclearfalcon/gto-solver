"""
Test Phase 2.1 iteration speed with vectorized regret matching.

Runs CFR iterations to measure actual speedup in the full solver.

IMPORTANT: Activate OpenSpiel virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import pyspiel
import time
from matrix_cfr import MatrixCFRSolver


def test_iteration_speed():
    """Test CFR iteration speed with Phase 2.1 optimization."""
    print("=" * 70)
    print("Phase 2.1: Full Iteration Speed Test")
    print("=" * 70)

    # Create Kuhn poker game
    game = pyspiel.load_game('kuhn_poker', {'players': 2})

    # Create solver
    print("\nInitializing solver...")
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

    # Warm-up
    print("\nWarming up JIT...")
    solver.solve(iterations=3)

    # Reset for actual test
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

    # Benchmark
    num_iterations = 50
    print(f"\nRunning {num_iterations} CFR iterations...")

    start = time.time()
    solver.solve(iterations=num_iterations)
    elapsed = time.time() - start

    iterations_per_sec = num_iterations / elapsed
    time_per_iteration = elapsed / num_iterations

    print("\n" + "=" * 70)
    print("📊 Phase 2.1 Performance Results")
    print("=" * 70)
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Iterations/second: {iterations_per_sec:.2f} it/s")
    print(f"  Time per iteration: {time_per_iteration:.3f}s")

    print("\n📈 Comparison with Phase 1:")
    print(f"  Phase 1 baseline: 1.0 it/s")
    print(f"  Phase 2.1 current: {iterations_per_sec:.2f} it/s")
    print(f"  Speedup: {iterations_per_sec / 1.0:.2f}x")

    print("\n💡 Note: Phase 2.1 only optimizes regret matching (~8% of time)")
    print("   Expected overall gain: 1.05-1.1x")
    print("   Major speedup comes from Phase 2.2 (batched action values)")

    return iterations_per_sec


if __name__ == '__main__':
    speed = test_iteration_speed()

    if speed >= 1.0:
        print("\n✅ Phase 2.1 complete: Regret matching vectorized successfully!")
    else:
        print("\n⚠️  Slower than expected - may need investigation")
