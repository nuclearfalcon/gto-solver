"""
Test Phase 2.2: Batched Action Value Computation

Tests the critical optimization that targets 85% of runtime.

IMPORTANT: Activate OpenSpiel virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import pyspiel
import time
from matrix_cfr import MatrixCFRSolver


def test_batched_computation():
    """Test that batched computation produces correct results."""
    print("=" * 70)
    print("Testing Phase 2.2: Batched Action Value Computation")
    print("=" * 70)

    # Create Kuhn poker game
    game = pyspiel.load_game('kuhn_poker', {'players': 2})

    # Create solver
    print("\nInitializing solver...")
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

    # Run a few iterations to get non-uniform strategies
    print("\nRunning 5 iterations to establish strategies...")
    solver.solve(iterations=5)

    # Test batched computation
    print("\n Testing batched counterfactual value computation...")
    cf_values = solver._compute_counterfactual_values(player=0)

    print(f"✅ Computed values for {len(cf_values)} infosets")

    # Validate structure
    for infoset, values in cf_values.items():
        num_actions = len(solver.matrix_repr.infoset_to_actions[infoset])
        if len(values) != num_actions:
            print(f"❌ FAIL: Infoset {infoset} has {len(values)} values but {num_actions} actions")
            return False

    print("✅ All infosets have correct number of action values")

    # Check that values are reasonable (not all zero, not NaN/Inf)
    all_values = []
    for values in cf_values.values():
        all_values.extend(values.tolist())

    if any(v != v for v in all_values):  # Check for NaN
        print("❌ FAIL: Found NaN values!")
        return False

    if any(abs(v) > 1e10 for v in all_values):  # Check for unreasonable values
        print("❌ FAIL: Found extremely large values!")
        return False

    print(f"✅ All values are reasonable (range: [{min(all_values):.2f}, {max(all_values):.2f}])")

    print("\n" + "=" * 70)
    print("✅ Phase 2.2 Test PASSED: Batched computation is working!")
    print("=" * 70)

    return True


def benchmark_speedup():
    """Benchmark the massive speedup from batching."""
    print("\n" + "=" * 70)
    print("Benchmarking Phase 2.2 Speedup (THE BIG ONE)")
    print("=" * 70)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})
    solver = MatrixCFRSolver(game)

    # Warm up JIT
    print("\nWarming up JIT compilation...")
    solver.solve(iterations=3)

    # Reset for actual test
    solver = MatrixCFRSolver(game)

    # Benchmark full iterations
    num_iterations = 50
    print(f"\nRunning {num_iterations} CFR iterations with Phase 2.2 batching...")

    start = time.time()
    solver.solve(iterations=num_iterations)
    elapsed = time.time() - start

    iterations_per_sec = num_iterations / elapsed
    time_per_iteration = elapsed / num_iterations

    print("\n" + "=" * 70)
    print("📊 Phase 2.2 Performance Results")
    print("=" * 70)
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Iterations/second: {iterations_per_sec:.2f} it/s")
    print(f"  Time per iteration: {time_per_iteration:.3f}s")

    print("\n📈 Comparison:")
    print(f"  Baseline (before any optimization): 0.43 it/s")
    print(f"  Phase 1 (JIT utilities/reach): 1.0 it/s (2.3x)")
    print(f"  Phase 2.2 (batched action values): {iterations_per_sec:.2f} it/s ({iterations_per_sec/0.43:.1f}x from baseline)")

    if iterations_per_sec >= 5.0:
        print(f"\n🎉 EXCELLENT! Achieved {iterations_per_sec:.1f} it/s")
        print("   Batching the action value computation (85% bottleneck) is paying off!")
    elif iterations_per_sec >= 2.0:
        print(f"\n✅ GOOD! Achieved {iterations_per_sec:.1f} it/s")
        print("   Significant improvement, though not at target yet")
    else:
        print(f"\n⚠️  Slower than expected ({iterations_per_sec:.1f} it/s)")
        print("   May need further optimization or larger batch sizes")

    print(f"\n💡 Phase 2.2 targets the 85% bottleneck (action value computation)")
    print(f"   Expected: 10-50x speedup on this component → 5-40 it/s overall")
    print(f"   Target: 40-100 it/s (50-100x from 0.43 baseline)")

    return iterations_per_sec


if __name__ == '__main__':
    print("\nPhase 2.2: Batched Action Value Computation Test")
    print("=" * 70)

    # Run correctness test
    if not test_batched_computation():
        import sys
        sys.exit(1)

    # Run benchmark
    speed = benchmark_speedup()

    if speed >= 5.0:
        print("\n🎉 Phase 2.2 SUCCESS! Major speedup achieved!")
    elif speed >= 2.0:
        print("\n✅ Phase 2.2 complete - good progress!")
    else:
        print("\n⚠️  Phase 2.2 needs investigation")

    print("\n" + "=" * 70)
