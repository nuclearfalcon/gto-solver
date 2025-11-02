#!/usr/bin/env python3
"""
Test Phase 4 Critical Optimizations (OPT-4.1 + 4.2) on Kuhn Poker

Tests:
1. Correctness: Verify learning still occurs
2. Performance: Benchmark speed (target: 5-6 it/s)
3. Comparison: Compare to Phase 3 baseline (2.66 it/s)

Requirements:
- source ~/open_spiel/venv/bin/activate

Phase 4 Optimizations Implemented:
- OPT-4.1: Array-based CF value extraction (eliminates 25% bottleneck)
- OPT-4.2: Vectorized regret updates (eliminates 20% bottleneck)

Expected speedup: 2x (2.66 → 5-6 it/s)
"""

import time
import pyspiel
import numpy as np
from matrix_cfr.game_to_matrix import GameTreeConverter
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver


def test_correctness(iterations=100):
    """Test that learning still occurs after Phase 4 optimizations."""
    print("=" * 70)
    print("CORRECTNESS TEST - Verify Learning")
    print("=" * 70)
    print()

    # Create Kuhn poker
    game = pyspiel.load_game("kuhn_poker")

    # Create solver (converts to matrix representation internally)
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)

    print(f"Game: {solver.matrix_repr.num_nodes} nodes, {solver.matrix_repr.num_infosets} infosets")
    print(f"Infoset-actions: {solver.matrix_repr.num_infoset_actions}")
    print()

    # Solve
    print(f"Running {iterations} iterations...")
    solver.solve(iterations=iterations, progress_interval=20)

    # Get final strategy
    avg_policy = solver.get_average_policy()

    # Check for non-uniform strategies (learning indicator)
    non_uniform_count = 0
    uniform_threshold = 0.01

    print("\nFinal strategies:")
    for infoset in sorted(avg_policy.keys()):
        probs = avg_policy[infoset]
        num_actions = len(probs)
        uniform_prob = 1.0 / num_actions

        # Check if any action deviates significantly from uniform
        is_non_uniform = any(abs(p - uniform_prob) > uniform_threshold for p in probs)

        if is_non_uniform:
            non_uniform_count += 1
            marker = "✓"
        else:
            marker = " "

        probs_str = ", ".join(f"{p:.3f}" for p in probs)
        print(f"  {marker} {infoset:3s}: [{probs_str}]")

    print()
    print(f"Non-uniform infosets: {non_uniform_count}/{len(avg_policy)}")

    if non_uniform_count >= 3:
        print("✅ SUCCESS! Learning is occurring!")
        return True
    else:
        print("❌ FAILURE! No learning detected!")
        return False


def benchmark_performance(num_runs=3, iterations=100):
    """Benchmark performance across multiple runs."""
    print()
    print("=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()

    # Create Kuhn poker
    game = pyspiel.load_game("kuhn_poker")

    speeds = []

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        print("-" * 70)

        # Create fresh solver
        solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)

        # Warmup
        solver.solve(iterations=10, progress_interval=1000)

        # Benchmark
        start_time = time.time()
        solver.solve(iterations=iterations, progress_interval=20)
        elapsed = time.time() - start_time

        speed = iterations / elapsed
        speeds.append(speed)

        print(f"\n  Run {run + 1}: {speed:.2f} it/s ({elapsed:.2f}s)")

    # Calculate statistics
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)

    print()
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print()
    print(f"Mean speed: {mean_speed:.2f} ± {std_speed:.2f} it/s")
    print()

    # Compare to baseline
    phase3_speed = 2.66  # From test_phase3_ablation.py results
    speedup = mean_speed / phase3_speed

    print("Comparison to Phase 3:")
    print(f"  Phase 3 baseline: {phase3_speed:.2f} it/s")
    print(f"  Phase 4 (OPT 4.1+4.2): {mean_speed:.2f} it/s")
    print(f"  Speedup: {speedup:.2f}x")
    print()

    # Target assessment
    target_speed = 5.0
    if mean_speed >= target_speed:
        print(f"✅ SUCCESS! Achieved target ({target_speed} it/s)")
    else:
        progress = (mean_speed / target_speed) * 100
        print(f"⚠️  Target not met: {progress:.1f}% of {target_speed} it/s target")

    print()

    return mean_speed, speeds


def main():
    """Run complete Phase 4 test suite."""
    print("=" * 70)
    print("PHASE 4 CRITICAL OPTIMIZATIONS TEST SUITE")
    print("=" * 70)
    print()
    print("Optimizations tested:")
    print("  - OPT-4.1: Array-based CF value extraction")
    print("  - OPT-4.2: Vectorized regret updates")
    print()
    print("Expected results:")
    print("  - Correctness: 3+ non-uniform infosets (learning confirmed)")
    print("  - Performance: 5-6 it/s (2x speedup from Phase 3)")
    print()

    # Test 1: Correctness
    correctness_ok = test_correctness(iterations=100)

    # Test 2: Performance
    mean_speed, speeds = benchmark_performance(num_runs=3, iterations=100)

    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    print(f"Correctness: {'✅ PASS' if correctness_ok else '❌ FAIL'}")
    print(f"Performance: {mean_speed:.2f} it/s")
    print()

    if correctness_ok and mean_speed >= 4.0:
        print("🎉 Phase 4 critical optimizations successful!")
        print("   Ready to test scaling on Leduc poker.")
    elif correctness_ok:
        print("✓ Learning preserved, but performance below target")
        print("   May need further optimization or better hardware cooling")
    else:
        print("❌ Critical issues detected - need debugging")

    print()


if __name__ == "__main__":
    main()
