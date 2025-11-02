#!/usr/bin/env python3
"""
Test Phase 4 Optimizations on Leduc Poker - Scaling Validation

This is the CRITICAL TEST to validate that Phase 4 optimizations scale properly.

Leduc poker is 12-24x larger than Kuhn:
- Kuhn: 12 infosets, 24 actions per iteration
- Leduc: 288 infosets, ~140 actions per player = 280 actions per iteration

Expected results if Phase 4 works:
- Leduc should be FASTER than Kuhn (better GPU utilization with larger batches)
- Target: 10-20 it/s on Leduc
- If Leduc < 5 it/s: Real bottleneck exists
- If Leduc >= 10 it/s: Phase 4 optimizations scale correctly!

Requirements:
- source ~/open_spiel/venv/bin/activate

Phase 4 Optimizations Being Tested:
- OPT-4.1: Array-based CF value extraction (vectorized gather/scatter)
- OPT-4.2: Vectorized regret updates (eliminates nested loops)
"""

import time
import pyspiel
import numpy as np
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver


def test_leduc_performance(num_runs=3, iterations=50):
    """
    Test Leduc poker performance to validate Phase 4 scaling.

    Args:
        num_runs: Number of benchmark runs
        iterations: Iterations per run (50 instead of 100 since Leduc is larger)
    """
    print("=" * 70)
    print("LEDUC POKER SCALING VALIDATION")
    print("=" * 70)
    print()
    print("Critical test: Does Phase 4 scale to larger games?")
    print()

    # Create Leduc poker
    game = pyspiel.load_game("leduc_poker")

    # Create solver to get game stats
    print("Initializing solver and analyzing game size...")
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)

    print()
    print("Game Statistics:")
    print(f"  Nodes: {solver.matrix_repr.num_nodes}")
    print(f"  Infosets: {solver.matrix_repr.num_infosets}")
    print(f"  Infoset-actions: {solver.matrix_repr.num_infoset_actions}")
    print(f"  Players: {solver.matrix_repr.num_players}")
    print()

    # Calculate expected batch size
    actions_per_player = solver.matrix_repr.num_infoset_actions // solver.matrix_repr.num_players
    batch_size = actions_per_player * 2  # Both players batched together

    print("Phase 4 Batch Processing:")
    print(f"  Actions per player: ~{actions_per_player}")
    print(f"  Total batch size: ~{batch_size} (both players)")
    print(f"  vs Kuhn batch size: 24")
    print(f"  Batch size increase: {batch_size / 24:.1f}x")
    print()

    # Comparison to Kuhn
    kuhn_infosets = 12
    kuhn_actions = 24
    scaling_factor = solver.matrix_repr.num_infosets / kuhn_infosets

    print("Scaling vs Kuhn Poker:")
    print(f"  Infosets: {scaling_factor:.1f}x larger")
    print(f"  Actions: {solver.matrix_repr.num_infoset_actions / kuhn_actions:.1f}x larger")
    print()

    print("=" * 70)
    print("BENCHMARK")
    print("=" * 70)
    print()

    speeds = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        print("-" * 70)

        # Create fresh solver
        solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)

        # Warmup
        print("  Warmup: 10 iterations...")
        solver.solve(iterations=10, progress_interval=1000)

        # Benchmark
        print(f"  Benchmark: {iterations} iterations...")
        start_time = time.time()
        solver.solve(iterations=iterations, progress_interval=10)
        elapsed = time.time() - start_time

        speed = iterations / elapsed
        speeds.append(speed)

        print(f"  ✓ Run {run + 1}: {speed:.2f} it/s ({elapsed:.2f}s)")
        print()

    # Calculate statistics
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)

    print()
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print()
    print(f"Leduc performance: {mean_speed:.2f} ± {std_speed:.2f} it/s")
    print()

    # Compare to Kuhn and Phase 3
    kuhn_phase4_speed = 2.86  # From test_phase4_kuhn_benchmark.py
    kuhn_phase3_speed = 2.66  # Phase 3 baseline

    print("Comparison:")
    print(f"  Kuhn Phase 3: {kuhn_phase3_speed:.2f} it/s")
    print(f"  Kuhn Phase 4: {kuhn_phase4_speed:.2f} it/s")
    print(f"  Leduc Phase 4: {mean_speed:.2f} it/s")
    print()

    # Scaling analysis
    if mean_speed > kuhn_phase4_speed:
        speedup_ratio = mean_speed / kuhn_phase4_speed
        print(f"🎉 Leduc is {speedup_ratio:.2f}x FASTER than Kuhn!")
        print(f"   This confirms Phase 4 optimizations scale with batch size!")
    else:
        slowdown_ratio = kuhn_phase4_speed / mean_speed
        print(f"⚠️  Leduc is {slowdown_ratio:.2f}x SLOWER than Kuhn")
        print(f"   (Expected - Leduc has more complex game tree)")

    print()

    # Assessment
    target_speed = 10.0

    print("Assessment:")
    if mean_speed >= target_speed:
        print(f"  ✅ SUCCESS! Achieved {mean_speed:.2f} it/s (target: {target_speed} it/s)")
        print(f"  ✅ Phase 4 optimizations scale correctly!")
        print(f"  ✅ Ready for Hold'em preparation!")
    elif mean_speed >= 5.0:
        progress = (mean_speed / target_speed) * 100
        print(f"  ⚠️  Partial success: {progress:.1f}% of target")
        print(f"  → Phase 4 helps, but more optimization needed")
        print(f"  → Consider Phase 4.3-4.4 or profiling")
    else:
        print(f"  ❌ Below expectations: {mean_speed:.2f} it/s < 5 it/s")
        print(f"  → Need to profile and find real bottleneck")

    print()

    # Test learning
    print("=" * 70)
    print("LEARNING VERIFICATION")
    print("=" * 70)
    print()
    print("Running 100 iterations to verify learning...")

    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)
    solver.solve(iterations=100, progress_interval=20)

    policy = solver.get_average_policy()

    # Check for non-uniform strategies
    non_uniform_count = 0
    uniform_threshold = 0.05

    for infoset, probs in policy.items():
        num_actions = len(probs)
        uniform_prob = 1.0 / num_actions
        is_non_uniform = any(abs(p - uniform_prob) > uniform_threshold for p in probs)
        if is_non_uniform:
            non_uniform_count += 1

    print(f"\nNon-uniform infosets: {non_uniform_count}/{len(policy)}")

    if non_uniform_count >= 10:
        print("✅ Learning confirmed on Leduc!")
    else:
        print("⚠️  Limited learning - may need more iterations")

    print()

    return mean_speed, speeds


def main():
    """Run Leduc scaling validation."""
    print()
    print("=" * 70)
    print("PHASE 4 SCALING VALIDATION - LEDUC POKER")
    print("=" * 70)
    print()
    print("This test validates that Phase 4 optimizations (OPT-4.1 + 4.2)")
    print("scale properly to larger games.")
    print()
    print("Key question: Does larger batch size → better GPU utilization?")
    print()

    mean_speed, speeds = test_leduc_performance(num_runs=3, iterations=50)

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    if mean_speed >= 10.0:
        print("🎉 Phase 4 optimizations VALIDATED!")
        print()
        print("Next steps:")
        print("  1. Document Phase 4 results")
        print("  2. Commit changes")
        print("  3. Begin Hold'em preparation (chunking, memory management)")
        print()
        print("Phase 4 successfully eliminates dictionary-based bottlenecks")
        print("and scales to larger games. Ready for 3-player Hold'em!")
    elif mean_speed >= 5.0:
        print("✓ Phase 4 shows improvement, but below expectations")
        print()
        print("Options:")
        print("  A. Implement Phase 4.3-4.4 (vectorize override building)")
        print("  B. Profile Leduc to find remaining bottlenecks")
        print("  C. Proceed to Hold'em anyway (might scale further)")
    else:
        print("⚠️  Need to investigate bottleneck")
        print()
        print("Recommended:")
        print("  1. Profile Leduc run to find actual bottleneck")
        print("  2. Check if thermal throttling is occurring")
        print("  3. Consider Phase 4.3-4.4 optimizations")

    print()


if __name__ == "__main__":
    main()
