#!/usr/bin/env python3
"""
Compare convergence speed of External MCCFR vs CFR+ using EXACT nash_conv.

This test uses OpenSpiel's exact nash_conv calculation instead of sampled
exploitability, avoiding measurement noise and methodological issues.

For small games (like 2p_5bb_fchpa_tiny.json), nash_conv is fast enough
to use directly.

Usage:
    source ~/open_spiel/venv/bin/activate
    python test_convergence_comparison.py
"""

import time
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import cfr
import pyspiel
from game_config import PokerGameConfig


def test_algorithm_convergence(algorithm_name, solver, game, iterations=100000, check_interval=10000, is_cfr_plus=False):
    """
    Test convergence of an algorithm using exact nash_conv.

    Args:
        algorithm_name: Name of algorithm (for display)
        solver: Solver object with iteration() or evaluate_and_update_policy() method
        game: OpenSpiel game object
        iterations: Number of iterations to run
        check_interval: How often to check nash_conv
        is_cfr_plus: If True, use evaluate_and_update_policy() instead of iteration()

    Returns:
        List of (iteration, nash_conv, time_elapsed) tuples
    """
    print(f"\n{'='*80}")
    print(f"Testing {algorithm_name}")
    print(f"{'='*80}")

    results = []
    start_time = time.time()

    print(f"Running {iterations:,} iterations...")
    print(f"Checking nash_conv every {check_interval:,} iterations")
    print()

    for i in range(1, iterations + 1):
        if is_cfr_plus:
            solver.evaluate_and_update_policy()
        else:
            solver.iteration()

        # Check nash_conv periodically
        if i % check_interval == 0:
            # Get average policy
            policy = solver.average_policy()

            # Compute exact nash_conv (fast for tiny games)
            nash_start = time.time()
            nash_conv_value = exploitability.nash_conv(game, policy)
            nash_time = time.time() - nash_start

            elapsed = time.time() - start_time
            it_per_sec = i / elapsed

            print(f"  [{i:,}/{iterations:,}] "
                  f"Nash conv: {nash_conv_value:8.4f} | "
                  f"Exploit: {nash_conv_value/2:8.4f} | "
                  f"Time: {elapsed:6.1f}s | "
                  f"{it_per_sec:6.1f} it/s | "
                  f"nash_conv took {nash_time:.3f}s")

            results.append((i, nash_conv_value, elapsed))

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s ({iterations/total_time:.1f} it/s)")

    return results


def main():
    """Run comparison test."""
    print("\n" + "="*80)
    print("CONVERGENCE COMPARISON: External MCCFR vs CFR+")
    print("="*80)
    print("\nUsing EXACT nash_conv (no sampling noise)")
    print("Testing on tiny 2-player game for fast iteration")

    # Create tiny 2-player game
    config = PokerGameConfig.from_json('configs/2p_5bb_fchpa_tiny.json')
    game = config.create_game()

    # Test parameters
    iterations = 100000
    check_interval = 10000

    # Test External MCCFR
    mccfr_solver = external_sampling_mccfr.ExternalSamplingSolver(
        game,
        average_type=external_sampling_mccfr.AverageType.SIMPLE
    )
    mccfr_results = test_algorithm_convergence(
        "External MCCFR (SIMPLE averaging)",
        mccfr_solver,
        game,
        iterations,
        check_interval
    )

    # Test CFR+
    cfrplus_solver = cfr.CFRPlusSolver(game)
    cfrplus_results = test_algorithm_convergence(
        "CFR+",
        cfrplus_solver,
        game,
        iterations,
        check_interval,
        is_cfr_plus=True
    )

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\n{'Iteration':<12} {'MCCFR':<12} {'CFR+':<12} {'MCCFR Better':<15} {'Time MCCFR':<12} {'Time CFR+':<12}")
    print("-" * 85)

    for (iter_mccfr, nash_mccfr, time_mccfr), (iter_cfr, nash_cfr, time_cfr) in zip(mccfr_results, cfrplus_results):
        improvement = (nash_mccfr - nash_cfr) / nash_mccfr * 100 if nash_mccfr > 0 else 0
        print(f"{iter_mccfr:<12} {nash_mccfr:<12.4f} {nash_cfr:<12.4f} {improvement:>13.1f}% "
              f"{time_mccfr:<12.1f} {time_cfr:<12.1f}")

    # Final comparison
    _, final_nash_mccfr, final_time_mccfr = mccfr_results[-1]
    _, final_nash_cfr, final_time_cfr = cfrplus_results[-1]

    improvement = (final_nash_mccfr - final_nash_cfr) / final_nash_mccfr * 100 if final_nash_mccfr > 0 else 0

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"External MCCFR final nash_conv: {final_nash_mccfr:.4f} (exploit: {final_nash_mccfr/2:.4f})")
    print(f"CFR+ final nash_conv:           {final_nash_cfr:.4f} (exploit: {final_nash_cfr/2:.4f})")
    print(f"Convergence improvement:        {improvement:.1f}%")
    print()
    print(f"External MCCFR total time: {final_time_mccfr:.1f}s")
    print(f"CFR+ total time:           {final_time_cfr:.1f}s")
    print(f"Speed ratio:               {final_time_cfr/final_time_mccfr:.2f}x")
    print()

    if improvement > 10:
        print(f"✓ CFR+ converges {abs(improvement):.1f}% better than External MCCFR")
    elif improvement < -10:
        print(f"✓ External MCCFR converges {abs(improvement):.1f}% better than CFR+")
    else:
        print(f"≈ Both algorithms converge similarly (within 10%)")

    print()
    print("KEY INSIGHT:")
    print("-" * 80)
    print("For convergence comparison, nash_conv is the right metric because:")
    print("  1. It's EXACT (no sampling variance)")
    print("  2. It's FAST for small games (0.3s per check)")
    print("  3. It works for 2+ player games")
    print("  4. It measures distance from Nash equilibrium correctly")


if __name__ == '__main__':
    main()
