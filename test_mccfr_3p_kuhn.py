#!/usr/bin/env python3
"""
Test External MCCFR on 3-player Kuhn Poker with high iteration count.

3-player Kuhn poker is a perfect test case:
- Small game tree (nash_conv is very fast)
- Requires nash_conv (not exploitability) since it's 3-player
- Well-studied game with known properties
- Can run millions of iterations to test convergence

Usage:
    source ~/open_spiel/venv/bin/activate
    python test_mccfr_3p_kuhn.py
"""

import time
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr
import pyspiel


def test_3p_kuhn_convergence(iterations=1_000_000, check_interval=50_000):
    """
    Test External MCCFR on 3-player Kuhn poker with high iteration count.

    Args:
        iterations: Number of iterations to run (default 1M)
        check_interval: How often to check nash_conv

    Returns:
        List of (iteration, nash_conv, time_elapsed) tuples
    """
    print("="*80)
    print("EXTERNAL MCCFR ON 3-PLAYER KUHN POKER")
    print("="*80)
    print()

    # Create 3-player Kuhn poker
    game = pyspiel.load_game("kuhn_poker", {"players": 3})

    print(f"Game: {game.get_type().long_name}")
    print(f"Players: {game.num_players()}")
    print(f"Max game length: {game.max_game_length()}")
    print()

    # Create External MCCFR solver with SIMPLE averaging
    solver = external_sampling_mccfr.ExternalSamplingSolver(
        game,
        average_type=external_sampling_mccfr.AverageType.SIMPLE
    )

    print(f"Algorithm: External Sampling MCCFR (SIMPLE averaging)")
    print(f"Target iterations: {iterations:,}")
    print(f"Checking nash_conv every {check_interval:,} iterations")
    print()

    # First, test how fast nash_conv is
    print("Testing nash_conv speed...")
    for i in range(100):
        solver.iteration()
    policy = solver.average_policy()
    nash_start = time.time()
    nash_conv_value = exploitability.nash_conv(game, policy)
    nash_time = time.time() - nash_start
    print(f"Nash_conv time: {nash_time*1000:.2f}ms (very fast!)")
    print(f"Initial nash_conv: {nash_conv_value:.4f}")
    print()

    # Run convergence test
    results = []
    start_time = time.time()
    last_report_time = start_time

    print("Starting convergence test...")
    print()

    for i in range(101, iterations + 1):
        solver.iteration()

        # Check nash_conv periodically
        if i % check_interval == 0 or i == iterations:
            policy = solver.average_policy()

            nash_start = time.time()
            nash_conv_value = exploitability.nash_conv(game, policy)
            nash_elapsed = time.time() - nash_start

            total_elapsed = time.time() - start_time
            it_per_sec = i / total_elapsed

            # Estimate time remaining
            remaining_iters = iterations - i
            eta_seconds = remaining_iters / it_per_sec if it_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60

            print(f"[{i:,}/{iterations:,}] "
                  f"Nash conv: {nash_conv_value:8.6f} | "
                  f"Time: {total_elapsed:6.1f}s | "
                  f"{it_per_sec:8.1f} it/s | "
                  f"nash_conv: {nash_elapsed*1000:5.1f}ms | "
                  f"ETA: {eta_minutes:.1f}m")

            results.append((i, nash_conv_value, total_elapsed))
            last_report_time = time.time()

    total_time = time.time() - start_time
    print()
    print(f"Completed {iterations:,} iterations in {total_time:.1f}s ({iterations/total_time:.1f} it/s)")

    return results


def analyze_results(results):
    """Analyze convergence results."""
    print()
    print("="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    print()

    # Print full results table
    print(f"{'Iteration':<12} {'Nash Conv':<15} {'Time (s)':<12} {'Improvement':<15}")
    print("-" * 60)

    for i, (iteration, nash_conv, elapsed) in enumerate(results):
        if i == 0:
            improvement = ""
        else:
            prev_nash = results[i-1][1]
            improvement_pct = (prev_nash - nash_conv) / prev_nash * 100 if prev_nash > 0 else 0
            improvement = f"{improvement_pct:>13.2f}%"

        print(f"{iteration:<12,} {nash_conv:<15.6f} {elapsed:<12.1f} {improvement:<15}")

    # Summary statistics
    initial_nash = results[0][1]
    final_nash = results[-1][1]
    total_improvement = (initial_nash - final_nash) / initial_nash * 100

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Initial nash_conv:     {initial_nash:.6f}")
    print(f"Final nash_conv:       {final_nash:.6f}")
    print(f"Total improvement:     {total_improvement:.2f}%")
    print(f"Absolute reduction:    {initial_nash - final_nash:.6f}")
    print()

    # Check convergence quality
    if final_nash < 0.001:
        print(f"✓ EXCELLENT convergence! Nash_conv < 0.001")
    elif final_nash < 0.01:
        print(f"✓ GOOD convergence! Nash_conv < 0.01")
    elif final_nash < 0.1:
        print(f"≈ Moderate convergence. Nash_conv < 0.1")
    else:
        print(f"⚠ Poor convergence. Nash_conv still > 0.1")

    # Convergence rate analysis
    print()
    print("Convergence rate by phase:")
    print("-" * 60)

    quarter = len(results) // 4
    phases = [
        ("First quarter", 0, quarter),
        ("Second quarter", quarter, quarter*2),
        ("Third quarter", quarter*2, quarter*3),
        ("Fourth quarter", quarter*3, len(results))
    ]

    for phase_name, start_idx, end_idx in phases:
        if end_idx > start_idx:
            start_nash = results[start_idx][1]
            end_nash = results[end_idx-1][1]
            improvement = (start_nash - end_nash) / start_nash * 100 if start_nash > 0 else 0
            print(f"{phase_name:<20}: {start_nash:.6f} → {end_nash:.6f} ({improvement:>6.2f}%)")


def main():
    """Run test."""
    print("\n" + "="*80)
    print("3-PLAYER KUHN POKER: HIGH-VOLUME MCCFR CONVERGENCE TEST")
    print("="*80)
    print()
    print("Testing hypothesis: Can External MCCFR reach strong convergence")
    print("on 3-player Kuhn poker with sheer volume of iterations?")
    print()
    print("Using EXACT nash_conv (no sampling) to measure convergence.")
    print()

    # Run 1 million iterations (adjust based on speed)
    results = test_3p_kuhn_convergence(
        iterations=1_000_000,
        check_interval=50_000
    )

    # Analyze results
    analyze_results(results)

    print()
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()
    print("1. Nash_conv is the CORRECT metric for 3-player games")
    print("2. It's exact (no sampling noise) and very fast for Kuhn poker")
    print("3. High iteration count tests if MCCFR can converge with volume")
    print("4. This validates our nash_conv methodology before using on larger games")


if __name__ == '__main__':
    main()
