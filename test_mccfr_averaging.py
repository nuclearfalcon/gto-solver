#!/usr/bin/env python3
"""
Test External MCCFR with different averaging types.

Compares SIMPLE vs FULL averaging to see convergence improvements.

Usage:
    source ~/open_spiel/venv/bin/activate
    python test_mccfr_averaging.py
"""

import time
from open_spiel.python.algorithms import external_sampling_mccfr
from game_config import PokerGameConfig
from exploitability_metrics import SampledExploitabilityCalculator


def test_averaging_type(avg_type_name, avg_type, iterations=50000, check_interval=10000):
    """
    Test External MCCFR with a specific averaging type.

    Args:
        avg_type_name: Name of averaging type (for display)
        avg_type: AverageType enum value
        iterations: Number of iterations to run
        check_interval: How often to check exploitability

    Returns:
        List of (iteration, exploitability) tuples
    """
    print(f"\n{'='*80}")
    print(f"Testing External MCCFR with {avg_type_name} averaging")
    print(f"{'='*80}")

    # Create tiny 2-player game
    config = PokerGameConfig.from_json('configs/2p_5bb_fchpa_tiny.json')
    game = config.create_game()

    # Create solver with specific averaging type
    solver = external_sampling_mccfr.ExternalSamplingSolver(
        game,
        average_type=avg_type
    )

    results = []
    start_time = time.time()

    print(f"Running {iterations:,} iterations...")
    print(f"Checking exploitability every {check_interval:,} iterations (low accuracy)")

    for i in range(1, iterations + 1):
        solver.iteration()

        # Check exploitability periodically
        if i % check_interval == 0:
            # Get average policy
            policy = solver.average_policy()

            # Compute exploitability with low sample count (fast)
            calc = SampledExploitabilityCalculator(game, policy)
            exploit_result = calc.calculate(
                confidence_level=0.95,
                max_ci_width=0.10,  # Accept 10% CI width (fast)
                min_samples=100,
                max_samples=1000
            )

            exploitability = exploit_result['exploitability']
            elapsed = time.time() - start_time
            it_per_sec = i / elapsed

            print(f"  [{i:,}/{iterations:,}] "
                  f"Exploit: {exploitability:.2f} | "
                  f"Time: {elapsed:.1f}s | "
                  f"{it_per_sec:.1f} it/s")

            results.append((i, exploitability))

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s ({iterations/total_time:.1f} it/s)")

    return results


def main():
    """Run comparison test."""
    print("\n" + "="*80)
    print("EXTERNAL MCCFR AVERAGING TYPE COMPARISON")
    print("="*80)
    print("\nTesting if FULL averaging converges faster than SIMPLE averaging")
    print("This is one of the 'quick win' optimizations for MCCFR")

    # Test parameters
    iterations = 50000
    check_interval = 10000

    # Test SIMPLE averaging (default)
    simple_results = test_averaging_type(
        "SIMPLE",
        external_sampling_mccfr.AverageType.SIMPLE,
        iterations,
        check_interval
    )

    # Test FULL averaging (linear/weighted)
    full_results = test_averaging_type(
        "FULL",
        external_sampling_mccfr.AverageType.FULL,
        iterations,
        check_interval
    )

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\n{'Iteration':<12} {'SIMPLE':<15} {'FULL':<15} {'Improvement':<15}")
    print("-" * 60)

    for (iter_simple, exploit_simple), (iter_full, exploit_full) in zip(simple_results, full_results):
        improvement = (exploit_simple - exploit_full) / exploit_simple * 100
        print(f"{iter_simple:<12} {exploit_simple:<15.2f} {exploit_full:<15.2f} {improvement:>13.1f}%")

    # Final comparison
    final_simple = simple_results[-1][1]
    final_full = full_results[-1][1]
    final_improvement = (final_simple - final_full) / final_simple * 100

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"SIMPLE averaging final exploitability: {final_simple:.2f}")
    print(f"FULL averaging final exploitability:   {final_full:.2f}")
    print(f"Improvement: {final_improvement:.1f}%")

    if final_improvement > 5:
        print(f"\n✓ FULL averaging is {final_improvement:.1f}% better than SIMPLE!")
        print("  Recommendation: Use FULL averaging by default")
    elif final_improvement < -5:
        print(f"\n✗ FULL averaging is {abs(final_improvement):.1f}% worse than SIMPLE")
        print("  Recommendation: Keep using SIMPLE averaging")
    else:
        print(f"\n≈ Both averaging types perform similarly (±5%)")
        print("  Recommendation: Either is fine, but FULL may converge faster")


if __name__ == '__main__':
    main()
