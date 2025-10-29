#!/usr/bin/env python3
"""
GTO Poker Solver - Algorithm Comparison

Test all solving algorithms sequentially on the same game configuration
and generate comparison report.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python solve_and_compare.py --config configs/2p_5bb.json --iterations 10000

    python solve_and_compare.py --config configs/2p_10bb.json --iterations 100000 --algorithms cfr_plus cpp_cfr_plus external_mccfr
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from game_config import PokerGameConfig
from poker_solver import UnifiedPokerSolver
from solver_metrics import AdaptiveSchedule
from solver_logger import ComparisonLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare multiple CFR algorithms on same game config',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Config
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to game configuration JSON file'
    )

    # Algorithms to test
    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='+',
        default=None,
        help='Algorithms to test (default: all except outcome_mccfr)'
    )

    # Solving parameters
    parser.add_argument(
        '--iterations',
        type=int,
        required=True,
        help='Number of iterations per algorithm'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results/)'
    )

    # Exploitability schedule
    parser.add_argument(
        '--check-exploitability',
        type=str,
        default='adaptive',
        choices=['adaptive', 'fixed'],
        help='Exploitability checking schedule (default: adaptive)'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=50000,
        help='Fixed interval for exploitability checks (default: 50000)'
    )
    parser.add_argument(
        '--use-full-exploitability',
        action='store_true',
        help='Use FULL exploitability (WARNING: Massive memory! Only for tiny test games. Default: False, uses sampled)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load game configuration
    try:
        game_config = PokerGameConfig.from_json(args.config)
        print(f"Loaded game configuration: {game_config}")
    except Exception as e:
        print(f"ERROR: Failed to load config from {args.config}: {e}")
        return 1

    # Determine algorithms to test
    if args.algorithms:
        algorithms = args.algorithms
    else:
        # Default: focus on practical algorithms (C++ versions + research variants)
        # Skip: vanilla_cfr (too slow), cfr_plus (C++ version is better), outcome_mccfr (too slow)
        algorithms = [
            'dcfr',
            'lcfr',
            'external_mccfr',
            'cpp_cfr',
            'cpp_cfr_plus',
        ]

    # Validate algorithms
    for algo in algorithms:
        if algo not in UnifiedPokerSolver.SUPPORTED_ALGORITHMS:
            print(f"ERROR: Unknown algorithm: {algo}")
            print(f"Supported: {list(UnifiedPokerSolver.SUPPORTED_ALGORITHMS.keys())}")
            return 1

    # Create comparison logger
    comp_logger = ComparisonLogger()
    comp_logger.log_comparison_start(len(algorithms), str(game_config))

    # Create adaptive schedule (shared across all algorithms)
    if args.check_exploitability == 'adaptive':
        schedule = AdaptiveSchedule()
    else:
        schedule = AdaptiveSchedule(
            thresholds=[0],
            intervals=[args.check_interval]
        )

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run each algorithm
    results = []

    for i, algorithm in enumerate(algorithms, 1):
        comp_logger.log_algorithm_start(
            algorithm_name=UnifiedPokerSolver.get_algorithm_description(algorithm),
            number=i,
            total=len(algorithms)
        )

        try:
            # Create solver
            solver = UnifiedPokerSolver(
                game_config=game_config,
                algorithm=algorithm
            )

            # Run solve
            solver.solve(
                max_iterations=args.iterations,
                adaptive_schedule=schedule,
                checkpoint_interval=None,  # No checkpoints during comparison
                progress_interval=100,  # Progress updates every 100 iterations
                skip_initial_exploitability=True,  # Skip initial check to save memory
                use_sampled_exploitability=not args.use_full_exploitability  # Default: True (sampled)
            )

            # Get summary
            summary = solver.get_metrics_summary()
            results.append(summary)

            # Save metrics for this algorithm
            base_name = f"{algorithm}_{game_config.get_short_description()}_{timestamp}"
            csv_path = output_dir / f"{base_name}_metrics.csv"
            json_path = output_dir / f"{base_name}_summary.json"
            solver.save_metrics(str(csv_path), str(json_path))

            # Save policy
            policy_path = output_dir / f"{base_name}_policy.pkl"
            solver.save_policy(str(policy_path))

            # Log completion
            comp_logger.log_algorithm_complete(
                algorithm_name=UnifiedPokerSolver.get_algorithm_description(algorithm),
                summary=summary
            )

        except KeyboardInterrupt:
            print("\n\nComparison interrupted by user")
            break
        except Exception as e:
            print(f"\nERROR: Algorithm {algorithm} failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next algorithm
            continue

    # Generate comparison report
    comp_logger.log_comparison_complete(len(results))

    if results:
        comp_logger.log_comparison_summary_table(results)

        # Save comparison report
        report_path = output_dir / f"comparison_{game_config.get_short_description()}_{timestamp}.md"
        generate_markdown_report(results, game_config, args.iterations, report_path)
        print(f"\nComparison report saved to: {report_path}")

    return 0


def generate_markdown_report(results, game_config, iterations, report_path):
    """
    Generate markdown comparison report.

    Args:
        results: List of summary dictionaries
        game_config: PokerGameConfig instance
        iterations: Number of iterations run
        report_path: Path to save report
    """
    with open(report_path, 'w') as f:
        f.write(f"# Algorithm Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Game configuration
        f.write(f"## Game Configuration\n\n")
        f.write(f"- **Players:** {game_config.num_players}\n")
        f.write(f"- **Stack sizes:** {game_config.stack_sizes}\n")
        f.write(f"- **Blinds:** {game_config.blinds}\n")
        f.write(f"- **Betting abstraction:** {game_config.betting_abstraction}\n")
        f.write(f"- **Effective stack (BB):** {game_config.get_big_blinds():.1f}\n")
        f.write(f"- **Target iterations:** {iterations:,}\n\n")

        # Results table
        f.write(f"## Results Summary\n\n")
        f.write(f"| Algorithm | Final Exploit | Time (s) | Avg Speed (it/s) | Peak Memory (MB) |\n")
        f.write(f"|-----------|---------------|----------|------------------|------------------|\n")

        for result in results:
            algo = result['algorithm']
            exploit = result['final_exploitability']
            time = result['total_time']
            speed = result['avg_iters_per_sec']
            memory = result['peak_memory_mb']

            f.write(f"| {algo} | {exploit:.6f} | {time:.1f} | {speed:.1f} | {memory:.1f} |\n")

        # Best performers
        f.write(f"\n## Best Performers\n\n")

        if results:
            # Lowest exploitability
            best_exploit = min(results, key=lambda x: x['final_exploitability'])
            f.write(f"- **Lowest Exploitability:** {best_exploit['algorithm']} ")
            f.write(f"({best_exploit['final_exploitability']:.6f})\n")

            # Fastest
            fastest = max(results, key=lambda x: x['avg_iters_per_sec'])
            f.write(f"- **Fastest (it/s):** {fastest['algorithm']} ")
            f.write(f"({fastest['avg_iters_per_sec']:.1f} it/s)\n")

            # Lowest memory
            lowest_mem = min(results, key=lambda x: x['peak_memory_mb'])
            f.write(f"- **Lowest Memory:** {lowest_mem['algorithm']} ")
            f.write(f"({lowest_mem['peak_memory_mb']:.1f} MB)\n")

        # Recommendations
        f.write(f"\n## Recommendations\n\n")
        f.write(f"Based on these results:\n\n")
        f.write(f"1. For best convergence (lowest exploitability), use: **{best_exploit['algorithm']}**\n")
        f.write(f"2. For fastest iterations, use: **{fastest['algorithm']}**\n")
        f.write(f"3. For lowest memory usage, use: **{lowest_mem['algorithm']}**\n\n")

        f.write(f"Note: C++ implementations generally provide 5-10x speedup over Python versions.\n")
        f.write(f"MCCFR algorithms use less memory but may require more iterations to converge.\n")


if __name__ == '__main__':
    sys.exit(main())
