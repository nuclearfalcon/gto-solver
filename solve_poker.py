#!/usr/bin/env python3
"""
GTO Poker Solver - Single Algorithm

Solve a poker game configuration using a single CFR algorithm.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python solve_poker.py --config configs/2p_10bb.json --algorithm cfr_plus --iterations 100000

    python solve_poker.py --config configs/3p_20bb.json --algorithm external_mccfr --iterations 1000000

    python solve_poker.py --resume checkpoints/cfr_plus_iter_50000.pkl --iterations 100000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from game_config import PokerGameConfig
from poker_solver import UnifiedPokerSolver
from solver_metrics import AdaptiveSchedule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Solve poker game configuration using CFR',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Config source (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to game configuration JSON file'
    )
    config_group.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint file to resume from'
    )

    # Algorithm selection
    parser.add_argument(
        '--algorithm',
        type=str,
        default='cfr_plus',
        choices=list(UnifiedPokerSolver.SUPPORTED_ALGORITHMS.keys()),
        help='Solving algorithm to use (default: cfr_plus)'
    )

    # Solving parameters
    parser.add_argument(
        '--iterations',
        type=int,
        required=True,
        help='Number of iterations to run'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=None,
        help='Save checkpoint every N iterations (default: no checkpoints)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints (default: checkpoints/)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results/)'
    )
    parser.add_argument(
        '--save-policy',
        action='store_true',
        help='Save final policy to file'
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
        '--progress-interval',
        type=int,
        default=100,
        help='Show progress update every N iterations (default: 100)'
    )
    parser.add_argument(
        '--skip-initial-check',
        action='store_true',
        default=True,
        help='Skip initial exploitability check (saves memory, default: True)'
    )
    parser.add_argument(
        '--no-skip-initial-check',
        action='store_false',
        dest='skip_initial_check',
        help='Do NOT skip initial exploitability check'
    )

    # Exploitability method
    parser.add_argument(
        '--use-full-exploitability',
        action='store_true',
        help='Use FULL exploitability (WARNING: Massive memory usage! Only for tiny test games. Default: False, uses sampled)'
    )
    parser.add_argument(
        '--final-sampled-exploitability',
        action='store_true',
        help='Run high-accuracy sampled exploitability at the end with tighter CI (default: False)'
    )
    parser.add_argument(
        '--sampled-confidence',
        type=float,
        default=0.99,
        help='Confidence level for sampled exploitability (default: 0.99 = 99%%)'
    )
    parser.add_argument(
        '--sampled-ci-width',
        type=float,
        default=0.05,
        help='Target CI width for sampled exploitability during solve (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--sampled-min-samples',
        type=int,
        default=50,
        help='Minimum samples for sampled exploitability (default: 50)'
    )
    parser.add_argument(
        '--sampled-max-samples',
        type=int,
        default=500,
        help='Maximum samples for sampled exploitability (default: 500)'
    )

    # Algorithm-specific parameters
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.5,
        help='DCFR alpha parameter (default: 1.5)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.0,
        help='DCFR beta parameter (default: 0.0)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=2.0,
        help='DCFR gamma parameter (default: 2.0)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.6,
        help='Outcome Sampling MCCFR epsilon parameter (default: 0.6)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load or create game configuration
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        print("Note: Game configuration will be loaded from checkpoint")
        # For simplicity, require --config even when resuming
        print("ERROR: Resume functionality requires specifying --config as well")
        print("This is a design limitation - checkpoints don't store game config")
        return 1
    else:
        try:
            game_config = PokerGameConfig.from_json(args.config)
            print(f"Loaded game configuration: {game_config}")
        except Exception as e:
            print(f"ERROR: Failed to load config from {args.config}: {e}")
            return 1

    # Create solver
    algorithm_kwargs = {}
    if args.algorithm == 'dcfr':
        algorithm_kwargs = {
            'alpha': args.alpha,
            'beta': args.beta,
            'gamma': args.gamma
        }
    elif args.algorithm == 'outcome_mccfr':
        algorithm_kwargs = {'epsilon': args.epsilon}

    try:
        solver = UnifiedPokerSolver(
            game_config=game_config,
            algorithm=args.algorithm,
            **algorithm_kwargs
        )
    except Exception as e:
        print(f"ERROR: Failed to create solver: {e}")
        return 1

    # Create adaptive schedule
    if args.check_exploitability == 'adaptive':
        schedule = AdaptiveSchedule()
    else:
        # Fixed interval
        schedule = AdaptiveSchedule(
            thresholds=[0],
            intervals=[args.check_interval]
        )

    # Run solver
    print("\nStarting solve...")
    try:
        solver.solve(
            max_iterations=args.iterations,
            adaptive_schedule=schedule,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            progress_interval=args.progress_interval,
            skip_initial_exploitability=args.skip_initial_check,
            use_sampled_exploitability=not args.use_full_exploitability,  # Default: True (sampled)
            final_sampled_exploitability=args.final_sampled_exploitability,
            sampled_exploit_confidence=args.sampled_confidence,
            sampled_exploit_ci_width=args.sampled_ci_width,
            sampled_exploit_min_samples=args.sampled_min_samples,
            sampled_exploit_max_samples=args.sampled_max_samples
        )
    except KeyboardInterrupt:
        print("\n\nSolve interrupted by user")
        print("Saving partial results...")
    except Exception as e:
        print(f"\nERROR: Solve failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate base filename
    base_name = f"{args.algorithm}_{game_config.get_short_description()}_{timestamp}"

    # Save metrics
    csv_path = output_dir / f"{base_name}_metrics.csv"
    json_path = output_dir / f"{base_name}_summary.json"
    solver.save_metrics(str(csv_path), str(json_path))

    print(f"\nMetrics saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")

    # Save policy if requested
    if args.save_policy:
        policy_path = output_dir / f"{base_name}_policy.pkl"
        solver.save_policy(str(policy_path))
        print(f"  Policy: {policy_path}")

    # Print summary
    summary = solver.get_metrics_summary()
    print("\n" + "=" * 80)
    print("SOLVE COMPLETE - SUMMARY")
    print("=" * 80)

    # Handle partial results (e.g., after interruption)
    if 'algorithm' in summary:
        print(f"Algorithm: {summary['algorithm']}")
    else:
        print(f"Algorithm: {args.algorithm}")

    if 'game_description' in summary:
        print(f"Game: {summary['game_description']}")
    else:
        print(f"Game: {game_config}")

    if 'total_iterations' in summary:
        print(f"Total iterations: {summary['total_iterations']:,}")
    else:
        print(f"Total iterations: {solver.current_iteration:,}")

    if 'final_exploitability' in summary and summary['final_exploitability'] is not None:
        print(f"Final exploitability: {summary['final_exploitability']:.6f}")
    else:
        print(f"Final exploitability: Not calculated (interrupted)")

    # Show sampled exploitability if available
    if solver.metrics_tracker.sampled_exploitability_result is not None:
        sampled_report = solver.metrics_tracker.get_formatted_sampled_exploit_report()
        print(f"Sampled exploitability: {sampled_report}")

    if 'total_time' in summary:
        print(f"Total time: {summary['total_time']:.1f}s")
    if 'avg_iters_per_sec' in summary:
        print(f"Average speed: {summary['avg_iters_per_sec']:.1f} it/s")
    if 'peak_memory_mb' in summary:
        print(f"Peak memory: {summary['peak_memory_mb']:.1f} MB")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
