#!/usr/bin/env python3
"""
Compare MCCFR Averaging Types on 3-Player Kuhn Poker

Tests SIMPLE vs FULL averaging to determine which converges faster for multiplayer games.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python compare_mccfr_averaging_3p.py --iterations 1000000 --check-interval 50000

    # With custom checkpoint prefix
    python compare_mccfr_averaging_3p.py \\
        --iterations 1000000 \\
        --check-interval 50000 \\
        --checkpoint-prefix "kuhn_3p_comparison"

    # Force restart
    python compare_mccfr_averaging_3p.py \\
        --iterations 1000000 \\
        --check-interval 50000 \\
        --force-restart
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import csv

import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr, exploitability

from poker_solver import UnifiedPokerSolver
from game_config import PokerGameConfig


class ComparisonRunner:
    """Runs side-by-side comparison of MCCFR averaging types."""

    def __init__(
        self,
        config_name: str = "3-player Kuhn Poker",
        iterations: int = 1000000,
        check_interval: int = 50000,
        checkpoint_interval: int = 100000,
        checkpoint_dir: str = "checkpoints",
        checkpoint_prefix: str = None,
        output_dir: str = "results",
        force_restart: bool = False
    ):
        """
        Initialize comparison runner.

        Args:
            config_name: Game configuration name
            iterations: Total iterations to run
            check_interval: Check exploitability every N iterations
            checkpoint_interval: Save checkpoint every N iterations
            checkpoint_dir: Directory for checkpoints
            checkpoint_prefix: Prefix for checkpoint files
            output_dir: Directory for output CSV
            force_restart: Ignore existing checkpoints
        """
        self.config_name = config_name
        self.iterations = iterations
        self.check_interval = check_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.force_restart = force_restart

        # Generate checkpoint prefix if not provided
        if checkpoint_prefix is None:
            self.checkpoint_prefix = "3p_kuhn_comparison"
        else:
            self.checkpoint_prefix = checkpoint_prefix

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize game (3-player Kuhn poker)
        self.game = pyspiel.load_game("kuhn_poker", {"players": 3})

        # Initialize solvers
        self.solvers = {}
        self.start_iterations = {}  # Track starting iteration for each algorithm

        for avg_type in ['SIMPLE', 'FULL']:
            prefix = f"{self.checkpoint_prefix}_{avg_type}"

            # Check for existing checkpoint
            if not force_restart and checkpoint_interval:
                latest_checkpoint = self._find_latest_checkpoint(prefix)
                if latest_checkpoint:
                    print(f"\n{'='*60}")
                    print(f"Found checkpoint for {avg_type}: {latest_checkpoint}")
                    print(f"Loading checkpoint...")
                    print(f"{'='*60}")
                    solver_data = self._load_checkpoint(latest_checkpoint)
                    self.solvers[avg_type] = solver_data['solver']
                    self.start_iterations[avg_type] = solver_data['iteration']
                else:
                    print(f"No checkpoint found for {avg_type}, starting from scratch")
                    if avg_type == 'SIMPLE':
                        self.solvers[avg_type] = external_sampling_mccfr.ExternalSamplingSolver(
                            self.game,
                            average_type=external_sampling_mccfr.AverageType.SIMPLE
                        )
                    else:
                        self.solvers[avg_type] = external_sampling_mccfr.ExternalSamplingSolver(
                            self.game,
                            average_type=external_sampling_mccfr.AverageType.FULL
                        )
                    self.start_iterations[avg_type] = 0
            else:
                if force_restart:
                    print(f"Force restart: Starting {avg_type} from scratch")
                if avg_type == 'SIMPLE':
                    self.solvers[avg_type] = external_sampling_mccfr.ExternalSamplingSolver(
                        self.game,
                        average_type=external_sampling_mccfr.AverageType.SIMPLE
                    )
                else:
                    self.solvers[avg_type] = external_sampling_mccfr.ExternalSamplingSolver(
                        self.game,
                        average_type=external_sampling_mccfr.AverageType.FULL
                    )
                self.start_iterations[avg_type] = 0

        # CSV output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = Path(self.output_dir) / f"3p_kuhn_comparison_{timestamp}.csv"

        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'algorithm', 'nash_conv', 'wall_time_sec', 'iterations_per_sec'])

        print(f"\nResults will be saved to: {self.csv_path}")

    def _find_latest_checkpoint(self, prefix: str):
        """Find latest checkpoint file for given prefix."""
        import glob
        import re

        pattern = f"{self.checkpoint_dir}/{prefix}_iter_*.pkl"
        files = glob.glob(pattern)

        if not files:
            return None

        # Find file with maximum iteration number
        max_iter = -1
        latest = None
        for filepath in files:
            match = re.search(r'iter_(\d+)\.pkl$', filepath)
            if match:
                iteration = int(match.group(1))
                if iteration > max_iter:
                    max_iter = iteration
                    latest = filepath

        return latest

    def _load_checkpoint(self, filepath: str):
        """Load checkpoint from file."""
        import pickle
        import re

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Handle both old and new format
        if isinstance(data, dict):
            return {
                'solver': data.get('solver', data),
                'iteration': data.get('current_iteration', 0)
            }
        else:
            # Old format - extract iteration from filename
            match = re.search(r'iter_(\d+)', filepath)
            iteration = int(match.group(1)) if match else 0
            return {
                'solver': data,
                'iteration': iteration
            }

    def _save_checkpoint(self, avg_type: str, iteration: int):
        """Save checkpoint for solver."""
        import pickle

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        prefix = f"{self.checkpoint_prefix}_{avg_type}"
        filepath = f"{self.checkpoint_dir}/{prefix}_iter_{iteration}.pkl"

        checkpoint_data = {
            'solver': self.solvers[avg_type],
            'current_iteration': iteration,
            'algorithm': f"external_mccfr_{avg_type}"
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f, pickle.HIGHEST_PROTOCOL)

        return filepath

    def run(self):
        """Run the comparison."""
        print("\n" + "="*80)
        print("3-PLAYER KUHN POKER: MCCFR AVERAGING COMPARISON")
        print("="*80)
        print(f"Total iterations: {self.iterations:,}")
        print(f"Check interval: {self.check_interval:,}")
        print(f"Checkpoint interval: {self.checkpoint_interval:,}" if self.checkpoint_interval else "Checkpointing: DISABLED")
        print(f"Algorithms: SIMPLE (default) vs FULL (better for 3+ players)")
        print("="*80)

        # Track metrics for each algorithm
        metrics = {
            'SIMPLE': {'nash_conv': [], 'times': [], 'iterations': []},
            'FULL': {'nash_conv': [], 'times': [], 'iterations': []}
        }

        start_time = time.time()
        last_check = {k: v for k, v in self.start_iterations.items()}

        # Main iteration loop
        for iteration in range(max(self.start_iterations.values()), self.iterations + 1):
            # Run one iteration for each solver
            for avg_type in ['SIMPLE', 'FULL']:
                if iteration >= self.start_iterations[avg_type]:
                    self.solvers[avg_type].iteration()

            # Show frequent progress updates (every 2k iterations)
            if iteration > 0 and (iteration == 1 or iteration % 2000 == 0):
                elapsed = time.time() - start_time
                rate = iteration / elapsed if elapsed > 0 else 0
                eta = (self.iterations - iteration) / rate if rate > 0 else 0
                print(f"\r{iteration:,}/{self.iterations:,} iterations | Rate: {rate:6.0f} it/s | ETA: {eta/60:5.1f}m | Elapsed: {elapsed:.1f}s", end='', flush=True)

            # Check exploitability at intervals
            if iteration > 0 and iteration % self.check_interval == 0:
                # Clear the progress line
                print()  # New line after progress updates
                elapsed = time.time() - start_time

                print(f"\n{'─'*80}")
                print(f"Iteration: {iteration:,}/{self.iterations:,} | Elapsed: {elapsed:.1f}s")
                print(f"{'─'*80}")

                # Calculate exploitability for each algorithm
                for avg_type in ['SIMPLE', 'FULL']:
                    if iteration >= self.start_iterations[avg_type]:
                        policy = self.solvers[avg_type].average_policy()

                        # Use C++ nash_conv for speed
                        try:
                            nash_conv = pyspiel.nash_conv(self.game, policy)
                        except:
                            nash_conv = exploitability.nash_conv(self.game, policy, return_only_nash_conv=True)

                        # Calculate rate
                        iters_since_start = iteration - self.start_iterations[avg_type]
                        rate = iters_since_start / elapsed if elapsed > 0 else 0
                        eta = (self.iterations - iteration) / rate if rate > 0 else 0

                        # Store metrics
                        metrics[avg_type]['nash_conv'].append(nash_conv)
                        metrics[avg_type]['times'].append(elapsed)
                        metrics[avg_type]['iterations'].append(iteration)

                        # Write to CSV
                        with open(self.csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([iteration, avg_type, nash_conv, elapsed, rate])

                        # Display progress
                        improvement = ""
                        if len(metrics[avg_type]['nash_conv']) > 1:
                            prev = metrics[avg_type]['nash_conv'][-2]
                            change = ((nash_conv - prev) / prev) * 100 if prev > 0 else 0
                            improvement = f" ({change:+.1f}%)"

                        print(f"{avg_type:10} | Nash: {nash_conv:.6f}{improvement:12} | "
                              f"Rate: {rate:6.0f} it/s | ETA: {eta/60:5.1f}m")

                # Comparative analysis
                if iteration >= max(self.start_iterations.values()):
                    simple_nc = metrics['SIMPLE']['nash_conv'][-1]
                    full_nc = metrics['FULL']['nash_conv'][-1]

                    if simple_nc > 0:
                        improvement = ((simple_nc - full_nc) / simple_nc) * 100
                        if improvement > 0:
                            print(f"{'─'*80}")
                            print(f"✓ FULL is {improvement:.1f}% better than SIMPLE")
                        else:
                            print(f"{'─'*80}")
                            print(f"✗ SIMPLE is {abs(improvement):.1f}% better than FULL")

            # Save checkpoints
            if self.checkpoint_interval and iteration > 0 and iteration % self.checkpoint_interval == 0:
                for avg_type in ['SIMPLE', 'FULL']:
                    if iteration >= self.start_iterations[avg_type]:
                        checkpoint_path = self._save_checkpoint(avg_type, iteration)
                        print(f"Checkpoint saved: {checkpoint_path}")

        # Final summary
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"\nFinal Nash Convergence:")
        for avg_type in ['SIMPLE', 'FULL']:
            if metrics[avg_type]['nash_conv']:
                final_nc = metrics[avg_type]['nash_conv'][-1]
                print(f"  {avg_type:10}: {final_nc:.6f}")

        if metrics['SIMPLE']['nash_conv'] and metrics['FULL']['nash_conv']:
            simple_final = metrics['SIMPLE']['nash_conv'][-1]
            full_final = metrics['FULL']['nash_conv'][-1]
            improvement = ((simple_final - full_final) / simple_final) * 100
            print(f"\nFULL is {improvement:+.1f}% vs SIMPLE")

        print(f"\nDetailed results saved to: {self.csv_path}")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare MCCFR averaging types on 3-player Kuhn poker',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=1000000,
        help='Total iterations to run (default: 1,000,000)'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=50000,
        help='Check exploitability every N iterations (default: 50,000)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=100000,
        help='Save checkpoint every N iterations (default: 100,000, 0=disable)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory for checkpoints (default: checkpoints/)'
    )
    parser.add_argument(
        '--checkpoint-prefix',
        type=str,
        default=None,
        help='Prefix for checkpoint files (default: 3p_kuhn_comparison)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory for output CSV (default: results/)'
    )
    parser.add_argument(
        '--force-restart',
        action='store_true',
        help='Ignore existing checkpoints and start from scratch'
    )

    args = parser.parse_args()

    # Disable checkpointing if interval is 0
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval > 0 else None

    try:
        runner = ComparisonRunner(
            iterations=args.iterations,
            check_interval=args.check_interval,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_prefix=args.checkpoint_prefix,
            output_dir=args.output_dir,
            force_restart=args.force_restart
        )
        runner.run()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
        print("Checkpoints have been saved - rerun to resume")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
