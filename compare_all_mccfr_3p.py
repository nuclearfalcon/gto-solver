#!/usr/bin/env python3
"""
Compare All MCCFR Variants on 3-Player Kuhn Poker

Tests SIMPLE vs FULL vs LCFR-ES variants to determine optimal algorithm for multiplayer.

Algorithms tested:
- SIMPLE: Standard external sampling (fast, but suboptimal for 3+ players)
- FULL: Reach-probability weighted averaging (better for 3+ players)
- LCFR-ES γ=1.0: Linear iteration weighting
- LCFR-ES γ=1.5: Moderate iteration weighting
- LCFR-ES γ=2.0: Quadratic iteration weighting

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python compare_all_mccfr_3p.py --iterations 1000000 --check-interval 50000

    # Shorter test
    python compare_all_mccfr_3p.py --iterations 100000 --check-interval 10000

    # With checkpointing
    python compare_all_mccfr_3p.py \\
        --iterations 1000000 \\
        --check-interval 50000 \\
        --checkpoint-interval 100000
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import csv

import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr, exploitability

from linear_external_mccfr import LinearExternalSamplingSolver


class MultiAlgorithmComparison:
    """Runs comprehensive comparison of all MCCFR variants."""

    def __init__(
        self,
        iterations: int = 1000000,
        check_interval: int = 50000,
        progress_interval: int = 2000,
        checkpoint_interval: int = None,
        checkpoint_dir: str = "checkpoints",
        checkpoint_prefix: str = None,
        output_dir: str = "results",
        force_restart: bool = False
    ):
        """
        Initialize multi-algorithm comparison.

        Args:
            iterations: Total iterations to run
            check_interval: Check exploitability every N iterations
            progress_interval: Show progress every N iterations
            checkpoint_interval: Save checkpoint every N iterations (None=disable)
            checkpoint_dir: Directory for checkpoints
            checkpoint_prefix: Prefix for checkpoint files
            output_dir: Directory for output CSV
            force_restart: Ignore existing checkpoints
        """
        self.iterations = iterations
        self.check_interval = check_interval
        self.progress_interval = progress_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.force_restart = force_restart

        # Generate checkpoint prefix
        if checkpoint_prefix is None:
            self.checkpoint_prefix = "3p_kuhn_full_comparison"
        else:
            self.checkpoint_prefix = checkpoint_prefix

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize game (3-player Kuhn poker)
        self.game = pyspiel.load_game("kuhn_poker", {"players": 3})

        # Define algorithms to test
        self.algorithms = {
            'SIMPLE': {
                'name': 'SIMPLE',
                'description': 'Standard external sampling',
                'create_fn': lambda: external_sampling_mccfr.ExternalSamplingSolver(
                    self.game,
                    average_type=external_sampling_mccfr.AverageType.SIMPLE
                )
            },
            'FULL': {
                'name': 'FULL',
                'description': 'Reach-probability weighted',
                'create_fn': lambda: external_sampling_mccfr.ExternalSamplingSolver(
                    self.game,
                    average_type=external_sampling_mccfr.AverageType.FULL
                )
            },
            'LCFR-ES_g1.0': {
                'name': 'LCFR-ES γ=1.0',
                'description': 'Linear iteration weighting',
                'create_fn': lambda: LinearExternalSamplingSolver(self.game, gamma=1.0)
            },
            'LCFR-ES_g1.5': {
                'name': 'LCFR-ES γ=1.5',
                'description': 'Moderate iteration weighting',
                'create_fn': lambda: LinearExternalSamplingSolver(self.game, gamma=1.5)
            },
            'LCFR-ES_g2.0': {
                'name': 'LCFR-ES γ=2.0',
                'description': 'Quadratic iteration weighting',
                'create_fn': lambda: LinearExternalSamplingSolver(self.game, gamma=2.0)
            }
        }

        # Initialize solvers
        self.solvers = {}
        self.start_iterations = {}

        for algo_key, algo_info in self.algorithms.items():
            prefix = f"{self.checkpoint_prefix}_{algo_key}"

            # Check for existing checkpoint
            if not force_restart and checkpoint_interval:
                latest_checkpoint = self._find_latest_checkpoint(prefix)
                if latest_checkpoint:
                    print(f"Found checkpoint for {algo_info['name']}: {latest_checkpoint}")
                    solver_data = self._load_checkpoint(latest_checkpoint)
                    self.solvers[algo_key] = solver_data['solver']
                    self.start_iterations[algo_key] = solver_data['iteration']
                else:
                    print(f"No checkpoint for {algo_info['name']}, starting from scratch")
                    self.solvers[algo_key] = algo_info['create_fn']()
                    self.start_iterations[algo_key] = 0
            else:
                if force_restart:
                    print(f"Force restart: {algo_info['name']} from scratch")
                self.solvers[algo_key] = algo_info['create_fn']()
                self.start_iterations[algo_key] = 0

        # CSV output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = Path(self.output_dir) / f"3p_kuhn_full_comparison_{timestamp}.csv"

        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'algorithm', 'nash_conv', 'wall_time_sec', 'iterations_per_sec'])

        print(f"\nResults will be saved to: {self.csv_path}\n")

    def _find_latest_checkpoint(self, prefix: str):
        """Find latest checkpoint file for given prefix."""
        import glob
        import re

        pattern = f"{self.checkpoint_dir}/{prefix}_iter_*.pkl"
        files = glob.glob(pattern)

        if not files:
            return None

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

        if isinstance(data, dict):
            return {
                'solver': data.get('solver', data),
                'iteration': data.get('current_iteration', 0)
            }
        else:
            match = re.search(r'iter_(\d+)', filepath)
            iteration = int(match.group(1)) if match else 0
            return {'solver': data, 'iteration': iteration}

    def _save_checkpoint(self, algo_key: str, iteration: int):
        """Save checkpoint for solver."""
        import pickle

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        prefix = f"{self.checkpoint_prefix}_{algo_key}"
        filepath = f"{self.checkpoint_dir}/{prefix}_iter_{iteration}.pkl"

        checkpoint_data = {
            'solver': self.solvers[algo_key],
            'current_iteration': iteration,
            'algorithm': algo_key
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f, pickle.HIGHEST_PROTOCOL)

        return filepath

    def run(self):
        """Run the comparison."""
        print("="*80)
        print("3-PLAYER KUHN POKER: COMPREHENSIVE MCCFR COMPARISON")
        print("="*80)
        print(f"Total iterations: {self.iterations:,}")
        print(f"Check interval: {self.check_interval:,}")
        print(f"Progress interval: {self.progress_interval:,}")
        if self.checkpoint_interval:
            print(f"Checkpoint interval: {self.checkpoint_interval:,}")
        else:
            print("Checkpointing: DISABLED")
        print("\nAlgorithms tested:")
        for algo_key, algo_info in self.algorithms.items():
            print(f"  {algo_info['name']:20} - {algo_info['description']}")
        print("="*80 + "\n")

        # Track metrics
        metrics = {algo_key: {'nash_conv': [], 'times': [], 'iterations': []}
                   for algo_key in self.algorithms.keys()}

        start_time = time.time()
        max_start = max(self.start_iterations.values())

        # Main iteration loop
        for iteration in range(max_start, self.iterations + 1):
            # Run one iteration for each solver
            for algo_key in self.algorithms.keys():
                if iteration >= self.start_iterations[algo_key]:
                    self.solvers[algo_key].iteration()

            # Show frequent progress updates
            if iteration > 0 and (iteration == 1 or iteration % self.progress_interval == 0):
                elapsed = time.time() - start_time
                rate = iteration / elapsed if elapsed > 0 else 0
                eta = (self.iterations - iteration) / rate if rate > 0 else 0
                print(f"\r{iteration:,}/{self.iterations:,} iterations | "
                      f"Rate: {rate:6.0f} it/s | ETA: {eta/60:5.1f}m | Elapsed: {elapsed:.1f}s",
                      end='', flush=True)

            # Check exploitability at intervals
            if iteration > 0 and iteration % self.check_interval == 0:
                print()  # New line after progress
                elapsed = time.time() - start_time

                print(f"\n{'─'*80}")
                print(f"Iteration: {iteration:,}/{self.iterations:,} | Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
                print(f"{'─'*80}")

                # Calculate exploitability for each algorithm
                results = []
                for algo_key, algo_info in self.algorithms.items():
                    if iteration >= self.start_iterations[algo_key]:
                        policy = self.solvers[algo_key].average_policy()

                        # Use C++ nash_conv for speed
                        try:
                            nash_conv = pyspiel.nash_conv(self.game, policy)
                        except:
                            nash_conv = exploitability.nash_conv(self.game, policy, return_only_nash_conv=True)

                        # Calculate rate
                        iters_since_start = iteration - self.start_iterations[algo_key]
                        rate = iters_since_start / elapsed if elapsed > 0 else 0

                        # Store metrics
                        metrics[algo_key]['nash_conv'].append(nash_conv)
                        metrics[algo_key]['times'].append(elapsed)
                        metrics[algo_key]['iterations'].append(iteration)

                        # Write to CSV
                        with open(self.csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([iteration, algo_info['name'], nash_conv, elapsed, rate])

                        # Calculate improvement vs previous
                        improvement = ""
                        if len(metrics[algo_key]['nash_conv']) > 1:
                            prev = metrics[algo_key]['nash_conv'][-2]
                            change = ((nash_conv - prev) / prev) * 100 if prev > 0 else 0
                            improvement = f" ({change:+.1f}%)"

                        results.append({
                            'key': algo_key,
                            'name': algo_info['name'],
                            'nash_conv': nash_conv,
                            'improvement': improvement,
                            'rate': rate
                        })

                        print(f"{algo_info['name']:20} | Nash: {nash_conv:.6f}{improvement:12} | Rate: {rate:6.0f} it/s")

                # Comparative analysis (vs SIMPLE baseline)
                if iteration >= max_start and results:
                    print(f"{'─'*80}")
                    simple_nc = next((r['nash_conv'] for r in results if r['key'] == 'SIMPLE'), None)

                    if simple_nc and simple_nc > 0:
                        print("Improvement vs SIMPLE:")
                        for result in results:
                            if result['key'] != 'SIMPLE':
                                improvement_pct = ((simple_nc - result['nash_conv']) / simple_nc) * 100
                                symbol = "✓" if improvement_pct > 0 else "✗"
                                print(f"  {symbol} {result['name']:20} is {improvement_pct:+6.1f}% vs SIMPLE")

                    # Find best performer
                    best = min(results, key=lambda r: r['nash_conv'])
                    print(f"\n★ Best performer: {best['name']} (Nash Conv: {best['nash_conv']:.6f})")

            # Save checkpoints
            if self.checkpoint_interval and iteration > 0 and iteration % self.checkpoint_interval == 0:
                print(f"\nSaving checkpoints at iteration {iteration:,}...")
                for algo_key in self.algorithms.keys():
                    if iteration >= self.start_iterations[algo_key]:
                        checkpoint_path = self._save_checkpoint(algo_key, iteration)
                        print(f"  Saved: {checkpoint_path}")

        # Final summary
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print(f"Total time: {time.time() - start_time:.1f}s ({(time.time() - start_time)/60:.1f}m)")

        print(f"\nFinal Nash Convergence:")
        final_results = []
        for algo_key, algo_info in self.algorithms.items():
            if metrics[algo_key]['nash_conv']:
                final_nc = metrics[algo_key]['nash_conv'][-1]
                final_results.append({
                    'key': algo_key,
                    'name': algo_info['name'],
                    'nash_conv': final_nc
                })
                print(f"  {algo_info['name']:20}: {final_nc:.6f}")

        # Rankings
        if final_results:
            print("\nRankings (best to worst):")
            ranked = sorted(final_results, key=lambda r: r['nash_conv'])
            for rank, result in enumerate(ranked, 1):
                print(f"  {rank}. {result['name']:20} - {result['nash_conv']:.6f}")

            # Improvement vs SIMPLE
            simple_final = next((r['nash_conv'] for r in final_results if r['key'] == 'SIMPLE'), None)
            if simple_final and simple_final > 0:
                print("\nFinal improvement vs SIMPLE:")
                for result in final_results:
                    if result['key'] != 'SIMPLE':
                        improvement = ((simple_final - result['nash_conv']) / simple_final) * 100
                        print(f"  {result['name']:20}: {improvement:+6.1f}%")

        print(f"\nDetailed results saved to: {self.csv_path}")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare all MCCFR variants on 3-player Kuhn poker',
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
        '--progress-interval',
        type=int,
        default=2000,
        help='Show progress every N iterations (default: 2,000)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=None,
        help='Save checkpoint every N iterations (default: None = disabled)'
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
        help='Prefix for checkpoint files (default: 3p_kuhn_full_comparison)'
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

    try:
        runner = MultiAlgorithmComparison(
            iterations=args.iterations,
            check_interval=args.check_interval,
            progress_interval=args.progress_interval,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_prefix=args.checkpoint_prefix,
            output_dir=args.output_dir,
            force_restart=args.force_restart
        )
        runner.run()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
        if args.checkpoint_interval:
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
