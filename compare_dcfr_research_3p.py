#!/usr/bin/env python3
"""
DCFR Research Validation: 3-Player Kuhn Poker

Tests the exact DCFR(α, β, γ) configurations from Brown & Sandholm research
to validate published claims about convergence performance.

Research Configurations Tested:
1. SIMPLE - Baseline external sampling
2. FULL - Reach-probability weighted external sampling
3. True LCFR - DCFR(1, 1, 1) - Original Linear CFR
4. SOTA DCFR - DCFR(1.5, 0, 2) - State-of-the-art (research best)
5. CFR+ Approximation - DCFR(∞, ∞, 2) - Quadratic averaging only
6. DCFR(0, 0, 1) - Research calls this "mismatched and suboptimal"

Research Claims to Validate:
- DCFR(1.5, 0, 2) should converge fastest
- γ=2 (quadratic) should beat γ=1 (linear)
- α=1.5 "scheduled" discount should beat no discount
- DCFR(0, 0, 1) should perform poorly

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    # Full research validation (1M iterations, ~2 hours)
    python compare_dcfr_research_3p.py --iterations 1000000 --check-interval 50000

    # Quick validation (100k iterations, ~12 minutes)
    python compare_dcfr_research_3p.py --iterations 100000 --check-interval 10000

    # With checkpointing for long runs
    python compare_dcfr_research_3p.py \\
        --iterations 1000000 \\
        --check-interval 50000 \\
        --checkpoint-interval 100000

Reference:
    Brown & Sandholm. "Solving Imperfect-Information Games via Discounted Regret Minimization"
    AAAI 2019. https://arxiv.org/abs/1809.04040
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


class ResearchValidationRunner:
    """Validates research-published DCFR configurations."""

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
        """Initialize research validation runner."""
        self.iterations = iterations
        self.check_interval = check_interval
        self.progress_interval = progress_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.force_restart = force_restart

        # Generate checkpoint prefix
        if checkpoint_prefix is None:
            self.checkpoint_prefix = "3p_kuhn_dcfr_research"
        else:
            self.checkpoint_prefix = checkpoint_prefix

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize game (3-player Kuhn poker)
        self.game = pyspiel.load_game("kuhn_poker", {"players": 3})

        # Define research-validated algorithms
        self.algorithms = {
            'SIMPLE': {
                'name': 'SIMPLE',
                'config': 'External Sampling',
                'research': 'Baseline (no weighting)',
                'create_fn': lambda: external_sampling_mccfr.ExternalSamplingSolver(
                    self.game,
                    average_type=external_sampling_mccfr.AverageType.SIMPLE
                )
            },
            'FULL': {
                'name': 'FULL',
                'config': 'External Sampling + Reach-Prob Weighting',
                'research': 'Baseline (reach-prob weighting)',
                'create_fn': lambda: external_sampling_mccfr.ExternalSamplingSolver(
                    self.game,
                    average_type=external_sampling_mccfr.AverageType.FULL
                )
            },
            'TRUE_LCFR': {
                'name': 'True LCFR',
                'config': 'DCFR(1, 1, 1)',
                'research': 'Original Linear CFR (Brown & Sandholm 2019)',
                'create_fn': lambda: LinearExternalSamplingSolver(
                    self.game, gamma=1.0, alpha=1.0, beta=1.0
                )
            },
            'SOTA_DCFR': {
                'name': 'SOTA DCFR',
                'config': 'DCFR(1.5, 0, 2)',
                'research': 'State-of-the-art (Brown & Sandholm 2019 - BEST)',
                'create_fn': lambda: LinearExternalSamplingSolver(
                    self.game, gamma=2.0, alpha=1.5, beta=0.0
                )
            },
            'CFR_PLUS_APPROX': {
                'name': 'CFR+ Approx',
                'config': 'DCFR(∞, ∞, 2)',
                'research': 'Quadratic averaging, no regret discounting',
                'create_fn': lambda: LinearExternalSamplingSolver(
                    self.game, gamma=2.0  # α=None, β=None
                )
            },
            'DCFR_0_0_1': {
                'name': 'DCFR(0,0,1)',
                'config': 'DCFR(0, 0, 1)',
                'research': 'Research calls "mismatched and suboptimal"',
                'create_fn': lambda: LinearExternalSamplingSolver(
                    self.game, gamma=1.0, alpha=0.0, beta=0.0
                )
            }
        }

        # Initialize solvers
        self.solvers = {}
        self.start_iterations = {}

        print("\n" + "="*80)
        print("INITIALIZING DCFR RESEARCH VALIDATION")
        print("="*80)

        for algo_key, algo_info in self.algorithms.items():
            prefix = f"{self.checkpoint_prefix}_{algo_key}"

            # Check for existing checkpoint
            if not force_restart and checkpoint_interval:
                latest_checkpoint = self._find_latest_checkpoint(prefix)
                if latest_checkpoint:
                    print(f"✓ {algo_info['name']:15} - Resuming from {latest_checkpoint}")
                    solver_data = self._load_checkpoint(latest_checkpoint)
                    self.solvers[algo_key] = solver_data['solver']
                    self.start_iterations[algo_key] = solver_data['iteration']
                else:
                    print(f"  {algo_info['name']:15} - Starting from scratch")
                    self.solvers[algo_key] = algo_info['create_fn']()
                    self.start_iterations[algo_key] = 0
            else:
                if force_restart:
                    print(f"⟳ {algo_info['name']:15} - Force restart")
                self.solvers[algo_key] = algo_info['create_fn']()
                self.start_iterations[algo_key] = 0

        # CSV output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = Path(self.output_dir) / f"dcfr_research_validation_{timestamp}.csv"

        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'algorithm', 'config', 'nash_conv', 'wall_time_sec', 'iterations_per_sec'])

        print(f"\nResults: {self.csv_path}\n")

    def _find_latest_checkpoint(self, prefix: str):
        """Find latest checkpoint file."""
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
            return {'solver': data.get('solver', data), 'iteration': data.get('current_iteration', 0)}
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
        """Run the research validation."""
        print("="*80)
        print("DCFR RESEARCH VALIDATION: 3-PLAYER KUHN POKER")
        print("="*80)
        print(f"Total iterations: {self.iterations:,}")
        print(f"Check interval: {self.check_interval:,}")
        print(f"Progress interval: {self.progress_interval:,}")
        if self.checkpoint_interval:
            print(f"Checkpoint interval: {self.checkpoint_interval:,}")
        else:
            print("Checkpointing: DISABLED")

        print("\nResearch Configurations:")
        print("-" * 80)
        for algo_key, algo_info in self.algorithms.items():
            print(f"{algo_info['name']:15} | {algo_info['config']:25} | {algo_info['research']}")
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
                print(f"Iteration: {iteration:,}/{self.iterations:,} | "
                      f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
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
                            writer.writerow([iteration, algo_info['name'], algo_info['config'],
                                           nash_conv, elapsed, rate])

                        # Calculate improvement vs previous
                        improvement = ""
                        if len(metrics[algo_key]['nash_conv']) > 1:
                            prev = metrics[algo_key]['nash_conv'][-2]
                            change = ((nash_conv - prev) / prev) * 100 if prev > 0 else 0
                            improvement = f" ({change:+.1f}%)"

                        results.append({
                            'key': algo_key,
                            'name': algo_info['name'],
                            'config': algo_info['config'],
                            'nash_conv': nash_conv,
                            'improvement': improvement
                        })

                        print(f"{algo_info['name']:15} | {algo_info['config']:25} | "
                              f"Nash: {nash_conv:.6f}{improvement:12}")

                # Comparative analysis
                if iteration >= max_start and results:
                    print(f"{'─'*80}")

                    # Find best and worst
                    best = min(results, key=lambda r: r['nash_conv'])
                    worst = max(results, key=lambda r: r['nash_conv'])

                    print(f"★ Best:  {best['name']:15} ({best['config']:25}) - Nash: {best['nash_conv']:.6f}")
                    print(f"✗ Worst: {worst['name']:15} ({worst['config']:25}) - Nash: {worst['nash_conv']:.6f}")

                    # Compare to research baseline (SIMPLE)
                    simple_nc = next((r['nash_conv'] for r in results if r['key'] == 'SIMPLE'), None)
                    if simple_nc and simple_nc > 0:
                        print(f"\nImprovement vs SIMPLE baseline:")
                        for result in results:
                            if result['key'] != 'SIMPLE':
                                improvement_pct = ((simple_nc - result['nash_conv']) / simple_nc) * 100
                                symbol = "✓" if improvement_pct > 0 else "✗"
                                print(f"  {symbol} {result['name']:15}: {improvement_pct:+6.1f}%")

                    # Validate research claims
                    print(f"\nResearch Claims Validation:")

                    # Claim 1: SOTA DCFR should be best
                    sota_nc = next((r['nash_conv'] for r in results if r['key'] == 'SOTA_DCFR'), None)
                    if sota_nc == best['nash_conv']:
                        print(f"  ✓ SOTA DCFR(1.5,0,2) is best performer (as claimed)")
                    else:
                        print(f"  ✗ SOTA DCFR(1.5,0,2) is NOT best (research claim FAILED)")

                    # Claim 2: DCFR(0,0,1) should be suboptimal
                    dcfr_001_nc = next((r['nash_conv'] for r in results if r['key'] == 'DCFR_0_0_1'), None)
                    if dcfr_001_nc and dcfr_001_nc == worst['nash_conv']:
                        print(f"  ✓ DCFR(0,0,1) is worst performer (as claimed)")
                    elif dcfr_001_nc and dcfr_001_nc > sota_nc:
                        print(f"  ≈ DCFR(0,0,1) is suboptimal but not worst")
                    else:
                        print(f"  ✗ DCFR(0,0,1) performs better than expected (research claim FAILED)")

                    # Claim 3: Gamma=2 should beat Gamma=1
                    cfr_plus_nc = next((r['nash_conv'] for r in results if r['key'] == 'CFR_PLUS_APPROX'), None)
                    true_lcfr_nc = next((r['nash_conv'] for r in results if r['key'] == 'TRUE_LCFR'), None)
                    if cfr_plus_nc and true_lcfr_nc and cfr_plus_nc < true_lcfr_nc:
                        improvement = ((true_lcfr_nc - cfr_plus_nc) / true_lcfr_nc) * 100
                        print(f"  ✓ γ=2 beats γ=1 by {improvement:.1f}% (as claimed)")
                    else:
                        print(f"  ✗ γ=2 does NOT beat γ=1 (research claim FAILED)")

            # Save checkpoints
            if self.checkpoint_interval and iteration > 0 and iteration % self.checkpoint_interval == 0:
                print(f"\nSaving checkpoints at iteration {iteration:,}...")
                for algo_key in self.algorithms.keys():
                    if iteration >= self.start_iterations[algo_key]:
                        checkpoint_path = self._save_checkpoint(algo_key, iteration)

        # Final summary
        print("\n" + "="*80)
        print("RESEARCH VALIDATION COMPLETE")
        print("="*80)
        print(f"Total time: {time.time() - start_time:.1f}s ({(time.time() - start_time)/60:.1f}m)")

        print(f"\nFinal Nash Convergence:")
        print("-" * 80)
        final_results = []
        for algo_key, algo_info in self.algorithms.items():
            if metrics[algo_key]['nash_conv']:
                final_nc = metrics[algo_key]['nash_conv'][-1]
                final_results.append({
                    'key': algo_key,
                    'name': algo_info['name'],
                    'config': algo_info['config'],
                    'nash_conv': final_nc
                })
                print(f"{algo_info['name']:15} | {algo_info['config']:25} | Nash: {final_nc:.6f}")

        # Final rankings
        if final_results:
            print("\n" + "="*80)
            print("FINAL RANKINGS (Best to Worst)")
            print("="*80)
            ranked = sorted(final_results, key=lambda r: r['nash_conv'])
            for rank, result in enumerate(ranked, 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                print(f"{medal:3} {result['name']:15} | {result['config']:25} | {result['nash_conv']:.6f}")

            # Final research validation
            print("\n" + "="*80)
            print("FINAL RESEARCH CLAIMS VALIDATION")
            print("="*80)

            best_final = ranked[0]
            worst_final = ranked[-1]

            if best_final['key'] == 'SOTA_DCFR':
                print(f"✓ SOTA DCFR(1.5,0,2) achieved best convergence")
            else:
                print(f"✗ SOTA DCFR(1.5,0,2) did NOT achieve best convergence")
                print(f"  Winner was: {best_final['name']} ({best_final['config']})")

            if worst_final['key'] == 'DCFR_0_0_1':
                print(f"✓ DCFR(0,0,1) had worst convergence (as claimed)")
            else:
                print(f"≈ DCFR(0,0,1) was not worst, but was suboptimal")

            # Improvement percentages vs SIMPLE
            simple_final = next((r['nash_conv'] for r in final_results if r['key'] == 'SIMPLE'), None)
            if simple_final and simple_final > 0:
                print(f"\nFinal Improvement vs SIMPLE Baseline:")
                for result in final_results:
                    if result['key'] != 'SIMPLE':
                        improvement = ((simple_final - result['nash_conv']) / simple_final) * 100
                        print(f"  {result['name']:15}: {improvement:+6.1f}%")

        print(f"\nDetailed results: {self.csv_path}")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate DCFR research configurations on 3-player Kuhn poker',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--iterations', type=int, default=1000000,
                       help='Total iterations (default: 1,000,000)')
    parser.add_argument('--check-interval', type=int, default=50000,
                       help='Check exploitability every N iterations (default: 50,000)')
    parser.add_argument('--progress-interval', type=int, default=2000,
                       help='Show progress every N iterations (default: 2,000)')
    parser.add_argument('--checkpoint-interval', type=int, default=None,
                       help='Save checkpoint every N iterations (default: None)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--checkpoint-prefix', type=str, default=None,
                       help='Prefix for checkpoint files')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output CSV')
    parser.add_argument('--force-restart', action='store_true',
                       help='Ignore existing checkpoints')

    args = parser.parse_args()

    try:
        runner = ResearchValidationRunner(
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
        print("\n\nValidation interrupted by user")
        if args.checkpoint_interval:
            print("Checkpoints saved - rerun to resume")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
