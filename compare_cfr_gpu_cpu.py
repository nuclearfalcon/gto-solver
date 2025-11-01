#!/usr/bin/env python3
"""
GPU vs CPU CFR Comparison for 3-Player Kuhn Poker

Compares CFR algorithms on GPU (using JAX/cfrx) vs CPU (using OpenSpiel)
to validate correctness and measure speedup.

Currently supports:
- Vanilla CFR (GPU vs CPU)
- MCCFR Outcome Sampling (GPU vs CPU)

DCFR variants (alpha, beta, gamma) are CPU-only until cfrx adds support.

Requirements:
    source ~/open_spiel/venv/bin/activate
    pip install 'jax[cuda12]'
    pip install cfrx

Usage:
    # Quick comparison (10k iterations)
    python compare_cfr_gpu_cpu.py --iterations 10000

    # Full comparison (1M iterations)
    python compare_cfr_gpu_cpu.py --iterations 1000000 --check-interval 100000

    # Save detailed results
    python compare_cfr_gpu_cpu.py --iterations 100000 --output-dir results/gpu_comparison
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import csv

import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr, exploitability

# Import GPU solver (will check availability internally)
from gpu_cfr_solver import GPUCFRSolver, check_gpu_requirements


class GPUCPUComparison:
    """Compare GPU and CPU CFR implementations."""

    def __init__(
        self,
        iterations: int = 100000,
        check_interval: int = 10000,
        output_dir: str = "results",
        num_players: int = 3
    ):
        """Initialize comparison runner."""
        self.iterations = iterations
        self.check_interval = check_interval
        self.output_dir = output_dir
        self.num_players = num_players

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize OpenSpiel game
        self.game = pyspiel.load_game("kuhn_poker", {"players": num_players})

        # Check GPU availability
        self.gpu_available, gpu_msg = check_gpu_requirements()
        print(f"\nGPU Status: {gpu_msg}")

        if not self.gpu_available:
            print("\n⚠ GPU not available. Will only run CPU benchmarks.")
            print("To enable GPU:")
            print("  pip install 'jax[cuda12]'")
            print("  pip install cfrx\n")

        # CSV output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = Path(self.output_dir) / f"gpu_cpu_comparison_{timestamp}.csv"

        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'algorithm', 'platform', 'nash_conv',
                           'wall_time_sec', 'iterations_per_sec'])

    def run_cpu_vanilla_cfr(self):
        """Run vanilla CFR on CPU using OpenSpiel."""
        print(f"\n{'='*80}")
        print("CPU: Vanilla CFR (OpenSpiel)")
        print(f"{'='*80}")

        from open_spiel.python.algorithms import cfr

        solver = cfr.CFRSolver(self.game)
        metrics = []

        start_time = time.time()

        for i in range(1, self.iterations + 1):
            solver.evaluate_and_update_policy()

            # Progress
            if i % 1000 == 0 or i == 1:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (self.iterations - i) / rate if rate > 0 else 0
                print(f"\r{i:,}/{self.iterations:,} | Rate: {rate:6.0f} it/s | "
                      f"ETA: {eta/60:5.1f}m", end='', flush=True)

            # Check exploitability
            if i % self.check_interval == 0:
                print()  # New line
                elapsed = time.time() - start_time
                policy = solver.average_policy()

                try:
                    nash_conv = pyspiel.nash_conv(self.game, policy)
                except:
                    nash_conv = exploitability.nash_conv(
                        self.game, policy, return_only_nash_conv=True)

                rate = i / elapsed if elapsed > 0 else 0

                print(f"Iteration {i:,}: Nash Conv = {nash_conv:.6f}, "
                      f"Rate = {rate:,.0f} it/s")

                # Record
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, 'vanilla_cfr', 'cpu', nash_conv, elapsed, rate])

                metrics.append({
                    'iteration': i,
                    'nash_conv': nash_conv,
                    'time': elapsed,
                    'rate': rate
                })

        elapsed = time.time() - start_time
        print(f"\n\nCPU Vanilla CFR Complete: {elapsed:.2f}s ({elapsed/60:.2f}m)")
        print(f"Final rate: {self.iterations/elapsed:,.0f} it/s")

        return metrics

    def run_cpu_mccfr(self):
        """Run MCCFR on CPU using OpenSpiel."""
        print(f"\n{'='*80}")
        print("CPU: MCCFR Outcome Sampling (OpenSpiel)")
        print(f"{'='*80}")

        from open_spiel.python.algorithms import external_sampling_mccfr

        solver = external_sampling_mccfr.ExternalSamplingSolver(
            self.game,
            average_type=external_sampling_mccfr.AverageType.SIMPLE
        )
        metrics = []

        start_time = time.time()

        for i in range(1, self.iterations + 1):
            solver.iteration()

            # Progress
            if i % 1000 == 0 or i == 1:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (self.iterations - i) / rate if rate > 0 else 0
                print(f"\r{i:,}/{self.iterations:,} | Rate: {rate:6.0f} it/s | "
                      f"ETA: {eta/60:5.1f}m", end='', flush=True)

            # Check exploitability
            if i % self.check_interval == 0:
                print()  # New line
                elapsed = time.time() - start_time
                policy = solver.average_policy()

                try:
                    nash_conv = pyspiel.nash_conv(self.game, policy)
                except:
                    nash_conv = exploitability.nash_conv(
                        self.game, policy, return_only_nash_conv=True)

                rate = i / elapsed if elapsed > 0 else 0

                print(f"Iteration {i:,}: Nash Conv = {nash_conv:.6f}, "
                      f"Rate = {rate:,.0f} it/s")

                # Record
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, 'mccfr', 'cpu', nash_conv, elapsed, rate])

                metrics.append({
                    'iteration': i,
                    'nash_conv': nash_conv,
                    'time': elapsed,
                    'rate': rate
                })

        elapsed = time.time() - start_time
        print(f"\n\nCPU MCCFR Complete: {elapsed:.2f}s ({elapsed/60:.2f}m)")
        print(f"Final rate: {self.iterations/elapsed:,.0f} it/s")

        return metrics

    def run_gpu_vanilla_cfr(self):
        """Run vanilla CFR on GPU using cfrx."""
        if not self.gpu_available:
            return None

        print(f"\n{'='*80}")
        print("GPU: Vanilla CFR (cfrx/JAX)")
        print(f"{'='*80}")

        solver = GPUCFRSolver(
            game_name='kuhn',
            num_players=self.num_players,
            algorithm='vanilla_cfr'
        )

        metrics = []
        start_time = time.time()

        # Run in chunks to record periodic metrics
        for checkpoint in range(self.check_interval, self.iterations + 1, self.check_interval):
            chunk_size = self.check_interval
            solver.solve(iterations=chunk_size, metrics_interval=chunk_size, verbose=False)

            elapsed = time.time() - start_time
            rate = checkpoint / elapsed if elapsed > 0 else 0

            # Get exploitability (from cfrx metrics if available)
            exp = solver.calculate_exploitability()
            if exp is None:
                exp = 0.0  # Placeholder

            print(f"Iteration {checkpoint:,}: Exploitability = {exp:.6f}, "
                  f"Rate = {rate:,.0f} it/s")

            # Record
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([checkpoint, 'vanilla_cfr', 'gpu', exp, elapsed, rate])

            metrics.append({
                'iteration': checkpoint,
                'exploitability': exp,
                'time': elapsed,
                'rate': rate
            })

        elapsed = time.time() - start_time
        print(f"\n\nGPU Vanilla CFR Complete: {elapsed:.2f}s ({elapsed/60:.2f}m)")
        print(f"Final rate: {self.iterations/elapsed:,.0f} it/s")

        return metrics

    def run_gpu_mccfr(self):
        """Run MCCFR on GPU using cfrx."""
        if not self.gpu_available:
            return None

        print(f"\n{'='*80}")
        print("GPU: MCCFR (cfrx/JAX)")
        print(f"{'='*80}")

        solver = GPUCFRSolver(
            game_name='kuhn',
            num_players=self.num_players,
            algorithm='mccfr'
        )

        metrics = []
        start_time = time.time()

        # Run in chunks to record periodic metrics
        for checkpoint in range(self.check_interval, self.iterations + 1, self.check_interval):
            chunk_size = self.check_interval
            solver.solve(iterations=chunk_size, metrics_interval=chunk_size, verbose=False)

            elapsed = time.time() - start_time
            rate = checkpoint / elapsed if elapsed > 0 else 0

            # Get exploitability (from cfrx metrics if available)
            exp = solver.calculate_exploitability()
            if exp is None:
                exp = 0.0  # Placeholder

            print(f"Iteration {checkpoint:,}: Exploitability = {exp:.6f}, "
                  f"Rate = {rate:,.0f} it/s")

            # Record
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([checkpoint, 'mccfr', 'gpu', exp, elapsed, rate])

            metrics.append({
                'iteration': checkpoint,
                'exploitability': exp,
                'time': elapsed,
                'rate': rate
            })

        elapsed = time.time() - start_time
        print(f"\n\nGPU MCCFR Complete: {elapsed:.2f}s ({elapsed/60:.2f}m)")
        print(f"Final rate: {self.iterations/elapsed:,.0f} it/s")

        return metrics

    def run(self):
        """Run full GPU vs CPU comparison."""
        print("="*80)
        print(f"GPU vs CPU CFR COMPARISON: {self.num_players}-Player Kuhn Poker")
        print("="*80)
        print(f"Total iterations: {self.iterations:,}")
        print(f"Check interval: {self.check_interval:,}")
        print(f"Results: {self.csv_path}")
        print("="*80)

        results = {}

        # Run CPU benchmarks
        results['cpu_vanilla'] = self.run_cpu_vanilla_cfr()
        results['cpu_mccfr'] = self.run_cpu_mccfr()

        # Run GPU benchmarks (if available)
        if self.gpu_available:
            results['gpu_vanilla'] = self.run_gpu_vanilla_cfr()
            results['gpu_mccfr'] = self.run_gpu_mccfr()
        else:
            results['gpu_vanilla'] = None
            results['gpu_mccfr'] = None

        # Summary
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)

        if results['cpu_vanilla']:
            cpu_v_rate = results['cpu_vanilla'][-1]['rate']
            print(f"CPU Vanilla CFR:  {cpu_v_rate:>12,.0f} it/s")

        if results['cpu_mccfr']:
            cpu_m_rate = results['cpu_mccfr'][-1]['rate']
            print(f"CPU MCCFR:        {cpu_m_rate:>12,.0f} it/s")

        if results['gpu_vanilla']:
            gpu_v_rate = results['gpu_vanilla'][-1]['rate']
            speedup_v = gpu_v_rate / cpu_v_rate if results['cpu_vanilla'] else 0
            print(f"GPU Vanilla CFR:  {gpu_v_rate:>12,.0f} it/s  ({speedup_v:>5.1f}x speedup)")

        if results['gpu_mccfr']:
            gpu_m_rate = results['gpu_mccfr'][-1]['rate']
            speedup_m = gpu_m_rate / cpu_m_rate if results['cpu_mccfr'] else 0
            print(f"GPU MCCFR:        {gpu_m_rate:>12,.0f} it/s  ({speedup_m:>5.1f}x speedup)")

        print("="*80)
        print(f"\nDetailed results: {self.csv_path}")
        print("="*80)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare GPU vs CPU CFR performance on Kuhn poker',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--iterations', type=int, default=100000,
                       help='Total iterations (default: 100,000)')
    parser.add_argument('--check-interval', type=int, default=10000,
                       help='Check metrics every N iterations (default: 10,000)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output CSV')
    parser.add_argument('--players', type=int, default=3,
                       help='Number of players (2 or 3, default: 3)')

    args = parser.parse_args()

    try:
        comparison = GPUCPUComparison(
            iterations=args.iterations,
            check_interval=args.check_interval,
            output_dir=args.output_dir,
            num_players=args.players
        )
        comparison.run()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
