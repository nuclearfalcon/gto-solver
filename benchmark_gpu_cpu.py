#!/usr/bin/env python3
"""
Simple GPU vs CPU Benchmark for CFR

Quick benchmark to measure speedup from GPU acceleration.
Runs short tests to estimate performance without full training.

Requirements:
    source ~/open_spiel/venv/bin/activate
    pip install 'jax[cuda12]'
    pip install cfrx

Usage:
    # Quick benchmark (10 seconds per test)
    python benchmark_gpu_cpu.py

    # Extended benchmark (60 seconds per test)
    python benchmark_gpu_cpu.py --duration 60

    # Test specific algorithm
    python benchmark_gpu_cpu.py --algorithm mccfr
"""

import argparse
import time
import sys

import pyspiel
from open_spiel.python.algorithms import cfr, external_sampling_mccfr

from gpu_cfr_solver import GPUCFRSolver, check_gpu_requirements


def benchmark_cpu_vanilla_cfr(duration_sec: int = 10, num_players: int = 3):
    """Benchmark vanilla CFR on CPU."""
    print(f"\n{'─'*80}")
    print("CPU: Vanilla CFR (OpenSpiel)")
    print(f"{'─'*80}")

    game = pyspiel.load_game("kuhn_poker", {"players": num_players})
    solver = cfr.CFRSolver(game)

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < duration_sec:
        solver.evaluate_and_update_policy()
        iterations += 1

    elapsed = time.time() - start_time
    rate = iterations / elapsed

    print(f"Iterations: {iterations:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Rate: {rate:,.0f} it/s")

    return rate


def benchmark_cpu_mccfr(duration_sec: int = 10, num_players: int = 3):
    """Benchmark MCCFR on CPU."""
    print(f"\n{'─'*80}")
    print("CPU: MCCFR External Sampling (OpenSpiel)")
    print(f"{'─'*80}")

    game = pyspiel.load_game("kuhn_poker", {"players": num_players})
    solver = external_sampling_mccfr.ExternalSamplingSolver(
        game,
        average_type=external_sampling_mccfr.AverageType.SIMPLE
    )

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < duration_sec:
        solver.iteration()
        iterations += 1

    elapsed = time.time() - start_time
    rate = iterations / elapsed

    print(f"Iterations: {iterations:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Rate: {rate:,.0f} it/s")

    return rate


def benchmark_gpu_vanilla_cfr(duration_sec: int = 10, num_players: int = 3):
    """Benchmark vanilla CFR on GPU."""
    print(f"\n{'─'*80}")
    print("GPU: Vanilla CFR (cfrx/JAX)")
    print(f"{'─'*80}")

    solver = GPUCFRSolver(
        game_name='kuhn',
        num_players=num_players,
        algorithm='vanilla_cfr'
    )

    # Estimate iterations we can do in duration_sec
    # Do a quick warmup to compile JAX functions
    print("Warming up GPU...")
    solver.solve(iterations=100, metrics_interval=100, verbose=False)

    # Now benchmark
    start_time = time.time()
    # Estimate: assume ~10000 it/s, scale by duration
    iterations = max(1000, int(10000 * duration_sec))

    result = solver.solve(iterations=iterations, metrics_interval=iterations, verbose=False)

    rate = result['iterations_per_second']

    print(f"Iterations: {iterations:,}")
    print(f"Time: {result['elapsed_time']:.2f}s")
    print(f"Rate: {rate:,.0f} it/s")

    return rate


def benchmark_gpu_mccfr(duration_sec: int = 10, num_players: int = 3):
    """Benchmark MCCFR on GPU."""
    print(f"\n{'─'*80}")
    print("GPU: MCCFR (cfrx/JAX)")
    print(f"{'─'*80}")

    solver = GPUCFRSolver(
        game_name='kuhn',
        num_players=num_players,
        algorithm='mccfr'
    )

    # Warmup
    print("Warming up GPU...")
    solver.solve(iterations=100, metrics_interval=100, verbose=False)

    # Benchmark
    start_time = time.time()
    iterations = max(1000, int(10000 * duration_sec))

    result = solver.solve(iterations=iterations, metrics_interval=iterations, verbose=False)

    rate = result['iterations_per_second']

    print(f"Iterations: {iterations:,}")
    print(f"Time: {result['elapsed_time']:.2f}s")
    print(f"Rate: {rate:,.0f} it/s")

    return rate


def main():
    """Run benchmarks and display speedup."""
    parser = argparse.ArgumentParser(
        description='Quick GPU vs CPU CFR benchmark'
    )

    parser.add_argument('--duration', type=int, default=10,
                       help='Duration per benchmark in seconds (default: 10)')
    parser.add_argument('--algorithm', type=str, default='both',
                       choices=['vanilla', 'mccfr', 'both'],
                       help='Which algorithm to benchmark (default: both)')
    parser.add_argument('--players', type=int, default=3,
                       help='Number of players (default: 3)')

    args = parser.parse_args()

    print("="*80)
    print(f"GPU vs CPU CFR BENCHMARK")
    print("="*80)
    print(f"Game: {args.players}-player Kuhn Poker")
    print(f"Duration: {args.duration}s per test")
    print("="*80)

    # Check GPU
    gpu_available, gpu_msg = check_gpu_requirements()
    print(f"\nGPU Status: {gpu_msg}")

    if not gpu_available:
        print("\n⚠ GPU not available. Only running CPU benchmarks.")
        print("To enable GPU:")
        print("  pip install 'jax[cuda12]'")
        print("  pip install cfrx\n")

    results = {}

    # Vanilla CFR
    if args.algorithm in ['vanilla', 'both']:
        results['cpu_vanilla'] = benchmark_cpu_vanilla_cfr(args.duration, args.players)

        if gpu_available:
            results['gpu_vanilla'] = benchmark_gpu_vanilla_cfr(args.duration, args.players)
        else:
            results['gpu_vanilla'] = None

    # MCCFR
    if args.algorithm in ['mccfr', 'both']:
        results['cpu_mccfr'] = benchmark_cpu_mccfr(args.duration, args.players)

        if gpu_available:
            results['gpu_mccfr'] = benchmark_gpu_mccfr(args.duration, args.players)
        else:
            results['gpu_mccfr'] = None

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    if 'cpu_vanilla' in results:
        cpu_v = results['cpu_vanilla']
        gpu_v = results['gpu_vanilla']
        print(f"\nVanilla CFR:")
        print(f"  CPU:     {cpu_v:>10,.0f} it/s")
        if gpu_v:
            speedup = gpu_v / cpu_v
            print(f"  GPU:     {gpu_v:>10,.0f} it/s  ({speedup:>5.1f}x speedup)")
        else:
            print(f"  GPU:     Not available")

    if 'cpu_mccfr' in results:
        cpu_m = results['cpu_mccfr']
        gpu_m = results['gpu_mccfr']
        print(f"\nMCCFR:")
        print(f"  CPU:     {cpu_m:>10,.0f} it/s")
        if gpu_m:
            speedup = gpu_m / cpu_m
            print(f"  GPU:     {gpu_m:>10,.0f} it/s  ({speedup:>5.1f}x speedup)")
        else:
            print(f"  GPU:     Not available")

    print("\n" + "="*80)

    # Estimate time savings for your research
    if gpu_available and 'gpu_mccfr' in results and results['gpu_mccfr']:
        print("\nTIME SAVINGS FOR YOUR RESEARCH:")
        print("─"*80)

        # Your compare_dcfr_research_3p.py uses 1M iterations
        total_iters = 1_000_000

        cpu_time = total_iters / results['cpu_mccfr']
        gpu_time = total_iters / results['gpu_mccfr']
        savings = cpu_time - gpu_time

        print(f"For 1M iterations (your research workload):")
        print(f"  CPU time:     {cpu_time/60:>8.1f} minutes ({cpu_time/3600:.1f} hours)")
        print(f"  GPU time:     {gpu_time/60:>8.1f} minutes ({gpu_time/3600:.1f} hours)")
        print(f"  Time saved:   {savings/60:>8.1f} minutes ({savings/3600:.1f} hours)")
        print("\n" + "="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
