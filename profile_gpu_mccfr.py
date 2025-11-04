#!/usr/bin/env python3
"""
Profiling script for GPU MCCFR solver.

Identifies performance bottlenecks in:
1. Recursive CFV computation (_compute_cfv_recursive)
2. Trajectory sampling (_sample_trajectory)
3. RegretTable operations (dict lookups, strategy computation)
4. State copying (JAX immutable updates)

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python profile_gpu_mccfr.py
    python profile_gpu_mccfr.py --game holdem --iterations 100

Phase 10.2: Performance Profiling
"""

import argparse
import cProfile
import pstats
import io
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr import kuhn_jax, holdem_jax
from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver, MCCFRConfig


def profile_kuhn(iterations: int = 1000, output_file: str = None):
    """
    Profile GPU MCCFR on Kuhn poker.

    Args:
        iterations: Number of iterations to profile
        output_file: Path to save profiling data (None = don't save)
    """
    print("=" * 70)
    print("GPU MCCFR PROFILING - KUHN POKER")
    print("=" * 70)
    print()

    # Setup
    print(f"[Configuration]")
    print(f"  Game: Kuhn Poker (2-player, 2-action)")
    print(f"  Iterations: {iterations:,}")
    print()

    config = MCCFRConfig(
        num_players=2,
        num_actions=2,
        use_linear_weighting=False
    )

    solver = GPUMCCFRSolver(kuhn_jax, config, seed=42)

    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])

    # Warm up (JIT compilation)
    print("[Warmup - JIT Compilation]")
    warmup_start = time.time()
    solver.solve(
        num_iterations=10,
        num_players=num_players,
        stacks=stacks,
        blinds=blinds,
        progress_interval=999
    )
    warmup_time = time.time() - warmup_start
    print(f"✓ Warmup complete ({warmup_time:.2f}s for 10 iterations)")
    print()

    # Profile main run
    print(f"[Profiling {iterations} iterations]")
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    solver.solve(
        num_iterations=iterations,
        num_players=num_players,
        stacks=stacks,
        blinds=blinds,
        progress_interval=max(iterations // 10, 1)
    )
    elapsed_time = time.time() - start_time

    profiler.disable()

    # Print results
    print()
    print("=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    print()

    # Basic stats
    print("[Performance Summary]")
    speed = iterations / elapsed_time
    print(f"  Total time:      {elapsed_time:.2f}s")
    print(f"  Iterations:      {iterations:,}")
    print(f"  Speed:           {speed:.2f} it/s")
    print(f"  Time per iter:   {elapsed_time/iterations*1000:.1f}ms")
    print()

    # Extract policy stats
    policy_p0 = solver.get_average_policy(player=0)
    policy_p1 = solver.get_average_policy(player=1)
    print(f"  Infosets (P0):   {len(policy_p0)}")
    print(f"  Infosets (P1):   {len(policy_p1)}")
    print(f"  Total infosets:  {len(policy_p0) + len(policy_p1)}")
    print()

    # Top time consumers
    print("[Top 20 Time Consumers]")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

    # Key functions to track
    print("[Key Function Breakdown]")
    print()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()

    # Look for specific functions
    key_functions = [
        '_compute_cfv_recursive',
        '_sample_trajectory',
        'compute_counterfactual_values',
        'get_strategy',
        'update_regrets',
        'apply_action',
        'legal_actions',
        'is_terminal'
    ]

    for func_name in key_functions:
        ps.sort_stats('cumulative')
        s = io.StringIO()
        ps.stream = s
        ps.print_stats(func_name)
        output = s.getvalue()

        if output and 'function calls' not in output:  # Found the function
            lines = [line for line in output.split('\n') if func_name in line]
            if lines:
                print(f"  {func_name}:")
                for line in lines[:3]:  # Print up to 3 matches
                    print(f"    {line.strip()}")

    print()

    # Save profiling data
    if output_file:
        profiler.dump_stats(output_file)
        print(f"[Profiling Data Saved]")
        print(f"  File: {output_file}")
        print(f"  View with: python -m pstats {output_file}")
        print()


def profile_holdem(iterations: int = 100, output_file: str = None):
    """
    Profile GPU MCCFR on Hold'em tiny.

    Args:
        iterations: Number of iterations to profile
        output_file: Path to save profiling data (None = don't save)
    """
    print("=" * 70)
    print("GPU MCCFR PROFILING - HOLD'EM TINY")
    print("=" * 70)
    print()

    # Setup
    print(f"[Configuration]")
    print(f"  Game: Hold'em Tiny (10-card deck, 2 players, 4 actions)")
    print(f"  Iterations: {iterations:,}")
    print()

    config = MCCFRConfig(
        num_players=2,
        num_actions=4,  # fold, call, pot, allin
        use_linear_weighting=False
    )

    solver = GPUMCCFRSolver(holdem_jax, config, seed=42)

    num_players = 2
    stacks = jnp.array([1000.0, 1000.0])
    blinds = jnp.array([50.0, 100.0])

    # Warm up
    print("[Warmup - JIT Compilation]")
    warmup_start = time.time()
    solver.solve(
        num_iterations=5,
        num_players=num_players,
        stacks=stacks,
        blinds=blinds,
        progress_interval=999
    )
    warmup_time = time.time() - warmup_start
    print(f"✓ Warmup complete ({warmup_time:.2f}s for 5 iterations)")
    print()

    # Profile main run
    print(f"[Profiling {iterations} iterations]")
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    solver.solve(
        num_iterations=iterations,
        num_players=num_players,
        stacks=stacks,
        blinds=blinds,
        progress_interval=max(iterations // 10, 1)
    )
    elapsed_time = time.time() - start_time

    profiler.disable()

    # Print results
    print()
    print("=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    print()

    # Basic stats
    print("[Performance Summary]")
    speed = iterations / elapsed_time
    print(f"  Total time:      {elapsed_time:.2f}s")
    print(f"  Iterations:      {iterations:,}")
    print(f"  Speed:           {speed:.2f} it/s")
    print(f"  Time per iter:   {elapsed_time/iterations*1000:.1f}ms")
    print()

    # Extract policy stats
    policy_p0 = solver.get_average_policy(player=0)
    policy_p1 = solver.get_average_policy(player=1)
    print(f"  Infosets (P0):   {len(policy_p0)}")
    print(f"  Infosets (P1):   {len(policy_p1)}")
    print(f"  Total infosets:  {len(policy_p0) + len(policy_p1)}")
    print()

    # Top time consumers
    print("[Top 20 Time Consumers]")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

    # Save profiling data
    if output_file:
        profiler.dump_stats(output_file)
        print(f"[Profiling Data Saved]")
        print(f"  File: {output_file}")
        print(f"  View with: python -m pstats {output_file}")
        print()


def compare_timings():
    """
    Quick comparison of Kuhn vs Hold'em performance.
    """
    print("=" * 70)
    print("QUICK PERFORMANCE COMPARISON")
    print("=" * 70)
    print()

    # Kuhn poker - 100 iterations
    print("[Kuhn Poker - 100 iterations]")
    config_kuhn = MCCFRConfig(num_players=2, num_actions=2, use_linear_weighting=False)
    solver_kuhn = GPUMCCFRSolver(kuhn_jax, config_kuhn, seed=42)

    # Warmup
    solver_kuhn.solve(
        num_iterations=10,
        num_players=2,
        stacks=jnp.array([100.0, 100.0]),
        blinds=jnp.array([1.0, 1.0]),
        progress_interval=999
    )

    # Time
    start = time.time()
    solver_kuhn.solve(
        num_iterations=100,
        num_players=2,
        stacks=jnp.array([100.0, 100.0]),
        blinds=jnp.array([1.0, 1.0]),
        progress_interval=999
    )
    kuhn_time = time.time() - start
    kuhn_speed = 100 / kuhn_time

    print(f"  Time: {kuhn_time:.2f}s")
    print(f"  Speed: {kuhn_speed:.2f} it/s")
    print()

    # Hold'em tiny - 100 iterations
    print("[Hold'em Tiny - 100 iterations]")
    config_holdem = MCCFRConfig(num_players=2, num_actions=4, use_linear_weighting=False)
    solver_holdem = GPUMCCFRSolver(holdem_jax, config_holdem, seed=42)

    # Warmup
    solver_holdem.solve(
        num_iterations=5,
        num_players=2,
        stacks=jnp.array([1000.0, 1000.0]),
        blinds=jnp.array([50.0, 100.0]),
        progress_interval=999
    )

    # Time
    start = time.time()
    solver_holdem.solve(
        num_iterations=100,
        num_players=2,
        stacks=jnp.array([1000.0, 1000.0]),
        blinds=jnp.array([50.0, 100.0]),
        progress_interval=999
    )
    holdem_time = time.time() - start
    holdem_speed = 100 / holdem_time

    print(f"  Time: {holdem_time:.2f}s")
    print(f"  Speed: {holdem_speed:.2f} it/s")
    print()

    # Comparison
    print("[Comparison]")
    print(f"  Kuhn speed:    {kuhn_speed:.2f} it/s")
    print(f"  Hold'em speed: {holdem_speed:.2f} it/s")
    print(f"  Ratio:         {kuhn_speed/holdem_speed:.2f}× (Kuhn is faster)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Profile GPU MCCFR solver performance'
    )
    parser.add_argument(
        '--game',
        type=str,
        choices=['kuhn', 'holdem', 'both'],
        default='both',
        help='Which game to profile (default: both)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Number of iterations (default: 1000 for Kuhn, 100 for Hold\'em)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='profiling_results',
        help='Output directory for profiling data (default: profiling_results)'
    )
    parser.add_argument(
        '--save-profile',
        action='store_true',
        help='Save profiling data to files'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick comparison only (100 iterations each)'
    )

    args = parser.parse_args()

    # Create output directory
    if args.save_profile:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("GPU MCCFR Performance Profiler")
    print("Phase 10.2 Optimization")
    print()

    if args.quick:
        compare_timings()
        return

    # Profile games
    if args.game in ['kuhn', 'both']:
        kuhn_iterations = args.iterations if args.iterations else 1000
        kuhn_output = f"{args.output_dir}/kuhn_{kuhn_iterations}iter.prof" if args.save_profile else None
        profile_kuhn(kuhn_iterations, kuhn_output)

    if args.game in ['holdem', 'both']:
        holdem_iterations = args.iterations if args.iterations else 100
        holdem_output = f"{args.output_dir}/holdem_{holdem_iterations}iter.prof" if args.save_profile else None

        if args.game == 'both':
            print("\n\n")

        profile_holdem(holdem_iterations, holdem_output)

    print("=" * 70)
    print("Profiling Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Analyze bottlenecks in the output above")
    print("2. Focus optimization on top time consumers")
    print("3. Implement batched trajectory sampling for speedup")


if __name__ == "__main__":
    main()
