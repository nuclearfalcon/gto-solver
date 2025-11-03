#!/usr/bin/env python3
"""
Detailed profiling of Matrix CFR solver iterations.

Focus on measuring:
1. Bottom-up utility propagation time
2. Reach probability computation time
3. Batch processing overhead
4. Regret matching time
"""

import time
import pyspiel
import jax
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver


def profile_leduc():
    print("=" * 70)
    print("DETAILED SOLVER PROFILING - LEDUC POKER")
    print("=" * 70)
    print()

    # Load game
    print("Loading Leduc poker...")
    game = pyspiel.load_game("leduc_poker")

    # Create solver
    print("Creating sparse solver...")
    solver = MatrixCFRSolver(game, use_sparse=True)

    print(f"Game size: {solver.matrix_repr.num_nodes} nodes")
    print(f"Infosets: {solver.matrix_repr.num_infosets}")
    print(f"Infoset-actions: {solver.matrix_repr.num_infoset_actions}")
    print()

    # Warm up (JIT compilation)
    print("Warming up (JIT compilation)...")
    solver.solve(iterations=2, progress_interval=999)
    print("✓ Warm-up complete")
    print()

    # Profile 10 iterations with detailed timing
    print("Profiling 10 iterations with detailed timing...")
    print()

    num_iterations = 10
    timings = {
        'total': [],
        'bottom_up': [],
        'reach': [],
        'regret': [],
        'strategy_update': []
    }

    for i in range(num_iterations):
        iter_start = time.time()

        # We'll need to manually step through the iteration to time components
        # For now, just time the full iteration
        solver.solve(iterations=1, progress_interval=999)

        iter_time = time.time() - iter_start
        timings['total'].append(iter_time)

        if (i + 1) % 2 == 0:
            avg = sum(timings['total'][-2:]) / 2
            print(f"Iteration {i+1}: {iter_time:.3f}s (avg last 2: {avg:.3f}s)")

    print()
    print("=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    print()

    avg_total = sum(timings['total']) / len(timings['total'])
    min_total = min(timings['total'])
    max_total = max(timings['total'])

    print(f"Iteration time:")
    print(f"  Average: {avg_total:.3f}s ({1/avg_total:.2f} it/s)")
    print(f"  Min: {min_total:.3f}s")
    print(f"  Max: {max_total:.3f}s")
    print()

    print(f"Estimated time for 1000 iterations: {avg_total * 1000 / 60:.1f} minutes")
    print(f"Estimated time for 10000 iterations: {avg_total * 10000 / 3600:.1f} hours")
    print()

    # Calculate target speed
    target_speed = 1.0  # 1 it/s
    current_speed = 1 / avg_total
    speedup_needed = target_speed / current_speed

    print(f"Current speed: {current_speed:.2f} it/s")
    print(f"Target speed: {target_speed:.2f} it/s")
    print(f"Speedup needed: {speedup_needed:.1f}x")
    print()


def profile_kuhn():
    print("=" * 70)
    print("DETAILED SOLVER PROFILING - KUHN POKER (BASELINE)")
    print("=" * 70)
    print()

    # Load game
    print("Loading Kuhn poker...")
    game = pyspiel.load_game("kuhn_poker")

    # Create solver
    print("Creating sparse solver...")
    solver = MatrixCFRSolver(game, use_sparse=True)

    print(f"Game size: {solver.matrix_repr.num_nodes} nodes")
    print(f"Infosets: {solver.matrix_repr.num_infosets}")
    print()

    # Warm up
    print("Warming up...")
    solver.solve(iterations=5, progress_interval=999)
    print("✓ Warm-up complete")
    print()

    # Profile 20 iterations
    print("Profiling 20 iterations...")
    start = time.time()
    solver.solve(iterations=20, progress_interval=999)
    elapsed = time.time() - start

    print()
    print(f"20 iterations in {elapsed:.3f}s")
    print(f"Average: {elapsed/20:.3f}s per iteration")
    print(f"Speed: {20/elapsed:.2f} it/s")
    print()


if __name__ == "__main__":
    profile_kuhn()
    print("\n\n")
    profile_leduc()
