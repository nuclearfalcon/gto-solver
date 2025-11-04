#!/usr/bin/env python3
"""
Test Phase 10.5: GPU-Resident Bucketed MCCFR - Hold'em Performance Benchmark

This test measures the actual speedup achieved by the GPU-resident approach
on full Hold'em poker and validates performance targets.

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.5.6: Testing & Validation (Final Phase)
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver, MCCFRConfig, GPURegretTable
from matrix_cfr import holdem_jax_v2


def test_gpu_resident_holdem():
    """Test GPU-resident MCCFR on Hold'em poker."""
    print("=" * 70)
    print("Phase 10.5.6: Hold'em GPU-Resident MCCFR Performance Benchmark")
    print("=" * 70)
    print()

    # Configuration
    num_players = 2
    stacks = jnp.array([1000.0, 1000.0])
    blinds = jnp.array([50.0, 100.0])
    num_iterations = 10
    batch_size = 100
    num_buckets = 10_000
    num_hand_buckets = 200
    num_pot_buckets = 10

    print("Configuration:")
    print(f"  Players: {num_players}")
    print(f"  Stacks: {stacks}")
    print(f"  Blinds: {blinds}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num buckets: {num_buckets:,}")
    print(f"  Hand buckets: {num_hand_buckets}")
    print(f"  Pot buckets: {num_pot_buckets}")
    print()

    # Create solver with GPU-resident mode
    config = MCCFRConfig(
        batch_size=batch_size,
        num_actions=4  # fold, call, bet, all-in
    )
    solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=42)

    # Initialize GPU regret tables
    print("Initializing GPU regret tables...")
    solver.regret_tables = [
        GPURegretTable(num_buckets=num_buckets, num_actions=config.num_actions)
        for _ in range(num_players)
    ]

    mem_usage = solver.regret_tables[0].get_memory_usage_mb() * num_players
    print(f"  GPU memory: {mem_usage:.2f} MB per player × {num_players} = {mem_usage * num_players:.2f} MB total")
    print()

    # Run GPU-resident iterations
    print(f"Running {num_iterations} GPU-resident iterations...")
    print(f"{'Iter':>5} | {'Time (s)':>8} | {'Speed (it/s)':>12} | {'Throughput (traj/s)':>20}")
    print("-" * 70)

    start_time = time.time()

    for i in range(num_iterations):
        iter_start = time.time()

        # Run GPU-resident iteration
        traj_length = solver.run_iteration_gpu_resident(
            num_players,
            stacks,
            blinds,
            num_buckets=num_buckets,
            num_hand_buckets=num_hand_buckets,
            num_pot_buckets=num_pot_buckets
        )

        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        speed = (i + 1) / elapsed
        throughput = speed * batch_size

        print(f"{i+1:5d} | {iter_time:8.2f} | {speed:12.2f} | {throughput:20.0f}")

    total_time = time.time() - start_time
    final_speed = num_iterations / total_time
    final_throughput = final_speed * batch_size

    print("-" * 70)
    print()

    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Total time: {total_time:.2f}s for {num_iterations} iterations")
    print(f"Speed: {final_speed:.3f} it/s")
    print(f"Throughput: {final_throughput:.0f} trajectories/s")
    print()

    # Compare to baseline
    baseline_speed = 0.0022  # From Phase 10.4 docs (sequential MCCFR)
    speedup = final_throughput / baseline_speed

    print("Baseline Comparison:")
    print(f"  Baseline: {baseline_speed} it/s (sequential MCCFR)")
    print(f"  GPU-Resident: {final_throughput:.0f} traj/s")
    print(f"  **Speedup: {speedup:.0f}×**")
    print()

    # Evaluate success criteria
    print("=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    print()

    criteria = [
        ("Minimum (454× speedup)", 1.0, 454),
        ("Target (900× speedup)", 2.0, 900),
        ("Stretch (1364× speedup)", 3.0, 1364),
    ]

    best_met = None
    for name, min_speed, min_speedup in criteria:
        met = final_speed >= min_speed
        symbol = "✓" if met else "✗"
        print(f"{symbol} {name:30s} | Speed ≥ {min_speed:.1f} it/s | Speedup ≥ {min_speedup}×")
        if met:
            best_met = name

    print()

    if best_met:
        print(f"✓ SUCCESS! Achieved {best_met}")
    else:
        print("⚠ Did not meet minimum success criteria")
        print(f"  Current speed: {final_speed:.3f} it/s")
        print(f"  Current throughput: {final_throughput:.0f} traj/s")
        print(f"  Current speedup: {speedup:.0f}×")

    print()
    print("=" * 70)
    print("Phase 10.5.6: Hold'em Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_gpu_resident_holdem()
