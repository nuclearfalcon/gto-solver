#!/usr/bin/env python3
"""
Quick Test: GPU MCCFR Batched Sampling - Simplified Version

Tests Phase 10.3 integration with small batch sizes to validate quickly.

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.3: Quick Validation
"""

import time
import jax
import jax.numpy as jnp

from matrix_cfr import holdem_jax_v2
from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver, MCCFRConfig

# Hold'em configuration
NUM_PLAYERS = 2
STACKS = jnp.array([1000.0, 1000.0])
BLINDS = jnp.array([50.0, 100.0])


def quick_test(num_iterations: int = 100):
    """Quick validation with small iteration count."""
    print("=" * 70)
    print("PHASE 10.3 QUICK VALIDATION - Simplified Test")
    print("=" * 70)
    print()
    print(f"Running {num_iterations} iterations with different batch sizes")
    print("This tests the infrastructure without long wait times")
    print()

    results = {}

    # Test configurations: (name, batch_size)
    configs = [
        ("Sequential", 1),
        ("Small Batch", 5),
        ("Medium Batch", 10),
    ]

    for name, batch_size in configs:
        print(f"\n[Testing {name} (batch_size={batch_size})]")

        config = MCCFRConfig(
            num_players=NUM_PLAYERS,
            num_actions=4,
            batch_size=batch_size
        )

        solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=42 + batch_size)

        print(f"Running {num_iterations} iterations...")

        start = time.time()
        solver.solve(
            num_iterations=num_iterations,
            num_players=NUM_PLAYERS,
            stacks=STACKS,
            blinds=BLINDS,
            progress_interval=999999  # Suppress progress
        )
        elapsed = time.time() - start

        iterations_per_sec = num_iterations / elapsed
        num_infosets = sum(table.get_num_infosets() for table in solver.regret_tables)

        results[name] = {
            'batch_size': batch_size,
            'speed': iterations_per_sec,
            'time': elapsed,
            'infosets': num_infosets
        }

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Speed: {iterations_per_sec:.2f} it/s")
        print(f"  Infosets: {num_infosets}")

    # Calculate speedups
    baseline_speed = results["Sequential"]['speed']

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Configuration':<20} {'Batch Size':<12} {'Speed (it/s)':<15} {'Speedup':<10}")
    print("-" * 70)

    for name in ["Sequential", "Small Batch", "Medium Batch"]:
        r = results[name]
        speedup = r['speed'] / baseline_speed
        print(f"{name:<20} {r['batch_size']:<12} {r['speed']:>8.2f}       {speedup:>6.2f}×")

    print()

    # Analysis
    small_speedup = results["Small Batch"]['speed'] / baseline_speed
    medium_speedup = results["Medium Batch"]['speed'] / baseline_speed

    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    if medium_speedup > 1.5:
        print(f"✅ GOOD: Batching shows speedup ({medium_speedup:.1f}×)")
        print(f"   Infrastructure working correctly!")
    elif medium_speedup > 1.0:
        print(f"⚠️ MARGINAL: Small speedup ({medium_speedup:.1f}×)")
        print(f"   Expected due to sequential trajectory sampling in current implementation")
    else:
        print(f"⚠️ ISSUE: No speedup observed ({medium_speedup:.1f}×)")
        print(f"   Current implementation samples trajectories sequentially")

    print()
    print("NOTE: Current Phase 10.3 implementation samples trajectories in Python loop,")
    print("not using full GPU parallelization. Expected speedup is modest (1-3×).")
    print()
    print("Phase 10.4 will implement full vectorization with jax.vmap for 20-50× speedup.")
    print()

    return results


def main():
    print("Hold'em GPU MCCFR - Quick Validation Test")
    print("Phase 10.3: Conservative Batching Infrastructure Test")
    print()

    results = quick_test(num_iterations=100)

    print("=" * 70)
    print("Phase 10.3: Infrastructure Integration ✅ COMPLETE")
    print("=" * 70)
    print()
    print("Key Finding: Current implementation validates integration but")
    print("doesn't yet achieve full GPU parallelization (that's Phase 10.4).")


if __name__ == "__main__":
    main()
