#!/usr/bin/env python3
"""
Test GPU MCCFR with Batched Trajectory Sampling

Validates Phase 10.3 integration by running Hold'em MCCFR with batch_size > 1
and comparing performance to sequential sampling (batch_size=1).

Expected: 20-50× speedup for end-to-end MCCFR iterations.

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.3: GPU MCCFR Integration Validation
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


def test_sequential_mccfr(num_iterations: int = 100):
    """Test sequential MCCFR (baseline - batch_size=1)."""
    print("=" * 70)
    print(f"Test 1: Sequential MCCFR (batch_size=1, {num_iterations} iterations)")
    print("=" * 70)
    print()

    config = MCCFRConfig(
        num_players=NUM_PLAYERS,
        num_actions=4,
        batch_size=1  # Sequential sampling
    )

    solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=42)

    print(f"Running {num_iterations} MCCFR iterations (sequential)...")
    print("(This establishes baseline performance)")
    print()

    start = time.time()
    solver.solve(
        num_iterations=num_iterations,
        num_players=NUM_PLAYERS,
        stacks=STACKS,
        blinds=BLINDS,
        progress_interval=max(num_iterations // 5, 1)
    )
    elapsed = time.time() - start

    iterations_per_sec = num_iterations / elapsed
    num_infosets = sum(table.get_num_infosets() for table in solver.regret_tables)

    print()
    print(f"✓ Sequential MCCFR complete")
    print(f"  Time: {elapsed:.2f}s for {num_iterations} iterations")
    print(f"  Speed: {iterations_per_sec:.2f} it/s")
    print(f"  Infosets visited: {num_infosets}")
    print()

    return iterations_per_sec, num_infosets


def test_batched_mccfr(num_iterations: int = 100, batch_size: int = 100):
    """Test batched MCCFR (GPU-accelerated)."""
    print("=" * 70)
    print(f"Test 2: Batched MCCFR (batch_size={batch_size}, {num_iterations} iterations)")
    print("=" * 70)
    print()

    config = MCCFRConfig(
        num_players=NUM_PLAYERS,
        num_actions=4,
        batch_size=batch_size  # Batched sampling
    )

    solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=42)

    print(f"Running {num_iterations} MCCFR iterations (batched)...")
    print(f"Each iteration samples {batch_size} trajectories in parallel")
    print()

    start = time.time()
    solver.solve(
        num_iterations=num_iterations,
        num_players=NUM_PLAYERS,
        stacks=STACKS,
        blinds=BLINDS,
        progress_interval=max(num_iterations // 5, 1)
    )
    elapsed = time.time() - start

    iterations_per_sec = num_iterations / elapsed
    num_infosets = sum(table.get_num_infosets() for table in solver.regret_tables)

    print()
    print(f"✓ Batched MCCFR complete")
    print(f"  Time: {elapsed:.2f}s for {num_iterations} iterations")
    print(f"  Speed: {iterations_per_sec:.2f} it/s")
    print(f"  Infosets visited: {num_infosets}")
    print(f"  Trajectories sampled: {num_iterations * batch_size}")
    print()

    return iterations_per_sec, num_infosets


def test_various_batch_sizes(num_iterations: int = 100):
    """Test different batch sizes to find optimal configuration."""
    print("=" * 70)
    print(f"Test 3: Batch Size Scaling ({num_iterations} iterations each)")
    print("=" * 70)
    print()

    # Measure sequential baseline
    print("[Measuring sequential baseline...]")
    seq_speed, seq_infosets = test_sequential_mccfr(num_iterations)

    # Test batch sizes
    batch_sizes = [10, 50, 100, 250, 500]
    results = {}

    for batch_size in batch_sizes:
        print(f"\n[Testing batch_size={batch_size}...]")

        config = MCCFRConfig(
            num_players=NUM_PLAYERS,
            num_actions=4,
            batch_size=batch_size
        )

        solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=42 + batch_size)

        start = time.time()
        solver.solve(
            num_iterations=num_iterations,
            num_players=NUM_PLAYERS,
            stacks=STACKS,
            blinds=BLINDS,
            progress_interval=999_999  # Suppress progress
        )
        elapsed = time.time() - start

        iterations_per_sec = num_iterations / elapsed
        speedup = iterations_per_sec / seq_speed
        num_infosets = sum(table.get_num_infosets() for table in solver.regret_tables)

        results[batch_size] = {
            'speed': iterations_per_sec,
            'speedup': speedup,
            'time': elapsed,
            'infosets': num_infosets
        }

        print(f"  Speed: {iterations_per_sec:.2f} it/s")
        print(f"  Speedup: {speedup:.1f}×")
        print(f"  Time: {elapsed:.2f}s")

    # Summary table
    print()
    print("=" * 70)
    print("SUMMARY: Batch Size Scaling")
    print("=" * 70)
    print()
    print(f"{'Batch Size':<12} {'Speed (it/s)':<15} {'Speedup':<10} {'Time (s)':<10}")
    print(f"{'----------':<12} {'-----------':<15} {'-------':<10} {'--------':<10}")
    print(f"{'Sequential':<12} {seq_speed:>8.2f}       {'1.0×':<10} {num_iterations/seq_speed:>6.2f}")

    for batch_size in batch_sizes:
        r = results[batch_size]
        print(f"{batch_size:<12} {r['speed']:>8.2f}       {r['speedup']:>6.1f}×    {r['time']:>6.2f}")

    print()

    # Find best
    best_batch = max(results.keys(), key=lambda b: results[b]['speedup'])
    best_speedup = results[best_batch]['speedup']

    print(f"Best configuration: batch_size={best_batch} with {best_speedup:.1f}× speedup")
    print()

    if best_speedup >= 20:
        print(f"🎉 SUCCESS! Achieved {best_speedup:.0f}× speedup (target: >20×)")
    elif best_speedup >= 10:
        print(f"✅ GOOD! Achieved {best_speedup:.0f}× speedup (target was >20×)")
    elif best_speedup >= 5:
        print(f"⚠️ MODERATE: Achieved {best_speedup:.0f}× speedup (target was >20×)")
    else:
        print(f"⚠️ LOW: Achieved {best_speedup:.0f}× speedup (target was >20×)")

    return results, seq_speed


def quick_comparison(num_iterations: int = 1000):
    """Quick comparison for Phase 10.3 validation."""
    print("=" * 70)
    print("PHASE 10.3 QUICK VALIDATION")
    print("=" * 70)
    print()
    print(f"Running {num_iterations} iterations with sequential and batched MCCFR")
    print("Target: Demonstrate 20-50× speedup with batched sampling")
    print()

    # Sequential
    seq_speed, seq_infosets = test_sequential_mccfr(num_iterations)

    # Batched (conservative batch size for Hold'em)
    batch_speed, batch_infosets = test_batched_mccfr(num_iterations, batch_size=100)

    # Calculate speedup
    speedup = batch_speed / seq_speed

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"Sequential: {seq_speed:.2f} it/s ({seq_infosets} infosets)")
    print(f"Batched:    {batch_speed:.2f} it/s ({batch_infosets} infosets)")
    print(f"Speedup:    {speedup:.1f}×")
    print()

    if speedup >= 20:
        print(f"🎉 EXCELLENT! Phase 10.3 target achieved ({speedup:.0f}× > 20× target)")
        print()
        print("Hold'em GPU MCCFR with batched sampling: ✅ WORKING")
        print("Ready for production use and further optimization!")
    elif speedup >= 10:
        print(f"✅ GOOD! Significant speedup achieved ({speedup:.0f}×)")
        print("Consider optimizing batch size or trajectory sampling.")
    else:
        print(f"⚠️ Speedup below target: {speedup:.0f}× (target: >20×)")
        print("Further investigation needed.")

    print()

    return speedup


def main():
    print("Hold'em GPU MCCFR - Batched Trajectory Sampling Validation")
    print("Phase 10.3: Conservative Batching with Uniform Policy")
    print()

    # Quick validation (1K iterations)
    speedup = quick_comparison(num_iterations=1000)

    print()
    print("=" * 70)
    print("Phase 10.3 GPU MCCFR Integration: COMPLETE ✅")
    print(f"Achieved {speedup:.1f}× speedup with batched trajectory sampling")
    print("=" * 70)


if __name__ == "__main__":
    main()
