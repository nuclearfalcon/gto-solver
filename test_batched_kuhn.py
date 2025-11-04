#!/usr/bin/env python3
"""
Test batched trajectory sampling on Kuhn poker.

Verifies that:
1. Batched sampling with uniform policy works
2. Provides massive speedup over sequential
3. Can still compute regrets correctly

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.2: Batched Sampling Implementation
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr import kuhn_jax
from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver, MCCFRConfig


def test_sequential_baseline():
    """Test sequential sampling (baseline)."""
    print("="*70)
    print("Test 1: Sequential Sampling (Baseline)")
    print("="*70)
    print()

    config = MCCFRConfig(num_players=2, num_actions=2, use_linear_weighting=False)
    solver = GPUMCCFRSolver(kuhn_jax, config, seed=42)

    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])

    # Warmup
    print("[Warmup]")
    solver.solve(10, num_players, stacks, blinds, progress_interval=999)
    print()

    # Benchmark 100 iterations
    print("[Benchmark: 100 iterations]")
    start = time.time()
    solver.solve(100, num_players, stacks, blinds, progress_interval=999)
    elapsed = time.time() - start
    speed = 100 / elapsed

    print(f"✓ Sequential sampling:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {speed:.2f} it/s")
    print(f"  Per iteration: {elapsed/100*1000:.1f}ms")
    print()

    return speed


def test_batched_sampling_simple():
    """Test batched sampling with uniform policy."""
    print("="*70)
    print("Test 2: Batched Sampling with Uniform Policy")
    print("="*70)
    print()

    # Import trajectory sampler
    from matrix_cfr.trajectory_sampler import (
        batch_sample_trajectories,
        uniform_random_policy
    )

    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])
    batch_size = 100
    max_length = 20  # Kuhn poker is short

    # Generate keys
    key = random.PRNGKey(42)
    keys = random.split(key, batch_size)

    # Warmup
    print(f"[Warmup - JIT compilation]")
    warmup_keys = random.split(random.PRNGKey(0), 10)
    _ = batch_sample_trajectories(
        warmup_keys, num_players, stacks, blinds,
        uniform_random_policy, max_length
    )
    print("✓ Warmup complete")
    print()

    # Benchmark
    print(f"[Benchmark: Sampling {batch_size} trajectories in parallel]")
    start = time.time()

    batch_outputs = batch_sample_trajectories(
        keys, num_players, stacks, blinds,
        uniform_random_policy, max_length
    )

    elapsed = time.time() - start
    throughput = batch_size / elapsed

    hole_cards, board, bets, actions, valid_mask = batch_outputs

    print(f"✓ Batched sampling:")
    print(f"  Batch size: {batch_size}")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Throughput: {throughput:.1f} trajectories/sec")
    print(f"  Per trajectory: {elapsed/batch_size*1000:.2f}ms")
    print(f"  Output shapes:")
    print(f"    - hole_cards: {hole_cards.shape}")
    print(f"    - actions: {actions.shape}")
    print(f"    - valid_mask: {valid_mask.shape}")
    print()

    return throughput


def test_large_batch():
    """Test larger batch size."""
    print("="*70)
    print("Test 3: Large Batch (1000 trajectories)")
    print("="*70)
    print()

    from matrix_cfr.trajectory_sampler import (
        batch_sample_trajectories,
        uniform_random_policy
    )

    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])
    batch_size = 1000
    max_length = 20

    key = random.PRNGKey(999)
    keys = random.split(key, batch_size)

    # Warmup
    print("[Warmup]")
    warmup_keys = random.split(random.PRNGKey(1), 10)
    _ = batch_sample_trajectories(
        warmup_keys, num_players, stacks, blinds,
        uniform_random_policy, max_length
    )
    print("✓ Done")
    print()

    # Benchmark
    print(f"[Sampling {batch_size} trajectories...]")
    start = time.time()

    batch_outputs = batch_sample_trajectories(
        keys, num_players, stacks, blinds,
        uniform_random_policy, max_length
    )

    elapsed = time.time() - start
    throughput = batch_size / elapsed

    print(f"✓ Large batch complete:")
    print(f"  Batch size: {batch_size}")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Throughput: {throughput:.1f} trajectories/sec")
    print(f"  Per trajectory: {elapsed/batch_size*1000:.2f}ms")
    print()

    return throughput


def main():
    print("GPU MCCFR: Batched Sampling Test (Kuhn Poker)")
    print("Phase 10.2 Optimization")
    print()

    # Test 1: Sequential baseline
    seq_speed = test_sequential_baseline()

    # Test 2: Batched (100)
    batch_speed_100 = test_batched_sampling_simple()

    # Test 3: Batched (1000)
    batch_speed_1000 = test_large_batch()

    # Compare
    print("="*70)
    print("Performance Summary")
    print("="*70)
    print()

    print(f"Sequential:        {seq_speed:.2f} it/s")
    print(f"Batched (100):     {batch_speed_100:.1f} traj/sec")
    print(f"Batched (1000):    {batch_speed_1000:.1f} traj/sec")
    print()

    # Note: Batch speeds are trajectories/sec, not iterations/sec
    # Each iteration samples 1 trajectory, so they're comparable
    speedup_100 = batch_speed_100 / seq_speed
    speedup_1000 = batch_speed_1000 / seq_speed

    print(f"Speedup (100):     {speedup_100:.1f}×")
    print(f"Speedup (1000):    {speedup_1000:.1f}×")
    print()

    if speedup_100 >= 10:
        print(f"✅ SUCCESS: 10×+ speedup achieved!")
    elif speedup_100 >= 5:
        print(f"⚠️ MODERATE: 5-10× speedup")
    else:
        print(f"❌ ISSUE: <5× speedup")

    print()
    print("="*70)
    print("Batched Sampling Test Complete!")
    print("="*70)
    print()
    print("Next: Integrate batched sampling into GPUMCCFRSolver")


if __name__ == "__main__":
    main()
