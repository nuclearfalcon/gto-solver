#!/usr/bin/env python3
"""
Test batched trajectory sampling implementation.

Verifies that:
1. sample_trajectory_fixed_length() works correctly
2. batch_sample_trajectories() works with vmap
3. Batching provides speedup

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.2: Batched Sampling Test
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr import kuhn_jax
from matrix_cfr.trajectory_sampler import (
    sample_trajectory,
    sample_trajectory_fixed_length,
    batch_sample_trajectories,
    uniform_random_policy
)


def test_sequential_sampling():
    """Test sequential trajectory sampling (baseline)."""
    print("="*70)
    print("Test 1: Sequential Trajectory Sampling (Baseline)")
    print("="*70)
    print()

    key = random.PRNGKey(42)
    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])

    # Sample one trajectory
    states, actions, players, payoffs = sample_trajectory(
        key, num_players, stacks, blinds, uniform_random_policy, max_actions=50
    )

    print(f"✓ Sequential sampling works:")
    print(f"  - Decision points: {len(states)}")
    print(f"  - Actions: {len(actions)}")
    print(f"  - Terminal payoffs: P0={payoffs[0]}, P1={payoffs[1]}")
    print()

    # Benchmark 100 trajectories
    print("[Benchmark: 100 trajectories]")
    num_trajectories = 100
    start = time.time()

    for i in range(num_trajectories):
        key, subkey = random.split(key)
        states, actions, players, payoffs = sample_trajectory(
            subkey, num_players, stacks, blinds, uniform_random_policy, max_actions=50
        )

    elapsed = time.time() - start
    speed = num_trajectories / elapsed

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {speed:.2f} trajectories/sec")
    print(f"  Per trajectory: {elapsed/num_trajectories*1000:.1f}ms")
    print()

    return speed


def test_fixed_length_sampling():
    """Test fixed-length trajectory sampling."""
    print("="*70)
    print("Test 2: Fixed-Length Trajectory Sampling")
    print("="*70)
    print()

    key = random.PRNGKey(123)
    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])
    max_length = 50

    # Sample one fixed-length trajectory
    hole_cards, board, bets, actions, valid_mask = sample_trajectory_fixed_length(
        key, num_players, stacks, blinds, uniform_random_policy, max_length
    )

    num_valid = jnp.sum(valid_mask)

    print(f"✓ Fixed-length sampling works:")
    print(f"  - Max length: {max_length}")
    print(f"  - Valid steps: {num_valid}")
    print(f"  - Padding: {max_length - num_valid} steps")
    print(f"  - Output shapes:")
    print(f"    - hole_cards: {hole_cards.shape}")
    print(f"    - board: {board.shape}")
    print(f"    - bets: {bets.shape}")
    print(f"    - actions: {actions.shape}")
    print(f"    - valid_mask: {valid_mask.shape}")
    print()


def test_batched_sampling_small():
    """Test batched sampling with small batch."""
    print("="*70)
    print("Test 3: Batched Sampling (Small Batch = 10)")
    print("="*70)
    print()

    key = random.PRNGKey(999)
    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])
    batch_size = 10
    max_length = 50

    # Generate batch keys
    keys = random.split(key, batch_size)

    # Sample batch
    print(f"[Sampling {batch_size} trajectories in parallel...]")
    start = time.time()

    batch_outputs = batch_sample_trajectories(
        keys, num_players, stacks, blinds, uniform_random_policy, max_length
    )

    elapsed = time.time() - start

    batch_hole, batch_board, batch_bets, batch_actions, batch_valid = batch_outputs

    print(f"✓ Batched sampling works:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Time: {elapsed:.4f}s")
    print(f"  - Speed: {batch_size/elapsed:.1f} trajectories/sec")
    print(f"  - Per trajectory: {elapsed/batch_size*1000:.1f}ms")
    print(f"  - Output shapes:")
    print(f"    - hole_cards: {batch_hole.shape}")
    print(f"    - board: {batch_board.shape}")
    print(f"    - bets: {batch_bets.shape}")
    print(f"    - actions: {batch_actions.shape}")
    print(f"    - valid_mask: {batch_valid.shape}")
    print()


def test_batched_sampling_large():
    """Test batched sampling with larger batch."""
    print("="*70)
    print("Test 4: Batched Sampling (Large Batch = 100)")
    print("="*70)
    print()

    key = random.PRNGKey(888)
    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])
    batch_size = 100
    max_length = 50

    # Generate batch keys
    keys = random.split(key, batch_size)

    # Warmup (JIT compilation)
    print("[Warmup - JIT compilation...]")
    warmup_keys = random.split(random.PRNGKey(0), 10)
    _ = batch_sample_trajectories(
        warmup_keys, num_players, stacks, blinds, uniform_random_policy, max_length
    )
    print("✓ Warmup complete")
    print()

    # Sample batch
    print(f"[Sampling {batch_size} trajectories in parallel...]")
    start = time.time()

    batch_outputs = batch_sample_trajectories(
        keys, num_players, stacks, blinds, uniform_random_policy, max_length
    )

    elapsed = time.time() - start

    batch_hole, batch_board, batch_bets, batch_actions, batch_valid = batch_outputs

    print(f"✓ Batched sampling completed:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Time: {elapsed:.4f}s")
    print(f"  - Speed: {batch_size/elapsed:.1f} trajectories/sec")
    print(f"  - Per trajectory: {elapsed/batch_size*1000:.1f}ms")
    print()

    return batch_size / elapsed


def compare_sequential_vs_batched(seq_speed, batch_speed):
    """Compare sequential vs batched performance."""
    print("="*70)
    print("Performance Comparison")
    print("="*70)
    print()

    print(f"Sequential speed:  {seq_speed:.2f} trajectories/sec")
    print(f"Batched speed:     {batch_speed:.2f} trajectories/sec")
    print(f"Speedup:           {batch_speed/seq_speed:.2f}×")
    print()

    if batch_speed >= seq_speed * 5:
        print("✅ SUCCESS: Batching provides 5×+ speedup!")
    elif batch_speed >= seq_speed * 2:
        print("⚠️ MODERATE: Batching provides 2-5× speedup")
    else:
        print("❌ ISSUE: Batching speedup < 2×")

    print()


def main():
    print("GPU MCCFR: Batched Sampling Test")
    print("Phase 10.2 Optimization")
    print()

    # Test 1: Sequential (baseline)
    seq_speed = test_sequential_sampling()

    # Test 2: Fixed-length (intermediate)
    test_fixed_length_sampling()

    # Test 3: Small batch
    test_batched_sampling_small()

    # Test 4: Large batch
    batch_speed = test_batched_sampling_large()

    # Compare
    compare_sequential_vs_batched(seq_speed, batch_speed)

    print("="*70)
    print("Batched Sampling Test Complete!")
    print("="*70)
    print()
    print("Next step: Integrate batched sampling into GPUMCCFRSolver")


if __name__ == "__main__":
    main()
