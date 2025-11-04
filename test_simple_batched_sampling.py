#!/usr/bin/env python3
"""
Test simple batched sampling using Python loops (not full JAX vmap).

This tests if we can at least parallelize the MCCFR iteration logic
by sampling multiple trajectories and batching the regret updates.

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.2: Simplified Batching Test
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr import kuhn_jax
from matrix_cfr.trajectory_sampler import (
    sample_trajectory,
    uniform_random_policy
)


def sample_batch_sequential(keys, num_players, stacks, blinds, policy_fn, max_actions):
    """
    Sample multiple trajectories sequentially (baseline).

    Args:
        keys: List of random keys (batch_size,)
        num_players: Number of players
        stacks: Stack sizes
        blinds: Blind amounts
        policy_fn: Policy function
        max_actions: Max actions per trajectory

    Returns:
        List of (states, actions, players, payoffs) tuples
    """
    results = []
    for key in keys:
        result = sample_trajectory(
            key, num_players, stacks, blinds, policy_fn, max_actions
        )
        results.append(result)
    return results


def test_sequential_sampling():
    """Test sequential sampling (baseline)."""
    print("="*70)
    print("Test 1: Sequential Sampling")
    print("="*70)
    print()

    key = random.PRNGKey(42)
    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])
    batch_size = 100

    # Generate keys
    keys = random.split(key, batch_size)

    # Sample batch
    print(f"[Sampling {batch_size} trajectories sequentially...]")
    start = time.time()

    results = sample_batch_sequential(
        keys, num_players, stacks, blinds,
        uniform_random_policy, max_actions=50
    )

    elapsed = time.time() - start
    throughput = batch_size / elapsed

    print(f"✓ Sequential sampling:")
    print(f"  Batch size: {batch_size}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} trajectories/sec")
    print(f"  Per trajectory: {elapsed/batch_size*1000:.1f}ms")
    print()

    # Show sample trajectory info
    states, actions, players, payoffs = results[0]
    print(f"Sample trajectory:")
    print(f"  Decision points: {len(states)}")
    print(f"  Actions: {actions}")
    print(f"  Players: {players}")
    print(f"  Payoffs: {payoffs}")
    print()

    return throughput


def test_parallel_processing():
    """
    Test if we can at least parallelize regret updates.

    Idea: Even if we can't fully vmap trajectory sampling,
    we can still parallelize the regret update computation.
    """
    print("="*70)
    print("Test 2: Parallel Regret Updates (Conceptual)")
    print("="*70)
    print()

    # Sample multiple trajectories
    key = random.PRNGKey(123)
    num_players = 2
    stacks = jnp.array([100.0, 100.0])
    blinds = jnp.array([1.0, 1.0])
    batch_size = 100

    keys = random.split(key, batch_size)
    results = sample_batch_sequential(
        keys, num_players, stacks, blinds,
        uniform_random_policy, max_actions=50
    )

    # Count total updates across all trajectories
    total_updates = 0
    for states, actions, players, payoffs in results:
        total_updates += len(states)

    print(f"✓ Batched {batch_size} trajectories:")
    print(f"  Total decision points: {total_updates}")
    print(f"  Average per trajectory: {total_updates/batch_size:.1f}")
    print()
    print("Next optimization: Vectorize regret update computation")
    print("  - Collect all (infoset, action, regret) tuples")
    print("  - Batch update regret table using JAX operations")
    print("  - Expected speedup: 2-5× over sequential updates")
    print()


def main():
    print("GPU MCCFR: Simple Batched Sampling Test")
    print("Phase 10.2 Optimization")
    print()

    # Test 1: Sequential baseline
    seq_speed = test_sequential_sampling()

    # Test 2: Parallel processing potential
    test_parallel_processing()

    print("="*70)
    print("Simple Batching Test Complete!")
    print("="*70)
    print()
    print(f"Sequential throughput: {seq_speed:.2f} trajectories/sec")
    print()
    print("Key findings:")
    print("1. Full JAX vmap/scan requires rewriting apply_action() to be JAX-native")
    print("2. Alternative: Parallelize regret updates, not trajectory sampling")
    print("3. Expected speedup from batched updates: 2-5×")
    print()
    print("Recommendation: Focus on batched regret updates first")


if __name__ == "__main__":
    main()
