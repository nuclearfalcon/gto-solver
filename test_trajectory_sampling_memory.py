"""
Isolation test for trajectory sampling memory leak.

This test checks if _sample_trajectory_fixed_length() causes memory accumulation
when run repeatedly. This isolates the trajectory sampling component.

Run with:
    source ~/open_spiel/venv/bin/activate
    python test_trajectory_sampling_memory.py
"""

import jax
import jax.numpy as jnp
import psutil
import gc
import time
from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver


def test_single_trajectory_memory():
    """Test memory growth from single trajectory sampling."""
    print("\n" + "="*70)
    print("Single Trajectory Sampling Memory Test")
    print("="*70)
    print("\nTesting: _sample_trajectory_fixed_length() in isolation")
    print("Expected: Should show ~0.4-0.8 MB/iter if NamedTuple carry leaks\n")

    # Create minimal solver
    num_players = 2
    num_buckets = 5000
    num_hand_buckets = 100
    num_pot_buckets = 5
    num_actions = 4
    batch_size = 1  # Single trajectory

    solver = GPUMCCFRSolver(
        num_players=num_players,
        num_buckets=num_buckets,
        num_hand_buckets=num_hand_buckets,
        num_pot_buckets=num_pot_buckets,
        num_actions=num_actions,
        batch_size=batch_size,
        seed=42
    )

    # Game parameters
    stacks = jnp.array([500.0, 500.0])
    blinds = jnp.array([50.0, 100.0])

    process = psutil.Process()

    # Initial memory
    gc.collect()
    time.sleep(0.5)
    initial_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Initial memory: {initial_mem:.1f} MB\n")

    num_iterations = 50

    for i in range(num_iterations):
        # Generate new key for this iteration
        key = jax.random.PRNGKey(i)

        # Sample single trajectory (this is what we suspect leaks)
        trajectory = solver._sample_trajectory_fixed_length(
            key=key,
            num_players=num_players,
            stacks=stacks,
            blinds=blinds,
            max_length=50
        )

        # Force computation
        states, actions, players, valid_mask, payoffs = trajectory
        _ = states.block_until_ready()

        # Check memory every 10 iterations
        if (i + 1) % 10 == 0:
            gc.collect()
            current_mem = process.memory_info().rss / (1024 * 1024)
            growth = current_mem - initial_mem
            per_iter = growth / (i + 1)
            print(f"Iteration {i+1:2d}: {current_mem:7.1f} MB (+{growth:6.1f} MB total, {per_iter:5.2f} MB/iter)")

    # Final memory
    gc.collect()
    time.sleep(0.5)
    final_mem = process.memory_info().rss / (1024 * 1024)
    total_growth = final_mem - initial_mem
    per_iter = total_growth / num_iterations

    print(f"\nResults:")
    print(f"  Initial:       {initial_mem:.1f} MB")
    print(f"  Final:         {final_mem:.1f} MB")
    print(f"  Total growth:  {total_growth:.1f} MB")
    print(f"  Per iteration: {per_iter:.2f} MB/iter")

    if total_growth < 100:
        print(f"  Status:        ✅ No significant memory leak!")
    else:
        print(f"  Status:        ❌ Memory leak: {per_iter:.2f} MB/iter")

    return total_growth, per_iter


if __name__ == "__main__":
    print("="*70)
    print("Trajectory Sampling Memory Leak Isolation Test")
    print("Testing if _sample_trajectory_fixed_length() leaks memory")
    print("="*70)

    growth, per_iter = test_single_trajectory_memory()

    print("\n" + "="*70)
    if growth < 100:
        print("✅ PASS: Single trajectory sampling is memory-safe")
    else:
        print(f"❌ FAIL: Single trajectory leaks {per_iter:.2f} MB/iter")
        print("\nThis confirms NamedTuple in while_loop is the leak source.")
    print("="*70)
