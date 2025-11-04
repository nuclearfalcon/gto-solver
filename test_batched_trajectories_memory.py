"""
Isolation test for batched trajectory sampling memory leak.

This test checks if _sample_batched_trajectories() causes the ~38-40 MB/iter
memory leak observed in the full solver.

Run with:
    source ~/open_spiel/venv/bin/activate
    python test_batched_trajectories_memory.py
"""

import jax
import jax.numpy as jnp
import psutil
import gc
import time
from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver


def test_batched_trajectories_memory():
    """Test memory growth from batched trajectory sampling."""
    print("\n" + "="*70)
    print("Batched Trajectory Sampling Memory Test")
    print("="*70)
    print("\nTesting: _sample_batched_trajectories() with batch_size=200")
    print("Expected: Should show ~38-40 MB/iter (reproduce actual solver leak)\n")

    # Create solver with actual parameters
    num_players = 2
    num_buckets = 5000
    num_hand_buckets = 100
    num_pot_buckets = 5
    num_actions = 4
    batch_size = 200  # Actual batch size used in solver

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

        # Sample batch of trajectories (this is the full pipeline we suspect leaks)
        batch_results = solver._sample_batched_trajectories(
            key=key,
            batch_size=batch_size,
            num_players=num_players,
            stacks=stacks,
            blinds=blinds,
            max_length=50
        )

        # Force computation
        states_batch, actions_batch, players_batch, valid_masks, payoffs_batch = batch_results
        _ = states_batch.block_until_ready()

        # Explicit cleanup (mimicking solver)
        del batch_results, states_batch, actions_batch, players_batch, valid_masks, payoffs_batch
        gc.collect()

        # Check memory every 10 iterations
        if (i + 1) % 10 == 0:
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

    if per_iter > 30:
        print(f"  Status:        ❌ LEAK REPRODUCED: {per_iter:.2f} MB/iter")
        print(f"  Match:         This matches the actual solver leak!")
    elif total_growth < 100:
        print(f"  Status:        ✅ No significant memory leak!")
    else:
        print(f"  Status:        ⚠️  Moderate leak: {per_iter:.2f} MB/iter")

    return total_growth, per_iter


if __name__ == "__main__":
    print("="*70)
    print("Batched Trajectory Sampling Memory Leak Test")
    print("This should reproduce the ~38-40 MB/iter leak from actual solver")
    print("="*70)

    growth, per_iter = test_batched_trajectories_memory()

    print("\n" + "="*70)
    if per_iter > 30:
        print(f"✅ LEAK REPRODUCED: {per_iter:.2f} MB/iter")
        print("\nThis confirms _sample_batched_trajectories() is the leak source.")
        print("Root cause: NamedTuples in vmap + while_loop")
    elif growth < 100:
        print("❓ UNEXPECTED: No leak found in isolation")
        print("Leak may be in regret table updates or other component")
    else:
        print(f"⚠️  PARTIAL LEAK: {per_iter:.2f} MB/iter")
    print("="*70)
