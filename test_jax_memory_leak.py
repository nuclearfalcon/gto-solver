"""
Minimal test to isolate JAX memory leak source.

This test checks if repeated calls to vmap with dynamic_slice cause memory accumulation.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
import psutil
import gc
import time


def process_flat_array(flat_state, player_id, num_players=2):
    """Simulate the bucketing operation with dynamic slicing."""
    # Extract player's data using dynamic_slice (like in bucketing)
    player_start = player_id * 2
    player_data = lax.dynamic_slice(flat_state, (player_start,), (2,))

    # Some computation
    result = jnp.sum(player_data) * 1.5
    return result.astype(jnp.int32)


# Create cached JIT function (simulating our bucketing function)
@jax.jit
def batch_process(states_batch, player_id):
    """Process batch of states."""
    process_fn = partial(process_flat_array, player_id=player_id, num_players=2)
    return jax.vmap(process_fn)(states_batch)


def main():
    """Run repeated iterations and monitor memory."""
    process = psutil.Process()

    # Initial memory
    gc.collect()
    time.sleep(0.5)
    initial_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Initial memory: {initial_mem:.1f} MB\n")

    # Simulate 50 iterations like the solver
    num_iterations = 50
    batch_size = 10000  # Same scale as solver (200 trajectories × 50 states)
    state_size = 73

    for i in range(num_iterations):
        # Generate random states (like trajectory sampling)
        key = jax.random.PRNGKey(i)
        states = jax.random.uniform(key, (batch_size, state_size))

        # Process batch (like bucketing)
        results = batch_process(states, player_id=0)

        # Force computation
        _ = results.block_until_ready()

        # Check memory every 10 iterations
        if (i + 1) % 10 == 0:
            gc.collect()
            current_mem = process.memory_info().rss / (1024 * 1024)
            growth = current_mem - initial_mem
            print(f"Iteration {i+1:2d}: {current_mem:7.1f} MB (+{growth:6.1f} MB)")

    # Final memory
    gc.collect()
    time.sleep(0.5)
    final_mem = process.memory_info().rss / (1024 * 1024)
    total_growth = final_mem - initial_mem
    per_iter = total_growth / num_iterations

    print(f"\nFinal memory:   {final_mem:.1f} MB")
    print(f"Total growth:   {total_growth:.1f} MB")
    print(f"Per iteration:  {per_iter:.1f} MB/iter")

    if total_growth < 100:
        print("\n✅ No significant memory leak!")
    else:
        print(f"\n❌ Memory leak detected: {per_iter:.1f} MB/iteration")


if __name__ == "__main__":
    print("="*70)
    print("JAX Memory Leak Isolation Test")
    print("Testing if vmap + dynamic_slice causes memory accumulation")
    print("="*70)
    print()

    main()
