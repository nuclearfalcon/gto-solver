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


def test_scenario(batch_size, state_size, num_iterations, scenario_name):
    """Test a specific scenario and report results."""
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"  Batch size:    {batch_size:,} states")
    print(f"  State size:    {state_size} values")
    print(f"  Iterations:    {num_iterations}")
    print(f"  Memory scale:  {(batch_size * state_size * 4) / (1024*1024):.1f} MB per batch")
    print(f"{'='*70}\n")

    process = psutil.Process()

    # Initial memory
    gc.collect()
    time.sleep(0.5)
    initial_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Initial memory: {initial_mem:.1f} MB\n")

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
            per_iter = growth / (i + 1)
            print(f"Iteration {i+1:2d}: {current_mem:7.1f} MB (+{growth:6.1f} MB total, {per_iter:5.2f} MB/iter)")

    # Final memory
    gc.collect()
    time.sleep(0.5)
    final_mem = process.memory_info().rss / (1024 * 1024)
    total_growth = final_mem - initial_mem
    per_iter = total_growth / num_iterations

    print(f"\nResults for {scenario_name}:")
    print(f"  Initial:       {initial_mem:.1f} MB")
    print(f"  Final:         {final_mem:.1f} MB")
    print(f"  Total growth:  {total_growth:.1f} MB")
    print(f"  Per iteration: {per_iter:.2f} MB/iter")

    if total_growth < 100:
        print(f"  Status:        ✅ No significant memory leak!")
    else:
        print(f"  Status:        ❌ Memory leak: {per_iter:.2f} MB/iter")

    return total_growth, per_iter


def main():
    """Run multiple test scenarios with different game sizes."""

    print("\n" + "="*70)
    print("Testing Multiple Game Sizes")
    print("="*70)

    scenarios = [
        # (batch_size, state_size, iterations, name)
        (10000, 73, 50, "Minimal (10K states, 73 dims)"),
        (10000, 200, 50, "Medium (10K states, 200 dims)"),
        (20000, 73, 50, "Larger batch (20K states, 73 dims)"),
        (10000, 500, 30, "Wide states (10K states, 500 dims)"),
        (5000, 1000, 30, "Very wide (5K states, 1000 dims)"),
        (1000, 10000, 30, "Huge (1K states, 10K dims)"),
        (500, 50000, 30, "MASSIVE (500 states, 50K dims)"),
    ]

    results = []
    for batch_size, state_size, iterations, name in scenarios:
        growth, per_iter = test_scenario(batch_size, state_size, iterations, name)
        results.append((name, growth, per_iter))

        # Clear JAX cache between scenarios
        jax.clear_caches()
        gc.collect()
        time.sleep(1)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL SCENARIOS")
    print("="*70)
    print(f"{'Scenario':<40} {'Total Growth':>12} {'Per Iter':>12}")
    print("-"*70)
    for name, growth, per_iter in results:
        status = "✅" if growth < 100 else "❌"
        print(f"{name:<40} {growth:>10.1f} MB {per_iter:>10.2f} MB {status}")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("JAX Memory Leak Isolation Test")
    print("Testing if vmap + dynamic_slice causes memory accumulation")
    print("="*70)
    print()

    main()
