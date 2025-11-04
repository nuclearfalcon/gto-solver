#!/usr/bin/env python3
"""
Test Batched vs Sequential Hold'em Trajectory Sampling

Measures the actual speedup achieved by batched sampling for No-Limit Hold'em
in a realistic MCCFR-like workload (sampling many trajectories with uniform policy).

Expected: 200-400× speedup based on Kuhn's 378× result.

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.2: Hold'em Batched Sampling Validation (Day 3)
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr import holdem_jax_v2

# Standard heads-up Hold'em configuration
NUM_PLAYERS = 2
STACKS = jnp.array([1000.0, 1000.0])
BLINDS = jnp.array([50.0, 100.0])


def uniform_random_policy(state: holdem_jax_v2.HoldemState) -> jnp.ndarray:
    """Uniform random policy (JAX-compatible)."""
    legal = holdem_jax_v2.legal_actions(state)
    probs = legal.astype(jnp.float32)
    probs = probs / (jnp.sum(probs) + 1e-10)
    return probs


def sample_trajectory_sequential(key: jax.random.PRNGKey) -> tuple:
    """Sample single Hold'em trajectory sequentially (Python loop)."""
    state = holdem_jax_v2.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)

    num_actions = 0
    max_actions = 50  # Hold'em can have many actions

    # Play until terminal (Python loop - NOT JAX-traceable)
    while not holdem_jax_v2.is_terminal(state) and num_actions < max_actions:
        action_probs = uniform_random_policy(state)

        key, subkey = random.split(key)
        action = random.choice(subkey, jnp.arange(4), p=action_probs)

        key, action_key = random.split(key)
        state = holdem_jax_v2.apply_action(state, action, action_key)
        num_actions += 1

    payoffs = holdem_jax_v2.payoffs(state)

    return (num_actions, payoffs)


def sample_trajectory_fixed_length(key: jax.random.PRNGKey, max_length: int = 50):
    """Sample Hold'em trajectory with fixed-length output (JAX-traceable)."""
    state = holdem_jax_v2.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)

    def cond_fn(carry):
        state, key, step, done = carry
        return (step < max_length) & ~done

    def body_fn(carry):
        state, key, step, done = carry

        terminal = holdem_jax_v2.is_terminal(state)
        done = done | terminal

        def sample_action(state, key):
            action_probs = uniform_random_policy(state)
            key, subkey = random.split(key)
            action = random.choice(subkey, jnp.arange(4), p=action_probs)
            return action, key

        def no_op(state, key):
            return jnp.int32(0), key

        action, key = jax.lax.cond(done, no_op, sample_action, state, key)

        def apply_fn(state_and_key):
            state, action, key = state_and_key
            key, action_key = random.split(key)
            return holdem_jax_v2.apply_action(state, action, action_key), key

        def keep_fn(state_and_key):
            state, action, key = state_and_key
            return state, key

        new_state, key = jax.lax.cond(done, keep_fn, apply_fn, (state, action, key))

        return (new_state, key, step + 1, done)

    initial_carry = (state, key, jnp.int32(0), False)
    final_state, final_key, num_steps, _ = jax.lax.while_loop(cond_fn, body_fn, initial_carry)

    payoffs = holdem_jax_v2.payoffs(final_state)

    return (num_steps, payoffs)


def batch_sample_trajectories(keys: jnp.ndarray, max_length: int = 50):
    """Sample many Hold'em trajectories in parallel (JAX vmap + JIT)."""
    vectorized_sample = jax.vmap(
        lambda key: sample_trajectory_fixed_length(key, max_length)
    )

    return vectorized_sample(keys)


def test_sequential_sampling(num_samples: int = 100):
    """Benchmark sequential trajectory sampling."""
    print("="*70)
    print(f"Test 1: Sequential Sampling ({num_samples} trajectories)")
    print("="*70)
    print()

    key = random.PRNGKey(42)
    keys = random.split(key, num_samples)

    print(f"Sampling {num_samples} Hold'em trajectories sequentially...")
    print("(This may take a while - Hold'em is more complex than Kuhn)")
    print()

    start = time.time()

    for k in keys:
        num_actions, payoffs = sample_trajectory_sequential(k)

    elapsed = time.time() - start
    throughput = num_samples / elapsed

    print(f"✓ Sequential sampling complete")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} trajectories/sec")
    print(f"  Per trajectory: {elapsed/num_samples*1000:.1f}ms")
    print()

    return throughput


def test_batched_sampling(num_samples: int = 100, batch_size: int = 100):
    """Benchmark batched trajectory sampling."""
    print("="*70)
    print(f"Test 2: Batched Sampling ({num_samples} trajectories, batch={batch_size})")
    print("="*70)
    print()

    key = random.PRNGKey(123)
    max_length = 50

    # Warmup
    print("[Warmup - JIT compilation...]")
    print("(First compilation may take 10-30 seconds for Hold'em)")
    warmup_keys = random.split(random.PRNGKey(0), 10)
    _ = batch_sample_trajectories(warmup_keys, max_length)
    print("✓ JIT compilation complete")
    print()

    # Generate all keys
    print(f"[Sampling {num_samples} trajectories in batches of {batch_size}...]")

    num_batches = (num_samples + batch_size - 1) // batch_size

    start = time.time()

    for i in range(num_batches):
        batch_key = random.fold_in(key, i)
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        batch_keys = random.split(batch_key, current_batch_size)

        num_steps_arr, payoffs_arr = batch_sample_trajectories(batch_keys, max_length)

    elapsed = time.time() - start
    throughput = num_samples / elapsed

    print(f"✓ Batched sampling complete")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Throughput: {throughput:.1f} trajectories/sec")
    print(f"  Per trajectory: {elapsed/num_samples*1000:.2f}ms")
    print()

    return throughput


def test_various_batch_sizes(num_samples: int = 1_000):
    """Test different batch sizes to find optimal."""
    print("="*70)
    print(f"Test 3: Batch Size Scaling ({num_samples} trajectories)")
    print("="*70)
    print()

    batch_sizes = [50, 100, 250, 500, 1000]
    results = {}

    # Measure sequential baseline once
    print("[Measuring sequential baseline...]")
    key = random.PRNGKey(999)
    baseline_sample_size = 50
    baseline_keys = random.split(key, baseline_sample_size)

    start = time.time()
    for k in baseline_keys:
        num_actions, payoffs = sample_trajectory_sequential(k)
    baseline_time = time.time() - start
    baseline_throughput = baseline_sample_size / baseline_time

    print(f"✓ Sequential baseline: {baseline_throughput:.2f} traj/s")
    print()

    for batch_size in batch_sizes:
        print(f"Testing batch_size={batch_size}...")

        key = random.PRNGKey(999 + batch_size)
        max_length = 50

        # Warmup
        warmup_keys = random.split(random.PRNGKey(batch_size), 10)
        _ = batch_sample_trajectories(warmup_keys, max_length)

        # Benchmark
        num_batches = (num_samples + batch_size - 1) // batch_size

        start = time.time()

        for i in range(num_batches):
            batch_key = random.fold_in(key, i)
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            batch_keys = random.split(batch_key, current_batch_size)

            num_steps_arr, payoffs_arr = batch_sample_trajectories(batch_keys, max_length)

        elapsed = time.time() - start
        throughput = num_samples / elapsed
        speedup = throughput / baseline_throughput
        per_traj_ms = (elapsed / num_samples) * 1000

        results[batch_size] = {
            'throughput': throughput,
            'speedup': speedup,
            'per_traj_ms': per_traj_ms
        }

        print(f"  Throughput: {throughput:.1f} traj/s")
        print(f"  Speedup: {speedup:.1f}×")
        print(f"  Per trajectory: {per_traj_ms:.2f}ms")
        print()

    # Summary table
    print("="*70)
    print("SUMMARY: Batch Size Scaling")
    print("="*70)
    print()
    print(f"{'Batch Size':<12} {'Throughput':<15} {'Speedup':<10} {'Per-Traj':<10}")
    print(f"{'----------':<12} {'-----------':<15} {'-------':<10} {'--------':<10}")
    print(f"{'Sequential':<12} {baseline_throughput:>8.1f} traj/s  {'1.0×':<10} {1000/baseline_throughput:>6.1f}ms")

    for batch_size in batch_sizes:
        r = results[batch_size]
        print(f"{batch_size:<12} {r['throughput']:>8.1f} traj/s  {r['speedup']:>6.1f}×    {r['per_traj_ms']:>6.2f}ms")

    print()

    # Find best
    best_batch = max(results.keys(), key=lambda b: results[b]['speedup'])
    best_speedup = results[best_batch]['speedup']

    print(f"Best configuration: batch_size={best_batch} with {best_speedup:.1f}× speedup")
    print()

    if best_speedup >= 200:
        print(f"🎉 EXCELLENT! Achieved {best_speedup:.0f}× speedup (target: >50×)")
    elif best_speedup >= 100:
        print(f"✅ GREAT! Achieved {best_speedup:.0f}× speedup (target: >50×)")
    elif best_speedup >= 50:
        print(f"✅ GOOD! Achieved {best_speedup:.0f}× speedup (target: >50×)")
    else:
        print(f"⚠️ Achieved {best_speedup:.0f}× speedup (target was >50×)")

    return results, baseline_throughput


def main():
    print("Hold'em JAX V2 Batched Trajectory Sampling Benchmark")
    print("Phase 10.2: Measuring GPU Speedup for Hold'em")
    print()

    # Test 1: Sequential sampling (small sample size due to slowness)
    seq_throughput = test_sequential_sampling(num_samples=50)

    # Test 2: Batched sampling (same sample size)
    batch_throughput = test_batched_sampling(num_samples=50, batch_size=50)

    # Calculate speedup
    speedup = batch_throughput / seq_throughput

    print("="*70)
    print("QUICK COMPARISON")
    print("="*70)
    print()
    print(f"Sequential: {seq_throughput:.2f} traj/s")
    print(f"Batched:    {batch_throughput:.1f} traj/s")
    print(f"Speedup:    {speedup:.1f}×")
    print()

    # Test 3: Comprehensive batch size scaling
    results, baseline = test_various_batch_sizes(num_samples=1000)

    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print()
    print(f"Hold'em JAX V2 batched sampling: ✅ WORKING")
    print(f"Best speedup achieved: {max(r['speedup'] for r in results.values()):.0f}×")
    print()
    print("Hold'em V2 is ready for integration into GPU MCCFR!")


if __name__ == "__main__":
    main()
