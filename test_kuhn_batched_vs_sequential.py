#!/usr/bin/env python3
"""
Test Batched vs Sequential Trajectory Sampling - Direct Comparison

Measures the actual speedup achieved by batched sampling in a realistic
MCCFR-like workload (sampling many trajectories with uniform policy).

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.2: Batched Sampling Validation (Day 2)
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr import kuhn_jax_v2


def uniform_random_policy(state: kuhn_jax_v2.KuhnState) -> jnp.ndarray:
    """Uniform random policy (JAX-compatible)."""
    legal = kuhn_jax_v2.legal_actions(state)
    probs = legal.astype(jnp.float32)
    probs = probs / (jnp.sum(probs) + 1e-10)
    return probs


def sample_trajectory_sequential(key: jax.random.PRNGKey) -> tuple:
    """Sample single trajectory sequentially (Python loop)."""
    state = kuhn_jax_v2.deal_initial_state(key)

    num_actions = 0
    max_actions = 10

    # Play until terminal (Python loop - NOT JAX-traceable)
    while not kuhn_jax_v2.is_terminal(state) and num_actions < max_actions:
        action_probs = uniform_random_policy(state)

        key, subkey = random.split(key)
        action = random.choice(subkey, jnp.arange(2), p=action_probs)

        state = kuhn_jax_v2.apply_action(state, action)
        num_actions += 1

    payoffs = kuhn_jax_v2.payoffs(state)

    return (num_actions, payoffs)


def sample_trajectory_fixed_length(key: jax.random.PRNGKey, max_length: int = 10):
    """Sample trajectory with fixed-length output (JAX-traceable)."""
    state = kuhn_jax_v2.deal_initial_state(key)

    def cond_fn(carry):
        state, key, step, done = carry
        return (step < max_length) & ~done

    def body_fn(carry):
        state, key, step, done = carry

        terminal = kuhn_jax_v2.is_terminal(state)
        done = done | terminal

        def sample_action(state, key):
            action_probs = uniform_random_policy(state)
            key, subkey = random.split(key)
            action = random.choice(subkey, jnp.arange(2), p=action_probs)
            return action, key

        def no_op(state, key):
            return jnp.int32(0), key

        action, key = jax.lax.cond(done, no_op, sample_action, state, key)

        def apply_fn(state, action):
            return kuhn_jax_v2.apply_action(state, action)

        def keep_fn(state, action):
            return state

        new_state = jax.lax.cond(done, keep_fn, apply_fn, state, action)

        return (new_state, key, step + 1, done)

    initial_carry = (state, key, jnp.int32(0), False)
    final_state, final_key, num_steps, _ = jax.lax.while_loop(cond_fn, body_fn, initial_carry)

    payoffs = kuhn_jax_v2.payoffs(final_state)

    return (num_steps, payoffs)


def batch_sample_trajectories(keys: jnp.ndarray, max_length: int = 10):
    """Sample many trajectories in parallel (JAX vmap + JIT)."""
    vectorized_sample = jax.vmap(
        lambda key: sample_trajectory_fixed_length(key, max_length)
    )

    return vectorized_sample(keys)


def test_sequential_sampling(num_samples: int = 1000):
    """Benchmark sequential trajectory sampling."""
    print("="*70)
    print(f"Test 1: Sequential Sampling ({num_samples} trajectories)")
    print("="*70)
    print()

    key = random.PRNGKey(42)
    keys = random.split(key, num_samples)

    print(f"Sampling {num_samples} trajectories sequentially...")

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


def test_batched_sampling(num_samples: int = 1000, batch_size: int = 1000):
    """Benchmark batched trajectory sampling."""
    print("="*70)
    print(f"Test 2: Batched Sampling ({num_samples} trajectories, batch={batch_size})")
    print("="*70)
    print()

    key = random.PRNGKey(123)
    max_length = 10

    # Warmup
    print("[Warmup - JIT compilation...]")
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


def test_various_batch_sizes(num_samples: int = 10_000):
    """Test different batch sizes to find optimal."""
    print("="*70)
    print(f"Test 3: Batch Size Scaling ({num_samples} trajectories)")
    print("="*70)
    print()

    batch_sizes = [100, 500, 1000, 2000, 5000]
    results = {}

    for batch_size in batch_sizes:
        print(f"Testing batch_size={batch_size}...")

        key = random.PRNGKey(999 + batch_size)
        max_length = 10

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

        results[batch_size] = throughput

        print(f"  Throughput: {throughput:.1f} traj/s")
        print()

    return results


def main():
    print("Kuhn Poker V2: Batched vs Sequential Sampling Benchmark")
    print("Phase 10.2: Direct Speedup Measurement")
    print()

    # Test 1: Sequential baseline (1K samples for fast test)
    seq_throughput = test_sequential_sampling(num_samples=1000)

    # Test 2: Batched (1K samples, batch=1000)
    batch_throughput = test_batched_sampling(num_samples=1000, batch_size=1000)

    # Test 3: Batch size scaling (10K samples)
    batch_results = test_various_batch_sizes(num_samples=10_000)

    # Summary
    print("="*70)
    print("Performance Summary")
    print("="*70)
    print()

    speedup = batch_throughput / seq_throughput

    print(f"Sequential:       {seq_throughput:.2f} traj/sec")
    print(f"Batched (1000):   {batch_throughput:.1f} traj/sec  ({speedup:.1f}× speedup)")
    print()

    print("Batch Size Scaling:")
    for batch_size, throughput in sorted(batch_results.items()):
        speedup_vs_seq = throughput / seq_throughput
        print(f"  Batch {batch_size:5d}: {throughput:7.1f} traj/s  ({speedup_vs_seq:.1f}× speedup)")
    print()

    # Find optimal batch size
    best_batch_size = max(batch_results, key=batch_results.get)
    best_throughput = batch_results[best_batch_size]
    best_speedup = best_throughput / seq_throughput

    print("="*70)
    print("Optimal Configuration")
    print("="*70)
    print()
    print(f"Best batch size: {best_batch_size}")
    print(f"Best throughput: {best_throughput:.1f} traj/s")
    print(f"Best speedup: {best_speedup:.1f}×")
    print()

    # GO/NO-GO decision
    print("="*70)
    print("GO/NO-GO DECISION")
    print("="*70)
    print()

    if best_speedup >= 50:
        print(f"✅ SUCCESS: {best_speedup:.1f}× speedup achieved!")
        print(f"   Target was >50×, we got {best_speedup:.1f}×")
        print()
        print("🚀 RECOMMENDATION: PROCEED with batched sampling integration")
        print("   The JAX-native approach is validated!")
        return True
    elif best_speedup >= 10:
        print(f"⚠️  MODERATE: {best_speedup:.1f}× speedup achieved")
        print(f"   Target was >50×, we got {best_speedup:.1f}×")
        print()
        print("🤔 RECOMMENDATION: Investigate bottlenecks")
        print("   10× is good, but we expected more")
        return False
    else:
        print(f"❌ INSUFFICIENT: Only {best_speedup:.1f}× speedup")
        print(f"   Target was >50×, this is not enough")
        print()
        print("🛑 RECOMMENDATION: ABORT JAX-native rewrite")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print()
        print("Next: Integrate batched sampling into full MCCFR solver!")
    else:
        print()
        print("Need to debug performance or try alternative approaches")
