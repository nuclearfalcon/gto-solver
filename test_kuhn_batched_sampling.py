#!/usr/bin/env python3
"""
Test Batched Trajectory Sampling for Kuhn Poker V2

This is the CRITICAL TEST that determines if the JAX-native rewrite was worth it!

Goal: Achieve >50× speedup through batched trajectory sampling.

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.2: Batched Sampling Proof-of-Concept (Day 1)
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr import kuhn_jax_v2


def uniform_random_policy_kuhn(state: kuhn_jax_v2.KuhnState) -> jnp.ndarray:
    """
    Uniform random policy for Kuhn poker.

    JAX-compatible: No string infosets!

    Returns action probabilities: [pass_prob, bet_prob]
    """
    legal = kuhn_jax_v2.legal_actions(state)
    # Uniform over legal actions
    probs = legal.astype(jnp.float32)
    probs = probs / (jnp.sum(probs) + 1e-10)
    return probs


def sample_trajectory_sequential(key: jax.random.PRNGKey) -> tuple:
    """
    Sample a single Kuhn poker trajectory sequentially.

    Returns:
        (num_actions, terminal_payoffs)
    """
    # Deal initial state
    state = kuhn_jax_v2.deal_initial_state(key)

    num_actions = 0
    max_actions = 10  # Kuhn poker max is 3, but add buffer

    # Play until terminal
    while not kuhn_jax_v2.is_terminal(state) and num_actions < max_actions:
        # Get action probabilities
        action_probs = uniform_random_policy_kuhn(state)

        # Sample action
        key, subkey = random.split(key)
        action = random.choice(subkey, jnp.arange(2), p=action_probs)

        # Apply action
        state = kuhn_jax_v2.apply_action(state, action)

        num_actions += 1

    # Get payoffs
    payoffs = kuhn_jax_v2.payoffs(state)

    return (num_actions, payoffs)


def sample_trajectory_fixed_length(key: jax.random.PRNGKey, max_length: int = 10):
    """
    Sample trajectory with fixed-length output for vmapping.

    Uses jax.lax.while_loop for efficient looping.

    Returns:
        (actions_taken, valid_mask, terminal_payoffs)
    """
    # Deal initial state
    state = kuhn_jax_v2.deal_initial_state(key)

    def cond_fn(carry):
        """Continue while not terminal and under max length."""
        state, key, step, done = carry
        return (step < max_length) & ~done

    def body_fn(carry):
        """Sample one action."""
        state, key, step, done = carry

        # Check if already terminal
        terminal = kuhn_jax_v2.is_terminal(state)
        done = done | terminal

        # Get action (or no-op if done)
        def sample_action(state, key):
            action_probs = uniform_random_policy_kuhn(state)
            key, subkey = random.split(key)
            action = random.choice(subkey, jnp.arange(2), p=action_probs)
            return action, key

        def no_op(state, key):
            return jnp.int32(0), key

        action, key = jax.lax.cond(done, no_op, sample_action, state, key)

        # Apply action (or keep state if done)
        def apply_fn(state, action):
            return kuhn_jax_v2.apply_action(state, action)

        def keep_fn(state, action):
            return state

        new_state = jax.lax.cond(done, keep_fn, apply_fn, state, action)

        return (new_state, key, step + 1, done)

    # Run loop
    initial_carry = (state, key, jnp.int32(0), False)
    final_state, final_key, num_steps, _ = jax.lax.while_loop(cond_fn, body_fn, initial_carry)

    # Get terminal payoffs
    payoffs = kuhn_jax_v2.payoffs(final_state)

    return (num_steps, payoffs)


def batch_sample_trajectories(keys: jnp.ndarray, max_length: int = 10):
    """
    Sample many trajectories in parallel using vmap.

    THIS IS THE KEY FUNCTION FOR GPU ACCELERATION!

    Args:
        keys: Array of random keys, shape (batch_size, 2)
        max_length: Max actions per trajectory

    Returns:
        (num_steps_array, payoffs_array)
    """
    # Vectorize over batch dimension
    vectorized_sample = jax.vmap(
        lambda key: sample_trajectory_fixed_length(key, max_length)
    )

    return vectorized_sample(keys)


def test_sequential_baseline():
    """Test sequential sampling (baseline)."""
    print("="*70)
    print("Test 1: Sequential Sampling (Baseline)")
    print("="*70)
    print()

    key = random.PRNGKey(42)
    batch_size = 100

    # Sample 100 trajectories sequentially
    print(f"Sampling {batch_size} trajectories sequentially...")

    keys = random.split(key, batch_size)

    start = time.time()

    for k in keys:
        num_actions, payoffs = sample_trajectory_sequential(k)

    elapsed = time.time() - start
    throughput = batch_size / elapsed

    print(f"✓ Sequential sampling:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} trajectories/sec")
    print(f"  Per trajectory: {elapsed/batch_size*1000:.1f}ms")
    print()

    return throughput


def test_batched_sampling():
    """Test batched sampling with vmap."""
    print("="*70)
    print("Test 2: Batched Sampling (vmap + JIT)")
    print("="*70)
    print()

    key = random.PRNGKey(123)
    batch_size = 100
    max_length = 10

    # Generate keys
    keys = random.split(key, batch_size)

    # Warmup (JIT compilation)
    print("[Warmup - JIT compilation...]")
    warmup_keys = random.split(random.PRNGKey(0), 10)
    _ = batch_sample_trajectories(warmup_keys, max_length)
    print("✓ JIT compilation complete")
    print()

    # Benchmark
    print(f"[Sampling {batch_size} trajectories in parallel...]")
    start = time.time()

    num_steps_arr, payoffs_arr = batch_sample_trajectories(keys, max_length)

    elapsed = time.time() - start
    throughput = batch_size / elapsed

    print(f"✓ Batched sampling:")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Throughput: {throughput:.1f} trajectories/sec")
    print(f"  Per trajectory: {elapsed/batch_size*1000:.2f}ms")
    print()
    print(f"Sample results:")
    print(f"  Num steps: {num_steps_arr[:5]}")
    print(f"  Payoffs P0: {payoffs_arr[:5, 0]}")
    print()

    return throughput


def test_large_batch():
    """Test with larger batch size."""
    print("="*70)
    print("Test 3: Large Batch (1000 trajectories)")
    print("="*70)
    print()

    key = random.PRNGKey(999)
    batch_size = 1000
    max_length = 10

    keys = random.split(key, batch_size)

    # Warmup
    print("[Warmup...]")
    warmup_keys = random.split(random.PRNGKey(1), 10)
    _ = batch_sample_trajectories(warmup_keys, max_length)
    print("✓ Done")
    print()

    # Benchmark
    print(f"[Sampling {batch_size} trajectories...]")
    start = time.time()

    num_steps_arr, payoffs_arr = batch_sample_trajectories(keys, max_length)

    elapsed = time.time() - start
    throughput = batch_size / elapsed

    print(f"✓ Large batch complete:")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Throughput: {throughput:.1f} trajectories/sec")
    print(f"  Per trajectory: {elapsed/batch_size*1000:.2f}ms")
    print()

    return throughput


def main():
    print("Kuhn Poker V2: Batched Sampling Benchmark")
    print("Phase 10.2 Proof-of-Concept")
    print()

    # Test 1: Sequential baseline
    seq_throughput = test_sequential_baseline()

    # Test 2: Batched (100)
    batch_throughput_100 = test_batched_sampling()

    # Test 3: Batched (1000)
    batch_throughput_1000 = test_large_batch()

    # Compare
    print("="*70)
    print("Performance Summary")
    print("="*70)
    print()

    speedup_100 = batch_throughput_100 / seq_throughput
    speedup_1000 = batch_throughput_1000 / seq_throughput

    print(f"Sequential:       {seq_throughput:.2f} traj/sec")
    print(f"Batched (100):    {batch_throughput_100:.1f} traj/sec  ({speedup_100:.1f}× speedup)")
    print(f"Batched (1000):   {batch_throughput_1000:.1f} traj/sec  ({speedup_1000:.1f}× speedup)")
    print()

    # GO/NO-GO Decision
    print("="*70)
    print("GO/NO-GO DECISION")
    print("="*70)
    print()

    if speedup_100 >= 50:
        print(f"✅ SUCCESS: {speedup_100:.1f}× speedup achieved!")
        print(f"   Target was >50×, we got {speedup_100:.1f}×")
        print()
        print("🚀 RECOMMENDATION: PROCEED to Hold'em rewrite")
        print("   The JAX-native approach is validated!")
        return True
    elif speedup_100 >= 10:
        print(f"⚠️  MODERATE: {speedup_100:.1f}× speedup achieved")
        print(f"   Target was >50×, we got {speedup_100:.1f}×")
        print()
        print("🤔 RECOMMENDATION: Investigate bottlenecks")
        print("   10× is good, but we expected more")
        print("   Check: JIT overhead, memory transfers, etc.")
        return False
    else:
        print(f"❌ INSUFFICIENT: Only {speedup_100:.1f}× speedup")
        print(f"   Target was >50×, this is not enough")
        print()
        print("🛑 RECOMMENDATION: ABORT JAX-native rewrite")
        print("   Pivot to Plan B: Vectorized regret updates")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print()
        print("Next: Integrate batched sampling into GPUMCCFRSolver!")
    else:
        print()
        print("Need to debug performance or pivot to Plan B")
