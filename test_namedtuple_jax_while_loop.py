"""
Minimal test to isolate JAX + NamedTuple interaction in while_loop.

This test checks if passing NamedTuples through jax.lax.while_loop
causes memory accumulation, independent of the poker solver.

Run with:
    source ~/open_spiel/venv/bin/activate
    python test_namedtuple_jax_while_loop.py
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import NamedTuple
import psutil
import gc
import time


class SimpleState(NamedTuple):
    """Minimal NamedTuple mimicking HoldemState structure."""
    cards: jnp.ndarray  # Shape: (4,)
    pot: float
    round: int
    counter: int


def test_namedtuple_while_loop():
    """Test if NamedTuples in while_loop carry cause memory leak."""
    print("\n" + "="*70)
    print("JAX NamedTuple in While Loop Memory Test")
    print("="*70)
    print("\nTesting: NamedTuple passed through jax.lax.while_loop")
    print("Expected: Should show memory growth if JAX+NamedTuple leaks\n")

    @jax.jit
    def simulate_with_namedtuple(key, max_steps=50):
        """Simulate trajectory sampling with NamedTuple in carry."""
        # Create initial NamedTuple
        initial_state = SimpleState(
            cards=jnp.array([0, 1, 2, 3]),
            pot=100.0,
            round=0,
            counter=0
        )

        # While loop with NamedTuple in carry
        def cond_fn(carry):
            state, step = carry
            return step < max_steps

        def body_fn(carry):
            state, step = carry
            # Update NamedTuple (creates new one each iteration)
            new_state = SimpleState(
                cards=state.cards + 1,
                pot=state.pot * 1.01,
                round=state.round + (step % 10 == 0),  # Increment every 10 steps
                counter=state.counter + 1
            )
            return new_state, step + 1

        final_state, final_step = lax.while_loop(
            cond_fn,
            body_fn,
            (initial_state, 0)
        )

        return final_state

    process = psutil.Process()

    # Initial memory
    gc.collect()
    time.sleep(0.5)
    initial_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Initial memory: {initial_mem:.1f} MB\n")

    num_iterations = 100  # More iterations to amplify effect

    for i in range(num_iterations):
        key = jax.random.PRNGKey(i)

        # Run simulation (NamedTuple through while_loop)
        result = simulate_with_namedtuple(key, max_steps=50)

        # Force computation
        _ = result.cards.block_until_ready()

        # Check memory every 10 iterations
        if (i + 1) % 10 == 0:
            gc.collect()
            current_mem = process.memory_info().rss / (1024 * 1024)
            growth = current_mem - initial_mem
            per_iter = growth / (i + 1)
            print(f"Iteration {i+1:3d}: {current_mem:7.1f} MB (+{growth:6.1f} MB total, {per_iter:5.2f} MB/iter)")

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


def test_flat_array_while_loop():
    """Control test: same logic but with flat arrays instead of NamedTuple."""
    print("\n" + "="*70)
    print("JAX Flat Array in While Loop Memory Test (Control)")
    print("="*70)
    print("\nTesting: Flat array passed through jax.lax.while_loop")
    print("Expected: Should show NO memory growth (control)\n")

    @jax.jit
    def simulate_with_flat_array(key, max_steps=50):
        """Simulate trajectory sampling with flat array in carry."""
        # Create initial flat array: [cards(4), pot, round, counter]
        initial_state_flat = jnp.array([0.0, 1.0, 2.0, 3.0, 100.0, 0.0, 0.0])

        # While loop with flat array in carry
        def cond_fn(carry):
            state_flat, step = carry
            return step < max_steps

        def body_fn(carry):
            state_flat, step = carry
            # Update flat array
            cards = state_flat[0:4] + 1
            pot = state_flat[4] * 1.01
            round_val = state_flat[5] + (step % 10 == 0)
            counter = state_flat[6] + 1
            new_state_flat = jnp.array([
                cards[0], cards[1], cards[2], cards[3],
                pot, round_val, counter
            ])
            return new_state_flat, step + 1

        final_state, final_step = lax.while_loop(
            cond_fn,
            body_fn,
            (initial_state_flat, 0)
        )

        return final_state

    process = psutil.Process()

    # Initial memory
    gc.collect()
    time.sleep(0.5)
    initial_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Initial memory: {initial_mem:.1f} MB\n")

    num_iterations = 100

    for i in range(num_iterations):
        key = jax.random.PRNGKey(i)

        # Run simulation (flat array through while_loop)
        result = simulate_with_flat_array(key, max_steps=50)

        # Force computation
        _ = result.block_until_ready()

        # Check memory every 10 iterations
        if (i + 1) % 10 == 0:
            gc.collect()
            current_mem = process.memory_info().rss / (1024 * 1024)
            growth = current_mem - initial_mem
            per_iter = growth / (i + 1)
            print(f"Iteration {i+1:3d}: {current_mem:7.1f} MB (+{growth:6.1f} MB total, {per_iter:5.2f} MB/iter)")

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
    print("JAX + NamedTuple While Loop Memory Leak Test")
    print("Minimal reproduction to isolate JAX interaction")
    print("="*70)

    growth_nt, per_iter_nt = test_namedtuple_while_loop()
    growth_flat, per_iter_flat = test_flat_while_loop()

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"NamedTuple: {per_iter_nt:5.2f} MB/iter")
    print(f"Flat Array: {per_iter_flat:5.2f} MB/iter")
    print(f"Difference: {per_iter_nt - per_iter_flat:5.2f} MB/iter")

    if growth_nt > 100 and growth_flat < 100:
        print("\n✅ CONFIRMED: NamedTuple in while_loop causes leak!")
        print("Solution: Use flat arrays throughout trajectory sampling")
    elif growth_nt < 100 and growth_flat < 100:
        print("\n❓ UNEXPECTED: Neither version leaks")
        print("Leak may be from vmap interaction or game engine specific")
    else:
        print("\n⚠️  INCONCLUSIVE: Both versions show memory growth")
    print("="*70)
