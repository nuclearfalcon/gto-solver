#!/usr/bin/env python3
"""
Test GPU-Resident Regret Tensor for Phase 10.5

Validates:
1. GPU regret tensor initialization
2. Single bucket regret updates
3. Batch scatter regret updates
4. Strategy computation (regret matching)
5. Strategy sum updates
6. Memory usage

Remember to activate virtual environment:
    source ~/open_spiel/venv/bin/activate
"""

import jax
import jax.numpy as jnp
import numpy as np

from matrix_cfr.gpu_mccfr_solver import GPURegretTable


def test_initialization():
    """Test GPU regret table initialization."""
    print("Testing GPU regret table initialization...")

    table = GPURegretTable(num_buckets=1000, num_actions=4)

    # Check shapes
    assert table.cumulative_regrets.shape == (1000, 4), "Regrets shape mismatch"
    assert table.strategy_sum.shape == (1000, 4), "Strategy sum shape mismatch"

    # Check initial values are zero
    assert jnp.all(table.cumulative_regrets == 0), "Initial regrets not zero"
    assert jnp.all(table.strategy_sum == 0), "Initial strategy sum not zero"

    # Check memory usage
    mem_mb = table.get_memory_usage_mb()
    expected_mb = (1000 * 4 * 4 * 2) / (1024 * 1024)  # 2 tensors, 4 bytes per float32
    assert abs(mem_mb - expected_mb) < 0.01, f"Memory usage mismatch: {mem_mb} vs {expected_mb}"

    print(f"  ✓ Initialization successful")
    print(f"    Memory usage: {mem_mb:.4f} MB")


def test_single_bucket_update():
    """Test single bucket regret update."""
    print("\nTesting single bucket regret update...")

    table = GPURegretTable(num_buckets=100, num_actions=4)

    # Update bucket 5
    bucket_idx = 5
    regrets = jnp.array([1.0, -2.0, 3.0, 0.5])
    table.update_regrets(bucket_idx, regrets)

    # Check update
    retrieved = table.get_regrets(bucket_idx)
    assert jnp.allclose(retrieved, regrets), "Single update failed"

    # Update again (should accumulate)
    table.update_regrets(bucket_idx, regrets)
    retrieved = table.get_regrets(bucket_idx)
    assert jnp.allclose(retrieved, regrets * 2), "Accumulation failed"

    print(f"  ✓ Single bucket updates work correctly")
    print(f"    After 2 updates: {retrieved}")


def test_batch_scatter_update():
    """Test batch scatter regret updates (key GPU operation)."""
    print("\nTesting batch scatter regret updates...")

    table = GPURegretTable(num_buckets=100, num_actions=4)

    # Update multiple buckets at once
    bucket_indices = jnp.array([5, 10, 5, 20])  # Note: bucket 5 appears twice!
    regret_deltas = jnp.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 0.5, 0.5, 0.5],
        [2.0, 1.0, 0.0, -1.0],  # Second update to bucket 5
        [-1.0, 3.0, 2.0, 1.0]
    ])

    table.batch_update_regrets(bucket_indices, regret_deltas)

    # Check bucket 5 (should have accumulated both updates)
    bucket_5_regrets = table.get_regrets(5)
    expected_5 = regret_deltas[0] + regret_deltas[2]  # Sum of two updates
    assert jnp.allclose(bucket_5_regrets, expected_5), \
        f"Bucket 5 accumulation failed: {bucket_5_regrets} vs {expected_5}"

    # Check bucket 10
    bucket_10_regrets = table.get_regrets(10)
    assert jnp.allclose(bucket_10_regrets, regret_deltas[1]), "Bucket 10 update failed"

    # Check bucket 20
    bucket_20_regrets = table.get_regrets(20)
    assert jnp.allclose(bucket_20_regrets, regret_deltas[3]), "Bucket 20 update failed"

    print(f"  ✓ Batch scatter updates work correctly")
    print(f"    Bucket 5 (2 updates): {bucket_5_regrets}")
    print(f"    Bucket 10: {bucket_10_regrets}")
    print(f"    Bucket 20: {bucket_20_regrets}")


def test_strategy_computation():
    """Test regret matching strategy computation."""
    print("\nTesting regret matching strategy computation...")

    table = GPURegretTable(num_buckets=10, num_actions=4)

    # Set up regrets for bucket 3
    bucket_idx = 3
    regrets = jnp.array([5.0, -2.0, 10.0, 0.0])
    table.update_regrets(bucket_idx, regrets)

    # Compute strategy
    legal_mask = jnp.array([True, True, True, True])
    strategy = table.get_strategy(bucket_idx, legal_mask)

    # CFR+ regret matching: max(regret, 0) / sum(max(regret, 0))
    # Positive regrets: [5.0, 0.0, 10.0, 0.0]
    # Sum: 15.0
    # Strategy: [5/15, 0/15, 10/15, 0/15] = [0.333, 0, 0.667, 0]
    expected = jnp.array([5.0, 0.0, 10.0, 0.0]) / 15.0
    assert jnp.allclose(strategy, expected, atol=1e-6), \
        f"Strategy mismatch: {strategy} vs {expected}"

    print(f"  ✓ Regret matching works correctly")
    print(f"    Regrets: {regrets}")
    print(f"    Strategy: {strategy}")

    # Test uniform fallback (all regrets zero or negative)
    table2 = GPURegretTable(num_buckets=10, num_actions=4)
    bucket_idx2 = 0
    table2.update_regrets(bucket_idx2, jnp.array([-1.0, -2.0, -3.0, 0.0]))
    strategy2 = table2.get_strategy(bucket_idx2, legal_mask)
    expected_uniform = jnp.array([0.25, 0.25, 0.25, 0.25])
    assert jnp.allclose(strategy2, expected_uniform, atol=1e-6), \
        "Uniform fallback failed"

    print(f"  ✓ Uniform fallback works correctly")
    print(f"    All negative regrets → Uniform strategy: {strategy2}")


def test_batch_strategy_computation():
    """Test vectorized strategy computation."""
    print("\nTesting batch strategy computation...")

    table = GPURegretTable(num_buckets=100, num_actions=4)

    # Set up regrets for multiple buckets
    bucket_indices = jnp.array([5, 10, 15])
    regret_deltas = jnp.array([
        [5.0, -2.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],  # All zero → uniform
        [1.0, 1.0, 1.0, 1.0]   # All equal → uniform
    ])
    table.batch_update_regrets(bucket_indices, regret_deltas)

    # Compute strategies for all buckets at once
    legal_masks = jnp.array([
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]
    ])
    strategies = table.batch_get_strategies(bucket_indices, legal_masks)

    # Check shapes
    assert strategies.shape == (3, 4), "Batch strategy shape mismatch"

    # Check bucket 5 strategy
    expected_5 = jnp.array([5.0, 0.0, 10.0, 0.0]) / 15.0
    assert jnp.allclose(strategies[0], expected_5, atol=1e-6), "Batch strategy 0 failed"

    # Check bucket 10 strategy (uniform)
    expected_10 = jnp.array([0.25, 0.25, 0.25, 0.25])
    assert jnp.allclose(strategies[1], expected_10, atol=1e-6), "Batch strategy 1 failed"

    # Check bucket 15 strategy (uniform)
    expected_15 = jnp.array([0.25, 0.25, 0.25, 0.25])
    assert jnp.allclose(strategies[2], expected_15, atol=1e-6), "Batch strategy 2 failed"

    print(f"  ✓ Batch strategy computation works correctly")
    print(f"    Strategy 0 (non-uniform): {strategies[0]}")
    print(f"    Strategy 1 (uniform): {strategies[1]}")
    print(f"    Strategy 2 (uniform): {strategies[2]}")


def test_strategy_sum_updates():
    """Test strategy sum updates for average policy."""
    print("\nTesting strategy sum updates...")

    table = GPURegretTable(num_buckets=10, num_actions=4)

    # Update strategy sum for bucket 2
    bucket_idx = 2
    strategy1 = jnp.array([0.5, 0.2, 0.2, 0.1])
    strategy2 = jnp.array([0.3, 0.3, 0.3, 0.1])

    table.update_strategy_sum(bucket_idx, strategy1, weight=1.0)
    table.update_strategy_sum(bucket_idx, strategy2, weight=1.0)

    # Check accumulated sum
    expected_sum = strategy1 + strategy2
    actual_sum = table.strategy_sum[bucket_idx]
    assert jnp.allclose(actual_sum, expected_sum), "Strategy sum accumulation failed"

    # Get average strategy
    legal_mask = jnp.array([True, True, True, True])
    avg_strategy = table.get_average_strategy(bucket_idx, legal_mask)

    # Average should be normalized sum
    expected_avg = expected_sum / jnp.sum(expected_sum)
    assert jnp.allclose(avg_strategy, expected_avg, atol=1e-6), \
        f"Average strategy mismatch: {avg_strategy} vs {expected_avg}"

    print(f"  ✓ Strategy sum updates work correctly")
    print(f"    Strategy sum: {actual_sum}")
    print(f"    Average strategy: {avg_strategy}")


def test_batch_strategy_sum_updates():
    """Test batch strategy sum updates."""
    print("\nTesting batch strategy sum updates...")

    table = GPURegretTable(num_buckets=100, num_actions=4)

    # Batch update strategy sums
    bucket_indices = jnp.array([10, 20, 10])  # Bucket 10 appears twice
    strategies = jnp.array([
        [0.5, 0.2, 0.2, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.3, 0.3, 0.3, 0.1]  # Second update to bucket 10
    ])

    table.batch_update_strategy_sum(bucket_indices, strategies, weight=1.0)

    # Check bucket 10 (should have both updates)
    bucket_10_sum = table.strategy_sum[10]
    expected_10_sum = strategies[0] + strategies[2]
    assert jnp.allclose(bucket_10_sum, expected_10_sum), "Bucket 10 sum accumulation failed"

    # Check bucket 20
    bucket_20_sum = table.strategy_sum[20]
    expected_20_sum = strategies[1]
    assert jnp.allclose(bucket_20_sum, expected_20_sum), "Bucket 20 sum failed"

    print(f"  ✓ Batch strategy sum updates work correctly")
    print(f"    Bucket 10 sum (2 updates): {bucket_10_sum}")
    print(f"    Bucket 20 sum: {bucket_20_sum}")


def test_legal_action_masking():
    """Test that illegal actions are properly masked."""
    print("\nTesting legal action masking...")

    table = GPURegretTable(num_buckets=10, num_actions=4)

    # Set up regrets
    bucket_idx = 0
    regrets = jnp.array([5.0, 10.0, 3.0, 2.0])
    table.update_regrets(bucket_idx, regrets)

    # Only actions 0 and 2 are legal
    legal_mask = jnp.array([True, False, True, False])
    strategy = table.get_strategy(bucket_idx, legal_mask)

    # Should only have probability on actions 0 and 2
    assert strategy[1] == 0.0, "Illegal action 1 has non-zero probability"
    assert strategy[3] == 0.0, "Illegal action 3 has non-zero probability"

    # Should sum to 1.0
    assert jnp.abs(jnp.sum(strategy) - 1.0) < 1e-6, "Strategy doesn't sum to 1"

    # Should be proportional to positive regrets of legal actions
    # Positive regrets: [5.0, 10.0 (illegal), 3.0, 2.0 (illegal)]
    # Legal positive: [5.0, 3.0]
    # Strategy: [5/8, 0, 3/8, 0]
    expected = jnp.array([5.0/8.0, 0.0, 3.0/8.0, 0.0])
    assert jnp.allclose(strategy, expected, atol=1e-6), \
        f"Legal masking failed: {strategy} vs {expected}"

    print(f"  ✓ Legal action masking works correctly")
    print(f"    Legal mask: {legal_mask}")
    print(f"    Strategy: {strategy}")


def test_memory_scaling():
    """Test memory usage scales correctly with bucket count."""
    print("\nTesting memory scaling...")

    configs = [
        (1_000, 4),
        (10_000, 4),
        (100_000, 4),
    ]

    for num_buckets, num_actions in configs:
        table = GPURegretTable(num_buckets=num_buckets, num_actions=num_actions)
        mem_mb = table.get_memory_usage_mb()
        expected_mb = (num_buckets * num_actions * 4 * 2) / (1024 * 1024)

        assert abs(mem_mb - expected_mb) < 0.01, \
            f"Memory scaling failed for {num_buckets} buckets"

        print(f"  {num_buckets:6d} buckets × {num_actions} actions = {mem_mb:.2f} MB")

    print(f"  ✓ Memory scaling is correct")


def main():
    """Run all GPU regret tensor tests."""
    print("=" * 60)
    print("Phase 10.5: GPU Regret Tensor Validation")
    print("=" * 60)

    test_initialization()
    test_single_bucket_update()
    test_batch_scatter_update()
    test_strategy_computation()
    test_batch_strategy_computation()
    test_strategy_sum_updates()
    test_batch_strategy_sum_updates()
    test_legal_action_masking()
    test_memory_scaling()

    print("\n" + "=" * 60)
    print("✓ All GPU regret tensor tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
