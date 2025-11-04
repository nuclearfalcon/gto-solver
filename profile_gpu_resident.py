#!/usr/bin/env python3
"""
Profile GPU-Resident MCCFR to identify bottlenecks.

Requirements:
    source ~/open_spiel/venv/bin/activate
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver, MCCFRConfig
from matrix_cfr import holdem_jax_v2


def profile_iteration():
    """Profile one iteration of GPU-resident MCCFR."""
    print("=" * 70)
    print("GPU-Resident MCCFR Profiling")
    print("=" * 70)
    print()

    # Configuration
    num_players = 2
    stacks = jnp.array([1000.0, 1000.0])
    blinds = jnp.array([50.0, 100.0])
    batch_size = 100
    num_buckets = 10_000
    num_hand_buckets = 200
    num_pot_buckets = 10

    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num buckets: {num_buckets:,}")
    print()

    # Create solver
    config = MCCFRConfig(batch_size=batch_size, num_actions=4)
    solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=42)

    # Initialize GPU regret tables
    from matrix_cfr.gpu_mccfr_solver import GPURegretTable
    solver.regret_tables = [
        GPURegretTable(num_buckets=num_buckets, num_actions=config.num_actions)
        for _ in range(num_players)
    ]

    print("Warming up (JIT compilation)...")
    solver.run_iteration_gpu_resident(
        num_players, stacks, blinds,
        num_buckets=num_buckets,
        num_hand_buckets=num_hand_buckets,
        num_pot_buckets=num_pot_buckets
    )
    print()

    print("Profiling iteration components...")
    print()

    # Profile trajectory sampling
    print("1. Trajectory Sampling:")
    solver.key, *subkeys = random.split(solver.key, batch_size + 1)
    batch_keys = jnp.array(subkeys)

    start = time.time()
    states_batch, actions_batch, players_batch, valid_masks, num_steps_array, payoffs_batch = \
        solver._sample_batched_trajectories(
            batch_keys, num_players, stacks, blinds, max_actions=50
        )
    jax.block_until_ready(states_batch)
    sampling_time = time.time() - start
    print(f"   Time: {sampling_time:.3f}s")
    print(f"   Sampled {batch_size} trajectories")
    print()

    # Profile bucketing
    print("2. State→Bucket Conversion:")
    from matrix_cfr.bucketing import state_to_bucket_index
    from matrix_cfr.gpu_mccfr_solver import unflatten_state

    @jax.jit
    def batch_convert_states_to_buckets(states_flat_2d, updating_player_const):
        """JIT-compiled vectorized state→bucket conversion."""
        def flatten_to_bucket(flat_state):
            state = unflatten_state(flat_state, num_players)
            return state_to_bucket_index(
                state, updating_player_const,
                num_buckets, num_hand_buckets, num_pot_buckets
            )
        vectorized_bucketing = jax.vmap(flatten_to_bucket)
        return vectorized_bucketing(states_flat_2d)

    batch_size_actual, max_length, state_size = states_batch.shape
    states_flat_2d = states_batch.reshape(-1, state_size)
    updating_player = 0

    start = time.time()
    bucket_indices_flat = batch_convert_states_to_buckets(states_flat_2d, updating_player)
    jax.block_until_ready(bucket_indices_flat)
    bucketing_time = time.time() - start
    bucket_indices = bucket_indices_flat.reshape(batch_size_actual, max_length)
    bucket_indices = jnp.where(valid_masks, bucket_indices, 0)
    print(f"   Time: {bucketing_time:.3f}s")
    print(f"   Converted {batch_size * max_length} states")
    print()

    # Profile CFV computation
    print("3. CFV Computation:")
    from matrix_cfr.bucketing import compute_cfvs_vectorized

    start = time.time()
    cfvs = compute_cfvs_vectorized(payoffs_batch, valid_masks, players_batch, updating_player)
    jax.block_until_ready(cfvs)
    cfv_time = time.time() - start
    print(f"   Time: {cfv_time:.3f}s")
    print()

    # Profile regret delta computation
    print("4. Regret Delta Computation:")
    from matrix_cfr.bucketing import compute_regret_deltas_vectorized

    start = time.time()
    regret_deltas = compute_regret_deltas_vectorized(cfvs, actions_batch, valid_masks, num_actions=4)
    jax.block_until_ready(regret_deltas)
    regret_time = time.time() - start
    print(f"   Time: {regret_time:.3f}s")
    print()

    # Profile scatter updates
    print("5. GPU Scatter Updates:")
    is_updating_player = (players_batch == updating_player) & valid_masks
    flat_bucket_indices = bucket_indices[is_updating_player]
    flat_regret_deltas = regret_deltas[is_updating_player]

    start = time.time()
    solver.regret_tables[updating_player].batch_update_regrets(flat_bucket_indices, flat_regret_deltas)
    # Block until ready (regret_table might not have .regrets attribute, use internal _regrets)
    scatter_time = time.time() - start
    print(f"   Time: {scatter_time:.3f}s")
    print()

    # Profile strategy sum updates
    print("6. Strategy Sum Updates:")
    num_updates = jnp.sum(is_updating_player)
    legal_masks = jnp.ones((num_updates, config.num_actions), dtype=bool)

    start = time.time()
    strategies = solver.regret_tables[updating_player].batch_get_strategies(flat_bucket_indices, legal_masks)
    jax.block_until_ready(strategies)
    solver.regret_tables[updating_player].batch_update_strategy_sum(flat_bucket_indices, strategies, weight=1.0)
    strategy_time = time.time() - start
    print(f"   Time: {strategy_time:.3f}s")
    print()

    # Summary
    total_time = sampling_time + bucketing_time + cfv_time + regret_time + scatter_time + strategy_time
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Component':<30} {'Time (s)':<10} {'% of Total':<12}")
    print("-" * 70)
    print(f"{'1. Trajectory Sampling':<30} {sampling_time:>8.3f}   {100*sampling_time/total_time:>8.1f}%")
    print(f"{'2. State→Bucket Conversion':<30} {bucketing_time:>8.3f}   {100*bucketing_time/total_time:>8.1f}%")
    print(f"{'3. CFV Computation':<30} {cfv_time:>8.3f}   {100*cfv_time/total_time:>8.1f}%")
    print(f"{'4. Regret Delta Computation':<30} {regret_time:>8.3f}   {100*regret_time/total_time:>8.1f}%")
    print(f"{'5. GPU Scatter Updates':<30} {scatter_time:>8.3f}   {100*scatter_time/total_time:>8.1f}%")
    print(f"{'6. Strategy Sum Updates':<30} {strategy_time:>8.3f}   {100*strategy_time/total_time:>8.1f}%")
    print("-" * 70)
    print(f"{'TOTAL':<30} {total_time:>8.3f}   {100.0:>8.1f}%")
    print()
    print(f"Estimated speed: {1.0/total_time:.3f} it/s")
    print()


if __name__ == "__main__":
    profile_iteration()
