#!/usr/bin/env python3
"""
Test Preflop-Only GPU-Resident MCCFR

Tests the preflop-only solver on both 2-player and 3+ player games.
Expected performance: 5-10 it/s (vs 0.292 it/s for full game)

Requirements:
    source ~/open_spiel/venv/bin/activate
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr.gpu_mccfr_solver import GPURegretTable, MCCFRConfig
from matrix_cfr import preflop_holdem_jax
from matrix_cfr.preflop_bucketing import preflop_state_to_bucket_index
from matrix_cfr.bucketing import compute_cfvs_vectorized, compute_regret_deltas_vectorized


class PreflopGPUMCCFRSolver:
    """GPU-Resident MCCFR Solver for Preflop-Only Poker."""

    def __init__(self, config: MCCFRConfig, seed: int = 42):
        """
        Initialize preflop GPU MCCFR solver.

        Args:
            config: MCCFR configuration
            seed: Random seed
        """
        self.config = config
        self.key = random.PRNGKey(seed)
        self.regret_tables = None
        self.iteration = 0

    def _sample_preflop_trajectory(
        self,
        key: jnp.ndarray,
        num_players: int,
        stacks: jnp.ndarray,
        blinds: jnp.ndarray,
        max_actions: int = 20
    ):
        """
        Sample a single preflop trajectory.

        Returns:
            states: List of PreflopState objects
            actions: List of action indices
            players: List of acting player indices
            payoffs: Final payoffs
        """
        state = preflop_holdem_jax.deal_initial_state(key, num_players, stacks, blinds)

        states = []
        actions_taken = []
        players = []

        step = 0
        while not preflop_holdem_jax.is_terminal(state) and step < max_actions:
            states.append(state)
            players.append(int(state.acting_player))

            # Uniform random policy
            legal = preflop_holdem_jax.legal_actions(state)
            probs = legal.astype(jnp.float32)
            probs = probs / (jnp.sum(probs) + 1e-10)

            key, subkey = random.split(key)
            action = int(random.choice(subkey, jnp.arange(4), p=probs))
            actions_taken.append(action)

            state = preflop_holdem_jax.apply_action(state, action)
            step += 1

        final_payoffs = preflop_holdem_jax.payoffs(state)

        return states, actions_taken, players, final_payoffs

    def run_iteration_preflop(
        self,
        num_players: int,
        stacks: jnp.ndarray,
        blinds: jnp.ndarray,
        num_buckets: int = 500,
        num_hand_buckets: int = 50,
        num_pot_buckets: int = 5
    ):
        """
        Run one preflop GPU-resident MCCFR iteration.

        Pipeline:
        1. Sample batch of preflop trajectories
        2. Convert states to bucket indices
        3. Compute CFVs
        4. Compute regret deltas
        5. Scatter updates to regret tensors
        6. Update strategy sums
        """
        # Choose updating player
        self.key, subkey = random.split(self.key)
        updating_player = int(random.randint(subkey, (), 0, num_players))

        batch_size = self.config.batch_size

        # Sample batch of trajectories
        self.key, *subkeys = random.split(self.key, batch_size + 1)

        trajectories = []
        for i in range(batch_size):
            key = subkeys[i]
            states, actions, players, payoffs = self._sample_preflop_trajectory(
                key, num_players, stacks, blinds, max_actions=20
            )
            trajectories.append((states, actions, players, payoffs))

        # Convert to batch format
        max_length = max(len(states) for states, _, _, _ in trajectories)

        batch_bucket_indices = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
        batch_actions = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
        batch_players = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
        batch_valid = jnp.zeros((batch_size, max_length), dtype=bool)
        batch_payoffs = jnp.zeros((batch_size, num_players), dtype=jnp.float32)

        for i, (states, actions, players, payoffs) in enumerate(trajectories):
            traj_len = len(states)

            # Convert states to bucket indices
            for t in range(traj_len):
                bucket_idx = preflop_state_to_bucket_index(
                    states[t],
                    updating_player,
                    num_buckets,
                    num_hand_buckets,
                    num_pot_buckets
                )
                batch_bucket_indices = batch_bucket_indices.at[i, t].set(bucket_idx)

            batch_actions = batch_actions.at[i, :traj_len].set(jnp.array(actions))
            batch_players = batch_players.at[i, :traj_len].set(jnp.array(players))
            batch_valid = batch_valid.at[i, :traj_len].set(True)
            batch_payoffs = batch_payoffs.at[i].set(payoffs)

        # Compute CFVs
        cfvs = compute_cfvs_vectorized(
            batch_payoffs,
            batch_valid,
            batch_players,
            updating_player
        )

        # Compute regret deltas
        regret_deltas = compute_regret_deltas_vectorized(
            cfvs,
            batch_actions,
            batch_valid,
            num_actions=self.config.num_actions
        )

        # Filter to updating player and flatten
        is_updating_player = (batch_players == updating_player) & batch_valid

        flat_bucket_indices = batch_bucket_indices[is_updating_player]
        flat_regret_deltas = regret_deltas[is_updating_player]

        # Scatter regret updates
        self.regret_tables[updating_player].batch_update_regrets(
            flat_bucket_indices,
            flat_regret_deltas
        )

        # Update strategy sums
        num_updates = jnp.sum(is_updating_player)
        legal_masks = jnp.ones((num_updates, self.config.num_actions), dtype=bool)

        strategies = self.regret_tables[updating_player].batch_get_strategies(
            flat_bucket_indices,
            legal_masks
        )

        weight = self.iteration + 1 if self.config.use_linear_weighting else 1.0
        self.regret_tables[updating_player].batch_update_strategy_sum(
            flat_bucket_indices,
            strategies,
            weight=weight
        )

        self.iteration += 1

        return int(jnp.sum(batch_valid))


def test_preflop_solver(num_players: int):
    """Test preflop solver with specified number of players."""
    print("=" * 70)
    print(f"Preflop-Only GPU-Resident MCCFR - {num_players} Players")
    print("=" * 70)
    print()

    # Configuration
    if num_players == 2:
        stacks = jnp.array([1000.0, 1000.0])
        blinds = jnp.array([50.0, 100.0])
    elif num_players == 3:
        stacks = jnp.array([1000.0, 1000.0, 1000.0])
        blinds = jnp.array([50.0, 100.0, 0.0])
    else:
        stacks = jnp.ones(num_players) * 1000.0
        blinds = jnp.zeros(num_players)
        blinds = blinds.at[0].set(50.0)
        blinds = blinds.at[1].set(100.0)

    num_iterations = 20
    batch_size = 100
    num_buckets = 500
    num_hand_buckets = 50
    num_pot_buckets = 5

    print("Configuration:")
    print(f"  Players: {num_players}")
    print(f"  Stacks: {stacks}")
    print(f"  Blinds: {blinds}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num buckets: {num_buckets}")
    print(f"  Hand buckets: {num_hand_buckets}")
    print(f"  Pot buckets: {num_pot_buckets}")
    print()

    # Create solver
    config = MCCFRConfig(batch_size=batch_size, num_actions=4)
    solver = PreflopGPUMCCFRSolver(config, seed=42)

    # Initialize GPU regret tables
    print("Initializing GPU regret tables...")
    solver.regret_tables = [
        GPURegretTable(num_buckets=num_buckets, num_actions=config.num_actions)
        for _ in range(num_players)
    ]

    mem_usage = solver.regret_tables[0].get_memory_usage_mb()
    print(f"  GPU memory: {mem_usage:.3f} MB per player × {num_players} = {mem_usage * num_players:.3f} MB total")
    print()

    # Run iterations
    print(f"Running {num_iterations} preflop iterations...")
    print(f"{'Iter':>5} | {'Time (s)':>8} | {'Speed (it/s)':>12} | {'Throughput (traj/s)':>20}")
    print("-" * 70)

    start_time = time.time()

    for i in range(num_iterations):
        iter_start = time.time()

        # Run preflop iteration
        traj_length = solver.run_iteration_preflop(
            num_players,
            stacks,
            blinds,
            num_buckets=num_buckets,
            num_hand_buckets=num_hand_buckets,
            num_pot_buckets=num_pot_buckets
        )

        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        speed = (i + 1) / elapsed
        throughput = speed * batch_size

        print(f"{i+1:5d} | {iter_time:8.2f} | {speed:12.2f} | {throughput:20.0f}")

    total_time = time.time() - start_time
    final_speed = num_iterations / total_time
    final_throughput = final_speed * batch_size

    print("-" * 70)
    print()

    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Total time: {total_time:.2f}s for {num_iterations} iterations")
    print(f"Speed: {final_speed:.3f} it/s")
    print(f"Throughput: {final_throughput:.0f} trajectories/s")
    print(f"GPU Memory: {mem_usage * num_players:.3f} MB")
    print()

    # Compare to baseline
    baseline_speed = 0.0022  # Sequential MCCFR
    speedup = final_throughput / baseline_speed

    print("Baseline Comparison:")
    print(f"  Baseline: {baseline_speed} it/s (sequential MCCFR)")
    print(f"  Preflop GPU-Resident: {final_throughput:.0f} traj/s")
    print(f"  **Speedup: {speedup:.0f}×**")
    print()

    # Success criteria
    print("=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    print()

    criteria = [
        ("Minimum (454× speedup)", 1.0, 454),
        ("Target (900× speedup)", 2.0, 900),
        ("Stretch (1364× speedup)", 3.0, 1364),
        ("Super Stretch (2273× speedup)", 5.0, 2273),
    ]

    best_met = None
    for name, min_speed, min_speedup in criteria:
        met = final_speed >= min_speed
        symbol = "✓" if met else "✗"
        print(f"{symbol} {name:30s} | Speed ≥ {min_speed:.1f} it/s | Speedup ≥ {min_speedup}×")
        if met:
            best_met = name

    print()

    if best_met:
        print(f"✓ SUCCESS! Achieved {best_met}")
    else:
        print("⚠ Did not meet minimum success criteria")

    print()
    print("=" * 70)
    print(f"{num_players}-Player Preflop Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("PREFLOP-ONLY GPU-RESIDENT MCCFR BENCHMARKS")
    print("=" * 70)
    print()

    # Test 2-player
    test_preflop_solver(num_players=2)

    print("\n\n")

    # Test 3-player
    test_preflop_solver(num_players=3)
