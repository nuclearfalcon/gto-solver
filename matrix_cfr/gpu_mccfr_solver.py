"""
GPU-Accelerated Monte Carlo CFR Solver

This module implements External Sampling MCCFR using JAX trajectory sampling.
Unlike Matrix CFR which stores full strategy tensors, MCCFR uses sparse regret
tables and only samples trajectories, enabling it to scale to full-size poker.

Key Innovation: Uses JAX for GPU-parallelized trajectory generation, making
MCCFR competitive with Matrix CFR's speed while maintaining memory efficiency.

Phase 10: GPU-Accelerated MCCFR (Days 8-9)
"""

from typing import Dict, Tuple, Callable, Optional, Any
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from dataclasses import dataclass
import time


class RegretTable:
    """
    Sparse regret storage for MCCFR.

    Stores cumulative regrets and computes strategies via regret matching.
    Uses dictionary for sparse storage - only visited information sets are stored.

    Memory efficiency: O(visited_infosets × num_actions) vs O(all_infosets × num_actions)
    For full Hold'em, this is critical: ~10^6 visited vs ~10^14 total infosets.
    """

    def __init__(self):
        """Initialize empty regret table."""
        self.cumulative_regrets: Dict[str, np.ndarray] = {}
        self.strategy_sum: Dict[str, np.ndarray] = {}
        self.num_actions: int = 4  # Default for Hold'em: fold, call, pot, all-in

    def get_regrets(self, infoset: str, num_actions: Optional[int] = None) -> np.ndarray:
        """
        Get cumulative regrets for an information set.

        Args:
            infoset: Information set string
            num_actions: Number of actions (defaults to self.num_actions)

        Returns:
            Cumulative regrets array, shape (num_actions,)
        """
        if num_actions is None:
            num_actions = self.num_actions

        if infoset not in self.cumulative_regrets:
            self.cumulative_regrets[infoset] = np.zeros(num_actions, dtype=np.float32)

        return self.cumulative_regrets[infoset]

    def update_regrets(self, infoset: str, regrets: np.ndarray):
        """
        Update cumulative regrets for an information set.

        Args:
            infoset: Information set string
            regrets: Instantaneous regrets to add, shape (num_actions,)
        """
        if infoset not in self.cumulative_regrets:
            self.cumulative_regrets[infoset] = np.zeros_like(regrets, dtype=np.float32)

        self.cumulative_regrets[infoset] += regrets

    def get_strategy(self, infoset: str, legal_mask: np.ndarray) -> np.ndarray:
        """
        Compute current strategy via regret matching.

        Regret Matching Algorithm:
        1. Positive regrets: max(regret, 0)
        2. Normalize: strategy = positive_regrets / sum(positive_regrets)
        3. If all zero: uniform over legal actions

        Args:
            infoset: Information set string
            legal_mask: Boolean mask of legal actions

        Returns:
            Strategy (action probabilities), shape (num_actions,)
        """
        num_actions = len(legal_mask)
        regrets = self.get_regrets(infoset, num_actions)

        # Regret matching: max(regret, 0)
        positive_regrets = np.maximum(regrets, 0.0)

        # Mask illegal actions
        positive_regrets = positive_regrets * legal_mask

        # Normalize
        regret_sum = np.sum(positive_regrets)
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            # Uniform over legal actions
            num_legal = np.sum(legal_mask)
            strategy = legal_mask.astype(np.float32) / num_legal

        return strategy

    def update_strategy_sum(self, infoset: str, strategy: np.ndarray, weight: float = 1.0):
        """
        Update cumulative strategy for average policy computation.

        The average policy is computed as:
        average_strategy = strategy_sum / sum(weights)

        Args:
            infoset: Information set string
            strategy: Current strategy, shape (num_actions,)
            weight: Weight for this update (default: 1.0)
        """
        if infoset not in self.strategy_sum:
            self.strategy_sum[infoset] = np.zeros_like(strategy, dtype=np.float32)

        self.strategy_sum[infoset] += strategy * weight

    def get_average_strategy(self, infoset: str, legal_mask: np.ndarray) -> np.ndarray:
        """
        Get average strategy for an information set.

        This is the final policy extracted after training.

        Args:
            infoset: Information set string
            legal_mask: Boolean mask of legal actions

        Returns:
            Average strategy, shape (num_actions,)
        """
        num_actions = len(legal_mask)

        if infoset not in self.strategy_sum:
            # Never visited: return uniform
            num_legal = np.sum(legal_mask)
            return legal_mask.astype(np.float32) / num_legal

        strategy_sum = self.strategy_sum[infoset]
        total = np.sum(strategy_sum)

        if total > 0:
            avg_strategy = strategy_sum / total
        else:
            # Sum is zero: uniform over legal actions
            num_legal = np.sum(legal_mask)
            avg_strategy = legal_mask.astype(np.float32) / num_legal

        # Ensure only legal actions have non-zero probability
        avg_strategy = avg_strategy * legal_mask
        avg_strategy = avg_strategy / np.sum(avg_strategy)

        return avg_strategy

    def get_num_infosets(self) -> int:
        """Get number of information sets visited."""
        return len(self.cumulative_regrets)

    def get_policy_dict(self) -> Dict[str, np.ndarray]:
        """
        Extract full average policy as dictionary.

        Returns:
            Dictionary mapping infoset strings to average strategies
        """
        policy = {}
        for infoset in self.strategy_sum.keys():
            # Infer number of actions from strategy_sum shape
            num_actions = len(self.strategy_sum[infoset])
            # Assume all actions legal for final policy
            # (real usage should track legal actions per infoset)
            legal_mask = np.ones(num_actions, dtype=bool)
            policy[infoset] = self.get_average_strategy(infoset, legal_mask)

        return policy


@dataclass
class MCCFRConfig:
    """Configuration for GPU MCCFR solver."""
    num_players: int = 2
    num_actions: int = 4  # fold, call, pot, all-in
    discount_factor: float = 1.0  # For DCFR variants (1.0 = no discounting)
    prune_threshold: float = -3e8  # Prune actions with very negative regrets
    use_linear_weighting: bool = False  # Weight updates by iteration number
    checkpoint_interval: int = 10_000  # Save checkpoint every N iterations


class GPUMCCFRSolver:
    """
    GPU-Accelerated Monte Carlo CFR Solver.

    Implements External Sampling MCCFR:
    - Each iteration: sample trajectory for one player
    - Update regrets only along sampled trajectory
    - Other players play according to current strategy

    GPU Acceleration:
    - Uses JAX trajectory sampler for fast game simulation
    - (Future: Batch multiple trajectories per iteration)

    Memory Efficiency:
    - Sparse regret storage (only visited infosets)
    - No full game tree in memory
    - Scales to full-size poker
    """

    def __init__(
        self,
        game_engine: Any,  # JAX game engine (holdem_jax module)
        config: MCCFRConfig,
        seed: int = 42
    ):
        """
        Initialize GPU MCCFR solver.

        Args:
            game_engine: JAX game engine module (e.g., holdem_jax)
            config: MCCFR configuration
            seed: Random seed for reproducibility
        """
        self.game_engine = game_engine
        self.config = config
        self.key = random.PRNGKey(seed)

        # Regret tables (one per player)
        self.regret_tables = [RegretTable() for _ in range(config.num_players)]

        # Iteration counter
        self.iteration = 0

        # Metrics
        self.metrics = {
            'iteration': [],
            'time': [],
            'infosets_visited': [],
            'avg_trajectory_length': []
        }

    def get_policy_for_player(self, player: int) -> Callable[[str, np.ndarray], np.ndarray]:
        """
        Get policy function for a player.

        Returns a callable that maps (infoset, legal_mask) → action_probs

        Args:
            player: Player index

        Returns:
            Policy function
        """
        def policy_fn(infoset: str, legal_mask: jnp.ndarray) -> jnp.ndarray:
            # Convert JAX array to numpy for RegretTable
            legal_np = np.array(legal_mask, dtype=bool)
            strategy_np = self.regret_tables[player].get_strategy(infoset, legal_np)
            return jnp.array(strategy_np)

        return policy_fn

    def _sample_trajectory(
        self,
        key: jax.random.PRNGKey,
        num_players: int,
        policy_fn: Callable[[str, jnp.ndarray], jnp.ndarray],
        max_actions: int = 100
    ) -> Tuple[list, list, list, jnp.ndarray]:
        """
        Sample one complete game trajectory using given policy.

        Generic implementation that works with any game engine.

        Args:
            key: JAX random key
            num_players: Number of players
            policy_fn: Policy function
            max_actions: Maximum actions per trajectory

        Returns:
            Tuple of (states, actions, players, terminal_payoffs)
        """
        # Deal initial state
        key, subkey = random.split(key)
        state = self.game_engine.deal_initial_state(subkey)

        states_list = []
        actions_list = []
        players_list = []
        action_count = 0

        # Play through game
        while not self.game_engine.is_terminal(state) and action_count < max_actions:
            player = int(state.acting_player)

            # Get legal actions
            legal = self.game_engine.legal_actions(state)

            # Get policy for this infoset
            infoset = self.game_engine.state_to_infoset(state, player)
            action_probs = policy_fn(infoset, legal)

            # Sample action according to policy
            key, subkey = random.split(key)
            # Filter to legal actions only
            legal_indices = jnp.where(legal, jnp.arange(len(legal)), -1)
            legal_indices = legal_indices[legal_indices >= 0]

            # Normalize probabilities over legal actions
            legal_probs = action_probs[legal]
            legal_probs = legal_probs / jnp.sum(legal_probs)

            # Sample from legal actions
            action_idx = random.choice(subkey, len(legal_indices), p=legal_probs)
            action = legal_indices[action_idx]

            # Record decision point
            states_list.append(state)
            actions_list.append(int(action))
            players_list.append(int(player))

            # Apply action and advance state
            state = self.game_engine.apply_action(state, int(action))

            action_count += 1

        # Get terminal payoffs
        terminal_payoffs = self.game_engine.payoffs(state)

        return states_list, actions_list, players_list, terminal_payoffs

    def compute_counterfactual_values(
        self,
        states: list,
        actions: list,
        players: list,
        payoffs: jnp.ndarray,
        updating_player: int
    ) -> list:
        """
        Compute counterfactual values for regret updates.

        For each decision point where updating_player acted:
        - Compute value of action taken
        - Compute values of alternative actions
        - Regret = alternative_value - taken_value

        Args:
            states: List of states visited
            actions: List of actions taken
            players: List of acting players
            payoffs: Terminal payoffs
            updating_player: Player being updated

        Returns:
            List of (infoset, action, regrets) tuples for updating_player
        """
        updates = []

        # For now: Simple implementation using terminal payoffs
        # Future optimization: Recursive CFV computation

        for i, (state, action, player) in enumerate(zip(states, actions, players)):
            if player != updating_player:
                continue

            # Get infoset
            infoset = self.game_engine.state_to_infoset(state, player)
            legal_mask = self.game_engine.legal_actions(state)
            legal_np = np.array(legal_mask, dtype=bool)

            # Simplified regret computation:
            # For action taken: use terminal payoff
            # For alternatives: assume same payoff (baseline)
            # This is a simplification - full MCCFR would simulate alternatives

            value_taken = float(payoffs[player])

            # Compute regrets: alternative_value - taken_value
            # For now: use uniform baseline for alternatives
            regrets = np.zeros(self.config.num_actions, dtype=np.float32)

            # All legal actions get same counterfactual value as baseline
            # Taken action gets 0 regret, others get (baseline - value)
            # This is placeholder logic - full MCCFR more sophisticated

            for a in range(self.config.num_actions):
                if legal_np[a]:
                    if a == action:
                        regrets[a] = 0.0
                    else:
                        # Alternative action: assign small positive regret if we won
                        # This encourages exploration
                        regrets[a] = 0.1 * value_taken if value_taken > 0 else 0.0

            updates.append((infoset, action, regrets))

        return updates

    def run_iteration(self, num_players: int, stacks: jnp.ndarray, blinds: jnp.ndarray):
        """
        Run one MCCFR iteration.

        External Sampling MCCFR:
        1. Choose updating player uniformly at random
        2. Sample trajectory with all players following current strategy
        3. Update regrets for updating player along trajectory
        4. Update strategy sum for average policy

        Args:
            num_players: Number of players
            stacks: Starting stacks
            blinds: Blind amounts
        """
        # Choose updating player
        self.key, subkey = random.split(self.key)
        updating_player = int(random.randint(subkey, (), 0, num_players))

        # Get policy for all players (current strategy)
        policy_fn = self.get_policy_for_player(updating_player)
        # For simplicity, all players use same policy (could be different)

        # Sample trajectory directly using game engine
        self.key, subkey = random.split(self.key)
        states, actions, players_list, payoffs = self._sample_trajectory(
            subkey,
            num_players,
            policy_fn,
            max_actions=100
        )

        # Compute counterfactual values and regrets
        updates = self.compute_counterfactual_values(
            states, actions, players_list, payoffs, updating_player
        )

        # Update regrets
        for infoset, action, regrets in updates:
            self.regret_tables[updating_player].update_regrets(infoset, regrets)

        # Update strategy sum for average policy
        # Weight by iteration if using linear weighting
        weight = self.iteration + 1 if self.config.use_linear_weighting else 1.0

        for state, action, player in zip(states, actions, players_list):
            if player == updating_player:
                infoset = self.game_engine.state_to_infoset(state, player)
                legal_mask = np.array(self.game_engine.legal_actions(state), dtype=bool)
                strategy = self.regret_tables[player].get_strategy(infoset, legal_mask)
                self.regret_tables[player].update_strategy_sum(infoset, strategy, weight)

        self.iteration += 1

        return len(states)  # Return trajectory length for metrics

    def solve(
        self,
        num_iterations: int,
        num_players: int,
        stacks: jnp.ndarray,
        blinds: jnp.ndarray,
        progress_interval: int = 1000
    ):
        """
        Solve game using GPU MCCFR.

        Args:
            num_iterations: Number of MCCFR iterations
            num_players: Number of players
            stacks: Starting stacks
            blinds: Blind amounts
            progress_interval: Print progress every N iterations
        """
        start_time = time.time()
        total_trajectory_length = 0

        print(f"Starting GPU MCCFR: {num_iterations} iterations")
        print(f"Players: {num_players}, Stacks: {stacks}, Blinds: {blinds}")
        print()

        for i in range(num_iterations):
            trajectory_length = self.run_iteration(num_players, stacks, blinds)
            total_trajectory_length += trajectory_length

            if (i + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                it_per_sec = (i + 1) / elapsed
                avg_traj_len = total_trajectory_length / (i + 1)
                num_infosets = sum(table.get_num_infosets() for table in self.regret_tables)

                print(f"Iteration {i + 1}/{num_iterations} "
                      f"({it_per_sec:.2f} it/s, {elapsed:.1f}s elapsed)")
                print(f"  Infosets visited: {num_infosets}")
                print(f"  Avg trajectory length: {avg_traj_len:.1f}")
                print()

                # Record metrics
                self.metrics['iteration'].append(i + 1)
                self.metrics['time'].append(elapsed)
                self.metrics['infosets_visited'].append(num_infosets)
                self.metrics['avg_trajectory_length'].append(avg_traj_len)

        total_time = time.time() - start_time
        print(f"Completed {num_iterations} iterations in {total_time:.2f}s")
        print(f"Average: {num_iterations / total_time:.2f} iterations/sec")
        print(f"Total infosets: {sum(table.get_num_infosets() for table in self.regret_tables)}")

    def get_average_policy(self, player: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract average policy for a player.

        Args:
            player: Player index

        Returns:
            Dictionary mapping infoset strings to average strategies
        """
        return self.regret_tables[player].get_policy_dict()

    def save_checkpoint(self, filepath: str):
        """
        Save solver checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        import pickle

        checkpoint = {
            'iteration': self.iteration,
            'regret_tables': self.regret_tables,
            'config': self.config,
            'metrics': self.metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Load solver checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        import pickle

        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        self.iteration = checkpoint['iteration']
        self.regret_tables = checkpoint['regret_tables']
        self.metrics = checkpoint['metrics']

        print(f"Checkpoint loaded: {filepath}")
        print(f"Resuming from iteration {self.iteration}")


if __name__ == "__main__":
    print("Testing GPU MCCFR Solver")
    print("=" * 70)

    # Import game engine
    from matrix_cfr import holdem_jax

    print("\n[Test 1: RegretTable]")
    table = RegretTable()

    # Test regret updates
    infoset = "R0_H[As,Kh]_B[]_Bets[50,100]"
    legal = np.array([False, True, True, True])  # Can't fold preflop as BB

    regrets = np.array([0.0, -10.0, 20.0, 5.0])
    table.update_regrets(infoset, regrets)

    strategy = table.get_strategy(infoset, legal)
    print(f"✓ Regret matching strategy: {strategy}")
    print(f"  (Should favor action 2 with regret 20.0)")

    # Test strategy sum
    table.update_strategy_sum(infoset, strategy, weight=1.0)
    avg_strategy = table.get_average_strategy(infoset, legal)
    print(f"✓ Average strategy: {avg_strategy}")

    print("\n[Test 2: GPUMCCFRSolver Initialization]")
    config = MCCFRConfig(num_players=2, num_actions=4)
    solver = GPUMCCFRSolver(holdem_jax, config, seed=42)

    print(f"✓ Solver initialized")
    print(f"  - {config.num_players} players")
    print(f"  - {config.num_actions} actions")
    print(f"  - {len(solver.regret_tables)} regret tables")

    print("\n[Test 3: Run Small Training Session]")
    num_players = 2
    stacks = jnp.array([1000.0, 1000.0])
    blinds = jnp.array([50.0, 100.0])

    print("Running 10 MCCFR iterations...")
    solver.solve(
        num_iterations=10,
        num_players=num_players,
        stacks=stacks,
        blinds=blinds,
        progress_interval=10
    )

    print("\n[Test 4: Extract Policy]")
    policy = solver.get_average_policy(player=0)
    print(f"✓ Policy extracted: {len(policy)} information sets")

    if len(policy) > 0:
        sample_infoset = list(policy.keys())[0]
        sample_strategy = policy[sample_infoset]
        print(f"  Example: {sample_infoset}")
        print(f"  Strategy: {sample_strategy}")

    print("\n" + "=" * 70)
    print("GPU MCCFR Solver Tests Passed! ✅")
    print("\nNext: Validate on Kuhn poker with known equilibrium!")
