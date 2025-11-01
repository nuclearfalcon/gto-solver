"""
Matrix-based GPU CFR Solver

Implements Counterfactual Regret Minimization using matrix operations on GPU.
Based on the approach from arXiv:2408.14778v5.

Instead of recursive tree traversal, CFR iterations are implemented as:
- Matrix-vector multiplications for reach probability propagation
- Sparse matrix operations for regret updates
- GPU-accelerated linear algebra using JAX

Key insight: Each CFR iteration can be expressed as a series of dense and sparse
matrix/vector operations, which are embarrassingly parallel and perfect for GPU.

Usage:
    from matrix_cfr import MatrixCFRSolver

    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')
    solver.solve(iterations=100000)
    policy = solver.get_average_policy()
"""

import pyspiel
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
import time

# JAX imports
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import sparse as jsparse

from .game_to_matrix import GameTreeConverter, MatrixRepresentation

logger = logging.getLogger(__name__)


class MatrixCFRSolver:
    """
    GPU-accelerated CFR solver using matrix operations.

    This solver reimplements CFR algorithms (vanilla, MCCFR, DCFR) using
    matrix operations that can be efficiently parallelized on GPU.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        algorithm: str = 'vanilla_cfr',
        use_gpu: bool = True
    ):
        """
        Initialize matrix-based CFR solver.

        Args:
            game: OpenSpiel game instance
            algorithm: CFR variant ('vanilla_cfr', 'mccfr', 'dcfr')
            use_gpu: Whether to use GPU acceleration (auto-detected if available)
        """
        self.game = game
        self.algorithm = algorithm
        self.use_gpu = use_gpu

        # Initialize JAX/GPU first
        self._init_gpu()

        # Convert game tree to matrix representation
        logger.info("Converting game tree to matrix representation...")
        self.converter = GameTreeConverter(game)
        self.matrix_repr = self.converter.build_matrices()

        # Convert matrices to JAX arrays on GPU
        logger.info("Transferring matrices to GPU...")
        self._convert_to_jax()

        # Initialize CFR state on GPU
        logger.info("Initializing CFR state...")
        self._init_cfr_state()

        logger.info(f"Matrix CFR solver ready: {self.matrix_repr.num_nodes} nodes, "
                   f"{self.matrix_repr.num_infoset_actions} infoset-actions on "
                   f"{'GPU' if self.use_gpu else 'CPU'}")

    def _init_gpu(self):
        """
        Initialize JAX and check GPU availability.
        """
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']

        if gpu_devices and self.use_gpu:
            logger.info(f"GPU detected: {gpu_devices[0].device_kind}")
            self.device = gpu_devices[0]
        else:
            logger.warning("No GPU detected or use_gpu=False, falling back to CPU")
            self.device = jax.devices('cpu')[0]
            self.use_gpu = False

    def _convert_to_jax(self):
        """
        Convert scipy sparse matrices to JAX arrays and transfer to GPU.

        Note: JAX sparse support is experimental, so we'll use dense arrays
        for now and optimize later with proper sparse operations.
        """
        # For now, convert sparse matrices to dense for simplicity
        # TODO: Use JAX sparse matrices when more stable

        self.level_matrices_jax = [
            jnp.array(L.toarray(), dtype=jnp.float32)
            for L in self.matrix_repr.level_matrices
        ]

        self.infoset_action_matrix_jax = jnp.array(
            self.matrix_repr.infoset_action_to_node_matrix.toarray(),
            dtype=jnp.float32
        )

        self.player_matrix_jax = jnp.array(
            self.matrix_repr.player_matrix,
            dtype=jnp.float32
        )

        self.terminal_utilities_jax = jnp.array(
            self.matrix_repr.terminal_utilities_matrix,
            dtype=jnp.float32
        )

    def _init_cfr_state(self):
        """
        Initialize CFR state vectors (regrets, strategies) on GPU.
        """
        num_ia = self.matrix_repr.num_infoset_actions

        # Build infoset action index mapping FIRST (needed for uniform strategy)
        self._build_infoset_indexing()

        # Cumulative regrets (one per infoset-action pair)
        self.cumulative_regrets = jnp.zeros(num_ia, dtype=jnp.float32)

        # Cumulative strategy (for averaging)
        self.cumulative_strategy = jnp.zeros(num_ia, dtype=jnp.float32)

        # Current strategy (uniform to start)
        self.current_strategy = self._init_uniform_strategy()

        # Iteration counter
        self.current_iteration = 0

    def _build_infoset_indexing(self):
        """
        Build indexing structures for efficient regret matching per infoset.

        Creates mappings from infosets to their action indices in the
        cumulative regrets/strategy vectors.
        """
        self.infoset_action_indices = {}  # infoset -> list of indices in regret/strategy vectors

        # Assign index to each (infoset, action) pair (must match game_to_matrix.py)
        ia_to_index = {}
        ia_index = 0
        for infoset in sorted(self.matrix_repr.infoset_to_actions.keys()):
            action_indices = []
            for action in self.matrix_repr.infoset_to_actions[infoset]:
                ia_to_index[(infoset, action)] = ia_index
                action_indices.append(ia_index)
                ia_index += 1
            self.infoset_action_indices[infoset] = jnp.array(action_indices, dtype=jnp.int32)

    def _init_uniform_strategy(self) -> jnp.ndarray:
        """
        Initialize uniform strategy for each infoset.

        Returns:
            JAX array of strategy probabilities (one per infoset-action)
        """
        strategy = jnp.zeros(self.matrix_repr.num_infoset_actions, dtype=jnp.float32)

        for infoset, action_indices in self.infoset_action_indices.items():
            num_actions = len(action_indices)
            uniform_prob = 1.0 / num_actions
            strategy = strategy.at[action_indices].set(uniform_prob)

        return strategy

    def solve(
        self,
        iterations: int,
        checkpoint_interval: Optional[int] = None,
        exploitability_interval: Optional[int] = None
    ):
        """
        Run CFR for specified number of iterations on GPU.

        Args:
            iterations: Number of CFR iterations to run
            checkpoint_interval: Save checkpoint every N iterations
            exploitability_interval: Calculate exploitability every N iterations
        """
        logger.info(f"Starting {self.algorithm} solver for {iterations} iterations on {'GPU' if self.use_gpu else 'CPU'}")

        start_time = time.time()

        for i in range(iterations):
            self.current_iteration += 1

            # Run one CFR iteration for each player
            for player in range(self.matrix_repr.num_players):
                self._cfr_iteration(player)

            # Periodic logging
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                it_per_sec = (i + 1) / elapsed
                logger.info(f"Iteration {i + 1}/{iterations} ({it_per_sec:.0f} it/s)")

            # TODO: Implement checkpointing
            # TODO: Implement exploitability calculation

        total_time = time.time() - start_time
        logger.info(f"Completed {iterations} iterations in {total_time:.2f}s ({iterations/total_time:.0f} it/s)")

    def _cfr_iteration(self, player: int):
        """
        Perform one CFR iteration for a player using matrix operations on GPU.

        This implements the three-phase algorithm from the paper:
        1. Tree traversal (compute utilities and reach probabilities)
        2. Strategy averaging
        3. Regret updates

        Args:
            player: Player index to update regrets for
        """
        # Phase 1: Compute counterfactual values and reach probabilities
        # This is a simplified version - full implementation would do level-by-level processing
        cf_values = self._compute_counterfactual_values(player)

        # Phase 2 & 3: Update regrets and strategy
        self._update_regrets_and_strategy(player, cf_values)

    def _compute_counterfactual_values(self, player: int) -> Dict[str, jnp.ndarray]:
        """
        Compute counterfactual values for all infosets of a player.

        This is a simplified version. Full implementation would use the
        level-by-level matrix operations from the paper.

        Args:
            player: Player to compute values for

        Returns:
            Dictionary mapping infosets to action values
        """
        # For now, use a simple traversal
        # TODO: Implement full matrix-based traversal with reach probabilities

        cf_values = {}

        for infoset in self.matrix_repr.infoset_to_actions.keys():
            # Get the node for this infoset
            if (infoset, self.matrix_repr.infoset_to_actions[infoset][0]) in self.matrix_repr.action_index_to_node:
                node_id = self.matrix_repr.action_index_to_node[(infoset, self.matrix_repr.infoset_to_actions[infoset][0])]
                node = self.matrix_repr.nodes[node_id]

                if node.player == player:
                    # Compute value for each action (simplified)
                    num_actions = len(self.matrix_repr.infoset_to_actions[infoset])
                    action_values = jnp.zeros(num_actions, dtype=jnp.float32)

                    # Placeholder: In full implementation, would traverse from this node
                    # using matrix operations to compute true counterfactual values
                    # For now, use random values to test the infrastructure
                    action_values = jnp.ones(num_actions, dtype=jnp.float32) * 0.1

                    cf_values[infoset] = action_values

        return cf_values

    def _update_regrets_and_strategy(self, player: int, cf_values: Dict[str, jnp.ndarray]):
        """
        Update cumulative regrets and strategy based on counterfactual values.

        Args:
            player: Player being updated
            cf_values: Counterfactual values per infoset
        """
        # Update regrets for this player's infosets
        for infoset, action_values in cf_values.items():
            action_indices = self.infoset_action_indices[infoset]

            # Compute strategy value (expected value under current strategy)
            current_probs = self.current_strategy[action_indices]
            strategy_value = jnp.sum(current_probs * action_values)

            # Instantaneous regrets (how much better each action is than current strategy)
            instant_regrets = action_values - strategy_value

            # Accumulate regrets
            self.cumulative_regrets = self.cumulative_regrets.at[action_indices].add(instant_regrets)

        # Update strategy via regret matching
        self.current_strategy = self._regret_matching()

        # Accumulate strategy for averaging
        self._update_cumulative_strategy()

    def _regret_matching(self) -> jnp.ndarray:
        """
        Convert cumulative regrets to strategy using regret matching.

        For each infoset:
        - Positive regrets → proportional probability
        - All non-positive → uniform distribution

        Returns:
            New strategy (JAX array)
        """
        new_strategy = jnp.zeros_like(self.current_strategy)

        for infoset, action_indices in self.infoset_action_indices.items():
            regrets = self.cumulative_regrets[action_indices]

            # Positive regrets only
            positive_regrets = jnp.maximum(regrets, 0.0)
            regret_sum = jnp.sum(positive_regrets)

            if regret_sum > 0:
                # Proportional to positive regrets
                probs = positive_regrets / regret_sum
            else:
                # Uniform if no positive regrets
                probs = jnp.ones(len(action_indices)) / len(action_indices)

            new_strategy = new_strategy.at[action_indices].set(probs)

        return new_strategy

    def _update_cumulative_strategy(self):
        """
        Accumulate current strategy for averaging.

        In vanilla CFR, we simply add the current strategy.
        """
        # Reach probability weight would go here in full implementation
        # For now, just accumulate uniformly
        self.cumulative_strategy += self.current_strategy

    def get_average_policy(self) -> Dict[str, np.ndarray]:
        """
        Get the average strategy policy.

        Returns:
            Dictionary mapping infosets to action probability distributions
        """
        # Normalize cumulative strategy
        strategy_sum = jnp.sum(self.cumulative_strategy)

        if strategy_sum > 0:
            avg_strategy_jax = self.cumulative_strategy / strategy_sum
        else:
            # If no strategy accumulated (shouldn't happen), return uniform
            logger.warning("No strategy accumulated, returning uniform")
            avg_strategy_jax = self._init_uniform_strategy()

        # Convert to numpy and extract per-infoset policies
        avg_strategy = np.array(avg_strategy_jax)

        policy = {}
        for infoset, action_indices in self.infoset_action_indices.items():
            action_probs = avg_strategy[action_indices]

            # Normalize (should already be normalized, but ensure it)
            prob_sum = action_probs.sum()
            if prob_sum > 0:
                action_probs = action_probs / prob_sum
            else:
                # Uniform if all zero
                action_probs = np.ones(len(action_indices)) / len(action_indices)

            policy[infoset] = action_probs

        return policy

    def get_strategy_dict(self) -> Dict[str, Dict[int, float]]:
        """
        Get average strategy in OpenSpiel-compatible format.

        Returns:
            Dict mapping infoset -> {action: probability}
        """
        avg_policy = self.get_average_policy()

        strategy_dict = {}
        for infoset, action_probs in avg_policy.items():
            actions = self.matrix_repr.infoset_to_actions[infoset]
            strategy_dict[infoset] = {
                action: float(prob)
                for action, prob in zip(actions, action_probs)
            }

        return strategy_dict

    def calculate_exploitability(self, sampled: bool = True):
        """
        Calculate exploitability of current policy.

        Args:
            sampled: Whether to use sampled exploitability (recommended)

        Returns:
            Exploitability value
        """
        # TODO: Implement full exploitability calculation
        # For now, return placeholder
        logger.warning("Exploitability calculation not yet implemented")
        return 0.0

    def save_checkpoint(self, filepath: str):
        """
        Save solver state to disk.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'cumulative_regrets': np.array(self.cumulative_regrets),
            'cumulative_strategy': np.array(self.cumulative_strategy),
            'current_strategy': np.array(self.current_strategy),
            'current_iteration': self.current_iteration,
        }

        np.savez(filepath, **checkpoint)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Load solver state from disk.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = np.load(filepath)

        self.cumulative_regrets = jnp.array(checkpoint['cumulative_regrets'])
        self.cumulative_strategy = jnp.array(checkpoint['cumulative_strategy'])
        self.current_strategy = jnp.array(checkpoint['current_strategy'])
        self.current_iteration = int(checkpoint['current_iteration'])

        logger.info(f"Checkpoint loaded from {filepath} (iteration {self.current_iteration})")


# Helper functions for creating solvers

def create_matrix_solver(
    game: pyspiel.Game,
    algorithm: str = 'vanilla_cfr',
    use_gpu: bool = True
) -> MatrixCFRSolver:
    """
    Create a matrix-based CFR solver.

    Args:
        game: OpenSpiel game instance
        algorithm: CFR variant ('vanilla_cfr', 'mccfr', 'dcfr')
        use_gpu: Whether to use GPU acceleration

    Returns:
        MatrixCFRSolver instance
    """
    return MatrixCFRSolver(game, algorithm=algorithm, use_gpu=use_gpu)


# TODO: Add MCCFR-specific subclass
# TODO: Add DCFR-specific subclass with α, β, γ parameters
# TODO: Add mixed-precision (FP16) support
# TODO: Add multi-GPU support
# TODO: Implement full matrix-based tree traversal (level-by-level processing from paper)
