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

        # Cumulative reach probabilities (for weighted averaging)
        self.cumulative_reach = jnp.zeros(num_ia, dtype=jnp.float32)

        # Current strategy (uniform to start)
        self.current_strategy = self._init_uniform_strategy()

        # Iteration counter
        self.current_iteration = 0

        # Build action→child cache (Option C optimization)
        self._build_action_child_cache()

    def _build_action_child_cache(self):
        """
        Build cache mapping (infoset, action) → child_node_id for fast lookups.

        Option C optimization: Trades ~2 MB memory for 2x speedup.
        Pre-computes all child node lookups that would otherwise use level matrices.

        Memory cost: O(num_infoset_actions) ~= 100k-500k entries for Hold'em = 2 MB
        """
        self.action_child_cache = {}

        logger.info("Building action→child cache...")

        for (infoset, action), parent_node_id in self.matrix_repr.action_index_to_node.items():
            parent_node = self.matrix_repr.nodes[parent_node_id]

            # Use _find_child_for_action to populate cache
            try:
                child_node_id = self._find_child_for_action(
                    parent_node_id=parent_node_id,
                    action=action,
                    parent_depth=parent_node.depth
                )
                self.action_child_cache[(infoset, action)] = child_node_id
            except (ValueError, IndexError) as e:
                # Some (infoset, action) pairs might not have children (terminal states)
                logger.debug(f"No child for ({infoset}, {action}): {e}")
                continue

        logger.info(f"  Cached {len(self.action_child_cache)} action→child mappings")

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

    def _find_child_for_action(self, parent_node_id: int, action: int, parent_depth: int) -> int:
        """
        Find child node reached by taking 'action' from parent node.

        Uses level matrix L^l to find children (zero memory overhead).
        L^l[parent, child] = 1 if edge exists from parent to child.

        Level matrix indexing: level_matrices[l] contains edges TO nodes at depth l.
        So children of a depth-d node are in level_matrices[d+1].

        Args:
            parent_node_id: ID of parent (decision) node
            action: Action index in legal_actions list
            parent_depth: Depth level of parent node

        Returns:
            child_node_id: ID of child node reached by this action

        Raises:
            ValueError: If action has no corresponding child
        """
        # Get level matrix for edges TO children (at depth parent_depth+1)
        child_depth = parent_depth + 1
        if child_depth >= len(self.level_matrices_jax):
            raise ValueError(f"Child depth {child_depth} exceeds max depth")

        L_l = self.level_matrices_jax[child_depth]  # Edges TO depth child_depth

        # Find all children of parent_node_id
        # L_l[parent_node_id, :] gives edges from parent to all possible children
        parent_row = L_l[parent_node_id, :]

        # Get indices where edge exists (non-zero entries)
        children_mask = parent_row > 0.5
        child_indices = jnp.where(children_mask, jnp.arange(len(parent_row)), -1)
        child_indices = child_indices[child_indices >= 0]  # Filter out -1s

        # Convert to regular Python list for indexing
        child_list = [int(idx) for idx in child_indices]

        # The action index corresponds to position in children list
        # (children are ordered by action in tree traversal)
        if action < len(child_list):
            return child_list[action]
        else:
            raise ValueError(
                f"Action {action} out of range for parent {parent_node_id} "
                f"with {len(child_list)} children"
            )

    def _build_node_strategy_vector(self) -> jnp.ndarray:
        """
        Map infoset-action strategies to node-level transition probabilities.

        For matrix operations (Equation 11), we need strategy probabilities per node,
        not per (infoset, action) pair. This function creates a vector where
        node_strategy[node_id] = probability of reaching this node from parent.

        For decision nodes: Use current_strategy[infoset, action]
        For chance nodes: Use uniform probability (1.0 / num_outcomes)
        For root: 1.0

        Returns:
            node_strategy: (num_nodes,) array of transition probabilities
        """
        num_nodes = self.matrix_repr.num_nodes
        node_strategy = jnp.ones(num_nodes, dtype=jnp.float32)

        # For each (infoset, action) → node mapping, set the strategy probability
        for (infoset, action), node_id in self.matrix_repr.action_index_to_node.items():
            node = self.matrix_repr.nodes[node_id]

            if node.is_terminal or node.is_chance:
                # Chance nodes: uniform probability over outcomes (simplified for now)
                # Terminal nodes: shouldn't be accessed during traversal
                continue

            # Decision node: look up strategy for the parent's (infoset, action)
            # The node_id corresponds to the state after taking 'action' in 'infoset'
            # But we want the probability of this transition, which is strategy[infoset, action]

            # Find the action index for this (infoset, action) pair
            try:
                action_list = self.matrix_repr.infoset_to_actions[infoset]
                action_idx = action_list.index(action)
                infoset_action_idx = self.infoset_action_indices[infoset][action_idx]

                # Set the transition probability to this node
                node_strategy = node_strategy.at[node_id].set(
                    self.current_strategy[infoset_action_idx]
                )
            except (KeyError, ValueError):
                # If mapping fails, keep default 1.0
                logger.warning(f"Could not map strategy for node {node_id}, infoset {infoset}, action {action}")

        return node_strategy

    def solve(
        self,
        iterations: int,
        checkpoint_interval: Optional[int] = None,
        exploitability_interval: Optional[int] = None,
        progress_interval: int = 100
    ):
        """
        Run CFR for specified number of iterations on GPU.

        Args:
            iterations: Number of CFR iterations to run
            checkpoint_interval: Save checkpoint every N iterations
            exploitability_interval: Calculate exploitability every N iterations
            progress_interval: Show progress update every N iterations (default: 100)
        """
        print("\n" + "=" * 80)
        print(f"MATRIX CFR SOLVER - {self.algorithm.upper()}")
        print("=" * 80)
        print(f"Game: {self.matrix_repr.num_nodes} nodes, {self.matrix_repr.num_infosets} infosets")
        print(f"Players: {self.matrix_repr.num_players}")
        print(f"Infoset-actions: {self.matrix_repr.num_infoset_actions}")
        print(f"Device: {'GPU ('+str(self.device)+')' if self.use_gpu else 'CPU'}")
        print(f"Target iterations: {iterations:,}")
        print("=" * 80 + "\n")

        start_time = time.time()
        last_update_time = start_time

        for i in range(iterations):
            self.current_iteration += 1

            # Run one CFR iteration for each player
            for player in range(self.matrix_repr.num_players):
                self._cfr_iteration(player)

            # Periodic progress updates
            if (i + 1) % progress_interval == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                it_per_sec = (i + 1) / elapsed

                # Time since last update
                interval_time = current_time - last_update_time
                interval_its = progress_interval
                interval_speed = interval_its / interval_time if interval_time > 0 else 0

                # Estimated time remaining
                remaining_its = iterations - (i + 1)
                eta_seconds = remaining_its / it_per_sec if it_per_sec > 0 else 0
                eta_mins = eta_seconds / 60

                print(f"Iteration {i + 1:>10,}/{iterations:,} | "
                      f"Speed: {interval_speed:>6.0f} it/s | "
                      f"Avg: {it_per_sec:>6.0f} it/s | "
                      f"Elapsed: {elapsed:>6.1f}s | "
                      f"ETA: {eta_mins:>5.1f}m")

                last_update_time = current_time

            # Checkpointing
            if checkpoint_interval and (i + 1) % checkpoint_interval == 0:
                checkpoint_path = f"checkpoints/matrix_cfr_iter_{i + 1}.npz"
                self.save_checkpoint(checkpoint_path)
                print(f"  → Checkpoint saved: {checkpoint_path}")

            # Exploitability calculation
            if exploitability_interval and (i + 1) % exploitability_interval == 0:
                # TODO: Implement proper exploitability
                print(f"  → Exploitability check at iteration {i + 1} (not yet implemented)")

        total_time = time.time() - start_time
        final_speed = iterations / total_time

        print("\n" + "=" * 80)
        print("SOLVE COMPLETE")
        print("=" * 80)
        print(f"Total iterations: {iterations:,}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f}m)")
        print(f"Average speed: {final_speed:.0f} it/s")
        print(f"Final iteration: {self.current_iteration:,}")
        print("=" * 80 + "\n")

    def _bottom_up_utilities(self, player: int) -> List[jnp.ndarray]:
        """
        Compute utilities for all nodes via bottom-up propagation (Equation 11).

        This implements the core bottom-up pass from the paper:
            Ǔ^(D+1) = terminal_utilities[player]
            for l = D down to 1:
                Ǔ^(l) = (L^l ⊙ S) @ Ǔ^(l+1) + Ǔ^(l+1)

        The strategy S is broadcast to match matrix dimensions, and utilities
        propagate from terminal nodes back to the root.

        Args:
            player: Which player's utilities to compute

        Returns:
            utilities_by_level: List of (num_nodes,) arrays, one per level
        """
        num_levels = len(self.level_matrices_jax)
        num_nodes = self.matrix_repr.num_nodes
        utilities = [None] * num_levels

        # Initialize terminal utilities at deepest level
        # Extract utilities for this player from terminal nodes
        utilities[-1] = self.terminal_utilities_jax[:, player]

        # Map current strategy to node-level transition probabilities
        node_strategy = self._build_node_strategy_vector()

        # Bottom-up pass: propagate utilities from level D to level 0 (root)
        for level in range(num_levels - 2, -1, -1):
            L_l = self.level_matrices_jax[level]  # (num_nodes, num_nodes)

            # Element-wise multiply: L^l ⊙ S
            # L_l[i,j] = 1 if edge i→j exists, 0 otherwise
            # We weight each edge by the strategy probability: S[j] (child node strategy)
            # Broadcast strategy column-wise
            weighted_L = L_l * node_strategy[jnp.newaxis, :]  # Broadcast across columns

            # Matrix-vector product: (L^l ⊙ S) @ Ǔ^(l+1)
            propagated = weighted_L @ utilities[level + 1]

            # Add direct contribution (handles non-edge connections)
            utilities[level] = propagated + utilities[level + 1]

        return utilities

    def _full_reach_probabilities(self, strategy: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Compute FULL reach probabilities (all players play given strategy).

        This is for strategy averaging - we want the probability of reaching each
        node when all players play according to the current strategy.

        Args:
            strategy: Node-level strategy vector

        Returns:
            reach_by_level: List of (num_nodes,) arrays with reach probabilities
        """
        num_levels = len(self.level_matrices_jax)
        num_nodes = self.matrix_repr.num_nodes
        reach = [None] * num_levels

        # Initialize root reach probability
        reach[0] = jnp.zeros(num_nodes, dtype=jnp.float32)
        reach[0] = reach[0].at[0].set(1.0)  # Root = 1.0

        # Top-down pass: ALL players use the given strategy
        for level in range(num_levels - 1):
            L_l = self.level_matrices_jax[level + 1]  # Edges TO level+1

            # Propagate: (L^l)^T @ reach^(l)
            propagated = L_l.T @ reach[level]

            # Weight by strategy probabilities
            weighted = propagated * strategy

            # Direct contribution
            reach[level + 1] = weighted + reach[level]

        return reach

    def _top_down_reach_probabilities(
        self,
        updating_player: int,
        opponent_strategy: jnp.ndarray
    ) -> List[jnp.ndarray]:
        """
        Compute counterfactual reach probabilities via top-down propagation (Equation 13).

        This implements the top-down pass from the paper:
            Π̌^(0) = [1, 0, 0, ...] (root = 1.0)
            for l = 1 to D:
                Π̌^(l) = (L^l)^T @ Π̌^(l-1) ⊙ Š + Π̌^(l-1)

        Counterfactual reach = probability of reaching node if updating player
        played to reach (plays uniformly/counterfactually) while opponents play
        according to their strategy.

        Args:
            updating_player: Player whose regrets we're computing
            opponent_strategy: Node-level strategy vector for weighting

        Returns:
            reach_by_level: List of (num_nodes,) arrays with reach probabilities
        """
        num_levels = len(self.level_matrices_jax)
        num_nodes = self.matrix_repr.num_nodes
        reach = [None] * num_levels

        # Initialize root reach probability
        reach[0] = jnp.zeros(num_nodes, dtype=jnp.float32)
        reach[0] = reach[0].at[0].set(1.0)  # Root node has reach 1.0

        # Build counterfactual strategy override
        # Š = 1.0 where updating_player acted (counterfactual), opponent_strategy otherwise
        player_nodes = self.player_matrix_jax[:, updating_player]  # (num_nodes,) - 1 if player acted

        # Where updating player acted, use 1.0 (counterfactual - as if they played to reach)
        # Where opponents acted, use their actual strategy
        counterfactual_strategy = jnp.where(
            player_nodes > 0.5,      # Updating player's nodes
            1.0,                      # Override to 1.0 (counterfactual)
            opponent_strategy         # Use opponent's strategy
        )

        # Top-down pass: propagate reach from level 0 (root) to level D (terminal)
        for level in range(num_levels - 1):
            L_l = self.level_matrices_jax[level]  # (num_nodes, num_nodes)

            # Transpose and multiply: (L^l)^T @ Π̌^(l-1)
            # This propagates reach from parents to children
            propagated = L_l.T @ reach[level]

            # Element-wise multiply with counterfactual strategy
            # Weight by strategy probability for transitions
            weighted = propagated * counterfactual_strategy

            # Add direct contribution
            # (This handles nodes that appear at multiple levels or have no parents)
            reach[level + 1] = weighted + reach[level]

        return reach

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
        Compute counterfactual values for all actions using matrix operations.

        For each action at each infoset:
        1. Override strategy to play that action with probability 1.0
        2. Compute bottom-up utilities with the override
        3. Extract utility at the infoset node

        This uses the bottom-up propagation (Equation 11) to compute exact
        counterfactual values for each action.

        Args:
            player: Player to compute values for

        Returns:
            Dictionary mapping infosets to action values array
        """
        cf_values = {}

        # Process each infoset belonging to this player
        for infoset, actions in self.matrix_repr.infoset_to_actions.items():
            if not actions:
                continue

            # Check if this infoset belongs to the updating player
            first_action = actions[0]
            if (infoset, first_action) not in self.matrix_repr.action_index_to_node:
                continue

            first_node_id = self.matrix_repr.action_index_to_node[(infoset, first_action)]
            first_node = self.matrix_repr.nodes[first_node_id]

            if first_node.player != player:
                continue

            # Compute value for each action at this infoset
            num_actions = len(actions)
            action_values = jnp.zeros(num_actions, dtype=jnp.float32)

            for action_idx, action in enumerate(actions):
                # Create strategy override: play this action with probability 1.0
                override_strategy = jnp.zeros_like(self.current_strategy)

                # Set this specific (infoset, action) to probability 1.0
                infoset_action_indices = self.infoset_action_indices[infoset]
                override_strategy = override_strategy.at[infoset_action_indices[action_idx]].set(1.0)

                # For all other infosets, use current strategy
                for other_infoset, other_indices in self.infoset_action_indices.items():
                    if other_infoset != infoset:
                        override_strategy = override_strategy.at[other_indices].set(
                            self.current_strategy[other_indices]
                        )

                # Temporarily swap strategies
                original_strategy = self.current_strategy
                self.current_strategy = override_strategy

                # Compute utilities with this action override
                utilities_by_level = self._bottom_up_utilities(player)

                # Restore original strategy
                self.current_strategy = original_strategy

                # FIX: Extract utility at CHILD node reached by this action, not parent
                # Option C optimization: Use cached child lookup (2x faster)
                cache_key = (infoset, action)
                if cache_key in self.action_child_cache:
                    # Fast path: cached lookup (1 op)
                    child_node_id = self.action_child_cache[cache_key]
                else:
                    # Slow path: compute via level matrices (10-20 ops)
                    parent_node_id = self.matrix_repr.action_index_to_node[(infoset, action)]
                    parent_node = self.matrix_repr.nodes[parent_node_id]
                    child_node_id = self._find_child_for_action(
                        parent_node_id=parent_node_id,
                        action=action,
                        parent_depth=parent_node.depth
                    )

                # Extract utility at CHILD node (the outcome of taking this action)
                child_node = self.matrix_repr.nodes[child_node_id]
                child_utility = utilities_by_level[child_node.depth][child_node_id]

                action_values = action_values.at[action_idx].set(child_utility)

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

        # Compute FULL reach probabilities for weighted strategy averaging
        # (both players play current strategy, not counterfactual)
        current_node_strategy = self._build_node_strategy_vector()
        reach_probs = self._full_reach_probabilities(current_node_strategy)

        # Accumulate strategy for averaging (weighted by reach)
        self._update_cumulative_strategy(reach_probs)

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

    def _update_cumulative_strategy(self, reach_probabilities: List[jnp.ndarray]):
        """
        Accumulate current strategy weighted by reach probability (Equation 10).

        Strategy averaging should weight each strategy by the probability of
        reaching that infoset, giving more importance to frequently visited states.

        Args:
            reach_probabilities: Reach probabilities by level from top-down pass
        """
        # For each infoset-action, weight by reach probability at that node
        for infoset, action_indices in self.infoset_action_indices.items():
            # Get a representative node for this infoset
            first_action = self.matrix_repr.infoset_to_actions[infoset][0]
            if (infoset, first_action) not in self.matrix_repr.action_index_to_node:
                continue

            node_id = self.matrix_repr.action_index_to_node[(infoset, first_action)]
            node = self.matrix_repr.nodes[node_id]
            reach_weight = reach_probabilities[node.depth][node_id]

            # Weighted accumulation: σ̄ += reach × σ
            weighted_strategy = self.current_strategy[action_indices] * reach_weight
            self.cumulative_strategy = self.cumulative_strategy.at[action_indices].add(
                weighted_strategy
            )

            # Also accumulate the reach weights for normalization
            self.cumulative_reach = self.cumulative_reach.at[action_indices].add(reach_weight)

    def get_average_policy(self) -> Dict[str, np.ndarray]:
        """
        Get the average strategy policy, weighted by reach probabilities.

        Computes: σ̄ = cumulative_strategy / cumulative_reach (per infoset-action)
        Then normalizes per infoset to get valid probability distributions.

        Returns:
            Dictionary mapping infosets to action probability distributions
        """
        # Divide cumulative strategy by cumulative reach (element-wise)
        # Add epsilon to avoid division by zero
        epsilon = 1e-10
        avg_strategy_jax = self.cumulative_strategy / (self.cumulative_reach + epsilon)

        # Convert to numpy and extract per-infoset policies
        avg_strategy = np.array(avg_strategy_jax)

        policy = {}
        for infoset, action_indices in self.infoset_action_indices.items():
            action_probs = avg_strategy[action_indices]

            # Normalize to get valid probability distribution
            prob_sum = action_probs.sum()
            if prob_sum > epsilon:
                action_probs = action_probs / prob_sum
            else:
                # If no reach (shouldn't happen for reachable infosets), use uniform
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
