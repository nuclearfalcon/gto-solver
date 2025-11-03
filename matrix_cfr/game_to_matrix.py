"""
Game Tree to Matrix Converter

Converts OpenSpiel game trees into sparse matrix representations suitable for
GPU-accelerated CFR computation.

This module implements the core transformation described in arXiv:2408.14778v5:
instead of recursive tree traversal, we represent the game as:
- Sparse transition matrices (state × action → next state)
- Reach probability matrices
- Utility/payoff vectors

The matrices are optimized for GPU operations using JAX sparse matrices.

Usage:
    converter = GameTreeConverter(game)
    matrices = converter.build_matrices()
    # matrices contains transition matrices, reach prob matrices, utility vectors
"""

import pyspiel
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
import logging
from dataclasses import dataclass
from scipy import sparse
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class GameNode:
    """
    Represents a single node in the game tree.

    This is an internal data structure used during tree traversal.
    """
    node_id: int
    depth: int
    parent_id: int  # -1 for root
    player: int  # -1 for chance, -2 for terminal
    infoset: Optional[str]  # None for chance/terminal
    legal_actions: List[int]
    is_terminal: bool
    is_chance: bool
    terminal_utilities: Optional[np.ndarray]  # Only for terminal nodes
    history_str: str  # For debugging


class MatrixRepresentation(NamedTuple):
    """
    Container for all matrices and metadata needed for GPU CFR.
    """
    # Tree structure
    level_matrices: List[sparse.csr_matrix]  # One per depth level
    num_nodes: int
    num_levels: int
    num_players: int  # Number of players in the game

    # Information sets
    infoset_to_actions: Dict[str, List[int]]  # infoset -> list of action indices
    action_index_to_node: Dict[Tuple[str, int], int]  # (infoset, action) -> node_id
    num_infosets: int
    num_infoset_actions: int  # Total (infoset, action) pairs

    # Mappings (sparse matrices)
    infoset_action_to_node_matrix: sparse.csr_matrix  # M^(Q+,V)
    player_matrix: np.ndarray  # M^(V,I+) (dense)

    # Terminal utilities
    terminal_utilities_matrix: np.ndarray  # (num_nodes, num_players)

    # Node metadata (for debugging/validation)
    nodes: List[GameNode]
    node_depths: np.ndarray


class GameTreeConverter:
    """
    Converts OpenSpiel game tree to matrix representation for GPU CFR.

    This is the core component that enables matrix-based CFR by transforming
    the game tree into linear algebra operations.
    """

    def __init__(self, game: pyspiel.Game):
        """
        Initialize converter for a given game.

        Args:
            game: OpenSpiel game instance (e.g., Kuhn poker, Hold'em)
        """
        self.game = game
        self.num_players = game.num_players()

        # Will be populated by build_matrices()
        self.nodes: List[GameNode] = []
        self.node_id_counter = 0
        self.max_depth = 0

        logger.info(f"Initialized GameTreeConverter for {game.get_type().short_name}")

    def build_matrices(self, starting_state: Optional[pyspiel.State] = None) -> MatrixRepresentation:
        """
        Build all matrix representations of the game tree.

        This is where the magic happens - we traverse the game tree once
        and construct sparse matrices that represent all possible transitions.

        Args:
            starting_state: Phase 9 - Optional custom starting state for true pre-dealing.
                           If None, uses game.new_initial_state() (default behavior).

        Returns:
            MatrixRepresentation containing all matrices needed for GPU CFR
        """
        logger.info("Building matrix representation of game tree...")

        # Step 1: Traverse tree and collect all nodes
        if starting_state is not None:
            # Phase 9: Start from custom state (true pre-dealing)
            initial_state = starting_state.clone()  # Clone to avoid modifying original
        else:
            # Default: Start from game's initial state
            initial_state = self.game.new_initial_state()

        self._traverse_tree(initial_state, depth=0, parent_id=-1)

        logger.info(f"  Enumerated {len(self.nodes)} nodes at {self.max_depth + 1} levels")

        # Step 2: Build level matrices
        level_matrices = self._build_level_matrices()
        logger.info(f"  Built {len(level_matrices)} level matrices")

        # Step 3: Build infoset mappings
        infoset_to_actions, action_index_to_node = self._build_infoset_mappings()
        num_infosets = len(infoset_to_actions)
        num_infoset_actions = sum(len(actions) for actions in infoset_to_actions.values())
        logger.info(f"  Found {num_infosets} infosets with {num_infoset_actions} total actions")

        # Step 4: Build infoset-action to node matrix (M^(Q+,V))
        infoset_action_matrix = self._build_infoset_action_matrix(
            infoset_to_actions, action_index_to_node
        )
        logger.info(f"  Built infoset-action matrix: {infoset_action_matrix.shape}")

        # Step 5: Build player matrix (M^(V,I+))
        player_matrix = self._build_player_matrix()
        logger.info(f"  Built player matrix: {player_matrix.shape}")

        # Step 6: Build terminal utilities matrix
        terminal_utilities = self._build_terminal_utilities()
        logger.info(f"  Built terminal utilities: {terminal_utilities.shape}")

        # Step 7: Collect node depths
        node_depths = np.array([node.depth for node in self.nodes], dtype=np.int32)

        logger.info("Matrix representation complete!")

        return MatrixRepresentation(
            level_matrices=level_matrices,
            num_nodes=len(self.nodes),
            num_levels=self.max_depth + 1,
            num_players=self.num_players,
            infoset_to_actions=infoset_to_actions,
            action_index_to_node=action_index_to_node,
            num_infosets=num_infosets,
            num_infoset_actions=num_infoset_actions,
            infoset_action_to_node_matrix=infoset_action_matrix,
            player_matrix=player_matrix,
            terminal_utilities_matrix=terminal_utilities,
            nodes=self.nodes,
            node_depths=node_depths
        )

    def _traverse_tree(self, state: pyspiel.State, depth: int, parent_id: int):
        """
        Recursively traverse game tree to build complete node list.

        Args:
            state: Current game state
            depth: Current depth in tree
            parent_id: Node ID of parent (-1 for root)
        """
        # Create node for current state
        node_id = self.node_id_counter
        self.node_id_counter += 1

        # Update max depth
        if depth > self.max_depth:
            self.max_depth = depth

        # Determine node type and properties
        is_terminal = state.is_terminal()
        is_chance = state.is_chance_node()

        if is_terminal:
            player = -2  # Terminal node marker
            infoset = None
            legal_actions = []
            terminal_utilities = np.array(state.returns(), dtype=np.float32)
        elif is_chance:
            player = -1  # Chance node marker
            infoset = None
            legal_actions = [action for action, _ in state.chance_outcomes()]
            terminal_utilities = None
        else:
            player = state.current_player()
            infoset = state.information_state_string(player)
            legal_actions = state.legal_actions()
            terminal_utilities = None

        # Create and store node
        node = GameNode(
            node_id=node_id,
            depth=depth,
            parent_id=parent_id,
            player=player,
            infoset=infoset,
            legal_actions=legal_actions,
            is_terminal=is_terminal,
            is_chance=is_chance,
            terminal_utilities=terminal_utilities,
            history_str=state.history_str()
        )
        self.nodes.append(node)

        # Recursively traverse children
        if not is_terminal:
            for action in legal_actions:
                child_state = state.child(action)
                self._traverse_tree(child_state, depth + 1, node_id)

    def _build_level_matrices(self) -> List[sparse.csr_matrix]:
        """
        Build sparse level matrices L^(l) for each depth.

        Each L^(l) is a sparse matrix of shape (num_nodes, num_nodes) where
        L^(l)[parent, child] = 1 if child is at depth l and parent -> child.

        Returns:
            List of sparse CSR matrices, one per depth level
        """
        num_nodes = len(self.nodes)
        level_matrices = []

        for level in range(self.max_depth + 1):
            # Collect edges for this level (parent -> child where child at depth=level)
            row_indices = []
            col_indices = []

            for node in self.nodes:
                if node.depth == level and node.parent_id >= 0:
                    row_indices.append(node.parent_id)
                    col_indices.append(node.node_id)

            # Build sparse matrix
            if row_indices:
                data = np.ones(len(row_indices), dtype=np.float32)
                level_matrix = sparse.csr_matrix(
                    (data, (row_indices, col_indices)),
                    shape=(num_nodes, num_nodes)
                )
            else:
                # Empty level (shouldn't happen but handle gracefully)
                level_matrix = sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float32)

            level_matrices.append(level_matrix)

        return level_matrices

    def _build_infoset_mappings(self) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, int], int]]:
        """
        Build mappings from infosets to actions and (infoset, action) to nodes.

        Returns:
            (infoset_to_actions, action_index_to_node)
        """
        infoset_to_actions = defaultdict(set)
        action_index_to_node = {}

        # Collect all (infoset, action) pairs from decision nodes
        for node in self.nodes:
            if node.infoset is not None:  # Decision node (not chance/terminal)
                for action in node.legal_actions:
                    infoset_to_actions[node.infoset].add(action)

        # Convert sets to sorted lists for consistent indexing
        infoset_to_actions = {
            infoset: sorted(list(actions))
            for infoset, actions in infoset_to_actions.items()
        }

        # Build (infoset, action) -> node mapping
        # Note: Multiple nodes can have same (infoset, action) pair
        # We'll store the first occurrence (they should have same semantics)
        for node in self.nodes:
            if node.infoset is not None:
                for action in node.legal_actions:
                    key = (node.infoset, action)
                    if key not in action_index_to_node:
                        action_index_to_node[key] = node.node_id

        return infoset_to_actions, action_index_to_node

    def _build_infoset_action_matrix(
        self,
        infoset_to_actions: Dict[str, List[int]],
        action_index_to_node: Dict[Tuple[str, int], int]
    ) -> sparse.csr_matrix:
        """
        Build M^(Q+,V) matrix mapping (infoset, action) pairs to nodes.

        Matrix shape: (num_infoset_actions, num_nodes)
        M[ia_idx, node_idx] = 1 if infoset-action pair ia_idx corresponds to node node_idx

        Returns:
            Sparse CSR matrix
        """
        num_nodes = len(self.nodes)

        # Assign index to each (infoset, action) pair
        ia_to_index = {}
        ia_index = 0
        for infoset in sorted(infoset_to_actions.keys()):
            for action in infoset_to_actions[infoset]:
                ia_to_index[(infoset, action)] = ia_index
                ia_index += 1

        num_infoset_actions = len(ia_to_index)

        # Build sparse matrix
        row_indices = []
        col_indices = []

        for (infoset, action), node_id in action_index_to_node.items():
            ia_idx = ia_to_index[(infoset, action)]
            row_indices.append(ia_idx)
            col_indices.append(node_id)

        data = np.ones(len(row_indices), dtype=np.float32)
        matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(num_infoset_actions, num_nodes)
        )

        return matrix

    def _build_player_matrix(self) -> np.ndarray:
        """
        Build M^(V,I+) dense matrix indicating which player acted at each node.

        Matrix shape: (num_nodes, num_players)
        M[node_idx, player] = 1 if player acted at node_idx, else 0

        Returns:
            Dense numpy array (this matrix is NOT sparse in general)
        """
        num_nodes = len(self.nodes)
        player_matrix = np.zeros((num_nodes, self.num_players), dtype=np.float32)

        for node in self.nodes:
            if 0 <= node.player < self.num_players:  # Valid player (not chance/terminal)
                player_matrix[node.node_id, node.player] = 1.0

        return player_matrix

    def _build_terminal_utilities(self) -> np.ndarray:
        """
        Build matrix of terminal utilities.

        Matrix shape: (num_nodes, num_players)
        M[node_idx, player] = utility for player if node_idx is terminal, else 0

        Returns:
            Dense numpy array
        """
        num_nodes = len(self.nodes)
        utilities = np.zeros((num_nodes, self.num_players), dtype=np.float32)

        for node in self.nodes:
            if node.is_terminal:
                utilities[node.node_id, :] = node.terminal_utilities

        return utilities


# Helper functions

def print_matrix_stats(matrix_repr: MatrixRepresentation):
    """
    Print statistics about the matrix representation.

    Args:
        matrix_repr: MatrixRepresentation to analyze
    """
    print("\n=== Matrix Representation Statistics ===")
    print(f"Game tree nodes: {matrix_repr.num_nodes}")
    print(f"Tree depth: {matrix_repr.num_levels} levels")
    print(f"Information sets: {matrix_repr.num_infosets}")
    print(f"Infoset-action pairs: {matrix_repr.num_infoset_actions}")

    print("\n=== Level Matrices ===")
    for i, L in enumerate(matrix_repr.level_matrices):
        nnz = L.nnz
        sparsity = 100 * (1 - nnz / (L.shape[0] * L.shape[1]))
        print(f"  Level {i}: {L.shape}, {nnz} non-zeros, {sparsity:.2f}% sparse")

    print("\n=== Mapping Matrices ===")
    ia_matrix = matrix_repr.infoset_action_to_node_matrix
    ia_sparsity = 100 * (1 - ia_matrix.nnz / (ia_matrix.shape[0] * ia_matrix.shape[1]))
    print(f"  Infoset-action to node: {ia_matrix.shape}, {ia_matrix.nnz} non-zeros, {ia_sparsity:.2f}% sparse")

    player_matrix = matrix_repr.player_matrix
    player_nnz = np.count_nonzero(player_matrix)
    player_sparsity = 100 * (1 - player_nnz / player_matrix.size)
    print(f"  Player matrix: {player_matrix.shape}, {player_nnz} non-zeros, {player_sparsity:.2f}% sparse")

    print("\n=== Memory Estimate ===")
    total_bytes = 0

    # Level matrices
    for L in matrix_repr.level_matrices:
        total_bytes += L.data.nbytes + L.indices.nbytes + L.indptr.nbytes

    # Other matrices
    total_bytes += ia_matrix.data.nbytes + ia_matrix.indices.nbytes + ia_matrix.indptr.nbytes
    total_bytes += player_matrix.nbytes
    total_bytes += matrix_repr.terminal_utilities_matrix.nbytes

    print(f"  Total matrix memory: {total_bytes / 1024:.2f} KB ({total_bytes / (1024**2):.4f} MB)")


def estimate_memory_gb(num_nodes: int, num_infosets: int, num_players: int) -> float:
    """
    Estimate memory requirements for matrix representation.

    Args:
        num_nodes: Number of nodes in game tree
        num_infosets: Number of information sets
        num_players: Number of players

    Returns:
        Estimated memory in GB
    """
    # Very rough estimation
    # Level matrices: ~num_nodes edges × 12 bytes (CSR overhead)
    # Player matrix: num_nodes × num_players × 4 bytes
    # Infoset matrices: ~num_infosets × 10 actions × 12 bytes

    level_matrix_bytes = num_nodes * 12
    player_matrix_bytes = num_nodes * num_players * 4
    infoset_matrix_bytes = num_infosets * 10 * 12

    total_bytes = level_matrix_bytes + player_matrix_bytes + infoset_matrix_bytes
    return total_bytes / (1024 ** 3)
