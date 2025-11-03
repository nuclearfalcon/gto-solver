"""
Subgame Solver for Chunked Hold'em

Enables solving large Hold'em games by decomposing into betting round chunks.
Each chunk is solved independently using previous round's policy as blueprint.

Phase 8: Chunking Architecture

Usage:
    # Solve preflop chunk
    preflop_solver = SubgameSolver(
        full_game_config=holdem_config,
        round_name="preflop"
    )
    preflop_policy = preflop_solver.solve(iterations=10000)

    # Solve flop chunk using preflop blueprint
    flop_solver = SubgameSolver(
        full_game_config=holdem_config,
        round_name="flop",
        blueprint_policy=preflop_policy
    )
    flop_policy = flop_solver.solve(iterations=10000)
"""

import pyspiel
import logging
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass
import json
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SubgameConfig:
    """Configuration for a single betting round subgame."""
    round_name: str  # "preflop", "flop", "turn", "river"
    num_players: int
    stack_sizes: List[int]
    blinds: List[int]
    betting_abstraction: str
    num_suits: int
    num_ranks: int
    num_hole_cards: int
    board_cards_at_start: int  # 0 for preflop, 3 for flop, etc.


class BlueprintPolicy:
    """
    Container for a solved policy from a previous betting round.

    Used to:
    1. Initialize strategies in next round
    2. Compute reach probabilities at subgame entry
    3. Provide terminal value estimates (future work)
    """

    def __init__(self, policy_dict: Dict[str, Dict[int, float]]):
        """
        Initialize from policy dictionary.

        Args:
            policy_dict: {infoset_str: {action: probability}}
        """
        self.policy = policy_dict

    def get_action_probs(self, infoset: str) -> Optional[Dict[int, float]]:
        """Get action probabilities for an infoset."""
        return self.policy.get(infoset, None)

    def save(self, filepath: str):
        """Save blueprint to JSON file."""
        with open(filepath, 'w') as f:
            # Convert to JSON-serializable format
            serializable = {
                infoset: {str(action): prob for action, prob in actions.items()}
                for infoset, actions in self.policy.items()
            }
            json.dump(serializable, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'BlueprintPolicy':
        """Load blueprint from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Convert back to int keys
            policy = {
                infoset: {int(action): prob for action, prob in actions.items()}
                for infoset, actions in data.items()
            }
            return cls(policy)


class CombinedPolicy:
    """
    Unified policy interface for querying across all 4 betting round chunks.

    Combines preflop, flop, turn, and river policies into a single queryable interface.

    Usage:
        # After solving all chunks:
        combined = CombinedPolicy(policies)  # policies is dict from ChunkedSolver
        probs = combined.get_action_probs(infoset, round_name="flop")

        # Save/load:
        combined.save("policies/")
        loaded = CombinedPolicy.load("policies/")
    """

    def __init__(self, policies: Dict[str, BlueprintPolicy]):
        """
        Initialize combined policy from chunk policies.

        Args:
            policies: Dict mapping round_name → BlueprintPolicy
                     (e.g., {"preflop": policy1, "flop": policy2, ...})
        """
        self.policies = policies
        self.rounds = ["preflop", "flop", "turn", "river"]

        # Validate all rounds present
        for round_name in self.rounds:
            if round_name not in policies:
                logger.warning(f"Missing policy for {round_name} round")

        logger.info(f"Initialized CombinedPolicy with {len(policies)} rounds")

    def get_action_probs(
        self,
        infoset: str,
        round_name: str
    ) -> Optional[Dict[int, float]]:
        """
        Get action probabilities for an infoset in a specific round.

        Args:
            infoset: Information set string
            round_name: Which round ("preflop", "flop", "turn", "river")

        Returns:
            Dict mapping action → probability, or None if not found
        """
        if round_name not in self.policies:
            logger.warning(f"No policy for round: {round_name}")
            return None

        policy = self.policies[round_name]
        return policy.get_action_probs(infoset)

    def get_total_infosets(self) -> int:
        """Get total number of infosets across all rounds."""
        total = 0
        for policy in self.policies.values():
            total += len(policy.policy)
        return total

    def get_infosets_by_round(self) -> Dict[str, int]:
        """Get number of infosets per round."""
        return {
            round_name: len(policy.policy)
            for round_name, policy in self.policies.items()
        }

    def save(self, output_dir: str):
        """
        Save all policies to directory.

        Args:
            output_dir: Directory to save policies to
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for round_name, policy in self.policies.items():
            filepath = os.path.join(output_dir, f"{round_name}_policy.json")
            policy.save(filepath)
            logger.info(f"Saved {round_name} policy to {filepath}")

    @classmethod
    def load(cls, output_dir: str) -> 'CombinedPolicy':
        """
        Load all policies from directory.

        Args:
            output_dir: Directory containing policy files

        Returns:
            CombinedPolicy instance with loaded policies
        """
        import os

        policies = {}
        rounds = ["preflop", "flop", "turn", "river"]

        for round_name in rounds:
            filepath = os.path.join(output_dir, f"{round_name}_policy.json")
            if os.path.exists(filepath):
                policies[round_name] = BlueprintPolicy.load(filepath)
                logger.info(f"Loaded {round_name} policy from {filepath}")
            else:
                logger.warning(f"Policy file not found: {filepath}")

        return cls(policies)

    def __repr__(self) -> str:
        """String representation showing round coverage."""
        infosets = self.get_infosets_by_round()
        total = self.get_total_infosets()
        rounds_str = ", ".join(f"{r}:{n}" for r, n in infosets.items())
        return f"CombinedPolicy({rounds_str}, total={total})"


# Phase 8.7: Helper functions for public card filtering

def _parse_public_cards_from_infoset(infoset: str) -> str:
    """
    Extract public cards from infoset string.

    Infoset format: "[...][Public: CARDS][...]"
    Example: "[Public: 2s3h4d5c]" -> "2s3h4d5c"

    Args:
        infoset: Information set string

    Returns:
        Public cards string (e.g., "2s3h4d5c") or empty string if none
    """
    import re
    match = re.search(r'\[Public: ([^\]]*)\]', infoset)
    return match.group(1) if match else ""


def _get_last_public_card(public_cards_str: str) -> str:
    """
    Extract the last card from public cards string.

    Args:
        public_cards_str: Public cards (e.g., "2s3h4d5c")

    Returns:
        Last card (e.g., "5c") or empty string if none
    """
    if not public_cards_str or len(public_cards_str) < 2:
        return ""
    # Each card is 2 characters (rank + suit)
    return public_cards_str[-2:]


def _infoset_matches_public_card(infoset: str, target_card: str) -> bool:
    """
    Check if infoset's last public card matches target.

    Phase 8.7: Used for filtering policies by public card in sub-chunking.

    Args:
        infoset: Information set string
        target_card: Target card (e.g., "2s", "Kh")

    Returns:
        True if infoset's last public card matches target
    """
    public_cards = _parse_public_cards_from_infoset(infoset)
    if not public_cards:
        return False

    last_card = _get_last_public_card(public_cards)
    return last_card == target_card


class SubgameSolver:
    """
    Solves a single betting round chunk of Hold'em.

    Decomposes full Hold'em into manageable pieces:
    - Preflop chunk: Deal → preflop betting → flop entry
    - Flop chunk: Flop deal → flop betting → turn entry
    - Turn chunk: Turn deal → turn betting → river entry
    - River chunk: River deal → river betting → showdown

    Each chunk is solved independently, using previous round's policy as blueprint.
    """

    def __init__(
        self,
        full_game_config: Dict[str, Any],
        round_name: str,
        blueprint_policy: Optional[BlueprintPolicy] = None,
        precision: str = 'fp32',
        micro_batch_size: int = 24,
        max_nodes: Optional[Union[int, Dict[str, int]]] = None,
        use_true_predealing: bool = True
    ):
        """
        Initialize subgame solver for a specific betting round.

        Args:
            full_game_config: Full Hold'em configuration (will be adapted for subgame)
            round_name: Which round to solve ("preflop", "flop", "turn", "river")
            blueprint_policy: Policy from previous round (None for preflop)
            precision: Tensor precision - 'fp32' (default) or 'fp16' for 50% memory savings
            micro_batch_size: Max batch size for utility computation (default 24, use 6-12 for large games)
            max_nodes: Phase 8.7 - Max nodes before auto-splitting. Either:
                       - int: Single threshold for all rounds
                       - Dict[str, int]: Per-round thresholds (e.g., {"turn": 10000, "river": 5000})
                       - None: Use default thresholds
            use_true_predealing: Phase 9 - Use true game pre-dealing (Option A) instead of filtered extraction (Option B).
                                Default True for 8× memory reduction. Set False for backward compatibility.
        """
        self.full_config = full_game_config
        self.round = round_name
        self.blueprint = blueprint_policy
        self.precision = precision
        self.micro_batch_size = micro_batch_size
        self.use_true_predealing = use_true_predealing

        # Phase 8.7: Parse max_nodes parameter to get threshold for this round
        if max_nodes is None:
            # Default per-round thresholds
            default_thresholds = {
                "preflop": 50000,  # Preflop is small, no split needed
                "flop": 20000,     # Flop is manageable
                "turn": 10000,     # Turn needs splitting (~57k nodes)
                "river": 5000      # River needs aggressive splitting (~200k+ nodes)
            }
            self.max_nodes_threshold = default_thresholds.get(round_name, 10000)
        elif isinstance(max_nodes, int):
            # Single threshold for all rounds
            self.max_nodes_threshold = max_nodes
        elif isinstance(max_nodes, dict):
            # Per-round thresholds
            self.max_nodes_threshold = max_nodes.get(round_name, 10000)
        else:
            raise ValueError(f"max_nodes must be int, dict, or None, got {type(max_nodes)}")

        # Phase 8.7: Cache for estimated chunk size (computed lazily)
        self._estimated_nodes = None

        # Validate
        valid_rounds = ["preflop", "flop", "turn", "river"]
        if round_name not in valid_rounds:
            raise ValueError(f"round_name must be one of {valid_rounds}, got {round_name}")

        if round_name != "preflop" and blueprint_policy is None:
            logger.warning(f"Solving {round_name} without blueprint policy - will use uniform initialization")

        # Create subgame-specific config
        self.subgame_config = self._create_subgame_config()

        logger.info(f"Initialized SubgameSolver for {round_name} round "
                   f"(precision={precision}, micro_batch={micro_batch_size}, "
                   f"max_nodes={self.max_nodes_threshold})")

    def _create_subgame_config(self) -> Dict[str, Any]:
        """
        Generate OpenSpiel config for just this betting round.

        Strategy:
        - Set num_rounds=1 (only this round's betting)
        - Set num_board_cards based on cumulative cards dealt up to this round
        - Keep other params (num_players, stacks, blinds, etc.)
        """
        config = self.full_config.copy()

        # Override for single round (OpenSpiel uses camelCase)
        config["numRounds"] = 1

        # Calculate cumulative board cards up to this round
        # Parse original board card string (e.g., "0 1 1 1")
        original_board_cards = config.get("numBoardCards", "0 3 1 1")
        board_per_round = [int(x) for x in original_board_cards.split()]

        # Map round name to index
        round_indices = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
        round_idx = round_indices[self.round]

        # Cumulative cards up to and including this round
        cumulative_cards = sum(board_per_round[:round_idx + 1])
        config["numBoardCards"] = str(cumulative_cards)

        # Adjust first player (who acts first this round)
        # For preflop: SB acts first (player 0 if 2p)
        # For postflop: first player left of button (usually player 0)
        if self.round == "preflop":
            config["firstPlayer"] = "2 1 1 1"  # Position 2 acts first preflop (after BB)
        else:
            config["firstPlayer"] = "1"  # Position 1 acts first postflop

        logger.info(f"Created {self.round} subgame config: {config['numBoardCards']} board cards, {config['numRounds']} round")

        return config

    def _estimate_chunk_size(self, num_samples: int = 500) -> int:
        """
        Estimate the number of nodes in this chunk's game tree using sampling.

        Phase 8.7: Uses Monte Carlo sampling to estimate game tree size before solving.
        This allows automatic detection of chunks that need sub-chunking.

        Strategy:
        1. Sample multiple random game trajectories
        2. Count nodes (chance + decision) per trajectory
        3. Average and extrapolate to estimate full tree size

        Args:
            num_samples: Number of game trajectories to sample (default 500)

        Returns:
            Estimated number of nodes in game tree
        """
        # Return cached value if already computed
        if self._estimated_nodes is not None:
            return self._estimated_nodes

        logger.info(f"  Estimating {self.round} chunk size ({num_samples} samples)...")

        # Create game for estimation
        game = pyspiel.load_game("universal_poker", self.subgame_config)

        total_nodes = 0

        for sample_idx in range(num_samples):
            state = game.new_initial_state()
            nodes_this_trajectory = 0

            # Simulate one complete hand
            while not state.is_terminal():
                nodes_this_trajectory += 1

                if state.is_chance_node():
                    # Random card dealing
                    outcomes = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes)
                    action = random.choices(action_list, weights=prob_list)[0]
                    state.apply_action(action)
                else:
                    # Player decision - random action
                    legal_actions = state.legal_actions()
                    action = random.choice(legal_actions)
                    state.apply_action(action)

            total_nodes += nodes_this_trajectory

        # Average nodes per trajectory
        avg_nodes_per_trajectory = total_nodes / num_samples

        # Estimate full game tree size
        # Since we're sampling paths, the full tree is much larger
        # Heuristic: multiply by branching factor estimate
        # For poker, branching factor varies by:
        # - Card dealing: ~deck_size choices
        # - Betting: ~4 choices (fold/call/bet/allin for FCPA)
        num_suits = self.full_config.get('numSuits', 4)
        num_ranks = self.full_config.get('numRanks', 13)
        deck_size = num_suits * num_ranks

        # Rough branching factor estimate
        # Chance nodes: ~deck_size / 2 (cards get removed)
        # Action nodes: ~3 (average legal actions)
        avg_branching_factor = (deck_size / 2 + 3) / 2

        # Estimate: avg_path_length * branching_factor^(depth)
        # Simplified: use exponential scaling
        # Conservative estimate: multiply by log(avg_branching_factor)^2
        import math
        depth_factor = max(1.0, math.log(avg_branching_factor) ** 2)

        estimated_nodes = int(avg_nodes_per_trajectory * depth_factor * 100)

        # Cache result
        self._estimated_nodes = estimated_nodes

        logger.info(f"    Estimated {self.round} chunk: ~{estimated_nodes:,} nodes "
                   f"(avg path: {avg_nodes_per_trajectory:.1f} nodes)")

        return estimated_nodes

    @property
    def needs_splitting(self) -> bool:
        """
        Check if this chunk needs sub-chunking based on size estimate.

        Phase 8.7: Only enable sub-chunking for "turn" and "river" rounds,
        as preflop/flop are typically small enough to solve directly.

        Returns:
            True if chunk should be split into sub-chunks
        """
        # Only split turn/river rounds (preflop/flop are small)
        if self.round not in ["turn", "river"]:
            return False

        # Check if estimated size exceeds threshold
        estimated_nodes = self._estimate_chunk_size()
        return estimated_nodes > self.max_nodes_threshold

    def _card_string_to_action(self, card_str: str, game: pyspiel.Game) -> int:
        """
        Convert card string (e.g., '2s', 'Ah') to OpenSpiel action index.

        Phase 9: Helper for creating starting states with pre-dealt cards.

        Args:
            card_str: Card string like '2s', 'Kh', etc.
            game: OpenSpiel game instance

        Returns:
            Action index for dealing this card
        """
        rank_chars = '23456789TJQKA'
        suit_chars = 'shdc'

        if len(card_str) != 2:
            raise ValueError(f"Card string must be 2 characters, got: {card_str}")

        rank_char = card_str[0]
        suit_char = card_str[1]

        if rank_char not in rank_chars or suit_char not in suit_chars:
            raise ValueError(f"Invalid card string: {card_str}")

        rank_idx = rank_chars.index(rank_char)
        suit_idx = suit_chars.index(suit_char)

        num_suits = self.full_config.get('numSuits', 4)

        # OpenSpiel card encoding: card_id = rank * num_suits + suit
        card_id = rank_idx * num_suits + suit_idx

        return card_id

    def _create_starting_state_with_card(
        self,
        game: pyspiel.Game,
        target_card: str
    ) -> pyspiel.State:
        """
        Create a starting state with specific public card pre-dealt.

        Phase 9: True game pre-dealing (Option A). Navigates through chance nodes
        to deal specific cards, returning a constrained game state that only
        contains relevant sub-tree.

        Args:
            game: OpenSpiel game instance
            target_card: Card to pre-deal (e.g., '2s' for turn, 'Kh' for river)

        Returns:
            Game state with target card dealt as last public card
        """
        state = game.new_initial_state()

        # Determine which public card position we're constraining
        # based on the round (turn = 4th board card, river = 5th)
        num_board_cards_str = self.full_config.get('numBoardCards', '0 3 1 1')
        board_cards_per_round = list(map(int, num_board_cards_str.split()))

        # Calculate total board cards up to this round
        round_idx = ["preflop", "flop", "turn", "river"].index(self.round)

        # For turn: we need to deal all hole cards, then 3 flop cards, then our specific turn card
        # For river: we need hole cards + 3 flop + 1 turn + our specific river card

        target_card_action = self._card_string_to_action(target_card, game)
        cards_dealt = []

        # Navigate through chance nodes until we reach the point where we need to deal target card
        chance_nodes_processed = 0
        num_players = self.full_config.get('numPlayers', 2)
        num_hole_cards = self.full_config.get('numHoleCards', 2)

        # Total hole cards to deal first
        total_hole_cards = num_players * num_hole_cards

        # Total board cards before our target card
        total_board_before_target = sum(board_cards_per_round[:round_idx + 1]) - 1

        # Total cards to deal before target
        total_cards_before_target = total_hole_cards + total_board_before_target

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                available_actions = [action for action, _ in outcomes]

                # Check if this is the chance node for our target card
                if chance_nodes_processed == total_cards_before_target:
                    # This is where we deal our target card
                    if target_card_action in available_actions:
                        state.apply_action(target_card_action)
                        cards_dealt.append(target_card)
                        chance_nodes_processed += 1
                        # We've dealt the target card, now we're done setting up
                        break
                    else:
                        # Target card already dealt earlier (conflict)
                        raise ValueError(
                            f"Cannot deal {target_card}: already dealt in previous rounds"
                        )
                else:
                    # Deal random card (but not our target card, save it for later)
                    valid_actions = [a for a in available_actions if a != target_card_action]
                    if not valid_actions:
                        raise ValueError(f"No valid cards to deal (target {target_card} is only option)")

                    action = random.choice(valid_actions)
                    state.apply_action(action)
                    chance_nodes_processed += 1
            else:
                # Player decision node - take random action to continue
                legal_actions = state.legal_actions()
                action = random.choice(legal_actions)
                state.apply_action(action)

        logger.debug(f"  Created starting state with {target_card} pre-dealt "
                    f"(dealt {len(cards_dealt)} target cards, passed through {chance_nodes_processed} chance nodes)")

        return state

    def _enumerate_public_cards(self) -> List[str]:
        """
        Enumerate all possible public cards for sub-chunking.

        Phase 8.7: For Turn/River chunks, returns all possible cards
        that could be dealt as the last public card.

        Returns:
            List of card strings (e.g., ['2s', '2h', '3s', ..., 'Ah', 'Ad', 'Ac'])
        """
        num_suits = self.full_config.get('numSuits', 4)
        num_ranks = self.full_config.get('numRanks', 13)

        # Rank encoding: 2=0, 3=1, ..., K=11, A=12
        rank_chars = '23456789TJQKA'[:num_ranks]
        # Suit encoding: s=0, h=1, d=2, c=3
        suit_chars = 'shdc'[:num_suits]

        # Generate all card strings
        cards = [rank + suit for rank in rank_chars for suit in suit_chars]

        logger.info(f"  Enumerated {len(cards)} possible public cards "
                   f"({num_ranks} ranks × {num_suits} suits)")

        return cards

    def _solve_with_public_card_filter(
        self,
        target_card: str,
        iterations: int,
        progress_interval: int,
        blueprint_policy: Optional[BlueprintPolicy] = None
    ) -> BlueprintPolicy:
        """
        Solve the full game and filter policy to only infosets with target public card.

        Phase 8.7: Hybrid approach between Option A (Game Wrapper) and Option B (Filtered Extraction).
        We solve the normal game but only keep infosets matching the target card.

        This provides memory benefits (smaller policy) while being simpler to implement
        than true game pre-dealing.

        Args:
            target_card: Card to filter by (e.g., "2s", "Kh")
            iterations: CFR iterations
            progress_interval: Logging frequency
            blueprint_policy: Optional blueprint for initialization

        Returns:
            BlueprintPolicy containing only infosets with target card
        """
        logger.info(f"    Solving sub-chunk for {target_card}...")

        # Import here to avoid circular dependency
        from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver

        # Create game normally
        game = pyspiel.load_game("universal_poker", self.subgame_config)

        # Create solver
        solver = MatrixCFRSolver(
            game,
            use_sparse=True,
            precision=self.precision,
            micro_batch_size=self.micro_batch_size
        )

        # Initialize from blueprint if provided
        if blueprint_policy is not None:
            logger.info(f"      Using blueprint for warm-start")
            self._initialize_from_blueprint_with_policy(solver, blueprint_policy)

        # Solve normally
        solver.solve(iterations=iterations, progress_interval=progress_interval)

        # Extract full policy
        full_policy_dict = solver.get_strategy_dict()

        # Filter to only infosets matching target card
        filtered_policy_dict = {}
        for infoset, action_probs in full_policy_dict.items():
            if _infoset_matches_public_card(infoset, target_card):
                filtered_policy_dict[infoset] = action_probs

        logger.info(f"      Filtered: {len(filtered_policy_dict)} / {len(full_policy_dict)} infosets "
                   f"({100.0 * len(filtered_policy_dict) / len(full_policy_dict):.1f}%)")

        # Cleanup
        del solver
        del game
        import gc
        gc.collect()

        return BlueprintPolicy(filtered_policy_dict)

    def _solve_with_true_predealing(
        self,
        target_card: str,
        iterations: int,
        progress_interval: int,
        blueprint_policy: Optional[BlueprintPolicy] = None
    ) -> BlueprintPolicy:
        """
        Solve with true game pre-dealing - constrains game tree BEFORE solving.

        Phase 9: Option A (True Pre-Dealing). Creates a starting state with target card
        pre-dealt, then builds matrix representation only for that constrained sub-tree.
        This achieves GENUINE memory reduction (8× smaller game tree) vs Option B which
        solves full tree then filters.

        Args:
            target_card: Card to pre-deal (e.g., "2s", "Kh")
            iterations: CFR iterations
            progress_interval: Logging frequency
            blueprint_policy: Optional blueprint for initialization

        Returns:
            BlueprintPolicy for this constrained game tree
        """
        logger.info(f"    Solving sub-chunk for {target_card} (true pre-dealing)...")

        # Import here to avoid circular dependency
        from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver
        from matrix_cfr.game_to_matrix import GameTreeConverter

        # Create game
        game = pyspiel.load_game("universal_poker", self.subgame_config)

        # Phase 9: Create starting state with target card pre-dealt
        starting_state = self._create_starting_state_with_card(game, target_card)

        logger.info(f"      Created starting state with {target_card} pre-dealt")

        # Build matrix representation from constrained starting state
        converter = GameTreeConverter(game)
        matrices = converter.build_matrices(starting_state=starting_state)

        logger.info(f"      Game tree: {matrices.num_nodes} nodes (vs full tree)")
        logger.info(f"      Infosets: {matrices.num_infosets}")

        # Create solver with constrained matrices
        solver = MatrixCFRSolver(
            game,
            use_sparse=True,
            precision=self.precision,
            micro_batch_size=self.micro_batch_size
        )

        # Replace solver's matrices with our constrained ones
        solver.matrices = matrices
        solver.num_infosets = matrices.num_infosets
        solver.infoset_to_actions = matrices.infoset_to_actions

        # Initialize from blueprint if provided
        if blueprint_policy is not None:
            logger.info(f"      Using blueprint for warm-start")
            self._initialize_from_blueprint_with_policy(solver, blueprint_policy)

        # Solve constrained game
        solver.solve(iterations=iterations, progress_interval=progress_interval)

        # Extract policy (NO filtering needed - all infosets are relevant!)
        policy_dict = solver.get_strategy_dict()

        logger.info(f"      Policy: {len(policy_dict)} infosets")

        # Cleanup
        del solver
        del converter
        del game
        import gc
        gc.collect()

        return BlueprintPolicy(policy_dict)

    def _initialize_from_blueprint_with_policy(
        self,
        solver,
        blueprint_policy: BlueprintPolicy
    ):
        """
        Initialize solver from blueprint policy.

        Phase 8.7: Simplified version that uses blueprint directly without reach probability estimation.

        Args:
            solver: MatrixCFRSolver instance
            blueprint_policy: Blueprint policy to initialize from
        """
        # Build strategy mapping
        strategy_dict = self._build_strategy_mapping(
            blueprint=blueprint_policy,
            solver=solver,
            reach_probs=None
        )

        # Set initial strategy
        stats = solver.set_initial_strategy_from_policy(strategy_dict)

        logger.info(f"        Blueprint coverage: {stats['coverage_pct']:.1f}%")

    def _solve_direct(self, iterations: int, progress_interval: int = 1000) -> BlueprintPolicy:
        """
        Solve this chunk directly without sub-chunking.

        Phase 8.7: Refactored from solve() - contains original solving logic.

        Args:
            iterations: Number of CFR iterations
            progress_interval: How often to log progress

        Returns:
            BlueprintPolicy containing the equilibrium strategy for this round
        """
        logger.info(f"  Solving {self.round} chunk directly ({iterations} iterations)...")

        # Import here to avoid circular dependency
        from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver

        # Create OpenSpiel game for this subgame
        game = pyspiel.load_game("universal_poker", self.subgame_config)

        # Phase 8.6: Create solver with memory optimization parameters
        solver = MatrixCFRSolver(
            game,
            use_sparse=True,
            precision=self.precision,
            micro_batch_size=self.micro_batch_size
        )

        # If blueprint provided, initialize strategies from blueprint
        if self.blueprint is not None:
            logger.info(f"    Using blueprint policy from previous round")
            self._initialize_from_blueprint(solver)
        else:
            logger.info(f"    No blueprint - using uniform initialization")

        # Solve
        solver.solve(iterations=iterations, progress_interval=progress_interval)

        # Extract policy (use get_strategy_dict for proper dict format)
        policy_dict = solver.get_strategy_dict()

        logger.info(f"    {self.round} chunk solved: {len(policy_dict)} infosets")

        # CRITICAL: Delete solver to free GPU memory before returning
        # This prevents GPU memory fragmentation between chunks
        del solver
        del game

        # Force garbage collection to free GPU arrays immediately
        import gc
        gc.collect()

        # Return as blueprint for next round
        return BlueprintPolicy(policy_dict)

    def solve(self, iterations: int, progress_interval: int = 1000) -> BlueprintPolicy:
        """
        Solve this subgame chunk to approximate equilibrium.

        Phase 8.7: Now supports automatic sub-chunking for large chunks.
        - If estimated nodes ≤ max_nodes: solve directly
        - If estimated nodes > max_nodes: split by public card, solve sequentially with warm-start

        Args:
            iterations: Number of CFR iterations per sub-chunk
            progress_interval: How often to log progress

        Returns:
            BlueprintPolicy containing the equilibrium strategy for this round
        """
        logger.info(f"Solving {self.round} chunk...")

        # Phase 8.7: Check if sub-chunking is needed
        if not self.needs_splitting:
            # Chunk is small enough - solve directly
            return self._solve_direct(iterations, progress_interval)

        # Phase 8.7: Sub-chunking path
        estimated_nodes = self._estimate_chunk_size()
        logger.info(f"  Chunk too large (~{estimated_nodes:,} nodes > {self.max_nodes_threshold:,} threshold)")
        logger.info(f"  Splitting into sub-chunks by public card...")

        # Enumerate all possible public cards for this round
        public_cards = self._enumerate_public_cards()
        num_sub_chunks = len(public_cards)

        logger.info(f"  Creating {num_sub_chunks} sub-chunks (target: ~{estimated_nodes // num_sub_chunks:,} nodes each)")

        # Solve each sub-chunk sequentially with warm-starting
        sub_policies = {}
        current_blueprint = self.blueprint  # Start with original blueprint (from previous round)

        for i, card in enumerate(public_cards):
            logger.info(f"\n  Sub-chunk {i+1}/{num_sub_chunks}: {self.round}|{card}")

            # Phase 9: Choose solving method based on use_true_predealing flag
            if self.use_true_predealing:
                # Option A: True pre-dealing (8× memory reduction, 8× speed improvement)
                sub_policy = self._solve_with_true_predealing(
                    target_card=card,
                    iterations=iterations,
                    progress_interval=progress_interval,
                    blueprint_policy=current_blueprint
                )
            else:
                # Option B: Filtered extraction (backward compatibility)
                sub_policy = self._solve_with_public_card_filter(
                    target_card=card,
                    iterations=iterations,
                    progress_interval=progress_interval,
                    blueprint_policy=current_blueprint
                )

            # Store result
            sub_chunk_key = f"{self.round}_{card}"
            sub_policies[sub_chunk_key] = sub_policy

            # Use this policy as blueprint for next sub-chunk (warm-start)
            current_blueprint = sub_policy

            # Memory cleanup between sub-chunks
            import gc
            import jax
            gc.collect()
            try:
                jax.clear_caches()
            except:
                pass

        # Merge all sub-chunk policies
        logger.info(f"\n  Merging {len(sub_policies)} sub-chunk policies...")
        merged_policy = self._merge_sub_policies(sub_policies)

        logger.info(f"  {self.round} chunk complete: {len(merged_policy.policy)} total infosets")

        return merged_policy

    def _merge_sub_policies(
        self,
        sub_policies: Dict[str, BlueprintPolicy]
    ) -> BlueprintPolicy:
        """
        Merge sub-chunk policies into a single unified policy.

        Phase 8.7: Each sub-chunk covers a disjoint set of infosets (conditioned on
        different public cards), so we can simply union all policies.

        Args:
            sub_policies: Dict mapping sub_chunk_key → BlueprintPolicy
                         (e.g., {"turn_2s": policy1, "turn_3s": policy2, ...})

        Returns:
            BlueprintPolicy containing all infosets from all sub-chunks
        """
        merged_dict = {}
        conflicts = 0

        for sub_chunk_key, policy in sub_policies.items():
            for infoset, action_probs in policy.policy.items():
                if infoset in merged_dict:
                    # This should never happen (policies should be disjoint)
                    logger.warning(f"    ⚠️ Duplicate infoset: {infoset} (in {sub_chunk_key})")
                    conflicts += 1
                else:
                    merged_dict[infoset] = action_probs

        # Report stats
        total_sub_infosets = sum(len(p.policy) for p in sub_policies.values())
        logger.info(f"    Merged {len(sub_policies)} sub-policies:")
        logger.info(f"      Total infosets: {len(merged_dict)}")
        logger.info(f"      Sub-policy infosets: {total_sub_infosets}")
        if conflicts > 0:
            logger.warning(f"      Conflicts detected: {conflicts}")
        else:
            logger.info(f"      ✓ No conflicts (policies are disjoint)")

        # Validate completeness (rough check)
        expected_per_sub = total_sub_infosets // len(sub_policies)
        logger.info(f"      Avg per sub-chunk: {expected_per_sub}")

        return BlueprintPolicy(merged_dict)

    def _estimate_reach_probabilities(
        self,
        blueprint: BlueprintPolicy,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Estimate reach probabilities at current subgame using Monte Carlo sampling.

        Simulates hands using blueprint policy from previous round(s) to estimate
        how likely each infoset in the current subgame is to be reached.

        Args:
            blueprint: Policy from previous round
            num_samples: Number of Monte Carlo samples

        Returns:
            Dictionary mapping infoset_str -> reach_probability
        """
        logger.info(f"  Estimating reach probabilities ({num_samples} samples)...")

        # Create game for this subgame
        game = pyspiel.load_game("universal_poker", self.subgame_config)

        # Track infoset visit counts
        infoset_counts = defaultdict(int)
        total_samples = 0

        for _ in range(num_samples):
            state = game.new_initial_state()

            # Simulate one complete hand
            while not state.is_terminal():
                if state.is_chance_node():
                    # Random card dealing
                    outcomes = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes)
                    action = random.choices(action_list, weights=prob_list)[0]
                    state.apply_action(action)
                else:
                    # Player decision - record infoset
                    current_player = state.current_player()
                    infoset_str = state.information_state_string(current_player)
                    infoset_counts[infoset_str] += 1

                    # Choose action using blueprint if available, else uniform
                    legal_actions = state.legal_actions()
                    action_probs = blueprint.get_action_probs(infoset_str)

                    if action_probs is not None:
                        # Use blueprint policy
                        # Build probability distribution over legal actions
                        probs = [action_probs.get(a, 0.0) for a in legal_actions]
                        prob_sum = sum(probs)
                        if prob_sum > 1e-10:
                            probs = [p / prob_sum for p in probs]
                            action = random.choices(legal_actions, weights=probs)[0]
                        else:
                            # Fallback to uniform if no valid probs
                            action = random.choice(legal_actions)
                    else:
                        # Infoset not in blueprint - uniform random
                        action = random.choice(legal_actions)

                    state.apply_action(action)

            total_samples += 1

        # Normalize counts to probabilities
        total_visits = sum(infoset_counts.values())
        reach_probs = {
            infoset: count / total_visits
            for infoset, count in infoset_counts.items()
        } if total_visits > 0 else {}

        logger.info(f"    Tracked {len(reach_probs)} unique infosets")
        logger.info(f"    Total visits: {total_visits}")

        return reach_probs

    def _build_strategy_mapping(
        self,
        blueprint: BlueprintPolicy,
        solver,
        reach_probs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Build strategy dictionary for current subgame based on blueprint policy.

        Uses simulation-based approach: for each infoset in the current subgame,
        determines the appropriate initial strategy based on blueprint guidance.

        Args:
            blueprint: Policy from previous round
            solver: MatrixCFRSolver instance for current subgame
            reach_probs: Optional reach probabilities for weighting (currently unused)

        Returns:
            Policy dictionary: {infoset_str: {action: probability}}
        """
        logger.info(f"  Building strategy mapping from blueprint...")

        # Get all infosets in current subgame
        current_infosets = set(solver.matrix_repr.infoset_to_actions.keys())

        # Start with blueprint policy
        strategy_dict = {}

        # For each infoset in current game, try to find blueprint guidance
        for infoset in current_infosets:
            if infoset in blueprint.policy:
                # Direct match - use blueprint strategy
                blueprint_actions = blueprint.policy[infoset]

                # Get legal actions for this infoset in current game
                legal_actions = solver.matrix_repr.infoset_to_actions[infoset]

                # Build strategy for current game's action space
                # (may need to normalize if action space differs)
                current_strategy = {}
                total_prob = 0.0

                for action in legal_actions:
                    prob = blueprint_actions.get(action, 0.0)
                    current_strategy[action] = prob
                    total_prob += prob

                # Normalize if needed
                if total_prob > 1e-10:
                    current_strategy = {
                        action: prob / total_prob
                        for action, prob in current_strategy.items()
                    }
                    strategy_dict[infoset] = current_strategy
                # else: skip this infoset, will use uniform fallback

        logger.info(f"    Mapped {len(strategy_dict)} / {len(current_infosets)} infosets")
        coverage_pct = 100.0 * len(strategy_dict) / len(current_infosets) if current_infosets else 0.0
        logger.info(f"    Coverage: {coverage_pct:.1f}%")

        return strategy_dict

    def _initialize_from_blueprint(self, solver):
        """
        Initialize solver strategies using blueprint from previous round.

        Phase 8.4: Complete implementation
        - Estimates reach probabilities via Monte Carlo sampling
        - Maps blueprint infosets to current subgame infosets
        - Sets initial strategies from blueprint (not uniform)

        Args:
            solver: MatrixCFRSolver instance to initialize
        """
        logger.info(f"Initializing from blueprint policy...")

        # Step 1: Estimate reach probabilities
        reach_probs = self._estimate_reach_probabilities(
            blueprint=self.blueprint,
            num_samples=1000
        )

        # Step 2: Build strategy mapping
        strategy_dict = self._build_strategy_mapping(
            blueprint=self.blueprint,
            solver=solver,
            reach_probs=reach_probs
        )

        # Step 3: Set initial strategy in solver
        stats = solver.set_initial_strategy_from_policy(strategy_dict)

        # Log initialization statistics
        logger.info(f"  Blueprint initialization complete:")
        logger.info(f"    Matched infosets: {stats['matched_infosets']}/{stats['total_infosets']}")
        logger.info(f"    Coverage: {stats['coverage_pct']:.1f}%")
        logger.info(f"    Uniform fallback: {stats['uniform_fallback']} infosets")


class ChunkedSolver:
    """
    Orchestrates solving Hold'em by decomposing into betting round chunks.

    Usage:
        chunked = ChunkedSolver(holdem_config)
        combined_policy = chunked.solve(iterations_per_chunk=10000)

        # With memory profiling:
        from matrix_cfr.gpu_memory import MemoryProfiler
        profiler = MemoryProfiler()
        chunked = ChunkedSolver(holdem_config, memory_profiler=profiler)
        combined_policy = chunked.solve(iterations_per_chunk=10000)
        profiler.print_report()
    """

    def __init__(
        self,
        full_game_config: Dict[str, Any],
        memory_profiler=None,
        precision: str = 'fp32',
        micro_batch_size: int = 24,
        max_nodes: Optional[Union[int, Dict[str, int]]] = None,
        use_true_predealing: bool = True
    ):
        """
        Initialize chunked solver for full Hold'em game.

        Args:
            full_game_config: Complete Hold'em configuration (all rounds)
            memory_profiler: Optional MemoryProfiler for tracking memory usage
            precision: Tensor precision - 'fp32' (default) or 'fp16' for 50% memory savings
            micro_batch_size: Max batch size for utility computation (default 24, use 6-12 for large games)
            max_nodes: Phase 8.7 - Max nodes per chunk before auto-splitting. Either:
                       - int: Single threshold for all rounds
                       - Dict[str, int]: Per-round thresholds (e.g., {"turn": 10000, "river": 5000})
                       - None: Use default thresholds (recommended)
            use_true_predealing: Phase 9 - Use true game pre-dealing (Option A) for 8× memory reduction.
                                Default True for maximum performance. Set False for backward compatibility.
        """
        self.config = full_game_config
        self.chunks = ["preflop", "flop", "turn", "river"]
        self.policies = {}  # Store policy for each chunk
        self.profiler = memory_profiler  # Optional memory profiler
        self.precision = precision
        self.micro_batch_size = micro_batch_size
        self.max_nodes = max_nodes  # Phase 8.7
        self.use_true_predealing = use_true_predealing  # Phase 9

        predealing_mode = "true pre-dealing (Option A)" if use_true_predealing else "filtered extraction (Option B)"
        logger.info(f"Initialized ChunkedSolver for {self.config.get('numPlayers', 2)}-player Hold'em "
                   f"(precision={precision}, micro_batch={micro_batch_size}, "
                   f"max_nodes={max_nodes}, mode={predealing_mode})")

    def solve(
        self,
        iterations_per_chunk: int = 10000,
        progress_interval: int = 1000
    ) -> Dict[str, BlueprintPolicy]:
        """
        Solve all chunks sequentially (preflop → flop → turn → river).

        Each chunk uses previous chunk's policy as blueprint.

        Args:
            iterations_per_chunk: CFR iterations per chunk
            progress_interval: Logging frequency

        Returns:
            Dictionary mapping round_name → BlueprintPolicy
        """
        logger.info("=" * 80)
        logger.info("CHUNKED HOLD'EM SOLVING")
        logger.info("=" * 80)

        # Take baseline memory snapshot if profiler provided
        if self.profiler:
            self.profiler.snapshot("baseline")

        blueprint = None

        for chunk_name in self.chunks:
            logger.info(f"\n{'='*80}")
            logger.info(f"CHUNK: {chunk_name.upper()}")
            logger.info(f"{'='*80}")

            # Memory snapshot before chunk
            if self.profiler:
                self.profiler.snapshot(f"before_{chunk_name}")

            # Phase 8.7-9: Create subgame solver with memory optimization + sub-chunking parameters
            subgame = SubgameSolver(
                full_game_config=self.config,
                round_name=chunk_name,
                blueprint_policy=blueprint,
                precision=self.precision,
                micro_batch_size=self.micro_batch_size,
                max_nodes=self.max_nodes,  # Phase 8.7: Enable auto-splitting
                use_true_predealing=self.use_true_predealing  # Phase 9: True pre-dealing
            )

            # Solve this chunk
            policy = subgame.solve(
                iterations=iterations_per_chunk,
                progress_interval=progress_interval
            )

            # Store result
            self.policies[chunk_name] = policy

            # Feed forward as blueprint for next chunk
            blueprint = policy

            # CRITICAL: Explicitly delete solver to free GPU memory
            # The solver holds JAX arrays that fragment GPU memory
            del subgame

            # Memory snapshot after chunk
            if self.profiler:
                self.profiler.snapshot(f"after_{chunk_name}")

            logger.info(f"✓ {chunk_name} chunk complete\n")

            # CRITICAL: Aggressive GPU memory cleanup between chunks
            # JAX accumulates memory across chunks, leading to fragmentation and OOM
            import gc
            import jax
            from jax.lib import xla_bridge

            logger.info("  Cleaning up GPU memory...")

            # Step 1: Clear JAX compilation caches
            try:
                jax.clear_caches()
            except:
                pass

            # Step 2: Force Python garbage collection
            gc.collect()

            # Step 3: Get backend and force memory release
            try:
                # Get the default backend (GPU if available)
                backend = xla_bridge.get_backend()

                # Force defragmentation by clearing live buffers
                # This is the key to preventing OOM from fragmentation
                if hasattr(backend, 'defragment'):
                    backend.defragment()
                    logger.info("  ✓ Defragmented GPU memory")

                logger.info("  ✓ Cleared JAX caches and triggered GPU cleanup")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not defragment: {e}")

            # Step 4: Final aggressive garbage collection
            gc.collect()
            gc.collect()  # Twice for good measure

        logger.info("=" * 80)
        logger.info("ALL CHUNKS SOLVED")
        logger.info("=" * 80)

        # Print memory report if profiler was provided
        if self.profiler:
            logger.info("")  # Blank line
            self.profiler.print_report()

        return self.policies

    def save_policies(self, output_dir: str):
        """Save all chunk policies to directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for round_name, policy in self.policies.items():
            filepath = os.path.join(output_dir, f"{round_name}_policy.json")
            policy.save(filepath)
            logger.info(f"Saved {round_name} policy to {filepath}")

    def load_policies(self, output_dir: str):
        """Load all chunk policies from directory."""
        import os

        for round_name in self.chunks:
            filepath = os.path.join(output_dir, f"{round_name}_policy.json")
            if os.path.exists(filepath):
                self.policies[round_name] = BlueprintPolicy.load(filepath)
                logger.info(f"Loaded {round_name} policy from {filepath}")
            else:
                logger.warning(f"Policy file not found: {filepath}")
