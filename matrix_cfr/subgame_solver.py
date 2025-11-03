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
from typing import Optional, Dict, List, Any, Tuple
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
        blueprint_policy: Optional[BlueprintPolicy] = None
    ):
        """
        Initialize subgame solver for a specific betting round.

        Args:
            full_game_config: Full Hold'em configuration (will be adapted for subgame)
            round_name: Which round to solve ("preflop", "flop", "turn", "river")
            blueprint_policy: Policy from previous round (None for preflop)
        """
        self.full_config = full_game_config
        self.round = round_name
        self.blueprint = blueprint_policy

        # Validate
        valid_rounds = ["preflop", "flop", "turn", "river"]
        if round_name not in valid_rounds:
            raise ValueError(f"round_name must be one of {valid_rounds}, got {round_name}")

        if round_name != "preflop" and blueprint_policy is None:
            logger.warning(f"Solving {round_name} without blueprint policy - will use uniform initialization")

        # Create subgame-specific config
        self.subgame_config = self._create_subgame_config()

        logger.info(f"Initialized SubgameSolver for {round_name} round")

    def _create_subgame_config(self) -> Dict[str, Any]:
        """
        Generate OpenSpiel config for just this betting round.

        Strategy:
        - Set num_rounds=1 (only this round's betting)
        - Set num_board_cards appropriately:
            - Preflop: "0" (no board yet)
            - Flop: "3" (flop cards)
            - Turn: "4" (flop + turn)
            - River: "5" (all board cards)
        - Keep other params (num_players, stacks, blinds, etc.)
        """
        config = self.full_config.copy()

        # Override for single round (OpenSpiel uses camelCase)
        config["numRounds"] = 1

        # Set board cards based on round
        board_cards_map = {
            "preflop": "0",
            "flop": "3",
            "turn": "4",
            "river": "5"
        }
        config["numBoardCards"] = board_cards_map[self.round]

        # Adjust first player (who acts first this round)
        # For preflop: SB acts first (player 0 if 2p)
        # For postflop: first player left of button (usually player 0)
        if self.round == "preflop":
            config["firstPlayer"] = "2 1 1 1"  # Position 2 acts first preflop (after BB)
        else:
            config["firstPlayer"] = "1"  # Position 1 acts first postflop

        logger.info(f"Created {self.round} subgame config: {config['numBoardCards']} board cards, {config['numRounds']} round")

        return config

    def solve(self, iterations: int, progress_interval: int = 1000) -> BlueprintPolicy:
        """
        Solve this subgame chunk to approximate equilibrium.

        Args:
            iterations: Number of CFR iterations
            progress_interval: How often to log progress

        Returns:
            BlueprintPolicy containing the equilibrium strategy for this round
        """
        logger.info(f"Solving {self.round} chunk for {iterations} iterations...")

        # Import here to avoid circular dependency
        from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver

        # Create OpenSpiel game for this subgame
        game = pyspiel.load_game("universal_poker", self.subgame_config)

        # Create solver
        solver = MatrixCFRSolver(game, use_sparse=True)

        # TODO Phase 8.4: If blueprint provided, initialize strategies from blueprint
        if self.blueprint is not None:
            logger.info(f"  Using blueprint policy from previous round")
            self._initialize_from_blueprint(solver)
        else:
            logger.info(f"  No blueprint - using uniform initialization")

        # Solve
        solver.solve(iterations=iterations, progress_interval=progress_interval)

        # Extract policy (use get_strategy_dict for proper dict format)
        policy_dict = solver.get_strategy_dict()

        logger.info(f"  {self.round} chunk solved: {len(policy_dict)} infosets")

        # Return as blueprint for next round
        return BlueprintPolicy(policy_dict)

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
    """

    def __init__(self, full_game_config: Dict[str, Any]):
        """
        Initialize chunked solver for full Hold'em game.

        Args:
            full_game_config: Complete Hold'em configuration (all rounds)
        """
        self.config = full_game_config
        self.chunks = ["preflop", "flop", "turn", "river"]
        self.policies = {}  # Store policy for each chunk

        logger.info(f"Initialized ChunkedSolver for {self.config.get('num_players', 2)}-player Hold'em")

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

        blueprint = None

        for chunk_name in self.chunks:
            logger.info(f"\n{'='*80}")
            logger.info(f"CHUNK: {chunk_name.upper()}")
            logger.info(f"{'='*80}")

            # Create subgame solver
            subgame = SubgameSolver(
                full_game_config=self.config,
                round_name=chunk_name,
                blueprint_policy=blueprint
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

            logger.info(f"✓ {chunk_name} chunk complete\n")

        logger.info("=" * 80)
        logger.info("ALL CHUNKS SOLVED")
        logger.info("=" * 80)

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
