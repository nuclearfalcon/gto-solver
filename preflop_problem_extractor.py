"""
Preflop Problem Extractor

Extracts preflop poker training problems from trained CFR policies.
Parses information state strings, queries GTO strategies, and builds
structured problem database.

Usage:
    from preflop_problem_extractor import PreflopProblemExtractor

    extractor = PreflopProblemExtractor('path/to/policy.pkl', game_config)
    problems = extractor.extract_problems()
    extractor.save_problems(problems, 'output.json')
"""

import re
import pickle
from typing import List, Dict, Optional, Tuple
import pyspiel

from gto_problem import PreflopProblem, categorize_hand, create_problem_id
from position_utils import get_position_name
from game_config import PokerGameConfig


class InfoStateParser:
    """Parses OpenSpiel information state strings into structured data."""

    # Regex pattern for parsing info state strings
    # Format: [Round X][Player: X][Pot: X][Money: X Y ...][Private: Xc][Public: ...][Sequences: ...]
    PATTERN = re.compile(
        r'\[Round (?P<round>\d+)\]'
        r'\[Player: (?P<player>\d+)\]'
        r'\[Pot: (?P<pot>\d+)\]'
        r'\[Money: (?P<money>[\d ]+)\]'
        r'\[Private: (?P<private>\w*)\]'
        r'\[Public: (?P<public>[\w ]*)\]'
        r'\[Sequences: (?P<sequences>[\w|]*)\]'
    )

    def __init__(self, num_players: int, big_blind: int = 100):
        """
        Initialize parser.

        Args:
            num_players: Number of players in the game
            big_blind: Big blind size (for converting to BB units)
        """
        self.num_players = num_players
        self.big_blind = big_blind

    def parse(self, info_state_str: str) -> Optional[Dict]:
        """
        Parse information state string into structured data.

        Args:
            info_state_str: OpenSpiel information state string

        Returns:
            Dictionary with parsed data, or None if parsing fails
        """
        match = self.PATTERN.match(info_state_str)
        if not match:
            return None

        data = match.groupdict()

        # Parse round number
        round_num = int(data['round'])

        # Parse current player
        current_player = int(data['player'])

        # Parse pot size
        pot_chips = int(data['pot'])
        pot_bb = pot_chips / self.big_blind

        # Parse stack sizes
        money_str = data['money'].strip()
        stacks_chips = [int(x) for x in money_str.split()]
        stacks_bb = [s / self.big_blind for s in stacks_chips]

        # Parse private cards
        private_cards = self._parse_cards(data['private'])

        # Parse public cards
        public_cards = self._parse_cards(data['public'])

        # Parse action sequence
        sequences = data['sequences']
        action_history = self._parse_sequences(sequences)

        return {
            'round': round_num,
            'current_player': current_player,
            'pot_chips': pot_chips,
            'pot_bb': pot_bb,
            'stacks_chips': stacks_chips,
            'stacks_bb': stacks_bb,
            'private_cards': private_cards,
            'public_cards': public_cards,
            'sequences': sequences,
            'action_history': action_history,
            'info_state_str': info_state_str
        }

    def _parse_cards(self, card_str: str) -> List[str]:
        """
        Parse card string into list of cards.

        Args:
            card_str: String like "2c3d" or "AhKs"

        Returns:
            List of card strings like ["2c", "3d"]
        """
        if not card_str or card_str.strip() == '':
            return []

        # Cards are 2 characters each: rank + suit
        cards = []
        i = 0
        while i < len(card_str):
            if i + 1 < len(card_str):
                cards.append(card_str[i:i+2])
                i += 2
            else:
                break

        return cards

    def _parse_sequences(self, sequences: str) -> List[Dict[str, str]]:
        """
        Parse action sequence string into structured actions.

        Args:
            sequences: String like "cr300" or "r200r500" or "cc|cr100"

        Returns:
            List of action dictionaries
        """
        if not sequences or sequences.strip() == '':
            return []

        # Split by rounds (separated by |)
        rounds = sequences.split('|')

        actions = []
        player_turn = 1 if self.num_players == 2 else 0  # BTN acts first in heads-up

        for round_idx, round_seq in enumerate(rounds):
            if not round_seq:
                continue

            # Parse individual actions in this round
            i = 0
            while i < len(round_seq):
                action_char = round_seq[i]

                if action_char == 'c':
                    # Call
                    actions.append({
                        'player_id': player_turn,
                        'action_type': 'call'
                    })
                    i += 1
                    player_turn = (player_turn + 1) % self.num_players

                elif action_char == 'f':
                    # Fold
                    actions.append({
                        'player_id': player_turn,
                        'action_type': 'fold'
                    })
                    i += 1
                    player_turn = (player_turn + 1) % self.num_players

                elif action_char == 'r':
                    # Raise - extract amount
                    i += 1
                    amount_str = ''
                    while i < len(round_seq) and round_seq[i].isdigit():
                        amount_str += round_seq[i]
                        i += 1

                    amount = int(amount_str) if amount_str else 0
                    actions.append({
                        'player_id': player_turn,
                        'action_type': 'raise',
                        'amount': amount
                    })
                    player_turn = (player_turn + 1) % self.num_players

                else:
                    # Unknown action, skip
                    i += 1

        return actions


class PreflopProblemExtractor:
    """Extracts preflop training problems from CFR policies."""

    def __init__(self, policy_path: str, game_config: PokerGameConfig):
        """
        Initialize extractor.

        Args:
            policy_path: Path to pickled TabularPolicy file
            game_config: Game configuration object
        """
        self.policy_path = policy_path
        self.game_config = game_config
        self.policy = None
        self.game = None
        self.parser = None

        self._load_policy()
        self._initialize()

    def _load_policy(self):
        """Load policy from pickle file."""
        with open(self.policy_path, 'rb') as f:
            self.policy = pickle.load(f)
        print(f"Loaded policy with {len(self.policy.state_lookup)} states")

    def _initialize(self):
        """Initialize game and parser."""
        self.game = self.game_config.create_game()

        # Get big blind size for BB conversions
        blinds = self.game_config.blinds
        self.big_blind = max(blinds) if blinds else 100

        self.parser = InfoStateParser(
            num_players=self.game_config.num_players,
            big_blind=self.big_blind
        )

    def extract_problems(self,
                         preflop_only: bool = True,
                         max_problems: Optional[int] = None) -> List[PreflopProblem]:
        """
        Extract training problems from policy.

        Args:
            preflop_only: Only extract preflop (Round 0) problems
            max_problems: Maximum number of problems to extract (None = all)

        Returns:
            List of PreflopProblem objects
        """
        problems = []
        problem_counter = 0

        print(f"Extracting problems from {len(self.policy.state_lookup)} states...")

        for info_state_str in self.policy.state_lookup.keys():
            # Filter for preflop if requested
            if preflop_only and '[Round 0]' not in info_state_str:
                continue

            # Parse info state
            parsed = self.parser.parse(info_state_str)
            if not parsed:
                continue

            # Skip if not Round 0 and preflop_only is True
            if preflop_only and parsed['round'] != 0:
                continue

            # Build problem
            problem = self._build_problem(parsed, problem_counter)
            if problem:
                problems.append(problem)
                problem_counter += 1

            # Check max limit
            if max_problems and len(problems) >= max_problems:
                break

        print(f"Extracted {len(problems)} problems")
        return problems

    def _build_problem(self, parsed: Dict, sequence_num: int) -> Optional[PreflopProblem]:
        """
        Build a PreflopProblem from parsed data.

        Args:
            parsed: Parsed information state data
            sequence_num: Sequential number for problem ID

        Returns:
            PreflopProblem object or None if construction fails
        """
        try:
            # Get position names
            current_player_id = parsed['current_player']
            hero_position = get_position_name(current_player_id, self.game_config.num_players)

            # Build stacks_bb dict
            stacks_bb = {}
            for i, stack_bb in enumerate(parsed['stacks_bb']):
                pos = get_position_name(i, self.game_config.num_players)
                stacks_bb[pos] = stack_bb

            # Determine active players (players who haven't folded)
            active_players = self._get_active_players(parsed['action_history'])

            # Format action history for display
            action_history_formatted = self._format_action_history(parsed['action_history'])

            # Get GTO strategy from policy
            gto_strategy = self._get_gto_strategy(parsed['info_state_str'])
            if not gto_strategy:
                return None

            # Categorize hand
            hero_cards = parsed['private_cards']
            if len(hero_cards) >= 2:
                hand_category = categorize_hand(hero_cards[0], hero_cards[1])
            else:
                hand_category = "unknown"

            # Generate tags
            tags = self._generate_tags(parsed, active_players, gto_strategy)

            # Create problem ID
            problem_id = create_problem_id(
                self.game_config.num_players,
                sequence_num,
                prefix=f"{self.game_config.num_players}p_preflop"
            )

            # Build problem object
            problem = PreflopProblem(
                problem_id=problem_id,
                num_players=self.game_config.num_players,
                hero_position=hero_position,
                hero_cards=hero_cards,
                stacks_bb=stacks_bb,
                pot_bb=parsed['pot_bb'],
                action_history=action_history_formatted,
                active_players=active_players,
                current_player=hero_position,
                gto_strategy=gto_strategy,
                tags=tags,
                hand_category=hand_category,
                info_state_str=parsed['info_state_str']
            )

            return problem

        except Exception as e:
            print(f"Warning: Failed to build problem: {e}")
            return None

    def _get_active_players(self, action_history: List[Dict]) -> List[str]:
        """Determine which players are still active (haven't folded)."""
        folded_players = set()

        for action in action_history:
            if action.get('action_type') == 'fold':
                player_id = action.get('player_id')
                if player_id is not None:
                    pos = get_position_name(player_id, self.game_config.num_players)
                    folded_players.add(pos)

        # All players minus folded ones
        all_positions = [get_position_name(i, self.game_config.num_players)
                        for i in range(self.game_config.num_players)]

        active = [pos for pos in all_positions if pos not in folded_players]
        return active

    def _format_action_history(self, action_history: List[Dict]) -> List[Dict[str, str]]:
        """Format action history for human readability."""
        formatted = []

        for action in action_history:
            player_id = action.get('player_id')
            action_type = action.get('action_type', 'unknown')
            amount = action.get('amount')

            pos = get_position_name(player_id, self.game_config.num_players)

            if action_type == 'call':
                action_str = "Call"
            elif action_type == 'fold':
                action_str = "Fold"
            elif action_type == 'raise':
                amount_bb = amount / self.big_blind if amount else 0
                action_str = f"Raise to {amount_bb:.1f}BB"
            else:
                action_str = action_type.capitalize()

            formatted.append({
                'player': pos,
                'action': action_str
            })

        return formatted

    def _get_gto_strategy(self, info_state_str: str) -> Optional[Dict[str, float]]:
        """
        Get GTO strategy from policy for this state.

        Args:
            info_state_str: Information state string

        Returns:
            Dictionary mapping action names to probabilities
        """
        # Look up state in policy
        if info_state_str not in self.policy.state_lookup:
            return None

        # Get action probabilities from policy
        state_index = self.policy.state_lookup[info_state_str]
        action_probs_array = self.policy.action_probability_array[state_index]
        legal_actions_mask = self.policy.legal_actions_mask[state_index]

        # Build strategy dict
        strategy = {}
        for action_id, prob in enumerate(action_probs_array):
            if legal_actions_mask[action_id]:
                # Get action name
                action_name = self._get_action_name(action_id)
                if action_name and prob > 0.001:  # Only include actions with >0.1% frequency
                    strategy[action_name] = float(prob)

        return strategy if strategy else None

    def _get_action_name(self, action_id: int) -> Optional[str]:
        """
        Convert action ID to human-readable name.

        This is a simplified version - ideally we'd reconstruct the state
        and use state.action_to_string(), but for now we'll use generic names.
        """
        # For betting abstractions like FCPA:
        # Typically: 0=Fold, 1=Call, 2=Half-pot, 3=Pot, 4=All-in
        action_names = {
            0: "Fold",
            1: "Call/Check",
            2: "Bet/Raise (Small)",
            3: "Bet/Raise (Pot)",
            4: "All-in"
        }

        return action_names.get(action_id, f"Action_{action_id}")

    def _generate_tags(self, parsed: Dict, active_players: List[str],
                       gto_strategy: Dict[str, float]) -> List[str]:
        """Generate tags for categorizing the problem."""
        tags = []

        # Add round tag
        tags.append(f"round_{parsed['round']}")
        tags.append("preflop")

        # Check if facing a raise
        action_history = parsed['action_history']
        if any(a.get('action_type') == 'raise' for a in action_history):
            tags.append("facing_raise")

        # Check for multiway pot
        if len(active_players) > 2:
            tags.append("multiway")

        # Check if close decision
        viable_actions = [freq for freq in gto_strategy.values() if freq >= 0.15]
        if len(viable_actions) >= 2:
            tags.append("close_decision")

        # Check pot size categories
        pot_bb = parsed['pot_bb']
        if pot_bb <= 3:
            tags.append("small_pot")
        elif pot_bb >= 10:
            tags.append("large_pot")

        return tags

    def save_problems(self, problems: List[PreflopProblem], output_path: str):
        """
        Save problems to JSON file.

        Args:
            problems: List of PreflopProblem objects
            output_path: Path to output JSON file
        """
        import json

        # Convert problems to dicts
        problems_data = [p.to_dict() for p in problems]

        with open(output_path, 'w') as f:
            json.dump(problems_data, f, indent=2)

        print(f"Saved {len(problems)} problems to {output_path}")


# Example usage
if __name__ == '__main__':
    print("Preflop Problem Extractor - Test")
    print("=" * 70)

    # Test with existing policy
    policy_path = 'results/cfr_plus_2p_5bb_fchpa_20251029_205453_policy.pkl'
    config_path = 'configs/2p_5bb_fchpa_tiny.json'

    print(f"\nLoading config: {config_path}")
    config = PokerGameConfig.from_json(config_path)

    print(f"Creating extractor...")
    extractor = PreflopProblemExtractor(policy_path, config)

    print(f"\nExtracting preflop problems...")
    problems = extractor.extract_problems(preflop_only=True, max_problems=10)

    print(f"\nSample problems:")
    for i, problem in enumerate(problems[:5]):
        print(f"\n{i+1}. {problem}")
        print(f"   Action history: {problem.format_action_history()}")
        print(f"   GTO strategy: {problem.format_gto_strategy()}")
        print(f"   Tags: {', '.join(problem.tags)}")

    # Save to file
    output_path = 'test_problems.json'
    print(f"\nSaving to {output_path}...")
    extractor.save_problems(problems, output_path)

    print("\n" + "=" * 70)
    print("Test complete!")
