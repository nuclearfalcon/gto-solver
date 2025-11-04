"""
GTO Poker Training Problem Data Structure

Defines the PreflopProblem dataclass for storing preflop poker training scenarios
with GTO action frequencies extracted from trained CFR policies.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json


@dataclass
class PreflopProblem:
    """
    Represents a preflop poker training problem with GTO solution.

    Attributes:
        problem_id: Unique identifier for this problem
        num_players: Number of players in the game (2-9)
        hero_position: Hero's position (e.g., "BTN", "BB", "UTG")
        hero_cards: Hero's hole cards as list of strings (e.g., ["As", "Kh"])
        stacks_bb: Stack sizes in big blinds for each position
        pot_bb: Current pot size in big blinds
        action_history: List of actions taken before hero's decision
        active_players: List of positions still in the hand
        current_player: Position of player to act (should be hero)
        gto_strategy: Dictionary mapping action names to frequencies (0.0-1.0)
        tags: List of tags categorizing this problem
        hand_category: Category of hero's starting hand
        info_state_str: Original OpenSpiel information state string (optional)
    """

    problem_id: str
    num_players: int
    hero_position: str
    hero_cards: List[str]
    stacks_bb: Dict[str, float]
    pot_bb: float
    action_history: List[Dict[str, str]]
    active_players: List[str]
    current_player: str
    gto_strategy: Dict[str, float]
    tags: List[str] = field(default_factory=list)
    hand_category: str = ""
    info_state_str: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PreflopProblem':
        """Create PreflopProblem from dictionary."""
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'PreflopProblem':
        """Create PreflopProblem from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_gto_action_sorted(self) -> List[tuple]:
        """
        Get GTO actions sorted by frequency (highest first).

        Returns:
            List of (action_name, frequency) tuples sorted by frequency
        """
        return sorted(self.gto_strategy.items(), key=lambda x: x[1], reverse=True)

    def get_primary_action(self) -> tuple:
        """
        Get the primary (highest frequency) GTO action.

        Returns:
            Tuple of (action_name, frequency)
        """
        sorted_actions = self.get_gto_action_sorted()
        if sorted_actions:
            return sorted_actions[0]
        return ("Unknown", 0.0)

    def is_close_decision(self, threshold: float = 0.15) -> bool:
        """
        Check if this is a close decision (multiple actions above threshold).

        Args:
            threshold: Minimum frequency to consider an action viable

        Returns:
            True if 2+ actions have frequency >= threshold
        """
        viable_actions = [freq for freq in self.gto_strategy.values() if freq >= threshold]
        return len(viable_actions) >= 2

    def get_decision_complexity(self) -> str:
        """
        Categorize decision complexity.

        Returns:
            One of: "clear" (one dominant action), "mixed" (close decision),
                    "balanced" (3+ viable options)
        """
        if self.is_close_decision(threshold=0.20):
            viable_count = sum(1 for freq in self.gto_strategy.values() if freq >= 0.15)
            if viable_count >= 3:
                return "balanced"
            else:
                return "mixed"
        return "clear"

    def format_action_history(self) -> str:
        """
        Format action history as a readable string.

        Returns:
            Human-readable action sequence (e.g., "UTG raises to 2.5BB, MP folds")
        """
        if not self.action_history:
            return "No actions yet"

        actions_str = []
        for action in self.action_history:
            player = action.get('player', '?')
            action_type = action.get('action', '?')
            actions_str.append(f"{player} {action_type}")

        return ", ".join(actions_str)

    def format_gto_strategy(self, max_actions: int = 5) -> str:
        """
        Format GTO strategy as a readable string.

        Args:
            max_actions: Maximum number of actions to display

        Returns:
            Formatted strategy string
        """
        sorted_actions = self.get_gto_action_sorted()[:max_actions]
        strategy_parts = []

        for action, freq in sorted_actions:
            if freq > 0.001:  # Only show actions with >0.1% frequency
                strategy_parts.append(f"{action}: {freq:.1%}")

        return ", ".join(strategy_parts)

    def __str__(self) -> str:
        """Human-readable string representation."""
        primary_action, primary_freq = self.get_primary_action()

        # Handle varying number of hole cards
        if len(self.hero_cards) >= 2:
            cards_str = f"{self.hero_cards[0]}{self.hero_cards[1]}"
        elif len(self.hero_cards) == 1:
            cards_str = self.hero_cards[0]
        else:
            cards_str = "??"

        return (f"Problem {self.problem_id}: {self.num_players}p, "
                f"{self.hero_position} with {cards_str}, "
                f"Primary: {primary_action} ({primary_freq:.1%})")


def categorize_hand(card1: str, card2: str) -> str:
    """
    Categorize a starting hand.

    Args:
        card1: First card (e.g., "As")
        card2: Second card (e.g., "Kh")

    Returns:
        Hand category string
    """
    # Extract ranks and suits
    ranks = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
             '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

    rank1 = card1[0]
    suit1 = card1[1]
    rank2 = card2[0]
    suit2 = card2[1]

    r1 = ranks.get(rank1, 0)
    r2 = ranks.get(rank2, 0)

    # Ensure higher rank first
    if r2 > r1:
        r1, r2, rank1, rank2, suit1, suit2 = r2, r1, rank2, rank1, suit2, suit1

    is_suited = (suit1 == suit2)
    is_pair = (r1 == r2)

    # Categorize
    if is_pair:
        if r1 >= 10:  # JJ+
            return "premium_pair"
        elif r1 >= 7:  # 77-TT
            return "medium_pair"
        else:  # 22-66
            return "small_pair"

    # Non-pairs
    if r1 >= 12 and r2 >= 10:  # Broadway (T-A)
        return "broadway_suited" if is_suited else "broadway_offsuit"

    if r1 >= 11 and r2 >= 10:  # High cards
        return "high_cards_suited" if is_suited else "high_cards"

    # Connectors
    if abs(r1 - r2) <= 1:
        return "suited_connector" if is_suited else "connector"

    # Gappers
    if abs(r1 - r2) <= 3:
        return "suited_gapper" if is_suited else "gapper"

    # Suited aces
    if r1 == 14 and is_suited:
        return "suited_ace"

    # Default
    return "other"


def create_problem_id(num_players: int, sequence_num: int, prefix: str = "") -> str:
    """
    Create a standardized problem ID.

    Args:
        num_players: Number of players
        sequence_num: Sequential number for this problem
        prefix: Optional prefix (e.g., config name)

    Returns:
        Problem ID string (e.g., "2p_00123" or "6max_00456")
    """
    if prefix:
        return f"{prefix}_{sequence_num:05d}"
    else:
        return f"{num_players}p_{sequence_num:05d}"


# Example usage
if __name__ == '__main__':
    print("GTO Problem - Test Cases")
    print("=" * 60)

    # Create a sample problem
    problem = PreflopProblem(
        problem_id="6p_00001",
        num_players=6,
        hero_position="BTN",
        hero_cards=["As", "Kh"],
        stacks_bb={"UTG": 10.0, "MP": 10.0, "CO": 10.0, "BTN": 10.0, "SB": 9.5, "BB": 9.0},
        pot_bb=1.5,
        action_history=[
            {"player": "UTG", "action": "Fold"},
            {"player": "MP", "action": "Raise to 2.5BB"},
            {"player": "CO", "action": "Fold"}
        ],
        active_players=["MP", "BTN", "SB", "BB"],
        current_player="BTN",
        gto_strategy={
            "Fold": 0.10,
            "Call": 0.25,
            "Raise to 7BB": 0.55,
            "All-in": 0.10
        },
        tags=["facing_raise", "position_advantage", "premium_hand"],
        hand_category="broadway_offsuit"
    )

    print("\nSample Problem:")
    print(problem)
    print()

    print("Action History:")
    print(f"  {problem.format_action_history()}")
    print()

    print("GTO Strategy:")
    print(f"  {problem.format_gto_strategy()}")
    print()

    print("Primary Action:")
    action, freq = problem.get_primary_action()
    print(f"  {action} ({freq:.1%})")
    print()

    print("Decision Complexity:")
    print(f"  {problem.get_decision_complexity()}")
    print(f"  Is close decision: {problem.is_close_decision()}")
    print()

    # Test JSON serialization
    print("JSON Export:")
    json_str = problem.to_json()
    print(json_str[:200] + "...")
    print()

    # Test JSON deserialization
    problem2 = PreflopProblem.from_json(json_str)
    print(f"Deserialized: {problem2}")
    print()

    # Test hand categorization
    print("Hand Categorization:")
    test_hands = [
        (["As", "Ah"], "premium_pair"),
        (["Kh", "Qs"], "broadway_suited"),
        (["9h", "8h"], "suited_connector"),
        (["Ad", "5d"], "suited_ace")
    ]
    for cards, expected in test_hands:
        category = categorize_hand(cards[0], cards[1])
        status = "✓" if category == expected else "✗"
        print(f"  {status} {cards[0]}{cards[1]} → {category} (expected: {expected})")

    print()
    print("=" * 60)
    print("All tests passed!")
