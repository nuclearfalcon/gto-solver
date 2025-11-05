#!/usr/bin/env python3
"""
Scenario Configuration for Conditional Solving

Extends PokerGameConfig with hero-specific parameters for targeted solving.

Requirements:
    source ~/open_spiel/venv/bin/activate

Example:
    scenario = ScenarioConfig.from_json("scenarios/hero_ak_btn.json")
    scenario = ScenarioConfig(
        game_config=base_config,
        hero_position=0,
        hero_cards=["As", "Kh"],
        depth_limit=1
    )
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import json

from game_config import PokerGameConfig


# Card notation mapping
RANKS = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
         'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
SUITS = {'h': 0, 'd': 1, 'c': 2, 's': 3}  # hearts, diamonds, clubs, spades

RANK_NAMES = {v: k for k, v in RANKS.items()}
SUIT_NAMES = {v: k for k, v in SUITS.items()}


def card_string_to_int(card_str: str) -> int:
    """
    Convert card notation (e.g., "As", "Kh") to OpenSpiel card integer.

    OpenSpiel uses: card_id = rank * num_suits + suit

    Args:
        card_str: Card in format "Rs" where R is rank (2-9,T,J,Q,K,A) and s is suit (h,d,c,s)

    Returns:
        Integer card ID for OpenSpiel

    Example:
        "As" -> 12 * 4 + 3 = 51
        "2h" -> 0 * 4 + 0 = 0
    """
    if len(card_str) != 2:
        raise ValueError(f"Card string must be 2 characters, got: {card_str}")

    rank_char = card_str[0].upper()
    suit_char = card_str[1].lower()

    if rank_char not in RANKS:
        raise ValueError(f"Invalid rank: {rank_char}. Must be 2-9, T, J, Q, K, or A")
    if suit_char not in SUITS:
        raise ValueError(f"Invalid suit: {suit_char}. Must be h, d, c, or s")

    rank = RANKS[rank_char]
    suit = SUITS[suit_char]

    return rank * 4 + suit


def card_int_to_string(card_int: int, num_suits: int = 4) -> str:
    """
    Convert OpenSpiel card integer to string notation.

    Args:
        card_int: Integer card ID
        num_suits: Number of suits in the deck (usually 4)

    Returns:
        Card string like "As", "Kh"

    Example:
        51 -> "As"
        0 -> "2h"
    """
    rank = card_int // num_suits
    suit = card_int % num_suits

    rank_char = RANK_NAMES.get(rank, '?')
    suit_char = SUIT_NAMES.get(suit, '?')

    return f"{rank_char}{suit_char}"


def parse_cards(cards_str: str) -> List[int]:
    """
    Parse space-separated card string to list of card integers.

    Args:
        cards_str: Space-separated cards like "As Kh" or "2h 2d"
                   Can also be raw card integers like "5" or "0 3"

    Returns:
        List of card integers

    Example:
        "As Kh" -> [51, 47]
        "5" -> [5]
        "0 3" -> [0, 3]
    """
    card_strings = cards_str.strip().split()
    result = []
    for c in card_strings:
        # Try to parse as integer first (for tiny decks)
        try:
            card_int = int(c)
            result.append(card_int)
        except ValueError:
            # Not an integer, try standard notation
            result.append(card_string_to_int(c))
    return result


@dataclass
class ScenarioConfig:
    """
    Configuration for a specific hero scenario.

    Extends base game configuration with hero-specific parameters
    for conditional solving.
    """

    game_config: PokerGameConfig
    hero_position: int
    hero_cards: List[int]  # List of card integers
    depth_limit: Optional[int] = None  # None = solve all streets, 1 = preflop only, etc.

    def __post_init__(self):
        """Validate configuration."""
        # Validate hero position
        if self.hero_position < 0 or self.hero_position >= self.game_config.num_players:
            raise ValueError(
                f"hero_position must be 0-{self.game_config.num_players-1}, "
                f"got {self.hero_position}"
            )

        # Validate hero cards
        if len(self.hero_cards) != self.game_config.num_hole_cards:
            raise ValueError(
                f"hero_cards must have {self.game_config.num_hole_cards} cards, "
                f"got {len(self.hero_cards)}"
            )

        # Check for duplicate cards
        if len(set(self.hero_cards)) != len(self.hero_cards):
            raise ValueError(f"hero_cards contains duplicates: {self.hero_cards}")

        # Validate card IDs are in valid range
        max_card = self.game_config.num_ranks * self.game_config.num_suits - 1
        for card in self.hero_cards:
            if card < 0 or card > max_card:
                raise ValueError(
                    f"Invalid card ID {card}. Must be 0-{max_card} for this deck"
                )

        # Validate depth limit
        if self.depth_limit is not None:
            if self.depth_limit < 1 or self.depth_limit > self.game_config.num_rounds:
                raise ValueError(
                    f"depth_limit must be 1-{self.game_config.num_rounds}, "
                    f"got {self.depth_limit}"
                )

    @classmethod
    def from_game_config(
        cls,
        game_config: PokerGameConfig,
        hero_position: int,
        hero_cards_str: str,
        depth_limit: Optional[int] = None
    ):
        """
        Create ScenarioConfig from PokerGameConfig and card string.

        Args:
            game_config: Base poker game configuration
            hero_position: Hero's seat (0-indexed)
            hero_cards_str: Hero's cards as space-separated string (e.g., "As Kh")
            depth_limit: Number of streets to solve (None = all streets)

        Returns:
            ScenarioConfig instance

        Example:
            config = PokerGameConfig.from_json("configs/2p_5bb_fchpa.json")
            scenario = ScenarioConfig.from_game_config(config, 0, "As Kh", depth_limit=1)
        """
        hero_cards = parse_cards(hero_cards_str)
        return cls(
            game_config=game_config,
            hero_position=hero_position,
            hero_cards=hero_cards,
            depth_limit=depth_limit
        )

    @classmethod
    def from_json(cls, json_path: str):
        """
        Load ScenarioConfig from JSON file.

        Expected JSON format:
        {
            "game_config": "configs/2p_5bb_fchpa.json",
            "hero_position": 0,
            "hero_cards": "As Kh",
            "depth_limit": 1
        }

        Args:
            json_path: Path to JSON configuration file

        Returns:
            ScenarioConfig instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Load base game config
        game_config = PokerGameConfig.from_json(data['game_config'])

        # Parse hero cards
        hero_cards = parse_cards(data['hero_cards'])

        return cls(
            game_config=game_config,
            hero_position=data['hero_position'],
            hero_cards=hero_cards,
            depth_limit=data.get('depth_limit')
        )

    def to_json(self, json_path: str, game_config_path: str):
        """
        Save ScenarioConfig to JSON file.

        Args:
            json_path: Path to save scenario JSON
            game_config_path: Path to reference for game config
        """
        data = {
            'game_config': game_config_path,
            'hero_position': self.hero_position,
            'hero_cards': self.get_hero_cards_string(),
            'depth_limit': self.depth_limit
        }

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_hero_cards_string(self) -> str:
        """Get hero cards as human-readable string."""
        return ' '.join(card_int_to_string(c, self.game_config.num_suits)
                       for c in self.hero_cards)

    def __str__(self) -> str:
        """String representation."""
        cards_str = self.get_hero_cards_string()
        depth_str = f"{self.depth_limit} streets" if self.depth_limit else "all streets"

        return (
            f"Scenario(hero=P{self.hero_position}, cards={cards_str}, "
            f"depth={depth_str}, game={self.game_config.get_short_description()})"
        )
