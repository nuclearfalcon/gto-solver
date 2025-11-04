#!/usr/bin/env python3
"""
Poker Game Configuration Management

Provides a structured way to define and manage OpenSpiel universal_poker
game configurations for GTO solving.

Requirements:
    source ~/open_spiel/venv/bin/activate

Example:
    # Create from code
    config = PokerGameConfig(
        num_players=2,
        stack_sizes=[1000, 1000],
        blinds=[100, 50],
        betting_abstraction='fchpa_1.5x'
    )

    # Load from file
    config = PokerGameConfig.from_json('configs/2p_10bb.json')

    # Create OpenSpiel game
    game = config.create_game()
"""

import json
import pyspiel
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class SolverSettings:
    """
    Optional solver configuration settings.

    These settings can be specified in config files to provide defaults
    for solver behavior. CLI arguments override these settings.

    Attributes:
        output_dir: Directory for results files
        checkpoint_dir: Directory for checkpoint files
        results_prefix: Custom prefix for result filenames
        best_policy_filename: Filename for best policy (None = auto-generate)
        max_iterations: Maximum iterations to run
        checkpoint_interval: Save checkpoint every N iterations
        progress_interval: Show progress update every N iterations
        exploitability_schedule: "adaptive" or "fixed"
        check_interval: Fixed interval for exploitability checks (if not adaptive)
        save_best_policy: Whether to automatically save best policy
        auto_resume: Automatically resume from checkpoint without prompt
        exploit_history_size: Number of recent exploitability tests to show (default: 10)
    """
    # Output paths
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    results_prefix: Optional[str] = None
    best_policy_filename: Optional[str] = None

    # Iteration settings
    max_iterations: Optional[int] = None
    checkpoint_interval: Optional[int] = None
    progress_interval: int = 100

    # Exploitability settings
    exploitability_schedule: str = "adaptive"  # "adaptive" or "fixed"
    check_interval: Optional[int] = None

    # Best policy tracking
    save_best_policy: bool = True

    # Auto-resume
    auto_resume: bool = False

    # Display settings
    exploit_history_size: int = 10


@dataclass
class PokerGameConfig:
    """
    Configuration for a universal_poker game.

    Attributes:
        num_players: Number of players (2-10)
        stack_sizes: Stack size for each player (in chips)
        blinds: Blind/ante amount for each player (in chips)
        betting_abstraction: One of 'fc', 'fcpa', 'fchpa', 'fchpa_1.5x', 'fullgame'
        num_rounds: Number of betting rounds (default: 4 for NLHE)
        num_suits: Number of suits (default: 4)
        num_ranks: Number of ranks (default: 13)
        num_hole_cards: Cards dealt to each player (default: 2)
        num_board_cards: Board cards per round (default: '0 3 1 1' for NLHE)
        first_player: Who acts first each round (default: '2 1 1 1' for NLHE)
        betting_type: 'nolimit', 'potlimit', or 'limit' (default: 'nolimit')
        description: Human-readable description
    """

    # Required parameters
    num_players: int
    stack_sizes: List[int]
    blinds: List[int]

    # Abstraction (default to fchpa)
    betting_abstraction: str = 'fchpa'

    # Standard NLHE defaults
    num_rounds: int = 4
    num_suits: int = 4
    num_ranks: int = 13
    num_hole_cards: int = 2
    num_board_cards: str = '0 3 1 1'  # Preflop, Flop, Turn, River
    first_player: str = '2 1 1 1'     # SB/BB act first preflop, BB acts first postflop
    betting_type: str = 'nolimit'

    # Metadata
    description: str = ""

    # Optional solver settings
    solver_settings: Optional[SolverSettings] = None

    def __post_init__(self):
        """Validate configuration."""
        # Validate num_players
        if not (2 <= self.num_players <= 10):
            raise ValueError(f"num_players must be 2-10, got {self.num_players}")

        # Validate stack_sizes
        if len(self.stack_sizes) != self.num_players:
            raise ValueError(
                f"stack_sizes must have {self.num_players} values, "
                f"got {len(self.stack_sizes)}"
            )

        # Validate blinds
        if len(self.blinds) != self.num_players:
            raise ValueError(
                f"blinds must have {self.num_players} values, "
                f"got {len(self.blinds)}"
            )

        # Validate abstraction
        valid_abstractions = ['fc', 'fcpa', 'fchpa', 'fullgame']
        if self.betting_abstraction not in valid_abstractions:
            raise ValueError(
                f"betting_abstraction must be one of {valid_abstractions}, "
                f"got {self.betting_abstraction}"
            )

    def to_openspiel_config(self) -> Dict[str, Any]:
        """
        Convert to OpenSpiel universal_poker configuration dict.

        Returns:
            Configuration dict suitable for pyspiel.load_game('universal_poker', config)
        """
        # Use betting abstraction directly
        openspiel_abstraction = self.betting_abstraction

        # Convert lists to space-separated strings
        stack_str = ' '.join(map(str, self.stack_sizes))
        blind_str = ' '.join(map(str, self.blinds))

        config = {
            'betting': self.betting_type,
            'numPlayers': self.num_players,
            'numRounds': self.num_rounds,
            'blind': blind_str,
            'firstPlayer': self.first_player,
            'numSuits': self.num_suits,
            'numRanks': self.num_ranks,
            'numHoleCards': self.num_hole_cards,
            'numBoardCards': self.num_board_cards,
            'stack': stack_str,
            'bettingAbstraction': openspiel_abstraction,
        }

        return config

    def uses_custom_abstraction(self) -> bool:
        """Check if this config uses a custom betting abstraction."""
        return False  # No longer using custom abstractions

    def create_game(self):
        """
        Create OpenSpiel game instance.

        Returns:
            OpenSpiel game object
        """
        config = self.to_openspiel_config()
        return pyspiel.load_game('universal_poker', config)

    def get_big_blinds(self) -> float:
        """
        Calculate effective stack size in big blinds.

        Returns:
            Average stack size in big blinds
        """
        big_blind = max(self.blinds)
        avg_stack = sum(self.stack_sizes) / len(self.stack_sizes)
        return avg_stack / big_blind if big_blind > 0 else 0

    def get_short_description(self) -> str:
        """
        Get short description for filenames.

        Returns:
            Description like "2p_10bb_fchpa_1.5x"
        """
        bb = int(self.get_big_blinds())
        return f"{self.num_players}p_{bb}bb_{self.betting_abstraction}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self, filepath: str):
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PokerGameConfig':
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            PokerGameConfig instance
        """
        # Handle solver_settings if present
        data_copy = data.copy()
        if 'solver_settings' in data_copy and data_copy['solver_settings'] is not None:
            data_copy['solver_settings'] = SolverSettings(**data_copy['solver_settings'])
        return cls(**data_copy)

    @classmethod
    def from_json(cls, filepath: str) -> 'PokerGameConfig':
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            PokerGameConfig instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation."""
        bb = self.get_big_blinds()
        return (
            f"PokerGameConfig({self.num_players}p, {bb:.1f}BB, "
            f"{self.betting_abstraction})"
        )


# Preset configurations

def create_2p_heads_up(stack_bb: int = 100, abstraction: str = 'fchpa') -> PokerGameConfig:
    """
    Create standard heads-up (2-player) configuration.

    Args:
        stack_bb: Stack size in big blinds
        abstraction: Betting abstraction to use

    Returns:
        PokerGameConfig for heads-up play
    """
    big_blind = 100
    small_blind = 50
    stack = big_blind * stack_bb

    return PokerGameConfig(
        num_players=2,
        stack_sizes=[stack, stack],
        blinds=[big_blind, small_blind],
        betting_abstraction=abstraction,
        description=f"{stack_bb}BB Heads-up NLHE with {abstraction}",
    )


def create_3p_config(stack_bb: int = 100, abstraction: str = 'fchpa') -> PokerGameConfig:
    """
    Create 3-player configuration.

    Args:
        stack_bb: Stack size in big blinds
        abstraction: Betting abstraction to use

    Returns:
        PokerGameConfig for 3-player game
    """
    big_blind = 100
    small_blind = 50
    stack = big_blind * stack_bb

    return PokerGameConfig(
        num_players=3,
        stack_sizes=[stack, stack, stack],
        blinds=[big_blind, small_blind, 0],  # BB, SB, Button
        betting_abstraction=abstraction,
        description=f"{stack_bb}BB 3-player NLHE with {abstraction}",
    )


def create_6max_config(stack_bb: int = 100, abstraction: str = 'fchpa') -> PokerGameConfig:
    """
    Create 6-max configuration.

    Args:
        stack_bb: Stack size in big blinds
        abstraction: Betting abstraction to use

    Returns:
        PokerGameConfig for 6-max game
    """
    big_blind = 100
    small_blind = 50
    stack = big_blind * stack_bb

    return PokerGameConfig(
        num_players=6,
        stack_sizes=[stack] * 6,
        blinds=[big_blind, small_blind, 0, 0, 0, 0],
        betting_abstraction=abstraction,
        description=f"{stack_bb}BB 6-max NLHE with {abstraction}",
    )


def create_validation_config() -> PokerGameConfig:
    """
    Create small configuration for pipeline validation.

    Returns:
        PokerGameConfig with small game tree (2p, 5BB, shallow)
    """
    return create_2p_heads_up(stack_bb=5, abstraction='fchpa')
