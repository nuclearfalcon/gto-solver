"""
GPU MCCFR Configuration Management

Configuration classes for GPU-accelerated MCCFR poker solver.
Provides JSON serialization, validation, and defaults.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class GPUMCCFRConfig:
    """Configuration for GPU MCCFR poker solver.

    Attributes:
        num_players: Number of players (2-10)
        stacks: Stack sizes per player (chips)
        blinds: Blind sizes per player (chips)
        batch_size: Number of trajectories per iteration (default: 100)
        num_buckets: Total number of state buckets (default: 10000)
        num_hand_buckets: Hand strength buckets (default: 200)
        num_pot_buckets: Pot size buckets (default: 10)
        num_actions: Number of actions (default: 4 for fold/call/bet/all-in)
        seed: Random seed (default: 42)
    """

    # Required parameters
    num_players: int
    stacks: List[float]
    blinds: List[float]

    # Training parameters with defaults
    batch_size: int = 100
    num_buckets: int = 10000
    num_hand_buckets: int = 200
    num_pot_buckets: int = 10
    num_actions: int = 4
    seed: int = 42

    # MCCFR parameters (for compatibility with MCCFRConfig)
    use_linear_weighting: bool = False  # Weight strategy updates by iteration number
    discount_factor: float = 1.0  # For DCFR variants (1.0 = no discounting)
    prune_threshold: float = -3e8  # Prune actions with very negative regrets
    checkpoint_interval: int = 10_000  # Save checkpoint every N iterations

    # Performance tuning
    cache_clear_interval: int = 20  # Clear JAX cache every N iterations (0 = never, 10 = min memory, 20-50 = speed)

    # Optional metadata
    name: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate configuration parameters."""
        # Player count validation
        if not (2 <= self.num_players <= 10):
            raise ValueError(f"num_players must be between 2 and 10, got {self.num_players}")

        # Stack size validation
        if len(self.stacks) != self.num_players:
            raise ValueError(
                f"Number of stacks ({len(self.stacks)}) must equal num_players ({self.num_players})"
            )

        if any(s <= 0 for s in self.stacks):
            raise ValueError("All stacks must be positive")

        # Blind validation
        if len(self.blinds) != self.num_players:
            raise ValueError(
                f"Number of blinds ({len(self.blinds)}) must equal num_players ({self.num_players})"
            )

        if any(b < 0 for b in self.blinds):
            raise ValueError("All blinds must be non-negative")

        # Training parameter validation
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_buckets <= 0:
            raise ValueError(f"num_buckets must be positive, got {self.num_buckets}")

        if self.num_hand_buckets <= 0:
            raise ValueError(f"num_hand_buckets must be positive, got {self.num_hand_buckets}")

        if self.num_pot_buckets <= 0:
            raise ValueError(f"num_pot_buckets must be positive, got {self.num_pot_buckets}")

        if self.num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {self.num_actions}")

        # Bucketing consistency
        if self.num_hand_buckets * self.num_pot_buckets > self.num_buckets:
            raise ValueError(
                f"num_hand_buckets ({self.num_hand_buckets}) × num_pot_buckets ({self.num_pot_buckets}) "
                f"= {self.num_hand_buckets * self.num_pot_buckets} exceeds num_buckets ({self.num_buckets})"
            )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, filepath: str):
        """Save configuration to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GPUMCCFRConfig':
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            GPUMCCFRConfig instance
        """
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> 'GPUMCCFRConfig':
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            GPUMCCFRConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def get_stack_bb(self) -> List[float]:
        """Get stack sizes in big blinds.

        Returns:
            List of stack sizes in BB
        """
        big_blind = max(self.blinds)
        return [s / big_blind for s in self.stacks]

    def get_game_description(self) -> str:
        """Get human-readable game description.

        Returns:
            Game description string
        """
        stack_bb = self.get_stack_bb()
        if all(s == stack_bb[0] for s in stack_bb):
            # All stacks equal
            return f"{self.num_players}p_{int(stack_bb[0])}bb"
        else:
            # Asymmetric stacks
            stacks_str = "_".join(str(int(s)) for s in stack_bb)
            return f"{self.num_players}p_{stacks_str}bb"


# Preset configurations
PRESET_CONFIGS = {
    "2p_10bb_holdem": GPUMCCFRConfig(
        num_players=2,
        stacks=[1000.0, 1000.0],
        blinds=[50.0, 100.0],
        name="2p_10bb_holdem",
        description="Heads-up 10BB Hold'em"
    ),

    "2p_20bb_holdem": GPUMCCFRConfig(
        num_players=2,
        stacks=[2000.0, 2000.0],
        blinds=[50.0, 100.0],
        name="2p_20bb_holdem",
        description="Heads-up 20BB Hold'em"
    ),

    "3p_10bb_holdem": GPUMCCFRConfig(
        num_players=3,
        stacks=[1000.0, 1000.0, 1000.0],
        blinds=[50.0, 100.0, 0.0],
        name="3p_10bb_holdem",
        description="3-player 10BB Hold'em"
    ),

    "6p_10bb_holdem": GPUMCCFRConfig(
        num_players=6,
        stacks=[1000.0] * 6,
        blinds=[50.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        name="6p_10bb_holdem",
        description="6-max 10BB Hold'em"
    ),

    "9p_10bb_holdem": GPUMCCFRConfig(
        num_players=9,
        stacks=[1000.0] * 9,
        blinds=[50.0, 100.0] + [0.0] * 7,
        name="9p_10bb_holdem",
        description="9-handed 10BB Hold'em"
    ),

    "2p_5bb_holdem_fast": GPUMCCFRConfig(
        num_players=2,
        stacks=[500.0, 500.0],
        blinds=[50.0, 100.0],
        batch_size=200,  # Larger batches for speed
        num_buckets=5000,  # Fewer buckets for speed
        num_hand_buckets=100,
        num_pot_buckets=5,
        name="2p_5bb_holdem_fast",
        description="Fast 2p 5BB for testing"
    ),
}


def get_preset(name: str) -> GPUMCCFRConfig:
    """Get a preset configuration by name.

    Args:
        name: Preset name (e.g., "2p_10bb_holdem")

    Returns:
        GPUMCCFRConfig instance

    Raises:
        KeyError: If preset not found
    """
    if name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise KeyError(f"Preset '{name}' not found. Available: {available}")

    return PRESET_CONFIGS[name]


def list_presets() -> List[str]:
    """List available preset configurations.

    Returns:
        List of preset names
    """
    return list(PRESET_CONFIGS.keys())
