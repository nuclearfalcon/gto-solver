"""
Simplified Bucketing for Preflop-Only Poker

Much simpler than full-game bucketing since we only need:
- Hand strength (169 combinations → ~50 strategic buckets)
- Pot size (3-5 categories)
- Position/action information

Total buckets: ~250-500 (vs 10,000 for full game)
"""

import jax
import jax.numpy as jnp
from matrix_cfr.preflop_holdem_jax import PreflopState


@jax.jit
def card_to_rank(card_index: int) -> int:
    """Convert card index (0-51) to rank (0-12)."""
    return card_index % 13


@jax.jit
def card_to_suit(card_index: int) -> int:
    """Convert card index (0-51) to suit (0-3)."""
    return card_index // 13


@jax.jit
def compute_preflop_hand_bucket(hole_cards: jnp.ndarray, num_buckets: int = 50) -> int:
    """
    Bucket preflop hands into strategic categories.

    Uses standard poker hand groupings:
    - Pairs: AA, KK, QQ, JJ, TT, 99-22 (group low pairs)
    - Suited: AKs, AQs, AJs, KQs, etc.
    - Offsuit: AKo, AQo, AJo, KQo, etc.

    Args:
        hole_cards: (2,) array of card indices
        num_buckets: Number of hand buckets (default 50)

    Returns:
        Bucket index in range [0, num_buckets)
    """
    card1, card2 = hole_cards[0], hole_cards[1]
    rank1, rank2 = card_to_rank(card1), card_to_rank(card2)
    suit1, suit2 = card_to_suit(card1), card_to_suit(card2)

    # Normalize so rank1 >= rank2
    high_rank = jnp.maximum(rank1, rank2)
    low_rank = jnp.minimum(rank1, rank2)

    # Features
    is_pair = (rank1 == rank2)
    is_suited = (suit1 == suit2)
    gap = high_rank - low_rank

    # Strategic bucketing
    # Pairs get their own buckets (0-12): AA=12, KK=11, ..., 22=0
    # Non-pairs: combine high rank, connectivity, and suitedness

    bucket = jax.lax.cond(
        is_pair,
        lambda: high_rank,  # Pairs: 0-12
        lambda: (
            13 +  # Offset past pairs
            high_rank * 3 +  # High card importance
            jnp.where(gap <= 2, 2, 0) +  # Connector bonus
            jnp.where(is_suited, 1, 0)  # Suited bonus
        )
    )

    # Modulo to fit in bucket range
    return bucket % num_buckets


@jax.jit
def compute_pot_bucket_preflop(
    pot: float,
    stacks: jnp.ndarray,
    num_pot_buckets: int = 5
) -> int:
    """
    Bucket pot size for preflop (coarser than full game).

    Categories (example for 1000 chip stacks):
    - 0: Limped pot (~150-300 chips)
    - 1: Small raise (~300-600 chips)
    - 2: Medium raise (~600-1200 chips)
    - 3: Large raise (~1200-2000 chips)
    - 4: All-in situation (>2000 chips)

    Args:
        pot: Current pot size
        stacks: Current stack sizes
        num_pot_buckets: Number of pot buckets (default 5)

    Returns:
        Bucket index in range [0, num_pot_buckets)
    """
    total_chips = pot + jnp.sum(stacks)
    pot_ratio = pot / (total_chips + 1e-6)

    # Logarithmic bucketing
    log_ratio = jnp.log2(pot_ratio + 1e-6)

    # Map to buckets: -10 to 0 -> 0 to num_pot_buckets
    bucket = jnp.floor((log_ratio + 10) * num_pot_buckets / 10.0)
    bucket = jnp.clip(bucket, 0, num_pot_buckets - 1)

    return bucket.astype(jnp.int32)


@jax.jit
def preflop_state_to_bucket_index(
    state: PreflopState,
    updating_player: int = 0,
    num_buckets: int = 500,
    num_hand_buckets: int = 50,
    num_pot_buckets: int = 5
) -> int:
    """
    Convert preflop state to numeric bucket index.

    Much simpler than full game bucketing:
    - Hand strength only (no board cards)
    - Pot size (coarse)
    - Position (who's acting)
    - Number of actions (betting pressure)

    Total buckets: hand (50) × pot (5) × position (3) × actions (2) = 1,500
    With modulo: 500 buckets

    Args:
        state: PreflopState to bucket
        updating_player: Which player's perspective
        num_buckets: Total number of buckets (default 500)
        num_hand_buckets: Hand strength buckets (default 50)
        num_pot_buckets: Pot size buckets (default 5)

    Returns:
        Bucket index in range [0, num_buckets)
    """
    # Get updating player's hole cards
    hole_cards = state.hole_cards[updating_player]

    # Compute hand bucket
    hand_bucket = compute_preflop_hand_bucket(hole_cards, num_hand_buckets)

    # Compute pot bucket
    pot_bucket = compute_pot_bucket_preflop(state.pot, state.stacks, num_pot_buckets)

    # Position category (early/middle/late)
    # This is simplified - in reality you'd consider button, blinds, etc.
    position_category = jnp.clip(updating_player, 0, 2)  # 0=early, 1=mid, 2=late

    # Action pressure (has there been significant betting?)
    max_bet = jnp.max(state.bets)
    avg_stack = jnp.mean(state.stacks + state.bets)
    betting_pressure = jnp.where(max_bet > avg_stack * 0.1, 1, 0)  # Binary: low/high

    # Combine into single index
    bucket_index = (
        hand_bucket +
        pot_bucket * num_hand_buckets +
        position_category * (num_hand_buckets * num_pot_buckets) +
        betting_pressure * (num_hand_buckets * num_pot_buckets * 3)
    )

    # Modulo to fit within num_buckets
    return bucket_index % num_buckets


# Export main API
__all__ = [
    'card_to_rank',
    'card_to_suit',
    'compute_preflop_hand_bucket',
    'compute_pot_bucket_preflop',
    'preflop_state_to_bucket_index',
]
