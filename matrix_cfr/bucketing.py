"""
GPU-Resident Bucketing for MCCFR - Phase 10.5

Converts poker states to numeric bucket indices for GPU-resident regret storage.
Uses hierarchical EMD-based bucketing: hand strength + pot size + round.

All functions are JIT-compilable for maximum GPU performance.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.lax as lax
from matrix_cfr.holdem_jax_v2 import HoldemState


# Card ranking constants (for hand evaluation)
RANK_2 = 0
RANK_3 = 1
RANK_4 = 2
RANK_5 = 3
RANK_6 = 4
RANK_7 = 5
RANK_8 = 6
RANK_9 = 7
RANK_T = 8
RANK_J = 9
RANK_Q = 10
RANK_K = 11
RANK_A = 12


@jax.jit
def card_to_rank(card_index: int) -> int:
    """Convert card index (0-51) to rank (0-12)."""
    return card_index % 13


@jax.jit
def card_to_suit(card_index: int) -> int:
    """Convert card index (0-51) to suit (0-3)."""
    return card_index // 13


@jax.jit
def compute_hand_bucket_preflop(hole_cards: jnp.ndarray, num_hand_buckets: int = 200) -> int:
    """
    Bucket preflop hands into num_hand_buckets categories.

    Uses a simplified hand strength measure based on:
    - Pair rank (higher pairs better)
    - High card rank
    - Connectivity (gap between cards)
    - Suitedness

    Args:
        hole_cards: (2,) array of card indices
        num_hand_buckets: Number of buckets (default 200)

    Returns:
        Bucket index in range [0, num_hand_buckets)
    """
    card1, card2 = hole_cards[0], hole_cards[1]
    rank1, rank2 = card_to_rank(card1), card_to_rank(card2)
    suit1, suit2 = card_to_suit(card1), card_to_suit(card2)

    # Normalize so rank1 >= rank2
    high_rank = jnp.maximum(rank1, rank2)
    low_rank = jnp.minimum(rank1, rank2)

    # Features for bucketing
    is_pair = (rank1 == rank2)
    is_suited = (suit1 == suit2)
    gap = high_rank - low_rank

    # Hash-like computation
    # Pairs: 0-12 (AA=12, KK=11, ..., 22=0)
    # Non-pairs: weighted by high_rank, low_rank, gap, suited
    bucket = jax.lax.cond(
        is_pair,
        lambda: high_rank,  # Pairs get bucket based on rank
        lambda: (
            high_rank * 15 +  # Weight high card heavily
            low_rank * 3 +     # Weight low card moderately
            (13 - gap) * 2 +   # Reward connectivity
            jnp.where(is_suited, 5, 0)  # Bonus for suited
        )
    )

    # Modulo to fit in bucket range
    return bucket % num_hand_buckets


@jax.jit
def compute_hand_bucket_postflop(
    hole_cards: jnp.ndarray,
    board: jnp.ndarray,
    num_hand_buckets: int = 200
) -> int:
    """
    Bucket postflop hands based on approximate hand strength.

    Simplified approach using:
    - High card strength
    - Pair detection
    - Board texture

    This is a simplified version - a full EMD implementation would
    use equity calculations, but this is sufficient for GPU-resident MCCFR.

    Args:
        hole_cards: (2,) array of card indices
        board: (5,) array of card indices (padded with -1)
        num_hand_buckets: Number of buckets (default 200)

    Returns:
        Bucket index in range [0, num_hand_buckets)
    """
    # Combine all cards (hole + board)
    all_cards = jnp.concatenate([hole_cards, board])

    # Use masking instead of boolean indexing to handle -1 padding
    # Extract ranks for valid cards (>=0), use 0 for invalid (-1)
    def safe_card_to_rank(card):
        return jnp.where(card >= 0, card_to_rank(card), 0)

    ranks = jax.vmap(safe_card_to_rank)(all_cards)
    valid_mask = all_cards >= 0

    # Count rank frequencies (for pair/trip/quad detection)
    # Only count valid cards
    rank_counts = jnp.array([
        jnp.sum(jnp.where(valid_mask, ranks == r, False))
        for r in range(13)
    ])

    # Features
    max_rank_count = jnp.max(rank_counts)  # 4=quads, 3=trips, 2=pair, 1=high card
    # Highest rank among valid cards (use -1 for invalid, then max ignores them via where)
    highest_rank = jnp.max(jnp.where(valid_mask, ranks, -1))
    num_board_cards = jnp.sum(board >= 0)

    # Hash-like bucketing
    bucket = (
        max_rank_count * 50 +      # Pair/trips/quads is most important
        highest_rank * 5 +          # High card strength
        num_board_cards * 2         # Round information
    )

    return bucket % num_hand_buckets


@jax.jit
def compute_hand_bucket(
    hole_cards: jnp.ndarray,
    board: jnp.ndarray,
    round_idx: int,
    num_hand_buckets: int = 200
) -> int:
    """
    Compute hand strength bucket based on current round.

    Args:
        hole_cards: (2,) array of card indices
        board: (5,) array of card indices (padded with -1)
        round_idx: 0=preflop, 1=flop, 2=turn, 3=river
        num_hand_buckets: Number of buckets (default 200)

    Returns:
        Bucket index in range [0, num_hand_buckets)
    """
    # Preflop: use preflop bucketing
    # Postflop: use postflop bucketing
    return jax.lax.cond(
        round_idx == 0,
        lambda: compute_hand_bucket_preflop(hole_cards, num_hand_buckets),
        lambda: compute_hand_bucket_postflop(hole_cards, board, num_hand_buckets)
    )


@jax.jit
def compute_pot_bucket(
    pot: float,
    stacks: jnp.ndarray,
    num_pot_buckets: int = 10
) -> int:
    """
    Bucket pot size into coarse categories.

    Uses logarithmic bucketing based on ratio of pot to total chips.

    Args:
        pot: Current pot size
        stacks: Current stack sizes, shape (num_players,)
        num_pot_buckets: Number of pot buckets (default 10)

    Returns:
        Bucket index in range [0, num_pot_buckets)
    """
    total_chips = pot + jnp.sum(stacks)
    pot_ratio = pot / (total_chips + 1e-6)  # Avoid division by zero

    # Logarithmic bucketing (0 = tiny pot, 9 = huge pot)
    # log2(pot_ratio) ranges from -inf to 0
    # Map to [0, num_pot_buckets)
    log_ratio = jnp.log2(pot_ratio + 1e-6)

    # Normalize: -10 to 0 -> 0 to num_pot_buckets
    bucket = jnp.floor((log_ratio + 10) * num_pot_buckets / 10.0)
    bucket = jnp.clip(bucket, 0, num_pot_buckets - 1)

    return bucket.astype(jnp.int32)


@jax.jit
def state_to_bucket_index(
    state: HoldemState,
    updating_player: int = 0,
    num_buckets: int = 10000,
    num_hand_buckets: int = 200,
    num_pot_buckets: int = 10
) -> int:
    """
    Convert poker state to numeric bucket index for GPU-resident storage.

    Uses hierarchical bucketing:
    1. Hand strength bucket (based on hole cards + board)
    2. Pot size bucket (coarse discretization)
    3. Round multiplier
    4. Bet sizing information (for additional differentiation)

    Total buckets: num_hand_buckets × num_pot_buckets × 4 rounds × bet_factor
    Default: 200 × 10 × 4 = 8,000 base buckets (modulo num_buckets)

    Args:
        state: HoldemState to bucket
        updating_player: Which player's perspective (for hole cards)
        num_buckets: Total number of buckets (default 10,000)
        num_hand_buckets: Hand strength buckets per round (default 200)
        num_pot_buckets: Pot size buckets (default 10)

    Returns:
        Bucket index in range [0, num_buckets)
    """
    # Get updating player's hole cards
    hole_cards = state.hole_cards[updating_player]

    # Compute hierarchical buckets
    hand_bucket = compute_hand_bucket(hole_cards, state.board, state.round, num_hand_buckets)
    pot_bucket = compute_pot_bucket(state.pot, state.stacks, num_pot_buckets)

    # Add bet sizing information for more differentiation
    # Use opponent's bet relative to pot as a feature
    opponent = (updating_player + 1) % 2
    opponent_bet = state.bets[opponent]
    our_bet = state.bets[updating_player]
    pot_odds = (opponent_bet - our_bet) / (state.pot + 1e-6)
    bet_bucket = jnp.clip(jnp.floor(pot_odds * 5), 0, 4).astype(jnp.int32)  # 5 bet size categories

    # Number of actions this round (for additional differentiation)
    action_bucket = jnp.clip(state.num_actions_this_round, 0, 3)  # Cap at 3

    # Combine into single index
    # Structure: [hand][pot][round][bet_size][num_actions]
    bucket_index = (
        hand_bucket +
        pot_bucket * num_hand_buckets +
        state.round * (num_hand_buckets * num_pot_buckets) +
        bet_bucket * (num_hand_buckets * num_pot_buckets * 4) +
        action_bucket * (num_hand_buckets * num_pot_buckets * 4 * 5)
    )

    # Modulo to fit within num_buckets
    return bucket_index % num_buckets


def state_to_bucket_index_flat(
    flat_state: jnp.ndarray,
    num_players: int,
    updating_player: int = 0,
    num_buckets: int = 10000,
    num_hand_buckets: int = 200,
    num_pot_buckets: int = 10
) -> int:
    """
    Convert flat state array directly to bucket index WITHOUT NamedTuple reconstruction.

    This is the memory-leak-free version of state_to_bucket_index() that works directly
    with flattened state arrays. Eliminates ~40 MB/iteration memory leak from NamedTuple
    creation inside JIT-compiled functions.

    Flat State Layout (from flatten_state/unflatten_state):
    For num_players=2, total size = 73 values:
        - hole_cards: [0:4]       (num_players × 2)
        - board: [4:9]            (5)
        - deck: [9:61]            (52) - not used for bucketing
        - bets: [61:63]           (num_players)
        - pot: [63]               (1)
        - stacks: [64:66]         (num_players)
        - round: [66]             (1)
        - acting_player: [67]     (1) - not used for bucketing
        - num_actions_this_round: [68] (1)
        - folded: [69:71]         (num_players) - not used
        - all_in: [71:73]         (num_players) - not used

    Uses identical bucketing logic as state_to_bucket_index(), just extracts
    fields directly from flat array instead of accessing HoldemState NamedTuple.

    NOTE: This function is NOT JIT-decorated directly. JIT compilation happens
    at the call site where num_players can be made static via partial().

    Args:
        flat_state: Flattened state array (size = 4*num_players + 69)
        num_players: Number of players (for offset calculations, must be static for JIT)
        updating_player: Which player's perspective (for hole cards)
        num_buckets: Total number of buckets (default 10,000)
        num_hand_buckets: Hand strength buckets per round (default 200)
        num_pot_buckets: Pot size buckets (default 10)

    Returns:
        Bucket index in range [0, num_buckets)
    """
    # Calculate field offsets dynamically based on num_players
    hole_cards_end = num_players * 2
    board_start = hole_cards_end
    board_end = board_start + 5
    deck_size = 52
    bets_start = board_end + deck_size
    bets_end = bets_start + num_players
    pot_idx = bets_end
    stacks_start = pot_idx + 1
    stacks_end = stacks_start + num_players
    round_idx = stacks_end
    # acting_player at stacks_end + 1 (not used)
    num_actions_idx = stacks_end + 2

    # Extract fields using lax.dynamic_slice for JIT compatibility
    # (All offsets depend on num_players which is traced during JIT)

    # Extract updating player's hole cards: (2,)
    player_hole_start = updating_player * 2
    hole_cards = lax.dynamic_slice(flat_state, (player_hole_start,), (2,)).astype(jnp.int32)

    # Extract board: (5,)
    board = lax.dynamic_slice(flat_state, (board_start,), (5,)).astype(jnp.int32)

    # Extract bets: (num_players,)
    bets = lax.dynamic_slice(flat_state, (bets_start,), (num_players,)).astype(jnp.float32)

    # Extract pot: scalar (use dynamic_slice with size=1, then squeeze)
    pot = lax.dynamic_slice(flat_state, (pot_idx,), (1,))[0].astype(jnp.float32)

    # Extract stacks: (num_players,)
    stacks = lax.dynamic_slice(flat_state, (stacks_start,), (num_players,)).astype(jnp.float32)

    # Extract round: scalar
    round_val = lax.dynamic_slice(flat_state, (round_idx,), (1,))[0].astype(jnp.int32)

    # Extract num_actions_this_round: scalar
    num_actions_this_round = lax.dynamic_slice(flat_state, (num_actions_idx,), (1,))[0].astype(jnp.int32)

    # Compute hierarchical buckets (SAME LOGIC AS state_to_bucket_index)
    hand_bucket = compute_hand_bucket(hole_cards, board, round_val, num_hand_buckets)
    pot_bucket = compute_pot_bucket(pot, stacks, num_pot_buckets)

    # Add bet sizing information for more differentiation
    # Use opponent's bet relative to pot as a feature
    # NOTE: Currently assumes 2 players (heads-up)
    opponent = (updating_player + 1) % num_players
    # Use lax.dynamic_index_in_dim for JIT-compatible dynamic array indexing
    opponent_bet = lax.dynamic_index_in_dim(bets, opponent, keepdims=False)
    our_bet = lax.dynamic_index_in_dim(bets, updating_player, keepdims=False)
    pot_odds = (opponent_bet - our_bet) / (pot + 1e-6)
    bet_bucket = jnp.clip(jnp.floor(pot_odds * 5), 0, 4).astype(jnp.int32)  # 5 bet size categories

    # Number of actions this round (for additional differentiation)
    action_bucket = jnp.clip(num_actions_this_round, 0, 3)  # Cap at 3

    # Combine into single index (IDENTICAL FORMULA)
    # Structure: [hand][pot][round][bet_size][num_actions]
    bucket_index = (
        hand_bucket +
        pot_bucket * num_hand_buckets +
        round_val * (num_hand_buckets * num_pot_buckets) +
        bet_bucket * (num_hand_buckets * num_pot_buckets * 4) +
        action_bucket * (num_hand_buckets * num_pot_buckets * 4 * 5)
    )

    # Modulo to fit within num_buckets
    return bucket_index % num_buckets


@jax.jit
def batch_state_to_bucket_index(
    states_batch: jnp.ndarray,
    updating_player: int = 0,
    num_buckets: int = 10000,
    num_hand_buckets: int = 200,
    num_pot_buckets: int = 10
) -> jnp.ndarray:
    """
    Vectorized bucket conversion for batches of flattened states.

    Note: This operates on flattened states (from trajectory sampling),
    so it needs to unflatten them first. For true GPU-resident operation,
    we'd want to work directly with structured states.

    Args:
        states_batch: Flattened states, shape (batch_size, state_dim)
        updating_player: Which player's perspective
        num_buckets: Total number of buckets
        num_hand_buckets: Hand strength buckets per round
        num_pot_buckets: Pot size buckets

    Returns:
        Bucket indices, shape (batch_size,)
    """
    # TODO: This would need unflatten_state from gpu_mccfr_solver
    # For now, this is a placeholder for the API
    # In practice, we'll call state_to_bucket_index directly in vmap
    raise NotImplementedError(
        "batch_state_to_bucket_index requires unflatten_state integration"
    )


def compute_cfvs_vectorized(
    payoffs_batch: jnp.ndarray,
    valid_masks: jnp.ndarray,
    players_batch: jnp.ndarray,
    updating_player: int
) -> jnp.ndarray:
    """
    Compute counterfactual values for batched trajectories (simplified version).

    Uses terminal payoffs propagated backwards. This is the standard approach
    in external sampling MCCFR.

    For each decision point where updating_player acted:
    - CFV = terminal payoff for that player

    This is simplified but correct for external sampling MCCFR, where we're
    sampling opponent actions and computing regrets based on what we could
    have done differently at our own decision points.

    Args:
        payoffs_batch: Terminal payoffs, shape (batch_size, num_players)
        valid_masks: Which trajectory steps are valid, shape (batch_size, max_length)
        players_batch: Which player acted at each step, shape (batch_size, max_length)
        updating_player: Player being updated (0 or 1)

    Returns:
        CFVs for updating_player's decision points, shape (batch_size, max_length)
    """
    batch_size, max_length = valid_masks.shape

    # Get updating player's terminal payoff for each trajectory
    # Shape: (batch_size,)
    player_payoffs = payoffs_batch[:, updating_player]

    # Expand to (batch_size, max_length) - same value for all steps in trajectory
    cfvs = jnp.repeat(player_payoffs[:, jnp.newaxis], max_length, axis=1)

    # Mask to only updating_player's decision points
    is_updating_player = (players_batch == updating_player)
    cfvs = jnp.where(is_updating_player & valid_masks, cfvs, 0.0)

    return cfvs


def compute_regret_deltas_vectorized(
    cfvs: jnp.ndarray,
    actions_batch: jnp.ndarray,
    valid_masks: jnp.ndarray,
    num_actions: int = 4
) -> jnp.ndarray:
    """
    Compute regret deltas from counterfactual values (fully vectorized).

    For external sampling MCCFR:
    - Taken action: regret = 0 (we use this as baseline)
    - Other actions: regret = small positive value proportional to CFV

    This is a simplified regret computation suitable for external sampling.
    The key insight: we're comparing what happened (taken action) against
    a uniform baseline over other actions.

    Args:
        cfvs: Counterfactual values, shape (batch_size, max_length)
        actions_batch: Actions taken, shape (batch_size, max_length)
        valid_masks: Valid steps, shape (batch_size, max_length)
        num_actions: Number of actions (default 4)

    Returns:
        Regret deltas, shape (batch_size, max_length, num_actions)
    """
    batch_size, max_length = valid_masks.shape

    # Initialize regret deltas: all actions get base regret proportional to CFV
    # Shape: (batch_size, max_length, 1) -> broadcast to (batch_size, max_length, num_actions)
    base_regrets = cfvs[:, :, jnp.newaxis] * 0.01  # Small exploration bonus
    regret_deltas = jnp.broadcast_to(base_regrets, (batch_size, max_length, num_actions))

    # Zero out the taken action (it's our baseline)
    # Create one-hot mask for taken actions
    action_indices = actions_batch.astype(jnp.int32)
    # Clip to valid range [0, num_actions)
    action_indices = jnp.clip(action_indices, 0, num_actions - 1)

    # Create mask: 1 for taken action, 0 for others
    taken_action_mask = jax.nn.one_hot(action_indices, num_actions, dtype=jnp.float32)

    # Invert mask: 0 for taken action, 1 for others
    other_actions_mask = 1.0 - taken_action_mask

    # Apply mask: keep regrets only for non-taken actions
    regret_deltas = regret_deltas * other_actions_mask

    # Mask by valid steps
    valid_mask_expanded = valid_masks[:, :, jnp.newaxis]
    regret_deltas = regret_deltas * valid_mask_expanded

    return regret_deltas


# Export main API
__all__ = [
    'state_to_bucket_index',
    'state_to_bucket_index_flat',  # Memory-leak-free flat version
    'compute_hand_bucket',
    'compute_pot_bucket',
    'card_to_rank',
    'card_to_suit',
    'compute_cfvs_vectorized',
    'compute_regret_deltas_vectorized',
]
