"""
Preflop-Only Hold'em Game Engine (JAX)

Simplified Hold'em poker that only plays the preflop round. This is perfect for:
- 3+ player games where full game is intractable
- Push/fold strategies
- ICM (Independent Chip Model) studies
- Tournament situations

Key simplifications:
- No board cards (flop/turn/river)
- Terminal after all preflop betting completes
- Showdown uses precomputed hand equity
- Much faster: ~3 states per trajectory vs ~50 for full game
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import random


class PreflopState(NamedTuple):
    """Simplified state for preflop-only poker."""
    hole_cards: jnp.ndarray  # (num_players, 2) - player hole cards
    deck: jnp.ndarray        # (52,) bool - which cards are available
    bets: jnp.ndarray        # (num_players,) - current bets this round
    pot: float               # Total pot
    stacks: jnp.ndarray      # (num_players,) - remaining stack sizes
    acting_player: int       # Current player to act
    num_actions_this_round: int  # Number of actions taken
    folded: jnp.ndarray      # (num_players,) bool - who has folded
    all_in: jnp.ndarray      # (num_players,) bool - who is all-in
    num_players: int         # Total number of players


def deal_initial_state(
    key: jnp.ndarray,
    num_players: int,
    stacks: jnp.ndarray,
    blinds: jnp.ndarray
) -> PreflopState:
    """
    Deal initial preflop state.

    Args:
        key: JAX random key
        num_players: Number of players (2-10)
        stacks: Initial stack sizes, shape (num_players,)
        blinds: Blind amounts, shape (num_players,) - typically [SB, BB, 0, 0, ...]

    Returns:
        Initial PreflopState with cards dealt and blinds posted
    """
    # Initialize deck (all cards available)
    deck = jnp.ones(52, dtype=bool)

    # Deal hole cards to each player
    hole_cards = jnp.zeros((num_players, 2), dtype=jnp.int32)

    for p in range(num_players):
        for c in range(2):
            # Sample a card from remaining deck
            key, subkey = random.split(key)
            available_cards = jnp.where(deck, jnp.arange(52), -1)
            # Filter to only available cards
            valid_indices = jnp.where(available_cards >= 0, available_cards, 52)
            card_idx = random.choice(subkey, valid_indices)
            card_idx = jnp.where(card_idx == 52, 0, card_idx)  # Fallback (shouldn't happen)

            hole_cards = hole_cards.at[p, c].set(card_idx)
            deck = deck.at[card_idx].set(False)

    # Post blinds
    bets = blinds.copy()
    pot = jnp.sum(blinds)
    stacks = stacks - blinds

    # First to act is after big blind (player 2 in heads-up, player 3 in 3+ player)
    acting_player = jnp.where(num_players == 2, 0, 2)  # HU: SB acts first preflop, else UTG

    return PreflopState(
        hole_cards=hole_cards,
        deck=deck,
        bets=bets,
        pot=pot,
        stacks=stacks,
        acting_player=acting_player,
        num_actions_this_round=0,
        folded=jnp.zeros(num_players, dtype=bool),
        all_in=jnp.zeros(num_players, dtype=bool),
        num_players=num_players
    )


def is_terminal(state: PreflopState) -> bool:
    """Check if state is terminal (hand is over)."""
    num_active = jnp.sum(~state.folded)

    # Terminal if only one player left (others folded)
    if num_active <= 1:
        return True

    # Terminal if betting round complete (all bets matched and everyone acted)
    max_bet = jnp.max(state.bets)
    all_bets_matched = jnp.all(
        (state.bets == max_bet) | state.folded | state.all_in
    )
    everyone_acted = state.num_actions_this_round >= state.num_players

    return bool(all_bets_matched and everyone_acted)


def legal_actions(state: PreflopState) -> jnp.ndarray:
    """
    Get legal actions for current player.

    Returns:
        Boolean array of shape (4,) indicating [FOLD, CALL, BET/RAISE, ALL_IN]
    """
    player = state.acting_player
    current_bet = state.bets[player]
    max_bet = jnp.max(state.bets)
    stack = state.stacks[player]
    to_call = max_bet - current_bet

    # FOLD: always legal (but not if already all-in or last player)
    can_fold = (to_call > 0) and (stack > 0)

    # CALL: legal if there's a bet to call and we have chips
    can_call = (to_call > 0) and (stack >= to_call)

    # BET/RAISE: legal if we have chips beyond the call
    # Simplified: bet/raise to 2x current max bet (or pot-sized)
    pot_after_call = state.pot + to_call
    raise_amount = jnp.maximum(max_bet * 2, pot_after_call)
    total_needed = raise_amount - current_bet
    can_bet = (stack > to_call) and (stack >= total_needed)

    # ALL_IN: always legal if we have chips
    can_allin = stack > 0

    # If there's no bet to us and we can't fold, we must check (use CALL for check)
    if to_call == 0:
        can_fold = False
        can_call = True  # CHECK

    return jnp.array([can_fold, can_call, can_bet, can_allin], dtype=bool)


def apply_action(state: PreflopState, action: int) -> PreflopState:
    """
    Apply action to state.

    Actions:
        0: FOLD
        1: CALL (or CHECK if no bet)
        2: BET/RAISE (2x or pot-sized)
        3: ALL_IN
    """
    player = state.acting_player
    current_bet = state.bets[player]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current_bet
    stack = state.stacks[player]

    # Initialize new state values
    new_bets = state.bets.copy()
    new_stacks = state.stacks.copy()
    new_folded = state.folded.copy()
    new_all_in = state.all_in.copy()
    new_pot = state.pot

    # Apply action
    if action == 0:  # FOLD
        new_folded = new_folded.at[player].set(True)

    elif action == 1:  # CALL / CHECK
        call_amount = jnp.minimum(to_call, stack)
        new_bets = new_bets.at[player].add(call_amount)
        new_stacks = new_stacks.at[player].add(-call_amount)
        new_pot = new_pot + call_amount
        if call_amount == stack:
            new_all_in = new_all_in.at[player].set(True)

    elif action == 2:  # BET/RAISE
        pot_after_call = state.pot + to_call
        raise_size = jnp.maximum(max_bet * 2, pot_after_call)
        total_to_put = raise_size - current_bet
        actual_raise = jnp.minimum(total_to_put, stack)
        new_bets = new_bets.at[player].add(actual_raise)
        new_stacks = new_stacks.at[player].add(-actual_raise)
        new_pot = new_pot + actual_raise
        if actual_raise == stack:
            new_all_in = new_all_in.at[player].set(True)

    elif action == 3:  # ALL_IN
        all_in_amount = stack
        new_bets = new_bets.at[player].add(all_in_amount)
        new_stacks = new_stacks.at[player].set(0)
        new_pot = new_pot + all_in_amount
        new_all_in = new_all_in.at[player].set(True)

    # Find next player to act
    next_player = (player + 1) % state.num_players
    # Skip folded and all-in players
    while (new_folded[next_player] or new_all_in[next_player]) and not is_terminal(
        PreflopState(
            state.hole_cards, state.deck, new_bets, new_pot, new_stacks,
            next_player, state.num_actions_this_round + 1,
            new_folded, new_all_in, state.num_players
        )
    ):
        next_player = (next_player + 1) % state.num_players

    return PreflopState(
        hole_cards=state.hole_cards,
        deck=state.deck,
        bets=new_bets,
        pot=new_pot,
        stacks=new_stacks,
        acting_player=next_player,
        num_actions_this_round=state.num_actions_this_round + 1,
        folded=new_folded,
        all_in=new_all_in,
        num_players=state.num_players
    )


def hand_strength(hole_cards: jnp.ndarray) -> float:
    """
    Compute approximate preflop hand strength (0-1).

    Simplified model based on:
    - Pair rank
    - High card strength
    - Connectivity
    - Suitedness

    Args:
        hole_cards: (2,) array of card indices

    Returns:
        Hand strength in range [0, 1]
    """
    card1, card2 = hole_cards[0], hole_cards[1]
    rank1 = card1 % 13  # 0=2, 1=3, ..., 12=A
    rank2 = card2 % 13
    suit1 = card1 // 13
    suit2 = card2 // 13

    # Normalize ranks (A=12, K=11, ..., 2=0)
    high_rank = jnp.maximum(rank1, rank2)
    low_rank = jnp.minimum(rank1, rank2)

    # Is it a pair?
    is_pair = (rank1 == rank2)

    # Is it suited?
    is_suited = (suit1 == suit2)

    # Gap between cards
    gap = high_rank - low_rank

    # Base strength from high card (Aces strongest)
    base_strength = (high_rank + 1) / 13.0  # 0.076 to 1.0

    # Pair bonus
    pair_bonus = jax.lax.cond(
        is_pair,
        lambda: 0.3 + (high_rank / 13.0) * 0.3,  # AA=0.6, 22=0.3
        lambda: 0.0
    )

    # Connectivity bonus (lower gap is better)
    connectivity_bonus = jnp.maximum(0, (5 - gap) / 20.0)  # Up to 0.25 for connectors

    # Suited bonus
    suited_bonus = jnp.where(is_suited, 0.05, 0.0)

    # High card bonus for second card
    low_card_bonus = (low_rank / 13.0) * 0.1

    # Total strength
    strength = base_strength + pair_bonus + connectivity_bonus + suited_bonus + low_card_bonus
    strength = jnp.clip(strength, 0.0, 1.0)

    return strength


def payoffs(state: PreflopState) -> jnp.ndarray:
    """
    Compute final payoffs for all players.

    For preflop-only, we use approximate equity based on hand strength
    rather than actual showdown evaluation.

    Returns:
        Payoffs array of shape (num_players,)
    """
    num_active = jnp.sum(~state.folded)

    # If only one player remains, they win the pot
    if num_active == 1:
        winner = jnp.argmax(~state.folded)
        payoff = jnp.zeros(state.num_players, dtype=jnp.float32)
        payoff = payoff.at[winner].set(state.pot)
        # Subtract bets from all players
        payoff = payoff - state.bets
        return payoff

    # Showdown: use hand strength as proxy for equity
    # In a real implementation, you'd use precomputed equity tables
    strengths = jnp.array([
        hand_strength(state.hole_cards[p]) if not state.folded[p] else 0.0
        for p in range(state.num_players)
    ])

    # Simple equity model: strength proportional to hand strength
    # (This is an approximation - real equity is more complex)
    total_strength = jnp.sum(strengths)
    equities = jnp.where(
        state.folded,
        0.0,
        strengths / (total_strength + 1e-10)
    )

    # Distribute pot according to equity
    payoff = equities * state.pot - state.bets

    return payoff


# Export main API
__all__ = [
    'PreflopState',
    'deal_initial_state',
    'is_terminal',
    'legal_actions',
    'apply_action',
    'payoffs',
    'hand_strength'
]
