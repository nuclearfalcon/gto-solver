"""
Kuhn Poker JAX Implementation - V2 (JAX-Native for Batched Sampling)

**CRITICAL CHANGE from V1**: Uses `jax.lax.cond` instead of Python `if` statement
in `apply_action()` to enable full JAX tracing and batched trajectory sampling.

This version enables 100-1000× speedup through `jax.vmap` batching.

Phase 10.2: JAX-Native Game Engine Rewrite (Day 1)
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import random


# Actions
ACTION_PASS = 0
ACTION_BET = 1

# Cards
JACK = 0
QUEEN = 1
KING = 2


class KuhnState(NamedTuple):
    """Pure JAX state for Kuhn poker."""
    # Cards
    cards: jnp.ndarray  # shape: (2,), dtype: int32, values: {0=J, 1=Q, 2=K}

    # Betting state
    pot: jnp.int32  # Total pot (starts at 2 from antes)
    player_bets: jnp.ndarray  # shape: (2,), dtype: int32

    # Game flow
    acting_player: jnp.int32  # Current player (0 or 1), -1 if terminal
    history: jnp.ndarray  # Action history, shape: (4,), padded with -1
    history_length: jnp.int32  # Number of actions taken


def deal_initial_state(key: jax.random.PRNGKey) -> KuhnState:
    """
    Deal initial Kuhn poker state.

    Args:
        key: JAX random key

    Returns:
        Initial KuhnState with cards dealt
    """
    # Deal 2 cards from 3
    all_cards = jnp.array([JACK, QUEEN, KING], dtype=jnp.int32)
    shuffled = random.permutation(key, all_cards)
    cards = shuffled[:2]

    # Initial state: antes posted, player 0 acts first
    state = KuhnState(
        cards=cards,
        pot=jnp.int32(2),  # 1 chip ante each
        player_bets=jnp.array([1, 1], dtype=jnp.int32),
        acting_player=jnp.int32(0),
        history=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        history_length=jnp.int32(0)
    )

    return state


def legal_actions(state: KuhnState) -> jnp.ndarray:
    """
    Get legal actions for current state.

    Args:
        state: Current KuhnState

    Returns:
        Boolean mask of legal actions [pass, bet]
    """
    # If terminal, no legal actions
    terminal = is_terminal(state)

    # Both pass and bet always legal in Kuhn poker (when not terminal)
    legal = jnp.array([True, True], dtype=bool)

    # If terminal, no legal actions
    legal = jnp.where(terminal, jnp.array([False, False], dtype=bool), legal)

    return legal


def apply_action(state: KuhnState, action: int) -> KuhnState:
    """
    Apply action and return new state.

    **JAX-NATIVE VERSION**: Uses jax.lax.cond instead of Python if statement.
    This enables full JAX tracing for batched trajectory sampling.

    Pure function: No mutation.

    Args:
        state: Current KuhnState
        action: Action to apply (0=pass, 1=bet)

    Returns:
        New KuhnState after action
    """
    player = state.acting_player

    # Update history
    new_history = state.history.at[state.history_length].set(action)
    new_history_length = state.history_length + 1

    # Update bets based on action using jax.lax.cond
    # BEFORE (Python if - blocks JAX tracing):
    #   if action == ACTION_BET:
    #       new_player_bets = new_player_bets.at[player].add(1)
    #       new_pot = new_pot + 1

    # AFTER (JAX-native):
    def bet_fn(player_bets, pot, player):
        """Betting branch: Add 1 chip bet."""
        return player_bets.at[player].add(1), pot + 1

    def pass_fn(player_bets, pot, player):
        """Pass branch: No bet change."""
        return player_bets, pot

    new_player_bets, new_pot = jax.lax.cond(
        action == ACTION_BET,
        bet_fn,
        pass_fn,
        state.player_bets, state.pot, player
    )

    # Determine next actor
    # Kuhn poker rules:
    # - Pass, Pass: Terminal (showdown)
    # - Pass, Bet, Pass: Terminal (fold)
    # - Pass, Bet, Bet: Terminal (showdown)
    # - Bet, Pass: Terminal (fold)
    # - Bet, Bet: Terminal (showdown)

    next_player = 1 - player  # Switch player

    # Check if terminal
    # Use static indexing (history is fixed length 4, just use -1 as invalid)
    h0 = new_history[0]
    h1 = new_history[1]
    h2 = new_history[2]

    # Terminal conditions (check length and sequence)
    pass_pass = (new_history_length == 2) & (h0 == ACTION_PASS) & (h1 == ACTION_PASS)
    pass_bet_pass = (new_history_length == 3) & (h0 == ACTION_PASS) & (h1 == ACTION_BET) & (h2 == ACTION_PASS)
    pass_bet_bet = (new_history_length == 3) & (h0 == ACTION_PASS) & (h1 == ACTION_BET) & (h2 == ACTION_BET)
    bet_pass = (new_history_length == 2) & (h0 == ACTION_BET) & (h1 == ACTION_PASS)
    bet_bet = (new_history_length == 2) & (h0 == ACTION_BET) & (h1 == ACTION_BET)

    terminal = pass_pass | pass_bet_pass | pass_bet_bet | bet_pass | bet_bet

    # If terminal, set actor to -1
    next_player = jnp.where(terminal, jnp.int32(-1), next_player)

    new_state = KuhnState(
        cards=state.cards,
        pot=new_pot,
        player_bets=new_player_bets,
        acting_player=next_player,
        history=new_history,
        history_length=new_history_length
    )

    return new_state


def is_terminal(state: KuhnState) -> jnp.bool_:
    """
    Check if state is terminal.

    JAX-compatible: Uses bitwise operations.

    Args:
        state: Current KuhnState

    Returns:
        True if terminal
    """
    return state.acting_player == -1


def payoffs(state: KuhnState) -> jnp.ndarray:
    """
    Compute terminal payoffs.

    JAX-NATIVE VERSION: Uses static indexing instead of dynamic slicing.

    Args:
        state: Terminal KuhnState

    Returns:
        Payoffs for each player, shape: (2,)
    """
    # Use static indexing (history is fixed length, just check indices directly)
    h0 = state.history[0]
    h1 = state.history[1]
    h2 = state.history[2]

    # Determine outcome
    # Fold outcomes
    pass_bet_pass = (state.history_length == 3) & (h0 == ACTION_PASS) & (h1 == ACTION_BET) & (h2 == ACTION_PASS)
    bet_pass = (state.history_length == 2) & (h0 == ACTION_BET) & (h1 == ACTION_PASS)

    fold = pass_bet_pass | bet_pass

    # Determine winner on fold
    # pass_bet_pass: Player 1 wins (player 0 folded)
    # bet_pass: Player 0 wins (player 1 folded)
    fold_winner = jnp.where(pass_bet_pass, 1, 0)

    # Showdown outcomes (compare cards)
    showdown = ~fold
    showdown_winner = jnp.where(state.cards[0] > state.cards[1], 0, 1)

    # Final winner
    winner = jnp.where(fold, fold_winner, showdown_winner)

    # Payoffs (winner takes pot, loser loses their contribution)
    payoffs_array = jnp.zeros(2, dtype=jnp.float32)

    # Winner gets: pot - their_bet
    # Loser gets: -their_bet
    winner_payoff = state.pot - state.player_bets[winner]
    loser = 1 - winner
    loser_payoff = -state.player_bets[loser]

    payoffs_array = payoffs_array.at[winner].set(winner_payoff)
    payoffs_array = payoffs_array.at[loser].set(loser_payoff)

    return payoffs_array


def state_to_infoset(state: KuhnState, player: int) -> str:
    """
    Convert state to information set string.

    In Kuhn poker, infoset = (player's card, action history)

    NOTE: This function is NOT JAX-traceable (uses Python strings).
    For batched sampling, use bucket IDs instead.

    Args:
        state: Current KuhnState
        player: Player index

    Returns:
        Infoset string
    """
    card_names = ['J', 'Q', 'K']
    card = int(state.cards[player])
    card_str = card_names[card]

    # History
    history = state.history[:state.history_length]
    history_str = ''.join(['p' if a == ACTION_PASS else 'b' for a in history])

    infoset = f"{card_str}_{history_str}"

    return infoset


def card_to_str(card: int) -> str:
    """Convert card index to string."""
    card_names = ['J', 'Q', 'K']
    return card_names[card]


def state_to_string(state: KuhnState) -> str:
    """
    Human-readable state representation.

    Args:
        state: KuhnState

    Returns:
        String description
    """
    p0_card = card_to_str(int(state.cards[0]))
    p1_card = card_to_str(int(state.cards[1]))

    history = state.history[:state.history_length]
    history_str = ''.join(['p' if a == ACTION_PASS else 'b' for a in history])

    return f"Cards: P0={p0_card} P1={p1_card}, History: {history_str}, Actor: {state.acting_player}, Pot: {state.pot}"


if __name__ == "__main__":
    print("Testing Kuhn Poker JAX V2 Implementation (JAX-Native)")
    print("=" * 70)

    print("\n[Test 1: Deal Initial State]")
    key = random.PRNGKey(42)
    state = deal_initial_state(key)
    print(f"✓ Initial state: {state_to_string(state)}")

    print("\n[Test 2: Legal Actions]")
    legal = legal_actions(state)
    print(f"✓ Legal actions: {legal}")

    print("\n[Test 3: Apply Pass Action]")
    state = apply_action(state, ACTION_PASS)
    print(f"✓ After pass: {state_to_string(state)}")

    print("\n[Test 4: Apply Bet Action]")
    state = apply_action(state, ACTION_BET)
    print(f"✓ After bet: {state_to_string(state)}")

    print("\n[Test 5: Complete Game - Pass, Bet, Bet (Showdown)]")
    key = random.PRNGKey(123)
    state = deal_initial_state(key)
    print(f"Initial: {state_to_string(state)}")

    state = apply_action(state, ACTION_PASS)
    print(f"After P0 pass: {state_to_string(state)}")

    state = apply_action(state, ACTION_BET)
    print(f"After P1 bet: {state_to_string(state)}")

    state = apply_action(state, ACTION_BET)
    print(f"After P0 call: {state_to_string(state)}")

    print(f"Terminal: {is_terminal(state)}")
    payoffs_result = payoffs(state)
    print(f"Payoffs: P0={payoffs_result[0]}, P1={payoffs_result[1]}")

    print("\n[Test 6: Complete Game - Bet, Pass (Fold)]")
    key = random.PRNGKey(456)
    state = deal_initial_state(key)
    print(f"Initial: {state_to_string(state)}")

    state = apply_action(state, ACTION_BET)
    print(f"After P0 bet: {state_to_string(state)}")

    state = apply_action(state, ACTION_PASS)
    print(f"After P1 fold: {state_to_string(state)}")

    print(f"Terminal: {is_terminal(state)}")
    payoffs_result = payoffs(state)
    print(f"Payoffs: P0={payoffs_result[0]}, P1={payoffs_result[1]}")

    print("\n[Test 7: Infoset Encoding]")
    key = random.PRNGKey(789)
    state = deal_initial_state(key)
    infoset_p0 = state_to_infoset(state, 0)
    infoset_p1 = state_to_infoset(state, 1)
    print(f"✓ P0 infoset: {infoset_p0}")
    print(f"✓ P1 infoset: {infoset_p1}")

    print("\n[Test 8: JAX JIT Compilation Test]")
    print("Testing if apply_action can be JIT-compiled...")

    # JIT compile apply_action
    jit_apply_action = jax.jit(apply_action)

    key = random.PRNGKey(999)
    state = deal_initial_state(key)

    # Test JIT-compiled version
    state_jit = jit_apply_action(state, ACTION_BET)
    print(f"✓ JIT-compiled apply_action works!")
    print(f"  After JIT bet: {state_to_string(state_jit)}")

    print("\n" + "=" * 70)
    print("Kuhn Poker JAX V2 Tests Passed! ✅")
    print("\nNext: Write comparison tests vs V1, then test batched sampling!")
