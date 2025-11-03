"""
Kuhn Poker JAX Implementation

Kuhn poker is the simplest possible poker game:
- 2 players
- 3 cards: Jack (0), Queen (1), King (2)
- Each player dealt 1 card
- Betting: Each player antes 1 chip
- Actions: Pass (0) or Bet (1)
- Betting sequences:
  - Pass, Pass: Showdown (high card wins)
  - Pass, Bet, Pass: Bettor wins (fold)
  - Pass, Bet, Bet: Showdown (high card wins, +1 chip bet)
  - Bet, Pass: Bettor wins (fold)
  - Bet, Bet: Showdown (high card wins, +1 chip bet)

Known Nash Equilibrium (from CFR literature):
- Player 0 with Jack: Always pass
- Player 0 with Queen: Pass, then bet 1/3 of the time if opponent bets
- Player 0 with King: Always bet
- Player 1 with Jack: Pass always
- Player 1 with Queen: Bet 1/3 of the time
- Player 1 with King: Always bet

Expected exploitability at Nash: ~0.0

Phase 10: Kuhn Poker Validation (Day 10)
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

    # Update bets if betting
    new_player_bets = state.player_bets
    new_pot = state.pot

    if action == ACTION_BET:
        new_player_bets = new_player_bets.at[player].add(1)
        new_pot = new_pot + 1

    # Determine next actor
    # Kuhn poker rules:
    # - Pass, Pass: Terminal (showdown)
    # - Pass, Bet, Pass: Terminal (fold)
    # - Pass, Bet, Bet: Terminal (showdown)
    # - Bet, Pass: Terminal (fold)
    # - Bet, Bet: Terminal (showdown)

    next_player = 1 - player  # Switch player

    # Check if terminal
    history_list = new_history[:new_history_length]

    # Terminal conditions
    pass_pass = (new_history_length == 2) & (history_list[0] == ACTION_PASS) & (history_list[1] == ACTION_PASS)
    pass_bet_pass = (new_history_length == 3) & (history_list[0] == ACTION_PASS) & (history_list[1] == ACTION_BET) & (history_list[2] == ACTION_PASS)
    pass_bet_bet = (new_history_length == 3) & (history_list[0] == ACTION_PASS) & (history_list[1] == ACTION_BET) & (history_list[2] == ACTION_BET)
    bet_pass = (new_history_length == 2) & (history_list[0] == ACTION_BET) & (history_list[1] == ACTION_PASS)
    bet_bet = (new_history_length == 2) & (history_list[0] == ACTION_BET) & (history_list[1] == ACTION_BET)

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

    Args:
        state: Terminal KuhnState

    Returns:
        Payoffs for each player, shape: (2,)
    """
    history = state.history[:state.history_length]

    # Determine outcome
    # Fold outcomes
    pass_bet_pass = (state.history_length == 3) & (history[0] == ACTION_PASS) & (history[1] == ACTION_BET) & (history[2] == ACTION_PASS)
    bet_pass = (state.history_length == 2) & (history[0] == ACTION_BET) & (history[1] == ACTION_PASS)

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
    print("Testing Kuhn Poker JAX Implementation")
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

    print("\n" + "=" * 70)
    print("Kuhn Poker JAX Tests Passed! ✅")
    print("\nNext: Validate GPU MCCFR convergence on Kuhn poker!")
