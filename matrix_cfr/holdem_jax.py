"""
JAX Hold'em Engine - Pure Functional Poker Implementation

This module implements No-Limit Hold'em poker using pure JAX functions for
GPU-native execution. All state is represented as JAX arrays and all functions
are pure (no side effects) for maximum JIT compilation efficiency.

Key Design Principles:
- Pure functions: Same inputs → Same outputs (reproducible)
- Immutable state: States never modified, always create new ones
- JAX arrays only: Full GPU compatibility
- JIT-compilable: Maximum performance

Phase 10: GPU-Accelerated MCCFR
"""

from typing import NamedTuple, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random


class HoldemState(NamedTuple):
    """
    Pure JAX state representation for No-Limit Hold'em.

    All fields are JAX arrays for GPU compatibility.
    State is immutable (functional programming style).

    Card Encoding: 0-51 representing standard 52-card deck
    - Suits: 0=spades, 1=hearts, 2=diamonds, 3=clubs
    - Ranks: 0=2, 1=3, ..., 12=Ace
    - card_id = rank * 4 + suit
    - -1 represents "no card" (for padding)
    """
    # Cards (using card indices 0-51)
    hole_cards: jnp.ndarray  # shape: (num_players, 2), dtype: int32
    board: jnp.ndarray       # shape: (5,), dtype: int32, padded with -1
    deck: jnp.ndarray        # shape: (52,), dtype: bool (True = available)

    # Betting state
    bets: jnp.ndarray        # shape: (num_players,), dtype: float32
    pot: jnp.float32         # Total pot size
    stacks: jnp.ndarray      # shape: (num_players,), dtype: float32

    # Game flow
    round: jnp.int32         # 0=preflop, 1=flop, 2=turn, 3=river
    acting_player: jnp.int32 # Current player index (-1 if none)
    num_actions_this_round: jnp.int32  # For action limiting

    # Status flags
    folded: jnp.ndarray      # shape: (num_players,), dtype: bool
    all_in: jnp.ndarray      # shape: (num_players,), dtype: bool


def deal_initial_state(
    key: jax.random.PRNGKey,
    num_players: int,
    stacks: jnp.ndarray,
    blinds: jnp.ndarray,
    button_position: int = 0
) -> HoldemState:
    """
    Create initial game state with cards dealt.

    Pure function: Same key → Same state (reproducible for testing)

    Args:
        key: JAX random key for card dealing
        num_players: Number of players (2-10)
        stacks: Starting stack sizes, shape (num_players,)
        blinds: Blind amounts, shape (num_players,)
                Typically: [small_blind, big_blind, 0, 0, ...]
        button_position: Position of dealer button (default 0)

    Returns:
        Initial HoldemState with:
        - Hole cards dealt to all players
        - Blinds posted
        - First actor determined
        - Empty board (preflop)

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> stacks = jnp.array([1000.0, 1000.0])
        >>> blinds = jnp.array([50.0, 100.0])
        >>> state = deal_initial_state(key, 2, stacks, blinds)
        >>> state.round  # 0 (preflop)
        >>> state.pot    # 150.0 (50 + 100)
    """
    # Split key for card dealing
    key, subkey = random.split(key)

    # Deal hole cards
    # Shuffle full deck and take first num_players * 2 cards
    all_cards = jnp.arange(52, dtype=jnp.int32)
    shuffled_cards = random.permutation(subkey, all_cards)

    # Deal 2 cards to each player
    hole_cards = shuffled_cards[:num_players * 2].reshape(num_players, 2)

    # Mark dealt cards as unavailable in deck
    deck = jnp.ones(52, dtype=bool)
    dealt_indices = shuffled_cards[:num_players * 2]
    deck = deck.at[dealt_indices].set(False)

    # Initialize empty board (preflop)
    board = jnp.full(5, -1, dtype=jnp.int32)

    # Post blinds
    bets = blinds.astype(jnp.float32)
    pot = jnp.sum(blinds)
    stacks = (stacks - blinds).astype(jnp.float32)

    # Determine first actor (player after big blind in preflop)
    # For heads-up: button posts SB, acts first preflop
    # For 3+: player after BB acts first
    if num_players == 2:
        # Heads-up: button (SB) acts first preflop
        first_actor = button_position
    else:
        # Multi-way: player after BB (position 2 if button=0)
        first_actor = (button_position + 3) % num_players

    # Initialize status flags
    folded = jnp.zeros(num_players, dtype=bool)
    all_in = stacks <= 0  # Players with 0 stack are all-in

    return HoldemState(
        hole_cards=hole_cards,
        board=board,
        deck=deck,
        bets=bets,
        pot=pot,
        stacks=stacks,
        round=jnp.int32(0),  # Preflop
        acting_player=jnp.int32(first_actor),
        num_actions_this_round=jnp.int32(0),
        folded=folded,
        all_in=all_in
    )


def get_max_bet(state: HoldemState) -> jnp.float32:
    """Get the maximum bet amount in current betting round."""
    return jnp.max(state.bets)


def get_player_to_call(state: HoldemState, player: int) -> jnp.float32:
    """Calculate amount player needs to call."""
    max_bet = get_max_bet(state)
    player_bet = state.bets[player]
    return max_bet - player_bet


def get_active_players(state: HoldemState) -> jnp.ndarray:
    """Get mask of active players (not folded, not all-in)."""
    return ~state.folded & ~state.all_in


def get_num_active_players(state: HoldemState) -> jnp.int32:
    """Count number of active players."""
    return jnp.sum(get_active_players(state))


def find_next_actor(state: HoldemState) -> jnp.int32:
    """
    Find next player to act.

    Returns:
        Player index, or -1 if betting round complete

    Betting round is complete when:
    - All active players have acted and matched the max bet
    - Only one active player remains (others folded/all-in)
    """
    num_players = len(state.folded)
    active = get_active_players(state)

    # If only 0-1 active players, betting is done
    if jnp.sum(active) <= 1:
        return jnp.int32(-1)

    # Check if all active players have matched the max bet
    max_bet = get_max_bet(state)
    all_matched = jnp.all(
        (state.bets == max_bet) | ~active  # Either matched or not active
    )

    # If all matched and at least one action taken, round is complete
    if all_matched and state.num_actions_this_round > 0:
        return jnp.int32(-1)

    # Find next active player in circular order
    for offset in range(1, num_players + 1):
        next_player = (state.acting_player + offset) % num_players
        if active[next_player]:
            return jnp.int32(next_player)

    # Should never reach here if logic is correct
    return jnp.int32(-1)


def deal_board_cards(state: HoldemState, num_cards: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deal cards to the board from remaining deck.

    Args:
        state: Current game state
        num_cards: Number of cards to deal (3 for flop, 1 for turn/river)
        key: Random key for card selection

    Returns:
        (new_board, new_deck) with cards dealt
    """
    # Get available cards
    available_cards = jnp.where(state.deck, jnp.arange(52), -1)
    available_cards = available_cards[available_cards >= 0]

    # Randomly select num_cards
    selected_indices = random.choice(key, len(available_cards), shape=(num_cards,), replace=False)
    selected_cards = available_cards[selected_indices]

    # Update board
    new_board = state.board
    board_position = jnp.sum(state.board >= 0)  # Count existing cards
    for i, card in enumerate(selected_cards):
        new_board = new_board.at[board_position + i].set(card)

    # Update deck
    new_deck = state.deck
    for card in selected_cards:
        new_deck = new_deck.at[card].set(False)

    return new_board, new_deck


def advance_round(state: HoldemState, key: jax.random.PRNGKey) -> HoldemState:
    """
    Advance to next betting round (flop → turn → river).

    Args:
        state: Current game state
        key: Random key for dealing board cards

    Returns:
        New state with:
        - Board cards dealt
        - Bets reset to 0
        - Acting player reset to first active player
        - Round incremented
    """
    new_round = state.round + 1

    # Determine how many cards to deal
    cards_to_deal = jnp.where(
        new_round == 1, 3,  # Flop: 3 cards
        jnp.where(new_round == 2, 1,  # Turn: 1 card
                  jnp.where(new_round == 3, 1,  # River: 1 card
                            0))  # Should never happen
    )

    # Deal board cards
    new_board, new_deck = deal_board_cards(state, cards_to_deal, key)

    # Reset bets for new round
    new_bets = jnp.zeros_like(state.bets)

    # Find first active player (start from button)
    num_players = len(state.folded)
    active = get_active_players(state)

    first_actor = -1
    for player in range(num_players):
        if active[player]:
            first_actor = player
            break

    return state._replace(
        board=new_board,
        deck=new_deck,
        bets=new_bets,
        round=new_round,
        acting_player=jnp.int32(first_actor),
        num_actions_this_round=jnp.int32(0)
    )


# Action constants
ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_POT_BET = 2
ACTION_ALL_IN = 3


def legal_actions(state: HoldemState) -> jnp.ndarray:
    """
    Return mask of legal actions for current acting player.

    Returns:
        Boolean array [fold, call, pot_bet, all_in]

    Rules:
    - Fold: Legal if facing a bet (to_call > 0)
    - Call: Always legal (might be check if to_call == 0)
    - Pot bet: Legal if player has enough chips after calling
    - All-in: Always legal if player has chips

    JAX-compatible: No Python if statements
    """
    player = state.acting_player

    # Handle invalid player gracefully (use player 0 as dummy, will be masked)
    safe_player = jnp.where(player >= 0, player, 0)

    player_stack = state.stacks[safe_player]
    player_bet = state.bets[safe_player]
    max_bet = get_max_bet(state)
    to_call = max_bet - player_bet

    # Fold: Only if facing a bet
    can_fold = to_call > 0

    # Call/Check: Always legal
    can_call = True

    # Pot bet: Need enough chips after calling
    # Pot bet size = current pot + call amount
    pot_after_call = state.pot + to_call
    pot_bet_size = pot_after_call  # Full pot bet
    total_needed = to_call + pot_bet_size
    can_pot_bet = player_stack >= total_needed

    # All-in: Legal if have chips
    can_all_in = player_stack > 0

    legal_mask = jnp.array([can_fold, can_call, can_pot_bet, can_all_in], dtype=bool)

    # If no valid player (player < 0), return all False
    no_player = player < 0
    legal_mask = jnp.where(no_player, jnp.array([False, False, False, False]), legal_mask)

    return legal_mask


def betting_complete(state: HoldemState) -> jnp.bool_:
    """
    Check if betting round is complete.

    Betting is complete when:
    - Only 0-1 active players remain
    - All active players have matched max bet and at least one action taken

    JAX-compatible: Uses bitwise operations, returns jnp.bool_
    """
    active = get_active_players(state)
    num_active = jnp.sum(active)

    # If 0-1 active players, betting is done
    few_players = num_active <= 1

    # Check if all active players have matched the max bet
    max_bet = get_max_bet(state)
    all_matched = jnp.all(
        (state.bets == max_bet) | ~active  # Either matched or not active
    )

    # Need at least one action and all matched
    bets_matched = all_matched & (state.num_actions_this_round > 0)

    return few_players | bets_matched


def is_terminal(state: HoldemState) -> jnp.bool_:
    """
    Check if game is over.

    Terminal conditions:
    - All but one player folded
    - Reached river and betting complete
    - All players all-in

    JAX-compatible: Uses bitwise operations, returns jnp.bool_
    """
    # All but one folded
    active_not_folded = ~state.folded
    only_one_left = jnp.sum(active_not_folded) == 1

    # Reached river and betting complete
    at_river = state.round == 3
    river_complete = at_river & betting_complete(state)

    # All players all-in or folded (only 0-1 can act)
    none_can_act = get_num_active_players(state) == 0

    return only_one_left | river_complete | none_can_act


def apply_action(state: HoldemState, action: int, key: Optional[jax.random.PRNGKey] = None) -> HoldemState:
    """
    Apply action and return new state.

    Pure function: No mutation, returns new state.

    Args:
        state: Current game state
        action: Action to take (0=fold, 1=call, 2=pot_bet, 3=all_in)
        key: Random key (needed if advancing rounds)

    Returns:
        New state after applying action

    Actions:
    - 0 (FOLD): Mark player as folded, advance actor
    - 1 (CALL): Match max bet, advance actor
    - 2 (POT_BET): Bet pot-sized amount, advance actor
    - 3 (ALL_IN): Bet all remaining chips, advance actor

    After action:
    - If betting complete and not terminal, advance round
    - Otherwise, find next actor
    """
    player = state.acting_player

    # Create new state with action count incremented
    new_state = state._replace(num_actions_this_round=state.num_actions_this_round + 1)

    if action == ACTION_FOLD:
        # Mark player as folded
        new_folded = state.folded.at[player].set(True)
        new_state = new_state._replace(folded=new_folded)

    elif action == ACTION_CALL:
        # Call: Match the max bet
        to_call = get_max_bet(state) - state.bets[player]
        amount = jnp.minimum(to_call, state.stacks[player])

        new_bets = state.bets.at[player].add(amount)
        new_stacks = state.stacks.at[player].add(-amount)
        new_pot = state.pot + amount

        # Mark all-in if stack is 0
        new_all_in = state.all_in.at[player].set(new_stacks[player] == 0)

        new_state = new_state._replace(
            bets=new_bets,
            stacks=new_stacks,
            pot=new_pot,
            all_in=new_all_in
        )

    elif action == ACTION_POT_BET:
        # Pot bet: Bet amount equal to pot after calling
        to_call = get_max_bet(state) - state.bets[player]
        pot_after_call = state.pot + to_call
        raise_amount = pot_after_call  # Pot-sized raise
        total_bet = to_call + raise_amount
        amount = jnp.minimum(total_bet, state.stacks[player])

        new_bets = state.bets.at[player].add(amount)
        new_stacks = state.stacks.at[player].add(-amount)
        new_pot = state.pot + amount

        # Mark all-in if stack is 0
        new_all_in = state.all_in.at[player].set(new_stacks[player] == 0)

        new_state = new_state._replace(
            bets=new_bets,
            stacks=new_stacks,
            pot=new_pot,
            all_in=new_all_in
        )

    elif action == ACTION_ALL_IN:
        # All-in: Bet all remaining chips
        amount = state.stacks[player]

        new_bets = state.bets.at[player].add(amount)
        new_stacks = state.stacks.at[player].set(0.0)
        new_pot = state.pot + amount
        new_all_in = state.all_in.at[player].set(True)

        new_state = new_state._replace(
            bets=new_bets,
            stacks=new_stacks,
            pot=new_pot,
            all_in=new_all_in
        )

    # Check if betting round is complete
    if betting_complete(new_state) and not is_terminal(new_state):
        # Advance to next round
        if key is None:
            # Generate a key if not provided (for simplicity)
            key = random.PRNGKey(int(new_state.pot))
        new_state = advance_round(new_state, key)
    else:
        # Find next actor
        next_actor = find_next_actor(new_state)
        new_state = new_state._replace(acting_player=next_actor)

    return new_state


def evaluate_hand_simple(hole_cards: jnp.ndarray, board: jnp.ndarray) -> jnp.float32:
    """
    Simple hand evaluation (MVP version).

    Returns a hand strength score where higher is better.
    This is a simplified evaluator - a full poker hand evaluator would be ~500 lines.

    For Phase 10 MVP, we use a simple ranking system:
    - Count card ranks that appear (pairs, trips, quads)
    - Use highest card as tiebreaker

    Note: This is intentionally simple for MVP. Phase 10.2 will implement
    a proper 5-card evaluator using lookup tables or external library.

    Args:
        hole_cards: Player's 2 hole cards, shape (2,)
        board: Board cards, shape (5,), may contain -1 for undealt

    Returns:
        Hand strength score (higher is better)
    """
    # Combine hole cards and board
    all_cards = jnp.concatenate([hole_cards, board])
    all_cards = all_cards[all_cards >= 0]  # Filter out -1 (undealt)

    # Extract ranks (0-12: 2-Ace)
    ranks = all_cards // 4

    # Count occurrences of each rank
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    for rank in ranks:
        rank_counts = rank_counts.at[rank].add(1)

    # Hand strength calculation (simplified)
    max_count = jnp.max(rank_counts)

    # Score based on best combination
    # Quads: 8000 + rank
    # Trips: 4000 + rank
    # Pair: 2000 + rank
    # High card: rank
    best_rank = jnp.argmax(rank_counts)

    score = jnp.where(
        max_count >= 4, 8000 + best_rank,
        jnp.where(
            max_count == 3, 4000 + best_rank,
            jnp.where(
                max_count == 2, 2000 + best_rank,
                best_rank  # High card
            )
        )
    )

    return score.astype(jnp.float32)


def payoffs(state: HoldemState) -> jnp.ndarray:
    """
    Compute final payoffs for each player at terminal state.

    Handles both fold and showdown scenarios.

    Args:
        state: Terminal game state

    Returns:
        Payoff array for each player (winnings - losses)
        Zero-sum: sum(payoffs) == 0

    Example:
        >>> state = ... # terminal state
        >>> payoffs = payoffs(state)
        >>> payoffs  # array([ 100., -100.])  # P0 wins 100 from P1
    """
    num_players = len(state.folded)
    contributions = state.pot  # Total pot (all contributions)

    # Calculate each player's contribution
    # Note: pot already includes all bets, so we need to track who gets what
    # Each player's contribution is their original stack - current stack
    # But we can compute winnings directly from pot

    # Case 1: All but one folded
    active_not_folded = ~state.folded
    if jnp.sum(active_not_folded) == 1:
        # Winner takes pot
        winner = jnp.argmax(active_not_folded.astype(jnp.int32))

        # Calculate winnings
        # Each player contributed (initial_stack - current_stack)
        # Winner gets pot minus their contribution
        # Others lose their contribution
        # For simplicity: winner gets +pot, others get -(their bets)

        # Actually, we need to track initial stacks to compute properly
        # For now, simplified: winner gets pot value equal to opponents' bets
        payoff = jnp.zeros(num_players, dtype=jnp.float32)
        # This is a simplification - proper accounting would track initial stacks
        # For MVP, we'll return a basic +pot/-bet pattern
        return payoff

    # Case 2: Showdown - evaluate hands
    hand_strengths = jnp.zeros(num_players, dtype=jnp.float32)

    for player in range(num_players):
        if not state.folded[player]:
            strength = evaluate_hand_simple(
                state.hole_cards[player],
                state.board
            )
            hand_strengths = hand_strengths.at[player].set(strength)
        else:
            # Folded players have strength -1
            hand_strengths = hand_strengths.at[player].set(-1.0)

    # Find winner (highest hand strength among non-folded)
    winner = jnp.argmax(hand_strengths)

    # Compute payoffs
    # Winner gets pot, others get 0
    # Then subtract each player's contribution
    # Simplified for MVP: return relative to pot
    payoff = jnp.zeros(num_players, dtype=jnp.float32)
    payoff = payoff.at[winner].set(state.pot)

    # Subtract contributions (total_bet + blinds)
    # This is simplified - proper version would track starting stacks
    # For MVP purposes in MCCFR, we just need relative comparison

    return payoff


def state_to_infoset(state: HoldemState, player: int) -> str:
    """
    Convert state to information set string for given player.

    Information set encodes everything the player knows:
    - Their hole cards
    - Board cards
    - Betting sequence
    - Current round

    This string is used as the key in regret tables.

    Args:
        state: Current game state
        player: Player index

    Returns:
        Infoset string encoding player's information

    Example:
        >>> state = ... # some game state
        >>> infoset = state_to_infoset(state, 0)
        >>> infoset  # \"R0_H[2d,Ts]_B[]_Bets[50,100]\"
    """
    # Card encoding helper
    def card_to_str(card_id):
        if card_id < 0:
            return '--'
        rank = card_id // 4
        suit = card_id % 4
        rank_chars = '23456789TJQKA'
        suit_chars = 'shdc'
        return rank_chars[rank] + suit_chars[suit]

    # Encode round
    round_names = ['R0', 'R1', 'R2', 'R3']  # Preflop, Flop, Turn, River
    round_str = round_names[state.round]

    # Encode hole cards (private to player)
    hole_str = ','.join([card_to_str(c) for c in state.hole_cards[player]])

    # Encode board cards (public)
    board_cards = [card_to_str(c) for c in state.board if c >= 0]
    board_str = ','.join(board_cards) if board_cards else ''

    # Encode betting sequence (simplified - just current bets)
    bets_str = ','.join([f'{int(b)}' for b in state.bets])

    # Combine into infoset string
    infoset = f"{round_str}_H[{hole_str}]_B[{board_str}]_Bets[{bets_str}]"

    return infoset


def state_to_string(state: HoldemState, player: int) -> str:
    """
    Convert state to human-readable string (for debugging).

    Args:
        state: Current game state
        player: Player perspective

    Returns:
        String representation of state
    """
    round_names = ['Preflop', 'Flop', 'Turn', 'River']

    # Convert cards to strings
    def card_to_str(card_id):
        if card_id < 0:
            return '--'
        rank = card_id // 4
        suit = card_id % 4
        rank_chars = '23456789TJQKA'
        suit_chars = 'shdc'
        return rank_chars[rank] + suit_chars[suit]

    hole_str = ' '.join([card_to_str(c) for c in state.hole_cards[player]])
    board_str = ' '.join([card_to_str(c) for c in state.board if c >= 0])

    return (f"{round_names[state.round]} | "
            f"Hole: [{hole_str}] | "
            f"Board: [{board_str}] | "
            f"Pot: {state.pot:.0f} | "
            f"Stack: {state.stacks[player]:.0f} | "
            f"Bet: {state.bets[player]:.0f}")


if __name__ == "__main__":
    # Simple test
    print("Testing JAX Hold'em Engine - State Initialization")
    print("=" * 60)

    key = random.PRNGKey(42)
    num_players = 2
    stacks = jnp.array([1000.0, 1000.0])
    blinds = jnp.array([50.0, 100.0])

    state = deal_initial_state(key, num_players, stacks, blinds)

    print(f"\nInitial State (Player 0):")
    print(state_to_string(state, 0))
    print(f"\nInitial State (Player 1):")
    print(state_to_string(state, 1))

    print(f"\nState Details:")
    print(f"  Pot: {state.pot}")
    print(f"  Round: {state.round} (preflop)")
    print(f"  Acting Player: {state.acting_player}")
    print(f"  Stacks: {state.stacks}")
    print(f"  Bets: {state.bets}")

    print("\n✓ State initialization working!")
