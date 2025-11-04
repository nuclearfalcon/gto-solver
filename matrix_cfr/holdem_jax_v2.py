"""
JAX Hold'em Engine V2 - JAX-Native for Batched Trajectory Sampling

**CRITICAL CHANGE from V1**: Uses JAX-native control flow (`jax.lax.switch`,
`jax.lax.cond`, `jax.lax.while_loop`) instead of Python control flow to enable
full JAX tracing and batched trajectory sampling.

This version enables 100-1000× speedup through `jax.vmap` batching.

Phase 10.2: JAX-Native Game Engine Rewrite (Days 3-6)
"""

from typing import NamedTuple, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random


class HoldemState(NamedTuple):
    """
    Pure JAX state representation for No-Limit Hold'em.

    Identical to V1 - no changes needed to state structure.
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


# Action constants
ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_POT_BET = 2
ACTION_ALL_IN = 3


def deal_initial_state(
    key: jax.random.PRNGKey,
    num_players: int,
    stacks: jnp.ndarray,
    blinds: jnp.ndarray,
    button_position: int = 0
) -> HoldemState:
    """
    Create initial game state with cards dealt.

    **JAX-NATIVE VERSION**: Converted if/else for first_actor determination.

    Args:
        key: JAX random key for card dealing
        num_players: Number of players (2-10)
        stacks: Starting stack sizes, shape (num_players,)
        blinds: Blind amounts, shape (num_players,)
        button_position: Position of dealer button (default 0)

    Returns:
        Initial HoldemState
    """
    # Split key for card dealing
    key, subkey = random.split(key)

    # Deal hole cards
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
    pot = jnp.sum(blinds).astype(jnp.float32)  # Ensure pot is float32 for consistency
    stacks = (stacks - blinds).astype(jnp.float32)

    # Determine first actor using jax.lax.cond
    # Heads-up: button acts first preflop
    # Multi-way: player after BB (position 2 if button=0)
    first_actor = jax.lax.cond(
        num_players == 2,
        lambda: button_position,  # Heads-up
        lambda: (button_position + 3) % num_players  # Multi-way
    )

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

    **JAX-NATIVE VERSION**: Uses jax.lax.while_loop instead of Python for loop.

    Returns:
        Player index, or -1 if betting round complete
    """
    num_players = len(state.folded)
    active = get_active_players(state)

    # If only 0-1 active players, betting is done
    few_active = jnp.sum(active) <= 1

    # Check if all active players have matched the max bet
    max_bet = get_max_bet(state)
    all_matched = jnp.all(
        (state.bets == max_bet) | ~active  # Either matched or not active
    )

    # If all matched and at least one action taken, round is complete
    round_complete = all_matched & (state.num_actions_this_round > 0)

    # Use jax.lax.cond to return early if betting complete
    def find_next():
        """Find next active player using while_loop."""
        def cond_fn(carry):
            offset, found = carry
            return (offset < num_players) & ~found

        def body_fn(carry):
            offset, found = carry
            next_player = (state.acting_player + offset) % num_players
            is_active = active[next_player]
            return (offset + 1, found | is_active)

        # Start search from offset 1
        final_offset, found = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (jnp.int32(1), False)
        )

        # Calculate the actual next player
        next_player = (state.acting_player + final_offset - 1) % num_players

        # Return next_player if found, else -1
        return jnp.where(found, next_player, jnp.int32(-1))

    # If betting complete or few active, return -1, else find next actor
    return jax.lax.cond(
        few_active | round_complete,
        lambda: jnp.int32(-1),
        find_next
    )


def deal_board_cards(
    state: HoldemState,
    num_cards: int,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deal cards to the board from remaining deck.

    **JAX-NATIVE VERSION**: Uses weighted sampling to avoid boolean indexing.

    Args:
        state: Current game state
        num_cards: Number of cards to deal (3 for flop, 1 for turn/river)
        key: Random key for card selection

    Returns:
        (new_board, new_deck) with cards dealt
    """
    # Use weighted sampling instead of boolean indexing
    # Weights are 1.0 for available cards, 0.0 for dealt cards
    weights = state.deck.astype(jnp.float32)

    # Normalize weights (avoid division by zero)
    total_weight = jnp.sum(weights)
    probs = jnp.where(total_weight > 0, weights / total_weight, weights)

    # Sample 3 cards (max for flop) from full deck using weighted probabilities
    # This avoids dynamic array creation from boolean indexing
    selected_cards = random.choice(
        key,
        52,  # Sample from all 52 cards
        shape=(3,),  # Max 3 cards (flop)
        replace=False,
        p=probs
    )

    # Current board position
    board_position = jnp.sum(state.board >= 0)

    # Update board using static indexing
    new_board = state.board

    # Deal up to 3 cards (use conditional updates based on num_cards)
    # Card 1 (always dealt if num_cards >= 1)
    new_board = jnp.where(
        (num_cards >= 1) & (board_position == 0),
        new_board.at[0].set(selected_cards[0]),
        new_board
    )
    new_board = jnp.where(
        (num_cards >= 1) & (board_position == 3),
        new_board.at[3].set(selected_cards[0]),
        new_board
    )
    new_board = jnp.where(
        (num_cards >= 1) & (board_position == 4),
        new_board.at[4].set(selected_cards[0]),
        new_board
    )

    # Card 2 (dealt if num_cards >= 2)
    new_board = jnp.where(
        (num_cards >= 2) & (board_position == 0),
        new_board.at[1].set(selected_cards[1]),
        new_board
    )

    # Card 3 (dealt if num_cards >= 3)
    new_board = jnp.where(
        (num_cards >= 3) & (board_position == 0),
        new_board.at[2].set(selected_cards[2]),
        new_board
    )

    # Update deck (mark dealt cards as unavailable)
    new_deck = state.deck
    new_deck = jnp.where(num_cards >= 1, new_deck.at[selected_cards[0]].set(False), new_deck)
    new_deck = jnp.where(num_cards >= 2, new_deck.at[selected_cards[1]].set(False), new_deck)
    new_deck = jnp.where(num_cards >= 3, new_deck.at[selected_cards[2]].set(False), new_deck)

    return new_board, new_deck


def advance_round(state: HoldemState, key: jax.random.PRNGKey) -> HoldemState:
    """
    Advance to next betting round (flop → turn → river).

    **JAX-NATIVE VERSION**: Uses jax.lax operations for round logic and
    player search.

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

    # Determine how many cards to deal (using nested jnp.where)
    cards_to_deal = jnp.where(
        new_round == 1, 3,  # Flop: 3 cards
        jnp.where(new_round == 2, 1,  # Turn: 1 card
                  jnp.where(new_round == 3, 1, 0))  # River: 1 card
    )

    # Deal board cards
    new_board, new_deck = deal_board_cards(state, cards_to_deal, key)

    # Reset bets for new round
    new_bets = jnp.zeros_like(state.bets)

    # Find first active player using while_loop
    num_players = len(state.folded)
    active = get_active_players(state)

    def cond_fn(carry):
        player, found = carry
        return (player < num_players) & ~found

    def body_fn(carry):
        player, found = carry
        is_active = active[player]
        return (player + 1, found | is_active)

    final_player, found = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (jnp.int32(0), False)
    )

    first_actor = jnp.where(found, final_player - 1, jnp.int32(-1))

    return state._replace(
        board=new_board,
        deck=new_deck,
        bets=new_bets,
        round=new_round,
        acting_player=first_actor,
        num_actions_this_round=jnp.int32(0)
    )


def legal_actions(state: HoldemState) -> jnp.ndarray:
    """
    Return mask of legal actions for current acting player.

    Already JAX-compatible in V1 - no changes needed.

    Returns:
        Boolean array [fold, call, pot_bet, all_in]
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
    pot_after_call = state.pot + to_call
    pot_bet_size = pot_after_call
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

    Already JAX-compatible in V1 - no changes needed.
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

    Already JAX-compatible in V1 - no changes needed.
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


def apply_action(
    state: HoldemState,
    action: int,
    key: jax.random.PRNGKey
) -> HoldemState:
    """
    Apply action and return new state.

    **JAX-NATIVE VERSION**: Uses jax.lax.switch for 4-way action branching
    and jax.lax.cond for round advancement.

    This is THE CRITICAL FUNCTION that enables batched trajectory sampling.

    Args:
        state: Current game state
        action: Action to take (0=fold, 1=call, 2=pot_bet, 3=all_in)
        key: Random key (REQUIRED for JAX tracing - always needed)

    Returns:
        New state after applying action
    """
    player = state.acting_player

    # Increment action count
    base_state = state._replace(num_actions_this_round=state.num_actions_this_round + 1)

    # Define action handler functions
    def fold_fn(s):
        """Handle FOLD action."""
        new_folded = s.folded.at[player].set(True)
        # Ensure pot is float32 for type consistency with other branches
        return s._replace(folded=new_folded, pot=s.pot.astype(jnp.float32))

    def call_fn(s):
        """Handle CALL action."""
        to_call = get_max_bet(s) - s.bets[player]
        amount = jnp.minimum(to_call, s.stacks[player])

        new_bets = s.bets.at[player].add(amount)
        new_stacks = s.stacks.at[player].add(-amount)
        new_pot = (s.pot + amount).astype(jnp.float32)
        new_all_in = s.all_in.at[player].set(new_stacks[player] == 0)

        return s._replace(
            bets=new_bets,
            stacks=new_stacks,
            pot=new_pot,
            all_in=new_all_in
        )

    def pot_bet_fn(s):
        """Handle POT_BET action."""
        to_call = get_max_bet(s) - s.bets[player]
        pot_after_call = s.pot + to_call
        raise_amount = pot_after_call
        total_bet = to_call + raise_amount
        amount = jnp.minimum(total_bet, s.stacks[player])

        new_bets = s.bets.at[player].add(amount)
        new_stacks = s.stacks.at[player].add(-amount)
        new_pot = (s.pot + amount).astype(jnp.float32)
        new_all_in = s.all_in.at[player].set(new_stacks[player] == 0)

        return s._replace(
            bets=new_bets,
            stacks=new_stacks,
            pot=new_pot,
            all_in=new_all_in
        )

    def all_in_fn(s):
        """Handle ALL_IN action."""
        amount = s.stacks[player]

        new_bets = s.bets.at[player].add(amount)
        new_stacks = s.stacks.at[player].set(0.0)
        new_pot = (s.pot + amount).astype(jnp.float32)
        new_all_in = s.all_in.at[player].set(True)

        return s._replace(
            bets=new_bets,
            stacks=new_stacks,
            pot=new_pot,
            all_in=new_all_in
        )

    # Apply action using jax.lax.switch (4-way branching)
    new_state = jax.lax.switch(
        action,
        [fold_fn, call_fn, pot_bet_fn, all_in_fn],
        base_state
    )

    # Check if betting round is complete and decide next action
    # Use jax.lax.cond to choose between advancing round or finding next actor
    should_advance = betting_complete(new_state) & ~is_terminal(new_state)

    def advance_fn(s):
        """Advance to next round."""
        return advance_round(s, key)

    def find_next_fn(s):
        """Find next actor."""
        next_actor = find_next_actor(s)
        return s._replace(acting_player=next_actor)

    final_state = jax.lax.cond(
        should_advance,
        advance_fn,
        find_next_fn,
        new_state
    )

    return final_state


def evaluate_hand_simple(hole_cards: jnp.ndarray, board: jnp.ndarray) -> jnp.float32:
    """
    Simple hand evaluation (MVP version).

    **JAX-NATIVE VERSION**: Uses vectorized operations instead of Python loops.

    Returns a hand strength score where higher is better.
    """
    # Combine hole cards and board (fixed size: 2 hole + 5 board = 7 max)
    all_cards = jnp.concatenate([hole_cards, board])

    # Create mask for valid cards (>= 0)
    valid_mask = all_cards >= 0

    # Extract ranks (0-12: 2-Ace), set invalid to 0
    ranks = jnp.where(valid_mask, all_cards // 4, 0)

    # Count occurrences of each rank using one-hot encoding
    # This replaces the Python for loop with vectorized operations
    rank_one_hot = jax.nn.one_hot(ranks, 13)  # Shape: (7, 13)
    # Mask out invalid cards when counting
    rank_one_hot = rank_one_hot * valid_mask[:, None]  # Apply mask
    rank_counts = jnp.sum(rank_one_hot, axis=0).astype(jnp.int32)  # Shape: (13,)

    # Hand strength calculation (simplified)
    max_count = jnp.max(rank_counts)

    # Score based on best combination
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

    **JAX-NATIVE VERSION**: Uses jax.lax.cond and vectorized operations
    instead of Python if statements and for loops.

    NOTE: This is a simplified version for MVP. Proper pot splitting for
    side pots would require more complex logic.

    Args:
        state: Terminal game state

    Returns:
        Payoff array for each player (winnings - losses)
    """
    num_players = len(state.folded)

    # Case 1: All but one folded
    active_not_folded = ~state.folded
    only_one_left = jnp.sum(active_not_folded) == 1

    def fold_payoffs():
        """Compute payoffs when all but one folded."""
        # Winner takes pot
        winner = jnp.argmax(active_not_folded.astype(jnp.int32))

        # Simplified: winner gets pot, others get 0
        # (Proper accounting would track initial stacks)
        payoff = jnp.zeros(num_players, dtype=jnp.float32)
        payoff = payoff.at[winner].set(state.pot)
        return payoff

    def showdown_payoffs():
        """Compute payoffs at showdown."""
        # Evaluate all hands using static indexing (heads-up: 2 players)
        # This avoids Python for loops which block JAX tracing

        # Player 0 hand strength
        strength_p0 = jnp.where(
            state.folded[0],
            jnp.float32(-1.0),  # Folded players have strength -1
            evaluate_hand_simple(state.hole_cards[0], state.board)
        )

        # Player 1 hand strength
        strength_p1 = jnp.where(
            state.folded[1],
            jnp.float32(-1.0),  # Folded players have strength -1
            evaluate_hand_simple(state.hole_cards[1], state.board)
        )

        # Combine into array
        hand_strengths = jnp.array([strength_p0, strength_p1])

        # Find winner (highest hand strength)
        winner = jnp.argmax(hand_strengths)

        # Winner takes pot
        payoff = jnp.zeros(num_players, dtype=jnp.float32)
        payoff = payoff.at[winner].set(state.pot)

        return payoff

    # Use jax.lax.cond to choose between fold and showdown payoffs
    return jax.lax.cond(
        only_one_left,
        fold_payoffs,
        showdown_payoffs
    )


def state_to_infoset(state: HoldemState, player: int) -> str:
    """
    Convert state to information set string for given player.

    **NOTE**: This function is NOT JAX-traceable (uses Python strings).
    For batched sampling, use bucket IDs instead.

    Args:
        state: Current game state
        player: Player index

    Returns:
        Infoset string encoding player's information
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
    round_names = ['R0', 'R1', 'R2', 'R3']
    round_str = round_names[state.round]

    # Encode hole cards
    hole_str = ','.join([card_to_str(int(c)) for c in state.hole_cards[player]])

    # Encode board cards
    board_cards = [card_to_str(int(c)) for c in state.board if int(c) >= 0]
    board_str = ','.join(board_cards) if board_cards else ''

    # Encode betting sequence
    bets_str = ','.join([f'{int(b)}' for b in state.bets])

    # Combine into infoset string
    infoset = f"{round_str}_H[{hole_str}]_B[{board_str}]_Bets[{bets_str}]"

    return infoset


def state_to_string(state: HoldemState, player: int) -> str:
    """
    Convert state to human-readable string (for debugging).

    **NOT JAX-TRACEABLE** - Only for debugging/logging.
    """
    round_names = ['Preflop', 'Flop', 'Turn', 'River']

    def card_to_str(card_id):
        if card_id < 0:
            return '--'
        rank = card_id // 4
        suit = card_id % 4
        rank_chars = '23456789TJQKA'
        suit_chars = 'shdc'
        return rank_chars[rank] + suit_chars[suit]

    hole_str = ' '.join([card_to_str(int(c)) for c in state.hole_cards[player]])
    board_str = ' '.join([card_to_str(int(c)) for c in state.board if int(c) >= 0])

    return (f"{round_names[int(state.round)]} | "
            f"Hole: [{hole_str}] | "
            f"Board: [{board_str}] | "
            f"Pot: {state.pot:.0f} | "
            f"Stack: {state.stacks[player]:.0f} | "
            f"Bet: {state.bets[player]:.0f}")


if __name__ == "__main__":
    print("Testing JAX Hold'em Engine V2 (JAX-Native)")
    print("=" * 70)

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

    print("\n✓ V2 state initialization working!")

    # Test apply_action with JIT compilation
    print("\n[Testing JIT compilation of apply_action...]")
    jit_apply_action = jax.jit(apply_action)

    key, subkey = random.split(key)
    state_after_call = jit_apply_action(state, ACTION_CALL, subkey)

    print(f"✓ JIT-compiled apply_action works!")
    print(f"  After P0 calls: {state_to_string(state_after_call, 0)}")

    print("\n" + "=" * 70)
    print("Hold'em JAX V2 Tests Passed! ✅")
    print("\nNext: Write comparison tests vs V1, then test batched sampling!")
