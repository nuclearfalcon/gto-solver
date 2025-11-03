"""
Phase 10 Tests: JAX Hold'em Engine

Tests for the pure-functional JAX-based Hold'em implementation.

Run with:
    source ~/open_spiel/venv/bin/activate
    python -m pytest tests/test_phase10_holdem.py -v
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr.holdem_jax import (
    HoldemState,
    deal_initial_state,
    get_max_bet,
    get_player_to_call,
    get_active_players,
    get_num_active_players,
    find_next_actor,
    advance_round,
    ACTION_FOLD,
    ACTION_CALL,
    ACTION_POT_BET,
    ACTION_ALL_IN,
)


class TestStateInitialization:
    """Test state representation and initialization."""

    def test_deal_initial_state_basic(self):
        """Test basic state initialization."""
        key = random.PRNGKey(42)
        num_players = 2
        stacks = jnp.array([1000.0, 1000.0])
        blinds = jnp.array([50.0, 100.0])

        state = deal_initial_state(key, num_players, stacks, blinds)

        # Check types
        assert isinstance(state, HoldemState)
        assert state.hole_cards.shape == (2, 2)
        assert state.board.shape == (5,)
        assert state.deck.shape == (52,)
        assert state.bets.shape == (2,)
        assert state.stacks.shape == (2,)
        assert state.folded.shape == (2,)
        assert state.all_in.shape == (2,)

    def test_deal_initial_state_cards(self):
        """Test that cards are dealt correctly."""
        key = random.PRNGKey(123)
        num_players = 2
        stacks = jnp.array([1000.0, 1000.0])
        blinds = jnp.array([50.0, 100.0])

        state = deal_initial_state(key, num_players, stacks, blinds)

        # Each player should have 2 cards
        assert state.hole_cards.shape == (2, 2)

        # All hole cards should be valid (0-51)
        assert jnp.all(state.hole_cards >= 0)
        assert jnp.all(state.hole_cards < 52)

        # All hole cards should be unique
        all_hole_cards = state.hole_cards.flatten()
        assert len(jnp.unique(all_hole_cards)) == 4

        # Dealt cards should be marked unavailable in deck
        for card in all_hole_cards:
            assert not state.deck[card]

        # Board should be empty (preflop)
        assert jnp.all(state.board == -1)

    def test_deal_initial_state_blinds(self):
        """Test that blinds are posted correctly."""
        key = random.PRNGKey(42)
        num_players = 2
        stacks = jnp.array([1000.0, 1000.0])
        blinds = jnp.array([50.0, 100.0])

        state = deal_initial_state(key, num_players, stacks, blinds)

        # Blinds should be posted
        assert float(state.bets[0]) == 50.0
        assert float(state.bets[1]) == 100.0

        # Pot should equal sum of blinds
        assert float(state.pot) == 150.0

        # Stacks should be reduced by blinds
        assert float(state.stacks[0]) == 950.0
        assert float(state.stacks[1]) == 900.0

    def test_deal_initial_state_game_flow(self):
        """Test initial game flow state."""
        key = random.PRNGKey(42)
        num_players = 2
        stacks = jnp.array([1000.0, 1000.0])
        blinds = jnp.array([50.0, 100.0])

        state = deal_initial_state(key, num_players, stacks, blinds)

        # Should be preflop
        assert int(state.round) == 0

        # Acting player should be set (button in heads-up)
        assert int(state.acting_player) == 0

        # No actions yet
        assert int(state.num_actions_this_round) == 0

        # No one folded or all-in initially
        assert not jnp.any(state.folded)
        assert not jnp.any(state.all_in)

    def test_deal_initial_state_reproducibility(self):
        """Test that same key produces same state."""
        key = random.PRNGKey(999)
        num_players = 2
        stacks = jnp.array([1000.0, 1000.0])
        blinds = jnp.array([50.0, 100.0])

        state1 = deal_initial_state(key, num_players, stacks, blinds)
        state2 = deal_initial_state(key, num_players, stacks, blinds)

        # Same key should produce identical states
        assert jnp.array_equal(state1.hole_cards, state2.hole_cards)
        assert jnp.array_equal(state1.deck, state2.deck)

    def test_deal_initial_state_different_keys(self):
        """Test that different keys produce different states."""
        key1 = random.PRNGKey(1)
        key2 = random.PRNGKey(2)
        num_players = 2
        stacks = jnp.array([1000.0, 1000.0])
        blinds = jnp.array([50.0, 100.0])

        state1 = deal_initial_state(key1, num_players, stacks, blinds)
        state2 = deal_initial_state(key2, num_players, stacks, blinds)

        # Different keys should (almost always) produce different cards
        assert not jnp.array_equal(state1.hole_cards, state2.hole_cards)

    def test_deal_initial_state_three_players(self):
        """Test initialization with 3 players."""
        key = random.PRNGKey(42)
        num_players = 3
        stacks = jnp.array([1000.0, 1000.0, 1000.0])
        blinds = jnp.array([50.0, 100.0, 0.0])  # SB, BB, no ante

        state = deal_initial_state(key, num_players, stacks, blinds)

        # Check shapes
        assert state.hole_cards.shape == (3, 2)
        assert state.bets.shape == (3,)
        assert state.stacks.shape == (3,)
        assert state.folded.shape == (3,)

        # 6 unique hole cards
        all_hole_cards = state.hole_cards.flatten()
        assert len(jnp.unique(all_hole_cards)) == 6

        # Pot and stacks correct
        assert float(state.pot) == 150.0
        assert float(state.stacks[0]) == 950.0  # Paid SB
        assert float(state.stacks[1]) == 900.0  # Paid BB
        assert float(state.stacks[2]) == 1000.0  # No blind


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_max_bet(self):
        """Test get_max_bet function."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 2,
            jnp.array([1000.0, 1000.0]),
            jnp.array([50.0, 100.0])
        )

        max_bet = get_max_bet(state)
        assert float(max_bet) == 100.0

    def test_get_player_to_call(self):
        """Test get_player_to_call function."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 2,
            jnp.array([1000.0, 1000.0]),
            jnp.array([50.0, 100.0])
        )

        # Player 0 needs to call 50 to match player 1's 100
        to_call_p0 = get_player_to_call(state, 0)
        assert float(to_call_p0) == 50.0

        # Player 1 already at max bet, needs 0 to call
        to_call_p1 = get_player_to_call(state, 1)
        assert float(to_call_p1) == 0.0

    def test_get_active_players(self):
        """Test get_active_players function."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 2,
            jnp.array([1000.0, 1000.0]),
            jnp.array([50.0, 100.0])
        )

        # Initially both players active
        active = get_active_players(state)
        assert jnp.array_equal(active, jnp.array([True, True]))

        # Mark player 0 as folded
        state = state._replace(folded=jnp.array([True, False]))
        active = get_active_players(state)
        assert jnp.array_equal(active, jnp.array([False, True]))

        # Mark player 1 as all-in
        state = state._replace(all_in=jnp.array([False, True]))
        active = get_active_players(state)
        assert jnp.array_equal(active, jnp.array([False, False]))

    def test_get_num_active_players(self):
        """Test get_num_active_players function."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 3,
            jnp.array([1000.0, 1000.0, 1000.0]),
            jnp.array([50.0, 100.0, 0.0])
        )

        # Initially all 3 active
        assert int(get_num_active_players(state)) == 3

        # Mark one folded
        state = state._replace(folded=jnp.array([True, False, False]))
        assert int(get_num_active_players(state)) == 2

        # Mark another all-in
        state = state._replace(all_in=jnp.array([False, True, False]))
        assert int(get_num_active_players(state)) == 1


class TestNextActor:
    """Test find_next_actor logic."""

    def test_find_next_actor_initial(self):
        """Test next actor from initial state."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 2,
            jnp.array([1000.0, 1000.0]),
            jnp.array([50.0, 100.0])
        )

        # In heads-up, after player 0 acts, player 1 should act
        next_actor = find_next_actor(state)
        # Since bets are not equal (50 vs 100), next is player 1
        assert int(next_actor) == 1

    def test_find_next_actor_after_action(self):
        """Test next actor after one action."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 2,
            jnp.array([1000.0, 1000.0]),
            jnp.array([50.0, 100.0])
        )

        # Simulate player 0 calling (matching 100)
        state = state._replace(
            bets=jnp.array([100.0, 100.0]),
            stacks=jnp.array([900.0, 900.0]),
            pot=200.0,
            num_actions_this_round=1
        )

        # Now bets are equal, so betting round should be complete
        next_actor = find_next_actor(state)
        assert int(next_actor) == -1  # No next actor, round complete

    def test_find_next_actor_one_folded(self):
        """Test next actor when one player folds."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 2,
            jnp.array([1000.0, 1000.0]),
            jnp.array([50.0, 100.0])
        )

        # Player 0 folds
        state = state._replace(
            folded=jnp.array([True, False]),
            num_actions_this_round=1
        )

        # Only 1 active player, betting complete
        next_actor = find_next_actor(state)
        assert int(next_actor) == -1


class TestRoundAdvancement:
    """Test advance_round function."""

    def test_advance_to_flop(self):
        """Test advancing from preflop to flop."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 2,
            jnp.array([1000.0, 1000.0]),
            jnp.array([50.0, 100.0])
        )

        # Advance to flop
        key_flop = random.PRNGKey(100)
        state_flop = advance_round(state, key_flop)

        # Round should be 1 (flop)
        assert int(state_flop.round) == 1

        # Board should have 3 cards
        board_cards = state_flop.board[state_flop.board >= 0]
        assert len(board_cards) == 3

        # All board cards should be unique and valid
        assert len(jnp.unique(board_cards)) == 3
        assert jnp.all(board_cards >= 0)
        assert jnp.all(board_cards < 52)

        # Board cards should not overlap with hole cards
        all_hole_cards = state.hole_cards.flatten()
        for board_card in board_cards:
            assert board_card not in all_hole_cards

        # Bets should be reset to 0
        assert jnp.all(state_flop.bets == 0)

        # Action counter reset
        assert int(state_flop.num_actions_this_round) == 0

    def test_advance_to_turn(self):
        """Test advancing from flop to turn."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 2,
            jnp.array([1000.0, 1000.0]),
            jnp.array([50.0, 100.0])
        )

        # Advance to flop
        key_flop = random.PRNGKey(100)
        state_flop = advance_round(state, key_flop)

        # Advance to turn
        key_turn = random.PRNGKey(200)
        state_turn = advance_round(state_flop, key_turn)

        # Round should be 2 (turn)
        assert int(state_turn.round) == 2

        # Board should have 4 cards
        board_cards = state_turn.board[state_turn.board >= 0]
        assert len(board_cards) == 4

        # All unique
        assert len(jnp.unique(board_cards)) == 4

    def test_advance_to_river(self):
        """Test advancing from turn to river."""
        key = random.PRNGKey(42)
        state = deal_initial_state(
            key, 2,
            jnp.array([1000.0, 1000.0]),
            jnp.array([50.0, 100.0])
        )

        # Advance through all rounds
        state = advance_round(state, random.PRNGKey(100))  # Flop
        state = advance_round(state, random.PRNGKey(200))  # Turn
        state = advance_round(state, random.PRNGKey(300))  # River

        # Round should be 3 (river)
        assert int(state.round) == 3

        # Board should have 5 cards
        board_cards = state.board[state.board >= 0]
        assert len(board_cards) == 5

        # All unique
        assert len(jnp.unique(board_cards)) == 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
