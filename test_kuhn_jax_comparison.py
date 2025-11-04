#!/usr/bin/env python3
"""
Test Kuhn JAX V1 vs V2 Comparison

Verifies that the JAX-native V2 implementation produces IDENTICAL
results to the original V1 implementation.

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.2: JAX-Native Kuhn Poker Validation (Day 1)
"""

import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr import kuhn_jax as kuhn_v1
from matrix_cfr import kuhn_jax_v2 as kuhn_v2


def test_deal_initial_state():
    """Test that initial state is identical."""
    print("="*70)
    print("Test 1: Deal Initial State")
    print("="*70)
    print()

    key = random.PRNGKey(42)

    state_v1 = kuhn_v1.deal_initial_state(key)
    state_v2 = kuhn_v2.deal_initial_state(key)

    # Compare all fields
    assert jnp.array_equal(state_v1.cards, state_v2.cards), "Cards mismatch!"
    assert state_v1.pot == state_v2.pot, "Pot mismatch!"
    assert jnp.array_equal(state_v1.player_bets, state_v2.player_bets), "Bets mismatch!"
    assert state_v1.acting_player == state_v2.acting_player, "Acting player mismatch!"
    assert jnp.array_equal(state_v1.history, state_v2.history), "History mismatch!"
    assert state_v1.history_length == state_v2.history_length, "History length mismatch!"

    print(f"✓ V1 state: {kuhn_v1.state_to_string(state_v1)}")
    print(f"✓ V2 state: {kuhn_v2.state_to_string(state_v2)}")
    print("✅ Initial states are IDENTICAL")
    print()


def test_legal_actions():
    """Test that legal actions are identical."""
    print("="*70)
    print("Test 2: Legal Actions")
    print("="*70)
    print()

    key = random.PRNGKey(123)

    state_v1 = kuhn_v1.deal_initial_state(key)
    state_v2 = kuhn_v2.deal_initial_state(key)

    legal_v1 = kuhn_v1.legal_actions(state_v1)
    legal_v2 = kuhn_v2.legal_actions(state_v2)

    assert jnp.array_equal(legal_v1, legal_v2), "Legal actions mismatch!"

    print(f"✓ V1 legal: {legal_v1}")
    print(f"✓ V2 legal: {legal_v2}")
    print("✅ Legal actions are IDENTICAL")
    print()


def test_apply_action_pass():
    """Test applying PASS action."""
    print("="*70)
    print("Test 3: Apply PASS Action")
    print("="*70)
    print()

    key = random.PRNGKey(456)

    state_v1 = kuhn_v1.deal_initial_state(key)
    state_v2 = kuhn_v2.deal_initial_state(key)

    new_state_v1 = kuhn_v1.apply_action(state_v1, kuhn_v1.ACTION_PASS)
    new_state_v2 = kuhn_v2.apply_action(state_v2, kuhn_v2.ACTION_PASS)

    # Compare all fields
    assert jnp.array_equal(new_state_v1.cards, new_state_v2.cards), "Cards mismatch!"
    assert new_state_v1.pot == new_state_v2.pot, f"Pot mismatch! V1={new_state_v1.pot}, V2={new_state_v2.pot}"
    assert jnp.array_equal(new_state_v1.player_bets, new_state_v2.player_bets), "Bets mismatch!"
    assert new_state_v1.acting_player == new_state_v2.acting_player, "Acting player mismatch!"
    assert jnp.array_equal(new_state_v1.history, new_state_v2.history), "History mismatch!"
    assert new_state_v1.history_length == new_state_v2.history_length, "History length mismatch!"

    print(f"✓ V1 after pass: {kuhn_v1.state_to_string(new_state_v1)}")
    print(f"✓ V2 after pass: {kuhn_v2.state_to_string(new_state_v2)}")
    print("✅ PASS action results are IDENTICAL")
    print()


def test_apply_action_bet():
    """Test applying BET action."""
    print("="*70)
    print("Test 4: Apply BET Action")
    print("="*70)
    print()

    key = random.PRNGKey(789)

    state_v1 = kuhn_v1.deal_initial_state(key)
    state_v2 = kuhn_v2.deal_initial_state(key)

    new_state_v1 = kuhn_v1.apply_action(state_v1, kuhn_v1.ACTION_BET)
    new_state_v2 = kuhn_v2.apply_action(state_v2, kuhn_v2.ACTION_BET)

    # Compare all fields
    assert jnp.array_equal(new_state_v1.cards, new_state_v2.cards), "Cards mismatch!"
    assert new_state_v1.pot == new_state_v2.pot, f"Pot mismatch! V1={new_state_v1.pot}, V2={new_state_v2.pot}"
    assert jnp.array_equal(new_state_v1.player_bets, new_state_v2.player_bets), "Bets mismatch!"
    assert new_state_v1.acting_player == new_state_v2.acting_player, "Acting player mismatch!"
    assert jnp.array_equal(new_state_v1.history, new_state_v2.history), "History mismatch!"
    assert new_state_v1.history_length == new_state_v2.history_length, "History length mismatch!"

    print(f"✓ V1 after bet: {kuhn_v1.state_to_string(new_state_v1)}")
    print(f"✓ V2 after bet: {kuhn_v2.state_to_string(new_state_v2)}")
    print("✅ BET action results are IDENTICAL")
    print()


def test_complete_game_showdown():
    """Test complete game ending in showdown."""
    print("="*70)
    print("Test 5: Complete Game - Pass, Bet, Bet (Showdown)")
    print("="*70)
    print()

    key = random.PRNGKey(111)

    state_v1 = kuhn_v1.deal_initial_state(key)
    state_v2 = kuhn_v2.deal_initial_state(key)

    # Play: Pass, Bet, Bet
    state_v1 = kuhn_v1.apply_action(state_v1, kuhn_v1.ACTION_PASS)
    state_v2 = kuhn_v2.apply_action(state_v2, kuhn_v2.ACTION_PASS)

    state_v1 = kuhn_v1.apply_action(state_v1, kuhn_v1.ACTION_BET)
    state_v2 = kuhn_v2.apply_action(state_v2, kuhn_v2.ACTION_BET)

    state_v1 = kuhn_v1.apply_action(state_v1, kuhn_v1.ACTION_BET)
    state_v2 = kuhn_v2.apply_action(state_v2, kuhn_v2.ACTION_BET)

    # Check terminal
    assert kuhn_v1.is_terminal(state_v1), "V1 should be terminal!"
    assert kuhn_v2.is_terminal(state_v2), "V2 should be terminal!"

    # Compare payoffs
    payoffs_v1 = kuhn_v1.payoffs(state_v1)
    payoffs_v2 = kuhn_v2.payoffs(state_v2)

    assert jnp.allclose(payoffs_v1, payoffs_v2, atol=1e-6), f"Payoffs mismatch! V1={payoffs_v1}, V2={payoffs_v2}"

    print(f"✓ V1 terminal: {kuhn_v1.state_to_string(state_v1)}")
    print(f"✓ V2 terminal: {kuhn_v2.state_to_string(state_v2)}")
    print(f"✓ V1 payoffs: {payoffs_v1}")
    print(f"✓ V2 payoffs: {payoffs_v2}")
    print("✅ Showdown game results are IDENTICAL")
    print()


def test_complete_game_fold():
    """Test complete game ending in fold."""
    print("="*70)
    print("Test 6: Complete Game - Bet, Pass (Fold)")
    print("="*70)
    print()

    key = random.PRNGKey(222)

    state_v1 = kuhn_v1.deal_initial_state(key)
    state_v2 = kuhn_v2.deal_initial_state(key)

    # Play: Bet, Pass
    state_v1 = kuhn_v1.apply_action(state_v1, kuhn_v1.ACTION_BET)
    state_v2 = kuhn_v2.apply_action(state_v2, kuhn_v2.ACTION_BET)

    state_v1 = kuhn_v1.apply_action(state_v1, kuhn_v1.ACTION_PASS)
    state_v2 = kuhn_v2.apply_action(state_v2, kuhn_v2.ACTION_PASS)

    # Check terminal
    assert kuhn_v1.is_terminal(state_v1), "V1 should be terminal!"
    assert kuhn_v2.is_terminal(state_v2), "V2 should be terminal!"

    # Compare payoffs
    payoffs_v1 = kuhn_v1.payoffs(state_v1)
    payoffs_v2 = kuhn_v2.payoffs(state_v2)

    assert jnp.allclose(payoffs_v1, payoffs_v2, atol=1e-6), f"Payoffs mismatch! V1={payoffs_v1}, V2={payoffs_v2}"

    print(f"✓ V1 terminal: {kuhn_v1.state_to_string(state_v1)}")
    print(f"✓ V2 terminal: {kuhn_v2.state_to_string(state_v2)}")
    print(f"✓ V1 payoffs: {payoffs_v1}")
    print(f"✓ V2 payoffs: {payoffs_v2}")
    print("✅ Fold game results are IDENTICAL")
    print()


def test_1000_random_games():
    """Test 1000 random games to ensure identical results."""
    print("="*70)
    print("Test 7: 1000 Random Games")
    print("="*70)
    print()

    print("Playing 1000 random games with both implementations...")
    print()

    mismatches = 0

    for seed in range(1000):
        key = random.PRNGKey(seed)

        # Play with V1
        state_v1 = kuhn_v1.deal_initial_state(key)

        # Random policy: 50% pass, 50% bet
        key, subkey = random.split(key)
        while not kuhn_v1.is_terminal(state_v1):
            action = random.choice(subkey, jnp.array([kuhn_v1.ACTION_PASS, kuhn_v1.ACTION_BET]))
            state_v1 = kuhn_v1.apply_action(state_v1, int(action))
            key, subkey = random.split(key)

        payoffs_v1 = kuhn_v1.payoffs(state_v1)

        # Play with V2 (same seed)
        key = random.PRNGKey(seed)
        state_v2 = kuhn_v2.deal_initial_state(key)

        key, subkey = random.split(key)
        while not kuhn_v2.is_terminal(state_v2):
            action = random.choice(subkey, jnp.array([kuhn_v2.ACTION_PASS, kuhn_v2.ACTION_BET]))
            state_v2 = kuhn_v2.apply_action(state_v2, int(action))
            key, subkey = random.split(key)

        payoffs_v2 = kuhn_v2.payoffs(state_v2)

        # Compare payoffs
        if not jnp.allclose(payoffs_v1, payoffs_v2, atol=1e-6):
            print(f"❌ MISMATCH in game {seed}:")
            print(f"   V1 payoffs: {payoffs_v1}")
            print(f"   V2 payoffs: {payoffs_v2}")
            mismatches += 1

    if mismatches == 0:
        print(f"✅ ALL 1000 games produced IDENTICAL payoffs!")
        print()
    else:
        print(f"❌ {mismatches} mismatches found!")
        print()
        raise AssertionError(f"{mismatches}/1000 games had mismatched payoffs!")


def test_jit_compilation():
    """Test that V2 can be JIT-compiled."""
    print("="*70)
    print("Test 8: JIT Compilation (V2 Only)")
    print("="*70)
    print()

    print("JIT-compiling V2 functions...")

    # JIT compile apply_action
    jit_apply_action = jax.jit(kuhn_v2.apply_action)

    # JIT compile payoffs
    jit_payoffs = jax.jit(kuhn_v2.payoffs)

    # Test JIT-compiled versions
    key = random.PRNGKey(999)
    state = kuhn_v2.deal_initial_state(key)

    state = jit_apply_action(state, kuhn_v2.ACTION_BET)
    state = jit_apply_action(state, kuhn_v2.ACTION_PASS)

    assert kuhn_v2.is_terminal(state), "Should be terminal!"

    payoffs_jit = jit_payoffs(state)

    print(f"✓ JIT-compiled apply_action works")
    print(f"✓ JIT-compiled payoffs works")
    print(f"✓ Terminal state: {kuhn_v2.state_to_string(state)}")
    print(f"✓ Payoffs: {payoffs_jit}")
    print("✅ V2 JIT compilation SUCCESSFUL")
    print()


def main():
    print("Kuhn JAX V1 vs V2 Comparison Test Suite")
    print("Phase 10.2: JAX-Native Validation")
    print()

    # Run all tests
    test_deal_initial_state()
    test_legal_actions()
    test_apply_action_pass()
    test_apply_action_bet()
    test_complete_game_showdown()
    test_complete_game_fold()
    test_1000_random_games()
    test_jit_compilation()

    print("="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70)
    print()
    print("Kuhn JAX V2 is FUNCTIONALLY IDENTICAL to V1")
    print("V2 is fully JAX-traceable and JIT-compilable")
    print()
    print("Next: Test batched trajectory sampling with V2!")


if __name__ == "__main__":
    main()
