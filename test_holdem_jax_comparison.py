#!/usr/bin/env python3
"""
Test Hold'em JAX V1 vs V2 Comparison

Verifies that the JAX-native V2 implementation produces IDENTICAL
results to the original V1 implementation for No-Limit Hold'em.

Requirements:
    source ~/open_spiel/venv/bin/activate

Phase 10.2: JAX-Native Hold'em Poker Validation
"""

import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr import holdem_jax as holdem_v1
from matrix_cfr import holdem_jax_v2 as holdem_v2


# Standard heads-up Hold'em configuration
NUM_PLAYERS = 2
STACKS = jnp.array([1000.0, 1000.0])
BLINDS = jnp.array([50.0, 100.0])


def test_deal_initial_state():
    """Test that initial state is identical."""
    print("="*70)
    print("Test 1: Deal Initial State")
    print("="*70)
    print()

    key = random.PRNGKey(42)

    state_v1 = holdem_v1.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)
    state_v2 = holdem_v2.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)

    # Compare all fields
    assert jnp.array_equal(state_v1.hole_cards, state_v2.hole_cards), "Hole cards mismatch!"
    assert jnp.array_equal(state_v1.board, state_v2.board), "Board mismatch!"
    assert jnp.array_equal(state_v1.deck, state_v2.deck), "Deck mismatch!"
    assert jnp.array_equal(state_v1.bets, state_v2.bets), "Bets mismatch!"
    assert state_v1.pot == state_v2.pot, "Pot mismatch!"
    assert jnp.array_equal(state_v1.stacks, state_v2.stacks), "Stacks mismatch!"
    assert state_v1.round == state_v2.round, "Round mismatch!"
    assert state_v1.acting_player == state_v2.acting_player, "Acting player mismatch!"
    assert jnp.array_equal(state_v1.folded, state_v2.folded), "Folded flags mismatch!"
    assert jnp.array_equal(state_v1.all_in, state_v2.all_in), "All-in flags mismatch!"

    print(f"✓ V1 hole cards: {state_v1.hole_cards}")
    print(f"✓ V2 hole cards: {state_v2.hole_cards}")
    print(f"✓ V1 pot: {state_v1.pot}, V2 pot: {state_v2.pot}")
    print(f"✓ V1 acting: {state_v1.acting_player}, V2 acting: {state_v2.acting_player}")
    print("✅ Initial states are IDENTICAL")
    print()


def test_legal_actions():
    """Test that legal actions are identical."""
    print("="*70)
    print("Test 2: Legal Actions")
    print("="*70)
    print()

    key = random.PRNGKey(123)

    state_v1 = holdem_v1.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)
    state_v2 = holdem_v2.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)

    legal_v1 = holdem_v1.legal_actions(state_v1)
    legal_v2 = holdem_v2.legal_actions(state_v2)

    assert jnp.array_equal(legal_v1, legal_v2), "Legal actions mismatch!"

    print(f"✓ V1 legal: {legal_v1}")
    print(f"✓ V2 legal: {legal_v2}")
    print("✅ Legal actions are IDENTICAL")
    print()


def test_apply_action_fold():
    """Test applying FOLD action."""
    print("="*70)
    print("Test 3: Apply FOLD Action")
    print("="*70)
    print()

    key = random.PRNGKey(456)

    state_v1 = holdem_v1.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)
    state_v2 = holdem_v2.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)

    k1, key = random.split(key)
    new_state_v1 = holdem_v1.apply_action(state_v1, holdem_v1.ACTION_FOLD, k1)

    k2, key = random.split(key)
    new_state_v2 = holdem_v2.apply_action(state_v2, holdem_v2.ACTION_FOLD, k2)

    # Compare terminal state
    assert holdem_v1.is_terminal(new_state_v1) == holdem_v2.is_terminal(new_state_v2), "Terminal status mismatch!"
    assert jnp.array_equal(new_state_v1.folded, new_state_v2.folded), "Folded flags mismatch!"

    print(f"✓ V1 folded: {new_state_v1.folded}")
    print(f"✓ V2 folded: {new_state_v2.folded}")
    print(f"✓ V1 terminal: {holdem_v1.is_terminal(new_state_v1)}")
    print(f"✓ V2 terminal: {holdem_v2.is_terminal(new_state_v2)}")
    print("✅ FOLD action results are IDENTICAL")
    print()


def test_apply_action_call():
    """Test applying CALL action."""
    print("="*70)
    print("Test 4: Apply CALL Action")
    print("="*70)
    print()

    key = random.PRNGKey(789)

    state_v1 = holdem_v1.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)
    state_v2 = holdem_v2.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)

    k1, key = random.split(key)
    new_state_v1 = holdem_v1.apply_action(state_v1, holdem_v1.ACTION_CALL, k1)

    k2, key = random.split(key)
    new_state_v2 = holdem_v2.apply_action(state_v2, holdem_v2.ACTION_CALL, k2)

    # Compare state after call
    assert jnp.array_equal(new_state_v1.bets, new_state_v2.bets), "Bets mismatch!"
    assert new_state_v1.pot == new_state_v2.pot, f"Pot mismatch! V1={new_state_v1.pot}, V2={new_state_v2.pot}"
    assert jnp.array_equal(new_state_v1.stacks, new_state_v2.stacks), "Stacks mismatch!"
    assert new_state_v1.acting_player == new_state_v2.acting_player, "Acting player mismatch!"

    print(f"✓ V1 bets: {new_state_v1.bets}, pot: {new_state_v1.pot}")
    print(f"✓ V2 bets: {new_state_v2.bets}, pot: {new_state_v2.pot}")
    print(f"✓ V1 acting: {new_state_v1.acting_player}")
    print(f"✓ V2 acting: {new_state_v2.acting_player}")
    print("✅ CALL action results are IDENTICAL")
    print()


def test_evaluate_hand_simple():
    """Test hand evaluation function."""
    print("="*70)
    print("Test 5: Hand Evaluation")
    print("="*70)
    print()

    # Test various hand types
    test_cases = [
        # (hole_cards, board, description)
        (jnp.array([0, 4]), jnp.array([8, 12, 16, 20, 24]), "Straight (2-6)"),
        (jnp.array([51, 47]), jnp.array([43, 39, 35, -1, -1]), "Pair of Aces"),
        (jnp.array([0, 1]), jnp.array([2, 3, 8, 12, -1]), "Four 2s"),
        (jnp.array([10, 14]), jnp.array([18, 22, 26, 30, -1]), "High card"),
    ]

    for hole_cards, board, desc in test_cases:
        score_v1 = holdem_v1.evaluate_hand_simple(hole_cards, board)
        score_v2 = holdem_v2.evaluate_hand_simple(hole_cards, board)

        assert jnp.allclose(score_v1, score_v2), f"Hand eval mismatch for {desc}! V1={score_v1}, V2={score_v2}"
        print(f"✓ {desc}: V1={score_v1:.1f}, V2={score_v2:.1f}")

    print("✅ Hand evaluations are IDENTICAL")
    print()


def simulate_random_hand_v1(key, num_players, stacks, blinds):
    """Simulate one random hand to completion using V1."""
    state = holdem_v1.deal_initial_state(key, num_players, stacks, blinds)

    max_actions = 100  # Safety limit
    for _ in range(max_actions):
        if holdem_v1.is_terminal(state):
            break

        legal = holdem_v1.legal_actions(state)
        k1, key = random.split(key)

        # Choose random legal action
        action_idx = random.randint(k1, (), 0, jnp.sum(legal).astype(jnp.int32))
        action = jnp.where(legal)[0][action_idx]

        k2, key = random.split(key)
        state = holdem_v1.apply_action(state, action, k2)

    payoffs_result = holdem_v1.payoffs(state)
    return payoffs_result


def simulate_random_hand_v2(key, num_players, stacks, blinds):
    """Simulate one random hand to completion using V2."""
    state = holdem_v2.deal_initial_state(key, num_players, stacks, blinds)

    max_actions = 100  # Safety limit
    for _ in range(max_actions):
        if holdem_v2.is_terminal(state):
            break

        legal = holdem_v2.legal_actions(state)
        k1, key = random.split(key)

        # Choose random legal action
        action_idx = random.randint(k1, (), 0, jnp.sum(legal).astype(jnp.int32))
        action = jnp.where(legal)[0][action_idx]

        k2, key = random.split(key)
        state = holdem_v2.apply_action(state, action, k2)

    payoffs_result = holdem_v2.payoffs(state)
    return payoffs_result


def test_complete_hand_fold():
    """Test a complete hand where player folds."""
    print("="*70)
    print("Test 6: Complete Hand (Fold)")
    print("="*70)
    print()

    key = random.PRNGKey(100)

    # P0 folds immediately
    state_v1 = holdem_v1.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)
    state_v2 = holdem_v2.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)

    k1, key = random.split(key)
    state_v1 = holdem_v1.apply_action(state_v1, holdem_v1.ACTION_FOLD, k1)

    k2, key = random.split(key)
    state_v2 = holdem_v2.apply_action(state_v2, holdem_v2.ACTION_FOLD, k2)

    payoffs_v1 = holdem_v1.payoffs(state_v1)
    payoffs_v2 = holdem_v2.payoffs(state_v2)

    assert jnp.allclose(payoffs_v1, payoffs_v2), f"Payoffs mismatch! V1={payoffs_v1}, V2={payoffs_v2}"

    print(f"✓ V1 payoffs: {payoffs_v1}")
    print(f"✓ V2 payoffs: {payoffs_v2}")
    print("✅ Complete fold hand payoffs are IDENTICAL")
    print()


def test_100_random_hands():
    """Test 100 random hands and verify identical payoffs."""
    print("="*70)
    print("Test 7: 100 Random Hands")
    print("="*70)
    print()

    print("Simulating 100 random hands with identical random seeds...")
    print()

    mismatches = 0
    mismatch_details = []

    for i in range(100):
        key = random.PRNGKey(i * 1000)

        # Simulate with V1
        payoffs_v1 = simulate_random_hand_v1(key, NUM_PLAYERS, STACKS, BLINDS)

        # Simulate with V2 (same seed)
        payoffs_v2 = simulate_random_hand_v2(key, NUM_PLAYERS, STACKS, BLINDS)

        # Check if payoffs match
        if not jnp.allclose(payoffs_v1, payoffs_v2, atol=1e-4):
            mismatches += 1
            mismatch_details.append((i, payoffs_v1, payoffs_v2))
            if mismatches <= 5:  # Only print first 5
                print(f"  ❌ Hand {i}: V1={payoffs_v1}, V2={payoffs_v2}")

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/100 hands... ({mismatches} mismatches so far)")

    print()
    if mismatches == 0:
        print(f"✅ ALL 100 hands produced IDENTICAL payoffs!")
        print()
    else:
        print(f"❌ {mismatches} mismatches found!")
        print()
        print("First few mismatches:")
        for i, v1, v2 in mismatch_details[:5]:
            print(f"  Hand {i}: V1={v1}, V2={v2}, diff={v1 - v2}")
        print()
        raise AssertionError(f"{mismatches}/100 hands had mismatched payoffs!")


def test_jit_compilation():
    """Test that V2 can be JIT-compiled."""
    print("="*70)
    print("Test 8: JIT Compilation (V2 Only)")
    print("="*70)
    print()

    print("JIT-compiling V2 functions...")

    # JIT compile apply_action
    @jax.jit
    def jit_apply_action(state, action, key):
        return holdem_v2.apply_action(state, action, key)

    # JIT compile payoffs
    jit_payoffs = jax.jit(holdem_v2.payoffs)

    # JIT compile evaluate_hand_simple
    jit_eval_hand = jax.jit(holdem_v2.evaluate_hand_simple)

    # Test JIT-compiled versions
    key = random.PRNGKey(999)
    state = holdem_v2.deal_initial_state(key, NUM_PLAYERS, STACKS, BLINDS)

    k1, key = random.split(key)
    state = jit_apply_action(state, holdem_v2.ACTION_POT_BET, k1)

    k2, key = random.split(key)
    state = jit_apply_action(state, holdem_v2.ACTION_FOLD, k2)

    assert holdem_v2.is_terminal(state), "Should be terminal!"

    payoffs_jit = jit_payoffs(state)

    # Test hand evaluation
    hole_cards = jnp.array([0, 4])
    board = jnp.array([8, 12, 16, 20, -1])
    score_jit = jit_eval_hand(hole_cards, board)

    print(f"✓ JIT-compiled apply_action works")
    print(f"✓ JIT-compiled payoffs works: {payoffs_jit}")
    print(f"✓ JIT-compiled evaluate_hand_simple works: {score_jit}")
    print("✅ V2 JIT compilation SUCCESSFUL")
    print()


def main():
    print("Hold'em JAX V1 vs V2 Comparison Test Suite")
    print("Phase 10.2: JAX-Native Validation")
    print()

    # Run all tests
    test_deal_initial_state()
    test_legal_actions()
    test_apply_action_fold()
    test_apply_action_call()
    test_evaluate_hand_simple()
    test_complete_hand_fold()
    test_100_random_hands()
    test_jit_compilation()

    print("="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70)
    print()
    print("Hold'em JAX V2 is FUNCTIONALLY IDENTICAL to V1")
    print("V2 is fully JAX-traceable and JIT-compilable")
    print()
    print("Next: Test batched trajectory sampling with V2!")


if __name__ == "__main__":
    main()
