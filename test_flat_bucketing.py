"""
Test suite for flat bucketing (memory-leak-free version).

Tests that state_to_bucket_index_flat() produces identical results to
state_to_bucket_index() while avoiding NamedTuple creation in JIT functions.

Run with:
    source ~/open_spiel/venv/bin/activate
    python test_flat_bucketing.py
"""

import jax
import jax.numpy as jnp
import jax.random as random
from matrix_cfr.holdem_jax_v2 import HoldemState, deal_initial_state
from matrix_cfr.bucketing import state_to_bucket_index, state_to_bucket_index_flat
from matrix_cfr.gpu_mccfr_solver import flatten_state, unflatten_state


def create_test_state(num_players=2, seed=42):
    """Create a test Hold'em state."""
    key = random.PRNGKey(seed)
    return deal_initial_state(
        key,
        num_players=num_players,
        stacks=jnp.array([1000.0, 1000.0]),
        blinds=jnp.array([50.0, 100.0])
    )


def test_flat_vs_namedtuple_bucketing():
    """
    Test that flat bucketing produces identical results to NamedTuple bucketing.

    This is the critical test - if this passes, the refactoring is correct.
    """
    print("\n" + "="*70)
    print("TEST 1: Flat vs NamedTuple Bucketing Equivalence")
    print("="*70)

    # Create test state
    state = create_test_state(num_players=2)
    flat_state = flatten_state(state, num_players=2)

    # Test for both players
    for updating_player in [0, 1]:
        # Bucket using old method (with NamedTuple)
        bucket_old = state_to_bucket_index(
            state,
            updating_player=updating_player,
            num_buckets=10000,
            num_hand_buckets=200,
            num_pot_buckets=10
        )

        # Bucket using new method (flat, no NamedTuple)
        bucket_new = state_to_bucket_index_flat(
            flat_state,
            num_players=2,
            updating_player=updating_player,
            num_buckets=10000,
            num_hand_buckets=200,
            num_pot_buckets=10
        )

        print(f"Player {updating_player}:")
        print(f"  Old (NamedTuple): {bucket_old}")
        print(f"  New (Flat):       {bucket_new}")
        print(f"  Match: {bucket_old == bucket_new}")

        assert bucket_old == bucket_new, f"Bucket mismatch for player {updating_player}!"

    print("\n✅ PASS: Flat bucketing matches NamedTuple bucketing")


def test_vectorized_flat_bucketing():
    """
    Test vectorized bucketing with multiple random states.

    This tests the vmap integration that will be used in run_iteration_gpu_resident().
    """
    print("\n" + "="*70)
    print("TEST 2: Vectorized Flat Bucketing (100 states)")
    print("="*70)

    num_test_states = 100
    num_players = 2

    # Generate 100 random flat states
    key = random.PRNGKey(42)
    keys = random.split(key, num_test_states)

    states = []
    flat_states = []
    for i in range(num_test_states):
        state = create_test_state(num_players=num_players, seed=int(keys[i][0]))
        states.append(state)
        flat_states.append(flatten_state(state, num_players=num_players))

    flat_states_batch = jnp.stack(flat_states)

    # Bucket using vectorized flat method
    # Use functools.partial to bind num_players as a constant
    from functools import partial

    @jax.jit
    def batch_bucket_flat(states_batch, updating_player_const):
        # Bind num_players as constant using partial
        bucket_one_fn = partial(
            state_to_bucket_index_flat,
            num_players=num_players,  # Static constant
            updating_player=updating_player_const,
            num_buckets=10000,
            num_hand_buckets=200,
            num_pot_buckets=10
        )
        return jax.vmap(bucket_one_fn)(states_batch)

    buckets_flat = batch_bucket_flat(flat_states_batch, 0)

    # Verify shape and range
    print(f"Batch shape: {buckets_flat.shape}")
    print(f"Min bucket: {jnp.min(buckets_flat)}")
    print(f"Max bucket: {jnp.max(buckets_flat)}")
    print(f"Unique buckets: {len(jnp.unique(buckets_flat))}")

    assert buckets_flat.shape == (num_test_states,), f"Wrong shape: {buckets_flat.shape}"
    assert jnp.all(buckets_flat >= 0), "Found negative bucket indices!"
    assert jnp.all(buckets_flat < 10000), "Found bucket indices >= num_buckets!"

    # Compare with old method (spot check 10 random states)
    matches = 0
    for i in range(0, num_test_states, 10):
        bucket_old = state_to_bucket_index(states[i], updating_player=0)
        bucket_new = buckets_flat[i]
        if bucket_old == bucket_new:
            matches += 1

    print(f"\nSpot check (every 10th state): {matches}/10 matches")
    assert matches == 10, f"Only {matches}/10 spot checks matched!"

    print("\n✅ PASS: Vectorized flat bucketing works correctly")


def test_field_extraction():
    """
    Test that field extraction from flat state is correct.

    This verifies the offset calculations in state_to_bucket_index_flat().
    """
    print("\n" + "="*70)
    print("TEST 3: Field Extraction from Flat State")
    print("="*70)

    num_players = 2
    state = create_test_state(num_players=num_players)
    flat_state = flatten_state(state, num_players=num_players)

    # Calculate offsets (same as in state_to_bucket_index_flat)
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
    num_actions_idx = stacks_end + 2

    # Extract fields
    hole_cards_p0 = flat_state[0:2].astype(jnp.int32)
    hole_cards_p1 = flat_state[2:4].astype(jnp.int32)
    board = flat_state[board_start:board_end].astype(jnp.int32)
    bets = flat_state[bets_start:bets_end].astype(jnp.float32)
    pot = flat_state[pot_idx].astype(jnp.float32)
    stacks = flat_state[stacks_start:stacks_end].astype(jnp.float32)
    round_val = flat_state[round_idx].astype(jnp.int32)
    num_actions = flat_state[num_actions_idx].astype(jnp.int32)

    # Verify against original state
    print(f"Hole cards P0 - Original: {state.hole_cards[0]}, Extracted: {hole_cards_p0}")
    print(f"Hole cards P1 - Original: {state.hole_cards[1]}, Extracted: {hole_cards_p1}")
    print(f"Board - Original: {state.board}, Extracted: {board}")
    print(f"Bets - Original: {state.bets}, Extracted: {bets}")
    print(f"Pot - Original: {state.pot}, Extracted: {pot}")
    print(f"Stacks - Original: {state.stacks}, Extracted: {stacks}")
    print(f"Round - Original: {state.round}, Extracted: {round_val}")
    print(f"Num actions - Original: {state.num_actions_this_round}, Extracted: {num_actions}")

    assert jnp.array_equal(state.hole_cards[0], hole_cards_p0), "Hole cards P0 mismatch!"
    assert jnp.array_equal(state.hole_cards[1], hole_cards_p1), "Hole cards P1 mismatch!"
    assert jnp.array_equal(state.board, board), "Board mismatch!"
    assert jnp.allclose(state.bets, bets), "Bets mismatch!"
    assert jnp.isclose(state.pot, pot), "Pot mismatch!"
    assert jnp.allclose(state.stacks, stacks), "Stacks mismatch!"
    assert state.round == round_val, "Round mismatch!"
    assert state.num_actions_this_round == num_actions, "Num actions mismatch!"

    print("\n✅ PASS: Field extraction is correct")


def test_multi_player_support():
    """
    Test that flat bucketing works for different numbers of players.

    Currently the code supports 2 players, but this test documents the behavior.
    """
    print("\n" + "="*70)
    print("TEST 4: Multi-player Support")
    print("="*70)

    # Test with 2 players
    num_players = 2
    state = create_test_state(num_players=num_players)
    flat_state = flatten_state(state, num_players=num_players)

    bucket = state_to_bucket_index_flat(
        flat_state,
        num_players=num_players,
        updating_player=0
    )

    print(f"2-player game - Bucket: {bucket}")
    assert bucket >= 0 and bucket < 10000, "Invalid bucket for 2 players!"

    print("\n✅ PASS: Multi-player support (2 players tested)")
    print("Note: 3+ player support requires generalizing opponent calculation")


def test_jit_compilation():
    """
    Test that flat bucketing can be JIT-compiled without issues.

    This is critical because the old version had memory leaks from
    JIT-compiling NamedTuple creation.
    """
    print("\n" + "="*70)
    print("TEST 5: JIT Compilation")
    print("="*70)

    from functools import partial

    # Create JIT-compiled function with num_players bound as constant
    jit_bucket_fn = jax.jit(partial(
        state_to_bucket_index_flat,
        num_players=2,
        num_buckets=10000,
        num_hand_buckets=200,
        num_pot_buckets=10
    ))

    state = create_test_state(num_players=2)
    flat_state = flatten_state(state, num_players=2)

    # First call (triggers compilation)
    print("Compiling...")
    bucket1 = jit_bucket_fn(flat_state, updating_player=0)
    print(f"First call: {bucket1}")

    # Second call (uses cached compilation)
    bucket2 = jit_bucket_fn(flat_state, updating_player=0)
    print(f"Second call: {bucket2}")

    assert bucket1 == bucket2, "JIT compilation not deterministic!"

    print("\n✅ PASS: JIT compilation works correctly")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Flat Bucketing Test Suite")
    print("Testing memory-leak-free bucketing implementation")
    print("="*70)

    try:
        test_flat_vs_namedtuple_bucketing()
        test_vectorized_flat_bucketing()
        test_field_extraction()
        test_multi_player_support()
        test_jit_compilation()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        print("\nFlat bucketing implementation is correct!")
        print("Memory leak should be eliminated.")

    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED ❌")
        print("="*70)
        print(f"\nError: {e}")
        raise
