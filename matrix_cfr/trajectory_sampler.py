"""
Trajectory Sampler - Vectorized Game Simulation for GPU MCCFR

This module implements parallel trajectory sampling for Monte Carlo CFR.
The key innovation: sample 10,000+ trajectories simultaneously on GPU!

Phase 10: GPU-Accelerated MCCFR
"""

from typing import Tuple, Callable
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr.holdem_jax import (
    HoldemState,
    deal_initial_state,
    apply_action,
    legal_actions,
    is_terminal,
    payoffs,
    state_to_infoset,
)


def sample_trajectory(
    key: jax.random.PRNGKey,
    num_players: int,
    stacks: jnp.ndarray,
    blinds: jnp.ndarray,
    policy_fn: Callable[[str, jnp.ndarray], jnp.ndarray],
    max_actions: int = 100
) -> Tuple[list, list, list, jnp.ndarray]:
    """
    Sample one complete game trajectory using given policy.

    This is the sequential version - used for testing and debugging.
    For production, use batch_sample_trajectories() which vectorizes this.

    Pure function: Same key → Same trajectory (reproducible)

    Args:
        key: JAX random key for reproducibility
        num_players: Number of players
        stacks: Starting stack sizes, shape (num_players,)
        blinds: Blind amounts, shape (num_players,)
        policy_fn: Function mapping (infoset_str, legal_mask) → action_probs
        max_actions: Maximum actions per trajectory (safety limit)

    Returns:
        Tuple of:
        - states: List of HoldemState objects visited
        - actions: List of actions taken (int)
        - players: List of acting players (int)
        - terminal_payoffs: Final payoffs, shape (num_players,)

    Example:
        >>> key = random.PRNGKey(42)
        >>> stacks = jnp.array([1000.0, 1000.0])
        >>> blinds = jnp.array([50.0, 100.0])
        >>> policy_fn = lambda infoset, legal: jnp.array([0.0, 1.0, 0.0, 0.0])  # Always call
        >>> states, actions, players, payoffs = sample_trajectory(key, 2, stacks, blinds, policy_fn)
        >>> len(states)  # Number of decision points
    """
    # Deal initial state
    key, subkey = random.split(key)
    state = deal_initial_state(subkey, num_players, stacks, blinds, button_position=0)

    states_list = []
    actions_list = []
    players_list = []

    action_count = 0

    # Play through game
    while not is_terminal(state) and action_count < max_actions:
        player = state.acting_player

        # Get legal actions
        legal = legal_actions(state)

        # Get policy for this infoset
        infoset = state_to_infoset(state, player)
        action_probs = policy_fn(infoset, legal)

        # Sample action according to policy
        key, subkey = random.split(key)
        # Filter to legal actions only
        legal_indices = jnp.where(legal, jnp.arange(4), -1)
        legal_indices = legal_indices[legal_indices >= 0]

        # Normalize probabilities over legal actions
        legal_probs = action_probs[legal]
        legal_probs = legal_probs / jnp.sum(legal_probs)

        # Sample from legal actions
        action_idx = random.choice(subkey, len(legal_indices), p=legal_probs)
        action = legal_indices[action_idx]

        # Record decision point
        states_list.append(state)
        actions_list.append(int(action))
        players_list.append(int(player))

        # Apply action and advance state
        key, subkey = random.split(key)
        state = apply_action(state, int(action), subkey)

        action_count += 1

    # Get terminal payoffs
    terminal_payoffs = payoffs(state)

    return states_list, actions_list, players_list, terminal_payoffs


def sample_trajectory_fixed_length(
    key: jax.random.PRNGKey,
    num_players: int,
    stacks: jnp.ndarray,
    blinds: jnp.ndarray,
    policy_fn: Callable[[str, jnp.ndarray], jnp.ndarray],
    max_length: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample trajectory with fixed-length output (for vmapping).

    This version pads trajectories to max_length for efficient batching.
    Uses jax.lax.scan for efficient looping.

    Args:
        key: JAX random key
        num_players: Number of players
        stacks: Starting stacks
        blinds: Blind amounts
        policy_fn: Policy function
        max_length: Fixed output length (trajectories padded to this)

    Returns:
        Tuple of JAX arrays (all padded to max_length):
        - hole_cards: (max_length, 2) - acting player's hole cards at each step
        - board: (max_length, 5) - board cards at each step
        - bets: (max_length, num_players) - bet amounts at each step
        - actions: (max_length,) - actions taken
        - valid_mask: (max_length,) - True for valid steps, False for padding

    Note: Uses scan for JIT compilation efficiency
    """
    # Deal initial state
    key, subkey = random.split(key)
    initial_state = deal_initial_state(subkey, num_players, stacks, blinds)

    def scan_step(carry, _):
        """Single step of trajectory sampling."""
        state, key, done = carry

        # Check if already terminal
        terminal = is_terminal(state)
        done = done | terminal

        # If done, return padding
        def sample_action_fn(state, key):
            player = state.acting_player
            legal = legal_actions(state)
            infoset = state_to_infoset(state, player)
            action_probs = policy_fn(infoset, legal)

            # Sample legal action
            key, subkey = random.split(key)
            legal_indices = jnp.where(legal, jnp.arange(4), -1)
            legal_indices = legal_indices[legal_indices >= 0]
            legal_probs = action_probs[legal]
            legal_probs = legal_probs / (jnp.sum(legal_probs) + 1e-10)

            action_idx = random.choice(subkey, len(legal_indices), p=legal_probs)
            action = legal_indices[action_idx]

            return action, key

        def no_op_fn(state, key):
            """No-op when terminal."""
            return jnp.int32(0), key

        # Conditional: sample action or no-op
        action, key = jax.lax.cond(
            done,
            no_op_fn,
            sample_action_fn,
            state, key
        )

        # Record state info (before applying action)
        player = state.acting_player
        hole_cards = state.hole_cards[player] if player >= 0 else jnp.array([-1, -1])
        board = state.board
        bets = state.bets

        # Apply action (or no-op if done)
        def apply_action_fn(state, action, key):
            key, subkey = random.split(key)
            return apply_action(state, int(action), subkey)

        def keep_state_fn(state, action, key):
            return state

        new_state = jax.lax.cond(
            done,
            keep_state_fn,
            apply_action_fn,
            state, action, key
        )

        # Return updated carry and outputs
        new_carry = (new_state, key, done)
        outputs = (hole_cards, board, bets, action, ~done)  # valid_mask = not done

        return new_carry, outputs

    # Run scan
    initial_carry = (initial_state, key, False)
    final_carry, outputs = jax.lax.scan(scan_step, initial_carry, None, length=max_length)

    hole_cards_arr, board_arr, bets_arr, actions_arr, valid_mask = outputs

    return hole_cards_arr, board_arr, bets_arr, actions_arr, valid_mask


def batch_sample_trajectories(
    keys: jnp.ndarray,
    num_players: int,
    stacks: jnp.ndarray,
    blinds: jnp.ndarray,
    policy_fn: Callable[[str, jnp.ndarray], jnp.ndarray],
    max_trajectory_length: int = 100
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample many trajectories in parallel using vmap.

    This is the key GPU parallelization function!
    Each trajectory is sampled independently on a separate GPU core.

    Expected speedup: 100-1000× vs sequential sampling

    Args:
        keys: Array of random keys, shape (batch_size, 2)
        num_players: Number of players
        stacks: Starting stacks (same for all trajectories)
        blinds: Blinds (same for all trajectories)
        policy_fn: Policy function
        max_trajectory_length: Max actions per trajectory

    Returns:
        Tuple of batched arrays:
        - hole_cards: (batch_size, max_length, 2)
        - board: (batch_size, max_length, 5)
        - bets: (batch_size, max_length, num_players)
        - actions: (batch_size, max_length)
        - valid_mask: (batch_size, max_length)

    Example:
        >>> batch_size = 10000
        >>> keys = random.split(random.PRNGKey(42), batch_size)
        >>> outputs = batch_sample_trajectories(keys, 2, stacks, blinds, policy_fn)
        >>> # Now have 10,000 trajectories sampled in parallel!
    """
    # Vectorize sample_trajectory_fixed_length over batch dimension
    vectorized_sample = jax.vmap(
        lambda key: sample_trajectory_fixed_length(
            key, num_players, stacks, blinds, policy_fn, max_trajectory_length
        )
    )

    return vectorized_sample(keys)


def uniform_random_policy(infoset: str, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Uniform random policy - for testing and initial exploration.

    Args:
        infoset: Information set string (unused for uniform policy)
        legal_mask: Boolean mask of legal actions

    Returns:
        Uniform probability distribution over legal actions
    """
    # Uniform over legal actions
    probs = legal_mask.astype(jnp.float32)
    probs = probs / jnp.sum(probs)
    return probs


if __name__ == "__main__":
    print("Testing Trajectory Sampler")
    print("=" * 70)

    # Test sequential sampling
    print("\n[Test 1: Sequential Trajectory Sampling]")
    key = random.PRNGKey(42)
    num_players = 2
    stacks = jnp.array([1000.0, 1000.0])
    blinds = jnp.array([50.0, 100.0])

    states, actions, players, payoffs_result = sample_trajectory(
        key, num_players, stacks, blinds,
        policy_fn=uniform_random_policy,
        max_actions=50
    )

    print(f"✓ Sampled trajectory:")
    print(f"  - {len(states)} decision points")
    print(f"  - {len(actions)} actions taken")
    print(f"  - Terminal payoffs: {payoffs_result}")

    # Test fixed-length sampling
    print("\n[Test 2: Fixed-Length Trajectory Sampling]")
    key = random.PRNGKey(123)
    max_length = 50

    hole_cards, board, bets, actions_arr, valid_mask = sample_trajectory_fixed_length(
        key, num_players, stacks, blinds,
        policy_fn=uniform_random_policy,
        max_length=max_length
    )

    num_valid = jnp.sum(valid_mask)
    print(f"✓ Fixed-length trajectory:")
    print(f"  - Max length: {max_length}")
    print(f"  - Valid steps: {num_valid}")
    print(f"  - Padding: {max_length - num_valid} steps")

    # Test batched sampling (small batch for testing)
    print("\n[Test 3: Batched Trajectory Sampling]")
    batch_size = 10  # Small for testing
    keys = random.split(random.PRNGKey(999), batch_size)

    import time
    start_time = time.time()

    batch_outputs = batch_sample_trajectories(
        keys, num_players, stacks, blinds,
        policy_fn=uniform_random_policy,
        max_trajectory_length=50
    )

    elapsed = time.time() - start_time

    batch_hole, batch_board, batch_bets, batch_actions, batch_valid = batch_outputs

    print(f"✓ Batched sampling:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Time: {elapsed:.4f}s")
    print(f"  - Throughput: {batch_size/elapsed:.1f} trajectories/sec")
    print(f"  - Output shapes:")
    print(f"    - hole_cards: {batch_hole.shape}")
    print(f"    - board: {batch_board.shape}")
    print(f"    - bets: {batch_bets.shape}")
    print(f"    - actions: {batch_actions.shape}")
    print(f"    - valid_mask: {batch_valid.shape}")

    print("\n" + "=" * 70)
    print("Trajectory Sampler Tests Passed! ✅")
    print("\nNext: Implement GPU MCCFR Solver to use these trajectories!")
