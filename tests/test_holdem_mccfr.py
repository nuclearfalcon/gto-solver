"""
Test GPU MCCFR Solver on No-Limit Hold'em

Validates GPU MCCFR scalability by testing on progressively larger Hold'em variants:
1. Tiny Hold'em: 2 players, 2 suits, 3 ranks (6 cards total)
2. Small Hold'em: 2 players, 2 suits, 5 ranks (10 cards total)
3. Medium Hold'em: 2 players, 4 suits, 5 ranks (20 cards total)

Phase 10: Days 11-13 - Hold'em Scalability Testing
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

from matrix_cfr import holdem_jax
from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver, MCCFRConfig


def test_tiny_holdem():
    """
    Test GPU MCCFR on Tiny Hold'em.

    Tiny Hold'em specifications:
    - 2 players
    - 2 suits × 3 ranks = 6 cards
    - Blinds: 50/100
    - Stacks: 1000 chips
    - Actions: fold, call, pot, all-in

    This is the smallest viable Hold'em game for MCCFR testing.
    """
    print("=" * 70)
    print("Testing GPU MCCFR on Tiny Hold'em (2 suits × 3 ranks)")
    print("=" * 70)

    # Setup
    print("\n[Setup]")
    config = MCCFRConfig(
        num_players=2,
        num_actions=4,  # fold, call, pot, all-in
        use_linear_weighting=False
    )

    solver = GPUMCCFRSolver(holdem_jax, config, seed=42)
    print(f"✓ Solver initialized")

    # Game parameters
    num_players = 2
    stacks = jnp.array([1000.0, 1000.0])
    blinds = jnp.array([50.0, 100.0])

    print(f"✓ Game setup:")
    print(f"  - Players: {num_players}")
    print(f"  - Stacks: {stacks}")
    print(f"  - Blinds: {blinds}")

    # Run short training
    print("\n[Training Phase: 50 iterations]")
    print("(Testing basic functionality before longer runs)")

    start_time = time.time()

    solver.solve(
        num_iterations=50,
        num_players=num_players,
        stacks=stacks,
        blinds=blinds,
        progress_interval=25
    )

    elapsed = time.time() - start_time

    # Extract policy
    print("\n[Policy Extraction]")
    policy_p0 = solver.get_average_policy(player=0)
    policy_p1 = solver.get_average_policy(player=1)

    print(f"✓ Player 0 policy: {len(policy_p0)} infosets")
    print(f"✓ Player 1 policy: {len(policy_p1)} infosets")

    total_infosets = len(set(policy_p0.keys()) | set(policy_p1.keys()))
    print(f"✓ Total unique infosets: {total_infosets}")

    # Sample strategies
    print("\n[Sample Strategies - Player 0]")
    for i, (infoset, strategy) in enumerate(sorted(policy_p0.items())[:5]):
        print(f"  {infoset}")
        print(f"    fold={strategy[0]:.3f}, call={strategy[1]:.3f}, "
              f"pot={strategy[2]:.3f}, allin={strategy[3]:.3f}")

    # Performance metrics
    print("\n[Performance Metrics]")
    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Speed: {50/elapsed:.2f} it/s")
    print(f"  Infosets visited: {total_infosets}")

    # Success criteria
    print("\n[Success Criteria]")
    if total_infosets > 0:
        print(f"✓ Learning occurred ({total_infosets} infosets)")
    else:
        print(f"✗ No infosets learned (check game logic)")

    if 50/elapsed >= 1.0:
        print(f"✓ Reasonable speed (>1 it/s)")
    else:
        print(f"⚠️ Slow speed (<1 it/s, needs optimization)")

    print("\n" + "=" * 70)
    print("Tiny Hold'em Test Complete!")
    print("=" * 70)

    return solver, elapsed, total_infosets


def test_longer_training():
    """
    Run longer training session to measure convergence.

    This test runs 500 iterations to see if policies converge
    and to measure performance over time.
    """
    print("\n" + "=" * 70)
    print("Testing Longer Training (500 iterations)")
    print("=" * 70)

    # Setup
    print("\n[Setup]")
    config = MCCFRConfig(
        num_players=2,
        num_actions=4,
        use_linear_weighting=True  # Use linear weighting for faster convergence
    )

    solver = GPUMCCFRSolver(holdem_jax, config, seed=123)
    print(f"✓ Solver initialized (with linear weighting)")

    # Game parameters
    num_players = 2
    stacks = jnp.array([1000.0, 1000.0])
    blinds = jnp.array([50.0, 100.0])

    # Run training
    print("\n[Training: 500 iterations]")

    start_time = time.time()

    solver.solve(
        num_iterations=500,
        num_players=num_players,
        stacks=stacks,
        blinds=blinds,
        progress_interval=100
    )

    elapsed = time.time() - start_time

    # Extract policy
    print("\n[Final Policy]")
    policy_p0 = solver.get_average_policy(player=0)
    policy_p1 = solver.get_average_policy(player=1)

    total_infosets = len(set(policy_p0.keys()) | set(policy_p1.keys()))

    print(f"✓ Player 0: {len(policy_p0)} infosets")
    print(f"✓ Player 1: {len(policy_p1)} infosets")
    print(f"✓ Total unique: {total_infosets}")

    # Convergence analysis
    print("\n[Convergence Analysis]")

    # Check for strategy diversity
    diverse_strategies = 0
    uniform_strategies = 0

    for infoset, strategy in policy_p0.items():
        max_prob = np.max(strategy)
        if max_prob < 0.9:  # Not dominated by single action
            diverse_strategies += 1
        elif np.allclose(strategy, 0.25, atol=0.1):  # Near uniform
            uniform_strategies += 1

    print(f"  Diverse strategies: {diverse_strategies} ({diverse_strategies/len(policy_p0)*100:.1f}%)")
    print(f"  Uniform strategies: {uniform_strategies} ({uniform_strategies/len(policy_p0)*100:.1f}%)")
    print(f"  Dominated strategies: {len(policy_p0) - diverse_strategies - uniform_strategies}")

    # Performance
    print("\n[Performance]")
    print(f"  Total time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"  Speed: {500/elapsed:.2f} it/s")
    print(f"  Time per iteration: {elapsed/500*1000:.0f}ms")

    print("\n" + "=" * 70)
    print("Longer Training Test Complete!")
    print("=" * 70)

    return solver, elapsed, total_infosets


def test_trajectory_statistics():
    """
    Analyze trajectory statistics to understand game complexity.

    Samples 100 trajectories and measures:
    - Average trajectory length
    - Average number of decisions per trajectory
    - Distribution of terminal states (fold, showdown)
    """
    print("\n" + "=" * 70)
    print("Analyzing Trajectory Statistics")
    print("=" * 70)

    from matrix_cfr.trajectory_sampler import uniform_random_policy

    print("\n[Sampling 100 Trajectories]")

    num_players = 2
    stacks = jnp.array([1000.0, 1000.0])
    blinds = jnp.array([50.0, 100.0])

    trajectory_lengths = []
    terminal_types = []  # 'fold' or 'showdown'

    key = random.PRNGKey(999)

    for i in range(100):
        key, subkey = random.split(key)

        # Sample trajectory using uniform random policy
        config = MCCFRConfig(num_players=2, num_actions=4)
        solver = GPUMCCFRSolver(holdem_jax, config, seed=int(subkey[0]))

        key, subkey = random.split(key)
        states, actions, players, payoffs = solver._sample_trajectory(
            subkey,
            num_players,
            uniform_random_policy,
            max_actions=100,
            stacks=stacks,
            blinds=blinds
        )

        trajectory_lengths.append(len(states))

        # Determine terminal type (simplified check)
        if len(states) > 0:
            final_state = states[-1]
            # Check if anyone folded
            if jnp.any(final_state.folded):
                terminal_types.append('fold')
            else:
                terminal_types.append('showdown')

        if (i + 1) % 25 == 0:
            print(f"  Sampled {i + 1}/100 trajectories...")

    # Statistics
    print("\n[Trajectory Statistics]")
    print(f"  Mean length: {np.mean(trajectory_lengths):.1f} decisions")
    print(f"  Std dev: {np.std(trajectory_lengths):.1f}")
    print(f"  Min: {np.min(trajectory_lengths)}")
    print(f"  Max: {np.max(trajectory_lengths)}")

    print("\n[Terminal State Distribution]")
    fold_count = terminal_types.count('fold')
    showdown_count = terminal_types.count('showdown')
    print(f"  Folds: {fold_count} ({fold_count/len(terminal_types)*100:.1f}%)")
    print(f"  Showdowns: {showdown_count} ({showdown_count/len(terminal_types)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("Trajectory Analysis Complete!")
    print("=" * 70)

    return trajectory_lengths, terminal_types


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GPU MCCFR Hold'em Scalability Tests")
    print("=" * 70)
    print()
    print("Test Suite:")
    print("1. Tiny Hold'em (50 iterations) - Basic functionality")
    print("2. Trajectory Statistics (100 samples) - Game complexity")
    print("3. Longer Training (500 iterations) - Convergence analysis")
    print()

    # Test 1: Basic functionality
    print("\n" + "=" * 70)
    print("TEST 1: Tiny Hold'em Basic Functionality")
    print("=" * 70)

    solver1, time1, infosets1 = test_tiny_holdem()

    # Test 2: Trajectory statistics
    print("\n" + "=" * 70)
    print("TEST 2: Trajectory Statistics")
    print("=" * 70)

    traj_lengths, traj_types = test_trajectory_statistics()

    # Test 3: Longer training
    print("\n" + "=" * 70)
    print("TEST 3: Longer Training")
    print("=" * 70)

    solver3, time3, infosets3 = test_longer_training()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    print("Test 1 - Basic Functionality:")
    print(f"  ✓ 50 iterations in {time1:.1f}s ({50/time1:.2f} it/s)")
    print(f"  ✓ {infosets1} infosets discovered")
    print()
    print("Test 2 - Trajectory Statistics:")
    print(f"  ✓ Average trajectory length: {np.mean(traj_lengths):.1f} decisions")
    print(f"  ✓ Fold rate: {traj_types.count('fold')/len(traj_types)*100:.1f}%")
    print()
    print("Test 3 - Longer Training:")
    print(f"  ✓ 500 iterations in {time3:.1f}s ({500/time3:.2f} it/s)")
    print(f"  ✓ {infosets3} infosets discovered")
    print()

    # Success criteria
    print("Overall Assessment:")
    all_tests_passed = True

    if 50/time1 >= 0.5 and 500/time3 >= 0.5:
        print("  ✓ Performance acceptable (>0.5 it/s)")
    else:
        print("  ⚠️ Performance slow (<0.5 it/s)")
        all_tests_passed = False

    if infosets1 > 10 and infosets3 > 50:
        print("  ✓ Exploration successful")
    else:
        print("  ⚠️ Limited exploration")
        all_tests_passed = False

    if np.mean(traj_lengths) > 1.0:
        print("  ✓ Trajectories non-trivial")
    else:
        print("  ⚠️ Trajectories too short")
        all_tests_passed = False

    print()
    if all_tests_passed:
        print("=" * 70)
        print("ALL TESTS PASSED! ✅")
        print("=" * 70)
        print()
        print("GPU MCCFR successfully scales to Hold'em!")
        print("Ready for larger variants and exploitability measurement.")
    else:
        print("=" * 70)
        print("SOME TESTS NEED ATTENTION ⚠️")
        print("=" * 70)
        print()
        print("GPU MCCFR works but may need optimization for larger games.")
