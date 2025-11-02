"""
Phase 3.4 Ablation Test - Does Vectorized Regret Update Help or Hurt?

Tests performance with and without Phase 3.4 optimization to determine
if it actually provides a speedup or causes a slowdown.

IMPORTANT: Activate OpenSpiel virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import pyspiel
import time
import numpy as np
from matrix_cfr import MatrixCFRSolver
import jax.numpy as jnp


def benchmark_current_implementation(iterations=100):
    """Benchmark WITH Phase 3.4 (vectorized regret updates)."""
    print("\n" + "="*70)
    print("Testing WITH Phase 3.4 (Vectorized Regret Updates)")
    print("="*70)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})

    # Warmup
    solver = MatrixCFRSolver(game)
    solver.solve(iterations=10)

    # Benchmark
    times = []
    for run in range(3):
        solver = MatrixCFRSolver(game)
        start = time.time()
        solver.solve(iterations=iterations)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {iterations/elapsed:.2f} it/s ({elapsed:.2f}s)")

    mean_time = np.mean(times)
    mean_speed = iterations / mean_time
    print(f"  Average: {mean_speed:.2f} it/s")

    return mean_speed


def benchmark_without_phase34(iterations=100):
    """Benchmark WITHOUT Phase 3.4 (sequential regret updates)."""
    print("\n" + "="*70)
    print("Testing WITHOUT Phase 3.4 (Sequential Regret Updates)")
    print("="*70)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})

    # Temporarily patch the method to use sequential updates
    from matrix_cfr import matrix_cfr_solver

    original_method = matrix_cfr_solver.MatrixCFRSolver._update_regrets_and_strategy

    def sequential_update(self, player, cf_values):
        """OLD implementation - sequential updates (pre Phase 3.4)."""
        for infoset, action_values in cf_values.items():
            action_indices = self.infoset_action_indices[infoset]
            current_probs = self.current_strategy[action_indices]

            # Compute strategy value (weighted average)
            strategy_value = jnp.sum(current_probs * action_values)

            # Compute instant regrets
            instant_regrets = action_values - strategy_value

            # SEQUENTIAL UPDATE (one .add() per action)
            for idx, action_idx in enumerate(action_indices):
                self.cumulative_regrets = self.cumulative_regrets.at[action_idx].add(
                    instant_regrets[idx]
                )

        # Update strategy
        self.current_strategy = self._regret_matching()

        # Update average strategy
        reach_weight = 1.0
        self.cumulative_strategy = self.cumulative_strategy.at[:].add(
            reach_weight * self.current_strategy
        )

    # Patch the method
    matrix_cfr_solver.MatrixCFRSolver._update_regrets_and_strategy = sequential_update

    # Warmup
    solver = MatrixCFRSolver(game)
    solver.solve(iterations=10)

    # Benchmark
    times = []
    for run in range(3):
        solver = MatrixCFRSolver(game)
        start = time.time()
        solver.solve(iterations=iterations)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {iterations/elapsed:.2f} it/s ({elapsed:.2f}s)")

    mean_time = np.mean(times)
    mean_speed = iterations / mean_time
    print(f"  Average: {mean_speed:.2f} it/s")

    # Restore original method
    matrix_cfr_solver.MatrixCFRSolver._update_regrets_and_strategy = original_method

    return mean_speed


if __name__ == '__main__':
    print("="*70)
    print("PHASE 3.4 ABLATION STUDY")
    print("="*70)
    print("\nTesting whether Phase 3.4 (vectorized regret updates)")
    print("actually improves performance or causes a slowdown.")

    iterations = 100

    # Test WITH Phase 3.4
    with_phase34 = benchmark_current_implementation(iterations)

    # Test WITHOUT Phase 3.4
    without_phase34 = benchmark_without_phase34(iterations)

    # Results
    print("\n" + "="*70)
    print("📊 ABLATION RESULTS")
    print("="*70)
    print(f"  WITH Phase 3.4 (vectorized):   {with_phase34:.2f} it/s")
    print(f"  WITHOUT Phase 3.4 (sequential): {without_phase34:.2f} it/s")
    print()

    if with_phase34 > without_phase34:
        improvement = (with_phase34 / without_phase34 - 1) * 100
        print(f"  ✅ Phase 3.4 HELPS: +{improvement:.1f}% speedup")
    else:
        regression = (1 - with_phase34 / without_phase34) * 100
        print(f"  ❌ Phase 3.4 HURTS: -{regression:.1f}% slowdown")
        print()
        print("  💡 RECOMMENDATION: Revert Phase 3.4 optimization")

    print("="*70)
