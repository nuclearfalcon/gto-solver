"""
Phase 3.4 Ablation Test - Robust Statistical Analysis

Runs MANY benchmark iterations to determine if Phase 3.4 actually helps or hurts.
Uses proper statistical testing to account for system noise.

IMPORTANT: Activate OpenSpiel virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import pyspiel
import time
import numpy as np
from matrix_cfr import MatrixCFRSolver
import jax.numpy as jnp
from scipy import stats


def benchmark_implementation(implementation_name, update_method, num_runs=10, iterations=100):
    """
    Benchmark a specific implementation with many runs.

    Args:
        implementation_name: Name for display
        update_method: Function to use for _update_regrets_and_strategy
        num_runs: Number of benchmark runs
        iterations: Iterations per run

    Returns:
        Array of speeds (it/s) for each run
    """
    print(f"\n{'='*70}")
    print(f"Testing {implementation_name}")
    print(f"{'='*70}")
    print(f"  Runs: {num_runs}")
    print(f"  Iterations per run: {iterations}")

    game = pyspiel.load_game('kuhn_poker', {'players': 2})

    # Patch the method
    from matrix_cfr import matrix_cfr_solver
    original_method = matrix_cfr_solver.MatrixCFRSolver._update_regrets_and_strategy
    matrix_cfr_solver.MatrixCFRSolver._update_regrets_and_strategy = update_method

    # Warmup
    print("  Warming up JIT...")
    solver = MatrixCFRSolver(game)
    solver.solve(iterations=10)

    # Benchmark runs
    print(f"  Running {num_runs} trials...")
    speeds = []

    for run in range(num_runs):
        solver = MatrixCFRSolver(game)

        start = time.time()
        solver.solve(iterations=iterations)
        elapsed = time.time() - start

        speed = iterations / elapsed
        speeds.append(speed)

        if (run + 1) % 5 == 0:
            print(f"    Completed {run+1}/{num_runs} runs")

    # Statistics
    speeds_array = np.array(speeds)
    mean = np.mean(speeds_array)
    std = np.std(speeds_array)
    sem = stats.sem(speeds_array)  # Standard error of mean

    print(f"\n  Results:")
    print(f"    Mean: {mean:.3f} it/s")
    print(f"    Std:  {std:.3f} it/s")
    print(f"    SEM:  {sem:.3f} it/s")
    print(f"    Min:  {np.min(speeds_array):.3f} it/s")
    print(f"    Max:  {np.max(speeds_array):.3f} it/s")

    # Restore original method
    matrix_cfr_solver.MatrixCFRSolver._update_regrets_and_strategy = original_method

    return speeds_array


def vectorized_update(self, player, cf_values):
    """Phase 3.4 implementation - vectorized regret updates."""
    if len(cf_values) == 0:
        self.current_strategy = self._regret_matching()
        current_node_strategy = self._build_node_strategy_vector()
        reach_probs = self._full_reach_probabilities(current_node_strategy)
        self._update_cumulative_strategy(reach_probs)
        return

    # Build full regret array
    instant_regrets_full = jnp.zeros_like(self.cumulative_regrets)

    for infoset, action_values in cf_values.items():
        action_indices = self.infoset_action_indices[infoset]
        current_probs = self.current_strategy[action_indices]
        strategy_value = jnp.sum(current_probs * action_values)
        instant_regrets = action_values - strategy_value
        instant_regrets_full = instant_regrets_full.at[action_indices].set(instant_regrets)

    # Single vectorized update
    self.cumulative_regrets = self.cumulative_regrets + instant_regrets_full

    self.current_strategy = self._regret_matching()
    current_node_strategy = self._build_node_strategy_vector()
    reach_probs = self._full_reach_probabilities(current_node_strategy)
    self._update_cumulative_strategy(reach_probs)


def sequential_update(self, player, cf_values):
    """Original implementation - sequential regret updates."""
    for infoset, action_values in cf_values.items():
        action_indices = self.infoset_action_indices[infoset]
        current_probs = self.current_strategy[action_indices]
        strategy_value = jnp.sum(current_probs * action_values)
        instant_regrets = action_values - strategy_value

        # Sequential updates
        for i, action_idx in enumerate(action_indices):
            self.cumulative_regrets = self.cumulative_regrets.at[action_idx].add(
                instant_regrets[i]
            )

    self.current_strategy = self._regret_matching()
    current_node_strategy = self._build_node_strategy_vector()
    reach_probs = self._full_reach_probabilities(current_node_strategy)
    self._update_cumulative_strategy(reach_probs)


if __name__ == '__main__':
    print("="*70)
    print("PHASE 3.4 ABLATION STUDY - ROBUST STATISTICAL ANALYSIS")
    print("="*70)
    print("\nThis test runs MANY iterations to account for system noise")
    print("and provides statistical significance testing.")

    num_runs = 15  # More runs for better statistics
    iterations = 100

    # Test both implementations
    print("\n" + "="*70)
    print("ROUND 1: Testing WITH Phase 3.4")
    print("="*70)
    with_speeds = benchmark_implementation(
        "WITH Phase 3.4 (Vectorized)",
        vectorized_update,
        num_runs=num_runs,
        iterations=iterations
    )

    print("\n" + "="*70)
    print("ROUND 2: Testing WITHOUT Phase 3.4")
    print("="*70)
    without_speeds = benchmark_implementation(
        "WITHOUT Phase 3.4 (Sequential)",
        sequential_update,
        num_runs=num_runs,
        iterations=iterations
    )

    # Statistical comparison
    print("\n" + "="*70)
    print("📊 STATISTICAL COMPARISON")
    print("="*70)

    with_mean = np.mean(with_speeds)
    without_mean = np.mean(without_speeds)

    print(f"\n  WITH Phase 3.4:    {with_mean:.3f} ± {stats.sem(with_speeds):.3f} it/s")
    print(f"  WITHOUT Phase 3.4: {without_mean:.3f} ± {stats.sem(without_speeds):.3f} it/s")

    # Paired t-test (tests if difference is significant)
    t_stat, p_value = stats.ttest_ind(with_speeds, without_speeds)

    print(f"\n  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  ✓ Difference is statistically significant (p < 0.05)")
    else:
        print(f"  ✗ Difference is NOT statistically significant (p >= 0.05)")
        print(f"    → Performance is effectively the same")

    # Effect size
    diff_pct = (with_mean / without_mean - 1) * 100

    print("\n" + "="*70)
    print("🎯 CONCLUSION")
    print("="*70)

    if p_value < 0.05:
        if with_mean > without_mean:
            print(f"  ✅ Phase 3.4 provides a SIGNIFICANT speedup: +{diff_pct:.1f}%")
            print(f"  💡 RECOMMENDATION: Keep Phase 3.4")
        else:
            print(f"  ❌ Phase 3.4 causes a SIGNIFICANT slowdown: {diff_pct:.1f}%")
            print(f"  💡 RECOMMENDATION: Revert Phase 3.4")
    else:
        print(f"  ⚠️  No significant performance difference (p={p_value:.4f})")
        print(f"  Observed difference: {diff_pct:+.1f}% (likely just noise)")
        if abs(diff_pct) < 2:
            print(f"  💡 RECOMMENDATION: Either implementation is fine")
            print(f"     Choose based on code simplicity (sequential is simpler)")
        else:
            print(f"  💡 RECOMMENDATION: Run more trials to get clearer answer")

    print("="*70)
