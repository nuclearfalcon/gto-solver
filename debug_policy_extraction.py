"""
Debug policy extraction to see why averaged strategy is still uniform.
"""

import pyspiel
import numpy as np
from matrix_cfr import MatrixCFRSolver

game = pyspiel.load_game("kuhn_poker")
solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

print("Running 10 iterations...")
solver.solve(iterations=10, progress_interval=10)

print("\n" + "=" * 80)
print("DEBUGGING STRATEGY ACCUMULATION")
print("=" * 80 + "\n")

# Check cumulative values
print("Cumulative regrets:")
print(f"  Min: {float(solver.cumulative_regrets.min()):.6f}")
print(f"  Max: {float(solver.cumulative_regrets.max()):.6f}")
print(f"  Non-zero: {int((solver.cumulative_regrets != 0).sum())}/{len(solver.cumulative_regrets)}")

print("\nCumulative strategy:")
print(f"  Min: {float(solver.cumulative_strategy.min()):.6f}")
print(f"  Max: {float(solver.cumulative_strategy.max()):.6f}")
print(f"  Sum: {float(solver.cumulative_strategy.sum()):.6f}")
print(f"  Non-zero: {int((solver.cumulative_strategy != 0).sum())}/{len(solver.cumulative_strategy)}")

print("\nCumulative reach:")
print(f"  Min: {float(solver.cumulative_reach.min()):.6f}")
print(f"  Max: {float(solver.cumulative_reach.max()):.6f}")
print(f"  Sum: {float(solver.cumulative_reach.sum()):.6f}")
print(f"  Non-zero: {int((solver.cumulative_reach != 0).sum())}/{len(solver.cumulative_reach)}")

# Sample values per infoset
print("\n" + "=" * 80)
print("SAMPLE VALUES PER INFOSET")
print("=" * 80 + "\n")

for infoset, indices in list(solver.infoset_action_indices.items())[:5]:
    print(f"Infoset: {infoset}")

    regrets = solver.cumulative_regrets[indices]
    strategy = solver.cumulative_strategy[indices]
    reach = solver.cumulative_reach[indices]
    current = solver.current_strategy[indices]

    print(f"  Cumulative regrets: {regrets}")
    print(f"  Cumulative strategy: {strategy}")
    print(f"  Cumulative reach: {reach}")
    print(f"  Current strategy: {current}")

    # Compute average as done in get_average_policy
    epsilon = 1e-10
    avg = strategy / (reach + epsilon)
    avg_normalized = avg / (avg.sum() + epsilon)
    print(f"  Average (strategy/reach): {avg}")
    print(f"  Average (normalized): {avg_normalized}")
    print()

print("=" * 80)
