"""
Debug reach probability computation.
"""

import pyspiel
import numpy as np
from matrix_cfr import MatrixCFRSolver

game = pyspiel.load_game("kuhn_poker")
solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

print("Testing reach probability computation:")
print("=" * 80)

opponent_strategy = solver._build_node_strategy_vector()
print(f"\nOpponent strategy vector:")
print(f"  Shape: {opponent_strategy.shape}")
print(f"  Min: {float(opponent_strategy.min()):.4f}")
print(f"  Max: {float(opponent_strategy.max()):.4f}")
print(f"  First 10: {opponent_strategy[:10]}")

reach_probs = solver._top_down_reach_probabilities(updating_player=0, opponent_strategy=opponent_strategy)
print(f"\nReach probabilities (player 0):")
print(f"  Num levels: {len(reach_probs)}")
for level, reach in enumerate(reach_probs):
    print(f"  Level {level}: min={float(reach.min()):.4f}, max={float(reach.max()):.4f}, "
          f"sum={float(reach.sum()):.4f}, nonzero={int((reach > 1e-6).sum())}")
    if level <= 3:
        print(f"    First 10: {reach[:10]}")

print("\n" + "=" * 80)
print("Testing strategy accumulation manually:")
print("=" * 80)

# Manually call the update
print("\nBefore accumulation:")
print(f"  Cumulative strategy sum: {float(solver.cumulative_strategy.sum())}")
print(f"  Cumulative reach sum: {float(solver.cumulative_reach.sum())}")

solver._update_cumulative_strategy(reach_probs)

print("\nAfter accumulation:")
print(f"  Cumulative strategy sum: {float(solver.cumulative_strategy.sum())}")
print(f"  Cumulative reach sum: {float(solver.cumulative_reach.sum())}")

# Check a specific infoset
infoset = "0"
if infoset in solver.infoset_action_indices:
    indices = solver.infoset_action_indices[infoset]
    print(f"\nInfoset '{infoset}':")
    print(f"  Indices: {indices}")
    print(f"  Cumulative strategy: {solver.cumulative_strategy[indices]}")
    print(f"  Cumulative reach: {solver.cumulative_reach[indices]}")
