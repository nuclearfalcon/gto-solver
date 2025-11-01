"""
Debug strategy accumulation step by step.
"""

import pyspiel
import numpy as np
from matrix_cfr import MatrixCFRSolver

game = pyspiel.load_game("kuhn_poker")
solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

# Build opponent strategy and reach probs
opponent_strategy = solver._build_node_strategy_vector()
reach_probs = solver._top_down_reach_probabilities(updating_player=0, opponent_strategy=opponent_strategy)

print("Debugging _update_cumulative_strategy():")
print("=" * 80)

# Manually iterate through the accumulation logic with debugging
for infoset, action_indices in list(solver.infoset_action_indices.items())[:5]:
    print(f"\nInfoset: {infoset}")
    print(f"  Action indices: {action_indices}")

    # Get representative node
    first_action = solver.matrix_repr.infoset_to_actions[infoset][0]
    if (infoset, first_action) not in solver.matrix_repr.action_index_to_node:
        print(f"  → SKIPPED: (infoset, action) not in mapping")
        continue

    node_id = solver.matrix_repr.action_index_to_node[(infoset, first_action)]
    node = solver.matrix_repr.nodes[node_id]

    print(f"  Node ID: {node_id}")
    print(f"  Node depth: {node.depth}")
    print(f"  Node player: {node.player}")

    # Get reach weight
    reach_weight = reach_probs[node.depth][node_id]
    print(f"  Reach weight: {float(reach_weight):.6f}")

    # Get current strategy
    current_strat = solver.current_strategy[action_indices]
    print(f"  Current strategy: {current_strat}")

    # Compute weighted strategy
    weighted_strategy = current_strat * reach_weight
    print(f"  Weighted strategy: {weighted_strategy}")

    if float(reach_weight) == 0.0:
        print(f"  ⚠️  ZERO REACH WEIGHT! This node is not reached.")

print("\n" + "=" * 80)
