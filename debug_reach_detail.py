"""
Debug reach probabilities in detail.
"""

import pyspiel
import numpy as np
from matrix_cfr import MatrixCFRSolver

game = pyspiel.load_game("kuhn_poker")
solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

opponent_strategy = solver._build_node_strategy_vector()
reach_probs = solver._top_down_reach_probabilities(updating_player=0, opponent_strategy=opponent_strategy)

print("Reach probabilities at depth 2:")
print("=" * 80)

depth_2_reach = reach_probs[2]
print(f"Reach at depth 2: {depth_2_reach}")
print(f"Non-zero count: {int((depth_2_reach > 1e-6).sum())}")
print(f"Non-zero indices: {np.where(np.array(depth_2_reach) > 1e-6)[0]}")

print("\nNodes at depth 2:")
for node in solver.matrix_repr.nodes:
    if node.depth == 2:
        reach_val = float(depth_2_reach[node.node_id])
        print(f"  Node {node.node_id}: player={node.player}, infoset={node.infoset}, reach={reach_val:.6f}")

print("\n" + "=" * 80)
print("Checking counterfactual strategy override:")
print("=" * 80)

player_matrix = np.array(solver.player_matrix_jax)
print(f"Player matrix shape: {player_matrix.shape}")
print(f"Player 0 nodes (first 20): {player_matrix[:20, 0]}")

# Check if player 0 nodes are being correctly identified
print("\nPlayer 0 decision nodes:")
for node in solver.matrix_repr.nodes[:20]:
    if node.player == 0:
        is_player_0 = player_matrix[node.node_id, 0]
        print(f"  Node {node.node_id}: depth={node.depth}, player_matrix[{node.node_id}, 0]={is_player_0}")
