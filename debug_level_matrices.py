"""
Debug level matrices to understand their structure.
"""

import pyspiel
import numpy as np
from matrix_cfr import MatrixCFRSolver

game = pyspiel.load_game("kuhn_poker")
solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

print("Level matrices analysis:")
print("=" * 80)

for level, L_l in enumerate(solver.level_matrices_jax):
    print(f"\nLevel {level} matrix: shape {L_l.shape}")

    # Convert to numpy for easier analysis
    L_np = np.array(L_l)

    # Count edges
    num_edges = int((L_np > 0.5).sum())
    print(f"  Total edges: {num_edges}")

    # For node 2 specifically
    if level == 2:
        print(f"\n  Node 2 (decision node at depth 2):")
        print(f"    Row 2 (outgoing edges): {L_np[2, :]}")
        print(f"    Children: {np.where(L_np[2, :] > 0.5)[0]}")

        # Also check if children are in the NEXT level matrix
        if level + 1 < len(solver.level_matrices_jax):
            L_next = np.array(solver.level_matrices_jax[level + 1])
            print(f"\n  Checking level {level + 1} (should have edges FROM node 2):")
            print(f"    Row 2: {L_next[2, :]}")
            print(f"    Children of node 2: {np.where(L_next[2, :] > 0.5)[0]}")

print("\n" + "=" * 80)
print("Node 2 details:")
node_2 = solver.matrix_repr.nodes[2]
print(f"  Depth: {node_2.depth}")
print(f"  Player: {node_2.player}")
print(f"  Infoset: {node_2.infoset}")
print(f"  Legal actions: {node_2.legal_actions}")
print(f"  Is terminal: {node_2.is_terminal}")

# Find all nodes at depth 3 (should be children of depth-2 nodes)
print("\nNodes at depth 3:")
for node in solver.matrix_repr.nodes:
    if node.depth == 3 and node.parent_id == 2:
        print(f"  Node {node.node_id}: parent={node.parent_id}, player={node.player}")
