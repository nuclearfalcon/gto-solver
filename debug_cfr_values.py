"""
Debug script to trace what's happening in counterfactual value computation.

This will run a single iteration and print:
- Node structure
- Utilities by level
- Counterfactual values
- Regrets

To identify where the algorithm is failing.
"""

import pyspiel
import numpy as np
from matrix_cfr import MatrixCFRSolver

def debug_single_iteration():
    """Debug a single CFR iteration."""
    print("\n" + "=" * 80)
    print("DEBUGGING MATRIX CFR")
    print("=" * 80 + "\n")

    # Load Kuhn poker
    game = pyspiel.load_game("kuhn_poker")
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

    print(f"Game: {solver.matrix_repr.num_nodes} nodes, {solver.matrix_repr.num_infosets} infosets")
    print(f"Levels: {len(solver.level_matrices_jax)}")
    print()

    # Print node structure
    print("Node structure:")
    print("-" * 80)
    for i, node in enumerate(solver.matrix_repr.nodes[:10]):  # First 10 nodes
        print(f"Node {i}: depth={node.depth}, player={node.player}, "
              f"terminal={node.is_terminal}, chance={node.is_chance}, "
              f"infoset={node.infoset}")
    print()

    # Test bottom-up utilities
    print("Testing bottom-up utilities for player 0:")
    print("-" * 80)
    utilities = solver._bottom_up_utilities(player=0)
    print(f"Number of levels: {len(utilities)}")
    for level, u in enumerate(utilities):
        print(f"Level {level}: shape={u.shape}, min={float(u.min()):.4f}, "
              f"max={float(u.max()):.4f}, mean={float(u.mean()):.4f}")
        # Show first few values
        print(f"  First 5 values: {u[:5]}")
    print()

    # Test counterfactual values
    print("Testing counterfactual values for player 0:")
    print("-" * 80)
    cf_values = solver._compute_counterfactual_values(player=0)
    print(f"Number of infosets for player 0: {len(cf_values)}")
    for infoset, values in list(cf_values.items())[:5]:  # First 5
        print(f"Infoset: {infoset}")
        print(f"  Action values: {values}")
        print(f"  All same? {np.allclose(values, values[0])}")
    print()

    # Check regrets after one iteration
    print("Running single CFR iteration...")
    print("-" * 80)
    solver._cfr_iteration(player=0)

    print(f"Cumulative regrets shape: {solver.cumulative_regrets.shape}")
    print(f"Regrets min: {float(solver.cumulative_regrets.min()):.6f}")
    print(f"Regrets max: {float(solver.cumulative_regrets.max()):.6f}")
    print(f"Regrets mean: {float(solver.cumulative_regrets.mean()):.6f}")
    print(f"Non-zero regrets: {int((solver.cumulative_regrets != 0).sum())}/{len(solver.cumulative_regrets)}")
    print()

    # Show some actual regret values
    print("Sample regrets:")
    for infoset, indices in list(solver.infoset_action_indices.items())[:5]:
        regrets = solver.cumulative_regrets[indices]
        print(f"{infoset}: {regrets}")
    print()

    print("=" * 80)

if __name__ == "__main__":
    try:
        debug_single_iteration()
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
