#!/usr/bin/env python3
"""
Quick test: Sparse vs Dense on Kuhn Poker

Verify that sparse and dense produce identical results.
"""

import pyspiel
import numpy as np
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver

def main():
    print("=" * 70)
    print("QUICK TEST: Sparse vs Dense on Kuhn Poker")
    print("=" * 70)
    print()

    game = pyspiel.load_game("kuhn_poker")

    # Dense solver (Phase 1-4)
    print("Creating dense solver...")
    solver_dense = MatrixCFRSolver(game, use_sparse=False)
    print("Running 50 iterations (dense)...")
    solver_dense.solve(iterations=50, progress_interval=50)
    policy_dense = solver_dense.get_average_policy()
    print("Dense complete!")
    print()

    # Sparse solver (Phase 5)
    print("Creating sparse solver...")
    solver_sparse = MatrixCFRSolver(game, use_sparse=True)
    print("Running 50 iterations (sparse)...")
    solver_sparse.solve(iterations=50, progress_interval=50)
    policy_sparse = solver_sparse.get_average_policy()
    print("Sparse complete!")
    print()

    # Compare policies
    print("Comparing policies...")
    max_diff = 0.0
    for infoset in policy_dense:
        diff = np.max(np.abs(np.array(policy_dense[infoset]) - np.array(policy_sparse[infoset])))
        max_diff = max(max_diff, diff)
        if diff > 0.001:
            print(f"  {infoset}: dense={policy_dense[infoset]}, sparse={policy_sparse[infoset]}, diff={diff}")

    print()
    print(f"Maximum difference: {max_diff}")
    print()

    if max_diff < 0.01:
        print("✅ SUCCESS! Sparse and dense produce identical results!")
        print()
        print("Ready for Leduc test!")
        return 0
    else:
        print("❌ FAILURE! Sparse and dense don't match!")
        return 1


if __name__ == "__main__":
    exit(main())
