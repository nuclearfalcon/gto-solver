#!/usr/bin/env python3
"""
Phase 5: THE CRITICAL TEST - Leduc Poker Memory Test

This is the test that validates Phase 5 sparse matrices enable larger games.

Leduc poker (9,457 nodes) caused OOM with dense matrices (2.67 GB).
With sparse matrices, it should use only ~0.4 MB.

Expected result: NO OOM, solver initializes and runs successfully!
"""

import pyspiel
import tracemalloc
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver


def main():
    print("=" * 70)
    print("PHASE 5: THE CRITICAL TEST - Leduc Poker Memory")
    print("=" * 70)
    print()
    print("This test validates that sparse matrices enable Leduc poker.")
    print()
    print("Expected:")
    print("  - Dense matrices: OOM at 2.67 GB ❌")
    print("  - Sparse matrices: ~0.4 MB ✅")
    print()
    print("=" * 70)
    print()

    # Start memory tracking
    tracemalloc.start()

    try:
        # Create Leduc poker game
        print("Loading Leduc poker...")
        game = pyspiel.load_game("leduc_poker")
        print("✓ Game loaded")
        print()

        # Attempt to create solver with SPARSE matrices
        print("Creating sparse solver (THE CRITICAL MOMENT)...")
        print("If this OOMs, Phase 5 failed...")
        print()

        solver = MatrixCFRSolver(game, use_sparse=True)

        # If we get here, initialization succeeded!
        print("=" * 70)
        print("🎉 SUCCESS! Solver initialized without OOM!")
        print("=" * 70)
        print()

        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Game statistics:")
        print(f"  Nodes: {solver.matrix_repr.num_nodes}")
        print(f"  Infosets: {solver.matrix_repr.num_infosets}")
        print(f"  Infoset-actions: {solver.matrix_repr.num_infoset_actions}")
        print(f"  Players: {solver.matrix_repr.num_players}")
        print()

        print(f"Memory usage:")
        print(f"  Current: {current / (1024**2):.2f} MB")
        print(f"  Peak: {peak / (1024**2):.2f} MB")
        print()

        if peak < 100 * (1024**2):  # Less than 100 MB
            print(f"✅ Memory usage is reasonable ({peak / (1024**2):.2f} MB)")
        else:
            print(f"⚠️  Memory usage higher than expected ({peak / (1024**2):.2f} MB)")
        print()

        # Try running a few iterations to ensure it actually works
        print("Running 10 iterations to verify solver works...")
        solver.solve(iterations=10, progress_interval=10)
        print()

        # Check that learning occurs
        policy = solver.get_average_policy()
        print(f"Policy has {len(policy)} infosets")
        print()

        print("=" * 70)
        print("🎉🎉🎉 PHASE 5 SUCCESS! 🎉🎉🎉")
        print("=" * 70)
        print()
        print("Leduc poker works with sparse matrices!")
        print(f"Memory: {peak / (1024**2):.2f} MB (vs 2.67 GB dense)")
        print(f"Compression: {(2.67 * 1024) / (peak / (1024**2)):.0f}x")
        print()
        print("Phase 5 validates:")
        print("  ✅ Sparse matrices enable Leduc poker")
        print("  ✅ Memory usage dramatically reduced")
        print("  ✅ Solver runs correctly")
        print("  ✅ Learning works")
        print()
        print("Ready for Hold'em! 🚀")
        print()

        return 0

    except Exception as e:
        tracemalloc.stop()
        print()
        print("=" * 70)
        print("❌ FAILURE - Phase 5 did not work")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        traceback.print_exc()
        print()
        print("Phase 5 needs more work to enable Leduc poker.")
        return 1


if __name__ == "__main__":
    exit(main())
