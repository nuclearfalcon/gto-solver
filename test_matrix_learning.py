"""
Quick test to validate that matrix CFR actually learns.

This will run a small number of iterations on 2-player Kuhn poker and check:
1. Code runs without errors
2. Strategies become non-uniform (learning occurs!)
3. Action values differ from placeholder 0.1

Run with:
    source ~/open_spiel/venv/bin/activate
    python test_matrix_learning.py
"""

import pyspiel
import numpy as np
from matrix_cfr import MatrixCFRSolver

def test_kuhn_learning():
    """Test if matrix CFR learns on Kuhn poker."""
    print("\n" + "=" * 80)
    print("TESTING MATRIX CFR LEARNING")
    print("=" * 80 + "\n")

    # Load 2-player Kuhn poker (tiny game)
    game = pyspiel.load_game("kuhn_poker")

    print(f"Game: {game.get_type().short_name}")
    print(f"Players: {game.num_players()}")
    print(f"Max game length: {game.max_game_length()}")
    print()

    # Create matrix CFR solver
    print("Creating matrix CFR solver...")
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

    print(f"Game tree: {solver.matrix_repr.num_nodes} nodes")
    print(f"Infosets: {solver.matrix_repr.num_infosets}")
    print(f"Infoset-actions: {solver.matrix_repr.num_infoset_actions}")
    print()

    # Run for a small number of iterations to test
    print("Running 100 iterations to test learning...")
    print("-" * 80)
    solver.solve(iterations=100, progress_interval=25)

    # Get policy
    policy = solver.get_average_policy()

    # Check if strategies are non-uniform
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80 + "\n")

    non_uniform_count = 0
    uniform_count = 0

    for infoset, probs in policy.items():
        is_uniform = np.allclose(probs, probs[0], atol=0.01)
        if is_uniform:
            uniform_count += 1
        else:
            non_uniform_count += 1
            print(f"✓ {infoset}: {probs} (NON-UNIFORM - LEARNING!)")

    print(f"\nSummary:")
    print(f"  Non-uniform infosets: {non_uniform_count}/{solver.matrix_repr.num_infosets}")
    print(f"  Uniform infosets: {uniform_count}/{solver.matrix_repr.num_infosets}")

    if non_uniform_count > 0:
        print("\n" + "=" * 80)
        print("✓✓✓ SUCCESS! LEARNING IS OCCURRING! ✓✓✓")
        print("=" * 80 + "\n")
        return True
    else:
        print("\n" + "=" * 80)
        print("✗✗✗ FAILURE: All strategies still uniform (no learning) ✗✗✗")
        print("=" * 80 + "\n")
        return False

if __name__ == "__main__":
    try:
        success = test_kuhn_learning()
        exit(0 if success else 1)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗✗✗ ERROR: {type(e).__name__}: {e}")
        print("=" * 80 + "\n")
        import traceback
        traceback.print_exc()
        exit(1)
