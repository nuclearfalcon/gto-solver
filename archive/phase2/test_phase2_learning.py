"""
Test that Phase 2 optimizations preserve learning correctness.

IMPORTANT: Activate OpenSpiel virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import pyspiel
from matrix_cfr import MatrixCFRSolver


def test_learning():
    """Test that the solver still learns with Phase 2 optimizations."""
    print("=" * 70)
    print("Testing Phase 2: Learning Validation")
    print("=" * 70)

    # Create Kuhn poker game
    game = pyspiel.load_game('kuhn_poker', {'players': 2})

    # Create solver
    print("\nInitializing solver with Phase 2 optimizations...")
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr')

    # Run iterations
    num_iterations = 100
    print(f"\nRunning {num_iterations} CFR iterations...")
    solver.solve(iterations=num_iterations)

    # Get final strategy
    policy_dict = solver.get_average_policy()

    # Count non-uniform strategies
    non_uniform_count = 0
    total_infosets = 0

    print("\n" + "=" * 70)
    print("Final Strategy Analysis")
    print("=" * 70)

    for infoset, strategy in sorted(policy_dict.items()):
        total_infosets += 1

        # Check if non-uniform (not all equal)
        is_uniform = all(abs(strategy[0] - p) < 0.01 for p in strategy)

        if not is_uniform:
            non_uniform_count += 1
            print(f"✓ {infoset:4s}: {[f'{p:.3f}' for p in strategy]} - Non-uniform")
        else:
            print(f"  {infoset:4s}: {[f'{p:.3f}' for p in strategy]} - Uniform")

    print("\n" + "=" * 70)
    print(f"Learning Results: {non_uniform_count}/{total_infosets} infosets learned non-uniform strategies")
    print("=" * 70)

    if non_uniform_count >= 3:
        print(f"\n✅ SUCCESS! Learning is occurring ({non_uniform_count}/12 = {non_uniform_count/12*100:.0f}%)")
        print("   Phase 2 optimizations preserve correctness!")
        return True
    else:
        print(f"\n❌ FAIL! Only {non_uniform_count}/12 infosets learning")
        print("   Phase 2 may have introduced bugs")
        return False


if __name__ == '__main__':
    import sys
    if not test_learning():
        sys.exit(1)

    print("\n✅ Phase 2 learning validation PASSED!")
