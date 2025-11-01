#!/usr/bin/env python3
"""
Basic Test for Matrix CFR Solver

Simple smoke test to verify the matrix CFR solver runs without errors.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python tests/test_matrix_solver_basic.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspiel
import logging

from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_solver_initialization():
    """Test that solver initializes correctly."""
    print("\n" + "=" * 60)
    print("TEST: Matrix CFR Solver Initialization")
    print("=" * 60)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})

    print("\nCreating matrix CFR solver...")
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)

    print("\n✅ Solver initialized successfully!")
    print(f"  Nodes: {solver.matrix_repr.num_nodes}")
    print(f"  Infosets: {solver.matrix_repr.num_infosets}")
    print(f"  Infoset-actions: {solver.matrix_repr.num_infoset_actions}")
    print(f"  Using GPU: {solver.use_gpu}")

    return True


def test_single_iteration():
    """Test that a single CFR iteration runs."""
    print("\n" + "=" * 60)
    print("TEST: Single CFR Iteration")
    print("=" * 60)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)

    print("\nRunning 1 iteration...")
    solver.solve(iterations=1)

    print("\n✅ Iteration completed successfully!")

    return True


def test_multiple_iterations():
    """Test that multiple CFR iterations run."""
    print("\n" + "=" * 60)
    print("TEST: Multiple CFR Iterations")
    print("=" * 60)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)

    print("\nRunning 100 iterations...")
    solver.solve(iterations=100)

    print("\n✅ All iterations completed successfully!")

    return True


def test_policy_extraction():
    """Test policy extraction."""
    print("\n" + "=" * 60)
    print("TEST: Policy Extraction")
    print("=" * 60)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)

    print("\nRunning 10 iterations...")
    solver.solve(iterations=10)

    print("\nExtracting average policy...")
    policy = solver.get_average_policy()

    print(f"\n✅ Policy extracted successfully!")
    print(f"  Policy has {len(policy)} infosets")

    # Show a sample infoset policy
    sample_infoset = list(policy.keys())[0]
    print(f"\nSample infoset: {sample_infoset}")
    print(f"  Actions: {solver.matrix_repr.infoset_to_actions[sample_infoset]}")
    print(f"  Probabilities: {policy[sample_infoset]}")
    print(f"  Sum: {policy[sample_infoset].sum():.6f} (should be 1.0)")

    # Verify all probabilities sum to 1
    for infoset, probs in policy.items():
        prob_sum = probs.sum()
        assert abs(prob_sum - 1.0) < 1e-5, f"Infoset {infoset} probs don't sum to 1: {prob_sum}"

    print("\n✅ All infoset probabilities sum to 1.0!")

    return True


def test_strategy_dict():
    """Test strategy dictionary extraction."""
    print("\n" + "=" * 60)
    print("TEST: Strategy Dictionary")
    print("=" * 60)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})
    solver = MatrixCFRSolver(game, algorithm='vanilla_cfr', use_gpu=True)

    solver.solve(iterations=10)

    strategy_dict = solver.get_strategy_dict()

    print(f"\n✅ Strategy dict extracted!")
    print(f"  Contains {len(strategy_dict)} infosets")

    # Show sample
    sample_infoset = list(strategy_dict.keys())[0]
    print(f"\nSample infoset: {sample_infoset}")
    print(f"  Strategy: {strategy_dict[sample_infoset]}")

    return True


def main():
    """Run all basic tests."""
    print("=" * 60)
    print("MATRIX CFR SOLVER BASIC TESTS")
    print("=" * 60)

    results = []

    try:
        results.append(("Solver Initialization", test_solver_initialization()))
    except Exception as e:
        print(f"\n❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Solver Initialization", False))

    try:
        results.append(("Single Iteration", test_single_iteration()))
    except Exception as e:
        print(f"\n❌ Single iteration test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Single Iteration", False))

    try:
        results.append(("Multiple Iterations", test_multiple_iterations()))
    except Exception as e:
        print(f"\n❌ Multiple iterations test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multiple Iterations", False))

    try:
        results.append(("Policy Extraction", test_policy_extraction()))
    except Exception as e:
        print(f"\n❌ Policy extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Policy Extraction", False))

    try:
        results.append(("Strategy Dictionary", test_strategy_dict()))
    except Exception as e:
        print(f"\n❌ Strategy dict test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Strategy Dictionary", False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    print("=" * 60)

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All basic tests passed!\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
