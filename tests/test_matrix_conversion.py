#!/usr/bin/env python3
"""
Test Matrix Conversion for Kuhn Poker

Validates that the GameTreeConverter correctly converts Kuhn poker
to matrix representation.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python tests/test_matrix_conversion.py
"""

import sys
import os

# Add parent directory to path to import matrix_cfr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspiel
import numpy as np
from matrix_cfr.game_to_matrix import GameTreeConverter, print_matrix_stats
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_kuhn_2p():
    """Test matrix conversion on 2-player Kuhn poker."""
    print("\n" + "=" * 60)
    print("TEST: 2-Player Kuhn Poker Matrix Conversion")
    print("=" * 60)

    # Create Kuhn poker game
    game = pyspiel.load_game('kuhn_poker', {'players': 2})
    print(f"Game: {game.get_type().long_name}")
    print(f"Players: {game.num_players()}")

    # Convert to matrix representation
    converter = GameTreeConverter(game)
    matrix_repr = converter.build_matrices()

    # Print statistics
    print_matrix_stats(matrix_repr)

    # Validation checks
    print("\n=== Validation Checks ===")

    # Check 1: All nodes should have consistent parent-child relationships
    for node in matrix_repr.nodes:
        if node.parent_id >= 0:
            parent = matrix_repr.nodes[node.parent_id]
            assert parent.depth == node.depth - 1, \
                f"Node {node.node_id} depth inconsistent with parent {node.parent_id}"
    print("✓ Parent-child depth relationships valid")

    # Check 2: Level matrices should have correct structure
    for level, L in enumerate(matrix_repr.level_matrices):
        nodes_at_level = [n for n in matrix_repr.nodes if n.depth == level]
        if nodes_at_level:
            # Number of edges should equal number of nodes at this level (minus root if level 0)
            if level > 0:
                assert L.nnz == len(nodes_at_level), \
                    f"Level {level} matrix has {L.nnz} edges but {len(nodes_at_level)} nodes"
    print("✓ Level matrices have correct edge counts")

    # Check 3: Terminal utilities should be zero-sum
    for node in matrix_repr.nodes:
        if node.is_terminal:
            util_sum = node.terminal_utilities.sum()
            assert abs(util_sum) < 1e-6, \
                f"Terminal node {node.node_id} utilities don't sum to zero: {node.terminal_utilities}"
    print("✓ Terminal utilities are zero-sum")

    # Check 4: Player matrix should have exactly one 1 per decision node
    decision_nodes = [n for n in matrix_repr.nodes if n.player >= 0]
    for node in decision_nodes:
        row_sum = matrix_repr.player_matrix[node.node_id, :].sum()
        assert abs(row_sum - 1.0) < 1e-6, \
            f"Decision node {node.node_id} player matrix row doesn't sum to 1"
    print("✓ Player matrix has valid structure")

    # Check 5: Infoset-action matrix should map to valid nodes
    ia_matrix = matrix_repr.infoset_action_to_node_matrix
    assert ia_matrix.shape[0] == matrix_repr.num_infoset_actions
    assert ia_matrix.shape[1] == matrix_repr.num_nodes
    print("✓ Infoset-action matrix has correct dimensions")

    print("\n✅ All validation checks passed for 2-player Kuhn poker!\n")
    return True


def test_kuhn_3p():
    """Test matrix conversion on 3-player Kuhn poker."""
    print("\n" + "=" * 60)
    print("TEST: 3-Player Kuhn Poker Matrix Conversion")
    print("=" * 60)

    # Create 3-player Kuhn poker game
    game = pyspiel.load_game('kuhn_poker', {'players': 3})
    print(f"Game: {game.get_type().long_name}")
    print(f"Players: {game.num_players()}")

    # Convert to matrix representation
    converter = GameTreeConverter(game)
    matrix_repr = converter.build_matrices()

    # Print statistics
    print_matrix_stats(matrix_repr)

    # Basic validation
    print("\n=== Validation Checks ===")

    assert matrix_repr.num_players == 3
    print("✓ Correct number of players")

    assert matrix_repr.num_nodes > 0
    print(f"✓ Enumerated {matrix_repr.num_nodes} nodes")

    assert matrix_repr.num_infosets > 0
    print(f"✓ Found {matrix_repr.num_infosets} information sets")

    # Check zero-sum
    for node in matrix_repr.nodes:
        if node.is_terminal:
            util_sum = node.terminal_utilities.sum()
            assert abs(util_sum) < 1e-6, \
                f"Terminal node {node.node_id} utilities don't sum to zero: {node.terminal_utilities}"
    print("✓ Terminal utilities are zero-sum")

    print("\n✅ All validation checks passed for 3-player Kuhn poker!\n")
    return True


def test_matrix_sparsity():
    """Test that matrices are actually sparse as expected."""
    print("\n" + "=" * 60)
    print("TEST: Matrix Sparsity Analysis")
    print("=" * 60)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})
    converter = GameTreeConverter(game)
    matrix_repr = converter.build_matrices()

    print("\n=== Sparsity Analysis ===")

    # Level matrices should be very sparse
    for i, L in enumerate(matrix_repr.level_matrices):
        density = L.nnz / (L.shape[0] * L.shape[1])
        sparsity_pct = 100 * (1 - density)
        print(f"Level {i}: {sparsity_pct:.2f}% sparse (density: {density:.6f})")

        # For Kuhn poker, should be >99% sparse
        assert sparsity_pct > 99.0, f"Level {i} matrix not sparse enough: {sparsity_pct}%"

    print("✓ All level matrices are highly sparse (>99%)")

    # Infoset-action matrix should also be sparse
    ia_matrix = matrix_repr.infoset_action_to_node_matrix
    ia_density = ia_matrix.nnz / (ia_matrix.shape[0] * ia_matrix.shape[1])
    ia_sparsity = 100 * (1 - ia_density)
    print(f"\nInfoset-action matrix: {ia_sparsity:.2f}% sparse")
    assert ia_sparsity > 90.0, f"Infoset-action matrix not sparse enough: {ia_sparsity}%"
    print("✓ Infoset-action matrix is sparse")

    print("\n✅ Sparsity test passed!\n")
    return True


def test_node_enumeration():
    """Test that all game states are enumerated."""
    print("\n" + "=" * 60)
    print("TEST: Complete Node Enumeration")
    print("=" * 60)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})
    converter = GameTreeConverter(game)
    matrix_repr = converter.build_matrices()

    print(f"\nTotal nodes enumerated: {matrix_repr.num_nodes}")
    print(f"Tree depth: {matrix_repr.num_levels} levels")

    # Count node types
    decision_nodes = sum(1 for n in matrix_repr.nodes if n.player >= 0)
    chance_nodes = sum(1 for n in matrix_repr.nodes if n.is_chance)
    terminal_nodes = sum(1 for n in matrix_repr.nodes if n.is_terminal)

    print(f"\nNode breakdown:")
    print(f"  Decision nodes: {decision_nodes}")
    print(f"  Chance nodes: {chance_nodes}")
    print(f"  Terminal nodes: {terminal_nodes}")
    print(f"  Total: {decision_nodes + chance_nodes + terminal_nodes}")

    assert decision_nodes + chance_nodes + terminal_nodes == matrix_repr.num_nodes
    print("✓ All nodes accounted for")

    # Show some example nodes
    print(f"\nExample nodes:")
    for i, node in enumerate(matrix_repr.nodes[:5]):
        node_type = "TERMINAL" if node.is_terminal else ("CHANCE" if node.is_chance else "DECISION")
        print(f"  Node {node.node_id}: depth={node.depth}, type={node_type}, player={node.player}")

    print("\n✅ Node enumeration test passed!\n")
    return True


def main():
    """Run all matrix conversion tests."""
    print("=" * 60)
    print("MATRIX CONVERSION TESTS FOR KUHN POKER")
    print("=" * 60)

    results = []

    try:
        results.append(("2-Player Kuhn", test_kuhn_2p()))
    except Exception as e:
        print(f"\n❌ 2-Player Kuhn test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("2-Player Kuhn", False))

    try:
        results.append(("3-Player Kuhn", test_kuhn_3p()))
    except Exception as e:
        print(f"\n❌ 3-Player Kuhn test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("3-Player Kuhn", False))

    try:
        results.append(("Sparsity Analysis", test_matrix_sparsity()))
    except Exception as e:
        print(f"\n❌ Sparsity test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Sparsity Analysis", False))

    try:
        results.append(("Node Enumeration", test_node_enumeration()))
    except Exception as e:
        print(f"\n❌ Node enumeration test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Node Enumeration", False))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    print("=" * 60)

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All matrix conversion tests passed!\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
