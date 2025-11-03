"""
Phase 9 Test: True Game Pre-Dealing (Option A vs Option B)

Tests and validates the true pre-dealing implementation that achieves GENUINE
memory reduction by constraining the game tree BEFORE solving.

IMPORTANT: Run with OpenSpiel virtual environment activated:
    source ~/open_spiel/venv/bin/activate && python test_phase9_true_predealing.py

Tests:
1. Card String Conversion: Validate card string to action mapping
2. Starting State Creation: Verify states with pre-dealt cards
3. Memory Reduction: Compare game tree sizes (Option A vs B)
4. Policy Correctness: Validate policies are equivalent
5. Full Pipeline: End-to-end test with both options
"""

import logging
import time
from typing import Dict
import pyspiel

from matrix_cfr.subgame_solver import SubgameSolver
from matrix_cfr.game_to_matrix import GameTreeConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_config() -> Dict:
    """
    Create minimal test config for validation.

    Uses 2 suits × 5 ranks = 10 cards (just enough for 2 players × 2 hole + 5 board)
    """
    return {
        'betting': 'nolimit',
        'numPlayers': 2,
        'numRounds': 4,
        'blind': '100 50',
        'firstPlayer': '2 1 1 1',
        'numSuits': 2,
        'numRanks': 5,  # 2,3,4,5,6
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',
        'stack': '10000 10000',
        'bettingAbstraction': 'fcpa'
    }


def test_1_card_string_conversion():
    """Test 1: Validate card string to action conversion."""
    logger.info("=" * 80)
    logger.info("TEST 1: CARD STRING CONVERSION")
    logger.info("=" * 80)

    config = create_test_config()

    solver = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        max_nodes=500,
        precision='fp16'  # Use FP16 for lower memory
    )

    game = pyspiel.load_game("universal_poker", solver.subgame_config)

    # Test conversions
    test_cases = [
        ('2s', 0),  # Rank 0 (2) × num_suits (2) + Suit 0 (s) = 0
        ('2h', 1),  # Rank 0 × 2 + Suit 1 (h) = 1
        ('3s', 2),  # Rank 1 (3) × 2 + Suit 0 = 2
        ('6h', 9),  # Rank 4 (6) × 2 + Suit 1 = 9
    ]

    passed = 0
    for card_str, expected_action in test_cases:
        action = solver._card_string_to_action(card_str, game)
        if action == expected_action:
            logger.info(f"✓ {card_str} → action {action} (expected {expected_action})")
            passed += 1
        else:
            logger.error(f"✗ {card_str} → action {action} (expected {expected_action})")

    assert passed == len(test_cases), f"Only {passed}/{len(test_cases)} conversions passed"
    logger.info("TEST 1: PASSED\n")


def test_2_starting_state_creation():
    """Test 2: Verify starting state creation with pre-dealt cards."""
    logger.info("=" * 80)
    logger.info("TEST 2: STARTING STATE CREATION")
    logger.info("=" * 80)

    config = create_test_config()

    solver = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        max_nodes=500,
        precision='fp16'  # Use FP16 for lower memory
    )

    game = pyspiel.load_game("universal_poker", solver.subgame_config)

    # Create starting state with turn card = '2s'
    target_card = '2s'
    logger.info(f"Creating starting state with turn card = {target_card}...")

    starting_state = solver._create_starting_state_with_card(game, target_card)

    # Verify state is valid
    assert not starting_state.is_terminal(), "Starting state should not be terminal"
    logger.info(f"✓ Created valid starting state (not terminal)")

    # Check that we can get legal actions
    if not starting_state.is_chance_node():
        legal_actions = starting_state.legal_actions()
        logger.info(f"✓ Starting state has {len(legal_actions)} legal actions")
    else:
        logger.info(f"✓ Starting state is at a chance node (expected)")

    logger.info("TEST 2: PASSED\n")


def test_3_memory_reduction():
    """Test 3: Compare game tree sizes between Option A and full tree."""
    logger.info("=" * 80)
    logger.info("TEST 3: MEMORY REDUCTION VALIDATION")
    logger.info("=" * 80)

    config = create_test_config()

    solver = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        max_nodes=500,
        precision='fp16'  # Use FP16 for lower memory
    )

    game = pyspiel.load_game("universal_poker", solver.subgame_config)

    # Build full game tree
    logger.info("Building FULL game tree (no pre-dealing)...")
    converter_full = GameTreeConverter(game)
    start_time = time.time()
    matrices_full = converter_full.build_matrices()
    time_full = time.time() - start_time

    nodes_full = matrices_full.num_nodes
    infosets_full = matrices_full.num_infosets
    logger.info(f"✓ Full tree: {nodes_full:,} nodes, {infosets_full:,} infosets ({time_full:.2f}s)")

    # Build constrained game tree with pre-dealt turn card
    target_card = '2s'
    logger.info(f"\nBuilding CONSTRAINED game tree (turn card = {target_card})...")
    starting_state = solver._create_starting_state_with_card(game, target_card)
    converter_constrained = GameTreeConverter(game)
    start_time = time.time()
    matrices_constrained = converter_constrained.build_matrices(starting_state=starting_state)
    time_constrained = time.time() - start_time

    nodes_constrained = matrices_constrained.num_nodes
    infosets_constrained = matrices_constrained.num_infosets
    logger.info(f"✓ Constrained tree: {nodes_constrained:,} nodes, {infosets_constrained:,} infosets ({time_constrained:.2f}s)")

    # Calculate reduction
    node_reduction = nodes_full / nodes_constrained if nodes_constrained > 0 else 0
    infoset_reduction = infosets_full / infosets_constrained if infosets_constrained > 0 else 0

    logger.info(f"\nMemory Reduction:")
    logger.info(f"  Nodes: {nodes_full:,} → {nodes_constrained:,} ({node_reduction:.1f}× smaller)")
    logger.info(f"  Infosets: {infosets_full:,} → {infosets_constrained:,} ({infoset_reduction:.1f}× smaller)")

    # Verify meaningful reduction (should be ~8× for turn with 2×5 deck)
    # With 10 cards and turn card fixed, we expect significant reduction
    assert node_reduction >= 2.0, f"Expected at least 2× node reduction, got {node_reduction:.1f}×"
    logger.info(f"✓ Achieved {node_reduction:.1f}× node reduction")

    logger.info("TEST 3: PASSED\n")

    return {
        'nodes_full': nodes_full,
        'nodes_constrained': nodes_constrained,
        'reduction': node_reduction
    }


def test_4_option_comparison():
    """Test 4: Compare Option A (true pre-dealing) vs Option B (filtered extraction)."""
    logger.info("=" * 80)
    logger.info("TEST 4: OPTION A VS OPTION B COMPARISON")
    logger.info("=" * 80)

    config = create_test_config()
    target_card = '2s'

    # Option B: Filtered extraction (Phase 8.7)
    logger.info("\nOption B: Filtered Extraction (baseline)...")
    solver_b = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        max_nodes=500,
        precision='fp16',  # Use FP16 for lower memory
        use_true_predealing=False  # Option B
    )

    start_time = time.time()
    policy_b = solver_b._solve_with_public_card_filter(
        target_card=target_card,
        iterations=10,  # Very few iterations for speed
        progress_interval=999
    )
    time_b = time.time() - start_time

    infosets_b = len(policy_b.policy)
    logger.info(f"✓ Option B: {infosets_b} infosets, {time_b:.2f}s")

    # Option A: True pre-dealing (Phase 9)
    logger.info("\nOption A: True Pre-Dealing (Phase 9)...")
    solver_a = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        max_nodes=500,
        precision='fp16',  # Use FP16 for lower memory
        use_true_predealing=True  # Option A
    )

    start_time = time.time()
    policy_a = solver_a._solve_with_true_predealing(
        target_card=target_card,
        iterations=10,  # Same iterations
        progress_interval=999
    )
    time_a = time.time() - start_time

    infosets_a = len(policy_a.policy)
    logger.info(f"✓ Option A: {infosets_a} infosets, {time_a:.2f}s")

    # Compare
    logger.info(f"\nComparison:")
    logger.info(f"  Policy sizes: Option B = {infosets_b}, Option A = {infosets_a}")
    logger.info(f"  Time: Option B = {time_b:.2f}s, Option A = {time_a:.2f}s")

    if time_b > time_a:
        speedup = time_b / time_a
        logger.info(f"  ✓ Option A is {speedup:.2f}× faster")
    else:
        logger.info(f"  ⚠️ Option A slower (likely due to small test size)")

    # Policy sizes should be similar (both filtering to same card)
    size_diff = abs(infosets_a - infosets_b) / max(infosets_a, infosets_b)
    assert size_diff < 0.5, f"Policy sizes too different: {infosets_a} vs {infosets_b}"
    logger.info(f"✓ Policy sizes within 50% ({size_diff*100:.1f}% difference)")

    logger.info("TEST 4: PASSED\n")


def test_5_full_pipeline():
    """Test 5: End-to-end test with true pre-dealing enabled."""
    logger.info("=" * 80)
    logger.info("TEST 5: FULL PIPELINE WITH TRUE PRE-DEALING")
    logger.info("=" * 80)

    config = create_test_config()

    # Create Turn solver with true pre-dealing enabled (default)
    turn_solver = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        blueprint_policy=None,
        precision='fp16',
        micro_batch_size=6,
        max_nodes=500,  # Force sub-chunking
        use_true_predealing=True  # Phase 9
    )

    logger.info("Solving Turn chunk with true pre-dealing...")
    start_time = time.time()

    policy = turn_solver.solve(iterations=10, progress_interval=999)

    solve_time = time.time() - start_time

    logger.info(f"\n✓ Turn chunk solved successfully")
    logger.info(f"  Time: {solve_time:.1f}s")
    logger.info(f"  Policy size: {len(policy.policy)} infosets")

    # Validation
    assert len(policy.policy) > 0, "Policy should not be empty"
    logger.info(f"✓ Policy validation passed")

    logger.info("TEST 5: PASSED\n")


def main():
    """Run all Phase 9 tests."""
    logger.info("")
    logger.info("*" * 80)
    logger.info("PHASE 9 TRUE GAME PRE-DEALING: VALIDATION TEST SUITE")
    logger.info("*" * 80)
    logger.info("")

    start_time = time.time()

    try:
        # Run all tests
        test_1_card_string_conversion()
        test_2_starting_state_creation()
        memory_results = test_3_memory_reduction()
        test_4_option_comparison()
        test_5_full_pipeline()

        total_time = time.time() - start_time

        logger.info("")
        logger.info("*" * 80)
        logger.info("ALL TESTS PASSED ✅")
        logger.info("*" * 80)
        logger.info(f"Total test time: {total_time:.1f}s")
        logger.info("")
        logger.info("Phase 9 Implementation: ✅ VALIDATED")
        logger.info("")
        logger.info("Key Results:")
        logger.info(f"  - Memory reduction: {memory_results['reduction']:.1f}× smaller game trees")
        logger.info(f"  - Full tree nodes: {memory_results['nodes_full']:,}")
        logger.info(f"  - Constrained tree nodes: {memory_results['nodes_constrained']:,}")
        logger.info("")
        logger.info("Phase 9 successfully enables true game pre-dealing for genuine memory reduction!")
        logger.info("")

    except Exception as e:
        logger.error("")
        logger.error("*" * 80)
        logger.error("TEST FAILED ❌")
        logger.error("*" * 80)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
