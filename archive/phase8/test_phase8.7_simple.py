"""
Phase 8.7 Simple Test: Validate Sub-Chunking Logic Without Full Solve

Tests the sub-chunking mechanism (threshold detection, enumeration, filtering)
without actually solving large games.

IMPORTANT: Run with OpenSpiel virtual environment activated:
    source ~/open_spiel/venv/bin/activate && python test_phase8.7_simple.py
"""

import logging
from typing import Dict

from matrix_cfr.subgame_solver import SubgameSolver, _infoset_matches_public_card

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_threshold_detection():
    """Test that needs_splitting property works correctly."""
    logger.info("=" * 80)
    logger.info("TEST 1: THRESHOLD DETECTION")
    logger.info("=" * 80)

    config = {
        'betting': 'nolimit',
        'numPlayers': 2,
        'numRounds': 4,
        'blind': '100 50',
        'firstPlayer': '2 1 1 1',
        'numSuits': 2,
        'numRanks': 5,
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',
        'stack': '10000 10000',
        'bettingAbstraction': 'fcpa'
    }

    # Test with high threshold (should NOT split)
    solver_no_split = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        max_nodes=100000
    )

    assert not solver_no_split.needs_splitting, "Should not split with high threshold"
    logger.info("✓ High threshold correctly prevents splitting")

    # Test with low threshold (should split)
    solver_split = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        max_nodes=500
    )

    assert solver_split.needs_splitting, "Should split with low threshold"
    logger.info("✓ Low threshold correctly triggers splitting")

    # Test preflop never splits
    solver_preflop = SubgameSolver(
        full_game_config=config,
        round_name="preflop",
        max_nodes=1
    )

    assert not solver_preflop.needs_splitting, "Preflop should never split"
    logger.info("✓ Preflop correctly never splits")

    logger.info("TEST 1: PASSED\n")


def test_card_enumeration():
    """Test public card enumeration."""
    logger.info("=" * 80)
    logger.info("TEST 2: CARD ENUMERATION")
    logger.info("=" * 80)

    config = {
        'betting': 'nolimit',
        'numPlayers': 2,
        'numRounds': 4,
        'blind': '100 50',
        'firstPlayer': '2 1 1 1',
        'numSuits': 2,
        'numRanks': 5,
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',
        'stack': '10000 10000',
        'bettingAbstraction': 'fcpa'
    }

    solver = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        max_nodes=500
    )

    cards = solver._enumerate_public_cards()

    assert len(cards) == 10, f"Expected 10 cards (2 suits × 5 ranks), got {len(cards)}"
    assert '2s' in cards, "Should contain 2s"
    assert '6h' in cards, "Should contain 6h"

    logger.info(f"✓ Enumerated {len(cards)} cards correctly")
    logger.info(f"  Cards: {cards}")
    logger.info("TEST 2: PASSED\n")


def test_infoset_filtering():
    """Test infoset filtering by public card."""
    logger.info("=" * 80)
    logger.info("TEST 3: INFOSET FILTERING")
    logger.info("=" * 80)

    # Test cases with different infoset formats
    test_cases = [
        # (infoset, target_card, should_match)
        ("[Public: 2s3h4d5c][Player: 0]", "5c", True),
        ("[Public: 2s3h4d5c][Player: 0]", "4d", False),
        ("[Public: 2s3h4d][Player: 1]", "4d", True),
        ("[Public: 2s][Player: 0]", "2s", True),
        ("[Public: ][Player: 0]", "2s", False),
    ]

    passed = 0
    for infoset, target, expected in test_cases:
        result = _infoset_matches_public_card(infoset, target)
        if result == expected:
            passed += 1
            logger.info(f"✓ {infoset} + '{target}' → {result} (expected {expected})")
        else:
            logger.error(f"✗ {infoset} + '{target}' → {result} (expected {expected})")

    assert passed == len(test_cases), f"Only {passed}/{len(test_cases)} tests passed"
    logger.info("TEST 3: PASSED\n")


def test_per_round_thresholds():
    """Test per-round threshold configuration."""
    logger.info("=" * 80)
    logger.info("TEST 4: PER-ROUND THRESHOLDS")
    logger.info("=" * 80)

    config = {
        'betting': 'nolimit',
        'numPlayers': 2,
        'numRounds': 4,
        'blind': '100 50',
        'firstPlayer': '2 1 1 1',
        'numSuits': 2,
        'numRanks': 5,
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',
        'stack': '10000 10000',
        'bettingAbstraction': 'fcpa'
    }

    # Test dict-based thresholds
    thresholds = {
        "preflop": 50000,
        "flop": 20000,
        "turn": 5000,
        "river": 2000
    }

    for round_name, expected_threshold in thresholds.items():
        solver = SubgameSolver(
            full_game_config=config,
            round_name=round_name,
            max_nodes=thresholds
        )

        assert solver.max_nodes_threshold == expected_threshold, \
            f"{round_name}: expected {expected_threshold}, got {solver.max_nodes_threshold}"

        logger.info(f"✓ {round_name}: threshold = {expected_threshold}")

    # Test single int threshold
    solver_uniform = SubgameSolver(
        full_game_config=config,
        round_name="turn",
        max_nodes=10000
    )

    assert solver_uniform.max_nodes_threshold == 10000, "Single int threshold should work"
    logger.info("✓ Single int threshold works")

    logger.info("TEST 4: PASSED\n")


def main():
    """Run all simple validation tests."""
    logger.info("")
    logger.info("*" * 80)
    logger.info("PHASE 8.7 SIMPLE VALIDATION (NO FULL SOLVE)")
    logger.info("*" * 80)
    logger.info("")

    try:
        test_threshold_detection()
        test_card_enumeration()
        test_infoset_filtering()
        test_per_round_thresholds()

        logger.info("*" * 80)
        logger.info("ALL TESTS PASSED ✅")
        logger.info("*" * 80)
        logger.info("")
        logger.info("Phase 8.7 Core Logic: VALIDATED")
        logger.info("")
        logger.info("Note: Full solve testing requires addressing the memory issue")
        logger.info("      where we solve the entire game tree per sub-chunk.")
        logger.info("      See comments in code for details on Option A vs Option B.")
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
