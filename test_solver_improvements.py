#!/usr/bin/env python3
"""
Test script for new solver improvements.

Tests:
1. Table-based progress display
2. Exploitability history table
3. Best policy tracking
4. Config file with solver_settings

Requirements:
    source ~/open_spiel/venv/bin/activate
"""

import sys
import pyspiel
from poker_solver import UnifiedPokerSolver
from solver_metrics import MetricsTracker, AdaptiveSchedule
from solver_logger import SolverLogger

def test_basic_solve():
    """Test basic solver with new table display using Kuhn poker."""
    print("=" * 80)
    print("Testing new solver improvements with Kuhn Poker...")
    print("=" * 80)
    print()

    # Use Kuhn poker - tiny game, perfect for testing display features
    game = pyspiel.load_game("kuhn_poker")
    print(f"Game: Kuhn Poker (3 cards, 2 players)")
    print()

    # Manually create a simple CFR solver for testing
    from open_spiel.python.algorithms import cfr
    solver_obj = cfr.CFRPlusSolver(game)

    # Create metrics and logger
    metrics = MetricsTracker("CFR+ (Test)", "Kuhn Poker")
    logger = SolverLogger("CFR+ (Test)", "Kuhn Poker", max_iterations=1000)

    print("Starting solve with new table display...")
    print()

    # Initial exploitability
    from open_spiel.python.algorithms import exploitability
    conv = exploitability.nash_conv(game, solver_obj.average_policy())
    metrics.record_checkpoint(0, conv)
    logger.log_info(f"Initial exploitability: {conv:.6f}")
    logger.log_info("")

    # Solve loop with table display
    for i in range(1, 1001):
        solver_obj.evaluate_and_update_policy()

        # Progress display every 100 iterations
        if i % 100 == 0:
            elapsed = metrics.start_time - metrics.start_time + (i * 0.001)  # Mock timing
            iters_per_sec = i / max(elapsed, 0.001) if elapsed > 0 else 0
            memory_mb = 50.0  # Mock memory

            logger.log_progress_table_row(
                iteration=i,
                time_elapsed=elapsed,
                iters_per_sec=iters_per_sec * 1000,  # Scale up for visibility
                memory_mb=memory_mb,
                last_exploitability=conv
            )

        # Exploitability check every 250 iterations
        if i % 250 == 0:
            conv = exploitability.nash_conv(game, solver_obj.average_policy())
            metrics.record_checkpoint(i, conv)

            # Show exploitability table
            recent = metrics.get_recent_checkpoints(5)
            logger.log_exploitability_table(recent, metrics.best_iteration)

    # Final
    print()
    print()
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    print()

    # Print summary
    summary = metrics.get_convergence_summary()
    print("Summary:")
    print(f"  Final exploitability: {summary['final_exploitability']:.6f}")
    print(f"  Best exploitability: {summary['best_exploitability']:.6f}")
    print(f"  Best iteration: {summary['best_iteration']:,}")
    print(f"  Total time: {summary['total_time']:.1f}s")
    print(f"  Total iterations: {summary['total_iterations']:,}")


def test_config_with_settings():
    """Test config file with solver_settings."""
    from game_config import PokerGameConfig, SolverSettings

    print("\n" + "=" * 80)
    print("Testing config with solver_settings...")
    print("=" * 80)
    print()

    # Create config with solver settings
    config = PokerGameConfig(
        num_players=2,
        stack_sizes=[500, 500],
        blinds=[100, 50],
        betting_abstraction='fcpa',
        description="Test 2p 5BB with solver settings",
        solver_settings=SolverSettings(
            output_dir="test_results",
            checkpoint_dir="test_checkpoints",
            max_iterations=500,
            checkpoint_interval=250,
            progress_interval=50,
            save_best_policy=True,
            exploit_history_size=3
        )
    )

    print(f"Config: {config}")
    print(f"Solver settings: {config.solver_settings}")
    print()

    # Verify settings are accessible
    assert config.solver_settings is not None
    assert config.solver_settings.max_iterations == 500
    assert config.solver_settings.save_best_policy is True
    assert config.solver_settings.exploit_history_size == 3

    print("✓ Config with solver_settings works correctly!")
    print()


if __name__ == "__main__":
    try:
        # Test config with settings first
        test_config_with_settings()

        # Then test actual solve with new features
        test_basic_solve()

        print()
        print("=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
