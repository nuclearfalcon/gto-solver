"""
Phase 8.6 Large Chunk Stress Test

Tests progressively larger game configurations to find OOM threshold
with Phase 8.6 memory optimizations (FP16 + micro-batching).

Goal: Validate that we can solve chunks >>1,600 nodes (Phase 8.5 OOM limit)

Requires: source ~/open_spiel/venv/bin/activate

Usage:
    python test_large_chunk_stress.py
"""

import pyspiel
import time
import sys
import gc
import jax
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver
from matrix_cfr.subgame_solver import SubgameSolver


def estimate_nodes(config_desc, config):
    """Rough estimate of tree size."""
    game = pyspiel.load_game('universal_poker', config)
    # Quick traversal to count nodes
    def count_nodes(state, max_depth=20):
        if state.is_terminal() or max_depth == 0:
            return 1
        if state.is_chance_node():
            # Sample a few chance outcomes
            outcomes = state.chance_outcomes()
            if len(outcomes) > 3:
                outcomes = outcomes[:3]  # Sample first 3
            total = 1
            for action, _ in outcomes:
                child = state.child(action)
                total += count_nodes(child, max_depth - 1)
            return total
        else:
            # Count all player actions
            actions = state.legal_actions()
            total = 1
            for action in actions[:2]:  # Sample first 2 actions
                child = state.child(action)
                total += count_nodes(child, max_depth - 1)
            return total * len(actions) // 2  # Scale up

    try:
        estimated = count_nodes(game.new_initial_state())
        print(f"  Estimated nodes: ~{estimated:,}")
        return estimated
    except:
        print(f"  Could not estimate nodes (too large)")
        return None


def test_configuration(name, config, precision='fp32', micro_batch_size=24, iterations=5):
    """Test a single configuration with given memory settings."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")
    print(f"Config: precision={precision}, micro_batch_size={micro_batch_size}")
    print(f"Settings:")
    print(f"  Deck: {config['numSuits']}×{config['numRanks']} = {config['numSuits']*config['numRanks']} cards")
    print(f"  Hole cards: {config['numHoleCards']}")
    print(f"  Board cards: {config['numBoardCards']}")
    print(f"  Betting: {config['bettingAbstraction']}")
    print(f"  Rounds: {config['numRounds']}")

    # Estimate size
    estimate_nodes(name, config)

    print(f"\nAttempting to solve {iterations} iterations...")

    try:
        # Create game
        game = pyspiel.load_game('universal_poker', config)

        # Create solver with memory optimizations
        solver = MatrixCFRSolver(
            game,
            use_sparse=True,
            precision=precision,
            micro_batch_size=micro_batch_size
        )

        print(f"✓ Solver created successfully")
        print(f"  Actual nodes: {solver.matrix_repr.num_nodes:,}")
        print(f"  Infosets: {len(solver.infoset_action_indices):,}")
        print(f"  Infoset-actions: {solver.matrix_repr.num_infoset_actions:,}")

        # Solve
        start = time.time()
        solver.solve(iterations=iterations, progress_interval=999)
        elapsed = time.time() - start

        speed = iterations / elapsed

        print(f"\n✅ SUCCESS!")
        print(f"  Time: {elapsed:.2f}s for {iterations} iterations")
        print(f"  Speed: {speed:.2f} it/s")
        print(f"  Nodes: {solver.matrix_repr.num_nodes:,}")

        # Clean up
        # Store node count before cleanup
        num_nodes = solver.matrix_repr.num_nodes

        # Clean up
        del solver
        del game
        gc.collect()
        jax.clear_caches()

        return {
            'status': 'SUCCESS',
            'nodes': num_nodes,
            'time': elapsed,
            'speed': speed
        }

    except Exception as e:
        error_msg = str(e)
        # Detect OOM/memory errors from multiple sources
        is_oom = (
            'out of memory' in error_msg.lower() or
            'oom' in error_msg.lower() or
            'RESOURCE_EXHAUSTED' in error_msg or
            'Failed to allocate' in error_msg or
            'memory' in error_msg.lower()
        )

        print(f"\n❌ FAILED: {error_msg[:200]}")
        if is_oom:
            print(f"  ⚠️  MEMORY ERROR DETECTED")

        # Clean up
        gc.collect()
        jax.clear_caches()

        return {
            'status': 'OOM' if is_oom else 'ERROR',
            'error': error_msg[:200],
            'nodes': None
        }


def main():
    """Run progressive stress tests."""
    print("="*80)
    print("PHASE 8.6 LARGE CHUNK STRESS TEST")
    print("="*80)
    print("\nTesting the 57k Turn chunk that caused OOM in Phase 8.5.")
    print()
    print("Background:")
    print("  - Phase 8.5 OOM limit: ~1,600 nodes")
    print("  - Phase 8.5 Turn chunk: 57,521 nodes (36x larger)")
    print("  - Phase 8.5 result: OOM error")
    print()
    print("Phase 8.6 Optimizations:")
    print("  - Micro-batching: Reduce peak memory 2-8x")
    print("  - FP16 precision: Reduce strategy memory 50%")
    print("  - Combined: Up to 16x memory reduction")
    print()
    print("Test Strategy:")
    print("  1. Validate baseline (1.6k nodes) still works")
    print("  2. Try 57k with default settings (expect OOM)")
    print("  3. Retry 57k with progressive optimizations until success or failure")
    print()
    print("Goal: Prove Phase 8.6 solves the OOM issue!")
    print()

    results = []

    # Test 1: Baseline (Phase 8.5 working config)
    print("\n" + "="*80)
    print("TEST SERIES 1: BASELINE (Phase 8.5 known working)")
    print("="*80)

    config_1600 = {
        "betting": "nolimit",
        "numPlayers": 2,
        "numRounds": 1,
        "blind": "50 100",
        "firstPlayer": "2",
        "numSuits": 2,
        "numRanks": 3,
        "numHoleCards": 1,
        "numBoardCards": "2",  # Turn-like (2 cumulative board cards)
        "stack": "1000 1000",
        "bettingAbstraction": "fc"
    }

    result = test_configuration(
        "Baseline ~1,600 nodes (FC)",
        config_1600,
        precision='fp32',
        micro_batch_size=24
    )
    results.append(('Baseline 1.6k (FP32, batch=24)', result))

    if result['status'] != 'SUCCESS':
        print("\n⚠️  Baseline failed! Cannot proceed.")
        return 1

    # Test 2: Same config with micro-batching
    print("\n" + "="*80)
    print("TEST SERIES 2: BASELINE WITH MICRO-BATCHING")
    print("="*80)

    result = test_configuration(
        "Baseline with micro-batch=6",
        config_1600,
        precision='fp32',
        micro_batch_size=6
    )
    results.append(('Baseline 1.6k (FP32, batch=6)', result))

    # Test 3: Larger config - FCPA betting (more actions) - THE CRITICAL 57K TEST
    print("\n" + "="*80)
    print("TEST SERIES 3: 57K TURN CHUNK (CRITICAL TEST)")
    print("="*80)
    print("This is the configuration that caused OOM in Phase 8.5")
    print()

    config_57k = {
        "betting": "nolimit",
        "numPlayers": 2,
        "numRounds": 1,
        "blind": "50 100",
        "firstPlayer": "2",
        "numSuits": 2,
        "numRanks": 4,  # Increased from 3
        "numHoleCards": 1,
        "numBoardCards": "2",
        "stack": "1000 1000",
        "bettingAbstraction": "fcpa"  # More actions per decision
    }

    # Try with default batch size first (expect OOM)
    result = test_configuration(
        "57k Turn chunk (FP32, batch=24)",
        config_57k,
        precision='fp32',
        micro_batch_size=24
    )
    results.append(('57k Turn (FP32, batch=24)', result))

    # CRITICAL: Retry with micro-batching (this is the key test!)
    if result['status'] == 'OOM':
        print("\n⚠️  OOM with default batch size (EXPECTED).")
        print("🔧 Retrying with micro-batch=12 (Phase 8.6 optimization)...")

        # Extra cleanup before retry
        import time
        gc.collect()
        jax.clear_caches()
        time.sleep(2)  # Let GPU release memory

        result = test_configuration(
            "57k Turn chunk (FP32, batch=12)",
            config_57k,
            precision='fp32',
            micro_batch_size=12
        )
        results.append(('57k Turn (FP32, batch=12)', result))

    # If still OOM, try batch=6
    if result['status'] == 'OOM':
        print("\n⚠️  Still OOM with batch=12.")
        print("🔧 Retrying with micro-batch=6 (aggressive optimization)...")

        gc.collect()
        jax.clear_caches()
        time.sleep(2)

        result = test_configuration(
            "57k Turn chunk (FP32, batch=6)",
            config_57k,
            precision='fp32',
            micro_batch_size=6
        )
        results.append(('57k Turn (FP32, batch=6)', result))

    # If STILL OOM, try FP16
    if result['status'] == 'OOM':
        print("\n⚠️  Still OOM with FP32 batch=6.")
        print("🔧 Retrying with FP16 + batch=6 (maximum optimization)...")

        gc.collect()
        jax.clear_caches()
        time.sleep(2)

        result = test_configuration(
            "57k Turn chunk (FP16, batch=6)",
            config_57k,
            precision='fp16',
            micro_batch_size=6
        )
        results.append(('57k Turn (FP16, batch=6)', result))

    # Last resort: FP16 + batch=3
    if result['status'] == 'OOM':
        print("\n⚠️  Still OOM with FP16 batch=6.")
        print("🔧 Final attempt: FP16 + batch=3 (extreme optimization)...")

        gc.collect()
        jax.clear_caches()
        time.sleep(2)

        result = test_configuration(
            "57k Turn chunk (FP16, batch=3)",
            config_57k,
            precision='fp16',
            micro_batch_size=3
        )
        results.append(('57k Turn (FP16, batch=3)', result))

    # Skip larger tests if 57k succeeded - we've proven the point!
    if result['status'] == 'SUCCESS':
        print("\n" + "="*80)
        print("🎉 57K TURN CHUNK SOLVED!")
        print("="*80)
        print("\nPhase 8.6 successfully handles the chunk that caused OOM in Phase 8.5!")
        print("Skipping larger tests - goal achieved.")
        print()
    else:
        print("\n" + "="*80)
        print("⚠️  57K TURN CHUNK STILL FAILS")
        print("="*80)
        print("\nPhase 8.6 optimizations were not sufficient for the 57k chunk.")
        print("This indicates hardware limitations beyond what software optimizations can solve.")
        print()

    # Print summary
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    print(f"{'Configuration':<40} {'Status':<10} {'Nodes':>10} {'Speed':>10}")
    print("-"*80)

    for name, result in results:
        status = result['status']
        nodes = f"{result.get('nodes', 0):,}" if result.get('nodes') else "N/A"
        speed = f"{result.get('speed', 0):.2f} it/s" if result.get('speed') else "N/A"
        print(f"{name:<40} {status:<10} {nodes:>10} {speed:>10}")

    # Find largest successful config
    successful = [(name, r) for name, r in results if r['status'] == 'SUCCESS']
    if successful:
        largest = max(successful, key=lambda x: x[1].get('nodes', 0))
        print(f"\n✅ Largest successful: {largest[0]}")
        print(f"   Nodes: {largest[1]['nodes']:,}")
        print(f"   Speed: {largest[1]['speed']:.2f} it/s")

        if largest[1]['nodes'] > 1600:
            improvement = largest[1]['nodes'] / 1600
            print(f"\n🎉 SUCCESS! Phase 8.6 handles {improvement:.1f}x larger chunks than Phase 8.5!")
        else:
            print(f"\n⚠️  Did not exceed Phase 8.5 limit (1,600 nodes)")
    else:
        print(f"\n❌ All tests failed!")

    # Find OOM threshold
    oom_tests = [(name, r) for name, r in results if r['status'] == 'OOM']
    if oom_tests:
        print(f"\n⚠️  OOM threshold found:")
        for name, r in oom_tests:
            print(f"   {name}")

    print("\n" + "="*80)
    print("PHASE 8.6 STRESS TEST COMPLETE")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
