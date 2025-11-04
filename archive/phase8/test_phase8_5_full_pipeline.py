"""
Test Phase 8.5: Full 4-Chunk Pipeline

Validates that the complete chunked Hold'em solving pipeline works:
- Preflop → Flop → Turn → River
- Memory profiling per chunk
- Policy combination and querying
- Validation testing

IMPORTANT: This test uses a minimal config (2 suits × 4 ranks = 8 cards, 1 hole card)
to keep game trees manageable:
  - Preflop: 1,913 nodes
  - Flop:    11,489 nodes
  - Turn:    57,521 nodes
  - River:   230,561 nodes

The turn and river are quite large, so we use reduced iterations (10-20) for testing.
Garbage collection is run between chunks to avoid GPU memory fragmentation.

Usage:
    source ~/open_spiel/venv/bin/activate
    python test_phase8_5_full_pipeline.py
"""

import pyspiel
from matrix_cfr.subgame_solver import (
    SubgameSolver,
    ChunkedSolver,
    BlueprintPolicy,
    CombinedPolicy
)
from matrix_cfr.gpu_memory import MemoryProfiler
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver
import tempfile
import os


def get_minimal_4round_config():
    """
    Ultra-minimal Hold'em config for full pipeline testing.

    Uses minimal deck to keep tree size manageable while testing
    all 4 betting rounds.

    Card accounting:
    - 2 players × 1 hole card = 2 cards
    - Board cards: 0 + 1 + 1 + 1 = 3 cards
    - Total needed: 5 cards minimum
    - Using: 2 suits × 4 ranks = 8 cards (safe buffer)

    Returns:
        Config dict suitable for 4-chunk pipeline
    """
    return {
        "betting": "nolimit",
        "numPlayers": 2,
        "numRounds": 4,  # All 4 rounds: preflop, flop, turn, river
        "blind": "50 100",
        "firstPlayer": "2 1 1 1",  # Preflop: position 2, postflop: position 1
        "numSuits": 2,
        "numRanks": 4,  # 2 suits × 4 ranks = 8 cards total (enough for 2 hole + 3 board)
        "numHoleCards": 1,  # Single hole card (like Leduc)
        "numBoardCards": "0 1 1 1",  # Preflop: 0, Flop: 1, Turn: 1, River: 1
        "stack": "1000 1000",
        "bettingAbstraction": "fcpa"
    }


def test_four_chunk_solve():
    """Test complete 4-chunk pipeline (preflop→flop→turn→river)."""
    print("\n" + "=" * 80)
    print("TEST 1: Full 4-Chunk Pipeline")
    print("=" * 80)

    config = get_minimal_4round_config()

    print("\nCreating ChunkedSolver for 4-round Hold'em...")
    print(f"  Config: {config['numSuits']} suits × {config['numRanks']} ranks")
    print(f"  Deck size: {config['numSuits'] * config['numRanks']} cards")
    print(f"  Hole cards: {config['numHoleCards']}")
    print(f"  Board cards: {config['numBoardCards']}")

    chunked = ChunkedSolver(full_game_config=config)

    # Verify all 4 chunks configured
    assert chunked.chunks == ["preflop", "flop", "turn", "river"], \
        f"Expected 4 chunks, got {chunked.chunks}"
    print(f"✓ Configured chunks: {chunked.chunks}")

    print("\nSolving all 4 chunks (20 iterations per chunk)...")
    print("  This will take several minutes...")
    print("  Note: Using 20 iterations for testing (turn/river are large)")
    print()

    # Solve all chunks (reduced iterations for large games)
    policies = chunked.solve(iterations_per_chunk=20, progress_interval=999)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

    # Verify all 4 policies created
    assert "preflop" in policies, "Preflop policy missing!"
    assert "flop" in policies, "Flop policy missing!"
    assert "turn" in policies, "Turn policy missing!"
    assert "river" in policies, "River policy missing!"

    print("\n📊 Policy Statistics:")
    for round_name in ["preflop", "flop", "turn", "river"]:
        policy = policies[round_name]
        num_infosets = len(policy.policy)
        print(f"  {round_name.capitalize():<8} {num_infosets:>6} infosets")

        # Verify policy is not empty
        assert num_infosets > 0, f"{round_name} policy is empty!"

    # Sample some strategies
    print("\n📋 Sample Strategies:")
    for round_name in ["preflop", "flop", "turn", "river"]:
        policy = policies[round_name]
        sample_infosets = list(policy.policy.keys())[:2]
        print(f"\n  {round_name.capitalize()}:")
        for infoset in sample_infosets:
            actions = policy.policy[infoset]
            print(f"    {infoset}: {actions}")

    print("\n✅ Full 4-chunk pipeline works!\n")

    return policies


def test_chunk_memory_usage():
    """Test memory profiling during 4-chunk solve."""
    print("\n" + "=" * 80)
    print("TEST 2: Memory Profiling Per Chunk")
    print("=" * 80)

    config = get_minimal_4round_config()

    print("\nCreating ChunkedSolver with MemoryProfiler...")
    profiler = MemoryProfiler()
    chunked = ChunkedSolver(full_game_config=config)

    # Take baseline snapshot
    profiler.snapshot("baseline")

    print("\nSolving chunks with memory tracking (10 iterations each)...")

    # Solve chunks manually to integrate profiling
    blueprint = None
    for chunk_name in chunked.chunks:
        print(f"\n{'='*80}")
        print(f"CHUNK: {chunk_name.upper()}")
        print(f"{'='*80}")

        # Snapshot before chunk
        profiler.snapshot(f"before_{chunk_name}")

        # Create and solve subgame
        subgame = SubgameSolver(
            full_game_config=config,
            round_name=chunk_name,
            blueprint_policy=blueprint
        )

        policy = subgame.solve(iterations=10, progress_interval=999)
        chunked.policies[chunk_name] = policy

        # Snapshot after chunk
        profiler.snapshot(f"after_{chunk_name}")

        # Feed forward
        blueprint = policy

        print(f"✓ {chunk_name} chunk complete")

    # Print memory report
    print("\n" + "=" * 80)
    print("MEMORY PROFILING RESULTS")
    print("=" * 80)
    profiler.print_report()

    # Verify reasonable memory usage
    snapshots = profiler.snapshots
    peak_cpu = max(s.cpu_mb for s in snapshots)
    peak_gpu = max((s.gpu_peak_mb for s in snapshots if s.gpu_peak_mb), default=None)

    print(f"\n🔍 Memory Analysis:")
    print(f"  Peak CPU: {peak_cpu:.1f} MB")
    if peak_gpu:
        print(f"  Peak GPU: {peak_gpu:.1f} MB")
        # Verify GPU usage is reasonable (< 8 GB for minimal config)
        assert peak_gpu < 8000, f"GPU usage too high: {peak_gpu:.1f} MB"

    print("\n✅ Memory profiling complete!\n")


def test_policy_save_load():
    """Test saving and loading all 4 chunk policies."""
    print("\n" + "=" * 80)
    print("TEST 3: Policy Save/Load")
    print("=" * 80)

    config = get_minimal_4round_config()

    print("\nSolving 4 chunks (10 iterations each)...")
    chunked = ChunkedSolver(full_game_config=config)
    policies = chunked.solve(iterations_per_chunk=10, progress_interval=999)

    print("\n📦 Testing save/load...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save all policies
        print(f"\nSaving policies to {tmpdir}...")
        chunked.save_policies(tmpdir)

        # Verify files exist
        for round_name in ["preflop", "flop", "turn", "river"]:
            filepath = os.path.join(tmpdir, f"{round_name}_policy.json")
            assert os.path.exists(filepath), f"Policy file not saved: {filepath}"
            print(f"  ✓ {round_name}_policy.json")

        # Load policies back
        print(f"\nLoading policies from {tmpdir}...")
        chunked2 = ChunkedSolver(full_game_config=config)
        chunked2.load_policies(tmpdir)

        # Verify all loaded
        for round_name in ["preflop", "flop", "turn", "river"]:
            assert round_name in chunked2.policies, f"{round_name} policy not loaded!"

            # Verify policy content matches
            original_size = len(policies[round_name].policy)
            loaded_size = len(chunked2.policies[round_name].policy)
            assert original_size == loaded_size, \
                f"{round_name} policy size mismatch: {original_size} != {loaded_size}"

            print(f"  ✓ {round_name}: {loaded_size} infosets")

    print("\n✅ Policy save/load works!\n")


def test_combined_policy():
    """Test CombinedPolicy interface for unified querying."""
    print("\n" + "=" * 80)
    print("TEST 4: CombinedPolicy Interface")
    print("=" * 80)

    config = get_minimal_4round_config()

    print("\nSolving 4 chunks (10 iterations each)...")
    chunked = ChunkedSolver(full_game_config=config)
    policies = chunked.solve(iterations_per_chunk=10, progress_interval=999)

    print("\n📦 Creating CombinedPolicy...")
    combined = CombinedPolicy(policies)
    print(f"✓ CombinedPolicy created: {combined}")

    # Test infoset statistics
    total_infosets = combined.get_total_infosets()
    by_round = combined.get_infosets_by_round()

    print(f"\n📊 Combined Policy Statistics:")
    print(f"  Total infosets: {total_infosets}")
    for round_name in ["preflop", "flop", "turn", "river"]:
        count = by_round.get(round_name, 0)
        print(f"    {round_name.capitalize():<8} {count:>6} infosets")

    # Test querying
    print(f"\n🔍 Testing policy queries...")
    for round_name in ["preflop", "flop", "turn", "river"]:
        policy = policies[round_name]
        if len(policy.policy) > 0:
            # Get a sample infoset
            sample_infoset = list(policy.policy.keys())[0]

            # Query via CombinedPolicy
            probs = combined.get_action_probs(sample_infoset, round_name)

            assert probs is not None, f"Query failed for {round_name}"
            print(f"  ✓ {round_name}: {sample_infoset[:20]}... → {len(probs)} actions")

    # Test save/load
    print(f"\n💾 Testing CombinedPolicy save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        combined.save(tmpdir)
        print(f"  ✓ Saved to {tmpdir}")

        # Load
        loaded = CombinedPolicy.load(tmpdir)
        print(f"  ✓ Loaded: {loaded}")

        # Verify
        assert loaded.get_total_infosets() == total_infosets, "Loaded policy mismatch!"
        print(f"  ✓ Verified: {loaded.get_total_infosets()} infosets")

    print("\n✅ CombinedPolicy interface works!\n")


if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 8.5: FULL 4-CHUNK PIPELINE TEST")
    print("=" * 80)
    print("\nValidating complete chunked Hold'em solving...")
    print()

    # Run tests
    try:
        # Test 1: Full pipeline
        policies = test_four_chunk_solve()

        # Test 2: Memory profiling
        test_chunk_memory_usage()

        # Test 3: Save/load
        test_policy_save_load()

        # Test 4: CombinedPolicy
        test_combined_policy()

        print("=" * 80)
        print("🎉 ALL PHASE 8.5 TESTS PASSED!")
        print("=" * 80)
        print("\nPhase 8.5 COMPLETE:")
        print("  ✓ Full 4-chunk pipeline operational")
        print("  ✓ Memory profiling per chunk working")
        print("  ✓ Policy save/load functional")
        print("  ✓ CombinedPolicy interface working")
        print("\nNext Steps:")
        print("  - Add validation tests")
        print("  - Document results")
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
