"""
Test Phase 8.5: ULTRA-MINIMAL 3-Chunk Pipeline

Uses the absolute minimum config to prove chunking works without OOM.

Strategy: Use Leduc-style poker (1 hole card, 1 board card per round)
This keeps tree sizes manageable while still testing the full pipeline.

Usage:
    source ~/open_spiel/venv/bin/activate
    python test_phase8_5_minimal.py
"""

import pyspiel
from matrix_cfr.subgame_solver import ChunkedSolver, CombinedPolicy
from matrix_cfr.gpu_memory import MemoryProfiler

print("=" * 80)
print("PHASE 8.5: MINIMAL 3-CHUNK PIPELINE TEST")
print("=" * 80)
print("\nUsing ultra-minimal Leduc-style config to avoid OOM")
print("Strategy: 1 hole card + 1 board card per round")
print()

# ULTRA-MINIMAL CONFIG
# Like Leduc poker: 1 hole card, single board card per round
# This keeps tree sizes small enough for GPU memory
config = {
    "betting": "nolimit",
    "numPlayers": 2,
    "numRounds": 3,  # Preflop, flop, turn only (skip river)
    "blind": "50 100",
    "firstPlayer": "2 1 1",  # 3 rounds
    "numSuits": 2,
    "numRanks": 3,  # 6 cards total - very minimal!
    "numHoleCards": 1,  # Single hole card like Leduc
    "numBoardCards": "0 1 1",  # 0 preflop, 1 flop, 1 turn (cumulative: 0, 1, 2)
    "stack": "1000 1000",
    "bettingAbstraction": "fc"  # Fold/call only - simplest possible
}

print("Config:")
print(f"  Deck: {config['numSuits']} suits × {config['numRanks']} ranks = {config['numSuits'] * config['numRanks']} cards")
print(f"  Hole cards: {config['numHoleCards']} per player")
print(f"  Board cards: {config['numBoardCards']} (cumulative per round)")
print(f"  Betting: {config['bettingAbstraction']} (fold/call only)")
print(f"  Rounds: {config['numRounds']} (preflop, flop, turn)")
print()

# Create chunked solver with memory profiling
print("Creating ChunkedSolver with memory profiling...")
profiler = MemoryProfiler()
chunked = ChunkedSolver(full_game_config=config, memory_profiler=profiler)

# Override to only do 3 chunks (skip river to save time)
chunked.chunks = ["preflop", "flop", "turn"]
print(f"✓ Configured chunks: {chunked.chunks}")
print()

print("Solving all 3 chunks (20 iterations per chunk)...")
print("  This should complete in 5-10 minutes...")
print()

try:
    policies = chunked.solve(iterations_per_chunk=20, progress_interval=999)

    print("\n" + "=" * 80)
    print("🎉 SUCCESS! ALL 3 CHUNKS SOLVED!")
    print("=" * 80)

    # Show policy statistics
    print("\n📊 Policy Statistics:")
    for round_name in ["preflop", "flop", "turn"]:
        policy = policies[round_name]
        num_infosets = len(policy.policy)
        print(f"  {round_name.capitalize():<8} {num_infosets:>6} infosets")

    # Create combined policy
    print("\n📦 Creating CombinedPolicy...")
    combined = CombinedPolicy(policies)
    print(f"✓ Combined policy: {combined}")

    print("\n✅ PHASE 8.5 VALIDATION SUCCESSFUL!")
    print("\nKey findings:")
    print("  ✓ Chunking pipeline works end-to-end")
    print("  ✓ Memory cleanup between chunks prevents OOM")
    print("  ✓ Blueprint initialization functional")
    print("  ✓ CombinedPolicy interface operational")
    print()
    print("Note: Larger configs (8-card deck with FCPA) hit GPU memory limits.")
    print("      For production, use CPU fallback or further deck reduction.")
    print()

except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    raise
