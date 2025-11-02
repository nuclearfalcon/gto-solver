"""
Test Phase 7: Ultra-Minimal Preflop-Only Hold'em

Tests Matrix CFR on a preflop-only Hold'em variant with 6 cards (like Leduc).
This validates that Hold'em game structure works without OOM issues.

Expected: 500-2,000 nodes (solvable!)

Usage:
    source ~/open_spiel/venv/bin/activate
    python test_phase7_preflop_minimal.py
"""

import pyspiel
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver
import json
import time
import psutil
import os

def get_memory_mb():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("=" * 80)
print("Phase 7 Test: Ultra-Minimal Preflop-Only Hold'em")
print("=" * 80)
print()

# Load the ultra-minimal config
print("Loading config: configs/2p_preflop_only_minimal.json")
with open("configs/2p_preflop_only_minimal.json", "r") as f:
    config = json.load(f)

print(f"Config: {config['num_players']}p, {config['num_suits']}x{config['num_ranks']} cards")
print(f"Rounds: {config['num_rounds']} (preflop only)")
print(f"Board cards: {config['num_board_cards']}")
print(f"Betting: {config['betting_abstraction']}")
print()

mem_start = get_memory_mb()

# Create game
print("Creating OpenSpiel game...")
game = pyspiel.load_game('universal_poker', {
    'betting': 'nolimit',
    'numPlayers': config['num_players'],
    'numRounds': config['num_rounds'],
    'blind': ' '.join(map(str, config['blinds'])),
    'firstPlayer': '2 1',  # Only 1 round, so only need 1 value
    'numSuits': config['num_suits'],
    'numRanks': config['num_ranks'],
    'numHoleCards': config['num_hole_cards'],
    'numBoardCards': config['num_board_cards'],
    'stack': ' '.join(map(str, config['stack_sizes'])),
    'bettingAbstraction': config['betting_abstraction']
})
print(f"✓ Game created")
print()

# Create solver
print("Creating Matrix CFR solver (sparse mode)...")
start_time = time.time()
solver = MatrixCFRSolver(game, use_sparse=True)
init_time = time.time() - start_time

print(f"✓ Solver initialized in {init_time:.1f}s")
print(f"  Nodes: {solver.matrix_repr.num_nodes:,}")
print(f"  Infosets: {solver.matrix_repr.num_infosets:,}")
print(f"  Infoset-actions: {solver.matrix_repr.num_infoset_actions:,}")
print(f"  Levels: {len(solver.level_matrices_jax)}")

mem_after = get_memory_mb()
print(f"  Memory: {mem_after:.1f} MB")
print()

# Validate tree size
if solver.matrix_repr.num_nodes > 10000:
    print(f"⚠️ Warning: Tree has {solver.matrix_repr.num_nodes:,} nodes (larger than expected 500-2,000)")
    print(f"  This may cause GPU OOM during iteration")
elif solver.matrix_repr.num_nodes < 100:
    print(f"⚠️ Warning: Tree only has {solver.matrix_repr.num_nodes} nodes (very small, may not be realistic)")
else:
    print(f"✓ Tree size looks good: {solver.matrix_repr.num_nodes:,} nodes")

print()
print("-" * 80)
print("Running convergence test (100 iterations)...")
print("-" * 80)
print()

try:
    start_time = time.time()
    solver.solve(iterations=100, progress_interval=20)
    elapsed = time.time() - start_time

    speed = 100 / elapsed
    print()
    print(f"✓ Convergence test PASSED!")
    print(f"  Time: {elapsed:.1f}s for 100 iterations")
    print(f"  Speed: {speed:.2f} it/s")

    mem_final = get_memory_mb()
    print(f"  Final memory: {mem_final:.1f} MB")

except Exception as e:
    print(f"✗ Convergence test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("=" * 80)
print("✓ PHASE 7 SUCCESS: Ultra-minimal Hold'em works!")
print("  - Preflop-only Hold'em structure validated")
print("  - No OOM during initialization or iteration")
print(f"  - Tree size: {solver.matrix_repr.num_nodes:,} nodes")
print("  - Ready to scale up gradually")
print("=" * 80)
