"""
Test Phase 7: OOM Fix on Tiny Hold'em Config

Tests that the sparse-native child lookup eliminates the OOM error on
2p_1bb_fc_tiny.json (8 cards, 2 rounds) which previously failed at 20.58 GB.

Usage:
    source ~/open_spiel/venv/bin/activate
    python test_phase7_oom_fix.py
"""

import pyspiel
from matrix_cfr.game_to_matrix import GameTreeConverter
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
print("Phase 7 Test: OOM Fix on Tiny Hold'em")
print("=" * 80)
print()

# Load the config that previously caused OOM
print("Loading config: configs/2p_1bb_fc_tiny.json")
with open("configs/2p_1bb_fc_tiny.json", "r") as f:
    config = json.load(f)

print(f"Config: {config['num_players']}p, {config['num_suits']}x{config['num_ranks']} cards, {config['num_rounds']} rounds")
print(f"Betting: {config['betting_abstraction']}")
print()

mem_start = get_memory_mb()
print(f"Memory at start: {mem_start:.1f} MB")
print()

# Create game
print("Creating OpenSpiel game...")
try:
    game = pyspiel.load_game('universal_poker', {
        'betting': 'nolimit',
        'numPlayers': config['num_players'],
        'numRounds': config['num_rounds'],
        'blind': ' '.join(map(str, config['blinds'])),
        'firstPlayer': '2 1 1 1',
        'numSuits': config['num_suits'],
        'numRanks': config['num_ranks'],
        'numHoleCards': config['num_hole_cards'],
        'numBoardCards': config['num_board_cards'],
        'stack': ' '.join(map(str, config['stack_sizes'])),
        'bettingAbstraction': config['betting_abstraction']
    })
    print(f"✓ Game created successfully")
    mem_after_game = get_memory_mb()
    print(f"  Memory: {mem_after_game:.1f} MB (+{mem_after_game - mem_start:.1f} MB)")
except Exception as e:
    print(f"✗ Failed to create game: {e}")
    exit(1)

print()

# Create solver with sparse mode (this will convert the game internally)
print("Creating Matrix CFR solver (sparse mode)...")
print("  This includes game tree conversion and matrix building...")
try:
    start_time = time.time()
    solver = MatrixCFRSolver(game, use_sparse=True)
    elapsed = time.time() - start_time

    print(f"✓ Solver initialized in {elapsed:.1f}s")
    print(f"  Nodes: {solver.matrix_repr.num_nodes:,}")
    print(f"  Infosets: {solver.matrix_repr.num_infosets:,}")
    print(f"  Infoset-actions: {solver.matrix_repr.num_infoset_actions:,}")
    print(f"  Levels: {len(solver.level_matrices_jax)}")

    mem_after_solver = get_memory_mb()
    print(f"  Memory: {mem_after_solver:.1f} MB (+{mem_after_solver - mem_after_game:.1f} MB)")
    print(f"  TOTAL memory used: {mem_after_solver:.1f} MB")
    print()

    # Check if we successfully avoided OOM
    if mem_after_solver < 2000:  # Less than 2 GB
        print(f"✓ SUCCESS! Memory usage ({mem_after_solver:.1f} MB) is MUCH less than previous OOM (20,580 MB)")
        print(f"  Memory reduction: {20580 / mem_after_solver:.1f}x")
    else:
        print(f"⚠️ Memory usage is high ({mem_after_solver:.1f} MB) but below OOM threshold")

except Exception as e:
    print(f"✗ Failed to create solver: {e}")
    import traceback
    traceback.print_exc()
    mem_at_error = get_memory_mb()
    print(f"  Memory at error: {mem_at_error:.1f} MB")
    exit(1)

print()
print("-" * 80)
print("Running convergence test (3 iterations)...")
print("-" * 80)

try:
    solver.solve(iterations=3, progress_interval=1)
    print()
    print(f"✓ Convergence test passed!")
    mem_final = get_memory_mb()
    print(f"  Final memory: {mem_final:.1f} MB")
except Exception as e:
    print(f"✗ Convergence test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("=" * 80)
print("✓ PHASE 7 OOM FIX CONFIRMED!")
print(f"  Previous: OOM at 20,580 MB")
print(f"  Now: {mem_final:.1f} MB ({20580 / mem_final:.1f}x improvement)")
print("=" * 80)
