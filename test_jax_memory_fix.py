"""
Test JAX Memory Configuration Fixes

Tests different JAX memory allocation strategies to fix the 12 GB pre-allocation issue.

Usage:
    source ~/open_spiel/venv/bin/activate
    python test_jax_memory_fix.py
"""

import os
import sys

# MUST set BEFORE importing JAX
print("Setting JAX memory environment variables...")
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't pre-allocate
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use platform allocator
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'  # Use max 80% of VRAM
print("  XLA_PYTHON_CLIENT_PREALLOCATE=false")
print("  XLA_PYTHON_CLIENT_ALLOCATOR=platform")
print("  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8")
print()

import pyspiel
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver
import json
import time
import subprocess

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        used, total = result.stdout.strip().split(',')
        return int(used), int(total)
    except Exception as e:
        return None, None

print("=" * 80)
print("Test: JAX Memory Configuration Fix")
print("=" * 80)
print()

# Baseline
baseline_used, total = get_gpu_memory()
print(f"Baseline GPU memory: {baseline_used:,} MB / {total:,} MB")
print()

# Test Kuhn poker
print("-" * 80)
print("Test 1: Kuhn poker (58 nodes)")
print("-" * 80)

game = pyspiel.load_game('kuhn_poker')
before_solver = get_gpu_memory()[0]

solver = MatrixCFRSolver(game, use_sparse=True)
after_init = get_gpu_memory()[0]
print(f"After solver init: {after_init:,} MB (+{after_init - before_solver:,} MB)")

solver.solve(iterations=1, progress_interval=999)
after_first_iter = get_gpu_memory()[0]
print(f"After first iteration: {after_first_iter:,} MB (+{after_first_iter - after_init:,} MB)")

solver.solve(iterations=10, progress_interval=999)
after_iters = get_gpu_memory()[0]
print(f"After 10 iterations: {after_iters:,} MB (+{after_iters - after_first_iter:,} MB)")

del solver
del game
import gc
gc.collect()

import jax
jax.clear_caches()
time.sleep(1)

after_cleanup = get_gpu_memory()[0]
print(f"After cleanup: {after_cleanup:,} MB")
print(f"  Memory retained: {after_cleanup - baseline_used:,} MB")
print()

# Test Leduc poker
print("-" * 80)
print("Test 2: Leduc poker (9,457 nodes)")
print("-" * 80)

game = pyspiel.load_game('leduc_poker')
before_solver = get_gpu_memory()[0]

solver = MatrixCFRSolver(game, use_sparse=True)
after_init = get_gpu_memory()[0]
print(f"After solver init: {after_init:,} MB (+{after_init - before_solver:,} MB)")

solver.solve(iterations=1, progress_interval=999)
after_first_iter = get_gpu_memory()[0]
print(f"After first iteration: {after_first_iter:,} MB (+{after_first_iter - after_init:,} MB)")

del solver
del game
gc.collect()
jax.clear_caches()
time.sleep(1)

after_cleanup2 = get_gpu_memory()[0]
print(f"After cleanup: {after_cleanup2:,} MB")
print()

# Test tiny Hold'em
print("-" * 80)
print("Test 3: Tiny Hold'em (74,321 nodes) - The OOM test")
print("-" * 80)

with open("configs/2p_1bb_fc_tiny.json", "r") as f:
    config = json.load(f)

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

before_solver = get_gpu_memory()[0]
print(f"Before solver: {before_solver:,} MB")
print(f"Available: {total - before_solver:,} MB")

try:
    solver = MatrixCFRSolver(game, use_sparse=True)
    after_init = get_gpu_memory()[0]
    print(f"After solver init: {after_init:,} MB (+{after_init - before_solver:,} MB)")
    print(f"  Nodes: {solver.matrix_repr.num_nodes:,}")
    print(f"  Available: {total - after_init:,} MB")

    print("Attempting first iteration...")
    solver.solve(iterations=1, progress_interval=999)
    after_iter = get_gpu_memory()[0]
    print(f"After iteration: {after_iter:,} MB (+{after_iter - after_init:,} MB)")
    print()
    print("✅ SUCCESS! Tiny Hold'em works with memory fix!")

except Exception as e:
    after_error = get_gpu_memory()[0]
    print(f"❌ Failed: {str(e)[:100]}")
    print(f"Memory at error: {after_error:,} MB")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"Total VRAM: {total:,} MB")
print(f"Baseline: {baseline_used:,} MB")
print(f"After Kuhn+Leduc: {after_cleanup2:,} MB")
print(f"Peak usage: {get_gpu_memory()[0]:,} MB")
print(f"Memory management: {'✅ IMPROVED' if after_cleanup2 < baseline_used + 5000 else '❌ STILL LEAKING'}")
