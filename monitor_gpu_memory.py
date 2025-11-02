"""
GPU Memory Monitor for Matrix CFR

Tracks VRAM usage throughout solver initialization and iteration to identify
memory bottlenecks and unexpected allocations.

Usage:
    source ~/open_spiel/venv/bin/activate
    python monitor_gpu_memory.py
"""

import pyspiel
from matrix_cfr.matrix_cfr_solver import MatrixCFRSolver
import json
import time
import subprocess
import threading

def get_gpu_memory():
    """Get current GPU memory usage in MB using nvidia-smi"""
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

def monitor_gpu_continuous(interval=0.5, duration=None):
    """Monitor GPU memory continuously in background"""
    measurements = []
    start_time = time.time()

    def monitor_loop():
        while True:
            if duration and (time.time() - start_time) > duration:
                break
            used, total = get_gpu_memory()
            if used is not None:
                timestamp = time.time() - start_time
                measurements.append((timestamp, used))
            time.sleep(interval)

    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    return measurements

def print_memory_snapshot(label, measurements_before):
    """Print memory usage snapshot"""
    used, total = get_gpu_memory()
    if used is not None:
        print(f"{label}")
        print(f"  GPU Memory: {used:,} MB / {total:,} MB ({100*used/total:.1f}% used)")
        if measurements_before:
            prev_used = measurements_before[-1][1]
            delta = used - prev_used
            if delta > 0:
                print(f"  Delta: +{delta:,} MB")
        return used
    return None

print("=" * 80)
print("GPU Memory Monitor: Matrix CFR Solver")
print("=" * 80)
print()

# Check GPU availability
used, total = get_gpu_memory()
if used is None:
    print("❌ No GPU detected or nvidia-smi not available")
    exit(1)

print(f"GPU detected: {total:,} MB total VRAM")
print()

# Test configurations
test_configs = [
    ("Kuhn poker", "kuhn_poker", None, 100),
    ("Leduc poker", "leduc_poker", None, 10),
    ("Preflop minimal", None, "configs/2p_preflop_only_minimal.json", 10),
]

for game_name, game_id, config_file, iterations in test_configs:
    print("=" * 80)
    print(f"Testing: {game_name}")
    print("=" * 80)
    print()

    # Baseline
    print("Stage 0: Baseline (before game creation)")
    baseline_used = print_memory_snapshot("", [])
    measurements = [(0, baseline_used)]
    print()

    # Create game
    print("Stage 1: Creating game...")
    if config_file:
        with open(config_file, "r") as f:
            config = json.load(f)
        game = pyspiel.load_game('universal_poker', {
            'betting': 'nolimit',
            'numPlayers': config['num_players'],
            'numRounds': config['num_rounds'],
            'blind': ' '.join(map(str, config['blinds'])),
            'firstPlayer': '2 1' if config['num_rounds'] == 1 else '2 1 1 1',
            'numSuits': config['num_suits'],
            'numRanks': config['num_ranks'],
            'numHoleCards': config['num_hole_cards'],
            'numBoardCards': config['num_board_cards'],
            'stack': ' '.join(map(str, config['stack_sizes'])),
            'bettingAbstraction': config['betting_abstraction']
        })
    else:
        game = pyspiel.load_game(game_id)

    time.sleep(0.5)  # Let memory settle
    print_memory_snapshot("After game creation", measurements)
    print()

    # Create solver (this triggers matrix building and GPU transfer)
    print("Stage 2: Creating solver (matrix conversion + GPU transfer)...")
    start_time = time.time()
    solver = MatrixCFRSolver(game, use_sparse=True)
    init_time = time.time() - start_time

    time.sleep(0.5)  # Let memory settle
    print(f"Initialization took {init_time:.1f}s")
    print_memory_snapshot("After solver initialization", measurements)
    print(f"  Game info: {solver.matrix_repr.num_nodes:,} nodes, "
          f"{solver.matrix_repr.num_infosets:,} infosets")
    print()

    # First iteration (triggers JIT compilation)
    print("Stage 3: First iteration (JIT compilation)...")
    start_time = time.time()
    solver.solve(iterations=1, progress_interval=999)
    first_iter_time = time.time() - start_time

    time.sleep(0.5)  # Let memory settle
    print(f"First iteration took {first_iter_time:.1f}s")
    after_first = print_memory_snapshot("After first iteration (JIT compiled)", measurements)
    print()

    # Subsequent iterations (compiled code)
    print(f"Stage 4: Running {iterations-1} more iterations...")
    start_time = time.time()
    solver.solve(iterations=iterations-1, progress_interval=999)
    remaining_time = time.time() - start_time

    time.sleep(0.5)  # Let memory settle
    print(f"Remaining {iterations-1} iterations took {remaining_time:.1f}s "
          f"({(iterations-1)/remaining_time:.2f} it/s)")
    after_iters = print_memory_snapshot("After all iterations", measurements)
    print()

    # Summary
    print("-" * 80)
    print("Memory allocation breakdown:")
    print(f"  Baseline: {baseline_used:,} MB")
    print(f"  + Solver init: +{after_first - baseline_used:,} MB")
    print(f"  + First iteration (JIT): +{0:,} MB (included in init)")
    print(f"  + Remaining iterations: +{after_iters - after_first:,} MB")
    print(f"  Total used: {after_iters:,} MB")
    print(f"  Available: {total - after_iters:,} MB remaining")
    print()

    # Clean up
    del solver
    del game
    import gc
    gc.collect()

    # Try to clear JAX cache
    try:
        import jax
        jax.clear_caches()
    except:
        pass

    time.sleep(2)  # Wait for cleanup

    print("Stage 5: After cleanup...")
    final_used = print_memory_snapshot("After cleanup (should be near baseline)", measurements)
    if final_used > baseline_used + 100:  # More than 100 MB retained
        print(f"  ⚠️ Warning: {final_used - baseline_used} MB not freed (possible memory leak)")
    print()

print("=" * 80)
print("GPU Memory Monitoring Complete")
print("=" * 80)
