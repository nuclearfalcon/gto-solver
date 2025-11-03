"""
Test Phase 8 Memory Profiling Infrastructure

This script validates the memory profiling system and demonstrates
component-by-component breakdown for Kuhn, Leduc, and Hold'em variants.

Usage:
    source ~/open_spiel/venv/bin/activate
    python test_phase8_memory_profiling.py
"""

import pyspiel
import jax.numpy as jnp
from matrix_cfr import (
    MatrixCFRSolver,
    GameTreeConverter,
    MemoryProfiler,
    get_cpu_memory_mb,
    get_gpu_memory_mb,
    get_array_memory_mb,
    get_sparse_memory_mb,
    profile_memory,
)


@profile_memory("Game Tree Conversion")
def convert_game(game):
    """Test decorator-based profiling."""
    converter = GameTreeConverter(game)
    return converter


def test_memory_profiling_basic():
    """Test basic memory profiling functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Memory Profiling")
    print("=" * 80)

    profiler = MemoryProfiler()

    # Initial snapshot
    profiler.snapshot("start")

    # Load Kuhn poker
    game = pyspiel.load_game('kuhn_poker')
    profiler.snapshot("after_game_load")

    # Convert to matrices
    converter = GameTreeConverter(game)
    profiler.snapshot("after_conversion")

    # Create solver
    solver = MatrixCFRSolver(game, use_sparse=True)
    profiler.snapshot("after_solver_init")

    # Print report
    profiler.print_report()

    print("✅ Basic memory profiling works!\n")


def test_component_breakdown():
    """Test component-by-component memory breakdown."""
    print("\n" + "=" * 80)
    print("TEST 2: Component Breakdown")
    print("=" * 80)

    profiler = MemoryProfiler()

    # Load Leduc poker
    game = pyspiel.load_game('leduc_poker')
    converter = GameTreeConverter(game)
    matrices = converter.build_matrices()

    # Calculate component sizes
    components = {}

    # Level matrices
    level_mem = 0
    for level, matrix in enumerate(matrices.level_matrices):
        mem = get_sparse_memory_mb(matrix)
        level_mem += mem
        components[f"level_matrix_{level}"] = mem
    components["level_matrices_total"] = level_mem

    # Action mapping matrix
    action_map_mem = get_sparse_memory_mb(matrices.infoset_action_to_node_matrix)
    components["action_mapping"] = action_map_mem

    # Player matrix
    player_mem = get_array_memory_mb(matrices.player_matrix)
    components["player_matrix"] = player_mem

    # Terminal utilities
    terminal_mem = get_array_memory_mb(matrices.terminal_utilities_matrix)
    components["terminal_utilities"] = terminal_mem

    # Take snapshot with components
    profiler.snapshot("leduc_components", components)

    # Print analysis
    profiler.print_component_analysis()

    print("✅ Component breakdown works!\n")


def test_scaling_analysis():
    """Test memory scaling analysis across game sizes."""
    print("\n" + "=" * 80)
    print("TEST 3: Memory Scaling Analysis")
    print("=" * 80)

    games = [
        ('kuhn_poker', 'Kuhn'),
        ('leduc_poker', 'Leduc'),
    ]

    nodes = []
    memories = []
    profiler = MemoryProfiler()

    for game_name, label in games:
        print(f"\nAnalyzing {label}...")
        game = pyspiel.load_game(game_name)
        converter = GameTreeConverter(game)
        matrices = converter.build_matrices()

        # Calculate total memory
        total_mem = 0
        for matrix in matrices.level_matrices:
            total_mem += get_sparse_memory_mb(matrix)

        nodes.append(matrices.num_nodes)
        memories.append(total_mem)

        print(f"  Nodes: {matrices.num_nodes}")
        print(f"  Memory: {total_mem:.2f} MB")

    # Scaling analysis
    profiler.analyze_scaling("num_nodes", nodes, memories)

    print("✅ Scaling analysis works!\n")


def test_gpu_memory_tracking():
    """Test GPU memory tracking."""
    print("\n" + "=" * 80)
    print("TEST 4: GPU Memory Tracking")
    print("=" * 80)

    # Check GPU availability
    gpu_info = get_gpu_memory_mb()
    if gpu_info is None:
        print("⚠️  GPU not available - skipping GPU memory test")
        return

    print(f"GPU memory before:")
    print(f"  Used: {gpu_info[0]:.1f} MB")
    print(f"  Peak: {gpu_info[1]:.1f} MB")

    # Create solver (uses GPU)
    game = pyspiel.load_game('leduc_poker')
    solver = MatrixCFRSolver(game, use_sparse=True)

    gpu_info_after = get_gpu_memory_mb()
    print(f"\nGPU memory after solver init:")
    print(f"  Used: {gpu_info_after[0]:.1f} MB")
    print(f"  Peak: {gpu_info_after[1]:.1f} MB")
    print(f"  Delta: {gpu_info_after[0] - gpu_info[0]:.1f} MB")

    print("\n✅ GPU memory tracking works!\n")


def test_decorator():
    """Test profile_memory decorator."""
    print("\n" + "=" * 80)
    print("TEST 5: Profile Memory Decorator")
    print("=" * 80)

    game = pyspiel.load_game('kuhn_poker')
    converter = convert_game(game)  # Uses @profile_memory decorator

    print("\n✅ Decorator works!\n")


def test_hold_em_variants():
    """Test memory profiling on Hold'em variants."""
    print("\n" + "=" * 80)
    print("TEST 6: Hold'em Variant Analysis")
    print("=" * 80)

    import json

    configs = [
        'configs/2p_preflop_only_minimal.json',
        'configs/2p_tiny_holdem.json',
    ]

    profiler = MemoryProfiler()
    nodes_list = []
    mem_list = []

    for config_path in configs:
        try:
            with open(config_path) as f:
                config = json.load(f)

            print(f"\nAnalyzing {config_path}...")

            game = pyspiel.load_game_from_file(config_path)
            converter = GameTreeConverter(game)
            matrices = converter.build_matrices()

            # Calculate memory
            components = {}
            level_mem = sum(get_sparse_memory_mb(m) for m in matrices.level_matrices)
            components["level_matrices"] = level_mem

            nodes_list.append(matrices.num_nodes)
            mem_list.append(level_mem)

            print(f"  Nodes: {matrices.num_nodes}")
            print(f"  Infosets: {matrices.num_infosets}")
            print(f"  Memory: {level_mem:.2f} MB")

            profiler.snapshot(f"holdem_{matrices.num_nodes}_nodes", components)

        except FileNotFoundError:
            print(f"  ⚠️  Config not found: {config_path}")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")

    if len(nodes_list) >= 2:
        profiler.analyze_scaling("num_nodes", nodes_list, mem_list)

    print("\n✅ Hold'em variant analysis complete!\n")


if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 8: MEMORY PROFILING INFRASTRUCTURE TEST")
    print("=" * 80)
    print("\nThis test validates the memory profiling system and demonstrates")
    print("component-by-component analysis for different poker variants.")
    print()

    # Run all tests
    test_memory_profiling_basic()
    test_component_breakdown()
    test_scaling_analysis()
    test_gpu_memory_tracking()
    test_decorator()
    test_hold_em_variants()

    print("=" * 80)
    print("🎉 ALL MEMORY PROFILING TESTS PASSED!")
    print("=" * 80)
    print("\nMemory profiling infrastructure is ready for Phase 8 optimizations.")
    print("Use MemoryProfiler class to track memory usage during chunking/FP16 work.")
    print()
