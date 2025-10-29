#!/usr/bin/env python3
"""
Test with 100 samples to verify memory stays stable.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python test_memory_100_samples.py
"""

import gc
import sys
import time
import pyspiel
from open_spiel.python.policy import UniformRandomPolicy
from game_config import PokerGameConfig
from exploitability_metrics import SampledExploitabilityCalculator
from test_utils import get_memory_mb


def main():
    print("=" * 70)
    print("100-SAMPLE MEMORY STABILITY TEST")
    print("=" * 70)

    # Initial memory
    mem_start = get_memory_mb()
    print(f"\n1. Initial memory: {mem_start:.2f} MB")

    # Load game
    config = PokerGameConfig.from_json("configs/2p_5bb_fchpa_tiny.json")
    game = config.create_game()
    policy = UniformRandomPolicy(game)

    mem_setup = get_memory_mb()
    print(f"2. After setup: {mem_setup:.2f} MB (+{mem_setup - mem_start:.2f} MB)")

    # Test with 100 samples
    print(f"\n3. Running with 100 samples...")
    print("   This will take ~30 seconds...")

    start_time = time.time()
    calc = SampledExploitabilityCalculator(game, policy)

    result = calc.calculate(
        confidence_level=0.95,
        max_ci_width=0.1,      # 10% CI width
        min_samples=50,
        max_samples=100,       # 100 samples max
        check_interval=10,
        gc_interval=10
    )

    elapsed_time = time.time() - start_time

    mem_after = get_memory_mb()
    print(f"\n4. After calculation: {mem_after:.2f} MB (+{mem_after - mem_setup:.2f} MB)")

    # Force GC
    gc.collect()
    mem_final = get_memory_mb()
    print(f"5. After GC: {mem_final:.2f} MB")

    # Results
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Samples: {result['num_samples']}")
    print(f"Exploitability: {result['exploitability']:.4f}")
    print(f"CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"Time: {elapsed_time:.2f} seconds")
    print(f"\nMemory increase: {mem_final - mem_start:.2f} MB")

    if mem_final - mem_start > 50:
        print("\n❌ FAIL: Memory increased by more than 50 MB!")
        return 1
    else:
        print("\n✅ PASS: Memory usage is stable!")
        print(f"Successfully processed {result['num_samples']} samples with minimal memory usage.")
        return 0


if __name__ == '__main__':
    sys.exit(main())