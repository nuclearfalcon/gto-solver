#!/usr/bin/env python3
"""
Minimal test to verify memory leak is fixed.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python test_memory_fix.py
"""

import gc
import sys
import pyspiel
from open_spiel.python.policy import UniformRandomPolicy
from game_config import PokerGameConfig
from exploitability_metrics import SampledExploitabilityCalculator
from test_utils import get_memory_mb


def main():
    print("=" * 70)
    print("MEMORY LEAK FIX VERIFICATION TEST")
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

    # Test with ONLY 20 samples
    print(f"\n3. Running with only 20 samples...")
    calc = SampledExploitabilityCalculator(game, policy)

    result = calc.calculate(
        confidence_level=0.90,
        max_ci_width=1.0,      # Don't care about CI
        min_samples=10,
        max_samples=20,        # ONLY 20 samples
        check_interval=5,
        gc_interval=5
    )

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
    print(f"Exploitability: {result['exploitability']:.6f}")
    print(f"\nMemory increase: {mem_final - mem_start:.2f} MB")

    if mem_final - mem_start > 50:
        print("\n❌ FAIL: Memory increased by more than 50 MB!")
        print("There may still be a memory leak.")
        return 1
    else:
        print("\n✅ PASS: Memory usage is acceptable!")
        print("The memory leak has been fixed.")
        return 0


if __name__ == '__main__':
    sys.exit(main())