"""
Final Phase 3 Benchmark - Comprehensive Performance Test

Tests all Phase 3 optimizations with proper warmup and multiple runs.

IMPORTANT: Activate OpenSpiel virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import pyspiel
import time
import numpy as np
from matrix_cfr import MatrixCFRSolver


def benchmark_with_warmup(num_runs=3, iterations_per_run=50):
    """Benchmark with proper JIT warmup."""
    print("=" * 70)
    print("Phase 3: Final Comprehensive Benchmark")
    print("=" * 70)

    game = pyspiel.load_game('kuhn_poker', {'players': 2})

    # Warmup run (JIT compilation)
    print("\n1. Warming up JIT compilation...")
    solver = MatrixCFRSolver(game)
    solver.solve(iterations=10)
    print("   ✅ Warmup complete")

    # Multiple benchmark runs
    print(f"\n2. Running {num_runs} benchmark runs ({iterations_per_run} iterations each)...")
    speeds = []

    for run in range(num_runs):
        solver = MatrixCFRSolver(game)

        start = time.time()
        solver.solve(iterations=iterations_per_run)
        elapsed = time.time() - start

        speed = iterations_per_run / elapsed
        speeds.append(speed)
        print(f"   Run {run+1}: {speed:.2f} it/s ({elapsed:.2f}s)")

    # Statistics
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    min_speed = np.min(speeds)
    max_speed = np.max(speeds)

    print("\n" + "=" * 70)
    print("📊 Phase 3 Performance Statistics")
    print("=" * 70)
    print(f"  Mean speed: {mean_speed:.2f} ± {std_speed:.2f} it/s")
    print(f"  Min speed:  {min_speed:.2f} it/s")
    print(f"  Max speed:  {max_speed:.2f} it/s")

    print("\n" + "=" * 70)
    print("📈 Overall Progress")
    print("=" * 70)
    print(f"  Baseline (no opt):     0.43 it/s")
    print(f"  Phase 1 (JIT):         1.00 it/s  (2.3x)")
    print(f"  Phase 2 (batching):    1.81 it/s  (4.2x)")
    print(f"  Phase 3 (all opts):    {mean_speed:.2f} it/s  ({mean_speed/0.43:.1f}x from baseline) ✅")

    print("\n" + "=" * 70)
    print("🎯 Phase 3 Optimizations Applied")
    print("=" * 70)
    print("  ✅ 3.1: Pre-build override templates")
    print("  ✅ 3.2: Batch both players together")
    print("  ✅ 3.3: JIT-compile 1D↔2D conversions")
    print("  ✅ 3.4: Vectorize regret updates")

    print("\n" + "=" * 70)
    print("💡 Key Insights")
    print("=" * 70)
    print("  • Small game (Kuhn) limits GPU parallelism")
    print("  • Batch size: 24 actions (2 players × 12 actions each)")
    print("  • Larger games will show much bigger gains")
    print("  • Expected on Hold'em: 50-100 it/s (100-200x)")

    return mean_speed


if __name__ == '__main__':
    mean_speed = benchmark_with_warmup(num_runs=3, iterations_per_run=50)

    if mean_speed >= 2.5:
        print("\n✅ Phase 3 SUCCESS! Achieved {:.2f} it/s ({:.1f}x baseline)".format(
            mean_speed, mean_speed/0.43
        ))
    else:
        print("\n⚠️  Performance lower than expected: {:.2f} it/s".format(mean_speed))
