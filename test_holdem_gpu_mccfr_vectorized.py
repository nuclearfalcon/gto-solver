"""
Test vectorized GPU MCCFR solver (Phase 10.4).

This test validates the jax.vmap-based trajectory sampling implementation
and measures end-to-end speedup vs sequential MCCFR.

**IMPORTANT**: Requires activating OpenSpiel virtual environment first:
    source ~/open_spiel/venv/bin/activate

Phase 10.4 Goal: Achieve 20-50× end-to-end MCCFR speedup using GPU-parallel
trajectory sampling with jax.vmap.

Tests:
1. Sequential MCCFR baseline (batch_size=1)
2. Vectorized MCCFR (batch_size=100)
3. Batch size scaling (10, 50, 100, 250)
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver, MCCFRConfig
from matrix_cfr import holdem_jax_v2


# Configuration
NUM_PLAYERS = 2
STACKS = jnp.array([1000, 1000])
BLINDS = jnp.array([50, 100])
NUM_ACTIONS = 4  # fold, call, pot, all-in


def test_sequential_mccfr(num_iterations: int = 100):
    """
    Test 1: Sequential MCCFR (batch_size=1) - baseline for comparison.
    """
    print("="*70)
    print(f"Test 1: Sequential MCCFR ({num_iterations} iterations)")
    print("="*70)
    print()

    print("Creating solver (batch_size=1)...")
    config = MCCFRConfig(
        num_players=NUM_PLAYERS,
        num_actions=NUM_ACTIONS,
        batch_size=1,  # Sequential mode
        discount_factor=1.0,
        use_linear_weighting=False
    )

    solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=42)
    print(f"✓ Solver created")
    print()

    print(f"Running {num_iterations} iterations...")
    start = time.time()

    for i in range(num_iterations):
        solver.run_iteration(NUM_PLAYERS, STACKS, BLINDS)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            speed = (i + 1) / elapsed
            print(f"  [{i+1}/{num_iterations}] {speed:.2f} it/s")

    elapsed = time.time() - start
    speed = num_iterations / elapsed

    print()
    print("="*70)
    print("Sequential MCCFR Results")
    print("="*70)
    print(f"  Iterations: {num_iterations}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {speed:.2f} it/s")
    print(f"  Infosets visited (P0): {len(solver.regret_tables[0].cumulative_regrets)}")
    print(f"  Infosets visited (P1): {len(solver.regret_tables[1].cumulative_regrets)}")
    print()

    return speed


def test_vectorized_mccfr(num_iterations: int = 100, batch_size: int = 100):
    """
    Test 2: Vectorized MCCFR with jax.vmap (Phase 10.4).
    """
    print("="*70)
    print(f"Test 2: Vectorized MCCFR ({num_iterations} iterations, batch_size={batch_size})")
    print("="*70)
    print()

    print(f"Creating solver (batch_size={batch_size})...")
    config = MCCFRConfig(
        num_players=NUM_PLAYERS,
        num_actions=NUM_ACTIONS,
        batch_size=batch_size,  # Vectorized mode
        discount_factor=1.0,
        use_linear_weighting=False
    )

    solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=42)
    print(f"✓ Solver created")
    print()

    print("[Warmup - JIT compilation...]")
    print("(First iteration compiles the vectorized sampler)")
    warmup_start = time.time()
    solver.run_iteration(NUM_PLAYERS, STACKS, BLINDS)
    warmup_time = time.time() - warmup_start
    print(f"✓ Warmup complete ({warmup_time:.1f}s)")
    print()

    print(f"Running {num_iterations} iterations...")
    start = time.time()

    for i in range(num_iterations):
        solver.run_iteration(NUM_PLAYERS, STACKS, BLINDS)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            speed = (i + 1) / elapsed
            print(f"  [{i+1}/{num_iterations}] {speed:.2f} it/s")

    elapsed = time.time() - start
    speed = num_iterations / elapsed

    print()
    print("="*70)
    print("Vectorized MCCFR Results")
    print("="*70)
    print(f"  Iterations: {num_iterations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {speed:.2f} it/s")
    print(f"  Warmup time: {warmup_time:.1f}s")
    print(f"  Infosets visited (P0): {len(solver.regret_tables[0].cumulative_regrets)}")
    print(f"  Infosets visited (P1): {len(solver.regret_tables[1].cumulative_regrets)}")
    print()

    return speed


def test_batch_size_scaling(batch_sizes: list = [10, 50, 100, 250], iterations_per_test: int = 50):
    """
    Test 3: Measure speedup across different batch sizes.
    """
    print("="*70)
    print(f"Test 3: Batch Size Scaling ({iterations_per_test} iterations each)")
    print("="*70)
    print()

    # Get sequential baseline first
    print("Running sequential baseline...")
    config_seq = MCCFRConfig(
        num_players=NUM_PLAYERS,
        num_actions=NUM_ACTIONS,
        batch_size=1,
        discount_factor=1.0,
        use_linear_weighting=False
    )
    solver_seq = GPUMCCFRSolver(config_seq, holdem_jax_v2, seed=42)

    start = time.time()
    for _ in range(iterations_per_test):
        solver_seq.run_iteration(NUM_PLAYERS, STACKS, BLINDS)
    baseline_time = time.time() - start
    baseline_speed = iterations_per_test / baseline_time

    print(f"✓ Sequential baseline: {baseline_speed:.2f} it/s")
    print()

    # Test each batch size
    results = []

    for batch_size in batch_sizes:
        print(f"Testing batch_size={batch_size}...")

        config = MCCFRConfig(
            num_players=NUM_PLAYERS,
            num_actions=NUM_ACTIONS,
            batch_size=batch_size,
            discount_factor=1.0,
            use_linear_weighting=False
        )
        solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=42)

        # Warmup
        solver.run_iteration(NUM_PLAYERS, STACKS, BLINDS)

        # Benchmark
        start = time.time()
        for _ in range(iterations_per_test):
            solver.run_iteration(NUM_PLAYERS, STACKS, BLINDS)
        elapsed = time.time() - start
        speed = iterations_per_test / elapsed
        speedup = speed / baseline_speed

        results.append({
            'batch_size': batch_size,
            'speed': speed,
            'speedup': speedup,
            'time': elapsed
        })

        print(f"  Speed: {speed:.2f} it/s | Speedup: {speedup:.2f}×")

    print()
    print("="*70)
    print("Batch Size Scaling Results")
    print("="*70)
    print(f"  Sequential baseline: {baseline_speed:.2f} it/s")
    print()
    print(f"  {'Batch Size':<12} {'Speed':<12} {'Speedup':<12} {'Time':<12}")
    print(f"  {'-'*11} {'-'*11} {'-'*11} {'-'*11}")

    for r in results:
        print(f"  {r['batch_size']:<12} {r['speed']:<12.2f} {r['speedup']:<12.2f}× {r['time']:<12.2f}s")

    print()
    print(f"  Best speedup: {max(r['speedup'] for r in results):.2f}× at batch_size={max(results, key=lambda r: r['speedup'])['batch_size']}")
    print()

    return results


def quick_comparison(num_iterations: int = 50):
    """
    Quick comparison for Phase 10.4 validation.
    """
    print("="*70)
    print("PHASE 10.4 - QUICK VALIDATION")
    print("="*70)
    print()

    # Sequential
    print("1. Sequential MCCFR (batch_size=1)")
    seq_speed = test_sequential_mccfr(num_iterations)

    print("\n")

    # Vectorized
    print("2. Vectorized MCCFR (batch_size=100)")
    vec_speed = test_vectorized_mccfr(num_iterations, batch_size=100)

    print("\n")

    # Summary
    speedup = vec_speed / seq_speed

    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Sequential: {seq_speed:.2f} it/s")
    print(f"  Vectorized (batch=100): {vec_speed:.2f} it/s")
    print(f"  **SPEEDUP: {speedup:.2f}×**")
    print()

    if speedup >= 20.0:
        print(f"  ✓ SUCCESS! Achieved ≥20× speedup (goal met)")
    elif speedup >= 10.0:
        print(f"  ⚠ PARTIAL SUCCESS: {speedup:.2f}× speedup (below 20× goal)")
    elif speedup >= 2.0:
        print(f"  ⚠ MODEST IMPROVEMENT: {speedup:.2f}× speedup")
    else:
        print(f"  ✗ NO SIGNIFICANT SPEEDUP: {speedup:.2f}×")

    print()
    print("="*70)
    print()

    return speedup


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*22 + "PHASE 10.4 VALIDATION" + " "*26 + "║")
    print("║" + " "*16 + "Vectorized GPU MCCFR with jax.vmap" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    print()

    # Skip sequential baseline - use known baseline of 0.17-0.27 it/s
    BASELINE_SPEED = 0.22  # Average of previous runs

    print("Baseline (from previous runs): 0.17-0.27 it/s (avg: 0.22 it/s)")
    print()

    # Test vectorized only
    vec_speed = test_vectorized_mccfr(num_iterations=50, batch_size=100)

    speedup = vec_speed / BASELINE_SPEED

    print("\n")
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Baseline (known): {BASELINE_SPEED:.2f} it/s")
    print(f"  Vectorized (batch=100): {vec_speed:.2f} it/s")
    print(f"  **SPEEDUP: {speedup:.2f}×**")
    print()

    if speedup >= 20.0:
        print(f"  ✓ SUCCESS! Achieved ≥20× speedup (goal met)")
    elif speedup >= 10.0:
        print(f"  ⚠ PARTIAL SUCCESS: {speedup:.2f}× speedup (below 20× goal)")
    elif speedup >= 2.0:
        print(f"  ⚠ MODEST IMPROVEMENT: {speedup:.2f}× speedup")
    else:
        print(f"  ✗ NO SIGNIFICANT SPEEDUP: {speedup:.2f}×")

    print()
    print("="*70)
    print()

    print("\nPhase 10.4 Status:")
    if speedup >= 20.0:
        print("  ✅ COMPLETE - Target speedup achieved!")
    else:
        print(f"  🔄 IN PROGRESS - Current speedup: {speedup:.2f}×")
    print()
