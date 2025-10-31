#!/usr/bin/env python3
"""
DCFR Research Validation: 3-Player Kuhn Poker (PARALLEL VERSION)

Parallel implementation using multiprocessing to run 5 solvers simultaneously.
Achieves ~5x speedup on multi-core systems with configurable CPU limiting.

Features:
    - True parallelism via ProcessPoolExecutor (bypasses Python GIL)
    - In-process CPU throttling (os.nice(10))
    - Optional external CPU limiting via cpulimit wrapper
    - Thread-safe CSV writing with file locking
    - Independent worker checkpoints for fault tolerance
    - Real-time progress monitoring from all workers
    - No pandas dependency (uses standard library only)

Tests the exact DCFR(α, β, γ) configurations from Brown & Sandholm research
to validate published claims about convergence performance.

Research Configurations Tested:
1. SIMPLE - Baseline external sampling
2. FULL - Reach-probability weighted external sampling
3. True LCFR - DCFR(1, 1, 1) - Original Linear CFR
4. SOTA DCFR - DCFR(1.5, 0, 2) - State-of-the-art (research best)
5. CFR+ Approximation - DCFR(∞, ∞, 2) - Quadratic averaging only

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    # Quick validation with parallel execution (100k iterations, ~2 minutes on 6+ cores)
    python compare_dcfr_research_3p_parallel.py --iterations 100000 --check-interval 10000

    # Full research validation (1M iterations, ~20 minutes on 6+ cores)
    python compare_dcfr_research_3p_parallel.py --iterations 1000000 --check-interval 50000

    # With CPU limiting wrapper - default 80% (recommended to prevent overheating)
    bash run_with_cpulimit.sh compare_dcfr_research_3p_parallel.py --iterations 1000000

    # With custom CPU limit (70%)
    bash run_with_cpulimit.sh --cpu-limit 70 compare_dcfr_research_3p_parallel.py --iterations 1000000

    # Conservative for laptops (50% CPU, 4 workers)
    bash run_with_cpulimit.sh -c 50 compare_dcfr_research_3p_parallel.py \\
        --iterations 1000000 --max-workers 4

    # With checkpointing for long runs
    python compare_dcfr_research_3p_parallel.py \\
        --iterations 1000000 \\
        --check-interval 50000 \\
        --checkpoint-interval 100000

Performance:
    - Sequential version: ~2 hours for 1M iterations
    - Parallel version: ~24 minutes for 1M iterations (5x speedup on 5+ cores)
    - Memory usage: ~5x more (all solvers active simultaneously)
    - Default workers: Half of available CPU cores (prevents overheating)
    - CPU limiting: Configurable via wrapper script (default 80%)

Cleanup:
    Press Ctrl+C to interrupt. The wrapper script includes:
    - Recursive process tree termination
    - Three-layer cleanup (graceful → forced → pattern-based)
    - Proper signal handling to prevent lingering processes

Reference:
    Brown & Sandholm. "Solving Imperfect-Information Games via Discounted Regret Minimization"
    AAAI 2019. https://arxiv.org/abs/1809.04040
"""

import argparse
import sys
import time
import os
import fcntl
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue
import pickle

import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr, exploitability

from linear_external_mccfr import LinearExternalSamplingSolver

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    print("\n\n⚠️  Interrupt received, shutting down gracefully...")
    print("Please wait while workers are terminated...")
    sys.stdout.flush()


def worker_run_solver(
    algo_key: str,
    algo_info: Dict[str, Any],
    iterations: int,
    start_iteration: int,
    check_interval: int,
    progress_interval: int,
    checkpoint_interval: int,
    checkpoint_dir: str,
    checkpoint_prefix: str,
    csv_path: str,
    progress_queue: Queue,
    worker_id: int,
    throttle_delay: float = 0.0
) -> Dict[str, Any]:
    """
    Worker function that runs a single solver in a separate process.

    Args:
        algo_key: Algorithm key (e.g., 'SIMPLE', 'SOTA_DCFR')
        algo_info: Algorithm configuration dict
        iterations: Total iterations to run
        start_iteration: Starting iteration (for checkpoint resume)
        check_interval: Calculate exploitability every N iterations
        progress_interval: Report progress every N iterations
        checkpoint_interval: Save checkpoint every N iterations (0 = disabled)
        checkpoint_dir: Directory for checkpoints
        checkpoint_prefix: Prefix for checkpoint files
        csv_path: Path to shared CSV file
        progress_queue: Queue for sending progress updates to main process
        worker_id: Unique worker ID for logging
        throttle_delay: Seconds to sleep after each exploitability check (thermal control)

    Returns:
        Dict with final metrics and status
    """
    try:
        # Set lower priority to reduce CPU contention
        os.nice(10)

        # Set up signal handling for graceful worker shutdown
        def worker_signal_handler(signum, frame):
            raise KeyboardInterrupt("Worker interrupted")

        signal.signal(signal.SIGINT, worker_signal_handler)
        signal.signal(signal.SIGTERM, worker_signal_handler)

        # Initialize game and solver
        game = pyspiel.load_game("kuhn_poker", {"players": 3})

        # Load checkpoint if resuming
        if start_iteration > 0:
            prefix = f"{checkpoint_prefix}_{algo_key}"
            checkpoint_file = f"{checkpoint_dir}/{prefix}_iter_{start_iteration}.pkl"
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            solver = checkpoint_data.get('solver', checkpoint_data)
        else:
            # Create solver using the create_fn from algo_info
            # Need to reconstruct the lambda since functions aren't picklable
            solver = _create_solver(game, algo_key, algo_info)

        # Track metrics
        metrics = []
        worker_start_time = time.time()
        min_nash_conv = float('inf')  # Track best (minimum) Nash seen
        min_nash_iteration = 0  # Track iteration where best Nash occurred

        # Main iteration loop
        for iteration in range(start_iteration, iterations + 1):
            solver.iteration()

            # Report progress to main process
            if iteration > 0 and (iteration == 1 or iteration % progress_interval == 0):
                progress_queue.put({
                    'worker_id': worker_id,
                    'algo_key': algo_key,
                    'iteration': iteration,
                    'type': 'progress'
                })

            # Check exploitability at intervals
            if iteration > 0 and iteration % check_interval == 0:
                elapsed = time.time() - worker_start_time
                policy = solver.average_policy()

                # Calculate Nash convergence
                try:
                    nash_conv = pyspiel.nash_conv(game, policy)
                except:
                    nash_conv = exploitability.nash_conv(game, policy, return_only_nash_conv=True)

                # Track minimum (best) Nash convergence
                if nash_conv < min_nash_conv:
                    min_nash_conv = nash_conv
                    min_nash_iteration = iteration

                    # Save "best" checkpoint when we find new minimum Nash
                    if checkpoint_interval:
                        _save_checkpoint_best(
                            solver, algo_key, iteration, min_nash_conv,
                            checkpoint_dir, checkpoint_prefix
                        )

                # Calculate iteration rate
                iters_since_start = iteration - start_iteration
                rate = iters_since_start / elapsed if elapsed > 0 else 0

                # Store metrics
                metric = {
                    'iteration': iteration,
                    'nash_conv': nash_conv,
                    'elapsed': elapsed,
                    'rate': rate
                }
                metrics.append(metric)

                # Write to CSV with file locking (thread-safe)
                _write_csv_locked(csv_path, [
                    iteration,
                    algo_info['name'],
                    algo_info['config'],
                    nash_conv,
                    elapsed,
                    rate
                ])

                # Send exploitability update to main process
                progress_queue.put({
                    'worker_id': worker_id,
                    'algo_key': algo_key,
                    'iteration': iteration,
                    'nash_conv': nash_conv,
                    'min_nash_conv': min_nash_conv,
                    'min_nash_iteration': min_nash_iteration,
                    'type': 'exploitability'
                })

                # Thermal throttling: sleep after exploitability check to reduce CPU heat
                if throttle_delay > 0:
                    time.sleep(throttle_delay)

            # Save checkpoints
            if checkpoint_interval and iteration > 0 and iteration % checkpoint_interval == 0:
                _save_checkpoint(
                    solver, algo_key, iteration,
                    checkpoint_dir, checkpoint_prefix
                )
                progress_queue.put({
                    'worker_id': worker_id,
                    'algo_key': algo_key,
                    'iteration': iteration,
                    'type': 'checkpoint'
                })

        # Worker complete
        total_time = time.time() - worker_start_time

        # Get final Nash convergence from last metric
        final_nash = metrics[-1]['nash_conv'] if metrics else None

        progress_queue.put({
            'worker_id': worker_id,
            'algo_key': algo_key,
            'type': 'complete',
            'metrics': metrics,
            'total_time': total_time,
            'final_nash': final_nash,
            'min_nash_conv': min_nash_conv if min_nash_conv != float('inf') else None,
            'min_nash_iteration': min_nash_iteration if min_nash_conv != float('inf') else None
        })

        return {
            'algo_key': algo_key,
            'status': 'success',
            'metrics': metrics,
            'total_time': total_time
        }

    except KeyboardInterrupt:
        # Graceful shutdown on interrupt
        progress_queue.put({
            'worker_id': worker_id,
            'algo_key': algo_key,
            'type': 'interrupted',
            'message': 'Worker interrupted by user'
        })
        return {
            'algo_key': algo_key,
            'status': 'interrupted',
            'message': 'Interrupted by user'
        }

    except Exception as e:
        import traceback
        error_msg = f"Worker {worker_id} ({algo_key}) failed: {str(e)}\n{traceback.format_exc()}"
        progress_queue.put({
            'worker_id': worker_id,
            'algo_key': algo_key,
            'type': 'error',
            'error': error_msg
        })
        return {
            'algo_key': algo_key,
            'status': 'error',
            'error': error_msg
        }


def _create_solver(game, algo_key: str, algo_info: Dict[str, Any]):
    """
    Create solver based on algorithm key.
    Needed because lambda functions aren't picklable for multiprocessing.
    """
    if algo_key == 'SIMPLE':
        return external_sampling_mccfr.ExternalSamplingSolver(
            game, average_type=external_sampling_mccfr.AverageType.SIMPLE
        )
    elif algo_key == 'FULL':
        return external_sampling_mccfr.ExternalSamplingSolver(
            game, average_type=external_sampling_mccfr.AverageType.FULL
        )
    elif algo_key == 'TRUE_LCFR':
        return LinearExternalSamplingSolver(
            game, gamma=1.0, alpha=1.0, beta=1.0
        )
    elif algo_key == 'SOTA_DCFR':
        return LinearExternalSamplingSolver(
            game, gamma=2.0, alpha=1.5, beta=0.0
        )
    elif algo_key == 'CFR_PLUS_APPROX':
        return LinearExternalSamplingSolver(
            game, gamma=2.0  # α=None, β=None
        )
    else:
        raise ValueError(f"Unknown algorithm key: {algo_key}")


def _write_csv_locked(csv_path: str, row: list):
    """Write a row to CSV with file locking for thread safety."""
    with open(csv_path, 'a', newline='') as f:
        # Acquire exclusive lock
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow(row)
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _save_checkpoint(solver, algo_key: str, iteration: int, checkpoint_dir: str, checkpoint_prefix: str):
    """Save checkpoint for solver."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    prefix = f"{checkpoint_prefix}_{algo_key}"
    filepath = f"{checkpoint_dir}/{prefix}_iter_{iteration}.pkl"

    checkpoint_data = {
        'solver': solver,
        'current_iteration': iteration,
        'algorithm': algo_key
    }

    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint_data, f, pickle.HIGHEST_PROTOCOL)


def _save_checkpoint_best(solver, algo_key: str, iteration: int, nash_conv: float, checkpoint_dir: str, checkpoint_prefix: str):
    """Save 'best' checkpoint when a new minimum Nash is found."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    prefix = f"{checkpoint_prefix}_{algo_key}"
    filepath = f"{checkpoint_dir}/{prefix}_best_iter_{iteration}_nash_{nash_conv:.6f}.pkl"

    checkpoint_data = {
        'solver': solver,
        'current_iteration': iteration,
        'algorithm': algo_key,
        'nash_conv': nash_conv,
        'is_best': True
    }

    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint_data, f, pickle.HIGHEST_PROTOCOL)


def _find_latest_checkpoint(checkpoint_dir: str, checkpoint_prefix: str, algo_key: str) -> Tuple[str, int]:
    """Find latest checkpoint file and iteration number."""
    import glob
    import re

    prefix = f"{checkpoint_prefix}_{algo_key}"
    pattern = f"{checkpoint_dir}/{prefix}_iter_*.pkl"
    files = glob.glob(pattern)
    if not files:
        return None, 0

    max_iter = -1
    latest = None
    for filepath in files:
        match = re.search(r'iter_(\d+)\.pkl$', filepath)
        if match:
            iteration = int(match.group(1))
            if iteration > max_iter:
                max_iter = iteration
                latest = filepath

    return latest, max_iter


class ParallelResearchValidationRunner:
    """Validates research-published DCFR configurations using parallel execution."""

    def __init__(
        self,
        iterations: int = 1000000,
        check_interval: int = 50000,
        progress_interval: int = 2000,
        checkpoint_interval: int = None,
        checkpoint_dir: str = "checkpoints",
        checkpoint_prefix: str = None,
        output_dir: str = "results",
        force_restart: bool = False,
        max_workers: int = None,
        throttle_delay: float = 0.0
    ):
        """Initialize parallel research validation runner."""
        self.iterations = iterations
        self.check_interval = check_interval
        self.progress_interval = progress_interval
        self.checkpoint_interval = checkpoint_interval or 0
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.force_restart = force_restart
        self.throttle_delay = throttle_delay

        # Default to half of available cores (more conservative, prevents overheating)
        if max_workers is None:
            default_workers = max(1, os.cpu_count() // 2)
            # Cap at number of algorithms (no benefit to more workers than algorithms)
            self.max_workers = min(default_workers, 5)
        else:
            self.max_workers = max_workers

        # Set lower priority for main process
        os.nice(10)

        # Generate checkpoint prefix
        if checkpoint_prefix is None:
            self.checkpoint_prefix = "3p_kuhn_dcfr_research"
        else:
            self.checkpoint_prefix = checkpoint_prefix

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Define research-validated algorithms
        self.algorithms = {
            'SIMPLE': {
                'name': 'SIMPLE',
                'config': 'External Sampling',
                'research': 'Baseline (no weighting)'
            },
            'FULL': {
                'name': 'FULL',
                'config': 'External Sampling + Reach-Prob Weighting',
                'research': 'Baseline (reach-prob weighting)'
            },
            'TRUE_LCFR': {
                'name': 'True LCFR',
                'config': 'DCFR(1, 1, 1)',
                'research': 'Original Linear CFR (Brown & Sandholm 2019)'
            },
            'SOTA_DCFR': {
                'name': 'SOTA DCFR',
                'config': 'DCFR(1.5, 0, 2)',
                'research': 'State-of-the-art (Brown & Sandholm 2019 - BEST)'
            },
            'CFR_PLUS_APPROX': {
                'name': 'CFR+ Approx',
                'config': 'DCFR(∞, ∞, 2)',
                'research': 'Quadratic averaging, no regret discounting'
            }
        }

        # Find checkpoint resume points
        self.start_iterations = {}

        print("\n" + "="*80)
        print("INITIALIZING PARALLEL DCFR RESEARCH VALIDATION")
        print("="*80)
        print(f"CPU cores available: {os.cpu_count()}")
        print(f"Max workers: {self.max_workers} processes (default: half of cores)")
        print(f"CPU priority: Lowered (nice +10)")

        for algo_key, algo_info in self.algorithms.items():
            # Check for existing checkpoint
            if not force_restart and checkpoint_interval:
                latest_checkpoint, max_iter = _find_latest_checkpoint(
                    checkpoint_dir, self.checkpoint_prefix, algo_key
                )
                if latest_checkpoint:
                    print(f"✓ {algo_info['name']:15} - Resuming from iteration {max_iter:,}")
                    self.start_iterations[algo_key] = max_iter
                else:
                    print(f"  {algo_info['name']:15} - Starting from scratch")
                    self.start_iterations[algo_key] = 0
            else:
                if force_restart:
                    print(f"⟳ {algo_info['name']:15} - Force restart")
                self.start_iterations[algo_key] = 0

        # CSV output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = Path(self.output_dir) / f"dcfr_research_validation_{timestamp}.csv"

        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'algorithm', 'config', 'nash_conv', 'wall_time_sec', 'iterations_per_sec'])

        print(f"\nResults: {self.csv_path}\n")

    def run(self):
        """Run the parallel research validation."""
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("="*80)
        print("DCFR RESEARCH VALIDATION: 3-PLAYER KUHN POKER (PARALLEL)")
        print("="*80)
        print(f"Total iterations: {self.iterations:,}")
        print(f"Check interval: {self.check_interval:,}")
        print(f"Progress interval: {self.progress_interval:,}")
        if self.checkpoint_interval:
            print(f"Checkpoint interval: {self.checkpoint_interval:,}")
        else:
            print("Checkpointing: DISABLED")

        print("\nResearch Configurations:")
        print("-" * 80)
        for algo_key, algo_info in self.algorithms.items():
            print(f"{algo_info['name']:15} | {algo_info['config']:25} | {algo_info['research']}")
        print("="*80 + "\n")

        # Create multiprocessing manager for shared queue
        manager = Manager()
        progress_queue = manager.Queue()

        # Track worker progress
        worker_progress = {algo_key: 0 for algo_key in self.algorithms.keys()}
        worker_exploitability = {}
        worker_status = {algo_key: 'ACTIVE' for algo_key in self.algorithms.keys()}
        worker_final_nash = {}  # Store final Nash for completed workers
        worker_min_nash = {}  # Store minimum (best) Nash seen for each worker
        worker_min_nash_iteration = {}  # Store iteration where min Nash occurred

        # Launch workers
        start_time = time.time()
        futures = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all workers
            for worker_id, (algo_key, algo_info) in enumerate(self.algorithms.items()):
                future = executor.submit(
                    worker_run_solver,
                    algo_key=algo_key,
                    algo_info=algo_info,
                    iterations=self.iterations,
                    start_iteration=self.start_iterations[algo_key],
                    check_interval=self.check_interval,
                    progress_interval=self.progress_interval,
                    checkpoint_interval=self.checkpoint_interval,
                    checkpoint_dir=self.checkpoint_dir,
                    checkpoint_prefix=self.checkpoint_prefix,
                    csv_path=str(self.csv_path),
                    progress_queue=progress_queue,
                    worker_id=worker_id,
                    throttle_delay=self.throttle_delay
                )
                futures[future] = algo_key

            print(f"Launched {len(futures)} parallel workers\n")

            # Monitor progress
            completed_workers = 0
            last_display = time.time()

            while completed_workers < len(self.algorithms):
                # Check for progress updates (non-blocking)
                try:
                    msg = progress_queue.get(timeout=0.1)

                    if msg['type'] == 'progress':
                        worker_progress[msg['algo_key']] = msg['iteration']

                    elif msg['type'] == 'exploitability':
                        worker_exploitability[msg['algo_key']] = {
                            'iteration': msg['iteration'],
                            'nash_conv': msg['nash_conv']
                        }
                        # Track minimum Nash
                        if 'min_nash_conv' in msg and msg['min_nash_conv'] is not None:
                            worker_min_nash[msg['algo_key']] = msg['min_nash_conv']
                            worker_min_nash_iteration[msg['algo_key']] = msg['min_nash_iteration']
                        # Nash values now shown in live display

                    elif msg['type'] == 'checkpoint':
                        # Checkpoint notifications now suppressed to keep display clean
                        pass

                    elif msg['type'] == 'complete':
                        completed_workers += 1
                        worker_status[msg['algo_key']] = 'COMPLETED'
                        if msg.get('final_nash') is not None:
                            worker_final_nash[msg['algo_key']] = msg['final_nash']
                        if msg.get('min_nash_conv') is not None:
                            worker_min_nash[msg['algo_key']] = msg['min_nash_conv']
                            worker_min_nash_iteration[msg['algo_key']] = msg['min_nash_iteration']
                        print(f"\n✓ {self.algorithms[msg['algo_key']]['name']:15} COMPLETED in {msg['total_time']:.1f}s")

                    elif msg['type'] == 'interrupted':
                        completed_workers += 1
                        worker_status[msg['algo_key']] = 'INTERRUPTED'
                        print(f"\n⚠  {self.algorithms[msg['algo_key']]['name']:15} INTERRUPTED")

                    elif msg['type'] == 'error':
                        completed_workers += 1
                        worker_status[msg['algo_key']] = 'FAILED'
                        print(f"\n✗ {self.algorithms[msg['algo_key']]['name']:15} FAILED:\n{msg['error']}")

                except:
                    # No message available, continue
                    pass

                # Check if shutdown was requested
                if shutdown_requested:
                    print("\n\n⚠️  Shutdown requested, terminating all workers...")
                    break

                # Display aggregate and individual progress every 2 seconds
                if time.time() - last_display > 2.0:
                    # Calculate progress only for active (non-completed) workers
                    active_worker_keys = [k for k in self.algorithms.keys() if worker_status[k] == 'ACTIVE']
                    active_progress = sum(worker_progress[k] for k in active_worker_keys)
                    active_start_iters = sum(self.start_iterations[k] for k in active_worker_keys)

                    # Total progress (all workers including completed)
                    total_progress = sum(worker_progress.values())
                    avg_progress = total_progress / len(worker_progress) if worker_progress else 0

                    elapsed = time.time() - start_time

                    # Rate: only from active workers (iterations completed since start / time)
                    active_iters_completed = active_progress - active_start_iters
                    rate = active_iters_completed / elapsed if elapsed > 0 else 0

                    # ETA: remaining work for active workers / current rate
                    remaining_work = self.iterations * len(active_worker_keys) - active_progress
                    eta = remaining_work / rate if rate > 0 else 0

                    active_workers = len(active_worker_keys)

                    # Clear screen and move to top (using ANSI escape codes)
                    print("\033[2J\033[H", end='')

                    # Display aggregate progress
                    print("="*80)
                    print(f"AGGREGATE: {avg_progress:7,.0f}/{self.iterations:,} avg | "
                          f"Rate: {rate:6,.0f} total it/s | ETA: {eta/60:5.1f}m | "
                          f"Active: {active_workers}/{len(self.algorithms)}")
                    print("="*80)
                    print()

                    # Display individual worker progress
                    for algo_key, algo_info in self.algorithms.items():
                        current_iter = worker_progress[algo_key]
                        status = worker_status[algo_key]

                        # Calculate per-worker iteration rate
                        start_iter = self.start_iterations[algo_key]
                        iters_completed = current_iter - start_iter
                        worker_rate = iters_completed / elapsed if elapsed > 0 else 0

                        # Build status string
                        if status == 'COMPLETED':
                            if algo_key in worker_final_nash and algo_key in worker_min_nash:
                                # Show both final and best (minimum) Nash for completed workers
                                final_nash = worker_final_nash[algo_key]
                                min_nash = worker_min_nash[algo_key]
                                min_iter = worker_min_nash_iteration[algo_key]
                                status_str = f"✓ COMPLETED (Final: {final_nash:.6f}, Best: {min_nash:.6f} @{min_iter:,})"
                            elif algo_key in worker_final_nash:
                                final_nash = worker_final_nash[algo_key]
                                status_str = f"✓ COMPLETED (Nash: {final_nash:.6f})"
                            else:
                                status_str = "✓ COMPLETED"
                        elif status == 'INTERRUPTED':
                            status_str = "⚠ INTERRUPTED"
                        elif status == 'FAILED':
                            status_str = "✗ FAILED"
                        else:
                            # Check if worker has actually started (made progress)
                            if iters_completed == 0 and elapsed > 1.0:
                                # No progress yet = queued (waiting for worker slot)
                                status_str = "⏳ Queued"
                            elif algo_key in worker_exploitability:
                                # Show current Nash and best (minimum) Nash if available
                                nash = worker_exploitability[algo_key]['nash_conv']
                                if algo_key in worker_min_nash:
                                    min_nash = worker_min_nash[algo_key]
                                    min_iter = worker_min_nash_iteration[algo_key]
                                    status_str = f"Nash: {nash:.6f} (Best: {min_nash:.6f} @{min_iter:,})"
                                else:
                                    status_str = f"Nash: {nash:.6f}"
                            else:
                                # Running but no Nash yet
                                status_str = "Working..."

                        # Calculate progress percentage
                        pct = (current_iter / self.iterations * 100) if self.iterations > 0 else 0

                        # Display worker line with rate
                        print(f"{algo_info['name']:15} | {current_iter:7,}/{self.iterations:,} ({pct:5.1f}%) | "
                              f"{worker_rate:6.0f} it/s | {status_str}")

                    print()
                    sys.stdout.flush()
                    last_display = time.time()

            # Shutdown handling
            if shutdown_requested:
                print("\n⚠️  Cancelling remaining workers...")
                # Cancel all pending futures
                for future in futures:
                    future.cancel()
                # Force shutdown
                executor.shutdown(wait=False, cancel_futures=True)
                print("✓ All workers terminated\n")
                print("="*80)
                print("VALIDATION INTERRUPTED BY USER")
                print("="*80)
                if self.checkpoint_interval:
                    print("Note: Partial checkpoints may have been saved.")
                return

            # Wait for all futures to complete (normal completion)
            print("\n\nWaiting for all workers to finish...")
            results = {}
            for future in as_completed(futures):
                algo_key = futures[future]
                try:
                    result = future.result()
                    results[algo_key] = result
                except Exception as e:
                    print(f"Worker {algo_key} raised exception: {e}")
                    results[algo_key] = {'status': 'error', 'error': str(e)}

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("PARALLEL RESEARCH VALIDATION COMPLETE")
        print("="*80)
        print(f"Total wall time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Speedup vs sequential: ~{len(self.algorithms)}x (theoretical)")

        # Analyze final results from CSV
        self._analyze_final_results()

        print(f"\nDetailed results: {self.csv_path}")
        print("="*80)

    def _analyze_final_results(self):
        """Analyze and display final results from CSV."""
        try:
            # Read CSV manually (no pandas dependency)
            algo_final_rows = {}

            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    algo_name = row['algorithm']
                    # Keep latest row for each algorithm
                    algo_final_rows[algo_name] = {
                        'iteration': int(row['iteration']),
                        'nash_conv': float(row['nash_conv']),
                        'config': row['config']
                    }

            # Build final results
            final_results = []
            for algo_key, algo_info in self.algorithms.items():
                if algo_info['name'] in algo_final_rows:
                    row_data = algo_final_rows[algo_info['name']]
                    final_results.append({
                        'key': algo_key,
                        'name': algo_info['name'],
                        'config': algo_info['config'],
                        'nash_conv': row_data['nash_conv'],
                        'iteration': row_data['iteration']
                    })

            if not final_results:
                print("\nNo results to analyze")
                return

            # Display final Nash convergence
            print(f"\nFinal Nash Convergence:")
            print("-" * 80)
            for result in final_results:
                print(f"{result['name']:15} | {result['config']:25} | Nash: {result['nash_conv']:.6f}")

            # Rankings
            print("\n" + "="*80)
            print("FINAL RANKINGS (Best to Worst)")
            print("="*80)
            ranked = sorted(final_results, key=lambda r: r['nash_conv'])
            for rank, result in enumerate(ranked, 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                print(f"{medal:3} {result['name']:15} | {result['config']:25} | {result['nash_conv']:.6f}")

            # Research validation
            print("\n" + "="*80)
            print("FINAL RESEARCH CLAIMS VALIDATION")
            print("="*80)

            best_final = ranked[0]

            if best_final['key'] == 'SOTA_DCFR':
                print(f"✓ SOTA DCFR(1.5,0,2) achieved best convergence (as claimed in research)")
            else:
                print(f"✗ SOTA DCFR(1.5,0,2) did NOT achieve best convergence")
                print(f"  Winner was: {best_final['name']} ({best_final['config']})")

            # Check if SOTA is at least in top 2
            sota_rank = next((i+1 for i, r in enumerate(ranked) if r['key'] == 'SOTA_DCFR'), None)
            if sota_rank and sota_rank <= 2:
                print(f"✓ SOTA DCFR ranked #{sota_rank} (top 2 is acceptable)")

            # Improvement vs SIMPLE
            simple_final = next((r['nash_conv'] for r in final_results if r['key'] == 'SIMPLE'), None)
            if simple_final and simple_final > 0:
                print(f"\nFinal Improvement vs SIMPLE Baseline:")
                for result in final_results:
                    if result['key'] != 'SIMPLE':
                        improvement = ((simple_final - result['nash_conv']) / simple_final) * 100
                        print(f"  {result['name']:15}: {improvement:+6.1f}%")

        except Exception as e:
            print(f"\nError analyzing results: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate DCFR research configurations on 3-player Kuhn poker (PARALLEL)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--iterations', type=int, default=1000000,
                       help='Total iterations (default: 1,000,000)')
    parser.add_argument('--check-interval', type=int, default=50000,
                       help='Check exploitability every N iterations (default: 50,000)')
    parser.add_argument('--progress-interval', type=int, default=2000,
                       help='Show progress every N iterations (default: 2,000)')
    parser.add_argument('--checkpoint-interval', type=int, default=None,
                       help='Save checkpoint every N iterations (default: None)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--checkpoint-prefix', type=str, default=None,
                       help='Prefix for checkpoint files')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output CSV')
    parser.add_argument('--force-restart', action='store_true',
                       help='Ignore existing checkpoints')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum parallel workers (default: half of CPU cores)')
    parser.add_argument('--throttle-delay', type=float, default=0.0,
                       help='Seconds to sleep after each exploitability check (thermal control, default: 0)')

    args = parser.parse_args()

    # Calculate default workers
    default_workers = max(1, os.cpu_count() // 2) if args.max_workers is None else args.max_workers

    print(f"\nCPU cores available: {os.cpu_count()}")
    print(f"Max workers: {default_workers} (default: half of cores)")
    print(f"Process priority: Lowered (nice +10)")
    if args.throttle_delay > 0:
        print(f"Thermal throttling: {args.throttle_delay}s sleep after each check (reduces heat)")
    print()

    try:
        runner = ParallelResearchValidationRunner(
            iterations=args.iterations,
            check_interval=args.check_interval,
            progress_interval=args.progress_interval,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_prefix=args.checkpoint_prefix,
            output_dir=args.output_dir,
            force_restart=args.force_restart,
            max_workers=args.max_workers,
            throttle_delay=args.throttle_delay
        )
        runner.run()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        if args.checkpoint_interval:
            print("Checkpoints saved - rerun to resume")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
