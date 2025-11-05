#!/usr/bin/env python3
"""
Solver Progress Logging

Provides formatted logging for GTO poker solving progress.

Requirements:
    source ~/open_spiel/venv/bin/activate

Example:
    logger = SolverLogger('CFR+', max_iterations=1000000)
    logger.log_start()
    logger.log_progress(iteration=1000, exploitability=0.5, memory_mb=250, next_check=50000)
    logger.log_completion()
"""

import logging
import sys
from typing import Optional
from solver_metrics import format_time, format_memory


class SolverLogger:
    """
    Formatted logger for solver progress.

    Provides consistent, readable output during solving.
    """

    def __init__(
        self,
        algorithm_name: str,
        game_description: str = "",
        max_iterations: Optional[int] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize solver logger.

        Args:
            algorithm_name: Name of algorithm (e.g., 'CFR+')
            game_description: Description of game being solved
            max_iterations: Maximum iterations (for progress percentage)
            log_level: Logging level
        """
        self.algorithm_name = algorithm_name
        self.game_description = game_description
        self.max_iterations = max_iterations

        # Setup logger
        self.logger = logging.getLogger(f"Solver.{algorithm_name}")
        self.logger.setLevel(log_level)

        # Remove existing handlers
        self.logger.handlers = []

        # Add console handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Simple format without timestamp (we include time in messages)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

        # Force flush after each message
        self.handler = handler

    def _log_and_flush(self, message: str):
        """Log message and flush to ensure immediate output."""
        self.logger.info(message)
        sys.stdout.flush()

    def log_start(self):
        """Log start of solving."""
        self._log_and_flush("=" * 80)
        self._log_and_flush(f"  Starting {self.algorithm_name} Solver")
        if self.game_description:
            self._log_and_flush(f"  Game: {self.game_description}")
        if self.max_iterations:
            self._log_and_flush(f"  Target iterations: {self.max_iterations:,}")
        self._log_and_flush("=" * 80)

    def log_progress_simple(
        self,
        iteration: int,
        time_elapsed: float,
        iters_per_sec: float,
        memory_mb: float,
        next_check: int,
        last_exploitability: float
    ):
        """
        Log simple progress update without calculating exploitability.

        Args:
            iteration: Current iteration
            time_elapsed: Time elapsed in seconds
            iters_per_sec: Iterations per second
            memory_mb: Memory usage in MB
            next_check: Next iteration to check exploitability
            last_exploitability: Last known exploitability value
        """
        # Calculate progress percentage
        if self.max_iterations:
            progress_pct = (iteration / self.max_iterations) * 100
            progress_str = f"[{progress_pct:5.1f}%]"
        else:
            progress_str = ""

        # Format iteration
        if self.max_iterations:
            iter_str = f"[{iteration:,}/{self.max_iterations:,}]"
        else:
            iter_str = f"[{iteration:,}]"

        # Format time and memory
        time_str = format_time(time_elapsed)
        mem_str = format_memory(memory_mb)

        # Build log message (without exploitability calculation)
        parts = [
            f"[{self.algorithm_name}]",
            iter_str,
            progress_str,
            f"| Time: {time_str}",
            f"| {iters_per_sec:.1f} it/s",
            f"| Mem: {mem_str}",
        ]

        # Add last known exploitability
        parts.append(f"| Last Exploit: {last_exploitability:.6f}")

        # Add next check info
        iters_until_check = next_check - iteration
        parts.append(f"| Next check: {iters_until_check:,} iters")

        message = " ".join(parts)
        self._log_and_flush(message)

    def log_progress_table(
        self,
        iteration: int,
        time_elapsed: float,
        iters_per_sec: float,
        memory_mb: float,
        next_checkpoint: int,
        best_exploitability: float,
        checkpoint_interval: Optional[int] = None
    ):
        """
        Log progress update in a clean table format.

        Args:
            iteration: Current iteration
            time_elapsed: Time elapsed in seconds
            iters_per_sec: Iterations per second
            memory_mb: Memory usage in MB
            next_checkpoint: Next checkpoint iteration
            best_exploitability: Best (lowest) exploitability seen so far
            checkpoint_interval: Checkpoint interval (for ETA calculation)
        """
        # Calculate ETA to next checkpoint
        if checkpoint_interval and iters_per_sec > 0:
            iters_remaining = next_checkpoint - iteration
            eta_seconds = iters_remaining / iters_per_sec
            eta_str = format_time(eta_seconds)
        else:
            eta_str = "N/A"

        # Format time and memory
        time_str = format_time(time_elapsed)
        mem_str = format_memory(memory_mb)

        # Format exploitability
        if best_exploitability == float('inf'):
            exploit_str = "Not yet calc"
        else:
            exploit_str = f"{best_exploitability:.6f}"

        # Print table
        self._log_and_flush("")
        self._log_and_flush("┌" + "─" * 78 + "┐")
        self._log_and_flush(f"│ {'Progress':<20} │ {iteration:>12,} / {self.max_iterations:<12,} {' ':>14} │")
        self._log_and_flush(f"│ {'Iterations/sec':<20} │ {iters_per_sec:>12.1f} {' ':>28} │")
        self._log_and_flush(f"│ {'Best Exploitability':<20} │ {exploit_str:>12} {' ':>28} │")
        self._log_and_flush(f"│ {'Time Elapsed':<20} │ {time_str:>12} {' ':>28} │")
        self._log_and_flush(f"│ {'Memory Usage':<20} │ {mem_str:>12} {' ':>28} │")
        self._log_and_flush(f"│ {'Next Checkpoint':<20} │ {next_checkpoint:>12,} {' ':>28} │")
        self._log_and_flush(f"│ {'ETA to Checkpoint':<20} │ {eta_str:>12} {' ':>28} │")
        self._log_and_flush("└" + "─" * 78 + "┘")
        self._log_and_flush("")

    def log_progress(
        self,
        iteration: int,
        exploitability: float,
        time_elapsed: float,
        iters_per_sec: float,
        memory_mb: float,
        next_check: int,
        convergence_rate: Optional[float] = None
    ):
        """
        Log full progress checkpoint with exploitability.

        Args:
            iteration: Current iteration
            exploitability: Current exploitability value
            time_elapsed: Time elapsed in seconds
            iters_per_sec: Iterations per second
            memory_mb: Memory usage in MB
            next_check: Next iteration to check exploitability
            convergence_rate: Convergence rate (% improvement)
        """
        # Calculate progress percentage
        if self.max_iterations:
            progress_pct = (iteration / self.max_iterations) * 100
            progress_str = f"[{progress_pct:5.1f}%]"
        else:
            progress_str = ""

        # Format iteration
        if self.max_iterations:
            iter_str = f"[{iteration:,}/{self.max_iterations:,}]"
        else:
            iter_str = f"[{iteration:,}]"

        # Format time and memory
        time_str = format_time(time_elapsed)
        mem_str = format_memory(memory_mb)

        # Build log message
        parts = [
            f"[{self.algorithm_name}]",
            iter_str,
            progress_str,
            f"| Exploit: {exploitability:.6f} ✓",  # Add checkmark to show this was calculated
            f"| Time: {time_str}",
            f"| {iters_per_sec:.1f} it/s",
            f"| Mem: {mem_str}",
        ]

        # Add convergence rate if available
        if convergence_rate is not None:
            parts.append(f"| Conv: {convergence_rate:+.2f}%")

        # Add next check info
        iters_until_check = next_check - iteration
        parts.append(f"| Next check: {iters_until_check:,} iters")

        message = " ".join(parts)
        self._log_and_flush(message)

    def log_exploitability_check(self, iteration: int, exploitability: float):
        """
        Log exploitability check.

        Args:
            iteration: Current iteration
            exploitability: Exploitability value
        """
        self._log_and_flush(
            f"  -> Checking exploitability at iteration {iteration:,}: "
            f"{exploitability:.6f}"
        )

    def log_checkpoint_save(self, iteration: int, filepath: str):
        """
        Log checkpoint save.

        Args:
            iteration: Current iteration
            filepath: Path where checkpoint was saved
        """
        self._log_and_flush(f"  -> Checkpoint saved at iteration {iteration:,}: {filepath}")

    def log_completion(
        self,
        total_iterations: int,
        final_exploitability: float,
        total_time: float,
        final_memory_mb: float
    ):
        """
        Log completion of solving.

        Args:
            total_iterations: Total iterations completed
            final_exploitability: Final exploitability value
            total_time: Total time in seconds
            final_memory_mb: Final memory usage in MB
        """
        self._log_and_flush("=" * 80)
        self._log_and_flush(f"  {self.algorithm_name} Solver Completed")
        self._log_and_flush(f"  Total iterations: {total_iterations:,}")
        self._log_and_flush(f"  Final exploitability: {final_exploitability:.6f}")
        self._log_and_flush(f"  Total time: {format_time(total_time)}")
        self._log_and_flush(f"  Average speed: {total_iterations / total_time:.1f} it/s")
        self._log_and_flush(f"  Final memory: {format_memory(final_memory_mb)}")
        self._log_and_flush("=" * 80)

    def log_error(self, message: str):
        """Log error message."""
        self.logger.error(f"ERROR: {message}")

    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(f"WARNING: {message}")

    def log_info(self, message: str):
        """Log info message."""
        self._log_and_flush(f"  {message}")

    def log_section(self, title: str):
        """Log section header."""
        self._log_and_flush("")
        self._log_and_flush("-" * 80)
        self._log_and_flush(f"  {title}")
        self._log_and_flush("-" * 80)


class ComparisonLogger:
    """
    Logger for comparing multiple algorithms.

    Provides formatted output when running multiple solvers sequentially.
    """

    def __init__(self):
        """Initialize comparison logger."""
        self.logger = logging.getLogger("Comparison")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers = []

        # Add console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _log_and_flush(self, message: str):
        """Log message and flush to ensure immediate output."""
        self.logger.info(message)
        sys.stdout.flush()

    def log_comparison_start(self, num_algorithms: int, game_description: str):
        """
        Log start of algorithm comparison.

        Args:
            num_algorithms: Number of algorithms to test
            game_description: Description of game
        """
        self._log_and_flush("")
        self._log_and_flush("=" * 80)
        self._log_and_flush("  ALGORITHM COMPARISON")
        self._log_and_flush("=" * 80)
        self._log_and_flush(f"  Game: {game_description}")
        self._log_and_flush(f"  Algorithms to test: {num_algorithms}")
        self._log_and_flush("=" * 80)
        self._log_and_flush("")

    def log_algorithm_start(self, algorithm_name: str, number: int, total: int):
        """
        Log start of testing a specific algorithm.

        Args:
            algorithm_name: Name of algorithm
            number: Algorithm number (1-indexed)
            total: Total number of algorithms
        """
        self._log_and_flush("")
        self._log_and_flush("=" * 80)
        self._log_and_flush(f"  ALGORITHM {number}/{total}: {algorithm_name}")
        self._log_and_flush("=" * 80)
        self._log_and_flush("")

    def log_algorithm_complete(self, algorithm_name: str, summary: dict):
        """
        Log completion of algorithm test.

        Args:
            algorithm_name: Name of algorithm
            summary: Summary statistics dict
        """
        self._log_and_flush("")
        self._log_and_flush("-" * 80)
        self._log_and_flush(f"  {algorithm_name} Complete")
        self._log_and_flush(f"    Final exploitability: {summary.get('final_exploitability', 'N/A')}")
        self._log_and_flush(f"    Total time: {format_time(summary.get('total_time', 0))}")
        self._log_and_flush(f"    Avg speed: {summary.get('avg_iters_per_sec', 0):.1f} it/s")
        self._log_and_flush(f"    Peak memory: {format_memory(summary.get('peak_memory_mb', 0))}")
        self._log_and_flush("-" * 80)
        self._log_and_flush("")

    def log_comparison_complete(self, num_algorithms: int):
        """
        Log completion of all algorithm tests.

        Args:
            num_algorithms: Number of algorithms tested
        """
        self._log_and_flush("")
        self._log_and_flush("=" * 80)
        self._log_and_flush(f"  COMPARISON COMPLETE")
        self._log_and_flush(f"  Tested {num_algorithms} algorithms")
        self._log_and_flush(f"  Results saved to results/ directory")
        self._log_and_flush("=" * 80)
        self._log_and_flush("")

    def log_comparison_summary_table(self, results: list):
        """
        Log comparison summary table.

        Args:
            results: List of result dictionaries
        """
        self._log_and_flush("")
        self._log_and_flush("=" * 80)
        self._log_and_flush("  SUMMARY TABLE")
        self._log_and_flush("=" * 80)
        self._log_and_flush("")

        # Header
        self._log_and_flush(
            f"{'Algorithm':<20} | {'Exploit':>12} | {'Time':>10} | "
            f"{'Speed':>12} | {'Memory':>10}"
        )
        self._log_and_flush("-" * 80)

        # Rows
        for result in results:
            algo = result.get('algorithm', 'Unknown')[:20]
            exploit = result.get('final_exploitability', 0)
            time_val = result.get('total_time', 0)
            speed = result.get('avg_iters_per_sec', 0)
            memory = result.get('peak_memory_mb', 0)

            self._log_and_flush(
                f"{algo:<20} | {exploit:>12.6f} | {format_time(time_val):>10} | "
                f"{speed:>10.1f} it/s | {format_memory(memory):>10}"
            )

        self._log_and_flush("")
