#!/usr/bin/env python3
"""
Solver Metrics Tracking

Tracks and stores metrics during GTO poker solving including:
- Exploitability over time
- Iterations per second
- Memory usage
- Convergence rate

Requirements:
    source ~/open_spiel/venv/bin/activate
    pip install psutil (for memory tracking)

Example:
    tracker = MetricsTracker(algorithm_name='CFR+')
    tracker.record_checkpoint(iteration=1000, exploitability=0.5, memory_mb=250)
    tracker.save_csv('results/cfr_plus_metrics.csv')
"""

import time
import csv
import json
import psutil
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class MetricCheckpoint:
    """Single checkpoint of metrics during solving."""
    iteration: int
    exploitability: float
    time_elapsed: float  # Seconds since start
    memory_mb: float
    iters_per_sec: float
    convergence_rate: Optional[float] = None  # % improvement since last check

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsTracker:
    """
    Track metrics during poker solving.

    Stores checkpoints and provides methods to save/load metrics.
    """

    def __init__(self, algorithm_name: str, game_description: str = ""):
        """
        Initialize metrics tracker.

        Args:
            algorithm_name: Name of solving algorithm (e.g., 'CFR+', 'MCCFR')
            game_description: Description of game being solved
        """
        self.algorithm_name = algorithm_name
        self.game_description = game_description
        self.checkpoints: List[MetricCheckpoint] = []
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.sampled_exploitability_result: Optional[Dict] = None  # Final sampled exploit result

    def record_checkpoint(
        self,
        iteration: int,
        exploitability: float,
        memory_mb: Optional[float] = None
    ):
        """
        Record a metrics checkpoint.

        Args:
            iteration: Current iteration number
            exploitability: Current exploitability value
            memory_mb: Memory usage in MB (auto-measured if None)
        """
        time_elapsed = time.time() - self.start_time

        # Calculate iters/sec
        iters_per_sec = iteration / time_elapsed if time_elapsed > 0 else 0

        # Measure memory if not provided
        if memory_mb is None:
            memory_mb = self.process.memory_info().rss / (1024 * 1024)

        # Calculate convergence rate
        convergence_rate = None
        if len(self.checkpoints) > 0:
            prev_exploit = self.checkpoints[-1].exploitability
            if prev_exploit > 0:
                convergence_rate = ((prev_exploit - exploitability) / prev_exploit) * 100

        checkpoint = MetricCheckpoint(
            iteration=iteration,
            exploitability=exploitability,
            time_elapsed=time_elapsed,
            memory_mb=memory_mb,
            iters_per_sec=iters_per_sec,
            convergence_rate=convergence_rate
        )

        self.checkpoints.append(checkpoint)

    def record_sampled_exploitability(self, result: Dict):
        """
        Record final sampled exploitability result.

        Args:
            result: Result dictionary from SampledExploitabilityCalculator
        """
        self.sampled_exploitability_result = result

    def get_formatted_sampled_exploit_report(self) -> Optional[str]:
        """
        Get formatted sampled exploitability report if available.

        Returns:
            Formatted string or None if not calculated yet
        """
        if self.sampled_exploitability_result is None:
            return None

        from exploitability_metrics import format_exploitability_report
        return format_exploitability_report(self.sampled_exploitability_result)

    def get_latest_checkpoint(self) -> Optional[MetricCheckpoint]:
        """Get most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def get_convergence_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with summary stats
        """
        if not self.checkpoints:
            return {}

        exploitabilities = [c.exploitability for c in self.checkpoints]
        iters_per_sec_values = [c.iters_per_sec for c in self.checkpoints]
        memory_values = [c.memory_mb for c in self.checkpoints]

        latest = self.checkpoints[-1]

        return {
            'algorithm': self.algorithm_name,
            'game_description': self.game_description,
            'total_iterations': latest.iteration,
            'total_time': latest.time_elapsed,
            'final_exploitability': latest.exploitability,
            'initial_exploitability': exploitabilities[0] if exploitabilities else None,
            'min_exploitability': min(exploitabilities) if exploitabilities else None,
            'avg_iters_per_sec': sum(iters_per_sec_values) / len(iters_per_sec_values),
            'peak_memory_mb': max(memory_values) if memory_values else None,
            'num_checkpoints': len(self.checkpoints),
        }

    def save_csv(self, filepath: str):
        """
        Save metrics to CSV file.

        Args:
            filepath: Path to save CSV file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            if not self.checkpoints:
                return

            fieldnames = [
                'iteration', 'exploitability', 'time_elapsed',
                'memory_mb', 'iters_per_sec', 'convergence_rate'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for checkpoint in self.checkpoints:
                writer.writerow(checkpoint.to_dict())

    def save_json_summary(self, filepath: str):
        """
        Save summary statistics to JSON.

        Args:
            filepath: Path to save JSON file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_convergence_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

    def load_csv(self, filepath: str):
        """
        Load metrics from CSV file.

        Args:
            filepath: Path to CSV file
        """
        self.checkpoints = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                checkpoint = MetricCheckpoint(
                    iteration=int(row['iteration']),
                    exploitability=float(row['exploitability']),
                    time_elapsed=float(row['time_elapsed']),
                    memory_mb=float(row['memory_mb']),
                    iters_per_sec=float(row['iters_per_sec']),
                    convergence_rate=float(row['convergence_rate']) if row.get('convergence_rate') else None
                )
                self.checkpoints.append(checkpoint)


class AdaptiveSchedule:
    """
    Adaptive exploitability testing schedule.

    Tests more frequently early in solving, less frequently later.
    """

    def __init__(
        self,
        thresholds: List[int] = None,
        intervals: List[int] = None
    ):
        """
        Initialize adaptive schedule.

        Args:
            thresholds: Iteration thresholds for changing intervals
            intervals: Intervals to use at each threshold

        Default schedule:
            - Iterations 0-500k: check every 50k
            - Iterations 500k-2M: check every 100k
            - Iterations 2M+: check every 250k
        """
        if thresholds is None:
            thresholds = [0, 500_000, 2_000_000]
        if intervals is None:
            intervals = [50_000, 100_000, 250_000]

        if len(thresholds) != len(intervals):
            raise ValueError("thresholds and intervals must have same length")

        self.thresholds = thresholds
        self.intervals = intervals

    def get_next_check(self, current_iteration: int) -> int:
        """
        Get next iteration to check exploitability.

        Args:
            current_iteration: Current iteration number

        Returns:
            Next iteration to check
        """
        # Find which interval we're in
        interval = self.intervals[-1]  # Default to last interval
        for i in range(len(self.thresholds) - 1, -1, -1):
            if current_iteration >= self.thresholds[i]:
                interval = self.intervals[i]
                break

        # Calculate next check point
        next_check = ((current_iteration // interval) + 1) * interval

        return next_check

    def get_current_interval(self, current_iteration: int) -> int:
        """
        Get current checking interval.

        Args:
            current_iteration: Current iteration number

        Returns:
            Current interval in iterations
        """
        for i in range(len(self.thresholds) - 1, -1, -1):
            if current_iteration >= self.thresholds[i]:
                return self.intervals[i]
        return self.intervals[0]

    def should_check(self, current_iteration: int, last_check: int) -> bool:
        """
        Determine if we should check exploitability now.

        Args:
            current_iteration: Current iteration number
            last_check: Iteration of last exploitability check

        Returns:
            True if we should check now
        """
        interval = self.get_current_interval(current_iteration)
        return (current_iteration - last_check) >= interval

    def get_schedule_description(self) -> str:
        """Get human-readable description of schedule."""
        parts = []
        for i in range(len(self.thresholds)):
            threshold = self.thresholds[i]
            interval = self.intervals[i]

            if i < len(self.thresholds) - 1:
                next_threshold = self.thresholds[i + 1]
                parts.append(
                    f"Iters {threshold:,}-{next_threshold:,}: every {interval:,}"
                )
            else:
                parts.append(
                    f"Iters {threshold:,}+: every {interval:,}"
                )

        return ", ".join(parts)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "2h 15m 30s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def format_memory(mb: float) -> str:
    """
    Format memory in human-readable format.

    Args:
        mb: Memory in megabytes

    Returns:
        Formatted string like "2.5 GB"
    """
    if mb < 1024:
        return f"{mb:.1f} MB"
    else:
        gb = mb / 1024
        return f"{gb:.2f} GB"
