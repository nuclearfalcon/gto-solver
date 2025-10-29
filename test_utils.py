#!/usr/bin/env python3
"""
Common Test Utilities for GTO Poker Training

This module provides shared utilities used across various test scripts,
ensuring a single source of truth for common test operations.

Requirements:
    source ~/open_spiel/venv/bin/activate
"""

import gc
import time
import psutil
from typing import Callable, Any, Dict, Tuple


def get_memory_mb() -> float:
    """
    Get current memory usage in megabytes.

    Returns:
        float: Current process memory usage in MB.
    """
    return psutil.Process().memory_info().rss / 1024 / 1024


def measure_memory_delta(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Execute a function and measure the memory increase.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (function result, memory increase in MB)
    """
    # Force garbage collection before measurement
    gc.collect()
    initial_memory = get_memory_mb()

    # Execute function
    result = func(*args, **kwargs)

    # Measure after execution
    final_memory = get_memory_mb()
    memory_increase = final_memory - initial_memory

    return result, memory_increase


def measure_time_and_memory(func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
    """
    Execute a function and measure both execution time and memory usage.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (function result, metrics dictionary with 'time_seconds' and 'memory_mb_increase')
    """
    # Force garbage collection before measurement
    gc.collect()
    initial_memory = get_memory_mb()
    start_time = time.time()

    # Execute function
    result = func(*args, **kwargs)

    # Measure after execution
    elapsed_time = time.time() - start_time
    final_memory = get_memory_mb()
    memory_increase = final_memory - initial_memory

    metrics = {
        'time_seconds': elapsed_time,
        'memory_mb_increase': memory_increase,
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory
    }

    return result, metrics


def assert_memory_stable(
    func: Callable,
    max_increase_mb: float = 50.0,
    gc_after: bool = True,
    *args, **kwargs
) -> Any:
    """
    Execute a function and assert that memory increase stays within bounds.

    Args:
        func: Function to execute
        max_increase_mb: Maximum allowed memory increase in MB
        gc_after: Whether to run garbage collection after execution
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Function result if memory test passes

    Raises:
        AssertionError: If memory increase exceeds max_increase_mb
    """
    result, memory_increase = measure_memory_delta(func, *args, **kwargs)

    if gc_after:
        gc.collect()
        final_memory = get_memory_mb()
        # Use post-GC memory for the check
        memory_increase = final_memory - (get_memory_mb() - memory_increase)

    assert memory_increase <= max_increase_mb, \
        f"Memory increased by {memory_increase:.2f} MB, exceeding limit of {max_increase_mb:.2f} MB"

    return result


def format_memory_report(metrics: Dict[str, float]) -> str:
    """
    Format a memory metrics dictionary into a readable report.

    Args:
        metrics: Dictionary with memory metrics

    Returns:
        Formatted string report
    """
    lines = []

    if 'initial_memory_mb' in metrics:
        lines.append(f"Initial memory: {metrics['initial_memory_mb']:.2f} MB")

    if 'final_memory_mb' in metrics:
        lines.append(f"Final memory: {metrics['final_memory_mb']:.2f} MB")

    if 'memory_mb_increase' in metrics:
        increase = metrics['memory_mb_increase']
        if increase > 0:
            lines.append(f"Memory increase: +{increase:.2f} MB")
        else:
            lines.append(f"Memory change: {increase:.2f} MB")

    if 'time_seconds' in metrics:
        lines.append(f"Execution time: {metrics['time_seconds']:.2f} seconds")

    return '\n'.join(lines)


def print_test_header(test_name: str, width: int = 70) -> None:
    """
    Print a formatted test header.

    Args:
        test_name: Name of the test
        width: Width of the header line
    """
    print("=" * width)
    print(test_name.center(width))
    print("=" * width)


def print_test_section(section_name: str, width: int = 70) -> None:
    """
    Print a formatted test section header.

    Args:
        section_name: Name of the section
        width: Width of the header line
    """
    print("\n" + "-" * width)
    print(f"  {section_name}")
    print("-" * width)


class MemoryMonitor:
    """
    Context manager for monitoring memory usage during a code block.

    Example:
        with MemoryMonitor() as monitor:
            # Your code here
            pass
        print(f"Memory increased by {monitor.memory_increase:.2f} MB")
    """

    def __init__(self, run_gc_before: bool = True, run_gc_after: bool = False):
        """
        Initialize the memory monitor.

        Args:
            run_gc_before: Whether to run garbage collection before starting
            run_gc_after: Whether to run garbage collection after finishing
        """
        self.run_gc_before = run_gc_before
        self.run_gc_after = run_gc_after
        self.initial_memory = 0.0
        self.final_memory = 0.0
        self.memory_increase = 0.0

    def __enter__(self):
        """Enter the context and record initial memory."""
        if self.run_gc_before:
            gc.collect()
        self.initial_memory = get_memory_mb()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and record final memory."""
        if self.run_gc_after:
            gc.collect()
        self.final_memory = get_memory_mb()
        self.memory_increase = self.final_memory - self.initial_memory

    def get_report(self) -> str:
        """Get a formatted memory report."""
        return format_memory_report({
            'initial_memory_mb': self.initial_memory,
            'final_memory_mb': self.final_memory,
            'memory_mb_increase': self.memory_increase
        })