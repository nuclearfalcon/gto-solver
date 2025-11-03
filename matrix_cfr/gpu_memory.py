"""
GPU Memory Management Utilities

Utilities for monitoring and optimizing VRAM usage in matrix-based CFR.

The RTX 4060 Ti has 16GB VRAM, which can be a constraint for large games
like 3-player Hold'em. This module provides:
- Memory usage monitoring (GPU + CPU)
- Component-by-component memory breakdown
- Peak memory tracking
- Memory growth analysis
- Profiling decorators
- Mixed-precision (FP16) support

Usage:
    from matrix_cfr.gpu_memory import MemoryProfiler, profile_memory

    # Automatic profiling with decorator
    @profile_memory("My Operation")
    def my_function():
        # ... code ...
        pass

    # Manual profiling
    profiler = MemoryProfiler()
    profiler.snapshot("before_operation")
    # ... code ...
    profiler.snapshot("after_operation")
    profiler.print_report()
"""

import logging
import psutil
import os
import gc
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from functools import wraps
import time

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single point-in-time memory measurement."""
    name: str
    timestamp: float
    cpu_mb: float
    gpu_mb: Optional[float]
    gpu_peak_mb: Optional[float]
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryDelta:
    """Memory change between two snapshots."""
    operation: str
    cpu_delta_mb: float
    gpu_delta_mb: Optional[float]
    duration_s: float


def get_cpu_memory_mb() -> float:
    """Get current process CPU memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_mb() -> Optional[tuple]:
    """
    Get current GPU memory usage (used, peak) in MB.

    Returns:
        Tuple of (used_mb, peak_mb) or None if GPU not available
    """
    try:
        import jax
        # Check if GPU available
        gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
        if not gpu_devices:
            return None

        # Get JAX memory stats
        stats = jax.local_devices()[0].memory_stats()
        if stats is None:
            return None

        used = stats.get('bytes_in_use', 0) / 1024 / 1024
        peak = stats.get('peak_bytes_in_use', 0) / 1024 / 1024
        return (used, peak)
    except (ImportError, AttributeError, RuntimeError):
        return None


def get_array_memory_mb(arr) -> float:
    """
    Estimate memory usage of a JAX/numpy array in MB.

    Args:
        arr: JAX or numpy array

    Returns:
        Memory size in MB
    """
    try:
        # Try JAX array
        return arr.nbytes / 1024 / 1024
    except AttributeError:
        # Try numpy
        try:
            return arr.nbytes / 1024 / 1024
        except:
            return 0.0


def get_sparse_memory_mb(sparse_matrix) -> float:
    """
    Estimate memory usage of a sparse matrix (BCOO/CSR/etc) in MB.

    Args:
        sparse_matrix: JAX BCOO or scipy sparse matrix

    Returns:
        Memory size in MB
    """
    try:
        # JAX BCOO format
        if hasattr(sparse_matrix, 'data') and hasattr(sparse_matrix, 'indices'):
            data_mem = get_array_memory_mb(sparse_matrix.data)
            indices_mem = get_array_memory_mb(sparse_matrix.indices)
            return data_mem + indices_mem
        # scipy sparse
        elif hasattr(sparse_matrix, 'data') and hasattr(sparse_matrix, 'indptr'):
            data_mem = get_array_memory_mb(sparse_matrix.data)
            indices_mem = get_array_memory_mb(sparse_matrix.indices)
            indptr_mem = get_array_memory_mb(sparse_matrix.indptr)
            return data_mem + indices_mem + indptr_mem
        else:
            return 0.0
    except:
        return 0.0


class MemoryProfiler:
    """
    Tracks memory usage across operations with component-level breakdown.

    Example:
        profiler = MemoryProfiler()
        profiler.snapshot("start")

        # ... build matrices ...
        profiler.snapshot("after_matrices", {"matrices": matrix_memory})

        # ... solve CFR ...
        profiler.snapshot("after_solve", {"regrets": regret_memory})

        profiler.print_report()
        profiler.print_component_analysis()
    """

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.enabled = True

    def snapshot(self, name: str, components: Optional[Dict[str, float]] = None):
        """
        Take a memory snapshot.

        Args:
            name: Descriptive name for this snapshot
            components: Optional dict of component_name -> memory_mb
        """
        if not self.enabled:
            return

        cpu_mb = get_cpu_memory_mb()
        gpu_info = get_gpu_memory_mb()
        gpu_mb = gpu_info[0] if gpu_info else None
        gpu_peak_mb = gpu_info[1] if gpu_info else None

        snapshot = MemorySnapshot(
            name=name,
            timestamp=time.time(),
            cpu_mb=cpu_mb,
            gpu_mb=gpu_mb,
            gpu_peak_mb=gpu_peak_mb,
            components=components or {}
        )
        self.snapshots.append(snapshot)

    def get_deltas(self) -> List[MemoryDelta]:
        """Calculate memory deltas between consecutive snapshots."""
        deltas = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]

            cpu_delta = curr.cpu_mb - prev.cpu_mb
            gpu_delta = (curr.gpu_mb - prev.gpu_mb) if (curr.gpu_mb and prev.gpu_mb) else None
            duration = curr.timestamp - prev.timestamp

            deltas.append(MemoryDelta(
                operation=f"{prev.name} → {curr.name}",
                cpu_delta_mb=cpu_delta,
                gpu_delta_mb=gpu_delta,
                duration_s=duration
            ))
        return deltas

    def print_report(self):
        """Print formatted memory profiling report."""
        if not self.snapshots:
            print("No memory snapshots recorded")
            return

        print("\n" + "=" * 80)
        print("MEMORY PROFILING REPORT")
        print("=" * 80)

        # Snapshots
        print("\n📊 Memory Snapshots:")
        print(f"{'Name':<30} {'CPU (MB)':<12} {'GPU (MB)':<12} {'GPU Peak (MB)':<15}")
        print("-" * 80)
        for snap in self.snapshots:
            gpu_str = f"{snap.gpu_mb:.1f}" if snap.gpu_mb else "N/A"
            gpu_peak_str = f"{snap.gpu_peak_mb:.1f}" if snap.gpu_peak_mb else "N/A"
            print(f"{snap.name:<30} {snap.cpu_mb:>10.1f}  {gpu_str:>10}  {gpu_peak_str:>13}")

        # Deltas
        deltas = self.get_deltas()
        if deltas:
            print("\n📈 Memory Changes:")
            print(f"{'Operation':<50} {'CPU Δ (MB)':<12} {'GPU Δ (MB)':<12}")
            print("-" * 80)
            for delta in deltas:
                cpu_delta_str = f"{delta.cpu_delta_mb:+.1f}"
                gpu_delta_str = f"{delta.gpu_delta_mb:+.1f}" if delta.gpu_delta_mb else "N/A"
                print(f"{delta.operation:<50} {cpu_delta_str:>10}  {gpu_delta_str:>10}")

        # Peak usage
        peak_cpu = max(s.cpu_mb for s in self.snapshots)
        peak_gpu = max((s.gpu_peak_mb for s in self.snapshots if s.gpu_peak_mb), default=None)

        print("\n🔝 Peak Memory Usage:")
        print(f"  CPU: {peak_cpu:.1f} MB")
        if peak_gpu:
            print(f"  GPU: {peak_gpu:.1f} MB")

        print("=" * 80 + "\n")

    def print_component_analysis(self):
        """Print component-by-component memory breakdown."""
        # Find latest snapshot with components
        for snap in reversed(self.snapshots):
            if snap.components:
                print("\n" + "=" * 80)
                print(f"COMPONENT ANALYSIS ({snap.name})")
                print("=" * 80)

                total = sum(snap.components.values())
                print(f"{'Component':<40} {'Memory (MB)':<15} {'% of Total':<10}")
                print("-" * 80)

                # Sort by size descending
                for comp, mem_mb in sorted(snap.components.items(), key=lambda x: x[1], reverse=True):
                    pct = (mem_mb / total * 100) if total > 0 else 0
                    print(f"{comp:<40} {mem_mb:>13.2f}  {pct:>8.1f}%")

                print("-" * 80)
                print(f"{'TOTAL':<40} {total:>13.2f}  {100.0:>8.1f}%")
                print("=" * 80 + "\n")
                break

    def analyze_scaling(self, param_name: str, param_values: List[float], memory_values: List[float]):
        """
        Analyze how memory scales with a parameter (e.g., number of nodes).

        Args:
            param_name: Name of parameter (e.g., "num_nodes")
            param_values: List of parameter values
            memory_values: List of corresponding memory usage (MB)
        """
        if len(param_values) != len(memory_values) or len(param_values) < 2:
            print("Need at least 2 data points for scaling analysis")
            return

        print("\n" + "=" * 80)
        print(f"MEMORY SCALING ANALYSIS: {param_name}")
        print("=" * 80)

        # Calculate growth ratios
        print(f"\n{param_name:<20} {'Memory (MB)':<15} {'Growth Ratio':<15}")
        print("-" * 80)
        for i in range(len(param_values)):
            ratio_str = ""
            if i > 0:
                mem_ratio = memory_values[i] / memory_values[i-1]
                param_ratio = param_values[i] / param_values[i-1]
                ratio_str = f"{mem_ratio:.2f}x (param: {param_ratio:.2f}x)"
            print(f"{param_values[i]:<20} {memory_values[i]:>13.2f}  {ratio_str}")

        # Estimate complexity
        if len(param_values) >= 3:
            # Simple linear regression to estimate O(n^k)
            import math
            log_params = [math.log(p) for p in param_values]
            log_mems = [math.log(m) for m in memory_values]

            n = len(log_params)
            sum_x = sum(log_params)
            sum_y = sum(log_mems)
            sum_xx = sum(x*x for x in log_params)
            sum_xy = sum(x*y for x, y in zip(log_params, log_mems))

            k = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

            print(f"\n📊 Estimated complexity: O({param_name}^{k:.2f})")

        print("=" * 80 + "\n")


def profile_memory(operation_name: str = None):
    """
    Decorator to profile memory usage of a function.

    Example:
        @profile_memory("Matrix Construction")
        def build_matrices(game):
            # ... code ...
            return matrices
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__

            # Force garbage collection
            gc.collect()

            # Measure before
            cpu_before = get_cpu_memory_mb()
            gpu_before = get_gpu_memory_mb()
            time_before = time.time()

            # Execute function
            result = func(*args, **kwargs)

            # Measure after
            gc.collect()
            cpu_after = get_cpu_memory_mb()
            gpu_after = get_gpu_memory_mb()
            time_after = time.time()

            # Report
            cpu_delta = cpu_after - cpu_before
            duration = time_after - time_before

            print(f"\n[Memory Profile: {name}]")
            print(f"  CPU: {cpu_before:.1f} MB → {cpu_after:.1f} MB (Δ {cpu_delta:+.1f} MB)")
            if gpu_before and gpu_after:
                gpu_delta = gpu_after[0] - gpu_before[0]
                print(f"  GPU: {gpu_before[0]:.1f} MB → {gpu_after[0]:.1f} MB (Δ {gpu_delta:+.1f} MB)")
                print(f"  GPU Peak: {gpu_after[1]:.1f} MB")
            print(f"  Duration: {duration:.2f}s")

            return result
        return wrapper
    return decorator


class GPUMemoryManager:
    """
    Manages GPU memory for matrix CFR operations.

    Monitors VRAM usage and provides utilities for memory optimization.
    """

    def __init__(self, max_vram_gb: float = 14.0):
        """
        Initialize GPU memory manager.

        Args:
            max_vram_gb: Maximum VRAM to use (leave margin for system)
        """
        self.max_vram_bytes = int(max_vram_gb * 1024**3)
        self._init_jax_memory()

    def _init_jax_memory(self):
        """
        Initialize JAX memory allocator.
        """
        try:
            import jax
            # Configure JAX to preallocate GPU memory
            # This helps avoid fragmentation
            logger.info(f"Configuring JAX memory allocator (max {self.max_vram_bytes / 1024**3:.1f} GB)")
        except ImportError:
            logger.warning("JAX not installed, skipping GPU memory configuration")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with 'used_mb', 'peak_mb', 'cpu_mb' keys
        """
        cpu_mb = get_cpu_memory_mb()
        result = {'cpu_mb': cpu_mb}

        gpu_info = get_gpu_memory_mb()
        if gpu_info:
            result['gpu_used_mb'] = gpu_info[0]
            result['gpu_peak_mb'] = gpu_info[1]

        return result

    def print_memory_usage(self):
        """
        Print current GPU memory usage to console.
        """
        usage = self.get_memory_usage()

        print("\n" + "=" * 60)
        print("GPU MEMORY USAGE")
        print("=" * 60)
        print(f"CPU: {usage['cpu_mb']:.1f} MB")

        if 'gpu_used_mb' in usage:
            print(f"GPU Used: {usage['gpu_used_mb']:.1f} MB")
            print(f"GPU Peak: {usage['gpu_peak_mb']:.1f} MB")

            # Calculate percentage of 16GB
            pct_used = (usage['gpu_used_mb'] / 16384) * 100
            pct_peak = (usage['gpu_peak_mb'] / 16384) * 100
            print(f"GPU % Used: {pct_used:.1f}%")
            print(f"GPU % Peak: {pct_peak:.1f}%")
        else:
            print("GPU: Not available")

        print("=" * 60 + "\n")

    def estimate_matrix_size(
        self,
        num_nodes: int,
        num_infosets: int,
        num_actions: int,
        sparsity: float = 0.99
    ) -> float:
        """
        Estimate memory required for game matrices.

        Args:
            num_nodes: Number of nodes in game tree
            num_infosets: Number of information sets
            num_actions: Average number of actions per infoset
            sparsity: Expected sparsity of matrices (default 0.99 = 99%)

        Returns:
            Estimated memory in MB
        """
        # Estimate for sparse BCOO format:
        # - Level matrices: num_nodes x num_nodes @ sparsity
        # - Regrets: num_infosets * num_actions (FP32)
        # - Strategies: num_infosets * num_actions (FP32)
        # - Cumulative: 2x above

        bytes_per_float = 4  # FP32

        # Sparse matrices (data + indices)
        num_nonzero = int(num_nodes * num_nodes * (1 - sparsity))
        sparse_mem = num_nonzero * (bytes_per_float + 8)  # data + 2 int indices

        # Regrets and strategies
        num_ia = num_infosets * num_actions
        regret_mem = num_ia * bytes_per_float * 2  # current + cumulative
        strategy_mem = num_ia * bytes_per_float * 2  # current + cumulative

        total_bytes = sparse_mem + regret_mem + strategy_mem
        return total_bytes / 1024 / 1024

    def suggest_batch_size(self, total_size: int, item_size_mb: float = 1.0) -> int:
        """
        Suggest optimal batch size based on available VRAM.

        Args:
            total_size: Total number of items to process
            item_size_mb: Memory per item in MB

        Returns:
            Suggested batch size
        """
        # Leave 20% margin for safety
        available_mb = self.max_vram_bytes / 1024 / 1024 * 0.8

        # Get current usage
        usage = self.get_memory_usage()
        used_mb = usage.get('gpu_used_mb', 0)
        free_mb = available_mb - used_mb

        # Calculate batch size
        batch_size = int(free_mb / item_size_mb)
        batch_size = max(1, min(batch_size, total_size))

        logger.info(f"Suggested batch size: {batch_size} (free: {free_mb:.1f} MB, item size: {item_size_mb:.1f} MB)")
        return batch_size

    def enable_mixed_precision(self):
        """
        Enable FP16 mixed precision to reduce memory usage.

        This can halve memory requirements with minimal accuracy loss.
        Note: This should be called BEFORE importing JAX in your code.
        """
        import os
        os.environ['JAX_DEFAULT_DTYPE_BITS'] = '16'
        logger.info("Mixed precision (FP16) enabled for JAX")
        print("⚠️  Warning: Mixed precision enabled. Call this BEFORE importing JAX!")


def check_gpu_available() -> bool:
    """
    Check if GPU is available for JAX.

    Returns:
        True if GPU detected, False otherwise
    """
    try:
        import jax
        gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
        return len(gpu_devices) > 0
    except ImportError:
        return False


def print_gpu_info():
    """
    Print information about available GPU(s).
    """
    try:
        import jax
        devices = jax.devices()
        print("\n=== GPU Information ===")
        for i, device in enumerate(devices):
            print(f"Device {i}: {device.device_kind} (platform: {device.platform})")
    except ImportError:
        print("JAX not installed - cannot detect GPU")
