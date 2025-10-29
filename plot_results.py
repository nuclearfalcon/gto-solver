#!/usr/bin/env python3
"""
Plot Solver Results

Visualize exploitability convergence and performance metrics from solver results.

Requirements:
    source ~/open_spiel/venv/bin/activate
    pip install matplotlib

Usage:
    python plot_results.py --metrics results/*_metrics.csv --output plots/convergence.png

    python plot_results.py --metrics results/cfr_plus_*.csv results/cpp_cfr_*.csv --output plots/comparison.png
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot solver convergence and performance metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        required=True,
        help='Paths to metrics CSV files'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for plot image (e.g., plots/convergence.png)'
    )

    parser.add_argument(
        '--title',
        type=str,
        default='Exploitability Convergence',
        help='Plot title'
    )

    parser.add_argument(
        '--log-scale',
        action='store_true',
        help='Use logarithmic scale for y-axis'
    )

    parser.add_argument(
        '--show-memory',
        action='store_true',
        help='Also plot memory usage (creates second subplot)'
    )

    return parser.parse_args()


def load_metrics(filepath: str) -> Dict[str, Any]:
    """
    Load metrics from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        Dictionary with metrics data
    """
    iterations = []
    exploitabilities = []
    times = []
    memories = []
    speeds = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['iteration']))
            exploitabilities.append(float(row['exploitability']))
            times.append(float(row['time_elapsed']))
            memories.append(float(row['memory_mb']))
            speeds.append(float(row['iters_per_sec']))

    # Extract algorithm name from filename
    filename = Path(filepath).stem
    # Assuming format like: cfr_plus_2p_10bb_fchpa_1.5x_20250128_123456_metrics
    parts = filename.split('_')
    if len(parts) >= 2:
        algorithm = parts[0] + '_' + parts[1] if parts[1] not in ['2p', '3p', '6p'] else parts[0]
    else:
        algorithm = filename

    return {
        'algorithm': algorithm,
        'filepath': filepath,
        'iterations': iterations,
        'exploitabilities': exploitabilities,
        'times': times,
        'memories': memories,
        'speeds': speeds,
    }


def plot_convergence(metrics_list: List[Dict[str, Any]], args):
    """
    Plot exploitability convergence.

    Args:
        metrics_list: List of metrics dictionaries
        args: Command line arguments
    """
    # Create figure
    if args.show_memory:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Plot exploitability
    for metrics in metrics_list:
        ax1.plot(
            metrics['iterations'],
            metrics['exploitabilities'],
            marker='o',
            markersize=4,
            label=metrics['algorithm'],
            linewidth=2
        )

    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Exploitability (NashConv)', fontsize=12)
    ax1.set_title(args.title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    if args.log_scale:
        ax1.set_yscale('log')

    # Plot memory if requested
    if args.show_memory:
        for metrics in metrics_list:
            ax2.plot(
                metrics['iterations'],
                metrics['memories'],
                marker='s',
                markersize=4,
                label=metrics['algorithm'],
                linewidth=2
            )

        ax2.set_xlabel('Iterations', fontsize=12)
        ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax2.set_title('Memory Usage Over Time', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Load all metrics files
    metrics_list = []
    for filepath in args.metrics:
        try:
            metrics = load_metrics(filepath)
            metrics_list.append(metrics)
            print(f"Loaded: {filepath} ({len(metrics['iterations'])} checkpoints)")
        except Exception as e:
            print(f"WARNING: Failed to load {filepath}: {e}")
            continue

    if not metrics_list:
        print("ERROR: No metrics files loaded successfully")
        return 1

    # Create plot
    try:
        plot_convergence(metrics_list, args)
    except Exception as e:
        print(f"ERROR: Failed to create plot: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
