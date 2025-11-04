#!/usr/bin/env python3
"""
GPU-Accelerated MCCFR Poker Solver

Production CLI tool for training poker policies using GPU-accelerated Monte Carlo CFR
with bucketing abstraction. Uses JAX for GPU compilation and batched trajectory sampling.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python solve_poker_gpu.py --config configs/gpu/2p_10bb_holdem_gpu.json --iterations 1000
    python solve_poker_gpu.py --num-players 2 --stacks 1000 1000 --blinds 50 100 --iterations 1000
"""

import argparse
import json
import time
import sys
import psutil
from pathlib import Path
from typing import Optional, List

import jax
import jax.numpy as jnp

from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver, MCCFRConfig, GPURegretTable
from matrix_cfr import holdem_jax_v2
from exploitability_metrics import SampledExploitabilityCalculator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated MCCFR Poker Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from config file
  python solve_poker_gpu.py --config configs/gpu/2p_10bb_holdem_gpu.json --iterations 1000

  # Train with explicit parameters
  python solve_poker_gpu.py --num-players 2 --stacks 1000 1000 --blinds 50 100 \\
      --iterations 1000 --batch-size 100 --num-buckets 10000

  # Resume from checkpoint
  python solve_poker_gpu.py --config configs/gpu/2p_10bb_holdem_gpu.json \\
      --iterations 1000 --checkpoint results/checkpoint_500.pkl
        """
    )

    # Config file OR explicit parameters
    config_group = parser.add_argument_group('configuration')
    config_group.add_argument('--config', type=str, help='Path to JSON config file')

    # Explicit game parameters
    game_group = parser.add_argument_group('game parameters')
    game_group.add_argument('--num-players', type=int, help='Number of players (2-10)')
    game_group.add_argument('--stacks', type=float, nargs='+', help='Stack sizes per player')
    game_group.add_argument('--blinds', type=float, nargs='+', help='Blind sizes per player')

    # Training parameters
    train_group = parser.add_argument_group('training parameters')
    train_group.add_argument('--iterations', type=int, required=True, help='Number of iterations to train')
    train_group.add_argument('--batch-size', type=int, default=100, help='Trajectories per iteration (default: 100)')
    train_group.add_argument('--num-buckets', type=int, default=10000, help='Total number of buckets (default: 10000)')
    train_group.add_argument('--num-hand-buckets', type=int, default=200, help='Hand strength buckets (default: 200)')
    train_group.add_argument('--num-pot-buckets', type=int, default=10, help='Pot size buckets (default: 10)')
    train_group.add_argument('--num-actions', type=int, default=4, help='Number of actions (default: 4)')

    # Output and checkpointing
    output_group = parser.add_argument_group('output and checkpointing')
    output_group.add_argument('--output-dir', type=str, default='results', help='Output directory (default: results)')
    output_group.add_argument('--checkpoint-interval', type=int, default=100, help='Save checkpoint every N iterations (default: 100)')
    output_group.add_argument('--checkpoint', type=str, help='Resume from checkpoint file')
    output_group.add_argument('--name', type=str, help='Experiment name (default: auto-generated)')

    # Display options
    display_group = parser.add_argument_group('display options')
    display_group.add_argument('--progress-interval', type=int, default=10, help='Print progress every N iterations (default: 10)')
    display_group.add_argument('--exploitability-interval', type=int, default=250, help='Calculate exploitability every N iterations (default: 250, 0=disable)')
    display_group.add_argument('--quiet', action='store_true', help='Suppress progress output')

    # Advanced options
    advanced_group = parser.add_argument_group('advanced options')
    advanced_group.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    # Validation
    if args.config is None and (args.num_players is None or args.stacks is None or args.blinds is None):
        parser.error("Either --config or (--num-players, --stacks, --blinds) must be provided")

    return args


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: dict, output_path: str):
    """Save configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_experiment_name(args) -> str:
    """Generate experiment name from arguments."""
    if args.name:
        return args.name

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.config:
        config_name = Path(args.config).stem
        return f"{config_name}_{timestamp}"
    else:
        return f"{args.num_players}p_{int(args.stacks[0])}bb_{timestamp}"


def print_header(title: str):
    """Print formatted header."""
    print("=" * 70)
    print(title.center(70))
    print("=" * 70)
    print()


def print_configuration(num_players: int, stacks: jnp.ndarray, blinds: jnp.ndarray,
                       iterations: int, batch_size: int, num_buckets: int,
                       num_hand_buckets: int, num_pot_buckets: int, num_actions: int):
    """Print training configuration."""
    print("Configuration:")
    print(f"  Players: {num_players}")
    print(f"  Stacks: {stacks}")
    print(f"  Blinds: {blinds}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num buckets: {num_buckets:,}")
    print(f"  Hand buckets: {num_hand_buckets}")
    print(f"  Pot buckets: {num_pot_buckets}")
    print(f"  Num actions: {num_actions}")
    print()


def print_progress_line(iteration: int, total_iterations: int, elapsed: float,
                       memory_mb: float, last_exploit: Optional[float], eta: float):
    """Print single-line progress update."""
    progress_pct = (iteration / total_iterations) * 100
    rate = iteration / elapsed if elapsed > 0 else 0

    # Clear line and print progress
    print(f"\r{iteration:>12,} │ {progress_pct:>7.2f}% │ {rate:>10.1f} it/s │ "
          f"{elapsed:>9.1f}s │ {memory_mb:>7.1f} MB │ "
          f"{last_exploit if last_exploit else 0.0:>12.6f} │ {eta:>9.1f}s",
          end='', flush=True)


def print_exploitability_table(exploit_history: List[dict], best_idx: int):
    """Print exploitability history table."""
    print("\n")  # New line after progress
    print(f"Recent Exploitability Tests (last {len(exploit_history)}):")
    print("─────────────┼──────────────────┼──────────────┼───────────")
    print("   Iteration │   Exploitability │         Time │     Status")
    print("─────────────┼──────────────────┼──────────────┼───────────")

    for idx, entry in enumerate(exploit_history):
        status = "     ★ BEST" if idx == best_idx else "           "
        print(f"{entry['iteration']:>12,} │ {entry['exploitability']:>16.6f} │ "
              f"{entry['time']:>11.1f}s │ {status}")

    print("─────────────┼──────────────────┼──────────────┼───────────")
    print()


def main():
    """Main training loop."""
    args = parse_args()

    # Load or construct configuration
    if args.config:
        config_dict = load_config(args.config)
        num_players = config_dict['num_players']
        stacks = jnp.array(config_dict['stacks'], dtype=jnp.float32)
        blinds = jnp.array(config_dict['blinds'], dtype=jnp.float32)
        batch_size = config_dict.get('batch_size', args.batch_size)
        num_buckets = config_dict.get('num_buckets', args.num_buckets)
        num_hand_buckets = config_dict.get('num_hand_buckets', args.num_hand_buckets)
        num_pot_buckets = config_dict.get('num_pot_buckets', args.num_pot_buckets)
        num_actions = config_dict.get('num_actions', args.num_actions)
    else:
        num_players = args.num_players
        stacks = jnp.array(args.stacks, dtype=jnp.float32)
        blinds = jnp.array(args.blinds, dtype=jnp.float32)
        batch_size = args.batch_size
        num_buckets = args.num_buckets
        num_hand_buckets = args.num_hand_buckets
        num_pot_buckets = args.num_pot_buckets
        num_actions = args.num_actions

    # Validate configuration
    if len(stacks) != num_players:
        print(f"ERROR: Number of stacks ({len(stacks)}) must equal num_players ({num_players})")
        sys.exit(1)

    if len(blinds) != num_players:
        print(f"ERROR: Number of blinds ({len(blinds)}) must equal num_players ({num_players})")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = get_experiment_name(args)

    # Print header
    if not args.quiet:
        print_header("GPU-Accelerated MCCFR Poker Solver")
        print_configuration(num_players, stacks, blinds, args.iterations, batch_size,
                          num_buckets, num_hand_buckets, num_pot_buckets, num_actions)

    # Create solver
    # Use GPUMCCFRConfig to include bucketing parameters (required for GPURegretTable)
    from gpu_mccfr_config import GPUMCCFRConfig
    config = GPUMCCFRConfig(
        num_players=num_players,
        stacks=stacks.tolist(),
        blinds=blinds.tolist(),
        batch_size=batch_size,
        num_buckets=num_buckets,
        num_hand_buckets=num_hand_buckets,
        num_pot_buckets=num_pot_buckets,
        num_actions=num_actions,
        seed=args.seed
    )

    if not args.quiet:
        print("Initializing GPU MCCFR solver...")

    # GPUMCCFRSolver now initializes GPURegretTable directly (when config has num_buckets)
    solver = GPUMCCFRSolver(holdem_jax_v2, config, seed=args.seed)

    # Verify GPURegretTable was initialized (should always be true for GPU configs)
    if hasattr(solver.regret_tables[0], 'get_memory_usage_mb'):
        mem_usage = solver.regret_tables[0].get_memory_usage_mb()
        total_mem = mem_usage * num_players

        if not args.quiet:
            print(f"GPU regret tables initialized:")
            print(f"  GPU memory: {mem_usage:.2f} MB per player × {num_players} = {total_mem:.2f} MB total")
            print()
    else:
        raise RuntimeError("GPUMCCFRConfig must be used with solve_poker_gpu.py (requires num_buckets/num_actions)")

    # Load checkpoint if specified
    start_iteration = 0
    if args.checkpoint:
        if not args.quiet:
            print(f"Loading checkpoint from {args.checkpoint}...")
        # TODO: Implement checkpoint loading
        # start_iteration = load_checkpoint(args.checkpoint, solver)
        print("WARNING: Checkpoint loading not yet implemented")
        print()

    # Training loop
    if not args.quiet:
        print(f"Training for {args.iterations} iterations...")
        print()

    training_start = time.time()
    process = psutil.Process()

    # Exploitability tracking
    exploit_history = []
    best_exploit_idx = -1
    last_exploit = None

    # Calculate initial exploitability if requested
    if args.exploitability_interval > 0 and not args.quiet:
        print("Calculating initial exploitability...")
        # Note: This requires implementing policy extraction from solver
        # For now, just store placeholder
        exploit_history.append({
            'iteration': 0,
            'exploitability': 0.0,  # Placeholder - implement actual calculation
            'time': 0.0
        })
        best_exploit_idx = 0
        last_exploit = 0.0
        print(f"Initial exploitability: {last_exploit:.6f}")
        print()

    # Print progress header
    if not args.quiet:
        print("─────────────┼──────────┼────────────┼────────────┼──────────┼──────────────┼───────────")
        print("   Iteration │ Progress │       Rate │    Elapsed │   Memory │    Last Expl │        ETA")
        print("─────────────┼──────────┼────────────┼────────────┼──────────┼──────────────┼───────────")

    for i in range(start_iteration, start_iteration + args.iterations):
        iter_start = time.time()

        # Run GPU-resident iteration
        traj_length = solver.run_iteration_gpu_resident(
            num_players,
            stacks,
            blinds,
            num_buckets=num_buckets,
            num_hand_buckets=num_hand_buckets,
            num_pot_buckets=num_pot_buckets
        )

        iter_time = time.time() - iter_start

        # Get memory usage
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # Update progress every iteration (inline)
        if not args.quiet:
            elapsed = time.time() - training_start
            completed_iters = i + 1 - start_iteration
            rate = completed_iters / elapsed if elapsed > 0 else 0
            eta = (args.iterations - completed_iters) / rate if rate > 0 else 0

            print_progress_line(i + 1, args.iterations, elapsed, memory_mb, last_exploit, eta)

        # Calculate exploitability periodically
        if args.exploitability_interval > 0 and (i + 1) % args.exploitability_interval == 0:
            exploit_start = time.time()

            # Note: Exploitability calculation requires policy extraction
            # Placeholder for now - implement actual calculation
            current_exploit = 0.0  # Placeholder

            exploit_time = time.time() - training_start
            exploit_history.append({
                'iteration': i + 1,
                'exploitability': current_exploit,
                'time': exploit_time
            })

            # Update best
            if best_exploit_idx < 0 or current_exploit < exploit_history[best_exploit_idx]['exploitability']:
                best_exploit_idx = len(exploit_history) - 1

            last_exploit = current_exploit

            # Print exploitability table
            if not args.quiet:
                print_exploitability_table(exploit_history, best_exploit_idx)
                # Reprint header
                print("─────────────┼──────────┼────────────┼────────────┼──────────┼──────────────┼───────────")
                print("   Iteration │ Progress │       Rate │    Elapsed │   Memory │    Last Expl │        ETA")
                print("─────────────┼──────────┼────────────┼────────────┼──────────┼──────────────┼───────────")

        # Save checkpoint
        if args.checkpoint_interval > 0 and (i + 1) % args.checkpoint_interval == 0:
            checkpoint_path = output_dir / f"{experiment_name}_checkpoint_{i+1}.pkl"
            # TODO: Implement checkpoint saving
            # save_checkpoint(checkpoint_path, solver, i + 1)
            pass

    training_time = time.time() - training_start
    final_speed = args.iterations / training_time
    final_throughput = final_speed * batch_size

    if not args.quiet:
        print()  # Newline after progress line
        print()

    # Print final exploitability table if we have data
    if exploit_history and not args.quiet:
        print()
        print_exploitability_table(exploit_history, best_exploit_idx)

    # Print final results
    if not args.quiet:
        print_header("Training Complete")
        print(f"Total time: {training_time:.2f}s for {args.iterations} iterations")
        print(f"Final speed: {final_speed:.3f} it/s")
        print(f"Final throughput: {final_throughput:.0f} trajectories/s")
        print(f"Average time per iteration: {training_time/args.iterations:.2f}s")
        if exploit_history and best_exploit_idx >= 0:
            best = exploit_history[best_exploit_idx]
            print(f"Best exploitability: {best['exploitability']:.6f} at iteration {best['iteration']:,}")
        print()

    # Save final policy
    policy_path = output_dir / f"{experiment_name}_policy_final.pkl"
    if not args.quiet:
        print(f"Saving final policy to {policy_path}...")
    # TODO: Implement policy saving
    # save_policy(policy_path, solver)

    # Save configuration
    config_output = {
        'num_players': num_players,
        'stacks': stacks.tolist(),
        'blinds': blinds.tolist(),
        'iterations': args.iterations,
        'batch_size': batch_size,
        'num_buckets': num_buckets,
        'num_hand_buckets': num_hand_buckets,
        'num_pot_buckets': num_pot_buckets,
        'num_actions': num_actions,
        'seed': args.seed,
        'training_time_seconds': training_time,
        'final_speed_it_per_sec': final_speed,
        'final_throughput_traj_per_sec': final_throughput,
    }

    config_path = output_dir / f"{experiment_name}_config.json"
    save_config(config_output, config_path)

    if not args.quiet:
        print(f"Configuration saved to {config_path}")
        print()
        print("=" * 70)
        print("All done!")
        print("=" * 70)


if __name__ == "__main__":
    main()
