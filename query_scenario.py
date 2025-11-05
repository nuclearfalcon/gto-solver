#!/usr/bin/env python3
"""
Query Scenario - Interactive Conditional Best Response Analysis

Computes hero's optimal strategy for specific hands against trained GTO policies.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    # Query a specific scenario
    python query_scenario.py \
        --policy results/cfr_plus_2p_2bb_fcpa_20251104_234832_policy.pkl \
        --config configs/2p_2bb_fcpa.json \
        --hero-position 0 \
        --hero-cards "As Kh" \
        --depth-limit 1

    # Query multiple scenarios from JSON
    python query_scenario.py \
        --policy results/policy.pkl \
        --scenario scenarios/hero_ak_btn.json

Example Outputs:
    Best action: Raise (action 2)
    Action EVs:
      Fold: -0.5000
      Call:  0.1234
      Raise: 0.2567  ← Best

    Confidence: ±0.0123 (99% CI)
    Samples used: 312
"""

import argparse
import pickle
import sys
import os
from typing import Dict

import pyspiel

from game_config import PokerGameConfig
from scenario_config import ScenarioConfig
from conditional_solver import ConditionalBestResponse


def format_action_name(game, state, action: int) -> str:
    """
    Get human-readable action name.

    Args:
        game: OpenSpiel game instance
        state: Game state (for context)
        action: Action integer

    Returns:
        String like "Fold", "Call", "Raise 250"
    """
    # For poker, actions typically map to:
    # 0 = Fold, 1 = Call/Check, 2+ = Bet/Raise
    if action == 0:
        return "Fold"
    elif action == 1:
        return "Call/Check"
    else:
        # Try to get more specific info from state
        try:
            action_str = state.action_to_string(state.current_player(), action)
            return action_str
        except:
            return f"Action {action}"


def print_results(result: Dict, game, scenario: ScenarioConfig):
    """
    Print conditional BR results in human-readable format.

    Args:
        result: Dict from ConditionalBestResponse.compute()
        game: OpenSpiel game instance
        scenario: ScenarioConfig describing the scenario
    """
    print()
    print("=" * 80)
    print(f"CONDITIONAL BEST RESPONSE ANALYSIS")
    print("=" * 80)
    print()
    print(f"Scenario: {scenario}")
    print()

    # Best response value
    print(f"Best Response Value: {result['br_value']:.4f}")
    print(f"Confidence Interval: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  ({int(result['confidence_level']*100)}% confidence, width = ±{result['ci_half_width']:.4f})")
    print(f"Samples used: {result['num_samples']}")
    print(f"Converged: {'Yes' if result['converged'] else 'No (reached max samples)'}")
    print()

    # Action EVs
    if result['action_evs']:
        print("Action Expected Values:")
        print("-" * 40)

        # Sort by EV (highest first)
        sorted_actions = sorted(result['action_evs'].items(), key=lambda x: x[1], reverse=True)

        for action, ev in sorted_actions:
            is_best = (action == result['best_action'])
            marker = "  ← BEST" if is_best else ""

            # Try to get readable action name
            action_name = f"Action {action}"
            print(f"  {action_name:20s}: {ev:>8.4f}{marker}")

        print()
        print(f"Recommended Action: Action {result['best_action']}")
    else:
        print("(No action information available)")

    print()
    print("=" * 80)
    print()


def load_policy(policy_path: str):
    """Load pickled policy from file."""
    print(f"Loading policy from {policy_path}...")
    with open(policy_path, 'rb') as f:
        policy_data = pickle.load(f)
    print("Policy loaded successfully")

    # If it's a checkpoint dict with solver, extract the average policy
    if isinstance(policy_data, dict) and 'solver' in policy_data:
        solver = policy_data['solver']
        policy = solver.average_policy()
        print(f"Extracted average policy from solver (iteration {policy_data.get('current_iteration', '?')})")
    # If it's a dict, wrap it in a TabularPolicy
    elif isinstance(policy_data, dict):
        from pyspiel import TabularPolicy
        policy = TabularPolicy(policy_data)
        print("Wrapped dict policy in TabularPolicy")
    else:
        policy = policy_data

    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Query conditional best response for specific hero scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query BTN with AKs preflop only
  python query_scenario.py --policy results/policy.pkl --config configs/2p_5bb_fcpa.json \\
      --hero-position 0 --hero-cards "As Kh" --depth-limit 1

  # Query from JSON scenario file
  python query_scenario.py --policy results/policy.pkl --scenario scenarios/ak_btn.json

  # Full depth analysis (all streets)
  python query_scenario.py --policy results/policy.pkl --config configs/2p_5bb_fcpa.json \\
      --hero-position 0 --hero-cards "Qs Qh"
        """
    )

    # Required
    parser.add_argument('--policy', required=True, help='Path to pickled policy file')

    # Scenario specification (either --scenario OR --config + hero args)
    parser.add_argument('--scenario', help='Path to JSON scenario file')
    parser.add_argument('--config', help='Path to game config JSON')
    parser.add_argument('--hero-position', type=int, help='Hero position (0-indexed)')
    parser.add_argument('--hero-cards', help='Hero cards, e.g., "As Kh"')
    parser.add_argument('--depth-limit', type=int, help='Solve N streets only (1=preflop, None=all)')

    # Sampling parameters
    parser.add_argument('--max-samples', type=int, default=500,
                       help='Maximum opponent card samples (default: 500)')
    parser.add_argument('--min-samples', type=int, default=50,
                       help='Minimum samples before checking convergence (default: 50)')
    parser.add_argument('--confidence', type=float, default=0.99,
                       help='Confidence level for CI (default: 0.99)')
    parser.add_argument('--max-ci-width', type=float, default=0.05,
                       help='Stop when CI width < this fraction of mean (default: 0.05)')
    parser.add_argument('--check-interval', type=int, default=50,
                       help='Check convergence every N samples (default: 50)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress during sampling')

    args = parser.parse_args()

    # Validate scenario specification
    if args.scenario:
        if args.config or args.hero_position is not None or args.hero_cards:
            print("Error: Cannot specify both --scenario and --config/--hero-* arguments")
            sys.exit(1)
    else:
        if not args.config or args.hero_position is None or not args.hero_cards:
            print("Error: Must specify either --scenario OR (--config + --hero-position + --hero-cards)")
            sys.exit(1)

    # Load scenario
    if args.scenario:
        print(f"Loading scenario from {args.scenario}...")
        scenario = ScenarioConfig.from_json(args.scenario)
    else:
        print(f"Loading game config from {args.config}...")
        game_config = PokerGameConfig.from_json(args.config)
        scenario = ScenarioConfig.from_game_config(
            game_config=game_config,
            hero_position=args.hero_position,
            hero_cards_str=args.hero_cards,
            depth_limit=args.depth_limit
        )

    print(f"Scenario: {scenario}")
    print()

    # Create game
    print("Creating game...")
    game = scenario.game_config.create_game()
    print(f"Game: {game.get_type().short_name}")
    print()

    # Load policy
    policy = load_policy(args.policy)

    # Create conditional solver
    print("Initializing conditional best response calculator...")
    cbr = ConditionalBestResponse(game, policy, scenario)
    print()

    # Compute conditional BR
    print("Computing conditional best response...")
    print(f"Sampling up to {args.max_samples} opponent card deals...")
    print(f"Confidence level: {args.confidence*100:.0f}%, target CI width: {args.max_ci_width*100:.1f}%")
    print()

    result = cbr.compute(
        num_samples=args.max_samples,
        confidence_level=args.confidence,
        max_ci_width=args.max_ci_width,
        min_samples=args.min_samples,
        check_interval=args.check_interval,
        verbose=args.verbose
    )

    # Print results
    print_results(result, game, scenario)

    return 0


if __name__ == '__main__':
    sys.exit(main())
