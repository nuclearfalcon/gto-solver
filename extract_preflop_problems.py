#!/usr/bin/env python3
"""
Extract Preflop GTO Training Problems - Command Line Tool

Extracts preflop poker training problems from trained CFR policies.

Usage:
    # Extract all preflop problems
    python extract_preflop_problems.py --policy results/policy.pkl --config configs/6p_10bb_holdem.json --output problems/6p_all.json

    # Limit number of problems
    python extract_preflop_problems.py --policy results/policy.pkl --config configs/6p_10bb_holdem.json --output problems/6p_sample.json --max 100

    # Filter by tag
    python extract_preflop_problems.py --policy results/policy.pkl --config configs/6p_10bb_holdem.json --output problems/6p_raises.json --filter-tag facing_raise

    # Show statistics without saving
    python extract_preflop_problems.py --policy results/policy.pkl --config configs/6p_10bb_holdem.json --stats-only

Make sure to activate the virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from collections import Counter

from preflop_problem_extractor import PreflopProblemExtractor
from gto_problem import PreflopProblem
from game_config import PokerGameConfig


def filter_problems(problems: List[PreflopProblem],
                    filter_tag: Optional[str] = None,
                    filter_position: Optional[str] = None,
                    close_decisions_only: bool = False) -> List[PreflopProblem]:
    """
    Filter problems based on criteria.

    Args:
        problems: List of problems to filter
        filter_tag: Only include problems with this tag
        filter_position: Only include problems where hero is in this position
        close_decisions_only: Only include close decisions (2+ viable actions)

    Returns:
        Filtered list of problems
    """
    filtered = problems

    if filter_tag:
        filtered = [p for p in filtered if filter_tag in p.tags]

    if filter_position:
        filtered = [p for p in filtered if p.hero_position == filter_position]

    if close_decisions_only:
        filtered = [p for p in filtered if p.is_close_decision()]

    return filtered


def print_statistics(problems: List[PreflopProblem]):
    """Print statistics about the extracted problems."""
    print("\n" + "=" * 70)
    print("PROBLEM STATISTICS")
    print("=" * 70)
    print(f"\nTotal problems: {len(problems)}")

    if not problems:
        print("No problems to analyze.")
        return

    # Position distribution
    positions = Counter(p.hero_position for p in problems)
    print("\nProblems by position:")
    for pos, count in positions.most_common():
        pct = 100 * count / len(problems)
        print(f"  {pos:6s}: {count:4d} ({pct:5.1f}%)")

    # Tag distribution
    all_tags = []
    for p in problems:
        all_tags.extend(p.tags)
    tag_counts = Counter(all_tags)

    print("\nMost common tags:")
    for tag, count in tag_counts.most_common(10):
        pct = 100 * count / len(problems)
        print(f"  {tag:20s}: {count:4d} ({pct:5.1f}%)")

    # Decision complexity
    close_decisions = sum(1 for p in problems if p.is_close_decision())
    pct_close = 100 * close_decisions / len(problems)
    print(f"\nClose decisions (2+ viable actions): {close_decisions} ({pct_close:.1f}%)")

    # Hand categories
    categories = Counter(p.hand_category for p in problems)
    print("\nProblems by hand category:")
    for cat, count in categories.most_common(10):
        pct = 100 * count / len(problems)
        if cat:  # Only show non-empty categories
            print(f"  {cat:20s}: {count:4d} ({pct:5.1f}%)")

    # Pot size distribution
    small_pot = sum(1 for p in problems if 'small_pot' in p.tags)
    large_pot = sum(1 for p in problems if 'large_pot' in p.tags)
    medium_pot = len(problems) - small_pot - large_pot

    print("\nPot size distribution:")
    print(f"  Small pots:  {small_pot:4d} ({100 * small_pot / len(problems):5.1f}%)")
    print(f"  Medium pots: {medium_pot:4d} ({100 * medium_pot / len(problems):5.1f}%)")
    print(f"  Large pots:  {large_pot:4d} ({100 * large_pot / len(problems):5.1f}%)")

    # Multiway vs heads-up
    multiway = sum(1 for p in problems if 'multiway' in p.tags)
    headsup = len(problems) - multiway
    print("\nPlayer count:")
    print(f"  Heads-up: {headsup:4d} ({100 * headsup / len(problems):5.1f}%)")
    print(f"  Multiway: {multiway:4d} ({100 * multiway / len(problems):5.1f}%)")

    print("\n" + "=" * 70)


def print_sample_problems(problems: List[PreflopProblem], num_samples: int = 5):
    """Print a few sample problems."""
    print("\n" + "=" * 70)
    print(f"SAMPLE PROBLEMS (showing {min(num_samples, len(problems))} of {len(problems)})")
    print("=" * 70)

    for i, problem in enumerate(problems[:num_samples]):
        print(f"\n{i+1}. {problem}")
        print(f"   Stacks: {', '.join(f'{k}={v:.1f}BB' for k, v in problem.stacks_bb.items())}")
        print(f"   Action: {problem.format_action_history()}")
        print(f"   GTO: {problem.format_gto_strategy()}")
        print(f"   Tags: {', '.join(problem.tags[:5])}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Extract preflop GTO training problems from CFR policies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all preflop problems from a policy
  %(prog)s --policy results/6p_policy.pkl --config configs/6p_10bb_holdem.json --output problems/6p_all.json

  # Extract only close decisions (mixed strategies)
  %(prog)s --policy results/6p_policy.pkl --config configs/6p_10bb_holdem.json --output problems/6p_close.json --close-only

  # Extract problems for specific position
  %(prog)s --policy results/6p_policy.pkl --config configs/6p_10bb_holdem.json --output problems/6p_btn.json --position BTN

  # Show statistics without saving
  %(prog)s --policy results/6p_policy.pkl --config configs/6p_10bb_holdem.json --stats-only

  # Limit to first 100 problems
  %(prog)s --policy results/6p_policy.pkl --config configs/6p_10bb_holdem.json --output problems/6p_sample.json --max 100
        """
    )

    parser.add_argument('--policy', required=True, help='Path to policy pickle file')
    parser.add_argument('--config', required=True, help='Path to game config JSON file')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--max', type=int, help='Maximum number of problems to extract')
    parser.add_argument('--filter-tag', help='Only include problems with this tag')
    parser.add_argument('--position', help='Only include problems for this position (e.g., BTN, BB)')
    parser.add_argument('--close-only', action='store_true', help='Only include close decisions')
    parser.add_argument('--stats-only', action='store_true', help='Show statistics without saving')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample problems to show (default: 5)')

    args = parser.parse_args()

    # Validate arguments
    if not args.stats_only and not args.output:
        parser.error("Either --output or --stats-only must be specified")

    # Check if files exist
    policy_path = Path(args.policy)
    config_path = Path(args.config)

    if not policy_path.exists():
        print(f"Error: Policy file not found: {args.policy}")
        sys.exit(1)

    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    print(f"Loading config: {args.config}")
    config = PokerGameConfig.from_json(args.config)
    print(f"  Game: {config.num_players}-player, {config.get_short_description()}")

    # Create extractor
    print(f"\nLoading policy: {args.policy}")
    extractor = PreflopProblemExtractor(args.policy, config)

    # Extract problems
    print("\nExtracting preflop problems...")
    problems = extractor.extract_problems(preflop_only=True, max_problems=args.max)

    # Apply filters
    if args.filter_tag or args.position or args.close_only:
        print("\nApplying filters...")
        original_count = len(problems)
        problems = filter_problems(
            problems,
            filter_tag=args.filter_tag,
            filter_position=args.position,
            close_decisions_only=args.close_only
        )
        print(f"  Filtered: {original_count} → {len(problems)} problems")

        if args.filter_tag:
            print(f"    Tag filter: {args.filter_tag}")
        if args.position:
            print(f"    Position filter: {args.position}")
        if args.close_only:
            print(f"    Close decisions only")

    # Print statistics
    print_statistics(problems)

    # Print sample problems
    print_sample_problems(problems, num_samples=args.samples)

    # Save to file
    if not args.stats_only and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving to: {args.output}")
        extractor.save_problems(problems, args.output)
        print(f"✓ Saved {len(problems)} problems")

    print("\nDone!")


if __name__ == '__main__':
    main()
