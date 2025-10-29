#!/usr/bin/env python3
"""
Query GTO Policy

Extract and display strategies from saved policy files.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python query_policy.py --policy results/cfr_plus_2p_10bb_policy.pkl --info-states 10
"""

import argparse
import pickle
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Query saved policy for strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--policy',
        type=str,
        required=True,
        help='Path to saved policy file (.pkl)'
    )

    parser.add_argument(
        '--info-states',
        type=int,
        default=10,
        help='Number of information states to display (default: 10)'
    )

    parser.add_argument(
        '--search',
        type=str,
        default=None,
        help='Search for specific information state (substring match)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load policy
    try:
        with open(args.policy, 'rb') as f:
            policy = pickle.load(f)
        print(f"Loaded policy from: {args.policy}\n")
    except Exception as e:
        print(f"ERROR: Failed to load policy from {args.policy}: {e}")
        return 1

    # Get policy type
    print(f"Policy type: {type(policy).__name__}")
    print("=" * 80)

    # Try to get state lookup if it's a TabularPolicy
    if hasattr(policy, 'state_lookup'):
        state_lookup = policy.state_lookup
        print(f"\nTotal information states: {len(state_lookup)}")

        # Get list of info states
        info_states = list(state_lookup.keys())

        # Filter if search term provided
        if args.search:
            info_states = [s for s in info_states if args.search in s]
            print(f"Filtered to {len(info_states)} states matching '{args.search}'")

        # Display first N info states
        num_to_show = min(args.info_states, len(info_states))
        print(f"\nShowing {num_to_show} information states:\n")

        for i, info_state in enumerate(info_states[:num_to_show], 1):
            print(f"\n[{i}] Information State:")
            print(f"    {info_state}")

            # Try to get action probabilities
            try:
                # For TabularPolicy, we need a state object
                # This is tricky without the game instance
                print(f"    (Action probabilities require game state object)")
            except Exception as e:
                print(f"    Error getting probabilities: {e}")

        print("\n" + "=" * 80)
        print("\nNote: Full strategy extraction requires the game instance.")
        print("This tool provides a preview of stored information states.")
        print("\nFor detailed strategy analysis, use the policy within a solve script")
        print("where you have access to the game object.")

    elif hasattr(policy, 'action_probability_array'):
        print(f"\nPolicy has action_probability_array attribute")
        print(f"Array shape: {policy.action_probability_array.shape if hasattr(policy.action_probability_array, 'shape') else 'N/A'}")
        print("\nThis policy stores action probabilities but requires game context for interpretation.")

    else:
        print("\nPolicy structure:")
        print(f"Attributes: {dir(policy)}")
        print("\nThis policy type may not support direct querying without game context.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
