#!/usr/bin/env python3
"""
Example: Running No-Limit Hold'em Poker Simulations with OpenSpiel

This script demonstrates how to use OpenSpiel to simulate hold'em poker games.
Make sure to activate the virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import pyspiel
import random


def create_holdem_game(num_players=2, stack_size=20000):
    """
    Create a No-Limit Hold'em game configuration.

    Args:
        num_players: Number of players (2-10)
        stack_size: Starting stack for each player

    Returns:
        OpenSpiel game object
    """
    # Build blind and stack strings based on number of players
    # Standard: Player 0 posts big blind (100), Player 1 posts small blind (50)
    blinds = ['100', '50'] + ['0'] * (num_players - 2)
    stacks = [str(stack_size)] * num_players

    game_config = {
        'betting': 'nolimit',
        'numPlayers': num_players,
        'numRounds': 4,                    # Preflop, Flop, Turn, River
        'blind': ' '.join(blinds),
        'firstPlayer': '2 1 1 1',          # First to act each round
        'numSuits': 4,
        'numRanks': 13,
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',        # 0 preflop, 3 flop, 1 turn, 1 river
        'stack': ' '.join(stacks)
    }

    return pyspiel.load_game('universal_poker', game_config)


def simulate_random_hand(game, verbose=True):
    """
    Simulate a single hand with random actions.

    Args:
        game: OpenSpiel game object
        verbose: Whether to print detailed output

    Returns:
        List of final returns for each player
    """
    state = game.new_initial_state()

    if verbose:
        print(f"\nStarting new hand with {game.num_players()} players")
        print("=" * 60)

    action_count = 0

    while not state.is_terminal():
        if state.is_chance_node():
            # Chance node: dealing cards
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = random.choices(action_list, weights=prob_list)[0]
            state.apply_action(action)
        else:
            # Player decision node
            current_player = state.current_player()
            legal_actions = state.legal_actions()
            action = random.choice(legal_actions)

            if verbose:
                action_str = state.action_to_string(current_player, action)
                print(f"Player {current_player}: {action_str}")

            state.apply_action(action)
            action_count += 1

    returns = state.returns()

    if verbose:
        print("\nFinal Results:")
        for player_id, player_return in enumerate(returns):
            print(f"  Player {player_id}: {player_return:+.0f} chips")
        print("=" * 60)

    return returns


def main():
    """Run example simulations"""

    print("OpenSpiel Hold'em Poker Simulation Examples")
    print("=" * 60)

    # Example 1: 2-player heads-up
    print("\n### Example 1: Heads-Up (2 players) ###")
    game_2p = create_holdem_game(num_players=2, stack_size=20000)
    simulate_random_hand(game_2p, verbose=True)

    # Example 2: 6-player table
    print("\n### Example 2: 6-Max Table ###")
    game_6p = create_holdem_game(num_players=6, stack_size=10000)
    simulate_random_hand(game_6p, verbose=True)

    # Example 3: Multiple hands statistics
    print("\n### Example 3: Simulating 100 hands ###")
    game = create_holdem_game(num_players=2)
    player_0_profit = 0

    for i in range(100):
        returns = simulate_random_hand(game, verbose=False)
        player_0_profit += returns[0]

    print(f"After 100 hands:")
    print(f"  Player 0 total profit: {player_0_profit:+.0f} chips")
    print(f"  Average per hand: {player_0_profit/100:+.2f} chips")

    print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    main()
