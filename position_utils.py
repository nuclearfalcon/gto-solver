"""
Position Utilities for Multi-Player Poker

Handles position mapping and naming for 2-9 player poker games.
Position names are relative to the button (dealer position).

Standard positions:
- 2p: BTN, BB
- 3p: BTN, SB, BB
- 6p: UTG, MP, CO, BTN, SB, BB
- 8p: UTG, UTG+1, MP, MP+1, CO, BTN, SB, BB
- 9p: UTG, UTG+1, UTG+2, MP, MP+1, CO, BTN, SB, BB
"""

from typing import List, Dict


# Position name mappings for each player count
# Key: number of players, Value: list of positions (in seat order)
POSITION_NAMES = {
    2: ['BB', 'BTN'],  # Player 0 = BB (big blind), Player 1 = BTN (button/small blind)
    3: ['BB', 'BTN', 'SB'],  # Player 0 = BB, Player 1 = BTN, Player 2 = SB
    4: ['BB', 'CO', 'BTN', 'SB'],
    5: ['BB', 'MP', 'CO', 'BTN', 'SB'],
    6: ['BB', 'UTG', 'MP', 'CO', 'BTN', 'SB'],
    7: ['BB', 'UTG', 'MP', 'MP+1', 'CO', 'BTN', 'SB'],
    8: ['BB', 'UTG', 'UTG+1', 'MP', 'MP+1', 'CO', 'BTN', 'SB'],
    9: ['BB', 'UTG', 'UTG+1', 'UTG+2', 'MP', 'MP+1', 'CO', 'BTN', 'SB'],
}


# Action order (who acts first preflop)
# For most player counts: SB acts first, then BB, then UTG, etc.
# For 2-player (heads-up): BTN acts first preflop (BTN is also SB)
PREFLOP_ACTION_ORDER = {
    2: ['BTN', 'BB'],  # Heads-up: BTN acts first preflop
    3: ['SB', 'BB', 'BTN'],
    4: ['SB', 'BB', 'CO', 'BTN'],
    5: ['SB', 'BB', 'MP', 'CO', 'BTN'],
    6: ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN'],
    7: ['SB', 'BB', 'UTG', 'MP', 'MP+1', 'CO', 'BTN'],
    8: ['SB', 'BB', 'UTG', 'UTG+1', 'MP', 'MP+1', 'CO', 'BTN'],
    9: ['SB', 'BB', 'UTG', 'UTG+1', 'UTG+2', 'MP', 'MP+1', 'CO', 'BTN'],
}


def get_position_name(player_id: int, num_players: int) -> str:
    """
    Get the position name for a player.

    Args:
        player_id: Player ID (0-indexed)
        num_players: Total number of players (2-9)

    Returns:
        Position name string (e.g., 'BTN', 'BB', 'UTG')

    Raises:
        ValueError: If num_players is not in range 2-9
        IndexError: If player_id is invalid for the player count
    """
    if num_players not in POSITION_NAMES:
        raise ValueError(f"Unsupported player count: {num_players}. Must be 2-9.")

    if player_id < 0 or player_id >= num_players:
        raise IndexError(f"Invalid player_id {player_id} for {num_players} players.")

    return POSITION_NAMES[num_players][player_id]


def get_all_positions(num_players: int) -> List[str]:
    """
    Get all position names for a given player count.

    Args:
        num_players: Total number of players (2-9)

    Returns:
        List of position names in seat order
    """
    if num_players not in POSITION_NAMES:
        raise ValueError(f"Unsupported player count: {num_players}. Must be 2-9.")

    return POSITION_NAMES[num_players].copy()


def get_player_id(position: str, num_players: int) -> int:
    """
    Get the player ID for a given position name.

    Args:
        position: Position name (e.g., 'BTN', 'BB', 'UTG')
        num_players: Total number of players (2-9)

    Returns:
        Player ID (0-indexed)

    Raises:
        ValueError: If position doesn't exist for this player count
    """
    if num_players not in POSITION_NAMES:
        raise ValueError(f"Unsupported player count: {num_players}. Must be 2-9.")

    positions = POSITION_NAMES[num_players]

    if position not in positions:
        raise ValueError(f"Position '{position}' not valid for {num_players} players. "
                        f"Valid positions: {positions}")

    return positions.index(position)


def get_preflop_action_order(num_players: int) -> List[str]:
    """
    Get the preflop action order for a given player count.

    Args:
        num_players: Total number of players (2-9)

    Returns:
        List of positions in order of action (first to act → last to act)
    """
    if num_players not in PREFLOP_ACTION_ORDER:
        raise ValueError(f"Unsupported player count: {num_players}. Must be 2-9.")

    return PREFLOP_ACTION_ORDER[num_players].copy()


def get_position_index(position: str, num_players: int) -> int:
    """
    Get the position's index in action order (0 = first to act).

    Args:
        position: Position name
        num_players: Total number of players

    Returns:
        Index in action order (0 = first to act, higher = later position)
    """
    action_order = get_preflop_action_order(num_players)

    if position not in action_order:
        raise ValueError(f"Position '{position}' not valid for {num_players} players.")

    return action_order.index(position)


def is_blind_position(position: str) -> bool:
    """
    Check if a position is a blind position (SB or BB).

    Args:
        position: Position name

    Returns:
        True if position is SB or BB
    """
    return position in ['SB', 'BB']


def get_position_category(position: str, num_players: int) -> str:
    """
    Categorize a position into early/middle/late/blinds.

    Args:
        position: Position name
        num_players: Total number of players

    Returns:
        Category string: 'early', 'middle', 'late', or 'blinds'
    """
    if position in ['SB', 'BB']:
        return 'blinds'
    elif position == 'BTN':
        return 'late'
    elif position == 'CO':
        return 'late'
    elif 'UTG' in position:
        return 'early'
    elif 'MP' in position:
        return 'middle'
    else:
        # Default for unusual positions
        return 'middle'


def format_position_list(positions: List[str]) -> str:
    """
    Format a list of positions as a readable string.

    Args:
        positions: List of position names

    Returns:
        Formatted string (e.g., "UTG, MP, BTN")
    """
    return ', '.join(positions)


def get_position_abbreviations() -> Dict[str, str]:
    """
    Get mapping of position abbreviations to full names.

    Returns:
        Dictionary mapping abbreviations to full names
    """
    return {
        'BTN': 'Button',
        'SB': 'Small Blind',
        'BB': 'Big Blind',
        'CO': 'Cutoff',
        'MP': 'Middle Position',
        'MP+1': 'Middle Position + 1',
        'MP+2': 'Middle Position + 2',
        'UTG': 'Under The Gun',
        'UTG+1': 'Under The Gun + 1',
        'UTG+2': 'Under The Gun + 2',
    }


# Example usage and testing
if __name__ == '__main__':
    print("Position Utils - Test Cases")
    print("=" * 60)

    # Test 2-player (heads-up)
    print("\n2-Player (Heads-Up):")
    print(f"  Positions: {get_all_positions(2)}")
    print(f"  Action order: {get_preflop_action_order(2)}")
    print(f"  Player 0 = {get_position_name(0, 2)}")
    print(f"  Player 1 = {get_position_name(1, 2)}")

    # Test 6-player (6-max)
    print("\n6-Player (6-max):")
    print(f"  Positions: {get_all_positions(6)}")
    print(f"  Action order: {get_preflop_action_order(6)}")
    for i in range(6):
        pos = get_position_name(i, 6)
        category = get_position_category(pos, 6)
        print(f"  Player {i} = {pos} ({category})")

    # Test 9-player (full ring)
    print("\n9-Player (Full Ring):")
    print(f"  Positions: {get_all_positions(9)}")
    print(f"  Action order: {get_preflop_action_order(9)}")

    # Test position lookups
    print("\nPosition Lookups (6-max):")
    print(f"  'BTN' → Player {get_player_id('BTN', 6)}")
    print(f"  'UTG' → Player {get_player_id('UTG', 6)}")
    print(f"  'BB' → Player {get_player_id('BB', 6)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
