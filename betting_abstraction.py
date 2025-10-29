#!/usr/bin/env python3
"""
Custom Betting Abstraction Helper

Provides filtering for custom betting abstractions when using OpenSpiel's
universal_poker with 'fullgame' betting mode.

Requirements:
    source ~/open_spiel/venv/bin/activate

Example:
    game = pyspiel.load_game('universal_poker', {..., 'bettingAbstraction': 'fullgame'})
    filter = CustomBettingFilter(abstraction_type='fchpa_1.5x')

    state = game.new_initial_state()
    # ... advance to decision node

    legal_actions = state.legal_actions()
    filtered_actions = filter.filter_actions(state, legal_actions)
"""

import pyspiel
from typing import List, Set


class CustomBettingFilter:
    """
    Filter legal actions for custom betting abstractions.

    Supports:
    - 'fchpa_1.5x': Fold, Call, Half-pot, Pot, 1.5×pot, All-in
    - 'fcpa': Fold, Call, Pot, All-in (matches OpenSpiel's fcpa)
    - 'fchpa': Fold, Call, Half-pot, Pot, All-in (matches OpenSpiel's fchpa)
    - 'custom': User-defined pot multipliers
    """

    ABSTRACTION_TYPES = {
        'fcpa': [1.0],           # Pot only
        'fchpa': [0.5, 1.0],     # Half-pot, Pot
        'fchpa_1.5x': [0.5, 1.0, 1.5],  # Half-pot, Pot, 1.5×pot
    }

    def __init__(self, abstraction_type: str = 'fchpa_1.5x',
                 tolerance: int = 10,
                 custom_multipliers: List[float] = None):
        """
        Initialize custom betting filter.

        Args:
            abstraction_type: One of 'fcpa', 'fchpa', 'fchpa_1.5x', or 'custom'
            tolerance: Allowed deviation from target bet size (in chips)
            custom_multipliers: For 'custom' type, list of pot multipliers
        """
        self.abstraction_type = abstraction_type
        self.tolerance = tolerance

        if abstraction_type == 'custom':
            if custom_multipliers is None:
                raise ValueError("custom_multipliers required for abstraction_type='custom'")
            self.pot_multipliers = sorted(custom_multipliers)
        elif abstraction_type in self.ABSTRACTION_TYPES:
            self.pot_multipliers = self.ABSTRACTION_TYPES[abstraction_type]
        else:
            raise ValueError(
                f"Unknown abstraction_type: {abstraction_type}. "
                f"Must be one of {list(self.ABSTRACTION_TYPES.keys())} or 'custom'"
            )

    def calculate_pot_size(self, state) -> int:
        """
        Calculate current pot size from game state.

        This is approximate - it sums the history of bets. For precise pot
        calculation, you may need to track pot size externally.

        Args:
            state: OpenSpiel game state

        Returns:
            Estimated pot size in chips
        """
        # Simple approximation: sum all positive actions in history
        # (actions represent bet sizes in universal_poker)
        pot = 0
        history = state.history()

        for action in history:
            if action > 0:  # Positive actions are bets/calls
                pot += action

        # If pot is still 0, we're likely at the start - use blinds
        # This is a fallback and may not be accurate for all cases
        if pot == 0:
            pot = 100  # Default estimate

        return pot

    def get_target_bet_sizes(self, pot: int, max_stack: int) -> Set[int]:
        """
        Calculate target bet sizes based on pot multipliers.

        Args:
            pot: Current pot size
            max_stack: Maximum stack (for all-in)

        Returns:
            Set of target bet sizes
        """
        targets = set()

        for multiplier in self.pot_multipliers:
            bet_size = int(pot * multiplier)
            # Clamp to max stack
            if bet_size > max_stack:
                bet_size = max_stack
            targets.add(bet_size)

        # Always include all-in
        targets.add(max_stack)

        return targets

    def filter_actions(self, state, legal_actions: List[int]) -> List[int]:
        """
        Filter legal actions to match custom betting abstraction.

        Args:
            state: Current game state
            legal_actions: Full list of legal actions from state.legal_actions()

        Returns:
            Filtered list of legal actions
        """
        if not legal_actions:
            return []

        # Always include fold (0) and call (1) if they're legal
        filtered = []
        for action in legal_actions:
            if action <= 1:
                filtered.append(action)

        # Find max stack (largest legal action)
        max_stack = max(legal_actions)

        # Calculate pot size
        pot = self.calculate_pot_size(state)

        # Get target bet sizes
        targets = self.get_target_bet_sizes(pot, max_stack)

        # Filter bet actions (action > 1)
        bet_actions = [a for a in legal_actions if a > 1]

        for target in targets:
            # Find closest legal action to target
            closest_action = None
            closest_distance = float('inf')

            for action in bet_actions:
                distance = abs(action - target)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_action = action

            # Include if within tolerance
            if closest_action is not None and closest_distance <= self.tolerance:
                if closest_action not in filtered:
                    filtered.append(closest_action)

        # Sort for consistency
        return sorted(filtered)

    def get_action_description(self, action: int, pot: int, max_stack: int) -> str:
        """
        Get human-readable description of an action.

        Args:
            action: Action number
            pot: Current pot size
            max_stack: Maximum stack

        Returns:
            Description string
        """
        if action == 0:
            return "Fold"
        elif action == 1:
            return "Call"
        elif action == max_stack:
            return f"All-in ({max_stack})"
        else:
            # Calculate pot multiplier
            multiplier = action / pot if pot > 0 else 0
            return f"Bet {action} ({multiplier:.2f}× pot)"


class ActionFilter:
    """
    Wrapper that provides a filtered action interface for a game state.

    Usage:
        filter = CustomBettingFilter('fchpa_1.5x')
        wrapper = ActionFilter(state, filter)

        filtered_actions = wrapper.legal_actions()
        wrapper.apply_filtered_action(action_index)
    """

    def __init__(self, state, betting_filter: CustomBettingFilter):
        """
        Initialize action filter wrapper.

        Args:
            state: OpenSpiel game state
            betting_filter: CustomBettingFilter instance
        """
        self.state = state
        self.filter = betting_filter
        self._filtered_actions = None

    def legal_actions(self) -> List[int]:
        """Get filtered legal actions."""
        if self._filtered_actions is None:
            full_actions = self.state.legal_actions()
            self._filtered_actions = self.filter.filter_actions(self.state, full_actions)
        return self._filtered_actions

    def num_legal_actions(self) -> int:
        """Get number of filtered legal actions."""
        return len(self.legal_actions())

    def apply_action(self, action: int):
        """
        Apply an action and return new state.

        Args:
            action: Action from filtered action list

        Returns:
            New state after applying action
        """
        if action not in self.legal_actions():
            raise ValueError(
                f"Action {action} not in filtered legal actions: {self.legal_actions()}"
            )
        self.state.apply_action(action)
        self._filtered_actions = None  # Invalidate cache
        return self.state

    def action_description(self, action: int) -> str:
        """Get description of a filtered action."""
        pot = self.filter.calculate_pot_size(self.state)
        max_stack = max(self.state.legal_actions())
        return self.filter.get_action_description(action, pot, max_stack)


# Convenience functions

def create_fchpa_1_5x_filter(tolerance: int = 10) -> CustomBettingFilter:
    """Create filter for FCHPA + 1.5×pot abstraction."""
    return CustomBettingFilter('fchpa_1.5x', tolerance=tolerance)


def create_custom_filter(pot_multipliers: List[float], tolerance: int = 10) -> CustomBettingFilter:
    """Create filter with custom pot multipliers."""
    return CustomBettingFilter('custom', tolerance=tolerance, custom_multipliers=pot_multipliers)
