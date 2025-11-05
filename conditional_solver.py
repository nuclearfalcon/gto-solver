#!/usr/bin/env python3
"""
Conditional Best Response Solver

Computes hero's optimal strategy for specific hands against GTO opponents.
Uses sampling-based approach similar to sampled exploitability.

Requirements:
    source ~/open_spiel/venv/bin/activate

Example:
    from conditional_solver import ConditionalBestResponse

    cbr = ConditionalBestResponse(game, policy, scenario)
    result = cbr.compute()

    print(f"Best action: {result['best_action']}")
    print(f"Action frequencies: {result['action_probs']}")
    print(f"Expected values: {result['action_evs']}")
"""

import random
import gc
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from scenario_config import ScenarioConfig
from exploitability_metrics import _compute_best_response_value_recursive


class ConditionalBestResponse:
    """
    Computes best response for hero with specific cards against GTO opponents.

    Uses Monte Carlo sampling over opponent card combinations to estimate
    hero's optimal strategy.
    """

    def __init__(self, game, policy, scenario: ScenarioConfig):
        """
        Initialize conditional best response calculator.

        Args:
            game: OpenSpiel game instance
            policy: GTO policy to compute best response against
            scenario: ScenarioConfig with hero position, cards, and depth limit
        """
        self.game = game
        self.policy = policy
        self.scenario = scenario

    def compute(
        self,
        num_samples: int = 500,
        confidence_level: float = 0.99,
        max_ci_width: float = 0.05,
        min_samples: int = 50,
        check_interval: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Compute best response strategy for hero.

        Args:
            num_samples: Maximum number of opponent card deals to sample
            confidence_level: Confidence level for CI (e.g., 0.99 for 99% CI)
            max_ci_width: Stop when CI width < this (as fraction of mean)
            min_samples: Minimum samples before checking convergence
            check_interval: Check convergence every N samples
            verbose: Print progress messages

        Returns:
            Dict with keys:
                - 'br_value': Best response EV estimate
                - 'ci_lower': Lower bound of confidence interval
                - 'ci_upper': Upper bound of confidence interval
                - 'num_samples': Number of samples used
                - 'action_evs': Dict of {action: expected_value}
                - 'best_action': Action with highest EV
                - 'converged': Whether sampling converged
        """
        if verbose:
            print(f"Computing conditional best response for {self.scenario}")
            print(f"Sampling up to {num_samples} opponent card deals...")
            print()

        # Initialize streaming statistics (Welford's algorithm)
        n = 0
        mean = 0.0
        m2 = 0.0  # Sum of squared differences from mean

        # Track action values across samples
        action_values = defaultdict(lambda: {'sum': 0.0, 'count': 0})

        converged = False

        for sample_num in range(num_samples):
            # Sample opponent cards and compute BR value
            sample_value, sample_actions = self._sample_one_deal()

            # Update streaming statistics
            n += 1
            delta = sample_value - mean
            mean += delta / n
            delta2 = sample_value - mean
            m2 += delta * delta2

            # Update action value tracking
            for action, value in sample_actions.items():
                action_values[action]['sum'] += value
                action_values[action]['count'] += 1

            # Periodic convergence check and GC
            if n >= min_samples and n % check_interval == 0:
                # Compute confidence interval
                if n > 1:
                    variance = m2 / (n - 1)
                    std_error = math.sqrt(variance / n)

                    # Z-score for confidence level (approximate)
                    z_score = 2.576 if confidence_level >= 0.99 else 1.96  # 99% or 95%

                    ci_half_width = z_score * std_error
                    ci_width_pct = (ci_half_width / abs(mean)) if abs(mean) > 1e-10 else float('inf')

                    if verbose and n % (check_interval * 2) == 0:
                        print(f"  Sample {n}/{num_samples}: "
                              f"Mean BR = {mean:.4f}, "
                              f"CI width = {ci_width_pct*100:.2f}%")

                    # Check convergence
                    if ci_width_pct < max_ci_width:
                        if verbose:
                            print(f"\n✓ Converged at {n} samples (CI width < {max_ci_width*100:.1f}%)")
                        converged = True
                        break

                # Memory cleanup
                gc.collect()

        # Final statistics
        if n > 1:
            variance = m2 / (n - 1)
            std_error = math.sqrt(variance / n)
            z_score = 2.576 if confidence_level >= 0.99 else 1.96
            ci_half_width = z_score * std_error
            ci_lower = mean - ci_half_width
            ci_upper = mean + ci_half_width
        else:
            std_error = 0.0
            ci_lower = mean
            ci_upper = mean

        # Compute average action EVs
        action_evs = {}
        for action, stats in action_values.items():
            if stats['count'] > 0:
                action_evs[action] = stats['sum'] / stats['count']

        # Find best action
        best_action = max(action_evs.items(), key=lambda x: x[1])[0] if action_evs else None

        if verbose:
            if not converged:
                print(f"\n⚠ Reached max samples ({num_samples}) without full convergence")
            print(f"\nFinal BR value: {mean:.4f} ± {ci_half_width:.4f}")
            print(f"Confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}] ({int(confidence_level*100)}% CI)")
            print(f"Samples used: {n}")
            print()

        return {
            'br_value': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_half_width': ci_half_width,
            'std_error': std_error,
            'num_samples': n,
            'confidence_level': confidence_level,
            'action_evs': action_evs,
            'best_action': best_action,
            'converged': converged
        }

    def _sample_one_deal(self) -> Tuple[float, Dict[int, float]]:
        """
        Sample one opponent card deal and compute BR value.

        Returns:
            Tuple of (br_value, action_values)
            - br_value: Best response value for this deal
            - action_values: Dict of {action: value} for hero's first decision
        """
        # Start from game root
        state = self.game.new_initial_state()

        # Deal hero's cards
        state = self._deal_hero_cards(state)

        # Sample opponent cards from remaining deck
        state = self._sample_opponent_cards(state)

        # Now we're past all private card deals, ready to start the game
        # Compute best response value from this state
        br_value = self._compute_conditional_br(
            state,
            self.scenario.hero_position,
            self.policy,
            self.scenario.depth_limit
        )

        # Also extract action values at hero's first decision point
        # (for action frequency analysis)
        action_values = self._get_action_values_at_first_decision(state)

        return br_value, action_values

    def _deal_hero_cards(self, state):
        """
        Navigate chance nodes to deal specific cards to hero.

        Args:
            state: Current game state (should be at root or early chance node)

        Returns:
            State after hero's cards have been dealt
        """
        cards_dealt = []
        target_cards = set(self.scenario.hero_cards)

        while state.is_chance_node() and len(cards_dealt) < self.scenario.game_config.num_hole_cards * self.scenario.game_config.num_players:
            # Determine whose cards are being dealt
            # In universal_poker, private cards are dealt in player order
            current_deal_player = len(cards_dealt) // self.scenario.game_config.num_hole_cards

            if current_deal_player == self.scenario.hero_position:
                # Deal one of hero's specific cards
                card_idx_for_player = len(cards_dealt) % self.scenario.game_config.num_hole_cards
                card_to_deal = self.scenario.hero_cards[card_idx_for_player]

                # Find the chance action that corresponds to this card
                # Chance outcomes are (action, probability) pairs
                # Action is the card integer
                state = state.child(card_to_deal)
                cards_dealt.append(card_to_deal)
            else:
                # Not hero's turn yet, need to advance through other players' deals
                # We'll handle this in _sample_opponent_cards
                break

        return state

    def _sample_opponent_cards(self, state):
        """
        Sample random cards for opponents from remaining deck.

        Args:
            state: State after hero's cards have been dealt

        Returns:
            State after all private cards have been dealt
        """
        # Track which cards are already dealt
        dealt_cards = set(self.scenario.hero_cards)

        # Continue dealing cards for remaining players
        while state.is_chance_node():
            # Get available chance outcomes
            outcomes = state.chance_outcomes()

            # Filter out already-dealt cards
            available_outcomes = [(action, prob) for action, prob in outcomes
                                 if action not in dealt_cards]

            if not available_outcomes:
                # All cards dealt or no valid cards remaining
                break

            # Sample from available cards
            actions, probs = zip(*available_outcomes)
            # Renormalize probabilities
            total_prob = sum(probs)
            normalized_probs = [p / total_prob for p in probs]

            # Sample one card
            action = random.choices(actions, weights=normalized_probs)[0]

            state = state.child(action)
            dealt_cards.add(action)

        return state

    def _compute_conditional_br(self, state, player_id: int, policy, depth_limit: Optional[int]) -> float:
        """
        Compute best response value with optional depth limit.

        Args:
            state: Starting state
            player_id: Hero player ID
            policy: Opponent policy
            depth_limit: Maximum depth (number of betting rounds), None for unlimited

        Returns:
            Best response value
        """
        if depth_limit is None:
            # No depth limit, use standard BR computation
            return _compute_best_response_value_recursive(state, player_id, policy)
        else:
            # Depth-limited BR
            return self._compute_br_with_depth_limit(state, player_id, policy, depth_limit, current_round=0)

    def _compute_br_with_depth_limit(self, state, player_id: int, policy, depth_limit: int, current_round: int) -> float:
        """
        Compute BR with depth limit (stops after N betting rounds).

        Args:
            state: Current state
            player_id: Player ID
            policy: Opponent policy
            depth_limit: Maximum rounds to explore
            current_round: Current round number (0 = preflop, 1 = flop, etc.)

        Returns:
            Best response value
        """
        if state.is_terminal():
            return state.returns()[player_id]

        # Check if we've exceeded depth limit
        # When we hit depth limit, evaluate using opponent policy
        if current_round >= depth_limit:
            # Use policy value as terminal evaluation
            return self._evaluate_with_policy(state, player_id, policy)

        if state.is_chance_node():
            # Expected value over chance outcomes
            value = 0.0
            for action, prob in state.chance_outcomes():
                next_state = state.child(action)
                # Check if this chance node transitions to next round
                next_round = current_round + 1 if self._is_new_round(state, next_state) else current_round
                value += prob * self._compute_br_with_depth_limit(
                    next_state, player_id, policy, depth_limit, next_round)
            return value

        current_player = state.current_player()

        if current_player == player_id:
            # Hero maximizes
            best_value = float('-inf')
            for action in state.legal_actions():
                next_state = state.child(action)
                next_round = current_round + 1 if self._is_new_round(state, next_state) else current_round
                value = self._compute_br_with_depth_limit(
                    next_state, player_id, policy, depth_limit, next_round)
                best_value = max(best_value, value)
            return best_value
        else:
            # Opponent follows policy
            value = 0.0
            action_probs = policy.action_probabilities(state)
            for action, prob in action_probs.items():
                next_state = state.child(action)
                next_round = current_round + 1 if self._is_new_round(state, next_state) else current_round
                value += prob * self._compute_br_with_depth_limit(
                    next_state, player_id, policy, depth_limit, next_round)
            return value

    def _is_new_round(self, state, next_state) -> bool:
        """Check if transitioning to next state starts a new betting round."""
        # This is a heuristic: if we go through a chance node dealing community cards,
        # it's a new round
        return state.is_chance_node() and not next_state.is_chance_node()

    def _evaluate_with_policy(self, state, player_id: int, policy) -> float:
        """
        Evaluate state using policy (all players follow policy).

        This is used as terminal evaluation when depth limit is reached.
        """
        if state.is_terminal():
            return state.returns()[player_id]

        if state.is_chance_node():
            value = 0.0
            for action, prob in state.chance_outcomes():
                next_state = state.child(action)
                value += prob * self._evaluate_with_policy(next_state, player_id, policy)
            return value

        # All players follow policy
        value = 0.0
        action_probs = policy.action_probabilities(state)
        for action, prob in action_probs.items():
            next_state = state.child(action)
            value += prob * self._evaluate_with_policy(next_state, player_id, policy)
        return value

    def _get_action_values_at_first_decision(self, state) -> Dict[int, float]:
        """
        Get EV for each action at hero's first decision point.

        Args:
            state: State after cards are dealt

        Returns:
            Dict mapping action to expected value
        """
        # Navigate to hero's first decision point
        while not state.is_terminal():
            if state.is_chance_node():
                # Take expected path through chance nodes
                outcomes = state.chance_outcomes()
                if not outcomes:
                    break
                # Take most likely outcome (or first one)
                action = outcomes[0][0]
                state = state.child(action)
            elif state.current_player() == self.scenario.hero_position:
                # Found hero's decision point!
                break
            else:
                # Opponent's decision, follow policy
                action_probs = self.policy.action_probabilities(state)
                if not action_probs:
                    break
                # Take highest probability action (or first one)
                action = max(action_probs.items(), key=lambda x: x[1])[0]
                state = state.child(action)

        # Compute value for each of hero's actions
        action_values = {}
        if not state.is_terminal() and state.current_player() == self.scenario.hero_position:
            for action in state.legal_actions():
                next_state = state.child(action)
                value = self._compute_conditional_br(
                    next_state,
                    self.scenario.hero_position,
                    self.policy,
                    self.scenario.depth_limit
                )
                action_values[action] = value

        return action_values
