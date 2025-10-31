#!/usr/bin/env python3
"""
Linear-Weighted External Sampling MCCFR (LCFR-ES)

Combines:
- External Sampling MCCFR (explores all acting player branches, samples opponents)
- Linear iteration weighting for average strategy (iteration^gamma)
- Optional: Linear regret discounting (iteration^alpha, iteration^beta)

This ensures all branches are "fully baked" for training applications while
achieving faster convergence than standard external sampling.

Requirements:
    source ~/open_spiel/venv/bin/activate

References:
    - External Sampling: Lanctot (2009), Lanctot et al. (2013)
    - Linear CFR (LCFR): Brown & Sandholm (2019)
    - Discounted CFR: Brown & Sandholm (2019)

Key insight for 3+ players:
    External sampling ensures all branches are updated every iteration,
    unlike outcome sampling which can leave zero-probability branches undertrained.
"""

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mccfr


class LinearExternalSamplingSolver(mccfr.MCCFRSolverBase):
    """
    Linear-Weighted External Sampling MCCFR (LCFR-ES).

    Extends external sampling MCCFR with:
    1. Linear iteration weighting for average strategy (t^gamma)
    2. Optional regret discounting (t^alpha, t^beta)

    For 3+ player games, this provides:
    - Fully explored game tree (external sampling)
    - Faster convergence (linear weighting)
    - All branches trained regardless of policy (no zero-prob issues)
    """

    def __init__(
        self,
        game,
        gamma: float = 1.0,
        alpha: float = None,
        beta: float = None
    ):
        """
        Initialize LCFR-ES solver.

        Args:
            game: OpenSpiel game instance
            gamma: Iteration weighting exponent for average strategy
                  gamma=1.0 (default): linear weighting (t)
                  gamma=2.0: quadratic weighting (t²)
                  gamma=0.0: no weighting (uniform)
            alpha: Regret discounting exponent for positive regrets (None = no discount)
            beta: Regret discounting exponent for negative regrets (None = no discount)

        Note on parameters:
            - gamma controls how much to favor later iterations in averaging
            - alpha/beta control regret growth/decay (LCFR uses alpha=beta=1)
            - For pure LCFR-ES, use gamma=1.0, alpha=None, beta=None
        """
        super().__init__(game)

        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        # Iteration counter (starts at 0, incremented before use, matching OpenSpiel)
        self._iteration = 0

        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
            "LCFR-ES requires sequential games. If you're trying to run it " +
            'on a simultaneous (or normal-form) game, please first transform it ' +
            'using turn_based_simultaneous_game.')

    def iteration(self):
        """
        Performs one iteration of LCFR-ES.

        An iteration consists of:
        1. INCREMENT iteration counter (CRITICAL: must be first, matching OpenSpiel)
        2. One regret update episode for each player (external sampling)
        3. Apply regret discounting (using current iteration)
        4. One weighted average policy update pass (using current iteration weight)
        """
        # INCREMENT FIRST (critical for correct discount factors and weights)
        self._iteration += 1

        # Update regrets for each player (external sampling)
        for player in range(self._num_players):
            self._update_regrets(self._game.new_initial_state(), player)

        # Apply regret discounting AFTER regret updates (using current iteration)
        if self.alpha is not None or self.beta is not None:
            self._discount_regrets()

        # Update average policy with current iteration weighting
        reach_probs = np.ones(self._num_players, dtype=np.float64)
        weight = self._iteration ** self.gamma
        self._weighted_update_average(self._game.new_initial_state(), reach_probs, weight)

    def _weighted_update_average(self, state, reach_probs, weight):
        """
        Update average policy with iteration weighting.

        This is similar to FULL averaging in external_sampling_mccfr.py,
        but multiplies the contribution by the iteration weight.

        Args:
            state: Current game state
            reach_probs: Reach probabilities for each player
            weight: Iteration weight (iteration^gamma)
        """
        if state.is_terminal():
            return

        if state.is_chance_node():
            for action in state.legal_actions():
                self._weighted_update_average(state.child(action), reach_probs, weight)
            return

        # Early exit if all reach probs are zero
        if np.sum(reach_probs) == 0:
            return

        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)

        # Get current policy (regret matching)
        infostate_info = self._lookup_infostate_info(info_state_key, num_legal_actions)
        policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX], num_legal_actions)

        # Recurse into children
        for action_idx in range(num_legal_actions):
            new_reach_probs = np.copy(reach_probs)
            new_reach_probs[cur_player] *= policy[action_idx]
            self._weighted_update_average(
                state.child(legal_actions[action_idx]),
                new_reach_probs,
                weight
            )

        # Update cumulative policy with iteration weighting
        for action_idx in range(num_legal_actions):
            # Weighted contribution: weight * reach_prob * policy
            contribution = weight * reach_probs[cur_player] * policy[action_idx]
            self._add_avstrat(info_state_key, action_idx, contribution)

    def _discount_regrets(self):
        """
        Apply linear regret discounting to all information states.

        Discounting formula (LCFR):
        - Positive regrets: multiply by t^alpha / (t^alpha + 1)
        - Negative regrets: multiply by t^beta / (t^beta + 1)

        CRITICAL FIX: When beta=0, research intends "no discounting" (1.0),
        NOT the formula result t^0/(t^0+1)=0.5 which causes "regret amnesia".

        This gradually reduces the influence of early regrets.
        """
        if self.alpha is None and self.beta is None:
            return

        t = self._iteration

        # Calculate positive regret discount factor
        if self.alpha is not None and self.alpha > 0:
            pos_discount = (t ** self.alpha) / ((t ** self.alpha) + 1)
        elif self.alpha is not None and self.alpha == 0:
            # alpha=0: explicit constant discount of 0.5 (exponential moving average)
            pos_discount = 0.5
        else:
            # alpha=None: no discounting
            pos_discount = 1.0

        # Calculate negative regret discount factor
        # CRITICAL FIX: beta=0 means "no discount" (1.0), NOT formula with t^0
        if self.beta is not None and self.beta > 0:
            neg_discount = (t ** self.beta) / ((t ** self.beta) + 1)
        elif self.beta is not None and self.beta == 0:
            # beta=0: NO DISCOUNTING (1.0), fixes "regret amnesia" bug
            neg_discount = 1.0
        else:
            # beta=None: no discounting
            neg_discount = 1.0

        # Apply discounting to all regrets
        for info_state_key in self._infostates:
            regret = self._infostates[info_state_key][mccfr.REGRET_INDEX]
            for action_idx in range(len(regret)):
                if regret[action_idx] > 0:
                    regret[action_idx] *= pos_discount
                elif regret[action_idx] < 0:
                    regret[action_idx] *= neg_discount

    def _update_regrets(self, state, player):
        """
        Run one episode of external sampling to update regrets.

        External sampling:
        - At opponent nodes: sample one action according to policy
        - At my nodes: iterate over all legal actions
        - At chance nodes: sample one outcome

        Args:
            state: Current game state
            player: Player to update regrets for

        Returns:
            value: Expected value for the player
        """
        if state.is_terminal():
            return state.player_return(player)

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            outcome = np.random.choice(outcomes, p=probs)
            return self._update_regrets(state.child(outcome), player)

        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)

        # Get current policy (regret matching)
        infostate_info = self._lookup_infostate_info(info_state_key, num_legal_actions)
        policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX], num_legal_actions)

        value = 0
        child_values = np.zeros(num_legal_actions, dtype=np.float64)

        if cur_player != player:
            # Opponent node: sample one action
            action_idx = np.random.choice(np.arange(num_legal_actions), p=policy)
            value = self._update_regrets(state.child(legal_actions[action_idx]), player)
        else:
            # My node: iterate over all actions (external sampling)
            for action_idx in range(num_legal_actions):
                child_values[action_idx] = self._update_regrets(
                    state.child(legal_actions[action_idx]),
                    player
                )
            value = np.dot(policy, child_values)

            # Update regrets (counterfactual regret)
            for action_idx in range(num_legal_actions):
                cfr = child_values[action_idx] - value
                self._add_regret(info_state_key, action_idx, cfr)

        return value

    def get_iteration(self):
        """Get current iteration count."""
        return self._iteration

    def get_parameters(self):
        """Get solver parameters for logging/checkpointing."""
        return {
            'gamma': self.gamma,
            'alpha': self.alpha,
            'beta': self.beta,
            'iteration': self._iteration
        }


def create_lcfr_es_variants():
    """
    Factory function to create common LCFR-ES variants.

    Returns:
        dict: Maps variant names to parameter configurations
    """
    return {
        # Pure LCFR-ES: Linear weighting, no discounting
        'lcfr_es': {
            'gamma': 1.0,
            'alpha': None,
            'beta': None,
            'description': 'Linear-weighted averaging, no regret discounting'
        },

        # LCFR-ES with quadratic weighting
        'lcfr_es_quad': {
            'gamma': 2.0,
            'alpha': None,
            'beta': None,
            'description': 'Quadratic-weighted averaging (t²), no regret discounting'
        },

        # Full LCFR: Linear weighting + linear regret discounting
        'full_lcfr_es': {
            'gamma': 1.0,
            'alpha': 1.0,
            'beta': 1.0,
            'description': 'Linear weighting + linear regret discounting (full LCFR)'
        },

        # DCFR-ES: Discounted CFR with external sampling
        'dcfr_es': {
            'gamma': 2.0,
            'alpha': 1.5,
            'beta': 0.0,
            'description': 'DCFR parameters (gamma=2, alpha=1.5, beta=0)'
        }
    }


# Example usage
if __name__ == '__main__':
    """
    Quick test of LCFR-ES on 3-player Kuhn poker.

    Requirements:
        source ~/open_spiel/venv/bin/activate

    Usage:
        python linear_external_mccfr.py
    """
    print("Testing LCFR-ES on 3-player Kuhn Poker\n")

    # Create 3-player Kuhn poker
    game = pyspiel.load_game("kuhn_poker", {"players": 3})

    # Create LCFR-ES solver
    print("Creating LCFR-ES solver (gamma=1.0, no discounting)")
    solver = LinearExternalSamplingSolver(game, gamma=1.0)

    print(f"Running {10000:,} iterations...\n")

    # Run 10k iterations with periodic checks
    import time
    check_interval = 2000
    start_time = time.time()

    for i in range(1, 10001):
        solver.iteration()

        # Show frequent progress (every 100 iterations or first iteration)
        if i == 1 or i % 100 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (10000 - i) / rate if rate > 0 else 0
            print(f"\r{i:,}/10,000 iterations | Rate: {rate:6.0f} it/s | ETA: {eta:.1f}s | Elapsed: {elapsed:.1f}s", end='', flush=True)

        if i % check_interval == 0:
            print()  # New line before detailed check
            # Calculate exploitability
            avg_policy = solver.average_policy()
            try:
                nash_conv = pyspiel.nash_conv(game, avg_policy)
            except:
                from open_spiel.python.algorithms import exploitability
                nash_conv = exploitability.nash_conv(game, avg_policy, return_only_nash_conv=True)

            print(f"Iteration {i:,}: Nash Conv = {nash_conv:.6f}")

    print("\nLCFR-ES test complete!")
    print(f"Parameters: {solver.get_parameters()}")
