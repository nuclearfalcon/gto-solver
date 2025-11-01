"""
Validation Utilities for GPU CFR

Tools for validating GPU CFR results against CPU solvers.

Since we're implementing a new solver from scratch, rigorous validation is
critical. This module provides utilities to:
- Compare GPU vs CPU policies
- Check exploitability convergence
- Validate regret values
- Ensure numerical stability

Usage:
    from matrix_cfr.validation import validate_against_cpu

    gpu_policy = gpu_solver.get_average_policy()
    cpu_policy = cpu_solver.get_average_policy()

    validate_against_cpu(gpu_policy, cpu_policy, tolerance=1e-6)
"""

import pyspiel
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_against_cpu(
    gpu_policy,
    cpu_policy,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Validate GPU solver results against CPU solver.

    Args:
        gpu_policy: Policy from GPU solver
        cpu_policy: Policy from CPU solver (ground truth)
        tolerance: Maximum allowed difference
        verbose: Print detailed comparison

    Returns:
        (is_valid, metrics_dict)
    """
    # TODO: Implement policy comparison
    # TODO: Compute L1/L2 distance between policies
    # TODO: Check per-infoset strategy differences
    # TODO: Return detailed metrics

    raise NotImplementedError("Policy validation not yet implemented")


def compare_exploitability(
    gpu_exploit: float,
    cpu_exploit: float,
    tolerance: float = 0.01
) -> bool:
    """
    Compare exploitability values from GPU and CPU.

    Args:
        gpu_exploit: Exploitability from GPU solver
        cpu_exploit: Exploitability from CPU solver
        tolerance: Maximum allowed difference (absolute)

    Returns:
        True if within tolerance
    """
    diff = abs(gpu_exploit - cpu_exploit)
    is_valid = diff <= tolerance

    if not is_valid:
        logger.warning(
            f"Exploitability mismatch: GPU={gpu_exploit:.6f}, "
            f"CPU={cpu_exploit:.6f}, diff={diff:.6f}"
        )

    return is_valid


def validate_regrets(
    gpu_regrets: np.ndarray,
    cpu_regrets: np.ndarray,
    tolerance: float = 1e-4
) -> Tuple[bool, float]:
    """
    Validate cumulative regret values.

    Args:
        gpu_regrets: Regrets from GPU solver
        cpu_regrets: Regrets from CPU solver
        tolerance: Maximum allowed relative error

    Returns:
        (is_valid, max_error)
    """
    # TODO: Implement regret comparison
    # TODO: Handle numerical precision issues
    # TODO: Check for catastrophic divergence

    raise NotImplementedError("Regret validation not yet implemented")


def check_strategy_sum(policy, tolerance: float = 1e-6) -> bool:
    """
    Check that strategies sum to 1.0 at each infoset.

    Args:
        policy: Policy to validate
        tolerance: Maximum allowed deviation from 1.0

    Returns:
        True if all infosets have valid probability distributions
    """
    # TODO: Implement strategy sum validation
    # TODO: Check for negative probabilities
    # TODO: Check for NaN/Inf values

    raise NotImplementedError("Strategy sum validation not yet implemented")


def validate_convergence_trajectory(
    gpu_trajectory: list,
    cpu_trajectory: list,
    tolerance: float = 0.05
) -> bool:
    """
    Validate that exploitability converges similarly on GPU and CPU.

    Args:
        gpu_trajectory: List of (iteration, exploitability) from GPU
        cpu_trajectory: List of (iteration, exploitability) from CPU
        tolerance: Maximum allowed relative difference in convergence rate

    Returns:
        True if convergence patterns match
    """
    # TODO: Implement trajectory comparison
    # TODO: Check convergence rates
    # TODO: Detect divergence early

    raise NotImplementedError("Convergence validation not yet implemented")


class ValidationSuite:
    """
    Comprehensive validation suite for GPU CFR implementation.

    Runs a battery of tests to ensure GPU implementation matches CPU.
    """

    def __init__(self, game: pyspiel.Game):
        """
        Initialize validation suite for a game.

        Args:
            game: OpenSpiel game to test
        """
        self.game = game
        self.test_results = {}

    def run_all_tests(
        self,
        gpu_solver,
        cpu_solver,
        iterations: int = 10000
    ) -> Dict:
        """
        Run all validation tests.

        Args:
            gpu_solver: MatrixCFRSolver instance
            cpu_solver: CPU solver instance (UnifiedPokerSolver)
            iterations: Number of iterations to test

        Returns:
            Dictionary of test results
        """
        # TODO: Implement comprehensive test suite
        # TODO: Test policy convergence
        # TODO: Test exploitability convergence
        # TODO: Test regret updates
        # TODO: Test numerical stability

        raise NotImplementedError("Validation suite not yet implemented")

    def print_results(self):
        """
        Print validation results in readable format.
        """
        # TODO: Implement results printing
        pass


# TODO: Add statistical tests (KS test, etc.)
# TODO: Add visualization utilities for convergence comparison
# TODO: Add automated regression tests
