#!/usr/bin/env python3
"""
Unified Poker Solver Interface

Provides unified interface for all OpenSpiel CFR algorithms.

Supported algorithms:
- vanilla_cfr: Python vanilla CFR
- cfr_plus: Python CFR+
- dcfr: Python Discounted CFR
- lcfr: Python Linear CFR
- external_mccfr: External Sampling MCCFR
- outcome_mccfr: Outcome Sampling MCCFR
- cpp_cfr: C++ CFR
- cpp_cfr_plus: C++ CFR+

Requirements:
    source ~/open_spiel/venv/bin/activate

Example:
    from game_config import create_validation_config
    config = create_validation_config()

    solver = UnifiedPokerSolver(config, algorithm='cfr_plus')
    solver.solve(max_iterations=10000, check_interval=1000)
    policy = solver.get_average_policy()
"""

import pyspiel
import pickle
import time
import gc
from typing import Optional, Dict, Any
from pathlib import Path

# Import OpenSpiel algorithms
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import discounted_cfr
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr
from open_spiel.python.algorithms import exploitability

from game_config import PokerGameConfig
from solver_metrics import MetricsTracker, AdaptiveSchedule
from solver_logger import SolverLogger
from exploitability_metrics import (
    SampledExploitabilityCalculator,
    format_exploitability_report
)
from linear_external_mccfr import LinearExternalSamplingSolver


class UnifiedPokerSolver:
    """
    Unified interface for poker GTO solving algorithms.

    Wraps OpenSpiel CFR implementations with consistent API.
    """

    SUPPORTED_ALGORITHMS = {
        'vanilla_cfr': 'Python Vanilla CFR',
        'cfr_plus': 'Python CFR+',
        'dcfr': 'Python Discounted CFR',
        'lcfr': 'Python Linear CFR',
        'external_mccfr': 'External Sampling MCCFR (SIMPLE averaging)',
        'external_mccfr_full': 'External Sampling MCCFR (FULL averaging)',
        'lcfr_es': 'Linear-Weighted External Sampling MCCFR (LCFR-ES)',
        'outcome_mccfr': 'Outcome Sampling MCCFR',
        'cpp_cfr': 'C++ CFR',
        'cpp_cfr_plus': 'C++ CFR+',
    }

    def __init__(
        self,
        game_config: PokerGameConfig,
        algorithm: str = 'cfr_plus',
        **algorithm_kwargs
    ):
        """
        Initialize unified solver.

        Args:
            game_config: PokerGameConfig instance
            algorithm: Algorithm name (see SUPPORTED_ALGORITHMS)
            **algorithm_kwargs: Additional kwargs for specific algorithm
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Supported: {list(self.SUPPORTED_ALGORITHMS.keys())}"
            )

        self.game_config = game_config
        self.algorithm = algorithm
        self.algorithm_kwargs = algorithm_kwargs

        # Create game
        self.game = game_config.create_game()

        # Create solver
        self.solver = self._create_solver()

        # Initialize metrics and logging
        self.metrics_tracker = MetricsTracker(
            algorithm_name=self.SUPPORTED_ALGORITHMS[algorithm],
            game_description=str(game_config)
        )

        self.logger = SolverLogger(
            algorithm_name=self.SUPPORTED_ALGORITHMS[algorithm],
            game_description=str(game_config)
        )

        # Track iterations
        self.current_iteration = 0

    def _create_solver(self):
        """Create algorithm-specific solver instance."""
        if self.algorithm == 'vanilla_cfr':
            return cfr.CFRSolver(self.game)

        elif self.algorithm == 'cfr_plus':
            return cfr.CFRPlusSolver(self.game)

        elif self.algorithm == 'dcfr':
            # DCFR has configurable parameters
            alpha = self.algorithm_kwargs.get('alpha', 1.5)
            beta = self.algorithm_kwargs.get('beta', 0.0)
            gamma = self.algorithm_kwargs.get('gamma', 2.0)
            return discounted_cfr.DCFRSolver(
                self.game,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )

        elif self.algorithm == 'lcfr':
            # LCFR is a special case of DCFR with alpha=beta=gamma=1
            return discounted_cfr.LCFRSolver(self.game)

        elif self.algorithm == 'external_mccfr':
            # Support both 'SIMPLE' and 'FULL' averaging types
            average_type_str = self.algorithm_kwargs.get('average_type', 'SIMPLE')
            if average_type_str == 'FULL':
                avg_type = external_sampling_mccfr.AverageType.FULL
            else:
                avg_type = external_sampling_mccfr.AverageType.SIMPLE
            return external_sampling_mccfr.ExternalSamplingSolver(
                self.game,
                average_type=avg_type
            )

        elif self.algorithm == 'external_mccfr_full':
            # Alias for external_mccfr with FULL averaging
            return external_sampling_mccfr.ExternalSamplingSolver(
                self.game,
                average_type=external_sampling_mccfr.AverageType.FULL
            )

        elif self.algorithm == 'lcfr_es':
            # Linear-Weighted External Sampling MCCFR
            gamma = self.algorithm_kwargs.get('gamma', 1.0)
            alpha = self.algorithm_kwargs.get('alpha', None)
            beta = self.algorithm_kwargs.get('beta', None)
            return LinearExternalSamplingSolver(
                self.game,
                gamma=gamma,
                alpha=alpha,
                beta=beta
            )

        elif self.algorithm == 'outcome_mccfr':
            epsilon = self.algorithm_kwargs.get('epsilon', 0.6)
            return outcome_sampling_mccfr.OutcomeSamplingSolver(
                self.game,
                epsilon=epsilon
            )

        elif self.algorithm == 'cpp_cfr':
            return pyspiel.CFRSolver(self.game)

        elif self.algorithm == 'cpp_cfr_plus':
            return pyspiel.CFRPlusSolver(self.game)

        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _run_iteration(self):
        """Run a single iteration of the algorithm."""
        if self.algorithm in ['external_mccfr', 'external_mccfr_full', 'lcfr_es', 'outcome_mccfr']:
            # MCCFR uses iteration() method
            self.solver.iteration()
        else:
            # CFR variants use evaluate_and_update_policy()
            self.solver.evaluate_and_update_policy()

        self.current_iteration += 1

    def get_average_policy(self):
        """Get current average policy."""
        return self.solver.average_policy()

    def get_current_policy(self):
        """Get current policy (for debugging)."""
        if hasattr(self.solver, 'current_policy'):
            return self.solver.current_policy()
        return None

    def calculate_exploitability(self) -> float:
        """
        Calculate exploitability of current average policy.

        Uses C++ implementation when available for better performance and memory usage.

        Returns:
            Exploitability value (NashConv)
        """
        avg_policy = self.get_average_policy()

        # Try to use C++ implementation first (much faster and less memory)
        try:
            return pyspiel.nash_conv(self.game, avg_policy)
        except (AttributeError, TypeError):
            # Fall back to Python implementation
            return exploitability.nash_conv(
                self.game,
                avg_policy,
                return_only_nash_conv=True
            )

    def calculate_sampled_exploitability(
        self,
        confidence_level: float = 0.99,
        max_ci_width: float = 0.002,
        min_samples: int = 1000,
        max_samples: int = 10_000_000,
        check_interval: int = 1000
    ) -> Dict:
        """
        Calculate exploitability using adaptive Monte Carlo sampling.

        Samples random card deals and computes exact best response for each deal.
        Continues sampling until confidence interval is narrow enough.

        Recommended for final validation when full exploitability is too expensive.

        Args:
            confidence_level: Confidence level (e.g., 0.99 for 99% CI)
            max_ci_width: Stop when CI width < this value (e.g., 0.002 = 0.2%)
            min_samples: Minimum samples before checking CI
            max_samples: Maximum samples to prevent infinite loops
            check_interval: Check CI every N samples

        Returns:
            Dict with keys:
                - exploitability: Mean exploitability estimate
                - ci_lower: Lower bound of CI
                - ci_upper: Upper bound of CI
                - ci_width: Width of CI
                - num_samples: Number of samples used
                - confidence_level: Confidence level used
                - std_error: Standard error of the mean
        """
        avg_policy = self.get_average_policy()
        calculator = SampledExploitabilityCalculator(self.game, avg_policy)
        return calculator.calculate(
            confidence_level=confidence_level,
            max_ci_width=max_ci_width,
            min_samples=min_samples,
            max_samples=max_samples,
            check_interval=check_interval
        )

    def get_exploitability_report(
        self,
        confidence_level: float = 0.99,
        max_ci_width: float = 0.001,
        **kwargs
    ) -> str:
        """
        Generate formatted exploitability report with confidence interval.

        Runs sampled exploitability and formats result as publication-quality string.

        Args:
            confidence_level: CI level (default: 0.99)
            max_ci_width: Target CI width (default: 0.001 = 0.1%)
            **kwargs: Additional arguments for calculate_sampled_exploitability

        Returns:
            Formatted string like: "0.34% ± 0.08% (99% CI, 2,450,000 samples)"
        """
        result = self.calculate_sampled_exploitability(
            confidence_level=confidence_level,
            max_ci_width=max_ci_width,
            **kwargs
        )
        return format_exploitability_report(result)

    def solve(
        self,
        max_iterations: int,
        adaptive_schedule: Optional[AdaptiveSchedule] = None,
        checkpoint_interval: Optional[int] = None,
        checkpoint_dir: str = "checkpoints",
        checkpoint_prefix: Optional[str] = None,
        progress_interval: int = 100,
        skip_initial_exploitability: bool = False,
        use_sampled_exploitability: bool = True,
        final_sampled_exploitability: bool = False,
        sampled_exploit_confidence: float = 0.99,
        sampled_exploit_ci_width: float = 0.05,
        sampled_exploit_min_samples: int = 50,
        sampled_exploit_max_samples: int = 500,
        save_best_policy: bool = True,
        use_table_display: bool = True,
        exploit_history_size: int = 10
    ):
        """
        Run solver for specified iterations.

        Args:
            max_iterations: Maximum iterations to run
            adaptive_schedule: AdaptiveSchedule for exploitability testing
                              (default: 50k/100k/250k schedule)
            checkpoint_interval: Save checkpoint every N iterations (None = no checkpoints)
            checkpoint_dir: Directory to save checkpoints
            checkpoint_prefix: Prefix for checkpoint filenames (default: algorithm name)
                             Used for auto-resume: checkpoints with same prefix can be resumed
            progress_interval: Show progress update every N iterations (default: 100)
            skip_initial_exploitability: Skip initial exploitability check (saves memory for large games)
            use_sampled_exploitability: Use memory-safe sampled exploitability during solve loop (default: True, RECOMMENDED)
            final_sampled_exploitability: Run high-accuracy sampled exploitability at end (default: False)
            sampled_exploit_confidence: Confidence level for sampled exploitability (default: 0.99)
            sampled_exploit_ci_width: Target CI width for sampled exploitability (default: 0.05 = 5%)
            sampled_exploit_min_samples: Minimum samples for sampled exploitability (default: 50)
            sampled_exploit_max_samples: Maximum samples for sampled exploitability (default: 500)
            save_best_policy: Automatically save best policy when found (default: True)
            use_table_display: Use table-based progress display instead of text (default: True)
            exploit_history_size: Number of recent exploitability tests to show in table (default: 10)
        """
        if adaptive_schedule is None:
            adaptive_schedule = AdaptiveSchedule()

        # Update logger with max iterations
        self.logger.max_iterations = max_iterations
        self.logger.log_start()

        # Log exploitability method
        if use_sampled_exploitability:
            self.logger.log_info(
                f"Using SAMPLED exploitability (DEFAULT, memory-safe)"
            )
            self.logger.log_info(
                f"  Sampling params: {sampled_exploit_min_samples}-{sampled_exploit_max_samples} samples, "
                f"target CI width: {sampled_exploit_ci_width*100:.1f}%"
            )
        else:
            self.logger.log_info("⚠️  Using FULL exploitability (WARNING: May use massive memory, only for tiny test games!)")

        # Initial exploitability check (optional)
        last_check_iteration = 0
        if skip_initial_exploitability:
            self.logger.log_info("Skipping initial exploitability check (will check at first scheduled interval)")
            exploitability_value = 0.0  # Placeholder
            last_exploitability = 0.0
        else:
            self.logger.log_info("Calculating initial exploitability (this may take several minutes for large games)...")

            if use_sampled_exploitability:
                result = self.calculate_sampled_exploitability(
                    confidence_level=sampled_exploit_confidence,
                    max_ci_width=sampled_exploit_ci_width,
                    min_samples=sampled_exploit_min_samples,
                    max_samples=sampled_exploit_max_samples
                )
                exploitability_value = result['exploitability']
                self.logger.log_info(f"Initial exploitability: {format_exploitability_report(result)}")
            else:
                exploitability_value = self.calculate_exploitability()
                self.logger.log_info(f"Initial exploitability: {exploitability_value:.6f}")

            self.metrics_tracker.record_checkpoint(
                iteration=0,
                exploitability=exploitability_value
            )
            last_exploitability = exploitability_value

        self.logger.log_info(
            f"Exploitability schedule: {adaptive_schedule.get_schedule_description()}"
        )
        self.logger.log_info(f"Progress updates every {progress_interval:,} iterations")
        self.logger.log_info("")

        # Main solving loop
        while self.current_iteration < max_iterations:
            # Run iteration
            self._run_iteration()

            # Show progress update (without calculating exploitability)
            if self.current_iteration % progress_interval == 0:
                # Calculate current stats without exploitability
                elapsed = time.time() - self.metrics_tracker.start_time
                iters_per_sec = self.current_iteration / elapsed if elapsed > 0 else 0
                memory_mb = self.metrics_tracker.process.memory_info().rss / (1024 * 1024)
                next_check = adaptive_schedule.get_next_check(self.current_iteration)

                # Use table display or simple text
                if use_table_display:
                    self.logger.log_progress_table_row(
                        iteration=self.current_iteration,
                        time_elapsed=elapsed,
                        iters_per_sec=iters_per_sec,
                        memory_mb=memory_mb,
                        last_exploitability=last_exploitability
                    )
                else:
                    # Original simple text progress
                    self.logger.log_progress_simple(
                        iteration=self.current_iteration,
                        time_elapsed=elapsed,
                        iters_per_sec=iters_per_sec,
                        memory_mb=memory_mb,
                        next_check=next_check,
                        last_exploitability=last_exploitability
                    )

                # Memory management
                if self.current_iteration % 1000 == 0:
                    gc.collect()  # Force garbage collection

                    # Warn if memory usage is very high
                    memory_gb = memory_mb / 1024
                    if memory_gb > 10:
                        self.logger.log_warning(
                            f"High memory usage: {memory_gb:.1f} GB. "
                            "Consider using C++ algorithms or MCCFR for large games."
                        )

            # Check if we should test exploitability
            if adaptive_schedule.should_check(self.current_iteration, last_check_iteration):
                if use_sampled_exploitability:
                    result = self.calculate_sampled_exploitability(
                        confidence_level=sampled_exploit_confidence,
                        max_ci_width=sampled_exploit_ci_width,
                        min_samples=sampled_exploit_min_samples,
                        max_samples=sampled_exploit_max_samples
                    )
                    exploitability_value = result['exploitability']
                else:
                    exploitability_value = self.calculate_exploitability()

                last_exploitability = exploitability_value

                # Record metrics
                self.metrics_tracker.record_checkpoint(
                    iteration=self.current_iteration,
                    exploitability=exploitability_value
                )

                # Force garbage collection after expensive exploitability calculation
                gc.collect()

                # Get metrics for logging
                latest = self.metrics_tracker.get_latest_checkpoint()
                next_check = adaptive_schedule.get_next_check(self.current_iteration)

                # Check if this is a new best and save if requested
                is_new_best = (self.metrics_tracker.best_iteration == self.current_iteration)
                if is_new_best and save_best_policy:
                    best_path = self._save_best_policy(checkpoint_dir, checkpoint_prefix)
                    self.logger.log_info(f"★ NEW BEST policy saved: {best_path} (exploit: {exploitability_value:.6f})")

                # Display exploitability history table
                if use_table_display:
                    recent_checkpoints = self.metrics_tracker.get_recent_checkpoints(exploit_history_size)
                    self.logger.log_exploitability_table(
                        recent_checkpoints,
                        best_iteration=self.metrics_tracker.best_iteration
                    )
                else:
                    # Original simple text progress
                    self.logger.log_progress(
                        iteration=self.current_iteration,
                        exploitability=exploitability_value,
                        time_elapsed=latest.time_elapsed,
                        iters_per_sec=latest.iters_per_sec,
                        memory_mb=latest.memory_mb,
                        next_check=next_check,
                        convergence_rate=latest.convergence_rate
                    )

                last_check_iteration = self.current_iteration

            # Save checkpoint if requested
            if checkpoint_interval and self.current_iteration % checkpoint_interval == 0:
                checkpoint_path = self._save_checkpoint(checkpoint_dir, checkpoint_prefix)
                self.logger.log_checkpoint_save(self.current_iteration, checkpoint_path)

        # Final exploitability check
        if use_sampled_exploitability:
            result = self.calculate_sampled_exploitability(
                confidence_level=sampled_exploit_confidence,
                max_ci_width=sampled_exploit_ci_width,
                min_samples=sampled_exploit_min_samples,
                max_samples=sampled_exploit_max_samples
            )
            final_exploitability = result['exploitability']
        else:
            final_exploitability = self.calculate_exploitability()

        self.metrics_tracker.record_checkpoint(
            iteration=self.current_iteration,
            exploitability=final_exploitability
        )

        # Optional: Run high-accuracy sampled exploitability at the end (with tighter CI than periodic checks)
        if final_sampled_exploitability:
            self.logger.log_info("")
            self.logger.log_info("=" * 60)
            self.logger.log_info("Running final HIGH-ACCURACY sampled exploitability calculation...")
            self.logger.log_info(f"Target: {sampled_exploit_confidence*100:.0f}% CI width < {sampled_exploit_ci_width/10*100:.2f}%")
            self.logger.log_info("=" * 60)
            self.logger.log_info("")

            # Use 10x tighter CI for final measurement
            sampled_result = self.calculate_sampled_exploitability(
                confidence_level=sampled_exploit_confidence,
                max_ci_width=sampled_exploit_ci_width / 10,  # 10x tighter
                min_samples=sampled_exploit_min_samples * 10,  # More samples
                max_samples=min(sampled_exploit_max_samples * 10, 10_000_000)  # Up to 10M
            )
            self.metrics_tracker.record_sampled_exploitability(sampled_result)

        # Log completion
        latest = self.metrics_tracker.get_latest_checkpoint()
        self.logger.log_completion(
            total_iterations=self.current_iteration,
            final_exploitability=final_exploitability,
            total_time=latest.time_elapsed,
            final_memory_mb=latest.memory_mb
        )

        # Log sampled exploitability if available
        if self.metrics_tracker.sampled_exploitability_result is not None:
            report = self.metrics_tracker.get_formatted_sampled_exploit_report()
            self.logger.log_info("")
            self.logger.log_info("=" * 60)
            self.logger.log_info(f"Final Sampled Exploitability: {report}")
            self.logger.log_info("=" * 60)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of solving metrics."""
        return self.metrics_tracker.get_convergence_summary()

    def save_metrics(self, csv_path: str, json_path: Optional[str] = None):
        """
        Save metrics to files.

        Args:
            csv_path: Path to save CSV metrics
            json_path: Path to save JSON summary (optional)
        """
        self.metrics_tracker.save_csv(csv_path)
        if json_path:
            self.metrics_tracker.save_json_summary(json_path)

    def save_policy(self, policy_path: str):
        """
        Save average policy to file.

        Args:
            policy_path: Path to save policy
        """
        Path(policy_path).parent.mkdir(parents=True, exist_ok=True)

        avg_policy = self.get_average_policy()
        with open(policy_path, 'wb') as f:
            pickle.dump(avg_policy, f, pickle.HIGHEST_PROTOCOL)

        self.logger.log_info(f"Policy saved to: {policy_path}")

    def _save_best_policy(self, checkpoint_dir: str, checkpoint_prefix: Optional[str] = None) -> str:
        """
        Save best policy encountered so far.

        Args:
            checkpoint_dir: Directory to save policy
            checkpoint_prefix: Optional prefix for filename (default: algorithm name)

        Returns:
            Path to saved policy
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Use prefix or default to algorithm name
        if checkpoint_prefix is None:
            prefix = self.algorithm
        else:
            prefix = checkpoint_prefix

        best_policy_path = f"{checkpoint_dir}/{prefix}_best_policy.pkl"

        avg_policy = self.get_average_policy()
        with open(best_policy_path, 'wb') as f:
            pickle.dump({
                'policy': avg_policy,
                'iteration': self.current_iteration,
                'exploitability': self.metrics_tracker.best_exploitability,
                'algorithm': self.algorithm,
                'game_config': self.game_config
            }, f, pickle.HIGHEST_PROTOCOL)

        return best_policy_path

    def _save_checkpoint(self, checkpoint_dir: str, checkpoint_prefix: Optional[str] = None) -> str:
        """
        Save solver checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
            checkpoint_prefix: Optional prefix for checkpoint filename (default: algorithm name)

        Returns:
            Path to saved checkpoint
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Use prefix or default to algorithm name
        if checkpoint_prefix is None:
            prefix = self.algorithm
        else:
            prefix = checkpoint_prefix

        checkpoint_path = (
            f"{checkpoint_dir}/{prefix}_iter_{self.current_iteration}.pkl"
        )

        # Save both solver and current iteration
        checkpoint_data = {
            'solver': self.solver,
            'current_iteration': self.current_iteration,
            'algorithm': self.algorithm,
            'game_config': self.game_config,
            'algorithm_kwargs': self.algorithm_kwargs
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f, pickle.HIGHEST_PROTOCOL)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load solver from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        # Handle both old format (just solver) and new format (dict with metadata)
        if isinstance(data, dict) and 'solver' in data:
            self.solver = data['solver']
            self.current_iteration = data.get('current_iteration', 0)
            self.logger.log_info(f"Loaded checkpoint from: {checkpoint_path}")
            self.logger.log_info(f"Resuming from iteration: {self.current_iteration:,}")
        else:
            # Old format - just the solver object
            self.solver = data
            # Try to extract iteration from filename
            import re
            match = re.search(r'iter_(\d+)', checkpoint_path)
            if match:
                self.current_iteration = int(match.group(1))
                self.logger.log_info(f"Loaded checkpoint from: {checkpoint_path}")
                self.logger.log_info(f"Resuming from iteration: {self.current_iteration:,} (extracted from filename)")
            else:
                self.logger.log_warning(f"Could not determine iteration from checkpoint filename: {checkpoint_path}")
                self.current_iteration = 0

    @classmethod
    def find_latest_checkpoint(cls, checkpoint_dir: str, checkpoint_prefix: str) -> Optional[str]:
        """
        Find the latest checkpoint file matching the given prefix.

        Args:
            checkpoint_dir: Directory containing checkpoints
            checkpoint_prefix: Prefix to match (e.g., 'cfr_plus_3p_kuhn')

        Returns:
            Path to latest checkpoint, or None if not found
        """
        import glob
        import re

        checkpoint_pattern = f"{checkpoint_dir}/{checkpoint_prefix}_iter_*.pkl"
        matching_files = glob.glob(checkpoint_pattern)

        if not matching_files:
            return None

        # Extract iteration numbers and find the maximum
        max_iter = -1
        latest_file = None

        for filepath in matching_files:
            match = re.search(r'iter_(\d+)\.pkl$', filepath)
            if match:
                iteration = int(match.group(1))
                if iteration > max_iter:
                    max_iter = iteration
                    latest_file = filepath

        return latest_file

    @classmethod
    def get_algorithm_description(cls, algorithm: str) -> str:
        """Get description of algorithm."""
        return cls.SUPPORTED_ALGORITHMS.get(algorithm, "Unknown")

    @classmethod
    def list_algorithms(cls) -> Dict[str, str]:
        """List all supported algorithms."""
        return cls.SUPPORTED_ALGORITHMS.copy()
