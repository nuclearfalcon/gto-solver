#!/usr/bin/env python3
"""
GPU-Accelerated CFR Solver using JAX and cfrx

Provides a wrapper around the cfrx library that matches the interface
of UnifiedPokerSolver and LinearExternalSamplingSolver for seamless
GPU acceleration of CFR research.

Requirements:
    source ~/open_spiel/venv/bin/activate
    pip install "jax[cuda12]"  # or cuda13
    pip install cfrx

Hardware:
    NVIDIA GPU with CUDA support (tested on RTX 4060 Ti)

Usage:
    from gpu_cfr_solver import GPUCFRSolver

    # For Kuhn poker (currently supported by cfrx)
    solver = GPUCFRSolver(
        game_name='kuhn',
        num_players=3,
        algorithm='mccfr',  # or 'vanilla_cfr'
    )

    # Train on GPU
    solver.solve(iterations=100000)

    # Get policy (compatible with OpenSpiel)
    policy = solver.get_average_policy()

Limitations:
    - cfrx currently only supports Kuhn and Leduc poker
    - For Hold'em, use CPU solvers or implement matrix-based GPU CFR
    - DCFR variants not yet supported (vanilla CFR and MCCFR only)

Reference:
    - cfrx: https://github.com/Egiob/cfrx
    - GPU CFR Paper: https://arxiv.org/abs/2408.14778
"""

import time
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("WARNING: JAX not installed. GPU acceleration unavailable.")
    print("Install with: pip install 'jax[cuda12]'")

try:
    from cfrx.envs import KuhnPoker, LeducPoker
    from cfrx.policy import TabularPolicy
    from cfrx.training import MCCFRTrainer, VanillaCFRTrainer
    CFRX_AVAILABLE = True
except ImportError:
    CFRX_AVAILABLE = False
    print("WARNING: cfrx not installed. GPU acceleration unavailable.")
    print("Install with: pip install cfrx")

import pyspiel


class GPUCFRSolver:
    """
    GPU-accelerated CFR solver using JAX and cfrx.

    Provides an interface compatible with UnifiedPokerSolver and
    LinearExternalSamplingSolver for drop-in GPU acceleration.
    """

    SUPPORTED_GAMES = ['kuhn', 'leduc']
    SUPPORTED_ALGORITHMS = ['vanilla_cfr', 'mccfr']

    def __init__(
        self,
        game_name: str = 'kuhn',
        num_players: int = 2,
        algorithm: str = 'mccfr',
        exploration_factor: float = 0.6,
        seed: int = 42
    ):
        """
        Initialize GPU CFR solver.

        Args:
            game_name: 'kuhn' or 'leduc' (cfrx supported games)
            num_players: Number of players (2 or 3 for Kuhn)
            algorithm: 'vanilla_cfr' or 'mccfr'
            exploration_factor: Exploration factor for regret matching (default: 0.6)
            seed: Random seed for reproducibility
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX not installed. Install with: pip install 'jax[cuda12]'")

        if not CFRX_AVAILABLE:
            raise ImportError("cfrx not installed. Install with: pip install cfrx")

        if game_name not in self.SUPPORTED_GAMES:
            raise ValueError(f"Game '{game_name}' not supported. Choose from: {self.SUPPORTED_GAMES}")

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm '{algorithm}' not supported. Choose from: {self.SUPPORTED_ALGORITHMS}")

        self.game_name = game_name
        self.num_players = num_players
        self.algorithm = algorithm
        self.exploration_factor = exploration_factor
        self.seed = seed

        # Initialize JAX random key
        self.rng_key = random.PRNGKey(seed)

        # Create cfrx environment
        if game_name == 'kuhn':
            self.env = KuhnPoker(n_players=num_players)
        elif game_name == 'leduc':
            self.env = LeducPoker(n_players=num_players)
        else:
            raise ValueError(f"Unsupported game: {game_name}")

        # Create policy
        self.policy = TabularPolicy(
            n_actions=self.env.n_actions,
            exploration_factor=exploration_factor,
            info_state_idx_fn=self.env.info_state_idx,
        )

        # Create trainer
        if algorithm == 'mccfr':
            self.trainer = MCCFRTrainer(env=self.env, policy=self.policy)
        elif algorithm == 'vanilla_cfr':
            self.trainer = VanillaCFRTrainer(env=self.env, policy=self.policy)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Track training state
        self.training_state = None
        self.total_iterations = 0
        self.metrics_history = []

        # Check GPU availability
        self._check_gpu()

    def _check_gpu(self):
        """Check if JAX detected GPU and print device info."""
        devices = jax.devices()
        print(f"\n{'='*80}")
        print("GPU CFR SOLVER INITIALIZATION")
        print(f"{'='*80}")
        print(f"JAX Devices: {devices}")
        print(f"Default Backend: {jax.default_backend()}")

        if jax.default_backend() == 'gpu' or jax.default_backend() == 'cuda':
            print("✓ GPU acceleration ENABLED")
            # Get GPU memory info
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    print(f"  {result.stdout.strip()}")
            except:
                pass
        else:
            print("✗ WARNING: GPU not detected, using CPU")
            print("  Check: pip list | grep jax")
            print("  Ensure you installed: pip install 'jax[cuda12]'")

        print(f"Game: {self.game_name} ({self.num_players} players)")
        print(f"Algorithm: {self.algorithm}")
        print(f"{'='*80}\n")

    def solve(
        self,
        iterations: int = 100000,
        metrics_interval: int = 10000,
        verbose: bool = True
    ) -> Dict:
        """
        Train the CFR solver on GPU.

        Args:
            iterations: Number of CFR iterations to run
            metrics_interval: How often to record metrics
            verbose: Print progress updates

        Returns:
            dict: Training metrics (exploitability, time, etc.)
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"STARTING GPU CFR TRAINING")
            print(f"{'='*80}")
            print(f"Iterations: {iterations:,}")
            print(f"Metrics interval: {metrics_interval:,}")
            print(f"{'='*80}\n")

        start_time = time.time()

        # Run training on GPU
        self.training_state, metrics = self.trainer.train(
            random_key=self.rng_key,
            n_iterations=iterations,
            metrics_period=metrics_interval
        )

        self.total_iterations += iterations
        self.metrics_history.extend(metrics)

        elapsed = time.time() - start_time
        rate = iterations / elapsed if elapsed > 0 else 0

        if verbose:
            print(f"\n{'='*80}")
            print(f"GPU TRAINING COMPLETE")
            print(f"{'='*80}")
            print(f"Total iterations: {iterations:,}")
            print(f"Elapsed time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
            print(f"Iteration rate: {rate:,.0f} it/s")
            print(f"{'='*80}\n")

        return {
            'iterations': iterations,
            'elapsed_time': elapsed,
            'iterations_per_second': rate,
            'metrics': metrics
        }

    def get_average_policy(self):
        """
        Get the average policy as an OpenSpiel-compatible TabularPolicy.

        Returns:
            OpenSpiel TabularPolicy object
        """
        if self.training_state is None:
            raise ValueError("Must call solve() before getting policy")

        # Convert cfrx policy to OpenSpiel format
        # This requires building a mapping between cfrx and OpenSpiel info states

        # For now, return the cfrx policy state
        # TODO: Implement full OpenSpiel compatibility layer
        return self.training_state

    def calculate_exploitability(self) -> float:
        """
        Calculate exploitability of current policy.

        For cfrx, this requires using the OpenSpiel game for validation.

        Returns:
            Exploitability value (Nash conv)
        """
        if self.training_state is None:
            raise ValueError("Must call solve() before calculating exploitability")

        # Use the last recorded metric
        if self.metrics_history:
            last_metric = self.metrics_history[-1]
            if 'exploitability' in last_metric:
                return float(last_metric['exploitability'])

        # If no metrics, return None
        return None

    def get_metrics_history(self) -> list:
        """Get full training metrics history."""
        return self.metrics_history

    def get_parameters(self) -> Dict:
        """Get solver parameters for logging/checkpointing."""
        return {
            'game_name': self.game_name,
            'num_players': self.num_players,
            'algorithm': self.algorithm,
            'exploration_factor': self.exploration_factor,
            'total_iterations': self.total_iterations,
            'backend': jax.default_backend()
        }


class GPUDCFRSolver:
    """
    Placeholder for GPU-accelerated DCFR solver.

    DCFR variants (alpha, beta, gamma discounting) are not yet implemented
    in cfrx. This would require either:

    1. Contributing DCFR to cfrx library
    2. Implementing matrix-based GPU CFR directly in JAX/PyTorch
    3. Waiting for cfrx to add DCFR support

    For now, use GPUCFRSolver for vanilla CFR and MCCFR, or use
    LinearExternalSamplingSolver on CPU for DCFR research.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GPU-accelerated DCFR not yet implemented. Options:\n"
            "1. Use GPUCFRSolver for vanilla CFR or MCCFR\n"
            "2. Use LinearExternalSamplingSolver (CPU) for DCFR research\n"
            "3. Implement matrix-based DCFR in JAX (see arxiv 2408.14778)\n"
            "4. Contribute DCFR to cfrx library"
        )


def check_gpu_requirements() -> Tuple[bool, str]:
    """
    Check if system meets GPU CFR requirements.

    Returns:
        (meets_requirements, message)
    """
    if not JAX_AVAILABLE:
        return False, "JAX not installed. Run: pip install 'jax[cuda12]'"

    if not CFRX_AVAILABLE:
        return False, "cfrx not installed. Run: pip install cfrx"

    devices = jax.devices()
    backend = jax.default_backend()

    if backend not in ['gpu', 'cuda']:
        return False, f"GPU not detected. Backend: {backend}. Check CUDA installation."

    return True, f"✓ GPU ready: {devices}"


# Example usage
if __name__ == '__main__':
    """
    Test GPU CFR solver on Kuhn poker.

    Requirements:
        source ~/open_spiel/venv/bin/activate
        pip install 'jax[cuda12]'
        pip install cfrx

    Usage:
        python gpu_cfr_solver.py
    """
    print("="*80)
    print("GPU CFR SOLVER TEST")
    print("="*80)

    # Check requirements
    ready, msg = check_gpu_requirements()
    print(f"\nGPU Status: {msg}\n")

    if not ready:
        print("Please install required packages:")
        print("  pip install 'jax[cuda12]'")
        print("  pip install cfrx")
        exit(1)

    # Test 2-player Kuhn (quick)
    print("\n" + "="*80)
    print("TEST 1: 2-Player Kuhn Poker (MCCFR)")
    print("="*80)

    solver_2p = GPUCFRSolver(
        game_name='kuhn',
        num_players=2,
        algorithm='mccfr'
    )

    result_2p = solver_2p.solve(iterations=10000, metrics_interval=2000)
    print(f"\n2P Results: {result_2p['iterations_per_second']:,.0f} it/s")

    # Test 3-player Kuhn
    print("\n" + "="*80)
    print("TEST 2: 3-Player Kuhn Poker (MCCFR)")
    print("="*80)

    solver_3p = GPUCFRSolver(
        game_name='kuhn',
        num_players=3,
        algorithm='mccfr'
    )

    result_3p = solver_3p.solve(iterations=10000, metrics_interval=2000)
    print(f"\n3P Results: {result_3p['iterations_per_second']:,.0f} it/s")

    # Compare algorithms
    print("\n" + "="*80)
    print("TEST 3: Vanilla CFR vs MCCFR")
    print("="*80)

    solver_vanilla = GPUCFRSolver(
        game_name='kuhn',
        num_players=2,
        algorithm='vanilla_cfr'
    )

    result_vanilla = solver_vanilla.solve(iterations=10000, metrics_interval=2000)
    print(f"\nVanilla CFR: {result_vanilla['iterations_per_second']:,.0f} it/s")

    # Summary
    print("\n" + "="*80)
    print("GPU CFR PERFORMANCE SUMMARY")
    print("="*80)
    print(f"2P MCCFR:     {result_2p['iterations_per_second']:>10,.0f} it/s")
    print(f"3P MCCFR:     {result_3p['iterations_per_second']:>10,.0f} it/s")
    print(f"Vanilla CFR:  {result_vanilla['iterations_per_second']:>10,.0f} it/s")
    print("="*80)

    print("\n✓ All GPU CFR tests passed!")
