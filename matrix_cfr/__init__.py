"""
Matrix-based GPU CFR Implementation

This package implements Counterfactual Regret Minimization using matrix operations
on GPU, based on the approach described in arXiv:2408.14778v5.

Key components:
- game_to_matrix: Converts OpenSpiel game trees to sparse matrix representation
- matrix_cfr_solver: GPU-accelerated CFR solver using JAX
- gpu_memory: VRAM management and optimization utilities
- validation: Tools for validating GPU results against CPU solvers

Requires:
- JAX with CUDA support
- NVIDIA GPU with CUDA capability
- OpenSpiel for game definitions

Usage:
    source ~/open_spiel/venv/bin/activate
    python -c "from matrix_cfr import MatrixCFRSolver; ..."
"""

__version__ = '0.1.0'
__author__ = 'GTO Poker Training Project'

# Import main classes when package is imported
# (will be implemented in subsequent files)
# from .matrix_cfr_solver import MatrixCFRSolver
# from .game_to_matrix import GameTreeConverter
# from .gpu_memory import GPUMemoryManager
# from .validation import validate_against_cpu

__all__ = [
    # 'MatrixCFRSolver',
    # 'GameTreeConverter',
    # 'GPUMemoryManager',
    # 'validate_against_cpu',
]
