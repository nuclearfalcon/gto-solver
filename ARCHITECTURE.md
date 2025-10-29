# GTO Poker Training - Architecture Documentation

This document describes the architecture of the GTO Poker Training codebase and provides guidance on using the single sources of truth for each component.

## Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Single Sources of Truth](#single-sources-of-truth)
3. [Module Reference](#module-reference)
4. [Usage Guidelines](#usage-guidelines)
5. [Architecture Diagram](#architecture-diagram)
6. [Development Best Practices](#development-best-practices)

## Core Philosophy

This codebase follows the **Single Source of Truth (SSOT)** principle:
- Each functionality has exactly ONE authoritative implementation
- All scripts use common modules for shared functionality
- Updates to methods automatically propagate throughout the codebase
- No code duplication for core operations

## Single Sources of Truth

### 1. Exploitability Calculation
**Module**: `exploitability_metrics.py`
**Primary Class**: `SampledExploitabilityCalculator`

**Purpose**: Calculate exploitability using Monte Carlo sampling with adaptive confidence intervals.

**Key Features**:
- Memory-efficient streaming algorithm (Welford's method)
- Adaptive sampling with confidence intervals
- Automatic garbage collection
- Unbiased estimates

**Usage**:
```python
from exploitability_metrics import SampledExploitabilityCalculator

# Create calculator
calc = SampledExploitabilityCalculator(game, policy)

# Calculate with adaptive sampling
result = calc.calculate(
    confidence_level=0.99,     # 99% confidence
    max_ci_width=0.001,        # 0.1% CI width
    min_samples=1000,
    max_samples=100_000
)
```

### 2. Solver Implementation
**Module**: `poker_solver.py`
**Primary Class**: `UnifiedPokerSolver`

**Purpose**: Unified interface for all CFR solving algorithms.

**Supported Algorithms**:
- `vanilla_cfr`: Python Vanilla CFR
- `cfr_plus`: Python CFR+
- `dcfr`: Python Discounted CFR
- `lcfr`: Python Linear CFR
- `external_mccfr`: External Sampling MCCFR
- `outcome_mccfr`: Outcome Sampling MCCFR
- `cpp_cfr`: C++ CFR
- `cpp_cfr_plus`: C++ CFR+

**Usage**:
```python
from poker_solver import UnifiedPokerSolver

# Create solver
solver = UnifiedPokerSolver(
    game_config=config,
    algorithm='cfr_plus'
)

# Solve game
solver.solve(
    max_iterations=100_000,
    adaptive_schedule=schedule
)

# Get exploitability
exploit = solver.calculate_exploitability()  # Full exploitability
sampled_exploit = solver.calculate_sampled_exploitability()  # Sampled
```

### 3. Game Configuration
**Module**: `game_config.py`
**Primary Class**: `PokerGameConfig`

**Purpose**: Centralized game configuration management.

**Features**:
- JSON serialization/deserialization
- Validation in `__post_init__`
- Conversion to OpenSpiel format
- Preset configurations

**Usage**:
```python
from game_config import PokerGameConfig

# Load from JSON
config = PokerGameConfig.from_json("configs/2p_10bb.json")

# Create game
game = config.create_game()

# Convert to OpenSpiel format
openspiel_config = config.to_openspiel_config()
```

### 4. Metrics Tracking
**Module**: `solver_metrics.py`
**Primary Classes**: `MetricsTracker`, `AdaptiveSchedule`

**Purpose**: Track and store solving metrics.

**Tracked Metrics**:
- Exploitability over time
- Iterations per second
- Memory usage
- Convergence rate

**Usage**:
```python
from solver_metrics import MetricsTracker, AdaptiveSchedule

# Create tracker
tracker = MetricsTracker(algorithm_name='CFR+')

# Create adaptive schedule
schedule = AdaptiveSchedule()  # Default: 50k, 100k, 250k intervals

# Record checkpoints
tracker.record_checkpoint(
    iteration=1000,
    exploitability=0.5,
    memory_mb=250
)

# Save results
tracker.save_csv('results/metrics.csv')
tracker.save_json('results/summary.json')
```

### 5. Logging
**Module**: `solver_logger.py`
**Primary Classes**: `SolverLogger`, `ComparisonLogger`

**Purpose**: Standardized logging for solving progress.

**Features**:
- Console output formatting
- Progress tracking
- Comparison reports

**Usage**:
```python
from solver_logger import SolverLogger

logger = SolverLogger()
logger.log_iteration(iteration=1000, exploit=0.5)
logger.log_exploitability_check(exploit=0.5, time_elapsed=60)
```

### 6. Test Utilities
**Module**: `test_utils.py`

**Purpose**: Common utilities for test scripts.

**Utilities**:
- `get_memory_mb()`: Get current memory usage
- `measure_memory_delta()`: Measure memory increase
- `measure_time_and_memory()`: Measure both time and memory
- `assert_memory_stable()`: Assert memory stays within bounds
- `MemoryMonitor`: Context manager for memory monitoring

**Usage**:
```python
from test_utils import get_memory_mb, MemoryMonitor

# Get current memory
memory = get_memory_mb()

# Monitor memory in block
with MemoryMonitor() as monitor:
    # Your code here
    pass
print(f"Memory increased by {monitor.memory_increase:.2f} MB")
```

## Module Reference

| Module | Purpose | Primary Classes/Functions |
|--------|---------|---------------------------|
| `exploitability_metrics.py` | Exploitability calculation | `SampledExploitabilityCalculator` |
| `poker_solver.py` | Solving algorithms | `UnifiedPokerSolver` |
| `game_config.py` | Game configuration | `PokerGameConfig` |
| `solver_metrics.py` | Metrics tracking | `MetricsTracker`, `AdaptiveSchedule` |
| `solver_logger.py` | Logging | `SolverLogger`, `ComparisonLogger` |
| `betting_abstraction.py` | Betting abstractions | `BettingAbstraction` |
| `test_utils.py` | Test utilities | `get_memory_mb()`, `MemoryMonitor` |

## Usage Guidelines

### DO ✅

1. **Always use the designated modules**:
   ```python
   # CORRECT: Use UnifiedPokerSolver
   from poker_solver import UnifiedPokerSolver
   solver = UnifiedPokerSolver(config, algorithm='cfr_plus')
   ```

2. **Load configurations through PokerGameConfig**:
   ```python
   # CORRECT: Use PokerGameConfig
   from game_config import PokerGameConfig
   config = PokerGameConfig.from_json("config.json")
   ```

3. **Use test_utils for memory monitoring in tests**:
   ```python
   # CORRECT: Use centralized utilities
   from test_utils import get_memory_mb
   memory = get_memory_mb()
   ```

4. **Calculate exploitability through designated methods**:
   ```python
   # CORRECT: Use solver's methods
   full_exploit = solver.calculate_exploitability()
   sampled_exploit = solver.calculate_sampled_exploitability()
   ```

### DON'T ❌

1. **Don't access OpenSpiel APIs directly**:
   ```python
   # WRONG: Direct OpenSpiel call
   from open_spiel.python.algorithms import exploitability
   exploit = exploitability.nash_conv(game, policy)

   # CORRECT: Use solver method
   exploit = solver.calculate_exploitability()
   ```

2. **Don't duplicate utility functions**:
   ```python
   # WRONG: Define your own memory function
   def get_memory():
       return psutil.Process().memory_info().rss / 1024 / 1024

   # CORRECT: Use test_utils
   from test_utils import get_memory_mb
   ```

3. **Don't hardcode game parameters**:
   ```python
   # WRONG: Hardcoded parameters
   game = pyspiel.load_game('universal_poker', {
       'numPlayers': 2,
       'numRounds': 4,
       ...
   })

   # CORRECT: Use configuration
   config = PokerGameConfig.from_json("config.json")
   game = config.create_game()
   ```

4. **Don't create custom solver wrappers**:
   ```python
   # WRONG: Custom wrapper
   class MyPokerSolver:
       def __init__(self):
           self.solver = cfr.CFRPlusSolver(game)

   # CORRECT: Use UnifiedPokerSolver
   solver = UnifiedPokerSolver(config, algorithm='cfr_plus')
   ```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Scripts                              │
├─────────────────────────────────────────────────────────────────┤
│  solve_poker.py    solve_and_compare.py    query_policy.py      │
│         ↓                    ↓                    ↓              │
└─────────────┬───────────────┬───────────────────┬───────────────┘
              │               │                   │
              ▼               ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Modules (SSOT)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │PokerGameConfig│  │UnifiedPoker │  │SampledExploit- │        │
│  │              │  │   Solver     │  │abilityCalculator│        │
│  └──────────────┘  └──────────────┘  └────────────────┘        │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │MetricsTracker│  │SolverLogger  │  │ Betting        │        │
│  │              │  │              │  │ Abstraction    │        │
│  └──────────────┘  └──────────────┘  └────────────────┘        │
│                                                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OpenSpiel Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  pyspiel.load_game('universal_poker')                           │
│  CFR Algorithms (C++ and Python implementations)                 │
│  Policy representations                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Development Best Practices

### 1. Adding New Features

When adding new functionality:
1. Check if it belongs in an existing module
2. If creating a new module, ensure it's a single source of truth
3. Update this documentation
4. Add comprehensive tests using `test_utils.py`

### 2. Modifying Existing Modules

When updating existing modules:
1. Ensure backward compatibility
2. Update all dependent scripts if API changes
3. Run all test scripts to verify nothing breaks
4. Update documentation if behavior changes

### 3. Creating New Scripts

When creating new scripts:
1. Import from the designated SSOT modules
2. Never duplicate functionality
3. Follow the established patterns from existing scripts
4. Add proper documentation headers

### 4. Testing

All test scripts should:
1. Use `test_utils.py` for common operations
2. Test the SSOT modules, not direct OpenSpiel calls
3. Include memory monitoring when appropriate
4. Follow the naming convention: `test_*.py`

### 5. Configuration Files

Game configurations should:
1. Be stored in `configs/` directory
2. Use descriptive names (e.g., `2p_10bb_fcpa.json`)
3. Include comments in a separate README if complex
4. Be validated by `PokerGameConfig` on load

## Version History

- **v1.0.0** (2025-10-29): Initial architecture documentation
  - Established single sources of truth
  - Fixed exploitability calculation inconsistency
  - Created test_utils module
  - Removed unused metrics fields

## Maintenance

This architecture is maintained to ensure:
- **Consistency**: All code uses the same implementations
- **Maintainability**: Updates propagate automatically
- **Testability**: Clear boundaries for unit testing
- **Performance**: Optimized implementations in one place
- **Documentation**: Clear guidance for developers

For questions or improvements, please refer to the project maintainers.