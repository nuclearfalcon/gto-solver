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
- `external_mccfr`: External Sampling MCCFR (SIMPLE or FULL averaging)
- `outcome_mccfr`: Outcome Sampling MCCFR
- `cpp_cfr`: C++ CFR
- `cpp_cfr_plus`: C++ CFR+
- `linear_external_mccfr`: Linear-Weighted External Sampling MCCFR with DCFR(α,β,γ) parameterization

**Algorithm Selection for 3+ Players**:

For 3+ player poker training, empirical validation (1M iterations on 3-player Kuhn poker) shows:
- **FULL averaging** (0.001109 Nash conv) significantly outperforms all DCFR variants
- **SIMPLE averaging** (0.024478 Nash conv) is 2107% worse
- DCFR(1.5, 0, 2) from research (0.004339 Nash conv) ranks 4th for 3+ players

**Recommendation**: Use External Sampling MCCFR with **FULL** averaging for 3+ player games.

See `DCFR_GUIDE.md` for complete algorithm selection guide and performance data.

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

### 3. DCFR/LCFR-ES Implementation
**Module**: `linear_external_mccfr.py`
**Primary Class**: `LinearExternalSamplingSolver`

**Purpose**: Linear-Weighted External Sampling MCCFR with full DCFR(α,β,γ) parameterization for 3+ player games.

**Key Features**:
- External Sampling ensures all branches are trained (required for 3+ players)
- Configurable discounting parameters (α, β, γ)
- Research-validated configurations: LCFR(1,1,1), SOTA DCFR(1.5,0,2), CFR+ Approx
- Checkpoint save/resume support
- Three critical bugs fixed (see `DCFR_BUGS_AND_FIXES.md`)

**DCFR Parameters**:
- **α (alpha)**: Positive regret discounting exponent
- **β (beta)**: Negative regret discounting exponent (β=0 means NO discount, not 0.5)
- **γ (gamma)**: Strategy averaging weighting exponent

**Usage**:
```python
from linear_external_mccfr import LinearExternalSamplingSolver

# True LCFR (linear weighting)
solver = LinearExternalSamplingSolver(game, gamma=1.0, alpha=1.0, beta=1.0)

# SOTA DCFR from research
solver = LinearExternalSamplingSolver(game, gamma=2.0, alpha=1.5, beta=0.0)

# Run iterations
for i in range(iterations):
    solver.iteration()

# Get policy
policy = solver.average_policy()
```

**Important Notes**:
- For 3+ players, **External Sampling MCCFR with FULL averaging** outperforms all DCFR variants
- DCFR research focused on 2-player games; results may not generalize to 3+ players
- Non-monotonic convergence is expected (track best Nash, not final Nash)
- See `DCFR_GUIDE.md` for algorithm selection and performance data

### 4. GPU MCCFR Solver (Phase 10+)
**Module**: `matrix_cfr/gpu_mccfr_solver.py`
**Primary Classes**: `GPUMCCFRSolver`, `GPURegretTable`

**Purpose**: GPU-accelerated Monte Carlo CFR for large-scale poker games using JAX and bucketing abstraction.

**Key Features**:
- GPU-resident computation (everything stays on GPU)
- Hierarchical bucketing (hand strength × pot size × round)
- Batched trajectory sampling (100-500 parallel games)
- Sparse or dense regret storage
- Memory: ~500 MB RAM, <10 MB VRAM (vs 10-20 GB for OpenSpiel)
- Speed: 100-1000× faster than sequential CPU for large games

**Usage**:
```python
from matrix_cfr.gpu_mccfr_solver import GPUMCCFRSolver
from gpu_mccfr_config import GPUMCCFRConfig
import jax.numpy as jnp

# Load configuration
config = GPUMCCFRConfig.from_json("configs/gpu/2p_10bb_holdem.json")

# Create solver
solver = GPUMCCFRSolver(
    num_players=config.num_players,
    num_buckets=config.num_buckets,
    num_hand_buckets=config.num_hand_buckets,
    num_pot_buckets=config.num_pot_buckets,
    num_actions=config.num_actions,
    batch_size=config.batch_size,
    seed=config.seed
)

# Run GPU-resident iteration
solver.run_iteration_gpu_resident(
    num_players=config.num_players,
    stacks=jnp.array(config.stacks),
    blinds=jnp.array(config.blinds),
    num_buckets=config.num_buckets,
    num_hand_buckets=config.num_hand_buckets,
    num_pot_buckets=config.num_pot_buckets
)

# Get average policy (bucket-level)
policy = solver.get_average_policy()
```

**When to use**:
- Large games (2-9 player Hold'em, 10BB+ stacks)
- RAM constrained (<2 GB available)
- Training speed matters (100-1000× vs CPU)
- Bucketing abstraction acceptable (~95% solution quality)

**When NOT to use**:
- Tiny games (Kuhn, Leduc) - use OpenSpiel
- Exact Nash equilibrium required - use OpenSpiel
- No GPU available - use OpenSpiel

**Related modules**:
- `matrix_cfr/holdem_jax_v2.py`: JAX Hold'em game engine
- `matrix_cfr/kuhn_jax_v2.py`: JAX Kuhn poker engine
- `matrix_cfr/bucketing.py`: Hierarchical bucketing functions
- `gpu_mccfr_config.py`: Configuration management
- `solve_poker_gpu.py`: CLI tool for GPU training

See `GPU_MCCFR_GUIDE.md` for detailed technical documentation.

### 5. JAX Game Engines
**Module**: `matrix_cfr/holdem_jax_v2.py`, `matrix_cfr/kuhn_jax_v2.py`
**Primary Functions**: `step_state()`, `is_terminal()`, `get_returns()`

**Purpose**: Pure JAX implementations of poker games enabling GPU compilation via JIT.

**Key Features**:
- Pure functional programming (no side effects)
- JAX control flow (`jax.lax.cond`, `jax.lax.while_loop`) instead of Python if/for
- GPU-compilable via `jax.jit`
- Vectorizable via `jax.vmap` (100+ parallel games)
- State as NamedTuple of JAX arrays (GPU-resident)

**Usage**:
```python
from matrix_cfr.holdem_jax_v2 import HoldemState, step_state, is_terminal, get_returns
import jax.numpy as jnp

# Create initial state
state = HoldemState(
    hole_cards=jnp.array([[0, 1], [2, 3]]),
    board=jnp.array([-1, -1, -1, -1, -1]),
    deck=jnp.ones(52, dtype=bool),
    bets=jnp.array([50.0, 100.0]),
    pot=150.0,
    stacks=jnp.array([950.0, 900.0]),
    round=0,
    acting_player=0,
    num_actions_this_round=0,
    folded=jnp.array([False, False]),
    all_in=jnp.array([False, False])
)

# Step state (GPU-compiled)
new_state = step_state(state, action=1)  # Call

# Check terminal
if is_terminal(new_state):
    payoffs = get_returns(new_state)
```

**Vectorization example** (100 parallel games):
```python
# Batch of initial states (100 games)
states_batch = jax.vmap(create_initial_state)(batch_keys)

# Batch of actions
actions = jnp.array([1, 0, 1, ...])  # 100 actions

# Vectorized step (all games in parallel on GPU!)
new_states = jax.vmap(step_state)(states_batch, actions)
```

### 6. Hierarchical Bucketing
**Module**: `matrix_cfr/bucketing.py`
**Primary Functions**: `state_to_bucket_index()`, `compute_cfvs_vectorized()`, `compute_regret_deltas_vectorized()`

**Purpose**: Reduce state space from ~10^14 infosets (full Hold'em) to ~10^4 buckets via abstraction.

**Bucketing dimensions**:
- Hand strength (200 buckets): Preflop equity, postflop hand type
- Pot size (10 buckets): Logarithmic bucketing of pot/chips ratio
- Round (4): Preflop, flop, turn, river
- Bet action history (abstracted)

**Usage**:
```python
from matrix_cfr.bucketing import state_to_bucket_index, compute_cfvs_vectorized

# Convert state to bucket
bucket_idx = state_to_bucket_index(
    state,
    num_buckets=10000,
    num_hand_buckets=200,
    num_pot_buckets=10,
    num_actions=4
)

# Compute CFVs for batch (GPU-vectorized)
cfvs = compute_cfvs_vectorized(
    payoffs,          # Terminal payoffs
    valid_masks,      # Valid state mask
    players,          # Player IDs
    updating_player   # Player to update
)
```

**Trade-off**:
- Memory reduction: 100,000,000× (10^14 → 10^4)
- Solution quality: ~95% of exact (empirically)
- Cannot query exact infoset strategies

### 7. GPU MCCFR Configuration
**Module**: `gpu_mccfr_config.py`
**Primary Class**: `GPUMCCFRConfig`

**Purpose**: Configuration management for GPU MCCFR training.

**Features**:
- JSON serialization/deserialization
- Validation in `__post_init__`
- 6 preset configurations
- Stack-in-BB calculations
- Game description generation

**Usage**:
```python
from gpu_mccfr_config import GPUMCCFRConfig

# Load from JSON
config = GPUMCCFRConfig.from_json("configs/gpu/2p_10bb_holdem.json")

# Create from preset
config = GPUMCCFRConfig.get_preset("2p_10bb_holdem")

# Create programmatically
config = GPUMCCFRConfig(
    num_players=2,
    stacks=[1000.0, 1000.0],
    blinds=[50.0, 100.0],
    batch_size=100,
    num_buckets=10000,
    num_hand_buckets=200,
    num_pot_buckets=10,
    num_actions=4,
    seed=42
)

# Save to JSON
config.to_json("configs/gpu/custom.json")

# Get game description
print(config.get_game_description())  # "2p 10bb Hold'em"
```

**Presets**: `2p_10bb_holdem`, `2p_20bb_holdem`, `3p_10bb_holdem`, `6p_10bb_holdem`, `9p_10bb_holdem`, `2p_5bb_holdem_fast`

### 8. Game Configuration (OpenSpiel)
**Module**: `game_config.py`
**Primary Class**: `PokerGameConfig`

**Purpose**: Centralized game configuration management for OpenSpiel track.

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

### 5. Metrics Tracking
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

### 6. Logging
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

### 7. Test Utilities
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
| `linear_external_mccfr.py` | DCFR/LCFR-ES algorithms | `LinearExternalSamplingSolver` |
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

- **v1.1.0** (2025-11-01): DCFR/LCFR-ES implementation and validation
  - Added `linear_external_mccfr.py` module with DCFR(α,β,γ) parameterization
  - Fixed three critical bugs in DCFR implementation (see `DCFR_BUGS_AND_FIXES.md`)
  - Validated algorithms on 3-player Kuhn poker (1M iterations)
  - Documented algorithm selection for 3+ players (FULL averaging recommended)
  - Created `DCFR_GUIDE.md` with performance data and selection guide
  - Added parallel validation framework for algorithm comparison

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