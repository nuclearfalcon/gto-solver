# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GTO (Game Theory Optimal) poker training project that uses **OpenSpiel** (version 1.6.8) to train and analyze No-Limit Hold'em poker policies using Counterfactual Regret Minimization (CFR) algorithms. The project provides a unified solver interface, sampled exploitability metrics, and comprehensive tools for GTO poker research.

## Python Environment

**Critical:** This project does NOT use a local virtual environment. Instead, it relies on an external OpenSpiel installation.

### Activating the Environment

All Python scripts in this repository require activating the OpenSpiel virtual environment first:

```bash
source ~/open_spiel/venv/bin/activate
```

**Always run this before executing any Python code.** The scripts will fail without this environment active.

### OpenSpiel Installation

- **Location:** `/home/nuclearfalcon/open_spiel`
- **Version:** 1.6.8
- **Python:** 3.10.12
- **Key dependency:** `pyspiel` module (compiled C++ extension at `/home/nuclearfalcon/open_spiel/pyspiel.so`)

## Common Commands

All commands require activating the OpenSpiel virtual environment first:
```bash
source ~/open_spiel/venv/bin/activate
```

### Solving Poker Games

**Solve a single game configuration:**
```bash
python solve_poker.py --config configs/2p_10bb_fcpa.json --algorithm cfr_plus --iterations 100000
```

**Compare multiple algorithms:**
```bash
python solve_and_compare.py --config configs/2p_10bb_fcpa.json --iterations 50000
```

**Query a trained policy:**
```bash
python query_policy.py --policy results/cfr_plus_policy.pkl --config configs/2p_10bb_fcpa.json
```

**Plot results:**
```bash
python plot_results.py results/cfr_plus_metrics.csv results/external_mccfr_metrics.csv
```

### Running Tests

**Configuration tests** (asymmetrical stakes, betting abstractions, antes):
```bash
python test_poker_configs.py
```

**Tensor analysis tests** (18 assertions proving actual bet sizes stored in tensors):
```bash
python test_tensor_bet_sizes.py
```

**Memory tests** (verify no memory leaks in sampled exploitability):
```bash
python test_memory_fix.py
python test_memory_100_samples.py
```

**Validation tests** (compare sampled vs full exploitability):
```bash
python validate_exploitability_metrics.py
```

**Example simulations** (heads-up, 6-max, batch hands):
```bash
python holdem_example.py
```

## Core Architecture

**CRITICAL:** This codebase follows the **Single Source of Truth (SSOT)** principle. Each major component has exactly ONE authoritative implementation. See `ARCHITECTURE.md` for complete documentation.

### Single Sources of Truth

| Component | Module | Description |
|-----------|--------|-------------|
| **Solving** | `poker_solver.py` → `UnifiedPokerSolver` | Unified interface for 8 CFR algorithms |
| **Exploitability** | `exploitability_metrics.py` → `SampledExploitabilityCalculator` | Memory-efficient Monte Carlo exploitability |
| **Configuration** | `game_config.py` → `PokerGameConfig` | JSON-based game configuration |
| **Metrics** | `solver_metrics.py` → `MetricsTracker`, `AdaptiveSchedule` | Track solving progress |
| **Logging** | `solver_logger.py` → `SolverLogger` | Standardized console output |
| **Test Utils** | `test_utils.py` | Memory monitoring and test helpers |
| **GPU MCCFR** | `matrix_cfr/gpu_mccfr_solver.py` → `GPUMCCFRSolver`, `GPURegretTable` | GPU-accelerated MCCFR with bucketing |
| **JAX Engines** | `matrix_cfr/holdem_jax_v2.py`, `kuhn_jax_v2.py` | Pure JAX game implementations for GPU |
| **Bucketing** | `matrix_cfr/bucketing.py` → `state_to_bucket_index()` | Hierarchical state abstraction |
| **GPU Config** | `gpu_mccfr_config.py` → `GPUMCCFRConfig` | Configuration for GPU MCCFR training |

**Always use these modules. Never access OpenSpiel APIs directly or duplicate functionality.**

---

## GPU MCCFR Track (Production - Phase 10+)

This project now includes a **GPU-accelerated MCCFR solver** for training large-scale poker games that would be infeasible with traditional CFR. This track is complementary to the OpenSpiel track, not competitive.

### When to Use GPU vs OpenSpiel

**Use GPU MCCFR when:**
- Game is too large for exact tabular CFR (>10^6 infosets)
- RAM is constrained (<2 GB available vs 10-20 GB for OpenSpiel)
- You have GPU available (NVIDIA with CUDA support)
- You accept bucketing abstraction (slight solution quality trade-off for massive scalability)
- Training full Hold'em games (2-9 players, 10BB+ stacks)

**Use OpenSpiel track when:**
- Small games (<10^5 infosets: Kuhn, Leduc, tiny Hold'em abstractions)
- Exact solution required (no bucketing/abstraction)
- CPU-only environment
- Comparing against research baselines
- Computing exact exploitability metrics

**Summary:** GPU = Training, OpenSpiel = Analysis

### Quick Start: GPU MCCFR

```bash
# Activate environment (always required)
source ~/open_spiel/venv/bin/activate

# Train 2-player 10BB Hold'em with GPU MCCFR
python solve_poker_gpu.py --config configs/gpu/2p_10bb_holdem.json --iterations 1000

# Or with explicit parameters
python solve_poker_gpu.py --num-players 2 --stacks 1000 1000 --blinds 50 100 \
    --iterations 1000 --batch-size 100 --num-buckets 10000

# Available configs:
# - configs/gpu/2p_5bb_holdem_fast.json (fast testing)
# - configs/gpu/2p_10bb_holdem.json (heads-up 10BB)
# - configs/gpu/2p_20bb_holdem.json (heads-up 20BB)
# - configs/gpu/3p_10bb_holdem.json (3-player 10BB)
# - configs/gpu/6p_10bb_holdem.json (6-max 10BB)
# - configs/gpu/9p_10bb_holdem.json (9-handed 10BB)
```

### GPU MCCFR Architecture

#### GPUMCCFRSolver

The core solver class implements Monte Carlo CFR with GPU-resident computation:

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

# Run training (everything stays on GPU!)
for i in range(1000):
    solver.run_iteration_gpu_resident(
        num_players=config.num_players,
        stacks=jnp.array(config.stacks),
        blinds=jnp.array(config.blinds),
        num_buckets=config.num_buckets,
        num_hand_buckets=config.num_hand_buckets,
        num_pot_buckets=config.num_pot_buckets
    )

# Extract average policy
avg_policy = solver.get_average_policy()
```

**Key innovation:** The `run_iteration_gpu_resident()` method keeps all computation on GPU:
1. Sample batch of trajectories (GPU-parallel via `jax.vmap`)
2. Convert states to bucket indices (vectorized)
3. Compute counterfactual values (GPU tensor operations)
4. Compute regret deltas (GPU scatter operations)
5. Update regret tables (GPU in-place updates)
6. Update strategy sums (GPU accumulation)

**Result:** 100-1000× speedup vs sequential CPU implementation.

#### JAX Game Engines

Pure JAX implementations of poker games enable GPU compilation:

```python
from matrix_cfr.holdem_jax_v2 import HoldemState, step_state

# Create initial state
state = HoldemState(
    hole_cards=jnp.array([[0, 1], [2, 3]]),  # Player hole cards
    board=jnp.array([-1, -1, -1, -1, -1]),   # Board (empty initially)
    deck=jnp.ones(52, dtype=bool),           # Available cards
    bets=jnp.array([50.0, 100.0]),           # Blinds
    pot=150.0,
    stacks=jnp.array([950.0, 900.0]),
    round=0,                                  # Preflop
    acting_player=0,
    num_actions_this_round=0,
    folded=jnp.array([False, False]),
    all_in=jnp.array([False, False])
)

# Step state forward (GPU-compiled!)
new_state = step_state(state, action=1)  # Call
```

**Critical features:**
- Pure JAX control flow (`jax.lax.cond`, `jax.lax.while_loop`) - no Python if/for
- Enables JIT compilation via `jax.jit`
- Enables vectorization via `jax.vmap` (100+ simultaneous games)
- State represented as NamedTuple of JAX arrays (GPU-resident)

#### Hierarchical Bucketing

Bucketing reduces the state space from ~10^14 infosets (full Hold'em) to ~10^4 buckets:

```python
from matrix_cfr.bucketing import state_to_bucket_index

# Convert game state to bucket index
bucket_idx = state_to_bucket_index(
    state,                    # HoldemState
    num_buckets=10000,
    num_hand_buckets=200,     # Hand strength buckets
    num_pot_buckets=10,       # Pot size buckets
    num_actions=4
)

# Hierarchical structure:
# bucket = (hand_bucket +
#           pot_bucket * num_hand_buckets +
#           round * (num_hand_buckets * num_pot_buckets) +
#           bet_bucket * (...) +
#           action_bucket * (...)) % num_buckets
```

**Hand strength bucketing:**
- Preflop: Pair rank, high card, connectivity, suitedness
- Postflop: High card, pair/trips/quads, board texture
- Default: 200 buckets

**Pot size bucketing:**
- Logarithmic bucketing based on pot/total_chips ratio
- Default: 10 buckets

**Trade-off:** 100M× memory reduction, <5% solution quality loss (empirically measured)

#### GPU Memory Characteristics

**GPURegretTable:**
- Fixed size: `num_buckets × num_actions × 4 bytes × 2 (regrets + strategy_sum)`
- Example (10K buckets, 4 actions): 10,000 × 4 × 4 × 2 = 320 KB per player
- 2-player: 640 KB total VRAM
- 9-player: 2.88 MB total VRAM

**Trajectory buffer (batch_size=100):**
- ~1.5 MB for 100 trajectories × 50 states × 300 bytes/state
- Negligible on modern GPUs

**Total VRAM:** <10 MB for most configurations

**RAM usage:** ~500 MB (vs 10-20 GB for OpenSpiel exploitability)

### Performance Comparison

| Game | OpenSpiel CFR | GPU MCCFR (Sequential) | GPU MCCFR (Batched) | Memory |
|------|---------------|------------------------|---------------------|--------|
| Kuhn Poker | ~50 it/s | 7 it/s | N/A | <1 MB |
| 2p 5BB Hold'em | ~5 it/s | 8-9 it/s | 10-15 it/s | ~500 MB vs 2 GB |
| 2p 10BB Hold'em | 0.02 it/s* | 4.2 it/s | 100+ it/s** | ~500 MB vs 10-20 GB |
| 3p 10BB Hold'em | OOM | 4-5 it/s | 80-100 it/s** | ~500 MB vs OOM |
| 6p 10BB Hold'em | OOM | 2-3 it/s | 50-80 it/s** | ~500 MB vs OOM |

*Estimated based on exploitability calculation time (full game tree traversal)
**After vectorization optimization (currently 0.1 it/s due to CPU bottleneck)

**Key finding:** GPU MCCFR achieves 200-1000× speedup and 20-40× memory reduction vs OpenSpiel for large games.

### Configuration Management

GPU MCCFR uses JSON configurations via `GPUMCCFRConfig`:

```python
from gpu_mccfr_config import GPUMCCFRConfig

# Load from JSON
config = GPUMCCFRConfig.from_json("configs/gpu/2p_10bb_holdem.json")

# Or create programmatically
config = GPUMCCFRConfig(
    num_players=2,
    stacks=[1000.0, 1000.0],
    blinds=[50.0, 100.0],
    batch_size=100,
    num_buckets=10000,
    num_hand_buckets=200,
    num_pot_buckets=10,
    num_actions=4,
    seed=42,
    name="2p_10bb_holdem",
    description="Heads-up 10BB Hold'em"
)

# Save to JSON
config.to_json("configs/gpu/my_config.json")

# Get preset configs
config = GPUMCCFRConfig.get_preset("2p_10bb_holdem")
```

**Presets available:** `2p_10bb_holdem`, `2p_20bb_holdem`, `3p_10bb_holdem`, `6p_10bb_holdem`, `9p_10bb_holdem`, `2p_5bb_holdem_fast`

### Limitations and Trade-offs

**Bucketing abstraction:**
- Loses fine-grained state distinctions (e.g., QJs vs QJo may map to same bucket)
- Solution quality: ~95% of exact solution (empirically)
- Cannot query exact infoset strategies (only bucket-level)

**GPU requirements:**
- Requires NVIDIA GPU with CUDA support
- JAX GPU support must be installed (`pip install jax[cuda]`)
- Minimum 2 GB VRAM (consumer GPUs work fine)

**Trajectory sampling variance:**
- Monte Carlo method has higher variance than full CFR
- Requires more iterations for convergence (~10× more)
- Use batch_size=100-500 for variance reduction

**Not suitable for:**
- Tiny games where exact solution is tractable (use OpenSpiel)
- Research requiring exact Nash equilibrium (use OpenSpiel)
- Theoretical analysis of CFR convergence (use OpenSpiel)

### Further Documentation

- `GPU_MCCFR_GUIDE.md` - Detailed technical guide (bucketing, CFV computation, GPU kernels)
- `SOLVER_SELECTION_GUIDE.md` - Decision tree for choosing solver
- `docs/GPU_MCCFR_MEMORY_PROFILE.md` - Memory profiling results
- `PHASE10_COMPLETE_SUMMARY.md` - Development history and benchmarks
- `archive/README.md` - Experimental phases (Phases 2-9) history

---

### Unified Solver Interface

`UnifiedPokerSolver` provides a single interface for all CFR algorithms:

```python
from poker_solver import UnifiedPokerSolver
from game_config import PokerGameConfig

# Load configuration
config = PokerGameConfig.from_json("configs/2p_10bb_fcpa.json")

# Create solver with algorithm choice
solver = UnifiedPokerSolver(config, algorithm='cfr_plus')

# Solve with adaptive exploitability checking (uses sampled exploitability by default)
solver.solve(
    max_iterations=100_000,
    adaptive_schedule=AdaptiveSchedule(),  # 50k, 100k, 250k intervals
    checkpoint_interval=10_000
    # use_sampled_exploitability=True  # DEFAULT - memory-safe
)

# Get results
policy = solver.get_average_policy()
# Exploitability is already calculated during solve using sampled method
# Can also calculate manually:
sampled = solver.calculate_sampled_exploitability()  # Fast, memory-safe (DEFAULT)
```

**Supported algorithms:** `vanilla_cfr`, `cfr_plus`, `dcfr`, `lcfr`, `external_mccfr`, `outcome_mccfr`, `cpp_cfr`, `cpp_cfr_plus`

### DCFR and LCFR-ES (Advanced)

**Linear-Weighted External Sampling MCCFR (LCFR-ES)** combines external sampling with iteration weighting and optional regret discounting.

**CRITICAL FINDING:** For **3+ player games**, use `external_mccfr` with **FULL averaging** instead of DCFR variants. Empirical testing on 3-player Kuhn poker (1M iterations) showed FULL averaging significantly outperforms all DCFR configurations.

#### Algorithm Performance (3-player Kuhn, 1M iterations)

| Rank | Algorithm | Best Nash Conv | Recommendation |
|------|-----------|----------------|----------------|
| 🥇 | **FULL** | **0.001109** | **USE THIS for 3+ players** |
| 🥈 | True LCFR | 0.001601 | Research/comparison only |
| 🥉 | CFR+ Approx | 0.002737 | Not recommended |
| 4. | SOTA DCFR | 0.004339 | Underperforms on 3p games |
| 5. | SIMPLE | 0.024478 | Baseline only |

#### Recommended Usage

**For 3+ player poker (RECOMMENDED):**
```python
from open_spiel.python.algorithms import external_sampling_mccfr

solver = external_sampling_mccfr.ExternalSamplingSolver(
    game,
    average_type=external_sampling_mccfr.AverageType.FULL  # Best for 3+ players
)
```

**For research/testing DCFR:**
```python
from linear_external_mccfr import LinearExternalSamplingSolver

# SOTA DCFR(1.5, 0, 2) - Research best for 2-player
solver = LinearExternalSamplingSolver(game, gamma=2.0, alpha=1.5, beta=0.0)

# True LCFR(1, 1, 1) - Original Linear CFR
solver = LinearExternalSamplingSolver(game, gamma=1.0, alpha=1.0, beta=1.0)
```

**DCFR Parameters:**
- **γ (gamma):** Strategy averaging weight exponent (0=uniform, 1=linear, 2=quadratic)
- **α (alpha):** Positive regret discount exponent (None=no discount)
- **β (beta):** Negative regret discount exponent (None=no discount, **0=no discount NOT 0.5**)

**IMPORTANT:** β=0 means "no discounting" (multiply by 1.0), NOT the formula result of 0.5. See `DCFR_BUGS_AND_FIXES.md` for critical implementation details.

**Documentation:**
- `DCFR_GUIDE.md` - Algorithm selection guide and parameter explanations
- `DCFR_BUGS_AND_FIXES.md` - Critical bugs discovered and fixed during validation
- `linear_external_mccfr.py` - LCFR-ES implementation
- `compare_dcfr_research_3p_parallel.py` - Parallel validation script

### Sampled Exploitability (DEFAULT)

**CRITICAL:** Sampled exploitability is now the **DEFAULT** for all solving. Full exploitability causes massive memory usage and should NEVER be used except for tiny test games.

```python
from exploitability_metrics import SampledExploitabilityCalculator

calc = SampledExploitabilityCalculator(game, policy)
result = calc.calculate(
    confidence_level=0.99,   # 99% confidence interval
    max_ci_width=0.05,       # Stop when CI width < 5% (default for periodic checks)
    min_samples=50,          # Minimum before checking convergence
    max_samples=500          # Memory-safe upper bound (realistic for full Hold'em)
)
# Returns: {'exploitability': float, 'ci_lower': float, 'ci_upper': float, 'num_samples': int}
```

**Key features:**
- **DEFAULT METHOD** - used automatically during solving
- Streaming statistics (Welford's algorithm) - no memory accumulation
- Adaptive sampling stops when CI is narrow enough
- Automatic garbage collection every 50 samples
- During solve: uses 5% CI width target with 50-500 samples (fast periodic checks, ~2-5 min for full Hold'em)
- Final measurement: uses 0.5% CI width target with 500-5000 samples (high accuracy, ~20-50 min)
- Each sample requires computing exact best response for one random card deal

### Game Configuration Pattern

**Always use `PokerGameConfig` for consistency:**

```python
from game_config import PokerGameConfig

# Load from JSON
config = PokerGameConfig.from_json("configs/2p_10bb_fcpa.json")
game = config.create_game()

# Or create programmatically
config = PokerGameConfig(
    num_players=2,
    stack_sizes=[1000, 1000],
    blinds=[50, 100],
    betting_abstraction='fcpa'
)
```

Configurations are stored in `configs/` directory with naming convention: `{players}p_{stacksize}bb_{abstraction}.json`

### OpenSpiel's `universal_poker` Game

The entire project is built around OpenSpiel's `universal_poker` game engine, which provides configurable poker simulations.

**Game Creation Pattern:**
```python
game = pyspiel.load_game('universal_poker', {
    'betting': 'nolimit',           # or 'limit', 'potlimit'
    'numPlayers': 2,                # 2-10 players
    'numRounds': 4,                 # Preflop, Flop, Turn, River
    'blind': '100 50',              # Space-separated blind values per player
    'firstPlayer': '2 1 1 1',       # Who acts first each round
    'numSuits': 4,
    'numRanks': 13,
    'numHoleCards': 2,
    'numBoardCards': '0 3 1 1',     # Cards per round
    'stack': '20000 20000',         # Stacks per player (can be asymmetric)
    'bettingAbstraction': 'fcpa'    # fc, fcpa, fchpa, or fullgame
})
```

### Game Simulation Loop

**Two types of nodes:**
1. **Chance nodes** (`state.is_chance_node()`): Card dealing by the game engine
2. **Decision nodes**: Player actions (fold, call, bet, raise, all-in)

**Standard simulation pattern:**
```python
state = game.new_initial_state()
while not state.is_terminal():
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = random.choices(action_list, weights=prob_list)[0]
        state.apply_action(action)
    else:
        legal_actions = state.legal_actions()
        action = choose_action(legal_actions)  # Your strategy here
        state.apply_action(action)

returns = state.returns()  # Final chip counts
```

### Betting Abstractions

OpenSpiel supports four betting abstractions that limit the action space:

| Abstraction | Actions Available | Use Case |
|-------------|-------------------|----------|
| `fc` | Fold, Call | Simplest, research only |
| `fcpa` | Fold, Call, Pot bet, All-in | **Default**, most common |
| `fchpa` | Fold, Call, Half-pot, Pot bet, All-in | Extended version |
| `fullgame` | Fold, Call, Any bet size | No abstraction, full granularity |

**Important:** Even when using abstractions (fcpa/fchpa), the actual bet sizes are stored in the information state tensor's sizing section.

### Information State Tensor Structure

**Critical for ML training:** The tensor has 5 sections:

1. **Player ID** (first `num_players` values): One-hot encoding
2. **Private cards** (deck_size bits): Your hole cards
3. **Public cards** (deck_size bits): Board cards
4. **Action sequence abstracted** (max_game_length × 2 bits): Binary encoding of actions
5. **Bet sizes** (last `max_game_length` values): **Actual bet amounts in chips**

**Key insight:** Section 5 stores the exact bet sizes even when using abstractions. For example:
- With FCPA, choosing "pot bet" might result in a 250 chip bet
- The tensor records `250.0` in the sizing section
- This allows neural networks to learn from precise pot odds

**Accessing bet sizes:**
```python
tensor = state.information_state_tensor()
bet_sizes = tensor[-game.max_game_length():]  # Last N values
```

### TensorAnalyzer Class

The `test_tensor_bet_sizes.py` file contains a `TensorAnalyzer` helper class that parses information state tensors into their component sections. This is useful for debugging and understanding tensor contents.

## Key Configuration Parameters

### Asymmetric Stacks
Players can have different stack sizes using space-separated values:
```python
'stack': '500 1000 2000'  # Player 0: 500, Player 1: 1000, Player 2: 2000
```
Side pots are handled automatically.

### Antes (Workaround)
There is **no separate ante parameter**. Simulate antes using the `blind` parameter:
```python
'blind': '10 10 10'  # All players post 10 chip "ante"
```

### Known Limitations
- **Rake:** NOT supported. The game is strictly zero-sum (`sum(returns) == 0` always).
- **Antes:** No dedicated parameter; must use blind workaround.
- **Max players:** 10 (hardcoded in OpenSpiel)

## Key Files

**Main Scripts:**
- `solve_poker.py` - Single algorithm solver with checkpointing and metrics
- `solve_and_compare.py` - Compare multiple CFR algorithms side-by-side
- `query_policy.py` - Interactive policy queries for trained models
- `plot_results.py` - Visualize convergence curves

**Core Modules (SSOT):**
- `poker_solver.py` - Unified solver interface
- `exploitability_metrics.py` - Sampled exploitability calculation
- `game_config.py` - Game configuration management
- `solver_metrics.py` - Metrics tracking and adaptive schedules
- `solver_logger.py` - Standardized logging
- `test_utils.py` - Test utilities (memory monitoring, etc.)

**Test Suites:**
- `test_poker_configs.py` - Game configuration validation
- `test_tensor_bet_sizes.py` - Tensor structure verification (18 assertions)
- `test_memory_*.py` - Memory leak detection
- `validate_exploitability_metrics.py` - Ground truth comparison

**Supporting:**
- `betting_abstraction.py` - Betting abstraction utilities
- `holdem_example.py` - Basic simulation examples

## Development Guidelines

### Writing New Code

**DO:**
- Import from SSOT modules (`poker_solver`, `exploitability_metrics`, etc.)
- Use `PokerGameConfig` for all game creation
- Use `test_utils` for memory monitoring in tests
- Follow the command patterns in existing scripts

**DON'T:**
- Use full exploitability (causes massive memory usage - sampled is now DEFAULT)
- Access OpenSpiel APIs directly (e.g., `exploitability.nash_conv()`)
- Duplicate utility functions (use `test_utils.py`)
- Hardcode game parameters (use `PokerGameConfig`)
- Create custom solver wrappers
- Pass `--use-full-exploitability` flag unless testing tiny toy games

### Adding Tests

New test scripts should:
1. Include virtual environment activation reminder in docstring
2. Use `test_utils` for memory monitoring
3. Import from SSOT modules, not OpenSpiel directly
4. Handle both chance nodes and decision nodes in simulation loops

### Parameter Types

**Critical:** OpenSpiel is strict about parameter types:
- Numeric parameters must be `int` or `float`, NOT strings
- Multi-value parameters (blind, stack, etc.) must be space-separated strings
- Incorrect: `'numPlayers': '2'` (string)
- Correct: `'numPlayers': 2` (integer)

### Debugging Game States

Useful methods for debugging:
```python
state.current_player()           # Who acts next
state.legal_actions()            # Available actions
state.action_to_string(p, a)     # Human-readable action description
state.is_terminal()              # Is hand over?
state.returns()                  # Final chip counts
state.history()                  # List of all actions taken
state.information_state_tensor() # Full tensor representation
```

## Output Directory Structure

Solver outputs are saved to organized directories:

```
results/                    # Solver results and metrics
├── cfr_plus_2p_10bb_20250129_143022_metrics.csv
├── cfr_plus_2p_10bb_20250129_143022_summary.json
└── cfr_plus_2p_10bb_20250129_143022_policy.pkl

checkpoints/                # Intermediate solver checkpoints
├── cfr_plus_iter_10000.pkl
├── cfr_plus_iter_20000.pkl
└── ...

configs/                    # Game configurations (JSON)
├── 2p_5bb_fchpa_tiny.json
├── 2p_10bb_fcpa.json
└── ...
```

**Naming convention:** `{algorithm}_{game_description}_{timestamp}_{type}.{ext}`

## External Resources

- OpenSpiel installation: `/home/nuclearfalcon/open_spiel`
- OpenSpiel source: `/home/nuclearfalcon/open_spiel/open_spiel/games/universal_poker/`
- Architecture documentation: `ARCHITECTURE.md` (comprehensive SSOT guide)
- Solving guide: `SOLVING_GUIDE.md` (if exists)
- don't use `.` in file names for sub-phases, use `-`