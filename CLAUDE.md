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

**Query conditional best response for specific hero scenarios:**
```bash
# Analyze hero's optimal strategy with specific hole cards
python query_scenario.py --policy checkpoints/cfr_plus_iter_200.pkl \
    --config configs/2p_5bb_fcpa.json \
    --hero-position 0 \
    --hero-cards "As Kh" \
    --depth-limit 1 \
    --max-samples 500
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

**Always use these modules. Never access OpenSpiel APIs directly or duplicate functionality.**

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

### Conditional Best Response (Hero Scenario Analysis)

**Purpose:** Analyze hero's optimal strategy with specific hole cards against GTO opponents, without solving the entire game tree.

```python
from conditional_solver import ConditionalBestResponse
from scenario_config import ScenarioConfig

# Define scenario: hero with specific cards
scenario = ScenarioConfig.from_game_config(
    game_config=config,
    hero_position=0,              # Button = 0, BB = 1
    hero_cards_str="As Kh",       # Hero's specific cards
    depth_limit=1                 # 1 = preflop only, None = all streets
)

# Compute conditional best response
cbr = ConditionalBestResponse(game, policy, scenario)
result = cbr.compute(
    num_samples=500,              # Max opponent card deals to sample
    confidence_level=0.99,        # 99% CI
    max_ci_width=0.05,           # Stop when CI width < 5%
    verbose=True
)

# Results
print(f"Best action: {result['best_action']}")
print(f"Action EVs: {result['action_evs']}")
print(f"BR value: {result['br_value']} ± {result['ci_half_width']}")
```

**Key features:**
- Monte Carlo sampling over opponent hand ranges
- Samples only opponent cards, hero cards are fixed
- Computes best response for each opponent card deal
- Depth-limited solving (preflop only, or any N streets)
- Streaming statistics with confidence intervals
- Memory-safe (same approach as sampled exploitability)

**Use cases:**
- Analyzing specific hero scenarios (e.g., "What should I do with AK on BTN?")
- Range analysis and action frequencies
- Debugging/validating trained policies on specific hands
- Training data generation for neural networks

**Card notation:**
- Standard notation: "As Kh" (Ace of spades, King of hearts)
- Raw integers: "51 47" (for tiny decks with non-standard ranks/suits)
- Parser tries integer first, then falls back to standard notation

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
- `query_scenario.py` - Conditional best response analysis for specific hero scenarios
- `plot_results.py` - Visualize convergence curves

**Core Modules (SSOT):**
- `poker_solver.py` - Unified solver interface
- `exploitability_metrics.py` - Sampled exploitability calculation
- `game_config.py` - Game configuration management
- `scenario_config.py` - Hero scenario configuration for conditional solving
- `conditional_solver.py` - Conditional best response calculator
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
