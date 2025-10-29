# GTO Poker Solving Guide

Complete guide to solving poker games for Game Theory Optimal (GTO) strategies using this framework.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Game Configuration](#game-configuration)
3. [Running Solves](#running-solves)
4. [Algorithm Comparison](#algorithm-comparison)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Prerequisites

```bash
# Activate OpenSpiel environment
source ~/open_spiel/venv/bin/activate

# Verify installation
python -c "import pyspiel; print('OpenSpiel ready')"
```

### Run Your First Solve

```bash
# Solve 2-player 5BB game with CFR+ (validation config)
python solve_poker.py \
  --config configs/2p_5bb_fchpa_1.5x.json \
  --algorithm cfr_plus \
  --iterations 10000 \
  --save-policy

# Results will be saved to results/ directory
```

### Compare Algorithms

```bash
# Test all algorithms on the same game
python solve_and_compare.py \
  --config configs/2p_5bb_fchpa_1.5x.json \
  --iterations 10000

# Generates comparison report in results/
```

---

## Game Configuration

### Configuration Files

Game configurations are JSON files in `configs/` directory:

```json
{
  "num_players": 2,
  "stack_sizes": [1000, 1000],
  "blinds": [100, 50],
  "betting_abstraction": "fchpa_1.5x",
  "description": "2-player 10BB heads-up NLHE"
}
```

### Provided Configurations

| Config | Players | Stack (BB) | Description |
|--------|---------|------------|-------------|
| `2p_5bb_fchpa_1.5x.json` | 2 | 5BB | Validation config (small, fast) |
| `2p_10bb_fchpa_1.5x.json` | 2 | 10BB | Standard short-stack |
| `2p_20bb_fchpa_1.5x.json` | 2 | 20BB | Medium stack |
| `3p_10bb_fchpa_1.5x.json` | 3 | 10BB | 3-player game |
| `6p_10bb_fchpa_1.5x.json` | 6 | 10BB | 6-max game |

### Creating Custom Configurations

```python
from game_config import PokerGameConfig

# Create config programmatically
config = PokerGameConfig(
    num_players=2,
    stack_sizes=[2000, 2000],
    blinds=[100, 50],
    betting_abstraction='fchpa_1.5x',
    description="Custom 20BB config"
)

# Save to file
config.to_json('configs/my_config.json')
```

### Betting Abstractions

- **`fchpa_1.5x`** (Recommended): Fold, Call, Half-pot, Pot, 1.5×pot, All-in
- **`fchpa`**: Fold, Call, Half-pot, Pot, All-in
- **`fcpa`**: Fold, Call, Pot, All-in
- **`fullgame`**: All bet sizes (very large game tree)

---

## Running Solves

### Single Algorithm Solve

```bash
python solve_poker.py \
  --config <config_file> \
  --algorithm <algorithm_name> \
  --iterations <num_iterations> \
  [options]
```

**Required Arguments:**
- `--config`: Path to game configuration JSON
- `--algorithm`: Algorithm to use (see [Available Algorithms](#available-algorithms))
- `--iterations`: Number of iterations to run

**Optional Arguments:**
- `--save-policy`: Save final policy to file
- `--checkpoint-interval N`: Save checkpoint every N iterations
- `--check-exploitability`: `adaptive` (default) or `fixed`
- `--output-dir`: Directory for results (default: `results/`)

### Available Algorithms

| Algorithm | Type | Speed | Memory | Best For |
|-----------|------|-------|--------|----------|
| `vanilla_cfr` | Python | Slow | Medium | Debugging |
| `cfr_plus` | Python | Medium | Medium | **Recommended baseline** |
| `dcfr` | Python | Medium | Medium | Research |
| `lcfr` | Python | Medium | Medium | Research |
| `external_mccfr` | Python | Fast | Low | Large games |
| `outcome_mccfr` | Python | Very Fast | Very Low | Very large games |
| `cpp_cfr` | C++ | Fast | Low | **Production** |
| `cpp_cfr_plus` | C++ | Very Fast | Low | **Best overall** |

**Recommendation**: Start with `cfr_plus` for testing, use `cpp_cfr_plus` for production solves.

### Example Solves

#### Quick Validation (2-5 minutes)
```bash
python solve_poker.py \
  --config configs/2p_5bb_fchpa_1.5x.json \
  --algorithm cfr_plus \
  --iterations 10000 \
  --save-policy
```

#### Medium Solve (30-60 minutes)
```bash
python solve_poker.py \
  --config configs/2p_10bb_fchpa_1.5x.json \
  --algorithm cpp_cfr_plus \
  --iterations 100000 \
  --save-policy \
  --checkpoint-interval 25000
```

#### Long Solve (hours to days)
```bash
python solve_poker.py \
  --config configs/2p_20bb_fchpa_1.5x.json \
  --algorithm cpp_cfr_plus \
  --iterations 1000000 \
  --save-policy \
  --checkpoint-interval 100000
```

### Monitoring Progress

During solving, you'll see output like:

```
[CFR+] [50000/1000000] [5.0%] | Exploit: 0.023456 | Time: 1234.5s | 40.5 it/s | Mem: 2.3 GB | Conv: -12.34% | Next: 50000 iters
```

**Fields explained:**
- **[50000/1000000]**: Current/total iterations
- **[5.0%]**: Progress percentage
- **Exploit**: Current exploitability (NashConv) - lower is better
- **Time**: Elapsed time
- **40.5 it/s**: Iterations per second
- **Mem**: Current memory usage
- **Conv**: Convergence rate (% improvement since last check)
- **Next**: Iterations until next exploitability check

### Exploitability Schedule

**Adaptive (Default)**:
- Iterations 0-500k: check every 50k
- Iterations 500k-2M: check every 100k
- Iterations 2M+: check every 250k

**Fixed**:
```bash
python solve_poker.py \
  --config configs/2p_10bb.json \
  --algorithm cfr_plus \
  --iterations 100000 \
  --check-exploitability fixed \
  --check-interval 10000
```

---

## Algorithm Comparison

### Run Comparison

```bash
python solve_and_compare.py \
  --config configs/2p_5bb_fchpa_1.5x.json \
  --iterations 10000
```

This will:
1. Run each algorithm sequentially on the same game
2. Record metrics for each (exploitability, speed, memory)
3. Generate comparison report

### Custom Algorithm List

```bash
python solve_and_compare.py \
  --config configs/2p_10bb.json \
  --iterations 50000 \
  --algorithms cfr_plus cpp_cfr_plus external_mccfr
```

### Comparison Output

Results are saved to `results/comparison_<game>_<timestamp>.md`:

```markdown
## Results Summary

| Algorithm | Final Exploit | Time (s) | Avg Speed (it/s) | Peak Memory (MB) |
|-----------|---------------|----------|------------------|------------------|
| Python CFR+ | 0.012345 | 1234.5 | 40.5 | 2300.0 |
| C++ CFR+ | 0.011234 | 156.2 | 320.1 | 450.0 |
...
```

---

## Understanding Results

### Output Files

After solving, you'll find in `results/`:

1. **`<algorithm>_<game>_<timestamp>_metrics.csv`**
   - Iteration-by-iteration metrics
   - Columns: iteration, exploitability, time_elapsed, memory_mb, iters_per_sec, convergence_rate

2. **`<algorithm>_<game>_<timestamp>_summary.json`**
   - Summary statistics
   - Final exploitability, total time, average speed, peak memory

3. **`<algorithm>_<game>_<timestamp>_policy.pkl`** (if `--save-policy`)
   - Pickled policy object
   - Can be loaded and queried (see [Policy Extraction](#policy-extraction))

### Interpreting Exploitability

**Exploitability** (NashConv) measures how much a best-response opponent can exploit the strategy:

- **< 0.01**: Excellent (< 1% exploitability)
- **< 0.001**: Very strong (< 0.1%)
- **< 0.0001**: Near-optimal (< 0.01%)

**Example**: Exploitability of 0.005 in a 10BB game means the opponent can exploit for ~0.05BB (0.5% of stack).

### Convergence

Monitor the **Conv** (convergence rate) field:
- **Negative values** (e.g., -12.34%): Exploitability decreasing (good!)
- **Positive values**: Exploitability increasing (bad - may indicate issues)
- **Values near 0%**: Convergence slowing (normal late in solve)

---

## Advanced Usage

### Policy Extraction

```bash
python query_policy.py \
  --policy results/cfr_plus_2p_10bb_policy.pkl \
  --info-states 20
```

### Visualization

```bash
# Plot single solve
python plot_results.py \
  --metrics results/cfr_plus_*_metrics.csv \
  --output plots/cfr_plus_convergence.png

# Compare multiple algorithms
python plot_results.py \
  --metrics results/cfr_plus_*_metrics.csv results/cpp_cfr_*_metrics.csv \
  --output plots/comparison.png \
  --log-scale

# Include memory usage
python plot_results.py \
  --metrics results/*_metrics.csv \
  --output plots/full_analysis.png \
  --show-memory
```

### Checkpointing

Save intermediate checkpoints:

```bash
python solve_poker.py \
  --config configs/2p_20bb.json \
  --algorithm cpp_cfr_plus \
  --iterations 1000000 \
  --checkpoint-interval 100000 \
  --checkpoint-dir checkpoints/
```

Checkpoints are saved to `checkpoints/<algorithm>_iter_<N>.pkl`.

---

## Tips and Best Practices

### 1. Start Small
- Always validate your pipeline on `2p_5bb` config first
- Confirm algorithms work before running long solves

### 2. Use C++ Implementations
- 5-10x faster than Python versions
- Lower memory usage
- Prefer `cpp_cfr_plus` for production

### 3. Monitor Memory
- If memory usage grows too large, switch to MCCFR algorithms
- `external_mccfr` uses ~10-100x less memory than vanilla CFR

### 4. Exploitability Goals
- For research: aim for < 0.001 (0.1%)
- For practical use: < 0.01 (1%) is often sufficient
- Diminishing returns after certain point

### 5. Iteration Guidelines
- **5BB game**: 10,000 - 50,000 iterations
- **10BB game**: 50,000 - 200,000 iterations
- **20BB game**: 200,000 - 1,000,000+ iterations

### 6. Algorithm Selection
- **Python CFR+**: Testing and debugging
- **C++ CFR+**: Production solves (best balance)
- **External MCCFR**: Very large games or low memory
- **Vanilla CFR**: Only for comparison/research

---

## Troubleshooting

### Out of Memory

```
MemoryError: Unable to allocate ...
```

**Solutions**:
1. Use MCCFR: `--algorithm external_mccfr`
2. Reduce game size (fewer players, lower stacks, simpler abstraction)
3. Use C++ implementation: `--algorithm cpp_cfr_plus`

### Slow Convergence

If exploitability isn't decreasing:

1. Check that algorithm is appropriate for game size
2. Increase iterations
3. Try different algorithm (e.g., CFR+ instead of vanilla CFR)
4. Verify game configuration is correct

### Import Errors

```
ModuleNotFoundError: No module named 'pyspiel'
```

**Solution**: Activate OpenSpiel environment:
```bash
source ~/open_spiel/venv/bin/activate
```

---

## Next Steps

- See `ALGORITHM_COMPARISON.md` for detailed algorithm analysis
- See `test_poker_configs.py` for game configuration examples
- See `test_tensor_bet_sizes.py` for understanding information state tensors

For questions or issues, refer to the OpenSpiel documentation or the project CLAUDE.md file.
