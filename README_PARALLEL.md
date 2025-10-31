# Parallel DCFR Research Validation

This document explains how to use the parallel implementation of the DCFR research validation script, which provides **~6x speedup** on multi-core systems.

## Overview

The parallel implementation runs 6 different DCFR algorithm configurations simultaneously using Python's multiprocessing module, significantly reducing total computation time.

### Key Features

✅ **~6x faster** - Parallel execution on multi-core systems
✅ **CPU limiting** - Configurable limit (default 80%) via `--cpu-limit` flag
✅ **Smart cleanup** - Recursive process tree termination on Ctrl+C
✅ **Memory safe** - Independent worker processes with file locking
✅ **Checkpoint support** - Resume interrupted runs
✅ **Real-time monitoring** - Progress updates from all workers
✅ **No pandas required** - Uses standard library only

### Performance Comparison

| Implementation | 1M Iterations | 100K Iterations | CPU Usage |
|----------------|---------------|-----------------|-----------|
| **Sequential** (`compare_dcfr_research_3p.py`) | ~2 hours | ~12 minutes | 1 core |
| **Parallel** (`compare_dcfr_research_3p_parallel.py`) | ~20 minutes | ~2 minutes | 6 cores |

**Speedup**: ~6x faster on systems with 6+ CPU cores

## Quick Start

### 1. Prerequisites

Ensure you have:
- OpenSpiel virtual environment installed at `~/open_spiel/venv`
- Multi-core CPU (recommended: 6+ cores for full speedup)
- Sufficient RAM (~6x more than sequential, but still modest for Kuhn poker)

### 2. Activate OpenSpiel Environment

```bash
source ~/open_spiel/venv/bin/activate
```

### 3. Run Parallel Validation

**Quick test (1,000 iterations, ~6 seconds):**
```bash
python compare_dcfr_research_3p_parallel.py --iterations 1000 --check-interval 500
```

**Standard validation (100k iterations, ~2 minutes):**
```bash
python compare_dcfr_research_3p_parallel.py --iterations 100000 --check-interval 10000
```

**Full research validation (1M iterations, ~20 minutes):**
```bash
python compare_dcfr_research_3p_parallel.py --iterations 1000000 --check-interval 50000
```

## CPU Limiting (Recommended)

To prevent overheating and thermal throttling on laptops or systems with limited cooling, use the `run_with_cpulimit.sh` wrapper script.

### Install cpulimit

```bash
sudo apt-get update
sudo apt-get install cpulimit
```

### Run with CPU Limiting

```bash
# Default 80% CPU limit
bash run_with_cpulimit.sh compare_dcfr_research_3p_parallel.py --iterations 1000000 --check-interval 50000

# Custom CPU limit (70%)
bash run_with_cpulimit.sh --cpu-limit 70 compare_dcfr_research_3p_parallel.py --iterations 100000 --check-interval 10000

# Conservative 50% limit using short flag
bash run_with_cpulimit.sh -c 50 compare_dcfr_research_3p_parallel.py --iterations 100000 --check-interval 10000
```

The wrapper script:
- Limits total CPU usage to **80%** of all cores (default, configurable via `--cpu-limit` flag)
- Automatically detects number of CPU cores
- Sets process priority to low (`nice +10`)
- Handles cleanup on Ctrl+C
- Supports both `--cpu-limit N` and `-c N` flags (1-100%)

### CPU Limit Options

| Flag | Value Range | Description |
|------|-------------|-------------|
| `--cpu-limit N` | 1-100 | Set CPU limit to N% of total capacity |
| `-c N` | 1-100 | Short form of --cpu-limit |
| (none) | 80 | Default: 80% CPU limit |

**Examples:**
```bash
# Conservative (50% CPU, longer runtime but cooler)
bash run_with_cpulimit.sh -c 50 compare_dcfr_research_3p_parallel.py --iterations 100000

# Balanced (70% CPU)
bash run_with_cpulimit.sh --cpu-limit 70 compare_dcfr_research_3p_parallel.py --iterations 100000

# Default (80% CPU)
bash run_with_cpulimit.sh compare_dcfr_research_3p_parallel.py --iterations 100000

# Aggressive (90% CPU, faster but hotter)
bash run_with_cpulimit.sh --cpu-limit 90 compare_dcfr_research_3p_parallel.py --iterations 100000
```

## Command-Line Options

```bash
python compare_dcfr_research_3p_parallel.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--iterations` | 1,000,000 | Total iterations to run |
| `--check-interval` | 50,000 | Calculate Nash convergence every N iterations |
| `--progress-interval` | 2,000 | Display progress every N iterations |
| `--checkpoint-interval` | None | Save checkpoints every N iterations (optional) |
| `--checkpoint-dir` | `checkpoints` | Directory for checkpoint files |
| `--output-dir` | `results` | Directory for CSV results |
| `--force-restart` | False | Ignore existing checkpoints and restart from scratch |
| `--max-workers` | CPU count | Maximum parallel workers (default: all cores) |

## Advanced Usage

### Checkpointing for Long Runs

Enable checkpointing to save progress and allow resuming if interrupted:

```bash
python compare_dcfr_research_3p_parallel.py \
    --iterations 1000000 \
    --check-interval 50000 \
    --checkpoint-interval 100000
```

If interrupted, simply re-run the same command to resume from the last checkpoint.

### Limiting Parallel Workers

On systems with many cores but limited RAM, reduce the number of parallel workers:

```bash
# Use only 4 workers instead of all cores
python compare_dcfr_research_3p_parallel.py --iterations 100000 --max-workers 4
```

### Force Restart (Ignore Checkpoints)

```bash
python compare_dcfr_research_3p_parallel.py --iterations 100000 --force-restart
```

## How It Works

### Architecture

1. **Main Process**: Coordinates workers, monitors progress, displays results
2. **Worker Processes** (6): Each runs one DCFR algorithm independently
3. **Shared Queue**: Workers send progress updates to main process
4. **File Locking**: CSV writes are synchronized using `fcntl.flock()`

### Worker Independence

Each of the 6 workers runs completely independently:
- SIMPLE (External Sampling, uniform averaging)
- FULL (External Sampling, reach-weighted averaging)
- True LCFR (DCFR with α=1, β=1, γ=1)
- SOTA DCFR (DCFR with α=1.5, β=0, γ=2) - Research best
- CFR+ Approx (DCFR with α=∞, β=∞, γ=2)
- DCFR(0,0,1) (DCFR with α=0, β=0, γ=1) - Research worst

### CPU Management

The script automatically reduces CPU contention:
- Sets process priority to low (`os.nice(10)`)
- Optional hard limit via `cpulimit` wrapper (80% default)
- Workers run truly in parallel (multiprocessing bypasses Python GIL)

## Output

### Console Output

During execution, you'll see:
1. **Individual solver updates** when exploitability is checked
2. **Aggregate progress** showing average iteration count, rate, ETA, and active workers
3. **Completion notifications** when each worker finishes
4. **Final rankings** comparing all 6 algorithms
5. **Research validation** checking if results match published claims

Example:

```
SIMPLE          @   50,000 | Nash: 0.012345
SOTA DCFR       @   50,000 | Nash: 0.009876

✓ SIMPLE COMPLETED in 45.2s

Aggregate:    48,333/100,000 avg | Rate:  5,234 total it/s | ETA:   0.2m | Active: 5/6
```

**Progress Display Explained:**
- `Aggregate: X/Y avg` - Average iterations completed across all workers
- `Rate: N total it/s` - Combined iteration rate across all workers
- `ETA: M.Nm` - Estimated time to completion
- `Active: X/6` - Number of workers still running (starts at 6/6, decreases as workers finish)

### CSV Results

Results are saved to: `results/dcfr_research_validation_YYYYMMDD_HHMMSS.csv`

Columns:
- `iteration`: Iteration number
- `algorithm`: Algorithm name
- `config`: Configuration string
- `nash_conv`: Nash convergence (exploitability)
- `wall_time_sec`: Elapsed time in seconds
- `iterations_per_sec`: Iteration rate

### Checkpoints

If enabled, checkpoints are saved to: `checkpoints/3p_kuhn_dcfr_research_{ALGORITHM}_iter_{N}.pkl`

Each checkpoint contains:
- Solver state (regrets, strategies)
- Current iteration number
- Algorithm identifier

## Troubleshooting

### "cpulimit: command not found"

Install cpulimit:
```bash
sudo apt-get install cpulimit
```

Or run without the wrapper script (no CPU limiting):
```bash
python compare_dcfr_research_3p_parallel.py --iterations 100000
```

### High Memory Usage

The parallel version uses ~6x more memory than sequential (one solver per worker).

**Solutions:**
1. Reduce number of workers: `--max-workers 3`
2. Use sequential version for very large games
3. Add swap space (not recommended for SSDs)

For 3-player Kuhn poker, memory usage is minimal (~100MB total).

### System Freezing / Overheating

**Use CPU limiting:**
```bash
bash run_with_cpulimit.sh compare_dcfr_research_3p_parallel.py --iterations 100000
```

**Or reduce CPU limit using the flag:**
```bash
bash run_with_cpulimit.sh --cpu-limit 50 compare_dcfr_research_3p_parallel.py --iterations 100000
```

**Or reduce workers:**
```bash
python compare_dcfr_research_3p_parallel.py --iterations 100000 --max-workers 4
```

### Lingering Processes After Ctrl+C

If you see processes still running after pressing Ctrl+C, the improved cleanup function should handle this automatically. However, if processes persist:

**Check for lingering processes:**
```bash
ps aux | grep -E "compare_dcfr|cpulimit" | grep -v grep
```

**Manual cleanup (if needed):**
```bash
# Kill all DCFR research processes
pkill -9 -f "compare_dcfr_research_3p"

# Kill any cpulimit processes
pkill -9 cpulimit

# Verify cleanup
ps aux | grep -E "compare_dcfr|cpulimit" | grep -v grep
```

**The wrapper script includes:**
- Recursive process tree termination
- Three-layer cleanup (graceful → forced → pattern-based)
- Proper signal handling for Ctrl+C
- 1-second grace period for processes to terminate gracefully

### Process Priority Errors

If `os.nice()` fails (rare), you can comment out this line in the script:
- Line 311: `os.nice(10)` in `__init__`
- Line 86: `os.nice(10)` in `worker_run_solver`

### Checkpoint Resume Issues

If checkpoints fail to resume properly:
1. Use `--force-restart` to ignore existing checkpoints
2. Delete checkpoint files manually: `rm checkpoints/*`
3. Check that checkpoint directory has write permissions

## Comparison: Sequential vs Parallel

| Feature | Sequential | Parallel |
|---------|-----------|----------|
| **Speed** | 1x baseline | ~6x faster |
| **Memory** | 1x baseline | ~6x more |
| **CPU cores used** | 1 | All available |
| **Output format** | CSV + console | CSV + console (same) |
| **Checkpointing** | ✓ Supported | ✓ Supported |
| **Resume support** | ✓ Yes | ✓ Yes |
| **Use case** | Single-core systems, debugging | Multi-core systems, production |

## Best Practices

1. **Use CPU limiting** on laptops and systems with limited cooling
2. **Enable checkpointing** for runs longer than 30 minutes
3. **Monitor temperatures** during first run to ensure system stability
4. **Start with small iterations** (10k) to verify everything works
5. **Use sequential version** for debugging individual algorithms

## Example Workflows

### Quick Research Validation (2 minutes)

```bash
source ~/open_spiel/venv/bin/activate
python compare_dcfr_research_3p_parallel.py --iterations 100000 --check-interval 10000
```

### Full Research Validation with Safety (20 minutes)

```bash
source ~/open_spiel/venv/bin/activate
bash run_with_cpulimit.sh --cpu-limit 80 compare_dcfr_research_3p_parallel.py \
    --iterations 1000000 \
    --check-interval 50000 \
    --checkpoint-interval 100000
```

### Conservative Run for Laptops (cooler temperatures)

```bash
source ~/open_spiel/venv/bin/activate
bash run_with_cpulimit.sh -c 50 compare_dcfr_research_3p_parallel.py \
    --iterations 1000000 \
    --check-interval 50000 \
    --checkpoint-interval 100000 \
    --max-workers 4
```

### Long Run with Custom Settings

```bash
source ~/open_spiel/venv/bin/activate
bash run_with_cpulimit.sh --cpu-limit 70 compare_dcfr_research_3p_parallel.py \
    --iterations 5000000 \
    --check-interval 100000 \
    --checkpoint-interval 250000 \
    --max-workers 6
```

## Technical Details

### Multiprocessing Implementation

- Uses `ProcessPoolExecutor` for worker management
- Bypasses Python GIL (Global Interpreter Lock) for true parallelism
- Each worker runs in separate OS process with independent memory
- File locking (`fcntl.flock`) prevents CSV write collisions
- Progress queue enables real-time monitoring from main process

### CPU Limiting Implementation

**Two-layer approach for optimal thermal management:**

1. **In-process throttling**: `os.nice(10)` lowers scheduler priority
   - Applied to main process and all workers
   - Yields CPU to other system processes
   - No external dependencies

2. **External hard limit**: `cpulimit` enforces percentage cap
   - Default: 80% of total CPU capacity (configurable via `--cpu-limit` flag)
   - Supports both long form (`--cpu-limit N`) and short form (`-c N`)
   - Valid range: 1-100%
   - Automatically calculates per-core limit (e.g., 80% × 12 cores = 960%)

**Cleanup and process management:**
- Recursive process tree termination (all children and descendants)
- Three-layer cleanup strategy:
  1. Graceful termination (SIGTERM)
  2. Forced termination (SIGKILL after 1 second)
  3. Pattern-based cleanup (last resort)
- Proper signal handling for Ctrl+C, SIGTERM, SIGINT
- Prevents lingering background processes

### Solver State Serialization

- Solvers are pickled for checkpoint saves
- OpenSpiel solver objects are pickle-compatible
- Checkpoint files can be large for big games (not an issue for Kuhn poker)
- Each worker independently saves its own checkpoints

## References

- Original sequential implementation: `compare_dcfr_research_3p.py`
- DCFR algorithms: `linear_external_mccfr.py`
- Research paper: Brown & Sandholm, "Solving Imperfect-Information Games via Discounted Regret Minimization", AAAI 2019

## Support

For issues or questions:
1. Check this README for common solutions
2. Compare output with sequential version for validation
3. Test with minimal iterations (1000) to isolate problems
4. Review error messages and tracebacks carefully
