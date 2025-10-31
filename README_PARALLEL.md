# Parallel DCFR Research Validation

This document explains how to use the parallel implementation of the DCFR research validation script, which provides **~5x speedup** on multi-core systems.

## Overview

The parallel implementation runs 5 different DCFR algorithm configurations simultaneously using Python's multiprocessing module, significantly reducing total computation time.

### Key Features

✅ **~5x faster** - Parallel execution on multi-core systems
✅ **CPU limiting** - Configurable limit (default 80%) via `--cpu-limit` flag
✅ **Thermal throttling** - Optional cooldown delays via `--throttle-delay` for temperature control
✅ **Smart cleanup** - Recursive process tree termination on Ctrl+C
✅ **Memory safe** - Independent worker processes with file locking
✅ **Checkpoint support** - Resume interrupted runs
✅ **Best Nash tracking** - Automatic tracking of minimum (best) Nash convergence with special checkpoints
✅ **Real-time monitoring** - Progress updates from all workers with current and historical best Nash
✅ **No pandas required** - Uses standard library only

### Performance Comparison

| Implementation | 1M Iterations | 100K Iterations | CPU Cores Used |
|----------------|---------------|-----------------|----------------|
| **Sequential** (`compare_dcfr_research_3p.py`) | ~2 hours | ~12 minutes | 1 core |
| **Parallel** (`compare_dcfr_research_3p_parallel.py`) | ~24 minutes | ~2.5 minutes | Half of available (default) |

**Speedup**: ~5x faster on systems with 5+ CPU cores
**Default workers**: Uses **half of available CPU cores** (prevents overheating, leaves headroom for system)

## Quick Start

### 1. Prerequisites

Ensure you have:
- OpenSpiel virtual environment installed at `~/open_spiel/venv`
- Multi-core CPU (recommended: 5+ cores for full speedup)
- Sufficient RAM (~5x more than sequential, but still modest for Kuhn poker)

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

**Full research validation (1M iterations, ~24 minutes):**
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
| `--max-workers` | Half of CPU cores | Maximum parallel workers (capped at 6 for this script) |
| `--throttle-delay` | 0.0 | Seconds to sleep after each exploitability check (thermal control) |

## Advanced Usage

### Thermal Throttling (NEW!)

If your CPU is still hitting max temperature even with `cpulimit`, you can add **throttle delays** to insert cooling periods during computation:

```bash
# Add 1-second cooldown after each exploitability check
python compare_dcfr_research_3p_parallel.py \
    --iterations 100000 \
    --check-interval 10000 \
    --throttle-delay 1.0

# More aggressive: 3-second cooldown (cooler CPU, slower runtime)
python compare_dcfr_research_3p_parallel.py \
    --iterations 100000 \
    --check-interval 10000 \
    --throttle-delay 3.0

# Combine with cpulimit and reduced workers for maximum cooling
bash run_with_cpulimit.sh -c 60 compare_dcfr_research_3p_parallel.py \
    --iterations 100000 \
    --check-interval 10000 \
    --max-workers 3 \
    --throttle-delay 2.0
```

**How it works:**
- After each exploitability check (every `--check-interval` iterations), workers sleep for the specified duration
- This gives the CPU periodic rest periods to cool down
- **Trade-off:** Increases total runtime proportionally to throttle delay
- **Example:** With `--check-interval 10000` and `--throttle-delay 2.0`, workers sleep for 2 seconds every 10,000 iterations

**When to use:**
- ✅ Laptops with limited cooling
- ✅ CPUs hitting thermal throttling despite `cpulimit`
- ✅ Long runs where sustained heat is a concern
- ✅ Background training while using system for other tasks

**Recommended values:**
- **Light throttling**: `--throttle-delay 0.5` (minimal impact, ~5-10% slower)
- **Moderate throttling**: `--throttle-delay 1.0` to `2.0` (noticeable cooling, 10-20% slower)
- **Heavy throttling**: `--throttle-delay 3.0` to `5.0` (significant cooling, 20-40% slower)

**Combined thermal control strategy:**
```bash
# Three-layer thermal protection (recommended for laptops):
bash run_with_cpulimit.sh --cpu-limit 70 compare_dcfr_research_3p_parallel.py \
    --iterations 1000000 \
    --check-interval 50000 \
    --checkpoint-interval 100000 \
    --max-workers 3 \
    --throttle-delay 1.5
```

This combines:
1. **cpulimit** at 70% (hard CPU cap)
2. **max-workers 3** (reduce concurrent work)
3. **throttle-delay 1.5s** (periodic cooldown breaks)

### Checkpointing for Long Runs

Enable checkpointing to save progress and allow resuming if interrupted:

```bash
python compare_dcfr_research_3p_parallel.py \
    --iterations 1000000 \
    --check-interval 50000 \
    --checkpoint-interval 100000
```

If interrupted, simply re-run the same command to resume from the last checkpoint.

### Minimum Nash Tracking and "Best" Checkpoints

The script automatically tracks the **minimum (best) Nash convergence** value throughout training, which is important because DCFR algorithms are **non-monotonic** - Nash convergence can temporarily increase before decreasing again.

**Key Features:**
- ✅ Tracks the lowest Nash convergence value and its iteration for each worker
- ✅ Displays both current and historical best Nash in real-time
- ✅ Automatically saves special "best" checkpoint when a new minimum is found
- ✅ Completed workers show both final and best Nash values

**Example Live Display:**

```
AGGREGATE:    50,000/100,000 avg | Rate:  5,234 total it/s | ETA:   0.2m | Active: 5/5
================================================================================

CFR+ Approx     |  50,000/100,000 ( 50.0%) |    892 it/s | Nash: 0.251203 (Best: 0.150311 @15,000)
```

In this example:
- **Current Nash**: 0.251203 (algorithm has temporarily regressed)
- **Best Nash**: 0.150311 (achieved at iteration 15,000)
- **Benefit**: You can resume from iteration 15,000 checkpoint instead of continuing from a worse state

**Best Checkpoint Files:**

When checkpointing is enabled, the script saves two types of checkpoints:

1. **Regular checkpoints** (interval-based):
   ```
   checkpoints/3p_kuhn_dcfr_research_CFR_PLUS_APPROX_iter_50000.pkl
   checkpoints/3p_kuhn_dcfr_research_CFR_PLUS_APPROX_iter_100000.pkl
   ```

2. **Best checkpoints** (saved when new minimum Nash found):
   ```
   checkpoints/3p_kuhn_dcfr_research_CFR_PLUS_APPROX_best_iter_15000_nash_0.150311.pkl
   checkpoints/3p_kuhn_dcfr_research_CFR_PLUS_APPROX_best_iter_45000_nash_0.098542.pkl
   ```

The filename includes:
- `best` marker
- Iteration number when best Nash was achieved
- The Nash convergence value (for easy identification)

**Resuming from Best Checkpoint:**

To resume training from the best checkpoint instead of the latest:

```bash
# 1. Find the best checkpoint for your algorithm
ls -lh checkpoints/*SOTA_DCFR*best*.pkl

# 2. Note the iteration number (e.g., 45000)

# 3. Remove later checkpoints to force resume from best
rm checkpoints/3p_kuhn_dcfr_research_SOTA_DCFR_iter_50000.pkl
rm checkpoints/3p_kuhn_dcfr_research_SOTA_DCFR_iter_100000.pkl

# 4. Rename best checkpoint to regular format
cp checkpoints/3p_kuhn_dcfr_research_SOTA_DCFR_best_iter_45000_nash_0.098542.pkl \
   checkpoints/3p_kuhn_dcfr_research_SOTA_DCFR_iter_45000.pkl

# 5. Resume training
python compare_dcfr_research_3p_parallel.py --iterations 100000 --checkpoint-interval 10000
```

**Why This Matters:**

DCFR algorithms often show **non-monotonic convergence** where Nash values can increase temporarily before improving again. By tracking the historical best, you can:
- Identify the optimal checkpoint to resume from
- Avoid wasting computation on regressed states
- Compare final vs best performance to understand convergence behavior

### Adjusting Parallel Workers

**Default**: Uses **half of available CPU cores** (e.g., 5 workers on a 10-core system, capped at 5 max)

To use a different number of workers:

```bash
# Use all 12 cores (maximum performance, higher heat)
python compare_dcfr_research_3p_parallel.py --iterations 100000 --max-workers 12

# Use only 3 workers (very conservative)
python compare_dcfr_research_3p_parallel.py --iterations 100000 --max-workers 3

# Use default (half of cores, recommended)
python compare_dcfr_research_3p_parallel.py --iterations 100000
```

### Force Restart (Ignore Checkpoints)

```bash
python compare_dcfr_research_3p_parallel.py --iterations 100000 --force-restart
```

## How It Works

### Architecture

1. **Main Process**: Coordinates workers, monitors progress, displays results
2. **Worker Processes** (5): Each runs one DCFR algorithm independently
3. **Shared Queue**: Workers send progress updates to main process
4. **File Locking**: CSV writes are synchronized using `fcntl.flock()`

### Worker Independence

Each of the 5 workers runs completely independently:
- SIMPLE (External Sampling, uniform averaging)
- FULL (External Sampling, reach-weighted averaging)
- True LCFR (DCFR with α=1, β=1, γ=1)
- SOTA DCFR (DCFR with α=1.5, β=0, γ=2) - Research best
- CFR+ Approx (DCFR with α=∞, β=∞, γ=2)

### CPU Management

The script automatically reduces CPU contention:
- Sets process priority to low (`os.nice(10)`)
- Optional hard limit via `cpulimit` wrapper (80% default)
- Workers run truly in parallel (multiprocessing bypasses Python GIL)

## Output

### Console Output

During execution, you'll see a **live updating display** that shows:
1. **Aggregate progress** - Overall statistics across all workers
2. **Individual worker progress** - Current iteration, percentage, and status for each algorithm
3. **Completion notifications** when each worker finishes
4. **Final rankings** comparing all 6 algorithms
5. **Research validation** checking if results match published claims

**Live Progress Display Example (6 workers, default):**

```
================================================================================
AGGREGATE:    48,333/100,000 avg | Rate:  5,234 total it/s | ETA:   0.2m | Active: 5/6
================================================================================

SIMPLE          |  50,000/100,000 ( 50.0%) |    920 it/s | ✓ COMPLETED (Final: 0.101773, Best: 0.098542 @45,000)
FULL            |  48,500/100,000 ( 48.5%) |    892 it/s | Nash: 0.123456 (Best: 0.115432 @42,000)
True LCFR       |  47,200/100,000 ( 47.2%) |    868 it/s | Nash: 0.234567 (Best: 0.234567 @47,200)
SOTA DCFR       |  49,100/100,000 ( 49.1%) |    903 it/s | Working...
CFR+ Approx     |  48,800/100,000 ( 48.8%) |    897 it/s | Nash: 0.345678 (Best: 0.320145 @38,500)
DCFR(0,0,1)     |  47,900/100,000 ( 47.9%) |    881 it/s | Working...
```

**With Limited Workers (e.g., --max-workers 3):**

```
================================================================================
AGGREGATE:    15,000/100,000 avg | Rate:  2,150 total it/s | ETA:   0.7m | Active: 3/6
================================================================================

SIMPLE          |  30,000/100,000 ( 30.0%) |    850 it/s | Working...
FULL            |  15,000/100,000 ( 15.0%) |    750 it/s | Working...
True LCFR       |  15,000/100,000 ( 15.0%) |    550 it/s | Working...
SOTA DCFR       |       0/100,000 (  0.0%) |      0 it/s | ⏳ Queued
CFR+ Approx     |       0/100,000 (  0.0%) |      0 it/s | ⏳ Queued
DCFR(0,0,1)     |       0/100,000 (  0.0%) |      0 it/s | ⏳ Queued
```

**Display refreshes every 2 seconds** and shows:
- **Aggregate stats** - Combined progress, rate, ETA, and active worker count
- **Per-worker stats** - Current iteration, completion percentage, **individual iteration rate**, and status
- **Iteration rates** - Shows how fast each algorithm is running (it/s = iterations per second)
- **Status indicators**:
  - `⏳ Queued` - Worker waiting for an available slot (when max_workers < 6)
  - `Working...` - Worker actively running iterations
  - `Nash: X.XXXXXX` - Most recent exploitability measurement
  - `Nash: X.XXXXXX (Best: Y.YYYYYY @N)` - Current Nash with historical best and its iteration
  - `✓ COMPLETED (Final: X.XXXXXX, Best: Y.YYYYYY @N)` - Worker finished, showing both final and best Nash
  - `⚠ INTERRUPTED` - Worker stopped by user (Ctrl+C)
  - `✗ FAILED` - Worker encountered an error

**Why different rates?** Each CFR algorithm variant has different computational complexity per iteration, so you'll see varying it/s rates across workers.

**Queued workers:** When you limit workers (e.g., `--max-workers 3`), up to 3 algorithms will show `Working...` while others show `⏳ Queued` until a slot opens up.

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

If enabled, the script saves two types of checkpoints:

**1. Regular Checkpoints** (interval-based):
- Path: `checkpoints/3p_kuhn_dcfr_research_{ALGORITHM}_iter_{N}.pkl`
- Saved every `--checkpoint-interval` iterations
- Used for resuming interrupted runs

**2. Best Checkpoints** (minimum Nash):
- Path: `checkpoints/3p_kuhn_dcfr_research_{ALGORITHM}_best_iter_{N}_nash_{VALUE}.pkl`
- Saved automatically when a new minimum Nash convergence is found
- Filename includes the Nash value for easy identification
- Allows resuming from the optimal point instead of the latest checkpoint

Each checkpoint contains:
- Solver state (regrets, strategies)
- Current iteration number
- Algorithm identifier
- Nash convergence value (for "best" checkpoints only)

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

The parallel version uses ~5x more memory than sequential (one solver per worker).

**Solutions:**
1. Reduce number of workers: `--max-workers 3`
2. Use sequential version for very large games
3. Add swap space (not recommended for SSDs)

For 3-player Kuhn poker, memory usage is minimal (~100MB total).

### System Freezing / Overheating

**Option 1: Use CPU limiting (recommended first step):**
```bash
bash run_with_cpulimit.sh compare_dcfr_research_3p_parallel.py --iterations 100000
```

**Option 2: Reduce CPU limit:**
```bash
bash run_with_cpulimit.sh --cpu-limit 50 compare_dcfr_research_3p_parallel.py --iterations 100000
```

**Option 3: Add thermal throttling (NEW - most effective for sustained heat):**
```bash
python compare_dcfr_research_3p_parallel.py \
    --iterations 100000 \
    --throttle-delay 1.5
```

**Option 4: Reduce workers:**
```bash
python compare_dcfr_research_3p_parallel.py --iterations 100000 --max-workers 4
```

**Option 5: Combine all strategies (maximum cooling):**
```bash
bash run_with_cpulimit.sh --cpu-limit 60 compare_dcfr_research_3p_parallel.py \
    --iterations 100000 \
    --max-workers 3 \
    --throttle-delay 2.0
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
| **Speed** | 1x baseline | ~5x faster |
| **Memory** | 1x baseline | ~5x more |
| **CPU cores used** | 1 | Half of available (default) |
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

### Quick Research Validation (2.5 minutes)

```bash
source ~/open_spiel/venv/bin/activate
python compare_dcfr_research_3p_parallel.py --iterations 100000 --check-interval 10000
```

### Full Research Validation with Safety (24 minutes)

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
    --max-workers 4 \
    --throttle-delay 1.0
```

### Maximum Thermal Control (for overheating systems)

```bash
source ~/open_spiel/venv/bin/activate
bash run_with_cpulimit.sh --cpu-limit 60 compare_dcfr_research_3p_parallel.py \
    --iterations 1000000 \
    --check-interval 50000 \
    --checkpoint-interval 100000 \
    --max-workers 3 \
    --throttle-delay 2.0
```

This combines all thermal control strategies:
- CPU limit at 60%
- Only 3 workers
- 2-second cooldown after each check
- Slowest but coolest option

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
