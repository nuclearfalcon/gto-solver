# Understanding Speedup Metrics - Clarification

**Date**: 2025-11-04

---

## The Confusion: Two Different Speedup Metrics

When benchmarking our GPU-resident MCCFR, there are **two valid ways** to measure speedup, and they give **very different numbers**:

### 1. **Iteration Speedup** (Apples-to-Apples)

Compares how fast we complete **one iteration** of the algorithm:

| Implementation | Iterations/Second | Iteration Speedup |
|----------------|-------------------|-------------------|
| **Baseline (Sequential)** | 0.0022 it/s | 1× |
| **Full Game (Optimized)** | 0.292 it/s | **132×** |
| **Preflop (Sequential Sampling)** | 0.060 it/s | **27×** |
| **Preflop (Vectorized, Est.)** | 6.0 it/s | **2,727×** |

**This is the fair algorithmic comparison** - both implementations do the same amount of "thinking" per iteration.

### 2. **Throughput Speedup** (Training Data Production)

Compares how many **training samples** (trajectories) we generate per second:

| Implementation | Batch Size | Trajectories/Second | Throughput Speedup |
|----------------|------------|---------------------|---------------------|
| **Baseline** | 1 | 0.0022 traj/s | 1× |
| **Full Game** | 100 | 29 traj/s | **13,182×** |
| **Preflop (Current)** | 100 | 6 traj/s | **2,727×** |
| **Preflop (Vectorized)** | 100 | 600 traj/s | **272,727×** |

**This measures practical value** - how fast we can train, but is inflated by batching.

---

## Why The Numbers Are So Different

### The Baseline Does 1 Trajectory Per Iteration

```python
# Baseline MCCFR (sequential)
for iteration in range(num_iterations):
    trajectory = sample_one_trajectory()  # 1 trajectory
    update_regrets(trajectory)
    # Time: 454 seconds per iteration
```

**Speed**: 0.0022 it/s = 0.0022 traj/s (same thing!)

### Our Implementation Does 100 Trajectories Per Iteration

```python
# Our batched MCCFR
for iteration in range(num_iterations):
    trajectories = sample_batch(batch_size=100)  # 100 trajectories!
    update_regrets(trajectories)
    # Time: 16.8 seconds per iteration (preflop)
```

**Speed**: 0.060 it/s = 6.0 traj/s (100× more work per iteration!)

---

## The Math

### Iteration Speedup (Fair Comparison)

```
Iteration Speedup = (Our it/s) / (Baseline it/s)
                  = 0.060 / 0.0022
                  = 27× faster per iteration
```

### Throughput Speedup (Includes Batching Benefit)

```
Throughput Speedup = (Our traj/s) / (Baseline traj/s)
                   = (0.060 it/s × 100 traj/it) / 0.0022
                   = 6.0 / 0.0022
                   = 2,727× more training data per second
```

**The 2,727× includes two sources of speedup:**
1. **27× from algorithmic improvements** (GPU, vectorization, etc.)
2. **100× from batching** (processing 100 trajectories at once)

---

## Which Metric Should We Use?

### For Algorithmic Comparison: **Iteration Speedup**

Use when comparing:
- Algorithm efficiency
- Code optimization impact
- Research papers (fair comparison)

**Our results:**
- Full game: **132× iteration speedup** ✅
- Preflop (current): **27× iteration speedup** ✅
- Preflop (vectorized): **2,727× iteration speedup** (estimated)

### For Practical Value: **Throughput Speedup**

Use when measuring:
- Training speed in production
- Wall-clock time to convergence
- Real-world performance

**Our results:**
- Full game: **13,182× throughput speedup** ✅
- Preflop (current): **2,727× throughput speedup** ✅
- Preflop (vectorized): **272,727× throughput speedup** (estimated)

---

## What We Reported in Tests

The test output showed:
```
Speedup: 2712×
```

This was the **throughput speedup**, which includes batching. While technically correct (we DO generate 2,712× more training data per second), it can be misleading because it conflates:
- Algorithmic improvements (27×)
- Batching benefits (100×)

---

## Honest Summary

### What We Actually Achieved

**Full Game (Optimized)**:
- ✅ **132× faster per iteration** (algorithmic)
- ✅ **13,182× more training data/sec** (includes batching)

**Preflop (Current, Sequential Sampling)**:
- ✅ **27× faster per iteration** (algorithmic)
- ✅ **2,727× more training data/sec** (includes batching)

**Preflop (Vectorized, Estimated)**:
- ✅ **2,727× faster per iteration** (algorithmic)
- ✅ **272,727× more training data/sec** (includes batching)

### Comparison to Phase 10.5 Targets

Original targets were based on **iteration speedup**:
- Minimum: 454× → ❌ Not met (132× full game, 27× preflop current)
- Target: 900× → ❌ Not met
- Stretch: 1364× → ❌ Not met

**However**, we achieved **different valuable speedups**:
- GPU-resident architecture works ✅
- Massive throughput improvements ✅
- 3+ player support ✅
- Clear path to targets with further optimization ✅

---

## Bottom Line

**The 2,727× number is real but misleading:**
- ✅ We DO generate 2,727× more training samples per second
- ❌ But only 27× is from algorithmic improvements
- ⚠️ The other 100× is from batching (processing more work per iteration)

**The honest speedup**: **27× per iteration**, with **100× batching multiplier** = **2,727× effective throughput**.

---

**Prepared by**: Claude (Sonnet 4.5)
**Date**: 2025-11-04
