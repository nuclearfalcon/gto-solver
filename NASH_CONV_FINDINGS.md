# Critical Findings: Nash Convergence for GTO Poker

## Executive Summary

**Major Discovery**: Our sampled exploitability implementation has a fundamental methodological bug that causes it to report values 4-5x higher than actual. The correct approach is to use **OpenSpiel's exact `nash_conv`** function, which is fast enough for small/medium games and works for 2+ players.

## The Bug in Sampled Exploitability

### What's Wrong

**File**: `exploitability_metrics.py`, lines 184-237 (`_sample_hand_exploitability`)

**The Problem**:
1. We deal cards first (skip chance nodes at lines 201-205)
2. Then compute exploitability from that specific card deal
3. We're measuring "conditional exploitability given cards"
4. NOT "expected exploitability from initial state"

**Impact**:
- Sampled: 69.3 chips exploitability
- Actual (nash_conv): 14.6 chips exploitability
- **Error**: 374% too high!

###The Methodology Issue

```python
# What we do (WRONG):
state = game.new_initial_state()
while state.is_chance_node():  # Skip chance nodes
    state.apply_action(random_action)
# Now compute exploitability from here
```

This computes exploitability AFTER cards are dealt, giving us conditional values that don't match the true game exploitability.

## The Correct Solution: Nash_conv

### Why Nash_conv is Correct

From OpenSpiel's documentation (`exploitability.py`):

```python
def exploitability(game, policy):
    """This is implemented only for 2 players constant-sum games, and is equivalent
    to NashConv / num_players in that case. Prefer using `nash_conv`."""
```

### Key Facts

1. **Nash_conv works for 2+ player games** (exploitability only works for 2-player)
2. **It's EXACT** (no sampling variance or measurement noise)
3. **It's FAST for small games** (0.33s per check on tiny config)
4. **Relationship**: For 2-player games, `exploitability = nash_conv / 2`

### How to Use Nash_conv

```python
from open_spiel.python.algorithms import exploitability

# Create solver
solver = external_sampling_mccfr.ExternalSamplingSolver(game)

# Run iterations
for i in range(iterations):
    solver.iteration()

# Compute exact nash convergence
policy = solver.average_policy()
nash_conv_value = exploitability.nash_conv(game, policy)

# For 2-player games
exploitability_value = nash_conv_value / 2
```

## Performance Analysis

### Tiny Config (2p_5bb_fchpa_tiny.json)

- **Nash_conv time**: 0.33s per check
- **Iteration time**: ~0.85ms per iteration
- **Overhead for 10 checks per 100k iterations**: ~3.3s (5% overhead)
- **Verdict**: Nash_conv is **FAST ENOUGH** for this game size

### Scalability

Nash_conv requires full game tree traversal, so it scales with tree size:

| Game Size | Nash_conv Time | Feasible? |
|-----------|----------------|-----------|
| Tiny (2p, 3 cards) | 0.33s | Yes (5% overhead) |
| Small (2p, 6 cards) | ~5-10s | Maybe (10-20% overhead) |
| Medium (2p, 13 cards) | ~60-300s | Marginal |
| Full Hold'em | Hours | No |

**Recommendation**:
- Use exact nash_conv for small/medium games
- For large games, need different approach (see below)

## Convergence Data (Exact Nash_conv)

### External MCCFR on Tiny Config

Using exact nash_conv (no sampling noise):

```
Iteration  Nash Conv  Exploitability
---------------------------------------
10,000     27.27      13.63
20,000     19.60       9.80
30,000     13.50       6.75
40,000     11.69       5.85
50,000     11.60       5.80
60,000     10.01       5.00
70,000      8.62       4.31
80,000      7.16       3.58
90,000      7.74       3.87
100,000     7.73       3.86
```

**Analysis**:
- 72% reduction in nash_conv (27.27 → 7.73)
- Good convergence trend
- Some noise at 90k-100k (typical for MCCFR)

## Answers to Original Questions

### Question: "How long does External MCCFR take to converge to 1% exploitability?"

**Answer (for tiny config)**:
- 1% = 0.01 exploitability = 0.02 nash_conv
- At 100k iterations: 7.73 nash_conv (still far from target)
- Extrapolating: Would need **~40M iterations** to reach 0.02 nash_conv
- At 1200 it/s: **~9 hours**

**But**: This is for the TINY config (only 3 cards). Full Hold'em would be MUCH slower.

### Question: "Should we use exploitability or nash_conv?"

**Answer**: **Always use nash_conv** because:
1. Works for 2+ player games (exploitability only works for 2-player)
2. Mathematically correct for all game types
3. OpenSpiel documentation explicitly recommends it
4. For 2-player games, exploitability = nash_conv / 2 anyway

## For Larger Games

### Options When Nash_conv is Too Slow

1. **Sampled Nash_conv**: Implement proper sampling that starts from initial state
2. **Head-to-head play**: Measure win rates instead of absolute exploitability
3. **Best response approximation**: Use sampling within best response calculation only
4. **Relative metrics**: Track improvement rather than absolute values
5. **Checkpoint-based**: Run expensive checks only at major milestones

### Fixing Our Sampled Implementation

To fix `exploitability_metrics.py`, we need to:
1. NOT skip chance nodes before computing values
2. Include chance outcome probabilities in the recursive calculation
3. This might require reimplementing to match OpenSpiel's approach

**Alternative**: Just use OpenSpiel's nash_conv when feasible, and accept that large games are hard to evaluate accurately.

## Action Items

1. **Update all tests** to use `exploitability.nash_conv()` instead of sampled exploitability
2. **Document game size limits** where nash_conv is feasible
3. **For large games**: Either fix sampling methodology or use relative/head-to-head metrics
4. **Consider**: Implementing OpenSpiel's exact nash_conv in C++ might be faster than Python

## Test Files Created

1. `test_convergence_comparison.py` - Compares External MCCFR vs CFR+ using exact nash_conv
2. `test_mccfr_averaging.py` - Tests SIMPLE vs FULL averaging (needs updating to use nash_conv)

## References

- OpenSpiel exploitability.py: `/home/nuclearfalcon/open_spiel/open_spiel/python/algorithms/exploitability.py`
- Nash Conv paper: https://arxiv.org/pdf/1711.00832.pdf
- Our buggy implementation: `exploitability_metrics.py:184-237`
