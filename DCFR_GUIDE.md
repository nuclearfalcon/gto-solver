# DCFR Algorithm Selection Guide

This guide helps you choose the right CFR algorithm variant for your poker training application based on empirical testing and research validation.

## Quick Recommendation

**For 3+ player poker training:** Use **External Sampling MCCFR with FULL averaging**

```python
from open_spiel.python.algorithms import external_sampling_mccfr

solver = external_sampling_mccfr.ExternalSamplingSolver(
    game,
    average_type=external_sampling_mccfr.AverageType.FULL
)
```

## Algorithm Performance Summary

Empirical validation on **3-player Kuhn poker** (1M iterations):

| Algorithm | Best Nash Conv | Improvement vs SIMPLE | Use Case |
|-----------|----------------|----------------------|----------|
| **FULL** | **0.001109** | **baseline (best)** | **3+ players (RECOMMENDED)** |
| True LCFR | 0.001601 | +44% worse | Research/comparison |
| CFR+ Approx | 0.002737 | +147% worse | Quadratic averaging only |
| SOTA DCFR | 0.004339 | +291% worse | 2-player games (underperforms on 3p) |
| SIMPLE | 0.024478 | +2107% worse | Baseline/debugging |

**Key Finding:** FULL averaging significantly outperforms all DCFR variants on 3-player games.

---

## Algorithm Descriptions

### 1. SIMPLE (Baseline)

**What it is:** Basic External Sampling MCCFR with simple averaging
- Updates player (i+1)'s average policy during player i's regret pass
- Fastest implementation (no separate averaging pass)
- Used in Pluribus (2-player and multi-player)

**When to use:**
- ❌ Not recommended for production
- ✓ Good for debugging/testing
- ✓ Baseline for comparisons

**Performance:** Worst performer on 3-player games

```python
solver = external_sampling_mccfr.ExternalSamplingSolver(
    game,
    average_type=external_sampling_mccfr.AverageType.SIMPLE
)
```

---

### 2. FULL (RECOMMENDED for 3+ players) ✅

**What it is:** External Sampling MCCFR with reach-probability weighted averaging
- Separate averaging pass with proper reach probability weighting
- Theoretically sound for 3+ player games
- More expensive than SIMPLE (extra tree traversal)

**When to use:**
- ✅ **3+ player poker (BEST PERFORMER)**
- ✅ When you want high-quality policies
- ✅ Training applications where all branches must be "fully baked"

**Performance:** 86% better than SIMPLE on 3-player Kuhn

```python
solver = external_sampling_mccfr.ExternalSamplingSolver(
    game,
    average_type=external_sampling_mccfr.AverageType.FULL
)
```

**Why it wins on 3+ players:**
- Correctly weights contributions by reach probability
- All player strategies updated with proper weighting
- No shortcuts that sacrifice correctness for speed

---

### 3. True LCFR (Linear CFR)

**What it is:** DCFR(1, 1, 1) - Linear weighting and discounting
- γ=1: Linear iteration weighting (t^1 = t)
- α=1: Linear positive regret discounting
- β=1: Linear negative regret discounting

**When to use:**
- ⚠️ Research and comparison purposes
- ❌ Underperforms FULL on 3-player games
- ✓ Historical/academic interest

**Performance:** 2nd best on 3-player Kuhn (but 44% worse than FULL)

```python
from linear_external_mccfr import LinearExternalSamplingSolver

solver = LinearExternalSamplingSolver(
    game,
    gamma=1.0,
    alpha=1.0,
    beta=1.0
)
```

---

### 4. SOTA DCFR (State-of-the-Art)

**What it is:** DCFR(1.5, 0, 2) - Research best for 2-player games
- γ=2: Quadratic iteration weighting (t^2)
- α=1.5: Aggressive positive regret discounting
- β=0: No negative regret discounting

**When to use:**
- ⚠️ Possibly good for 2-player games (research claim)
- ❌ **NOT recommended for 3+ player games** (ranked 4th)
- ⚠️ Experimental/research only

**Performance:** 4th place on 3-player Kuhn (underperformed expectations)

```python
from linear_external_mccfr import LinearExternalSamplingSolver

solver = LinearExternalSamplingSolver(
    game,
    gamma=2.0,
    alpha=1.5,
    beta=0.0
)
```

**Note:** Research showed this as best for 2-player games (Heads-Up Hold'em, Leduc). Our testing suggests it may not generalize to 3+ players.

---

### 5. CFR+ Approximation

**What it is:** DCFR(∞, ∞, 2) - Quadratic averaging, no regret discounting
- γ=2: Quadratic iteration weighting
- α=None: No positive regret discounting
- β=None: No negative regret discounting

**When to use:**
- ⚠️ Testing quadratic averaging effects
- ❌ Not recommended for production

**Performance:** 3rd place on 3-player Kuhn

```python
from linear_external_mccfr import LinearExternalSamplingSolver

solver = LinearExternalSamplingSolver(
    game,
    gamma=2.0,
    alpha=None,
    beta=None
)
```

---

## Decision Tree

```
Do you have 3+ players?
├─ YES → Use FULL ✅
│         (Best empirical performance)
│
└─ NO (2 players)
   ├─ Want research-validated method?
   │  └─ Use SOTA DCFR (1.5, 0, 2)
   │     (Research best for 2p)
   │
   └─ Want simplicity?
      └─ Use FULL or SIMPLE
         (Both work well for 2p)
```

---

## DCFR Parameters Explained

### γ (Gamma) - Strategy Averaging Weighting

Controls how much to weight recent iterations when computing average policy.

| Value | Meaning | Effect |
|-------|---------|--------|
| 0.0 | Uniform | All iterations weighted equally |
| 1.0 | Linear | Weight = iteration number (t) |
| 2.0 | Quadratic | Weight = t² (strongly favors recent) |

**Higher γ** = More weight on recent iterations = Faster adaptation but less stability

### α (Alpha) - Positive Regret Discounting

Controls how quickly positive regrets decay over time.

| Value | Meaning | Effect |
|-------|---------|--------|
| None | No discount | Keep all positive regrets |
| 0.0 | Constant 0.5 | Halve positive regrets every iteration |
| 1.0 | Linear decay | Gradual reduction: t/(t+1) |
| 1.5 | Faster decay | More aggressive reduction |

**Higher α** = Faster forgetting of old good actions

### β (Beta) - Negative Regret Discounting

Controls how quickly negative regrets decay over time.

| Value | Meaning | Effect |
|-------|---------|--------|
| None | No discount | Keep all negative regrets |
| **0.0** | **No discount** | **Keep negative regrets (NOT 0.5!)** |
| 1.0 | Linear decay | Gradual reduction: t/(t+1) |

**CRITICAL:** β=0 means "no discounting" (multiply by 1.0), NOT the formula result t^0/(t^0+1)=0.5. See [DCFR_BUGS_AND_FIXES.md](DCFR_BUGS_AND_FIXES.md) for details.

---

## Common Configurations

### Research-Validated Configurations

From Brown & Sandholm (2019):

```python
# SOTA DCFR (research best for 2-player)
solver = LinearExternalSamplingSolver(game, gamma=2.0, alpha=1.5, beta=0.0)

# True LCFR (original Linear CFR)
solver = LinearExternalSamplingSolver(game, gamma=1.0, alpha=1.0, beta=1.0)

# CFR+ approximation
solver = LinearExternalSamplingSolver(game, gamma=2.0, alpha=None, beta=None)
```

### Empirically-Validated Configurations

From our testing on 3-player Kuhn poker:

```python
# BEST for 3+ players
solver = external_sampling_mccfr.ExternalSamplingSolver(
    game,
    average_type=external_sampling_mccfr.AverageType.FULL
)

# Good baseline
solver = external_sampling_mccfr.ExternalSamplingSolver(
    game,
    average_type=external_sampling_mccfr.AverageType.SIMPLE
)
```

---

## Performance Considerations

### Iteration Speed

| Algorithm | Speed | Reason |
|-----------|-------|--------|
| SIMPLE | Fastest | No separate averaging pass |
| FULL | Slower | Separate averaging tree traversal |
| DCFR variants | Slower | Discounting calculations + averaging |

**For large games:** Use C++ implementations for 100-1000x speedup:
```python
import pyspiel

# C++ External Sampling (FAST!)
solver = pyspiel.ExternalSamplingMCCFRSolver(
    game,
    avg_type=pyspiel.MCCFRAverageType.FULL
)
```

### Memory Usage

All variants have similar memory usage (regrets + average policy for each information state).

DCFR variants may use slightly more due to discounting overhead, but difference is negligible.

---

## Game Size Limitations

### Python MCCFR

**Practical limits:**
- Small games (Kuhn poker): ✅ 200+ it/s
- Tiny Hold'em (2 suits, 3 ranks): ✅ 0.5-2 it/s
- Full Hold'em (52 cards, 3 players): ❌ <0.01 it/s (impractical)

**Recommendation:** Use Kuhn poker for algorithm testing, C++ for larger games

### C++ MCCFR

**100-1000x faster** than Python, but still limited:
- Small games (Kuhn): ✅ 10,000+ it/s
- Medium games: ✅ 10-100 it/s
- Full 3-player Hold'em: ❌ <0.01 it/s (too large)

**Recommendation:** For real Hold'em, use heavy card/action abstractions or outcome sampling

---

## Testing Your Implementation

### Validation Checklist

When implementing or testing DCFR:

1. **Test on Kuhn poker first** - Known equilibrium, fast convergence
2. **Compare against SIMPLE and FULL** - Baselines to detect bugs
3. **Check for non-monotonic convergence** - DCFR Nash values can increase temporarily
4. **Verify β=0 behavior** - Should NOT halve negative regrets (see bugs doc)
5. **Test iteration resumption** - Checkpoint save/load should work
6. **Run long enough** - At least 100k iterations for meaningful comparison

### Expected Convergence

On 3-player Kuhn poker (1M iterations):

```
FULL:        ~0.001 final Nash convergence
SIMPLE:      ~0.02 final Nash convergence
DCFR variants: ~0.001-0.005 final Nash convergence
```

If results are far from these, check for implementation bugs.

---

## Frequently Asked Questions

### Why does FULL outperform SOTA DCFR on 3-player games?

Research focused on 2-player games (Heads-Up Hold'em, Leduc). For 3+ players:
- Reach-probability weighting becomes more important
- Multi-player dynamics differ from heads-up
- DCFR tuning may be 2-player specific

### Should I use DCFR for my poker app?

**For 3+ players:** No, use FULL averaging instead (empirically better)

**For 2 players:** Possibly, but FULL also works well

### Why is Nash convergence non-monotonic?

DCFR uses regret discounting, which can temporarily increase exploitability as the algorithm "forgets" old information. This is expected behavior - focus on best Nash over time, not final Nash.

### Can I use DCFR on large games?

Only with C++ implementations and even then, full Hold'em is impractical. Use:
- Card abstractions (fewer suits/ranks)
- Action abstractions (FCPA, FCHPA)
- Outcome Sampling MCCFR (faster for large games)
- Or stick with smaller games like Kuhn for algorithm testing

---

## References

- Brown & Sandholm (2019). "Solving Imperfect-Information Games via Discounted Regret Minimization". AAAI 2019.
- OpenSpiel external_sampling_mccfr.py implementation
- [DCFR_BUGS_AND_FIXES.md](DCFR_BUGS_AND_FIXES.md) - Implementation bugs and fixes
- Validation results: `results/dcfr_research_validation_*.csv`

---

## Summary

**For your GTO poker training application (3+ players):**

✅ **Use FULL averaging** - Best empirical performance, theoretically sound

❌ **Don't use DCFR variants** - Underperform on 3-player games

✅ **Use C++ implementations** - 100-1000x faster for production

✅ **Test on Kuhn poker** - Validate algorithms before scaling
