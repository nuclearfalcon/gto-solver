# Solver Selection Guide

**Which poker solver should you use?**

This guide helps you choose between the **GPU MCCFR track** and the **OpenSpiel track** based on your specific use case.

---

## Quick Decision Tree

```
START: What's your goal?
    │
    ├─ Training GTO policy for large Hold'em game (2-9 players, 10BB+ stacks)?
    │   │
    │   ├─ Do you have NVIDIA GPU available?
    │   │   │
    │   │   ├─ YES → ✅ Use GPU MCCFR
    │   │   │        python solve_poker_gpu.py --config configs/gpu/2p_10bb_holdem.json --iterations 1000
    │   │   │
    │   │   └─ NO → ⚠️ Use OpenSpiel (slower, more RAM)
    │   │            python solve_poker.py --config configs/2p_10bb_fcpa.json --algorithm external_mccfr
    │   │
    │   └─ (If RAM constrained: GPU MCCFR uses 500MB vs 10-20GB for OpenSpiel)
    │
    ├─ Exact Nash equilibrium required (research, validation, theoretical analysis)?
    │   └─ ✅ Use OpenSpiel (tabular CFR+)
    │       python solve_poker.py --config configs/2p_kuhn.json --algorithm cfr_plus
    │
    ├─ Small game testing (Kuhn, Leduc, toy Hold'em)?
    │   └─ ✅ Use OpenSpiel (simpler, no GPU needed)
    │       python solve_poker.py --config configs/kuhn_poker.json --algorithm cfr_plus
    │
    ├─ Exploitability metric calculation?
    │   │
    │   ├─ Small game (<10^5 infosets)?
    │   │   └─ ✅ Use OpenSpiel (exact exploitability)
    │   │
    │   └─ Large game?
    │       └─ ✅ Use sampled exploitability (both tracks support this)
    │           from exploitability_metrics import SampledExploitabilityCalculator
    │
    └─ Comparing CFR algorithm variants (CFR, CFR+, DCFR, LCFR)?
        └─ ✅ Use OpenSpiel (supports 8 algorithms)
            python solve_and_compare.py --config configs/2p_kuhn.json --iterations 10000
```

---

## Detailed Comparison

### Performance Matrix

| Criterion | GPU MCCFR | OpenSpiel |
|-----------|-----------|-----------|
| **Speed (2p 10BB)** | 100+ it/s | 0.02 it/s |
| **RAM Usage** | 500 MB | 10-20 GB |
| **GPU Required** | Yes (NVIDIA CUDA) | No |
| **Max Game Size** | 10^14+ infosets (bucketed) | ~10^5 infosets (exact) |
| **Solution Quality** | ~95% (bucketed) | 100% (exact) |
| **Algorithms** | MCCFR only | 8 variants (CFR, CFR+, DCFR, etc.) |
| **Exploitability** | Sampled only | Exact or sampled |
| **Learning Curve** | Medium (JAX, bucketing) | Low (standard CFR) |

### Use Case Recommendations

| Use Case | Recommended Solver | Why |
|----------|-------------------|-----|
| **Training 2-9 player Hold'em** | 🟢 GPU MCCFR | 100-1000× faster, 20-40× less RAM |
| **Deep stack Hold'em (20BB+)** | 🟢 GPU MCCFR | OpenSpiel runs out of memory |
| **Kuhn/Leduc poker** | 🟢 OpenSpiel | Simple, exact solution, no GPU needed |
| **Algorithm comparison** | 🟢 OpenSpiel | Supports 8 CFR variants |
| **Exact Nash equilibrium** | 🟢 OpenSpiel | GPU MCCFR uses bucketing (~5% error) |
| **CPU-only environment** | 🟢 OpenSpiel | GPU MCCFR requires NVIDIA GPU |
| **Research paper replication** | 🟢 OpenSpiel | Standard reference implementation |
| **Production poker bot** | 🟢 GPU MCCFR | Scales to realistic game sizes |

---

## Workflow Examples

### Example 1: Training a Heads-Up 10BB Bot

**Goal:** Train GTO policy for 2-player 10BB No-Limit Hold'em

**Recommended:** GPU MCCFR

**Workflow:**
```bash
# 1. Activate environment
source ~/open_spiel/venv/bin/activate

# 2. Train policy (1M iterations, ~3 hours on RTX 3060)
python solve_poker_gpu.py \
    --config configs/gpu/2p_10bb_holdem.json \
    --iterations 1000000 \
    --batch-size 100

# 3. Outputs:
# - results/gpu_mccfr_2p_10bb_policy.pkl (average policy)
# - results/gpu_mccfr_2p_10bb_metrics.json (exploitability over time)

# 4. Query policy
python query_gpu_policy.py --policy results/gpu_mccfr_2p_10bb_policy.pkl
```

**Alternative (OpenSpiel):**
- Would require 10-20 GB RAM
- ~500× slower (weeks instead of hours)
- Only feasible for tiny abstractions (FCPA with few states)

---

### Example 2: Validating CFR+ Convergence on Kuhn Poker

**Goal:** Verify CFR+ converges to Nash equilibrium on Kuhn poker

**Recommended:** OpenSpiel

**Workflow:**
```bash
# 1. Activate environment
source ~/open_spiel/venv/bin/activate

# 2. Solve with CFR+
python solve_poker.py \
    --config configs/kuhn_poker.json \
    --algorithm cfr_plus \
    --iterations 100000

# 3. Calculate exact exploitability (Kuhn has only 12 infosets)
# Outputs:
# - Exploitability: 0.0001 (near-perfect Nash)

# 4. Compare with other algorithms
python solve_and_compare.py \
    --config configs/kuhn_poker.json \
    --iterations 100000

# Compares: vanilla_cfr, cfr_plus, dcfr, lcfr, external_mccfr, outcome_mccfr
```

**Why not GPU MCCFR?**
- Overkill for 12 infosets
- Bucketing would lose accuracy unnecessarily
- OpenSpiel gives exact solution in seconds

---

### Example 3: Training 6-max Tournament Poker (10BB)

**Goal:** Train GTO strategy for 6-player tournament (10BB effective stacks)

**Recommended:** GPU MCCFR

**Workflow:**
```bash
# 1. Create custom config
cat > configs/gpu/6p_10bb_tournament.json <<EOF
{
  "num_players": 6,
  "stacks": [1000, 1000, 1000, 1000, 1000, 1000],
  "blinds": [50, 100, 0, 0, 0, 0],
  "batch_size": 100,
  "num_buckets": 20000,
  "num_hand_buckets": 300,
  "num_pot_buckets": 15,
  "num_actions": 4,
  "seed": 42,
  "name": "6p_10bb_tournament",
  "description": "6-max tournament 10BB"
}
EOF

# 2. Train (5M iterations, ~24 hours on RTX 3060)
python solve_poker_gpu.py \
    --config configs/gpu/6p_10bb_tournament.json \
    --iterations 5000000 \
    --batch-size 200 \
    --checkpoint-interval 100000

# 3. Monitor convergence
tail -f results/gpu_mccfr_6p_10bb_tournament_metrics.json
```

**Why not OpenSpiel?**
- 6-player Hold'em has ~10^8 infosets (exact CFR impossible)
- Would require >100 GB RAM
- GPU MCCFR uses ~650 MB RAM, 2 MB VRAM

---

### Example 4: Computing Exploitability for Trained Policy

**Goal:** Measure how far a trained policy is from Nash equilibrium

**Small game (<10^5 infosets):**
```bash
# Use OpenSpiel's exact exploitability
python solve_poker.py \
    --config configs/2p_kuhn.json \
    --algorithm cfr_plus \
    --iterations 100000

# Exploitability computed automatically during solving
```

**Large game (>10^6 infosets):**
```bash
# Use sampled exploitability (works with both tracks)
from exploitability_metrics import SampledExploitabilityCalculator

calc = SampledExploitabilityCalculator(game, policy)
result = calc.calculate(
    confidence_level=0.99,
    max_ci_width=0.01,  # 1% CI width
    min_samples=500,
    max_samples=5000
)

print(f"Exploitability: {result['exploitability']:.4f}")
print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
```

---

## Migration Guide

### Moving from OpenSpiel to GPU MCCFR

**When to migrate:**
- OpenSpiel CFR runs out of memory (>10 GB)
- Training takes too long (days/weeks)
- You acquire GPU hardware

**Migration steps:**

1. **Create GPU config from OpenSpiel config:**
```python
from game_config import PokerGameConfig
from gpu_mccfr_config import GPUMCCFRConfig

# Load OpenSpiel config
openspiel_config = PokerGameConfig.from_json("configs/2p_10bb_fcpa.json")

# Convert to GPU config
gpu_config = GPUMCCFRConfig(
    num_players=openspiel_config.num_players,
    stacks=openspiel_config.stack_sizes,
    blinds=openspiel_config.blinds,
    batch_size=100,  # New parameter
    num_buckets=10000,  # Adjust based on game size
    num_hand_buckets=200,
    num_pot_buckets=10,
    num_actions=4,
    seed=42
)

gpu_config.to_json("configs/gpu/2p_10bb_holdem.json")
```

2. **Run GPU training:**
```bash
python solve_poker_gpu.py \
    --config configs/gpu/2p_10bb_holdem.json \
    --iterations 1000000
```

3. **Validate solution quality:**
```python
# Compare OpenSpiel policy vs GPU MCCFR policy on small test cases
# Expect ~95% agreement on strategy
```

**Expected differences:**
- GPU MCCFR uses bucketing (~5% solution quality loss)
- Cannot query exact infosets (only bucket-level)
- Faster convergence in wall-clock time (100-1000×)

---

## Hardware Requirements

### GPU MCCFR

**Minimum:**
- NVIDIA GPU (GTX 1060 6GB or better)
- 2 GB GPU VRAM
- 2 GB system RAM
- CUDA 11.0+

**Recommended:**
- NVIDIA RTX 3060 or better
- 8+ GB GPU VRAM
- 8 GB system RAM
- CUDA 11.8+

**Optimal:**
- NVIDIA RTX 4090
- 24 GB GPU VRAM
- 32 GB system RAM
- CUDA 12.0+

### OpenSpiel

**Minimum:**
- CPU only (no GPU needed)
- 4 GB RAM (for small games)

**Recommended:**
- Multi-core CPU (CFR is CPU-only, but can parallelize exploitability)
- 16+ GB RAM (for medium games)

**For large games:**
- 64+ GB RAM (still may OOM on full Hold'em)

---

## Troubleshooting Decision Issues

### "Which solver for my custom game?"

**Ask yourself:**

1. **How many infosets does the game have?**
   - <10^5: OpenSpiel (exact)
   - >10^6: GPU MCCFR (bucketed)

2. **Do I have GPU?**
   - Yes: GPU MCCFR (if game is large)
   - No: OpenSpiel (only option)

3. **Do I need exact Nash?**
   - Yes: OpenSpiel (no bucketing)
   - No (~95% okay): GPU MCCFR (faster)

4. **How much RAM available?**
   - <2 GB: GPU MCCFR only option
   - 2-8 GB: GPU MCCFR recommended
   - 16+ GB: Both viable

### "My OpenSpiel solver runs out of memory!"

**Solutions (in order of preference):**

1. **Switch to GPU MCCFR** (20-40× less RAM)
```bash
python solve_poker_gpu.py --config configs/gpu/2p_10bb_holdem.json --iterations 100000
```

2. **Use sampled exploitability** (instead of exact)
```python
solver.solve(max_iterations=100000, use_sampled_exploitability=True)
```

3. **Reduce game size** (smaller stacks, fewer players)
```python
config = PokerGameConfig(
    num_players=2,
    stack_sizes=[500, 500],  # 5BB instead of 10BB
    blinds=[50, 100],
    betting_abstraction='fc'  # Fold/call only (simplest)
)
```

### "My GPU MCCFR trains too slowly!"

**Solutions:**

1. **Increase batch_size** (if VRAM available)
```bash
python solve_poker_gpu.py --config ... --batch-size 500  # Default: 100
```

2. **Check GPU utilization**
```bash
nvidia-smi  # Should show >80% GPU usage
```

3. **Verify JAX is using GPU**
```python
import jax
print(jax.devices())  # Should show 'gpu:0'
```

4. **Reduce num_buckets** (if convergence okay)
```python
# 5K buckets instead of 10K (2× less memory, slightly faster)
config.num_buckets = 5000
config.num_hand_buckets = 150
config.num_pot_buckets = 8
```

---

## Summary Table

| Solver | Best For | Speed | RAM | Accuracy | GPU Required |
|--------|----------|-------|-----|----------|--------------|
| **GPU MCCFR** | Large games, production | 100+ it/s | 500 MB | ~95% | ✅ Yes |
| **OpenSpiel CFR** | Small games, research | 0.02-50 it/s | 0.1-20 GB | 100% | ❌ No |

**General rule:**
- **Game size <10^5 infosets:** Use OpenSpiel
- **Game size >10^6 infosets:** Use GPU MCCFR (if GPU available)
- **Exact solution required:** Use OpenSpiel
- **Production deployment:** Use GPU MCCFR

---

## Further Reading

- `CLAUDE.md` - Quick start guide for both tracks
- `GPU_MCCFR_GUIDE.md` - Technical deep dive into GPU MCCFR
- `ARCHITECTURE.md` - SSOT documentation for all components
- `PHASE10_COMPLETE_SUMMARY.md` - Development history and benchmarks
- `archive/README.md` - Lessons learned from Phases 2-9

**Questions?**
- Check `CLAUDE.md` for common commands
- See `GPU_MCCFR_GUIDE.md` troubleshooting section
- Review benchmark data in `PHASE10_COMPLETE_SUMMARY.md`
