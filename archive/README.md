# Archived Experimental Tests

This directory contains historical test files from experimental development phases (Phases 2-9). These experiments led to the production GPU MCCFR implementation (Phase 10+), but are no longer actively maintained.

**Current Production Code:** Phase 10.5+ tests remain in the root directory and in `tests/`.

---

## Phase Timeline

### Phase 2: Early Learning Experiments (Feb 2025)
**Goal:** Explore basic matrix CFR learning dynamics

**Files:**
- `test_phase2_learning.py` - Basic learning tests
- `test_phase2_1_regret_matching.py` - Regret matching validation
- `test_phase2_1_iteration_speed.py` - Early speed benchmarks
- `test_phase2_2_batched_values.py` - Batched value computation

**Results:**
- Established baseline regret matching correctness
- Identified iteration speed as key bottleneck (~0.14 it/s for Leduc)

**Status:** Superseded by Phase 10 GPU implementations

---

### Phase 3: Ablation Studies (Feb 2025)
**Goal:** Isolate performance factors in matrix CFR

**Files:**
- `test_phase3_ablation.py` - Component ablation tests
- `test_phase3_ablation_robust.py` - Robust ablation validation
- `test_phase3_final_benchmark.py` - Final baseline benchmarks

**Results:**
- Scatter operations identified as major bottleneck
- Established 0.36 it/s as scatter-optimized baseline

**Status:** Superseded by sparse matrix approach (Phase 5+)

---

### Phase 4: Pre-GPU Benchmarks (Feb 2025)
**Goal:** Establish baseline performance for comparison

**Files:**
- `test_phase4_kuhn_benchmark.py` - Kuhn poker baseline

**Results:**
- Kuhn: ~10 it/s with dense matrices
- Baseline for future GPU comparisons

**Status:** Superseded by Phase 10 GPU benchmarks

---

### Phase 5: Sparse Matrix Experiments (Feb 2025)
**Goal:** Reduce memory usage with sparse representations

**Files:**
- `test_phase5_sparse_kuhn_quick.py` - Sparse Kuhn validation
- `test_phase5_leduc_memory.py` - Leduc memory profiling
- `test_phase5_bcoo_conversion.py` - JAX BCOO format tests

**Results:**
- ✅ Reduced memory from dense O(all_states) to sparse O(visited_states)
- ⚠️ Slower iteration speed (~0.25 it/s, 30% regression)
- Sparse updates more complex than dense

**Key Insight:** Memory efficiency at cost of speed

**Status:** Partially integrated into Phase 10 (GPURegretTable uses sparse-like dict storage on CPU)

---

### Phase 7: OOM Fixes & Preflop Experiments (Feb 2025)
**Goal:** Fix out-of-memory errors for large games

**Files:**
- `test_phase7_oom_fix.py` - OOM mitigation strategies
- `test_phase7_sparse_fix.py` - Sparse implementation fixes
- `test_phase7_preflop_minimal.py` - Preflop-only solver
- `test_phase7_final_validation.py` - Validation suite

**Results:**
- Identified full exploitability calculation as memory killer (10-20 GB for full Hold'em)
- Led to sampled exploitability implementation (now DEFAULT)
- Preflop-only solver: memory-efficient but limited

**Key Contribution:** Sampled exploitability now production feature

**Status:** Sampled exploitability extracted to production (`exploitability_metrics.py`)

---

### Phase 8: Chunking & Batching Experiments (Feb-Mar 2025)
**Goal:** Process large games in memory-efficient chunks

**Files:**
- `test_phase8_batch_ops.py` - Batched operation tests
- `test_phase8_chunking.py` - Chunk-based processing
- `test_phase8_memory_profiling.py` - Memory profiling tools
- `test_phase8.5_minimal.py` - Minimal chunked solver
- `test_phase8.5_three_chunk.py` - Three-chunk validation
- `test_phase8.5_full_pipeline.py` - Full chunked pipeline

**Results:**
- ✅ Reduced peak memory usage
- ⚠️ Complex coordination logic
- ⚠️ Still too slow for practical use

**Status:** Superseded by Phase 10 bucketing approach (simpler, faster)

---

### Phase 9: Pre-dealing Experiments (Mar 2025)
**Goal:** Pre-deal cards to reduce game tree size

**Files:**
- `test_phase9_true_predealing.py` - Card pre-dealing validation

**Results:**
- ⚠️ Reduces game size but changes equilibrium
- Not suitable for GTO training (alters information structure)

**Status:** Abandoned (theoretical issues)

---

## Why These Were Superseded

**Phase 10 (Current)** introduced a fundamentally different approach:

1. **GPU-Accelerated MCCFR:** JAX-based game engines compiled to XLA
2. **Bucketing:** State abstraction (10K buckets) instead of exact states
3. **Batched Sampling:** 100+ trajectories per iteration
4. **GPU-Resident Regrets:** Dense tensors on GPU, no Python dict overhead

**Result:**
- **Speed:** 0.100 it/s actual (target 1-10 it/s with optimizations)
- **Memory:** ~1 GB RAM, ~105 MB VRAM (vs 10-20 GB for OpenSpiel)
- **Scalability:** Can handle full Hold'em games

**Key Paradigm Shift:** Exact states → Bucketed abstraction

---

## Lessons Learned

### What Worked
- ✅ Sparse storage reduces memory (Phase 5)
- ✅ Sampled exploitability avoids OOM (Phase 7)
- ✅ JAX + GPU >>> pure Python (Phase 10)
- ✅ Bucketing enables large games (Phase 10)

### What Didn't Work
- ❌ Pure chunking too complex (Phase 8)
- ❌ Pre-dealing breaks GTO (Phase 9)
- ❌ Dense matrices don't scale (Phase 2-4)

### Critical Insights
1. **Memory vs Speed tradeoff:** Sparse is slower but necessary for large games
2. **Exploitability is expensive:** Full calculation requires massive memory
3. **Python loops are death:** Must compile to GPU kernels (JAX/XLA)
4. **Abstraction is key:** Can't store exact states for full Hold'em

---

## If You Need These Files

All archived tests are preserved for:
- Historical reference
- Algorithm comparison
- Research validation
- Understanding design decisions

**To run an archived test:**
```bash
source ~/open_spiel/venv/bin/activate
python archive/phase5/test_phase5_leduc_memory.py
```

**Note:** Some tests may fail due to API changes in production code.

---

## Current Production Tests

**In root directory:**
- `test_poker_configs.py` - Game configuration validation
- `test_tensor_bet_sizes.py` - Tensor structure validation
- `test_sampled_exploitability.py` - Exploitability validation
- `test_memory_fix.py` - Memory leak detection
- `test_phase10-5_*.py` - Current GPU MCCFR tests

**In tests/ directory:**
- `test_phase10_holdem.py` - Full Hold'em validation (17/17 passing)
- `test_kuhn_mccfr.py` - Kuhn poker validation
- `test_holdem_mccfr.py` - Scalability tests

---

**Archive Created:** 2025-02-04

**Production Solver:** `solve_poker_gpu.py` (Phase 10.5+)
