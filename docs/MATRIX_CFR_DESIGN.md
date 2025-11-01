# Matrix-Based GPU CFR Design Document

**Project**: GPU-Accelerated CFR for 3-Player Hold'em
**Goal**: Enable solving 3-player No-Limit Hold'em using matrix-based CFR on GPU
**Hardware**: NVIDIA GeForce RTX 4060 Ti (16GB VRAM)
**Based on**: arXiv:2408.14778v5 "GPU-Accelerated CFR"
**Date**: November 2025

---

## Executive Summary

This document outlines the design for a matrix-based GPU CFR implementation that transforms recursive tree traversal into level-by-level matrix operations. The goal is to achieve 50-200x speedup on 3-player Hold'em, enabling solutions that are infeasible on CPU.

**Key Innovation**: Process game tree by depth levels using sparse adjacency matrices instead of recursive traversal, enabling GPU parallelization across thousands of nodes simultaneously.

---

## 1. Core Algorithm

### 1.1 Level-by-Level Processing

Instead of depth-first recursive traversal:
```python
# Traditional CFR (recursive)
def cfr(state, player, reach_probs):
    if state.is_terminal():
        return state.returns()[player]
    if state.current_player() != player:
        # Traverse all opponent actions
        ...
    # Recursive calls for each action
```

We use breadth-first level processing:
```python
# Matrix CFR (level-by-level)
for level in range(max_depth, 0, -1):
    # Process ALL nodes at this level in parallel
    utilities[level] = level_matrix[level] @ utilities[level+1]
```

### 1.2 Three Phases per Iteration

Each CFR iteration consists of three matrix operation phases:

#### Phase 1: Tree Traversal (Utilities & Reach Probabilities)

**Bottom-up utility propagation** (Equation 11 from paper):
```
Ǔ^(D+1) = terminal_utilities
for l = D down to 1:
    Ǔ^(l) = (L^l ⊙ S) Ǔ^(l+1) + Ǔ^(l+1)
```

Where:
- `L^l`: Sparse level matrix (parent-child adjacency at depth l)
- `S`: Strategy vector broadcast to nodes
- `⊙`: Element-wise product
- In-place addition ensures each node visited once

**Top-down reach probability propagation** (Equation 13):
```
Π̌^(0) = [1, 0, 0, ...] (1 at root only)
for l = 1 to D:
    Π̌^(l) = ((L^l)^T Π̌^(l-1)) ⊙ Š + Π̌^(l-1)
```

Where `Š` sets probability to 1 when player acted (counterfactual).

#### Phase 2: Strategy Averaging

Weighted average by reach probability (Equation 10):
```
σ̄^(T) = (M^(Q+,V)^T (Π̄ ⊙ σ^(T))) ⊘ (M^(Q+,V)^T Π̄)
```

#### Phase 3: Regret Update

1. Compute counterfactual utility per action (override strategy)
2. Calculate instantaneous regret: `r̃ = π̃(h) × (ũ_override − ũ_current)`
3. Accumulate: `r̄^(T) += r̃`
4. Regret matching: `σ^(T+1) = max(r̄, 0) / Σ max(r̄, 0)`

---

## 2. Matrix Representation

### 2.1 Required Matrices

| Matrix | Dimensions | Type | Description |
|--------|-----------|------|-------------|
| **L^(l)** | \|V\| × \|V\| | Sparse CSR | Level-l parent-child adjacency |
| **M^(Q+,V)** | \|Q+\| × \|V\| | Sparse CSR | Infoset-action to nodes |
| **M^(H+,Q+)** | \|H+\| × \|Q+\| | Sparse CSR | Infoset to actions |
| **M^(V,I+)** | \|V\| × \|I+\| | Dense | Which player acted at node |

Where:
- `|V|`: Total nodes in game tree
- `|Q+|`: Number of (infoset, action) pairs
- `|H+|`: Number of infosets
- `|I+|`: Number of players

### 2.2 Vector Representations

| Vector | Dimensions | Description |
|--------|-----------|-------------|
| **σ** | \|Q+\| | Current strategy probabilities |
| **r̄** | \|Q+\| | Cumulative regrets |
| **σ̄** | \|Q+\| | Average strategy |
| **Ǔ** | \|V\| | Node utilities |
| **Π̌** | \|V\| | Counterfactual reach probabilities |

### 2.3 Sparsity Analysis

**Critical insight from paper**: All matrices except M^(V,I+) are highly sparse.

For poker games:
- **Level matrices L^l**: ~0.1-1% density (each node has 2-5 children)
- **Infoset mappings**: ~0.01% density (information hiding creates many infosets)
- **Player matrix M^(V,I+)**: Dense (every node has a player)

**Memory optimization**: Use CSR (Compressed Sparse Row) format for all sparse matrices.

---

## 3. Game Tree Construction

### 3.1 Tree Traversal Algorithm

```python
def build_game_matrices(game):
    # Step 1: Full tree enumeration
    nodes = []
    infosets = {}

    def traverse(state, depth, parent_idx):
        node_idx = len(nodes)
        nodes.append({
            'state': state,
            'depth': depth,
            'parent': parent_idx,
            'player': state.current_player(),
            'infoset': state.information_state_string() if not terminal
        })

        if state.is_terminal():
            return

        for action in state.legal_actions():
            child_state = state.child(action)
            traverse(child_state, depth+1, node_idx)

    traverse(game.new_initial_state(), 0, -1)

    # Step 2: Build matrices from traversal
    return construct_matrices(nodes, infosets)
```

### 3.2 Matrix Construction

```python
def construct_matrices(nodes, infosets):
    max_depth = max(n['depth'] for n in nodes)

    # Level matrices (one per depth)
    level_matrices = []
    for l in range(max_depth + 1):
        L_l = build_level_adjacency(nodes, l)
        level_matrices.append(sparse.csr_matrix(L_l))

    # Infoset-to-node mapping
    M_qv = build_infoset_action_map(nodes, infosets)

    # Player matrix (dense)
    M_vi = build_player_matrix(nodes, num_players)

    return {
        'level_matrices': level_matrices,
        'infoset_action_map': M_qv,
        'player_matrix': M_vi,
        'num_nodes': len(nodes),
        'num_infosets': len(infosets)
    }
```

---

## 4. Implementation Strategy

### 4.1 Framework: JAX

**Choice: JAX with CUDA backend**

Reasons:
- Best balance of flexibility and performance
- Native sparse matrix support (jax.experimental.sparse)
- JIT compilation for kernel fusion
- Familiar NumPy-like API
- Excellent GPU memory management

Alternatives considered:
- **PyTorch**: More mature but worse sparse support
- **CuPy**: Lower-level, harder to optimize
- **Raw CUDA**: Too much development time

### 4.2 Development Phases

**Phase 1: Kuhn Poker Prototype (Week 2-3)**
- Small game tree (~12 nodes)
- Validate matrix construction
- Implement vanilla CFR only
- Verify exact match with CPU

**Phase 2: Scaling to Leduc (Week 4-5)**
- Medium game tree (~1000 nodes)
- Add MCCFR (sampling)
- Add DCFR (discounting)
- Memory optimization

**Phase 3: Hold'em Implementation (Week 6-8)**
- Large game tree (millions of nodes)
- Chunking by betting round
- FP16 mixed precision
- Exploit RTX 4060 Ti fully

---

## 5. Memory Management

### 5.1 VRAM Budget (16GB RTX 4060 Ti)

| Component | Allocation | Notes |
|-----------|------------|-------|
| System overhead | 2 GB | CUDA runtime, driver |
| Matrix storage | 4-6 GB | Sparse matrices (CSR) |
| Vector storage | 2-4 GB | Strategies, regrets, utilities |
| Working memory | 4-6 GB | Intermediate computations |
| Safety margin | 2 GB | Prevent OOM crashes |

**Target**: Keep total usage < 14GB for safety.

### 5.2 Optimization Techniques

1. **Sparse CSR matrices**: 100-1000x less memory than dense
2. **Mixed precision (FP16)**: Halve vector memory usage
3. **Gradient checkpointing**: Recompute instead of store intermediates
4. **Batch by betting round**: Process preflop/flop/turn/river separately
5. **Lazy matrix construction**: Build level matrices on-demand

### 5.3 Memory Estimation

For 3-player Hold'em (5bb stacks, FCPA):
- Estimated nodes: ~10-50 million
- Sparse matrix memory: ~2-8 GB
- Vector memory (FP32): ~1-4 GB
- Vector memory (FP16): ~0.5-2 GB
- **Total: 4-14 GB** ✅ Fits on 16GB card

---

## 6. Extension to MCCFR and DCFR

### 6.1 MCCFR (Monte Carlo CFR)

**Challenge**: Paper only covers full tree traversal.

**Solution**: Sparse matrix masking

Instead of processing all nodes at level l:
```python
# Sample one action at chance nodes
sampled_nodes = sample_chance_outcomes(level_l_nodes)
mask = create_sparse_mask(sampled_nodes)

# Masked matrix multiply
Ǔ^(l) = (mask ⊙ L^l ⊙ S) Ǔ^(l+1)
```

This drastically reduces computation (only traverse one path instead of all paths).

### 6.2 DCFR (Discounted CFR)

**Challenge**: Paper doesn't cover regret discounting.

**Solution**: Vector scaling

After regret update (Phase 3):
```python
# Discount positive regrets
r̄_positive *= iteration^(-alpha) where r̄ > 0

# Discount negative regrets
r̄_negative *= iteration^(-beta) where r̄ < 0

# Weighted averaging (gamma parameter)
weight = iteration^gamma
σ̄ = (σ̄ * total_weight + σ * weight) / (total_weight + weight)
```

These are simple vector operations, easily parallelizable.

---

## 7. Integration with Existing Codebase

### 7.1 UnifiedPokerSolver Interface

Add to `poker_solver.py`:
```python
class UnifiedPokerSolver:
    def __init__(self, config, algorithm='cfr_plus'):
        if algorithm.startswith('matrix_'):
            # Use GPU solver
            from matrix_cfr import MatrixCFRSolver
            self.solver = MatrixCFRSolver(...)
        else:
            # Use CPU solver
            ...
```

### 7.2 New Algorithms

- `'matrix_vanilla_cfr'`: Vanilla CFR on GPU
- `'matrix_mccfr'`: MCCFR with external sampling on GPU
- `'matrix_dcfr'`: DCFR with custom (α, β, γ) on GPU
- `'matrix_auto'`: Auto-select best variant for game size

### 7.3 Backward Compatibility

All existing scripts (`solve_poker.py`, etc.) work unchanged:
```bash
# Automatically uses GPU if available
python solve_poker.py --config configs/3p_5bb_fcpa.json --algorithm matrix_mccfr --iterations 100000
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

1. **Matrix construction** (`test_matrix_construction.py`):
   - Verify dimensions
   - Check sparsity patterns
   - Validate parent-child relationships

2. **Phase operations** (`test_cfr_phases.py`):
   - Test utility propagation (Phase 1)
   - Test strategy averaging (Phase 2)
   - Test regret updates (Phase 3)

3. **Memory management** (`test_gpu_memory.py`):
   - Monitor VRAM usage
   - Detect memory leaks
   - Test chunking

### 8.2 Integration Tests

1. **Kuhn poker** (`test_kuhn_gpu.py`):
   - Exact match with CPU vanilla_cfr
   - Convergence to Nash equilibrium
   - Exploitability < 0.01 at 10k iterations

2. **Leduc poker** (`test_leduc_gpu.py`):
   - Compare GPU vs CPU MCCFR
   - Validate DCFR variants
   - Performance benchmark (>20x speedup target)

3. **Hold'em** (`test_holdem_gpu.py`):
   - Small stacks (2bb): Full solve
   - Medium stacks (5bb): Partial solve
   - Verify VRAM usage < 14GB

### 8.3 Validation Against CPU

For each game and algorithm:
```python
# Run both solvers
gpu_solver = MatrixCFRSolver(game, 'vanilla_cfr')
cpu_solver = UnifiedPokerSolver(config, 'vanilla_cfr')

gpu_solver.solve(10000)
cpu_solver.solve(10000)

# Compare policies
validate_policies(gpu_solver.get_average_policy(),
                 cpu_solver.get_average_policy(),
                 tolerance=1e-6)
```

---

## 9. Performance Targets

### 9.1 Expected Speedups

| Game | Nodes | CPU (it/s) | GPU Target (it/s) | Speedup |
|------|-------|------------|-------------------|---------|
| Kuhn poker | ~12 | 50,000 | 200,000 | 4x |
| Leduc poker | ~1,000 | 500 | 10,000 | 20x |
| Hold'em 2p 2bb | ~100k | 10 | 500 | 50x |
| Hold'em 3p 2bb | ~500k | 2 | 200 | 100x |
| Hold'em 3p 5bb | ~10M | 0.1 | 20 | 200x |

**Note**: Small games (Kuhn) have overhead; large games (Hold'em) see massive gains.

### 9.2 Success Criteria

**Minimum Viable Product** (end of Week 3):
- ✅ Kuhn poker: Exact match with CPU
- ✅ Any measurable speedup (even 2x is progress)
- ✅ Clean test suite (all passing)

**Target Product** (end of Week 8):
- ✅ Hold'em 3p 5bb: Can complete 100k iterations in < 1 hour
- ✅ VRAM usage: < 12GB (safe margin)
- ✅ Exploitability convergence matches CPU behavior
- ✅ All DCFR variants working

**Stretch Goals**:
- ✅ Hold'em 3p 10bb: Solvable in < 24 hours
- ✅ 100x+ speedup on Hold'em
- ✅ Multi-GPU support

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| VRAM overflow on Hold'em | High | Critical | Implement chunking early; use FP16 |
| Slow sparse matmul | Medium | High | Profile and optimize; try CuSPARSE directly |
| Numerical instability | Medium | Medium | Use FP64 for testing; add clipping |
| JAX bugs/limitations | Low | High | Have CuPy backup plan |

### 10.2 Contingency Plans

**If VRAM is insufficient**:
1. Use FP16 everywhere (halves usage)
2. Process betting rounds separately
3. Implement gradient checkpointing
4. Consider cloud GPU (A100 80GB)

**If speedup is disappointing (<10x)**:
1. Profile to find bottleneck
2. Try different sparse matrix formats (COO, ELL)
3. Optimize matrix construction (do once, reuse)
4. Consider hybrid CPU/GPU (CPU for small games)

**If Hold'em tree is too large**:
1. Start with deeper abstractions (fc instead of fcpa)
2. Use smaller stacks (1-2bb)
3. Build 2-player first, then 3-player
4. Consider bucketing/clustering techniques

---

## 11. Implementation Roadmap

### Week 1 (Current)
- [x] Branch created
- [x] Project structure
- [x] Research paper analysis
- [x] Design document
- [ ] JAX installation
- [ ] GPU validation

### Week 2-3: Kuhn Poker Prototype
- [ ] Game tree to matrix converter (Kuhn)
- [ ] Vanilla CFR matrix implementation
- [ ] Test suite
- [ ] Validation against CPU
- [ ] Benchmark

### Week 4-5: Scaling & Extensions
- [ ] Leduc poker support
- [ ] MCCFR sampling
- [ ] DCFR discounting
- [ ] Memory optimization
- [ ] Performance profiling

### Week 6-8: Hold'em Implementation
- [ ] Hold'em matrix construction
- [ ] Incremental scaling tests (2bb → 5bb)
- [ ] 3-player implementation
- [ ] Production integration
- [ ] Comprehensive benchmarks

### Week 9-10: Polish
- [ ] Documentation
- [ ] API cleanup
- [ ] UnifiedPokerSolver integration
- [ ] Final validation
- [ ] Performance tuning

---

## 12. Open Questions

1. **Chance node handling**: Should we build full matrices including chance nodes, or skip them and sample in the traversal?
   - **Decision**: Build full matrices, add sampling mask for MCCFR

2. **Level matrix storage**: Store all L^l matrices, or recompute on-demand?
   - **Decision**: Store (better performance), but recompute if VRAM tight

3. **Multi-GPU**: Worth implementing for 2-GPU setups?
   - **Decision**: Defer to stretch goals; single GPU first

4. **FP16 vs FP32**: Use mixed precision from start, or only when needed?
   - **Decision**: Start with FP32, add FP16 in Week 4

---

## 13. References

- **Primary**: arXiv:2408.14778v5 "GPU-Accelerated CFR"
- OpenSpiel documentation: `/home/nuclearfalcon/open_spiel/`
- JAX documentation: https://jax.readthedocs.io/
- Existing codebase: `ARCHITECTURE.md`, `DCFR_GUIDE.md`

---

## Conclusion

This design provides a clear path to implementing matrix-based GPU CFR for 3-player Hold'em. By following the level-by-level processing approach from the research paper and carefully managing VRAM constraints on the RTX 4060 Ti, we can achieve 50-200x speedups that make previously infeasible solves practical.

The phased approach (Kuhn → Leduc → Hold'em) ensures we validate correctness on small games before tackling the complexity of full Hold'em.

**Next steps**: Install JAX, validate GPU setup, begin Kuhn poker prototype.
