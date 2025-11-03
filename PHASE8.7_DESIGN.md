# Phase 8.7 Design: Hierarchical Sub-Chunking

**Date:** 2025-01-03
**Goal:** Enable solving arbitrarily large chunks by automatic sub-division
**Status:** 🎯 Design Phase

---

## Problem Statement

From Phase 8.6 stress testing:
- **Working limit**: ~1,600 nodes per chunk
- **Turn chunk size**: 57,521 nodes (36× too large)
- **Root cause**: Memory fragmentation, not allocation size
- **Solution needed**: Split large chunks into smaller sub-chunks

---

## Architecture Design

### Current Hierarchy (Phase 8.5)

```
FullGame
  └─ ChunkedSolver
      ├─ PreflContent chunk (SubgameSolver)
      ├─ Flop chunk (SubgameSolver)
      ├─ Turn chunk (SubgameSolver) ← 57k nodes, OOM
      └─ River chunk (SubgameSolver) ← 200k+ nodes, OOM
```

### Proposed Hierarchy (Phase 8.7)

```
FullGame
  └─ ChunkedSolver
      ├─ Preflop chunk (SubgameSolver)
      ├─ Flop chunk (SubgameSolver)
      ├─ Turn chunk (SubgameSolver with auto-split)
      │   ├─ Turn|2♠ sub-chunk (~7k nodes)
      │   ├─ Turn|3♠ sub-chunk (~7k nodes)
      │   ├─ Turn|4♠ sub-chunk (~7k nodes)
      │   ├─ ... (8 total)
      │   └─ Turn|K♠ sub-chunk (~7k nodes)
      └─ River chunk (SubgameSolver with auto-split)
          ├─ River|2♠2♥ sub-chunk (~6k nodes)
          ├─ ... (32 total)
          └─ River|K♠K♥ sub-chunk (~6k nodes)
```

---

## Key Design Principles

### 1. Automatic Threshold Detection

```python
class SubgameSolver:
    def __init__(self, config, round_name, max_nodes=10000):
        self.max_nodes = max_nodes  # Configurable threshold
        self.estimated_nodes = self._estimate_chunk_size()
        self.needs_splitting = self.estimated_nodes > max_nodes
```

**Threshold logic:**
- Default: 10,000 nodes (safety margin above 1,600 limit)
- User-configurable for different hardware
- Automatic: No manual split configuration needed

### 2. Public Card Grouping

Split chunks by **turn/river card** combinations:

**For Turn chunks:**
```
Original: All possible turn cards in one game
Sub-chunks: One game per turn card
  - Turn|2♠: Only hands where turn card is 2♠
  - Turn|3♠: Only hands where turn card is 3♠
  - ... (num_suits × num_ranks total sub-chunks)
```

**For River chunks:**
```
Original: All possible river cards in one game
Sub-chunks: One game per river card
  - River|2♠: Only hands where river card is 2♠
  - ... (larger number of combinations)
```

### 3. Blueprint Propagation

Each sub-chunk uses the **previous sub-chunk's policy** as blueprint:

```python
Turn|2♠ → solve with flop blueprint
Turn|3♠ → solve with Turn|2♠ blueprint (warm-start)
Turn|4♠ → solve with Turn|3♠ blueprint (warm-start)
...
```

**Rationale:** Sequential solving with warm-starting converges faster than independent solving.

### 4. Policy Merging

Combine sub-chunk policies into unified policy:

```python
turn_policy = CombinedPolicy({
    "turn_2s": policy_2s,
    "turn_3s": policy_3s,
    "turn_4s": policy_4s,
    ...
})
```

Uses existing `CombinedPolicy` class (Phase 8.5).

---

## Implementation Strategy

### Phase 8.7.1: Node Estimation

Add method to estimate chunk size before solving:

```python
def _estimate_chunk_size(self) -> int:
    """
    Estimate number of nodes in this chunk.

    Strategy:
    1. Create game with truncated depth
    2. Sample a few game trees
    3. Extrapolate to full depth

    Returns:
        Estimated number of nodes (rough approximation)
    """
    game = pyspiel.load_game("universal_poker", self.subgame_config)

    # Quick sampling-based estimation
    sample_size = 100
    total_nodes = 0

    for _ in range(sample_size):
        state = game.new_initial_state()
        nodes = self._count_nodes_sample(state, max_depth=10)
        total_nodes += nodes

    avg_nodes = total_nodes / sample_size
    estimated = int(avg_nodes * self._depth_multiplier())

    return estimated
```

**Alternative:** Use heuristic formula based on deck size, betting actions, etc.

### Phase 8.7.2: Public Card Sub-Chunking

Split by fixing a public card:

```python
def _create_sub_chunks(self) -> List['SubgameSolver']:
    """
    Split this chunk into sub-chunks by public card.

    For Turn: Split by turn card (last board card)
    For River: Split by river card (last board card)
    For Flop: No split needed (already small)

    Returns:
        List of SubgameSolver instances, one per public card combination
    """
    if self.round not in ['turn', 'river']:
        return [self]  # No split needed

    sub_chunks = []
    public_cards = self._enumerate_public_cards()

    for card in public_cards:
        # Create modified config fixing this public card
        sub_config = self._fix_public_card(card)

        sub_chunk = SubgameSolver(
            full_game_config=sub_config,
            round_name=self.round,
            blueprint_policy=self.blueprint,
            precision=self.precision,
            micro_batch_size=self.micro_batch_size,
            max_nodes=self.max_nodes  # Propagate threshold
        )
        sub_chunks.append(sub_chunk)

    return sub_chunks
```

**Challenge:** OpenSpiel doesn't directly support "fixing" a public card. We need to:
1. Modify the game config to filter out unwanted cards
2. OR: Solve full game but only extract policy for relevant infosets
3. OR: Create a wrapper game that pre-deals the public card

### Phase 8.7.3: Sequential Sub-Chunk Solving

```python
def solve(self, iterations, progress_interval=1000) -> BlueprintPolicy:
    """
    Solve this subgame chunk, automatically sub-chunking if needed.

    If estimated nodes > max_nodes:
      - Split into sub-chunks by public card
      - Solve each sub-chunk sequentially with warm-starting
      - Merge policies

    Otherwise:
      - Solve directly (current behavior)
    """
    if not self.needs_splitting:
        # Current path: solve directly
        return self._solve_direct(iterations, progress_interval)

    # New path: hierarchical solving
    logger.info(f"  {self.round} chunk too large ({self.estimated_nodes} nodes)")
    logger.info(f"  Splitting into sub-chunks (target: <{self.max_nodes} nodes each)")

    sub_chunks = self._create_sub_chunks()
    sub_policies = {}
    current_blueprint = self.blueprint

    for i, sub_chunk in enumerate(sub_chunks):
        logger.info(f"\n  Sub-chunk {i+1}/{len(sub_chunks)}: {sub_chunk.descriptor}")

        # Solve with warm-start from previous sub-chunk
        sub_chunk.blueprint = current_blueprint
        policy = sub_chunk._solve_direct(iterations, progress_interval)
        sub_policies[sub_chunk.descriptor] = policy

        # Use this policy as blueprint for next sub-chunk (warm-start)
        current_blueprint = policy

        # Memory cleanup between sub-chunks
        gc.collect()
        jax.clear_caches()

    # Merge sub-chunk policies
    logger.info(f"\n  Merging {len(sub_policies)} sub-chunk policies...")
    merged_policy = self._merge_sub_policies(sub_policies)

    return merged_policy
```

### Phase 8.7.4: Policy Merging

```python
def _merge_sub_policies(
    self,
    sub_policies: Dict[str, BlueprintPolicy]
) -> BlueprintPolicy:
    """
    Merge sub-chunk policies into single unified policy.

    Strategy:
    - Each sub-chunk covers disjoint set of infosets (conditioned on public card)
    - Simply union all policies (no conflicts possible)

    Returns:
        BlueprintPolicy containing all sub-chunk policies
    """
    merged_dict = {}

    for descriptor, policy in sub_policies.items():
        # Add all infosets from this sub-policy
        for infoset, actions in policy.policy.items():
            if infoset in merged_dict:
                logger.warning(f"  Duplicate infoset: {infoset} (shouldn't happen!)")
            merged_dict[infoset] = actions

    logger.info(f"  Merged policy: {len(merged_dict)} total infosets")

    return BlueprintPolicy(merged_dict)
```

---

## Public Card Fixing: Implementation Options

### Option A: Game Wrapper (Clean but Complex)

Create wrapper game that pre-deals public cards:

```python
class FixedPublicCardGame:
    """Wrapper around universal_poker that fixes public cards."""

    def __init__(self, base_game, fixed_cards: List[int]):
        self.base_game = base_game
        self.fixed_cards = fixed_cards

    def new_initial_state(self):
        state = self.base_game.new_initial_state()
        # Force-deal fixed public cards
        for card in self.fixed_cards:
            state.apply_action(card)
        return state
```

**Pros:** Clean separation, easy to understand
**Cons:** Requires wrapping OpenSpiel API, complex

### Option B: Filtered Policy Extraction (Simpler)

Solve full game, extract only relevant infosets:

```python
def _solve_with_public_card_filter(self, fixed_card: int):
    """Solve full game, filter policy by public card."""

    # Solve normally
    game = pyspiel.load_game("universal_poker", self.subgame_config)
    solver = MatrixCFRSolver(game, ...)
    solver.solve(iterations)
    full_policy = solver.get_strategy_dict()

    # Filter to only infosets matching fixed_card
    filtered_policy = {
        infoset: actions
        for infoset, actions in full_policy.items()
        if self._infoset_matches_public_card(infoset, fixed_card)
    }

    return BlueprintPolicy(filtered_policy)
```

**Pros:** Simpler, no game wrapper needed
**Cons:** Solves full game (wastes compute on filtered-out infosets)

### Option C: Config Modification (Hacky but Fast)

Modify game config to reduce deck size:

```python
# If fixing turn card = 2♠, remove it from deck
modified_config = config.copy()
modified_config['available_cards'] = [c for c in cards if c != card_2s]
# Force turn card to be 2♠ (would need OpenSpiel modification)
```

**Pros:** Most efficient, only solves relevant game tree
**Cons:** Requires OpenSpiel modifications (not feasible)

### Recommended: Option B (Filtered Extraction)

**Rationale:**
- Simplest to implement (no game wrapper, no OpenSpiel mods)
- Still provides memory benefits (only store filtered policy)
- Computational waste is acceptable (solving is parallelizable)
- Can optimize later with Option A if needed

---

## Memory Benefits

### Expected Memory Reduction

**Without sub-chunking (Turn, 57k nodes):**
- Base memory: ~14 GB
- Utilities: ~95 MB (with Phase 8.6 optimizations)
- **Total: ~15 GB → OOM on 16GB VRAM**

**With sub-chunking (Turn → 8 sub-chunks, ~7k nodes each):**
- Base memory per sub-chunk: ~2 GB
- Utilities: ~12 MB (with Phase 8.6 optimizations)
- **Total per sub-chunk: ~2.5 GB → ✅ Fits easily**

**Reduction: 15 GB → 2.5 GB (6× smaller)**

### Sequential Solving Overhead

Solving 8 sub-chunks sequentially vs 1 full chunk:
- **Compute**: 8× (unavoidable)
- **Wall time**: 8× (but each chunk converges faster with warm-start)
- **Memory**: 6× reduction (enables solving at all)

**Trade-off is acceptable** since the alternative is OOM (infinite time).

---

## Warm-Starting Benefits

### Blueprint Propagation Across Sub-Chunks

```
Sub-chunk 1 (Turn|2♠): Solve from uniform → Policy A
Sub-chunk 2 (Turn|3♠): Solve from Policy A → Policy B (faster convergence)
Sub-chunk 3 (Turn|4♠): Solve from Policy B → Policy C (even faster)
...
```

**Expected speedup:** 2-5× faster convergence per sub-chunk after the first.

**Rationale:** Strategies for different turn cards are correlated. A good strategy for Turn|2♠ provides a strong starting point for Turn|3♠.

---

## Configuration API

### User-Facing Parameters

```python
# Simple usage (automatic sub-chunking)
chunked_solver = ChunkedSolver(
    full_game_config=holdem_config,
    max_nodes_per_chunk=10000,  # Auto-split chunks larger than this
    precision='fp16',
    micro_batch_size=6
)
```

### Advanced Usage

```python
# Disable auto-splitting for specific rounds
chunked_solver = ChunkedSolver(
    full_game_config=holdem_config,
    max_nodes_per_chunk={
        'preflop': 50000,  # Never split preflop
        'flop': 20000,
        'turn': 10000,  # Split turn if >10k
        'river': 5000   # Aggressive split for river
    }
)
```

---

## Testing Strategy

### Phase 8.7 Validation Tests

1. **Test 1: Turn Sub-Chunking (Critical)**
   - Config: 2 suits, 4 ranks, FCPA (57k nodes)
   - Expected: 8 sub-chunks of ~7k nodes each
   - Target: ✅ All sub-chunks solve successfully
   - Metric: Combined policy converges to equilibrium

2. **Test 2: Warm-Starting Benefit**
   - Measure: Iterations to convergence for sub-chunk 1 vs sub-chunk 8
   - Expected: 2-5× faster convergence with warm-start

3. **Test 3: Policy Merging Correctness**
   - Compare: Merged policy exploitability vs monolithic policy (if feasible)
   - Expected: Similar exploitability (within 5%)

4. **Test 4: Full 4-Round Pipeline**
   - Solve: Preflop → Flop → Turn (8 sub-chunks) → River (32 sub-chunks)
   - Target: Complete end-to-end solve on 16GB VRAM

---

## Implementation Checklist

### Phase 8.7.1: Foundation
- [ ] Add `max_nodes` parameter to `SubgameSolver.__init__()`
- [ ] Implement `_estimate_chunk_size()` method
- [ ] Add `needs_splitting` property

### Phase 8.7.2: Sub-Chunking Logic
- [ ] Implement `_enumerate_public_cards()` method
- [ ] Implement `_solve_with_public_card_filter()` method (Option B)
- [ ] Implement `_infoset_matches_public_card()` helper

### Phase 8.7.3: Sequential Solving
- [ ] Modify `solve()` to check `needs_splitting`
- [ ] Implement sub-chunk loop with warm-starting
- [ ] Add progress logging for sub-chunks

### Phase 8.7.4: Policy Merging
- [ ] Implement `_merge_sub_policies()` method
- [ ] Add conflict detection (should never happen)
- [ ] Validate merged policy completeness

### Phase 8.7.5: Integration
- [ ] Update `ChunkedSolver` to pass `max_nodes` parameter
- [ ] Add configuration options
- [ ] Update documentation

### Phase 8.7.6: Validation
- [ ] Create `test_phase8.7_turn_subchunking.py`
- [ ] Test Turn chunk (57k → 8×7k)
- [ ] Measure warm-starting speedup
- [ ] Validate policy correctness

---

## Expected Timeline

- **Phase 8.7.1**: 1 hour (foundation + estimation)
- **Phase 8.7.2**: 2 hours (sub-chunking logic + filtering)
- **Phase 8.7.3**: 1 hour (sequential solving)
- **Phase 8.7.4**: 30 minutes (policy merging)
- **Phase 8.7.5**: 30 minutes (integration)
- **Phase 8.7.6**: 2 hours (validation + testing)

**Total: ~7 hours** (1 session)

---

## Success Criteria

✅ **Phase 8.7 Complete** when:

1. Turn chunk (57k nodes) solves successfully via sub-chunking
2. Sub-chunks are ≤10k nodes each
3. Warm-starting provides measurable speedup
4. Merged policy is valid and complete
5. Full 4-round pipeline runs without OOM
6. Performance is acceptable (<24 hours for full solve)

---

## Risks & Mitigation

### Risk 1: OpenSpiel Public Card Filtering

**Problem:** Hard to filter infosets by public card without game modification

**Mitigation:**
- Infoset strings contain board cards → parse string
- Fallback: Solve multiple times with different random seeds (inefficient but works)

### Risk 2: Warm-Starting Doesn't Help

**Problem:** Sub-chunk policies may not correlate

**Mitigation:**
- Even without warm-starting, sub-chunking still solves memory issue
- Speedup is bonus, not requirement

### Risk 3: Policy Merging Introduces Errors

**Problem:** Merged policy might have gaps or conflicts

**Mitigation:**
- Validate merged policy covers all expected infosets
- Compare exploitability against baseline (if computable)
- Add comprehensive unit tests

---

## Conclusion

Phase 8.7 completes the hierarchical architecture by adding **automatic sub-chunking** for large chunks. Combined with Phase 8.6's memory optimizations, this enables solving arbitrarily large poker games on consumer hardware.

**Next:** Implement Phase 8.7 and validate on 57k Turn chunk.
