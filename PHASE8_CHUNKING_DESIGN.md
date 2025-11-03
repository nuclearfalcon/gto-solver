# Phase 8: Chunking Architecture Design

**Goal**: Enable full Hold'em solving by dividing game tree into manageable chunks

## Problem Statement

**Current limitation**: Full Hold'em game trees are too large for memory
- Tiny Hold'em (74K nodes): ✅ Working (142 MB)
- **Full Hold'em estimate**: 10-100M nodes → 10-100 GB (OOM!)

**Solution**: **Subgame decomposition** (chunking by betting round)

---

## Chunking Strategy: Betting Round Decomposition

### Core Concept

Divide Hold'em into 4 independent subgames:
1. **Preflop chunk**: Solve from deal → end of preflop betting
2. **Flop chunk**: Solve from flop deal → end of flop betting (using preflop policy as input)
3. **Turn chunk**: Solve from turn → end of turn betting (using flop policy)
4. **River chunk**: Solve from river → showdown (using turn policy)

**Key insight**: Each chunk is ~100K-500K nodes instead of 10M+

### Tree Size Reduction

| Chunk | Cards Dealt | Betting Rounds | Est. Nodes | Memory |
|-------|-------------|----------------|------------|--------|
| **Preflop** | 2 hole cards | 1 round | ~10K | ~20 MB |
| **Flop** | + 3 board | 1 round | ~500K | ~800 MB |
| **Turn** | + 1 card | 1 round | ~2M | ~3 GB |
| **River** | + 1 card | 1 round | ~8M | ~12 GB |
| **TOTAL (chunked)** | - | 4 chunks | **~10.5M** | **~16 GB** (sequential) |
| **TOTAL (monolithic)** | - | Combined | **~100M+** | **~150+ GB** (OOM!) |

**Savings**: 10-100x memory reduction by solving incrementally

---

## Chunking Approaches in Literature

### 1. **Subgame Solving (Johanson et al., 2012)**
- Solve trunk game → get reach probabilities at subgame roots
- Solve subgames independently using reach-weighted equilibria
- **Pro**: Sound game-theoretic foundation
- **Con**: Requires reach probability tracking

### 2. **Blueprint Strategy + Refinement (Brown & Sandholm, 2017)**
- Solve coarse abstraction → "blueprint strategy"
- Refine in realtime for specific subgames
- **Pro**: Used in Libratus (beat pros!)
- **Con**: Complex realtime refinement

### 3. **Depth-Limited Solving (Brown et al., 2019)**
- Solve to depth D, treat deeper as terminal with value function
- **Pro**: Simple cutoff
- **Con**: Needs value function approximation

### 4. **Sequential Decomposition (Our Approach)**
- Solve round 1 → use as "policy network" for round 2 → ...
- **Pro**: Simple, no reach probability tracking
- **Con**: May lose some EV if boundaries not carefully chosen

---

## Proposed Architecture

### API Design

```python
class SubgameSolver:
    """
    Solves a single betting round chunk using blueprint strategy from previous round.
    """

    def __init__(
        self,
        game_config: dict,
        round_name: str,  # "preflop", "flop", "turn", "river"
        blueprint_policy: Optional[Policy] = None  # From previous round
    ):
        self.config = game_config
        self.round = round_name
        self.blueprint = blueprint_policy

    def solve(self, iterations: int) -> Policy:
        """
        Solve this subgame to equilibrium.

        If blueprint_policy provided, use it to:
        1. Initialize strategies at subgame entry points
        2. Compute reach probabilities for weighting
        3. Provide terminal values at subgame boundaries
        """
        pass

    def create_subgame_config(self) -> dict:
        """Generate game config for just this betting round."""
        pass


class ChunkedSolver:
    """
    Orchestrates solving Hold'em in chunks (preflop → flop → turn → river).
    """

    def __init__(self, full_game_config: dict):
        self.config = full_game_config
        self.chunks = self._decompose_into_chunks()

    def solve(self, iterations_per_chunk: int) -> CombinedPolicy:
        """
        Solve all chunks sequentially, feeding each solution forward.

        Returns:
            Combined policy spanning all rounds
        """
        blueprint = None

        for chunk_name in ["preflop", "flop", "turn", "river"]:
            print(f"Solving {chunk_name} chunk...")

            subgame = SubgameSolver(
                game_config=self.config,
                round_name=chunk_name,
                blueprint_policy=blueprint
            )

            policy = subgame.solve(iterations=iterations_per_chunk)
            blueprint = policy  # Feed forward

        return self._combine_policies(policies)
```

### Key Components

#### 1. **Subgame Boundary Identification**
```python
def identify_subgame_roots(game_tree, round_name):
    """
    Find all nodes that start a betting round.

    For "flop" chunk:
    - Roots: All states where flop just dealt (before any flop betting)
    - Terminals: All states where flop betting complete (call/fold/reach turn)
    """
    roots = []
    for node in game_tree:
        if node.round == round_name and node.is_round_start:
            roots.append(node)
    return roots
```

#### 2. **Blueprint Strategy Integration**
```python
def apply_blueprint_at_boundaries(subgame, blueprint_policy):
    """
    Use previous round's policy at subgame entry points.

    Two uses:
    1. Initialize strategies: Start solving from blueprint, not uniform
    2. Compute reach probabilities: How likely to reach each subgame root
    """
    for root in subgame.roots:
        # Extract infoset from previous round
        prev_infoset = root.history_up_to_previous_round()

        if prev_infoset in blueprint_policy:
            # Use blueprint action distribution
            reach_prob = compute_reach_from_blueprint(root, blueprint_policy)
            subgame.set_initial_reach(root, reach_prob)
```

#### 3. **OpenSpiel Game Config Modification**

Challenge: OpenSpiel doesn't natively support "start from flop" games.

**Solution**: Modify game config to skip rounds
```python
def create_flop_only_config(base_config):
    """
    Create config that starts game at flop.

    Approach:
    - Set num_rounds=1 (only flop betting)
    - Set num_board_cards="3" (flop already dealt)
    - Set num_hole_cards=2 (already dealt)
    - Blind structure irrelevant (use placeholder)
    """
    return {
        "num_players": base_config["num_players"],
        "num_rounds": 1,  # Just flop betting
        "num_board_cards": "3",  # Flop cards
        "num_hole_cards": 2,
        # ... other params
    }
```

**Problem**: This creates a NEW game, not a true subgame. Need to map states.

**Alternative**: Use full game but **filter** to only solve flop nodes
```python
def filter_to_subgame(full_game, round_name):
    """
    Build game tree but only include nodes from specific round.

    Treat previous rounds as:
    - Chance nodes (cards dealt with uniform prob)
    - Action nodes with fixed blueprint strategy (not trainable)
    """
    pass
```

---

## Implementation Plan

### Phase 8.2: Subgame Infrastructure (Week 1-2)

1. **SubgameSolver class** (`matrix_cfr/subgame_solver.py`)
   - Basic structure
   - Config generation for single rounds
   - Policy import/export

2. **Game Tree Filtering** (`matrix_cfr/game_tree_filter.py`)
   - Identify round boundaries
   - Extract subgame nodes
   - Handle transition states

3. **Blueprint Policy Format** (`matrix_cfr/blueprint_policy.py`)
   - Save/load policies
   - Convert between formats (dict ↔ matrix)
   - Merge policies from different rounds

### Phase 8.3: Preflop Chunk (Week 2-3)

1. **Preflop Solver**
   - Standalone preflop game
   - No blueprint needed (first chunk)
   - Validate convergence

2. **Policy Export**
   - Save preflop equilibrium
   - Format for consumption by flop chunk

### Phase 8.4: Flop Chunk (Week 3-4)

1. **Blueprint Integration**
   - Load preflop policy
   - Compute reach probabilities at flop entry
   - Initialize flop strategies from blueprint

2. **Flop Solver**
   - Solve conditioned on preflop
   - Validate combined preflop→flop policy

### Phase 8.5: Turn & River (Week 4-6)

1. **Repeat for Turn**
   - Use flop blueprint
   - Solve turn chunk

2. **Repeat for River**
   - Use turn blueprint
   - Solve river chunk

3. **Full Pipeline Test**
   - Preflop → Flop → Turn → River
   - Measure total convergence

---

## Challenges & Solutions

### Challenge 1: State Mapping Between Chunks

**Problem**: Flop game trees have different state IDs than full game

**Solution**: Use **information sets as keys**, not node IDs
- Infosets are deterministic (history string)
- Same infoset in preflop chunk = same in flop chunk
- Policy mapping: `dict[infoset] -> action_distribution`

### Challenge 2: Reach Probability Tracking

**Problem**: Need to know how likely each flop root is

**Solution**: **Simple Monte Carlo estimation**
```python
def estimate_reach_probabilities(preflop_policy, num_samples=10000):
    """
    Sample hands using preflop policy to estimate reach at flop roots.
    """
    reach_counts = defaultdict(int)

    for _ in range(num_samples):
        # Sample a hand
        hole_cards = deal_random_hand()

        # Play through preflop using policy
        history = play_preflop(hole_cards, preflop_policy)

        # Increment count for this flop entry point
        if not history.is_terminal:  # Reached flop
            flop_root_key = get_flop_root_infoset(history)
            reach_counts[flop_root_key] += 1

    # Normalize
    total = sum(reach_counts.values())
    return {k: v/total for k, v in reach_counts.items()}
```

### Challenge 3: Terminal Values at Boundaries

**Problem**: When flop betting ends (call/check), what's the value?

**Solution A**: **Rollout with blueprint**
- Use turn+river blueprint to estimate EV
- Requires all chunks solved first (circular dependency)

**Solution B**: **Iterative refinement**
- First pass: Use simplified terminal values (equity-based)
- Second pass: Re-solve with better value estimates
- Repeat until converged

**Solution C**: **Decomposition without terminal values** (SIMPLEST)
- Each chunk solves to its natural terminal (fold/showdown)
- No artificial cutoffs
- May miss some EV boundary effects

**Recommendation**: Start with **Solution C** (simplest), upgrade to B if needed

---

## Testing Strategy

### Unit Tests
1. Test subgame config generation
2. Test blueprint policy save/load
3. Test reach probability estimation

### Integration Tests
1. **Preflop-only**: Solve, verify convergence
2. **Preflop→Flop**: Verify flop uses preflop correctly
3. **Full pipeline**: Preflop→Flop→Turn→River

### Validation Tests
1. **Compare with monolithic**: On small games (2-card deck), compare chunked vs full-tree
2. **Exploitability**: Measure Nash conv of combined policy
3. **Human inspection**: Sample hands, verify strategies make sense

---

## Success Criteria

### Minimum (Proof of Concept)
- ✅ Preflop chunk solves correctly
- ✅ Flop chunk uses preflop blueprint
- ✅ Combined policy playable (no crashes)

### Target (Working System)
- ✅ All 4 chunks solve
- ✅ Combined policy has reasonable exploitability
- ✅ Solves full 2-player Hold'em (no OOM)

### Stretch (Production Quality)
- ✅ 3-player Hold'em works
- ✅ Exploitability competitive with monolithic (within 10%)
- ✅ Iteration over chunks (refine boundaries)

---

## Next Steps (Immediate)

1. **Create `SubgameSolver` skeleton class**
2. **Implement preflop-only config generation**
3. **Test preflop chunk solving**
4. **Design blueprint policy format**
5. **Prototype flop chunk with dummy blueprint**

---

## References

- Johanson et al. (2012): "Finding Optimal Abstract Strategies in Extensive-Form Games"
- Brown & Sandholm (2017): "Safe and Nested Subgame Solving for Imperfect-Information Games"
- arXiv:2408.14778v5: Our matrix CFR paper (baseline)

---

**Status**: Design phase complete, ready for implementation

**Estimated timeline**: 4-6 weeks to working 2-player Hold'em chunking
