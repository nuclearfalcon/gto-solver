# Phase 10 Design: GPU-Accelerated MCCFR

**Date:** 2025-01-03
**Goal:** Implement JAX-based Hold'em engine with GPU-parallelized MCCFR sampling
**Status:** 🎯 Design Phase
**Timeline:** 2-3 weeks

---

## Executive Summary

Phase 9 revealed that Matrix-based CFR (and MCCFR) both hit fundamental limits:
- **Matrix CFR**: Fast on GPU but memory-bound (~100K nodes max)
- **CPU MCCFR**: Memory-efficient but extremely slow (0.01-1 it/s)

**The Breakthrough:** Combine both strengths through **GPU-parallelized MCCFR**
- Sample 10,000 trajectories **simultaneously** on GPU (one per core)
- Accumulate regrets in parallel using JAX
- **Result: Low memory (sampling) + High speed (GPU parallelism)**

**Expected Performance:**
- Speed: **100-1000 it/s** (1000-100000× faster than CPU MCCFR)
- Memory: **~1-2 GB** (vs 14-16 GB for Matrix CFR)
- **Can potentially solve full 52-card Hold'em**

---

## Problem Statement

### Current Limitations

| Approach | Speed | Memory | Can Solve Full Hold'em? | Limitation |
|----------|-------|--------|------------------------|------------|
| Python MCCFR | 0.01 it/s | ~100 MB | ❌ Too slow (weeks) | CPU bottleneck |
| C++ MCCFR | ~1 it/s | ~100 MB | ❌ Still too slow (days) | Single-threaded |
| Matrix CFR | 0.14-1.66 it/s | 14-16 GB | ❌ OOM on large games | Full tree required |

**Conclusion:** Neither tree-based (Matrix) nor CPU sampling (MCCFR) can solve arbitrarily large games.

### The Key Insight

**Why is MCCFR slow?**
- Samples ONE trajectory at a time sequentially
- CPU-bound (can't parallelize)
- Each iteration processes ~100-1000 nodes

**GPU Advantage:**
- 10,000+ cores available (RTX 4060 Ti)
- Can sample 10,000 trajectories **simultaneously**
- Each core is independent (embarrassingly parallel)

**The Solution:**
Implement MCCFR where each GPU core samples its own trajectory in parallel!

---

## Architecture Design

### Component Overview

```
Phase 10: GPU-Accelerated MCCFR
├─ holdem_jax.py              # JAX Hold'em engine (pure functions)
├─ gpu_mccfr_solver.py        # Main solver with batched sampling
├─ trajectory_sampler.py      # Vectorized trajectory generation
├─ regret_table.py            # Sparse regret/strategy storage
└─ test_phase10_*.py          # Comprehensive test suite
```

### 1. JAX Hold'em Engine

**File:** `holdem_jax.py` (~400 lines)

**Objective:** Implement Hold'em as pure JAX functions for GPU-native execution

#### State Representation

```python
from typing import NamedTuple
import jax.numpy as jnp

class HoldemState(NamedTuple):
    """
    Pure JAX state representation.

    All fields are JAX arrays for GPU compatibility.
    Immutable for functional programming style.
    """
    # Cards (using card indices 0-51)
    hole_cards: jnp.ndarray  # shape: (num_players, 2), dtype: int32
    board: jnp.ndarray       # shape: (5,), dtype: int32, padded with -1
    deck: jnp.ndarray        # shape: (52,), dtype: bool (available cards)

    # Betting state
    bets: jnp.ndarray        # shape: (num_players,), dtype: float32
    pot: jnp.float32         # Total pot size
    stacks: jnp.ndarray      # shape: (num_players,), dtype: float32

    # Game flow
    round: jnp.int32         # 0=preflop, 1=flop, 2=turn, 3=river
    acting_player: jnp.int32 # Current player index
    num_actions_this_round: jnp.int32  # For action limiting

    # Status flags
    folded: jnp.ndarray      # shape: (num_players,), dtype: bool
    all_in: jnp.ndarray      # shape: (num_players,), dtype: bool
```

#### Core Functions

```python
def deal_initial_state(key: jax.random.PRNGKey, num_players: int,
                      stacks: jnp.ndarray, blinds: jnp.ndarray) -> HoldemState:
    """
    Create initial game state with cards dealt.

    Pure function: Same key → Same state (reproducible)
    """
    # Deal hole cards
    key, subkey = jax.random.split(key)
    all_cards = jnp.arange(52)
    shuffled = jax.random.permutation(subkey, all_cards)

    hole_cards = shuffled[:num_players*2].reshape(num_players, 2)
    remaining = shuffled[num_players*2:]

    # Initialize state
    return HoldemState(
        hole_cards=hole_cards,
        board=jnp.full(5, -1, dtype=jnp.int32),  # No board yet
        deck=jnp.ones(52, dtype=bool),  # All cards available initially
        bets=blinds,
        pot=jnp.sum(blinds),
        stacks=stacks - blinds,
        round=0,  # Preflop
        acting_player=find_first_actor(num_players),
        num_actions_this_round=0,
        folded=jnp.zeros(num_players, dtype=bool),
        all_in=jnp.zeros(num_players, dtype=bool)
    )

def apply_action(state: HoldemState, action: int) -> HoldemState:
    """
    Apply action and return new state.

    Pure function: No mutation, returns new state.

    Actions:
    - 0: Fold
    - 1: Call/Check
    - 2: Pot-sized bet/raise
    - 3: All-in
    """
    # Immutable updates using NamedTuple._replace()
    if action == 0:  # Fold
        new_folded = state.folded.at[state.acting_player].set(True)
        return state._replace(
            folded=new_folded,
            acting_player=find_next_actor(state, new_folded)
        )

    elif action == 1:  # Call/Check
        to_call = jnp.max(state.bets) - state.bets[state.acting_player]
        amount = jnp.minimum(to_call, state.stacks[state.acting_player])

        new_bets = state.bets.at[state.acting_player].add(amount)
        new_stacks = state.stacks.at[state.acting_player].add(-amount)
        new_pot = state.pot + amount

        return state._replace(
            bets=new_bets,
            stacks=new_stacks,
            pot=new_pot,
            acting_player=find_next_actor(state, state.folded)
        )

    # ... similar for pot bet and all-in

def legal_actions(state: HoldemState) -> jnp.ndarray:
    """
    Return mask of legal actions.

    Returns: boolean array [fold, call, bet, allin]
    """
    player = state.acting_player
    current_bet = jnp.max(state.bets)
    player_bet = state.bets[player]
    to_call = current_bet - player_bet

    can_fold = to_call > 0  # Only if facing a bet
    can_call = True  # Always (might be check)
    can_bet = state.stacks[player] > to_call  # Need chips to raise
    can_allin = state.stacks[player] > 0

    return jnp.array([can_fold, can_call, can_bet, can_allin], dtype=bool)

def is_terminal(state: HoldemState) -> bool:
    """Check if game is over."""
    # All but one folded
    active = ~state.folded
    if jnp.sum(active) == 1:
        return True

    # Reached river and betting complete
    if state.round == 3 and betting_complete(state):
        return True

    return False

def payoffs(state: HoldemState) -> jnp.ndarray:
    """
    Compute final payoffs for each player.

    Handles both showdown and fold scenarios.
    """
    if jnp.sum(~state.folded) == 1:
        # Everyone folded except one
        winner = jnp.argmax(~state.folded)
        payoff = jnp.zeros(state.hole_cards.shape[0])
        payoff = payoff.at[winner].set(state.pot)
        return payoff - (state.bets + state.pot / state.hole_cards.shape[0])

    # Showdown - evaluate hands
    hand_strengths = evaluate_hands_vectorized(state.hole_cards, state.board)
    winner = jnp.argmax(hand_strengths * ~state.folded)

    payoff = jnp.zeros(state.hole_cards.shape[0])
    payoff = payoff.at[winner].set(state.pot)
    return payoff - (state.bets + state.pot / state.hole_cards.shape[0])
```

**Key Properties:**
- All functions are **pure** (no side effects)
- All data is **JAX arrays** (GPU-compatible)
- States are **immutable** (functional style)
- Fully **JIT-compilable** for maximum speed

---

### 2. Vectorized Trajectory Sampling

**File:** `trajectory_sampler.py` (~300 lines)

**Objective:** Sample thousands of game trajectories in parallel

#### Single Trajectory Sampling

```python
def sample_trajectory(
    key: jax.random.PRNGKey,
    num_players: int,
    stacks: jnp.ndarray,
    blinds: jnp.ndarray,
    policy_fn: callable
) -> Tuple[List[HoldemState], List[int], List[int]]:
    """
    Sample one complete game trajectory.

    Args:
        key: Random key for reproducibility
        num_players: Number of players
        stacks: Starting stack sizes
        blinds: Blind amounts
        policy_fn: Function mapping (state, player) → action probabilities

    Returns:
        states: List of states visited
        actions: List of actions taken
        players: List of acting players
    """
    key, subkey = jax.random.split(key)
    state = deal_initial_state(subkey, num_players, stacks, blinds)

    states = []
    actions = []
    players = []

    while not is_terminal(state):
        player = state.acting_player
        legal = legal_actions(state)

        # Get policy for this infoset
        infoset = state_to_infoset(state, player)
        policy = policy_fn(infoset, legal)

        # Sample action
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, jnp.arange(4), p=policy)

        # Record and advance
        states.append(state)
        actions.append(action)
        players.append(player)

        state = apply_action(state, action)

    return states, actions, players
```

#### Batched Parallel Sampling

```python
def batch_sample_trajectories(
    keys: jnp.ndarray,  # shape: (batch_size, 2)
    num_players: int,
    stacks: jnp.ndarray,
    blinds: jnp.ndarray,
    policy_fn: callable,
    max_trajectory_length: int = 100
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample many trajectories in parallel using vmap.

    Args:
        keys: Array of random keys (one per trajectory)
        ... (same as single trajectory)
        max_trajectory_length: Max actions per trajectory (for padding)

    Returns:
        states: (batch_size, max_length, state_dims)
        actions: (batch_size, max_length)
        players: (batch_size, max_length)

    Note: Trajectories are padded to max_length with sentinel values
    """
    # Vectorize trajectory sampling over batch dimension
    vectorized_sample = jax.vmap(
        lambda key: sample_trajectory_fixed_length(
            key, num_players, stacks, blinds, policy_fn, max_trajectory_length
        )
    )

    return vectorized_sample(keys)

def sample_trajectory_fixed_length(
    key: jax.random.PRNGKey,
    num_players: int,
    stacks: jnp.ndarray,
    blinds: jnp.ndarray,
    policy_fn: callable,
    max_length: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample trajectory with fixed-length output (for vmapping).

    Uses JAX's scan for efficient looping with fixed iterations.
    """
    key, subkey = jax.random.split(key)
    initial_state = deal_initial_state(subkey, num_players, stacks, blinds)

    def scan_fn(carry, _):
        state, key = carry

        # Check if terminal
        done = is_terminal(state)

        # Sample action (or dummy if terminal)
        player = state.acting_player
        legal = legal_actions(state)
        infoset = state_to_infoset(state, player)
        policy = policy_fn(infoset, legal)

        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, jnp.arange(4), p=policy)

        # Apply action (or no-op if terminal)
        new_state = jax.lax.cond(
            done,
            lambda s: s,  # Keep state if terminal
            lambda s: apply_action(s, action),  # Advance if not
            state
        )

        return (new_state, key), (state, action, player, done)

    _, (states, actions, players, dones) = jax.lax.scan(
        scan_fn,
        (initial_state, key),
        None,
        length=max_length
    )

    return states, actions, players, dones
```

**Key Properties:**
- **Vectorized**: `vmap` applies sampling over batch of keys
- **Parallel**: Each trajectory computed independently on separate GPU core
- **Fixed-length**: Padding enables efficient GPU operations
- **JIT-compatible**: Full JAX compilation for maximum speed

---

### 3. GPU MCCFR Solver

**File:** `gpu_mccfr_solver.py` (~500 lines)

**Objective:** Implement external sampling MCCFR with GPU-parallel trajectory sampling

#### Regret Storage

```python
class RegretTable:
    """
    Sparse storage for regrets and strategies.

    Only stores infosets that have been visited (MCCFR property).
    Gradually builds complete strategy.
    """
    def __init__(self, num_actions: int = 4):
        self.regrets = {}  # {infoset_str: jnp.array(num_actions)}
        self.cumulative_strategy = {}  # {infoset_str: jnp.array(num_actions)}
        self.num_actions = num_actions

    def get_regrets(self, infoset: str) -> jnp.ndarray:
        """Get regrets for infoset (initialize if new)."""
        if infoset not in self.regrets:
            self.regrets[infoset] = jnp.zeros(self.num_actions)
        return self.regrets[infoset]

    def update_regrets(self, infoset: str, regret_updates: jnp.ndarray):
        """Accumulate regrets for infoset."""
        current = self.get_regrets(infoset)
        self.regrets[infoset] = current + regret_updates

    def get_strategy(self, infoset: str, legal_actions: jnp.ndarray) -> jnp.ndarray:
        """
        Compute current strategy from regrets using regret matching.

        Args:
            infoset: Infoset string
            legal_actions: Boolean mask of legal actions

        Returns:
            Probability distribution over actions
        """
        regrets = self.get_regrets(infoset)

        # Regret matching: prob ∝ max(regret, 0)
        positive_regrets = jnp.maximum(regrets, 0)

        # Mask illegal actions
        positive_regrets = positive_regrets * legal_actions

        # Normalize
        regret_sum = jnp.sum(positive_regrets)
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            # Uniform over legal actions
            strategy = legal_actions / jnp.sum(legal_actions)

        return strategy
```

#### Main CFR Loop

```python
class GPUMCCFRSolver:
    """GPU-accelerated Monte Carlo CFR using batched trajectory sampling."""

    def __init__(
        self,
        num_players: int = 2,
        stacks: List[float] = [1000, 1000],
        blinds: List[float] = [50, 100],
        batch_size: int = 10000,  # 10K parallel trajectories
        max_trajectory_length: int = 100
    ):
        self.num_players = num_players
        self.stacks = jnp.array(stacks)
        self.blinds = jnp.array(blinds)
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length

        self.regret_table = RegretTable()
        self.iteration = 0

    def solve(
        self,
        iterations: int = 1000,
        progress_interval: int = 100
    ):
        """
        Main CFR loop with GPU-parallel sampling.

        Each iteration:
        1. Sample batch_size trajectories in parallel on GPU
        2. Compute counterfactual values for each trajectory
        3. Update regrets for visited infosets
        4. Accumulate strategy
        """
        for it in range(iterations):
            # Generate batch of random keys
            key = jax.random.PRNGKey(self.iteration)
            keys = jax.random.split(key, self.batch_size)

            # Sample trajectories in parallel (GPU)
            trajectories = batch_sample_trajectories(
                keys,
                self.num_players,
                self.stacks,
                self.blinds,
                policy_fn=self._policy_fn,
                max_trajectory_length=self.max_trajectory_length
            )

            # Update regrets for each trajectory
            self._update_from_trajectories(trajectories)

            self.iteration += 1

            if it % progress_interval == 0:
                logger.info(f"Iteration {it}/{iterations}: "
                           f"{len(self.regret_table.regrets)} infosets visited")

    def _policy_fn(self, infoset: str, legal_actions: jnp.ndarray) -> jnp.ndarray:
        """Policy function for sampling (wraps regret table)."""
        return self.regret_table.get_strategy(infoset, legal_actions)

    def _update_from_trajectories(self, trajectories):
        """Update regrets from batch of trajectories."""
        states, actions, players, dones = trajectories

        # Process each trajectory
        for traj_idx in range(self.batch_size):
            self._update_from_single_trajectory(
                states[traj_idx],
                actions[traj_idx],
                players[traj_idx],
                dones[traj_idx]
            )

    def _update_from_single_trajectory(self, states, actions, players, dones):
        """
        External sampling update for one trajectory.

        For each player, compute counterfactual values:
        - What if I played differently at each infoset?
        - Update regrets = (counterfactual value - actual value)
        """
        # Find where trajectory ended
        trajectory_length = jnp.sum(~dones)

        # Get terminal payoffs
        final_state = states[trajectory_length - 1]
        utilities = payoffs(final_state)

        # Backward pass: compute counterfactual values
        for t in range(trajectory_length - 1, -1, -1):
            state = states[t]
            action_taken = actions[t]
            player = players[t]

            # Compute counterfactual values for all actions
            infoset = state_to_infoset(state, player)
            legal = legal_actions(state)

            # What's the value of each alternative action?
            alt_values = jnp.zeros(4)
            for a in range(4):
                if legal[a]:
                    # Simulate taking action a instead
                    alt_state = apply_action(state, a)
                    alt_value = self._evaluate_state(alt_state, utilities, player)
                    alt_values = alt_values.at[a].set(alt_value)

            # Actual value from action taken
            actual_value = alt_values[action_taken]

            # Regret = (alternative value - actual value)
            regrets = alt_values - actual_value

            # Update regret table
            self.regret_table.update_regrets(infoset, regrets)
```

**Key Properties:**
- **Batched sampling**: 10K trajectories per iteration
- **Sparse storage**: Only visited infosets stored
- **External sampling**: One player updates per trajectory
- **GPU-accelerated**: All trajectory sampling on GPU

---

## Expected Performance

### Theoretical Analysis

**Assumptions:**
- GPU: RTX 4060 Ti (10,752 CUDA cores)
- Batch size: 10,000 trajectories
- Average trajectory: 30 actions
- CPU MCCFR baseline: 1 it/s

**GPU Advantage:**
1. **Parallelization**: 10,000 trajectories simultaneously
   - Expected speedup: **10,000×** (if perfectly parallel)
   - Realistic (with overhead): **1,000-5,000×**

2. **JIT Compilation**: JAX compiles to optimized GPU kernels
   - Expected speedup: **2-5×** over Python

3. **Vectorization**: Batched operations reduce kernel launch overhead
   - Expected speedup: **2-3×** over sequential GPU calls

**Combined:** 1,000-5,000 × 2-5 × 2-3 = **4,000-75,000× speedup**

**Conservative Estimate:** **100-1000 it/s** (vs 0.01-1 it/s for CPU)

### Memory Requirements

**Per Trajectory:**
- State: ~200 bytes
- Actions/players: ~100 bytes
- **Total: ~300 bytes**

**Batch of 10K:**
- 10,000 × 300 bytes = **3 MB**

**Regret Tables:**
- Sparse storage (only visited infosets)
- Full Hold'em: ~100M infosets max
- At 4 floats per infoset: 100M × 16 bytes = **1.6 GB**
- In practice: Much less due to sparse visiting

**Total: ~1-2 GB** (vs 14-16 GB for Matrix CFR)

---

## Implementation Roadmap

### Week 1: JAX Hold'em Engine (Days 1-5)

**Day 1-2: State Representation & Core Functions**
- [ ] Define `HoldemState` NamedTuple
- [ ] Implement `deal_initial_state()`
- [ ] Implement `apply_action()` for all 4 actions
- [ ] Write unit tests

**Day 3-4: Game Logic**
- [ ] Implement `legal_actions()`
- [ ] Implement `is_terminal()`
- [ ] Implement `find_next_actor()` helper
- [ ] Implement betting round advancement
- [ ] Write comprehensive tests

**Day 5: Payoff Evaluation**
- [ ] Implement `payoffs()` for fold scenarios
- [ ] Implement `evaluate_hands_vectorized()` for showdown
- [ ] Integrate hand evaluator library (or implement simple version)
- [ ] Test on known scenarios

### Week 2: GPU MCCFR Implementation (Days 6-10)

**Day 6-7: Trajectory Sampling**
- [ ] Implement `sample_trajectory()` (sequential)
- [ ] Implement `batch_sample_trajectories()` (vectorized)
- [ ] Test vectorization correctness
- [ ] Benchmark speedup

**Day 8-9: MCCFR Core**
- [ ] Implement `RegretTable` class
- [ ] Implement `GPUMCCFRSolver` class
- [ ] Implement `_update_from_trajectories()`
- [ ] Implement counterfactual value computation

**Day 10: Testing & Validation**
- [ ] Test on Kuhn poker (validate correctness)
- [ ] Test on Leduc poker (validate scaling)
- [ ] Benchmark vs CPU MCCFR
- [ ] Measure memory usage

### Week 3: Optimization & Documentation (Days 11-14)

**Day 11-12: Performance Optimization**
- [ ] Profile GPU utilization
- [ ] Optimize batch size
- [ ] Optimize trajectory length
- [ ] JIT compile critical paths

**Day 13: Testing & Validation**
- [ ] Compare vs Matrix CFR (Kuhn/Leduc)
- [ ] Measure convergence rates
- [ ] Validate exploitability
- [ ] Stress test on larger games

**Day 14: Documentation**
- [ ] Complete test suite
- [ ] Write usage examples
- [ ] Create benchmark report
- [ ] Update PROJECT_STATUS.md

---

## Success Criteria

### Minimum Viable Product (Week 2)

- ✅ JAX Hold'em engine passes all unit tests
- ✅ GPU MCCFR solves Kuhn poker correctly
- ✅ Batched sampling is 100× faster than sequential
- ✅ Memory usage <500 MB for Leduc

### Target Product (Week 3)

- ✅ GPU MCCFR is 100-1000× faster than CPU MCCFR
- ✅ Memory usage <2 GB for large games
- ✅ Convergence rate similar to CPU MCCFR (validated on Kuhn/Leduc)
- ✅ Can solve larger games than Matrix CFR

### Stretch Goals (Future)

- ✅ Solve full 52-card Hold'em (all betting rounds)
- ✅ 1000+ it/s performance
- ✅ <1 GB memory for full Hold'em

---

## Risks & Mitigation

### Risk 1: Vectorization Overhead

**Problem:** GPU parallelization overhead might reduce speedup

**Mitigation:**
- Start with large batch sizes (10K)
- Profile GPU utilization
- Optimize kernel launch patterns
- Use JAX's pmap for multi-GPU if needed

### Risk 2: Trajectory Length Variability

**Problem:** Variable-length trajectories hard to vectorize

**Mitigation:**
- Pad to max length with sentinel values
- Use JAX's scan for fixed-length loops
- Mask out invalid entries in updates

### Risk 3: Sparse Regret Table on GPU

**Problem:** Python dictionaries don't work well on GPU

**Mitigation:**
- Keep regret tables on CPU (small memory)
- Only trajectory sampling on GPU
- Transfer is fast (<1ms per iteration)
- Alternative: Explore JAX pytrees for GPU dicts

### Risk 4: Convergence vs CPU MCCFR

**Problem:** Parallelization might affect convergence

**Mitigation:**
- Use proven external sampling algorithm
- Each trajectory is independent (no interference)
- Theory guarantees still apply
- Validate empirically on Kuhn/Leduc

---

## Literature Review

### Existing Work

1. **GPU-CFR Paper (arXiv:2408.14778)**:
   - Matrix-based approach
   - 352× speedup on GPU
   - **Limitation:** Requires full tree in memory

2. **MCCFR Papers (Lanctot et al. 2009)**:
   - External/outcome sampling
   - CPU-only implementations
   - **Limitation:** Single-threaded, slow

3. **Parallel MCCFR (Li et al. 2012)**:
   - Multi-core CPU parallelization
   - ~4-8× speedup on 8 cores
   - **Limitation:** Not GPU, limited scaling

### Novel Contribution

**No existing work on GPU-parallelized MCCFR with massive batching (10K+ trajectories)**

This is genuinely novel research combining:
- MCCFR's sampling (low memory)
- GPU's massive parallelism (high speed)
- JAX's JIT compilation (optimization)

**Potential for publication or open-source release!**

---

## Conclusion

Phase 10 represents a fundamental architectural shift from tree-based CFR to sampling-based CFR with GPU acceleration.

**Key Innovation:** GPU-parallelized MCCFR combines the best of both worlds:
- Low memory (sampling, no full tree)
- High speed (GPU, 10K parallel trajectories)

**Expected Impact:**
- 100-1000× faster than CPU MCCFR
- ~10× lower memory than Matrix CFR
- Can solve arbitrarily large games (no tree size limit)

**If successful, this enables:**
- Full 52-card Hold'em solving
- 3+ player games without constraints
- Real-time strategy computation
- Novel research contributions

The failed Phase 9 experiment was necessary to reach this breakthrough insight. Sometimes the most valuable experiments show what **doesn't** work, pushing us toward better solutions.

---

## References

1. GPU-CFR Paper: https://arxiv.org/pdf/2408.14778v5
2. MCCFR Papers: Lanctot et al. (2009)
3. External Sampling: https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf
4. JAX Documentation: https://jax.readthedocs.io/
5. Hold'em Hand Evaluation: https://github.com/worldveil/deuces

---

**Next:** Begin Week 1 implementation - JAX Hold'em Engine
