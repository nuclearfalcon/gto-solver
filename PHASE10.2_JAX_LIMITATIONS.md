# Phase 10.2: JAX Batched Sampling Limitations

**Date**: 2025-01-30
**Status**: Blocked - Requires major refactor of game engine

## Summary

We attempted to implement batched trajectory sampling using JAX's `vmap` and `scan` to achieve 100-1000× speedup. However, we hit a fundamental limitation: **JAX cannot trace Python control flow with traced values**.

## The Problem

### JAX Tracing Errors Encountered

1. **String infosets block tracing** (Line 165 `trajectory_sampler.py`)
   - `state_to_infoset()` uses string formatting with traced values
   - Fixed by passing `None` for infoset and using uniform policy

2. **Boolean indexing not allowed** (Line 174)
   - `legal_indices[legal_indices >= 0]` uses boolean array indexing
   - Fixed by using `jnp.where` and `random.choice` directly

3. **Python if-else blocks tracing** (Line 195)
   - `state.hole_cards[player] if player >= 0 else jnp.array([-1, -1])`
   - Fixed by using `jnp.where` for conditional selection

4. **Cannot convert traced values to Python int** (Line 208)
   - `apply_action(state, int(action), subkey)` tries to convert JAX array to Python int
   - Fixed by removing `int()` cast

5. **Python if statements in apply_action() block tracing** (`holdem_jax.py:428`)
   - **ROOT CAUSE**: `if action == ACTION_FOLD:` uses Python control flow
   - **BLOCKER**: Entire `apply_action()` function uses Python `if` statements
   - **FIX REQUIRED**: Rewrite entire game engine to use `jax.lax.cond` and `jax.lax.switch`

### The Fundamental Issue

```python
# Current implementation (NOT JAX-traceable)
def apply_action(state, action, key):
    if action == ACTION_FOLD:  # ❌ Python if statement
        # ... fold logic
    elif action == ACTION_CALL:  # ❌ Python elif statement
        # ... call logic
    # ...
```

```python
# Required for JAX tracing (major refactor)
def apply_action(state, action, key):
    # Use jax.lax.switch for multi-way branching
    return jax.lax.switch(
        action,
        [fold_fn, call_fn, pot_bet_fn, allin_fn],
        state, key
    )
```

## Performance Measurements

### Sequential Trajectory Sampling (Baseline)
- **Kuhn Poker**: 5.93 trajectories/sec (169ms per trajectory)
- **Average decision points**: 3.2 per trajectory
- **GPU MCCFR Speed**: 19.28 iterations/sec

### What We Learned
1. Full JAX vmap/scan requires **JAX-native control flow** (`jax.lax.cond`, `jax.lax.switch`)
2. Current game engine uses **Python control flow** throughout (not JAX-compatible)
3. Rewriting the game engine is a **massive refactor** (several days of work)

## Why This is Hard

### Files That Need Rewriting
1. `matrix_cfr/holdem_jax.py` - All functions with Python if statements:
   - `apply_action()` - 100+ lines with complex branching
   - `_next_actor()` - Player rotation logic
   - `_advance_round()` - Round progression logic
   - `legal_actions()` - Action availability logic
   - `is_terminal()` - Terminal state detection
   - `payoffs()` - Payoff computation with folded players

2. `matrix_cfr/kuhn_jax.py` - Similar issues in Kuhn poker engine

### Estimated Effort
- **Time**: 3-5 days for complete refactor
- **Risk**: High - easy to introduce bugs in game logic
- **Testing**: Extensive validation needed to ensure correctness
- **Benefit**: 100-1000× speedup IF it works
- **Probability of success**: Medium (JAX nested conditionals are tricky)

## Alternative Optimization Strategy

Instead of full batched trajectory sampling, we pivot to **vectorized regret updates**:

### Approach
1. Sample multiple trajectories sequentially (keep current code)
2. Collect all (infoset, action, regret) tuples across trajectories
3. Batch the regret table updates using JAX operations
4. Expected speedup: **2-5× vs 100-1000×**

### Why This is Better
- ✅ No game engine rewrite required
- ✅ Lower risk of bugs
- ✅ Faster to implement (1 day vs 5 days)
- ✅ Still provides meaningful speedup
- ✅ Can be implemented incrementally

### Implementation Plan
1. Modify `compute_counterfactual_values()` to return batched updates
2. Vectorize regret table updates using JAX array operations
3. Test on Kuhn poker (simple case)
4. Extend to Hold'em if successful

## Decision

**PIVOT to vectorized regret updates** instead of full batched trajectory sampling.

**Rationale**:
- Full batching requires game engine rewrite (high risk, high effort)
- Vectorized updates provide 2-5× speedup with much less risk
- We can always revisit full batching in Phase 11 if needed

## Lessons Learned

1. **JAX tracing is strict**: Any Python control flow with traced values fails
2. **Check JAX compatibility early**: Should have tested tracing before full implementation
3. **Start simple**: Should have tested with toy game first (single action type)
4. **Read JAX docs carefully**: `jax.lax.cond` and `jax.lax.switch` are required for branching
5. **Estimate refactor scope**: Game engine rewrite is much larger than anticipated

## Next Steps

1. ✅ Document JAX limitations (this file)
2. ⏳ Implement vectorized regret updates
3. ⏳ Benchmark vectorized vs sequential updates
4. ⏳ Run validation tests (Kuhn 10K iterations)
5. ⏳ Document Phase 10.2 results

## References

- JAX Documentation: https://docs.jax.dev/en/latest/errors.html
- JAX Control Flow: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow
- TracerBoolConversionError: https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError
