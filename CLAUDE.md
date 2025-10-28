# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GTO (Game Theory Optimal) poker training project that uses **OpenSpiel** (version 1.6.8) to simulate and analyze No-Limit Hold'em poker games. The project focuses on understanding poker game configurations, betting abstractions, and information state tensors for machine learning applications.

## Python Environment

**Critical:** This project does NOT use a local virtual environment. Instead, it relies on an external OpenSpiel installation.

### Activating the Environment

All Python scripts in this repository require activating the OpenSpiel virtual environment first:

```bash
source ~/open_spiel/venv/bin/activate
```

**Always run this before executing any Python code.** The scripts will fail without this environment active.

### OpenSpiel Installation

- **Location:** `/home/nuclearfalcon/open_spiel`
- **Version:** 1.6.8
- **Python:** 3.10.12
- **Key dependency:** `pyspiel` module (compiled C++ extension at `/home/nuclearfalcon/open_spiel/pyspiel.so`)

## Running Tests

This repository contains comprehensive test suites. All tests must be run with the OpenSpiel venv activated.

### Run All Configuration Tests
```bash
source ~/open_spiel/venv/bin/activate
python test_poker_configs.py
```

Tests: asymmetrical stakes, betting abstractions (fc/fcpa/fchpa/fullgame), ante simulations, and known limitations.

### Run Tensor Analysis Tests
```bash
source ~/open_spiel/venv/bin/activate
python test_tensor_bet_sizes.py
```

**Critical test suite** proving that information state tensors store actual bet sizes even when using betting abstractions. Contains 18 assertions across 5 test categories.

### Run Example Simulations
```bash
source ~/open_spiel/venv/bin/activate
python holdem_example.py
```

Demonstrates basic poker simulation: heads-up, 6-max, and batch hand simulation.

## Core Architecture

### OpenSpiel's `universal_poker` Game

The entire project is built around OpenSpiel's `universal_poker` game engine, which provides configurable poker simulations.

**Game Creation Pattern:**
```python
game = pyspiel.load_game('universal_poker', {
    'betting': 'nolimit',           # or 'limit', 'potlimit'
    'numPlayers': 2,                # 2-10 players
    'numRounds': 4,                 # Preflop, Flop, Turn, River
    'blind': '100 50',              # Space-separated blind values per player
    'firstPlayer': '2 1 1 1',       # Who acts first each round
    'numSuits': 4,
    'numRanks': 13,
    'numHoleCards': 2,
    'numBoardCards': '0 3 1 1',     # Cards per round
    'stack': '20000 20000',         # Stacks per player (can be asymmetric)
    'bettingAbstraction': 'fcpa'    # fc, fcpa, fchpa, or fullgame
})
```

### Game Simulation Loop

**Two types of nodes:**
1. **Chance nodes** (`state.is_chance_node()`): Card dealing by the game engine
2. **Decision nodes**: Player actions (fold, call, bet, raise, all-in)

**Standard simulation pattern:**
```python
state = game.new_initial_state()
while not state.is_terminal():
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = random.choices(action_list, weights=prob_list)[0]
        state.apply_action(action)
    else:
        legal_actions = state.legal_actions()
        action = choose_action(legal_actions)  # Your strategy here
        state.apply_action(action)

returns = state.returns()  # Final chip counts
```

### Betting Abstractions

OpenSpiel supports four betting abstractions that limit the action space:

| Abstraction | Actions Available | Use Case |
|-------------|-------------------|----------|
| `fc` | Fold, Call | Simplest, research only |
| `fcpa` | Fold, Call, Pot bet, All-in | **Default**, most common |
| `fchpa` | Fold, Call, Half-pot, Pot bet, All-in | Extended version |
| `fullgame` | Fold, Call, Any bet size | No abstraction, full granularity |

**Important:** Even when using abstractions (fcpa/fchpa), the actual bet sizes are stored in the information state tensor's sizing section.

### Information State Tensor Structure

**Critical for ML training:** The tensor has 5 sections:

1. **Player ID** (first `num_players` values): One-hot encoding
2. **Private cards** (deck_size bits): Your hole cards
3. **Public cards** (deck_size bits): Board cards
4. **Action sequence abstracted** (max_game_length × 2 bits): Binary encoding of actions
5. **Bet sizes** (last `max_game_length` values): **Actual bet amounts in chips**

**Key insight:** Section 5 stores the exact bet sizes even when using abstractions. For example:
- With FCPA, choosing "pot bet" might result in a 250 chip bet
- The tensor records `250.0` in the sizing section
- This allows neural networks to learn from precise pot odds

**Accessing bet sizes:**
```python
tensor = state.information_state_tensor()
bet_sizes = tensor[-game.max_game_length():]  # Last N values
```

### TensorAnalyzer Class

The `test_tensor_bet_sizes.py` file contains a `TensorAnalyzer` helper class that parses information state tensors into their component sections. This is useful for debugging and understanding tensor contents.

## Key Configuration Parameters

### Asymmetric Stacks
Players can have different stack sizes using space-separated values:
```python
'stack': '500 1000 2000'  # Player 0: 500, Player 1: 1000, Player 2: 2000
```
Side pots are handled automatically.

### Antes (Workaround)
There is **no separate ante parameter**. Simulate antes using the `blind` parameter:
```python
'blind': '10 10 10'  # All players post 10 chip "ante"
```

### Known Limitations
- **Rake:** NOT supported. The game is strictly zero-sum (`sum(returns) == 0` always).
- **Antes:** No dedicated parameter; must use blind workaround.
- **Max players:** 10 (hardcoded in OpenSpiel)

## File Organization

- **`holdem_example.py`**: Basic examples showing how to create games and simulate hands
- **`test_poker_configs.py`**: Comprehensive test suite for game configurations (asymmetric stacks, abstractions, antes, limitations)
- **`test_tensor_bet_sizes.py`**: Detailed test suite proving that actual bet sizes are stored in tensors (18 test assertions)
- **`README.md`**: User-facing documentation with quick start guide and parameter reference

## Development Notes

### Adding New Tests

When creating new test scripts:
1. Always include the shebang and virtual environment activation reminder in the docstring
2. Use `pyspiel.load_game('universal_poker', config_dict)` pattern
3. Handle both chance nodes and decision nodes in simulation loops
4. Test assertions should verify specific behaviors (see `test_tensor_bet_sizes.py` for examples)

### Parameter Types

**Critical:** OpenSpiel is strict about parameter types:
- Numeric parameters must be `int` or `float`, NOT strings
- Multi-value parameters (blind, stack, etc.) must be space-separated strings
- Incorrect: `'numPlayers': '2'` (string)
- Correct: `'numPlayers': 2` (integer)

### Debugging Game States

Useful methods for debugging:
```python
state.current_player()           # Who acts next
state.legal_actions()            # Available actions
state.action_to_string(p, a)     # Human-readable action description
state.is_terminal()              # Is hand over?
state.returns()                  # Final chip counts
state.history()                  # List of all actions taken
state.information_state_tensor() # Full tensor representation
```

### External Resources

- OpenSpiel installation: `/home/nuclearfalcon/open_spiel`
- OpenSpiel source code: `/home/nuclearfalcon/open_spiel/open_spiel/games/universal_poker/`
- Universal poker implementation: `universal_poker.cc` and `universal_poker.h`
- Test examples: `/home/nuclearfalcon/open_spiel/open_spiel/games/universal_poker/universal_poker_test.cc`
