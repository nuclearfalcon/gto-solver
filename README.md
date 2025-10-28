# GTO Poker Training with OpenSpiel

This project uses OpenSpiel to simulate and analyze No-Limit Hold'em poker games.

## OpenSpiel Installation

**OpenSpiel is already installed** at `/home/nuclearfalcon/open_spiel` (version 1.6.8)

All system dependencies are installed:
- Python 3.10.12
- git, cmake, build-essential
- clang and development tools
- Virtual environment with all required packages

## Quick Start

### 1. Activate the Virtual Environment

```bash
source ~/open_spiel/venv/bin/activate
```

### 2. Run the Example Script

```bash
python holdem_example.py
```

This will run three example simulations:
- Heads-up (2 players)
- 6-max table (6 players)
- 100 hands simulation with statistics

## Using OpenSpiel in Your Code

### Basic Import

```python
import pyspiel
```

### Create a Hold'em Game

```python
game = pyspiel.load_game('universal_poker', {
    'betting': 'nolimit',
    'numPlayers': 2,
    'numRounds': 4,
    'blind': '100 50',
    'firstPlayer': '2 1 1 1',
    'numSuits': 4,
    'numRanks': 13,
    'numHoleCards': 2,
    'numBoardCards': '0 3 1 1',
    'stack': '20000 20000'
})
```

### Simulate a Hand

```python
state = game.new_initial_state()

while not state.is_terminal():
    if state.is_chance_node():
        # Deal cards randomly
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = random.choices(action_list, weights=prob_list)[0]
        state.apply_action(action)
    else:
        # Player makes a decision
        legal_actions = state.legal_actions()
        action = choose_action(legal_actions)  # Your strategy here
        state.apply_action(action)

# Get final results
returns = state.returns()
```

## Game Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `betting` | Betting structure | `'nolimit'`, `'limit'`, `'potlimit'` |
| `numPlayers` | Number of players | `2` to `10` |
| `numRounds` | Number of betting rounds | `4` (preflop, flop, turn, river) |
| `blind` | Blind sizes per player | `'100 50'` (big, small) |
| `firstPlayer` | First to act each round | `'2 1 1 1'` |
| `numHoleCards` | Cards per player | `2` |
| `numBoardCards` | Board cards per round | `'0 3 1 1'` |
| `stack` | Starting stack per player | `'20000 20000'` |

## Available Poker Games in OpenSpiel

- `universal_poker` - Configurable poker game (use this for hold'em)
- `kuhn_poker` - Simple 3-card poker
- `leduc_poker` - 6-card poker variant

## Resources

- [OpenSpiel Documentation](https://github.com/deepmind/open_spiel)
- [Universal Poker Guide](https://github.com/deepmind/open_spiel/blob/master/docs/games/universal_poker.md)
- OpenSpiel Installation: `/home/nuclearfalcon/open_spiel`

## Verification

To verify your installation:

```bash
source ~/open_spiel/venv/bin/activate
python -c "import pyspiel; print('OpenSpiel version:', pyspiel.__version__)"
python -c "import pyspiel; game = pyspiel.load_game('universal_poker'); print('universal_poker works!')"
```

Expected output:
```
OpenSpiel version: 1.6.8
universal_poker works!
```
