# Preflop GTO Training Problems

This system extracts preflop poker training problems from trained CFR policies and exports them as structured JSON data for use in external training applications.

## Overview

The GTO problem extraction system allows you to:
1. Train CFR policies on poker games (2-9 players)
2. Extract preflop decision points from the trained policies
3. Query GTO strategies (action frequencies) for each decision point
4. Filter and categorize problems by position, tags, complexity
5. Export to JSON for integration with web apps or training tools

## Quick Start

### 1. Train a Policy

First, train a CFR policy on a poker configuration:

```bash
source ~/open_spiel/venv/bin/activate

# Train a 6-max policy (10BB stacks)
python solve_poker.py \
  --config configs/6p_10bb_holdem.json \
  --algorithm external_mccfr \
  --iterations 100000 \
  --output results/6p_10bb_policy.pkl
```

### 2. Extract Problems

Extract preflop problems from the trained policy:

```bash
# Extract all preflop problems
python extract_preflop_problems.py \
  --policy results/6p_10bb_policy.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/6p_all.json

# Extract only close decisions (mixed strategies)
python extract_preflop_problems.py \
  --policy results/6p_10bb_policy.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/6p_close.json \
  --close-only

# Extract problems for specific position
python extract_preflop_problems.py \
  --policy results/6p_10bb_policy.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/6p_btn.json \
  --position BTN
```

### 3. Use in Your Application

Load the JSON file in your web app or training tool and present problems to users.

## System Architecture

### Core Components

1. **`position_utils.py`** - Position mapping for 2-9 players
   - Maps player IDs to poker positions (BTN, BB, UTG, etc.)
   - Handles action order and position categories

2. **`gto_problem.py`** - Problem data structure
   - `PreflopProblem` dataclass with all game state info
   - GTO strategy with action frequencies
   - JSON serialization/deserialization
   - Helper methods for analysis

3. **`preflop_problem_extractor.py`** - Extraction engine
   - Loads and parses CFR policies
   - Extracts preflop decision points
   - Queries GTO strategies from policies
   - Auto-tags problems (facing_raise, close_decision, etc.)

4. **`extract_preflop_problems.py`** - CLI tool
   - Command-line interface for extraction
   - Filtering and statistics
   - Batch processing

## Available Configurations

Prebuilt configs for training GTO policies:

| Config | Players | Stacks | Description |
|--------|---------|--------|-------------|
| `configs/2p_5bb_holdem.json` | 2 | 5BB | Heads-up, short stack |
| `configs/3p_5bb_holdem.json` | 3 | 5BB | 3-handed, short stack |
| `configs/6p_10bb_holdem.json` | 6 | 10BB | 6-max (most popular online) |
| `configs/9p_10bb_holdem.json` | 9 | 10BB | Full ring |

All configs use:
- **Full 52-card deck** (4 suits, 13 ranks)
- **FCPA betting abstraction** (Fold, Call, Pot bet, All-in)
- **Standard blinds** (100 big blind, 50 small blind)

## JSON Format Specification

Each problem is exported as a JSON object with the following structure:

```json
{
  "problem_id": "6p_preflop_00123",
  "num_players": 6,
  "hero_position": "CO",
  "hero_cards": ["Ah", "Kd"],
  "stacks_bb": {
    "UTG": 10.0,
    "MP": 10.0,
    "CO": 10.0,
    "BTN": 10.0,
    "SB": 9.5,
    "BB": 9.0
  },
  "pot_bb": 1.5,
  "action_history": [
    {"player": "UTG", "action": "Fold"},
    {"player": "MP", "action": "Raise to 2.5BB"}
  ],
  "active_players": ["MP", "CO", "BTN", "SB", "BB"],
  "current_player": "CO",
  "gto_strategy": {
    "Fold": 0.60,
    "Call/Check": 0.15,
    "Bet/Raise (Pot)": 0.20,
    "All-in": 0.05
  },
  "tags": ["facing_raise", "mp_open", "close_decision"],
  "hand_category": "broadway_offsuit",
  "info_state_str": "[Round 0][Player: 2]..."
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `problem_id` | string | Unique identifier |
| `num_players` | int | Number of players (2-9) |
| `hero_position` | string | Hero's position (BTN, BB, UTG, etc.) |
| `hero_cards` | array | Hero's hole cards (e.g., ["As", "Kh"]) |
| `stacks_bb` | object | Stack sizes in big blinds for each position |
| `pot_bb` | float | Current pot size in big blinds |
| `action_history` | array | Actions taken before hero's decision |
| `active_players` | array | Players still in the hand (haven't folded) |
| `current_player` | string | Position of player to act (should be hero) |
| `gto_strategy` | object | GTO mixed strategy (action → frequency) |
| `tags` | array | Categorization tags |
| `hand_category` | string | Hand strength category |
| `info_state_str` | string | Original OpenSpiel state string (optional) |

## Position Names

### 2-Player (Heads-up)
- **BTN** - Button (also small blind, acts first preflop)
- **BB** - Big blind

### 3-Player
- **BTN** - Button
- **SB** - Small blind
- **BB** - Big blind

### 6-Player (6-max)
- **UTG** - Under the gun (first to act)
- **MP** - Middle position
- **CO** - Cutoff
- **BTN** - Button (dealer)
- **SB** - Small blind
- **BB** - Big blind

### 9-Player (Full Ring)
- **UTG** - Under the gun
- **UTG+1** - Under the gun + 1
- **UTG+2** - Under the gun + 2
- **MP** - Middle position
- **MP+1** - Middle position + 1
- **CO** - Cutoff
- **BTN** - Button
- **SB** - Small blind
- **BB** - Big blind

## Problem Tags

Problems are automatically tagged for easy filtering:

### Round Tags
- `round_0` - Round 0 (always present for preflop)
- `preflop` - Preflop decision point

### Action Tags
- `facing_raise` - Hero is facing a raise
- `facing_3bet` - Hero is facing a 3-bet
- `multiway` - 3+ players still in hand

### Decision Complexity
- `close_decision` - 2+ actions with ≥15% frequency (mixed strategy)

### Pot Size
- `small_pot` - Pot ≤ 3BB
- `large_pot` - Pot ≥ 10BB
- (No tag = medium pot)

## CLI Tool Usage

### Basic Commands

```bash
# Show statistics without saving
python extract_preflop_problems.py \
  --policy results/policy.pkl \
  --config configs/6p_10bb_holdem.json \
  --stats-only

# Extract all problems
python extract_preflop_problems.py \
  --policy results/policy.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/all.json

# Limit number of problems
python extract_preflop_problems.py \
  --policy results/policy.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/sample.json \
  --max 100
```

### Filtering

```bash
# Extract only close decisions
python extract_preflop_problems.py \
  --policy results/policy.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/close.json \
  --close-only

# Extract for specific position
python extract_preflop_problems.py \
  --policy results/policy.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/btn.json \
  --position BTN

# Extract problems with specific tag
python extract_preflop_problems.py \
  --policy results/policy.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/facing_raise.json \
  --filter-tag facing_raise
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--policy PATH` | Path to policy pickle file (required) |
| `--config PATH` | Path to game config JSON file (required) |
| `--output PATH` | Output JSON file path |
| `--max N` | Maximum number of problems to extract |
| `--filter-tag TAG` | Only include problems with this tag |
| `--position POS` | Only include problems for this position |
| `--close-only` | Only include close decisions (2+ viable actions) |
| `--stats-only` | Show statistics without saving |
| `--samples N` | Number of sample problems to show (default: 5) |

## Training Policies

### Recommended Algorithm

For 3+ players, use **External Sampling MCCFR with FULL averaging**:

```bash
python solve_poker.py \
  --config configs/6p_10bb_holdem.json \
  --algorithm external_mccfr \
  --iterations 100000
```

### Training Time Estimates

| Config | Iterations | Time | Exploitability Check |
|--------|-----------|------|---------------------|
| 2p_5bb | 50k-100k | 1-3 hours | Every 10k iterations |
| 3p_5bb | 100k-200k | 3-8 hours | Every 20k iterations |
| 6p_10bb | 200k-500k | 8-24 hours | Every 50k iterations |
| 9p_10bb | 500k-1M | 1-3 days | Every 100k iterations |

Times are approximate and depend on hardware.

### Memory Considerations

- Use **sampled exploitability** (default) for all configs
- Full exploitability causes OOM errors on large games
- The solver automatically uses sampled exploitability during training

## Integration Guide

### Loading Problems in Python

```python
import json
from gto_problem import PreflopProblem

# Load problems from JSON
with open('problems/6p_all.json', 'r') as f:
    problems_data = json.load(f)

# Convert to PreflopProblem objects
problems = [PreflopProblem.from_dict(p) for p in problems_data]

# Access problem data
for problem in problems[:5]:
    print(f"Position: {problem.hero_position}")
    print(f"Cards: {problem.hero_cards}")
    print(f"GTO: {problem.format_gto_strategy()}")
    print()
```

### Loading Problems in JavaScript/TypeScript

```javascript
// Load JSON file
const response = await fetch('/problems/6p_all.json');
const problems = await response.json();

// Display a problem to the user
function displayProblem(problem) {
  console.log(`Position: ${problem.hero_position}`);
  console.log(`Cards: ${problem.hero_cards.join('')}`);
  console.log(`Pot: ${problem.pot_bb}BB`);

  // Show action history
  problem.action_history.forEach(action => {
    console.log(`${action.player}: ${action.action}`);
  });

  // Show GTO strategy
  for (const [action, freq] of Object.entries(problem.gto_strategy)) {
    console.log(`${action}: ${(freq * 100).toFixed(1)}%`);
  }
}
```

## Example Workflow

### Complete Training & Extraction Pipeline

```bash
# 1. Activate environment
source ~/open_spiel/venv/bin/activate

# 2. Train a 6-max policy
python solve_poker.py \
  --config configs/6p_10bb_holdem.json \
  --algorithm external_mccfr \
  --iterations 200000 \
  --checkpoint-interval 50000

# 3. Extract all problems
python extract_preflop_problems.py \
  --policy results/external_mccfr_6p_10bb_*.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/6p_all.json

# 4. Extract filtered subsets
python extract_preflop_problems.py \
  --policy results/external_mccfr_6p_10bb_*.pkl \
  --config configs/6p_10bb_holdem.json \
  --output problems/6p_btn_close.json \
  --position BTN \
  --close-only

# 5. Generate statistics
python extract_preflop_problems.py \
  --policy results/external_mccfr_6p_10bb_*.pkl \
  --config configs/6p_10bb_holdem.json \
  --stats-only
```

## Troubleshooting

### Common Issues

**"Policy file not found"**
- Ensure the policy path is correct
- Check that training completed successfully
- Policy files are saved in `results/` directory

**"Config file not found"**
- Config must exist in `configs/` directory
- Use one of the provided configs or create your own

**"No problems extracted"**
- Check that policy was trained (not empty)
- Verify config matches the policy's game parameters
- Try without filters first (`--stats-only`)

**"Memory error during training"**
- Reduce stack sizes (5BB instead of 10BB)
- Use fewer players (2p or 3p instead of 6p/9p)
- Sampled exploitability is enabled by default

### Getting Help

- Check `ARCHITECTURE.md` for system design details
- See `CLAUDE.md` for project overview
- Review example scripts in the repository

## Advanced Usage

### Custom Filtering

You can programmatically filter problems:

```python
from preflop_problem_extractor import PreflopProblemExtractor
from game_config import PokerGameConfig

# Load policy
config = PokerGameConfig.from_json('configs/6p_10bb_holdem.json')
extractor = PreflopProblemExtractor('results/policy.pkl', config)

# Extract all problems
all_problems = extractor.extract_problems()

# Custom filter: UTG opens facing no action
utg_opens = [p for p in all_problems
             if p.hero_position == 'UTG'
             and len(p.action_history) == 0]

# Save filtered set
extractor.save_problems(utg_opens, 'problems/utg_opens.json')
```

### Batch Processing

Process multiple policies:

```bash
#!/bin/bash
for policy in results/*_policy.pkl; do
  config=$(basename $policy .pkl | sed 's/_policy//')
  python extract_preflop_problems.py \
    --policy "$policy" \
    --config "configs/${config}.json" \
    --output "problems/${config}_problems.json"
done
```

## Next Steps

1. **Train policies** using the provided configs
2. **Extract problems** for your target player counts
3. **Integrate JSON data** into your web application
4. **Present problems** to users for GTO training
5. **Track user performance** and adapt problem difficulty

## File Reference

### Python Modules
- `position_utils.py` - Position mapping (2-9 players)
- `gto_problem.py` - Problem data structure
- `preflop_problem_extractor.py` - Extraction engine
- `extract_preflop_problems.py` - CLI tool

### Configuration Files
- `configs/2p_5bb_holdem.json` - Heads-up config
- `configs/3p_5bb_holdem.json` - 3-handed config
- `configs/6p_10bb_holdem.json` - 6-max config
- `configs/9p_10bb_holdem.json` - Full ring config

### Directories
- `problems/` - Exported problem JSON files
- `results/` - Trained policy files (.pkl)
- `configs/` - Game configuration files (.json)

## License

This project is part of the GTO Poker Training repository.
