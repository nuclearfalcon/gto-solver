"""
Quick test script for Phase 8.5 - Check game tree sizes before full solve

Run this to verify the config works and check tree sizes before running
the full expensive test suite.

Usage:
    source ~/open_spiel/venv/bin/activate
    python quick_test_phase8_5.py
"""

from matrix_cfr.subgame_solver import SubgameSolver, ChunkedSolver, BlueprintPolicy
from matrix_cfr.game_to_matrix import GameTreeConverter
import pyspiel

def get_minimal_4round_config():
    """Same config as test_phase8_5_full_pipeline.py"""
    return {
        "betting": "nolimit",
        "numPlayers": 2,
        "numRounds": 4,
        "blind": "50 100",
        "firstPlayer": "2 1 1 1",
        "numSuits": 2,
        "numRanks": 4,  # 8 cards total
        "numHoleCards": 1,
        "numBoardCards": "0 1 1 1",  # Cumulative: 0, 1, 2, 3
        "stack": "1000 1000",
        "bettingAbstraction": "fcpa"
    }

print("=" * 80)
print("QUICK TEST: Phase 8.5 Config Validation")
print("=" * 80)

config = get_minimal_4round_config()

print("\n1. Testing subgame config generation...")
print("-" * 80)
for round_name in ['preflop', 'flop', 'turn', 'river']:
    solver = SubgameSolver(config, round_name, blueprint_policy=None)
    board_cards = solver.subgame_config['numBoardCards']
    print(f"  {round_name:8} → {board_cards} board cards")

print("\n2. Checking game tree sizes for all chunks...")
print("-" * 80)
print(f"{'Round':<10} {'Nodes':>10} {'Infosets':>10} {'Board Cards':>12}")
print("-" * 80)

for round_name in ['preflop', 'flop', 'turn', 'river']:
    solver = SubgameSolver(config, round_name, blueprint_policy=None)
    game = pyspiel.load_game('universal_poker', solver.subgame_config)

    # Build matrix representation to count nodes
    converter = GameTreeConverter(game)
    matrix_repr = converter.build_matrices()

    num_nodes = matrix_repr.num_nodes
    num_infosets = len(matrix_repr.infoset_to_actions)
    num_board = solver.subgame_config['numBoardCards']

    print(f"{round_name:<10} {num_nodes:>10,} {num_infosets:>10,} {num_board:>12}")

print("\n3. Card accounting check...")
print("-" * 80)
total_deck = config['numSuits'] * config['numRanks']
hole_cards = config['numPlayers'] * config['numHoleCards']
board_cards = sum(int(x) for x in config['numBoardCards'].split())
total_needed = hole_cards + board_cards

print(f"  Deck size:     {total_deck} cards ({config['numSuits']} suits × {config['numRanks']} ranks)")
print(f"  Hole cards:    {hole_cards} cards ({config['numPlayers']} players × {config['numHoleCards']})")
print(f"  Board cards:   {board_cards} cards (total across all rounds)")
print(f"  Total needed:  {total_needed} cards")
print(f"  Remaining:     {total_deck - total_needed} cards")

if total_needed <= total_deck:
    print(f"\n  ✓ Config is valid (enough cards)")
else:
    print(f"\n  ✗ Config is INVALID (not enough cards!)")

print("\n4. Testing a quick 2-iteration solve on preflop...")
print("-" * 80)
preflop_solver = SubgameSolver(config, 'preflop', blueprint_policy=None)
print("  Solving preflop (2 iterations)...")
policy = preflop_solver.solve(iterations=2, progress_interval=999)
print(f"  ✓ Preflop solved: {len(policy.policy)} infosets")

print("\n" + "=" * 80)
print("✅ QUICK TEST COMPLETE!")
print("=" * 80)
print("\nIf all checks passed, you can run the full test suite:")
print("  python test_phase8_5_full_pipeline.py")
print()
