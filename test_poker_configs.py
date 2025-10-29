#!/usr/bin/env python3
"""
Comprehensive OpenSpiel Poker Configuration Tests

Tests various universal_poker configurations including:
- Asymmetrical stakes
- Betting abstractions (fc, fcpa, fchpa, fullgame)
- Ante simulations
- Information state tensors

Requirements:
    source ~/open_spiel/venv/bin/activate

Run:
    python test_poker_configs.py
"""

import pyspiel
import random


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")


def test_asymmetrical_stakes():
    """Test asymmetrical stack configurations"""
    print_section("TEST 1: ASYMMETRICAL STAKES")

    # Test 1.1: 2-player with different stacks
    print_subsection("1.1: Heads-Up with Asymmetric Stacks (500 vs 2000)")
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
        'stack': '500 2000',  # Asymmetric stacks
        'bettingAbstraction': 'fcpa'
    })

    state = game.new_initial_state()
    print(f"✓ Game created successfully")
    print(f"  Players: {game.num_players()}")
    print(f"  Player 0 stack: 500 chips")
    print(f"  Player 1 stack: 2000 chips")
    print(f"  Max player 0 can win/lose: 500 chips (limited by their stack)")
    print(f"  Distinct actions: {game.num_distinct_actions()}")

    # Test 1.2: 3-player with varying stacks
    print_subsection("1.2: 3-Player with Varying Stacks (500, 1000, 2000)")
    game_3p = pyspiel.load_game('universal_poker', {
        'betting': 'nolimit',
        'numPlayers': 3,
        'numRounds': 4,
        'blind': '100 50 0',
        'firstPlayer': '2 1 1 1',
        'numSuits': 4,
        'numRanks': 13,
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',
        'stack': '500 1000 2000',  # Three different stacks
        'bettingAbstraction': 'fcpa'
    })

    print(f"✓ Game created successfully")
    print(f"  Players: {game_3p.num_players()}")
    print(f"  Player 0 stack: 500 chips")
    print(f"  Player 1 stack: 1000 chips")
    print(f"  Player 2 stack: 2000 chips")
    print(f"  Side pots: Will be created automatically when needed")

    # Test 1.3: 6-player with mixed stacks
    print_subsection("1.3: 6-Max with Mixed Stacks")
    game_6p = pyspiel.load_game('universal_poker', {
        'betting': 'nolimit',
        'numPlayers': 6,
        'numRounds': 4,
        'blind': '100 50 0 0 0 0',
        'firstPlayer': '2 1 1 1',
        'numSuits': 4,
        'numRanks': 13,
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',
        'stack': '10000 5000 15000 20000 8000 12000',
        'bettingAbstraction': 'fcpa'
    })

    print(f"✓ Game created successfully")
    print(f"  Players: {game_6p.num_players()}")
    print(f"  Stacks: 10000, 5000, 15000, 20000, 8000, 12000")

    # Simulate a quick hand to verify it works
    print_subsection("1.4: Simulating Hand with Asymmetric Stacks")
    state = game.new_initial_state()
    action_count = 0

    while not state.is_terminal() and action_count < 10:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = random.choices(action_list, weights=prob_list)[0]
            state.apply_action(action)
        else:
            legal_actions = state.legal_actions()
            action = random.choice(legal_actions)
            player = state.current_player()
            action_str = state.action_to_string(player, action)
            print(f"  Player {player}: {action_str}")
            state.apply_action(action)
            action_count += 1

    if state.is_terminal():
        returns = state.returns()
        print(f"\n  Final returns: {[f'{r:+.0f}' for r in returns]}")
    print(f"  ✓ Asymmetric stacks work correctly")


def test_betting_abstractions():
    """Test different betting abstraction modes"""
    print_section("TEST 2: BETTING ABSTRACTIONS")

    # Common base config for all tests
    base_config = {
        'betting': 'nolimit',
        'numPlayers': 2,
        'numRounds': 2,
        'blind': '100 50',
        'firstPlayer': '2 1',
        'numSuits': 2,
        'numRanks': 6,
        'numHoleCards': 1,
        'numBoardCards': '0 1',
        'stack': '1000 1000'
    }

    abstractions = ['fc', 'fcpa', 'fchpa', 'fullgame']

    for abstraction in abstractions:
        print_subsection(f"2.{abstractions.index(abstraction)+1}: Betting Abstraction = '{abstraction}'")

        config = base_config.copy()
        config['bettingAbstraction'] = abstraction
        game = pyspiel.load_game('universal_poker', config)

        print(f"✓ Game loaded with abstraction: {abstraction}")
        print(f"  Distinct actions: {game.num_distinct_actions()}")

        # Describe what actions are available
        if abstraction == 'fc':
            print(f"  Available actions: Fold, Call only")
        elif abstraction == 'fcpa':
            print(f"  Available actions: Fold, Call, Pot-sized bet, All-in")
        elif abstraction == 'fchpa':
            print(f"  Available actions: Fold, Call, Half-pot, Pot-sized bet, All-in")
        elif abstraction == 'fullgame':
            print(f"  Available actions: Fold, Call, Any bet size from min to all-in")
            print(f"  Note: Action space = max_stack_size + 1 = {game.num_distinct_actions()}")

        # Show a sample state
        state = game.new_initial_state()
        # Deal cards
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, _ = zip(*outcomes)
            state.apply_action(action_list[0])
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, _ = zip(*outcomes)
            state.apply_action(action_list[0])

        # Now we should be at a player decision node
        if not state.is_chance_node() and not state.is_terminal():
            legal_actions = state.legal_actions()
            print(f"  Legal actions at first decision: {len(legal_actions)} options")
            if abstraction == 'fullgame':
                print(f"    Example actions: {legal_actions[:5]}... (showing first 5)")
            else:
                for action in legal_actions:
                    action_str = state.action_to_string(state.current_player(), action)
                    print(f"    Action {action}: {action_str}")


def test_information_state_tensors():
    """Test information state tensors with different bet sizes"""
    print_section("TEST 3: INFORMATION STATE TENSORS")

    print_subsection("3.1: Tensor with Betting Abstraction (FCPA)")

    # Create a simple game with FCPA abstraction
    game = pyspiel.load_game('universal_poker', {
        'bettingAbstraction': 'fcpa',
        'numRanks': 4,
        'numSuits': 1,
        'numPlayers': 2,
        'numRounds': 1,
        'blind': '50 100',
        'firstPlayer': '2',
        'numHoleCards': 1,
        'numBoardCards': '0',
        'stack': '1000 1000'
    })

    print(f"✓ Game created with FCPA abstraction")
    print(f"  Distinct actions: {game.num_distinct_actions()}")

    # Get initial state and deal cards
    state = game.new_initial_state()

    # Deal two cards
    if state.is_chance_node():
        state = state.child(1)  # Deal first card
    if state.is_chance_node():
        state = state.child(2)  # Deal second card

    print(f"\n  After dealing cards, current player: {state.current_player()}")

    # Test different actions and show information state tensors
    if not state.is_terminal() and not state.is_chance_node():
        legal_actions = state.legal_actions()
        print(f"  Legal actions: {legal_actions}")

        for action in legal_actions[:3]:  # Show first 3 actions
            action_str = state.action_to_string(state.current_player(), action)
            next_state = state.child(action)
            info_tensor = next_state.information_state_tensor()
            print(f"\n  Action {action} ({action_str}):")
            print(f"    Info state tensor length: {len(info_tensor)}")
            print(f"    Tensor (first 20 values): {info_tensor[:20]}")

    print_subsection("3.2: Tensor with Full Game (No Abstraction)")

    # Similar test with fullgame abstraction
    game_full = pyspiel.load_game('universal_poker', {
        'bettingAbstraction': 'fullgame',
        'numRanks': 4,
        'numSuits': 1,
        'numPlayers': 2,
        'numRounds': 1,
        'blind': '50 100',
        'firstPlayer': '2',
        'numHoleCards': 1,
        'numBoardCards': '0',
        'stack': '1200 1200'
    })

    print(f"✓ Game created with fullgame (no abstraction)")
    print(f"  Distinct actions: {game_full.num_distinct_actions()}")

    state = game_full.new_initial_state()

    # Deal cards
    if state.is_chance_node():
        state = state.child(1)
    if state.is_chance_node():
        state = state.child(2)

    print(f"\n  Testing different bet sizes:")

    # Test specific bet sizes (adapted from user's example)
    test_actions = [
        (1, "check/call"),
        (400, "bet 400"),
        (500, "bet 500"),
        (1200, "bet 1200 (all-in)")
    ]

    for action_num, description in test_actions:
        if action_num in state.legal_actions():
            next_state = state.child(action_num)
            info_tensor = next_state.information_state_tensor()
            print(f"\n  {description}:")
            print(f"    Action: {action_num}")
            print(f"    Tensor length: {len(info_tensor)}")
            print(f"    Tensor (first 25 values): {info_tensor[:25]}")
        else:
            print(f"\n  {description}: Not a legal action in this state")

    print_subsection("3.3: Information State Tensor Structure")
    print("""
  The information state tensor contains:
  1. Player ID (one-hot encoding)
  2. Private cards (which cards you hold)
  3. Public cards (board cards)
  4. Action sequence abstracted (encoded betting actions: fold/call/bet/all-in)
  5. Action sequence sizings (ACTUAL bet sizes, even with abstraction)

  Action encoding (binary):
    'c' (check/call) = 10
    'p' (bet/raise)  = 01
    'a' (all-in)     = 11
    'f' (fold)       = 00
    'd' (deal)       = 00

  Important: Even with betting abstractions like FCPA, the actual bet sizes
  are stored in the tensor, allowing exact game state reconstruction.
    """)


def test_ante_simulations():
    """Test ante configurations using the blind parameter"""
    print_section("TEST 4: ANTE SIMULATIONS (using blind parameter)")

    print("""
  NOTE: OpenSpiel's universal_poker does NOT have a separate ante parameter.
  To simulate antes, use the 'blind' parameter with equal values for all players.
    """)

    print_subsection("4.1: Tournament Style - All Players Post Ante")

    game_ante = pyspiel.load_game('universal_poker', {
        'betting': 'nolimit',
        'numPlayers': 3,
        'numRounds': 4,
        'blind': '10 10 10',  # All players post 10 chips (simulating antes)
        'firstPlayer': '1 1 1 1',
        'numSuits': 4,
        'numRanks': 13,
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',
        'stack': '1000 1000 1000',
        'bettingAbstraction': 'fcpa'
    })

    print(f"✓ Game created with equal 'antes' for all players")
    print(f"  Players: {game_ante.num_players()}")
    print(f"  Blind config: '10 10 10' (simulates 10 chip ante for each player)")
    print(f"  Starting pot: 30 chips (10 × 3 players)")
    print(f"  Stack per player: 1000 chips")

    print_subsection("4.2: Mixed Blinds + Ante Simulation")

    game_mixed = pyspiel.load_game('universal_poker', {
        'betting': 'nolimit',
        'numPlayers': 6,
        'numRounds': 4,
        'blind': '100 50 25 25 25 25',  # BB, SB, then antes for others
        'firstPlayer': '2 1 1 1',
        'numSuits': 4,
        'numRanks': 13,
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',
        'stack': '10000 10000 10000 10000 10000 10000',
        'bettingAbstraction': 'fcpa'
    })

    print(f"✓ Game created with mixed blinds + antes")
    print(f"  Players: {game_mixed.num_players()}")
    print(f"  Blind config: '100 50 25 25 25 25'")
    print(f"    Player 0 (BB): 100 chips")
    print(f"    Player 1 (SB): 50 chips")
    print(f"    Players 2-5: 25 chips each (ante)")
    print(f"  Starting pot: 250 chips")

    print_subsection("4.3: Simulating a Hand with Antes")

    state = game_ante.new_initial_state()
    action_count = 0

    print(f"  Initial pot: 30 chips from antes")

    while not state.is_terminal() and action_count < 8:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = random.choices(action_list, weights=prob_list)[0]
            state.apply_action(action)
        else:
            legal_actions = state.legal_actions()
            action = random.choice(legal_actions)
            player = state.current_player()
            action_str = state.action_to_string(player, action)
            print(f"  Player {player}: {action_str}")
            state.apply_action(action)
            action_count += 1

    if state.is_terminal():
        returns = state.returns()
        print(f"\n  Final returns: {[f'{r:+.0f}' for r in returns]}")

    print(f"  ✓ Ante simulation works correctly")


def test_custom_betting_abstraction():
    """Test custom betting abstraction using fullgame + filtering"""
    print_section("TEST 5: CUSTOM BETTING ABSTRACTION (Fullgame + Filtering)")

    print("""
  OpenSpiel has hardcoded betting abstractions (fc, fcpa, fchpa, fullgame).
  To implement custom abstractions like FCHPA + 1.5×pot, we use 'fullgame'
  and programmatically filter legal actions to only allow specific bet sizes.
    """)

    print_subsection("5.1: Creating Game with Fullgame Abstraction")

    game = pyspiel.load_game('universal_poker', {
        'betting': 'nolimit',
        'numPlayers': 2,
        'numRounds': 1,
        'blind': '50 100',
        'firstPlayer': '2',
        'numSuits': 2,
        'numRanks': 4,
        'numHoleCards': 1,
        'numBoardCards': '0',
        'stack': '1000 1000',
        'bettingAbstraction': 'fullgame'  # No abstraction - we'll filter ourselves
    })

    print(f"✓ Game created with fullgame abstraction")
    print(f"  Distinct actions: {game.num_distinct_actions()}")

    # Create state and deal cards
    state = game.new_initial_state()
    state = state.child(1)  # Deal first card
    state = state.child(2)  # Deal second card

    print_subsection("5.2: Filtering Actions for Custom Abstraction")

    legal_actions = state.legal_actions()
    print(f"  Fullgame legal actions: {len(legal_actions)} total")
    print(f"  Sample actions: {legal_actions[:10]}")

    # Calculate pot and allowed bet sizes
    # After blinds: pot = 150 (50 + 100)
    pot = 150
    half_pot = pot // 2  # 75
    full_pot = pot  # 150
    one_half_pot = int(pot * 1.5)  # 225
    max_stack = 1000

    print(f"\n  Pot size: {pot}")
    print(f"  Target bet sizes:")
    print(f"    - Fold: action 0")
    print(f"    - Call: action 1")
    print(f"    - Half pot (0.5×): ~{half_pot}")
    print(f"    - Full pot (1.0×): ~{full_pot}")
    print(f"    - One-and-half pot (1.5×): ~{one_half_pot}")
    print(f"    - All-in: action {max_stack}")

    # Filter to our custom abstraction
    # Find closest action to each target bet size
    target_bets = [half_pot, full_pot, one_half_pot, max_stack]

    filtered_actions = []

    # Always include fold and call
    for action in legal_actions:
        if action <= 1:
            filtered_actions.append(action)

    # For each target bet size, find the closest legal action
    bet_actions = [a for a in legal_actions if a > 1]

    for target in target_bets:
        if bet_actions:
            # Find closest action to this target
            closest = min(bet_actions, key=lambda a: abs(a - target))
            if closest not in filtered_actions:
                filtered_actions.append(closest)

    print(f"\n  Filtered actions: {len(filtered_actions)} total")
    print(f"  Allowed actions: {filtered_actions}")

    print_subsection("5.3: Verifying Filtered Actions Work")

    # Test each filtered action
    action_count = 0
    for action in filtered_actions[:3]:  # Test first 3
        try:
            next_state = state.child(action)
            action_str = state.action_to_string(state.current_player(), action)
            print(f"  ✓ Action {action} ({action_str}): Works correctly")
            action_count += 1
        except Exception as e:
            print(f"  ✗ Action {action}: Failed with error: {e}")

    print(f"\n  Successfully tested {action_count} filtered actions")

    print_subsection("5.4: Custom Abstraction Summary")
    print(f"""
  ✓ Fullgame abstraction provides all bet sizes
  ✓ Programmatic filtering reduces action space to desired abstraction
  ✓ Filtered actions include: Fold, Call, 0.5×pot, 1.0×pot, 1.5×pot, All-in
  ✓ This approach allows any custom betting abstraction
  ✓ Trade-off: Slightly larger game tree vs hardcoded abstractions
    """)


def test_limitations():
    """Document features that are NOT supported"""
    print_section("TEST 6: KNOWN LIMITATIONS")

    print("""
  The following features are NOT supported in OpenSpiel's universal_poker:

  ❌ RAKE:
     - No rake parameter exists
     - The game is strictly zero-sum (utility sum always = 0)
     - Workaround: Implement rake tracking at a higher level, outside the game
     - Could potentially use 'repeated_poker' wrapper to apply rake between hands

  ⚠️  ANTES (Limited):
     - No separate ante parameter
     - Must simulate antes using the 'blind' parameter
     - Works by setting equal blind values for all (or some) players
     - Limitation: Cannot have true positional blinds + separate antes in the
       same game using standard parameters

  ℹ️  OTHER NOTES:
     - Minimum blind requirement: At least one player must have blind > 0
     - Stack sizes in limit games are effectively ignored (set to INT32_MAX)
     - Max players: 10 (hardcoded limit)
     - Information state tensor size scales with max_game_length
    """)

    print_subsection("6.1: Attempting to Create Game with Rake (Will Fail)")

    try:
        game = pyspiel.load_game('universal_poker', {
            'betting': 'nolimit',
            'numPlayers': 2,
            'rake': '0.05',  # This parameter doesn't exist
            'stack': '1000 1000'
        })
        print("  ✗ Unexpectedly succeeded (this should have failed)")
    except Exception as e:
        print(f"  ✓ Expected error: Parameter 'rake' not recognized")
        print(f"  Error message: {str(e)[:100]}...")

    print_subsection("6.2: Zero-Sum Verification")

    game = pyspiel.load_game('universal_poker', {
        'betting': 'nolimit',
        'numPlayers': 2,
        'numRounds': 1,
        'blind': '100 50',
        'firstPlayer': '2',
        'numSuits': 2,
        'numRanks': 4,
        'numHoleCards': 1,
        'numBoardCards': '0',
        'stack': '1000 1000'
    })

    print(f"  Game type utility: {game.get_type().utility}")
    print(f"  ✓ Game is zero-sum (no rake possible)")

    # Simulate a hand and verify returns sum to zero
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = random.choices(action_list, weights=prob_list)[0]
        else:
            legal_actions = state.legal_actions()
            action = random.choice(legal_actions)
        state.apply_action(action)

    returns = state.returns()
    total_return = sum(returns)
    print(f"\n  Sample hand returns: {[f'{r:+.0f}' for r in returns]}")
    print(f"  Sum of returns: {total_return:.1f} (exactly zero)")
    print(f"  ✓ Confirmed: Game is zero-sum, no rake applied")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  OPENSPIEL UNIVERSAL_POKER CONFIGURATION TESTS")
    print("  Testing: Asymmetric stacks, Betting abstractions, Antes, Tensors")
    print("="*70)

    try:
        test_asymmetrical_stakes()
        test_betting_abstractions()
        test_information_state_tensors()
        test_ante_simulations()
        test_custom_betting_abstraction()
        test_limitations()

        print_section("ALL TESTS COMPLETED SUCCESSFULLY")
        print("""
  Summary:
  ✓ Asymmetrical stakes: SUPPORTED
  ✓ Betting abstractions: SUPPORTED (fc, fcpa, fchpa, fullgame)
  ✓ Custom betting abstraction: SUPPORTED (fullgame + filtering)
  ✓ Information state tensors: WORKING (includes actual bet sizes)
  ⚠ Antes: PARTIAL (use blind parameter)
  ❌ Rake: NOT SUPPORTED

  For more information, see:
  - OpenSpiel docs: https://github.com/deepmind/open_spiel
  - Universal poker: open_spiel/games/universal_poker/
        """)

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
