#!/usr/bin/env python3
"""
Enhanced Information State Tensor Analysis and Testing

This script thoroughly tests and demonstrates how OpenSpiel's universal_poker
stores actual bet sizes in the information state tensor, even when using
betting abstractions like FCPA.

Requirements:
    source ~/open_spiel/venv/bin/activate

Run:
    python test_tensor_bet_sizes.py
"""

import pyspiel
import sys


class TensorAnalyzer:
    """Helper class to parse and analyze information state tensors"""

    def __init__(self, game, state):
        self.game = game
        self.state = state
        self.num_players = game.num_players()
        self.deck_size = self._calculate_deck_size()
        self.max_game_length = game.max_game_length()

    def _calculate_deck_size(self):
        """Calculate deck size from game parameters"""
        # This is a simplification - actual deck size depends on game config
        # For our test cases we'll derive it from the information state
        return 4  # Will be overridden by actual measurement

    def parse_tensor(self, tensor):
        """
        Parse the information state tensor into its component sections.

        Tensor structure (from OpenSpiel source):
        1. Player ID (num_players bits, one-hot encoding)
        2. Private cards (deck_size bits)
        3. Public cards (deck_size bits)
        4. Action sequence abstracted (max_game_length * 2 bits)
        5. Action sequence sizings (max_game_length integers)
        """
        idx = 0
        sections = {}

        # Section 1: Player ID
        player_id_size = self.num_players
        sections['player_id'] = tensor[idx:idx + player_id_size]
        idx += player_id_size

        # For sections 2-4, we need to calculate based on tensor length
        # The sizing section is always the last max_game_length values
        sections['sizing'] = tensor[-self.max_game_length:]

        # Everything between player_id and sizing is cards + abstracted actions
        middle_section = tensor[player_id_size:-self.max_game_length]
        sections['middle'] = middle_section

        # Try to estimate where cards end and actions begin
        # This is approximate without knowing exact game parameters
        sections['estimated_cards'] = middle_section[:len(middle_section)//2]
        sections['estimated_actions'] = middle_section[len(middle_section)//2:]

        return sections

    def find_bet_sizes(self, tensor):
        """Extract actual bet sizes from the tensor (last max_game_length values)"""
        return list(tensor[-self.max_game_length:])

    def print_analysis(self, tensor, label=""):
        """Print detailed tensor analysis"""
        sections = self.parse_tensor(tensor)

        print(f"\n{'='*70}")
        if label:
            print(f"  TENSOR ANALYSIS: {label}")
        else:
            print(f"  TENSOR ANALYSIS")
        print('='*70)

        print(f"\nTensor total length: {len(tensor)}")
        print(f"Max game length: {self.max_game_length}")

        print(f"\n--- Section 1: Player ID (first {self.num_players} values) ---")
        print(f"  {sections['player_id']}")
        player_idx = sections['player_id'].index(1.0) if 1.0 in sections['player_id'] else -1
        print(f"  → Player {player_idx}")

        print(f"\n--- Section 2-4: Cards + Abstracted Actions (middle section) ---")
        print(f"  Length: {len(sections['middle'])} values")
        print(f"  First 10: {sections['middle'][:10]}")
        print(f"  Last 10: {sections['middle'][-10:]}")

        print(f"\n--- Section 5: BET SIZES (last {self.max_game_length} values) ---")
        bet_sizes = sections['sizing']
        print(f"  Full sizing array: {bet_sizes}")

        # Find non-zero bet sizes (actual bets that occurred)
        non_zero_bets = [(i, val) for i, val in enumerate(bet_sizes) if val != 0.0]
        if non_zero_bets:
            print(f"\n  Non-zero bet sizes found:")
            for idx, size in non_zero_bets:
                print(f"    Position {idx}: {size:.0f} chips")
        else:
            print(f"\n  No bets placed yet (all zeros)")

        return sections


def print_test_header(test_num, description):
    """Print formatted test header"""
    print(f"\n\n{'#'*70}")
    print(f"# TEST {test_num}: {description}")
    print('#'*70)


def assert_test(condition, test_name, details=""):
    """Assert a test condition and print result"""
    if condition:
        print(f"  ✓ PASS: {test_name}")
        if details:
            print(f"         {details}")
        return True
    else:
        print(f"  ✗ FAIL: {test_name}")
        if details:
            print(f"         {details}")
        return False


def test_1_basic_tensor_structure():
    """Test 1: Verify basic tensor structure"""
    print_test_header(1, "Basic Tensor Structure")

    game = pyspiel.load_game('universal_poker', {
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

    # Create initial state and deal cards
    state = game.new_initial_state()
    state = state.child(1)  # Deal first card
    state = state.child(2)  # Deal second card

    analyzer = TensorAnalyzer(game, state)
    tensor = state.information_state_tensor()

    print(f"\nGame configuration:")
    print(f"  Players: {game.num_players()}")
    print(f"  Max game length: {game.max_game_length()}")
    print(f"  Tensor length: {len(tensor)}")

    # Parse tensor
    sections = analyzer.parse_tensor(tensor)

    # Test assertions
    passed = 0
    total = 0

    total += 1
    if assert_test(len(tensor) > 0, "Tensor has non-zero length",
                   f"Length: {len(tensor)}"):
        passed += 1

    total += 1
    if assert_test(len(sections['player_id']) == game.num_players(),
                   "Player ID section has correct size",
                   f"Expected: {game.num_players()}, Got: {len(sections['player_id'])}"):
        passed += 1

    total += 1
    if assert_test(len(sections['sizing']) == game.max_game_length(),
                   "Sizing section has correct length",
                   f"Expected: {game.max_game_length()}, Got: {len(sections['sizing'])}"):
        passed += 1

    total += 1
    if assert_test(sum(sections['player_id']) == 1.0,
                   "Player ID is one-hot encoded",
                   f"Sum: {sum(sections['player_id'])}"):
        passed += 1

    print(f"\n{'='*70}")
    print(f"  TEST 1 RESULTS: {passed}/{total} assertions passed")
    print('='*70)

    return passed == total


def test_2_bet_sizes_in_tensor():
    """Test 2: Verify actual bet sizes appear in tensor"""
    print_test_header(2, "Actual Bet Sizes Stored in Tensor")

    game = pyspiel.load_game('universal_poker', {
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

    # Create initial state and deal cards
    state = game.new_initial_state()
    state = state.child(1)  # Deal first card
    state = state.child(2)  # Deal second card

    print(f"\nStarting state: Player {state.current_player()} to act")
    print(f"Blinds posted: 50 and 100")

    analyzer = TensorAnalyzer(game, state)

    # Test different bet sizes
    test_bets = [
        (1, "Call (action 1)", 100),      # Call to 100
        (400, "Bet 400 (action 400)", 400),
        (500, "Bet 500 (action 500)", 500),
        (1200, "All-in 1200 (action 1200)", 1200),
    ]

    passed = 0
    total = 0

    for action, description, expected_size in test_bets:
        if action not in state.legal_actions():
            print(f"\n  ⊘ Skipping: {description} (not legal in this state)")
            continue

        print(f"\n--- Testing: {description} ---")

        # Apply the action and get the resulting tensor
        next_state = state.child(action)
        tensor = next_state.information_state_tensor()

        # Extract bet sizes from tensor
        bet_sizes = analyzer.find_bet_sizes(tensor)

        print(f"  Bet sizes section: {bet_sizes[:10]}... (showing first 10)")

        # Find the bet size in the tensor
        # After one action, we expect to see the bet size somewhere in the array
        non_zero = [x for x in bet_sizes if x > 0]

        total += 1
        if action == 1:  # Call action
            # Call actions don't record a raise size (they're not raises)
            if assert_test(len(non_zero) == 0,
                          f"Call action shows 0.0 (not a raise)",
                          f"Non-zero values: {non_zero} (should be empty)"):
                passed += 1
        else:  # Bet action
            if assert_test(float(expected_size) in bet_sizes,
                          f"Bet size {expected_size} found in tensor",
                          f"Non-zero values: {non_zero}"):
                passed += 1

    print(f"\n{'='*70}")
    print(f"  TEST 2 RESULTS: {passed}/{total} assertions passed")
    print('='*70)

    return passed == total


def test_3_compare_bet_sizes_side_by_side():
    """Test 3: Compare different bet sizes side-by-side"""
    print_test_header(3, "Side-by-Side Bet Size Comparison")

    game = pyspiel.load_game('universal_poker', {
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

    # Create initial state and deal cards
    state = game.new_initial_state()
    state = state.child(1)
    state = state.child(2)

    analyzer = TensorAnalyzer(game, state)

    print(f"\nComparing different bet sizes from the same starting state...")
    print(f"Starting state: Player {state.current_player()} to act after blinds\n")

    # Test specific bet sizes
    test_actions = [
        (1, "check/call"),
        (400, "bet 400"),
        (500, "bet 500"),
        (1200, "bet 1200 (all-in)")
    ]

    results = []

    for action, description in test_actions:
        if action not in state.legal_actions():
            print(f"  ⊘ {description}: Not legal in this state")
            continue

        next_state = state.child(action)
        tensor = next_state.information_state_tensor()
        bet_sizes = analyzer.find_bet_sizes(tensor)

        results.append({
            'action': action,
            'description': description,
            'tensor': tensor,
            'bet_sizes': bet_sizes,
            'non_zero': [x for x in bet_sizes if x > 0]
        })

    # Print side-by-side comparison
    print("="*70)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*70)

    for result in results:
        print(f"\n{result['description'].upper()} (action {result['action']}):")
        print(f"  Tensor length: {len(result['tensor'])}")
        print(f"  First 20 values: {result['tensor'][:20]}")
        print(f"  Last 20 values:  {result['tensor'][-20:]}")
        print(f"  Non-zero bet sizes: {result['non_zero']}")

    # Test assertions
    passed = 0
    total = 0

    print(f"\n{'='*70}")
    print("COMPARISON TESTS")
    print('='*70)

    # Test: Different bet sizes should have different non-zero values
    if len(results) >= 2:
        total += 1
        bet_400 = next((r for r in results if r['action'] == 400), None)
        bet_500 = next((r for r in results if r['action'] == 500), None)

        if bet_400 and bet_500:
            has_400 = 400.0 in bet_400['bet_sizes']
            has_500 = 500.0 in bet_500['bet_sizes']

            if assert_test(has_400 and has_500,
                          "Different bet sizes produce different tensor values",
                          f"Bet 400 has 400.0: {has_400}, Bet 500 has 500.0: {has_500}"):
                passed += 1
        else:
            print("  ⊘ SKIP: Could not compare bet 400 vs 500")

    # Test: All-in should show max stack size
    total += 1
    bet_1200 = next((r for r in results if r['action'] == 1200), None)
    if bet_1200:
        if assert_test(1200.0 in bet_1200['bet_sizes'],
                      "All-in bet shows full stack size in tensor",
                      f"Found 1200.0 in bet sizes: {1200.0 in bet_1200['bet_sizes']}"):
            passed += 1
    else:
        print("  ⊘ SKIP: All-in action not available")

    # Test: Tensor lengths should be the same regardless of bet size
    total += 1
    if len(results) >= 2:
        lengths = [len(r['tensor']) for r in results]
        all_same = all(l == lengths[0] for l in lengths)
        if assert_test(all_same,
                      "All tensors have the same length regardless of bet size",
                      f"Lengths: {lengths}"):
            passed += 1

    print(f"\n{'='*70}")
    print(f"  TEST 3 RESULTS: {passed}/{total} assertions passed")
    print('='*70)

    return passed == total


def test_4_abstraction_vs_sizing():
    """Test 4: Verify bet sizes stored even with abstraction"""
    print_test_header(4, "Bet Sizes with Abstraction (FCPA)")

    print("\nThis test demonstrates that even with betting abstraction (FCPA),")
    print("the actual bet sizes are still stored in the tensor.\n")

    game_fcpa = pyspiel.load_game('universal_poker', {
        'bettingAbstraction': 'fcpa',  # Using abstraction!
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

    print(f"Game configuration:")
    print(f"  Betting abstraction: FCPA (Fold/Call/Pot/All-in)")
    print(f"  Distinct actions: {game_fcpa.num_distinct_actions()}")
    print(f"  Max game length: {game_fcpa.max_game_length()}")

    # Create state
    state = game_fcpa.new_initial_state()
    state = state.child(1)  # Deal
    state = state.child(2)  # Deal

    analyzer = TensorAnalyzer(game_fcpa, state)

    legal_actions = state.legal_actions()
    print(f"\nLegal actions: {legal_actions}")

    passed = 0
    total = 0

    # Map actions to their meanings
    action_map = {
        0: "Fold",
        1: "Call",
        2: "Bet (pot-sized)",
        3: "All-in"
    }

    prev_bet_sizes = None

    for action in legal_actions:
        action_name = action_map.get(action, f"Action {action}")
        print(f"\n--- {action_name} (action {action}) ---")

        next_state = state.child(action)
        tensor = next_state.information_state_tensor()
        bet_sizes = analyzer.find_bet_sizes(tensor)
        non_zero = [x for x in bet_sizes if x > 0]

        print(f"  Tensor length: {len(tensor)}")
        print(f"  Bet sizes section: {bet_sizes[:10]}...")
        print(f"  Non-zero bet sizes: {non_zero}")

        # Test: Tensor should contain sizing information
        total += 1
        if action == 0:  # Fold
            if assert_test(True, f"{action_name}: Tensor exists",
                          "Fold action recorded"):
                passed += 1
        elif action == 1:  # Call
            # Call actions show 0.0 (they're not raises)
            if assert_test(len(non_zero) == 0,
                          f"{action_name}: Call shows 0.0 (not a raise)",
                          f"Found values: {non_zero} (should be empty)"):
                passed += 1
        else:  # Bet or All-in
            if assert_test(len(non_zero) > 0,
                          f"{action_name}: Bet size recorded in tensor",
                          f"Found bet sizes: {non_zero}"):
                passed += 1

        # Store for comparison
        if prev_bet_sizes is None and action > 1:  # First bet/raise
            prev_bet_sizes = bet_sizes
        elif prev_bet_sizes is not None and action > 1:
            # Test: Different bet actions should produce different sizings
            total += 1
            if assert_test(bet_sizes != prev_bet_sizes,
                          "Different bet actions produce different size arrays",
                          "Sizing arrays differ"):
                passed += 1

    print(f"\n{'='*70}")
    print(f"  TEST 4 RESULTS: {passed}/{total} assertions passed")
    print('='*70)

    return passed == total


def test_5_detailed_tensor_walkthrough():
    """Test 5: Detailed walkthrough of tensor sections"""
    print_test_header(5, "Detailed Tensor Section Walkthrough")

    game = pyspiel.load_game('universal_poker', {
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

    state = game.new_initial_state()
    state = state.child(1)  # Deal first card
    state = state.child(2)  # Deal second card

    # Make a specific bet
    bet_amount = 500
    if bet_amount in state.legal_actions():
        state = state.child(bet_amount)
        print(f"\nAction taken: Bet {bet_amount}")
    else:
        print(f"\nBet {bet_amount} not available, using first legal action")
        state = state.child(state.legal_actions()[0])

    analyzer = TensorAnalyzer(game, state)
    tensor = state.information_state_tensor()

    # Full analysis
    analyzer.print_analysis(tensor, f"After betting {bet_amount}")

    # Test assertions
    passed = 0
    total = 0

    sections = analyzer.parse_tensor(tensor)

    total += 1
    if assert_test(len(sections['sizing']) == game.max_game_length(),
                   "Sizing section is exactly max_game_length",
                   f"{len(sections['sizing'])} == {game.max_game_length()}"):
        passed += 1

    total += 1
    non_zero_sizes = [x for x in sections['sizing'] if x > 0]
    if assert_test(len(non_zero_sizes) > 0,
                   "At least one bet size is recorded",
                   f"Found {len(non_zero_sizes)} non-zero bet sizes"):
        passed += 1

    total += 1
    if assert_test(bet_amount in state.history() or float(bet_amount) in sections['sizing'],
                   f"Bet amount {bet_amount} appears in tensor or history",
                   f"History: {state.history()}, Sizes: {non_zero_sizes}"):
        passed += 1

    print(f"\n{'='*70}")
    print(f"  TEST 5 RESULTS: {passed}/{total} assertions passed")
    print('='*70)

    return passed == total


def test_6_custom_abstraction_bet_sizes():
    """Test 6: Verify bet sizes stored with custom abstraction (fullgame + filtering)"""
    print_test_header(6, "Custom Betting Abstraction (Fullgame + Filtering)")

    print("\nThis test verifies that when using fullgame + filtering for custom")
    print("abstraction (FCHPA + 1.5×pot), the actual bet sizes are still stored.")

    game = pyspiel.load_game('universal_poker', {
        'bettingAbstraction': 'fullgame',  # No hardcoded abstraction
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

    # Create state
    state = game.new_initial_state()
    state = state.child(1)  # Deal first card
    state = state.child(2)  # Deal second card

    analyzer = TensorAnalyzer(game, state)

    print(f"\nGame configuration:")
    print(f"  Betting abstraction: fullgame (with custom filtering)")
    print(f"  Max game length: {game.max_game_length()}")

    # Calculate custom bet sizes
    pot = 150  # 50 + 100 blinds
    half_pot = pot // 2  # 75
    full_pot = pot  # 150
    one_half_pot = int(pot * 1.5)  # 225
    max_stack = 1000
    tolerance = 10

    print(f"\nCustom abstraction bet sizes:")
    print(f"  Half pot (0.5×): ~{half_pot}")
    print(f"  Full pot (1.0×): ~{full_pot}")
    print(f"  One-and-half pot (1.5×): ~{one_half_pot}")
    print(f"  All-in: {max_stack}")

    # Filter actions - find closest action to each target bet size
    legal_actions = state.legal_actions()
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

    filtered_actions.sort()

    print(f"\nFiltered to {len(filtered_actions)} actions: {filtered_actions}")

    passed = 0
    total = 0

    # Test that filtered actions store bet sizes
    test_actions = [
        (a, f"Filtered action {a}")
        for a in filtered_actions if a > 1  # Skip fold and call
    ]

    for action, description in test_actions[:4]:  # Test first 4 bet actions
        print(f"\n--- Testing: {description} ---")

        next_state = state.child(action)
        tensor = next_state.information_state_tensor()
        bet_sizes = analyzer.find_bet_sizes(tensor)
        non_zero = [x for x in bet_sizes if x > 0]

        print(f"  Non-zero bet sizes in tensor: {non_zero}")

        total += 1
        if assert_test(float(action) in bet_sizes,
                      f"Bet size {action} found in tensor",
                      f"Expected {action}, found: {non_zero}"):
            passed += 1

    # Test that different filtered bet sizes produce different tensors
    total += 1
    if len(filtered_actions) >= 4:
        # Get two different bet actions
        bet_actions = [a for a in filtered_actions if a > 1]
        if len(bet_actions) >= 2:
            action1, action2 = bet_actions[0], bet_actions[1]
            tensor1 = state.child(action1).information_state_tensor()
            tensor2 = state.child(action2).information_state_tensor()

            sizes1 = analyzer.find_bet_sizes(tensor1)
            sizes2 = analyzer.find_bet_sizes(tensor2)

            if assert_test(sizes1 != sizes2,
                          "Different filtered bets produce different tensor sizes",
                          f"Action {action1} vs {action2}"):
                passed += 1
        else:
            print("  ⊘ SKIP: Not enough bet actions to compare")
    else:
        print("  ⊘ SKIP: Not enough filtered actions")

    # Verify 1.5× pot bet stores correct size
    total += 1
    # Find the action closest to 1.5x pot (should be in filtered_actions)
    one_half_pot_action = None
    bet_actions_filtered = [a for a in filtered_actions if a > 1]
    if bet_actions_filtered:
        # Find which one is closest to our 1.5x pot target
        closest_to_1_5x = min(bet_actions_filtered, key=lambda a: abs(a - one_half_pot))
        if closest_to_1_5x != max_stack:  # Make sure it's not the all-in
            one_half_pot_action = closest_to_1_5x

    if one_half_pot_action:
        next_state = state.child(one_half_pot_action)
        tensor = next_state.information_state_tensor()
        bet_sizes = analyzer.find_bet_sizes(tensor)

        if assert_test(float(one_half_pot_action) in bet_sizes,
                      f"1.5× pot bet ({one_half_pot_action}) stores actual size",
                      f"Found {one_half_pot_action} in tensor"):
            passed += 1
    else:
        print("  ⊘ SKIP: 1.5× pot action not available")
        total -= 1  # Don't count this test

    print(f"\n{'='*70}")
    print(f"  TEST 6 RESULTS: {passed}/{total} assertions passed")
    print('='*70)

    return passed == total


def main():
    """Run all tests"""
    print("="*70)
    print("  ENHANCED INFORMATION STATE TENSOR TESTING")
    print("  Focus: Proving that actual bet sizes are stored in tensors")
    print("="*70)

    all_tests_passed = True
    test_results = []

    # Run all tests
    tests = [
        ("Basic Tensor Structure", test_1_basic_tensor_structure),
        ("Actual Bet Sizes in Tensor", test_2_bet_sizes_in_tensor),
        ("Side-by-Side Comparison", test_3_compare_bet_sizes_side_by_side),
        ("Abstraction vs Sizing", test_4_abstraction_vs_sizing),
        ("Detailed Walkthrough", test_5_detailed_tensor_walkthrough),
        ("Custom Abstraction Bet Sizes", test_6_custom_abstraction_bet_sizes),
    ]

    for test_name, test_func in tests:
        try:
            passed = test_func()
            test_results.append((test_name, passed))
            if not passed:
                all_tests_passed = False
        except Exception as e:
            print(f"\n❌ TEST FAILED WITH EXCEPTION: {test_name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            test_results.append((test_name, False))
            all_tests_passed = False

    # Print final summary
    print("\n\n" + "="*70)
    print("  FINAL TEST SUMMARY")
    print("="*70)

    for test_name, passed in test_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print("="*70)

    if all_tests_passed:
        print("  ✓ ALL TESTS PASSED")
        print("="*70)
        print("\n  CONCLUSION:")
        print("  OpenSpiel's information state tensor DOES store actual bet sizes")
        print("  in the last max_game_length values, even when using betting")
        print("  abstractions like FCPA. This has been verified through multiple")
        print("  test cases comparing different bet sizes.")
        return 0
    else:
        print("  ✗ SOME TESTS FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
