#!/usr/bin/env python3
"""
Minimal 3-player Hold'em CFR test with periodic exploitability checks.

Run 100 iterations with:
- Progress updates every 5 iterations
- Exploitability measurements every 20 iterations

Make sure to activate the virtual environment first:
    source ~/open_spiel/venv/bin/activate
"""

import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr
import time


def create_minimal_3p_holdem():
    """Create a minimal 3-player Hold'em game configuration."""
    game_config = {
        'betting': 'nolimit',
        'numPlayers': 3,
        'numRounds': 4,                    # Preflop, Flop, Turn, River
        'blind': '100 50 0',               # Player 0: BB, Player 1: SB, Player 2: no blind
        'firstPlayer': '2 1 1 1',          # First to act each round
        'numSuits': 4,
        'numRanks': 13,
        'numHoleCards': 2,
        'numBoardCards': '0 3 1 1',        # 0 preflop, 3 flop, 1 turn, 1 river
        'stack': '10000 10000 10000',      # Small stacks for faster testing
        'bettingAbstraction': 'fcpa'       # Fold, Call, Pot, All-in
    }

    return pyspiel.load_game('universal_poker', game_config)


def calculate_exploitability(game, solver):
    """Calculate exploitability using sampled method."""
    from exploitability_metrics import SampledExploitabilityCalculator

    avg_policy = solver.average_policy()
    calc = SampledExploitabilityCalculator(game, avg_policy)

    # Use minimal sampling for quick periodic checks
    result = calc.calculate(
        confidence_level=0.95,
        max_ci_width=0.10,      # 10% CI width (relaxed for speed)
        min_samples=20,          # Minimum samples
        max_samples=100          # Maximum samples (keep it fast)
    )

    return result


def main():
    print("=" * 70)
    print("Minimal 3-Player Hold'em CFR Test")
    print("=" * 70)
    print()

    # Create game
    print("Creating 3-player Hold'em game (FCPA abstraction, 10k stacks)...")
    game = create_minimal_3p_holdem()
    print(f"  Game: {game.get_type().short_name}")
    print(f"  Players: {game.num_players()}")
    print(f"  Max game length: {game.max_game_length()}")
    print()

    # Create solver (using External Sampling MCCFR with FULL averaging for 3+ players)
    print("Initializing External Sampling MCCFR solver...")
    print("  Using FULL averaging (recommended for 3+ players)")
    solver = external_sampling_mccfr.ExternalSamplingSolver(
        game,
        average_type=external_sampling_mccfr.AverageType.FULL
    )
    print()

    # Training loop
    print("Starting training: 100 iterations")
    print("  Progress updates: every 5 iterations")
    print("  Exploitability checks: every 20 iterations")
    print("-" * 70)
    print()

    total_iterations = 100
    progress_interval = 5
    exploit_interval = 20

    start_time = time.time()

    for i in range(1, total_iterations + 1):
        solver.iteration()

        # Progress updates
        if i % progress_interval == 0:
            elapsed = time.time() - start_time
            speed = i / elapsed
            print(f"Iteration {i:3d}/{total_iterations} | "
                  f"Elapsed: {elapsed:6.2f}s | "
                  f"Speed: {speed:5.2f} it/s", end="")

            # Exploitability check
            if i % exploit_interval == 0:
                print(" | Checking exploitability...", flush=True)
                result = calculate_exploitability(game, solver)
                print(f"  → Exploitability: {result['exploitability']:.6f} "
                      f"(±{(result['ci_upper'] - result['ci_lower'])/2:.6f}, "
                      f"{result['num_samples']} samples)")
            else:
                print()

    # Final results
    print()
    print("-" * 70)
    total_time = time.time() - start_time
    final_speed = total_iterations / total_time

    print(f"\nTraining complete!")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average speed: {final_speed:.2f} it/s")
    print()

    # Final exploitability measurement
    print("Computing final exploitability (higher accuracy)...")
    from exploitability_metrics import SampledExploitabilityCalculator

    avg_policy = solver.average_policy()
    calc = SampledExploitabilityCalculator(game, avg_policy)

    final_result = calc.calculate(
        confidence_level=0.99,
        max_ci_width=0.05,      # 5% CI width for final measurement
        min_samples=50,
        max_samples=200
    )

    print(f"  Exploitability: {final_result['exploitability']:.6f}")
    print(f"  95% CI: [{final_result['ci_lower']:.6f}, {final_result['ci_upper']:.6f}]")
    print(f"  Samples used: {final_result['num_samples']}")
    print()
    print("=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
