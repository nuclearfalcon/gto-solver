#!/usr/bin/env python3
"""
Quick test: C++ MCCFR on 3-player 5BB Hold'em

Tests if C++ External Sampling MCCFR can handle the full game.

Requirements:
    source ~/open_spiel/venv/bin/activate

Usage:
    python test_cpp_mccfr_holdem.py
"""

import time
import sys
import pyspiel
from game_config import PokerGameConfig

print("="*80)
print("C++ MCCFR PERFORMANCE TEST: 3-PLAYER 5BB HOLD'EM")
print("="*80)
print()

# Load full 3-player 5BB Hold'em config
print("Loading game configuration...")
config = PokerGameConfig.from_json("configs/3p_5bb_fchpa.json")
game = config.create_game()

print(f"✓ Game loaded: {config.description}")
print()
print("Game Configuration:")
print("-" * 80)
print(f"  Players:          {config.num_players}")
print(f"  Stack sizes:      {config.stack_sizes}")
print(f"  Blinds:           {config.blinds}")
print(f"  Betting:          {config.betting_abstraction}")
print(f"  Rounds:           {config.num_rounds}")
print(f"  Suits:            {config.num_suits}")
print(f"  Ranks:            {config.num_ranks}")
print(f"  Hole cards:       {config.num_hole_cards}")
print(f"  Board cards:      {config.num_board_cards}")
print()

# Get game info
game_type = game.get_type()
print("Game Complexity:")
print("-" * 80)
print(f"  Max game length:  {game.max_game_length()}")
print(f"  Game type:        {game_type.short_name}")
print(f"  Chance mode:      {game_type.chance_mode}")
print(f"  Information:      {game_type.information}")
print(f"  Utility:          {game_type.utility}")
print()

# Create C++ External Sampling MCCFR solver
print("Initializing C++ ExternalSamplingMCCFRSolver...")
print("  - Algorithm:      External Sampling MCCFR")
print("  - Averaging:      FULL (reach-probability weighted)")
print("  - Implementation: C++ (via pyspiel bindings)")
solver = pyspiel.ExternalSamplingMCCFRSolver(
    game,
    avg_type=pyspiel.MCCFRAverageType.FULL
)
print("✓ Solver initialized")
print()

print("="*80)
print("RUNNING 10 ITERATIONS TO MEASURE PERFORMANCE")
print("="*80)
print()
print("Iteration | Time/Iter | Cumul Time | Avg Rate | ETA (100 its) | ETA (1k its)")
print("-" * 80)
sys.stdout.flush()

start_time = time.time()
iteration_times = []

for i in range(1, 11):
    iter_start = time.time()

    # Run iteration (this is the expensive part)
    solver.run_iteration()

    iter_time = time.time() - iter_start
    iteration_times.append(iter_time)

    elapsed = time.time() - start_time
    rate = i / elapsed
    eta_100 = (100 - i) / rate if rate > 0 else 0
    eta_1k = (1000 - i) / rate if rate > 0 else 0

    # Format: Iteration | Time/Iter | Cumul Time | Avg Rate | ETA (100 its) | ETA (1k its)
    print(f"    {i:2}/10 | {iter_time:9.1f}s | {elapsed:10.1f}s | {rate:8.3f}/s | "
          f"{eta_100/60:7.1f} min | {eta_1k/60:8.1f} min")
    sys.stdout.flush()

total_time = time.time() - start_time
final_rate = 10 / total_time
avg_iter_time = sum(iteration_times) / len(iteration_times)
min_iter_time = min(iteration_times)
max_iter_time = max(iteration_times)

print()
print("="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print()
print("Iteration Timing Statistics:")
print("-" * 80)
print(f"  Total time (10 iterations):  {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"  Average time per iteration:  {avg_iter_time:.1f}s")
print(f"  Fastest iteration:           {min_iter_time:.1f}s")
print(f"  Slowest iteration:           {max_iter_time:.1f}s")
print(f"  Average iteration rate:      {final_rate:.3f} iterations/second")
print()

print("Projected Time Estimates:")
print("-" * 80)
print(f"  100 iterations:    {100/final_rate:8.1f}s = {100/final_rate/60:6.1f} minutes")
print(f"  1,000 iterations:  {1000/final_rate:8.1f}s = {1000/final_rate/60:6.1f} minutes = {1000/final_rate/3600:5.2f} hours")
print(f"  10,000 iterations: {10000/final_rate/60:8.1f} min = {10000/final_rate/3600:5.1f} hours")
print(f"  100,000 iterations:{100000/final_rate/3600:7.1f} hours = {100000/final_rate/86400:5.1f} days")
print()

# Calculate exploitability (this is expensive for large games)
print("="*80)
print("CALCULATING EXPLOITABILITY")
print("="*80)
print()
print("Getting average policy from solver...")
policy = solver.average_policy()
print(f"✓ Policy retrieved (contains {len(policy)} information states)")
print()

print("Attempting to calculate full exploitability...")
print("  NOTE: This may be very expensive for large games!")
print("  If this hangs, use sampled exploitability instead.")
print()

exploit_start = time.time()
try:
    exploit = pyspiel.exploitability(game, policy)
    exploit_time = time.time() - exploit_start
    print(f"✓ Exploitability calculated in {exploit_time:.1f}s")
    print(f"  Exploitability after 10 iterations: {exploit:.6f}")
    print()
    print(f"  NOTE: This is NOT Nash convergence.")
    print(f"  For 3-player games, divide by 2 to get Nash convergence equivalent.")
    nash_equiv = exploit / 2.0
    print(f"  Nash convergence equivalent: {nash_equiv:.6f}")
except Exception as e:
    print(f"✗ Full exploitability calculation failed: {e}")
    print()
    print("  This is expected for very large games.")
    print("  Use SampledExploitabilityCalculator from exploitability_metrics.py")
    print("  for practical exploitability measurements on large games.")
print()

print("="*80)
print("FINAL ASSESSMENT")
print("="*80)
print()

# Assess feasibility
if final_rate >= 1.0:
    print("✓ C++ MCCFR IS FAST ENOUGH FOR THIS GAME!")
    print()
    print(f"  Performance:        {final_rate:.2f} iterations/second")
    print(f"  Practical training: 10k iterations in {10000/final_rate/60:.0f} minutes")
    print(f"                      100k iterations in {100000/final_rate/3600:.1f} hours")
    print()
    print("  Recommendation:     USE THIS for algorithm comparison")
    print("                      C++ MCCFR can handle this game size")
elif final_rate >= 0.1:
    print("⚠ C++ MCCFR IS SLOW BUT POTENTIALLY USABLE")
    print()
    print(f"  Performance:        {final_rate:.3f} iterations/second")
    print(f"  Training time:      10k iterations = {10000/final_rate/3600:.1f} hours")
    print()
    print("  Recommendation:     Consider one of these options:")
    print("                      1. Use smaller game (fewer suits/ranks/rounds)")
    print("                      2. Try Outcome Sampling MCCFR (faster)")
    print("                      3. Stick with Kuhn poker for comparisons")
elif final_rate >= 0.01:
    print("✗ C++ MCCFR IS TOO SLOW FOR PRACTICAL USE")
    print()
    print(f"  Performance:        {final_rate:.4f} iterations/second")
    print(f"  Training time:      10k iterations = {10000/final_rate/3600:.1f} hours")
    print()
    print("  Recommendation:     Game is too large!")
    print("                      1. Use much smaller game (Kuhn poker)")
    print("                      2. Try Outcome Sampling MCCFR")
    print("                      3. Consider card/action abstractions")
else:
    print("✗ C++ MCCFR CANNOT HANDLE THIS GAME SIZE")
    print()
    print(f"  Performance:        {final_rate:.6f} iterations/second")
    print(f"  Per iteration:      {avg_iter_time:.0f} seconds/iteration")
    print()
    print("  Recommendation:     This game is FAR too large")
    print("                      Use Kuhn poker for algorithm testing")
    print("                      For real Hold'em, need heavy abstractions")

print("="*80)
