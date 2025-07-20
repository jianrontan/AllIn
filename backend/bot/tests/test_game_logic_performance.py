import time
from src.cfr.poker_game import PokerGame


def test_game_logic_performance():
    """Test Cython vs Python game logic performance"""

    game = PokerGame()

    # Test scenarios
    test_histories = [
        [],
        ['check'],
        ['bet_small', 'call'],
        ['bet_medium', 'raise_large', 'call'],
        ['check', 'bet_large', 'raise_large', 'call'],
        ['bet_small', 'raise_medium', 'raise_large', 'call']
    ]

    iterations = 50000

    print(f"🏃 Testing game logic performance ({iterations} iterations)...")

    # Test pot size calculation
    start_time = time.time()
    for _ in range(iterations):
        for history in test_histories:
            pot_size = game.calculate_current_pot_size(history)
    pot_time = time.time() - start_time

    # Test round completion
    start_time = time.time()
    for _ in range(iterations):
        for history in test_histories:
            is_complete = game.is_round_complete(history)
    round_time = time.time() - start_time

    # Test stack calculation
    start_time = time.time()
    for _ in range(iterations):
        for history in test_histories:
            stack = game.calculate_player_stack_after_history(0, history)
    stack_time = time.time() - start_time

    print(f"📊 Results:")
    print(
        f"   Pot calculation: {pot_time:.3f}s ({pot_time/iterations*1000:.3f}ms per 1000 calls)")
    print(
        f"   Round completion: {round_time:.3f}s ({round_time/iterations*1000:.3f}ms per 1000 calls)")
    print(
        f"   Stack calculation: {stack_time:.3f}s ({stack_time/iterations*1000:.3f}ms per 1000 calls)")
    print(f"   Total: {pot_time + round_time + stack_time:.3f}s")


if __name__ == "__main__":
    test_game_logic_performance()
