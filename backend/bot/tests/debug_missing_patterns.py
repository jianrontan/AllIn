from src.cfr.poker_game import PokerGame
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def debug_missing_patterns():
    """Debug why kml and ml patterns don't appear"""
    print("=" * 60)
    print("DEBUGGING MISSING PATTERNS: kml and ml")
    print("=" * 60)

    game = PokerGame()

    # Test the specific sequences that should create missing patterns
    test_cases = [
        (['check', 'bet_medium'], 'km', 'Should allow raise_large to create kml'),
        (['bet_medium'], 'm', 'Should allow raise_large to create ml'),
    ]

    for history, current_pattern, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"History: {history}")
        print(f"Current pattern: '{current_pattern}'")

        # Check both players' situations
        for player in [0, 1]:
            current_player = len(history) % 2
            if player != current_player:
                continue  # Skip non-acting player

            print(f"\n  Player {player} to act:")

            # Calculate game state
            stack = game.calculate_player_stack_after_history(player, history)
            pot = game.calculate_current_pot_size(history)
            legal_actions = game.get_legal_actions(history)

            print(f"    Stack: ${stack:.2f}")
            print(f"    Pot: ${pot:.2f}")
            print(f"    Legal actions: {legal_actions}")

            # Check why raise_large might be missing
            if 'raise_large' not in legal_actions:
                print(f"    ❌ raise_large NOT available")

                # Debug the requirements
                if history and history[-1].startswith('bet_'):
                    call_amount = game.get_last_bet_amount_from_history(
                        history)
                    remaining_after_call = stack - call_amount
                    raise_large_requirement = 1.0 * pot

                    print(f"    Call amount: ${call_amount:.2f}")
                    print(
                        f"    Remaining after call: ${remaining_after_call:.2f}")
                    print(
                        f"    Large raise requirement: ${raise_large_requirement:.2f}")
                    print(
                        f"    Can afford large raise: {remaining_after_call >= raise_large_requirement}")

                    # Check raise count
                    raise_count = sum(
                        1 for action in history if action.startswith('raise_'))
                    print(f"    Raise count: {raise_count} (limit: 1)")
                    print(f"    Raises allowed: {raise_count < 1}")
            else:
                print(f"    ✅ raise_large IS available")


if __name__ == "__main__":
    debug_missing_patterns()
