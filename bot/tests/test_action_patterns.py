from src.abstractions.action_abstractions import ActionAbstraction
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_action_pattern_generation():
    """Test if action abstraction can generate missing patterns"""
    print("=" * 60)
    print("TESTING ACTION PATTERN GENERATION")
    print("=" * 60)

    abstraction = ActionAbstraction()

    # Test sequences that should produce missing patterns
    test_sequences = [
        # (action_sequence, expected_pattern, description)
        ([{'action': 'check', 'amount': 0},
          {'action': 'bet', 'amount': 66}], 'km', 'Check, medium bet'),

        ([{'action': 'check', 'amount': 0},
          {'action': 'bet', 'amount': 100}], 'kl', 'Check, large bet'),

        ([{'action': 'check', 'amount': 0},
          {'action': 'check', 'amount': 0}], 'kk', 'Check, check'),

        ([{'action': 'bet', 'amount': 33},
          {'action': 'call', 'amount': 0}], 'sc', 'Small bet, call'),

        ([{'action': 'check', 'amount': 0},
          {'action': 'call', 'amount': 0}], 'kc', 'Check, call (after opponent bet)'),

        ([{'action': 'bet', 'amount': 66},
          {'action': 'bet', 'amount': 100}], 'ml', 'Medium bet, large raise'),

        ([{'action': 'bet', 'amount': 33},
          {'action': 'bet', 'amount': 100}], 'sl', 'Small bet, large raise'),
    ]

    # Test each sequence
    for actions, expected, description in test_sequences:
        game_state = {
            'pot_size': 100,
            'player_stack': 1000,
            'current_bet': 0,
            'player_contribution': 0,
            'big_blind': 2
        }

        result = abstraction.abstract_action_history(actions, game_state)

        print(f"Test: {description}")
        print(
            f"  Actions: {[a['action'] + ('_' + str(a['amount']) if a['amount'] > 0 else '') for a in actions]}")
        print(f"  Expected pattern: '{expected}'")
        print(f"  Actual pattern: '{result}'")
        print(f"  ✅ Match: {result == expected}")
        print()


def test_bet_categorization():
    """Test bet size categorization specifically"""
    print("=" * 60)
    print("TESTING BET SIZE CATEGORIZATION")
    print("=" * 60)

    abstraction = ActionAbstraction()

    test_bets = [
        (33, 100, 'small'),   # 33% pot
        (66, 100, 'medium'),  # 66% pot
        (100, 100, 'large'),  # 100% pot
        (150, 100, 'overbet')  # 150% pot
    ]

    for amount, pot_size, expected in test_bets:
        action = {'action': 'bet', 'amount': amount}
        game_state = {'pot_size': pot_size, 'player_stack': 1000}

        category = abstraction.categorize_bet_size(action, game_state)
        first_letter = category[0] if category else '?'

        print(f"Bet ${amount} in ${pot_size} pot:")
        print(f"  Ratio: {amount/pot_size:.2f}")
        print(f"  Category: '{category}' (expected: '{expected}')")
        print(f"  First letter: '{first_letter}'")
        print(f"  ✅ Correct: {category == expected}")
        print()


if __name__ == "__main__":
    test_action_pattern_generation()
    test_bet_categorization()
