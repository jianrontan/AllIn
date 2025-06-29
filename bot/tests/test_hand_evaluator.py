from src.abstractions.card_abstractions import CardAbstraction
from src.abstractions.hand_evaluator import HandEvaluator


def test_hand_evaluator():
    """Test hand evaluator integration"""
    evaluator = HandEvaluator()

    # Test royal flush
    hand_type, strength = evaluator.evaluate_hand_strength(
        ['AH', 'KH'], ['QH', 'JH', 'TH'])
    print(f"Royal flush: {hand_type}, strength: {strength}")

    # Test pair
    hand_type, strength = evaluator.evaluate_hand_strength(
        ['AH', '7S'], ['AC', '7H', '2D'])
    print(f"Two pair: {hand_type}, strength: {strength}")

    # Test high card
    hand_type, strength = evaluator.evaluate_hand_strength(
        ['AH', '7S'], ['KC', 'JH', '2D'])
    print(f"High card: {hand_type}, strength: {strength}")


def test_card_abstraction_integration():
    """Test that card abstraction uses hand evaluator"""
    abstraction = CardAbstraction()

    # Test postflop bucketing
    bucket = abstraction.postflop_bucket(['AH', '7S'], ['AC', '7H', '2D'])
    print(f"A7 with two pair gets bucket: {bucket}")

    bucket = abstraction.postflop_bucket(['AH', '7S'], ['KC', 'JH', '2D'])
    print(f"A7 with ace high gets bucket: {bucket}")


if __name__ == "__main__":
    test_hand_evaluator()
    test_card_abstraction_integration()
