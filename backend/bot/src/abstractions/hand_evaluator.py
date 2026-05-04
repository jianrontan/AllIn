# backend/bot/abstractions/hand_evaluator.py
from phevaluator.evaluator import evaluate_cards


class HandEvaluator:
    """
    Based on Henry R Lee's PokerHandEvaluator and StackWild's approach
    Maps cards to integers 0-51 and uses lookup tables for fast evaluation
    """

    def __init__(self):
        # Hand strength rankings (higher number = stronger hand)
        self.HAND_TYPES = {
            'high_card': 0,
            'pair': 1,
            'two_pair': 2,
            'three_of_kind': 3,
            'straight': 4,
            'flush': 5,
            'full_house': 6,
            'four_of_kind': 7,
            'straight_flush': 8
        }

    def convert_card_format(self, card):
        """
        Convert PyPokerEngine format to phevaluator format
        PyPokerEngine: 'CT' (Club Ten) -> phevaluator: 'Tc' (Ten of clubs)
        """
        if len(card) != 2:
            return card  # Already in correct format or invalid

        # PyPokerEngine format: [Suit][Rank]
        suit = card[0]
        rank = card[1]

        # Convert suit to lowercase for phevaluator
        suit_map = {
            'S': 's',  # Spades
            'H': 'h',  # Hearts
            'D': 'd',  # Diamonds
            'C': 'c'   # Clubs
        }

        # Return in phevaluator format: [Rank][suit]
        return rank + suit_map.get(suit, suit.lower())

    def evaluate_hand_strength(self, hole_cards, community_cards):
        """
        Main interface - takes PyPokerEngine cards and returns hand type + strength
        hole_cards: ['CT', 'H2'] (PyPokerEngine format)
        community_cards: ['C9', 'DT', 'ST'] (PyPokerEngine format)
        """
        # Convert all cards to phevaluator format
        converted_hole = [self.convert_card_format(
            card) for card in hole_cards]
        converted_community = [self.convert_card_format(
            card) for card in community_cards]

        all_cards = converted_hole + converted_community

        # Get hand strength from phevaluator
        hand_value = evaluate_cards(*all_cards)

        # Convert to hand type and relative strength
        hand_type = self.get_hand_type_from_value(hand_value)
        relative_strength = self.get_relative_strength(hand_value)

        return hand_type, relative_strength

    def get_hand_type_from_value(self, hand_value):
        """Convert phevaluator result to readable hand type"""
        # phevaluator returns lower numbers for stronger hands
        # Need to map these ranges to hand types
        if hand_value <= 10:
            return 'straight_flush'
        elif hand_value <= 166:
            return 'four_of_kind'
        elif hand_value <= 322:
            return 'full_house'
        elif hand_value <= 1599:
            return 'flush'
        elif hand_value <= 1609:
            return 'straight'
        elif hand_value <= 2467:
            return 'three_of_kind'
        elif hand_value <= 3325:
            return 'two_pair'
        elif hand_value <= 6185:
            return 'pair'
        else:
            return 'high_card'

    def get_relative_strength(self, hand_value):
        """Convert to 0-8 scale for easy bucketing"""
        hand_type = self.get_hand_type_from_value(hand_value)
        return self.HAND_TYPES[hand_type]

    def has_draw_potential(self, hole_cards, community_cards):
        """Check for flush/straight draws"""
        if len(community_cards) >= 5:  # River - no more potential
            return False

        all_cards = hole_cards + community_cards

        # Simple draw detection
        return (self.has_flush_draw(all_cards) or
                self.has_straight_draw(all_cards))

    def has_flush_draw(self, cards):
        """Check for 4+ cards of same suit"""
        suits = {}
        for card in cards:
            # PyPokerEngine format: first char is suit
            suit = card[0]
            suits[suit] = suits.get(suit, 0) + 1

        return any(count >= 4 for count in suits.values())

    def has_straight_draw(self, cards):
        """Check for 4+ cards that could make a straight"""
        ranks = set()
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

        for card in cards:
            # PyPokerEngine format: second char is rank
            rank = card[1]
            if rank in rank_map:
                ranks.add(rank_map[rank])

        # Check for 4+ consecutive ranks
        sorted_ranks = sorted(ranks)
        consecutive = 1
        max_consecutive = 1

        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i-1] + 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1

        return max_consecutive >= 4
