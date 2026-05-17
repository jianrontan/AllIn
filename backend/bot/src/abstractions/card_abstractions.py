# backend/bot/abstractions/card_abstractions.py
from .hand_evaluator import HandEvaluator, BoardTextureEvaluator


class CardAbstraction:
    """
    Start with 8 preflop buckets
    """

    def __init__(self):
        self.preflop_buckets = {
            'premium_pair': ['AA', 'KK', 'QQ'],
            'medium_pair': ['JJ', 'TT', '99'],
            'small_pair': ['88', '77', '66', '55', '44', '33', '22'],
            'ace_king': ['AKs', 'AKo'],
            'strong_ace': ['AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'ATo'],
            'ace_x': ['A9s', 'A9o', 'A8s', 'A8o', 'A7s', 'A7o', 'A6s', 'A6o', 'A5s', 'A5o', 'A4s', 'A4o', 'A3s', 'A3o', 'A2s', 'A2o'],
            'broadway': ['KQs', 'KQo', 'KJs', 'KJo', 'KTs', 'KTo', 'QJs', 'QJo', 'QTs', 'QTo', 'JTs', 'JTo'],
            'suited_connector': ['T9s', '98s', '87s', '76s', '65s', '54s'],
            'offsuit_connector': ['T9o', '98o', '87o', '76o', '65o', '54o'],
            'weak': []  # Everything else
        }
        self.hand_evaluator = HandEvaluator()
        self.board_texture = BoardTextureEvaluator()
        self._bucket_cache = {}
        # Reverse lookup: hand string → bucket name, built once for O(1) preflop lookup
        self._preflop_lookup = {
            hand: bucket
            for bucket, hands in self.preflop_buckets.items()
            for hand in hands
        }

    def get_bucket(self, hole_cards, community_cards=None):
        """
        Similar to get_info_set() logic but for 2-card hands
        hole_cards: [Card, Card] from PyPokerEngine
        """
        key = (tuple(hole_cards), tuple(community_cards) if community_cards else None)
        cached = self._bucket_cache.get(key)
        if cached is not None:
            return cached
        result = (self.preflop_bucket(hole_cards) if not community_cards
                  else self.postflop_bucket(hole_cards, community_cards))
        self._bucket_cache[key] = result
        return result

    def preflop_bucket(self, hole_cards):
        """Convert 2-card hand to bucket"""
        hand_str = self.cards_to_string(hole_cards)
        return self._preflop_lookup.get(hand_str, 'weak')

    def postflop_bucket(self, hole_cards, community_cards):
        """Return integer bucket 0–7 encoding board-adjusted hand strength."""
        return self.board_texture.compute_postflop_bucket(
            hole_cards, community_cards, self.hand_evaluator
        )

    def cards_to_string(self, hole_cards):
        """Convert PyPokerEngine cards to readable format"""

        # Handle different PyPokerEngine card formats
        if isinstance(hole_cards[0], str):
            # If cards are already strings like "AH", "KS"
            return self.parse_string_cards(hole_cards)
        else:
            # If cards are Card objects with .rank and .suit
            return self.parse_card_objects(hole_cards)

    def parse_string_cards(self, hole_cards):
        """Handle [Suit][Rank] format: PyPokerEngine ('HA') and training deck ('HA')"""
        card1_str, card2_str = hole_cards

        suit1 = card1_str[0]
        rank1 = card1_str[1]
        suit2 = card2_str[0]
        rank2 = card2_str[1]

        return self.format_hand_string(rank1, rank2, suit1, suit2)

    def parse_card_objects(self, hole_cards):
        """Handle PyPokerEngine Card objects with .rank and .suit"""
        rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
                    9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}

        card1, card2 = hole_cards
        rank1 = rank_map[card1.rank]
        rank2 = rank_map[card2.rank]

        # Convert suit numbers to letters (if needed for suited/offsuit)
        suited = card1.suit == card2.suit

        return self.format_hand_string(rank1, rank2, card1.suit, card2.suit)

    def format_hand_string(self, rank1, rank2, suit1, suit2):
        """Format two cards into standard notation like 'AKs' or 'AKo'"""
        # Determine rank order (higher rank first)
        rank_order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
                      '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

        if rank_order.get(rank1, 0) >= rank_order.get(rank2, 0):
            high_rank, low_rank = rank1, rank2
            suited = suit1 == suit2
        else:
            high_rank, low_rank = rank2, rank1
            suited = suit1 == suit2

        # Format like "AKs" (suited) or "AKo" (offsuit)
        if high_rank == low_rank:  # Pair
            return f"{high_rank}{low_rank}"  # "AA", "KK", etc.
        else:
            suffix = 's' if suited else 'o'
            return f"{high_rank}{low_rank}{suffix}"
