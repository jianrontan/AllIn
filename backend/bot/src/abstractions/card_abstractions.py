# backend/bot/abstractions/card_abstractions.py
from .hand_evaluator import HandEvaluator
from .postflop_v2 import PostflopV2

# Precomputed via scripts/compute_preflop_equity.py
# 10,000 Monte Carlo simulations vs random opponent, seed=42, 15 equal-quantile buckets
_PREFLOP_EQUITY = {
    '22': 0.5088,
    '32o': 0.3208,
    '32s': 0.3620,
    '33': 0.5338,
    '42o': 0.3290,
    '42s': 0.3739,
    '43o': 0.3510,
    '43s': 0.3907,
    '44': 0.5710,
    '52o': 0.3483,
    '52s': 0.3760,
    '53o': 0.3660,
    '53s': 0.3998,
    '54o': 0.3881,
    '54s': 0.4132,
    '55': 0.5970,
    '62o': 0.3418,
    '62s': 0.3812,
    '63o': 0.3618,
    '63s': 0.4066,
    '64o': 0.3824,
    '64s': 0.4125,
    '65o': 0.4062,
    '65s': 0.4324,
    '66': 0.6308,
    '72o': 0.3584,
    '72s': 0.3842,
    '73o': 0.3775,
    '73s': 0.3916,
    '74o': 0.3902,
    '74s': 0.4199,
    '75o': 0.4023,
    '75s': 0.4232,
    '76o': 0.4220,
    '76s': 0.4565,
    '77': 0.6577,
    '82o': 0.3673,
    '82s': 0.4011,
    '83o': 0.3795,
    '83s': 0.4028,
    '84o': 0.3923,
    '84s': 0.4329,
    '85o': 0.4206,
    '85s': 0.4385,
    '86o': 0.4304,
    '86s': 0.4707,
    '87o': 0.4501,
    '87s': 0.4803,
    '88': 0.6984,
    '92o': 0.3983,
    '92s': 0.4215,
    '93o': 0.3907,
    '93s': 0.4374,
    '94o': 0.3946,
    '94s': 0.4386,
    '95o': 0.4235,
    '95s': 0.4515,
    '96o': 0.4406,
    '96s': 0.4808,
    '97o': 0.4657,
    '97s': 0.4927,
    '98o': 0.4813,
    '98s': 0.5061,
    '99': 0.7169,
    'A2o': 0.5527,
    'A2s': 0.5745,
    'A3o': 0.5534,
    'A3s': 0.5764,
    'A4o': 0.5698,
    'A4s': 0.5883,
    'A5o': 0.5826,
    'A5s': 0.5995,
    'A6o': 0.5686,
    'A6s': 0.5992,
    'A7o': 0.5857,
    'A7s': 0.6092,
    'A8o': 0.5967,
    'A8s': 0.6261,
    'A9o': 0.6136,
    'A9s': 0.6319,
    'AA': 0.8562,
    'AJo': 0.6303,
    'AJs': 0.6589,
    'AKo': 0.6481,
    'AKs': 0.6701,
    'AQo': 0.6358,
    'AQs': 0.6591,
    'ATo': 0.6300,
    'ATs': 0.6492,
    'J2o': 0.4405,
    'J2s': 0.4726,
    'J3o': 0.4512,
    'J3s': 0.4818,
    'J4o': 0.4676,
    'J4s': 0.4936,
    'J5o': 0.4736,
    'J5s': 0.4948,
    'J6o': 0.4824,
    'J6s': 0.5077,
    'J7o': 0.4998,
    'J7s': 0.5211,
    'J8o': 0.5130,
    'J8s': 0.5411,
    'J9o': 0.5322,
    'J9s': 0.5591,
    'JJ': 0.7810,
    'JTo': 0.5527,
    'JTs': 0.5850,
    'K2o': 0.5060,
    'K2s': 0.5352,
    'K3o': 0.5129,
    'K3s': 0.5413,
    'K4o': 0.5245,
    'K4s': 0.5481,
    'K5o': 0.5321,
    'K5s': 0.5532,
    'K6o': 0.5459,
    'K6s': 0.5612,
    'K7o': 0.5506,
    'K7s': 0.5739,
    'K8o': 0.5645,
    'K8s': 0.5813,
    'K9o': 0.5735,
    'K9s': 0.6008,
    'KJo': 0.6043,
    'KJs': 0.6237,
    'KK': 0.8133,
    'KQo': 0.6217,
    'KQs': 0.6271,
    'KTo': 0.5929,
    'KTs': 0.6131,
    'Q2o': 0.4762,
    'Q2s': 0.5023,
    'Q3o': 0.4781,
    'Q3s': 0.5171,
    'Q4o': 0.4891,
    'Q4s': 0.5220,
    'Q5o': 0.4894,
    'Q5s': 0.5254,
    'Q6o': 0.5102,
    'Q6s': 0.5380,
    'Q7o': 0.5195,
    'Q7s': 0.5490,
    'Q8o': 0.5344,
    'Q8s': 0.5586,
    'Q9o': 0.5541,
    'Q9s': 0.5818,
    'QJo': 0.5793,
    'QJs': 0.5979,
    'QQ': 0.7973,
    'QTo': 0.5749,
    'QTs': 0.5802,
    'T2o': 0.4138,
    'T2s': 0.4420,
    'T3o': 0.4238,
    'T3s': 0.4562,
    'T4o': 0.4353,
    'T4s': 0.4627,
    'T5o': 0.4496,
    'T5s': 0.4721,
    'T6o': 0.4667,
    'T6s': 0.4876,
    'T7o': 0.4759,
    'T7s': 0.5075,
    'T8o': 0.4935,
    'T8s': 0.5256,
    'T9o': 0.5233,
    'T9s': 0.5433,
    'TT': 0.7534,
}

# Custom bucket assignment: 169 hands → 15 buckets (pf_0 weakest, pf_14 strongest)
# pf_14: hands with equity >= 0.75 (TT+). Remaining 164 hands: equal-quantile pf_0–pf_13.
# This ensures 77/88/99 (equity 0.658–0.717) are separated from TT+ (0.753–0.856).
_PREFLOP_BUCKET_MAP = {
    '22': 'pf_7',
    '32o': 'pf_0',
    '32s': 'pf_0',
    '33': 'pf_8',
    '42o': 'pf_0',
    '42s': 'pf_0',
    '43o': 'pf_0',
    '43s': 'pf_1',
    '44': 'pf_10',
    '52o': 'pf_0',
    '52s': 'pf_0',
    '53o': 'pf_0',
    '53s': 'pf_2',
    '54o': 'pf_1',
    '54s': 'pf_2',
    '55': 'pf_11',
    '62o': 'pf_0',
    '62s': 'pf_1',
    '63o': 'pf_0',
    '63s': 'pf_2',
    '64o': 'pf_1',
    '64s': 'pf_2',
    '65o': 'pf_2',
    '65s': 'pf_3',
    '66': 'pf_13',
    '72o': 'pf_0',
    '72s': 'pf_1',
    '73o': 'pf_1',
    '73s': 'pf_1',
    '74o': 'pf_1',
    '74s': 'pf_2',
    '75o': 'pf_2',
    '75s': 'pf_3',
    '76o': 'pf_3',
    '76s': 'pf_4',
    '77': 'pf_13',
    '82o': 'pf_0',
    '82s': 'pf_2',
    '83o': 'pf_1',
    '83s': 'pf_2',
    '84o': 'pf_1',
    '84s': 'pf_3',
    '85o': 'pf_2',
    '85s': 'pf_3',
    '86o': 'pf_3',
    '86s': 'pf_5',
    '87o': 'pf_4',
    '87s': 'pf_5',
    '88': 'pf_13',
    '92o': 'pf_2',
    '92s': 'pf_3',
    '93o': 'pf_1',
    '93s': 'pf_3',
    '94o': 'pf_1',
    '94s': 'pf_4',
    '95o': 'pf_3',
    '95s': 'pf_4',
    '96o': 'pf_4',
    '96s': 'pf_5',
    '97o': 'pf_4',
    '97s': 'pf_6',
    '98o': 'pf_5',
    '98s': 'pf_7',
    '99': 'pf_13',
    'A2o': 'pf_9',
    'A2s': 'pf_10',
    'A3o': 'pf_9',
    'A3s': 'pf_10',
    'A4o': 'pf_10',
    'A4s': 'pf_11',
    'A5o': 'pf_11',
    'A5s': 'pf_12',
    'A6o': 'pf_10',
    'A6s': 'pf_11',
    'A7o': 'pf_11',
    'A7s': 'pf_12',
    'A8o': 'pf_11',
    'A8s': 'pf_12',
    'A9o': 'pf_12',
    'A9s': 'pf_13',
    'AA': 'pf_14',
    'AJo': 'pf_12',
    'AJs': 'pf_13',
    'AKo': 'pf_13',
    'AKs': 'pf_13',
    'AQo': 'pf_13',
    'AQs': 'pf_13',
    'ATo': 'pf_12',
    'ATs': 'pf_13',
    'J2o': 'pf_4',
    'J2s': 'pf_5',
    'J3o': 'pf_4',
    'J3s': 'pf_6',
    'J4o': 'pf_5',
    'J4s': 'pf_6',
    'J5o': 'pf_5',
    'J5s': 'pf_6',
    'J6o': 'pf_6',
    'J6s': 'pf_7',
    'J7o': 'pf_6',
    'J7s': 'pf_7',
    'J8o': 'pf_7',
    'J8s': 'pf_8',
    'J9o': 'pf_8',
    'J9s': 'pf_9',
    'JJ': 'pf_14',
    'JTo': 'pf_9',
    'JTs': 'pf_11',
    'K2o': 'pf_7',
    'K2s': 'pf_8',
    'K3o': 'pf_7',
    'K3s': 'pf_8',
    'K4o': 'pf_8',
    'K4s': 'pf_9',
    'K5o': 'pf_8',
    'K5s': 'pf_9',
    'K6o': 'pf_9',
    'K6s': 'pf_10',
    'K7o': 'pf_9',
    'K7s': 'pf_10',
    'K8o': 'pf_10',
    'K8s': 'pf_11',
    'K9o': 'pf_10',
    'K9s': 'pf_12',
    'KJo': 'pf_12',
    'KJs': 'pf_12',
    'KK': 'pf_14',
    'KQo': 'pf_12',
    'KQs': 'pf_12',
    'KTo': 'pf_11',
    'KTs': 'pf_12',
    'Q2o': 'pf_5',
    'Q2s': 'pf_6',
    'Q3o': 'pf_5',
    'Q3s': 'pf_7',
    'Q4o': 'pf_6',
    'Q4s': 'pf_7',
    'Q5o': 'pf_6',
    'Q5s': 'pf_8',
    'Q6o': 'pf_7',
    'Q6s': 'pf_8',
    'Q7o': 'pf_7',
    'Q7s': 'pf_9',
    'Q8o': 'pf_8',
    'Q8s': 'pf_9',
    'Q9o': 'pf_9',
    'Q9s': 'pf_11',
    'QJo': 'pf_10',
    'QJs': 'pf_11',
    'QQ': 'pf_14',
    'QTo': 'pf_10',
    'QTs': 'pf_11',
    'T2o': 'pf_2',
    'T2s': 'pf_4',
    'T3o': 'pf_3',
    'T3s': 'pf_4',
    'T4o': 'pf_3',
    'T4s': 'pf_4',
    'T5o': 'pf_4',
    'T5s': 'pf_5',
    'T6o': 'pf_5',
    'T6s': 'pf_6',
    'T7o': 'pf_5',
    'T7s': 'pf_7',
    'T8o': 'pf_6',
    'T8s': 'pf_8',
    'T9o': 'pf_8',
    'T9s': 'pf_9',
    'TT': 'pf_14',
}

# Number of preflop buckets, derived from the map above so consumers (e.g. the
# API's Key Explorer vocabulary) can't drift from the actual abstraction.
NUM_PREFLOP_BUCKETS = len(set(_PREFLOP_BUCKET_MAP.values()))


class CardAbstraction:
    """
    15 equity-based preflop buckets (pf_0 weakest → pf_14 strongest).
    Postflop: distribution-aware (potential-aware) buckets via PostflopV2 --
    12 flop / 12 turn / 10 river, from precomputed equity-distribution centroids
    + baked lookup tables (see scripts/compute_postflop_buckets.py and
    scripts/bake_postflop_table.py). This replaced the old 8-bucket
    BoardTextureEvaluator heuristic; blueprints must be (re)trained under it.
    """

    # Bound the cache so a long training run (10M+ hands, mostly-distinct
    # postflop boards) can't grow it without limit. Cleared wholesale on
    # overflow -- simple and safe: inference sessions never approach the cap,
    # and postflop boards rarely repeat during training so evictions cost little.
    _CACHE_CAP = 500_000

    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.postflop = PostflopV2()
        self._bucket_cache = {}

    def get_bucket(self, hole_cards, community_cards=None):
        key = (tuple(hole_cards), tuple(community_cards) if community_cards else None)
        cached = self._bucket_cache.get(key)
        if cached is not None:
            return cached
        result = (self.preflop_bucket(hole_cards) if not community_cards
                  else self.postflop_bucket(hole_cards, community_cards))
        if len(self._bucket_cache) >= self._CACHE_CAP:
            self._bucket_cache.clear()
        self._bucket_cache[key] = result
        return result

    def preflop_bucket(self, hole_cards):
        hand_str = self.cards_to_string(hole_cards)
        bucket = _PREFLOP_BUCKET_MAP.get(hand_str)
        if bucket is None:
            import warnings
            warnings.warn(f"Unrecognized hand string '{hand_str}' — falling back to pf_7")
            return 'pf_7'
        return bucket

    def postflop_bucket(self, hole_cards, community_cards):
        """Distribution-aware postflop bucket (12 flop / 12 turn / 10 river)."""
        return self.postflop.bucket(list(hole_cards), list(community_cards))

    def cards_to_string(self, hole_cards):
        """Convert PyPokerEngine cards to readable format"""
        if isinstance(hole_cards[0], str):
            return self.parse_string_cards(hole_cards)
        else:
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

        return self.format_hand_string(rank1, rank2, card1.suit, card2.suit)

    def format_hand_string(self, rank1, rank2, suit1, suit2):
        """Format two cards into standard notation like 'AKs' or 'AKo'"""
        rank_order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
                      '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

        suited = suit1 == suit2
        if rank_order.get(rank1, 0) >= rank_order.get(rank2, 0):
            high_rank, low_rank = rank1, rank2
        else:
            high_rank, low_rank = rank2, rank1

        if high_rank == low_rank:
            return f"{high_rank}{low_rank}"
        else:
            suffix = 's' if suited else 'o'
            return f"{high_rank}{low_rank}{suffix}"
