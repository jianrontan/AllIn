# backend/bot/abstractions/card_abstractions.py
from .hand_evaluator import HandEvaluator
from .postflop_v2 import PostflopV2

# Precomputed via scripts/compute_preflop_equity.py
# 10,000 Monte Carlo simulations vs random opponent, seed=42. This is the raw equity
# table; the fine (30) and coarse (10) bucket MAPS are derived from it below by
# _quantile_buckets (no pasted bucket literal).
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

# --- Preflop bucketing: DECOUPLED fine (preflop) vs coarse (postflop) ------------
# Two independent equal-frequency quantilings of the SAME equity table above:
#   * FINE  (30): identifies the hand for PREFLOP keys -- sharp preflop play. Preflop
#                 is only 169 distinct hands, so fine resolution is cheap.
#   * COARSE (10): the preflop-hand summary carried into POSTFLOP keys as `startBucket`
#                 -- imperfect recall (Libratus/Pluribus structure). The postflop key
#                 still knows roughly which range you came in with; it just doesn't
#                 distinguish all 30 fine buckets, which collapses the postflop
#                 info-set count (startBucket x strength) far below carrying the fine id.
# The fine->coarse collapse happens in cfr/keys.make_info_set_key for postflop streets
# (FINE_TO_COARSE below), so callers keep passing the fine bucket and the contract
# (postflop keys carry coarse) is impossible to violate at a call site.
#
# Both maps are DERIVED here from _PREFLOP_EQUITY by the same method as
# scripts/compute_preflop_equity.py:assign_buckets -- the script remains the equity
# GENERATOR, but the bucket maps are derived from the committed equity table so the
# two can never drift (and there is no 169-line literal to maintain per scheme).
NUM_PREFLOP_BUCKETS = 169  # fine, LOSSLESS: one bucket per canonical preflop hand
                           # (pf_0 weakest .. pf_168 strongest). _quantile_buckets with
                           # n == len(table) assigns each hand its own rank -> perfect
                           # preflop resolution. Only PREFLOP keys grow (169 hands);
                           # the coarse-10 carry below keeps POSTFLOP key count unchanged.
NUM_PREFLOP_COARSE = 10    # coarse (postflop startBucket: 0 weakest .. 9 strongest)


def _quantile_buckets(equity_map, n_buckets):
    """Equal-frequency quantile bucketing of the 169 hands by equity (ascending):
    0 = weakest .. n_buckets-1 = strongest, ~169/n_buckets hands each. Identical
    formula to scripts/compute_preflop_equity.py:assign_buckets. Stable sort over the
    committed equity dict -> deterministic across runs (dict insertion order breaks ties)."""
    ranked = sorted(equity_map.items(), key=lambda kv: kv[1])
    total = len(ranked)
    return {hand: min(int(i * n_buckets / total), n_buckets - 1)
            for i, (hand, _eq) in enumerate(ranked)}


_FINE_IDX = _quantile_buckets(_PREFLOP_EQUITY, NUM_PREFLOP_BUCKETS)      # hand -> 0..29
_COARSE_IDX = _quantile_buckets(_PREFLOP_EQUITY, NUM_PREFLOP_COARSE)     # hand -> 0..9
_PREFLOP_BUCKET_MAP = {h: f"pf_{b}" for h, b in _FINE_IDX.items()}      # hand -> 'pf_<n>'


def _build_fine_to_coarse():
    """Fine bucket int -> coarse class int. With LOSSLESS fine buckets (169 = one per
    canonical hand) each fine bucket holds exactly ONE hand, so it maps to exactly one
    coarse class trivially -- the collapse is exact and the assertion below can never
    fire here. (It still guards the general case: were fine ever a coarser quantiling
    that didn't nest inside coarse, a straddling bucket would trip it.)"""
    from collections import defaultdict
    by_fine = defaultdict(list)
    for hand, fb in _FINE_IDX.items():
        by_fine[fb].append(hand)
    out = [0] * NUM_PREFLOP_BUCKETS
    for fb, hands in by_fine.items():
        coarse_classes = {_COARSE_IDX[h] for h in hands}
        assert len(coarse_classes) == 1, (
            f"fine bucket pf_{fb} straddles coarse classes {sorted(coarse_classes)} "
            f"-- the fine->coarse collapse is no longer exact (check NUM_PREFLOP_BUCKETS "
            f"/ NUM_PREFLOP_COARSE divisibility)")
        out[fb] = coarse_classes.pop()
    return tuple(out)


# Indexed by fine bucket int; consumed by cfr/keys.make_info_set_key for postflop keys.
FINE_TO_COARSE = _build_fine_to_coarse()

assert len(set(_PREFLOP_BUCKET_MAP.values())) == NUM_PREFLOP_BUCKETS
assert len(set(_COARSE_IDX.values())) == NUM_PREFLOP_COARSE


class CardAbstraction:
    """
    Preflop: DECOUPLED buckets -- 169 FINE buckets (LOSSLESS: one per canonical hand,
    pf_0 weakest → pf_168 strongest) for preflop keys, collapsed to 10 COARSE classes
    (0..9) for the postflop `startBucket` (imperfect recall; see _build_fine_to_coarse). The
    collapse lives in cfr/keys.make_info_set_key, so preflop_bucket() always returns
    the fine id and callers never choose.
    Postflop: distribution-aware (potential-aware) buckets via PostflopV2 --
    20 flop / 16 turn / 10 river, from precomputed equity-distribution centroids
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
        """FINE preflop bucket 'pf_0'..'pf_168' (for PREFLOP keys). Postflop keys
        collapse this to the coarse class inside make_info_set_key -- callers always
        pass this fine id."""
        hand_str = self.cards_to_string(hole_cards)
        bucket = _PREFLOP_BUCKET_MAP.get(hand_str)
        if bucket is None:
            import warnings
            mid = f"pf_{NUM_PREFLOP_BUCKETS // 2}"
            warnings.warn(f"Unrecognized hand string '{hand_str}' — falling back to {mid}")
            return mid
        return bucket

    def preflop_class(self, hole_cards):
        """COARSE preflop class 0..9 (the postflop `startBucket`). Rarely needed
        directly -- make_info_set_key collapses the fine bucket for postflop keys --
        but exposed for diagnostics / the API vocabulary."""
        hand_str = self.cards_to_string(hole_cards)
        cls = _COARSE_IDX.get(hand_str)
        return cls if cls is not None else NUM_PREFLOP_COARSE // 2

    def postflop_bucket(self, hole_cards, community_cards):
        """Distribution-aware postflop bucket (20 flop / 16 turn / 10 river; the
        per-street K is whatever the loaded centroids define)."""
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
