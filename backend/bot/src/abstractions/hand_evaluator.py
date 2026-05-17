# backend/bot/abstractions/hand_evaluator.py
from phevaluator.evaluator import evaluate_cards

RANK_MAP = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

_HAND_BOUNDARIES = [
    (10,   'straight_flush'),
    (166,  'four_of_kind'),
    (322,  'full_house'),
    (1599, 'flush'),
    (1609, 'straight'),
    (2467, 'three_of_kind'),
    (3325, 'two_pair'),
    (6185, 'pair'),
]

_HAND_STRENGTH = {
    'high_card': 0, 'pair': 1, 'two_pair': 2, 'three_of_kind': 3,
    'straight': 4, 'flush': 5, 'full_house': 6, 'four_of_kind': 7,
    'straight_flush': 8,
}

_PHEVALUATOR_MAX = 7462
_PHEVALUATOR_RANGE = 7461.0

# Raw score of the worst possible made hand (pair of 2s, worst kickers).
# Made hands are re-normalised within [_MADE_HAND_FLOOR, 1.0] before bucketing.
_MADE_HAND_FLOOR = (_PHEVALUATOR_MAX - 6185) / _PHEVALUATOR_RANGE  # ≈ 0.171


class HandEvaluator:
    """
    Wraps phevaluator. Cards are in SuitRank format (e.g. 'HA' = Heart Ace):
    card[0] = suit letter, card[1] = rank character.
    """

    def __init__(self):
        self.HAND_TYPES = _HAND_STRENGTH

    def convert_card_format(self, card):
        """SuitRank 'HA' → phevaluator 'Ah'."""
        if len(card) != 2:
            return card
        suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
        return card[1] + suit_map.get(card[0], card[0].lower())

    def get_raw_hand_value(self, hole_cards, community_cards):
        """Raw phevaluator score (1–7462, lower = stronger)."""
        converted = [self.convert_card_format(c) for c in hole_cards + community_cards]
        return evaluate_cards(*converted)

    def evaluate_hand_strength(self, hole_cards, community_cards):
        """Return (hand_type_str, strength_int 0–8)."""
        hand_value = self.get_raw_hand_value(hole_cards, community_cards)
        hand_type = self.get_hand_type_from_value(hand_value)
        return hand_type, _HAND_STRENGTH[hand_type]

    def get_hand_type_from_value(self, hand_value):
        for boundary, name in _HAND_BOUNDARIES:
            if hand_value <= boundary:
                return name
        return 'high_card'

    def get_relative_strength(self, hand_value):
        return _HAND_STRENGTH[self.get_hand_type_from_value(hand_value)]

    def has_draw_potential(self, hole_cards, community_cards):
        """True if any draw exists. Always False on the river."""
        if len(community_cards) >= 5:
            return False
        all_cards = hole_cards + community_cards
        return self.has_flush_draw(all_cards) or self.has_straight_draw(all_cards)

    def has_flush_draw(self, cards):
        """Four or more cards of the same suit."""
        suit_counts = {}
        for card in cards:
            suit_counts[card[0]] = suit_counts.get(card[0], 0) + 1
        return any(v >= 4 for v in suit_counts.values())

    def has_straight_draw(self, cards):
        """
        True for open-ended or gutshot draws: 4 cards fall inside any 5-rank window.
        Handles ace-low (A-2-3-4-5) by treating ace as rank 1 as well.
        """
        ranks = set(RANK_MAP[c[1]] for c in cards if c[1] in RANK_MAP)
        if 14 in ranks:
            ranks = ranks | {1}
        for low in range(1, 11):            # windows A–5 through T–A
            window = set(range(low, low + 5))
            if sum(1 for r in window if r in ranks) >= 4:
                return True
        return False

    def is_oesd(self, cards):
        """
        True only for open-ended straight draws: 4 strictly consecutive ranks.
        Handles ace-low.
        """
        ranks = set(RANK_MAP[c[1]] for c in cards if c[1] in RANK_MAP)
        if 14 in ranks:
            ranks = ranks | {1}
        sorted_ranks = sorted(ranks)
        run = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i - 1] + 1:
                run += 1
                if run >= 4:
                    return True
            else:
                run = 1
        return False


class BoardTextureEvaluator:
    """
    Combines phevaluator output with board texture to produce a single
    integer bucket (0–7) representing board-adjusted postflop hand strength.

    Bucket layout:
      0  pure bluff    — high card, no meaningful draw (<4 outs)
      1  weak draw     — gutshot (~4 outs)
      2  strong draw   — flush draw or OESD (~8–9 outs)
      3  combo draw    — flush draw + straight draw (~12+ outs, near-coinflip)
      4  weakest made  — bottom/middle pair or heavily discounted hand
      5  medium made   — top pair, overpair on wet board, weak two pair
      6  strong made   — two pair, trips, straight
      7  near-nuts     — flush, full house, quads, straight flush
    """

    _NUM_DRAW_BUCKETS = 4   # buckets 0-3
    _NUM_MADE_BUCKETS = 4   # buckets 4-7
    _PENALTY_CAP = 0.25     # maximum raw-score discount from board danger (~1-2 buckets)

    def compute_postflop_bucket(self, hole_cards, community_cards, hand_evaluator):
        """Return integer bucket 0–7."""
        hand_value = hand_evaluator.get_raw_hand_value(hole_cards, community_cards)
        hand_type = hand_evaluator.get_hand_type_from_value(hand_value)

        if hand_type == 'high_card':
            return self._draw_bucket(hole_cards, community_cards, hand_evaluator)

        # Normalise phevaluator score: 0.0 (worst pair) → 1.0 (royal flush)
        raw = (_PHEVALUATOR_MAX - hand_value) / _PHEVALUATOR_RANGE

        # Board danger: 0.0 (dry) → 1.0 (maximally threatening)
        danger = self.get_board_danger(community_cards)

        # Vulnerability peaks for mid-strength hands (pairs/two-pair).
        # Monsters and bluffs barely move; one-pair hands move most.
        vulnerability = 4.0 * raw * (1.0 - raw)

        adjusted = max(0.0, raw - danger * vulnerability * self._PENALTY_CAP)

        # Re-normalise within the made-hand range then map to buckets 4–7
        made_norm = max(0.0, (adjusted - _MADE_HAND_FLOOR) / (1.0 - _MADE_HAND_FLOOR))
        made_bucket = min(self._NUM_MADE_BUCKETS - 1, int(made_norm * self._NUM_MADE_BUCKETS))
        return self._NUM_DRAW_BUCKETS + made_bucket

    # ------------------------------------------------------------------
    # Draw classification
    # ------------------------------------------------------------------

    def _draw_bucket(self, hole_cards, community_cards, hand_evaluator):
        """Classify high-card hands by estimated out count."""
        if len(community_cards) >= 5:   # river — no draws remain
            return 0

        all_cards = hole_cards + community_cards
        flush_outs = 9 if hand_evaluator.has_flush_draw(all_cards) else 0

        oesd = hand_evaluator.is_oesd(all_cards)
        gutshot = (not oesd) and hand_evaluator.has_straight_draw(all_cards)
        straight_outs = 8 if oesd else (4 if gutshot else 0)

        # At most 2 cards complete both draws simultaneously
        overlap = 2 if (flush_outs and straight_outs) else 0
        outs = flush_outs + straight_outs - overlap

        if outs >= 12:
            return 3    # combo draw
        if outs >= 8:
            return 2    # strong draw
        if outs >= 4:
            return 1    # weak draw (gutshot)
        return 0        # pure bluff

    # ------------------------------------------------------------------
    # Board danger scoring
    # ------------------------------------------------------------------

    def get_board_danger(self, community_cards):
        """
        0.0–1.0 threat level, river-aware.
        Flop/turn: measures draw potential.
        River:     measures what strong hands have completed.
        """
        if not community_cards:
            return 0.0
        if len(community_cards) == 5:
            return self._river_danger(community_cards)

        fs = self._flush_score(community_cards) / 2.0         # 0.0–1.0
        ss = self._straight_score(community_cards) / 2.0      # 0.0–1.0
        ps = self._paired_board_score(community_cards) * 0.5  # 0.0–0.5
        return min(1.0, fs + ss + ps)

    def _river_danger(self, community_cards):
        """Score based on what strong hands have materialised by the river."""
        suit_counts = {}
        for card in community_cards:
            suit_counts[card[0]] = suit_counts.get(card[0], 0) + 1
        flush_made = 1.0 if max(suit_counts.values()) >= 3 else 0.0

        ranks = sorted(set(RANK_MAP.get(card[1], 0) for card in community_cards))
        straight_possible = 0.0
        for i in range(len(ranks)):
            for j in range(i + 2, len(ranks)):
                if ranks[j] - ranks[i] <= 4:
                    straight_possible = 1.0
                    break

        ps = self._paired_board_score(community_cards) * 0.5
        return min(1.0, flush_made + straight_possible + ps)

    def _paired_board_score(self, community_cards):
        """1 if the board contains a pair (trips/full house now in opponent range)."""
        rank_counts = {}
        for card in community_cards:
            rank_counts[card[1]] = rank_counts.get(card[1], 0) + 1
        return 1 if any(v >= 2 for v in rank_counts.values()) else 0

    def _flush_score(self, community_cards):
        """0 = rainbow, 1 = flush draw possible, 2 = many flush draws."""
        suit_counts = {}
        for card in community_cards:
            suit_counts[card[0]] = suit_counts.get(card[0], 0) + 1
        top = max(suit_counts.values()) if suit_counts else 0
        if top >= 3:
            return 2
        if top >= 2:
            return 1
        return 0

    def _straight_score(self, community_cards):
        """0 = disconnected, 1 = some straight draws, 2 = very connected."""
        if len(community_cards) < 2:
            return 0
        ranks = sorted(set(RANK_MAP.get(card[1], 0) for card in community_cards))
        connected_pairs = sum(
            1 for i in range(len(ranks))
            for j in range(i + 1, len(ranks))
            if ranks[j] - ranks[i] <= 4
        )
        if connected_pairs >= 3:
            return 2
        if connected_pairs >= 1:
            return 1
        return 0
