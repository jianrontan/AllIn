# backend/bot/src/game/cards.py
"""
Card helpers and a deck.

Two card formats exist in this codebase:

  * "engine" format  — SuitRank, e.g. 'HA' (Heart Ace), 'D2' (Diamond 2).
    This is what the training deck, CardAbstraction and HandEvaluator expect.
  * "display" format — RankSuit with a lowercase suit, e.g. 'Ah', '2d'.
    Standard poker notation; what the frontend sends and shows.

The game core works in engine format internally and converts only at the
API boundary, so the rest of the codebase never sees display format.
"""
import random

SUITS = ['H', 'D', 'C', 'S']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

_SUIT_SET = set(SUITS)
_RANK_SET = set(RANKS)


def make_deck():
    """Return a fresh 52-card deck in engine format."""
    return [suit + rank for rank in RANKS for suit in SUITS]


def shuffled_deck(rng=None):
    """Return a shuffled 52-card deck. Pass an rng for reproducibility."""
    deck = make_deck()
    (rng or random).shuffle(deck)
    return deck


def to_engine(card):
    """
    Normalise any reasonable card spelling to engine format 'SuitRank'.

    Accepts 'Ah', 'AH', 'ha', 'HA', etc. Raises ValueError on a bad card.
    """
    if not card or len(card) != 2:
        raise ValueError(f"Invalid card: {card!r}")
    a, b = card[0].upper(), card[1].upper()
    if a in _SUIT_SET and b in _RANK_SET:
        return a + b              # already SuitRank
    if a in _RANK_SET and b in _SUIT_SET:
        return b + a              # RankSuit -> SuitRank
    raise ValueError(f"Invalid card: {card!r}")


def to_display(card):
    """Engine format 'HA' -> display format 'Ah'."""
    engine = to_engine(card)
    return engine[1] + engine[0].lower()


def to_display_list(cards):
    return [to_display(c) for c in cards]
