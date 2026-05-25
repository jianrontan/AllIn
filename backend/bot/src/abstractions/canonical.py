# backend/bot/src/abstractions/canonical.py
"""
Suit-isomorphism canonicalisation of poker situations.

A situation's strategy/equity is invariant under any *consistent* relabeling of
the four suits: e.g. (A_h K_h on Q_h J_h 2_h) plays identically to
(A_s K_s on Q_s J_s 2_s) -- swap hearts<->spades. So every (hole, board) maps to
a CANONICAL representative shared by all 24 suit-permutations of it. This is what
lets the postflop abstraction store/compute one bucket per equivalence class
instead of per concrete situation (a ~10-24x reduction).

`canonical_key(hole, board)` returns a hashable canonical representation:
  ((rank, suit), ...) for the hole, and likewise for the board, under the suit
  permutation that minimises the pair lexicographically. Ranks are never
  permuted (they carry absolute meaning); only suit *labels* are.

Hole and board are kept distinct (a card in hand plays differently than one on
the board). Within hole and within board, cards are order-insensitive (sorted).
"""
from itertools import permutations

_RANKS = '23456789TJQKA'
_RANK_ORD = {r: i for i, r in enumerate(_RANKS)}
_SUITS = ['H', 'D', 'C', 'S']
_SUIT_PERMS = [dict(zip(_SUITS, p)) for p in permutations(_SUITS)]   # all 24


def _rep(cards, smap):
    """Sorted ((rank_ord, suit), ...) for `cards` under suit relabeling `smap`.
    Cards are SuitRank strings, e.g. 'HA' = Ace of hearts (card[0]=suit, [1]=rank)."""
    return tuple(sorted((_RANK_ORD[c[1]], smap[c[0]]) for c in cards))


def canonical_key(hole, board=()):
    """
    Canonical (suit-isomorphic) key for a situation. Two situations are equal
    under some suit permutation iff their canonical_key values are equal.
    """
    hole = tuple(hole)
    board = tuple(board)
    best = None
    for smap in _SUIT_PERMS:
        rep = (_rep(hole, smap), _rep(board, smap))
        if best is None or rep < best:
            best = rep
    return best


def canonical_str(hole, board=()):
    """Human-readable canonical key, e.g. 'Ah Ks | Qh Jd 2c' (debug/logging)."""
    inv = {i: r for r, i in _RANK_ORD.items()}
    h, b = canonical_key(hole, board)
    fmt = lambda cs: ' '.join(f"{inv[r]}{s.lower()}" for r, s in cs)
    return f"{fmt(h)} | {fmt(b)}" if b else fmt(h)
