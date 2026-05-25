# backend/bot/src/abstractions/postflop_features.py
"""
Shared postflop-abstraction primitives, used by BOTH the offline clustering
script (scripts/compute_postflop_buckets.py) and the offline table baker
(scripts/bake_postflop_table.py), and at runtime by CardAbstraction.

Keeping the equity-distribution feature in ONE place is essential: the baker
must compute features the *same way* the centroids were fit, or bucket
assignments would be inconsistent with the clustering.

Contents:
  - equity_distribution(): the per-runout win-rate histogram (the bucket feature)
  - emd() / assign(): 1-D Earth Mover's Distance + nearest-centroid assignment
  - load_centroids(): read analysis/abstractions/postflop_centroids_<street>.npz
  - encode_situation() / enumerate_canonical(): compact int id per canonical
    (hole, board) situation, and an enumerator over all of them for a street.
"""
import os
from itertools import combinations

import numpy as np
from phevaluator.card import Card
from phevaluator.evaluator import _evaluate_cards

from .canonical import canonical_key

SUITS = ['H', 'D', 'C', 'S']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
DECK = [s + r for r in RANKS for s in SUITS]
ALL_HANDS = list(combinations(DECK, 2))
_SUIT_MAP = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
_SUIT_IDX = {s: i for i, s in enumerate(SUITS)}    # matches canonical._SUITS order
STREET_BOARD = {'flop': 3, 'turn': 4, 'river': 5}

_ABSTRACTIONS_DIR = (os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'analysis', 'abstractions'))


def to_phev(card):
    return card[1] + _SUIT_MAP[card[0]]


# Precompute each engine card ('HA' ...) -> phevaluator integer id ONCE. The
# public evaluate_cards() re-parses card strings via Card.to_id on every call
# (~30s / 30M calls in a training profile); passing precomputed ids straight to
# the internal _evaluate_cards bypasses that entirely.
_PHEV_ID = {c: Card.to_id(to_phev(c)) for c in DECK}


def rank7(cards):
    return _evaluate_cards(*[_PHEV_ID[c] for c in cards])


# ----------------------------------------------------------------------
# Equity distribution feature (must stay identical between clustering & baking)
# ----------------------------------------------------------------------

def hand_winrate(hole, board5, opp_hands):
    """Win-rate (win + 0.5 tie) of `hole` vs a uniform opp range on a 5-card board."""
    hr = rank7(list(hole) + list(board5))
    win = tie = 0
    for opp in opp_hands:
        orank = rank7(list(opp) + list(board5))
        if hr < orank:        # lower phevaluator score = stronger
            win += 1
        elif hr == orank:
            tie += 1
    n = len(opp_hands)
    return (win + 0.5 * tie) / n if n else 0.5


def equity_distribution(hole, board, bins, n_runout, n_opp, rng):
    """Histogram of per-runout win-rates -> the hand's equity distribution."""
    dead = set(hole) | set(board)
    deck = [c for c in DECK if c not in dead]
    need = 5 - len(board)

    if need == 0:
        runouts = [()]
    elif need == 1:
        runouts = [(c,) for c in deck]                  # enumerate the 44-46 rivers
    else:
        runouts = [tuple(rng.sample(deck, need)) for _ in range(n_runout)]

    winrates = []
    for ro in runouts:
        full = board + list(ro)
        blocked = dead | set(ro)
        opp = [h for h in ALL_HANDS if h[0] not in blocked and h[1] not in blocked]
        if n_opp and len(opp) > n_opp:
            opp = rng.sample(opp, n_opp)
        winrates.append(hand_winrate(hole, full, opp))

    hist, _ = np.histogram(winrates, bins=bins, range=(0.0, 1.0))
    h = hist.astype(float)
    s = h.sum()
    return h / s if s > 0 else h


def emd(a, b):
    """1-D Earth Mover's Distance = L1 between CDFs."""
    return np.abs(np.cumsum(a) - np.cumsum(b)).sum()


def assign(feat, centroids):
    """Nearest centroid (bucket id) by EMD."""
    return int(np.argmin([emd(feat, c) for c in centroids]))


def load_centroids(street):
    """Load the Phase-A centroids for a street: returns (centroids, bins)."""
    path = os.path.join(_ABSTRACTIONS_DIR, f'postflop_centroids_{street}.npz')
    data = np.load(path)
    return data['centroids'], int(data['bins'])


_CARD_IDX = {c: i for i, c in enumerate(DECK)}     # global 0..51


def board_winrates(board5):
    """
    Win-rate (win + 0.5*tie) vs a UNIFORM opponent range, with exact card
    removal, for EVERY 2-card hand on a fixed 5-card board -- all at once.

    Returns (c1, c2, winrate): card-index arrays (0..51) for the two cards of
    each hand over the 47 cards not on the board, and that hand's equity. Uses
    the same O(H)
    per-card running-sums method validated in evaluation/best_response.py
    (weights = 1 here, so masses are counts). This is the hot primitive that
    makes baking tractable: one board's ranking is shared across all holes.
    """
    dead = set(board5)
    cards = [c for c in DECK if c not in dead]            # 47
    hands = list(combinations(cards, 2))
    H = len(hands)                                        # 1081
    bl = list(board5)
    raw = np.array([rank7([h[0], h[1]] + bl) for h in hands])
    c1 = np.array([_CARD_IDX[h[0]] for h in hands])
    c2 = np.array([_CARD_IDX[h[1]] for h in hands])
    NC = 52

    uniq = np.unique(raw)
    g = np.searchsorted(uniq, raw)                        # 0 = strongest
    G = len(uniq)

    cardTot = (np.bincount(c1, minlength=NC) + np.bincount(c2, minlength=NC)).astype(float)
    groupSum = np.bincount(g, minlength=G).astype(float)
    strongerGroupCum = np.cumsum(groupSum) - groupSum
    gNC = g * NC
    gc = (np.bincount(gNC + c1, minlength=G * NC) +
          np.bincount(gNC + c2, minlength=G * NC)).astype(float).reshape(G, NC)
    strongerCardCum = np.cumsum(gc, axis=0) - gc

    sg = strongerGroupCum[g]
    grp = groupSum[g]
    wk = H - sg - grp                                     # weaker (hero beats), no removal
    scc1, scc2 = strongerCardCum[g, c1], strongerCardCum[g, c2]
    gcc1, gcc2 = gc[g, c1], gc[g, c2]
    winM = wk - (cardTot[c1] - scc1 - gcc1) - (cardTot[c2] - scc2 - gcc2)
    tieM = grp - gcc1 - gcc2 + 1.0                        # disjoint ties (drop self)
    compat = H - (cardTot[c1] + cardTot[c2] - 1.0)        # disjoint opp count (=990)
    return c1, c2, (winM + 0.5 * tieM) / compat


# ----------------------------------------------------------------------
# Canonical situation encoding + enumeration (Phase B table support)
# ----------------------------------------------------------------------

def encode_situation(key):
    """
    Pack a canonical_key ((rank,suit) tuples for hole then board) into a single
    int id. Injective for a fixed board length, so different canonical
    situations get different ids -- usable as a compact table key.
    """
    sid = 0
    for r, s in key[0] + key[1]:          # hole cards then board cards
        sid = sid * 52 + (r * 4 + _SUIT_IDX[s])
    return sid


def enumerate_canonical(street):
    """
    Yield (situation_id, hole, board) for every canonical (hole, board) on
    `street`, each with one concrete representative. Works by enumerating
    canonical BOARDS (small) then all holes on each -- so it never has to touch
    the billions of concrete situations, only the ~millions of canonical ones.
    """
    nboard = STREET_BOARD[street]

    # One representative concrete board per canonical board.
    rep_boards = {}
    for board in combinations(DECK, nboard):
        bkey = canonical_key((), board)
        if bkey not in rep_boards:
            rep_boards[bkey] = list(board)

    seen = set()
    for board in rep_boards.values():
        dead = set(board)
        remaining = [c for c in DECK if c not in dead]
        for hole in combinations(remaining, 2):
            sid = encode_situation(canonical_key(hole, board))
            if sid not in seen:
                seen.add(sid)
                yield sid, list(hole), board
