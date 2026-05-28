# backend/bot/src/evaluation/showdown_kernel.py
"""
Shared, exact range-vs-range showdown kernel + per-board precompute.

Extracted from best_response.py so the best-response evaluator AND the Phase-4
river subgame solver use ONE validated core, rather than the solver
re-implementing card-removal showdown and risking drift. Pure NumPy -- no DB, no
Flask, no game/CFR imports. Validated against a brute-force O(H^2) oracle in
tests/test_showdown_kernel.py (and, via the delegating evaluator, in
tests/test_best_response_vectorized.py).

WHAT'S HERE
-----------
* build_board_arrays(board, evaluator, cards) -> dict of per-board arrays that
  do NOT depend on the betting line or which seat is hero: the hand list,
  showdown ranks, integer card ids, dense strength groups, per-hand preflop /
  postflop buckets, and precomputed villain-key group masks (hands sharing a
  blueprint key share a strategy row).
* compatible_mass(ba, rv) -> per-hero-hand reach of villain hands that share NO
  card with the hero hand (the O(H) per-card running-sums card-removal trick).
  This is the denominator/normaliser for any reach-weighted range-vs-range
  quantity and the mass a fold terminal transfers.
* showdown_measure(ba, rv, final_pot, hero_total) -> per-hero-hand showdown value
  (a MEASURE, hero perspective) vs a villain reach vector, with card removal.

Everything here is a MEASURE: the villain reach `rv` is used as-is. The uniform-
exploitability normaliser (dividing by the constant compatible-hand count) lives
in the BR estimator, not here -- a subgame solve passes a NON-uniform reach and
must not divide by that constant.
"""
from itertools import combinations

import numpy as np

_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_SUITS = ['H', 'D', 'C', 'S']
_FULL_DECK = [s + r for r in _RANKS for s in _SUITS]
_CARD_ID = {c: i for i, c in enumerate(_FULL_DECK)}
_NUM_CARDS = len(_FULL_DECK)  # 52

# Every card appears in 46 of the C(47,2)=1081 hands; each hero hand is blocked
# by 46 + 46 - 1 = 91 villain hands, leaving C(45,2) = 990 compatible. Only valid
# for a 5-card board (H = 1081); it is the uniform normaliser, not used here.
_COMPATIBLE = 990


def build_board_arrays(board, evaluator, cards):
    """
    Precompute, for one board, everything independent of the betting line or
    which seat is hero:

      hands : list of (cardA, cardB) for all H hands not using a board card
      raw   : showdown rank per hand  (lower = stronger)
      c1,c2 : integer card ids per hand (for card-removal at terminals)
      g, G  : dense strength-group id per hand (0 = strongest) and group count
      gNC   : g * 52, precomputed offset for the per-(group,card) bincounts
      pf    : preflop bucket per hand   (object array, for villain keys)
      strg  : {street: bucket-per-hand} for flop(1)/turn(2)/river(3) keys
      groups: {street: [(mask, rep_idx), ...]} hands sharing a blueprint key,
              so the model is queried once per distinct key
    """
    board_set = set(board)
    pool = [c for c in _FULL_DECK if c not in board_set]
    hands = list(combinations(pool, 2))
    H = len(hands)

    raw = np.empty(H)
    c1 = np.empty(H, dtype=np.int64)
    c2 = np.empty(H, dtype=np.int64)
    pf = [None] * H
    s1 = [None] * H
    s2 = [None] * H
    s3 = [None] * H
    for i, (a, b) in enumerate(hands):
        hl = [a, b]
        raw[i] = evaluator.get_raw_hand_value(hl, board)
        c1[i] = _CARD_ID[a]
        c2[i] = _CARD_ID[b]
        pf[i] = cards.get_bucket(hl, None)
        s1[i] = cards.get_bucket(hl, board[:3])
        s2[i] = cards.get_bucket(hl, board[:4])
        s3[i] = cards.get_bucket(hl, board[:5])

    # Dense strength groups: ascending raw -> group 0 is the strongest.
    uniq = np.unique(raw)
    g = np.searchsorted(uniq, raw)
    G = len(uniq)

    pf = np.array(pf, dtype=object)
    strg = {1: np.array(s1, dtype=object),
            2: np.array(s2, dtype=object),
            3: np.array(s3, dtype=object)}

    # Villain hands sharing the same blueprint key share a strategy row. The
    # grouping (preflop bucket, or (preflop, strength) postflop) is independent
    # of position/pattern/legal-set, so precompute masks + a representative hand
    # per group ONCE. groups[street] = list of (mask, rep_idx).
    def build_groups(labels):
        out = []
        for lab in set(labels.tolist()):
            mask = labels == lab
            out.append((mask, int(np.argmax(mask))))
        return out

    groups = {0: build_groups(pf)}
    for s in (1, 2, 3):
        labels = np.array([f"{pf[i]}|{strg[s][i]}" for i in range(H)], dtype=object)
        groups[s] = build_groups(labels)

    return {
        'hands': hands, 'H': H, 'raw': raw, 'c1': c1, 'c2': c2,
        'g': g, 'G': G, 'gNC': g.astype(np.int64) * _NUM_CARDS,
        'pf': pf, 'strg': strg, 'groups': groups,
    }


def compatible_mass(ba, rv):
    """
    Per-hero-hand reach of villain hands sharing NO card with the hero hand:
    compatM[h] = sum over villain v compatible with h of rv[v]. O(H) via per-card
    running sums. Used as the reach a fold terminal transfers, and anywhere a
    reach-weighted range-vs-range quantity needs the compatible denominator.
    """
    c1 = ba['c1']
    c2 = ba['c2']
    total = float(rv.sum())
    cardTot = (np.bincount(c1, weights=rv, minlength=_NUM_CARDS) +
               np.bincount(c2, weights=rv, minlength=_NUM_CARDS))
    return total - (cardTot[c1] + cardTot[c2] - rv)


def showdown_measure(ba, rv, final_pot, hero_total):
    """
    Vectorized showdown value (measure, hero perspective) per hero hand, with
    card removal. Lower raw = stronger; hero wins vs WEAKER villains (villain
    raw > hero raw). O(H) via per-card running sums. Validated against a
    brute-force oracle.

    payoff = winnings - hero_total; winnings = final_pot (win), final_pot/2
    (tie), 0 (lose). The returned value is reach-weighted (a measure); compatM
    (== winM + tieM + loseM) is the compatible villain mass per hero hand.
    """
    c1 = ba['c1']
    c2 = ba['c2']
    g = ba['g']
    G = ba['G']

    total = float(rv.sum())
    cardTot = (np.bincount(c1, weights=rv, minlength=_NUM_CARDS) +
               np.bincount(c2, weights=rv, minlength=_NUM_CARDS))
    compatM = total - (cardTot[c1] + cardTot[c2] - rv)

    groupSum = np.bincount(g, weights=rv, minlength=G)
    cum = np.cumsum(groupSum)
    strongerGroupCum = cum - groupSum            # reach of strictly stronger groups
    gNC = ba['gNC']
    flat = (np.bincount(gNC + c1, weights=rv, minlength=G * _NUM_CARDS) +
            np.bincount(gNC + c2, weights=rv, minlength=G * _NUM_CARDS))
    gc = flat.reshape(G, _NUM_CARDS)
    gcum = np.cumsum(gc, axis=0)
    strongerCardCum = gcum - gc                  # reach of stronger groups, per card

    sg = strongerGroupCum[g]                     # stronger total (no removal)
    grp = groupSum[g]                            # tie-group total
    wk = total - sg - grp                        # weaker total (no removal)

    scc1 = strongerCardCum[g, c1]
    scc2 = strongerCardCum[g, c2]
    gcc1 = gc[g, c1]
    gcc2 = gc[g, c2]

    loseM = sg - scc1 - scc2                     # stronger villains, compatible
    tieM = grp - gcc1 - gcc2 + rv                # tied villains, compatible (drop self)
    wcc1 = cardTot[c1] - scc1 - gcc1
    wcc2 = cardTot[c2] - scc2 - gcc2
    winM = wk - wcc1 - wcc2                       # weaker villains, compatible

    # compatM == winM + tieM + loseM. payoff = final_pot (win) / final_pot/2
    # (tie) / 0 (lose), minus hero_total over every compatible villain hand.
    return final_pot * winM + (final_pot / 2.0) * tieM - hero_total * compatM
