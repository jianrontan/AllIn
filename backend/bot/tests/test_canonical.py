# backend/bot/tests/test_canonical.py
"""
Validate the suit-isomorphism canonicaliser:

  1. Textbook counts: all 1326 two-card hands collapse to 169 canonical; all
     22,100 three-card flops collapse to 1,755 canonical. (Classic results --
     a strong end-to-end correctness check.)
  2. Isomorphism: a situation and any suit-permutation of it share a key.
  3. Distinctness: non-isomorphic situations get different keys.
  4. Equity preservation: same canonical key => identical equity (sanity).
"""
import os
import sys
import random
from itertools import combinations, permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions.canonical import canonical_key
from src.abstractions.hand_evaluator import HandEvaluator

_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_SUITS = ['H', 'D', 'C', 'S']
_DECK = [s + r for r in _RANKS for s in _SUITS]
EV = HandEvaluator()


def test_counts():
    hands = {canonical_key(list(h)) for h in combinations(_DECK, 2)}
    flops = {canonical_key((), list(b)) for b in combinations(_DECK, 3)}
    print(f"canonical 2-card hands: {len(hands)} (expect 169)")
    print(f"canonical 3-card flops: {len(flops)} (expect 1755)")
    assert len(hands) == 169, len(hands)
    assert len(flops) == 1755, len(flops)


def test_isomorphism():
    rng = random.Random(0)
    for _ in range(2000):
        cards = rng.sample(_DECK, 7)
        hole, board = cards[:2], cards[2:]
        # apply a random suit permutation
        perm = dict(zip(_SUITS, rng.sample(_SUITS, 4)))
        h2 = [perm[c[0]] + c[1] for c in hole]
        b2 = [perm[c[0]] + c[1] for c in board]
        assert canonical_key(hole, board) == canonical_key(h2, b2), (hole, board, perm)
    print("isomorphism: 2000 random situations invariant under suit permutation OK")


def test_distinctness():
    # Different rank structure -> different key.
    assert canonical_key(['HA', 'HK']) != canonical_key(['HA', 'HQ'])
    # Suited vs offsuit AK -> different key.
    assert canonical_key(['HA', 'HK']) != canonical_key(['HA', 'SK'])
    # Hole/board roles are not interchangeable.
    assert canonical_key(['HA', 'HK'], ['DQ', 'DJ', 'D2']) != \
           canonical_key(['DQ', 'DJ'], ['HA', 'HK', 'D2'])
    print("distinctness: rank/suitedness/role differences produce different keys OK")


def test_equity_preserved():
    """All suit-permutations of a situation must share both key and equity."""
    rng = random.Random(1)
    for _ in range(200):
        cards = rng.sample(_DECK, 7)
        hole, board5 = cards[:2], cards[2:]
        base = EV.get_raw_hand_value(hole, board5)
        for perm_t in random.sample(list(permutations(_SUITS)), 5):
            perm = dict(zip(_SUITS, perm_t))
            h2 = [perm[c[0]] + c[1] for c in hole]
            b2 = [perm[c[0]] + c[1] for c in board5]
            assert canonical_key(hole, board5) == canonical_key(h2, b2)
            assert EV.get_raw_hand_value(h2, b2) == base   # equity invariant
    print("equity preservation: permuted situations keep key AND raw value OK")


def main():
    test_counts()
    test_isomorphism()
    test_distinctness()
    test_equity_preserved()
    print("\nPASS: canonicaliser correct (169 hands / 1755 flops, isomorphic, distinct).")


if __name__ == '__main__':
    main()
