# backend/bot/tests/test_lbr_equity.py
"""
Validate LBREvaluator.equity_vs_range against independent brute-force oracles:
  * river  -> exact (no runout), compared to a direct weighted tally.
  * turn   -> enumerate rivers, compared to an independent enumeration.
  * flop   -> sampled, compared to full two-card enumeration (loose tolerance).
"""
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.lbr import LBREvaluator, _FULL_DECK
from src.abstractions.hand_evaluator import HandEvaluator

EV = HandEvaluator()


def brute_equity(lbr_hand, board, range_hands, range_w):
    """Full enumeration of all runouts; exact reference."""
    dead = set(board) | set(lbr_hand)
    undealt = [c for c in _FULL_DECK if c not in dead]
    need = 5 - len(board)
    if need <= 0:
        runouts = [()]
    elif need == 1:
        runouts = [(c,) for c in undealt]
    else:
        runouts = list(combinations(undealt, 2))

    total, n = 0.0, 0
    for ro in runouts:
        ro_set = set(ro)
        full = board + list(ro)
        lbr_raw = EV.get_raw_hand_value(list(lbr_hand), full)
        eq, wt = 0.0, 0.0
        for h, w in zip(range_hands, range_w):
            if w <= 0 or h[0] in dead or h[1] in dead:
                continue
            if h[0] in ro_set or h[1] in ro_set:
                continue
            hr = EV.get_raw_hand_value(list(h), full)
            eq += w * (1.0 if lbr_raw < hr else 0.5 if lbr_raw == hr else 0.0)
            wt += w
        if wt > 0:
            total += eq / wt
            n += 1
    return total / n if n else 0.5


def main():
    lbr = LBREvaluator(blueprint_db=None, seed=1, flop_runout_samples=400)

    # A fixed scenario: LBR holds As Ks, a small mixed bot range.
    lbr_hand = ('SA', 'SK')
    rng_hands = [('HQ', 'DQ'), ('C7', 'D2'), ('HA', 'CA'),
                 ('S5', 'S6'), ('DT', 'DJ'), ('C9', 'H9')]
    rng_w = [2.0, 1.0, 0.5, 1.5, 1.0, 0.8]

    worst = 0.0
    # River (5-card board) -- exact.
    river = ['HT', 'D4', 'C2', 'S9', 'DK']
    got = lbr.equity_vs_range(lbr_hand, river, rng_hands, rng_w)
    want = brute_equity(lbr_hand, river, rng_hands, rng_w)
    print(f"river: got={got:.6f} want={want:.6f} |err|={abs(got-want):.2e}")
    worst = max(worst, abs(got - want))

    # Turn (4-card board) -- both enumerate rivers, exact.
    turn = ['HT', 'D4', 'C2', 'S9']
    got = lbr.equity_vs_range(lbr_hand, turn, rng_hands, rng_w)
    want = brute_equity(lbr_hand, turn, rng_hands, rng_w)
    print(f"turn : got={got:.6f} want={want:.6f} |err|={abs(got-want):.2e}")
    worst = max(worst, abs(got - want))

    # Flop (3-card board) -- sampled vs full enumeration (loose tolerance).
    flop = ['HT', 'D4', 'C2']
    got = lbr.equity_vs_range(lbr_hand, flop, rng_hands, rng_w)
    want = brute_equity(lbr_hand, flop, rng_hands, rng_w)
    print(f"flop : got={got:.6f} want={want:.6f} |err|={abs(got-want):.2e}  (sampled)")

    assert worst < 1e-9, f"river/turn must be exact, worst={worst}"
    assert abs(got - want) < 0.02, f"flop sampled too far from exact: {abs(got-want)}"
    print("\nPASS: equity_vs_range matches brute force (river/turn exact, flop within tol).")


if __name__ == '__main__':
    main()
