# backend/bot/tests/test_best_response_vectorized.py
"""
Validate the vectorized showdown terminal (the only non-mechanical part of the
vectorized best-response rewrite) against a brute-force O(H^2) oracle.

The walk's sum/max propagation is a mechanical lift of the previously-validated
scalar version; the risk is concentrated in `_showdown_measure` (card removal +
the O(H) per-card running-sums trick). We check it on random boards and random
villain reach vectors against the naive definition:

    value[h] = sum over villain v sharing NO card with h of rv[v] * payoff(h, v)

with payoff = final_pot - hero_total (hero stronger, lower raw),
              -hero_total          (hero weaker),
              final_pot/2 - hero_total (tie).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.best_response import BestResponseEvaluator, _FULL_DECK, _COMPATIBLE


def _brute_force(ba, rv, final_pot, hero_total):
    raw = ba['raw']
    c1 = ba['c1']
    c2 = ba['c2']
    H = ba['H']
    out = np.empty(H)
    for h in range(H):
        a1, a2 = c1[h], c2[h]
        raw_h = raw[h]
        acc = 0.0
        for v in range(H):
            if c1[v] == a1 or c1[v] == a2 or c2[v] == a1 or c2[v] == a2:
                continue  # blocked: shares a card with hero hand
            if raw[v] > raw_h:
                payoff = final_pot - hero_total      # hero stronger -> wins
            elif raw[v] < raw_h:
                payoff = -hero_total                 # hero weaker -> loses
            else:
                payoff = final_pot / 2.0 - hero_total
            acc += rv[v] * payoff
        out[h] = acc
    return out


def main():
    ev = BestResponseEvaluator(blueprint_db=None, seed=0)
    rng = np.random.default_rng(123)

    worst = 0.0
    n_boards = 4
    for b in range(n_boards):
        board = list(np.random.default_rng(b).choice(_FULL_DECK, size=5, replace=False))
        ba = ev._board_arrays(board)
        H = ba['H']

        # Sanity: every hero hand blocks exactly the same number of villains.
        c1, c2 = ba['c1'], ba['c2']
        blocked0 = sum(1 for v in range(H)
                       if c1[v] in (c1[0], c2[0]) or c2[v] in (c1[0], c2[0]))
        assert H - blocked0 == _COMPATIBLE, (H, blocked0, H - blocked0)

        for trial in range(3):
            rv = rng.random(H)
            final_pot = float(rng.integers(4, 400))
            hero_total = float(rng.integers(1, int(final_pot)))
            got = ev._showdown_measure(ba, rv, final_pot, hero_total)
            want = _brute_force(ba, rv, final_pot, hero_total)
            err = float(np.max(np.abs(got - want)))
            worst = max(worst, err)
            print(f"board {b} trial {trial}: H={H} pot={final_pot:.0f} "
                  f"hero_total={hero_total:.0f} max|err|={err:.3e}")

    print(f"\nworst max|err| across all cases: {worst:.3e}")
    assert worst < 1e-6, f"vectorized showdown disagrees with brute force: {worst}"
    print("PASS: vectorized showdown matches brute-force oracle.")


if __name__ == '__main__':
    main()
