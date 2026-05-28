# backend/bot/tests/test_showdown_kernel.py
"""
Validate the shared showdown kernel (src/evaluation/showdown_kernel.py) against
brute-force O(H^2) oracles. This is the exact range-vs-range core that BOTH the
best-response evaluator and the Phase-4 river subgame solver build on, so it is
pinned independently of either consumer:

  * showdown_measure(ba, rv, final_pot, hero_total) — per-hero-hand reach-weighted
    showdown value with card removal.
  * compatible_mass(ba, rv) — per-hero-hand reach of card-compatible villains.
  * build_board_arrays — basic shape/consistency (and that the delegating
    BestResponseEvaluator still produces identical board arrays).

The kernel must work with NON-uniform reach (the tracker's belief / a subgame
range), so the oracle uses random reach vectors, not just the uniform vector.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.showdown_kernel import (
    build_board_arrays, showdown_measure, compatible_mass,
    _FULL_DECK, _COMPATIBLE)
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction


def _brute_showdown(ba, rv, final_pot, hero_total):
    raw, c1, c2, H = ba['raw'], ba['c1'], ba['c2'], ba['H']
    out = np.empty(H)
    for h in range(H):
        a1, a2, raw_h = c1[h], c2[h], raw[h]
        acc = 0.0
        for v in range(H):
            if c1[v] in (a1, a2) or c2[v] in (a1, a2):
                continue  # shares a card with hero hand -> impossible
            if raw[v] > raw_h:
                payoff = final_pot - hero_total       # hero stronger -> wins
            elif raw[v] < raw_h:
                payoff = -hero_total                  # hero weaker -> loses
            else:
                payoff = final_pot / 2.0 - hero_total
            acc += rv[v] * payoff
        out[h] = acc
    return out


def _brute_compatible(ba, rv):
    c1, c2, H = ba['c1'], ba['c2'], ba['H']
    out = np.empty(H)
    for h in range(H):
        a1, a2 = c1[h], c2[h]
        out[h] = sum(rv[v] for v in range(H)
                     if c1[v] not in (a1, a2) and c2[v] not in (a1, a2))
    return out


# One shared, deterministic set of deps (no DB needed for the kernel).
_EVAL = HandEvaluator()
_CARDS = CardAbstraction()


def test_showdown_matches_oracle():
    rng = np.random.default_rng(123)
    worst = 0.0
    for b in range(4):
        board = list(np.random.default_rng(b).choice(_FULL_DECK, size=5, replace=False))
        ba = build_board_arrays(board, _EVAL, _CARDS)
        for _ in range(3):
            rv = rng.random(ba['H'])
            final_pot = float(rng.integers(4, 400))
            hero_total = float(rng.integers(1, int(final_pot)))
            got = showdown_measure(ba, rv, final_pot, hero_total)
            want = _brute_showdown(ba, rv, final_pot, hero_total)
            worst = max(worst, float(np.max(np.abs(got - want))))
    assert worst < 1e-6, f"showdown_measure disagrees with oracle: {worst}"
    print(f"PASS test_showdown_matches_oracle (worst |err|={worst:.2e})")


def test_compatible_mass_matches_oracle():
    rng = np.random.default_rng(7)
    worst = 0.0
    for b in range(3):
        board = list(np.random.default_rng(100 + b).choice(_FULL_DECK, size=5, replace=False))
        ba = build_board_arrays(board, _EVAL, _CARDS)
        for _ in range(2):
            rv = rng.random(ba['H'])
            got = compatible_mass(ba, rv)
            want = _brute_compatible(ba, rv)
            worst = max(worst, float(np.max(np.abs(got - want))))
    assert worst < 1e-6, f"compatible_mass disagrees with oracle: {worst}"
    print(f"PASS test_compatible_mass_matches_oracle (worst |err|={worst:.2e})")


def test_uniform_compatible_count_is_990():
    """On a 5-card board, every hero hand has exactly C(45,2)=990 compatible
    villain hands (the uniform normaliser the BR estimator divides by)."""
    board = ['HA', 'DK', 'CQ', 'SJ', 'H9']
    ba = build_board_arrays(board, _EVAL, _CARDS)
    assert ba['H'] == 1081, ba['H']
    compat = compatible_mass(ba, np.ones(ba['H']))
    assert np.allclose(compat, _COMPATIBLE), (compat.min(), compat.max())
    print("PASS test_uniform_compatible_count_is_990")


def test_board_arrays_shape_consistency():
    board = ['HA', 'DK', 'CQ', 'SJ', 'H9']
    ba = build_board_arrays(board, _EVAL, _CARDS)
    H = ba['H']
    for k in ('raw', 'c1', 'c2', 'g', 'pf'):
        assert len(ba[k]) == H, (k, len(ba[k]), H)
    # No hand uses a board card; group ids are dense 0..G-1.
    bset = {c for c in board}
    assert all(h[0] not in bset and h[1] not in bset for h in ba['hands'])
    assert set(np.unique(ba['g']).tolist()) == set(range(ba['G']))
    # groups partition every hand exactly once, per street.
    for street in (0, 1, 2, 3):
        covered = np.zeros(H, dtype=int)
        for mask, _rep in ba['groups'][street]:
            covered += mask.astype(int)
        assert np.all(covered == 1), (street, covered.min(), covered.max())
    print("PASS test_board_arrays_shape_consistency")


def test_evaluator_delegates_to_kernel():
    """The refactored BestResponseEvaluator must produce arrays/showdown values
    identical to the kernel it now delegates to (guards against drift)."""
    from src.evaluation.best_response import BestResponseEvaluator
    ev = BestResponseEvaluator(blueprint_db=None, seed=0)
    board = ['HA', 'DK', 'CQ', 'SJ', 'H9']
    ba_ev = ev._board_arrays(board)
    ba_k = build_board_arrays(board, _EVAL, _CARDS)
    assert ba_ev['H'] == ba_k['H']
    assert np.array_equal(ba_ev['raw'], ba_k['raw'])
    assert np.array_equal(ba_ev['c1'], ba_k['c1']) and np.array_equal(ba_ev['c2'], ba_k['c2'])
    rv = np.random.default_rng(1).random(ba_k['H'])
    a = ev._showdown_measure(ba_ev, rv, 100.0, 30.0)
    b = showdown_measure(ba_k, rv, 100.0, 30.0)
    assert np.allclose(a, b)
    print("PASS test_evaluator_delegates_to_kernel")


TESTS = [
    test_showdown_matches_oracle,
    test_compatible_mass_matches_oracle,
    test_uniform_compatible_count_is_990,
    test_board_arrays_shape_consistency,
    test_evaluator_delegates_to_kernel,
]

if __name__ == '__main__':
    passed = failed = 0
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\nResults: {passed} passed, {failed} failed out of {len(TESTS)}")
    sys.exit(1 if failed else 0)
