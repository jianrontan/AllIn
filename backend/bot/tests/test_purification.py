# backend/bot/tests/test_purification.py
"""
Strategy purification transform (src/cfr/purification.py) + its wiring into the
blueprint strategy and the BR evaluator.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.purification import purify_probs, purify_dist


def test_off_is_identity():
    p = np.array([0.6, 0.3, 0.1])
    assert np.array_equal(purify_probs(p, 0.0), p)
    d = {'a': 0.6, 'b': 0.3, 'c': 0.1}
    assert purify_dist(d, 0.0) is d
    print("PASS test_off_is_identity")


def test_threshold_drops_and_renormalises():
    p = np.array([0.62, 0.31, 0.05, 0.015, 0.005])
    out = purify_probs(p, 0.01)                 # drop the 0.005 (only)
    assert out[4] == 0.0 and out[3] > 0.0, out
    assert abs(out.sum() - 1.0) < 1e-12
    # surviving mass keeps its relative proportions
    assert abs(out[0] / out[1] - 0.62 / 0.31) < 1e-9
    print("PASS test_threshold_drops_and_renormalises")


def test_full_purification_keeps_argmax():
    p = np.array([0.55, 0.30, 0.15])
    out = purify_probs(p, 1.0)                  # nothing clears 1.0 -> argmax only
    assert np.array_equal(out, np.array([1.0, 0.0, 0.0])), out
    print("PASS test_full_purification_keeps_argmax")


def test_full_purification_keeps_ties():
    p = np.array([0.5, 0.5, 0.0])
    out = purify_probs(p, 1.0)                  # genuine tie stays mixed
    assert abs(out[0] - 0.5) < 1e-12 and abs(out[1] - 0.5) < 1e-12
    print("PASS test_full_purification_keeps_ties")


def test_dist_wrapper_matches_vector():
    d = {'fold': 0.62, 'call': 0.31, 'raise': 0.05, 'allin': 0.02}
    out = purify_dist(d, 0.05)                  # drop raise(0.05? >=) ... keep >=0.05
    # 0.05 and 0.02: 0.05 >= 0.05 keeps, 0.02 drops
    assert out['allin'] == 0.0 and out['raise'] > 0.0
    assert abs(sum(out.values()) - 1.0) < 1e-12
    print("PASS test_dist_wrapper_matches_vector")


def test_blueprint_strategy_purifies_lookup():
    """BlueprintStrategy(purify_threshold=) purifies the trained-key distribution it
    samples from; the opponent model (range_model_fn) is left UNpurified."""
    from src.game.bot_strategy import BlueprintStrategy

    class _DB:
        def get_average_strategy(self, key):
            return {'fold': 0.62, 'call': 0.31, 'raise_small': 0.05, 'allin': 0.02}

    legal = ['fold', 'call', 'raise_small', 'allin']
    plain = BlueprintStrategy(_DB())._distribution('k', legal)
    assert plain['allin'] > 0.0
    pur = BlueprintStrategy(_DB(), purify_threshold=0.05)._distribution('k', legal)
    assert pur['allin'] == 0.0 and abs(sum(pur.values()) - 1.0) < 1e-12
    # opponent model unaffected by purification
    fn = BlueprintStrategy(_DB(), purify_threshold=1.0).range_model_fn()
    w = fn('k', legal)
    assert (w > 0).sum() == 4, "range model must NOT be purified"
    print("PASS test_blueprint_strategy_purifies_lookup")


def test_br_evaluator_applies_purification():
    """The BR scoreboard wiring (best_response.py:_restricted_probs) actually purifies the
    villain blueprint probs, not just the util -- review gap M2. Stub DB + menu_mode avoids
    a DB read; threshold 0.1 must drop the sub-10% action from the restricted vector."""
    from src.evaluation.best_response import BestResponseEvaluator

    class _StubDB:
        db_path = ':memory:'
        def get_average_strategy(self, key):
            return {'check': 0.62, 'bet_small': 0.33, 'bet_medium': 0.05}

    legal = ('check', 'bet_small', 'bet_medium')
    off = BestResponseEvaluator(_StubDB(), menu_mode='control', purify_threshold=0.0)
    on = BestResponseEvaluator(_StubDB(), menu_mode='control', purify_threshold=0.1)
    p_off = off._restricted_probs('k', legal)
    p_on = on._restricted_probs('k', legal)
    assert p_off[2] > 0.0, "baseline keeps the 5% action"
    assert p_on[2] == 0.0, "purify=0.1 must drop the 5% action"
    assert abs(p_on.sum() - 1.0) < 1e-12
    assert abs(p_on[0] / p_on[1] - 0.62 / 0.33) < 1e-9, "survivors keep proportions"
    print("PASS test_br_evaluator_applies_purification")


TESTS = [
    test_off_is_identity,
    test_br_evaluator_applies_purification,
    test_threshold_drops_and_renormalises,
    test_full_purification_keeps_argmax,
    test_full_purification_keeps_ties,
    test_dist_wrapper_matches_vector,
    test_blueprint_strategy_purifies_lookup,
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
