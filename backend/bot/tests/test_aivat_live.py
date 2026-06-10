# backend/bot/tests/test_aivat_live.py
"""
AIVAT control-variate variance reduction ported to the LIVE GameSession path
(compare_gadget_policies._play_and_record -> aivat.AIVATEstimator). Validates the
record-building (esp. the all-in detection that c3 keys off) is correct and that the
estimator stays UNBIASED while cutting variance. Runs against the active blueprint;
skips if none.
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _blueprint_db():
    try:
        from src.config import resolve_blueprint_path
        from src.storage.blueprint_db import BlueprintDB
        return BlueprintDB(resolve_blueprint_path(), read_only=True)
    except Exception as e:
        print(f"  (no blueprint DB: {e})")
        return None


def test_aivat_live_unbiased_and_reduces_variance():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_aivat_live_unbiased_and_reduces_variance (no blueprint)")
        return
    from src.abstractions.sizing import db_menu_mode
    from src.game.game_session import GameSession
    from src.subgame.river_subgame_solver import RiverSubgameSolver
    from src.evaluation.aivat import AIVATEstimator
    from compare_gadget_policies import _play_and_record, _POLICIES

    bot = RiverSubgameSolver(db, max_iters=80, check_every=40, time_budget=2.0,
                             **_POLICIES['off'])
    random.seed(7)
    s = GameSession.new('t', 'maniac', strategy_fn=bot.range_model_fn(),
                        menu_mode=db_menu_mode(db), max_raises_per_street=float('inf'))
    records = []
    prev = 0.0
    hands = 250
    for h in range(hands):
        if h > 0:
            s.start_next_hand()
        bot_seat = 1 - s.data['human_seat']
        rec = _play_and_record(s, bot, 'jam', bot_seat)
        net = s.data['human_net']
        rec['result'] = -(net - prev)
        prev = net
        records.append(rec)

    # all-in detection must fire (the bug that broke c3 produced ZERO -> ~no reduction).
    allin = sum(1 for r in records if r['allin_street'] is not None)
    assert allin > 0, "no all-in showdowns detected -- c3 would be inert"
    # every all-in record's invested must be the FULL committed stacks (pot == both stacks).
    for r in records:
        if r['allin_street'] is not None:
            assert abs(r['invested'][0] - r['invested'][1]) < 1e-6
            assert abs(sum(r['invested']) - 400.0) < 1e-6, r['invested']

    est = AIVATEstimator(db, seed=7).estimate(records)
    db.close()
    # UNBIASED: AIVAT changes variance, not the mean -- within a few raw stderrs.
    drift = abs(est['aivat_mbb'] - est['raw_mbb'])
    assert drift <= 5.0 * est['raw_stderr_mbb'], (
        est['aivat_mbb'], est['raw_mbb'], est['raw_stderr_mbb'])
    # and it must actually REDUCE variance (c1 + c3 working).
    assert est['var_reduction'] > 0.10, est['var_reduction']
    print(f"PASS test_aivat_live_unbiased_and_reduces_variance "
          f"(allin={allin}/{hands}, var -{est['var_reduction'] * 100:.0f}%, "
          f"raw {est['raw_mbb']:+.0f}+/-{est['raw_stderr_mbb']:.0f} vs "
          f"aivat {est['aivat_mbb']:+.0f}+/-{est['aivat_stderr_mbb']:.0f} mbb)")


TESTS = [test_aivat_live_unbiased_and_reduces_variance]

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
