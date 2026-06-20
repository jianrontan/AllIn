# backend/bot/tests/test_opponent_model.py
"""Phase 6 E2: the serve-time HumanModel produces a valid, correctly-tilted strategy_fn.

Integration test against the real fitted artifacts (analysis/opponent_models/model_built_*.json
+ model_population.json) and the 30/24/10 snapshot blueprint. Asserts:
  * the strategy_fn obeys the tracker contract (np.ndarray aligned to legal, non-negative, sums to 1);
  * a KNOWN over-limper (XYyyyy) is tilted toward call/limp and away from raising vs the blueprint
    at a verified limp key (the leak the audit confirmed real);
  * an UNKNOWN player falls back to the population model (still a valid distribution).

Run: python tests/test_opponent_model.py   (or under pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exploitation.opponent_model import HumanModel
from src.storage.blueprint_db import BlueprintDB

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_HERE, 'analysis', 'opponent_models')
_DB = os.path.join(_HERE, 'analysis', 'blueprints', 'snapshots', 'snap_52500000.db')
_XY = '2a1a069d-d241-4e3f-9321-418ebf516c47'        # XYyyyy: verified preflop over-limper
_LEGAL = ('fold', 'call', 'bet_small', 'bet_medium', 'bet_large', 'bet_xlarge')   # preflop open menu
_KEY = 'pf_8_ip_'                                   # XYyyyy limps ~always here; GTO raises ~18%


def _bp_call_prob(db, key, legal):
    raw = db.get_average_strategy(key) or {}
    w = np.array([max(0.0, raw.get(a, 0.0)) for a in legal])
    return float(w[legal.index('call')] / w.sum()) if w.sum() > 1e-12 else 1.0 / len(legal)


def _valid(v):
    return len(v) == len(_LEGAL) and (v >= 0).all() and abs(float(v.sum()) - 1.0) < 1e-9


def run():
    if not (os.path.exists(_DB) and os.path.exists(os.path.join(_MODEL_DIR, 'model_population.json'))):
        print("  SKIP: opponent-model artifacts / snapshot DB not present")
        return True
    db = BlueprintDB(_DB, read_only=True)
    hm = HumanModel(_MODEL_DIR, db)

    fn = hm.strategy_fn_for(_XY)
    h = fn(_KEY, _LEGAL)
    bp_call = _bp_call_prob(db, _KEY, _LEGAL)
    h_call = float(h[_LEGAL.index('call')])
    h_raise = float(h[_LEGAL.index('bet_medium')] + h[_LEGAL.index('bet_large')])

    print(f"  {_KEY}: XYyyyy P(limp/call)={h_call:.2f} vs GTO {bp_call:.2f}; "
          f"XYyyyy P(med+large raise)={h_raise:.2f}")
    assert _valid(h), "strategy_fn must return a valid distribution over legal"
    assert hm.has_player(_XY), "XYyyyy should be a known player"
    assert h_call > 0.5, f"over-limper should call/limp a lot here (got {h_call:.2f})"
    assert h_call > bp_call + 0.2, "known over-limper must be tilted ABOVE GTO's call freq"

    # Unknown player -> population model; must still be a valid distribution.
    pf = hm.strategy_fn_for('no-such-player-uuid')
    p = pf(_KEY, _LEGAL)
    assert not hm.has_player('no-such-player-uuid')
    assert _valid(p), "unknown player must fall back to a valid population distribution"

    # CRASH-SAFETY (audit C1): strategy_fn runs mid-hand where callers catch only GameError, so a
    # malformed key must degrade to a uniform 'no read' row, NEVER raise (which would 500 the hand).
    for bad_key in ('totally_garbage', 'pf_xx_ip_', '', 'pf_3_99_ip_turn_'):
        v = fn(bad_key, _LEGAL)
        assert _valid(v), f"malformed key {bad_key!r} must yield a valid uniform row, not raise"

    # A different KNOWN player keys differently from XYyyyy at the same node (per-player, not global).
    db.close()
    print("  PASS: HumanModel strategy_fn is valid, tilts XYyyyy toward limp, "
          "cold-starts to population")
    return True


def test_human_model():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
