"""
Tests for the strategy-shape sanity probe (src/cfr/strategy_shape.py) -- the
checkpoint health check that catches a BUG-014-style preflop collapse (open one size
with 100% of hands, never fold). Guards that the probe FLAGS a collapsed strategy and
PASSES a healthy one.

Run from backend/bot/:
    python tests/test_strategy_shape.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.strategy_shape import strategy_shape_report, format_shape_line, OK, COLLAPSE


def _collapsed_strat(key):
    """The BUG-014 fingerprint: every open is one big size with ~0 fold; every
    BB-vs-5BB is one re-raise with ~0 fold -- for ALL buckets (no gradient)."""
    if key.endswith('_ip_'):                       # pf_N_ip_ open node
        return {'bet_xlarge': 0.97, 'bet_large': 0.03, 'fold': 0.0}
    if key.endswith('_oop_x'):                     # pf_N_oop_x BB-vs-5BB node
        return {'raise_large': 0.99, 'fold': 0.01}
    return None


def _healthy_strat(key):
    """A sane strength gradient: weak buckets fold a lot, strong buckets open/call;
    opens spread across sizes; BB-vs-5BB folds weak, 3-bets strong."""
    parts = key.split('_')
    if key.endswith('_ip_'):                       # pf_N_ip_
        n = int(parts[1])
        if n <= 5:
            return {'fold': 0.65, 'bet_medium': 0.12, 'bet_small': 0.1, 'call': 0.08,
                    'bet_large': 0.05}
        if n >= 24:
            return {'bet_large': 0.3, 'bet_medium': 0.25, 'call': 0.2, 'bet_xlarge': 0.15,
                    'fold': 0.02, 'bet_small': 0.08}
        return {'fold': 0.3, 'call': 0.25, 'bet_medium': 0.2, 'bet_small': 0.15,
                'bet_large': 0.1}
    if key.endswith('_oop_x'):                     # pf_N_oop_x
        n = int(parts[1])
        if n <= 5:
            return {'fold': 0.8, 'call': 0.1, 'raise_medium': 0.06, 'raise_large': 0.04}
        return {'raise_medium': 0.35, 'raise_large': 0.25, 'call': 0.2, 'fold': 0.1,
                'raise_small': 0.1}
    return None


def test_probe_flags_collapse():
    rep = strategy_shape_report(_collapsed_strat, num_preflop_buckets=30)
    assert rep['verdict'] == COLLAPSE, rep
    assert rep['n_open'] == 30 and rep['n_bbx'] == 30
    assert rep['weak_open_fold'] < 0.05, rep['weak_open_fold']
    # both collapse nodes should be named in the reasons
    joined = ' '.join(rep['reasons'])
    assert 'open' in joined and 'BB-vs-5BB' in joined, rep['reasons']
    assert format_shape_line(rep).isascii(), "checkpoint line must be ASCII (Windows cp1252)"
    print("  PASS: probe flags a collapsed strategy (both nodes)")


def test_probe_passes_healthy():
    rep = strategy_shape_report(_healthy_strat, num_preflop_buckets=30)
    assert rep['verdict'] == OK, rep
    assert rep['weak_open_fold'] > 0.4, rep['weak_open_fold']
    assert rep['gradient_pf0_minus_strong'] > 0.4, rep['gradient_pf0_minus_strong']
    assert format_shape_line(rep).isascii()
    print("  PASS: probe passes a healthy strength-graded strategy")


def test_probe_handles_missing_keys():
    """An empty/early blueprint (no keys) must not crash and must not false-alarm."""
    rep = strategy_shape_report(lambda k: None, num_preflop_buckets=30)
    assert rep['n_open'] == 0 and rep['n_bbx'] == 0
    assert rep['verdict'] == OK, rep        # no data -> no collapse claim
    print("  PASS: probe handles a missing/empty blueprint without false alarm")


if __name__ == '__main__':
    tests = [test_probe_flags_collapse, test_probe_passes_healthy,
             test_probe_handles_missing_keys]
    print("Running strategy-shape probe tests...\n")
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll {len(tests)} tests passed.")
