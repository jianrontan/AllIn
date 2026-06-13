# backend/bot/tests/test_range_tracker.py
"""Unit tests for the Phase-3 hand-level Bayesian opponent range tracker
(src/game/range_tracker.py). Run from backend/bot/:  python tests/test_range_tracker.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.game.range_tracker import RangeTracker
from src.abstractions.card_abstractions import CardAbstraction, NUM_PREFLOP_BUCKETS

CARDS = CardAbstraction()

# Strongest preflop bucket index (pf_0..pf_{MAX}). Derived from the abstraction so
# the stub models below normalise to the real bucket range. (History: hardcoding the
# old 15-bucket max (14) made `frac` exceed 1 once preflop went to 40 buckets, driving
# the stub fold-probability negative and silently no-op'ing the Bayesian update. Now
# decoupled to 30 fine buckets -- this tracks NUM_PREFLOP_BUCKETS so it can't rot again.)
_MAX_PF = NUM_PREFLOP_BUCKETS - 1


def _pf_index(key):
    """Parse the preflop bucket integer from an info-set key 'pf_<n>_...'."""
    return int(key.split('_')[1])


def uniform_fn(key, legal):
    n = len(legal)
    return np.ones(n) / n


def peaked_fn(key, legal):
    """call-heavy: fold 0.02 / call 0.96 / raise 0.02 (key-independent)."""
    m = {'fold': 0.02, 'call': 0.96, 'raise': 0.02}
    return np.array([m[a] for a in legal])


def weakfolds_fn(key, legal):
    """Weak preflop buckets fold a lot, strong ones rarely (key-DEPENDENT)."""
    frac = _pf_index(key) / _MAX_PF
    fold = 0.85 * (1.0 - frac)
    rest = 1.0 - fold
    m = {'fold': fold, 'call': rest * 0.5, 'raise': rest * 0.5}
    return np.array([m[a] for a in legal])


def test_initial_hand_count():
    t = RangeTracker(('HA', 'DK'), CARDS)
    # opponent holds 2 of the 50 cards not in hero's hand: C(50,2) = 1225
    assert len(t.hands) == 1225, len(t.hands)
    assert t.confidence == 1.0
    assert np.allclose(t.w, 1.0)
    print(f"PASS test_initial_hand_count: {len(t.hands)} combos")


def test_reveal_card_removal():
    t = RangeTracker(('HA', 'DK'), CARDS)
    board = ['SQ', 'SJ', 'ST']
    t.reveal(board)
    bset = set(board)
    for h, w in zip(t.hands, t.w):
        if h[0] in bset or h[1] in bset:
            assert w == 0.0, f"{h} collides with board but weight {w}"
    # hands using neither hero nor board cards keep weight 1
    assert any(w == 1.0 for w in t.w)
    print("PASS test_reveal_card_removal")


def test_uniform_model_no_confidence_loss():
    """A uniform model can never be surprised -> confidence stays 1.0."""
    t = RangeTracker(('HA', 'DK'), CARDS)
    legal = ['fold', 'call', 'raise']
    for a in ['call', 'fold', 'raise']:
        t.observe(uniform_fn, a, 0, 'oop', '', legal, [])
    assert abs(t.confidence - 1.0) < 1e-9, t.confidence
    print(f"PASS test_uniform_model_no_confidence_loss: conf={t.confidence:.4f}")


def test_confidence_drops_on_offmodel_action():
    """call-heavy model: observing a rare 'fold' tanks confidence; 'call' doesn't."""
    legal = ['fold', 'call', 'raise']
    on = RangeTracker(('HA', 'DK'), CARDS)
    on.observe(peaked_fn, 'call', 0, 'oop', '', legal, [])
    assert on.confidence > 0.95, f"on-model call should barely move conf: {on.confidence}"

    off = RangeTracker(('HA', 'DK'), CARDS)
    off.observe(peaked_fn, 'fold', 0, 'oop', '', legal, [])
    assert off.confidence < 0.2, f"off-model fold should tank conf: {off.confidence}"
    print(f"PASS test_confidence_drops_on_offmodel_action: "
          f"call->{on.confidence:.3f}  fold->{off.confidence:.3f}")


def test_bayesian_update_shifts_toward_consistent_hands():
    """With a model where weak hands fold more, observing 'fold' must shift the
    belief's mass toward weaker preflop buckets."""
    t = RangeTracker(('HA', 'DK'), CARDS)
    legal = ['fold', 'call', 'raise']

    pf = np.array([_pf_index(_key0(h)) for h in t.hands], dtype=float)
    prior = t.normalized_weights()
    mean_pf_prior = float((prior * pf).sum())

    t.observe(weakfolds_fn, 'fold', 0, 'oop', '', legal, [])
    post = t.normalized_weights()
    mean_pf_post = float((post * pf).sum())

    assert mean_pf_post < mean_pf_prior - 0.5, (mean_pf_prior, mean_pf_post)
    print(f"PASS test_bayesian_update_shifts_toward_consistent_hands: "
          f"mean pf {mean_pf_prior:.2f} -> {mean_pf_post:.2f} after fold")


def _key0(h):
    from src.cfr.keys import make_info_set_key
    return make_info_set_key(0, 'oop', CARDS.get_bucket(list(h), None), None, '')


def test_offmodel_action_preserves_range():
    """C1: an action with ~zero model-prob across ALL live hands must NOT zero
    the belief — the range is preserved (not annihilated) and the bot stays
    informed; only confidence drops."""
    t = RangeTracker(('HA', 'DK'), CARDS)
    legal = ['fold', 'call', 'raise']

    def never_folds(key, legal):
        m = {'fold': 0.0, 'call': 0.5, 'raise': 0.5}
        return np.array([m[a] for a in legal])

    before = t.w.copy()
    conf_before = t.confidence
    t.observe(never_folds, 'fold', 0, 'oop', '', legal, [])   # impossible under model
    assert t.w.sum() > 0.0, "range was zeroed (C1 regression!)"
    assert np.allclose(t.w, before), "zero-prob update should leave the range unchanged"
    assert t.confidence < conf_before, "confidence should drop on the off-model action"
    eq = t.hero_equity(['HA', 'DK'], ['HQ', 'HJ', 'HT', 'C2', 'S4'])
    assert 0.0 <= eq <= 1.0
    print(f"PASS test_offmodel_action_preserves_range: conf {conf_before:.2f}->{t.confidence:.3f}")


def test_offmenu_action_collapses_confidence_keeps_range():
    """observe(action) where `action` is NOT in `legal` -- an emergent/custom all-in
    outside the node's abstract menu (e.g. a 100BB jam over a min-open normalized to
    'allin' when 'allin' isn't a listed legal action) -- must (a) not crash / 500 the
    live hand [regression 2026-06-04], (b) leave the RANGE unchanged (can't Bayesian-
    reweight on an action the model can't represent), and (c) COLLAPSE confidence
    [BUG-022 2026-06-09]: such an action is maximally off-model, and leaving confidence
    untouched let the bot trust a uniform 'jams any-two' belief and call off 100BB with
    trash. Confidence must drop below the guards' trust threshold (0.2)."""
    t = RangeTracker(('HA', 'DK'), CARDS)
    legal = ['fold', 'call', 'raise_medium', 'raise_large']   # note: no 'allin'
    before = t.w.copy()
    conf_before = t.confidence
    t.observe(weakfolds_fn, 'allin', 0, 'oop', '', legal, [])   # 'allin' not in legal
    assert np.allclose(t.w, before), "off-menu action must leave the range unchanged"
    assert t.confidence < conf_before, "off-menu action must collapse confidence"
    assert t.confidence < 0.2, "off-menu confidence must drop below the guard threshold"
    print("PASS test_offmenu_action_collapses_confidence_keeps_range")


def test_serialization_roundtrip():
    t = RangeTracker(('HA', 'DK'), CARDS)
    legal = ['fold', 'call', 'raise']
    t.observe(weakfolds_fn, 'call', 0, 'oop', '', legal, [])
    t.reveal(['SQ', 'SJ', 'ST'])
    d = t.to_dict()
    import json
    d2 = json.loads(json.dumps(d))            # ensure it's JSON-clean
    r = RangeTracker.from_dict(d2, CARDS)
    assert r.hero_hole == t.hero_hole
    assert np.allclose(r.w, t.w)
    assert abs(r.confidence - t.confidence) < 1e-12
    assert r.hands == t.hands
    print("PASS test_serialization_roundtrip")


# =====================================================================
# GameSession integration
# =====================================================================
from src.game.game_session import GameSession


def _stub_strategy_fn(key, legal):
    """Non-uniform, never-zero model: weak preflop buckets fold/check more,
    strong ones bet/raise more. Works for any legal set and street."""
    pf = int(key.split('_')[1])
    frac = pf / _MAX_PF
    vals = []
    for a in legal:
        if a == 'fold':
            vals.append(0.55 * (1.0 - frac) + 0.05)
        elif a in ('check', 'call'):
            vals.append(0.40)
        else:                       # bets / raises / allin -> stronger hands
            vals.append(0.15 + 0.45 * frac)
    arr = np.array(vals)
    return arr / arr.sum()


def test_gamesession_tracking_disabled_without_model():
    s = GameSession.new('s1', 'p1')          # no strategy_fn
    assert s.data['opp_range'] is None
    assert s.opponent_read() is None
    print("PASS test_gamesession_tracking_disabled_without_model")


def test_gamesession_tracks_and_serializes():
    import json
    s = GameSession.new('s2', 'p2', strategy_fn=_stub_strategy_fn)
    assert s.data['opp_range'] is not None
    read0 = s.opponent_read()
    assert read0 is not None
    assert read0['confidence'] == 1.0 and len(read0['topHands']) > 0
    assert s.is_human_turn(), "human is the button and acts first preflop"

    before = list(s.data['opp_range']['w'])
    legal = s.legal_actions()
    action = 'call' if 'call' in legal else ('check' if 'check' in legal else legal[0])
    s.apply_action(action)
    after = list(s.data['opp_range']['w'])
    assert not np.allclose(before, after), "human action should update the belief"

    # round-trip through JSON with the model re-injected (API does this each request)
    d = json.loads(json.dumps(s.to_dict()))
    s2 = GameSession.from_dict(d, strategy_fn=_stub_strategy_fn)
    assert np.allclose(s2.data['opp_range']['w'], s.data['opp_range']['w'])
    print("PASS test_gamesession_tracks_and_serializes")


def test_read_group_label():
    """The bot's-read grouping: suit-equivalent combos collapse, suits show only
    when flush-relevant, suited/offsuit stays distinct, and any flush-suit letter
    held is appended at the END of the token (ranks first, suit never wedged
    between them). An offsuit combo that holds one flush card is tagged once
    (e.g. 'AKoh') without distinguishing WHICH card is the flush card -- the
    trailing format trades that blocker detail for a consistently readable label."""
    from src.game.game_session import _read_group_label
    # Rainbow board (no flush-relevant suit): suits vanish, suited/offsuit kept.
    assert _read_group_label(('HA', 'CA'), set()) == 'AA'
    assert _read_group_label(('HA', 'SA'), set()) == 'AA'
    assert _read_group_label(('HA', 'HK'), set()) == 'AKs'
    assert _read_group_label(('HA', 'CK'), set()) == 'AKo'
    assert _read_group_label(('H2', 'D2'), set()) == '22'
    # Hearts flush-relevant: a held heart shows as a trailing 'h'; no heart collapses.
    assert _read_group_label(('HA', 'CA'), {'H'}) == 'AAh'     # pair holding Ah (blocker)
    assert _read_group_label(('CA', 'SA'), {'H'}) == 'AA'
    assert _read_group_label(('HA', 'HK'), {'H'}) == 'AKh'     # suited in hearts (flush draw)
    assert _read_group_label(('HA', 'CK'), {'H'}) == 'AKoh'    # offsuit, holds one heart (Ah)
    assert _read_group_label(('CA', 'HK'), {'H'}) == 'AKoh'    # offsuit, holds one heart (Kh)
    # Two flush-relevant suits at once: both held suit letters trail, sorted.
    assert _read_group_label(('HK', 'SQ'), {'H', 'S'}) == 'KQohs'   # offsuit, holds H and S
    assert _read_group_label(('HA', 'CA'), {'H', 'C'}) == 'AAch'    # pair, holds C and H
    print("PASS test_read_group_label")


def test_opponent_read_groups_combos():
    """opponent_read returns grouped labels (not raw combos): e.g. preflop the six
    pocket-ace combos collapse into a single 'AA' entry."""
    s = GameSession.new('grp', 'p', strategy_fn=_stub_strategy_fn)
    read = s.opponent_read(k=200)
    assert read is not None
    labels = [h['label'] for h in read['topHands']]
    assert len(labels) == len(set(labels)), "labels must be unique groups"
    assert 'AA' in labels and 'AKs' in labels and 'AKo' in labels
    # Grouping must not lose mass: total over groups ~ 1.
    assert abs(sum(h['prob'] for h in read['topHands']) - 1.0) < 0.02
    print("PASS test_opponent_read_groups_combos")


def test_public_view_json_serializable():
    """public_view() is what the API actually returns (incl. botRead derived
    from the live tracker); it must be JSON-clean -- no numpy scalars leaking
    from confidence/top-hand probs. to_dict()'s round-trip is tested above; this
    guards the DIFFERENT, untested API-response path."""
    import json
    s = GameSession.new('pv', 'pv', strategy_fn=_stub_strategy_fn)
    # Play a few human/bot actions so the tracker has observed + (likely) a
    # street has been revealed -> botRead.topHands is populated.
    guard = 0
    while s.data['status'] == 'in_hand' and guard < 8:
        if s.is_human_turn():
            legal = s.legal_actions()
            a = 'call' if 'call' in legal else ('check' if 'check' in legal else legal[0])
            s.apply_action(a)
        else:
            break
        guard += 1

    view = s.public_view()
    # Must serialize without a "not JSON serializable" TypeError.
    json.dumps(view)
    if view['botRead'] is not None:
        assert isinstance(view['botRead']['confidence'], float)
        for th in view['botRead']['topHands']:
            assert isinstance(th['prob'], float)
    print("PASS test_public_view_json_serializable")


def test_gamesession_reveal_removes_board_combos():
    s = GameSession.new('s3', 'p3', strategy_fn=_stub_strategy_fn)
    guard = 0
    while s.data['status'] == 'in_hand' and s.data['street'] == 0 and guard < 12:
        legal = s.legal_actions()
        a = 'call' if 'call' in legal else ('check' if 'check' in legal else legal[0])
        s.apply_action(a)
        guard += 1
    if s.data['street'] >= 1 and s.data['opp_range'] is not None:
        t = RangeTracker.from_dict(s.data['opp_range'], CARDS)
        board = set(s.data['community'][:3])
        assert all(w == 0.0 for h, w in zip(t.hands, t.w)
                   if h[0] in board or h[1] in board)
        print("PASS test_gamesession_reveal_removes_board_combos")
    else:
        print("PASS test_gamesession_reveal_removes_board_combos (hand ended preflop)")


# =====================================================================
# hero_equity + ConfidenceAwareStrategy consumer
# =====================================================================
from src.game.bot_strategy import ConfidenceAwareStrategy


def test_hero_equity_nuts_is_one():
    """Hero holding the nuts on a finished board beats a forced worse hand 100%."""
    t = RangeTracker(('HA', 'HK'), CARDS)
    t.w[:] = 0.0
    t.w[t.hands.index(('S2', 'S3'))] = 1.0           # opponent forced to 2-3
    board = ['HQ', 'HJ', 'HT', 'C2', 'D3']           # hero AK hearts = royal flush
    eq = t.hero_equity(['HA', 'HK'], board)
    assert eq == 1.0, eq
    print("PASS test_hero_equity_nuts_is_one")


def _hand_index(tracker, a, b):
    """Index of the {a,b} combo regardless of stored card order."""
    target = {a, b}
    return next(i for i, h in enumerate(tracker.hands) if set(h) == target)


def test_hero_equity_dominated_is_zero():
    """Hero drawing dead has equity 0 vs a forced better hand."""
    t = RangeTracker(('C7', 'D2'), CARDS)
    t.w[:] = 0.0
    t.w[_hand_index(t, 'HA', 'HK')] = 1.0            # opponent has the royal
    board = ['HQ', 'HJ', 'HT', 'C2', 'S4']           # opp = royal flush; hero = junk
    eq = t.hero_equity(['C7', 'D2'], board)
    assert eq == 0.0, eq
    print("PASS test_hero_equity_dominated_is_zero")


def test_equity_action_mapping():
    f = ConfidenceAwareStrategy._equity_action
    assert f(0.70, 0, 10, ['check', 'bet_medium']) == 'bet_medium'   # ahead, can bet
    assert f(0.40, 0, 10, ['check', 'bet_medium']) == 'check'        # mediocre, check
    assert f(0.20, 10, 10, ['fold', 'call', 'raise_medium']) == 'fold'   # eq<pot odds
    assert f(0.80, 10, 90, ['fold', 'call', 'raise_medium']) == 'raise_medium'  # crush
    assert f(0.60, 10, 10, ['fold', 'call', 'raise_medium']) == 'call'  # priced in
    print("PASS test_equity_action_mapping")


def test_consumer_routes_by_confidence():
    """High confidence -> blueprint path (equity NOT consulted); low -> equity path."""
    strat = ConfidenceAwareStrategy(None)            # no DB -> blueprint = uniform
    legal = ['check', 'bet_medium']
    t = RangeTracker(('HA', 'HK'), CARDS)

    def boom(*a, **k):
        raise RuntimeError("equity consulted")
    t.hero_equity = boom
    ps = {'opp_range': t, 'hole_cards': ['HA', 'HK'], 'community': [], 'to_call': 0, 'pot': 3}

    t.confidence = 1.0                               # confident -> must not touch equity
    strat.decide('pf_9_ip_', legal, ps)

    t.confidence = 0.01                              # collapsed -> must use equity
    try:
        strat.decide('pf_9_ip_', legal, ps)
        raise AssertionError("expected equity path")
    except RuntimeError as e:
        assert 'equity consulted' in str(e)
    print("PASS test_consumer_routes_by_confidence")


# =====================================================================
# Process-global bucket cache (perf optimization, must be correctness-neutral)
# =====================================================================
from src.game import range_tracker as _rt
from src.game.range_tracker import (
    _clear_bucket_caches, _PF_BUCKET_CACHE, _POSTFLOP_BUCKET_CACHE)


def test_bucket_cache_matches_direct_get_bucket():
    """The cached _buckets() output must equal a direct per-hand get_bucket()
    computation, on every street. This is THE correctness guard for the cache."""
    _clear_bucket_caches()
    board = ['SQ', 'SJ', 'ST', 'C2', 'D4']
    t = RangeTracker(('HA', 'DK'), CARDS)
    for street in (0, 1, 2, 3):
        slice_board = None if street == 0 else board[:2 + street]
        if slice_board is not None:
            t.reveal(slice_board)                    # match real usage (reveal before bucket)
        pf, strength = t._buckets(street, board)
        bset = set(slice_board) if slice_board else set()
        for i, h in enumerate(t.hands):
            if t.w[i] <= 0.0:
                continue                             # dead hands are left None by design
            assert pf[i] == CARDS.get_bucket(list(h), None), (street, h, 'preflop')
            if street > 0:
                assert h[0] not in bset and h[1] not in bset  # live => non-colliding
                assert strength[i] == CARDS.get_bucket(list(h), slice_board), (street, h)
    print("PASS test_bucket_cache_matches_direct_get_bucket")


def test_bucket_cache_shared_across_trackers():
    """Two trackers with DIFFERENT heroes on the SAME board reuse the one cached
    per-board map (no second board entry), and both get correct buckets — this is
    the cross-tracker / cross-session sharing that makes the cache worthwhile."""
    _clear_bucket_caches()
    board = ['SQ', 'SJ', 'ST']
    t1 = RangeTracker(('HA', 'DK'), CARDS)
    t1.reveal(board)
    t1._buckets(1, board)
    assert len(_PF_BUCKET_CACHE) == 1326, "preflop map fills all combos once"
    assert tuple(board) in _POSTFLOP_BUCKET_CACHE
    n_boards = len(_POSTFLOP_BUCKET_CACHE)

    t2 = RangeTracker(('C7', 'C8'), CARDS)           # different hero, same board
    t2.reveal(board)
    pf2, str2 = t2._buckets(1, board)
    assert len(_POSTFLOP_BUCKET_CACHE) == n_boards, "same board must not add a 2nd entry"
    for i, h in enumerate(t2.hands):
        if t2.w[i] <= 0.0:
            continue
        assert str2[i] == CARDS.get_bucket(list(h), board)
    print("PASS test_bucket_cache_shared_across_trackers")


def test_bucket_cache_warm_equals_cold_for_observe():
    """observe() must produce identical weights + confidence whether the global
    cache was cold or pre-warmed — proves the optimization is behavior-neutral."""
    legal = ['fold', 'call', 'raise']
    _clear_bucket_caches()
    cold = RangeTracker(('HA', 'DK'), CARDS)
    cold.observe(weakfolds_fn, 'fold', 0, 'oop', '', legal, [])

    # Pre-warm by bucketing with an unrelated tracker, then observe afresh.
    RangeTracker(('C2', 'C3'), CARDS)._buckets(0, [])
    warm = RangeTracker(('HA', 'DK'), CARDS)
    warm.observe(weakfolds_fn, 'fold', 0, 'oop', '', legal, [])

    assert np.allclose(cold.w, warm.w)
    assert abs(cold.confidence - warm.confidence) < 1e-12
    print("PASS test_bucket_cache_warm_equals_cold_for_observe")


def test_bucket_cache_lru_bound():
    """The per-board cache is LRU-bounded: past the cap, the oldest board is
    evicted and the most-recent retained (so it can't grow without limit)."""
    _clear_bucket_caches()
    old_cap = _rt._POSTFLOP_BUCKET_MAX_BOARDS
    _rt._POSTFLOP_BUCKET_MAX_BOARDS = 3
    try:
        boards = [['SQ', 'SJ', 'ST'], ['HQ', 'HJ', 'HT'], ['DQ', 'DJ', 'DT'],
                  ['CQ', 'CJ', 'CT'], ['S2', 'S3', 'S4']]
        for b in boards:
            t = RangeTracker(('HA', 'DK'), CARDS)
            t.reveal(b)
            t._buckets(1, b)
        assert len(_POSTFLOP_BUCKET_CACHE) == 3, len(_POSTFLOP_BUCKET_CACHE)
        assert tuple(boards[-1]) in _POSTFLOP_BUCKET_CACHE      # newest kept
        assert tuple(boards[0]) not in _POSTFLOP_BUCKET_CACHE   # oldest evicted
    finally:
        _rt._POSTFLOP_BUCKET_MAX_BOARDS = old_cap
        _clear_bucket_caches()
    print("PASS test_bucket_cache_lru_bound")


TESTS = [
    test_initial_hand_count,
    test_reveal_card_removal,
    test_uniform_model_no_confidence_loss,
    test_confidence_drops_on_offmodel_action,
    test_bayesian_update_shifts_toward_consistent_hands,
    test_offmodel_action_preserves_range,
    test_offmenu_action_collapses_confidence_keeps_range,
    test_serialization_roundtrip,
    test_gamesession_tracking_disabled_without_model,
    test_gamesession_tracks_and_serializes,
    test_gamesession_reveal_removes_board_combos,
    test_read_group_label,
    test_opponent_read_groups_combos,
    test_public_view_json_serializable,
    test_hero_equity_nuts_is_one,
    test_hero_equity_dominated_is_zero,
    test_equity_action_mapping,
    test_consumer_routes_by_confidence,
    test_bucket_cache_matches_direct_get_bucket,
    test_bucket_cache_shared_across_trackers,
    test_bucket_cache_warm_equals_cold_for_observe,
    test_bucket_cache_lru_bound,
]

if __name__ == '__main__':
    passed = failed = 0
    fails = []
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            fails.append((fn.__name__, repr(e)))
            print(f"FAIL {fn.__name__}: {e!r}")
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    if fails:
        print("Failed:")
        for name, err in fails:
            print(f"  {name}: {err}")
    sys.exit(1 if failed else 0)
