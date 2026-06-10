# backend/bot/tests/test_deep_raise_guard.py
"""
Unit tests for the facing-a-NON-all-in-deep-raise guard
(RiverSubgameSolver._facing_deep_raise_guard).

The leak it fixes: training caps aggression at 3/street, but LIVE play uncaps
re-raises, so a human 5-bet reaches a node (e.g. pf_29_ip_slll) the blueprint
never trained. BlueprintStrategy's passive fallback then plays UNIFORM call/fold
-- folding the nuts ~half the time. The guard instead decides call/fold by the
tracked-range equity vs pot odds, but ONLY when (a) the key is untrained and
(b) money is still behind (to_call < bot_stack -- the all-in case belongs to
_facing_allin_guard). It never raises (no stray raise from an untrained node).

A stub tracker (fixed equity + confidence) makes every branch deterministic; db
defaults to None (-> the key is "untrained", the case that matters most), with a
stub DB only for the trained-key-defers test.

Run: python tests/test_deep_raise_guard.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_subgame_solver import RiverSubgameSolver

# A deep 5-bet+ node: facing a raise, with sized re-raises also legal (uncapped
# live), money still behind (to_call 66 < stack 74). Pot odds = 66/(114+66) = 0.367.
_LEGAL = ['fold', 'call', 'raise_small', 'raise_medium', 'raise_large']
_passed = _failed = 0


class _StubTracker:
    def __init__(self, eq, confidence=1.0):
        self._eq = eq
        self.confidence = confidence

    def hero_equity(self, hole, board, n_runouts=None, rng=None):
        return self._eq


class _StubDB:
    """A blueprint that DOES have a trained strategy at any key (mass on legal)."""
    def get_average_strategy(self, key):
        return {'call': 0.5, 'raise_small': 0.5}


def _solver():
    return RiverSubgameSolver(None, guard_confidence=0.2, guard_margin=1.0)


def _ps(eq, *, street='preflop', to_call=66.0, bot_stack=74.0, pot=114.0,
        confidence=1.0, seat=0, community=None, hole=('CA', 'SA')):
    return {
        'street': street,
        'community': [] if community is None else community,
        'hole_cards': list(hole),
        'seat': seat,
        'p0_stack': bot_stack if seat == 0 else 200.0,
        'p1_stack': bot_stack if seat == 1 else 200.0,
        'to_call': to_call,
        'pot': pot,
        'opp_range': _StubTracker(eq, confidence),
    }


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


def test_untrained_premium_calls():
    """THE fix: AA at a beyond-cap 5-bet node (untrained key) -> CALL, not the
    50/50 fold of the passive fallback. eq 0.85 >= pot odds 0.367."""
    s = _solver()
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL, _ps(0.85))
    check('untrained premium -> call', out == 'call', f"got {out!r}")


def test_untrained_trash_folds():
    """A hand below pot odds at the untrained node folds (eq 0.10 < 0.367)."""
    s = _solver()
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL, _ps(0.10))
    check('untrained trash -> fold', out == 'fold', f"got {out!r}")


def test_never_raises_from_untrained_node():
    """The guard only ever returns call/fold -- never a sized raise/jam from an
    untrained node (the BUG-011 stray-raise class), even with monster equity."""
    s = _solver()
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL, _ps(0.99))
    check('never raises from untrained node', out in ('call', 'fold'), f"got {out!r}")


def test_trained_key_defers():
    """A TRAINED key (blueprint has usable mass on the legal actions) is left to its
    learned mixed strategy -- the guard returns None."""
    s = _solver()
    s.db = _StubDB()
    out = s._facing_deep_raise_guard('pf_29_ip_sl', _LEGAL, _ps(0.85))
    check('trained key defers to blueprint', out is None, f"got {out!r}")


def test_trained_but_no_mass_on_legal_acts():
    """A key trained under a DIFFERENT legal set (zero mass on the CURRENT legal
    actions) is effectively untrained here -> the guard acts (mirrors exactly when
    _distribution would fall into its passive fallback)."""
    class _OffLegalDB:
        def get_average_strategy(self, key):
            return {'bet_small': 1.0}      # not in _LEGAL
    s = _solver()
    s.db = _OffLegalDB()
    out = s._facing_deep_raise_guard('k', _LEGAL, _ps(0.85))
    check('trained off-legal -> acts (call)', out == 'call', f"got {out!r}")


def test_untrained_faced_allin_acts_on_equity():
    """A faced all-in (to_call >= bot_stack) at an UNTRAINED key is no longer punted
    to a coin-flip: the deep guard decides it on equity too (a called jam runs the
    board out -> exact). AA -> call. (At decide() level the all-in guard runs FIRST
    and owns the TRUSTED jam; this fires only when it has deferred -- see decide
    tests.)"""
    s = _solver()
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL,
                                     _ps(0.85, to_call=74.0, bot_stack=74.0))
    check('untrained faced all-in -> call on equity', out == 'call', f"got {out!r}")


def test_not_facing_bet_defers():
    s = _solver()
    out = s._facing_deep_raise_guard('k', ['check', 'bet_small'], _ps(0.85))
    check('not facing a bet defers', out is None, f"got {out!r}")


def test_river_defers():
    """River is handled by the full subgame solver, never this guard."""
    s = _solver()
    out = s._facing_deep_raise_guard('k', _LEGAL, _ps(0.85, street='river',
                                                      community=['HK', 'SQ', 'D7', 'C2', 'H9']))
    check('river defers to solver', out is None, f"got {out!r}")


def test_low_confidence_premium_still_calls_via_uniform():
    """The core of the AA-fold fix: when a maniac has COLLAPSED confidence, the guard
    does NOT defer to the blueprint's coin-flip -- it judges against a UNIFORM range
    (real equity, card removal) and a premium still clears pot odds -> CALL. Uses the
    turn (exact 1-card runout) so the real-equity path is fast + deterministic. The
    stub's fixed eq is bypassed here (the uniform fallback computes real equity)."""
    s = _solver()
    out = s._facing_deep_raise_guard(
        'pf_9_5_ip_turn_lll', _LEGAL,
        _ps(0.0, street='turn', community=['HK', 'SQ', 'D7', 'C2'],
            hole=('CA', 'SA'), confidence=0.05))
    check('low-confidence premium -> call (uniform floor)', out == 'call', f"got {out!r}")


def test_low_confidence_trash_still_folds_via_uniform():
    """The uniform fallback is not a blanket call: genuine trash facing big pot odds
    still folds (3-high vs a uniform range, to_call 150 of a 200 stack -> pot odds
    0.71)."""
    s = _solver()
    out = s._facing_deep_raise_guard(
        'pf_0_0_ip_turn_lll', _LEGAL,
        _ps(0.0, street='turn', community=['HK', 'SQ', 'D7', 'C9'],
            hole=('H3', 'S2'), to_call=150.0, bot_stack=200.0, pot=60.0,
            confidence=0.05))
    check('low-confidence trash -> fold (uniform floor)', out == 'fold', f"got {out!r}")


def test_seat1_reads_bot_stack():
    """Seat-1 bot: bot_stack must come from p1_stack."""
    s = _solver()
    out = s._facing_deep_raise_guard('pf_29_oop_slll', _LEGAL, _ps(0.85, seat=1))
    check('seat-1 reads p1_stack -> call', out == 'call', f"got {out!r}")


def test_postflop_street_applies():
    """The guard also covers flop/turn deep raises, not just preflop."""
    s = _solver()
    out = s._facing_deep_raise_guard('pf_9_5_ip_turn_lll', _LEGAL,
                                     _ps(0.85, street='turn',
                                         community=['HK', 'SQ', 'D7', 'C2']))
    check('turn deep raise applies', out == 'call', f"got {out!r}")


# --- decide() integration -----------------------------------------------------

def test_decide_routes_through_deep_guard():
    """A faced deep (non-all-in) raise at an untrained key reaches the guard via
    decide(), decide() RETURNS its action, the stat ticks, and last_debug records
    mode='deep_raise_guard'. db=None makes the key untrained."""
    s = _solver()
    before = s.stats['deep_raise_guard']
    action = s.decide('pf_29_ip_slll', _LEGAL, _ps(0.85))
    check('decide returns deep-guard action', action == 'call', f"got {action!r}")
    check('decide ticked deep_raise_guard stat',
          s.stats['deep_raise_guard'] == before + 1)
    check('decide last_debug mode',
          (s.last_debug or {}).get('mode') == 'deep_raise_guard', s.last_debug)


def test_decide_trusted_allin_handled_by_allin_guard():
    """Ordering: a TRUSTED faced all-in (confidence high) is handled by the all-in
    guard FIRST -- decide() records mode='allin_guard', not 'deep_raise_guard'."""
    s = _solver()
    action = s.decide('pf_29_ip_slll', ['fold', 'call'],
                      _ps(0.85, to_call=74.0, bot_stack=74.0))
    check('all-in guard wins ordering', action == 'call', f"got {action!r}")
    check('mode is allin_guard not deep',
          (s.last_debug or {}).get('mode') == 'allin_guard', s.last_debug)
    check('deep stat NOT ticked', s.stats['deep_raise_guard'] == 0)


def test_decide_untrained_jam_low_conf_routes_to_deep_guard():
    """The hole #2 closes: a faced all-in at an UNTRAINED key with COLLAPSED
    confidence. The all-in guard DEFERS (low conf) -> the deep guard catches it and
    calls a premium on uniform-floor equity (mode='deep_raise_guard'), instead of the
    blueprint's 50/50. Turn board -> fast, exact real equity (no stub)."""
    s = _solver()
    ps = _ps(0.0, street='turn', community=['HK', 'SQ', 'D7', 'C2'],
             hole=('CA', 'SA'), to_call=80.0, bot_stack=80.0, pot=60.0,
             confidence=0.05)
    action = s.decide('pf_9_5_ip_turn_lll', ['fold', 'call'], ps)
    check('untrained jam low-conf -> deep guard call', action == 'call', f"got {action!r}")
    check('mode is deep_raise_guard',
          (s.last_debug or {}).get('mode') == 'deep_raise_guard', s.last_debug)


# --- #1 uniform-floor cache (latency fix) -------------------------------------

def test_uniform_floor_caches_and_is_sane():
    s = _solver()
    hole, board = ('CA', 'SA'), ['HK', 'SQ', 'D7', 'C2']   # turn = exact + fast
    e1 = s._uniform_floor_equity(hole, board)
    cached = (frozenset(hole), frozenset(board)) in s._uniform_eq_cache
    e2 = s._uniform_floor_equity(hole, board)
    check('uniform floor cached', cached and e1 == e2, f"{e1},{e2}")
    check('AA on dry turn floor is high', e1 > 0.7, f"{e1}")


# --- #2 exception path -> safe call, never the coin-flip ----------------------

def test_safe_untrained_call_untrained_facing_bet():
    s = _solver()
    check('untrained faced bet -> call', s._safe_untrained_call('k', _LEGAL, _ps(0.0)) == 'call')


def test_safe_untrained_call_trained_defers():
    s = _solver()
    s.db = _StubDB()
    check('trained -> None', s._safe_untrained_call('k', _LEGAL, _ps(0.0)) is None)


def test_safe_untrained_call_not_facing_bet_defers():
    s = _solver()
    check('no bet -> None', s._safe_untrained_call('k', ['check', 'bet_small'], _ps(0.0)) is None)


def test_decide_deep_guard_exception_returns_safe_call():
    """A throw inside the deep guard at an untrained faced-bet node must yield a safe
    CALL, not decide()'s fall-through to the blueprint coin-flip. Boom tracker with
    high confidence forces the TRUSTED branch (which calls the throwing hero_equity)."""
    class _Boom:
        confidence = 1.0
        def hero_equity(self, *a, **k):
            raise ValueError('boom')
    ps = _ps(0.0)
    ps['opp_range'] = _Boom()
    s = _solver()
    action = s.decide('pf_29_ip_slll', _LEGAL, ps)
    check('deep-guard exception -> safe call', action == 'call', f"got {action!r}")
    check('counted the fallback', s.stats['fallback'] >= 1)


# --- #3 AA/KK never fold preflop floor ---------------------------------------

def test_premium_floor_upgrades_aa_fold_to_call():
    s = _solver()
    out = s._premium_no_fold('fold', _LEGAL, _ps(0.0, hole=('CA', 'SA')))
    check('AA preflop fold -> call', out == 'call', f"got {out!r}")


def test_premium_floor_upgrades_kk_fold_to_call():
    s = _solver()
    out = s._premium_no_fold('fold', _LEGAL, _ps(0.0, hole=('CK', 'SK')))
    check('KK preflop fold -> call', out == 'call', f"got {out!r}")


def test_premium_floor_leaves_qq_alone():
    """QQ is NOT a never-fold hand (foldable to a 4-bet jam) -- the floor must not touch
    it (it is scoped to AA/KK by rank, not the equity bucket which also holds QQ/AKs)."""
    s = _solver()
    out = s._premium_no_fold('fold', _LEGAL, _ps(0.0, hole=('CQ', 'SQ')))
    check('QQ preflop fold left as fold', out == 'fold', f"got {out!r}")


def test_premium_floor_postflop_aa_untouched():
    """AA can correctly fold postflop on a scary board -> the floor is preflop-only."""
    s = _solver()
    out = s._premium_no_fold('fold', _LEGAL, _ps(0.0, street='turn',
                             community=['HK', 'SQ', 'D7', 'C2'], hole=('CA', 'SA')))
    check('AA postflop fold left as fold', out == 'fold', f"got {out!r}")


def test_premium_floor_passes_through_nonfold():
    s = _solver()
    out = s._premium_no_fold('raise_large', _LEGAL, _ps(0.0, hole=('CA', 'SA')))
    check('AA raise passes through', out == 'raise_large', f"got {out!r}")


# --- _is_untrained helper + B1 LRU cache + C1 value-jam ----------------------

_LEGAL_ALLIN = ['fold', 'call', 'allin']


def _jam_solver():
    return RiverSubgameSolver(None, guard_confidence=0.2, guard_margin=1.0, value_jam=True)


def test_is_untrained_helper():
    s = _solver()
    check('db=None -> untrained', s._is_untrained('k', _LEGAL) is True)
    s2 = _solver()
    s2.db = _StubDB()
    check('trained -> not untrained', s2._is_untrained('k', _LEGAL) is False)

    class _Off:
        def get_average_strategy(self, k):
            return {'bet_small': 1.0}        # no mass on _LEGAL
    s3 = _solver()
    s3.db = _Off()
    check('off-legal mass -> untrained', s3._is_untrained('k', _LEGAL) is True)


def test_uniform_cache_lru_evicts_and_is_order_free():
    s = _solver()
    s._uniform_eq_cap = 2
    board = ['HK', 'SQ', 'D7', 'C2']
    s._uniform_floor_equity(('CA', 'SA'), board)
    s._uniform_floor_equity(('CK', 'SK'), board)
    s._uniform_floor_equity(('CQ', 'SQ'), board)   # len 3 > 2 -> evict the oldest (AA)
    check('cache capped (LRU evict, no flush)', len(s._uniform_eq_cache) == 2,
          len(s._uniform_eq_cache))
    e1 = s._uniform_floor_equity(('CA', 'SA'), board)
    e2 = s._uniform_floor_equity(('CA', 'SA'), ['C2', 'D7', 'SQ', 'HK'])  # permuted board
    check('board key is order-free (cache hit)', e1 == e2, f"{e1},{e2}")


def test_value_jam_monster_jams_when_enabled():
    s = _jam_solver()
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL_ALLIN, _ps(0.90))
    check('value-jam: monster -> allin', out == 'allin', f"got {out!r}")


def test_value_jam_off_by_default():
    s = _solver()                                  # value_jam=False
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL_ALLIN, _ps(0.90))
    check('value-jam off -> call', out == 'call', f"got {out!r}")


def test_value_jam_needs_a_legal_allin():
    s = _jam_solver()
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL, _ps(0.90))   # no 'allin'
    check('value-jam: no legal allin -> call (never construct one)', out == 'call', f"got {out!r}")


def test_value_jam_not_for_marginal_equity():
    s = _jam_solver()
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL_ALLIN, _ps(0.55))
    check('value-jam: marginal eq -> call not jam', out == 'call', f"got {out!r}")


def test_value_jam_never_upgrades_a_fold():
    s = _jam_solver()
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL_ALLIN, _ps(0.10))
    check('value-jam: a fold stays a fold', out == 'fold', f"got {out!r}")


def test_value_jam_not_when_already_allin():
    s = _jam_solver()
    out = s._facing_deep_raise_guard('pf_29_ip_slll', _LEGAL_ALLIN,
                                     _ps(0.90, to_call=74.0, bot_stack=74.0))
    check('value-jam: already all-in -> call (no money behind)', out == 'call', f"got {out!r}")


# --- Part 2: top-X% jam range for an uninformed faced all-in (BUG-022) ---------

def test_jam_range_equity_below_uniform_for_dominated():
    s = _solver()
    eu = s._uniform_floor_equity(('D8', 'HT'), [])       # T8o vs uniform
    ej = s._jam_range_equity(('D8', 'HT'), [], 0.20)     # T8o vs a top-20% jam range
    check('jam-range eq << uniform eq for T8o', ej < eu - 0.03 and ej < 0.40, f"{ej} vs {eu}")


def test_jam_range_equity_caches():
    s = _solver()
    e1 = s._jam_range_equity(('CA', 'SA'), [], 0.20)
    cached = (frozenset(('CA', 'SA')), frozenset([]), round(0.20, 3)) in s._jam_eq_cache
    e2 = s._jam_range_equity(('CA', 'SA'), [], 0.20)
    check('jam-range cached + stable', cached and e1 == e2, f"{e1},{e2}")


def test_jam_range_postflop_ranks_by_centroid_mean():
    """Postflop: the EMD bucket index isn't equity-ordered, but ranking by the bucket's
    centroid MEAN equity is -- and is draw-aware. A weak made hand on a turn gets a much
    lower jam-range equity than vs uniform (so it folds to a turn jam); a monster stays
    high."""
    s = _solver()
    board = ['HK', 'SQ', 'D7', 'C2']
    eu_weak = s._uniform_floor_equity(('D8', 'HT'), board)
    ej_weak = s._jam_range_equity(('D8', 'HT'), board, 0.20)
    ej_aa = s._jam_range_equity(('CA', 'SA'), board, 0.20)
    check('postflop jam-range drops a weak hand vs uniform', ej_weak < eu_weak - 0.03,
          f"{ej_weak} vs {eu_weak}")
    check('postflop jam-range keeps AA strong', ej_aa > 0.60, f"{ej_aa}")


def test_uninformed_preflop_allin_folds_dominated():
    """BUG-022: a faced all-in preflop with a collapsed read judges T8o vs the top-20%
    jam range (eq ~0.33 < pot odds 0.495) -> FOLD, instead of calling off vs uniform."""
    s = _solver()
    ps = _ps(0.0, to_call=99.0, bot_stack=99.0, pot=101.0, confidence=0.05, hole=('D8', 'HT'))
    out = s._facing_deep_raise_guard('pf_13_oop_a', _LEGAL_ALLIN, ps)
    check('uninformed preflop all-in: T8o -> fold', out == 'fold', f"got {out!r}")


def test_uninformed_preflop_allin_premium_still_calls():
    s = _solver()
    ps = _ps(0.0, to_call=99.0, bot_stack=99.0, pot=101.0, confidence=0.05, hole=('CA', 'SA'))
    out = s._facing_deep_raise_guard('pf_29_oop_a', _LEGAL_ALLIN, ps)
    check('uninformed preflop all-in: AA -> call', out == 'call', f"got {out!r}")


def test_decide_preflop_jam_uninformed_folds_dominated_end_to_end():
    """Full path: faced all-in preflop, collapsed confidence -> all-in guard defers ->
    deep guard judges vs the jam range -> T8o folds (the served-bot behavior)."""
    s = _solver()
    ps = _ps(0.0, to_call=99.0, bot_stack=99.0, pot=101.0, confidence=0.05, hole=('D8', 'HT'))
    out = s.decide('pf_13_oop_a', ['fold', 'call'], ps)
    check('decide end-to-end: T8o jam -> fold', out == 'fold', f"got {out!r}")


def test_turn_uninformed_allin_premium_calls():
    """A faced all-in on the TURN with an uninformed read now uses the centroid-mean jam
    range (the postflop-gap fix). AA (overpair) clears it -> call."""
    s = _solver()
    ps = _ps(0.0, street='turn', community=['HK', 'SQ', 'D7', 'C2'],
             to_call=80.0, bot_stack=80.0, pot=60.0, confidence=0.05, hole=('CA', 'SA'))
    out = s._facing_deep_raise_guard('pf_9_5_oop_turn_a', _LEGAL_ALLIN, ps)
    check('turn all-in, uninformed: AA calls', out == 'call', f"got {out!r}")


def test_turn_uninformed_allin_weak_made_hand_folds():
    """The postflop-gap fix: a WEAK made hand facing a turn jam with no read folds vs the
    centroid-mean jam range (it called off vs the uniform floor before). 87 on K-Q-7-2 =
    bottom-ish pair, a dog vs a top-20% turn jam range."""
    s = _solver()
    ps = _ps(0.0, street='turn', community=['HK', 'SQ', 'D7', 'C2'],
             to_call=80.0, bot_stack=80.0, pot=60.0, confidence=0.05, hole=('S8', 'H7'))
    out = s._facing_deep_raise_guard('pf_3_4_oop_turn_a', _LEGAL_ALLIN, ps)
    check('turn all-in, uninformed: weak pair folds', out == 'fold', f"got {out!r}")


def test_b1_preflop_money_behind_uninformed_folds_dominated():
    """B1: an uninformed PREFLOP money-behind (non-all-in) deep raise at a beyond-cap node
    also uses the jam range (the range is selected), so a dominated hand folds instead of
    calling vs uniform."""
    s = _solver()
    ps = _ps(0.0, to_call=60.0, bot_stack=120.0, pot=70.0, confidence=0.05, hole=('D8', 'HT'))
    out = s._facing_deep_raise_guard('pf_13_ip_llm', _LEGAL, ps)
    check('B1: preflop money-behind T8o -> fold', out == 'fold', f"got {out!r}")


def test_informativeness_gate():
    """A6/BUG-022 generalization: a UNIFORM belief at high confidence is NOT trusted (it
    carries no read); a CONCENTRATED belief is. A stub without weights defers to the
    confidence gate (so existing stub tests are unaffected)."""
    import numpy as np

    class _W:
        def __init__(self, w, c):
            self.w = np.array(w, float)
            self.confidence = c
    uni = _W([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1.0)
    conc = _W([10, 1, 0, 0, 0, 0, 0, 0, 0, 0], 1.0)
    mild = _W([1.5] * 4 + [1.0] * 16, 1.0)               # a real ~1.5x tilt (ratio ~0.968)
    s = _solver()
    check('uniform belief: not informative', s._belief_is_informative(uni) is False)
    check('concentrated belief: informative', s._belief_is_informative(conc) is True)
    # Regression guard: a genuine MILD read must be TRUSTED, not discarded as "uniform"
    # (the 0.95-threshold bug reverted these to un-adapted blueprint play at trained
    # all-in nodes). At 0.99 the 1.5x tilt is informative.
    check('mild real read: informative (not discarded)', s._belief_is_informative(mild) is True)
    check('trust_read False on uniform@high-conf', s._trust_read(uni) is False)
    check('trust_read True on concentrated@high-conf', s._trust_read(conc) is True)
    check('trust_read defers to confidence for a stub (no weights)',
          s._trust_read(_StubTracker(0.9, 1.0)) is True)


# --- Inference #1: first-to-act untrained value-bet -------------------------

_LEGAL_FIRST_ACT = ['check', 'bet_small', 'bet_medium', 'bet_large']
_FLOP = ['HK', 'SQ', 'D7']


def test_first_act_value_bets_strong_untrained():
    """An untrained first-to-act flop node with a strong hand value-bets a trained size
    instead of checking 100% (the passive fallback). Stub tracker @ high confidence ->
    _trust_read uses its eq 0.85 >= FIRST_ACT_VALUE_EQ -> bet_medium."""
    s = _solver()
    out = s._first_act_value_guard('flopkey', _LEGAL_FIRST_ACT,
                                   _ps(0.85, street='flop', community=_FLOP, hole=('CA', 'SA')))
    check('first-act untrained strong -> value bet', out == 'bet_medium', f"got {out!r}")


def test_first_act_value_checks_weak():
    s = _solver()
    out = s._first_act_value_guard('flopkey', _LEGAL_FIRST_ACT,
                                   _ps(0.40, street='flop', community=_FLOP, hole=('C7', 'D2')))
    check('first-act untrained weak -> check (None)', out is None, f"got {out!r}")


def test_first_act_value_defers_facing_bet():
    s = _solver()
    out = s._first_act_value_guard('flopkey', _LEGAL,
                                   _ps(0.85, street='flop', community=_FLOP))
    check('facing a bet -> defer (guards own it)', out is None, f"got {out!r}")


def test_first_act_value_defers_trained():
    class _FlopDB:                                      # mass ON the first-act legal set
        def get_average_strategy(self, k):
            return {'check': 0.6, 'bet_medium': 0.4}
    s = _solver()
    s.db = _FlopDB()
    out = s._first_act_value_guard('k', _LEGAL_FIRST_ACT,
                                   _ps(0.85, street='flop', community=_FLOP, hole=('CA', 'SA')))
    check('trained -> blueprint plays (None)', out is None, f"got {out!r}")


def test_first_act_value_preflop_and_river_defer():
    s = _solver()
    pf = s._first_act_value_guard('pf', _LEGAL_FIRST_ACT, _ps(0.85, street='preflop', hole=('CA', 'SA')))
    rv = s._first_act_value_guard('rk', _LEGAL_FIRST_ACT,
                                  _ps(0.85, street='river', community=_FLOP + ['C2', 'H9'], hole=('CA', 'SA')))
    check('preflop defers (trained opens)', pf is None, f"got {pf!r}")
    check('river defers (solver owns it)', rv is None, f"got {rv!r}")


TESTS = [
    test_untrained_premium_calls,
    test_untrained_trash_folds,
    test_never_raises_from_untrained_node,
    test_trained_key_defers,
    test_trained_but_no_mass_on_legal_acts,
    test_untrained_faced_allin_acts_on_equity,
    test_not_facing_bet_defers,
    test_river_defers,
    test_low_confidence_premium_still_calls_via_uniform,
    test_low_confidence_trash_still_folds_via_uniform,
    test_seat1_reads_bot_stack,
    test_postflop_street_applies,
    test_decide_routes_through_deep_guard,
    test_decide_trusted_allin_handled_by_allin_guard,
    test_decide_untrained_jam_low_conf_routes_to_deep_guard,
    test_first_act_value_bets_strong_untrained,
    test_first_act_value_checks_weak,
    test_first_act_value_defers_facing_bet,
    test_first_act_value_defers_trained,
    test_first_act_value_preflop_and_river_defer,
    test_uniform_floor_caches_and_is_sane,
    test_safe_untrained_call_untrained_facing_bet,
    test_safe_untrained_call_trained_defers,
    test_safe_untrained_call_not_facing_bet_defers,
    test_decide_deep_guard_exception_returns_safe_call,
    test_premium_floor_upgrades_aa_fold_to_call,
    test_premium_floor_upgrades_kk_fold_to_call,
    test_premium_floor_leaves_qq_alone,
    test_premium_floor_postflop_aa_untouched,
    test_premium_floor_passes_through_nonfold,
    test_is_untrained_helper,
    test_uniform_cache_lru_evicts_and_is_order_free,
    test_value_jam_monster_jams_when_enabled,
    test_value_jam_off_by_default,
    test_value_jam_needs_a_legal_allin,
    test_value_jam_not_for_marginal_equity,
    test_value_jam_never_upgrades_a_fold,
    test_value_jam_not_when_already_allin,
    test_jam_range_equity_below_uniform_for_dominated,
    test_jam_range_equity_caches,
    test_jam_range_postflop_ranks_by_centroid_mean,
    test_uninformed_preflop_allin_folds_dominated,
    test_uninformed_preflop_allin_premium_still_calls,
    test_decide_preflop_jam_uninformed_folds_dominated_end_to_end,
    test_turn_uninformed_allin_premium_calls,
    test_turn_uninformed_allin_weak_made_hand_folds,
    test_b1_preflop_money_behind_uninformed_folds_dominated,
    test_informativeness_gate,
]

if __name__ == '__main__':
    for fn in TESTS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            _failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
