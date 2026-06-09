# backend/bot/tests/test_allin_guard.py
"""
Unit tests for the flop/turn FACING-an-all-in guard
(RiverSubgameSolver._facing_allin_guard).

The guard is a pure equity-vs-pot-odds call/fold decision against the live
opponent belief, applied only when calling commits the bot's WHOLE stack (the
near-terminal case: the board runs out, no continuation to value). A stub tracker
with a fixed equity + confidence makes every branch deterministic -- no DB, no
blueprint, no Monte Carlo.

Run: python tests/test_allin_guard.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_subgame_solver import RiverSubgameSolver


class _StubTracker:
    """Minimal RangeTracker stand-in: fixed equity + confidence."""
    def __init__(self, eq, confidence=1.0):
        self._eq = eq
        self.confidence = confidence

    def hero_equity(self, hole, board, n_runouts=None, rng=None):
        # Fixed equity regardless of street/runouts (the guard passes n_runouts).
        return self._eq


def _solver():
    # No DB needed: the guard never touches the blueprint or a solve.
    return RiverSubgameSolver(None, guard_confidence=0.2, guard_margin=1.0)


def _ps(eq, *, street='flop', to_call=80.0, bot_stack=80.0, pot=40.0,
        confidence=1.0, seat=0, community=None):
    """A facing-an-all-in public_state. Default: pot 40, opponent jams so the bot
    must call its whole 80 stack -> pot odds = 80/(40+80) = 0.667. `community`
    defaults to a flop; pass [] for preflop."""
    return {
        'street': street,
        'community': ['HK', 'SQ', 'D7'] if community is None else community,
        'hole_cards': ['HA', 'DA'],
        'seat': seat,
        'p0_stack': bot_stack if seat == 0 else 200.0,
        'p1_stack': bot_stack if seat == 1 else 200.0,
        'to_call': to_call,
        'pot': pot,
        'opp_range': _StubTracker(eq, confidence),
    }


_LEGAL = ['fold', 'call']
_passed = _failed = 0


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


def test_calls_when_equity_beats_pot_odds():
    s = _solver()
    # eq 0.90: EV(call) = 0.9*120 - 80 = +28 -> call.
    check('calls with high equity', s._facing_allin_guard(_LEGAL, _ps(0.90)) == 'call')


def test_folds_when_equity_below_pot_odds():
    s = _solver()
    # eq 0.10: EV(call) = 0.1*120 - 80 = -68 -> fold.
    check('folds with low equity', s._facing_allin_guard(_LEGAL, _ps(0.10)) == 'fold')


def test_pot_odds_boundary_defers_when_knife_edge():
    s = _solver()
    # eq exactly at pot odds 0.6667: EV(call) ~ 0 -> within margin -> defer (None).
    out = s._facing_allin_guard(_LEGAL, _ps(80.0 / 120.0))
    check('knife-edge defers to blueprint', out is None, f"got {out!r}")


def test_clear_call_just_past_margin():
    s = _solver()
    # eq 0.70: EV(call) = 0.70*120 - 80 = +4 (> margin 1) -> call.
    check('clear call past margin', s._facing_allin_guard(_LEGAL, _ps(0.70)) == 'call')


def test_defers_when_money_behind():
    s = _solver()
    # to_call 40 < stack 80: calling leaves 40 behind -> NOT near-terminal -> defer,
    # even though folding the strong hand would be wrong (that's the blueprint's job).
    out = s._facing_allin_guard(_LEGAL, _ps(0.10, to_call=40.0, bot_stack=80.0))
    check('defers when not all-in (money behind)', out is None, f"got {out!r}")


def test_defers_on_river():
    s = _solver()
    # River routes through the full solver, never the guard.
    out = s._facing_allin_guard(_LEGAL, _ps(0.10, street='river'))
    check('defers on river', out is None, f"got {out!r}")


def test_defers_when_not_facing_bet():
    s = _solver()
    out = s._facing_allin_guard(['check', 'bet_small'], _ps(0.10))
    check('defers when not facing a bet', out is None, f"got {out!r}")


def test_defers_when_confidence_low():
    s = _solver()
    # Belief untrusted -> defer to blueprint even with a clear equity edge.
    out = s._facing_allin_guard(_LEGAL, _ps(0.10, confidence=0.05))
    check('defers when confidence below threshold', out is None, f"got {out!r}")


def test_works_for_seat_1():
    s = _solver()
    # Same decision, bot in seat 1 (stack read from p1_stack).
    check('seat-1 call', s._facing_allin_guard(_LEGAL, _ps(0.90, seat=1)) == 'call')


def test_turn_street_applies():
    s = _solver()
    ps = _ps(0.90, street='turn')
    ps['community'] = ['HK', 'SQ', 'D7', 'C2']
    check('guard applies on the turn', s._facing_allin_guard(_LEGAL, ps) == 'call')


def test_high_spr_overbet_jam_acts_when_trusted():
    """A massive OVERBET jam from a high-SPR start (small pot, deep stacks) still
    triggers the guard -- calling IS all-in, so it's near-terminal regardless of the
    pre-jam SPR. With a trusted belief the pot-odds call/fold is correct (trash folds:
    pot 6, jam 198 -> needs 198/204 = 0.97 equity)."""
    s = _solver()
    ps = _ps(0.15, to_call=198.0, bot_stack=198.0, pot=6.0)
    check('high-SPR overbet, trusted -> acts (fold trash)',
          s._facing_allin_guard(_LEGAL, ps) == 'fold')


def test_high_spr_overbet_jam_defers_when_untrusted():
    """The real high-SPR danger: a giant overbet is off-model, the call risks the
    whole stack, so a bad belief is expensive. An off-model jam decays confidence ->
    the guard DEFERS to the blueprint rather than act on an untrusted belief. (Here
    even a 'call'-leaning equity must yield None because confidence is below gate.)"""
    s = _solver()
    ps = _ps(0.95, to_call=198.0, bot_stack=198.0, pot=6.0, confidence=0.05)
    out = s._facing_allin_guard(_LEGAL, ps)
    check('high-SPR overbet, untrusted -> defers', out is None, f"got {out!r}")


def test_preflop_jam_calls_with_high_equity():
    """Fix #1: the guard now covers PREFLOP (empty board). High equity vs the
    tracked range -> call (pot 40, jam to 80 -> EV(call)=0.90*120-80=+28)."""
    s = _solver()
    out = s._facing_allin_guard(_LEGAL, _ps(0.90, street='preflop', community=[]))
    check('preflop jam: high equity -> call', out == 'call', f"got {out!r}")


def test_preflop_jam_folds_with_low_equity():
    """The leak this fixes: vs a strong believed jamming range a marginal hand FOLDS
    preflop instead of the blueprint's un-adapted GTO call (eq 0.40 ->
    EV(call)=0.40*120-80=-32 -> fold)."""
    s = _solver()
    out = s._facing_allin_guard(_LEGAL, _ps(0.40, street='preflop', community=[]))
    check('preflop jam: low equity -> fold', out == 'fold', f"got {out!r}")


def test_preflop_empty_board_accepted():
    """An empty community ([] preflop) must be ACCEPTED -- the reject is `board is
    None`, not falsy-empty. Regression guard for the preflop extension."""
    s = _solver()
    out = s._facing_allin_guard(_LEGAL, _ps(0.90, street='preflop', community=[]))
    check('preflop empty board accepted', out in ('call', 'fold'), f"got {out!r}")


# --- boundary + robustness cases (audit follow-ups) ----------------------------

def test_opponent_short_jam_defers_known_limitation():
    """KNOWN LIMITATION (accepted for fixed-size H2H, 2026-06-01): when the OPPONENT
    jams for LESS than the bot's stack (to_call < bot_stack), calling is still
    near-terminal (opp all-in, board runs out) -- but the gate `to_call < bot_stack
    - eps` REJECTS it, so the guard defers to the blueprint. Unreachable in HU today
    (stacks reset equal each street, so a jam is always the full equal stack); this
    test PINS the current behavior so a future unequal-stack change is a conscious
    one, not a silent regression. eq=0.10 (a clear fold) is deliberately ignored."""
    s = _solver()
    out = s._facing_allin_guard(_LEGAL, _ps(0.10, to_call=60.0, bot_stack=80.0))
    check('opponent short jam defers (known limitation)', out is None, f"got {out!r}")


def test_confidence_exactly_at_threshold_acts():
    """The gate is `confidence < guard_confidence` (strict). At EXACTLY the
    threshold (0.2) the belief is trusted and the guard ACTS. Boundary lock."""
    s = _solver()  # guard_confidence=0.2
    out = s._facing_allin_guard(_LEGAL, _ps(0.90, confidence=0.2))
    check('confidence exactly at threshold acts', out == 'call', f"got {out!r}")


def test_confidence_just_below_threshold_defers():
    """Just under the threshold -> defer. Pairs with the at-threshold test to pin
    the strict `<` boundary from both sides."""
    s = _solver()
    out = s._facing_allin_guard(_LEGAL, _ps(0.90, confidence=0.199))
    check('confidence just below threshold defers', out is None, f"got {out!r}")


def test_missing_confidence_defers():
    """A tracker with NO `confidence` attribute must DEFER (the audit fix flipped
    the getattr default 1.0 -> 0.0: a malformed/foreign tracker is untrusted, and
    deferring is the safe direction for a whole-stack decision)."""
    class _NoConfTracker:
        def hero_equity(self, hole, board, n_runouts=None, rng=None):
            return 0.90
    ps = _ps(0.90)
    ps['opp_range'] = _NoConfTracker()  # no .confidence
    s = _solver()
    out = s._facing_allin_guard(_LEGAL, ps)
    check('missing confidence defers', out is None, f"got {out!r}")


def test_seat1_preflop_reads_bot_stack():
    """Seat-1 PREFLOP (empty board): bot_stack must be read from p1_stack, not p0.
    Combines the two extensions (preflop + seat 1) the audit found untested
    together. High equity -> call."""
    s = _solver()
    out = s._facing_allin_guard(
        _LEGAL, _ps(0.90, street='preflop', community=[], seat=1))
    check('seat-1 preflop reads bot stack -> call', out == 'call', f"got {out!r}")


# --- decide() integration (the guard reached through the real entry point) -----

def test_decide_routes_through_guard():
    """Integration: a faced jam reaches the guard via decide() (not just the unit
    method), decide() RETURNS the guard's action, and the allin_guard stat ticks.
    This is the path advance_bot_turns actually calls."""
    s = _solver()
    before = s.stats['allin_guard']
    action = s.decide('irrelevant_key', _LEGAL, _ps(0.90))
    check('decide returns guard action', action == 'call', f"got {action!r}")
    check('decide ticked allin_guard stat', s.stats['allin_guard'] == before + 1)
    check('decide last_debug mode', (s.last_debug or {}).get('mode') == 'allin_guard',
          s.last_debug)


def test_decide_guard_exception_defers_to_blueprint():
    """A guard exception must NOT crash decide() (advance_bot_turns only catches
    GameError -> an unguarded raise would 500 the turn). decide() catches it,
    records mode='guard_error', and falls through to the blueprint, returning a
    LEGAL action. With db=None the blueprint lookup yields uniform-over-legal."""
    class _BoomTracker:
        confidence = 1.0
        def hero_equity(self, hole, board, n_runouts=None, rng=None):
            raise ValueError("boom in equity")
    ps = _ps(0.90)
    ps['opp_range'] = _BoomTracker()
    s = _solver()
    before = s.stats['fallback']
    action = s.decide('irrelevant_key', _LEGAL, ps)
    check('decide survives guard exception', action in _LEGAL, f"got {action!r}")
    # A broken tracker trips BOTH equity guards: the all-in guard catches it (defers),
    # then the deep-raise guard's own equity call also throws and is caught -- each
    # ticks the fallback counter once (now +2). On the deep-guard error the catch block
    # then degrades SAFELY: at this untrained faced-bet node it returns 'call' rather
    # than the blueprint coin-flip (the #2 safety net), so the action is 'call'.
    check('decide counted both guard fallbacks', s.stats['fallback'] == before + 2,
          s.stats['fallback'])
    check('decide degraded to a safe call', action == 'call', f"got {action!r}")
    check('decide did NOT tick allin_guard on error', s.stats['allin_guard'] == 0,
          s.stats['allin_guard'])


TESTS = [
    test_calls_when_equity_beats_pot_odds,
    test_folds_when_equity_below_pot_odds,
    test_pot_odds_boundary_defers_when_knife_edge,
    test_clear_call_just_past_margin,
    test_defers_when_money_behind,
    test_defers_on_river,
    test_defers_when_not_facing_bet,
    test_defers_when_confidence_low,
    test_works_for_seat_1,
    test_turn_street_applies,
    test_high_spr_overbet_jam_acts_when_trusted,
    test_high_spr_overbet_jam_defers_when_untrusted,
    test_preflop_jam_calls_with_high_equity,
    test_preflop_jam_folds_with_low_equity,
    test_preflop_empty_board_accepted,
    test_opponent_short_jam_defers_known_limitation,
    test_confidence_exactly_at_threshold_acts,
    test_confidence_just_below_threshold_defers,
    test_missing_confidence_defers,
    test_seat1_preflop_reads_bot_stack,
    test_decide_routes_through_guard,
    test_decide_guard_exception_defers_to_blueprint,
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
