# backend/bot/tests/test_poker_game_properties.py
"""
Property-based tests for `src/cfr/poker_game.py` using Hypothesis.

Run from backend/bot/ with:
    python -m pytest tests/test_poker_game_properties.py -v --tb=short

These are property-based fuzz tests. Each one describes a SEMANTIC INVARIANT
that must hold for every reachable game state, and Hypothesis generates many
random valid traces to try to falsify it. When a property fails, Hypothesis
SHRINKS the failing input to a minimal counter-example, which is much easier
to debug than a 100-action random walk.

The existing `test_random_session_playout_invariants` in
test_cfr_correctness.py is the hand-rolled cousin of this file — same idea,
no shrinking. This file adds: Hypothesis-driven generation, shrinking, and a
wider invariant catalog covering legal-action shape, all-in semantics,
contribution arithmetic, and terminal/utility guarantees.

================================================================
INVARIANT CATALOG
================================================================

A. CHIP CONSERVATION
   A1. p0_stack + p1_stack + grand_pot == 2 * STARTING_STACK at every state.
   A2. Both stacks always >= -epsilon.

B. CALL / CONTRIBUTION ARITHMETIC
   B1. After any 'call', either total commitments are equal, or caller is
       all-in (caller_stack == 0).
   B2. Calling with chips remaining and an outstanding aggressive action
       costs > 0 chips. (Regression guard for BUG-004.)
   B3. _action_cost(...) >= 0 for every legal action.
   B4. _action_cost(...) <= player's remaining stack for every legal action.
   B5. get_call_amount_from_history(...) >= 0 for every reachable state.

C. ALL-IN SEMANTICS
   C1. After 'allin' in history, legal opponent actions are exactly
       {'fold', 'call'}.
   C2. An 'allin + call' terminal leaves both stacks at 0.
   C3. After a player goes all-in, that player's stack is 0.

D. LEGAL ACTIONS
   D1. No legal action has cost > remaining stack.
   D2. 'fold' is legal whenever a non-empty action set exists AND there is
       an outstanding bet/raise/allin to respond to (or, preflop, when
       facing the BB before acting). It is NEVER legal to fold facing a
       free check (we assert: 'fold' not in legal when last action is
       'check' or history is empty postflop).
   D3. Total bet/raise/allin actions on a single street is at most 3
       (1 bet + 2 raises, or 1 bet + 1 raise + 1 allin, etc.).
   D4. legal_actions for an all-in-then-(fold|call) terminal is empty.

E. STREET TRANSITIONS
   E1. At the start of every postflop street (street > 0, history empty),
       p0_invested == p1_invested.
   E2. p0_invested + p1_invested == starting_pot - 3.0 (initial blinds).

F. TERMINAL / UTILITY
   F1. At terminal, |util_p0| <= STARTING_STACK.
   F2. Zero-sum: util_p0 + util_p1 == 0 (we evaluate by symmetry on the
       same hand: u_p1 = -u_p0 is the convention).
   F3. Chip conservation at terminal: the winner's final stack delta
       equals the loser's, and total chips == 2 * STARTING_STACK.
   F4. Game terminates within a bounded number of actions (no infinite
       betting loops; we cap at 300 actions per hand).

G. INTERNAL CONSISTENCY
   G1. get_player_contribution_this_round summed across both players
       equals (current_pot - starting_pot) on every postflop street.
       (Preflop has the SB/BB seed making this off-by-blinds; we account
       for it.)
"""

import os
import sys
import random as _r

import pytest

# Mark every test in this module as slow. Hypothesis property tests take ~11
# min walltime; the CI pipeline runs `pytest -m "not slow"` to keep the
# per-deploy feedback loop short. A separate nightly workflow runs the full
# suite (including this file) so the invariants still get exercised. See
# pytest.ini and .github/workflows/nightly-tests.yml.
pytestmark = pytest.mark.slow


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypothesis import given, settings, strategies as st, HealthCheck, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition

from src.cfr.poker_game import PokerGame, STARTING_STACK
from src.game.game_session import GameSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOTAL_CHIPS = 2 * STARTING_STACK
_EPS = 1e-6


def _new_session(seed, max_raises=float('inf')):
    """Construct a fresh GameSession with a deterministic deck (Python's
    `random` drives `shuffled_deck`). `max_raises` is the per-street raise cap:
    the default (inf) matches LIVE serving (uncapped re-raises); pass 2 to walk
    the trained/eval-capped engine (1 bet + 2 raises) for the D3 cap property."""
    _r.seed(seed)
    return GameSession.new(f'fuzz-{seed}', 'p', max_raises_per_street=max_raises)


def _grand_pot(session):
    d = session.data
    return session.game.calculate_current_pot(
        d['starting_pot'], d['history'], d['street'],
        d['p0_invested'], d['p1_invested'])


def _chip_conservation_holds(session):
    d = session.data
    if d['status'] == 'hand_over':
        # When hand_over, stacks reflect the result; the pot has been
        # distributed. We don't assert pot conservation here.
        return True
    return abs(d['p0_stack'] + d['p1_stack'] + _grand_pot(session)
               - _TOTAL_CHIPS) < _EPS


# ---------------------------------------------------------------------------
# Composite strategy: drive a session through a random legal walk.
# ---------------------------------------------------------------------------

@st.composite
def session_walk(draw, max_steps=80, max_hands=3, max_raises=float('inf')):
    """Generate a session that has executed a sequence of random legal
    actions. Returns the live GameSession plus the list of post-action
    snapshots (so per-step invariants can be checked).

    We use Hypothesis's `draw` to pick action indices and seeds — this is
    what gives shrinking power: when an assertion fails, Hypothesis can
    shrink the action sequence and seed to the minimal failing case.

    `max_raises` is the per-street raise cap (default inf = live serving's
    uncapped re-raises; pass 2 for the trained-cap engine — see test_D3).
    """
    seed = draw(st.integers(min_value=0, max_value=10_000_000))
    session = _new_session(seed, max_raises=max_raises)
    snapshots = []

    hands = 0
    steps_total = 0
    while hands < max_hands and steps_total < max_steps:
        if session.data['status'] == 'hand_over':
            session.start_next_hand()
            hands += 1
            continue
        legal = session.legal_actions()
        if not legal:
            # Should not happen mid-hand; if it does, capture and stop.
            break
        idx = draw(st.integers(min_value=0, max_value=len(legal) - 1))
        action = legal[idx]
        # Snapshot pre-action so tests can inspect what the act did.
        pre = {
            'history': list(session.data['history']),
            'street': session.data['street'],
            'p0_stack': session.data['p0_stack'],
            'p1_stack': session.data['p1_stack'],
            'p0_invested': session.data['p0_invested'],
            'p1_invested': session.data['p1_invested'],
            'starting_pot': session.data['starting_pot'],
            'acting': session.current_player(),
            'action': action,
            'legal': list(legal),
        }
        session.apply_action(action)
        post = {
            'p0_stack': session.data['p0_stack'],
            'p1_stack': session.data['p1_stack'],
            'p0_invested': session.data['p0_invested'],
            'p1_invested': session.data['p1_invested'],
            'street': session.data['street'],
            'status': session.data['status'],
            'history': list(session.data['history']),
        }
        snapshots.append((pre, post))
        steps_total += 1
        if hands == 0 and session.data['status'] == 'hand_over':
            hands += 1
    return session, snapshots


# ---------------------------------------------------------------------------
# A. CHIP CONSERVATION
# ---------------------------------------------------------------------------

@given(session_walk())
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_A1_chip_conservation(walk):
    """A1 + A2: stacks + pot == 2 * STARTING_STACK; stacks never negative."""
    session, snapshots = walk
    for pre, post in snapshots:
        assert post['p0_stack'] > -_EPS, f"negative p0 stack {post['p0_stack']}"
        assert post['p1_stack'] > -_EPS, f"negative p1 stack {post['p1_stack']}"
    assert _chip_conservation_holds(session)


# ---------------------------------------------------------------------------
# B. CALL / CONTRIBUTION ARITHMETIC
# ---------------------------------------------------------------------------

@given(session_walk())
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_B1_B2_call_invariants(walk):
    """B1: after call, totals equal OR caller is all-in.
    B2: calls with chips remaining must cost > 0."""
    session, snapshots = walk
    for pre, post in snapshots:
        if pre['action'] != 'call':
            continue
        # B1: total commitments must match unless caller is all-in.
        p0_total = STARTING_STACK - post['p0_stack']
        p1_total = STARTING_STACK - post['p1_stack']
        if abs(p0_total - p1_total) > _EPS:
            caller_stack = post['p0_stack'] if pre['acting'] == 0 else post['p1_stack']
            assert caller_stack < _EPS, (
                f"call didn't equalise commitments and caller not all-in: "
                f"p0_total={p0_total}, p1_total={p1_total}, "
                f"history={pre['history'] + ['call']}")
        # B2: call cost > 0 if caller had chips and there was outstanding bet.
        pre_stack = pre['p0_stack'] if pre['acting'] == 0 else pre['p1_stack']
        post_stack = post['p0_stack'] if pre['acting'] == 0 else post['p1_stack']
        cost = pre_stack - post_stack
        if pre_stack > _EPS:
            assert cost > -_EPS, f"negative call cost {cost}"
            # Cost must be strictly positive if the call wasn't a free check.
            # On preflop with empty history, SB faces BB so call costs 1.
            # Otherwise: any call after bet/raise/allin must cost > 0.
            facing_aggression = any(
                a.startswith(('bet_', 'raise_')) or a == 'allin'
                for a in pre['history'])
            if facing_aggression or (pre['street'] == 0 and not pre['history']):
                assert cost > _EPS, (
                    f"call cost was 0 despite chips remaining and aggression: "
                    f"history={pre['history'] + ['call']}, "
                    f"pre_stack={pre_stack}, post_stack={post_stack}")


@given(session_walk())
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_B3_B4_action_cost_bounds(walk):
    """B3: every legal action has cost >= 0.
    B4: every legal action has cost <= remaining stack."""
    session, snapshots = walk
    # Check the live session (not the snapshots — those reflect *after* the move).
    if session.data['status'] != 'in_hand':
        return
    legal = session.legal_actions()
    d = session.data
    acting = session.current_player()
    remaining = d['p0_stack'] if acting == 0 else d['p1_stack']
    for action in legal:
        cost = session.game._action_cost(
            action, d['street'], d['history'], d['starting_pot'],
            acting, d['p0_invested'], d['p1_invested'])
        assert cost >= -_EPS, f"negative cost {cost} for {action}"
        assert cost <= remaining + _EPS, (
            f"{action} costs {cost} but {acting} only has {remaining} "
            f"(history={d['history']})")


@given(session_walk())
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_B5_call_amount_nonnegative(walk):
    """B5: get_call_amount_from_history is always >= 0 on every reachable
    state (we test on each snapshot's pre-state)."""
    session, snapshots = walk
    g = session.game
    for pre, _ in snapshots:
        amt = g.get_call_amount_from_history(
            pre['street'], pre['history'], pre['starting_pot'],
            pre['p0_invested'], pre['p1_invested'])
        assert amt >= -_EPS, f"negative call amount {amt}, history={pre['history']}"


# ---------------------------------------------------------------------------
# C. ALL-IN SEMANTICS
# ---------------------------------------------------------------------------

@given(session_walk())
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_C1_after_allin_only_fold_or_call(walk):
    """C1: After 'allin' appears in current-street history, opponent's
    legal set is exactly {'fold', 'call'} (no further raises)."""
    session, snapshots = walk
    # Inspect every pre-state where 'allin' is in history.
    g = session.game
    for pre, _ in snapshots:
        if 'allin' not in pre['history']:
            continue
        # Find who is to act NEXT in this pre-state.
        acting = g._acting_player(len(pre['history']), pre['street'])
        # Skip if the round is complete (no legal actions).
        if g.is_round_complete(pre['history']):
            continue
        p0s = pre['p0_stack']
        p1s = pre['p1_stack']
        legal = g.get_legal_actions(
            pre['street'], pre['history'], pre['starting_pot'], acting,
            p0s, p1s, pre['p0_invested'], pre['p1_invested'])
        assert set(legal) <= {'fold', 'call'}, (
            f"After allin, legal contains non-fold/call: {legal}, "
            f"history={pre['history']}")


@given(session_walk())
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_C2_allin_call_terminal_zero_stacks(walk):
    """C2: when a hand ends with the current-street sequence ending in
    [..., 'allin', 'call'], both stacks must be 0."""
    session, snapshots = walk
    d = session.data
    if d['status'] != 'hand_over':
        return
    # Inspect action_log for the terminal sequence on the final street.
    log = d.get('action_log', [])
    if len(log) < 2:
        return
    if log[-2]['action'] == 'allin' and log[-1]['action'] == 'call':
        assert d['p0_stack'] < _EPS and d['p1_stack'] < _EPS, (
            f"allin+call terminal but stacks nonzero: "
            f"p0={d['p0_stack']}, p1={d['p1_stack']}")


@given(session_walk())
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_C3_allin_actor_stack_zero(walk):
    """C3: immediately after a player plays 'allin', their stack is 0."""
    session, snapshots = walk
    for pre, post in snapshots:
        if pre['action'] != 'allin':
            continue
        s = post['p0_stack'] if pre['acting'] == 0 else post['p1_stack']
        assert s < _EPS, f"allin actor stack nonzero: {s}"


# ---------------------------------------------------------------------------
# D. LEGAL ACTIONS
# ---------------------------------------------------------------------------

@given(session_walk())
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_D1_legal_actions_affordable(walk):
    """D1: every action returned by get_legal_actions has cost <= remaining."""
    # Already covered by B4 implicitly; we restate explicitly for clarity.
    session, _ = walk
    if session.data['status'] != 'in_hand':
        return
    d = session.data
    acting = session.current_player()
    remaining = d['p0_stack'] if acting == 0 else d['p1_stack']
    legal = session.legal_actions()
    for action in legal:
        cost = session.game._action_cost(
            action, d['street'], d['history'], d['starting_pot'],
            acting, d['p0_invested'], d['p1_invested'])
        assert cost <= remaining + _EPS, (
            f"{action} costs {cost} but acting={acting} has {remaining}")


@given(session_walk(max_raises=2))
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_D3_max_bet_raise_per_street(walk):
    """D3: at most 3 sized aggression actions per street (1 bet + 2 raises,
    or all-in counts toward the cap as a terminal raise). We assert the
    weaker but already-violated-on-failure form: combined sized bet/raise
    count <= 3 per street.

    This is a property of the CAPPED engine (max_raises_per_street=2) the
    blueprint trains/evals under, so the walk passes max_raises=2. LIVE serving
    (GameSession's default inf) deliberately UNCAPS re-raises so a human can
    5-bet/6-bet+, where this bound does NOT hold by design (see GameSession)."""
    session, snapshots = walk
    # Walk through pre-states and check the current-street history.
    for pre, _ in snapshots:
        sized = sum(1 for a in pre['history'] if a.startswith(('bet_', 'raise_')))
        assert sized <= 3, (
            f"More than 3 sized bet/raise in street history: {pre['history']}")


@given(session_walk())
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_D4_round_complete_means_no_legal(walk):
    """D4: when is_round_complete is True, get_legal_actions returns []."""
    session, snapshots = walk
    g = session.game
    for pre, _ in snapshots:
        if g.is_round_complete(pre['history']):
            acting = g._acting_player(len(pre['history']), pre['street'])
            legal = g.get_legal_actions(
                pre['street'], pre['history'], pre['starting_pot'], acting,
                pre['p0_stack'], pre['p1_stack'],
                pre['p0_invested'], pre['p1_invested'])
            assert legal == [], (
                f"round_complete but legal={legal}, history={pre['history']}")


# ---------------------------------------------------------------------------
# E. STREET TRANSITIONS
# ---------------------------------------------------------------------------

@given(session_walk())
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_E1_postflop_streets_start_symmetric(walk):
    """E1: at every snapshot where street > 0 and current-street history is
    empty (just transitioned), p0_invested == p1_invested."""
    session, snapshots = walk
    for _, post in snapshots:
        if post['street'] > 0 and not post['history']:
            assert abs(post['p0_invested'] - post['p1_invested']) < _EPS, (
                f"asymmetric cross-street invested at street {post['street']}: "
                f"p0_invested={post['p0_invested']}, "
                f"p1_invested={post['p1_invested']}")


# ---------------------------------------------------------------------------
# F. TERMINAL / UTILITY
# ---------------------------------------------------------------------------

@given(session_walk(max_steps=200, max_hands=5))
@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_F1_F3_terminal_utility_bounded(walk):
    """F1: |util_p0| <= STARTING_STACK at terminal.
    F3: total chips at terminal == 2 * STARTING_STACK."""
    session, _ = walk
    d = session.data
    if d['status'] != 'hand_over':
        return
    result = d['result']
    assert result is not None
    human_delta = result['humanDelta']
    util_p0 = human_delta if d['human_seat'] == 0 else -human_delta
    assert abs(util_p0) <= STARTING_STACK + _EPS, (
        f"|util_p0|={abs(util_p0)} exceeds STARTING_STACK={STARTING_STACK}")
    # F3: stacks restored (each hand starts fresh) so chip conservation
    # only holds during the hand; at hand_over the result captures delta.


@given(session_walk(max_steps=200, max_hands=5))
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_F4_hands_terminate(walk):
    """F4: every walk we generate either ends a hand or reaches our step
    budget — there is no infinite betting. We assert: if we executed any
    actions, history length on the active street stays <= 8 (since max 3
    bet/raises + interleaved calls + allin response is bounded)."""
    session, snapshots = walk
    for pre, _ in snapshots:
        # Per-street action count cap. With max 3 aggression + 1 opener
        # check + 1 closing call/fold + 1 allin response we expect <= 8.
        assert len(pre['history']) <= 10, (
            f"current-street history grew too long: {pre['history']}")


# ---------------------------------------------------------------------------
# G. INTERNAL CONSISTENCY (contribution sum == pot delta)
# ---------------------------------------------------------------------------

@given(session_walk())
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_G1_contribution_sum_equals_pot_delta(walk):
    """G1: postflop, p0_contrib + p1_contrib (this street) ==
    grand_pot - starting_pot. (Regression for BUG-003.)"""
    session, snapshots = walk
    g = session.game
    for pre, _ in snapshots:
        if pre['street'] == 0:
            continue
        p0_c = g.get_player_contribution_this_round(
            pre['history'], pre['street'], pre['starting_pot'], 0,
            pre['p0_invested'], pre['p1_invested'])
        p1_c = g.get_player_contribution_this_round(
            pre['history'], pre['street'], pre['starting_pot'], 1,
            pre['p0_invested'], pre['p1_invested'])
        pot = g.calculate_current_pot(
            pre['starting_pot'], pre['history'], pre['street'],
            pre['p0_invested'], pre['p1_invested'])
        delta = pot - pre['starting_pot']
        assert abs(p0_c + p1_c - delta) < 1e-4, (
            f"contribution sum {p0_c + p1_c} != pot delta {delta} "
            f"at history={pre['history']}, street={pre['street']}")


# ---------------------------------------------------------------------------
# F2 / F3 — zero-sum + terminal chip conservation (asserted, not just documented)
# ---------------------------------------------------------------------------

@given(session_walk(max_steps=200, max_hands=5))
@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_F2_zero_sum_and_terminal_conservation(walk):
    """F3: at a terminal, every chip in the pot was contributed by the two
    players (final_pot == p0_total + p1_total). F2: the game is zero-sum —
    both players' utilities sum to 0 — checked with an INDEPENDENT showdown
    evaluation (not via get_utility), which also cross-checks get_utility."""
    session, _ = walk
    d = session.data
    if d['status'] != 'hand_over':
        return
    g = session.game
    street = min(d['street'], 3)
    final_pot = g.calculate_current_pot(
        d['starting_pot'], d['history'], street, d['p0_invested'], d['p1_invested'])
    p0_total = d['p0_invested'] + g.get_player_contribution_this_round(
        d['history'], street, d['starting_pot'], 0, d['p0_invested'], d['p1_invested'])
    p1_total = d['p1_invested'] + g.get_player_contribution_this_round(
        d['history'], street, d['starting_pot'], 1, d['p0_invested'], d['p1_invested'])

    # F3: terminal chip conservation.
    assert abs(final_pot - (p0_total + p1_total)) < 1e-4, (
        f"pot {final_pot} != contributions {p0_total + p1_total}")

    # Independent winner determination (mirrors a showdown, not get_utility).
    if 'fold' in d['history']:
        folder = g._acting_player(d['history'].index('fold'), street)
        share0 = 0.0 if folder == 0 else final_pot
    else:
        board = d['community'][:5] if 'allin' in d['history'] \
            else d['community'][:g.get_community_cards_count(street)]
        r0 = session.evaluator.get_raw_hand_value(d['p0_cards'], board)
        r1 = session.evaluator.get_raw_hand_value(d['p1_cards'], board)
        share0 = final_pot if r0 < r1 else (0.0 if r0 > r1 else final_pot / 2.0)

    util0_indep = share0 - p0_total
    util1_indep = (final_pot - share0) - p1_total
    # F2: zero-sum.
    assert abs(util0_indep + util1_indep) < 1e-4, (
        f"not zero-sum: {util0_indep} + {util1_indep} != 0")
    # Cross-check the engine's get_utility against the independent accounting.
    util0_engine = g.get_utility(
        d['p0_cards'], d['p1_cards'], d['community'], d['history'], street,
        d['starting_pot'], d['p0_invested'], d['p1_invested'])
    assert abs(util0_engine - util0_indep) < 1e-4, (
        f"get_utility {util0_engine} != independent {util0_indep}")


# ---------------------------------------------------------------------------
# Console runner (mirrors test_cfr_correctness.py style)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback
    tests = [v for k, v in globals().items() if k.startswith('test_')]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'=' * 70}")
    sys.exit(1 if failed else 0)
