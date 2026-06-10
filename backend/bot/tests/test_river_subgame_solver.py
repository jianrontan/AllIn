# backend/bot/tests/test_river_subgame_solver.py
"""
Validate the RiverSubgameSolver assembly (step 6a): the end-to-end
ranges->tree->solve->read-off path produces a valid action for the bot's hand at
its actual decision node (both seats), and decide() falls back to the blueprint
off-river / when solver inputs are absent.

Runs against the active blueprint DB (read-only); skips cleanly if none exists.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_subgame_solver import RiverSubgameSolver, blueprint_to_tree_dist
from src.subgame.river_tree import build_river_tree
from src.game.range_tracker import RangeTracker
from src.abstractions.card_abstractions import CardAbstraction

_CARDS = CardAbstraction()
_BOARD = ['CQ', 'SJ', 'H9', 'D5', 'C2']
_HOLE = ['HA', 'DK']


def _blueprint_db():
    try:
        from src.config import resolve_blueprint_path
        from src.storage.blueprint_db import BlueprintDB
        path = resolve_blueprint_path()
        return BlueprintDB(path, read_only=True)
    except Exception as e:
        print(f"  (no blueprint DB available: {e})")
        return None


def _trackers():
    villain = RangeTracker(_HOLE, _CARDS)   # opponent: bot's cards removed
    villain.reveal(_BOARD)
    hero = RangeTracker((), _CARDS)         # bot's range: spans all hands
    hero.reveal(_BOARD)
    return villain, hero


def _solver(db):
    # Small iteration budget keeps the test fast; correctness, not convergence.
    return RiverSubgameSolver(db, max_iters=80, check_every=40, time_budget=30.0)


def test_solve_for_action_oop():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_solve_for_action_oop (no blueprint)")
        return
    villain, hero = _trackers()
    solver = _solver(db)
    # Bot is OOP (seat 1): with an empty river path, the root IS the bot's node.
    dist, node, info = solver.solve_for_action(
        board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=1, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0, river_path=[])
    db.close()
    assert node.player == 1
    assert set(dist.keys()) == set(node.actions)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert all(p >= -1e-12 for p in dist.values())
    print(f"PASS test_solve_for_action_oop (iters={info['iters']}, gap={info['gap']:.3f}, "
          f"dist={ {k: round(v,2) for k,v in dist.items()} })")


def test_solve_for_action_ip_after_villain_checks():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_solve_for_action_ip_after_villain_checks (no blueprint)")
        return
    villain, hero = _trackers()
    solver = _solver(db)
    # Bot is IP (seat 0): its decision is AFTER OOP acts. Path = OOP checks.
    dist, node, info = solver.solve_for_action(
        board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=0, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0,
        river_path=['check'])
    db.close()
    assert node.player == 0, "after OOP checks it's the IP (seat-0) bot's turn"
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    # The bot's hand should have a real (non-uniform) strategy at its own node.
    n = len(dist)
    assert not np.allclose(list(dist.values()), 1.0 / n), dist
    print(f"PASS test_solve_for_action_ip_after_villain_checks "
          f"(node {node.node_id}, dist={ {k: round(v,2) for k,v in dist.items()} })")


def test_navigate_injects_offmenu_bet():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_navigate_injects_offmenu_bet (no blueprint)")
        return
    villain, hero = _trackers()
    solver = _solver(db)
    # Villain (OOP) makes an off-menu bet of 13 chips; the bot (IP) faces it. Nested
    # solving injects 13 as a REAL tree edge, so the bot's node faces the EXACT pot
    # (not a snapped 12/18): to_call must be exactly 13, pot_mid = 24 + 13 = 37.
    dist, node, info = solver.solve_for_action(
        board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=0, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0,
        river_path=[('bet', 13.0)])
    db.close()
    assert node.player == 0, "bot faces the injected bet"
    assert abs(node.to_call - 13.0) < 1e-9, f"exact off-grid pot, got to_call={node.to_call}"
    assert abs(node.pot_mid - 37.0) < 1e-9, node.pot_mid
    assert 'fold' in node.actions and 'call' in node.actions
    print(f"PASS test_navigate_injects_offmenu_bet (exact to_call={node.to_call:.1f}, "
          f"pot_mid={node.pot_mid:.1f})")


def test_decide_falls_back_off_river():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_decide_falls_back_off_river (no blueprint)")
        return
    solver = _solver(db)
    legal = ['check', 'bet_small', 'bet_medium', 'bet_large']
    # Flop -> must delegate to the blueprint (returns a legal action, no solve).
    a = solver.decide('pf_9_5_ip_flop_', legal, {'street': 'flop'})
    assert a in legal
    # River but missing the solver inputs -> also falls back.
    a2 = solver.decide('pf_9_5_ip_river_', legal, {'street': 'river', 'community': _BOARD})
    assert a2 in legal
    db.close()
    print("PASS test_decide_falls_back_off_river")


def test_blueprint_to_tree_dist_mapping():
    """Blueprint actions redistribute onto the tree menu by nearest size fraction;
    no blueprint DB needed."""
    tree = build_river_tree(pot_entry=24.0, stacks=(88.0, 88.0))
    root = tree.root                                   # OOP, to_call=0: check + bets + allin
    # tree root bets: 0.5/0.75/1.0/1.5 * 24 = 12/18/24/36
    bp = {'check': 0.5, 'bet_small': 0.3, 'bet_large': 0.2}   # fracs 0.33 / 1.0
    out = blueprint_to_tree_dist(bp, root)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert abs(out['check'] - 0.5) < 1e-9
    assert abs(out['bet:12'] - 0.3) < 1e-9             # 0.33 -> nearest 0.5 -> bet:12
    assert abs(out['bet:24'] - 0.2) < 1e-9             # 1.0 -> bet:24

    # A facing-bet node: fold/call direct, raise mapped to nearest raise edge.
    ip = root.children[root.actions.index('bet:24')]   # IP faces a pot bet
    assert ip.to_call > 0
    bp2 = {'fold': 0.4, 'call': 0.4, 'raise_medium': 0.2}
    out2 = blueprint_to_tree_dist(bp2, ip)
    assert abs(out2['fold'] - 0.4) < 1e-9 and abs(out2['call'] - 0.4) < 1e-9
    raise_mass = sum(v for a, v in out2.items() if a.startswith('raise:') or a == 'allin')
    assert abs(raise_mass - 0.2) < 1e-9
    print("PASS test_blueprint_to_tree_dist_mapping")


def test_blueprint_to_tree_dist_reraise_node():
    """Re-raise node where the dropped-sc offset STRADDLES a nearest-neighbour
    boundary, so the fixed and buggy formulas give DIFFERENT answers (the prior
    test was vacuous -- both formulas agreed on it). Node: sc=(54,34), actor
    (player 1) has 34 in, faces 20, pot 80 -> offset delta = 34/100 = 0.34.

      raise edge:   raise:36  raise:61  raise:86  raise:136
      fixed frac :   0.50      0.75      1.00      1.50
      buggy frac :   0.16      0.41      0.66      1.16    (= fixed - 0.34)

    A blueprint raise_medium (frac 0.66) maps to raise:61 (nearest 0.75) under the
    FIX, but to raise:86 (exact 0.66) under the BUG -- so this discriminates."""
    import types
    node = types.SimpleNamespace(
        player=1, sc=(54.0, 34.0), pot_mid=80.0, to_call=20.0,
        actions=['fold', 'call', 'raise:36', 'raise:61', 'raise:86', 'raise:136', 'allin'])
    out = blueprint_to_tree_dist({'raise_medium': 1.0}, node)
    assert abs(out['raise:61'] - 1.0) < 1e-9, ("fix->raise:61, bug->raise:86", out)
    print("PASS test_blueprint_to_tree_dist_reraise_node (discriminates fix vs bug)")


def test_hero_zero_reach_falls_back():
    """A bot hand with ~zero hero reach must raise (so decide() falls back) rather
    than silently read off a uniform strategy."""
    db = _blueprint_db()
    if db is None:
        print("SKIP test_hero_zero_reach_falls_back (no blueprint)")
        return
    villain, hero = _trackers()
    hi = next(i for i, h in enumerate(hero.hands) if set(h) == set(_HOLE))
    hero.w[hi] = 0.0                                   # bot's actual hand: zero reach
    solver = _solver(db)
    try:
        solver.solve_for_action(
            board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=1, hole=_HOLE,
            villain_tracker=villain, hero_tracker=hero, confidence=1.0, river_path=[])
        raise AssertionError("expected zero hero reach to raise")
    except ValueError as e:
        assert 'reach' in str(e), e
    db.close()
    print("PASS test_hero_zero_reach_falls_back")


def test_allin_emits_shove_not_check_at_deep_stack():
    """At a deep-stack node the engine offers no discrete 'allin' (sized bets are
    affordable), but the solver may still choose to shove. The mapping must emit a
    full-stack custom shove, NOT silently fall through to 'check'."""
    import types
    solver = RiverSubgameSolver(None)        # no DB needed for action mapping
    legal = ['check', 'bet_small', 'bet_medium', 'bet_large']   # NB: no 'allin'
    node = types.SimpleNamespace(to_call=0.0, sc=(0.0, 0.0))
    spec = {'bot_seat': 1, 'stacks': (88.0, 88.0)}
    out = solver._pick_engine_action({'check': 0.0, 'allin': 1.0}, legal, spec, node)
    assert out != 'check', "a chosen shove must not degrade to check"
    assert out.startswith('bet_custom_'), out          # no bet to call -> bet shove
    assert abs(float(out.rsplit('_', 1)[1]) - 88.0) < 1e-9, out   # full stack
    # Facing a bet -> raise shove.
    node2 = types.SimpleNamespace(to_call=20.0, sc=(40.0, 10.0))
    out2 = solver._pick_engine_action(
        {'allin': 1.0}, ['fold', 'call', 'raise_small'], spec, node2)
    assert out2.startswith('raise_custom_'), out2
    print("PASS test_allin_emits_shove_not_check_at_deep_stack")


def test_decide_river_emits_action_with_ev_gate():
    """The full decide() path on a river state: solve + EV gate + emit. The action
    is a valid engine action -- check/call/fold/allin, an abstract size, or an
    exact custom_ size (the solver's edge)."""
    db = _blueprint_db()
    if db is None:
        print("SKIP test_decide_river_emits_action_with_ev_gate (no blueprint)")
        return
    villain, hero = _trackers()
    solver = _solver(db)
    legal = ['check', 'bet_small', 'bet_medium', 'bet_large', 'allin']
    ps = {
        'street': 'river', 'community': _BOARD, 'hole_cards': _HOLE,
        'riverEntryPot': 24.0, 'riverEntryStacks': (88.0, 88.0), 'botSeat': 1,
        'opp_range': villain, 'hero_range': hero, 'riverPath': [],
    }
    a = solver.decide('pf_9_5_oop_river_', legal, ps)
    db.close()
    ok = (a in legal) or a.startswith('bet_custom_') or a.startswith('raise_custom_')
    assert ok, f"unexpected action: {a}"
    print(f"PASS test_decide_river_emits_action_with_ev_gate (action={a})")


def test_solver_gated_off_on_high_spr_small_pot():
    """A high-SPR river (small pot, deep stacks) builds a huge tree that blows the live
    solve budget (~20s, unconverged), so the solver must SKIP it -> blueprint. Only
    low-SPR (meaningful-pot) spots are solved, where the tree is small + fast. The
    all-in guard runs earlier in decide(), so jams stay covered regardless."""
    from src.subgame.river_subgame_solver import RiverSubgameSolver, SOLVER_MAX_SPR
    solver = RiverSubgameSolver.__new__(RiverSubgameSolver)  # _solver_inputs is pure(ps)
    base = dict(street='river', community=_BOARD, botSeat=1, hole_cards=_HOLE,
                opp_range=object(), hero_range=object(), riverPath=[])
    # high SPR (pot 6, stacks 197 -> SPR ~33) -> skip
    assert solver._solver_inputs(
        {**base, 'riverEntryPot': 6.0, 'riverEntryStacks': (197.0, 197.0)}) is None
    # also a borderline just above the threshold -> skip
    assert solver._solver_inputs(
        {**base, 'riverEntryPot': 10.0,
         'riverEntryStacks': (10.0 * (SOLVER_MAX_SPR + 1), ) * 2}) is None
    # low SPR (pot 80, stacks 40 -> SPR 0.5) -> solve
    spec = solver._solver_inputs(
        {**base, 'riverEntryPot': 80.0, 'riverEntryStacks': (40.0, 40.0)})
    assert spec is not None and spec['pot_entry'] == 80.0
    print(f"PASS test_solver_gated_off_on_high_spr_small_pot (SOLVER_MAX_SPR={SOLVER_MAX_SPR})")


def test_solve_deep_reraise_war():
    """#1 fix: a river re-raise war BEYOND the blueprint's 3-aggression cap (now legal
    via uncapped live re-raises) must SOLVE, not abort to the blueprint. The live solver
    builds a depth-`LIVE_RIVER_MAX_AGGRESSIONS` tree, so a 4-aggression river path lands
    on a real decision node instead of raising 'river path did not land on a decision
    node'. Deep stacks + escalating off-menu raises keep money behind so the 4th
    aggression is a genuine (non-clamped) node."""
    from src.subgame.river_subgame_solver import LIVE_RIVER_MAX_AGGRESSIONS
    assert LIVE_RIVER_MAX_AGGRESSIONS >= 4, LIVE_RIVER_MAX_AGGRESSIONS
    db = _blueprint_db()
    if db is None:
        print("SKIP test_solve_deep_reraise_war (no blueprint)")
        return
    villain, hero = _trackers()
    solver = _solver(db)
    # bet(OOP,agg1), raise(IP,agg2), raise(OOP,agg3), raise(IP,agg4) -> OOP (seat 1)
    # bot faces the 4th aggression. Off-menu chips -> injected as real tree edges.
    path = [('bet', 8.0), ('raise', 16.0), ('raise', 32.0), ('raise', 64.0)]
    dist, node, info = solver.solve_for_action(
        board=_BOARD, pot_entry=24.0, stacks=(400.0, 400.0), bot_seat=1, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0, river_path=path)
    db.close()
    assert node.agg >= 4, f"expected a >=4-aggression node (depth fix), got agg={node.agg}"
    assert node.player == 1, node.player
    assert set(dist.keys()) == set(node.actions)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    print(f"PASS test_solve_deep_reraise_war (node agg={node.agg}, "
          f"dist={ {k: round(v,2) for k,v in dist.items()} })")


def test_safe_gadget_solve_for_action():
    """Phase 5a: solve_for_action with safe_gadget=True runs the re-solving gadget
    end-to-end (blueprint opt-out CFVs computed + gadget solve + read-off) and returns
    a valid action distribution for the bot's hand. Same spot as the unsafe path; this
    just proves the safe branch produces a well-formed strategy on the real blueprint."""
    db = _blueprint_db()
    if db is None:
        print("SKIP test_safe_gadget_solve_for_action (no blueprint)")
        return
    villain, hero = _trackers()
    solver = RiverSubgameSolver(db, max_iters=80, check_every=40, time_budget=30.0,
                                safe_gadget=True)
    dist, node, info = solver.solve_for_action(
        board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=1, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0, river_path=[])
    db.close()
    assert node.player == 1
    assert set(dist.keys()) == set(node.actions)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert all(p >= -1e-12 for p in dist.values())
    assert info['gap'] is None, "gadget solve reports no Nash gap (range-reshaped)"
    print(f"PASS test_safe_gadget_solve_for_action (iters={info['iters']}, "
          f"converged={info['converged']}, dist={ {k: round(v,2) for k,v in dist.items()} })")


def test_auto_anchor_branches_wiring():
    """Phase 5a 'auto'/'confidence' wiring (review gap H1): solve_for_action must return
    the right info['anchor'] for EVERY branch -- including the CLAMP (auto_safe_fallback),
    the safety point of the feature, which the maniac never triggered live. The self-check
    outcome is forced deterministically via _AUTO_SAFE_MARGIN (+/- huge) so the branch
    WIRING (correct anchor label, the second gadget solve on clamp) is exercised without
    depending on a particular belief over-exploiting. A uniform belief is auto-untrusted
    (uninformative); a concentrated one is trusted."""
    import numpy as np
    db = _blueprint_db()
    if db is None:
        print("SKIP test_auto_anchor_branches_wiring (no blueprint)")
        return

    def anchor_for(gadget_anchor, margin, *, trusted):
        villain, hero = _trackers()
        if trusted:
            live = np.where(np.asarray(villain.w) > 0)[0]
            villain.w[:] = 0.0
            villain.w[live[0]] = 1.0            # dominant hand -> informative
            villain.w[live[1:20]] = 0.01        # tiny mass -> non-degenerate range
            villain.confidence = 1.0
        else:
            villain.confidence = 0.0            # untrusted (also uniform = uninformative)
        s = RiverSubgameSolver(db, max_iters=60, check_every=30, time_budget=30.0,
                               safe_gadget=True, gadget_anchor=gadget_anchor)
        if margin is not None:
            s._AUTO_SAFE_MARGIN = margin
        _, _, info = s.solve_for_action(
            board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=1, hole=_HOLE,
            villain_tracker=villain, hero_tracker=hero, confidence=villain.confidence,
            river_path=[])
        return info['anchor']

    assert anchor_for('auto', None, trusted=True) == 'auto_exploit_trusted'
    assert anchor_for('auto', 1e9, trusted=False) == 'auto_exploit_safe'
    assert anchor_for('auto', -1e9, trusted=False) == 'auto_safe_fallback'   # the CLAMP
    assert anchor_for('confidence', None, trusted=True) == 'confidence_exploit'
    assert anchor_for('confidence', None, trusted=False) == 'confidence_safe'
    db.close()
    print("PASS test_auto_anchor_branches_wiring (all 5 anchor branches reached)")


def test_safe_gadget_auto_records_decision():
    """Phase 5a 'auto' anchor: the solver runs the unsafe solve, then either exploits
    (confidence pre-filter or self-check passes) or falls back to the blueprint-anchored
    gadget -- and records which via info['anchor']. Just proves the auto path runs end-to-
    end on the real blueprint and tags a decision (the exploitability guarantee is covered
    by tests/test_safe_river_gadget.py)."""
    db = _blueprint_db()
    if db is None:
        print("SKIP test_safe_gadget_auto_records_decision (no blueprint)")
        return
    villain, hero = _trackers()
    solver = RiverSubgameSolver(db, max_iters=80, check_every=40, time_budget=30.0,
                                safe_gadget=True, gadget_anchor='auto')
    dist, node, info = solver.solve_for_action(
        board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=1, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=0.0, river_path=[])
    db.close()
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert info.get('anchor') in (
        'auto_exploit_trusted', 'auto_exploit_safe', 'auto_safe_fallback'), info.get('anchor')
    print(f"PASS test_safe_gadget_auto_records_decision (anchor={info['anchor']}, "
          f"selfCheck={info.get('autoSelfCheck')})")


TESTS = [
    test_solver_gated_off_on_high_spr_small_pot,
    test_safe_gadget_solve_for_action,
    test_auto_anchor_branches_wiring,
    test_safe_gadget_auto_records_decision,
    test_solve_deep_reraise_war,
    test_blueprint_to_tree_dist_mapping,
    test_blueprint_to_tree_dist_reraise_node,
    test_hero_zero_reach_falls_back,
    test_allin_emits_shove_not_check_at_deep_stack,
    test_decide_river_emits_action_with_ev_gate,
    test_solve_for_action_oop,
    test_solve_for_action_ip_after_villain_checks,
    test_navigate_injects_offmenu_bet,
    test_decide_falls_back_off_river,
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
