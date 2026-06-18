# backend/bot/tests/test_safe_river_gadget.py
"""
Phase 5a piece 4 -- the exploitability validation gate for the safe-river gadget.

The safety claim is directly testable: on a battery of river spots, the villain's
best-response value against the HERO strategy (measured over the villain's TRUE /
full range) must be no greater for the gadget-solved hero than for the blueprint
hero. We measure four hero strategies per spot:

  * blueprint  -- the blueprint's river play projected onto the tree (the baseline).
  * unsafe-v1  -- solve_river directly against the input BELIEF (today's served path).
  * gadget(belief)    -- the re-solving gadget anchored to the tracked belief.
  * gadget(blueprint) -- the gadget anchored to a UNIFORM card-removal villain range
                         (the provable anchor: safe vs ANY villain hand).

Two belief scenarios per spot:
  * CORRECT  -- belief == the true (uniform) range. The honest case.
  * WRONG    -- a narrow, confidently-WRONG belief (villain "has only the nuts").
                This is the danger the gadget exists to fix: the unsafe solve best-
                responds to a phantom and can become MORE exploitable than the
                blueprint; the blueprint-anchored gadget must clamp to <= blueprint.

HARD ASSERT (the ship gate): gadget(blueprint) <= blueprint on EVERY spot/scenario.
The rest is reported as a measurement table (the data the anchor decision rests on).

Run: python tests/test_safe_river_gadget.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_tree import build_river_tree
from src.subgame.river_cfr import RiverCFR
from src.subgame.solve_control import solve_river
from src.subgame.blueprint_projection import (
    blueprint_cfv, blueprint_strategy_on_tree)
from src.evaluation.showdown_kernel import build_board_arrays, compatible_mass
from src.subgame.range_inputs import hand_index_map, project_tracker
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction
from src.game.range_tracker import RangeTracker

_EVAL = HandEvaluator()
_CARDS = CardAbstraction()

# A small battery of river spots (board, bot hole, bot seat, entry pot, behind stacks).
_SPOTS = [
    (['HA', 'DK', 'CQ', 'SJ', 'H9'], ['HK', 'DQ'], 1, 30.0, 70.0),   # two pair-ish OOP
    (['CQ', 'SJ', 'H9', 'D5', 'C2'], ['HA', 'DK'], 1, 24.0, 60.0),   # ace-high OOP
    (['S8', 'H7', 'D4', 'CK', 'SA'], ['HQ', 'DQ'], 0, 40.0, 50.0),   # mid pair IP
]

_ITERS = 300


def _blueprint_raw():
    """raw_strategy(key) from the active blueprint DB, or a uniform fallback (None)."""
    try:
        from src.config import resolve_blueprint_path
        from src.storage.blueprint_db import BlueprintDB
        db = BlueprintDB(resolve_blueprint_path(), read_only=True)
        return db.get_average_strategy, db
    except Exception as e:
        print(f"  (no blueprint DB: {e}; using uniform-blueprint baseline)")
        return (lambda k: None), None


def _uniform_reach(board, hole, ba, idx):
    """Uniform card-removal range over a player's legal hands (the 'true range'/blueprint
    anchor). `hole` removed too when it's the villain (the bot's known cards)."""
    t = RangeTracker(tuple(hole) if hole else (), _CARDS)
    t.reveal(list(board))
    return project_tracker(t, ba, idx)


def _narrow_wrong(board, hole, ba, idx):
    """A sharp, WRONG villain belief: all mass on the single strongest combo by board
    pairing (a phantom 'nuts'). Card-removal correct, but concentrated on hands the
    villain almost surely does NOT uniquely hold -- the narrow-wrong danger case."""
    proj = _uniform_reach(board, hole, ba, idx)
    w = np.zeros_like(proj)
    live = np.where(proj > 0)[0]
    # pick a deterministic 'wrong' spike: the last live index (arbitrary but fixed).
    w[live[-1]] = 1.0
    return w


def _villain_br(meas, tree, ba, villain_seat, hero_reach, villain_true, hero_fn):
    """Villain best-response value vs `hero_fn` over the villain's TRUE range, per dealt
    matchup (chips). meas: any RiverCFR on this tree/ba (used only for _br_value)."""
    br = meas._br_value(tree.root, villain_seat, np.asarray(hero_reach, float), hero_fn)
    Z = float((np.asarray(villain_true, float)
               * compatible_mass(ba, np.asarray(hero_reach, float))).sum())
    if Z <= 0:
        return 0.0
    return float((np.asarray(villain_true, float) * br).sum()) / Z


def _solve_unsafe(tree, ba, hero_reach, belief, bot_seat):
    reach0, reach1 = (hero_reach, belief) if bot_seat == 0 else (belief, hero_reach)
    cfr, _ = solve_river(tree, ba, reach0, reach1, max_iters=_ITERS, check_every=_ITERS)
    return cfr.average_strategy


def _solve_gadget(tree, ba, raw, hero_reach, anchor_villain, villain_seat, menu):
    g0, g1 = (hero_reach, anchor_villain) if villain_seat == 1 else (anchor_villain, hero_reach)
    optout = blueprint_cfv(tree, ba, raw, g0, g1, villain_seat, menu)
    cfr = RiverCFR(tree, ba)
    cfr.run_gadget(hero_reach, anchor_villain, optout, villain_seat, iters=_ITERS)
    return cfr.average_strategy


def run():
    raw, db = _blueprint_raw()
    menu = None
    if db is not None:
        from src.abstractions.sizing import db_menu_mode, postflop_menu_for
        menu = postflop_menu_for(db_menu_mode(db))

    rows = []
    worst_violation = 0.0
    for board, hole, bot_seat, pot, behind in _SPOTS:
        ba = build_board_arrays(board, _EVAL, _CARDS)
        idx = hand_index_map(ba)
        tree = build_river_tree(pot, (behind, behind))
        villain_seat = 1 - bot_seat
        meas = RiverCFR(tree, ba)               # for _br_value only

        hero_reach = _uniform_reach(board, [], ba, idx)         # bot's range (true)
        villain_true = _uniform_reach(board, hole, ba, idx)     # villain's TRUE range
        bp_strat = blueprint_strategy_on_tree(tree, ba, raw, menu)
        bp_fn = lambda nid, _b=bp_strat: _b[nid]
        bp_expl = _villain_br(meas, tree, ba, villain_seat, hero_reach, villain_true, bp_fn)

        for scenario, belief in (('correct', villain_true),
                                 ('wrong', _narrow_wrong(board, hole, ba, idx))):
            unsafe_fn = _solve_unsafe(tree, ba, hero_reach, belief, bot_seat)
            gbel_fn = _solve_gadget(tree, ba, raw, hero_reach, belief, villain_seat, menu)
            gbp_fn = _solve_gadget(tree, ba, raw, hero_reach, villain_true, villain_seat, menu)

            e_unsafe = _villain_br(meas, tree, ba, villain_seat, hero_reach, villain_true, unsafe_fn)
            e_gbel = _villain_br(meas, tree, ba, villain_seat, hero_reach, villain_true, gbel_fn)
            e_gbp = _villain_br(meas, tree, ba, villain_seat, hero_reach, villain_true, gbp_fn)
            # 'auto' self-check (RiverSubgameSolver gadget_anchor='auto', no pre-filter):
            # measure the unsafe strategy vs the blueprint over the same UNIFORM range the
            # live check uses, exploit iff within the blueprint BR + margin, else gad(bp).
            from src.subgame.river_subgame_solver import RiverSubgameSolver as _RSS
            uni = villain_true                                  # the uniform card-removal range
            chk_u = float((uni * meas._br_value(tree.root, villain_seat, hero_reach, unsafe_fn)).sum())
            chk_b = float((uni * meas._br_value(tree.root, villain_seat, hero_reach, bp_fn)).sum())
            exploit = chk_u <= chk_b + _RSS._AUTO_SAFE_MARGIN * (abs(chk_b) + 1.0)
            e_auto = e_unsafe if exploit else e_gbp

            rows.append((board, scenario, bp_expl, e_unsafe, e_gbel, e_gbp, e_auto))
            # HARD GATE: the blueprint-anchored gadget AND the auto policy are both
            # no-more-exploitable than the blueprint (small slack for finite iters + float).
            slack = 1e-3 * (abs(bp_expl) + 1.0)
            worst_violation = max(worst_violation, e_gbp - bp_expl - slack,
                                  e_auto - bp_expl - slack)

    if db is not None:
        db.close()

    print("\nVillain best-response value vs hero (chips/dealt matchup; lower = safer)\n")
    print(f"  {'board':<20} {'scen':<8} {'blueprint':>10} {'unsafe-v1':>10} "
          f"{'gad(blf)':>10} {'gad(bp)':>10} {'auto':>10}")
    for board, scen, bp, us, gb, gp, au in rows:
        bstr = ''.join(board)
        print(f"  {bstr:<20} {scen:<8} {bp:>10.3f} {us:>10.3f} {gb:>10.3f} "
              f"{gp:>10.3f} {au:>10.3f}")
    print()

    ok = worst_violation <= 0.0
    print(f"  HARD GATE (gadget(blueprint) AND auto <= blueprint on every spot): "
          f"{'PASS' if ok else 'FAIL'} (worst excess={worst_violation:+.4f})")
    # Informational: did the unsafe solve over-exploit a WRONG belief past the blueprint?
    wrong = [(us, bp) for (_, scen, bp, us, _gb, _gp, _au) in rows if scen == 'wrong']
    over = sum(1 for us, bp in wrong if us > bp + 1e-6)
    print(f"  INFO: unsafe-v1 more exploitable than blueprint in {over}/{len(wrong)} "
          f"WRONG-belief spots (the leak the gadget removes)")
    # Informational: on CORRECT-belief spots, auto should recover the unsafe exploitation.
    corr = [(au, us) for (_, scen, _bp, us, _gb, _gp, au) in rows if scen == 'correct']
    kept = sum(1 for au, us in corr if abs(au - us) <= 1e-6)
    print(f"  INFO: auto kept unsafe-v1 exploitation in {kept}/{len(corr)} "
          f"CORRECT-belief spots (exploits when safe)")
    return ok


def test_gadget_blueprint_anchor_no_more_exploitable():
    """The ship gate as a regression assert: gadget(blueprint) <= blueprint everywhere."""
    assert run(), "gadget(blueprint) exceeded blueprint exploitability on some spot"
    print("PASS test_gadget_blueprint_anchor_no_more_exploitable")


def test_gadget_increment_equivalence():
    """run_gadget split into check_every-sized chunks MUST equal a single-shot solve.

    run_gadget is fully deterministic (vectorized, no sampling), so a chunked solve
    can only differ from one continuous solve if some per-iteration state fails to
    persist across calls. The villain gadget regret (g_regret) was exactly such a
    bug: a function-local reset every chunk, so the served check_every-chunked path
    (solve_river_gadget runs 10x40 by default) restarted the villain opt-out belief
    from uniform every chunk while the hero strat_sum accumulated -- the gadget never
    converged and the no-more-exploitable-than-blueprint guarantee was silently lost.
    Every prior safety test ran run_gadget single-shot, so none caught it. This asserts
    the increment-equivalence that self.regret/self._iter were always documented to hold."""
    raw, db = _blueprint_raw()
    menu = None
    if db is not None:
        from src.abstractions.sizing import db_menu_mode, postflop_menu_for
        menu = postflop_menu_for(db_menu_mode(db))
    board, hole, bot_seat, pot, behind = _SPOTS[0]
    ba = build_board_arrays(board, _EVAL, _CARDS)
    idx = hand_index_map(ba)
    tree = build_river_tree(pot, (behind, behind))
    villain_seat = 1 - bot_seat
    hero_reach = _uniform_reach(board, [], ba, idx)
    villain_true = _uniform_reach(board, hole, ba, idx)
    g0, g1 = (hero_reach, villain_true) if villain_seat == 1 else (villain_true, hero_reach)
    optout = blueprint_cfv(tree, ba, raw, g0, g1, villain_seat, menu)

    one = RiverCFR(tree, ba)
    one.run_gadget(hero_reach, villain_true, optout, villain_seat, iters=300)
    chunked = RiverCFR(tree, ba)                 # mirror the served chunked path
    for _ in range(10):
        chunked.run_gadget(hero_reach, villain_true, optout, villain_seat, iters=30)
    if db is not None:
        db.close()

    worst = max(float(np.abs(one.average_strategy(nid) - chunked.average_strategy(nid)).max())
                for nid in range(len(tree.decision_nodes)))
    print(f"  increment-equivalence: max avg-strategy diff (10x30 vs 1x300) = {worst:.2e}")
    assert worst < 1e-9, (f"chunked run_gadget != single-shot ({worst:.2e}) -- the villain "
                          "gadget regret is not persisting across increments")
    print("PASS test_gadget_increment_equivalence")


TESTS = [test_gadget_blueprint_anchor_no_more_exploitable,
         test_gadget_increment_equivalence]

if __name__ == '__main__':
    sys.exit(0 if run() else 1)
