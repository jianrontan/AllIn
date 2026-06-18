# backend/bot/tests/test_safe_turn_gadget.py
"""
Phase 5b Component #1 -- the safety gate for the TURN safe-solving gadget
(solve_control.solve_turn_gadget + blueprint_projection.blueprint_cfv_turn).

The turn analogue of test_safe_river_gadget. The safety claim is the same shape, but
on the DEPTH-LIMITED turn tree: the villain best-responds in TURN actions only (the
river is the FROZEN blueprint continuation in the leaf matrices, identical for the
blueprint hero and the gadget hero). So this gate proves the gadget is no-more-
exploitable than the blueprint *within the depth-limited game* -- which is exactly what
Component #1 is for (kill the off-model −170 regression). Full safety also needs an
accurate leaf (Component #2, nested river solve) -- NOT tested here.

HARD ASSERT: on a battery of turn spots, the villain's BR value vs the gadget(blueprint-
anchor) hero <= vs the blueprint hero, over the villain's TRUE (uniform card-removal)
range. Slack for finite iters + float.

Run: python tests/test_safe_turn_gadget.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.turn_tree import build_turn_tree
from src.subgame.turn_cfr import TurnCFR
from src.subgame.solve_control import solve_turn_gadget
from src.subgame.blueprint_projection import (
    blueprint_cfv_turn, blueprint_turn_strategy_on_tree)
from src.subgame.cfv import (turn_strength, equal_freq_partition,
                             turn_leaf_matrix_both, FULL_DECK)
from src.evaluation.showdown_kernel import build_turn_board_arrays, compatible_mass
from src.subgame.range_inputs import hand_index_map, project_tracker
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction
from src.game.range_tracker import RangeTracker

_EVAL = HandEvaluator()
_CARDS = CardAbstraction()

# 4-card turn spots: (board4, bot hole, bot seat, entry pot, behind stacks).
_SPOTS = [
    (['CQ', 'SJ', 'H9', 'D5'], ['HK', 'DQ'], 1, 20.0, 60.0),   # OOP
    (['HA', 'DK', 'CQ', 'SJ'], ['HQ', 'CJ'], 0, 24.0, 55.0),   # IP
]
_N_BUCKETS = 12
_RIVER_SAMPLE = 12
_ITERS = 300


def _raw_blueprint():
    """(raw_strategy, db, menu). Prefer a 30/24 snapshot matching the local abstraction;
    else the active blueprint; else uniform fallback (the safety math holds for ANY
    blueprint, incl. uniform -- like test_safe_river_gadget)."""
    from src.storage.blueprint_db import BlueprintDB
    cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'analysis', 'blueprints', 'snapshots', 'snap_52500000.db')
    try:
        if not os.path.exists(cand):
            from src.config import resolve_blueprint_path
            cand = resolve_blueprint_path()
        db = BlueprintDB(cand, read_only=True)
        from src.abstractions.sizing import db_menu_mode, postflop_menu_for
        menu = postflop_menu_for(db_menu_mode(db))
        return db.get_average_strategy, db, menu
    except Exception as e:
        print(f"  (no blueprint DB: {e}; uniform-blueprint baseline)")
        return (lambda k: None), None, None


def _uniform(board, hole, ba, idx):
    t = RangeTracker(tuple(hole) if hole else (), _CARDS)
    t.reveal(list(board))
    return project_tracker(t, ba, idx)


def _villain_br(meas, tree, ba, villain_seat, hero_reach, villain_true, hero_fn):
    """Villain BR value vs hero_fn over the villain's TRUE range (chips/dealt matchup).
    In-model-river: TurnCFR._br_value best-responds in TURN actions, frozen blueprint leaf."""
    br = meas._br_value(tree.root, villain_seat, np.asarray(hero_reach, float), hero_fn)
    Z = float((np.asarray(villain_true, float)
               * compatible_mass(ba, np.asarray(hero_reach, float))).sum())
    return 0.0 if Z <= 0 else float((np.asarray(villain_true, float) * br).sum()) / Z


def _build(board, pot, behind, db, menu):
    ba = build_turn_board_arrays(board, _CARDS)               # FULL: strg2/groups2
    idx = hand_index_map(ba)
    rivers = [c for c in FULL_DECK if c not in set(board)]
    if 0 < _RIVER_SAMPLE < len(rivers):
        r = np.random.default_rng(12345)
        rivers = sorted(r.choice(rivers, size=_RIVER_SAMPLE, replace=False).tolist())
    cache = {}
    strength = turn_strength(board, _EVAL, _CARDS, rivers=rivers, ba_cache=cache)
    part = equal_freq_partition(strength, _N_BUCKETS)
    buckets = sorted(set(part.values()))
    bidx = {b: i for i, b in enumerate(buckets)}
    tb_idx = np.array([bidx[part.get(h, buckets[0])] for h in ba['hands']], dtype=np.int64)
    tree = build_turn_tree(pot, (behind, behind))

    def leaf_fn(p, st):
        M0, M1, _, _, _ = turn_leaf_matrix_both(
            board, p, st, db, _EVAL, _CARDS,
            menu=menu, rivers=rivers, partition=part, ba_cache=cache)
        return M0, M1

    return ba, idx, tb_idx, tree, leaf_fn


def run():
    raw, db, menu = _raw_blueprint()
    if db is None:                       # turn_leaf_matrix_both needs a db for the rollout
        print("  SKIP: turn leaf rollout needs a blueprint DB; none available.")
        return True

    rows = []
    worst = 0.0
    for board, hole, bot_seat, pot, behind in _SPOTS:
        ba, idx, tb_idx, tree, leaf_fn = _build(board, pot, behind, db, menu)
        villain_seat = 1 - bot_seat
        meas = TurnCFR(tree, ba, tb_idx, leaf_fn)             # _br_value only

        hero_reach = _uniform(board, [], ba, idx)
        villain_true = _uniform(board, hole, ba, idx)

        bp = blueprint_turn_strategy_on_tree(tree, ba, raw, menu)
        bp_fn = lambda nid, _b=bp: _b[nid]
        bp_expl = _villain_br(meas, tree, ba, villain_seat, hero_reach, villain_true, bp_fn)

        # gadget anchored to the uniform (true) villain range -- the provable anchor.
        g0, g1 = (hero_reach, villain_true) if villain_seat == 1 else (villain_true, hero_reach)
        optout = blueprint_cfv_turn(tree, ba, raw, g0, g1, villain_seat, tb_idx, leaf_fn, menu)
        cfr, _ = solve_turn_gadget(tree, ba, tb_idx, leaf_fn, hero_reach, villain_true,
                                   optout, villain_seat, max_iters=_ITERS, check_every=_ITERS)
        e_gbp = _villain_br(meas, tree, ba, villain_seat, hero_reach, villain_true,
                            cfr.average_strategy)

        rows.append((''.join(board), bp_expl, e_gbp))
        slack = 1e-3 * (abs(bp_expl) + 1.0)
        worst = max(worst, e_gbp - bp_expl - slack)

    db.close()
    print("\nVillain BR vs hero (chips/dealt matchup; lower = safer)\n")
    print(f"  {'board':<14} {'blueprint':>10} {'gadget(bp)':>11}")
    for b, bp, gp in rows:
        print(f"  {b:<14} {bp:>10.3f} {gp:>11.3f}")
    ok = worst <= 0.0
    print(f"\n  HARD GATE (gadget(blueprint) <= blueprint, depth-limited): "
          f"{'PASS' if ok else 'FAIL'} (worst excess={worst:+.4f})")
    return ok


def test_turn_gadget_no_more_exploitable():
    assert run(), "turn gadget exceeded blueprint exploitability (depth-limited) on some spot"
    print("PASS test_turn_gadget_no_more_exploitable")


TESTS = [test_turn_gadget_no_more_exploitable]

if __name__ == '__main__':
    sys.exit(0 if run() else 1)
