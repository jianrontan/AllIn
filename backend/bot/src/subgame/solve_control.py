# backend/bot/src/subgame/solve_control.py
"""
Solve control for the river subgame (Phase-4, step 5): convergence-based early
stop, and the EV gate.

  * solve_river(...) -- run CFR+ in increments, checking exploitability (the BR
    gap) periodically, and stop as soon as it is small enough (or a max-iters /
    time budget is hit). The Linear-CFR clock persists across the increments
    (river_cfr.RiverCFR._iter), so incremental solving matches one big run. This
    is both the quality guarantee (we solve until actually converged) and the
    speed win (we stop early on easy spots instead of grinding a fixed count).

  * ev_gate(...) -- after solving, deviate from the blueprint ONLY if the solved
    strategy beats it by a margin under our own belief. The gate does NOT protect
    against a wrong input range (a confidently-wrong belief yields a high solved
    EV the gate happily passes -- only the confidence widening guards that); what
    it catches is (a) a numerically-broken / under-converged solve that fails to
    beat the blueprint even under its own belief, and (b) negligible-edge spots
    where deviating just adds variance/churn for no real gain.

The blueprint baseline (and the warm-start prior) are fed in at step-6 wiring,
where the blueprint<->tree bridge is built; here the gate is a pure function over
an explicit baseline distribution.
"""
import math
import time

import numpy as np

from .river_cfr import RiverCFR
from ..evaluation.showdown_kernel import compatible_mass


def solve_river(tree, ba, reach0, reach1, *, max_iters=1000, check_every=50,
                gap_threshold=None, time_budget=None,
                warm_start=None, warm_weight=0.0):
    """Solve the river subgame with convergence-based early stop.

    gap_threshold is in chips per dealt hand-pair (default: 1% of the entry pot).
    Returns (cfr, info) where info has iters / gap / seconds / converged.
    """
    cfr = RiverCFR(tree, ba)
    if warm_start is not None and warm_weight > 0:
        cfr.warm_start(warm_start, warm_weight)
    if gap_threshold is None:
        gap_threshold = 0.01 * tree.pot_entry

    t0 = time.time()
    done = 0
    gap = float('inf')
    per_iter = None                       # measured cost of one CFR iteration (s)
    while done < max_iters:
        step = min(check_every, max_iters - done)
        # Budget-granularity: a fixed `check_every` block runs to completion before the
        # post-block time check, so on a bigger (turn) tree one block can overrun the
        # budget by a lot. Once we know the per-iteration cost, shrink the next block to
        # what fits in the remaining budget so the overrun is bounded to ~one iteration.
        if time_budget is not None:
            remaining = time_budget - (time.time() - t0)
            if remaining <= 0:
                break
            if per_iter and per_iter > 0 and math.isfinite(remaining):
                step = max(1, min(step, int(remaining / per_iter)))
        bt = time.time()
        cfr.run(reach0, reach1, iters=step)
        per_iter = (time.time() - bt) / step
        done += step
        gap = cfr.exploitability(reach0, reach1)
        if gap <= gap_threshold:
            break
        if time_budget is not None and (time.time() - t0) >= time_budget:
            break
    return cfr, {'iters': done, 'gap': float(gap),
                 'seconds': time.time() - t0,
                 'converged': bool(gap <= gap_threshold)}


def solve_river_gadget(tree, ba, hero_reach, villain_reach, optout, villain_seat, *,
                       max_iters=1000, check_every=50, time_budget=None):
    """Solve the river subgame under the SAFE re-solving gadget (RiverCFR.run_gadget):
    the villain gets a per-hand opt-out paying `optout` (the blueprint river-entry CFV
    from blueprint_projection.blueprint_cfv), so the solved HERO strategy is
    no-more-exploitable than the blueprint (Phase 5a). Mirrors solve_river's
    increment / time-budget loop.

    NOTE there is no cheap per-iteration convergence GAP here: the gadget deliberately
    reshapes the villain's effective range, so the ordinary subgame Nash gap is not the
    safety target. We therefore run to the iteration/time budget and report
    `converged = (ran the full max_iters within budget)` -- a budget proxy the EV gate's
    non-converged margin (H1) consumes the same way it does for solve_river."""
    cfr = RiverCFR(tree, ba)
    hero_reach = np.asarray(hero_reach, float)
    villain_reach = np.asarray(villain_reach, float)
    optout = np.asarray(optout, float)
    t0 = time.time()
    done = 0
    per_iter = None
    while done < max_iters:
        step = min(check_every, max_iters - done)
        if time_budget is not None:
            remaining = time_budget - (time.time() - t0)
            if remaining <= 0:
                break
            if per_iter and per_iter > 0 and math.isfinite(remaining):
                step = max(1, min(step, int(remaining / per_iter)))
        bt = time.time()
        cfr.run_gadget(hero_reach, villain_reach, optout, villain_seat, iters=step)
        per_iter = (time.time() - bt) / step
        done += step
        if time_budget is not None and (time.time() - t0) >= time_budget:
            break
    # NOTE: 'converged' here is a BUDGET PROXY (hit max_iters), NOT a Nash-gap check -- the gadget
    # reshapes the villain range each iteration, so there is no cheap per-iteration subgame gap to
    # measure (unlike solve_river). The <=blueprint SAFETY GUARANTEE is strongest when the gadget
    # actually converges, so the time_budget must be generous enough to hit max_iters on served spots
    # (it is: 275 iters fit well inside 24s). A time-truncated gadget (converged=False) is bounded only
    # by the EV gate's margin -- which guards EV trustworthiness, NOT exploitability -- so don't rely on
    # it for safety; keep the budget ample. (Review note 2026-06-22.)
    return cfr, {'iters': done, 'gap': None, 'seconds': time.time() - t0,
                 'converged': bool(done >= max_iters)}


def solve_turn_gadget(tree, ba, tb_idx, leaf_matrix_fn, hero_reach, villain_reach,
                      optout, villain_seat, *, max_iters=1000, check_every=50,
                      time_budget=None):
    """Solve the depth-limited TURN subgame under the SAFE re-solving gadget -- the turn
    analogue of solve_river_gadget. `TurnCFR.run_gadget` is INHERITED from RiverCFR and
    automatically uses TurnCFR's depth-limited leaf (the river collapsed into the leaf
    matrices) via _cfr -> _terminal, so no turn-specific gadget code is needed. The
    villain gets a per-hand opt-out paying `optout` (blueprint TURN-entry CFV from
    blueprint_projection.blueprint_cfv_turn), so the solved HERO turn strategy is
    no-more-exploitable than the blueprint within the depth-limited game.

    Same budget loop + `converged = ran the full max_iters within budget` proxy as
    solve_river_gadget (the gadget reshapes the villain range, so there is no cheap
    per-iteration Nash gap; the EV gate's non-converged margin consumes the proxy)."""
    from .turn_cfr import TurnCFR
    cfr = TurnCFR(tree, ba, tb_idx, leaf_matrix_fn)
    hero_reach = np.asarray(hero_reach, float)
    villain_reach = np.asarray(villain_reach, float)
    optout = np.asarray(optout, float)
    t0 = time.time()
    done = 0
    per_iter = None
    while done < max_iters:
        step = min(check_every, max_iters - done)
        if time_budget is not None:
            remaining = time_budget - (time.time() - t0)
            if remaining <= 0:
                break
            if per_iter and per_iter > 0 and math.isfinite(remaining):
                step = max(1, min(step, int(remaining / per_iter)))
        bt = time.time()
        cfr.run_gadget(hero_reach, villain_reach, optout, villain_seat, iters=step)
        per_iter = (time.time() - bt) / step
        done += step
        if time_budget is not None and (time.time() - t0) >= time_budget:
            break
    return cfr, {'iters': done, 'gap': None, 'seconds': time.time() - t0,
                 'converged': bool(done >= max_iters)}


# Below this compatible-villain-mass the per-action EVs are undefined: dividing the
# tiny `vals` row by a tiny `z` blows the chip-EVs up to garbage that would dominate the
# EV-gate margin and spuriously deviate/keep. Return None so the caller keeps the
# blueprint baseline instead (M1).
_EV_MIN_MASS = 1e-9


def hand_action_evs(cfr, node, hand_row, reach0, reach1):
    """Per-action chip EV (length = #actions at `node`) for the hand at `hand_row`,
    under the solved average strategy. Normalised by the hand's compatible villain
    mass so the values are in chips per dealt matchup (makes the EV-gate margin an
    interpretable chip amount). Returns None when that mass is ~zero (EV undefined)."""
    p = node.player
    villain = np.asarray(reach1 if p == 0 else reach0, float)
    z = compatible_mass(cfr.ba, villain)[hand_row]
    if z <= _EV_MIN_MASS:
        return None                                            # no compatible villain mass
    vals = cfr.node_action_values(node, reach0, reach1)        # [H, A] measures
    return vals[hand_row] / z


def ev_gate(actions, solved_dist, baseline_dist, action_evs, margin):
    """Pick the solved strategy over the baseline only if it is materially better.

    actions      : ordered action labels at the node.
    solved_dist  : {action: prob} from the solve (the bot's actual-hand strategy).
    baseline_dist: {action: prob} the blueprint would play here (mapped to the
                   tree menu at step-6 wiring).
    action_evs   : chip EV per action, aligned to `actions` (from hand_action_evs).
    margin       : minimum chip EV advantage required to deviate from the baseline.

    Returns (chosen_dist, info). info.used is 'solved' or 'baseline'.
    """
    ev_s = float(sum(solved_dist.get(a, 0.0) * v for a, v in zip(actions, action_evs)))
    ev_b = float(sum(baseline_dist.get(a, 0.0) * v for a, v in zip(actions, action_evs)))
    use_solved = (ev_s - ev_b) >= margin
    chosen = solved_dist if use_solved else baseline_dist
    return chosen, {'ev_solved': ev_s, 'ev_baseline': ev_b,
                    'delta': ev_s - ev_b, 'used': 'solved' if use_solved else 'baseline'}
