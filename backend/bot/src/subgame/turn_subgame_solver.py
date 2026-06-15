# backend/bot/src/subgame/turn_subgame_solver.py
"""
TurnSubgameSolver (M4) -- serves the depth-limited TURN solve as a BotStrategy.

Subclasses RiverSubgameSolver and adds ONLY the turn path; everything else is
inherited, so the full served stack composes cleanly, each street solved once:

    decide():  near-terminal all-in guard  (inherited _run_guard)
            -> TURN  -> depth-limited turn solve (here)
            -> RIVER -> river subgame solve   (inherited)
            -> else  -> blueprint             (inherited)

The turn solve reuses the M0-M2 machinery: build_turn_tree + the reach-conditioned
leaf matrices (turn_leaf_matrix_both, M1 = -M0^T) + TurnCFR, then EV-gates the result
against the blueprint baseline exactly like the river path (inherited _gate_and_pick).

IMPORTANT -- the leaf models the bot's river as the BLUEPRINT, but at serve time the
bot actually plays the RIVER SOLVER on the river (the inherited river path). So the
leaf is a (conservative) GUIDE for turn betting; real river robustness comes from the
downstream river solve, not from the leaf. (This is why we don't need a multi-valued /
K-set leaf: the river solver is the continuation robustness K-set would approximate.)

LATENCY -- the leaf-matrix build (per distinct leaf x n buckets x rivers river-tree
evals) dominates and is SLOW (~20-50s single-core at n=48-64). So:
  * `n_buckets` / `leaf_rivers` are deliberately LOW (live fidelity), set from the
    minimum that still passes the LBR gate;
  * a turn SPR gate (`max_spr_turn`) skips deep/small-pot turns (tree blow-up + lowest
    stakes), like the river SOLVER_MAX_SPR;
  * this class is correct but NOT yet live-fast -- serving it needs either a parallel
    leaf build or BAKED leaf matrices (see ROADMAP M4). It is usable now for the
    OFFLINE LBR gate (slow solves are fine for measurement).
"""
import logging
import time

import numpy as np

from ..game.bot_strategy import BlueprintStrategy
from ..evaluation.showdown_kernel import build_turn_board_arrays
from .river_subgame_solver import RiverSubgameSolver, SOLVER_MAX_SPR
from .turn_tree import build_turn_tree
from .turn_cfr import TurnCFR
from .cfv import (turn_strength, equal_freq_partition, turn_leaf_matrix_both, FULL_DECK)
from .range_inputs import (project_tracker, blend_villain, hand_index_map, hand_row,
                           read_action_strategy)

_LOG = logging.getLogger(__name__)

# Live turn tree can represent an uncapped re-raise war (like the river solver), since
# GameSession uncaps re-raises live. Deep nodes collapse to jam/call once stacks commit.
LIVE_TURN_MAX_AGGRESSIONS = 5


def _iter_leaf_nodes(node):
    """Yield the depth-limit LEAF nodes (terminal with folder is None -> river still to
    come) of a turn tree. CFR traverses the whole tree each iteration, so eager-building
    these costs nothing extra vs the lazy path -- it just lets the wall-clock budget cover
    the (dominant) leaf build."""
    if node.terminal:
        if node.folder is None:
            yield node
        return
    for ch in node.children:
        yield from _iter_leaf_nodes(ch)


class TurnSubgameSolver(RiverSubgameSolver):
    # SPR cap for attempting a turn solve. The river's SOLVER_MAX_SPR=6.0 is tuned for the
    # RIVER tree and reusing it for the turn skipped ~95% of turns (vacuous-gate finding).
    # But the turn carries a LEAF-BUILD cost that EXPLODES with tree size (= SPR): a high-SPR
    # turn's leaf build blows the wall-clock budget and just wastes it on a timeout->fallback.
    # So the cap is set to the band that actually SOLVES within TURN_TIME_BUDGET (low SPR =
    # small tree = fast), NOT as high as possible -- admitting un-solvable high-SPR turns only
    # burns time. Pots that bring turns into this band come from real aggression (see the
    # maxbet opponent in the gate); a passive line stays high-SPR and is correctly skipped.
    TURN_MAX_SPR = 10.0
    # Hard wall-clock cap per turn solve (leaf-build + CFR). The leaf build is lazy inside
    # the first CFR chunk, so the budget covers it; on timeout the solve stops early and is
    # marked NON-converged -> the inherited EV gate's 4x non-converged margin makes a
    # deviation very unlikely (graceful degrade to blueprint) rather than acting on a
    # half-built/under-solved turn. ~10s matches the product latency target for a solve.
    TURN_TIME_BUDGET = 10.0

    def __init__(self, blueprint_db, *, n_buckets=48, leaf_rivers=6,
                 turn_max_iters=300, max_spr_turn=None, turn_time_budget=None, **kw):
        super().__init__(blueprint_db, **kw)
        self.n_buckets = int(n_buckets)
        self.leaf_rivers = int(leaf_rivers)       # 0 -> all river runouts (offline only)
        self.turn_max_iters = int(turn_max_iters)
        self.max_spr_turn = float(self.TURN_MAX_SPR if max_spr_turn is None else max_spr_turn)
        self.turn_time_budget = float(self.TURN_TIME_BUDGET if turn_time_budget is None
                                      else turn_time_budget)
        for k in ('turn_calls', 'turn_deviated', 'turn_kept', 'turn_timeout'):
            self.stats.setdefault(k, 0)
        self.turn_deviation_evs = []   # gate evDelta on each turn DEVIATION (per-deviation ledger)
        self.turn_solve_seconds = []   # wall-clock of each turn solve (latency calibration)

    # -- BotStrategy: turn here, everything else to the parent ------------------
    def decide(self, info_set_key, legal_actions, public_state):
        ps = public_state or {}
        if ps.get('street') != 'turn':
            return super().decide(info_set_key, legal_actions, public_state)
        self.last_debug = None
        guard = self._run_guard(legal_actions, ps)
        if guard is not None:
            return guard
        spec = self._turn_solver_inputs(ps)
        if spec is None:
            self.last_debug = {'mode': 'blueprint', 'street': 'turn'}
            return BlueprintStrategy.decide(self, info_set_key, legal_actions, public_state)
        self.stats['turn_calls'] += 1
        try:
            dist, node, info = self.solve_turn_for_action(**spec)
            self.turn_solve_seconds.append(float(info.get('seconds', 0.0)))
            result = self._gate_and_pick(dist, node, info, info_set_key,
                                         legal_actions, ps, spec, 'turn_solver', 'turn')
            dbg = self.last_debug or {}
            if dbg.get('deviated'):
                self.stats['turn_deviated'] += 1
                self.turn_deviation_evs.append(float(dbg.get('evDelta', 0.0)))
            else:
                self.stats['turn_kept'] += 1
            self._attach_hero_range_update(node, info)
            return result
        except Exception:
            self._fallback_count += 1
            self.stats['fallback'] += 1
            self.last_debug = {'mode': 'fallback', 'street': 'turn', 'solved': False}
            n = self._fallback_count
            if n <= 5 or n % 100 == 0:
                _LOG.warning("TurnSubgameSolver fell back to blueprint (#%d)", n,
                             exc_info=True)
            return BlueprintStrategy.decide(self, info_set_key, legal_actions, public_state)

    def _attach_hero_range_update(self, node, info):
        """1a (continual re-solving): when the turn solve DEVIATED from the blueprint,
        expose the SOLVED per-hand probability of the played action as
        last_debug['heroRangeUpdate'] so GameSession updates the bot's OWN range off the
        solved turn play. The river-entry hero range the inherited river solver then
        consumes is consistent with how the bot ACTUALLY played the turn -- the N0 fix
        (NN_LEAF_PLAN 6d thread 1: a turn deviation must not feed the river solver a
        blueprint range). No-op when the EV gate KEPT the blueprint (the bot effectively
        played the blueprint, so the usual blueprint range update downstream is right)."""
        dbg = self.last_debug
        if not dbg or not dbg.get('deviated') or 'chosenTreeAction' not in dbg:
            return
        choice = dbg['chosenTreeAction']
        if choice not in node.actions:
            return
        ai = node.actions.index(choice)
        col = info['cfr'].average_strategy(node.node_id)[:, ai]   # per-hand P(played action)
        hands = info['ba']['hands']
        dbg['heroRangeUpdate'] = {frozenset(h): float(col[i]) for i, h in enumerate(hands)}

    def _hand_action_evs(self, info, node, row):
        """1c: grade the turn deviation on the EXACT rollout leaf, not the bucketed matrix
        the solve optimised -- otherwise the EV gate SELF-GRADES and waves through a
        deviation that loses under the true continuation (a root cause of the N0 failure;
        NN_LEAF_PLAN 6d). Builds an ExactLeafTurnCFR that shares the SOLVED average
        strategy (strat_sum copied) but values every depth-limit leaf via
        turn_leaf_value_exact (the two-sided _terminal override), then reads the bot's row
        and normalises by compatible villain mass exactly like the bucketed hand_action_evs.
        The rollout cost is fine: the turn solver is offline-only (not live-served)."""
        from .turn_cfr import ExactLeafTurnCFR
        from ..evaluation.showdown_kernel import compatible_mass
        # _gate_and_pick is SHARED with the inherited RIVER path: a TurnSubgameSolver also
        # handles river decisions, whose info carries a base RiverCFR (no tb_idx/board4) and
        # is already exact at showdown. Only TURN info stashes 'board4' -> use it as the
        # discriminator and defer river decisions to the base (bucketed-but-exact) impl.
        if 'board4' not in info:
            return super()._hand_action_evs(info, node, row)
        solver = info['cfr']
        exact = ExactLeafTurnCFR(
            solver.tree, info['ba'], solver.tb_idx, solver.leaf_matrix_fn,
            info['board4'], self.db, self._evaluator, self._cards,
            menu=self._postflop_menu, rivers=info['rivers'],
            ba_cache=info['ba_cache'], adversary='blueprint')
        exact.strat_sum = [s.copy() for s in solver.strat_sum]   # match the solved average
        reach0, reach1 = info['reach0'], info['reach1']
        villain = np.asarray(reach1 if node.player == 0 else reach0, float)
        z = compatible_mass(exact.ba, villain)[row]
        if z <= 1e-9:                                            # no compatible villain mass
            return None
        vals = exact.node_action_values(node, reach0, reach1)    # [H, A], EXACT leaves
        return vals[row] / z

    def _turn_solver_inputs(self, ps):
        """Turn-solve inputs from public_state, or None to fall back. The turn-entry
        fields (turnEntryPot/turnEntryStacks/turnPath) are added to bot_public_state at
        wiring; absent them, return None. Skips high-SPR turns (tree blow-up + lowest
        stakes -- the guard + blueprint cover those)."""
        if ps.get('street') != 'turn':
            return None
        required = ('turnEntryPot', 'turnEntryStacks', 'botSeat', 'hole_cards',
                    'opp_range', 'hero_range', 'turnPath', 'community')
        if any(ps.get(k) is None for k in required):
            return None
        pot_entry = ps['turnEntryPot']
        stacks = ps['turnEntryStacks']
        eff = stacks[0] if isinstance(stacks, (list, tuple)) else stacks
        if pot_entry <= 0 or eff / pot_entry > self.max_spr_turn:
            return None
        tracker = ps['opp_range']
        return {
            'board': ps['community'],
            'pot_entry': pot_entry,
            'stacks': stacks,
            'bot_seat': ps['botSeat'],
            'hole': ps['hole_cards'],
            'villain_tracker': tracker,
            'hero_tracker': ps['hero_range'],
            'confidence': getattr(tracker, 'confidence', 1.0),
            'turn_path': ps['turnPath'],
        }

    # -- core: solve the turn subgame + read off -------------------------------
    def solve_turn_for_action(self, *, board, pot_entry, stacks, bot_seat, hole,
                              villain_tracker, hero_tracker, confidence, turn_path):
        """Solve the depth-limited turn subgame and return (action_dist, node, info)
        for the bot's actual hand at its decision node along `turn_path`. board: 4
        SuitRank turn cards. The leaf matrices use the blueprint's postflop menu
        (self._postflop_menu) for projection; the tree uses the solver menu."""
        ba_cache = {}
        rivers = [c for c in FULL_DECK if c not in set(board)]
        if self.leaf_rivers and 0 < self.leaf_rivers < len(rivers):
            # UNBIASED sample, not rivers[:k]: FULL_DECK is rank-ascending, so the slice
            # took only the lowest-rank rivers (deuces/treys) -> the leaf, the strength
            # PARTITION, and the 1c exact gate all saw low runouts only, biasing toward
            # made hands (the N0 overaggression symptom). Sample uniformly with a
            # per-board-deterministic RNG (reproducible per board, no systematic bias).
            rsamp = np.random.default_rng(abs(hash(tuple(board))) % (2 ** 32))
            rivers = sorted(rsamp.choice(rivers, size=self.leaf_rivers,
                                         replace=False).tolist())

        strength = turn_strength(board, self._evaluator, self._cards,
                                 rivers=rivers, ba_cache=ba_cache)
        part = equal_freq_partition(strength, self.n_buckets)
        buckets = sorted(set(part.values()))
        bidx = {b: i for i, b in enumerate(buckets)}
        ba = build_turn_board_arrays(board)               # light: hands/c1/c2 (no buckets)
        idx = hand_index_map(ba)
        tb_idx = np.array([bidx[part[h]] for h in ba['hands']], dtype=np.int64)
        tree = build_turn_tree(pot_entry, stacks, menu=self.menu,
                               max_aggressions=LIVE_TURN_MAX_AGGRESSIONS)

        def leaf_fn(pot, st):
            M0, M1, _, _, _ = turn_leaf_matrix_both(
                board, pot, st, self.db, self._evaluator, self._cards,
                menu=self._postflop_menu, rivers=rivers, partition=part, ba_cache=ba_cache)
            return M0, M1

        villain = blend_villain(project_tracker(villain_tracker, ba, idx),
                                confidence, self.temper_beta)
        hero = project_tracker(hero_tracker, ba, idx)
        reach0, reach1 = (hero, villain) if bot_seat == 0 else (villain, hero)

        row = hand_row(ba, hole, idx)
        if row is None:
            raise ValueError("bot hole cards collide with the board")
        if hero[row] <= 1e-12:
            raise ValueError("bot hand has ~zero hero reach; solve can't represent it")
        node, edge_path = self._navigate_turn(tree, turn_path)
        if node is None or node.terminal:
            raise ValueError("turn path did not land on a decision node")
        if node.player != bot_seat:
            raise ValueError(f"path landed on seat {node.player}, not the bot ({bot_seat})")

        solver = TurnCFR(tree, ba, tb_idx, leaf_fn)
        # Wall-clock cap covering BOTH the (dominant) leaf build and CFR. Build the leaf
        # matrices EAGERLY with a deadline so the budget actually bounds them -- the lazy
        # in-CFR build would let one heavy leaf pass blow the cap. If the leaves can't build
        # in budget (SPR too high for the tree size), abort -> blueprint fallback (decide's
        # except). Then CFR runs on the remaining budget; a CFR timeout -> non-converged ->
        # the EV gate's 4x margin guards a deviation on a half-solved turn.
        t0 = time.perf_counter()
        for leaf in _iter_leaf_nodes(tree.root):
            solver._leaf_mats(leaf)
            if time.perf_counter() - t0 >= self.turn_time_budget:
                raise TimeoutError(
                    f"turn leaf-build exceeded {self.turn_time_budget:g}s budget "
                    "(SPR too high for the tree size); falling back to blueprint")
        chunk = max(1, int(getattr(self, 'check_every', 40)))
        done = 0
        while done < self.turn_max_iters:
            solver.run(reach0, reach1, iters=min(chunk, self.turn_max_iters - done))
            done += min(chunk, self.turn_max_iters - done)
            if time.perf_counter() - t0 >= self.turn_time_budget:
                break
        seconds = time.perf_counter() - t0
        converged = done >= self.turn_max_iters
        if not converged:
            self.stats['turn_timeout'] += 1

        dist = read_action_strategy(solver, node, hole, ba, idx)
        node_reach0, node_reach1 = solver.reach_into(edge_path, reach0, reach1)
        info = {'cfr': solver, 'ba': ba, 'idx': idx,
                'reach0': node_reach0, 'reach1': node_reach1,
                'iters': done, 'gap': None, 'converged': converged, 'seconds': seconds,
                # 1c: inputs for the exact-leaf EV gate (grade the deviation on the TRUE
                # rollout continuation, not the bucketed leaf the solve used).
                'board4': board, 'rivers': rivers, 'ba_cache': ba_cache}
        return dist, node, info

    def _navigate_turn(self, tree, turn_path):
        """Walk the turn tree following `turn_path`, snapping off-grid sizes to the
        nearest edge via the inherited _match_edge. (The turn tree has no
        inject_realized_edge yet -- off-grid turn bets snap; nested injection is a
        later refinement, like the river path's.) Returns (node, edge_indices)."""
        node = tree.root
        edges = []
        for spec in turn_path:
            if node.terminal:
                return None, edges
            i = self._match_edge(node, spec)
            if i is None:
                return None, edges
            edges.append(i)
            node = node.children[i]
        return node, edges
