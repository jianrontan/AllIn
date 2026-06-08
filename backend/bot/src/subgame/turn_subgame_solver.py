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


class TurnSubgameSolver(RiverSubgameSolver):
    def __init__(self, blueprint_db, *, n_buckets=48, leaf_rivers=6,
                 turn_max_iters=300, max_spr_turn=SOLVER_MAX_SPR, **kw):
        super().__init__(blueprint_db, **kw)
        self.n_buckets = int(n_buckets)
        self.leaf_rivers = int(leaf_rivers)       # 0 -> all river runouts (offline only)
        self.turn_max_iters = int(turn_max_iters)
        self.max_spr_turn = float(max_spr_turn)
        self.stats.setdefault('turn_calls', 0)

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
            return self._gate_and_pick(dist, node, info, info_set_key,
                                       legal_actions, ps, spec, 'turn_solver', 'turn')
        except Exception:
            self._fallback_count += 1
            self.stats['fallback'] += 1
            self.last_debug = {'mode': 'fallback', 'street': 'turn', 'solved': False}
            n = self._fallback_count
            if n <= 5 or n % 100 == 0:
                _LOG.warning("TurnSubgameSolver fell back to blueprint (#%d)", n,
                             exc_info=True)
            return BlueprintStrategy.decide(self, info_set_key, legal_actions, public_state)

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
        if self.leaf_rivers and self.leaf_rivers > 0:
            rivers = rivers[:self.leaf_rivers]

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
        solver.run(reach0, reach1, iters=self.turn_max_iters)

        dist = read_action_strategy(solver, node, hole, ba, idx)
        node_reach0, node_reach1 = solver.reach_into(edge_path, reach0, reach1)
        info = {'cfr': solver, 'ba': ba, 'idx': idx,
                'reach0': node_reach0, 'reach1': node_reach1,
                'iters': self.turn_max_iters, 'gap': None, 'converged': True}
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
