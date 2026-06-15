# backend/bot/src/subgame/turn_cfr.py
"""
Two-sided vectorized CFR+ over the TURN betting tree (M2) -- depth-limited solving.

This is RiverCFR with the turn's two differences, and NOTHING else: the CFR+
mechanics (reach-carrying traversal, regret-matching+, linear-weighted average,
best-response walk, exploitability, the persisted Linear-CFR clock) are subtle and
must stay bit-for-bit in sync with the river solver, so TurnCFR SUBCLASSES RiverCFR
and overrides ONLY the two terminal evaluators:

  * FOLD terminal -- identical to the river (pot transfer x compatible_mass), but on
    the 4-card turn basis (H = 1128).
  * LEAF terminal (a non-fold close: the turn betting is over but the RIVER is still
    to come) -- value = the blueprint's river-continuation value, supplied as the M0
    reach-conditioned leaf matrices (M0 for seat-0 hero, M1 for seat-1 hero; the river
    is positional) dotted with the LIVE reach at the leaf. This is what makes the
    solve depth-limited rather than solve-to-showdown: the river subtree is collapsed
    into one value object, re-weighted by the current reach every traversal.

The leaf matrices depend on the leaf's (final_pot, leaf_stacks) but NOT on the live
reach, so they are built once per distinct leaf (via `leaf_matrix_fn`) and cached;
each traversal only re-dots them with the current reach (cheap). An all-in-and-called
leaf has leaf_stacks==(0,0) -> its matrices encode pure equity-to-river (no inner
river betting), handled with no special case (see turn_tree.py).
"""
import numpy as np

from ..evaluation.showdown_kernel import compatible_mass
from .river_cfr import RiverCFR
from .cfv import leaf_value_vec, turn_leaf_value_exact, turn_leaf_br_value


class TurnCFR(RiverCFR):
    def __init__(self, tree, ba, tb_idx, leaf_matrix_fn):
        """tree: a built TurnTree. ba: build_turn_board_arrays(board4) (the H=1128
        basis; carries c1/c2 for card removal). tb_idx: [H] leaf-bucket ROW-index per
        turn hand (== bidx[tb[hand]], the shared partition). leaf_matrix_fn:
        (final_pot, leaf_stacks) -> (M0, M1) reach-conditioned leaf matrices for that
        leaf (caller builds + may cache the heavy river pass; TurnCFR also caches by
        the leaf's pot/stacks so it is built at most once per distinct leaf)."""
        # These MUST be set BEFORE super().__init__: the overridden _terminal /
        # _terminal_one depend on them. (Safe today because RiverCFR.__init__ only
        # allocates tables and never traverses, but set them first so a future
        # __init__ that does traverse can't hit an unset field.)
        self.c1 = ba['c1']
        self.c2 = ba['c2']
        self.tb_idx = np.asarray(tb_idx, dtype=np.int64)
        self.leaf_matrix_fn = leaf_matrix_fn
        self._leaf_cache = {}
        super().__init__(tree, ba)

    def _leaf_mats(self, node):
        key = (round(node.final_pot, 6),
               round(node.leaf_stacks[0], 6), round(node.leaf_stacks[1], 6))
        m = self._leaf_cache.get(key)
        if m is None:
            m = self.leaf_matrix_fn(node.final_pot, node.leaf_stacks)
            self._leaf_cache[key] = m
        return m

    # -- terminal values (the ONLY override vs RiverCFR) ------------------------
    def _terminal(self, node, reach0, reach1):
        if node.folder is None:                   # LEAF: blueprint river continuation
            M0, M1 = self._leaf_mats(node)
            v0 = leaf_value_vec(M0, self.tb_idx, self.c1, self.c2, reach1)
            v1 = leaf_value_vec(M1, self.tb_idx, self.c1, self.c2, reach0)
            return v0, v1
        fp = node.final_pot                        # FOLD: constant payoff per hand
        c0, c1 = node.contrib
        if node.folder == 0:
            p0, p1 = -c0, fp - c1
        else:
            p0, p1 = fp - c0, -c1
        v0 = p0 * compatible_mass(self.ba, reach1)
        v1 = p1 * compatible_mass(self.ba, reach0)
        return v0, v1

    def _terminal_one(self, node, hero, reach_villain):
        """Hero's value vector vs a single villain reach (for the BR walk)."""
        if node.folder is None:
            M0, M1 = self._leaf_mats(node)
            M = M0 if hero == 0 else M1
            return leaf_value_vec(M, self.tb_idx, self.c1, self.c2, reach_villain)
        fp = node.final_pot
        ch = node.contrib[hero]
        payoff = -ch if node.folder == hero else fp - ch
        return payoff * compatible_mass(self.ba, reach_villain)

    def exploitability(self, reach0, reach1, strat_fn=None):
        """DEPTH-LIMITED exploitability of the turn strategy: BR0(s1)+BR1(s0) where, at
        every depth-limit LEAF, BOTH the strategy AND the best-responder are valued by
        the FROZEN blueprint river continuation (the leaf matrices) -- the BR can only
        deviate in TURN actions, not inside the leaf's river. So this is the convergence
        gap of the turn solve *given the bucketed-leaf game it defines*, NOT the true
        exploitability of the turn+river subgame. It does NOT account for a villain who
        deviates in the river (the unsafe-v1 frozen-range trap). Use it to confirm the
        CFR+ run converged on its own game; use the M2 Stage-3 exact-rollout gate and
        the M4 LBR gate (a villain that can deviate downstream) to judge real quality.
        Behaviour is identical to RiverCFR.exploitability -- only the meaning differs on
        the turn tree, hence this override exists purely to document it."""
        return super().exploitability(reach0, reach1, strat_fn=strat_fn)


class ExactLeafTurnCFR(TurnCFR):
    """A TurnCFR whose BEST-RESPONSE walk values leaves by the EXACT blueprint river
    rollout (turn_leaf_value_exact) instead of the bucketed matrix the solver used --
    the M2 Stage-3 'out-of-bucket' adversary. Used ONLY to MEASURE exploitability
    (exploitability / _br_value), not to solve: the regret-update path (_terminal) still
    uses the matrix. Overriding _terminal_one means `exploitability(reach0, reach1,
    strat_fn=...)` grades any strategy (blueprint projection OR a solved average) against
    an adversary that gets the TRUE blueprint continuation at the depth limit -- so a
    solver cannot win the gate by gaming its own leaf bucketing.

    Still IN-MODEL-RIVER (the rollout assumes blueprint river play); a villain that
    DEVIATES in the river is only caught by M4's LBR. So this gate is necessary, not
    sufficient -- it isolates 'did the turn betting solve improve over the blueprint,
    judged on the true continuation values'.

    COST: each leaf BR eval is one full rollout (~all river runouts x project+eval) and
    is reach-dependent (no caching across BR calls) -- heavy; for offline measurement on
    a sample of turn spots, not live."""

    def __init__(self, tree, ba, tb_idx, leaf_matrix_fn, board4, db, evaluator, cards,
                 menu=None, rivers=None, ba_cache=None, adversary='blueprint'):
        """adversary: 'blueprint' -> leaf BR vs the EXACT blueprint river continuation
        (out-of-bucket, IN-model-river; the M2 gate). 'river_br' -> the BR player ALSO
        best-responds on the river (out-of-MODEL-river; exposes the frozen-range trap,
        the M3 gate)."""
        self.board4 = board4
        self.db = db
        self.evaluator = evaluator
        self.cards = cards
        self.menu = menu
        self.rivers = rivers
        self.exact_ba_cache = ba_cache if ba_cache is not None else {}
        self._hands = ba['hands']
        assert adversary in ('blueprint', 'river_br')
        self.adversary = adversary
        super().__init__(tree, ba, tb_idx, leaf_matrix_fn)

    def _terminal(self, node, reach0, reach1):
        """Two-sided EXACT-leaf terminal so node_action_values / _eval grade a strategy
        on the TRUE continuation (1c: the turn EV gate must not self-grade on the bucketed
        leaf). The leaf is just _terminal_one per seat (which does the exact rollout);
        a fold is the exact pot transfer (the bucketed branch is already exact there)."""
        if node.folder is None:
            return (self._terminal_one(node, 0, reach1),
                    self._terminal_one(node, 1, reach0))
        return super()._terminal(node, reach0, reach1)

    def _terminal_one(self, node, hero, reach_villain):
        if node.folder is None:                       # LEAF
            vill = {self._hands[i]: float(reach_villain[i])
                    for i in range(self.H) if reach_villain[i] > 0.0}
            if self.adversary == 'river_br':          # hero ALSO best-responds on the river
                val = turn_leaf_br_value(
                    self.board4, node.final_pot, node.leaf_stacks, hero, vill,
                    self.db, self.evaluator, self.cards, menu=self.menu,
                    rivers=self.rivers, ba_cache=self.exact_ba_cache)
            else:                                     # hero plays the blueprint river (exact)
                hero_range = {h: 1.0 for h in self._hands}   # value independent of hero reach
                val = turn_leaf_value_exact(
                    self.board4, node.final_pot, node.leaf_stacks, hero, hero_range, vill,
                    self.db, self.evaluator, self.cards, menu=self.menu,
                    rivers=self.rivers, ba_cache=self.exact_ba_cache)
            return np.array([val.get(h, 0.0) for h in self._hands])
        return super()._terminal_one(node, hero, reach_villain)   # fold: pot transfer
