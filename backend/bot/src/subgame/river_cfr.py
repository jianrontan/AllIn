# backend/bot/src/subgame/river_cfr.py
"""
Two-sided vectorized CFR+ over the river betting tree (Phase-4, step 3).

Solves the river subgame to an approximate Nash equilibrium for BOTH players'
hand ranges at once, then exposes the average strategy per node (the bot reads
off the row for its actual hand in step 4).

VECTORIZED OVER HANDS
---------------------
Each player has H (=1081 on a 5-card board) possible hands. Every decision node
owned by player p carries [H, A] cumulative-regret and strategy-sum tables. One
CFR traversal carries BOTH players' reach vectors down the tree and returns a
value vector for EACH player:

    cfr(node, reach0, reach1) -> (v0, v1)

KEY INVARIANT that makes this correct in one pass: player 0's value depends only
on player 1's reach (and vice versa) -- at a terminal v0 is the kernel showdown
measure against reach1, never reach0. So the value returned for the ACTING player
down action a is its true counterfactual value (opponent-reach-weighted,
independent of its own reach split), exactly what the regret update needs; while
the NON-acting player's value is summed over the acting player's strategy-split
children. Terminals use the shared showdown kernel (exact card removal).

This is CFR+: regrets are floored at 0 each iteration; the average strategy is
linearly weighted (weight = iteration t). Exploitability (the convergence gap) is
a best-response walk of the same tree against the current average strategy.
"""
import numpy as np

from ..evaluation.showdown_kernel import showdown_measure, compatible_mass


class RiverCFR:
    def __init__(self, tree, ba):
        """tree: a built RiverTree. ba: build_board_arrays(board, evaluator, cards)
        for that river board (the H-hand basis both ranges live on)."""
        self.tree = tree
        self.ba = ba
        self.H = ba['H']
        self.nodes = tree.decision_nodes
        self.regret = [np.zeros((self.H, len(n.actions))) for n in self.nodes]
        self.strat_sum = [np.zeros((self.H, len(n.actions))) for n in self.nodes]
        # Cumulative Linear-CFR clock, PERSISTED across run() calls. The strategy
        # sum is weighted by the iteration number, so the clock must keep counting
        # when a solve is split into increments (step 5's convergence-based early
        # stop / warm-start). Resetting it per run() would under-weight the later,
        # better strategies. run(100)+run(100) is therefore bit-identical to run(200).
        self._iter = 0
        self._t_weight = 1.0

    # -- current strategy (regret-matching+) -----------------------------------
    def _strategy(self, nid):
        r = self.regret[nid]                      # already floored at >= 0 (CFR+)
        s = r.sum(axis=1, keepdims=True)
        A = r.shape[1]
        return np.where(s > 1e-12, r / np.where(s > 1e-12, s, 1.0), 1.0 / A)

    def average_strategy(self, nid):
        """Linearly-weighted average strategy at a node: [H, A] rows summing to 1."""
        s = self.strat_sum[nid]
        tot = s.sum(axis=1, keepdims=True)
        A = s.shape[1]
        return np.where(tot > 1e-12, s / np.where(tot > 1e-12, tot, 1.0), 1.0 / A)

    # -- terminal values -------------------------------------------------------
    def _terminal(self, node, reach0, reach1):
        fp = node.final_pot
        c0, c1 = node.contrib
        if node.folder is None:                   # showdown (exact, card removal)
            v0 = showdown_measure(self.ba, reach1, fp, c0)
            v1 = showdown_measure(self.ba, reach0, fp, c1)
        else:                                     # fold: constant payoff per hand
            if node.folder == 0:
                p0, p1 = -c0, fp - c1
            else:
                p0, p1 = fp - c0, -c1
            v0 = p0 * compatible_mass(self.ba, reach1)
            v1 = p1 * compatible_mass(self.ba, reach0)
        return v0, v1

    def _terminal_one(self, node, hero, reach_villain):
        """Hero's value vector vs a single villain reach (for the BR walk)."""
        fp = node.final_pot
        ch = node.contrib[hero]
        if node.folder is None:
            return showdown_measure(self.ba, reach_villain, fp, ch)
        if node.folder == hero:
            payoff = -ch
        else:
            payoff = fp - ch
        return payoff * compatible_mass(self.ba, reach_villain)

    # -- one CFR+ traversal ----------------------------------------------------
    def _cfr(self, node, reach0, reach1):
        if node.terminal:
            return self._terminal(node, reach0, reach1)

        nid = node.node_id
        p = node.player
        strat = self._strategy(nid)
        A = len(node.actions)

        u = np.zeros((self.H, A))                 # acting player's action values
        v_other = np.zeros(self.H)
        for a in range(A):
            if p == 0:
                cv0, cv1 = self._cfr(node.children[a], reach0 * strat[:, a], reach1)
                u[:, a] = cv0
                v_other = v_other + cv1
            else:
                cv0, cv1 = self._cfr(node.children[a], reach0, reach1 * strat[:, a])
                u[:, a] = cv1
                v_other = v_other + cv0

        v_p = (strat * u).sum(axis=1)

        # CFR+ regret update (accumulate then floor at 0).
        self.regret[nid] += u - v_p[:, None]
        np.maximum(self.regret[nid], 0.0, out=self.regret[nid])
        # Linearly-weighted strategy sum, own-reach weighted.
        reach_p = reach0 if p == 0 else reach1
        self.strat_sum[nid] += (reach_p[:, None] * strat) * self._t_weight

        return (v_p, v_other) if p == 0 else (v_other, v_p)

    def run(self, reach0, reach1, iters=500):
        """Run `iters` CFR+ iterations from the given root reach vectors. The
        Linear-CFR clock continues across calls (see _iter), so solving in
        increments matches one big run() exactly."""
        reach0 = np.asarray(reach0, dtype=float)
        reach1 = np.asarray(reach1, dtype=float)
        for _ in range(iters):
            self._iter += 1
            self._t_weight = float(self._iter)
            self._cfr(self.tree.root, reach0, reach1)
        return self

    def run_gadget(self, hero_reach, villain_gadget_reach, optout, villain_seat,
                   iters=500):
        """Safe re-solving via the re-solve (reach) gadget (Burch/Moravcik/Brown,
        "safe and nested subgame solving"). Phase 5a.

        The villain (`villain_seat`) is given a per-hand OPT-OUT paying `optout[h]`
        -- the blueprint's river-entry counterfactual value for villain hand h (from
        blueprint_projection.blueprint_cfv), in the SAME MEASURE units as the subgame
        values here (both weighted by the HERO's reach). Each iteration the villain
        regret-matches, per hand, between FOLLOW (enter the subgame) and TERMINATE
        (take the opt-out). So the villain only brings hands into the subgame that do
        at least as well as the blueprint already guaranteed them; the solver player
        (hero) must therefore make the subgame no better for the villain than the
        blueprint did -> the resulting HERO strategy is no-more-exploitable than the
        blueprint. Reads off via average_strategy as usual.

        hero_reach: the hero's FIXED river-entry reach (length H).
        villain_gadget_reach: the villain's river-entry reach -- the gadget WEIGHTING;
            the solved follow-probability modulates it each iteration (do NOT pre-scale).
        The Linear-CFR clock continues across calls, like run()."""
        hero_reach = np.asarray(hero_reach, float)
        vg = np.asarray(villain_gadget_reach, float)
        optout = np.asarray(optout, float)
        g_regret = np.zeros((self.H, 2))          # per villain hand: [Follow, Terminate]
        for _ in range(iters):
            self._iter += 1
            self._t_weight = float(self._iter)
            # villain gadget strategy (regret-matching+); uniform until regret accrues.
            gr = np.maximum(g_regret, 0.0)
            gs = gr.sum(axis=1, keepdims=True)
            gstrat = np.where(gs > 1e-12, gr / np.where(gs > 1e-12, gs, 1.0), 0.5)
            follow = gstrat[:, 0]
            villain_reach = vg * follow
            if villain_seat == 0:
                v0, v1 = self._cfr(self.tree.root, villain_reach, hero_reach)
                v_follow = v0
            else:
                v0, v1 = self._cfr(self.tree.root, hero_reach, villain_reach)
                v_follow = v1
            # gadget regret update (CFR+): villain maximizes Follow vs Terminate.
            node_val = follow * v_follow + gstrat[:, 1] * optout
            g_regret[:, 0] += v_follow - node_val
            g_regret[:, 1] += optout - node_val
            np.maximum(g_regret, 0.0, out=g_regret)
        return self

    # -- value + exploitability (convergence gap) ------------------------------
    def current_values(self, reach0, reach1):
        """(v0, v1) value vectors at the root under the CURRENT strategy."""
        return self._eval(self.tree.root, np.asarray(reach0, float),
                          np.asarray(reach1, float), self._strategy)

    def _eval(self, node, reach0, reach1, strat_fn):
        """Read-only value propagation under the strategy returned by strat_fn(nid)
        (self._strategy for current, self.average_strategy for the average). No
        regret/strategy updates."""
        if node.terminal:
            return self._terminal(node, reach0, reach1)
        nid = node.node_id
        p = node.player
        strat = strat_fn(nid)
        u = np.zeros((self.H, len(node.actions)))
        v_other = np.zeros(self.H)
        for a in range(len(node.actions)):
            if p == 0:
                cv0, cv1 = self._eval(node.children[a], reach0 * strat[:, a], reach1, strat_fn)
                u[:, a] = cv0
            else:
                cv0, cv1 = self._eval(node.children[a], reach0, reach1 * strat[:, a], strat_fn)
                u[:, a] = cv1
            v_other = v_other + (cv1 if p == 0 else cv0)
        v_p = (strat * u).sum(axis=1)
        return (v_p, v_other) if p == 0 else (v_other, v_p)

    def node_action_values(self, node, reach0, reach1):
        """Per-action value [H, A] for node.player at `node`, assuming BOTH players
        play the AVERAGE strategy from here on. reach0/reach1 are the ranges
        reaching `node`. MEASURE units (weighted by the opponent's reach); the
        acting player's own reach does not affect its own value (the value-passing
        invariant), so action a's value is just child_a's value for the actor.
        This is the EV gate's building block (compare solved vs baseline action)."""
        p = node.player
        reach0 = np.asarray(reach0, float)
        reach1 = np.asarray(reach1, float)
        vals = np.zeros((self.H, len(node.actions)))
        for a, child in enumerate(node.children):
            cv0, cv1 = self._eval(child, reach0, reach1, self.average_strategy)
            vals[:, a] = cv0 if p == 0 else cv1
        return vals

    def reach_into(self, edge_indices, reach0, reach1):
        """Reaches into the node reached by following `edge_indices` (child indices
        from the root), each player's root reach multiplied by their AVERAGE-strategy
        probability of the action taken at each node they own along the path. This is
        the reach the EV gate needs at a non-root node: a villain hand that would
        rarely take the realized line is correctly down-weighted, instead of using
        the (unconditioned) root reach. Mirrors the reach split in _eval/_cfr."""
        r0 = np.asarray(reach0, float).copy()
        r1 = np.asarray(reach1, float).copy()
        node = self.tree.root
        for a in edge_indices:
            strat = self.average_strategy(node.node_id)
            if node.player == 0:
                r0 = r0 * strat[:, a]
            else:
                r1 = r1 * strat[:, a]
            node = node.children[a]
        return r0, r1

    def warm_start(self, seed_strategies, weight):
        """Seed the average-strategy accumulator with `weight` worth of a prior
        strategy per node (e.g. the blueprint mapped onto the tree). An
        under-converged solve then degrades gracefully toward the prior; many
        iterations (linear weighting) wash it out. Does NOT seed regrets, so the
        SAFETY (graceful fallback) property is what this buys -- NOT speed (current
        play still starts uniform). seed_strategies: indexed by node_id, each an
        [H, A] row-stochastic array. Call once before run().

        NOTE: the seed weight is fixed, while real strategy-sum mass is reach-
        weighted, so the seed persists MORE at low-reach nodes (where the solve has
        little signal) and washes out at well-reached ones. That is intentional --
        it falls back toward the prior exactly where the solve is uninformed."""
        if weight <= 0:
            return
        for nid in range(len(self.nodes)):
            self.strat_sum[nid] += weight * np.asarray(seed_strategies[nid], float)

    def _br_value(self, node, br_player, reach_fixer, strat_fn):
        """br_player best-responds (max per hand); the other plays `strat_fn`.
        reach_fixer = the fixed player's reach into this node. Returns br_player's
        value vector."""
        if node.terminal:
            return self._terminal_one(node, br_player, reach_fixer)
        if node.player == br_player:
            vals = [self._br_value(c, br_player, reach_fixer, strat_fn) for c in node.children]
            return np.maximum.reduce(vals)
        strat = strat_fn(node.node_id)
        total = np.zeros(self.H)
        for a, c in enumerate(node.children):
            total = total + self._br_value(c, br_player, reach_fixer * strat[:, a], strat_fn)
        return total

    def exploitability(self, reach0, reach1, strat_fn=None):
        """Exploitability (chips per dealt hand-pair): BR0(σ1) + BR1(σ0) of the
        strategy σ = `strat_fn` (default: this solver's AVERAGE strategy -> the
        convergence gap). Pass a different strat_fn (e.g. the blueprint projected
        onto this tree) to measure how exploitable THAT strategy's river play is."""
        sf = strat_fn if strat_fn is not None else self.average_strategy
        reach0 = np.asarray(reach0, float)
        reach1 = np.asarray(reach1, float)
        br0 = self._br_value(self.tree.root, 0, reach1, sf)
        br1 = self._br_value(self.tree.root, 1, reach0, sf)
        Z = float((reach0 * compatible_mass(self.ba, reach1)).sum())
        if Z <= 0:
            return 0.0
        return (float((reach0 * br0).sum()) + float((reach1 * br1).sum())) / Z
