"""Offline DCFR(1.5,0,2) experiment; never selected by serving.

Brown/Sandholm 2019: https://arxiv.org/abs/1809.04040.
Regrets discounted after each alternating player update; t**2 averaging.
"""
import json
import numpy as np
from bench_baseline import BOARD, RIVER_CONFIGS
from src.subgame.river_cfr import RiverCFR
from src.subgame.river_tree import build_river_tree
from src.abstractions.card_abstractions import CardAbstraction
from src.abstractions.hand_evaluator import HandEvaluator
from src.evaluation.showdown_kernel import build_board_arrays


class DCFR(RiverCFR):
    def _strategy(self, nid):
        r = np.maximum(self.regret[nid], 0.)
        total = r.sum(axis=1, keepdims=True)
        return np.where(total > 1e-12, r / np.where(total > 1e-12, total, 1.), 1. / r.shape[1])

    def _cfr(self, node, r0, r1):
        if node.terminal:
            return self._terminal(node, r0, r1)
        strat = self._strategy(node.node_id)
        u = np.zeros((self.H, len(node.actions)))
        other = np.zeros(self.H)
        for a, child in enumerate(node.children):
            if node.player == 0:
                v0, v1 = self._cfr(child, r0 * strat[:, a], r1)
                u[:, a], other = v0, other + v1
            else:
                v0, v1 = self._cfr(child, r0, r1 * strat[:, a])
                u[:, a], other = v1, other + v0
        value = (strat * u).sum(axis=1)
        if node.player == self.updating:
            self.regret[node.node_id] += u - value[:, None]
            reach = r0 if node.player == 0 else r1
            self.strat_sum[node.node_id] += reach[:, None] * strat * self._iter ** 2
        return (value, other) if node.player == 0 else (other, value)

    def run(self, r0, r1, iters=500):
        for _ in range(iters):
            self._iter += 1
            t = self._iter ** 1.5
            for self.updating in (0, 1):
                self._cfr(self.tree.root, r0, r1)
                for node in self.nodes:
                    if node.player == self.updating:
                        r = self.regret[node.node_id]
                        r *= np.where(r >= 0., t / (t + 1.), .5)
        return self

    def run_gadget(self, hero, vg, optout, villain_seat, iters=500):
        if self._g_regret is None:
            self._g_regret = np.zeros((self.H, 2))
        for _ in range(iters):
            self._iter += 1
            t = self._iter ** 1.5
            for self.updating in (0, 1):
                positive = np.maximum(self._g_regret, 0.)
                total = positive.sum(axis=1, keepdims=True)
                strat = np.where(total > 1e-12, positive / np.where(total > 1e-12, total, 1.), .5)
                villain = vg * strat[:, 0]
                reaches = (villain, hero) if villain_seat == 0 else (hero, villain)
                values = self._cfr(self.tree.root, *reaches)[villain_seat]
                for node in self.nodes:
                    if node.player == self.updating:
                        r = self.regret[node.node_id]
                        r *= np.where(r >= 0., t / (t + 1.), .5)
                if self.updating == villain_seat:
                    value = strat[:, 0] * values + strat[:, 1] * optout
                    self._g_regret += np.column_stack((values - value, optout - value))
                    self._g_regret *= np.where(self._g_regret >= 0., t / (t + 1.), .5)
        return self


if __name__ == '__main__':
    ba = build_board_arrays(BOARD, HandEvaluator(), CardAbstraction(), streets=(3,))
    rng = np.random.default_rng(371)
    r0, r1 = rng.random((2, ba['H'])) + .1
    for pot, stack in RIVER_CONFIGS:
        tree = build_river_tree(pot, (stack, stack))
        for cls in (RiverCFR, DCFR):
            solver = cls(tree, ba)
            previous = 0
            for checkpoint in (50, 100, 150, 275):
                solver.run(r0, r1, checkpoint - previous)
                previous = checkpoint
                print(json.dumps(dict(algorithm=cls.__name__, pot=pot, iterations=checkpoint,
                                      traversals=checkpoint * (2 if cls is DCFR else 1),
                                      gap=solver.exploitability(r0, r1))), flush=True)
