"""Rust iteration engine over the exact Python serving tree.

Read-only EV and safety walks reuse the reference implementation and a cached
snapshot of Rust's average strategy. No second betting tree is reconstructed.
"""
import numpy as np

from .river_cfr import RiverCFR


def serialize_tree(tree):
    nodes = []

    def visit(node):
        index = len(nodes)
        row = {'terminal': node.terminal}
        nodes.append(row)
        if node.terminal:
            row.update(final_pot=node.final_pot, contrib=node.contrib, folder=node.folder)
        else:
            row.update(player=node.player, node_id=node.node_id, actions=list(node.actions),
                       children=[visit(child) for child in node.children])
        return index

    visit(tree.root)
    return nodes


class RustRiverCFR(RiverCFR):
    def __init__(self, tree, ba, *, exact=False, algorithm="cfrplus"):
        from allin_core import RiverCFR as NativeRiverCFR
        self.tree, self.ba, self.H = tree, ba, ba['H']
        self.nodes = tree.decision_nodes
        self._native = NativeRiverCFR.from_tree(
            tree.pot_entry, serialize_tree(tree),
            np.asarray(ba['raw'], dtype=float), np.asarray(ba['c1'], dtype=np.int64),
            np.asarray(ba['c2'], dtype=np.int64), exact=exact)
        if algorithm == 'dcfr':
            self._native.enable_dcfr()
        elif algorithm != 'cfrplus':
            raise ValueError(f"Unknown river algorithm: {algorithm!r}")
        self._averages = {}

    @property
    def _iter(self):
        return self._native.iterations

    def average_strategy(self, nid):
        if nid not in self._averages:
            self._averages[nid] = self._native.average_strategy(nid)
        return self._averages[nid]

    def _strategy(self, nid):
        r = np.maximum(self._native.regret(nid), 0.)
        total = r.sum(axis=1, keepdims=True)
        return np.where(total > 1e-12, r / np.where(total > 1e-12, total, 1), 1 / r.shape[1])

    def run(self, reach0, reach1, iters=500):
        self._native.run(np.asarray(reach0, float), np.asarray(reach1, float), iters)
        self._averages.clear()

    def run_gadget(self, hero_reach, villain_gadget_reach, optout, villain_seat, iters=500):
        self._native.run_gadget(np.asarray(hero_reach, float),
                                np.asarray(villain_gadget_reach, float),
                                np.asarray(optout, float), villain_seat, iters)
        self._averages.clear()

    def warm_start(self, seed_strategies, weight):
        self._native.warm_start([np.asarray(seed_strategies[i], float)
                                 for i in range(len(self.nodes))], weight)
        self._averages.clear()

    def current_values(self, reach0, reach1):
        return self._native.current_values(np.asarray(reach0, float), np.asarray(reach1, float))

    def exploitability(self, reach0, reach1, strat_fn=None):
        if strat_fn is not None:
            return super().exploitability(reach0, reach1, strat_fn)
        return self._native.exploitability(np.asarray(reach0, float), np.asarray(reach1, float))
