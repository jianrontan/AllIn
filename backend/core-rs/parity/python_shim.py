"""The parity CONTRACT implemented with the Python reference code.

Used with ALLIN_PARITY_TARGET=python to prove the harness + fixtures are
self-consistent (every test must pass against Python itself) and that the
contract is implementable. It is also executable documentation of what each
Rust binding has to do.
"""
import os
import sys

import numpy as np

_BOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bot'))
if _BOT not in sys.path:
    sys.path.insert(0, _BOT)

from src.evaluation.showdown_kernel import (  # noqa: E402
    _FULL_DECK, _NUM_CARDS, showdown_measure as _sd, compatible_mass as _cm)

__version__ = 'python-reference'
COMPONENTS = ('eval_rank7', 'buckets', 'keys', 'engine',
              'river_showdown', 'river_tree', 'river_cfr', 'river_gadget',
              'mccfr', 'mccfr_merge', 'keys_spr', 'mccfr_spr', 'buckets_v3')


def ported():
    return list(COMPONENTS)


def _cards(row):
    return [_FULL_DECK[int(i)] for i in row]


# -- eval ---------------------------------------------------------------------
def rank_batch(cards):
    from src.abstractions.postflop_features import rank7
    return np.array([rank7(_cards(r)) for r in cards], dtype=np.int32)


# -- buckets ------------------------------------------------------------------
_CA = None


def _ca():
    global _CA
    if _CA is None:
        from src.abstractions.card_abstractions import CardAbstraction
        _CA = CardAbstraction()
    return _CA


def preflop_buckets(cards):
    from src.abstractions.card_abstractions import FINE_TO_COARSE
    fine = np.array([int(_ca().preflop_bucket(_cards(r)).split('_')[1]) for r in cards],
                    dtype=np.int32)
    return fine, np.array([FINE_TO_COARSE[f] for f in fine], dtype=np.int32)


class Buckets:
    def __init__(self, abstractions_dir, river_cache_boards=None):
        from src.abstractions.abstraction_set import is_root
        from src.abstractions.postflop_v2 import PostflopV2
        self.dir = abstractions_dir
        # The root shares the cached CardAbstraction; a named abstraction gets its own.
        self._pv2 = None if is_root(abstractions_dir) else PostflopV2(abstraction_dir=abstractions_dir)

    def _postflop(self):
        return self._pv2 if self._pv2 is not None else _ca().postflop

    def n_buckets(self, street):
        return len(self._postflop()._centroid(('flop', 'turn', 'river')[street])[0])

    def situation_ids(self, cards):
        from src.abstractions.canonical import canonical_key
        from src.abstractions.postflop_features import encode_situation
        return np.array([encode_situation(canonical_key(tuple(_cards(r[:2])), tuple(_cards(r[2:]))))
                         for r in cards], dtype=np.int64)

    def bucket_batch(self, cards):
        pv2 = self._postflop()
        return np.array([pv2.bucket(_cards(r[:2]), _cards(r[2:])) for r in cards], dtype=np.int32)


# -- keys ---------------------------------------------------------------------
def make_info_set_key(street, position, preflop_bucket, strength, pattern, spr=None):
    from src.cfr.keys import make_info_set_key as mk
    return mk(street, position, preflop_bucket, strength, pattern, spr)


def spr_bucket(key_format, street, street_pot):
    from src.cfr.keys import spr_bucket as sb
    return sb(key_format, street, street_pot)


def action_char(action):
    from src.cfr.keys import action_char as ac
    return ac(action)


# -- engine -------------------------------------------------------------------
class Engine:
    def __init__(self, mode):
        from src.cfr.blueprint_trainer import BlueprintTrainer
        self.g = BlueprintTrainer(menu_mode=mode).game

    def init_node_state(self, street, starting_pot):
        return self.g.init_node_state(street, starting_pot)

    def acting_player(self, n, street):
        return self.g._acting_player(n, street)

    def legal_actions(self, street, st, cp, stack_cp):
        return self.g.state_legal_actions(street, st, cp, stack_cp)

    def action_cost(self, action, street, st, cp, stack_cp):
        return self.g.state_action_cost(action, street, st, cp, stack_cp)

    def advance(self, st, action, street, cp, stack_cp, p_inv):
        return self.g.advance_node_state(st, action, street, cp, stack_cp, p_inv)

    def is_terminal(self, history, street):
        return self.g.is_terminal(history, street)

    def utility(self, p0, p1, board, history, street, starting_pot, p0_inv, p1_inv,
                final_pot, p0_total):
        self.g._calc_cache.clear()
        return self.g.get_utility(p0, p1, board, history, street, starting_pot, p0_inv, p1_inv,
                                  _final_pot=final_pot, _p0_total=p0_total)


# -- river --------------------------------------------------------------------
def _ba(raw, c1, c2):
    """The subset of build_board_arrays the river solver reads, from raw ranks."""
    raw = np.asarray(raw, float)
    uniq = np.unique(raw)
    g = np.searchsorted(uniq, raw)
    return {'H': len(raw), 'raw': raw, 'c1': np.asarray(c1, np.int64),
            'c2': np.asarray(c2, np.int64), 'g': g, 'G': len(uniq),
            'gNC': g.astype(np.int64) * _NUM_CARDS}


def showdown_measure(raw, c1, c2, rv, final_pot, hero_total, exact=False):
    return _sd(_ba(raw, c1, c2), np.asarray(rv, float), final_pot, hero_total)


def compatible_mass(c1, c2, rv, exact=False):
    return _cm({'c1': np.asarray(c1, np.int64), 'c2': np.asarray(c2, np.int64)},
               np.asarray(rv, float))


def river_tree_nodes(pot_entry, stack, menu, max_aggressions):
    from src.subgame.river_tree import build_river_tree
    sys.path.insert(0, os.path.join(_BOT, 'scripts'))
    from export_rust_fixtures import _tree_to_list
    return _tree_to_list(build_river_tree(pot_entry, (stack, stack), menu=tuple(menu),
                                          max_aggressions=max_aggressions))


class RiverCFR:
    def __init__(self, pot_entry, stack, menu, max_aggressions, raw, c1, c2, exact=False):
        from src.subgame.river_tree import build_river_tree
        from src.subgame.river_cfr import RiverCFR as _R
        tree = build_river_tree(pot_entry, (stack, stack), menu=tuple(menu),
                                max_aggressions=max_aggressions)
        self._c = _R(tree, _ba(raw, c1, c2))
        self.n_decision = len(tree.decision_nodes)

    def run(self, r0, r1, iters):
        self._c.run(r0, r1, iters=iters)

    def run_gadget(self, hero, villain, optout, villain_seat, iters):
        self._c.run_gadget(hero, villain, optout, villain_seat, iters=iters)

    def average_strategy(self, nid):
        return self._c.average_strategy(nid)

    def regret(self, nid):
        return self._c.regret[nid]

    def current_values(self, r0, r1):
        return self._c.current_values(r0, r1)

    def exploitability(self, r0, r1):
        return self._c.exploitability(r0, r1)

    def gadget_regret(self):
        return self._c._g_regret


# -- mccfr --------------------------------------------------------------------
class Trainer:
    """The Python BlueprintTrainer behind the Rust Trainer contract: deals and
    the uniform stream are injected exactly as export_rust_fixtures records
    them (buckets are recomputed by Python, so fine/post/raw are unused)."""

    def __init__(self, menu_mode='control', alpha=1.5, gamma=1.0, key_format='v2'):
        from src.cfr.blueprint_trainer import BlueprintTrainer
        self._t = BlueprintTrainer(menu_mode=menu_mode, key_format=key_format)
        self._t.alpha, self._t.gamma = alpha, gamma

    @property
    def num_info_sets(self):
        return len(self._t.info_sets)

    @property
    def ev_sum(self):
        return self._t.ev_sum

    @property
    def ev_count(self):
        return self._t.ev_count

    def run_fixture(self, deals, fine, post, raw, fine_to_coarse, uniforms, start_iteration=0):
        import random
        import src.cfr.blueprint_trainer as bt

        class _Replay(random.Random):
            pos = 0

            def random(self):
                v = uniforms[self.pos]
                self.pos += 1
                return v

        cards = [[_FULL_DECK[i] for i in d] for d in deals]
        feed = iter(cards)
        self._t.deal_random_hand = lambda: (lambda d: (d[0:2], d[2:4], d[4:9]))(next(feed))
        stream, saved = _Replay(0), bt.random
        bt.random = stream
        try:
            for i in range(len(cards)):
                self._t._run_iteration(start_iteration + i)
        finally:
            bt.random = saved
        return stream.pos

    def checkpoint(self, db_path, iteration):
        from src.storage.blueprint_db import BlueprintDB
        db = BlueprintDB(db_path)
        n = len(self._t._dirty)
        try:
            self._t.checkpoint_to_db(db, iteration)
        finally:
            db.close()
        return n


def merge_round(glob, workers, alpha, gamma):
    from src.cfr.information_set import InformationSet
    from src.cfr.parallel_trainer import merge_round as _merge
    info = {}
    for k, v in glob.items():
        i = InformationSet()
        i.legal_actions = list(v['legal'])
        i.cumulative_regrets = dict(v['regret'])
        i.cumulative_strategy = dict(v['strategy'])
        i.visit_count = v['visit_count']
        i.strategy_visit_count = v['strategy_visit_count']
        info[k] = i
    touched = _merge(info, workers, alpha, gamma)
    return ({k: {'legal': i.legal_actions, 'regret': i.cumulative_regrets,
                 'strategy': i.cumulative_strategy, 'visit_count': i.visit_count,
                 'strategy_visit_count': i.strategy_visit_count} for k, i in info.items()},
            sorted(touched))
