# backend/bot/scripts/export_rust_fixtures.py
"""
Export golden fixtures from the PYTHON reference implementation, for the Rust
port in backend/core-rs (see docs/RUST_CORE_PLAN.md).

The Python code is the oracle: every Rust component must reproduce these outputs
(exactly for integer/discrete things, to a tight float tolerance for the river
solver). The parity harness (backend/core-rs/parity/) loads these files.

Fixtures are DETERMINISTIC given --seed and the current Python code/abstraction,
and are git-ignored (regenerate, don't commit). manifest.json records the commit,
library versions and centroid hashes they were produced under; the harness warns
if they look stale.

Card-id convention (all fixtures): id = rank_index * 4 + suit_index, with ranks
'23456789TJQKA' and suits 'HDCS' -- i.e. the index into showdown_kernel._FULL_DECK.
Engine card strings are SuitRank ('HA' = ace of hearts).

Run from backend/bot/:
    python scripts/export_rust_fixtures.py                       # all components
    python scripts/export_rust_fixtures.py --only river_cfr,keys
    python scripts/export_rust_fixtures.py --quick               # smaller, for a fast loop
"""
import argparse
import datetime
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
sys.path.insert(0, BOT)

from src.evaluation.showdown_kernel import (  # noqa: E402
    _FULL_DECK, _CARD_ID, build_board_arrays, showdown_measure, compatible_mass)

COMPONENTS = ('eval_rank7', 'buckets', 'keys', 'engine',
              'river_showdown', 'river_tree', 'river_cfr', 'river_gadget',
              'mccfr', 'mccfr_merge', 'keys_spr', 'mccfr_spr')
DEFAULT_OUT = os.path.normpath(os.path.join(BOT, '..', 'core-rs', 'fixtures'))


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _ids(cards):
    return [_CARD_ID[c] for c in cards]


def _deal(rng, n):
    """n distinct engine cards, uniformly at random."""
    idx = rng.choice(52, size=n, replace=False)
    return [_FULL_DECK[i] for i in idx]


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))


def _git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BOT,
                                       text=True).strip()
    except Exception:
        return None


def _git_dirty():
    try:
        out = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src'],
                                      cwd=BOT, text=True)
        return bool(out.strip())
    except Exception:
        return None


# ----------------------------------------------------------------------------
# eval_rank7 -- showdown ranks (phevaluator score, lower = stronger)
# ----------------------------------------------------------------------------
def export_eval_rank7(out, rng, quick):
    from src.abstractions.postflop_features import rank7
    n = 20_000 if quick else 200_000
    res = {}
    for k in (5, 6, 7):
        m = n if k == 7 else n // 4
        cards = np.empty((m, k), dtype=np.uint8)
        ranks = np.empty(m, dtype=np.int32)
        for i in range(m):
            cs = _deal(rng, k)
            cards[i] = _ids(cs)
            ranks[i] = rank7(cs)
        res[f'cards{k}'] = cards
        res[f'ranks{k}'] = ranks
    np.savez_compressed(os.path.join(out, 'eval_rank7.npz'), **res)
    return {k: int(v.shape[0]) for k, v in res.items() if k.startswith('ranks')}


# ----------------------------------------------------------------------------
# buckets -- preflop fine/coarse + postflop (flop/turn table lookup, river runtime)
# ----------------------------------------------------------------------------
def export_buckets(out, rng, quick):
    from itertools import combinations
    from src.abstractions.card_abstractions import CardAbstraction, FINE_TO_COARSE
    from src.abstractions.canonical import canonical_key
    from src.abstractions.postflop_features import encode_situation
    ca = CardAbstraction()
    pv2 = ca.postflop

    # Preflop: exhaustive over all 1326 combos.
    combos = list(combinations(range(52), 2))
    pf_cards = np.array(combos, dtype=np.uint8)
    pf_fine = np.empty(len(combos), dtype=np.int32)
    pf_coarse = np.empty(len(combos), dtype=np.int32)
    for i, (a, b) in enumerate(combos):
        fine = ca.preflop_bucket([_FULL_DECK[a], _FULL_DECK[b]])
        pf_fine[i] = int(fine.split('_')[1])
        pf_coarse[i] = FINE_TO_COARSE[pf_fine[i]]     # tuple indexed by fine id

    res = {'pf_cards': pf_cards, 'pf_fine': pf_fine, 'pf_coarse': pf_coarse}
    counts = {'preflop': len(combos)}
    # flop/turn are baked-table lookups (cheap); river is runtime equity -> fewer,
    # and several holes per board so the per-board cache path is exercised.
    plan = {'flop': (3, 2_000 if quick else 20_000, 1),
            'turn': (4, 2_000 if quick else 20_000, 1),
            'river': (5, 40 if quick else 300, 8)}
    for street, (nb, nboards, per_board) in plan.items():
        rows, buckets, sids = [], [], []
        for _ in range(nboards):
            cs = _deal(rng, nb + 2 * per_board)
            board = cs[:nb]
            for j in range(per_board):
                hole = cs[nb + 2 * j: nb + 2 * j + 2]
                rows.append(_ids(hole + board))
                buckets.append(pv2.bucket(hole, board))
                sids.append(int(encode_situation(canonical_key(tuple(hole), tuple(board)))))
        res[f'{street}_cards'] = np.array(rows, dtype=np.uint8)   # [hole0, hole1, board...]
        res[f'{street}_bucket'] = np.array(buckets, dtype=np.int32)
        res[f'{street}_sid'] = np.array(sids, dtype=np.int64)
        counts[street] = len(rows)
    np.savez_compressed(os.path.join(out, 'buckets.npz'), **res)
    return counts


# ----------------------------------------------------------------------------
# keys -- info-set key strings + action chars
# ----------------------------------------------------------------------------
def export_keys(out, rng, quick):
    from src.cfr.keys import make_info_set_key, action_char, ACTION_CHARS
    from src.abstractions.card_abstractions import NUM_PREFLOP_BUCKETS
    chars = sorted(set(ACTION_CHARS.values()))
    cases = []
    n = 2_000 if quick else 20_000
    for _ in range(n):
        street = int(rng.integers(0, 4))
        pos = 'ip' if rng.random() < 0.5 else 'oop'
        fine = f"pf_{int(rng.integers(0, NUM_PREFLOP_BUCKETS))}"
        strength = None if street == 0 else int(rng.integers(0, 30))
        pat = ''.join(rng.choice(chars) for _ in range(int(rng.integers(0, 6))))
        cases.append({'street': street, 'position': pos, 'preflop_bucket': fine,
                      'strength': strength, 'pattern': pat,
                      'key': make_info_set_key(street, pos, fine, strength, pat)})
    _write_json(os.path.join(out, 'keys.json'), {
        'action_chars': {a: action_char(a) for a in ACTION_CHARS},
        'cases': cases})
    return {'cases': len(cases), 'actions': len(ACTION_CHARS)}


# ----------------------------------------------------------------------------
# engine -- the state-threaded betting engine the trainer runs (Lever A path)
# ----------------------------------------------------------------------------
def _engine_games(mode, n_games, seed):
    """Random walks mirroring BlueprintTrainer.cfr's control flow, recording every
    node's inputs and outputs. Floats are JSON-exact (repr round-trips)."""
    from src.cfr.blueprint_trainer import BlueprintTrainer
    from src.cfr.poker_game import STARTING_STACK
    g = BlueprintTrainer(menu_mode=mode).game
    r = random.Random(seed)
    games = []
    for _ in range(n_games):
        g._calc_cache.clear()
        cs = r.sample(_FULL_DECK, 9)
        p0c, p1c, comm = cs[0:2], cs[2:4], cs[4:9]
        street, starting_pot = 0, 3.0
        p_inv = [0.0, 0.0]
        stacks = [STARTING_STACK - 1, STARTING_STACK - 2]
        history = []
        st = g.init_node_state(0, starting_pot)
        nodes = []
        terminal = None
        for _guard in range(80):
            if g.is_terminal(history, street):
                terminal = {'street': street}
                break
            cp = g._acting_player(len(history), street)
            stack_cp = stacks[cp]
            legal = g.state_legal_actions(street, st, cp, stack_cp)
            if not legal:
                if street < 3:
                    p_inv = [p_inv[0] + st['c'][0], p_inv[1] + st['c'][1]]
                    stacks = [STARTING_STACK - p_inv[0], STARTING_STACK - p_inv[1]]
                    starting_pot = st['pot']
                    street += 1
                    history = []
                    st = g.init_node_state(street, starting_pot)
                    nodes.append({'transition_to': street, 'p_inv': list(p_inv),
                                  'stacks': list(stacks), 'starting_pot': starting_pot,
                                  'st': st})
                    continue
                terminal = {'street': street}
                break
            per_action = []
            for a in legal:
                cost = g.state_action_cost(a, street, st, cp, stack_cp)
                child = g.advance_node_state(st, a, street, cp, stack_cp, tuple(p_inv))
                per_action.append({'action': a, 'cost': cost, 'child': child})
            # Bias away from folds (taken 10% of the time when legal) so walks
            # reach flop/turn/river betting instead of ending preflop.
            non_fold = [i for i, x in enumerate(legal) if x != 'fold']
            if non_fold and (len(non_fold) == len(legal) or r.random() >= 0.10):
                pick = r.choice(non_fold)
            else:
                pick = r.randrange(len(legal))
            a = legal[pick]
            nodes.append({'street': street, 'history': list(history), 'cp': cp,
                          'stack_cp': stack_cp, 'p_inv': list(p_inv), 'st': st,
                          'legal': legal, 'per_action': per_action, 'pick': pick})
            stacks[cp] -= per_action[pick]['cost']
            st = per_action[pick]['child']
            history = history + [a]
        if terminal is None:
            raise RuntimeError(f"engine walk did not terminate: {history}")
        s = min(terminal['street'], 3)
        terminal['utility_p0'] = g.get_utility(
            p0c, p1c, comm, history, s, starting_pot, p_inv[0], p_inv[1],
            _final_pot=st['pot'], _p0_total=p_inv[0] + st['c'][0])
        games.append({'p0_cards': p0c, 'p1_cards': p1c, 'board': comm,
                      'nodes': nodes, 'final_history': history,
                      'final_st': st, 'terminal': terminal})
    return games


def export_engine(out, rng, quick):
    n = 300 if quick else 3_000
    counts = {}
    for k, mode in enumerate(('control', 'capped')):
        games = _engine_games(mode, n, seed=int(rng.integers(1 << 31)) + k)
        _write_json(os.path.join(out, f'engine_{mode}.json'),
                    {'mode': mode, 'games': games})
        counts[mode] = {'games': len(games),
                        'nodes': sum(len(gm['nodes']) for gm in games)}
    return counts


# ----------------------------------------------------------------------------
# river -- shared case generation
# ----------------------------------------------------------------------------
def _river_cases(rng, n_boards):
    """(board, raw, c1, c2) for n random river boards."""
    from src.abstractions.hand_evaluator import HandEvaluator
    from src.abstractions.card_abstractions import CardAbstraction
    ev, ca = HandEvaluator(), CardAbstraction()
    out = []
    for _ in range(n_boards):
        board = _deal(rng, 5)
        ba = build_board_arrays(board, ev, ca, streets=(3,))
        out.append((board, ba))
    return out


def _reach(rng, H, kind):
    if kind == 'uniform':
        return np.ones(H)
    if kind == 'random':
        return rng.random(H)
    if kind == 'sparse':                  # many exact zeros (blocked / folded-out hands)
        r = rng.random(H)
        r[rng.random(H) < 0.7] = 0.0
        return r
    raise ValueError(kind)


def _ba_arrays(prefix, board, ba):
    return {f'{prefix}board': np.array(_ids(board), dtype=np.uint8),
            f'{prefix}raw': ba['raw'], f'{prefix}c1': ba['c1'], f'{prefix}c2': ba['c2'],
            f'{prefix}g': ba['g'], f'{prefix}G': np.array(ba['G'])}


def _tree_to_list(tree):
    """Pre-order serialization; children are indices into the returned list."""
    nodes = []

    def walk(n):
        i = len(nodes)
        rec = {'terminal': n.terminal}
        nodes.append(rec)
        if n.terminal:
            rec.update(final_pot=n.final_pot, contrib=list(n.contrib), folder=n.folder)
            return i
        rec.update(player=n.player, to_call=n.to_call, pot_mid=n.pot_mid,
                   sc=list(n.sc), actions=list(n.actions), node_id=n.node_id,
                   agg=n.agg, last_inc=n.last_inc)
        rec['children'] = [walk(c) for c in n.children]
        return i

    walk(tree.root)
    return nodes


# Tree configs: (pot_entry, stack_behind). Covers deep / normal / short / micro
# stacks (micro forces all-in collapse) and odd pots (non-round sizing).
_TREE_CONFIGS = [(6.0, 197.0), (12.0, 194.0), (30.0, 185.0), (45.5, 177.25),
                 (80.0, 160.0), (150.0, 125.0), (200.0, 100.0), (300.0, 50.0),
                 (390.0, 5.0), (396.0, 2.0), (398.0, 1.0), (7.0, 196.5),
                 (100.0, 150.0), (250.0, 75.0), (20.0, 190.0), (64.0, 168.0)]


def export_river_tree(out, rng, quick):
    from src.subgame.river_tree import build_river_tree, DEFAULT_MENU, MAX_AGGRESSIONS
    cases = []
    for pot, stack in _TREE_CONFIGS:
        t = build_river_tree(pot_entry=pot, stacks=(stack, stack))
        cases.append({'pot_entry': pot, 'stack': stack, 'menu': list(DEFAULT_MENU),
                      'max_aggressions': MAX_AGGRESSIONS,
                      'n_decision': len(t.decision_nodes), 'nodes': _tree_to_list(t)})
    _write_json(os.path.join(out, 'river_tree.json'), {'cases': cases})
    return {'trees': len(cases),
            'decision_nodes': [c['n_decision'] for c in cases]}


def export_river_showdown(out, rng, quick):
    boards = _river_cases(rng, 4 if quick else 20)
    res = {}
    n = 0
    for b, (board, ba) in enumerate(boards):
        res.update(_ba_arrays(f'b{b}_', board, ba))
        for k, kind in enumerate(('uniform', 'random', 'sparse')):
            rv = _reach(rng, ba['H'], kind)
            fp, ht = float(rng.uniform(4, 400)), float(rng.uniform(1, 200))
            res[f'b{b}_k{k}_rv'] = rv
            res[f'b{b}_k{k}_args'] = np.array([fp, ht])
            res[f'b{b}_k{k}_showdown'] = showdown_measure(ba, rv, fp, ht)
            res[f'b{b}_k{k}_compat'] = compatible_mass(ba, rv)
            n += 1
    res['n_boards'] = np.array(len(boards))
    np.savez_compressed(os.path.join(out, 'river_showdown.npz'), **res)
    return {'boards': len(boards), 'cases': n}


def _cfr_snapshot(prefix, cfr, r0, r1):
    nn = len(cfr.nodes)
    return {f'{prefix}strat': np.concatenate([cfr.average_strategy(i).ravel() for i in range(nn)]),
            f'{prefix}regret': np.concatenate([cfr.regret[i].ravel() for i in range(nn)]),
            f'{prefix}v0': cfr.current_values(r0, r1)[0],
            f'{prefix}v1': cfr.current_values(r0, r1)[1],
            f'{prefix}expl': np.array(cfr.exploitability(r0, r1))}


# CFR cases: (tree config index, reach kinds). Kept to small/medium trees so the
# fixture files stay ~tens of MB; checkpoints show divergence early AND late.
_CFR_CHECKPOINTS = (1, 10, 50)


def export_river_cfr(out, rng, quick):
    from src.subgame.river_tree import build_river_tree
    from src.subgame.river_cfr import RiverCFR
    cfgs = [(30.0, 185.0, 'uniform', 'uniform'), (80.0, 160.0, 'random', 'random'),
            (12.0, 194.0, 'random', 'sparse'), (200.0, 100.0, 'sparse', 'random'),
            (390.0, 5.0, 'uniform', 'random'), (45.5, 177.25, 'random', 'random')]
    if quick:
        cfgs = cfgs[:2]
    boards = _river_cases(rng, len(cfgs))
    meta = []
    for i, ((pot, stack, k0, k1), (board, ba)) in enumerate(zip(cfgs, boards)):
        tree = build_river_tree(pot_entry=pot, stacks=(stack, stack))
        H = ba['H']
        r0, r1 = _reach(rng, H, k0), _reach(rng, H, k1)
        res = _ba_arrays('', board, ba)
        res.update(r0=r0, r1=r1,
                   shapes=np.array([len(n.actions) for n in tree.decision_nodes]))
        cfr = RiverCFR(tree, ba)
        done = 0
        for cp in _CFR_CHECKPOINTS:
            cfr.run(r0, r1, iters=cp - done)
            done = cp
            res.update(_cfr_snapshot(f'it{cp}_', cfr, r0, r1))
        np.savez_compressed(os.path.join(out, f'river_cfr_{i}.npz'), **res)
        meta.append({'file': f'river_cfr_{i}.npz', 'pot_entry': pot, 'stack': stack,
                     'reach': [k0, k1], 'H': H, 'n_decision': len(tree.decision_nodes)})
    _write_json(os.path.join(out, 'river_cfr.json'),
                {'checkpoints': list(_CFR_CHECKPOINTS), 'cases': meta})
    return {'cases': len(meta)}


def export_river_gadget(out, rng, quick):
    from src.subgame.river_tree import build_river_tree
    from src.subgame.river_cfr import RiverCFR
    cfgs = [(30.0, 185.0, 0), (80.0, 160.0, 1), (150.0, 125.0, 1), (12.0, 194.0, 0)]
    if quick:
        cfgs = cfgs[:2]
    boards = _river_cases(rng, len(cfgs))
    meta = []
    for i, ((pot, stack, vseat), (board, ba)) in enumerate(zip(cfgs, boards)):
        tree = build_river_tree(pot_entry=pot, stacks=(stack, stack))
        H = ba['H']
        hero = _reach(rng, H, 'random')
        vill = _reach(rng, H, 'random')
        # Opt-out values in the same MEASURE units as the subgame values (weighted
        # by the hero's reach): scale a per-hand payoff by the compatible hero mass.
        optout = rng.uniform(-0.3, 0.6, H) * pot * compatible_mass(ba, hero)
        res = _ba_arrays('', board, ba)
        res.update(hero=hero, villain=vill, optout=optout,
                   shapes=np.array([len(n.actions) for n in tree.decision_nodes]))
        cfr = RiverCFR(tree, ba)
        r0, r1 = (vill, hero) if vseat == 0 else (hero, vill)
        done = 0
        for cp in _CFR_CHECKPOINTS:
            cfr.run_gadget(hero, vill, optout, vseat, iters=cp - done)
            done = cp
            res.update(_cfr_snapshot(f'it{cp}_', cfr, r0, r1))
            res[f'it{cp}_g_regret'] = cfr._g_regret.copy()
        np.savez_compressed(os.path.join(out, f'river_gadget_{i}.npz'), **res)
        meta.append({'file': f'river_gadget_{i}.npz', 'pot_entry': pot, 'stack': stack,
                     'villain_seat': vseat, 'H': H,
                     'n_decision': len(tree.decision_nodes)})
    _write_json(os.path.join(out, 'river_gadget.json'),
                {'checkpoints': list(_CFR_CHECKPOINTS), 'cases': meta})
    return {'cases': len(meta)}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# mccfr -- single-thread BlueprintTrainer under injected randomness
# ----------------------------------------------------------------------------
class _StreamRandom(random.Random):
    """random.Random whose random() replays (and records) a numpy uniform
    stream. Random.choices calls self.random(), so the trainer's opponent
    sampling consumes exactly this stream. numpy's random() is k * 2**-53, like
    CPython's, and JSON round-trips it exactly."""

    def __init__(self, rng):
        super().__init__(0)
        self._rng = rng
        self.drawn = []

    def random(self):
        v = float(self._rng.random())
        self.drawn.append(v)
        return v


def _mccfr_run(out, rng, mode, gamma, n_iter, key_format='v2'):
    import src.cfr.blueprint_trainer as bt
    from src.abstractions.card_abstractions import FINE_TO_COARSE
    from src.abstractions.postflop_features import rank7
    from src.storage.blueprint_db import BlueprintDB

    deals = [_deal(rng, 9) for _ in range(n_iter)]
    trainer = bt.BlueprintTrainer(menu_mode=mode, key_format=key_format)
    trainer.gamma = gamma
    ca = trainer.game_adapter.card_abstractions
    feed = iter(deals)
    trainer.deal_random_hand = lambda: (lambda d: (d[0:2], d[2:4], d[4:9]))(next(feed))
    stream = _StreamRandom(rng)
    saved = bt.random
    bt.random = stream
    try:
        for i in range(n_iter):
            trainer._run_iteration(i)
    finally:
        bt.random = saved
    kf = '' if key_format == 'v2' else f'_{key_format}'
    name = f'mccfr_{mode}{kf}_g{gamma:g}'
    db_path = os.path.join(out, f'{name}.db')
    for suffix in ('', '-wal', '-shm'):
        if os.path.exists(db_path + suffix):
            os.remove(db_path + suffix)
    db = BlueprintDB(db_path)
    trainer.checkpoint_to_db(db, n_iter - 1)
    db.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    # Everything the Rust side needs to replay the run without the bucketing
    # component: per-deal preflop fine buckets, postflop buckets per street, and
    # the two 7-card ranks (every trainer showdown uses the full board).
    rows = []
    for d in deals:
        holes, board = (d[0:2], d[2:4]), d[4:9]
        rows.append({
            'cards': _ids(d),
            'fine': [int(ca.get_bucket(h, None).split('_')[1]) for h in holes],
            'post': [[int(ca.get_bucket(h, board[:n])) for n in (3, 4, 5)] for h in holes],
            'raw': [int(rank7(list(h) + list(board))) for h in holes],
        })
    _write_json(os.path.join(out, f'{name}.json'), {
        'mode': mode, 'key_format': key_format,
        'alpha': trainer.alpha, 'gamma': gamma, 'iterations': n_iter,
        'fine_to_coarse': list(FINE_TO_COARSE), 'deals': rows, 'uniforms': stream.drawn,
        'ev_sum': trainer.ev_sum, 'ev_count': trainer.ev_count,
        'info_sets': len(trainer.info_sets), 'db': os.path.basename(db_path)})
    return {'iterations': n_iter, 'info_sets': len(trainer.info_sets),
            'uniforms': len(stream.drawn)}


def export_mccfr(out, rng, quick):
    n = 60 if quick else 600
    return {f'{mode}_g{gamma:g}': _mccfr_run(out, rng, mode, gamma, n)
            for mode, gamma in (('control', 1.0), ('capped', 1.0), ('control', 2.0))}


def export_mccfr_spr(out, rng, quick):
    n = 60 if quick else 600
    return {'capped_spr5_g1': _mccfr_run(out, rng, 'capped', 1.0, n, key_format='spr5')}


# ----------------------------------------------------------------------------
# keys_spr -- spr_bucket + spr5 key strings (Phase 6)
# ----------------------------------------------------------------------------
def export_keys_spr(out, rng, quick):
    from src.cfr.keys import make_info_set_key, spr_bucket, SPR_EDGES
    from src.abstractions.card_abstractions import NUM_PREFLOP_BUCKETS
    # Pots exactly at, and one ulp around, every bucket edge (SPR = 200/pot - 0.5
    # => edge pot = 200 / (edge + 0.5)), plus random realistic pots.
    pots = []
    for e in SPR_EDGES:
        p = 200.0 / (e + 0.5)
        pots += [p, float(np.nextafter(p, 0)), float(np.nextafter(p, 1e9))]
    pots += [3.0, 4.0, 8.0, 10.0, 16.0, 80.0, 400.0, 399.5, 0.5]
    n = 2_000 if quick else 20_000
    pots += [float(x) for x in rng.uniform(2.0, 400.0, n)]
    cases = []
    for i, pot in enumerate(pots):
        for kf in ('v2', 'spr5'):
            street = int(rng.integers(0, 4))
            b = spr_bucket(kf, street, pot)
            pos = 'ip' if rng.random() < 0.5 else 'oop'
            fine = f"pf_{int(rng.integers(0, NUM_PREFLOP_BUCKETS))}"
            strength = None if street == 0 else int(rng.integers(0, 30))
            pat = ''.join(rng.choice(list('kcfsmlxo2a')) for _ in range(int(rng.integers(0, 6))))
            cases.append({'key_format': kf, 'street': street, 'street_pot': pot, 'spr': b,
                          'position': pos, 'preflop_bucket': fine, 'strength': strength,
                          'pattern': pat,
                          'key': make_info_set_key(street, pos, fine, strength, pat, b)})
    _write_json(os.path.join(out, 'keys_spr.json'), {'cases': cases})
    return {'cases': len(cases)}


# ----------------------------------------------------------------------------
# mccfr_merge -- parallel_trainer.merge_round on synthetic worker deltas
# ----------------------------------------------------------------------------
def _rand_map(rng, actions, p_missing=0.25, neg=False):
    out = {}
    for a in actions:
        if rng.random() < p_missing:
            continue
        v = float(rng.random() * 10.0 ** int(rng.integers(-2, 3)))
        out[a] = -v if (neg and rng.random() < 0.3) else v
    return out


def export_mccfr_merge(out, rng, quick):
    """merge_round is a pure function of (global info sets, worker deltas,
    alpha, gamma); record inputs and outputs on random cases incl. keys absent
    from the global table, actions absent from the baseline, regret increments
    that go negative (master floor), clocks at t=1 and gamma=0."""
    from src.cfr.information_set import InformationSet
    from src.cfr.parallel_trainer import merge_round
    from src.cfr.keys import ACTION_CHARS
    names = list(ACTION_CHARS)
    cases = []
    for ci in range(20 if quick else 200):
        keys = [f'pf_{ci}_{k}_ip_flop_' for k in range(int(rng.integers(1, 8)))]
        # One action pool per key (a real node offers at most 8 actions; the
        # Rust table caps an info set at 10 and errors beyond that).
        pool = {k: list(rng.choice(names, size=int(rng.integers(2, 9)), replace=False))
                for k in keys}
        glob = {}
        for k in keys:
            if rng.random() < 0.3:
                continue                      # brand-new key this round
            acts = pool[k]
            info = InformationSet()
            info.legal_actions = acts
            info.cumulative_regrets = _rand_map(rng, acts)
            info.cumulative_strategy = _rand_map(rng, acts)
            info.visit_count = int(rng.integers(0, 4))
            info.strategy_visit_count = int(rng.integers(0, 4))
            glob[k] = info
        workers = []
        for _w in range(int(rng.integers(1, 5))):
            res = {'regret': {}, 'strategy': {}, 'legal': {}}
            for k in keys:
                if rng.random() < 0.4:
                    continue
                acts = (glob[k].legal_actions if k in glob else
                        [a for a in pool[k] if rng.random() < 0.8] or pool[k][:1])
                res['legal'][k] = list(acts)
                if rng.random() < 0.7:
                    res['regret'][k] = _rand_map(rng, acts)
                if rng.random() < 0.7:
                    res['strategy'][k] = _rand_map(rng, acts)
            workers.append(res)
        alpha = float(rng.choice([1.0, 1.5, 2.0]))
        gamma = float(rng.choice([0.0, 1.0, 2.0]))

        def dump(d):
            return {k: {'legal': v.legal_actions, 'regret': v.cumulative_regrets,
                        'strategy': v.cumulative_strategy, 'visit_count': v.visit_count,
                        'strategy_visit_count': v.strategy_visit_count} for k, v in d.items()}
        before = dump(glob)
        touched = merge_round(glob, workers, alpha, gamma)
        cases.append({'alpha': alpha, 'gamma': gamma, 'global': before, 'workers': workers,
                      'after': dump(glob), 'touched': sorted(touched)})
    _write_json(os.path.join(out, 'mccfr_merge.json'), {'cases': cases})
    return {'cases': len(cases)}


def _manifest(out, args, results, timings):
    from src.abstractions import postflop_features as pf
    hashes = {}
    try:
        from src.abstractions.postflop_v2 import PostflopV2
        p = PostflopV2()
        for street in ('flop', 'turn', 'river'):
            c, bins = p._centroid(street)
            hashes[street] = pf.centroid_hash(c, bins)
    except Exception as e:                                   # informational only
        hashes = {'error': repr(e)}
    prev = {}
    path = os.path.join(out, 'manifest.json')
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f).get('components', {})
    for name, summary in results.items():
        prev[name] = {'summary': summary, 'seconds': round(timings[name], 2),
                      'quick': args.quick, 'seed': args.seed}
    _write_json(path, {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_sha': _git_sha(), 'src_dirty': _git_dirty(),
        'python': platform.python_version(), 'numpy': np.__version__,
        'card_id': "rank_index*4 + suit_index; ranks '23456789TJQKA', suits 'HDCS'",
        'centroid_hashes': hashes, 'components': prev})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=20261108)
    ap.add_argument('--only', default='', help='comma-separated subset of: ' + ','.join(COMPONENTS))
    ap.add_argument('--quick', action='store_true', help='smaller fixtures for a fast loop')
    args = ap.parse_args()
    want = [c for c in args.only.split(',') if c] or list(COMPONENTS)
    bad = set(want) - set(COMPONENTS)
    if bad:
        ap.error(f"unknown component(s): {sorted(bad)}")
    os.makedirs(args.out, exist_ok=True)
    results, timings = {}, {}
    for name in want:
        # Each component gets its OWN seeded stream, so --only reproduces the same
        # fixture as a full run.
        rng = np.random.default_rng([args.seed, COMPONENTS.index(name)])
        t = time.time()
        results[name] = globals()[f'export_{name}'](args.out, rng, args.quick)
        timings[name] = time.time() - t
        print(f"  {name:15s} {timings[name]:7.1f}s  {results[name]}", flush=True)
    _manifest(args.out, args, results, timings)
    print(f"fixtures -> {args.out}")


if __name__ == '__main__':
    main()
