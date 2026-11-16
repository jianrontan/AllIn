#!/usr/bin/env python
"""V3.15: how different are the ranges that one postflop key averages over?

Postflop keys carry only the current street's betting (plus, for spr5, the street-start
SPR bucket), so every earlier-street line that reaches the same street with the same SPR
bucket shares a key. This walks the blueprint's public betting tree on sampled boards,
carrying each player's per-hand reach, and at the first decision of each recorded street
stores both players' ranges and the line's probability. Then, per board and per grouping
of those street-start nodes, it reports the reach-weighted total-variation distance
between each line's range and the group's pooled range (the range the shared key is
trained against), per player:

  street         one key per street (no SPR, no history): the upper reference
  street+spr     what the served keys distinguish today
  street+spr+pot   + preflop pot type (srp/3bp/4bp)
  street+spr+aggr  + who made the previous street's last aggression (none/ip/oop)
  street+spr+tok   + both: the proposed history token
  street+spr+aggr+pre  previous street's last aggressor + preflop's (turn keys: both)
  street+spr+aggr+n    previous street's last aggressor + how many aggressions it had
  street+spr+all   all four
  full history   0 by definition

Two numbers per grouping: TV over hands (descriptive), and the decision-relevant one, the
key's equity error: for the player deciding at the key, each hand's equity against the
line's real opponent range minus its equity against the opponent range the key pools for
that hand's bucket (reach-weighted over the merged lines, which is what training sees),
averaged over hands and lines by joint reach, plus the share of that mass off by more
than 5 points. Turn equities are exact over the river; flop equities use sampled
runouts. Both are exact on the walked tree otherwise; lines below
`--min-reach` (line probability) are pruned and the pruned mass is reported.

    python scripts/measure_street_history.py --db <blueprint.db> --boards 24 --workers 6
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cfr.keys import action_char, spr_bucket  # noqa: E402
from src.cfr.poker_game import STARTING_STACK  # noqa: E402
from src.evaluation.best_response import BestResponseEvaluator  # noqa: E402
from src.evaluation.showdown_kernel import build_board_arrays  # noqa: E402
from src.exploitation.jam_read import pot_type  # noqa: E402
from src.abstractions.postflop_features import rank7  # noqa: E402
from src.game.range_tracker import _FULL_DECK  # noqa: E402
from src.storage.blueprint_db import BlueprintDB  # noqa: E402

GROUPINGS = ('street', 'street+spr', 'street+spr+pot', 'street+spr+aggr', 'street+spr+tok',
             'street+spr+aggr+pre', 'street+spr+aggr+n', 'street+spr+all')
_POS = {0: 'ip', 1: 'oop'}


def _aggressive(a):
    return a == 'allin' or a.startswith(('bet_', 'raise_'))


def last_aggressor(game, street, history):
    """'ip' / 'oop' for whoever made the street's last aggression, else 'none'."""
    for i in range(len(history) - 1, -1, -1):
        if _aggressive(history[i]):
            return _POS[game._acting_player(i, street)]
    return 'none'


class Walker:
    def __init__(self, db_path, record_streets, min_reach):
        self.ev = BestResponseEvaluator(BlueprintDB(db_path, read_only=True), seed=0)
        self.game = self.ev.game
        self.kf = self.ev.key_format
        self.record_streets = tuple(record_streets)
        self.max_street = max(self.record_streets)
        self.min_reach = min_reach

    def board(self, board):
        ba = build_board_arrays(board, self.ev.evaluator, self.ev.cards,
                                streets=tuple(s for s in (1, 2, 3) if s <= self.max_street))
        H = ba['H']
        game, ev = self.game, self.ev
        key_cache = {}
        nodes = []          # (street, spr, token, line, w, r0, r1)
        pruned = [0.0]

        def probs_matrix(street, pattern, legal, pos, spr):
            from src.cfr.keys import make_info_set_key
            mat = np.empty((H, len(legal)))
            for gi, (mask, rep) in enumerate(ba['groups'][street]):
                kc = (street, gi, pattern, pos, spr)
                key = key_cache.get(kc)
                if key is None:
                    if street == 0:
                        key = make_info_set_key(0, pos, ba['pf'][rep], None, pattern)
                    else:
                        key = make_info_set_key(street, pos, ba['pf'][rep],
                                                ba['strg'][street][rep], pattern, spr)
                    key_cache[kc] = key
                mat[mask] = ev._restricted_probs(key, legal)
            return mat

        def walk(history, street, starting_pot, p0_inv, p1_inv, p0_stack, p1_stack,
                 pattern, r0, r1, prior, depth):
            if depth > 60 or street > self.max_street or game.is_terminal(history, street):
                return
            cp = game._acting_player(len(history), street)
            legal = game.get_legal_actions(street, history, starting_pot, cp,
                                           p0_stack, p1_stack, p0_inv, p1_inv)
            if not legal:
                if street >= self.max_street:
                    return
                p0_this = game.get_player_contribution_this_round(
                    history, street, starting_pot, 0, p0_inv, p1_inv)
                p1_this = game.get_player_contribution_this_round(
                    history, street, starting_pot, 1, p0_inv, p1_inv)
                new_pot = game.calculate_current_pot(starting_pot, history, street,
                                                     p0_inv, p1_inv)
                n0, n1 = p0_inv + p0_this, p1_inv + p1_this
                walk([], street + 1, new_pot, n0, n1, STARTING_STACK - n0,
                     STARTING_STACK - n1, '', r0, r1,
                     prior + ((street, tuple(history)),), depth + 1)
                return
            spr = spr_bucket(self.kf, street, starting_pot)
            if not history and street in self.record_streets:
                pre = prior[0][1]
                prev_street, prev_hist = prior[-1]
                tok = (pot_type(list(pre)), last_aggressor(game, prev_street, list(prev_hist)),
                       last_aggressor(game, 0, list(pre)),
                       min(3, sum(1 for x in prev_hist if _aggressive(x))))
                line = '/'.join(''.join(action_char(a) for a in h) for _, h in prior)
                w = float(r0.mean() * r1.mean())
                nodes.append((street, spr, tok, line, w, r0.copy(), r1.copy()))
                if street >= self.max_street:
                    return
            probs = probs_matrix(street, pattern, tuple(legal), _POS[cp], spr)
            for ai, a in enumerate(legal):
                n0 = r0 * probs[:, ai] if cp == 0 else r0
                n1 = r1 * probs[:, ai] if cp == 1 else r1
                w = float(n0.mean() * n1.mean())
                if w < self.min_reach:
                    pruned[0] += w
                    continue
                cost = game._action_cost(a, street, history, starting_pot, cp, p0_inv, p1_inv)
                walk(history + [a], street, starting_pot, p0_inv, p1_inv,
                     p0_stack - (cost if cp == 0 else 0), p1_stack - (cost if cp == 1 else 0),
                     pattern + action_char(a), n0, n1, prior, depth + 1)

        walk([], 0, 3, 0.0, 0.0, STARTING_STACK - 1, STARTING_STACK - 2, '',
             np.ones(H), np.ones(H), (), 0)
        return ba, nodes, pruned[0]


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def equity_matrices(ba, board, street, n_runouts, seed):
    """W[h, o] = P(h beats o) + half the ties over the street's remaining runouts, pooled
    over (pair, runout) like jam_read._bin_equities; C[h, o] = 1 if h and o share no card
    (and some runout is compatible). Turn: every river (exact); flop: seeded runouts."""
    hands, H = ba['hands'], ba['H']
    known = list(board[:2 + street])
    c1, c2 = ba['c1'], ba['c2']
    dead = set(board)          # the walked board; hands colliding with it are not in ba
    undealt = [c for c in _FULL_DECK if c not in dead]
    need = 5 - len(known)
    if need == 1:
        runouts = [(c,) for c in undealt]
    else:
        rng = random.Random(seed)
        runouts = [tuple(rng.sample(undealt, need)) for _ in range(n_runouts)]
    wins = np.zeros((H, H))
    cnt = np.zeros((H, H))
    for ro in runouts:
        roset = set(ro)
        valid = np.array([not (a in roset or b in roset) for a, b in hands])
        full = known + list(ro)
        ranks = np.array([rank7([a, b] + full) if v else 0 for (a, b), v in zip(hands, valid)])
        m = valid[:, None] & valid[None, :]
        wins += m * ((ranks[:, None] < ranks[None, :]) + 0.5 * (ranks[:, None] == ranks[None, :]))
        cnt += m
    compat = ((c1[:, None] != c1[None, :]) & (c1[:, None] != c2[None, :])
              & (c2[:, None] != c1[None, :]) & (c2[:, None] != c2[None, :]) & (cnt > 0))
    W = np.where(compat, wins / np.maximum(cnt, 1), 0.0)
    return W, compat.astype(float)


def _key_labels(ba, street):
    lab = np.empty(ba['H'], dtype=np.int64)
    for gi, (mask, _) in enumerate(ba['groups'][street]):
        lab[mask] = gi
    return lab


def _group_key(n, g):
    """n = (street, spr, tok, line, ...); tok = (pot type, previous street's last
    aggressor, preflop's last aggressor, aggressions on the previous street capped at 3)."""
    t = n[2]
    return {'street': (n[0],), 'street+spr': (n[0], n[1]),
            'street+spr+pot': (n[0], n[1], t[0]), 'street+spr+aggr': (n[0], n[1], t[1]),
            'street+spr+tok': (n[0], n[1], t[0], t[1]),
            'street+spr+aggr+pre': (n[0], n[1], t[1], t[2]),
            'street+spr+aggr+n': (n[0], n[1], t[1], t[3]),
            'street+spr+all': (n[0], n[1]) + tuple(t)}[g]


def summarize_board(ba, nodes, board, n_runouts, seed, big=0.05):
    """Per street and grouping, reach-weighted sums (divide by the street's total):
    tv[pos]   line weight x TV(line range of `pos`, the group's pooled range)
    deq[pos]  for the player at `pos` deciding at the key: hand reach x |equity vs the
              line's opponent range - equity vs the opponent range the key pools for
              that hand's own bucket|  (the key's equity error from forgetting the line)
    big[pos]  the same weight on |delta equity| > `big`"""
    out = {}
    for street in sorted({n[0] for n in nodes}):
        sn = [n for n in nodes if n[0] == street]
        W, C = equity_matrices(ba, board, street, n_runouts, seed)
        lab = _key_labels(ba, street)
        nb = int(lab.max()) + 1
        res = {'w': sum(n[4] for n in sn), 'lines': len(sn)}
        for g in GROUPINGS:
            groups = defaultdict(list)
            for n in sn:
                groups[_group_key(n, g)].append(n)
            acc = {k: {'ip': 0.0, 'oop': 0.0} for k in ('tv', 'deq', 'big', 'hw')}
            for members in groups.values():
                lw = np.array([m[4] for m in members])
                if lw.sum() <= 0:
                    continue
                for pos, j, jo in (('ip', 5, 6), ('oop', 6, 5)):
                    R = np.stack([m[j] for m in members])          # this player's reach
                    P = R / R.sum(1, keepdims=True)
                    pooled = (lw[:, None] * P).sum(0) / lw.sum()
                    acc['tv'][pos] += float(sum(w * tv(p, pooled) for w, p in zip(lw, P)))
                    # pos deciding: its hands' reach x the opponent's line range
                    Ro = np.stack([m[jo] for m in members])
                    mo = Ro.mean(1)
                    Po = Ro / Ro.sum(1, keepdims=True)
                    E_line = (Po @ W.T) / np.maximum(Po @ C.T, 1e-12)   # [L, H]
                    hw = R * mo[:, None]                                 # [L, H] joint reach
                    Bw = np.zeros((nb, len(members)))
                    np.add.at(Bw, lab, hw.T)                             # [B, L]
                    pooled_b = (Bw @ Po) / np.maximum(Bw.sum(1, keepdims=True), 1e-300)
                    PB = pooled_b[lab]                                    # [H, H]
                    E_pool = (W * PB).sum(1) / np.maximum((C * PB).sum(1), 1e-12)
                    d = np.abs(E_line - E_pool[None, :])
                    ok = (Po @ C.T) > 1e-12
                    hw = np.where(ok, hw, 0.0)
                    acc['deq'][pos] += float((hw * d).sum()) / ba['H']
                    acc['big'][pos] += float((hw * (d > big)).sum()) / ba['H']
                    acc['hw'][pos] += float(hw.sum()) / ba['H']
            res[g] = {'groups': len(groups), **acc}
        out[street] = res
    return out


_W = None


_RUNOUTS = 40


def _init(db, streets, min_reach, runouts):
    global _W, _RUNOUTS
    _W = Walker(db, streets, min_reach)
    _RUNOUTS = runouts


def _one(board):
    t = time.time()
    ba, nodes, pruned = _W.board(board)
    s = summarize_board(ba, nodes, board, _RUNOUTS, ''.join(board))
    return {'board': board, 'secs': round(time.time() - t, 1), 'pruned': pruned, 'streets': s}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--db', required=True)
    ap.add_argument('--boards', type=int, default=24)
    ap.add_argument('--seed', type=int, default=1515)
    ap.add_argument('--streets', default='1,2', help='street starts to record (1=flop, 2=turn, 3=river)')
    ap.add_argument('--min-reach', type=float, default=1e-6)
    ap.add_argument('--flop-runouts', type=int, default=40,
                    help='sampled turn+river runouts for flop equities (turn is exact)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--json', help='write per-board results here')
    a = ap.parse_args()
    streets = [int(s) for s in a.streets.split(',')]
    ncards = {1: 3, 2: 4, 3: 5}[max(streets)]
    rng = random.Random(a.seed)
    boards = [rng.sample(_FULL_DECK, ncards) for _ in range(a.boards)]
    if a.workers > 1:
        with Pool(a.workers, initializer=_init, initargs=(a.db, streets, a.min_reach, a.flop_runouts)) as p:
            results = p.map(_one, boards)
    else:
        _init(a.db, streets, a.min_reach, a.flop_runouts)
        results = [_one(b) for b in boards]

    print(f'{len(results)} boards; mean {np.mean([r["secs"] for r in results]):.1f} s/board; '
          f'pruned line mass {np.mean([r["pruned"] for r in results]):.2e}')
    names = {1: 'flop', 2: 'turn', 3: 'river'}
    for st in streets:
        rows = [r['streets'][st] for r in results if st in r['streets']]
        W = sum(r['w'] for r in rows)
        print(f'\n{names[st]} start: {np.mean([r["lines"] for r in rows]):.0f} lines/board '
              f'(reach {W / len(rows):.3f} of hands reach it)')
        print(f'  {"grouping":<19} {"keys":>5}  {"TV range ip/oop":>15}  '
              f'{"|d equity| ip/oop":>17}  {">5pp ip/oop":>13}')
        for g in GROUPINGS:
            k = np.mean([r[g]['groups'] for r in rows])
            tvs = [sum(r[g]['tv'][p] for r in rows) / W for p in ('ip', 'oop')]
            hw = [sum(r[g]['hw'][p] for r in rows) for p in ('ip', 'oop')]
            dq = [sum(r[g]['deq'][p] for r in rows) / h for p, h in zip(('ip', 'oop'), hw)]
            bg = [sum(r[g]['big'][p] for r in rows) / h for p, h in zip(('ip', 'oop'), hw)]
            print(f'  {g:<19} {k:5.1f}  {tvs[0]:6.3f} / {tvs[1]:.3f}  '
                  f'{100 * dq[0]:6.2f} / {100 * dq[1]:.2f}pp  {100 * bg[0]:5.1f} / {100 * bg[1]:.1f}%')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(results, f, indent=1, default=str)


if __name__ == '__main__':
    main()
