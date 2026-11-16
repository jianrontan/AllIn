#!/usr/bin/env python
"""V3.15: how much does a postflop key lose by forgetting the earlier streets?

Postflop keys carry only the current street's betting, plus (spr5) the street-start SPR
bucket on the flop and turn; river keys carry no SPR. So every earlier-street line that
reaches a street with the same SPR bucket shares a key. This walks the blueprint's public
betting tree on sampled boards, carrying each player's per-hand reach, and at the first
decision of each recorded street stores both players' reach vectors.

The number reported is the key's equity error. Training (external-sampling MCCFR) updates
a key's regrets from every line that reaches it, weighted by the OPPONENT's reach only:
the traverser explores all its own actions, so its own reach never enters
(core-rs/src/mccfr/trainer.rs, cfr()). The opponent range a merged key is trained
against is therefore the sum of the opponent's reach over the merged lines, the same for
every hand of the deciding player. For each hand h of the player deciding and each line
l, the error is |equity(h vs the line's real opponent range) - equity(h vs that pooled
range)|, averaged with weight = how often h really stands at l (joint reach). Reported:
the mean error and the share of that mass off by more than 5 equity points, per player.

Groupings of the recorded lines (what the key would distinguish):

  street only      no SPR, no history (the lower reference)
  spr              the spr5 street-start bucket (today's flop/turn keys; NEW on the river)
  spr+prev         + the previous street's last aggressor (none/ip/oop)
  spr+init         + "initiative": the last aggressor of the most recent street that had
                   any aggression, and whether that street was the previous one or the
                   previous street checked through (5 values)
  spr+prev+prev2   + the last aggressors of the previous two streets (turn: flop and
                   preflop; river: turn and flop)
  spr+prev+prev2+prev3   river only: turn, flop and preflop aggressors
  spr+prev+prev2+size    + the size class of the previous street's last aggression
  sprfine          a finer SPR split (SPR_FINE_EDGES) instead of spr5, no history
  sprfine+prev+prev2     the finer split + two aggressor tags
  spr+randK        each SPR group's lines split at random (a hash of the line) into K
                   parts: the "more keys" baseline a real tag has to beat

Turn and river equities are exact over the remaining cards; flop equities use sampled
runouts. Lines below `--min-reach` are pruned and the pruned mass is reported. When the
recorded streets include later ones, the hands that collide with later board cards are
dropped from the earlier streets too (a small bias, the same for every grouping).

    python scripts/measure_street_history.py --db <blueprint.db> --boards 48 --workers 8
    python scripts/measure_street_history.py --db <blueprint.db> --streets 3 --boards 32
"""
import argparse
import json
import random
import sys
import time
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cfr.keys import SPR_EDGES, action_char, spr_bucket  # noqa: E402
from src.cfr.poker_game import STARTING_STACK  # noqa: E402
from src.evaluation.best_response import BestResponseEvaluator  # noqa: E402
from src.evaluation.showdown_kernel import build_board_arrays  # noqa: E402
from src.abstractions.postflop_features import rank7  # noqa: E402
from src.game.range_tracker import _FULL_DECK  # noqa: E402
from src.storage.blueprint_db import BlueprintDB  # noqa: E402

# A finer SPR split than the served spr5 edges (2, 5, 12, 40), for comparison.
SPR_FINE_EDGES = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0, 12.0, 18.0, 25.0, 40.0)
_POS = {0: 'ip', 1: 'oop'}


def _aggressive(a):
    return a == 'allin' or a.startswith(('bet_', 'raise_'))


def last_aggressor(game, street, history):
    """'ip' / 'oop' for whoever made the street's last aggression, else 'none'."""
    for i in range(len(history) - 1, -1, -1):
        if _aggressive(history[i]):
            return _POS[game._acting_player(i, street)]
    return 'none'


def last_size(history):
    """Size class of the street's last aggression: none / s / m / big (l, x, o, 2) / a."""
    for a in reversed(list(history)):
        if _aggressive(a):
            c = action_char(a)
            return c if c in ('s', 'm', 'a') else 'big'
    return 'none'


def spr_of(street_pot):
    return (STARTING_STACK - street_pot / 2.0) / street_pot


def spr5(street_pot):
    """The spr5 bucket, computed on any street (the served keys use it on flop/turn)."""
    return sum(1 for e in SPR_EDGES if spr_of(street_pot) >= e)


def spr_fine(street_pot):
    return sum(1 for e in SPR_FINE_EDGES if spr_of(street_pot) >= e)


def tags(game, prior):
    """The candidate key additions for a street start, from the earlier streets'
    histories `prior` = ((street, history), ...) oldest first."""
    aggr = [last_aggressor(game, s, list(h)) for s, h in prior]     # oldest first
    back = aggr[::-1]                                                # previous street first
    init = 'none'
    for i, a in enumerate(back):
        if a != 'none':
            init = (a, 'prev' if i == 0 else 'older')
            break
    return {'prev': back[0], 'prev2': back[1] if len(back) > 1 else '-',
            'prev3': back[2] if len(back) > 2 else '-', 'init': init,
            'size': last_size(prior[-1][1])}


def _rand(line, k):
    return zlib.crc32(line.encode()) % k


GROUPINGS = {
    'street only': lambda t: (),
    'spr': lambda t: (t['spr'],),
    'spr+prev': lambda t: (t['spr'], t['prev']),
    'spr+init': lambda t: (t['spr'], t['init']),
    'spr+prev+prev2': lambda t: (t['spr'], t['prev'], t['prev2']),
    'spr+prev+prev2+prev3': lambda t: (t['spr'], t['prev'], t['prev2'], t['prev3']),
    'spr+prev+prev2+size': lambda t: (t['spr'], t['prev'], t['prev2'], t['size']),
    'sprfine': lambda t: (t['sprfine'],),
    'sprfine+prev+prev2': lambda t: (t['sprfine'], t['prev'], t['prev2']),
    'spr+rand3': lambda t: (t['spr'], _rand(t['line'], 3)),
    'spr+rand9': lambda t: (t['spr'], _rand(t['line'], 9)),
    'spr+rand27': lambda t: (t['spr'], _rand(t['line'], 27)),
}


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
        nodes = []          # (street, tag dict, r0, r1)
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
            if not history and street in self.record_streets:
                t = tags(game, prior)
                t['spr'] = spr5(starting_pot)
                t['sprfine'] = spr_fine(starting_pot)
                t['line'] = '/'.join(''.join(action_char(a) for a in h) for _, h in prior)
                nodes.append((street, t, r0.copy(), r1.copy()))
                if street >= self.max_street:
                    return
            spr = spr_bucket(self.kf, street, starting_pot)
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


def equity_matrices(ba, board, street, n_runouts, seed):
    """W[h, o] = P(h beats o) + half the ties over the street's remaining runouts, pooled
    over (pair, runout) like jam_read._bin_equities; C[h, o] = 1 if h and o share no card
    (and some runout is compatible). River: showdown; turn: every river; flop: seeded
    runouts."""
    hands, H = ba['hands'], ba['H']
    known = list(board[:2 + street])
    c1, c2 = ba['c1'], ba['c2']
    dead = set(board)          # the walked board; hands colliding with it are not in ba
    undealt = [c for c in _FULL_DECK if c not in dead]
    need = 5 - len(known)
    if need == 0:
        runouts = [()]
    elif need == 1:
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


def summarize_board(ba, nodes, board, n_runouts, seed, big=0.05):
    """Per street and grouping, sums over (line, hand) of joint reach x |equity error|
    (deq), of joint reach on errors > `big` (big), and of joint reach (hw), per player
    deciding at the key; plus the number of groups."""
    out = {}
    for street in sorted({n[0] for n in nodes}):
        sn = [n for n in nodes if n[0] == street]
        W, C = equity_matrices(ba, board, street, n_runouts, seed)
        res = {'lines': len(sn), 'w': sum(float(n[2].mean() * n[3].mean()) for n in sn)}
        for g, fn in GROUPINGS.items():
            groups = defaultdict(list)
            for n in sn:
                groups[fn(n[1])].append(n)
            acc = {k: {'ip': 0.0, 'oop': 0.0} for k in ('deq', 'big', 'hw')}
            for members in groups.values():
                for pos, j, jo in (('ip', 2, 3), ('oop', 3, 2)):
                    R = np.stack([m[j] for m in members])           # deciding player's reach
                    Ro = np.stack([m[jo] for m in members])         # opponent's reach
                    mo = Ro.mean(1)
                    if mo.sum() <= 0:
                        continue
                    Po = Ro / np.maximum(Ro.sum(1, keepdims=True), 1e-300)
                    num, den = Po @ W.T, Po @ C.T
                    E_line = num / np.maximum(den, 1e-12)            # [L, H]
                    pooled = Ro.sum(0)                               # training's pool
                    E_pool = (W @ pooled) / np.maximum(C @ pooled, 1e-12)
                    d = np.abs(E_line - E_pool[None, :])
                    hw = np.where(den > 1e-12, R * mo[:, None], 0.0)  # joint reach
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
                    help='sampled turn+river runouts for flop equities (turn/river are exact)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--json', help='write per-board results here')
    a = ap.parse_args()
    streets = [int(s) for s in a.streets.split(',')]
    ncards = {1: 3, 2: 4, 3: 5}[max(streets)]
    rng = random.Random(a.seed)
    boards = [rng.sample(_FULL_DECK, ncards) for _ in range(a.boards)]
    if a.workers > 1:
        with Pool(a.workers, initializer=_init,
                  initargs=(a.db, streets, a.min_reach, a.flop_runouts)) as p:
            results = p.map(_one, boards)
    else:
        _init(a.db, streets, a.min_reach, a.flop_runouts)
        results = [_one(b) for b in boards]

    print(f'{len(results)} boards; mean {np.mean([r["secs"] for r in results]):.1f} s/board; '
          f'pruned line mass {np.mean([r["pruned"] for r in results]):.2e}')
    names = {1: 'flop', 2: 'turn', 3: 'river'}
    for st in streets:
        rows = [r['streets'][st] for r in results if st in r['streets']]
        print(f'\n{names[st]} start: {np.mean([r["lines"] for r in rows]):.0f} lines/board '
              f'(reach {np.mean([r["w"] for r in rows]):.3f})')
        print(f'  {"grouping":<22} {"groups":>6}  {"|d equity| ip/oop":>17}  {">5pp ip/oop":>13}')
        for g in GROUPINGS:
            if (st < 3 and g.endswith('prev3')) or (st == 1 and 'prev2' in g):
                continue        # identical to a shorter row on this street
            k = np.mean([r[g]['groups'] for r in rows])
            hw = [sum(r[g]['hw'][p] for r in rows) for p in ('ip', 'oop')]
            dq = [sum(r[g]['deq'][p] for r in rows) / h for p, h in zip(('ip', 'oop'), hw)]
            bg = [sum(r[g]['big'][p] for r in rows) / h for p, h in zip(('ip', 'oop'), hw)]
            print(f'  {g:<22} {k:6.1f}  {100 * dq[0]:6.2f} / {100 * dq[1]:.2f}pp  '
                  f'{100 * bg[0]:5.1f} / {100 * bg[1]:.1f}%')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(results, f, indent=1, default=str)


if __name__ == '__main__':
    main()
