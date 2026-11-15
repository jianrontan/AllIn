#!/usr/bin/env python3
"""Logged-spot evaluator for the facing-a-jam decision (v3 step V3.11).

For every logged hand where the bot faced a human all-in before the river and answered
call or fold, the value of calling is exact: the server dealt both hands, so the recap
holds the human's real hole cards even when the bot folded (no showdown selection bias).

    EV(call) - EV(fold) = equity(bot hand vs human hand, remaining board) * final pot - call cost

(flop/turn: every runout; preflop: Monte Carlo, seeded). A policy that calls with
probability p in a spot scores p * that number; summed over spots it is an unbiased,
response-variance-free score of any facing-jam policy on real human jams.

Policies scored:
  logged       what the bot that played the hand did (v1/v2 in today's export)
  served       the bot served on --db today: the hand is replayed through a GameSession
               bound to that blueprint (real cards, logged actions, its range tracker) and
               the served RiverSubgameSolver decides; P(call) is read off its decision path
  jam_read     served, with the V3.11 pooled jam read installed (--jam-read: a fitted
               artifact, or 'lopo' = refit leaving the spot's player out)
  always_call / oracle   reference points (always_fold scores 0)

Run from backend/bot (seconds to a few minutes):
    python scripts/eval_facing_jams.py --in <hands_export.jsonl> --db <blueprint.db>
        [--model-dir <opponent models fit for --db>]   # exploit arm: the human model
        [--jam-read lopo|<artifact.json>] [--json out.json]
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions.postflop_features import rank7               # noqa: E402
from src.cfr.keys import BOARD_COUNT, STREET_NAMES                  # noqa: E402
from src.game import game_session as gs_mod                         # noqa: E402
from src.game.cards import to_engine                                # noqa: E402
from src.game.game_session import GameSession, session_kwargs, GameError  # noqa: E402
from src.cfr.poker_game import make_custom_action                   # noqa: E402
from src.game.range_tracker import _FULL_DECK                       # noqa: E402
from src.storage.blueprint_db import BlueprintDB                    # noqa: E402

_BLIND = {0: 1.0, 1: 2.0}
_STACK = 200.0
JAM_STREETS = ('preflop', 'flop', 'turn')


# ------------------------------------------------------------------- spots + exact value

def facing_jam_spots(rec):
    """Indices into rec['actionLog'] of human all-ins (preflop/flop/turn) that the bot
    answered with call or fold, each with the pot arithmetic of the bot's decision."""
    hs = rec.get('humanSeat')
    if hs not in (0, 1):
        return []
    bs = 1 - hs
    contrib = dict(_BLIND)
    log = rec.get('actionLog') or []
    out = []
    for i, a in enumerate(log):
        p = a.get('player')
        if p not in (0, 1):
            return out
        contrib[p] += float(a.get('chips') or 0.0)
        if p != hs or a.get('action') != 'allin' or a.get('street') not in JAM_STREETS:
            continue
        if i + 1 >= len(log):
            continue
        nxt = log[i + 1]
        if (nxt.get('player') != bs or nxt.get('street') != a.get('street')
                or nxt.get('action') not in ('call', 'fold')):
            continue
        to_call = contrib[hs] - contrib[bs]
        remaining = _STACK - contrib[bs]
        cost = min(to_call, remaining)
        pot = contrib[0] + contrib[1]
        final_pot = pot + cost if to_call <= remaining else pot - to_call + 2 * remaining
        out.append({'index': i, 'street': a['street'], 'cost': cost, 'final_pot': final_pot,
                    'logged_call': nxt['action'] == 'call'})
    return out


def exact_equity(a, b, board, seed, n_mc=20000):
    """Equity of hand `a` vs hand `b` over the remaining board: exact when two or fewer
    cards are to come, else a seeded Monte Carlo (preflop)."""
    dead = set(a) | set(b) | set(board)
    deck = [c for c in _FULL_DECK if c not in dead]
    need = 5 - len(board)

    def score(full):
        ra, rb = rank7(a + full), rank7(b + full)
        return 1.0 if ra < rb else 0.5 if ra == rb else 0.0
    if need == 0:
        return score(list(board))
    if need <= 2:
        vals = [score(list(board) + list(x)) for x in combinations(deck, need)]
        return sum(vals) / len(vals)
    rng = random.Random(seed)
    return sum(score(list(board) + rng.sample(deck, need)) for _ in range(n_mc)) / n_mc


# ------------------------------------------------------------------- replay

def _fixed_deck(rec):
    hs = rec['humanSeat']
    hh = [to_engine(c) for c in rec['humanHole']]
    bh = [to_engine(c) for c in rec['botHole']]
    p = {hs: hh, 1 - hs: bh}
    comm = [to_engine(c) for c in rec.get('community') or []]
    used = set(hh) | set(bh) | set(comm)
    pad = [c for c in _FULL_DECK if c not in used][:5 - len(comm)]
    return p[0] + p[1] + comm + pad


def _replay_action(s, a):
    """The action that reproduces log entry `a` in session `s`. A logged custom bet is
    recorded under its snapped grid name with the real chips, so a sized bet whose
    grid cost differs from the logged chips is replayed as a custom bet to the real
    raise-to total (the player's street contribution plus the chips). Otherwise the
    candidate would decide in a different pot from the one being scored."""
    act = a['action']
    if not act.startswith(('bet_', 'raise_')) or '_custom_' in act:
        return act
    chips = float(a.get('chips') or 0.0)
    if abs(s._action_cost(act) - chips) <= 1e-6:
        return act
    d = s.data
    contrib = s.game.get_player_contribution_this_round(
        d['history'], d['street'], d['starting_pot'], s.current_player(),
        d['p0_invested'], d['p1_invested'])
    return make_custom_action(act.startswith('raise_'), contrib + chips)


def same_spot(s, sp, tol=0.05):
    """True when the replayed decision has the logged call cost and final pot (the log
    rounds chips to 2 decimals, so a few actions can leave a cent of drift)."""
    d = s.data
    bot = s.current_player()
    stack = d['p0_stack'] if bot == 0 else d['p1_stack']
    to_call = s._action_cost('call')
    pot = s._current_pot()
    cost = min(to_call, stack)
    final_pot = pot + cost if to_call <= stack else pot - to_call + 2 * stack
    return abs(cost - sp['cost']) <= tol and abs(final_pot - sp['final_pot']) <= tol


def replay_to(rec, idx, kw, strategy_fn, hero_fn, exploit=None):
    """A GameSession at the bot's decision after log entry `idx`, or None when the logged
    line is illegal under this blueprint's menu (e.g. a bot 5-bet that capped_pj plays
    as its jam)."""
    deck = _fixed_deck(rec)
    orig = gs_mod.shuffled_deck
    gs_mod.shuffled_deck = lambda: list(deck)
    try:
        s = GameSession({'session_id': 'eval', 'player_id': rec.get('playerId'),
                         'human_net': 0.0}, strategy_fn=strategy_fn, hero_strategy_fn=hero_fn,
                        max_raises_per_street=float('inf'), **kw)
        s._deal_hand(1, rec['humanSeat'])
    finally:
        gs_mod.shuffled_deck = orig
    s.exploit = exploit
    for a in rec['actionLog'][:idx + 1]:
        if s.data['status'] != 'in_hand' or s.current_player() != a['player']:
            return None
        try:
            s.apply_action(_replay_action(s, a))
        except GameError:
            return None
    bot = 1 - rec['humanSeat']
    if s.data['status'] != 'in_hand' or s.current_player() != bot:
        return None
    if 'call' not in s.legal_actions() or 'fold' not in s.legal_actions():
        return None
    return s


def served_call_prob(bot, s):
    """P(call) of the served strategy at session `s`'s bot decision, read off the path
    the decision took (guards are deterministic; the tilt and the blueprint are mixed)."""
    key = s.info_set_key(s.current_player())
    legal = s.legal_actions()
    ps = s.bot_public_state()
    action = bot.decide(key, legal, ps)
    dbg = bot.last_debug or {}
    mode = dbg.get('mode')
    if mode == 'exploit_tilt':
        t = dbg.get('tiltedStrategy') or {}
        tot = sum(t.values())
        p = sum(v for a, v in t.items() if a in ('call', 'allin')) / tot if tot else 0.0
    elif mode == 'blueprint':
        dist = bot._state_distribution(key, legal, ps)
        tot = sum(dist.values())
        p = sum(v for a, v in dist.items() if a in ('call', 'allin')) / tot if tot else 0.0
        if bot._premium_no_fold('fold', legal, ps) == 'call':
            p = 1.0
    else:
        p = 1.0 if action in ('call', 'allin') else 0.0
    if dbg.get('jamRead'):
        mode = 'jam_read'                # the guard acted on the V3.11 read
    return p, mode or 'unknown'


# ------------------------------------------------------------------- main

def _summary(rows, policy):
    by = defaultdict(list)
    for r in rows:
        if policy in r:
            by[r['street']].append(r[policy] * r['ev'])
            by['all'].append(r[policy] * r['ev'])
    return {st: (len(v), sum(v)) for st, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--db', required=True)
    ap.add_argument('--model-dir', default=None,
                    help='opponent models fit for --db: the exploit arm (range tracker on the '
                         'human model, the tilt on). Default: the control arm (blueprint model)')
    ap.add_argument('--jam-read', default=None,
                    help="'lopo' (refit leaving each spot's player out) or a fitted artifact")
    ap.add_argument('--read-param', action='append', default=[], metavar='NAME=VALUE',
                    help='override a JamRead parameter (alpha, p_max, s, conf, margin, ...)')
    ap.add_argument('--json', default=None, help='per-spot rows (no player ids) + summary')
    args = ap.parse_args()

    from src.subgame.river_subgame_solver import RiverSubgameSolver
    db = BlueprintDB(args.db, read_only=True)
    kw = session_kwargs(db)
    bot = RiverSubgameSolver(db, max_iters=275, check_every=40, time_budget=24.0,
                             safe_gadget=True, gadget_anchor='auto', purify_threshold=0.01,
                             guard_confidence=0.2)
    range_fn, hero_fn = bot.range_model_fn(), bot.hero_range_model_fn()
    hm = None
    if args.model_dir:
        from src.exploitation.opponent_model import HumanModel
        hm = HumanModel(args.model_dir, db)

    recs = [json.loads(line) for line in open(args.inp, encoding='utf-8') if line.strip()]
    recs = [r for r in recs if len(r.get('humanHole') or []) == 2
            and len(r.get('botHole') or []) == 2]

    reads = {}
    if args.jam_read:
        from src.exploitation import jam_read as jr
        params = {k: float(v) for k, v in (x.split('=', 1) for x in args.read_param)}
        if args.jam_read == 'lopo':
            pids = {r.get('playerId') for r in recs}
            cache = jr.JamObservations.from_recaps(recs, bot._combo_strength)
            reads = {p: jr.JamRead.fit(cache.excluding(p), jr.read_stamp(db), **params)
                     for p in pids}
        else:
            read = jr.JamRead.load(args.jam_read, db, **params)
            reads = defaultdict(lambda: read)

    rows, skipped, mismatched = [], 0, 0
    for rec in recs:
        for sp in facing_jam_spots(rec):
            st_idx = STREET_NAMES.index(sp['street'])
            comm = [to_engine(c) for c in rec.get('community') or []]
            board = comm[:BOARD_COUNT[st_idx]] if st_idx else []
            hh = [to_engine(c) for c in rec['humanHole']]
            bh = [to_engine(c) for c in rec['botHole']]
            eq = exact_equity(bh, hh, board, seed=rec.get('handKey') or '')
            row = {'street': sp['street'], 'ev': eq * sp['final_pot'] - sp['cost'],
                   'need': sp['cost'] / sp['final_pot'], 'logged': float(sp['logged_call']),
                   'always_call': 1.0}
            row['oracle'] = 1.0 if row['ev'] > 0 else 0.0
            pid = rec.get('playerId')
            row['_player'] = pid                 # for the clustered SE only; never written out
            exploit = None
            strategy_fn = range_fn
            if hm is not None:
                strategy_fn = hm.strategy_fn_for(pid)
                exploit = {'chardist': hm.chardist_for(pid), 'delta': 0.05}
            s = replay_to(rec, sp['index'], kw, strategy_fn, hero_fn, exploit)
            if s is None:
                skipped += 1
                continue
            if not same_spot(s, sp):
                mismatched += 1          # the candidate would decide a different pot
                continue
            row['served'], row['served_mode'] = served_call_prob(bot, s)
            if reads:
                bot.jam_read = reads[pid]
                try:
                    s2 = replay_to(rec, sp['index'], kw, strategy_fn, hero_fn, exploit)
                    row['jam_read'], row['jam_read_mode'] = served_call_prob(bot, s2)
                finally:
                    bot.jam_read = None
            rows.append(row)

    print(f"{len(rows)} spots scored ({skipped} skipped: line illegal under this menu; "
          f"{mismatched} skipped: replayed pot/call cost differ from the log)")
    policies = ['logged', 'served'] + (['jam_read'] if reads else []) + ['always_call', 'oracle']
    print(f"  {'policy':<12}" + ''.join(f"{st:>16}" for st in ('preflop', 'flop', 'turn', 'all')))
    summary = {}
    for pol in policies:
        sm = _summary(rows, pol)
        summary[pol] = sm
        cells = ''.join(f"{sm.get(st, (0, 0.0))[1]:>11.0f} ({sm.get(st, (0, 0))[0]:>2})"
                        for st in ('preflop', 'flop', 'turn', 'all'))
        print(f"  {pol:<12}{cells}")
    print("  (sum of EV(call) - EV(fold) in chips taken, spots in brackets; always_fold = 0)")
    if reads:
        d = [r['ev'] * (r['jam_read'] - r['served']) for r in rows]
        n = len(d)
        m = sum(d) / n
        se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float('nan')
        # Spots of one player are not independent: also a player-clustered SE.
        by_p = defaultdict(float)
        cnt_p = defaultdict(int)
        for r, x in zip(rows, d):
            by_p[r['_player']] += x
            cnt_p[r['_player']] += 1
        g = len(by_p)
        cse = (math.sqrt(sum((by_p[p] - cnt_p[p] * m) ** 2 for p in by_p) * g / (g - 1)) / n
               if g > 1 else float('nan'))
        more = [r for r in rows if r['jam_read'] > r['served']]
        less = [r for r in rows if r['jam_read'] < r['served']]
        print(f"  jam_read - served: {m:+.2f} chips/spot (se {se:.2f}; clustered by player "
              f"{cse:.2f}, {g} players; n {n})")
        print(f"    calls more in {len(more)} spots (their mean EV(call) "
              f"{sum(r['ev'] for r in more) / max(1, len(more)):+.1f}), calls less in "
              f"{len(less)} (mean EV(call) {sum(r['ev'] for r in less) / max(1, len(less)):+.1f})")
        jm = defaultdict(int)
        for r in rows:
            jm[r['jam_read_mode']] += 1
        print(f"    jam_read decision paths: {dict(jm)}")
    modes = defaultdict(int)
    for r in rows:
        modes[r['served_mode']] += 1
    print(f"  served decision paths: {dict(modes)}")
    if args.json:
        out = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]
        with open(args.json, 'w') as f:
            json.dump({'summary': summary, 'rows': out}, f, indent=1)


if __name__ == '__main__':
    main()
