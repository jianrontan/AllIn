# backend/bot/scripts/bug027_replay.py
"""
BUG-027 replay: the exact live line, read through each blueprint's own strategy.

The line (100BB, hero = the bot in the SB holding J9hh): bot opens to 3.5BB
(bet_large), the BB 3-bets to 8.5BB (a custom size, credited to the small/medium
bracket by translation), the bot 4-bets (raise_medium, to 22BB), the BB calls.
Flop Qs9c3c, pot 88 chips, 156 behind; the BB leads all-in and the bot calls.

For each blueprint, with the blueprint as the model of both players (a Bayesian
range tracker per side, bucketed with the blueprint's own abstraction):
  - the BB's response to the 4-bet (fold / call / 5-bet jam where the menu has it),
    overall and for the top hands;
  - the bot's 4-bet range: how often J9s / J9o 4-bet here;
  - the BB's flop lead-shove range: frequency and made-hand composition;
  - J9hh's exact equity and call EV against that shove range, and how much of the
    bot's own range calls the shove.
Plus, once: J9hh's equity against realistic human shove ranges.

Run from backend/bot/ (a minute or two per blueprint):
    python scripts/bug027_replay.py --db analysis/blueprints/blueprint_final_v2.db [--db ...] [--json out.json]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.abstractions.abstraction_set import resolve_abstraction_dir   # noqa: E402
from src.abstractions.card_abstractions import CardAbstraction   # noqa: E402
from src.abstractions.postflop_features import rank7   # noqa: E402
from src.abstractions.sizing import db_menu_mode   # noqa: E402
from src.cfr.keys import make_info_set_key, spr_bucket, db_key_format   # noqa: E402
from src.evaluation.match import _legal_actions   # noqa: E402
from src.game import range_tracker as rt   # noqa: E402
from src.game.range_tracker import RangeTracker, _FULL_DECK   # noqa: E402
from src.storage.blueprint_db import BlueprintDB   # noqa: E402

HERO = ['HJ', 'H9']
FLOP = ['SQ', 'C9', 'C3']
R = '23456789TJQKA'
# The BB's custom 3-bet to 17 chips, translated onto the 3-bet grid (pseudo-harmonic;
# the live session's translation of this exact size, recorded in the original replay).
THREEBET_BLEND = [('raise_small', 0.3718), ('raise_medium', 0.6282)]
FLOP_POT, BEHIND = 88.0, 156.0
TOP = ('AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo', 'AQs', 'AQo', 'KQs', 'A5s', 'J9s', 'J9o')


def model_fn(db):
    """BlueprintStrategy.range_model_fn: the raw average strategy restricted to the
    legal actions and renormalised; uniform where the key has no mass."""
    def fn(key, legal):
        st = db.get_average_strategy(key)
        n = len(legal)
        if st:
            w = np.array([max(0.0, st.get(a, 0.0)) for a in legal])
            t = w.sum()
            if t > 1e-12:
                return w / t
        return np.ones(n) / n
    return fn


def hand_class(h):
    a, b = h
    ra, rb = R.index(a[1]), R.index(b[1])
    if ra < rb:
        a, b, ra, rb = b, a, rb, ra
    if ra == rb:
        return R[ra] * 2
    return R[ra] + R[rb] + ('s' if a[0] == b[0] else 'o')


def category(h):
    """Made-hand category on Qs9c3c, plus flush-draw / straight-draw flags."""
    cards = list(h) + FLOP
    ranks = [c[1] for c in cards]
    hr = [c[1] for c in h]
    if hr[0] == hr[1] and hr[0] in 'Q93':
        cat = 'set'
    elif hr[0] != hr[1] and sum(1 for x in hr if x in 'Q93') == 2:
        cat = 'two pair'
    elif hr[0] == hr[1] and R.index(hr[0]) > R.index('Q'):
        cat = 'overpair'
    elif 'Q' in hr:
        cat = 'top pair'
    elif hr[0] == hr[1] and R.index(hr[0]) > R.index('9'):
        cat = 'pocket TT-JJ'
    elif '9' in hr:
        cat = 'second pair'
    elif hr[0] == hr[1] or '3' in hr:
        cat = 'weak pair'
    else:
        cat = 'no pair'
    fd = sum(1 for c in cards if c[0] == 'C') >= 4
    rs = set(R.index(x) for x in ranks) | ({-1} if 'A' in ranks else set())
    sd = any(len(rs & set(range(lo, lo + 5))) >= 4 for lo in range(-1, 9))
    return cat, fd, sd


def exact_equity(hero, board, villain):
    """Hero equity vs weighted villain hands over every turn+river runout."""
    dead = set(hero) | set(board)
    deck = [c for c in _FULL_DECK if c not in dead]
    live = [(h, w) for h, w in villain if w > 0 and h[0] not in dead and h[1] not in dead]
    eq = tot = 0.0
    for t, r in combinations(deck, 2):
        full = board + [t, r]
        hr = rank7(hero + full)
        for h, w in live:
            if h[0] in (t, r) or h[1] in (t, r):
                continue
            o = rank7(list(h) + full)
            eq += w * (1.0 if hr < o else 0.5 if hr == o else 0.0)
            tot += w
    return eq / tot


def _preflop_rows(tr, fn, pos, pattern, legal):
    """Per-hand action probabilities at a preflop node: {index: row}."""
    pf, _ = tr._buckets(0, [])
    return {i: fn(make_info_set_key(0, pos, pf[i], None, pattern), tuple(legal))
            for i in range(len(tr.hands)) if tr.w[i] > 0}


def _by_class(tr, rows, legal, classes):
    num, den = defaultdict(lambda: np.zeros(len(legal))), defaultdict(float)
    tot, wsum = np.zeros(len(legal)), 0.0
    for i, row in rows.items():
        c = hand_class(tr.hands[i])
        num[c] += tr.w[i] * row
        den[c] += tr.w[i]
        tot += tr.w[i] * row
        wsum += tr.w[i]
    out = {'all': dict(zip(legal, np.round(tot / wsum, 4).tolist()))}
    for c in classes:
        if den.get(c):
            out[c] = dict(zip(legal, np.round(num[c] / den[c], 4).tolist()))
    return out


def replay(path):
    db = BlueprintDB(path, read_only=True)
    mm, kf = db_menu_mode(db), db_key_format(db) or 'v2'
    adir = resolve_abstraction_dir(db, 'bug027 replay')
    cards = CardAbstraction(adir)
    # The tracker's bucket maps are process-global and keyed by board only, so clear
    # them before bucketing with a different abstraction.
    rt._PF_BUCKET_CACHE.clear()
    rt._POSTFLOP_BUCKET_CACHE.clear()
    fn = model_fn(db)
    vil = RangeTracker(HERO, cards)          # the BB's range (hero's cards removed)
    bot = RangeTracker((), cards)            # the bot's own range
    open_legal = _legal_actions(0, 1, 0, 199, 3, mm)
    three_legal = _legal_actions(0, 5, 1, 196, 9, mm)
    four_legal = _legal_actions(0, 10, 2, 190, 24, mm)
    cap_legal = _legal_actions(0, 27, 3, 183, 61, mm)

    bot.observe(fn, 'bet_large', 0, 'ip', '', open_legal, [])
    vil.observe(fn, None, 0, 'oop', 'l', three_legal, [], blend=THREEBET_BLEND)
    four = _by_class(bot, _preflop_rows(bot, fn, 'ip', 'lm', four_legal), four_legal, TOP)
    bot.observe(fn, 'raise_medium', 0, 'ip', 'lm', four_legal, [])
    cap = _by_class(vil, _preflop_rows(vil, fn, 'oop', 'lmm', cap_legal), cap_legal, TOP)
    vil.observe(fn, 'call', 0, 'oop', 'lmm', cap_legal, [])

    vil.reveal(FLOP)
    bot.reveal(FLOP)
    spr = spr_bucket(kf, 1, FLOP_POT)
    pf, st = vil._buckets(1, FLOP)
    keys = {i: make_info_set_key(1, 'oop', pf[i], st[i], '', spr=spr)
            for i in range(len(vil.hands)) if vil.w[i] > 0}
    legal = sorted({a for k in set(keys.values()) for a in (db.get_average_strategy(k) or {})})
    if 'allin' not in legal:
        raise SystemExit(f"{path}: no flop lead all-in in this blueprint's tree ({legal})")
    pre = vil.normalized_weights().copy()
    shove = np.zeros(len(vil.hands))
    for i, k in keys.items():
        shove[i] = fn(k, tuple(legal))[legal.index('allin')]
    post = pre * shove
    p_shove = float(post.sum())
    post /= post.sum()
    cats = defaultdict(float)
    draws = {'flush draw': 0.0, 'straight draw only': 0.0, 'air': 0.0}
    classes = defaultdict(float)
    for i, h in enumerate(vil.hands):
        if post[i] <= 0:
            continue
        c, fd, sd = category(h)
        cats[c] += post[i]
        classes[hand_class(h)] += post[i]
        if fd:
            draws['flush draw'] += post[i]
        elif sd:
            draws['straight draw only'] += post[i]
        elif c == 'no pair':
            draws['air'] += post[i]
    villain = [(h, post[i]) for i, h in enumerate(vil.hands) if post[i] > 1e-9]
    eq = exact_equity(HERO, FLOP, villain)
    final_pot = FLOP_POT + 2 * BEHIND
    # The bot's call frequency facing the shove, over its own range.
    bw = bot.normalized_weights()
    bpf, bst = bot._buckets(1, FLOP)
    callmass = tot = 0.0
    hero_t = tuple(sorted(HERO, key=_FULL_DECK.index))
    hero_key = None
    for i, h in enumerate(bot.hands):
        if bw[i] <= 0:
            continue
        k = make_info_set_key(1, 'ip', bpf[i], bst[i], 'a', spr=spr)
        if h == hero_t:
            hero_key = k
        callmass += bw[i] * fn(k, ('fold', 'call'))[1]
        tot += bw[i]
    db.close()
    top = lambda d, n: [(k, round(v, 4)) for k, v in sorted(d.items(), key=lambda x: -x[1])[:n]]
    return {
        'db': os.path.basename(path), 'menu_mode': mm, 'key_format': kf,
        'abstraction': 'root' if os.path.basename(adir) == 'abstractions' else os.path.basename(adir),
        'bot_4bet_at_lm': four, 'bb_vs_4bet_at_lmm': cap,
        'flop_spr_bucket': spr, 'flop_lead_legal': legal,
        'bb_lead_shove_freq': round(p_shove, 4),
        'shove_categories': top(cats, 10), 'shove_draws': {k: round(v, 4) for k, v in draws.items()},
        'shove_top_classes': top(classes, 15),
        'hero_equity_vs_shove': round(eq, 4), 'needed_equity': round(BEHIND / final_pot, 4),
        'hero_call_ev_chips': round(eq * final_pot - BEHIND, 2),
        'hero_key': hero_key, 'hero_key_strategy': fn(hero_key, ('fold', 'call')).round(4).tolist(),
        'bot_range_call_freq': round(callmass / tot, 4),
    }


def realistic():
    """J9hh's equity vs human-style shove ranges from a 4-bet-calling range."""
    dead = set(HERO) | set(FLOP)
    base = [h for h in combinations(_FULL_DECK, 2) if h[0] not in dead and h[1] not in dead
            and hand_class(h) in ('AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo', 'AQs', 'AQo',
                                  'KQs', 'AJs')]
    strict = [h for h in base if category(h)[0] in ('set', 'top pair', 'two pair')
              or (category(h)[1] and hand_class(h) in ('AKs', 'AJs'))]
    plus = strict + [h for h in base if category(h)[0] == 'overpair']
    return {name: round(exact_equity(HERO, FLOP, [(h, 1.0) for h in r]), 4)
            for name, r in (('Qx, sets, nut flush draws', strict),
                            ('plus AA/KK', plus), ('the whole 4-bet-calling range', base))}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--db', action='append', required=True)
    p.add_argument('--json', default=None)
    args = p.parse_args()
    out = {'blueprints': []}
    for path in args.db:
        r = replay(path)
        out['blueprints'].append(r)
        print(f"\n== {r['db']}  ({r['menu_mode']}, {r['key_format']}, abstraction {r['abstraction']})")
        print(f"  bot 4-bets at 'lm' (raise_medium): all {r['bot_4bet_at_lm']['all'].get('raise_medium', 0):.3f}  "
              f"J9s {r['bot_4bet_at_lm'].get('J9s', {}).get('raise_medium', 0):.3f}  "
              f"J9o {r['bot_4bet_at_lm'].get('J9o', {}).get('raise_medium', 0):.3f}")
        print("  BB facing the 4-bet:")
        for c, d in r['bb_vs_4bet_at_lmm'].items():
            print(f"    {c:4s} " + '  '.join(f"{a} {v:.3f}" for a, v in d.items()))
        print(f"  BB flop lead-shove freq {r['bb_lead_shove_freq']:.3f} (spr bucket {r['flop_spr_bucket']})")
        print(f"  shove range: {r['shove_categories']}")
        print(f"               {r['shove_draws']}")
        print(f"  J9hh equity vs shove {r['hero_equity_vs_shove']:.3f} (needs {r['needed_equity']:.3f}), "
              f"call EV {r['hero_call_ev_chips']:+.1f} chips; hero key {r['hero_key']} "
              f"fold/call {r['hero_key_strategy']}; bot range calls {r['bot_range_call_freq']:.3f}")
    out['realistic'] = realistic()
    print(f"\n  J9hh equity vs realistic human shove ranges: {out['realistic']}")
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1, default=str)


if __name__ == '__main__':
    main()
