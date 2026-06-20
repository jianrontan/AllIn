#!/usr/bin/env python3
"""Build the hierarchical-backoff opponent model from the fitted counts (Phase 6 / E1).

Turns the per-player full-key counts (fit_opponent_models.py) into a serve-time model that,
for ANY blueprint info-set key, returns the player's estimated action distribution P(a|key).
The estimate is a backoff cascade (so a sparse fine key borrows from data-rich coarse levels
instead of collapsing to GTO):

    blueprint(key)  ->  strength x pos x street  ->  + facing-bet-size  ->  full key

Each rung is an empirical-Bayes posterior whose PRIOR is the coarser rung (the blueprint is the
bottom prior). Keeps the ORIGINAL blueprint info-set scheme -- the output is keyed by the full
key, a drop-in strategy_fn for the range tracker; the coarse rungs are only how we ESTIMATE it.

Outputs serve-time model_built_<id>.json (backoff count tables + alpha) and prints, per
high-volume player, the top leaks vs GTO (the exploitation signal, in plain terms).

Run from backend/bot:
    python scripts/build_opponent_model.py --db analysis/blueprints/snapshots/snap_52500000.db
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.abstractions.postflop_features import load_centroids      # noqa: E402
from src.storage.blueprint_db import BlueprintDB                   # noqa: E402
from src.exploitation.opponent_model import (                      # noqa: E402  the SHARED estimator
    _levels, _eb, _bp_chardist, estimate, _player_levels)          #   (one impl, used by serving too)

_CHAR_NAME = {'f': 'fold', 'c': 'call', 'k': 'check', 's': 'bet/raise-small',
              'm': 'bet/raise-med', 'l': 'bet/raise-large', 'x': 'open-xlarge',
              'o': 'overbet', '2': 'overbet2', 'a': 'all-in'}


def build_player(level_counts, key, bp, alpha):
    """Serving estimate: fetch the blueprint prior for `key`, then run the shared estimator."""
    return estimate(level_counts, key, _bp_chardist(bp, key), alpha)


def check_abstraction(bp):
    """Guard the silent strength-bucket misalignment (audit C1): the fitted model's keys are
    bucketed by the LOCAL centroids, but the prior comes from --db. If --db was trained under a
    DIFFERENT-K abstraction, the strength digit means different things in the two and lookups are
    silently wrong. We can't read the DB's K directly (not stamped), so we infer it from the
    largest strength index actually seen in the DB's keys (== K-1 once every bucket is used over
    a long run) and compare to the centroid K per street."""
    ck = {s: len(load_centroids(s)[0]) for s in ('flop', 'turn', 'river')}
    obs = {'flop': -1, 'turn': -1, 'river': -1}
    for (key,) in bp.conn.execute("SELECT key FROM info_sets"):
        p = key.split('_')
        if len(p) >= 5 and p[4] in obs:                 # pf_C_S_pos_street_pattern
            try:
                obs[p[4]] = max(obs[p[4]], int(p[2]))
            except ValueError:
                pass
    print("  abstraction check (centroid K vs blueprint observed strength):")
    fatal = []
    for s in ('flop', 'turn', 'river'):
        db_k = obs[s] + 1
        gap = ck[s] - db_k
        tag = ('OK' if gap == 0 else
               'minor (unused top bucket?)' if gap == 1 else 'MISMATCH')
        print(f"    {s:<6} centroids K={ck[s]:>3}  blueprint max-strength+1={db_k:>3}   {tag}")
        if gap < 0 or gap >= 2:                          # DB finer, or centroids >=2 finer than DB
            fatal.append(s)
    if fatal:
        raise SystemExit(
            f"\nABSTRACTION MISMATCH on {fatal}: the local postflop centroids do NOT match the "
            f"--db blueprint's bucketing -> the model's strength buckets would be keyed to a "
            f"different quantization than the blueprint (silently wrong lookups). Point --db at "
            f"the blueprint trained under THESE centroids, or re-bake/re-fit. Aborting.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='analysis/opponent_models')
    ap.add_argument('--db', default='analysis/blueprints/snapshots/snap_52500000.db')
    ap.add_argument('--alpha', type=float, default=10.0,
                    help='EB shrinkage strength per rung. Default 10 = the held-out-tuned '
                         'KNOWN-player optimum (tune_alpha.py). The population/cold-start model '
                         'wants ~160 (build it with --alpha 160 from model_population.json).')
    ap.add_argument('--min-visits', type=int, default=15,
                    help='min full-key visits for a leak to appear in the demo. The z>=2 '
                         'credible-interval gate is the real phantom filter; this is a light floor.')
    ap.add_argument('--top', type=int, default=8, help='leaks to show per player')
    args = ap.parse_args()

    bp = BlueprintDB(args.db, read_only=True)
    check_abstraction(bp)                              # audit C1: fail fast on a wrong --db
    files = sorted(glob.glob(os.path.join(args.dir, 'model_*.json')))
    players = []
    for f in files:
        base = os.path.basename(f)
        if base == 'model_population.json' or base.startswith('model_built_'):
            continue
        with open(f, encoding='utf-8') as fh:
            m = json.load(fh)
        players.append(m)
    players.sort(key=lambda m: m['hands'], reverse=True)

    for m in players:
        lv = _player_levels(m['counts'])
        # Save the serve-time model (backoff tables + alpha).
        # 'hands' lets the serve-time min-hands gate (HumanModel) skip thin per-player models;
        # alpha is NOT stored -- serving sets it from env (ALLIN_EXPLOIT_ALPHA), one knob for all.
        out = {'playerId': m['playerId'], 'handle': m['handle'], 'hands': m['hands'],
               'levels': {r: {k: dict(c) for k, c in tbl.items()} for r, tbl in lv.items()}}
        with open(os.path.join(args.dir, f"model_built_{m['playerId']}.json"), 'w',
                  encoding='utf-8') as fh:
            json.dump(out, fh, separators=(',', ':'))

        if m['hands'] < 300:                       # only demo the high-volume players
            continue
        leaks = []
        for key, chars in m['counts'].items():
            n = sum(chars.values())
            if n < args.min_visits:
                continue
            est = build_player(lv, key, bp, args.alpha)
            base = _bp_chardist(bp, key)
            if base is None:
                continue
            # CREDIBLE-INTERVAL gate (audit #4): keep a leak only if GTO falls OUTSIDE the
            # action's ~95% posterior interval (z>=2) AND the effect is material (>=10pts) --
            # this is the multiple-comparisons / regression-to-the-mean control, replacing the
            # old point-threshold that surfaced phantom leaks at thin keys.
            for a in set(est) | set(base):
                e, g = est.get(a, 0.0), base.get(a, 0.0)
                se = (max(e * (1.0 - e), 1e-6) / (n + args.alpha)) ** 0.5
                z = abs(e - g) / se
                if z >= 2.0 and abs(e - g) >= 0.10:
                    leaks.append((z, key, a, g, e, n))
        leaks.sort(reverse=True)
        print(f"\n=== {m['handle']} ({m['hands']} hands) - top leaks vs GTO ===")
        if not leaks:
            print("  (no well-supported deviation - plays close to GTO on its frequent spots)")
        seen = set()
        shown = 0
        for _, key, a, gto, hum, n in leaks:
            if key in seen:
                continue
            seen.add(key)
            aggr = ('s', 'm', 'l', 'o', '2', 'a')
            if a == 'f':
                hint = 'over-folds -> bluff MORE' if hum > gto else 'under-folds -> value thinner'
            elif a in ('c', 'k'):
                hint = ('calls/checks too much -> value thinner, bluff LESS' if hum > gto
                        else 'checks/calls too little -> more aggressive than GTO')
            else:                                   # a is a bet/raise size
                hint = ('over-aggressive -> call/trap wider, bluff LESS' if hum > gto
                        else 'under-bets this spot')
            print(f"  {key:<28} {_CHAR_NAME.get(a, a):<16} GTO {gto*100:4.0f}% -> "
                  f"{hum*100:4.0f}%  (n={n:>3})  {hint}")
            shown += 1
            if shown >= args.top:
                break
    bp.close()
    print(f"\nwrote serve-time model_built_<id>.json to {args.dir}")


if __name__ == '__main__':
    main()
