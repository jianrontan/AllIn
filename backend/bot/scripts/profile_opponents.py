#!/usr/bin/env python3
"""Profile + quality-screen exported opponents (Phase 6 / E0 cleaning step).

Reads the JSONL produced by `export_hands.py` and, per player (the HUMAN side of each
hand recap), prints: hands/sessions, the action-category mix, VPIP/PFR, postflop
aggression frequency, overall fold rate, mean BB/hand, showdown rate, plus suspicious-
pattern FLAGS and a keep / population-only / drop SUGGESTION. Also dumps a JSON summary.

This is the gate that decides which of the leaderboard players are real (bad) players worth
modeling vs button-mashers/trolls whose data would poison the fit (see EXPLOITATION_PLAN §5.3).
It is read-only and needs no AWS / no blueprint.

Run from backend/bot:
    python scripts/profile_opponents.py --in analysis/opponent_models/hands_export.jsonl
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

_STREETS = ('preflop', 'flop', 'turn', 'river')
_BB = 2.0           # recap result.humanDelta is in CHIPS; BB = 2 chips (SB=1/BB=2)


def _street(s):
    """actionLog street may be an int 0-3 or a name; normalise to a name."""
    if isinstance(s, bool):
        return str(s)
    if isinstance(s, (int, float)):
        i = int(s)
        return _STREETS[i] if 0 <= i < 4 else str(s)
    return str(s).lower()


def _cat(action):
    """Coarse action category, defensive against the exact engine vocabulary
    (fold / check / call / aggr=bet|raise|custom / allin / other). 'other' is surfaced
    so an unrecognised string can't be silently miscounted."""
    a = str(action).lower()
    if a.startswith('fold'):
        return 'fold'
    if a.startswith('check'):
        return 'check'
    if a.startswith('call'):
        return 'call'
    if a.startswith('all') or a == 'allin':           # allin / all_in / all-in
        return 'allin'
    if a.startswith(('bet', 'raise')) or 'custom' in a:
        return 'aggr'
    return 'other'


def _load(path):
    by_player = defaultdict(list)
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_player[(r.get('_handle', 'anon'), r.get('playerId'))].append(r)
    return by_player


def _profile(hands):
    """Aggregate one player's recaps into a metrics dict."""
    sessions = set()
    cats = Counter()                 # human action categories (all streets)
    raw_actions = Counter()          # raw action strings (vocabulary inventory)
    post_cats = Counter()            # postflop only
    n_vpip = n_pfr = 0
    folds = decisions = 0
    showdowns = 0
    deltas = []
    per_session_hands = Counter()

    for r in hands:
        sid = r.get('sessionId')
        sessions.add(sid)
        per_session_hands[sid] += 1
        hs = r.get('humanSeat')
        log = r.get('actionLog') or []
        res = r.get('result') or {}
        if res.get('reason') == 'showdown':
            showdowns += 1
        deltas.append(float(res.get('humanDelta') or 0.0))

        pf_voluntary = pf_raised = False
        for a in log:
            if a.get('player') != hs:
                continue
            st = _street(a.get('street'))
            c = _cat(a.get('action'))
            cats[c] += 1
            raw_actions[str(a.get('action'))] += 1
            decisions += 1
            if c == 'fold':
                folds += 1
            if st != 'preflop':
                post_cats[c] += 1
            else:
                if c in ('call', 'aggr', 'allin'):
                    pf_voluntary = True
                if c in ('aggr', 'allin'):
                    pf_raised = True
        n_vpip += int(pf_voluntary)
        n_pfr += int(pf_raised)

    n = len(hands)
    post_total = sum(post_cats.values())
    post_aggr = post_cats['aggr'] + post_cats['allin']
    aggr_freq = (post_aggr / post_total) if post_total else 0.0          # bet+raise / all postflop
    af = (post_aggr / post_cats['call']) if post_cats['call'] else float('inf') if post_aggr else 0.0
    fold_rate = (folds / decisions) if decisions else 0.0

    # Near-uniform action mix => button-masher signal. Normalised Shannon entropy over the
    # categories actually used (1.0 = perfectly uniform across them).
    used = [v for v in cats.values() if v > 0]
    tot = sum(used)
    ent = -sum((v / tot) * math.log(v / tot) for v in used) if tot else 0.0
    norm_ent = ent / math.log(len(used)) if len(used) > 1 else 0.0

    med_sess = sorted(per_session_hands.values())
    med_sess = med_sess[len(med_sess) // 2] if med_sess else 0

    return {
        'hands': n,
        'sessions': len(sessions),
        'median_hands_per_session': med_sess,
        'vpip': round(n_vpip / n, 3) if n else 0.0,
        'pfr': round(n_pfr / n, 3) if n else 0.0,
        'postflop_aggr_freq': round(aggr_freq, 3),
        'aggression_factor': (round(af, 2) if math.isfinite(af) else None),
        'fold_rate': round(fold_rate, 3),
        'showdown_rate': round(showdowns / n, 3) if n else 0.0,
        'mean_bb_per_hand': round(sum(deltas) / n / _BB, 2) if n else 0.0,
        'action_mix': {k: round(v / tot, 3) for k, v in cats.items()} if tot else {},
        'action_mix_entropy': round(norm_ent, 3),
        'raw_actions': dict(raw_actions),
        'decisions': decisions,
    }


def _flags(m, min_hands):
    f = []
    if m['hands'] < min_hands:
        f.append('few-hands')                                  # too thin for a per-player model
    if m['fold_rate'] > 0.85:
        f.append('high-fold')                                  # folds almost everything (troll proxy)
    if m['action_mix_entropy'] > 0.95 and m['decisions'] >= 50:
        f.append('uniform-mix')                                # near-random action choice
    if m['median_hands_per_session'] <= 1:
        f.append('tiny-sessions')                              # one-and-done / not engaged
    if m['action_mix'].get('other', 0) > 0:
        f.append('unknown-actions')                            # vocabulary the parser didn't recognise
    return f


def _suggest(flags):
    if 'high-fold' in flags or 'uniform-mix' in flags:
        return 'DROP'
    if 'few-hands' in flags:
        return 'population-only'
    return 'per-player'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='analysis/opponent_models/hands_export.jsonl')
    ap.add_argument('--out', default='analysis/opponent_models/quality_report.json')
    ap.add_argument('--min-hands', type=int, default=100,
                    help='below this many hands -> population-only (per-player too noisy)')
    args = ap.parse_args()

    by_player = _load(args.inp)
    report = []
    for (handle, pid), hands in by_player.items():
        m = _profile(hands)
        m['flags'] = _flags(m, args.min_hands)
        m['suggestion'] = _suggest(m['flags'])
        m['handle'], m['playerId'] = handle, pid
        report.append(m)
    report.sort(key=lambda x: x['hands'], reverse=True)

    print(f"profiled {len(report)} players from {args.inp}\n")
    hdr = (f"  {'handle':<16}{'hands':>6}{'VPIP':>6}{'PFR':>6}{'aggF':>6}{'fold':>6}"
           f"{'sd':>6}{'BB/h':>7}  {'suggestion':<15} flags")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in report:
        print(f"  {m['handle']:<16}{m['hands']:>6}{m['vpip']:>6.2f}{m['pfr']:>6.2f}"
              f"{m['postflop_aggr_freq']:>6.2f}{m['fold_rate']:>6.2f}{m['showdown_rate']:>6.2f}"
              f"{m['mean_bb_per_hand']:>7.2f}  {m['suggestion']:<15} {','.join(m['flags'])}")

    # Surface the full raw-action vocabulary once (so we tune the fitter to real strings).
    vocab = Counter()
    for m in report:
        vocab.update(m['raw_actions'])
    print(f"\n  raw action vocabulary ({len(vocab)} distinct): "
          + ", ".join(f"{a} x{n}" for a, n in vocab.most_common()))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == '__main__':
    main()
