#!/usr/bin/env python3
"""Live exploitation A/B analysis: split recaps by their recorded `exploitArm` ('on' = exploit,
'off' = pure-blueprint control) and compare the BOT's win-rate (bb/100) per arm.

The arm is assigned per session (`ALLIN_AB_ARM=random` for a prod 50/50 split by player; `on`/`off`
forces it for dev to play each side) and stamped on every recap by `recap_from_session`. This is the
LIVE EV ORACLE the offline scoreboard (`measure_exploit.py`) cannot be -- but it only resolves a small
(sub-BB/hand) delta with MANY hands across MANY players; a single dev player is far too noisy.

Reads an export JSONL (re-run `scripts/export_hands.py` against the live store first):
  python scripts/analyze_ab.py --in analysis/opponent_models/hands_export.jsonl
"""
import argparse
import json
import math
from collections import defaultdict

_BB = 2.0   # SB=1/BB=2; humanDelta is in chips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='analysis/opponent_models/hands_export.jsonl')
    args = ap.parse_args()

    arms = defaultdict(list)        # arm -> list of BOT bb/hand
    players = defaultdict(set)      # arm -> distinct players
    n_total = n_tagged = 0
    with open(args.inp, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            n_total += 1
            arm = r.get('exploitArm')
            if arm not in ('on', 'off'):
                continue            # untagged (pre-A/B) hands
            hd = (r.get('result') or {}).get('humanDelta')
            if hd is None:
                continue
            n_tagged += 1
            arms[arm].append(-float(hd) / _BB)          # BOT bb/hand = -human
            players[arm].add(r.get('playerId'))

    print(f"{n_total} recaps, {n_tagged} tagged with an A/B arm\n")
    stats = {}
    for arm in ('on', 'off'):
        xs = arms.get(arm, [])
        n = len(xs)
        if n == 0:
            print(f"  arm {arm:3}: 0 hands")
            continue
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n if n > 1 else 0.0
        se = math.sqrt(var / n) if n > 1 else 0.0
        stats[arm] = (mean, se, n)
        print(f"  arm {arm:3}: {n:6} hands / {len(players[arm])} players | "
              f"bot {mean * 100:+.1f} bb/100  +/- {se * 100 * 1.96:.1f} (95%)")

    if 'on' in stats and 'off' in stats:
        (m1, s1, _), (m0, s0, _) = stats['on'], stats['off']
        diff = m1 - m0
        sed = math.sqrt(s1 * s1 + s0 * s0)
        t = diff / sed if sed > 0 else 0.0
        print(f"\n  EXPLOIT - CONTROL: {diff * 100:+.1f} bb/100  +/- {sed * 100 * 1.96:.1f} (95%)  "
              f"|t|={abs(t):.2f}")
        print("  (|t|>=2 ~ significant; needs MANY hands across MANY players to resolve a small delta)")
    else:
        print("\n  Need BOTH arms with data. Play/serve some hands on each (ALLIN_AB_ARM=random in prod,"
              " or alternate on/off in dev), re-export, and re-run.")


if __name__ == '__main__':
    main()
