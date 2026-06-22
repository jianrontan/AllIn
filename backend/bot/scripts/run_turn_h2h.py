#!/usr/bin/env python3
"""Parallel turn-solver-vs-river H2H edge test (8 shards) + combine.

Runs `measure_turn_match.py --paired --measure turn` (the CRN-paired turn-solver gain over river-only,
AIVAT-reduced) across N shards with different seeds, then pools the per-shard (mean, se, n) into one
aggregate. THE question: does a higher-fidelity (24-river) turn solve ADD EV over river-only? Uses
--opponent maxbet so pots build -> low-SPR turns -> the turn solver actually fires.

  python scripts/run_turn_h2h.py --rivers 24 --shards 8 --hands-per 450
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT = os.path.dirname(_HERE)

_PAIRED = re.compile(r'PAIRED \([^)]*\) = ([+-]?[\d.]+) \+/- ([\d.]+) mbb/hand over (\d+) hands.*?diverged on (\d+)')
_AIVAT = re.compile(r'AIVAT \([^)]*\) = ([+-]?[\d.]+) \+/- ([\d.]+) mbb/hand')
_DEVNET = re.compile(r'PER-DEVIATION \((\d+) diverged\): (\d+) win \(\+([\d.]+)\) / (\d+) lose \(([+-]?[\d.]+)\).*?net ([+-]?[\d.]+)')


def pool(rows):
    """rows = [(mean, se, n)] -> (pooled_mean, pooled_se, total_n)."""
    rows = [r for r in rows if r and r[2] > 0]
    if not rows:
        return None
    N = sum(n for _, _, n in rows)
    m = sum(mean * n for mean, _, n in rows) / N
    se = math.sqrt(sum((n * se) ** 2 for _, se, n in rows)) / N   # pooled SE of the mean
    return m, se, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rivers', type=int, default=24)
    ap.add_argument('--buckets', type=int, default=24)
    ap.add_argument('--shards', type=int, default=8)
    ap.add_argument('--hands-per', type=int, default=450)
    ap.add_argument('--turn-budget', type=float, default=150.0)
    ap.add_argument('--opponent', default='maxbet')
    ap.add_argument('--multivalued', action='store_true',
                    help="use the robust Modicum multi-valued leaf (fixes the optimistic-leaf trap)")
    ap.add_argument('--logdir', default=os.path.join(_BOT, 'analysis', 'h2h_logs'))
    args = ap.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    cmd_base = [sys.executable, os.path.join(_HERE, 'measure_turn_match.py'),
                '--paired', '--measure', 'turn', '--aivat',
                '--leaf-rivers', str(args.rivers), '--n-buckets', str(args.buckets),
                '--turn-budget', str(args.turn_budget), '--turn-max-spr', '10',
                '--opponent', args.opponent, '--hands', str(args.hands_per)]
    if args.multivalued:
        cmd_base.append('--multivalued-leaf')
    procs, logs = [], []
    t0 = time.time()
    print(f"launching {args.shards} shards @ {args.rivers} rivers, {args.hands_per} hands each "
          f"({args.shards * args.hands_per} total), opponent={args.opponent} ...", flush=True)
    for i in range(args.shards):
        lp = os.path.join(args.logdir, f'shard_{i}.log')
        lf = open(lp, 'w')
        p = subprocess.Popen(cmd_base + ['--seed', str(1000 + i)], cwd=_BOT, stdout=lf,
                             stderr=subprocess.STDOUT, env={**os.environ, 'ALLIN_MMAP_POSTFLOP': '1'})
        procs.append((p, lf)); logs.append(lp)
    print(f"  shards running; tail logs in {args.logdir}/shard_*.log", flush=True)
    for p, lf in procs:
        p.wait(); lf.close()
    print(f"  all shards done ({(time.time()-t0)/60:.0f} min). combining ...\n", flush=True)

    paired, aivat = [], []
    dev_win = dev_lose = dev_n = 0
    for lp in logs:
        txt = open(lp, encoding='utf-8', errors='ignore').read()
        mp = _PAIRED.search(txt)
        if mp:
            paired.append((float(mp.group(1)), float(mp.group(2)), int(mp.group(3))))
        ma = _AIVAT.search(txt)
        if ma:
            aivat.append((float(ma.group(1)), float(ma.group(2)), int(mp.group(3)) if mp else 0))
        md = _DEVNET.search(txt)
        if md:
            dev_n += int(md.group(1)); dev_win += float(md.group(3)); dev_lose += float(md.group(5))

    pp = pool(paired)
    pa = pool(aivat)
    print("=" * 60)
    if pp:
        m, se, n = pp
        print(f"POOLED PAIRED (turn - river): {m:+.1f} +/- {se:.1f} mbb/hand  over {n} hands  "
              f"(|t|={abs(m)/se if se else 0:.2f})")
    if pa:
        m, se, n = pa
        print(f"POOLED AIVAT  (turn - river): {m:+.1f} +/- {se:.1f} mbb/hand  (|t|={abs(m)/se if se else 0:.2f})")
    print(f"PER-DEVIATION: {dev_n} diverged -> win +{dev_win:.0f} / lose {dev_lose:.0f} -> "
          f"net {dev_win+dev_lose:+.0f} mbb-total")
    print("  >0 beyond ~2se => the 24-river turn solve ADDS EV over river-only -> NN justified.")
    print("  ~0 / negative => higher fidelity doesn't help -> stop (no NN).")


if __name__ == '__main__':
    main()
