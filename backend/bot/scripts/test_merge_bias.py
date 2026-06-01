"""
Decisive A/B test for the parallel-CFR+ high-variance regret-bias hypothesis.

The block-CFR+ merge floors each worker's regret against its own copy, so
cross-worker negative regret can't cancel positive regret WITHIN a round -> a
bounded upward bias on high-variance actions (jam / call-off) that scales with
`workers * merge_every`. If that bias is what's making the served bot over-jam
and over-call all-ins, then a run with a MUCH smaller merge_every (≈ no batching
bias) should jam/call-off LESS at the same info-sets.

So: train two FRESH parallel runs, identical except merge_every, and compare the
average-strategy jam frequency (P(allin)) and call-off frequency (P(call) when
facing an all-in) at PREFLOP info-sets present in both.

  Arm A: merge_every=4000  (the production setting -> large rounds -> more bias)
  Arm B: merge_every=250   (16x smaller rounds -> ~16x less bias)

Verdict:
  * A jams/calls-off NOTICEABLY more than B  -> the parallel bias is real and
    material; the served blueprint is compromised; retrain small-merge/single-thread.
  * A ≈ B                                    -> NOT the merge bias; look to the
    SPR-blind key (M1) and/or genuine GTO looseness.

Run from backend/bot/:  python scripts/test_merge_bias.py --iters 500000 --workers 6
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.blueprint_trainer import BlueprintTrainer
from src.cfr.parallel_trainer import train_blueprint_parallel


def run_arm(label, merge_every, iters, workers, seed):
    print(f"\n########## ARM {label}: merge_every={merge_every}, "
          f"workers={workers}, iters={iters:,} ##########", flush=True)
    random.seed(seed)
    t = BlueprintTrainer()
    t0 = time.time()
    train_blueprint_parallel(t, iters, db=None, workers=workers,
                             merge_every=merge_every, seed=seed)
    print(f"ARM {label} done in {time.time()-t0:.0f}s, {len(t.info_sets):,} info sets",
          flush=True)
    return t.info_sets


def _is_preflop_key(key):
    # preflop: {fineBucket}_{pos}_{pattern} -> 4 tokens (e.g. pf_27_ip_a, pf_3_oop_).
    # postflop: {coarse}_{strength}_{pos}_{street}_{pattern} -> 6 tokens.
    return len(key.split('_')) == 4


def _metrics(info_sets):
    """Per preflop key: P(allin), and P(call) when the pattern shows a faced
    all-in. Returns {key: (mass, p_allin_or_None, p_calloff_or_None)}."""
    out = {}
    for key, info in info_sets.items():
        if not _is_preflop_key(key):
            continue
        legal = info.legal_actions
        if not legal:
            continue
        strat = info.get_average_strategy(legal)
        prob = {a: float(p) for a, p in zip(legal, strat)}
        mass = float(sum(info.cumulative_strategy.values()))
        pattern = key.split('_')[3]
        p_allin = prob.get('allin') if 'allin' in legal else None
        # Facing an all-in: last action in the line was a jam (pattern ends 'a')
        # and the only responses are fold/call -> P(call) is the call-off rate.
        p_calloff = prob.get('call') if (pattern.endswith('a') and 'call' in legal) else None
        out[key] = (mass, p_allin, p_calloff)
    return out


def _wmean(pairs):
    """Mass-weighted mean of (mass, value) pairs, ignoring value=None."""
    num = den = 0.0
    for mass, val in pairs:
        if val is not None:
            num += mass * val
            den += mass
    return (num / den) if den else float('nan'), den


def compare(infoA, infoB, min_mass):
    mA, mB = _metrics(infoA), _metrics(infoB)
    keys = [k for k in mA.keys() & mB.keys()
            if mA[k][0] >= min_mass and mB[k][0] >= min_mass]
    keys.sort()

    jamA = [(mA[k][0], mA[k][1]) for k in keys if mA[k][1] is not None]
    jamB = [(mB[k][0], mB[k][1]) for k in keys if mB[k][1] is not None]
    coA = [(mA[k][0], mA[k][2]) for k in keys if mA[k][2] is not None]
    coB = [(mB[k][0], mB[k][2]) for k in keys if mB[k][2] is not None]

    jA, _ = _wmean(jamA)
    jB, _ = _wmean(jamB)
    cA, _ = _wmean(coA)
    cB, _ = _wmean(coB)

    print("\n==================== VERDICT ====================")
    print(f"matched preflop info-sets (mass >= {min_mass}): {len(keys):,}")
    print(f"  keys where allin is legal      : {len(jamA):,}")
    print(f"  keys facing an all-in (call-off): {len(coA):,}")
    print(f"\n  mass-weighted P(allin)   ARM A (merge 4000): {jA:.4f}")
    print(f"  mass-weighted P(allin)   ARM B (merge  250): {jB:.4f}")
    if jB:
        print(f"    -> A/B ratio: {jA/jB:.2f}x   (A jams {(jA-jB):+.4f} more)")
    print(f"\n  mass-weighted P(call-off) ARM A (merge 4000): {cA:.4f}")
    print(f"  mass-weighted P(call-off) ARM B (merge  250): {cB:.4f}")
    if cB:
        print(f"    -> A/B ratio: {cA/cB:.2f}x   (A calls {(cA-cB):+.4f} more)")

    # Biggest per-key jam-frequency gaps (A - B).
    gaps = [(k, mA[k][1] - mB[k][1]) for k in keys
            if mA[k][1] is not None and mB[k][1] is not None]
    gaps.sort(key=lambda kv: -kv[1])
    print("\n  top 12 preflop keys by P(allin) gap (A - B):")
    print(f"  {'key':<22}{'A jam':>8}{'B jam':>8}{'gap':>8}")
    for k, g in gaps[:12]:
        print(f"  {k:<22}{mA[k][1]:>8.3f}{mB[k][1]:>8.3f}{g:>8.3f}")
    print("=================================================")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=500_000)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--merge-a', type=int, default=4000)
    p.add_argument('--merge-b', type=int, default=250)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--min-mass', type=float, default=5.0,
                   help="Min cumulative-strategy mass for a key to count (noise filter).")
    args = p.parse_args()

    infoA = run_arm('A', args.merge_a, args.iters, args.workers, args.seed)
    infoB = run_arm('B', args.merge_b, args.iters, args.workers, args.seed)
    compare(infoA, infoB, args.min_mass)


if __name__ == '__main__':
    main()
