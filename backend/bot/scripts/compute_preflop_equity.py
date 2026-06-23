# backend/bot/scripts/compute_preflop_equity.py
"""
One-time script: compute equity of all 169 canonical preflop hands vs. a random
opponent using Monte Carlo simulation with phevaluator.

Run from backend/bot/:
    python scripts/compute_preflop_equity.py

Prints a Python dict literal ready to paste into card_abstractions.py.
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from phevaluator.evaluator import evaluate_cards

SUITS = ['H', 'D', 'C', 'S']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
DECK  = [s + r for r in RANKS for s in SUITS]
RANK_VAL = {r: i for i, r in enumerate(RANKS)}

SIMULATIONS = 10000
RANDOM_SEED = 42
# Decoupled preflop scheme: FINE buckets identify the hand for preflop keys; COARSE
# classes are the postflop startBucket (imperfect recall). card_abstractions.py DERIVES
# both maps from the committed _PREFLOP_EQUITY table at import (NOT from these counts), so
# this script is ONLY the equity GENERATOR. NUM_BUCKETS/NUM_COARSE below are DIAGNOSTIC-ONLY
# (they drive this script's own printout, not the live abstraction). The live scheme is
# lossless 169 fine / 10 coarse (card_abstractions.NUM_PREFLOP_BUCKETS / NUM_PREFLOP_COARSE).
NUM_BUCKETS = 169         # fine (diagnostic only; lossless = one bucket per canonical hand)
NUM_COARSE = 10           # coarse (diagnostic only)


def to_phev(card):
    suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
    return card[1] + suit_map[card[0]]


def canonical(c1, c2):
    r1, s1 = c1[1], c1[0]
    r2, s2 = c2[1], c2[0]
    if RANK_VAL[r1] < RANK_VAL[r2]:
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ('s' if s1 == s2 else 'o')


def collect_hand_reps():
    reps = {}
    for i, c1 in enumerate(DECK):
        for c2 in DECK[i + 1:]:
            k = canonical(c1, c2)
            if k not in reps:
                reps[k] = (c1, c2)
    return reps


def compute_equities(hand_reps):
    random.seed(RANDOM_SEED)
    results = {}
    total = len(hand_reps)
    for idx, (hand_key, (c1, c2)) in enumerate(sorted(hand_reps.items()), 1):
        rest = [c for c in DECK if c not in (c1, c2)]
        wins = ties = 0
        for _ in range(SIMULATIONS):
            sample = random.sample(rest, 7)
            o1, o2 = sample[0], sample[1]
            board   = sample[2:7]
            me  = evaluate_cards(*[to_phev(x) for x in [c1, c2] + board])
            opp = evaluate_cards(*[to_phev(x) for x in [o1, o2] + board])
            if me < opp:
                wins += 1
            elif me == opp:
                ties += 1
        results[hand_key] = (wins + 0.5 * ties) / SIMULATIONS
        if idx % 20 == 0 or idx == total:
            print(f"  {idx}/{total} hands computed...", flush=True)
    return results


def assign_buckets(results, n_buckets):
    # Pure equal-frequency quantile over all 169 hands by equity (ascending):
    # pf_0 weakest .. pf_(n_buckets-1) strongest, ~169/n_buckets hands each. Called
    # for BOTH the fine (30) and coarse (10) maps; card_abstractions.py derives both
    # from the committed equity table via this same formula. The strongest bucket
    # separates naturally (JJ/QQ/KK/AA), so no hardcoded top-bucket special case.
    ranked = sorted(results.items(), key=lambda x: x[1])
    total = len(ranked)
    bucket_map = {}
    for i, (hand, eq) in enumerate(ranked):
        bucket = min(int(i * n_buckets / total), n_buckets - 1)
        bucket_map[hand] = f"pf_{bucket}"
    return bucket_map


def main():
    print(f"Computing equity for all 169 hands ({SIMULATIONS} simulations each)...")
    hand_reps = collect_hand_reps()
    assert len(hand_reps) == 169, f"Expected 169 hands, got {len(hand_reps)}"

    results = compute_equities(hand_reps)
    bucket_map = assign_buckets(results, NUM_BUCKETS)

    sorted_by_eq = sorted(results.items(), key=lambda x: -x[1])

    print("\n--- Equity table (strongest to weakest) ---")
    for hand, eq in sorted_by_eq:
        print(f"  {hand:6s}  equity={eq:.4f}  bucket={bucket_map[hand]}")

    print("\n--- Bucket summary ---")
    from collections import defaultdict
    buckets = defaultdict(list)
    for hand, bucket in bucket_map.items():
        buckets[bucket].append((hand, results[hand]))
    for b in sorted(buckets.keys(), key=lambda x: int(x.split('_')[1]), reverse=True):
        hands = sorted(buckets[b], key=lambda x: -x[1])
        eq_range = f"{hands[-1][1]:.3f}â€“{hands[0][1]:.3f}"
        names = ', '.join(h for h, _ in hands)
        print(f"  {b}: [{eq_range}]  {names}")

    coarse_map = assign_buckets(results, NUM_COARSE)
    print("\n--- Coarse class summary (postflop startBucket) ---")
    cbuckets = defaultdict(list)
    for hand, b in coarse_map.items():
        cbuckets[int(b.split('_')[1])].append((hand, results[hand]))
    for b in sorted(cbuckets, reverse=True):
        hands = sorted(cbuckets[b], key=lambda x: -x[1])
        print(f"  class {b}: {', '.join(h for h, _ in hands)}")

    print("\n--- Paste ONLY _PREFLOP_EQUITY into card_abstractions.py ---")
    print("# card_abstractions.py DERIVES the fine (30) and coarse (10) bucket maps")
    print("# from this table at import (via _quantile_buckets), so there is no bucket")
    print("# literal to paste -- only the equity table below.")
    print("_PREFLOP_EQUITY = {")
    for hand, eq in sorted(results.items()):
        print(f"    '{hand}': {eq:.4f},")
    print("}")


if __name__ == "__main__":
    main()

