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
NUM_BUCKETS = 15


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


TOP_EQUITY_THRESHOLD = 0.75  # equity >= threshold → pf_14 (TT+)


def assign_buckets(results, n_buckets):
    # pf_14: hardcoded top bucket for hands with equity >= TOP_EQUITY_THRESHOLD (TT+)
    # Remaining hands: equal-quantile into pf_0 through pf_(n_buckets-2)
    top = {k: v for k, v in results.items() if v >= TOP_EQUITY_THRESHOLD}
    rest = sorted([(k, v) for k, v in results.items() if v < TOP_EQUITY_THRESHOLD],
                  key=lambda x: x[1])
    n_rest_buckets = n_buckets - 1
    total = len(rest)
    bucket_map = {}
    for i, (hand, eq) in enumerate(rest):
        bucket = int(i * n_rest_buckets / total)
        bucket = min(bucket, n_rest_buckets - 1)
        bucket_map[hand] = f"pf_{bucket}"
    for hand in top:
        bucket_map[hand] = f"pf_{n_buckets - 1}"
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

    print("\n--- Paste this into card_abstractions.py ---")
    print("_PREFLOP_EQUITY = {")
    for hand, eq in sorted(results.items()):
        print(f"    '{hand}': {eq:.4f},")
    print("}")
    print()
    print("_PREFLOP_BUCKET_MAP = {")
    for hand, bucket in sorted(bucket_map.items()):
        print(f"    '{hand}': '{bucket}',")
    print("}")


if __name__ == "__main__":
    main()

