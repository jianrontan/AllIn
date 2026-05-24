# backend/bot/scripts/compute_postflop_buckets.py
"""
One-time script: build DISTRIBUTION-AWARE postflop card buckets.

WHY (vs the current heuristic)
------------------------------
Today `BoardTextureEvaluator` returns one of 8 buckets from a hand-tuned
"current strength + board danger" rule. That collapses hands whose equity is
similar NOW but evolves very differently across runouts (a static made hand vs a
polarized draw), so the blueprint is forced to average two strategies that
should differ. We instead describe each hand by the *distribution* of its equity
over future board cards and cluster those distributions -- which automatically
separates static from polarized hands and gives finer gradations.

THE FEATURE
-----------
For a (hole, board) situation we compute, for each runout completion to the
river, the hand's win-rate vs a uniform opponent range on the completed board,
then HISTOGRAM those win-rates over EQUITY_BINS bins. That histogram is the
hand's "equity distribution":
  * river  (0 cards to come) -> a spike: equity is a single number (correct --
            with no cards left, equity is the whole story).
  * turn   (1 to come)       -> histogram over the 44 possible rivers.
  * flop   (2 to come)       -> histogram over sampled turn+river runouts.

CLUSTERING
----------
k-means with EARTH MOVER'S DISTANCE (EMD) between histograms. For 1-D histograms
EMD is just the L1 distance between CDFs -- cheap and exact -- which is the
right metric for distributions (L2 on bins wrongly treats neighbouring equity
bins as unrelated). Output: K centroid histograms per street. At runtime a hand
is bucketed by computing its histogram and taking the nearest centroid (EMD).

Run from backend/bot/:
    python scripts/compute_postflop_buckets.py --street flop --buckets 12 --situations 3000
    python scripts/compute_postflop_buckets.py --street river --buckets 10 --situations 5000

This DRAFT prints cluster occupancy and assigns two textbook hands (top pair vs
nut flush draw on the same flop) to show they land in different buckets. It
saves centroids to analysis/postflop_centroids_<street>.npz.
"""
import argparse
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from phevaluator.evaluator import evaluate_cards

SUITS = ['H', 'D', 'C', 'S']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
DECK = [s + r for r in RANKS for s in SUITS]
ALL_HANDS = list(combinations(DECK, 2))
_SUIT_MAP = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}

STREET_BOARD = {'flop': 3, 'turn': 4, 'river': 5}


def to_phev(card):
    return card[1] + _SUIT_MAP[card[0]]


def rank7(cards):
    return evaluate_cards(*[to_phev(c) for c in cards])


# ----------------------------------------------------------------------
# Equity distribution feature
# ----------------------------------------------------------------------

def hand_winrate(hole, board5, opp_hands):
    """Win-rate (win + 0.5 tie) of `hole` vs a uniform opp range on a 5-card board."""
    hr = rank7(list(hole) + list(board5))
    win = tie = 0
    for opp in opp_hands:
        orank = rank7(list(opp) + list(board5))
        if hr < orank:        # lower phevaluator score = stronger
            win += 1
        elif hr == orank:
            tie += 1
    n = len(opp_hands)
    return (win + 0.5 * tie) / n if n else 0.5


def equity_distribution(hole, board, bins, n_runout, n_opp, rng):
    """Histogram of per-runout win-rates -> the hand's equity distribution."""
    dead = set(hole) | set(board)
    deck = [c for c in DECK if c not in dead]
    need = 5 - len(board)

    if need == 0:
        runouts = [()]
    elif need == 1:
        runouts = [(c,) for c in deck]                  # enumerate the 44-46 rivers
    else:
        runouts = [tuple(rng.sample(deck, need)) for _ in range(n_runout)]

    winrates = []
    for ro in runouts:
        full = board + list(ro)
        blocked = dead | set(ro)
        opp = [h for h in ALL_HANDS if h[0] not in blocked and h[1] not in blocked]
        if n_opp and len(opp) > n_opp:
            opp = rng.sample(opp, n_opp)
        winrates.append(hand_winrate(hole, full, opp))

    hist, _ = np.histogram(winrates, bins=bins, range=(0.0, 1.0))
    h = hist.astype(float)
    s = h.sum()
    return h / s if s > 0 else h


# ----------------------------------------------------------------------
# EMD k-means (1-D histograms)
# ----------------------------------------------------------------------

def emd(a, b):
    """1-D Earth Mover's Distance = L1 between CDFs."""
    return np.abs(np.cumsum(a) - np.cumsum(b)).sum()


def kmeans_emd(feats, k, iters, rng):
    n = len(feats)
    idx = rng.sample(range(n), k)
    centroids = [feats[i].copy() for i in idx]
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        # assign
        for i, f in enumerate(feats):
            labels[i] = int(np.argmin([emd(f, c) for c in centroids]))
        # update (mean histogram is a standard, cheap EMD-centroid approximation)
        moved = 0.0
        for c in range(k):
            members = feats[labels == c] if isinstance(feats, np.ndarray) else \
                np.array([feats[i] for i in range(n) if labels[i] == c])
            if len(members):
                new = members.mean(axis=0)
                moved += emd(new, centroids[c])
                centroids[c] = new
        if moved < 1e-6:
            break
    # order clusters by mean equity so bucket index is monotonic-ish
    bin_centers = (np.arange(len(centroids[0])) + 0.5) / len(centroids[0])
    order = np.argsort([float((c * bin_centers).sum()) for c in centroids])
    centroids = [centroids[o] for o in order]
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[int(l)] for l in labels])
    return np.array(centroids), labels


def assign(feat, centroids):
    return int(np.argmin([emd(feat, c) for c in centroids]))


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def sample_situation(street, rng):
    nboard = STREET_BOARD[street]
    cards = rng.sample(DECK, 2 + nboard)
    return tuple(cards[:2]), cards[2:]


def main():
    ap = argparse.ArgumentParser(description="Distribution-aware postflop buckets.")
    ap.add_argument('--street', choices=list(STREET_BOARD), default='flop')
    ap.add_argument('--buckets', type=int, default=12)
    ap.add_argument('--situations', type=int, default=3000,
                    help="Sampled (hole,board) situations to fit clusters on.")
    ap.add_argument('--bins', type=int, default=30, help="Equity histogram bins.")
    ap.add_argument('--runout-samples', type=int, default=60,
                    help="Sampled runouts per flop situation (turn/river enumerate).")
    ap.add_argument('--opp-samples', type=int, default=200,
                    help="Sampled opponent hands per board (0 = full range).")
    ap.add_argument('--kmeans-iters', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    print(f"Street {args.street}: sampling {args.situations} situations, "
          f"{args.bins} equity bins, K={args.buckets}")
    feats = np.empty((args.situations, args.bins))
    for i in range(args.situations):
        hole, board = sample_situation(args.street, rng)
        feats[i] = equity_distribution(
            hole, board, args.bins, args.runout_samples, args.opp_samples, rng)
        if (i + 1) % 500 == 0:
            print(f"  features {i + 1}/{args.situations}", flush=True)

    print("clustering (EMD k-means)...")
    centroids, labels = kmeans_emd(feats, args.buckets, args.kmeans_iters, rng)

    counts = np.bincount(labels, minlength=args.buckets)
    bin_centers = (np.arange(args.bins) + 0.5) / args.bins
    print("\nbucket | share  | mean-eq | shape (E[eq], spread)")
    for b in range(args.buckets):
        c = centroids[b]
        mean_eq = float((c * bin_centers).sum())
        spread = float(np.sqrt(((bin_centers - mean_eq) ** 2 * c).sum()))
        bar = '#' * int(40 * counts[b] / max(1, counts.sum()))
        print(f"  {b:>4} | {counts[b] / counts.sum():5.1%} | {mean_eq:6.3f} "
              f"| spread={spread:5.3f} {bar}")

    out = os.path.join(os.path.dirname(__file__), '..', 'analysis',
                       f'postflop_centroids_{args.street}.npz')
    np.savez(out, centroids=centroids, bins=args.bins)
    print(f"\nsaved centroids -> {os.path.relpath(out)}")

    # --- Demo: the made-vs-draw question, on one fixed flop -----------------
    if args.street == 'flop':
        flop = ['SA', 'H7', 'C2']
        top_pair = ('DA', 'C5')          # pair of aces, static
        flush_draw = ('SK', 'SQ')        # nut flush draw + overs, polarized
        for name, hole in [('top pair Ax (static)', top_pair),
                           ('K-high nut FD+overs (polarized)', flush_draw)]:
            f = equity_distribution(hole, flop, args.bins, 200, args.opp_samples, rng)
            mean_eq = float((f * bin_centers).sum())
            spread = float(np.sqrt(((bin_centers - mean_eq) ** 2 * f).sum()))
            print(f"  demo {name:32} -> bucket {assign(f, centroids):>2} "
                  f"(eq={mean_eq:.3f}, spread={spread:.3f})")


if __name__ == '__main__':
    main()
