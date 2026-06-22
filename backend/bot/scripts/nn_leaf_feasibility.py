#!/usr/bin/env python3
"""NN-leaf FEASIBILITY test (TURN_BAKE_VS_NN_SPEC Pipeline B step 1-3): can a small net learn
board->M0/pot well enough? Data-gen (board-amortized) -> train an MLP -> held-out-BY-BOARD rel-RMSE.

Held-out by BOARD (not by sample) is the real test: the net must generalize to boards it never saw.
Baseline = predict the training-mean M0 (rel-RMSE ~1.0 = learned nothing).

  python scripts/nn_leaf_feasibility.py --boards 600 --sprs 8
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.storage.blueprint_db import BlueprintDB
from src.subgame.turn_subgame_solver import TurnSubgameSolver
from src.subgame.turn_leaf_gen import board_rivers_and_partition, m0_for
from src.game.cards import shuffled_deck

_RANKS = '23456789TJQKA'
_RANK_VAL = {r: i for i, r in enumerate(_RANKS)}


def board_features(board4):
    """Suit-INVARIANT, near-lossless canonical-board features: sorted rank values, rank-count histogram,
    sorted suit-count pattern, and texture flags (paired/flush-draw/connected/high)."""
    ranks = sorted((_RANK_VAL[c[1]] for c in board4))
    suits = [c[0] for c in board4]
    rank_hist = np.zeros(13)
    for r in ranks:
        rank_hist[r] += 1
    suit_counts = sorted((suits.count(s) for s in set(suits)), reverse=True)
    suit_counts = (suit_counts + [0, 0, 0, 0])[:4]
    paired = float(max(rank_hist) >= 2)
    trips = float(max(rank_hist) >= 3)
    flushy = float(max(suit_counts) >= 3)
    gaps = np.diff(ranks)
    connected = float(np.sum(gaps <= 2))
    feat = ([r / 12.0 for r in ranks] + list(rank_hist / 4.0) + [c / 4.0 for c in suit_counts]
            + [paired, trips, flushy, connected / 3.0, ranks[-1] / 12.0, ranks[0] / 12.0])
    return np.array(feat, float)


def gen(db, bot, n_boards, sprs, seed=0):
    pm = bot._postflop_menu
    rng = np.random.default_rng(seed)
    spr_vals = np.linspace(0.75, 7.5, sprs)
    X, Y, board_id = [], [], []
    t0 = time.time()
    deck = shuffled_deck()
    bid = 0
    made = 0
    while made < n_boards:
        b = list(rng.choice(deck, size=4, replace=False))
        try:
            rivers, part, bac = board_rivers_and_partition(b, bot._evaluator, bot._cards, 24, 4)
            bf = board_features(b)
            ok = False
            for spr in spr_vals:
                m0, _ = m0_for(b, 1.0, (float(spr), float(spr)), db, bot._evaluator, bot._cards,
                               menu=pm, rivers=rivers, partition=part, ba_cache=bac)
                if m0.shape != (24, 24):
                    continue
                X.append(np.concatenate([bf, [spr / 7.5]]))
                Y.append(m0.ravel())
                board_id.append(bid)
                ok = True
            if ok:
                bid += 1
                made += 1
        except Exception:
            continue
        if made % 50 == 0 and made:
            print(f"  gen {made}/{n_boards} boards ({time.time()-t0:.0f}s)", flush=True)
    return np.array(X), np.array(Y), np.array(board_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', type=int, default=600)
    ap.add_argument('--sprs', type=int, default=8)
    args = ap.parse_args()

    db = BlueprintDB('analysis/blueprints/snapshots/snap_52500000.db', read_only=True)
    bot = TurnSubgameSolver(db, n_buckets=24, leaf_rivers=4, safe_gadget=True, gadget_anchor='auto')
    print(f"generating ~{args.boards} boards x {args.sprs} SPRs ...", flush=True)
    X, Y, bid = gen(db, bot, args.boards, args.sprs)
    db.close()
    print(f"dataset: X{X.shape} Y{Y.shape}  ({bid.max()+1} distinct boards)")

    # split BY BOARD
    nb = bid.max() + 1
    rng = np.random.default_rng(1)
    test_boards = set(rng.choice(nb, size=max(1, nb // 5), replace=False).tolist())
    te = np.array([b in test_boards for b in bid])
    tr = ~te
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    print(f"train {Xtr.shape[0]} / test {Xte.shape[0]} (by board)")

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    net = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=300, early_stopping=True,
                       random_state=0)
    t0 = time.time()
    net.fit(sc.transform(Xtr), Ytr)
    pred = net.predict(sc.transform(Xte))

    def rel_rmse(P, T):
        return float(np.sqrt(np.mean((P - T) ** 2)) / (np.sqrt(np.mean(T ** 2)) + 1e-12))
    base = np.tile(Ytr.mean(0), (Yte.shape[0], 1))
    print(f"\ntrain {time.time()-t0:.0f}s")
    print(f"  NN   held-out-by-board rel-RMSE: {rel_rmse(pred, Yte):.3f}")
    print(f"  mean-baseline       rel-RMSE: {rel_rmse(base, Yte):.3f}  (1.0 = learned nothing)")
    print(f"  => NN beats baseline by {rel_rmse(base,Yte)-rel_rmse(pred,Yte):+.3f}; "
          f"lower NN = better. <~0.15 promising, >~0.5 = featurization/data problem.")


if __name__ == '__main__':
    main()
