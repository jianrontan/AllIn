#!/usr/bin/env python3
"""NN-leaf trainer (TURN_BAKE_VS_NN_SPEC Pipeline B step 2-3): load the M0 dataset, featurize the board
(near-lossless canonical encoding), train a torch MLP (CPU), report TRAIN + HELD-OUT-BY-BOARD rel-RMSE.

Held-out by BOARD = honest generalization test. But since we generate ALL canonical boards, the NN is a
COMPRESSION of the full table -> low TRAIN rel-RMSE = the net has capacity; low HELD-OUT = it's a smooth
function (genuine compression), not memorization. Target ~<=0.15 (the EV gate grades on the exact leaf,
so the NN need only be accurate enough to PROPOSE good deviations -- safety is gate-protected).

  python scripts/nn_leaf_train.py --data analysis/nn_leaf/m0_data.npz --epochs 200
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_RANKS = '23456789TJQKA'
_RANK_VAL = {r: i for i, r in enumerate(_RANKS)}


def board_features(board_str):
    """Near-LOSSLESS suit-invariant canonical-board features (a 4-card board is identified by its sorted
    ranks + which cards share a suit). 4 cards x rank-one-hot (52) + 6 suit-sharing pairs + texture flags."""
    cards = [board_str[i:i + 2] for i in range(0, 8, 2)]
    order = sorted(range(4), key=lambda i: _RANK_VAL[cards[i][1]])
    cards = [cards[i] for i in order]
    ranks = [_RANK_VAL[c[1]] for c in cards]
    suits = [c[0] for c in cards]
    onehot = np.zeros(4 * 13, np.float32)
    for i, r in enumerate(ranks):
        onehot[i * 13 + r] = 1.0
    share = np.array([1.0 if suits[i] == suits[j] else 0.0
                      for i in range(4) for j in range(i + 1, 4)], np.float32)   # 6
    hist = np.zeros(13, np.float32)
    for r in ranks:
        hist[r] += 1
    suit_counts = sorted((suits.count(s) for s in set(suits)), reverse=True)
    suit_counts = np.array((suit_counts + [0, 0, 0, 0])[:4], np.float32) / 4.0
    gaps = np.diff(sorted(ranks))
    flags = np.array([float(max(hist) >= 2), float(max(hist) >= 3),
                      float(max(suit_counts * 4) >= 3), float(np.sum(gaps <= 2)) / 3.0,
                      ranks[-1] / 12.0, ranks[0] / 12.0], np.float32)
    return np.concatenate([onehot, share, hist / 4.0, suit_counts, flags])   # 52+6+13+4+6 = 81


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='analysis/nn_leaf/m0_data.npz')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='analysis/nn_leaf/nn_leaf.pt')
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    z = np.load(args.data, allow_pickle=True)
    bstr, spr, Y = z['boards'], z['spr'].astype(np.float32), z['m0'].astype(np.float32)
    print(f"dataset: {len(Y)} samples, {len(set(bstr.tolist()))} boards, M0 dim {Y.shape[1]}")

    # featurize (cache board features so we don't recompute per sample)
    uniq = {b: board_features(b) for b in set(bstr.tolist())}
    BF = np.stack([uniq[b] for b in bstr])
    X = np.concatenate([BF, (spr / 8.0).reshape(-1, 1)], axis=1).astype(np.float32)

    # split BY BOARD
    boards = np.array(sorted(set(bstr.tolist())))
    rng = np.random.default_rng(1)
    test_b = set(rng.choice(len(boards), size=max(1, len(boards) // 6), replace=False).tolist())
    is_test = np.array([i in test_b for i in
                        [np.searchsorted(boards, b) for b in bstr]])
    tr, te = ~is_test, is_test

    # standardize X by train stats; standardize Y by a GLOBAL scale (keep matrix structure)
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xs = (X - xm) / xs
    yscale = float(np.sqrt(np.mean(Y[tr] ** 2)))           # one global scale -> preserves zero-sum shape
    Ys = Y / yscale

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(8)
    Xtr = torch.tensor(Xs[tr], device=dev); Ytr = torch.tensor(Ys[tr], device=dev)
    Xte = torch.tensor(Xs[te], device=dev); Yte = torch.tensor(Ys[te], device=dev)
    print(f"train {Xtr.shape[0]} / test {Yte.shape[0]} (by board), in_dim {X.shape[1]}, device {dev}")

    layers, d = [], X.shape[1]
    for _ in range(args.layers):
        layers += [nn.Linear(d, args.hidden), nn.ReLU()]
        d = args.hidden
    layers += [nn.Linear(d, Y.shape[1])]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.MSELoss()

    def rel_rmse(P, T):
        return float(torch.sqrt(((P - T) ** 2).mean()) / (torch.sqrt((T ** 2).mean()) + 1e-12))

    n = Xtr.shape[0]
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(net(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                rtr = rel_rmse(net(Xtr), Ytr)
                rte = rel_rmse(net(Xte), Yte)
            print(f"  ep {ep:4} ({time.time()-t0:.0f}s)  train {rtr:.3f}  held-out-by-board {rte:.3f}",
                  flush=True)

    net.eval()
    with torch.no_grad():
        rtr, rte = rel_rmse(net(Xtr), Ytr), rel_rmse(net(Xte), Yte)
    print(f"\nFINAL  train rel-RMSE {rtr:.3f}  |  held-out-by-board rel-RMSE {rte:.3f}")
    print(f"  target <=0.15. >0.5 = featurization/capacity problem. param count: "
          f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M  (~{sum(p.numel() for p in net.parameters())*4/1e6:.1f}MB)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'state': net.state_dict(), 'xm': xm, 'xs': xs, 'yscale': yscale,
                'hidden': args.hidden, 'layers': args.layers}, args.out)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
