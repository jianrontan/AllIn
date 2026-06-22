#!/usr/bin/env python3
"""NN-leaf data-gen (TURN_BAKE_VS_NN_SPEC Pipeline B step 1) -- PARALLEL + RESUMABLE.

Enumerates ALL canonical turn boards and computes M0/pot for each x an SPR grid (the full bake table,
which the NN then compresses). Saves RAW (board, leaf-SPR, M0) so features can be iterated at TRAIN time
without re-gen. Checkpoints so an overnight run survives interruption.

  python scripts/nn_leaf_datagen.py --workers 15 --out analysis/nn_leaf/m0_data.npz
"""
import argparse
import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = 'analysis/blueprints/snapshots/snap_52500000.db'
SPR_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]   # leaf SPR = leaf_stacks/final_pot
# Rivers per M0. The LIVE solve samples 4 (jumpy per board -> hard to learn + low-river BIAS). With the
# NN leaf, more rivers cost NOTHING at serve time (the net predicts the high-river M0), so we spend the
# overnight headroom here: a SMOOTHER, less-biased M0 = an easier learning target + better quality.
# CONSISTENCY: whatever this is, the SERVED turn solver's leaf_rivers (partition + exact gate) must MATCH.
_LEAF_RIVERS = int(os.environ.get('NN_LEAF_RIVERS', '12'))

_W = {}   # per-worker singletons


def _init_worker():
    # mmap the postflop tables so ALL workers SHARE one copy via the OS page cache (instead of each
    # loading ~127MB into its own heap) -- essential on a low-RAM (<16GB) box. Set BEFORE the import.
    os.environ['ALLIN_MMAP_POSTFLOP'] = '1'
    from src.storage.blueprint_db import BlueprintDB
    from src.subgame.turn_subgame_solver import TurnSubgameSolver
    db = BlueprintDB(DB_PATH, read_only=True)
    bot = TurnSubgameSolver(db, n_buckets=24, leaf_rivers=4, safe_gadget=True, gadget_anchor='auto')
    _W['db'] = db
    _W['bot'] = bot
    _W['pm'] = bot._postflop_menu


def _gen_board(board):
    """Return (board_str, [(spr, m0_flat_float32), ...]) for one canonical board, or None on failure."""
    from src.subgame.turn_leaf_gen import board_rivers_and_partition, m0_for
    bot, db, pm = _W['bot'], _W['db'], _W['pm']
    try:
        rivers, part, bac = board_rivers_and_partition(board, bot._evaluator, bot._cards, 24, _LEAF_RIVERS)
    except Exception:
        return None
    rows = []
    for spr in SPR_GRID:
        try:
            beh = float(spr)
            m0, _ = m0_for(board, 1.0, (beh, beh), db, bot._evaluator, bot._cards,
                           menu=pm, rivers=rivers, partition=part, ba_cache=bac)
            if m0.shape == (24, 24):
                rows.append((spr, m0.ravel().astype(np.float32)))
        except Exception:
            continue
    return (''.join(board), rows) if rows else None


def canonical_boards():
    from src.game.cards import shuffled_deck
    from src.abstractions.canonical import canonical_board_perm
    deck = shuffled_deck()
    seen, reps = set(), []
    for i, b in enumerate(combinations(deck, 4)):     # 270,725 concrete -> 16,432 canonical
        cb, _ = canonical_board_perm(list(b))
        key = tuple(cb)
        if key not in seen:
            seen.add(key)
            reps.append(list(cb))
        if (i + 1) % 60000 == 0:
            print(f"  enumerating... {i+1}/270725 scanned, {len(reps)} canonical so far", flush=True)
    return reps


def main():
    ap = argparse.ArgumentParser()
    # 8 PHYSICAL cores, but <16GB RAM (Chrome) -> default 6 workers (leave cores+RAM for the system).
    # Workers SHARE the mmap'd postflop table, so RAM scales gently; bump to 8 if RAM allows.
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default='analysis/nn_leaf/m0_data.npz')
    ap.add_argument('--limit', type=int, default=0, help="0 = all canonical boards")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("enumerating canonical turn boards ...", flush=True)
    boards = canonical_boards()
    if args.limit:
        boards = boards[:args.limit]
    print(f"{len(boards)} canonical boards x {len(SPR_GRID)} SPRs", flush=True)

    done = set()
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        done = set(z['boards'].tolist())
        bstr, sprs, Y = list(z['boards']), list(z['spr']), list(z['m0'])
        print(f"resuming: {len(done)} boards already done", flush=True)
    else:
        bstr, sprs, Y = [], [], []
    todo = [b for b in boards if ''.join(b) not in done]
    print(f"{len(todo)} boards to go", flush=True)

    t0 = time.time()
    n_done = 0
    print(f"starting {args.workers} workers @ {_LEAF_RIVERS} rivers (progress every 100 boards, "
          f"checkpoint every 500) ...", flush=True)
    with Pool(args.workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(_gen_board, todo, chunksize=4):
            n_done += 1
            if res is not None:
                bs, rows = res
                for spr, m0 in rows:
                    bstr.append(bs)
                    sprs.append(spr)
                    Y.append(m0)
            if n_done % 100 == 0:                              # frequent progress (~every few min)
                el = time.time() - t0
                rate = n_done / el
                eta = (len(todo) - n_done) / rate / 60
                print(f"  {n_done}/{len(todo)} boards  {rate:.1f}/s  elapsed {el/60:.0f}min  "
                      f"ETA {eta:.0f}min  ({len(Y)} samples)", flush=True)
            if n_done % 500 == 0:                              # less-frequent checkpoint (heavier I/O)
                np.savez_compressed(args.out, boards=np.array(bstr), spr=np.array(sprs, np.float32),
                                    m0=np.array(Y, np.float32))
    np.savez_compressed(args.out, boards=np.array(bstr), spr=np.array(sprs, np.float32),
                        m0=np.array(Y, np.float32))
    print(f"DONE: {len(Y)} samples over {len(set(bstr))} boards -> {args.out} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
