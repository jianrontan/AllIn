# backend/bot/src/subgame/turn_leaf_gen.py
"""Standalone turn-leaf M0 generator -- the shared upstream for the BAKE and NN-leaf pipelines
(docs/TURN_BAKE_VS_NN_SPEC.md).

Produces the range-INDEPENDENT leaf value matrix M0 for a (board4, leaf pot/stacks), reproducing the
EXACT river sample + strength partition the live `TurnSubgameSolver` uses (so a baked/learned M0 is
bit-identical to what the live solve would compute). M0 scales linearly with pot, so the natural
storage/learning key is (canonical board, leaf-SPR) with M0/pot as the target.
"""
import zlib

import numpy as np

from .cfv import (turn_strength, equal_freq_partition, turn_leaf_matrix_both, FULL_DECK)


def board_rivers_and_partition(board4, evaluator, cards, n_buckets, leaf_rivers):
    """The deterministic river sample + strength partition for a turn board -- IDENTICAL to
    TurnSubgameSolver.solve_turn_for_action (per-board crc32 seed, unbiased sample, equal-freq
    partition). Returns (rivers, partition, ba_cache). ba_cache is shared so repeated M0 calls on the
    same board reuse the per-river board arrays."""
    ba_cache = {}
    rivers = [c for c in FULL_DECK if c not in set(board4)]
    if leaf_rivers and 0 < leaf_rivers < len(rivers):
        rsamp = np.random.default_rng(zlib.crc32(''.join(sorted(board4)).encode()))
        rivers = sorted(rsamp.choice(rivers, size=leaf_rivers, replace=False).tolist())
    strength = turn_strength(board4, evaluator, cards, rivers=rivers, ba_cache=ba_cache)
    part = equal_freq_partition(strength, n_buckets)
    return rivers, part, ba_cache


def m0_for(board4, final_pot, leaf_stacks, db, evaluator, cards, *, menu,
           rivers, partition, ba_cache):
    """M0 for one leaf config, given the board's precomputed (rivers, partition, ba_cache).
    Returns (M0, buckets) -- M0 is [B x B] float64; buckets is the sorted bucket id list."""
    M0, _M1, buckets, _bidx, _tb = turn_leaf_matrix_both(
        board4, float(final_pot), tuple(float(s) for s in leaf_stacks), db, evaluator, cards,
        menu=menu, rivers=rivers, partition=partition, ba_cache=ba_cache)
    return M0, buckets


def m0_normalized(board4, leaf_spr, db, evaluator, cards, *, menu, n_buckets, leaf_rivers,
                  ref_pot=1.0):
    """Convenience: the POT-NORMALIZED M0 (= M0 / pot) for a (board4, leaf-SPR) -- the storage/learning
    target. Computes the board's sample+partition internally. leaf_spr = leaf_stacks/final_pot (equal
    seats). M0(k.pot, k.stacks) = k.M0(pot,stacks), so M0/ref_pot is scale-free."""
    rivers, part, ba_cache = board_rivers_and_partition(board4, evaluator, cards, n_buckets, leaf_rivers)
    beh = leaf_spr * ref_pot
    M0, buckets = m0_for(board4, ref_pot, (beh, beh), db, evaluator, cards,
                         menu=menu, rivers=rivers, partition=part, ba_cache=ba_cache)
    return M0 / ref_pot, buckets
