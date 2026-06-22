#!/usr/bin/env python3
"""Phase 1 enumeration for TURN_LATENCY_PLAN: size the bake/NN key space.

KEY INSIGHT (from turn_tree.py): the turn BETTING tree is BOARD-INDEPENDENT -- the leaf
(final_pot, leaf_stacks) configs depend only on (pot_entry, stacks, menu) = effectively SPR. The board
only sets M0's VALUES, not the leaf config set. So the bake table = canonical_boards x leaf_config(SPR).

This measures, across the live SPR range (<= the gate):
  1. distinct leaf (final_pot, leaf_stacks) configs PER TREE (the "~75/board" assumption -- really /SPR),
  2. whether POT-NORMALIZING (final_pot/pot, leaf_stacks/pot) collapses configs across different absolute
     pots at the same SPR -> if yes, the key is (board, SPR-bucket), a clean 1-D discretization (small,
     grid-clean). If no, it's a 2-D (pot, stacks) continuum (large, grid-miss risk).
  3. total distinct NORMALIZED leaf configs over an SPR grid -> the per-board multiplier for table size.

  python scripts/enumerate_turn_leaves.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.subgame.turn_tree import build_turn_tree, MAX_AGGRESSIONS

try:
    from src.subgame.turn_tree import DEFAULT_MENU
except Exception:
    DEFAULT_MENU = None


def leaf_configs(pot, stacks, menu, max_agg, normalize=False):
    """Distinct leaf (final_pot, leaf_stacks) of the turn tree, optionally pot-normalized + rounded."""
    t = build_turn_tree(pot, stacks, menu=menu, max_aggressions=max_agg)
    out = set()
    for n in getattr(t, 'leaves', None) or _iter_leaves(t):
        fp, ls = n.final_pot, n.leaf_stacks
        if normalize:
            out.add((round(fp / pot, 3), round(ls[0] / pot, 3), round(ls[1] / pot, 3)))
        else:
            out.add((round(fp, 2), round(ls[0], 2), round(ls[1], 2)))
    return out


def _iter_leaves(tree):
    # leaves = nodes with no children (final_pot set). Walk decision_nodes' children.
    seen, stack = [], [tree.root]
    while stack:
        n = stack.pop()
        ch = getattr(n, 'children', None) or {}
        kids = list(ch.values()) if isinstance(ch, dict) else list(ch)
        if not kids:
            seen.append(n)
        else:
            stack.extend(k for k in kids if k is not None)
    return seen


def main():
    menu = DEFAULT_MENU
    max_agg = MAX_AGGRESSIONS
    print(f"menu={menu}  max_aggressions={max_agg}\n")

    # 1. leaf count per SPR (fixed pot=100, vary behind stacks)
    print("SPR  leaves(raw)  leaves(pot-normalized)")
    spr_norm_union = set()
    for spr in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]:
        pot = 100.0
        beh = spr * pot
        raw = leaf_configs(pot, (beh, beh), menu, max_agg, normalize=False)
        nrm = leaf_configs(pot, (beh, beh), menu, max_agg, normalize=True)
        spr_norm_union |= {(spr, *c) for c in nrm}
        print(f"{spr:4}  {len(raw):10}  {len(nrm):10}")

    # 2. does normalizing collapse across DIFFERENT absolute pots at the SAME SPR?
    print("\nNormalization check (same SPR=3, different absolute pots) -> identical normalized set?")
    s = []
    for pot in [40.0, 100.0, 173.0, 250.0]:
        s.append(leaf_configs(pot, (3 * pot, 3 * pot), menu, max_agg, normalize=True))
    allsame = all(x == s[0] for x in s)
    print(f"  pots {[40,100,173,250]} -> normalized leaf sets identical: {allsame} (|set|={len(s[0])})")

    # 3. total distinct normalized leaf configs over a fine SPR grid (the per-board table multiplier)
    grid = [round(0.5 + 0.25 * i, 2) for i in range(31)]   # SPR 0.5..8.0 step 0.25
    union = set()
    for spr in grid:
        pot = 100.0
        beh = spr * pot
        union |= leaf_configs(pot, (beh, beh), menu, max_agg, normalize=True)
    print(f"\nDistinct NORMALIZED leaf configs over SPR grid {grid[0]}..{grid[-1]} step 0.25: {len(union)}")
    print("  => bake table rows ~= (canonical turn boards) x (this number).")
    print("  If normalization collapses (check 2 = True), key = (board, SPR-bucket): SMALL + grid-clean.")


if __name__ == '__main__':
    main()
