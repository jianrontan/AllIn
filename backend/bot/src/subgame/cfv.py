# backend/bot/src/subgame/cfv.py
"""
Leaf value function for depth-limited TURN solving (Phase-4 keystone, M0).

The turn depth limit is "turn betting closed -> river to come." The leaf value is
the blueprint's continuation value through the river: for each hero hand, the
expected value (MEASURE units, opponent-reach-weighted, matching the river solver's
terminal convention) of dealing the river and both players playing the river per the
BLUEPRINT, against the opponent's range entering the leaf.

`turn_leaf_value_exact` is the EXACT, per-hand, reach-conditioned reference (the
"rollout"): it averages the blueprint's river value over every river runout. It
reuses the validated river machinery as-is (the river is street 3, a 5-card board, so
NONE of the 4-card-turn-basis landmines apply here -- those bite the turn TREE, not
this leaf):
  build_board_arrays (5-card river) -> build_river_tree(leaf pot/stacks) ->
  blueprint_strategy_on_tree (blueprint projected onto the river tree) ->
  RiverCFR._eval (read-only value propagation under that fixed strategy).

This is the M0 reference against which the cheaper BUCKETED leaf (next step) is
validated -- including under a SHIFTED opponent range, the real test of whether a
bucketed/frozen leaf is adequate (the agent-review P0). Ranges are passed as
dicts {(cardA, cardB): weight} over TURN hands, where the tuple order matches
showdown_kernel's hand enumeration (FULL_DECK order).
"""
from collections import defaultdict
from itertools import combinations

import numpy as np

from ..evaluation.showdown_kernel import build_board_arrays, compatible_mass
from .river_tree import build_river_tree
from .river_cfr import RiverCFR
from .blueprint_projection import blueprint_strategy_on_tree

# MUST match showdown_kernel._FULL_DECK ordering (hand tuples are combinations over it).
_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_SUITS = ['H', 'D', 'C', 'S']
FULL_DECK = [s + r for r in _RANKS for s in _SUITS]


def turn_hands(board4):
    """All turn hands (cardA, cardB), FULL_DECK order, not using a board card."""
    pool = [c for c in FULL_DECK if c not in set(board4)]
    return list(combinations(pool, 2))


def _get_ba(board5, evaluator, cards, ba_cache):
    """build_board_arrays for a 5-card river board, memoized in `ba_cache` (keyed by
    the board) when provided. `ba` depends only on the board (not pot/stacks/range), so
    a caller sweeping many (pot, stacks, range) configs on the SAME board can pass one
    dict and pay the expensive per-board rank pass once. None -> no caching."""
    if ba_cache is None:
        return build_board_arrays(board5, evaluator, cards)
    key = tuple(board5)
    ba = ba_cache.get(key)
    if ba is None:
        ba = build_board_arrays(board5, evaluator, cards)
        ba_cache[key] = ba
    return ba


def turn_leaf_value_exact(board4, pot, stacks, hero_seat, hero_range, villain_range,
                          db, evaluator, cards, menu=None, rivers=None, ba_cache=None):
    """EXACT reach-conditioned turn-leaf value (the rollout reference).

    board4: 4 SuitRank turn cards. pot/stacks: pot + (equal) behind stacks AT the leaf
    (turn betting closed). hero_seat: 0/1. hero_range/villain_range: {hand_tuple: weight}
    over turn hands. db: BlueprintDB (read-only). menu: postflop bet menu
    (postflop_menu_for(db_menu_mode(db))); None -> river_tree default.

    Returns {hero_hand_tuple: leaf_value} -- the blueprint's expected river
    continuation value (MEASURE units) for that hero hand vs `villain_range`, averaged
    over river runouts. Hands that block no runout see all 46 rivers; the mean is over
    the runouts where the hand exists (uniform chance, card removal handled by the
    per-river 5-card basis dropping blocked hands)."""
    board_set = set(board4)
    acc = defaultdict(float)
    cnt = defaultdict(int)
    # `rivers` restricts the runout set (a fast smoke); None = all 46+ live rivers.
    # Pass the SAME `rivers` to both seats to keep the zero-sum identity exact.
    river_iter = [r for r in (rivers if rivers is not None else FULL_DECK)
                  if r not in board_set]

    for r in river_iter:
        board5 = list(board4) + [r]
        ba = _get_ba(board5, evaluator, cards, ba_cache)
        hands5 = ba['hands']
        reach_hero = np.array([hero_range.get(hb, 0.0) for hb in hands5], dtype=float)
        reach_vill = np.array([villain_range.get(hb, 0.0) for hb in hands5], dtype=float)
        if reach_vill.sum() <= 0.0:
            continue
        reach0, reach1 = ((reach_hero, reach_vill) if hero_seat == 0
                          else (reach_vill, reach_hero))

        # The river TREE uses the solver's own bet menu (river_tree.DEFAULT_MENU),
        # like the live river solver -- NOT the blueprint's postflop menu. `menu` here
        # is the blueprint's postflop menu, used only to PROJECT the blueprint's chars
        # onto the tree (capped vs control). Two different "menu" concepts.
        tree = build_river_tree(pot, stacks)
        bp = blueprint_strategy_on_tree(tree, ba, db.get_average_strategy,
                                        postflop_menu=menu)
        cfr = RiverCFR(tree, ba)
        v0, v1 = cfr._eval(tree.root, reach0, reach1, lambda nid: bp[nid])
        vhero = v0 if hero_seat == 0 else v1

        for i, hb in enumerate(hands5):
            acc[hb] += float(vhero[i])
            cnt[hb] += 1

    return {hb: acc[hb] / cnt[hb] for hb in acc if cnt[hb] > 0}


def turn_leaf_br_value(board4, pot, stacks, br_seat, fixed_range,
                       db, evaluator, cards, menu=None, rivers=None, ba_cache=None):
    """Leaf value where the BR player (`br_seat`) BEST-RESPONDS on the river while the
    fixed (opponent) player plays the BLUEPRINT river -- the OUT-OF-MODEL-RIVER leaf.

    turn_leaf_value_exact has BOTH seats play the blueprint river (in-model); the gap
    between THIS and that is the 'frozen-range trap' (Brown-Sandholm): a real opponent
    deviates on the river, beyond the single frozen blueprint continuation, which K=1
    depth-limited solving does not see and M3's multi-valued (K-set) leaf must close.

    Per river: project the blueprint (the FIXED player's river strategy), then
    RiverCFR._br_value lets `br_seat` best-respond against it. Returns {br_hand: value}
    (MEASURE units, vs `fixed_range`), runout-averaged. `fixed_range` is the opponent's
    range entering the leaf (the reach the BR is computed against)."""
    board_set = set(board4)
    acc = defaultdict(float)
    cnt = defaultdict(int)
    river_iter = [r for r in (rivers if rivers is not None else FULL_DECK)
                  if r not in board_set]
    for r in river_iter:
        board5 = list(board4) + [r]
        ba = _get_ba(board5, evaluator, cards, ba_cache)
        hands5 = ba['hands']
        fixed_reach = np.array([fixed_range.get(hb, 0.0) for hb in hands5], dtype=float)
        if fixed_reach.sum() <= 0.0:
            continue
        tree = build_river_tree(pot, stacks)
        bp = blueprint_strategy_on_tree(tree, ba, db.get_average_strategy, postflop_menu=menu)
        cfr = RiverCFR(tree, ba)
        vbr = cfr._br_value(tree.root, br_seat, fixed_reach, lambda nid: bp[nid])
        for i, hb in enumerate(hands5):
            acc[hb] += float(vbr[i])
            cnt[hb] += 1
    return {hb: acc[hb] / cnt[hb] for hb in acc if cnt[hb] > 0}


def turn_bucket(hand, board4, cards):
    """The hero's TURN strength bucket (street 2) for `hand` on the 4-card board --
    computed directly via the bucketer (NO build_board_arrays, which assumes a 5-card
    board). This is the resolution at which the turn solver indexes its leaf."""
    return cards.get_bucket(list(hand), list(board4))


def turn_strength(board4, evaluator, cards, rivers=None, ba_cache=None, with_var=False):
    """Observable, range-INDEPENDENT strength scalar per turn hand: the hand's mean
    showdown rank (`raw`, lower = stronger) averaged over river runouts. This is a
    feature of the cards alone (no opponent range, no blueprint), so it is a legitimate
    basis for a FINER leaf partition than the blueprint's turn buckets -- unlike the leaf
    value itself, which would be circular. Returns {turn_hand: mean_rank}.

    with_var=True instead returns (mean_dict, var_dict) where var_dict is the VARIANCE
    of the hand's rank across runouts -- a range-independent 'draw-iness' proxy (made
    hands have stable rank across rivers; draws swing) used to build leaf-stress range
    shifts that are ORTHOGONAL to mean strength."""
    board_set = set(board4)
    river_iter = [r for r in (rivers if rivers is not None else FULL_DECK)
                  if r not in board_set]
    acc = defaultdict(float)
    sq = defaultdict(float)
    cnt = defaultdict(int)
    for r in river_iter:
        ba = _get_ba(list(board4) + [r], evaluator, cards, ba_cache)
        for hb, rk in zip(ba['hands'], ba['raw']):
            rk = float(rk)
            acc[hb] += rk
            sq[hb] += rk * rk
            cnt[hb] += 1
    mean = {hb: acc[hb] / cnt[hb] for hb in acc if cnt[hb] > 0}
    if not with_var:
        return mean
    var = {hb: max(0.0, sq[hb] / cnt[hb] - mean[hb] ** 2) for hb in mean}
    return mean, var


def equal_freq_partition(strength, n):
    """Collapse a {hand: scalar} strength map into n equal-frequency bins (bin 0 =
    strongest, since lower rank = stronger). Returns {hand: bin_id in 0..n-1}.

    Ties are broken by the hand key so the partition is fully deterministic regardless
    of dict insertion order (hands with identical strength would otherwise land in
    adjacent bins by sort order alone)."""
    items = sorted(strength.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    return {h: min(n - 1, i * n // m) for i, (h, _) in enumerate(items)}


def turn_leaf_matrix(board4, pot, stacks, hero_seat, db, evaluator, cards,
                     menu=None, rivers=None, partition=None):
    """Reach-conditioned bucketed leaf: M[hero_turn_bucket, villain_turn_bucket] =
    the AVERAGE blueprint river-continuation payoff of a hero hand in hero_bucket vs a
    villain hand in villain_bucket, runout-averaged. Range-INDEPENDENT (the live range
    is applied at runtime via `bucketed_measure_leaf`), which is exactly why a per-bucket
    SCALAR CFV is wrong (it bakes in one range) and this matrix is right.

    Built efficiently: the blueprint projection onto the river tree is reach-independent,
    so it is computed ONCE per river card, then one cheap `_eval` per villain bucket
    (reach = that bucket's indicator). Per-(hero hand, villain bucket) payoff =
    measure / compatible-mass; collapsed to hero buckets at the end.

    `partition` (optional) overrides the hero/villain partition with a custom
    {turn_hand: bucket_id} map (e.g. a FINER strength partition from
    equal_freq_partition(turn_strength(...), n)) -- the leaf resolution is a free lever
    independent of the blueprint's turn buckets. None -> the blueprint turn buckets.

    Returns (M, buckets, bidx, tb): M is [B, B]; buckets the sorted turn-bucket ids;
    bidx {bucket: row/col index}; tb {turn_hand: turn_bucket}."""
    tb = partition if partition is not None else {
        h: turn_bucket(h, board4, cards) for h in turn_hands(board4)}
    buckets = sorted(set(tb.values()))
    bidx = {b: i for i, b in enumerate(buckets)}
    B = len(buckets)
    board_set = set(board4)
    river_iter = [r for r in (rivers if rivers is not None else FULL_DECK)
                  if r not in board_set]

    # per hero hand: sum of (avg payoff vs vb) over rivers, and the river count.
    psum = {h: np.zeros(B) for h in tb}
    pcnt = {h: np.zeros(B) for h in tb}

    for r in river_iter:
        board5 = list(board4) + [r]
        ba = build_board_arrays(board5, evaluator, cards)
        hands5 = ba['hands']
        tb5 = [tb[hb] for hb in hands5]
        full = np.ones(len(hands5))
        tree = build_river_tree(pot, stacks)
        bp = blueprint_strategy_on_tree(tree, ba, db.get_average_strategy, postflop_menu=menu)
        cfr = RiverCFR(tree, ba)
        for vb in buckets:
            vmask = np.fromiter((1.0 if b == vb else 0.0 for b in tb5), float, len(tb5))
            if vmask.sum() <= 0:
                continue
            reach0, reach1 = (full, vmask) if hero_seat == 0 else (vmask, full)
            v0, v1 = cfr._eval(tree.root, reach0, reach1, lambda nid: bp[nid])
            vhero = v0 if hero_seat == 0 else v1
            compat = compatible_mass(ba, vmask)         # # vb hands compatible per hero hand
            j = bidx[vb]
            for i, hb in enumerate(hands5):
                if compat[i] > 1e-9:
                    psum[hb][j] += vhero[i] / compat[i]
                    pcnt[hb][j] += 1.0

    # runout-average per hand, then collapse hero hands -> hero buckets
    M = np.zeros((B, B))
    Mc = np.zeros((B, B))
    for h, s in psum.items():
        c = pcnt[h]
        avg = np.divide(s, c, out=np.zeros(B), where=c > 0)
        i = bidx[tb[h]]
        M[i] += avg
        Mc[i] += (c > 0)
    M = np.divide(M, Mc, out=np.zeros((B, B)), where=Mc > 0)
    return M, buckets, bidx, tb


def _collapse_matrix(psum, pcnt, tb, bidx, B):
    """Runout-average per hero hand, then collapse hero hands -> hero buckets."""
    M = np.zeros((B, B))
    Mc = np.zeros((B, B))
    for h, s in psum.items():
        c = pcnt[h]
        avg = np.divide(s, c, out=np.zeros(B), where=c > 0)
        i = bidx[tb[h]]
        M[i] += avg
        Mc[i] += (c > 0)
    return np.divide(M, Mc, out=np.zeros((B, B)), where=Mc > 0)


def turn_leaf_matrix_both(board4, pot, stacks, db, evaluator, cards,
                          menu=None, rivers=None, partition=None, ba_cache=None):
    """The turn solver's leaf matrices (M0, M1), where M1 := -M0^T (ENFORCED, not
    independently averaged). The river is positional, so seat-1's leaf is genuinely a
    different object than seat-0's -- but because the river is ZERO-SUM, the seat-1
    matrix IS the negated transpose of seat-0's: -M0^T[hb,vb] = -M0[vb,hb] = (seat-1
    hero in hb vs seat-0 villain in vb), exactly M1's definition. Computing M1
    INDEPENDENTLY (a second per-bucket `_eval`) instead gives a numerically DIFFERENT
    matrix because the hero-bucket collapse normalizes asymmetrically (unequal bucket
    sizes / compatible counts) -- which silently breaks the leaf's zero-sum property
    (measured ~3.5% root zero-sum violation) and turns the turn subgame non-zero-sum,
    where CFR+'s exploitability no longer converges to ~0. Enforcing M1 = -M0^T makes
    the subgame EXACTLY zero-sum (root E0+E1 == 0 identically) and halves the build
    cost (one `_eval` per bucket, not two). Standard subgame-solver practice.

    Returns (M0, M1, buckets, bidx, tb)."""
    tb = partition if partition is not None else {
        h: turn_bucket(h, board4, cards) for h in turn_hands(board4)}
    buckets = sorted(set(tb.values()))
    bidx = {b: i for i, b in enumerate(buckets)}
    B = len(buckets)
    board_set = set(board4)
    river_iter = [r for r in (rivers if rivers is not None else FULL_DECK)
                  if r not in board_set]

    psum0 = {h: np.zeros(B) for h in tb}
    pcnt0 = {h: np.zeros(B) for h in tb}

    for r in river_iter:
        board5 = list(board4) + [r]
        ba = _get_ba(board5, evaluator, cards, ba_cache)
        hands5 = ba['hands']
        tb5 = [tb[hb] for hb in hands5]
        full = np.ones(len(hands5))
        tree = build_river_tree(pot, stacks)
        bp = blueprint_strategy_on_tree(tree, ba, db.get_average_strategy, postflop_menu=menu)
        cfr = RiverCFR(tree, ba)
        for vb in buckets:
            vmask = np.fromiter((1.0 if b == vb else 0.0 for b in tb5), float, len(tb5))
            if vmask.sum() <= 0:
                continue
            compat = compatible_mass(ba, vmask)
            j = bidx[vb]
            v0, _ = cfr._eval(tree.root, full, vmask, lambda nid: bp[nid])   # hero seat0
            for i, hb in enumerate(hands5):
                if compat[i] > 1e-9:
                    psum0[hb][j] += v0[i] / compat[i]
                    pcnt0[hb][j] += 1.0

    M0 = _collapse_matrix(psum0, pcnt0, tb, bidx, B)
    M1 = -M0.T                                  # zero-sum by construction (see docstring)
    # M1 = -M0^T is the correct seat-1 leaf ONLY because both seats share this one
    # partition `tb` (so row/col indices mean the same bucket for both). The solver
    # (TurnCFR) is handed a single tb_idx for both seats, preserving this.
    return M0, M1, buckets, bidx, tb


def _bias_strategy(bp, tree, villain_seat, bias, menu):
    """Modicum bias continuation: a COPY of `bp` (node_id -> [H,A]) with the VILLAIN's
    decision nodes reweighted toward `bias` -- x10 on the matching action columns then
    renormalized per row. Hero nodes unchanged. bias in {'baseline','fold','call','raise'}
    ('baseline' returns bp unchanged). 'raise' targets every bet/raise/all-in char."""
    from .blueprint_projection import tree_action_char
    if bias == 'baseline':
        return bp
    call_chars = ('c', 'k')
    out = list(bp)
    for node in tree.decision_nodes:
        if node.player != villain_seat:
            continue
        mat = bp[node.node_id]
        if mat is None:
            continue
        w = np.ones(len(node.actions))
        for a_i, action in enumerate(node.actions):
            ch = tree_action_char(action, node, menu)
            hit = ({'fold': ch == 'f', 'call': ch in call_chars,
                    'raise': ch not in ('f',) + call_chars}[bias])
            if hit:
                w[a_i] = 10.0
        biased = mat * w[None, :]
        rs = biased.sum(axis=1, keepdims=True)
        out[node.node_id] = np.divide(biased, rs, out=np.zeros_like(biased), where=rs > 1e-12)
    return out


def turn_leaf_matrix_multivalued(board4, pot, stacks, db, evaluator, cards,
                                 menu=None, rivers=None, partition=None, ba_cache=None,
                                 biases=('baseline', 'fold', 'call', 'raise')):
    """MULTI-VALUED (Modicum) turn leaf: the OPPONENT picks the worst-for-hero among
    `biases` blueprint river continuations, instead of trusting a single blueprint
    continuation (the source of the −76 over-prediction). Independent M0/M1 (NOT −M0^T):
    M0 (hero=seat0) biases the VILLAIN=seat1 strategy each way and takes the element-wise
    MIN over biases; M1 (hero=seat1) biases seat0 and mins. Multi-valued states are
    intentionally NOT zero-sum at the leaf (the opponent's continuation choice is a max
    for them, not mirrored), so CFR+'s root zero-sum/exploitability no longer hits ~0 --
    fine for an OFFLINE EV gate (we read realized play, not the convergence gap).
    Drop-in for turn_leaf_matrix_both: returns (M0, M1, buckets, bidx, tb)."""
    tb = partition if partition is not None else {
        h: turn_bucket(h, board4, cards) for h in turn_hands(board4)}
    buckets = sorted(set(tb.values()))
    bidx = {b: i for i, b in enumerate(buckets)}
    B = len(buckets)
    board_set = set(board4)
    river_iter = [r for r in (rivers if rivers is not None else FULL_DECK)
                  if r not in board_set]
    nb = list(biases)
    # per-bias accumulators for both seats' leaf matrices
    ps0 = [{h: np.zeros(B) for h in tb} for _ in nb]   # M0: hero seat0, villain seat1
    pc0 = [{h: np.zeros(B) for h in tb} for _ in nb]
    ps1 = [{h: np.zeros(B) for h in tb} for _ in nb]   # M1: hero seat1, villain seat0
    pc1 = [{h: np.zeros(B) for h in tb} for _ in nb]

    for r in river_iter:
        board5 = list(board4) + [r]
        ba = _get_ba(board5, evaluator, cards, ba_cache)
        hands5 = ba['hands']
        tb5 = [tb[hb] for hb in hands5]
        full = np.ones(len(hands5))
        tree = build_river_tree(pot, stacks)
        bp = blueprint_strategy_on_tree(tree, ba, db.get_average_strategy, postflop_menu=menu)
        cfr = RiverCFR(tree, ba)
        for k, bias in enumerate(nb):
            bp1 = _bias_strategy(bp, tree, 1, bias, menu)   # villain=seat1 (for M0)
            bp0 = _bias_strategy(bp, tree, 0, bias, menu)   # villain=seat0 (for M1)
            for vb in buckets:
                vmask = np.fromiter((1.0 if b == vb else 0.0 for b in tb5), float, len(tb5))
                if vmask.sum() <= 0:
                    continue
                compat = compatible_mass(ba, vmask)
                j = bidx[vb]
                v0, _ = cfr._eval(tree.root, full, vmask, lambda nid, _b=bp1: _b[nid])
                _, v1 = cfr._eval(tree.root, vmask, full, lambda nid, _b=bp0: _b[nid])
                for i, hb in enumerate(hands5):
                    if compat[i] > 1e-9:
                        ps0[k][hb][j] += v0[i] / compat[i]; pc0[k][hb][j] += 1.0
                        ps1[k][hb][j] += v1[i] / compat[i]; pc1[k][hb][j] += 1.0

    M0s = [_collapse_matrix(ps0[k], pc0[k], tb, bidx, B) for k in range(len(nb))]
    M1s = [_collapse_matrix(ps1[k], pc1[k], tb, bidx, B) for k in range(len(nb))]

    def _opp_pick(Ms):
        # The opponent picks ONE continuation per VILLAIN bucket (column), worst-for-hero,
        # by that column's (uniform-reach) hero value -- the Modicum multi-valued choice.
        # NOT an element-wise (per-cell) min: that lets the opponent pick a different
        # continuation per cell, a min-of-k order-statistic bias that makes the leaf wildly
        # over-pessimistic (and the solver over-defensive). Per-column is realistic.
        S = np.stack(Ms)                              # [k, B, B]
        pick = S.sum(axis=1).argmin(axis=0)           # [B]: per-column worst-for-hero
        return np.stack([S[pick[j], :, j] for j in range(S.shape[2])], axis=1)

    M0 = _opp_pick(M0s)                               # hero seat0, villain seat1 picks per col
    M1 = _opp_pick(M1s)                               # hero seat1, villain seat0 picks per col
    return M0, M1, buckets, bidx, tb


def leaf_value_vec(M, tb_idx, c1, c2, reach):
    """VECTORIZED card-removal-aware reach-conditioned leaf value, per hero hand
    (the solve-time form of bucketed_measure_leaf_cr). M: [B,B] hero-bucket x
    villain-bucket payoff. tb_idx: [H] villain/hero bucket ROW-index per hand (==
    bidx[tb[hand]], shared partition). c1,c2: [H] card ids (from build_turn_board_arrays).
    reach: [H] villain reach. Returns [H] hero leaf MEASURE values.

    compat[h,b] = (villain reach in bucket b) - (vb-mass sharing hero card 1) -
    (sharing card 2) + (the hero==villain combo, subtracted twice); leaf[h] =
    M[bucket(h)] . compat[h]. Same inclusion-exclusion as compatible_mass, per bucket."""
    B = M.shape[0]
    NC = 52
    H = tb_idx.shape[0]
    total = np.bincount(tb_idx, weights=reach, minlength=B)               # [B]
    flat = (np.bincount(tb_idx * NC + c1, weights=reach, minlength=B * NC) +
            np.bincount(tb_idx * NC + c2, weights=reach, minlength=B * NC))
    bcard = flat.reshape(B, NC)                                           # [B, 52]
    compat = total[None, :] - bcard[:, c1].T - bcard[:, c2].T            # [H, B]
    compat[np.arange(H), tb_idx] += reach                                 # add back own combo
    return np.einsum('hb,hb->h', M[tb_idx], compat)


def bucketed_measure_leaf(M, bidx, tb, hero_hands, villain_range):
    """Reconstruct the per-hero-hand MEASURE leaf from the bucket matrix M and a live
    villain range (dict {hand: weight}): leaf[h] = M[bucket(h)] . villain_mass_by_bucket
    (the bucket abstraction: average payoff x total villain reach per bucket, ignoring
    fine card removal). This is the reach-conditioned leaf the turn solver would use."""
    B = M.shape[0]
    mass = np.zeros(B)
    for h, w in villain_range.items():
        b = tb.get(h)
        if b is not None and w:
            mass[bidx[b]] += w
    return {h: float(M[bidx[tb[h]]] @ mass) for h in hero_hands if h in tb}


_CID = {c: i for i, c in enumerate(FULL_DECK)}


def bucketed_measure_leaf_cr(M, bidx, tb, hero_hands, villain_range):
    """Card-removal-AWARE reconstruction: like bucketed_measure_leaf, but the villain
    mass per bucket is the mass COMPATIBLE with the hero hand's two cards (inclusion-
    exclusion per bucket), matching the exact leaf's card removal -- not the raw total.
    Removes the ~5% total-vs-compatible bias that grows as buckets get finer."""
    B = M.shape[0]
    btot = np.zeros(B)
    bcard = np.zeros((B, len(_CID)))
    hw = {}
    for h, w in villain_range.items():
        b = tb.get(h)
        if b is None or not w:
            continue
        j = bidx[b]
        btot[j] += w
        bcard[j, _CID[h[0]]] += w
        bcard[j, _CID[h[1]]] += w
        hw[h] = (j, w)
    out = {}
    for h in hero_hands:
        if h not in tb:
            continue
        ia, ic = _CID[h[0]], _CID[h[1]]
        compat = btot - bcard[:, ia] - bcard[:, ic]   # exclude villains sharing a card
        v = hw.get(h)
        if v is not None:                              # add back the (a,b)==hero combo (subtracted twice)
            compat[v[0]] += v[1]
        out[h] = float(M[bidx[tb[h]]] @ compat)
    return out
