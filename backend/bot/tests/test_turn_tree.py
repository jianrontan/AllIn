# backend/bot/tests/test_turn_tree.py
"""
Validate the turn betting tree (src/subgame/turn_tree.py): structure, exact chip
conservation, all-in capping, the aggression cap, min-raise legality, AND the
depth-limit leaf semantics that distinguish it from the river tree -- a non-fold
close is a LEAF (river to come), not a showdown, and carries the (final_pot,
leaf_stacks) the M0 leaf value function consumes. The tree is the scaffold the M2
turn CFR+ solver runs on, so its accounting must be airtight before any solve.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.turn_tree import (
    build_turn_tree, TurnNode, is_leaf, is_sized, sized_chips,
    MAX_AGGRESSIONS, MIN_BET, OOP_SEAT, IP_SEAT)


def _walk_terminals(node, path=()):
    if node.terminal:
        yield node, path
        return
    for label, child in zip(node.actions, node.children):
        yield from _walk_terminals(child, path + (label,))


def _all_nodes(node):
    yield node
    if not node.terminal:
        for c in node.children:
            yield from _all_nodes(c)


def test_root_structure():
    t = build_turn_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    r = t.root
    assert not r.terminal
    assert r.player == OOP_SEAT, "OOP (BB seat 1) acts first on the turn"
    assert abs(r.to_call) < 1e-9 and abs(r.pot_mid - 20.0) < 1e-9
    assert 'check' in r.actions
    assert any(is_sized(a) for a in r.actions), "root must offer at least one bet"
    print("PASS test_root_structure")


def test_chip_conservation_all_terminals():
    """Every terminal: pot = sum of contributions, no seat invests more than it has;
    and for LEAVES the behind (leaf) stacks + the pot account for ALL table chips."""
    t = build_turn_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    cap = [t.entry_contrib + t.stacks[0], t.entry_contrib + t.stacks[1]]
    table_chips = t.pot_entry + t.stacks[0] + t.stacks[1]
    n = 0
    for term, path in _walk_terminals(t.root):
        c0, c1 = term.contrib
        assert abs(term.final_pot - (c0 + c1)) < 1e-6, (path, term.final_pot, c0, c1)
        assert c0 <= cap[0] + 1e-6 and c1 <= cap[1] + 1e-6, (path, c0, c1, cap)
        assert c0 >= t.entry_contrib - 1e-9 and c1 >= t.entry_contrib - 1e-9
        assert term.final_pot >= t.pot_entry - 1e-9
        if is_leaf(term):
            # pot entering river + both behind stacks == all chips at the table.
            s0, s1 = term.leaf_stacks
            assert abs((term.final_pot + s0 + s1) - table_chips) < 1e-6, (path, s0, s1)
            assert s0 >= -1e-9 and s1 >= -1e-9, (path, term.leaf_stacks)
        n += 1
    assert n > 0
    print(f"PASS test_chip_conservation_all_terminals ({n} terminals)")


def test_fold_and_leaf_terminals():
    """Folds carry a folder + are NOT leaves; non-fold closes are LEAVES with folder
    None and EQUAL behind stacks (the inner-river equal-stack precondition)."""
    t = build_turn_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    folds = leaves = 0
    for term, path in _walk_terminals(t.root):
        if 'fold' in path:
            assert term.folder in (0, 1) and not is_leaf(term)
            folds += 1
        else:
            assert is_leaf(term), path
            s0, s1 = term.leaf_stacks
            assert abs(s0 - s1) < 1e-6, ("leaf behind stacks must be equal", path, term.leaf_stacks)
            # behind == turn-entry stack minus this seat's turn chips
            assert abs(s0 - (t.stacks[0] - (term.contrib[0] - t.entry_contrib))) < 1e-6
            leaves += 1
    assert folds > 0 and leaves > 0
    print(f"PASS test_fold_and_leaf_terminals ({folds} folds, {leaves} leaves)")


def test_no_river_showdown_terminal_kind():
    """Sanity: the turn tree must NOT produce showdown terminals -- every non-fold
    terminal is a depth-limit leaf (this is the river-vs-turn difference)."""
    t = build_turn_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    for term, path in _walk_terminals(t.root):
        assert term.folder is not None or is_leaf(term), path
    print("PASS test_no_river_showdown_terminal_kind")


def test_allin_called_leaf_has_zero_behind():
    """An all-in that gets CALLED closes the turn with both stacks committed -> the
    leaf has zero behind on both sides, so the leaf value fn yields pure
    equity-to-river (no inner river betting). Needs no special case in the tree."""
    t = build_turn_tree(pot_entry=20.0, stacks=(40.0, 40.0))
    found = 0
    for term, path in _walk_terminals(t.root):
        if is_leaf(term) and path and path[-1] == 'call' and 'allin' in path:
            s0, s1 = term.leaf_stacks
            assert abs(s0) < 1e-6 and abs(s1) < 1e-6, (path, term.leaf_stacks)
            found += 1
    assert found > 0, "expected at least one all-in-called leaf"
    print(f"PASS test_allin_called_leaf_has_zero_behind ({found} all-in-called leaves)")


def test_positional_alternation():
    """After OOP checks, the actor flips to IP (seat 0) -- positional correctness
    beyond just the root."""
    t = build_turn_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    ci = t.root.actions.index('check')
    assert t.root.children[ci].player == IP_SEAT, t.root.children[ci].player
    print("PASS test_positional_alternation")


def test_node_and_leaf_counts_pinned():
    """Pin the exact tree shape for a known config, so a silent builder regression
    (an extra/missing close or aggression) is caught even when chip-conservation holds."""
    t = build_turn_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    terms = list(_walk_terminals(t.root))
    leaves = sum(1 for n, _ in terms if is_leaf(n))
    folds = sum(1 for n, _ in terms if n.folder is not None)
    assert len(t.decision_nodes) == 72, len(t.decision_nodes)
    assert len(terms) == 141, len(terms)
    assert leaves == 71 and folds == 70, (leaves, folds)
    print(f"PASS test_node_and_leaf_counts_pinned (72 nodes, 141 terminals, 71 leaves, 70 folds)")


def test_aggression_cap():
    t = build_turn_tree(pot_entry=20.0, stacks=(200.0, 200.0))
    worst = 0
    for _term, path in _walk_terminals(t.root):
        aggs = sum(1 for a in path if is_sized(a) or a == 'allin')
        worst = max(worst, aggs)
        assert aggs <= MAX_AGGRESSIONS, (path, aggs)
    assert worst == MAX_AGGRESSIONS, "deep stacks should reach the cap somewhere"
    print(f"PASS test_aggression_cap (deepest path has {worst} aggressions)")


def test_aggression_cap_is_a_runtime_param():
    """The aggression cap is a runtime parameter, not fixed at the blueprint's 3 --
    the live solver can pass a deeper cap. Deep stacks so the cap binds."""
    deep = (1000.0, 1000.0)

    def deepest(t):
        return max(sum(1 for a in path if is_sized(a) or a == 'allin')
                   for _term, path in _walk_terminals(t.root))

    t3 = build_turn_tree(pot_entry=20.0, stacks=deep, max_aggressions=3)
    t5 = build_turn_tree(pot_entry=20.0, stacks=deep, max_aggressions=5)
    assert deepest(t3) == 3, deepest(t3)
    assert deepest(t5) == 5, "max_aggressions=5 must reach a 5th aggression"
    print(f"PASS test_aggression_cap_is_a_runtime_param (cap 3 -> {deepest(t3)}, "
          f"cap 5 -> {deepest(t5)})")


def test_no_duplicate_action_amounts():
    t = build_turn_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    for node in _all_nodes(t.root):
        if node.terminal:
            continue
        sized_only = [round(sized_chips(a), 6) for a in node.actions if is_sized(a)]
        assert len(sized_only) == len(set(sized_only)), (node.actions,)
    print("PASS test_no_duplicate_action_amounts")


def test_min_bet_respected():
    t = build_turn_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    for node in _all_nodes(t.root):
        if node.terminal or node.to_call > 1e-9:
            continue
        for a in node.actions:
            if a.startswith('bet:'):
                assert sized_chips(a) >= MIN_BET - 1e-9, (a,)
    print("PASS test_min_bet_respected")


def test_facing_allin_only_fold_or_call():
    t = build_turn_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    for node in _all_nodes(t.root):
        if node.terminal:
            continue
        for label, child in zip(node.actions, node.children):
            if label == 'allin' and not child.terminal:
                assert set(child.actions) <= {'fold', 'call'}, child.actions
    print("PASS test_facing_allin_only_fold_or_call")


def test_short_stack_collapses_to_allin():
    t = build_turn_tree(pot_entry=40.0, stacks=(6.0, 6.0))   # 0.5*40=20 > 6 behind
    r = t.root
    assert 'allin' in r.actions
    assert not any(is_sized(a) for a in r.actions), r.actions
    print("PASS test_short_stack_collapses_to_allin")


def test_raise_sizing_matches_engine():
    """Pin the turn tree's bet/raise-to chip amounts against the engine's postflop
    sizing for the shared 1.0x ('large') fraction at street=2 (turn). Chip
    conservation can't catch a wrong-but-self-consistent size, so this guards drift."""
    from src.cfr.poker_game import PokerGame
    g = PokerGame()
    P = 20.0
    t = build_turn_tree(pot_entry=P, stacks=(200.0, 200.0))

    bet_large = g.calculate_bet_amount('bet_large', 2, P, [], 0.0, 0.0)
    bet_label = f"bet:{bet_large:.6g}"
    assert bet_label in t.root.actions, (t.root.actions, bet_label)

    ip = t.root.children[t.root.actions.index(bet_label)]
    raise_large = g.calculate_raise_amount('raise_large', 2, P, ['bet_large'], 1, 0.0, 0.0)
    raise_label = f"raise:{raise_large:.6g}"
    assert raise_label in ip.actions, (ip.actions, raise_label)
    print(f"PASS test_raise_sizing_matches_engine "
          f"(engine bet={bet_large:.6g}, raise cost={raise_large:.6g})")


def test_unequal_stacks_rejected():
    """The equal-stack invariant is enforced loudly so leaf behind-stacks stay equal
    (the inner river eval's precondition)."""
    try:
        build_turn_tree(pot_entry=20.0, stacks=(90.0, 80.0))
        raise AssertionError("expected unequal stacks to be rejected")
    except ValueError as e:
        assert 'equal' in str(e).lower()
    print("PASS test_unequal_stacks_rejected")


TESTS = [
    test_root_structure,
    test_positional_alternation,
    test_node_and_leaf_counts_pinned,
    test_raise_sizing_matches_engine,
    test_unequal_stacks_rejected,
    test_chip_conservation_all_terminals,
    test_fold_and_leaf_terminals,
    test_no_river_showdown_terminal_kind,
    test_allin_called_leaf_has_zero_behind,
    test_aggression_cap,
    test_aggression_cap_is_a_runtime_param,
    test_no_duplicate_action_amounts,
    test_min_bet_respected,
    test_facing_allin_only_fold_or_call,
    test_short_stack_collapses_to_allin,
]

if __name__ == '__main__':
    passed = failed = 0
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\nResults: {passed} passed, {failed} failed out of {len(TESTS)}")
    sys.exit(1 if failed else 0)
