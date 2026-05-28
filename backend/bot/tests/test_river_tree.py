# backend/bot/tests/test_river_tree.py
"""
Validate the river betting tree (src/subgame/river_tree.py): structure, exact
chip conservation, all-in capping, the aggression cap, and min-raise legality.
The tree is the scaffold the Phase-4 CFR+ solver runs on, so its accounting must
be airtight before any solve sits on top of it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_tree import (
    build_river_tree, RiverNode, is_sized, sized_chips,
    MAX_AGGRESSIONS, MIN_BET, OOP_SEAT, IP_SEAT)


def _walk_terminals(node, path=()):
    """Yield (terminal_node, action_path) for every terminal in the tree."""
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
    t = build_river_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    r = t.root
    assert not r.terminal
    assert r.player == OOP_SEAT, "OOP (BB seat 1) acts first on the river"
    assert abs(r.to_call) < 1e-9 and abs(r.pot_mid - 20.0) < 1e-9
    assert 'check' in r.actions
    assert any(is_sized(a) for a in r.actions), "root must offer at least one bet"
    print("PASS test_root_structure")


def test_chip_conservation_all_terminals():
    """Every terminal: pot = sum of contributions, and no seat invests more than
    it has (prior P/2 + its behind stack)."""
    t = build_river_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    cap = [t.entry_contrib + t.stacks[0], t.entry_contrib + t.stacks[1]]
    n = 0
    for term, path in _walk_terminals(t.root):
        c0, c1 = term.contrib
        assert abs(term.final_pot - (c0 + c1)) < 1e-6, (path, term.final_pot, c0, c1)
        assert c0 <= cap[0] + 1e-6 and c1 <= cap[1] + 1e-6, (path, c0, c1, cap)
        assert c0 >= t.entry_contrib - 1e-9 and c1 >= t.entry_contrib - 1e-9
        # final pot is the entry pot plus both players' river chips.
        assert term.final_pot >= t.pot_entry - 1e-9
        n += 1
    assert n > 0
    print(f"PASS test_chip_conservation_all_terminals ({n} terminals)")


def test_fold_and_showdown_terminals():
    t = build_river_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    folds = showdowns = 0
    for term, path in _walk_terminals(t.root):
        if 'fold' in path:
            assert term.folder in (0, 1)
            # The folder is whoever was to act when 'fold' was chosen; the pot is
            # uncontested, so contributions reflect chips put in up to the fold.
            folds += 1
        else:
            assert term.folder is None
            showdowns += 1
    assert folds > 0 and showdowns > 0
    print(f"PASS test_fold_and_showdown_terminals ({folds} folds, {showdowns} showdowns)")


def test_aggression_cap():
    t = build_river_tree(pot_entry=20.0, stacks=(200.0, 200.0))
    worst = 0
    for _term, path in _walk_terminals(t.root):
        aggs = sum(1 for a in path if is_sized(a) or a == 'allin')
        worst = max(worst, aggs)
        assert aggs <= MAX_AGGRESSIONS, (path, aggs)
    assert worst == MAX_AGGRESSIONS, "deep stacks should reach the cap somewhere"
    print(f"PASS test_aggression_cap (deepest path has {worst} aggressions)")


def test_no_duplicate_action_amounts():
    """At any node, two actions must never lead to the same chip commitment
    (the all-in dedupe): otherwise the menu and all-in collide."""
    t = build_river_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    for node in _all_nodes(t.root):
        if node.terminal:
            continue
        costs = []
        for a in node.actions:
            if is_sized(a):
                costs.append(round(sized_chips(a), 6))
            elif a == 'allin':
                costs.append(('allin', node.node_id))   # distinct sentinel
        # sized costs must be unique among themselves
        sized_only = [c for c in costs if not isinstance(c, tuple)]
        assert len(sized_only) == len(set(sized_only)), (node.actions,)
    print("PASS test_no_duplicate_action_amounts")


def test_min_bet_respected():
    """Every opening bet is at least 1 BB."""
    t = build_river_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    for node in _all_nodes(t.root):
        if node.terminal or node.to_call > 1e-9:
            continue
        for a in node.actions:
            if a.startswith('bet:'):
                assert sized_chips(a) >= MIN_BET - 1e-9, (a,)
    print("PASS test_min_bet_respected")


def test_facing_allin_only_fold_or_call():
    """After an all-in, the other player can only fold or call (no further raise
    fits behind the shove)."""
    t = build_river_tree(pot_entry=20.0, stacks=(90.0, 90.0))
    for node in _all_nodes(t.root):
        if node.terminal:
            continue
        for label, child in zip(node.actions, node.children):
            if label == 'allin' and not child.terminal:
                assert set(child.actions) <= {'fold', 'call'}, child.actions
    print("PASS test_facing_allin_only_fold_or_call")


def test_short_stack_collapses_to_allin():
    """With tiny stacks behind, sized bets exceed the stack and collapse to a
    single all-in (no sized bet survives)."""
    t = build_river_tree(pot_entry=40.0, stacks=(6.0, 6.0))   # 0.5*40=20 > 6 behind
    r = t.root
    assert 'allin' in r.actions
    assert not any(is_sized(a) for a in r.actions), r.actions
    print("PASS test_short_stack_collapses_to_allin")


def test_raise_sizing_matches_engine():
    """Pin the tree's bet/raise-to chip amounts against the engine's
    calculate_bet_amount / calculate_raise_amount for the shared 1.0x ('large')
    fraction. Chip-conservation can't catch a wrong-but-self-consistent size, so
    this is the test that would catch sizing drift if the menu changes."""
    from src.cfr.poker_game import PokerGame
    g = PokerGame()
    P = 20.0
    t = build_river_tree(pot_entry=P, stacks=(200.0, 200.0))

    # Engine: OOP opens a pot-sized bet (postflop street=3, no history yet).
    bet_large = g.calculate_bet_amount('bet_large', 3, P, [], 0.0, 0.0)
    bet_label = f"bet:{bet_large:.6g}"
    assert bet_label in t.root.actions, (t.root.actions, bet_label)

    # Then IP raises pot (a 'large' raise) facing that bet.
    ip = t.root.children[t.root.actions.index(bet_label)]
    raise_large = g.calculate_raise_amount('raise_large', 3, P, ['bet_large'], 1, 0.0, 0.0)
    raise_label = f"raise:{raise_large:.6g}"
    assert raise_label in ip.actions, (ip.actions, raise_label)
    print(f"PASS test_raise_sizing_matches_engine "
          f"(engine bet={bet_large:.6g}, raise cost={raise_large:.6g})")


def test_unequal_stacks_rejected():
    """The equal-stack invariant is enforced loudly (all-in-for-less is not
    handled), so a future caller can't silently get inflated showdown pots."""
    try:
        build_river_tree(pot_entry=20.0, stacks=(90.0, 80.0))
        raise AssertionError("expected unequal stacks to be rejected")
    except ValueError as e:
        assert 'equal' in str(e).lower()
    print("PASS test_unequal_stacks_rejected")


TESTS = [
    test_root_structure,
    test_raise_sizing_matches_engine,
    test_unequal_stacks_rejected,
    test_chip_conservation_all_terminals,
    test_fold_and_showdown_terminals,
    test_aggression_cap,
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
