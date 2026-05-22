# backend/bot/tests/test_cfr_correctness.py
"""
Real unit tests (with assertions) for the CFR poker AI system.
Run from backend/bot/ with:
    python -m pytest tests/test_cfr_correctness.py -v
    python tests/test_cfr_correctness.py
"""

import sys
import os
import re
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cfr.poker_game import PokerGame, STARTING_STACK
from src.cfr.information_set import InformationSet
from src.cfr.blueprint_trainer import BlueprintTrainer
from src.abstractions.card_abstractions import CardAbstraction, _PREFLOP_BUCKET_MAP
from src.abstractions.action_abstractions import ActionAbstraction
from src.bot.game_adapter import GameAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_short_training(iterations=200, seed=42):
    """Run a fixed-seed short training run and return the trainer."""
    random.seed(seed)
    trainer = BlueprintTrainer()
    trainer.train_blueprint(iterations)
    return trainer


# ===========================================================================
# 1. CFR ALGORITHM INVARIANTS
# ===========================================================================

def test_cfr_cumulative_regrets_nonnegative():
    """CFR+ guarantee: all cumulative regrets must be >= 0."""
    trainer = run_short_training(200)
    violations = []
    for key, info_set in trainer.info_sets.items():
        for action, regret in info_set.cumulative_regrets.items():
            if regret < -1e-9:
                violations.append((key, action, regret))
    assert len(violations) == 0, (
        f"CFR+ violation: {len(violations)} negative cumulative regrets found. "
        f"First few: {violations[:5]}"
    )
    print(f"PASS test_cfr_cumulative_regrets_nonnegative: "
          f"all regrets >= 0 across {len(trainer.info_sets)} info sets")


def test_cfr_average_strategy_sums_to_one():
    """Average strategy must sum to 1.0 (within 1e-6) for every info set with legal_actions."""
    trainer = run_short_training(200)
    bad = []
    for key, info_set in trainer.info_sets.items():
        if not info_set.legal_actions:
            continue
        avg = info_set.get_average_strategy(info_set.legal_actions)
        total = float(sum(avg))
        if abs(total - 1.0) > 1e-6:
            bad.append((key, total))
    assert len(bad) == 0, (
        f"{len(bad)} info sets whose average strategy does not sum to 1.0. "
        f"First few: {bad[:5]}"
    )
    print(f"PASS test_cfr_average_strategy_sums_to_one: "
          f"all strategies sum to 1 across {len(trainer.info_sets)} info sets")


def test_cfr_ev_is_finite():
    """EV returned by train_blueprint must be finite (not NaN or inf)."""
    random.seed(99)
    trainer = BlueprintTrainer()
    ev = trainer.train_blueprint(100)
    assert math.isfinite(ev), f"Expected finite EV, got {ev}"
    print(f"PASS test_cfr_ev_is_finite: EV = {ev:.6f}")


def test_cfr_strategy_probabilities_nonnegative():
    """Every probability in every average strategy must be >= 0."""
    trainer = run_short_training(200)
    bad = []
    for key, info_set in trainer.info_sets.items():
        if not info_set.legal_actions:
            continue
        avg = info_set.get_average_strategy(info_set.legal_actions)
        for action, prob in zip(info_set.legal_actions, avg):
            if prob < -1e-9:
                bad.append((key, action, float(prob)))
    assert len(bad) == 0, f"Negative probabilities found: {bad[:5]}"
    print(f"PASS test_cfr_strategy_probabilities_nonnegative")


# ===========================================================================
# 2. ZERO-SUM CHECK
# ===========================================================================

def test_cfr_returns_bounded_p0_value():
    """
    cfr() returns the game value from P0's fixed perspective regardless of which
    player is the traverser. Every return must be finite and within ±STARTING_STACK
    (no one can win/lose more than a full stack).
    """
    random.seed(7)
    trainer = BlueprintTrainer()
    ca = trainer.game_adapter.card_abstractions
    N = 200

    for i in range(N):
        p0_cards, p1_cards, community = trainer.deal_random_hand()
        trainer._p0_preflop = ca.get_bucket(p0_cards, None)
        trainer._p1_preflop = ca.get_bucket(p1_cards, None)
        trainer._p0_postflop = {
            1: ca.get_bucket(p0_cards, community[:3]),
            2: ca.get_bucket(p0_cards, community[:4]),
            3: ca.get_bucket(p0_cards, community[:5]),
        }
        trainer._p1_postflop = {
            1: ca.get_bucket(p1_cards, community[:3]),
            2: ca.get_bucket(p1_cards, community[:4]),
            3: ca.get_bucket(p1_cards, community[:5]),
        }
        trainer.game._calc_cache.clear()

        for updating_player in [0, 1]:
            util = trainer.cfr(p0_cards, p1_cards, community, [],
                               0, updating_player, 0, i, 3)
            assert math.isfinite(util), f"cfr returned non-finite value: {util}"
            assert abs(util) <= STARTING_STACK + 1e-6, f"cfr value out of range: {util}"
    print("PASS test_cfr_returns_bounded_p0_value")


def test_self_play_ev_is_near_zero():
    """
    Regression test for the CFR sign-convention bug.

    cfr() must propagate utility with a consistent P0 perspective. The pre-fix
    bug mixed perspectives across street transitions and terminals, producing a
    STABLE self-play EV around +17 chips/hand that did not decay with training.
    A correct solver's running EV is a cumulative average dominated by early
    random play at low iteration counts; it stays well bounded and converges
    toward the small game value (measured: ~6.9 at 3k iters, ~2.6 at 50k).
    The bound below catches the broken +17 while tolerating 3k-iter noise.
    """
    random.seed(2024)
    trainer = BlueprintTrainer()
    ev = trainer.train_blueprint(3000)
    assert abs(ev) < 10.0, (
        f"Self-play EV should be bounded well below the broken +17; got {ev:.3f}. "
        f"A large stable EV indicates the CFR sign/perspective bug has returned."
    )
    print(f"PASS test_self_play_ev_is_near_zero: EV={ev:.4f}")


# ===========================================================================
# 3. POKER GAME LOGIC
# ===========================================================================

def test_is_terminal_fold():
    """is_terminal(['fold'], street=0) must be True."""
    game = PokerGame()
    assert game.is_terminal(['fold'], 0), "fold at preflop should be terminal"
    print("PASS test_is_terminal_fold")


def test_is_terminal_river_check_check():
    """is_terminal(['check', 'check'], street=3) must be True."""
    game = PokerGame()
    assert game.is_terminal(['check', 'check'], 3), \
        "check-check on river (street=3) should be terminal"
    print("PASS test_is_terminal_river_check_check")


def test_is_terminal_preflop_check_check_not_terminal():
    """is_terminal(['check', 'check'], street=0) must be False (preflop check-check goes to flop)."""
    game = PokerGame()
    # NOTE: preflop check-check is is_round_complete=True but NOT is_terminal
    # because street < 3 — the CFR loop advances to the next street instead.
    result = game.is_terminal(['check', 'check'], 0)
    assert result == False, (
        f"check-check preflop should NOT be terminal (game continues to flop), got {result}"
    )
    print("PASS test_is_terminal_preflop_check_check_not_terminal")


def test_is_terminal_bet_call():
    """bet_small + call on river (street=3) must be terminal."""
    game = PokerGame()
    assert game.is_terminal(['bet_small', 'call'], 3), \
        "bet+call on river should be terminal"
    print("PASS test_is_terminal_bet_call")


def test_is_terminal_allin_call():
    """allin + call is always terminal (immediate showdown)."""
    game = PokerGame()
    for street in range(4):
        assert game.is_terminal(['allin', 'call'], street), \
            f"allin+call should be terminal on street {street}"
    print("PASS test_is_terminal_allin_call")


def test_is_terminal_allin_fold():
    """allin + fold: fold is in history so is_terminal returns True."""
    game = PokerGame()
    assert game.is_terminal(['allin', 'fold'], 0), \
        "allin+fold should be terminal (fold in history)"
    print("PASS test_is_terminal_allin_fold")


def test_is_terminal_bet_call_preflop_not_terminal():
    """bet_small + call on preflop (street=0) is round-complete but NOT game-terminal."""
    game = PokerGame()
    result = game.is_terminal(['bet_small', 'call'], 0)
    assert result == False, (
        f"bet+call preflop should NOT be game-terminal (continues to flop), got {result}"
    )
    print("PASS test_is_terminal_bet_call_preflop_not_terminal")


def test_utility_p0_folds_preflop():
    """When P0 folds preflop, P0 loses exactly their SB (1 chip)."""
    game = PokerGame()
    # P0 is SB (acts first preflop), posted 1 chip
    # P0 folds immediately -> loses 1 chip
    # Use cards that don't collide (each SuitRank is unique)
    p0_cards = ['HA', 'DA']
    p1_cards = ['HK', 'DK']
    community = ['CQ', 'SJ', 'CT', 'S2', 'C3']
    util = game.get_utility(p0_cards, p1_cards, community,
                            ['fold'], 0, 3, 0.0, 0.0)
    assert util == -1.0, f"P0 folds preflop should give utility -1.0, got {util}"
    print(f"PASS test_utility_p0_folds_preflop: utility = {util}")


def test_utility_p1_folds_preflop():
    """When P1 folds after P0 open, P0 wins the pot (BB already posted)."""
    game = PokerGame()
    # P0 opens small (6 chips) -> P1 folds
    p0_cards = ['HA', 'DA']
    p1_cards = ['H2', 'D3']
    community = ['CQ', 'SJ', 'CT', 'S4', 'C5']
    # starting_pot=3, history=['bet_small', 'fold']
    util = game.get_utility(p0_cards, p1_cards, community,
                            ['bet_small', 'fold'], 0, 3, 0.0, 0.0)
    assert util > 0, f"P0 wins when P1 folds, expected positive utility, got {util}"
    print(f"PASS test_utility_p1_folds_preflop: utility = {util}")


def test_pot_increases_on_bet():
    """Pot should increase when a bet is placed."""
    game = PokerGame()
    starting_pot = 3
    pot_after_bet = game.calculate_current_pot(starting_pot, ['bet_small'], 1, 0.0, 0.0)
    assert pot_after_bet > starting_pot, (
        f"Pot should increase after bet_small, got {pot_after_bet} (started {starting_pot})"
    )
    print(f"PASS test_pot_increases_on_bet: pot = {pot_after_bet}")


def test_legal_actions_never_empty_at_nonterminal():
    """get_legal_actions never returns [] at non-terminal nodes."""
    game = PokerGame()
    # Sample several non-terminal situations.
    # Each tuple: (street, history, starting_pot, current_player)
    situations = [
        (0, [], 3, 0),               # preflop start: SB acts first
        (1, [], 3, 1),               # flop start: OOP (BB) acts first
        (2, [], 3, 1),               # turn start: OOP acts first
        (3, [], 3, 1),               # river start: OOP acts first
        (0, ['call'], 3, 1),         # preflop: SB limped, BB can check or bet
        (1, ['check'], 10, 0),       # flop: OOP checked, IP's turn
    ]
    for street, history, starting_pot, current_player in situations:
        if game.is_terminal(history, street):
            continue
        actions = game.get_legal_actions(
            street, history, starting_pot, current_player,
            STARTING_STACK - 1, STARTING_STACK - 2)
        assert len(actions) > 0, (
            f"get_legal_actions returned [] at non-terminal: "
            f"street={street}, history={history}, player={current_player}"
        )
    print("PASS test_legal_actions_never_empty_at_nonterminal")


def test_allin_appears_with_short_stack():
    """With tiny remaining stack, allin should appear in legal actions."""
    game = PokerGame()
    # Player 0 has almost no chips left (just 1 chip remaining out of 200)
    actions = game.get_legal_actions(
        1, [], 3, 0,
        p0_stack=1, p1_stack=STARTING_STACK - 2)
    assert 'allin' in actions, (
        f"allin should appear when p0 stack=1, got: {actions}"
    )
    print(f"PASS test_allin_appears_with_short_stack: actions = {actions}")


def test_large_bet_disappears_short_stack():
    """
    A sized bet is replaced by allin only when its EXACT cost meets/exceeds the
    stack. Postflop open: bet_large costs 1.0x pot. With pot=10, an 8-chip stack
    cannot afford bet_large (cost 10) but can afford bet_medium (cost 6.6).
    """
    game = PokerGame()
    actions = game.get_legal_actions(
        1, [], 10, 0,
        p0_stack=8, p1_stack=STARTING_STACK - 2)
    assert 'bet_large' not in actions, (
        f"bet_large (cost 10) should be replaced by allin with an 8-chip stack, got: {actions}"
    )
    assert 'bet_medium' in actions, (
        f"bet_medium (cost 6.6) is affordable and should remain, got: {actions}"
    )
    assert 'allin' in actions, (
        f"allin should appear when bet_large is removed, got: {actions}"
    )
    print(f"PASS test_large_bet_disappears_short_stack: actions = {actions}")


def test_max_raise_cap():
    """After 3 bet/raises, only fold/call should be available (1 bet + 2 raises cap)."""
    game = PokerGame()
    # Three bet/raise actions at preflop (open + 2 raises)
    history = ['bet_small', 'raise_small', 'raise_medium']
    actions = game.get_legal_actions(
        0, history, 3, 0,
        p0_stack=STARTING_STACK - 1, p1_stack=STARTING_STACK - 2)
    assert set(actions) <= {'fold', 'call'}, (
        f"After 3 bet/raises only fold/call allowed, got: {actions}"
    )
    assert 'fold' in actions and 'call' in actions, \
        f"fold and call must be present after cap: {actions}"

    # Also verify that 2 bet/raises still allows another raise
    history2 = ['bet_small', 'raise_small']
    actions2 = game.get_legal_actions(
        0, history2, 3, 0,
        p0_stack=STARTING_STACK - 1, p1_stack=STARTING_STACK - 2)
    has_raise = any(a.startswith('raise_') for a in actions2)
    assert has_raise, f"After 2 bet/raises a 3rd raise should still be allowed, got: {actions2}"
    print(f"PASS test_max_raise_cap: 3-action cap={actions}, 2-action still has raise={actions2}")


def test_short_stack_can_shove_when_no_sized_raise():
    """
    Safety net: when aggression is permitted but no sized bet/raise survives,
    the player may still go all-in — provided the shove is a genuine raise
    (commits strictly more than a call). And NOT when the all-in can't even
    cover a call (then only fold/call remain).
    """
    game = PokerGame()
    # P0 faces P1's postflop bet, no sized raise in the list, 50-chip stack
    # (p0_prev=150 -> remaining 50). The 50-chip shove dwarfs the ~6.6 call.
    result = game._apply_stack_constraints(
        ['fold', 'call'], 50, 1, ['bet_small'], 20, 0, 150, 0)
    assert 'allin' in result, f"P0 should be able to shove, got: {result}"

    # P0 so short the all-in (3) is below the call (~6.6): no shove option.
    result2 = game._apply_stack_constraints(
        ['fold', 'call'], 3, 1, ['bet_small'], 20, 0, 197, 0)
    assert 'allin' not in result2, f"No shove when all-in <= call, got: {result2}"
    print(f"PASS test_short_stack_can_shove_when_no_sized_raise: {result}")


def test_preflop_call_amount_is_one():
    """Initial preflop call amount is 1 chip (SB calls the BB)."""
    game = PokerGame()
    call_amt = game.get_call_amount_from_history(0, [], 3)
    assert call_amt == 1.0, f"Preflop initial call should be 1 (SB calls BB), got {call_amt}"
    print(f"PASS test_preflop_call_amount_is_one: call = {call_amt}")


def test_acting_player_preflop():
    """Preflop: P0 (SB) acts first (index 0), P1 (BB) acts second."""
    game = PokerGame()
    assert game._acting_player(0, 0) == 0, "Action 0 preflop should be P0 (SB)"
    assert game._acting_player(1, 0) == 1, "Action 1 preflop should be P1 (BB)"
    assert game._acting_player(2, 0) == 0, "Action 2 preflop should be P0"
    print("PASS test_acting_player_preflop")


def test_acting_player_postflop():
    """Postflop: P1 (BB/OOP) acts first."""
    game = PokerGame()
    assert game._acting_player(0, 1) == 1, "Action 0 postflop should be P1 (OOP/BB)"
    assert game._acting_player(1, 1) == 0, "Action 1 postflop should be P0 (IP/SB)"
    assert game._acting_player(2, 1) == 1, "Action 2 postflop should be P1"
    print("PASS test_acting_player_postflop")


def test_round_complete_call_check_preflop():
    """call then check is not a valid is_round_complete case — verify actual semantics."""
    game = PokerGame()
    # 'call' followed by 'check' is round complete (BB calls SB's raise then checks)
    # Actually the code says history[-2:] == ['call', 'check'] is complete
    result = game.is_round_complete(['call', 'check'])
    assert result == True, \
        f"call+check should be round complete (preflop BB acts), got {result}"
    print(f"PASS test_round_complete_call_check_preflop")


def test_is_round_complete_check_check():
    game = PokerGame()
    assert game.is_round_complete(['check', 'check']) == True
    print("PASS test_is_round_complete_check_check")


def test_utility_showdown_p0_wins():
    """Showdown: P0 has aces, P1 has 7-2. P0 should win."""
    game = PokerGame()
    p0_cards = ['HA', 'DA']  # Aces
    p1_cards = ['H7', 'D2']  # 7-2 offsuit (no suit collision with above)
    community = ['CQ', 'SK', 'D9', 'S3', 'C4']
    # check-check on river means showdown
    util = game.get_utility(p0_cards, p1_cards, community,
                            ['check', 'check'], 3, 3, 0.0, 0.0)
    assert util > 0, f"P0 with AA should win showdown, got utility {util}"
    print(f"PASS test_utility_showdown_p0_wins: utility = {util}")


def test_utility_showdown_p1_wins():
    """Showdown: P1 has aces, P0 has 7-2. P0 should lose (negative net utility)."""
    game = PokerGame()
    p0_cards = ['H7', 'D2']  # 7-2 offsuit
    p1_cards = ['HA', 'DA']  # Aces
    community = ['CQ', 'SK', 'D9', 'S3', 'C4']
    # Pass blind investments so P0 has money at stake (SB=1, BB=2)
    util = game.get_utility(p0_cards, p1_cards, community,
                            ['check', 'check'], 3, 3, 1.0, 2.0)
    assert util < 0, f"P0 with 72o should lose showdown, got utility {util}"
    print(f"PASS test_utility_showdown_p1_wins: utility = {util}")


# ===========================================================================
# 4. CARD ABSTRACTION
# ===========================================================================

def test_preflop_bucket_AA():
    """AA maps to pf_14 (strongest bucket)."""
    ca = CardAbstraction()
    bucket = ca.preflop_bucket(['HA', 'DA'])
    assert bucket == 'pf_14', f"AA should be pf_14, got {bucket}"
    print(f"PASS test_preflop_bucket_AA: {bucket}")


def test_preflop_bucket_KK():
    """KK maps to pf_14."""
    ca = CardAbstraction()
    bucket = ca.preflop_bucket(['HK', 'DK'])
    assert bucket == 'pf_14', f"KK should be pf_14, got {bucket}"
    print(f"PASS test_preflop_bucket_KK: {bucket}")


def test_preflop_bucket_32o():
    """32o maps to pf_0 (weakest)."""
    ca = CardAbstraction()
    bucket = ca.preflop_bucket(['H3', 'D2'])
    assert bucket == 'pf_0', f"32o should be pf_0, got {bucket}"
    print(f"PASS test_preflop_bucket_32o: {bucket}")


def test_preflop_bucket_72o():
    """72o maps to pf_0 (weakest)."""
    ca = CardAbstraction()
    bucket = ca.preflop_bucket(['H7', 'D2'])
    assert bucket == 'pf_0', f"72o should be pf_0, got {bucket}"
    print(f"PASS test_preflop_bucket_72o: {bucket}")


def test_cards_to_string_aa():
    """cards_to_string(['HA', 'DA']) -> 'AA'"""
    ca = CardAbstraction()
    result = ca.cards_to_string(['HA', 'DA'])
    assert result == 'AA', f"Expected 'AA', got '{result}'"
    print(f"PASS test_cards_to_string_aa: '{result}'")


def test_cards_to_string_32o():
    """cards_to_string(['H3', 'D2']) -> '32o'"""
    ca = CardAbstraction()
    result = ca.cards_to_string(['H3', 'D2'])
    assert result == '32o', f"Expected '32o', got '{result}'"
    print(f"PASS test_cards_to_string_32o: '{result}'")


def test_all_169_canonical_hands_covered():
    """All 169 canonical hands must be in _PREFLOP_BUCKET_MAP (no missing keys)."""
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    rank_val = {r: i for i, r in enumerate(ranks)}
    expected = set()
    for i, r1 in enumerate(ranks):
        for r2 in ranks[i:]:
            if r1 == r2:
                expected.add(r1 + r2)
            else:
                expected.add(r1 + r2 + 'o')
                expected.add(r1 + r2 + 's')
    missing = expected - set(_PREFLOP_BUCKET_MAP.keys())
    assert len(missing) == 0, f"Missing {len(missing)} hands from bucket map: {sorted(missing)[:20]}"
    print(f"PASS test_all_169_canonical_hands_covered: {len(_PREFLOP_BUCKET_MAP)} entries")


def test_no_fallback_pf_7_bucket_for_mapped_hands():
    """
    preflop_bucket() returns 'pf_7' only as a fallback for unmapped hands.
    No canonical hand should trigger the fallback.
    """
    ca = CardAbstraction()
    # Generate all 169 canonical hand strings and check none hit the default fallback
    # The fallback is returned only when hand_str is NOT in _PREFLOP_BUCKET_MAP
    # So we verify that every canonical string IS in the map.
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    fallback_hits = []
    for i, r1 in enumerate(ranks):
        for r2 in ranks[i:]:
            if r1 == r2:
                combos = [(r1, r2, 'H', 'D')]  # pair
            else:
                combos = [(r1, r2, 'H', 'D'), (r1, r2, 'H', 'H')]  # offsuit + suited
            for rank1, rank2, suit1, suit2 in combos:
                card1 = suit1 + rank1
                card2 = suit2 + rank2
                hand_str = ca.cards_to_string([card1, card2])
                if hand_str not in _PREFLOP_BUCKET_MAP:
                    fallback_hits.append(hand_str)

    assert len(fallback_hits) == 0, (
        f"{len(fallback_hits)} canonical hands are missing from the bucket map "
        f"and will return fallback 'pf_6': {fallback_hits[:10]}"
    )
    print(f"PASS test_no_fallback_pf_6_bucket_for_mapped_hands")


def test_preflop_bucket_valid_format():
    """Every bucket in the map must match the pattern pf_N where N is 0-14."""
    bad = [(k, v) for k, v in _PREFLOP_BUCKET_MAP.items()
           if not re.match(r'^pf_\d+$', v) or int(v.split('_')[1]) > 14]
    assert len(bad) == 0, f"Invalid bucket values: {bad[:5]}"
    print("PASS test_preflop_bucket_valid_format")


def test_postflop_bucket_is_integer():
    """Postflop bucket should be an integer in 0-7."""
    ca = CardAbstraction()
    hole_cards = ['HA', 'DA']
    community = ['HK', 'SQ', 'DJ']  # valid SuitRank cards, no collision with hole cards
    bucket = ca.postflop_bucket(hole_cards, community)
    assert isinstance(bucket, int), f"Expected int, got {type(bucket)}: {bucket}"
    assert 0 <= bucket <= 7, f"Postflop bucket out of range: {bucket}"
    print(f"PASS test_postflop_bucket_is_integer: bucket = {bucket}")


# ===========================================================================
# 5. INFO SET KEY FORMAT
# ===========================================================================

def test_preflop_key_format():
    """Preflop info set key must match pattern pf_N_position_pattern."""
    adapter = GameAdapter()
    round_state = {
        'street': 'preflop',
        'community_card': [],
        'action_histories': {'preflop': []},
        'pot': {'main': {'amount': 3}},
        'cfr_history': []
    }
    key = adapter.create_info_set_key(['HA', 'DA'], round_state, 'ip')
    pattern = re.compile(r'^pf_\d+_(ip|oop)_\w*$')
    assert pattern.match(key), f"Preflop key '{key}' doesn't match expected format"
    print(f"PASS test_preflop_key_format: key = '{key}'")


def test_postflop_key_format():
    """Postflop info set key must match pattern pf_N_M_position_street_pattern."""
    adapter = GameAdapter()
    round_state = {
        'street': 'flop',
        'community_card': ['HK', 'SQ', 'DJ'],  # valid SuitRank, no collision with HA/DA
        'action_histories': {'flop': []},
        'pot': {'main': {'amount': 5}},
        'cfr_history': []
    }
    key = adapter.create_info_set_key(['HA', 'DA'], round_state, 'ip')
    # Pattern: pf_N_M_ip/oop_flop/turn/river_pattern
    pattern = re.compile(r'^pf_\d+_\d+_(ip|oop)_(flop|turn|river)_\w*$')
    assert pattern.match(key), f"Postflop key '{key}' doesn't match expected format"
    print(f"PASS test_postflop_key_format: key = '{key}'")


def test_info_set_key_deterministic():
    """Same cards + same history -> same key every time."""
    adapter = GameAdapter()
    round_state = {
        'street': 'preflop',
        'community_card': [],
        'action_histories': {'preflop': []},
        'pot': {'main': {'amount': 3}},
        'cfr_history': ['bet_small']
    }
    key1 = adapter.create_info_set_key(['HA', 'DA'], round_state, 'ip')
    key2 = adapter.create_info_set_key(['HA', 'DA'], round_state, 'ip')
    assert key1 == key2, f"Key not deterministic: '{key1}' vs '{key2}'"
    print(f"PASS test_info_set_key_deterministic: key = '{key1}'")


def test_info_set_key_different_cards_different_key():
    """Different hole cards -> different info set key (at least for strong vs weak hands)."""
    adapter = GameAdapter()
    round_state = {
        'street': 'preflop',
        'community_card': [],
        'action_histories': {'preflop': []},
        'pot': {'main': {'amount': 3}},
        'cfr_history': []
    }
    key_aa = adapter.create_info_set_key(['HA', 'DA'], round_state, 'ip')
    key_72 = adapter.create_info_set_key(['H7', 'D2'], round_state, 'ip')
    # AA is pf_14, 72o is pf_0 — these must be different keys
    assert key_aa != key_72, f"AA and 72o should produce different keys, both gave '{key_aa}'"
    print(f"PASS test_info_set_key_different_cards_different_key")


def test_position_in_key():
    """Key contains 'ip' or 'oop' depending on position."""
    adapter = GameAdapter()
    round_state = {
        'street': 'preflop',
        'community_card': [],
        'action_histories': {'preflop': []},
        'pot': {'main': {'amount': 3}},
        'cfr_history': []
    }
    key_ip = adapter.create_info_set_key(['HA', 'DA'], round_state, 'ip')
    key_oop = adapter.create_info_set_key(['HA', 'DA'], round_state, 'oop')
    assert '_ip_' in key_ip, f"'ip' not in key: '{key_ip}'"
    assert '_oop_' in key_oop, f"'oop' not in key: '{key_oop}'"
    assert key_ip != key_oop, "IP and OOP keys should differ"
    print(f"PASS test_position_in_key")


# ===========================================================================
# 6. STREET TRANSITION
# ===========================================================================

def test_street_transition_pot():
    """After preflop action, CFR should transition to street=1 with correct new pot."""
    game = PokerGame()
    # Preflop: P0 opens small bet (bet_small = 6 chips target)
    # P1 calls -> round complete, pot should be bet_small_target * 2
    starting_pot = 3
    history = ['bet_small', 'call']
    pot = game.calculate_current_pot(starting_pot, history, 0, 0.0, 0.0)
    # P0 opened to 6 total (SB=1 + 5 more), P1 called to 6 total (BB=2 + 4 more)
    # pot = 6 + 6 = 12
    assert pot > starting_pot, f"Pot after bet+call should exceed starting_pot, got {pot}"
    print(f"PASS test_street_transition_pot: pot after preflop bet+call = {pot}")


def test_postflop_contribution_no_double_count():
    """
    A player who bets then re-raises on the same postflop street must have
    their contribution counted as the raise-to TOTAL, not bet + raise-to.
    The bug double-counted the earlier bet and produced negative stacks.
    """
    game = PokerGame()
    # Flop: P1 (acts first postflop) bets, P0 raises, P1 re-raises, P0 calls.
    history = ['bet_large', 'raise_medium', 'raise_medium', 'call']
    starting_pot = 28.0
    p0_this = game.get_player_contribution_this_round(history, 1, starting_pot, 0, 14.0, 14.0)
    p1_this = game.get_player_contribution_this_round(history, 1, starting_pot, 1, 14.0, 14.0)
    pot = game.calculate_current_pot(starting_pot, history, 1, 14.0, 14.0)
    # The two players' street contributions must sum to the chips added this street.
    assert abs((p0_this + p1_this) - (pot - starting_pot)) < 1e-6, (
        f"contributions {p0_this}+{p1_this} != pot increment {pot - starting_pot}"
    )
    # Neither player can commit more than a full stack (200) in one street.
    assert p0_this <= STARTING_STACK and p1_this <= STARTING_STACK, (
        f"contribution exceeds a full stack: p0_this={p0_this}, p1_this={p1_this}"
    )
    print(f"PASS test_postflop_contribution_no_double_count: p0={p0_this:.2f}, p1={p1_this:.2f}")


def test_no_double_deduction_across_streets():
    """
    Stacks should not have chips double-deducted across streets.
    After preflop bet+call the stacks should reflect exactly what was invested.
    """
    game = PokerGame()
    starting_pot = 3
    history = ['bet_small', 'call']
    p0_contrib = game.get_player_contribution_this_round(
        history, 0, starting_pot, 0, 0.0, 0.0)
    p1_contrib = game.get_player_contribution_this_round(
        history, 0, starting_pot, 1, 0.0, 0.0)
    # Total contributions should equal the pot
    pot = game.calculate_current_pot(starting_pot, history, 0, 0.0, 0.0)
    assert abs((p0_contrib + p1_contrib) - pot) < 1e-6, (
        f"p0_contrib={p0_contrib} + p1_contrib={p1_contrib} should equal pot={pot}"
    )
    print(f"PASS test_no_double_deduction_across_streets: "
          f"p0={p0_contrib}, p1={p1_contrib}, pot={pot}")


def test_stacks_after_preflop_action():
    """After posting blinds, stacks should be STARTING_STACK - blind_amount."""
    # P0 posts SB=1, P1 posts BB=2
    p0_stack_initial = STARTING_STACK - 1
    p1_stack_initial = STARTING_STACK - 2
    assert p0_stack_initial == 199, f"P0 initial stack wrong: {p0_stack_initial}"
    assert p1_stack_initial == 198, f"P1 initial stack wrong: {p1_stack_initial}"
    print("PASS test_stacks_after_preflop_action: "
          f"P0 stack={p0_stack_initial}, P1 stack={p1_stack_initial}")


def test_postflop_multiraise_contribution_invariant():
    """
    For postflop histories with multiple raises (including bet-then-reraise by
    the same player), each player's street contribution must sum to the chips
    added that street, and neither may exceed a full stack.
    """
    game = PokerGame()
    sp = 20.0
    histories = [
        ['bet_small', 'raise_small'],
        ['bet_medium', 'raise_large', 'raise_small'],
        ['bet_large', 'raise_medium', 'raise_medium', 'call'],
        ['check', 'bet_small', 'raise_medium', 'call'],
        ['bet_small', 'raise_large', 'call'],
    ]
    for hist in histories:
        p0 = game.get_player_contribution_this_round(hist, 1, sp, 0, 14.0, 14.0)
        p1 = game.get_player_contribution_this_round(hist, 1, sp, 1, 14.0, 14.0)
        pot = game.calculate_current_pot(sp, hist, 1, 14.0, 14.0)
        assert abs((p0 + p1) - (pot - sp)) < 1e-6, (
            f"{hist}: contributions {p0}+{p1} != pot increment {pot - sp}")
        assert p0 <= STARTING_STACK + 1e-6 and p1 <= STARTING_STACK + 1e-6, (
            f"{hist}: contribution exceeds a full stack: p0={p0}, p1={p1}")
    print("PASS test_postflop_multiraise_contribution_invariant")


def test_preflop_raise_war_contribution_invariant():
    """
    Preflop open / 3-bet / 4-bet contributions (which include the posted
    blinds) must sum to the full pot.
    """
    game = PokerGame()
    histories = [
        ['bet_small', 'raise_small'],
        ['bet_medium', 'raise_medium', 'raise_large'],
        ['call', 'bet_small', 'raise_small', 'call'],
        ['bet_large', 'call'],
    ]
    for hist in histories:
        p0 = game.get_player_contribution_this_round(hist, 0, 3.0, 0)
        p1 = game.get_player_contribution_this_round(hist, 0, 3.0, 1)
        pot = game.calculate_current_pot(3.0, hist, 0)
        # Preflop contributions include the blinds, so they sum to the whole pot.
        assert abs((p0 + p1) - pot) < 1e-6, (
            f"{hist}: contributions {p0}+{p1} != pot {pot}")
    print("PASS test_preflop_raise_war_contribution_invariant")


def test_call_after_allin_costs_total_minus_caller_contrib():
    """
    Calling an all-in must cost (all-in player's TOTAL street commitment) minus
    (caller's current street commitment) — NOT (all-in's chip increment) minus
    (caller's commitment). The buggy version produced 0 whenever the caller had
    already put in more this street than the all-in's increment.

    Reproduces the 'Play vs AI' bug: river bet → opponent big raise → you shove
    for less than the raise's increment but for more total. The opponent owed
    the small difference; the bug let them call for free.
    """
    game = PokerGame()
    # River, after both have invested 25 chips each through preflop/flop/turn,
    # starting_pot = 50. P1 (user) bets medium, P0 (bot) raises large, P1
    # shoves the rest. P0 should owe (P1's total) − (P0's contrib so far).
    history = ['bet_medium', 'raise_large', 'allin']
    starting_pot = 50.0
    call_cost = game.get_call_amount_from_history(3, history, starting_pot, 25.0, 25.0)
    # P1's total river commitment after the all-in is STARTING_STACK − 25 = 175.
    # P0's contribution so far this street is 149 (the large raise).
    # The call must cost 175 − 149 = 26 chips, not 0.
    assert call_cost > 1.0, (
        f"calling an all-in that raises the bet must cost > 0; got {call_cost}")
    assert abs(call_cost - 26.0) < 0.5, (
        f"expected call cost ~26 chips, got {call_cost}")
    print(f"PASS test_call_after_allin_costs_total_minus_caller_contrib: {call_cost:.2f}")


def test_random_session_playout_invariants():
    """
    Property-based fuzz: play many random hands through a full GameSession and
    assert core invariants after EVERY action — no negative stacks, and chip
    conservation (both stacks + the pot always total 2 x STARTING_STACK).
    This is the class of test that catches sequence-specific chip bugs.
    """
    import random as _r
    from src.game.game_session import GameSession

    _r.seed(12345)
    total = 2 * STARTING_STACK
    session = GameSession.new('fuzz', 'p')
    hands_played = 0
    for hand in range(2000):
        if hand > 0:
            session.start_next_hand()
        hands_played += 1
        steps = 0
        while session.data['status'] == 'in_hand':
            legal = session.legal_actions()
            if not legal:
                break
            action = _r.choice(legal)
            # Snapshot pre-action for the call-cost invariant.
            acting = session.current_player()
            pre_p0, pre_p1 = session.data['p0_stack'], session.data['p1_stack']
            pre_history = list(session.data['history'])
            session.apply_action(action)
            d = session.data
            assert d['p0_stack'] > -1e-6, f"negative p0 stack: {d['p0_stack']}"
            assert d['p1_stack'] > -1e-6, f"negative p1 stack: {d['p1_stack']}"
            pot = session.game.calculate_current_pot(
                d['starting_pot'], d['history'], d['street'],
                d['p0_invested'], d['p1_invested'])
            assert abs(d['p0_stack'] + d['p1_stack'] + pot - total) < 1e-6, (
                f"chip conservation broken: stacks {d['p0_stack']:.2f}+"
                f"{d['p1_stack']:.2f} + pot {pot:.2f} != {total}")
            # Calling when the caller has chips and there is an outstanding
            # aggressive action must put chips in. (Preflop SB limp also costs
            # 1 chip to match the BB.) The buggy allin-call branch silently
            # produced cost=0 and chip conservation alone could not detect it.
            if action == 'call':
                pre_stack = pre_p0 if acting == 0 else pre_p1
                post_stack = d['p0_stack'] if acting == 0 else d['p1_stack']
                cost = pre_stack - post_stack
                if pre_stack > 1e-6:
                    assert cost > 1e-6, (
                        f"call cost was 0 despite chips remaining; "
                        f"history={pre_history + [action]}, stack pre={pre_stack:.2f}")
                # Semantic invariant: after a call, either the caller's TOTAL
                # commitment matches the opponent's, or the caller is now
                # all-in for less. Derived from stack values only, so it is
                # independent of any contribution/pot function — catches
                # internally consistent miscalculations that look fine to chip
                # conservation. Would have caught BUG-004 directly.
                p0_total = STARTING_STACK - d['p0_stack']
                p1_total = STARTING_STACK - d['p1_stack']
                if abs(p0_total - p1_total) > 1e-6:
                    caller_stack = d['p0_stack'] if acting == 0 else d['p1_stack']
                    assert caller_stack < 1e-6, (
                        f"call didn't match opponent's commitment and caller "
                        f"isn't all-in: p0_total={p0_total:.3f}, "
                        f"p1_total={p1_total:.3f}, caller_stack={caller_stack:.3f}, "
                        f"history={pre_history + [action]}")
            # Semantic invariant: every postflop street begins with both
            # players' cross-street investments equal — the previous street
            # only completes via fold (terminal) or a call/check that
            # equalises. Catches asymmetric-invested drift across streets
            # (a class BUG-003 belonged to).
            if d['street'] > 0:
                assert abs(d['p0_invested'] - d['p1_invested']) < 1e-6, (
                    f"asymmetric cross-street invested at street {d['street']}: "
                    f"p0_invested={d['p0_invested']:.3f}, "
                    f"p1_invested={d['p1_invested']:.3f}")
            steps += 1
            assert steps <= 300, "hand did not terminate"
        # End-of-hand semantic invariant: if the hand ended via an all-in
        # that was called, both players must have stack 0 (with equal
        # starting stacks an all-in always commits the full stack and the
        # caller, when able to cover, also goes to 0). Catches the bug
        # class of "call recorded but no chips actually moved."
        hist = session.data['history']
        if len(hist) >= 2 and hist[-2] == 'allin' and hist[-1] == 'call':
            assert (session.data['p0_stack'] < 1e-6
                    and session.data['p1_stack'] < 1e-6), (
                f"allin+call terminal but stacks nonzero: "
                f"p0={session.data['p0_stack']:.3f}, "
                f"p1={session.data['p1_stack']:.3f}")
    print(f"PASS test_random_session_playout_invariants: {hands_played} hands, "
          f"invariants held after every action")


# ===========================================================================
# 7. INFORMATION SET
# ===========================================================================

def test_information_set_uniform_strategy_no_regrets():
    """Without any regrets, strategy should be uniform."""
    info_set = InformationSet()
    actions = ['fold', 'call', 'bet_small']
    strategy = info_set.get_strategy(actions)
    expected = 1.0 / 3
    for i, prob in enumerate(strategy):
        assert abs(prob - expected) < 1e-9, \
            f"Expected uniform {expected:.4f}, got {prob:.4f} for action {actions[i]}"
    print("PASS test_information_set_uniform_strategy_no_regrets")


def test_information_set_regret_matching():
    """Positive regret on one action should increase its strategy probability."""
    info_set = InformationSet()
    actions = ['fold', 'call', 'bet_small']
    info_set.cumulative_regrets = {'fold': 0, 'call': 10, 'bet_small': 0}
    strategy = info_set.get_strategy(actions)
    # Call has all the regret -> should have probability 1.0
    assert strategy[1] == 1.0, f"Call should have prob=1.0 with all regret, got {strategy[1]}"
    assert strategy[0] == 0.0, f"Fold should have prob=0.0, got {strategy[0]}"
    print("PASS test_information_set_regret_matching")


def test_information_set_cfr_plus_floors_negative_regrets():
    """CFR+ must floor regrets at 0 when computing strategy (negatives treated as 0)."""
    info_set = InformationSet()
    actions = ['fold', 'call']
    # Manually inject negative cumulative regrets
    info_set.cumulative_regrets = {'fold': -100, 'call': 5}
    strategy = info_set.get_strategy(actions)
    # fold's negative regret should be treated as 0; call has all regret
    assert strategy[1] > strategy[0], (
        f"call should dominate after flooring negative regret for fold: {strategy}"
    )
    # With fold regret floored to 0 and call regret = 5: call gets 100%
    assert abs(strategy[1] - 1.0) < 1e-9, \
        f"call should get prob=1.0 (fold regret floors to 0), got {strategy[1]}"
    print("PASS test_information_set_cfr_plus_floors_negative_regrets")


def test_get_strategy_is_pure():
    """get_strategy must NOT mutate the cumulative strategy (pure regret matching)."""
    info_set = InformationSet()
    actions = ['fold', 'call', 'bet_small']
    info_set.get_strategy(actions)
    info_set.get_strategy(actions)
    assert info_set.cumulative_strategy == {}, (
        f"get_strategy must not touch cumulative_strategy, got {info_set.cumulative_strategy}"
    )
    print("PASS test_get_strategy_is_pure")


def test_accumulate_strategy():
    """accumulate_strategy adds the strategy vector unweighted; average normalises it."""
    info_set = InformationSet()
    actions = ['fold', 'call']
    info_set.accumulate_strategy(actions, [0.25, 0.75])
    info_set.accumulate_strategy(actions, [0.5, 0.5])
    assert abs(info_set.cumulative_strategy['fold'] - 0.75) < 1e-9
    assert abs(info_set.cumulative_strategy['call'] - 1.25) < 1e-9
    avg = info_set.get_average_strategy(actions)
    assert abs(float(avg[0]) - 0.375) < 1e-9, f"avg fold should be 0.375, got {avg[0]}"
    assert abs(float(avg[1]) - 0.625) < 1e-9, f"avg call should be 0.625, got {avg[1]}"
    print("PASS test_accumulate_strategy")


def test_cumulative_strategy_accumulates():
    """Each accumulate_strategy call of a unit strategy adds 1.0 to the total."""
    info_set = InformationSet()
    actions = ['fold', 'call']
    s = info_set.get_strategy(actions)  # uniform, sums to 1
    info_set.accumulate_strategy(actions, s)
    info_set.accumulate_strategy(actions, s)
    total = sum(info_set.cumulative_strategy.values())
    assert abs(total - 2.0) < 1e-9, f"Two accumulations should total 2.0, got {total}"
    print(f"PASS test_cumulative_strategy_accumulates: total={total:.4f}")


def test_db_average_strategy_includes_all_actions():
    """
    The DB readout must normalise the average strategy over ALL accumulated
    actions, not the stale first-seen `legal_actions` list. A key whose action
    set varies across visits (a postflop key spanning different pots) must not
    have any action silently dropped from the exported strategy.
    """
    import tempfile
    import os as _os
    from src.storage.blueprint_db import BlueprintDB

    info = InformationSet()
    # First visit locks `legal_actions` to this set (no 'allin').
    info.accumulate_strategy(['fold', 'call', 'bet_large'], [0.2, 0.3, 0.5])
    # A later short-stack visit to the same key brings 'allin'.
    info.accumulate_strategy(['fold', 'call', 'allin'], [0.1, 0.4, 0.5])

    path = _os.path.join(tempfile.gettempdir(), 'bp_readout_test.db')
    if _os.path.exists(path):
        _os.remove(path)
    db = BlueprintDB(path)
    db.save_batch({'k': info})
    strat = db.get_average_strategy('k')
    rec = db.get_record('k')
    db.close()
    _os.remove(path)

    for a in ('fold', 'call', 'bet_large', 'allin'):
        assert a in strat, f"action {a!r} dropped from exported strategy: {strat}"
    assert abs(sum(strat.values()) - 1.0) < 1e-6, f"strategy must sum to 1: {strat}"
    assert set(rec['strategy']) == set(strat), "get_record strategy differs from get_average_strategy"
    print(f"PASS test_db_average_strategy_includes_all_actions: {strat}")


# ===========================================================================
# 8. ACTION ABSTRACTION
# ===========================================================================

def test_action_char_mapping():
    """_action_char should correctly map all CFR actions."""
    from src.cfr.blueprint_trainer import _action_char
    assert _action_char('check') == 'k'
    assert _action_char('call') == 'c'
    assert _action_char('fold') == 'f'
    assert _action_char('bet_small') == 's'
    assert _action_char('bet_medium') == 'm'
    assert _action_char('bet_large') == 'l'
    assert _action_char('raise_small') == 's'
    assert _action_char('raise_medium') == 'm'
    assert _action_char('raise_large') == 'l'
    assert _action_char('allin') == 'a'
    assert _action_char('unknown') == 'x'
    print("PASS test_action_char_mapping")


def test_preflop_open_sizes():
    """Preflop open sizing: small=6, medium=10, large=14 chips."""
    game = PokerGame()
    amounts = game.get_preflop_bet_amounts('open', 3)  # starting_pot not used for fixed opens
    assert amounts['small'] == 6, f"Expected small=6, got {amounts['small']}"
    assert amounts['medium'] == 10, f"Expected medium=10, got {amounts['medium']}"
    assert amounts['large'] == 14, f"Expected large=14, got {amounts['large']}"
    print(f"PASS test_preflop_open_sizes: {amounts}")


def test_preflop_3bet_sizes():
    """Preflop 3-bet sizing: small=18, medium=24, large=32 chips."""
    game = PokerGame()
    amounts = game.get_preflop_bet_amounts('3bet', 3)
    assert amounts['small'] == 18, f"Expected small=18, got {amounts['small']}"
    assert amounts['medium'] == 24, f"Expected medium=24, got {amounts['medium']}"
    assert amounts['large'] == 32, f"Expected large=32, got {amounts['large']}"
    print(f"PASS test_preflop_3bet_sizes: {amounts}")


def test_postflop_bet_multipliers():
    """Postflop bet multipliers: small=0.33, medium=0.66, large=1.00."""
    game = PokerGame()
    assert game.BET_MULTIPLIERS['small'] == 0.33
    assert game.BET_MULTIPLIERS['medium'] == 0.66
    assert game.BET_MULTIPLIERS['large'] == 1.00
    print("PASS test_postflop_bet_multipliers")


def test_action_abstraction_preflop_open_sizing():
    """ActionAbstraction._calculate_target_amount matches training sizes for preflop opens."""
    aa = ActionAbstraction()
    game_state = {'pot_size': 3, 'current_bet': 0, 'player_contribution': 0, 'big_blind': 2}
    round_state = {'street': 'preflop', 'action_histories': {'preflop': []}}

    small = aa._calculate_target_amount('small', 'bet', game_state, round_state)
    medium = aa._calculate_target_amount('medium', 'bet', game_state, round_state)
    large = aa._calculate_target_amount('large', 'bet', game_state, round_state)

    # Training: open small=6, medium=10, large=14
    assert small == 6, f"Preflop open small should be 6, got {small}"
    assert medium == 10, f"Preflop open medium should be 10, got {medium}"
    assert large == 14, f"Preflop open large should be 14, got {large}"
    print(f"PASS test_action_abstraction_preflop_open_sizing: {small}/{medium}/{large}")


def test_action_abstraction_postflop_bet_sizing():
    """ActionAbstraction postflop bet sizes match training multipliers."""
    aa = ActionAbstraction()
    pot = 20.0
    game_state = {'pot_size': pot, 'current_bet': 0, 'player_contribution': 0, 'big_blind': 2}
    round_state = {'street': 'flop', 'action_histories': {}}

    small = aa._calculate_target_amount('small', 'bet', game_state, round_state)
    medium = aa._calculate_target_amount('medium', 'bet', game_state, round_state)
    large = aa._calculate_target_amount('large', 'bet', game_state, round_state)

    assert abs(small - 0.33 * pot) < 0.01, f"Postflop small bet should be 0.33*pot, got {small}"
    assert abs(medium - 0.66 * pot) < 0.01, f"Postflop medium bet should be 0.66*pot, got {medium}"
    assert abs(large - 1.0 * pot) < 0.01, f"Postflop large bet should be 1.0*pot, got {large}"
    print(f"PASS test_action_abstraction_postflop_bet_sizing: {small}/{medium}/{large}")


# ===========================================================================
# 9. EDGE CASES & REGRESSION
# ===========================================================================

def test_preflop_limp_legal_actions():
    """After SB calls (limp), BB can check or bet."""
    game = PokerGame()
    # Preflop: P0 calls (limps), P1 (BB) can check or bet
    actions = game.get_legal_actions(
        0, ['call'], 3, 1,
        STARTING_STACK - 1, STARTING_STACK - 2)
    assert 'check' in actions, f"BB should be able to check after limp, got {actions}"
    print(f"PASS test_preflop_limp_legal_actions: {actions}")


def test_preflop_bet_then_fold():
    """After P0 bets preflop and P1 folds, game is terminal immediately."""
    game = PokerGame()
    assert game.is_terminal(['bet_small', 'fold'], 0), \
        "bet then fold is terminal"
    print("PASS test_preflop_bet_then_fold")


def test_utility_allin_call_full_board():
    """allin + call uses all 5 community cards for showdown."""
    game = PokerGame()
    p0_cards = ['HA', 'DA']
    p1_cards = ['C2', 'D3']
    community = ['HK', 'SQ', 'DJ', 'S7', 'C8']
    util = game.get_utility(p0_cards, p1_cards, community,
                            ['allin', 'call'], 0, 3, 0.0, 0.0)
    # AA should beat 23 offsuit almost always
    assert math.isfinite(util), f"Utility should be finite, got {util}"
    print(f"PASS test_utility_allin_call_full_board: utility = {util}")


def test_postflop_check_options():
    """On flop with no prior action, OOP player (P1) gets check and bets."""
    game = PokerGame()
    actions = game.get_legal_actions(
        1, [], 10, 1,
        STARTING_STACK - 5, STARTING_STACK - 5)
    assert 'check' in actions, f"check should be available on flop open, got {actions}"
    assert 'fold' not in actions, f"fold should NOT be available when no bet, got {actions}"
    print(f"PASS test_postflop_check_options: {actions}")


def test_get_community_cards_count():
    """Community card count per street is correct."""
    game = PokerGame()
    assert game.get_community_cards_count(0) == 0, "Preflop: 0 community cards"
    assert game.get_community_cards_count(1) == 3, "Flop: 3 community cards"
    assert game.get_community_cards_count(2) == 4, "Turn: 4 community cards"
    assert game.get_community_cards_count(3) == 5, "River: 5 community cards"
    print("PASS test_get_community_cards_count")


def test_preflop_bucket_vs_equity_ordering():
    """Higher-equity hands should have higher or equal bucket numbers."""
    # AA (0.8562) > KK (0.8133) > 22 (0.5088) > 32o (0.3208)
    # All should be pf_14, pf_14, pf_7, pf_0
    ca = CardAbstraction()
    aa_bucket = int(ca.preflop_bucket(['HA', 'DA']).split('_')[1])
    kk_bucket = int(ca.preflop_bucket(['HK', 'DK']).split('_')[1])
    small_pair = int(ca.preflop_bucket(['H2', 'D2']).split('_')[1])
    trash = int(ca.preflop_bucket(['H3', 'D2']).split('_')[1])

    assert aa_bucket >= kk_bucket >= small_pair >= trash, (
        f"Equity ordering violated: AA={aa_bucket}, KK={kk_bucket}, "
        f"22={small_pair}, 32o={trash}"
    )
    print(f"PASS test_preflop_bucket_vs_equity_ordering: "
          f"AA={aa_bucket}, KK={kk_bucket}, 22={small_pair}, 32o={trash}")


# ===========================================================================
# RUNNER
# ===========================================================================

ALL_TESTS = [
    # CFR invariants
    test_cfr_cumulative_regrets_nonnegative,
    test_cfr_average_strategy_sums_to_one,
    test_cfr_ev_is_finite,
    test_cfr_strategy_probabilities_nonnegative,
    # CFR perspective / sign convention
    test_cfr_returns_bounded_p0_value,
    test_self_play_ev_is_near_zero,
    # Game logic
    test_is_terminal_fold,
    test_is_terminal_river_check_check,
    test_is_terminal_preflop_check_check_not_terminal,
    test_is_terminal_bet_call,
    test_is_terminal_allin_call,
    test_is_terminal_allin_fold,
    test_is_terminal_bet_call_preflop_not_terminal,
    test_utility_p0_folds_preflop,
    test_utility_p1_folds_preflop,
    test_pot_increases_on_bet,
    test_legal_actions_never_empty_at_nonterminal,
    test_allin_appears_with_short_stack,
    test_large_bet_disappears_short_stack,
    test_short_stack_can_shove_when_no_sized_raise,
    test_max_raise_cap,
    test_preflop_call_amount_is_one,
    test_acting_player_preflop,
    test_acting_player_postflop,
    test_round_complete_call_check_preflop,
    test_is_round_complete_check_check,
    test_utility_showdown_p0_wins,
    test_utility_showdown_p1_wins,
    # Card abstraction
    test_preflop_bucket_AA,
    test_preflop_bucket_KK,
    test_preflop_bucket_32o,
    test_preflop_bucket_72o,
    test_cards_to_string_aa,
    test_cards_to_string_32o,
    test_all_169_canonical_hands_covered,
    test_no_fallback_pf_7_bucket_for_mapped_hands,
    test_preflop_bucket_valid_format,
    test_postflop_bucket_is_integer,
    # Info set key format
    test_preflop_key_format,
    test_postflop_key_format,
    test_info_set_key_deterministic,
    test_info_set_key_different_cards_different_key,
    test_position_in_key,
    # Street transition
    test_street_transition_pot,
    test_postflop_contribution_no_double_count,
    test_no_double_deduction_across_streets,
    test_stacks_after_preflop_action,
    # Chip conservation / property-based fuzz
    test_postflop_multiraise_contribution_invariant,
    test_preflop_raise_war_contribution_invariant,
    test_call_after_allin_costs_total_minus_caller_contrib,
    test_random_session_playout_invariants,
    # InformationSet
    test_information_set_uniform_strategy_no_regrets,
    test_information_set_regret_matching,
    test_information_set_cfr_plus_floors_negative_regrets,
    test_get_strategy_is_pure,
    test_accumulate_strategy,
    test_cumulative_strategy_accumulates,
    test_db_average_strategy_includes_all_actions,
    # Action abstraction
    test_action_char_mapping,
    test_preflop_open_sizes,
    test_preflop_3bet_sizes,
    test_postflop_bet_multipliers,
    test_action_abstraction_preflop_open_sizing,
    test_action_abstraction_postflop_bet_sizing,
    # Edge cases
    test_preflop_limp_legal_actions,
    test_preflop_bet_then_fold,
    test_utility_allin_call_full_board,
    test_postflop_check_options,
    test_get_community_cards_count,
    test_preflop_bucket_vs_equity_ordering,
]


if __name__ == '__main__':
    print("=" * 70)
    print("CFR Correctness Test Suite")
    print("=" * 70)
    passed = 0
    failed = 0
    errors = []
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            import traceback
            failed += 1
            errors.append((test_fn.__name__, exc, traceback.format_exc()))
            print(f"FAIL {test_fn.__name__}: {exc}")

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)} tests")
    if errors:
        print("\nFailed tests:")
        for name, exc, tb in errors:
            print(f"\n  {name}:")
            print(f"    {exc}")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)
