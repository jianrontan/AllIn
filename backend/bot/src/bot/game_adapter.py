# backend/bot/src/bot/game_adapter.py
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.action_abstractions import ActionAbstraction
from ..cfr.keys import make_info_set_key, action_char, STREET_NAMES


class GameAdapter:
    def __init__(self, menu_mode='control'):
        self.card_abstractions = CardAbstraction()
        # menu_mode drives the action abstraction so the PyPokerEngine path serves a
        # capped blueprint on its own menu (2.0x tier, no voluntary all-in).
        self.action_abstractions = ActionAbstraction(menu_mode=menu_mode)

    def create_info_set_key(self, hole_card, round_state, position='ip'):
        """
        Generate info set key from round state action history.
        position: 'ip' (SB/BTN, acts last postflop) or 'oop' (BB, acts first postflop).
        Built through keys.py (single source of truth) so this PyPokerEngine path
        can't drift from the trainer / live key format.
        """
        betting_pattern = self._extract_betting_pattern(round_state)
        street_str = round_state.get('street', 'preflop')
        street = STREET_NAMES.index(street_str) if street_str in STREET_NAMES else 0

        if street == 0:
            return make_info_set_key(
                0, position, self.card_abstractions.get_bucket(hole_card, None),
                None, betting_pattern)
        starting_hand = self.card_abstractions.get_bucket(hole_card, None)
        current_strength = self.card_abstractions.get_bucket(
            hole_card, round_state.get('community_card'))
        return make_info_set_key(
            street, position, starting_hand, current_strength, betting_pattern)

    def _extract_betting_pattern(self, round_state):
        """Extract betting pattern string from round_state action histories"""
        # Training path: synthetic round_state has cfr_history directly
        if 'cfr_history' in round_state:
            return ''.join([self.cfr_action_to_char(a) for a in round_state['cfr_history']])

        # Inference path: read from PyPokerEngine action_histories
        current_street = round_state.get('street', 'preflop')
        street_actions = round_state.get('action_histories', {}).get(current_street, [])

        # Reconstruct pot at start of current street
        current_pot = round_state.get('pot', {}).get('main', {}).get('amount', 3)
        paid_this_street = sum(a.get('paid', 0) for a in street_actions)
        running_pot = current_pot - paid_this_street

        pattern = ''
        for idx, action in enumerate(street_actions):
            action_type = action.get('action', '').lower()
            if action_type in ('smallblind', 'bigblind'):
                continue  # blinds are not betting-pattern actions
            if action_type == 'fold':
                pattern += 'f'
            elif action_type in ('call', 'check'):
                # PyPokerEngine has no explicit CHECK — a check is a CALL that
                # pays 0 chips. Map paid==0 to 'k' (check), otherwise 'c' (call).
                pattern += 'k' if action.get('paid', 0) == 0 else 'c'
            elif action_type in ('raise', 'bet'):
                amount = action.get('amount', 0)
                game_state = {'pot_size': running_pot, 'big_blind': 2}
                category = self.action_abstractions.categorize_bet_size(
                    {'action': action_type, 'amount': amount},
                    game_state,
                    street_actions[:idx],
                    current_street
                )
                pattern += category[0]
            running_pot += action.get('paid', 0)

        return pattern

    def cfr_action_to_char(self, cfr_action):
        """Convert CFR action to single character.

        Routes through keys.action_char (single source of truth) so this
        PyPokerEngine path can't drift from the trainer / live key format and
        correctly handles bet_xlarge ('x') / bet_overbet / raise_overbet ('o').
        Defaulting to 'x' here used to alias the real xlarge-open char and
        corrupt info-set keys."""
        return action_char(cfr_action)

