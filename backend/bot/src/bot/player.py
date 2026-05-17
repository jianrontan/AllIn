# backend/bot/src/bot/player.py
from pypokerengine.players import BasePokerPlayer
from src.bot.game_adapter import GameAdapter
from src.storage.blueprint_db import BlueprintDB
import random
from pathlib import Path


class Player(BasePokerPlayer):
    def __init__(self):
        self.db = None
        self.game_adapter = GameAdapter()
        self.my_uuid = None
        self._load_db()

    def _load_db(self):
        try:
            current_dir = Path(__file__).parent
            db_path = (current_dir / ".." / ".." / "analysis" / "blueprint.db").resolve()
            self.db = BlueprintDB(db_path)
            total_iterations = self.db.get_metadata('total_iterations', 0)
            print(f"Loaded blueprint DB: {db_path}")
            print(f"Training iterations: {total_iterations}")
        except Exception as e:
            print(f"Error loading blueprint DB: {e}")

    def declare_action(self, valid_actions, hole_card, round_state):
        """Simple blueprint action selection with debug output"""
        action, amount = self._get_blueprint_action(
            valid_actions, hole_card, round_state)

        # Debug output
        print(f"[CFR_Bot] Hand: {hole_card}, Action: {action}:{amount}")

        return action, amount

    def _get_my_position(self, round_state):
        """Return 'ip' if bot is SB/BTN (acts last postflop), 'oop' if BB."""
        preflop_actions = round_state.get('action_histories', {}).get('preflop', [])
        for action in preflop_actions:
            if action.get('action', '').upper() == 'SMALLBLIND':
                return 'ip' if action.get('uuid') == self.my_uuid else 'oop'
        return 'ip'

    def _get_blueprint_action(self, valid_actions, hole_card, round_state):
        """Get action using trained blueprint strategy"""
        position = self._get_my_position(round_state)
        info_set_key = self.game_adapter.create_info_set_key(hole_card, round_state, position)
        game_state = self.extract_game_state(round_state)
        cfr_actions = self.game_adapter.action_abstractions.pypoker_to_cfr_actions(
            valid_actions, game_state
        )

        strategy_dict = self.db.get_average_strategy(info_set_key) if self.db else None

        if strategy_dict:
            strategy = [strategy_dict.get(a, 0.0) for a in cfr_actions]
            print(f"[CFR_Bot] Found: {info_set_key}")
        else:
            strategy = [1.0 / len(cfr_actions)] * len(cfr_actions)
            print(f"[CFR_Bot] Unknown: {info_set_key}, using uniform strategy")

        print(f"[CFR_Bot] Strategy: {dict(zip(cfr_actions, strategy))}")

        selected_cfr_action = random.choices(cfr_actions, weights=strategy)[0]
        print(f"[CFR_Bot] Selected: {selected_cfr_action}")

        action, amount = self.game_adapter.action_abstractions.cfr_to_pypoker_action(
            selected_cfr_action, valid_actions, round_state, game_state
        )
        return action, amount

    def extract_game_state(self, round_state):
        """Extract relevant game state info for abstractions"""
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        big_blind = self.extract_big_blind(round_state)
        current_bet = self.extract_current_bet(round_state)

        if self.my_uuid:
            player_stack = self.extract_player_stack(round_state, self.my_uuid)
            player_contribution = self.extract_player_contribution(
                round_state, self.my_uuid)
        else:
            player_stack = 100
            player_contribution = 0

        return {
            'pot_size': pot_size,
            'player_stack': player_stack,
            'current_bet': current_bet,
            'player_contribution': player_contribution,
            'big_blind': big_blind
        }

    def extract_big_blind(self, round_state):
        """Extract big blind directly from action history"""
        preflop_actions = round_state.get(
            'action_histories', {}).get('preflop', [])
        for action in preflop_actions:
            if action.get('action', '').upper() == 'BIGBLIND':
                return action.get('amount', 2)
        return 2

    def extract_current_bet(self, round_state):
        """Extract current bet - the total amount a player needs to have contributed"""
        current_street = round_state.get('street', 'preflop')
        actions = round_state.get(
            'action_histories', {}).get(current_street, [])

        current_bet = 0
        for action in actions:
            action_type = action.get('action', '').upper()
            if action_type in ['BET', 'RAISE', 'BIGBLIND']:
                current_bet = action.get('amount', 0)

        return current_bet

    def extract_player_stack(self, round_state, player_uuid):
        """Extract specific player's stack"""
        seats = round_state.get('seats', [])
        for seat in seats:
            if seat.get('uuid') == player_uuid:
                return seat.get('stack', 0)
        return 0

    def extract_player_contribution(self, round_state, player_uuid):
        """Extract how much specific player has contributed this street"""
        current_street = round_state.get('street', 'preflop')
        actions = round_state.get(
            'action_histories', {}).get(current_street, [])

        total_contribution = 0
        for action in actions:
            if action.get('uuid') == player_uuid:
                action_type = action.get('action', '').upper()
                if action_type in ['BET', 'RAISE', 'CALL', 'BIGBLIND', 'SMALLBLIND']:
                    if action_type == 'CALL':
                        total_contribution += action.get('amount', 0)
                    else:
                        # For bets/raises, this IS the total contribution
                        total_contribution = action.get('amount', 0)

        return total_contribution

    def receive_round_start_message(self, round_count, hole_card, seats):
        if self.my_uuid is None:
            for seat in seats:
                if seat.get('name') == 'CFR_Bot':
                    self.my_uuid = seat.get('uuid')
                    break

    def receive_game_start_message(self, game_info):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        """Track actions for history building"""
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """Learn from results"""
        pass
