import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
from src.bot.player import Player
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.game import setup_config, start_poker


class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        action = random.choice(valid_actions)
        action_type = action['action']

        if 'amount' in action:
            if isinstance(action['amount'], dict):
                amount = random.randint(
                    action['amount']['min'], action['amount']['max'])
            else:
                amount = action['amount']
        else:
            amount = 0

        return action_type, amount

    def receive_game_start_message(self, game_info):
        """Required by PyPokerEngine"""
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        """Required by PyPokerEngine"""
        pass

    def receive_street_start_message(self, street, round_state):
        """Required by PyPokerEngine"""
        pass

    def receive_game_update_message(self, action, round_state):
        """Required by PyPokerEngine"""
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """Required by PyPokerEngine"""
        pass


def test_game():
    try:
        # Setup game
        config = setup_config(
            max_round=10, initial_stack=100, small_blind_amount=1)
        config.register_player(name="CFR_Bot", algorithm=Player())
        config.register_player(name="Random_Bot", algorithm=RandomPlayer())

        # Run game
        print("Starting poker game...")
        game_result = start_poker(config, verbose=1)
        print(f"Game completed: {game_result}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_game()
