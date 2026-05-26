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
    """The CFR bot plays a full PyPokerEngine match vs a random player without
    crashing, and chips are conserved end-to-end. (Previously this swallowed all
    exceptions and asserted nothing, so it 'passed' even if the bot crashed or
    mis-bet.)"""
    initial_stack = 100
    config = setup_config(
        max_round=10, initial_stack=initial_stack, small_blind_amount=1)
    config.register_player(name="CFR_Bot", algorithm=Player())
    config.register_player(name="Random_Bot", algorithm=RandomPlayer())

    # No try/except: a bot crash or an illegal action must FAIL the test.
    game_result = start_poker(config, verbose=0)

    assert game_result is not None, "start_poker returned no result"
    players = game_result['players']
    assert len(players) == 2, f"expected 2 players, got {len(players)}"
    # Heads-up is zero-sum: total chips are conserved across the whole match.
    total = sum(p['stack'] for p in players)
    assert total == 2 * initial_stack, (
        f"chips not conserved: {total} != {2 * initial_stack}")
    assert all(p['stack'] >= 0 for p in players), "a stack went negative"


if __name__ == "__main__":
    test_game()
    print("PASS test_game (bot played a full match, chips conserved)")
