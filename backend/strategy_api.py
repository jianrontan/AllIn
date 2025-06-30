import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from bot.src.bot.game_adapter import GameAdapter
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/api/evaluate-hand', methods=['POST'])
def evaluate_hand():
    data = request.get_json(force=True)
    if data is None:
        return jsonify({'error': 'No JSON data received'}), 400

    hole_cards = data.get('holeCards')
    if hole_cards is None:
        return jsonify({'error': 'holeCards field is required'}), 400

    community_cards = data.get('communityCards', [])
    actions = data.get('actions', [])
    game_state = data.get('gameState', {})

    game_adapter = GameAdapter()

    try:
        street = determine_street(community_cards)
        actual_pot_size = game_state.get('potSize', 3)
        actual_player_stack = game_state.get(
            'playerStack', 100)

        round_state = {
            'street': street,
            'community_card': community_cards,
            'action_histories': {
                street: actions
            },
            'pot': {
                'main': {
                    'amount': actual_pot_size
                }
            },
            'seats': [
                {'stack': actual_player_stack},
                {'stack': 100}
            ]
        }

        # Generate info set key using actual game state
        info_set_key = game_adapter.create_info_set_key(
            hole_cards, round_state)

        # Get abstractions using actual game state
        card_bucket = game_adapter.card_abstractions.get_bucket(
            hole_cards, None)

        strength_bucket = None
        if community_cards:
            strength_bucket = game_adapter.card_abstractions.get_bucket(
                hole_cards, community_cards)

        action_pattern = game_adapter.extract_betting_history(round_state)

        return jsonify({
            'infoSetKey': info_set_key,
            'cardBucket': card_bucket,
            'strengthBucket': strength_bucket,
            'actionPattern': action_pattern,
            'debugInfo': {
                'actualPotSize': actual_pot_size,
                'actualPlayerStack': actual_player_stack,
                'street': street,
                'numCommunityCards': len(community_cards)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def determine_street(community_cards):
    """Helper function to determine street from community cards"""
    if len(community_cards) == 0:
        return 'preflop'
    elif len(community_cards) == 3:
        return 'flop'
    elif len(community_cards) == 4:
        return 'turn'
    elif len(community_cards) == 5:
        return 'river'
    else:
        return 'preflop'


if __name__ == '__main__':
    app.run(debug=True, port=5000)
