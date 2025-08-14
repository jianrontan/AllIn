from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

print(f"DEBUG: Current dir: {current_dir}")
print(f"DEBUG: Backend dir: {backend_dir}")

app = Flask(__name__)

CORS(app, origins=['http://localhost:5173', 'http://localhost:5174'], 
     supports_credentials=True)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        print(f"DEBUG: Handling OPTIONS request for {request.path}")
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

def test_imports():
    """Test critical imports on startup"""
    try:
        from bot.src.bot.game_adapter import GameAdapter
        from bot.src.cfr.poker_game import PokerGame
        print("✅ Critical imports successful")
        return True
    except Exception as e:
        print(f"❌ Imports failed: {e}")
        return False

@app.route('/api/evaluate-hand', methods=['POST'])
def evaluate_hand():
    print(f"DEBUG: Received {request.method} request on /api/evaluate-hand")
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        hole_cards = data.get("holeCards", [])
        community_cards = data.get("communityCards", [])
        actions = data.get("actions", [])
        game_state = data.get("gameState", {})

        if len(hole_cards) < 2:
            return jsonify({"error": "At least two hole cards required"}), 400

        hole_cards = [card.strip().upper() for card in hole_cards if card]

        # Import here
        from bot.src.bot.game_adapter import GameAdapter
        from bot.src.cfr.poker_game import PokerGame

        poker_game = PokerGame()
        game_adapter = GameAdapter()

        # Determine street
        street = determine_street(community_cards)

        # Convert frontend actions to CFR format
        cfr_history = []
        for a in actions:
            act = a.get('action')
            if act in ['fold', 'call', 'check']:
                cfr_history.append(act)
            elif act in ['bet', 'raise']:
                size = a.get('size', 'medium')
                cfr_history.append(f"{act}_{size}")

        # Create round state
        round_state = {
            'street': street,
            'community_card': community_cards,
            'cfr_history': cfr_history,
            'pot': {'main': {'amount': game_state.get('potSize', 3)}}
        }

        info_key = game_adapter.create_info_set_key(hole_cards, round_state)
        card_bucket = game_adapter.card_abstractions.get_bucket(hole_cards, None)

        strength_bucket = None
        if community_cards:
            strength_bucket = game_adapter.card_abstractions.get_bucket(hole_cards, community_cards)

        action_pattern = ''.join([game_adapter.cfr_action_to_char(act) for act in cfr_history])

        return jsonify({
            "infoSetKey": info_key,
            "cardBucket": card_bucket,
            "strengthBucket": strength_bucket,
            "actionPattern": action_pattern,
            "debugInfo": {
                "cfr_history": cfr_history,
                "street": street,
                "numCommunityCards": len(community_cards)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-legal-actions', methods=['POST'])
def get_legal_actions():
    print(f"DEBUG: Received {request.method} request on /api/get-legal-actions")
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        from bot.src.cfr.poker_game import PokerGame
        poker_game = PokerGame()

        community_cards = data.get("communityCards", [])
        actions = data.get("actions", [])
        game_state = data.get("gameState", {})

        street_num = 0
        if len(community_cards) == 3:
            street_num = 1
        elif len(community_cards) == 4:
            street_num = 2
        elif len(community_cards) == 5:
            street_num = 3

        # Build CFR style history
        history = []
        for a in actions:
            act = a.get('action')
            if act in ['fold', 'call', 'check']:
                history.append(act)
            elif act in ['bet', 'raise']:
                size = a.get('size', 'medium')
                history.append(f"{act}_{size}")

        player_to_act = len(history) % 2
        legal = poker_game.get_legal_actions(street_num, history, 3, player_to_act)

        return jsonify({
            "legalActions": legal,
            "street": street_num,
            "history": history
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"status": "Server is working", "message": "HTTP communication successful"})

def determine_street(community_cards):
    if len(community_cards) == 0:
        return "preflop"
    elif len(community_cards) == 3:
        return "flop"
    elif len(community_cards) == 4:
        return "turn"
    elif len(community_cards) == 5:
        return "river"
    else:
        return "preflop"

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    print(f"📁 Working directory: {os.getcwd()}")
    
    if not test_imports():
        print("❌ Import test failed - server may not work properly")
    else:
        print("✅ All imports successful")
    
    print("\n📊 Routes registered:")
    for rule in app.url_map.iter_rules():
        methods = list(rule.methods) if rule.methods else []
        print(f"  {rule.endpoint}: {rule.rule} {methods}")
    
    print(f"\n🌐 Server starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
