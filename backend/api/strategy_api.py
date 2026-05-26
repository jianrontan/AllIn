from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sys
import os
import uuid

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

print(f"DEBUG: Current dir: {current_dir}")
print(f"DEBUG: Backend dir: {backend_dir}")

# --- Shared backend resources (loaded once at startup) -----------------------
from bot.src.config import resolve_blueprint_path
from bot.src.storage.blueprint_db import BlueprintDB
from bot.src.game.bot_strategy import BlueprintStrategy, ConfidenceAwareStrategy
from bot.src.game.session_store import InMemorySessionStore
from bot.src.game.game_session import GameSession, advance_bot_turns, GameError, BIG_BLIND
from bot.src.game.cards import to_engine
from bot.src.cfr.poker_game import make_custom_action

# The blueprint DB is opened read-only so a concurrent training run is safe.
_BLUEPRINT_PATH = resolve_blueprint_path()
BLUEPRINT_DB = BlueprintDB(_BLUEPRINT_PATH, read_only=True)
# Confidence-aware bot: plays the blueprint while the opponent looks on-model,
# falls back to equity-vs-range when the range tracker's confidence collapses.
# Behaves identically to BlueprintStrategy in the common (confident) case.
BOT = ConfidenceAwareStrategy(BLUEPRINT_DB)
# Opponent model for the hand-level range tracker (Phase 3); injected into every
# GameSession so the bot maintains a belief over the human's hand as it plays.
BOT_RANGE_FN = BOT.range_model_fn()

# SessionStore: in-memory for now. Swap for a Redis/DynamoDB-backed store to
# run multiple backend processes (see session_store.py).
SESSIONS = InMemorySessionStore()

print(f"DEBUG: Blueprint: {_BLUEPRINT_PATH.name} "
      f"({BLUEPRINT_DB.get_metadata('total_iterations', 0):,} iterations)")

app = Flask(__name__)

# CORS origins are env-driven (ALLIN_CORS_ORIGINS, comma-separated) so the same
# code serves localhost in dev and the real domain once deployed.
_origins_env = os.environ.get("ALLIN_CORS_ORIGINS")
ALLOWED_ORIGINS = ([o.strip() for o in _origins_env.split(",") if o.strip()]
                   if _origins_env
                   else ['http://localhost:5173', 'http://localhost:5174'])
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response


# =============================================================================
# Strategy lookup
# =============================================================================

# Postflop bucket COUNTS per street, read from the live PostflopV2 centroids
# (distribution-aware/EMD clusters — integers, no human labels). Derived from the
# centroid files so the Key Explorer can't drift from the real abstraction
# (e.g. the old hardcoded 8-bucket 0-7 list). 12 flop / 12 turn / 10 river today.
from bot.src.abstractions.postflop_features import load_centroids
_POSTFLOP_BUCKET_COUNTS = {
    street: len(load_centroids(street)[0]) for street in ('flop', 'turn', 'river')
}
_PATTERN_CHARS = {
    'k': 'check', 'c': 'call', 'f': 'fold',
    's': 'small bet/raise', 'm': 'medium bet/raise',
    'l': 'large bet/raise', 'a': 'all-in',
}


@app.route('/api/strategy', methods=['GET'])
def get_strategy():
    """Look up the blueprint strategy for one info-set key."""
    key = (request.args.get('key') or '').strip()
    if not key:
        return jsonify({"error": "Query parameter 'key' is required"}), 400

    record = BLUEPRINT_DB.get_record(key)
    if record is None:
        # Not an error: an untrained key is a valid, informative answer.
        return jsonify({
            "key": key, "found": False,
            "strategy": None, "visitCount": 0,
        })
    return jsonify({
        "key": key,
        "found": True,
        "strategy": record["strategy"],
        "legalActions": record["legalActions"],
        "visitCount": record["visitCount"],
    })


@app.route('/api/strategy/from-hand', methods=['POST'])
def strategy_from_hand():
    """
    Hand Explorer: take real cards + a betting line, derive the info-set key,
    and return the blueprint strategy. One round-trip.
    """
    data = request.get_json(force=True) or {}
    hole = [c for c in data.get('holeCards', []) if c]
    community_in = [c for c in data.get('communityCards', []) if c]
    actions = data.get('actions', [])
    position = data.get('position', 'ip')

    if position not in ('ip', 'oop'):
        return jsonify({"error": "position must be 'ip' or 'oop'"}), 400
    if len(hole) != 2:
        return jsonify({"error": "exactly two hole cards required"}), 400
    if len(community_in) not in (0, 3, 4, 5):
        return jsonify({"error": "community cards must number 0, 3, 4, or 5"}), 400

    try:
        hole_e = [to_engine(c) for c in hole]
        community_e = [to_engine(c) for c in community_in]
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    all_cards = hole_e + community_e
    if len(set(all_cards)) != len(all_cards):
        return jsonify({"error": "duplicate cards in hand"}), 400

    from bot.src.bot.game_adapter import GameAdapter
    adapter = GameAdapter()

    street = determine_street(community_e)
    cfr_history = []
    for a in actions:
        act = a.get('action')
        if act in ('fold', 'call', 'check', 'allin'):
            cfr_history.append(act)
        elif act in ('bet', 'raise'):
            cfr_history.append(f"{act}_{a.get('size', 'medium')}")

    round_state = {
        'street': street,
        'community_card': community_e,
        'cfr_history': cfr_history,
    }
    key = adapter.create_info_set_key(hole_e, round_state, position)
    card_bucket = adapter.card_abstractions.get_bucket(hole_e, None)
    strength_bucket = (adapter.card_abstractions.get_bucket(hole_e, community_e)
                       if community_e else None)

    record = BLUEPRINT_DB.get_record(key)
    return jsonify({
        "key": key,
        "cardBucket": card_bucket,
        "strengthBucket": strength_bucket,
        "street": street,
        "found": record is not None,
        "strategy": record["strategy"] if record else None,
        "legalActions": record["legalActions"] if record else None,
        "visitCount": record["visitCount"] if record else 0,
    })


@app.route('/api/abstractions', methods=['GET'])
def get_abstractions():
    """Vocabulary the frontend Key Explorer dropdowns are built from."""
    return jsonify({
        "preflopBuckets": [f"pf_{i}" for i in range(15)],
        # Per-street postflop bucket ids (distribution-aware; no semantic labels).
        "postflopBuckets": {
            street: list(range(n)) for street, n in _POSTFLOP_BUCKET_COUNTS.items()
        },
        "positions": [
            {"value": "ip", "label": "in position (button/SB)"},
            {"value": "oop", "label": "out of position (BB)"},
        ],
        "streets": ["preflop", "flop", "turn", "river"],
        "patternChars": _PATTERN_CHARS,
        "keyFormat": {
            "preflop": "{bucket}_{position}_{pattern}",
            "postflop": "{startBucket}_{strength}_{position}_{street}_{pattern}",
        },
    })


# =============================================================================
# Play against the bot
# =============================================================================

def _load_session(session_id):
    """Return a GameSession or (None, error_response)."""
    if not session_id:
        return None, (jsonify({"error": "session id required"}), 400)
    data = SESSIONS.get(session_id)
    if data is None:
        return None, (jsonify({"error": "session not found or expired"}), 404)
    return GameSession.from_dict(data, strategy_fn=BOT_RANGE_FN), None


@app.route('/api/game/new', methods=['POST'])
def game_new():
    """Start a new game session and deal the first hand."""
    data = request.get_json(silent=True) or {}
    player_id = data.get('playerId') or str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    session = GameSession.new(session_id, player_id, strategy_fn=BOT_RANGE_FN)
    advance_bot_turns(session, BOT)          # bot may act first (when it is SB)
    SESSIONS.put(session_id, session.to_dict())

    view = session.public_view()
    view['playerId'] = player_id
    return jsonify(view)


@app.route('/api/game/state', methods=['GET'])
def game_state():
    """Current redacted state. The frontend reads bot moves from here."""
    session, err = _load_session(request.args.get('id'))
    if err:
        return err
    return jsonify(session.public_view())


@app.route('/api/game/action', methods=['POST'])
def game_action():
    """Apply the human's action, then let the bot respond."""
    data = request.get_json(force=True) or {}
    session, err = _load_session(data.get('id'))
    if err:
        return err

    if not session.is_human_turn():
        return jsonify({"error": "not your turn"}), 409

    action = data.get('action')
    # Unrestricted sizing: the UI sends {action: 'bet_custom'|'raise_custom',
    # amountBb: <raise-to TOTAL in big blinds>}. Convert BB -> chips and build the
    # internal custom action string; GameSession validates against poker rules.
    if action in ('bet_custom', 'raise_custom'):
        try:
            amount_bb = float(data.get('amountBb'))
        except (TypeError, ValueError):
            return jsonify({"error": "amountBb must be a number"}), 400
        chips = round(amount_bb * BIG_BLIND, 2)
        action = make_custom_action(action == 'raise_custom', chips)

    try:
        session.apply_action(action)
        advance_bot_turns(session, BOT)
    except GameError as e:
        return jsonify({"error": str(e)}), 400

    SESSIONS.put(session.data['session_id'], session.to_dict())
    return jsonify(session.public_view())


@app.route('/api/game/next-hand', methods=['POST'])
def game_next_hand():
    """Deal the next hand in an existing session."""
    data = request.get_json(force=True) or {}
    session, err = _load_session(data.get('id'))
    if err:
        return err

    try:
        session.start_next_hand()
        advance_bot_turns(session, BOT)
    except GameError as e:
        return jsonify({"error": str(e)}), 400

    SESSIONS.put(session.data['session_id'], session.to_dict())
    return jsonify(session.public_view())


# =============================================================================
# Health
# =============================================================================

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        "status": "Server is working",
        "blueprint": _BLUEPRINT_PATH.name,
        "iterations": BLUEPRINT_DB.get_metadata('total_iterations', 0),
    })


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
    print("Starting Flask server...")
    print(f"Working directory: {os.getcwd()}")
    print(f"CORS origins: {ALLOWED_ORIGINS}")

    print("\nRoutes registered:")
    for rule in app.url_map.iter_rules():
        methods = list(rule.methods) if rule.methods else []
        print(f"  {rule.endpoint}: {rule.rule} {methods}")

    print(f"\nServer starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
