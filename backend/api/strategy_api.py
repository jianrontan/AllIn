from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import math
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
from bot.src.cfr.poker_game import make_custom_action, STARTING_STACK
from bot.src.cfr import translation
from bot.src.abstractions.sizing import PREFLOP_RAISE_MULT, preflop_open_chips
from bot.src.subgame.river_subgame_solver import RiverSubgameSolver

# The blueprint DB is opened read-only so a concurrent training run is safe.
_BLUEPRINT_PATH = resolve_blueprint_path()
BLUEPRINT_DB = BlueprintDB(_BLUEPRINT_PATH, read_only=True)
# Phase-4 bot: the river subgame solver. Off the river (or whenever the solve
# inputs are missing) it delegates to the blueprint exactly like
# ConfidenceAwareStrategy; on the river it solves the actual subgame and plays the
# exact size it finds (validated ~24x less river-exploitable than the blueprint).
# A river decision can take up to time_budget seconds (early-stops sooner on easy
# spots); GameSession's advance_bot_turns has a safe-fallback guard so a hand
# never crashes. NOTE: resolve_blueprint_path picks the highest-iteration DB; set
# ALLIN_BLUEPRINT_DB to a CURRENT-sizing blueprint/snapshot if the auto-resolved
# one predates the active abstraction (see the serve note in the river-solver docs).
BOT = RiverSubgameSolver(BLUEPRINT_DB, max_iters=200, check_every=40, time_budget=5.0)
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
        # Echo the Origin only if it's in the allow-list (a wildcard "*" is both
        # invalid alongside credentials and contradicts the restricted allow-list
        # flask-cors applies to the actual response).
        origin = request.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            response.headers.add("Access-Control-Allow-Origin", origin)
            response.headers.add("Access-Control-Allow-Credentials", "true")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type")
        response.headers.add('Access-Control-Allow-Methods', "GET, POST, OPTIONS")
        return response


# =============================================================================
# Strategy lookup
# =============================================================================

# Postflop bucket COUNTS per street, read from the live PostflopV2 centroids
# (distribution-aware/EMD clusters — integers, no human labels). Derived from the
# centroid files so the Key Explorer can't drift from the real abstraction
# (e.g. the old hardcoded 8-bucket 0-7 list). 20 flop / 16 turn / 10 river today.
# Read ONCE at import — restart the API if the centroids are re-fit to a new K.
from bot.src.abstractions.postflop_features import load_centroids
from bot.src.abstractions.card_abstractions import (
    NUM_PREFLOP_BUCKETS, NUM_PREFLOP_COARSE)
_POSTFLOP_BUCKET_COUNTS = {
    street: len(load_centroids(street)[0]) for street in ('flop', 'turn', 'river')
}
_PATTERN_CHARS = {
    'k': 'check', 'c': 'call', 'f': 'fold',
    's': 'small bet/raise', 'm': 'medium bet/raise',
    'l': 'large bet/raise', 'o': 'overbet (1.5x pot)',
    'x': 'xlarge open (5 BB)', 'a': 'all-in',
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


# Canonical size->char lives in sizing.py (shared with the LBR victim model) so the
# explorer's grid and the exploitability harness can't drift.
from bot.src.abstractions.sizing import SIZE_CHAR as _SIZE_CHAR
_SB_CHIPS, _BB_CHIPS = 1, 2


def _postflop_pattern(actions):
    """Pattern + trailing off-grid blend for a postflop betting line. A bet/raise
    may carry a `fraction` (pot fraction); it's translated onto the postflop grid
    (⅓ / ⅔ / pot). Returns (pattern, trailing_blend, error_or_None)."""
    pattern, trailing_blend, last = '', None, len(actions) - 1
    for idx, a in enumerate(actions):
        act = a.get('action')
        char, blend = None, None
        if act == 'check':
            char = 'k'
        elif act == 'call':
            char = 'c'
        elif act == 'allin':
            char = 'a'
        elif act in ('bet', 'raise'):
            frac = a.get('fraction')
            if frac is not None:
                try:
                    frac = float(frac)
                except (TypeError, ValueError):
                    return None, None, "bet fraction must be a number"
                if not math.isfinite(frac) or frac <= 0:
                    return None, None, "bet fraction must be a positive, finite number"
                tb = translation.translate_bet(frac, translation.POSTFLOP_GRID)
                char = translation.nearest_char(frac, translation.POSTFLOP_GRID)
                if len(tb) > 1:
                    blend = tb
            else:
                char = _SIZE_CHAR.get(a.get('size', 'medium'), 'm')
        if char is None:
            continue   # fold / unknown: never a valid queryable pattern char
        pattern += char
        if idx == last:
            trailing_blend = blend
    return pattern, trailing_blend, None


def _preflop_grid(num_aggr, committed_actor, to_call, pot):
    """Trained preflop bet-size grid at a node, as sorted [(char, eff_frac), ...].
    Thin wrapper over the shared translation.preflop_grid (the one definition the
    LBR victim model also uses) so the explorer and the harness can't drift."""
    return translation.preflop_grid(
        num_aggr, committed_actor, to_call, pot,
        preflop_open_chips(), PREFLOP_RAISE_MULT, _SIZE_CHAR)


def _preflop_pattern(actions):
    """Pattern + trailing off-grid blend for a preflop line. Preflop sizes are
    absolute (a BB ladder, or pot-relative 3-bets), not plain pot fractions, so a
    custom raise carries a `bb` raise-TO total. We replay the committed chips
    (SB=1, BB=2, both 100 BB deep) to recover each raise's pot fraction, then
    translate it onto the preflop grid. Returns (pattern, trailing_blend, err)."""
    committed = [float(_SB_CHIPS), float(_BB_CHIPS)]   # p0 = SB (ip), p1 = BB (oop)
    actor, num_aggr = 0, 0                              # SB acts first preflop
    pattern, trailing_blend, last = '', None, len(actions) - 1
    for idx, a in enumerate(actions):
        act = a.get('action')
        other = 1 - actor
        to_call = max(0.0, committed[other] - committed[actor])
        pot = committed[0] + committed[1]
        char, blend = None, None
        if act == 'check':
            char = 'k'
        elif act == 'call':
            char = 'c'
            committed[actor] = committed[other]
        elif act == 'allin':
            char = 'a'
            committed[actor] = float(STARTING_STACK)
            num_aggr += 1
        elif act in ('bet', 'raise'):
            bb = a.get('bb')
            if bb is not None:
                try:
                    total = float(bb) * BIG_BLIND
                except (TypeError, ValueError):
                    return None, None, "raise-to (bb) must be a number"
                if not math.isfinite(total) or total <= committed[actor]:
                    return None, None, "raise-to must exceed the current bet"
            elif num_aggr == 0:
                total = preflop_open_chips().get(
                    a.get('size', 'medium'), preflop_open_chips()['medium'])
            else:
                mult = PREFLOP_RAISE_MULT.get(a.get('size', 'medium'), 1.0)
                total = committed[actor] + to_call + mult * (pot + to_call)
            grid = _preflop_grid(num_aggr, committed[actor], to_call, pot)
            eff = translation.eff_fraction(total - committed[actor], to_call, pot)
            tb = translation.translate_bet(eff, grid)
            char = translation.nearest_char(eff, grid)
            if len(tb) > 1:
                blend = tb
            committed[actor] = total
            num_aggr += 1
        if char is None:
            continue
        pattern += char
        if idx == last:
            trailing_blend = blend
        actor = other
    return pattern, trailing_blend, None


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
    from bot.src.cfr.keys import make_info_set_key, STREET_NAMES
    adapter = GameAdapter()

    street = determine_street(community_e)
    street_idx = STREET_NAMES.index(street)
    is_preflop = (street_idx == 0)

    card_bucket = adapter.card_abstractions.get_bucket(hole_e, None)
    strength_bucket = (adapter.card_abstractions.get_bucket(hole_e, community_e)
                       if community_e else None)

    def build_key(pat):
        return (make_info_set_key(0, position, card_bucket, None, pat)
                if is_preflop else
                make_info_set_key(street_idx, position, card_bucket,
                                  strength_bucket, pat))

    # Build the current-street betting pattern + any trailing off-grid blend.
    # Preflop sizes are absolute (BB ladder / pot-relative 3-bets) so a custom
    # raise is given as a `bb` raise-to total; postflop a custom bet is a pot
    # `fraction`. Either way an off-grid FINAL (faced) bet is mapped onto the
    # trained grid and the two bracketing responses are blended -- exactly what
    # the live bot does via pending_translation.
    pattern, trailing_blend, err = (
        _preflop_pattern(actions) if is_preflop else _postflop_pattern(actions))
    if err:
        return jsonify({"error": err}), 400

    # Off-grid final bet -> blend the bracketing keys' responses (action
    # translation). Falls through to the plain single-key lookup if neither
    # bracket was trained.
    if trailing_blend:
        base = pattern[:-1]
        combined, total_w, brackets = {}, 0.0, []
        for char, w in trailing_blend:
            k = build_key(base + char)
            rec = BLUEPRINT_DB.get_record(k)
            brackets.append({"key": k, "char": char,
                             "weight": round(w, 4), "found": rec is not None})
            if rec:
                for a_, p_ in rec["strategy"].items():
                    combined[a_] = combined.get(a_, 0.0) + w * p_
                total_w += w
        if total_w > 1e-12:
            combined = {a_: p_ / total_w for a_, p_ in combined.items()}
            return jsonify({
                "key": build_key(pattern),
                "cardBucket": card_bucket, "strengthBucket": strength_bucket,
                "street": street, "found": True, "translated": True,
                "brackets": brackets, "strategy": combined,
                "legalActions": sorted(combined.keys()), "visitCount": None,
            })

    key = build_key(pattern)
    record = BLUEPRINT_DB.get_record(key)
    return jsonify({
        "key": key,
        "cardBucket": card_bucket,
        "strengthBucket": strength_bucket,
        "street": street,
        "found": record is not None,
        "translated": False,
        "strategy": record["strategy"] if record else None,
        "legalActions": record["legalActions"] if record else None,
        "visitCount": record["visitCount"] if record else 0,
    })


@app.route('/api/abstractions', methods=['GET'])
def get_abstractions():
    """Vocabulary the frontend Key Explorer dropdowns are built from."""
    return jsonify({
        # Decoupled preflop buckets: PREFLOP keys use the FINE id (pf_0..pf_29);
        # POSTFLOP keys carry the COARSE class (pf_0..pf_9) as startBucket. The Key
        # Explorer must offer the fine list for preflop and the coarse list for the
        # postflop startBucket -- offering the fine list postflop would let a user
        # build a key the (coarse-keyed) blueprint never wrote.
        "preflopBuckets": [f"pf_{i}" for i in range(NUM_PREFLOP_BUCKETS)],   # fine
        "preflopStartBuckets": [f"pf_{i}" for i in range(NUM_PREFLOP_COARSE)],  # coarse (postflop)
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

    # No contention on a freshly-minted id, but lock for uniformity.
    with SESSIONS.lock(session_id):
        session = GameSession.new(session_id, player_id, strategy_fn=BOT_RANGE_FN)
        advance_bot_turns(session, BOT)      # bot may act first (when it is SB)
        SESSIONS.put(session_id, session.to_dict())
        view = session.public_view()
    view['playerId'] = player_id
    return jsonify(view)


@app.route('/api/game/state', methods=['GET'])
def game_state():
    """Current redacted state. The frontend reads bot moves from here."""
    session_id = request.args.get('id')
    with SESSIONS.lock(session_id):          # serialize with any in-flight writer
        session, err = _load_session(session_id)
        if err:
            return err
        return jsonify(session.public_view())


@app.route('/api/game/action', methods=['POST'])
def game_action():
    """Apply the human's action only. The bot responds in a separate
    /api/game/bot-action call so the client can reveal the new card first."""
    data = request.get_json(force=True) or {}
    session_id = data.get('id')
    # One lock per session for the whole load-modify-put: concurrent requests for
    # the same session (double-click, retry, or a /bot-action racing this) can't
    # clobber each other or double-apply.
    with SESSIONS.lock(session_id):
        session, err = _load_session(session_id)
        if err:
            return err

        if not session.is_human_turn():
            return jsonify({"error": "not your turn"}), 409

        action = data.get('action')
        # Unrestricted sizing: the UI sends {action: 'bet_custom'|'raise_custom',
        # amountBb: <raise-to TOTAL in big blinds>}. Convert BB -> chips and build
        # the internal custom action string; GameSession validates against poker
        # rules.
        if action in ('bet_custom', 'raise_custom'):
            try:
                amount_bb = float(data.get('amountBb'))
            except (TypeError, ValueError):
                return jsonify({"error": "amountBb must be a number"}), 400
            # Reject NaN/inf: they slip past every chip bound check and would
            # silently corrupt stacks/pot (stack -= nan).
            if not math.isfinite(amount_bb) or amount_bb <= 0:
                return jsonify({"error": "amountBb must be a positive, finite number"}), 400
            chips = round(amount_bb * BIG_BLIND, 2)
            action = make_custom_action(action == 'raise_custom', chips)

        try:
            session.apply_action(action)
        except GameError as e:
            return jsonify({"error": str(e)}), 400

        SESSIONS.put(session.data['session_id'], session.to_dict())
        return jsonify(session.public_view())


@app.route('/api/game/bot-action', methods=['POST'])
def game_bot_action():
    """Run the bot's pending turn(s). Split out from /action so the client can
    render the freshly-dealt board and a 'thinking' indicator before the (possibly
    slow) river solve. A no-op if it isn't the bot's turn."""
    data = request.get_json(force=True) or {}
    session_id = data.get('id')
    with SESSIONS.lock(session_id):
        session, err = _load_session(session_id)
        if err:
            return err
        try:
            advance_bot_turns(session, BOT)
        except GameError as e:
            return jsonify({"error": str(e)}), 400
        SESSIONS.put(session.data['session_id'], session.to_dict())
        return jsonify(session.public_view())


@app.route('/api/game/next-hand', methods=['POST'])
def game_next_hand():
    """Deal the next hand in an existing session."""
    data = request.get_json(force=True) or {}
    session_id = data.get('id')
    with SESSIONS.lock(session_id):
        session, err = _load_session(session_id)
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
