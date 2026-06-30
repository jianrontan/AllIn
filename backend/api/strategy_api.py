from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import hashlib
import json
import logging
import math
import sys
import os
import threading
import time
import uuid
from collections import OrderedDict

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)
sys.path.insert(0, current_dir)         # so sibling `auth` module imports resolve

# Load a local .env for DEV convenience (repo root, then backend/), BEFORE any
# os.environ reads below. No-op if python-dotenv isn't installed (prod) or the
# file is absent; override=False so real host env vars (Lightsail) always win.
try:
    from dotenv import load_dotenv
    _repo_root = os.path.dirname(backend_dir)
    load_dotenv(os.path.join(_repo_root, '.env'))
    load_dotenv(os.path.join(backend_dir, '.env'))
except ImportError:
    pass

# Logging is configured once at the WSGI entrypoint (wsgi.py) / dev-server block;
# here we just get a module logger. No top-level print() so production logs
# (CloudWatch / Lightsail) stay clean.
_LOG = logging.getLogger(__name__)
_LOG.debug("sys.path set: current=%s backend=%s", current_dir, backend_dir)

# --- Shared backend resources (loaded once at startup) -----------------------
from bot.src.storage.blueprint_source import make_blueprint_source
from bot.src.storage.blueprint_db import BlueprintDB
from bot.src.game.session_store import make_session_store, SessionLockTimeout
from bot.src.game.player_store import (
    make_player_store, InvalidHandle, HandleTaken, AccountConflict,
    sanitize_display_name)
from bot.src.game.global_stats_store import make_global_stats_store
from bot.src.game.hand_store import (make_hand_store, recap_from_session, recap_version,
                                      InMemoryHandStore)
from bot.src.game.game_session import GameSession, advance_bot_turns, GameError, BIG_BLIND
from bot.src.game.cards import to_engine
from bot.src.cfr.poker_game import make_custom_action, STARTING_STACK
from bot.src.cfr import translation
from bot.src.abstractions.sizing import PREFLOP_RAISE_MULT, preflop_open_chips
from bot.src.subgame.river_subgame_solver import RiverSubgameSolver
from bot.src.game.range_tracker import RangeTracker, _FULL_DECK
from bot.src.subgame.river_tree import is_sized, sized_chips

# The blueprint DB is opened read-only so a concurrent training run is safe. The
# source (local file in v1; an S3 object later) is chosen by ALLIN_BLUEPRINT_SOURCE
# behind make_blueprint_source(); local_path() returns a path BlueprintDB can open.
#
# Wrapped so a misconfigured deploy (missing blueprint, corrupt DB) doesn't kill
# Flask at import. Without this, `gunicorn wsgi:app` import fails → workers
# CrashLoopBackOff → Lightsail's health-check sees nothing on port 5000 →
# deployment fails opaquely with no signal. Now the import survives, every
# strategy/game endpoint 503s with a clear "blueprint not loaded" reason, and
# /api/healthz exposes the load error for ops.
_BLUEPRINT_LOAD_ERROR = None         # full detail -- logs only, never a response
_BLUEPRINT_LOAD_ERROR_PUBLIC = None  # exception TYPE only -- safe for healthz
_BLUEPRINT_PATH = None
BLUEPRINT_DB = None
BLUEPRINT_MENU_MODE = 'control'
try:
    _BLUEPRINT_PATH = make_blueprint_source().local_path()
    BLUEPRINT_DB = BlueprintDB(_BLUEPRINT_PATH, read_only=True)
    from bot.src.abstractions.sizing import db_menu_mode
    BLUEPRINT_MENU_MODE = db_menu_mode(BLUEPRINT_DB)
except Exception as _bp_err:
    _BLUEPRINT_LOAD_ERROR = f"{type(_bp_err).__name__}: {_bp_err}"
    # The full message can embed container filesystem paths (e.g. a
    # FileNotFoundError's path) -- keep those in the server logs, expose only
    # the exception type on the public healthz endpoint.
    _BLUEPRINT_LOAD_ERROR_PUBLIC = (
        f"blueprint failed to load ({type(_bp_err).__name__}); see server logs")
    _LOG.error("blueprint load failed; serving in degraded mode (healthz=503): %s",
               _BLUEPRINT_LOAD_ERROR)
# Postflop action-translation grid for the SERVED blueprint's menu (capped adds the
# 2.0x 'overbet2' tier). The Hand Explorer must translate off-grid bets onto THIS
# grid, or a 1.75-2.0x bet clamps to 'o' (1.5x) and the explorer shows a different
# line than the live bot (which translates against the same menu). See BUG-019.
_POSTFLOP_GRID = translation.postflop_grid_for(BLUEPRINT_MENU_MODE)
# The served bot's version label (e.g. 'v1'/'v2'), stamped on every hand's GLOBAL per-version
# counter so the +EV card's v1/v2 numbers are LIVE (no hand-table scan). Constant per process --
# the SAME derivation recap_version() applies to this process's recaps, so the live counter and the
# recap history agree on which version a hand belongs to.
_BOT_VERSION_LABEL = recap_version({
    # NB: NO ALLIN_GIT_SHA fallback -- the version bucket must be COARSE (v1/v2), or a per-commit SHA
    # would mint unbounded vh_<sha>/vn_<sha> attrs on the global row (400KB-item leak) and mis-bucket
    # the card. Unset ALLIN_BOT_VERSION degrades to the blueprint-name-derived label (v1/v2).
    'botVersion': os.environ.get('ALLIN_BOT_VERSION') or None,
    'blueprint': _BLUEPRINT_PATH.name if _BLUEPRINT_PATH else '',
})
# Sentinel for the degraded-mode bail-out: every strategy/game endpoint short-
# circuits with 503 + this message if the blueprint failed to load at import.
def _blueprint_unavailable_503():
    return jsonify({
        "error": "blueprint not loaded; the deploy is in a degraded state. "
                 "Check /api/healthz for details."
    }), 503
# Phase-4 bot: the river subgame solver. Off the river (or whenever the solve
# inputs are missing) it delegates to the blueprint exactly like
# ConfidenceAwareStrategy; on the river it solves the actual subgame and plays the
# exact size it finds (validated ~24x less river-exploitable than the blueprint).
# A river decision can take up to time_budget seconds (early-stops sooner on easy
# spots); GameSession's advance_bot_turns has a safe-fallback guard so a hand
# never crashes. NOTE: resolve_blueprint_path globs only the TOP-LEVEL
# analysis/blueprints/blueprint_*.db (snapshots/ is not searched), so the served
# default is blueprint_final.db (the 25M snapshot). Pin ALLIN_BLUEPRINT_DB only to
# override that (e.g. a different-sizing blueprint).
# Served live bot. safe_gadget + gadget_anchor=os.environ.get('ALLIN_GADGET_ANCHOR', 'auto') (Phase 5a, "policy B"): on the
# river it exploits the tracked read (unsafe-v1) ONLY when a per-spot safety self-check
# proves that is no-more-exploitable than the blueprint, else it clamps to the
# blueprint-anchored re-solving gadget -- provably ≤ blueprint on the wrong-belief tail.
# Validated to match unsafe-v1 on EV + latency vs a maniac while carrying the guarantee
# (docs/SAFE_RIVER_SOLVING_PLAN.md; tests/test_safe_river_gadget.py,
# tests/compare_gadget_policies.py). To revert to pure max-exploit: drop both kwargs.
# purify_threshold=0.01: 1% strategy purification of the blueprint-path play -- the BR
# sweep (seed 42, 50 samples) found 1% the optimum (off 14621 -> 1% 14534 mbb; 5% and
# full=argmax both WORSE), a small safe win that keeps genuine mixes. EXPLORER_BOT below
# is left UNpurified (it inspects the raw blueprint). See docs/ROADMAP.md (purify track).
# Validate the gadget anchor ONCE -- an invalid env value would otherwise raise deep inside BOT
# construction and crash-loop the worker (no degraded-mode JSON). Unknown -> warn + safe default.
_GADGET_ANCHOR = os.environ.get('ALLIN_GADGET_ANCHOR', 'auto')
if _GADGET_ANCHOR not in ('belief', 'blueprint', 'confidence', 'auto'):
    _LOG.warning("ALLIN_GADGET_ANCHOR=%r invalid; falling back to 'auto'.", _GADGET_ANCHOR)
    _GADGET_ANCHOR = 'auto'
if _GADGET_ANCHOR == 'belief':
    _LOG.warning("ALLIN_GADGET_ANCHOR='belief' GIVES UP the safe-gadget <=blueprint floor "
                 "(read-driven, no wrong-belief guarantee) -- intended for personal/exploit "
                 "testing, NOT public production. Use 'auto' in prod.")
# Confidence gate for the exploit tilt + the all-in guard's belief-trust. Default 0.2 (prod). LOWER
# it (e.g. 0.1, via dev_launch) so exploitation engages on weaker reads for testing -- riskier
# (acts on a less-certain belief at a stack-off), so keep 0.2 in production.
_GUARD_CONFIDENCE = float(os.environ.get('ALLIN_GUARD_CONFIDENCE', '0.2'))
if BLUEPRINT_DB is not None:
    # Served bot is RIVER-ONLY: blueprint everywhere, the river subgame solver on the river
    # (validated, ~+207 mbb real) + the pre-river exploit tilt. Real-time TURN solving is SHELVED
    # (dead end): an H2H over 48k hands lost -66 mbb/hand because the depth-limited turn leaf models
    # the BLUEPRINT river while the bot plays the RIVER SOLVER -- an optimistic, self-inconsistent
    # leaf the turn CFR overfits. Every cheap leaf (blueprint, multi-valued) loses; the only correct
    # fix is a range-conditional CFV net (DeepStack-scale, $$$). The turn-solver code is kept as the
    # recorded dead-end (TurnSubgameSolver / cfv.py); it is NO LONGER WIRED into serving. See
    # docs/private/ROADMAP.md "Dead Ends" + backend/bot/docs/TURN_BAKE_VS_NN_SPEC.md.
    # max_iters=275, time_budget=24s: RUN the river CFR to the full 275 iters (it converges, 1% gap, at
    # ~280-640; 200 was under-converged). The budget is raised 10s -> 24s SO 275 ITERS ACTUALLY RUN on the
    # 0.25 vCPU box (200 iters there ~12s, so 275 ~15-16s -- well under 24s) -- a DELIBERATE latency-for-
    # convergence trade. 24s is the safety cap for pathological deep/high-SPR spots only. UX: the human
    # waits ~15s (up to 24s worst case) for a river decision. (Convergence data: 2026-06-22.)
    BOT = RiverSubgameSolver(BLUEPRINT_DB, max_iters=275, check_every=40,
                             time_budget=24.0, safe_gadget=True, gadget_anchor=_GADGET_ANCHOR,
                             purify_threshold=0.01, guard_confidence=_GUARD_CONFIDENCE)
    # A separate solver for the Strategy Explorer's on-demand river solve. It is
    # NOT latency-critical (an explicit "solve" click, not a live turn), so it
    # gets more iterations for a better-converged answer and a slightly larger
    # time budget; the smaller check_every bounds a deep-stack/small-pot (high-
    # SPR) solve closer to the budget. Shares the same read-only blueprint DB.
    # check_every=10 (not 25): the explorer is UNGATED on SPR (it solves arbitrary user spots, incl.
    # deep-stack/high-SPR ones with a big tree), and the time budget is only enforced at each block
    # boundary -- a coarser check_every let one deep solve overrun ~8s and pin the single explorer
    # permit (starving concurrent solves -> 503). A tighter check honors the 8s budget sooner.
    EXPLORER_BOT = RiverSubgameSolver(BLUEPRINT_DB, max_iters=2000, check_every=10,
                                      time_budget=8.0)
else:
    BOT = EXPLORER_BOT = None
# Opponent model for the hand-level range tracker (Phase 3); injected into every
# GameSession so the bot maintains a belief over the human's hand as it plays.
BOT_RANGE_FN = BOT.range_model_fn() if BOT is not None else None
# The HERO (bot's-own) range model: the bot's ACTUAL served play (deep-jam routed + purified), so the
# subgame solver's hero range matches how the bot really plays -- not the raw blueprint. VILLAIN range
# stays BOT_RANGE_FN (the opponent plays the raw blueprint; doesn't route/purify). C1 keeps hero on the
# blueprint, NOT the exploit model -- this just makes that blueprint range faithful to routing+purify.
HERO_RANGE_FN = BOT.hero_range_model_fn() if BOT is not None else None

# Phase 6 opponent EXPLOITATION (default OFF). When ALLIN_EXPLOIT=1, the range tracker's
# assumed opponent model is swapped from the blueprint ("opponent = GTO") to the fitted
# per-player / population HUMAN model ("opponent = this human"). Still gadget-protected
# (gadget-protected via _GADGET_ANCHOR, default 'auto'), so the served strategy stays <= blueprint.
# HumanModel verifies abstraction consistency with the served blueprint on load and RAISES
# on a mismatch -> we catch it and stay on the blueprint rather than mis-serve.
HUMAN_MODEL = None
if BLUEPRINT_DB is not None and os.environ.get('ALLIN_EXPLOIT', '0') == '1':
    try:
        from bot.src.exploitation.opponent_model import HumanModel
        _OPP_DIR = os.environ.get('ALLIN_OPPONENT_MODEL_DIR') or os.path.join(
            os.path.dirname(os.path.dirname(_BLUEPRINT_PATH)), 'opponent_models')
        HUMAN_MODEL = HumanModel(
            _OPP_DIR, BLUEPRINT_DB,
            alpha_player=float(os.environ.get('ALLIN_EXPLOIT_ALPHA', '10')),
            alpha_pop=float(os.environ.get('ALLIN_EXPLOIT_ALPHA_POP', '160')))
        _LOG.info("opponent exploitation ENABLED: %d player models from %s",
                  len(HUMAN_MODEL.players), _OPP_DIR)
    except Exception as exc:                 # mismatch / missing artifacts -> stay on blueprint
        HUMAN_MODEL = None
        _LOG.error("opponent exploitation requested but failed to load (%s); "
                   "falling back to the blueprint opponent model", exc)


# Live A/B arm for exploitation measurement. 'on' (treatment) plays the exploit; 'off' (control) plays
# the PURE BLUEPRINT even with HUMAN_MODEL loaded. `ALLIN_AB_ARM` forces it ('on'/'off' for dev to play
# each side) or 'random' = a STABLE 50/50 split by player_id (prod A/B). The arm is recorded per hand
# (recap 'exploitArm') so EV(on) vs EV(off) is measurable (scripts/analyze_ab.py). Default 'on' = the
# exploit runs whenever configured (no split); unchanged byte-for-byte when exploitation is off.
_AB_ARM = os.environ.get('ALLIN_AB_ARM', 'on').strip().lower()


def _ab_arm(player_id):
    if _AB_ARM in ('on', 'off'):
        return _AB_ARM
    # 'random': STABLE per-player (hashlib, not hash() -- survives restarts so a player stays one arm).
    h = hashlib.md5(str(player_id).encode()).digest()[0] if player_id else 0
    return 'on' if h % 2 == 0 else 'off'


def _exploit_enabled(player_id):
    """Exploitation runs for this session iff HUMAN_MODEL is loaded AND the A/B arm is 'on'."""
    return HUMAN_MODEL is not None and _ab_arm(player_id) == 'on'


def _range_fn_for(player_id):
    """The range tracker's opponent model for a session: the per-player HUMAN model when exploitation
    is enabled AND the A/B arm is 'on' (population fallback for an unknown player), else the blueprint
    (the default + the A/B CONTROL arm -- byte-for-byte today's behavior when ALLIN_EXPLOIT is off)."""
    if _exploit_enabled(player_id):
        return HUMAN_MODEL.strategy_fn_for(player_id)
    return BOT_RANGE_FN

# Pre-river exploitation tilt risk budget (total-variation cap from the blueprint). Small by
# default -- provably near-blueprint; widen only as the offline scoreboard's measured EV justifies.
_EXPLOIT_DELTA = float(os.environ.get('ALLIN_EXPLOIT_DELTA', '0.05'))


def _exploit_for(player_id):
    """Per-player PRE-RIVER exploitation payload ({chardist, delta}) for the bot, or None when
    exploitation is off. Set on the GameSession so bot_public_state passes it to the solver, which
    tilts trained pre-river nodes toward the opponent's off-GTO fold response (capped at delta)."""
    if not _exploit_enabled(player_id):              # off, or the A/B CONTROL arm -> no tilt
        return None
    return {'chardist': HUMAN_MODEL.chardist_for(player_id), 'delta': _EXPLOIT_DELTA}


# Live last-N (recency): ALLIN_EXPLOIT_RECENT_N>0 refits the opponent model from the player's most
# RECENT N hands at session start, so the read reflects CURRENT play (vs the static fitted profile).
# A hard last-N window IS recency; EB shrinkage handles the smaller sample (thin -> defers to the
# population model). Dev source = the export JSONL; prod would query the allin-hands DynamoDB table
# (same recap shape -- TODO). Cached per player per process (re-fit on restart) so it isn't re-fit
# on every request. OFF by default (=0) -> the static fitted profile, unchanged.
_EXPLOIT_RECENT_N = int(os.environ.get('ALLIN_EXPLOIT_RECENT_N', '0'))
# Shrinkage for the live last-N personal layer. Defaults to alpha_player (no change); LOWER it
# (e.g. 4-5) so the well-sampled COARSE rung moves at small N -- but TUNE it (chronological-holdout
# LL on recent windows), don't guess (EXPLOITATION_PLAN v3).
_EXPLOIT_RECENT_ALPHA = float(os.environ.get('ALLIN_EXPLOIT_RECENT_ALPHA',
                                             os.environ.get('ALLIN_EXPLOIT_ALPHA', '10')))
# Re-fit cadence: re-fit the live last-N window every K completed hands (WITHIN-session adaptation --
# the sliding window). 0 = re-fit only per game (/game/new). The source is the live recap store
# (HANDS: in-memory on dev, DynamoDB in prod -- the SAME code path), so the window actually advances
# as the player keeps playing; e.g. N=100 + REFRESH=50 = a 100-hand window re-fit every 50 hands.
_EXPLOIT_RECENT_REFRESH = int(os.environ.get('ALLIN_EXPLOIT_RECENT_REFRESH', '0'))
_RECENT_MISS = object()                          # sentinel: fit ran, player has NO recent history
_RECENT_LOCK = threading.Lock()                  # guards the caches under gunicorn --threads
_RECENT_CAP = int(os.environ.get('ALLIN_RECENT_CACHE_CAP', '2000'))   # per-process LRU bound
_RECENT_CD = OrderedDict()                       # player_id -> chardist fn | _RECENT_MISS (LRU-capped)
_RECENT_HANDS = OrderedDict()                    # player_id -> hands since last fit (evicted with _RECENT_CD)


def _apply_live_recent(session, player_id, refresh=False):
    """Override the session's opponent model + tilt payload with a fit from the player's RECENT
    hands (live last-N), when enabled. `refresh=True` (at /game/new) RE-FITS the latest window so a
    returning player isn't frozen at first-sighting; else reuses the per-process cache so the
    per-request _load_session path stays O(1). Reuses session.cards (no extra table load); caches a
    MISS sentinel for historyless players so they don't re-query the store every request. Crash-safe;
    only touches strategy_fn + exploit.chardist (hero_strategy_fn / the C1 blueprint hero range is
    never overridden). No-op when off / unknown player / no recent history."""
    if not player_id or not _exploit_enabled(player_id) or _EXPLOIT_RECENT_N <= 0:
        return                                        # off, control arm, or recency disabled
    try:
        cd = None
        if not refresh:
            with _RECENT_LOCK:
                cd = _RECENT_CD.get(player_id)
                if cd is not None:
                    _RECENT_CD.move_to_end(player_id)     # LRU touch
        if cd is None:                            # cold, /game/new refresh, or an every-K invalidation
            recaps = HANDS.list_for_player(player_id, n=_EXPLOIT_RECENT_N)  # live store: dev mem/prod DDB
            cd = (HUMAN_MODEL.chardist_from_recent(recaps, session.cards, alpha=_EXPLOIT_RECENT_ALPHA)
                  if recaps else _RECENT_MISS)
            with _RECENT_LOCK:
                _RECENT_CD[player_id] = cd
                _RECENT_CD.move_to_end(player_id)
                _RECENT_HANDS[player_id] = 0      # reset the hands-since-fit counter on every fit
                while len(_RECENT_CD) > _RECENT_CAP:      # bound per-process memory (LRU evict)
                    old_pid, _ = _RECENT_CD.popitem(last=False)
                    _RECENT_HANDS.pop(old_pid, None)
            if recaps:
                _LOG.info("live last-N: refit %s from %d recent hands",
                          str(player_id)[:8], len(recaps))
        if cd is _RECENT_MISS:
            return                                # no recent history -> keep the static model
        session.strategy_fn = HUMAN_MODEL._strategy_fn(cd)
        if getattr(session, 'exploit', None) is not None:
            session.exploit = {**session.exploit, 'chardist': cd}
    except Exception as exc:                      # never break session creation
        _LOG.warning("live last-N refit failed for %s (%s); using static model", player_id, exc)


def _note_hand_for_recent(player_id):
    """Advance the live last-N refresh cadence on a completed hand: every ALLIN_EXPLOIT_RECENT_REFRESH
    hands, invalidate the player's cached fit so the next session-load re-fits from the now-larger
    recent window (within-session adaptation -- the sliding window). No-op when off / refresh disabled."""
    if (HUMAN_MODEL is None or _EXPLOIT_RECENT_N <= 0
            or _EXPLOIT_RECENT_REFRESH <= 0 or not player_id):
        return
    with _RECENT_LOCK:
        n = _RECENT_HANDS.get(player_id, 0) + 1
        if n >= _EXPLOIT_RECENT_REFRESH:
            _RECENT_CD.pop(player_id, None)       # invalidate -> next _apply_live_recent re-fits
            _RECENT_HANDS[player_id] = 0
        else:
            _RECENT_HANDS[player_id] = n


# SessionStore: chosen by ALLIN_SESSION_STORE (default 'memory'). Set it to
# 'dynamodb' for a multi-worker / multi-box deploy so games are shared and
# survive restarts (see session_store.py / make_session_store).
SESSIONS = make_session_store()
# Leaderboard datastores (the +EV counter + per-player rows). Same backend switch
# as sessions (ALLIN_STORE_BACKEND: memory default, dynamodb in prod).
PLAYERS = make_player_store()
GLOBAL = make_global_stats_store()
# Per-hand recap capture (write-only in v1; the UI / coach / training pipeline
# that consume it are post-launch — but capturing now means launch-window hands
# aren't lost forever). One PutItem per completed hand inside the same hook.
HANDS = make_hand_store()

# Dev warm-start (live last-N): ALLIN_SEED_HANDS_JSONL=<export.jsonl> seeds the IN-MEMORY hand store
# with the player's exported history, so the live last-N fit has real hands to read on dev WITHOUT
# connecting to (or writing into) prod DynamoDB. The JSONL is an export of DynamoDB (scripts/
# export_hands.py) -- re-run that to refresh with the latest. No-op unless set AND the store is
# in-memory (prod DynamoDB already holds the history). Play as your real playerId so the seeded
# hands (keyed by that id) feed your session.
_SEED_HANDS = os.environ.get('ALLIN_SEED_HANDS_JSONL')
if _SEED_HANDS and isinstance(HANDS, InMemoryHandStore):
    try:
        _seeded = 0
        with open(_SEED_HANDS, encoding='utf-8') as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    HANDS.put(json.loads(_line))
                    _seeded += 1
                except Exception:
                    continue
        _LOG.info("seeded %d hands into the in-memory store from %s", _seeded, _SEED_HANDS)
    except OSError as _e:
        _LOG.warning("could not seed hands from %s: %s", _SEED_HANDS, _e)

# River-solve concurrency cap: a single river decision can burn several seconds
# of CFR (BOT.time_budget), so without a bound N simultaneous solves pin every
# core and stall the whole server. This BoundedSemaphore caps how many bot turns
# run a solve at once PER PROCESS (under gunicorn, total = workers x this). Sized
# to leave one core free. Acquired around advance_bot_turns (see _run_bot).
#
# ALLIN_SOLVE_PERMITS overrides the cpu-derived default. On a small container
# (Lightsail Micro: fractional vCPU but cpu_count() reports the host's cores)
# the auto-derived number can be wrong in either direction; the override lets
# ops tune concurrency-vs-latency without a code change. The solver is anytime
# CFR (returns its best answer at the time budget), so 2 concurrent solves on
# one core both finish slower rather than one of them 503ing -- usually the
# better trade for live play.
def _permits_from_env(var, default):
    try:
        v = int(os.environ.get(var, ''))
        return max(1, v)
    except ValueError:
        return default
_SOLVE_PERMITS = _permits_from_env(
    'ALLIN_SOLVE_PERMITS', max(1, (os.cpu_count() or 2) - 1))
_SOLVE_SEMAPHORE = threading.BoundedSemaphore(_SOLVE_PERMITS)
# The explorer's on-demand river solve (rate-limited per IP but unauthenticated,
# up to 8s each) gets a SEPARATE, smaller pool so a burst of
# /api/strategy/river-solve can't hold every permit and starve live gameplay
# (which uses _SOLVE_SEMAPHORE).
_EXPLORER_PERMITS = _permits_from_env(
    'ALLIN_EXPLORER_PERMITS', max(1, _SOLVE_PERMITS // 2))
_EXPLORER_SEMAPHORE = threading.BoundedSemaphore(_EXPLORER_PERMITS)
# How long a request will wait for a solve permit before giving up with 503.
_SOLVE_WAIT_SECONDS = 30.0

# Inactivity auto-fold: an in-hand session with no client progress for longer than
# this is considered abandoned (tab closed / disconnected) and resolved by a
# background sweeper, so it doesn't pin a session open and can't be used to dodge a
# loss on the +EV leaderboard. The deadline is stamped on every in-hand persist
# (regardless of whose turn it is) so a hand abandoned while the BOT is to act --
# tab closed after /action but before /bot-action -- is also reaped. A normal
# client re-stamps it within ms by driving the next step, so only a truly idle hand
# expires. Backwards-compatible: a pre-existing session has no `inactivity_deadline`
# so it's treated as legacy and left to the normal TTL -- never retroactively folded.
_INACTIVITY_DEADLINE_SECONDS = int(os.environ.get('ALLIN_INACTIVITY_DEADLINE_SECONDS', '600'))
_SWEEP_INTERVAL_SECONDS = int(os.environ.get('ALLIN_INACTIVITY_SWEEP_SECONDS', '60'))
_SWEEP_ENABLED = os.environ.get('ALLIN_INACTIVITY_SWEEP', '1') == '1'

# Debug overlay (the per-decision bot trace, `botDebug`) exposes the bot's bucketed
# hand class MID-HAND -- a spoiler. Policy: ON by default (dev experience), the
# PROD Dockerfile sets ALLIN_DEBUG_OVERLAY=0 explicitly so the public deploy is
# off by default. To disable locally: ALLIN_DEBUG_OVERLAY=0. To enable in Docker:
# ALLIN_DEBUG_OVERLAY=1. The frontend Debug toggle gates the UI side too.
_DEBUG_OVERLAY = os.environ.get("ALLIN_DEBUG_OVERLAY", "1") == "1"


def _redact_view(view):
    """Return a public view with the debug overlay removed unless explicitly
    enabled. Applied to every game response so botDebug never ships hot."""
    if not _DEBUG_OVERLAY:
        view.pop('botDebug', None)
    return view


if BLUEPRINT_DB is not None:
    _LOG.info("Loaded blueprint: %s (%s iterations)", _BLUEPRINT_PATH.name,
              BLUEPRINT_DB.get_metadata('total_iterations', 0))

app = Flask(__name__)

# Cap request bodies: every endpoint takes a tiny JSON object, so a few KB is
# plenty. This rejects oversized/garbage bodies (a cheap DoS vector) with 413
# before Flask buffers them. 64 KB is generous headroom.
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024

# CORS origins are env-driven (ALLIN_CORS_ORIGINS, comma-separated) so the same
# code serves localhost in dev and the real domain once deployed. No credentials
# mode: auth is stateless (session id + playerId in the request, no cookies), so
# supports_credentials would be a needless risk amplifier.
_origins_env = os.environ.get("ALLIN_CORS_ORIGINS")
ALLOWED_ORIGINS = ([o.strip() for o in _origins_env.split(",") if o.strip()]
                   if _origins_env
                   else ['http://localhost:5173', 'http://localhost:5174'])
CORS(app, origins=ALLOWED_ORIGINS)


@app.errorhandler(413)
def _too_large(_e):
    return jsonify({"error": "request body too large"}), 413


@app.errorhandler(SessionLockTimeout)
@app.errorhandler(TimeoutError)
def _server_busy(e):
    # Couldn't get a session lock or a river-solve permit in time: shed load.
    # Log with the contended session id (when available via the URL) so a wedged
    # session can be diagnosed instead of just seeing 503s in aggregate.
    sid = request.args.get('id') or '(unknown)'
    _LOG.warning("server busy on %s (session=%s, ip=%s): %s",
                 request.path, sid, request.remote_addr or '?', e)
    return jsonify({"error": "server busy, please retry"}), 503


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        # Echo the Origin only if it's in the allow-list (a wildcard "*"
        # contradicts the restricted allow-list flask-cors applies to the
        # actual response).
        origin = request.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add('Access-Control-Allow-Headers', "Content-Type")
        response.headers.add('Access-Control-Allow-Methods', "GET, POST, OPTIONS")
        return response


@app.before_request
def _degraded_mode_guard():
    """When the blueprint failed to load at import, every endpoint EXCEPT
    /api/healthz returns 503 (instead of 500-ing on a None reference deeper in).
    Lightsail's rolling-deploy probe hits /api/healthz, sees 503, fails fast,
    and ops can read the load error from the healthz JSON. Live users get a
    clean "service unavailable" instead of an opaque crash."""
    if _BLUEPRINT_LOAD_ERROR is None:
        return None
    if request.method == "OPTIONS":
        return None                  # let preflight pass; the next request 503s
    if request.path in ("/api/healthz", "/api/test"):
        return None
    return _blueprint_unavailable_503()


@app.after_request
def _security_headers(resp):
    """Defense-in-depth response headers. Cheap, harmless on a JSON API. HSTS is
    also added by the TLS terminator (Lightsail/Cloudflare) but a second copy is
    fine. Specifically NOT adding CSP — JSON responses don't execute, and a CSP
    on the API would only constrain a misconfigured browser anyway."""
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    resp.headers.setdefault('Cross-Origin-Resource-Policy', 'cross-origin')
    resp.headers.setdefault('Strict-Transport-Security',
                            'max-age=31536000; includeSubDomains')
    return resp


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
    '2': 'overbet2 (2.0x pot, capped menu)',
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


def _postflop_pattern(actions, session=None):
    """Pattern + trailing off-grid blend for a postflop betting line. A bet/raise
    may carry a `fraction` (pot fraction); it's translated onto the postflop grid
    (⅓ / ⅔ / pot). Returns (pattern, trailing_blend, error_or_None).

    If `session` is provided, each off-grid bet is translated against the
    NODE-specific grid (session._node_grid) — the same grid the live bot uses,
    which excludes any size whose chips would exceed the actor's remaining
    stack. Each action is then APPLIED to the session so the next iteration
    sees the live pot/stack state. Without a session the static _POSTFLOP_GRID
    is used (correct only for 100-BB-deep current-street-fresh assumptions)."""
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
                grid = (session._node_grid(session.legal_actions(), include_allin=False)
                        if session is not None else _POSTFLOP_GRID)
                tb = translation.translate_bet(frac, grid)
                char = translation.nearest_char(frac, grid)
                if len(tb) > 1:
                    blend = tb
            else:
                char = _SIZE_CHAR.get(a.get('size', 'medium'), 'm')
        if char is None:
            continue   # fold / unknown: never a valid queryable pattern char
        pattern += char
        if idx == last:
            trailing_blend = blend
        # Walk the session forward so the next action's node grid is correct.
        if session is not None and act in ('check', 'call', 'fold', 'allin', 'bet', 'raise'):
            try:
                session.apply_action(_engine_action_for_replay(a, session))
            except (GameError, ValueError, TypeError) as e:
                return None, None, f"could not replay action {idx + 1}: {e}"
    return pattern, trailing_blend, None


def _preflop_grid(num_aggr, committed_actor, to_call, pot):
    """Trained preflop bet-size grid at a node, as sorted [(char, eff_frac), ...].
    Thin wrapper over the shared translation.preflop_grid (the one definition the
    LBR victim model also uses) so the explorer and the harness can't drift."""
    return translation.preflop_grid(
        num_aggr, committed_actor, to_call, pot,
        preflop_open_chips(), PREFLOP_RAISE_MULT, _SIZE_CHAR)


def _preflop_pattern(actions, session=None):
    """Pattern + trailing off-grid blend for a preflop line. Preflop sizes are
    absolute (a BB ladder, or pot-relative 3-bets), not plain pot fractions, so a
    custom raise carries a `bb` raise-TO total.

    Two paths, gated on `session`:
      * session is None — standalone replay of committed chips (SB=1, BB=2, both
        100 BB deep) to recover each raise's pot fraction, translated against the
        SHARED grid (translation.preflop_grid). Backwards-compatible default.
      * session given — translation uses the session's NODE grid
        (session._node_grid), and each action is applied to the session so the
        next iteration's grid reflects the actor's real remaining stack. This is
        what the live bot does, so the explorer matches the bot in deep/short
        spots where a size on the static grid would be illegal at the node.
    Returns (pattern, trailing_blend, err)."""
    pattern, trailing_blend, last = '', None, len(actions) - 1
    # Standalone (no session) bookkeeping. Unused when session is given.
    committed = [float(_SB_CHIPS), float(_BB_CHIPS)]
    actor, num_aggr = 0, 0
    for idx, a in enumerate(actions):
        act = a.get('action')
        if session is not None:
            # Session is authoritative: read state from it instead of the
            # standalone bookkeeping. (Both kept in sync for the session=None
            # fall-through inside the bet/raise branch.)
            legal = session.legal_actions()
            actor = session.current_player()
            other = 1 - actor
            committed_actor = session.data['p0_invested' if actor == 0 else 'p1_invested']
            to_call = session._action_cost('call') if 'call' in legal else 0.0
            pot = session._current_pot()
        else:
            other = 1 - actor
            to_call = max(0.0, committed[other] - committed[actor])
            pot = committed[0] + committed[1]
            committed_actor = committed[actor]
        char, blend = None, None
        if act == 'check':
            char = 'k'
        elif act == 'call':
            char = 'c'
            if session is None:
                committed[actor] = committed[other]
        elif act == 'allin':
            char = 'a'
            if session is None:
                committed[actor] = float(STARTING_STACK)
                num_aggr += 1
        elif act in ('bet', 'raise'):
            bb = a.get('bb')
            if bb is not None:
                try:
                    total = float(bb) * BIG_BLIND
                except (TypeError, ValueError):
                    return None, None, "raise-to (bb) must be a number"
                if not math.isfinite(total) or total <= committed_actor:
                    return None, None, "raise-to must exceed the current bet"
            elif (session is None and num_aggr == 0) or \
                 (session is not None and not (to_call > 0)):
                # First-in open: BB ladder.
                total = preflop_open_chips().get(
                    a.get('size', 'medium'), preflop_open_chips()['medium'])
            else:
                # 3-bet / 4-bet+: pot-relative.
                mult = PREFLOP_RAISE_MULT.get(a.get('size', 'medium'), 1.0)
                total = committed_actor + to_call + mult * (pot + to_call)
            if session is not None:
                grid = session._node_grid(legal, include_allin=False)
            else:
                grid = _preflop_grid(num_aggr, committed_actor, to_call, pot)
            eff = translation.eff_fraction(total - committed_actor, to_call, pot)
            tb = translation.translate_bet(eff, grid)
            char = translation.nearest_char(eff, grid)
            if len(tb) > 1:
                blend = tb
            if session is None:
                committed[actor] = total
                num_aggr += 1
        if char is None:
            continue
        pattern += char
        if idx == last:
            trailing_blend = blend
        # Apply the action so the next iteration's grid + state is correct.
        if session is not None and act in ('check', 'call', 'fold', 'allin', 'bet', 'raise'):
            try:
                session.apply_action(_engine_action_for_replay(a, session))
            except (GameError, ValueError, TypeError) as e:
                return None, None, f"could not replay action {idx + 1}: {e}"
        else:
            actor = other
    return pattern, trailing_blend, None


@app.route('/api/strategy/from-hand', methods=['POST'])
def strategy_from_hand():
    """
    Hand Explorer: take real cards + a betting line, derive the info-set key,
    and return the blueprint strategy. One round-trip.
    """
    data = request.get_json(silent=True) or {}
    hole = [c for c in data.get('holeCards', []) if c]
    community_in = [c for c in data.get('communityCards', []) if c]
    actions = data.get('actions', [])
    history = data.get('history') or {}
    position = data.get('position', 'ip')

    if position not in ('ip', 'oop'):
        return jsonify({"error": "position must be 'ip' or 'oop'"}), 400
    if not isinstance(actions, list):
        return jsonify({"error": "'actions' must be a list"}), 400
    if not isinstance(history, dict) or any(not isinstance(v, list)
                                            for v in history.values()):
        return jsonify({"error": "each 'history' street must be a list"}), 400
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
    from bot.src.cfr.keys import make_info_set_key, STREET_NAMES, street_from_board_count
    adapter = GameAdapter()

    # community length was validated to 0/3/4/5 above, so this never raises.
    street_idx = street_from_board_count(len(community_e))
    street = STREET_NAMES[street_idx]
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
    #
    # If sufficient `history` was sent, build a session and use NODE-specific
    # grids (matching the live bot exactly, even in deep/short-stack and capped
    # menu spots where the static grid contains illegal sizes). If history is
    # missing/incomplete, fall back to the static grid (preserves prior UX for
    # quick lookups without prior-street setup).
    lookup_session = None
    try:
        lookup_session = _try_build_lookup_session(
            hole_e, community_e, position, history, street_idx)
    except Exception:
        _LOG.warning("failed to build lookup session; falling back to static grid",
                     exc_info=True)
        lookup_session = None
    pattern, trailing_blend, err = (
        _preflop_pattern(actions, session=lookup_session) if is_preflop
        else _postflop_pattern(actions, session=lookup_session))
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


# =============================================================================
# River subgame solver (on-demand, for the Strategy Explorer)
# =============================================================================

def _river_action_label(a, node, bot_seat):
    """Friendly label (in BB) for a river-tree action token."""
    if a in ('check', 'call', 'fold', 'allin'):
        return a.capitalize() if a != 'allin' else 'All-in'
    if is_sized(a):
        add = sized_chips(a)
        if a.startswith('raise:'):
            total = node.sc[bot_seat] + add            # raise-TO street total
            return f"Raise to {total / BIG_BLIND:g} BB"
        return f"Bet {add / BIG_BLIND:g} BB"
    return a


def _engine_action_for_replay(a, session):
    """Convert one explorer action into an engine action string for
    GameSession.apply_action, using the session's live pot / to_call.

    Three forms accepted:
      * sizeless: {action:'check'|'call'|'fold'|'allin'} -> returned as-is.
      * named: {action:'bet'|'raise', size:'small'|...} -> returned as
        'bet_<size>' / 'raise_<size>', applied by the engine like the live bot's
        own named action.
      * custom: {action:'bet'|'raise', fraction|bb} -> converted to a custom
        raise-to total ('bet_custom_<chips>' / 'raise_custom_<chips>'). The
        engine then snaps it onto the trained grid via pseudo-harmonic
        translation, exactly like the live bot's pending_translation path.
    Raises ValueError on a malformed size or unsupported action."""
    act = a.get('action')
    if act in ('check', 'call', 'fold', 'allin'):
        return act
    if act not in ('bet', 'raise'):
        raise ValueError(f"unsupported action {act!r}")
    facing = 'call' in session.legal_actions()         # a raise iff there's a bet to call
    bb = a.get('bb')
    frac_in = a.get('fraction')
    if bb is None and frac_in is None:
        # Named size — apply directly as 'bet_<size>' / 'raise_<size>'.
        size = a.get('size', 'medium')
        return f"{act}_{size}"
    if bb is not None:                                  # preflop: absolute BB raise-to total
        total = float(bb) * BIG_BLIND
    else:                                               # postflop: pot-fraction sized
        try:
            frac = float(frac_in)
        except (TypeError, ValueError):
            raise ValueError("bet/raise fraction must be a number")
        if not math.isfinite(frac) or frac <= 0:
            raise ValueError("bet/raise fraction must be positive and finite")
        pot = session._current_pot()
        to_call = session._action_cost('call') if facing else 0.0
        total = frac * (pot + to_call) + to_call if facing else frac * pot
    return make_custom_action(facing, round(float(total), 2))


def _explorer_session(hole_e, board_e, position):
    """A GameSession rigged for the explorer: the bot seat (= the user's position)
    holds the user's hole cards, the opponent holds throwaway (unused) cards, and
    the board is the user's five cards. Range tracking is on (BOT_RANGE_FN), so
    replaying the line builds both blueprint-projected ranges exactly as live play
    does. Returns (session, bot_seat)."""
    session = GameSession.new('explorer', 'explorer', strategy_fn=BOT_RANGE_FN,
                              menu_mode=BLUEPRINT_MENU_MODE)
    d = session.data
    bot_seat = 0 if position == 'ip' else 1            # ip = SB/button = seat 0
    used = set(hole_e) | set(board_e)
    dummy = [c for c in _FULL_DECK if c not in used][:2]   # opponent's (unused) cards
    d['human_seat'] = 1 - bot_seat                      # the opponent is the "human"
    d['p0_cards'], d['p1_cards'] = ((hole_e, dummy) if bot_seat == 0
                                    else (dummy, hole_e))
    d['community'] = list(board_e)
    # Reset both trackers to the rigged bot cards (new() seeded them off a random deal).
    d['opp_range'] = RangeTracker(hole_e, session.cards).to_dict()
    d['bot_range'] = RangeTracker((), session.cards).to_dict()
    return session, bot_seat


def _replay_history_up_to(session, history, target_street):
    """Replay history streets 0..target_street-1 through the session. Each
    prior street's betting must close. After success the session sits at
    `target_street` with empty current-street betting. Returns None on
    success or an error string (the caller falls back to the static-grid
    path on any error — history insufficient or inconsistent)."""
    streets = [('preflop', 0), ('flop', 1), ('turn', 2)]
    for name, idx in streets:
        if idx >= target_street:
            break
        line = history.get(name) or []
        if not line:
            return f"history.{name} is empty"
        for a in line:
            if session.data['status'] != 'in_hand':
                return f"the betting line ends the hand before {name} closes"
            if session.data['street'] != idx:
                return f"actions out of order at {name}"
            try:
                session.apply_action(_engine_action_for_replay(a, session))
            except (GameError, ValueError, TypeError) as e:
                return f"illegal action in {name}: {e}"
        if session.data['status'] == 'in_hand' \
                and session.data['street'] == idx:
            return f"{name} betting did not close"
    if session.data['status'] != 'in_hand' \
            or session.data['street'] != target_street:
        return "history did not advance to the requested street"
    return None


def _try_build_lookup_session(hole_e, board_e, position, history, street_idx):
    """Build an explorer session positioned at `street_idx`, with `history`
    replayed through prior streets. Returns the session, or None if the
    history is insufficient (caller falls back to the static-grid lookup).
    Preflop (street_idx == 0) always succeeds — no prior streets needed."""
    if street_idx == 0:
        sess, _ = _explorer_session(hole_e, board_e, position)
        return sess
    sess, _ = _explorer_session(hole_e, board_e, position)
    err = _replay_history_up_to(sess, history, street_idx)
    return None if err else sess


def _replay_history(session, history, river_actions):
    """Replay the scripted pre-river streets + the river line (before the bot's
    decision) through the session, in action order. Each pre-river street's betting
    must CLOSE before the next begins (the engine advances streets automatically).
    Returns an error string, or None on success (the session then sits at a river
    decision)."""
    streets = [('preflop', 0, history.get('preflop', [])),
               ('flop', 1, history.get('flop', [])),
               ('turn', 2, history.get('turn', [])),
               ('river', 3, river_actions or [])]
    for name, idx, line in streets:
        for a in line:
            if session.data['status'] != 'in_hand':
                return f"the betting line ends the hand before the river (in the {name})"
            if session.data['street'] != idx:
                return (f"too many actions before the {name} — each street's betting "
                        "must complete before the next")
            try:
                session.apply_action(_engine_action_for_replay(a, session))
            except GameError as e:
                return f"illegal action in the {name} line: {e}"
            except (ValueError, TypeError) as e:
                return f"bad {name} action: {e}"
        if name != 'river' and session.data['status'] == 'in_hand' \
                and session.data['street'] == idx:
            return f"the {name} betting is incomplete (it never closed)"
    if session.data['status'] != 'in_hand' or session.data['street'] != 3:
        return "the line does not reach a river decision"
    return None


@app.route('/api/strategy/river-solve', methods=['POST'])
def strategy_river_solve():
    """Run the river subgame solver for a concrete spot and return its SOLVED
    strategy -- UNGATED (no SPR skip, no EV gate, unlike live play).

    The previous-streets betting (`history`: preflop/flop/turn) plus the river line
    (`actions`, before the bot's decision) are REPLAYED through a GameSession, so
    both players' ranges are the blueprint-projected beliefs the live bot would
    hold at river entry -- not uniform. The river-entry pot, behind stacks, and
    realized river path all come from that replay. Bets are full-custom-sized and
    snapped onto the trained grid exactly as the live bot does.
    """
    # Per-IP floor: each solve burns up to ~8s of CPU on a small box, making
    # this the cheapest CPU-DoS pivot on the API. The Cloudflare edge rule
    # (5/10s) is the primary fence; this floor holds if CF is bypassed via the
    # raw Lightsail URL. A human exploring spots clicks a few times a minute.
    if _rate_limited('river_solve', _client_ip(), limit=10, window_seconds=60):
        return jsonify({"error": "too many solver requests — slow down"}), 429
    data = request.get_json(silent=True) or {}
    hole = [c for c in data.get('holeCards', []) if c]
    community_in = [c for c in data.get('communityCards', []) if c]
    river_actions = data.get('actions', [])
    history = data.get('history') or {}
    position = data.get('position', 'ip')

    if position not in ('ip', 'oop'):
        return jsonify({"error": "position must be 'ip' or 'oop'"}), 400
    if not isinstance(river_actions, list) or not isinstance(history, dict) \
            or any(not isinstance(v, list) for v in history.values()):
        return jsonify({"error": "'actions' and each 'history' street must be lists"}), 400
    if len(hole) != 2:
        return jsonify({"error": "exactly two hole cards required"}), 400
    if len(community_in) != 5:
        return jsonify({"error": "the river solver needs all five community cards"}), 400

    try:
        hole_e = [to_engine(c) for c in hole]
        community_e = [to_engine(c) for c in community_in]
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    all_cards = hole_e + community_e
    if len(set(all_cards)) != len(all_cards):
        return jsonify({"error": "duplicate cards in hand/board"}), 400

    session, bot_seat = _explorer_session(hole_e, community_e, position)
    err = _replay_history(session, history, river_actions)
    if err:
        return jsonify({"error": err}), 400
    if session.current_player() != bot_seat:
        return jsonify({"error": "the betting line must end on your turn on the river "
                                 "(it currently lands on the opponent)"}), 400

    ps = session.bot_public_state()
    if ps.get('riverEntryPot') is None or ps.get('hero_range') is None:
        return jsonify({"error": "could not assemble the river solve inputs"}), 400

    if not _EXPLORER_SEMAPHORE.acquire(timeout=_SOLVE_WAIT_SECONDS):
        raise TimeoutError("server busy: no solver permit available")
    try:
        dist, node, info = EXPLORER_BOT.solve_for_action(
            board=ps['community'], pot_entry=ps['riverEntryPot'],
            stacks=ps['riverEntryStacks'], bot_seat=ps['botSeat'],
            hole=ps['hole_cards'], villain_tracker=ps['opp_range'],
            hero_tracker=ps['hero_range'],
            confidence=getattr(ps['opp_range'], 'confidence', 1.0),
            river_path=ps['riverPath'])
    except ValueError as e:
        # A spot the solver can't represent (hole collides with the board, the
        # bot's hand has ~zero blueprint reach for this line, ...) -> clean 400.
        return jsonify({"error": f"river solve not applicable: {e}"}), 400
    finally:
        _EXPLORER_SEMAPHORE.release()

    strategy = {}
    for a, p in dist.items():
        lbl = _river_action_label(a, node, bot_seat)
        strategy[lbl] = strategy.get(lbl, 0.0) + float(p)

    villain = ps['opp_range']
    return jsonify({
        "solver": {
            "strategy": strategy,
            "converged": bool(info.get('converged', False)),
            "iters": int(info.get('iters', 0)),
            "gap": (round(float(info['gap']), 4) if info.get('gap') is not None else None),
            "potEntryBb": ps['riverEntryPot'] / BIG_BLIND,
            "effectiveStackBb": ps['riverEntryStacks'][0] / BIG_BLIND,
            "confidence": round(float(getattr(villain, 'confidence', 1.0)), 3),
        },
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

def _load_session(session_id, claimed_player_id):
    """Return a GameSession or (None, error_response).

    Ownership check (C3): the session id alone is an unguessable bearer
    capability, but we additionally require the caller to present the playerId
    the session was created with. A mismatch (or a missing playerId) is rejected
    so a leaked/guessed session id can't be read or acted on by someone else.
    """
    if not session_id:
        return None, (jsonify({"error": "session id required"}), 400)
    data = SESSIONS.get(session_id)
    if data is None:
        return None, (jsonify({"error": "session not found or expired"}), 404)
    if not claimed_player_id or claimed_player_id != data.get('player_id'):
        # 404 (not 403) so we don't confirm a session id exists to a non-owner.
        return None, (jsonify({"error": "session not found or expired"}), 404)
    sess = GameSession.from_dict(data, strategy_fn=_range_fn_for(data.get('player_id')),
                                 menu_mode=BLUEPRINT_MENU_MODE, hero_strategy_fn=HERO_RANGE_FN)
    sess.exploit = _exploit_for(data.get('player_id'))
    _apply_live_recent(sess, data.get('player_id'))
    return sess, None


def _run_bot(session, **kwargs):
    """Advance the bot's turn(s) under the river-solve concurrency cap (H2).

    advance_bot_turns may run a multi-second river solve; the BoundedSemaphore
    limits how many run at once so concurrent requests can't pin every core.
    Raises SessionLockTimeout-style 503 handling is done by the caller; here we
    surface a permit timeout as a RuntimeError the caller maps to 503."""
    if not _SOLVE_SEMAPHORE.acquire(timeout=_SOLVE_WAIT_SECONDS):
        raise TimeoutError("server busy: no solver permit available")
    try:
        advance_bot_turns(session, BOT, **kwargs)
    finally:
        _SOLVE_SEMAPHORE.release()


def _record_hand_end(session, pre_status, pre_net):
    """If the hand just transitioned in_hand -> hand_over, record ONE result to the
    leaderboard stores. Idempotent: it fires only on the transition (a retried
    request on an already-finished hand has pre_status == 'hand_over'), and the
    session lock is held, so each hand is counted exactly once. Never breaks a hand
    -- a store failure is logged, not raised."""
    if pre_status == 'hand_over' or session.data.get('status') != 'hand_over':
        return
    if session.data.get('result_recorded'):
        return                                  # persisted idempotency anchor (below)
    player_id = session.data.get('player_id')
    if not player_id:
        return
    delta_bb = (session.data.get('human_net', 0.0) - pre_net) / BIG_BLIND
    if not math.isfinite(delta_bb):
        # A NaN/inf would poison netBB and the leaderboard sort; never record it.
        _LOG.warning("non-finite hand delta (%r); skipping leaderboard update", delta_bb)
        return
    # Persist the "recorded" flag with the session BEFORE writing the stats, so a
    # crash between the two can't double-count on retry (a lost stat is acceptable;
    # a double-count is not). The endpoint's own put() afterward is idempotent.
    session.data['result_recorded'] = True
    # Pin the recap timestamp ONCE so a retried hook (e.g. throttled DynamoDB +
    # adaptive retry path) produces the IDENTICAL handKey instead of a fresh ms
    # epoch each call → duplicate rows in `allin-hands`. The session-stored
    # timestamp survives any in-process retry path.
    if 'recap_ts_ms' not in session.data:
        from bot.src.game.hand_store import _now_ms
        session.data['recap_ts_ms'] = _now_ms()
    sid = session.data['session_id']
    try:
        SESSIONS.put(sid, session.to_dict())
    except Exception:
        session.data['result_recorded'] = False
        session.data.pop('recap_ts_ms', None)
        _LOG.warning("could not persist hand-end anchor for session %s; deferring",
                     sid, exc_info=True)
        return
    # Split the leaderboard updates into separate try/excepts so a per-store failure (e.g. throttle
    # on one table) doesn't strand the other. NOTE: unlike the recap (keyed on handKey, an idempotent
    # overwrite), the PlayerStore/GlobalStats counters are blind ADD increments with NO per-hand dedup
    # key. They are safe from double-counting ONLY because the result_recorded anchor is persisted
    # under the session lock BEFORE these writes, so _record_hand_end runs at most once per hand. Do
    # NOT move these above the anchor put, or weaken the lock, without first keying them by handKey.
    try:
        PLAYERS.record_hand_result(player_id, delta_bb)
    except Exception:
        _LOG.warning("PLAYERS.record_hand_result failed for player=%s session=%s",
                     player_id, sid, exc_info=True)
    try:
        GLOBAL.record_hand_result(delta_bb, is_new_player=False, version=_BOT_VERSION_LABEL)
    except Exception:
        _LOG.warning("GLOBAL.record_hand_result failed for player=%s session=%s",
                     player_id, sid, exc_info=True)
    # Capture the recap (write-only; no v1 consumer). Same per-call try/except
    # discipline + a pinned ts so retries don't pile up duplicate rows.
    try:
        HANDS.put(recap_from_session(
            session, blueprint_name=_BLUEPRINT_PATH.name,
            ts_ms=session.data['recap_ts_ms']))
    except Exception:
        _LOG.warning("hand-recap capture failed for player=%s session=%s",
                     player_id, sid, exc_info=True)
    _note_hand_for_recent(player_id)              # live last-N: advance the every-K re-fit cadence


def _persist(session):
    """Persist a session, (re)stamping the inactivity deadline. The deadline is live
    for the whole duration a hand is in progress (either turn), so a hand abandoned
    while the bot is to act is reaped too; it's cleared once the hand is over. Use
    this in place of a bare SESSIONS.put for any game-state write."""
    d = session.data
    if d.get('status') == 'in_hand':
        d['inactivity_deadline'] = time.time() + _INACTIVITY_DEADLINE_SECONDS
    else:
        d.pop('inactivity_deadline', None)
    SESSIONS.put(d['session_id'], session.to_dict())


def _resolve_abandoned(session):
    """Resolve an abandoned in-hand session. If the bot is to act (e.g. the tab
    closed after /action but before /bot-action ran), advance the bot first; then,
    if control is back with the human and the hand is still live, FOLD (the agreed
    abandonment policy). Either branch reaches a terminal hand and records the
    result. Fold is always legal for the human -- facing a bet, or via the live
    free-fold when a check was available."""
    pre_status = session.data['status']
    pre_net = session.data.get('human_net', 0.0)
    if not session.is_human_turn():
        _run_bot(session)                        # advance the pending bot turn(s)
    if session.data['status'] == 'in_hand' and session.is_human_turn():
        session.apply_action('fold')
    _record_hand_end(session, pre_status, pre_net)


def _sweep_once(max_resolved=20):
    """One inactivity-sweep pass: resolve in-hand sessions whose deadline has
    passed, up to `max_resolved` per pass. Idempotent and lock-guarded so
    concurrent sweepers (one per gunicorn worker) and a racing request can't
    double-resolve a hand.

    The per-pass cap bounds worst-case sweep duration: resolving an abandoned
    session can run a bot turn (a river solve, seconds of CPU, competing for
    the same solve permit as live traffic) UNDER that session's lock. Without
    a cap, a pile of abandoned sessions (e.g. after a burst of visitors) makes
    one pass run for minutes; the leftovers just resolve on later passes
    (interval is _SWEEP_INTERVAL_SECONDS)."""
    now = time.time()
    resolved = 0
    for sid, data in SESSIONS.iter_active():
        if resolved >= max_resolved:
            break
        # Cheap pre-filter on the snapshot before taking the lock.
        dl = data.get('inactivity_deadline')
        if dl is None or now <= dl or data.get('status') != 'in_hand':
            continue
        try:
            with SESSIONS.lock(sid):
                fresh = SESSIONS.get(sid)
                if fresh is None:
                    continue
                dl = fresh.get('inactivity_deadline')
                if (dl is None or time.time() <= dl
                        or fresh.get('status') != 'in_hand'):
                    continue                     # changed since the snapshot
                session = GameSession.from_dict(
                    fresh, strategy_fn=_range_fn_for(fresh.get('player_id')),
                    menu_mode=BLUEPRINT_MENU_MODE, hero_strategy_fn=HERO_RANGE_FN)
                session.exploit = _exploit_for(fresh.get('player_id'))
                _apply_live_recent(session, fresh.get('player_id'))
                _resolve_abandoned(session)
                _persist(session)
                resolved += 1
                _LOG.info("auto-resolved abandoned hand (session %s)", sid)
        except (GameError, SessionLockTimeout, TimeoutError):
            _LOG.warning("inactivity sweep could not resolve session %s", sid,
                         exc_info=True)
        except Exception:
            _LOG.warning("inactivity sweep error on session %s", sid, exc_info=True)


_sweeper_started = False
_sweeper_guard = threading.Lock()


def _start_sweeper():
    """Start the daemon inactivity sweeper once per process (idempotent). Each
    gunicorn worker runs its own; the per-session lock keeps them from clashing."""
    global _sweeper_started
    if not _SWEEP_ENABLED:
        return
    if 'pytest' in sys.modules:
        return                                   # tests drive _sweep_once() directly
    with _sweeper_guard:
        if _sweeper_started:
            return
        _sweeper_started = True

    def _loop():
        while True:
            time.sleep(_SWEEP_INTERVAL_SECONDS)
            try:
                _sweep_once()
            except Exception:
                _LOG.warning("inactivity sweep pass failed", exc_info=True)

    threading.Thread(target=_loop, name="inactivity-sweeper", daemon=True).start()
    _LOG.info("inactivity sweeper started (deadline=%ss interval=%ss)",
              _INACTIVITY_DEADLINE_SECONDS, _SWEEP_INTERVAL_SECONDS)


import re as _re

# playerId is opaque to the backend: a client-generated identifier that becomes
# a DynamoDB key and the subject of ownership checks. We don't require strict
# UUID shape (test fixtures use simple ids like "alice"; the frontend api.js
# generates v4 UUIDs in production — both are fine). What we DO enforce is a
# length cap (defends against a 64KB-body attack stuffing a giant key) and a
# safe character set (defends against injection-shaped values and accidental
# whitespace/control bytes).
_PLAYER_ID_RE = _re.compile(r'^[A-Za-z0-9_-]{1,64}$')


def _valid_player_id(pid):
    return isinstance(pid, str) and bool(_PLAYER_ID_RE.match(pid))


# Lightweight in-process rate limit. NOT a substitute for the Cloudflare edge
# limits the deploy doc relies on; this just prevents a single client from
# accidentally hammering an endpoint (e.g. a misbehaving fetch loop) and a
# floor for the "no Cloudflare in front yet" early-launch window.
_RATE_LIMITS = {}                         # (route, key) -> [count, window_start_epoch]
_RATE_LIMITS_LOCK = threading.Lock()
# Prune lapsed windows once the dict grows past this, so a stream of fresh
# keys (one per rotating UUID / spoofed IP) can't grow the dict unboundedly
# (~120 bytes per entry, forever, per worker). The prune is O(n) but amortized:
# it only runs when the size threshold is crossed, and removes every entry
# whose window has lapsed (those can never influence a future decision).
_RATE_LIMITS_MAX = 10_000


def _rate_limited(route, key, *, limit, window_seconds):
    """True if (route, key) has burst past `limit` within `window_seconds`."""
    now = time.time()
    with _RATE_LIMITS_LOCK:
        entry = _RATE_LIMITS.get((route, key))
        if entry is None or now - entry[1] >= window_seconds:
            if len(_RATE_LIMITS) >= _RATE_LIMITS_MAX:
                # Window starts are at most `window_seconds` old for any entry
                # still in force; anything older is dead weight. 2x slack so a
                # caller-specific longer window (none today) stays safe.
                cutoff = now - 2 * max(window_seconds, 60)
                for k in [k for k, v in _RATE_LIMITS.items() if v[1] < cutoff]:
                    del _RATE_LIMITS[k]
            _RATE_LIMITS[(route, key)] = [1, now]
            return False
        entry[0] += 1
        return entry[0] > limit


def _client_ip():
    """The originating client IP, used to key the per-IP rate limits.

    X-Forwarded-For is CLIENT-CONTROLLED unless the request provably came through a
    trusted proxy: on the raw Lightsail origin a caller can spoof the leftmost XFF
    entry and mint a fresh rate-limit bucket per request (defeating the CPU-DoS /
    row-spam fences). So XFF is trusted ONLY when ALLIN_TRUST_XFF=1 -- set that in
    prod *after* locking the origin to Cloudflare (security group / CF-injected
    shared-secret header) so XFF can't reach the container except via the edge.
    Default (unset) keys on remote_addr, which is unspoofable but coarse (collapses
    all Cloudflare-fronted traffic to the edge IP -- acceptable as a floor; the
    Cloudflare WAF rate rule is the real per-client limiter)."""
    if os.environ.get('ALLIN_TRUST_XFF') == '1':
        xff = request.headers.get('X-Forwarded-For') or ''
        if xff:
            return xff.split(',')[0].strip() or (request.remote_addr or 'unknown')
    return request.remote_addr or 'unknown'


def _hand_cap_response(player_id):
    """A 429 (+ Retry-After) response if the player is over the rolling hand cap,
    else None. Used to gate starting a NEW hand (game/new, next-hand); an in-flight
    hand is never interrupted."""
    try:
        allowed, retry = PLAYERS.hand_cap_status(player_id)
    except Exception:
        _LOG.warning("hand-cap check failed; allowing", exc_info=True)
        return None
    if allowed:
        return None
    resp = jsonify({"error": "hand limit reached — take a break and come back later",
                    "retryAfter": int(retry)})
    resp.status_code = 429
    resp.headers['Retry-After'] = str(int(retry))
    return resp


@app.route('/api/game/new', methods=['POST'])
def game_new():
    """Start a new game session and deal the first hand."""
    # Per-IP floor (belt-and-suspenders to the Cloudflare edge rule, and the
    # only fence if the raw Lightsail URL is hit directly): a rotating-UUID
    # loop would otherwise mint a fresh player row + bump totalPlayers on the
    # public /api/stats ticker every call. A real player starts a handful of
    # sessions per hour at most.
    if _rate_limited('game_new', _client_ip(), limit=20, window_seconds=60):
        return jsonify({"error": "too many new games — slow down"}), 429
    data = request.get_json(silent=True) or {}
    # ALLIN_DEV_FORCE_PLAYER (dev only) forces every new session to a chosen playerId, so you can play
    # AS a profiled player (e.g. 'ron') and watch the bot exploit their model. Ignored in prod (unset).
    raw_pid = os.environ.get('ALLIN_DEV_FORCE_PLAYER') or data.get('playerId')
    # Mint a fresh UUID if the client didn't send one OR sent garbage. Don't
    # blindly trust a non-UUID-shaped string into a DynamoDB key.
    if raw_pid and _valid_player_id(raw_pid):
        player_id = raw_pid
    else:
        player_id = str(uuid.uuid4())

    # Rolling hand-cap: refuse a new hand once the player is over quota (429).
    capped = _hand_cap_response(player_id)
    if capped is not None:
        return capped

    # Count a newly-seen player exactly once (create_if_absent is atomic).
    try:
        if PLAYERS.create_if_absent(player_id):
            GLOBAL.record_new_player()
    except Exception:
        _LOG.warning("new-player counting failed", exc_info=True)

    session_id = str(uuid.uuid4())
    # No contention on a freshly-minted id, but lock for uniformity.
    with SESSIONS.lock(session_id):
        session = GameSession.new(session_id, player_id,
                                  strategy_fn=_range_fn_for(player_id),
                                  menu_mode=BLUEPRINT_MENU_MODE, hero_strategy_fn=HERO_RANGE_FN)
        session.exploit = _exploit_for(player_id)
        session.data['ab_arm'] = _ab_arm(player_id)            # recorded per hand for the EV A/B
        _apply_live_recent(session, player_id, refresh=True)   # new game -> re-fit the latest window
        pre_status, pre_net = session.data['status'], session.data.get('human_net', 0.0)
        _run_bot(session)                    # bot may act first (when it is SB)
        _record_hand_end(session, pre_status, pre_net)   # rare: bot folds hand 1 outright
        _persist(session)
        view = _redact_view(session.public_view())
    view['playerId'] = player_id
    return jsonify(view)


@app.route('/api/game/state', methods=['GET'])
def game_state():
    """Current redacted state. The frontend reads bot moves from here."""
    session_id = request.args.get('id')
    player_id = request.args.get('playerId')
    with SESSIONS.lock(session_id):          # serialize with any in-flight writer
        session, err = _load_session(session_id, player_id)
        if err:
            return err
        return jsonify(_redact_view(session.public_view()))


@app.route('/api/game/action', methods=['POST'])
def game_action():
    """Apply the human's action only. The bot responds in a separate
    /api/game/bot-action call so the client can reveal the new card first."""
    data = request.get_json(silent=True) or {}
    session_id = data.get('id')
    player_id = data.get('playerId')
    # One lock per session for the whole load-modify-put: concurrent requests for
    # the same session (double-click, retry, or a /bot-action racing this) can't
    # clobber each other or double-apply.
    with SESSIONS.lock(session_id):
        session, err = _load_session(session_id, player_id)
        if err:
            return err

        if not session.is_human_turn():
            return jsonify({"error": "not your turn"}), 409

        pre_status, pre_net = session.data['status'], session.data.get('human_net', 0.0)
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

        _record_hand_end(session, pre_status, pre_net)
        _persist(session)
        return jsonify(_redact_view(session.public_view()))


@app.route('/api/game/bot-action', methods=['POST'])
def game_bot_action():
    """Run the bot's pending turn(s). Split out from /action so the client can
    render the freshly-dealt board and a 'thinking' indicator before the (possibly
    slow) river solve. A no-op if it isn't the bot's turn."""
    # Bound abuse of the solve-triggering endpoint (the _SOLVE_SEMAPHORE caps CONCURRENCY; this caps
    # per-IP VOLUME). 120/min is generous for legit play (a few bot-actions per hand x a few hands/min).
    if _rate_limited('bot_action', _client_ip(), limit=120, window_seconds=60):
        return jsonify({"error": "too many requests — slow down"}), 429
    data = request.get_json(silent=True) or {}
    session_id = data.get('id')
    player_id = data.get('playerId')
    with SESSIONS.lock(session_id):
        session, err = _load_session(session_id, player_id)
        if err:
            return err
        pre_status, pre_net = session.data['status'], session.data.get('human_net', 0.0)
        try:
            # Pause when a bot action deals a new board card so the client can
            # render it before the bot's (possibly slow, river-solve) decision on
            # the new street; the client's bot-turn loop calls back to resume.
            _run_bot(session, stop_on_new_card=True)
        except GameError as e:
            return jsonify({"error": str(e)}), 400
        _record_hand_end(session, pre_status, pre_net)
        _persist(session)
        return jsonify(_redact_view(session.public_view()))


@app.route('/api/game/next-hand', methods=['POST'])
def game_next_hand():
    """Deal the next hand in an existing session."""
    data = request.get_json(silent=True) or {}
    session_id = data.get('id')
    player_id = data.get('playerId')
    with SESSIONS.lock(session_id):
        session, err = _load_session(session_id, player_id)
        if err:
            return err
        # Rolling hand-cap: refuse the next hand once over quota (the just-finished
        # hand was allowed to complete; this stops the next deal).
        capped = _hand_cap_response(session.data.get('player_id'))
        if capped is not None:
            return capped
        try:
            session.start_next_hand()
            pre_status = session.data['status']          # 'in_hand' after the deal
            pre_net = session.data.get('human_net', 0.0)
            _run_bot(session)
        except GameError as e:
            return jsonify({"error": str(e)}), 400
        _record_hand_end(session, pre_status, pre_net)
        _persist(session)
        return jsonify(_redact_view(session.public_view()))


# =============================================================================
# +EV leaderboard + accounts
# =============================================================================

_STATS_CACHE = {'data': None, 'ts': 0.0}
_STATS_TTL_SECONDS = 5.0          # /api/stats is polled ~every 30s by every client
# /api/leaderboard does a full table Scan; cache per (n,min_hands,accounts_only)
# so an unauthenticated burst can't amplify into O(table) Scans every request.
# min_hands is snapped to this menu (largest entry <= requested) BOTH so the
# cache keyspace is bounded and so off-menu values can't bypass the cache and
# turn into per-request table Scans.
_LEADERBOARD_CACHE = {}           # (version, min_hands, accounts_only) -> (ts, full_board)
_LEADERBOARD_TTL_SECONDS = 10.0
# The 'all' cold-start fallback (lifetime player-store board, used while the recap scan is empty) is
# cached with THIS short TTL instead of the full one: short enough that 'all' self-corrects to the
# recap board within a few seconds of the scan landing, long enough that a poll storm during the cold
# window can't turn the fallback into a Scan-per-request.
_LEADERBOARD_FALLBACK_TTL = 3.0
_MIN_HANDS_MENU = (0, 10, 50, 100, 500, 1000)
# The cached board holds up to this many rows; pagination slices it (so page/offset
# never multiply the cache keyspace). 200 = 10 pages of 20 -- nobody scrolls a
# leaderboard past that, and it bounds the per-entry payload.
_LEADERBOARD_MAX = 200

# Per-bot-version per-PLAYER aggregate for the leaderboard rows. version_aggregates() is a FULL
# hand-table scan (~25s) -- far too slow for a request path. It USED to be cached PER WORKER, so the 2
# gunicorn workers scanned on independent clocks and served leaderboard snapshots up to 2 min apart
# (the board "flickered" / "wasn't synced" between refreshes). Now the scan result is a SINGLE SHARED
# snapshot in the global store (GLOBAL.{get,put}_version_snapshot): ONE worker per window wins
# GLOBAL.try_acquire_version_refresh(), runs the scan in a background thread, and publishes it; every
# worker reads the SAME blob, so the board is coherent across workers. A short in-process cache avoids
# a GetItem on every leaderboard hit. (NB the +EV card's per-version totals come from the shared
# global COUNTERS, not this scan; this snapshot is only the per-player ranking rows.)
_VERSION_TTL_SECONDS = 90.0               # snapshot older than this -> trigger a re-scan
_VERSION_READ_TTL = 5.0                   # in-process cache of the shared-snapshot READ
_VERSION_CACHE = {'data': {}, 'computedAt': 0, 'read_ts': 0.0}
_VERSION_LOCK = threading.Lock()
_VERSION_REFRESHING = {'v': False}


def _do_version_refresh():
    """Run the slow recap scan and publish it as the shared snapshot. Runs only on the worker that won
    the cross-worker lease, in a background thread, so no request ever blocks on the scan."""
    try:
        data = HANDS.version_aggregates()                  # {'totals': {...}, 'byPlayer': {...}}
        GLOBAL.put_version_snapshot(data, int(time.time()))
    except Exception:
        _LOG.warning("version snapshot refresh failed", exc_info=True)
    finally:
        with _VERSION_LOCK:
            _VERSION_REFRESHING['v'] = False


def _version_data():
    """Last-known SHARED version aggregates {'totals':..., 'byPlayer':...}: reads the shared snapshot
    (briefly cached in-process) and, when it's stale, lets ONE worker (cross-worker lease) kick a
    background re-scan. NEVER blocks. {} until the first scan finishes."""
    now = time.time()
    c = _VERSION_CACHE
    if now - c['read_ts'] > _VERSION_READ_TTL:
        try:
            snap = GLOBAL.get_version_snapshot()
        except Exception:
            snap = None
            _LOG.warning("version snapshot read failed", exc_info=True)
        if snap is not None:
            c['data'] = snap.get('data') or {}
            c['computedAt'] = int(snap.get('computedAt') or 0)
        c['read_ts'] = now
    if now - c['computedAt'] > _VERSION_TTL_SECONDS:       # stale (or never computed) -> refresh
        with _VERSION_LOCK:
            if not _VERSION_REFRESHING['v']:
                try:
                    won = GLOBAL.try_acquire_version_refresh(lease_seconds=int(_VERSION_TTL_SECONDS))
                except Exception:
                    won = False
                if won:
                    _VERSION_REFRESHING['v'] = True
                    threading.Thread(target=_do_version_refresh, daemon=True).start()
    return c['data']


_KNOWN_VERSIONS_CACHE = {'v': [], 'ts': 0.0}


def _known_versions():
    """Bot-version labels from the SHARED global COUNTERS (instant + coherent across workers) -- used
    to validate the ?version= filter and populate the dropdown, so a cold per-worker scan can't make
    one worker forget a version exists. Briefly cached (the counters are themselves ~5s-cached)."""
    now = time.time()
    c = _KNOWN_VERSIONS_CACHE
    if now - c['ts'] > 5.0:
        try:
            c['v'] = sorted((GLOBAL.get().get('byVersion') or {}).keys())
        except Exception:
            pass                                            # keep last-known on a store fault
        c['ts'] = now
    return c['v']


def _version_board(version, min_hands, accounts_only):
    """Rank players for ONE bot version from the cached recap aggregates (joined with player_store
    for handle/isRegistered). Operates on the cached aggregate, not a scan; the per-(version,cut)
    result is itself cached by the leaderboard endpoint. Rows match PlayerStore.top(include_id=True)
    so the endpoint shares the pagination/isYou/yourRank logic."""
    by_player_all = _version_data().get('byPlayer') or {}
    if version is None or version == 'all':
        # 'all' = sum across EVERY version, so the board reconciles with v1+v2+... (one source of
        # truth: the recap rows). Merge each version's per-player counts before resolving merges.
        by_player = {}
        for vmap in by_player_all.values():
            for pid, d in vmap.items():
                a = by_player.setdefault(pid, {'hands': 0, 'humanNetBB': 0.0})
                a['hands'] += d['hands']
                a['humanNetBB'] += d['humanNetBB']
    else:
        by_player = by_player_all.get(version, {})
    # Resolve each recap pid through its merge chain so a signed-in user's pre-merge (anon) hands
    # attribute to their CANONICAL account -- matching the lifetime board (which already moves them)
    # and so isYou/yourRank work for the caller's canonical id. Combine entries that resolve together;
    # apply min_hands AFTER combining (so split anon+canonical hands count as one).
    # Pull ALL player rows in ONE (cached) scan -- not a GetItem per player (that was ~1 DynamoDB
    # round-trip per player, i.e. seconds on a full board) -- and resolve merges locally.
    prows = _player_rows_cached()
    resolved = {}
    for pid, d in by_player.items():
        prow = prows.get(pid) or {}
        cid = prow.get('merged_into') or pid
        crow = (prows.get(cid) or {}) if prow.get('merged_into') else prow
        agg = resolved.setdefault(cid, {'hands': 0, 'humanNetBB': 0.0, 'row': crow})
        agg['hands'] += d['hands']
        agg['humanNetBB'] += d['humanNetBB']
    rows = []
    for cid, agg in resolved.items():
        crow = agg['row']
        if agg['hands'] < min_hands or crow.get('merged_into') \
                or (accounts_only and not crow.get('isRegistered')):
            continue
        net = round(agg['humanNetBB'], 2)
        rows.append({'handle': crow.get('handle') or 'Anonymous', 'hands': agg['hands'], 'netBB': net,
                     'bbPer100': round(net / agg['hands'] * 100.0, 2) if agg['hands'] else 0.0,
                     'isRegistered': bool(crow.get('isRegistered')), '_pid': cid})
    # Rank by NET BB desc (ties: more hands first). bb/100 stays as a secondary display stat; ranking
    # by rate made the Net BB column non-monotonic (a +20-over-2-hands row above +280-over-150), which
    # read as "BB/hand not aligned with Net BB". min_hands still gates out tiny-sample noise.
    rows.sort(key=lambda r: (r['netBB'], r['hands']), reverse=True)
    return rows[:_LEADERBOARD_MAX]


_version_data()   # pre-warm: start the first scan at import so the breakdown is ready ~asap


def _player_public_self(row):
    """The caller's OWN row, curated. The canonical playerId is returned so the
    client can adopt it after sign-in. Email is intentionally NOT returned."""
    h = row.get('hands') or 0
    net = float(row.get('netBB') or 0.0)
    return {
        'playerId': row.get('playerId'),
        'handle': row.get('handle'),
        'hands': int(h),
        'netBB': round(net, 2),
        'bbPer100': round((net / h * 100.0) if h else 0.0, 2),
        'isRegistered': bool(row.get('isRegistered')),
        'usernameSet': bool(row.get('handle')),
    }


# Cache the players-table scan (all_rows) briefly. It's the merge-resolution map needed by BOTH the
# leaderboard board build and /api/me -- and /api/me is polled per hand. Without this cache, /api/me
# would run a full O(table) Scan PER request (a cost/latency/DoS regression). Short TTL so a new or
# newly-merged player still shows up within seconds.
_PLAYER_ROWS_CACHE = {'data': None, 'ts': 0.0}
_PLAYER_ROWS_TTL_SECONDS = 10.0
_PLAYER_ROWS_LOCK = threading.Lock()


def _player_rows_cached():
    now = time.time()
    c = _PLAYER_ROWS_CACHE
    if c['data'] is None or now - c['ts'] > _PLAYER_ROWS_TTL_SECONDS:
        with _PLAYER_ROWS_LOCK:
            if c['data'] is None or now - c['ts'] > _PLAYER_ROWS_TTL_SECONDS:
                c['data'] = PLAYERS.all_rows()
                c['ts'] = now
    return c['data']


@app.route('/api/stats', methods=['GET'])
def stats():
    """The +EV counter source: global totals + the served blueprint. Cached a few
    seconds (it's the hottest endpoint — every client polls it)."""
    now = time.time()
    c = _STATS_CACHE
    if c['data'] is None or now - c['ts'] > _STATS_TTL_SECONDS:
        try:
            g = GLOBAL.get()
            # byVersion is now LIVE per-version running counters on the global row (no hand-table
            # scan) -- {version: {hands, humanNetBB}}. netBB is the human field's net (the card
            # negates it for the bot's perspective), matching the version-blind totalNetBB.
            by_version = {v: {'hands': d['hands'], 'humanNetBB': round(float(d['netBB']), 2)}
                          for v, d in (g.get('byVersion') or {}).items()}
            c['data'] = {
                'totalHands': g['totalHands'],
                'totalNetBB': round(float(g['totalNetBB']), 2),
                'totalPlayers': g['totalPlayers'],
                'byVersion': by_version,
                'blueprint': _BLUEPRINT_PATH.name,
                'iterations': BLUEPRINT_DB.get_metadata('total_iterations', 0),
            }
            c['ts'] = now
        except Exception:
            # A store fault must NOT 500 the hottest endpoint (every client polls it). Serve the
            # last-known-good cache, or a zero-state if we never had one.
            _LOG.warning("stats: GLOBAL.get failed; serving stale/zero-state", exc_info=True)
            if c['data'] is None:
                c['data'] = {'totalHands': 0, 'totalNetBB': 0.0, 'totalPlayers': 0,
                             'byVersion': {}, 'blueprint': _BLUEPRINT_PATH.name,
                             'iterations': BLUEPRINT_DB.get_metadata('total_iterations', 0)}
    return jsonify(c['data'])


@app.route('/api/leaderboard', methods=['GET'])
def leaderboard():
    """Ranked by bb/100 over >= min_hands. accounts_only=true is the ranked board
    (signed-in accounts); omit it for an 'all players' cut that includes anonymous
    players. `version` selects a bot-version cut (v1/v2/...) from the recap aggregate,
    'all' (default) sums across versions; `you` marks the caller's own row. Paginated:
    n = page size, offset = rows to skip. Returns {players, total, yourRank, versions}.
    Rows are redacted (no playerId/email)."""
    if _rate_limited('leaderboard', _client_ip(), limit=60, window_seconds=60):
        return jsonify({"error": "rate limit"}), 429
    def _int(name, default, lo, hi):
        try:
            return max(lo, min(hi, int(request.args.get(name, default))))
        except (TypeError, ValueError):
            return default
    n = _int('n', 20, 1, 100)                              # page size
    offset = _int('offset', 0, 0, _LEADERBOARD_MAX)
    # Snap min_hands to a small menu so off-menu values can't bypass the cache and
    # turn into per-request table Scans (and so the cache keyspace stays bounded).
    raw_min_hands = _int('min_hands', 50, 0, 10 ** 9)
    min_hands = max((m for m in _MIN_HANDS_MENU if m <= raw_min_hands), default=0)
    accounts_only = request.args.get('accounts_only', '').lower() in ('1', 'true', 'yes')
    you = request.args.get('you') or None        # the caller's OWN playerId, to mark their row
    version = request.args.get('version') or None
    # VALIDATE version against the KNOWN set (from the SHARED global counters -- instant + coherent,
    # never the cold per-worker scan) before it keys the cache: an arbitrary client string would
    # otherwise mint unbounded _LEADERBOARD_CACHE entries (memory-exhaustion DoS).
    known = _known_versions()
    if version == 'all':
        version = None                           # None -> the recap board summed across all versions
    elif version and version not in known:
        version = None                           # unknown label -> 'all' (bounds the cache keyspace)
    # A KNOWN per-version cut whose recap snapshot hasn't landed yet (cold start, or right after a new
    # version's first hands while the ~90s shared scan catches up): report `pending` rather than
    # SILENTLY downgrading to 'all' (which would show the wrong cut on the v2 tab). The client renders
    # "updating…" until the snapshot lands. 'all' (version is None) keeps its lifetime cold-fallback.
    if version is not None and version not in (_version_data().get('totals') or {}):
        return jsonify({'players': [], 'total': 0, 'yourRank': None,
                        'versions': known, 'pending': True})
    # Cache the FULL board (with an internal '_pid' for caller-row matching) per
    # (version, min_hands, accounts_only); pagination + the per-caller 'isYou' are applied AFTER the
    # cache, so page/offset and 'you' don't multiply cache keys. EVERY cut (incl. 'all') ranks from
    # the recap aggregate (_version_board) so the version dropdown is internally consistent: the 'all'
    # row always equals that player's v1+v2+... rows summed (one source of truth, not the separate
    # lifetime counter which can drift from the recap rows).
    ck = (version, min_hands, accounts_only)
    now = time.time()
    # 'all' (version is None) can fall back to the durable lifetime player-store board when the recap
    # aggregate is cold/empty. PlayerStore.top(include_id=True) rows are shaped EXACTLY like
    # _version_board rows (handle/hands/netBB/bbPer100/isRegistered/_pid, sorted by Net BB), so
    # pagination/yourRank/isYou work identically. In prod the lifetime board ~= the recap board; they
    # only diverge on seeded dev data.
    def _lifetime_board():
        return PLAYERS.top(n=_LEADERBOARD_MAX, min_hands=min_hands,
                           accounts_only=accounts_only, include_id=True)
    cached = _LEADERBOARD_CACHE.get(ck)
    if cached is not None and now - cached[0] <= _LEADERBOARD_TTL_SECONDS:
        full = cached[1]
    else:
        try:
            full = _version_board(version, min_hands, accounts_only)
            store_ts = now
            if not full and version is None:
                # COLD-START / SCAN-FAILED FALLBACK: the version board comes from the background
                # hand-scan, which is empty until the first scan lands, after every worker restart,
                # and forever if that scan persistently fails. Don't render a blank 'all' board while
                # the card shows a number -- serve the lifetime store. CACHE it (so a poll storm in the
                # cold window can't Scan-per-request), but with a SHORT effective TTL (backdated ts) so
                # 'all' self-corrects to the recap board within seconds of the scan landing. Per-version
                # cuts have no lifetime equivalent -> they just populate once the scan completes.
                full = _lifetime_board()
                store_ts = now - _LEADERBOARD_TTL_SECONDS + _LEADERBOARD_FALLBACK_TTL
            if len(_LEADERBOARD_CACHE) > 256:        # defensive backstop; version is validated so the
                _LEADERBOARD_CACHE.clear()           # keyspace is already bounded
            _LEADERBOARD_CACHE[ck] = (store_ts, full)
        except Exception:
            # A store fault (missing table / throttle) must NOT 500 a public endpoint. Prefer a stale
            # cached board; else 'all' can still serve the lifetime store (never blank); else empty.
            _LOG.warning("leaderboard: board build failed; serving stale/empty", exc_info=True)
            if cached is not None:
                full = cached[1]
            else:
                try:
                    full = _lifetime_board() if version is None else []
                except Exception:
                    full = []
    # LIVE caller row: the board is the ~90s shared snapshot, so the caller's OWN row lags their latest
    # hand -- while the "You" header and the +EV card update instantly off the live counters, which makes
    # a stale own-row look broken. For the 'all' cut, overlay the caller's LIVE lifetime PlayerStore
    # counter (version-blind = their all-versions total, the SAME source as /api/me) onto their row and
    # re-rank, so YOUR row is always current even while everyone else's comes from the snapshot. Copy --
    # never mutate the cached board. (Per-version cuts have no per-player live counter, so they still
    # track the snapshot; only refresh an EXISTING row -- a brand-new/sub-min-hands player appears once
    # the snapshot catches up.)
    if you and version is None:
        try:
            live = PLAYERS.get(you)
        except Exception:
            live = None
        if live and not live.get('merged_into') and any(r.get('_pid') == you for r in full):
            h = int(live.get('hands') or 0)
            net = round(float(live.get('netBB') or 0.0), 2)
            full = [dict(r) for r in full]                       # copy; the cached list is shared
            for r in full:
                if r.get('_pid') == you:
                    r['hands'], r['netBB'] = h, net
                    r['bbPer100'] = round(net / h * 100.0, 2) if h else 0.0
                    break
            full.sort(key=lambda r: (r['netBB'], r['hands']), reverse=True)
    # The caller's 1-based rank in the FULL board (page-independent) so the client can open the
    # page that contains them; None if they're off the board (sub-min-hands, or beyond the cap).
    your_rank = next((i + 1 for i, r in enumerate(full) if you and r.get('_pid') == you), None)
    # Strip the internal '_pid' and mark ONLY the caller's own row (never expose another id).
    out = []
    for r in full[offset:offset + n]:
        row = {k: v for k, v in r.items() if k != '_pid'}
        if you and r.get('_pid') == you:
            row['isYou'] = True
        out.append(row)
    return jsonify({'players': out, 'total': len(full), 'yourRank': your_rank,
                    'versions': known})


@app.route('/api/me', methods=['GET'])
def me():
    """Return the caller's own player row (lifetime stats for the AiGame header).
    Public-by-UUID-by-design: anyone who knows your playerId UUID can read your
    curated stats (handle, hands, netBB, bbPer100, isRegistered). UUIDs are
    unguessable and not enumerable, so this is a deliberate design — the
    leaderboard already publishes curated rows. Missing/invalid playerId → empty
    0-state so the UI renders cleanly on first load."""
    player_id = (request.args.get('playerId') or '').strip()
    if not player_id or not _valid_player_id(player_id):
        return jsonify({"playerId": None, "handle": None, "hands": 0,
                        "netBB": 0.0, "bbPer100": 0.0, "isRegistered": False})
    try:
        row = PLAYERS.get(player_id)
    except Exception:
        _LOG.warning("me: PLAYERS.get failed; serving 0-state", exc_info=True)
        row = None
    # Follow the merge chain so a pre-merge (anon) id returns the CANONICAL row (the client should
    # adopt that id; the canonical row already holds the merged hands+netBB from link_account).
    if row and row.get('merged_into'):
        try:
            canon = PLAYERS.get(row['merged_into'])
            if canon:
                row = canon
        except Exception:
            pass
    if not row:
        return jsonify({"playerId": player_id, "handle": None, "hands": 0,
                        "netBB": 0.0, "bbPer100": 0.0, "isRegistered": False})
    # Lifetime stats come from the durable PlayerStore COUNTER (incremented on every hand_over), so
    # the "You" header updates LIVE per hand. The recap-aggregate leaderboard is a ~120s background
    # scan that would lag per-hand; in prod the counter == that aggregate (both written per hand), so
    # this stays consistent with the board. (On seeded/cloned dev data the two can differ.)
    return jsonify(_player_public_self(row))


@app.route('/api/player', methods=['POST'])
def player_upsert():
    """Set the caller's unique username (signed-in players pick one on sign-in, or
    rename later). Validates the handle (regex + profanity -> 400) and uniqueness
    (-> 409)."""
    data = request.get_json(silent=True) or {}
    player_id = data.get('playerId')
    handle = data.get('handle')
    if not _valid_player_id(player_id):
        return jsonify({"error": "playerId required (UUID format)"}), 400
    # Rate-limit handle changes per-player AND per-IP. The per-IP guard catches a
    # malicious caller fanning out across UUIDs; the per-player guard catches a
    # legit-but-buggy client looping the rename.
    if _rate_limited('player', player_id, limit=10, window_seconds=60) or \
       _rate_limited('player_ip', _client_ip(), limit=30, window_seconds=60):
        return jsonify({"error": "too many handle changes; slow down"}), 429
    # Count a newly-seen player exactly once. /api/game/new is the usual first
    # touch, but a player can hit /api/player first (setting a name before
    # playing). Without this, upsert_handle creates the row silently and
    # GLOBAL.totalPlayers under-counts.
    try:
        if PLAYERS.create_if_absent(player_id):
            GLOBAL.record_new_player()
    except Exception:
        _LOG.warning("new-player counting failed for player=%s", player_id, exc_info=True)
    try:
        row = PLAYERS.upsert_handle(player_id, handle)
    except InvalidHandle as e:
        return jsonify({"error": str(e)}), 400
    except HandleTaken:
        return jsonify({"error": "that username is already taken"}), 409
    return jsonify(_player_public_self(row))


@app.route('/api/auth/google', methods=['POST'])
def auth_google():
    """Verify a Cognito-issued Google ID token and resolve the canonical account.

    One account per Google `sub`: a returning user (even on a new device) ADOPTS
    the existing account (the response's `playerId` is the canonical id the client
    must adopt) AND has this device's anonymous hands+netBB merged into the
    canonical row (non-destructive); a first sign-in binds this browser's row,
    absorbing its anonymous history. The client then sets a unique username if `usernameSet` is false
    (`suggestedHandle` pre-fills the prompt). 503 unconfigured, 401 bad/unverified
    token, 403 if this browser is already bound to a different account.
    """
    import auth
    if not auth.is_configured():
        return jsonify({"error": "auth not configured"}), 503
    # Rate-limit per IP BEFORE the JWT verify (which costs CPU on every attempt).
    # 20/min is generous for legitimate retries; an attacker bursting fake tokens
    # bounces off here without burning JWKS lookups + RSA verifies.
    if _rate_limited('auth_google', _client_ip(), limit=20, window_seconds=60):
        return jsonify({"error": "too many auth attempts; slow down"}), 429
    data = request.get_json(silent=True) or {}
    id_token = data.get('idToken')
    player_id = data.get('playerId')
    if not isinstance(id_token, str) or not _valid_player_id(player_id):
        return jsonify({"error": "idToken and a valid playerId required"}), 400
    try:
        claims = auth.verify_cognito_id_token(id_token)
    except auth.AuthNotConfigured:
        return jsonify({"error": "auth not configured"}), 503
    except auth.AuthError as e:
        # Generic message to the client — PyJWT's reason ("Signature has expired",
        # "no JWKS key matches the token's kid") is an oracle for token-forging
        # attempts. Log the detail for ops.
        _LOG.info("auth rejected (ip=%s, reason=%s)", _client_ip(), e)
        return jsonify({"error": "invalid token"}), 401
    # Trust `sub` as identity; require a verified email before storing/using it.
    ev = claims.get('email_verified')
    if not (ev is True or str(ev).lower() == 'true'):
        return jsonify({"error": "Google email is not verified"}), 401
    sub, email = claims.get('sub'), claims.get('email')
    # Same one-shot new-player count as /api/game/new and /api/player: if this is
    # the first server-side touch for player_id (sign-in before any hand), count
    # them. Otherwise link_account would silently create the row.
    try:
        if PLAYERS.create_if_absent(player_id):
            GLOBAL.record_new_player()
    except Exception:
        _LOG.warning("new-player counting failed for player=%s", player_id, exc_info=True)
    try:
        row = PLAYERS.link_account(player_id, email=email, auth_provider='google',
                                   provider_sub=sub)
    except AccountConflict:
        return jsonify({"error": "this browser is already signed in to another "
                                 "account; sign out first"}), 403
    # If THIS call merged the device's anon row into an existing account, that anon was counted at
    # /game/new but is no longer a distinct player -> decrement once. `_mergedThisCall` is set ONLY on
    # the actual merge transition (not the permanent merged_into), so a retried/replayed sign-in with
    # the same anon id can't double-decrement totalPlayers (the "67 vs 66" fix, made idempotent).
    if isinstance(row, dict) and row.pop('_mergedThisCall', False):
        try:
            GLOBAL.record_merged_player()
        except Exception:
            _LOG.warning("merge-count adjustment failed for player=%s", player_id, exc_info=True)
    out = _player_public_self(row)
    out['suggestedHandle'] = sanitize_display_name(
        claims.get('name') or (email.split('@')[0] if email else ''))
    return jsonify(out)


# =============================================================================
# Health
# =============================================================================

def _postflop_table_status():
    """Whether the baked postflop lookup tables are present. If a table is
    missing, PostflopV2 falls back to slow lazy bucketing -- a misbuilt image
    signal worth surfacing in the healthcheck."""
    base = os.path.join(backend_dir, 'bot', 'analysis', 'abstractions')
    return {street: os.path.exists(os.path.join(base, f'postflop_table_{street}.npz'))
            for street in ('flop', 'turn')}


# Store-reachability probe for healthz. Without this, a revoked/expired IAM key
# leaves healthz green while every game write 500s -- UptimeRobot stays quiet
# and the outage goes unnoticed (boto3 constructs its client lazily, so the
# import-time path can't catch it). The probe is a single cheap GetItem on a
# key that never exists.
#
# Failure policy: THREE consecutive failed probes flip healthz to 503. One
# blip (a throttle, a transient network error) must NOT flip it -- Lightsail's
# health check consumes healthz, and a flapping 503 would restart the
# container, which doesn't fix a broken store and adds churn. With the 30s
# probe cache, 3 strikes ≈ a sustained ~90s outage before we go red.
_STORE_PROBE = {'ts': 0.0, 'ok': True, 'strikes': 0}
_STORE_PROBE_TTL_SECONDS = 30.0
_STORE_PROBE_STRIKES_TO_FAIL = 3
_STORE_PROBE_LOCK = threading.Lock()


def _stores_ok():
    """True while the player store answers reads (cached ~30s, 3-strike)."""
    now = time.time()
    with _STORE_PROBE_LOCK:
        if now - _STORE_PROBE['ts'] < _STORE_PROBE_TTL_SECONDS:
            return _STORE_PROBE['ok'] or \
                _STORE_PROBE['strikes'] < _STORE_PROBE_STRIKES_TO_FAIL
        _STORE_PROBE['ts'] = now
    try:
        # GetItem on a key that never exists: exercises credentials + table
        # reachability for ~free. ('#' can't appear in a real playerId.)
        PLAYERS.get('healthz#probe')
        probe_ok = True
    except Exception:
        _LOG.warning("healthz store probe failed", exc_info=True)
        probe_ok = False
    with _STORE_PROBE_LOCK:
        _STORE_PROBE['ok'] = probe_ok
        _STORE_PROBE['strikes'] = 0 if probe_ok else _STORE_PROBE['strikes'] + 1
        return probe_ok or _STORE_PROBE['strikes'] < _STORE_PROBE_STRIKES_TO_FAIL


def _health_payload():
    if _BLUEPRINT_LOAD_ERROR is not None:
        # Degraded: blueprint failed to load at import. /api/healthz returns 503
        # with the exception TYPE so a rolling-deploy probe correctly fails AND
        # ops gets a pointer (the full message -- which can embed container
        # filesystem paths -- stays in the server logs only).
        return {
            "status": "degraded",
            "error": _BLUEPRINT_LOAD_ERROR_PUBLIC,
            "blueprint": None,
            "sessionStore": type(SESSIONS).__name__,
            "commit": os.environ.get('ALLIN_GIT_SHA'),
        }
    stores_ok = _stores_ok()
    return {
        "status": "ok" if stores_ok else "degraded",
        **({} if stores_ok else
           {"error": "player store unreachable (see server logs)"}),
        "blueprint": _BLUEPRINT_PATH.name,
        "iterations": BLUEPRINT_DB.get_metadata('total_iterations', 0),
        "postflopTables": _postflop_table_status(),
        "sessionStore": type(SESSIONS).__name__,
        "storesOk": stores_ok,
        "debugOverlay": _DEBUG_OVERLAY,
        # River safe-gadget policy + purification threshold actually in force on the served
        # bot -- so a deploy can confirm them without reading the code. None in degraded
        # mode (BOT failed to load), so guard against it (healthz must still answer).
        "riverGadget": (None if BOT is None
                        else ("off" if not BOT.safe_gadget else BOT.gadget_anchor)),
        "purify": (None if BOT is None else BOT.purify_threshold),
        # Phase 6 opponent exploitation: false when off (default), else #player models + tilt budget.
        "exploit": (False if HUMAN_MODEL is None
                    else {"players": len(HUMAN_MODEL.players), "delta": _EXPLOIT_DELTA}),
        "commit": os.environ.get('ALLIN_GIT_SHA'),
    }


@app.route('/api/test', methods=['GET'])
@app.route('/api/healthz', methods=['GET'])
def test():
    """Health/diagnostics for the load balancer + a quick manual check. The
    /api/healthz alias is the rolling-deploy probe. Returns 503 (not 200) when
    the blueprint failed to load OR the player store has failed 3 consecutive
    probes (sustained ~90s outage; see _stores_ok), so Lightsail rolling-deploy
    aborts and UptimeRobot alerts."""
    payload = _health_payload()
    status = 200 if payload.get("status") == "ok" else 503
    return jsonify(payload), status


# Start the inactivity sweeper at import time so it runs under BOTH gunicorn (which
# imports `app` via wsgi.py) and the dev server below. Idempotent + env-gated.
_start_sweeper()


if __name__ == "__main__":
    # DEVELOPMENT server only. This path is for local work; it is NOT for
    # production (the Werkzeug debugger is a remote-code-execution risk and the
    # dev server is not built for real load). Deploy via the WSGI entrypoint
    # instead -- gunicorn on Linux, waitress on Windows -- which imports `app`
    # from wsgi.py and never runs this block. See docs/DEPLOYMENT.md.
    #
    # Defaults are safe-ish for dev: bind loopback (not 0.0.0.0) and gate the
    # debugger behind ALLIN_DEBUG (default on for convenience). Override host via
    # ALLIN_DEV_HOST if you really need LAN access.
    logging.basicConfig(
        level=os.environ.get("ALLIN_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    debug = os.environ.get("ALLIN_DEBUG", "1") == "1"
    host = os.environ.get("ALLIN_DEV_HOST", "127.0.0.1")
    port = int(os.environ.get("ALLIN_DEV_PORT", "5000"))

    print("Starting Flask DEVELOPMENT server (do not use in production)...")
    print(f"Working directory: {os.getcwd()}")
    print(f"CORS origins: {ALLOWED_ORIGINS}")
    print(f"Session store: {type(SESSIONS).__name__} · solve permits: {_SOLVE_PERMITS}")

    print("\nRoutes registered:")
    for rule in app.url_map.iter_rules():
        methods = list(rule.methods) if rule.methods else []
        print(f"  {rule.endpoint}: {rule.rule} {methods}")

    print(f"\nServer starting on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
