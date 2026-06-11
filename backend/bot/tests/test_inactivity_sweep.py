# backend/bot/tests/test_inactivity_sweep.py
"""
Tests for the inactivity auto-fold sweeper (strategy_api: _persist /
_resolve_abandoned / _sweep_once). Drives the WSGI app directly (Flask's
test_client is broken under this env's Flask/Werkzeug skew), mirroring
test_leaderboard_api.py.

Importing strategy_api starts no background sweeper here: _start_sweeper() skips
when 'pytest' is in sys.modules, so these tests call _sweep_once() deterministically.

Covers: an abandoned human-to-act hand is folded + recorded; a hand abandoned while
the BOT is to act (tab closed after /action, before /bot-action) is also resolved
(the H1 fix); fresh and legacy (no-deadline) sessions are left alone.
"""
import io
import json
import os
import sys
import time

import pytest

_BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BOT)
sys.path.insert(0, os.path.join(os.path.dirname(_BOT), 'api'))

# Force memory stores + no Cognito before importing the app.
os.environ.pop('ALLIN_STORE_BACKEND', None)
os.environ.pop('ALLIN_SESSION_STORE', None)
os.environ.pop('ALLIN_COGNITO_REGION', None)

import strategy_api as s


def call(method, path, body=None, query=''):
    raw = b'' if body is None else json.dumps(body).encode()
    env = {'REQUEST_METHOD': method, 'PATH_INFO': path, 'QUERY_STRING': query,
           'SERVER_NAME': 'l', 'SERVER_PORT': '5000', 'SERVER_PROTOCOL': 'HTTP/1.1',
           'wsgi.version': (1, 0), 'wsgi.url_scheme': 'http',
           'wsgi.input': io.BytesIO(raw), 'wsgi.errors': io.StringIO(),
           'wsgi.multithread': True, 'wsgi.multiprocess': False, 'wsgi.run_once': False,
           'CONTENT_LENGTH': str(len(raw)), 'CONTENT_TYPE': 'application/json'}
    box = {}
    def sr(status, headers, exc_info=None):
        box['code'] = int(status.split()[0])
        return lambda b: None
    data = b''.join(s.app(env, sr))
    return box['code'], (json.loads(data) if data else {})


def _new_game(pid):
    code, v = call('POST', '/api/game/new', {'playerId': pid})
    assert code == 200, (code, v)
    return v['sessionId']


def _backdate(sid):
    """Push the session's inactivity deadline into the past so the next sweep
    treats it as abandoned."""
    d = s.SESSIONS.get(sid)
    assert d is not None and d['status'] == 'in_hand'
    assert 'inactivity_deadline' in d, "in-hand persist must stamp a deadline"
    d['inactivity_deadline'] = time.time() - 1
    s.SESSIONS.put(sid, d)


def test_sweep_folds_abandoned_human_to_act():
    """A hand left with the human to act past the deadline is folded and recorded."""
    sid = _new_game('pid-sweep-human')
    # Fresh /new leaves the human (SB) to act, with a deadline stamped.
    assert s.SESSIONS.get(sid)['status'] == 'in_hand'
    _backdate(sid)
    s._sweep_once()
    after = s.SESSIONS.get(sid)
    assert after['status'] == 'hand_over', "abandoned human hand must be folded"
    assert after.get('result_recorded') is True, "swept fold must hit the leaderboard hook"
    print("PASS test_sweep_folds_abandoned_human_to_act")


def test_sweep_resolves_when_bot_to_act():
    """H1: a hand abandoned while the BOT is to act (tab closed after /action but
    before /bot-action) is still resolved -- the sweeper advances the bot, then
    folds the human."""
    sid = _new_game('pid-sweep-bot')
    # Human limps; the bot's turn is NOT run (no /bot-action call), so the hand is
    # left in_hand with the bot to act.
    code, v = call('POST', '/api/game/action',
                   {'id': sid, 'playerId': 'pid-sweep-bot', 'action': 'call'})
    assert code == 200, (code, v)
    mid = s.SESSIONS.get(sid)
    assert mid['status'] == 'in_hand'
    assert v.get('toAct') == 'bot', v                 # bot to act, human idle
    _backdate(sid)
    s._sweep_once()
    after = s.SESSIONS.get(sid)
    assert after['status'] == 'hand_over', "abandoned bot-to-act hand must be resolved (H1)"
    print("PASS test_sweep_resolves_when_bot_to_act")


def test_sweep_skips_fresh_and_legacy():
    """A fresh session (future deadline) and a legacy session (no deadline at all,
    i.e. created before this feature) are both left untouched -- never retroactively
    folded."""
    sid = _new_game('pid-sweep-skip')
    s._sweep_once()                                   # fresh deadline -> skipped
    assert s.SESSIONS.get(sid)['status'] == 'in_hand'
    # Legacy: drop the deadline entirely and confirm it's not swept.
    d = s.SESSIONS.get(sid)
    d.pop('inactivity_deadline', None)
    s.SESSIONS.put(sid, d)
    s._sweep_once()
    assert s.SESSIONS.get(sid)['status'] == 'in_hand', "legacy session must not be swept"
    print("PASS test_sweep_skips_fresh_and_legacy")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, '-q']))
