# backend/bot/tests/test_leaderboard_api.py
"""
API-level tests for the +EV leaderboard + accounts endpoints and the per-hand
result hook (Items 3 & 4). Drives the WSGI app directly (Flask's test_client is
broken under this env's Flask/Werkzeug skew).

Covers: hand-end bump records once and is idempotent on replay; the rolling
hand-cap returns 429 + Retry-After; /api/player handle validation; /api/stats +
/api/leaderboard shapes; /api/auth/google 503 when Cognito is unconfigured.
"""
import io
import json
import os
import sys

import pytest

_BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BOT)
sys.path.insert(0, os.path.join(os.path.dirname(_BOT), 'api'))

# Force memory stores + no Cognito before importing the app.
os.environ.pop('ALLIN_STORE_BACKEND', None)
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
        box['headers'] = dict(headers)
        return lambda b: None
    data = b''.join(s.app(env, sr))
    return box['code'], box.get('headers', {}), (json.loads(data) if data else {})


def test_stats_and_leaderboard_shapes():
    code, _, v = call('GET', '/api/stats')
    assert code == 200
    for k in ('totalHands', 'totalNetBB', 'totalPlayers', 'blueprint', 'iterations'):
        assert k in v
    code, _, v = call('GET', '/api/leaderboard', query='n=5&min_hands=0')
    assert code == 200 and isinstance(v['players'], list)


def test_stats_byversion_is_live():
    """The card's byVersion now comes from LIVE global per-version counters: a completed hand
    bumps it immediately, with no hand-table scan."""
    code, _, before = call('GET', '/api/stats')
    assert code == 200 and 'byVersion' in before
    bv0 = sum(d['hands'] for d in (before.get('byVersion') or {}).values())
    pid = 'ZZ-bv'
    code, _, nv = call('POST', '/api/game/new', {'playerId': pid})
    call('POST', '/api/game/action', {'id': nv['sessionId'], 'playerId': pid, 'action': 'fold'})
    s._STATS_CACHE['data'] = None                   # bust the 5s cache so we read the fresh counter
    code, _, after = call('GET', '/api/stats')
    bv1 = sum(d['hands'] for d in (after.get('byVersion') or {}).values())
    assert bv1 == bv0 + 1, (bv0, bv1)               # the just-played hand is in byVersion live


def test_player_handle_validation():
    code, _, v = call('POST', '/api/player', {'playerId': 'pid-h', 'handle': 'Ron_1'})
    assert code == 200 and v['handle'] == 'Ron_1' and v['playerId'] == 'pid-h'
    code, _, v = call('POST', '/api/player', {'playerId': 'pid-h', 'handle': 'bad handle!'})
    assert code == 400
    code, _, v = call('POST', '/api/player', {'playerId': 'pid-h', 'handle': 'shithead'})
    assert code == 400                                   # profanity
    code, _, v = call('POST', '/api/player', {'handle': 'x'})
    assert code == 400                                   # missing playerId


def test_auth_google_503_when_unconfigured():
    code, _, v = call('POST', '/api/auth/google', {'idToken': 'x', 'playerId': 'p'})
    assert code == 503 and 'not configured' in v['error']


def test_auth_google_canonical_flow(monkeypatch):
    import auth
    monkeypatch.setattr(auth, 'is_configured', lambda: True)
    claims = {'sub': 'g-1', 'email': 'jane@x.com', 'email_verified': True,
              'name': 'Jane Doe'}
    monkeypatch.setattr(auth, 'verify_cognito_id_token', lambda tok, **k: dict(claims))

    # First sign-in from dev-A: binds, no username yet, suggestion from Google name.
    code, _, v = call('POST', '/api/auth/google', {'idToken': 't', 'playerId': 'dev-A'})
    assert code == 200, (code, v)
    assert v['playerId'] == 'dev-A' and v['usernameSet'] is False
    assert v['suggestedHandle'] == 'Jane Doe' and 'email' not in v

    # Second device dev-B, same Google sub: ADOPTS canonical dev-A (no new account).
    code, _, v = call('POST', '/api/auth/google', {'idToken': 't', 'playerId': 'dev-B'})
    assert code == 200 and v['playerId'] == 'dev-A'      # canonical id returned

    # Unverified email -> 401.
    monkeypatch.setattr(auth, 'verify_cognito_id_token',
                        lambda tok, **k: {**claims, 'email_verified': False})
    code, _, v = call('POST', '/api/auth/google', {'idToken': 't', 'playerId': 'dev-C'})
    assert code == 401 and 'not verified' in v['error']


def test_hand_end_records_once_and_is_idempotent():
    # New game for a fresh player.
    pid = 'pid-hand-end'
    code, _, v = call('POST', '/api/game/new', {'playerId': pid})
    assert code == 200
    sid = v['sessionId']
    before_player = s.PLAYERS.get(pid)
    before_global = s.GLOBAL.get()
    assert before_player is not None and before_player['hands'] == 0

    # It's the human's turn preflop (facing the blinds) -> fold ends the hand.
    assert v['toAct'] == 'you', v.get('toAct')
    assert any(a['action'] == 'fold' for a in v['legalActions'])
    code, _, v2 = call('POST', '/api/game/action', {'id': sid, 'playerId': pid, 'action': 'fold'})
    assert code == 200 and v2['status'] == 'hand_over'

    after_player = s.PLAYERS.get(pid)
    after_global = s.GLOBAL.get()
    assert after_player['hands'] == 1                    # exactly one hand recorded
    assert after_global['totalHands'] == before_global['totalHands'] + 1
    assert after_player['netBB'] < 0                     # folder forfeits the blind

    # Replay the same hand-ending request: hand is over -> no second count.
    code, _, _ = call('POST', '/api/game/action', {'id': sid, 'playerId': pid, 'action': 'fold'})
    assert s.PLAYERS.get(pid)['hands'] == 1
    assert s.GLOBAL.get()['totalHands'] == before_global['totalHands'] + 1


def test_new_player_counted_once():
    pid = 'pid-count-once'
    g0 = s.GLOBAL.get()['totalPlayers']
    call('POST', '/api/game/new', {'playerId': pid})
    g1 = s.GLOBAL.get()['totalPlayers']
    call('POST', '/api/game/new', {'playerId': pid})     # same player again
    g2 = s.GLOBAL.get()['totalPlayers']
    assert g1 == g0 + 1 and g2 == g1                     # counted once


def test_hand_cap_returns_429(monkeypatch):
    pid = 'pid-capped'
    monkeypatch.setattr(s.PLAYERS, 'hand_cap_status', lambda p: (False, 321))
    code, headers, v = call('POST', '/api/game/new', {'playerId': pid})
    assert code == 429 and headers.get('Retry-After') == '321'
    assert 'retryAfter' in v


def _seed_recaps(pid, version, delta_chips, n):
    """Put n recaps for (pid, version), each humanDelta=delta_chips. Distinct handKeys
    per version so the (playerId, handKey) idempotency doesn't collapse them."""
    for i in range(n):
        s.HANDS.put({'playerId': pid, 'sessionId': f'sess-{version}', 'handNumber': i,
                     'handKey': f'{i:013d}#sess-{version}#{i}',
                     'botVersion': version, 'result': {'humanDelta': float(delta_chips)}})


def test_leaderboard_version_cuts():
    """The version-filtered leaderboard reads the RECAP aggregate: 'all' == v1+v2, a version
    cut isolates that version, yourRank/versions are returned, and _pid never leaks."""
    pid = 'ZZ-recap'
    _seed_recaps(pid, 'v1', 10.0, 60)        # 60 hands
    _seed_recaps(pid, 'v2', 6.0, 40)         # 40 hands
    s.PLAYERS.create_if_absent(pid)
    s._PLAYER_ROWS_CACHE['data'] = None      # bust the cached players-scan
    s._LEADERBOARD_CACHE.clear()             # bust any board cached before this seed
    s._refresh_version_cache()               # populate byVersion synchronously

    code, _, lb = call('GET', '/api/leaderboard', query=f'version=all&min_hands=0&you={pid}')
    assert code == 200 and 'total' in lb and 'yourRank' in lb and 'versions' in lb
    assert all('_pid' not in row for row in lb['players'])      # internal id never leaked
    me_row = next((r for r in lb['players'] if r.get('isYou')), None)
    assert me_row and me_row['hands'] == 100, me_row            # all = v1 + v2

    code, _, lb1 = call('GET', '/api/leaderboard', query=f'version=v1&min_hands=0&you={pid}')
    v1_row = next((r for r in lb1['players'] if r.get('isYou')), None)
    assert v1_row and v1_row['hands'] == 60, v1_row             # the v1 cut isolates


def test_me_is_counter_based_and_live():
    """The 'You' header reads the live PlayerStore COUNTER (not the lagging recap scan), so it
    ticks up the instant a hand completes -- no version-cache refresh in between."""
    pid = 'ZZ-live'
    before = call('GET', '/api/me', query=f'playerId={pid}')[2]['hands']    # 0 (unknown player)
    code, _, nv = call('POST', '/api/game/new', {'playerId': pid})
    assert code == 200
    # human is in the BB facing the blinds -> fold ends the hand immediately
    call('POST', '/api/game/action', {'id': nv['sessionId'], 'playerId': pid, 'action': 'fold'})
    after = call('GET', '/api/me', query=f'playerId={pid}')[2]['hands']
    assert after == before + 1, (before, after)                 # live, no scan needed


def test_me_never_500s_on_store_fault(monkeypatch):
    """/api/me degrades to 0-state on a store fault, never 500s."""
    monkeypatch.setattr(s.PLAYERS, 'get',
                        lambda p: (_ for _ in ()).throw(RuntimeError('ddb')))
    code, _, me = call('GET', '/api/me', query='playerId=ZZ-fault')
    assert code == 200, (code, me)


def test_merge_decrements_total_players_once(monkeypatch):
    """The '67 vs 66' fix: a cross-device merge decrements totalPlayers exactly once,
    and a replayed sign-in does not double-decrement."""
    import auth
    monkeypatch.setattr(auth, 'is_configured', lambda: True)
    claims = {'sub': 'g-merge', 'email': 'm@x.com', 'email_verified': True, 'name': 'M'}
    monkeypatch.setattr(auth, 'verify_cognito_id_token', lambda tok, **k: dict(claims))

    # Device M-A signs in first -> canonical account, counted as a new player.
    call('POST', '/api/auth/google', {'idToken': 't', 'playerId': 'M-A'})
    # Device M-B plays + finishes a hand (so it has hands > 0 and is counted), then signs in.
    code, _, nv = call('POST', '/api/game/new', {'playerId': 'M-B'})
    call('POST', '/api/game/action', {'id': nv['sessionId'], 'playerId': 'M-B', 'action': 'fold'})
    g_before = s.GLOBAL.get()['totalPlayers']
    code, _, v = call('POST', '/api/auth/google', {'idToken': 't', 'playerId': 'M-B'})
    assert code == 200 and v['playerId'] == 'M-A'              # adopted the canonical account
    assert s.GLOBAL.get()['totalPlayers'] == g_before - 1      # M-B merged -> one fewer player

    # Replay the same sign-in: already merged -> NO second decrement.
    call('POST', '/api/auth/google', {'idToken': 't', 'playerId': 'M-B'})
    assert s.GLOBAL.get()['totalPlayers'] == g_before - 1      # idempotent


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
