# backend/api/auth.py
"""
Cognito ID-token validation for "Sign in with Google" (federated through an AWS
Cognito User Pool's Hosted UI).

The backend trusts no client claim: it validates the ID token's RS256 signature
against the User Pool's JWKS, plus `iss`, `aud`, `exp`, and `token_use == 'id'`,
before binding the account (see PlayerStore.link_account).

Configuration is the user's job at deploy time (User Pool + Google IdP). This
module just needs three env vars; when any is unset, `is_configured()` is False
and the auth endpoint returns 503 so gameplay (playerId-routed, no token) runs
fine in dev without Cognito.

Uses PyJWT (+ cryptography) — already available — not python-jose.
"""
import json
import logging
import os
import threading
import time
import urllib.request

import jwt
from jwt.algorithms import RSAAlgorithm

_LOG = logging.getLogger(__name__)

_JWKS_TTL_SECONDS = 3600          # Cognito keys rotate; a fixed 1h cache is fine.
_JWKS_CACHE = {'url': None, 'keys': None, 'fetched': 0.0}
_JWKS_LOCK = threading.Lock()


class AuthNotConfigured(Exception):
    """Cognito env vars are unset — the auth endpoint should 503."""


class AuthError(Exception):
    """The token is present but invalid (expired / wrong aud|iss / tampered)."""


def _config():
    return (os.environ.get('ALLIN_COGNITO_REGION'),
            os.environ.get('ALLIN_COGNITO_USER_POOL_ID'),
            os.environ.get('ALLIN_COGNITO_APP_CLIENT_ID'))


def is_configured():
    return all(_config())


def _issuer(region, pool_id):
    return f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"


def _jwks_url(region, pool_id):
    return _issuer(region, pool_id) + "/.well-known/jwks.json"


def _default_fetcher(url):
    with urllib.request.urlopen(url, timeout=5) as r:   # nosec - fixed Cognito URL
        return json.loads(r.read().decode())['keys']


def _get_jwks(url, fetcher):
    now = time.time()
    with _JWKS_LOCK:
        if (_JWKS_CACHE['keys'] is not None and _JWKS_CACHE['url'] == url
                and now - _JWKS_CACHE['fetched'] < _JWKS_TTL_SECONDS):
            return _JWKS_CACHE['keys']
    keys = (fetcher or _default_fetcher)(url)
    with _JWKS_LOCK:
        _JWKS_CACHE.update(url=url, keys=keys, fetched=now)
    return keys


def verify_cognito_id_token(token, *, jwks_fetcher=None, leeway=0):
    """Validate a Cognito ID token and return its claims dict, or raise.

    Checks: RS256 signature against the JWKS key matching the token's `kid`,
    `iss` == the User Pool issuer, `aud` == the App Client ID, `exp` in the
    future, and `token_use == 'id'`. Raises AuthNotConfigured if env is unset,
    AuthError on any validation failure.
    """
    region, pool_id, app_client_id = _config()
    if not all((region, pool_id, app_client_id)):
        raise AuthNotConfigured("Cognito env vars not configured")

    issuer = _issuer(region, pool_id)
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as e:
        raise AuthError(f"malformed token header: {e}")
    kid = header.get('kid')

    keys = _get_jwks(_jwks_url(region, pool_id), jwks_fetcher)
    jwk = next((k for k in keys if k.get('kid') == kid), None)
    if jwk is None:
        raise AuthError("no JWKS key matches the token's kid")
    public_key = RSAAlgorithm.from_jwk(json.dumps(jwk))

    try:
        claims = jwt.decode(
            token, key=public_key, algorithms=['RS256'],
            audience=app_client_id, issuer=issuer, leeway=leeway,
            options={'require': ['exp', 'iss', 'aud']})
    except jwt.PyJWTError as e:
        raise AuthError(f"invalid token: {e}")

    if claims.get('token_use') != 'id':
        raise AuthError("token_use is not 'id'")
    return claims


def require_account(fn):
    """Flask decorator for endpoints that REQUIRE a signed-in account. Unused in
    v1 (every gameplay endpoint stays playerId-routed); stubbed for v1.1 saved
    hands. 503 if auth isn't configured, 401 if the bearer token is missing/invalid;
    on success stashes the verified claims on flask.g.account_claims."""
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        from flask import request, jsonify, g
        if not is_configured():
            return jsonify({"error": "auth not configured"}), 503
        authz = request.headers.get('Authorization', '')
        token = authz[7:].strip() if authz.lower().startswith('bearer ') else None
        if not token:
            return jsonify({"error": "missing bearer token"}), 401
        try:
            g.account_claims = verify_cognito_id_token(token)
        except AuthError as e:
            return jsonify({"error": str(e)}), 401
        return fn(*args, **kwargs)

    return wrapper
