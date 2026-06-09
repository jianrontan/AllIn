# backend/bot/tests/test_auth.py
"""
Cognito ID-token validation (backend/api/auth.py).

Self-contained: generates an RSA keypair, publishes it as a JWKS, signs ID
tokens with PyJWT, and asserts verify_cognito_id_token accepts a well-formed one
and rejects expired / wrong-aud / wrong-iss / tampered / non-id / unknown-kid
tokens, and raises AuthNotConfigured when the env is unset.
"""
import json
import os
import sys
import time

import pytest

# auth.py lives under backend/api, not backend/bot.
_BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_BACKEND, 'api'))

jwt = pytest.importorskip("jwt")
pytest.importorskip("cryptography")
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

import auth as authmod
from auth import verify_cognito_id_token, AuthError, AuthNotConfigured, is_configured

REGION = 'us-east-1'
POOL = 'us-east-1_TestPool'
CLIENT = 'app-client-123'
KID = 'test-key-1'
ISS = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL}"

_PRIV = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_JWK = json.loads(RSAAlgorithm.to_jwk(_PRIV.public_key()))
_JWK['kid'] = KID
_FETCHER = lambda url: [_JWK]      # noqa: E731 - stubbed JWKS, no network


def _token(claims=None, kid=KID, key=None):
    payload = {
        'iss': ISS, 'aud': CLIENT, 'token_use': 'id',
        'sub': 'google-sub-xyz', 'email': 'p@example.com',
        'exp': int(time.time()) + 3600, 'iat': int(time.time()),
    }
    payload.update(claims or {})
    return jwt.encode(payload, key or _PRIV, algorithm='RS256', headers={'kid': kid})


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv('ALLIN_COGNITO_REGION', REGION)
    monkeypatch.setenv('ALLIN_COGNITO_USER_POOL_ID', POOL)
    monkeypatch.setenv('ALLIN_COGNITO_APP_CLIENT_ID', CLIENT)
    authmod._JWKS_CACHE.update(url=None, keys=None, fetched=0.0)   # reset cache
    yield


def test_accepts_valid_token():
    claims = verify_cognito_id_token(_token(), jwks_fetcher=_FETCHER)
    assert claims['sub'] == 'google-sub-xyz' and claims['email'] == 'p@example.com'


def test_rejects_expired():
    tok = _token({'exp': int(time.time()) - 10})
    with pytest.raises(AuthError):
        verify_cognito_id_token(tok, jwks_fetcher=_FETCHER)


def test_rejects_wrong_audience():
    tok = _token({'aud': 'some-other-client'})
    with pytest.raises(AuthError):
        verify_cognito_id_token(tok, jwks_fetcher=_FETCHER)


def test_rejects_wrong_issuer():
    tok = _token({'iss': 'https://evil.example.com/pool'})
    with pytest.raises(AuthError):
        verify_cognito_id_token(tok, jwks_fetcher=_FETCHER)


def test_rejects_tampered_signature():
    tok = _token()
    tampered = tok[:-3] + ('aaa' if tok[-3:] != 'aaa' else 'bbb')
    with pytest.raises(AuthError):
        verify_cognito_id_token(tampered, jwks_fetcher=_FETCHER)


def test_rejects_non_id_token():
    tok = _token({'token_use': 'access'})
    with pytest.raises(AuthError):
        verify_cognito_id_token(tok, jwks_fetcher=_FETCHER)


def test_rejects_unknown_kid():
    tok = _token(kid='not-in-jwks')
    with pytest.raises(AuthError):
        verify_cognito_id_token(tok, jwks_fetcher=_FETCHER)


def test_rejects_wrong_signing_key():
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    tok = _token(key=other)                       # signed by a key not in the JWKS
    with pytest.raises(AuthError):
        verify_cognito_id_token(tok, jwks_fetcher=_FETCHER)


def test_auth_not_configured(monkeypatch):
    monkeypatch.delenv('ALLIN_COGNITO_REGION', raising=False)
    assert is_configured() is False
    with pytest.raises(AuthNotConfigured):
        verify_cognito_id_token(_token(), jwks_fetcher=_FETCHER)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
