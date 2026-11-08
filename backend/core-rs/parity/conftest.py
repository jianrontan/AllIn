"""Shared setup for the Rust-vs-Python parity harness.

Run from backend/core-rs/ after `maturin develop --release`:
    ../../.venv/bin/python -m pytest parity -q

A component's tests SKIP until its name is in `allin_core.ported()`. Set
ALLIN_PARITY_STRICT=1 to turn those skips into failures (phase gates).
"""
import json
import os
import subprocess
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.normpath(os.path.join(HERE, '..', 'fixtures'))
BOT = os.path.normpath(os.path.join(HERE, '..', '..', 'bot'))
STRICT = os.environ.get('ALLIN_PARITY_STRICT') == '1'

# ALLIN_PARITY_TARGET=python runs the harness against the Python reference
# itself (parity/python_shim.py) -- proves fixtures + harness are consistent.
TARGET = os.environ.get('ALLIN_PARITY_TARGET', 'rust')
_IMPORT_ERR = None
if TARGET == 'python':
    import python_shim as allin_core
else:
    try:
        import allin_core  # noqa: F401
    except ImportError as e:                 # extension not built in this venv
        allin_core = None
        _IMPORT_ERR = e


def pytest_report_header(config):
    if allin_core is None:
        return f"allin_core: NOT IMPORTABLE ({_IMPORT_ERR}) -- run maturin develop"
    return (f"target={TARGET}; allin_core {allin_core.__version__}; ported={allin_core.ported()}; "
            f"strict={STRICT}; fixtures={FIXTURES}")


def _check_manifest():
    path = os.path.join(FIXTURES, 'manifest.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        m = json.load(f)
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BOT, text=True).strip()
    except Exception:
        return
    if m.get('git_sha') != head or m.get('src_dirty'):
        warnings.warn(
            f"fixtures were generated at {m.get('git_sha', '?')[:10]} "
            f"(src dirty={m.get('src_dirty')}), HEAD is {head[:10]}; if backend/bot/src "
            "changed since, regenerate: python scripts/export_rust_fixtures.py")


_check_manifest()


def need(component, *fixture_files):
    """Skip (or fail in strict mode) unless `component` is ported and its
    fixture files exist. Returns the allin_core module."""
    if allin_core is None:
        pytest.skip(f"allin_core not importable: {_IMPORT_ERR}")
    missing = [f for f in fixture_files if not os.path.exists(os.path.join(FIXTURES, f))]
    if missing:
        pytest.skip(f"missing fixtures {missing}: run backend/bot/scripts/export_rust_fixtures.py")
    if component not in allin_core.ported():
        msg = f"{component}: not ported yet"
        if STRICT:
            pytest.fail(msg)
        pytest.skip(msg)
    return allin_core


def fixture_path(name):
    return os.path.join(FIXTURES, name)
