# backend/api/wsgi.py
"""
Production WSGI entrypoint.

The Flask dev server (`python strategy_api.py`) is for local work only. In
production a real WSGI server runs this `app` object instead -- it has no
debugger exposed and is built for concurrent load:

    # Linux (deploy box):
    gunicorn --chdir backend/api wsgi:app \
        --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:5000

    # Windows (run the production server locally to test it):
    waitress-serve --listen=0.0.0.0:5000 --call wsgi:create_app

Notes
-----
* Worker model: a river solve is CPU-bound and can take a few seconds, so use a
  small worker count (~CPUs) plus threads, and a generous --timeout. The
  per-process river-solve semaphore in strategy_api caps concurrent solves.
* Session store: set ALLIN_SESSION_STORE=dynamodb for >1 worker (the default
  in-memory store is per-process and would split games across workers).
* Blueprint: pin ALLIN_BLUEPRINT_DB to the served snapshot (e.g. the 25M one).
"""
import logging
import os
import sys

# Load a local .env for dev (no-op if python-dotenv absent / file missing; real
# host env vars win). strategy_api loads it again for its own reads — harmless.
try:
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(_here)), '.env'))
    load_dotenv(os.path.join(os.path.dirname(_here), '.env'))
except ImportError:
    pass

# Configure logging ONCE, before importing the app (so the module-load "Loaded
# blueprint" line surfaces). Level is env-driven; format suits CloudWatch/Lightsail.
logging.basicConfig(
    level=os.environ.get("ALLIN_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Make `import strategy_api` work regardless of the launcher's CWD (gunicorn's
# --chdir already does this; this is belt-and-suspenders for other launchers).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from strategy_api import app


def create_app():
    """Factory form, for servers that want a callable (e.g. waitress --call)."""
    return app


if __name__ == "__main__":
    # Convenience: `python wsgi.py` runs the cross-platform waitress production
    # server (not the Flask debugger). Falls back with a clear message if
    # waitress isn't installed.
    host = os.environ.get("ALLIN_HOST", "0.0.0.0")
    port = int(os.environ.get("ALLIN_PORT", "5000"))
    try:
        from waitress import serve
    except ImportError:
        sys.exit("waitress is not installed. `pip install waitress`, or run "
                 "gunicorn (Linux): gunicorn --chdir backend/api wsgi:app")
    print(f"Serving (waitress) on http://{host}:{port}")
    serve(app, host=host, port=port)
