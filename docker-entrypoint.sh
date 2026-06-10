#!/bin/sh
# Container entrypoint: pick the gunicorn worker count based on whether the
# stores are shared.
#
# The in-memory stores (PlayerStore, GlobalStatsStore, HandStore, SessionStore)
# are PER-PROCESS dicts. Run >1 worker against them and reads oscillate as
# gunicorn round-robins requests between workers that hold different state.
# DynamoDB-backed stores are shared, so multi-worker is safe (and is the prod
# config -- two workers give us headroom for concurrent river solves).
#
# Default workers: 1 if any store is in-memory; 2 if both are DynamoDB. Override
# either branch with ALLIN_WORKERS=<N>. Threads / timeout / bind are also
# overridable for flexibility.
set -e

# Lowercase env values so 'Memory'/'MEMORY' don't silently fall into the wrong
# branch. (The Python store factories also lowercase, so a misspelling there is
# caught at import time — but if THIS script branches on the wrong value first
# the worker count goes to 2 with an in-memory store, which oscillates.)
SESSION_STORE=$(printf '%s' "${ALLIN_SESSION_STORE:-memory}" | tr '[:upper:]' '[:lower:]')
STORE_BACKEND=$(printf '%s' "${ALLIN_STORE_BACKEND:-memory}" | tr '[:upper:]' '[:lower:]')

# Validate. A typo (e.g. 'dyanmodb') silently falls through to the else branch
# AND the Python factory will then raise at import time → opaque CrashLoop.
# Surface it here, before gunicorn even spawns.
case "$SESSION_STORE" in
    memory|inmemory|dynamodb|dynamo) ;;
    *) echo "[entrypoint] ERROR: ALLIN_SESSION_STORE=$SESSION_STORE (expected: memory, dynamodb)" >&2
       exit 64 ;;
esac
case "$STORE_BACKEND" in
    memory|inmemory|dynamodb|dynamo) ;;
    *) echo "[entrypoint] ERROR: ALLIN_STORE_BACKEND=$STORE_BACKEND (expected: memory, dynamodb)" >&2
       exit 64 ;;
esac

if [ "$SESSION_STORE" = "memory" ] || [ "$SESSION_STORE" = "inmemory" ] \
        || [ "$STORE_BACKEND" = "memory" ] || [ "$STORE_BACKEND" = "inmemory" ]; then
    WORKERS="${ALLIN_WORKERS:-1}"
    echo "[entrypoint] in-memory store detected (sessions=$SESSION_STORE, backend=$STORE_BACKEND) -> --workers $WORKERS"
else
    WORKERS="${ALLIN_WORKERS:-2}"
    echo "[entrypoint] shared store (sessions=$SESSION_STORE, backend=$STORE_BACKEND) -> --workers $WORKERS"
fi

# --max-requests cycles workers periodically to shed any memory creep from
# long-lived numpy/CFR allocations. --access-logfile - sends per-request logs
# to stdout so Lightsail's container logs include the trace ops needs to
# diagnose user reports. --graceful-timeout matches our hand timeout so a
# SIGTERM mid-solve gets enough room to finish.
exec gunicorn --chdir backend/api wsgi:app \
    --workers "$WORKERS" \
    --threads "${ALLIN_THREADS:-4}" \
    --timeout "${ALLIN_TIMEOUT:-120}" \
    --graceful-timeout "${ALLIN_GRACEFUL_TIMEOUT:-120}" \
    --max-requests "${ALLIN_MAX_REQUESTS:-500}" \
    --max-requests-jitter "${ALLIN_MAX_REQUESTS_JITTER:-50}" \
    --access-logfile - \
    --error-logfile - \
    --bind "${ALLIN_BIND:-0.0.0.0:5000}"
