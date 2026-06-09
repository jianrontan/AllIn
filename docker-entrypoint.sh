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

SESSION_STORE="${ALLIN_SESSION_STORE:-memory}"
STORE_BACKEND="${ALLIN_STORE_BACKEND:-memory}"

if [ "$SESSION_STORE" = "memory" ] || [ "$STORE_BACKEND" = "memory" ]; then
    WORKERS="${ALLIN_WORKERS:-1}"
    echo "[entrypoint] in-memory store detected (sessions=$SESSION_STORE, backend=$STORE_BACKEND) -> --workers $WORKERS"
else
    WORKERS="${ALLIN_WORKERS:-2}"
    echo "[entrypoint] shared store (sessions=$SESSION_STORE, backend=$STORE_BACKEND) -> --workers $WORKERS"
fi

exec gunicorn --chdir backend/api wsgi:app \
    --workers "$WORKERS" \
    --threads "${ALLIN_THREADS:-4}" \
    --timeout "${ALLIN_TIMEOUT:-120}" \
    --bind "${ALLIN_BIND:-0.0.0.0:5000}"
