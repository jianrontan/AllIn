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
#
# LOGGING (BUG-030). The 2026-09-22 latency incident was undiagnosable from the
# container log because the access log carried no request DURATION: a 20 ms
# reply and a 20 s reply looked identical. UptimeRobot saw a 3.7 s average and a
# 26 s peak while the log showed an unbroken wall of "200"s, so we could not
# tell whether the container was slow or whether something in FRONT of it was
# dropping requests. %(D)s (microseconds) is the field that answers that, and it
# is overridable via ALLIN_ACCESS_LOGFORMAT.
#
# DO NOT point --error-logfile at /dev/stdout (or any pipe). gunicorn 23 opens
# the error log with open(errorlog, 'a+') (glogging.py:205), and 'a+' requires a
# SEEKABLE file; a container's stdout is a PIPE, so it raises
#     io.UnsupportedOperation: File or stream is not seekable
# during Arbiter setup -- before any worker boots, before anything is logged.
# That is exactly what failed Lightsail deployment v18 on 2026-09-22: the
# container never listened, health checks never passed, and the rolling deploy
# correctly kept the previous version serving. Keep '-' (stderr).
#
# STILL OPEN: whether Lightsail surfaces gunicorn's stderr at all (a 24h filter
# for "INFO" returned zero events). If it does not, app-level _LOG.* WARNING and
# ERROR are invisible -- but the fix for that is a stdout logging handler inside
# the app, NOT an errorlog path. See BUG-030.
ACCESS_FMT="${ALLIN_ACCESS_LOGFORMAT:-%(h)s %(t)s \"%(r)s\" %(s)s %(b)s %(D)s \"%(a)s\"}"

exec gunicorn --chdir backend/api wsgi:app \
    --workers "$WORKERS" \
    --threads "${ALLIN_THREADS:-4}" \
    --timeout "${ALLIN_TIMEOUT:-120}" \
    --graceful-timeout "${ALLIN_GRACEFUL_TIMEOUT:-120}" \
    --max-requests "${ALLIN_MAX_REQUESTS:-500}" \
    --max-requests-jitter "${ALLIN_MAX_REQUESTS_JITTER:-50}" \
    --access-logfile - \
    --access-logformat "$ACCESS_FMT" \
    --error-logfile - \
    --bind "${ALLIN_BIND:-0.0.0.0:5000}"
