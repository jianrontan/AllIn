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
# LOGGING (BUG-030). Two blind spots made the 2026-09-22 latency incident
# undiagnosable from the container log:
#
#  1. The access log carried no request DURATION, so a 20 ms reply and a 20 s
#     reply looked identical. UptimeRobot saw 3.7 s average / 26 s peak while
#     the log showed an unbroken wall of "200" -- we could not tell whether the
#     container was slow or something in front of it was. %(D)s (microseconds)
#     closes that; it is the single field that would have answered it.
#  2. gunicorn writes its ERROR log (startup, worker recycles, tracebacks,
#     WORKER TIMEOUT, OOM kills) to STDERR, and Lightsail's log pipeline appears
#     to surface only STDOUT: a `--filter-pattern INFO` over 24h returned ZERO
#     events. (Weak-ish evidence on its own -- the image bakes
#     ALLIN_MAX_REQUESTS=50000, so workers recycle only ~daily and would emit
#     just a couple of "Booting worker" lines in that window -- but app-level
#     _LOG.* WARNING/ERROR goes to stderr too, so the whole error channel was
#     unverifiable.) Point the error log at stdout and add --capture-output so
#     stray print()/tracebacks land in the same stream; after deploy, a filter
#     for "Booting" CONFIRMS whether stderr was the gap.
#
# Both are env-overridable: if /dev/stdout is ever unopenable in the runtime
# (it is /proc/self/fd/1 -- the process owns fd 1, so `USER allin` can write it)
# set ALLIN_ERROR_LOGFILE=- to fall back to stderr WITHOUT a rebuild.
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
    --error-logfile "${ALLIN_ERROR_LOGFILE:-/dev/stdout}" \
    --capture-output \
    --bind "${ALLIN_BIND:-0.0.0.0:5000}"
