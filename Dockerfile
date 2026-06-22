# syntax=docker/dockerfile:1
# Production image: gunicorn serving the Flask API against the pinned 25M
# blueprint snapshot, with the postflop lookup tables baked in. ~200-250 MB.
#
# Build (from repo root):   docker build -t allin .
# Run (local smoke test):   docker run -p 5000:5000 \
#                             -e ALLIN_SESSION_STORE=memory allin
#                           then GET http://localhost:5000/api/healthz
#
# Multi-stage layout:
#   base -> shared layers (prod deps + code) — never tagged directly
#   test -> base + dev deps; used by CI to run pytest against the same image
#           the prod stage extends. Locally:
#               docker build --target test -t allin:test .
#               docker run --rm -w /app/backend/bot --entrypoint python \
#                 allin:test -m pytest tests/ -q
#   prod (default, last stage) -> base + non-root user + entrypoint. This is what
#           a plain `docker build .` produces, and what ships to Lightsail.
#
# 3.12-slim matches the project's Python (README). If a wheel (e.g. phevaluator)
# lacks a 3.12 build, either pin 3.11-slim or add build-essential before pip.

# === BASE STAGE — prod deps + code, no entrypoint ===
FROM python:3.12-slim AS base

WORKDIR /app

# Do not write .pyc files at runtime. The container's source tree is owned by
# root (from COPY) and run by `USER allin`; without this flag Python tries to
# create __pycache__/ next to source files and either silently fails (warnings
# in logs) or writes into a per-user cache (~/.cache) that's wiped on restart.
ENV PYTHONDONTWRITEBYTECODE=1

# Python deps first (cached layer). gunicorn / waitress / boto3 are pinned in
# requirements.txt, so no extra installs here.
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# App code + committed centroids + the 25M `blueprint_final.db` + the
# precomputed baked postflop tables. The .dockerignore keeps everything except
# the snapshots/ tree and other regenerable artifacts (training curves etc.).
# No bake step: the tables are already on disk and shipped as-is.
COPY backend/ /app/backend/

# Make every file under backend/ world-readable, directories world-traversable.
# Why this is a RUN step (and not just a chmod in CI before COPY): Docker's
# buildx layer cache is content-hashed, and when the source files are byte-
# identical to a previous build, the cached COPY layer is reused with whatever
# permissions THAT build had. CI's `gh release download` creates files with
# mode 0600, so without this RUN the cached layer still carries 0600 root:root
# files -- and `USER allin` (UID 10001) can't open them, the blueprint load
# fails, and the container 503s health checks. The RUN command is itself a
# new deterministic layer so the chmod always takes effect.
# `a+rX` (capital X) means: add read for all, and add execute for all only on
# things that ALREADY have execute somewhere (i.e. directories), so the .db
# and .npz data files don't accidentally become executable.
RUN chmod -R a+rX /app/backend/

# Switch the embedded blueprint from WAL to DELETE journal mode. The training
# process uses WAL so it can read-while-write during checkpoints; in prod
# nothing ever writes, and WAL mode makes SQLite require write access to the
# DB's directory (to create/manage the -wal/-shm sidecars) even when we open
# with `mode=ro`. That fails under `USER allin` and bricks the deploy. Doing
# the conversion at BUILD time (as root, before USER allin) leaves the local
# blueprint file on the developer's disk untouched -- only the image copy is
# converted. `wal_checkpoint(TRUNCATE)` first to flush any pending WAL data,
# then PRAGMA journal_mode=DELETE to remove the WAL flag from the header.
RUN python -c "import sqlite3; \
    p = '/app/backend/bot/analysis/blueprints/blueprint_final.db'; \
    c = sqlite3.connect(p); \
    c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); \
    print('journal_mode:', c.execute('PRAGMA journal_mode=DELETE').fetchone()); \
    c.commit(); c.close()"


# === TEST STAGE — base + dev deps; used by CI only ===
# Extends base so the test environment is bit-identical to prod for all prod
# imports — tests + prod resolve the same Python wheels, same code tree, same
# OS. Dev deps (pytest, moto, hypothesis) are added on top; they never enter
# the prod image. The test stage runs as root and has no entrypoint so CI can
# `docker run --entrypoint python allin:test -m pytest ...`.
FROM base AS test

COPY backend/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt


# === PROD STAGE (default — the image that ships) ===
FROM base AS prod

# Defaults safe for a bare `docker run`. Prod overrides at deploy time (Lightsail
# env): ALLIN_SESSION_STORE=dynamodb, ALLIN_STORE_BACKEND=dynamodb,
# ALLIN_CORS_ORIGINS=https://allin.jianrontan.com, AWS_REGION, ALLIN_GIT_SHA,
# ALLIN_COGNITO_*, etc.
# Blueprint resolution: NO ALLIN_BLUEPRINT_DB pin — the auto-resolver
# (config.resolve_blueprint_path) globs only the top-level analysis/blueprints
# and picks blueprint_final.db, which is the 25M snapshot.
#
# Debug overlay OFF by default IN THE IMAGE (secure-by-default): the per-decision
# bot trace (`botDebug`) exposes the bot's hand-bucket mid-hand (a spoiler), so the
# public artifact must not ship it even if the Lightsail env var is forgotten. The
# strategy_api.py CODE default stays ON for local dev (`python strategy_api.py`); a
# local container that wants the overlay can `docker run -e ALLIN_DEBUG_OVERLAY=1`.
# ALLIN_MAX_REQUESTS: gunicorn's per-worker recycle threshold. The entrypoint
# default used to be 500, which sounds generous until you count health checks:
# Lightsail's LB probes /api/healthz every 5s from 3 addresses (~36 req/min),
# burning 500 in ~15-30 min per worker. Each recycle re-imports the whole app
# (blueprint + 127MB postflop table) on a fractional vCPU, pinning the CPU for
# tens of seconds and stalling the OTHER worker's in-flight game requests --
# observed as intermittent ~10s non-river actions with a single player. 50000
# recycles roughly daily, which is still plenty for leak hygiene.
# ALLIN_RIVER_CACHE_BOARDS: with workers now long-lived, bound the one cache
# that grows (river-board equity LRU): 20k boards ~= 52MB/worker vs the 100k
# default's ~260MB -- the right trade on a 1GB instance with 2 workers.
# ALLIN_SOLVE_PERMITS/ALLIN_EXPLORER_PERMITS=1: cap concurrent river solves at 1 on this fractional
#   vCPU. The code default is cpu_count()-1, but Lightsail reports the HOST's cores (not the ~0.25
#   vCPU allocation), so the auto-default over-subscribes and multiple 24s solves thrash one core.
# --- Phase 6 exploitation (flag-gated, SECURE-OFF in the image) ---
# ALLIN_MMAP_POSTFLOP: memory-map the baked tables (extract members to .npy + mmap them) to keep the
#   ~107MB turn `ids` array off the heap. OFF in the image -- the instance already handles the full
#   load, and the extraction needs a WRITABLE table dir (the image's is read-only, so it would just
#   fall back to the full load). It's a dev/measurement aid on a writable, RAM-tight box. NOTE:
#   np.load(mmap_mode=) does NOT mmap .npz members -- only the extracted .npy work.
# ALLIN_GADGET_ANCHOR=auto: the SAFE re-solving anchor (provably <= blueprint). NEVER set 'belief' in
#   prod -- that gives up the safety floor; 'belief' is for LOCAL single-player exploit testing only.
# ALLIN_EXPLOIT: OFF in the image (secure-by-default, like ALLIN_DEBUG_OVERLAY). Enabling it in prod is
#   NOT just a flag flip: ALLIN_EXPLOIT=1 needs the SERVED blueprint and the shipped opponent_models/ to
#   share the SAME abstraction. This image serves blueprint_final (20/16); the models were fit on the
#   30/24 retrain snapshot, so exploit would hit HumanModel's abstraction guard and SELF-DISABLE. To run
#   it in prod, EITHER ship+serve the 30/24 blueprint (un-ignore snapshots/, pin ALLIN_BLUEPRINT_DB) OR
#   re-fit the opponent models on blueprint_final. The models DO ship (not .dockerignored).
# (Real-time TURN solving is REMOVED/dead -- ALLIN_TURN_SOLVE is no longer read; serving is river-only.)
ENV PYTHONUNBUFFERED=1 \
    ALLIN_LOG_LEVEL=INFO \
    ALLIN_SESSION_STORE=memory \
    ALLIN_DEBUG_OVERLAY=0 \
    ALLIN_MAX_REQUESTS=50000 \
    ALLIN_RIVER_CACHE_BOARDS=20000 \
    ALLIN_MMAP_POSTFLOP=0 \
    ALLIN_GADGET_ANCHOR=auto \
    ALLIN_EXPLOIT=0 \
    ALLIN_SOLVE_PERMITS=1 \
    ALLIN_EXPLORER_PERMITS=1

EXPOSE 5000

# Run as a non-root user (prod hardening). The app only READS its files at runtime
# (read-only blueprint; state is in-memory/DynamoDB), and COPY leaves them
# world-readable, so an unprivileged user is sufficient.
RUN useradd --create-home --uid 10001 allin

# Entrypoint chooses --workers based on the store backend: 1 worker for the
# in-memory default (per-process dicts can't be shared, so >1 worker oscillates),
# 2 workers when both stores point at DynamoDB (the prod config). Override via
# ALLIN_WORKERS=<N>. See docker-entrypoint.sh.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# Defensive CRLF strip: if the entrypoint was checked out on Windows it has \r\n
# line endings, and Linux's exec(2) will fail with "no such file or directory"
# trying to run the shebang (/bin/sh\r doesn't exist). sed normalises to LF.
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
 && chmod +x /usr/local/bin/docker-entrypoint.sh

USER allin
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
