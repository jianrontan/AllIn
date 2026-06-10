# syntax=docker/dockerfile:1
# Production image: gunicorn serving the Flask API against the pinned 25M
# blueprint snapshot, with the postflop lookup tables baked in. ~200-250 MB.
#
# Build (from repo root):   docker build -t allin .
# Run (local smoke test):   docker run -p 5000:5000 \
#                             -e ALLIN_SESSION_STORE=memory allin
#                           then GET http://localhost:5000/api/healthz
#
# 3.12-slim matches the project's Python (README). If a wheel (e.g. phevaluator)
# lacks a 3.12 build, either pin 3.11-slim or add build-essential before pip.
FROM python:3.12-slim

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
ENV PYTHONUNBUFFERED=1 \
    ALLIN_LOG_LEVEL=INFO \
    ALLIN_SESSION_STORE=memory \
    ALLIN_DEBUG_OVERLAY=0

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
