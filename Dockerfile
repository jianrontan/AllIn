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
# ALLIN_COGNITO_*, etc. The debug overlay stays OFF (ALLIN_DEBUG_OVERLAY unset)
# so the bot's bucket never leaks in production.
# Blueprint resolution: NO ALLIN_BLUEPRINT_DB pin -- the auto-resolver
# (config.resolve_blueprint_path) globs only the top-level analysis/blueprints
# and picks blueprint_final.db, which is the 25M snapshot.
ENV PYTHONUNBUFFERED=1 \
    ALLIN_LOG_LEVEL=INFO \
    ALLIN_SESSION_STORE=memory

EXPOSE 5000

# Run as a non-root user (prod hardening). The app only READS its files at runtime
# (read-only blueprint; state is in-memory/DynamoDB), and COPY leaves them
# world-readable, so an unprivileged user is sufficient.
RUN useradd --create-home --uid 10001 allin
USER allin

# A river solve is CPU-bound (a few seconds); small worker count + threads + a
# generous timeout. The per-process river-solve semaphore caps concurrent solves.
CMD ["gunicorn", "--chdir", "backend/api", "wsgi:app", \
     "--workers", "2", "--threads", "4", "--timeout", "120", \
     "--bind", "0.0.0.0:5000"]
