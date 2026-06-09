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

# App code + committed centroids + the pinned blueprint snapshot. The
# .dockerignore keeps ONLY the 25M snapshot among the blueprint DBs.
COPY backend/ /app/backend/

# Bake the postflop lookup tables INTO the image (centroids are committed; the
# tables are git-ignored and regenerable). Without them PostflopV2 falls back to
# slow lazy bucketing and /api/healthz.postflopTables reads false.
RUN cd /app/backend/bot \
    && python scripts/bake_postflop_table.py --street flop \
    && python scripts/bake_postflop_table.py --street turn

# Defaults safe for a bare `docker run`. Prod overrides at deploy time (Lightsail
# env): ALLIN_SESSION_STORE=dynamodb, ALLIN_CORS_ORIGINS=https://allin.jianrontan.com,
# AWS_REGION, ALLIN_GIT_SHA, etc. The debug overlay stays OFF (ALLIN_DEBUG_OVERLAY
# unset) so the bot's bucket never leaks in production.
ENV PYTHONUNBUFFERED=1 \
    ALLIN_LOG_LEVEL=INFO \
    ALLIN_SESSION_STORE=memory \
    ALLIN_BLUEPRINT_DB=/app/backend/bot/analysis/blueprints/snapshots/snap_20260604_114512_25550000.db

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
