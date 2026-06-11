# Deployment

> **STATUS: ✅ SHIPPED. Live at https://allin.jianrontan.com as of v1.0.0.**
> CI/CD is automated via GitHub Actions; everyday deploys are `git push`
> triggered. The full operational + historical reference is split across
> the three docs below.

This page is an index. Pick the doc that matches what you're trying to do.

## The three deployment docs

| Doc | What it is | When you want it |
|---|---|---|
| **[DEPLOYMENT_RUNBOOK.md](DEPLOYMENT_RUNBOOK.md)** | Operational reference: env vars, Cognito console setup, the original 11-step launch sequence (preserved as the reproducibility / fallback path), post-launch ops runbooks (rotate IAM key, upgrade blueprint, roll back a deploy), and the post-launch hardening checklist. | You're reproducing the launch, setting up a fresh environment, doing a one-shot ops task, or auditing what runs in prod. |
| **[DEPLOYMENT_HISTORY.md](DEPLOYMENT_HISTORY.md)** | Engineering narrative + case study: cost model + two-phase residency/launch plan, stack glossary (every layer one sentence), original D0–D4 roadmap, security & runtime hardening pass, the v1-surface code map (auth, leaderboard, hand recaps), WebAssembly analysis, +EV leaderboard design. | You're explaining the project in an interview, writing a portfolio narrative, or curious *why* a particular decision was made. |
| **`docs/private/MAINTENANCE.md`** (gitignored, author-only) | Day-to-day operations: feature/fix/chore branch flow, local dev with the venv + Vite + Docker smoke tests, recipes for the 14 common ops tasks (blueprint upgrade, re-bake postflop, IAM key rotation, leaderboard wipe, CF rate-limit tuning, …), versioning conventions, diagnostic recipes for CI / Lightsail / healthz failures, and the "lessons learned during v1.0 launch" pitfall list. | You're the author iterating on the live system. |

## At-a-glance — what's deployed

- **Frontend** — Cloudflare Pages serves the React/Vite SPA at
  `allin.jianrontan.com`. CI/CD: `.github/workflows/frontend-deploy.yml` runs
  `wrangler pages deploy` on `git push origin main` (paths under `frontend/`).
- **Backend** — Flask + gunicorn in a Docker container on AWS Lightsail
  Containers (Micro, 1 node, `ap-southeast-1`), reachable at
  `api-allin.jianrontan.com` behind Cloudflare's edge (TLS, WAF rate limit on
  `/api/strategy/river-solve`). CI/CD: `.github/workflows/backend-deploy.yml`
  builds + tests inside the Dockerfile's `test` stage, pushes the `prod`
  stage to ECR, and rolls a new Lightsail deployment.
- **Persistent state** — DynamoDB with PITR on all four tables (`allin-sessions`
  with TTL, `allin-players`, `allin-global`, `allin-hands`).
- **Auth** — AWS Cognito User Pool federated to Google IdP (implicit grant,
  no client secret on the public SPA).
- **Large artifacts** — `blueprint_final.db` (~37 MB) + `postflop_table_*.npz`
  (~131 MB combined) live as GitHub Release assets on the `assets-v1` tag.
  CI fetches them at image-build time so they ship inside the image without
  bloating the git tree.

## See also

- **[diagrams/infrastructure.puml](diagrams/infrastructure.puml)** — the
  canonical "what's deployed where" architecture diagram (PlantUML).
- **[diagrams/system_architecture.puml](diagrams/system_architecture.puml)** —
  the application-layer diagram (frontend pages, API endpoints, store
  interfaces, game-core wiring).
- **[ROADMAP.md](ROADMAP.md)** — the build arc through v1.0 and the deferred
  Phase-5b/post-launch items.
- **[../backend/bot/docs/BUG_LOG.md](../backend/bot/docs/BUG_LOG.md)** —
  interesting bugs caught during build + launch.
- **[RELEASE_NOTES_v1.0.0.md](RELEASE_NOTES_v1.0.0.md)** — the launch
  announcement copy.
