# Deployment Runbook

**Operational reference** for the AllIn v1.0.0 deployment. Captures the
one-shot launch sequence (preserved for reproducibility + auditing), the
Cognito + AWS console setup, the env-var contract, post-launch ops
runbooks, and the post-launch hardening checklist.

For the **journey + design tradeoffs** (cost model, stack glossary, what
was built and why), see [DEPLOYMENT_HISTORY.md](DEPLOYMENT_HISTORY.md).
For **day-to-day operations** (rotating IAM keys, upgrading the blueprint,
diagnosing CI failures, dev workflow) — that's in
`docs/private/MAINTENANCE.md` (gitignored, author-only).

> **CI/CD is automated.** Everyday deploys are `git push` triggered by
> `.github/workflows/backend-deploy.yml` + `frontend-deploy.yml`. The
> manual 11-step runbook below is the **original one-shot launch
> sequence**, kept as a reproducibility reference and a fallback if CI
> ever breaks.

## Environment variables (deploy-relevant)

- `ALLIN_BLUEPRINT_DB` — explicit path to the blueprint DB (overrides auto-resolution). The
  resolver serves the top-level 25M (`blueprint_final.db`) by default; set this only to override.
- `ALLIN_BLUEPRINT_SOURCE` — where the blueprint comes from: `local` (default, a file —
  what the image bakes in) or `s3` (download once + cache; stub, not wired in v1).
- `ALLIN_BLUEPRINT_S3_URI` — `s3://bucket/key` of the blueprint (only when source = `s3`).
- `ALLIN_CORS_ORIGINS` — comma-separated allowed CORS origins. **In prod set to exactly the
  Cloudflare Pages domain (e.g. `https://allin.jianrontan.com`), never `*`.**
- `ALLIN_SESSION_STORE` — `memory` (default) or `dynamodb`. **Must be `dynamodb` for >1 worker.**
- `ALLIN_STORE_BACKEND` — `memory` (default) or `dynamodb`: backs the **leaderboard** stores
  (players + global counter). Set to `dynamodb` in prod alongside `ALLIN_SESSION_STORE`.
- `ALLIN_DYNAMODB_TABLE` — sessions table name (default `allin-sessions`).
- `ALLIN_PLAYERS_TABLE` / `ALLIN_GLOBAL_TABLE` — leaderboard table names (defaults
  `allin-players` / `allin-global`). Provision via the stores' `create_table_if_missing()`.
- `ALLIN_HANDS_TABLE` — recap-capture table name (default `allin-hands`). Provision via
  `DynamoDBHandStore.create_table_if_missing()`. Backed by `ALLIN_STORE_BACKEND` along with
  the leaderboard stores.
- `ALLIN_DYNAMODB_ENDPOINT` — override endpoint for DynamoDB Local testing (e.g.
  `http://localhost:8000`); unset in prod.
- `AWS_REGION` / `AWS_DEFAULT_REGION` — region for the DynamoDB stores.
- `ALLIN_SESSION_TTL_SECONDS` — session lifetime (default `86400` = 24h; DynamoDB TTL).
- `ALLIN_HANDS_PER_WINDOW` / `ALLIN_HAND_WINDOW_SECONDS` — rolling hand cap (default `500`
  per `3600`s); over it, `/api/game/new` + `/next-hand` return 429 + `Retry-After`.
- `ALLIN_COGNITO_REGION` / `ALLIN_COGNITO_USER_POOL_ID` / `ALLIN_COGNITO_APP_CLIENT_ID` —
  backend, for Google-sign-in ID-token validation. **Optional in dev**: when any is unset,
  `/api/auth/google` returns 503 and gameplay (playerId-routed) is unaffected.
- `ALLIN_DEBUG_OVERLAY` — `1` exposes the per-decision bot trace (`botDebug`, which leaks the
  bot's bucketed hand class mid-hand) in game responses. **In v1.0 the Dockerfile bakes
  `ALLIN_DEBUG_OVERLAY=0` as the secure-by-default**, so the live public bot's bucket can't
  leak even if the Lightsail env var is forgotten. The code default (when run outside Docker)
  remains `1` so the Debug button works in local dev. Override in Lightsail to `1` only for
  one-off debugging.
- `ALLIN_LOG_LEVEL` — log level (default `INFO`).
- `ALLIN_GIT_SHA` — build commit, surfaced in `/api/healthz` (set by CI; absent in dev is fine).
- `ALLIN_BLUEPRINT_CACHE_DIR` — only when `ALLIN_BLUEPRINT_SOURCE=s3`: local dir the
  blueprint is downloaded into and re-opened from. Default is `tempfile.gettempdir()`; in a
  container, set this to a stable mount (e.g. `/var/lib/allin/blueprints`) since `/tmp`
  can be wiped on restart by some orchestrators.
- `ALLIN_RIVER_CACHE_BOARDS` — `PostflopV2._RIVER_BOARD_CACHE` cap (default 100k;
  the Docker image bakes `20000`). Larger → more memory, faster eval; smaller → less
  RAM but more recomputation.
- `ALLIN_TRACKER_BUCKET_BOARDS` — process-global range-tracker bucket cache cap, in
  distinct boards (default `128`, ~32 MB/worker). The tracker re-buckets ~1,300 hands
  per action; this cache shares that work across both trackers / all actions on a
  board / concurrent sessions, so only the first action on a given board pays the cost.
  The dominant per-action cost on a fractional vCPU before this existed (~1–1.5s →
  near-zero after the first action). See `range_tracker.py`.
- `ALLIN_SOLVE_PERMITS` / `ALLIN_EXPLORER_PERMITS` — per-process concurrency caps for
  the live river solver / the explorer's on-demand solve (defaults: `cpu_count - 1`
  and half that, floored at 1). The solver is anytime CFR, so on a small instance
  raising the live cap to 2 lets two concurrent solves both finish (slower) instead
  of one queueing 30s into a 503 — usually the better trade for live play.
- `ALLIN_DEBUG` — dev server only: `1` (default) enables the Werkzeug debugger, `0` disables.
  Irrelevant under gunicorn/waitress (that code path never runs).
- `ALLIN_DEV_HOST` / `ALLIN_DEV_PORT` — dev server bind (default `127.0.0.1:5000`).
- **Gunicorn / entrypoint tuning** (Docker only — read by `docker-entrypoint.sh`):
  - `ALLIN_WORKERS` — override the entrypoint's auto-pick (1 if any memory store, 2 if both DynamoDB).
  - `ALLIN_THREADS` — gunicorn threads per worker (default `4`).
  - `ALLIN_TIMEOUT` — request timeout in seconds (default `120`; matches the river-solve ceiling).
  - `ALLIN_GRACEFUL_TIMEOUT` — graceful shutdown timeout (default `120`).
  - `ALLIN_MAX_REQUESTS` / `ALLIN_MAX_REQUESTS_JITTER` — worker cycling for memory hygiene
    (Docker image bakes `50000` / `50`; the entrypoint fallback is `500`). Sized so the
    LB health checker (~36 req/min of baseline traffic) doesn't churn workers every
    ~15 min — each recycle re-imports the app on a fractional vCPU and stalls in-flight
    requests for ~10s (see MAINTENANCE.md "Common pitfalls").
  - `ALLIN_BIND` — gunicorn listen address (default `0.0.0.0:5000`).
- `VITE_API_BASE` — frontend API base URL (set at build time).
- `VITE_COGNITO_DOMAIN` / `VITE_COGNITO_APP_CLIENT_ID` / `VITE_COGNITO_REDIRECT_URI` —
  frontend (build-time), for the "Sign in with Google" Hosted-UI redirect. **Public values**
  (not secrets). When unset, the sign-in button hides itself. The Google OAuth *client secret*
  is NEVER here — it lives only in the Cognito IdP config in AWS.

## Cognito setup — one-time, in the AWS console

You can't run this remotely; it's a sequence of clicks in two consoles (AWS + Google).
**Do this once, in this exact order** — the Google Client ID has to be created BEFORE the
Cognito IdP can be configured, and the Cognito Hosted UI domain has to exist BEFORE you can
paste its `/oauth2/idpresponse` URL back into Google's allowed-redirect list.

**Step 1 — Google Cloud Console:**
1. Create a new project (e.g. `allin-prod`).
2. **APIs & Services → OAuth consent screen.** External user type. App name `AllIn`,
   support email. Scopes: `openid`, `email`, `profile`. Add your email as a test user
   (or publish if you want anyone to sign in).
3. **Credentials → Create credentials → OAuth client ID.** Type **Web application**,
   name `allin-cognito`. **Leave authorized URIs blank for now** — Cognito will tell you
   the redirect URI to paste in.
4. **Save the Client ID and Client Secret.** You'll paste them into Cognito in Step 4.

**Step 2 — Cognito User Pool (`ap-southeast-1`):**
1. Cognito → Create user pool. Name `allin-users`. Sign-in option: **email**. Required attributes: `email`.
2. Default password policy is fine. MFA: optional or none for v1.
3. **App client:** "Public client". Name `allin-web`. **UNCHECK "Generate a client secret"** —
   the frontend is a public SPA and can't carry a secret.
4. **Hosted UI:** enable. **Cognito domain:** pick a prefix (e.g. `allin-prod`) →
   gives you `allin-prod.auth.ap-southeast-1.amazoncognito.com`. (Custom domain
   `auth.jianrontan.com` is post-launch — needs an ACM cert + DNS.)
5. **Allowed callback URLs:** `https://allin.jianrontan.com/auth/callback` AND
   `http://localhost:5173/auth/callback` (for dev).
6. **Allowed sign-out URLs:** `https://allin.jianrontan.com/` AND `http://localhost:5173/`.
7. **OAuth 2.0 grant types:** **Implicit grant** (the frontend uses `response_type=token`).
8. **Allowed OAuth scopes:** `openid`, `email`, `profile`.
9. Note the **User Pool ID** (e.g. `ap-southeast-1_XXXXXXX`) and the **App Client ID**.

**Step 3 — Cognito → Google IdP federation:**
1. In the User Pool → **Federation → Identity providers → Add Google.**
2. **Google client ID** + **Google client secret**: paste from Step 1.
3. **Authorized scopes:** `profile email openid`.
4. **Attribute mapping:** `email → email`, `sub → username`, `name → name`.
   Also map `email_verified → email_verified` (without this, the backend rejects every sign-in
   because the claim is missing).
5. Save. Back in your App Client settings, **enable Google** under "Enabled identity providers".

**Step 4 — back to Google Cloud Console:**
1. Open your OAuth Client (from Step 1) → **Authorized redirect URIs** → add
   `https://<your-cognito-domain>/oauth2/idpresponse` (the Hosted UI URL from Step 2.4 + that path).
2. Save.

**Step 5 — wire the env vars into Lightsail (and Cloudflare Pages):**

Backend (Lightsail container deployment env):
```
ALLIN_COGNITO_REGION=ap-southeast-1
ALLIN_COGNITO_USER_POOL_ID=ap-southeast-1_XXXXXXX
ALLIN_COGNITO_APP_CLIENT_ID=<App Client ID>
```

Frontend (Cloudflare Pages build env / `frontend/.env.production`):
```
VITE_COGNITO_DOMAIN=<allin-prod>.auth.ap-southeast-1.amazoncognito.com
VITE_COGNITO_APP_CLIENT_ID=<same App Client ID>
VITE_COGNITO_REDIRECT_URI=https://allin.jianrontan.com/auth/callback
```

**Verification:** `/api/healthz` should show your blueprint loaded. The frontend's
"Sign in with Google" button should appear (it hides itself if `VITE_COGNITO_DOMAIN` is
empty). Clicking it should bounce through `<cognito-domain>/oauth2/authorize` → Google →
back to `/auth/callback` with a token. The callback should show "Welcome back, …".

## Deploy runbook — one-shot launch sequence (preserved)

This is the **original 11-step manual launch sequence** used for the v1.0.0 cut.
After v1.0.0, [CI/CD](#cicd-replacement-everyday-deploys) replaced steps 6–7 + 10.
Preserved here verbatim as the reproducibility reference + the fallback if CI breaks.

You've done Step 0 (AWS account + IAM + Cloudflare domain). The rest, in order:

**1. Provision the four DynamoDB tables.** From your laptop with `aws configure` already done:

```powershell
python scripts/provision_dynamodb.py --region ap-southeast-1
```

The script (in `scripts/provision_dynamodb.py`) imports each store class and calls
`create_table_if_missing()` for `allin-sessions`, `allin-players`, `allin-global`,
`allin-hands`. All four are PAY_PER_REQUEST; PITR is enabled idempotently.

**2. Create the ECR repo.**

```powershell
aws ecr create-repository --repository-name allin --region ap-southeast-1
```

Note the URL it prints (`<account>.dkr.ecr.ap-southeast-1.amazonaws.com/allin`).

**3. Create an IAM user for the container runtime.** Console → IAM → Users → Create.
Attach a least-privilege inline policy granting DynamoDB read/write on the four `allin-*` tables.
Save its access key + secret.

(v1.0 launch used `AmazonDynamoDBFullAccess` to ship; the post-launch hardening
checklist below tracks tightening this.)

**4. Cognito setup.** Follow the "Cognito setup" section above.

**5. Create the Lightsail container service.** `ap-southeast-1`, name `allin`, plan
**Micro** for residency.

**6. Build, tag, push the Docker image.**

```powershell
docker build --pull -t allin .
# Smoke-test locally:
docker run -p 5000:5000 -e ALLIN_SESSION_STORE=memory -e ALLIN_STORE_BACKEND=memory allin
# In another shell:
curl http://localhost:5000/api/healthz   # expect 200 + blueprint name

aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin <YOUR-ECR-URL>
docker tag allin:latest <YOUR-ECR-URL>:v1.0.0
docker push <YOUR-ECR-URL>:v1.0.0
```

**7. Deploy to Lightsail.** Lightsail console → your container service → Deployments → Create.
Image = `<YOUR-ECR-URL>:v1.0.0`. Open port `5000` publicly. Health-check path `/api/healthz`.

You also need to grant the Lightsail service permission to pull from your private ECR — the
one-time setup:

```powershell
aws lightsail update-container-service --service-name allin --region ap-southeast-1 `
  --private-registry-access "ecrImagePullerRole={isActive=true}"

# Get the role ARN, then add a repository policy to ECR that allows that role:
aws lightsail get-container-services --service-name allin --region ap-southeast-1 `
  --query "containerServices[0].privateRegistryAccess.ecrImagePullerRole.principalArn"
# Use the ARN in an ECR repository policy with ecr:BatchGetImage + ecr:GetDownloadUrlForLayer.
```

**Environment variables — ALL of these (the ⚠️ ones are easy to forget):**

```
ALLIN_SESSION_STORE=dynamodb
ALLIN_STORE_BACKEND=dynamodb
AWS_REGION=ap-southeast-1
AWS_ACCESS_KEY_ID=<from step 3>
AWS_SECRET_ACCESS_KEY=<from step 3>
ALLIN_CORS_ORIGINS=https://allin.jianrontan.com
ALLIN_COGNITO_REGION=ap-southeast-1
ALLIN_COGNITO_USER_POOL_ID=<from Cognito step 2.9>
ALLIN_COGNITO_APP_CLIENT_ID=<from Cognito step 2.9>
ALLIN_DEBUG_OVERLAY=0                     ⚠️ (now baked OFF in the Dockerfile default, but pin explicitly)
ALLIN_GIT_SHA=v1.0.0
```

**8. Cloudflare Pages for the frontend.** Connect the repo. Root `frontend/`, build
`npm install && npm run build`, output `dist`. Production env (see "Cognito setup → Step 5").

In v1.0 we instead set up the frontend via the GitHub Actions workflow
`.github/workflows/frontend-deploy.yml` calling `wrangler pages deploy` — that path
is also valid, and lets you avoid CF Pages' Git-connect flow if it's been moved/renamed
in CF's dashboard (CF Pages projects created via `wrangler pages deploy` can't be
retroactively Git-connected through the dashboard).

**9. Cloudflare DNS + edge rate limits.** `allin.jianrontan.com` → CF Pages
(CNAME, orange cloud / proxied). `api-allin.jianrontan.com` → Lightsail public URL
(CNAME, orange cloud + Origin Rules + Lightsail-issued ACM cert for the custom domain).

> **Why `api-allin` not `api.allin`:** CF Universal SSL covers `*.jianrontan.com` (one
> level of subdomain) but NOT `*.allin.jianrontan.com` (two levels). Using a flat
> `api-allin.jianrontan.com` stays within the free Universal SSL cert. The Lightsail
> custom-domain feature provisions its own ACM cert for the same hostname, attached
> to the public endpoint — required so origin TLS validates with CF Full mode.

Add a **Cloudflare Rate Limiting rule** at this stage:

| Path | Limit | Reason |
|---|---|---|
| `api-allin.jianrontan.com/api/strategy/river-solve` | **5 req / 10s per IP**, block | CPU DoS (each solve ~50-200 ms; ~20 concurrent saturates Micro) |

The Free plan caps periods at 10s, so the original "10/min" translates to ~5/10s with the
same effective rate. Pro plan ($25/mo) unlocks longer windows + more rules; second priority
is `/api/game/new` at 30/min to defeat +EV-counter farming via rotating UUIDs.

**10. Smoke test.** Open `https://allin.jianrontan.com`. Play a hand. Sign in with Google.
`/api/healthz` should show `debugOverlay: false` + the blueprint loaded.

**11. Tag `v1.0.0`** and draft a GitHub Release with `RELEASE_NOTES_v1.0.0.md`.

### CI/CD replacement (everyday deploys)

After step 11, everyday deploys are `git push origin main` triggered:

- `.github/workflows/backend-deploy.yml` — runs pytest inside the Dockerfile's `test`
  stage, builds the `prod` stage, pushes to ECR with the short commit SHA as tag,
  fetches the existing Lightsail deployment config, swaps the image tag + updates
  `ALLIN_GIT_SHA` (preserving every other env var), creates a new deployment, polls
  `currentDeployment.version` until the new one is promoted, then verifies
  `/api/healthz` reports the new SHA.
- `.github/workflows/frontend-deploy.yml` — `npm install && npm run build` with the
  committed `frontend/.env.production` values, then `wrangler pages deploy`.

AWS auth in both flows is OIDC via the IAM role `github-actions-allin-deploy` — no
long-lived AWS access keys in GitHub Secrets. CF auth uses the `CLOUDFLARE_API_TOKEN`
secret (Pages:Edit scope). See `docs/private/CICD_SETUP.md` for the one-time IAM role
+ trust policy setup.

## Runbooks (post-launch operations)

- **Rotate the IAM access key.** Console → IAM → Users → `allin-runtime` → Security credentials
  → Create a new access key. Paste both into the Lightsail container deployment env (replacing
  the old pair). Save. Once Lightsail confirms the deploy is healthy, deactivate the old key
  back in IAM. Delete it after a few days.

- **Upgrade the blueprint.** Rename the new DB to `blueprint_final.db` locally,
  `gh release upload assets-v1 backend/bot/analysis/blueprints/blueprint_final.db --clobber`,
  then push a trivial backend change (or `gh workflow run backend-deploy.yml`) to trigger CI.
  The Dockerfile pulls from the `assets-v1` release on every build, so the new blueprint
  flows into the next image automatically. **No DB migration is needed** — the blueprint is
  immutable runtime data baked into the image.

- **Re-bake postflop tables (centroid change).** Same flow as blueprint upgrade: bake locally,
  `gh release upload assets-v1 postflop_table_*.npz --clobber`, trigger backend CI.

- **Wipe the leaderboard.** Console → DynamoDB → table → "Delete items" by scanning, OR drop the
  whole table and re-create via `create_table_if_missing()` (PITR-restorable for the next 35d).
  Don't wipe `allin-hands` if you ever want the recap-derived analytics.

- **Diagnose a failed deploy.** First: `curl https://api-allin.jianrontan.com/api/healthz`.
  A 503 with `"status": "degraded"` shows the blueprint load error. A 502/connect-refused means
  the container is restarting — Lightsail console → Containers → Logs. The entrypoint prints
  `[entrypoint] shared store (sessions=dynamodb, backend=dynamodb) -> --workers N` on each
  start; if you see the oscillation symptom, that's `ALLIN_STORE_BACKEND` accidentally on memory.

- **Roll back a bad deploy.** Lightsail console → your container service → Deployments →
  "Deploy a previous version" → pick the last green tag. DynamoDB state survives (the schema
  is forward-compatible for the changes in v1).

## Post-launch hardening checklist

These were flagged HIGH/MED in pre-launch review but were not blockers. Work through them in
the first 1–2 weeks after v1.0:

- **Measure the deployed bot's true exploitability.** The BR evaluator
  (`run_evaluation.py`) reads the raw blueprint table, and the LBR victim model still plays
  the raw blueprint on the river — so neither captures what the *served* bot actually does
  (gadget anchor, untrained-node equity guards, purification). Wire the served river solver
  into LBR's river decision and run a paired BR/LBR pass. The number we publish in
  `docs/ROADMAP.md` is **stale until this is done.** Severity: HIGH (truth-in-numbers).

- ~~**In-code per-IP rate limits as belt-and-suspenders to the CF edge rules.**~~
  ✅ **DONE post-launch (2026-06-12).** `/api/game/new` (20/min/IP) and
  `/api/strategy/river-solve` (10/min/IP) now carry in-code `_rate_limited()`
  floors, so the raw Lightsail URL is no longer an unprotected path. The same
  pass added a healthz store-reachability probe (3-strike 503 so a revoked IAM
  key alerts UptimeRobot) and `ALLIN_SOLVE_PERMITS` / `ALLIN_EXPLORER_PERMITS`
  env overrides for solver concurrency tuning.

- **Drop the `AmazonDynamoDBFullAccess` policy on `allin-runtime`.** The launch IAM user
  has full DynamoDB rights for speed; tighten to a custom inline policy granting only
  `Get/Put/Update/Query/Scan` on `arn:aws:dynamodb:ap-southeast-1:*:table/allin-*`. Severity:
  MED.

- ~~**CI/CD via GitHub Actions.**~~ ✅ **DONE post-launch (2026-06-11).**
  `.github/workflows/backend-deploy.yml` (OIDC → ECR push → Lightsail rolling deploy +
  healthz verify) and `.github/workflows/frontend-deploy.yml` (wrangler pages deploy) now
  drive everyday deploys.

- **ECR lifecycle policy.** Each deploy pushes a new image (~300 MB). Set "keep last 20
  tags" to cap storage cost at a constant ~$0.30/mo. Severity: LOW.

- **Move the IAM access key to AWS Secrets Manager.** Long-lived keys in Lightsail env
  vars are tolerable but not ideal. Once the runtime IAM user has a least-privilege policy
  (above), publish the keys via Secrets Manager and have the entrypoint fetch on boot.
  Severity: LOW.

- ~~**Uptime monitoring + auto-renew the domain.**~~ ✅ **DONE
  post-launch (2026-06-12).** UptimeRobot free tier (permanent; 50-monitor
  cap, using 1) pings `https://api-allin.jianrontan.com/api/healthz` every
  5 min; email alerts on 503/timeout. Public status page at
  `stats.uptimerobot.com/tzx21x76mt`. Cloudflare Registrar auto-renew is
  ON.
