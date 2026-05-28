# Deployment

Last updated: 2026-05-27

How AllIn goes from a local Flask + Vite dev setup to a public, online heads-up game
with a live "+EV counter" — the planned LinkedIn launch. For how the system works see
[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md); for the build arc see [ROADMAP.md](ROADMAP.md);
for not-yet-committed ideas (WASM client-offload) see [IDEAS.md](IDEAS.md).

Status legend: ✅ done · 🚧 in progress · 📅 planned

---

## Sequencing: solver-first

Deploy **after the Phase-4 river solver ships**, not before. The LinkedIn launch is built
around the *strongest* bot plus a live public **+EV counter** of the bot's lifetime record
vs the field — so the launch waits for the solver (~2–3 weeks out), and the leaderboard is
built in parallel during the fine-tuning + frontend week.

This accepts that **solver-under-burst is the harder scaling case** (CPU/GIL-bound CFR+
solves, up to a 10s/decision ceiling). Blueprint inference is a cheap SQLite lookup; the
solver is the only real compute. See the cost analysis below — it stays cheap because the
workload is human-paced and only *river* decisions solve.

The architecture is already shaped for this: the `game/` core has no Flask imports, and the
`SessionStore` / `BotStrategy` interfaces are thin, so each step below is **additive, not a
rewrite**.

---

## Deployment roadmap

### D0 — Pre-deploy gaps (host-agnostic) 📅
- **Bake the postflop tables on build** (`scripts/bake_postflop_table.py`) or ship the
  `.npz` — they're git-ignored (the turn table is ~126 MB). A fresh box without them falls
  back to slow lazy bucketing. **This is the main deploy gotcha.** (River is runtime-cached
  by design — fine.)
- **Ship the blueprint `.db`** into the image/instance; confirm `resolve_blueprint_path()`
  picks it, or pin `ALLIN_BLUEPRINT_DB`.
- **Real WSGI** (gunicorn), not the Flask dev server. In-memory sessions only survive one
  worker — another reason for the persistent session store (D2).
- Set **`ALLIN_CORS_ORIGINS`** to the real frontend domain; terminate **HTTPS** at the load
  balancer / App Runner / CloudFront.

### D1 — Frontend 📅
Static Vite build → **AWS Amplify Hosting** (or S3 + CloudFront). Set `VITE_API_BASE` at
build time. Scales on the free tier at this audience.

### D2 — Backend + session store (persistent, mandatory) 📅
**AWS App Runner** (container, built-in HTTPS, autoscales) + a **`DynamoDBSessionStore`**
behind the existing `SessionStore` interface → a stateless API that scales horizontally
safely. In-memory single-box is **off the table** because the public counter must survive
restarts. DynamoDB on-demand is ~free at this scale.

### D3 — Polish 📅
Health checks, request logging, a per-IP rate limit on new-session creation (anti-farm), a
per-session hand cap, and a landing-page "what is this" for LinkedIn visitors.

### D4 — Accounts (v1.1, fast-follow) 📅
**Deliberately after launch.** Add the Tier-2 account layer (see Identity tiers below): managed
auth, persistent saved hands + hand review/coach, and the ranked leaderboard. Launch ships
Tier-1 only with the schema already account-ready (`playerId`-as-PK), so this is additive.

> **Caution:** auth is the biggest scope-creep risk in the project (token storage, verification,
> recovery, data-deletion expectations, login UI). It must not block the launch — the +EV
> counter and the solver do not need accounts. Ship Tier 1, then add D4.

---

## +EV leaderboard

A public "bot's lifetime record vs the field" counter plus a per-player leaderboard — the
launch hook.

### Identity tiers (plan of record)

Two tiers, with the account as an *upgrade* of the anonymous identity — never a separate one.

- **Tier 1 — Anonymous (launch).** Client-generated **UUID in localStorage**, *not* IP. (IP
  collapses NAT-shared users together and splits mobile users apart.) First visit → generate a
  UUID + prompt for a handle, persist both; a return visit in the same browser keeps the name
  and record. No auth (friction). IP is kept only as an **abuse signal** (rate limiting), never
  as identity. Erasable/non-portable by design — acceptable for the public counter.
- **Tier 2 — Account (fast-follow, v1.1).** Optional email/OAuth signup unlocks **persistent
  saved hands, hand review/coach, a ranked leaderboard slot, and cross-device** play. The
  account **binds to the existing localStorage UUID** so all previously-played hands carry over
  — `playerId` stays the PK, signup just adds `{email, authProvider, isRegistered}` to the same
  row. **Non-destructive upgrade is the rule:** never mint a fresh identity on signup (a
  "you now have 0 hands" reset is the worst possible moment to lose a user).
- **Why tiered (not all-or-nothing):** anonymous keeps launch friction at zero; once *saved
  hands / review* exist, the account is a feature the user **wants** (persistence), so the
  friction argument flips — auth becomes opt-in value, not a tax.

### Leaderboard split

- **Global +EV counter** — counts **everyone** (anonymous + accounts). Headline "bot vs the
  field" number; maximize volume, so it is **not** account-gated.
- **Ranked leaderboard** — **accounts only**, still gated on the min-hands floor (≥50–100).
  Better than an all-UUID board: kills the clear-storage-to-re-roll abuse path, and a named,
  persistent board is a stronger flex than disposable handles.

So: anonymous play *feeds the counter*; accounts *compete on the board*.
- **Metric** — show net BB (headline) plus **BB/100 over N hands** (the credible +EV claim).
  Source is per-session `human_net` (bot net = −human_net). **Do not AIVAT the public number**
  (misleading for a public scoreboard); show real observed results with an "over N hands"
  qualifier (early variance can read negative).
- **Data model (DynamoDB):**
  - `players` — PK `playerId` → `{handle, hands, netBB, firstSeen, lastSeen}`; account upgrade
    adds `{email, authProvider, isRegistered}` to the **same** row (UUID-as-PK, see tiers above).
  - `global` — single item → `{totalHands, totalNetBB}`
  - `sessions` — PK `sessionId` → serialized `GameSession.data` + `playerId`
  - On each completed hand: update the session, bump the player + global rows.
  - Saved-hands / review (v1.1) reuse the persisted hand-recap object (IDEAS §3), keyed by
    `playerId` — no new identity needed.
- **Auth (v1.1):** use a **managed provider** (Cognito fits — already in AWS; or Clerk/Auth0),
  not hand-rolled passwords. On-brand with the "boring, correct infrastructure" story.
- **Leaderboard rules** — rank by BB/100 **gated on a minimum hand count** (≥50–100) to avoid
  one-hand luckboxes; also surface biggest-net-winner and most-hands cuts. The flex line:
  *"X players, none beat the bot over 100+ hands."* Count **completed** hands only.
- **New API** — `/api/leaderboard` + `/api/stats`; a frontend panel.

---

## Cost

Researched 2026-05-27 (AWS App Runner, us-east; [pricing](https://aws.amazon.com/apprunner/pricing/)).

**What actually costs money.** In the Phase-4 v1 design, **only river decisions invoke the
solver** — flop/turn/preflop fall back to the cheap blueprint lookup. So a hand has ~0–2
solves, each bounded by the 10s ceiling (realistically 1–several seconds). The workload is
**human-paced** (players act every ~10–30s), so even with 10–20 people "online" the number of
*simultaneous* solves stays tiny. It's a mostly-idle box that occasionally spikes one core —
**not** flat-out 24/7.

App Runner rates: **$0.064 / vCPU-hour** (active) + **$0.007 / GB-hour** (memory, charged even
when idle if kept warm), billed per second; compute drops to ~0 when not processing.

| Scenario | Pays for | Cost |
|---|---|---|
| Idle-warm (no traffic) | 4 GB memory only | **~$0.028/hr ≈ $0.67/day ≈ $20/mo floor** |
| Active solving (2 cores busy) | 2 vCPU + 4 GB | ~$0.156/hr |
| **Realistic launch week** (10–20 users, bursty) | mostly memory + occasional compute | **~$1–3/day → ~$10–25 total** |
| Unlikely 100–200 burst day | scale to 4 vCPU / 8 GB during spike | a few $/day while it lasts |

Caveats:
- App Runner **caps at 4 vCPU / 12 GB per instance**. If a single solve ever needs >4 cores to
  hit 10s, move to ECS/Fargate or EC2 — a Phase-5 (turn/flop, bigger trees) concern, not river v1.
- Run with **min-instances = 1 (warm)**: a cold solver mid-hand is bad UX, and the live counter
  wants an always-on box — hence the ~$20/mo floor. `min = 0` drops idle cost to ~$0 but adds
  cold starts.
- The rest of the stack is ~free at this scale: **Amplify** frontend (free tier), **DynamoDB**
  on-demand (~$0 at thousands of hands).

This **confirms the server-side launch over WASM client-offload**: offloading saves only
single-digit dollars while costing leaderboard integrity (a tampered client could farm the +EV
counter). WASM stays a portfolio flex / load pressure-valve, not day-one. See [IDEAS.md §1](IDEAS.md).

### Domain

Plan: buy the apex **`jianrontan.com`**, point the subdomain **`allin.jianrontan.com`** at the
app. A subdomain is **free** once you own the apex (just a DNS record), and it leaves the apex
free for a portfolio landing page (`jianrontan.com` = you, `allin.jianrontan.com` = the project).

- **Registrar** — Cloudflare is **at-cost ~$10.44/yr** for both registration and renewal
  (zero markup; requires Cloudflare DNS, which is free). Namecheap is ~$6.79 first year /
  ~$13.98 renewal. Avoid GoDaddy (~$22/yr renewal).
- **DNS** — Cloudflare DNS is free (pairs with the registrar); Route 53 is $0.50/mo per hosted
  zone (only worth it if going all-in on AWS).
- **Recommended:** register + DNS at **Cloudflare (~$10/yr flat)**, CNAME `allin.jianrontan.com`
  → the App Runner URL (App Runner's custom-domain feature gives auto HTTPS).

**All-in:** ~$10–30 for the active launch week + ~$10/yr domain.

---

## Environment variables (deploy-relevant)

- `ALLIN_BLUEPRINT_DB` — explicit path to the blueprint DB (overrides auto-resolution).
- `ALLIN_CORS_ORIGINS` — comma-separated allowed CORS origins (set to the real frontend domain).
- `VITE_API_BASE` — frontend API base URL (set at build time).
