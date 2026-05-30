# Deployment

Last updated: 2026-05-30

How AllIn goes from a local Flask + Vite dev setup to a public, online heads-up game
with a live "+EV counter" — the planned LinkedIn launch. For how the system works see
[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md); for the build arc see [ROADMAP.md](ROADMAP.md);
for not-yet-committed ideas (WASM client-offload) see [IDEAS.md](IDEAS.md).

Status legend: ✅ done · 🚧 in progress · 📅 planned

---

## Stack glossary — what each layer is and does

In the order a visitor's request travels through it. The one-sentence path: a visitor hits
`allin.jianrontan.com` → **Cloudflare DNS** routes them → **Cloudflare Pages** serves the React
app to their browser → the app calls the API → which lives in a **Docker container** (run by
**gunicorn**) on a **Lightsail box** → that runs **Flask + the game engine + the blueprint** to
decide the bot's move and reads/writes the **DynamoDB** leaderboard → and the whole AWS side is
defined in **Terraform**.

1. **Domain + DNS — Cloudflare Registrar + DNS.** The registrar is who you rent the name
   `jianrontan.com` from; DNS is the phonebook that turns a name into a server address. A
   subdomain (`allin.`) is just one more phonebook entry — free once you own the apex. Cloudflare
   does both at cost (~$10/yr).
2. **Frontend hosting — Cloudflare Pages.** Hosts the **static** files `npm run build` emits
   (HTML/CSS/JS). Static = no compute; the browser downloads and runs them. Serves `AiGame.jsx`
   etc. on a global CDN. Holds **no game logic and no blueprint** — it calls the API for those.
   One Pages project **per portfolio project** (apex = landing page, each project on its own
   subdomain), $0 each, unmetered bandwidth, auto-HTTPS, deploy-on-git-push.
3. **The boundary — the API client.** `frontend/src/api.js` + `VITE_API_BASE`: the seam where the
   static frontend stops and the live backend begins. Every `/api/...` call crosses from
   Cloudflare to the backend box; `VITE_API_BASE` (set at build) names the backend URL.
4. **Backend compute — Lightsail instance.** The always-on Linux box (CPU + RAM) — the **only**
   piece that runs Python and *thinks*. Runs the Flask API, `GameSession`, `BlueprintStrategy`,
   and (in the launch window) the Phase-4 river solver. Its size is what costs money, because the
   solver is the only real compute.
5. **The container — Docker + gunicorn.** **Docker** packages app + Python + numpy + the blueprint
   `.db` + the baked tables into one reproducible **image** (fixes "works on my laptop"). **gunicorn**
   is the production WSGI server that accepts public HTTP and runs Flask in multiple workers (the
   `python strategy_api.py` dev server is single-threaded, dev-only). ⚠️ The git-ignored baked
   postflop tables (turn ~126 MB) **must be inside the image**, or the bot falls back to slow lazy
   bucketing (see D0).
6. **Image registry — ECR.** A storage locker for built Docker images: build → push here → the
   box pulls from here to run. The handoff between "I built a new version" and "the server runs it."
7. **Datastore — DynamoDB (on-demand).** A managed NoSQL DB (no server to run; pay-per-request,
   ≈free at this scale). Holds the **mutable** runtime data — game sessions (so a hand survives a
   restart) and the leaderboard (`players` / `global` / `sessions`), behind the `SessionStore`
   seam. Note the blueprint itself is **not** here — it's the read-only SQLite `.db` baked into the
   image (layer 5); it never changes at runtime, so it needs no database service.
8. **Infrastructure-as-Code — Terraform.** Describes the AWS resources (Lightsail service, ECR repo,
   DynamoDB tables, IAM) in text; `terraform apply` builds them for real. Reproducible, self-
   documenting, and the portfolio/IaC signal. DNS stays **out** of Terraform — it lives in the
   Cloudflare dashboard (changed maybe twice ever).

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
- Set **`ALLIN_CORS_ORIGINS`** to the real frontend domain; terminate **HTTPS** at the host
  (Lightsail/Pages both bundle it).

### D1 — Frontend 📅
Static Vite build → **Cloudflare Pages** ($0, unmetered bandwidth, auto-HTTPS, deploy-on-push).
Set `VITE_API_BASE` at build time. This **consolidates with the Cloudflare registrar + DNS**
(one vendor for domain + DNS + all frontend hosting) and is the right shape for a **multi-project
portfolio**: apex `jianrontan.com` = a landing-page Pages project, each project on its own
subdomain as a separate Pages deployment, $0 each. (All-AWS alternative: S3 + CloudFront folded
into the same Terraform — more AWS IaC signal, slightly more setup.)

### D2 — Backend + session store (persistent, mandatory) 📅
> **App Runner is retired** — AWS put it in maintenance mode (no new customers after 2026-04-30,
> no new features). Do **not** onboard. AWS's own successor is **ECS Express Mode** (Fargate +
> ALB), but for this workload it's overkill (see Cost).

**AWS Lightsail Containers/instance** (flat-rate, bundled HTTPS, no ALB/NAT cost traps) + a
**`DynamoDBSessionStore`** behind the existing `SessionStore` interface → a stateless API that
scales horizontally safely. In-memory single-box is **off the table** because the public counter
must survive restarts. DynamoDB on-demand is ~free at this scale. **Why Lightsail over Fargate:**
the box is kept always-warm (min-instance = 1) for the live counter + solver, and always-on
favors flat-rate over pay-per-use — Lightsail is cheaper *and* simpler here, and is still
Terraformable (`aws_lightsail_container_service`), so the IaC story is intact. Fargate's
fine-grained autoscaling is its edge, and this human-paced workload doesn't need it. (Non-AWS
value champion: a Hetzner CX22, 2 vCPU/4 GB ~$5/mo, runs the full solver too.)

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

Researched 2026-05-30. (Supersedes the earlier App Runner pricing — App Runner is retired,
see D2.)

**What actually costs money.** In the Phase-4 v1 design, **only river decisions invoke the
solver** — flop/turn/preflop fall back to the cheap blueprint lookup. So a hand has ~0–2
solves, each bounded by the 10s ceiling (realistically 1–several seconds). The workload is
**human-paced** (players act every ~10–30s), so even with 10–20 people "online" the number of
*simultaneous* solves stays tiny. It's a mostly-idle box that occasionally spikes one core —
**not** flat-out 24/7. Because the box is kept **always-warm**, flat-rate pricing beats
pay-per-use, which is why Lightsail wins over Fargate here.

### Two-phase cost model

The key decoupling: the solver is the *only* expensive part — blueprint inference is just
SQLite lookups. So you don't pay launch prices forever. Run a cheap **residency** indefinitely
and bump to a bigger box only during the **launch window**.

| Phase | What's live | Box | Cost |
|---|---|---|---|
| **Long-term residency** (recruiters, indefinite) | Blueprint bot only | tiny always-on | **~$5–10/mo** |
| **Launch window** (LinkedIn push, ~1–2 wks) | + Phase-4 solver behind the +EV counter | bump to 2 vCPU | **~$40/mo**, then scale back |

Backend options at the residency tier:

| Option | Specs | Cost | Runs full solver? |
|---|---|---|---|
| **AWS Lightsail instance** (Docker) | 2 vCPU burstable / 1 GB | **$7/mo** | Blueprint yes; solver via CPU-burst, slower |
| AWS Lightsail Containers Micro | 0.5 vCPU / 1 GB | $10/mo (free 3 mo) | Blueprint yes; solver no |
| **Hetzner CX22** (non-AWS) | 2 vCPU / 4 GB | **~$5/mo** | **Yes — solver too**, no two-phase split needed |
| Oracle Cloud always-free | 4 vCPU / 24 GB ARM | $0 | Yes, but flaky availability |

Notes:
- For the launch window, Lightsail Containers **Medium** (2 vCPU / 4 GB, ~$40/mo flat) is the
  pick; scale back to the residency box after. Lightsail scales by **adding nodes** for the
  unlikely 100–200 burst (the API is stateless behind DynamoDB, so horizontal scaling is safe).
- The rest of the stack is ~free at this scale: **Cloudflare Pages** frontend ($0), **DynamoDB**
  on-demand (~$0 at thousands of hands).
- **All-in:** ~$5–10/mo residency + a ~$40 launch month + ~$10/yr domain.

This **confirms the server-side launch over WASM client-offload**: offloading saves only
single-digit dollars while costing leaderboard integrity (a tampered client could farm the +EV
counter). WASM stays a portfolio flex / load pressure-valve, not day-one (see the WASM section
below and [IDEAS.md §1](IDEAS.md)).

### Domain

Plan: buy the apex **`jianrontan.com`**, point the subdomain **`allin.jianrontan.com`** at the
app. A subdomain is **free** once you own the apex (just a DNS record), and it leaves the apex
free for a portfolio landing page (`jianrontan.com` = you, `allin.jianrontan.com` = the project).

- **Registrar** — Cloudflare is **at-cost ~$10.44/yr** for both registration and renewal
  (zero markup; requires Cloudflare DNS, which is free). Namecheap is ~$6.79 first year /
  ~$13.98 renewal. Avoid GoDaddy (~$22/yr renewal).
- **DNS** — Cloudflare DNS is free (pairs with the registrar); Route 53 is $0.50/mo per hosted
  zone (only worth it if going all-in on AWS).
- **Recommended:** register + DNS at **Cloudflare (~$10/yr flat)**. Point `allin.jianrontan.com`
  at the **Lightsail** backend (custom-domain → bundled HTTPS); the frontend is a **Cloudflare
  Pages** project on its own subdomain (auto-HTTPS). Apex `jianrontan.com` = a portfolio landing
  Pages project; **add future projects as more subdomains**, $0 each.

**All-in:** ~$5–10/mo residency + a ~$40 launch month + ~$10/yr domain.

---

## If you later add WebAssembly — does the stack change?

**Mostly no — WASM is additive, not a different stack.** It doesn't replace any layer; it
**shifts the solver's compute from the backend (layer 4) into the user's browser** (layer 2). The
recommended hybrid (IDEAS §1, option D): the server still builds the two ~1225-combo range vectors
(cheap, but needs the baked tables), and the browser runs **only** the table-free river CFR+ in a
Rust→WASM module. What each layer does differently:

- **Frontend (Cloudflare Pages)** — *gains weight*: ships an extra **Rust→WASM** artifact, runs the
  solve in a **Web Worker** (so the UI doesn't freeze), and needs **COOP/COEP headers** for WASM
  threads (set via a Pages `_headers` file — Pages supports this). Still $0, still Pages.
- **Backend (Lightsail)** — *shrinks permanently*: the expensive solve leaves the server, so it only
  builds range vectors + serves blueprint lookups + the leaderboard. **No launch-window 2-vCPU bump
  needed** — the residency box handles everything, indefinitely. This is the real payoff.
- **DynamoDB, ECR, Docker, Terraform, domain/DNS** — **unchanged**.

So the stack's *shape* is identical; one box gets lighter and one gets a new artifact + headers.
Two caveats that are design constraints, not stack changes:
- **The 126 MB turn table can't ship to the browser** — which is exactly why it's the *hybrid*
  (server builds ranges using the tables; client does only the table-free river solve).
- **Leaderboard integrity:** client-computed bot decisions can be tampered to farm the +EV counter.
  Mitigate by **load-adaptive offload** (solve server-side by default, offload only under real
  burst) or by **excluding client-solved hands** from the leaderboard. Re-evaluate only **after
  Phase 4 ships** and real solve-time/concurrency is measured.

---

## Environment variables (deploy-relevant)

- `ALLIN_BLUEPRINT_DB` — explicit path to the blueprint DB (overrides auto-resolution).
- `ALLIN_CORS_ORIGINS` — comma-separated allowed CORS origins (set to the real frontend domain).
- `VITE_API_BASE` — frontend API base URL (set at build time).
