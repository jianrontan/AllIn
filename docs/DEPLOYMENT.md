# Deployment

Last updated: 2026-06-07 (was 2026-05-30; this update folds in the M0–M4 turn-solver
findings, fresh cost/WASM research, and the bake decision).

> **⚠️ SUPERSEDED LATER THE SAME DAY (2026-06-07) — read this first.** The "Update 2026-06-07"
> section below committed to building a **baked n=64 turn solver**. That decision was **reversed**
> hours later when the turn solver's **N0 real-game gate FAILED** (it lowered exploitability but
> did NOT beat the blueprint head-to-head). **The turn bake and the NN leaf are NOT being built.**
> The shipped plan is **Rung 1 = the 25M blueprint + the river subgame solver only** (see the
> "Sequencing" section below and ROADMAP Phase 4 / NN_LEAF_PLAN.md). Treat the turn-bake bullets
> in the next section as historical, not the plan of record.

How AllIn goes from a local Flask + Vite dev setup to a public, online heads-up game
with a live "+EV counter" — the planned LinkedIn launch. For how the system works see
[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md); for the build arc see [ROADMAP.md](ROADMAP.md);
for not-yet-committed ideas (WASM client-offload) see [IDEAS.md](IDEAS.md).

Status legend: ✅ done · 🚧 in progress · 📅 planned

---

## Update 2026-06-07 — what's changed since the original plan

- **The deployable bot = blueprint + RIVER solver (Slumbot-class).** Measured serving
  footprint **~200 MB** (river solve incl. the 126 MB postflop tables). Turn-based, mostly
  cheap blueprint lookups, no per-user state → exactly the cheap-to-serve profile (Slumbot
  is the same blueprint+real-time-solver shape, served free online). **Ready now**; the
  cost section below holds.
- **The served default is the 25M blueprint, not the over-trained 30M.** The capped 30M run
  over-trained past its best point; 25M is least exploitable on BR+LBR. The 25M is the
  top-level `blueprint_final.db` (served by default); the 30M is kept in `snapshots/`, which
  the resolver doesn't search. Pin `ALLIN_BLUEPRINT_DB` only to override (see the
  `capped-blueprint-ship-25m` memory).
- **The TURN solver is PAUSED.** It's proven in the lab (M0–M2: ~99% less exploitable at
  high fidelity) but a *live* turn solve is **~20–54 s** (building the leaf value table
  dominates), vs the river solver's ~8 s. Serving it needs one of:
  - **Bake the leaf matrices offline → live lookup** (live solve drops to ~1–2 s). Use
    **n=64** (~5 GB; passed the gate as well as n=128's ~18 GB). **Key cost point: the bake
    is MEMORY-MAPPED from SSD, so the box does NOT need 5 GB of RAM** — only the ~12 leaves
    a solve touches page in (a few ms of disk read, then cached). So a **~$5/mo Hetzner CX22
    (4 GB RAM / 40 GB SSD)** or a ~$10–20/mo Lightsail serves it — NOT a $74/mo big-RAM box.
    One-time bake: ~5 GB, parallelizable + resumable → free-but-slow on an 8-core laptop
    (~a few days, run in chunks) or ~$20–60 fast on a rented many-core spot box.
  - **A neural-net (DeepStack-style CFV) leaf** — small + instant + WASM-able; the proper
    long-term fix (enables turn-in-browser), but a real ML project. See
    `poker-bot-deployment-feasibility` memory.
  - **DECISION (2026-06-07, user) — REVERSED same day.** The original call was to pursue the
    BAKED n=64 turn solver on a cheap SSD box. It was abandoned hours later when the turn solver
    **failed the N0 real-game gate** (−611 mbb vs the blueprint, while the river-only stack won
    +1801 mbb). **No turn bake, no NN leaf.** The "residual value risk" flagged here is exactly
    what N0 confirmed: lower exploitability did not translate to real-game EV. Ship Rung 1 (25M
    blueprint + river solver) instead. See the "Sequencing" section below.
- **WASM is feasible (precedent found): WASM Postflop** is an open-source CFR Hold'em
  solver running in-browser (Rust→WASM, ~2× native, multithreaded, ~16 GB browser limit).
  So a **river-WASM** bot (solve client-side, zero server cost, ~infinite scale) is real —
  but the prior tradeoff stands: **client results are tamperable**, so they can't power a
  *trustworthy* +EV leaderboard (keep that server-side / exclude client-solved hands). The
  ~5–18 GB turn bake can't ship to a browser → WASM turn solving needs the NN leaf. See the
  WASM section below.

**Decisions (2026-06-07, user):** budget **~$10–15/mo sustained, ~$30–40 launch month** max;
host at **allin.jianrontan.com** (+ ~$10/yr domain). **Server-side** (keeps the +EV leaderboard
trustworthy). **WASM deferred.** ~~Turn solver IN v1 via a baked n=64 leaf~~ — **REVERSED the same
day after the N0 gate failed; v1 = the 25M blueprint + the river solver only** (no turn bake). The
turn/flop solver is shelved pending a continual-re-solving redesign. See "Sequencing" below.

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

## Sequencing: deploy the river bot NOW (don't wait for turn/flop)

**Resolved 2026-06-07:** the river subgame solver is the solver we ship — it's done and
wins on both exploitability and real-game EV. The **turn/flop** solver is **deferred** (it
failed the N0 real-game gate — see ROADMAP Phase 4 case study — and needs a continual-re-
solving rebuild). So **do NOT wait for more solver work**; deploy `25M blueprint + river
solver` now. The LinkedIn launch is built around this bot plus the live **+EV counter**;
the leaderboard is built in parallel during the fine-tuning + frontend week.

(Original note, still valid: deploy *after the river solver ships, not before* — which is
now satisfied.)

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
- **Ship the blueprint `.db`** into the image/instance. The image's `.dockerignore` keeps only
  the 25M snapshot, and the resolver serves the top-level 25M (`blueprint_final.db`) by default,
  so no pin is required; set `ALLIN_BLUEPRINT_DB` only to override (see the
  `capped-blueprint-ship-25m` memory).
- **Real WSGI** (gunicorn/waitress), not the Flask dev server. ✅ **Done in code** —
  `backend/api/wsgi.py` (see "Security & runtime hardening" below). In-memory sessions only
  survive one worker — another reason for the persistent session store (D2).
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
scales horizontally safely. ✅ **`DynamoDBSessionStore` is implemented** (TTL + lease lock,
`ALLIN_SESSION_STORE=dynamodb`; see "Security & runtime hardening"); what remains is
provisioning the table and setting the env vars. In-memory single-box is **off the table** because the public counter
must survive restarts. DynamoDB on-demand is ~free at this scale. **Why Lightsail over Fargate:**
the box is kept always-warm (min-instance = 1) for the live counter + solver, and always-on
favors flat-rate over pay-per-use — Lightsail is cheaper *and* simpler here, and is still
Terraformable (`aws_lightsail_container_service`), so the IaC story is intact. Fargate's
fine-grained autoscaling is its edge, and this human-paced workload doesn't need it. (Non-AWS
value champion: a Hetzner CX22, 2 vCPU/4 GB ~$5/mo, runs the full solver too.)

### D3 — Polish 📅
Health checks, request logging, a per-IP rate limit on new-session creation (anti-farm), a
per-session hand cap, and a landing-page "what is this" for LinkedIn visitors. (Body-size cap,
the river-solve concurrency cap, and stateless ownership checks already shipped — see "Security
& runtime hardening"; the **per-IP rate limit** is the remaining anti-farm item, best done at
the Cloudflare edge.)

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

## Security & runtime hardening (implemented 2026-06-08)

A read-only security review (C1–C3 critical, H1–H4 high, M1–M4 medium) was actioned. What
shipped, and how it changes deployment:

- **C1 — real WSGI entrypoint.** `backend/api/wsgi.py` exposes `app` (and a `create_app()`
  factory). `python strategy_api.py` is now **dev-only**: it binds **loopback** (`127.0.0.1`)
  and gates the Werkzeug debugger behind `ALLIN_DEBUG` (default on for dev). Production never
  runs that block — it imports `wsgi:app`. Run commands:
  ```bash
  # Linux deploy box:
  gunicorn --chdir backend/api wsgi:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:5000
  # Windows / cross-platform (test the prod server locally):
  waitress-serve --listen=0.0.0.0:5000 --call wsgi:create_app   # or: python backend/api/wsgi.py
  ```
  Worker sizing: a river solve is CPU-bound (a few seconds), so keep `--workers ≈ CPUs` with
  threads and a generous `--timeout`. A per-process semaphore caps concurrent solves (H2).
- **C2 — persistent, multi-worker session store.** `DynamoDBSessionStore` + a
  `make_session_store()` factory selected by `ALLIN_SESSION_STORE` (`memory` default for
  dev/tests, `dynamodb` for prod). Native DynamoDB **TTL** on the `expiry` attribute expires
  stale games; a **lease-based conditional-write lock** preserves the `with SESSIONS.lock(...)`
  contract across workers/boxes. **You MUST set `ALLIN_SESSION_STORE=dynamodb` whenever you run
  >1 worker** — the in-memory store is per-process and would split games across workers.
  One-time table provisioning: `DynamoDBSessionStore.create_table_if_missing(table, region)`
  (on-demand billing, TTL enabled on `expiry`). Test locally without AWS via DynamoDB Local +
  `ALLIN_DYNAMODB_ENDPOINT=http://localhost:8000`.
- **C3 — session ownership.** Every game request must carry the `playerId` the session was
  created with; a mismatch returns 404 (we don't confirm a session id to a non-owner). The
  frontend sends it automatically (`api.setPlayerId`, persisted in localStorage).
- **H1/H2/H3 hardening.** `get_json(silent=True)` everywhere (malformed body → 400, never a
  500/traceback); `MAX_CONTENT_LENGTH=64 KB` (oversized → 413); a river-solve **semaphore**
  (`max(1, CPUs−1)` permits/process, 503 when saturated); dropped `supports_credentials` from
  CORS (auth is stateless). Per-IP **rate limiting** is still expected at the **Cloudflare**
  edge (D3) — not done in-app.
- **M3** — `BlueprintDB` now uses **thread-local** SQLite connections (the Flask thread pool no
  longer shares one connection). **M4** — frontend error message uses `API_BASE`, not a
  hardcoded `:5000`. **M2** (the debug overlay leaking the bot's bucket) is **intentionally left
  in** as an inspection feature — gate it before a competitive deploy.
- **M1** — `requirements.txt` now includes `gunicorn`, `waitress`, and `boto3` (it was already
  pinned; the review's "unpinned" flag was wrong).

This closes the D0 "real WSGI" gap and the D2 "DynamoDBSessionStore (mandatory)" gap in code;
provisioning the table + setting the env vars below is what remains at deploy time.

## v1 surface — the deployment-prep code (added 2026-06-09)

Architectural tour of everything *added on top of* the existing engine
(training, blueprint, river solver, GameSession, range tracker) to make the
bot servable as a public website with a +EV leaderboard and Google sign-in.

### Module map

New files. All wire through factory functions selected by env vars — nothing
hard-codes the prod backend.

| Layer | New file | What it does |
|---|---|---|
| **Storage** | `bot/src/storage/blueprint_source.py` | Where the blueprint comes from. `LocalFileSource` (file baked into image) is the only impl wired in v1; `S3ObjectSource` stubbed for later. Factory: `make_blueprint_source()` from `ALLIN_BLUEPRINT_SOURCE`. |
| **Leaderboard data** | `bot/src/game/player_store.py` | Per-player rows: handle, lifetime hands/netBB, rolling hand-cap window, optional linked account. `InMemory*` (dev) + `DynamoDB*` (prod). Factory: `make_player_store()` from `ALLIN_STORE_BACKEND`. |
| | `bot/src/game/global_stats_store.py` | The single "+EV counter" row: `{totalHands, totalNetBB, totalPlayers}`. Same factory pattern. |
| | `bot/src/game/hand_store.py` | Per-hand recap capture (added 2026-06-09). One PutItem per completed hand inside the same `_record_hand_end` hook. Write-only in v1 — no UI/coach consumer yet; the point is to have the data when those features ship. `recap_from_session()` builds the recap dict from `GameSession.data` at `hand_over`; cards in display format, full `actionLog` across streets, result + before/after net P/L, blueprint snapshot tag. Schema: PK `playerId`, SK `<13-digit ms-epoch>#<sessionId>#<handNumber>` so a single Query (`ScanIndexForward=False`) returns newest-first. |
| **Auth** | `api/auth.py` | Cognito ID-token validation (PyJWT + JWKS cache). `is_configured()` is the dev-vs-prod gate; `verify_cognito_id_token()` is the validator; `@require_account` is the (unused-in-v1) decorator stub. |
| **Frontend config** | `frontend/src/config.js` | Cognito public config + `hostedUiUrl()` (the Hosted UI redirect URL builder). |
| **Frontend UI** | `components/EvCounter.jsx` | +EV counter widget, polls `/api/stats` every 30s. |
| | `components/Leaderboard.jsx` | The signed-in leaderboard (Home renders one cut: accounts with 50+ hands). |
| | `components/UsernameModal.jsx` | Required unique-username picker shown on sign-in (pre-filled from the Google name). |
| | `components/LoginPrompt.jsx` | Optional "sign in to join the leaderboard" popup. |
| | `components/GoogleSignInButton.jsx` | Shows a DISABLED button if Cognito unconfigured (so its placement is visible); else redirects to Hosted UI. |
| | `components/IntroModal.jsx` | First-visit popup. Gated on `allin.introDismissed`. Wired into `AiGame.jsx` so it triggers the first time someone plays, not on the marketing page. |
| | `pages/AuthCallback.jsx` | `/auth/callback` route. Reads ID token from URL fragment, calls `/api/auth/google`, scrubs the token from the URL bar. |

Modified files: `api/strategy_api.py` (hand-end hook, hand-cap gate, the four
new endpoints, `_redact_view` for the debug overlay), `api/wsgi.py`
(`logging.basicConfig` once), `game/session_store.py` (env-driven TTL),
`game/game_session.py` (`max_raises_per_street` param so tests can hit the
trained cap without disturbing live serving's `inf`), `frontend/src/api.js`
(localStorage UUID + cached account state + new endpoints),
`frontend/src/App.jsx` (the `/auth/callback` route), `frontend/src/pages/Home.jsx`
(EvCounter + the signed-in Leaderboard + GoogleSignInButton). Sign-in adopts the
canonical playerId and prompts for a unique username (UsernameModal).

### Stores — what's actually stored

`PlayerStore` row (DynamoDB table `allin-players`, PK `playerId`):

```
{
  playerId,                              # the browser's localStorage UUID
  handle,                                # display name, validated
  hands, netBB,                          # lifetime (human perspective; bot = -netBB)
  firstSeen, lastSeen,                   # epoch seconds
  window_start, hands_in_window,         # rolling 500/1h cap
  isRegistered,                          # True once an account links
  # added by link_account():
  email, authProvider, providerSub,      # set on Google sign-in
  merged_into,                           # set if this row was merged into another
}
```

`GlobalStatsStore` row (table `allin-global`, PK `statId='global'`):

```
{ totalHands, totalNetBB, totalPlayers }
```

Both DynamoDB impls use `UpdateItem ADD/SET` — atomic at the table level, no
read-modify-write races. The hand-cap window reset is a *conditional* update
(only fires when `window_start < now - 3600`), so two concurrent hands
resetting the window can't double-reset.

### The hand-end hook — how a hand becomes a leaderboard bump

The bump lives in the **API layer** (`strategy_api._record_hand_end`), not in
`GameSession`. The engine stays transport-agnostic; the API knows about stores.

It fires on the `status != 'hand_over' → status == 'hand_over'` transition,
inside the per-session lock. Three properties this gets right:

1. **Idempotent.** A retried `/api/game/action` for the same hand: the second
   call sees `pre_status == 'hand_over'` and returns early. One hand → one bump.
2. **Concurrent-safe.** The session lock guarantees only one request at a time
   owns the transition.
3. **Failure-safe.** A store outage is logged and swallowed — the hand still
   ends correctly for the player. Counters can drift down (a missed bump) but
   never up (a double bump).

Plus a separate one-shot at `/api/game/new`: `PLAYERS.create_if_absent()`
returns `True` exactly once per playerId; that's when `GLOBAL.record_new_player()`
fires.

The **hand cap** (500 hands per rolling 1h, per playerId) gates `/api/game/new`
and `/api/game/next-hand` with a 429 + `Retry-After`. An in-flight hand is
never interrupted — only the next deal is refused.

### Auth — from-scratch explanation

The user signs in with their existing Google account. We never touch their
password. The whole flow rests on three open standards (**OAuth 2.0**,
**OpenID Connect / OIDC**, **JWT**) and one AWS service (**Cognito**).

#### Vocabulary

- **OAuth 2.0** — protocol for "let app A use my account from app B without
  giving A my password." Browser-redirect dance: A sends the user to B, B asks
  the user to authorize, B redirects back to A with a token.
- **OpenID Connect (OIDC)** — a thin layer *on top of* OAuth 2.0 that adds
  *identity*: "who is this user?", as opposed to OAuth's "what may A do on
  behalf of the user?". OIDC introduces the **ID token**.
- **JWT (JSON Web Token)** — a token format: three base64 segments separated
  by dots (`header.payload.signature`). The payload is JSON claims (who,
  audience, when issued, when expires); the signature proves the issuer
  signed it. **Signed, not encrypted** — anyone can decode and read the
  payload; only the issuer can produce a valid signature.
- **ID token** — a JWT *whose payload says who the user is*: `sub` (subject,
  a stable per-user ID), `email`, `iss` (issuer URL), `aud` (audience — *who*
  the token is for), `exp` (expiry), `token_use: "id"`.
- **JWKS (JSON Web Key Set)** — the issuer's *public keys*, fetched at a
  well-known URL. Used to verify signatures. The issuer rotates these
  periodically.
- **Cognito User Pool** — AWS's managed identity database + authentication
  endpoints. Stores users, can federate to external IdPs (Google, Apple,
  etc.), exposes a **Hosted UI** that handles the OAuth flow for you, mints
  the ID tokens.
- **Hosted UI** — the AWS-hosted login page. You redirect the user there;
  they sign in (via Google in our case); AWS redirects back to your callback
  URL with the token in the URL.

#### What our flow actually does

```
1. Anonymous user clicks "Sign in with Google" on Home
   → frontend builds the Hosted UI URL via config.hostedUiUrl()
   → browser navigates to https://<cognito-domain>/oauth2/authorize?...

2. Cognito Hosted UI sees identity_provider=Google in the URL
   → it redirects the user to Google's standard OAuth screen

3. User authenticates with Google
   → Google redirects back to Cognito with an internal authorization code

4. Cognito mints an ID token (a JWT signed by the User Pool's private key)
   → redirects to https://allin.jianrontan.com/auth/callback#id_token=eyJ...

5. AuthCallback.jsx reads the token from window.location.hash
   → scrubs it from the URL bar (so it doesn't sit in browser history)
   → POSTs { idToken, playerId: <localStorage UUID> } to /api/auth/google

6. Backend auth.verify_cognito_id_token(idToken):
     - fetches the User Pool's JWKS (cached ~1h)
     - finds the JWK matching the token's `kid` (key id) header
     - verifies the RS256 signature against that public key
     - checks iss == https://cognito-idp.<region>.amazonaws.com/<poolId>
     - checks aud == our App Client ID
     - checks exp > now and token_use == 'id'
     - on any failure → AuthError → 401 to the client

7. PlayerStore.link_account(playerId, email=..., authProvider='google',
                              providerSub=token.sub):
     - if a DIFFERENT playerId already has this providerSub (the user signed in
       on a different browser before), MERGE non-destructively: prefer the
       higher-hands row's stats onto the surviving row, mark the other merged.
     - else: just add {email, authProvider, providerSub, isRegistered=True}
       to the existing anonymous row. Their handle and lifetime stats survive.

8. /auth/callback shows "Welcome back, <handle>. You've played N hands."
   → never a 0-state. localStorage caches { handle, isRegistered } so the header
     can render the signed-in chip without a self-lookup endpoint.
```

#### Why this is safe (the security properties that matter)

- **The backend trusts no client claim.** Even if the frontend says "I am
  jianrontan", the backend re-verifies the token. The frontend is a hostile
  environment in principle.
- **The signature anchors trust.** Cognito's private signing key never leaves
  AWS. The JWKS gives us the *public* key. A token we can verify against the
  JWKS came from Cognito — full stop.
- **`aud` pins which app the token is for.** A token issued for app X can't
  be replayed against app Y. We check `aud == <our App Client ID>`.
- **`exp` bounds replay windows.** Cognito ID tokens default to 1h. After
  that, the token is rejected; the user re-authenticates (or uses a refresh
  token — we don't, see Deviation below).
- **Hostile playerId binding is blocked.** If you somehow obtain another
  user's playerId UUID and try to bind your Google account to it,
  `/api/auth/google` checks `existing.providerSub != claims.sub` and returns
  403.

#### Deviation from the original plan

The implementation uses **implicit flow** (`response_type=token`, ID token
returned directly in the URL fragment). The original plan mentioned **PKCE
authorization code flow**, which also returns a *refresh* token (so the user
doesn't have to re-sign-in when the 1h ID token expires).

Implicit flow is simpler and zero-server-secret. The cost is the user
re-authenticates every ~1h. **For v1 that's fine** — they're not hitting auth
endpoints continuously, only on first sign-in. Upgrading to PKCE later is
additive: add a `/api/auth/refresh` endpoint that calls Cognito's token
endpoint with the refresh token; frontend persists the refresh token in an
httpOnly cookie (or memory + localStorage); silent refresh on expiry.

### Endpoint inventory

Strategy lookup (unchanged): `/api/strategy`, `/api/strategy/from-hand`,
`/api/strategy/river-solve`, `/api/abstractions`.

Game (unchanged shape; only internal `_record_hand_end` + `_hand_cap_response`
plumbing added): `/api/game/new`, `/api/game/state`, `/api/game/action`,
`/api/game/bot-action`, `/api/game/next-hand`.

**New in v1 surface:**

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /api/stats` | The +EV counter source. | 5s in-process cache (hottest endpoint, polled by every browser every 30s). |
| `GET /api/leaderboard?n=10&min_hands=50&accounts_only=true` | Ranked board (accounts only) or Most-active (anonymous OK). | Rows redacted (no `playerId`/`email`). |
| `POST /api/player` | Upsert the caller's handle. | Validates regex + profanity, 400 on reject. |
| `POST /api/auth/google` | Bind a Google account (via Cognito) to the caller's existing playerId. | 503 if Cognito unconfigured, 401 on bad token, 403 if the playerId already belongs to another account. |
| `GET /api/healthz` (alias of `/api/test`) | Health probe. | Reports blueprint, iterations, baked-table presence, session store class, debug overlay flag, `ALLIN_GIT_SHA`. |

### What's stubbed vs wired

| Thing | State |
|---|---|
| `LocalFileSource` (blueprint from baked-in file) | Wired ✅ |
| `S3ObjectSource` (blueprint from S3) | Implemented, not wired (no env var flips to it in v1) |
| `InMemory*` stores | Wired, default in dev |
| `DynamoDB*` stores | Wired, default in image (env override at deploy) |
| `HandStore` recap capture | Wired ✅ (write-only — no v1 consumer; data is captured from launch onward for the post-launch hand-history UI / coach / RAG) |
| Hand-history / coach / RAG endpoints | Not built; post-launch features that READ the data captured above |
| Cognito auth (`/api/auth/google`) | Wired; **503 until** `ALLIN_COGNITO_*` env vars are set at deploy |
| Hosted UI redirect (frontend `GoogleSignInButton`) | Wired; **hidden until** `VITE_COGNITO_*` build vars are set |
| `@require_account` decorator | Stubbed; nothing in v1 uses it (gameplay stays playerId-routed). Used by v1.1 saved-hands. |
| Refresh tokens / silent re-auth | Not done (implicit flow). v1.1. |

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
  bot's bucketed hand class mid-hand) in game responses; **default `0` (off)**. Set `1` only in
  local dev. Must stay unset/`0` in prod — the repo is public, so a hot debug overlay is a leak.
- `ALLIN_LOG_LEVEL` — log level for the WSGI/dev server (default `INFO`).
- `ALLIN_GIT_SHA` — build commit, surfaced in `/api/healthz` (set by CI; absent in dev is fine).
- `ALLIN_DEBUG` — dev server only: `1` (default) enables the Werkzeug debugger, `0` disables.
  Irrelevant under gunicorn/waitress (that code path never runs).
- `ALLIN_DEV_HOST` / `ALLIN_DEV_PORT` — dev server bind (default `127.0.0.1:5000`).
- `VITE_API_BASE` — frontend API base URL (set at build time).
- `VITE_COGNITO_DOMAIN` / `VITE_COGNITO_APP_CLIENT_ID` / `VITE_COGNITO_REDIRECT_URI` —
  frontend (build-time), for the "Sign in with Google" Hosted-UI redirect. **Public values**
  (not secrets). When unset, the sign-in button hides itself. The Google OAuth *client secret*
  is NEVER here — it lives only in the Cognito IdP config in AWS.
