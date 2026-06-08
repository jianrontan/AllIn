# Ideas / Feature Backlog

Last updated: 2026-06-08 (positioning notes §6 updated: river solver shipped, turn/flop shelved)

Speculative features and "could we…" explorations that aren't yet committed phases.
The committed, sequenced work lives in [ROADMAP.md](ROADMAP.md); this file is the
holding pen for ideas, with enough research attached to make a later go/no-go cheap.

Status legend: 💡 idea · 🔬 researched · 🧪 prototyping · ❄️ parked

---

## 1. Offloading compute to the user (WebAssembly & friends) 🔬

**The problem it solves.** The expensive part of the system is the Phase-4 river
solver: a vectorized CFR+ solve with a locked **10s per-decision budget**, CPU/GIL-bound
(numpy). Blueprint inference is cheap (a SQLite lookup); the *solve* is what costs server
CPU. With a possible 100–200 concurrent-user burst at launch, many simultaneous 10s solves
are the one thing that can tip the server over. If the solve ran **on the user's own
machine**, that cost goes to zero and scales linearly with users for free — and "the poker
AI solves the river live, in your browser" is a strong portfolio line.

**The hard constraint to remember (the bucketing dependency).** The river *solve itself*
is exact-hand and needs **no abstraction tables**. But it consumes input ranges built from
blueprint *reach*, and the blueprint is bucketed. Bucketing a flop/turn situation needs
either the baked postflop tables (**the turn table is ~126 MB** — a non-starter to ship to
a browser) or the slow lazy per-situation path. The blueprint `.db` itself is small (a few
MB, shippable). So **full** client-side play is blocked by the table; a **hybrid** is not
(see Option D).

### Options researched (2026-05-25)

| Option | What runs client-side | Rewrite cost | Verdict |
|---|---|---|---|
| **A. Pyodide (Python-in-WASM)** | The existing Python solver, unchanged | ~zero code | ❌ Not viable for the hot path |
| **B. Rust → WASM** | Solver inner loop ported to Rust | High (2nd impl) | ✅ Best perf if we go client-side |
| **C. Go → WASM (TinyGo)** | Solver loop ported to Go | High (2nd impl) | ⚠️ Possible, but Rust is the better WASM target |
| **D. Hybrid: server builds ranges, browser solves** | Only the exact river CFR (no buckets) | Medium | ✅ **Most promising** — dodges the 126 MB table |

**A — Pyodide.** Tempting because it runs our *existing* Python+numpy with near-zero
rewrite. Killed by the runtime realities: ~6.4 MB (core) to ~15 MB (with numpy/scipy)
first-load download, 4–5 s init, **no threading/multiprocessing** (single Web Worker),
and meaningfully slower than native. Fine for a one-off demo, wrong for a per-hand 10s
solve users do repeatedly. Park it.

**B — Rust → WASM.** The performance path. 2025 benchmarks put Rust/WASM at ~8–10× over
pure JS for compute-heavy loops, and **near-native** with the now-broadly-shipped **128-bit
SIMD** (all major browsers, late-2024/2025) and **WASM threads** (WebAssembly 3.0, end of
2025: GC, threads, Memory64, SIMD across all major browsers). Cost: the CFR inner loop is a
**second implementation** in a second language to maintain alongside the Python trainer/eval.
Note `wasm-bindgen` marshalling overhead at the JS↔WASM boundary — keep the boundary coarse
(hand the solver flat typed arrays, get a strategy vector back), don't chatter.

**C — Go → WASM.** Works (TinyGo for a smaller binary than the stdlib toolchain, which emits
multi-MB blobs), but for a numeric hot loop Rust produces leaner, faster WASM and has the
SIMD/threads story more mature. If we ever offload, prefer Rust. (Go as a *backend* rewrite
is separately not worth it — we'd lose numpy and rewrite the working Flask-free game core
for a concurrency win a Python worker pool already buys at this scale.)

**D — Hybrid (recommended shape if we offload at all).** Server does the cheap, table-
dependent work (bucket the history, build both range vectors via blueprint reach) and ships
the two ~1225-combo range vectors + board + pot/stacks to the browser. The browser runs
**only the exact river CFR+** (no buckets, no tables) in WASM and returns the strategy for
the bot's hand. This moves the expensive compute off the server **without** shipping the
126 MB table. The seam is clean because the solver's input contract is already small and
fully specified (see [river-solver-design] memory): board, pot, stacks, two range vectors.

### Supporting browser tech (all shipping as of late 2025)
- **WASM SIMD** (128-bit) — broad support; big win for the vectorized payoff/regret math.
- **WASM threads** via SharedArrayBuffer + atomics — requires the page to send
  `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`
  headers, else it silently falls back to single-threaded. Note when we set up hosting.
- **Web Workers** — run the solve off the main thread regardless of WASM, so the UI never
  freezes during a solve (do this even server-side-rendered).
- **sql.js / sql.js-httpvfs** — read the blueprint `.db` in-browser if we ever go fully
  client-side (only relevant once the table problem is solved; parked with Option A).

### Option D performance analysis
Data flow: `human acts → [server] bucket history + build both range vectors (~ms) → ship
{board,pot,stacks,heroRange[1225],villainRange[1225]} (~10 KB) → [browser WASM] exact river
CFR+ → bot action`. The split is at the table-dependent (server) / table-free (client) boundary.

- **Solve speed: neutral, possibly FASTER.** The relevant baseline is Python+**numpy**, not
  native C. numpy vectorizes the array math but the CFR *tree walk* runs in the interpreter;
  Rust→WASM compiles the walk too. The interpreter overhead Python pays on traversal usually
  exceeds the WASM-vs-native penalty, so a Rust/WASM solve lands neutral-to-faster. (The more
  the solve is one big numpy/BLAS op, the more numpy holds its own; the more it's tree walk,
  the more Rust wins.)
- **Latency: neutral / better.** Range-build round trip is ms; the 10s solve then runs locally
  with no long-held server connection. One-time WASM download is a few hundred KB (cached) —
  nothing like Pyodide's 15 MB.
- **Server cost: the big win.** Per decision drops from ~10s CFR to ~ms range-building; the
  100–200 burst stops being a server-CPU problem.
- **Real cost = client-hardware variance.** Trades uniform server CPU for the user's device;
  fast on a laptop, possibly slow on an old phone (the 10s ceiling protects correctness but a
  weak device can hit it). Worst case moves from a busy server to slow client devices.

### ⚠️ The trust / leaderboard tradeoff (decisive here)
Not a performance issue but the decisive one: **a client-computed bot decision can be tampered
with.** A user could modify the WASM to make the bot fold everything and FARM the public +EV
leaderboard — the exact credibility the leaderboard is supposed to demonstrate. Server-side
solving keeps the bot provably honest. Resolutions: (a) **load-adaptive hybrid** — solve
server-side by default, offload to the browser only under real burst load (integrity in the
common case + a pressure valve); (b) mark/exclude any client-solved hands from the leaderboard;
(c) accept it for a showcase and re-verify suspicious streaks.

### Recommendation
Keep the launch **server-side Python** (least resistance, matches the deployment plan), with a
**worker pool on a modestly bigger instance** — likely enough for a 10–20 user load with a rare
100–200 burst, *without* offloading at all. Frame Option D as **(a) a portfolio flex** ("the AI
solves in your browser") **and (b) a load pressure-valve**, not a day-one need; if built, use the
**load-adaptive** form (server-side by default) so leaderboard integrity is preserved, in **Rust**,
porting only the river CFR inner loop. Re-evaluate once Phase 4 ships and we've *measured* real
solve time and concurrency (don't optimize a burst we haven't seen). Prereq: Phase 4 exists + measured.

---

## 2. Bot personality / commentary layer 💡

**The idea.** The bot reacts to the human — taunts on a loss, calls them lucky on a win,
flexes its read. The differentiator: the bot has **rich true internal state**, so the banter
can be *specific and correct*, not generic. "Nice catch" is a chatbot; *"You called my river
shove with 18% equity and hit your gutshot — enjoy it"* shows off the AI. That specificity is
the portfolio flex.

**Signals already computed that can drive lines:**
- **Suckout vs cooler vs hero-call** — from all-in/showdown equity (you were behind and won →
  *lucky*, justified with the number; both strong → *nothing you could do*; you were ahead and
  lost → respect/salt).
- **The bot's read** (`public_view().botRead`: confidence + top hands) — *"I had you on a flush
  draw the whole way."*
- **Range-tracker confidence collapse** (off-model play, `RangeTracker`) — *"...that was a choice."*
- **Its own bluff vs value** (the bot knows its line) — *"You really folded that? Seven-high."*
- **Solver flex** (Phase 4) — *"I solved that river exactly. There was no profitable call."*
- **The +EV leaderboard record** (deployment plan) — *"You're down 40 BB to me, by the way."*

**Two generation strategies:**
1. **Templated lines** keyed by signal + magnitude. Zero latency, fully controlled. Solid v1.
2. **LLM persona** (Claude Haiku) fed a structured hand recap → witty, varied, specific banter.
   *"The bot writes its own trash talk from its actual read"* is a killer launch line; Haiku
   keeps per-hand cost/latency tiny.

**Design notes (build in from the start):**
- Emit a small structured **hand-recap object** from the engine (per-decision equity, was-bluff,
  showdown result, confidence, board). That object is the clean seam for *both* the templated and
  LLM paths — and it doubles as the data for a post-hand "what the bot was thinking" panel.
- Make it a **drop-in layer** (mirror the `BotStrategy` pattern — additive, no Flask coupling, AWS-safe).
- **Needle/tone toggle** (Friendly / Snarky / Silent) — some testers love it, some bounce.
- **PG, never genuinely mean** — punch at the play, not the person.

This competes with the WASM idea for the "one cool feature" budget; the talking persona is
likely the higher-wattage flex for less engineering. Pairs naturally with the leaderboard launch.

---

## 3. The keystone primitive — a structured hand-recap object 💡

The single highest-leverage thing to build. A per-hand object emitted by `GameSession`,
holding **per-street snapshots** of: the bot's read (`opponent_read()` — confidence + top
hands), the bot's equity-vs-believed-range at each of its bets (`hero_equity()`), a
bluff/thin-value/nuts classification of each bot bet, the showdown story tag (suckout / cooler
/ hero-call from pre-runout equity vs result), and the `human_net` delta. Today `botRead`
computes the *current* read live and discards history — persist the **history** instead.

Why it's the keystone: it independently powers feature §2 (persona), §4 (fun features below),
and the leaderboard analytics (`deployment-plan`). Build once, reuse five ways. Seam: it's the
input contract for both the templated and LLM commentary paths, and it doubles as the data for a
post-hand "what the bot was thinking" panel. Key files: `game_session.py` (`opponent_read`,
`_resolve`, `bot_public_state`), `bot_strategy.py` (`explain`, `hero_equity`), `range_tracker.py`.

---

## 4. Fun / engaging feature menu 🔬

Brainstormed 2026-05-26; all grounded in real bot state (no generic chatbot fluff). Effort S/M/L,
wattage = shareability. **Launch bundle = #a + #b + #c + #f + #h** (all reuse §3's recap object,
S–M, no LLM/model needed) — together they manufacture the shareable artifacts that spread a launch.

| # | Feature | Signal | Effort | Wattage |
|---|---------|--------|--------|---------|
| a | **"The Read" reveal** — flip a card after each hand showing what the bot thought you had, turn by turn, with receipts | persisted `opponent_read()` history + showdown truth | S–M | ★★★★★ signature shareable |
| b | **Bluff Confession / Value Reveal** — "That was seven-high. You folded the best hand." | bot `hole_cards` + `hero_equity` at bet time | S | ★★★★ |
| c | **Bad-Beat-o-Meter** — auto-tag suckout / cooler / hero-call | pre-runout equity vs result | S–M | ★★★★ poker players love bad-beat content |
| d | **Post-hand Coach** — grades your decisions vs the blueprint, flags your biggest leak | `BlueprintStrategy.explain()` on the human's info-set key | M | ★★★★ "AI poker coach" |
| e | **Equity / "sweat" graph** — live win-prob line across the hand | true all-in equity per street | M | ★★★ TV-poker premium feel |
| f | **Shareable hand replays** — permalink + auto OG image per hand | serialized session + recap | M | ★★★★ the growth loop |
| g | **"Beat the Bot" daily** — fixed-seed gauntlet, compare to the field + GTO line | seeded deck + `explain()` | M | ★★★★ Wordle-style virality ⚠️ separate board |
| h | **Confidence meter HUD** — live gauge that visibly drops on off-model play | `RangeTracker.confidence` (already shipped to client) | S | ★★★ makes the Bayesian tracker tactile |
| i | **River Solver Spotlight** — "Solving river… 47k CFR iterations. No profitable call." | Phase-4 solver metadata | L (gated) | ★★★★★ if Phase 4 ships |
| j | **Range heatmap** — the 13×13 grid of the bot's believed range, per street | `RangeTracker.weighted_hands()` → grid | M | ★★★★ instantly legible, screenshots well |

**Integrity flags:** daily/challenge (#g) must use a *separate* board (fixed deals aren't a fair
sample); never leak the bot's hole cards mid-hand (`botCards` correctly nulled until `hand_over`).

---

## 5. Stack-enrichment / tech-to-learn menu 🔬

Brainstormed 2026-05-26. The stack only *looks* flat — the CFR solver, EMD abstraction, AIVAT,
and Bayesian tracker are already deep. The job is making the depth **legible** (observability,
real-time, typed seams) + *one* genuine flex (WASM). Scored Learn / Signal / Fit (1–5), effort S/M/L.

**Shortlist (if you do only 2–3):**
1. **Exploitability-over-training dashboard** (Learn4/Signal5/Fit5, S→M) — *highest signal-per-effort
   in the whole brief.* You already compute BR/LBR/AIVAT; log it over iterations and chart it.
   S-version: rows in `training_metadata` + a Recharts panel reading `/api/training/history` (no new
   infra). Resume version: Prometheus + Grafana, add live solve-latency p99. **The screenshot that
   explains the project.** Do the S-version this week regardless.
2. **Rust + WASM equity kernel** (Learn5/Signal4/Fit5, M) — the *learnable on-ramp*, NOT the solver.
   Port the equity-vs-range rollout to a `wasm-pack` crate (flat `Float32Array` boundary, Web Worker,
   then `std::arch::wasm32` SIMD); drop it live into the Strategy Explorer. ~300 lines Rust, ships now,
   no leaderboard-trust caveat. The SIMD benchmark *is* the LinkedIn post. Builds the exact skills for
   the §1 hybrid-solver capstone, which stays gated on Phase 4 ("more of the same, bigger").
3. **WebSockets + containerize + CI** (Fit5, S–M) — the transport upgrade the architecture was *designed*
   to accept (Flask-free `game/` core), so adding it *proves* the design; the "bot is thinking…" stream
   ties to the slow solver and the WASM work. Bundle the cheap DevOps hygiene (Dockerfile that bakes the
   126MB table; GitHub Actions running the existing pytest/Hypothesis suite).

**Also good, in proportion:** typed API layer (Pydantic→OpenAPI→generated TS client, pairs with the
Phase-5 `{action,amount}` widening); IaC (Terraform edge for job-listing signal / CDK for learning) —
one clean stack, don't gold-plate; FastAPI migration only if doing WebSockets + typed API together.
**Resist (gold-plating traps):** a Rust *microservice* (same cost as WASM, none of the "in your browser"
story) and GPU/batched-CFR (MCCFR is branchy tree-walk, poor GPU fit — a research project, not a resume line).

---

## 6. Positioning for resume / LinkedIn 🔬

Brainstormed 2026-05-26. **You are underselling this** — it's the Libratus/Pluribus algorithm family.

**Strongest true-today talking points (lead with the first):**
- **The measurement harness (BR/LBR/AIVAT) is the secret weapon** — anyone can train a model; almost
  nobody builds the instrument that *proves* it's good. Signals research maturity ("measure first, then
  optimize"). Lead with this for any serious technical audience.
- MCCFR+ with Linear-CFR-style discounting, the Libratus/Pluribus algorithm family (honest — same technique family).
- Potential-aware EMD abstraction (equity-distribution histograms + suit-isomorphism canonicalization).
- Hand-level Bayesian range tracker with entropy-relative confidence (correct for mixed strategies).
- **The RIVER subgame solver IS built and shipped** (Phase 4) — you can claim real-time river
  re-solving (Slumbot-class: blueprint + live river solve). Be precise: the **turn/flop** depth-limited
  solver was built and lab-validated but **shelved** (it didn't beat the blueprint in real games), so
  don't claim full multi-street solving — "river solving live; turn/flop is a validated lab result
  pending a continual-re-solving rebuild" is the honest, still-impressive framing.

**LinkedIn posts (hook + why it travels), suggested order:**
1. *"How do you even know if a poker AI is good? Build the instrument first."* — credibility, converts ML/quant followers.
2. *"Watch a poker AI teach itself to stop being exploitable."* — the exploitability-drops graph; rare ML post with a legible down-and-right metric.
3. *"I built a poker AI that writes its own trash talk — from its actual read of your hand."* — highest reach; funny + the grounding is the flex in disguise.
4. *"+EV leaderboard" launch* — "come try to beat my AI; it keeps a public record." Interactive CTA = the acquisition flywheel; the capstone post.
5. Educational carousel on CFR/regret-minimization in plain English — establishes you as someone who can explain hard ideas.

**Cheapest credibility additions (ranked):** (1) a 60–90s demo video/GIF in the README — *the biggest gap,
non-negotiable*; (2) the exploitability-over-training graph committed at the top of the README; (3) a
vs-baseline win-rate (BB/100 vs random + a rule bot) with AIVAT confidence intervals — you have
`run_match.py` + AIVAT, mostly run-and-plot; (4) a ~1,500-word architecture writeup (CLAUDE.md is the raw
material); (5) the live hosted demo. The gap is *presentation and proof, not engineering* — don't add features for credibility.

**Audience framings:** ML/research → lead rigor + lineage (harness, EMD, entropy-confidence). Backend/systems
→ lead transport-agnostic design, WAL read-while-training, interface-driven AWS-readiness. Generalist recruiter
→ "same game-theory family that beat pro players, live and playable, keeps a public record" + the demo link.

**Preempt skeptical questions:** the **river** solver is built (claim that); the **turn/flop** solver is
shelved (a validated lab result, not shipped — say so); the BR baseline is genuinely exploitable in absolute
terms (frame as a *trend you're driving down*, and note the river solver refines the endgame on top of it);
the abstraction omits SPR (the documented M1 limitation — naming it first is a senior move); 3-size betting
abstraction is deliberate; you wrote the CFR/engine/harness yourself (only `phevaluator` for raw hand strength).

---

## 7. AI coaching layer — grounded LLM (RAG / tool-use / LangGraph) 🔬

Brainstormed 2026-05-26. Potentially the STRONGEST resume play — combines the existing game-theory
depth with current LLM/AI-engineering skills, a pairing almost nobody has. Same pipeline as the
persona layer (§2) and the post-hand Coach (§4d): **recap object → LLM layer → grounded in solver
truth**; persona = the entertainment voice, coach = the educational voice. One architecture, two voices.

**THE key insight — ground the LLM, don't let it do poker math.** LLMs hallucinate frequencies/equities
and any player spots it instantly. We already have the correct answer computed: `BlueprintStrategy.explain()`
= the GTO mixed strategy for any info-set; `hero_equity()` + the range tracker = real numbers. The coach's
job is to *explain numbers the solver produced*, not invent strategy. This grounding IS the differentiator
vs a generic "poker coach GPT" and the real engineering story.

**Technique → fit (use each where it belongs; be able to defend each choice — interviewers WILL ask):**
- **Tool-use / function-calling (ESSENTIAL, the core):** LLM calls `explain(key)`, `hero_equity()`, range
  lookups to get the real GTO line + numbers, then explains them. This is what makes the coach correct.
- **RAG (good fit — for CONCEPTS, not data):** vector store over a curated poker-concept KB (~50–200 short
  notes: pot odds, range advantage, polarization, blockers, board texture, c-bet theory…) → retrieve the
  *why* + vocabulary to turn "fold 85%" into a lesson. ANTI-PATTERN to avoid: do NOT RAG over the user's
  hand history — "how often did I punt this spot" is a SQL query over recap objects, not semantic retrieval.
- **LangGraph (fits IF interactive/agentic):** the multi-turn coaching loop with state — analyze recap →
  diagnose biggest leak → retrieve concept + tool-grounded numbers → explain → quiz → adapt. Overkill for
  one-shot explanations; earns its place for the longitudinal version below.
- **LangChain (optional glue):** recognizable but not required; the raw Anthropic SDK + a vector store is
  often cleaner. Know it, don't force it.

**The differentiator = LONGITUDINAL.** A one-shot explainer is fine; the impressive version consumes recap
objects across MANY hands, builds a persistent **leak profile** ("you over-fold rivers, under-3bet the SB"),
and runs a personalized curriculum (spaced repetition / drilling). That's where LangGraph's persistent state
earns its keep and beats a stateless chatbot. The §3 recap object is exactly the structured input it needs.

**Sequencing:** M core = grounded one-shot post-hand explainer (tool-use + small concept KB) — impressive
+ shippable, build first. L extension = interactive LangGraph loop + leak profile + curriculum. Post-hand
feature (not the decision loop), so the 10s-solve budget doesn't apply; Haiku keeps cost/latency low.
**Caveat:** defend every library or it reads as buzzword-stuffing — a smaller honest stack beats a padded one.

---

## 8. More technologies to learn (MCP + others) 🔬

Surveyed 2026-05-26. Beyond WASM (§1/§5) and the LLM stack (§7). Scored Learn/Signal/Fit (1–5).

**MCP (Model Context Protocol)** — open Anthropic standard (late 2024), "USB-C for AI tools": an MCP
*server* exposes **tools** (callable functions), **resources** (readable data), **prompts** (templates);
any MCP *client* (Claude Desktop, Claude Code, your app) connects over a standard transport (stdio/HTTP).
Write the integration once, any client uses it. **Fits AllIn unusually well:** wrap the solver as an MCP
server — `get_gto_strategy()`, `compute_equity()`, `query_hand_history()`, `explain_range()`. Same grounding
philosophy as §7 tool-use (LLM calls the solver for truth), but standardized + reusable. FLEX: connect
**Claude Desktop** to it and analyze hands conversationally through Claude itself — demoable artifact.
Caveat: if the coach is the only consumer, bespoke tool-use is simpler; MCP wins on current-skill signal +
the Claude-Desktop demo. Build the §7 coach's tools as MCP from the start. **Learn4/Signal4/Fit5.**
Resume line: *"Exposed a CFR poker solver as an MCP server — GTO engine available as grounded tools to any LLM client."*

**🟢 High-leverage natural fits (new):**
- **Redis sorted sets for the leaderboard** (L3/S3/Fit5) — the +EV leaderboard is the textbook `ZSET` use
  (O(log n) ranked insert, top-N / my-rank free). Redis also = session store (Phase-5 swap) + pub/sub for
  WebSockets. One tool, three roadmap jobs.
- **Postgres + pgvector** (L4/S4/Fit5) — move live APP DATA (recap objects, leaderboard, profiles, analytics)
  off SQLite to Postgres/RDS; pgvector makes the SAME DB the §7 RAG vector store. Keep SQLite for the blueprint
  (right tool there). Unifies analytics + RAG embeddings.
- **LLM eval harness for the coach** (L4/S5/Fit5) — THE sleeper pick, on-brand with the exploitability harness:
  since `explain()` is ground truth, AUTO-CHECK whether the coach's stated frequencies match the solver →
  catch hallucinations. Langfuse/LangSmith tracing + a custom grounded-correctness eval. Most people don't
  eval their LLM features; you can, because you have truth.

**🟡 Good in proportion:**
- **Async task queue / worker** (L4/S4/Fit4) — Celery/RQ local, AWS **SQS** + workers prod; the concrete
  "worker pool" — offload slow CFR solves + training jobs off the request thread. Pairs with WebSockets.
- **OpenTelemetry** (L3/S3/Fit4) — current tracing standard; pairs with the observability dashboard (§5.3a).

**🔴 Honest skips (resume-padding traps here):** Kubernetes (overkill vs App Runner/ECS), GraphQL (REST is
fine — solves a problem we don't have), gRPC/Protobuf (tied to the gold-plating microservice), heavy auth
(Cognito/Auth0 — deliberately chose a lightweight handle; revisit only if the leaderboard needs real identity).

**Top 3 NEW picks for breadth-without-padding:** MCP server, Redis sorted-set leaderboard, LLM eval harness.

---

## 9. Provably-fair deck — cryptographic commit–reveal 🔬

**The problem it solves.** A skeptical player (or recruiter) has no reason to trust that the
bot isn't cheating. "Cheating" is really **two distinct claims**, and they need different proofs —
conflating them is the trap:

1. **The deck isn't rigged** — the bot isn't dealt better hole cards, and the turn/river aren't
   chosen *after* seeing how the human bet. → provable cryptographically (this idea).
2. **The bot doesn't peek at the human's hole cards** — a property of the *decision code*, not of
   any data commitment, so no seed can prove it. → already enforced architecturally: the strategy's
   `decide(info_set_key, legal_actions, public_state)` only ever receives the actor's **own** cards
   (`game_session.py:bot_public_state` → `'hole_cards': p0/p1 ... # never the opponent's`). The
   `RangeTracker` is a *Bayesian belief* inferred from public betting — the opposite of peeking. The
   honest proof of claim 2 is **open-sourcing the repo** and pointing at that one line; the only
   residual gap (does the deployed binary match the source?) needs remote attestation (TEE/SGX) and
   is overkill for a portfolio piece — don't claim it. **This idea covers claim 1 only.**

**Why it's cheap here.** The deal path is already 90% plumbed: `cards.py:shuffled_deck(rng=None)`
takes an injectable RNG ("Pass an rng for reproducibility"). Today `game_session.py:_deal_hand`
calls it with the global RNG (not reproducible); the change is to feed it a *seeded* RNG and persist
the seed material in the (already fully-JSON-serializable) session state.

### The commit–reveal scheme
1. **Hand start** (`/api/game/new`, `/api/game/next-hand`): server generates a fresh random
   `server_seed`, computes `commit = SHA256(server_seed)`, stores `server_seed` secret, and returns
   **only** `commit` + `nonce` (= hand number). The client supplies a `client_seed` (random, or
   user-chosen).
2. **Deterministic deal**: `deck_seed = HMAC_SHA256(server_seed, client_seed || nonce)`, feed into
   `random.Random(deck_seed)` and pass as `shuffled_deck(rng=...)`. Same inputs → same deck.
3. **Hand end** (showdown/fold): server **reveals** `server_seed`. Client checks
   `SHA256(server_seed) == commit`, then re-derives the shuffle from `(server_seed, client_seed,
   nonce)` and confirms the dealt cards match what it saw.

**Why each half is load-bearing:**
- The **commit** (sent before any action) means the server can't change the runout after seeing the
  human's bets — turn and river were fixed in advance.
- The **client_seed** means the server can't grind `server_seed`s to pre-pick a deck favorable to the
  bot, because it doesn't control the human's half of the input.
- Bonus: the bot's hole cards are just `deck[0:2]` of the committed shuffle, so the **same proof also
  pins the bot's cards in advance** — it can't be dealt a better hand.

**Gotchas:** fresh `server_seed` per hand; reveal **only after** the hand completes (revealing early
would let the player compute future board cards); `nonce = hand_number` prevents seed reuse.

### Scope of work (if built)
Self-contained, touches no training/blueprint code:
- `cards.py` — seed-derivation helper (HMAC → `random.Random`).
- `game_session.py` — `commit`/`server_seed`/`client_seed`/`nonce` fields in `self.data`; seed the
  deal; redact `server_seed` from `public_view()` until `hand_over`, then reveal.
- API — accept an optional `client_seed`; surface `commit` (live) and `server_seed` (post-hand).
- `AiGame.jsx` — a "Verify this hand" affordance that re-derives the deck client-side.
- Pairs with the existing per-hand `bot_debug` trace (`game_session.py`) as a downloadable replay.

**Resume framing:** *"provably-fair deck (cryptographic commit–reveal) + open-source, hole-card-blind
decision path."* Accurate and defensible — note it proves the **deck**, not the absence of peeking
(that's the open-source claim). Effort: S. Wattage: ★★★ (trust/credibility, pairs with the
leaderboard launch where "is the bot honest?" is the obvious question).

---

## Cross-references
- Committed phases & status: [ROADMAP.md](ROADMAP.md)
- Deployment + leaderboard: `deployment-plan` memory
- River solver input contract (the offload seam): `river-solver-design` memory
