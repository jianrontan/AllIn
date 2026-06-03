# backend/bot/src/cfr/parallel_trainer.py
"""
Data-parallel external-sampling MCCFR+ for blueprint training.

External-sampling MCCFR draws an independent random hand per iteration, so
iterations are embarrassingly parallel. The inner cfr() loop is pure Python, so
the GIL forces *processes* (not threads) -- this module runs W worker processes,
each training a chunk of iterations into its own in-memory copy of the blueprint,
then the master merges their contributions and re-broadcasts. Repeat per round.

Single-thread BlueprintTrainer.train_blueprint stays the default and the
reproducible oracle; this is an *approximation* of it (see merge_round) validated
by exploitability, not bit-identity.

Discount handling (the subtle part). Single-thread applies the Linear-CFR alpha/
gamma decay per-iteration, keyed on per-info-set clocks. Workers here run with
`discount_enabled = False` (raw, CFR+-floored accumulation only); the master
applies the decay ONCE PER MERGE ROUND in merge_round -- "block Linear-CFR". This
decouples the discount schedule from the worker count entirely (the global clock
now counts rounds-an-info-set-appeared-in, advanced once per round), at the cost
of treating all iterations within a round as equally weighted. Keep rounds small
(merge_every) and the error stays in the band the data-parallel-CFR literature
tolerates. All workers + master must share identical alpha/gamma.

CFR+ flooring. Single-thread floors the cumulative regret at 0 on EVERY write
(blueprint_trainer.py, the `max(0.0, prior+regret)` in the traverser loop) --
canonical CFR+. The block scheme can only approximate that per-iteration floor.
Two design points:
  - The MASTER applies the CFR+ floor once per round, to the summed total
    (merge_round). This is the per-round-granularity stand-in for the per-iteration
    floor.
  - The WORKERS report RAW (unfloored) regret increments (Fix #2, 2026-05-31).
    Previously each worker floored its own cumulative on write too, so a losing
    action's negative regret was clamped to 0 INSIDE each worker before the master
    ever saw it -- it could never cancel another worker's positive regret within a
    round. That double-floor put a bounded UPWARD bias on high-variance actions
    (e.g. jam), growing with workers*merge_every. Reporting raw deltas restores
    cross-worker cancellation (the master's single floor still keeps the global
    >= 0, matching CFR+ at round granularity).
    THE TRADE-OFF (be precise): storing raw regret changes a worker's OWN
    within-chunk regret matching from CFR+ to vanilla-CFR re-activation. Under CFR+
    a zeroed action re-enters the strategy the instant one positive regret arrives;
    under raw storage an action driven deeply negative must first climb all the way
    back above 0 before get_strategy's read-floor lets it back in -- so within a
    chunk it can stay suppressed longer than CFR+ would (vanilla CFR, still
    convergent, not "slightly slower"). This is bounded because the master floors
    and re-broadcasts a floored baseline EVERY merge, so the raw drift only
    accumulates over one chunk (~merge_every iters) and then resets. Net: we trade
    a systematic cross-worker bias that does NOT vanish with iterations for a
    per-chunk vanilla-CFR approximation that resets each merge -- a good trade, but
    it argues for keeping merge_every modest. Validated by exploitability /
    agreement-with-single-thread (scripts/validate_parallel_merge.py), not bit-
    identity.

This leaves TWO approximations of single-thread, both shrinking with merge_every:
  1. Discount timing -- all iterations within a round share one decay weight
     (block Linear-CFR, above), and workers don't see each other's mid-round
     updates (each anchors increments to the same pre-round baseline).
  2. Floor granularity -- per-round + within-chunk raw accumulation instead of a
     per-iteration floor. Fix #2 shrank this (cancellation restored) but did not
     remove it.
Both vanish at merge_every=1; the single-thread trainer remains the oracle, and
this path is validated by exploitability / agreement-with-single-thread, not
bit-identity.

Platform note: on Windows multiprocessing uses 'spawn' (no fork copy-on-write),
so the baseline blueprint is pickled to each worker each round. This is the
correctness-first path (P1); a persistent-worker / incremental-broadcast and a
shared-memory layout are the documented follow-ups for the larger redesigned
abstraction.
"""
import os
import time
import signal
import random
from multiprocessing import Pool, TimeoutError as _mp_TimeoutError

from .information_set import InformationSet
from .blueprint_trainer import BlueprintTrainer, _format_duration


def _worker_init():
    """Pool-worker initializer: make workers IGNORE SIGINT (Ctrl+C). On Windows,
    Ctrl+C is delivered to the WHOLE process group, so without this every worker
    raises KeyboardInterrupt mid-cfr() AND the pool's maintenance thread respawns
    them (SpawnPoolWorker-9..16 in the traceback) -- an endless "Ctrl+C does
    nothing" loop. With workers ignoring SIGINT, Ctrl+C reaches ONLY the master,
    which then calls pool.terminate() once to SIGKILL the (idle-waiting) workers.
    The master re-enables default SIGINT handling for itself right after pool
    creation."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# --------------------------------------------------------------------------- #
# (De)serialization of the blueprint for cross-process transport.
# Workers only need cumulative_regrets (for regret matching), cumulative_strategy
# (so increments are relative to the same baseline the master holds), and
# legal_actions. The discount clocks live ONLY on the master.
# --------------------------------------------------------------------------- #

def export_baseline(info_sets):
    """Global blueprint -> a plain picklable dict for broadcasting to workers."""
    return {
        key: (info.cumulative_regrets, info.cumulative_strategy, info.legal_actions)
        for key, info in info_sets.items()
    }


def _import_baseline(info_sets, baseline):
    """Load a broadcast baseline into a worker's empty info_sets dict."""
    for key, (regrets, strategy, legal_actions) in baseline.items():
        info = InformationSet()
        info.cumulative_regrets = dict(regrets)
        info.cumulative_strategy = dict(strategy)
        info.legal_actions = list(legal_actions)
        info_sets[key] = info


# --------------------------------------------------------------------------- #
# Worker entry point (module-level so it is picklable under 'spawn').
# --------------------------------------------------------------------------- #

def _worker_run_chunk(payload):
    """Run `chunk` iterations from a baseline with the discount disabled, and
    return only this round's deltas (the info sets touched, split by whether they
    got a regret update, a strategy update, or both)."""
    baseline, start_iteration, chunk, seed, alpha, gamma, menu_mode = payload
    random.seed(seed)

    # The worker MUST build its game under the same action abstraction as the master
    # (capped vs control); otherwise it would generate the wrong legal-action set and
    # info-set keys for the arm being trained.
    trainer = BlueprintTrainer(menu_mode=menu_mode)
    trainer.alpha = alpha
    trainer.gamma = gamma
    trainer.discount_enabled = False        # master applies the block discount
    _import_baseline(trainer.info_sets, baseline)
    trainer._dirty_regret = set()
    trainer._dirty_strategy = set()

    ev_sum0, ev_count0 = trainer.ev_sum, trainer.ev_count
    for i in range(chunk):
        trainer._run_iteration(start_iteration + i)

    info_sets = trainer.info_sets
    regret = {k: info_sets[k].cumulative_regrets for k in trainer._dirty_regret}
    strategy = {k: info_sets[k].cumulative_strategy for k in trainer._dirty_strategy}
    legal = {k: info_sets[k].legal_actions
             for k in (trainer._dirty_regret | trainer._dirty_strategy)}
    return {
        'regret': regret,
        'strategy': strategy,
        'legal': legal,
        'ev_sum': trainer.ev_sum - ev_sum0,
        'ev_count': trainer.ev_count - ev_count0,
    }


# --------------------------------------------------------------------------- #
# Master-side merge (block Linear-CFR discount + summed worker increments).
# Pure function of (global info_sets, worker_results) -- unit-tested directly.
# --------------------------------------------------------------------------- #

def merge_round(info_sets, worker_results, alpha, gamma):
    """Fold one round of worker deltas into the global blueprint, in place.

    For every key a worker touched, sum the per-worker increments relative to the
    pre-round baseline, advance that key's discount clock by one round, decay the
    existing cumulative by the Linear-CFR factor, then add the increments and apply
    the CFR+ floor ONCE to the summed total. The workers report RAW (unfloored)
    increments (Fix #2), so opposite-signed contributions cancel here before this
    single floor -- the floor stays. Returns the set of keys mutated (for dirtying
    the checkpoint).

    Regret and strategy are clocked independently: a key only advances its regret
    clock if some worker updated its regret this round (i.e. visited it as the
    traverser), and likewise for strategy -- mirroring the single-thread split
    where regrets update at traverser nodes and the average strategy at opponent
    nodes.
    """
    regret_keys = set()
    strategy_keys = set()
    for res in worker_results:
        regret_keys.update(res['regret'])
        strategy_keys.update(res['strategy'])

    # Snapshot pre-round baselines BEFORE any mutation, so every worker's
    # increment is measured against the same R0/S0.
    R0 = {k: dict(info_sets[k].cumulative_regrets) if k in info_sets else {}
          for k in regret_keys}
    S0 = {k: dict(info_sets[k].cumulative_strategy) if k in info_sets else {}
          for k in strategy_keys}

    def _ensure(key, res_legal_lookup):
        info = info_sets.get(key)
        if info is None:
            info = InformationSet()
            for res in res_legal_lookup:
                if key in res['legal']:
                    info.legal_actions = list(res['legal'][key])
                    break
            info_sets[key] = info
        elif not info.legal_actions:
            for res in res_legal_lookup:
                if key in res['legal']:
                    info.legal_actions = list(res['legal'][key])
                    break
        # Every dirty key is built from the union (_dirty_regret | _dirty_strategy),
        # for which the worker always emits res['legal'], so the scan above must
        # succeed. Guard the contract: an empty legal_actions here would silently
        # seed an info set that later NaNs in get_average_strategy / regret matching.
        # Use an explicit raise (not assert) so the guard survives `python -O`,
        # which strips asserts -- a long training run launched with -O must still
        # fail loud rather than corrupt the blueprint.
        if not info.legal_actions:
            raise RuntimeError(f"merge_round: no legal_actions for key {key!r}")
        return info

    # --- Regret merge ---
    for key in regret_keys:
        info = _ensure(key, worker_results)
        base = R0[key]
        increment = {}
        for res in worker_results:
            wr = res['regret'].get(key)
            if wr:
                for a, v in wr.items():
                    increment[a] = increment.get(a, 0.0) + (v - base.get(a, 0.0))
        info.visit_count += 1
        t = info.visit_count
        decay = ((t - 1) / t) ** alpha if t > 1 else 1.0
        merged = {}
        for a in set(base) | set(increment):
            # The master applies the CFR+ floor ONCE per round, to the SUMMED total.
            # Single-thread floors on every write (the max(0.,.) in cfr()'s traverser
            # loop); the block scheme floors at round granularity instead. Fix #2 is
            # NOT here -- it is that the WORKERS now report RAW (unfloored) regret
            # increments (see _worker_run_chunk / blueprint_trainer's worker mode), so
            # a losing action's negative increment from one worker can cancel a winning
            # one from another WITHIN this sum before the single floor below. Previously
            # each worker floored locally first, discarding those negatives -> a bounded
            # upward bias on high-variance actions (jam). Keep this floor.
            merged[a] = max(0.0, decay * base.get(a, 0.0) + increment.get(a, 0.0))
        info.cumulative_regrets = merged

    # --- Strategy merge (no floor; gamma clock) ---
    for key in strategy_keys:
        info = _ensure(key, worker_results)
        base = S0[key]
        increment = {}
        for res in worker_results:
            ws = res['strategy'].get(key)
            if ws:
                for a, v in ws.items():
                    increment[a] = increment.get(a, 0.0) + (v - base.get(a, 0.0))
        info.strategy_visit_count += 1
        s = info.strategy_visit_count
        decay = ((s - 1) / s) ** gamma if (s > 1 and gamma) else 1.0
        merged = {}
        for a in set(base) | set(increment):
            merged[a] = decay * base.get(a, 0.0) + increment.get(a, 0.0)
        info.cumulative_strategy = merged

    return regret_keys | strategy_keys


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #

def train_blueprint_parallel(trainer, iterations, db=None, start_iteration=0,
                             checkpoint_every=10000, workers=None,
                             merge_every=2000, seed=None, pool=None):
    """Data-parallel training driver. Mutates `trainer.info_sets` in place.

    Args:
        trainer:          a BlueprintTrainer holding the (possibly resumed) global
                          blueprint and the alpha/gamma schedule.
        iterations:       total iterations to run this session.
        merge_every:      iterations PER WORKER between merges (the round size).
                          Smaller = more faithful discount, more broadcast cost.
        workers:          process count (default: os.cpu_count()).
        pool:             an existing multiprocessing Pool (else one is created).
    """
    if workers is None:
        workers = os.cpu_count() or 1
    alpha, gamma = trainer.alpha, trainer.gamma
    menu_mode = trainer.menu_mode      # workers must build the same abstraction arm
    # Per-worker seed = base_seed + round_idx*workers + w (unique per (round, worker)).
    # Fold start_iteration in so a SEEDED resume doesn't restart round_idx at 0 and
    # re-deal the exact hands the original run already trained on (correlated work).
    base_seed = (seed if seed is not None else random.randrange(1 << 30)) + start_iteration

    own_pool = pool is None
    if own_pool:
        # Workers ignore SIGINT (see _worker_init) so Ctrl+C hits only the master.
        # The Pool maintenance thread inherits SIG_IGN from creation context; set it
        # here, create the pool, then restore the master's own SIGINT so our
        # except-KeyboardInterrupt below still fires.
        prev_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            pool = Pool(processes=workers, initializer=_worker_init)
        finally:
            signal.signal(signal.SIGINT, prev_sigint)

    total_target = start_iteration + iterations
    print(f"Starting PARALLEL blueprint CFR training")
    print(f"  Target iterations : {total_target:,}")
    print(f"  Workers           : {workers}")
    print(f"  Merge every       : {merge_every:,} iters/worker "
          f"({merge_every * workers:,} iters/round)")
    print(f"  Checkpoint every  : {checkpoint_every:,}")
    print()

    t_start = time.time()
    done = 0
    cursor = start_iteration
    since_checkpoint = 0
    round_idx = 0
    ev_ema = None                 # smoothed per-round EV (see EV(round) print below)
    EV_EMA_BETA = 0.2             # ~last 5 rounds

    try:
        while done < iterations:
            t_round = time.time()
            remaining = iterations - done
            round_total = min(merge_every * workers, remaining)
            # Split this round's iterations across workers as evenly as possible.
            base_chunk, extra = divmod(round_total, workers)
            baseline = export_baseline(trainer.info_sets)

            payloads = []
            offset = 0
            for w in range(workers):
                chunk = base_chunk + (1 if w < extra else 0)
                if chunk == 0:
                    continue
                payloads.append((
                    baseline,
                    cursor + offset,
                    chunk,
                    base_seed + round_idx * workers + w,
                    alpha, gamma, menu_mode,
                ))
                offset += chunk

            # map_async(...).get(timeout) instead of the blocking pool.map: a bare
            # blocking map() on Windows does NOT return control to the main thread on
            # Ctrl+C (the interrupt can't surface until the C-level wait ends), so the
            # except below never fires. Polling .get() with a finite timeout keeps the
            # main thread interruptible, so KeyboardInterrupt propagates immediately.
            async_res = pool.map_async(_worker_run_chunk, payloads)
            while True:
                try:
                    results = async_res.get(timeout=1.0)
                    break
                except _mp_TimeoutError:
                    continue

            touched = merge_round(trainer.info_sets, results, alpha, gamma)
            trainer._dirty.update(touched)
            round_ev_sum = round_ev_count = 0.0
            for res in results:
                trainer.ev_sum += res['ev_sum']
                trainer.ev_count += res['ev_count']
                round_ev_sum += res['ev_sum']
                round_ev_count += res['ev_count']
            # Per-round EV: mean root value of THIS round's frozen broadcast strategy.
            # The least-lagged EV signal the parallel path has (vs the lifetime
            # EV(cum) which barely moves on a resume). Still lagged ~one round behind
            # single-thread's per-iteration EV(sess) -- all workers measure the
            # round-START strategy, not within-round updates -- so watch its TREND
            # (should fall toward the game value as it converges), not its absolute
            # level, and never compare it to single-thread EV(sess) or to BR/LBR.
            round_ev = round_ev_sum / round_ev_count if round_ev_count else 0.0
            ev_ema = round_ev if ev_ema is None else (
                EV_EMA_BETA * round_ev + (1.0 - EV_EMA_BETA) * ev_ema)

            done += round_total
            cursor += round_total
            since_checkpoint += round_total
            round_idx += 1

            round_secs = time.time() - t_round
            elapsed = time.time() - t_start
            ips = done / elapsed if elapsed > 0 else 0
            round_ips = round_total / round_secs if round_secs > 0 else 0
            eta = _format_duration((iterations - done) / ips) if ips > 0 else "?"
            # NOTE: EV(cum) here READS HIGH vs single-thread and is NOT comparable to
            # it. Each worker records the value of the STALE round-start broadcast
            # strategy (frozen per round; workers don't see within-round updates), so
            # this gauge measures a strategy that advances only once per round
            # (merge_every*workers iters) -- structurally less converged -> inflated
            # EV. Measured: parallel reads ~+1.4 (control) to ~+2.2 (capped) above
            # single-thread at matched iters. The accumulation is arithmetically
            # correct (ev_count==total_iterations); it's the gauge's reference
            # strategy that lags. For TRUE strength use the eval harness (BR/LBR), not
            # this number. Labelled EV(cum,lagged) to flag it. See
            # ev-cum-investigation memory.
            cum_ev = trainer.ev_sum / trainer.ev_count if trainer.ev_count else 0.0
            print(f"  round {round_idx:>4} | iter {cursor:>11,}/{total_target:,} "
                  f"(+{round_total:,}) | EV(cum,lagged): {cum_ev:+.5f} | "
                  f"EV(round): {round_ev:+.4f} | EV(round,ema): {ev_ema:+.4f} | "
                  f"info sets: {len(trainer.info_sets):>8,} | "
                  f"{round_ips:>7.1f} it/s round ({ips:>6.1f} avg) | "
                  f"round: {_format_duration(round_secs)} | "
                  f"elapsed: {_format_duration(elapsed)} | ETA: {eta}")

            if db is not None and since_checkpoint >= checkpoint_every:
                trainer.checkpoint_to_db(db, cursor - 1)
                since_checkpoint = 0
                # Self-play value of the SERVED (average) strategy. Unlike
                # EV(cum,lagged)/EV(round) above (the CURRENT iterate, which can cycle
                # at a large value forever), this settles to a small STABLE constant
                # (the button's game-value edge, ~0). It is a SEAT-BALANCE/convergence
                # sanity check, NOT a strength metric: a seat-lopsided or unconverged
                # served strategy reads large, but two equally-bad strategies also
                # self-play near the constant -- for exploitability/strength use LBR.
                served_ev = trainer.evaluate_served_ev()
                print(f"  EV(served, avg strategy): {served_ev:+.4f}  <- served "
                      f"self-play value (seat-balance check, NOT strength -> use LBR)")

        # Flush the tail. since_checkpoint advances in round_total units and rarely
        # lands exactly on a checkpoint boundary, so the final rounds' work is
        # otherwise left only in memory -- a silent data loss AND an undercounted
        # total_iterations that would make the next resume replay lost iterations.
        if db is not None and since_checkpoint > 0:
            trainer.checkpoint_to_db(db, cursor - 1)
            since_checkpoint = 0
            served_ev = trainer.evaluate_served_ev()
            print(f"  EV(served, avg strategy): {served_ev:+.4f}  <- served "
                  f"self-play value (seat-balance check, NOT strength -> use LBR)")
    except KeyboardInterrupt:
        # Ctrl+C: kill the workers IMMEDIATELY (terminate, not close). pool.close()
        # waits for every queued chunk to finish before join() returns -- with 8
        # workers each mid-chunk that's "Ctrl+C does nothing for minutes", the
        # symptom that forced killing the terminal. terminate() SIGKILLs them now.
        # Work since the last checkpoint is lost (resume picks up from the last
        # checkpointed iteration), which is the correct interrupt semantics.
        print("\nKeyboardInterrupt -- terminating workers (progress since the last "
              "checkpoint is discarded; resume from the checkpointed iteration).")
        if own_pool:
            pool.terminate()
            pool.join()
            own_pool = False             # already torn down; skip the finally close()
        raise
    finally:
        if own_pool:
            pool.close()
            pool.join()

    print(f"\nParallel training completed in {_format_duration(time.time() - t_start)}.")
    return trainer.ev_sum / trainer.ev_count if trainer.ev_count else 0.0
