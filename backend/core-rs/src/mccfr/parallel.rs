//! Training drivers: single-thread and data-parallel (Phase 5).
//!
//! The parallel scheme is the one `cfr/parallel_trainer.py` validated ("block
//! Linear-CFR"): each round, W workers run iterations against a frozen
//! baseline with the discount disabled (CFR+-floored on every write), then the
//! master merges their increments and applies the discount once per round
//! (`merge_round`, ported bit-exactly). Differences from the Python driver are
//! mechanical only: workers are threads sharing the baseline read-only (each
//! copies an info set on first touch) instead of processes receiving a pickled
//! copy, and iterations are handed out dynamically in small blocks so fast and
//! efficiency cores finish a round together. Randomness is per ITERATION
//! (seeded from `seed` and the iteration index), so a deal does not depend on
//! which worker ran it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};

use super::rng::Xoshiro;
use super::table::{ActMap, InfoSet, Legal, floor0};
use super::trainer::{Abstraction, Table, Trainer};
use crate::buckets::Buckets;

/// The production card abstraction: the ported bucketing + evaluator.
pub struct RealAbstraction {
    pub buckets: Buckets,
}

impl Abstraction for RealAbstraction {
    fn preflop_fine(&self, hole: [u8; 2]) -> u8 {
        crate::buckets::preflop::preflop_fine(hole[0], hole[1])
    }

    fn coarse(&self, fine: u8) -> u8 {
        crate::buckets::preflop::fine_to_coarse()[fine as usize]
    }

    fn postflop(&self, hole: [u8; 2], board: &[u8]) -> Result<u16, String> {
        self.buckets.bucket(hole, board).map(u16::from)
    }

    fn rank(&self, cards: &[u8]) -> i32 {
        crate::eval::rank(cards)
    }
}

/// One worker's round output (`_worker_run_chunk`): absolute cumulative values
/// of the info sets it updated, split by update kind.
#[derive(Default)]
pub struct WorkerDelta {
    pub regret: FxHashMap<u64, ActMap>,
    pub strategy: FxHashMap<u64, ActMap>,
    pub legal: FxHashMap<u64, Legal>,
    pub ev_sum: f64,
    pub ev_count: i64,
}

impl Trainer {
    fn into_delta(self) -> WorkerDelta {
        let mut d = WorkerDelta {
            ev_sum: self.ev_sum,
            ev_count: self.ev_count,
            ..Default::default()
        };
        for k in &self.dirty_regret {
            d.regret.insert(*k, self.table[k].regrets);
        }
        for k in &self.dirty_strategy {
            d.strategy.insert(*k, self.table[k].strategy);
        }
        for k in self.dirty_regret.iter().chain(&self.dirty_strategy) {
            d.legal.insert(*k, self.table[k].legal);
        }
        d
    }
}

/// The pending merged value of `key`: this round's update so far, else the
/// table's entry, else a new info set; legal actions filled from the workers.
fn ensure<'u>(
    table: &Table,
    updates: &'u mut FxHashMap<u64, InfoSet>,
    key: u64,
    workers: &[WorkerDelta],
) -> Result<&'u mut InfoSet, String> {
    let is = updates
        .entry(key)
        .or_insert_with(|| table.get(&key).copied().unwrap_or_default());
    if is.legal.is_empty() {
        if let Some(l) = workers.iter().find_map(|w| w.legal.get(&key)) {
            is.legal = *l;
        }
    }
    if is.legal.is_empty() {
        return Err(format!("merge_round: no legal_actions for key {key:#x}"));
    }
    Ok(is)
}

/// `{a: base[a]*decay + inc[a]}` over base ∪ increment (base order first),
/// with the CFR+ floor for regrets.
fn merged(base: &ActMap, inc: &ActMap, decay: f64, floor: bool) -> Result<ActMap, String> {
    let mut out = ActMap::default();
    for (a, b) in base.iter() {
        let v = decay * b + inc.get(a).unwrap_or(0.0);
        out.set(a, if floor { floor0(v) } else { v })?;
    }
    for (a, i) in inc.iter() {
        if base.get(a).is_none() {
            let v = decay * 0.0 + i;
            out.set(a, if floor { floor0(v) } else { v })?;
        }
    }
    Ok(out)
}

/// Per-action sum over workers (in worker order) of `value - base`.
fn increment<'w>(base: &ActMap, maps: impl Iterator<Item = &'w ActMap>) -> Result<ActMap, String> {
    let mut inc = ActMap::default();
    for m in maps {
        for (a, v) in m.iter() {
            let prev = inc.get(a).unwrap_or(0.0);
            inc.set(a, prev + (v - base.get(a).unwrap_or(0.0)))?;
        }
    }
    Ok(inc)
}

/// Port of `parallel_trainer.merge_round`. Returns the touched keys.
/// All-or-nothing: updates are computed first and applied only if every key
/// merged cleanly, so an error never leaves the table half-merged.
pub fn merge_round(
    table: &mut Table,
    workers: &[WorkerDelta],
    alpha: f64,
    gamma: f64,
) -> Result<FxHashSet<u64>, String> {
    let mut regret_keys: FxHashSet<u64> = FxHashSet::default();
    let mut strategy_keys: FxHashSet<u64> = FxHashSet::default();
    for w in workers {
        regret_keys.extend(w.regret.keys());
        strategy_keys.extend(w.strategy.keys());
    }
    let mut updates: FxHashMap<u64, InfoSet> = FxHashMap::default();
    let bump = |c: &mut i64| -> Result<(), String> {
        *c = c.checked_add(1).ok_or("visit counter overflow")?;
        Ok(())
    };
    for &key in &regret_keys {
        let is = ensure(table, &mut updates, key, workers)?;
        let base = is.regrets;
        let inc = increment(&base, workers.iter().filter_map(|w| w.regret.get(&key)))?;
        bump(&mut is.visit_count)?;
        let t = is.visit_count;
        let decay = if t > 1 {
            ((t - 1) as f64 / t as f64).powf(alpha)
        } else {
            1.0
        };
        is.regrets = merged(&base, &inc, decay, true)?;
    }
    for &key in &strategy_keys {
        let is = ensure(table, &mut updates, key, workers)?;
        let base = is.strategy;
        let inc = increment(&base, workers.iter().filter_map(|w| w.strategy.get(&key)))?;
        bump(&mut is.strategy_visit_count)?;
        let s = is.strategy_visit_count;
        let decay = if s > 1 && gamma != 0.0 {
            ((s - 1) as f64 / s as f64).powf(gamma)
        } else {
            1.0
        };
        is.strategy = merged(&base, &inc, decay, false)?;
    }
    let touched: FxHashSet<u64> = updates.keys().copied().collect();
    table.extend(updates);
    Ok(touched)
}

/// Deterministic per-iteration generator.
#[inline]
pub fn iteration_rng(seed: u64, iteration: i64) -> Xoshiro {
    Xoshiro::new(seed ^ (iteration as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RunStats {
    pub iterations: u64,
    pub rounds: u64,
    pub ev_sum: f64,
    pub ev_count: i64,
    pub seconds: f64,
}

/// Single-thread training (the reproducible oracle path, discount per visit).
pub fn train_single<A: Abstraction + ?Sized>(
    t: &mut Trainer,
    abs: &A,
    iterations: u64,
    start: i64,
    seed: u64,
) -> Result<RunStats, String> {
    let t0 = Instant::now();
    let (s0, c0) = (t.ev_sum, t.ev_count);
    for i in 0..iterations {
        let it = start + i as i64;
        let mut rng = iteration_rng(seed, it);
        let deal = rng.deal();
        t.run_iteration(it, deal, abs, &mut rng)?;
    }
    Ok(RunStats {
        iterations,
        rounds: 0,
        ev_sum: t.ev_sum - s0,
        ev_count: t.ev_count - c0,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

const BLOCK: u64 = 16;
/// Upper bound on worker threads (sanity limit, far above any core count).
pub const MAX_WORKERS: usize = 1024;

/// Data-parallel training for `iterations` starting at `start`; rounds of
/// `merge_every * workers` iterations. `on_round(stats_so_far)` runs after
/// each merge (logging); returning `false` stops after that round.
#[allow(clippy::too_many_arguments)]
pub fn train_parallel<A: Abstraction + ?Sized>(
    master: &mut Trainer,
    abs: &A,
    iterations: u64,
    start: i64,
    workers: usize,
    merge_every: u64,
    seed: u64,
    mut on_round: impl FnMut(&Trainer, &RunStats) -> bool,
) -> Result<RunStats, String> {
    if workers == 0 || workers > MAX_WORKERS || merge_every == 0 {
        return Err(format!(
            "workers must be 1..={MAX_WORKERS} and merge_every >= 1"
        ));
    }
    let per_round = merge_every
        .checked_mul(workers as u64)
        .ok_or("merge_every * workers overflows")?;
    if start < 0
        || (start as u64)
            .checked_add(iterations)
            .is_none_or(|e| e > i64::MAX as u64)
    {
        return Err("iteration range out of bounds".into());
    }
    if !master.discount_enabled {
        return Err("the master trainer must have the discount enabled".into());
    }
    let t0 = Instant::now();
    let mut stats = RunStats::default();
    let (mode, sched, key_format) = (master.menu_mode.clone(), master.sched, master.key_format);
    while stats.iterations < iterations {
        let round_total = per_round.min(iterations - stats.iterations);
        let cursor = start + stats.iterations as i64;
        let base = Arc::new(std::mem::take(&mut master.table));
        let next = AtomicU64::new(0);
        let results: Vec<Result<WorkerDelta, String>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    let base = Arc::clone(&base);
                    let (next, mode) = (&next, &mode);
                    // Builder::spawn_scoped returns an error (instead of
                    // panicking) if the OS refuses a thread.
                    std::thread::Builder::new().spawn_scoped(
                        s,
                        move || -> Result<WorkerDelta, String> {
                            let mut wt = Trainer::worker(mode, sched, base)?;
                            wt.key_format = key_format;
                            loop {
                                let k = next.fetch_add(BLOCK, Ordering::Relaxed);
                                if k >= round_total {
                                    break;
                                }
                                for j in k..(k + BLOCK).min(round_total) {
                                    let it = cursor + j as i64;
                                    let mut rng = iteration_rng(seed, it);
                                    let deal = rng.deal();
                                    wt.run_iteration(it, deal, abs, &mut rng)?;
                                }
                            }
                            Ok(wt.into_delta())
                        },
                    )
                })
                .collect();
            handles
                .into_iter()
                .map(|h| match h {
                    Err(e) => Err(format!("could not spawn a worker thread: {e}")),
                    Ok(h) => h
                        .join()
                        .unwrap_or_else(|_| Err("worker thread panicked".into())),
                })
                .collect()
        });
        // Always restore the table before reporting any worker error: all
        // workers are joined here, so the Arc is unique (clone defensively
        // rather than ever lose the blueprint).
        master.table = Arc::try_unwrap(base).unwrap_or_else(|a| (*a).clone());
        let deltas = results.into_iter().collect::<Result<Vec<_>, String>>()?;
        let touched = merge_round(&mut master.table, &deltas, sched.alpha, sched.gamma)?;
        master.dirty.extend(touched);
        for d in &deltas {
            master.ev_sum += d.ev_sum;
            master.ev_count += d.ev_count;
            stats.ev_sum += d.ev_sum;
            stats.ev_count += d.ev_count;
        }
        stats.iterations += round_total;
        stats.rounds += 1;
        stats.seconds = t0.elapsed().as_secs_f64();
        if !on_round(master, &stats) {
            break;
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mccfr::table::{Pattern, pack_key};

    fn map(v: &[(u8, f64)]) -> ActMap {
        let mut m = ActMap::default();
        for &(a, x) in v {
            m.set(a, x).unwrap();
        }
        m
    }

    #[test]
    fn merge_floors_regrets_and_sums_increments() {
        let k = pack_key(1, 0, 3, 4, Pattern::default(), None).unwrap();
        let mut table = Table::default();
        let mut is = InfoSet::default();
        is.legal = Legal::from_ids(&[0, 4]).unwrap();
        is.regrets = map(&[(0, 4.0), (4, 1.0)]);
        is.visit_count = 1;
        table.insert(k, is);
        let w = |r: &[(u8, f64)]| {
            let mut d = WorkerDelta::default();
            d.regret.insert(k, map(r));
            d.legal.insert(k, Legal::from_ids(&[0, 4]).unwrap());
            d
        };
        // worker A: action 0 +2, action 4 -1 ; worker B: action 0 -3
        let ws = [w(&[(0, 6.0), (4, 0.0)]), w(&[(0, 1.0), (4, 1.0)])];
        let touched = merge_round(&mut table, &ws, 1.0, 1.0).unwrap();
        assert!(touched.contains(&k));
        let r = table[&k].regrets;
        // t=2, decay=0.5: a0 = 0.5*4 + (2 - 3) = 1 ; a4 = 0.5*1 + (-1 + 0) = -0.5 -> 0
        assert_eq!(r.get(0), Some(1.0));
        assert_eq!(r.get(4), Some(0.0));
        assert_eq!(table[&k].visit_count, 2);
    }
}
