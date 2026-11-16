//! Experimental zero-probability action pruning. Signed shadow scores and raw
//! visit counts have the regret clock's discount; workers never floor them.
use super::table::ActMap;
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PruneConfig {
    pub c: f64,
    pub n_min: f64,
    pub probability: f64,
}
impl PruneConfig {
    pub fn new(c: f64, n_min: f64, probability: f64) -> Result<Self, String> {
        if !c.is_finite()
            || c <= 0.0
            || !n_min.is_finite()
            || n_min < 1.0
            || !probability.is_finite()
            || !(0.0..=1.0).contains(&probability)
        {
            return Err("pruning needs c > 0, n_min >= 1 and probability in [0,1]".into());
        }
        Ok(Self {
            c,
            n_min,
            probability,
        })
    }
    pub fn json(self) -> serde_json::Value {
        serde_json::json!({"version": 1, "c": self.c, "n_min": self.n_min, "probability": self.probability})
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct PruneState {
    pub shadow: ActMap,
    pub visits: f64,
    /// Largest pot encountered at this key; conservative bound for its recovery floor.
    pub pot: f64,
    pub dormant: u16,
}
pub type PruneTable = FxHashMap<u64, PruneState>;
impl PruneState {
    pub fn floor(&mut self, config: PruneConfig) -> Result<(), String> {
        let floor = -config.c * self.pot * self.visits * (31.0 / 30.0);
        let old = self.shadow;
        for (a, s) in old.iter() {
            self.shadow.set(a, s.max(floor))?;
        }
        Ok(())
    }
    /// Clear dormant bits once regrets are final for this visit or merge round and
    /// return how many actions re-activated: a dormant action whose regret turned
    /// positive, or every dormant action when the node has no positive regret (it then
    /// plays uniformly). A bit whose action is merely no longer eligible (its score
    /// recovered past the threshold, or a smaller pot) is cleared without counting, so
    /// the re-activation count means "a pruned action regained probability" and a
    /// count of zero is a real alarm. Worker bits are provisional; only the master's
    /// merge (or a single-threaded node) calls this.
    pub fn refresh_dormant(&mut self, regrets: &ActMap, config: PruneConfig) -> u32 {
        let old = self.dormant;
        if !regrets.iter().any(|(_, r)| r > 0.0) {
            self.dormant = 0;
            return old.count_ones();
        }
        let mut revived = 0u32;
        for a in 0..16u8 {
            let bit = 1u16 << a;
            if old & bit == 0 {
                continue;
            }
            if regrets.get(a).unwrap_or(0.0) > 0.0 {
                self.dormant &= !bit;
                revived += 1;
            } else if !self.eligible(a, self.pot, config) {
                self.dormant &= !bit;
            }
        }
        revived
    }

    pub fn eligible(&self, a: u8, pot: f64, config: PruneConfig) -> bool {
        self.visits >= config.n_min
            && self.shadow.get(a).unwrap_or(0.0) < -config.c * pot * self.visits
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct PruneStats {
    pub skipped: u64,
    pub explored: u64,
    pub dormant: u64,
    pub reactivated: u64,
    pub update_mass: f64,
    pub low_reach_mass: f64,
}
impl PruneStats {
    pub fn add(&mut self, b: Self) {
        self.skipped += b.skipped;
        self.explored += b.explored;
        self.dormant += b.dormant;
        self.reactivated += b.reactivated;
        self.update_mass += b.update_mass;
        self.low_reach_mass += b.low_reach_mass;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mccfr::db::BlueprintDb;
    use crate::mccfr::parallel::{train_parallel, train_single};
    use crate::mccfr::{Abstraction, Schedule, Trainer};
    struct Abs;
    impl Abstraction for Abs {
        fn preflop_fine(&self, h: [u8; 2]) -> u8 {
            h[0]
        }
        fn coarse(&self, _: u8) -> u8 {
            0
        }
        fn postflop(&self, _: [u8; 2], _: &[u8]) -> Result<u16, String> {
            Ok(0)
        }
        fn rank(&self, c: &[u8]) -> i32 {
            c[0] as i32
        }
    }
    fn trainer(config: Option<PruneConfig>) -> Trainer {
        let mut t = Trainer::new("capped_pj", Schedule::default()).unwrap();
        t.key_format = crate::keys::KeyFormat::Spr5a;
        t.pruning = config;
        t
    }
    #[test]
    fn disabled_coin_preserves_every_cfr_value() {
        let mut a = trainer(None);
        let mut b = trainer(Some(PruneConfig::new(0.01, 1.0, 0.0).unwrap()));
        train_single(&mut a, &Abs, 120, 0, 711).unwrap();
        train_single(&mut b, &Abs, 120, 0, 711).unwrap();
        assert_eq!(a.table, b.table);
        assert_eq!(a.ev_sum, b.ev_sum);
        assert_eq!(b.prune_stats.skipped, 0);
    }
    #[test]
    fn enabled_pruning_skips_and_preserves_shadow_bound() {
        let config = PruneConfig::new(1e-8, 1.0, 0.95).unwrap();
        let mut t = trainer(Some(config));
        train_single(&mut t, &Abs, 250, 0, 711).unwrap();
        assert!(t.prune_stats.skipped > 0);
        assert!(t.prune_stats.dormant > 0);
        assert!(t.prune_stats.reactivated > 0);
        for (k, state) in &t.prune_table {
            for a in 0..16 {
                if state.dormant & (1 << a) != 0 {
                    assert_eq!(t.table[k].regrets.get(a).unwrap_or(0.0), 0.0);
                }
            }
            for (a, s) in state.shadow.iter() {
                assert!(s <= t.table[k].regrets.get(a).unwrap_or(0.0) + 1e-8);
                assert!(s >= -config.c * state.pot * state.visits * 31.0 / 30.0 - 1e-8);
            }
        }
    }
    #[test]
    fn parallel_merges_shadow_increments_once() {
        let config = PruneConfig::new(1e-8, 1.0, 0.95).unwrap();
        let mut t = trainer(Some(config));
        train_parallel(&mut t, &Abs, 250, 0, 2, 25, 711, |_, _| true).unwrap();
        assert!(t.prune_stats.skipped > 0);
        for (k, state) in &t.prune_table {
            assert!(state.visits > 0.0);
            for a in 0..16 {
                if state.dormant & (1 << a) != 0 {
                    assert_eq!(t.table[k].regrets.get(a).unwrap_or(0.0), 0.0);
                }
            }
            for (a, s) in state.shadow.iter() {
                assert!(s <= t.table[k].regrets.get(a).unwrap_or(0.0) + 1e-8);
            }
        }
    }
    #[test]
    fn checkpoint_resume_preserves_scores_counts_and_refuses_mismatch() {
        let config = PruneConfig::new(1e-8, 1.0, 0.95).unwrap();
        let mut full = trainer(Some(config));
        train_single(&mut full, &Abs, 200, 0, 711).unwrap();
        let mut t = trainer(Some(config));
        train_single(&mut t, &Abs, 100, 0, 711).unwrap();
        let path = std::env::temp_dir().join(format!("allin-prune-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut db = BlueprintDb::open(path.to_str().unwrap()).unwrap();
        t.checkpoint(&mut db, 99).unwrap();
        assert!(
            trainer(None)
                .resume(&db, None)
                .unwrap_err()
                .contains("pruning mismatch")
        );
        let mut resumed = trainer(Some(config));
        let start = resumed.resume(&db, None).unwrap();
        train_single(&mut resumed, &Abs, 100, start, 711).unwrap();
        assert_eq!(full.table, resumed.table);
        assert_eq!(full.ev_sum, resumed.ev_sum);
        for (k, s) in &full.prune_table {
            let b = resumed.prune_table[k];
            assert_eq!(s.shadow, b.shadow);
            assert_eq!(s.visits, b.visits);
            assert_eq!(s.pot, b.pot);
        }
        drop(db);
        let _ = std::fs::remove_file(&path);
    }
}
