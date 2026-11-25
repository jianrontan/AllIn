//! Data-biased response: fixed empirical human counts plus a free opponent.
//! Two role tables share the same packed key representation. No production
//! strategy is changed unless an explicitly forked DBR asset is selected.
use super::table::{ActMap, id_action, parse_key};
use crate::keys::action_char;
use rustc_hash::FxHashMap;
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Debug)]
pub struct ModelRow {
    pub counts: ActMap,
    pub n: f64,
}
#[derive(Debug)]
pub struct DbrModel {
    pub rows: FxHashMap<u64, ModelRow>,
    pub metadata: Value,
    pub digest: String,
}
impl DbrModel {
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let m: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if m["format"] != "allin-dbr-counts-v1" || m["estimator"] != "empirical-full-v1" {
            return Err("unsupported DBR count cache format/estimator".into());
        }
        for field in ["base_sha256", "split_sha256"] {
            if !m[field]
                .as_str()
                .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                return Err(format!("invalid DBR cache {field}"));
            }
        }
        let mut rows = FxHashMap::default();
        for (key, counts) in m["rows"].as_object().ok_or("missing DBR rows")? {
            let mut c = ActMap::default();
            let mut n = 0.0;
            for (ch, count) in counts.as_object().ok_or("invalid DBR count map")? {
                let v = count
                    .as_f64()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or("invalid DBR count")?;
                if ch.len() != 1 || !b"kcfsmlxo2a".contains(&ch.as_bytes()[0]) {
                    return Err("invalid DBR action character".into());
                }
                c.set(ch.as_bytes()[0], v)?;
                n += v;
            }
            if !n.is_finite() {
                return Err("DBR count sum overflow".into());
            }
            if n > 0.0 {
                rows.insert(parse_key(key)?, ModelRow { counts: c, n });
            }
        }
        Ok(Self {
            rows,
            metadata: m,
            digest: crate::buckets::sha1::sha1_hex(&bytes),
        })
    }
    pub fn validate_base(&self, db: &super::db::BlueprintDb, cap_jam: bool) -> Result<(), String> {
        let stamp = &self.metadata["stamp"];
        let kf = db.metadata("key_format")?.unwrap_or(json!("v2"));
        if stamp["key_format"] != kf
            || stamp["cap_jam"] != json!(cap_jam)
            || stamp["abstraction_id"] != db.metadata("abstraction_id")?.unwrap_or(Value::Null)
            || self.metadata["base_iterations"]
                != db.metadata("total_iterations")?.unwrap_or(Value::Null)
        {
            return Err("DBR cache/base stamp mismatch".into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForkRecipe {
    LegacyResetAll,
    PreserveRegretClock,
}
impl ForkRecipe {
    pub fn name(self) -> &'static str {
        match self {
            Self::LegacyResetAll => "reset_all_clocks_v1",
            Self::PreserveRegretClock => "preserve_regret_clock_v2",
        }
    }
    fn recorded(config: &Value) -> Result<Self, String> {
        match (config["version"].as_u64(), config.get("fork_recipe")) {
            (Some(1), None) => Ok(Self::LegacyResetAll),
            (Some(2), Some(Value::String(recipe)))
                if recipe == Self::PreserveRegretClock.name() =>
            {
                Ok(Self::PreserveRegretClock)
            }
            _ => Err("unknown DBR fork recipe/version".into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DbrConfig {
    pub model: Arc<DbrModel>,
    pub p_max: f64,
    pub strength: f64,
    pub fork_recipe: ForkRecipe,
}
impl DbrConfig {
    pub fn new(model: Arc<DbrModel>, p_max: f64, strength: f64) -> Result<Self, String> {
        if !p_max.is_finite()
            || !(0.0..=1.0).contains(&p_max)
            || !strength.is_finite()
            || strength <= 0.0
        {
            return Err("DBR requires p_max in [0,1] and positive finite strength".into());
        }
        Ok(Self {
            model,
            p_max,
            strength,
            fork_recipe: ForkRecipe::PreserveRegretClock,
        })
    }
    pub fn json(&self) -> Value {
        let legacy = self.fork_recipe == ForkRecipe::LegacyResetAll;
        let mut value = json!({"version":if legacy {1} else {2},"p_max":self.p_max,"strength":self.strength,
            "cache_sha1":self.model.digest,"base_sha256":self.model.metadata["base_sha256"],
            "split_sha256":self.model.metadata["split_sha256"],"estimator":"empirical-full-v1"});
        if !legacy {
            value["fork_recipe"] = json!(self.fork_recipe.name());
        }
        value
    }
    /// Unsupported observed mass stays free rather than being silently reassigned.
    /// With no supported observation this is exactly the free strategy.
    pub fn mixture(
        &self,
        key: u64,
        legal: &[u8],
        free: &[f64],
        out: &mut [f64],
    ) -> Result<f64, String> {
        out[..legal.len()].copy_from_slice(&free[..legal.len()]);
        let Some(row) = self.model.rows.get(&key) else {
            return Ok(0.0);
        };
        let mut counts = [0.0; super::table::MAX_ACTIONS];
        let mut supported = 0.0;
        for (i, &a) in legal.iter().enumerate() {
            counts[i] = row.counts.get(action_char(id_action(a))?).unwrap_or(0.0);
            supported += counts[i];
        }
        if supported <= 0.0 || self.p_max == 0.0 {
            return Ok(0.0);
        };
        let q = self.p_max * row.n / (self.strength + row.n) * (supported / row.n);
        for i in 0..legal.len() {
            out[i] = (1.0 - q) * free[i] + q * counts[i] / supported;
        }
        Ok(q)
    }
}

/// Retain regret mass AND its visit counter; start a fresh served average.
/// Last-seen iteration markers belong to the old iteration namespace and reset.
pub fn reset_fork(table: &mut super::trainer::Table) {
    for is in table.values_mut() {
        is.strategy = ActMap::default();
        is.last_visited = -1;
        is.strategy_visit_count = 0;
        is.last_strategy = -1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::table::action_id;
    use super::*;
    fn config() -> (u64, DbrConfig) {
        let k = parse_key("pf_1_ip_").unwrap();
        let mut c = ActMap::default();
        c.set(b'c', 10.0).unwrap();
        let rows = FxHashMap::from_iter([(k, ModelRow { counts: c, n: 10.0 })]);
        (
            k,
            DbrConfig::new(
                Arc::new(DbrModel {
                    rows,
                    metadata: json!({}),
                    digest: "test".into(),
                }),
                0.5,
                10.0,
            )
            .unwrap(),
        )
    }
    #[test]
    fn fork_preserves_discount_history_in_the_real_parallel_merge() {
        use super::super::parallel::{WorkerDelta, merge_round};
        use super::super::table::{InfoSet, Legal};
        use super::super::trainer::Table;
        let key = parse_key("pf_1_ip_").unwrap();
        let mut node = InfoSet::default();
        node.legal = Legal::from_ids(&[1, 2]).unwrap();
        node.regrets.set(1, 100.0).unwrap();
        node.regrets.set(2, 40.0).unwrap();
        node.strategy.set(1, 1000.0).unwrap();
        node.visit_count = 10;
        node.strategy_visit_count = 20;
        node.last_visited = 900;
        node.last_strategy = 901;
        let mut warm = Table::from_iter([(key, node)]);
        reset_fork(&mut warm);
        assert_eq!(warm[&key].visit_count, 10);
        assert_eq!(warm[&key].last_visited, -1);
        assert!(warm[&key].strategy.is_empty());
        assert_eq!(warm[&key].strategy_visit_count, 0);
        assert_eq!(warm[&key].last_strategy, -1);
        let mut cold = warm.clone();
        cold.get_mut(&key).unwrap().visit_count = 0;
        for table in [&mut warm, &mut cold] {
            for _ in 0..2 {
                // A zero-increment worker isolates the inherited regret term.
                // Exercise production merge_round, not a duplicate decay helper.
                let mut delta = WorkerDelta::default();
                delta.regret.insert(key, table[&key].regrets);
                merge_round(table, &[delta], 1.5, 1.0).unwrap();
            }
        }
        assert_eq!(warm[&key].visit_count, 12);
        assert!(
            (warm[&key].regrets.get(1).unwrap() - 100.0 * (10.0f64 / 12.0).powf(1.5)).abs() < 1e-12
        );
        assert!((cold[&key].regrets.get(1).unwrap() - 100.0 * 2.0f64.powf(-1.5)).abs() < 1e-12);
    }

    #[test]
    fn mixture_has_analytic_confidence_and_no_data_is_free() {
        let (k, c) = config();
        let legal = [action_id(crate::engine::Action::Call).unwrap(), 2];
        let mut out = [0.0; 2];
        assert_eq!(c.mixture(k, &legal, &[0.4, 0.6], &mut out).unwrap(), 0.25);
        assert!((out[0] - 0.55).abs() < 1e-14);
        assert!((out[1] - 0.45).abs() < 1e-14);
        assert_eq!(
            c.mixture(k + 1, &legal, &[0.4, 0.6], &mut out).unwrap(),
            0.0
        );
        assert_eq!(out, [0.4, 0.6]);
        assert_eq!(c.mixture(k, &[0, 2], &[0.4, 0.6], &mut out).unwrap(), 0.0);
        assert_eq!(out, [0.4, 0.6]);
    }
    #[test]
    fn v2_pairs_stamp_format_and_resume_legacy_implicit_format() {
        use super::super::db::BlueprintDb;
        use super::super::{Schedule, Trainer};
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let folder =
            std::env::temp_dir().join(format!("allin-dbr-v2-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&folder).unwrap();
        let bot_path = folder.join("bot.db");
        let opp_path = folder.join("opponent.db");
        let schedule = Schedule {
            alpha: 1.5,
            gamma: 1.0,
        };
        let mut trainer = Trainer::new("capped_pj", schedule).unwrap();
        let (_, config) = config();
        trainer.dbr = Some(config.clone());
        {
            let mut bot = BlueprintDb::open(bot_path.to_str().unwrap()).unwrap();
            let mut opponent = BlueprintDb::open(opp_path.to_str().unwrap()).unwrap();
            trainer
                .checkpoint_dbr(&mut bot, &mut opponent, -1, "test", &json!({}))
                .unwrap();
            assert_eq!(bot.metadata("key_format").unwrap(), Some(json!("v2")));
            assert_eq!(opponent.metadata("key_format").unwrap(), Some(json!("v2")));
        }
        for implicit_format in [false, true] {
            if implicit_format {
                for path in [&bot_path, &opp_path] {
                    let db = rusqlite::Connection::open(path).unwrap();
                    db.execute("DELETE FROM training_metadata WHERE key='key_format'", [])
                        .unwrap();
                }
            }
            let bot = BlueprintDb::open_read_only(bot_path.to_str().unwrap()).unwrap();
            let opponent = BlueprintDb::open_read_only(opp_path.to_str().unwrap()).unwrap();
            let mut resumed = Trainer::new("capped_pj", schedule).unwrap();
            assert_eq!(
                resumed
                    .resume_dbr(&bot, &opponent, config.clone(), &json!({}))
                    .unwrap(),
                0
            );
        }
        std::fs::remove_dir_all(folder).unwrap();
    }

    // ---- Traversal oracles (2026-11-25 review). They run the real trainer on a
    // tiny card abstraction and take subtree values from a paired P_max=0 run,
    // so they check the restricted-node maths without a toy-game refactor.
    use super::super::parallel::{train_parallel, train_single};
    use super::super::table::unpack_key;
    use super::super::{Abstraction, Schedule, Trainer};
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
    fn plain() -> Trainer {
        let mut t = Trainer::new("capped_pj", Schedule::default()).unwrap();
        t.key_format = crate::keys::KeyFormat::Spr5a;
        t
    }
    fn forked_table() -> super::super::trainer::Table {
        let mut base = plain();
        train_single(&mut base, &Abs, 200, 0, 5).unwrap();
        let mut table = base.table.clone();
        reset_fork(&mut table);
        table
    }
    fn dbr_trainer(
        table: &super::super::trainer::Table,
        rows: FxHashMap<u64, ModelRow>,
        p: f64,
    ) -> Trainer {
        let mut t = plain();
        t.table = table.clone();
        t.opponent_table = table.clone();
        let model = DbrModel {
            rows,
            metadata: json!({}),
            digest: "t".into(),
        };
        t.dbr = Some(DbrConfig::new(Arc::new(model), p, 10.0).unwrap());
        t
    }

    #[test]
    fn restricted_root_update_matches_closed_form() {
        // Iteration 2: bot_seat=1, updater=0, so the SB root is restricted AND updating.
        let it = 2;
        let mut table = forked_table();
        let mut probe = dbr_trainer(&table, FxHashMap::default(), 0.0);
        train_single(&mut probe, &Abs, 1, it, 711).unwrap();
        let root = *probe
            .opponent_dirty
            .iter()
            .find(|k| {
                let p = unpack_key(**k);
                p.seat == 0 && p.street == 0 && p.pattern.len == 0
            })
            .unwrap();
        let legal_set = probe.opponent_table[&root].legal;
        let legal: Vec<u8> = legal_set.ids().to_vec();
        {
            // Uniform free policy, no flooring, no decay.
            let is = table.entry(root).or_default();
            is.legal = legal_set;
            is.visit_count = 0;
            for &a in &legal {
                is.regrets.set(a, 1e6).unwrap();
            }
        }
        let rows = || {
            let mut c = ActMap::default();
            c.set(b'f', 6.0).unwrap();
            c.set(b'c', 4.0).unwrap();
            FxHashMap::from_iter([(root, ModelRow { counts: c, n: 10.0 })])
        };
        let mut d0 = dbr_trainer(&table, rows(), 0.0);
        let mut dp = dbr_trainer(&table, rows(), 0.8);
        train_single(&mut d0, &Abs, 1, it, 711).unwrap();
        train_single(&mut dp, &Abs, 1, it, 711).unwrap();
        let q = 0.8 * 10.0 / (10.0 + 10.0);
        let na = legal.len() as f64;
        let (fold, call) = (
            action_id(crate::engine::Action::Fold).unwrap(),
            action_id(crate::engine::Action::Call).unwrap(),
        );
        let mut expect_ev = 0.0;
        for &a in &legal {
            // At P_max=0 the increment is v_a - v_free, and v_free is the root value.
            let r0 = d0.opponent_table[&root].regrets.get(a).unwrap() - 1e6;
            let rp = dp.opponent_table[&root].regrets.get(a).unwrap() - 1e6;
            // Free regret is (1-q)(v_a - v_free): the FREE baseline, not the mixed one.
            assert!(
                (rp - (1.0 - q) * r0).abs() < 1e-6,
                "regret {a}: {rp} vs {}",
                (1.0 - q) * r0
            );
            let v = r0 + d0.ev_sum;
            let fixed = if a == fold {
                0.6
            } else if a == call {
                0.4
            } else {
                0.0
            };
            expect_ev += ((1.0 - q) / na + q * fixed) * v;
        }
        // The MIXED value passes up (to the root's EV here).
        assert!(
            (dp.ev_sum - expect_ev).abs() < 1e-6,
            "ev {} vs {expect_ev}",
            dp.ev_sum
        );
        // The bot role is untouched by restricted updates.
        assert_eq!(dp.table, d0.table);
    }

    #[test]
    fn non_updating_restricted_node_samples_the_mixture_and_averages_the_free_policy() {
        // Iteration 3: bot_seat=1, updater=1, so the SB root is restricted, not updating.
        let it = 3;
        let table = forked_table();
        let mut bot_changed_without_model = false;
        for seed in 900..908 {
            // Each seed deals a different SB hand, hence a different SB root key.
            let mut probe = dbr_trainer(&table, FxHashMap::default(), 0.0);
            train_single(&mut probe, &Abs, 1, it, seed).unwrap();
            let root = *probe
                .opponent_dirty
                .iter()
                .find(|k| {
                    let p = unpack_key(**k);
                    p.seat == 0 && p.street == 0 && p.pattern.len == 0
                })
                .unwrap();
            // All observed mass on fold with q ~ 1: the sampled SB action is a fold,
            // so the updating BB (bot) never acts and its table cannot change.
            let rows = || {
                let mut c = ActMap::default();
                c.set(b'f', 1e9).unwrap();
                FxHashMap::from_iter([(root, ModelRow { counts: c, n: 1e9 })])
            };
            let mut m = dbr_trainer(&table, rows(), 1.0);
            m.dbr.as_mut().unwrap().strength = 1e-9;
            let mut f = dbr_trainer(&table, rows(), 0.0);
            train_single(&mut m, &Abs, 1, it, seed).unwrap();
            train_single(&mut f, &Abs, 1, it, seed).unwrap();
            assert!(
                m.table == table,
                "sampling ignored the model mixture (seed {seed})"
            );
            // The opponent's average accumulates the FREE policy in both runs.
            assert_eq!(
                m.opponent_table[&root].strategy,
                f.opponent_table[&root].strategy
            );
            bot_changed_without_model |= f.table != table;
        }
        assert!(
            bot_changed_without_model,
            "the free policy never let the BB act"
        );
    }

    #[test]
    fn pmax0_dbr_equals_ordinary_training_on_each_role_pairing() {
        // Bot seat 0 + opponent seat 1 train on iterations = 0,1 (mod 4);
        // bot seat 1 + opponent seat 0 on iterations = 2,3 (mod 4).
        let table = forked_table();
        let mut d = dbr_trainer(&table, FxHashMap::default(), 0.0);
        let n = 400i64;
        train_single(&mut d, &Abs, n as u64, 0, 711).unwrap();
        let (mut a, mut b) = (plain(), plain());
        a.table = table.clone();
        b.table = table.clone();
        for it in 0..n {
            let t = if it.rem_euclid(4) < 2 { &mut a } else { &mut b };
            train_single(t, &Abs, 1, it, 711).unwrap();
        }
        let seat = |t: &super::super::trainer::Table, s: u8| {
            let mut v: Vec<_> = t
                .iter()
                .filter(|(k, _)| unpack_key(**k).seat == s)
                .map(|(k, x)| (*k, *x))
                .collect();
            v.sort_by_key(|x| x.0);
            v
        };
        assert_eq!(seat(&d.table, 0), seat(&a.table, 0));
        assert_eq!(seat(&d.opponent_table, 1), seat(&a.table, 1));
        assert_eq!(seat(&d.table, 1), seat(&b.table, 1));
        assert_eq!(seat(&d.opponent_table, 0), seat(&b.table, 0));
        assert_ne!(seat(&d.table, 0), seat(&table, 0));
    }

    #[test]
    fn parallel_merges_and_checkpoints_the_opponent_table() {
        let table = forked_table();
        let mut t = dbr_trainer(&table, FxHashMap::default(), 0.0);
        train_parallel(&mut t, &Abs, 64, 0, 2, 8, 711, |_, _| true).unwrap();
        let changed: Vec<u64> = t
            .opponent_table
            .iter()
            .filter(|(k, v)| table.get(k) != Some(v))
            .map(|(k, _)| *k)
            .collect();
        assert!(!changed.is_empty(), "opponent merge was dropped");
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let folder =
            std::env::temp_dir().join(format!("allin-dbr-par-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&folder).unwrap();
        let (bp, op) = (folder.join("bot.db"), folder.join("opponent.db"));
        {
            let mut bot = super::super::db::BlueprintDb::open(bp.to_str().unwrap()).unwrap();
            let mut opp = super::super::db::BlueprintDb::open(op.to_str().unwrap()).unwrap();
            t.checkpoint_dbr(&mut bot, &mut opp, 63, "g", &json!({}))
                .unwrap();
        }
        let opp = super::super::db::BlueprintDb::open_read_only(op.to_str().unwrap()).unwrap();
        let saved = opp.load_all().unwrap();
        for k in &changed {
            // Every merged opponent update reached the checkpoint (dirty tracking).
            let (s, m) = (&saved[k], &t.opponent_table[k]);
            assert_eq!(
                (s.regrets, s.strategy, s.visit_count),
                (m.regrets, m.strategy, m.visit_count)
            );
        }
        drop(opp);
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn dbr_rejects_pruning() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let folder =
            std::env::temp_dir().join(format!("allin-dbr-prune-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&folder).unwrap();
        let path = folder.join("x.db");
        drop(super::super::db::BlueprintDb::open(path.to_str().unwrap()).unwrap());
        let db = super::super::db::BlueprintDb::open_read_only(path.to_str().unwrap()).unwrap();
        let (_, cfg) = config();
        let mut t = plain();
        t.pruning = Some(super::super::pruning::PruneConfig::new(0.01, 100.0, 0.95).unwrap());
        assert!(t.fork_dbr(&db, cfg.model.clone(), 0.5, 10.0).is_err());
        assert!(t.resume_dbr(&db, &db, cfg, &json!({})).is_err());
        drop(db);
        std::fs::remove_dir_all(folder).unwrap();
    }
}

impl super::trainer::Trainer {
    pub fn fork_dbr(
        &mut self,
        db: &super::db::BlueprintDb,
        model: Arc<DbrModel>,
        p_max: f64,
        strength: f64,
    ) -> Result<i64, String> {
        if self.pruning.is_some() || self.dbr.is_some() {
            return Err("fork DBR only from an ordinary unpruned trainer".into());
        }
        model.validate_base(db, self.engine.preflop_cap_jam)?;
        let config = DbrConfig::new(model, p_max, strength)?;
        let base_iterations = self.resume(db, None)?;
        reset_fork(&mut self.table);
        self.opponent_table = self.table.clone();
        self.dirty = self.table.keys().copied().collect();
        self.opponent_dirty = self.opponent_table.keys().copied().collect();
        self.ev_sum = 0.0;
        self.ev_count = 0;
        self.dbr = Some(config);
        Ok(base_iterations)
    }
    pub fn checkpoint_dbr(
        &mut self,
        bot: &mut super::db::BlueprintDb,
        opp: &mut super::db::BlueprintDb,
        iteration: i64,
        generation: &str,
        run: &Value,
    ) -> Result<(usize, usize), String> {
        let cfg = self.dbr.as_ref().ok_or("not a DBR trainer")?;
        let mut meta = self.checkpoint_meta(iteration);
        // Ordinary v2 checkpoints omit this legacy-default field. Paired DBR
        // checkpoints stamp it explicitly so their resume contract is self-contained.
        if self.key_format == crate::keys::KeyFormat::V2 {
            meta.push(("key_format", json!("v2").to_string()));
        }
        meta.push(("dbr_generation", json!(generation).to_string()));
        meta.push(("dbr_run", run.to_string()));
        meta.push((
            "fork_base_iterations",
            cfg.model.metadata["base_iterations"].to_string(),
        ));
        for k in ["abstraction_id", "abstraction"] {
            if let Some(v) = cfg.model.metadata["stamp"].get(k) {
                meta.push((k, v.to_string()));
            }
        }
        meta.push(("dbr_role", json!("bot").to_string()));
        let b = bot.save_checkpoint(&self.table, self.dirty.iter(), &meta)?;
        meta.pop();
        meta.push(("dbr_role", json!("opponent").to_string()));
        let o = opp.save_checkpoint(&self.opponent_table, self.opponent_dirty.iter(), &meta)?;
        // The Python driver publishes the immutable pair only after BOTH writes succeed.
        self.dirty.clear();
        self.opponent_dirty.clear();
        Ok((b, o))
    }
    pub fn resume_dbr(
        &mut self,
        bot: &super::db::BlueprintDb,
        opp: &super::db::BlueprintDb,
        mut config: DbrConfig,
        run: &Value,
    ) -> Result<i64, String> {
        if self.pruning.is_some() {
            return Err("DBR + pruning is not supported".into());
        }
        let recorded = bot.metadata("dbr")?.ok_or("missing DBR configuration")?;
        config.fork_recipe = ForkRecipe::recorded(&recorded)?;
        let generation = bot
            .metadata("dbr_generation")?
            .ok_or("missing DBR pair generation")?;
        let total = bot
            .metadata("total_iterations")?
            .and_then(|v| v.as_i64())
            .ok_or("missing DBR iteration")?;
        for (db, role) in [(bot, "bot"), (opp, "opponent")] {
            for (k, want) in [
                ("dbr", config.json()),
                ("dbr_run", run.clone()),
                ("dbr_role", json!(role)),
                ("dbr_generation", generation.clone()),
                ("total_iterations", json!(total)),
                ("key_format", json!(self.key_format.name())),
                ("menu_mode", json!(self.menu_mode)),
                ("alpha", json!(self.sched.alpha)),
                ("gamma", json!(self.sched.gamma)),
            ] {
                let stored = db
                    .metadata(k)?
                    .or_else(|| (k == "key_format").then(|| json!("v2")));
                if stored.as_ref() != Some(&want) {
                    return Err(format!("DBR resume mismatch: {k} ({role})"));
                }
            }
        }
        let b = bot.load_all()?;
        let o = opp.load_all()?;
        if total < 0
            || b.values()
                .chain(o.values())
                .any(|v| v.last_visited >= total || v.last_strategy >= total)
        {
            return Err("DBR resume iteration is behind role-table clocks".into());
        }
        self.table = b;
        self.opponent_table = o;
        self.dbr = Some(config);
        self.dirty.clear();
        self.opponent_dirty.clear();
        self.ev_sum = bot
            .metadata("ev_sum")?
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        self.ev_count = bot
            .metadata("ev_count")?
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        Ok(total)
    }
}

pub fn free_increment(value: f64, free_value: f64, q: f64) -> f64 {
    (1.0 - q) * (value - free_value)
}
