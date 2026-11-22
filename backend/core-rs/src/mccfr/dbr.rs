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

    #[test]
    fn free_regret_uses_free_baseline_and_mixture_value_passes_up() {
        // Free=(1,0), fixed=(0,1), q=.25, utilities=(0,10).
        // Mixed value2.5; changing the free policy gains7.5, not5.625.
        let free_value = 0.0;
        let mixed_value = 2.5;
        assert_eq!(free_increment(10.0, free_value, 0.25), 7.5);
        assert_ne!((1.0 - 0.25) * (10.0 - mixed_value), 7.5);
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
