//! PyO3 bindings for the MCCFR trainer (`allin_core.Trainer`).

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::buckets::Buckets;
use crate::mccfr::db::BlueprintDb;
use crate::mccfr::fixture::FixtureAbstraction;
use crate::mccfr::parallel::{
    RealAbstraction, RunStats, WorkerDelta, merge_round as merge_round_rs, train_parallel,
    train_single,
};
use crate::mccfr::rng::StreamRng;
use crate::mccfr::table::{ActMap, InfoSet, Legal, id_action, id_from_name, key_string, parse_key};
use crate::mccfr::{Schedule, Table, Trainer};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn sched(alpha: f64, gamma: f64) -> PyResult<Schedule> {
    if !(alpha.is_finite() && gamma.is_finite() && alpha >= 0.0 && gamma >= 0.0) {
        return Err(err("alpha and gamma must be finite and >= 0".into()));
    }
    Ok(Schedule { alpha, gamma })
}

/// `Trainer(menu_mode='control', alpha=1.5, gamma=1.0, key_format='v2')`.
#[pyclass(name = "Trainer", module = "allin_core")]
pub(crate) struct PyTrainer {
    pub(crate) inner: Trainer,
    abs: Option<Arc<RealAbstraction>>,
}

#[pymethods]
impl PyTrainer {
    #[new]
    #[pyo3(signature = (menu_mode="control", alpha=1.5, gamma=1.0, key_format="v2"))]
    fn new(menu_mode: &str, alpha: f64, gamma: f64, key_format: &str) -> PyResult<Self> {
        let mut inner = Trainer::new(menu_mode, sched(alpha, gamma)?).map_err(err)?;
        inner.key_format = crate::keys::KeyFormat::parse(key_format).map_err(err)?;
        Ok(PyTrainer { inner, abs: None })
    }

    #[getter]
    fn key_format(&self) -> &'static str {
        self.inner.key_format.name()
    }

    #[getter]
    fn num_info_sets(&self) -> usize {
        self.inner.table.len()
    }

    #[getter]
    fn ev_sum(&self) -> f64 {
        self.inner.ev_sum
    }

    #[getter]
    fn ev_count(&self) -> i64 {
        self.inner.ev_count
    }

    #[getter]
    fn menu_mode(&self) -> String {
        self.inner.menu_mode.clone()
    }

    /// Replay iterations `start..start+N` from fixture deals and an injected
    /// uniform stream (parity test). Returns the number of uniforms consumed.
    /// `deals[i]` = 9 card ids, `fine[i]` = 2 fine buckets, `post[i]` = 2x3
    /// postflop buckets (3/4/5 board cards), `raw[i]` = 2 seven-card ranks.
    #[pyo3(signature = (deals, fine, post, raw, fine_to_coarse, uniforms, start_iteration=0))]
    #[allow(clippy::too_many_arguments)]
    fn run_fixture(
        &mut self,
        py: Python<'_>,
        deals: Vec<Vec<u8>>,
        fine: Vec<Vec<u8>>,
        post: Vec<Vec<Vec<u16>>>,
        raw: Vec<Vec<i32>>,
        fine_to_coarse: Vec<u8>,
        uniforms: Vec<f64>,
        start_iteration: i64,
    ) -> PyResult<usize> {
        let n = deals.len();
        if fine.len() != n || post.len() != n || raw.len() != n {
            return Err(err("deals/fine/post/raw lengths differ".into()));
        }
        let mut abs = FixtureAbstraction::new(fine_to_coarse.clone());
        let mut cards9 = Vec::with_capacity(n);
        for i in 0..n {
            let c: [u8; 9] = deals[i]
                .as_slice()
                .try_into()
                .map_err(|_| err(format!("deal {i}: need 9 cards")))?;
            for (j, &x) in c.iter().enumerate() {
                if x >= 52 || c[..j].contains(&x) {
                    return Err(err(format!("deal {i}: bad or duplicate card {x}")));
                }
            }
            let f: [u8; 2] = fine[i]
                .as_slice()
                .try_into()
                .map_err(|_| err(format!("deal {i}: need 2 fine buckets")))?;
            if f.iter().any(|&b| b as usize >= fine_to_coarse.len()) {
                return Err(err(format!("deal {i}: fine bucket out of range")));
            }
            let mut p = [[0u16; 3]; 2];
            for s in 0..2 {
                p[s] = post[i]
                    .get(s)
                    .and_then(|v| v.as_slice().try_into().ok())
                    .ok_or_else(|| err(format!("deal {i}: need 2x3 postflop buckets")))?;
            }
            let r: [i32; 2] = raw[i]
                .as_slice()
                .try_into()
                .map_err(|_| err(format!("deal {i}: need 2 ranks")))?;
            abs.add(c, f, p, r);
            cards9.push(c);
        }
        let inner = &mut self.inner;
        py.detach(|| {
            let mut rng = StreamRng {
                vals: &uniforms,
                pos: 0,
            };
            for (i, c) in cards9.iter().enumerate() {
                inner.run_iteration(start_iteration + i as i64, *c, &abs, &mut rng)?;
            }
            Ok(rng.pos)
        })
        .map_err(err)
    }

    /// Persist the info sets touched since the last checkpoint + metadata,
    /// atomically. Returns the number of rows written.
    fn checkpoint(&mut self, py: Python<'_>, db_path: &str, iteration: i64) -> PyResult<usize> {
        let inner = &mut self.inner;
        py.detach(|| {
            let mut db = BlueprintDb::open(db_path)?;
            inner.checkpoint(&mut db, iteration)
        })
        .map_err(err)
    }

    /// Stamp `training_mode` / `menu_mode` like `run_blueprint_trainer`.
    fn stamp_mode(&self, db_path: &str, mode: &str) -> PyResult<()> {
        if mode != "single" && mode != "parallel" {
            return Err(err("mode must be 'single' or 'parallel'".into()));
        }
        let mut db = BlueprintDb::open(db_path).map_err(err)?;
        let mut meta = vec![
            ("training_mode", format!("\"{mode}\"")),
            ("menu_mode", format!("\"{}\"", self.inner.menu_mode)),
        ];
        if self.inner.key_format != crate::keys::KeyFormat::V2 {
            meta.push((
                "key_format",
                format!("\"{}\"", self.inner.key_format.name()),
            ));
        }
        db.save_checkpoint(&self.inner.table, std::iter::empty(), &meta)
            .map(|_| ())
            .map_err(err)
    }

    /// Load a blueprint and return the iteration to continue from.
    #[pyo3(signature = (db_path, mode=None))]
    fn resume(&mut self, py: Python<'_>, db_path: &str, mode: Option<String>) -> PyResult<i64> {
        let inner = &mut self.inner;
        py.detach(|| {
            let db = BlueprintDb::open_read_only(db_path)?;
            inner.resume(&db, mode.as_deref())
        })
        .map_err(err)
    }
}

fn act_map(d: &Bound<'_, PyDict>) -> PyResult<ActMap> {
    let mut m = ActMap::default();
    for (k, v) in d.iter() {
        let name: String = k.extract()?;
        let id = id_from_name(&name).map_err(err)?;
        m.set(id, v.extract::<f64>()?).map_err(err)?;
    }
    Ok(m)
}

fn act_dict<'py>(py: Python<'py>, m: &ActMap) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (a, v) in m.iter() {
        d.set_item(id_action(a).name(), v)?;
    }
    Ok(d)
}

fn legal_from(v: &Bound<'_, PyAny>) -> PyResult<Legal> {
    let names: Vec<String> = v.extract()?;
    let ids = names
        .iter()
        .map(|s| id_from_name(s))
        .collect::<Result<Vec<u8>, String>>()
        .map_err(err)?;
    Legal::from_ids(&ids).map_err(err)
}

fn item<'py>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(k)?.ok_or_else(|| err(format!("missing {k:?}")))
}

/// `parallel_trainer.merge_round` on plain dicts (parity contract).
#[pyfunction]
fn merge_round<'py>(
    py: Python<'py>,
    global: &Bound<'py, PyDict>,
    workers: Vec<Bound<'py, PyDict>>,
    alpha: f64,
    gamma: f64,
) -> PyResult<(Bound<'py, PyDict>, Vec<String>)> {
    let sch = sched(alpha, gamma)?;
    let mut table = Table::default();
    for (k, v) in global.iter() {
        let key = parse_key(&k.extract::<String>()?).map_err(err)?;
        let v = v.cast_into::<PyDict>()?;
        let is = InfoSet {
            legal: legal_from(&item(&v, "legal")?)?,
            regrets: act_map(&item(&v, "regret")?.cast_into()?)?,
            strategy: act_map(&item(&v, "strategy")?.cast_into()?)?,
            visit_count: item(&v, "visit_count")?.extract()?,
            strategy_visit_count: item(&v, "strategy_visit_count")?.extract()?,
            ..InfoSet::default()
        };
        table.insert(key, is);
    }
    let mut deltas = Vec::with_capacity(workers.len());
    for w in &workers {
        let mut d = WorkerDelta::default();
        for (col, out) in [("regret", &mut d.regret), ("strategy", &mut d.strategy)] {
            for (k, v) in item(w, col)?.cast_into::<PyDict>()?.iter() {
                let key = parse_key(&k.extract::<String>()?).map_err(err)?;
                out.insert(key, act_map(&v.cast_into()?)?);
            }
        }
        for (k, v) in item(w, "legal")?.cast_into::<PyDict>()?.iter() {
            let key = parse_key(&k.extract::<String>()?).map_err(err)?;
            d.legal.insert(key, legal_from(&v)?);
        }
        deltas.push(d);
    }
    let touched = merge_round_rs(&mut table, &deltas, sch.alpha, sch.gamma).map_err(err)?;
    let out = PyDict::new(py);
    for (k, is) in &table {
        let d = PyDict::new(py);
        let legal: Vec<String> = is
            .legal
            .ids()
            .iter()
            .map(|&a| id_action(a).name())
            .collect();
        d.set_item("legal", legal)?;
        d.set_item("regret", act_dict(py, &is.regrets)?)?;
        d.set_item("strategy", act_dict(py, &is.strategy)?)?;
        d.set_item("visit_count", is.visit_count)?;
        d.set_item("strategy_visit_count", is.strategy_visit_count)?;
        out.set_item(key_string(*k), d)?;
    }
    let mut t: Vec<String> = touched.into_iter().map(key_string).collect();
    t.sort();
    Ok((out, t))
}

#[pymethods]
impl PyTrainer {
    /// Load the card abstraction (baked flop/turn tables + centroids) used by
    /// `train`. `river_cache_boards` caps the canonical river-board cache.
    #[pyo3(signature = (abstractions_dir, river_cache_boards=None))]
    fn load_abstraction(
        &mut self,
        py: Python<'_>,
        abstractions_dir: &str,
        river_cache_boards: Option<usize>,
    ) -> PyResult<()> {
        let dir = std::path::PathBuf::from(abstractions_dir);
        let b = py
            .detach(|| {
                let b = Buckets::new(&dir, river_cache_boards)?;
                b.preload()?;
                Ok::<_, String>(b)
            })
            .map_err(err)?;
        self.abs = Some(Arc::new(RealAbstraction { buckets: b }));
        Ok(())
    }

    /// Train `iterations` from `start_iteration`. workers == 1: the
    /// single-thread path (per-visit discount, 'single' clocks); workers > 1:
    /// data-parallel rounds of `merge_every * workers` iterations ('parallel'
    /// clocks). `progress(dict)` (optional) is called after every round /
    /// `progress_every` single-thread iterations. Ctrl+C stops at the next
    /// round / progress boundary and returns with `interrupted=True`; the table
    /// is consistent there and `iterations` says how many completed, so the
    /// caller can checkpoint. Returns run stats.
    #[pyo3(signature = (iterations, start_iteration, seed, workers=1, merge_every=2000, progress=None, progress_every=10000))]
    #[allow(clippy::too_many_arguments)]
    fn train<'py>(
        &mut self,
        py: Python<'py>,
        iterations: u64,
        start_iteration: i64,
        seed: u64,
        workers: usize,
        merge_every: u64,
        progress: Option<Py<PyAny>>,
        progress_every: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if start_iteration < 0 {
            return Err(err("start_iteration must be >= 0".into()));
        }
        let abs = self
            .abs
            .clone()
            .ok_or_else(|| err("call load_abstraction(dir) before train()".into()))?;
        let inner = &mut self.inner;
        // Ctrl+C sets `interrupted` (returned to the caller, who knows exactly how
        // many iterations completed and can checkpoint); a failing progress
        // callback is re-raised.
        let mut py_err: Option<PyErr> = None;
        let mut interrupted = false;
        let report = |t: &Trainer,
                      s: &RunStats,
                      py_err: &mut Option<PyErr>,
                      interrupted: &mut bool|
         -> bool {
            Python::attach(|py| {
                if py.check_signals().is_err() {
                    *interrupted = true;
                    return false;
                }
                if let Some(cb) = &progress {
                    let d = PyDict::new(py);
                    let r = (|| {
                        d.set_item("iterations", s.iterations)?;
                        d.set_item("rounds", s.rounds)?;
                        d.set_item("seconds", s.seconds)?;
                        d.set_item("ev_sum", s.ev_sum)?;
                        d.set_item("ev_count", s.ev_count)?;
                        d.set_item("info_sets", t.table.len())?;
                        // Lifetime totals (the trainer itself is borrowed
                        // while training, so the callback can't read it).
                        d.set_item("total_ev_sum", t.ev_sum)?;
                        d.set_item("total_ev_count", t.ev_count)?;
                        cb.call1(py, (d,)).map(|_| ())
                    })();
                    if let Err(e) = r {
                        *py_err = Some(e);
                        return false;
                    }
                }
                true
            })
        };
        let stats = py
            .detach(|| {
                if workers <= 1 {
                    let mut total = RunStats::default();
                    let step = progress_every.max(1);
                    while total.iterations < iterations {
                        let n = step.min(iterations - total.iterations);
                        let s = train_single(
                            inner,
                            &*abs,
                            n,
                            start_iteration + total.iterations as i64,
                            seed,
                        )?;
                        total.iterations += s.iterations;
                        total.ev_sum += s.ev_sum;
                        total.ev_count += s.ev_count;
                        total.seconds += s.seconds;
                        if !report(inner, &total, &mut py_err, &mut interrupted) {
                            break;
                        }
                    }
                    Ok(total)
                } else {
                    train_parallel(
                        inner,
                        &*abs,
                        iterations,
                        start_iteration,
                        workers,
                        merge_every,
                        seed,
                        |t, s| report(t, s, &mut py_err, &mut interrupted),
                    )
                }
            })
            .map_err(err)?;
        if let Some(e) = py_err {
            return Err(e);
        }
        let d = PyDict::new(py);
        d.set_item("iterations", stats.iterations)?;
        d.set_item("rounds", stats.rounds)?;
        d.set_item("seconds", stats.seconds)?;
        d.set_item("ev_sum", stats.ev_sum)?;
        d.set_item("ev_count", stats.ev_count)?;
        d.set_item("interrupted", interrupted)?;
        Ok(d)
    }

    /// Raw cumulative (average-strategy) sums for `key`, or None — the input
    /// of `strategy_shape.strategy_shape_report`.
    fn cumulative_strategy<'py>(
        &self,
        py: Python<'py>,
        key: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Ok(k) = parse_key(key) else {
            return Ok(None);
        };
        match self.inner.table.get(&k) {
            None => Ok(None),
            Some(is) => Ok(Some(act_dict(py, &is.strategy)?)),
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrainer>()?;
    m.add_function(wrap_pyfunction!(merge_round, m)?)?;
    Ok(())
}
