//! PyO3 bindings: the `allin_core` Python module. Keep this layer thin —
//! conversion and argument checking only; logic lives in the pure-Rust modules.
//! The contract each binding implements is documented in
//! `backend/core-rs/parity/test_parity.py`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::river::tree::Node;
use crate::river::{Board, RiverCfr, RiverTree};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn vec1(a: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    a.as_array().to_vec()
}

fn veci(a: &PyReadonlyArray1<'_, i64>) -> Vec<i64> {
    a.as_array().to_vec()
}

fn arr2<'py>(py: Python<'py>, v: Vec<f64>, h: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = v.len().checked_div(h).unwrap_or(0);
    Ok(Array2::from_shape_vec((h, a), v)
        .map_err(|e| err(e.to_string()))?
        .into_pyarray(py))
}

/// Names of components that are ported and pass parity (see `crate::PORTED`).
#[pyfunction]
fn ported() -> Vec<&'static str> {
    crate::PORTED.to_vec()
}

#[pyfunction]
#[pyo3(signature = (raw, c1, c2, rv, final_pot, hero_total, exact=false))]
#[allow(clippy::too_many_arguments)]
fn showdown_measure<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray1<'py, f64>,
    c1: PyReadonlyArray1<'py, i64>,
    c2: PyReadonlyArray1<'py, i64>,
    rv: PyReadonlyArray1<'py, f64>,
    final_pot: f64,
    hero_total: f64,
    exact: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut b = Board::new(&vec1(&raw), &veci(&c1), &veci(&c2)).map_err(err)?;
    b.exact = exact;
    let rv = vec1(&rv);
    if rv.len() != b.h {
        return Err(err(format!("rv length {} != H={}", rv.len(), b.h)));
    }
    let mut out = vec![0.0; b.h];
    b.showdown_measure(&rv, final_pot, hero_total, &mut out);
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (c1, c2, rv, exact=false))]
fn compatible_mass<'py>(
    py: Python<'py>,
    c1: PyReadonlyArray1<'py, i64>,
    c2: PyReadonlyArray1<'py, i64>,
    rv: PyReadonlyArray1<'py, f64>,
    exact: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (c1, c2) = (veci(&c1), veci(&c2));
    let mut b = Board::new(&vec![0.0; c1.len()], &c1, &c2).map_err(err)?;
    b.exact = exact;
    let rv = vec1(&rv);
    if rv.len() != b.h {
        return Err(err(format!("rv length {} != H={}", rv.len(), b.h)));
    }
    let mut out = vec![0.0; b.h];
    b.compatible_mass(&rv, &mut out);
    Ok(PyArray1::from_vec(py, out))
}

fn build_tree(
    pot_entry: f64,
    stack: f64,
    menu: Vec<f64>,
    max_aggressions: u32,
) -> PyResult<RiverTree> {
    RiverTree::build(pot_entry, [stack, stack], &menu, max_aggressions).map_err(err)
}

/// Pre-order node list, same fields as `export_rust_fixtures._tree_to_list`.
#[pyfunction]
fn river_tree_nodes<'py>(
    py: Python<'py>,
    pot_entry: f64,
    stack: f64,
    menu: Vec<f64>,
    max_aggressions: u32,
) -> PyResult<Bound<'py, PyList>> {
    let t = build_tree(pot_entry, stack, menu, max_aggressions)?;
    let out = PyList::empty(py);
    for n in &t.nodes {
        let d = PyDict::new(py);
        match n {
            Node::Terminal(x) => {
                d.set_item("terminal", true)?;
                d.set_item("final_pot", x.final_pot)?;
                d.set_item("contrib", x.contrib.to_vec())?;
                d.set_item("folder", x.folder)?;
            }
            Node::Decision(x) => {
                d.set_item("terminal", false)?;
                d.set_item("player", x.player)?;
                d.set_item("to_call", x.to_call)?;
                d.set_item("pot_mid", x.pot_mid)?;
                d.set_item("sc", x.sc.to_vec())?;
                d.set_item("actions", x.actions.clone())?;
                d.set_item("node_id", x.node_id)?;
                d.set_item("agg", x.agg)?;
                d.set_item("last_inc", x.last_inc)?;
                d.set_item("children", x.children.clone())?;
            }
        }
        out.append(d)?;
    }
    Ok(out)
}

/// `RiverCFR(pot_entry, stack, menu, max_aggressions, raw, c1, c2)`.
#[pyclass(name = "RiverCFR", module = "allin_core")]
struct PyRiverCfr {
    inner: RiverCfr,
}

impl PyRiverCfr {
    fn nid(&self, nid: usize) -> PyResult<()> {
        if nid >= self.inner.tree.num_decision() {
            return Err(err(format!("node_id {nid} out of range")));
        }
        Ok(())
    }
}

#[pymethods]
impl PyRiverCfr {
    #[new]
    #[pyo3(signature = (pot_entry, stack, menu, max_aggressions, raw, c1, c2, exact=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        pot_entry: f64,
        stack: f64,
        menu: Vec<f64>,
        max_aggressions: u32,
        raw: PyReadonlyArray1<'_, f64>,
        c1: PyReadonlyArray1<'_, i64>,
        c2: PyReadonlyArray1<'_, i64>,
        exact: bool,
    ) -> PyResult<Self> {
        let tree = build_tree(pot_entry, stack, menu, max_aggressions)?;
        let mut board = Board::new(&vec1(&raw), &veci(&c1), &veci(&c2)).map_err(err)?;
        board.exact = exact;
        Ok(PyRiverCfr {
            inner: RiverCfr::new(tree, board),
        })
    }

    #[getter]
    fn n_decision(&self) -> usize {
        self.inner.tree.num_decision()
    }

    #[getter]
    fn exact(&self) -> bool {
        self.inner.board.exact
    }

    #[getter]
    fn iterations(&self) -> u64 {
        self.inner.iterations()
    }

    fn run(
        &mut self,
        py: Python<'_>,
        r0: PyReadonlyArray1<'_, f64>,
        r1: PyReadonlyArray1<'_, f64>,
        iters: u64,
    ) -> PyResult<()> {
        let (r0, r1) = (vec1(&r0), vec1(&r1));
        let inner = &mut self.inner;
        py.detach(|| inner.run(&r0, &r1, iters)).map_err(err)
    }

    fn run_gadget(
        &mut self,
        py: Python<'_>,
        hero: PyReadonlyArray1<'_, f64>,
        villain: PyReadonlyArray1<'_, f64>,
        optout: PyReadonlyArray1<'_, f64>,
        villain_seat: u8,
        iters: u64,
    ) -> PyResult<()> {
        let (hero, villain, optout) = (vec1(&hero), vec1(&villain), vec1(&optout));
        let inner = &mut self.inner;
        py.detach(|| inner.run_gadget(&hero, &villain, &optout, villain_seat, iters))
            .map_err(err)
    }

    fn average_strategy<'py>(
        &self,
        py: Python<'py>,
        nid: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.nid(nid)?;
        arr2(py, self.inner.average_strategy(nid), self.inner.h())
    }

    fn regret<'py>(&self, py: Python<'py>, nid: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.nid(nid)?;
        arr2(py, self.inner.regret(nid).to_vec(), self.inner.h())
    }

    fn current_values<'py>(
        &self,
        py: Python<'py>,
        r0: PyReadonlyArray1<'py, f64>,
        r1: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (r0, r1) = (vec1(&r0), vec1(&r1));
        let inner = &self.inner;
        let (v0, v1) = py.detach(|| inner.current_values(&r0, &r1)).map_err(err)?;
        Ok((PyArray1::from_vec(py, v0), PyArray1::from_vec(py, v1)))
    }

    fn exploitability(
        &self,
        py: Python<'_>,
        r0: PyReadonlyArray1<'_, f64>,
        r1: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        let (r0, r1) = (vec1(&r0), vec1(&r1));
        let inner = &self.inner;
        py.detach(|| inner.exploitability(&r0, &r1, None))
            .map_err(err)
    }

    fn gadget_regret<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match self.inner.gadget_regret() {
            None => Ok(None),
            Some(g) => Ok(Some(arr2(py, g.to_vec(), self.inner.h())?)),
        }
    }
}

#[pymodule]
fn allin_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", crate::VERSION)?;
    m.add_function(wrap_pyfunction!(ported, m)?)?;
    m.add_function(wrap_pyfunction!(showdown_measure, m)?)?;
    m.add_function(wrap_pyfunction!(compatible_mass, m)?)?;
    m.add_function(wrap_pyfunction!(river_tree_nodes, m)?)?;
    m.add_class::<PyRiverCfr>()?;
    crate::py_engine::register(m)?;
    crate::py_buckets::register(m)?;
    Ok(())
}
