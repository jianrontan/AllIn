//! PyO3 bindings for Phase 2: `eval_rank7` (`rank_batch`) and `buckets`
//! (`preflop_buckets`, `Buckets`). Contracts: `parity/test_parity.py`.
//!
//! Card arrays are 2-D `[N, k]` of card ids (`rank_idx * 4 + suit_idx`); uint8
//! is the contract dtype, any other integer dtype is accepted and
//! range-checked. All validation errors are `ValueError`; batch work runs with
//! the GIL released.

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::buckets;
use crate::eval;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// Copy a 2-D integer array into row-major u8 ids; returns (ids, row width).
fn rows(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<u8>, usize)> {
    fn conv<T: numpy::Element + Copy + TryInto<u8> + std::fmt::Debug>(
        a: PyReadonlyArray2<'_, T>,
    ) -> PyResult<(Vec<u8>, usize)> {
        let v = a.as_array();
        let w = v.ncols();
        let mut out = Vec::with_capacity(v.len());
        for &x in v.iter() {
            out.push(
                x.try_into()
                    .map_err(|_| err(format!("card id {x:?} out of range 0..52")))?,
            );
        }
        Ok((out, w))
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray2<'_, u8>>() {
        let v = a.as_array();
        return Ok((v.iter().copied().collect(), v.ncols()));
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray2<'_, i64>>() {
        return conv(a);
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray2<'_, i32>>() {
        return conv(a);
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray2<'_, i16>>() {
        return conv(a);
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray2<'_, i8>>() {
        return conv(a);
    }
    Err(PyTypeError::new_err(
        "expected a 2-D integer numpy array of card ids [N, k]",
    ))
}

/// `rank_batch(cards: uint8[N, k]) -> int32[N]`, k = 5, 6, 7 (lower = stronger).
#[pyfunction]
fn rank_batch<'py>(
    py: Python<'py>,
    cards: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let (ids, k) = rows(cards)?;
    let out = py.detach(|| eval::rank_batch(&ids, k)).map_err(err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// `preflop_buckets(cards: uint8[N, 2]) -> (fine int32[N], coarse int32[N])`.
#[pyfunction]
fn preflop_buckets<'py>(
    py: Python<'py>,
    cards: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
    let (ids, k) = rows(cards)?;
    if k != 2 {
        return Err(err(format!("preflop rows must have 2 cards, got {k}")));
    }
    let (f, c) = py.detach(|| buckets::preflop_batch(&ids)).map_err(err)?;
    Ok((PyArray1::from_vec(py, f), PyArray1::from_vec(py, c)))
}

/// `fine_to_coarse() -> list[int]` (`card_abstractions.FINE_TO_COARSE`).
#[pyfunction]
fn fine_to_coarse() -> Vec<u8> {
    buckets::fine_to_coarse().to_vec()
}

/// `Buckets(abstractions_dir, river_cache_boards=None)`.
#[pyclass(name = "Buckets", module = "allin_core", frozen)]
struct PyBuckets {
    inner: Arc<buckets::Buckets>,
}

#[pymethods]
impl PyBuckets {
    #[new]
    #[pyo3(signature = (abstractions_dir, river_cache_boards=None))]
    fn new(
        py: Python<'_>,
        abstractions_dir: PathBuf,
        river_cache_boards: Option<usize>,
    ) -> PyResult<Self> {
        let b = py
            .detach(|| buckets::Buckets::new(&abstractions_dir, river_cache_boards))
            .map_err(err)?;
        Ok(PyBuckets { inner: Arc::new(b) })
    }

    /// `situation_ids(cards: uint8[N, 2 + nb]) -> int64[N]`.
    fn situation_ids<'py>(
        &self,
        py: Python<'py>,
        cards: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (ids, w) = rows(cards)?;
        let out = py.detach(|| buckets::situation_ids(&ids, w)).map_err(err)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// `bucket_batch(cards: uint8[N, 2 + nb]) -> int32[N]`.
    fn bucket_batch<'py>(
        &self,
        py: Python<'py>,
        cards: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let (ids, w) = rows(cards)?;
        let inner = &self.inner;
        let out = py.detach(|| inner.bucket_batch(&ids, w)).map_err(err)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Load and stamp-check every baked table now (otherwise lazy on first use).
    fn preload(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.detach(|| inner.preload()).map_err(err)
    }

    /// Number of clusters per street (0 flop, 1 turn, 2 river).
    fn n_buckets(&self, street: usize) -> PyResult<usize> {
        if street > 2 {
            return Err(err(format!("street index {street} not in 0..3")));
        }
        Ok(self.inner.n_buckets(street))
    }

    /// `(entries, capacity, hits, misses)` of the per-board river equity cache.
    fn river_cache_info(&self) -> (usize, usize, u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let c = self.inner.river_cache();
        (
            c.len(),
            c.capacity(),
            c.hits.load(Relaxed),
            c.misses.load(Relaxed),
        )
    }

    fn clear_river_cache(&self) {
        self.inner.clear_river_cache();
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rank_batch, m)?)?;
    m.add_function(wrap_pyfunction!(preflop_buckets, m)?)?;
    m.add_function(wrap_pyfunction!(fine_to_coarse, m)?)?;
    m.add_class::<PyBuckets>()?;
    Ok(())
}
