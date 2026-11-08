//! PyO3 bindings for the betting engine and info-set keys (contracts in
//! `parity/test_parity.py`, sections `keys` and `engine`). Conversion and
//! validation only; logic lives in `engine.rs` / `keys.rs`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::engine::{self, Action, Engine, Kind, NodeState};
use crate::keys;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// Engine card string (`SuitRank`, e.g. `HA`) -> id = rank_idx * 4 + suit_idx
/// with suits H, D, C, S (the `DECK` order used everywhere in Python).
pub(crate) fn card_id(s: &str) -> PyResult<u8> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return Err(err(format!("bad card {s:?}")));
    }
    let suit = b"HDCS".iter().position(|&c| c == b[0]);
    let rank = b"23456789TJQKA".iter().position(|&c| c == b[1]);
    match (suit, rank) {
        (Some(su), Some(r)) => Ok((r * 4 + su) as u8),
        _ => Err(err(format!("bad card {s:?}"))),
    }
}

fn cards(v: &[String]) -> PyResult<Vec<u8>> {
    let ids = v
        .iter()
        .map(|s| card_id(s))
        .collect::<PyResult<Vec<u8>>>()?;
    for (i, a) in ids.iter().enumerate() {
        if ids[..i].contains(a) {
            return Err(err(format!("duplicate card {}", v[i])));
        }
    }
    Ok(ids)
}

fn street_arg(street: i64) -> PyResult<u8> {
    if (0..=3).contains(&street) {
        Ok(street as u8)
    } else {
        Err(err(format!("street must be 0..3, got {street}")))
    }
}

fn seat_arg(cp: i64) -> PyResult<u8> {
    if cp == 0 || cp == 1 {
        Ok(cp as u8)
    } else {
        Err(err(format!("player must be 0 or 1, got {cp}")))
    }
}

fn finite(x: f64, what: &str) -> PyResult<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(err(format!("{what} must be finite")))
    }
}

fn stack_arg(x: f64) -> PyResult<f64> {
    if x.is_finite() && x >= 0.0 {
        Ok(x)
    } else {
        Err(err("stack_cp must be finite and >= 0".into()))
    }
}

fn get<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<T> {
    d.get_item(k)?
        .ok_or_else(|| err(format!("state dict missing {k:?}")))?
        .extract::<T>()
        .map_err(|e| {
            let e: PyErr = e.into();
            err(format!("state[{k:?}]: {e}"))
        })
}

fn state_from(d: &Bound<'_, PyDict>) -> PyResult<NodeState> {
    let c: Vec<f64> = get(d, "c")?;
    if c.len() != 2 {
        return Err(err("state['c'] must have 2 entries".into()));
    }
    let kind = |k: &str| -> PyResult<Kind> {
        let s: String = get(d, k)?;
        Kind::parse(&s).ok_or_else(|| err(format!("state[{k:?}]: bad kind {s:?}")))
    };
    Ok(NodeState {
        pot: finite(get(d, "pot")?, "pot")?,
        c: [finite(c[0], "c")?, finite(c[1], "c")?],
        bet_to: finite(get(d, "bet_to")?, "bet_to")?,
        prev_bet_to: finite(get(d, "prev_bet_to")?, "prev_bet_to")?,
        num_br: get(d, "num_br")?,
        last_kind: kind("last_kind")?,
        prev_kind: kind("prev_kind")?,
        allin_seen: get(d, "allin_seen")?,
    })
}

fn state_to<'py>(py: Python<'py>, st: &NodeState) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("pot", st.pot)?;
    d.set_item("c", st.c.to_vec())?;
    d.set_item("bet_to", st.bet_to)?;
    d.set_item("prev_bet_to", st.prev_bet_to)?;
    d.set_item("num_br", st.num_br)?;
    d.set_item("last_kind", st.last_kind.name())?;
    d.set_item("prev_kind", st.prev_kind.name())?;
    d.set_item("allin_seen", st.allin_seen)?;
    Ok(d)
}

fn action(s: &str) -> PyResult<Action> {
    Action::parse(s).map_err(err)
}

fn history(h: &[String]) -> PyResult<Vec<Action>> {
    h.iter().map(|s| action(s)).collect()
}

#[pyfunction]
#[pyo3(signature = (street, position, preflop_bucket, strength, pattern, spr=None))]
fn make_info_set_key(
    street: i64,
    position: &str,
    preflop_bucket: &str,
    strength: Option<i64>,
    pattern: &str,
    spr: Option<u8>,
) -> PyResult<String> {
    let street = street_arg(street)?;
    if street == 0 {
        if spr.is_some() {
            return Err(err("preflop keys carry no SPR bucket".into()));
        }
        // Python formats the bucket id verbatim preflop.
        return Ok(format!("{preflop_bucket}_{position}_{pattern}"));
    }
    let fine = keys::parse_bucket(preflop_bucket).map_err(err)?;
    keys::make_info_set_key(
        street,
        position,
        fine,
        strength,
        pattern,
        crate::buckets::fine_to_coarse(),
        spr,
    )
    .map_err(err)
}

/// `keys.spr_bucket(key_format, street, street_pot)`.
#[pyfunction]
fn spr_bucket(key_format: &str, street: i64, street_pot: f64) -> PyResult<Option<u8>> {
    let fmt = keys::KeyFormat::parse(key_format).map_err(err)?;
    keys::spr_bucket(fmt, street_arg(street)?, street_pot).map_err(err)
}

#[pyfunction]
fn action_char(a: &str) -> PyResult<String> {
    let c = keys::action_char(action(a)?).map_err(err)?;
    Ok((c as char).to_string())
}

/// `Engine(mode)`: the trainer's state-threaded betting engine.
#[pyclass(name = "Engine", module = "allin_core")]
struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new(mode: &str) -> PyResult<Self> {
        Ok(PyEngine {
            inner: Engine::for_mode(mode).map_err(err)?,
        })
    }

    fn init_node_state<'py>(
        &self,
        py: Python<'py>,
        street: i64,
        starting_pot: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let st = self
            .inner
            .init_node_state(street_arg(street)?, finite(starting_pot, "starting_pot")?);
        state_to(py, &st)
    }

    fn acting_player(&self, n: usize, street: i64) -> PyResult<u8> {
        Ok(Engine::acting_player(n, street_arg(street)?))
    }

    fn legal_actions(
        &self,
        street: i64,
        st: &Bound<'_, PyDict>,
        cp: i64,
        stack_cp: f64,
    ) -> PyResult<Vec<String>> {
        let st = state_from(st)?;
        let legal = self.inner.legal_actions(
            street_arg(street)?,
            &st,
            seat_arg(cp)?,
            stack_arg(stack_cp)?,
        );
        Ok(legal.into_iter().map(Action::name).collect())
    }

    fn action_cost(
        &self,
        action_name: &str,
        street: i64,
        st: &Bound<'_, PyDict>,
        cp: i64,
        stack_cp: f64,
    ) -> PyResult<f64> {
        let st = state_from(st)?;
        self.inner
            .action_cost(
                action(action_name)?,
                street_arg(street)?,
                &st,
                seat_arg(cp)?,
                stack_arg(stack_cp)?,
            )
            .map_err(err)
    }

    #[allow(clippy::too_many_arguments)]
    fn advance<'py>(
        &self,
        py: Python<'py>,
        st: &Bound<'py, PyDict>,
        action_name: &str,
        street: i64,
        cp: i64,
        stack_cp: f64,
        p_inv: (f64, f64),
    ) -> PyResult<Bound<'py, PyDict>> {
        let st = state_from(st)?;
        let new = self
            .inner
            .advance(
                &st,
                action(action_name)?,
                street_arg(street)?,
                seat_arg(cp)?,
                stack_arg(stack_cp)?,
                [finite(p_inv.0, "p_inv")?, finite(p_inv.1, "p_inv")?],
            )
            .map_err(err)?;
        state_to(py, &new)
    }

    fn is_terminal(&self, history_names: Vec<String>, street: i64) -> PyResult<bool> {
        Ok(engine::is_terminal(
            &history(&history_names)?,
            street_arg(street)?,
        ))
    }

    /// P0-perspective utility with the threaded final pot / P0 total.
    /// `starting_pot` and the investments are accepted for contract fidelity
    /// (the threaded values make them unnecessary, as in Python).
    #[allow(clippy::too_many_arguments)]
    fn utility(
        &self,
        p0_cards: Vec<String>,
        p1_cards: Vec<String>,
        board: Vec<String>,
        history_names: Vec<String>,
        street: i64,
        starting_pot: f64,
        p0_inv: f64,
        p1_inv: f64,
        final_pot: f64,
        p0_total: f64,
    ) -> PyResult<f64> {
        let _ = (starting_pot, p0_inv, p1_inv);
        let mut all = p0_cards.clone();
        all.extend(p1_cards.iter().cloned());
        all.extend(board.iter().cloned());
        let ids = cards(&all)?;
        if p0_cards.len() != 2 || p1_cards.len() != 2 || board.len() > 5 {
            return Err(err("need 2+2 hole cards and at most 5 board cards".into()));
        }
        let h = history(&history_names)?;
        let street = street_arg(street)?;
        // Every trainer showdown is on the full board (river, or an all-in
        // run-out); anything shorter would rank incomplete hands.
        if !h.contains(&Action::Fold) && board.len() < 5 {
            return Err(err("a showdown needs all 5 board cards".into()));
        }
        Ok(engine::utility(
            [ids[0], ids[1]],
            [ids[2], ids[3]],
            &ids[4..],
            &h,
            street,
            finite(final_pot, "final_pot")?,
            finite(p0_total, "p0_total")?,
            crate::eval::rank,
        ))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(make_info_set_key, m)?)?;
    m.add_function(wrap_pyfunction!(spr_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(action_char, m)?)?;
    m.add_class::<PyEngine>()?;
    Ok(())
}
