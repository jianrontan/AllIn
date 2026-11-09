//! AllIn Rust core.
//!
//! The Python code under `backend/bot/src/` is the reference implementation:
//! every component ported here must match it on the golden fixtures produced
//! by `backend/bot/scripts/export_rust_fixtures.py` (checked by
//! `backend/core-rs/parity/`). See `docs/RUST_CORE_PLAN.md`.
//!
//! Layout: pure-Rust modules (testable with plain `cargo test`) + a thin PyO3
//! layer in `python.rs`, compiled only with the `python` feature.

// Explicit index loops are deliberate: they mirror the numpy expressions of the
// Python reference line by line, which is what parity review needs.
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::collapsible_if,
    clippy::field_reassign_with_default
)]

/// Components that are ported AND pass parity. The parity harness skips
/// fixtures for any component not listed here, so add a name only once its
/// parity tests are green. Names match the fixture groups in
/// `export_rust_fixtures.py` (`COMPONENTS`).
pub const PORTED: &[&str] = &[
    "river_showdown",
    "river_tree",
    "river_cfr",
    "river_gadget",
    "eval_rank7",
    "buckets",
    "keys",
    "engine",
    "mccfr",
    "mccfr_merge",
    "keys_spr",
    "mccfr_spr",
    "engine_pj",
    "mccfr_pj",
];

/// Crate version, surfaced to Python for diagnostics.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod buckets;
pub mod engine;
pub mod eval;
pub mod keys;
pub mod mccfr;
pub mod river;

#[cfg(feature = "python")]
mod py_buckets;
#[cfg(feature = "python")]
mod py_engine;
#[cfg(feature = "python")]
mod py_train;
#[cfg(feature = "python")]
mod python;
