//! River subgame solver (Phase 1): a port of
//! `backend/bot/src/evaluation/showdown_kernel.py` (the showdown / card-removal
//! kernel), `backend/bot/src/subgame/river_tree.py` (the betting tree) and
//! `backend/bot/src/subgame/river_cfr.py` (two-sided vectorized CFR+, the safe
//! re-solving gadget, best response). The Python code is the reference; parity
//! is checked by `backend/core-rs/parity/test_parity.py`.

pub mod cfr;
pub mod showdown;
pub mod tree;

pub use cfr::RiverCfr;
pub use showdown::Board;
pub use tree::{Node, RiverTree};
