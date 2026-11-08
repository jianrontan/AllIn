//! Blueprint MCCFR+ trainer (port of `cfr/blueprint_trainer.py`, Phase 4).

pub mod db;
pub mod fixture;
pub mod parallel;
pub mod pyjson;
pub mod rng;
pub mod table;
pub mod trainer;

pub use trainer::{Abstraction, Schedule, Table, Trainer};
