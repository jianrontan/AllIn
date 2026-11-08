//! External-sampling MCCFR+ with Linear-CFR-style discounting. Port of
//! `BlueprintTrainer.cfr` / `_run_iteration` (the Lever-A threaded-state path).
//!
//! Bit-parity notes (vs Python, same deal + same uniform stream):
//! - regret matching sums with numpy's order (`np_sum`);
//! - `node_value` is Python's left-to-right `sum` starting from +0.0;
//! - the discount is `((t-1)/t).powf(alpha)` (libm `pow`, as CPython);
//! - opponent sampling replays `random.choices`: cumulative weights summed left
//!   to right, `bisect_right(cum, u * total, 0, n - 1)`.

use rustc_hash::{FxHashMap, FxHashSet};

use super::rng::Uniform;
use super::table::{InfoSet, MAX_ACTIONS, Pattern, action_id, floor0, id_action, pack_key};
use crate::engine::{self, Action, Engine, NodeState, STARTING_STACK};

/// Card abstraction the trainer reads (the bucketing component, or fixture
/// values in the parity test). Must be shareable across worker threads.
pub trait Abstraction: Sync {
    /// FINE preflop bucket (0..169).
    fn preflop_fine(&self, hole: [u8; 2]) -> u8;
    /// Coarse class of a fine bucket (postflop keys).
    fn coarse(&self, fine: u8) -> u8;
    /// Postflop strength bucket of `hole` on `board` (3, 4 or 5 cards).
    fn postflop(&self, hole: [u8; 2], board: &[u8]) -> Result<u16, String>;
    /// 5..7-card rank, lower = stronger.
    fn rank(&self, cards: &[u8]) -> i32;
}

pub type Table = FxHashMap<u64, InfoSet>;

const UNSET: i32 = i32::MIN;

/// Per-iteration card context with the trainer's lazy memo (`_postflop_memo`,
/// `_raw_cached`).
struct Hand<'a, A: Abstraction + ?Sized> {
    abs: &'a A,
    hole: [[u8; 2]; 2],
    board: [u8; 5],
    fine: [u8; 2],
    coarse: [u8; 2],
    post: [[i32; 3]; 2],
    raw5: [i32; 2],
}

impl<A: Abstraction + ?Sized> Hand<'_, A> {
    fn strength(&mut self, seat: usize, street: u8) -> Result<u16, String> {
        let i = street as usize - 1;
        if self.post[seat][i] == UNSET {
            let b = self
                .abs
                .postflop(self.hole[seat], &self.board[..street as usize + 2])?;
            self.post[seat][i] = b as i32;
        }
        Ok(self.post[seat][i] as u16)
    }

    fn rank(&mut self, cards: &[u8]) -> i32 {
        // Every showdown in the trainer uses the full board; memoize that case.
        if cards.len() == 7 && cards[2..] == self.board {
            let seat = if cards[..2] == self.hole[0] { 0 } else { 1 };
            if cards[..2] == self.hole[seat] {
                if self.raw5[seat] == UNSET {
                    self.raw5[seat] = self.abs.rank(cards);
                }
                return self.raw5[seat];
            }
        }
        self.abs.rank(cards)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule {
    pub alpha: f64,
    pub gamma: f64,
}

impl Default for Schedule {
    fn default() -> Self {
        Schedule {
            alpha: 1.5,
            gamma: 1.0,
        }
    }
}

pub struct Trainer {
    pub engine: Engine,
    pub menu_mode: String,
    pub sched: Schedule,
    pub table: Table,
    /// Keys touched since the last checkpoint.
    pub dirty: FxHashSet<u64>,
    /// Worker mode (discount off): keys with a regret / strategy update.
    pub dirty_regret: FxHashSet<u64>,
    pub dirty_strategy: FxHashSet<u64>,
    pub discount_enabled: bool,
    pub ev_sum: f64,
    pub ev_count: i64,
    /// Read-only fallback table (parallel workers read the round baseline here
    /// and copy an entry into `table` on first touch).
    base: Option<std::sync::Arc<Table>>,
    max_depth_warned: bool,
}

/// Arguments that change per recursion step.
#[derive(Clone, Copy)]
struct Node {
    street: u8,
    depth: u32,
    p_inv: [f64; 2],
    pattern: Pattern,
    stacks: [f64; 2],
    st: NodeState,
    /// Index in the history buffer where this street's actions start.
    h0: usize,
}

impl Trainer {
    pub fn new(menu_mode: &str, sched: Schedule) -> Result<Trainer, String> {
        Ok(Trainer {
            engine: Engine::for_mode(menu_mode)?,
            menu_mode: menu_mode.to_string(),
            sched,
            table: Table::default(),
            dirty: FxHashSet::default(),
            dirty_regret: FxHashSet::default(),
            dirty_strategy: FxHashSet::default(),
            discount_enabled: true,
            ev_sum: 0.0,
            ev_count: 0,
            base: None,
            max_depth_warned: false,
        })
    }

    /// Worker over a shared, read-only round baseline (discount disabled).
    pub fn worker(
        menu_mode: &str,
        sched: Schedule,
        base: std::sync::Arc<Table>,
    ) -> Result<Trainer, String> {
        let mut t = Trainer::new(menu_mode, sched)?;
        t.discount_enabled = false;
        t.base = Some(base);
        Ok(t)
    }

    #[inline]
    fn entry(&mut self, key: u64) -> &mut InfoSet {
        let base = &self.base;
        self.table.entry(key).or_insert_with(|| {
            base.as_ref()
                .and_then(|b| b.get(&key).copied())
                .unwrap_or_default()
        })
    }

    /// One MCCFR+ iteration for `deal` = [p0 hole x2, p1 hole x2, board x5].
    /// Returns the sampled root utility (P0 perspective).
    pub fn run_iteration<A: Abstraction + ?Sized, R: Uniform>(
        &mut self,
        iteration: i64,
        deal: [u8; 9],
        abs: &A,
        rng: &mut R,
    ) -> Result<f64, String> {
        let hole = [[deal[0], deal[1]], [deal[2], deal[3]]];
        let fine = [abs.preflop_fine(hole[0]), abs.preflop_fine(hole[1])];
        let mut hand = Hand {
            abs,
            hole,
            board: deal[4..9].try_into().unwrap(),
            fine,
            coarse: [abs.coarse(fine[0]), abs.coarse(fine[1])],
            post: [[UNSET; 3]; 2],
            raw5: [UNSET; 2],
        };
        let updating = (iteration.rem_euclid(2)) as u8;
        let mut hist: Vec<Action> = Vec::with_capacity(64);
        let root = Node {
            street: 0,
            depth: 0,
            p_inv: [0.0, 0.0],
            pattern: Pattern::default(),
            stacks: [STARTING_STACK - 1.0, STARTING_STACK - 2.0],
            st: self.engine.init_node_state(0, 3.0),
            h0: 0,
        };
        let util = self.cfr(&mut hand, &mut hist, root, updating, iteration, rng)?;
        self.ev_sum += util;
        self.ev_count += 1;
        Ok(util)
    }

    fn terminal<A: Abstraction + ?Sized>(
        &self,
        hand: &mut Hand<'_, A>,
        hist: &[Action],
        n: &Node,
        street: u8,
    ) -> f64 {
        let board = hand.board;
        let (p0, p1) = (hand.hole[0], hand.hole[1]);
        engine::utility(
            p0,
            p1,
            &board,
            &hist[n.h0..],
            street.min(3),
            n.st.pot,
            n.p_inv[0] + n.st.c[0],
            |c| hand.rank(c),
        )
    }

    fn cfr<A: Abstraction + ?Sized, R: Uniform>(
        &mut self,
        hand: &mut Hand<'_, A>,
        hist: &mut Vec<Action>,
        n: Node,
        upd: u8,
        iteration: i64,
        rng: &mut R,
    ) -> Result<f64, String> {
        if n.depth > 50 {
            if !self.max_depth_warned {
                eprintln!("WARNING: Max depth reached at street {}", n.street);
                self.max_depth_warned = true;
            }
            return Ok(self.terminal(hand, hist, &n, n.street));
        }
        if n.street > 3 {
            return Ok(self.terminal(hand, hist, &n, 3));
        }
        if engine::is_terminal(&hist[n.h0..], n.street) {
            return Ok(self.terminal(hand, hist, &n, n.street));
        }
        let cp = Engine::acting_player(hist.len() - n.h0, n.street);
        let stack_cp = n.stacks[cp as usize];
        let mut legal_buf: Vec<Action> = Vec::with_capacity(MAX_ACTIONS);
        self.engine
            .legal_actions_into(n.street, &n.st, cp, stack_cp, &mut legal_buf);

        if legal_buf.is_empty() {
            if n.street < 3 {
                let p_inv = [n.p_inv[0] + n.st.c[0], n.p_inv[1] + n.st.c[1]];
                let next = Node {
                    street: n.street + 1,
                    depth: n.depth + 1,
                    p_inv,
                    pattern: Pattern::default(),
                    stacks: [STARTING_STACK - p_inv[0], STARTING_STACK - p_inv[1]],
                    st: self.engine.init_node_state(n.street + 1, n.st.pot),
                    h0: hist.len(),
                };
                return self.cfr(hand, hist, next, upd, iteration, rng);
            }
            return Ok(self.terminal(hand, hist, &n, n.street));
        }

        let na = legal_buf.len();
        let mut ids = [0u8; MAX_ACTIONS];
        for (i, &a) in legal_buf.iter().enumerate() {
            ids[i] = action_id(a)?;
        }
        let legal = &ids[..na];

        let seat = cp as usize;
        let (bucket, strength) = if n.street == 0 {
            (hand.fine[seat], 0)
        } else {
            (hand.coarse[seat], hand.strength(seat, n.street)?)
        };
        let key = pack_key(n.street, cp, bucket, strength, n.pattern)?;
        self.dirty.insert(key);

        let mut strat = [0.0f64; MAX_ACTIONS];
        let discount = self.discount_enabled;
        let sched = self.sched;

        let child = |this: &Trainer, a: Action| -> Result<Node, String> {
            let cost = this.engine.action_cost(a, n.street, &n.st, cp, stack_cp)?;
            let st = this
                .engine
                .advance(&n.st, a, n.street, cp, stack_cp, n.p_inv)?;
            let mut stacks = n.stacks;
            stacks[seat] -= cost;
            Ok(Node {
                depth: n.depth + 1,
                pattern: n.pattern.push(crate::keys::action_char(a)?)?,
                stacks,
                st,
                ..n
            })
        };

        if cp == upd {
            if !discount {
                self.dirty_regret.insert(key);
            }
            {
                let is = self.entry(key);
                // Python computes the strategy BEFORE the discount (scaling then
                // normalising is not bit-identical to normalising).
                is.current_strategy(legal, &mut strat)?;
                if discount && is.last_visited != iteration {
                    is.visit_count += 1;
                    is.last_visited = iteration;
                    let t = is.visit_count;
                    if t > 1 {
                        let decay = ((t - 1) as f64 / t as f64).powf(sched.alpha);
                        is.regrets.scale(decay);
                    }
                }
            }
            let mut vals = [0.0f64; MAX_ACTIONS];
            for i in 0..na {
                let a = id_action(legal[i]);
                let c = child(self, a)?;
                hist.push(a);
                let v = self.cfr(hand, hist, c, upd, iteration, rng);
                hist.pop();
                vals[i] = v?;
            }
            let sign = if cp == 0 { 1.0 } else { -1.0 };
            let mut own = [0.0f64; MAX_ACTIONS];
            let mut node_value = 0.0f64;
            for i in 0..na {
                own[i] = sign * vals[i];
                node_value += strat[i] * own[i];
            }
            let is = self.table.get_mut(&key).expect("info set present");
            for i in 0..na {
                let regret = own[i] - node_value;
                let prior = is.regrets.get(legal[i]).unwrap_or(0.0);
                is.regrets.set(legal[i], floor0(prior + regret))?;
            }
            Ok(sign * node_value)
        } else {
            if !discount {
                self.dirty_strategy.insert(key);
            }
            {
                let is = self.entry(key);
                is.current_strategy(legal, &mut strat)?;
                if discount && is.last_strategy != iteration {
                    is.strategy_visit_count += 1;
                    is.last_strategy = iteration;
                    let s = is.strategy_visit_count;
                    if s > 1 && sched.gamma != 0.0 {
                        let decay = ((s - 1) as f64 / s as f64).powf(sched.gamma);
                        is.strategy.scale(decay);
                    }
                }
                is.accumulate(legal, &strat[..na])?;
            }
            let pick = sample(&strat[..na], rng)?;
            let a = id_action(legal[pick]);
            let c = child(self, a)?;
            hist.push(a);
            let v = self.cfr(hand, hist, c, upd, iteration, rng);
            hist.pop();
            v
        }
    }
}

/// `random.choices(range(n), weights=w)[0]` with `u = rng.uniform()`.
#[inline]
pub fn sample<R: Uniform>(w: &[f64], rng: &mut R) -> Result<usize, String> {
    let n = w.len();
    let mut cum = [0.0f64; MAX_ACTIONS];
    let mut acc = w[0];
    cum[0] = acc;
    for i in 1..n {
        acc += w[i];
        cum[i] = acc;
    }
    let total = cum[n - 1] + 0.0;
    if total <= 0.0 || !total.is_finite() {
        return Err("total of weights must be finite and greater than zero".into());
    }
    let x = rng.uniform()? * total;
    // bisect_right over cum[0..n-1]
    let (mut lo, mut hi) = (0usize, n - 1);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x < cum[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(lo)
}

impl Trainer {
    /// Metadata written with every checkpoint (`checkpoint_to_db`).
    pub fn checkpoint_meta(&self, iteration: i64) -> Vec<(&'static str, String)> {
        use super::pyjson::py_repr;
        vec![
            ("total_iterations", (iteration + 1).to_string()),
            ("alpha", py_repr(self.sched.alpha)),
            ("gamma", py_repr(self.sched.gamma)),
            ("menu_mode", format!("\"{}\"", self.menu_mode)),
            ("ev_sum", py_repr(self.ev_sum)),
            ("ev_count", self.ev_count.to_string()),
        ]
    }

    /// Atomic incremental checkpoint of the dirty info sets + metadata.
    pub fn checkpoint(
        &mut self,
        db: &mut super::db::BlueprintDb,
        iteration: i64,
    ) -> Result<usize, String> {
        let meta = self.checkpoint_meta(iteration);
        let n = db.save_checkpoint(&self.table, self.dirty.iter(), &meta)?;
        self.dirty.clear();
        Ok(n)
    }

    /// Load a blueprint to continue training (`resume_from_db`), with the same
    /// guards: alpha/gamma and menu-mode must match, and a known training mode
    /// (`single` / `parallel`) must match `mode`. Returns the start iteration.
    pub fn resume(
        &mut self,
        db: &super::db::BlueprintDb,
        mode: Option<&str>,
    ) -> Result<i64, String> {
        for (name, ours) in [("alpha", self.sched.alpha), ("gamma", self.sched.gamma)] {
            if let Some(v) = db.metadata(name)? {
                let stored = v
                    .as_f64()
                    .ok_or_else(|| format!("metadata {name} not numeric"))?;
                if (stored - ours).abs() > 1e-9 {
                    return Err(format!(
                        "Resume {name} mismatch: blueprint trained with {name}={stored}, run configured with {ours}"
                    ));
                }
            }
        }
        let stored_menu = db
            .metadata("menu_mode")?
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "control".into());
        if stored_menu != self.menu_mode {
            return Err(format!(
                "Resume menu-mode mismatch: blueprint menu_mode={stored_menu:?}, run {:?}",
                self.menu_mode
            ));
        }
        if let (Some(mode), Some(stored)) = (mode, db.metadata("training_mode")?) {
            if let Some(s) = stored.as_str() {
                if s != mode {
                    return Err(format!(
                        "Resume mode mismatch: blueprint trained in {s:?} mode, run is {mode:?}"
                    ));
                }
            }
        }
        self.table = db.load_all()?;
        self.dirty.clear();
        self.ev_sum = db
            .metadata("ev_sum")?
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        self.ev_count = db
            .metadata("ev_count")?
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let start = db
            .metadata("total_iterations")?
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        // H2 monotonicity guard: stored clocks must not be ahead of the counter,
        // or the once-per-iteration discount mis-fires.
        let max_clock = self
            .table
            .values()
            .map(|is| is.last_visited.max(is.last_strategy))
            .max()
            .unwrap_or(-1);
        if self.discount_enabled && start < max_clock {
            return Err(format!(
                "resume start_iteration={start} is behind the loaded discount clocks (max={max_clock})"
            ));
        }
        Ok(start)
    }
}
