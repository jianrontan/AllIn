//! Two-sided vectorized CFR+ over the river tree. Port of `subgame/river_cfr.py`.
//!
//! Every decision node carries regret and strategy-sum tables over (hand,
//! action). One traversal carries both players' reach vectors and returns a
//! value vector for each player (player 0's value depends only on player 1's
//! reach and vice versa, so the acting player's child values are counterfactual
//! values). CFR+: regrets floored at 0; strategy sum weighted by the iteration
//! number, whose clock persists across `run` / `run_gadget` calls.
//!
//! Internally the tables are ACTION-MAJOR (`[A][H]`, so every inner loop is a
//! contiguous, vectorizable pass over hands); the public accessors take and
//! return Python's `[H, A]` row-major layout. Element-wise arithmetic follows
//! the Python expression order and per-row sums run over actions left to right
//! (numpy's order for < 8 actions), so in exact mode (`Board::exact`) the
//! results are bit-identical to `river_cfr.py`.

use super::showdown::{Board, np_dot};
use super::tree::{Decision, Node, RiverTree, Terminal};

/// Regret-matching / averaging threshold used by the Python code.
const EPS: f64 = 1e-12;

pub struct RiverCfr {
    pub tree: RiverTree,
    pub board: Board,
    tables: Tables,
}

struct Tables {
    h: usize,
    /// Per decision node, `[A][H]`.
    regret: Vec<Vec<f64>>,
    strat_sum: Vec<Vec<f64>>,
    iter: u64,
    t_weight: f64,
    dcfr: bool,
    updating: Option<u8>,
    /// Gadget regret `[H, 2]` (Follow, Terminate), persisted across calls.
    g_regret: Option<Vec<f64>>,
}

/// Normalise an action-major table per hand: `x / sum_a x`, or uniform `1/A`
/// where that sum is <= EPS (Python `_strategy` / `average_strategy`).
fn normalise(x: &[f64], h: usize, a: usize, out: &mut [f64]) {
    let mut s = vec![0.0f64; h];
    for k in 0..a {
        let col = &x[k * h..(k + 1) * h];
        for i in 0..h {
            s[i] += col[i];
        }
    }
    let uni = 1.0 / a as f64;
    for k in 0..a {
        let col = &x[k * h..(k + 1) * h];
        let o = &mut out[k * h..(k + 1) * h];
        for i in 0..h {
            o[i] = if s[i] > EPS { col[i] / s[i] } else { uni };
        }
    }
}

/// `[H, A]` row-major -> `[A][H]`.
fn to_action_major(x: &[f64], h: usize, a: usize) -> Vec<f64> {
    let mut out = vec![0.0; h * a];
    for i in 0..h {
        for k in 0..a {
            out[k * h + i] = x[i * a + k];
        }
    }
    out
}

/// `[A][H]` -> `[H, A]` row-major.
fn to_row_major(x: &[f64], h: usize, a: usize) -> Vec<f64> {
    let mut out = vec![0.0; h * a];
    for k in 0..a {
        for i in 0..h {
            out[i * a + k] = x[k * h + i];
        }
    }
    out
}

/// Both players' values at a terminal (`RiverCFR._terminal`).
fn terminal_values(
    board: &Board,
    t: &Terminal,
    r0: &[f64],
    r1: &[f64],
    v0: &mut [f64],
    v1: &mut [f64],
) {
    let fp = t.final_pot;
    let [c0, c1] = t.contrib;
    match t.folder {
        None => {
            board.showdown_measure(r1, fp, c0, v0);
            board.showdown_measure(r0, fp, c1, v1);
        }
        Some(f) => {
            let (p0, p1) = if f == 0 {
                (-c0, fp - c1)
            } else {
                (fp - c0, -c1)
            };
            board.compatible_mass(r1, v0);
            board.compatible_mass(r0, v1);
            v0.iter_mut().for_each(|x| *x *= p0);
            v1.iter_mut().for_each(|x| *x *= p1);
        }
    }
}

/// Seat `seat`'s value at a terminal vs the opponent reach `rv`.
fn terminal_one(board: &Board, t: &Terminal, seat: usize, rv: &[f64], out: &mut [f64]) {
    let ch = t.contrib[seat];
    match t.folder {
        None => board.showdown_measure(rv, t.final_pot, ch, out),
        Some(f) => {
            let payoff = if f as usize == seat {
                -ch
            } else {
                t.final_pot - ch
            };
            board.compatible_mass(rv, out);
            out.iter_mut().for_each(|x| *x *= payoff);
        }
    }
}

/// Seat `seat`'s value at a terminal from precomputed fast-mode showdown stats
/// of the opponent reach (`eq`, `compat`) -- equal to `terminal_one` in fast mode.
fn terminal_from_stats(t: &Terminal, seat: usize, eq: &[f64], compat: &[f64], out: &mut [f64]) {
    let ch = t.contrib[seat];
    match t.folder {
        None => {
            let fp = t.final_pot;
            for i in 0..out.len() {
                out[i] = fp * eq[i] - ch * compat[i];
            }
        }
        Some(f) => {
            let payoff = if f as usize == seat {
                -ch
            } else {
                t.final_pot - ch
            };
            for i in 0..out.len() {
                out[i] = compat[i] * payoff;
            }
        }
    }
}

/// Lazily computed fast-mode stats of the opponent reach at one node: every
/// child sees the same opponent reach, so the acting player's terminal values
/// for all terminal children share one kernel call.
struct OppStats {
    eq: Vec<f64>,
    compat: Vec<f64>,
    ready: bool,
}

impl OppStats {
    fn new() -> Self {
        OppStats {
            eq: Vec::new(),
            compat: Vec::new(),
            ready: false,
        }
    }

    fn get(&mut self, board: &Board, r_opp: &[f64]) -> (&[f64], &[f64]) {
        if !self.ready {
            self.eq = vec![0.0; board.h];
            self.compat = vec![0.0; board.h];
            board.showdown_stats(r_opp, &mut self.eq, &mut self.compat);
            self.ready = true;
        }
        (&self.eq, &self.compat)
    }
}

/// Values of one child for (actor, other), writing into `v_actor` / `v_other`.
/// `child_reach` is the actor's reach times its strategy for this action.
/// `walk` evaluates a non-terminal child as `walk(child, r0, r1, v0, v1)`.
#[allow(clippy::too_many_arguments)]
fn child_values(
    tree: &RiverTree,
    board: &Board,
    d: &Decision,
    child: usize,
    r_opp: &[f64],
    child_reach: &[f64],
    stats: &mut OppStats,
    v_actor: &mut [f64],
    v_other: &mut [f64],
    walk: &mut dyn FnMut(usize, &[f64], &[f64], &mut [f64], &mut [f64]),
) {
    let p = d.player as usize;
    if let (Node::Terminal(t), false) = (&tree.nodes[child], board.exact) {
        let (eq, cm) = stats.get(board, r_opp);
        terminal_from_stats(t, p, eq, cm, v_actor);
        terminal_one(board, t, 1 - p, child_reach, v_other);
        return;
    }
    if p == 0 {
        walk(child, child_reach, r_opp, v_actor, v_other);
    } else {
        walk(child, r_opp, child_reach, v_other, v_actor);
    }
}

impl Tables {
    fn current_strategy(&self, nid: usize, a: usize, out: &mut [f64]) {
        if self.dcfr {
            let positive: Vec<f64> = self.regret[nid].iter().map(|r| r.max(0.0)).collect();
            normalise(&positive, self.h, a, out);
        } else {
            normalise(&self.regret[nid], self.h, a, out);
        }
    }

    fn average_strategy(&self, nid: usize, a: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.h * a];
        normalise(&self.strat_sum[nid], self.h, a, &mut out);
        out
    }
}

/// One CFR+ traversal (`RiverCFR._cfr`), writing the root values into v0 / v1.
#[allow(clippy::too_many_arguments)]
fn cfr<const DCFR: bool>(
    tree: &RiverTree,
    board: &Board,
    tb: &mut Tables,
    ni: usize,
    r0: &[f64],
    r1: &[f64],
    v0: &mut [f64],
    v1: &mut [f64],
) {
    let d = match &tree.nodes[ni] {
        Node::Terminal(t) => return terminal_values(board, t, r0, r1, v0, v1),
        Node::Decision(d) => d,
    };
    let h = tb.h;
    let a = d.actions.len();
    let nid = d.node_id;
    let p = d.player as usize;
    let mut strat = vec![0.0; h * a];
    if DCFR {
        let positive: Vec<f64> = tb.regret[nid].iter().map(|r| r.max(0.0)).collect();
        normalise(&positive, h, a, &mut strat);
    } else {
        normalise(&tb.regret[nid], h, a, &mut strat);
    }
    let (r_p, r_opp) = if p == 0 { (r0, r1) } else { (r1, r0) };

    let mut u = vec![0.0; h * a];
    let mut v_other = vec![0.0; h];
    let mut tmp = vec![0.0; h];
    let mut child_reach = vec![0.0; h];
    let mut stats = OppStats::new();
    for (k, &child) in d.children.iter().enumerate() {
        let s = &strat[k * h..(k + 1) * h];
        for i in 0..h {
            child_reach[i] = r_p[i] * s[i];
        }
        let uk = &mut u[k * h..(k + 1) * h];
        child_values(
            tree,
            board,
            d,
            child,
            r_opp,
            &child_reach,
            &mut stats,
            uk,
            &mut tmp,
            &mut |c, a0, a1, o0, o1| cfr::<DCFR>(tree, board, tb, c, a0, a1, o0, o1),
        );
        for i in 0..h {
            v_other[i] += tmp[i];
        }
    }
    let mut v_p = vec![0.0; h];
    for k in 0..a {
        let (s, uk) = (&strat[k * h..(k + 1) * h], &u[k * h..(k + 1) * h]);
        for i in 0..h {
            v_p[i] += s[i] * uk[i];
        }
    }
    if !DCFR || tb.updating.is_none_or(|seat| seat as usize == p) {
        let tw = tb.t_weight;
        let reg = &mut tb.regret[nid];
        let ss = &mut tb.strat_sum[nid];
        for k in 0..a {
            let o = k * h;
            for i in 0..h {
                let j = o + i;
                let r = reg[j] + (u[j] - v_p[i]);
                reg[j] = if DCFR || r > 0.0 { r } else { 0.0 };
                ss[j] += (r_p[i] * strat[j]) * tw;
            }
        }
    }
    let (out_p, out_o) = if p == 0 { (v0, v1) } else { (v1, v0) };
    out_p.copy_from_slice(&v_p);
    out_o.copy_from_slice(&v_other);
}

/// Read-only value propagation under fixed action-major strategies (`_eval`).
#[allow(clippy::too_many_arguments)]
fn eval(
    tree: &RiverTree,
    board: &Board,
    strats: &[Vec<f64>],
    ni: usize,
    r0: &[f64],
    r1: &[f64],
    v0: &mut [f64],
    v1: &mut [f64],
) {
    let d = match &tree.nodes[ni] {
        Node::Terminal(t) => return terminal_values(board, t, r0, r1, v0, v1),
        Node::Decision(d) => d,
    };
    let h = board.h;
    let a = d.actions.len();
    let p = d.player as usize;
    let strat = &strats[d.node_id];
    let (r_p, r_opp) = if p == 0 { (r0, r1) } else { (r1, r0) };
    let mut u = vec![0.0; h * a];
    let mut v_other = vec![0.0; h];
    let mut tmp = vec![0.0; h];
    let mut child_reach = vec![0.0; h];
    let mut stats = OppStats::new();
    for (k, &child) in d.children.iter().enumerate() {
        let s = &strat[k * h..(k + 1) * h];
        for i in 0..h {
            child_reach[i] = r_p[i] * s[i];
        }
        let uk = &mut u[k * h..(k + 1) * h];
        child_values(
            tree,
            board,
            d,
            child,
            r_opp,
            &child_reach,
            &mut stats,
            uk,
            &mut tmp,
            &mut |c, a0, a1, o0, o1| eval(tree, board, strats, c, a0, a1, o0, o1),
        );
        for i in 0..h {
            v_other[i] += tmp[i];
        }
    }
    let mut v_p = vec![0.0; h];
    for k in 0..a {
        let (s, uk) = (&strat[k * h..(k + 1) * h], &u[k * h..(k + 1) * h]);
        for i in 0..h {
            v_p[i] += s[i] * uk[i];
        }
    }
    let (out_p, out_o) = if p == 0 { (v0, v1) } else { (v1, v0) };
    out_p.copy_from_slice(&v_p);
    out_o.copy_from_slice(&v_other);
}

/// `br` best-responds per hand; the other seat plays `strats` (`_br_value`).
/// `reach_fixer` = the fixed seat's reach into this node.
fn br_value(
    tree: &RiverTree,
    board: &Board,
    strats: &[Vec<f64>],
    ni: usize,
    br: usize,
    reach_fixer: &[f64],
    out: &mut [f64],
) {
    let d = match &tree.nodes[ni] {
        Node::Terminal(t) => return terminal_one(board, t, br, reach_fixer, out),
        Node::Decision(d) => d,
    };
    let h = board.h;
    let mut tmp = vec![0.0; h];
    if d.player as usize == br {
        // The fixed seat's reach is the same for every child: share the stats.
        let mut stats = OppStats::new();
        for (k, &c) in d.children.iter().enumerate() {
            let dst: &mut [f64] = if k == 0 { out } else { &mut tmp };
            match (&tree.nodes[c], board.exact) {
                (Node::Terminal(t), false) => {
                    let (eq, cm) = stats.get(board, reach_fixer);
                    terminal_from_stats(t, br, eq, cm, dst);
                }
                _ => br_value(tree, board, strats, c, br, reach_fixer, dst),
            }
            if k > 0 {
                for i in 0..h {
                    // np.maximum: NaN propagates; otherwise the larger value.
                    if tmp[i] > out[i] || tmp[i].is_nan() {
                        out[i] = tmp[i];
                    }
                }
            }
        }
        return;
    }
    let strat = &strats[d.node_id];
    let mut child_reach = vec![0.0; h];
    out.fill(0.0);
    for (k, &c) in d.children.iter().enumerate() {
        let s = &strat[k * h..(k + 1) * h];
        for i in 0..h {
            child_reach[i] = reach_fixer[i] * s[i];
        }
        br_value(tree, board, strats, c, br, &child_reach, &mut tmp);
        for i in 0..h {
            out[i] += tmp[i];
        }
    }
}

impl RiverCfr {
    pub fn new(tree: RiverTree, board: Board) -> Self {
        let h = board.h;
        let n = tree.num_decision();
        let sizes: Vec<usize> = (0..n)
            .map(|i| h * tree.decision_node(i).actions.len())
            .collect();
        let tables = Tables {
            h,
            regret: sizes.iter().map(|&s| vec![0.0; s]).collect(),
            strat_sum: sizes.iter().map(|&s| vec![0.0; s]).collect(),
            iter: 0,
            t_weight: 1.0,
            dcfr: false,
            updating: None,
            g_regret: None,
        };
        RiverCfr {
            tree,
            board,
            tables,
        }
    }

    pub fn h(&self) -> usize {
        self.board.h
    }

    pub fn iterations(&self) -> u64 {
        self.tables.iter
    }

    pub fn num_actions(&self, nid: usize) -> usize {
        self.tree.decision_node(nid).actions.len()
    }

    fn check_len(&self, what: &str, v: &[f64]) -> Result<(), String> {
        if v.len() != self.board.h {
            return Err(format!("{what}: length {} != H={}", v.len(), self.board.h));
        }
        Ok(())
    }

    fn root(&mut self, r0: &[f64], r1: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let h = self.board.h;
        let (mut v0, mut v1) = (vec![0.0; h], vec![0.0; h]);
        if self.tables.dcfr {
            cfr::<true>(
                &self.tree,
                &self.board,
                &mut self.tables,
                0,
                r0,
                r1,
                &mut v0,
                &mut v1,
            );
        } else {
            cfr::<false>(
                &self.tree,
                &self.board,
                &mut self.tables,
                0,
                r0,
                r1,
                &mut v0,
                &mut v1,
            );
        }
        (v0, v1)
    }

    /// Experimental alternating DCFR(1.5,0,2); enable only before solving.
    pub fn enable_dcfr(&mut self) -> Result<(), String> {
        if self.tables.iter != 0 {
            return Err("cannot change algorithm after solving".into());
        }
        self.tables.dcfr = true;
        Ok(())
    }

    fn discount_player(&mut self, seat: u8) {
        let t = (self.tables.iter as f64).powf(1.5);
        for (nid, r) in self.tables.regret.iter_mut().enumerate() {
            if self.tree.decision_node(nid).player == seat {
                for x in r {
                    *x *= if *x >= 0.0 { t / (t + 1.0) } else { 0.5 };
                }
            }
        }
    }

    /// `run(reach0, reach1, iters)`.
    pub fn run(&mut self, r0: &[f64], r1: &[f64], iters: u64) -> Result<(), String> {
        self.check_len("reach0", r0)?;
        self.check_len("reach1", r1)?;
        for _ in 0..iters {
            self.tables.iter += 1;
            self.tables.t_weight = self.tables.iter as f64;
            if self.tables.dcfr {
                self.tables.t_weight *= self.tables.t_weight;
                for seat in 0..2 {
                    self.tables.updating = Some(seat);
                    self.root(r0, r1);
                    self.discount_player(seat);
                }
                self.tables.updating = None;
            } else {
                self.root(r0, r1);
            }
        }
        Ok(())
    }

    /// `run_gadget(hero_reach, villain_gadget_reach, optout, villain_seat, iters)`.
    pub fn run_gadget(
        &mut self,
        hero: &[f64],
        vg: &[f64],
        optout: &[f64],
        villain_seat: u8,
        iters: u64,
    ) -> Result<(), String> {
        self.check_len("hero_reach", hero)?;
        self.check_len("villain_gadget_reach", vg)?;
        self.check_len("optout", optout)?;
        if villain_seat > 1 {
            return Err(format!("villain_seat must be 0 or 1, got {villain_seat}"));
        }
        if self.tables.dcfr {
            return self.run_dcfr_gadget(hero, vg, optout, villain_seat, iters);
        }
        let h = self.board.h;
        let mut g = self
            .tables
            .g_regret
            .take()
            .unwrap_or_else(|| vec![0.0; 2 * h]);
        let mut follow = vec![0.0; h];
        let mut term = vec![0.0; h];
        let mut villain = vec![0.0; h];
        for _ in 0..iters {
            self.tables.iter += 1;
            self.tables.t_weight = self.tables.iter as f64;
            for i in 0..h {
                let (f, t) = (g[2 * i].max(0.0), g[2 * i + 1].max(0.0));
                let s = f + t;
                if s > EPS {
                    follow[i] = f / s;
                    term[i] = t / s;
                } else {
                    follow[i] = 0.5;
                    term[i] = 0.5;
                }
                villain[i] = vg[i] * follow[i];
            }
            let v_follow = if villain_seat == 0 {
                self.root(&villain, hero).0
            } else {
                self.root(hero, &villain).1
            };
            for i in 0..h {
                let node_val = follow[i] * v_follow[i] + term[i] * optout[i];
                let f = g[2 * i] + (v_follow[i] - node_val);
                let t = g[2 * i + 1] + (optout[i] - node_val);
                g[2 * i] = if f > 0.0 { f } else { 0.0 };
                g[2 * i + 1] = if t > 0.0 { t } else { 0.0 };
            }
        }
        self.tables.g_regret = Some(g);
        Ok(())
    }

    fn run_dcfr_gadget(
        &mut self,
        hero: &[f64],
        vg: &[f64],
        optout: &[f64],
        villain_seat: u8,
        iters: u64,
    ) -> Result<(), String> {
        let h = self.board.h;
        let mut g = self
            .tables
            .g_regret
            .take()
            .unwrap_or_else(|| vec![0.0; 2 * h]);
        for _ in 0..iters {
            self.tables.iter += 1;
            self.tables.t_weight = (self.tables.iter as f64).powi(2);
            let t = (self.tables.iter as f64).powf(1.5);
            for seat in 0..2 {
                let mut follow = vec![0.0; h];
                let mut term = vec![0.0; h];
                let mut villain = vec![0.0; h];
                for i in 0..h {
                    let f = g[2 * i].max(0.0);
                    let q = g[2 * i + 1].max(0.0);
                    let sum = f + q;
                    follow[i] = if sum > EPS { f / sum } else { 0.5 };
                    term[i] = if sum > EPS { q / sum } else { 0.5 };
                    villain[i] = vg[i] * follow[i];
                }
                self.tables.updating = Some(seat);
                let values = if villain_seat == 0 {
                    self.root(&villain, hero).0
                } else {
                    self.root(hero, &villain).1
                };
                self.discount_player(seat);
                if seat == villain_seat {
                    for i in 0..h {
                        let value = follow[i] * values[i] + term[i] * optout[i];
                        g[2 * i] += values[i] - value;
                        g[2 * i + 1] += optout[i] - value;
                    }
                    for r in &mut g {
                        *r *= if *r >= 0.0 { t / (t + 1.0) } else { 0.5 };
                    }
                }
            }
        }
        self.tables.updating = None;
        self.tables.g_regret = Some(g);
        Ok(())
    }

    /// Cumulative regret at a node, `[H, A]` row-major.
    pub fn regret(&self, nid: usize) -> Vec<f64> {
        to_row_major(
            &self.tables.regret[nid],
            self.board.h,
            self.num_actions(nid),
        )
    }

    /// Strategy sum at a node, `[H, A]` row-major.
    pub fn strat_sum(&self, nid: usize) -> Vec<f64> {
        to_row_major(
            &self.tables.strat_sum[nid],
            self.board.h,
            self.num_actions(nid),
        )
    }

    /// Gadget regret `[H, 2]` row-major (None before the first `run_gadget`).
    pub fn gadget_regret(&self) -> Option<&[f64]> {
        self.tables.g_regret.as_deref()
    }

    /// Average strategy at a node, `[H, A]` row-major.
    pub fn average_strategy(&self, nid: usize) -> Vec<f64> {
        let a = self.num_actions(nid);
        to_row_major(&self.tables.average_strategy(nid, a), self.board.h, a)
    }

    fn average_am(&self) -> Vec<Vec<f64>> {
        (0..self.tree.num_decision())
            .map(|n| self.tables.average_strategy(n, self.num_actions(n)))
            .collect()
    }

    fn current_am(&self) -> Vec<Vec<f64>> {
        (0..self.tree.num_decision())
            .map(|n| {
                let a = self.num_actions(n);
                let mut s = vec![0.0; self.board.h * a];
                self.tables.current_strategy(n, a, &mut s);
                s
            })
            .collect()
    }

    /// Validate `[H, A]` row-major strategies per node and convert to action-major.
    fn strats_am(&self, strats: &[Vec<f64>], what: &str) -> Result<Vec<Vec<f64>>, String> {
        if strats.len() != self.tree.num_decision() {
            return Err(format!(
                "{what}: {} arrays for {} nodes",
                strats.len(),
                self.tree.num_decision()
            ));
        }
        let h = self.board.h;
        strats
            .iter()
            .enumerate()
            .map(|(n, s)| {
                let a = self.num_actions(n);
                if s.len() != h * a {
                    return Err(format!(
                        "{what}: node {n} has {} entries, expected {}x{}",
                        s.len(),
                        h,
                        a
                    ));
                }
                Ok(to_action_major(s, h, a))
            })
            .collect()
    }

    /// `warm_start(seed_strategies, weight)`: add `weight * seed` (`[H, A]` per
    /// node) to each node's strategy sum.
    pub fn warm_start(&mut self, seeds: &[Vec<f64>], weight: f64) -> Result<(), String> {
        if weight <= 0.0 {
            return Ok(());
        }
        let seeds = self.strats_am(seeds, "seed strategies")?;
        for (ss, s) in self.tables.strat_sum.iter_mut().zip(&seeds) {
            for (x, &y) in ss.iter_mut().zip(s) {
                *x += weight * y;
            }
        }
        Ok(())
    }

    fn eval_am(
        &self,
        ni: usize,
        r0: &[f64],
        r1: &[f64],
        strats: &[Vec<f64>],
    ) -> (Vec<f64>, Vec<f64>) {
        let h = self.board.h;
        let (mut v0, mut v1) = (vec![0.0; h], vec![0.0; h]);
        eval(
            &self.tree,
            &self.board,
            strats,
            ni,
            r0,
            r1,
            &mut v0,
            &mut v1,
        );
        (v0, v1)
    }

    fn check_node(&self, ni: usize) -> Result<(), String> {
        if ni >= self.tree.nodes.len() {
            return Err(format!("node index {ni} out of range"));
        }
        Ok(())
    }

    /// `_eval` from node index `ni` (into `tree.nodes`); `strats` = `[H, A]`
    /// per node, or None for the average strategy.
    pub fn eval_with(
        &self,
        ni: usize,
        r0: &[f64],
        r1: &[f64],
        strats: Option<&[Vec<f64>]>,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        self.check_node(ni)?;
        self.check_len("reach0", r0)?;
        self.check_len("reach1", r1)?;
        let sf = match strats {
            Some(s) => self.strats_am(s, "strategies")?,
            None => self.average_am(),
        };
        Ok(self.eval_am(ni, r0, r1, &sf))
    }

    /// `current_values(reach0, reach1)`.
    pub fn current_values(&self, r0: &[f64], r1: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        self.check_len("reach0", r0)?;
        self.check_len("reach1", r1)?;
        Ok(self.eval_am(0, r0, r1, &self.current_am()))
    }

    /// `_br_value(node, br_player, reach_fixer, strat_fn)` from node index `ni`;
    /// `strats` = `[H, A]` per node, or None for the average strategy.
    pub fn br_value_with(
        &self,
        ni: usize,
        br: u8,
        reach_fixer: &[f64],
        strats: Option<&[Vec<f64>]>,
    ) -> Result<Vec<f64>, String> {
        self.check_node(ni)?;
        self.check_len("reach_fixer", reach_fixer)?;
        if br > 1 {
            return Err(format!("br_player must be 0 or 1, got {br}"));
        }
        let sf = match strats {
            Some(s) => self.strats_am(s, "strategies")?,
            None => self.average_am(),
        };
        let mut out = vec![0.0; self.board.h];
        br_value(
            &self.tree,
            &self.board,
            &sf,
            ni,
            br as usize,
            reach_fixer,
            &mut out,
        );
        Ok(out)
    }

    /// `exploitability(reach0, reach1, strat_fn)`; `strats` = `[H, A]` per node,
    /// or None for the average strategy (the convergence gap).
    pub fn exploitability(
        &self,
        r0: &[f64],
        r1: &[f64],
        strats: Option<&[Vec<f64>]>,
    ) -> Result<f64, String> {
        self.check_len("reach0", r0)?;
        self.check_len("reach1", r1)?;
        let sf = match strats {
            Some(s) => self.strats_am(s, "strategies")?,
            None => self.average_am(),
        };
        let h = self.board.h;
        let (mut br0, mut br1) = (vec![0.0; h], vec![0.0; h]);
        br_value(&self.tree, &self.board, &sf, 0, 0, r1, &mut br0);
        br_value(&self.tree, &self.board, &sf, 0, 1, r0, &mut br1);
        let mut cm = vec![0.0; h];
        self.board.compatible_mass(r1, &mut cm);
        let z = np_dot(r0, &cm);
        if z <= 0.0 {
            return Ok(0.0);
        }
        Ok((np_dot(r0, &br0) + np_dot(r1, &br1)) / z)
    }
}

#[cfg(test)]
mod tests {
    use super::super::showdown::tests::{lcg, random_board};
    use super::super::tree::{DEFAULT_MENU, MAX_AGGRESSIONS};
    use super::*;

    fn reaches(h: usize) -> (Vec<f64>, Vec<f64>) {
        let mut s = 5u64;
        (
            (0..h).map(|_| lcg(&mut s)).collect(),
            (0..h).map(|_| lcg(&mut s)).collect(),
        )
    }

    #[test]
    fn incremental_runs_match_one_run_and_converge() {
        let (board, ..) = random_board(3);
        let (r0, r1) = reaches(board.h);
        let tree = RiverTree::build(80.0, [160.0, 160.0], &DEFAULT_MENU, MAX_AGGRESSIONS).unwrap();
        let mut a = RiverCfr::new(tree.clone(), board.clone());
        let mut b = RiverCfr::new(tree, board);
        a.run(&r0, &r1, 30).unwrap();
        b.run(&r0, &r1, 10).unwrap();
        b.run(&r0, &r1, 20).unwrap();
        for n in 0..a.tree.num_decision() {
            assert_eq!(a.regret(n), b.regret(n));
            assert_eq!(a.strat_sum(n), b.strat_sum(n));
        }
        let e30 = a.exploitability(&r0, &r1, None).unwrap();
        a.run(&r0, &r1, 270).unwrap();
        let e300 = a.exploitability(&r0, &r1, None).unwrap();
        assert!(
            e300 < e30 && e300 >= -1e-9,
            "exploitability {e30} -> {e300}"
        );
    }

    /// The shared-stats shortcut (fast mode) must give the same values as the
    /// plain per-terminal path; compare against an explicitly passed strategy
    /// table evaluated both ways.
    #[test]
    fn fast_and_exact_modes_agree() {
        let (mut board, ..) = random_board(9);
        let (r0, r1) = reaches(board.h);
        let tree = RiverTree::build(30.0, [185.0, 185.0], &DEFAULT_MENU, MAX_AGGRESSIONS).unwrap();
        let mut fast = RiverCfr::new(tree.clone(), board.clone());
        board.exact = true;
        let mut exact = RiverCfr::new(tree, board);
        fast.run(&r0, &r1, 5).unwrap();
        exact.run(&r0, &r1, 5).unwrap();
        for n in 0..fast.tree.num_decision() {
            for (x, y) in fast.regret(n).iter().zip(exact.regret(n)) {
                assert!(
                    (x - y).abs() <= 1e-9 * (1.0 + y.abs()),
                    "node {n}: {x} vs {y}"
                );
            }
        }
        let (ef, ee) = (
            fast.exploitability(&r0, &r1, None).unwrap(),
            exact.exploitability(&r0, &r1, None).unwrap(),
        );
        assert!((ef - ee).abs() <= 1e-9 * ee.abs(), "{ef} vs {ee}");
    }
}
