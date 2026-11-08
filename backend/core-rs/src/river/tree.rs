//! River betting tree. Port of `subgame/river_tree.py` (`RiverTree._build`,
//! `_add_aggressive`, `_terminal`), producing the same nodes in the same
//! pre-order with bit-identical chip amounts and identical action labels.
//!
//! Conventions (see the Python docstring): OOP = seat 1 acts first; each seat
//! already invested `pot_entry / 2`; bet = frac * pot; raise-to =
//! frac * (pot + to_call) + to_call; at most `max_aggressions` bets/raises;
//! a sized action that would cost the whole stack collapses into all-in.

/// 1 big blind in chips (`sizing.BIG_BLIND`): the minimum opening bet.
pub const MIN_BET: f64 = 2.0;
pub const DEFAULT_MENU: [f64; 4] = [0.5, 0.75, 1.0, 1.5];
pub const MAX_AGGRESSIONS: u32 = 3;
pub const OOP_SEAT: u8 = 1;

#[derive(Clone, Debug)]
pub struct Decision {
    pub player: u8,
    pub to_call: f64,
    pub pot_mid: f64,
    pub sc: [f64; 2],
    pub actions: Vec<String>,
    /// Indices into `RiverTree::nodes`, parallel to `actions`.
    pub children: Vec<usize>,
    /// Dense index among decision nodes (pre-order), keys the CFR tables.
    pub node_id: usize,
    pub agg: u32,
    pub last_inc: f64,
}

#[derive(Clone, Debug)]
pub struct Terminal {
    pub final_pot: f64,
    pub contrib: [f64; 2],
    /// Seat that folded, or `None` for a showdown.
    pub folder: Option<u8>,
}

#[derive(Clone, Debug)]
pub enum Node {
    Decision(Decision),
    Terminal(Terminal),
}

#[derive(Clone, Debug)]
pub struct RiverTree {
    /// All nodes in pre-order; the root is `nodes[0]`.
    pub nodes: Vec<Node>,
    /// `decision[node_id]` = index into `nodes`.
    pub decision: Vec<usize>,
    pub pot_entry: f64,
}

/// Python's `f"{x:.6g}"`.
pub fn fmt_g6(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".into()
        } else if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    // Exponent after rounding to 6 significant digits.
    let sci = format!("{x:.5e}");
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", strip(mant), sign, exp.abs())
    } else {
        strip(&format!("{:.*}", (5 - exp) as usize, x))
    }
}

/// Python's `round(x, 6)` as a hashable key (equal keys iff equal rounded floats).
fn cost_key(x: f64) -> String {
    format!("{x:.6}")
}

fn fmax(a: f64, b: f64) -> f64 {
    // Python max(a, b): returns a unless b > a.
    if b > a { b } else { a }
}

struct Builder<'a> {
    pot_entry: f64,
    stacks: [f64; 2],
    menu: &'a [f64],
    max_aggressions: u32,
    entry_contrib: f64,
    nodes: Vec<Node>,
    decision: Vec<usize>,
}

impl RiverTree {
    /// `RiverTree(pot_entry, (stack, stack), menu, max_aggressions)`.
    pub fn build(
        pot_entry: f64,
        stacks: [f64; 2],
        menu: &[f64],
        max_aggressions: u32,
    ) -> Result<Self, String> {
        if pot_entry <= 0.0 {
            return Err("pot_entry must be positive".into());
        }
        if (stacks[0] - stacks[1]).abs() > 1e-6 {
            return Err(format!(
                "river tree assumes equal river-entry stacks (got {stacks:?}); \
                 unequal stacks need all-in-for-less handling"
            ));
        }
        let mut b = Builder {
            pot_entry,
            stacks,
            menu,
            max_aggressions,
            entry_contrib: pot_entry / 2.0,
            nodes: Vec::new(),
            decision: Vec::new(),
        };
        b.build(OOP_SEAT, [0.0, 0.0], 0, MIN_BET, false);
        Ok(RiverTree {
            nodes: b.nodes,
            decision: b.decision,
            pot_entry,
        })
    }

    /// Assemble a tree from already-built nodes (e.g. a Python `RiverTree`
    /// after `inject_realized_edge`). Checks the structure the solver relies on.
    pub fn from_nodes(nodes: Vec<Node>, pot_entry: f64) -> Result<Self, String> {
        if nodes.is_empty() {
            return Err("empty tree".into());
        }
        let n_dec = nodes
            .iter()
            .filter(|n| matches!(n, Node::Decision(_)))
            .count();
        let mut decision = vec![usize::MAX; n_dec];
        for (i, n) in nodes.iter().enumerate() {
            if let Node::Decision(d) = n {
                if d.node_id >= n_dec || decision[d.node_id] != usize::MAX {
                    return Err(format!("node {i}: bad or duplicate node_id {}", d.node_id));
                }
                if d.player > 1 || d.children.len() != d.actions.len() || d.children.is_empty() {
                    return Err(format!("node {i}: bad player/actions/children"));
                }
                if d.children.iter().any(|&c| c <= i || c >= nodes.len()) {
                    return Err(format!(
                        "node {i}: child index must point forward (pre-order)"
                    ));
                }
                decision[d.node_id] = i;
            }
        }
        Ok(RiverTree {
            nodes,
            decision,
            pot_entry,
        })
    }

    pub fn num_decision(&self) -> usize {
        self.decision.len()
    }

    pub fn decision_node(&self, node_id: usize) -> &Decision {
        match &self.nodes[self.decision[node_id]] {
            Node::Decision(d) => d,
            Node::Terminal(_) => unreachable!(),
        }
    }
}

impl Builder<'_> {
    fn terminal(&mut self, sc: [f64; 2], folder: Option<u8>) -> usize {
        self.nodes.push(Node::Terminal(Terminal {
            final_pot: self.pot_entry + sc[0] + sc[1],
            contrib: [self.entry_contrib + sc[0], self.entry_contrib + sc[1]],
            folder,
        }));
        self.nodes.len() - 1
    }

    fn build(
        &mut self,
        player: u8,
        sc: [f64; 2],
        agg: u32,
        last_inc: f64,
        prev_check: bool,
    ) -> usize {
        let p = player as usize;
        let max_sc = fmax(sc[0], sc[1]);
        let to_call = max_sc - sc[p];
        let pot_mid = self.pot_entry + sc[0] + sc[1];
        let idx = self.nodes.len();
        let node_id = self.decision.len();
        self.decision.push(idx);
        self.nodes.push(Node::Decision(Decision {
            player,
            to_call,
            pot_mid,
            sc,
            actions: Vec::new(),
            children: Vec::new(),
            node_id,
            agg,
            last_inc,
        }));
        let stack_p = self.stacks[p] - sc[p];
        let other = 1 - player;
        let mut edges: Vec<(String, usize)> = Vec::new();

        if to_call <= 1e-9 {
            let child = if prev_check {
                self.terminal(sc, None)
            } else {
                self.build(other, sc, agg, last_inc, true)
            };
            edges.push(("check".into(), child));
            self.add_aggressive(
                &mut edges, player, sc, agg, last_inc, pot_mid, to_call, stack_p, true,
            );
        } else {
            let fold = self.terminal(sc, Some(player));
            edges.push(("fold".into(), fold));
            let call_amt = if stack_p < to_call { stack_p } else { to_call };
            let mut sc_call = sc;
            sc_call[p] += call_amt;
            let call = self.terminal(sc_call, None);
            edges.push(("call".into(), call));
            let opp_behind = self.stacks[other as usize] - sc[other as usize];
            if agg < self.max_aggressions && opp_behind > 1e-9 {
                self.add_aggressive(
                    &mut edges, player, sc, agg, last_inc, pot_mid, to_call, stack_p, false,
                );
            }
        }
        if let Node::Decision(d) = &mut self.nodes[idx] {
            for (a, c) in edges {
                d.actions.push(a);
                d.children.push(c);
            }
        }
        idx
    }

    #[allow(clippy::too_many_arguments)]
    fn add_aggressive(
        &mut self,
        edges: &mut Vec<(String, usize)>,
        player: u8,
        sc: [f64; 2],
        agg: u32,
        last_inc: f64,
        pot_mid: f64,
        to_call: f64,
        stack_p: f64,
        opening: bool,
    ) {
        if stack_p <= 1e-9 {
            return;
        }
        let p = player as usize;
        let other = 1 - player;
        let max_sc = fmax(sc[0], sc[1]);
        let mut need_allin = false;
        let mut seen: Vec<String> = Vec::new();
        for &frac in self.menu {
            let (amount, new_sc_p, inc, legal_min);
            if opening {
                amount = frac * pot_mid;
                new_sc_p = sc[p] + amount;
                inc = amount;
                legal_min = amount >= MIN_BET - 1e-9;
            } else {
                new_sc_p = frac * (pot_mid + to_call) + to_call;
                amount = new_sc_p - sc[p];
                inc = new_sc_p - max_sc;
                legal_min = inc >= last_inc - 1e-9 && amount > 0.0;
            }
            if !legal_min {
                continue;
            }
            if amount >= stack_p - 1e-9 {
                need_allin = true;
                continue;
            }
            let key = cost_key(amount);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let mut child_sc = sc;
            child_sc[p] = new_sc_p;
            let kind = if opening { "bet" } else { "raise" };
            let child = self.build(other, child_sc, agg + 1, inc, false);
            edges.push((format!("{kind}:{}", fmt_g6(amount)), child));
        }
        let allin_new_sc = sc[p] + stack_p;
        if need_allin || allin_new_sc > max_sc + 1e-9 {
            let mut child_sc = sc;
            child_sc[p] = allin_new_sc;
            let inc = allin_new_sc - max_sc;
            let child = self.build(other, child_sc, agg + 1, fmax(inc, last_inc), false);
            edges.push(("allin".into(), child));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g6_matches_python() {
        for (x, s) in [
            (6.0, "6"),
            (2.5, "2.5"),
            (6.666666666666667, "6.66667"),
            (123456.7, "123457"),
            (1234567.0, "1.23457e+06"),
            (0.0001234, "0.0001234"),
            (0.00001234, "1.234e-05"),
            (97.0, "97"),
            (999999.5, "1e+06"),
        ] {
            assert_eq!(fmt_g6(x), s, "{x}");
        }
    }

    #[test]
    fn chips_are_conserved() {
        let t = RiverTree::build(12.0, [194.0, 194.0], &DEFAULT_MENU, MAX_AGGRESSIONS).unwrap();
        assert_eq!(t.num_decision(), 194);
        for n in &t.nodes {
            if let Node::Terminal(x) = n {
                assert!((x.contrib[0] + x.contrib[1] - x.final_pot).abs() < 1e-9);
            }
        }
    }
}
