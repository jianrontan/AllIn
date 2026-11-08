//! State-threaded betting engine. Port of the "Lever A" path of
//! `cfr/poker_game.py` (`init_node_state`, `state_legal_actions`,
//! `state_action_cost`, `advance_node_state`, `is_terminal`, `get_utility`
//! with threaded pot/total) — the code `BlueprintTrainer.cfr` runs.
//!
//! Every float expression keeps the Python operation order so chip amounts are
//! bit-identical (the parity fixtures compare with `==`).

/// `poker_game.STARTING_STACK` (100 BB at BB = 2 chips).
pub const STARTING_STACK: f64 = 200.0;
/// `sizing.BIG_BLIND`; also the minimum postflop opening bet.
pub const BIG_BLIND: f64 = 2.0;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Size {
    Small,
    Medium,
    Large,
    XLarge,
    Overbet,
    Overbet2,
}

impl Size {
    pub fn name(self) -> &'static str {
        match self {
            Size::Small => "small",
            Size::Medium => "medium",
            Size::Large => "large",
            Size::XLarge => "xlarge",
            Size::Overbet => "overbet",
            Size::Overbet2 => "overbet2",
        }
    }

    fn parse(s: &str) -> Option<Size> {
        Some(match s {
            "small" => Size::Small,
            "medium" => Size::Medium,
            "large" => Size::Large,
            "xlarge" => Size::XLarge,
            "overbet" => Size::Overbet,
            "overbet2" => Size::Overbet2,
            _ => return None,
        })
    }

    /// `sizing.PREFLOP_OPEN_BB[size] * BIG_BLIND` (raise-to total, chips).
    fn open_chips(self) -> Option<f64> {
        Some(match self {
            Size::Small => 2.0 * BIG_BLIND,
            Size::Medium => 2.5 * BIG_BLIND,
            Size::Large => 3.5 * BIG_BLIND,
            Size::XLarge => 5.0 * BIG_BLIND,
            _ => return None,
        })
    }

    /// `sizing.PREFLOP_RAISE_MULT[size]` (3-bet / 4-bet+, pot-relative).
    fn preflop_raise_mult(self) -> Option<f64> {
        Some(match self {
            Size::Small => 0.66,
            Size::Medium => 1.0,
            Size::Large => 1.5,
            _ => return None,
        })
    }
}

const PREFLOP_RAISE_SIZES: [Size; 3] = [Size::Small, Size::Medium, Size::Large];
const OPEN_SIZES: [Size; 4] = [Size::Small, Size::Medium, Size::Large, Size::XLarge];

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Action {
    Check,
    Call,
    Fold,
    Bet(Size),
    Raise(Size),
    AllIn,
    /// Off-grid human bet/raise; the payload is the raise-to street total.
    BetCustom(f64),
    RaiseCustom(f64),
}

impl Action {
    /// `startswith(('bet_', 'raise_'))`.
    pub fn is_bet_or_raise(self) -> bool {
        matches!(
            self,
            Action::Bet(_) | Action::Raise(_) | Action::BetCustom(_) | Action::RaiseCustom(_)
        )
    }

    pub fn name(self) -> String {
        match self {
            Action::Check => "check".into(),
            Action::Call => "call".into(),
            Action::Fold => "fold".into(),
            Action::AllIn => "allin".into(),
            Action::Bet(s) => format!("bet_{}", s.name()),
            Action::Raise(s) => format!("raise_{}", s.name()),
            Action::BetCustom(t) => format!("bet_custom_{}", crate::river::tree::fmt_g6(t)),
            Action::RaiseCustom(t) => format!("raise_custom_{}", crate::river::tree::fmt_g6(t)),
        }
    }

    pub fn parse(s: &str) -> Result<Action, String> {
        Ok(match s {
            "check" => Action::Check,
            "call" => Action::Call,
            "fold" => Action::Fold,
            "allin" => Action::AllIn,
            _ => {
                let bad = || format!("unknown action {s:?}");
                let (kind, rest) = s.split_once('_').ok_or_else(bad)?;
                if let Some(total) = rest.strip_prefix("custom_") {
                    let t: f64 = total.parse().map_err(|_| bad())?;
                    if !t.is_finite() || t < 0.0 {
                        return Err(bad());
                    }
                    match kind {
                        "bet" => Action::BetCustom(t),
                        "raise" => Action::RaiseCustom(t),
                        _ => return Err(bad()),
                    }
                } else {
                    let size = Size::parse(rest).ok_or_else(bad)?;
                    match kind {
                        "bet" => Action::Bet(size),
                        "raise" => Action::Raise(size),
                        _ => return Err(bad()),
                    }
                }
            }
        })
    }
}

/// `last_kind` / `prev_kind` of the threaded state.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Kind {
    None,
    Start,
    Check,
    Call,
    Aggr,
    Fold,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::None => "none",
            Kind::Start => "start",
            Kind::Check => "check",
            Kind::Call => "call",
            Kind::Aggr => "aggr",
            Kind::Fold => "fold",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "none" => Kind::None,
            "start" => Kind::Start,
            "check" => Kind::Check,
            "call" => Kind::Call,
            "aggr" => Kind::Aggr,
            "fold" => Kind::Fold,
            _ => return None,
        })
    }
}

/// Within-street betting state (the Python state dict).
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NodeState {
    pub pot: f64,
    /// This-street contributions per seat.
    pub c: [f64; 2],
    pub bet_to: f64,
    pub prev_bet_to: f64,
    pub num_br: u32,
    pub last_kind: Kind,
    pub prev_kind: Kind,
    pub allin_seen: bool,
}

/// Python `max(a, b)`: `a` unless `b > a`.
#[inline]
fn pymax(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

#[derive(Clone, Debug)]
pub struct Engine {
    /// Postflop bet/raise menu in dict order (name, pot multiplier).
    pub menu: Vec<(Size, f64)>,
    pub voluntary_allin: bool,
    /// Raises allowed beyond the opening bet (f64 so live play can pass inf).
    pub max_raises_per_street: f64,
}

impl Engine {
    /// `BlueprintTrainer(menu_mode).game`.
    pub fn for_mode(mode: &str) -> Result<Engine, String> {
        let base = vec![
            (Size::Small, 0.33),
            (Size::Medium, 0.66),
            (Size::Large, 1.0),
            (Size::Overbet, 1.5),
        ];
        Ok(match mode {
            "control" => Engine {
                menu: base,
                voluntary_allin: true,
                max_raises_per_street: 2.0,
            },
            "capped" => {
                let mut m = base;
                m.push((Size::Overbet2, 2.0));
                Engine {
                    menu: m,
                    voluntary_allin: false,
                    max_raises_per_street: 2.0,
                }
            }
            "capped_no2" => Engine {
                menu: base,
                voluntary_allin: false,
                max_raises_per_street: 2.0,
            },
            _ => {
                return Err(format!(
                    "menu_mode must be 'control', 'capped', or 'capped_no2', got {mode:?}"
                ));
            }
        })
    }

    fn menu_mult(&self, size: Size) -> Option<f64> {
        self.menu.iter().find(|(s, _)| *s == size).map(|&(_, m)| m)
    }

    pub fn init_node_state(&self, street: u8, starting_pot: f64) -> NodeState {
        let (c, bet_to) = if street == 0 {
            ([1.0, 2.0], 2.0)
        } else {
            ([0.0, 0.0], 0.0)
        };
        NodeState {
            pot: starting_pot,
            c,
            bet_to,
            prev_bet_to: 0.0,
            num_br: 0,
            last_kind: Kind::Start,
            prev_kind: Kind::None,
            allin_seen: false,
        }
    }

    /// Preflop SB (0) acts first; postflop BB (1) acts first.
    #[inline]
    pub fn acting_player(n_actions: usize, street: u8) -> u8 {
        let offset = if street > 0 { 1 } else { 0 };
        ((n_actions + offset) & 1) as u8
    }

    #[inline]
    fn round_complete(st: &NodeState) -> bool {
        (st.last_kind == Kind::Call && st.num_br >= 1)
            || (st.last_kind == Kind::Check && matches!(st.prev_kind, Kind::Check | Kind::Call))
    }

    #[inline]
    pub fn to_call(st: &NodeState, cp: u8) -> f64 {
        let x = st.bet_to - st.c[cp as usize];
        pymax(0.0, x)
    }

    #[inline]
    fn min_raise(st: &NodeState) -> f64 {
        if st.num_br == 0 {
            2.0
        } else {
            st.bet_to + (st.bet_to - st.prev_bet_to)
        }
    }

    /// Raise-to street total for a sized bet/raise (`_ns_sized_total`).
    fn sized_total(&self, size: Size, street: u8, st: &NodeState, cp: u8) -> Result<f64, String> {
        let to_call = Self::to_call(st, cp);
        let bad = || format!("size {} not offered here", size.name());
        if street == 0 {
            if st.num_br == 0 {
                return size.open_chips().ok_or_else(bad);
            }
            let m = size.preflop_raise_mult().ok_or_else(bad)?;
            return Ok(m * (st.pot + to_call) + to_call);
        }
        let mult = self.menu_mult(size).ok_or_else(bad)?;
        Ok(if to_call > 0.0 {
            mult * (st.pot + to_call) + to_call
        } else {
            mult * st.pot
        })
    }

    /// Chips `cp` adds for `action` (`state_action_cost`).
    pub fn action_cost(
        &self,
        action: Action,
        street: u8,
        st: &NodeState,
        cp: u8,
        stack_cp: f64,
    ) -> Result<f64, String> {
        Ok(match action {
            Action::Check | Action::Fold => 0.0,
            Action::Call => Self::to_call(st, cp),
            Action::AllIn => stack_cp,
            Action::BetCustom(t) | Action::RaiseCustom(t) => t - st.c[cp as usize],
            Action::Bet(s) | Action::Raise(s) => {
                self.sized_total(s, street, st, cp)? - st.c[cp as usize]
            }
        })
    }

    /// Legal actions in Python's order (`state_legal_actions`).
    pub fn legal_actions(&self, street: u8, st: &NodeState, cp: u8, stack_cp: f64) -> Vec<Action> {
        let mut out = Vec::with_capacity(8);
        self.legal_actions_into(street, st, cp, stack_cp, &mut out);
        out
    }

    /// Allocation-free variant for the training hot path (`out` is cleared).
    pub fn legal_actions_into(
        &self,
        street: u8,
        st: &NodeState,
        cp: u8,
        stack_cp: f64,
        out: &mut Vec<Action>,
    ) {
        out.clear();
        if Self::round_complete(st) {
            return;
        }
        if st.allin_seen || (st.num_br as f64) >= self.max_raises_per_street + 1.0 {
            out.extend([Action::Fold, Action::Call]);
            return;
        }
        let lk = st.last_kind;
        if street == 0 {
            if lk == Kind::Start {
                out.extend([Action::Fold, Action::Call]);
                out.extend(OPEN_SIZES.iter().map(|&s| Action::Bet(s)));
            } else if lk == Kind::Call && st.num_br == 0 {
                out.push(Action::Check);
                out.extend(OPEN_SIZES.iter().map(|&s| Action::Bet(s)));
            } else {
                self.fold_call_raises(street, st, cp, out);
            }
        } else if matches!(lk, Kind::Start | Kind::Check) {
            out.push(Action::Check);
            for &(s, m) in &self.menu {
                if m * st.pot >= 2.0 {
                    out.push(Action::Bet(s));
                }
            }
        } else {
            self.fold_call_raises(street, st, cp, out);
        }
        self.stack_constrain(street, st, cp, stack_cp, out);
    }

    fn fold_call_raises(&self, street: u8, st: &NodeState, cp: u8, out: &mut Vec<Action>) {
        out.extend([Action::Fold, Action::Call]);
        let raise_count = st.num_br.saturating_sub(1);
        if (raise_count as f64) < self.max_raises_per_street {
            let mr = Self::min_raise(st);
            let mut push = |s: Size| {
                // sized_total cannot fail for these sizes (menu / 3-bet sizes).
                if let Ok(t) = self.sized_total(s, street, st, cp) {
                    if t >= mr {
                        out.push(Action::Raise(s));
                    }
                }
            };
            if street > 0 {
                for &(s, _) in &self.menu {
                    push(s);
                }
            } else {
                for s in PREFLOP_RAISE_SIZES {
                    push(s);
                }
            }
        }
    }

    fn stack_constrain(
        &self,
        street: u8,
        st: &NodeState,
        cp: u8,
        stack_cp: f64,
        out: &mut Vec<Action>,
    ) {
        let mut needs_allin = false;
        out.retain(|&a| {
            if !a.is_bet_or_raise() {
                return true;
            }
            let cost = self
                .action_cost(a, street, st, cp, stack_cp)
                .unwrap_or(f64::INFINITY);
            if cost >= stack_cp {
                needs_allin = true;
                false
            } else {
                true
            }
        });
        if !needs_allin && self.voluntary_allin && stack_cp > Self::to_call(st, cp) {
            needs_allin = true;
        }
        if needs_allin && !out.contains(&Action::AllIn) {
            let i = out
                .iter()
                .position(|a| a.is_bet_or_raise())
                .unwrap_or(out.len());
            out.insert(i, Action::AllIn);
        }
    }

    /// New state after `cp` takes `action` (`advance_node_state`). `p_inv` is
    /// the cross-street prior investment per seat.
    pub fn advance(
        &self,
        st: &NodeState,
        action: Action,
        street: u8,
        cp: u8,
        stack_cp: f64,
        p_inv: [f64; 2],
    ) -> Result<NodeState, String> {
        let mut new = NodeState {
            prev_kind: st.last_kind,
            allin_seen: st.allin_seen || action == Action::AllIn,
            ..*st
        };
        let cost = self.action_cost(action, street, st, cp, stack_cp)?;
        new.pot += cost;
        let p = cp as usize;
        match action {
            Action::Check => new.last_kind = Kind::Check,
            Action::Fold => new.last_kind = Kind::Fold,
            Action::Call => {
                new.c[p] += cost;
                new.last_kind = Kind::Call;
            }
            _ => {
                new.prev_bet_to = st.bet_to;
                let match_level = if action == Action::AllIn {
                    new.c[p] += cost;
                    STARTING_STACK - p_inv[p]
                } else {
                    let total = match action {
                        Action::BetCustom(t) | Action::RaiseCustom(t) => t,
                        Action::Bet(s) | Action::Raise(s) => self.sized_total(s, street, st, cp)?,
                        _ => unreachable!(),
                    };
                    new.c[p] = total;
                    total
                };
                new.bet_to = pymax(st.bet_to, match_level);
                new.num_br += 1;
                new.last_kind = Kind::Aggr;
            }
        }
        Ok(new)
    }
}

/// `PokerGame.is_round_complete(history)`.
pub fn is_round_complete(h: &[Action]) -> bool {
    let n = h.len();
    if n == 0 {
        return false;
    }
    if h.contains(&Action::Fold) {
        return true;
    }
    if n >= 2 {
        let (a, b) = (h[n - 2], h[n - 1]);
        if (a == Action::Call || a == Action::Check) && b == Action::Check {
            return true;
        }
        if b == Action::Call && a.is_bet_or_raise() {
            return true;
        }
        if a == Action::AllIn && (b == Action::Call || b == Action::Fold) {
            return true;
        }
    }
    false
}

/// `PokerGame.is_terminal(history, street)`.
pub fn is_terminal(h: &[Action], street: u8) -> bool {
    if h.contains(&Action::Fold) {
        return true;
    }
    let n = h.len();
    if n >= 2 && h[n - 2] == Action::AllIn && h[n - 1] == Action::Call {
        return true;
    }
    street == 3 && is_round_complete(h)
}

/// Community cards visible per street (`get_community_cards_count`).
pub fn board_count(street: u8) -> usize {
    [0, 3, 4, 5][street.min(3) as usize]
}

/// Terminal utility from P0's perspective with threaded `final_pot` /
/// `p0_total` (`get_utility(..., _final_pot, _p0_total)`). `rank(cards)`
/// returns the 5..7-card rank (lower = stronger).
#[allow(clippy::too_many_arguments)]
pub fn utility<F: FnMut(&[u8]) -> i32>(
    p0: [u8; 2],
    p1: [u8; 2],
    board: &[u8],
    history: &[Action],
    street: u8,
    final_pot: f64,
    p0_total: f64,
    mut rank: F,
) -> f64 {
    if let Some(i) = history.iter().position(|&a| a == Action::Fold) {
        return if Engine::acting_player(i, street) == 0 {
            -p0_total
        } else {
            final_pot - p0_total
        };
    }
    let nb = if history.contains(&Action::AllIn) {
        board.len().min(5)
    } else {
        board_count(street).min(board.len())
    };
    let mut buf = [0u8; 7];
    let n = 2 + nb;
    buf[2..n].copy_from_slice(&board[..nb]);
    buf[..2].copy_from_slice(&p0);
    let r0 = rank(&buf[..n]);
    buf[..2].copy_from_slice(&p1);
    let r1 = rank(&buf[..n]);
    if r0 < r1 {
        final_pot - p0_total
    } else if r1 < r0 {
        -p0_total
    } else {
        final_pot / 2.0 - p0_total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_names_round_trip() {
        for s in [
            "check",
            "call",
            "fold",
            "allin",
            "bet_small",
            "bet_xlarge",
            "raise_overbet2",
            "bet_custom_37.5",
            "raise_custom_120",
        ] {
            assert_eq!(Action::parse(s).unwrap().name(), s);
        }
        for bad in [
            "",
            "bet",
            "bet_huge",
            "jam_small",
            "bet_custom_x",
            "bet_custom_nan",
            "bet_custom_-1",
        ] {
            assert!(Action::parse(bad).is_err(), "{bad}");
        }
    }

    /// Random walks: chips are conserved and every walk terminates within the
    /// trained aggression cap.
    #[test]
    fn walks_conserve_chips_and_terminate() {
        for mode in ["control", "capped", "capped_no2"] {
            let e = Engine::for_mode(mode).unwrap();
            let mut seed = 12345u64;
            let mut rnd = |n: usize| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as usize) % n
            };
            for _ in 0..2000 {
                let (mut street, mut pot) = (0u8, 3.0);
                let mut p_inv = [0.0, 0.0];
                let mut stacks = [STARTING_STACK - 1.0, STARTING_STACK - 2.0];
                let mut st = e.init_node_state(0, pot);
                let mut h: Vec<Action> = Vec::new();
                let mut steps = 0;
                loop {
                    steps += 1;
                    assert!(steps < 80);
                    if is_terminal(&h, street) {
                        break;
                    }
                    let cp = Engine::acting_player(h.len(), street);
                    let legal = e.legal_actions(street, &st, cp, stacks[cp as usize]);
                    if legal.is_empty() {
                        if street == 3 {
                            break;
                        }
                        p_inv = [p_inv[0] + st.c[0], p_inv[1] + st.c[1]];
                        stacks = [STARTING_STACK - p_inv[0], STARTING_STACK - p_inv[1]];
                        pot = st.pot;
                        street += 1;
                        h.clear();
                        st = e.init_node_state(street, pot);
                        continue;
                    }
                    let a = legal[rnd(legal.len())];
                    let cost = e
                        .action_cost(a, street, &st, cp, stacks[cp as usize])
                        .unwrap();
                    assert!(cost >= 0.0 && cost <= stacks[cp as usize] + 1e-9);
                    st = e
                        .advance(&st, a, street, cp, stacks[cp as usize], p_inv)
                        .unwrap();
                    stacks[cp as usize] -= cost;
                    h.push(a);
                    let total = p_inv[0] + st.c[0] + p_inv[1] + st.c[1];
                    assert!(
                        (total - st.pot).abs() < 1e-9,
                        "{mode}: pot {} vs {total}",
                        st.pot
                    );
                    assert!(st.num_br <= 3);
                }
            }
        }
    }
}
