//! Info-set storage for the MCCFR trainer: compact action ids, an inline
//! insertion-ordered action→value map (Python's dict semantics, no heap), the
//! `InformationSet` port, and 64-bit packed info-set keys that decode to the
//! exact `keys.make_info_set_key` string.

use crate::engine::{Action, Size};
use crate::keys::STREET_NAMES;

/// Most distinct actions one info set can accumulate. The largest node is a
/// postflop raise node under the capped menu (fold, call, allin, 5 raises = 8).
pub const MAX_ACTIONS: usize = 10;

const SIZES: [Size; 6] = [
    Size::Small,
    Size::Medium,
    Size::Large,
    Size::XLarge,
    Size::Overbet,
    Size::Overbet2,
];

/// Grid action -> compact id (0..16). Custom (off-grid) actions never occur in
/// training and are rejected.
pub fn action_id(a: Action) -> Result<u8, String> {
    let size_idx = |s: Size| SIZES.iter().position(|&x| x == s).unwrap() as u8;
    Ok(match a {
        Action::Check => 0,
        Action::Call => 1,
        Action::Fold => 2,
        Action::AllIn => 3,
        Action::Bet(s) => 4 + size_idx(s),
        Action::Raise(s) => 10 + size_idx(s),
        _ => return Err(format!("not a grid action: {}", a.name())),
    })
}

pub fn id_action(id: u8) -> Action {
    match id {
        0 => Action::Check,
        1 => Action::Call,
        2 => Action::Fold,
        3 => Action::AllIn,
        4..=9 => Action::Bet(SIZES[(id - 4) as usize]),
        10..=15 => Action::Raise(SIZES[(id - 10) as usize]),
        _ => panic!("bad action id {id}"),
    }
}

pub fn id_from_name(name: &str) -> Result<u8, String> {
    action_id(Action::parse(name)?)
}

/// Insertion-ordered `{action: float}` map with inline storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActMap {
    n: u8,
    ids: [u8; MAX_ACTIONS],
    vals: [f64; MAX_ACTIONS],
}

impl Default for ActMap {
    fn default() -> Self {
        ActMap {
            n: 0,
            ids: [0; MAX_ACTIONS],
            vals: [0.0; MAX_ACTIONS],
        }
    }
}

impl ActMap {
    #[inline]
    pub fn len(&self) -> usize {
        self.n as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    #[inline]
    fn find(&self, id: u8) -> Option<usize> {
        self.ids[..self.n as usize].iter().position(|&x| x == id)
    }

    #[inline]
    pub fn get(&self, id: u8) -> Option<f64> {
        self.find(id).map(|i| self.vals[i])
    }

    /// `d[id] = v` (appends a new key at the end, like a dict).
    #[inline]
    pub fn set(&mut self, id: u8, v: f64) -> Result<(), String> {
        match self.find(id) {
            Some(i) => self.vals[i] = v,
            None => {
                let n = self.n as usize;
                if n == MAX_ACTIONS {
                    return Err(format!("info set exceeds {MAX_ACTIONS} actions"));
                }
                self.ids[n] = id;
                self.vals[n] = v;
                self.n += 1;
            }
        }
        Ok(())
    }

    #[inline]
    pub fn scale(&mut self, f: f64) {
        for v in &mut self.vals[..self.n as usize] {
            *v *= f;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (u8, f64)> + '_ {
        self.ids[..self.n as usize]
            .iter()
            .copied()
            .zip(self.vals[..self.n as usize].iter().copied())
    }

    pub fn ids(&self) -> &[u8] {
        &self.ids[..self.n as usize]
    }
}

/// First-seen legal action list (reference only, like Python).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Legal {
    n: u8,
    ids: [u8; MAX_ACTIONS],
}

impl Legal {
    pub fn from_ids(ids: &[u8]) -> Result<Legal, String> {
        if ids.len() > MAX_ACTIONS {
            return Err(format!("legal list exceeds {MAX_ACTIONS} actions"));
        }
        let mut l = Legal::default();
        l.ids[..ids.len()].copy_from_slice(ids);
        l.n = ids.len() as u8;
        Ok(l)
    }

    pub fn ids(&self) -> &[u8] {
        &self.ids[..self.n as usize]
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Port of `cfr/information_set.py:InformationSet` (+ the trainer's clocks).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InfoSet {
    pub legal: Legal,
    pub regrets: ActMap,
    pub strategy: ActMap,
    pub visit_count: i64,
    pub last_visited: i64,
    pub strategy_visit_count: i64,
    pub last_strategy: i64,
}

impl Default for InfoSet {
    fn default() -> Self {
        InfoSet {
            legal: Legal::default(),
            regrets: ActMap::default(),
            strategy: ActMap::default(),
            visit_count: 0,
            last_visited: -1,
            strategy_visit_count: 0,
            last_strategy: -1,
        }
    }
}

/// Python `max(0.0, x)`.
#[inline]
pub fn floor0(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

impl InfoSet {
    /// `get_strategy(legal)`: CFR+ regret matching, numpy reduction order.
    /// Writes `legal.len()` probabilities into `out`.
    #[inline]
    pub fn current_strategy(&mut self, legal: &[u8], out: &mut [f64]) -> Result<(), String> {
        if self.legal.is_empty() {
            self.legal = Legal::from_ids(legal)?;
        }
        let n = legal.len();
        let mut r = [0.0f64; MAX_ACTIONS];
        for (i, &a) in legal.iter().enumerate() {
            r[i] = floor0(self.regrets.get(a).unwrap_or(0.0));
        }
        let total = crate::river::showdown::np_sum(&r[..n]);
        if total > 0.0 {
            for i in 0..n {
                out[i] = r[i] / total;
            }
        } else {
            let u = 1.0 / n as f64;
            out[..n].fill(u);
        }
        Ok(())
    }

    /// `accumulate_strategy(legal, strategy)`.
    #[inline]
    pub fn accumulate(&mut self, legal: &[u8], strat: &[f64]) -> Result<(), String> {
        if self.legal.is_empty() {
            self.legal = Legal::from_ids(legal)?;
        }
        for (i, &a) in legal.iter().enumerate() {
            let prev = self.strategy.get(a).unwrap_or(0.0);
            self.strategy.set(a, prev + strat[i])?;
        }
        Ok(())
    }

    /// Normalised average strategy over the accumulated actions
    /// (`BlueprintDB.get_average_strategy`; uniform over `legal` when empty).
    pub fn average_strategy(&self) -> Vec<(u8, f64)> {
        let total: f64 = self.strategy.iter().map(|(_, v)| v).sum();
        if total > 1e-12 {
            self.strategy.iter().map(|(a, v)| (a, v / total)).collect()
        } else {
            let ids = self.legal.ids();
            let u = 1.0 / ids.len().max(1) as f64;
            ids.iter().map(|&a| (a, u)).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Packed keys
// ---------------------------------------------------------------------------

/// Betting pattern of the current street: up to 10 chars, 4 bits each
/// (codes 1..10; 0 terminates).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    pub code: u64,
    pub len: u8,
}

pub const MAX_PATTERN: usize = 10;
const PATTERN_CHARS: &[u8; 10] = b"kcfsmlxo2a";

impl Pattern {
    #[inline]
    pub fn push(self, ch: u8) -> Result<Pattern, String> {
        let c = PATTERN_CHARS
            .iter()
            .position(|&x| x == ch)
            .ok_or_else(|| format!("bad pattern char {:?}", ch as char))? as u64
            + 1;
        if self.len as usize >= MAX_PATTERN {
            return Err(format!("betting pattern longer than {MAX_PATTERN} chars"));
        }
        Ok(Pattern {
            code: self.code | (c << (4 * self.len as u64)),
            len: self.len + 1,
        })
    }

    pub fn parse(s: &str) -> Result<Pattern, String> {
        s.bytes().try_fold(Pattern::default(), |p, ch| p.push(ch))
    }

    pub fn text(self) -> String {
        (0..self.len)
            .map(|i| PATTERN_CHARS[(((self.code >> (4 * i as u64)) & 0xF) - 1) as usize] as char)
            .collect()
    }
}

const PAT_BITS: u64 = 40;
const STR_BITS: u64 = 10;
const BKT_BITS: u64 = 8;

/// Pack a key. `bucket` is the FINE preflop id preflop and the COARSE class
/// postflop (exactly what the key string prints); `strength` is ignored preflop.
#[inline]
pub fn pack_key(
    street: u8,
    seat: u8,
    bucket: u8,
    strength: u16,
    pat: Pattern,
) -> Result<u64, String> {
    if street > 3 || seat > 1 {
        return Err("bad street/seat".into());
    }
    if strength as u64 >= 1 << STR_BITS {
        return Err(format!(
            "strength bucket {strength} does not fit the packed key"
        ));
    }
    let s = if street == 0 { 0 } else { strength as u64 };
    Ok(pat.code
        | (s << PAT_BITS)
        | ((bucket as u64) << (PAT_BITS + STR_BITS))
        | ((seat as u64) << (PAT_BITS + STR_BITS + BKT_BITS))
        | ((street as u64) << (PAT_BITS + STR_BITS + BKT_BITS + 1)))
}

pub struct KeyParts {
    pub street: u8,
    pub seat: u8,
    pub bucket: u8,
    pub strength: u16,
    pub pattern: Pattern,
}

pub fn unpack_key(k: u64) -> KeyParts {
    let code = k & ((1 << PAT_BITS) - 1);
    let mut len = 0u8;
    while (len as usize) < MAX_PATTERN && (code >> (4 * len as u64)) & 0xF != 0 {
        len += 1;
    }
    KeyParts {
        street: (k >> (PAT_BITS + STR_BITS + BKT_BITS + 1)) as u8 & 3,
        seat: (k >> (PAT_BITS + STR_BITS + BKT_BITS)) as u8 & 1,
        bucket: (k >> (PAT_BITS + STR_BITS)) as u8,
        strength: ((k >> PAT_BITS) & ((1 << STR_BITS) - 1)) as u16,
        pattern: Pattern { code, len },
    }
}

/// The `make_info_set_key` string of a packed key.
pub fn key_string(k: u64) -> String {
    let p = unpack_key(k);
    let pos = crate::keys::position_name(p.seat);
    let pat = p.pattern.text();
    if p.street == 0 {
        format!("pf_{}_{pos}_{pat}", p.bucket)
    } else {
        format!(
            "pf_{}_{}_{pos}_{}_{pat}",
            p.bucket, p.strength, STREET_NAMES[p.street as usize]
        )
    }
}

/// Parse a key string written by Python (resume from an existing DB).
pub fn parse_key(s: &str) -> Result<u64, String> {
    let bad = || format!("unparseable info-set key {s:?}");
    let parts: Vec<&str> = s.split('_').collect();
    let num = |x: &str| x.parse::<u16>().map_err(|_| bad());
    let seat = |x: &str| match x {
        "ip" => Ok(0u8),
        "oop" => Ok(1u8),
        _ => Err(bad()),
    };
    match parts.as_slice() {
        ["pf", b, pos, pat] => {
            let b = num(b)?;
            if b > 255 {
                return Err(bad());
            }
            pack_key(0, seat(pos)?, b as u8, 0, Pattern::parse(pat)?)
        }
        ["pf", b, st, pos, street, pat] => {
            let b = num(b)?;
            let street = STREET_NAMES
                .iter()
                .position(|x| x == street)
                .filter(|&i| i > 0)
                .ok_or_else(bad)? as u8;
            if b > 255 {
                return Err(bad());
            }
            pack_key(street, seat(pos)?, b as u8, num(st)?, Pattern::parse(pat)?)
        }
        _ => Err(bad()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip() {
        for s in [
            "pf_160_ip_",
            "pf_0_oop_csss",
            "pf_9_5_ip_turn_m",
            "pf_3_29_oop_river_kos2a",
            "pf_0_0_ip_flop_",
            "pf_168_oop_xa",
        ] {
            let k = parse_key(s).unwrap();
            assert_eq!(key_string(k), s);
        }
        for bad in [
            "",
            "pf_1_ip",
            "pf_x_ip_",
            "pf_1_mid_",
            "pf_1_2_ip_preflop_",
            "pf_1_ip_q",
            "pf_1_2_ip_flop_kkkkkkkkkkk",
        ] {
            assert!(parse_key(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn act_map_keeps_insertion_order_and_caps() {
        let mut m = ActMap::default();
        m.set(5, 1.0).unwrap();
        m.set(1, 2.0).unwrap();
        m.set(5, 3.0).unwrap();
        assert_eq!(m.iter().collect::<Vec<_>>(), vec![(5, 3.0), (1, 2.0)]);
        for i in 0..8 {
            m.set(20 + i, 0.0).unwrap();
        }
        assert!(m.set(99, 0.0).is_err());
    }

    #[test]
    fn action_ids_round_trip() {
        for id in 0..16u8 {
            assert_eq!(action_id(id_action(id)).unwrap(), id);
        }
    }
}
