//! Info-set keys. Port of `cfr/keys.py` (`make_info_set_key`, `action_char`).
//!
//! Preflop : `pf_{fine}_{position}_{pattern}`
//! Postflop: `pf_{coarse}_{strength}_{position}_{street}_{pattern}` where the
//! fine preflop bucket is collapsed to its coarse class (imperfect recall).
//! spr5    : `pf_{coarse}_{strength}_{position}_{street}_r{spr}_{pattern}` on flop
//! and turn (Phase 6; `keys.spr_bucket`).

use crate::engine::{Action, Size};

pub const STREET_NAMES: [&str; 4] = ["preflop", "flop", "turn", "river"];

/// `keys.ACTION_CHARS`. Custom (off-grid) actions have no char — Python raises.
pub fn action_char(a: Action) -> Result<u8, String> {
    Ok(match a {
        Action::Check => b'k',
        Action::Call => b'c',
        Action::Fold => b'f',
        Action::AllIn => b'a',
        Action::Bet(s) | Action::Raise(s) => match (a, s) {
            (Action::Raise(_), Size::XLarge) => {
                return Err(
                    "action_char: unmapped action 'raise_xlarge' (not a grid action)".into(),
                );
            }
            (_, Size::Small) => b's',
            (_, Size::Medium) => b'm',
            (_, Size::Large) => b'l',
            (_, Size::XLarge) => b'x',
            (_, Size::Overbet) => b'o',
            (_, Size::Overbet2) => b'2',
        },
        Action::BetCustom(_) | Action::RaiseCustom(_) => {
            return Err(format!(
                "action_char: unmapped action {:?} (not a grid action)",
                a.name()
            ));
        }
    })
}

/// `keys.KEY_FORMATS`: which info-set key format a blueprint uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KeyFormat {
    #[default]
    V2,
    Spr5,
}

impl KeyFormat {
    pub fn parse(s: &str) -> Result<KeyFormat, String> {
        match s {
            "v2" => Ok(KeyFormat::V2),
            "spr5" => Ok(KeyFormat::Spr5),
            _ => Err(format!("unknown key_format {s:?}")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            KeyFormat::V2 => "v2",
            KeyFormat::Spr5 => "spr5",
        }
    }
}

/// `keys.SPR_EDGES` / `keys._KEY_STARTING_STACK`.
pub const SPR_EDGES: [f64; 4] = [2.0, 5.0, 12.0, 40.0];
const KEY_STARTING_STACK: f64 = 200.0;

/// `keys.spr_bucket`: the SPR bucket (0..=4) a flop/turn key carries in the
/// spr5 format, from the pot when the street began; None for v2 / preflop /
/// river. Same float operations as Python, so the bucket is bit-identical.
pub fn spr_bucket(fmt: KeyFormat, street: u8, street_pot: f64) -> Result<Option<u8>, String> {
    if fmt == KeyFormat::V2 || !(street == 1 || street == 2) {
        return Ok(None);
    }
    if !street_pot.is_finite() || street_pot <= 0.0 {
        return Err(format!("street_pot must be > 0, got {street_pot}"));
    }
    let spr = (KEY_STARTING_STACK - street_pot / 2.0) / street_pot;
    Ok(Some(SPR_EDGES.iter().filter(|&&e| spr >= e).count() as u8))
}

pub fn position_name(seat: u8) -> &'static str {
    if seat == 0 { "ip" } else { "oop" }
}

/// `make_info_set_key`. `fine` is the acting player's FINE preflop bucket;
/// `fine_to_coarse` maps it to the coarse class for postflop keys.
/// `strength` is ignored preflop; `None` postflop prints as Python's `None`.
pub fn make_info_set_key(
    street: u8,
    position: &str,
    fine: usize,
    strength: Option<i64>,
    pattern: &str,
    fine_to_coarse: &[u8],
    spr: Option<u8>,
) -> Result<String, String> {
    if street == 0 {
        if spr.is_some() {
            return Err("preflop keys carry no SPR bucket".into());
        }
        return Ok(format!("pf_{fine}_{position}_{pattern}"));
    }
    let name = STREET_NAMES
        .get(street as usize)
        .ok_or_else(|| format!("bad street {street}"))?;
    let coarse = fine_to_coarse
        .get(fine)
        .ok_or_else(|| format!("fine preflop bucket {fine} out of range"))?;
    let s = match strength {
        Some(v) => v.to_string(),
        None => "None".to_string(),
    };
    match spr {
        None => Ok(format!("pf_{coarse}_{s}_{position}_{name}_{pattern}")),
        Some(_) if street > 2 => Err(format!("street {street} keys carry no SPR bucket")),
        Some(b) if b as usize > SPR_EDGES.len() => Err(format!("SPR bucket {b} out of range")),
        Some(b) => Ok(format!("pf_{coarse}_{s}_{position}_{name}_r{b}_{pattern}")),
    }
}

/// Parse a `pf_<n>` bucket id.
pub fn parse_bucket(id: &str) -> Result<usize, String> {
    id.strip_prefix("pf_")
        .and_then(|n| n.parse::<usize>().ok())
        .ok_or_else(|| format!("not a preflop bucket id: {id:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_have_python_shape() {
        let f2c: Vec<u8> = (0..169).map(|i| (i / 17) as u8).collect();
        assert_eq!(
            make_info_set_key(0, "ip", 160, None, "", &f2c, None).unwrap(),
            "pf_160_ip_"
        );
        assert_eq!(
            make_info_set_key(2, "ip", 160, Some(5), "m", &f2c, None).unwrap(),
            "pf_9_5_ip_turn_m"
        );
        assert!(make_info_set_key(1, "oop", 169, Some(1), "", &f2c, None).is_err());
        assert!(make_info_set_key(4, "oop", 1, Some(1), "", &f2c, None).is_err());
    }

    #[test]
    fn spr_buckets_follow_pot_types() {
        let f = KeyFormat::Spr5;
        assert_eq!(spr_bucket(f, 1, 4.0), Ok(Some(4)));
        assert_eq!(spr_bucket(f, 1, 8.0), Ok(Some(3)));
        assert_eq!(spr_bucket(f, 2, 20.0), Ok(Some(2)));
        assert_eq!(spr_bucket(f, 2, 45.0), Ok(Some(1)));
        assert_eq!(spr_bucket(f, 1, 100.0), Ok(Some(0)));
        assert_eq!(spr_bucket(f, 0, 3.0), Ok(None));
        assert_eq!(spr_bucket(f, 3, 10.0), Ok(None));
        assert_eq!(spr_bucket(KeyFormat::V2, 1, 10.0), Ok(None));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(spr_bucket(f, 1, bad).is_err());
        }
        let f2c: Vec<u8> = (0..169).map(|i| (i / 17) as u8).collect();
        assert_eq!(
            make_info_set_key(1, "ip", 160, Some(5), "km", &f2c, Some(3)).unwrap(),
            "pf_9_5_ip_flop_r3_km"
        );
        assert!(make_info_set_key(1, "ip", 160, Some(5), "", &f2c, Some(5)).is_err());
        assert!(make_info_set_key(3, "ip", 160, Some(5), "", &f2c, Some(1)).is_err());
        assert!(make_info_set_key(0, "ip", 160, None, "", &f2c, Some(1)).is_err());
    }

    #[test]
    fn chars_match_python_table() {
        for (a, c) in [
            ("check", 'k'),
            ("call", 'c'),
            ("fold", 'f'),
            ("bet_small", 's'),
            ("raise_small", 's'),
            ("bet_medium", 'm'),
            ("raise_medium", 'm'),
            ("bet_large", 'l'),
            ("raise_large", 'l'),
            ("bet_xlarge", 'x'),
            ("bet_overbet", 'o'),
            ("raise_overbet", 'o'),
            ("bet_overbet2", '2'),
            ("raise_overbet2", '2'),
            ("allin", 'a'),
        ] {
            assert_eq!(
                action_char(Action::parse(a).unwrap()).unwrap() as char,
                c,
                "{a}"
            );
        }
        assert!(action_char(Action::parse("raise_xlarge").unwrap()).is_err());
        assert!(action_char(Action::parse("bet_custom_10").unwrap()).is_err());
    }
}
