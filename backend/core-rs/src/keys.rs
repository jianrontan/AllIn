//! Info-set keys. Port of `cfr/keys.py` (`make_info_set_key`, `action_char`).
//!
//! Preflop : `pf_{fine}_{position}_{pattern}`
//! Postflop: `pf_{coarse}_{strength}_{position}_{street}_{pattern}` where the
//! fine preflop bucket is collapsed to its coarse class (imperfect recall).

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
) -> Result<String, String> {
    if street == 0 {
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
    Ok(format!("pf_{coarse}_{s}_{position}_{name}_{pattern}"))
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
            make_info_set_key(0, "ip", 160, None, "", &f2c).unwrap(),
            "pf_160_ip_"
        );
        assert_eq!(
            make_info_set_key(2, "ip", 160, Some(5), "m", &f2c).unwrap(),
            "pf_9_5_ip_turn_m"
        );
        assert!(make_info_set_key(1, "oop", 169, Some(1), "", &f2c).is_err());
        assert!(make_info_set_key(4, "oop", 1, Some(1), "", &f2c).is_err());
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
