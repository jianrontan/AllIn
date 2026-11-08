//! Python-compatible JSON text for the blueprint DB columns, so a Rust-written
//! blueprint is byte-identical to one written by `BlueprintDB` (`json.dumps`
//! with default separators, insertion order, `repr(float)` numbers).

/// Python `repr(x)` for a float (`json.dumps` uses it, mapping non-finite values
/// to `Infinity` / `-Infinity` / `NaN`).
pub fn py_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Rust's `{:e}` is the shortest round-trip digit string, like Python's repr.
    let sci = format!("{:e}", x.abs());
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let body = if exp >= 0 {
            let e = exp as usize;
            if digits.len() <= e + 1 {
                format!("{digits}{}.0", "0".repeat(e + 1 - digits.len()))
            } else {
                format!("{}.{}", &digits[..e + 1], &digits[e + 1..])
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let es = if exp < 0 { '-' } else { '+' };
        format!("{sign}{m}e{es}{:02}", exp.abs())
    }
}

/// `json.dumps({name: value, ...})` in the given order.
pub fn dumps_map<'a>(items: impl Iterator<Item = (&'a str, f64)>) -> String {
    let body: Vec<String> = items
        .map(|(k, v)| format!("\"{k}\": {}", py_repr(v)))
        .collect();
    format!("{{{}}}", body.join(", "))
}

/// `json.dumps([name, ...])`.
pub fn dumps_list<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let body: Vec<String> = items.map(|k| format!("\"{k}\"")).collect();
    format!("[{}]", body.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_matches_python() {
        for (x, s) in [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (-2.25, "-2.25"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (123456789012345.6, "123456789012345.6"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.2345e-7, "1.2345e-07"),
            (0.1 + 0.2, "0.30000000000000004"),
            (2.0f64.powi(-1074), "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (100.0, "100.0"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(py_repr(x), s, "{x:e}");
        }
    }

    #[test]
    fn dumps_shape() {
        assert_eq!(
            dumps_map([("call", 0.5), ("fold", 2.0)].into_iter()),
            "{\"call\": 0.5, \"fold\": 2.0}"
        );
        assert_eq!(dumps_map(std::iter::empty()), "{}");
        assert_eq!(
            dumps_list(["fold", "call"].into_iter()),
            "[\"fold\", \"call\"]"
        );
    }
}
