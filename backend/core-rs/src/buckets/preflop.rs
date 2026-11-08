//! Preflop buckets: port of `card_abstractions._quantile_buckets` /
//! `FINE_TO_COARSE` / `CardAbstraction.preflop_bucket`.
//!
//! Both maps are DERIVED (as in Python) from the committed equity table
//! (`preflop_equity.rs`, generated from `_PREFLOP_EQUITY` in dict order):
//! a stable ascending sort by equity (ties keep dict order), then
//! `bucket = min(int(i * n / 169), n - 1)` with Python's float division.

use std::sync::OnceLock;

use super::preflop_equity::PREFLOP_EQUITY;

pub const NUM_PREFLOP_BUCKETS: usize = 169;
pub const NUM_PREFLOP_COARSE: usize = 10;

const RANKS: &[u8; 13] = b"23456789TJQKA";

struct Maps {
    /// `[hi][lo][suited]` -> fine bucket (hi >= lo rank index; pairs use suited=0).
    fine: [[[u8; 2]; 13]; 13],
    fine_to_coarse: [u8; NUM_PREFLOP_BUCKETS],
}

fn quantile(n_buckets: usize) -> Vec<(&'static str, usize)> {
    let mut ranked: Vec<(&'static str, f64)> = PREFLOP_EQUITY.to_vec();
    // Stable sort (Python `sorted(..., key=eq)`); equities are finite literals.
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let total = ranked.len();
    ranked
        .iter()
        .enumerate()
        .map(|(i, (h, _))| {
            let q = ((i * n_buckets) as f64 / total as f64) as usize; // int(i*n/total)
            (*h, q.min(n_buckets - 1))
        })
        .collect()
}

fn rank_idx(c: u8) -> usize {
    RANKS
        .iter()
        .position(|&r| r == c)
        .expect("generated table uses valid ranks")
}

fn build() -> Maps {
    let fine = quantile(NUM_PREFLOP_BUCKETS);
    let coarse: std::collections::HashMap<&str, usize> =
        quantile(NUM_PREFLOP_COARSE).into_iter().collect();
    let mut m = Maps {
        fine: [[[u8::MAX; 2]; 13]; 13],
        fine_to_coarse: [u8::MAX; NUM_PREFLOP_BUCKETS],
    };
    for (hand, fb) in fine {
        let b = hand.as_bytes();
        let (hi, lo) = (rank_idx(b[0]), rank_idx(b[1]));
        let suited = usize::from(b.len() == 3 && b[2] == b's');
        m.fine[hi][lo][suited] = fb as u8;
        let c = coarse[hand] as u8;
        // Python asserts each fine bucket maps to ONE coarse class (lossless:
        // one hand per fine bucket, so this always holds).
        assert!(m.fine_to_coarse[fb] == u8::MAX || m.fine_to_coarse[fb] == c);
        m.fine_to_coarse[fb] = c;
    }
    m
}

fn maps() -> &'static Maps {
    static M: OnceLock<Maps> = OnceLock::new();
    M.get_or_init(build)
}

/// `card_abstractions.FINE_TO_COARSE`: fine bucket id (0..169) -> coarse class (0..10).
pub fn fine_to_coarse() -> &'static [u8; NUM_PREFLOP_BUCKETS] {
    &maps().fine_to_coarse
}

/// Fine preflop bucket (0 = weakest .. 168 = strongest) of two VALID, distinct
/// card ids (`rank_idx * 4 + suit_idx`).
pub fn preflop_fine(a: u8, b: u8) -> u8 {
    let (ra, rb) = (((a >> 2) % 13) as usize, ((b >> 2) % 13) as usize);
    let (hi, lo) = if ra >= rb { (ra, rb) } else { (rb, ra) };
    let suited = usize::from(hi != lo && (a & 3) == (b & 3));
    maps().fine[hi][lo][suited]
}

/// `(fine, coarse)` preflop buckets of two VALID card ids.
pub fn preflop_buckets(a: u8, b: u8) -> (u8, u8) {
    let f = preflop_fine(a, b);
    (f, fine_to_coarse()[f as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_complete_and_lossless() {
        let m = maps();
        let mut seen = [false; NUM_PREFLOP_BUCKETS];
        for hi in 0..13 {
            for lo in 0..=hi {
                for s in 0..2 {
                    if hi == lo && s == 1 {
                        continue;
                    }
                    let f = m.fine[hi][lo][s];
                    assert!((f as usize) < NUM_PREFLOP_BUCKETS);
                    assert!(!seen[f as usize], "fine bucket {f} reused");
                    seen[f as usize] = true;
                }
            }
        }
        assert!(
            m.fine_to_coarse
                .iter()
                .all(|&c| (c as usize) < NUM_PREFLOP_COARSE)
        );
        // AA strongest, 32o weakest
        assert_eq!(preflop_fine(48, 49), 168);
        assert_eq!(preflop_fine(0, 5), 0);
        assert_eq!(fine_to_coarse()[168], 9);
        assert_eq!(fine_to_coarse()[0], 0);
    }
}
