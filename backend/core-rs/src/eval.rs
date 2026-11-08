//! 5/6/7-card hand evaluator (Phase 2, component `eval_rank7`).
//!
//! Port of `postflop_features.rank7` / `HandEvaluator.get_raw_hand_value`, which
//! call phevaluator. The value is the Cactus-Kev equivalence class of the best
//! 5-card hand: 1 = royal flush ... 7462 = 7-5-4-3-2 offsuit; LOWER = STRONGER.
//!
//! | class | range |
//! |---|---|
//! | straight flush | 1..=10 |
//! | four of a kind | 11..=166 |
//! | full house | 167..=322 |
//! | flush | 323..=1599 |
//! | straight | 1600..=1609 |
//! | three of a kind | 1610..=2467 |
//! | two pair | 2468..=3325 |
//! | one pair | 3326..=6185 |
//! | high card | 6186..=7462 |
//!
//! Within a class, hands are ordered by their relevant ranks, high first
//! (lexicographic). Card id = `rank_idx * 4 + suit_idx` (ranks `23456789TJQKA`,
//! suits `HDCS`), the index into the Python `DECK`.
//!
//! Method: per-suit 13-bit rank masks; multiplicity masks (>=2, >=3, 4) by
//! bitwise AND/OR; the within-class index is a colex rank of the kicker mask
//! (the numeric order of k-subsets as bitmasks is the colex order, which is the
//! lexicographic high-first order poker uses). All tables are `const` (built at
//! compile time), so the evaluator is allocation-free and trivially `Sync`.

/// Number of distinct hand values (Cactus-Kev classes).
pub const NUM_CLASSES: i32 = 7462;

const fn binom(n: u32, k: u32) -> u32 {
    if k > n {
        return 0;
    }
    let mut r: u32 = 1;
    let mut i = 0;
    while i < k {
        r = r * (n - i) / (i + 1);
        i += 1;
    }
    r
}

/// `BINOM[n][k]` for n <= 13, k <= 5.
const BINOM: [[u32; 6]; 14] = {
    let mut t = [[0u32; 6]; 14];
    let mut n = 0;
    while n < 14 {
        let mut k = 0;
        while k < 6 {
            t[n][k] = binom(n as u32, k as u32);
            k += 1;
        }
        n += 1;
    }
    t
};

/// Highest straight in a 13-bit rank mask: 1 = A-high ... 9 = 6-high,
/// 10 = 5-high (wheel), 0 = none.
const STRAIGHT: [u8; 8192] = {
    let mut t = [0u8; 8192];
    let mut m = 0usize;
    while m < 8192 {
        let mut hi = 12i32; // top card of the window, A=12 down to 6=4
        let mut found = 0u8;
        while hi >= 4 {
            let w = 0x1Fusize << (hi - 4);
            if m & w == w {
                found = (13 - hi) as u8; // A-high (12) -> 1, 6-high (4) -> 9
                break;
            }
            hi -= 1;
        }
        if found == 0 && (m & 0x100F) == 0x100F {
            found = 10; // A-2-3-4-5
        }
        t[m] = found;
        m += 1;
    }
    t
};

/// Descending index (0 = strongest) of each 5-bit non-straight rank mask among
/// all 1277 such masks; used for both flushes and high-card hands.
const FIVE_IDX: [u16; 8192] = {
    let mut t = [0u16; 8192];
    let mut idx = 0u16;
    let mut m = 8191i32;
    while m >= 0 {
        let mu = m as usize;
        if (mu as u32).count_ones() == 5 && STRAIGHT[mu] == 0 {
            t[mu] = idx;
            idx += 1;
        }
        m -= 1;
    }
    t
};

/// Highest set bit (0..=12); 0 for an empty mask (never reached for valid hands).
#[inline(always)]
fn hi_bit(m: u32) -> u32 {
    31u32.saturating_sub(m.leading_zeros())
}

/// Keep only the `k` highest set bits of `m`.
#[inline(always)]
fn top_bits(mut m: u32, k: u32) -> u32 {
    let mut n = m.count_ones();
    while n > k {
        m &= m - 1; // drop the lowest set bit (m != 0 because n > k >= 0)
        n -= 1;
    }
    m
}

/// Remove bit position `b` from `m`, shifting the higher bits down by one.
#[inline(always)]
fn remove_bit(m: u32, b: u32) -> u32 {
    let low = m & ((1u32 << b) - 1);
    ((m >> (b + 1)) << b) | low
}

/// Colex rank of a k-subset given as a bitmask (= its rank in numeric order).
#[inline(always)]
fn colex(mut m: u32) -> u32 {
    let mut r = 0;
    let mut i = 1usize;
    while m != 0 && i <= 5 {
        let p = m.trailing_zeros() as usize;
        r += BINOM[p.min(13)][i];
        m &= m - 1;
        i += 1;
    }
    r
}

/// Descending index (0 = strongest) of a k-bit mask over n positions.
#[inline(always)]
fn desc_idx(m: u32, n: usize, k: usize) -> u32 {
    BINOM[n][k].wrapping_sub(1).wrapping_sub(colex(m))
}

/// Value of the best 5-card hand from per-suit rank masks (13 bits each).
#[inline(always)]
fn rank_masks(s: [u32; 4]) -> i32 {
    let any = s[0] | s[1] | s[2] | s[3];
    let two = (s[0] & s[1])
        | (s[0] & s[2])
        | (s[0] & s[3])
        | (s[1] & s[2])
        | (s[1] & s[3])
        | (s[2] & s[3]);
    let three =
        (s[0] & s[1] & s[2]) | (s[0] & s[1] & s[3]) | (s[0] & s[2] & s[3]) | (s[1] & s[2] & s[3]);
    let four = s[0] & s[1] & s[2] & s[3];

    // Flush: at most one suit can hold >= 5 of <= 7 cards, and with a flush
    // present no quads / full house is possible in 7 cards, so it decides.
    for &sm in &s {
        if sm.count_ones() >= 5 {
            let st = STRAIGHT[(sm & 0x1FFF) as usize];
            if st != 0 {
                return st as i32;
            }
            return 323 + FIVE_IDX[top_bits(sm, 5) as usize] as i32;
        }
    }
    if four != 0 {
        let q = hi_bit(four);
        let k = hi_bit(any & !(1 << q));
        let kp = remove_bit(1 << k, q);
        return 11 + (12 - q as i32) * 12 + desc_idx(kp, 12, 1) as i32;
    }
    if three != 0 {
        let t = hi_bit(three);
        let pairs = two & !(1 << t);
        if pairs != 0 {
            let p = hi_bit(pairs);
            let pp = remove_bit(1 << p, t);
            return 167 + (12 - t as i32) * 12 + desc_idx(pp, 12, 1) as i32;
        }
    }
    let st = STRAIGHT[(any & 0x1FFF) as usize];
    if st != 0 {
        return 1599 + st as i32;
    }
    if three != 0 {
        let t = hi_bit(three);
        let kick = remove_bit(top_bits(any & !(1 << t), 2), t);
        return 1610 + (12 - t as i32) * 66 + desc_idx(kick, 12, 2) as i32;
    }
    if two.count_ones() >= 2 {
        let pm = top_bits(two, 2);
        let p1 = hi_bit(pm);
        let p2 = pm.trailing_zeros().min(12);
        let k = hi_bit(any & !pm);
        let kp = remove_bit(remove_bit(1 << k, p1), p2);
        return 2468 + desc_idx(pm, 13, 2) as i32 * 11 + desc_idx(kp, 11, 1) as i32;
    }
    if two != 0 {
        let p = hi_bit(two);
        let kick = remove_bit(top_bits(any & !(1 << p), 3), p);
        return 3326 + (12 - p as i32) * 220 + desc_idx(kick, 12, 3) as i32;
    }
    6186 + FIVE_IDX[top_bits(any, 5) as usize] as i32
}

/// Per-suit rank masks of a set of VALID card ids (< 52).
#[inline(always)]
pub fn suit_masks(cards: &[u8]) -> [u32; 4] {
    let mut s = [0u32; 4];
    for &c in cards {
        s[(c & 3) as usize] |= 1 << ((c >> 2) % 13);
    }
    s
}

/// Value of a hand given precomputed suit masks of some cards plus two more
/// cards (the river-bucketing hot path: board masks once, then every hole).
#[inline(always)]
pub fn rank_with(base: [u32; 4], a: u8, b: u8) -> i32 {
    let mut s = base;
    s[(a & 3) as usize] |= 1 << ((a >> 2) % 13);
    s[(b & 3) as usize] |= 1 << ((b >> 2) % 13);
    rank_masks(s)
}

/// Hand value of 5, 6 or 7 cards (same integer as Python `rank7`; lower =
/// stronger). Thread-safe and allocation-free.
///
/// The caller is responsible for validity (see [`validate_row`]); invalid
/// input never panics: a wrong length or a card id >= 52 returns 0 (not a
/// valid hand value), duplicates give an unspecified value.
#[inline]
pub fn rank(cards: &[u8]) -> i32 {
    if !(5..=7).contains(&cards.len()) || cards.iter().any(|&c| c >= 52) {
        return 0;
    }
    rank_masks(suit_masks(cards))
}

/// Check one row of card ids: every id < 52 and no duplicates.
pub fn validate_row(cards: &[u8]) -> Result<(), String> {
    let mut seen = 0u64;
    for &c in cards {
        if c >= 52 {
            return Err(format!("card id {c} out of range 0..52"));
        }
        let bit = 1u64 << c;
        if seen & bit != 0 {
            return Err(format!("duplicate card id {c} in row {cards:?}"));
        }
        seen |= bit;
    }
    Ok(())
}

/// Batch evaluation of a row-major `[n, k]` card array (k = 5, 6, 7).
/// Validates everything first; returns an error instead of panicking.
pub fn rank_batch(cards: &[u8], k: usize) -> Result<Vec<i32>, String> {
    if !(5..=7).contains(&k) {
        return Err(format!(
            "rank_batch: rows must have 5, 6 or 7 cards, got {k}"
        ));
    }
    if !cards.len().is_multiple_of(k) {
        return Err(format!(
            "rank_batch: {} ids is not a multiple of k={k}",
            cards.len()
        ));
    }
    for (i, row) in cards.chunks_exact(k).enumerate() {
        validate_row(row).map_err(|e| format!("row {i}: {e}"))?;
    }
    Ok(cards
        .chunks_exact(k)
        .map(|r| rank_masks(suit_masks(r)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Independent brute-force reference: classify a 5-card hand by rank counts
    /// into (category, tiebreak ranks high-first), enumerate every distinct
    /// class, sort strongest-first and number them 1..=7462.
    type Key = (u8, [u8; 5]);

    fn classify5(ranks: [u8; 5], flush: bool) -> Key {
        let mut cnt = [0u8; 13];
        for &r in &ranks {
            cnt[r as usize] += 1;
        }
        // groups sorted by (count desc, rank desc)
        let mut groups: Vec<(u8, u8)> = (0..13u8)
            .rev()
            .filter(|&r| cnt[r as usize] > 0)
            .map(|r| (cnt[r as usize], r))
            .collect();
        groups.sort_by(|a, b| b.cmp(a));
        let mut tb = [0u8; 5];
        for (i, g) in groups.iter().enumerate() {
            tb[i] = g.1;
        }
        let distinct = groups.len() == 5;
        let straight_top = if distinct {
            let mut s: Vec<u8> = ranks.to_vec();
            s.sort_unstable_by(|a, b| b.cmp(a));
            if s[0] - s[4] == 4 {
                Some(s[0])
            } else if s == [12, 3, 2, 1, 0] {
                Some(3)
            } else {
                None
            }
        } else {
            None
        };
        // category: higher = stronger
        let cat = match (straight_top, flush, groups[0].0, groups.get(1).map(|g| g.0)) {
            (Some(_), true, _, _) => 8,
            (_, _, 4, _) => 7,
            (_, _, 3, Some(2)) => 6,
            (None, true, _, _) => 5,
            (Some(_), false, _, _) => 4,
            (_, _, 3, _) => 3,
            (_, _, 2, Some(2)) => 2,
            (_, _, 2, _) => 1,
            _ => 0,
        };
        if let Some(t) = straight_top {
            tb = [t, 0, 0, 0, 0];
        }
        (cat, tb)
    }

    fn class_table() -> HashMap<Key, i32> {
        let mut keys = std::collections::BTreeSet::new();
        for a in 0..13u8 {
            for b in a..13 {
                for c in b..13 {
                    for d in c..13 {
                        for e in d..13 {
                            let r = [a, b, c, d, e];
                            let mut cnt = [0; 13];
                            for &x in &r {
                                cnt[x as usize] += 1;
                            }
                            if cnt.iter().any(|&x| x > 4) {
                                continue;
                            }
                            keys.insert(classify5(r, false));
                            if cnt.iter().all(|&x| x <= 1) {
                                keys.insert(classify5(r, true));
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(keys.len(), 7462);
        keys.into_iter()
            .rev()
            .enumerate()
            .map(|(i, k)| (k, i as i32 + 1))
            .collect()
    }

    fn brute(cards: &[u8], tab: &HashMap<Key, i32>) -> i32 {
        let n = cards.len();
        let mut best = i32::MAX;
        let idx: Vec<usize> = (0..n).collect();
        // all 5-subsets
        fn rec(
            start: usize,
            pick: &mut Vec<usize>,
            idx: &[usize],
            cards: &[u8],
            tab: &HashMap<Key, i32>,
            best: &mut i32,
        ) {
            if pick.len() == 5 {
                let r: [u8; 5] = std::array::from_fn(|i| cards[pick[i]] >> 2);
                let s0 = cards[pick[0]] & 3;
                let flush = pick.iter().all(|&p| cards[p] & 3 == s0);
                *best = (*best).min(tab[&classify5(r, flush)]);
                return;
            }
            for i in start..idx.len() {
                pick.push(i);
                rec(i + 1, pick, idx, cards, tab, best);
                pick.pop();
            }
        }
        rec(0, &mut Vec::new(), &idx, cards, tab, &mut best);
        best
    }

    #[test]
    fn known_values() {
        // royal flush (hearts): TH JH QH KH AH -> ids rank*4+0
        assert_eq!(rank(&[32, 36, 40, 44, 48]), 1);
        // 7-5-4-3-2 offsuit -> 7462
        assert_eq!(rank(&[20, 13, 8, 5, 2]), 7462);
        // wheel straight flush -> 10
        assert_eq!(rank(&[48, 0, 4, 8, 12]), 10);
        // invalid input never panics
        assert_eq!(rank(&[52, 1, 2, 3, 4]), 0);
        assert_eq!(rank(&[1, 2, 3]), 0);
        let _ = rank(&[1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn exhaustive_5_card_vs_brute_force() {
        let tab = class_table();
        let mut seen = vec![false; 7463];
        let mut n = 0u64;
        for a in 0..52u8 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    for d in c + 1..52 {
                        for e in d + 1..52 {
                            let h = [a, b, c, d, e];
                            let r5: [u8; 5] = h.map(|x| x >> 2);
                            let fl = h.iter().all(|&x| x & 3 == a & 3);
                            let want = tab[&classify5(r5, fl)];
                            assert_eq!(rank(&h), want, "{h:?}");
                            seen[want as usize] = true;
                            n += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(n, 2_598_960);
        assert!(seen[1..].iter().all(|&s| s), "every class reached");
    }

    /// splitmix64: deterministic, dependency-free test RNG.
    fn next(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn deal(s: &mut u64, k: usize) -> Vec<u8> {
        let mut deck: Vec<u8> = (0..52).collect();
        for i in 0..k {
            let j = i + (next(s) % (52 - i) as u64) as usize;
            deck.swap(i, j);
        }
        deck.truncate(k);
        deck
    }

    #[test]
    fn random_6_and_7_card_vs_brute_force() {
        let tab = class_table();
        let n = if cfg!(debug_assertions) {
            20_000
        } else {
            500_000
        };
        let mut s = 12345u64;
        for k in [6usize, 7] {
            for _ in 0..n {
                let h = deal(&mut s, k);
                assert_eq!(rank(&h), brute(&h, &tab), "{h:?}");
            }
        }
    }

    #[test]
    fn batch_validation() {
        assert!(rank_batch(&[0, 1, 2, 3, 4], 5).is_ok());
        assert!(rank_batch(&[0, 1, 2, 3, 52], 5).is_err());
        assert!(rank_batch(&[0, 1, 2, 3, 3], 5).is_err());
        assert!(rank_batch(&[0, 1, 2, 3], 4).is_err());
        assert!(rank_batch(&[0, 1, 2, 3, 4, 5], 5).is_err());
        let b = rank_batch(&[0, 4, 8, 12, 48, 1, 2], 7).unwrap();
        assert_eq!(b, vec![rank(&[0, 4, 8, 12, 48, 1, 2])]);
    }
}
