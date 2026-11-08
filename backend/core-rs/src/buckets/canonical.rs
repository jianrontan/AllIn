//! Suit-isomorphic canonicalisation: port of `canonical.canonical_key`,
//! `canonical.canonical_board_perm` and `postflop_features.encode_situation`.
//!
//! Python represents a card as `(rank_ord, suit_letter)` and minimises the pair
//! `(sorted hole reps, sorted board reps)` over the 24 suit relabelings, with
//! suits compared as STRINGS (`'C' < 'D' < 'H' < 'S'`), not by their deck
//! index (`H D C S`). Here a rep is the 6-bit value `rank * 4 + letter_order`,
//! so tuple order == numeric order, and a whole (hole, board) rep packs into a
//! u64 whose numeric order is the lexicographic order Python compares.

/// Deck suit index (H, D, C, S) -> letter order (C=0, D=1, H=2, S=3).
const LETTER: [u8; 4] = [2, 1, 0, 3];
/// Letter order -> deck suit index.
const SUIT_OF_LETTER: [u8; 4] = [2, 1, 0, 3];

/// `itertools.permutations(['H','D','C','S'])` as index maps, in the same
/// order: `PERMS[p][suit] = relabeled suit`.
pub const PERMS: [[u8; 4]; 24] = {
    let mut out = [[0u8; 4]; 24];
    let mut n = 0;
    let mut a = 0;
    while a < 4 {
        let mut b = 0;
        while b < 4 {
            let mut c = 0;
            while c < 4 {
                let mut d = 0;
                while d < 4 {
                    if a != b && a != c && a != d && b != c && b != d && c != d {
                        out[n] = [a as u8, b as u8, c as u8, d as u8];
                        n += 1;
                    }
                    d += 1;
                }
                c += 1;
            }
            b += 1;
        }
        a += 1;
    }
    out
};

#[inline(always)]
fn rep(card: u8, perm: &[u8; 4]) -> u8 {
    ((card >> 2) % 13) * 4 + LETTER[perm[(card & 3) as usize] as usize]
}

#[inline(always)]
fn sort_small(v: &mut [u8]) {
    for i in 1..v.len() {
        let x = v[i];
        let mut j = i;
        while j > 0 && v[j - 1] > x {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = x;
    }
}

#[inline(always)]
fn pack(v: &[u8]) -> u64 {
    v.iter().fold(0u64, |acc, &x| (acc << 6) | x as u64)
}

/// Canonical reps (hole sorted, then board sorted) for <= 2 hole + <= 5 board
/// VALID cards; returns the rep buffer and its length.
fn canonical_reps(hole: &[u8], board: &[u8]) -> ([u8; 7], usize) {
    let (nh, nb) = (hole.len().min(2), board.len().min(5));
    let n = nh + nb;
    let mut best = [0u8; 7];
    let mut best_key = u64::MAX;
    for perm in &PERMS {
        let mut r = [0u8; 7];
        for i in 0..nh {
            r[i] = rep(hole[i], perm);
        }
        for i in 0..nb {
            r[nh + i] = rep(board[i], perm);
        }
        sort_small(&mut r[..nh]);
        sort_small(&mut r[nh..n]);
        let k = pack(&r[..n]);
        if k < best_key {
            best_key = k;
            best = r;
        }
    }
    (best, n)
}

/// `encode_situation(canonical_key(hole, board))` for VALID card ids.
pub fn situation_id(hole: &[u8], board: &[u8]) -> i64 {
    let (r, n) = canonical_reps(hole, board);
    let mut sid: i64 = 0;
    for &x in &r[..n] {
        let id = (x >> 2) as i64 * 4 + SUIT_OF_LETTER[(x & 3) as usize] as i64;
        sid = sid * 52 + id;
    }
    sid
}

/// `canonical_board_perm(board)`: the canonical board (as a packed key usable
/// for caching, plus its card ids) and the suit relabeling that produced it
/// (the FIRST permutation reaching the minimum, as in Python).
pub struct CanonBoard {
    pub key: u64,
    pub cards: [u8; 5],
    pub n: usize,
    pub perm: [u8; 4],
}

pub fn canonical_board(board: &[u8]) -> CanonBoard {
    let nb = board.len().min(5);
    let mut best = [0u8; 5];
    let mut best_key = u64::MAX;
    let mut best_perm = PERMS[0];
    for perm in &PERMS {
        let mut r = [0u8; 5];
        for i in 0..nb {
            r[i] = rep(board[i], perm);
        }
        sort_small(&mut r[..nb]);
        let k = pack(&r[..nb]);
        if k < best_key {
            best_key = k;
            best = r;
            best_perm = *perm;
        }
    }
    let mut cards = [0u8; 5];
    for i in 0..nb {
        cards[i] = (best[i] >> 2) * 4 + SUIT_OF_LETTER[(best[i] & 3) as usize];
    }
    CanonBoard {
        key: best_key | ((nb as u64) << 60),
        cards,
        n: nb,
        perm: best_perm,
    }
}

/// Apply a suit relabeling to a card id.
#[inline(always)]
pub fn relabel(card: u8, perm: &[u8; 4]) -> u8 {
    (card & !3) | perm[(card & 3) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perms_match_itertools_order() {
        assert_eq!(PERMS[0], [0, 1, 2, 3]);
        assert_eq!(PERMS[1], [0, 1, 3, 2]);
        assert_eq!(PERMS[23], [3, 2, 1, 0]);
    }

    #[test]
    fn invariant_under_suit_relabeling() {
        let hole = [48u8, 45]; // AH, KD
        let board = [40u8, 37, 2]; // QH, JD, 2C
        let sid = situation_id(&hole, &board);
        for p in &PERMS {
            let h2 = hole.map(|c| relabel(c, p));
            let b2 = board.map(|c| relabel(c, p));
            assert_eq!(situation_id(&h2, &b2), sid);
            assert_eq!(canonical_board(&b2).key, canonical_board(&board).key);
        }
    }

    #[test]
    fn canonical_board_perm_maps_board_onto_canon() {
        let board = [48u8, 45, 41, 3, 6];
        let cb = canonical_board(&board);
        let mut mapped: Vec<u8> = board.iter().map(|&c| relabel(c, &cb.perm)).collect();
        let mut canon = cb.cards[..cb.n].to_vec();
        mapped.sort_unstable();
        canon.sort_unstable();
        assert_eq!(mapped, canon);
    }
}
