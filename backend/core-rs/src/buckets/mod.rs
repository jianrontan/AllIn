//! Card abstraction (Phase 2, component `buckets`): port of
//! `card_abstractions.py` (preflop fine/coarse), `canonical.py`,
//! `postflop_features.py` and `postflop_v2.py`.
//!
//! * preflop: [`preflop`] (fine 0..169 lossless, coarse 0..10).
//! * flop / turn: canonical situation id -> binary search in the baked table
//!   `postflop_table_{street}.npz`, whose centroid stamp is verified on load
//!   exactly as `PostflopV2._verify_stamp` does (mismatch = hard error).
//! * river: equity vs a uniform range on the complete board (`board_winrates`,
//!   computed once per CANONICAL board and cached in a bounded thread-safe LRU,
//!   like `_RIVER_BOARD_CACHE`), then the spike histogram's nearest river
//!   centroid by EMD.
//!
//! Deviation (documented, not silent): when a flop/turn table is missing or a
//! situation is absent from it, Python falls back to a slow Monte Carlo
//! equity-distribution feature drawn from its own RNG (not reproducible across
//! languages). Rust refuses with an error instead; the shipped tables are
//! complete (every canonical flop/turn situation).

pub mod cache;
pub mod canonical;
pub mod npz;
pub mod preflop;
mod preflop_equity;
pub mod sha1;

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub use preflop::{fine_to_coarse, preflop_buckets, preflop_fine};

use crate::eval;
use crate::river::showdown::np_sum;
use cache::BoardCache;
use canonical::{canonical_board, relabel, situation_id};
use npz::{NpyData, Npz};

/// Python's `_RIVER_BOARD_CACHE_CAP` default.
pub const DEFAULT_RIVER_CACHE_BOARDS: usize = 100_000;
/// Hands on a complete board: C(47, 2).
pub const RIVER_HANDS: usize = 1081;
const RIVER_LIVE: usize = 47;
const RIVER_2NM1: usize = 2 * RIVER_LIVE - 1;
/// `_RIVER_EQ_SCALE`: 2 * 990 * equity is an exact integer in [0, 1980].
const RIVER_EQ_SCALE: f64 = 1980.0;

const STREETS: [&str; 3] = ["flop", "turn", "river"];

/// Street index (0 flop, 1 turn, 2 river) for a board length.
fn street_of(nb: usize) -> Result<usize, String> {
    match nb {
        3 => Ok(0),
        4 => Ok(1),
        5 => Ok(2),
        _ => Err(format!("board must have 3, 4 or 5 cards, got {nb}")),
    }
}

/// One street's centroids (`postflop_centroids_<street>.npz`).
pub struct Centroids {
    pub k: usize,
    pub bins: usize,
    /// row-major `[k][bins]`, float64 (as the hash and EMD use them)
    pub data: Vec<f64>,
}

impl Centroids {
    fn load(dir: &Path, street: &str) -> Result<Centroids, String> {
        let path = dir.join(format!("postflop_centroids_{street}.npz"));
        let mut z = Npz::open(&path)?;
        let c = z.read("centroids")?;
        let bins = z.read("bins")?.scalar_i64()?;
        let ctx = |m: &str| format!("{}: {m}", path.display());
        if c.shape.len() != 2 {
            return Err(ctx("centroids must be 2-D"));
        }
        let (k, nb) = (c.shape[0], c.shape[1]);
        if k == 0 || k > 255 || nb == 0 || nb > 10_000 || bins != nb as i64 {
            return Err(ctx(&format!(
                "bad centroid shape {:?} / bins {bins}",
                c.shape
            )));
        }
        let data = match c.data {
            NpyData::F64(v) => v,
            NpyData::I64(v) => v.into_iter().map(|x| x as f64).collect(),
            NpyData::U8(v) => v.into_iter().map(|x| x as f64).collect(),
            NpyData::Str(_) => return Err(ctx("centroids are strings")),
        };
        if data.iter().any(|x| !x.is_finite()) {
            return Err(ctx("non-finite centroid value"));
        }
        Ok(Centroids { k, bins: nb, data })
    }

    /// `postflop_features.centroid_hash(centroids, bins)`: sha1 over the
    /// float64 C-order bytes, then `b'|'`, then `str(int(bins))`.
    pub fn hash(&self) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(self.data.len() * 8 + 8);
        for x in &self.data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        bytes.push(b'|');
        bytes.extend_from_slice(self.bins.to_string().as_bytes());
        sha1::sha1_hex(&bytes)
    }

    /// `assign(feat, centroids)` = `argmin_k emd(feat, c_k)` with
    /// `emd(a, b) = np.abs(np.cumsum(a) - np.cumsum(b)).sum()`. Replays numpy's
    /// order: sequential cumsums, element-wise |diff|, pairwise sum; first
    /// minimum wins (np.argmin).
    pub fn assign(&self, feat: &[f64]) -> usize {
        let n = self.bins;
        let mut cf = vec![0.0; n];
        let mut acc = 0.0;
        for i in 0..n {
            acc += feat[i];
            cf[i] = acc;
        }
        let mut best = (0usize, f64::INFINITY);
        let mut d = vec![0.0; n];
        for k in 0..self.k {
            let row = &self.data[k * n..(k + 1) * n];
            let mut acc = 0.0;
            for i in 0..n {
                acc += row[i];
                d[i] = (cf[i] - acc).abs();
            }
            let e = np_sum(&d);
            if e < best.1 || (k == 0 && e.is_nan()) {
                best = (k, e);
            }
        }
        best.0
    }
}

/// A baked flop/turn (or optional river) table.
struct Table {
    ids: Vec<i64>,
    buckets: Vec<u8>,
}

impl Table {
    fn lookup(&self, sid: i64) -> Option<u8> {
        self.ids.binary_search(&sid).ok().map(|i| self.buckets[i])
    }
}

/// Per-board river entry: the exact equity numerator (`2*win + tie`, in
/// [0, 1980]) of every hand, in combination (i < j over live cards) order.
pub type RiverEntry = [u16; RIVER_HANDS];

/// Postflop + preflop bucketing. `Send + Sync`: share one instance (e.g. in an
/// `Arc`) across threads so they share the table memory and the river cache.
pub struct Buckets {
    dir: PathBuf,
    centroids: [Centroids; 3],
    tables: [OnceLock<Result<Option<Arc<Table>>, String>>; 3],
    /// river: equity numerator (0..=1980) -> bucket
    river_by_num: Vec<u8>,
    river_cache: BoardCache<RiverEntry>,
}

impl Buckets {
    /// `river_cache_boards`: None = `ALLIN_RIVER_CACHE_BOARDS` env var, else
    /// 100k (Python's default). 0 disables the cache.
    pub fn new(dir: &Path, river_cache_boards: Option<usize>) -> Result<Buckets, String> {
        let centroids = [
            Centroids::load(dir, "flop")?,
            Centroids::load(dir, "turn")?,
            Centroids::load(dir, "river")?,
        ];
        let cap = match river_cache_boards {
            Some(c) => c,
            None => match std::env::var("ALLIN_RIVER_CACHE_BOARDS") {
                Ok(s) => s
                    .trim()
                    .parse()
                    .map_err(|_| format!("ALLIN_RIVER_CACHE_BOARDS={s:?} is not an integer"))?,
                Err(_) => DEFAULT_RIVER_CACHE_BOARDS,
            },
        };
        // River: hist = one-hot at min(int(eq * bins), bins - 1) with
        // eq = num / 1980.0 (Python float ops), so the bucket is a function of
        // the integer numerator only -> precompute all 1981.
        let rc = &centroids[2];
        let spike: Vec<u8> = (0..rc.bins)
            .map(|b| {
                let mut h = vec![0.0; rc.bins];
                h[b] = 1.0;
                rc.assign(&h) as u8
            })
            .collect();
        let river_by_num = (0..=1980u32)
            .map(|num| {
                let eq = num as f64 / RIVER_EQ_SCALE;
                let b = ((eq * rc.bins as f64) as usize).min(rc.bins - 1);
                spike[b]
            })
            .collect();
        Ok(Buckets {
            dir: dir.to_path_buf(),
            centroids,
            tables: [OnceLock::new(), OnceLock::new(), OnceLock::new()],
            river_by_num,
            river_cache: BoardCache::new(cap),
        })
    }

    pub fn centroids(&self, street: usize) -> &Centroids {
        &self.centroids[street]
    }

    pub fn n_buckets(&self, street: usize) -> usize {
        self.centroids[street].k
    }

    pub fn river_cache(&self) -> &BoardCache<RiverEntry> {
        &self.river_cache
    }

    /// Load + verify a street's table (once). Ok(None) = no table file.
    fn table(&self, street: usize) -> Result<Option<Arc<Table>>, String> {
        self.tables[street]
            .get_or_init(|| self.load_table(street).map(|t| t.map(Arc::new)))
            .clone()
    }

    /// Eagerly load (and stamp-check) the flop and turn tables.
    pub fn preload(&self) -> Result<(), String> {
        for s in 0..3 {
            self.table(s)?;
        }
        Ok(())
    }

    fn load_table(&self, street: usize) -> Result<Option<Table>, String> {
        let name = STREETS[street];
        let path = self.dir.join(format!("postflop_table_{name}.npz"));
        if !path.exists() {
            return Ok(None);
        }
        let mut z = Npz::open(&path)?;
        let ctx = |m: String| format!("{}: {m}", path.display());
        // _verify_stamp
        let c = &self.centroids[street];
        if z.has("centroid_hash") {
            // A partial checkpoint from an interrupted bake (complete=0); tables
            // baked before the field existed are complete.
            if z.has("complete") && z.read("complete")?.scalar_i64()? == 0 {
                return Err(format!(
                    "postflop_table_{name}.npz is a partial checkpoint from an interrupted \
                     bake; re-run python scripts/bake_postflop_table.py --street {name}"
                ));
            }
            let got = z.read("centroid_hash")?.scalar_str()?;
            let want = c.hash();
            if got != want {
                return Err(format!(
                    "postflop_table_{name}.npz was baked from DIFFERENT centroids (stamp {}... != \
                     current {}...). The table is stale; re-bake it: python \
                     scripts/bake_postflop_table.py --street {name}",
                    got.get(..8).unwrap_or(&got),
                    &want[..8]
                ));
            }
            let nb = z.read("n_buckets")?.scalar_i64()?;
            let tb = z.read("bins")?.scalar_i64()?;
            if nb != c.k as i64 || tb != c.bins as i64 {
                return Err(format!(
                    "postflop_table_{name}.npz K/bins ({nb}/{tb}) != current centroids ({}/{}); \
                     re-bake --street {name}.",
                    c.k, c.bins
                ));
            }
        } else {
            eprintln!(
                "warning: postflop_table_{name}.npz has no centroid stamp (legacy bake) -- cannot \
                 verify it matches the current centroids. Re-bake to enable the consistency check."
            );
        }
        let ids = match z.read("ids")?.data {
            NpyData::I64(v) => v,
            _ => return Err(ctx("ids must be an integer array".into())),
        };
        let buckets = match z.read("buckets")? {
            a if a.shape.len() == 1 => match a.data {
                NpyData::U8(v) => v,
                NpyData::I64(v) => v
                    .into_iter()
                    .map(|x| u8::try_from(x).map_err(|_| ctx(format!("bucket {x} out of range"))))
                    .collect::<Result<_, _>>()?,
                _ => return Err(ctx("buckets must be an integer array".into())),
            },
            _ => return Err(ctx("buckets must be 1-D".into())),
        };
        if ids.len() != buckets.len() {
            return Err(ctx(format!(
                "{} ids vs {} buckets",
                ids.len(),
                buckets.len()
            )));
        }
        if ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err(ctx("ids are not strictly increasing".into()));
        }
        if let Some(b) = buckets.iter().find(|&&b| b as usize >= c.k) {
            return Err(ctx(format!("bucket {b} >= K={}", c.k)));
        }
        Ok(Some(Table { ids, buckets }))
    }

    /// Bucket of one VALID situation (hole = 2 cards, board = 3..=5 cards,
    /// all distinct, < 52).
    pub fn bucket(&self, hole: [u8; 2], board: &[u8]) -> Result<u8, String> {
        let street = street_of(board.len())?;
        if let Some(t) = self.table(street)?
            && let Some(b) = t.lookup(situation_id(&hole, board))
        {
            return Ok(b);
        }
        if street == 2 {
            let cb = canonical_board(board);
            let entry = self.river_entry(&cb)?;
            return self.river_from_entry(&entry, &cb, hole);
        }
        Err(format!(
            "{} situation {hole:?} | {board:?} not in postflop_table_{}.npz (or the table is \
             missing); Python would fall back to non-reproducible Monte Carlo bucketing -- \
             re-bake the table",
            STREETS[street], STREETS[street]
        ))
    }

    fn river_entry(&self, cb: &canonical::CanonBoard) -> Result<Arc<RiverEntry>, String> {
        self.river_cache
            .get_or_compute(cb.key, || Ok(board_entry(&cb.cards[..cb.n])))
    }

    fn river_from_entry(
        &self,
        entry: &RiverEntry,
        cb: &canonical::CanonBoard,
        hole: [u8; 2],
    ) -> Result<u8, String> {
        let mut on_board = 0u64;
        for &c in &cb.cards[..cb.n] {
            on_board |= 1u64 << c;
        }
        let pos = |c: u8| -> Option<usize> {
            if on_board >> c & 1 == 1 {
                None
            } else {
                Some((!on_board & ((1u64 << c) - 1)).count_ones() as usize)
            }
        };
        let (ca, cb_) = (relabel(hole[0], &cb.perm), relabel(hole[1], &cb.perm));
        let (Some(mut i), Some(mut j)) = (pos(ca), pos(cb_)) else {
            return Err(format!("river hole {hole:?} overlaps the board"));
        };
        if i == j {
            return Err(format!("river hole {hole:?} has a duplicate card"));
        }
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        let slot = i * (RIVER_2NM1 - i) / 2 + (j - i - 1);
        Ok(self.river_by_num[entry[slot] as usize])
    }

    /// Batch over row-major `[n, width]` rows `[hole0, hole1, board...]`
    /// (width 5, 6 or 7). Validates every row first.
    pub fn bucket_batch(&self, cards: &[u8], width: usize) -> Result<Vec<i32>, String> {
        let nb = check_rows(cards, width)?;
        let street = street_of(nb)?;
        let table = self.table(street)?;
        let mut out = Vec::with_capacity(cards.len() / width);
        // consecutive rows on the same board reuse its canonical form + entry
        let mut last: Option<(Vec<u8>, canonical::CanonBoard, Arc<RiverEntry>)> = None;
        for row in cards.chunks_exact(width) {
            let (hole, board) = ([row[0], row[1]], &row[2..]);
            if let Some(t) = &table
                && let Some(b) = t.lookup(situation_id(&hole, board))
            {
                out.push(b as i32);
                continue;
            }
            if street != 2 {
                out.push(self.bucket(hole, board)? as i32); // produces the error
                continue;
            }
            let fresh = !matches!(&last, Some((b, _, _)) if b.as_slice() == board);
            if fresh {
                let cb = canonical_board(board);
                let e = self.river_entry(&cb)?;
                last = Some((board.to_vec(), cb, e));
            }
            if let Some((_, cb, e)) = &last {
                out.push(self.river_from_entry(e, cb, hole)? as i32);
            }
        }
        Ok(out)
    }

    /// `encode_situation(canonical_key(hole, board))` per row.
    pub fn situation_ids(&self, cards: &[u8], width: usize) -> Result<Vec<i64>, String> {
        situation_ids(cards, width)
    }

    pub fn clear_river_cache(&self) {
        self.river_cache.clear();
    }
}

/// Validate `[n, width]` rows (width 5..=7): ids < 52, distinct per row.
/// Returns the board length.
fn check_rows(cards: &[u8], width: usize) -> Result<usize, String> {
    if !(5..=7).contains(&width) {
        return Err(format!(
            "rows must be [hole0, hole1, board(3..5)], got width {width}"
        ));
    }
    if !cards.len().is_multiple_of(width) {
        return Err(format!(
            "{} ids is not a multiple of width {width}",
            cards.len()
        ));
    }
    for (i, row) in cards.chunks_exact(width).enumerate() {
        eval::validate_row(row).map_err(|e| format!("row {i}: {e}"))?;
    }
    Ok(width - 2)
}

/// `encode_situation(canonical_key(...))` for validated rows.
pub fn situation_ids(cards: &[u8], width: usize) -> Result<Vec<i64>, String> {
    check_rows(cards, width)?;
    Ok(cards
        .chunks_exact(width)
        .map(|r| situation_id(&r[..2], &r[2..]))
        .collect())
}

/// Preflop `(fine, coarse)` for `[n, 2]` rows.
pub fn preflop_batch(cards: &[u8]) -> Result<(Vec<i32>, Vec<i32>), String> {
    let (pairs, rest) = cards.as_chunks::<2>();
    if !rest.is_empty() {
        return Err("preflop rows must have 2 cards".into());
    }
    let n = pairs.len();
    let (mut f, mut c) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for (i, r) in pairs.iter().enumerate() {
        eval::validate_row(r).map_err(|e| format!("row {i}: {e}"))?;
        let (fb, cb) = preflop_buckets(r[0], r[1]);
        f.push(fb as i32);
        c.push(cb as i32);
    }
    Ok((f, c))
}

/// `PostflopV2._compute_board_entry(board)` / `board_winrates`: for every hand
/// over the 47 live cards (combination order), `rint(wr * 1980)` where
/// `wr = (winM + 0.5 * tieM) / compat` vs a uniform opponent range with exact
/// card removal. Counts are integers (exact in f64), and the final division /
/// rounding replays Python's float expression.
pub fn board_entry(board: &[u8]) -> RiverEntry {
    let base = eval::suit_masks(board);
    let mut on_board = 0u64;
    for &c in board {
        on_board |= 1u64 << c;
    }
    let live: Vec<u8> = (0..52u8).filter(|&c| on_board >> c & 1 == 0).collect();
    let h = live.len() * (live.len().saturating_sub(1)) / 2;
    let mut c1 = Vec::with_capacity(h);
    let mut c2 = Vec::with_capacity(h);
    let mut keyed: Vec<u32> = Vec::with_capacity(h);
    for i in 0..live.len() {
        for j in i + 1..live.len() {
            let idx = c1.len() as u32;
            let r = eval::rank_with(base, live[i], live[j]) as u32;
            keyed.push((r << 11) | idx);
            c1.push(live[i]);
            c2.push(live[j]);
        }
    }
    keyed.sort_unstable();
    let card_tot = (live.len() as i64) - 1; // hands per live card (46)
    let hh = h as i64;
    let compat = (hh - (card_tot + card_tot - 1)) as f64; // 990
    let mut out = [0u16; RIVER_HANDS];
    let mut scc = [0i64; 52]; // stronger-group hands per card
    let mut gc = [0i64; 52]; // this group's hands per card
    let mut sg = 0i64; // stronger hands
    let mut s = 0usize;
    while s < keyed.len() {
        let r = keyed[s] >> 11;
        let mut e = s;
        while e < keyed.len() && keyed[e] >> 11 == r {
            let k = (keyed[e] & 0x7FF) as usize;
            gc[c1[k] as usize] += 1;
            gc[c2[k] as usize] += 1;
            e += 1;
        }
        let grp = (e - s) as i64;
        for &kv in &keyed[s..e] {
            let k = (kv & 0x7FF) as usize;
            let (a, b) = (c1[k] as usize, c2[k] as usize);
            let wk = hh - sg - grp;
            let win = wk - (card_tot - scc[a] - gc[a]) - (card_tot - scc[b] - gc[b]);
            let tie = grp - gc[a] - gc[b] + 1;
            let wr = (win as f64 + 0.5 * tie as f64) / compat;
            if k < RIVER_HANDS {
                out[k] = (wr * RIVER_EQ_SCALE).round_ties_even() as u16;
            }
        }
        for &kv in &keyed[s..e] {
            let k = (kv & 0x7FF) as usize;
            for c in [c1[k] as usize, c2[k] as usize] {
                scc[c] += gc[c];
                gc[c] = 0;
            }
        }
        sg += grp;
        s = e;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force equity numerator (2*win + tie vs every disjoint hand).
    fn brute_num(board: &[u8], a: u8, b: u8) -> u16 {
        let me = eval::rank(&[a, b, board[0], board[1], board[2], board[3], board[4]]);
        let (mut w, mut t) = (0, 0);
        for x in 0..52u8 {
            for y in x + 1..52 {
                if board.contains(&x)
                    || board.contains(&y)
                    || [a, b].contains(&x)
                    || [a, b].contains(&y)
                {
                    continue;
                }
                let o = eval::rank(&[x, y, board[0], board[1], board[2], board[3], board[4]]);
                if me < o {
                    w += 1;
                } else if me == o {
                    t += 1;
                }
            }
        }
        2 * w + t
    }

    #[test]
    fn board_entry_matches_brute_force() {
        for board in [
            [0u8, 5, 10, 20, 51],
            [48, 49, 50, 44, 45],
            [0, 4, 8, 12, 30],
        ] {
            let e = board_entry(&board);
            let live: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
            let mut k = 0;
            for i in 0..live.len() {
                for j in i + 1..live.len() {
                    if k % 7 == 0 {
                        assert_eq!(e[k], brute_num(&board, live[i], live[j]), "{board:?} {k}");
                    }
                    k += 1;
                }
            }
        }
    }

    // ---- stale-table guard + loader robustness (synthetic assets) -----------

    fn npy(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(' ');
        }
        h.push('\n');
        let mut v = b"\x93NUMPY\x01\x00".to_vec();
        v.extend_from_slice(&(h.len() as u16).to_le_bytes());
        v.extend_from_slice(h.as_bytes());
        v.extend_from_slice(data);
        v
    }

    /// Minimal STORED zip writer (CRC left 0: the reader does not check it).
    fn write_npz(path: &Path, members: &[(&str, Vec<u8>)]) {
        let (mut out, mut cd) = (Vec::new(), Vec::new());
        for (name, data) in members {
            let name = format!("{name}.npy");
            let off = out.len() as u32;
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
            cd.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            cd.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
            cd.extend_from_slice(&[0u8; 12]);
            cd.extend_from_slice(&off.to_le_bytes());
            cd.extend_from_slice(name.as_bytes());
        }
        let cd_off = out.len() as u32;
        out.extend_from_slice(&cd);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(cd.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        std::fs::write(path, out).unwrap();
    }

    fn f64s(v: &[f64]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn unicode(s: &str) -> Vec<u8> {
        s.chars().flat_map(|c| (c as u32).to_le_bytes()).collect()
    }

    fn synthetic_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("allin_buckets_test_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cents = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]; // K=2, bins=3
        for s in STREETS {
            write_npz(
                &dir.join(format!("postflop_centroids_{s}.npz")),
                &[
                    ("centroids", npy("<f8", "(2, 3)", &f64s(&cents))),
                    ("bins", npy("<i8", "()", &3i64.to_le_bytes())),
                ],
            );
        }
        dir
    }

    fn write_flop_table(dir: &Path, stamp: &str, ids: &[i64], buckets: &[u8]) {
        let idb: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
        write_npz(
            &dir.join("postflop_table_flop.npz"),
            &[
                ("ids", npy("<i8", &format!("({},)", ids.len()), &idb)),
                (
                    "buckets",
                    npy("|u1", &format!("({},)", buckets.len()), buckets),
                ),
                (
                    "centroid_hash",
                    npy(&format!("<U{}", stamp.len()), "()", &unicode(stamp)),
                ),
                ("n_buckets", npy("<i8", "()", &2i64.to_le_bytes())),
                ("bins", npy("<i8", "()", &3i64.to_le_bytes())),
            ],
        );
    }

    #[test]
    fn stale_table_is_a_hard_error() {
        let dir = synthetic_dir("stale");
        let b = Buckets::new(&dir, Some(8)).unwrap();
        let (hole, board) = ([48u8, 44], [40u8, 36, 2]);
        let sid = situation_id(&hole, &board);
        // matching stamp: loads and looks up
        write_flop_table(&dir, &b.centroids(0).hash(), &[sid - 1, sid], &[0, 1]);
        assert_eq!(b.bucket(hole, &board).unwrap(), 1);
        // a different stamp: hard error (fresh instance: tables load once)
        write_flop_table(
            &dir,
            "0000000000000000000000000000000000000000",
            &[sid],
            &[1],
        );
        let e = Buckets::new(&dir, Some(8))
            .unwrap()
            .bucket(hole, &board)
            .unwrap_err();
        assert!(e.contains("DIFFERENT centroids"), "{e}");
        // unsorted ids / out-of-range bucket / truncated file: errors, no panic
        let h = b.centroids(0).hash();
        write_flop_table(&dir, &h, &[sid, sid - 1], &[0, 1]);
        assert!(Buckets::new(&dir, None).unwrap().preload().is_err());
        write_flop_table(&dir, &h, &[sid], &[7]);
        assert!(Buckets::new(&dir, None).unwrap().preload().is_err());
        write_flop_table(&dir, &h, &[sid], &[1]);
        let p = dir.join("postflop_table_flop.npz");
        let full = std::fs::read(&p).unwrap();
        for cut in [full.len() / 3, full.len() - 30, full.len() - 1] {
            std::fs::write(&p, &full[..cut]).unwrap();
            assert!(Buckets::new(&dir, None).unwrap().preload().is_err());
        }
        // missing flop table: flop bucketing refuses instead of guessing
        std::fs::remove_file(&p).unwrap();
        assert!(
            Buckets::new(&dir, None)
                .unwrap()
                .bucket(hole, &board)
                .is_err()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn real_assets_load_and_stamps_match() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bot/analysis/abstractions");
        if !dir.join("postflop_table_flop.npz").exists() {
            eprintln!("skipping: baked tables not present");
            return;
        }
        let b = Buckets::new(&dir, Some(1000)).unwrap();
        b.preload().unwrap(); // verifies both stamps against Python's centroid_hash
        let mut rows = Vec::new();
        for (h0, h1) in [(0u8, 1u8), (48, 49), (20, 33)] {
            rows.extend_from_slice(&[h0, h1, 4, 9, 14, 30, 51]);
        }
        let out = b.bucket_batch(&rows, 7).unwrap();
        assert!(out.iter().all(|&x| (0..10).contains(&x)));
        // invalid rows -> errors
        assert!(b.bucket_batch(&[0, 0, 4, 9, 14], 5).is_err());
        assert!(b.bucket_batch(&[0, 1, 4, 9, 99], 5).is_err());
        assert!(b.bucket_batch(&[0, 1, 4, 9], 4).is_err());
    }
}
