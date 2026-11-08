//! Exact range-vs-range showdown kernel with card removal.
//!
//! Port of `showdown_kernel.compatible_mass` / `showdown_measure`. Python does
//! it with `np.bincount` over a [groups x 52] table; here the hands are sorted
//! by strength once per board and each call is a single O(H) sweep with 52-entry
//! running sums, which gives the same quantities (up to summation order).
//!
//! Conventions (same as Python): lower `raw` = stronger hand; card ids are
//! 0..52; every value is a MEASURE (weighted by the villain reach `rv`).

//!
//! Two arithmetic modes:
//! * **fast** (default): the O(H) sweep below.
//! * **exact**: replays numpy's operation order (`np.bincount`, `np.cumsum`,
//!   pairwise `sum`) so results are bit-identical to the Python kernel. CFR+
//!   is chaotic at exact ties (a 1e-11 noise regret crosses the 1e-12
//!   regret-matching threshold and flips a hand's strategy), so cross-language
//!   trajectory parity is only meaningful in this mode. It is slower (a
//!   [groups x 52] table per call) and exists for verification.

pub const NUM_CARDS: usize = 52;
/// C(52, 2): no river range has more hands.
pub const MAX_HANDS: usize = 1326;

/// numpy's pairwise summation (`pairwise_sum` in numpy 1.26 `loops_utils.h`),
/// which is what `ndarray.sum()` uses for a contiguous float64 vector. For
/// fewer than 8 elements it is a plain left-to-right sum.
pub fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = -0.0;
        for &x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

/// numpy `sum(x * y)` for two vectors (element-wise product, then pairwise sum).
pub fn np_dot(x: &[f64], y: &[f64]) -> f64 {
    let p: Vec<f64> = x.iter().zip(y).map(|(a, b)| a * b).collect();
    np_sum(&p)
}

/// The per-board arrays the river solver needs (the subset of Python's
/// `build_board_arrays` it reads): card ids per hand plus the strength order.
#[derive(Clone, Debug)]
pub struct Board {
    pub h: usize,
    pub c1: Vec<u8>,
    pub c2: Vec<u8>,
    /// Dense strength group per hand, 0 = strongest (Python's `g`).
    g: Vec<u32>,
    /// Hand indices sorted strongest-first (ascending raw), stable.
    order: Vec<u32>,
    /// Exclusive end offset into `order` of each tie group, strongest first.
    group_end: Vec<u32>,
    // ---- fast-kernel index (all positions are into the strength-sorted order) ----
    /// Group start / end position of each sorted position.
    gs: Vec<u32>,
    ge: Vec<u32>,
    /// Per card c: its hands' sorted positions live at `cl[base[c] .. base[c] + cnt[c]]`
    /// (ascending), followed by one sentinel slot (the card's total).
    cl: Vec<u32>,
    base: [u32; NUM_CARDS],
    cnt: [u32; NUM_CARDS],
    /// Per sorted position and card slot (c1, c2): the range in `cl` of that
    /// card's hands in the same tie group (everything before `lo` is stronger).
    lo: Vec<[u32; 2]>,
    hi: Vec<[u32; 2]>,
    /// Sorted position -> its two cards.
    sc: Vec<[u8; 2]>,
    /// Use the numpy-order (bit-exact) kernel instead of the fast sweep.
    pub exact: bool,
}

impl Board {
    /// Build from showdown ranks and card ids (as Python's `build_board_arrays`
    /// produces them). Tie groups are hands with exactly equal `raw`.
    pub fn new(raw: &[f64], c1: &[i64], c2: &[i64]) -> Result<Self, String> {
        let h = raw.len();
        if h > MAX_HANDS {
            return Err(format!("at most {MAX_HANDS} hands (got {h})"));
        }
        if c1.len() != h || c2.len() != h {
            return Err(format!(
                "raw/c1/c2 length mismatch: {h}/{}/{}",
                c1.len(),
                c2.len()
            ));
        }
        let card = |c: i64| -> Result<u8, String> {
            if (0..NUM_CARDS as i64).contains(&c) {
                Ok(c as u8)
            } else {
                Err(format!("card id {c} out of range 0..52"))
            }
        };
        let c1: Vec<u8> = c1.iter().map(|&c| card(c)).collect::<Result<_, _>>()?;
        let c2: Vec<u8> = c2.iter().map(|&c| card(c)).collect::<Result<_, _>>()?;
        if raw.iter().any(|r| r.is_nan()) {
            return Err("raw contains NaN".into());
        }
        let mut order: Vec<u32> = (0..h as u32).collect();
        order.sort_by(|&a, &b| raw[a as usize].partial_cmp(&raw[b as usize]).unwrap());
        let mut group_end = Vec::new();
        let mut g = vec![0u32; h];
        for i in 0..h {
            if i > 0 && raw[order[i] as usize] != raw[order[i - 1] as usize] {
                group_end.push(i as u32);
            }
            g[order[i] as usize] = group_end.len() as u32;
        }
        if h > 0 {
            group_end.push(h as u32);
        }
        // Fast-kernel index.
        let sc: Vec<[u8; 2]> = order
            .iter()
            .map(|&o| [c1[o as usize], c2[o as usize]])
            .collect();
        let (mut gs, mut ge) = (vec![0u32; h], vec![0u32; h]);
        let mut start = 0u32;
        for &end in &group_end {
            for i in start..end {
                gs[i as usize] = start;
                ge[i as usize] = end;
            }
            start = end;
        }
        let mut cnt = [0u32; NUM_CARDS];
        for s in &sc {
            if s[0] == s[1] {
                return Err("a hand has two identical cards".into());
            }
            cnt[s[0] as usize] += 1;
            cnt[s[1] as usize] += 1;
        }
        let mut base = [0u32; NUM_CARDS];
        let mut next = 0u32;
        for c in 0..NUM_CARDS {
            base[c] = next;
            next += cnt[c] + 1; // + sentinel
        }
        let mut cl = vec![u32::MAX; next as usize];
        let mut fill = base;
        for (i, s) in sc.iter().enumerate() {
            for &c in s {
                cl[fill[c as usize] as usize] = i as u32;
                fill[c as usize] += 1;
            }
        }
        let mut lo = vec![[0u32; 2]; h];
        let mut hi = vec![[0u32; 2]; h];
        for (i, s) in sc.iter().enumerate() {
            for slot in 0..2 {
                let c = s[slot] as usize;
                let (b, n) = (base[c] as usize, cnt[c] as usize);
                let list = &cl[b..b + n];
                lo[i][slot] = (b + list.partition_point(|&p| p < gs[i])) as u32;
                hi[i][slot] = (b + list.partition_point(|&p| p < ge[i])) as u32;
            }
        }
        Ok(Board {
            h,
            c1,
            c2,
            g,
            order,
            group_end,
            gs,
            ge,
            cl,
            base,
            cnt,
            lo,
            hi,
            sc,
            exact: false,
        })
    }

    /// Number of distinct strength groups (Python's `G`).
    pub fn num_groups(&self) -> usize {
        self.group_end.len()
    }

    /// Total reach and per-card reach sums. Four accumulator banks break the
    /// store-to-load dependency chains when consecutive hands share a card.
    fn card_totals(&self, rv: &[f64]) -> (f64, [f64; NUM_CARDS]) {
        let mut banks = [[0.0f64; NUM_CARDS]; 4];
        let mut tot = [0.0f64; 4];
        for (h, &r) in rv.iter().enumerate().take(self.h) {
            let k = h & 3;
            tot[k] += r;
            banks[k][self.c1[h] as usize] += r;
            banks[k][self.c2[h] as usize] += r;
        }
        let mut ct = [0.0f64; NUM_CARDS];
        for c in 0..NUM_CARDS {
            ct[c] = (banks[0][c] + banks[1][c]) + (banks[2][c] + banks[3][c]);
        }
        ((tot[0] + tot[1]) + (tot[2] + tot[3]), ct)
    }

    /// Per-hero-hand reach of villain hands sharing no card with the hero hand.
    pub fn compatible_mass(&self, rv: &[f64], out: &mut [f64]) {
        if self.exact {
            return self.compatible_mass_exact(rv, out);
        }
        let (total, ct) = self.card_totals(rv);
        for h in 0..self.h {
            out[h] = total - (ct[self.c1[h] as usize] + ct[self.c2[h] as usize] - rv[h]);
        }
    }

    /// Showdown statistics vs villain reach `rv`, per hero hand:
    /// `eq[h] = win + tie/2` (compatible villain mass) and `compat[h]`
    /// (identical to `compatible_mass`). Fast mode only.
    /// The showdown measure is then `final_pot * eq - hero_total * compat`.
    pub fn showdown_stats(&self, rv: &[f64], eq: &mut [f64], compat: &mut [f64]) {
        let n = self.h;
        // Reach in strength order, and its exclusive prefix sums t[i] (t[n] = total).
        let rs: Vec<f64> = self.order.iter().map(|&o| rv[o as usize]).collect();
        let mut t = vec![0.0f64; n + 1];
        let mut acc = 0.0;
        for i in 0..n {
            t[i] = acc;
            acc += rs[i];
        }
        t[n] = acc;
        let total = acc;
        // Per card, exclusive prefix sums over its hands in strength order; the
        // sentinel slot holds the card's total. The 52 chains are independent,
        // so they overlap in the CPU instead of serialising through memory.
        let mut p = vec![0.0f64; self.cl.len()];
        let mut ct = [0.0f64; NUM_CARDS];
        for c in 0..NUM_CARDS {
            let (b, k) = (self.base[c] as usize, self.cnt[c] as usize);
            let mut a = 0.0;
            for j in b..b + k {
                p[j] = a;
                a += rs[self.cl[j] as usize];
            }
            p[b + k] = a;
            ct[c] = a;
        }
        for i in 0..n {
            let h = self.order[i] as usize;
            let [ca, cb] = self.sc[i];
            let [la, lb] = self.lo[i];
            let [ha, hb] = self.hi[i];
            let (la, lb, ha, hb) = (la as usize, lb as usize, ha as usize, hb as usize);
            let (gs, ge) = (self.gs[i] as usize, self.ge[i] as usize);
            let r = rs[i];
            let cm = total - (ct[ca as usize] + ct[cb as usize] - r);
            // Compatible villain mass strictly stronger, and tied (excluding self).
            let lose = t[gs] - p[la] - p[lb];
            let tie = (t[ge] - t[gs]) - (p[ha] - p[la]) - (p[hb] - p[lb]) + r;
            compat[h] = cm;
            eq[h] = cm - lose - 0.5 * tie; // = win + tie/2
        }
    }

    /// Python `showdown_measure(ba, rv, final_pot, hero_total)`.
    pub fn showdown_measure(&self, rv: &[f64], final_pot: f64, hero_total: f64, out: &mut [f64]) {
        if self.exact {
            return self.showdown_measure_exact(rv, final_pot, hero_total, out);
        }
        let mut compat = vec![0.0; self.h];
        self.showdown_stats(rv, out, &mut compat);
        for h in 0..self.h {
            out[h] = final_pot * out[h] - hero_total * compat[h];
        }
    }

    // ---- exact (numpy operation order) -------------------------------------

    /// `np.bincount(c1, rv, 52) + np.bincount(c2, rv, 52)`.
    fn card_tot_exact(&self, rv: &[f64]) -> [f64; NUM_CARDS] {
        let mut b1 = [0.0f64; NUM_CARDS];
        let mut b2 = [0.0f64; NUM_CARDS];
        for h in 0..self.h {
            b1[self.c1[h] as usize] += rv[h];
        }
        for h in 0..self.h {
            b2[self.c2[h] as usize] += rv[h];
        }
        let mut t = [0.0f64; NUM_CARDS];
        for c in 0..NUM_CARDS {
            t[c] = b1[c] + b2[c];
        }
        t
    }

    fn compatible_mass_exact(&self, rv: &[f64], out: &mut [f64]) {
        let total = np_sum(rv);
        let ct = self.card_tot_exact(rv);
        for h in 0..self.h {
            out[h] = total - (ct[self.c1[h] as usize] + ct[self.c2[h] as usize] - rv[h]);
        }
    }

    /// Line-for-line replay of `showdown_kernel.showdown_measure`.
    fn showdown_measure_exact(&self, rv: &[f64], fp: f64, ht: f64, out: &mut [f64]) {
        const NC: usize = NUM_CARDS;
        let ng = self.num_groups();
        let total = np_sum(rv);
        let ct = self.card_tot_exact(rv);
        let mut group_sum = vec![0.0f64; ng];
        for h in 0..self.h {
            group_sum[self.g[h] as usize] += rv[h];
        }
        let mut stronger_group = vec![0.0f64; ng]; // cumsum(groupSum) - groupSum
        let mut cum = 0.0;
        for k in 0..ng {
            cum = if k == 0 {
                group_sum[0]
            } else {
                cum + group_sum[k]
            };
            stronger_group[k] = cum - group_sum[k];
        }
        let mut f1 = vec![0.0f64; ng * NC];
        let mut f2 = vec![0.0f64; ng * NC];
        for h in 0..self.h {
            f1[self.g[h] as usize * NC + self.c1[h] as usize] += rv[h];
        }
        for h in 0..self.h {
            f2[self.g[h] as usize * NC + self.c2[h] as usize] += rv[h];
        }
        let gc: Vec<f64> = f1.iter().zip(&f2).map(|(a, b)| a + b).collect();
        // strongerCardCum = cumsum(gc, axis=0) - gc
        let mut scc = vec![0.0f64; ng * NC];
        let mut col = [0.0f64; NC];
        for k in 0..ng {
            for c in 0..NC {
                let x = gc[k * NC + c];
                col[c] = if k == 0 { x } else { col[c] + x };
                scc[k * NC + c] = col[c] - x;
            }
        }
        for h in 0..self.h {
            let gi = self.g[h] as usize;
            let (a, b) = (self.c1[h] as usize, self.c2[h] as usize);
            let r = rv[h];
            let compat = total - (ct[a] + ct[b] - r);
            let sg = stronger_group[gi];
            let grp = group_sum[gi];
            let wk = total - sg - grp;
            let (scc1, scc2) = (scc[gi * NC + a], scc[gi * NC + b]);
            let (gcc1, gcc2) = (gc[gi * NC + a], gc[gi * NC + b]);
            // (Python also forms loseM = sg - scc1 - scc2; the payoff doesn't use it.)
            let tie = grp - gcc1 - gcc2 + r;
            let wcc1 = ct[a] - scc1 - gcc1;
            let wcc2 = ct[b] - scc2 - gcc2;
            let win = wk - wcc1 - wcc2;
            out[h] = fp * win + (fp / 2.0) * tie - ht * compat;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Tiny deterministic generator so the tests need no dependencies.
    pub(crate) fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Every 2-card hand from the 47 cards off a fixed board, with coarse random
    /// ranks (many ties) and some zero reach.
    pub(crate) fn random_board(seed: u64) -> (Board, Vec<f64>, Vec<i64>, Vec<i64>) {
        let board = [0u8, 5, 10, 15, 20];
        let pool: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
        let (mut c1, mut c2) = (Vec::new(), Vec::new());
        for i in 0..pool.len() {
            for j in i + 1..pool.len() {
                c1.push(pool[i] as i64);
                c2.push(pool[j] as i64);
            }
        }
        let mut s = seed;
        let raw: Vec<f64> = (0..c1.len())
            .map(|_| (lcg(&mut s) * 40.0).floor())
            .collect();
        (Board::new(&raw, &c1, &c2).unwrap(), raw, c1, c2)
    }

    #[test]
    fn showdown_matches_brute_force() {
        let (mut b, raw, c1, c2) = random_board(7);
        let h = b.h;
        let mut seed = 11u64;
        let rv: Vec<f64> = (0..h)
            .map(|_| {
                if lcg(&mut seed) < 0.2 {
                    0.0
                } else {
                    lcg(&mut seed)
                }
            })
            .collect();
        let (fp, ht) = (37.5, 11.25);
        for exact in [false, true] {
            b.exact = exact;
            check_brute_force(&b, &raw, &c1, &c2, &rv, fp, ht);
        }
    }

    fn check_brute_force(
        b: &Board,
        raw: &[f64],
        c1: &[i64],
        c2: &[i64],
        rv: &[f64],
        fp: f64,
        ht: f64,
    ) {
        let h = b.h;
        let mut got = vec![0.0; h];
        b.showdown_measure(rv, fp, ht, &mut got);
        let mut cm = vec![0.0; h];
        b.compatible_mass(rv, &mut cm);
        for x in 0..h {
            let (mut want, mut want_cm) = (0.0, 0.0);
            for v in 0..h {
                let share = c1[x] == c1[v] || c1[x] == c2[v] || c2[x] == c1[v] || c2[x] == c2[v];
                if share {
                    continue;
                }
                want_cm += rv[v];
                let win = if raw[x] < raw[v] {
                    fp
                } else if raw[x] == raw[v] {
                    fp / 2.0
                } else {
                    0.0
                };
                want += rv[v] * (win - ht);
            }
            assert!(
                (got[x] - want).abs() < 1e-9 * (1.0 + want.abs()),
                "hand {x}: {} vs {want}",
                got[x]
            );
            assert!((cm[x] - want_cm).abs() < 1e-9 * (1.0 + want_cm.abs()));
        }
    }
}
