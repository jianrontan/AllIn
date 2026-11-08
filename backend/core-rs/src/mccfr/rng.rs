//! Randomness for training: a uniform stream (opponent sampling) and deals.
//! `Xoshiro` is the production generator; `StreamRng` replays an injected
//! uniform stream so the parity fixtures can drive Rust and Python identically.

pub trait Uniform {
    /// A float in [0, 1), like Python's `random.random()`.
    fn uniform(&mut self) -> Result<f64, String>;
}

/// xoshiro256++ (Blackman & Vigna), seeded through SplitMix64.
#[derive(Clone, Debug)]
pub struct Xoshiro {
    s: [u64; 4],
}

impl Xoshiro {
    pub fn new(seed: u64) -> Xoshiro {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Xoshiro {
            s: [next(), next(), next(), next()],
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Unbiased integer in [0, n) (Lemire's method).
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        let n64 = n as u64;
        let mut m = (self.next_u64() >> 32) * n64;
        if (m as u32) < n {
            let t = n.wrapping_neg() % n;
            while (m as u32) < t {
                m = (self.next_u64() >> 32) * n64;
            }
        }
        (m >> 32) as u32
    }

    /// Nine distinct cards: p0 hole, p1 hole, five board (partial Fisher-Yates).
    pub fn deal(&mut self) -> [u8; 9] {
        let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
        for i in 0..9 {
            let j = i + self.below((52 - i) as u32) as usize;
            deck.swap(i, j);
        }
        deck[..9].try_into().unwrap()
    }
}

impl Uniform for Xoshiro {
    #[inline]
    fn uniform(&mut self) -> Result<f64, String> {
        Ok((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }
}

/// Replays a fixed stream of uniforms; running out is an error, not a wrap.
pub struct StreamRng<'a> {
    pub vals: &'a [f64],
    pub pos: usize,
}

impl Uniform for StreamRng<'_> {
    fn uniform(&mut self) -> Result<f64, String> {
        let v = *self
            .vals
            .get(self.pos)
            .ok_or("injected uniform stream exhausted")?;
        self.pos += 1;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deals_are_distinct_and_uniformish() {
        let mut r = Xoshiro::new(7);
        let mut counts = [0u32; 52];
        for _ in 0..20000 {
            let d = r.deal();
            for i in 0..9 {
                assert!(d[i] < 52 && !d[..i].contains(&d[i]));
                counts[d[i] as usize] += 1;
            }
        }
        let mean = 20000.0 * 9.0 / 52.0;
        for c in counts {
            assert!((c as f64 - mean).abs() < 0.08 * mean, "{c} vs {mean}");
        }
        for _ in 0..1000 {
            let u = r.uniform().unwrap();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
