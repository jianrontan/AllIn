//! Card abstraction backed by precomputed per-deal values (the `mccfr` parity
//! fixture), so trainer parity is tested independently of the bucketing port.

use rustc_hash::FxHashMap;

use super::trainer::Abstraction;

#[derive(Default)]
pub struct FixtureAbstraction {
    fine: FxHashMap<[u8; 2], u8>,
    post: FxHashMap<([u8; 2], Vec<u8>), u16>,
    raw: FxHashMap<Vec<u8>, i32>,
    fine_to_coarse: Vec<u8>,
}

impl FixtureAbstraction {
    pub fn new(fine_to_coarse: Vec<u8>) -> Self {
        FixtureAbstraction {
            fine_to_coarse,
            ..Default::default()
        }
    }

    /// Register one deal: `cards` = [p0 x2, p1 x2, board x5], `post[seat]` =
    /// buckets on 3/4/5 board cards, `raw[seat]` = 7-card rank.
    pub fn add(&mut self, cards: [u8; 9], fine: [u8; 2], post: [[u16; 3]; 2], raw: [i32; 2]) {
        for s in 0..2 {
            let hole = [cards[2 * s], cards[2 * s + 1]];
            self.fine.insert(hole, fine[s]);
            for (i, nb) in [3usize, 4, 5].into_iter().enumerate() {
                self.post
                    .insert((hole, cards[4..4 + nb].to_vec()), post[s][i]);
            }
            let mut seven = hole.to_vec();
            seven.extend_from_slice(&cards[4..9]);
            self.raw.insert(seven, raw[s]);
        }
    }
}

impl Abstraction for FixtureAbstraction {
    fn preflop_fine(&self, hole: [u8; 2]) -> u8 {
        self.fine[&hole]
    }

    fn coarse(&self, fine: u8) -> u8 {
        self.fine_to_coarse[fine as usize]
    }

    fn postflop(&self, hole: [u8; 2], board: &[u8]) -> Result<u16, String> {
        self.post
            .get(&(hole, board.to_vec()))
            .copied()
            .ok_or_else(|| "fixture has no bucket for this situation".into())
    }

    fn rank(&self, cards: &[u8]) -> i32 {
        *self
            .raw
            .get(cards)
            .expect("fixture has no rank for this hand")
    }
}
