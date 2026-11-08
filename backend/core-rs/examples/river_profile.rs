//! Micro-profile of the river solver: kernel cost vs a full CFR iteration.
//! `cargo run --release --example river_profile`
use std::time::Instant;

use allin_core::river::tree::{DEFAULT_MENU, MAX_AGGRESSIONS, Node};
use allin_core::river::{Board, RiverCfr, RiverTree};

fn main() {
    // All 1081 hands off a fixed board; ranks with realistic tie structure.
    let board = [0u8, 5, 10, 15, 20];
    let pool: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
    let (mut c1, mut c2) = (Vec::new(), Vec::new());
    for i in 0..pool.len() {
        for j in i + 1..pool.len() {
            c1.push(pool[i] as i64);
            c2.push(pool[j] as i64);
        }
    }
    let h = c1.len();
    let raw: Vec<f64> = (0..h).map(|i| ((i * 7919) % 600) as f64).collect();
    let b = Board::new(&raw, &c1, &c2).unwrap();
    let rv: Vec<f64> = (0..h)
        .map(|i| 0.5 + ((i * 31) % 17) as f64 / 17.0)
        .collect();
    let (mut eq, mut cm) = (vec![0.0; h], vec![0.0; h]);
    let n = 20000;
    let t = Instant::now();
    for _ in 0..n {
        b.showdown_stats(&rv, &mut eq, &mut cm);
    }
    println!(
        "showdown_stats: {:.2} us/call (G={})",
        t.elapsed().as_secs_f64() / n as f64 * 1e6,
        b.num_groups()
    );
    let t = Instant::now();
    for _ in 0..n {
        b.compatible_mass(&rv, &mut cm);
    }
    println!(
        "compatible_mass: {:.2} us/call",
        t.elapsed().as_secs_f64() / n as f64 * 1e6
    );

    for (pot, stack) in [(12.0, 194.0), (30.0, 185.0), (80.0, 160.0)] {
        let tree = RiverTree::build(pot, [stack, stack], &DEFAULT_MENU, MAX_AGGRESSIONS).unwrap();
        let terms = tree
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Terminal(_)))
            .count();
        let showdowns = tree
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Terminal(t) if t.folder.is_none()))
            .count();
        let acts: usize = (0..tree.num_decision())
            .map(|i| tree.decision_node(i).actions.len())
            .sum();
        let nd = tree.num_decision();
        let mut cfr = RiverCfr::new(tree, b.clone());
        let ones = vec![1.0; h];
        cfr.run(&ones, &ones, 3).unwrap();
        let it = 200;
        let t = Instant::now();
        cfr.run(&ones, &ones, it).unwrap();
        println!(
            "pot {pot} stack {stack}: {nd} decisions, {acts} edges, {terms} terminals ({showdowns} showdown): {:.3} ms/iter",
            t.elapsed().as_secs_f64() / it as f64 * 1e3
        );
    }
}
