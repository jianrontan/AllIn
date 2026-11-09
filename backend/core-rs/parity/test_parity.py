"""Rust (allin_core) vs Python reference parity, per component.

Each section below is also the API CONTRACT the Rust bindings must expose for
that component. The contract may be revised during a phase (update this file
and docs/RUST_CORE_PLAN.md together), but the comparison semantics may not be
loosened without recording why in the plan's progress log.

Tolerances:
  * discrete outputs (ranks, buckets, keys, legal actions, tree shape): EXACT
  * engine chip math: EXACT (the Python engine is float-exact by design; Lever A
    relies on assignment, not accumulation -- see BlueprintTrainer.cfr)
  * river solver, EXACT mode (`exact=True`, replays numpy's operation order):
    BIT-IDENTICAL to Python -- kernel outputs, regrets, average strategies,
    values, exploitability, gadget regret, at every checkpoint.
  * river solver, FAST mode (default, the served path; O(H) showdown sweep):
    kernel within rtol=1e-9 / atol=1e-9*max|expected|; CFR trajectories within
    the same tolerance on non-degenerate reaches. When BOTH root reaches are
    uniform the ranges are symmetric and many action values tie exactly; CFR+
    then breaks ties on float noise (a ~1e-11 regret crosses the 1e-12
    regret-matching threshold), so the trajectory legitimately differs from
    numpy's -- Python's own path there is noise-determined. That case is held
    to solution quality instead: exploitability within FAST_TIE_EXPL_RTOL.
    (Progress log 2026-11-08 records this split.)
"""
import json
import re

import numpy as np
import pytest

from conftest import fixture_path, need

RTOL = 1e-9
FAST_TIE_EXPL_RTOL = 0.02   # fast mode, both-uniform reaches only (see module docstring)


def assert_bit_equal(actual, expected, what):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    bad = np.flatnonzero(actual.ravel() != expected.ravel())
    assert bad.size == 0, (f"{what}: {bad.size} entries differ, first {bad[:5]}, max abs diff "
                           f"{np.max(np.abs(actual - expected)):.3e}")


def assert_close(actual, expected, what):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    scale = max(1.0, float(np.max(np.abs(expected)))) if expected.size else 1.0
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=RTOL * scale, err_msg=what)


# ----------------------------------------------------------------------------
# eval_rank7 -- contract: rank_batch(cards: uint8[N, k]) -> int32[N]  (k = 5, 6, 7)
# ----------------------------------------------------------------------------
def test_eval_rank7():
    core = need('eval_rank7', 'eval_rank7.npz')
    d = np.load(fixture_path('eval_rank7.npz'))
    for k in (5, 6, 7):
        got = np.asarray(core.rank_batch(d[f'cards{k}']))
        bad = np.flatnonzero(got != d[f'ranks{k}'])
        assert bad.size == 0, f"{bad.size} mismatches for {k}-card hands, first rows {bad[:5]}"


# ----------------------------------------------------------------------------
# buckets -- contract:
#   preflop_buckets(cards: uint8[N, 2]) -> (fine int32[N], coarse int32[N])
#   Buckets(abstractions_dir: str)
#     .situation_ids(cards: uint8[N, 2 + nb]) -> int64[N]   (canonical encode)
#     .bucket_batch(cards: uint8[N, 2 + nb]) -> int32[N]    (nb = 3 flop, 4 turn, 5 river)
#   card rows are [hole0, hole1, board...]
# ----------------------------------------------------------------------------
def test_buckets():
    core = need('buckets', 'buckets.npz')
    import os
    from conftest import BOT
    d = np.load(fixture_path('buckets.npz'))
    fine, coarse = core.preflop_buckets(d['pf_cards'])
    assert np.array_equal(fine, d['pf_fine']), "preflop fine buckets"
    assert np.array_equal(coarse, d['pf_coarse']), "preflop coarse classes"
    b = core.Buckets(os.path.join(BOT, 'analysis', 'abstractions'))
    for street in ('flop', 'turn', 'river'):
        cards = d[f'{street}_cards']
        assert np.array_equal(b.situation_ids(cards), d[f'{street}_sid']), f"{street} situation ids"
        got = np.asarray(b.bucket_batch(cards))
        bad = np.flatnonzero(got != d[f'{street}_bucket'])
        assert bad.size == 0, f"{street}: {bad.size}/{len(got)} bucket mismatches, rows {bad[:5]}"


# ----------------------------------------------------------------------------
# keys -- contract: make_info_set_key(street, position, preflop_bucket, strength|None,
#                   pattern) -> str ;  action_char(action) -> str
# ----------------------------------------------------------------------------
def test_keys():
    core = need('keys', 'keys.json')
    with open(fixture_path('keys.json')) as f:
        d = json.load(f)
    for action, ch in d['action_chars'].items():
        assert core.action_char(action) == ch, action
    for c in d['cases']:
        got = core.make_info_set_key(c['street'], c['position'], c['preflop_bucket'],
                                     c['strength'], c['pattern'])
        assert got == c['key'], c


# ----------------------------------------------------------------------------
# engine -- contract: Engine(mode: 'control' | 'capped') with
#   .init_node_state(street, starting_pot) -> dict
#   .acting_player(n_actions_this_street, street) -> int
#   .legal_actions(street, st, cp, stack_cp) -> list[str]
#   .action_cost(action, street, st, cp, stack_cp) -> float
#   .advance(st, action, street, cp, stack_cp, p_inv) -> dict
#   .is_terminal(history, street) -> bool
#   .utility(p0_cards, p1_cards, board, history, street, starting_pot,
#            p0_inv, p1_inv, final_pot, p0_total) -> float   (P0 perspective)
# State dicts must equal Python's EXACTLY (same keys, same float values).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize('mode', ['control', 'capped', 'capped_pj'])
def test_engine(mode):
    core = need('engine_pj' if mode == 'capped_pj' else 'engine', f'engine_{mode}.json')
    with open(fixture_path(f'engine_{mode}.json')) as f:
        d = json.load(f)
    eng = core.Engine(mode)
    for gi, g in enumerate(d['games']):
        street, starting_pot = 0, 3.0
        p_inv = [0.0, 0.0]
        history = []
        st = eng.init_node_state(0, starting_pot)
        for ni, n in enumerate(g['nodes']):
            where = f"game {gi} node {ni}"
            if 'transition_to' in n:
                street, p_inv, starting_pot = n['transition_to'], n['p_inv'], n['starting_pot']
                history = []
                st = eng.init_node_state(street, starting_pot)
                assert st == n['st'], f"{where}: init state"
                continue
            assert st == n['st'], f"{where}: state"
            assert not eng.is_terminal(history, street), f"{where}: is_terminal"
            assert eng.acting_player(len(history), street) == n['cp'], f"{where}: acting player"
            legal = eng.legal_actions(street, st, n['cp'], n['stack_cp'])
            assert legal == n['legal'], f"{where}: legal {legal} != {n['legal']}"
            for pa in n['per_action']:
                a = pa['action']
                assert eng.action_cost(a, street, st, n['cp'], n['stack_cp']) == pa['cost'], \
                    f"{where}: cost of {a}"
                assert eng.advance(st, a, street, n['cp'], n['stack_cp'], tuple(p_inv)) == pa['child'], \
                    f"{where}: child state after {a}"
            pick = n['per_action'][n['pick']]
            st = pick['child']
            history = history + [pick['action']]
        assert history == g['final_history'], f"game {gi}: final history"
        t = g['terminal']
        u = eng.utility(g['p0_cards'], g['p1_cards'], g['board'], history, min(t['street'], 3),
                        starting_pot, p_inv[0], p_inv[1], st['pot'], p_inv[0] + st['c'][0])
        assert u == t['utility_p0'], f"game {gi}: utility {u} != {t['utility_p0']}"


# ----------------------------------------------------------------------------
# river_showdown -- contract:
#   showdown_measure(raw f64[H], c1 i64[H], c2 i64[H], rv f64[H], final_pot, hero_total,
#                    exact=False) -> f64[H]
#   compatible_mass(c1, c2, rv, exact=False) -> f64[H]
# ----------------------------------------------------------------------------
@pytest.mark.parametrize('exact', [True, False], ids=['exact', 'fast'])
def test_river_showdown(exact):
    core = need('river_showdown', 'river_showdown.npz')
    check = assert_bit_equal if exact else assert_close
    d = np.load(fixture_path('river_showdown.npz'))
    for b in range(int(d['n_boards'])):
        raw, c1, c2 = d[f'b{b}_raw'], d[f'b{b}_c1'], d[f'b{b}_c2']
        for k in range(3):
            rv = d[f'b{b}_k{k}_rv']
            fp, ht = d[f'b{b}_k{k}_args']
            check(core.showdown_measure(raw, c1, c2, rv, fp, ht, exact=exact),
                  d[f'b{b}_k{k}_showdown'], f"board {b} case {k} showdown")
            check(core.compatible_mass(c1, c2, rv, exact=exact), d[f'b{b}_k{k}_compat'],
                  f"board {b} case {k} compatible_mass")


# ----------------------------------------------------------------------------
# river_tree -- contract: river_tree_nodes(pot_entry, stack, menu, max_aggressions)
#   -> list[dict], pre-order, same fields as export_rust_fixtures._tree_to_list.
#   Action labels must match exactly ('bet:<chips:.6g>' etc.).
# ----------------------------------------------------------------------------
def test_river_tree():
    core = need('river_tree', 'river_tree.json')
    with open(fixture_path('river_tree.json')) as f:
        d = json.load(f)
    for c in d['cases']:
        got = core.river_tree_nodes(c['pot_entry'], c['stack'], c['menu'], c['max_aggressions'])
        assert len(got) == len(c['nodes']), f"pot={c['pot_entry']} stack={c['stack']}: node count"
        for i, (g, e) in enumerate(zip(got, c['nodes'])):
            assert g == e, f"pot={c['pot_entry']} stack={c['stack']} node {i}: {g} != {e}"


# ----------------------------------------------------------------------------
# river_cfr / river_gadget -- contract:
#   RiverCFR(pot_entry, stack, menu, max_aggressions, raw, c1, c2, exact=False)
#     .n_decision -> int
#     .run(r0, r1, iters)                      (Linear-CFR clock persists across calls)
#     .run_gadget(hero, villain, optout, villain_seat, iters)
#     .average_strategy(nid) -> f64[H, A] ; .regret(nid) -> f64[H, A]
#     .current_values(r0, r1) -> (f64[H], f64[H])
#     .exploitability(r0, r1) -> float
#     .gadget_regret() -> f64[H, 2]
#   Default tree params (menu, max_aggressions) are those in river_tree.json.
# ----------------------------------------------------------------------------
def _tree_defaults():
    with open(fixture_path('river_tree.json')) as f:
        c = json.load(f)['cases'][0]
    return c['menu'], c['max_aggressions']


def _check_snapshot(cfr, d, prefix, r0, r1, label, check):
    shapes = d['shapes']
    strat = np.concatenate([np.asarray(cfr.average_strategy(i)).ravel() for i in range(len(shapes))])
    regret = np.concatenate([np.asarray(cfr.regret(i)).ravel() for i in range(len(shapes))])
    check(strat, d[f'{prefix}strat'], f"{label} average strategy")
    check(regret, d[f'{prefix}regret'], f"{label} regret")
    v0, v1 = cfr.current_values(r0, r1)
    check(v0, d[f'{prefix}v0'], f"{label} v0")
    check(v1, d[f'{prefix}v1'], f"{label} v1")
    check(cfr.exploitability(r0, r1), d[f'{prefix}expl'], f"{label} exploitability")


def _tie_degenerate(case):
    return case.get('reach') == ['uniform', 'uniform']


def _check_tie_case(cfr, d, prefix, r0, r1, label):
    """Fast mode on symmetric (both-uniform) reaches: solution quality only."""
    got, want = cfr.exploitability(r0, r1), float(d[f'{prefix}expl'])
    assert abs(got - want) <= FAST_TIE_EXPL_RTOL * abs(want), \
        f"{label} exploitability {got} vs {want} (tie-degenerate case, rtol {FAST_TIE_EXPL_RTOL})"


@pytest.mark.parametrize('exact', [True, False], ids=['exact', 'fast'])
def test_river_cfr(exact):
    core = need('river_cfr', 'river_cfr.json', 'river_tree.json')
    menu, max_agg = _tree_defaults()
    with open(fixture_path('river_cfr.json')) as f:
        meta = json.load(f)
    for case in meta['cases']:
        d = np.load(fixture_path(case['file']))
        cfr = core.RiverCFR(case['pot_entry'], case['stack'], menu, max_agg,
                            d['raw'], d['c1'], d['c2'], exact=exact)
        assert cfr.n_decision == case['n_decision']
        r0, r1 = d['r0'], d['r1']
        done = 0
        for cp in meta['checkpoints']:
            cfr.run(r0, r1, cp - done)
            done = cp
            label = f"{case['file']} @ {cp} iters ({'exact' if exact else 'fast'})"
            if exact:
                _check_snapshot(cfr, d, f'it{cp}_', r0, r1, label, assert_bit_equal)
            elif _tie_degenerate(case):
                _check_tie_case(cfr, d, f'it{cp}_', r0, r1, label)
            else:
                _check_snapshot(cfr, d, f'it{cp}_', r0, r1, label, assert_close)


@pytest.mark.parametrize('exact', [True, False], ids=['exact', 'fast'])
def test_river_gadget(exact):
    core = need('river_gadget', 'river_gadget.json', 'river_tree.json')
    menu, max_agg = _tree_defaults()
    check = assert_bit_equal if exact else assert_close
    with open(fixture_path('river_gadget.json')) as f:
        meta = json.load(f)
    for case in meta['cases']:
        d = np.load(fixture_path(case['file']))
        cfr = core.RiverCFR(case['pot_entry'], case['stack'], menu, max_agg,
                            d['raw'], d['c1'], d['c2'], exact=exact)
        hero, vill, optout, seat = d['hero'], d['villain'], d['optout'], case['villain_seat']
        r0, r1 = (vill, hero) if seat == 0 else (hero, vill)
        done = 0
        for cp in meta['checkpoints']:
            cfr.run_gadget(hero, vill, optout, seat, cp - done)
            done = cp
            label = f"{case['file']} @ {cp} iters ({'exact' if exact else 'fast'})"
            _check_snapshot(cfr, d, f'it{cp}_', r0, r1, label, check)
            check(cfr.gadget_regret(), d[f'it{cp}_g_regret'], f"{label} gadget regret")


# ----------------------------------------------------------------------------
# mccfr -- contract:
#   Trainer(menu_mode, alpha, gamma)
#     .run_fixture(deals, fine, post, raw, fine_to_coarse, uniforms, start_iteration=0)
#        -> number of uniforms consumed (opponent sampling replays the stream
#           exactly like random.choices; buckets/ranks come from the fixture)
#     .checkpoint(db_path, iteration) -> rows written
#     .num_info_sets, .ev_sum, .ev_count
#   The checkpointed blueprint DB must equal the Python trainer's row for row:
#   every column byte-identical (keys, JSON text incl. float repr and dict
#   order, clocks) and the same metadata.
# ----------------------------------------------------------------------------
def _db_rows(path):
    import sqlite3
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT * FROM info_sets ORDER BY key").fetchall()
        meta = dict(con.execute("SELECT key, value FROM training_metadata").fetchall())
    finally:
        con.close()
    return rows, meta


@pytest.mark.parametrize('name', ['mccfr_control_g1', 'mccfr_capped_g1', 'mccfr_control_g2',
                                  'mccfr_capped_spr5_g1', 'mccfr_capped_pj_spr5_g1'])
def test_mccfr(name, tmp_path):
    component = 'mccfr_pj' if '_pj_' in name else 'mccfr_spr' if 'spr' in name else 'mccfr'
    core = need(component, f'{name}.json', f'{name}.db')
    with open(fixture_path(f'{name}.json')) as f:
        d = json.load(f)
    t = core.Trainer(d['mode'], d['alpha'], d['gamma'], key_format=d.get('key_format', 'v2'))
    rows = d['deals']
    used = t.run_fixture([r['cards'] for r in rows], [r['fine'] for r in rows],
                         [r['post'] for r in rows], [r['raw'] for r in rows],
                         d['fine_to_coarse'], d['uniforms'])
    assert used == len(d['uniforms']), f"consumed {used} uniforms, Python {len(d['uniforms'])}"
    assert t.num_info_sets == d['info_sets']
    assert (t.ev_sum, t.ev_count) == (d['ev_sum'], d['ev_count'])
    out = str(tmp_path / 'rust.db')
    assert t.checkpoint(out, d['iterations'] - 1) == d['info_sets']
    got, got_meta = _db_rows(out)
    want, want_meta = _db_rows(fixture_path(d['db']))
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g == w, f"row differs:\n rust   {g}\n python {w}"
    for k in ('total_iterations', 'alpha', 'gamma', 'menu_mode', 'ev_sum', 'ev_count'):
        assert got_meta[k] == want_meta[k], f"metadata {k}: {got_meta[k]} != {want_meta[k]}"
    assert got_meta.get('key_format') == want_meta.get('key_format')
    if d.get('key_format', 'v2') != 'v2':
        assert any(re.search(r'_(flop|turn)_r[0-4]_', r[0]) for r in got), 'spr5 run produced no SPR keys'
    if d['mode'] == 'capped_pj':
        # The 5-bet jam: three preflop aggressions, then 'a'.
        assert any(re.match(r'pf_\d+_(ip|oop)_c?[smlx][sml]{2}a$', r[0]) for r in got), \
            'capped_pj run never reached the preflop cap jam'


# ----------------------------------------------------------------------------
# keys_spr -- contract: spr_bucket(key_format, street, street_pot) -> int|None ;
#   make_info_set_key(..., spr) carries it as an `r<b>` token on flop/turn keys
# ----------------------------------------------------------------------------
def test_keys_spr():
    core = need('keys_spr', 'keys_spr.json')
    with open(fixture_path('keys_spr.json')) as f:
        d = json.load(f)
    for c in d['cases']:
        assert core.spr_bucket(c['key_format'], c['street'], c['street_pot']) == c['spr'], c
        got = core.make_info_set_key(c['street'], c['position'], c['preflop_bucket'],
                                     c['strength'], c['pattern'], c['spr'])
        assert got == c['key'], c


# ----------------------------------------------------------------------------
# mccfr_merge -- contract: merge_round(global, workers, alpha, gamma) -> (after, touched)
#   global/after: {key: {legal, regret, strategy, visit_count, strategy_visit_count}};
#   workers: parallel_trainer worker results ({regret, strategy, legal}).
#   Values bit-identical. Per-key ACTION ORDER is not compared: Python builds
#   the merged dict from set iteration (hash-seed dependent).
# ----------------------------------------------------------------------------
def test_mccfr_merge():
    core = need('mccfr_merge', 'mccfr_merge.json')
    with open(fixture_path('mccfr_merge.json')) as f:
        d = json.load(f)
    for i, c in enumerate(d['cases']):
        after, touched = core.merge_round(c['global'], c['workers'], c['alpha'], c['gamma'])
        assert sorted(touched) == c['touched'], f"case {i}: touched"
        assert set(after) == set(c['after']), f"case {i}: keys"
        for k, w in c['after'].items():
            g = after[k]
            assert g['legal'] == w['legal'], f"case {i} {k}: legal"
            for col in ('regret', 'strategy'):
                assert g[col] == w[col], f"case {i} {k}: {col} {g[col]} != {w[col]}"
            for col in ('visit_count', 'strategy_visit_count'):
                assert g[col] == w[col], f"case {i} {k}: {col}"
