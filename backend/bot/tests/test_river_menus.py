"""Solver menus are distinct from trained blueprint menus."""
import numpy as np
import pytest
from .test_solve_control import _setup
from .test_lbr_engine_mirror import _NullDB
from .test_river_tree import _walk_terminals
from src.subgame.river_tree import build_river_tree, river_menu_profile, is_sized
from src.subgame.river_cfr import RiverCFR
from src.subgame.rust_river_cfr import RustRiverCFR
from src.subgame.river_subgame_solver import RiverSubgameSolver


@pytest.mark.parametrize('name', ['A', 'F', 'F_EXTENDED'])
def test_profile_tree_accounting_and_injected_edge(name):
    bets, raises = river_menu_profile(name)
    tree = build_river_tree(8, (192,192), menu=bets, raise_menu=raises)
    assert len(tree.decision_nodes) == {'A':210,'F':124,'F_EXTENDED':162}[name]
    assert tree.inject_realized_edge(tree.root, 'bet', 2.123456789) is not None
    for terminal, path in _walk_terminals(tree.root):
        assert terminal.final_pot == pytest.approx(sum(terminal.contrib))
        assert max(terminal.contrib) <= 196 + 1e-9
        # The injected realized edge consumes one aggression too.
        assert sum(is_sized(a) or a == 'allin' for a in path) <= 3


@pytest.mark.parametrize('name', ['F', 'F_EXTENDED'])
def test_native_solver_accepts_separate_bet_and_raise_menus(name):
    pytest.importorskip('allin_core')
    _, ba = _setup(30, (40,40))
    bets, raises = river_menu_profile(name)
    tree = build_river_tree(30, (40,40), menu=bets, raise_menu=raises)
    py, rs = RiverCFR(tree,ba), RustRiverCFR(tree,ba,exact=True)
    r0,r1 = np.random.default_rng(39).random((2,ba['H'])) + .1
    for solver in (py,rs):
        solver.run(r0,r1,iters=5)
    for node in tree.decision_nodes:
        np.testing.assert_array_equal(py.average_strategy(node.node_id),rs.average_strategy(node.node_id))


def test_live_profile_defaults_and_explicit_override(monkeypatch):
    monkeypatch.delenv('ALLIN_RIVER_MENU', raising=False)
    assert RiverSubgameSolver(_NullDB()).max_aggressions == 5
    monkeypatch.setenv('ALLIN_RIVER_MENU','F')
    candidate = RiverSubgameSolver(_NullDB())
    assert candidate.menu == river_menu_profile('F')[0]
    assert candidate.raise_menu == (.5,1.)
    assert candidate.max_aggressions == 3
    explicit = RiverSubgameSolver(_NullDB(),menu=(.75,))
    assert explicit.menu == explicit.raise_menu == (.75,)
    assert explicit.max_aggressions == 5
    monkeypatch.setenv('ALLIN_RIVER_MENU','typo')
    with pytest.raises(ValueError,match='Unknown river menu'):
        RiverSubgameSolver(_NullDB())


def test_zero_aggression_cap_has_checks_only():
    tree = build_river_tree(20,(90,90),max_aggressions=0)
    assert tree.root.actions == ['check']
    assert tree.root.children[0].actions == ['check']


@pytest.mark.parametrize('name', ['F', 'F_EXTENDED'])
def test_candidate_gadget_safety_battery(name, monkeypatch):
    pytest.importorskip('allin_core')
    monkeypatch.setenv('ALLIN_RIVER_BACKEND', 'rust')
    from .test_safe_river_gadget import run
    assert run(menu_name=name)


def test_match_views_are_synchronized_and_reproducible():
    import importlib.util
    import random
    import sys
    from pathlib import Path
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('river_menu_match_test',scripts/'river_menu_match.py')
        runner=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(scripts))
    from src.game.bot_strategy import BlueprintStrategy
    class DB(_NullDB):
        def get_metadata(self, key, default=None):
            return default
    db=DB()
    bots=[BlueprintStrategy(db),BlueprintStrategy(db)]
    before=random.getstate()
    first=runner.play_hand(db,bots,39,0)
    assert random.getstate()==before
    second=runner.play_hand(db,bots,39,0)
    assert first==second
    assert first['river_range'] is None
    assert first['seat_of_A']==0
    assert runner.play_hand(db,bots,39,1)['seat_of_A']==1


def test_native_zero_cap_matches_python():
    core=pytest.importorskip('allin_core')
    decisions=[n for n in core.river_tree_nodes(20.,90.,[.5,.75,1.,1.5],0) if not n['terminal']]
    assert len(decisions)==2
    assert all(n['actions']==['check'] for n in decisions)


def test_checkpoint_validation_rejects_policy_and_hand_mismatch(tmp_path):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('checkpoint_runner',scripts/'river_menu_match.py')
        runner=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(scripts))
    config=dict(db='snapshot.db',a='F',b='A',hands=4,iters=275,budget=24.,seed=101,
                depth_a=None,depth_b=None)
    sources={'tree':'same'}
    identity={'version':1,'settings':{'anchor':'0'}}
    saved=dict(config=config.copy(),source_sha256=sources.copy(),policy_identity=identity,completed=1,
               records=[{'seat_of_A':0}])
    path=tmp_path/'partial.json'
    path.write_text(json.dumps(saved))
    assert runner.load_checkpoint(path,config,sources,identity)['completed']==1
    with pytest.raises(ValueError,match='source hashes'):
        runner.load_checkpoint(path,config,{'tree':'changed'},identity)
    with pytest.raises(ValueError,match='seed'):
        runner.load_checkpoint(path,{**config,'seed':102},sources,identity)
    with pytest.raises(ValueError,match='policy identity'):
        runner.load_checkpoint(path,config,sources,{'version':1,'settings':{'anchor':'1'}})
    with pytest.raises(ValueError,match='policy identity'):
        runner.load_checkpoint(path,config,sources)
    saved['completed']=2
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='hand count'):
        runner.load_checkpoint(path,config,sources,identity)
