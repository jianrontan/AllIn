"""Zero-size anchor boundaries and the real GameSession -> strategy path."""
import pytest
from .test_custom_betting import GameSession, make_custom_action, BlueprintStrategy, _FakeDB
from src.cfr.translation import translate_bet, pseudo_harmonic_weight


@pytest.mark.parametrize('fraction', [.001, .1, .25, .33, .5])
def test_zero_anchor_boundaries(fraction):
    grid = [('s', .33), ('m', .66)]
    result = translate_bet(fraction, grid, zero_anchor=True)
    assert sum(w for _, w in result) == pytest.approx(1)
    if fraction < .33:
        assert result[0][0] is None
        assert result[0][1] == pytest.approx(pseudo_harmonic_weight(fraction, 0, .33))
    else:
        assert result == translate_bet(fraction, grid)


@pytest.mark.parametrize('enabled', [False, True])
def test_session_anchor_has_no_synthetic_blueprint_key(monkeypatch, enabled):
    monkeypatch.setenv('ALLIN_SUBMINIMUM_ANCHOR', '1' if enabled else '0')
    session = GameSession.new('tiny', 'p')
    session.apply_action(make_custom_action(False, 10))
    session.apply_action('call')
    session.data['human_seat'] = 1
    assert session.data['street'] == 1
    session.apply_action(make_custom_action(False, 2))
    state = session.bot_public_state()
    class DB(_FakeDB):
        def get_average_strategy(self, key):
            assert key is not None
            return super().get_average_strategy(key)
    bot = BlueprintStrategy(DB())
    dist = bot.explain(session.info_set_key(session.current_player()), session.legal_actions(), state)
    expected = .1 * (1 - pseudo_harmonic_weight(.1, 0, .33)) if enabled else .1
    assert dist['fold'] == pytest.approx(expected)
    assert 'None' not in session.data['bet_pattern']
    if enabled:
        restored = GameSession.from_dict(session.to_dict())
        assert restored.bot_public_state()['translation'] == state['translation']


def test_hero_model_tracks_the_served_zero_anchor():
    import numpy as np
    bot = BlueprintStrategy(_FakeDB())
    model = bot.hero_range_model_fn()
    weights = translate_bet(.1, [('s', .33), ('m', .66)], zero_anchor=True)
    legal = ('fold', 'call')
    row = model.translated('pf_0_0_ip_flop_s', legal, weights)
    expected = .1 * (1 - pseudo_harmonic_weight(.1, 0, .33))
    np.testing.assert_allclose(row, [expected, 1 - expected])


def test_zero_anchor_is_not_observed_as_a_human_shove(monkeypatch):
    from src.game.range_tracker import RangeTracker
    monkeypatch.setenv('ALLIN_SUBMINIMUM_ANCHOR', '1')
    bot = BlueprintStrategy(_FakeDB())
    session = GameSession.new('tiny', 'p', strategy_fn=bot.range_model_fn())
    session.apply_action(make_custom_action(False, 10))
    session.apply_action('call')
    session.data['human_seat'] = 1
    observed = []
    def observe(self, model, action, *args, **kwargs):
        observed.append((action, kwargs.get('blend')))
    monkeypatch.setattr(RangeTracker, 'observe', observe)
    session.apply_action(make_custom_action(False, 2))
    # The zero-size point describes our RESPONSE, not a human action.
    assert observed == [('bet_small', None)]


@pytest.mark.parametrize('amount', [2, 10])
def test_session_self_range_uses_the_served_blend(monkeypatch, amount):
    from src.game.range_tracker import RangeTracker
    monkeypatch.setenv('ALLIN_SUBMINIMUM_ANCHOR', '1')
    bot = BlueprintStrategy(_FakeDB())
    session = GameSession.new('tiny', 'p', strategy_fn=bot.range_model_fn(),
                              hero_strategy_fn=bot.hero_range_model_fn())
    session.apply_action(make_custom_action(False, 10))
    session.apply_action('call')
    session.data['human_seat'] = 1
    session.apply_action(make_custom_action(False, amount))
    rows = []
    def observe(self, model, action, street, position, pattern, legal, board, **kwargs):
        rows.append((action, model('pf_0_0_ip_flop_s', tuple(legal)), list(legal)))
    monkeypatch.setattr(RangeTracker, 'observe', observe)
    session.apply_action('call')
    action, row, legal = rows[0]
    assert action == 'call'
    expected = (.1 * (1 - pseudo_harmonic_weight(.1, 0, .33)) if amount == 2
                else .1 * pseudo_harmonic_weight(.5, .33, .66)
                + .3 * (1 - pseudo_harmonic_weight(.5, .33, .66)))
    assert row[legal.index('fold')] == pytest.approx(expected)


def test_hero_model_also_tracks_existing_between_size_translation():
    import numpy as np
    bot = BlueprintStrategy(_FakeDB())
    legal = ('fold', 'call')
    weights = translate_bet(.5, [('s', .33), ('m', .66)])
    row = bot.hero_range_model_fn().translated('pf_0_0_ip_flop_m', legal, weights)
    w = pseudo_harmonic_weight(.5, .33, .66)
    fold = .1 * w + .3 * (1 - w)
    np.testing.assert_allclose(row, [fold, 1 - fold])
