# backend/bot/src/game/bot_strategy.py
"""
The bot's decision-making, behind a swappable interface.

`BotStrategy.decide()` takes:
  * info_set_key  — the bucketed key identifying the situation
  * legal_actions — the CFR actions actually available right now
  * public_state  — full public game state (pot, stacks, board, street, ...)

`public_state` is intentionally richer than the bucketed key. The blueprint
only needs the key, but a future SubgameSolvingStrategy needs real pot/stack/
board values, so it is passed in from day one. Adding the smarter bot later
means writing a new class here — nothing else changes.
"""
import random
from abc import ABC, abstractmethod

from ..cfr import translation


class BotStrategy(ABC):
    @abstractmethod
    def decide(self, info_set_key, legal_actions, public_state):
        """Return one action from legal_actions."""

    def explain(self, info_set_key, legal_actions, public_state):
        """
        Optional: return {action: probability} over legal_actions for UI
        ("show me what the bot was thinking"). Default: uniform.
        """
        n = len(legal_actions)
        return {a: 1.0 / n for a in legal_actions} if n else {}


class BlueprintStrategy(BotStrategy):
    """Looks the situation up in the trained blueprint DB."""

    def __init__(self, blueprint_db):
        self.db = blueprint_db

    def _distribution(self, info_set_key, legal_actions):
        """
        Map the blueprint's stored strategy onto the live legal actions.

        The blueprint may have been trained with a slightly different legal-
        action set for a key, so we re-restrict to the current legal_actions
        and renormalise. Unknown key or zero mass -> uniform.
        """
        if not legal_actions:
            return {}
        stored = self.db.get_average_strategy(info_set_key) if self.db else None
        if stored:
            weights = {a: max(0.0, stored.get(a, 0.0)) for a in legal_actions}
            total = sum(weights.values())
            if total > 1e-12:
                return {a: w / total for a, w in weights.items()}
        n = len(legal_actions)
        return {a: 1.0 / n for a in legal_actions}

    def _state_distribution(self, info_set_key, legal_actions, public_state):
        """Distribution accounting for action translation. When the opponent
        just made an off-grid bet, `public_state['translation']` carries the
        bracketing blueprint keys + pseudo-harmonic weights; blend their
        responses. Otherwise it is the plain single-key lookup."""
        trans = (public_state or {}).get('translation')
        if trans:
            blended = translation.blend(
                trans, lambda key: self._distribution(key, legal_actions))
            if blended:
                return blended
        return self._distribution(info_set_key, legal_actions)

    def decide(self, info_set_key, legal_actions, public_state):
        dist = self._state_distribution(info_set_key, legal_actions, public_state)
        actions = list(dist.keys())
        weights = list(dist.values())
        return random.choices(actions, weights=weights)[0]

    def explain(self, info_set_key, legal_actions, public_state):
        return self._state_distribution(info_set_key, legal_actions, public_state)

    def range_model_fn(self):
        """Return a strategy_fn(key, legal)->np.ndarray for the opponent range
        tracker: the blueprint's average strategy restricted to `legal` and
        renormalised (uniform if the key is unknown). This is the opponent model
        the tracker assumes (opponent plays the blueprint); the tracker's
        confidence score guards against that assumption being wrong."""
        import numpy as np

        def fn(key, legal):
            stored = self.db.get_average_strategy(key) if self.db else None
            n = len(legal)
            if stored:
                w = np.array([max(0.0, stored.get(a, 0.0)) for a in legal])
                t = w.sum()
                if t > 1e-12:
                    return w / t
            return np.ones(n) / n
        return fn


class ConfidenceAwareStrategy(BlueprintStrategy):
    """
    Plays the blueprint while the opponent looks like the blueprint, and falls
    back to a direct equity-vs-range decision when the range tracker's confidence
    collapses (the opponent is playing off-model, so the blueprint's "opponent =
    blueprint" assumption -- and thus its balance -- no longer holds).

    This is the Phase-3 consumer of the hand-level range tracker. It needs three
    things from public_state that the plain blueprint ignores: the bot's own
    `hole_cards`, the live `opp_range` tracker, and `to_call`/`pot` for pot odds.
    When any are missing (or confidence is high) it is exactly BlueprintStrategy.

    NOTE: the equity fallback policy below is a deliberately simple v1 -- a
    pot-odds/equity rule, not a solve. It is the safety net for "opponent is
    doing something the blueprint never expected"; the principled replacement is
    the Phase-4 river subgame solver, which consumes this same tracked range.
    """

    # Confidence below this => stop trusting the blueprint's balance. Set low so
    # the fallback only fires when the opponent is clearly off-model (a few
    # near-impossible actions drive confidence well under 0.1 in practice).
    CONFIDENCE_THRESHOLD = 0.15

    def decide(self, info_set_key, legal_actions, public_state):
        tracker = (public_state or {}).get('opp_range')
        hole = (public_state or {}).get('hole_cards')
        if tracker is None or hole is None or tracker.confidence >= self.CONFIDENCE_THRESHOLD:
            return super().decide(info_set_key, legal_actions, public_state)

        board = public_state.get('community', []) or []
        eq = tracker.hero_equity(hole, board)
        to_call = public_state.get('to_call', 0) or 0
        pot = public_state.get('pot', 0) or 0
        return self._equity_action(eq, to_call, pot, legal_actions)

    @staticmethod
    def _equity_action(eq, to_call, pot, legal):
        """Map (equity, pot odds) to a legal action. Value-bet/raise when well
        ahead, fold when below pot odds, otherwise check/call."""
        def first(*opts):
            for a in opts:
                if a in legal:
                    return a
            return None

        if to_call <= 0:                                   # no bet to face
            if eq >= 0.62:
                a = first('bet_medium', 'bet_small', 'bet_large')
                if a:
                    return a
            return first('check') or legal[0]

        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
        if eq < pot_odds:
            return first('fold') or first('check') or legal[0]
        if eq >= 0.75:
            a = first('raise_medium', 'raise_small', 'raise_large')
            if a:
                return a
        return first('call') or first('check') or legal[0]
