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
import threading
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
        # Per-decision diagnostics for the optional "what is the bot thinking"
        # debug overlay, populated by decide() and read by advance_bot_turns.
        # None for a plain blueprint lookup (the info-set key + strategy already
        # tell the whole story); the river solver fills it with solve details.
        #
        # THREAD-LOCAL: the deployed BOT / EXPLORER_BOT are MODULE-LEVEL
        # singletons serving multiple Flask worker threads concurrently. A naive
        # `self.last_debug = ...` lets thread B overwrite A's debug between A's
        # decide() returning and advance_bot_turns reading it → A's session log
        # gets B's data. The thread-local store gives each thread its own
        # last_debug; reads default to None if the current thread never wrote.
        self._tls = threading.local()

    @property
    def last_debug(self):
        return getattr(self._tls, 'last_debug', None)

    @last_debug.setter
    def last_debug(self, value):
        self._tls.last_debug = value

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
        # Untrained key: fall back to PASSIVE actions only (check/call/fold), never
        # uniform over the full legal set. Live play uncaps re-raises
        # (GameSession passes max_raises_per_street=inf), so at a beyond-cap node
        # legal_actions includes sized raises + all-in; uniform over those would make
        # the bot stray-raise/jam from a key it never trained (the BUG-011 failure
        # mode). The bot must never PROPOSE an untrained aggressive size. A faced
        # all-in is already handled upstream by the near-terminal guard; this passive
        # fallback is the non-jam deep-raise stopgap until the Phase-4 deep-raise
        # solver. (Falls through to the full legal set only if no passive action is
        # legal -- a degenerate node that shouldn't occur in normal play.)
        passive = [a for a in legal_actions if a in ('check', 'call', 'fold')]
        pool = passive or legal_actions
        n = len(pool)
        return {a: 1.0 / n for a in pool}

    def _blend_lookup(self, info_set_key, legal_actions):
        """Restricted blueprint dist for a translation bracket, or {} when the key
        is UNTRAINED. Unlike _distribution (which returns uniform for an unknown
        key, the right default for a single lookup), this returns {} so blend()
        can route an untrained bracket's weight to fold rather than uniform --
        otherwise a too-big open's overflow bracket dilutes to a random response."""
        if not legal_actions:
            return {}
        stored = self.db.get_average_strategy(info_set_key) if self.db else None
        if stored:
            weights = {a: max(0.0, stored.get(a, 0.0)) for a in legal_actions}
            total = sum(weights.values())
            if total > 1e-12:
                return {a: w / total for a, w in weights.items()}
        return {}

    def _state_distribution(self, info_set_key, legal_actions, public_state):
        """Distribution accounting for action translation. When the opponent
        just made an off-grid bet, `public_state['translation']` carries the
        bracketing blueprint keys + pseudo-harmonic weights; blend their
        responses (untrained brackets fold, see translation.blend). Otherwise it
        is the plain single-key lookup."""
        trans = (public_state or {}).get('translation')
        if trans:
            blended = translation.blend(
                trans, lambda key: self._blend_lookup(key, legal_actions))
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

    # Runouts for the flop equity estimate on the fallback path. The tracker's
    # default (120) is fine for the UI "bot's read" but too noisy/biased to base
    # a fold/raise decision on (~1.5 equity-pts std + a ~1-pt low-n mean bias;
    # measured). The fallback fires rarely (only when confidence has collapsed),
    # so a high count here is cheap. River/turn are exact regardless of this.
    EQUITY_RUNOUTS = 1000

    def decide(self, info_set_key, legal_actions, public_state):
        tracker = (public_state or {}).get('opp_range')
        hole = (public_state or {}).get('hole_cards')
        if tracker is None or hole is None or tracker.confidence >= self.CONFIDENCE_THRESHOLD:
            return super().decide(info_set_key, legal_actions, public_state)

        board = public_state.get('community', []) or []
        eq = tracker.hero_equity(hole, board, n_runouts=self.EQUITY_RUNOUTS)
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

        # Absolute last resort prefers a PASSIVE action over `legal[0]` (which could
        # be an aggressive size): the equity fallback must never emit an untrained
        # raise/jam either (the BUG-011 class). legal[0] is only reached at a
        # degenerate node with no check/call/fold legal, which doesn't arise in normal
        # play. (This strategy isn't the deployed one -- RiverSubgameSolver is -- but
        # keep it safe in case it's ever served directly.)
        if to_call <= 0:                                   # no bet to face
            if eq >= 0.62:
                a = first('bet_medium', 'bet_small', 'bet_large')
                if a:
                    return a
            return first('check', 'call', 'fold') or legal[0]

        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
        if eq < pot_odds:
            return first('fold', 'check', 'call') or legal[0]
        if eq >= 0.75:
            a = first('raise_medium', 'raise_small', 'raise_large')
            if a:
                return a
        return first('call', 'check', 'fold') or legal[0]
