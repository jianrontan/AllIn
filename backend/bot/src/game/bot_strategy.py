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

    def decide(self, info_set_key, legal_actions, public_state):
        dist = self._distribution(info_set_key, legal_actions)
        actions = list(dist.keys())
        weights = list(dist.values())
        return random.choices(actions, weights=weights)[0]

    def explain(self, info_set_key, legal_actions, public_state):
        return self._distribution(info_set_key, legal_actions)
