# backend/bot/src/game/game package
"""
Transport-agnostic game core for playing against the blueprint bot.

Nothing in this package imports Flask or any web framework — the API layer is
a thin adapter on top. This keeps the game logic reusable if the transport
later changes (e.g. WebSockets for live online play).
"""
