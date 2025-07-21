# backend/bot/src/subgame/player_blueprint_adapter.py
class PlayerBlueprintAdapter:
    """Adapter to make player.py compatible with confidence detection"""

    def __init__(self, player):
        self.info_sets = player.info_sets
        self.total_training_iterations = player.total_training_iterations
