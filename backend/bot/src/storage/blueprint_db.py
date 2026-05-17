# backend/bot/src/storage/blueprint_db.py
import sqlite3
import json
from ..cfr.information_set import InformationSet


class BlueprintDB:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        # WAL mode: allows reads while a write is in progress
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS info_sets (
                key                    TEXT PRIMARY KEY,
                legal_actions          TEXT NOT NULL,
                cumulative_regrets     TEXT NOT NULL,
                cumulative_strategy    TEXT NOT NULL,
                visit_count            INTEGER NOT NULL DEFAULT 0,
                last_visited_iteration INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS training_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def save_batch(self, info_sets_dict):
        """Bulk upsert all info sets — used for checkpointing during training."""
        rows = [
            (
                key,
                json.dumps(info_set.legal_actions),
                json.dumps(info_set.cumulative_regrets),
                json.dumps(info_set.cumulative_strategy),
                info_set.visit_count,
                info_set.last_visited_iteration,
            )
            for key, info_set in info_sets_dict.items()
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO info_sets
                (key, legal_actions, cumulative_regrets, cumulative_strategy,
                 visit_count, last_visited_iteration)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def get_average_strategy(self, key):
        """
        Inference lookup: returns {action: probability} for one info set key,
        or None if the key was never trained on.
        """
        row = self.conn.execute(
            "SELECT legal_actions, cumulative_strategy FROM info_sets WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None

        legal_actions = json.loads(row[0])
        cumulative_strategy = json.loads(row[1])
        total = sum(cumulative_strategy.get(a, 0.0) for a in legal_actions)

        if total > 1e-12:
            return {a: cumulative_strategy.get(a, 0.0) / total for a in legal_actions}
        return {a: 1.0 / len(legal_actions) for a in legal_actions}

    def load_all_to_memory(self):
        """
        Resume path: load every row back into InformationSet objects so training
        can continue from where it left off.
        """
        rows = self.conn.execute(
            "SELECT key, legal_actions, cumulative_regrets, cumulative_strategy, "
            "visit_count, last_visited_iteration FROM info_sets"
        ).fetchall()

        info_sets = {}
        for key, legal_actions, cumulative_regrets, cumulative_strategy, visit_count, last_visited in rows:
            info_set = InformationSet()
            info_set.legal_actions = json.loads(legal_actions)
            info_set.cumulative_regrets = json.loads(cumulative_regrets)
            info_set.cumulative_strategy = json.loads(cumulative_strategy)
            info_set.visit_count = visit_count
            info_set.last_visited_iteration = last_visited
            info_sets[key] = info_set

        return info_sets

    def get_metadata(self, key, default=None):
        row = self.conn.execute(
            "SELECT value FROM training_metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def set_metadata(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO training_metadata (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
