# backend/bot/src/storage/blueprint_db.py
import sqlite3
import json
from ..cfr.information_set import InformationSet


class BlueprintDB:
    def __init__(self, db_path, read_only=False):
        """
        read_only=True opens the file with SQLite's URI `mode=ro` flag: the
        connection can never write, so it is safe to point at a blueprint while
        a separate training process holds it open. Inference (API, bot) should
        always use read_only=True; training uses the default read/write mode.
        """
        self.db_path = str(db_path)
        self.read_only = read_only

        if read_only:
            # check_same_thread=False: the Flask API serves requests from a
            # thread pool. Read-only queries across threads are safe here.
            self.conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        else:
            self.conn = sqlite3.connect(self.db_path)
            # WAL mode: allows reads while a write is in progress
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS info_sets (
                key                     TEXT PRIMARY KEY,
                legal_actions           TEXT NOT NULL,
                cumulative_regrets      TEXT NOT NULL,
                cumulative_strategy     TEXT NOT NULL,
                visit_count             INTEGER NOT NULL DEFAULT 0,
                last_visited_iteration  INTEGER NOT NULL DEFAULT 0,
                strategy_visit_count    INTEGER NOT NULL DEFAULT 0,
                last_strategy_iteration INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS training_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        # Migrate DBs created before the DCFR gamma clock existed: add the two
        # columns if missing. Resuming such a run continues the gamma discount
        # from the correct iteration count instead of silently restarting it.
        for col in ("strategy_visit_count", "last_strategy_iteration"):
            try:
                self.conn.execute(
                    f"ALTER TABLE info_sets ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # column already exists
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
                info_set.strategy_visit_count,
                info_set.last_strategy_iteration,
            )
            for key, info_set in info_sets_dict.items()
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO info_sets
                (key, legal_actions, cumulative_regrets, cumulative_strategy,
                 visit_count, last_visited_iteration,
                 strategy_visit_count, last_strategy_iteration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def get_average_strategy(self, key):
        """
        Inference lookup: returns {action: probability} for one info set key,
        or None if the key was never trained on.

        Normalised over EVERY accumulated action. The stored `legal_actions` is
        only the first-seen action set and can omit actions a key picked up on
        later visits (e.g. 'allin' on a short-stack visit) — normalising over it
        would silently drop those actions from the exported strategy.
        """
        row = self.conn.execute(
            "SELECT legal_actions, cumulative_strategy FROM info_sets WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None

        cumulative_strategy = json.loads(row[1])
        total = sum(cumulative_strategy.values())
        if total > 1e-12:
            return {a: v / total for a, v in cumulative_strategy.items()}

        # No strategy mass accumulated — fall back to uniform over legal actions.
        legal_actions = json.loads(row[0])
        return {a: 1.0 / len(legal_actions) for a in legal_actions}

    def get_record(self, key):
        """
        Inference + UI lookup: full info-set record for one key, or None.
        Returns the normalised average strategy plus visit metadata.
        """
        row = self.conn.execute(
            "SELECT legal_actions, cumulative_strategy, visit_count, "
            "last_visited_iteration FROM info_sets WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None

        legal_actions = json.loads(row[0])
        cumulative_strategy = json.loads(row[1])
        # Normalise over every accumulated action (see get_average_strategy).
        total = sum(cumulative_strategy.values())
        if total > 1e-12:
            strategy = {a: v / total for a, v in cumulative_strategy.items()}
        else:
            strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}

        return {
            "strategy": strategy,
            "legalActions": list(strategy.keys()),
            "visitCount": row[2],
            "lastVisitedIteration": row[3],
        }

    def load_all_to_memory(self):
        """
        Resume path: load every row back into InformationSet objects so training
        can continue from where it left off.
        """
        rows = self.conn.execute(
            "SELECT key, legal_actions, cumulative_regrets, cumulative_strategy, "
            "visit_count, last_visited_iteration, "
            "strategy_visit_count, last_strategy_iteration FROM info_sets"
        ).fetchall()

        info_sets = {}
        for (key, legal_actions, cumulative_regrets, cumulative_strategy,
             visit_count, last_visited, strat_visit, last_strat) in rows:
            info_set = InformationSet()
            info_set.legal_actions = json.loads(legal_actions)
            info_set.cumulative_regrets = json.loads(cumulative_regrets)
            info_set.cumulative_strategy = json.loads(cumulative_strategy)
            info_set.visit_count = visit_count
            info_set.last_visited_iteration = last_visited
            info_set.strategy_visit_count = strat_visit
            info_set.last_strategy_iteration = last_strat
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
