# backend/bot/src/storage/blueprint_db.py
import sqlite3
import json
import threading
import hashlib
import os
from pathlib import Path
from ..cfr.information_set import InformationSet


class BlueprintDB:
    def __init__(self, db_path, read_only=False, *, fallback_base_db=None):
        """
        read_only=True opens the file with SQLite's URI `mode=ro` flag: the
        connection can never write, so it is safe to point at a blueprint while
        a separate training process holds it open. Inference (API, bot) should
        always use read_only=True; training uses the default read/write mode.

        Connections are THREAD-LOCAL. The Flask API serves requests from a thread
        pool, and a single sqlite3.Connection shared across threads is NOT safe
        for concurrent use (interleaved cursors / "recursive use of cursors", and
        result corruption under load) even for read-only queries. Each thread
        lazily gets its own connection via the `conn` property; every connection
        opened is tracked so `close()` can dispose of all of them.
        """
        self.db_path = str(db_path)
        self.read_only = read_only
        if fallback_base_db is not None and not read_only:
            raise ValueError("DBR fallback is for read-only policy access only")
        self._fallback_path = fallback_base_db
        self._fallback_db = None
        self._policy_checked = False
        self._policy_lock = threading.Lock()
        self._dbr_policy = False
        self._local = threading.local()
        self._all_conns = []
        self._all_conns_lock = threading.Lock()

        if not read_only:
            # Eagerly open the writable connection (and create the schema) in the
            # calling thread. Training is single-threaded, so this thread's
            # thread-local connection is the sole writer.
            self._open_connection()

    def _open_connection(self):
        """Open a fresh connection for the CURRENT thread, register it for
        close(), apply pragmas + ensure the schema for a writable handle, and
        return it. Called lazily by the `conn` property the first time each
        thread touches the DB."""
        if self.read_only:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        else:
            # check_same_thread=False only so close() can run from another thread;
            # actual queries stay on the single (training) thread that opened it.
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # WAL mode: allows reads while a write is in progress
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        self._local.conn = conn
        with self._all_conns_lock:
            self._all_conns.append(conn)
        if not self.read_only:
            self._create_tables()
        return conn

    @property
    def conn(self):
        """This thread's connection, opened on first use in that thread."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open_connection()
        return conn

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
        # Migrate DBs created before the gamma-discount clock existed: add the two
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

    def _info_set_rows(self, info_sets_dict):
        return [
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

    def save_checkpoint(self, info_sets_dict, metadata):
        """Persist info sets AND metadata in a SINGLE atomic transaction.

        Writing the info sets and `total_iterations` in separate commits is a
        resume-corruption hazard: an interrupt (crash / Ctrl-C) between them
        would leave the info sets at iteration N but the counter at an earlier
        M, so a resume replays M..N into already-updated info sets and
        double-counts their regrets. `with self.conn` commits on success and
        rolls back on any exception (including KeyboardInterrupt), and SQLite's
        transaction is atomic at the DB level — so a checkpoint either lands
        completely or not at all. Use this instead of save_batch + set_metadata
        for training checkpoints.
        """
        rows = self._info_set_rows(info_sets_dict)
        meta_rows = [(k, json.dumps(v)) for k, v in metadata.items()]
        with self.conn:                       # atomic: one commit, rollback on error
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
            self.conn.executemany(
                "INSERT OR REPLACE INTO training_metadata (key, value) VALUES (?, ?)",
                meta_rows,
            )

    def validate_policy(self):
        """Bind DBR bot reads to their exact frozen fork source, once per handle.

        Set ALLIN_DBR_BASE_DB (or fallback_base_db) for serving AND evaluation.
        Metadata/raw SQL/resume remain raw; writable training is never affected.
        Opponent-role tables are checkpoints, not playable bot policies.
        """
        if not self.read_only or self._policy_checked:
            return
        with self._policy_lock:
            if self._policy_checked:
                return
            role = self.get_metadata('dbr_role')
            config = self.get_metadata('dbr')
            if role is None and config is None:
                self._policy_checked = True
                return
            if role != 'bot' or not isinstance(config, dict):
                raise ValueError('only DBR bot-role checkpoints can supply a policy')
            path = self._fallback_path or os.environ.get('ALLIN_DBR_BASE_DB')
            if not path:
                raise ValueError('DBR policy requires ALLIN_DBR_BASE_DB pointing to the frozen fork base')
            path = Path(path).resolve(strict=True)
            wal = Path(str(path) + '-wal')
            if wal.exists() and wal.stat().st_size:
                raise ValueError('DBR fallback base must be a quiescent snapshot without a nonempty WAL')
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest != config.get('base_sha256'):
                raise ValueError('DBR fallback base SHA256 differs from the fork source')
            base = BlueprintDB(path, read_only=True)
            try:
                if base.get_metadata('dbr') is not None or base.get_metadata('dbr_role') is not None:
                    raise ValueError('DBR fallback base must be an ordinary frozen blueprint')
                for field, default in [('key_format', 'v2'), ('menu_mode', 'control'),
                                       ('abstraction_id', None)]:
                    if base.get_metadata(field, default) != self.get_metadata(field, default):
                        raise ValueError(f'DBR fallback base {field} mismatch')
                if base.get_metadata('total_iterations') != self.get_metadata('fork_base_iterations'):
                    raise ValueError('DBR fallback base iteration mismatch')
            except Exception:
                base.close()
                raise
            self._fallback_db = base
            self._dbr_policy = True
            self._policy_checked = True

    @property
    def policy_fallback_path(self):
        """Validated physical source path for evaluation subprocesses."""
        self.validate_policy()
        return self._fallback_db.db_path if self._fallback_db is not None else None

    def policy_provenance(self):
        """Public-safe completion rule; raw average coverage must be measured separately."""
        self.validate_policy()
        if not self._dbr_policy:
            return None
        return {'emptyAverage': 'frozen_base',
                'baseSha256': self.get_metadata('dbr')['base_sha256'],
                'unavailableBaseAverage': 'untrained'}

    @staticmethod
    def _normalise_average(cumulative):
        # Use EVERY accumulated action: first-seen legal_actions can omit an
        # all-in added on a later short-stack visit to the same abstract key.
        total = sum(cumulative.values())
        if total > 1e-12:
            return {a: v / total for a, v in cumulative.items()}
        return None

    def get_average_strategy(self, key):
        """Effective policy, shared by serving and evaluation; missing = None.

        DBR empty averages use the validated fork base. Ordinary blueprints keep
        their historical uniform-on-empty behavior. Training state is untouched.
        """
        record = self.get_record(key)
        return record['strategy'] if record is not None else None

    def get_record(self, key):
        """Effective policy + candidate visits; DBR records identify its source."""
        self.validate_policy()
        row = self.conn.execute(
            "SELECT legal_actions, cumulative_strategy, visit_count, "
            "last_visited_iteration FROM info_sets WHERE key = ?", (key,),
        ).fetchone()
        if row is None:
            return None
        strategy = self._normalise_average(json.loads(row[1]))
        source = 'dbr_average'
        if strategy is None:
            if self._dbr_policy:
                base_row = self._fallback_db.conn.execute(
                    "SELECT cumulative_strategy FROM info_sets WHERE key = ?", (key,),
                ).fetchone()
                strategy = self._normalise_average(json.loads(base_row[0])) if base_row else None
                if strategy is None:
                    # Newly discovered keys may have no trained D policy. Use
                    # consumers' existing untrained-key handling, never uniform.
                    return None
                source = 'frozen_base'
            else:
                actions = json.loads(row[0])
                strategy = {a: 1.0 / len(actions) for a in actions}
        record = {
            "strategy": strategy,
            "legalActions": list(strategy.keys()),
            "visitCount": row[2],
            "lastVisitedIteration": row[3],
        }
        if self._dbr_policy:
            record['strategySource'] = source
        return record

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
        """Close every per-thread connection this DB opened."""
        with self._all_conns_lock:
            conns = list(self._all_conns)
            self._all_conns.clear()
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        self._local.conn = None
        if self._fallback_db is not None:
            self._fallback_db.close()


class FrozenBlueprint:
    """Memoizing read snapshot of a BlueprintDB's average strategy.

    The FIRST read of each info-set key is cached; all later reads return that same
    value. This makes a measurement that does MULTIPLE passes over the blueprint
    (e.g. building an exact-leaf reference and a bucketed leaf matrix in separate
    passes) internally CONSISTENT even when a training process is still checkpointing
    the same .db underneath -- without it, two passes seconds apart can read DIFFERENT
    blueprint values and silently disagree (a live-training artifact, not a real
    error). Use it for any OFFLINE measurement against a possibly-being-trained DB.

    Wraps only `get_average_strategy` (the sole method the leaf/subgame builders use);
    `db_menu_mode`/`postflop_menu_for` etc. read the underlying db directly. Not for
    training (which needs regrets), and unnecessary for live inference against a
    served, quiescent blueprint."""

    def __init__(self, db):
        self.db = db
        self._cache = {}

    def get_average_strategy(self, key):
        c = self._cache
        if key not in c:
            c[key] = self.db.get_average_strategy(key)
        return c[key]
