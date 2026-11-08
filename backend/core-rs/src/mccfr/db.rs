//! Blueprint SQLite I/O with the exact schema and row encoding of
//! `storage/blueprint_db.py`, so serving reads a Rust-trained blueprint
//! unchanged and Rust can resume a Python-trained one.

use rusqlite::{Connection, OpenFlags, params};
use serde_json::Value;

use super::pyjson::{dumps_list, dumps_map};
use super::table::{InfoSet, Legal, id_action, id_from_name, key_string, parse_key};
use super::trainer::Table;

const SCHEMA: &str = "
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
);";

fn e(x: rusqlite::Error) -> String {
    format!("sqlite: {x}")
}

/// The bundled SQLite is built with URI filenames enabled globally, so a path
/// like `file:x.db?mode=memory` would be parsed as a URI even without
/// SQLITE_OPEN_URI. Prefixing `./` makes any relative path a plain file name
/// (absolute paths start with `/` and are never URIs). Matches Python's
/// `sqlite3.connect(path)`.
fn plain_path(path: &str) -> String {
    if path.starts_with('/') {
        path.to_string()
    } else {
        format!("./{path}")
    }
}

pub struct BlueprintDb {
    conn: Connection,
}

impl BlueprintDb {
    /// Open (creating if needed) a writable blueprint, WAL + synchronous=NORMAL.
    pub fn open(path: &str) -> Result<BlueprintDb, String> {
        // Explicit flags: no SQLITE_OPEN_URI, so a "file:...?mode=memory" string
        // is a plain file name (like Python's sqlite3.connect), never a URI.
        let conn = Connection::open_with_flags(
            plain_path(path),
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(e)?;
        conn.pragma_update(None, "journal_mode", "WAL").map_err(e)?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(e)?;
        conn.execute_batch(SCHEMA).map_err(e)?;
        for col in ["strategy_visit_count", "last_strategy_iteration"] {
            // Old DBs lack these columns; "duplicate column" is the normal case.
            let _ = conn.execute(
                &format!("ALTER TABLE info_sets ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0"),
                [],
            );
        }
        Ok(BlueprintDb { conn })
    }

    /// Read-only handle (never writes; safe on a file a trainer holds open).
    pub fn open_read_only(path: &str) -> Result<BlueprintDb, String> {
        let conn = Connection::open_with_flags(
            plain_path(path),
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(e)?;
        Ok(BlueprintDb { conn })
    }

    /// Info sets `keys` + metadata in ONE transaction (`save_checkpoint`).
    /// `meta` values are JSON text (e.g. `"1.5"`, `"\"control\""`).
    pub fn save_checkpoint<'k>(
        &mut self,
        table: &Table,
        keys: impl Iterator<Item = &'k u64>,
        meta: &[(&str, String)],
    ) -> Result<usize, String> {
        let tx = self.conn.transaction().map_err(e)?;
        let mut n = 0;
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT OR REPLACE INTO info_sets
                        (key, legal_actions, cumulative_regrets, cumulative_strategy,
                         visit_count, last_visited_iteration,
                         strategy_visit_count, last_strategy_iteration)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                )
                .map_err(e)?;
            for k in keys {
                let Some(is) = table.get(k) else { continue };
                // A NaN/inf would make the blueprint unresumable (and means the
                // run is already corrupt): fail the checkpoint instead. The
                // transaction rolls back, so the DB keeps the last good state.
                if is
                    .regrets
                    .iter()
                    .chain(is.strategy.iter())
                    .any(|(_, v)| !v.is_finite())
                {
                    return Err(format!("non-finite value in info set {}", key_string(*k)));
                }
                let (legal, regrets, strategy) = encode_row(is);
                stmt.execute(params![
                    key_string(*k),
                    legal,
                    regrets,
                    strategy,
                    is.visit_count,
                    is.last_visited,
                    is.strategy_visit_count,
                    is.last_strategy
                ])
                .map_err(e)?;
                n += 1;
            }
            let mut ms = tx
                .prepare_cached(
                    "INSERT OR REPLACE INTO training_metadata (key, value) VALUES (?1, ?2)",
                )
                .map_err(e)?;
            for (k, v) in meta {
                ms.execute(params![k, v]).map_err(e)?;
            }
        }
        tx.commit().map_err(e)?;
        Ok(n)
    }

    /// JSON-decoded metadata value, or None.
    pub fn metadata(&self, key: &str) -> Result<Option<Value>, String> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT value FROM training_metadata WHERE key = ?1")
            .map_err(e)?;
        let mut rows = stmt.query(params![key]).map_err(e)?;
        match rows.next().map_err(e)? {
            None => Ok(None),
            Some(r) => {
                let s: String = r.get(0).map_err(e)?;
                serde_json::from_str(&s)
                    .map(Some)
                    .map_err(|x| format!("metadata {key:?}: {x}"))
            }
        }
    }

    /// Every info set (`load_all_to_memory`).
    pub fn load_all(&self) -> Result<Table, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT key, legal_actions, cumulative_regrets, cumulative_strategy,
                        visit_count, last_visited_iteration,
                        strategy_visit_count, last_strategy_iteration FROM info_sets",
            )
            .map_err(e)?;
        let mut rows = stmt.query([]).map_err(e)?;
        let mut table = Table::default();
        while let Some(r) = rows.next().map_err(e)? {
            let key: String = r.get(0).map_err(e)?;
            let k = parse_key(&key)?;
            let mut is = InfoSet {
                visit_count: r.get(4).map_err(e)?,
                last_visited: r.get(5).map_err(e)?,
                strategy_visit_count: r.get(6).map_err(e)?,
                last_strategy: r.get(7).map_err(e)?,
                ..InfoSet::default()
            };
            let ctx = |what: &str, x: String| format!("row {key:?} {what}: {x}");
            let legal: Vec<String> = serde_json::from_str(&r.get::<_, String>(1).map_err(e)?)
                .map_err(|x| ctx("legal_actions", x.to_string()))?;
            let ids = legal
                .iter()
                .map(|s| id_from_name(s))
                .collect::<Result<Vec<u8>, String>>()
                .map_err(|x| ctx("legal_actions", x))?;
            is.legal = Legal::from_ids(&ids).map_err(|x| ctx("legal_actions", x))?;
            for (col, map) in [(2usize, &mut is.regrets), (3, &mut is.strategy)] {
                let text: String = r.get(col).map_err(e)?;
                let obj: serde_json::Map<String, Value> =
                    serde_json::from_str(&text).map_err(|x| ctx("map", x.to_string()))?;
                for (name, v) in obj {
                    let v = v
                        .as_f64()
                        .ok_or_else(|| ctx("map", format!("non-numeric value for {name}")))?;
                    map.set(id_from_name(&name).map_err(|x| ctx("map", x))?, v)
                        .map_err(|x| ctx("map", x))?;
                }
            }
            table.insert(k, is);
        }
        Ok(table)
    }
}

fn encode_row(is: &InfoSet) -> (String, String, String) {
    let names = |ids: &[u8]| -> Vec<String> { ids.iter().map(|&a| id_action(a).name()).collect() };
    let legal = names(is.legal.ids());
    let rn = names(is.regrets.ids());
    let sn = names(is.strategy.ids());
    (
        dumps_list(legal.iter().map(|s| s.as_str())),
        dumps_map(
            rn.iter()
                .map(|s| s.as_str())
                .zip(is.regrets.iter().map(|(_, v)| v)),
        ),
        dumps_map(
            sn.iter()
                .map(|s| s.as_str())
                .zip(is.strategy.iter().map(|(_, v)| v)),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mccfr::table::{Pattern, pack_key};

    #[test]
    fn checkpoint_round_trips() {
        let dir = std::env::temp_dir().join(format!("allin_db_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bp.db");
        let _ = std::fs::remove_file(&path);
        let p = path.to_str().unwrap();
        let mut t = Table::default();
        let k = pack_key(2, 1, 7, 13, Pattern::parse("ks").unwrap()).unwrap();
        let mut is = InfoSet::default();
        is.legal = Legal::from_ids(&[2, 1, 3]).unwrap();
        is.regrets.set(1, 0.1 + 0.2).unwrap();
        is.regrets.set(2, 1e-17).unwrap();
        is.strategy.set(3, 12345.678).unwrap();
        is.visit_count = 4;
        is.last_visited = 99;
        t.insert(k, is);
        let mut db = BlueprintDb::open(p).unwrap();
        db.save_checkpoint(
            &t,
            t.keys(),
            &[
                ("total_iterations", "100".into()),
                ("menu_mode", "\"control\"".into()),
            ],
        )
        .unwrap();
        let back = BlueprintDb::open_read_only(p).unwrap();
        assert_eq!(back.load_all().unwrap().get(&k), Some(&is));
        assert_eq!(
            back.metadata("total_iterations").unwrap(),
            Some(Value::from(100))
        );
        assert_eq!(back.metadata("missing").unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
