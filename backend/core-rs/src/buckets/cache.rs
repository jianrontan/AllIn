//! Bounded, thread-safe LRU cache for per-board river equities (the Rust
//! counterpart of `postflop_v2._RIVER_BOARD_CACHE`).
//!
//! Sharded (`SHARDS` independent `Mutex`-guarded LRUs, picked by key hash) so
//! many threads bucketing different boards rarely contend; each shard is an
//! O(1) intrusive doubly-linked LRU. Values are `Arc`s, so a hit is a refcount
//! bump and the expensive compute runs OUTSIDE any lock (two threads racing on
//! one board both compute it; the value is identical, the second insert wins).
//! Eviction is per shard (approximately global LRU), which is invisible to
//! callers because the cached value is a pure function of the key.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const SHARDS: usize = 16;
const NIL: usize = usize::MAX;

struct Node<V> {
    key: u64,
    val: Arc<V>,
    prev: usize,
    next: usize,
}

struct Lru<V> {
    cap: usize,
    map: HashMap<u64, usize>,
    nodes: Vec<Node<V>>,
    head: usize, // most recently used
    tail: usize, // least recently used
}

impl<V> Lru<V> {
    fn new(cap: usize) -> Self {
        Lru {
            cap,
            map: HashMap::new(),
            nodes: Vec::new(),
            head: NIL,
            tail: NIL,
        }
    }

    fn unlink(&mut self, i: usize) {
        let (p, n) = (self.nodes[i].prev, self.nodes[i].next);
        if p != NIL {
            self.nodes[p].next = n;
        } else {
            self.head = n;
        }
        if n != NIL {
            self.nodes[n].prev = p;
        } else {
            self.tail = p;
        }
    }

    fn push_front(&mut self, i: usize) {
        self.nodes[i].prev = NIL;
        self.nodes[i].next = self.head;
        if self.head != NIL {
            self.nodes[self.head].prev = i;
        }
        self.head = i;
        if self.tail == NIL {
            self.tail = i;
        }
    }

    fn get(&mut self, key: u64) -> Option<Arc<V>> {
        let i = *self.map.get(&key)?;
        self.unlink(i);
        self.push_front(i);
        Some(self.nodes[i].val.clone())
    }

    fn insert(&mut self, key: u64, val: Arc<V>) {
        if self.cap == 0 {
            return;
        }
        if let Some(&i) = self.map.get(&key) {
            self.nodes[i].val = val;
            self.unlink(i);
            self.push_front(i);
            return;
        }
        let i = if self.nodes.len() < self.cap {
            self.nodes.push(Node {
                key,
                val,
                prev: NIL,
                next: NIL,
            });
            self.nodes.len() - 1
        } else {
            // reuse the LRU slot
            let i = self.tail;
            self.unlink(i);
            let old = self.nodes[i].key;
            self.map.remove(&old);
            self.nodes[i].key = key;
            self.nodes[i].val = val;
            i
        };
        self.map.insert(key, i);
        self.push_front(i);
    }

    fn clear(&mut self) {
        self.map.clear();
        self.nodes.clear();
        self.head = NIL;
        self.tail = NIL;
    }
}

pub struct BoardCache<V> {
    shards: Vec<Mutex<Lru<V>>>,
    cap: usize,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

#[inline]
fn shard_of(key: u64) -> usize {
    // splitmix-style finalizer: spreads packed board keys over the shards
    let mut z = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^= z >> 29;
    (z % SHARDS as u64) as usize
}

impl<V> BoardCache<V> {
    /// `cap` = total entries (0 disables caching).
    pub fn new(cap: usize) -> Self {
        let per = if cap == 0 { 0 } else { cap.div_ceil(SHARDS) };
        BoardCache {
            shards: (0..SHARDS).map(|_| Mutex::new(Lru::new(per))).collect(),
            cap,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    fn lock(&self, key: u64) -> std::sync::MutexGuard<'_, Lru<V>> {
        // A poisoned shard only means another thread panicked mid-update of a
        // pure cache; recover the guard rather than propagating the panic.
        self.shards[shard_of(key)]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Cached value for `key`, computing (outside the lock) and inserting on a miss.
    pub fn get_or_compute<E>(
        &self,
        key: u64,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<Arc<V>, E> {
        if let Some(v) = self.lock(key).get(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(v);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let v = Arc::new(compute()?);
        self.lock(key).insert(key, v.clone());
        Ok(v)
    }

    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().unwrap_or_else(|e| e.into_inner()).map.len())
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        for s in &self.shards {
            s.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_evicts_least_recent() {
        let mut l: Lru<u32> = Lru::new(2);
        l.insert(1, Arc::new(10));
        l.insert(2, Arc::new(20));
        assert_eq!(l.get(1).as_deref(), Some(&10)); // 1 is now MRU
        l.insert(3, Arc::new(30)); // evicts 2
        assert!(l.get(2).is_none());
        assert_eq!(l.get(1).as_deref(), Some(&10));
        assert_eq!(l.get(3).as_deref(), Some(&30));
        assert_eq!(l.map.len(), 2);
    }

    #[test]
    fn cache_bounded_and_threadsafe() {
        let c: Arc<BoardCache<u64>> = Arc::new(BoardCache::new(64));
        std::thread::scope(|s| {
            for t in 0..8u64 {
                let c = c.clone();
                s.spawn(move || {
                    for k in 0..2000u64 {
                        let key = (k * 7 + t) % 500;
                        let v = c.get_or_compute::<()>(key, || Ok(key * 3)).unwrap();
                        assert_eq!(*v, key * 3);
                    }
                });
            }
        });
        assert!(c.len() <= 64 + SHARDS);
        let z: BoardCache<u64> = BoardCache::new(0);
        assert_eq!(*z.get_or_compute::<()>(5, || Ok(1)).unwrap(), 1);
        assert_eq!(z.len(), 0);
    }
}
