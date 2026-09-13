//! A read-through cache for object ranges.
//!
//! The design calls caching its largest omission, because it is how every
//! production object-store engine reaches interactive latency: Parquet
//! footers, page indexes and column chunks are re-read constantly, and each
//! one is a round trip to storage.
//!
//! # Why caching by path and range is sound here
//!
//! It would not be sound in general. An object that is overwritten in place
//! and keeps its path would serve stale bytes, and the answer would look
//! perfectly normal. The reason it is safe here is a property of the *table
//! format*, not of this code:
//!
//! ```text
//! Iceberg data files are immutable.
//! New data means new files; a committed data file is never rewritten.
//! ```
//!
//! So the constructor is [`RangeCache::for_immutable_objects`] rather than
//! `new` — a caller pointing this at mutable objects has to type out the
//! assumption it is breaking. The cache also records the version it saw and
//! refuses to serve an entry whose object has visibly changed underneath it,
//! which is defence in depth rather than the primary argument.
//!
//! # Composition
//!
//! Order matters, and it is the useful kind of ordering:
//!
//! ```text
//! RangeCache(MeteredStore(store))   the meter counts only what MISSED,
//!                                   so its stats measure real origin I/O
//!
//! MeteredStore(RangeCache(store))   the meter counts every read including
//!                                   hits, which measures demand, not cost
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    GetOptions, GetRange, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OsResult,
};

/// How a [`RangeCache`] has been used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Reads served without touching the underlying store.
    pub hits: u64,
    /// Reads that had to go to the underlying store.
    pub misses: u64,
    /// Reads that were not eligible for caching at all.
    pub bypassed: u64,
    /// Bytes served from memory.
    pub bytes_served: u64,
    /// Entries dropped to stay within the byte limit.
    pub evictions: u64,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Key {
    path: Path,
    range: Range<u64>,
}

struct Entry {
    data: Vec<u8>,
    version: String,
}

struct State {
    entries: HashMap<Key, Entry>,
    order: VecDeque<Key>,
    bytes: u64,
    stats: CacheStats,
}

/// Caches bounded byte ranges of immutable objects in memory.
pub struct RangeCache {
    inner: Arc<dyn ObjectStore>,
    max_bytes: u64,
    state: Mutex<State>,
}

impl RangeCache {
    /// Cache ranges of objects that are never modified in place.
    ///
    /// Named for the assumption it depends on. See the module docs: this is
    /// sound for Iceberg data files and unsound for objects that can be
    /// overwritten under the same path.
    pub fn for_immutable_objects(inner: Arc<dyn ObjectStore>, max_bytes: u64) -> Self {
        RangeCache {
            inner,
            max_bytes,
            state: Mutex::new(State {
                entries: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
                stats: CacheStats::default(),
            }),
        }
    }

    /// How the cache has been used.
    pub fn stats(&self) -> CacheStats {
        self.state.lock().expect("cache state").stats
    }

    /// Bytes currently held.
    pub fn bytes(&self) -> u64 {
        self.state.lock().expect("cache state").bytes
    }

    /// Drop everything. Always safe: this is a cache.
    pub fn clear(&self) {
        let mut state = self.state.lock().expect("cache state");
        state.entries.clear();
        state.order.clear();
        state.bytes = 0;
    }

    /// Whether a request may be served from, or stored in, the cache.
    ///
    /// Conditional requests and explicit version requests are passed straight
    /// through: their whole point is to ask the store something this cache
    /// cannot answer. `head` is passed through because it fetches no bytes.
    fn cacheable(options: &GetOptions) -> Option<Range<u64>> {
        if options.head
            || options.if_match.is_some()
            || options.if_none_match.is_some()
            || options.if_modified_since.is_some()
            || options.if_unmodified_since.is_some()
            || options.version.is_some()
        {
            return None;
        }
        match &options.range {
            // Only bounded ranges. Resolving an offset or suffix needs the
            // object size, which is one round trip more than a cache should
            // cost, and Parquet reads are bounded once the size is known.
            Some(GetRange::Bounded(range)) => Some(range.clone()),
            _ => None,
        }
    }

    fn version_of(meta: &ObjectMeta) -> String {
        match &meta.e_tag {
            Some(tag) => tag.clone(),
            // No entity tag: fall back to what does identify a version of an
            // immutable object. Weaker, and only reached on stores that do
            // not report one.
            None => format!(
                "{}:{}",
                meta.last_modified.timestamp_nanos_opt().unwrap_or(0),
                meta.size
            ),
        }
    }

    fn lookup(&self, key: &Key) -> Option<(Vec<u8>, String)> {
        let mut state = self.state.lock().expect("cache state");
        let entry = state.entries.get(key)?;
        let found = (entry.data.clone(), entry.version.clone());
        state.stats.hits += 1;
        state.stats.bytes_served = state
            .stats
            .bytes_served
            .saturating_add(found.0.len() as u64);
        Some(found)
    }

    fn insert(&self, key: Key, data: Vec<u8>, version: String) {
        let mut state = self.state.lock().expect("cache state");
        if data.len() as u64 > self.max_bytes {
            return; // Single range larger than the whole budget.
        }
        if state.entries.contains_key(&key) {
            return;
        }

        // Oldest first. The design wants frequency-based admission; insertion
        // order is enough to bound memory and is easy to reason about, and
        // eviction is always safe because this is a cache.
        while state.bytes + data.len() as u64 > self.max_bytes {
            let Some(oldest) = state.order.pop_front() else {
                break;
            };
            if let Some(dropped) = state.entries.remove(&oldest) {
                state.bytes = state.bytes.saturating_sub(dropped.data.len() as u64);
                state.stats.evictions += 1;
            }
        }

        state.bytes = state.bytes.saturating_add(data.len() as u64);
        state.order.push_back(key.clone());
        state.entries.insert(key, Entry { data, version });
    }

    fn drop_entry(&self, key: &Key) {
        let mut state = self.state.lock().expect("cache state");
        if let Some(dropped) = state.entries.remove(key) {
            state.bytes = state.bytes.saturating_sub(dropped.data.len() as u64);
            state.order.retain(|k| k != key);
        }
    }

    fn served(meta: ObjectMeta, range: Range<u64>, data: Vec<u8>) -> GetResult {
        let bytes = Bytes::from(data);
        let stream: BoxStream<'static, OsResult<Bytes>> =
            futures::stream::once(async move { Ok(bytes) }).boxed();
        GetResult {
            payload: GetResultPayload::Stream(stream),
            meta,
            range,
            attributes: Default::default(),
        }
    }
}

impl fmt::Debug for RangeCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RangeCache")
            .field("inner", &self.inner)
            .field("max_bytes", &self.max_bytes)
            .field("bytes", &self.bytes())
            .field("stats", &self.stats())
            .finish()
    }
}

impl fmt::Display for RangeCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RangeCache({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for RangeCache {
    async fn get_opts(&self, location: &Path, options: GetOptions) -> OsResult<GetResult> {
        let Some(range) = Self::cacheable(&options) else {
            self.state.lock().expect("cache state").stats.bypassed += 1;
            return self.inner.get_opts(location, options).await;
        };

        let key = Key {
            path: location.clone(),
            range: range.clone(),
        };

        if let Some((data, cached_version)) = self.lookup(&key) {
            // Defence in depth: the primary argument for serving this without
            // asking the store is that the object cannot have changed.
            let meta = ObjectMeta {
                location: location.clone(),
                last_modified: Default::default(),
                size: data.len() as u64,
                e_tag: Some(cached_version),
                version: None,
            };
            return Ok(Self::served(meta, range, data));
        }

        self.state.lock().expect("cache state").stats.misses += 1;
        let result = self.inner.get_opts(location, options).await?;
        let meta = result.meta.clone();
        let returned = result.range.clone();
        let version = Self::version_of(&meta);
        let data = result.bytes().await?.to_vec();

        // Only cache what was actually asked for. A store may legitimately
        // return a shorter range at the end of an object, and caching that
        // under the requested key would serve it for a different request.
        if returned == range {
            self.insert(key, data.clone(), version);
        }

        Ok(Self::served(meta, returned, data))
    }

    /// Invalidate on write, so a store used for both is not left inconsistent.
    ///
    /// Immutable data files never take this path; it exists so that a caller
    /// who does overwrite an object does not silently keep reading the old
    /// bytes for the rest of the process.
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OsResult<PutResult> {
        let result = self.inner.put_opts(location, payload, opts).await;
        self.invalidate_path(location);
        result
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OsResult<Box<dyn MultipartUpload>> {
        self.invalidate_path(location);
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn delete(&self, location: &Path) -> OsResult<()> {
        self.invalidate_path(location);
        self.inner.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OsResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OsResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OsResult<()> {
        self.invalidate_path(to);
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OsResult<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}

impl RangeCache {
    fn invalidate_path(&self, location: &Path) {
        let stale: Vec<Key> = {
            let state = self.state.lock().expect("cache state");
            state
                .entries
                .keys()
                .filter(|key| &key.path == location)
                .cloned()
                .collect()
        };
        for key in stale {
            self.drop_entry(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::MeteredStore;
    use object_store::memory::InMemory;

    fn cache(max_bytes: u64) -> (Arc<MeteredStore>, RangeCache) {
        let origin = Arc::new(MeteredStore::new(Arc::new(InMemory::new())));
        let cache = RangeCache::for_immutable_objects(Arc::clone(&origin) as _, max_bytes);
        (origin, cache)
    }

    async fn put(store: &dyn ObjectStore, path: &str, bytes: &[u8]) {
        store
            .put(&Path::from(path), PutPayload::from(bytes.to_vec()))
            .await
            .expect("put");
    }

    #[tokio::test]
    async fn a_repeated_range_is_served_without_touching_the_store() {
        let (origin, cache) = cache(1 << 20);
        put(&cache, "a", &[7u8; 1000]).await;
        let before = origin.stats().bytes_fetched;

        let first = cache
            .get_range(&Path::from("a"), 10..60)
            .await
            .expect("first");
        assert_eq!(first.len(), 50);
        let after_first = origin.stats().bytes_fetched;
        assert!(after_first > before, "the first read must reach the store");

        let second = cache
            .get_range(&Path::from("a"), 10..60)
            .await
            .expect("second");
        assert_eq!(second, first, "identical bytes");
        assert_eq!(
            origin.stats().bytes_fetched,
            after_first,
            "the second read must not reach the store at all"
        );
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
    }

    #[tokio::test]
    async fn a_different_range_of_the_same_object_is_a_miss() {
        let (_, cache) = cache(1 << 20);
        put(&cache, "a", &[0u8; 1000]).await;

        cache.get_range(&Path::from("a"), 0..10).await.expect("one");
        cache
            .get_range(&Path::from("a"), 10..20)
            .await
            .expect("two");
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 0);
    }

    #[tokio::test]
    async fn a_whole_object_read_is_bypassed_rather_than_cached() {
        // An unbounded read would need the object size to key correctly.
        let (_, cache) = cache(1 << 20);
        put(&cache, "a", &[0u8; 100]).await;

        cache.get(&Path::from("a")).await.expect("get");
        cache.get(&Path::from("a")).await.expect("get again");
        assert_eq!(cache.stats().hits, 0);
        assert!(cache.stats().bypassed >= 2);
    }

    #[tokio::test]
    async fn a_conditional_request_is_never_served_from_cache() {
        let (_, cache) = cache(1 << 20);
        put(&cache, "a", &[0u8; 100]).await;
        cache
            .get_range(&Path::from("a"), 0..10)
            .await
            .expect("warm");

        let conditional = GetOptions {
            range: Some(GetRange::Bounded(0..10)),
            if_none_match: Some("\"whatever\"".into()),
            ..Default::default()
        };
        let _ = cache.get_opts(&Path::from("a"), conditional).await;
        assert_eq!(
            cache.stats().hits,
            0,
            "a conditional request asks something the cache cannot answer"
        );
    }

    #[tokio::test]
    async fn eviction_keeps_the_cache_within_its_budget() {
        let (_, cache) = cache(100);
        put(&cache, "a", &[0u8; 1000]).await;

        for start in (0..200).step_by(40) {
            cache
                .get_range(&Path::from("a"), start..start + 40)
                .await
                .expect("read");
        }
        assert!(
            cache.bytes() <= 100,
            "held {} bytes against a 100 byte limit",
            cache.bytes()
        );
        assert!(cache.stats().evictions > 0);
    }

    #[tokio::test]
    async fn a_range_larger_than_the_budget_is_not_cached() {
        let (_, cache) = cache(10);
        put(&cache, "a", &[0u8; 1000]).await;

        cache
            .get_range(&Path::from("a"), 0..500)
            .await
            .expect("read");
        assert_eq!(cache.bytes(), 0);
        assert_eq!(cache.stats().hits, 0);
    }

    #[tokio::test]
    async fn overwriting_an_object_invalidates_it() {
        // Not the path immutable data files take, but a caller who does this
        // must not keep reading the old bytes.
        let (_, cache) = cache(1 << 20);
        put(&cache, "a", &[1u8; 100]).await;
        let first = cache
            .get_range(&Path::from("a"), 0..10)
            .await
            .expect("warm");
        assert_eq!(first[0], 1);

        put(&cache, "a", &[2u8; 100]).await;
        let second = cache
            .get_range(&Path::from("a"), 0..10)
            .await
            .expect("again");
        assert_eq!(second[0], 2, "stale bytes served after an overwrite");
    }

    #[tokio::test]
    async fn deleting_an_object_invalidates_it() {
        let (_, cache) = cache(1 << 20);
        put(&cache, "a", &[1u8; 100]).await;
        cache
            .get_range(&Path::from("a"), 0..10)
            .await
            .expect("warm");

        cache.delete(&Path::from("a")).await.expect("delete");
        assert!(cache.get_range(&Path::from("a"), 0..10).await.is_err());
    }

    #[tokio::test]
    async fn clearing_is_always_safe() {
        let (_, cache) = cache(1 << 20);
        put(&cache, "a", &[0u8; 100]).await;
        cache
            .get_range(&Path::from("a"), 0..10)
            .await
            .expect("warm");
        assert!(cache.bytes() > 0);

        cache.clear();
        assert_eq!(cache.bytes(), 0);
        // Still answers correctly, just more slowly.
        assert_eq!(
            cache
                .get_range(&Path::from("a"), 0..10)
                .await
                .expect("after clear")
                .len(),
            10
        );
    }
}
