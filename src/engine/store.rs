//! An object store that counts what it fetches, and stops when a budget says
//! so.
//!
//! Two things become real here that were previously only modelled:
//!
//! - **Pruning saves measurable I/O.** The rule decides which files to read;
//!   this counts the bytes that actually left the store, so the saving is a
//!   number rather than a claim.
//! - **Budgets are enforced, not estimated.** A [`Budget`] wired in here
//!   refuses reads once the ceiling is crossed, which aborts the query
//!   mid-scan. A ceiling checked only during planning would be decoration,
//!   because cardinality estimates are routinely wrong by orders of magnitude.

use std::fmt;
use std::sync::{Arc, Mutex};

use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OsResult,
};

use crate::budget::{Budget, Exceeded, Meter, Outcome, Permit};
use crate::cost::PriceTable;
use crate::facts::{OpaqueStorage, StorageFacts, resolve};
use crate::place::Place;

/// What a [`MeteredStore`] has fetched.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StoreStats {
    /// Bytes returned by the underlying store.
    pub bytes_fetched: u64,
    /// Requests made to the underlying store.
    pub requests: u64,
    /// Seconds the requests took, summed per request.
    ///
    /// A sum, not wall time: parallel reads each contribute their own
    /// duration, so this is directly comparable to what the meter charges
    /// *before* the overlap discount. Whether the [`Link`](crate::cost::Link)
    /// figures describe reality is checkable as `observed / requests` against
    /// `first_byte_seconds`.
    pub observed_seconds: f64,
}

/// Wraps an object store to count reads and enforce a budget.
///
/// What a read *costs* comes from asking the backend, through
/// [`StorageFacts`]. A backend that cannot answer resolves to hot and
/// [`Far`](crate::place::Distance::Far), which is what plain object storage is;
/// one that reports placement makes the same budget buy far more bytes. Either
/// way this type consults [`resolve`] and never asks which backend it has.
pub struct MeteredStore {
    inner: Arc<dyn ObjectStore>,
    facts: Arc<dyn StorageFacts>,
    reader: Place,
    state: Mutex<State>,
}

struct State {
    stats: StoreStats,
    meter: Option<Meter>,
}

impl MeteredStore {
    /// Count reads against `inner`, without any ceiling.
    ///
    /// The backend is assumed to know nothing about placement; see
    /// [`MeteredStore::with_facts`].
    pub fn new(inner: Arc<dyn ObjectStore>) -> Self {
        MeteredStore {
            inner,
            facts: Arc::new(OpaqueStorage),
            reader: Place::unknown(),
            state: Mutex::new(State {
                stats: StoreStats::default(),
                meter: None,
            }),
        }
    }

    /// Refuse reads once `budget` is exceeded.
    pub fn with_budget(mut self, budget: Budget, prices: PriceTable) -> Self {
        self.state = Mutex::new(State {
            stats: StoreStats::default(),
            meter: Some(Meter::new(budget, prices)),
        });
        self
    }

    /// Ask `facts` what each object costs, from a reader at `reader`.
    pub fn with_facts(mut self, facts: Arc<dyn StorageFacts>, reader: Place) -> Self {
        self.facts = facts;
        self.reader = reader;
        self
    }

    /// The commit log the backend pushes into, if it offers one.
    ///
    /// Hand this to [`Optimizer::with_commits`](super::Optimizer::with_commits):
    /// a backend that can hear commits is how the loop stops polling.
    pub fn commits(&self) -> Option<crate::snapshot::Commits> {
        self.facts.commits()
    }

    /// What has been fetched so far.
    pub fn stats(&self) -> StoreStats {
        self.state.lock().expect("store state").stats
    }

    /// What has been read, priced.
    ///
    /// Zero unless a budget is set, since pricing is the meter's job and a
    /// store without one only counts.
    pub fn spent(&self) -> crate::cost::Cost {
        self.state
            .lock()
            .expect("store state")
            .meter
            .as_ref()
            .map(Meter::spent)
            .unwrap_or(crate::cost::Cost::ZERO)
    }

    /// Whether a budget stopped the reads.
    pub fn outcome(&self) -> Outcome {
        self.state
            .lock()
            .expect("store state")
            .meter
            .as_ref()
            .map(Meter::outcome)
            .unwrap_or(Outcome::Complete)
    }

    /// Account for a read of `object`, and refuse it if the budget is spent.
    ///
    /// Charged *before* the ceiling is tested, so the reported spend reflects
    /// what was consumed rather than the last amount that happened to fit.
    ///
    /// The price comes from the backend rather than from a field here, so a
    /// colocated hot object and a cold remote one are charged differently
    /// without this code knowing which is which.
    /// `elapsed` is what the request actually took, recorded so the charged
    /// wait can be checked against reality rather than trusted.
    fn charge(&self, object: &Path, bytes: u64, elapsed: f64) -> OsResult<()> {
        let resolved = resolve(self.facts.as_ref(), object.as_ref(), &self.reader);

        let mut state = self.state.lock().expect("store state");
        state.stats.bytes_fetched = state.stats.bytes_fetched.saturating_add(bytes);
        state.stats.requests += 1;
        state.stats.observed_seconds += elapsed;

        let Some(meter) = state.meter.as_mut() else {
            return Ok(());
        };
        match meter.charge(bytes, resolved.tier, resolved.distance) {
            Permit::Continue => Ok(()),
            Permit::Stop(exceeded) => Err(object_store::Error::Generic {
                store: "quarry",
                source: Box::new(BudgetExceeded(exceeded)),
            }),
        }
    }

    /// Charge a request that returns no payload: a HEAD, or a page of
    /// listing. Counted like a read and billed like one too — AWS does not
    /// distinguish "how much did you fetch" from "how often did you ask".
    ///
    /// `object` is what the request was about, so a HEAD of a cold or far
    /// object is charged like one. A listing has no single object; "" is
    /// whatever the backend reports for a path it does not know.
    fn charge_meta(&self, object: &str, listed: bool, elapsed: f64) -> OsResult<()> {
        let resolved = resolve(self.facts.as_ref(), object, &self.reader);

        let mut state = self.state.lock().expect("store state");
        state.stats.requests += 1;
        state.stats.observed_seconds += elapsed;

        let Some(meter) = state.meter.as_mut() else {
            return Ok(());
        };
        let permit = if listed {
            meter.charge_list(resolved.tier, resolved.distance)
        } else {
            meter.charge_meta(resolved.tier, resolved.distance)
        };
        match permit {
            Permit::Continue => Ok(()),
            Permit::Stop(exceeded) => Err(object_store::Error::Generic {
                store: "quarry",
                source: Box::new(BudgetExceeded(exceeded)),
            }),
        }
    }
}

/// The error a [`MeteredStore`] returns when a budget stops a read.
///
/// Surfaces through DataFusion as an execution error, so an over-budget query
/// fails loudly rather than quietly returning fewer rows. Silently truncating
/// would be far worse: the answer would look complete.
#[derive(Debug)]
pub struct BudgetExceeded(pub Exceeded);

impl fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Exceeded::Bytes { limit, spent } => write!(
                f,
                "query budget exceeded: read {spent} bytes, limit {limit}"
            ),
            Exceeded::Usd { limit, spent } => write!(
                f,
                "query budget exceeded: spent ${spent:.8}, limit ${limit:.8}"
            ),
        }
    }
}

impl std::error::Error for BudgetExceeded {}

impl fmt::Debug for MeteredStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MeteredStore")
            .field("inner", &self.inner)
            .field("facts", &self.facts)
            .field("reader", &self.reader)
            .field("stats", &self.stats())
            .finish()
    }
}

impl fmt::Display for MeteredStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeteredStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for MeteredStore {
    /// The counted read.
    ///
    /// `get`, `get_range`, `get_ranges` and `head` all reach the store through
    /// here by default — `head` arrives as `options.head` and is charged as a
    /// request rather than a read, since it returns no payload for the byte
    /// price to attach to.
    async fn get_opts(&self, location: &Path, options: GetOptions) -> OsResult<GetResult> {
        let started = std::time::Instant::now();
        let head = options.head;
        let result = self.inner.get_opts(location, options).await?;
        let elapsed = started.elapsed().as_secs_f64();
        if head {
            self.charge_meta(location.as_ref(), false, elapsed)?;
        } else {
            let fetched = result.range.end.saturating_sub(result.range.start);
            self.charge(location, fetched, elapsed)?;
        }
        Ok(result)
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OsResult<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OsResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn delete(&self, location: &Path) -> OsResult<()> {
        self.inner.delete(location).await
    }

    /// Charged once per call.
    ///
    /// The returned stream paginates *inside* the inner store, so the number
    /// of LIST requests is invisible here — a long listing is undercharged
    /// rather than uncharged. `list_with_delimiter` is exact.
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OsResult<ObjectMeta>> {
        // A listing that is refused still has to return a stream, so the
        // refusal becomes its one item.
        match self.charge_meta(prefix.map_or("", |p| p.as_ref()), true, 0.0) {
            Ok(()) => self.inner.list(prefix),
            Err(err) => Box::pin(futures::stream::once(async move { Err(err) })),
        }
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OsResult<ListResult> {
        let started = std::time::Instant::now();
        let result = self.inner.list_with_delimiter(prefix).await?;
        self.charge_meta(
            prefix.map_or("", |p| p.as_ref()),
            true,
            started.elapsed().as_secs_f64(),
        )?;
        Ok(result)
    }

    async fn copy(&self, from: &Path, to: &Path) -> OsResult<()> {
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OsResult<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::Tier;
    use crate::facts::{PlacedStorage, Placement};
    use crate::place::Distance;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    async fn put(store: &dyn ObjectStore, path: &str, bytes: &[u8]) {
        store
            .put(&Path::from(path), PutPayload::from(bytes.to_vec()))
            .await
            .expect("put");
    }

    #[tokio::test]
    async fn reads_are_counted() {
        let metered = MeteredStore::new(store());
        put(&metered, "a", &[0u8; 100]).await;

        assert_eq!(
            metered.stats(),
            StoreStats::default(),
            "writes are not reads"
        );

        metered.get(&Path::from("a")).await.expect("get");
        assert_eq!(metered.stats().bytes_fetched, 100);
        assert_eq!(metered.stats().requests, 1);
    }

    #[tokio::test]
    async fn ranged_reads_count_only_the_range() {
        let metered = MeteredStore::new(store());
        put(&metered, "a", &[0u8; 1000]).await;

        metered
            .get_range(&Path::from("a"), 10..50)
            .await
            .expect("get_range");
        assert_eq!(metered.stats().bytes_fetched, 40);
    }

    #[tokio::test]
    async fn a_budget_refuses_reads_once_spent() {
        let metered =
            MeteredStore::new(store()).with_budget(Budget::bytes(150), PriceTable::default());
        put(&metered, "a", &[0u8; 100]).await;

        assert!(metered.get(&Path::from("a")).await.is_ok());
        let second = metered.get(&Path::from("a")).await;
        assert!(second.is_err(), "200 bytes exceeds a 150 byte ceiling");

        let message = second.unwrap_err().to_string();
        assert!(
            message.contains("budget exceeded"),
            "unhelpful error: {message}"
        );
        assert!(matches!(
            metered.outcome(),
            Outcome::Aborted(Exceeded::Bytes { .. })
        ));
    }

    #[tokio::test]
    async fn without_a_budget_nothing_is_refused() {
        let metered = MeteredStore::new(store());
        put(&metered, "a", &[0u8; 10_000]).await;
        for _ in 0..10 {
            metered.get(&Path::from("a")).await.expect("get");
        }
        assert_eq!(metered.stats().bytes_fetched, 100_000);
        assert!(metered.outcome().is_complete());
    }

    #[tokio::test]
    async fn what_the_backend_reports_decides_how_much_a_budget_buys() {
        use crate::facts::{PlacedStorage, Placement};

        // Priced for bytes leaving the region, because that is the deployment
        // where distance costs money. Under same-region rates — the default —
        // transfer is not billed at all, so a colocated byte and a remote one
        // cost exactly the same and this would have nothing to measure. See
        // `PriceTable::aws_s3_same_region`.
        let prices = PriceTable::aws_s3_internet();
        // A ceiling that affords one far read and not two. Priced with
        // `price` rather than `byte_usd * bytes`, because most of what a small
        // remote read costs is the round trip waited for, not the bytes moved.
        let budget = Budget::usd(prices.price(150, Tier::Hot, Distance::Far, 0.0).usd);
        let here = Place::parse("/onprem/dc1/rack2/node7");

        // A backend that says nothing: everything is far.
        let opaque = MeteredStore::new(store()).with_budget(budget, prices);

        // A backend that reports the object as sitting on this very node.
        let colocated = MeteredStore::new(store())
            .with_facts(
                Arc::new(PlacedStorage::new().with_fallback(Placement::hot(here.clone()))),
                here,
            )
            .with_budget(budget, prices);

        put(&opaque, "a", &[0u8; 100]).await;
        put(&colocated, "a", &[0u8; 100]).await;

        assert!(opaque.get(&Path::from("a")).await.is_ok());
        assert!(
            opaque.get(&Path::from("a")).await.is_err(),
            "200 unplaced bytes are priced as far and exceed the ceiling"
        );

        for _ in 0..10 {
            assert!(
                colocated.get(&Path::from("a")).await.is_ok(),
                "the same money buys far more bytes the backend says are local"
            );
        }
    }

    #[tokio::test]
    async fn a_cold_object_is_charged_more_than_a_hot_one() {
        use crate::facts::{PlacedStorage, Placement};

        let here = Place::parse("/onprem/dc1/rack2/node7");
        let facts = Arc::new(
            PlacedStorage::new()
                .with_prefix("hot/", Placement::hot(here.clone()))
                .with_prefix("cold/", Placement::cold(here.clone())),
        );

        let metered = MeteredStore::new(store())
            .with_facts(facts, here)
            .with_budget(Budget::UNLIMITED, PriceTable::default());

        put(&metered, "hot/a", &[0u8; 100]).await;
        metered.get(&Path::from("hot/a")).await.expect("hot read");
        let after_hot = metered.spent().usd;

        put(&metered, "cold/a", &[0u8; 100]).await;
        metered.get(&Path::from("cold/a")).await.expect("cold read");
        let cold_cost = metered.spent().usd - after_hot;

        assert!(
            cold_cost > after_hot,
            "the same 100 bytes cost {cold_cost} cold against {after_hot} hot"
        );
    }

    #[tokio::test]
    async fn writes_and_listing_pass_through() {
        let metered = MeteredStore::new(store());
        put(&metered, "dir/a", &[1u8; 10]).await;
        put(&metered, "dir/b", &[2u8; 10]).await;

        let listed = metered.list_with_delimiter(Some(&Path::from("dir"))).await;
        assert_eq!(listed.expect("list").objects.len(), 2);

        metered.delete(&Path::from("dir/a")).await.expect("delete");
        assert!(metered.get(&Path::from("dir/a")).await.is_err());
    }

    /// Charged waiting is an estimate; `observed_seconds` is what actually
    /// happened. This test exists so the two cannot silently diverge.
    ///
    /// The `Link` figures come from benchmarks of real storage. A local
    /// filesystem read measured ~45us against a charged 100us — within 2.2x.
    /// If this starts failing, the right response is to update the table or
    /// the store, not to widen the tolerance: the number exists to be checked.
    #[tokio::test]
    async fn charged_waiting_stays_within_sight_of_observed() {
        let dir = std::env::temp_dir().join(format!("quarry_obs_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let local = LocalFileSystem::new_with_prefix(&dir).expect("local store");
        local
            .put(&Path::from("x"), PutPayload::from(vec![0u8; 4096]))
            .await
            .expect("put");

        let store = MeteredStore::new(Arc::new(local)).with_facts(
            Arc::new(PlacedStorage::new().with_fallback(Placement::hot(Place::parse("/a/b")))),
            Place::parse("/a/b"),
        );
        for _ in 0..20 {
            store.get(&Path::from("x")).await.expect("get");
        }

        let stats = store.stats();
        assert_eq!(stats.requests, 20);
        assert!(stats.observed_seconds > 0.0, "observed time was recorded");

        let observed_per_read = stats.observed_seconds / stats.requests as f64;
        let charged_per_read = PriceTable::default().local_link.first_byte_seconds;
        let ratio = charged_per_read / observed_per_read;
        assert!(
            (1.0 / 25.0..25.0).contains(&ratio),
            "charged {charged_per_read}s vs observed {observed_per_read}s per read ({ratio}x)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `head` reaches the meter: it routes through `get_opts` with an empty
    /// range, which used to be counted and charged nothing.
    #[tokio::test]
    async fn a_head_is_counted_and_charged() {
        let metered = MeteredStore::new(store());
        put(&metered, "a", b"data").await;

        metered.head(&Path::from("a")).await.expect("head");

        let stats = metered.stats();
        assert_eq!(stats.requests, 1);
        assert!(stats.observed_seconds > 0.0);
        assert_eq!(stats.bytes_fetched, 0, "a HEAD moves no payload");
    }

    #[tokio::test]
    async fn a_budget_stops_repeated_metadata_calls() {
        // A workload of many small metadata calls — exactly the pattern that
        // went unmetered — is now inside the budget like any other spend.
        let prices = PriceTable::default();
        let per_head = prices.request_usd
            + prices.wait_seconds(0, Tier::Hot, Distance::Far) * prices.cpu_second_usd;
        let metered = MeteredStore::new(store()).with_budget(Budget::usd(per_head * 1.5), prices);
        put(&metered, "a", b"data").await;

        assert!(metered.head(&Path::from("a")).await.is_ok(), "one fits");
        assert!(
            metered.head(&Path::from("a")).await.is_err(),
            "the second does not"
        );
    }

    #[tokio::test]
    async fn a_listing_is_charged() {
        use futures::TryStreamExt;
        let metered = MeteredStore::new(store());
        put(&metered, "dir/a", b"x").await;
        put(&metered, "dir/b", b"y").await;

        let found: Vec<_> = metered
            .list(Some(&Path::from("dir")))
            .try_collect()
            .await
            .expect("list");
        assert_eq!(found.len(), 2);
        assert_eq!(metered.stats().requests, 1, "one chargeable call");

        // And a budget can refuse it — the refusal arrives as the stream's
        // first item, since a refused listing still has to produce a stream.
        let priced = PriceTable::default();
        let refused = MeteredStore::new(store()).with_budget(Budget::usd(0.0), priced);
        let result = refused
            .list(Some(&Path::from("dir")))
            .try_collect::<Vec<_>>()
            .await;
        assert!(result.is_err(), "a zero budget refuses the listing");
    }
}
