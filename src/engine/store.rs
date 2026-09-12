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
use crate::cost::{PriceTable, Tier};
use crate::place::Distance;

/// What a [`MeteredStore`] has fetched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreStats {
    /// Bytes returned by the underlying store.
    pub bytes_fetched: u64,
    /// Requests made to the underlying store.
    pub requests: u64,
}

/// Wraps an object store to count reads and enforce a budget.
///
/// Reads are priced as hot and [`Distance::Far`] by default, which is what
/// remote object storage is. A colocated store should say otherwise via
/// [`MeteredStore::with_locality`], and then the same budget buys far more
/// bytes — which is the whole point of pricing distance.
pub struct MeteredStore {
    inner: Arc<dyn ObjectStore>,
    tier: Tier,
    distance: Distance,
    state: Mutex<State>,
}

struct State {
    stats: StoreStats,
    meter: Option<Meter>,
}

impl MeteredStore {
    /// Count reads against `inner`, without any ceiling.
    pub fn new(inner: Arc<dyn ObjectStore>) -> Self {
        MeteredStore {
            inner,
            tier: Tier::Hot,
            distance: Distance::Far,
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

    /// Declare how far away, and how cold, this store's bytes are.
    pub fn with_locality(mut self, tier: Tier, distance: Distance) -> Self {
        self.tier = tier;
        self.distance = distance;
        self
    }

    /// What has been fetched so far.
    pub fn stats(&self) -> StoreStats {
        self.state.lock().expect("store state").stats
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

    /// Account for a read, and refuse it if the budget is spent.
    ///
    /// Charged *before* the ceiling is tested, so the reported spend reflects
    /// what was consumed rather than the last amount that happened to fit.
    fn charge(&self, bytes: u64) -> OsResult<()> {
        let mut state = self.state.lock().expect("store state");
        state.stats.bytes_fetched = state.stats.bytes_fetched.saturating_add(bytes);
        state.stats.requests += 1;

        let Some(meter) = state.meter.as_mut() else {
            return Ok(());
        };
        match meter.charge(bytes, self.tier, self.distance) {
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
            .field("tier", &self.tier)
            .field("distance", &self.distance)
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
    /// The one counted method.
    ///
    /// Every other read path — `get`, `get_range`, `get_ranges`, `head` —
    /// reaches the store through here by default, so counting once covers all
    /// of them. `GetResult::range` gives the byte count without consuming the
    /// payload stream.
    async fn get_opts(&self, location: &Path, options: GetOptions) -> OsResult<GetResult> {
        let result = self.inner.get_opts(location, options).await?;
        let fetched = result.range.end.saturating_sub(result.range.start);
        self.charge(fetched)?;
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

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OsResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OsResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
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
    async fn distance_decides_how_many_bytes_a_money_budget_buys() {
        let prices = PriceTable::default();
        let budget = Budget::usd(prices.byte_usd(Tier::Hot, Distance::Far) * 150.0);

        let far = MeteredStore::new(store())
            .with_locality(Tier::Hot, Distance::Far)
            .with_budget(budget, prices);
        let local = MeteredStore::new(store())
            .with_locality(Tier::Hot, Distance::Local)
            .with_budget(budget, prices);

        put(&far, "a", &[0u8; 100]).await;
        put(&local, "a", &[0u8; 100]).await;

        assert!(far.get(&Path::from("a")).await.is_ok());
        assert!(
            far.get(&Path::from("a")).await.is_err(),
            "200 far bytes cost more than the ceiling"
        );

        for _ in 0..10 {
            assert!(
                local.get(&Path::from("a")).await.is_ok(),
                "the same money buys far more local bytes"
            );
        }
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
}
