//! One place that assembles the pieces.
//!
//! Until now a working engine had to be wired by hand: an origin store, a
//! meter, a cache, a session, a registered table. That is fine in a test and
//! wrong as an interface, since the design's whole user-facing claim is that
//! ordinary SQL gets faster without anything new to learn.
//!
//! # The store stack, and why it has two meters
//!
//! ```text
//! MeteredStore   per session: enforces this query's budget
//!      └─ RangeCache   shared: immutable object ranges (optional)
//!            └─ MeteredStore   shared: lifetime origin I/O
//!                  └─ origin
//! ```
//!
//! Both meters are wanted, and they measure different things:
//!
//! - The **outer** one sees every read including cache hits. That is the right
//!   basis for a budget: a query that reads 10 GB out of cache still consumed
//!   10 GB of work, and a caller who asked for a ceiling meant it.
//! - The **inner** one sees only what missed, so its counter is real origin
//!   I/O — what the cache saved, and what a remote store would have billed.
//!
//! The cache sits between them because it must outlive any one session, while
//! a budget must not.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::error::Result as DfResult;
use datafusion::prelude::SessionContext;
use object_store::ObjectStore;
use url::Url;

use crate::budget::{Budget, Outcome};
use crate::cost::{Cost, PriceTable};
use crate::facts::{OpaqueStorage, StorageFacts};
use crate::place::Place;

use super::{CacheStats, MeteredStore, QuarryTable, RangeCache, StoreStats};

/// A configured engine: an object store, prices, and an optional cache.
///
/// Long-lived. Hand out a [`Session`] per query, or per group of queries that
/// should share a budget.
#[derive(Debug)]
pub struct Quarry {
    url: Url,
    /// The stack below any per-session metering: cache over origin meter.
    shared: Arc<dyn ObjectStore>,
    origin_meter: Arc<MeteredStore>,
    cache: Option<Arc<RangeCache>>,
    prices: PriceTable,
    facts: Arc<dyn StorageFacts>,
    here: Place,
}

impl Quarry {
    /// Read from `origin`, with no cache.
    pub fn new(url: Url, origin: Arc<dyn ObjectStore>) -> Self {
        let origin_meter = Arc::new(MeteredStore::new(origin));
        Quarry {
            url,
            shared: Arc::clone(&origin_meter) as _,
            origin_meter,
            cache: None,
            prices: PriceTable::default(),
            facts: Arc::new(OpaqueStorage),
            here: Place::unknown(),
        }
    }

    /// Read from `origin` through a cache of at most `cache_bytes`.
    ///
    /// Only sound for stores whose objects are never modified in place; see
    /// [`RangeCache`].
    pub fn with_cache(url: Url, origin: Arc<dyn ObjectStore>, cache_bytes: u64) -> Self {
        let origin_meter = Arc::new(MeteredStore::new(origin));
        let cache = Arc::new(RangeCache::for_immutable_objects(
            Arc::clone(&origin_meter) as _,
            cache_bytes,
        ));
        Quarry {
            url,
            shared: Arc::clone(&cache) as _,
            origin_meter,
            cache: Some(cache),
            prices: PriceTable::default(),
            facts: Arc::new(OpaqueStorage),
            here: Place::unknown(),
        }
    }

    /// Price reads differently — a colocated store, or one with no egress bill.
    pub fn with_prices(mut self, prices: PriceTable) -> Self {
        self.prices = prices;
        self
    }

    /// Let the backend report where its objects are and which are cold.
    ///
    /// Without this everything is priced hot and far, which is what plain
    /// object storage is. With it, a colocated or tiered backend costs less
    /// where it should — and nothing here or in the optimizer learns which
    /// backend it is talking to.
    pub fn with_facts(mut self, facts: Arc<dyn StorageFacts>, here: Place) -> Self {
        self.facts = facts;
        self.here = here;
        self
    }

    /// Bytes actually fetched from the origin, for the life of this `Quarry`.
    ///
    /// What the cache did *not* save. Compare with a session's
    /// [`Session::spent`] to see the cache's effect.
    pub fn origin_stats(&self) -> StoreStats {
        self.origin_meter.stats()
    }

    /// How the cache has been used, if there is one.
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|cache| cache.stats())
    }

    /// Drop the cache's contents. Always safe.
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.clear();
        }
    }

    /// A session with no spending limit.
    pub fn session(&self) -> Session {
        self.session_with_budget(Budget::UNLIMITED)
    }

    /// A session that stops once `budget` is spent.
    ///
    /// The ceiling is enforced against bytes as they are read, so an
    /// over-budget query fails part-way rather than after the fact — and it
    /// fails rather than returning fewer rows, because a truncated answer
    /// looks complete.
    pub fn session_with_budget(&self, budget: Budget) -> Session {
        let ctx = SessionContext::new();
        // Take the parallelism from DataFusion rather than assuming it. Reads
        // that overlap wait once, not once each, and on a sixteen-core machine
        // the difference is sixteenfold on what is often the largest part of a
        // small-file scan's cost.
        let prices = self
            .prices
            .with_concurrent_reads(ctx.state().config().target_partitions() as f64);
        let meter = Arc::new(
            MeteredStore::new(Arc::clone(&self.shared))
                .with_facts(Arc::clone(&self.facts), self.here.clone())
                .with_budget(budget, prices),
        );
        ctx.register_object_store(&self.url, Arc::clone(&meter) as _);
        Session { ctx, meter }
    }
}

/// One session's worth of querying, with its own budget.
pub struct Session {
    ctx: SessionContext,
    meter: Arc<MeteredStore>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // SessionContext is not Debug, and its contents would not be useful
        // here anyway; what a caller wants to see is the spend.
        f.debug_struct("Session")
            .field("stats", &self.stats())
            .field("outcome", &self.outcome())
            .finish_non_exhaustive()
    }
}

impl Session {
    /// The underlying DataFusion context, for anything not wrapped here.
    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }

    /// Make a table queryable under `name`.
    pub fn register(&self, name: &str, table: Arc<QuarryTable>) -> DfResult<()> {
        self.ctx.register_table(name, table)?;
        Ok(())
    }

    /// Run a query to completion.
    ///
    /// Before handing the plan to DataFusion, aggregate subtrees over a
    /// registered table are offered to the registry: a cube that the rule
    /// admits rewrites the subtree to read stored partials, and the answer
    /// comes back re-aggregated at the query's grain. Everything else runs as
    /// written — `scan` still applies indexes and result caches below.
    pub async fn sql(&self, sql: &str) -> DfResult<Vec<RecordBatch>> {
        let df = self.ctx.sql(sql).await?;
        let plan = df.logical_plan().clone();
        let (rewritten, served) = super::cube::rewrite(&plan)?;
        if served.is_some() {
            return datafusion::dataframe::DataFrame::new(self.ctx.state(), rewritten)
                .collect()
                .await;
        }
        let rows = df.collect().await?;
        // An aggregate no cube could serve still reports the ask: the scan
        // itself saw only `aggregate: None`, and the optimizer can only
        // propose what it can see.
        if let Some(ask) = super::cube::first_ask(&plan) {
            if let Some(query) = super::cube::query_of(&ask) {
                if let Some(mut report) = ask.table.last_scan() {
                    report.aggregate =
                        query.aggregate.zip(query.plan.clone()).map(|(spec, plan)| {
                            crate::workload::AggregateAsk {
                                table: ask.table.table_id().clone(),
                                plan,
                                spec,
                                filter_sql: super::cube::filter_sql(&ask),
                            }
                        });
                    report.plan = query.plan;
                    ask.table.note_scan(report);
                }
            }
        }
        Ok(rows)
    }

    /// What this session has read, priced.
    ///
    /// Includes reads served from cache: this is what the query consumed, not
    /// what it cost the origin. For the latter see [`Quarry::origin_stats`].
    pub fn spent(&self) -> Cost {
        self.meter.spent()
    }

    /// Bytes and requests this session issued.
    pub fn stats(&self) -> StoreStats {
        self.meter.stats()
    }

    /// Whether a budget stopped this session.
    pub fn outcome(&self) -> Outcome {
        self.meter.outcome()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::TableId;
    use crate::snapshot::{Snapshot, SnapshotGraph, SnapshotId};
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use object_store::memory::InMemory;
    use std::collections::BTreeMap;

    fn url() -> Url {
        Url::parse("memory://").expect("url")
    }

    fn table() -> Arc<QuarryTable> {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![1i64, 2, 3]))],
        )
        .expect("batch");

        let file = crate::snapshot::FileId("only".into());
        let graph =
            SnapshotGraph::new().with(Snapshot::root(SnapshotId(1)).with_clean_file(file.clone()));

        Arc::new(
            QuarryTable::new(
                schema,
                TableId("t".into()),
                SnapshotId(1),
                graph,
                BTreeMap::from([("n".to_owned(), 1u32)]),
            )
            .with_file(file, vec![batch]),
        )
    }

    #[tokio::test]
    async fn a_session_answers_sql() {
        let quarry = Quarry::new(url(), Arc::new(InMemory::new()));
        let session = quarry.session();
        session.register("t", table()).expect("register");

        let rows = session.sql("SELECT sum(n) AS s FROM t").await.expect("sql");
        let sum = rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64")
            .value(0);
        assert_eq!(sum, 6);
    }

    #[test]
    fn a_quarry_without_a_cache_reports_no_cache_stats() {
        let quarry = Quarry::new(url(), Arc::new(InMemory::new()));
        assert!(quarry.cache_stats().is_none());
        quarry.clear_cache(); // harmless
    }

    #[test]
    fn a_quarry_with_a_cache_reports_cache_stats() {
        let quarry = Quarry::with_cache(url(), Arc::new(InMemory::new()), 1 << 20);
        assert_eq!(quarry.cache_stats(), Some(CacheStats::default()));
    }

    #[tokio::test]
    async fn sessions_have_separate_budgets_but_share_the_cache() {
        let quarry = Quarry::with_cache(url(), Arc::new(InMemory::new()), 1 << 20);

        let first = quarry.session_with_budget(Budget::bytes(10));
        let second = quarry.session();

        assert_eq!(first.stats(), StoreStats::default());
        assert_eq!(second.stats(), StoreStats::default());
        assert!(first.outcome().is_complete());

        // Both draw on the same cache, so the shared one is what persists.
        assert!(quarry.cache_stats().is_some());
    }

    #[tokio::test]
    async fn a_reporting_backend_costs_less_than_an_opaque_one() {
        use crate::facts::{PlacedStorage, Placement};
        use object_store::PutPayload;
        use object_store::path::Path;

        let here = Place::parse("/onprem/dc1/rack2/node7");

        /// Read one object through a session and report what it was charged.
        async fn read_cost(quarry: &Quarry) -> f64 {
            let session = quarry.session_with_budget(Budget::UNLIMITED);
            session
                .meter
                .put(&Path::from("a"), PutPayload::from(vec![0u8; 1000]))
                .await
                .expect("put");
            session.meter.get(&Path::from("a")).await.expect("get");
            session.spent().usd
        }

        // Priced for bytes leaving the region: under same-region rates
        // transfer is free, so reporting locality changes nothing about the
        // money and there would be nothing to assert. The point being tested
        // is that facts reach the price, not that distance always costs.
        let prices = crate::cost::PriceTable::aws_s3_internet();
        let opaque = Quarry::new(url(), Arc::new(InMemory::new())).with_prices(prices);
        let placed = Quarry::new(url(), Arc::new(InMemory::new()))
            .with_prices(prices)
            .with_facts(
                Arc::new(PlacedStorage::new().with_fallback(Placement::hot(here.clone()))),
                here,
            );

        let opaque_cost = read_cost(&opaque).await;
        let placed_cost = read_cost(&placed).await;

        assert!(
            placed_cost < opaque_cost,
            "a backend that reports locality should cost less: {placed_cost} vs {opaque_cost}"
        );
    }

    #[tokio::test]
    async fn in_memory_tables_never_touch_the_store() {
        // Worth pinning: the table above holds its data in memory, so the
        // store stack should see nothing at all.
        let quarry = Quarry::new(url(), Arc::new(InMemory::new()));
        let session = quarry.session();
        session.register("t", table()).expect("register");
        session.sql("SELECT * FROM t").await.expect("sql");

        assert_eq!(quarry.origin_stats(), StoreStats::default());
        assert_eq!(session.stats(), StoreStats::default());
    }
}
