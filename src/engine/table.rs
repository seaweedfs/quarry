//! A [`TableProvider`] whose scan is planned by the rule.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};

use async_trait::async_trait;
use datafusion::arrow::array::{RecordBatch, UInt64Array};
use datafusion::arrow::compute::take;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{DataFusionError, Result as DfResult};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
use datafusion::datasource::physical_plan::{FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown, TableType};
use datafusion::parquet::arrow::arrow_reader::{ArrowReaderMetadata, RowSelection};
use datafusion::parquet::arrow::async_reader::ParquetObjectReader;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::union::UnionExec;
use datafusion::scalar::ScalarValue;

use crate::cost::PriceTable;
use crate::derived::{
    AggFunc, Aggregate, Decision, DerivedId, FieldId, Filter, Measure, Plan, PolicyFingerprint,
    Predicate, Query, Scope,
};
use crate::registry::{Composed, Registry, compose};
use crate::snapshot::{FileId, SnapshotGraph, SnapshotId, TableId};
use crate::stable_hash::StableHasher;
use crate::workload::{Fingerprint, Observation};
use object_store::path::Path as ObjectPath;

/// A registry several things can hold at once.
///
/// A plain `RwLock`: planning takes a read guard and never awaits while
/// holding it, and the optimizer takes a write guard between rounds. An async
/// lock would buy nothing and make the planning path await.
pub type SharedRegistry = Arc<RwLock<Registry>>;

/// Wrap a registry so it can be shared.
pub fn shared(registry: Registry) -> SharedRegistry {
    Arc::new(RwLock::new(registry))
}

/// Hash a literal the way [`QuarryTable`] does when probing an index.
///
/// Anything building an index must agree with this, or a probe will miss.
/// Uses [`StableHasher`], so an index written to storage still matches after a
/// compiler upgrade — see [`HASH_VERSION`](crate::stable_hash::HASH_VERSION)
/// for what remains versioned, and why.
///
/// A collision here is safe: two values hashing alike means the index reports
/// files holding the other value too, so the scan over-selects and the
/// predicate is re-applied to rows anyway.
pub fn hash_scalar(value: &ScalarValue) -> u64 {
    StableHasher::of(value)
}

/// What the rule decided for the most recent scan.
///
/// Recorded so a caller can see the decision after the fact — the same
/// information `EXPLAIN` reports, kept here because a scan is where it is
/// actually acted on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanReport {
    /// Files the plan reads.
    pub files_read: BTreeSet<FileId>,
    /// How much of each file the plan reads.
    ///
    /// The composed scope map — a file absent from it was not part of a
    /// pruned plan at all; `Scope::Whole` entries are read end to end.
    /// Empty for a substitute or an unaided scan.
    pub scopes: BTreeMap<FileId, Scope>,
    /// Which derived state was used, if any.
    ///
    /// Plural because prunes compose: `a = 1 AND b = 2` can be served by two
    /// indexes whose candidate file sets intersected, and each is named here.
    /// A piece is named only if it actually narrowed the plan.
    pub used: Vec<String>,
    /// Files read because they were added after that derived state was built.
    pub also_scanned: BTreeSet<FileId>,
    /// Whether stored rows were read in place of the table.
    pub substituted: bool,
    /// Whether the served answer is an estimate rather than exact.
    ///
    /// `true` when stored sketch partials answered a measure — the query
    /// asked `approx_distinct`, or asked `count(distinct)` with the
    /// session's `quarry.approximate` opt-in. Never set by omission.
    pub approximate: bool,
    /// Identity of the plan that was looked up.
    ///
    /// A caller wanting to *store* this query's result needs the same key the
    /// table will look it up under, so it is reported rather than left to be
    /// recomputed.
    pub plan_hash: u64,
    /// What the query would have cost unaided.
    ///
    /// The sum of the live data files' sizes at the queried snapshot, known
    /// exactly from metadata. This is what makes a saving measurable instead
    /// of estimated — see [`crate::workload`].
    pub bytes_if_full_scan: u64,
    /// The shape of the query, with literals stripped.
    pub fingerprint: Fingerprint,
    /// What this scan computes, when it can be described exactly.
    ///
    /// A caller storing this query's result needs it: a stored result is
    /// matched by plan, not by name.
    pub plan: Option<Plan>,
    /// What the query aggregated, if it did.
    ///
    /// Set only when the whole plan was visible — `Session::sql` sees the
    /// `GROUP BY` that `scan` cannot — and `None` otherwise means nothing
    /// more than "no aggregate was visible".
    pub aggregate: Option<crate::workload::AggregateAsk>,
    /// The filters this scan ran, kept for the optimizer's proposals.
    ///
    /// Each is a [`FilterAsk`](crate::workload::FilterAsk): the canonical
    /// filter plus the SQL that re-executes it. A filter that cannot be
    /// unparsed is absent — it cannot be rebuilt, so proposing it would be
    /// noise.
    pub filters: Vec<crate::workload::FilterAsk>,
}

impl ScanReport {
    /// Bytes this scan expects to read.
    ///
    /// Derived-state bytes are not counted: what a query avoids is measured
    /// against the table, and a piece of derived state small enough to be
    /// worth using is negligible beside the files it replaced.
    pub fn bytes_planned(&self, sizes: &BTreeMap<FileId, u64>) -> u64 {
        self.files_read
            .iter()
            .filter_map(|file| sizes.get(file))
            .sum()
    }

    /// What to record about this scan.
    pub fn observation(&self, bytes_read: u64) -> Observation {
        Observation {
            fingerprint: self.fingerprint.clone(),
            aggregate: self.aggregate.clone(),
            filters: self.filters.clone(),
            bytes_read,
            bytes_if_full_scan: self.bytes_if_full_scan,
            used: self.used.iter().cloned().map(DerivedId).collect(),
        }
    }
}

/// Where a table's data actually lives.
#[derive(Debug)]
enum Files {
    /// Batches held in memory. Convenient for tests and examples, and enough
    /// to exercise every decision the rule makes.
    Memory(BTreeMap<FileId, Vec<RecordBatch>>),
    /// Parquet objects in an object store, addressed by path.
    ///
    /// [`FileId`] holds the object path, so the identity the rule reasons
    /// about and the identity the store reads are the same string. That is
    /// what an Iceberg data file path will be too.
    Parquet {
        url: ObjectStoreUrl,
        sizes: BTreeMap<FileId, u64>,
    },
}

/// A table planned through the rule.
///
/// Stands in for an Iceberg table until the catalog arrives: the point is that
/// `scan` consults [`Derived::may_serve`](crate::derived::Derived::may_serve)
/// and reads only what it permits, which is the same path a real table will
/// take.
#[derive(Debug)]
pub struct QuarryTable {
    schema: SchemaRef,
    table: TableId,
    snapshot: SnapshotId,
    graph: SnapshotGraph,
    /// Shared, because derived state changes while the table lives.
    ///
    /// The optimizer registers what it builds and drops what it retires
    /// without the table being rebuilt — which is the difference between a
    /// system that *can* be optimized and one that optimizes itself.
    registry: SharedRegistry,
    files: Files,
    field_ids: BTreeMap<String, FieldId>,
    policy: PolicyFingerprint,
    prices: PriceTable,
    /// Per-file value ranges per field, when the catalog records them.
    ///
    /// Iceberg keeps a lower and upper bound per field per data file, which is
    /// what decides whether an index could beat the format's own pruning. It
    /// belongs on the table because it describes the table's files, and it is
    /// empty for a table built by hand — in which case the optimizer declines
    /// to judge rather than guessing.
    field_bounds: BTreeMap<FieldId, Vec<(f64, f64)>>,
    /// Stored rows for substituting derived state, keyed by its id.
    ///
    /// Kept beside the registry rather than inside the kind so that no
    /// downcasting is needed: the registry decides *whether* derived state may
    /// be used, and the engine knows *how* to read it. Neither has to know the
    /// other's types.
    materialized: Mutex<BTreeMap<DerivedId, Vec<RecordBatch>>>,
    last_scan: Mutex<Option<ScanReport>>,
    /// Per-group row counts for Parquet files already opened.
    ///
    /// A scoped prune needs them at plan time — a `ParquetAccessPlan` must
    /// name exactly the file's groups, and a row-level selection must know
    /// each group's length — and the footer holding them is a few kilobytes
    /// read once per file through the session's store, so it is metered
    /// like any other read.
    row_groups: Mutex<BTreeMap<FileId, Vec<i64>>>,
}

impl QuarryTable {
    /// A table with no files and no derived state.
    ///
    /// `field_ids` maps column names to Iceberg field ids. Derived state is
    /// keyed on field ids, never names, so that renaming a column cannot
    /// silently mismatch it.
    pub fn new(
        schema: SchemaRef,
        table: TableId,
        snapshot: SnapshotId,
        graph: SnapshotGraph,
        field_ids: BTreeMap<String, FieldId>,
    ) -> Self {
        QuarryTable {
            schema,
            table,
            snapshot,
            graph,
            registry: shared(Registry::new()),
            files: Files::Memory(BTreeMap::new()),
            field_ids,
            policy: PolicyFingerprint(0),
            prices: PriceTable::default(),
            field_bounds: BTreeMap::new(),
            materialized: Mutex::new(BTreeMap::new()),
            last_scan: Mutex::new(None),
            row_groups: Mutex::new(BTreeMap::new()),
        }
    }

    /// Read data files as Parquet objects from `url` instead of from memory.
    ///
    /// Discards any in-memory files already added.
    pub fn on_object_store(mut self, url: ObjectStoreUrl) -> Self {
        self.files = Files::Parquet {
            url,
            sizes: BTreeMap::new(),
        };
        self
    }

    /// Add an in-memory data file's contents.
    ///
    /// Ignored if this table reads Parquet; see
    /// [`QuarryTable::with_parquet_file`].
    pub fn with_file(mut self, file: FileId, batches: Vec<RecordBatch>) -> Self {
        if let Files::Memory(files) = &mut self.files {
            files.insert(file, batches);
        }
        self
    }

    /// Register a Parquet object as a data file.
    ///
    /// `file` is the object path within the store, and `size` its length in
    /// bytes, which the Parquet reader needs in order to locate the footer.
    pub fn with_parquet_file(mut self, file: FileId, size: u64) -> Self {
        if let Files::Parquet { sizes, .. } = &mut self.files {
            sizes.insert(file, size);
        }
        self
    }

    /// Supply the stored rows for a piece of substituting derived state.
    ///
    /// The id must match one registered in the [`Registry`]; the rule decides
    /// whether it may be used, and these are the rows read when it is.
    ///
    /// Rows are stored **table-shaped** — the table's full schema, projected
    /// above the scan — and a batch that is not is rejected here rather than
    /// left to fail inside Arrow mid-query. What cannot be checked is whether
    /// the rows are the *complete* answer: verifying that would take the scan
    /// being avoided, so it stays the caller's precondition, documented on
    /// [`MaterializedResult::rows_of`](super::MaterializedResult::rows_of).
    ///
    /// # Panics
    ///
    /// If any batch's schema is not the table's.
    pub fn with_materialized(self, id: DerivedId, batches: Vec<RecordBatch>) -> Self {
        for batch in &batches {
            assert_eq!(
                batch.schema().as_ref(),
                self.schema.as_ref(),
                "stored rows for {id:?} must be table-shaped: the full schema, \
                 projected above the scan"
            );
        }
        self.materialized
            .lock()
            .expect("materialized")
            .insert(id, batches);
        self
    }

    /// Supply a cube's stored rows: partial aggregates at `rollup`'s grain.
    ///
    /// The stored schema must be exactly the rollup's columns, in order — a
    /// row that does not match its own spec fails here, where the mistake was
    /// made, rather than inside the re-aggregation mid-query.
    ///
    /// # Panics
    ///
    /// If any batch's column names are not `rollup`'s.
    pub fn with_cube(
        self,
        id: DerivedId,
        rollup: &super::Rollup,
        batches: Vec<RecordBatch>,
    ) -> Self {
        for batch in &batches {
            let names: Vec<String> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            assert_eq!(
                names,
                rollup.columns(),
                "cube rows for {id:?} must match their spec"
            );
        }
        self.materialized
            .lock()
            .expect("materialized")
            .insert(id, batches);
        self
    }

    /// Supply a projection's stored rows: a subset of the table's columns.
    ///
    /// Each batch is canonicalised before storing — reordered to the
    /// table's column order and rebuilt under the table's own fields — so a
    /// union with scanned files sees one schema, metadata included. A
    /// column the table does not have, or one with a different type, is
    /// rejected here rather than left to fail mid-query.
    ///
    /// # Panics
    ///
    /// If any batch holds a column that is not a table column.
    pub fn with_projection(self, id: DerivedId, batches: Vec<RecordBatch>) -> Self {
        let canonical = batches
            .iter()
            .map(|batch| {
                // (table index, stored index) pairs, sorted by table order.
                let mut order: Vec<(usize, usize)> = batch
                    .schema()
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(stored, field)| {
                        let table = self.schema.index_of(field.name()).unwrap_or_else(|_| {
                            panic!("stored column {:?} is not a table column", field.name())
                        });
                        assert_eq!(
                            field.data_type(),
                            self.schema.field(table).data_type(),
                            "stored column {:?} has a different type than the table's",
                            field.name()
                        );
                        (table, stored)
                    })
                    .collect();
                order.sort();
                order.dedup();
                let schema = Arc::new(
                    self.schema
                        .project(&order.iter().map(|(table, _)| *table).collect::<Vec<_>>())
                        .expect("stored columns are table columns"),
                );
                RecordBatch::try_new(
                    schema,
                    order
                        .iter()
                        .map(|(_, stored)| batch.column(*stored).clone())
                        .collect(),
                )
                .expect("canonicalised batch")
            })
            .collect();
        self.materialized
            .lock()
            .expect("materialized")
            .insert(id, canonical);
        self
    }

    /// The policy this table enforces.
    pub fn policy(&self) -> PolicyFingerprint {
        self.policy
    }

    /// Use this registry of derived state.
    pub fn with_registry(mut self, registry: Registry) -> Self {
        self.registry = shared(registry);
        self
    }

    /// Share an existing registry, so changes to it reach this table.
    pub fn with_shared_registry(mut self, registry: SharedRegistry) -> Self {
        self.registry = registry;
        self
    }

    /// The registry this table consults.
    pub fn registry(&self) -> &SharedRegistry {
        &self.registry
    }

    /// Read as this principal.
    pub fn with_policy(mut self, policy: PolicyFingerprint) -> Self {
        self.policy = policy;
        self
    }

    /// The rule's decision for the most recent scan.
    pub fn last_scan(&self) -> Option<ScanReport> {
        self.last_scan.lock().expect("scan report lock").clone()
    }

    /// Where this table's Parquet objects are, and how big they are.
    ///
    /// `None` for an in-memory table, which has no objects to read.
    pub fn parquet_files(&self) -> Option<(&ObjectStoreUrl, &BTreeMap<FileId, u64>)> {
        match &self.files {
            Files::Parquet { url, sizes } => Some((url, sizes)),
            Files::Memory(_) => None,
        }
    }

    /// The field id a column name maps to.
    pub fn field_id_of(&self, column: &str) -> Option<FieldId> {
        self.field_ids.get(column).copied()
    }

    /// The column a field id names.
    ///
    /// The inverse of [`QuarryTable::field_id_of`], needed to go from a
    /// proposal — which speaks in field ids, because derived state is keyed on
    /// them — back to something a reader can project.
    pub fn column_of(&self, field: FieldId) -> Option<&str> {
        self.field_ids
            .iter()
            .find(|(_, id)| **id == field)
            .map(|(name, _)| name.as_str())
    }

    /// The snapshot this table reads.
    pub fn snapshot(&self) -> SnapshotId {
        self.snapshot
    }

    /// Record per-file value ranges per field, as a catalog reports them.
    pub fn with_field_bounds(mut self, bounds: BTreeMap<FieldId, Vec<(f64, f64)>>) -> Self {
        self.field_bounds = bounds;
        self
    }

    /// How many data files the queried snapshot holds.
    pub fn file_count(&self) -> u64 {
        self.graph
            .get(self.snapshot)
            .map(|snapshot| snapshot.files().len() as u64)
            .unwrap_or(0)
    }

    /// Per-file value ranges for one field, if the catalog recorded any.
    pub fn bounds_of(&self, field: FieldId) -> Option<&[(f64, f64)]> {
        self.field_bounds
            .get(&field)
            .filter(|ranges| !ranges.is_empty())
            .map(Vec::as_slice)
    }

    /// The table's snapshot history.
    pub fn graph(&self) -> &SnapshotGraph {
        &self.graph
    }

    /// Bytes of live data at the queried snapshot.
    ///
    /// The unaided cost of a query, and the denominator for asking how much of
    /// the table a stale piece of derived state can no longer help with.
    pub fn live_bytes(&self) -> u64 {
        let Some(snapshot) = self.graph.get(self.snapshot) else {
            return 0;
        };
        match &self.files {
            Files::Parquet { sizes, .. } => snapshot
                .files()
                .keys()
                .filter_map(|file| sizes.get(file))
                .sum(),
            Files::Memory(_) => 0,
        }
    }

    /// This table's identity.
    pub fn table_id(&self) -> &TableId {
        &self.table
    }

    /// What this table pays for reads.
    pub fn prices(&self) -> &PriceTable {
        &self.prices
    }

    /// The stored rows for a piece of substituting derived state.
    pub fn stored(&self, id: &DerivedId) -> Option<Vec<RecordBatch>> {
        self.materialized
            .lock()
            .expect("materialized")
            .get(id)
            .cloned()
    }

    /// Store rows for `id` after construction — how the optimizer makes what
    /// it built visible to the table it was built on.
    pub fn store_rows(&self, id: DerivedId, batches: Vec<RecordBatch>) {
        self.materialized
            .lock()
            .expect("materialized")
            .insert(id, batches);
    }

    /// Record a plan-level report: `Session::sql` can see the whole plan,
    /// including the aggregate a scan cannot, and reports through the same
    /// channel a scan does.
    pub fn note_scan(&self, report: ScanReport) {
        *self.last_scan.lock().expect("last_scan") = Some(report);
    }

    /// The aggregate query `group`/`aggr` over `filters` asks of this table,
    /// if it can be described exactly enough for a cube to match it.
    ///
    /// `None` means the shape is outside what a cube can be checked against
    /// — a group key that is an expression, a `DISTINCT` or `FILTER`ed
    /// measure, an unsupported function — and the caller runs the plan as
    /// written. Returning `Some` for anything else would be a wrong answer.
    pub fn aggregate_query(
        &self,
        group: &[Expr],
        aggr: &[Expr],
        filters: &[Expr],
        approximate: bool,
    ) -> Option<Query> {
        let group_by = group
            .iter()
            .map(|expr| match expr {
                Expr::Column(column) => self.field_ids.get(column.name()).copied(),
                _ => None,
            })
            .collect::<Option<BTreeSet<FieldId>>>()?;
        let measures = aggr
            .iter()
            .map(|expr| self.measure(expr))
            .collect::<Option<BTreeSet<Measure>>>()?;
        let aggregate = Aggregate { group_by, measures };
        let projected: BTreeSet<FieldId> = aggregate
            .group_by
            .iter()
            .copied()
            .chain(aggregate.measures.iter().filter_map(|m| m.field))
            .collect();
        let plan = Plan::new(
            projected.clone(),
            filters.iter().map(|expr| self.canonical_filter(expr)),
        );
        let plan_hash = {
            let mut hasher = StableHasher::new();
            (plan.clone(), aggregate.clone()).hash(&mut hasher);
            hasher.finish()
        };
        Some(Query {
            table: self.table.clone(),
            snapshot: self.snapshot,
            policy: self.policy,
            plan_hash,
            plan: Some(plan),
            projected,
            predicates: self.predicates(filters),
            aggregate: Some(aggregate),
            nearest: None,
            approximate,
        })
    }

    /// One measure expression, or `None` if it is not one a cube can hold.
    ///
    /// Only a plain `func(column)` or `count(*)` qualifies: a `FILTER` or an
    /// expression argument computes something a stored partial cannot
    /// reproduce. `DISTINCT` qualifies for `count` alone — it asks for the
    /// estimate a sketch cube stores — and `approx_distinct` is that
    /// sketch's own name.
    pub(crate) fn measure(&self, expr: &Expr) -> Option<Measure> {
        let Expr::AggregateFunction(aggregate) = expr else {
            return None;
        };
        if aggregate.params.filter.is_some() || aggregate.params.order_by.is_some() {
            return None;
        }
        let func = match (aggregate.func.name(), aggregate.params.distinct) {
            ("count", true) => AggFunc::CountDistinct,
            ("count", false) => AggFunc::Count,
            ("sum", false) => AggFunc::Sum,
            ("min", false) => AggFunc::Min,
            ("max", false) => AggFunc::Max,
            ("approx_distinct" | "approx_count_distinct", false) => AggFunc::ApproxDistinct,
            _ => return None,
        };
        let field = match aggregate.params.args.as_slice() {
            // `count(*)` lands as `count(1)` after analysis: a literal arg is
            // a row count, not a field count.
            [Expr::Literal(..)] | [] if func == AggFunc::Count => None,
            [Expr::Column(column)] => Some(*self.field_ids.get(column.name())?),
            _ => return None,
        };
        Some(Measure { func, field })
    }

    /// Translate DataFusion filters into the predicates the rule understands.
    ///
    /// A comparison of a known column against a literal becomes
    /// [`Predicate::Eq`]; the operands are normalised so that `a = 1` and
    /// `1 = a` produce the same predicate. Anything else touching a known
    /// column becomes [`Predicate::Opaque`], which marks the column filtered
    /// without claiming it can be probed. Filters on unknown columns are
    /// dropped, since no derived state is keyed on them.
    fn predicates(&self, filters: &[Expr]) -> Vec<Predicate> {
        filters
            .iter()
            .filter_map(|expr| self.predicate(expr))
            .collect()
    }

    fn predicate(&self, expr: &Expr) -> Option<Predicate> {
        if let Expr::BinaryExpr(binary) = expr {
            let flipped = match (binary.left.as_ref(), binary.right.as_ref()) {
                (Expr::Column(c), Expr::Literal(v, _)) => Some((c, v)),
                (Expr::Literal(v, _), Expr::Column(c)) => Some((c, v)),
                _ => None,
            };
            if let Some((column, value)) = flipped {
                let field = *self.field_ids.get(column.name())?;
                return Some(match binary.op {
                    Operator::Eq => Predicate::Eq {
                        field,
                        value: hash_scalar(value),
                    },
                    _ => Predicate::Opaque { field },
                });
            }
        }
        // Not a shape we model: mark every known column it mentions as
        // filtered, so a kind needing an unfiltered column will not match.
        let mut columns = Vec::new();
        expr.apply(|node| {
            if let Expr::Column(c) = node {
                if let Some(field) = self.field_ids.get(c.name()) {
                    columns.push(*field);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .ok()?;
        columns
            .first()
            .map(|field| Predicate::Opaque { field: *field })
    }

    /// Which files this scan must read, and why.
    ///
    /// Consults the rule, and falls back to every live file when no derived
    /// state applies. A full scan is always a correct answer, so every
    /// failure path here leads to one.
    fn plan_files(
        &self,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        approximate: bool,
    ) -> ScanReport {
        let live: BTreeSet<FileId> = self
            .graph
            .get(self.snapshot)
            .map(|s| s.files().keys().cloned().collect())
            .unwrap_or_default();

        let plan_hash = self.plan_hash(projection, filters);
        let query = Query {
            table: self.table.clone(),
            snapshot: self.snapshot,
            policy: self.policy,
            plan_hash,
            plan: self.exact_plan(projection, filters),
            projected: self.projected_fields(projection),
            predicates: self.predicates(filters),
            aggregate: None,
            nearest: None,
            approximate,
        };
        let fingerprint = Fingerprint::of(&query);

        // Each conjunct kept in both its forms: canonical for matching,
        // SQL for rebuilding. One that cannot unparse cannot be built, so
        // it is not worth observing.
        let asks: Vec<crate::workload::FilterAsk> = filters
            .iter()
            .filter_map(|expr| {
                let sql = datafusion::sql::unparser::expr_to_sql(expr)
                    .ok()?
                    .to_string();
                Some(crate::workload::FilterAsk {
                    table: self.table.clone(),
                    filter: self.canonical_filter(expr),
                    sql,
                })
            })
            .collect();

        // The unaided baseline, known exactly rather than estimated: every
        // live data file's size. Only available where sizes are known, which
        // is the Parquet path; an in-memory table reports zero and its
        // observations are correspondingly uninformative.
        let bytes_if_full_scan = match &self.files {
            Files::Parquet { sizes, .. } => live.iter().filter_map(|file| sizes.get(file)).sum(),
            Files::Memory(_) => 0,
        };

        // The cheapest candidate the engine can actually execute. A
        // substituting one is executable only if its rows were supplied to
        // `with_materialized`; otherwise it is skipped, which is the safe
        // direction — the answer is right, merely slower.
        // A read guard, held only across planning, which does no I/O.
        let registry = self.registry.read().expect("registry lock");
        let materialized = self.materialized.lock().expect("materialized");
        let usable = match compose(
            &registry.candidates(&query, &self.graph, &self.prices),
            |d| materialized.contains_key(&d.id),
        ) {
            Composed::Substitute(candidate) => {
                let also_scanned = match candidate.decision {
                    Decision::UseWith { also_scan, .. } => also_scan,
                    _ => BTreeSet::new(),
                };
                Some(ScanReport {
                    aggregate: None,
                    filters: asks.clone(),
                    files_read: BTreeSet::new(),
                    scopes: BTreeMap::new(),
                    used: vec![candidate.derived.id.0.clone()],
                    also_scanned,
                    substituted: true,
                    approximate: false,
                    plan_hash,
                    bytes_if_full_scan,
                    fingerprint: fingerprint.clone(),
                    plan: query.plan.clone(),
                })
            }
            Composed::Pruned {
                files,
                residual,
                pieces,
            } => Some(ScanReport {
                aggregate: None,
                filters: asks.clone(),
                scopes: files.clone(),
                files_read: files.into_keys().collect(),
                used: pieces.iter().map(|p| p.derived.id.0.clone()).collect(),
                also_scanned: residual,
                substituted: false,
                approximate: false,
                plan_hash,
                bytes_if_full_scan,
                fingerprint: fingerprint.clone(),
                plan: query.plan.clone(),
            }),
            Composed::Scan => None,
        };

        match usable {
            None => ScanReport {
                aggregate: None,
                filters: asks,
                files_read: live,
                scopes: BTreeMap::new(),
                used: Vec::new(),
                also_scanned: BTreeSet::new(),
                substituted: false,
                approximate: false,
                plan_hash,
                bytes_if_full_scan,
                fingerprint,
                plan: query.plan.clone(),
            },
            Some(mut report) => {
                report
                    .files_read
                    .extend(report.also_scanned.iter().cloned());
                report
            }
        }
    }

    fn projected_fields(&self, projection: Option<&Vec<usize>>) -> BTreeSet<FieldId> {
        let names: Vec<&str> = match projection {
            None => self
                .schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect(),
            Some(indices) => indices
                .iter()
                .filter_map(|i| self.schema.fields().get(*i).map(|f| f.name().as_str()))
                .collect(),
        };
        names
            .into_iter()
            .filter_map(|name| self.field_ids.get(name).copied())
            .collect()
    }

    /// An exact description of this scan, or `None` if one cannot be made.
    ///
    /// Substituting derived state is admitted only on an exact match, so this
    /// must never approximate. It returns `None` when a projected column has
    /// no field id, because the mapping is not injective in that case: an
    /// unknown column silently vanishes, and two different projections would
    /// describe identically.
    fn exact_plan(&self, projection: Option<&Vec<usize>>, filters: &[Expr]) -> Option<Plan> {
        let mut projected = BTreeSet::new();
        let indices: Vec<usize> = match projection {
            None => (0..self.schema.fields().len()).collect(),
            Some(indices) => indices.clone(),
        };
        for index in indices {
            let field = self.schema.fields().get(index)?;
            projected.insert(self.field_ids.get(field.name()).copied()?);
        }

        Some(Plan::new(
            projected,
            filters.iter().map(|expr| self.canonical_filter(expr)),
        ))
    }

    /// A faithful rendering of one filter.
    ///
    /// `Debug` rather than `Display`, because `Display` erases types: an
    /// `Int64(1)` and a `Utf8("1")` both print as `1`, so two different
    /// filters would render the same string and compare equal. `Debug` names
    /// the variant, which makes the rendering injective for the purpose it is
    /// used for.
    ///
    /// Column-against-literal comparisons are normalised to a fixed operand
    /// order, and only for the symmetric operators, so that `status = 500` and
    /// `500 = status` are one plan. Failing to normalise something costs a
    /// missed match; normalising it *wrongly* would cost a wrong answer, so
    /// anything else is left exactly as written.
    pub(crate) fn canonical_filter(&self, expr: &Expr) -> Filter {
        if let Expr::BinaryExpr(binary) = expr {
            let symmetric = matches!(binary.op, Operator::Eq | Operator::NotEq);
            let mut sides = vec![(binary.left.as_ref(), binary.right.as_ref())];
            if symmetric {
                // `1 = a` and `a = 1` are the same predicate; render both as
                // `a = 1`. Asymmetric operators keep their operand order.
                sides.push((binary.right.as_ref(), binary.left.as_ref()));
            }
            for (column, value) in sides {
                if let (Expr::Column(column), Expr::Literal(value, _)) = (column, value) {
                    return Filter {
                        field: self.field_ids.get(column.name()).copied(),
                        text: format!("{} {} {:?}", column.name(), binary.op, value),
                    };
                }
            }
        }
        // Not column-against-literal: the field it restricts is the first
        // known column it mentions, or none if it mentions none.
        let mut field = None;
        let _ = expr.apply(|node| {
            if field.is_none() {
                if let Expr::Column(c) = node {
                    field = self.field_ids.get(c.name()).copied();
                }
            }
            Ok(TreeNodeRecursion::Continue)
        });
        Filter {
            field,
            text: format!("{expr:?}"),
        }
    }

    /// A short name for this scan's plan.
    ///
    /// Hashes the *exact* plan, not the lossy predicate translation, so the
    /// name at least corresponds to the identity it labels. Queries with no
    /// exact plan all share one value, which is harmless: nothing decides
    /// anything from this, and matching compares plans.
    fn plan_hash(&self, projection: Option<&Vec<usize>>, filters: &[Expr]) -> u64 {
        let mut hasher = StableHasher::new();
        self.table.0.hash(&mut hasher);
        self.exact_plan(projection, filters).hash(&mut hasher);
        hasher.finish()
    }
}

#[async_trait]
impl TableProvider for QuarryTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Filters are used to *prune*, never to filter rows, so DataFusion must
    /// re-apply them above the scan.
    ///
    /// [`TableProviderFilterPushDown::Inexact`] says exactly that. Claiming
    /// `Exact` would be wrong: pruning is conservative, so a file the rule
    /// admits still contains rows the predicate rejects.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let report = self.plan_files(projection, filters, super::options::approximate(state));

        // Stored rows first when substituting, then whatever files the rule
        // says must still be read. Concatenating the two is only valid
        // because the rule refused any derived state that cannot supply
        // exactly the projected columns.
        let mut parts: Vec<Arc<dyn ExecutionPlan>> = Vec::new();

        if report.substituted {
            if let Some(batches) = report
                .used
                .first()
                .map(|id| DerivedId(id.clone()))
                .and_then(|id| {
                    self.materialized
                        .lock()
                        .expect("materialized")
                        .get(&id)
                        .cloned()
                })
            {
                parts.push(self.derived_plan(&batches, projection)?);
            }
        }

        if !report.files_read.is_empty() {
            parts.push(
                self.file_plan(&report.files_read, &report.scopes, projection, state)
                    .await?,
            );
        }

        *self.last_scan.lock().expect("scan report lock") = Some(report);

        match parts.len() {
            // Nothing to read: an empty plan of the right shape, not an error.
            0 => self.memory_plan(&[], projection),
            1 => Ok(parts.remove(0)),
            _ => Ok(Arc::new(UnionExec::new(parts))),
        }
    }
}

impl QuarryTable {
    fn memory_plan(
        &self,
        partitions: &[Vec<RecordBatch>],
        projection: Option<&Vec<usize>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let exec = MemorySourceConfig::try_new_exec(
            partitions,
            Arc::clone(&self.schema),
            projection.cloned(),
        )?;
        Ok(exec)
    }

    /// A plan over stored rows, whose schema may be narrower than the
    /// table's — a projection's stored columns.
    ///
    /// `projection` is expressed in table-schema indices; when the stored
    /// schema is narrower it is remapped to stored positions by name, which
    /// is safe because the rule only let the state serve a query whose
    /// columns it covers. `with_projection` canonicalised the batches under
    /// the table's own fields, so the output schema is byte-identical to
    /// what a full scan would project.
    fn derived_plan(
        &self,
        batches: &[RecordBatch],
        projection: Option<&Vec<usize>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let Some(stored) = batches.first().map(|batch| batch.schema()) else {
            return self.memory_plan(std::slice::from_ref(&batches.to_vec()), projection);
        };
        if stored == self.schema {
            return self.memory_plan(std::slice::from_ref(&batches.to_vec()), projection);
        }

        let positions: Vec<usize> = stored
            .fields()
            .iter()
            .map(|field| self.schema.index_of(field.name()))
            .collect::<Result<_, _>>()?;
        let remapped = match projection {
            Some(indices) => indices
                .iter()
                .map(|index| {
                    positions
                        .iter()
                        .position(|position| position == index)
                        .ok_or_else(|| {
                            DataFusionError::Plan(format!(
                                "stored projection does not hold column {:?}",
                                self.schema.field(*index).name()
                            ))
                        })
                })
                .collect::<DfResult<Vec<_>>>()?,
            None => {
                return Err(DataFusionError::Plan(
                    "a stored projection cannot serve a whole-table scan".to_owned(),
                ));
            }
        };
        let exec = MemorySourceConfig::try_new_exec(&[batches.to_vec()], stored, Some(remapped))?;
        Ok(exec)
    }

    /// A plan reading exactly `files` — and within them, only the row
    /// groups `groups` names — from wherever this table's data lives.
    async fn file_plan(
        &self,
        files: &BTreeSet<FileId>,
        scopes: &BTreeMap<FileId, Scope>,
        projection: Option<&Vec<usize>>,
        state: &dyn Session,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        match &self.files {
            Files::Memory(available) => {
                let partitions: Vec<Vec<RecordBatch>> = files
                    .iter()
                    .filter_map(|file| {
                        let batches = available.get(file)?;
                        Some(match scopes.get(file) {
                            None | Some(Scope::Whole) => batches.clone(),
                            // Batches play the role of row groups.
                            Some(Scope::Groups(groups)) => batches
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| groups.contains(&(*i as u32)))
                                .map(|(_, batch)| batch.clone())
                                .collect(),
                            Some(Scope::Rows(rows)) => batches
                                .iter()
                                .enumerate()
                                .filter_map(|(i, batch)| {
                                    let rows = rows.get(&(i as u32))?;
                                    take_rows(batch, rows)
                                })
                                .collect(),
                        })
                    })
                    .collect();
                self.memory_plan(&partitions, projection)
            }
            Files::Parquet { url, sizes } => {
                let mut builder = FileScanConfigBuilder::new(
                    url.clone(),
                    Arc::clone(&self.schema),
                    Arc::new(ParquetSource::default()),
                )
                .with_projection(projection.cloned());

                // One file group per file, so DataFusion can read them in
                // parallel and the plan shows exactly what was selected.
                for file in files {
                    let size = sizes.get(file).copied().unwrap_or(0);
                    let mut partitioned = PartitionedFile::new(file.0.clone(), size);
                    match scopes.get(file) {
                        None | Some(Scope::Whole) => {}
                        Some(scope) => {
                            let groups = self.row_group_rows(state, url, file, size).await?;
                            let mut plan = ParquetAccessPlan::new_all(groups.len());
                            for (group, &len) in groups.iter().enumerate() {
                                match scope {
                                    Scope::Whole => {}
                                    Scope::Groups(admit) => {
                                        if !admit.contains(&(group as u32)) {
                                            plan.skip(group);
                                        }
                                    }
                                    Scope::Rows(rows) => match rows.get(&(group as u32)) {
                                        Some(rows) => {
                                            plan.scan_selection(group, row_selection(rows, len));
                                        }
                                        None => plan.skip(group),
                                    },
                                }
                            }
                            partitioned = partitioned.with_extensions(Arc::new(plan));
                        }
                    }
                    builder = builder.with_file_group(vec![partitioned].into());
                }

                Ok(DataSourceExec::from_data_source(builder.build()))
            }
        }
    }

    /// A file's per-group row counts, read once from its footer and cached.
    ///
    /// The footer is a few kilobytes at the file's tail; the read resolves
    /// through the session's registered store, so it is metered like any
    /// other — and `ParquetAccessPlan` is strict about it: a plan naming the
    /// wrong number of groups fails rather than guessing.
    async fn row_group_rows(
        &self,
        state: &dyn Session,
        url: &ObjectStoreUrl,
        file: &FileId,
        size: u64,
    ) -> DfResult<Vec<i64>> {
        if let Some(groups) = self.row_groups.lock().expect("row-group cache").get(file) {
            return Ok(groups.clone());
        }
        let store = state.runtime_env().object_store(url)?;
        let mut reader =
            ParquetObjectReader::new(store, ObjectPath::from(file.0.as_str())).with_file_size(size);
        let metadata = ArrowReaderMetadata::load_async(&mut reader, Default::default()).await?;
        let groups: Vec<i64> = metadata
            .metadata()
            .row_groups()
            .iter()
            .map(|group| group.num_rows())
            .collect();
        self.row_groups
            .lock()
            .expect("row-group cache")
            .insert(file.clone(), groups.clone());
        Ok(groups)
    }
}

/// The `RowSelection` admitting `rows` of a group `total` rows long.
///
/// Admit-only, like the scope it came from: an offset past the end cannot
/// name a real row and is ignored rather than erroring.
fn row_selection(rows: &BTreeSet<u64>, total: i64) -> RowSelection {
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for &row in rows {
        if row as i64 >= total {
            continue;
        }
        let row = row as usize;
        match ranges.last_mut() {
            Some(last) if last.end == row => last.end += 1,
            _ => ranges.push(row..row + 1),
        }
    }
    RowSelection::from_consecutive_ranges(ranges.into_iter(), total as usize)
}

/// `batch` restricted to the named rows, or `None` if it holds none of them.
fn take_rows(batch: &RecordBatch, rows: &BTreeSet<u64>) -> Option<RecordBatch> {
    let indices: Vec<u64> = rows
        .iter()
        .cloned()
        .filter(|row| *row < batch.num_rows() as u64)
        .collect();
    if indices.is_empty() {
        return None;
    }
    let indices = UInt64Array::from(indices);
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column, &indices, None).expect("indices are in range"))
        .collect();
    Some(RecordBatch::try_new(batch.schema(), columns).expect("taken batch"))
}
