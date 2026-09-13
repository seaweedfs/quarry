//! A [`TableProvider`] whose scan is planned by the rule.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};

use async_trait::async_trait;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::Result as DfResult;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::datasource::physical_plan::{FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::union::UnionExec;
use datafusion::scalar::ScalarValue;

use crate::cost::PriceTable;
use crate::derived::{Decision, DerivedId, FieldId, PolicyFingerprint, Predicate, Query, Rewrite};
use crate::registry::Registry;
use crate::snapshot::{FileId, SnapshotGraph, SnapshotId, TableId};
use crate::workload::{Fingerprint, Observation};

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
/// Not stable across Rust releases, so derived state that outlives a process
/// will eventually need a fixed hash rather than [`DefaultHasher`]; in-memory
/// state does not care yet.
pub fn hash_scalar(value: &ScalarValue) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
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
    /// Which derived state was used, if any.
    pub used: Option<String>,
    /// Files read because they were added after that derived state was built.
    pub also_scanned: BTreeSet<FileId>,
    /// Whether stored rows were read in place of the table.
    pub substituted: bool,
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
            bytes_read,
            bytes_if_full_scan: self.bytes_if_full_scan,
            used: self.used.clone().map(DerivedId),
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
    /// Stored rows for substituting derived state, keyed by its id.
    ///
    /// Kept beside the registry rather than inside the kind so that no
    /// downcasting is needed: the registry decides *whether* derived state may
    /// be used, and the engine knows *how* to read it. Neither has to know the
    /// other's types.
    materialized: BTreeMap<DerivedId, Vec<RecordBatch>>,
    last_scan: Mutex<Option<ScanReport>>,
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
            materialized: BTreeMap::new(),
            last_scan: Mutex::new(None),
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
    pub fn with_materialized(mut self, id: DerivedId, batches: Vec<RecordBatch>) -> Self {
        self.materialized.insert(id, batches);
        self
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
    fn plan_files(&self, projection: Option<&Vec<usize>>, filters: &[Expr]) -> ScanReport {
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
            projected: self.projected_fields(projection),
            predicates: self.predicates(filters),
        };
        let fingerprint = Fingerprint::of(&query);

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
        let usable = registry
            .candidates(&query, &self.graph, &self.prices)
            .into_iter()
            .find_map(|candidate| {
                let used = candidate.derived.id.0.clone();
                let (rewrite, also_scanned) = match candidate.decision {
                    Decision::Use(rewrite) => (rewrite, BTreeSet::new()),
                    Decision::UseWith { rewrite, also_scan } => (rewrite, also_scan),
                    Decision::Reject(_) => return None,
                };
                match rewrite {
                    Rewrite::Prune { files } => Some(ScanReport {
                        files_read: files,
                        used: Some(used),
                        also_scanned,
                        substituted: false,
                        plan_hash,
                        bytes_if_full_scan,
                        fingerprint: fingerprint.clone(),
                    }),
                    Rewrite::Substitute { .. }
                        if self.materialized.contains_key(&candidate.derived.id) =>
                    {
                        Some(ScanReport {
                            files_read: BTreeSet::new(),
                            used: Some(used),
                            also_scanned,
                            substituted: true,
                            plan_hash,
                            bytes_if_full_scan,
                            fingerprint: fingerprint.clone(),
                        })
                    }
                    Rewrite::Substitute { .. } => None,
                }
            });

        match usable {
            None => ScanReport {
                files_read: live,
                used: None,
                also_scanned: BTreeSet::new(),
                substituted: false,
                plan_hash,
                bytes_if_full_scan,
                fingerprint,
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

    /// A stand-in for a canonical plan hash.
    ///
    /// Hashes the projected fields and the translated predicates rather than
    /// the `Expr` tree, which makes it insensitive to spelling — `a = 1` and
    /// `1 = a` agree. It is *not* yet canonical in the full sense the design
    /// asks for: a query that a rewrite could reduce to this one will hash
    /// differently, so the result cache under-hits rather than mis-hits.
    fn plan_hash(&self, projection: Option<&Vec<usize>>, filters: &[Expr]) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.table.0.hash(&mut hasher);
        self.projected_fields(projection).hash(&mut hasher);
        let mut predicates = self.predicates(filters);
        predicates.sort_by_key(|p| p.field());
        for predicate in predicates {
            match predicate {
                Predicate::Eq { field, value } => (0u8, field, value).hash(&mut hasher),
                Predicate::Opaque { field } => (1u8, field, 0u64).hash(&mut hasher),
            }
        }
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
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let report = self.plan_files(projection, filters);

        // Stored rows first when substituting, then whatever files the rule
        // says must still be read. Concatenating the two is only valid because
        // the rule refused any derived state whose rows are not table-shaped.
        let mut parts: Vec<Arc<dyn ExecutionPlan>> = Vec::new();

        if report.substituted {
            if let Some(batches) = report
                .used
                .as_deref()
                .map(|id| DerivedId(id.to_owned()))
                .and_then(|id| self.materialized.get(&id))
            {
                parts.push(self.memory_plan(std::slice::from_ref(batches), projection)?);
            }
        }

        if !report.files_read.is_empty() {
            parts.push(self.file_plan(&report.files_read, projection)?);
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

    /// A plan reading exactly `files`, from wherever this table's data lives.
    fn file_plan(
        &self,
        files: &BTreeSet<FileId>,
        projection: Option<&Vec<usize>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        match &self.files {
            Files::Memory(available) => {
                let partitions: Vec<Vec<RecordBatch>> = files
                    .iter()
                    .filter_map(|file| available.get(file).cloned())
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
                    builder = builder
                        .with_file_group(vec![PartitionedFile::new(file.0.clone(), size)].into());
                }

                Ok(DataSourceExec::from_data_source(builder.build()))
            }
        }
    }
}
