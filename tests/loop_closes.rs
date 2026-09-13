//! The loop, end to end, on real queries over real Parquet.
//!
//! Everything before this used derived state that a test decided to build.
//! Here the system decides: it watches queries, proposes an index, and once
//! that index exists it stops proposing one and starts crediting it with what
//! it measurably saved.
//!
//! ```sh
//! cargo test --features engine --test loop_closes
//! ```

#![cfg(feature = "engine")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::scalar::ScalarValue;
use object_store::local::LocalFileSystem;
use url::Url;

use quarry::cost::PriceTable;
use quarry::derived::{Derived, DerivedId, PolicyFingerprint, Source};
use quarry::engine::{
    Declined, Optimizer, Quarry, QuarryTable, build_index, build_proposed_index, estimate_overlap,
    hash_scalar, index_id, shared,
};
use quarry::kinds::Index;
use quarry::layout::Spread;
use quarry::registry::Registry;
use quarry::snapshot::{FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};
use quarry::workload::{Policy, Workload};

const TENANT_FIELD: u32 = 4;
const MESSAGE_FIELD: u32 = 7;
const POLICY: PolicyFingerprint = PolicyFingerprint(1);

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("tenant_id", DataType::Int64, false),
        Field::new("message", DataType::Utf8, false),
    ]))
}

fn field_ids() -> BTreeMap<String, u32> {
    BTreeMap::from([
        ("tenant_id".to_owned(), TENANT_FIELD),
        ("message".to_owned(), 7),
    ])
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// A data file big enough that avoiding it is worth measuring.
fn write_parquet(dir: &Path, name: &str, tenant: i64, rows: usize) -> (FileId, u64) {
    let path = dir.join(name);
    let file = fs::File::create(&path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema(), None).expect("writer");

    let tenants: Int64Array = (0..rows).map(|_| tenant).collect();
    let messages: StringArray = (0..rows)
        .map(|i| {
            Some(format!(
                "message number {i} with some padding to take up room"
            ))
        })
        .collect();
    let batch =
        RecordBatch::try_new(schema(), vec![Arc::new(tenants), Arc::new(messages)]).expect("batch");
    writer.write(&batch).expect("write");
    writer.close().expect("close");

    let size = fs::metadata(&path).expect("stat").len();
    (FileId(path.to_string_lossy().into_owned()), size)
}

struct Fixture {
    dir: PathBuf,
    sizes: BTreeMap<FileId, u64>,
    graph: SnapshotGraph,
    files: Vec<FileId>,
    at: SnapshotId,
}

impl Fixture {
    /// Commit a new snapshot appending one file of `rows` rows for `tenant`.
    ///
    /// What a table does between rounds, and what makes an index built earlier
    /// progressively less useful.
    fn append(&mut self, tenant: i64, rows: usize) -> FileId {
        let name = format!("appended-{}.parquet", self.files.len());
        let (file, size) = write_parquet(&self.dir, &name, tenant, rows);
        self.sizes.insert(file.clone(), size);
        self.files.push(file.clone());

        let next = SnapshotId(self.at.0 + 1);
        let mut snapshot = Snapshot::child_of(next, self.at);
        for existing in &self.files {
            snapshot = snapshot.with_clean_file(existing.clone());
        }
        self.graph.insert(snapshot);
        self.at = next;
        file
    }
}

/// Four files: tenant 1 in the first, tenants 2..5 in the rest.
fn fixture(name: &str) -> Fixture {
    let dir = scratch(name);
    let mut sizes = BTreeMap::new();
    let mut files = Vec::new();

    for (index, tenant) in [1i64, 2, 3, 4].into_iter().enumerate() {
        let (file, size) = write_parquet(&dir, &format!("{index}.parquet"), tenant, 500);
        sizes.insert(file.clone(), size);
        files.push(file);
    }

    let mut snapshot = Snapshot::root(SnapshotId(1));
    for file in &files {
        snapshot = snapshot.with_clean_file(file.clone());
    }

    Fixture {
        dir,
        sizes,
        graph: SnapshotGraph::new().with(snapshot),
        files,
        at: SnapshotId(1),
    }
}

fn table(fixture: &Fixture, registry: Registry) -> QuarryTable {
    let mut table = QuarryTable::new(
        schema(),
        TableId("events".into()),
        SnapshotId(1),
        fixture.graph.clone(),
        field_ids(),
    )
    .with_policy(POLICY)
    .with_registry(registry)
    .on_object_store(ObjectStoreUrl::local_filesystem());
    for (file, size) in &fixture.sizes {
        table = table.with_parquet_file(file.clone(), *size);
    }
    table
}

/// A policy for the tests that exercise the loop's *mechanism*.
///
/// `min_index_advantage_pct: 0.0` switches off the gate that asks whether an
/// index beats the file format's own pruning. These tests are about
/// propose/build/refresh/retire working at all, and this fixture puts one
/// tenant in each file — perfectly disjoint ranges, which is precisely the
/// case where measurement says an index is *not* worth building. Leaving the
/// gate on would make them all decline, correctly, and test nothing.
///
/// The judgement itself is covered by `layout.rs` and by
/// `the_gate_refuses_an_index_the_format_makes_redundant` below.
fn mechanism_policy(base: Policy) -> Policy {
    Policy {
        min_index_advantage_pct: 0.0,
        ..base
    }
}

fn quarry() -> Quarry {
    Quarry::new(
        Url::parse("file://").expect("url"),
        Arc::new(LocalFileSystem::new()),
    )
}

/// Run one query and record what it cost.
async fn run_and_observe(
    fixture: &Fixture,
    registry: Registry,
    workload: &mut Workload,
    sql: &str,
) -> usize {
    let table = Arc::new(table(fixture, registry));
    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::clone(&table))
        .expect("register");

    let rows: usize = session
        .sql(sql)
        .await
        .expect("query")
        .iter()
        .map(RecordBatch::num_rows)
        .sum();

    let report = table.last_scan().expect("a scan happened");
    // Bytes attributable to the table, from the plan rather than the socket:
    // the store also fetches Parquet footers, which are not what an index is
    // credited with avoiding.
    let bytes_read = report.bytes_planned(&fixture.sizes);
    workload.observe(report.observation(bytes_read));
    rows
}

/// An index written by hand, for the tests that are not about building.
///
/// `run_and_observe` needs a fresh registry per query because `QuarryTable`
/// owns one, and rebuilding by reading the data for each is wasteful when the
/// test is about what the index *does* rather than how it was made.
fn hand_written_index(fixture: &Fixture, field: u32, tenant: i64) -> Registry {
    let mut index = Index::new(field);
    // Only the first file holds tenant 1, which is what makes the index worth
    // having; a real builder would learn this by reading the data.
    index.insert(
        hash_scalar(&ScalarValue::Int64(Some(tenant))),
        fixture.files[0].clone(),
    );

    let mut registry = Registry::new();
    registry.register(Derived::new(
        DerivedId("tenant_idx".into()),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(1),
        },
        POLICY,
        4096,
        Box::new(index.with_bytes(4096)),
    ));
    registry
}

/// The whole loop with nothing hand-fed: the index is built by reading data.
#[tokio::test]
async fn the_loop_builds_its_own_index_from_the_data() {
    let fixture = fixture("loop_self_built");
    let prices = PriceTable::default();
    let sql = "SELECT * FROM events WHERE tenant_id = 1";
    let mut workload = Workload::new();

    // OBSERVE: unaided queries, reading the whole table each time.
    for _ in 0..10 {
        run_and_observe(&fixture, Registry::new(), &mut workload, sql).await;
    }

    // PROPOSE.
    let proposals = workload.proposals(&prices, 5);
    assert_eq!(proposals.len(), 1);
    let field = proposals[0].field;

    // BUILD, for real: read the proposed column off the objects.
    let quarry = quarry();
    let session = quarry.session();
    let bare = table(&fixture, Registry::new());
    let derived = build_proposed_index(
        &session,
        &bare,
        field,
        index_id(&TableId("events".into()), field),
        POLICY,
    )
    .await
    .expect("build the proposed index");

    // Building reads only the indexed column, so it costs a fraction of the
    // table. The fixture's other column is the bulky one.
    let build_bytes = quarry.origin_stats().bytes_fetched;
    let table_bytes: u64 = fixture.sizes.values().sum();
    assert!(
        build_bytes < table_bytes,
        "building read {build_bytes} of {table_bytes} bytes; projection \
         pushdown should have skipped the message column"
    );

    let built_id = derived.id.clone();
    assert!(derived.bytes > 0, "the index should price itself");

    let mut registry = Registry::new();
    registry.register(derived);

    // MEASURE: the same query, served by the index just built — not by a
    // hand-written stand-in. This is the step that makes the loop real.
    let served = Arc::new(table(&fixture, registry));
    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    let rows: usize = session
        .sql(sql)
        .await
        .expect("query")
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 500, "the answer must not change");

    let report = served.last_scan().expect("a scan happened");
    assert_eq!(
        report.used.as_deref(),
        Some(built_id.0.as_str()),
        "the self-built index should have served the query"
    );

    let mut after = Workload::new();
    after.observe(report.observation(report.bytes_planned(&fixture.sizes)));

    let credited = after.credited(&built_id).expect("the index served a query");
    assert!(
        credited.bytes_saved() > 0,
        "an index it built itself must still save bytes"
    );

    // STOP PROPOSING.
    assert!(after.proposals(&prices, 1).is_empty());
}

/// The optimizer driving itself: no caller sequences the steps.
#[tokio::test]
async fn the_optimizer_runs_the_loop_on_its_own() {
    let fixture = fixture("loop_driven");
    let sql = "SELECT * FROM events WHERE tenant_id = 1";
    let table_bytes: u64 = fixture.sizes.values().sum();

    // A shared registry, so what the optimizer builds reaches the table
    // without the table being rebuilt.
    let registry = shared(Registry::new());
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));

    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        mechanism_policy(Policy::automatic_pct(table_bytes, 5.0).with_min_queries(3)),
    )
    .for_reader(POLICY);

    let quarry = quarry();

    // Round 0: nothing observed, so nothing to do.
    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");
    let first = optimizer.round(&session, &served).await;
    assert_eq!(first, quarry::engine::Round::default(), "nothing to go on");

    // Queries run; the optimizer is told what they cost.
    for _ in 0..5 {
        let rows: usize = session
            .sql(sql)
            .await
            .expect("query")
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        assert_eq!(rows, 500);

        let report = served.last_scan().expect("a scan happened");
        optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
        assert_eq!(report.used, None, "nothing is built yet");
    }

    // Round 1: it proposes, builds, and registers, by itself.
    let round = optimizer.round(&session, &served).await;
    assert_eq!(round.built.len(), 1, "one index built: {round:?}");
    assert!(round.retired.is_empty());
    assert_eq!(registry.read().expect("lock").len(), 1);

    // The same table now uses it — no rebuild, because the registry is shared.
    let rows: usize = session
        .sql(sql)
        .await
        .expect("query")
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 500, "the answer must not change");

    let report = served.last_scan().expect("a scan happened");
    assert_eq!(
        report.used.as_deref(),
        Some(round.built[0].0.as_str()),
        "the index the optimizer built should now be serving queries"
    );
    optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));

    // Round 2: nothing left to do, and it does not rebuild what it made.
    let round = optimizer.round(&session, &served).await;
    assert!(round.built.is_empty(), "must not rebuild: {round:?}");
    assert!(round.retired.is_empty(), "it is earning its keep");
}

#[tokio::test]
async fn an_advisory_optimizer_proposes_but_changes_nothing() {
    let fixture = fixture("loop_advisory");
    let sql = "SELECT * FROM events WHERE tenant_id = 1";

    let registry = shared(Registry::new());
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
    let mut optimizer = Optimizer::new(Arc::clone(&registry), Policy::ADVISORY.with_min_queries(3));

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    for _ in 0..5 {
        session.sql(sql).await.expect("query");
        let report = served.last_scan().expect("scan");
        optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
    }

    assert_eq!(optimizer.proposals().len(), 1, "it has an opinion");

    let round = optimizer.round(&session, &served).await;
    assert!(round.built.is_empty(), "advisory must not act");
    assert_eq!(
        round.declined,
        vec![(
            round.declined[0].0.clone(),
            quarry::engine::Declined::Advisory
        )],
    );
    assert_eq!(registry.read().expect("lock").len(), 0);
}

#[tokio::test]
async fn a_budget_of_nothing_declines_the_build() {
    let fixture = fixture("loop_budget");
    let sql = "SELECT * FROM events WHERE tenant_id = 1";

    let registry = shared(Registry::new());
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
    // Automatic, but with no room to keep anything.
    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        mechanism_policy(Policy::automatic(0).with_min_queries(3)),
    );

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    for _ in 0..5 {
        session.sql(sql).await.expect("query");
        let report = served.last_scan().expect("scan");
        optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
    }

    let round = optimizer.round(&session, &served).await;
    assert!(round.built.is_empty());
    assert_eq!(
        round.declined.first().map(|(_, why)| *why),
        Some(quarry::engine::Declined::OverBudget),
        "the ceiling is checked on what was built, not on a guess"
    );
    assert_eq!(registry.read().expect("lock").len(), 0);
}

/// A table sharing `registry`, so the optimizer's builds reach it.
///
/// Reads whatever snapshot the fixture is currently at, as a table rebuilt
/// from a catalog would.
fn shared_table(fixture: &Fixture, registry: quarry::engine::SharedRegistry) -> QuarryTable {
    let mut table = QuarryTable::new(
        schema(),
        TableId("events".into()),
        fixture.at,
        fixture.graph.clone(),
        field_ids(),
    )
    .with_policy(POLICY)
    .with_shared_registry(registry)
    .on_object_store(ObjectStoreUrl::local_filesystem());
    for (file, size) in &fixture.sizes {
        table = table.with_parquet_file(file.clone(), *size);
    }
    table
}

/// An index the table has grown past is rebuilt, not left to decay.
#[tokio::test]
async fn a_stale_index_is_refreshed_when_the_table_grows() {
    let mut fixture = fixture("loop_refresh");
    let sql = "SELECT * FROM events WHERE tenant_id = 1";
    let registry = shared(Registry::new());

    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        // A tenth of the table is as far behind as an index may fall.
        mechanism_policy(Policy {
            max_residual_pct: 10.0,
            ..Policy::automatic(1 << 30).with_min_queries(3)
        }),
    )
    .for_reader(POLICY);

    // Build an index at snapshot 1.
    {
        let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
        let session = quarry().session();
        session
            .register("events", Arc::clone(&served))
            .expect("register");
        for _ in 0..5 {
            session.sql(sql).await.expect("query");
            let report = served.last_scan().expect("scan");
            optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
        }
        let round = optimizer.round(&session, &served).await;
        assert_eq!(round.built.len(), 1, "built at snapshot 1: {round:?}");
    }

    // The table grows substantially, with more rows for tenant 1 in the new
    // file. The old index cannot know about it.
    let appended = fixture.append(1, 2_000);
    assert_eq!(fixture.at, SnapshotId(2));

    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
    let session = quarry().session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    // Correct but decayed: the stale index still serves, and the appended file
    // is read alongside it every single query.
    let rows: usize = session
        .sql(sql)
        .await
        .expect("query")
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 2_500, "every tenant-1 row, old and new");
    let report = served.last_scan().expect("scan");
    assert!(
        report.also_scanned.contains(&appended),
        "the appended file must be scanned alongside a stale index"
    );
    optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));

    // The round rebuilds it at the new snapshot.
    let round = optimizer.round(&session, &served).await;
    assert_eq!(round.refreshed.len(), 1, "should refresh: {round:?}");
    assert!(round.built.is_empty(), "a refresh is not a new build");

    // Now the appended file is indexed, not merely tolerated.
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
    let session = quarry().session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");
    let rows: usize = session
        .sql(sql)
        .await
        .expect("query")
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 2_500, "the answer must not change");

    let report = served.last_scan().expect("scan");
    assert!(
        report.also_scanned.is_empty(),
        "nothing should be residual now: {report:?}"
    );
    assert!(
        report.files_read.contains(&appended),
        "the appended file is now reached through the index"
    );
    assert_eq!(
        report.files_read.len(),
        2,
        "only the two files holding tenant 1"
    );
}

#[tokio::test]
async fn a_small_append_does_not_trigger_a_rebuild() {
    let mut fixture = fixture("loop_no_refresh");
    let sql = "SELECT * FROM events WHERE tenant_id = 1";
    let registry = shared(Registry::new());

    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        mechanism_policy(Policy {
            max_residual_pct: 25.0,
            ..Policy::automatic(1 << 30).with_min_queries(3)
        }),
    )
    .for_reader(POLICY);

    {
        let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
        let session = quarry().session();
        session
            .register("events", Arc::clone(&served))
            .expect("register");
        for _ in 0..5 {
            session.sql(sql).await.expect("query");
            let report = served.last_scan().expect("scan");
            optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
        }
        assert_eq!(optimizer.round(&session, &served).await.built.len(), 1);
    }

    // A handful of rows against four files of five hundred: well inside the
    // threshold, and rebuilding would cost more than it recovers.
    fixture.append(9, 5);

    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
    let session = quarry().session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    let round = optimizer.round(&session, &served).await;
    assert!(
        round.refreshed.is_empty(),
        "a trivial append is not worth a rebuild: {round:?}"
    );
}

#[tokio::test]
async fn a_built_index_finds_every_value_in_the_data() {
    let fixture = fixture("loop_build_contents");
    let session = quarry().session();
    let bare = table(&fixture, Registry::new());

    let index = build_index(&session, &bare, TENANT_FIELD)
        .await
        .expect("build index");

    // The fixture puts one tenant per file, so each tenant maps to one file.
    assert_eq!(index.values(), 4, "four distinct tenants across four files");
    for (position, tenant) in [1i64, 2, 3, 4].into_iter().enumerate() {
        let hash = hash_scalar(&ScalarValue::Int64(Some(tenant)));
        assert_eq!(
            index.files_for(hash),
            Some(&std::collections::BTreeSet::from([
                fixture.files[position].clone()
            ])),
            "tenant {tenant} should be found in exactly its own file"
        );
    }
}

#[tokio::test]
async fn building_an_index_on_an_unknown_field_fails_clearly() {
    let fixture = fixture("loop_build_bad_field");
    let session = quarry().session();
    let bare = table(&fixture, Registry::new());

    let error = build_index(&session, &bare, 999)
        .await
        .expect_err("field 999 is not a column");
    assert!(
        error.to_string().contains("999"),
        "unhelpful error: {error}"
    );
}

#[tokio::test]
async fn the_loop_proposes_builds_and_then_stops_proposing() {
    let fixture = fixture("loop_full");
    let prices = PriceTable::default();
    let sql = "SELECT * FROM events WHERE tenant_id = 1";
    let mut workload = Workload::new();

    // 1. OBSERVE. Nothing is built, so every query scans the whole table.
    for _ in 0..10 {
        let rows = run_and_observe(&fixture, Registry::new(), &mut workload, sql).await;
        assert_eq!(rows, 500, "tenant 1 has 500 rows");
    }

    // 2. PROPOSE. The workload asks for an index on the field it keeps
    //    filtering, and nothing else.
    let proposals = workload.proposals(&prices, 5);
    assert_eq!(proposals.len(), 1, "one field, one proposal");
    assert_eq!(proposals[0].field, TENANT_FIELD);
    assert_eq!(proposals[0].queries, 10);
    assert!(
        proposals[0].bytes_scanned > 0,
        "the unaided queries read real bytes"
    );

    // 3. BUILD what was proposed.
    let registry = hand_written_index(&fixture, proposals[0].field, 1);

    // 4. MEASURE. The same queries now read less, for the same answer.
    let mut after = Workload::new();
    for _ in 0..10 {
        let rows = run_and_observe(
            &fixture,
            hand_written_index(&fixture, TENANT_FIELD, 1),
            &mut after,
            sql,
        )
        .await;
        assert_eq!(rows, 500, "the answer must not change");
    }

    let credited = after
        .credited(&DerivedId("tenant_idx".into()))
        .expect("the index was credited");
    assert_eq!(credited.queries, 10);
    assert!(
        credited.bytes_saved() > 0,
        "the index avoided reading three of the four files"
    );

    // 5. STOP PROPOSING. The shape is served, so it no longer asks.
    assert!(
        after.proposals(&prices, 1).is_empty(),
        "a shape already served needs no index"
    );

    // 6. KEEP. What it saved outweighs what keeping it costs.
    assert!(
        after.retirements(&registry, &prices, 30.0).is_empty(),
        "an index earning its keep must not be retired"
    );
}

#[tokio::test]
async fn an_index_nobody_uses_is_retired() {
    let fixture = fixture("loop_retire");
    let prices = PriceTable::default();

    // An index on a field no query filters on.
    let mut index = Index::new(99);
    index.insert(1, fixture.files[0].clone());
    let mut registry = Registry::new();
    registry.register(Derived::new(
        DerivedId("unused_idx".into()),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(1),
        },
        POLICY,
        1 << 30, // a gigabyte of index nobody probes
        Box::new(index.with_bytes(1 << 30)),
    ));

    let mut workload = Workload::new();
    for _ in 0..5 {
        run_and_observe(
            &fixture,
            Registry::new(),
            &mut workload,
            "SELECT * FROM events WHERE tenant_id = 1",
        )
        .await;
    }

    assert_eq!(
        workload.retirements(&registry, &prices, 30.0),
        vec![DerivedId("unused_idx".into())],
        "a gigabyte that has served nothing is indistinguishable from a leak"
    );
}

#[tokio::test]
async fn the_measured_saving_matches_the_files_avoided() {
    let fixture = fixture("loop_measure");
    let sql = "SELECT * FROM events WHERE tenant_id = 1";

    let mut unaided = Workload::new();
    run_and_observe(&fixture, Registry::new(), &mut unaided, sql).await;

    let mut aided = Workload::new();
    run_and_observe(
        &fixture,
        hand_written_index(&fixture, TENANT_FIELD, 1),
        &mut aided,
        sql,
    )
    .await;

    let credited = aided
        .credited(&DerivedId("tenant_idx".into()))
        .expect("credited");

    // The index leaves exactly one of four files to read, so the saving is
    // the other three — computed from metadata, not inferred from timings.
    let total: u64 = fixture.sizes.values().sum();
    let kept = fixture.sizes[&fixture.files[0]];
    assert_eq!(
        credited.bytes_saved(),
        total - kept,
        "the saving is the three files the index ruled out"
    );
    assert_eq!(
        credited.bytes_if_full_scan, total,
        "the baseline is the whole table, known from file sizes"
    );
}

/// The judgement the measurement said was missing.
///
/// `examples/measure.rs` found three regimes where the loop built an index
/// with equal enthusiasm and one of them saved 95% of a scan while two saved
/// nothing. The gate is what tells them apart, from evidence available before
/// anything is built.
#[tokio::test]
async fn the_gate_refuses_an_index_the_format_makes_redundant() {
    let fixture = fixture("loop_gate");
    let sql = "SELECT * FROM events WHERE tenant_id = 1";

    // This fixture puts one tenant in each file, so per-file ranges are
    // disjoint: exactly the CLUSTERED regime, where Parquet's own row-group
    // statistics already skip the files an index would skip.
    let disjoint: Vec<(f64, f64)> = (1..=4).map(|t| (t as f64, t as f64)).collect();
    let clustered = Spread::from_bounds(4, &disjoint, 500.0);
    assert!(
        clustered.index_advantage_pct() < 10.0,
        "disjoint ranges should show no advantage, got {}%",
        clustered.index_advantage_pct()
    );

    let registry = shared(Registry::new());
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        Policy::automatic(1 << 30).with_min_queries(3),
    )
    .for_reader(POLICY)
    .with_spreads(BTreeMap::from([(TENANT_FIELD, clustered)]));

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    for _ in 0..5 {
        session.sql(sql).await.expect("query");
        let report = served.last_scan().expect("scan");
        optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
    }

    // It still wants an index — the shape goes unaided — and refuses anyway.
    assert_eq!(optimizer.proposals().len(), 1, "the shape is unaided");

    let round = optimizer.round(&session, &served).await;
    assert!(round.built.is_empty(), "must not build: {round:?}");
    assert_eq!(
        round.declined.first().map(|(_, why)| *why),
        Some(Declined::NoAdvantage),
        "and it should say why: {round:?}"
    );
    assert_eq!(registry.read().expect("lock").len(), 0);
}

#[tokio::test]
async fn the_gate_allows_an_index_the_format_cannot_replace() {
    let fixture = fixture("loop_gate_allows");
    let sql = "SELECT * FROM events WHERE tenant_id = 1";

    // The SELECTIVE regime: every file's range spans the whole domain, so
    // ranges prune nothing, and a value occupies about one file.
    let overlapping: Vec<(f64, f64)> = (0..4).map(|_| (0.0, 5_000_000.0)).collect();
    let selective = Spread::from_bounds(4, &overlapping, 1.1);
    assert!(selective.index_advantage_pct() > 50.0);

    let registry = shared(Registry::new());
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        Policy::automatic(1 << 30).with_min_queries(3),
    )
    .for_reader(POLICY)
    .with_spreads(BTreeMap::from([(TENANT_FIELD, selective)]));

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    for _ in 0..5 {
        session.sql(sql).await.expect("query");
        let report = served.last_scan().expect("scan");
        optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
    }

    let round = optimizer.round(&session, &served).await;
    assert_eq!(round.built.len(), 1, "should build: {round:?}");
}

#[tokio::test]
async fn the_gate_refuses_when_it_knows_nothing() {
    // Building on no evidence is what measurement showed to be wrong, so the
    // default is to decline and say so rather than to hope.
    let fixture = fixture("loop_gate_blind");
    let sql = "SELECT * FROM events WHERE tenant_id = 1";

    let registry = shared(Registry::new());
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        Policy::automatic(1 << 30).with_min_queries(3),
    )
    .for_reader(POLICY);

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    for _ in 0..5 {
        session.sql(sql).await.expect("query");
        let report = served.last_scan().expect("scan");
        optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
    }

    let round = optimizer.round(&session, &served).await;
    assert!(round.built.is_empty());
    assert_eq!(
        round.declined.first().map(|(_, why)| *why),
        Some(Declined::NoEvidence)
    );
}

/// The gate deciding from evidence it gathered itself.
///
/// Everything above hands the optimizer a `Spread`. Here the overlap is
/// *measured* off the data, which is the input that removes the need for a
/// distinct-value count Iceberg does not record.
#[tokio::test]
async fn overlap_is_measured_from_two_files_not_the_whole_table() {
    // The fixture puts one tenant per file, so no two files share a value.
    let clustered = fixture("spread_clustered");
    let session = quarry().session();
    let table = table(&clustered, Registry::new());

    let overlap = estimate_overlap(&session, &table, TENANT_FIELD)
        .await
        .expect("estimate")
        .expect("four files, so there is an overlap to measure");
    assert_eq!(
        overlap, 0.0,
        "one tenant per file means no shared values at all"
    );

    // Combined with the fixture's disjoint ranges, that is the regime where
    // the file format already prunes and an index adds nothing.
    let disjoint: Vec<(f64, f64)> = (1..=4).map(|t| (t as f64, t as f64)).collect();
    let spread = Spread::from_overlap(4, &disjoint, overlap);
    assert!(
        spread.index_advantage_pct() < 10.0,
        "claimed {}%",
        spread.index_advantage_pct()
    );
}

#[tokio::test]
async fn a_column_every_file_shares_measures_as_fully_overlapping() {
    // Every file holds the same handful of values, which is the regime where
    // an index prunes nothing however selective it looks.
    let dir = scratch("spread_shared");
    let mut sizes = BTreeMap::new();
    let mut files = Vec::new();
    for index in 0..4 {
        // The same four tenants in every file.
        let path = dir.join(format!("{index}.parquet"));
        let file = std::fs::File::create(&path).expect("create");
        let mut writer =
            datafusion::parquet::arrow::ArrowWriter::try_new(file, schema(), None).expect("writer");
        let tenants: Int64Array = (0..200).map(|row| (row % 4) as i64).collect();
        let messages: StringArray = (0..200).map(|row| Some(format!("m{row}"))).collect();
        let batch = RecordBatch::try_new(schema(), vec![Arc::new(tenants), Arc::new(messages)])
            .expect("batch");
        writer.write(&batch).expect("write");
        writer.close().expect("close");

        let id = FileId(path.to_string_lossy().into_owned());
        sizes.insert(id.clone(), std::fs::metadata(&path).expect("stat").len());
        files.push(id);
    }

    let mut snapshot = Snapshot::root(SnapshotId(1));
    for file in &files {
        snapshot = snapshot.with_clean_file(file.clone());
    }
    let shared_fixture = Fixture {
        dir,
        sizes,
        graph: SnapshotGraph::new().with(snapshot),
        files,
        at: SnapshotId(1),
    };

    let session = quarry().session();
    let table = table(&shared_fixture, Registry::new());
    let overlap = estimate_overlap(&session, &table, TENANT_FIELD)
        .await
        .expect("estimate")
        .expect("four files");

    assert_eq!(overlap, 1.0, "every file holds every value");

    // Ranges overlap too, so this is the SCATTERED regime: an index would be
    // built, cost storage, and prune nothing.
    let everywhere: Vec<(f64, f64)> = (0..4).map(|_| (0.0, 3.0)).collect();
    let spread = Spread::from_overlap(4, &everywhere, overlap);
    assert_eq!(spread.index_advantage_pct(), 0.0);
}

#[tokio::test]
async fn a_single_file_table_has_no_overlap_to_measure() {
    let dir = scratch("spread_one_file");
    let (file, size) = write_parquet(&dir, "only.parquet", 1, 100);
    let fixture = Fixture {
        dir,
        sizes: BTreeMap::from([(file.clone(), size)]),
        graph: SnapshotGraph::new()
            .with(Snapshot::root(SnapshotId(1)).with_clean_file(file.clone())),
        files: vec![file],
        at: SnapshotId(1),
    };

    let session = quarry().session();
    let table = table(&fixture, Registry::new());
    assert!(
        estimate_overlap(&session, &table, TENANT_FIELD)
            .await
            .expect("estimate")
            .is_none(),
        "one file cannot overlap with anything, and guessing would be worse"
    );
}

/// Partial clustering: the case that broke the overlap estimator twice.
///
/// Each tenant lives in three consecutive files of eight. Sampling only
/// distant files reports every value as living in one file; sampling only
/// neighbours reports far too many. The estimate has to land near three.
#[tokio::test]
async fn overlap_is_estimated_correctly_under_partial_clustering() {
    let dir = scratch("spread_partial");
    let files = 8usize;
    let tenants = 64i64;

    let mut sizes = BTreeMap::new();
    let mut ids = Vec::new();
    for index in 0..files {
        // Tenants whose home is this file, the one before, or the one before
        // that — so every tenant occupies exactly three consecutive files.
        let mine: Vec<i64> = (0..tenants)
            .filter(|tenant| {
                let home = (*tenant as usize) % files;
                [home, (home + 1) % files, (home + 2) % files].contains(&index)
            })
            .collect();

        let path = dir.join(format!("{index}.parquet"));
        let file = std::fs::File::create(&path).expect("create");
        let mut writer =
            datafusion::parquet::arrow::ArrowWriter::try_new(file, schema(), None).expect("writer");
        let tenant_ids: Int64Array = (0..400).map(|row| mine[row % mine.len()]).collect();
        let messages: StringArray = (0..400).map(|row| Some(format!("m{row}"))).collect();
        let batch = RecordBatch::try_new(schema(), vec![Arc::new(tenant_ids), Arc::new(messages)])
            .expect("batch");
        writer.write(&batch).expect("write");
        writer.close().expect("close");

        let id = FileId(path.to_string_lossy().into_owned());
        sizes.insert(id.clone(), std::fs::metadata(&path).expect("stat").len());
        ids.push(id);
    }

    let mut snapshot = Snapshot::root(SnapshotId(1));
    for id in &ids {
        snapshot = snapshot.with_clean_file(id.clone());
    }
    let fixture = Fixture {
        dir,
        sizes,
        graph: SnapshotGraph::new().with(snapshot),
        files: ids,
        at: SnapshotId(1),
    };

    let session = quarry().session();
    let table = table(&fixture, Registry::new());

    // Ground truth, from the index itself.
    let index = build_index(&session, &table, TENANT_FIELD)
        .await
        .expect("build");
    let postings: u64 = index.postings().map(|(_, f)| f.len() as u64).sum();
    let truth = postings as f64 / index.values() as f64;
    assert!(
        (truth - 3.0).abs() < 0.2,
        "the fixture should put each tenant in three files, got {truth}"
    );

    // And what the gate estimates from a sample, without building anything.
    let overlap = estimate_overlap(&session, &table, TENANT_FIELD)
        .await
        .expect("estimate")
        .expect("eight files");
    let everywhere: Vec<(f64, f64)> = (0..files).map(|_| (0.0, tenants as f64)).collect();
    let spread = Spread::from_overlap(files as u64, &everywhere, overlap);

    assert!(
        (spread.files_by_index - truth).abs() < 1.5,
        "estimated {:.2} files per value against a true {truth:.2}; sampling \
         only distant files gives 1.0 and only neighbours gives far too many",
        spread.files_by_index
    );
    // Three of eight files is still well worth an index.
    assert!(spread.index_advantage_pct() > 40.0);
}

/// Ranking: the biggest ceiling is not the best candidate.
///
/// Nothing measured whether proposals were *ordered* well, only whether
/// individual decisions were right. They were not: `max_builds_per_round` was
/// applied while walking proposals in `ceiling_usd` order, and the ceiling was
/// measured 1775x wrong on one regime. With the default of one build per
/// round, a round could build the worst candidate and decline the best.
#[tokio::test]
async fn the_best_candidate_is_built_not_the_one_with_the_biggest_ceiling() {
    let fixture = fixture("loop_ranking");
    let registry = shared(Registry::new());
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));

    // Two fields. `tenant_id` is scanned twice as much, so it has the larger
    // ceiling — but the format nearly prunes it already. `message` is scanned
    // less and an index would genuinely help.
    //
    //     ceiling   tenant 1000 MB  >  message 500 MB
    //     expected  tenant  120 MB  <  message 475 MB
    let barely_helps = Spread {
        files: 20,
        files_by_bounds: 20.0,
        files_by_index: 17.6, // 12% advantage, just over the threshold
    };
    let helps_a_lot = Spread {
        files: 20,
        files_by_bounds: 20.0,
        files_by_index: 1.0, // 95% advantage
    };

    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        Policy::automatic(1 << 30)
            .with_min_queries(3)
            .with_max_builds(1),
    )
    .for_reader(POLICY)
    .with_spreads(BTreeMap::from([
        (TENANT_FIELD, barely_helps),
        (MESSAGE_FIELD, helps_a_lot),
    ]));

    // Ten queries on tenant_id, five on message, the same size each.
    for _ in 0..10 {
        optimizer.observe(observation_on(TENANT_FIELD, 100_000_000));
    }
    for _ in 0..5 {
        optimizer.observe(observation_on(MESSAGE_FIELD, 100_000_000));
    }

    let proposals = optimizer.proposals();
    assert_eq!(proposals.len(), 2);
    assert_eq!(
        proposals[0].field, TENANT_FIELD,
        "by ceiling, tenant_id looks like the one to build"
    );

    // But by expected saving it is not: 1000 MB x 12% against 500 MB x 95%.
    let prices = PriceTable::default();
    let tenant = proposals
        .iter()
        .find(|p| p.field == TENANT_FIELD)
        .expect("tenant proposal");
    let message = proposals
        .iter()
        .find(|p| p.field == MESSAGE_FIELD)
        .expect("message proposal");
    assert!(tenant.ceiling_usd > message.ceiling_usd);
    assert!(
        message.expected_usd(&helps_a_lot, &prices) > tenant.expected_usd(&barely_helps, &prices),
        "the cheaper query's index should be worth more"
    );

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    let round = optimizer.round(&session, &served).await;
    assert_eq!(round.built.len(), 1, "one build allowed: {round:?}");
    assert_eq!(
        round.built[0],
        index_id(&TableId("events".into()), MESSAGE_FIELD),
        "the round must spend its one build on the better candidate: {round:?}"
    );
    assert_eq!(
        round
            .declined
            .iter()
            .find(|(id, _)| *id == index_id(&TableId("events".into()), TENANT_FIELD))
            .map(|(_, why)| *why),
        Some(Declined::RoundFull),
        "and defer the other rather than skip it: {round:?}"
    );
}

/// An observation of a query filtering `field`, having read `bytes`.
fn observation_on(field: u32, bytes: u64) -> quarry::workload::Observation {
    use quarry::derived::{Predicate, Query};
    use quarry::workload::{Fingerprint, Observation};

    let query = Query {
        table: TableId("events".into()),
        snapshot: SnapshotId(1),
        policy: POLICY,
        plan_hash: field as u64,
        plan: None,
        projected: std::collections::BTreeSet::from([field]),
        predicates: vec![Predicate::Eq { field, value: 1 }],
    };
    Observation {
        fingerprint: Fingerprint::of(&query),
        bytes_read: bytes,
        bytes_if_full_scan: bytes,
        used: None,
    }
}
