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
use quarry::engine::{Quarry, QuarryTable, hash_scalar};
use quarry::kinds::Index;
use quarry::registry::Registry;
use quarry::snapshot::{FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};
use quarry::workload::Workload;

const TENANT_FIELD: u32 = 4;
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
    sizes: BTreeMap<FileId, u64>,
    graph: SnapshotGraph,
    files: Vec<FileId>,
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
        sizes,
        graph: SnapshotGraph::new().with(snapshot),
        files,
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

/// The index the loop proposed, built for real.
fn build_index(fixture: &Fixture, field: u32, tenant: i64) -> Registry {
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
    let registry = build_index(&fixture, proposals[0].field, 1);

    // 4. MEASURE. The same queries now read less, for the same answer.
    let mut after = Workload::new();
    for _ in 0..10 {
        let rows = run_and_observe(
            &fixture,
            build_index(&fixture, TENANT_FIELD, 1),
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
        build_index(&fixture, TENANT_FIELD, 1),
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
