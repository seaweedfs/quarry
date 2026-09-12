//! Real Parquet objects, read from an object store, selected by the rule.
//!
//! The in-memory tests prove the *decisions*. These prove the decisions reach
//! actual bytes on actual storage: files the rule excludes are never opened,
//! which is checkable here because opening a file that does not exist fails.
//!
//! ```sh
//! cargo test --features engine --test engine_parquet
//! ```

#![cfg(feature = "engine")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;

use quarry::derived::{Derived, DerivedId, PolicyFingerprint, Source};
use quarry::engine::{QuarryTable, hash_scalar};
use quarry::kinds::Index;
use quarry::registry::Registry;
use quarry::snapshot::{FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};

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

fn batch(rows: &[(i64, &str)]) -> RecordBatch {
    let tenants: Int64Array = rows.iter().map(|(t, _)| *t).collect();
    let messages: StringArray = rows.iter().map(|(_, m)| Some(*m)).collect();
    RecordBatch::try_new(schema(), vec![Arc::new(tenants), Arc::new(messages)])
        .expect("batch matches schema")
}

/// A directory of its own per test, so runs cannot interfere.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Write one Parquet file and return its object path and byte length.
fn write_parquet(dir: &Path, name: &str, rows: &[(i64, &str)]) -> (FileId, u64) {
    let path = dir.join(name);
    let file = fs::File::create(&path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema(), None).expect("writer");
    writer.write(&batch(rows)).expect("write batch");
    writer.close().expect("close writer");

    let size = fs::metadata(&path).expect("stat").len();
    // The object path is absolute, matching how a catalog records data files.
    (FileId(path.to_string_lossy().into_owned()), size)
}

fn tenant_hash(tenant: i64) -> u64 {
    hash_scalar(&ScalarValue::Int64(Some(tenant)))
}

async fn run(table: Arc<QuarryTable>, sql: &str) -> Vec<RecordBatch> {
    let ctx = SessionContext::new();
    ctx.register_table("events", table).expect("register");
    ctx.sql(sql)
        .await
        .expect("plan")
        .collect()
        .await
        .expect("execute")
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// Three Parquet files on the local filesystem: tenant 1 in a and c,
/// tenant 2 in b.
fn parquet_table(dir: &Path, snapshot: i64) -> (QuarryTable, [FileId; 3]) {
    let (a, a_size) = write_parquet(dir, "a.parquet", &[(1, "a1"), (1, "a2")]);
    let (b, b_size) = write_parquet(dir, "b.parquet", &[(2, "b1")]);
    let (c, c_size) = write_parquet(dir, "c.parquet", &[(1, "c1")]);

    let graph = SnapshotGraph::new().with(
        Snapshot::root(SnapshotId(snapshot))
            .with_clean_file(a.clone())
            .with_clean_file(b.clone())
            .with_clean_file(c.clone()),
    );

    let table = QuarryTable::new(
        schema(),
        TableId("events".into()),
        SnapshotId(snapshot),
        graph,
        field_ids(),
    )
    .with_policy(POLICY)
    .on_object_store(ObjectStoreUrl::local_filesystem())
    .with_parquet_file(a.clone(), a_size)
    .with_parquet_file(b.clone(), b_size)
    .with_parquet_file(c.clone(), c_size);

    (table, [a, b, c])
}

fn tenant_index(at: i64, entries: &[(i64, &FileId)]) -> Derived {
    let mut index = Index::new(TENANT_FIELD);
    for (tenant, file) in entries {
        index.insert(tenant_hash(*tenant), (*file).clone());
    }
    Derived::new(
        DerivedId("tenant_idx".into()),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(at),
        },
        POLICY,
        4096,
        Box::new(index.with_bytes(4096)),
    )
}

#[tokio::test]
async fn parquet_is_read_from_the_object_store() {
    let dir = scratch("parquet_full_scan");
    let (table, _) = parquet_table(&dir, 810);
    let table = Arc::new(table);

    let rows = run(Arc::clone(&table), "SELECT * FROM events ORDER BY message").await;
    assert_eq!(total_rows(&rows), 4, "every row across the three files");
    assert_eq!(table.last_scan().expect("scan").files_read.len(), 3);
}

#[tokio::test]
async fn an_index_prunes_which_parquet_objects_are_opened() {
    let dir = scratch("parquet_pruned");
    let (table, [a, b, c]) = parquet_table(&dir, 810);

    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, &a), (1, &c), (2, &b)]));

    let table = Arc::new(table.with_registry(registry));

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(total_rows(&rows), 3, "same answer as a full scan");

    let report = table.last_scan().expect("scan");
    assert_eq!(report.used.as_deref(), Some("tenant_idx"));
    assert_eq!(report.files_read, BTreeSet::from([a, c]));
}

#[tokio::test]
async fn a_pruned_object_is_never_opened() {
    // The strongest available proof that pruning reaches the storage layer:
    // delete the file the index excludes. If the plan touched it, the query
    // would fail rather than merely be slower.
    let dir = scratch("parquet_deleted_file");
    let (table, [a, b, c]) = parquet_table(&dir, 810);

    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, &a), (1, &c), (2, &b)]));

    let table = Arc::new(table.with_registry(registry));

    fs::remove_file(&b.0).expect("remove the excluded file");

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(
        total_rows(&rows),
        3,
        "the query must not depend on the excluded object existing"
    );
}

#[tokio::test]
async fn projection_reaches_the_parquet_reader() {
    let dir = scratch("parquet_projection");
    let (table, _) = parquet_table(&dir, 810);
    let table = Arc::new(table);

    let rows = run(Arc::clone(&table), "SELECT tenant_id FROM events").await;
    assert_eq!(rows[0].num_columns(), 1, "only the projected column");
    assert_eq!(total_rows(&rows), 4);
}

#[tokio::test]
async fn aggregation_over_pruned_parquet_is_correct() {
    let dir = scratch("parquet_aggregate");
    let (table, [a, b, c]) = parquet_table(&dir, 810);

    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, &a), (1, &c), (2, &b)]));
    let table = Arc::new(table.with_registry(registry));

    let rows = run(
        Arc::clone(&table),
        "SELECT count(*) AS n FROM events WHERE tenant_id = 1",
    )
    .await;

    let n = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is int64")
        .value(0);
    assert_eq!(n, 3);
    assert_eq!(table.last_scan().expect("scan").files_read.len(), 2);
}

#[tokio::test]
async fn a_stale_index_reads_the_parquet_added_since() {
    let dir = scratch("parquet_stale");
    let (a, a_size) = write_parquet(&dir, "a.parquet", &[(1, "a1"), (1, "a2")]);
    let (b, b_size) = write_parquet(&dir, "b.parquet", &[(2, "b1")]);
    let (c, c_size) = write_parquet(&dir, "c.parquet", &[(1, "c1")]);

    // The index was built at 810, before c existed.
    let graph = SnapshotGraph::new()
        .with(
            Snapshot::root(SnapshotId(810))
                .with_clean_file(a.clone())
                .with_clean_file(b.clone()),
        )
        .with(
            Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                .with_clean_file(a.clone())
                .with_clean_file(b.clone())
                .with_clean_file(c.clone()),
        );

    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, &a), (2, &b)]));

    let table = Arc::new(
        QuarryTable::new(
            schema(),
            TableId("events".into()),
            SnapshotId(811),
            graph,
            field_ids(),
        )
        .with_policy(POLICY)
        .with_registry(registry)
        .on_object_store(ObjectStoreUrl::local_filesystem())
        .with_parquet_file(a.clone(), a_size)
        .with_parquet_file(b, b_size)
        .with_parquet_file(c.clone(), c_size),
    );

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(
        total_rows(&rows),
        3,
        "the row in the unindexed object must not be lost"
    );

    let report = table.last_scan().expect("scan");
    assert_eq!(report.also_scanned, BTreeSet::from([c.clone()]));
    assert_eq!(report.files_read, BTreeSet::from([a, c]));
}
