//! Derived state surviving a restart.
//!
//! Everything the optimizer learned and built lived in memory, so a restart
//! lost it — and then, with credits gone, retirement would delete the indexes
//! whose bytes were still in storage. These tests simulate a restart by
//! throwing away every in-memory structure and recovering from storage alone.
//!
//! ```sh
//! cargo test --features engine,iceberg --test persistence
//! ```

#![cfg(all(feature = "engine", feature = "iceberg"))]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::scalar::ScalarValue;
use iceberg::io::FileIO;
use object_store::local::LocalFileSystem;
use url::Url;

use quarry::derived::PolicyFingerprint;
use quarry::engine::{
    Layout, Quarry, QuarryTable, Store, build_index, hash_scalar, index_id, read_index, shared,
    write_index,
};
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

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn write_parquet(dir: &Path, name: &str, tenant: i64, rows: usize) -> (FileId, u64) {
    let path = dir.join(name);
    let file = fs::File::create(&path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema(), None).expect("writer");

    let tenants: Int64Array = (0..rows).map(|_| tenant).collect();
    let messages: StringArray = (0..rows).map(|i| Some(format!("row {i}"))).collect();
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
}

fn fixture(name: &str) -> Fixture {
    let dir = scratch(name);
    let mut sizes = BTreeMap::new();
    let mut files = Vec::new();

    for (index, tenant) in [1i64, 2, 3].into_iter().enumerate() {
        let (file, size) = write_parquet(&dir, &format!("{index}.parquet"), tenant, 200);
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

/// Both IO stacks, as a deployment would have.
fn stacks(fixture: &Fixture) -> (Arc<FileIO>, Store) {
    let file_io = Arc::new(
        FileIO::from_path(fixture.dir.to_str().expect("utf-8"))
            .expect("file io")
            .build()
            .expect("build file io"),
    );
    let store = Store::new(
        Arc::clone(&file_io),
        Arc::new(LocalFileSystem::new()),
        Layout::beside(fixture.dir.to_str().expect("utf-8")),
    );
    (file_io, store)
}

fn events() -> TableId {
    TableId("events".into())
}

#[tokio::test]
async fn an_index_survives_being_written_and_read_back() {
    let fixture = fixture("persist_roundtrip");
    let (file_io, store) = stacks(&fixture);

    let session = quarry().session();
    let built = build_index(&session, &table(&fixture, Registry::new()), TENANT_FIELD)
        .await
        .expect("build");
    assert_eq!(built.values(), 3, "three tenants, one per file");

    let path = store
        .write(&events(), SnapshotId(1), POLICY, &built)
        .await
        .expect("write");
    assert!(
        Path::new(&path).exists(),
        "the blob should be on disk at {path}"
    );

    let (read, policy) = read_index(&file_io, &path, TENANT_FIELD)
        .await
        .expect("read")
        .expect("a usable index");

    assert_eq!(policy, POLICY);
    assert_eq!(read.field(), TENANT_FIELD);
    assert_eq!(read.values(), built.values());
    for tenant in [1i64, 2, 3] {
        let hash = hash_scalar(&ScalarValue::Int64(Some(tenant)));
        assert_eq!(
            read.files_for(hash),
            built.files_for(hash),
            "tenant {tenant} should map to the same files"
        );
    }
}

#[tokio::test]
async fn the_registry_is_rebuilt_by_listing_after_a_restart() {
    let fixture = fixture("persist_recover");
    let (_, store) = stacks(&fixture);
    let sql = "SELECT * FROM events WHERE tenant_id = 1";

    // --- Before the restart: build an index and write it down.
    {
        let session = quarry().session();
        let built = build_index(&session, &table(&fixture, Registry::new()), TENANT_FIELD)
            .await
            .expect("build");
        store
            .write(&events(), SnapshotId(1), POLICY, &built)
            .await
            .expect("write");
    }

    // --- The restart. Nothing in memory survives; storage is all there is.
    let recovered = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("recover");

    assert_eq!(recovered.registry.len(), 1, "the index should be found");
    assert!(recovered.discarded.is_empty());
    let id = index_id(&events(), TENANT_FIELD);
    assert_eq!(
        recovered.fields.get(&id),
        Some(&TENANT_FIELD),
        "the optimizer needs to know what to refresh"
    );

    let derived = recovered.registry.get(&id).expect("registered");
    assert_eq!(
        derived.source.snapshot,
        SnapshotId(1),
        "the lineage anchor must come back intact, or the rule cannot judge it"
    );
    assert_eq!(derived.policy, POLICY);

    // --- And it prunes, exactly as it did before the restart.
    let served = Arc::new(table(&fixture, recovered.registry));
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
    assert_eq!(rows, 200);

    let report = served.last_scan().expect("scan");
    assert_eq!(report.used.as_deref(), Some(id.0.as_str()));
    assert_eq!(
        report.files_read,
        std::collections::BTreeSet::from([fixture.files[0].clone()]),
        "a recovered index must prune as well as a fresh one"
    );
}

#[tokio::test]
async fn an_index_from_a_forgotten_snapshot_is_discarded() {
    let fixture = fixture("persist_expired");
    let (_, store) = stacks(&fixture);

    let session = quarry().session();
    let built = build_index(&session, &table(&fixture, Registry::new()), TENANT_FIELD)
        .await
        .expect("build");
    store
        .write(&events(), SnapshotId(1), POLICY, &built)
        .await
        .expect("write");

    // Snapshot 1 has been expired: the rule could never admit this index, so
    // recovery should not carry it, and the bytes should be reclaimed.
    let recovered = store
        .recover(&events(), &[SnapshotId(7)])
        .await
        .expect("recover");

    assert_eq!(recovered.registry.len(), 0);
    assert_eq!(recovered.discarded.len(), 1);

    assert_eq!(
        store.discard(&recovered.discarded).await.expect("discard"),
        1
    );
    let gone = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("recover");
    assert_eq!(gone.registry.len(), 0, "the object should be deleted");
}

#[tokio::test]
async fn a_table_that_was_never_optimized_recovers_to_nothing() {
    let fixture = fixture("persist_empty");
    let (_, store) = stacks(&fixture);

    let recovered = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("a missing prefix is not an error");
    assert_eq!(recovered.registry.len(), 0);
    assert!(recovered.discarded.is_empty());
}

#[tokio::test]
async fn a_corrupt_blob_is_discarded_rather_than_trusted() {
    let fixture = fixture("persist_corrupt");
    let (_, store) = stacks(&fixture);

    let session = quarry().session();
    let built = build_index(&session, &table(&fixture, Registry::new()), TENANT_FIELD)
        .await
        .expect("build");
    let path = store
        .write(&events(), SnapshotId(1), POLICY, &built)
        .await
        .expect("write");

    // Truncating a Puffin file destroys its footer. A partially readable
    // index is the dangerous case: it would prune away files holding matching
    // rows, so it must be refused rather than salvaged.
    let bytes = fs::read(&path).expect("read");
    fs::write(&path, &bytes[..bytes.len() / 2]).expect("truncate");

    // Before the footer guard existed this panicked inside iceberg-rust, so
    // it is asserted precisely: recovery must succeed, register nothing, and
    // offer the object for deletion.
    let recovered = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("a corrupt object must not fail recovery, let alone panic");

    assert_eq!(
        recovered.registry.len(),
        0,
        "a corrupt index must not be registered"
    );
    assert_eq!(
        recovered.discarded.len(),
        1,
        "and it should be offered for deletion"
    );
    assert_eq!(
        store.discard(&recovered.discarded).await.expect("discard"),
        1
    );
}

#[tokio::test]
async fn a_file_that_is_not_puffin_at_all_is_ignored() {
    let fixture = fixture("persist_not_puffin");
    let (_, store) = stacks(&fixture);
    let layout = Layout::beside(fixture.dir.to_str().expect("utf-8"));

    // Something else entirely, at a path that parses as ours.
    let path = layout.index_path(&events(), TENANT_FIELD, SnapshotId(1));
    fs::create_dir_all(Path::new(&path).parent().expect("parent")).expect("dirs");
    fs::write(&path, b"not a puffin file").expect("write");

    let recovered = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("should not fail");
    assert_eq!(recovered.registry.len(), 0);
    assert_eq!(recovered.discarded, vec![path]);
}

#[tokio::test]
async fn an_empty_file_does_not_panic() {
    // Zero bytes is the degenerate truncation, and the one most likely to
    // exist after an interrupted write.
    let fixture = fixture("persist_empty_file");
    let (_, store) = stacks(&fixture);
    let layout = Layout::beside(fixture.dir.to_str().expect("utf-8"));

    let path = layout.index_path(&events(), TENANT_FIELD, SnapshotId(1));
    fs::create_dir_all(Path::new(&path).parent().expect("parent")).expect("dirs");
    fs::write(&path, b"").expect("write");

    let recovered = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("should not fail");
    assert_eq!(recovered.registry.len(), 0);
    assert_eq!(recovered.discarded.len(), 1);
}

#[tokio::test]
async fn a_stale_hash_version_is_not_probed() {
    // The scenario HASH_VERSION exists for: postings built by a different
    // hashing of values. Matching nothing is silent, so it must be refused.
    let fixture = fixture("persist_hash_version");
    let (file_io, _) = stacks(&fixture);
    let layout = Layout::beside(fixture.dir.to_str().expect("utf-8"));

    let session = quarry().session();
    let built = build_index(&session, &table(&fixture, Registry::new()), TENANT_FIELD)
        .await
        .expect("build");
    let path = write_index(&file_io, &layout, &events(), SnapshotId(1), POLICY, &built)
        .await
        .expect("write");

    // Written with the current version, so it reads back.
    assert!(
        read_index(&file_io, &path, TENANT_FIELD)
            .await
            .expect("read")
            .is_some(),
        "the current version must be usable"
    );

    // Rewrite the recorded version so it no longer matches, byte for byte in
    // the footer's JSON.
    let raw = fs::read(&path).expect("read");
    let needle = b"\"quarry.hash-version\":\"1\"";
    let replacement = b"\"quarry.hash-version\":\"9\"";
    let position = raw
        .windows(needle.len())
        .position(|window| window == needle);

    if let Some(position) = position {
        let mut patched = raw.clone();
        patched[position..position + needle.len()].copy_from_slice(replacement);
        fs::write(&path, &patched).expect("patch");

        assert!(
            read_index(&file_io, &path, TENANT_FIELD)
                .await
                .expect("read")
                .is_none(),
            "an index built with a different hash version must be refused"
        );
    }
}

#[tokio::test]
async fn recovery_composes_with_a_shared_registry() {
    // The recovered registry is the one the live table consults, so a
    // restarted process is indistinguishable from one that never stopped.
    let fixture = fixture("persist_shared");
    let (_, store) = stacks(&fixture);

    {
        let session = quarry().session();
        let built = build_index(&session, &table(&fixture, Registry::new()), TENANT_FIELD)
            .await
            .expect("build");
        store
            .write(&events(), SnapshotId(1), POLICY, &built)
            .await
            .expect("write");
    }

    let recovered = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("recover");
    let registry = shared(recovered.registry);

    let mut served = QuarryTable::new(
        schema(),
        events(),
        SnapshotId(1),
        fixture.graph.clone(),
        field_ids(),
    )
    .with_policy(POLICY)
    .with_shared_registry(Arc::clone(&registry))
    .on_object_store(ObjectStoreUrl::local_filesystem());
    for (file, size) in &fixture.sizes {
        served = served.with_parquet_file(file.clone(), *size);
    }
    let served = Arc::new(served);

    let session = quarry().session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");
    session
        .sql("SELECT * FROM events WHERE tenant_id = 2")
        .await
        .expect("query");

    let report = served.last_scan().expect("scan");
    assert_eq!(
        report.files_read,
        std::collections::BTreeSet::from([fixture.files[1].clone()]),
        "tenant 2 lives only in the second file"
    );
    assert_eq!(registry.read().expect("lock").len(), 1);
}
