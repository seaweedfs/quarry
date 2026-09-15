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

use std::collections::{BTreeMap, BTreeSet, HashMap};
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

use iceberg::puffin::{Blob, CompressionCodec, PuffinWriter};
use quarry::derived::PolicyFingerprint;
use quarry::derived::{Decision, Plan, Predicate, Query, Rewrite, Scope};
use quarry::engine::{
    Advertised, Layout, Optimizer, Quarry, QuarryTable, Retired, Store, advertised, build_bitmap,
    build_filter_set, build_index, filter_set_id, hash_scalar, index_id, read_bitmap,
    read_filter_set, read_index, shared, write_index,
};
use quarry::registry::Registry;
use quarry::snapshot::{FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};
use quarry::workload::{Policy, Workload};

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

/// A data file with one row group per entry — the shape a bitmap posts.
fn write_parquet_grouped(dir: &Path, name: &str, groups: &[Vec<i64>]) -> (FileId, u64) {
    let path = dir.join(name);
    let file = fs::File::create(&path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema(), None).expect("writer");

    for tenants in groups {
        let tenants_col: Int64Array = tenants.iter().map(|t| Some(*t)).collect();
        let messages: StringArray = tenants
            .iter()
            .enumerate()
            .map(|(i, _)| Some(format!("row {i}")))
            .collect();
        let batch = RecordBatch::try_new(schema(), vec![Arc::new(tenants_col), Arc::new(messages)])
            .expect("batch");
        writer.write(&batch).expect("write");
        writer.flush().expect("close row group");
    }
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
    assert_eq!(report.used.first().map(String::as_str), Some(id.0.as_str()));
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

/// The whole restart: indexes recovered, credits recovered, nothing deleted.
#[tokio::test]
async fn a_restart_recovers_everything_and_destroys_nothing() {
    let fixture = fixture("persist_restart");
    let (_, store) = stacks(&fixture);
    let sql = "SELECT * FROM events WHERE tenant_id = 1";
    let policy = Policy {
        retire_after_queries: 3,
        // This test is about surviving a restart, not about whether the index
        // is worth having. The table is built by hand so it carries no
        // per-file bounds, and the gate correctly declines to judge without
        // them — which would leave nothing to restart with.
        min_index_advantage_pct: 0.0,
        ..Policy::automatic(1 << 30).with_min_queries(3)
    };

    // --- Process one: learn, build, and write everything down.
    let saved_credits = {
        let registry = shared(Registry::new());
        let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
        let mut optimizer = Optimizer::new(Arc::clone(&registry), policy).for_reader(POLICY);

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
        assert_eq!(round.built.len(), 1, "built an index: {round:?}");

        // Queries now served by it, so it accrues credit.
        for _ in 0..5 {
            session.sql(sql).await.expect("query");
            let report = served.last_scan().expect("scan");
            assert!(!report.used.is_empty(), "the index should be serving");
            optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
        }

        let id = index_id(&events(), TENANT_FIELD);
        let credits = optimizer
            .workload()
            .credited(&id)
            .expect("the index earned credit");
        assert!(credits.bytes_saved() > 0);

        // Write the index and the telemetry down.
        let derived_bytes = registry.read().expect("lock").bytes();
        assert!(derived_bytes > 0);
        let index = build_index(&session, &served, TENANT_FIELD)
            .await
            .expect("rebuild for writing");
        store
            .write(&events(), SnapshotId(1), POLICY, &index)
            .await
            .expect("write index");
        store
            .save_workload(&events(), optimizer.workload())
            .await
            .expect("save workload");

        credits
    };

    // --- Process two. Nothing in memory carried over.
    let recovered = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("recover");
    assert_eq!(recovered.registry.len(), 1, "the index came back");

    let workload = store
        .load_workload(&events())
        .await
        .expect("load")
        .expect("a saved workload");

    let id = index_id(&events(), TENANT_FIELD);
    assert_eq!(
        workload.credited(&id).map(|seen| seen.bytes_saved()),
        Some(saved_credits.bytes_saved()),
        "credits must survive, or the first round deletes what was recovered"
    );

    let registry = shared(recovered.registry);
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));
    let mut optimizer = Optimizer::new(Arc::clone(&registry), policy)
        .for_reader(POLICY)
        .with_workload(workload);
    optimizer.adopt(recovered.fields);

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    // The first round after a restart must not destroy what it recovered.
    let round = optimizer.round(&session, &served).await;
    assert!(
        round.retired.is_empty(),
        "a recovered index that has paid for itself must be kept: {round:?}"
    );
    assert!(
        round.built.is_empty(),
        "and it must not be built a second time: {round:?}"
    );
    assert_eq!(registry.read().expect("lock").len(), 1);

    // And it is genuinely in use, not merely present.
    session.sql(sql).await.expect("query");
    let report = served.last_scan().expect("scan");
    assert_eq!(report.used.first().map(String::as_str), Some(id.0.as_str()));
}

#[tokio::test]
async fn a_cold_start_without_credits_still_does_not_delete() {
    // The destructive case: an index recovered with no telemetry looks as
    // though it has saved nothing, and retirement reads that as a reason to
    // delete. The grace window is what stops it.
    let fixture = fixture("persist_cold_start");
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
    let served = Arc::new(shared_table(&fixture, Arc::clone(&registry)));

    // No workload at all, as though the telemetry were lost.
    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        Policy {
            retire_after_queries: 100,
            ..Policy::automatic(1 << 30)
        },
    );
    optimizer.adopt(recovered.fields);

    let session = quarry().session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    let round = optimizer.round(&session, &served).await;
    assert!(
        round.retired.is_empty(),
        "nothing may be retired on no evidence: {round:?}"
    );
    assert_eq!(registry.read().expect("lock").len(), 1);
}

#[tokio::test]
async fn a_saved_workload_survives_a_round_trip() {
    let fixture = fixture("persist_workload");
    let (_, store) = stacks(&fixture);

    let mut workload = Workload::new();
    let served = Arc::new(table(&fixture, Registry::new()));
    let session = quarry().session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    session
        .sql("SELECT tenant_id FROM events WHERE tenant_id = 3")
        .await
        .expect("query");
    let report = served.last_scan().expect("scan");
    workload.observe(report.observation(report.bytes_planned(&fixture.sizes)));

    store
        .save_workload(&events(), &workload)
        .await
        .expect("save");
    let read = store
        .load_workload(&events())
        .await
        .expect("load")
        .expect("present");

    assert_eq!(read.shapes(), workload.shapes());
    assert_eq!(read.observed(), workload.observed());
    let (shape, seen) = workload.shapes_seen().next().expect("one shape");
    assert_eq!(
        read.seen(shape).map(|s| s.bytes_if_full_scan),
        Some(seen.bytes_if_full_scan),
        "the unaided baseline must survive, since savings are measured from it"
    );
}

#[tokio::test]
async fn no_saved_workload_is_not_an_error() {
    let fixture = fixture("persist_no_workload");
    let (_, store) = stacks(&fixture);
    assert!(
        store
            .load_workload(&events())
            .await
            .expect("should not fail")
            .is_none()
    );
}

/// A table sharing `registry`, for the restart tests.
fn shared_table(fixture: &Fixture, registry: quarry::engine::SharedRegistry) -> QuarryTable {
    let mut table = QuarryTable::new(
        schema(),
        events(),
        SnapshotId(1),
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

/// A retired index whose blob survives is *re-registered* by recovery: its
/// snapshot is still known, so nothing in `recover` knows it was dropped.
/// Reclaiming the bytes is therefore not housekeeping — it is what makes
/// retirement stick across a restart.
#[tokio::test]
async fn a_retired_index_stays_retired_only_if_reclaimed() {
    let fixture = fixture("persist_reclaim");
    let (_, store) = stacks(&fixture);

    let index = {
        let session = quarry().session();
        build_index(&session, &table(&fixture, Registry::new()), TENANT_FIELD)
            .await
            .expect("build")
    };
    store
        .write(&events(), SnapshotId(1), POLICY, &index)
        .await
        .expect("write");

    // Retired the way a round reports it: enough to find the blob.
    let retired = [Retired {
        id: index_id(&events(), TENANT_FIELD),
        field: Some(TENANT_FIELD),
        at: SnapshotId(1),
    }];

    // Without reclamation the restart undoes the retirement.
    let resurrected = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("recover without reclaim");
    assert_eq!(
        resurrected.registry.len(),
        1,
        "recovery cannot tell retired from live"
    );
    store
        .discard(&resurrected.discarded)
        .await
        .expect("discard");
    // The blob was never discarded (the snapshot is known), so delete it now
    // the way a caller should have.
    assert_eq!(
        store.reclaim(&events(), &retired).await.expect("reclaim"),
        1,
        "one blob to delete"
    );

    // With reclamation, the restart leaves the retirement in place.
    let gone = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("recover after reclaim");
    assert_eq!(gone.registry.len(), 0, "the retirement should stick");
}

#[tokio::test]
async fn a_published_manifest_recovers_without_listing() {
    let fixture = fixture("persist_publish");
    let (_, store) = stacks(&fixture);

    let session = quarry().session();
    let built = build_index(&session, &table(&fixture, Registry::new()), TENANT_FIELD)
        .await
        .expect("build");
    let path = store
        .write(&events(), SnapshotId(1), POLICY, &built)
        .await
        .expect("write");

    // The manifest advertises the index by pointing at its path; the kind is
    // the blob's own type, so a reader sees exactly what exists.
    let stats = store
        .manifest(
            &events(),
            SnapshotId(1),
            &[Advertised {
                kind: quarry::engine::QUARRY_EQ_INDEX_V2.to_owned(),
                fields: vec![TENANT_FIELD as i32],
                path: path.clone(),
                properties: HashMap::new(),
            }],
            None,
        )
        .await
        .expect("manifest");

    let entries = advertised(&stats);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, path);
    assert_eq!(entries[0].fields, vec![TENANT_FIELD as i32]);

    // Recovery through the published copy: no listing, just the paths the
    // catalog's metadata names.
    let recovered = store
        .recover_published(&events(), &stats, &[SnapshotId(1)])
        .await
        .expect("recover");
    assert_eq!(recovered.registry.len(), 1, "the index should be found");
    assert!(recovered.discarded.is_empty());
}

#[tokio::test]
async fn publishing_carries_forward_statistics_already_there() {
    let fixture = fixture("persist_merge");
    let (file_io, store) = stacks(&fixture);

    // Another engine already published statistics for the snapshot.
    let prior_path = fixture.dir.join("their-stats.puffin");
    let output = file_io
        .new_output(prior_path.to_str().expect("utf-8"))
        .expect("output");
    let mut writer = PuffinWriter::new(&output, HashMap::new(), false)
        .await
        .expect("writer");
    writer
        .add(
            Blob::builder()
                .r#type("ndv".to_owned())
                .fields(vec![1])
                .snapshot_id(1)
                .sequence_number(0)
                .data(vec![1, 2, 3])
                .properties(HashMap::new())
                .build(),
            CompressionCodec::None,
        )
        .await
        .expect("add");
    writer.close().await.expect("close");
    let prior = iceberg::spec::StatisticsFile {
        snapshot_id: 1,
        statistics_path: prior_path.to_string_lossy().into_owned(),
        file_size_in_bytes: 0,
        file_footer_size_in_bytes: 0,
        key_metadata: None,
        blob_metadata: vec![],
    };

    let stats = store
        .manifest(
            &events(),
            SnapshotId(1),
            &[Advertised {
                kind: quarry::engine::QUARRY_EQ_INDEX_V2.to_owned(),
                fields: vec![TENANT_FIELD as i32],
                path: "anywhere.puffin".to_owned(),
                properties: HashMap::new(),
            }],
            Some(&prior),
        )
        .await
        .expect("manifest");

    let types: Vec<_> = stats
        .blob_metadata
        .iter()
        .map(|blob| blob.r#type.as_str())
        .collect();
    assert_eq!(
        types,
        vec!["ndv", quarry::engine::QUARRY_EQ_INDEX_V2],
        "the prior engine's blob must be carried forward, not replaced"
    );

    // Ours advertise a path; theirs does not, so it is never mistaken for
    // derived state.
    let entries = advertised(&stats);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "anywhere.puffin");
}

#[tokio::test]
async fn a_manifest_for_a_forgotten_snapshot_is_discarded() {
    let fixture = fixture("persist_publish_stale");
    let (_, store) = stacks(&fixture);

    let stats = store
        .manifest(&events(), SnapshotId(7), &[], None)
        .await
        .expect("manifest");

    let recovered = store
        .recover_published(&events(), &stats, &[SnapshotId(1)])
        .await
        .expect("recover");
    assert_eq!(recovered.registry.len(), 0);
    assert_eq!(
        recovered.discarded,
        vec![stats.statistics_path.clone()],
        "a manifest for a snapshot the table forgot is disposable"
    );
}

#[tokio::test]
async fn a_bitmap_survives_being_written_and_read_back() {
    let dir = scratch("persist_bitmap");

    // Two files, two groups each — postings with real rows to name.
    let mut sizes = BTreeMap::new();
    let mut files = Vec::new();
    for (index, tenants) in [
        vec![vec![1; 100], vec![2; 100]],
        vec![vec![3; 100], vec![1; 100]],
    ]
    .into_iter()
    .enumerate()
    {
        let (file, size) = write_parquet_grouped(&dir, &format!("{index}.parquet"), &tenants);
        sizes.insert(file.clone(), size);
        files.push(file);
    }
    let mut snapshot = Snapshot::root(SnapshotId(1));
    for file in &files {
        snapshot = snapshot.with_clean_file(file.clone());
    }
    let graph = SnapshotGraph::new().with(snapshot);

    let mut table = QuarryTable::new(
        schema(),
        TableId("events".into()),
        SnapshotId(1),
        graph.clone(),
        field_ids(),
    )
    .with_policy(POLICY)
    .on_object_store(ObjectStoreUrl::local_filesystem());
    for (file, size) in &sizes {
        table = table.with_parquet_file(file.clone(), *size);
    }

    let fixture_dir = Fixture {
        dir: dir.clone(),
        sizes,
        graph,
        files,
    };
    let (file_io, store) = stacks(&fixture_dir);

    let session = quarry().session();
    let built = build_bitmap(&session, &table, TENANT_FIELD)
        .await
        .expect("build");

    let path = store
        .write_bitmap(&events(), SnapshotId(1), POLICY, &built)
        .await
        .expect("write");
    let (read, policy) = read_bitmap(&file_io, &path, TENANT_FIELD)
        .await
        .expect("read")
        .expect("a usable bitmap");
    assert_eq!(policy, POLICY);
    assert_eq!(read, built, "postings round-trip exactly");

    // And a restart finds it, serving at row granularity.
    let recovered = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("recover");
    let id = index_id(&events(), TENANT_FIELD);
    let derived = recovered.registry.get(&id).expect("registered");

    let query = Query {
        table: events(),
        snapshot: SnapshotId(1),
        policy: POLICY,
        plan_hash: 0,
        plan: None,
        projected: BTreeSet::from([TENANT_FIELD]),
        predicates: vec![Predicate::Eq {
            field: TENANT_FIELD,
            value: hash_scalar(&ScalarValue::Int64(Some(1))),
        }],
        aggregate: None,
            nearest: None,
        approximate: false,
    };
    match derived.may_serve(&query, &fixture_dir.graph) {
        Decision::Use(Rewrite::Prune { files }) => {
            assert!(
                files.values().all(|scope| matches!(scope, Scope::Rows(_))),
                "the recovered piece posts rows: {files:?}"
            );
        }
        other => panic!("expected the recovered bitmap to serve, got {other:?}"),
    }
}

/// A filter set round-trips: write it, read it, and a restart finds and
/// serves it — identity carried by the blob's metadata, not the path.
#[tokio::test]
async fn a_filter_set_survives_being_written_and_read_back() {
    let dir = scratch("persist_filter_set");

    let mut sizes = BTreeMap::new();
    let mut files = Vec::new();
    for (index, tenants) in [
        vec![vec![1; 100], vec![2; 100]],
        vec![vec![3; 100], vec![1; 100]],
    ]
    .into_iter()
    .enumerate()
    {
        let (file, size) = write_parquet_grouped(&dir, &format!("{index}.parquet"), &tenants);
        sizes.insert(file.clone(), size);
        files.push(file);
    }
    let mut snapshot = Snapshot::root(SnapshotId(1));
    for file in &files {
        snapshot = snapshot.with_clean_file(file.clone());
    }
    let graph = SnapshotGraph::new().with(snapshot);

    let mut table = QuarryTable::new(
        schema(),
        TableId("events".into()),
        SnapshotId(1),
        graph.clone(),
        field_ids(),
    )
    .with_policy(POLICY)
    .on_object_store(ObjectStoreUrl::local_filesystem());
    for (file, size) in &sizes {
        table = table.with_parquet_file(file.clone(), *size);
    }

    let fixture_dir = Fixture {
        dir: dir.clone(),
        sizes,
        graph,
        files,
    };
    let (file_io, store) = stacks(&fixture_dir);

    let session = quarry().session();
    let (built, _) = build_filter_set(&session, &table, "tenant_id = 1")
        .await
        .expect("build");

    let path = store
        .write_filter_set(&events(), SnapshotId(1), POLICY, &built)
        .await
        .expect("write");
    let (read, policy) = read_filter_set(&file_io, &path)
        .await
        .expect("read")
        .expect("a usable filter set");
    assert_eq!(policy, POLICY);
    assert_eq!(read.filter(), built.filter());
    assert_eq!(
        read.postings(),
        built.postings(),
        "postings round-trip exactly"
    );

    // A restart finds it by listing, and it serves the filter it names.
    let recovered = store
        .recover(&events(), &[SnapshotId(1)])
        .await
        .expect("recover");
    let id = filter_set_id(&events(), built.filter());
    let derived = recovered.registry.get(&id).expect("registered");

    let query = Query {
        table: events(),
        snapshot: SnapshotId(1),
        policy: POLICY,
        plan_hash: 0,
        plan: Some(Plan::new(
            BTreeSet::from([TENANT_FIELD]),
            [built.filter().clone()],
        )),
        projected: BTreeSet::from([TENANT_FIELD]),
        predicates: Vec::new(),
        aggregate: None,
            nearest: None,
        approximate: false,
    };
    match derived.may_serve(&query, &fixture_dir.graph) {
        Decision::Use(Rewrite::Prune { files }) => {
            assert!(
                files.values().all(|scope| matches!(scope, Scope::Rows(_))),
                "the recovered piece posts rows: {files:?}"
            );
        }
        other => panic!("expected the recovered filter set to serve, got {other:?}"),
    }
}
