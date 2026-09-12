//! End-to-end: real SQL through DataFusion, with the rule choosing files.
//!
//! The point of these tests is not that SQL works — DataFusion's job — but
//! that `Derived::may_serve` is genuinely on the scan path, and that its
//! decisions hold up when a query engine re-applies the filter above the scan.
//!
//! ```sh
//! cargo test --features engine --test engine_sql
//! ```

#![cfg(feature = "engine")]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;

use quarry::derived::{Derived, DerivedId, PolicyFingerprint, Source};
use quarry::engine::{MaterializedResult, QuarryTable, hash_scalar};
use quarry::kinds::{Index, ResultCache};
use quarry::registry::Registry;
use quarry::snapshot::{DeleteState, FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};

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

/// One data file holding the given (tenant, message) rows.
fn batch(rows: &[(i64, &str)]) -> RecordBatch {
    let tenants: Int64Array = rows.iter().map(|(t, _)| *t).collect();
    let messages: StringArray = rows.iter().map(|(_, m)| Some(*m)).collect();
    RecordBatch::try_new(schema(), vec![Arc::new(tenants), Arc::new(messages)])
        .expect("batch matches schema")
}

fn file(name: &str) -> FileId {
    FileId(name.to_owned())
}

fn tenant_hash(tenant: i64) -> u64 {
    hash_scalar(&ScalarValue::Int64(Some(tenant)))
}

/// Three files: tenant 1 in "a", tenant 2 in "b", tenant 1 again in "c".
fn table_with_three_files(graph: SnapshotGraph, snapshot: SnapshotId) -> QuarryTable {
    QuarryTable::new(
        schema(),
        TableId("events".into()),
        snapshot,
        graph,
        field_ids(),
    )
    .with_policy(POLICY)
    .with_file(file("a"), vec![batch(&[(1, "a1"), (1, "a2")])])
    .with_file(file("b"), vec![batch(&[(2, "b1")])])
    .with_file(file("c"), vec![batch(&[(1, "c1")])])
}

fn flat_graph(files: &[&str], at: i64) -> SnapshotGraph {
    let mut snapshot = Snapshot::root(SnapshotId(at));
    for f in files {
        snapshot = snapshot.with_clean_file(file(f));
    }
    SnapshotGraph::new().with(snapshot)
}

/// An index over tenant_id built at `at`, covering the given files per tenant.
fn tenant_index(at: i64, entries: &[(i64, &str)]) -> Derived {
    let mut index = Index::new(TENANT_FIELD);
    for (tenant, f) in entries {
        index.insert(tenant_hash(*tenant), file(f));
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

#[tokio::test]
async fn without_derived_state_every_file_is_read() {
    let table = Arc::new(table_with_three_files(
        flat_graph(&["a", "b", "c"], 810),
        SnapshotId(810),
    ));

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(total_rows(&rows), 3);

    let report = table.last_scan().expect("a scan happened");
    assert_eq!(report.used, None);
    assert_eq!(
        report.files_read,
        BTreeSet::from([file("a"), file("b"), file("c")])
    );
}

#[tokio::test]
async fn an_index_prunes_the_files_read_without_changing_the_answer() {
    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, "a"), (1, "c"), (2, "b")]));

    let table = Arc::new(
        table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
            .with_registry(registry),
    );

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(total_rows(&rows), 3, "same answer as the full scan");

    let report = table.last_scan().expect("a scan happened");
    assert_eq!(report.used.as_deref(), Some("tenant_idx"));
    assert_eq!(
        report.files_read,
        BTreeSet::from([file("a"), file("c")]),
        "file b holds only tenant 2 and must not be read"
    );
}

#[tokio::test]
async fn a_stale_index_still_returns_every_row() {
    // The index was built at 810, before file "c" existed. The rule must add
    // it back, or tenant 1's row in "c" would silently disappear.
    let graph = SnapshotGraph::new()
        .with(
            Snapshot::root(SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b")),
        )
        .with(
            Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b"))
                .with_clean_file(file("c")),
        );

    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, "a"), (2, "b")]));

    let table = Arc::new(table_with_three_files(graph, SnapshotId(811)).with_registry(registry));

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(
        total_rows(&rows),
        3,
        "the row in the unindexed file must not be lost"
    );

    let report = table.last_scan().expect("a scan happened");
    assert_eq!(report.also_scanned, BTreeSet::from([file("c")]));
    assert_eq!(report.files_read, BTreeSet::from([file("a"), file("c")]));
}

#[tokio::test]
async fn an_index_from_an_abandoned_branch_is_not_used() {
    // 811 is abandoned; 812 is committed against 810 instead.
    let graph = SnapshotGraph::new()
        .with(
            Snapshot::root(SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b")),
        )
        .with(Snapshot::child_of(SnapshotId(811), SnapshotId(810)).with_clean_file(file("a")))
        .with(
            Snapshot::child_of(SnapshotId(812), SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b"))
                .with_clean_file(file("c")),
        );

    let mut registry = Registry::new();
    registry.register(tenant_index(811, &[(1, "a")]));

    let table = Arc::new(table_with_three_files(graph, SnapshotId(812)).with_registry(registry));

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(total_rows(&rows), 3);

    let report = table.last_scan().expect("a scan happened");
    assert_eq!(report.used, None, "derived state off the branch is refused");
}

#[tokio::test]
async fn a_compacted_away_file_is_never_read() {
    // Compaction replaced a and b with "merged"; the index still points at a.
    let graph = SnapshotGraph::new()
        .with(
            Snapshot::root(SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b")),
        )
        .with(Snapshot::child_of(SnapshotId(811), SnapshotId(810)).with_clean_file(file("merged")));

    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, "a"), (2, "b")]));

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
        .with_file(file("a"), vec![batch(&[(1, "stale")])])
        .with_file(file("merged"), vec![batch(&[(1, "m1"), (2, "m2")])]),
    );

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(
        total_rows(&rows),
        1,
        "the compacted-away file must not resurrect its rows"
    );

    let report = table.last_scan().expect("a scan happened");
    assert!(
        !report.files_read.contains(&file("a")),
        "read {:?}, which includes a file the table no longer references",
        report.files_read
    );
    assert_eq!(report.files_read, BTreeSet::from([file("merged")]));
}

#[tokio::test]
async fn a_policy_mismatch_falls_back_to_a_full_scan() {
    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, "a"), (1, "c"), (2, "b")]));

    let table = Arc::new(
        table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
            .with_registry(registry)
            .with_policy(PolicyFingerprint(999)),
    );

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(total_rows(&rows), 3);
    assert_eq!(table.last_scan().expect("scan").used, None);
}

#[tokio::test]
async fn a_predicate_the_index_cannot_probe_falls_back() {
    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, "a"), (1, "c"), (2, "b")]));

    let table = Arc::new(
        table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
            .with_registry(registry),
    );

    // A range, which the index has no postings for.
    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id > 1",
    )
    .await;
    assert_eq!(total_rows(&rows), 1);
    assert_eq!(
        table.last_scan().expect("scan").used,
        None,
        "an unprobeable predicate must not prune"
    );
}

#[tokio::test]
async fn the_operand_order_of_an_equality_does_not_matter() {
    let build = || {
        let mut registry = Registry::new();
        registry.register(tenant_index(810, &[(1, "a"), (1, "c"), (2, "b")]));
        Arc::new(
            table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
                .with_registry(registry),
        )
    };

    let normal = build();
    run(
        Arc::clone(&normal),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;

    let flipped = build();
    run(
        Arc::clone(&flipped),
        "SELECT * FROM events WHERE 1 = tenant_id",
    )
    .await;

    assert_eq!(
        normal.last_scan().expect("scan").files_read,
        flipped.last_scan().expect("scan").files_read,
        "a = 1 and 1 = a must prune identically"
    );
}

#[tokio::test]
async fn aggregation_over_a_pruned_scan_is_correct() {
    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, "a"), (1, "c"), (2, "b")]));

    let table = Arc::new(
        table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
            .with_registry(registry),
    );

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
    assert_eq!(
        table.last_scan().expect("scan").files_read,
        BTreeSet::from([file("a"), file("c")])
    );
}

/// A materialised answer to `SELECT * FROM events WHERE tenant_id = 1`.
///
/// The plan hash has to match the one the table computes, so it is read back
/// from a scan rather than guessed.
async fn plan_hash_of(sql: &str, table: Arc<QuarryTable>) -> u64 {
    run(Arc::clone(&table), sql).await;
    table.last_scan().expect("scan").plan_hash
}

#[tokio::test]
async fn a_materialised_result_is_read_instead_of_the_table() {
    let graph = flat_graph(&["a", "b", "c"], 810);
    let sql = "SELECT * FROM events WHERE tenant_id = 1";

    // Learn the plan hash the table will look up.
    let probe = Arc::new(table_with_three_files(graph.clone(), SnapshotId(810)));
    let hash = plan_hash_of(sql, probe).await;

    let stored = vec![batch(&[(1, "cached-1"), (1, "cached-2"), (1, "cached-3")])];
    let id = DerivedId("answer".into());

    let mut registry = Registry::new();
    registry.register(Derived::new(
        id.clone(),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
        },
        POLICY,
        128,
        Box::new(MaterializedResult::rows_of(hash, stored.clone())),
    ));

    let table = Arc::new(
        table_with_three_files(graph, SnapshotId(810))
            .with_registry(registry)
            .with_materialized(id, stored),
    );

    let rows = run(Arc::clone(&table), sql).await;
    assert_eq!(total_rows(&rows), 3);

    let report = table.last_scan().expect("scan");
    assert!(report.substituted, "the stored rows should have been read");
    assert!(
        report.files_read.is_empty(),
        "no data file should be touched, read {:?}",
        report.files_read
    );
}

#[tokio::test]
async fn a_stale_materialised_result_is_read_with_the_files_added_since() {
    let sql = "SELECT * FROM events WHERE tenant_id = 1";
    let graph = SnapshotGraph::new()
        .with(
            Snapshot::root(SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b")),
        )
        .with(
            Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b"))
                .with_clean_file(file("c")),
        );

    let probe = Arc::new(table_with_three_files(graph.clone(), SnapshotId(811)));
    let hash = plan_hash_of(sql, probe).await;

    // Stored at 810: the two rows of tenant 1 that existed in file "a".
    let stored = vec![batch(&[(1, "a1"), (1, "a2")])];
    let id = DerivedId("answer".into());

    let mut registry = Registry::new();
    registry.register(Derived::new(
        id.clone(),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
        },
        POLICY,
        128,
        Box::new(MaterializedResult::rows_of(hash, stored.clone())),
    ));

    let table = Arc::new(
        table_with_three_files(graph, SnapshotId(811))
            .with_registry(registry)
            .with_materialized(id, stored),
    );

    let rows = run(Arc::clone(&table), sql).await;
    assert_eq!(
        total_rows(&rows),
        3,
        "two stored rows plus the one in the file added since"
    );

    let report = table.last_scan().expect("scan");
    assert!(report.substituted);
    assert_eq!(report.also_scanned, BTreeSet::from([file("c")]));
}

#[tokio::test]
async fn an_aggregated_result_is_refused_once_a_file_is_added() {
    let sql = "SELECT * FROM events WHERE tenant_id = 1";
    let graph = SnapshotGraph::new()
        .with(
            Snapshot::root(SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b")),
        )
        .with(
            Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b"))
                .with_clean_file(file("c")),
        );

    let probe = Arc::new(table_with_three_files(graph.clone(), SnapshotId(811)));
    let hash = plan_hash_of(sql, probe).await;

    let stored = vec![batch(&[(1, "aggregated")])];
    let id = DerivedId("cube".into());

    let mut registry = Registry::new();
    registry.register(Derived::new(
        id.clone(),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
        },
        POLICY,
        16,
        // The same rows, declared aggregated rather than table-shaped.
        Box::new(MaterializedResult::aggregate_of(hash, stored.clone())),
    ));

    let table = Arc::new(
        table_with_three_files(graph, SnapshotId(811))
            .with_registry(registry)
            .with_materialized(id, stored),
    );

    let rows = run(Arc::clone(&table), sql).await;
    assert_eq!(
        total_rows(&rows),
        3,
        "a full scan, not the stored row plus new files"
    );

    let report = table.last_scan().expect("scan");
    assert_eq!(
        report.used, None,
        "an aggregate cannot be concatenated with raw rows"
    );
    assert!(!report.substituted);
}

#[tokio::test]
async fn a_substituting_candidate_without_stored_rows_is_skipped() {
    // The registry knows a result exists; the engine has no bytes for it.
    let sql = "SELECT * FROM events WHERE tenant_id = 1";
    let graph = flat_graph(&["a", "b", "c"], 810);

    let probe = Arc::new(table_with_three_files(graph.clone(), SnapshotId(810)));
    let hash = plan_hash_of(sql, probe).await;

    let mut registry = Registry::new();
    registry.register(Derived::new(
        DerivedId("recorded-only".into()),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
        },
        POLICY,
        1,
        Box::new(ResultCache::rows_of(hash, 3, 1)),
    ));

    let table = Arc::new(table_with_three_files(graph, SnapshotId(810)).with_registry(registry));

    let rows = run(Arc::clone(&table), sql).await;
    assert_eq!(total_rows(&rows), 3);
    assert_eq!(
        table.last_scan().expect("scan").used,
        None,
        "skipping is the safe direction: right answer, merely slower"
    );
}

#[tokio::test]
async fn deletes_do_not_disqualify_a_pruning_index() {
    // A delete-only commit: rows removed from "a", no file added or removed.
    let graph = SnapshotGraph::new()
        .with(
            Snapshot::root(SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b")),
        )
        .with(
            Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                .with_file(file("a"), DeleteState(7))
                .with_clean_file(file("b")),
        );

    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, "a"), (2, "b")]));

    let table = Arc::new(table_with_three_files(graph, SnapshotId(811)).with_registry(registry));

    run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(
        table.last_scan().expect("scan").used.as_deref(),
        Some("tenant_idx"),
        "the engine still reads the file and applies deletes, so pruning is safe"
    );
}
