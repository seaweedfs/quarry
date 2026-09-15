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

use quarry::cost::{Cost, PriceTable};
use quarry::derived::{
    Derived, DerivedId, FieldId, Kind, Plan, PolicyFingerprint, Query, Refreshed, Rewrite, Scope,
    Source,
};
use quarry::engine::{MaterializedResult, QuarryTable, Rollup, hash_scalar};
use quarry::kinds::{Index, Projection, ResultCache};
use quarry::registry::Registry;
use quarry::snapshot::{DeleteState, Diff, FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};

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
    field_index("tenant_idx", at, index)
}

/// An index over message built at `at`, under the given id.
fn message_index(id: &str, at: i64, entries: &[(&str, &str)]) -> Derived {
    let mut index = Index::new(7);
    for (message, f) in entries {
        index.insert(
            hash_scalar(&ScalarValue::Utf8(Some(message.to_string()))),
            file(f),
        );
    }
    field_index(id, at, index)
}

/// A kind admitting only the named batches of the named files — stands in
/// for a kind that knows row groups, which an in-memory file's batches are.
#[derive(Debug)]
struct ScopedIndex {
    field: FieldId,
    files: BTreeMap<FileId, Scope>,
}

impl Kind for ScopedIndex {
    fn name(&self) -> &'static str {
        "scoped"
    }
    fn matches(&self, query: &Query) -> Option<Rewrite> {
        (!query.equalities(self.field).is_empty()).then(|| Rewrite::Prune {
            files: self.files.clone(),
        })
    }
    fn cost(&self, _prices: &PriceTable) -> Cost {
        Cost::ZERO
    }
    fn refresh(&mut self, _diff: &Diff) -> Refreshed {
        Refreshed::UpToDate
    }
}

fn scoped_index(at: i64, files: &[(&str, Scope)]) -> Derived {
    Derived::new(
        DerivedId("scoped_idx".into()),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(at),
        },
        POLICY,
        1,
        Box::new(ScopedIndex {
            field: TENANT_FIELD,
            files: files
                .iter()
                .map(|(f, scope)| (file(f), scope.clone()))
                .collect(),
        }),
    )
}

fn field_index(id: &str, at: i64, index: Index) -> Derived {
    Derived::new(
        DerivedId(id.into()),
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
    assert!(report.used.is_empty());
    assert_eq!(
        report.files_read,
        BTreeSet::from([file("a"), file("b"), file("c")])
    );
}

#[tokio::test]
async fn two_indexes_intersect_their_candidate_files() {
    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, "a"), (1, "c"), (2, "b")]));
    // The a1 posting names b too — postings may over-claim (the engine
    // re-checks rows), and here it makes the message index alone insufficient.
    registry.register(message_index(
        "message_idx",
        810,
        &[
            ("a1", "a"),
            ("a1", "b"),
            ("a2", "a"),
            ("b1", "b"),
            ("c1", "c"),
        ],
    ));

    let table = Arc::new(
        table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
            .with_registry(registry),
    );

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1 AND message = 'a1'",
    )
    .await;
    assert_eq!(total_rows(&rows), 1, "same answer as the full scan");

    let report = table.last_scan().expect("a scan happened");
    assert_eq!(
        report.files_read,
        BTreeSet::from([file("a")]),
        "message says {{a,b}}, tenant says {{a,c}}: the conjunct reads only a"
    );
    assert_eq!(
        report.used.len(),
        2,
        "both indexes narrowed the plan and both are credited"
    );
}

#[tokio::test]
async fn an_index_that_narrows_nothing_is_not_credited() {
    let mut registry = Registry::new();
    registry.register(tenant_index(810, &[(1, "a"), (1, "c"), (2, "b")]));
    // A conservative index naming every file still prunes correctly, but for
    // this query it adds nothing over the tenant index.
    // The id sorts after "tenant_idx" so the tenant index leads and this
    // one is the candidate that fails to narrow it.
    registry.register(message_index(
        "wide_idx",
        810,
        &[("a1", "a"), ("a1", "b"), ("a1", "c")],
    ));

    let table = Arc::new(
        table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
            .with_registry(registry),
    );

    run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1 AND message = 'a1'",
    )
    .await;

    let report = table.last_scan().expect("a scan happened");
    assert_eq!(report.files_read, BTreeSet::from([file("a"), file("c")]),);
    assert_eq!(report.used, ["tenant_idx"], "only the piece that narrowed");
}

#[tokio::test]
async fn a_scoped_prune_reads_only_the_named_batches() {
    // File a holds tenant 1 in batch 0 and tenant 2 in batch 1 — an
    // in-memory file's batches playing the role of row groups — and the
    // kind names batch 0 only. File c is admitted whole.
    let mut registry = Registry::new();
    registry.register(scoped_index(
        810,
        &[
            ("a", Scope::Groups(BTreeSet::from([0]))),
            ("c", Scope::Whole),
        ],
    ));
    let table = Arc::new(
        QuarryTable::new(
            schema(),
            TableId("events".into()),
            SnapshotId(810),
            flat_graph(&["a", "c"], 810),
            field_ids(),
        )
        .with_policy(POLICY)
        .with_file(file("a"), vec![batch(&[(1, "a1")]), batch(&[(2, "a2")])])
        .with_file(file("c"), vec![batch(&[(1, "c1")])])
        .with_registry(registry),
    );

    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(
        total_rows(&rows),
        2,
        "a1 and c1 — the scope never changes the answer"
    );

    let report = table.last_scan().expect("a scan happened");
    assert_eq!(
        report.groups,
        BTreeMap::from([(file("a"), BTreeSet::from([0]))]),
        "the plan carries the scope through; c is whole and absent"
    );
    assert_eq!(report.used, ["scoped_idx"]);
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
    assert_eq!(report.used.first().map(String::as_str), Some("tenant_idx"));
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
    assert!(
        report.used.is_empty(),
        "derived state off the branch is refused"
    );
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
    assert!(table.last_scan().expect("scan").used.is_empty());
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
    assert!(
        table.last_scan().expect("scan").used.is_empty(),
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

/// The exact plan the table computes for `sql`.
///
/// Read back from a real scan rather than guessed: a stored result is matched
/// against the plan itself, not against a hash of it, so the plan has to be
/// the one the table actually produces.
async fn plan_of(sql: &str, table: Arc<QuarryTable>) -> Plan {
    run(Arc::clone(&table), sql).await;
    table
        .last_scan()
        .expect("scan")
        .plan
        .expect("this query is exactly describable")
}

/// Two different filters must not share one cached answer.
///
/// This was a real wrong answer reachable through plain SQL, not a theoretical
/// one. Plan identity used to be a hash of the *translated predicates*, and
/// `Predicate` collapses everything it does not model to `Opaque { field }` —
/// so `tenant_id > 1` and `tenant_id < 3` were the same predicate, hashed to
/// the same value, and a result stored for one was served for the other.
///
/// Nothing about hashing caused it: the representation being hashed was lossy.
/// The fix compares an exact rendering of the plan instead.
#[tokio::test]
async fn two_different_filters_do_not_share_a_cached_answer() {
    let graph = flat_graph(&["a", "b", "c"], 810);
    let stored_for = "SELECT * FROM events WHERE tenant_id > 1";
    let asked = "SELECT * FROM events WHERE tenant_id < 3";

    let probe = Arc::new(table_with_three_files(graph.clone(), SnapshotId(810)));

    run(Arc::clone(&probe), stored_for).await;
    let for_gt = probe.last_scan().expect("scan");
    run(Arc::clone(&probe), asked).await;
    let for_lt = probe.last_scan().expect("scan");

    // The reason an exact plan is needed, asserted rather than argued: the
    // lossy shape cannot tell these two queries apart. Both filters collapse
    // to `Opaque { tenant_id }`, so anything derived from `Predicate` — a
    // hash of it very much included — calls them the same query.
    assert_eq!(
        for_gt.fingerprint, for_lt.fingerprint,
        "the lossy shape is expected to be blind to the difference"
    );

    // The exact plan is not.
    let stored_plan = for_gt.plan.clone().expect("describable");
    let asked_plan = for_lt.plan.clone().expect("describable");
    assert_ne!(stored_plan, asked_plan);

    // The table holds tenants 1, 1, 2, 1 — so `< 3` is all four rows and
    // `> 1` is one. These stored rows are neither, which is what makes the
    // assertions below able to tell substitution from a real scan.
    let stored = vec![batch(&[(2, "gt-1"), (3, "gt-1")])];
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
        Box::new(MaterializedResult::rows_of(stored_plan, stored.clone())),
    ));

    let table = Arc::new(
        table_with_three_files(graph, SnapshotId(810))
            .with_registry(registry)
            .with_materialized(id, stored),
    );

    let rows = run(Arc::clone(&table), asked).await;
    let report = table.last_scan().expect("scan");
    assert!(
        !report.substituted,
        "a result stored for `tenant_id > 1` must not answer `tenant_id < 3`"
    );

    // And the answer is the table's own: every row, since all four tenants
    // recorded are below three.
    assert_eq!(total_rows(&rows), 4);
    assert!(
        !report.files_read.is_empty(),
        "it should have gone to the table"
    );
}

/// The same bug class one step further out: filters that are not
/// `BinaryExpr` at all.
///
/// `tenant_id IN (1, 2)` does not translate to a `Predicate` — `predicate()`
/// drops to marking the column opaque — so the shape loses the entire
/// filter. Whether the *plan* still tells `IN (1, 2)` from `IN (1, 3)` is
/// the Phase-12 question again, and it is only answered if the fallback
/// rendering is faithful.
#[tokio::test]
async fn an_in_list_does_not_share_a_cached_answer_with_a_different_one() {
    let graph = flat_graph(&["a", "b", "c"], 810);
    let stored_for = "SELECT * FROM events WHERE tenant_id IN (1, 2)";
    let asked = "SELECT * FROM events WHERE tenant_id IN (1, 3)";

    let probe = Arc::new(table_with_three_files(graph.clone(), SnapshotId(810)));
    run(Arc::clone(&probe), stored_for).await;
    let for_first = probe.last_scan().expect("scan");
    run(Arc::clone(&probe), asked).await;
    let for_second = probe.last_scan().expect("scan");

    // Same shape blindness, one level deeper: to the predicates these are
    // identical AND carry no comparison at all, just `Opaque { tenant_id }`.
    assert_eq!(
        for_first.fingerprint, for_second.fingerprint,
        "the shape cannot see inside an IN list"
    );

    let stored_plan = for_first.plan.clone().expect("describable");
    let asked_plan = for_second.plan.clone().expect("describable");
    assert_ne!(stored_plan, asked_plan, "the plan must see inside it");

    let stored = vec![batch(&[(1, "in12"), (2, "in12")])];
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
        Box::new(MaterializedResult::rows_of(stored_plan, stored.clone())),
    ));
    let table = Arc::new(
        table_with_three_files(graph, SnapshotId(810))
            .with_registry(registry)
            .with_materialized(id, stored),
    );

    let rows = run(Arc::clone(&table), asked).await;
    let report = table.last_scan().expect("scan");
    assert!(
        !report.substituted,
        "a result stored for IN (1, 2) must not answer IN (1, 3)"
    );
    // IN (1, 3) over tenants 1, 1, 2, 1 keeps the three 1s.
    assert_eq!(total_rows(&rows), 3);
}

/// The checkable half of the completeness precondition.
///
/// Row *count* cannot be verified — that would take the scan being avoided —
/// but schema can: stored rows are served under the table's schema, so a
/// batch that does not have it is a caller bug, and it is louder to refuse it
/// at registration than to let Arrow error mid-query.
#[test]
#[should_panic(expected = "must be table-shaped")]
fn rows_stored_under_the_wrong_schema_are_rejected_on_the_way_in() {
    let graph = flat_graph(&["a"], 810);
    let wrong_schema = Arc::new(Schema::new(vec![Field::new(
        "tenant_id",
        DataType::Utf8, // the table's is Int64
        false,
    )]));
    let wrong = RecordBatch::try_new(
        wrong_schema,
        vec![Arc::new(StringArray::from(vec!["not-an-int"]))],
    )
    .expect("batch matches its own schema");

    table_with_three_files(graph, SnapshotId(810))
        .with_materialized(DerivedId("bad".into()), vec![wrong]);
}

#[tokio::test]
async fn a_materialised_result_is_read_instead_of_the_table() {
    let graph = flat_graph(&["a", "b", "c"], 810);
    let sql = "SELECT * FROM events WHERE tenant_id = 1";

    // Learn the plan hash the table will look up.
    let probe = Arc::new(table_with_three_files(graph.clone(), SnapshotId(810)));
    let plan = plan_of(sql, probe).await;

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
        Box::new(MaterializedResult::rows_of(plan.clone(), stored.clone())),
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
    let plan = plan_of(sql, probe).await;

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
        Box::new(MaterializedResult::rows_of(plan.clone(), stored.clone())),
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
    let plan = plan_of(sql, probe).await;

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
        Box::new(MaterializedResult::aggregate_of(
            plan.clone(),
            Rollup::new(
                quarry::derived::Aggregate {
                    group_by: BTreeSet::from([TENANT_FIELD]),
                    measures: BTreeSet::from([quarry::derived::Measure {
                        func: quarry::derived::AggFunc::Count,
                        field: None,
                    }]),
                },
                BTreeMap::from([(TENANT_FIELD, "tenant_id".to_owned())]),
            ),
            stored.clone(),
        )),
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
    assert!(
        report.used.is_empty(),
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
    let plan = plan_of(sql, probe).await;

    let mut registry = Registry::new();
    registry.register(Derived::new(
        DerivedId("recorded-only".into()),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
        },
        POLICY,
        1,
        Box::new(ResultCache::rows_of(plan.clone(), 3, 1)),
    ));

    let table = Arc::new(table_with_three_files(graph, SnapshotId(810)).with_registry(registry));

    let rows = run(Arc::clone(&table), sql).await;
    assert_eq!(total_rows(&rows), 3);
    assert!(
        table.last_scan().expect("scan").used.is_empty(),
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
        table
            .last_scan()
            .expect("scan")
            .used
            .first()
            .map(String::as_str),
        Some("tenant_idx"),
        "the engine still reads the file and applies deletes, so pruning is safe"
    );
}

/// A stored column subset: `message` alone, narrower than the table.
fn message_batch(rows: &[&str]) -> RecordBatch {
    let messages: StringArray = rows.iter().map(|m| Some(*m)).collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "message",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(messages)],
    )
    .expect("batch")
}

fn message_projection(at: i64) -> Derived {
    Derived::new(
        DerivedId("message_proj".into()),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(at),
        },
        POLICY,
        1,
        Box::new(Projection::covering([7], 1)),
    )
}

#[tokio::test]
async fn a_projection_serves_a_query_touching_only_its_columns() {
    // The projection holds every row's `message`; the filter still applies
    // above the scan, so the answer is unchanged.
    let mut registry = Registry::new();
    registry.register(message_projection(810));
    let table = Arc::new(
        table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
            .with_registry(registry)
            .with_projection(
                DerivedId("message_proj".into()),
                vec![message_batch(&["a1", "a2", "b1", "c1"])],
            ),
    );

    let rows = run(
        Arc::clone(&table),
        "SELECT message FROM events WHERE message = 'a1'",
    )
    .await;
    assert_eq!(total_rows(&rows), 1, "only a1 matches");

    let report = table.last_scan().expect("scan");
    assert!(report.files_read.is_empty(), "no file was read");
    assert_eq!(report.used, vec!["message_proj".to_string()]);
}

#[tokio::test]
async fn a_projection_cannot_serve_a_column_it_does_not_hold() {
    let mut registry = Registry::new();
    registry.register(message_projection(810));
    let table = Arc::new(
        table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
            .with_registry(registry)
            .with_projection(
                DerivedId("message_proj".into()),
                vec![message_batch(&["a1", "a2", "b1", "c1"])],
            ),
    );

    // tenant_id is not stored, so nothing may substitute.
    let rows = run(
        Arc::clone(&table),
        "SELECT * FROM events WHERE tenant_id = 1",
    )
    .await;
    assert_eq!(total_rows(&rows), 3);
    assert_eq!(
        table.last_scan().expect("scan").files_read,
        ["a", "b", "c"].iter().map(|f| file(f)).collect(),
        "every file was read"
    );
}

#[tokio::test]
async fn a_stale_projection_unions_with_the_file_added_since() {
    // Built at 810 holding the messages of a and b; c arrives at 811 and is
    // read whole — the union is the answer.
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
    registry.register(message_projection(810));
    let table = Arc::new(
        table_with_three_files(graph, SnapshotId(811))
            .with_registry(registry)
            .with_projection(
                DerivedId("message_proj".into()),
                vec![message_batch(&["a1", "a2", "b1"])],
            ),
    );

    let rows = run(
        Arc::clone(&table),
        "SELECT message FROM events WHERE message = 'c1'",
    )
    .await;
    assert_eq!(total_rows(&rows), 1, "only c holds it");

    let report = table.last_scan().expect("scan");
    assert_eq!(report.files_read, BTreeSet::from([file("c")]));
    assert_eq!(report.used, vec!["message_proj".to_string()]);
    assert_eq!(report.also_scanned, BTreeSet::from([file("c")]));
}

#[tokio::test]
async fn a_projection_stored_out_of_order_is_served_in_table_order() {
    // Every column, stored in the opposite order — canonicalised on the way
    // in, so serving is the full-schema path.
    let tenants: Int64Array = vec![1, 2].into_iter().map(Some).collect();
    let messages: StringArray = vec!["a1", "b1"].into_iter().map(Some).collect();
    let reordered = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("message", DataType::Utf8, false),
            Field::new("tenant_id", DataType::Int64, false),
        ])),
        vec![Arc::new(messages), Arc::new(tenants)],
    )
    .expect("batch");

    let mut registry = Registry::new();
    registry.register(Derived::new(
        DerivedId("all_proj".into()),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
        },
        POLICY,
        1,
        Box::new(Projection::covering([TENANT_FIELD, 7], 1)),
    ));
    let table = Arc::new(
        table_with_three_files(flat_graph(&["a", "b", "c"], 810), SnapshotId(810))
            .with_registry(registry)
            .with_projection(DerivedId("all_proj".into()), vec![reordered]),
    );

    let rows = run(Arc::clone(&table), "SELECT * FROM events").await;
    assert_eq!(total_rows(&rows), 2, "the two stored rows");
    let names: Vec<String> = rows[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, vec!["tenant_id", "message"], "table order restored");
}
