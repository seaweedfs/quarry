//! Aggregate queries served from a cube — the plan-level half of the rule.
//!
//! A scan never sees a group-by, so all of this goes through `Session::sql`:
//! the aggregate is recognised in the logical plan, the registry decides
//! whether a cube may serve it, and the stored partials are re-aggregated at
//! the query's grain.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use url::Url;

use quarry::derived::{
    AggFunc, Aggregate, Derived, DerivedId, Measure, Plan, PolicyFingerprint, Source,
};
use quarry::engine::{MaterializedResult, Quarry, QuarryTable, Rollup, Session};
use quarry::registry::Registry;
use quarry::snapshot::{FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};

const DAY: u32 = 1;
const BYTES: u32 = 3;
const TENANT: u32 = 4;
const POLICY: PolicyFingerprint = PolicyFingerprint(1);

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("day", DataType::Int64, false),
        Field::new("bytes", DataType::Int64, false),
        Field::new("tenant_id", DataType::Int64, false),
    ]))
}

fn field_ids() -> BTreeMap<String, u32> {
    BTreeMap::from([
        ("day".to_owned(), DAY),
        ("bytes".to_owned(), BYTES),
        ("tenant_id".to_owned(), TENANT),
    ])
}

/// The base rows: six events across two days and two tenants.
fn batch() -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 1, 1, 2, 2, 2])),
            Arc::new(Int64Array::from(vec![10i64, 20, 30, 40, 50, 60])),
            Arc::new(Int64Array::from(vec![1i64, 1, 2, 1, 2, 2])),
        ],
    )
    .expect("batch")
}

fn table(at: SnapshotId) -> QuarryTable {
    QuarryTable::new(
        schema(),
        TableId("events".into()),
        at,
        SnapshotGraph::new().with(Snapshot::root(at).with_clean_file(FileId("a".into()))),
        field_ids(),
    )
    .with_policy(POLICY)
    .with_file(FileId("a".into()), vec![batch()])
}

fn count_star() -> Measure {
    Measure {
        func: AggFunc::Count,
        field: None,
    }
}

fn sum_bytes() -> Measure {
    Measure {
        func: AggFunc::Sum,
        field: Some(BYTES),
    }
}

/// The cube's grain and stored column names.
fn rollup() -> Rollup {
    Rollup::new(
        Aggregate {
            group_by: BTreeSet::from([DAY, TENANT]),
            measures: BTreeSet::from([count_star(), sum_bytes()]),
        },
        BTreeMap::from([
            (DAY, "day".to_owned()),
            (TENANT, "tenant_id".to_owned()),
            (BYTES, "bytes".to_owned()),
        ]),
    )
}

/// Partials at the cube's grain, in `Rollup::columns` order.
fn cube_rows() -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("day", DataType::Int64, false),
        Field::new("tenant_id", DataType::Int64, false),
        Field::new("count(*)", DataType::Int64, false),
        Field::new("sum(bytes)", DataType::Int64, false),
    ]));
    vec![
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![1i64, 2, 1, 2])),
                Arc::new(Int64Array::from(vec![2i64, 1, 1, 2])),
                Arc::new(Int64Array::from(vec![30i64, 30, 40, 110])),
            ],
        )
        .expect("cube"),
    ]
}

fn build_plan() -> Plan {
    // The cube was built over the whole table: no filters baked in.
    Plan::new(BTreeSet::from([DAY, BYTES, TENANT]), Vec::<String>::new())
}

fn with_cube(at: SnapshotId) -> Arc<QuarryTable> {
    let id = DerivedId("cube".into());
    let mut registry = Registry::new();
    registry.register(Derived::new(
        id.clone(),
        Source {
            table: TableId("events".into()),
            snapshot: at,
        },
        POLICY,
        128,
        Box::new(MaterializedResult::aggregate_of(
            build_plan(),
            rollup(),
            cube_rows(),
        )),
    ));
    Arc::new(
        table(at)
            .with_registry(registry)
            .with_cube(id, &rollup(), cube_rows()),
    )
}

fn session() -> Session {
    Quarry::new(
        Url::parse("memory://").expect("url"),
        Arc::new(object_store::memory::InMemory::new()),
    )
    .session()
}

/// (day, sum) pairs out of a result, whatever order they came back in.
fn pairs(rows: &[RecordBatch]) -> BTreeMap<i64, i64> {
    let mut out = BTreeMap::new();
    for batch in rows {
        let keys = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i64 keys");
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i64 values");
        for i in 0..batch.num_rows() {
            out.insert(keys.value(i), values.value(i));
        }
    }
    out
}

#[tokio::test]
async fn a_cube_serves_the_grain_it_was_built_at() {
    let table = with_cube(SnapshotId(810));
    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql("SELECT day, tenant_id, count(*), sum(bytes) FROM events GROUP BY day, tenant_id")
        .await
        .expect("sql");
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 4);
    assert_eq!(table.last_scan().expect("report").used, Some("cube".into()));
    assert!(table.last_scan().expect("report").substituted);
}

#[tokio::test]
async fn a_coarser_query_rolls_the_cube_up() {
    let table = with_cube(SnapshotId(810));
    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql("SELECT day, sum(bytes) FROM events GROUP BY day")
        .await
        .expect("sql");
    assert_eq!(
        pairs(&rows),
        BTreeMap::from([(1, 60), (2, 150)]),
        "day 1: 10+20+30, day 2: 40+50+60"
    );
    assert_eq!(table.last_scan().expect("report").used, Some("cube".into()));
}

#[tokio::test]
async fn a_stored_count_is_summed_not_counted() {
    let table = with_cube(SnapshotId(810));
    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql("SELECT tenant_id, count(*) FROM events GROUP BY tenant_id")
        .await
        .expect("sql");
    assert_eq!(pairs(&rows), BTreeMap::from([(1, 3), (2, 3)]));
    assert_eq!(table.last_scan().expect("report").used, Some("cube".into()));
}

#[tokio::test]
async fn a_query_with_no_grouping_rolls_everything_up() {
    let table = with_cube(SnapshotId(810));
    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql("SELECT count(*) FROM events")
        .await
        .expect("sql");
    let total: i64 = rows
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("i64")
                .values()
                .to_vec()
        })
        .sum();
    assert_eq!(total, 6);
    assert_eq!(table.last_scan().expect("report").used, Some("cube".into()));
}

#[tokio::test]
async fn a_filter_on_a_group_key_is_applied_to_the_partials() {
    let table = with_cube(SnapshotId(810));
    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql("SELECT day, count(*) FROM events WHERE day = 2 GROUP BY day")
        .await
        .expect("sql");
    assert_eq!(pairs(&rows), BTreeMap::from([(2, 3)]));
    assert_eq!(table.last_scan().expect("report").used, Some("cube".into()));
}

#[tokio::test]
async fn a_filter_on_a_non_key_field_cannot_be_served() {
    let table = with_cube(SnapshotId(810));
    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");

    // `bytes` is a measure field, not a key: filtering it after aggregation
    // would filter partial sums, which means something else entirely.
    let rows = session
        .sql("SELECT day, sum(bytes) FROM events WHERE bytes > 20 GROUP BY day")
        .await
        .expect("sql");
    assert_eq!(
        pairs(&rows),
        BTreeMap::from([(1, 30), (2, 150)]),
        "computed from raw rows: only bytes > 20 count"
    );
    assert_eq!(table.last_scan().expect("report").used, None);
}

#[tokio::test]
async fn a_measure_the_cube_does_not_hold_cannot_be_served() {
    let table = with_cube(SnapshotId(810));
    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql("SELECT day, min(bytes) FROM events GROUP BY day")
        .await
        .expect("sql");
    assert_eq!(pairs(&rows), BTreeMap::from([(1, 10), (2, 40)]));
    assert_eq!(table.last_scan().expect("report").used, None);
}

#[tokio::test]
async fn a_finer_grouping_cannot_be_served() {
    let table = with_cube(SnapshotId(810));
    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");

    // The cube collapsed bytes: grouping by it again cannot be un-aggregated.
    let rows = session
        .sql("SELECT day, bytes, count(*) FROM events GROUP BY day, bytes")
        .await
        .expect("sql");
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 6);
    assert_eq!(table.last_scan().expect("report").used, None);
}

#[tokio::test]
async fn a_cube_does_not_serve_across_a_snapshot_boundary() {
    // The cube knows snapshot 810; the table has moved to 811 with a file
    // added. Unionable is false — the partials cannot be concatenated with
    // raw rows — so the query must scan.
    let at = SnapshotId(811);
    let mut graph = SnapshotGraph::new()
        .with(Snapshot::root(SnapshotId(810)).with_clean_file(FileId("a".into())));
    graph = graph.with(
        Snapshot::child_of(at, SnapshotId(810))
            .with_clean_file(FileId("a".into()))
            .with_clean_file(FileId("b".into())),
    );
    let id = DerivedId("cube".into());
    let mut registry = Registry::new();
    registry.register(Derived::new(
        id.clone(),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
        },
        POLICY,
        128,
        Box::new(MaterializedResult::aggregate_of(
            build_plan(),
            rollup(),
            cube_rows(),
        )),
    ));
    let table = Arc::new(
        QuarryTable::new(schema(), TableId("events".into()), at, graph, field_ids())
            .with_policy(POLICY)
            .with_file(FileId("a".into()), vec![batch()])
            .with_file(FileId("b".into()), vec![batch()])
            .with_registry(registry)
            .with_cube(id, &rollup(), cube_rows()),
    );

    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");
    let rows = session
        .sql("SELECT count(*) FROM events")
        .await
        .expect("sql");
    let total: i64 = rows
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("i64")
                .values()
                .to_vec()
        })
        .sum();
    assert_eq!(total, 12, "both files' rows, not the stale cube's six");
    assert_eq!(table.last_scan().expect("report").used, None);
}

#[tokio::test]
async fn repeated_asks_make_the_optimizer_build_the_cube() {
    let mut registry = Registry::new();
    let shared = quarry::engine::shared(std::mem::take(&mut registry));
    let table = Arc::new(table(SnapshotId(810)).with_shared_registry(shared.clone()));
    let session = session();
    session.register("events", Arc::clone(&table)).expect("reg");

    let mut optimizer = quarry::engine::Optimizer::new(
        shared,
        quarry::workload::Policy::automatic(1_000_000_000).with_min_queries(2),
    );

    let sql = "SELECT day, sum(bytes) FROM events GROUP BY day";
    for _ in 0..2 {
        session.sql(sql).await.expect("sql");
        let report = table.last_scan().expect("scan");
        optimizer.observe(report.observation(1024));
        assert_eq!(report.used, None, "nothing to serve it yet");
    }

    let round = optimizer.round(&session, &table).await;
    assert_eq!(round.built.len(), 1, "the ask earned a cube");

    let rows = session
        .sql("SELECT day, sum(bytes) FROM events GROUP BY day")
        .await
        .expect("sql");
    assert_eq!(pairs(&rows), BTreeMap::from([(1, 60), (2, 150)]));
    let report = table.last_scan().expect("report");
    assert!(report.substituted);
    assert!(
        report
            .used
            .as_deref()
            .is_some_and(|id| id.starts_with("cube:")),
        "served by the cube the round built: {:?}",
        report.used
    );
}
