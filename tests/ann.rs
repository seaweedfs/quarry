//! Top-k nearest-neighbour queries served from a vector index — the
//! plan-level half of the rule, for the `ORDER BY distance LIMIT k` shape.
//!
//! A scan never sees the sort or the limit, so all of this goes through
//! `Session::sql`: the shape is recognised in the logical plan, the registry
//! decides whether a vector index may serve it, and the stored rows are
//! re-ranked by the same distance function the query named.
//!
//! ```sh
//! cargo test --features engine --test ann
//! ```

#![cfg(feature = "engine")]

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use url::Url;

use quarry::derived::{Derived, DerivedId, Metric, PolicyFingerprint, Source};
use quarry::engine::{Quarry, QuarryTable, Session};
use quarry::kinds::VectorIndex;
use quarry::registry::Registry;
use quarry::snapshot::{DeleteState, FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};

const ID: u32 = 1;
const EMBEDDING: u32 = 2;
const TENANT: u32 = 3;
const DIM: i32 = 3;
const POLICY: PolicyFingerprint = PolicyFingerprint(1);

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
            true,
        ),
        Field::new("tenant_id", DataType::Int64, false),
    ]))
}

fn field_ids() -> BTreeMap<String, u32> {
    BTreeMap::from([
        ("id".to_owned(), ID),
        ("embedding".to_owned(), EMBEDDING),
        ("tenant_id".to_owned(), TENANT),
    ])
}

/// Five rows whose vectors sit at known distances from the origin, so a
/// top-k answer is checkable by hand rather than by re-running the query.
///
/// ```text
/// id  vector          L2 from [0,0,0]   tenant
/// 1   [1, 0, 0]                     1        1
/// 2   [0, 2, 0]                     4        1
/// 3   [0, 0, 3]                     9        2
/// 4   [4, 0, 0]                    16        2
/// 5   [0, 5, 0]                    25        1
/// ```
fn rows() -> Vec<(i64, [f32; 3], i64)> {
    vec![
        (1, [1.0, 0.0, 0.0], 1),
        (2, [0.0, 2.0, 0.0], 1),
        (3, [0.0, 0.0, 3.0], 2),
        (4, [4.0, 0.0, 0.0], 2),
        (5, [0.0, 5.0, 0.0], 1),
    ]
}

fn batch() -> RecordBatch {
    let ids: Int64Array = rows().iter().map(|(id, _, _)| *id).collect();
    let tenants: Int64Array = rows().iter().map(|(_, _, t)| *t).collect();
    let flat: Vec<f32> = rows().iter().flat_map(|(_, v, _)| *v).collect();
    let embeddings = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM,
        Arc::new(Float32Array::from(flat)),
        None,
    )
    .expect("embeddings");

    RecordBatch::try_new(
        schema(),
        vec![Arc::new(ids), Arc::new(embeddings), Arc::new(tenants)],
    )
    .expect("batch")
}

fn graph(at: SnapshotId) -> SnapshotGraph {
    SnapshotGraph::new().with(Snapshot::root(at).with_clean_file(FileId("a".into())))
}

fn table(at: SnapshotId) -> QuarryTable {
    QuarryTable::new(schema(), TableId("docs".into()), at, graph(at), field_ids())
        .with_policy(POLICY)
        .with_file(FileId("a".into()), vec![batch()])
}

/// A vector index over `embedding`, with the table's rows stored beside it.
fn vector_index(id: &str, at: SnapshotId, metric: Metric, dimension: u32) -> Derived {
    Derived::new(
        DerivedId(id.to_owned()),
        Source {
            table: TableId("docs".into()),
            snapshot: at,
        },
        POLICY,
        4096,
        Box::new(VectorIndex::new(EMBEDDING, metric, dimension, 4096)),
    )
}

fn session() -> Session {
    Quarry::new(
        Url::parse("memory://").expect("url"),
        Arc::new(object_store::memory::InMemory::new()),
    )
    .session()
}

/// A table with a usable L2 index over `embedding`, rows supplied.
fn with_index(metric: Metric, dimension: u32) -> Arc<QuarryTable> {
    let mut registry = Registry::new();
    registry.register(vector_index("vec", SnapshotId(810), metric, dimension));
    Arc::new(
        table(SnapshotId(810))
            .with_registry(registry)
            .with_projection(DerivedId("vec".into()), vec![batch()]),
    )
}

/// The `id` column of an answer, in the order it came back.
fn ids(batches: &[RecordBatch]) -> Vec<i64> {
    let mut out = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name("id")
            .expect("id")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64");
        for row in 0..batch.num_rows() {
            out.push(column.value(row));
        }
    }
    out
}

const ORIGIN: &str = "arrow_cast(make_array(0.0, 0.0, 0.0), 'FixedSizeList(3, Float32)')";

#[tokio::test]
async fn a_top_k_search_is_served_by_a_vector_index() {
    let table = with_index(Metric::L2, DIM as u32);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 3"
        ))
        .await
        .expect("query");

    assert_eq!(
        ids(&rows),
        vec![1, 2, 3],
        "the three nearest the origin, in order"
    );

    let report = table.last_scan().expect("a scan happened");
    assert!(report.substituted, "the index supplied the rows");
    assert_eq!(report.used, vec!["vec".to_owned()]);
    assert!(
        report.files_read.is_empty(),
        "no data file was read: that is the saving"
    );
    assert!(
        !report.approximate,
        "a flat index holds every row and the sort re-ranks them: exact"
    );
}

#[tokio::test]
async fn the_answer_matches_the_unaided_query_exactly() {
    // The property that matters: substituting must not change the answer.
    let served = with_index(Metric::L2, DIM as u32);
    let unaided = Arc::new(table(SnapshotId(810)));

    let sql =
        format!("SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 4");

    let one = session();
    one.register("docs", Arc::clone(&served)).expect("reg");
    let with = one.sql(&sql).await.expect("query");

    let other = session();
    other.register("docs", Arc::clone(&unaided)).expect("reg");
    let without = other.sql(&sql).await.expect("query");

    assert_eq!(ids(&with), ids(&without));
    assert!(served.last_scan().expect("scan").substituted);
    assert!(!unaided.last_scan().expect("scan").substituted);
}

#[tokio::test]
async fn a_filter_below_the_sort_still_applies_after_substituting() {
    let table = with_index(Metric::L2, DIM as u32);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql(&format!(
            "SELECT id FROM docs WHERE tenant_id = 1 \
             ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 2"
        ))
        .await
        .expect("query");

    assert_eq!(
        ids(&rows),
        vec![1, 2],
        "tenant 1's nearest two, not the table's"
    );
    assert!(table.last_scan().expect("scan").substituted);
}

#[tokio::test]
async fn a_wrong_dimension_index_does_not_serve() {
    // An index built over 128-dimensional vectors cannot answer a
    // 3-dimensional sort, and must not be allowed to try.
    let table = with_index(Metric::L2, 128);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 3"
        ))
        .await
        .expect("query");

    assert_eq!(ids(&rows), vec![1, 2, 3], "still the right answer");
    let report = table.last_scan().expect("scan");
    assert!(!report.substituted, "the scan ran; the index did not serve");
}

#[tokio::test]
async fn a_cosine_index_does_not_serve_an_l2_sort() {
    let table = with_index(Metric::Cosine, DIM as u32);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 3"
        ))
        .await
        .expect("query");

    assert_eq!(ids(&rows), vec![1, 2, 3]);
    assert!(!table.last_scan().expect("scan").substituted);
}

#[tokio::test]
async fn a_descending_sort_is_not_a_nearest_neighbour_ask() {
    // DESC asks for the *farthest* rows. A nearest-neighbour index has no
    // claim to them, so the shape must not be recognised.
    let table = with_index(Metric::L2, DIM as u32);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) DESC LIMIT 3"
        ))
        .await
        .expect("query");

    assert_eq!(ids(&rows), vec![5, 4, 3], "the farthest three");
    assert!(!table.last_scan().expect("scan").substituted);
}

#[tokio::test]
async fn an_offset_is_not_a_top_k() {
    let table = with_index(Metric::L2, DIM as u32);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) \
             LIMIT 2 OFFSET 2"
        ))
        .await
        .expect("query");

    assert_eq!(ids(&rows), vec![3, 4], "still correct");
    assert!(!table.last_scan().expect("scan").substituted);
}

#[tokio::test]
async fn a_sort_with_no_limit_is_not_a_top_k() {
    let table = with_index(Metric::L2, DIM as u32);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN})"
        ))
        .await
        .expect("query");

    assert_eq!(ids(&rows), vec![1, 2, 3, 4, 5]);
    assert!(!table.last_scan().expect("scan").substituted);
}

#[tokio::test]
async fn an_ordinary_sort_is_not_rewritten() {
    let table = with_index(Metric::L2, DIM as u32);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql("SELECT id FROM docs ORDER BY tenant_id, id LIMIT 3")
        .await
        .expect("query");

    assert_eq!(ids(&rows), vec![1, 2, 5]);
    assert!(
        !table.last_scan().expect("scan").substituted,
        "a sort on an ordinary column is nothing a vector index answers"
    );
}

#[tokio::test]
async fn a_delete_since_the_build_disqualifies_the_index() {
    // Substituting kinds tolerate only additive change: stored rows would
    // include a row the table no longer has.
    let mut registry = Registry::new();
    registry.register(vector_index("vec", SnapshotId(810), Metric::L2, DIM as u32));

    let graph = SnapshotGraph::new()
        .with(Snapshot::root(SnapshotId(810)).with_clean_file(FileId("a".into())))
        .with(
            Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                .with_file(FileId("a".into()), DeleteState(7)),
        );

    let table = Arc::new(
        QuarryTable::new(
            schema(),
            TableId("docs".into()),
            SnapshotId(811),
            graph,
            field_ids(),
        )
        .with_policy(POLICY)
        .with_file(FileId("a".into()), vec![batch()])
        .with_registry(registry)
        .with_projection(DerivedId("vec".into()), vec![batch()]),
    );

    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");
    let rows = session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 3"
        ))
        .await
        .expect("query");

    assert_eq!(ids(&rows), vec![1, 2, 3], "read from the table");
    assert!(!table.last_scan().expect("scan").substituted);
}

#[tokio::test]
async fn an_append_since_the_build_leaves_the_answer_correct() {
    // A flat index is the whole table, so an append leaves it holding fewer
    // rows than the table has. The leaf swap reads the stored rows *instead*
    // of the files, so it cannot also read the residual — and rather than
    // return a top-k over the wrong row set, it declines and the scan runs.
    let mut registry = Registry::new();
    registry.register(vector_index("vec", SnapshotId(810), Metric::L2, DIM as u32));

    // A second file, nearer the origin than anything the index holds.
    let nearer = {
        let ids: Int64Array = vec![6i64].into();
        let tenants: Int64Array = vec![1i64].into();
        let embeddings = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            DIM,
            Arc::new(Float32Array::from(vec![0.0f32, 0.0, 0.5])),
            None,
        )
        .expect("embeddings");
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(ids), Arc::new(embeddings), Arc::new(tenants)],
        )
        .expect("batch")
    };

    let graph = SnapshotGraph::new()
        .with(Snapshot::root(SnapshotId(810)).with_clean_file(FileId("a".into())))
        .with(
            Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                .with_clean_file(FileId("a".into()))
                .with_clean_file(FileId("b".into())),
        );

    let table = Arc::new(
        QuarryTable::new(
            schema(),
            TableId("docs".into()),
            SnapshotId(811),
            graph,
            field_ids(),
        )
        .with_policy(POLICY)
        .with_file(FileId("a".into()), vec![batch()])
        .with_file(FileId("b".into()), vec![nearer])
        .with_registry(registry)
        .with_projection(DerivedId("vec".into()), vec![batch()]),
    );

    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");
    let rows = session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 2"
        ))
        .await
        .expect("query");

    assert_eq!(
        ids(&rows),
        vec![6, 1],
        "row 6 is nearest and the index has never seen it"
    );
    let report = table.last_scan().expect("scan");
    assert!(
        !report.substituted,
        "serving from a stale flat index would have missed row 6"
    );
}

#[tokio::test]
async fn a_built_index_is_keyed_on_what_it_covers_not_on_the_k() {
    // Why this matters for refresh: the optimizer rebuilds onto the same id,
    // so an index whose table has moved is *replaced* rather than
    // accumulated beside. (The refresh step itself needs a Parquet table —
    // `stale` measures residual bytes, which an in-memory table has none of
    // — and is exercised in tests/loop_closes.rs for the other kinds.)
    let shared = quarry::engine::shared(Registry::new());
    let table = Arc::new(table(SnapshotId(810)).with_shared_registry(Arc::clone(&shared)));
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let mut optimizer = quarry::engine::Optimizer::new(
        Arc::clone(&shared),
        quarry::workload::Policy::automatic(1_000_000_000).with_min_queries(2),
    )
    .for_reader(POLICY);

    let sql =
        format!("SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 3");
    for _ in 0..2 {
        session.sql(&sql).await.expect("query");
        let report = table.last_scan().expect("scan");
        optimizer.observe(report.observation(report.bytes_if_full_scan));
    }
    let built = optimizer.round(&session, &table).await;
    assert_eq!(built.built.len(), 1, "{built:?}");

    // It serves at the snapshot it was built for.
    session.sql(&sql).await.expect("query");
    assert!(table.last_scan().expect("scan").substituted);

    // A rebuild onto the same id is what a refresh does; the id is keyed on
    // what the index covers, not on the snapshot, so it replaces in place.
    let id = quarry::engine::vector_index_id(
        &TableId("docs".into()),
        &quarry::derived::Nearest {
            field: EMBEDDING,
            metric: Metric::L2,
            dimension: DIM as u32,
            k: 3,
        },
    );
    assert!(
        built.built.contains(&id),
        "the id names what it covers, not the k: {:?}",
        built.built
    );
}

#[tokio::test]
async fn an_unserved_top_k_still_reports_the_ask() {
    // What lets the optimizer propose an index: the scan itself cannot see
    // the sort, so `Session::sql` reports the ask even when nothing served.
    let table = Arc::new(table(SnapshotId(810)));
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 7"
        ))
        .await
        .expect("query");

    let report = table.last_scan().expect("scan");
    let ask = report.nearest.expect("the top-k ask was reported");
    assert_eq!(ask.nearest.field, EMBEDDING);
    assert_eq!(ask.nearest.metric, Metric::L2);
    assert_eq!(ask.nearest.dimension, DIM as u32);
    assert_eq!(ask.nearest.k, 7);
}

#[tokio::test]
async fn a_tie_breaking_sort_key_is_still_served() {
    // `ORDER BY distance, id` is the usual way to make a top-k reproducible.
    // Safe because the whole Sort node runs unchanged over identical rows.
    let table = with_index(Metric::L2, DIM as u32);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let rows = session
        .sql(&format!(
            "SELECT id FROM docs \
             ORDER BY quarry_l2_distance(embedding, {ORIGIN}), id LIMIT 3"
        ))
        .await
        .expect("query");

    assert_eq!(ids(&rows), vec![1, 2, 3]);
    assert!(table.last_scan().expect("scan").substituted);
}

#[tokio::test]
async fn repeated_top_k_searches_make_the_optimizer_build_the_index() {
    // The whole loop for the vector shape: observe, propose, build, serve.
    let shared = quarry::engine::shared(Registry::new());
    let table = Arc::new(table(SnapshotId(810)).with_shared_registry(Arc::clone(&shared)));
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let mut optimizer = quarry::engine::Optimizer::new(
        Arc::clone(&shared),
        quarry::workload::Policy::automatic(1_000_000_000).with_min_queries(2),
    )
    .for_reader(POLICY);

    let sql =
        format!("SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT 3");

    // Two unaided searches. Each reports its ask even though nothing served.
    for _ in 0..2 {
        session.sql(&sql).await.expect("query");
        let report = table.last_scan().expect("scan");
        assert!(!report.substituted, "nothing to serve it yet");
        optimizer.observe(report.observation(report.bytes_if_full_scan));
    }

    let round = optimizer.round(&session, &table).await;
    assert_eq!(
        round.built.len(),
        1,
        "the repeated top-k earned an index: {round:?}"
    );

    // And now it serves, with the same answer.
    let rows = session.sql(&sql).await.expect("query");
    assert_eq!(ids(&rows), vec![1, 2, 3]);
    let report = table.last_scan().expect("scan");
    assert!(report.substituted, "the built index served the next search");
    assert!(report.files_read.is_empty());
}

#[tokio::test]
async fn one_index_serves_every_k() {
    // A search for the nearest 2 and one for the nearest 4 are the same
    // proposal: k is not part of what an index covers.
    let shared = quarry::engine::shared(Registry::new());
    let table = Arc::new(table(SnapshotId(810)).with_shared_registry(Arc::clone(&shared)));
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    let mut optimizer = quarry::engine::Optimizer::new(
        Arc::clone(&shared),
        quarry::workload::Policy::automatic(1_000_000_000).with_min_queries(2),
    )
    .for_reader(POLICY);

    for k in [2, 4] {
        session
            .sql(&format!(
                "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT {k}"
            ))
            .await
            .expect("query");
        let report = table.last_scan().expect("scan");
        optimizer.observe(report.observation(report.bytes_if_full_scan));
    }

    let round = optimizer.round(&session, &table).await;
    assert_eq!(round.built.len(), 1, "one index, not one per k: {round:?}");

    // Both k's are served by it.
    for (k, want) in [(1usize, vec![1i64]), (5, vec![1, 2, 3, 4, 5])] {
        let rows = session
            .sql(&format!(
                "SELECT id FROM docs ORDER BY quarry_l2_distance(embedding, {ORIGIN}) LIMIT {k}"
            ))
            .await
            .expect("query");
        assert_eq!(ids(&rows), want, "k={k}");
        assert!(table.last_scan().expect("scan").substituted, "k={k}");
    }
}

#[tokio::test]
async fn a_cosine_search_is_served_by_a_cosine_index() {
    let table = with_index(Metric::Cosine, DIM as u32);
    let session = session();
    session.register("docs", Arc::clone(&table)).expect("reg");

    // Along +x: row 1 and row 4 both point exactly that way, so both sit at
    // distance zero and the rest at one.
    let query = "arrow_cast(make_array(1.0, 0.0, 0.0), 'FixedSizeList(3, Float32)')";
    let rows = session
        .sql(&format!(
            "SELECT id FROM docs ORDER BY quarry_cosine_distance(embedding, {query}), id LIMIT 2"
        ))
        .await
        .expect("query");

    assert_eq!(ids(&rows), vec![1, 4], "both point along +x");
    assert!(table.last_scan().expect("scan").substituted);
}
