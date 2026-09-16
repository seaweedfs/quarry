//! Joins served from a cached build-side relation — the plan-level half of
//! the rule, for the `JOIN ... ON` shape.
//!
//! A scan never sees the join, so all of this goes through `Session::sql`:
//! the shape is recognised in the logical plan, the registry decides
//! whether a join hash may serve it, and the build-side `TableScan` is
//! swapped for a `MemTable` of stored rows.
//!
//! ```sh
//! cargo test --features engine --test join
//! ```

#![cfg(feature = "engine")]

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use url::Url;

use quarry::derived::{Derived, DerivedId, PolicyFingerprint, Source};
use quarry::engine::{Quarry, QuarryTable, Session};
use quarry::kinds::JoinHash;
use quarry::registry::Registry;
use quarry::snapshot::{FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};

const ORDER_ID: u32 = 1;
const CUSTOMER_ID: u32 = 2;
const CUSTOMER_NAME: u32 = 3;
const POLICY: PolicyFingerprint = PolicyFingerprint(1);

/// `orders(order_id, customer_id)` — the fact table, probe side.
fn orders_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("customer_id", DataType::Int64, false),
    ]))
}

fn orders_field_ids() -> BTreeMap<String, u32> {
    BTreeMap::from([
        ("order_id".to_owned(), ORDER_ID),
        ("customer_id".to_owned(), CUSTOMER_ID),
    ])
}

/// `customers(customer_id, name)` — the dimension table, build side.
fn customers_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("customer_id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

fn customers_field_ids() -> BTreeMap<String, u32> {
    BTreeMap::from([
        ("customer_id".to_owned(), CUSTOMER_ID),
        ("name".to_owned(), CUSTOMER_NAME),
    ])
}

/// Three orders for two customers.
fn orders_batch() -> RecordBatch {
    let order_ids = Int64Array::from(vec![100, 200, 300]);
    let customer_ids = Int64Array::from(vec![1, 2, 1]);
    RecordBatch::try_new(
        orders_schema(),
        vec![Arc::new(order_ids), Arc::new(customer_ids)],
    )
    .expect("orders batch")
}

/// Two customers.
fn customers_batch() -> RecordBatch {
    let customer_ids = Int64Array::from(vec![1, 2]);
    let names = StringArray::from(vec!["Alice", "Bob"]);
    RecordBatch::try_new(
        customers_schema(),
        vec![Arc::new(customer_ids), Arc::new(names)],
    )
    .expect("customers batch")
}

fn graph(at: SnapshotId) -> SnapshotGraph {
    SnapshotGraph::new().with(Snapshot::root(at).with_clean_file(FileId("a".into())))
}

fn orders_table(at: SnapshotId) -> QuarryTable {
    QuarryTable::new(
        orders_schema(),
        TableId("orders".into()),
        at,
        graph(at),
        orders_field_ids(),
    )
    .with_policy(POLICY)
    .with_file(FileId("a".into()), vec![orders_batch()])
}

fn customers_table(at: SnapshotId) -> QuarryTable {
    QuarryTable::new(
        customers_schema(),
        TableId("customers".into()),
        at,
        graph(at),
        customers_field_ids(),
    )
    .with_policy(POLICY)
    .with_file(FileId("a".into()), vec![customers_batch()])
}

/// A join hash over `customers.customer_id`, covering `customer_id` and
/// `name`, with the table's rows stored beside it.
fn join_hash(id: &str, at: SnapshotId) -> Derived {
    Derived::new(
        DerivedId(id.to_owned()),
        Source {
            table: TableId("customers".into()),
            snapshot: at,
        },
        POLICY,
        4096,
        Box::new(JoinHash::covering(
            [CUSTOMER_ID],
            [CUSTOMER_ID, CUSTOMER_NAME],
            4096,
        )),
    )
}

fn session() -> Session {
    Quarry::new(
        Url::parse("memory://").expect("url"),
        Arc::new(object_store::memory::InMemory::new()),
    )
    .session()
}

/// A customers table with a usable join hash, rows supplied.
fn with_hash() -> Arc<QuarryTable> {
    let mut registry = Registry::new();
    registry.register(join_hash("jh", SnapshotId(810)));
    Arc::new(
        customers_table(SnapshotId(810))
            .with_registry(registry)
            .with_projection(DerivedId("jh".into()), vec![customers_batch()]),
    )
}

/// The `(order_id, customer_name)` pairs of an answer, in the order they
/// came back.
fn pairs(batches: &[RecordBatch]) -> Vec<(i64, String)> {
    let mut out = Vec::new();
    for batch in batches {
        let order_ids = batch
            .column_by_name("order_id")
            .expect("order_id")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64");
        let names = batch
            .column_by_name("name")
            .expect("name")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("utf8");
        for row in 0..batch.num_rows() {
            out.push((order_ids.value(row), names.value(row).to_owned()));
        }
    }
    out
}

#[tokio::test]
async fn a_join_is_served_by_a_join_hash() {
    let customers = with_hash();
    let orders = Arc::new(orders_table(SnapshotId(810)));
    let session = session();
    session
        .register("orders", Arc::clone(&orders))
        .expect("reg");
    session
        .register("customers", Arc::clone(&customers))
        .expect("reg");

    let rows = session
        .sql(
            "SELECT o.order_id, c.name \
             FROM customers AS c \
             INNER JOIN orders AS o ON c.customer_id = o.customer_id \
             ORDER BY o.order_id",
        )
        .await
        .expect("query");

    assert_eq!(
        pairs(&rows),
        vec![
            (100, "Alice".to_owned()),
            (200, "Bob".to_owned()),
            (300, "Alice".to_owned()),
        ],
        "the join result is correct"
    );

    let report = customers.last_scan().expect("a scan happened");
    assert!(report.substituted, "the join hash supplied the rows");
    assert_eq!(report.used, vec!["jh".to_owned()]);
    assert!(
        report.files_read.is_empty(),
        "no files read — the rows came from the hash"
    );
}

#[tokio::test]
async fn a_join_without_a_hash_falls_back_to_the_table() {
    let customers = Arc::new(customers_table(SnapshotId(810)));
    let orders = Arc::new(orders_table(SnapshotId(810)));
    let session = session();
    session
        .register("orders", Arc::clone(&orders))
        .expect("reg");
    session
        .register("customers", Arc::clone(&customers))
        .expect("reg");

    let rows = session
        .sql(
            "SELECT o.order_id, c.name \
             FROM customers AS c \
             INNER JOIN orders AS o ON c.customer_id = o.customer_id \
             ORDER BY o.order_id",
        )
        .await
        .expect("query");

    assert_eq!(
        pairs(&rows),
        vec![
            (100, "Alice".to_owned()),
            (200, "Bob".to_owned()),
            (300, "Alice".to_owned()),
        ],
        "the join result is still correct without a hash"
    );

    let report = customers.last_scan().expect("a scan happened");
    assert!(!report.substituted, "no hash — the table was read");
}

#[tokio::test]
async fn a_join_reports_the_ask_even_without_a_hash() {
    let customers = Arc::new(customers_table(SnapshotId(810)));
    let orders = Arc::new(orders_table(SnapshotId(810)));
    let session = session();
    session
        .register("orders", Arc::clone(&orders))
        .expect("reg");
    session
        .register("customers", Arc::clone(&customers))
        .expect("reg");

    let _ = session
        .sql(
            "SELECT o.order_id, c.name \
             FROM customers AS c \
             INNER JOIN orders AS o ON c.customer_id = o.customer_id",
        )
        .await
        .expect("query");

    let report = customers.last_scan().expect("a scan happened");
    let ask = report.join.as_ref().expect("join ask reported");
    assert_eq!(ask.table, TableId("customers".into()));
    assert_eq!(ask.keys, std::collections::BTreeSet::from([CUSTOMER_ID]));
    assert_eq!(
        ask.columns,
        std::collections::BTreeSet::from([CUSTOMER_ID, CUSTOMER_NAME])
    );
}

#[tokio::test]
async fn a_right_join_is_not_served_by_a_join_hash() {
    // A right join makes the right side the build side in DataFusion's
    // physical plan, so the left-side hash does not apply.
    let customers = with_hash();
    let orders = Arc::new(orders_table(SnapshotId(810)));
    let session = session();
    session
        .register("orders", Arc::clone(&orders))
        .expect("reg");
    session
        .register("customers", Arc::clone(&customers))
        .expect("reg");

    let _ = session
        .sql(
            "SELECT o.order_id, c.name \
             FROM orders AS o \
             RIGHT JOIN customers AS c ON c.customer_id = o.customer_id",
        )
        .await
        .expect("query");

    // The customers table was the right side, so the join rewrite should
    // not have fired for it. The scan report should show no substitution.
    // (The join ask is also not reported, because the rewrite only
    // recognises inner and left joins.)
    if let Some(report) = customers.last_scan() {
        assert!(
            !report.substituted,
            "right join does not use the left-side hash"
        );
    }
}
