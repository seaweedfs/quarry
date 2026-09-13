//! Measuring the things the optimizer decides with.
//!
//! Every test in this repository runs on three or four files and a couple of
//! thousand rows. The loop makes *cost-based* decisions — what to build, what
//! to keep, what fits in a budget — from constants that were chosen by
//! reasoning rather than measurement. This runs the same machinery on a table
//! large enough for the numbers to mean something, and prints them.
//!
//! ```sh
//! cargo run --release --features engine --example measure
//! cargo run --release --features engine --example measure -- 40 200000 20000
//! ```
//!
//! # What it is looking for
//!
//! ```text
//! index size        the budget is enforced against it, so an estimate that
//!                   is wrong by 10x makes `optimize_budget_pct` meaningless
//! build cost        one ScalarValue per row; viable or not at millions
//! realized pruning  the proposer reports a CEILING that assumes perfect
//!                   selectivity. How far below it does reality land?
//! clustering        an index over data with no locality prunes nothing,
//!                   which the cost model currently has no way to know
//! ```
//!
//! Release mode matters: a debug build measures the wrong thing entirely.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::parquet::arrow::ArrowWriter;
use object_store::local::LocalFileSystem;
use url::Url;

use quarry::cost::PriceTable;
use quarry::derived::{Derived, PolicyFingerprint, Source};
use quarry::engine::{Quarry, QuarryTable, build_index, index_id};
use quarry::kinds::Index;
use quarry::registry::Registry;
use quarry::snapshot::{FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};
use quarry::workload::{Policy, Workload};

const TENANT_FIELD: u32 = 4;
const POLICY: PolicyFingerprint = PolicyFingerprint(1);

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("tenant_id", DataType::Int64, false),
        Field::new("ts", DataType::Int64, false),
        Field::new("message", DataType::Utf8, false),
    ]))
}

fn field_ids() -> BTreeMap<String, u32> {
    BTreeMap::from([
        ("tenant_id".to_owned(), TENANT_FIELD),
        ("ts".to_owned(), 5),
        ("message".to_owned(), 7),
    ])
}

/// How tenants are distributed across files.
#[derive(Clone, Copy, PartialEq)]
enum Layout {
    /// Each file holds a contiguous range of tenants, as time-ordered
    /// ingestion with tenant locality produces.
    Clustered,
    /// Every file holds a sample of every tenant, which is what happens when
    /// writes are interleaved. An index still answers correctly and prunes
    /// almost nothing.
    Scattered,
    /// Cardinality far above the row count, assigned at random: any one value
    /// lives in one or two files, but per-file min/max ranges cover
    /// everything, so Parquet's own statistics cannot prune.
    ///
    /// This is the case a file-level index exists for, and leaving it out of
    /// the measurement would have flattered the two above.
    Selective,
}

/// A cheap deterministic generator, so runs are comparable.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state >> 33
}

fn write_file(
    dir: &Path,
    index: usize,
    rows: usize,
    tenants: u64,
    files: usize,
    layout: Layout,
    state: &mut u64,
) -> (FileId, u64) {
    let path = dir.join(format!("part-{index:05}.parquet"));
    let file = fs::File::create(&path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema(), None).expect("writer");

    let per_file = (tenants / files as u64).max(1);
    let low = index as u64 * per_file;

    let mut tenant_ids = Vec::with_capacity(rows);
    let mut timestamps = Vec::with_capacity(rows);
    let mut messages = Vec::with_capacity(rows);
    for row in 0..rows {
        let tenant = match layout {
            Layout::Clustered => low + next(state) % per_file,
            Layout::Scattered => next(state) % tenants,
            // Spread over a space much larger than the number of rows, so a
            // given value is rare, while neighbouring rows are unrelated so
            // every file's range spans the whole space.
            Layout::Selective => next(state) % (tenants * 1_000),
        };
        tenant_ids.push(tenant as i64);
        timestamps.push((index * rows + row) as i64);
        messages.push(format!(
            "event {row} for tenant {tenant} with enough payload to make the \
             message column the bulky one, as it usually is"
        ));
    }

    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(tenant_ids)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(StringArray::from(messages)),
        ],
    )
    .expect("batch");
    writer.write(&batch).expect("write");
    writer.close().expect("close");

    let size = fs::metadata(&path).expect("stat").len();
    (FileId(path.to_string_lossy().into_owned()), size)
}

struct Table {
    sizes: BTreeMap<FileId, u64>,
    graph: SnapshotGraph,
    bytes: u64,
}

fn generate(dir: &Path, files: usize, rows: usize, tenants: u64, layout: Layout) -> Table {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).expect("dir");

    let mut state = 0x2545F4914F6CDD1D;
    let mut sizes = BTreeMap::new();
    let mut snapshot = Snapshot::root(SnapshotId(1));
    for index in 0..files {
        let (file, size) = write_file(dir, index, rows, tenants, files, layout, &mut state);
        snapshot = snapshot.with_clean_file(file.clone());
        sizes.insert(file, size);
    }

    Table {
        bytes: sizes.values().sum(),
        sizes,
        graph: SnapshotGraph::new().with(snapshot),
    }
}

fn table_of(table: &Table, registry: Registry) -> QuarryTable {
    let mut built = QuarryTable::new(
        schema(),
        TableId("events".into()),
        SnapshotId(1),
        table.graph.clone(),
        field_ids(),
    )
    .with_policy(POLICY)
    .with_registry(registry)
    .on_object_store(ObjectStoreUrl::local_filesystem());
    for (file, size) in &table.sizes {
        built = built.with_parquet_file(file.clone(), *size);
    }
    built
}

fn quarry() -> Quarry {
    Quarry::new(
        Url::parse("file://").expect("url"),
        Arc::new(LocalFileSystem::new()),
    )
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000.0
}

/// The exact number of bytes this index occupies when written down.
///
/// Mirrors the on-disk encoding: a count, then per value its hash, a file
/// count, and each path with its length. Computed here rather than taken from
/// the index so that this measurement is independent of what the code claims.
fn encoded_len(index: &Index) -> u64 {
    let mut total = 8u64;
    for (_, files) in index.postings() {
        total += 8 + 8;
        for file in files {
            total += 8 + file.0.len() as u64;
        }
    }
    total
}

fn postings_of(index: &Index) -> u64 {
    index.postings().map(|(_, files)| files.len() as u64).sum()
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let files: usize = args.first().and_then(|a| a.parse().ok()).unwrap_or(20);
    let rows: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(50_000);
    let tenants: u64 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(5_000);

    if cfg!(debug_assertions) {
        println!(
            "WARNING: debug build. Timings below measure the wrong thing; \
             re-run with --release.\n"
        );
    }

    println!(
        "{files} files x {rows} rows = {} rows, {tenants} distinct tenants\n",
        files * rows
    );

    for layout in [Layout::Clustered, Layout::Scattered, Layout::Selective] {
        let name = match layout {
            Layout::Clustered => "CLUSTERED (each file holds a tenant range)",
            Layout::Scattered => "SCATTERED (every file holds every tenant)",
            Layout::Selective => "SELECTIVE (high cardinality, ranges overlap)",
        };
        println!("=== {name}");

        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("measure")
            .join(match layout {
                Layout::Clustered => "clustered",
                Layout::Scattered => "scattered",
                Layout::Selective => "selective",
            });

        let started = Instant::now();
        let table = generate(&dir, files, rows, tenants, layout);
        println!(
            "  table            {:.1} MB in {files} files, written in {:.1}s",
            mb(table.bytes),
            started.elapsed().as_secs_f64()
        );

        // --- Building the index.
        let session = quarry().session();
        let bare = table_of(&table, Registry::new());

        let started = Instant::now();
        let index = build_index(&session, &bare, TENANT_FIELD)
            .await
            .expect("build index");
        let build_time = started.elapsed();

        let postings = postings_of(&index);
        let encoded = encoded_len(&index);
        println!(
            "  index            {} values, {postings} postings, built in {:.2}s",
            index.values(),
            build_time.as_secs_f64()
        );
        println!(
            "  index size       {:.2} MB encoded ({:.0} bytes/posting), \
             {:.2}% of the table",
            mb(encoded),
            encoded as f64 / postings.max(1) as f64,
            100.0 * encoded as f64 / table.bytes as f64
        );
        println!(
            "  what it reports  {:.2} MB  <-- what the budget is enforced against",
            mb(index.bytes_estimate())
        );
        println!(
            "  selectivity      {:.1} files per value out of {files}  \
             <-- the predictor of whether this is worth building",
            postings as f64 / index.values().max(1) as f64
        );

        // --- What it saves, on a query for one tenant.
        // A value that exists: for the selective layout most of the space is
        // empty, so read one out of the data rather than guessing.
        let tenant = a_present_tenant(&table, layout, tenants).await;
        let sql = format!("SELECT * FROM events WHERE tenant_id = {tenant}");

        let full = measure_query(&table, Registry::new(), &sql).await;
        let with_index = measure_query(&table, registry_for(index), &sql).await;

        assert_eq!(
            full.rows, with_index.rows,
            "the answer must not change: {} vs {}",
            full.rows, with_index.rows
        );

        println!(
            "  full scan        {:.1} MB fetched, {} files, {:.2}s, {} rows",
            mb(full.bytes),
            full.files,
            full.seconds,
            full.rows
        );
        println!(
            "  with the index   {:.1} MB fetched, {} files, {:.2}s",
            mb(with_index.bytes),
            with_index.files,
            with_index.seconds
        );

        let saved = full.bytes.saturating_sub(with_index.bytes);
        println!(
            "  realized saving  {:.1} MB ({:.1}% of what a full scan fetched)",
            mb(saved),
            100.0 * saved as f64 / full.bytes.max(1) as f64
        );

        // --- What the proposer would have promised.
        let prices = PriceTable::default();
        let mut workload = Workload::new();
        for _ in 0..Policy::ADVISORY.min_queries {
            workload.observe(full.observation.clone());
        }
        if let Some(proposal) = workload.proposals(&prices, 1).first() {
            let realized_usd = saved as f64
                * prices.byte_usd(quarry::cost::Tier::Hot, quarry::place::Distance::Far)
                * Policy::ADVISORY.min_queries as f64;
            if realized_usd > 0.0 {
                println!(
                    "  proposer ceiling ${:.6} vs ${:.6} realized  ({:.0}x optimistic)",
                    proposal.ceiling_usd,
                    realized_usd,
                    proposal.ceiling_usd / realized_usd
                );
            } else {
                println!(
                    "  proposer ceiling ${:.6} vs $0 realized  (unboundedly optimistic)",
                    proposal.ceiling_usd
                );
            }
        }
        println!();
    }
}

struct Measured {
    bytes: u64,
    files: usize,
    seconds: f64,
    rows: usize,
    observation: quarry::workload::Observation,
}

fn registry_for(index: Index) -> Registry {
    let bytes = index.bytes_estimate();
    let mut registry = Registry::new();
    registry.register(Derived::new(
        index_id(&TableId("events".into()), TENANT_FIELD),
        Source {
            table: TableId("events".into()),
            snapshot: SnapshotId(1),
        },
        POLICY,
        bytes,
        Box::new(index),
    ));
    registry
}

async fn measure_query(table: &Table, registry: Registry, sql: &str) -> Measured {
    // A fresh Quarry, so `origin_stats` counts this query and nothing else.
    let quarry = quarry();
    let served = Arc::new(table_of(table, registry));

    let session = quarry.session();
    session
        .register("events", Arc::clone(&served))
        .expect("register");

    let started = Instant::now();
    let batches = session.sql(sql).await.expect("query");
    let seconds = started.elapsed().as_secs_f64();

    let report = served.last_scan().expect("scan");
    Measured {
        bytes: quarry.origin_stats().bytes_fetched,
        files: report.files_read.len(),
        seconds,
        rows: batches.iter().map(RecordBatch::num_rows).sum(),
        observation: report.observation(report.bytes_planned(&table.sizes)),
    }
}

/// A tenant id that actually occurs, read out of the first file.
///
/// The selective layout leaves most of its value space empty, so a guessed id
/// would usually match nothing and the measurement would compare two scans
/// that both return zero rows.
async fn a_present_tenant(table: &Table, layout: Layout, tenants: u64) -> i64 {
    if layout != Layout::Selective {
        return (tenants / 2) as i64;
    }
    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::new(table_of(table, Registry::new())))
        .expect("register");
    let batches = session
        .sql("SELECT tenant_id FROM events LIMIT 1")
        .await
        .expect("query");
    batches
        .first()
        .and_then(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(|column| column.value(0))
        })
        .expect("at least one row")
}
