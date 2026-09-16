//! Full-text search: a `quarry_matches` predicate pruned by an inverted
//! index.
//!
//! Unlike a cube or a top-k, a match is a *filter*, so it is pushed into the
//! scan and needs no plan-level rewrite. These tests go through
//! `Session::sql` anyway, because that is where the predicate function is
//! registered, and they read real Parquet through a metered store so the
//! saving is a measurement rather than a claim about the plan.
//!
//! ```sh
//! cargo test --features engine --test fts
//! ```

#![cfg(feature = "engine")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::parquet::arrow::ArrowWriter;
use object_store::local::LocalFileSystem;
use url::Url;

use quarry::derived::{Derived, DerivedId, PolicyFingerprint, Source};
use quarry::engine::{Optimizer, Quarry, QuarryTable, Session, shared, terms, text_index_id};
use quarry::kinds::TextIndex;
use quarry::registry::Registry;
use quarry::snapshot::{FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};
use quarry::workload::Policy;

const ID: u32 = 1;
const BODY: u32 = 2;
const POLICY: PolicyFingerprint = PolicyFingerprint(1);

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("body", DataType::Utf8, true),
    ]))
}

fn field_ids() -> BTreeMap<String, u32> {
    BTreeMap::from([("id".to_owned(), ID), ("body".to_owned(), BODY)])
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// A file whose rows all contain `word`, plus padding shared by every file.
///
/// Padding matters twice: it makes the file big enough that skipping it is
/// worth measuring, and it gives the index a term that prunes *nothing*, so
/// the tests distinguish a real narrowing from an accidental one.
fn write_parquet(dir: &Path, name: &str, first_id: i64, word: &str, rows: usize) -> (FileId, u64) {
    let path = dir.join(name);
    let file = fs::File::create(&path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema(), None).expect("writer");

    let ids: Int64Array = (0..rows).map(|i| first_id + i as i64).collect();
    let bodies: StringArray = (0..rows)
        .map(|i| {
            Some(format!(
                "{word} shared padding to take up room, line {i} of the log"
            ))
        })
        .collect();
    let batch = RecordBatch::try_new(schema(), vec![Arc::new(ids), Arc::new(bodies)])
        .expect("batch matches schema");
    writer.write(&batch).expect("write");
    writer.close().expect("close");

    let size = fs::metadata(&path).expect("stat").len();
    (FileId(path.to_string_lossy().into_owned()), size)
}

struct Fixture {
    sizes: BTreeMap<FileId, u64>,
    graph: SnapshotGraph,
    files: [FileId; 3],
}

/// Three files, each with a distinguishing word: alpha, beta, gamma.
fn fixture(name: &str) -> Fixture {
    let dir = scratch(name);
    let (a, a_size) = write_parquet(&dir, "a.parquet", 0, "alpha", 400);
    let (b, b_size) = write_parquet(&dir, "b.parquet", 400, "beta", 400);
    let (c, c_size) = write_parquet(&dir, "c.parquet", 800, "gamma", 400);

    let graph = SnapshotGraph::new().with(
        Snapshot::root(SnapshotId(810))
            .with_clean_file(a.clone())
            .with_clean_file(b.clone())
            .with_clean_file(c.clone()),
    );

    Fixture {
        sizes: BTreeMap::from([
            (a.clone(), a_size),
            (b.clone(), b_size),
            (c.clone(), c_size),
        ]),
        graph,
        files: [a, b, c],
    }
}

fn table(fixture: &Fixture, registry: Registry) -> QuarryTable {
    let mut table = QuarryTable::new(
        schema(),
        TableId("logs".into()),
        SnapshotId(810),
        fixture.graph.clone(),
        field_ids(),
    )
    .with_policy(POLICY)
    .on_object_store(
        datafusion::datasource::object_store::ObjectStoreUrl::parse("file://").expect("url"),
    )
    .with_registry(registry);
    for (file, size) in &fixture.sizes {
        table = table.with_parquet_file(file.clone(), *size);
    }
    table
}

/// A metered session over the local filesystem, with the predicate
/// registered.
fn session() -> (Quarry, Session) {
    let quarry = Quarry::new(
        Url::parse("file://").expect("url"),
        Arc::new(LocalFileSystem::new()),
    );
    let session = quarry.session();
    (quarry, session)
}

/// A text index built by hand, so a probe can be tested without a build.
fn text_index(fixture: &Fixture, entries: &[(&str, usize)]) -> Derived {
    let mut index = TextIndex::new(BODY);
    for (word, which) in entries {
        for term in terms(word) {
            index.insert(term, fixture.files[*which].clone());
        }
    }
    let bytes = index.encoded_len();
    Derived::new(
        DerivedId("body_txt".into()),
        Source {
            table: TableId("logs".into()),
            snapshot: SnapshotId(810),
        },
        POLICY,
        bytes,
        Box::new(index.with_bytes(bytes)),
    )
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

#[tokio::test]
async fn a_match_prunes_to_the_file_holding_the_term() {
    let fixture = fixture("fts_prune");
    let mut registry = Registry::new();
    // Every file holds "shared"; only b holds "beta".
    registry.register(text_index(
        &fixture,
        &[
            ("alpha", 0),
            ("beta", 1),
            ("gamma", 2),
            ("shared", 0),
            ("shared", 1),
            ("shared", 2),
        ],
    ));
    let table = Arc::new(table(&fixture, registry));

    let (_quarry, session) = session();
    session.register("logs", Arc::clone(&table)).expect("reg");
    let rows = session
        .sql("SELECT id FROM logs WHERE quarry_matches(body, 'beta')")
        .await
        .expect("query");

    assert_eq!(total_rows(&rows), 400, "every row of b matches");
    let report = table.last_scan().expect("a scan happened");
    assert_eq!(
        report.files_read,
        BTreeSet::from([fixture.files[1].clone()]),
        "only the file holding the term is read"
    );
    assert_eq!(report.used, vec!["body_txt".to_owned()]);
}

#[tokio::test]
async fn a_term_every_file_holds_prunes_nothing() {
    // The control for the test above: the same index, a term with postings
    // everywhere, and all three files read. Without this, a pruning result
    // could be an accident of the fixture rather than the index working.
    let fixture = fixture("fts_no_prune");
    let mut registry = Registry::new();
    registry.register(text_index(
        &fixture,
        &[("shared", 0), ("shared", 1), ("shared", 2)],
    ));
    let table = Arc::new(table(&fixture, registry));

    let (_quarry, session) = session();
    session.register("logs", Arc::clone(&table)).expect("reg");
    let rows = session
        .sql("SELECT id FROM logs WHERE quarry_matches(body, 'shared')")
        .await
        .expect("query");

    assert_eq!(total_rows(&rows), 1200, "every row matches");
    assert_eq!(
        table.last_scan().expect("scan").files_read.len(),
        3,
        "nothing to narrow"
    );
}

#[tokio::test]
async fn several_terms_intersect_so_an_impossible_conjunction_reads_nothing() {
    let fixture = fixture("fts_intersect");
    let mut registry = Registry::new();
    registry.register(text_index(&fixture, &[("alpha", 0), ("beta", 1)]));
    let table = Arc::new(table(&fixture, registry));

    let (quarry, session) = session();
    session.register("logs", Arc::clone(&table)).expect("reg");
    let rows = session
        .sql("SELECT id FROM logs WHERE quarry_matches(body, 'alpha beta')")
        .await
        .expect("query");

    assert_eq!(
        total_rows(&rows),
        0,
        "no row holds both words, and the answer says so"
    );
    let report = table.last_scan().expect("scan");
    assert!(
        report.files_read.is_empty(),
        "the intersection is empty, so no file can match: {:?}",
        report.files_read
    );
    assert_eq!(
        quarry.origin_stats().bytes_fetched,
        0,
        "and provably nothing was fetched"
    );
}

#[tokio::test]
async fn the_answer_matches_the_unaided_query() {
    // The property that matters: pruning must not change the answer.
    let fixture = fixture("fts_same_answer");
    let mut registry = Registry::new();
    registry.register(text_index(
        &fixture,
        &[("alpha", 0), ("beta", 1), ("gamma", 2)],
    ));
    let served = Arc::new(table(&fixture, registry));
    let unaided = Arc::new(table(&fixture, Registry::new()));

    let sql = "SELECT count(*) AS n FROM logs WHERE quarry_matches(body, 'gamma')";

    let (_q1, one) = session();
    one.register("logs", Arc::clone(&served)).expect("reg");
    let with = one.sql(sql).await.expect("query");

    let (_q2, other) = session();
    other.register("logs", Arc::clone(&unaided)).expect("reg");
    let without = other.sql(sql).await.expect("query");

    assert_eq!(format!("{with:?}"), format!("{without:?}"));
    assert_eq!(served.last_scan().expect("scan").files_read.len(), 1);
    assert_eq!(unaided.last_scan().expect("scan").files_read.len(), 3);
}

#[tokio::test]
async fn a_prefix_is_not_a_term() {
    // `quarry_matches` is not a LIKE. The index cannot answer a substring,
    // so the predicate must not claim to either — and the rows returned are
    // what proves it, not the plan.
    let fixture = fixture("fts_prefix");
    let table = Arc::new(table(&fixture, Registry::new()));

    let (_quarry, session) = session();
    session.register("logs", Arc::clone(&table)).expect("reg");
    let rows = session
        .sql("SELECT id FROM logs WHERE quarry_matches(body, 'alph')")
        .await
        .expect("query");

    assert_eq!(total_rows(&rows), 0, "'alph' is not the word 'alpha'");
}

#[tokio::test]
async fn matching_ignores_case_and_punctuation() {
    let fixture = fixture("fts_case");
    let table = Arc::new(table(&fixture, Registry::new()));

    let (_quarry, session) = session();
    session.register("logs", Arc::clone(&table)).expect("reg");
    let rows = session
        .sql("SELECT id FROM logs WHERE quarry_matches(body, 'ALPHA!')")
        .await
        .expect("query");

    assert_eq!(total_rows(&rows), 400);
}

#[tokio::test]
async fn a_stale_index_reads_the_files_added_since() {
    // The pruning staleness rule, for this kind: an index that has never
    // seen a file must not cause its rows to be missed.
    let mut fixture = fixture("fts_stale");
    let mut registry = Registry::new();
    registry.register(text_index(&fixture, &[("alpha", 0)]));

    // A fourth file, holding "alpha" too, committed after the index was built.
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fts_stale");
    let (d, d_size) = write_parquet(&dir, "d.parquet", 1200, "alpha", 100);
    fixture.sizes.insert(d.clone(), d_size);
    let mut next = Snapshot::child_of(SnapshotId(811), SnapshotId(810));
    for file in fixture.sizes.keys() {
        next = next.with_clean_file(file.clone());
    }
    fixture.graph.insert(next);

    let mut table = QuarryTable::new(
        schema(),
        TableId("logs".into()),
        SnapshotId(811),
        fixture.graph.clone(),
        field_ids(),
    )
    .with_policy(POLICY)
    .on_object_store(
        datafusion::datasource::object_store::ObjectStoreUrl::parse("file://").expect("url"),
    )
    .with_registry(registry);
    for (file, size) in &fixture.sizes {
        table = table.with_parquet_file(file.clone(), *size);
    }
    let table = Arc::new(table);

    let (_quarry, session) = session();
    session.register("logs", Arc::clone(&table)).expect("reg");
    let rows = session
        .sql("SELECT id FROM logs WHERE quarry_matches(body, 'alpha')")
        .await
        .expect("query");

    assert_eq!(
        total_rows(&rows),
        500,
        "400 from the indexed file and 100 from the one it never saw"
    );
    let report = table.last_scan().expect("scan");
    assert!(report.also_scanned.contains(&d), "d was unioned back in");
}

#[tokio::test]
async fn repeated_matches_make_the_optimizer_build_the_index() {
    let fixture = fixture("fts_loop");
    let registry = shared(Registry::new());
    let table =
        Arc::new(table(&fixture, Registry::new()).with_shared_registry(Arc::clone(&registry)));

    let (_quarry, session) = session();
    session.register("logs", Arc::clone(&table)).expect("reg");

    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        Policy::automatic(1_000_000_000).with_min_queries(2),
    )
    .for_reader(POLICY);

    let sql = "SELECT id FROM logs WHERE quarry_matches(body, 'beta')";
    for _ in 0..2 {
        session.sql(sql).await.expect("query");
        let report = table.last_scan().expect("scan");
        assert!(report.used.is_empty(), "nothing to serve it yet");
        optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
    }

    let round = optimizer.build_recommended(&session, &table).await;
    assert!(
        round
            .built
            .contains(&text_index_id(&TableId("logs".into()), BODY)),
        "the repeated match earned a text index: {round:?}"
    );

    // And it prunes on the next query, with the same answer.
    let rows = session.sql(sql).await.expect("query");
    assert_eq!(total_rows(&rows), 400);
    let report = table.last_scan().expect("scan");
    assert_eq!(
        report.files_read,
        BTreeSet::from([fixture.files[1].clone()]),
        "the built index narrowed the scan to b"
    );
}

#[tokio::test]
async fn a_repeated_match_does_not_earn_an_equality_index() {
    // A match is not an equality, and a scalar index could never serve one.
    // Keeping matches out of `Fingerprint::probeable` is what prevents the
    // optimizer from spending a build learning that.
    let fixture = fixture("fts_not_equality");
    let registry = shared(Registry::new());
    let table =
        Arc::new(table(&fixture, Registry::new()).with_shared_registry(Arc::clone(&registry)));

    let (_quarry, session) = session();
    session.register("logs", Arc::clone(&table)).expect("reg");

    let mut optimizer = Optimizer::new(
        Arc::clone(&registry),
        Policy::automatic(1_000_000_000).with_min_queries(2),
    )
    .for_reader(POLICY);

    for _ in 0..3 {
        session
            .sql("SELECT id FROM logs WHERE quarry_matches(body, 'beta')")
            .await
            .expect("query");
        let report = table.last_scan().expect("scan");
        optimizer.observe(report.observation(report.bytes_planned(&fixture.sizes)));
    }

    assert!(
        optimizer.proposals().is_empty(),
        "no equality-index proposal: {:?}",
        optimizer.proposals()
    );
}
