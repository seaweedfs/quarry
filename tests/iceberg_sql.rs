//! SQL over a real Iceberg table, planned by the rule.
//!
//! Everything here is genuine: Iceberg metadata JSON, Avro manifest lists and
//! manifests written by `iceberg-rust`, and Parquet data files those manifests
//! point at. The query goes through DataFusion, and which objects it opens is
//! decided by `Derived::may_serve`.
//!
//! ```sh
//! cargo test --features engine,iceberg --test iceberg_sql
//! ```

#![cfg(all(feature = "engine", feature = "iceberg"))]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::parquet::arrow::ArrowWriter;
use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, ManifestListWriter, ManifestWriterBuilder,
    Struct, TableMetadata,
};
use object_store::local::LocalFileSystem;
use url::Url;

use quarry::derived::{Derived, DerivedId, PolicyFingerprint, Source};
use quarry::engine::{Quarry, hash_scalar, table_from_iceberg};
use quarry::kinds::Index;
use quarry::registry::Registry;
use quarry::snapshot::{FileId, SnapshotId, TableId};

const TENANT_FIELD: u32 = 1;
const POLICY: PolicyFingerprint = PolicyFingerprint(1);
const TABLE_UUID: &str = "9c12d441-03fe-4693-9a96-a0705ddf69c1";

fn arrow_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("tenant_id", DataType::Int64, false),
        Field::new("message", DataType::Utf8, false),
    ]))
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("metadata")).expect("metadata dir");
    fs::create_dir_all(dir.join("data")).expect("data dir");
    dir
}

/// Write a real Parquet data file, returning its absolute path and size.
fn write_parquet(dir: &Path, name: &str, rows: &[(i64, &str)]) -> (String, u64) {
    let path = dir.join("data").join(name);
    let file = fs::File::create(&path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, arrow_schema(), None).expect("writer");

    let tenants: Int64Array = rows.iter().map(|(t, _)| *t).collect();
    let messages: StringArray = rows.iter().map(|(_, m)| Some(*m)).collect();
    let batch = RecordBatch::try_new(arrow_schema(), vec![Arc::new(tenants), Arc::new(messages)])
        .expect("batch");
    writer.write(&batch).expect("write");
    writer.close().expect("close");

    let size = fs::metadata(&path).expect("stat").len();
    (path.to_string_lossy().into_owned(), size)
}

fn metadata_json(dir: &Path) -> String {
    let location = dir.to_string_lossy();
    format!(
        r#"{{
            "format-version": 2,
            "table-uuid": "{TABLE_UUID}",
            "location": "{location}",
            "last-sequence-number": 1,
            "last-updated-ms": 1,
            "last-column-id": 2,
            "current-schema-id": 0,
            "schemas": [{{
                "type": "struct",
                "schema-id": 0,
                "fields": [
                    {{"id": 1, "name": "tenant_id", "required": true, "type": "long"}},
                    {{"id": 2, "name": "message", "required": true, "type": "string"}}
                ]
            }}],
            "default-spec-id": 0,
            "partition-specs": [{{"spec-id": 0, "fields": []}}],
            "last-partition-id": 999,
            "default-sort-order-id": 0,
            "sort-orders": [{{"order-id": 0, "fields": []}}],
            "properties": {{}},
            "current-snapshot-id": 1,
            "snapshots": [{{
                "snapshot-id": 1,
                "sequence-number": 1,
                "timestamp-ms": 1,
                "summary": {{"operation": "append"}},
                "manifest-list": "{location}/metadata/list-1.avro",
                "schema-id": 0
            }}],
            "snapshot-log": [],
            "metadata-log": [],
            "refs": {{"main": {{"snapshot-id": 1, "type": "branch"}}}}
        }}"#
    )
}

/// A table of three real Parquet files: tenant 1 in a and c, tenant 2 in b.
async fn iceberg_table(name: &str) -> (TableMetadata, FileIO, PathBuf, [String; 3]) {
    let dir = scratch(name);
    let file_io = FileIO::from_path(dir.to_str().expect("utf-8"))
        .expect("file io")
        .build()
        .expect("build file io");

    let (a, a_size) = write_parquet(&dir, "a.parquet", &[(1, "a1"), (1, "a2")]);
    let (b, b_size) = write_parquet(&dir, "b.parquet", &[(2, "b1")]);
    let (c, c_size) = write_parquet(&dir, "c.parquet", &[(1, "c1")]);

    let metadata: TableMetadata =
        serde_json::from_str(&metadata_json(&dir)).expect("valid metadata");

    let mut writer = ManifestWriterBuilder::new(
        file_io
            .new_output(format!(
                "{}/metadata/manifest-1.avro",
                dir.to_string_lossy()
            ))
            .expect("manifest output"),
        Some(1),
        None,
        metadata.current_schema().clone(),
        metadata.default_partition_spec().as_ref().clone(),
    )
    .build_v2_data();

    for (path, size) in [(&a, a_size), (&b, b_size), (&c, c_size)] {
        writer
            .add_file(
                DataFileBuilder::default()
                    .partition_spec_id(0)
                    .content(DataContentType::Data)
                    .file_path(path.clone())
                    .file_format(DataFileFormat::Parquet)
                    .file_size_in_bytes(size)
                    .record_count(1)
                    .partition(Struct::empty())
                    .build()
                    .expect("data file"),
                1,
            )
            .expect("add file");
    }
    let manifest = writer.write_manifest_file().await.expect("write manifest");

    let mut list = ManifestListWriter::v2(
        file_io
            .new_output(format!("{}/metadata/list-1.avro", dir.to_string_lossy()))
            .expect("list output"),
        1,
        None,
        1,
    );
    list.add_manifests(std::iter::once(manifest))
        .expect("add manifest");
    list.close().await.expect("close list");

    (metadata, file_io, dir, [a, b, c])
}

fn quarry() -> Quarry {
    Quarry::new(
        Url::parse("file://").expect("url"),
        Arc::new(LocalFileSystem::new()),
    )
}

/// An index over tenant_id covering the given (tenant, file) pairs.
fn tenant_index(entries: &[(i64, &str)]) -> Registry {
    let mut index = Index::new(TENANT_FIELD);
    for (tenant, path) in entries {
        index.insert(
            hash_scalar(&datafusion::scalar::ScalarValue::Int64(Some(*tenant))),
            FileId(quarry::from_iceberg::object_path(path)),
        );
    }
    let mut registry = Registry::new();
    registry.register(Derived::new(
        DerivedId("tenant_idx".into()),
        Source {
            table: TableId(TABLE_UUID.into()),
            snapshot: SnapshotId(1),
        },
        POLICY,
        4096,
        Box::new(index.with_bytes(4096)),
    ));
    registry
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

#[tokio::test]
async fn the_schema_and_field_ids_come_from_iceberg() {
    let (metadata, file_io, _, _) = iceberg_table("iceberg_sql_schema").await;
    let table = table_from_iceberg(
        &metadata,
        &file_io,
        ObjectStoreUrl::local_filesystem(),
        None,
    )
    .await
    .expect("build table");

    let ids = quarry::engine::field_ids(&metadata);
    assert_eq!(ids.get("tenant_id"), Some(&1));
    assert_eq!(ids.get("message"), Some(&2));

    // The Arrow schema is Iceberg's, not one we declared alongside it.
    let schema = datafusion::catalog::TableProvider::schema(&table);
    assert_eq!(schema.fields().len(), 2);
    assert_eq!(schema.field(0).name(), "tenant_id");
}

#[tokio::test]
async fn sql_reads_a_real_iceberg_table() {
    let (metadata, file_io, _, _) = iceberg_table("iceberg_sql_scan").await;
    let table = table_from_iceberg(
        &metadata,
        &file_io,
        ObjectStoreUrl::local_filesystem(),
        None,
    )
    .await
    .expect("build table");

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::new(table))
        .expect("register");

    let rows = session
        .sql("SELECT * FROM events ORDER BY message")
        .await
        .expect("query");
    assert_eq!(
        total_rows(&rows),
        4,
        "every row across the three data files"
    );
    assert!(
        quarry.origin_stats().bytes_fetched > 0,
        "the Parquet objects were actually read"
    );
}

#[tokio::test]
async fn an_index_prunes_a_real_iceberg_table() {
    let (metadata, file_io, _, [a, _b, c]) = iceberg_table("iceberg_sql_prune").await;

    let table = table_from_iceberg(
        &metadata,
        &file_io,
        ObjectStoreUrl::local_filesystem(),
        None,
    )
    .await
    .expect("build table")
    .with_registry(tenant_index(&[(1, &a), (1, &c)]))
    .with_policy(POLICY);
    let table = Arc::new(table);

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::clone(&table))
        .expect("register");

    let rows = session
        .sql("SELECT * FROM events WHERE tenant_id = 1")
        .await
        .expect("query");
    assert_eq!(total_rows(&rows), 3, "same answer as a full scan");

    let report = table.last_scan().expect("a scan happened");
    assert_eq!(report.used.as_deref(), Some("tenant_idx"));
    assert_eq!(
        report.files_read,
        BTreeSet::from([
            FileId(quarry::from_iceberg::object_path(&a)),
            FileId(quarry::from_iceberg::object_path(&c)),
        ]),
        "the file holding only tenant 2 must not be read"
    );
}

#[tokio::test]
async fn a_pruned_iceberg_object_is_never_opened() {
    // The strongest available proof: delete the excluded object. If the plan
    // reached for it, the query would fail rather than merely be slower.
    let (metadata, file_io, _, [a, b, c]) = iceberg_table("iceberg_sql_deleted").await;

    let table = table_from_iceberg(
        &metadata,
        &file_io,
        ObjectStoreUrl::local_filesystem(),
        None,
    )
    .await
    .expect("build table")
    .with_registry(tenant_index(&[(1, &a), (1, &c)]))
    .with_policy(POLICY);

    fs::remove_file(&b).expect("remove the excluded object");

    let quarry = quarry();
    let session = quarry.session();
    session
        .register("events", Arc::new(table))
        .expect("register");

    let rows = session
        .sql("SELECT * FROM events WHERE tenant_id = 1")
        .await
        .expect("query must not depend on the excluded object");
    assert_eq!(total_rows(&rows), 3);
}

#[tokio::test]
async fn aggregation_over_a_pruned_iceberg_table_is_correct() {
    let (metadata, file_io, _, [a, _b, c]) = iceberg_table("iceberg_sql_aggregate").await;

    let table = table_from_iceberg(
        &metadata,
        &file_io,
        ObjectStoreUrl::local_filesystem(),
        None,
    )
    .await
    .expect("build table")
    .with_registry(tenant_index(&[(1, &a), (1, &c)]))
    .with_policy(POLICY);

    let session = quarry().session();
    session
        .register("events", Arc::new(table))
        .expect("register");

    let rows = session
        .sql("SELECT count(*) AS n FROM events WHERE tenant_id = 1")
        .await
        .expect("query");
    let n = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("int64")
        .value(0);
    assert_eq!(n, 3);
}

#[tokio::test]
async fn a_repeated_iceberg_query_stops_touching_the_store() {
    let (metadata, file_io, _, _) = iceberg_table("iceberg_sql_cached").await;
    let sql = "SELECT * FROM events";

    let quarry = Quarry::with_cache(
        Url::parse("file://").expect("url"),
        Arc::new(LocalFileSystem::new()),
        16 << 20,
    );

    for run in 0..2 {
        let table = table_from_iceberg(
            &metadata,
            &file_io,
            ObjectStoreUrl::local_filesystem(),
            None,
        )
        .await
        .expect("build table");

        let session = quarry.session();
        session
            .register("events", Arc::new(table))
            .expect("register");
        assert_eq!(total_rows(&session.sql(sql).await.expect("query")), 4);

        if run == 0 {
            assert!(quarry.origin_stats().bytes_fetched > 0);
        }
    }

    assert!(
        quarry.cache_stats().expect("a cache").hits > 0,
        "the second run should have been served from cache"
    );
}
