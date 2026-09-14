//! SQL over a table loaded from an Iceberg REST catalog.
//!
//! The catalog is real: `iceberg-catalog-rest` talking HTTP over a real socket
//! to a mock server that answers `/v1/config` and `loadTable` the way a
//! catalog does. What comes back is a genuine `Table`, and everything below it
//! — manifest lists, manifests, Parquet — is on disk.
//!
//! This is the deployment shape that matters: point at a catalog URI, name a
//! table, query it. The library depends on no catalog implementation, because
//! [`table_from_catalog`] takes the `Table` every catalog hands back.
//!
//! ```sh
//! cargo test --features rest-catalog --test iceberg_rest
//! ```

#![cfg(feature = "rest-catalog")]

use std::collections::{BTreeSet, HashMap};
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
use iceberg::{Catalog, NamespaceIdent, TableIdent};
use iceberg_catalog_rest::{RestCatalog, RestCatalogConfig};
use object_store::local::LocalFileSystem;
use url::Url;

use quarry::derived::{Derived, DerivedId, PolicyFingerprint, Source};
use quarry::engine::{Advertised, Layout, Quarry, Store, hash_scalar, table_from_catalog};
use quarry::from_iceberg::object_path;
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

/// Write a real table on disk and return its metadata JSON and data paths.
async fn table_on_disk(name: &str) -> (PathBuf, String, [String; 3]) {
    let dir = scratch(name);
    let file_io = FileIO::from_path(dir.to_str().expect("utf-8"))
        .expect("file io")
        .build()
        .expect("build file io");

    let (a, a_size) = write_parquet(&dir, "a.parquet", &[(1, "a1"), (1, "a2")]);
    let (b, b_size) = write_parquet(&dir, "b.parquet", &[(2, "b1")]);
    let (c, c_size) = write_parquet(&dir, "c.parquet", &[(1, "c1")]);

    let json = metadata_json(&dir);
    let metadata: TableMetadata = serde_json::from_str(&json).expect("valid metadata");

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

    (dir, json, [a, b, c])
}

/// A REST catalog serving one table, over HTTP on a real socket.
///
/// `/v1/config` returns no `warehouse` override on purpose: the client then
/// derives its `FileIO` from the metadata location, which is a local path here.
/// Returning an `s3://` warehouse would send it looking for S3 credentials.
async fn rest_catalog(dir: &Path, metadata: &str) -> (mockito::ServerGuard, RestCatalog) {
    let mut server = mockito::Server::new_async().await;

    server
        .mock("GET", "/v1/config")
        .with_status(200)
        .with_body(r#"{"overrides": {}, "defaults": {}}"#)
        .create_async()
        .await;

    let load_table = format!(
        r#"{{
            "metadata-location": "{}/metadata/v1.metadata.json",
            "metadata": {metadata},
            "config": {{}}
        }}"#,
        dir.to_string_lossy()
    );
    server
        .mock("GET", "/v1/namespaces/db/tables/events")
        .with_status(200)
        .with_body(load_table)
        .create_async()
        .await;

    let catalog = RestCatalog::new(RestCatalogConfig::builder().uri(server.url()).build());
    (server, catalog)
}

fn tenant_index(entries: &[(i64, &str)]) -> Registry {
    let mut index = Index::new(TENANT_FIELD);
    for (tenant, path) in entries {
        index.insert(
            hash_scalar(&datafusion::scalar::ScalarValue::Int64(Some(*tenant))),
            FileId(object_path(path)),
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

fn events() -> TableIdent {
    TableIdent::new(NamespaceIdent::new("db".to_owned()), "events".to_owned())
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

#[tokio::test]
async fn a_table_loaded_over_rest_is_queryable() {
    let (dir, metadata, _) = table_on_disk("rest_scan").await;
    let (_server, catalog) = rest_catalog(&dir, &metadata).await;

    let loaded = catalog.load_table(&events()).await.expect("load table");
    assert_eq!(loaded.identifier(), &events());

    let table = table_from_catalog(&loaded, ObjectStoreUrl::local_filesystem(), None)
        .await
        .expect("build table");

    let quarry = Quarry::new(
        Url::parse("file://").expect("url"),
        Arc::new(LocalFileSystem::new()),
    );
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
        "every row, fetched via a catalog over HTTP"
    );
}

#[tokio::test]
async fn the_rule_prunes_a_table_loaded_over_rest() {
    let (dir, metadata, [a, _b, c]) = table_on_disk("rest_prune").await;
    let (_server, catalog) = rest_catalog(&dir, &metadata).await;

    let loaded = catalog.load_table(&events()).await.expect("load table");
    let table = Arc::new(
        table_from_catalog(&loaded, ObjectStoreUrl::local_filesystem(), None)
            .await
            .expect("build table")
            .with_registry(tenant_index(&[(1, &a), (1, &c)]))
            .with_policy(POLICY),
    );

    let quarry = Quarry::new(
        Url::parse("file://").expect("url"),
        Arc::new(LocalFileSystem::new()),
    );
    let session = quarry.session();
    session
        .register("events", Arc::clone(&table))
        .expect("register");

    let rows = session
        .sql("SELECT * FROM events WHERE tenant_id = 1")
        .await
        .expect("query");
    assert_eq!(total_rows(&rows), 3);

    let report = table.last_scan().expect("a scan happened");
    assert_eq!(report.used.first().map(String::as_str), Some("tenant_idx"));
    assert_eq!(
        report.files_read,
        BTreeSet::from([FileId(object_path(&a)), FileId(object_path(&c))]),
        "the catalog path must prune exactly as the local path does"
    );
}

#[tokio::test]
async fn the_table_identity_survives_the_catalog() {
    // Derived state is keyed on the table's UUID, not the name the catalog
    // happens to serve it under, so state built earlier still matches.
    let (dir, metadata, _) = table_on_disk("rest_identity").await;
    let (_server, catalog) = rest_catalog(&dir, &metadata).await;

    let loaded = catalog.load_table(&events()).await.expect("load table");
    assert_eq!(
        quarry::from_iceberg::table_id(loaded.metadata()).0,
        TABLE_UUID
    );
    assert_eq!(
        quarry::from_iceberg::current_snapshot(loaded.metadata()),
        Some(SnapshotId(1))
    );
}

#[tokio::test]
async fn publish_commits_the_manifest_to_table_metadata() {
    let (dir, metadata, _) = table_on_disk("rest_publish").await;
    let mut server = mockito::Server::new_async().await;

    server
        .mock("GET", "/v1/config")
        .with_status(200)
        .with_body(r#"{"overrides": {}, "defaults": {}}"#)
        .create_async()
        .await;
    let load_table = format!(
        r#"{{"metadata-location": "{}/metadata/v1.metadata.json",
            "metadata": {metadata}, "config": {{}}}}"#,
        dir.to_string_lossy()
    );
    server
        .mock("GET", "/v1/namespaces/db/tables/events")
        .with_status(200)
        .with_body(load_table)
        .create_async()
        .await;
    // The commit: a POST of requirements + updates to the table resource.
    let committed = server
        .mock("POST", "/v1/namespaces/db/tables/events")
        .match_body(mockito::Matcher::Regex(
            "set-statistics.*manifest/1\\.puffin".to_owned(),
        ))
        .with_status(200)
        .with_body(format!(
            r#"{{"metadata-location": "{}/metadata/v2.metadata.json",
                "metadata": {metadata}}}"#,
            dir.to_string_lossy()
        ))
        .create_async()
        .await;

    let catalog = RestCatalog::new(RestCatalogConfig::builder().uri(server.url()).build());
    let loaded = catalog.load_table(&events()).await.expect("load table");

    // Real derived state on disk first, advertised second — the same order as
    // `write`, so a published path never dangles.
    let file_io = Arc::new(
        FileIO::from_path(dir.to_str().expect("utf-8"))
            .expect("file io")
            .build()
            .expect("build file io"),
    );
    let store = Store::new(
        file_io,
        Arc::new(LocalFileSystem::new()),
        Layout::beside(&dir.to_string_lossy()),
    );
    let index_path = store
        .write(&TableId(TABLE_UUID.into()), SnapshotId(1), POLICY, &{
            let mut index = Index::new(TENANT_FIELD);
            index.insert(
                hash_scalar(&datafusion::scalar::ScalarValue::Int64(Some(1))),
                FileId("a.parquet".into()),
            );
            index.with_bytes(128)
        })
        .await
        .expect("write index");

    let stats = store
        .publish(
            &loaded,
            &catalog,
            SnapshotId(1),
            &[Advertised {
                kind: quarry::engine::QUARRY_EQ_INDEX_V2.to_owned(),
                fields: vec![TENANT_FIELD as i32],
                path: index_path.clone(),
                properties: HashMap::new(),
            }],
        )
        .await
        .expect("publish");

    committed.assert_async().await;
    assert_eq!(stats.snapshot_id, 1);
    assert!(
        stats.statistics_path.ends_with("manifest/1.puffin"),
        "published path: {}",
        stats.statistics_path
    );

    // And the advertisement resolves: what the catalog now records points at
    // bytes that recover.
    let recovered = store
        .recover_published(&TableId(TABLE_UUID.into()), &stats, &[SnapshotId(1)])
        .await
        .expect("recover");
    assert_eq!(recovered.registry.len(), 1);
}
