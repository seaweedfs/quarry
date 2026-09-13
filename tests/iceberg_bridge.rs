//! Deriving a snapshot graph from a real Iceberg table.
//!
//! The table here is genuine: real metadata JSON, real Avro manifest lists and
//! manifests, written with `iceberg-rust`'s own writers and read back through
//! its own readers. Only the Parquet data files are absent, because the bridge
//! never opens them — it reads metadata, which is exactly the point.
//!
//! ```sh
//! cargo test --features iceberg --test iceberg_bridge
//! ```

#![cfg(feature = "iceberg")]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, ManifestListWriter, ManifestWriterBuilder,
    Struct, TableMetadata,
};

use quarry::from_iceberg::{current_snapshot, live_data_files, snapshot_graph, table_id};
use quarry::snapshot::{DeleteState, FileId, SnapshotId};

/// One data or delete file to record in a snapshot.
struct Entry {
    path: &'static str,
    content: DataContentType,
}

fn data(path: &'static str) -> Entry {
    Entry {
        path,
        content: DataContentType::Data,
    }
}

fn position_deletes(path: &'static str) -> Entry {
    Entry {
        path,
        content: DataContentType::PositionDeletes,
    }
}

/// A directory of its own per test.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("metadata")).expect("create scratch dir");
    dir
}

/// Metadata for a table whose snapshots are 1 → 2 → 3, unpartitioned.
fn metadata_json(dir: &Path, snapshots: usize) -> String {
    let location = dir.to_string_lossy();
    let entries: Vec<String> = (1..=snapshots)
        .map(|id| {
            let parent = if id > 1 {
                format!(r#""parent-snapshot-id": {},"#, id - 1)
            } else {
                String::new()
            };
            format!(
                r#"{{
                    "snapshot-id": {id},
                    {parent}
                    "sequence-number": {id},
                    "timestamp-ms": {id},
                    "summary": {{"operation": "append"}},
                    "manifest-list": "{location}/metadata/list-{id}.avro",
                    "schema-id": 0
                }}"#
            )
        })
        .collect();

    format!(
        r#"{{
            "format-version": 2,
            "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
            "location": "{location}",
            "last-sequence-number": {snapshots},
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
            "current-snapshot-id": {snapshots},
            "snapshots": [{}],
            "snapshot-log": [],
            "metadata-log": [],
            "refs": {{"main": {{"snapshot-id": {snapshots}, "type": "branch"}}}}
        }}"#,
        entries.join(",")
    )
}

/// Write the manifest list and manifest for one snapshot.
async fn write_snapshot(
    dir: &Path,
    file_io: &FileIO,
    metadata: &TableMetadata,
    snapshot_id: i64,
    entries: &[Entry],
) {
    let schema = metadata.current_schema().clone();
    let spec = metadata.default_partition_spec().as_ref().clone();

    let build = |kind: &str| {
        ManifestWriterBuilder::new(
            file_io
                .new_output(format!(
                    "{}/metadata/manifest-{snapshot_id}-{kind}.avro",
                    dir.to_string_lossy()
                ))
                .expect("manifest output"),
            Some(snapshot_id),
            None,
            schema.clone(),
            spec.clone(),
        )
    };

    let file_of = |entry: &Entry| {
        DataFileBuilder::default()
            .partition_spec_id(0)
            .content(entry.content)
            .file_path(entry.path.to_owned())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1000)
            .record_count(10)
            .partition(Struct::empty())
            .build()
            .expect("data file")
    };

    // Iceberg keeps data files and delete files in manifests of different
    // content types, so a snapshot with both needs two manifests. The bridge
    // has to read both kinds, which is exactly what this exercises.
    let mut manifests = Vec::new();

    let data_entries: Vec<&Entry> = entries
        .iter()
        .filter(|entry| entry.content == DataContentType::Data)
        .collect();
    if !data_entries.is_empty() {
        let mut writer = build("data").build_v2_data();
        for entry in data_entries {
            writer
                .add_file(file_of(entry), snapshot_id)
                .expect("add data file");
        }
        manifests.push(writer.write_manifest_file().await.expect("write data"));
    }

    let delete_entries: Vec<&Entry> = entries
        .iter()
        .filter(|entry| entry.content != DataContentType::Data)
        .collect();
    if !delete_entries.is_empty() {
        let mut writer = build("deletes").build_v2_deletes();
        for entry in delete_entries {
            // `add_file`, not `add_delete_file`. The latter reads as though it
            // adds a delete file, but it sets the entry's status to `Deleted`,
            // which per the spec means "this file was REMOVED from the table".
            // A delete file that is live in the snapshot is an `Added` entry in
            // a deletes-content manifest.
            writer
                .add_file(file_of(entry), snapshot_id)
                .expect("add delete file");
        }
        manifests.push(writer.write_manifest_file().await.expect("write deletes"));
    }

    let list_path = format!("{}/metadata/list-{snapshot_id}.avro", dir.to_string_lossy());
    let mut list = ManifestListWriter::v2(
        file_io.new_output(&list_path).expect("list output"),
        snapshot_id,
        if snapshot_id > 1 {
            Some(snapshot_id - 1)
        } else {
            None
        },
        snapshot_id,
    );
    list.add_manifests(manifests.into_iter())
        .expect("add manifests");
    list.close().await.expect("close list");
}

/// Build a table whose snapshot `n` holds `snapshots[n - 1]`.
async fn table(name: &str, snapshots: &[Vec<Entry>]) -> (TableMetadata, FileIO) {
    let dir = scratch(name);
    let file_io = FileIO::from_path(dir.to_str().expect("utf-8 path"))
        .expect("file io")
        .build()
        .expect("build file io");

    let metadata: TableMetadata =
        serde_json::from_str(&metadata_json(&dir, snapshots.len())).expect("valid metadata");

    for (index, entries) in snapshots.iter().enumerate() {
        write_snapshot(&dir, &file_io, &metadata, index as i64 + 1, entries).await;
    }

    (metadata, file_io)
}

fn f(path: &str) -> FileId {
    FileId(path.to_owned())
}

#[tokio::test]
async fn the_table_uuid_is_the_table_identity() {
    let (metadata, _) = table("iceberg_identity", &[vec![data("a.parquet")]]).await;
    assert_eq!(
        table_id(&metadata).0,
        "9c12d441-03fe-4693-9a96-a0705ddf69c1",
        "identity must be the uuid, so renaming a table does not orphan \
         derived state"
    );
    assert_eq!(current_snapshot(&metadata), Some(SnapshotId(1)));
}

#[tokio::test]
async fn lineage_comes_from_parent_snapshot_ids() {
    let (metadata, file_io) = table(
        "iceberg_lineage",
        &[
            vec![data("a.parquet")],
            vec![data("a.parquet"), data("b.parquet")],
            vec![data("a.parquet"), data("b.parquet"), data("c.parquet")],
        ],
    )
    .await;

    let graph = snapshot_graph(&metadata, &file_io)
        .await
        .expect("build graph");

    assert!(graph.is_descendant_or_self(SnapshotId(3), SnapshotId(1)));
    assert!(graph.is_descendant_or_self(SnapshotId(3), SnapshotId(2)));
    assert!(!graph.is_descendant_or_self(SnapshotId(1), SnapshotId(3)));
}

#[tokio::test]
async fn live_files_come_from_the_manifests() {
    let (metadata, file_io) = table(
        "iceberg_live_files",
        &[
            vec![data("a.parquet")],
            vec![data("a.parquet"), data("b.parquet")],
        ],
    )
    .await;

    let graph = snapshot_graph(&metadata, &file_io)
        .await
        .expect("build graph");

    let at_one: BTreeSet<FileId> = graph
        .get(SnapshotId(1))
        .expect("snapshot 1")
        .files()
        .keys()
        .cloned()
        .collect();
    assert_eq!(at_one, BTreeSet::from([f("a.parquet")]));

    let at_two: BTreeSet<FileId> = graph
        .get(SnapshotId(2))
        .expect("snapshot 2")
        .files()
        .keys()
        .cloned()
        .collect();
    assert_eq!(at_two, BTreeSet::from([f("a.parquet"), f("b.parquet")]));
}

#[tokio::test]
async fn an_append_only_table_diffs_as_purely_additive() {
    let (metadata, file_io) = table(
        "iceberg_append",
        &[
            vec![data("a.parquet")],
            vec![data("a.parquet"), data("b.parquet")],
        ],
    )
    .await;

    let graph = snapshot_graph(&metadata, &file_io)
        .await
        .expect("build graph");

    let diff = graph
        .diff(SnapshotId(1), SnapshotId(2))
        .expect("both snapshots known");
    assert_eq!(diff.added, BTreeSet::from([f("b.parquet")]));
    assert!(
        diff.is_purely_additive(),
        "an append must leave substituting derived state usable"
    );
}

#[tokio::test]
async fn a_delete_file_shows_up_as_subtractive_change() {
    // Snapshot 2 adds a position delete file. No data file was added or
    // removed, so a file-set diff alone would see nothing at all.
    let (metadata, file_io) = table(
        "iceberg_deletes",
        &[
            vec![data("a.parquet")],
            vec![data("a.parquet"), position_deletes("d.parquet")],
        ],
    )
    .await;

    let graph = snapshot_graph(&metadata, &file_io)
        .await
        .expect("build graph");

    let one = graph.get(SnapshotId(1)).expect("snapshot 1");
    let two = graph.get(SnapshotId(2)).expect("snapshot 2");
    assert_eq!(
        one.files().get(&f("a.parquet")),
        Some(&DeleteState::NONE),
        "nothing is deleted at snapshot 1"
    );
    assert_ne!(
        two.files().get(&f("a.parquet")),
        Some(&DeleteState::NONE),
        "the delete file must change the data file's delete state"
    );

    // The delete file itself is not a data file.
    assert_eq!(
        two.files().keys().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from([f("a.parquet")])
    );

    let diff = graph
        .diff(SnapshotId(1), SnapshotId(2))
        .expect("both snapshots known");
    assert!(diff.added.is_empty(), "no data file was added");
    assert_eq!(diff.redeleted, BTreeSet::from([f("a.parquet")]));
    assert!(
        !diff.is_purely_additive(),
        "a delete must disqualify substituting derived state"
    );
}

#[tokio::test]
async fn compaction_shows_up_as_files_removed() {
    let (metadata, file_io) = table(
        "iceberg_compaction",
        &[
            vec![data("small-1.parquet"), data("small-2.parquet")],
            vec![data("merged.parquet")],
        ],
    )
    .await;

    let graph = snapshot_graph(&metadata, &file_io)
        .await
        .expect("build graph");

    let diff = graph
        .diff(SnapshotId(1), SnapshotId(2))
        .expect("both snapshots known");
    assert_eq!(diff.added, BTreeSet::from([f("merged.parquet")]));
    assert_eq!(
        diff.removed,
        BTreeSet::from([f("small-1.parquet"), f("small-2.parquet")])
    );
    assert!(!diff.is_purely_additive());
}

#[tokio::test]
async fn file_sizes_are_reported_for_registering_objects() {
    let (metadata, file_io) = table(
        "iceberg_sizes",
        &[vec![data("a.parquet"), data("b.parquet")]],
    )
    .await;

    let sizes = live_data_files(&metadata, &file_io, SnapshotId(1))
        .await
        .expect("read sizes");
    assert_eq!(sizes.len(), 2);
    assert_eq!(sizes.get(&f("a.parquet")), Some(&1000));
}

#[tokio::test]
async fn an_unknown_snapshot_yields_no_files_rather_than_an_error() {
    let (metadata, file_io) = table("iceberg_unknown", &[vec![data("a.parquet")]]).await;
    let sizes = live_data_files(&metadata, &file_io, SnapshotId(999))
        .await
        .expect("no error");
    assert!(sizes.is_empty());
}
