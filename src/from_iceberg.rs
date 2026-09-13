//! Deriving a [`SnapshotGraph`] from real Iceberg table metadata.
//!
//! The rule needs two things from a table: which snapshot descends from which,
//! and which data files are live at each — with enough fidelity that a change
//! in what is *deleted* shows up even when no data file was added or removed.
//! Iceberg has all of it; this walks the metadata and manifests to extract it.
//!
//! Behind the `iceberg` feature, and independent of `engine`: the bridge is
//! pure metadata, so it needs no query engine, and the engine needs no Iceberg.
//!
//! # How deletes are handled, and why coarsely
//!
//! [`DeleteState`] only has to *change* when the deletes affecting a data file
//! change; it is an opaque fingerprint, not a description. Attributing deletes
//! to individual data files is possible in part — a position delete file may
//! name a `referenced_data_file` — but equality deletes apply by value across a
//! whole partition, so exact attribution is not generally available.
//!
//! So this errs coarse and safe:
//!
//! ```text
//! no delete files in the snapshot
//!     → every data file is DeleteState::NONE
//!
//! any delete files in the snapshot
//!     → every data file gets the SAME fingerprint, derived from the
//!       whole delete-file set
//! ```
//!
//! The consequence is over-invalidation, never under: a delete anywhere in the
//! table makes every substituting piece of derived state look stale, while
//! pruning keeps working because pruning tolerates any change. Wrong in the
//! direction of slow rather than the direction of wrong.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use iceberg::io::FileIO;
use iceberg::spec::{DataContentType, ManifestStatus, TableMetadata};

use crate::snapshot::{DeleteState, FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};

/// The table's identity, independent of its name.
///
/// Iceberg's table UUID, so that renaming a table does not orphan the derived
/// state built from it.
pub fn table_id(metadata: &TableMetadata) -> TableId {
    TableId(metadata.uuid().to_string())
}

/// The path a storage layer would use for a file Iceberg records.
///
/// Iceberg records data file locations as URIs, which the object store
/// addresses as a path *within* a store identified separately. So
/// `s3://bucket/data/a.parquet` becomes `data/a.parquet`, and the bucket is
/// carried by the store's own URL.
///
/// This is the one place catalog identity and storage identity are reconciled,
/// and doing it here means [`FileId`] means the same thing everywhere: the rule
/// reasons about it, and the scan reads it, without either converting.
pub fn object_path(file_path: &str) -> String {
    match file_path.find("://") {
        None => file_path.to_owned(),
        Some(scheme_end) => {
            let after_scheme = &file_path[scheme_end + 3..];
            match after_scheme.find('/') {
                // scheme://authority/path → path
                Some(authority_end) => after_scheme[authority_end + 1..].to_owned(),
                // scheme://authority with no path at all
                None => String::new(),
            }
        }
    }
}

/// The snapshot a query would read by default.
pub fn current_snapshot(metadata: &TableMetadata) -> Option<SnapshotId> {
    metadata
        .current_snapshot()
        .map(|snapshot| SnapshotId(snapshot.snapshot_id()))
}

/// Build a [`SnapshotGraph`] covering every snapshot the metadata retains.
///
/// Reads each snapshot's manifest list and manifests, so it costs one round
/// trip per manifest. Snapshots that have been expired are simply absent, and
/// the rule already refuses derived state whose source snapshot it cannot find.
pub async fn snapshot_graph(
    metadata: &TableMetadata,
    file_io: &FileIO,
) -> iceberg::Result<SnapshotGraph> {
    let mut graph = SnapshotGraph::new();

    for snapshot in metadata.snapshots() {
        let contents = read_snapshot(snapshot, metadata, file_io).await?;

        let mut built = match snapshot.parent_snapshot_id() {
            Some(parent) => {
                Snapshot::child_of(SnapshotId(snapshot.snapshot_id()), SnapshotId(parent))
            }
            None => Snapshot::root(SnapshotId(snapshot.snapshot_id())),
        };
        let deletes = fingerprint(&contents.delete_files);
        for file in contents.data_files {
            built = built.with_file(file, deletes);
        }
        graph.insert(built);
    }

    Ok(graph)
}

/// Every data file live at `snapshot`, with the deletes applying to it.
///
/// Exposed because it is useful on its own: a caller wanting to register the
/// table's Parquet objects needs exactly this list.
pub async fn live_data_files(
    metadata: &TableMetadata,
    file_io: &FileIO,
    at: SnapshotId,
) -> iceberg::Result<BTreeMap<FileId, u64>> {
    let Some(snapshot) = metadata.snapshot_by_id(at.0) else {
        return Ok(BTreeMap::new());
    };
    let contents = read_snapshot(snapshot, metadata, file_io).await?;
    Ok(contents.sizes)
}

struct Contents {
    data_files: BTreeSet<FileId>,
    delete_files: BTreeSet<String>,
    sizes: BTreeMap<FileId, u64>,
}

async fn read_snapshot(
    snapshot: &iceberg::spec::Snapshot,
    metadata: &TableMetadata,
    file_io: &FileIO,
) -> iceberg::Result<Contents> {
    let mut contents = Contents {
        data_files: BTreeSet::new(),
        delete_files: BTreeSet::new(),
        sizes: BTreeMap::new(),
    };

    let manifest_list = snapshot.load_manifest_list(file_io, metadata).await?;
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(file_io).await?;
        for entry in manifest.entries() {
            // A `Deleted` entry records that a file left the table in this
            // snapshot; it is not live. `Existing` and `Added` both are.
            if entry.status() == ManifestStatus::Deleted {
                continue;
            }
            let data_file = entry.data_file();
            match data_file.content_type() {
                DataContentType::Data => {
                    let id = FileId(object_path(data_file.file_path()));
                    contents
                        .sizes
                        .insert(id.clone(), data_file.file_size_in_bytes());
                    contents.data_files.insert(id);
                }
                DataContentType::PositionDeletes | DataContentType::EqualityDeletes => {
                    contents
                        .delete_files
                        .insert(object_path(data_file.file_path()));
                }
            }
        }
    }

    Ok(contents)
}

/// A fingerprint of a snapshot's delete files.
///
/// Any change to the set changes the value, which is all [`DeleteState`]
/// promises. An empty set is [`DeleteState::NONE`] so that an append-only table
/// never looks as though anything was deleted.
fn fingerprint(delete_files: &BTreeSet<String>) -> DeleteState {
    if delete_files.is_empty() {
        return DeleteState::NONE;
    }
    let mut hasher = crate::stable_hash::StableHasher::new();
    for path in delete_files {
        path.hash(&mut hasher);
    }
    // Never collide with NONE: a table that has deletes must not look clean.
    DeleteState(hasher.finish() | 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_path_is_left_alone() {
        assert_eq!(
            object_path("/tmp/table/data/a.parquet"),
            "/tmp/table/data/a.parquet"
        );
        assert_eq!(object_path("data/a.parquet"), "data/a.parquet");
    }

    #[test]
    fn a_uri_loses_its_scheme_and_authority() {
        // The authority is carried by the object store's own URL, so the path
        // here is the path within that store.
        assert_eq!(object_path("s3://bucket/data/a.parquet"), "data/a.parquet");
        assert_eq!(object_path("file:///tmp/t/a.parquet"), "tmp/t/a.parquet");
        assert_eq!(object_path("gs://b/x/y.parquet"), "x/y.parquet");
    }

    #[test]
    fn a_uri_with_no_path_yields_nothing_rather_than_panicking() {
        assert_eq!(object_path("s3://bucket"), "");
    }

    #[test]
    fn an_empty_delete_set_is_clean() {
        assert_eq!(fingerprint(&BTreeSet::new()), DeleteState::NONE);
    }

    #[test]
    fn any_delete_set_is_distinguishable_from_clean() {
        let one = BTreeSet::from(["deletes-1.parquet".to_owned()]);
        assert_ne!(fingerprint(&one), DeleteState::NONE);
    }

    #[test]
    fn the_fingerprint_changes_when_the_delete_set_does() {
        let one = BTreeSet::from(["a".to_owned()]);
        let two = BTreeSet::from(["a".to_owned(), "b".to_owned()]);
        assert_ne!(
            fingerprint(&one),
            fingerprint(&two),
            "adding a delete file must invalidate substituting derived state"
        );
    }

    #[test]
    fn the_fingerprint_is_stable_for_the_same_set() {
        let set = BTreeSet::from(["b".to_owned(), "a".to_owned()]);
        assert_eq!(fingerprint(&set), fingerprint(&set));
    }
}
