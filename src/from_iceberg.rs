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

use crate::derived::FieldId;
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

/// Per-file value ranges for one field, as the manifests record them.
///
/// The input the index gate needs, and it costs nothing to obtain: Iceberg
/// already stores a lower and upper bound per field per data file, so whether
/// an index could beat the format's own pruning is answerable from metadata
/// alone. See [`Spread`](crate::layout::Spread).
///
/// Files that record no bound for the field are omitted, which the estimate
/// treats as files that must always be read — so a table with no statistics
/// degrades to "assume the worst" rather than to a wrong answer.
///
/// Only orderable numeric and temporal types are mapped. Strings and binary
/// are left out: they have bounds, but projecting them onto a number to
/// compare range *widths* would invent a distance that does not exist.
pub async fn bounds_by_field(
    metadata: &TableMetadata,
    file_io: &FileIO,
    at: SnapshotId,
) -> iceberg::Result<BTreeMap<FieldId, Vec<(f64, f64)>>> {
    let Some(snapshot) = metadata.snapshot_by_id(at.0) else {
        return Ok(BTreeMap::new());
    };

    // Every field in one pass. Walking the manifests per field would multiply
    // the metadata reads by the width of the table for no reason.
    let mut bounds: BTreeMap<FieldId, Vec<(f64, f64)>> = BTreeMap::new();
    let manifest_list = snapshot.load_manifest_list(file_io, metadata).await?;
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(file_io).await?;
        for entry in manifest.entries() {
            if entry.status() == ManifestStatus::Deleted {
                continue;
            }
            let data_file = entry.data_file();
            if data_file.content_type() != DataContentType::Data {
                continue;
            }
            for (key, low) in data_file.lower_bounds() {
                let Some(low) = as_number(low) else { continue };
                let Some(high) = data_file.upper_bounds().get(key).and_then(as_number) else {
                    continue;
                };
                bounds.entry(*key as FieldId).or_default().push((low, high));
            }
        }
    }
    Ok(bounds)
}

/// One bound as a number, if the type has a meaningful distance.
fn as_number(bound: &iceberg::spec::Datum) -> Option<f64> {
    use iceberg::spec::PrimitiveLiteral;
    match bound.literal() {
        PrimitiveLiteral::Int(value) => Some(*value as f64),
        PrimitiveLiteral::Long(value) => Some(*value as f64),
        PrimitiveLiteral::Float(value) => Some(value.into_inner() as f64),
        PrimitiveLiteral::Double(value) => Some(value.into_inner()),
        // Booleans, strings, binary, decimals and UUIDs are ordered but have
        // no distance that range widths could be compared across.
        _ => None,
    }
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
