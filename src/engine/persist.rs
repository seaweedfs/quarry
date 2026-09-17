//! Writing derived state down, and finding it again.
//!
//! The registry, the workload and the optimizer's record of its own builds all
//! lived in memory, so a restart lost everything learned and everything built
//! — and then, with credits gone, retirement would have deleted the indexes
//! whose bytes were still sitting in storage. Disposable state is fine;
//! destroying it on every restart is not.
//!
//! # The format is Iceberg's, not ours
//!
//! Puffin is the format Iceberg already defines for derived blobs — sketches,
//! deletion vectors, statistics. Its metadata carries almost exactly what
//! [`Derived`] needs:
//!
//! ```text
//! Puffin BlobMetadata          Quarry
//!   snapshot_id            ≡     Derived.source.snapshot   ← the lineage anchor
//!   fields: Vec<i32>       ≡     Index.field               ← a Vec, so composite is free
//!   type: String           ≡     Kind::name()
//!   properties: Map              policy fingerprint, hash version, postings
//! ```
//!
//! Using it means an unfamiliar engine encountering these files skips them
//! safely as an unknown blob type, rather than tripping over a private format.
//! It does not mean another engine can *use* them: this is a standard
//! location, not a standard index format.
//!
//! # The path is the metadata
//!
//! ```text
//! <prefix>/<table-uuid>/<field>/<source-snapshot>.puffin
//! ```
//!
//! so the registry is recoverable by listing. Nothing has to be read to know
//! what a file is or which snapshot it belongs to, there is no second store to
//! keep in step, and a crash between writing a blob and registering it leaves
//! an orphan that listing finds — an orphan costs storage, whereas a registry
//! entry with no blob behind it costs confidence in every decision.
//!
//! Write order follows from that: **blob first, register second.**
//!
//! # Two IO stacks, named rather than hidden
//!
//! `iceberg::io::FileIO` has no listing operation — it can read, write, delete
//! and test existence, and that is all. So recovery needs both:
//!
//! ```text
//! FileIO        reading and writing Puffin blobs, because that is what
//!               PuffinReader and PuffinWriter take
//!
//! ObjectStore   listing, because FileIO cannot, and because this is the
//!               stack that is already metered and cached
//! ```
//!
//! The two name paths differently — FileIO takes the location Iceberg records,
//! an object store addresses a path within a store — so [`Layout::parse`]
//! reads identity from the *tail* of a path rather than by stripping a prefix.
//! Three segments is all it needs, and that works in either path space.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, UInt64Array};
use datafusion::arrow::compute::take;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::parquet::arrow::arrow_reader::ArrowReaderMetadata;
use datafusion::parquet::arrow::async_reader::ParquetObjectReader;
use datafusion::scalar::ScalarValue;
use iceberg::Catalog;
use iceberg::io::{FileIO, FileRead};
use iceberg::puffin::{Blob, CompressionCodec, PuffinReader, PuffinWriter};
use iceberg::spec::StatisticsFile;
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use object_store::ObjectStore;
use object_store::path::Path as ObjectPath;

use crate::derived::{Derived, DerivedId, FieldId, Filter, Metric, PolicyFingerprint, Source};
use crate::kinds::{Bitmap, FilterSet, Index, JoinHash, TextIndex, VectorIndex};
use crate::registry::Registry;
use crate::snapshot::{FileId, SnapshotId, TableId};
use crate::stable_hash::{HASH_VERSION, StableHasher};
use crate::workload::{Fingerprint, Seen, Workload};

use super::table::RowFiles;

/// The blob type for an equality index.
///
/// Versioned in the name: a change to the layout below becomes a different
/// type, which an older reader skips rather than misreads.
pub const QUARRY_EQ_INDEX_V2: &str = "quarry-eq-index-v2";

/// The blob type for a row-level [`Bitmap`] index.
pub const QUARRY_BITMAP_V1: &str = "quarry-bitmap-v1";

/// The blob type for a persisted filter set.
pub const QUARRY_FILTER_SET_V1: &str = "quarry-filter-set-v1";

/// The blob type for a persisted text index.
pub const QUARRY_TEXT_INDEX_V1: &str = "quarry-text-index-v1";

/// The blob type for a saved vector index's metadata. The rows themselves
/// are written as Parquet at a companion path — see `Layout::vector_rows_path`
/// — because they are the bulk of the index and Parquet is their native
/// columnar format. The Puffin blob carries only what recovery needs to
/// reconstruct the kind and find the rows.
pub const QUARRY_VECTOR_INDEX_V1: &str = "quarry-vector-index-v1";

/// The blob type for a saved join hash's metadata. Like the vector index,
/// the build-side rows are written as Parquet at a companion path — see
/// `Layout::join_hash_rows_path` — and the Puffin blob carries only the
/// join keys and covered columns recovery needs to reconstruct the kind.
pub const QUARRY_JOIN_HASH_V1: &str = "quarry-join-hash-v1";

/// The blob property holding the canonical filter text a filter set answers.
const FILTER_TEXT_PROPERTY: &str = "quarry.filter";

/// Property carrying the hash version the postings were built with.
const HASH_VERSION_PROPERTY: &str = "quarry.hash-version";
/// Property carrying the policy the index was built for.
const POLICY_PROPERTY: &str = "quarry.policy";
/// Property carrying how many Parquet shards a join hash's rows were
/// written across. Absent on the earliest blobs, whose single rows file
/// lives at `{snapshot}.parquet`.
const SHARDS_PROPERTY: &str = "quarry.shards";

/// The target bytes per join-hash shard when a `Store` writes one.
///
/// Rows are bucketed by join key so each shard is a contiguous slice of the
/// key space; 128 MiB keeps a build of a few gigabytes at tens of files
/// rather than thousands, each still large enough to read efficiently.
const JOIN_HASH_SHARD_BYTES: u64 = 128 * 1024 * 1024;

/// An upper bound on shards, so a pathological `shard_bytes` cannot turn a
/// write into a small-file problem.
const JOIN_HASH_MAX_SHARDS: u64 = 1024;

/// Where derived state for a table lives.
///
/// One directory per table, one per field beneath it, named by the snapshot
/// the index was built from.
#[derive(Clone, Debug)]
pub struct Layout {
    prefix: String,
}

impl Layout {
    /// Keep derived state under `prefix`.
    ///
    /// Typically the table's own location plus a reserved directory, so
    /// derived state travels with the table and is removed with it.
    pub fn new(prefix: impl Into<String>) -> Self {
        Layout {
            prefix: prefix.into().trim_end_matches('/').to_owned(),
        }
    }

    /// Derived state beside the table it came from.
    pub fn beside(table_location: &str) -> Self {
        Layout::new(format!(
            "{}/_quarry/idx",
            table_location.trim_end_matches('/')
        ))
    }

    /// Where one index belongs, in the path space `FileIO` uses.
    pub fn index_path(&self, table: &TableId, field: FieldId, at: SnapshotId) -> String {
        format!("{}/{}/{}/{}.puffin", self.prefix, table.0, field, at.0)
    }

    /// Where a vector index's rows belong, as Parquet.
    ///
    /// Distinct from [`index_path`](Self::index_path) by extension: a
    /// `.parquet` file beside the `.puffin` metadata blob. The two travel
    /// together and share the `(table, field, snapshot)` identity.
    pub fn vector_rows_path(&self, table: &TableId, field: FieldId, at: SnapshotId) -> String {
        format!("{}/{}/{}/{}.parquet", self.prefix, table.0, field, at.0)
    }

    /// Where a join hash's metadata belongs — keyed by the join shape's
    /// stable hash, not a single field, because the join shape (keys +
    /// columns) is the identity.
    pub fn join_hash_path(&self, table: &TableId, id_hash: u64, at: SnapshotId) -> String {
        format!(
            "{}/{}/jh-{:016x}/{}.puffin",
            self.prefix, table.0, id_hash, at.0
        )
    }

    /// Where a join hash's rows belong, as Parquet.
    ///
    /// Distinct from [`join_hash_path`](Self::join_hash_path) by extension,
    /// the same convention as `vector_rows_path`. Written only by blobs that
    /// predate sharding — new writes use [`join_hash_shard_path`](Self::join_hash_shard_path).
    pub fn join_hash_rows_path(&self, table: &TableId, id_hash: u64, at: SnapshotId) -> String {
        format!(
            "{}/{}/jh-{:016x}/{}.parquet",
            self.prefix, table.0, id_hash, at.0
        )
    }

    /// Where shard `shard` of a join hash's rows belongs, as Parquet.
    pub fn join_hash_shard_path(
        &self,
        table: &TableId,
        id_hash: u64,
        at: SnapshotId,
        shard: u32,
    ) -> String {
        format!(
            "{}/{}/jh-{:016x}/{}.{:04}.parquet",
            self.prefix, table.0, id_hash, at.0, shard
        )
    }

    /// Every rows path a join hash's metadata names: `shards` numbered
    /// files, or the single pre-sharding path when `None`.
    pub fn join_hash_rows_paths(
        &self,
        table: &TableId,
        id_hash: u64,
        at: SnapshotId,
        shards: Option<u32>,
    ) -> Vec<String> {
        match shards {
            None => vec![self.join_hash_rows_path(table, id_hash, at)],
            Some(shards) => (0..shards)
                .map(|shard| self.join_hash_shard_path(table, id_hash, at, shard))
                .collect(),
        }
    }

    /// The `ObjectStoreUrl` a session addresses this layout's store by —
    /// the prefix's scheme and authority, or the local filesystem when the
    /// prefix is a bare path.
    ///
    /// The `ObjectStoreUrl` is the lookup key for a store registered with
    /// the session; paths within it are the [`object_path`](crate::from_iceberg::object_path)
    /// of each location.
    pub fn object_store_url(&self) -> ObjectStoreUrl {
        let parsed = match self.prefix.find("://") {
            // A bare path is a local file.
            None => Ok(ObjectStoreUrl::local_filesystem()),
            Some(i) => match self.prefix[i + 3..].find('/') {
                Some(j) => ObjectStoreUrl::parse(&self.prefix[..i + 3 + j + 1]),
                None => ObjectStoreUrl::parse(format!("{}/", self.prefix)),
            },
        };
        parsed.unwrap_or_else(|_| ObjectStoreUrl::local_filesystem())
    }

    /// Where one filter set belongs — keyed by the filter's hash, not a
    /// field, because the filter is the identity.
    pub fn filter_set_path(&self, table: &TableId, filter: &Filter, at: SnapshotId) -> String {
        self.filter_set_path_for(table, crate::stable_hash::StableHasher::of(filter), at)
    }

    /// The same path, keyed by the hash directly — for recovery, which has
    /// the hash from the listing before it has read the filter inside.
    fn filter_set_path_for(&self, table: &TableId, hash: u64, at: SnapshotId) -> String {
        format!("{}/{}/fset-{}/{}.puffin", self.prefix, table.0, hash, at.0)
    }

    /// Where a table's saved workload belongs.
    ///
    /// Outside the `<field>/<snapshot>` tree on purpose, so listing for
    /// indexes cannot mistake it for one.
    pub fn workload_path(&self, table: &TableId) -> String {
        format!("{}/{}/workload.puffin", self.prefix, table.0)
    }

    /// Everything for one table, in the path space `FileIO` uses.
    pub fn table_prefix(&self, table: &TableId) -> String {
        format!("{}/{}/", self.prefix, table.0)
    }

    /// Where the manifest published for one snapshot lives.
    ///
    /// `manifest` does not parse as a field id, so [`Layout::parse`] — and
    /// therefore listing-driven recovery — cannot mistake it for derived
    /// state.
    pub fn statistics_path(&self, table: &TableId, at: SnapshotId) -> String {
        format!("{}/{}/manifest/{}.puffin", self.prefix, table.0, at.0)
    }

    /// The same prefix as an object store addresses it.
    ///
    /// Listing happens through `ObjectStore`, which names paths within a store
    /// rather than by full location, so the scheme and authority come off —
    /// the same reconciliation
    /// [`object_path`](crate::from_iceberg::object_path) performs for data
    /// files.
    pub fn object_prefix(&self, table: &TableId) -> String {
        crate::from_iceberg::object_path(&self.table_prefix(table))
            .trim_start_matches('/')
            .to_owned()
    }

    /// The [`RowFiles`] a `write_join_hash` result describes — `files` are
    /// `(path, size)` in the path space `FileIO` uses; this converts them to
    /// the space the store addresses.
    pub fn row_files(&self, files: &[(String, u64)], schema: SchemaRef) -> RowFiles {
        RowFiles {
            url: self.object_store_url(),
            files: files
                .iter()
                .map(|(path, size)| (store_path(path), *size))
                .collect(),
            schema,
        }
    }

    /// The `FileIO` path for an object-space `path` under this table's
    /// prefix — the inverse of [`object_prefix`](Self::object_prefix), so
    /// listed objects can be named the way reads and deletes expect.
    pub fn file_io_path(&self, table: &TableId, object: &str) -> Option<String> {
        let rel = object.strip_prefix(&self.object_prefix(table))?;
        Some(format!("{}{rel}", self.table_prefix(table)))
    }

    /// Recover `(table, field, snapshot)` from a path this layout produced.
    ///
    /// Read from the **tail**, not by stripping the prefix, because the same
    /// object is named one way by `FileIO` and another by an object store and
    /// this has to recognise both. The last three segments carry everything:
    /// identity does not depend on how the path was reached.
    ///
    /// Anything that does not parse is not ours, and is ignored rather than
    /// treated as corrupt — other things may share the prefix.
    pub fn parse(&self, path: &str) -> Option<(TableId, FieldId, SnapshotId)> {
        let mut segments = path.rsplit('/');
        let snapshot = segments.next()?.strip_suffix(".puffin")?.parse().ok()?;
        let field = segments.next()?.parse().ok()?;
        let table = segments.next()?;
        if table.is_empty() {
            return None;
        }
        // The remaining prefix must be ours, in one naming or the other.
        let tail = format!("/{table}/{field}/{snapshot}.puffin");
        let head = path.strip_suffix(&tail)?;
        if !self.prefix.ends_with(head) && !head.ends_with(&self.prefix) {
            return None;
        }
        Some((TableId(table.to_owned()), field, SnapshotId(snapshot)))
    }

    /// Recover `(table, filter hash, snapshot)` from a
    /// [`Layout::filter_set_path`].
    ///
    /// The hash is not the identity itself — the filter text inside the
    /// blob is — so the caller must read the blob before trusting it. The
    /// segment exists only so listing can route the read.
    pub fn parse_filter_set(&self, path: &str) -> Option<(TableId, u64, SnapshotId)> {
        let mut segments = path.rsplit('/');
        let snapshot = segments.next()?.strip_suffix(".puffin")?.parse().ok()?;
        let hash = segments.next()?.strip_prefix("fset-")?.parse().ok()?;
        let table = segments.next()?;
        if table.is_empty() {
            return None;
        }
        Some((TableId(table.to_owned()), hash, SnapshotId(snapshot)))
    }

    /// Recover `(table, id_hash, snapshot)` from a
    /// [`Layout::join_hash_path`].
    ///
    /// Like [`parse_filter_set`](Self::parse_filter_set), the hash is not the
    /// identity itself — the keys and columns inside the blob are — but the
    /// segment lets listing route the read. The caller reconstructs the
    /// `DerivedId` from `(table, id_hash)`.
    pub fn parse_join_hash(&self, path: &str) -> Option<(TableId, u64, SnapshotId)> {
        let mut segments = path.rsplit('/');
        let snapshot = segments.next()?.strip_suffix(".puffin")?.parse().ok()?;
        let hash = u64::from_str_radix(segments.next()?.strip_prefix("jh-")?, 16).ok()?;
        let table = segments.next()?;
        if table.is_empty() {
            return None;
        }
        Some((TableId(table.to_owned()), hash, SnapshotId(snapshot)))
    }
}

/// Serialise an index's postings.
///
/// A file table first, then postings referring to it by number. Paths are
/// 60–100 bytes and a value can appear in many files, so writing each path
/// once rather than once per posting is where most of the size went: measured
/// on a million-posting index, 90 MB became 15 MB.
///
/// Compression is Puffin's job, not this function's — though `iceberg-rust`
/// 0.6 does not implement any codec yet, which is why this is worth doing by
/// hand.
fn encode(index: &Index) -> Vec<u8> {
    let paths: Vec<&FileId> = index
        .postings()
        .flat_map(|(_, files)| files)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let number_of: BTreeMap<&FileId, u32> = paths
        .iter()
        .enumerate()
        .map(|(index, file)| (*file, index as u32))
        .collect();

    let mut out = Vec::new();
    out.extend_from_slice(&(paths.len() as u32).to_le_bytes());
    for file in &paths {
        let bytes = file.0.as_bytes();
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }

    out.extend_from_slice(&(index.values() as u64).to_le_bytes());
    for (value, files) in index.postings() {
        out.extend_from_slice(&value.to_le_bytes());
        out.extend_from_slice(&(files.len() as u32).to_le_bytes());
        for file in files {
            out.extend_from_slice(&number_of[file].to_le_bytes());
        }
    }
    out
}

/// Read postings back, or `None` if the bytes are not a well-formed index.
///
/// Returns `None` rather than a partial index on any inconsistency. A
/// truncated index is not a smaller index: it would prune away files that
/// hold matching rows, which is the one direction that returns wrong answers.
fn decode(field: FieldId, bytes: &[u8]) -> Option<Index> {
    let mut cursor = Cursor { bytes, at: 0 };

    let path_count = cursor.u32()?;
    let mut paths = Vec::with_capacity(path_count.min(1 << 16) as usize);
    for _ in 0..path_count {
        let len = cursor.u32()? as usize;
        paths.push(FileId(String::from_utf8(cursor.take(len)?.to_vec()).ok()?));
    }

    let values = cursor.u64()?;
    let mut index = Index::new(field);
    for _ in 0..values {
        let value = cursor.u64()?;
        let files = cursor.u32()?;
        for _ in 0..files {
            // A number outside the table means these bytes are not what we
            // think they are. Skipping it would silently drop a file the
            // index should have named, which under-selects.
            let file = paths.get(cursor.u32()? as usize)?;
            index.insert(value, file.clone());
        }
    }

    if cursor.at != bytes.len() {
        return None; // Trailing bytes mean this is not what we think it is.
    }
    Some(index)
}

fn encode_bitmap(bitmap: &Bitmap) -> Vec<u8> {
    let paths: Vec<&FileId> = bitmap
        .postings()
        .flat_map(|(_, files)| files.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let number_of: BTreeMap<&FileId, u32> = paths
        .iter()
        .enumerate()
        .map(|(index, file)| (*file, index as u32))
        .collect();

    let mut out = Vec::new();
    out.extend_from_slice(&(paths.len() as u32).to_le_bytes());
    for file in &paths {
        let bytes = file.0.as_bytes();
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }

    let values = bitmap.postings().count() as u64;
    out.extend_from_slice(&values.to_le_bytes());
    for (value, files) in bitmap.postings() {
        out.extend_from_slice(&value.to_le_bytes());
        out.extend_from_slice(&(files.len() as u32).to_le_bytes());
        for (file, groups) in files {
            out.extend_from_slice(&number_of[file].to_le_bytes());
            out.extend_from_slice(&(groups.len() as u32).to_le_bytes());
            for (group, rows) in groups {
                out.extend_from_slice(&group.to_le_bytes());
                out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
                for row in rows {
                    out.extend_from_slice(&row.to_le_bytes());
                }
            }
        }
    }
    out
}

/// Read row-level postings back, or `None` if the bytes are not a
/// well-formed bitmap.
///
/// Same discipline [`decode`] keeps: any inconsistency is a `None`, because
/// a partial bitmap under-selects rows and that is the wrong direction.
fn decode_bitmap(field: FieldId, bytes: &[u8]) -> Option<Bitmap> {
    let mut cursor = Cursor { bytes, at: 0 };

    let path_count = cursor.u32()?;
    let mut paths = Vec::with_capacity(path_count.min(1 << 16) as usize);
    for _ in 0..path_count {
        let len = cursor.u32()? as usize;
        paths.push(FileId(String::from_utf8(cursor.take(len)?.to_vec()).ok()?));
    }

    let values = cursor.u64()?;
    let mut bitmap = Bitmap::new(field);
    for _ in 0..values {
        let value = cursor.u64()?;
        let files = cursor.u32()?;
        for _ in 0..files {
            let file = paths.get(cursor.u32()? as usize)?.clone();
            let groups = cursor.u32()?;
            for _ in 0..groups {
                let group = cursor.u32()?;
                let rows = cursor.u32()?;
                for _ in 0..rows {
                    bitmap.insert(value, file.clone(), group, cursor.u64()?);
                }
            }
        }
    }

    if cursor.at != bytes.len() {
        return None;
    }
    Some(bitmap)
}

/// Serialise a filter set's postings: the path table, then
/// `file → group → rows`. The filter itself travels in blob metadata —
/// it is the piece's identity, and identity does not belong in the data.
fn encode_filter_set(set: &FilterSet) -> Vec<u8> {
    let paths: Vec<&FileId> = set
        .postings()
        .keys()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let number_of: BTreeMap<&FileId, u32> = paths
        .iter()
        .enumerate()
        .map(|(index, file)| (*file, index as u32))
        .collect();

    let mut out = Vec::new();
    out.extend_from_slice(&(paths.len() as u32).to_le_bytes());
    for file in &paths {
        let bytes = file.0.as_bytes();
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }

    out.extend_from_slice(&(set.postings().len() as u32).to_le_bytes());
    for (file, groups) in set.postings() {
        out.extend_from_slice(&number_of[file].to_le_bytes());
        out.extend_from_slice(&(groups.len() as u32).to_le_bytes());
        for (group, rows) in groups {
            out.extend_from_slice(&group.to_le_bytes());
            out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
            for row in rows {
                out.extend_from_slice(&row.to_le_bytes());
            }
        }
    }
    out
}

/// Read filter-set postings back; the filter arrives from the caller,
/// which took it from the blob's metadata.
fn decode_filter_set(filter: Filter, bytes: &[u8]) -> Option<FilterSet> {
    let mut cursor = Cursor { bytes, at: 0 };

    let path_count = cursor.u32()?;
    let mut paths = Vec::with_capacity(path_count.min(1 << 16) as usize);
    for _ in 0..path_count {
        let len = cursor.u32()? as usize;
        paths.push(FileId(String::from_utf8(cursor.take(len)?.to_vec()).ok()?));
    }

    let files = cursor.u32()?;
    let mut set = FilterSet::of(filter);
    for _ in 0..files {
        let file = paths.get(cursor.u32()? as usize)?.clone();
        let groups = cursor.u32()?;
        for _ in 0..groups {
            let group = cursor.u32()?;
            let rows = cursor.u32()?;
            for _ in 0..rows {
                set.insert(file.clone(), group, cursor.u64()?);
            }
        }
    }

    if cursor.at != bytes.len() {
        return None;
    }
    Some(set)
}

/// Serialise a text index's postings: the path table, then
/// `term → files`. Same shape as [`encode`], because the postings are
/// structurally identical — a hash to a file set.
fn encode_text_index(text: &TextIndex) -> Vec<u8> {
    let paths: Vec<&FileId> = text
        .postings()
        .flat_map(|(_, files)| files.iter())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let number_of: BTreeMap<&FileId, u32> = paths
        .iter()
        .enumerate()
        .map(|(index, file)| (*file, index as u32))
        .collect();

    let mut out = Vec::new();
    out.extend_from_slice(&(paths.len() as u32).to_le_bytes());
    for file in &paths {
        let bytes = file.0.as_bytes();
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }

    out.extend_from_slice(&(text.values() as u64).to_le_bytes());
    for (term, files) in text.postings() {
        out.extend_from_slice(&term.to_le_bytes());
        out.extend_from_slice(&(files.len() as u32).to_le_bytes());
        for file in files {
            out.extend_from_slice(&number_of[file].to_le_bytes());
        }
    }
    out
}

/// Read text postings back, or `None` if the bytes are not well-formed.
///
/// Same discipline as [`decode`]: any inconsistency is `None`, because a
/// partial text index under-selects files and that is the wrong direction.
fn decode_text_index(field: FieldId, bytes: &[u8]) -> Option<TextIndex> {
    let mut cursor = Cursor { bytes, at: 0 };

    let path_count = cursor.u32()?;
    let mut paths = Vec::with_capacity(path_count.min(1 << 16) as usize);
    for _ in 0..path_count {
        let len = cursor.u32()? as usize;
        paths.push(FileId(String::from_utf8(cursor.take(len)?.to_vec()).ok()?));
    }

    let terms = cursor.u64()?;
    let mut text = TextIndex::new(field);
    for _ in 0..terms {
        let term = cursor.u64()?;
        let files = cursor.u32()?;
        for _ in 0..files {
            let file = paths.get(cursor.u32()? as usize)?;
            text.insert(term, file.clone());
        }
    }

    if cursor.at != bytes.len() {
        return None;
    }
    Some(text)
}

/// Encode a vector index's metadata: field, metric, dimension.
///
/// The rows themselves are Parquet, written separately; this is only what
/// recovery needs to reconstruct the kind and locate them.
fn encode_vector_index(vector: &VectorIndex) -> Vec<u8> {
    let (field, metric, dimension) = vector.covers();
    let mut out = Vec::new();
    out.extend_from_slice(&field.to_le_bytes());
    out.extend_from_slice(&(metric as u32).to_le_bytes());
    out.extend_from_slice(&dimension.to_le_bytes());
    out
}

/// Read vector metadata back, or `None` if the bytes are not well-formed.
fn decode_vector_index(bytes: &[u8]) -> Option<(FieldId, Metric, u32)> {
    let mut cursor = Cursor { bytes, at: 0 };
    let field = cursor.u32()?;
    let metric = match cursor.u32()? {
        0 => Metric::L2,
        1 => Metric::Cosine,
        _ => return None,
    };
    let dimension = cursor.u32()?;
    if cursor.at != bytes.len() {
        return None;
    }
    Some((field, metric, dimension))
}

/// Encode a join hash's metadata: the join keys and covered columns.
///
/// The rows themselves are Parquet, written separately; this is only what
/// recovery needs to reconstruct the kind.
fn encode_join_hash(hash: &JoinHash) -> Vec<u8> {
    let (keys, columns) = hash.covers();
    let mut out = Vec::new();
    put_fields(&mut out, keys);
    put_fields(&mut out, columns);
    out
}

/// Read join hash metadata back, or `None` if the bytes are not well-formed.
fn decode_join_hash(bytes: &[u8]) -> Option<(BTreeSet<FieldId>, BTreeSet<FieldId>)> {
    let mut cursor = Cursor { bytes, at: 0 };
    let keys = cursor.fields()?;
    let columns = cursor.fields()?;
    if cursor.at != bytes.len() {
        return None;
    }
    Some((keys, columns))
}

/// Extract the stable hash from a join hash's `DerivedId` (`jh:<table>:<hex>`),
/// so `reclaim` can reconstruct the blob path without carrying a separate
/// field through `Retired`.
fn parse_join_hash_id(id: &DerivedId) -> Option<u64> {
    let s = id.0.strip_prefix("jh:")?;
    let hash = s.rsplit_once(':')?.1;
    u64::from_str_radix(hash, 16).ok()
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(len)?;
        let slice = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(slice)
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn string(&mut self) -> Option<String> {
        let len = self.u64()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).ok()
    }

    fn fields(&mut self) -> Option<BTreeSet<FieldId>> {
        let count = self.u64()?;
        let mut fields = BTreeSet::new();
        for _ in 0..count {
            fields.insert(self.u32()?);
        }
        Some(fields)
    }

    fn seen(&mut self) -> Option<Seen> {
        Some(Seen {
            queries: self.u64()?,
            bytes_read: self.u64()?,
            bytes_if_full_scan: self.u64()?,
            helped: self.u64()?,
        })
    }
}

/// Write an index to storage as a Puffin blob.
///
/// Returns the path written. Registering it is the caller's next step, and
/// deliberately not this function's: a blob with no registry entry is an
/// orphan, while a registry entry with no blob is a lie.
pub async fn write_index(
    file_io: &FileIO,
    layout: &Layout,
    table: &TableId,
    at: SnapshotId,
    policy: PolicyFingerprint,
    index: &Index,
) -> iceberg::Result<String> {
    let path = layout.index_path(table, index.field(), at);
    let output = file_io.new_output(&path)?;

    let mut writer = PuffinWriter::new(&output, HashMap::new(), false).await?;
    writer
        .add(
            Blob::builder()
                .r#type(QUARRY_EQ_INDEX_V2.to_owned())
                .fields(vec![index.field() as i32])
                .snapshot_id(at.0)
                .sequence_number(0)
                .data(encode(index))
                .properties(HashMap::from([
                    (HASH_VERSION_PROPERTY.to_owned(), HASH_VERSION.to_string()),
                    (POLICY_PROPERTY.to_owned(), policy.0.to_string()),
                ]))
                .build(),
            // Uncompressed, because `iceberg-rust` 0.6 declares `Lz4` and
            // `Zstd` but returns FeatureUnsupported for both. Postings are
            // hashes and repeated path strings, so they would compress well;
            // this costs bytes on disk and nothing in correctness, and the
            // codec is recorded per blob so a later writer can change it
            // without invalidating anything already written.
            CompressionCodec::None,
        )
        .await?;
    writer.close().await?;
    Ok(path)
}

/// Read one index back, if it is one we can still use.
///
/// Refuses, rather than returns something wrong, when:
///
/// - the blob type is not ours, or is a version we do not know
/// - the hash version differs, so the postings were built by a different
///   hashing of values and would match nothing
/// - the bytes do not decode exactly
///
/// The hash-version check is the reason
/// [`HASH_VERSION`](crate::stable_hash::HASH_VERSION) exists: without it a
/// changed hash produces an index that silently matches nothing, which reads
/// as "mysteriously slower" rather than as a fault.
pub async fn read_index(
    file_io: &FileIO,
    path: &str,
    field: FieldId,
) -> iceberg::Result<Option<(Index, PolicyFingerprint)>> {
    if !is_plausible_puffin(file_io, path).await? {
        return Ok(None);
    }
    let reader = PuffinReader::new(file_io.new_input(path)?);
    let metadata = reader.file_metadata().await?;

    for blob_metadata in metadata.blobs() {
        if blob_metadata.blob_type() != QUARRY_EQ_INDEX_V2 {
            continue;
        }
        let properties = blob_metadata.properties();
        if properties.get(HASH_VERSION_PROPERTY).map(String::as_str)
            != Some(&HASH_VERSION.to_string())
        {
            continue;
        }
        let policy = properties
            .get(POLICY_PROPERTY)
            .and_then(|raw| raw.parse().ok())
            .map(PolicyFingerprint);
        let Some(policy) = policy else {
            continue;
        };

        let blob = reader.blob(blob_metadata).await?;
        if let Some(index) = decode(field, blob.data()) {
            let bytes = blob.data().len() as u64;
            return Ok(Some((index.with_bytes(bytes), policy)));
        }
    }
    Ok(None)
}

/// Write a bitmap to storage as a Puffin blob — [`write_index`]'s contract
/// verbatim, only the blob type and postings differ.
pub async fn write_bitmap(
    file_io: &FileIO,
    layout: &Layout,
    table: &TableId,
    at: SnapshotId,
    policy: PolicyFingerprint,
    bitmap: &Bitmap,
) -> iceberg::Result<String> {
    let path = layout.index_path(table, bitmap.field(), at);
    let output = file_io.new_output(&path)?;

    let mut writer = PuffinWriter::new(&output, HashMap::new(), false).await?;
    writer
        .add(
            Blob::builder()
                .r#type(QUARRY_BITMAP_V1.to_owned())
                .fields(vec![bitmap.field() as i32])
                .snapshot_id(at.0)
                .sequence_number(0)
                .data(encode_bitmap(bitmap))
                .properties(HashMap::from([
                    (HASH_VERSION_PROPERTY.to_owned(), HASH_VERSION.to_string()),
                    (POLICY_PROPERTY.to_owned(), policy.0.to_string()),
                ]))
                .build(),
            CompressionCodec::None,
        )
        .await?;
    writer.close().await?;
    Ok(path)
}

/// Read one bitmap back, with [`read_index`]'s refusals: wrong type, wrong
/// hash version, or bytes that do not decode exactly are all `None`.
pub async fn read_bitmap(
    file_io: &FileIO,
    path: &str,
    field: FieldId,
) -> iceberg::Result<Option<(Bitmap, PolicyFingerprint)>> {
    if !is_plausible_puffin(file_io, path).await? {
        return Ok(None);
    }
    let reader = PuffinReader::new(file_io.new_input(path)?);
    let metadata = reader.file_metadata().await?;

    for blob_metadata in metadata.blobs() {
        if blob_metadata.blob_type() != QUARRY_BITMAP_V1 {
            continue;
        }
        let properties = blob_metadata.properties();
        if properties.get(HASH_VERSION_PROPERTY).map(String::as_str)
            != Some(&HASH_VERSION.to_string())
        {
            continue;
        }
        let policy = properties
            .get(POLICY_PROPERTY)
            .and_then(|raw| raw.parse().ok())
            .map(PolicyFingerprint);
        let Some(policy) = policy else {
            continue;
        };

        let blob = reader.blob(blob_metadata).await?;
        if let Some(bitmap) = decode_bitmap(field, blob.data()) {
            let bytes = blob.data().len() as u64;
            return Ok(Some((bitmap.with_bytes(bytes), policy)));
        }
    }
    Ok(None)
}

/// Write a filter set to storage as a Puffin blob.
///
/// The canonical filter text goes in the blob's properties — it is the
/// piece's identity, so it belongs in the metadata, not the data.
pub async fn write_filter_set(
    file_io: &FileIO,
    layout: &Layout,
    table: &TableId,
    at: SnapshotId,
    policy: PolicyFingerprint,
    set: &FilterSet,
) -> iceberg::Result<String> {
    let path = layout.filter_set_path(table, set.filter(), at);
    let output = file_io.new_output(&path)?;

    let mut writer = PuffinWriter::new(&output, HashMap::new(), false).await?;
    writer
        .add(
            Blob::builder()
                .r#type(QUARRY_FILTER_SET_V1.to_owned())
                .fields(set.filter().field.iter().map(|f| *f as i32).collect())
                .snapshot_id(at.0)
                .sequence_number(0)
                .data(encode_filter_set(set))
                .properties(HashMap::from([
                    (HASH_VERSION_PROPERTY.to_owned(), HASH_VERSION.to_string()),
                    (POLICY_PROPERTY.to_owned(), policy.0.to_string()),
                    (FILTER_TEXT_PROPERTY.to_owned(), set.filter().text.clone()),
                ]))
                .build(),
            CompressionCodec::None,
        )
        .await?;
    writer.close().await?;
    Ok(path)
}

/// Read one filter set back. The filter's field comes from the blob's
/// `fields`; the text from its properties — either missing means the blob
/// is not what the path claimed.
pub async fn read_filter_set(
    file_io: &FileIO,
    path: &str,
) -> iceberg::Result<Option<(FilterSet, PolicyFingerprint)>> {
    if !is_plausible_puffin(file_io, path).await? {
        return Ok(None);
    }
    let reader = PuffinReader::new(file_io.new_input(path)?);
    let metadata = reader.file_metadata().await?;

    for blob_metadata in metadata.blobs() {
        if blob_metadata.blob_type() != QUARRY_FILTER_SET_V1 {
            continue;
        }
        let properties = blob_metadata.properties();
        if properties.get(HASH_VERSION_PROPERTY).map(String::as_str)
            != Some(&HASH_VERSION.to_string())
        {
            continue;
        }
        let Some(policy) = properties
            .get(POLICY_PROPERTY)
            .and_then(|raw| raw.parse().ok())
            .map(PolicyFingerprint)
        else {
            continue;
        };
        let Some(text) = properties.get(FILTER_TEXT_PROPERTY).cloned() else {
            continue;
        };
        let field = match blob_metadata.fields() {
            [] => None,
            [f] => u32::try_from(*f).ok(),
            _ => continue,
        };
        let filter = Filter { field, text };

        let blob = reader.blob(blob_metadata).await?;
        if let Some(set) = decode_filter_set(filter, blob.data()) {
            let bytes = blob.data().len() as u64;
            return Ok(Some((set.with_bytes(bytes), policy)));
        }
    }
    Ok(None)
}

/// Write a text index to storage as a Puffin blob — [`write_index`]'s
/// contract verbatim, only the blob type and postings differ.
///
/// A text index shares the `(table, field, snapshot)` path with equality
/// indexes and bitmaps: one field gets one accelerator, whichever shape it
/// took. Recovery distinguishes them by blob type.
pub async fn write_text_index(
    file_io: &FileIO,
    layout: &Layout,
    table: &TableId,
    at: SnapshotId,
    policy: PolicyFingerprint,
    text: &TextIndex,
) -> iceberg::Result<String> {
    let path = layout.index_path(table, text.field(), at);
    let output = file_io.new_output(&path)?;

    let mut writer = PuffinWriter::new(&output, HashMap::new(), false).await?;
    writer
        .add(
            Blob::builder()
                .r#type(QUARRY_TEXT_INDEX_V1.to_owned())
                .fields(vec![text.field() as i32])
                .snapshot_id(at.0)
                .sequence_number(0)
                .data(encode_text_index(text))
                .properties(HashMap::from([
                    (HASH_VERSION_PROPERTY.to_owned(), HASH_VERSION.to_string()),
                    (POLICY_PROPERTY.to_owned(), policy.0.to_string()),
                ]))
                .build(),
            CompressionCodec::None,
        )
        .await?;
    writer.close().await?;
    Ok(path)
}

/// Read one text index back, with [`read_index`]'s refusals: wrong type,
/// wrong hash version, or bytes that do not decode exactly are all `None`.
pub async fn read_text_index(
    file_io: &FileIO,
    path: &str,
    field: FieldId,
) -> iceberg::Result<Option<(TextIndex, PolicyFingerprint)>> {
    if !is_plausible_puffin(file_io, path).await? {
        return Ok(None);
    }
    let reader = PuffinReader::new(file_io.new_input(path)?);
    let metadata = reader.file_metadata().await?;

    for blob_metadata in metadata.blobs() {
        if blob_metadata.blob_type() != QUARRY_TEXT_INDEX_V1 {
            continue;
        }
        let properties = blob_metadata.properties();
        if properties.get(HASH_VERSION_PROPERTY).map(String::as_str)
            != Some(&HASH_VERSION.to_string())
        {
            continue;
        }
        let policy = properties
            .get(POLICY_PROPERTY)
            .and_then(|raw| raw.parse().ok())
            .map(PolicyFingerprint);
        let Some(policy) = policy else {
            continue;
        };

        let blob = reader.blob(blob_metadata).await?;
        if let Some(text) = decode_text_index(field, blob.data()) {
            let bytes = blob.data().len() as u64;
            return Ok(Some((text.with_bytes(bytes), policy)));
        }
    }
    Ok(None)
}

/// Write a vector index: the rows as Parquet, then the metadata as a Puffin
/// blob beside them.
///
/// The Parquet file is the bulk of the index — the rows the kind substitutes
/// for the scan. The Puffin blob carries only the field, metric, and
/// dimension recovery needs to reconstruct the kind, plus the policy and
/// hash version the same gate as every other kind checks.
///
/// Returns the Puffin metadata path; the rows path is derived from it by
/// swapping `.puffin` for `.parquet`.
pub async fn write_vector_index(
    file_io: &FileIO,
    layout: &Layout,
    table: &TableId,
    at: SnapshotId,
    policy: PolicyFingerprint,
    vector: &VectorIndex,
    rows: &[datafusion::arrow::record_batch::RecordBatch],
) -> iceberg::Result<String> {
    let field = vector.covers().0;
    let rows_path = layout.vector_rows_path(table, field, at);
    let meta_path = layout.index_path(table, field, at);

    // Parquet rows first; the Puffin metadata is the registry entry, and a
    // recovered metadata blob whose rows are absent is discarded.
    let schema = rows
        .first()
        .ok_or_else(|| {
            iceberg::Error::new(
                iceberg::ErrorKind::Unexpected,
                format!("vector index for {table:?} field {field} has no rows"),
            )
        })?
        .schema();
    let mut buffer = Vec::new();
    let mut writer = datafusion::parquet::arrow::ArrowWriter::try_new(&mut buffer, schema, None)?;
    for batch in rows {
        writer.write(batch)?;
    }
    writer.close()?;
    file_io
        .new_output(&rows_path)?
        .write(bytes::Bytes::from(buffer))
        .await?;

    let output = file_io.new_output(&meta_path)?;
    let mut writer = PuffinWriter::new(&output, HashMap::new(), false).await?;
    writer
        .add(
            Blob::builder()
                .r#type(QUARRY_VECTOR_INDEX_V1.to_owned())
                .fields(vec![field as i32])
                .snapshot_id(at.0)
                .sequence_number(0)
                .data(encode_vector_index(vector))
                .properties(HashMap::from([
                    (HASH_VERSION_PROPERTY.to_owned(), HASH_VERSION.to_string()),
                    (POLICY_PROPERTY.to_owned(), policy.0.to_string()),
                ]))
                .build(),
            CompressionCodec::None,
        )
        .await?;
    writer.close().await?;
    Ok(meta_path)
}

/// Read a vector index's metadata blob, returning the kind and the path to
/// its Parquet rows.
///
/// `None` for wrong type, wrong hash version, missing policy, or bytes that
/// do not decode exactly — same discipline as every other kind.
pub async fn read_vector_index(
    file_io: &FileIO,
    path: &str,
) -> iceberg::Result<Option<(FieldId, Metric, u32, PolicyFingerprint, String)>> {
    if !is_plausible_puffin(file_io, path).await? {
        return Ok(None);
    }
    let reader = PuffinReader::new(file_io.new_input(path)?);
    let metadata = reader.file_metadata().await?;

    for blob_metadata in metadata.blobs() {
        if blob_metadata.blob_type() != QUARRY_VECTOR_INDEX_V1 {
            continue;
        }
        let properties = blob_metadata.properties();
        if properties.get(HASH_VERSION_PROPERTY).map(String::as_str)
            != Some(&HASH_VERSION.to_string())
        {
            continue;
        }
        let policy = properties
            .get(POLICY_PROPERTY)
            .and_then(|raw| raw.parse().ok())
            .map(PolicyFingerprint);
        let Some(policy) = policy else {
            continue;
        };

        let blob = reader.blob(blob_metadata).await?;
        let Some((field, metric, dimension)) = decode_vector_index(blob.data()) else {
            continue;
        };
        let rows_path = path.strip_suffix(".puffin").map(|p| format!("{p}.parquet"));
        let Some(rows_path) = rows_path else {
            continue;
        };
        return Ok(Some((field, metric, dimension, policy, rows_path)));
    }
    Ok(None)
}

/// The path `store` addresses a `FileIO` location by — the scheme and
/// authority stripped, any leading slash dropped.
fn store_path(file_io_path: &str) -> String {
    crate::from_iceberg::object_path(file_io_path)
        .trim_start_matches('/')
        .to_owned()
}

/// The Arrow schema a Parquet object was written with.
///
/// Only the footer is read — a few kilobytes at the file's tail, which
/// `size` lets the reader seek straight to. `None` for a file that is
/// missing or does not read as Parquet, which for derived rows means the
/// state they belong to cannot serve.
async fn parquet_schema(store: &Arc<dyn ObjectStore>, path: &str, size: u64) -> Option<SchemaRef> {
    let mut reader =
        ParquetObjectReader::new(Arc::clone(store), ObjectPath::from(path)).with_file_size(size);
    let metadata = ArrowReaderMetadata::load_async(&mut reader, Default::default())
        .await
        .ok()?;
    Some(Arc::clone(metadata.schema()))
}

/// The [`RowFiles`] descriptor for a piece of derived state's persisted
/// rows.
///
/// Every path is `head`ed — a missing one is a hard failure, since a
/// substitute holding partial rows would answer wrongly. The first file's
/// footer supplies the schema the rows were written with; the footer read
/// doubles as a check that the file is Parquet at all.
///
/// `None` when any file is missing or unreadable — the caller discards the
/// state rather than registering a substitute with nothing to serve.
async fn parquet_row_files(
    store: &Arc<dyn ObjectStore>,
    layout: &Layout,
    rows_paths: &[String],
) -> Option<(RowFiles, u64)> {
    let mut files = Vec::with_capacity(rows_paths.len());
    let mut schema = None;
    let mut total = 0;
    for path in rows_paths {
        let object = store_path(path);
        let meta = store.head(&ObjectPath::from(object.as_str())).await.ok()?;
        if schema.is_none() {
            schema = parquet_schema(store, &object, meta.size).await;
        }
        total += meta.size;
        files.push((object, meta.size));
    }
    Some((
        RowFiles {
            url: layout.object_store_url(),
            files,
            schema: schema?,
        },
        total,
    ))
}

/// Read the Parquet rows a vector index stored, as `RecordBatch`es.
///
/// Called after [`read_vector_index`] locates the rows path. Returns `None`
/// if the file is missing or unreadable — the caller discards the index.
pub async fn read_vector_rows(
    store: &Arc<dyn ObjectStore>,
    rows_path: &str,
) -> Option<Vec<datafusion::arrow::record_batch::RecordBatch>> {
    let path = ObjectPath::from(rows_path);
    let reader =
        datafusion::parquet::arrow::async_reader::ParquetObjectReader::new(Arc::clone(store), path);
    let builder =
        datafusion::parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .ok()?;
    let stream = builder.build().ok()?;
    use futures::TryStreamExt;
    stream.try_collect().await.ok()
}

/// How many shards `bytes` of rows should be written across.
///
/// One shard per `shard_bytes`, bounded below by one and above by
/// [`JOIN_HASH_MAX_SHARDS`].
fn join_shard_count(bytes: u64, shard_bytes: u64) -> u32 {
    (bytes.div_ceil(shard_bytes.max(1))).clamp(1, JOIN_HASH_MAX_SHARDS) as u32
}

/// Split `rows` into `shards` buckets by join key.
///
/// Every row lands in exactly one bucket — the key columns' values hashed
/// with [`StableHasher`], the same hash the blob's `quarry.hash-version`
/// stamps. The bucket boundaries are a partition of the key space, which is
/// what makes a shard independently servable: it holds every stored row
/// whose key falls in its range.
fn shard_rows(
    hash: &JoinHash,
    rows: &[RecordBatch],
    field_ids: &BTreeMap<String, FieldId>,
    shards: u32,
) -> iceberg::Result<Vec<Vec<RecordBatch>>> {
    let (keys, _) = hash.covers();
    let Some(schema) = rows.first().map(|b| b.schema()) else {
        return Ok(Vec::new());
    };
    // The batch columns that are join keys, in column order.
    let key_positions: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| field_ids.get(f.name()).is_some_and(|id| keys.contains(id)))
        .map(|(i, _)| i)
        .collect();
    if key_positions.len() != keys.len() {
        return Err(iceberg::Error::new(
            iceberg::ErrorKind::Unexpected,
            format!(
                "join hash covers {} keys but the rows hold {}",
                keys.len(),
                key_positions.len()
            ),
        ));
    }

    let mut out: Vec<Vec<RecordBatch>> = vec![Vec::new(); shards as usize];
    for batch in rows {
        // Bucket each row once, then take its columns per shard.
        let mut bucket_rows: Vec<Vec<u64>> = vec![Vec::new(); shards as usize];
        for row in 0..batch.num_rows() {
            let mut hasher = StableHasher::new();
            for &position in &key_positions {
                let value =
                    ScalarValue::try_from_array(batch.column(position), row).map_err(|e| {
                        iceberg::Error::new(
                            iceberg::ErrorKind::Unexpected,
                            format!("cannot read join key at row {row}: {e}"),
                        )
                    })?;
                value.hash(&mut hasher);
            }
            bucket_rows[(hasher.finish() % shards as u64) as usize].push(row as u64);
        }
        for (shard, rows) in bucket_rows.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let indices = UInt64Array::from(rows);
            let columns = batch
                .columns()
                .iter()
                .map(|column| take(column, &indices, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    iceberg::Error::new(
                        iceberg::ErrorKind::Unexpected,
                        format!("cannot take shard rows: {e}"),
                    )
                })?;
            out[shard].push(
                RecordBatch::try_new(Arc::clone(&batch.schema()), columns).map_err(|e| {
                    iceberg::Error::new(
                        iceberg::ErrorKind::Unexpected,
                        format!("cannot build shard batch: {e}"),
                    )
                })?,
            );
        }
    }
    Ok(out)
}

/// Write a join hash: the rows as Parquet shards, then the metadata as a
/// Puffin blob beside them.
///
/// Same contract as [`write_vector_index`]: the Parquet files are the bulk
/// of the index — the build-side rows the kind substitutes for the scan.
/// The Puffin blob carries the join keys and covered columns recovery needs
/// to reconstruct the kind, the policy and hash version, and the shard
/// count the rows were written across.
///
/// Rows are bucketed by join-key hash into `shard_bytes`-sized shards, so a
/// large build side is a set of files read in parallel rather than one
/// object — and each shard holds every stored row of a key range, which is
/// what keeps a missing shard a hard failure rather than a partial answer.
///
/// The path key is the same stable hash [`super::build::join_hash_id_from`]
/// computes, so the blob lands where recovery and `reclaim` look for it.
///
/// Returns the Puffin metadata path and each written shard's `(path, size)`.
#[allow(clippy::too_many_arguments)]
pub async fn write_join_hash(
    file_io: &FileIO,
    layout: &Layout,
    table: &TableId,
    at: SnapshotId,
    policy: PolicyFingerprint,
    hash: &JoinHash,
    rows: &[RecordBatch],
    field_ids: &BTreeMap<String, FieldId>,
    shard_bytes: u64,
) -> iceberg::Result<(String, Vec<(String, u64)>)> {
    let (keys, columns) = hash.covers();
    let id = super::build::join_hash_id_from(table, keys, columns);
    let id_hash = parse_join_hash_id(&id).ok_or_else(|| {
        iceberg::Error::new(
            iceberg::ErrorKind::Unexpected,
            format!("join hash id {id:?} does not carry a hash"),
        )
    })?;
    let meta_path = layout.join_hash_path(table, id_hash, at);

    let schema = rows
        .first()
        .ok_or_else(|| {
            iceberg::Error::new(
                iceberg::ErrorKind::Unexpected,
                format!("join hash for {table:?} id {id_hash:016x} has no rows"),
            )
        })?
        .schema();
    let total: u64 = rows.iter().map(|b| b.get_array_memory_size() as u64).sum();
    let shards = join_shard_count(total, shard_bytes);
    let sharded = shard_rows(hash, rows, field_ids, shards)?;

    // Parquet shards first; the Puffin metadata is the registry entry, and
    // a recovered metadata blob whose rows are absent is discarded.
    let mut written = Vec::with_capacity(shards as usize);
    for (shard, batches) in sharded.into_iter().enumerate() {
        let path = layout.join_hash_shard_path(table, id_hash, at, shard as u32);
        let mut buffer = Vec::new();
        let mut writer = datafusion::parquet::arrow::ArrowWriter::try_new(
            &mut buffer,
            Arc::clone(&schema),
            None,
        )?;
        for batch in &batches {
            writer.write(batch)?;
        }
        writer.close()?;
        let size = buffer.len() as u64;
        file_io
            .new_output(&path)?
            .write(bytes::Bytes::from(buffer))
            .await?;
        written.push((path, size));
    }

    let output = file_io.new_output(&meta_path)?;
    let mut writer = PuffinWriter::new(&output, HashMap::new(), false).await?;
    writer
        .add(
            Blob::builder()
                .r#type(QUARRY_JOIN_HASH_V1.to_owned())
                .fields(Vec::new())
                .snapshot_id(at.0)
                .sequence_number(0)
                .data(encode_join_hash(hash))
                .properties(HashMap::from([
                    (HASH_VERSION_PROPERTY.to_owned(), HASH_VERSION.to_string()),
                    (POLICY_PROPERTY.to_owned(), policy.0.to_string()),
                    (SHARDS_PROPERTY.to_owned(), shards.to_string()),
                ]))
                .build(),
            CompressionCodec::None,
        )
        .await?;
    writer.close().await?;
    Ok((meta_path, written))
}

/// Read a join hash's metadata blob, returning the keys, columns, policy,
/// and the paths to its Parquet rows.
///
/// `None` for wrong type, wrong hash version, missing policy, a shard count
/// that does not parse, or bytes that do not decode exactly — same
/// discipline as every other kind.
pub async fn read_join_hash(
    file_io: &FileIO,
    path: &str,
) -> iceberg::Result<
    Option<(
        BTreeSet<FieldId>,
        BTreeSet<FieldId>,
        PolicyFingerprint,
        Vec<String>,
    )>,
> {
    if !is_plausible_puffin(file_io, path).await? {
        return Ok(None);
    }
    let reader = PuffinReader::new(file_io.new_input(path)?);
    let metadata = reader.file_metadata().await?;

    for blob_metadata in metadata.blobs() {
        if blob_metadata.blob_type() != QUARRY_JOIN_HASH_V1 {
            continue;
        }
        let properties = blob_metadata.properties();
        if properties.get(HASH_VERSION_PROPERTY).map(String::as_str)
            != Some(&HASH_VERSION.to_string())
        {
            continue;
        }
        let policy = properties
            .get(POLICY_PROPERTY)
            .and_then(|raw| raw.parse().ok())
            .map(PolicyFingerprint);
        let Some(policy) = policy else {
            continue;
        };

        let blob = reader.blob(blob_metadata).await?;
        let Some((keys, columns)) = decode_join_hash(blob.data()) else {
            continue;
        };
        let Some(stem) = path.strip_suffix(".puffin") else {
            continue;
        };
        // The shard count the writer recorded; absent means the blob
        // predates sharding and its single rows file sits at `{snap}.parquet`.
        let rows_paths = match properties.get(SHARDS_PROPERTY) {
            None => vec![format!("{stem}.parquet")],
            Some(raw) => {
                let Ok(shards) = raw.parse::<u32>() else {
                    continue;
                };
                if shards == 0 {
                    continue;
                }
                (0..shards)
                    .map(|shard| format!("{stem}.{shard:04}.parquet"))
                    .collect()
            }
        };
        return Ok(Some((keys, columns, policy, rows_paths)));
    }
    Ok(None)
}

/// The blob type for a saved workload.
pub const QUARRY_WORKLOAD_V1: &str = "quarry-workload-v1";

/// Serialise a workload: shapes, then credits.
fn encode_workload(workload: &Workload) -> Vec<u8> {
    let mut out = Vec::new();

    let shapes: Vec<_> = workload.shapes_seen().collect();
    out.extend_from_slice(&(shapes.len() as u64).to_le_bytes());
    for (fingerprint, seen) in shapes {
        put_str(&mut out, &fingerprint.table.0);
        put_fields(&mut out, &fingerprint.probeable);
        put_fields(&mut out, &fingerprint.opaque);
        put_fields(&mut out, &fingerprint.projected);
        put_seen(&mut out, seen);
    }

    let credits: Vec<_> = workload.credits().collect();
    out.extend_from_slice(&(credits.len() as u64).to_le_bytes());
    for (id, seen) in credits {
        put_str(&mut out, &id.0);
        put_seen(&mut out, seen);
    }
    out
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_fields(out: &mut Vec<u8>, fields: &BTreeSet<FieldId>) {
    out.extend_from_slice(&(fields.len() as u64).to_le_bytes());
    for field in fields {
        out.extend_from_slice(&field.to_le_bytes());
    }
}

fn put_seen(out: &mut Vec<u8>, seen: &Seen) {
    out.extend_from_slice(&seen.queries.to_le_bytes());
    out.extend_from_slice(&seen.bytes_read.to_le_bytes());
    out.extend_from_slice(&seen.bytes_if_full_scan.to_le_bytes());
    out.extend_from_slice(&seen.helped.to_le_bytes());
}

/// Read a workload back, or `None` if the bytes are not one.
///
/// Same discipline as the index decoder: nothing partial. A half-read
/// workload would under-report what derived state has saved, and
/// under-reporting savings is what causes a useful index to be retired.
fn decode_workload(bytes: &[u8]) -> Option<Workload> {
    let mut cursor = Cursor { bytes, at: 0 };

    let shape_count = cursor.u64()?;
    let mut shapes = Vec::new();
    for _ in 0..shape_count {
        let table = TableId(cursor.string()?);
        let probeable = cursor.fields()?;
        let opaque = cursor.fields()?;
        let projected = cursor.fields()?;
        shapes.push((
            Fingerprint {
                table,
                probeable,
                opaque,
                projected,
            },
            cursor.seen()?,
        ));
    }

    let credit_count = cursor.u64()?;
    let mut credits = Vec::new();
    for _ in 0..credit_count {
        credits.push((DerivedId(cursor.string()?), cursor.seen()?));
    }

    if cursor.at != bytes.len() {
        return None;
    }
    Some(Workload::restore(shapes, credits))
}

/// Write the workload down.
///
/// One object per table prefix. Concurrent optimizers over the same storage
/// overwrite each other, which is why
/// [`retire_after_queries`](crate::workload::Policy::retire_after_queries) exists:
/// losing telemetry must not be able to delete a useful index.
pub async fn write_workload(
    file_io: &FileIO,
    layout: &Layout,
    table: &TableId,
    workload: &Workload,
) -> iceberg::Result<String> {
    let path = layout.workload_path(table);
    let output = file_io.new_output(&path)?;

    let mut writer = PuffinWriter::new(&output, HashMap::new(), false).await?;
    writer
        .add(
            Blob::builder()
                .r#type(QUARRY_WORKLOAD_V1.to_owned())
                .fields(Vec::new())
                .snapshot_id(0)
                .sequence_number(0)
                .data(encode_workload(workload))
                .properties(HashMap::from([(
                    HASH_VERSION_PROPERTY.to_owned(),
                    HASH_VERSION.to_string(),
                )]))
                .build(),
            CompressionCodec::None,
        )
        .await?;
    writer.close().await?;
    Ok(path)
}

/// Read the workload back, or `None` if there is none to read.
pub async fn read_workload(
    file_io: &FileIO,
    layout: &Layout,
    table: &TableId,
) -> iceberg::Result<Option<Workload>> {
    let path = layout.workload_path(table);
    if !file_io.exists(&path).await.unwrap_or(false) {
        return Ok(None);
    }
    if !is_plausible_puffin(file_io, &path).await? {
        return Ok(None);
    }

    let reader = PuffinReader::new(file_io.new_input(&path)?);
    let metadata = reader.file_metadata().await?;
    for blob_metadata in metadata.blobs() {
        if blob_metadata.blob_type() != QUARRY_WORKLOAD_V1 {
            continue;
        }
        // Shape fingerprints hold field ids, not hashes, so they do not
        // depend on the hash version. Credits are keyed on derived ids, which
        // do not either. The stamp is recorded anyway, so that a future
        // format change has a version to refuse on.
        let blob = reader.blob(blob_metadata).await?;
        return Ok(decode_workload(blob.data()));
    }
    Ok(None)
}

/// Puffin's file magic, `PFA1`.
const PUFFIN_MAGIC: [u8; 4] = [0x50, 0x46, 0x41, 0x31];
/// Leading magic, plus the footer's payload-length, flags and trailing magic.
const PUFFIN_MINIMUM_LENGTH: u64 = 4 + 4 + 4 + 4;

/// Whether a file is structurally a Puffin file, checked before parsing it.
///
/// This exists because of a real defect in `iceberg-rust` 0.6, not out of
/// caution. Its footer reader computes
///
/// ```text
/// start = input_file_length - footer_length
/// ```
///
/// where `footer_length` comes from four bytes read out of the file. On a
/// truncated object those bytes are whatever happened to land there, the
/// subtraction underflows, and the process **panics** — in release builds it
/// wraps instead and asks for an absurd range. Either way a half-written
/// object takes down an engine rather than being skipped, which is not an
/// acceptable failure mode for state that is meant to be disposable.
///
/// So the footer is validated first, with two small ranged reads: the magic at
/// both ends, and a declared footer length that actually fits inside the file.
/// That covers truncation and gross corruption. It is not a checksum and does
/// not claim to be — the decoder's exact-length check is the second line.
async fn is_plausible_puffin(file_io: &FileIO, path: &str) -> iceberg::Result<bool> {
    let input = file_io.new_input(path)?;
    let length = input.metadata().await?.size;
    if length < PUFFIN_MINIMUM_LENGTH {
        return Ok(false);
    }

    let reader = input.reader().await?;
    if reader.read(0..4).await?.as_ref() != PUFFIN_MAGIC {
        return Ok(false);
    }
    if reader.read(length - 4..length).await?.as_ref() != PUFFIN_MAGIC {
        return Ok(false);
    }

    // The footer struct is the last twelve bytes: payload length, flags,
    // magic. The payload precedes it.
    let declared = reader.read(length - 12..length - 8).await?;
    let payload: u32 = u32::from_le_bytes(declared.as_ref().try_into().map_err(|_| {
        iceberg::Error::new(
            iceberg::ErrorKind::DataInvalid,
            "puffin footer payload length is not four bytes",
        )
    })?);

    // 4 leading magic + payload + 12 footer struct must fit.
    Ok(4 + payload as u64 + 12 <= length)
}

/// What was recovered, and what was found but could not be used.
#[derive(Debug, Default)]
pub struct Recovered {
    /// Derived state loaded and ready to register.
    pub registry: Registry,
    /// What each recovered piece indexes, so the optimizer can refresh it.
    pub fields: std::collections::BTreeMap<DerivedId, FieldId>,
    /// Where a substituting kind's rows live, keyed by derived id.
    ///
    /// The rows stay on the store: the descriptor names the files and the
    /// schema they were written with, so the caller can
    /// `QuarryTable::store_row_files` and let the substitute stream them
    /// rather than paying recovery for rows it may never read. Kinds that
    /// need no rows (equality, bitmap, text, filter set) leave this empty.
    pub row_files: BTreeMap<DerivedId, RowFiles>,
    /// Objects that are ours but unusable, and should be deleted.
    ///
    /// A stale hash version, a superseded snapshot, or corrupt bytes. Not an
    /// error: derived state is disposable, and the right response is to drop
    /// it and let the loop rebuild.
    pub discarded: Vec<String>,
}

/// Rebuild the registry for one table by listing what is in storage.
///
/// No side store and no catalog read: the paths carry the identities. Only
/// indexes built from a snapshot the graph still knows are kept, since the
/// rule would refuse the rest anyway.
pub async fn recover(
    file_io: &FileIO,
    store: &Arc<dyn ObjectStore>,
    layout: &Layout,
    table: &TableId,
    known_snapshots: &[SnapshotId],
) -> iceberg::Result<Recovered> {
    let mut recovered = Recovered::default();

    // `.puffin` paths that registered, in object-store space, and `.parquet`
    // payloads seen along the way. A rows file whose metadata blob did not
    // register is an orphan — a crash between the two writes, or a blob
    // removed by other means — and is discarded at the end.
    let mut kept: BTreeSet<String> = BTreeSet::new();
    let mut stray_rows: Vec<String> = Vec::new();

    for path in list_paths(store, &layout.object_prefix(table)).await {
        if path.ends_with(".parquet") {
            stray_rows.push(path);
            continue;
        }
        // Two path shapes: a field id names an index or bitmap, an
        // `fset-<hash>` segment names a filter set.
        if let Some((found_table, field, at)) = layout.parse(&path) {
            if &found_table != table {
                continue;
            }
            // Listing gave an object-store path; reading and deleting both
            // need the FileIO one. Recording the wrong form here made
            // `discard` report success for deletes that hit nothing, because
            // object stores treat deleting an absent key as a no-op.
            let readable = layout.index_path(table, field, at);

            if !known_snapshots.contains(&at) {
                // Built from a snapshot the table no longer retains, so the
                // rule could not admit it. Disposable; drop it.
                recovered.discarded.push(readable);
                continue;
            }
            let read = match read_index(file_io, &readable, field).await? {
                Some((index, policy)) => {
                    let bytes = index.bytes_estimate();
                    Some((
                        Box::new(index) as Box<dyn crate::derived::Kind>,
                        bytes,
                        policy,
                    ))
                }
                None => match read_bitmap(file_io, &readable, field).await? {
                    Some((bitmap, policy)) => {
                        let bytes = bitmap.bytes_estimate();
                        Some((
                            Box::new(bitmap) as Box<dyn crate::derived::Kind>,
                            bytes,
                            policy,
                        ))
                    }
                    None => match read_text_index(file_io, &readable, field).await? {
                        Some((text, policy)) => {
                            let bytes = text.bytes_estimate();
                            Some((
                                Box::new(text) as Box<dyn crate::derived::Kind>,
                                bytes,
                                policy,
                            ))
                        }
                        None => None,
                    },
                },
            };
            match read {
                Some((kind, bytes, policy)) => {
                    let id = super::index_id(table, field);
                    recovered.fields.insert(id.clone(), field);
                    kept.insert(store_path(&readable));
                    recovered.registry.register(Derived::new(
                        id,
                        Source {
                            table: table.clone(),
                            snapshot: at,
                        },
                        policy,
                        bytes,
                        kind,
                    ));
                }
                None => match read_vector_index(file_io, &readable).await? {
                    Some((field, metric, dimension, policy, rows_path)) => {
                        let Some((files, bytes)) =
                            parquet_row_files(store, layout, std::slice::from_ref(&rows_path))
                                .await
                        else {
                            recovered.discarded.push(readable);
                            recovered.discarded.push(rows_path);
                            continue;
                        };
                        let id = super::index_id(table, field);
                        recovered.fields.insert(id.clone(), field);
                        recovered.row_files.insert(id.clone(), files);
                        kept.insert(store_path(&readable));
                        recovered.registry.register(Derived::new(
                            id,
                            Source {
                                table: table.clone(),
                                snapshot: at,
                            },
                            policy,
                            bytes,
                            Box::new(VectorIndex::new(field, metric, dimension, bytes)),
                        ));
                    }
                    None => recovered.discarded.push(readable),
                },
            }
        } else if let Some((found_table, _hash, at)) = layout.parse_filter_set(&path) {
            if &found_table != table {
                continue;
            }
            let readable = layout.filter_set_path_for(table, _hash, at);
            if !known_snapshots.contains(&at) {
                recovered.discarded.push(readable);
                continue;
            }
            match read_filter_set(file_io, &readable).await? {
                Some((set, policy)) => {
                    let bytes = set.bytes_estimate();
                    let id = super::filter_set_id(table, set.filter());
                    kept.insert(store_path(&readable));
                    recovered.registry.register(Derived::new(
                        id,
                        Source {
                            table: table.clone(),
                            snapshot: at,
                        },
                        policy,
                        bytes,
                        Box::new(set),
                    ));
                }
                None => recovered.discarded.push(readable),
            }
        } else if let Some((found_table, id_hash, at)) = layout.parse_join_hash(&path) {
            if &found_table != table {
                continue;
            }
            let readable = layout.join_hash_path(table, id_hash, at);
            if !known_snapshots.contains(&at) {
                // The row files need no naming: the sweep below discards any
                // `.parquet` whose `.puffin` was not registered.
                recovered.discarded.push(readable);
                continue;
            }
            match read_join_hash(file_io, &readable).await? {
                Some((keys, columns, policy, rows_paths)) => {
                    let id = super::join_hash_id_from(table, &keys, &columns);
                    match parquet_row_files(store, layout, &rows_paths).await {
                        Some((files, bytes)) => {
                            recovered.row_files.insert(id.clone(), files);
                            kept.insert(store_path(&readable));
                            recovered.registry.register(Derived::new(
                                id,
                                Source {
                                    table: table.clone(),
                                    snapshot: at,
                                },
                                policy,
                                bytes,
                                Box::new(JoinHash::covering(keys, columns, bytes)),
                            ));
                        }
                        // A shard missing or unreadable is a hard failure:
                        // partial rows would answer joins wrongly.
                        None => {
                            recovered.discarded.push(readable);
                            recovered.discarded.extend(rows_paths);
                        }
                    }
                }
                None => recovered.discarded.push(readable),
            }
        }
    }

    // Orphan sweep: a `.parquet` under the prefix is a rows payload for the
    // `.puffin` of the same stem (`{snap}.parquet`, or `{snap}.{i}.parquet`
    // for a shard). One whose blob did not register is dropped — it could
    // only ever be read through that blob.
    for path in stray_rows {
        let Some(stem) = path.strip_suffix(".parquet") else {
            continue;
        };
        let Some((dir, name)) = stem.rsplit_once('/') else {
            continue;
        };
        let Some(snap) = name.split('.').next().filter(|s| !s.is_empty()) else {
            continue;
        };
        if kept.contains(&format!("{dir}/{snap}.puffin")) {
            continue;
        }
        let Some(file_io_path) = layout.file_io_path(table, &path) else {
            continue;
        };
        recovered.discarded.push(file_io_path);
    }
    Ok(recovered)
}

/// Object paths under a prefix.
///
/// A listing failure yields nothing rather than an error: a missing prefix is
/// the normal state of a table that has never been optimized, and the caller's
/// response to either is the same — start from scratch.
async fn list_paths(store: &dyn ObjectStore, prefix: &str) -> Vec<String> {
    use futures::StreamExt;

    let mut paths = Vec::new();
    let mut listing = store.list(Some(&ObjectPath::from(prefix)));
    while let Some(entry) = listing.next().await {
        if let Ok(meta) = entry {
            paths.push(meta.location.to_string());
        }
    }
    paths.sort();
    paths
}

/// Delete objects that are ours but unusable.
///
/// Separate from [`recover`] so that recovery is read-only and a caller can
/// inspect what would be removed before removing it.
///
/// The count is of objects that existed and are now gone, not of calls that
/// returned `Ok`. Deleting an absent key succeeds on every object store, so
/// counting successes would report confident progress while deleting nothing
/// — which is exactly how the wrong path form went unnoticed.
pub async fn discard(file_io: &FileIO, paths: &[String]) -> iceberg::Result<usize> {
    let mut removed = 0;
    for path in paths {
        if !file_io.exists(path).await.unwrap_or(false) {
            continue;
        }
        if file_io.delete(path).await.is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Property carrying the path an advertised piece's bytes live at.
///
/// A manifest blob has no payload; it points at the file that does. Entries
/// lacking this property are another engine's statistics, not ours.
pub const QUARRY_PATH_PROPERTY: &str = "quarry.path";

/// One piece of derived state to advertise in a snapshot's statistics.
#[derive(Clone, Debug)]
pub struct Advertised {
    /// The blob type a reader would find at `path`.
    pub kind: String,
    /// The field ids the piece covers.
    pub fields: Vec<i32>,
    /// Where the bytes are, in `FileIO` path space.
    pub path: String,
    /// Whatever the real blob recorded — policy, hash version.
    pub properties: HashMap<String, String>,
}

/// What a published [`StatisticsFile`] advertises.
///
/// Reads `blob_metadata` directly rather than the file: the catalog's copy
/// is the authority, so discovery costs no read at all.
pub fn advertised(stats: &StatisticsFile) -> Vec<Advertised> {
    stats
        .blob_metadata
        .iter()
        .filter_map(|blob| {
            let path = blob.properties.get(QUARRY_PATH_PROPERTY)?.clone();
            Some(Advertised {
                kind: blob.r#type.clone(),
                fields: blob.fields.clone(),
                path,
                properties: blob.properties.clone(),
            })
        })
        .collect()
}

/// Write a manifest advertising `entries` for `at`, and return the
/// [`StatisticsFile`] committing it needs.
///
/// Statistics give a snapshot one file, so a `prior` file's blobs are carried
/// forward first: replacing it without them would delete another engine's
/// statistics rather than add ours.
pub async fn write_manifest(
    file_io: &FileIO,
    layout: &Layout,
    table: &TableId,
    at: SnapshotId,
    entries: &[Advertised],
    prior: Option<&StatisticsFile>,
) -> iceberg::Result<StatisticsFile> {
    let path = layout.statistics_path(table, at);
    let output = file_io.new_output(&path)?;
    let mut writer = PuffinWriter::new(&output, HashMap::new(), false).await?;

    if let Some(stats) = prior {
        if file_io.exists(&stats.statistics_path).await? {
            let reader = PuffinReader::new(file_io.new_input(&stats.statistics_path)?);
            let metadata = reader.file_metadata().await?;
            for prior_blob in metadata.blobs() {
                let blob = reader.blob(prior_blob).await?;
                writer.add(blob, prior_blob.compression_codec()).await?;
            }
        }
    }

    for entry in entries {
        let mut properties = entry.properties.clone();
        properties.insert(QUARRY_PATH_PROPERTY.to_owned(), entry.path.clone());
        writer
            .add(
                Blob::builder()
                    .r#type(entry.kind.clone())
                    .fields(entry.fields.clone())
                    .snapshot_id(at.0)
                    .sequence_number(0)
                    .data(Vec::new())
                    .properties(properties)
                    .build(),
                CompressionCodec::None,
            )
            .await?;
    }
    writer.close().await?;

    let input = file_io.new_input(&path)?;
    let size = input.metadata().await?.size as i64;
    let reader = PuffinReader::new(input);
    let metadata = reader.file_metadata().await?;
    let blobs_end = metadata
        .blobs()
        .iter()
        .map(|blob| blob.offset() + blob.length())
        .max()
        .unwrap_or(4) as i64;
    Ok(StatisticsFile {
        snapshot_id: at.0,
        statistics_path: path,
        file_size_in_bytes: size,
        file_footer_size_in_bytes: size - blobs_end,
        key_metadata: None,
        blob_metadata: metadata
            .blobs()
            .iter()
            .map(|blob| iceberg::spec::BlobMetadata {
                r#type: blob.blob_type().to_owned(),
                snapshot_id: blob.snapshot_id(),
                sequence_number: blob.sequence_number(),
                fields: blob.fields().to_vec(),
                properties: blob.properties().clone(),
            })
            .collect(),
    })
}

/// Shared handle to a table's on-disk derived state.
#[derive(Clone, Debug)]
pub struct Store {
    file_io: Arc<FileIO>,
    store: Arc<dyn ObjectStore>,
    layout: Layout,
    join_shard_bytes: u64,
}

impl Store {
    /// Keep derived state under `layout`.
    ///
    /// Both stacks are required and for different reasons: `file_io` reads and
    /// writes Puffin blobs, `store` lists them, because `FileIO` cannot.
    pub fn new(file_io: Arc<FileIO>, store: Arc<dyn ObjectStore>, layout: Layout) -> Self {
        Store {
            file_io,
            store,
            layout,
            join_shard_bytes: JOIN_HASH_SHARD_BYTES,
        }
    }

    /// Size join-hash shards at `bytes` — mostly for tests; the default
    /// targets 128 MiB per shard.
    pub fn with_join_shard_bytes(mut self, bytes: u64) -> Self {
        self.join_shard_bytes = bytes;
        self
    }

    /// The layout in use.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The object store in use, for reading Parquet rows.
    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// Write an index and report where it went.
    pub async fn write(
        &self,
        table: &TableId,
        at: SnapshotId,
        policy: PolicyFingerprint,
        index: &Index,
    ) -> iceberg::Result<String> {
        write_index(&self.file_io, &self.layout, table, at, policy, index).await
    }

    /// Write a bitmap and report where it went — [`Store::write`]'s contract
    /// for the row-level kind. Both share the `(table, field, snapshot)`
    /// path: one field gets one accelerator, whichever shape it took.
    pub async fn write_bitmap(
        &self,
        table: &TableId,
        at: SnapshotId,
        policy: PolicyFingerprint,
        bitmap: &Bitmap,
    ) -> iceberg::Result<String> {
        write_bitmap(&self.file_io, &self.layout, table, at, policy, bitmap).await
    }

    /// Persist a filter set beside the table it indexes.
    pub async fn write_filter_set(
        &self,
        table: &TableId,
        at: SnapshotId,
        policy: PolicyFingerprint,
        set: &FilterSet,
    ) -> iceberg::Result<String> {
        write_filter_set(&self.file_io, &self.layout, table, at, policy, set).await
    }

    /// Persist a text index beside the table it indexes.
    pub async fn write_text_index(
        &self,
        table: &TableId,
        at: SnapshotId,
        policy: PolicyFingerprint,
        text: &TextIndex,
    ) -> iceberg::Result<String> {
        write_text_index(&self.file_io, &self.layout, table, at, policy, text).await
    }

    /// Persist a vector index: the rows as Parquet, the metadata as a Puffin
    /// blob beside them.
    pub async fn write_vector_index(
        &self,
        table: &TableId,
        at: SnapshotId,
        policy: PolicyFingerprint,
        vector: &VectorIndex,
        rows: &[datafusion::arrow::record_batch::RecordBatch],
    ) -> iceberg::Result<String> {
        write_vector_index(&self.file_io, &self.layout, table, at, policy, vector, rows).await
    }

    /// Write a join hash down: rows as Parquet shards, metadata as a Puffin
    /// blob.
    ///
    /// The path key is derived from the join shape the hash covers, so the
    /// blob lands where recovery and `reclaim` look for it. Returns the
    /// metadata path and each shard's `(path, size)` — the caller can hand
    /// them to `QuarryTable::store_row_files` and drop the in-memory copy.
    pub async fn write_join_hash(
        &self,
        table: &TableId,
        at: SnapshotId,
        policy: PolicyFingerprint,
        hash: &JoinHash,
        rows: &[datafusion::arrow::record_batch::RecordBatch],
        field_ids: &BTreeMap<String, FieldId>,
    ) -> iceberg::Result<(String, Vec<(String, u64)>)> {
        write_join_hash(
            &self.file_io,
            &self.layout,
            table,
            at,
            policy,
            hash,
            rows,
            field_ids,
            self.join_shard_bytes,
        )
        .await
    }

    /// Rebuild a table's registry from storage.
    pub async fn recover(
        &self,
        table: &TableId,
        known_snapshots: &[SnapshotId],
    ) -> iceberg::Result<Recovered> {
        recover(
            &self.file_io,
            &self.store,
            &self.layout,
            table,
            known_snapshots,
        )
        .await
    }

    /// Delete unusable objects.
    pub async fn discard(&self, paths: &[String]) -> iceberg::Result<usize> {
        discard(&self.file_io, paths).await
    }

    /// Write a manifest for `entries`, carrying forward whatever `prior`
    /// already published for the same snapshot.
    pub async fn manifest(
        &self,
        table: &TableId,
        at: SnapshotId,
        entries: &[Advertised],
        prior: Option<&StatisticsFile>,
    ) -> iceberg::Result<StatisticsFile> {
        write_manifest(&self.file_io, &self.layout, table, at, entries, prior).await
    }

    /// Advertise derived state through the table's own metadata.
    ///
    /// The merge runs against the metadata `table` carries, so publish the
    /// freshest load available.
    pub async fn publish(
        &self,
        table: &Table,
        catalog: &dyn Catalog,
        at: SnapshotId,
        entries: &[Advertised],
    ) -> iceberg::Result<StatisticsFile> {
        let metadata = table.metadata();
        let stats = self
            .manifest(
                &crate::from_iceberg::table_id(metadata),
                at,
                entries,
                metadata.statistics_for_snapshot(at.0),
            )
            .await?;
        Transaction::new(table)
            .update_statistics()
            .set_statistics(stats.clone())
            .apply(Transaction::new(table))?
            .commit(catalog)
            .await?;
        Ok(stats)
    }

    /// Register what a published statistics file advertises.
    ///
    /// The published discovery path, where [`Store::recover`'s](Self::recover)
    /// is listing: the catalog's `blob_metadata` names every entry, so this
    /// costs no listing and no read of the manifest itself.
    pub async fn recover_published(
        &self,
        table: &TableId,
        stats: &StatisticsFile,
        known_snapshots: &[SnapshotId],
    ) -> iceberg::Result<Recovered> {
        let mut recovered = Recovered::default();
        let at = SnapshotId(stats.snapshot_id);
        if !known_snapshots.contains(&at) {
            recovered.discarded.push(stats.statistics_path.clone());
            return Ok(recovered);
        }
        for entry in advertised(stats) {
            // A filter set names no field it must parse; its blob carries
            // the filter itself.
            if entry.kind == QUARRY_FILTER_SET_V1 {
                match read_filter_set(&self.file_io, &entry.path).await? {
                    Some((set, policy)) => {
                        let bytes = set.bytes_estimate();
                        let id = super::filter_set_id(table, set.filter());
                        recovered.registry.register(Derived::new(
                            id,
                            Source {
                                table: table.clone(),
                                snapshot: at,
                            },
                            policy,
                            bytes,
                            Box::new(set),
                        ));
                    }
                    None => recovered.discarded.push(entry.path),
                }
                continue;
            }
            if entry.kind == QUARRY_JOIN_HASH_V1 {
                match read_join_hash(&self.file_io, &entry.path).await? {
                    Some((keys, columns, policy, rows_paths)) => {
                        let id = super::join_hash_id_from(table, &keys, &columns);
                        match parquet_row_files(&self.store, &self.layout, &rows_paths).await {
                            Some((files, bytes)) => {
                                recovered.row_files.insert(id.clone(), files);
                                recovered.registry.register(Derived::new(
                                    id,
                                    Source {
                                        table: table.clone(),
                                        snapshot: at,
                                    },
                                    policy,
                                    bytes,
                                    Box::new(JoinHash::covering(keys, columns, bytes)),
                                ));
                            }
                            // A shard missing or unreadable is a hard
                            // failure: partial rows would answer wrongly.
                            None => {
                                recovered.discarded.push(entry.path);
                                recovered.discarded.extend(rows_paths);
                            }
                        }
                    }
                    None => recovered.discarded.push(entry.path),
                }
                continue;
            }
            if entry.kind != QUARRY_EQ_INDEX_V2
                && entry.kind != QUARRY_BITMAP_V1
                && entry.kind != QUARRY_TEXT_INDEX_V1
                && entry.kind != QUARRY_VECTOR_INDEX_V1
            {
                continue;
            }
            let Some(&field) = entry.fields.first() else {
                continue;
            };
            let Ok(field) = u32::try_from(field) else {
                continue;
            };
            if entry.kind == QUARRY_VECTOR_INDEX_V1 {
                match read_vector_index(&self.file_io, &entry.path).await? {
                    Some((field, metric, dimension, policy, rows_path)) => {
                        let Some((files, bytes)) = parquet_row_files(
                            &self.store,
                            &self.layout,
                            std::slice::from_ref(&rows_path),
                        )
                        .await
                        else {
                            recovered.discarded.push(entry.path);
                            recovered.discarded.push(rows_path);
                            continue;
                        };
                        let id = super::index_id(table, field);
                        recovered.fields.insert(id.clone(), field);
                        recovered.row_files.insert(id.clone(), files);
                        recovered.registry.register(Derived::new(
                            id,
                            Source {
                                table: table.clone(),
                                snapshot: at,
                            },
                            policy,
                            bytes,
                            Box::new(VectorIndex::new(field, metric, dimension, bytes)),
                        ));
                    }
                    None => recovered.discarded.push(entry.path),
                }
                continue;
            }
            let read = match read_index(&self.file_io, &entry.path, field).await? {
                Some((index, policy)) => {
                    let bytes = index.bytes_estimate();
                    Some((
                        Box::new(index) as Box<dyn crate::derived::Kind>,
                        bytes,
                        policy,
                    ))
                }
                None => match read_bitmap(&self.file_io, &entry.path, field).await? {
                    Some((bitmap, policy)) => {
                        let bytes = bitmap.bytes_estimate();
                        Some((
                            Box::new(bitmap) as Box<dyn crate::derived::Kind>,
                            bytes,
                            policy,
                        ))
                    }
                    None => match read_text_index(&self.file_io, &entry.path, field).await? {
                        Some((text, policy)) => {
                            let bytes = text.bytes_estimate();
                            Some((
                                Box::new(text) as Box<dyn crate::derived::Kind>,
                                bytes,
                                policy,
                            ))
                        }
                        None => None,
                    },
                },
            };
            match read {
                Some((kind, bytes, policy)) => {
                    let id = super::index_id(table, field);
                    recovered.fields.insert(id.clone(), field);
                    recovered.registry.register(Derived::new(
                        id,
                        Source {
                            table: table.clone(),
                            snapshot: at,
                        },
                        policy,
                        bytes,
                        kind,
                    ));
                }
                None => recovered.discarded.push(entry.path),
            }
        }
        Ok(recovered)
    }

    /// Reclaim the bytes behind what a round retired.
    ///
    /// Retirement removes an index from the registry but not from storage —
    /// the optimizer holds none — so without this every retired index is an
    /// orphan that only a forgotten snapshot would ever clean up.
    ///
    /// Pieces whose field the optimizer never knew are skipped: their blob
    /// name cannot be reconstructed, and deleting every index of the same
    /// snapshot to be sure would throw away work the caller did not retire.
    /// Such orphans are still collected by [`Store::recover`], which discards
    /// what it cannot use.
    pub async fn reclaim(
        &self,
        table: &TableId,
        retired: &[super::optimizer::Retired],
    ) -> iceberg::Result<usize> {
        let mut paths = Vec::new();
        for r in retired {
            if let Some(field) = r.field {
                paths.push(self.layout.index_path(table, field, r.at));
            } else if let Some(id_hash) = parse_join_hash_id(&r.id) {
                let meta = self.layout.join_hash_path(table, id_hash, r.at);
                paths.push(meta.clone());
                // The shard count lives in the blob. An unreadable one still
                // has the pre-sharding rows path to try; numbered shards it
                // cannot name become orphans the next `recover` sweeps.
                match read_join_hash(&self.file_io, &meta).await? {
                    Some((_, _, _, rows_paths)) => paths.extend(rows_paths),
                    None => paths.push(self.layout.join_hash_rows_path(table, id_hash, r.at)),
                }
            }
        }
        discard(&self.file_io, &paths).await
    }

    /// Write the workload down.
    pub async fn save_workload(
        &self,
        table: &TableId,
        workload: &Workload,
    ) -> iceberg::Result<String> {
        write_workload(&self.file_io, &self.layout, table, workload).await
    }

    /// Read a saved workload, if there is one.
    pub async fn load_workload(&self, table: &TableId) -> iceberg::Result<Option<Workload>> {
        read_workload(&self.file_io, &self.layout, table).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> TableId {
        TableId("9c12d441".into())
    }

    fn layout() -> Layout {
        Layout::new("_quarry/idx")
    }

    fn index_of(pairs: &[(u64, &str)]) -> Index {
        let mut index = Index::new(4);
        for (value, file) in pairs {
            index.insert(*value, FileId((*file).to_owned()));
        }
        index
    }

    #[test]
    fn a_path_carries_the_identity() {
        let path = layout().index_path(&table(), 4, SnapshotId(810));
        assert_eq!(path, "_quarry/idx/9c12d441/4/810.puffin");
        assert_eq!(
            layout().parse(&path),
            Some((table(), 4, SnapshotId(810))),
            "listing alone must be enough to rebuild the registry"
        );
    }

    #[test]
    fn a_trailing_slash_in_the_prefix_does_not_double_up() {
        assert_eq!(
            Layout::new("_quarry/idx/").index_path(&table(), 4, SnapshotId(1)),
            "_quarry/idx/9c12d441/4/1.puffin"
        );
    }

    #[test]
    fn paths_that_are_not_ours_are_ignored() {
        let layout = layout();
        for path in [
            "somewhere/else/4/1.puffin",
            "_quarry/idx/9c12d441/4/1.parquet",
            "_quarry/idx/9c12d441/notafield/1.puffin",
            "_quarry/idx/9c12d441/4/notasnapshot.puffin",
            "_quarry/idx/9c12d441/4/1/extra.puffin",
            "_quarry/idx/9c12d441",
        ] {
            assert_eq!(layout.parse(path), None, "should not parse: {path}");
        }
    }

    #[test]
    fn the_encoder_produces_exactly_the_size_the_index_reports() {
        // `Index::encoded_len` is what the storage budget is enforced against,
        // and what a freshly built index reports. A recovered one reports the
        // length of the blob it was read from. If these two ever disagree, the
        // same index is sized differently either side of a restart.
        for pairs in [
            &[][..],
            &[(1, "a.parquet")][..],
            &[(1, "a.parquet"), (1, "c.parquet"), (2, "b.parquet")][..],
            &[(7, "a/very/long/path/to/an/object/somewhere.parquet")][..],
        ] {
            let index = index_of(pairs);
            assert_eq!(
                encode(&index).len() as u64,
                index.encoded_len(),
                "disagreement for {pairs:?}"
            );
        }
    }

    #[test]
    fn an_index_survives_a_round_trip() {
        let index = index_of(&[(1, "a.parquet"), (1, "c.parquet"), (2, "b.parquet")]);
        let decoded = decode(4, &encode(&index)).expect("decodes");

        assert_eq!(decoded.field(), 4);
        assert_eq!(decoded.values(), 2);
        assert_eq!(decoded.files_for(1), index.files_for(1));
        assert_eq!(decoded.files_for(2), index.files_for(2));
    }

    #[test]
    fn an_empty_index_survives_a_round_trip() {
        let decoded = decode(4, &encode(&Index::new(4))).expect("decodes");
        assert_eq!(decoded.values(), 0);
    }

    #[test]
    fn truncated_bytes_decode_to_nothing_rather_than_less() {
        // A short index is not a smaller index: it would prune away files
        // holding matching rows, which is the direction that returns wrong
        // answers. Refusing is the only safe response.
        let encoded = encode(&index_of(&[(1, "a.parquet"), (2, "b.parquet")]));
        for cut in 1..encoded.len() {
            assert!(
                decode(4, &encoded[..cut]).is_none(),
                "truncating to {cut} bytes must be refused"
            );
        }
    }

    #[test]
    fn trailing_bytes_are_refused() {
        let mut encoded = encode(&index_of(&[(1, "a.parquet")]));
        encoded.push(0);
        assert!(decode(4, &encoded).is_none());
    }

    #[test]
    fn an_absurd_length_does_not_panic() {
        // A corrupt length prefix must not be trusted into an allocation or
        // an out-of-bounds read.
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&1u64.to_le_bytes()); // one value
        encoded.extend_from_slice(&7u64.to_le_bytes()); // the value
        encoded.extend_from_slice(&1u64.to_le_bytes()); // one file
        encoded.extend_from_slice(&u64::MAX.to_le_bytes()); // impossible length
        assert!(decode(4, &encoded).is_none());
    }

    #[test]
    fn a_file_number_outside_the_table_is_refused() {
        // Introduced with interning: postings name files by number now, so a
        // number past the end of the table is a new way for bytes to be wrong.
        // Skipping it would drop a file the index should have named, which
        // under-selects — the one direction that returns wrong answers.
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&1u32.to_le_bytes()); // one path
        encoded.extend_from_slice(&9u32.to_le_bytes()); // of length nine
        encoded.extend_from_slice(b"a.parquet");
        encoded.extend_from_slice(&1u64.to_le_bytes()); // one value
        encoded.extend_from_slice(&7u64.to_le_bytes()); // the value
        encoded.extend_from_slice(&1u32.to_le_bytes()); // in one file
        encoded.extend_from_slice(&5u32.to_le_bytes()); // file 5, of one
        assert!(decode(4, &encoded).is_none());
    }

    #[test]
    fn interning_makes_repeated_paths_nearly_free() {
        // The measured problem: a path is 60-100 bytes and was written once
        // per posting. One long path across many values should now cost about
        // four bytes per posting rather than ninety.
        let long = "warehouse/db/events/data/00000-0-a1b2c3d4-e5f6.parquet";
        let mut index = Index::new(4);
        for value in 0..1_000u64 {
            index.insert(value, FileId(long.to_owned()));
        }

        let per_posting = index.encoded_len() as f64 / 1_000.0;
        assert!(
            per_posting < 20.0,
            "{per_posting} bytes per posting; the path should be stored once"
        );
        assert_eq!(encode(&index).len() as u64, index.encoded_len());
    }

    #[test]
    fn invalid_utf8_in_a_path_is_refused() {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&1u64.to_le_bytes());
        encoded.extend_from_slice(&7u64.to_le_bytes());
        encoded.extend_from_slice(&1u64.to_le_bytes());
        encoded.extend_from_slice(&2u64.to_le_bytes());
        encoded.extend_from_slice(&[0xff, 0xfe]);
        assert!(decode(4, &encoded).is_none());
    }
}
