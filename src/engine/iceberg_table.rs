//! Turning a real Iceberg table into one the rule can plan.
//!
//! The two halves were built to meet here: [`crate::from_iceberg`] extracts
//! everything the rule needs from Iceberg metadata, and [`QuarryTable`] reads
//! Parquet objects the rule selects. This joins them, and needs both features.
//!
//! It is a small function because the pieces were designed to fit. That is the
//! test: if joining them had needed either side to change, the boundary would
//! have been in the wrong place.

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::datasource::object_store::ObjectStoreUrl;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::FileIO;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;

use crate::derived::FieldId;
use crate::from_iceberg::{
    bounds_by_field, current_snapshot, live_data_files, snapshot_graph, table_id,
};
use crate::snapshot::SnapshotId;

use super::QuarryTable;

/// Build a [`QuarryTable`] over a real Iceberg table.
///
/// Reads the table's manifests, so it costs one round trip per manifest. Pass
/// `at` to read a snapshot other than the current one; a table with no
/// snapshots at all yields a table with no files, which queries as empty.
///
/// `url` identifies the object store the data files live in. Iceberg records
/// file locations as URIs and this addresses them as paths within that store,
/// which is what [`object_path`](crate::from_iceberg::object_path) reconciles.
pub async fn table_from_iceberg(
    metadata: &TableMetadata,
    file_io: &FileIO,
    url: ObjectStoreUrl,
    at: Option<SnapshotId>,
) -> iceberg::Result<QuarryTable> {
    let schema = Arc::new(arrow_schema(metadata)?);
    let field_ids = field_ids(metadata);
    let graph = snapshot_graph(metadata, file_io).await?;

    // A table with no snapshots has nothing to read. Snapshot 0 is not a valid
    // Iceberg id, so it cannot collide with a real one, and the rule will
    // refuse any derived state claiming to come from it.
    let snapshot = at
        .or_else(|| current_snapshot(metadata))
        .unwrap_or(SnapshotId(0));

    // Per-file bounds come along, because they are what decides whether an
    // index would beat the format's own pruning and they cost one metadata
    // pass. Without them the optimizer declines to judge.
    let mut table = QuarryTable::new(schema, table_id(metadata), snapshot, graph, field_ids)
        .on_object_store(url)
        .with_field_bounds(bounds_by_field(metadata, file_io, snapshot).await?);
    for (file, size) in live_data_files(metadata, file_io, snapshot).await? {
        table = table.with_parquet_file(file, size);
    }
    Ok(table)
}

/// Build a [`QuarryTable`] over a table loaded from any Iceberg catalog.
///
/// Takes [`iceberg::table::Table`] rather than a particular catalog type,
/// which is why this crate depends on no catalog implementation at all: REST,
/// in-memory, Glue and anything else all hand back the same `Table`, carrying
/// the metadata and the `FileIO` configured for wherever it lives.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use datafusion::datasource::object_store::ObjectStoreUrl;
/// # use iceberg::{Catalog, TableIdent};
/// # async fn example(catalog: &dyn Catalog) -> iceberg::Result<()> {
/// let table = catalog.load_table(&TableIdent::from_strs(["db", "events"])?).await?;
/// let queryable = quarry::engine::table_from_catalog(
///     &table,
///     ObjectStoreUrl::local_filesystem(),
///     None,
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn table_from_catalog(
    table: &Table,
    url: ObjectStoreUrl,
    at: Option<SnapshotId>,
) -> iceberg::Result<QuarryTable> {
    table_from_iceberg(table.metadata(), table.file_io(), url, at).await
}

/// The table's Arrow schema.
pub fn arrow_schema(metadata: &TableMetadata) -> iceberg::Result<ArrowSchema> {
    schema_to_arrow_schema(metadata.current_schema())
}

/// Column name to Iceberg field id, for the current schema.
///
/// Derived state is keyed on field ids, never names, so this mapping is what
/// lets a renamed column keep matching the index built on it — and what stops a
/// *different* column that reuses an old name from matching by accident.
pub fn field_ids(metadata: &TableMetadata) -> BTreeMap<String, FieldId> {
    metadata
        .current_schema()
        .as_struct()
        .fields()
        .iter()
        .map(|field| (field.name.clone(), field.id as FieldId))
        .collect()
}
