//! Building the derived state the loop proposed.
//!
//! This is where the system stops advising and starts optimizing. Everything
//! else either uses derived state or decides that some would help; this reads
//! the data and produces it.
//!
//! # Why it can read so little
//!
//! An index over one field needs only that field. Projecting to a single
//! column means Parquet reads one column chunk per row group and skips the
//! rest, so building an index over a wide table costs a small fraction of a
//! full scan. That is the same projection pushdown a query gets, reached
//! through the same DataFusion file source rather than a separate reader.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use datafusion::common::Result as DfResult;
use datafusion::common::exec_err;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::datasource::physical_plan::{FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::collect;
use datafusion::scalar::ScalarValue;

use crate::derived::{Derived, DerivedId, FieldId, PolicyFingerprint, Source};
use crate::kinds::Index;
use crate::snapshot::FileId;

use super::{QuarryTable, Session, hash_scalar};

/// Build an index over `field` by reading that column from `table`'s objects.
///
/// Reads only the indexed column. Null values are skipped: `x = NULL` matches
/// nothing in SQL, and `IS NULL` is an opaque predicate the index would not be
/// probed for anyway.
///
/// Values are hashed with [`hash_scalar`], the same function the scan uses to
/// probe, so build and probe cannot disagree about what a value hashes to.
pub async fn build_index(
    session: &Session,
    table: &QuarryTable,
    field: FieldId,
) -> DfResult<Index> {
    let Some(column) = table.column_of(field) else {
        return exec_err!("field {field} is not a column of this table");
    };
    let Some((url, files)) = table.parquet_files() else {
        return exec_err!("an in-memory table has no objects to index");
    };

    let schema = datafusion::catalog::TableProvider::schema(table);
    let Some((position, _)) = schema.column_with_name(column) else {
        return exec_err!("column {column} is not in the table's schema");
    };

    let mut index = Index::new(field);

    for (file, size) in files {
        // One value may appear in many row groups; recording it once per file
        // is all the index holds.
        let mut seen = HashSet::new();
        for value in read_column(session, url, &schema, position, file, *size).await? {
            if seen.insert(value) {
                index.insert(value, file.clone());
            }
        }
    }

    // The exact encoded size, not an estimate: this is what the storage
    // budget is enforced against, and it must match what a recovered index
    // reports or the two disagree across a restart.
    let bytes = index.encoded_len();
    Ok(index.with_bytes(bytes))
}

/// Build the index a proposal asked for, ready to register.
///
/// The last step of the loop: a [`Proposal`](crate::workload::Proposal) names a
/// table and a field, and this returns the [`Derived`] to hand to the registry.
pub async fn build_proposed_index(
    session: &Session,
    table: &QuarryTable,
    field: FieldId,
    id: DerivedId,
    policy: PolicyFingerprint,
) -> DfResult<Derived> {
    let index = build_index(session, table, field).await?;
    let bytes = index.bytes_estimate();
    Ok(Derived::new(
        id,
        Source {
            table: table.table_id().clone(),
            snapshot: table.snapshot(),
        },
        policy,
        bytes,
        Box::new(index),
    ))
}

/// Every non-null value of one column of one object, hashed.
async fn read_column(
    session: &Session,
    url: &ObjectStoreUrl,
    schema: &datafusion::arrow::datatypes::SchemaRef,
    position: usize,
    file: &FileId,
    size: u64,
) -> DfResult<Vec<u64>> {
    let config = FileScanConfigBuilder::new(
        url.clone(),
        Arc::clone(schema),
        Arc::new(ParquetSource::default()),
    )
    .with_projection(Some(vec![position]))
    .with_file(PartitionedFile::new(file.0.clone(), size))
    .build();

    let plan = DataSourceExec::from_data_source(config);
    let batches = collect(plan, session.context().task_ctx()).await?;

    let mut values = Vec::new();
    for batch in batches {
        let column = batch.column(0);
        for row in 0..batch.num_rows() {
            if column.is_null(row) {
                continue;
            }
            // Via ScalarValue rather than reading the array's native type, so
            // that a value hashes identically whether it came from data here or
            // from a literal in a query. Slower than a typed path, and the
            // obvious thing to optimise once correctness is pinned.
            values.push(hash_scalar(&ScalarValue::try_from_array(column, row)?));
        }
    }
    Ok(values)
}

/// Names for derived state the loop builds.
///
/// Stable and derived from what the index is *on*, so rebuilding after a
/// snapshot moves replaces the old entry rather than accumulating beside it.
pub fn index_id(table: &crate::snapshot::TableId, field: FieldId) -> DerivedId {
    DerivedId(format!("idx:{}:{field}", table.0))
}

/// Column name to field id for every column of a table, for convenience.
pub fn columns(table: &QuarryTable) -> BTreeMap<String, FieldId> {
    let schema = datafusion::catalog::TableProvider::schema(table);
    schema
        .fields()
        .iter()
        .filter_map(|f| table.field_id_of(f.name()).map(|id| (f.name().clone(), id)))
        .collect()
}
