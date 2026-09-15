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
use datafusion::parquet::arrow::arrow_reader::ArrowReaderMetadata;
use datafusion::parquet::arrow::async_reader::ParquetObjectReader;
use datafusion::parquet::file::statistics::Statistics;
use datafusion::physical_plan::collect;
use datafusion::scalar::ScalarValue;
use object_store::path::Path as ObjectPath;

use crate::derived::{Derived, DerivedId, FieldId, PolicyFingerprint, Source};
use crate::kinds::{Bitmap, Index};
use crate::snapshot::FileId;
use crate::workload::AggregateAsk;

use super::{MaterializedResult, QuarryTable, Rollup, Session, hash_scalar};

/// The most distinct values a field may hold for a bitmap to be the right
/// instrument — past it, per-row postings are just a dearer file index.
const MAX_BITMAP_CARDINALITY: usize = 64;

/// How many *positions* in the file list to sample around.
///
/// Each position contributes a file and its neighbour, so the sample contains
/// both adjacent and widely separated pairs. Three positions is six files and
/// fifteen pairs.
const SAMPLE_POSITIONS: usize = 3;

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
    let reads = column_reads(session, table, field).await?;
    let index = populate_index(field, &reads);
    // The exact encoded size, not an estimate: this is what the storage
    // budget is enforced against, and it must match what a recovered index
    // reports or the two disagree across a restart.
    let bytes = index.encoded_len();
    Ok(index.with_bytes(bytes))
}

/// Every file's column read, positions preserved and footer facts attached.
async fn column_reads(
    session: &Session,
    table: &QuarryTable,
    field: FieldId,
) -> DfResult<Vec<(FileId, ColumnRead)>> {
    let Some(column) = table.column_of(field) else {
        return exec_err!("field {field} is not a column of this table");
    };
    let Some((url, files)) = table.parquet_files() else {
        return exec_err!("an in-memory table has no objects to index");
    };
    let store = session.context().runtime_env().object_store(url)?;

    let schema = datafusion::catalog::TableProvider::schema(table);
    let Some((position, _)) = schema.column_with_name(column) else {
        return exec_err!("column {column} is not in the table's schema");
    };

    let mut reads = Vec::with_capacity(files.len());
    for (file, size) in files {
        let read =
            read_column_grouped(session, url, &store, &schema, position, file, *size).await?;
        reads.push((file.clone(), read));
    }
    Ok(reads)
}

/// The file-level index the reads support.
fn populate_index(field: FieldId, reads: &[(FileId, ColumnRead)]) -> Index {
    let mut index = Index::new(field);
    for (file, read) in reads {
        // One value may appear in many row groups; recording it once per file
        // is all the index holds.
        let mut seen = HashSet::new();
        for value in read.values.iter().flatten() {
            if seen.insert(*value) {
                index.insert(*value, file.clone());
            }
        }
    }
    index
}

/// The row-level bitmap the reads support.
fn populate_bitmap(field: FieldId, reads: &[(FileId, ColumnRead)]) -> Bitmap {
    let mut bitmap = Bitmap::new(field);
    for (file, read) in reads {
        for (row, value) in read.values.iter().enumerate() {
            let row = row as u64;
            let group = read.group_ends.partition_point(|end| row >= *end) as u32;
            let start = if group == 0 {
                0
            } else {
                read.group_ends[group as usize - 1]
            };
            if let Some(value) = value {
                bitmap.insert(*value, file.clone(), group, row - start);
            }
        }
    }
    bitmap
}

/// Build a bitmap over `field`: postings down to the row inside its group.
///
/// The same read [`build_index`] makes, plus the footer's per-group row
/// counts — they turn each value's ordinal in the file into the
/// (group, row-in-group) pair the postings address by.
pub async fn build_bitmap(
    session: &Session,
    table: &QuarryTable,
    field: FieldId,
) -> DfResult<Bitmap> {
    let reads = column_reads(session, table, field).await?;
    let bitmap = populate_bitmap(field, &reads);
    let bytes = bitmap.encoded_len();
    Ok(bitmap.with_bytes(bytes))
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
    let reads = column_reads(session, table, field).await?;

    // The proposal names a field, not a shape — the build picks the
    // granularity from what the read showed. A field of few distinct
    // values earns the row-level bitmap, whose postings stay small and
    // which prunes inside files; but only where files actually have
    // interior structure — against single-group files the finer postings
    // cost more to store and skip nothing the coarser ones would not.
    let distinct: HashSet<u64> = reads
        .iter()
        .flat_map(|(_, read)| read.values.iter().flatten().copied())
        .collect();
    let multi_group = reads.iter().any(|(_, read)| read.group_ends.len() > 1);

    let (kind, bytes): (Box<dyn crate::derived::Kind>, u64) =
        if distinct.len() <= MAX_BITMAP_CARDINALITY && multi_group {
            let bitmap = populate_bitmap(field, &reads);
            let bytes = bitmap.encoded_len();
            (Box::new(bitmap.with_bytes(bytes)), bytes)
        } else {
            let index = populate_index(field, &reads);
            let bytes = index.encoded_len();
            (Box::new(index.with_bytes(bytes)), bytes)
        };

    Ok(Derived::new(
        id,
        Source {
            table: table.table_id().clone(),
            snapshot: table.snapshot(),
        },
        policy,
        bytes,
        kind,
    ))
}

/// How much of one file's values another file also holds, for `field`.
///
/// The input [`Spread::from_overlap`](crate::layout::Spread::from_overlap)
/// needs, and the reason the gate is affordable: two column reads rather than
/// the whole table. Building the index to find out whether the index is worth
/// building would defeat the point.
///
/// Returns `None` when there are fewer than two files, or when no sampled file
/// holds any value of the field — in neither case is there an overlap to
/// measure, and inventing one would be worse than admitting ignorance.
///
/// # Which files to sample, arrived at by being wrong twice
///
/// Against a table where each value occupied three files of twenty:
///
/// ```text
/// sample                    estimate   truth
/// first and last only         13.7      3.0
/// four files, evenly spread    1.0      3.0
/// three adjacent pairs         ~3.5      3.0
/// ```
///
/// The first was chosen on the reasoning that neighbouring files in an
/// ingestion-ordered table resemble each other and would flatter any column.
/// That was wrong twice over: "first and last" is only distant in *path
/// order*, which need not relate to content, and in that table they were
/// neighbours.
///
/// Spreading the sample out then failed the opposite way. A value spanning
/// three consecutive files shows **zero** overlap between files five apart, so
/// widely separated samples cannot see locality among nearby ones and report
/// every value as living in one file.
///
/// So the sample contains both: a few positions across the list, each
/// contributing a file *and its neighbour*. Adjacent pairs reveal local
/// clustering, distant pairs reveal global spread, and averaging over all of
/// them lands close. Adjacent pairs are slightly over-represented relative to
/// a uniform sample of pairs, which biases the estimate toward *more* files
/// per value — less advantage, so fewer indexes built. The conservative
/// direction.
pub async fn estimate_overlap(
    session: &Session,
    table: &QuarryTable,
    field: FieldId,
) -> DfResult<Option<f64>> {
    let Some(column) = table.column_of(field) else {
        return exec_err!("field {field} is not a column of this table");
    };
    let Some((url, files)) = table.parquet_files() else {
        return Ok(None);
    };
    if files.len() < 2 {
        return Ok(None);
    }

    let schema = datafusion::catalog::TableProvider::schema(table);
    let Some((position, _)) = schema.column_with_name(column) else {
        return exec_err!("column {column} is not in the table's schema");
    };

    let all: Vec<(&FileId, &u64)> = files.iter().collect();

    // A file and its neighbour at each of a few positions, so the pairs span
    // both spacings. Deduplicated, since the positions coincide on a short
    // list.
    let positions = SAMPLE_POSITIONS.min(all.len());
    let stride = (all.len() / positions).max(1);
    let mut chosen: Vec<usize> = Vec::new();
    for step in 0..positions {
        let at = step * stride;
        for index in [at, at + 1] {
            if index < all.len() && !chosen.contains(&index) {
                chosen.push(index);
            }
        }
    }

    let mut samples: Vec<HashSet<u64>> = Vec::with_capacity(chosen.len());
    for index in chosen {
        let (file, size) = all[index];
        let values: HashSet<u64> = read_column(session, url, &schema, position, file, *size)
            .await?
            .into_iter()
            .flatten()
            .collect();
        if !values.is_empty() {
            samples.push(values);
        }
    }
    if samples.len() < 2 {
        return Ok(None);
    }

    // Of one file's values, what fraction does another hold? Averaged over
    // every pair, in both directions, since the two are not symmetric when
    // the files hold different numbers of distinct values.
    let mut total = 0.0;
    let mut pairs = 0u32;
    for (position, one) in samples.iter().enumerate() {
        for other in samples.iter().skip(position + 1) {
            let shared = one.iter().filter(|value| other.contains(value)).count();
            total += shared as f64 / one.len() as f64;
            let shared = other.iter().filter(|value| one.contains(value)).count();
            total += shared as f64 / other.len() as f64;
            pairs += 2;
        }
    }
    Ok(Some(total / pairs as f64))
}

/// Per-file value ranges for `field`, read from the Parquet footers.
///
/// The other half of what the index gate needs, for tables that do not come
/// from a catalog. Iceberg records these bounds in its manifests and they cost
/// nothing to read; a plain Parquet table has them too, in each file's footer,
/// and without them the gate has to decline every proposal.
///
/// Only footers are read — a few kilobytes per file, no column data — so this
/// stays far cheaper than the build it informs.
///
/// A file whose footer records no statistics for the column is **omitted**
/// rather than guessed at, and [`Spread`](crate::layout::Spread) treats a
/// missing range as a file that must always be read. Assuming a range would
/// claim the format prunes something it does not.
pub async fn parquet_bounds(
    session: &Session,
    table: &QuarryTable,
    field: FieldId,
) -> DfResult<Vec<(f64, f64)>> {
    let Some(column) = table.column_of(field) else {
        return exec_err!("field {field} is not a column of this table");
    };
    let Some((url, files)) = table.parquet_files() else {
        return Ok(Vec::new());
    };
    let store = session.context().runtime_env().object_store(url)?;

    let schema = datafusion::catalog::TableProvider::schema(table);
    let Some((position, _)) = schema.column_with_name(column) else {
        return exec_err!("column {column} is not in the table's schema");
    };

    let mut ranges = Vec::new();
    for (file, size) in files {
        let mut reader =
            ParquetObjectReader::new(Arc::clone(&store), ObjectPath::from(file.0.as_str()))
                .with_file_size(*size);
        let Ok(metadata) = ArrowReaderMetadata::load_async(&mut reader, Default::default()).await
        else {
            continue; // Not readable as Parquet; the caller will treat it as unknown.
        };

        // A file's range is the union of its row groups' ranges.
        let mut low = f64::MAX;
        let mut high = f64::MIN;
        for group in metadata.metadata().row_groups() {
            let Some(statistics) = group.columns().get(position).and_then(|c| c.statistics())
            else {
                continue;
            };
            if let Some((group_low, group_high)) = numeric_range(statistics) {
                low = low.min(group_low);
                high = high.max(group_high);
            }
        }
        if low <= high {
            ranges.push((low, high));
        }
    }
    Ok(ranges)
}

/// One column chunk's range, if the type has a meaningful distance.
///
/// Booleans, strings and byte arrays are ordered but have no distance that
/// range *widths* could be compared across, which is what the estimate needs.
fn numeric_range(statistics: &Statistics) -> Option<(f64, f64)> {
    match statistics {
        Statistics::Int32(values) => Some((*values.min_opt()? as f64, *values.max_opt()? as f64)),
        Statistics::Int64(values) => Some((*values.min_opt()? as f64, *values.max_opt()? as f64)),
        Statistics::Float(values) => Some((*values.min_opt()? as f64, *values.max_opt()? as f64)),
        Statistics::Double(values) => Some((*values.min_opt()?, *values.max_opt()?)),
        _ => None,
    }
}

/// One file's column read and the footer facts a row-level build needs.
struct ColumnRead {
    /// Every row's value, hashed — `None` where null, so the row's ordinal
    /// in the file survives in the vector's.
    values: Vec<Option<u64>>,
    /// Cumulative row counts: group `g` holds ordinals `ends[g-1]..ends[g]`.
    group_ends: Vec<u64>,
}

async fn read_column_grouped(
    session: &Session,
    url: &ObjectStoreUrl,
    store: &Arc<dyn object_store::ObjectStore>,
    schema: &datafusion::arrow::datatypes::SchemaRef,
    position: usize,
    file: &FileId,
    size: u64,
) -> DfResult<ColumnRead> {
    let mut reader = ParquetObjectReader::new(Arc::clone(store), ObjectPath::from(file.0.as_str()))
        .with_file_size(size);
    let metadata = ArrowReaderMetadata::load_async(&mut reader, Default::default()).await?;

    let mut group_ends = Vec::new();
    let mut total = 0u64;
    for group in metadata.metadata().row_groups() {
        total += group.num_rows() as u64;
        group_ends.push(total);
    }

    Ok(ColumnRead {
        values: read_column(session, url, schema, position, file, size).await?,
        group_ends,
    })
}

/// Every value of one column of one object, hashed — `None` where null, so
/// the row's position in the file survives in the vector's.
async fn read_column(
    session: &Session,
    url: &ObjectStoreUrl,
    schema: &datafusion::arrow::datatypes::SchemaRef,
    position: usize,
    file: &FileId,
    size: u64,
) -> DfResult<Vec<Option<u64>>> {
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
                values.push(None);
                continue;
            }
            // Via ScalarValue rather than reading the array's native type, so
            // that a value hashes identically whether it came from data here or
            // from a literal in a query. Slower than a typed path, and the
            // obvious thing to optimise once correctness is pinned.
            values.push(Some(hash_scalar(&ScalarValue::try_from_array(
                column, row,
            )?)));
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

/// A cube's id names what it answers, so the same ask rebuilds onto the same
/// id rather than accumulating beside it.
pub fn cube_id(table: &crate::snapshot::TableId, ask: &AggregateAsk) -> DerivedId {
    use std::hash::{Hash, Hasher};
    let mut hasher = crate::stable_hash::StableHasher::new();
    (table.clone(), ask.clone()).hash(&mut hasher);
    DerivedId(format!("cube:{}:{:016x}", table.0, hasher.finish()))
}

/// Build the cube `ask` proposes: group keys plus measure partials, computed
/// by running the aggregation against the table itself.
///
/// The build goes through the session's context rather than `Session::sql`
/// deliberately — the point is to compute from the table, not to be answered
/// by a cube that already exists.
pub async fn build_cube(
    session: &Session,
    table: &Arc<QuarryTable>,
    ask: &AggregateAsk,
    id: DerivedId,
) -> DfResult<Derived> {
    let names: BTreeMap<FieldId, String> = ask
        .spec
        .group_by
        .iter()
        .copied()
        .chain(ask.spec.measures.iter().filter_map(|m| m.field))
        .filter_map(|field| Some((field, table.column_of(field)?.to_owned())))
        .collect();
    let rollup = Rollup::new(ask.spec.clone(), names);

    let source = format!("__quarry_cube_{}", id.0.replace(':', "_"));
    session
        .context()
        .register_table(
            source.as_str(),
            Arc::clone(table) as Arc<dyn datafusion::catalog::TableProvider>,
        )
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

    let keys: Vec<&str> = ask
        .spec
        .group_by
        .iter()
        .filter_map(|field| rollup.key_name(*field))
        .collect();
    let measures: Vec<String> = ask
        .spec
        .measures
        .iter()
        .filter_map(|m| rollup.measure_name(m))
        .map(|name| {
            let (func, arg) = name.split_once('(').expect("measure name");
            format!(
                "{func}({arg}) AS \"{name}\"",
                arg = arg.trim_end_matches(')')
            )
        })
        .collect();
    let sql = format!(
        "SELECT {}, {} FROM {source}{} GROUP BY {}",
        keys.join(", "),
        measures.join(", "),
        ask.filter_sql
            .iter()
            .map(|f| format!(" WHERE {f}"))
            .collect::<Vec<_>>()
            .join(" AND "),
        keys.join(", "),
    );
    let batches = session.context().sql(&sql).await?.collect().await?;
    session
        .context()
        .deregister_table(source.as_str())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

    let bytes = batches
        .iter()
        .map(|b| b.get_array_memory_size() as u64)
        .sum();
    table.store_rows(id.clone(), batches.clone());
    Ok(Derived::new(
        id,
        Source {
            table: table.table_id().clone(),
            snapshot: table.snapshot(),
        },
        table.policy(),
        bytes,
        Box::new(MaterializedResult::aggregate_of(
            ask.plan.clone(),
            rollup,
            batches,
        )),
    ))
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
