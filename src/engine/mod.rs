//! Where the rule meets a real query engine.
//!
//! Everything outside this module is dependency-free, so the logic that has to
//! be correct is tested without a query engine at all. This module is the
//! boundary: it translates DataFusion's view of a scan into a
//! [`Query`](crate::derived::Query), asks the rule which files to read, and
//! hands back a plan that reads only those.
//!
//! Behind the `engine` feature, so `cargo test` stays fast and the core stays
//! honest about having no dependencies.
//!
//! # A kind defined outside the core
//!
//! [`MaterializedResult`] lives here rather than in
//! [`kinds`](crate::kinds) because it holds Arrow `RecordBatch`es. It is the
//! test of the claim that adding a kind is a file rather than a refactor:
//! nothing in `derived`, `registry`, or `explain` knows it exists, and none of
//! them changed to accommodate it.

mod ann;
mod build;
mod cache;
mod cube;
mod distance;
mod hll;
#[cfg(feature = "iceberg")]
mod iceberg_table;
mod join;
mod materialized;
mod optimizer;
mod options;
#[cfg(feature = "iceberg")]
mod persist;
mod quarry;
mod store;
mod table;
mod text;

pub use build::{
    build_bitmap, build_cube, build_filter_set, build_index, build_join_hash,
    build_proposed_filter_set, build_proposed_index, build_proposed_text_index, build_text_index,
    build_vector_index, columns, cube_id, estimate_overlap, filter_set_id, index_id, join_hash_id,
    join_hash_id_from, parquet_bounds, text_index_id, vector_index_id,
};
pub use cache::{CacheStats, RangeCache};
#[cfg(feature = "iceberg")]
pub use iceberg_table::{arrow_schema, field_ids, table_from_catalog, table_from_iceberg};
pub use materialized::{MaterializedResult, Rollup, hash_plan};
pub use optimizer::{
    BuildKind, BuildRecommendation, Calibration, Declined, Optimizer, Recommendation,
    RetireRecommendation, Retired, Round,
};
pub use options::QuarryOptions;
#[cfg(feature = "iceberg")]
pub use persist::{
    Advertised, Layout, QUARRY_BITMAP_V1, QUARRY_EQ_INDEX_V2, QUARRY_FILTER_SET_V1,
    QUARRY_JOIN_HASH_V1, QUARRY_PATH_PROPERTY, QUARRY_TEXT_INDEX_V1, QUARRY_VECTOR_INDEX_V1,
    QUARRY_WORKLOAD_V1, Recovered, Store, advertised, discard, read_bitmap, read_filter_set,
    read_index, read_join_hash, read_text_index, read_vector_index, read_vector_rows,
    read_workload, recover, write_bitmap, write_filter_set, write_index, write_join_hash,
    write_manifest, write_text_index, write_vector_index, write_workload,
};
pub use quarry::{Quarry, Session};
pub use store::{BudgetExceeded, MeteredStore, StoreStats};
pub use table::{QuarryTable, ScanReport, SharedRegistry, hash_scalar, shared};
pub use text::terms;
