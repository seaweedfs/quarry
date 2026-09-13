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

mod cache;
mod materialized;
mod quarry;
mod store;
mod table;

pub use cache::{CacheStats, RangeCache};
pub use materialized::{MaterializedResult, hash_plan};
pub use quarry::{Quarry, Session};
pub use store::{BudgetExceeded, MeteredStore, StoreStats};
pub use table::{QuarryTable, ScanReport, hash_scalar};
