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
//! # What is deliberately not here yet
//!
//! Only *pruning* rewrites reach the physical plan. A substituting candidate
//! is skipped, because the bytes it would substitute are not stored anywhere
//! the engine can read from yet — the [`ResultCache`](crate::kinds::ResultCache)
//! kind records that a result exists and how big it is, not the result itself.
//! Skipping is the safe direction: the answer is right, just slower.

mod table;

pub use table::{QuarryTable, ScanReport, hash_scalar};
