//! Session-level opt-ins, carried in DataFusion's config extensions.

use datafusion::catalog::Session;
use datafusion::common::config::{ConfigExtension, ConfigOptions};
use datafusion::common::extensions_options;

extensions_options! {
    /// Quarry's session options, `SET`-able as `quarry.<name>`.
    pub struct QuarryOptions {
        /// Whether stored approximate state may serve exact queries — a
        /// sketch answering `count(distinct)`, say. `false` means only a
        /// query that asks for an approximation gets one.
        pub approximate: bool, default = false
        /// Whether a non-unionable substitute may serve a query when new data
        /// has been added since it was built — a vector index answering a
        /// top-k over a table that has grown, say. `false` means the query
        /// reads the table instead, which is correct but slower. `true` means
        /// the stored rows are served as-is, which is fast but may miss rows
        /// added since the index was built.
        ///
        /// Only additive staleness is covered: a delete still rejects, because
        /// serving rows that should not exist is fabrication, not staleness.
        pub stale: bool, default = false
    }
}

impl ConfigExtension for QuarryOptions {
    const PREFIX: &'static str = "quarry";
}

/// Whether `state`'s session accepts approximate answers for exact asks.
pub(crate) fn approximate(state: &dyn Session) -> bool {
    state
        .config_options()
        .extensions
        .get::<QuarryOptions>()
        .is_some_and(|options| options.approximate)
}

/// Whether `options` accept approximate answers — the [`Session::sql`] side,
/// which sees the plan but not a `scan` call's `Session`.
pub(crate) fn approximate_of(options: &ConfigOptions) -> bool {
    options
        .extensions
        .get::<QuarryOptions>()
        .is_some_and(|options| options.approximate)
}

/// Whether `state`'s session accepts stale substitutes for non-unionable
/// kinds when only data has been added.
pub(crate) fn stale(state: &dyn Session) -> bool {
    state
        .config_options()
        .extensions
        .get::<QuarryOptions>()
        .is_some_and(|options| options.stale)
}

/// Whether `options` accept stale substitutes — the [`Session::sql`] side,
/// which sees the plan but not a `scan` call's `Session`.
pub(crate) fn stale_of(options: &ConfigOptions) -> bool {
    options
        .extensions
        .get::<QuarryOptions>()
        .is_some_and(|options| options.stale)
}
