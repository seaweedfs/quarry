//! A result cache that actually holds the rows.
//!
//! This kind is defined *outside* the core, in the engine module, because it
//! holds Arrow `RecordBatch`es. It is the test of the design's claim that
//! adding a kind is a file rather than a refactor: nothing in `derived`,
//! `registry`, or `explain` knows this type exists, and none of them changed
//! to accommodate it.
//!
//! It also completes the substitution path. [`ResultCache`](crate::kinds::ResultCache)
//! records that a result exists and how big it is, which is enough to plan
//! with but not to answer from. This one can answer.

use std::hash::{Hash, Hasher};

use datafusion::arrow::array::RecordBatch;

use crate::cost::{Cost, PriceTable};
use crate::derived::{Kind, Plan, Query, Refreshed, Rewrite};
use crate::snapshot::Diff;

/// A stored answer to one canonical plan, with the rows.
#[derive(Clone, Debug)]
pub struct MaterializedResult {
    plan: Plan,
    batches: Vec<RecordBatch>,
    bytes: u64,
    unionable: bool,
}

impl MaterializedResult {
    /// Store table-shaped rows: the result of a filter or projection.
    ///
    /// Rows appended to the table afterwards can be read alongside these, so
    /// this stays usable as the table grows.
    /// The stored rows must be the **complete** answer to `plan`. DataFusion
    /// passes `scan` a row limit as a hint and applies `LIMIT` above the scan
    /// itself, so returning more rows than asked for is safe and returning
    /// fewer is not: storing a truncated result here would silently shorten
    /// every later query that matches.
    pub fn rows_of(plan: Plan, batches: Vec<RecordBatch>) -> Self {
        let bytes = batches.iter().map(batch_bytes).sum();
        MaterializedResult {
            plan,
            batches,
            bytes,
            unionable: true,
        }
    }

    /// Store an aggregated answer.
    ///
    /// Usable only while the table has not moved, since merging is not
    /// concatenating.
    pub fn aggregate_of(plan: Plan, batches: Vec<RecordBatch>) -> Self {
        let bytes = batches.iter().map(batch_bytes).sum();
        MaterializedResult {
            plan,
            batches,
            bytes,
            unionable: false,
        }
    }

    /// The stored rows.
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// The plan this answers.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
}

/// A short name for a plan, for keying stored bytes.
///
/// # This names; it does not identify
///
/// Two different plans can share a 64-bit hash. Nothing decides that a query
/// may be served from stored rows on the strength of this value — matching
/// compares the [`Plan`] itself — so a collision here costs a redundant
/// lookup, never a wrong answer.
///
/// That was not always true. Matching used to compare hashes, which meant a
/// collision handed one query another's result, silently: vanishingly unlikely
/// by accident and trivial to arrange deliberately.
pub fn hash_plan<H: Hash>(parts: &H) -> u64 {
    let mut hasher = crate::stable_hash::StableHasher::new();
    parts.hash(&mut hasher);
    hasher.finish()
}

fn batch_bytes(batch: &RecordBatch) -> u64 {
    batch.get_array_memory_size() as u64
}

impl Kind for MaterializedResult {
    fn name(&self) -> &'static str {
        "materialized"
    }

    fn matches(&self, query: &Query) -> Option<Rewrite> {
        (query.plan.as_ref() == Some(&self.plan)).then_some(Rewrite::Substitute {
            unionable: self.unionable,
            rollup: None,
        })
    }

    fn cost(&self, prices: &PriceTable) -> Cost {
        crate::kinds::price_local(prices, self.bytes)
    }

    /// Stored rows cannot be brought forward in place; the plan has to run
    /// again. Serving a *stale* entry is separate, and the rule handles it.
    fn refresh(&mut self, diff: &Diff) -> Refreshed {
        if diff.is_empty() {
            Refreshed::UpToDate
        } else {
            Refreshed::NeedsRebuild
        }
    }
}
