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
use crate::derived::{Kind, Query, Refreshed, Rewrite};
use crate::snapshot::Diff;

/// A stored answer to one canonical plan, with the rows.
#[derive(Clone, Debug)]
pub struct MaterializedResult {
    plan_hash: u64,
    batches: Vec<RecordBatch>,
    bytes: u64,
    unionable: bool,
}

impl MaterializedResult {
    /// Store table-shaped rows: the result of a filter or projection.
    ///
    /// Rows appended to the table afterwards can be read alongside these, so
    /// this stays usable as the table grows.
    pub fn rows_of(plan_hash: u64, batches: Vec<RecordBatch>) -> Self {
        let bytes = batches.iter().map(batch_bytes).sum();
        MaterializedResult {
            plan_hash,
            batches,
            bytes,
            unionable: true,
        }
    }

    /// Store an aggregated answer.
    ///
    /// Usable only while the table has not moved, since merging is not
    /// concatenating.
    pub fn aggregate_of(plan_hash: u64, batches: Vec<RecordBatch>) -> Self {
        let bytes = batches.iter().map(batch_bytes).sum();
        MaterializedResult {
            plan_hash,
            batches,
            bytes,
            unionable: false,
        }
    }

    /// The stored rows.
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// The canonical plan this answers.
    pub fn plan_hash(&self) -> u64 {
        self.plan_hash
    }
}

/// Hash a canonical plan description the way a query's `plan_hash` is built.
///
/// Exposed so a caller storing a result can key it the same way the table
/// will look it up.
///
/// # A collision here is not safe
///
/// Unlike an index probe, where a collision costs extra I/O, two *different*
/// plans hashing alike means one query is served the other's stored answer —
/// a wrong result, silently. Being 64-bit and non-cryptographic, this is
/// vanishingly unlikely by accident and trivial to arrange on purpose.
///
/// Substituting derived state therefore must not be persisted or shared
/// between principals until a match is *verified* rather than assumed, by
/// keeping the plan description beside the hash and comparing it. Pruning
/// kinds have no such restriction, which is why they are the ones being
/// written to storage first.
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
        (query.plan_hash == self.plan_hash).then_some(Rewrite::Substitute {
            unionable: self.unionable,
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
