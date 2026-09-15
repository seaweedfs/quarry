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

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use datafusion::arrow::array::RecordBatch;

use crate::cost::{Cost, PriceTable};
use crate::derived::{AggFunc, Aggregate, FieldId, Kind, Measure, Plan, Query, Refreshed, Rewrite};
use crate::snapshot::Diff;

/// A stored answer to one canonical plan, with the rows.
#[derive(Clone, Debug)]
pub struct MaterializedResult {
    plan: Plan,
    batches: Vec<RecordBatch>,
    bytes: u64,
    unionable: bool,
    /// The grain the rows are aggregated at, if they are.
    rollup: Option<Rollup>,
}

/// What a cube is aggregated at, and what its columns are called.
///
/// `spec` is the coverage decision; `names` is the serving detail — the
/// stored rows are only useful if the re-aggregation can name its columns.
#[derive(Clone, Debug)]
pub struct Rollup {
    spec: Aggregate,
    names: BTreeMap<FieldId, String>,
}

impl Rollup {
    /// A cube's grain and measures, with the table's column name per field.
    pub fn new(spec: Aggregate, names: BTreeMap<FieldId, String>) -> Self {
        Rollup { spec, names }
    }

    /// The grain the stored rows sit at.
    pub fn spec(&self) -> &Aggregate {
        &self.spec
    }

    /// The column a group key is stored under.
    pub fn key_name(&self, field: FieldId) -> Option<&str> {
        self.names.get(&field).map(String::as_str)
    }

    /// The column a measure's partials are stored under, if the cube holds
    /// the measure.
    pub fn measure_name(&self, measure: &Measure) -> Option<String> {
        let inner = match measure.field {
            Some(field) => self.names.get(&field)?.clone(),
            None => "*".to_owned(),
        };
        Some(format!(
            "{}({inner})",
            match measure.func {
                AggFunc::Count => "count",
                AggFunc::Sum => "sum",
                AggFunc::Min => "min",
                AggFunc::Max => "max",
                AggFunc::CountDistinct => "count_distinct",
                AggFunc::ApproxDistinct => "approx_distinct",
            }
        ))
    }

    /// Every column the stored schema should carry.
    pub fn columns(&self) -> Vec<String> {
        self.spec
            .group_by
            .iter()
            .filter_map(|field| self.key_name(*field).map(str::to_owned))
            .chain(
                self.spec
                    .measures
                    .iter()
                    .filter_map(|m| self.measure_name(m)),
            )
            .collect()
    }
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
            rollup: None,
        }
    }

    /// Store an aggregated answer: `plan`'s filters applied at build, its
    /// rows rolled up to `rollup`'s grain.
    ///
    /// Usable only while the table has not moved, since merging is not
    /// concatenating.
    pub fn aggregate_of(plan: Plan, rollup: Rollup, batches: Vec<RecordBatch>) -> Self {
        let bytes = batches.iter().map(batch_bytes).sum();
        MaterializedResult {
            plan,
            batches,
            bytes,
            unionable: false,
            rollup: Some(rollup),
        }
    }

    /// The grain the stored rows sit at, if this is a cube.
    pub fn rollup(&self) -> Option<&Rollup> {
        self.rollup.as_ref()
    }

    /// The stored rows.
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// The plan this answers.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Whether a cube answers `query` exactly.
    ///
    /// Three exact conditions — grain, then both directions of the filter
    /// set:
    ///
    /// ```text
    /// grain      the query's keys and measures roll up from the cube's
    /// build     every filter baked into the cube is one the query's own
    ///             filters imply — a narrower range still honours a wider
    ///             one, so `ts >= Mar` answers `ts >= Jan`'s cube
    /// query      every remaining query filter sits on a stored group key,
    ///            so it can be applied to the partials before re-aggregating
    /// ```
    ///
    /// The second direction is what makes this subsumption rather than a
    /// hash: the query may restrict a key field the cube never filtered on.
    fn covers(&self, query: &Query, rollup: &Rollup) -> bool {
        let (Some(want), Some(plan)) = (&query.aggregate, &query.plan) else {
            return false;
        };
        want.covered_by(&rollup.spec, query.approximate)
            && self
                .plan
                .filters
                .iter()
                .all(|baked| plan.filters.iter().any(|q| q.implies(baked)))
            && plan.filters.iter().all(|f| {
                self.plan.filters.contains(f)
                    || f.field
                        .is_some_and(|field| rollup.spec.group_by.contains(&field))
            })
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
        match &self.rollup {
            None => (query.plan.as_ref() == Some(&self.plan)).then_some(Rewrite::Substitute {
                unionable: self.unionable,
                rollup: None,
            }),
            Some(rollup) => self.covers(query, rollup).then_some(Rewrite::Substitute {
                unionable: false,
                rollup: Some(rollup.spec.clone()),
            }),
        }
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
