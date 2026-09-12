//! A stored answer to one exact query.
//!
//! The simplest possible kind, and deliberately the first: it proves the
//! [`Kind`] trait and the registry before anything with interesting matching
//! logic arrives.
//!
//! It is also the kind with the *strongest* staleness requirement, being
//! substituting, which makes it a good early test of the rule: nothing
//! downstream re-reads the table, so a stale hit is a wrong answer rather
//! than a slow one.

use crate::cost::{Cost, PriceTable};
use crate::derived::{Kind, Query, Refreshed, Rewrite};
use crate::snapshot::Diff;

/// A stored result for one canonical plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultCache {
    plan_hash: u64,
    rows: u64,
    bytes: u64,
    unionable: bool,
}

impl ResultCache {
    /// A stored result whose rows have the same shape as the table's.
    ///
    /// The result of a filter or projection: rows added to the table after
    /// this was stored can be read alongside it, so it stays usable as the
    /// table grows.
    ///
    /// `plan_hash` must identify the *canonical* form of the plan, so that
    /// queries differing only in spelling hit the same entry.
    pub fn rows_of(plan_hash: u64, rows: u64, bytes: u64) -> Self {
        ResultCache {
            plan_hash,
            rows,
            bytes,
            unionable: true,
        }
    }

    /// A stored result that has been aggregated.
    ///
    /// A `count(*)` or `sum(x)`, which cannot simply be read alongside newly
    /// added rows — combining them needs a merge step. Usable only while the
    /// table has not moved at all.
    pub fn aggregate_of(plan_hash: u64, rows: u64, bytes: u64) -> Self {
        ResultCache {
            plan_hash,
            rows,
            bytes,
            unionable: false,
        }
    }

    /// The canonical plan this answers.
    pub fn plan_hash(&self) -> u64 {
        self.plan_hash
    }

    /// How many rows are stored.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// How many bytes are stored.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether the stored rows have the same shape as the table's.
    pub fn is_unionable(&self) -> bool {
        self.unionable
    }
}

impl Kind for ResultCache {
    fn name(&self) -> &'static str {
        "result"
    }

    /// Matches one plan exactly.
    ///
    /// Exact matching finds far less than subsumption would — a cached
    /// 30-day filter cannot serve a 7-day query here. That is deferred
    /// deliberately: exact matching is cheap and obviously correct, and its
    /// measured hit rate is what should justify building a matcher.
    fn matches(&self, query: &Query) -> Option<Rewrite> {
        (query.plan_hash == self.plan_hash).then_some(Rewrite::Substitute {
            unionable: self.unionable,
        })
    }

    fn cost(&self, prices: &PriceTable) -> Cost {
        super::price_local(prices, self.bytes)
    }

    /// A stored result cannot be brought forward in place.
    ///
    /// Updating it would require knowing the plan's structure well enough to
    /// combine the old answer with new rows, which a generic result does not
    /// carry. Recomputation is the only option, so this reports
    /// [`Refreshed::NeedsRebuild`] for any change at all.
    ///
    /// Note this is separate from serving a *stale* entry: the rule may still
    /// admit one by unioning the files added since, without the stored bytes
    /// changing.
    fn refresh(&mut self, diff: &Diff) -> Refreshed {
        if diff.is_empty() {
            Refreshed::UpToDate
        } else {
            Refreshed::NeedsRebuild
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::{PolicyFingerprint, Predicate};
    use crate::snapshot::{FileId, SnapshotId, TableId};
    use std::collections::BTreeSet;

    fn query(plan_hash: u64) -> Query {
        Query {
            table: TableId("events".into()),
            snapshot: SnapshotId(812),
            policy: PolicyFingerprint(1),
            plan_hash,
            projected: BTreeSet::from([4]),
            predicates: vec![Predicate::Eq { field: 4, value: 9 }],
        }
    }

    #[test]
    fn an_identical_plan_substitutes() {
        let c = ResultCache::rows_of(0xBEEF, 10, 100);
        assert_eq!(
            c.matches(&query(0xBEEF)),
            Some(Rewrite::Substitute { unionable: true })
        );
    }

    #[test]
    fn a_different_plan_does_not_match() {
        let c = ResultCache::rows_of(0xBEEF, 10, 100);
        assert_eq!(c.matches(&query(0xFEED)), None);
    }

    #[test]
    fn cost_scales_with_stored_bytes() {
        let prices = PriceTable::default();
        let small = ResultCache::rows_of(1, 1, 100).cost(&prices);
        let large = ResultCache::rows_of(1, 1, 100_000).cost(&prices);
        assert!(large.usd > small.usd);
        assert_eq!(small.bytes, 100);
    }

    #[test]
    fn an_unchanged_table_leaves_it_up_to_date() {
        let mut c = ResultCache::rows_of(1, 1, 1);
        assert_eq!(c.refresh(&Diff::default()), Refreshed::UpToDate);
    }

    #[test]
    fn any_change_requires_a_rebuild() {
        let mut c = ResultCache::rows_of(1, 1, 1);

        let appended = Diff {
            added: BTreeSet::from([FileId("new".into())]),
            ..Diff::default()
        };
        assert_eq!(c.refresh(&appended), Refreshed::NeedsRebuild);

        let deleted = Diff {
            redeleted: BTreeSet::from([FileId("old".into())]),
            ..Diff::default()
        };
        assert_eq!(c.refresh(&deleted), Refreshed::NeedsRebuild);
    }

    #[test]
    fn accessors_report_what_was_stored() {
        let c = ResultCache::rows_of(7, 42, 4096);
        assert_eq!(c.plan_hash(), 7);
        assert_eq!(c.rows(), 42);
        assert_eq!(c.bytes(), 4096);
        assert_eq!(c.name(), "result");
    }
}
