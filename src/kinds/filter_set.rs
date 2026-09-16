//! A filter set: the rows that passed a remembered predicate.
//!
//! The taxonomy's reusable-computation layer, mapped onto the pruning
//! machinery: a query that repeats a filter another query paid to evaluate
//! can skip straight to the rows that passed it. No interior-plan
//! substitution is needed — the scan's scope carries the candidate set.
//!
//! Matching is by the filter's canonical text, which is why a query must
//! carry an exact [`Plan`] to be served at all: `plan: None` means the
//! engine could not render the filters faithfully, and matching text it
//! cannot see would be a guess.

use std::collections::{BTreeMap, BTreeSet};

use crate::cost::{Cost, PriceTable};
use crate::derived::{Filter, Kind, Query, Refreshed, Rewrite, Scope};
use crate::snapshot::{Diff, FileId};

/// Which rows of which files satisfy one filter.
///
/// Postings are `file → group → row offsets`, the same shape a
/// [`Bitmap`](super::Bitmap) produces — and they compose the same way: a
/// query asking this filter *and* another's intersects both sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterSet {
    /// The predicate this answers, as it appears in a [`Plan`]'s filters.
    filter: Filter,
    postings: BTreeMap<FileId, BTreeMap<u32, BTreeSet<u64>>>,
    bytes: u64,
}

impl FilterSet {
    /// The rows that passed `filter`, by file and group.
    pub fn of(filter: Filter) -> Self {
        FilterSet {
            filter,
            postings: BTreeMap::new(),
            bytes: 0,
        }
    }

    /// Record that `file`'s `group` satisfies the filter at `row`.
    pub fn insert(&mut self, file: FileId, group: u32, row: u64) {
        self.postings
            .entry(file)
            .or_default()
            .entry(group)
            .or_default()
            .insert(row);
    }

    /// Builder form of [`FilterSet::insert`].
    pub fn with(mut self, file: FileId, group: u32, row: u64) -> Self {
        self.insert(file, group, row);
        self
    }

    /// Bytes the postings occupy once encoded: two u64s per row.
    ///
    /// `pub`, like [`Index::encoded_len`](super::Index::encoded_len) and
    /// [`Bitmap::encoded_len`](super::Bitmap::encoded_len): it is what a
    /// storage budget is enforced against, so a caller sizing derived state
    /// needs it for every kind or for none.
    pub fn encoded_len(&self) -> u64 {
        self.postings
            .values()
            .flat_map(|groups| groups.values())
            .map(|rows| 16 * rows.len() as u64)
            .sum()
    }

    /// Declare how many bytes this set occupies.
    pub fn with_bytes(mut self, bytes: u64) -> Self {
        self.bytes = bytes;
        self
    }

    /// The filter this answers.
    pub fn filter(&self) -> &Filter {
        &self.filter
    }

    /// The postings, for persistence.
    pub fn postings(&self) -> &BTreeMap<FileId, BTreeMap<u32, BTreeSet<u64>>> {
        &self.postings
    }

    /// Bytes held, as declared at build.
    pub fn bytes_estimate(&self) -> u64 {
        self.bytes
    }
}

impl Kind for FilterSet {
    fn name(&self) -> &'static str {
        "filter-set"
    }

    /// The query must ask *this* filter — one conjunct of it, at least.
    ///
    /// A query asking the filter among others is still served: the set
    /// holds every row that passed it, which is a superset of the rows the
    /// whole conjunction passes — over-admission, the safe direction, and
    /// the remaining conjuncts still evaluate above.
    fn matches(&self, query: &Query) -> Option<Rewrite> {
        let plan = query.plan.as_ref()?;
        plan.filters.contains(&self.filter).then(|| Rewrite::Prune {
            files: self
                .postings
                .iter()
                .map(|(file, rows)| (file.clone(), Scope::Rows(rows.clone())))
                .collect(),
        })
    }

    fn cost(&self, prices: &PriceTable) -> Cost {
        super::price_local(prices, self.bytes)
    }

    /// Row positions cannot be brought forward; the filter has to run again.
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
    use crate::derived::Plan;
    use crate::snapshot::{SnapshotId, TableId};

    const STATUS: FieldId = 2;
    const POLICY: PolicyFingerprint = PolicyFingerprint(1);

    use crate::derived::FieldId;
    use crate::derived::PolicyFingerprint;

    fn f(name: &str) -> FileId {
        FileId(name.into())
    }

    fn filter() -> Filter {
        Filter {
            field: Some(STATUS),
            text: "status = Int64(1)".into(),
        }
    }

    fn set() -> FilterSet {
        FilterSet::of(filter())
            .with(f("a"), 0, 1)
            .with(f("b"), 0, 0)
            .with(f("b"), 0, 2)
            .with_bytes(256)
    }

    fn query(plan: Option<Plan>) -> Query {
        Query {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
            policy: POLICY,
            plan_hash: 1,
            plan,
            projected: BTreeSet::from([STATUS]),
            predicates: Vec::new(),
            aggregate: None,
            nearest: None,
            join: None,
            approximate: false,
        }
    }

    fn asking(filters: Vec<Filter>) -> Query {
        query(Some(Plan::new(BTreeSet::from([STATUS]), filters)))
    }

    #[test]
    fn a_query_asking_this_filter_prunes_to_its_rows() {
        let got = set().matches(&asking(vec![filter()]));
        assert_eq!(
            got,
            Some(Rewrite::Prune {
                files: BTreeMap::from([
                    (
                        f("a"),
                        Scope::Rows(BTreeMap::from([(0, BTreeSet::from([1]))]))
                    ),
                    (
                        f("b"),
                        Scope::Rows(BTreeMap::from([(0, BTreeSet::from([0, 2]))]))
                    ),
                ])
            })
        );
    }

    #[test]
    fn the_filter_among_others_still_serves() {
        // Asking the stored filter AND another is safe: the set is a
        // superset of the conjunction's rows; the rest filters above.
        let other = Filter {
            field: Some(9),
            text: "region = Utf8(\"us\")".into(),
        };
        assert!(set().matches(&asking(vec![filter(), other])).is_some());
    }

    #[test]
    fn an_unasked_filter_serves_nothing() {
        let other = Filter {
            field: Some(9),
            text: "region = Utf8(\"us\")".into(),
        };
        assert_eq!(set().matches(&asking(vec![other])), None);
    }

    #[test]
    fn without_an_exact_plan_there_is_nothing_to_match() {
        assert_eq!(set().matches(&query(None)), None);
    }

    #[test]
    fn cost_scales_with_set_size() {
        let prices = PriceTable::default();
        let small = FilterSet::of(filter()).with_bytes(1_000).cost(&prices);
        let large = FilterSet::of(filter()).with_bytes(1_000_000).cost(&prices);
        assert!(large.usd > small.usd);
    }
}
