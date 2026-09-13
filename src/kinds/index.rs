//! An equality index over one field: which files can contain which values.
//!
//! The counterpart to [`ResultCache`](super::ResultCache), and the reason the
//! two are worth having early: this one *prunes* rather than *substitutes*, so
//! between them they cover both halves of the rule's staleness asymmetry.
//!
//! Pruning is the weaker, safer obligation. Naming too many files is merely
//! slow, because the engine still reads real data and applies deletes. Naming
//! too few would lose rows — which is why the rule unions in any files added
//! since the index was built, rather than trusting the index to be complete.

use std::collections::{BTreeMap, BTreeSet};

use crate::cost::{Cost, PriceTable};
use crate::derived::{FieldId, Kind, Query, Refreshed, Rewrite};
use crate::snapshot::{Diff, FileId};

/// Which files contain which values of one field.
///
/// One field, not several. A composite index is a different matching problem —
/// prefix rules, column order — and belongs in its own kind rather than as a
/// flag on this one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Index {
    field: FieldId,
    postings: BTreeMap<u64, BTreeSet<FileId>>,
    bytes: u64,
}

impl Index {
    /// An index over `field` with no entries.
    pub fn new(field: FieldId) -> Self {
        Index {
            field,
            postings: BTreeMap::new(),
            bytes: 0,
        }
    }

    /// Record that `file` contains rows where `field` hashes to `value`.
    pub fn insert(&mut self, value: u64, file: FileId) {
        self.postings.entry(value).or_default().insert(file);
    }

    /// Builder form of [`Index::insert`].
    pub fn with(mut self, value: u64, file: FileId) -> Self {
        self.insert(value, file);
        self
    }

    /// Declare how many bytes this index occupies.
    ///
    /// Set explicitly rather than computed: the in-memory layout here is not
    /// the on-disk layout that will eventually be priced.
    pub fn with_bytes(mut self, bytes: u64) -> Self {
        self.bytes = bytes;
        self
    }

    /// The field indexed.
    pub fn field(&self) -> FieldId {
        self.field
    }

    /// How many distinct values are indexed.
    pub fn values(&self) -> usize {
        self.postings.len()
    }

    /// Files recorded as containing `value`.
    pub fn files_for(&self, value: u64) -> Option<&BTreeSet<FileId>> {
        self.postings.get(&value)
    }

    /// How many bytes this index was declared to occupy.
    pub fn bytes_estimate(&self) -> u64 {
        self.bytes
    }

    /// Every hashed value and the files holding it, in a fixed order.
    ///
    /// Ordered because it is written to storage: a `BTreeMap` makes the bytes
    /// a function of the contents alone, so the same index encodes
    /// identically every time and two writers cannot disagree.
    pub fn postings(&self) -> impl Iterator<Item = (u64, &BTreeSet<FileId>)> {
        self.postings.iter().map(|(value, files)| (*value, files))
    }
}

impl Kind for Index {
    fn name(&self) -> &'static str {
        "index"
    }

    /// Prunes to the files that can contain any of the queried values.
    ///
    /// Only equality predicates on the indexed field are probed. A predicate
    /// the optimizer does not model contributes nothing, so a query filtering
    /// the indexed field only opaquely does not match at all — better to scan
    /// than to prune on a predicate whose meaning is unknown.
    ///
    /// Several equalities on the same field are treated as alternatives and
    /// unioned, which is what `IN (…)` and `a = 1 OR a = 2` reduce to.
    ///
    /// A value with no postings yields an *empty* file set, which is a useful
    /// answer rather than a failure: no file at this snapshot can match. The
    /// rule still adds any files appended since.
    fn matches(&self, query: &Query) -> Option<Rewrite> {
        let values = query.equalities(self.field);
        if values.is_empty() {
            return None;
        }
        let mut files = BTreeSet::new();
        for value in values {
            if let Some(posting) = self.postings.get(&value) {
                files.extend(posting.iter().cloned());
            }
        }
        Some(Rewrite::Prune { files })
    }

    fn cost(&self, prices: &PriceTable) -> Cost {
        super::price_local(prices, self.bytes)
    }

    /// Extending an index means reading the data files that were added, which
    /// this type cannot do, so any change reports
    /// [`Refreshed::NeedsRebuild`].
    ///
    /// Less pressing than it sounds: a stale index stays *usable* under the
    /// rule, which unions the added files. Refresh is an efficiency concern
    /// here, not a correctness one.
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
    use crate::derived::{
        Decision, Derived, DerivedId, PolicyFingerprint, Predicate, Reason, Source,
    };
    use crate::snapshot::{DeleteState, Snapshot, SnapshotGraph, SnapshotId, TableId};

    const POLICY: PolicyFingerprint = PolicyFingerprint(1);
    const TENANT: FieldId = 4;

    fn f(name: &str) -> FileId {
        FileId(name.to_owned())
    }

    fn index() -> Index {
        Index::new(TENANT)
            .with(100, f("a"))
            .with(100, f("b"))
            .with(200, f("c"))
            .with_bytes(4096)
    }

    fn query_with(predicates: Vec<Predicate>) -> Query {
        Query {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
            policy: POLICY,
            plan_hash: 1,
            plan: None,
            projected: BTreeSet::from([TENANT]),
            predicates,
        }
    }

    fn eq(field: FieldId, value: u64) -> Predicate {
        Predicate::Eq { field, value }
    }

    #[test]
    fn an_equality_prunes_to_the_files_holding_that_value() {
        let got = index().matches(&query_with(vec![eq(TENANT, 100)]));
        assert_eq!(
            got,
            Some(Rewrite::Prune {
                files: BTreeSet::from([f("a"), f("b")])
            })
        );
    }

    #[test]
    fn alternatives_on_one_field_are_unioned() {
        let got = index().matches(&query_with(vec![eq(TENANT, 100), eq(TENANT, 200)]));
        assert_eq!(
            got,
            Some(Rewrite::Prune {
                files: BTreeSet::from([f("a"), f("b"), f("c")])
            })
        );
    }

    #[test]
    fn an_absent_value_prunes_to_nothing() {
        // Not a failure to match: no file at this snapshot can contain it.
        let got = index().matches(&query_with(vec![eq(TENANT, 999)]));
        assert_eq!(
            got,
            Some(Rewrite::Prune {
                files: BTreeSet::new()
            })
        );
    }

    #[test]
    fn an_unmodelled_predicate_on_the_indexed_field_does_not_match() {
        let got = index().matches(&query_with(vec![Predicate::Opaque { field: TENANT }]));
        assert_eq!(
            got, None,
            "better to scan than to prune on a predicate we cannot interpret"
        );
    }

    #[test]
    fn a_predicate_on_another_field_does_not_match() {
        let got = index().matches(&query_with(vec![eq(99, 100)]));
        assert_eq!(got, None);
    }

    #[test]
    fn no_predicates_do_not_match() {
        assert_eq!(index().matches(&query_with(vec![])), None);
    }

    #[test]
    fn cost_scales_with_index_size() {
        let prices = PriceTable::default();
        let small = Index::new(TENANT).with_bytes(1_000).cost(&prices);
        let large = Index::new(TENANT).with_bytes(1_000_000).cost(&prices);
        assert!(large.usd > small.usd);
    }

    #[test]
    fn refresh_reports_up_to_date_only_when_nothing_changed() {
        let mut i = index();
        assert_eq!(i.refresh(&Diff::default()), Refreshed::UpToDate);
        let appended = Diff {
            added: BTreeSet::from([f("new")]),
            ..Diff::default()
        };
        assert_eq!(i.refresh(&appended), Refreshed::NeedsRebuild);
    }

    #[test]
    fn accessors_report_what_was_built() {
        let i = index();
        assert_eq!(i.field(), TENANT);
        assert_eq!(i.values(), 2);
        assert_eq!(i.files_for(100), Some(&BTreeSet::from([f("a"), f("b")])));
        assert_eq!(i.files_for(999), None);
        assert_eq!(i.name(), "index");
    }

    // The point of having both kinds early: the same staleness that
    // disqualifies a substituting kind leaves a pruning one usable.

    fn derived_index(at: SnapshotId) -> Derived {
        Derived::new(
            DerivedId("idx".into()),
            Source {
                table: TableId("events".into()),
                snapshot: at,
            },
            POLICY,
            4096,
            Box::new(index()),
        )
    }

    #[test]
    fn a_stale_index_is_usable_with_the_files_added_since() {
        let g = SnapshotGraph::new()
            .with(
                Snapshot::root(SnapshotId(810))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b"))
                    .with_clean_file(f("c")),
            )
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b"))
                    .with_clean_file(f("c"))
                    .with_clean_file(f("d")),
            );

        let mut q = query_with(vec![eq(TENANT, 100)]);
        q.snapshot = SnapshotId(811);

        assert_eq!(
            derived_index(SnapshotId(810)).may_serve(&q, &g),
            Decision::UseWith {
                rewrite: Rewrite::Prune {
                    files: BTreeSet::from([f("a"), f("b")])
                },
                also_scan: BTreeSet::from([f("d")]),
            },
            "the index has never seen file d; scanning it is what keeps rows from being lost"
        );
    }

    #[test]
    fn a_delete_leaves_a_pruning_index_usable() {
        let g = SnapshotGraph::new()
            .with(Snapshot::root(SnapshotId(810)).with_clean_file(f("a")))
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                    .with_file(f("a"), DeleteState(7)),
            );

        let mut q = query_with(vec![eq(TENANT, 100)]);
        q.snapshot = SnapshotId(811);

        let decision = derived_index(SnapshotId(810)).may_serve(&q, &g);
        assert_ne!(
            decision,
            Decision::Reject(Reason::SubtractiveChange),
            "the engine still reads the file and applies deletes as it goes"
        );
        assert!(decision.is_admitted());
    }
}
