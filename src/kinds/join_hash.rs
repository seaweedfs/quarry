//! A join hash: the build-side rows of a join, sorted by the join key and held
//! in memory so a hash join reads them locally instead of from object storage.
//!
//! DataFusion's `HashJoinExec` keeps its built hash table private, so the
//! hash table is rebuilt on every query. What *is* persistable and reusable
//! is the build-side relation itself — and sorting it by the join key at
//! build time is a pre-computation that costs nothing per query and enables
//! alternative join strategies (sort-merge, index nested loops) should the
//! probe side also be sorted.
//!
//! The win is I/O, not computation: the rows are local where the table's
//! Parquet objects are remote. The hash table build is O(rows in the build
//! side), which for the typical small-dimension-table build side is negligible
//! beside the I/O it saves.
//!
//! Like [`VectorIndex`](super::VectorIndex), the rows live in the table's
//! materialized store (through `QuarryTable::store_rows`); this kind only
//! says which joins may use them. Also like the vector index, it declares
//! itself non-unionable: the sorted rows cannot simply be concatenated with
//! appended rows because the sort order would be wrong, and a re-sort is a
//! rebuild.

use std::collections::BTreeSet;

use crate::cost::{Cost, PriceTable};
use crate::derived::{FieldId, Kind, Query, Refreshed, Rewrite};
use crate::snapshot::Diff;

/// The build-side rows of a join, sorted by the join key.
#[derive(Debug)]
pub struct JoinHash {
    /// The join key fields on the build side.
    keys: BTreeSet<FieldId>,
    /// All columns the stored rows cover: join keys plus whatever the join
    /// reads from the build side.
    columns: BTreeSet<FieldId>,
    /// Stored bytes, for pricing.
    bytes: u64,
}

impl JoinHash {
    /// A join hash over `keys`, covering `columns`, occupying `bytes`.
    pub fn covering(
        keys: impl IntoIterator<Item = FieldId>,
        columns: impl IntoIterator<Item = FieldId>,
        bytes: u64,
    ) -> Self {
        JoinHash {
            keys: keys.into_iter().collect(),
            columns: columns.into_iter().collect(),
            bytes,
        }
    }

    /// The join keys and columns this hash covers.
    pub fn covers(&self) -> (&BTreeSet<FieldId>, &BTreeSet<FieldId>) {
        (&self.keys, &self.columns)
    }
}

impl Kind for JoinHash {
    fn name(&self) -> &'static str {
        "join_hash"
    }

    /// Matches a join ask whose keys are exactly this hash's keys and whose
    /// projected columns are a subset of what the stored rows cover.
    ///
    /// The join keys must match exactly — a different key set is a different
    /// join shape, and the stored rows were sorted for this one. The projected
    /// columns must be covered, or the substituted rows would lack a column
    /// the plan references.
    fn matches(&self, query: &Query) -> Option<Rewrite> {
        let keys = query.join.as_ref()?;
        if keys != &self.keys {
            return None;
        }
        let mut needed = query.projected.clone();
        needed.extend(query.filtered_fields());
        if !needed.is_subset(&self.columns) {
            return None;
        }
        Some(Rewrite::Substitute {
            // The sorted rows cannot be unioned with appended rows without
            // re-sorting, which is a rebuild. Declaring non-unionable makes
            // the rule reject an appended-to table by name (Reason::
            // ResidualNotUnionable) rather than the serve path declining
            // quietly.
            unionable: false,
            rollup: None,
        })
    }

    fn cost(&self, prices: &PriceTable) -> Cost {
        super::price_local(prices, self.bytes)
    }

    /// Stored rows cannot be brought forward in place; the join hash must be
    /// rebuilt.
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
    use crate::derived::{Derived, DerivedId, PolicyFingerprint, Source};
    use crate::snapshot::{Snapshot, SnapshotGraph, SnapshotId, TableId};

    const POLICY: PolicyFingerprint = PolicyFingerprint(1);
    const KEY: FieldId = 1;
    const VAL: FieldId = 2;

    fn hash() -> JoinHash {
        JoinHash::covering([KEY], [KEY, VAL], 4096)
    }

    fn query_with(join: Option<BTreeSet<FieldId>>, projected: BTreeSet<FieldId>) -> Query {
        Query {
            table: TableId("orders".into()),
            snapshot: SnapshotId(810),
            policy: POLICY,
            plan_hash: 1,
            plan: None,
            projected,
            predicates: Vec::new(),
            aggregate: None,
            nearest: None,
            join,
            approximate: false,
        }
    }

    #[test]
    fn a_matching_join_substitutes() {
        let got = hash().matches(&query_with(
            Some(BTreeSet::from([KEY])),
            BTreeSet::from([KEY, VAL]),
        ));
        assert_eq!(
            got,
            Some(Rewrite::Substitute {
                unionable: false,
                rollup: None
            })
        );
    }

    #[test]
    fn a_wrong_key_does_not_match() {
        assert_eq!(
            hash().matches(&query_with(
                Some(BTreeSet::from([VAL])),
                BTreeSet::from([KEY, VAL])
            )),
            None
        );
    }

    #[test]
    fn a_missing_column_does_not_match() {
        assert_eq!(
            hash().matches(&query_with(
                Some(BTreeSet::from([KEY])),
                BTreeSet::from([KEY, VAL, 99])
            )),
            None
        );
    }

    #[test]
    fn no_join_does_not_match() {
        assert_eq!(
            hash().matches(&query_with(None, BTreeSet::from([KEY, VAL]))),
            None
        );
    }

    #[test]
    fn cost_scales_with_size() {
        let prices = PriceTable::default();
        let small = JoinHash::covering([KEY], [KEY, VAL], 1_000).cost(&prices);
        let large = JoinHash::covering([KEY], [KEY, VAL], 1_000_000).cost(&prices);
        assert!(large.usd > small.usd);
    }

    #[test]
    fn refresh_reports_up_to_date_only_when_nothing_changed() {
        let mut h = hash();
        assert_eq!(h.refresh(&Diff::default()), Refreshed::UpToDate);
        let appended = Diff {
            added: std::collections::BTreeSet::from([crate::snapshot::FileId("new".into())]),
            ..Diff::default()
        };
        assert_eq!(h.refresh(&appended), Refreshed::NeedsRebuild);
    }

    fn derived_hash(at: SnapshotId) -> Derived {
        Derived::new(
            DerivedId("jh".into()),
            Source {
                table: TableId("orders".into()),
                snapshot: at,
            },
            POLICY,
            4096,
            Box::new(hash()),
        )
    }

    #[test]
    fn an_append_rejects_rather_than_union() {
        let g = SnapshotGraph::new()
            .with(
                Snapshot::root(SnapshotId(810))
                    .with_clean_file(crate::snapshot::FileId("a".into())),
            )
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                    .with_clean_file(crate::snapshot::FileId("a".into()))
                    .with_clean_file(crate::snapshot::FileId("b".into())),
            );
        let mut q = query_with(Some(BTreeSet::from([KEY])), BTreeSet::from([KEY, VAL]));
        q.snapshot = SnapshotId(811);
        assert_eq!(
            derived_hash(SnapshotId(810)).may_serve(&q, &g),
            crate::derived::Decision::Reject(crate::derived::Reason::ResidualNotUnionable)
        );
    }

    #[test]
    fn an_unchanged_table_admits() {
        let g = SnapshotGraph::new().with(
            Snapshot::root(SnapshotId(810)).with_clean_file(crate::snapshot::FileId("a".into())),
        );
        assert!(
            derived_hash(SnapshotId(810))
                .may_serve(
                    &query_with(Some(BTreeSet::from([KEY])), BTreeSet::from([KEY, VAL])),
                    &g
                )
                .is_admitted()
        );
    }

    #[test]
    fn a_delete_disqualifies() {
        let g = SnapshotGraph::new()
            .with(
                Snapshot::root(SnapshotId(810))
                    .with_clean_file(crate::snapshot::FileId("a".into())),
            )
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810)).with_file(
                    crate::snapshot::FileId("a".into()),
                    crate::snapshot::DeleteState(7),
                ),
            );
        let mut q = query_with(Some(BTreeSet::from([KEY])), BTreeSet::from([KEY, VAL]));
        q.snapshot = SnapshotId(811);
        assert!(
            !derived_hash(SnapshotId(810))
                .may_serve(&q, &g)
                .is_admitted(),
            "a delete means the stored rows hold deleted ones"
        );
    }
}
