//! A flat vector index: all rows stored in memory, served in place of the
//! scan so a top-k sort re-ranks them without touching the object store.
//!
//! The first vector kind is deliberately the simplest one that is correct by
//! construction. It holds every row table-shaped — the same shape
//! [`Projection`](super::Projection) keeps — and substitutes them for the
//! scan. The `Sort { distance(col, v) } LIMIT k` above the scan does the
//! real work: it computes exact distances against the substituted rows and
//! keeps the top k. Nothing is approximate, so no opt-in is required.
//!
//! The win is I/O, not computation: the rows are local where the table's
//! Parquet objects are remote. A future approximate kind (IVF, HNSW) would
//! return only candidates and would need the `quarry.approximate` opt-in,
//! because candidates can miss true neighbours. This one cannot.

use crate::cost::{Cost, PriceTable};
use crate::derived::{FieldId, Kind, Metric, Nearest, Query, Refreshed, Rewrite};
use crate::snapshot::Diff;

/// A flat vector index over one field, under one metric and dimension.
#[derive(Debug)]
pub struct VectorIndex {
    field: FieldId,
    metric: Metric,
    dimension: u32,
    bytes: u64,
}

impl VectorIndex {
    /// A flat index over `field` measuring with `metric` in `dimension`.
    pub fn new(field: FieldId, metric: Metric, dimension: u32, bytes: u64) -> Self {
        VectorIndex {
            field,
            metric,
            dimension,
            bytes,
        }
    }

    /// The field, metric, and dimension this index covers.
    pub fn covers(&self) -> (FieldId, Metric, u32) {
        (self.field, self.metric, self.dimension)
    }
}

impl Kind for VectorIndex {
    fn name(&self) -> &'static str {
        "vector"
    }

    /// Matches a top-k ask on this index's field, metric, and dimension.
    ///
    /// The query vector is not compared — it is the probe, not the identity,
    /// and the plan above re-ranks with it. A mismatched metric or dimension
    /// means the stored rows cannot answer this query's sort, so it does not
    /// match.
    fn matches(&self, query: &Query) -> Option<Rewrite> {
        let Nearest {
            field,
            metric,
            dimension,
            ..
        } = query.nearest.as_ref()?;
        if *field == self.field && *metric == self.metric && *dimension == self.dimension {
            Some(Rewrite::Substitute {
                // `false`, though the rows *are* table-shaped and could in
                // principle be read alongside an appended file. A top-k is
                // why not: the k nearest of (stored ∪ residual) is not the k
                // nearest of the stored rows plus the k nearest of the
                // residual, so a union would have to re-rank across both —
                // which the leaf-swap rewrite does not do.
                //
                // Claiming `true` would make the rule admit a `UseWith` the
                // serve path then declines, hiding the reason. Declaring it
                // here means the rule rejects with
                // `Reason::ResidualNotUnionable` and `EXPLAIN` says so.
                unionable: false,
                rollup: None,
            })
        } else {
            None
        }
    }

    fn cost(&self, prices: &PriceTable) -> Cost {
        super::price_local(prices, self.bytes)
    }

    /// Stored rows cannot be brought forward in place; the index must be
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
    use std::collections::BTreeSet;

    use crate::derived::{Derived, DerivedId, PolicyFingerprint, Source};
    use crate::snapshot::{Snapshot, SnapshotGraph, SnapshotId, TableId};

    const POLICY: PolicyFingerprint = PolicyFingerprint(1);
    const VEC: FieldId = 9;

    fn index() -> VectorIndex {
        VectorIndex::new(VEC, Metric::L2, 3, 4096)
    }

    fn query_with(nearest: Option<Nearest>) -> Query {
        Query {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
            policy: POLICY,
            plan_hash: 1,
            plan: None,
            projected: BTreeSet::from([VEC]),
            predicates: Vec::new(),
            aggregate: None,
            nearest,
            approximate: false,
        }
    }

    #[test]
    fn a_matching_nearest_substitutes() {
        let ask = Nearest {
            field: VEC,
            metric: Metric::L2,
            dimension: 3,
            k: 10,
        };
        let got = index().matches(&query_with(Some(ask)));
        assert_eq!(
            got,
            Some(Rewrite::Substitute {
                unionable: false,
                rollup: None
            })
        );
    }

    #[test]
    fn a_wrong_metric_does_not_match() {
        let ask = Nearest {
            field: VEC,
            metric: Metric::Cosine,
            dimension: 3,
            k: 10,
        };
        assert_eq!(index().matches(&query_with(Some(ask))), None);
    }

    #[test]
    fn a_wrong_dimension_does_not_match() {
        let ask = Nearest {
            field: VEC,
            metric: Metric::L2,
            dimension: 128,
            k: 10,
        };
        assert_eq!(index().matches(&query_with(Some(ask))), None);
    }

    #[test]
    fn a_wrong_field_does_not_match() {
        let ask = Nearest {
            field: 99,
            metric: Metric::L2,
            dimension: 3,
            k: 10,
        };
        assert_eq!(index().matches(&query_with(Some(ask))), None);
    }

    #[test]
    fn no_nearest_does_not_match() {
        assert_eq!(index().matches(&query_with(None)), None);
    }

    #[test]
    fn cost_scales_with_index_size() {
        let prices = PriceTable::default();
        let small = VectorIndex::new(VEC, Metric::L2, 3, 1_000).cost(&prices);
        let large = VectorIndex::new(VEC, Metric::L2, 3, 1_000_000).cost(&prices);
        assert!(large.usd > small.usd);
    }

    #[test]
    fn refresh_reports_up_to_date_only_when_nothing_changed() {
        let mut v = index();
        assert_eq!(v.refresh(&Diff::default()), Refreshed::UpToDate);
        let appended = Diff {
            added: std::collections::BTreeSet::from([crate::snapshot::FileId("new".into())]),
            ..Diff::default()
        };
        assert_eq!(v.refresh(&appended), Refreshed::NeedsRebuild);
    }

    fn derived_vector(at: SnapshotId) -> Derived {
        Derived::new(
            DerivedId("vec".into()),
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
    fn an_append_makes_the_rule_reject_rather_than_union() {
        // The k nearest of (stored ∪ appended) is not the k nearest of each,
        // so there is no union that answers the query — and the rule says so
        // by name rather than the serve path declining quietly.
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

        let ask = Nearest {
            field: VEC,
            metric: Metric::L2,
            dimension: 3,
            k: 10,
        };
        let mut q = query_with(Some(ask));
        q.snapshot = SnapshotId(811);

        assert_eq!(
            derived_vector(SnapshotId(810)).may_serve(&q, &g),
            crate::derived::Decision::Reject(crate::derived::Reason::ResidualNotUnionable)
        );
    }

    #[test]
    fn an_unchanged_table_admits_the_index() {
        let g = SnapshotGraph::new().with(
            Snapshot::root(SnapshotId(810)).with_clean_file(crate::snapshot::FileId("a".into())),
        );
        let ask = Nearest {
            field: VEC,
            metric: Metric::L2,
            dimension: 3,
            k: 10,
        };
        assert!(
            derived_vector(SnapshotId(810))
                .may_serve(&query_with(Some(ask)), &g)
                .is_admitted()
        );
    }

    #[test]
    fn a_delete_disqualifies_a_substituting_vector_index() {
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

        let ask = Nearest {
            field: VEC,
            metric: Metric::L2,
            dimension: 3,
            k: 10,
        };
        let mut q = query_with(Some(ask));
        q.snapshot = SnapshotId(811);

        let decision = derived_vector(SnapshotId(810)).may_serve(&q, &g);
        assert!(
            !decision.is_admitted(),
            "a delete means the stored rows hold deleted ones"
        );
    }
}
