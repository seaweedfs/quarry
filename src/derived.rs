//! Derived state, and the one rule that decides when it may replace a scan.
//!
//! Derived state is bytes computed from a table at a known snapshot that can
//! answer some queries more cheaply than the table can: a cached result, an
//! index, a projection, a pre-aggregate. All of them are the same thing with
//! different [`Kind`]s, so all of them are admitted by one function,
//! [`Derived::may_serve`].
//!
//! Checking admissibility in exactly one place is what makes a wrong answer
//! prevented by construction rather than by care. A new kind implements three
//! methods and inherits the correctness argument.

use std::collections::BTreeSet;
use std::fmt;

use crate::cost::{Cost, PriceTable};
use crate::snapshot::{Diff, FileId, SnapshotGraph, SnapshotId, TableId};

/// Identifies one piece of derived state.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DerivedId(pub String);

/// A hash of the row filters, column masks, and grants that apply to a
/// principal reading the source tables.
///
/// Part of a piece of derived state's identity, because two principals issuing
/// the same query may be entitled to different rows. Reusing one principal's
/// derived state for another with a different effective policy would leak
/// them. Different policy means different derived state, at the cost of a
/// lower hit rate — which is the correct trade, not one to tune away.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PolicyFingerprint(pub u64);

/// A field within a table, identified the way Iceberg identifies it.
///
/// Field *ids*, never names: renaming a column must not silently invalidate or
/// mismatch the derived state built from it.
pub type FieldId = u32;

/// What a query needs, reduced to what the rule and the kinds have to reason
/// about.
///
/// A stand-in for a real logical plan until the engine is wired up in phase 9.
/// It carries identity (`plan_hash`), shape (`projected`, `filtered`), and the
/// context that admissibility depends on (`table`, `snapshot`, `policy`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    /// The table being read.
    pub table: TableId,
    /// The snapshot being read.
    pub snapshot: SnapshotId,
    /// The effective policy of the principal issuing the query.
    pub policy: PolicyFingerprint,
    /// Identity of the canonical logical plan.
    ///
    /// Canonical, so that `status = 500` and `500 = status` hash equal.
    pub plan_hash: u64,
    /// Fields the query reads.
    pub projected: BTreeSet<FieldId>,
    /// Fields the query has predicates on.
    pub filtered: BTreeSet<FieldId>,
}

/// How derived state changes a plan.
///
/// The distinction carries the design's central safety property, so it is a
/// type rather than a convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rewrite {
    /// Narrow which files must be read.
    ///
    /// The engine still reads real data and applies deletes, so naming too
    /// many files is merely slow. Naming too *few* would lose rows, which is
    /// why files added since the derived state was built are unioned back in
    /// by [`Derived::may_serve`].
    Prune {
        /// Files that may contain matching rows.
        files: BTreeSet<FileId>,
    },
    /// Replace the data source with the derived state.
    ///
    /// Nothing downstream re-reads the table, so staleness is not slow, it is
    /// wrong.
    Substitute,
}

impl Rewrite {
    /// Whether this rewrite replaces the data source rather than narrowing it.
    pub fn is_substituting(&self) -> bool {
        matches!(self, Rewrite::Substitute)
    }
}

/// Why a piece of derived state was not admitted.
///
/// Kept specific so that `EXPLAIN` can say why a query was slower than the
/// user expected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Built from a different table.
    WrongTable,
    /// The kind cannot answer this query.
    NoMatch,
    /// The queried snapshot does not descend from the one it was built from,
    /// so it may have been built on an abandoned branch.
    NotDescendant,
    /// The queried snapshot, or the one it was built from, is unknown.
    UnknownSnapshot,
    /// Built under a different effective policy.
    PolicyMismatch,
    /// Rows have been removed since it was built, so it holds rows that are
    /// no longer live. Unusable, not repairable.
    SubtractiveChange,
}

/// The rule's verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Apply the rewrite; nothing else is needed.
    Use(Rewrite),
    /// Apply the rewrite, and also scan these files, which were added after
    /// the derived state was built.
    UseWith {
        /// The rewrite to apply.
        rewrite: Rewrite,
        /// Files to read alongside it.
        also_scan: BTreeSet<FileId>,
    },
    /// Scan the table.
    Reject(Reason),
}

impl Decision {
    /// Whether the derived state may be used at all.
    pub fn is_admitted(&self) -> bool {
        !matches!(self, Decision::Reject(_))
    }
}

/// Whether a refresh brought derived state up to date.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refreshed {
    /// Up to date with the target snapshot.
    UpToDate,
    /// Cannot be updated in place; rebuild from the source table.
    NeedsRebuild,
}

/// A kind of derived state.
///
/// Three methods. Adding a kind should touch no other file: the registry, the
/// rule, and `EXPLAIN` all work in terms of this trait.
pub trait Kind: fmt::Debug {
    /// A short name, for `EXPLAIN`.
    fn name(&self) -> &'static str;

    /// Whether and how this can answer `query`.
    ///
    /// Concerned only with *shape* — the columns, predicates, or plan identity
    /// it can serve. Snapshot lineage, policy, and staleness are the rule's
    /// job, not the kind's, so that no kind can forget them.
    fn matches(&self, query: &Query) -> Option<Rewrite>;

    /// What keeping this costs.
    fn cost(&self, prices: &PriceTable) -> Cost;

    /// Bring this up to date across `diff`.
    fn refresh(&mut self, diff: &Diff) -> Refreshed;
}

/// Where a piece of derived state came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    /// The table it was computed from.
    pub table: TableId,
    /// The snapshot it was computed from.
    pub snapshot: SnapshotId,
}

/// Bytes computed from a table at a known snapshot, which can answer some
/// queries more cheaply than the table can.
///
/// Always disposable. Losing all derived state costs performance and never
/// correctness, which is what makes every optimization in the engine safe to
/// attempt.
#[derive(Debug)]
pub struct Derived {
    /// Its identity.
    pub id: DerivedId,
    /// What it was computed from.
    pub source: Source,
    /// The policy it was computed under.
    pub policy: PolicyFingerprint,
    /// How many bytes it occupies.
    pub bytes: u64,
    kind: Box<dyn Kind>,
    uses: u64,
}

impl Derived {
    /// Register a piece of derived state.
    pub fn new(
        id: DerivedId,
        source: Source,
        policy: PolicyFingerprint,
        bytes: u64,
        kind: Box<dyn Kind>,
    ) -> Self {
        Derived {
            id,
            source,
            policy,
            bytes,
            kind,
            uses: 0,
        }
    }

    /// Its kind.
    pub fn kind(&self) -> &dyn Kind {
        self.kind.as_ref()
    }

    /// How many times it has served a query.
    pub fn uses(&self) -> u64 {
        self.uses
    }

    /// Note that it served a query. Drives retirement.
    pub fn record_use(&mut self) {
        self.uses = self.uses.saturating_add(1);
    }

    /// What keeping it costs.
    pub fn cost(&self, prices: &PriceTable) -> Cost {
        self.kind.cost(prices)
    }

    /// Bring it up to date across `diff`.
    pub fn refresh(&mut self, diff: &Diff) -> Refreshed {
        self.kind.refresh(diff)
    }

    /// **The one rule.** Whether this may serve `query`, and how.
    ///
    /// The four conditions, in order, so that a rejection names the first
    /// thing that failed:
    ///
    /// 1. `MATCH` — the kind can answer the query's shape
    /// 2. `LINEAGE` — the queried snapshot descends from the built-from one
    /// 3. `POLICY` — the principal's effective policy matches
    /// 4. `RESIDUAL` — what changed since it was built is tolerable
    ///
    /// Condition 4 differs by rewrite, and this is the design's central
    /// asymmetry:
    ///
    /// - [`Rewrite::Prune`] tolerates *any* change. Files added since must be
    ///   scanned too, or rows would be lost; files removed or re-deleted are
    ///   harmless, because the engine reads only live files and applies
    ///   deletes as it goes. Over-selection is conservative.
    /// - [`Rewrite::Substitute`] tolerates only *additive* change. Once rows
    ///   have been removed, the derived state holds rows that are no longer
    ///   live and no amount of extra reading removes them.
    pub fn may_serve(&self, query: &Query, graph: &SnapshotGraph) -> Decision {
        if self.source.table != query.table {
            return Decision::Reject(Reason::WrongTable);
        }

        let Some(rewrite) = self.kind.matches(query) else {
            return Decision::Reject(Reason::NoMatch);
        };

        if graph.get(self.source.snapshot).is_none() || graph.get(query.snapshot).is_none() {
            return Decision::Reject(Reason::UnknownSnapshot);
        }
        if !graph.is_descendant_or_self(query.snapshot, self.source.snapshot) {
            return Decision::Reject(Reason::NotDescendant);
        }

        if self.policy != query.policy {
            return Decision::Reject(Reason::PolicyMismatch);
        }

        let Some(diff) = graph.diff(self.source.snapshot, query.snapshot) else {
            return Decision::Reject(Reason::UnknownSnapshot);
        };

        if rewrite.is_substituting() && !diff.is_purely_additive() {
            return Decision::Reject(Reason::SubtractiveChange);
        }

        if diff.added.is_empty() {
            Decision::Use(rewrite)
        } else {
            Decision::UseWith {
                rewrite,
                also_scan: diff.added,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{DeleteState, Snapshot};

    const POLICY: PolicyFingerprint = PolicyFingerprint(1);
    const OTHER_POLICY: PolicyFingerprint = PolicyFingerprint(2);

    fn t() -> TableId {
        TableId("events".into())
    }

    fn f(name: &str) -> FileId {
        FileId(name.to_owned())
    }

    fn s(id: i64) -> SnapshotId {
        SnapshotId(id)
    }

    /// Prunes to a fixed file set when the query filters on field 4.
    #[derive(Debug)]
    struct FakeIndex {
        files: BTreeSet<FileId>,
    }

    impl Kind for FakeIndex {
        fn name(&self) -> &'static str {
            "index"
        }
        fn matches(&self, query: &Query) -> Option<Rewrite> {
            query.filtered.contains(&4).then(|| Rewrite::Prune {
                files: self.files.clone(),
            })
        }
        fn cost(&self, _prices: &PriceTable) -> Cost {
            Cost::ZERO
        }
        fn refresh(&mut self, _diff: &Diff) -> Refreshed {
            Refreshed::UpToDate
        }
    }

    /// Substitutes for one exact plan.
    #[derive(Debug)]
    struct FakeResult {
        plan_hash: u64,
    }

    impl Kind for FakeResult {
        fn name(&self) -> &'static str {
            "result"
        }
        fn matches(&self, query: &Query) -> Option<Rewrite> {
            (query.plan_hash == self.plan_hash).then_some(Rewrite::Substitute)
        }
        fn cost(&self, _prices: &PriceTable) -> Cost {
            Cost::ZERO
        }
        fn refresh(&mut self, _diff: &Diff) -> Refreshed {
            Refreshed::NeedsRebuild
        }
    }

    fn index_at(snapshot: SnapshotId, files: &[&str]) -> Derived {
        Derived::new(
            DerivedId("idx".into()),
            Source {
                table: t(),
                snapshot,
            },
            POLICY,
            1024,
            Box::new(FakeIndex {
                files: files.iter().map(|n| f(n)).collect(),
            }),
        )
    }

    fn result_at(snapshot: SnapshotId, plan_hash: u64) -> Derived {
        Derived::new(
            DerivedId("res".into()),
            Source {
                table: t(),
                snapshot,
            },
            POLICY,
            64,
            Box::new(FakeResult { plan_hash }),
        )
    }

    fn query_at(snapshot: SnapshotId) -> Query {
        Query {
            table: t(),
            snapshot,
            policy: POLICY,
            plan_hash: 99,
            projected: BTreeSet::from([4, 7]),
            filtered: BTreeSet::from([4]),
        }
    }

    /// 810 -> 811 (appends "b") -> 812 (appends "c")
    fn appends() -> SnapshotGraph {
        SnapshotGraph::new()
            .with(Snapshot::root(s(810)).with_clean_file(f("a")))
            .with(
                Snapshot::child_of(s(811), s(810))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b")),
            )
            .with(
                Snapshot::child_of(s(812), s(811))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b"))
                    .with_clean_file(f("c")),
            )
    }

    #[test]
    fn current_derived_state_is_used_as_is() {
        let g = appends();
        let d = index_at(s(812), &["a"]);
        assert_eq!(
            d.may_serve(&query_at(s(812)), &g),
            Decision::Use(Rewrite::Prune {
                files: BTreeSet::from([f("a")])
            })
        );
    }

    #[test]
    fn a_stale_index_is_used_with_the_added_files() {
        // Pruning tolerates staleness, but the files added since must be
        // scanned too or their rows would be lost.
        let g = appends();
        let d = index_at(s(810), &["a"]);
        assert_eq!(
            d.may_serve(&query_at(s(812)), &g),
            Decision::UseWith {
                rewrite: Rewrite::Prune {
                    files: BTreeSet::from([f("a")])
                },
                also_scan: BTreeSet::from([f("b"), f("c")]),
            }
        );
    }

    #[test]
    fn a_stale_result_is_used_with_the_added_files() {
        let g = appends();
        let d = result_at(s(811), 99);
        let decision = d.may_serve(&query_at(s(812)), &g);
        assert_eq!(
            decision,
            Decision::UseWith {
                rewrite: Rewrite::Substitute,
                also_scan: BTreeSet::from([f("c")]),
            }
        );
    }

    #[test]
    fn a_delete_disqualifies_a_substituting_kind_but_not_a_pruning_one() {
        // 811 deletes rows from "a" without adding or removing any file.
        let g = SnapshotGraph::new()
            .with(Snapshot::root(s(810)).with_clean_file(f("a")))
            .with(Snapshot::child_of(s(811), s(810)).with_file(f("a"), DeleteState(7)));

        let result = result_at(s(810), 99);
        assert_eq!(
            result.may_serve(&query_at(s(811)), &g),
            Decision::Reject(Reason::SubtractiveChange),
            "a cached result would return rows that are no longer live"
        );

        let index = index_at(s(810), &["a"]);
        assert_eq!(
            index.may_serve(&query_at(s(811)), &g),
            Decision::Use(Rewrite::Prune {
                files: BTreeSet::from([f("a")])
            }),
            "pruning is safe: the engine still reads the file and applies deletes"
        );
    }

    #[test]
    fn compaction_disqualifies_a_substituting_kind() {
        let g = SnapshotGraph::new()
            .with(
                Snapshot::root(s(810))
                    .with_clean_file(f("small-1"))
                    .with_clean_file(f("small-2")),
            )
            .with(Snapshot::child_of(s(811), s(810)).with_clean_file(f("merged")));

        assert_eq!(
            result_at(s(810), 99).may_serve(&query_at(s(811)), &g),
            Decision::Reject(Reason::SubtractiveChange)
        );
    }

    #[test]
    fn derived_state_from_an_abandoned_branch_is_refused() {
        let g = SnapshotGraph::new()
            .with(Snapshot::root(s(810)).with_clean_file(f("a")))
            .with(Snapshot::child_of(s(811), s(810)).with_clean_file(f("a")))
            .with(Snapshot::child_of(s(812), s(810)).with_clean_file(f("a")));

        assert_eq!(
            index_at(s(811), &["a"]).may_serve(&query_at(s(812)), &g),
            Decision::Reject(Reason::NotDescendant)
        );
    }

    #[test]
    fn a_newer_query_snapshot_is_required() {
        let g = appends();
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&query_at(s(810)), &g),
            Decision::Reject(Reason::NotDescendant)
        );
    }

    #[test]
    fn a_policy_mismatch_is_refused() {
        let g = appends();
        let mut q = query_at(s(812));
        q.policy = OTHER_POLICY;
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&q, &g),
            Decision::Reject(Reason::PolicyMismatch)
        );
    }

    #[test]
    fn a_different_table_is_refused() {
        let g = appends();
        let mut q = query_at(s(812));
        q.table = TableId("other".into());
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&q, &g),
            Decision::Reject(Reason::WrongTable)
        );
    }

    #[test]
    fn a_kind_that_cannot_answer_is_refused() {
        let g = appends();
        let mut q = query_at(s(812));
        q.filtered.clear(); // the fake index needs a predicate on field 4
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&q, &g),
            Decision::Reject(Reason::NoMatch)
        );

        let mut q = query_at(s(812));
        q.plan_hash = 12345;
        assert_eq!(
            result_at(s(812), 99).may_serve(&q, &g),
            Decision::Reject(Reason::NoMatch)
        );
    }

    #[test]
    fn an_unknown_snapshot_is_refused() {
        let g = appends();
        assert_eq!(
            index_at(s(777), &["a"]).may_serve(&query_at(s(812)), &g),
            Decision::Reject(Reason::UnknownSnapshot)
        );
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&query_at(s(777)), &g),
            Decision::Reject(Reason::UnknownSnapshot)
        );
    }

    #[test]
    fn use_counting_saturates() {
        let mut d = index_at(s(812), &["a"]);
        assert_eq!(d.uses(), 0);
        d.record_use();
        assert_eq!(d.uses(), 1);
        d.uses = u64::MAX;
        d.record_use();
        assert_eq!(d.uses(), u64::MAX);
    }
}
