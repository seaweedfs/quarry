//! What queries asked for, what would have helped, and what stopped helping.
//!
//! This is the part that makes the system *self*-optimizing rather than merely
//! optimizable. Everything else uses derived state that someone decided to
//! build; this decides.
//!
//! # The counterfactual problem, and where it does not apply
//!
//! Measuring an optimizer's benefit is usually circular: you cannot measure
//! what a candidate would have saved without building it, and once you build
//! it the un-optimized baseline is gone. The usual answers are holdout
//! sampling or shadow execution, both of which cost something.
//!
//! For *pruning*, none of that is needed, because the baseline is **computable
//! from metadata**:
//!
//! ```text
//! a full scan's cost = the sum of the live data files' sizes
//! ```
//!
//! That is known exactly, at every snapshot, without reading anything. So the
//! saving from having pruned is `full_scan_bytes - bytes_read`, measured rather
//! than estimated, for every query that ran.
//!
//! Where it genuinely does not apply, and this design says so rather than
//! pretending:
//!
//! ```text
//! latency          not modelled at all; bytes are a proxy, and a poor one
//!                  for a query whose cost is cpu rather than I/O
//!
//! substituting     a cached aggregate's alternative is not "read N bytes"
//! kinds            but "read N bytes and then compute", and the compute is
//!                  not priced here
//!
//! what was never   a proposal's saving cannot be measured, only bounded.
//! built            See `Proposal::ceiling_usd`, which is a ceiling and is
//!                  documented as one.
//! ```

use std::collections::{BTreeMap, BTreeSet};

use crate::cost::PriceTable;
use crate::derived::{DerivedId, FieldId, Predicate, Query};
use crate::layout::Spread;
use crate::registry::Registry;
use crate::snapshot::{SnapshotId, TableId};

/// The shape of a query, with literals stripped.
///
/// Two queries differing only in which tenant they ask about share a
/// fingerprint, which is the whole point: one index serves both. Literals are
/// dropped rather than hashed, so nothing here can leak a value.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fingerprint {
    /// The table read.
    pub table: TableId,
    /// Fields an index could probe, because they were compared for equality.
    pub probeable: BTreeSet<FieldId>,
    /// Fields restricted in a way nothing can currently probe.
    pub opaque: BTreeSet<FieldId>,
    /// Fields read.
    pub projected: BTreeSet<FieldId>,
}

impl Fingerprint {
    /// The shape of `query`.
    pub fn of(query: &Query) -> Self {
        let mut probeable = BTreeSet::new();
        let mut opaque = BTreeSet::new();
        for predicate in &query.predicates {
            match predicate {
                Predicate::Eq { field, .. } => probeable.insert(*field),
                Predicate::Opaque { field } => opaque.insert(*field),
            };
        }
        Fingerprint {
            table: query.table.clone(),
            probeable,
            opaque,
            projected: query.projected.clone(),
        }
    }
}

/// What one query actually cost, and what it would have cost unaided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    /// The query's shape.
    pub fingerprint: Fingerprint,
    /// Bytes the query read.
    pub bytes_read: u64,
    /// Bytes a full scan of the queried snapshot would have read.
    ///
    /// Known exactly from the live file sizes, which is what makes the saving
    /// below a measurement rather than a guess.
    pub bytes_if_full_scan: u64,
    /// Which piece of derived state served it, if any.
    pub used: Option<DerivedId>,
}

impl Observation {
    /// Bytes not read, thanks to whatever served this query.
    ///
    /// Saturating: a query that somehow read more than a full scan is recorded
    /// as having saved nothing rather than as a negative saving.
    pub fn bytes_saved(&self) -> u64 {
        self.bytes_if_full_scan.saturating_sub(self.bytes_read)
    }
}

/// A scan another engine ran, as it can describe it.
///
/// The optimizer only sees queries that came through this engine, which in a
/// real lakehouse is a minority of them: Spark and Trino read the same tables
/// and their traffic is invisible here. An index built for what *we* happen to
/// serve optimizes for a sample, not for the workload.
///
/// # Why this is not Iceberg's `ScanReport`
///
/// `iceberg-rust` 0.6 has no metrics reporting at all — no `ScanReport`, no
/// `MetricsReporter` — so there is no type to adapt from. Coupling to an absent
/// API would be worse than this in any case: the point is to accept telemetry
/// from *any* engine. The shape below deliberately mirrors the fields Iceberg's
/// REST `report-metrics` payload carries, so a handler for it can populate this
/// directly.
///
/// # What a foreign engine does not have to share
///
/// Notably, not literal values. A [`Fingerprint`] keeps only *which* fields
/// were restricted and whether an equality could probe them — literals are
/// dropped, so nothing here depends on another engine hashing values the way
/// this one does, and no value crosses the boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignScan {
    /// The table read.
    pub table: TableId,
    /// The snapshot read.
    pub snapshot: SnapshotId,
    /// Fields the query read.
    pub projected: BTreeSet<FieldId>,
    /// Fields compared for equality, which an index could probe.
    pub equalities: BTreeSet<FieldId>,
    /// Fields restricted some other way.
    pub restrictions: BTreeSet<FieldId>,
    /// Bytes the scan read.
    pub bytes_read: u64,
}

impl ForeignScan {
    /// What to record about this scan.
    ///
    /// `bytes_if_full_scan` is the caller's to supply from the table's own file
    /// sizes rather than the reporter's to claim — it is the baseline every
    /// saving is measured against, and a foreign engine has no reason to be
    /// trusted with it.
    ///
    /// `used` is always `None`: another engine cannot have been served by
    /// derived state it does not know about, so these scans inform *what to
    /// build* without ever crediting anything.
    pub fn observation(&self, bytes_if_full_scan: u64) -> Observation {
        Observation {
            fingerprint: Fingerprint {
                table: self.table.clone(),
                probeable: self.equalities.clone(),
                opaque: self.restrictions.clone(),
                projected: self.projected.clone(),
            },
            bytes_read: self.bytes_read,
            bytes_if_full_scan,
            used: None,
        }
    }
}

/// What has been seen for one query shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Seen {
    /// How many queries of this shape ran.
    pub queries: u64,
    /// Bytes they read in total.
    pub bytes_read: u64,
    /// Bytes full scans would have read.
    pub bytes_if_full_scan: u64,
    /// How many were served by derived state.
    pub helped: u64,
}

impl Seen {
    /// Bytes not read across every query of this shape.
    pub fn bytes_saved(&self) -> u64 {
        self.bytes_if_full_scan.saturating_sub(self.bytes_read)
    }
}

/// Something worth building.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    /// The table it would be built on.
    pub table: TableId,
    /// The field it would index.
    pub field: FieldId,
    /// How many queries of this shape went unaided.
    pub queries: u64,
    /// Bytes those queries read.
    pub bytes_scanned: u64,
    /// The most this could possibly have saved.
    ///
    /// A **ceiling**, not an estimate: it assumes the index prunes everything.
    /// Measured against real data it was 1775x optimistic, unboundedly
    /// optimistic, and exactly right, across three regimes — so it does *not*
    /// order proposals usefully, which an earlier version of this comment
    /// claimed it did.
    ///
    /// Use [`Proposal::expected_usd`] to rank. This remains only as an upper
    /// bound, which is occasionally worth knowing and never worth deciding on.
    pub ceiling_usd: f64,
}

impl Proposal {
    /// What this is expected to save, given how the field is laid out.
    ///
    /// The ceiling scaled by the fraction of a scan an index actually removes
    /// that the file format does not. Against measured data this lands within
    /// a few points of what was realized, where the ceiling was out by three
    /// orders of magnitude.
    pub fn expected_usd(&self, spread: &Spread, prices: &PriceTable) -> f64 {
        self.bytes_scanned as f64
            * spread.index_advantage()
            * prices.byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far)
    }
}

/// What the optimizer is allowed to do.
///
/// The design's user surface is one table property and one number:
///
/// ```sql
/// ALTER TABLE events SET (auto_optimize = true, optimize_budget_pct = 5);
/// ```
///
/// so this is deliberately small. Everything here is a ceiling or a threshold;
/// none of it is a hint the optimizer may ignore.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    /// Whether the optimizer may act at all.
    ///
    /// When false a round still *reports* what it would do, so advisory and
    /// automatic modes share one code path and one set of decisions.
    pub auto_optimize: bool,
    /// Most bytes of derived state to keep for this table.
    pub budget_bytes: u64,
    /// Queries of a shape before it justifies building anything.
    pub min_queries: u64,
    /// Days over which retention is priced when deciding what has paid off.
    pub horizon_days: f64,
    /// Most builds in one round, so a cold start cannot build everything at
    /// once.
    pub max_builds_per_round: usize,
    /// How much of a table derived state may fall behind before it is rebuilt,
    /// as a percentage.
    ///
    /// A pruning index stays *correct* as a table grows — the rule reads the
    /// files added since alongside it — but it helps less and less, because
    /// that residual is scanned every query. Left alone it decays toward
    /// useless while still being credited with the savings it once earned, so
    /// retirement will not catch it.
    ///
    /// Measured against bytes rather than commits: ten tiny appends matter
    /// less than one large one, and the file sizes are known exactly.
    pub max_residual_pct: f64,
    /// Queries that must be observed before anything may be retired.
    ///
    /// A guard against judging derived state on no evidence. Retirement asks
    /// what a piece has *measurably* saved, and treats nothing as a reason to
    /// delete — which is right once queries have run and wrong immediately
    /// after a restart, when a recovered index has served nothing yet.
    ///
    /// Persisted credits mostly close that window, but not entirely: two
    /// processes sharing storage overwrite each other's telemetry, so a
    /// recovered piece can legitimately look unused. Deleting an index is
    /// cheap to get wrong in one direction only — rebuilding costs a scan,
    /// while keeping a useless index for another round costs almost nothing.
    pub retire_after_queries: u64,
    /// How much better than the file format's own pruning an index must be,
    /// as a percentage of a full scan, before it is worth building.
    ///
    /// Without this the loop builds an index whenever a shape goes unaided,
    /// which measurement showed to be badly wrong: of three regimes, one saved
    /// 95% of a scan and two saved nothing, and nothing in the proposal could
    /// tell them apart. See [`Spread`].
    ///
    /// Zero would restore the old behaviour of building on hope alone.
    pub min_index_advantage_pct: f64,
}

impl Policy {
    /// Observe and report, but change nothing.
    pub const ADVISORY: Policy = Policy {
        auto_optimize: false,
        budget_bytes: 0,
        min_queries: 10,
        horizon_days: 30.0,
        max_builds_per_round: 1,
        max_residual_pct: 25.0,
        retire_after_queries: 100,
        min_index_advantage_pct: 10.0,
    };

    /// Act, keeping derived state under `budget_bytes`.
    pub fn automatic(budget_bytes: u64) -> Self {
        Policy {
            auto_optimize: true,
            budget_bytes,
            ..Policy::ADVISORY
        }
    }

    /// Act, keeping derived state under `percent` of the table's own size.
    ///
    /// The design's unit. A percentage is the one a user can set without
    /// knowing how big an index turns out to be.
    pub fn automatic_pct(table_bytes: u64, percent: f64) -> Self {
        Policy::automatic((table_bytes as f64 * percent / 100.0) as u64)
    }

    /// Require `min_queries` of a shape before building for it.
    pub fn with_min_queries(mut self, min_queries: u64) -> Self {
        self.min_queries = min_queries;
        self
    }

    /// Build at most `max` pieces per round.
    pub fn with_max_builds(mut self, max: usize) -> Self {
        self.max_builds_per_round = max;
        self
    }
}

/// What queries have asked for, accumulated by shape.
#[derive(Clone, Debug, Default)]
pub struct Workload {
    by_shape: BTreeMap<Fingerprint, Seen>,
    by_derived: BTreeMap<DerivedId, Seen>,
}

impl Workload {
    /// An empty workload.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record what a query cost.
    pub fn observe(&mut self, observation: Observation) {
        let shape = self.by_shape.entry(observation.fingerprint).or_default();
        shape.queries += 1;
        shape.bytes_read = shape.bytes_read.saturating_add(observation.bytes_read);
        shape.bytes_if_full_scan = shape
            .bytes_if_full_scan
            .saturating_add(observation.bytes_if_full_scan);

        if let Some(id) = observation.used {
            shape.helped += 1;
            let credited = self.by_derived.entry(id).or_default();
            credited.queries += 1;
            credited.helped += 1;
            credited.bytes_read = credited.bytes_read.saturating_add(observation.bytes_read);
            credited.bytes_if_full_scan = credited
                .bytes_if_full_scan
                .saturating_add(observation.bytes_if_full_scan);
        }
    }

    /// How many query shapes have been seen.
    pub fn shapes(&self) -> usize {
        self.by_shape.len()
    }

    /// What was seen for one shape.
    pub fn seen(&self, fingerprint: &Fingerprint) -> Option<Seen> {
        self.by_shape.get(fingerprint).copied()
    }

    /// What one piece of derived state has actually saved.
    pub fn credited(&self, id: &DerivedId) -> Option<Seen> {
        self.by_derived.get(id).copied()
    }

    /// Every shape seen, in a fixed order.
    ///
    /// Ordered because this is written to storage: the bytes should be a
    /// function of the contents alone.
    pub fn shapes_seen(&self) -> impl Iterator<Item = (&Fingerprint, &Seen)> {
        self.by_shape.iter()
    }

    /// What each piece of derived state has saved, in a fixed order.
    pub fn credits(&self) -> impl Iterator<Item = (&DerivedId, &Seen)> {
        self.by_derived.iter()
    }

    /// Rebuild a workload that was written down.
    ///
    /// Credits matter far more than shapes here. Shapes only affect how
    /// quickly the optimizer re-learns what to propose; credits are what
    /// [`Workload::retirements`] judges against, and without them every
    /// recovered piece looks like it has saved nothing and is deleted.
    pub fn restore(
        shapes: impl IntoIterator<Item = (Fingerprint, Seen)>,
        credits: impl IntoIterator<Item = (DerivedId, Seen)>,
    ) -> Self {
        Workload {
            by_shape: shapes.into_iter().collect(),
            by_derived: credits.into_iter().collect(),
        }
    }

    /// How many queries have been observed in total.
    ///
    /// Used to hold retirement off until enough has been seen to judge by —
    /// see [`Policy::retire_after_queries`].
    pub fn observed(&self) -> u64 {
        self.by_shape.values().map(|seen| seen.queries).sum()
    }

    /// Indexes worth building, most promising first.
    ///
    /// Only proposes for queries that went **unaided**: a shape already served
    /// by derived state does not need more. Requires at least `min_queries`
    /// of a shape, so that one expensive one-off does not cause a build.
    ///
    /// One proposal per (table, field), with counts summed across every shape
    /// that would benefit — otherwise ten shapes filtering the same field
    /// would each propose the same index.
    pub fn proposals(&self, prices: &PriceTable, min_queries: u64) -> Vec<Proposal> {
        let mut merged: BTreeMap<(TableId, FieldId), Proposal> = BTreeMap::new();

        for (shape, seen) in &self.by_shape {
            let unaided = seen.queries.saturating_sub(seen.helped);
            if unaided == 0 {
                continue;
            }
            for field in &shape.probeable {
                let key = (shape.table.clone(), *field);
                let entry = merged.entry(key).or_insert_with(|| Proposal {
                    table: shape.table.clone(),
                    field: *field,
                    queries: 0,
                    bytes_scanned: 0,
                    ceiling_usd: 0.0,
                });
                entry.queries += unaided;
                entry.bytes_scanned = entry.bytes_scanned.saturating_add(seen.bytes_read);
            }
        }

        let mut proposals: Vec<Proposal> = merged
            .into_values()
            .filter(|proposal| proposal.queries >= min_queries)
            .map(|mut proposal| {
                proposal.ceiling_usd = proposal.bytes_scanned as f64
                    * prices.byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far);
                proposal
            })
            .collect();

        // Most promising first; ties broken by field so the order is total.
        proposals.sort_by(|a, b| {
            b.ceiling_usd
                .total_cmp(&a.ceiling_usd)
                .then_with(|| a.field.cmp(&b.field))
        });
        proposals
    }

    /// Derived state that has not paid for itself over `horizon_days`.
    ///
    /// Compares what a piece has *measurably* saved against what keeping it
    /// costs. A piece nothing has used yet is proposed for retirement, which is
    /// deliberate: something built and never touched is indistinguishable from
    /// a leak, and rebuilding it later is cheap because derived state is
    /// disposable.
    ///
    /// Returns ids only; the caller decides whether to act, so that an
    /// advisory mode and an automatic one share this code.
    pub fn retirements(
        &self,
        registry: &Registry,
        prices: &PriceTable,
        horizon_days: f64,
    ) -> Vec<DerivedId> {
        let mut retire = Vec::new();
        for id in registry.ids() {
            let Some(derived) = registry.get(&id) else {
                continue;
            };
            let keeping = prices.retention_usd(derived.bytes, horizon_days);
            let saved = self
                .credited(&id)
                .map(|seen| {
                    seen.bytes_saved() as f64
                        * prices.byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far)
                })
                .unwrap_or(0.0);
            if saved < keeping {
                retire.push(id);
            }
        }
        retire
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::Cost;
    use crate::derived::{Derived, Kind, PolicyFingerprint, Refreshed, Rewrite, Source};
    use crate::snapshot::{Diff, SnapshotId};

    const GB: u64 = 1_000_000_000;

    fn table() -> TableId {
        TableId("events".into())
    }

    fn query_on(fields: &[FieldId]) -> Query {
        Query {
            table: table(),
            snapshot: SnapshotId(1),
            policy: PolicyFingerprint(1),
            plan_hash: 1,
            plan: None,
            projected: BTreeSet::from([7]),
            predicates: fields
                .iter()
                .map(|field| Predicate::Eq {
                    field: *field,
                    value: 42,
                })
                .collect(),
        }
    }

    fn unaided(fields: &[FieldId], bytes: u64) -> Observation {
        Observation {
            fingerprint: Fingerprint::of(&query_on(fields)),
            bytes_read: bytes,
            bytes_if_full_scan: bytes,
            used: None,
        }
    }

    fn foreign(fields: &[FieldId], bytes: u64) -> ForeignScan {
        ForeignScan {
            table: table(),
            snapshot: SnapshotId(1),
            projected: BTreeSet::from([7]),
            equalities: fields.iter().copied().collect(),
            restrictions: BTreeSet::new(),
            bytes_read: bytes,
        }
    }

    /// The property the whole thing rests on.
    ///
    /// If a scan reported by Spark does not land in the same bucket as the same
    /// query run here, cross-engine telemetry aggregates nothing and the
    /// optimizer still only sees its own traffic.
    #[test]
    fn a_foreign_scan_shares_a_shape_with_an_identical_local_query() {
        let local = unaided(&[4], GB);
        let remote = foreign(&[4], GB).observation(GB);
        assert_eq!(
            local.fingerprint, remote.fingerprint,
            "the same question asked by two engines is one shape"
        );

        let mut workload = Workload::new();
        workload.observe(local);
        workload.observe(remote);
        assert_eq!(workload.shapes(), 1, "they must aggregate, not sit apart");
        assert_eq!(workload.observed(), 2);
    }

    #[test]
    fn foreign_traffic_alone_can_justify_an_index() {
        // A table this engine barely serves, and another hammers.
        let mut workload = Workload::new();
        for _ in 0..10 {
            workload.observe(foreign(&[4], GB).observation(GB));
        }

        let proposals = workload.proposals(&PriceTable::default(), 5);
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].field, 4);
        assert_eq!(proposals[0].queries, 10);
    }

    #[test]
    fn a_foreign_scan_credits_nothing() {
        // Another engine cannot have used derived state it does not know
        // about, so these observations must never make an index look useful.
        let mut workload = Workload::new();
        workload.observe(foreign(&[4], GB / 10).observation(GB));
        assert!(workload.credits().next().is_none());
    }

    #[test]
    fn the_baseline_is_the_callers_to_supply_not_the_reporters() {
        // A reporter that under-states what it read cannot inflate a saving,
        // because the baseline comes from the table's own file sizes.
        let scan = foreign(&[4], GB / 4);
        let observation = scan.observation(GB);
        assert_eq!(observation.bytes_if_full_scan, GB);
        assert_eq!(observation.bytes_saved(), GB - GB / 4);
    }

    #[test]
    fn a_budget_percentage_is_of_the_table() {
        let policy = Policy::automatic_pct(100 * GB, 5.0);
        assert!(policy.auto_optimize);
        assert_eq!(policy.budget_bytes, 5 * GB);
    }

    // `Policy::ADVISORY` is a const, so asserting on its fields asserts on
    // compile-time constants and proves nothing. What matters is that an
    // advisory optimizer declines to act, which tests/loop_closes.rs checks
    // against a real table.

    #[test]
    fn a_fingerprint_drops_literals_but_keeps_shape() {
        let mut one = query_on(&[4]);
        one.predicates = vec![Predicate::Eq { field: 4, value: 1 }];
        let mut two = query_on(&[4]);
        two.predicates = vec![Predicate::Eq {
            field: 4,
            value: 999,
        }];

        assert_eq!(
            Fingerprint::of(&one),
            Fingerprint::of(&two),
            "two tenants asking the same question share one index"
        );
    }

    #[test]
    fn probeable_and_opaque_predicates_are_distinguished() {
        let mut query = query_on(&[]);
        query.predicates = vec![
            Predicate::Eq { field: 4, value: 1 },
            Predicate::Opaque { field: 9 },
        ];
        let fingerprint = Fingerprint::of(&query);
        assert_eq!(fingerprint.probeable, BTreeSet::from([4]));
        assert_eq!(fingerprint.opaque, BTreeSet::from([9]));
    }

    #[test]
    fn an_unaided_shape_produces_a_proposal() {
        let mut workload = Workload::new();
        for _ in 0..10 {
            workload.observe(unaided(&[4], GB));
        }

        let proposals = workload.proposals(&PriceTable::default(), 5);
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].field, 4);
        assert_eq!(proposals[0].queries, 10);
        assert_eq!(proposals[0].bytes_scanned, 10 * GB);
        assert!(proposals[0].ceiling_usd > 0.0);
    }

    #[test]
    fn a_rare_shape_does_not_justify_a_build() {
        let mut workload = Workload::new();
        workload.observe(unaided(&[4], 100 * GB));
        assert!(
            workload.proposals(&PriceTable::default(), 5).is_empty(),
            "one expensive query is not a workload"
        );
    }

    #[test]
    fn an_already_helped_shape_produces_no_proposal() {
        let mut workload = Workload::new();
        for _ in 0..10 {
            workload.observe(Observation {
                fingerprint: Fingerprint::of(&query_on(&[4])),
                bytes_read: GB / 10,
                bytes_if_full_scan: GB,
                used: Some(DerivedId("idx".into())),
            });
        }
        assert!(workload.proposals(&PriceTable::default(), 1).is_empty());
    }

    #[test]
    fn one_index_is_proposed_per_field_not_per_shape() {
        // Two shapes filtering the same field differ only in projection.
        let mut workload = Workload::new();
        for _ in 0..5 {
            workload.observe(unaided(&[4], GB));
        }
        let mut other = query_on(&[4]);
        other.projected = BTreeSet::from([7, 11]);
        for _ in 0..5 {
            workload.observe(Observation {
                fingerprint: Fingerprint::of(&other),
                bytes_read: GB,
                bytes_if_full_scan: GB,
                used: None,
            });
        }
        assert_eq!(workload.shapes(), 2);

        let proposals = workload.proposals(&PriceTable::default(), 1);
        assert_eq!(proposals.len(), 1, "one field, one index");
        assert_eq!(proposals[0].queries, 10, "counts sum across shapes");
    }

    #[test]
    fn proposals_are_ordered_by_what_is_at_stake() {
        let mut workload = Workload::new();
        for _ in 0..3 {
            workload.observe(unaided(&[4], GB));
            workload.observe(unaided(&[9], 100 * GB));
        }
        let proposals = workload.proposals(&PriceTable::default(), 1);
        assert_eq!(proposals[0].field, 9, "the expensive one first");
        assert_eq!(proposals[1].field, 4);
    }

    #[test]
    fn saving_is_measured_not_estimated() {
        let observation = Observation {
            fingerprint: Fingerprint::of(&query_on(&[4])),
            bytes_read: GB / 4,
            bytes_if_full_scan: GB,
            used: Some(DerivedId("idx".into())),
        };
        assert_eq!(observation.bytes_saved(), GB - GB / 4);
    }

    #[test]
    fn reading_more_than_a_full_scan_saves_nothing_rather_than_less() {
        let observation = Observation {
            fingerprint: Fingerprint::of(&query_on(&[4])),
            bytes_read: 2 * GB,
            bytes_if_full_scan: GB,
            used: Some(DerivedId("idx".into())),
        };
        assert_eq!(observation.bytes_saved(), 0);
    }

    // Retirement needs a registry, so a minimal kind to populate it with.

    #[derive(Debug)]
    struct Nothing;

    impl Kind for Nothing {
        fn name(&self) -> &'static str {
            "nothing"
        }
        fn matches(&self, _query: &Query) -> Option<Rewrite> {
            None
        }
        fn cost(&self, _prices: &PriceTable) -> Cost {
            Cost::ZERO
        }
        fn refresh(&mut self, _diff: &Diff) -> Refreshed {
            Refreshed::UpToDate
        }
    }

    fn registry_with(id: &str, bytes: u64) -> Registry {
        let mut registry = Registry::new();
        registry.register(Derived::new(
            DerivedId(id.into()),
            Source {
                table: table(),
                snapshot: SnapshotId(1),
            },
            PolicyFingerprint(1),
            bytes,
            Box::new(Nothing),
        ));
        registry
    }

    #[test]
    fn derived_state_nothing_has_used_is_retired() {
        let registry = registry_with("idle", 10 * GB);
        let retire = Workload::new().retirements(&registry, &PriceTable::default(), 30.0);
        assert_eq!(
            retire,
            vec![DerivedId("idle".into())],
            "built and never touched is indistinguishable from a leak"
        );
    }

    #[test]
    fn derived_state_that_has_paid_for_itself_is_kept() {
        let registry = registry_with("useful", 1_000_000);
        let mut workload = Workload::new();
        for _ in 0..1_000 {
            workload.observe(Observation {
                fingerprint: Fingerprint::of(&query_on(&[4])),
                bytes_read: 0,
                bytes_if_full_scan: GB,
                used: Some(DerivedId("useful".into())),
            });
        }
        assert!(
            workload
                .retirements(&registry, &PriceTable::default(), 30.0)
                .is_empty(),
            "a terabyte of avoided reads pays for a megabyte of storage"
        );
    }

    #[test]
    fn a_huge_piece_saving_little_is_retired() {
        let registry = registry_with("bloated", 500 * GB);
        let mut workload = Workload::new();
        workload.observe(Observation {
            fingerprint: Fingerprint::of(&query_on(&[4])),
            bytes_read: 0,
            bytes_if_full_scan: 1_000,
            used: Some(DerivedId("bloated".into())),
        });
        assert_eq!(
            workload.retirements(&registry, &PriceTable::default(), 30.0),
            vec![DerivedId("bloated".into())]
        );
    }

    #[test]
    fn credit_goes_to_the_piece_that_served_the_query() {
        let mut workload = Workload::new();
        workload.observe(Observation {
            fingerprint: Fingerprint::of(&query_on(&[4])),
            bytes_read: 1,
            bytes_if_full_scan: GB,
            used: Some(DerivedId("a".into())),
        });

        let credited = workload.credited(&DerivedId("a".into())).expect("credited");
        assert_eq!(credited.queries, 1);
        assert_eq!(credited.bytes_saved(), GB - 1);
        assert!(workload.credited(&DerivedId("b".into())).is_none());
    }
}
