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
use crate::derived::{Aggregate, DerivedId, FieldId, Filter, Nearest, Plan, Predicate, Query};
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

/// What an aggregate query asked, kept whole enough to propose a cube.
///
/// Unlike a [`Fingerprint`], this keeps the plan — literal filter text
/// included — because a cube proposal must name exactly which filters to bake
/// in. It is observed, grouped, and proposed from, but deliberately **not**
/// persisted: literals are a principal's own data and do not belong on disk
/// beside shape counts.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AggregateAsk {
    /// The table the query aggregated over.
    pub table: TableId,
    /// The exact plan the query ran under.
    pub plan: Plan,
    /// Keys and measures.
    pub spec: Aggregate,
    /// The query's filters rendered back to SQL, so a cube proposal can bake
    /// them in. Canonical text identifies; this executes.
    pub filter_sql: Vec<String>,
}

/// What a top-k nearest-neighbour query asked, kept whole enough to propose
/// a vector index.
///
/// Unlike [`AggregateAsk`] and [`FilterAsk`] this carries **no literals**: a
/// [`Nearest`] holds the field, metric, dimension, and k, never the query
/// vector. One index over `(field, metric, dimension)` serves every vector,
/// so the vector is not part of what is proposed — which also means this is
/// safe to persist, though nothing does yet.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NearestAsk {
    /// The table searched.
    pub table: TableId,
    /// The ask, minus the query vector.
    pub nearest: Nearest,
}

/// A vector index worth building.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorProposal {
    /// The table it would be built on.
    pub table: TableId,
    /// What it would cover.
    pub ask: NearestAsk,
    /// How many unaided queries asked for this.
    pub queries: u64,
    /// Bytes those queries read.
    pub bytes_scanned: u64,
    /// The most it could possibly have saved: every byte those queries moved.
    ///
    /// A flat vector index replaces the whole scan, so this is as close to
    /// honest as a cube's ceiling — and still an upper bound, since the
    /// index's own storage and build are not in it.
    pub ceiling_usd: f64,
}

/// A cube worth building.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeProposal {
    /// The table it would be built on.
    pub table: TableId,
    /// The shape of what it answers: grain, measures, and the filters to bake.
    pub ask: AggregateAsk,
    /// How many unaided queries asked for this.
    pub queries: u64,
    /// Bytes those queries read.
    pub bytes_scanned: u64,
    /// The most it could possibly have saved: every byte those queries moved.
    ///
    /// A cube's own read is a few partial rows beside a full scan, so unlike
    /// an index's ceiling this one is close to honest — and still only an
    /// upper bound, since the cube's build cost is not in it.
    pub ceiling_usd: f64,
}

/// What one query actually cost, and what it would have cost unaided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    /// The query's shape.
    pub fingerprint: Fingerprint,
    /// What the query aggregated, if it did.
    pub aggregate: Option<AggregateAsk>,
    /// What the query searched for, if it was a top-k nearest ask.
    pub nearest: Option<NearestAsk>,
    /// Bytes the query read.
    pub bytes_read: u64,
    /// Bytes a full scan of the queried snapshot would have read.
    ///
    /// Known exactly from the live file sizes, which is what makes the saving
    /// below a measurement rather than a guess.
    pub bytes_if_full_scan: u64,
    /// Which pieces of derived state served it, if any.
    ///
    /// Plural because prunes compose: several indexes may have intersected
    /// their candidate file sets, and each that narrowed the plan is named.
    /// Every named piece is credited with the observation — attribution of
    /// the saving between them is inherently ambiguous.
    pub used: Vec<DerivedId>,
    /// The filters the query ran, kept whole — canonical identity plus the
    /// SQL to rebuild it.
    ///
    /// Not persisted: the text carries literals. See [`FilterAsk`].
    pub filters: Vec<FilterAsk>,
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
            aggregate: None,
            // A foreign engine's sort expression is in its dialect, not
            // this one's distance functions — nothing to key an index by.
            nearest: None,
            bytes_read: self.bytes_read,
            bytes_if_full_scan,
            used: Vec::new(),
            // A foreign engine's filter text is in its dialect, not this
            // one's canonical form — nothing to match a FilterSet by.
            filters: Vec::new(),
        }
    }
}

/// One filter worth remembering: what it is, and how to run it.
///
/// Like [`AggregateAsk`], this keeps literal text — the filter's canonical
/// rendering identifies it and its SQL re-executes it — so it is observed
/// and proposed from but **not** persisted. A filter whose clause cannot be
/// rendered back to SQL never becomes one of these.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FilterAsk {
    /// The table the filter ran on.
    pub table: TableId,
    /// The canonical filter — the identity a [`FilterSet`] matches on.
    pub filter: Filter,
    /// The clause as SQL, so building one can evaluate it.
    pub sql: String,
}

/// A filter set worth building.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterProposal {
    /// The table it would be built on.
    pub table: TableId,
    /// Which filter it caches.
    pub ask: FilterAsk,
    /// How many unaided queries asked it.
    pub queries: u64,
    /// Bytes those queries read.
    pub bytes_scanned: u64,
    /// The most it could possibly have saved: every byte those queries moved.
    ///
    /// A filter set still reads the candidate rows, so this is further from
    /// honest than a cube's ceiling — but it only gates building, where a
    /// bound suffices and an estimate would only pretend.
    pub ceiling_usd: f64,
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
    ///
    /// Two terms, because skipping a file saves two different things:
    ///
    /// ```text
    /// bytes not moved     dominates when files are large
    /// files not opened    dominates when files are small and many
    /// ```
    ///
    /// The second term is why this is not simply a byte count. A thousand
    /// hundred-kilobyte files — the state Iceberg compaction exists to fix —
    /// cost mostly round trips, so an index over them is worth considerably
    /// more than the bytes it saves. Counting only bytes would rank such a
    /// table below a fatter one that actually benefits less.
    pub fn expected_usd(&self, spread: &Spread, prices: &PriceTable) -> f64 {
        use crate::cost::Tier;
        use crate::place::Distance;

        let bytes = self.bytes_scanned as f64 * spread.index_advantage();
        // Files the index skips that the format's own ranges would not.
        let opens_avoided =
            self.queries as f64 * (spread.files_by_bounds - spread.files_by_index).max(0.0);

        // Divided by the reads in flight, because a scan that skips sixteen
        // files in parallel saves one wave of waiting, not sixteen. Left
        // undivided this term would overstate the benefit of an index over
        // many small files by exactly the engine's parallelism.
        let waves_avoided = opens_avoided / prices.concurrent_reads.max(1.0);

        bytes * prices.byte_usd(Tier::Hot, Distance::Far)
            + waves_avoided * prices.link(Distance::Far).first_byte_seconds * prices.cpu_second_usd
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
    /// Aggregate asks, kept whole: a cube proposal needs the spec and the
    /// filters to bake, which a fingerprint has deliberately dropped.
    /// In-memory only — the literals inside never reach storage.
    by_ask: BTreeMap<AggregateAsk, Seen>,
    /// In-memory only, for the same reason.
    by_filter: BTreeMap<FilterAsk, Seen>,
    /// Top-k asks. Unlike the two above this carries no literals — a
    /// [`NearestAsk`] holds no query vector — so nothing keeps it in memory
    /// except that nothing persists it yet.
    by_nearest: BTreeMap<NearestAsk, Seen>,
}

impl Workload {
    /// An empty workload.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record what a query cost.
    pub fn observe(&mut self, observation: Observation) {
        let shape = self
            .by_shape
            .entry(observation.fingerprint.clone())
            .or_default();
        shape.queries += 1;
        shape.bytes_read = shape.bytes_read.saturating_add(observation.bytes_read);
        shape.bytes_if_full_scan = shape
            .bytes_if_full_scan
            .saturating_add(observation.bytes_if_full_scan);

        if let Some(ask) = observation.aggregate {
            let seen = self.by_ask.entry(ask).or_default();
            seen.queries += 1;
            seen.bytes_read = seen.bytes_read.saturating_add(observation.bytes_read);
            seen.bytes_if_full_scan = seen
                .bytes_if_full_scan
                .saturating_add(observation.bytes_if_full_scan);
            if !observation.used.is_empty() {
                seen.helped += 1;
            }
        }

        // Grouped ignoring k: one index answers any k, so a query for the
        // nearest 10 and one for the nearest 100 are the same proposal. The
        // largest k seen is kept, since it bounds what the index must return.
        if let Some(ask) = observation.nearest {
            let key = self
                .by_nearest
                .keys()
                .find(|seen| {
                    seen.table == ask.table
                        && seen.nearest.field == ask.nearest.field
                        && seen.nearest.metric == ask.nearest.metric
                        && seen.nearest.dimension == ask.nearest.dimension
                })
                .cloned();
            let key = match key {
                Some(existing) if existing.nearest.k >= ask.nearest.k => existing,
                Some(existing) => {
                    // A larger k arrived: re-key the accumulated counts.
                    let seen = self.by_nearest.remove(&existing).unwrap_or_default();
                    self.by_nearest.insert(ask.clone(), seen);
                    ask
                }
                None => ask,
            };
            let seen = self.by_nearest.entry(key).or_default();
            seen.queries += 1;
            seen.bytes_read = seen.bytes_read.saturating_add(observation.bytes_read);
            seen.bytes_if_full_scan = seen
                .bytes_if_full_scan
                .saturating_add(observation.bytes_if_full_scan);
            if !observation.used.is_empty() {
                seen.helped += 1;
            }
        }

        // Filters an index can serve are its business; a filter set is
        // what remains for the ones it cannot.
        for ask in observation.filters {
            let eligible = match ask.filter.field {
                Some(field) => !observation.fingerprint.probeable.contains(&field),
                None => true,
            };
            if !eligible {
                continue;
            }
            let seen = self.by_filter.entry(ask).or_default();
            seen.queries += 1;
            seen.bytes_read = seen.bytes_read.saturating_add(observation.bytes_read);
            seen.bytes_if_full_scan = seen
                .bytes_if_full_scan
                .saturating_add(observation.bytes_if_full_scan);
            if !observation.used.is_empty() {
                seen.helped += 1;
            }
        }

        if !observation.used.is_empty() {
            shape.helped += 1;
        }
        for id in &observation.used {
            let credited = self.by_derived.entry(id.clone()).or_default();
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
            by_ask: BTreeMap::new(),
            by_filter: BTreeMap::new(),
            by_nearest: BTreeMap::new(),
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

    /// Cubes worth building, most promising first.
    ///
    /// Grouped by the ask itself rather than the shape, because the filters
    /// to bake are part of what is proposed. Only unaided asks propose: a
    /// query a cube already served does not need a second one.
    pub fn cube_proposals(&self, prices: &PriceTable, min_queries: u64) -> Vec<CubeProposal> {
        let mut proposals: Vec<CubeProposal> = self
            .by_ask
            .iter()
            .filter(|(_, seen)| seen.queries.saturating_sub(seen.helped) >= min_queries)
            .map(|(ask, seen)| CubeProposal {
                table: ask.table.clone(),
                ask: ask.clone(),
                queries: seen.queries,
                bytes_scanned: seen.bytes_read,
                ceiling_usd: seen.bytes_read as f64
                    * prices.byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far),
            })
            .collect();
        proposals.sort_by(|a, b| b.ceiling_usd.total_cmp(&a.ceiling_usd));
        proposals
    }

    /// Filter sets worth building, most promising first.
    ///
    /// Same rule as cubes: only unaided asks propose, and only once the
    /// filter has repeated enough to amortize the build. A filter an index
    /// could serve never reaches `by_filter` at all — see `observe`.
    pub fn filter_proposals(&self, prices: &PriceTable, min_queries: u64) -> Vec<FilterProposal> {
        let mut proposals: Vec<FilterProposal> = self
            .by_filter
            .iter()
            .filter(|(_, seen)| seen.queries.saturating_sub(seen.helped) >= min_queries)
            .map(|(ask, seen)| FilterProposal {
                table: ask.table.clone(),
                ask: ask.clone(),
                queries: seen.queries,
                bytes_scanned: seen.bytes_read,
                ceiling_usd: seen.bytes_read as f64
                    * prices.byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far),
            })
            .collect();
        proposals.sort_by(|a, b| b.ceiling_usd.total_cmp(&a.ceiling_usd));
        proposals
    }

    /// Vector indexes worth building, most promising first.
    ///
    /// Same rule as cubes: only unaided asks propose, grouped by what an
    /// index would cover rather than by query shape — the query vector is
    /// not part of the identity, so every search of one column under one
    /// metric counts toward the same proposal.
    pub fn vector_proposals(&self, prices: &PriceTable, min_queries: u64) -> Vec<VectorProposal> {
        let mut proposals: Vec<VectorProposal> = self
            .by_nearest
            .iter()
            .filter(|(_, seen)| seen.queries.saturating_sub(seen.helped) >= min_queries)
            .map(|(ask, seen)| VectorProposal {
                table: ask.table.clone(),
                ask: ask.clone(),
                queries: seen.queries,
                bytes_scanned: seen.bytes_read,
                ceiling_usd: seen.bytes_read as f64
                    * prices.byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far),
            })
            .collect();
        proposals.sort_by(|a, b| b.ceiling_usd.total_cmp(&a.ceiling_usd));
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
    use crate::layout::Spread;
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
            aggregate: None,
            nearest: None,
            approximate: false,
        }
    }

    fn unaided(fields: &[FieldId], bytes: u64) -> Observation {
        Observation {
            fingerprint: Fingerprint::of(&query_on(fields)),
            aggregate: None,
            nearest: None,
            bytes_read: bytes,
            bytes_if_full_scan: bytes,
            used: Vec::new(),
            filters: Vec::new(),
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
                aggregate: None,
                nearest: None,
                bytes_read: GB / 10,
                bytes_if_full_scan: GB,
                used: vec![DerivedId("idx".into())],
                filters: Vec::new(),
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
                aggregate: None,
                nearest: None,
                bytes_read: GB,
                bytes_if_full_scan: GB,
                used: Vec::new(),
                filters: Vec::new(),
            });
        }
        assert_eq!(workload.shapes(), 2);

        let proposals = workload.proposals(&PriceTable::default(), 1);
        assert_eq!(proposals.len(), 1, "one field, one index");
        assert_eq!(proposals[0].queries, 10, "counts sum across shapes");
    }

    /// One observation carrying a filter nothing probeable answers.
    fn filtered(field: FieldId, text: &str, bytes: u64, used: Vec<DerivedId>) -> Observation {
        Observation {
            fingerprint: Fingerprint {
                table: table(),
                probeable: BTreeSet::new(),
                opaque: BTreeSet::from([field]),
                projected: BTreeSet::from([7]),
            },
            aggregate: None,
            nearest: None,
            bytes_read: bytes,
            bytes_if_full_scan: bytes,
            used,
            filters: vec![FilterAsk {
                table: table(),
                filter: Filter {
                    field: Some(field),
                    text: text.into(),
                },
                sql: text.into(),
            }],
        }
    }

    #[test]
    fn a_repeated_unindexable_filter_earns_a_set() {
        let mut workload = Workload::new();
        for _ in 0..3 {
            workload.observe(filtered(4, "tenant_id > Int64(2)", GB, Vec::new()));
        }
        let proposals = workload.filter_proposals(&PriceTable::default(), 2);
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].ask.sql, "tenant_id > Int64(2)");
        assert_eq!(proposals[0].queries, 3);
    }

    #[test]
    fn a_filter_an_index_could_serve_is_not_proposed_for_a_set() {
        // `tenant_id = 1` is probeable — the index proposal covers it, and a
        // filter set would duplicate the same work at row granularity.
        let mut workload = Workload::new();
        let mut observation = filtered(4, "tenant_id = Int64(1)", GB, Vec::new());
        observation.fingerprint.probeable = BTreeSet::from([4]);
        observation.fingerprint.opaque = BTreeSet::new();
        for _ in 0..5 {
            workload.observe(observation.clone());
        }
        assert!(
            workload
                .filter_proposals(&PriceTable::default(), 1)
                .is_empty()
        );
        assert_eq!(workload.proposals(&PriceTable::default(), 1).len(), 1);
    }

    #[test]
    fn a_served_filter_does_not_propose_again() {
        let mut workload = Workload::new();
        for _ in 0..5 {
            workload.observe(filtered(
                4,
                "tenant_id > Int64(2)",
                GB,
                vec![DerivedId("fset:events:x".into())],
            ));
        }
        assert!(
            workload
                .filter_proposals(&PriceTable::default(), 1)
                .is_empty()
        );
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
            aggregate: None,
            nearest: None,
            bytes_read: GB / 4,
            bytes_if_full_scan: GB,
            used: vec![DerivedId("idx".into())],
            filters: Vec::new(),
        };
        assert_eq!(observation.bytes_saved(), GB - GB / 4);
    }

    #[test]
    fn reading_more_than_a_full_scan_saves_nothing_rather_than_less() {
        let observation = Observation {
            fingerprint: Fingerprint::of(&query_on(&[4])),
            aggregate: None,
            nearest: None,
            bytes_read: 2 * GB,
            bytes_if_full_scan: GB,
            used: vec![DerivedId("idx".into())],
            filters: Vec::new(),
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
                aggregate: None,
                nearest: None,
                bytes_read: 0,
                bytes_if_full_scan: GB,
                used: vec![DerivedId("useful".into())],
                filters: Vec::new(),
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
            aggregate: None,
            nearest: None,
            bytes_read: 0,
            bytes_if_full_scan: 1_000,
            used: vec![DerivedId("bloated".into())],
            filters: Vec::new(),
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
            aggregate: None,
            nearest: None,
            bytes_read: 1,
            bytes_if_full_scan: GB,
            used: vec![DerivedId("a".into())],
            filters: Vec::new(),
        });

        let credited = workload.credited(&DerivedId("a".into())).expect("credited");
        assert_eq!(credited.queries, 1);
        assert_eq!(credited.bytes_saved(), GB - 1);
        assert!(workload.credited(&DerivedId("b".into())).is_none());
    }

    /// Two tables saving the same bytes, ranked apart by how many files they
    /// stop opening.
    ///
    /// Both prune 90% of a scan, so a benefit made only of bytes would call
    /// them equal. They are not: one skips nine files, the other nine hundred,
    /// and every skipped file is a round trip not waited for.
    ///
    /// This is the small-files case that Iceberg compaction exists to address,
    /// and the reason `expected_usd` has a second term.
    #[test]
    fn skipping_many_small_files_is_worth_more_than_skipping_a_few_large_ones() {
        // A non-unit parallelism, so that the division is actually exercised.
        let prices = PriceTable::default().with_concurrent_reads(8.0);
        let queries = 10;
        let bytes_scanned = 10 * GB;

        let few_large = Spread {
            files: 10,
            files_by_bounds: 10.0,
            files_by_index: 1.0,
        };
        let many_small = Spread {
            files: 1_000,
            files_by_bounds: 1_000.0,
            files_by_index: 100.0,
        };

        // Ground truth for "a benefit made only of bytes cannot tell these
        // apart": the fraction of the scan removed is identical.
        assert_eq!(few_large.index_advantage(), many_small.index_advantage());

        let proposal = Proposal {
            table: table(),
            field: 1,
            queries,
            bytes_scanned,
            ceiling_usd: 0.0,
        };
        let large = proposal.expected_usd(&few_large, &prices);
        let small = proposal.expected_usd(&many_small, &prices);
        assert!(
            small > large,
            "many small files should rank higher: {small} vs {large}"
        );

        // And by exactly the round trips avoided, computed independently
        // here — including the parallelism, since skipping eight files at once
        // saves one wave of waiting rather than eight.
        let extra_opens = queries as f64 * ((1_000.0 - 100.0) - (10.0 - 1.0));
        let expected_gap = (extra_opens / prices.concurrent_reads)
            * prices.far_link.first_byte_seconds
            * prices.cpu_second_usd;
        assert!(
            (small - large - expected_gap).abs() < 1e-12,
            "gap {} should be {}",
            small - large,
            expected_gap
        );
    }
}
