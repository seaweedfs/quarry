//! Collects usage stats, recommends derived state to build or retire, and
//! refreshes state that has fallen behind.
//!
//! The optimizer does not build or retire automatically. It observes
//! workload through [`observe`](Optimizer::observe), produces structured
//! recommendations with cost analysis through [`recommend`](Optimizer::recommend),
//! and the caller acts on them explicitly through [`build`](Optimizer::build),
//! [`build_recommended`](Optimizer::build_recommended),
//! [`retire`](Optimizer::retire), or [`retire_recommended`](Optimizer::retire_recommended).
//!
//! What `round` does automatically is refresh: rebuilding state that has
//! fallen too far behind the table's current snapshot. This is correctness
//! maintenance, not optimization — a stale piece is still correct but helps
//! less and less, so it is rebuilt in place.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::cost::PriceTable;
use crate::derived::{DerivedId, FieldId, PolicyFingerprint};
use crate::layout::Spread;
use crate::snapshot::{Commits, SnapshotId, TableId};
use crate::workload::{
    AggregateAsk, FilterAsk, JoinAsk, NearestAsk, Observation, Policy, Proposal, TextAsk, Workload,
};

use super::{
    QuarryTable, Session, SharedRegistry, build_cube, build_join_hash, build_proposed_filter_set,
    build_proposed_index, build_proposed_text_index, build_vector_index, cube_id, estimate_overlap,
    filter_set_id, index_id, join_hash_id, parquet_bounds, text_index_id, vector_index_id,
};

/// A piece of derived state a round dropped, identified enough to delete it.
///
/// `field` and `at` are what [`Layout::index_path`](super::persist::Layout)
/// needs, so a caller holding a [`Store`](super::persist::Store) can
/// reclaim the bytes rather than leaving them as orphans.
#[derive(Clone, Debug, PartialEq)]
pub struct Retired {
    /// What was dropped.
    pub id: DerivedId,
    /// The field it indexed, when the optimizer knows it.
    ///
    /// `None` for state this optimizer neither built nor adopted: the
    /// registry keeps no field, so the caller falls back to listing the
    /// table's prefix and matching `at`.
    pub field: Option<FieldId>,
    /// The snapshot it was built from.
    pub at: SnapshotId,
}

/// What one round did, or would have done.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Round {
    /// Derived state built and registered.
    pub built: Vec<DerivedId>,
    /// Derived state rebuilt because the table had moved out from under it.
    pub refreshed: Vec<DerivedId>,
    /// Derived state dropped because it had not paid for itself.
    ///
    /// In-memory it is gone; anything it persisted is not. Deleting the blob
    /// is the caller's — the optimizer holds no storage — and [`Retired`]
    /// carries what it needs to do it.
    pub retired: Vec<Retired>,
    /// Proposals not acted on, and why.
    pub declined: Vec<(DerivedId, Declined)>,
}

impl Round {
    /// Whether anything changed.
    pub fn changed_anything(&self) -> bool {
        !self.built.is_empty() || !self.refreshed.is_empty() || !self.retired.is_empty()
    }
}

/// Why a proposal was not acted on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Declined {
    /// Something already covers this.
    AlreadyBuilt,
    /// Building it would exceed the byte budget.
    OverBudget,
    /// This round has already built as much as it may.
    RoundFull,
    /// The build itself failed.
    BuildFailed,
    /// The file format's own pruning is already about as good.
    ///
    /// Measurement found this to be the common case rather than a rarity: an
    /// index over a column the data is already clustered on skips files the
    /// reader was barely touching. See [`Spread`](crate::layout::Spread).
    NoAdvantage,
    /// Nothing is known about how the field's values are laid out.
    ///
    /// Only reported when the policy requires an advantage to be demonstrated.
    /// Refusing to build without evidence is the conservative direction: a
    /// useless index costs storage and a build, forever, for nothing.
    NoEvidence,
    /// Built before, and it did not pay.
    ///
    /// Retirement removes a piece; this is what stops the next round from
    /// rebuilding it. The negative evidence the design's calibration section
    /// calls for: a node retained and never reused says the prediction was
    /// wrong, and building it again would spend a scan relearning that.
    NotWorthIt,
}

/// Predicted versus realized for one piece of derived state.
///
/// `realized` counts only bytes not read: the wait a skipped file would have
/// cost is not metered per observation, so it understates what a piece
/// actually saved. Honest, and biased in the direction that underclaims.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    /// What was measured.
    pub id: DerivedId,
    /// What the proposal expected it to save when it was built.
    pub predicted_usd: f64,
    /// What it measurably did save: bytes not read, priced.
    pub realized_usd: f64,
    /// Queries it has served.
    pub queries: u64,
}

/// What kind of derived state a build recommendation targets.
#[derive(Clone, Debug, PartialEq)]
pub enum BuildKind {
    /// A scalar equality index on one field.
    Index {
        /// The field the index hashes.
        field: FieldId,
    },
    /// An aggregate cube.
    Cube {
        /// The aggregate the cube materializes.
        ask: AggregateAsk,
    },
    /// A cached filter result set.
    FilterSet {
        /// The filter whose result set is cached.
        ask: FilterAsk,
    },
    /// A flat vector index for top-k search.
    Vector {
        /// The nearest-neighbour ask the index serves.
        ask: NearestAsk,
    },
    /// An inverted text index.
    Text {
        /// The text ask the index serves.
        ask: TextAsk,
    },
    /// A join hash: the build-side rows of a join, sorted by the join key.
    JoinHash {
        /// The join ask the hash serves.
        ask: JoinAsk,
    },
}

/// A recommendation to build a piece of derived state, with the cost
/// analysis that justifies it.
///
/// The optimizer produces these from observed workload but does not act on
/// them. The caller reviews the expected savings against the build cost and
/// decides which to build — explicitly, through [`Optimizer::build`] or
/// [`Optimizer::build_recommended`].
#[derive(Clone, Debug, PartialEq)]
pub struct BuildRecommendation {
    /// The id the built state would have.
    pub id: DerivedId,
    /// What kind of state to build, and the ask that identifies it.
    pub kind: BuildKind,
    /// The table it would be built on.
    pub table: TableId,
    /// How many unaided queries asked for this shape.
    pub queries: u64,
    /// Bytes those queries scanned.
    pub bytes_scanned: u64,
    /// The most this could possibly save — assumes perfect pruning.
    pub ceiling_usd: f64,
    /// What it is expected to save, scaled by measured evidence.
    ///
    /// For indexes this uses [`Spread`]-based advantage; for cubes, filter
    /// sets, and vector indexes it falls back to the ceiling. Scaled by
    /// [`Optimizer::proven`] when prior builds exist.
    pub expected_savings_usd: f64,
    /// What one build costs: a full scan of the table, priced.
    pub build_cost_usd: f64,
}

/// A recommendation to retire a piece of derived state that has not paid off.
#[derive(Clone, Debug, PartialEq)]
pub struct RetireRecommendation {
    /// What would be dropped.
    pub id: DerivedId,
    /// The field it indexed, if the optimizer knows.
    pub field: Option<FieldId>,
    /// The snapshot it was built from.
    pub at: SnapshotId,
    /// What it measurably saved, in dollars.
    pub realized_savings_usd: f64,
    /// How many queries it served.
    pub queries_served: u64,
}

/// What the optimizer recommends, without acting.
///
/// Builds and retirements are recommendations; refresh stays automatic
/// because it is correctness maintenance, not optimization. The caller
/// approves each action explicitly.
#[derive(Clone, Debug, PartialEq)]
pub enum Recommendation {
    /// Build a new piece of derived state.
    Build(BuildRecommendation),
    /// Retire a piece that has not paid for itself.
    Retire(RetireRecommendation),
}

impl Recommendation {
    /// The id of the derived state this recommendation concerns.
    pub fn id(&self) -> &DerivedId {
        match self {
            Recommendation::Build(b) => &b.id,
            Recommendation::Retire(r) => &r.id,
        }
    }
}

/// Watches a table, and keeps its derived state worth having.
#[derive(Debug)]
pub struct Optimizer {
    workload: Workload,
    registry: SharedRegistry,
    policy: Policy,
    prices: PriceTable,
    reader: PolicyFingerprint,
    /// What this optimizer built, and on which field.
    ///
    /// Needed to rebuild a piece that has fallen behind, and kept here rather
    /// than asked of the piece itself: a `Kind` describes what it can answer,
    /// not how to remake it, and growing that trait to carry build
    /// instructions would make every kind pay for this one's convenience.
    ///
    /// A consequence worth naming: the optimizer maintains only what it built.
    /// Derived state registered by hand is left alone, which is the right
    /// default — and it is lost on restart, along with the in-memory registry
    /// itself.
    built: BTreeMap<DerivedId, FieldId>,
    /// Cubes this optimizer built, keyed the same way.
    built_cubes: BTreeMap<DerivedId, AggregateAsk>,
    /// Same for filter sets — the clause a rebuild re-runs.
    built_filters: BTreeMap<DerivedId, FilterAsk>,
    /// Same for vector indexes — the ask a rebuild re-reads the table for.
    built_vectors: BTreeMap<DerivedId, NearestAsk>,
    /// Same for text indexes.
    built_texts: BTreeMap<DerivedId, TextAsk>,
    /// Same for join hashes — the ask a rebuild re-reads the table for.
    built_joins: BTreeMap<DerivedId, JoinAsk>,
    /// What each build was expected to save, in dollars.
    ///
    /// Kept so a prediction can be held against the realized credit later:
    /// without it, what a proposal claimed evaporates at build time and there
    /// is nothing to calibrate against.
    predicted: BTreeMap<DerivedId, f64>,
    /// Commits heard about between rounds.
    ///
    /// A notification's only fact is "this table moved to at least this
    /// snapshot", which is exactly what `failed_at` suppression and `stale`
    /// ask about. Drained each round; an empty log changes nothing.
    commits: Commits,
    /// The snapshot at which each id's last build failed.
    ///
    /// A failed build is retried only when the table has moved, because a
    /// build that failed on an unchanged table fails the same way: the inputs
    /// are identical. Retrying every round would read the whole table once
    /// per round forever, for a failure that cannot resolve itself.
    failed_at: BTreeMap<DerivedId, SnapshotId>,
    /// What is known about how each field's values sit across the files.
    ///
    /// Supplied rather than derived, because the two inputs come from
    /// different places: per-file ranges from Iceberg manifest bounds, and
    /// rows-per-value from an estimate of distinct values that Iceberg does
    /// not record. Taking it as an input keeps the requirement visible.
    spreads: BTreeMap<FieldId, Spread>,
}

impl Optimizer {
    /// Watch the derived state in `registry`, under `policy`.
    ///
    /// Takes the *shared* registry, so what it builds becomes visible to every
    /// table holding the same one, without those tables being rebuilt.
    pub fn new(registry: SharedRegistry, policy: Policy) -> Self {
        Optimizer {
            workload: Workload::new(),
            registry,
            policy,
            prices: PriceTable::default(),
            reader: PolicyFingerprint(0),
            built: BTreeMap::new(),
            built_cubes: BTreeMap::new(),
            built_filters: BTreeMap::new(),
            built_vectors: BTreeMap::new(),
            built_texts: BTreeMap::new(),
            built_joins: BTreeMap::new(),
            predicted: BTreeMap::new(),
            commits: Commits::new(),
            failed_at: BTreeMap::new(),
            spreads: BTreeMap::new(),
        }
    }

    /// Price with this table instead of the default.
    pub fn with_prices(mut self, prices: PriceTable) -> Self {
        self.prices = prices;
        self
    }

    /// Listen for commits pushed by a catalog or storage backend.
    ///
    /// The log is shared: whoever hears the commit notes it, the next round
    /// drains it. A missed or absent notification is safe — the table's own
    /// snapshot is always the fallback — and a commit the graph cannot
    /// relate is ignored rather than trusted.
    pub fn with_commits(mut self, commits: Commits) -> Self {
        self.commits = commits;
        self
    }

    /// Build derived state readable by this principal.
    ///
    /// Derived state is keyed on the policy it was built under, so an
    /// optimizer building for one principal produces nothing usable by
    /// another. That is the correct behaviour and the reason this is explicit
    /// rather than defaulted silently.
    pub fn for_reader(mut self, reader: PolicyFingerprint) -> Self {
        self.reader = reader;
        self
    }

    /// Resume from a workload read back from storage.
    ///
    /// Credits are the part that matters: without them every recovered piece
    /// of derived state looks as though it has saved nothing, and the first
    /// round deletes all of it — worse than not recovering at all, because
    /// the bytes were there and are now gone.
    pub fn with_workload(mut self, workload: Workload) -> Self {
        self.workload = workload;
        self
    }

    /// Take over maintenance of derived state recovered from storage.
    ///
    /// Without this the optimizer would not know what a recovered index is
    /// on, so it could neither refresh it nor recognise it as already built —
    /// and would build a second one beside it.
    pub fn adopt(&mut self, fields: impl IntoIterator<Item = (DerivedId, FieldId)>) {
        self.built.extend(fields);
    }

    /// Say how each field's values are laid out across the files.
    ///
    /// Without this, a policy requiring a demonstrated advantage declines
    /// every proposal — deliberately, since building on no evidence is what
    /// measurement showed to be wrong.
    pub fn with_spreads(mut self, spreads: BTreeMap<FieldId, Spread>) -> Self {
        self.spreads = spreads;
        self
    }

    /// What is known about one field's layout.
    pub fn spread(&self, field: FieldId) -> Option<Spread> {
        self.spreads.get(&field).copied()
    }

    /// Record what a query cost.
    ///
    /// The caller passes the observation rather than the optimizer hooking the
    /// scan, because a query may be observed from more than one place — this
    /// engine, or another engine's scan reports.
    pub fn observe(&mut self, observation: Observation) {
        self.workload.observe(observation);
    }

    /// What has been observed.
    pub fn workload(&self) -> &Workload {
        &self.workload
    }

    /// The policy in force.
    pub fn policy(&self) -> Policy {
        self.policy
    }

    /// Predicted versus realized for everything measured on both sides.
    ///
    /// Only pieces this optimizer built have a prediction, and only pieces
    /// that have served have a realized figure — the intersection is where
    /// calibration is possible. The deferred future-reuse work needs exactly
    /// this comparison to exist.
    pub fn calibration(&self) -> Vec<Calibration> {
        use crate::cost::Tier;
        use crate::place::Distance;

        self.predicted
            .iter()
            .filter_map(|(id, predicted)| {
                let seen = self.workload.credited(id)?;
                Some(Calibration {
                    id: id.clone(),
                    predicted_usd: *predicted,
                    realized_usd: seen.bytes_saved() as f64
                        * self.prices.byte_usd(Tier::Hot, Distance::Far),
                    queries: seen.helped,
                })
            })
            .collect()
    }

    /// How much this id's last build proved out, as a multiplier on its next
    /// expectation: `realized / predicted`, capped at one — evidence can
    /// deflate a prediction, never inflate it.
    ///
    /// Reached only for ids absent from the registry, so a piece with a
    /// prediction but no credit is one that was built and then dropped —
    /// retired or rejected — having served nothing. That is a zero.
    fn proven(&self, id: &DerivedId) -> f64 {
        use crate::cost::Tier;
        use crate::place::Distance;

        let Some(&predicted) = self.predicted.get(id) else {
            return 1.0;
        };
        let realized = self
            .workload
            .credited(id)
            .map(|seen| seen.bytes_saved() as f64 * self.prices.byte_usd(Tier::Hot, Distance::Far))
            .unwrap_or(0.0);
        if predicted <= 0.0 {
            return if realized > 0.0 { 1.0 } else { 0.0 };
        }
        (realized / predicted).clamp(0.0, 1.0)
    }

    /// What a round would do, without doing it.
    ///
    /// The same decisions a round acts on, which is what makes the advisory
    /// mode trustworthy: it is not a separate code path that might disagree.
    pub fn proposals(&self) -> Vec<Proposal> {
        self.workload
            .proposals(&self.prices, self.policy.min_queries)
    }

    /// What the optimizer recommends, without acting.
    ///
    /// Produces build and retire recommendations with cost analysis: expected
    /// savings, build cost, and ceiling for builds; realized savings and
    /// queries served for retirements. The caller reviews these and explicitly
    /// calls [`build`](Self::build) or [`retire`](Self::retire) on the ones it
    /// approves — or [`build_recommended`](Self::build_recommended) and
    /// [`retire_recommended`](Self::retire_recommended) for all of them.
    ///
    /// Build recommendations are ordered by expected savings, most first.
    /// Retire recommendations are unordered.
    pub async fn recommend(
        &self,
        session: &Session,
        table: &Arc<QuarryTable>,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Retire recommendations: state that has not paid for itself.
        if self.workload.observed() >= self.policy.retire_after_queries {
            let registry = self.registry.read().expect("registry lock");
            for id in self
                .workload
                .retirements(&registry, &self.prices, self.policy.horizon_days)
            {
                let Some(derived) = registry.get(&id) else {
                    continue;
                };
                let seen = self.workload.credited(&id);
                let realized_usd = seen
                    .map(|s| {
                        s.bytes_saved() as f64
                            * self
                                .prices
                                .byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far)
                    })
                    .unwrap_or(0.0);
                recommendations.push(Recommendation::Retire(RetireRecommendation {
                    field: self.built.get(&id).copied(),
                    at: derived.source.snapshot,
                    queries_served: seen.map(|s| s.helped).unwrap_or(0),
                    realized_savings_usd: realized_usd,
                    id,
                }));
            }
        }

        // Build recommendations: the same proposals round() would act on,
        // with the same evidence and proven scaling, but returned rather than
        // built.
        let existing: BTreeSet<DerivedId> = self
            .registry
            .read()
            .expect("registry lock")
            .ids()
            .into_iter()
            .collect();

        let build_cost_usd = table.live_bytes() as f64
            * self
                .prices
                .byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far);

        // Index proposals — ranked by expected savings with evidence.
        let mut ranked: Vec<(BuildRecommendation, f64)> = Vec::new();
        for proposal in self.proposals() {
            let id = index_id(&proposal.table, proposal.field);
            if existing.contains(&id) {
                continue;
            }
            if proposal.table != *table.table_id() {
                continue;
            }
            let spread = self.evidence(session, table, proposal.field).await;
            let expected = spread
                .map(|s| proposal.expected_usd(&s, &self.prices))
                .unwrap_or(proposal.ceiling_usd)
                * self.proven(&id);
            ranked.push((
                BuildRecommendation {
                    id,
                    kind: BuildKind::Index {
                        field: proposal.field,
                    },
                    table: proposal.table.clone(),
                    queries: proposal.queries,
                    bytes_scanned: proposal.bytes_scanned,
                    ceiling_usd: proposal.ceiling_usd,
                    expected_savings_usd: expected,
                    build_cost_usd,
                },
                expected,
            ));
        }
        ranked.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        recommendations.extend(ranked.into_iter().map(|(r, _)| Recommendation::Build(r)));

        // Cube proposals.
        for proposal in self
            .workload
            .cube_proposals(&self.prices, self.policy.min_queries)
        {
            if proposal.table != *table.table_id() {
                continue;
            }
            let id = cube_id(&proposal.table, &proposal.ask);
            if existing.contains(&id) {
                continue;
            }
            let expected = proposal.ceiling_usd * self.proven(&id);
            recommendations.push(Recommendation::Build(BuildRecommendation {
                id,
                kind: BuildKind::Cube {
                    ask: proposal.ask.clone(),
                },
                table: proposal.table.clone(),
                queries: proposal.queries,
                bytes_scanned: proposal.bytes_scanned,
                ceiling_usd: proposal.ceiling_usd,
                expected_savings_usd: expected,
                build_cost_usd,
            }));
        }

        // Filter set proposals.
        for proposal in self
            .workload
            .filter_proposals(&self.prices, self.policy.min_queries)
        {
            if proposal.table != *table.table_id() {
                continue;
            }
            let id = filter_set_id(&proposal.table, &proposal.ask.filter);
            if existing.contains(&id) {
                continue;
            }
            let expected = proposal.ceiling_usd * self.proven(&id);
            recommendations.push(Recommendation::Build(BuildRecommendation {
                id,
                kind: BuildKind::FilterSet {
                    ask: proposal.ask.clone(),
                },
                table: proposal.table.clone(),
                queries: proposal.queries,
                bytes_scanned: proposal.bytes_scanned,
                ceiling_usd: proposal.ceiling_usd,
                expected_savings_usd: expected,
                build_cost_usd,
            }));
        }

        // Vector proposals.
        for proposal in self
            .workload
            .vector_proposals(&self.prices, self.policy.min_queries)
        {
            if proposal.table != *table.table_id() {
                continue;
            }
            let id = vector_index_id(&proposal.table, &proposal.ask.nearest);
            if existing.contains(&id) {
                continue;
            }
            let expected = proposal.ceiling_usd * self.proven(&id);
            recommendations.push(Recommendation::Build(BuildRecommendation {
                id,
                kind: BuildKind::Vector {
                    ask: proposal.ask.clone(),
                },
                table: proposal.table.clone(),
                queries: proposal.queries,
                bytes_scanned: proposal.bytes_scanned,
                ceiling_usd: proposal.ceiling_usd,
                expected_savings_usd: expected,
                build_cost_usd,
            }));
        }

        // Text proposals.
        for proposal in self
            .workload
            .text_proposals(&self.prices, self.policy.min_queries)
        {
            if proposal.table != *table.table_id() {
                continue;
            }
            let id = text_index_id(&proposal.table, proposal.ask.field);
            if existing.contains(&id) {
                continue;
            }
            let expected = proposal.ceiling_usd * self.proven(&id);
            recommendations.push(Recommendation::Build(BuildRecommendation {
                id,
                kind: BuildKind::Text {
                    ask: proposal.ask.clone(),
                },
                table: proposal.table.clone(),
                queries: proposal.queries,
                bytes_scanned: proposal.bytes_scanned,
                ceiling_usd: proposal.ceiling_usd,
                expected_savings_usd: expected,
                build_cost_usd,
            }));
        }

        // Join hash proposals.
        for proposal in self
            .workload
            .join_proposals(&self.prices, self.policy.min_queries)
        {
            if proposal.table != *table.table_id() {
                continue;
            }
            let id = join_hash_id(&proposal.table, &proposal.ask);
            if existing.contains(&id) {
                continue;
            }
            let expected = proposal.ceiling_usd * self.proven(&id);
            recommendations.push(Recommendation::Build(BuildRecommendation {
                id,
                kind: BuildKind::JoinHash {
                    ask: proposal.ask.clone(),
                },
                table: proposal.table.clone(),
                queries: proposal.queries,
                bytes_scanned: proposal.bytes_scanned,
                ceiling_usd: proposal.ceiling_usd,
                expected_savings_usd: expected,
                build_cost_usd,
            }));
        }

        recommendations
    }

    /// Build one piece of derived state from a recommendation.
    ///
    /// The caller picks a [`BuildRecommendation`] from [`recommend`](Self::recommend)
    /// and passes it here. Returns the id on success, or the reason it was
    /// declined. Checks the byte budget on what was actually produced, since
    /// an index's size is not knowable before building it.
    pub async fn build(
        &mut self,
        session: &Session,
        table: &Arc<QuarryTable>,
        rec: &BuildRecommendation,
    ) -> Result<DerivedId, Declined> {
        let head = self.head(table);

        if self
            .registry
            .read()
            .expect("registry lock")
            .get(&rec.id)
            .is_some()
        {
            return Err(Declined::AlreadyBuilt);
        }
        if self.failed_at.get(&rec.id).is_some_and(|at| *at >= head) {
            return Err(Declined::BuildFailed);
        }

        // A build whose own evidence says it will not pay the scan it costs is
        // declined rather than re-tried. Only measured under-prediction
        // triggers this — a first-time proposal is judged by its estimate
        // alone, where `proven` is 1.0.
        if self.proven(&rec.id) < 1.0 && rec.expected_savings_usd <= rec.build_cost_usd {
            return Err(Declined::NotWorthIt);
        }

        let derived = match &rec.kind {
            BuildKind::Index { field } => {
                // The gate: would this beat what the file format prunes for
                // free? Asked before building, because the answer is usually
                // no, and building first and measuring after means paying for
                // the build and the storage to learn it was pointless.
                if self.policy.min_index_advantage_pct > 0.0 {
                    let spread = self.evidence(session, table, *field).await;
                    match spread {
                        None => return Err(Declined::NoEvidence),
                        Some(spread)
                            if spread.index_advantage_pct()
                                < self.policy.min_index_advantage_pct =>
                        {
                            return Err(Declined::NoAdvantage);
                        }
                        Some(_) => {}
                    }
                }
                build_proposed_index(session, table, *field, rec.id.clone(), self.reader).await
            }
            BuildKind::Cube { ask } => build_cube(session, table, ask, rec.id.clone()).await,
            BuildKind::FilterSet { ask } => {
                match build_proposed_filter_set(session, table, ask, rec.id.clone(), self.reader)
                    .await
                {
                    Ok(Some(d)) => Ok(d),
                    Ok(None) => return Err(Declined::NoAdvantage),
                    Err(e) => Err(e),
                }
            }
            BuildKind::Vector { ask } => {
                build_vector_index(session, table, &ask.nearest, rec.id.clone()).await
            }
            BuildKind::Text { ask } => {
                match build_proposed_text_index(
                    session,
                    table,
                    ask.field,
                    rec.id.clone(),
                    self.reader,
                )
                .await
                {
                    Ok(Some(d)) => Ok(d),
                    Ok(None) => return Err(Declined::NoAdvantage),
                    Err(e) => Err(e),
                }
            }
            BuildKind::JoinHash { ask } => {
                build_join_hash(session, table, ask, rec.id.clone()).await
            }
        };

        let derived = match derived {
            Ok(d) => d,
            Err(_) => {
                self.failed_at.insert(rec.id.clone(), head);
                return Err(Declined::BuildFailed);
            }
        };
        self.failed_at.remove(&rec.id);

        let held = self.registry.read().expect("registry lock").bytes();
        if held + derived.bytes > self.policy.budget_bytes {
            return Err(Declined::OverBudget);
        }

        self.registry
            .write()
            .expect("registry lock")
            .register(derived);

        // Record what was built so refresh and retirement can find it.
        match &rec.kind {
            BuildKind::Index { field } => {
                self.built.insert(rec.id.clone(), *field);
            }
            BuildKind::Cube { ask } => {
                self.built_cubes.insert(rec.id.clone(), ask.clone());
            }
            BuildKind::FilterSet { ask } => {
                self.built_filters.insert(rec.id.clone(), ask.clone());
            }
            BuildKind::Vector { ask } => {
                self.built_vectors.insert(rec.id.clone(), ask.clone());
            }
            BuildKind::Text { ask } => {
                self.built_texts.insert(rec.id.clone(), ask.clone());
            }
            BuildKind::JoinHash { ask } => {
                self.built_joins.insert(rec.id.clone(), ask.clone());
            }
        }
        self.predicted
            .insert(rec.id.clone(), rec.expected_savings_usd);
        Ok(rec.id.clone())
    }

    /// Build all recommended state, respecting budget and `max_builds_per_round`.
    ///
    /// Calls [`recommend`](Self::recommend) and acts on every build
    /// recommendation, in ranked order. Returns a [`Round`] with what was built
    /// and what was declined.
    pub async fn build_recommended(
        &mut self,
        session: &Session,
        table: &Arc<QuarryTable>,
    ) -> Round {
        let mut round = Round::default();
        let recommendations = self.recommend(session, table).await;

        for rec in recommendations {
            let Recommendation::Build(rec) = rec else {
                continue;
            };
            if round.built.len() >= self.policy.max_builds_per_round {
                round.declined.push((rec.id.clone(), Declined::RoundFull));
                continue;
            }
            match self.build(session, table, &rec).await {
                Ok(id) => round.built.push(id),
                Err(reason) => round.declined.push((rec.id.clone(), reason)),
            }
        }
        round
    }

    /// Retire one piece of derived state by id.
    ///
    /// Returns the [`Retired`] record so the caller can reclaim storage.
    pub fn retire(&mut self, id: &DerivedId) -> Option<Retired> {
        let snapshot = {
            let registry = self.registry.read().expect("registry lock");
            registry.get(id).map(|d| d.source.snapshot)
        };
        let snapshot = snapshot?;
        let retired = Retired {
            field: self.built.get(id).copied(),
            at: snapshot,
            id: id.clone(),
        };
        if self
            .registry
            .write()
            .expect("registry lock")
            .remove(id)
            .is_some()
        {
            self.built.remove(id);
            self.built_cubes.remove(id);
            self.built_filters.remove(id);
            self.built_vectors.remove(id);
            self.built_texts.remove(id);
            // The prediction is kept, not dropped: it is the evidence that the
            // next round's `NotWorthIt` check needs. A retired piece served
            // nothing, so `proven` reads it as zero, and a proposal that
            // returns is declined rather than rebuilt — without spending a
            // scan to relearn what the last build already proved.
            Some(retired)
        } else {
            None
        }
    }

    /// Retire all state the optimizer recommends retiring.
    ///
    /// Returns a [`Round`] with what was retired.
    pub async fn retire_recommended(
        &mut self,
        session: &Session,
        table: &Arc<QuarryTable>,
    ) -> Round {
        let mut round = Round::default();
        let recommendations = self.recommend(session, table).await;

        for rec in recommendations {
            let Recommendation::Retire(rec) = rec else {
                continue;
            };
            if let Some(retired) = self.retire(&rec.id) {
                round.retired.push(retired);
            }
        }
        round
    }

    /// The newest snapshot the table is known to have reached.
    fn head(&self, table: &QuarryTable) -> SnapshotId {
        self.commits
            .drain()
            .get(table.table_id())
            .copied()
            .filter(|moved| table.graph().get(*moved).is_some())
            .unwrap_or_else(|| table.snapshot())
    }

    /// Run one maintenance round against `table`.
    ///
    /// Refreshes derived state that has fallen too far behind the table's
    /// current snapshot. This is correctness maintenance, not optimization:
    /// a stale piece is still *correct* (the rule reads residual files
    /// alongside it) but it helps less and less, so it is rebuilt in place.
    ///
    /// Builds and retirements are **not** done here. The optimizer collects
    /// usage stats and produces recommendations through [`recommend`](Self::recommend);
    /// the caller acts on them explicitly through [`build`](Self::build),
    /// [`build_recommended`](Self::build_recommended),
    /// [`retire`](Self::retire), or [`retire_recommended`](Self::retire_recommended).
    pub async fn round(&mut self, session: &Session, table: &Arc<QuarryTable>) -> Round {
        let mut round = Round::default();
        let head = self.head(table);

        // Refresh what the table has moved out from under.
        //
        // Not reachable through proposals: the stale piece is still serving
        // queries, so its shape counts as helped and nothing asks for it
        // again. Left alone it decays while still being credited with what it
        // once saved, which is why retirement does not catch it either.
        for id in self.stale(table, head) {
            if self.failed_at.get(&id).is_some_and(|at| *at >= head) {
                continue;
            }
            let rebuilt = if let Some(&field) = self.built.get(&id) {
                build_proposed_index(session, table, field, id.clone(), self.reader).await
            } else if let Some(ask) = self.built_cubes.get(&id) {
                build_cube(session, table, ask, id.clone()).await
            } else if let Some(ask) = self.built_filters.get(&id) {
                match build_proposed_filter_set(session, table, ask, id.clone(), self.reader).await
                {
                    Ok(None) => datafusion::common::exec_err!("the filter admits every row"),
                    Ok(Some(derived)) => Ok(derived),
                    Err(e) => Err(e),
                }
            } else if let Some(ask) = self.built_vectors.get(&id) {
                build_vector_index(session, table, &ask.nearest, id.clone()).await
            } else if let Some(ask) = self.built_texts.get(&id) {
                match build_proposed_text_index(session, table, ask.field, id.clone(), self.reader)
                    .await
                {
                    Ok(None) => datafusion::common::exec_err!("the index prunes nothing"),
                    Ok(Some(derived)) => Ok(derived),
                    Err(e) => Err(e),
                }
            } else if let Some(ask) = self.built_joins.get(&id) {
                build_join_hash(session, table, ask, id.clone()).await
            } else {
                continue;
            };
            let Ok(derived) = rebuilt else {
                self.failed_at.insert(id.clone(), head);
                round.declined.push((id, Declined::BuildFailed));
                continue;
            };
            self.failed_at.remove(&id);
            self.registry
                .write()
                .expect("registry lock")
                .register(derived);
            round.refreshed.push(id);
        }

        round
    }

    /// How a field's values sit across the files, gathered if not supplied.
    ///
    /// Two inputs, from two places, both cheap:
    ///
    /// ```text
    /// per-file ranges   from the table, which took them from the catalog's
    ///                   manifest bounds at no extra cost
    /// value overlap     measured from two files, not the whole table
    /// ```
    ///
    /// `None` when the ranges cannot be had at all, which is the honest answer
    /// rather than the convenient one. Absent bounds would make the format
    /// look as though it prunes nothing, and an index maximally valuable —
    /// optimistic in exactly the direction that produced the useless indexes
    /// to begin with. A caller who knows better can still say so with
    /// [`Optimizer::with_spreads`].
    async fn evidence(
        &self,
        session: &Session,
        table: &QuarryTable,
        field: FieldId,
    ) -> Option<Spread> {
        if let Some(supplied) = self.spread(field) {
            return Some(supplied);
        }

        // From the catalog if the table came with them, otherwise from the
        // Parquet footers. Both describe the same thing; the catalog's copy is
        // simply already in hand, while footers cost a small read per file.
        let ranges = match table.bounds_of(field) {
            Some(ranges) => ranges.to_vec(),
            None => parquet_bounds(session, table, field).await.ok()?,
        };
        if ranges.is_empty() {
            return None;
        }

        let overlap = estimate_overlap(session, table, field).await.ok()??;
        Some(Spread::from_overlap(table.file_count(), &ranges, overlap))
    }

    /// Derived state this optimizer built that has fallen too far behind.
    ///
    /// Staleness is the fraction of the table the piece cannot help with —
    /// the files added since it was built, which every query must scan
    /// alongside it. Bytes rather than commits, because ten tiny appends
    /// matter less than one large one, and both are known exactly.
    /// `head` is the newest snapshot the table is known to have reached —
    /// pushed or polled. When a commit names one the graph cannot relate,
    /// the caller has already reduced it to the table's own.
    fn stale(&self, table: &QuarryTable, head: SnapshotId) -> Vec<DerivedId> {
        let live = table.live_bytes();
        if live == 0 {
            return Vec::new();
        }
        let Some((_, sizes)) = table.parquet_files() else {
            return Vec::new();
        };
        let limit = self.policy.max_residual_pct / 100.0;
        let registry = self.registry.read().expect("registry lock");

        self.built
            .keys()
            .chain(self.built_cubes.keys())
            .chain(self.built_filters.keys())
            .chain(self.built_vectors.keys())
            .chain(self.built_texts.keys())
            .filter(|id| registry.get(id).is_some())
            .filter(|id| {
                let Some(derived) = registry.get(id) else {
                    return false;
                };
                if derived.source.table != *table.table_id() {
                    return false;
                }
                let Some(diff) = table.graph().diff(derived.source.snapshot, head) else {
                    // Lineage the graph cannot relate: the rule already
                    // refuses it, so it saves nothing and retirement will
                    // drop it. Rebuilding is not this step's job.
                    return false;
                };
                let residual: u64 = diff.added.iter().filter_map(|file| sizes.get(file)).sum();
                residual as f64 / live as f64 > limit
            })
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Registry;
    use crate::snapshot::TableId;
    use crate::workload::Fingerprint;

    use super::super::shared;

    fn optimizer(policy: Policy) -> Optimizer {
        Optimizer::new(shared(Registry::new()), policy)
    }

    fn observation(table: &str, field: u32, bytes: u64) -> Observation {
        use crate::derived::{Predicate, Query};
        use crate::snapshot::SnapshotId;
        use std::collections::BTreeSet;

        let query = Query {
            table: TableId(table.into()),
            snapshot: SnapshotId(1),
            policy: PolicyFingerprint(0),
            plan_hash: 1,
            plan: None,
            projected: BTreeSet::from([field]),
            predicates: vec![Predicate::Eq { field, value: 1 }],
            aggregate: None,
            nearest: None,
            join: None,
            approximate: false,
        };
        Observation {
            fingerprint: Fingerprint::of(&query),
            aggregate: None,
            nearest: None,
            join: None,
            text: Vec::new(),
            bytes_read: bytes,
            bytes_if_full_scan: bytes,
            used: Vec::new(),
            filters: Vec::new(),
        }
    }

    #[test]
    fn an_advisory_optimizer_still_proposes() {
        let mut optimizer = optimizer(Policy::ADVISORY.with_min_queries(3));
        for _ in 0..5 {
            optimizer.observe(observation("events", 4, 1_000_000));
        }
        let proposals = optimizer.proposals();
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].field, 4);
    }

    #[test]
    fn proposals_respect_the_minimum_query_count() {
        let mut optimizer = optimizer(Policy::ADVISORY.with_min_queries(10));
        for _ in 0..5 {
            optimizer.observe(observation("events", 4, 1_000_000));
        }
        assert!(optimizer.proposals().is_empty());
    }

    #[test]
    fn recommend_reports_what_could_be_built() {
        let mut optimizer = optimizer(Policy::ADVISORY.with_min_queries(1));
        optimizer.observe(observation("events", 4, 1_000_000));

        let proposals = optimizer.proposals();
        assert_eq!(proposals.len(), 1);
    }

    #[test]
    fn a_percentage_budget_scales_with_the_table() {
        let policy = Policy::automatic_pct(1_000_000_000, 5.0);
        assert_eq!(policy.budget_bytes, 50_000_000);
    }

    #[test]
    fn a_round_that_changed_nothing_says_so() {
        assert!(!Round::default().changed_anything());
        assert!(
            Round {
                built: vec![DerivedId("x".into())],
                ..Round::default()
            }
            .changed_anything()
        );
    }
}
