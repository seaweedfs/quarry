//! The thing that runs the loop without being asked.
//!
//! Every step existed already — observe, propose, build, measure, retire —
//! and a caller had to sequence them. This owns the sequence, and owns the
//! two pieces of state the sequence needs: what queries asked for, and what
//! has been built.
//!
//! # Order within a round, and why it is that order
//!
//! ```text
//! 1. retire first    frees budget, so a useful index is not refused
//!                    because a useless one is occupying the space
//! 2. then build      cheapest-first is wrong here; most-at-stake first,
//!                    which is the order proposals already come in
//! 3. budget after    an index's size is not knowable until it is built,
//!                    so the ceiling is enforced on what was produced
//!                    rather than on a guess
//! ```
//!
//! Step 3 is the same distinction drawn everywhere else in this crate:
//! estimation informs, enforcement decides. A build that turns out too large
//! for the remaining budget is discarded rather than kept and apologised for.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::cost::PriceTable;
use crate::derived::{DerivedId, FieldId, PolicyFingerprint};
use crate::layout::Spread;
use crate::snapshot::{Commits, SnapshotId};
use crate::workload::{AggregateAsk, Observation, Policy, Proposal, Workload};

use super::{
    QuarryTable, Session, SharedRegistry, build_cube, build_proposed_index, cube_id,
    estimate_overlap, index_id, parquet_bounds,
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
    /// The policy forbids acting.
    Advisory,
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

    /// Run one round against `table`.
    ///
    /// Retires first, then builds, enforcing the byte budget on what was
    /// actually produced. Reports what it declined and why, whether or not it
    /// was allowed to act.
    pub async fn round(&mut self, session: &Session, table: &Arc<QuarryTable>) -> Round {
        let mut round = Round::default();
        let proposals = self.proposals();

        // The newest position the table is known to have reached: the pushed
        // commit when there is one, what the table says otherwise. A commit
        // the graph cannot relate reduces to the table's own snapshot.
        let head = self
            .commits
            .drain()
            .get(table.table_id())
            .copied()
            .filter(|moved| table.graph().get(*moved).is_some())
            .unwrap_or_else(|| table.snapshot());

        if !self.policy.auto_optimize {
            for proposal in &proposals {
                round.declined.push((
                    index_id(&proposal.table, proposal.field),
                    Declined::Advisory,
                ));
            }
            return round;
        }

        // 1. Retire, so freed bytes are available to what follows.
        //
        // Held off until enough queries have been seen to judge by.
        // Retirement asks what a piece has measurably saved and reads nothing
        // as a reason to delete, which is right once traffic has run and
        // wrong immediately after a restart, when a recovered index has
        // served nothing yet. The asymmetry decides it: rebuilding a deleted
        // index costs a scan, keeping a useless one another round costs
        // almost nothing.
        let retire = if self.workload.observed() < self.policy.retire_after_queries {
            Vec::new()
        } else {
            let registry = self.registry.read().expect("registry lock");
            self.workload
                .retirements(&registry, &self.prices, self.policy.horizon_days)
        };
        if !retire.is_empty() {
            let mut registry = self.registry.write().expect("registry lock");
            for id in retire {
                // Read before removing: the caller needs the field and the
                // snapshot to find the blob, and both live on the entry.
                let Some(derived) = registry.get(&id) else {
                    continue;
                };
                let retired = Retired {
                    field: self.built.get(&id).copied(),
                    at: derived.source.snapshot,
                    id,
                };
                if registry.remove(&retired.id).is_some() {
                    round.retired.push(retired);
                }
            }
        }

        // 2. Refresh what the table has moved out from under.
        //
        // Not reachable through proposals: the stale piece is still serving
        // queries, so its shape counts as helped and nothing asks for it
        // again. Left alone it decays while still being credited with what it
        // once saved, which is why retirement does not catch it either.
        for id in self.stale(table, head) {
            if self.failed_at.get(&id).is_some_and(|at| *at >= head) {
                // Already failed on the newest known table state.
                continue;
            }
            let rebuilt = if let Some(&field) = self.built.get(&id) {
                build_proposed_index(session, table, field, id.clone(), self.reader).await
            } else if let Some(ask) = self.built_cubes.get(&id) {
                build_cube(session, table, ask, id.clone()).await
            } else {
                continue;
            };
            let Ok(derived) = rebuilt else {
                self.failed_at.insert(id.clone(), head);
                round.declined.push((id, Declined::BuildFailed));
                continue;
            };
            self.failed_at.remove(&id);
            // Replaced only once the replacement exists, so a failed rebuild
            // leaves the stale-but-correct piece in place rather than nothing.
            self.registry
                .write()
                .expect("registry lock")
                .register(derived);
            round.refreshed.push(id);
        }

        // 3. Build, most at stake first.
        let existing: BTreeSet<DerivedId> = self
            .registry
            .read()
            .expect("registry lock")
            .ids()
            .into_iter()
            .collect();

        // Cheap refusals first, then evidence, then rank, then build. The
        // order matters: `max_builds_per_round` used to be applied while
        // walking proposals in `ceiling_usd` order, which measurement showed
        // to be 1775x wrong on one regime and unbounded on another — so with
        // the default of one build per round, a round could build the worst
        // candidate and decline the best as `RoundFull`.
        let mut ranked: Vec<(Proposal, f64)> = Vec::new();
        for proposal in proposals {
            let id = index_id(&proposal.table, proposal.field);

            if existing.contains(&id) {
                // A freshly built index has not served a query yet, so the
                // workload would keep proposing it. Checking by id is what
                // stops a round from rebuilding what the last one made.
                round.declined.push((id, Declined::AlreadyBuilt));
                continue;
            }

            // Would this beat what the file format prunes for free? Asked
            // before building, because the answer is usually no, and because
            // building first and measuring after means paying for the build
            // and the storage to learn it was pointless.
            let spread = self.evidence(session, table, proposal.field).await;
            if self.policy.min_index_advantage_pct > 0.0 {
                match spread {
                    None => {
                        round.declined.push((id, Declined::NoEvidence));
                        continue;
                    }
                    Some(spread)
                        if spread.index_advantage_pct() < self.policy.min_index_advantage_pct =>
                    {
                        round.declined.push((id, Declined::NoAdvantage));
                        continue;
                    }
                    Some(_) => {}
                }
            }

            // Ranked by what it is expected to save, not by the ceiling.
            // Without evidence there is nothing to scale the ceiling by, which
            // only happens when the policy does not require evidence.
            let expected = spread
                .map(|spread| proposal.expected_usd(&spread, &self.prices))
                .unwrap_or(proposal.ceiling_usd)
                * self.proven(&id);
            ranked.push((proposal, expected));
        }

        ranked.sort_by(|(left, one), (right, other)| {
            other
                .total_cmp(one)
                .then_with(|| left.field.cmp(&right.field))
        });

        for (proposal, expected) in ranked {
            let id = index_id(&proposal.table, proposal.field);
            if round.built.len() >= self.policy.max_builds_per_round {
                round.declined.push((id, Declined::RoundFull));
                continue;
            }

            // Building costs a scan; a build whose own evidence says it will
            // not pay that back is declined rather than re-tried. Only
            // measured under-prediction triggers this — a first-time proposal
            // is judged by its estimate alone.
            if self.proven(&id) < 1.0
                && expected
                    <= table.live_bytes() as f64
                        * self
                            .prices
                            .byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far)
            {
                round.declined.push((id, Declined::NotWorthIt));
                continue;
            }

            if self.failed_at.get(&id).is_some_and(|at| *at >= head) {
                continue;
            }
            let built =
                build_proposed_index(session, table, proposal.field, id.clone(), self.reader).await;

            let Ok(derived) = built else {
                self.failed_at.insert(id.clone(), head);
                round.declined.push((id, Declined::BuildFailed));
                continue;
            };
            self.failed_at.remove(&id);

            // The ceiling, applied to what was produced. An index's size
            // cannot be known before building it, so this is the only honest
            // place to check.
            let held = self.registry.read().expect("registry lock").bytes();
            if held + derived.bytes > self.policy.budget_bytes {
                round.declined.push((id, Declined::OverBudget));
                continue;
            }

            self.registry
                .write()
                .expect("registry lock")
                .register(derived);
            self.built.insert(id.clone(), proposal.field);
            self.predicted.insert(id.clone(), expected);
            round.built.push(id);
        }

        // 4. Cubes: an aggregate asked often enough is worth materialising.
        for proposal in self
            .workload
            .cube_proposals(&self.prices, self.policy.min_queries)
        {
            if proposal.table != *table.table_id() {
                continue;
            }
            let id = cube_id(&proposal.table, &proposal.ask);
            if existing.contains(&id) {
                round.declined.push((id, Declined::AlreadyBuilt));
                continue;
            }
            if round.built.len() >= self.policy.max_builds_per_round {
                round.declined.push((id, Declined::RoundFull));
                continue;
            }
            // Same gate as indexes: a ceiling scaled by how the last build of
            // this ask proved out, against the scan a rebuild costs. A cube's
            // prediction is a bound rather than an estimate, but a served
            // cube replaces the whole scan, so realized lands close to it.
            let proven = self.proven(&id);
            if proven < 1.0
                && proposal.ceiling_usd * proven
                    <= table.live_bytes() as f64
                        * self
                            .prices
                            .byte_usd(crate::cost::Tier::Hot, crate::place::Distance::Far)
            {
                round.declined.push((id, Declined::NotWorthIt));
                continue;
            }
            if self.failed_at.get(&id).is_some_and(|at| *at >= head) {
                continue;
            }
            let derived = match build_cube(session, table, &proposal.ask, id.clone()).await {
                Ok(derived) => derived,
                Err(_) => {
                    self.failed_at.insert(id.clone(), head);
                    round.declined.push((id, Declined::BuildFailed));
                    continue;
                }
            };
            self.failed_at.remove(&id);
            let held = self.registry.read().expect("registry lock").bytes();
            if held + derived.bytes > self.policy.budget_bytes {
                round.declined.push((id, Declined::OverBudget));
                continue;
            }
            self.registry
                .write()
                .expect("registry lock")
                .register(derived);
            self.built_cubes.insert(id.clone(), proposal.ask);
            self.predicted.insert(id.clone(), proposal.ceiling_usd);
            round.built.push(id);
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
        };
        Observation {
            fingerprint: Fingerprint::of(&query),
            aggregate: None,
            bytes_read: bytes,
            bytes_if_full_scan: bytes,
            used: Vec::new(),
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
    fn a_round_reports_what_it_would_have_done() {
        // Advisory: no session or table is needed, because nothing is built.
        let mut optimizer = optimizer(Policy::ADVISORY.with_min_queries(1));
        optimizer.observe(observation("events", 4, 1_000_000));

        // `round` needs a session to build; advisory returns before that, so
        // the declined list is checkable without one.
        let proposals = optimizer.proposals();
        assert_eq!(proposals.len(), 1);
        assert!(!optimizer.policy().auto_optimize);
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
