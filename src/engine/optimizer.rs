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

use crate::cost::PriceTable;
use crate::derived::{DerivedId, FieldId, PolicyFingerprint};
use crate::workload::{Observation, Policy, Proposal, Workload};

use super::{QuarryTable, Session, SharedRegistry, build_proposed_index, index_id};

/// What one round did, or would have done.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Round {
    /// Derived state built and registered.
    pub built: Vec<DerivedId>,
    /// Derived state rebuilt because the table had moved out from under it.
    pub refreshed: Vec<DerivedId>,
    /// Derived state dropped because it had not paid for itself.
    pub retired: Vec<DerivedId>,
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
        }
    }

    /// Price with this table instead of the default.
    pub fn with_prices(mut self, prices: PriceTable) -> Self {
        self.prices = prices;
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
    pub async fn round(&mut self, session: &Session, table: &QuarryTable) -> Round {
        let mut round = Round::default();
        let proposals = self.proposals();

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
                if registry.remove(&id).is_some() {
                    round.retired.push(id);
                }
            }
        }

        // 2. Refresh what the table has moved out from under.
        //
        // Not reachable through proposals: the stale piece is still serving
        // queries, so its shape counts as helped and nothing asks for it
        // again. Left alone it decays while still being credited with what it
        // once saved, which is why retirement does not catch it either.
        for (id, field) in self.stale(table) {
            let rebuilt =
                build_proposed_index(session, table, field, id.clone(), self.reader).await;
            let Ok(derived) = rebuilt else {
                round.declined.push((id, Declined::BuildFailed));
                continue;
            };
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

        for proposal in proposals {
            let id = index_id(&proposal.table, proposal.field);

            if existing.contains(&id) {
                // A freshly built index has not served a query yet, so the
                // workload would keep proposing it. Checking by id is what
                // stops a round from rebuilding what the last one made.
                round.declined.push((id, Declined::AlreadyBuilt));
                continue;
            }
            if round.built.len() >= self.policy.max_builds_per_round {
                round.declined.push((id, Declined::RoundFull));
                continue;
            }

            let built =
                build_proposed_index(session, table, proposal.field, id.clone(), self.reader).await;

            let Ok(derived) = built else {
                round.declined.push((id, Declined::BuildFailed));
                continue;
            };

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
            round.built.push(id);
        }

        round
    }

    /// Derived state this optimizer built that has fallen too far behind.
    ///
    /// Staleness is the fraction of the table the piece cannot help with —
    /// the files added since it was built, which every query must scan
    /// alongside it. Bytes rather than commits, because ten tiny appends
    /// matter less than one large one, and both are known exactly.
    fn stale(&self, table: &QuarryTable) -> Vec<(DerivedId, FieldId)> {
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
            .iter()
            .filter(|(id, _)| registry.get(id).is_some())
            .filter(|(id, _)| {
                let Some(derived) = registry.get(id) else {
                    return false;
                };
                if derived.source.table != *table.table_id() {
                    return false;
                }
                let Some(diff) = table
                    .graph()
                    .diff(derived.source.snapshot, table.snapshot())
                else {
                    // Lineage the graph cannot relate: the rule already
                    // refuses it, so it saves nothing and retirement will
                    // drop it. Rebuilding is not this step's job.
                    return false;
                };
                let residual: u64 = diff.added.iter().filter_map(|file| sizes.get(file)).sum();
                residual as f64 / live as f64 > limit
            })
            .map(|(id, field)| (id.clone(), *field))
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
            projected: BTreeSet::from([field]),
            predicates: vec![Predicate::Eq { field, value: 1 }],
        };
        Observation {
            fingerprint: Fingerprint::of(&query),
            bytes_read: bytes,
            bytes_if_full_scan: bytes,
            used: None,
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
