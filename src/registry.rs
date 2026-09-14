//! What derived state exists, and which piece to use.
//!
//! The registry does three things: it holds [`Derived`] values, it asks each
//! one whether it may serve a query, and it keeps the total within a byte
//! budget. It contains no per-kind logic — everything it needs comes through
//! the [`Kind`](crate::derived::Kind) trait — which is what lets a new kind be
//! added without touching this file.

use std::collections::BTreeMap;

use crate::cost::PriceTable;
use crate::derived::{Decision, Derived, DerivedId, Query};
use crate::snapshot::SnapshotGraph;

/// One admissible candidate for serving a query.
#[derive(Clone, Debug)]
pub struct Candidate<'a> {
    /// The derived state.
    pub derived: &'a Derived,
    /// How it would be used.
    pub decision: Decision,
    /// What using it would cost.
    pub cost: crate::cost::Cost,
}

/// The derived state the engine knows about.
#[derive(Debug, Default)]
pub struct Registry {
    entries: BTreeMap<DerivedId, Derived>,
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add derived state, replacing any with the same id.
    pub fn register(&mut self, derived: Derived) {
        self.entries.insert(derived.id.clone(), derived);
    }

    /// Forget derived state.
    pub fn remove(&mut self, id: &DerivedId) -> Option<Derived> {
        self.entries.remove(id)
    }

    /// Look up derived state.
    pub fn get(&self, id: &DerivedId) -> Option<&Derived> {
        self.entries.get(id)
    }

    /// Every registered id, in order.
    pub fn ids(&self) -> Vec<DerivedId> {
        self.entries.keys().cloned().collect()
    }

    /// How many pieces are registered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes held.
    pub fn bytes(&self) -> u64 {
        self.entries.values().map(|d| d.bytes).sum()
    }

    /// Note that a piece served a query.
    pub fn record_use(&mut self, id: &DerivedId) {
        if let Some(d) = self.entries.get_mut(id) {
            d.record_use();
        }
    }

    /// Every registered piece and the rule's verdict on it, admitted or not.
    ///
    /// Ordered by id so the result is stable, which matters because this is
    /// what `EXPLAIN` reports: the most useful line of an explain output is
    /// often *why* a piece of derived state was not used.
    pub fn assess<'a>(
        &'a self,
        query: &Query,
        graph: &SnapshotGraph,
    ) -> Vec<(&'a Derived, Decision)> {
        self.entries
            .values()
            .map(|derived| (derived, derived.may_serve(query, graph)))
            .collect()
    }

    /// Every piece that may serve `query`, cheapest first.
    ///
    /// Inadmissible pieces are omitted entirely: the rule
    /// ([`Derived::may_serve`]) decides, and the registry never second-guesses
    /// it. Ordering is total and deterministic — cost, then fewest extra files
    /// to scan, then id — so plan choice is reproducible.
    pub fn candidates<'a>(
        &'a self,
        query: &Query,
        graph: &SnapshotGraph,
        prices: &PriceTable,
    ) -> Vec<Candidate<'a>> {
        let mut found: Vec<Candidate<'a>> = self
            .assess(query, graph)
            .into_iter()
            .filter(|(_, decision)| decision.is_admitted())
            .map(|(derived, decision)| Candidate {
                derived,
                decision,
                cost: derived.cost(prices),
            })
            .collect();

        found.sort_by(|a, b| {
            a.cost
                .usd
                .total_cmp(&b.cost.usd)
                .then_with(|| extra_files(&a.decision).cmp(&extra_files(&b.decision)))
                .then_with(|| a.derived.id.cmp(&b.derived.id))
        });
        found
    }

    /// The cheapest piece that may serve `query`.
    pub fn best<'a>(
        &'a self,
        query: &Query,
        graph: &SnapshotGraph,
        prices: &PriceTable,
    ) -> Option<Candidate<'a>> {
        self.candidates(query, graph, prices).into_iter().next()
    }

    /// Evict until at most `max_bytes` are held, least valuable first.
    ///
    /// Value is uses per byte: a small piece that serves many queries outranks
    /// a large one that serves few. This is a placeholder for the expected
    /// remaining value the design calls for, which cannot be computed until
    /// realized benefit is measured; the signature will not change when it is.
    ///
    /// Eviction is always safe. Derived state is disposable by construction, so
    /// the worst outcome is a slower query.
    pub fn evict_to(&mut self, max_bytes: u64) -> Vec<DerivedId> {
        let mut ranked: Vec<(DerivedId, u64)> = self
            .entries
            .values()
            .map(|d| (d.id.clone(), d.bytes))
            .collect();

        // Least valuable first; ties broken by id so eviction is deterministic.
        ranked.sort_by(|(a_id, _), (b_id, _)| {
            let a = &self.entries[a_id];
            let b = &self.entries[b_id];
            value_density(a)
                .total_cmp(&value_density(b))
                .then_with(|| a_id.cmp(b_id))
        });

        let mut held = self.bytes();
        let mut evicted = Vec::new();
        for (id, bytes) in ranked {
            if held <= max_bytes {
                break;
            }
            self.entries.remove(&id);
            held = held.saturating_sub(bytes);
            evicted.push(id);
        }
        evicted
    }
}

/// Uses per byte. Zero-byte state is infinitely valuable, so never evicted.
fn value_density(d: &Derived) -> f64 {
    if d.bytes == 0 {
        f64::INFINITY
    } else {
        d.uses() as f64 / d.bytes as f64
    }
}

fn extra_files(decision: &Decision) -> usize {
    match decision {
        Decision::UseWith { also_scan, .. } => also_scan.len(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::Cost;
    use crate::derived::{
        Derived, DerivedId, FieldId, Kind, PolicyFingerprint, Predicate, Refreshed, Rewrite, Source,
    };
    use crate::snapshot::{Diff, FileId, Snapshot, SnapshotId, TableId};
    use std::collections::BTreeSet;

    const POLICY: PolicyFingerprint = PolicyFingerprint(1);

    /// Substitutes for any query on its table, at a fixed price.
    #[derive(Debug)]
    struct Fake {
        usd: f64,
        wants: Option<FieldId>,
    }

    impl Kind for Fake {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn matches(&self, query: &Query) -> Option<Rewrite> {
            match self.wants {
                None => Some(Rewrite::Substitute { unionable: true }),
                Some(field) => query
                    .filtered_fields()
                    .contains(&field)
                    .then_some(Rewrite::Substitute { unionable: true }),
            }
        }
        fn cost(&self, _prices: &PriceTable) -> Cost {
            Cost {
                usd: self.usd,
                ..Cost::ZERO
            }
        }
        fn refresh(&mut self, _diff: &Diff) -> Refreshed {
            Refreshed::UpToDate
        }
    }

    fn table() -> TableId {
        TableId("events".into())
    }

    fn entry(id: &str, usd: f64, bytes: u64, wants: Option<FieldId>) -> Derived {
        Derived::new(
            DerivedId(id.into()),
            Source {
                table: table(),
                snapshot: SnapshotId(810),
            },
            POLICY,
            bytes,
            Box::new(Fake { usd, wants }),
        )
    }

    fn graph() -> SnapshotGraph {
        SnapshotGraph::new()
            .with(Snapshot::root(SnapshotId(810)).with_clean_file(FileId("a".into())))
    }

    fn query() -> Query {
        Query {
            table: table(),
            snapshot: SnapshotId(810),
            policy: POLICY,
            plan_hash: 1,
            plan: None,
            projected: BTreeSet::from([4]),
            predicates: vec![Predicate::Eq { field: 4, value: 7 }],
            aggregate: None,
        }
    }

    #[test]
    fn the_cheapest_admissible_candidate_wins() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(entry("expensive", 10.0, 1, None));
        r.register(entry("cheap", 1.0, 1, None));
        r.register(entry("middling", 5.0, 1, None));

        let best = r.best(&query(), &graph(), &prices).expect("a candidate");
        assert_eq!(best.derived.id, DerivedId("cheap".into()));
    }

    #[test]
    fn inadmissible_candidates_are_omitted() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        // Cheaper, but needs a predicate on field 9, which the query lacks.
        r.register(entry("wrong-shape", 0.1, 1, Some(9)));
        r.register(entry("usable", 1.0, 1, None));

        let candidates = r.candidates(&query(), &graph(), &prices);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].derived.id, DerivedId("usable".into()));
    }

    #[test]
    fn no_candidates_means_scan_the_table() {
        let prices = PriceTable::default();
        let r = Registry::new();
        assert!(r.best(&query(), &graph(), &prices).is_none());
    }

    #[test]
    fn ordering_is_deterministic_on_ties() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(entry("b", 1.0, 1, None));
        r.register(entry("a", 1.0, 1, None));
        r.register(entry("c", 1.0, 1, None));

        let ids: Vec<_> = r
            .candidates(&query(), &graph(), &prices)
            .into_iter()
            .map(|c| c.derived.id.clone())
            .collect();
        assert_eq!(
            ids,
            vec![
                DerivedId("a".into()),
                DerivedId("b".into()),
                DerivedId("c".into())
            ]
        );
    }

    #[test]
    fn eviction_frees_enough_and_keeps_the_valuable() {
        let mut r = Registry::new();
        r.register(entry("hot", 1.0, 100, None)); // many uses per byte
        r.register(entry("cold", 1.0, 100, None)); // never used
        for _ in 0..50 {
            r.record_use(&DerivedId("hot".into()));
        }
        assert_eq!(r.bytes(), 200);

        let evicted = r.evict_to(100);
        assert_eq!(evicted, vec![DerivedId("cold".into())]);
        assert!(r.bytes() <= 100);
        assert!(r.get(&DerivedId("hot".into())).is_some());
    }

    #[test]
    fn eviction_under_budget_does_nothing() {
        let mut r = Registry::new();
        r.register(entry("a", 1.0, 10, None));
        assert!(r.evict_to(1_000).is_empty());
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn eviction_can_empty_the_registry() {
        let mut r = Registry::new();
        r.register(entry("a", 1.0, 10, None));
        r.register(entry("b", 1.0, 10, None));
        assert_eq!(r.evict_to(0).len(), 2);
        assert!(r.is_empty());
        assert_eq!(r.bytes(), 0);
    }

    #[test]
    fn registering_the_same_id_replaces() {
        let mut r = Registry::new();
        r.register(entry("a", 1.0, 10, None));
        r.register(entry("a", 1.0, 20, None));
        assert_eq!(r.len(), 1);
        assert_eq!(r.bytes(), 20);
    }

    #[test]
    fn uses_are_recorded_and_survive_lookup() {
        let mut r = Registry::new();
        r.register(entry("a", 1.0, 10, None));
        r.record_use(&DerivedId("a".into()));
        r.record_use(&DerivedId("a".into()));
        assert_eq!(r.get(&DerivedId("a".into())).expect("present").uses(), 2);
    }

    #[test]
    fn recording_a_use_of_something_absent_is_harmless() {
        let mut r = Registry::new();
        r.record_use(&DerivedId("nope".into()));
        assert!(r.is_empty());
    }

    #[test]
    fn assess_reports_refusals_too_so_explain_can_say_why() {
        let mut r = Registry::new();
        r.register(entry("usable", 1.0, 1, None));
        r.register(entry("wrong-shape", 1.0, 1, Some(9)));

        let verdicts = r.assess(&query(), &graph());
        assert_eq!(
            verdicts.len(),
            2,
            "refused entries are reported, not hidden"
        );

        let refused: Vec<_> = verdicts
            .iter()
            .filter(|(_, d)| !d.is_admitted())
            .map(|(d, _)| d.id.clone())
            .collect();
        assert_eq!(refused, vec![DerivedId("wrong-shape".into())]);
    }

    #[test]
    fn assess_is_ordered_by_id() {
        let mut r = Registry::new();
        for id in ["c", "a", "b"] {
            r.register(entry(id, 1.0, 1, None));
        }
        let ids: Vec<_> = r
            .assess(&query(), &graph())
            .into_iter()
            .map(|(d, _)| d.id.clone())
            .collect();
        assert_eq!(
            ids,
            vec![
                DerivedId("a".into()),
                DerivedId("b".into()),
                DerivedId("c".into())
            ]
        );
    }
}
