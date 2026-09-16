//! What derived state exists, and which piece to use.
//!
//! The registry does three things: it holds [`Derived`] values, it asks each
//! one whether it may serve a query, and it keeps the total within a byte
//! budget. It contains no per-kind logic — everything it needs comes through
//! the [`Kind`](crate::derived::Kind) trait — which is what lets a new kind be
//! added without touching this file.

use std::collections::{BTreeMap, BTreeSet};

use crate::cost::PriceTable;
use crate::derived::{Decision, Derived, DerivedId, Query, Rewrite, Scope};
use crate::snapshot::{FileId, SnapshotGraph};

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

/// The plan the rule composes from a query's candidates.
///
/// The cheapest executable candidate leads. A leading substitute answers
/// alone — stored rows replace the scan outright. When a prune leads, every
/// later admissible prune intersects its candidate files with the running
/// set; `a = 1 AND b = 2` reads the files both indexes name, not whichever
/// index is cheaper alone.
pub enum Composed<'a> {
    /// One substitute answers the query.
    Substitute(Candidate<'a>),
    /// Intersected pruning: the files that may match every predicate, the
    /// added-since files among them, and the pieces that contributed.
    ///
    /// A piece contributes when it strictly narrows the running set; the
    /// leader is always listed, even when it narrows nothing, because it is
    /// still the piece the scan reads its candidate files from.
    Pruned {
        /// Files to read, and how much of each: every contributing index's
        /// set intersected, each already unioned with the files added since
        /// it was built. Residual files are always whole.
        files: BTreeMap<FileId, Scope>,
        /// The added-since portion of `files`.
        residual: BTreeSet<FileId>,
        /// The pieces that narrowed the set, cheapest first.
        pieces: Vec<Candidate<'a>>,
    },
    /// Nothing admissible: a full scan.
    Scan,
}

/// Turn a query's candidates into a plan.
///
/// `can_substitute` is the caller's executability check: stored rows may not
/// have been supplied for an otherwise admissible substitute, and one that
/// cannot answer is skipped rather than leading. `Explain` passes `|_| true`
/// — it reports what the rule admits, executability being a runtime fact.
pub fn compose<'a>(
    candidates: &'a [Candidate<'a>],
    can_substitute: impl Fn(&Derived) -> bool,
) -> Composed<'a> {
    let mut files: Option<BTreeMap<FileId, Scope>> = None;
    let mut residual = BTreeSet::new();
    let mut pieces = Vec::new();
    for candidate in candidates {
        let (rewrite, also_scan) = match &candidate.decision {
            Decision::Use(rewrite) => (rewrite, None),
            Decision::UseStale(rewrite) => (rewrite, None),
            Decision::UseWith { rewrite, also_scan } => (rewrite, Some(also_scan)),
            Decision::Reject(_) => continue,
        };
        match rewrite {
            Rewrite::Substitute { .. } => {
                if files.is_none() && can_substitute(candidate.derived) {
                    return Composed::Substitute(candidate.clone());
                }
            }
            Rewrite::Prune { files: named } => {
                // Files added since the piece was built may match anything;
                // they join its set at whole-file scope.
                let mut effective = named.clone();
                for file in also_scan.into_iter().flatten() {
                    effective.entry(file.clone()).or_insert(Scope::Whole);
                }
                match &mut files {
                    None => {
                        residual.extend(also_scan.into_iter().flatten().cloned());
                        files = Some(effective);
                        pieces.push(candidate.clone());
                    }
                    Some(current) => {
                        let mut next = BTreeMap::new();
                        for (file, scope) in current.iter() {
                            if let Some(scope) =
                                effective.get(file).and_then(|o| scope.intersect(o))
                            {
                                next.insert(file.clone(), scope);
                            }
                        }
                        if *current != next {
                            residual.extend(also_scan.into_iter().flatten().cloned());
                            *current = next;
                            pieces.push(candidate.clone());
                        }
                    }
                }
            }
        }
    }
    match files {
        Some(files) => Composed::Pruned {
            residual: residual
                .into_iter()
                .filter(|f| files.contains_key(f))
                .collect(),
            files,
            pieces,
        },
        None => Composed::Scan,
    }
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
    use std::collections::{BTreeMap, BTreeSet};

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
                None => Some(Rewrite::Substitute {
                    unionable: true,
                    rollup: None,
                }),
                Some(field) => {
                    query
                        .filtered_fields()
                        .contains(&field)
                        .then_some(Rewrite::Substitute {
                            unionable: true,
                            rollup: None,
                        })
                }
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

    /// Names a fixed file set whenever its field carries an equality.
    #[derive(Debug)]
    struct Pruner {
        field: FieldId,
        files: BTreeMap<FileId, Scope>,
        usd: f64,
    }

    impl Kind for Pruner {
        fn name(&self) -> &'static str {
            "pruner"
        }
        fn matches(&self, query: &Query) -> Option<Rewrite> {
            query
                .equalities(self.field)
                .first()
                .map(|_| Rewrite::Prune {
                    files: self.files.clone(),
                })
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

    fn prune_entry(id: &str, field: u32, usd: f64, files: &[&str]) -> Derived {
        pruner(
            id,
            field,
            usd,
            files
                .iter()
                .map(|f| (FileId(f.to_string()), Scope::Whole))
                .collect(),
        )
    }

    /// A prune entry whose postings name row groups, not just files.
    fn scoped_entry(id: &str, field: u32, usd: f64, files: &[(&str, &[u32])]) -> Derived {
        pruner(
            id,
            field,
            usd,
            files
                .iter()
                .map(|(f, groups)| {
                    (
                        FileId(f.to_string()),
                        Scope::Groups(groups.iter().cloned().collect()),
                    )
                })
                .collect(),
        )
    }

    /// file → group → row offsets, as fixtures spell it.
    type RowPostings<'a> = &'a [(&'a str, &'a [(u32, &'a [u64])])];

    /// A prune entry whose postings name rows inside groups.
    fn row_entry(id: &str, field: u32, usd: f64, files: RowPostings) -> Derived {
        pruner(
            id,
            field,
            usd,
            files
                .iter()
                .map(|(f, groups)| {
                    (
                        FileId(f.to_string()),
                        Scope::Rows(
                            groups
                                .iter()
                                .map(|(g, rows)| (*g, rows.iter().cloned().collect()))
                                .collect(),
                        ),
                    )
                })
                .collect(),
        )
    }

    fn pruner(id: &str, field: u32, usd: f64, files: BTreeMap<FileId, Scope>) -> Derived {
        Derived::new(
            DerivedId(id.into()),
            Source {
                table: table(),
                snapshot: SnapshotId(810),
            },
            POLICY,
            files.len() as u64,
            Box::new(Pruner { field, files, usd }),
        )
    }

    fn and_query() -> Query {
        Query {
            predicates: vec![
                Predicate::Eq { field: 4, value: 7 },
                Predicate::Eq { field: 9, value: 2 },
            ],
            ..query()
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
        let mut root = Snapshot::root(SnapshotId(810));
        for f in ["a", "b", "c", "d"] {
            root = root.with_clean_file(FileId(f.into()));
        }
        SnapshotGraph::new().with(root)
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
            nearest: None,
            join: None,
            approximate: false,
            stale: false,
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

    #[test]
    fn admissible_prunes_intersect_their_candidate_files() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(prune_entry("a_idx", 4, 1.0, &["a", "b", "c"]));
        r.register(prune_entry("b_idx", 9, 2.0, &["b", "d"]));

        match compose(&r.candidates(&and_query(), &graph(), &prices), |_| true) {
            Composed::Pruned {
                files,
                residual,
                pieces,
            } => {
                assert_eq!(files, BTreeMap::from([(FileId("b".into()), Scope::Whole)]));
                assert!(residual.is_empty());
                assert_eq!(
                    pieces
                        .iter()
                        .map(|p| p.derived.id.clone())
                        .collect::<Vec<_>>(),
                    vec![DerivedId("a_idx".into()), DerivedId("b_idx".into())],
                    "the cheaper piece leads; the second still narrowed the set"
                );
            }
            _ => panic!("two admissible prunes must compose"),
        }
    }

    #[test]
    fn a_prune_that_narrows_nothing_is_not_listed() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(prune_entry("narrow", 4, 1.0, &["a"]));
        r.register(prune_entry("wide", 9, 2.0, &["a", "b", "c"]));

        match compose(&r.candidates(&and_query(), &graph(), &prices), |_| true) {
            Composed::Pruned { files, pieces, .. } => {
                assert_eq!(files, BTreeMap::from([(FileId("a".into()), Scope::Whole)]));
                assert_eq!(
                    pieces.len(),
                    1,
                    "a piece whose set is a superset contributed nothing"
                );
                assert_eq!(pieces[0].derived.id, DerivedId("narrow".into()));
            }
            _ => panic!("expected a pruned plan"),
        }
    }

    #[test]
    fn prunes_compose_to_no_files_at_all() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(prune_entry("a_idx", 4, 1.0, &["a"]));
        r.register(prune_entry("b_idx", 9, 2.0, &["b"]));

        match compose(&r.candidates(&and_query(), &graph(), &prices), |_| true) {
            Composed::Pruned { files, pieces, .. } => {
                assert!(files.is_empty(), "disjoint candidates: nothing can match");
                assert_eq!(pieces.len(), 2);
            }
            _ => panic!("expected a pruned plan"),
        }
    }

    #[test]
    fn residuals_intersect_with_postings() {
        let prices = PriceTable::default();
        let graph = SnapshotGraph::new()
            .with(
                Snapshot::root(SnapshotId(810))
                    .with_clean_file(FileId("a".into()))
                    .with_clean_file(FileId("b".into())),
            )
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                    .with_clean_file(FileId("a".into()))
                    .with_clean_file(FileId("b".into()))
                    .with_clean_file(FileId("d".into())),
            );
        let query = Query {
            snapshot: SnapshotId(811),
            ..and_query()
        };

        let mut r = Registry::new();
        r.register(prune_entry("a_idx", 4, 1.0, &["a", "b"]));
        r.register(prune_entry("b_idx", 9, 2.0, &["b"]));

        match compose(&r.candidates(&query, &graph, &prices), |_| true) {
            Composed::Pruned {
                files, residual, ..
            } => {
                // Both indexes predate file d, so each may-serve set is
                // postings + {d}: {a,b,d} ∩ {b,d} = {b,d}.
                assert_eq!(
                    files,
                    BTreeMap::from([
                        (FileId("b".into()), Scope::Whole),
                        (FileId("d".into()), Scope::Whole)
                    ])
                );
                assert_eq!(residual, BTreeSet::from([FileId("d".into())]));
            }
            _ => panic!("expected a pruned plan"),
        }
    }

    #[test]
    fn an_executable_substitute_answers_alone() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(entry("sub", 0.5, 1, None));
        r.register(prune_entry("p", 4, 1.0, &["a"]));

        match compose(&r.candidates(&and_query(), &graph(), &prices), |_| true) {
            Composed::Substitute(candidate) => {
                assert_eq!(candidate.derived.id, DerivedId("sub".into()));
            }
            _ => panic!("the cheapest executable candidate leads"),
        }
    }

    #[test]
    fn a_substitute_priced_above_a_leading_prune_is_ignored() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(prune_entry("p", 4, 0.5, &["a"]));
        r.register(entry("sub", 5.0, 1, None));

        match compose(&r.candidates(&and_query(), &graph(), &prices), |_| true) {
            Composed::Pruned { files, pieces, .. } => {
                assert_eq!(files, BTreeMap::from([(FileId("a".into()), Scope::Whole)]));
                assert_eq!(pieces.len(), 1);
            }
            _ => panic!("the cheaper prune leads; stored rows never get asked"),
        }
    }

    #[test]
    fn a_non_executable_substitute_falls_through_to_prunes() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(entry("no-rows", 0.5, 1, None));
        r.register(prune_entry("p", 4, 1.0, &["a", "b"]));

        match compose(&r.candidates(&and_query(), &graph(), &prices), |_| false) {
            Composed::Pruned { pieces, .. } => {
                assert_eq!(pieces.len(), 1);
                assert_eq!(pieces[0].derived.id, DerivedId("p".into()));
            }
            _ => panic!("a substitute without rows is skipped, not leading"),
        }
    }

    #[test]
    fn nothing_admissible_is_a_full_scan() {
        assert!(matches!(compose(&[], |_| true), Composed::Scan));
    }

    #[test]
    fn scopes_intersect_within_a_shared_file() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(scoped_entry(
            "a_idx",
            4,
            1.0,
            &[("a", &[0, 1]), ("b", &[0])],
        ));
        r.register(scoped_entry("b_idx", 9, 2.0, &[("a", &[1, 2])]));

        match compose(&r.candidates(&and_query(), &graph(), &prices), |_| true) {
            Composed::Pruned { files, pieces, .. } => {
                assert_eq!(
                    files,
                    BTreeMap::from([(FileId("a".into()), Scope::Groups(BTreeSet::from([1])))]),
                    "b is dropped — no index names it — and a keeps only group 1"
                );
                assert_eq!(pieces.len(), 2);
            }
            _ => panic!("expected a pruned plan"),
        }
    }

    #[test]
    fn disjoint_scopes_drop_the_file() {
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(scoped_entry("a_idx", 4, 1.0, &[("a", &[0])]));
        r.register(scoped_entry("b_idx", 9, 2.0, &[("a", &[1])]));

        match compose(&r.candidates(&and_query(), &graph(), &prices), |_| true) {
            Composed::Pruned { files, .. } => {
                assert!(files.is_empty(), "no group satisfies both, so no file can");
            }
            _ => panic!("expected a pruned plan"),
        }
    }

    #[test]
    fn row_masks_intersect_within_a_shared_group() {
        // One bitmap names rows {0,1} of a's group 0, the other {1,2} —
        // the conjunct holds only at row 1.
        let prices = PriceTable::default();
        let mut r = Registry::new();
        r.register(row_entry("a_idx", 4, 1.0, &[("a", &[(0, &[0, 1])])]));
        r.register(row_entry("b_idx", 9, 2.0, &[("a", &[(0, &[1, 2])])]));

        match compose(&r.candidates(&and_query(), &graph(), &prices), |_| true) {
            Composed::Pruned { files, pieces, .. } => {
                assert_eq!(
                    files,
                    BTreeMap::from([(
                        FileId("a".into()),
                        Scope::Rows(BTreeMap::from([(0, BTreeSet::from([1]))])),
                    )]),
                    "only row 1 satisfies both"
                );
                assert_eq!(pieces.len(), 2, "both narrowed");
            }
            _ => panic!("expected a pruned plan"),
        }
    }
}
