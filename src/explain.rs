//! What a query will cost, and why.
//!
//! This is the engine's entire introspection surface, and also its entire
//! agent-facing API. `EXPLAIN` *is* `plan()`: it is free, it consults the same
//! rule and the same cost model that execution will, and it reports the answer
//! before anything is spent. An agent that can read this and set a
//! [`Budget`](crate::budget::Budget) needs no separate protocol.
//!
//! Two properties make the output trustworthy, and both are deliberate:
//!
//! - It names what was **not** used and why. A query that is slower than
//!   expected is usually a query whose derived state was refused, and the
//!   reason is the only thing that tells a user what to fix.
//! - It distinguishes what is **known** from what is **estimated**. Byte
//!   counts come from immutable metadata; latency does not, and is therefore
//!   absent rather than guessed.

use std::collections::BTreeSet;
use std::fmt;

use crate::budget::Exceeded;
use crate::cost::{Cost, PriceTable};
use crate::derived::{Decision, DerivedId, Query, Reason, Rewrite};
use crate::registry::Registry;
use crate::snapshot::{FileId, SnapshotGraph, SnapshotId};

/// How much of the authoritative data the answer accounts for.
///
/// Under the current kinds this is always [`Coverage::Exact`] unless a budget
/// stopped the query: the rule refuses any derived state that would produce a
/// wrong answer, so exactness is an invariant rather than a hope. Approximate
/// kinds — samples, sketches — would add a `Bounded` variant carrying a
/// confidence interval, and that is the point of naming it here now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coverage {
    /// Every live row at the queried snapshot is accounted for.
    Exact,
    /// Stopped at a budget ceiling. Rows are missing, and no bound can be put
    /// on how many: the ceiling fell wherever the scan happened to be.
    Incomplete(Exceeded),
}

impl Coverage {
    /// Whether the answer accounts for everything.
    pub fn is_exact(&self) -> bool {
        matches!(self, Coverage::Exact)
    }
}

/// Derived state the plan will use.
#[derive(Clone, Debug, PartialEq)]
pub struct Used {
    /// Which piece.
    pub id: DerivedId,
    /// Its kind, for display.
    pub kind: &'static str,
    /// The snapshot it was built from.
    pub built_at: SnapshotId,
    /// Whether it prunes the scan or replaces it.
    pub rewrite: Rewrite,
}

/// Derived state the plan will not use, and why not.
#[derive(Clone, Debug, PartialEq)]
pub struct Refused {
    /// Which piece.
    pub id: DerivedId,
    /// Its kind, for display.
    pub kind: &'static str,
    /// The first condition of the rule that it failed.
    pub reason: Reason,
}

/// What a query will do.
#[derive(Clone, Debug, PartialEq)]
pub struct Explain {
    /// The snapshot being read.
    pub snapshot: SnapshotId,
    /// The derived state chosen, if any. `None` means a full scan.
    pub used: Option<Used>,
    /// Files that must be read alongside the derived state, because they were
    /// added after it was built.
    pub also_scan: BTreeSet<FileId>,
    /// Derived state that was considered and refused.
    pub refused: Vec<Refused>,
    /// What using the chosen derived state costs. Bytes are known; time is
    /// not modelled and so is not reported.
    pub cost: Cost,
    /// How much of the data the answer accounts for.
    pub coverage: Coverage,
}

impl Explain {
    /// Plan `query` without executing it.
    ///
    /// Free by construction: it reads the registry and the snapshot graph, and
    /// touches no data.
    pub fn plan(
        query: &Query,
        registry: &Registry,
        graph: &SnapshotGraph,
        prices: &PriceTable,
    ) -> Explain {
        // Only admitted candidates come back from `best`, so a rejection here
        // is impossible; it is treated as "no candidate" rather than a panic,
        // since a full scan is always a correct answer.
        let chosen = registry
            .best(query, graph, prices)
            .and_then(|c| match c.decision {
                Decision::Use(rewrite) => Some((c.derived, rewrite, BTreeSet::new(), c.cost)),
                Decision::UseWith { rewrite, also_scan } => {
                    Some((c.derived, rewrite, also_scan, c.cost))
                }
                Decision::Reject(_) => None,
            });

        let refused = registry
            .assess(query, graph)
            .into_iter()
            .filter_map(|(derived, decision)| match decision {
                Decision::Reject(reason) => Some(Refused {
                    id: derived.id.clone(),
                    kind: derived.kind().name(),
                    reason,
                }),
                _ => None,
            })
            .collect();

        match chosen {
            None => Explain {
                snapshot: query.snapshot,
                used: None,
                also_scan: BTreeSet::new(),
                refused,
                cost: Cost::ZERO,
                coverage: Coverage::Exact,
            },
            Some((derived, rewrite, also_scan, cost)) => Explain {
                snapshot: query.snapshot,
                used: Some(Used {
                    id: derived.id.clone(),
                    kind: derived.kind().name(),
                    built_at: derived.source.snapshot,
                    rewrite,
                }),
                also_scan,
                refused,
                cost,
                coverage: Coverage::Exact,
            },
        }
    }

    /// Record that execution stopped at a budget ceiling.
    ///
    /// Downgrades coverage, because a query that was cut short has rows
    /// missing and the gap cannot be bounded.
    pub fn aborted(mut self, exceeded: Exceeded) -> Self {
        self.coverage = Coverage::Incomplete(exceeded);
        self
    }

    /// Whether the plan reads the table directly.
    pub fn is_full_scan(&self) -> bool {
        self.used.is_none()
    }
}

impl fmt::Display for Explain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "snapshot:  {}", self.snapshot.0)?;
        match &self.used {
            None => writeln!(f, "used:      none (full scan)")?,
            Some(used) => {
                let how = match used.rewrite {
                    Rewrite::Prune { .. } => "prunes",
                    Rewrite::Substitute { .. } => "substitutes",
                };
                writeln!(
                    f,
                    "used:      {} ({}, {}, built at {})",
                    used.id.0, used.kind, how, used.built_at.0
                )?;
            }
        }
        if !self.also_scan.is_empty() {
            writeln!(f, "also scan: {} file(s) added since", self.also_scan.len())?;
        }
        writeln!(f, "bytes:     {} (known)", self.cost.bytes)?;
        // Waiting is shown separately because it is the part a caller can act
        // on by moving work closer or reading in fewer, larger pieces, and
        // because in the same region it is the *only* thing distance changes.
        writeln!(f, "waiting:   {:.4}s", self.cost.wait_seconds)?;
        writeln!(f, "cost:      ${:.8}", self.cost.usd)?;
        match self.coverage {
            Coverage::Exact => writeln!(f, "coverage:  exact")?,
            Coverage::Incomplete(exceeded) => {
                writeln!(f, "coverage:  INCOMPLETE, stopped at {exceeded:?}")?
            }
        }
        for refused in &self.refused {
            writeln!(
                f,
                "refused:   {} ({}) - {:?}",
                refused.id.0, refused.kind, refused.reason
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Exceeded;
    use crate::derived::{Derived, PolicyFingerprint, Predicate, Source};
    use crate::kinds::{Index, ResultCache};
    use crate::snapshot::{DeleteState, Snapshot, TableId};

    const POLICY: PolicyFingerprint = PolicyFingerprint(1);
    const TENANT: u32 = 4;

    fn f(name: &str) -> FileId {
        FileId(name.to_owned())
    }

    fn table() -> TableId {
        TableId("events".into())
    }

    fn source(at: i64) -> Source {
        Source {
            table: table(),
            snapshot: SnapshotId(at),
        }
    }

    fn query(at: i64) -> Query {
        Query {
            table: table(),
            snapshot: SnapshotId(at),
            policy: POLICY,
            plan_hash: 42,
            plan: Some(the_plan()),
            projected: BTreeSet::from([TENANT]),
            predicates: vec![Predicate::Eq {
                field: TENANT,
                value: 100,
            }],
        }
    }

    /// The one plan these tests use, so a result cache can match a query.
    fn the_plan() -> crate::derived::Plan {
        crate::derived::Plan::new(BTreeSet::from([TENANT]), ["tenant = 100".to_owned()])
    }

    fn index_derived(at: i64) -> Derived {
        Derived::new(
            DerivedId("idx".into()),
            source(at),
            POLICY,
            4096,
            Box::new(Index::new(TENANT).with(100, f("a")).with_bytes(4096)),
        )
    }

    fn result_derived(at: i64) -> Derived {
        Derived::new(
            DerivedId("res".into()),
            source(at),
            POLICY,
            64,
            Box::new(ResultCache::rows_of(the_plan(), 3, 64)),
        )
    }

    fn graph_with_append() -> SnapshotGraph {
        SnapshotGraph::new()
            .with(Snapshot::root(SnapshotId(810)).with_clean_file(f("a")))
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b")),
            )
    }

    #[test]
    fn an_empty_registry_explains_a_full_scan() {
        let e = Explain::plan(
            &query(810),
            &Registry::new(),
            &graph_with_append(),
            &PriceTable::default(),
        );
        assert!(e.is_full_scan());
        assert!(e.refused.is_empty());
        assert!(e.coverage.is_exact());
        assert!(e.to_string().contains("full scan"));
    }

    #[test]
    fn the_chosen_derived_state_is_named() {
        let mut r = Registry::new();
        r.register(result_derived(810));

        let e = Explain::plan(
            &query(810),
            &r,
            &graph_with_append(),
            &PriceTable::default(),
        );
        let used = e.used.as_ref().expect("something was used");
        assert_eq!(used.id, DerivedId("res".into()));
        assert_eq!(used.kind, "result");
        assert_eq!(used.rewrite, Rewrite::Substitute { unionable: true });
        assert_eq!(e.cost.bytes, 64);
        assert!(e.to_string().contains("substitutes"));
    }

    #[test]
    fn files_added_since_are_reported() {
        let mut r = Registry::new();
        r.register(index_derived(810));

        let e = Explain::plan(
            &query(811),
            &r,
            &graph_with_append(),
            &PriceTable::default(),
        );
        assert_eq!(e.also_scan, BTreeSet::from([f("b")]));
        assert!(e.to_string().contains("1 file(s) added since"));
    }

    #[test]
    fn a_refusal_is_reported_with_its_reason() {
        // A delete-only commit: the result cache is refused, the index is not.
        let graph = SnapshotGraph::new()
            .with(Snapshot::root(SnapshotId(810)).with_clean_file(f("a")))
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                    .with_file(f("a"), DeleteState(7)),
            );

        let mut r = Registry::new();
        r.register(result_derived(810));
        r.register(index_derived(810));

        let e = Explain::plan(&query(811), &r, &graph, &PriceTable::default());

        assert_eq!(
            e.used.as_ref().map(|u| u.kind),
            Some("index"),
            "the pruning kind survives the delete"
        );
        assert_eq!(
            e.refused,
            vec![Refused {
                id: DerivedId("res".into()),
                kind: "result",
                reason: Reason::SubtractiveChange,
            }]
        );

        let text = e.to_string();
        assert!(text.contains("refused:"));
        assert!(text.contains("SubtractiveChange"));
    }

    #[test]
    fn everything_refused_explains_a_full_scan_and_says_why() {
        let graph = SnapshotGraph::new()
            .with(Snapshot::root(SnapshotId(810)))
            .with(Snapshot::child_of(SnapshotId(811), SnapshotId(810)))
            .with(Snapshot::child_of(SnapshotId(812), SnapshotId(810)));

        let mut r = Registry::new();
        r.register(index_derived(811)); // abandoned branch

        let e = Explain::plan(&query(812), &r, &graph, &PriceTable::default());
        assert!(e.is_full_scan());
        assert_eq!(e.refused.len(), 1);
        assert_eq!(e.refused[0].reason, Reason::NotDescendant);
    }

    #[test]
    fn planning_is_free_and_repeatable() {
        let mut r = Registry::new();
        r.register(index_derived(810));
        r.register(result_derived(810));
        let graph = graph_with_append();
        let prices = PriceTable::default();

        let first = Explain::plan(&query(810), &r, &graph, &prices);
        let second = Explain::plan(&query(810), &r, &graph, &prices);
        assert_eq!(first, second, "planning must not depend on hidden state");
        assert_eq!(
            r.get(&DerivedId("idx".into())).expect("present").uses(),
            0,
            "planning must not count as a use"
        );
    }

    #[test]
    fn an_abort_downgrades_coverage() {
        let e = Explain::plan(
            &query(810),
            &Registry::new(),
            &graph_with_append(),
            &PriceTable::default(),
        )
        .aborted(Exceeded::Bytes {
            limit: 100,
            spent: 200,
        });

        assert!(!e.coverage.is_exact());
        assert!(e.to_string().contains("INCOMPLETE"));
    }

    #[test]
    fn the_cheapest_candidate_is_the_one_explained() {
        let mut r = Registry::new();
        r.register(index_derived(810)); // 4096 bytes
        r.register(result_derived(810)); // 64 bytes, so cheaper

        let e = Explain::plan(
            &query(810),
            &r,
            &graph_with_append(),
            &PriceTable::default(),
        );
        assert_eq!(e.used.expect("used").id, DerivedId("res".into()));
        assert!(
            e.refused.is_empty(),
            "a candidate that lost on cost was not refused"
        );
    }
}
