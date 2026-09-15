//! An inverted index over one text field: which files can contain which
//! terms.
//!
//! Structurally [`Index`](super::Index)'s twin — postings from a hash to a
//! file set, pruning rather than substituting — and different in exactly one
//! way that matters: it **intersects** where the equality index unions.
//!
//! ```text
//! Index        a = 1 OR a = 2   alternatives   union the postings
//! TextIndex    'error timeout'  conjunction    intersect the postings
//! ```
//!
//! That follows from what the predicate means, not from a choice made here:
//! a document matches `error timeout` only if it holds both words, so a file
//! that can match must hold both, so a file absent from either posting list
//! cannot contain a matching row.
//!
//! Pruning, so over-selection is merely slow — the engine still reads real
//! rows and re-applies `quarry_matches` above the scan, which is what makes
//! a term-level index safe despite knowing nothing about row-level
//! positions.

use std::collections::{BTreeMap, BTreeSet};

use crate::cost::{Cost, PriceTable};
use crate::derived::{FieldId, Kind, Query, Refreshed, Rewrite};
use crate::snapshot::{Diff, FileId};

/// Which files contain which terms of one text field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextIndex {
    field: FieldId,
    postings: BTreeMap<u64, BTreeSet<FileId>>,
    bytes: u64,
}

impl TextIndex {
    /// A text index over `field` with no entries.
    pub fn new(field: FieldId) -> Self {
        TextIndex {
            field,
            postings: BTreeMap::new(),
            bytes: 0,
        }
    }

    /// Record that `file` contains a row whose text holds `term`.
    ///
    /// `term` is a hash from [`terms`](crate::engine::terms) — the tokenizer
    /// a probe uses too, since an index built by any other definition of a
    /// word answers a different question than the one asked.
    pub fn insert(&mut self, term: u64, file: FileId) {
        self.postings.entry(term).or_default().insert(file);
    }

    /// Builder form of [`TextIndex::insert`].
    pub fn with(mut self, term: u64, file: FileId) -> Self {
        self.insert(term, file);
        self
    }

    /// Declare how many bytes this index occupies.
    pub fn with_bytes(mut self, bytes: u64) -> Self {
        self.bytes = bytes;
        self
    }

    /// The field indexed.
    pub fn field(&self) -> FieldId {
        self.field
    }

    /// How many distinct terms are indexed.
    pub fn values(&self) -> usize {
        self.postings.len()
    }

    /// Files recorded as containing `term`.
    pub fn files_for(&self, term: u64) -> Option<&BTreeSet<FileId>> {
        self.postings.get(&term)
    }

    /// How many bytes this index was declared to occupy.
    pub fn bytes_estimate(&self) -> u64 {
        self.bytes
    }

    /// Exactly how many bytes this index occupies when written down.
    ///
    /// The same accounting [`Index::encoded_len`](super::Index::encoded_len)
    /// does, and for the same reason: it is what a storage budget is enforced
    /// against, so it must be computed rather than estimated and must agree
    /// with what a recovered index reports.
    ///
    /// File paths are written once and referred to by number, because a path
    /// is 60–100 bytes and a term index repeats the same handful of them
    /// across every posting.
    pub fn encoded_len(&self) -> u64 {
        let paths: BTreeSet<&FileId> = self.postings.values().flatten().collect();

        let mut total = 4u64;
        for file in &paths {
            total += 4 + file.0.len() as u64;
        }
        total += 8; // the number of terms
        for files in self.postings.values() {
            total += 8 + 4; // the hashed term, and how many files hold it
            total += 4 * files.len() as u64;
        }
        total
    }

    /// Every term and the files holding it, in a fixed order.
    ///
    /// Ordered because it is written to storage: the bytes are a function of
    /// the contents alone, so the same index encodes identically every time.
    pub fn postings(&self) -> impl Iterator<Item = (u64, &BTreeSet<FileId>)> {
        self.postings.iter().map(|(term, files)| (*term, files))
    }
}

impl Kind for TextIndex {
    fn name(&self) -> &'static str {
        "text"
    }

    /// Prunes to the files that can contain **every** queried term.
    ///
    /// The intersection, because the predicate conjoins: a file missing one
    /// term cannot hold a row holding all of them. A term with no postings
    /// therefore prunes to nothing, which is a useful answer rather than a
    /// failure — no file at this snapshot contains that word, and the rule
    /// still adds any files appended since.
    ///
    /// A query that does not match this field contributes no terms and does
    /// not match at all: better to scan than to prune on a predicate whose
    /// meaning this index does not know.
    fn matches(&self, query: &Query) -> Option<Rewrite> {
        let terms = query.match_terms(self.field);
        if terms.is_empty() {
            return None;
        }
        let mut files: Option<BTreeSet<FileId>> = None;
        for term in terms {
            let holding = self.postings.get(&term).cloned().unwrap_or_default();
            files = Some(match files {
                None => holding,
                Some(sofar) => sofar.intersection(&holding).cloned().collect(),
            });
        }
        Some(Rewrite::prune(files.unwrap_or_default()))
    }

    fn cost(&self, prices: &PriceTable) -> Cost {
        super::price_local(prices, self.bytes)
    }

    /// Extending the index means reading the files that were added, which
    /// this type cannot do, so any change reports
    /// [`Refreshed::NeedsRebuild`].
    ///
    /// Less pressing than it sounds, exactly as for an equality index: a
    /// stale text index stays *usable*, because the rule unions the added
    /// files. Refresh is efficiency here, not correctness.
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
    use crate::derived::{
        Decision, Derived, DerivedId, PolicyFingerprint, Predicate, Reason, Scope, Source,
    };
    use crate::snapshot::{DeleteState, Snapshot, SnapshotGraph, SnapshotId, TableId};

    const POLICY: PolicyFingerprint = PolicyFingerprint(1);
    const BODY: FieldId = 7;

    // Terms as opaque hashes; the tokenizer's own behaviour is tested where
    // it lives, in `engine::text`.
    const ERROR: u64 = 100;
    const TIMEOUT: u64 = 200;
    const RETRY: u64 = 300;

    fn f(name: &str) -> FileId {
        FileId(name.to_owned())
    }

    /// `error` in a and b, `timeout` in b and c, `retry` in a.
    fn index() -> TextIndex {
        TextIndex::new(BODY)
            .with(ERROR, f("a"))
            .with(ERROR, f("b"))
            .with(TIMEOUT, f("b"))
            .with(TIMEOUT, f("c"))
            .with(RETRY, f("a"))
            .with_bytes(4096)
    }

    fn query_with(predicates: Vec<Predicate>) -> Query {
        Query {
            table: TableId("logs".into()),
            snapshot: SnapshotId(810),
            policy: POLICY,
            plan_hash: 1,
            plan: None,
            projected: BTreeSet::from([BODY]),
            predicates,
            aggregate: None,
            nearest: None,
            approximate: false,
        }
    }

    fn matching(field: FieldId, terms: &[u64]) -> Predicate {
        Predicate::Matches {
            field,
            terms: terms.to_vec(),
        }
    }

    fn pruned(files: &[&str]) -> Option<Rewrite> {
        Some(Rewrite::Prune {
            files: files.iter().map(|name| (f(name), Scope::Whole)).collect(),
        })
    }

    #[test]
    fn one_term_prunes_to_the_files_holding_it() {
        assert_eq!(
            index().matches(&query_with(vec![matching(BODY, &[ERROR])])),
            pruned(&["a", "b"])
        );
    }

    /// The one way this differs from an equality index, and the reason it is
    /// a separate kind rather than a flag.
    #[test]
    fn several_terms_intersect_rather_than_union() {
        assert_eq!(
            index().matches(&query_with(vec![matching(BODY, &[ERROR, TIMEOUT])])),
            pruned(&["b"]),
            "only b holds both words"
        );
    }

    #[test]
    fn two_match_predicates_on_one_field_also_intersect() {
        // Two conjuncts, which must behave as one conjunction.
        assert_eq!(
            index().matches(&query_with(vec![
                matching(BODY, &[ERROR]),
                matching(BODY, &[TIMEOUT]),
            ])),
            pruned(&["b"])
        );
    }

    #[test]
    fn terms_with_no_common_file_prune_to_nothing() {
        assert_eq!(
            index().matches(&query_with(vec![matching(BODY, &[RETRY, TIMEOUT])])),
            pruned(&[]),
            "retry is only in a, timeout only in b and c"
        );
    }

    #[test]
    fn an_absent_term_prunes_to_nothing() {
        // Not a failure to match: no file at this snapshot holds that word.
        assert_eq!(
            index().matches(&query_with(vec![matching(BODY, &[999])])),
            pruned(&[])
        );
    }

    #[test]
    fn an_unmodelled_predicate_on_the_indexed_field_does_not_match() {
        assert_eq!(
            index().matches(&query_with(vec![Predicate::Opaque { field: BODY }])),
            None,
            "better to scan than to prune on a predicate we cannot interpret"
        );
    }

    #[test]
    fn an_equality_on_the_indexed_field_does_not_match() {
        // A text index knows words, not whole values.
        assert_eq!(
            index().matches(&query_with(vec![Predicate::Eq {
                field: BODY,
                value: ERROR
            }])),
            None
        );
    }

    #[test]
    fn a_match_on_another_field_does_not_match() {
        assert_eq!(
            index().matches(&query_with(vec![matching(99, &[ERROR])])),
            None
        );
    }

    #[test]
    fn no_predicates_do_not_match() {
        assert_eq!(index().matches(&query_with(vec![])), None);
    }

    #[test]
    fn cost_scales_with_index_size() {
        let prices = PriceTable::default();
        let small = TextIndex::new(BODY).with_bytes(1_000).cost(&prices);
        let large = TextIndex::new(BODY).with_bytes(1_000_000).cost(&prices);
        assert!(large.usd > small.usd);
    }

    #[test]
    fn refresh_reports_up_to_date_only_when_nothing_changed() {
        let mut i = index();
        assert_eq!(i.refresh(&Diff::default()), Refreshed::UpToDate);
        let appended = Diff {
            added: BTreeSet::from([f("new")]),
            ..Diff::default()
        };
        assert_eq!(i.refresh(&appended), Refreshed::NeedsRebuild);
    }

    #[test]
    fn accessors_report_what_was_built() {
        let i = index();
        assert_eq!(i.field(), BODY);
        assert_eq!(i.values(), 3);
        assert_eq!(i.files_for(ERROR), Some(&BTreeSet::from([f("a"), f("b")])));
        assert_eq!(i.files_for(999), None);
        assert_eq!(i.name(), "text");
        assert!(i.encoded_len() > 0);
    }

    fn derived_index(at: SnapshotId) -> Derived {
        Derived::new(
            DerivedId("txt".into()),
            Source {
                table: TableId("logs".into()),
                snapshot: at,
            },
            POLICY,
            4096,
            Box::new(index()),
        )
    }

    #[test]
    fn a_stale_text_index_is_usable_with_the_files_added_since() {
        let g = SnapshotGraph::new()
            .with(
                Snapshot::root(SnapshotId(810))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b"))
                    .with_clean_file(f("c")),
            )
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b"))
                    .with_clean_file(f("c"))
                    .with_clean_file(f("d")),
            );

        let mut q = query_with(vec![matching(BODY, &[ERROR])]);
        q.snapshot = SnapshotId(811);

        assert_eq!(
            derived_index(SnapshotId(810)).may_serve(&q, &g),
            Decision::UseWith {
                rewrite: Rewrite::Prune {
                    files: BTreeMap::from([(f("a"), Scope::Whole), (f("b"), Scope::Whole)])
                },
                also_scan: BTreeSet::from([f("d")]),
            },
            "the index has never read d; scanning it is what keeps rows from being lost"
        );
    }

    #[test]
    fn a_delete_leaves_a_pruning_text_index_usable() {
        let g = SnapshotGraph::new()
            .with(Snapshot::root(SnapshotId(810)).with_clean_file(f("a")))
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                    .with_file(f("a"), DeleteState(7)),
            );

        let mut q = query_with(vec![matching(BODY, &[ERROR])]);
        q.snapshot = SnapshotId(811);

        let decision = derived_index(SnapshotId(810)).may_serve(&q, &g);
        assert_ne!(decision, Decision::Reject(Reason::SubtractiveChange));
        assert!(
            decision.is_admitted(),
            "the engine still reads the file and applies deletes as it goes"
        );
    }
}
