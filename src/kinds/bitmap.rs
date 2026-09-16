//! A bitmap index over one field: which rows can hold which values.
//!
//! [`Index`](super::Index)'s finer-grained counterpart — its postings name
//! files, a bitmap's name positions inside them. Most useful on
//! low-cardinality fields, where per-row postings stay small and a lookup
//! skips whole row groups and pages rather than just objects.
//!
//! The safety argument is unchanged: the rewrite only narrows what is read,
//! never what is answered, so a bitmap that names too many rows is slow and
//! one that names too few is prevented — the rule unions in files added
//! since the build, and in-file deletes are still applied above the scan.

use std::collections::{BTreeMap, BTreeSet};

use crate::cost::{Cost, PriceTable};
use crate::derived::{FieldId, Kind, Query, Refreshed, Rewrite, Scope};
use crate::snapshot::{Diff, FileId};

/// Which rows of which files contain which values of one field.
///
/// Postings are group-local — `value → file → group → row offsets inside
/// the group` — because that is the granularity the reader selects at, and
/// because group-local offsets compose with a [`Scope::Groups`] without
/// knowing where each group begins in the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitmap {
    field: FieldId,
    postings: BTreeMap<u64, BTreeMap<FileId, BTreeMap<u32, BTreeSet<u64>>>>,
    bytes: u64,
}

impl Bitmap {
    /// A bitmap over `field` with no entries.
    pub fn new(field: FieldId) -> Self {
        Bitmap {
            field,
            postings: BTreeMap::new(),
            bytes: 0,
        }
    }

    /// Record that `file`'s `group` holds `field = value` at `row`.
    pub fn insert(&mut self, value: u64, file: FileId, group: u32, row: u64) {
        self.postings
            .entry(value)
            .or_default()
            .entry(file)
            .or_default()
            .entry(group)
            .or_default()
            .insert(row);
    }

    /// Builder form of [`Bitmap::insert`].
    pub fn with(mut self, value: u64, file: FileId, group: u32, row: u64) -> Self {
        self.insert(value, file, group, row);
        self
    }

    /// Declare how many bytes this bitmap occupies.
    pub fn with_bytes(mut self, bytes: u64) -> Self {
        self.bytes = bytes;
        self
    }

    /// The field indexed.
    pub fn field(&self) -> FieldId {
        self.field
    }

    /// Every hashed value and its `file → group → rows` postings, in a
    /// fixed order — written to storage, so the bytes are a function of the
    /// contents alone.
    pub fn postings(
        &self,
    ) -> impl Iterator<Item = (u64, &BTreeMap<FileId, BTreeMap<u32, BTreeSet<u64>>>)> {
        self.postings.iter().map(|(value, files)| (*value, files))
    }

    /// Exactly how many bytes this bitmap occupies when written down.
    ///
    /// The same discipline [`Index::encoded_len`](super::Index::encoded_len)
    /// keeps: computed because a
    /// storage budget is enforced against it, and agreeing with what a
    /// recovered bitmap reports so the same piece is not sized differently
    /// either side of a restart.
    pub fn encoded_len(&self) -> u64 {
        let paths: BTreeSet<&FileId> = self
            .postings
            .values()
            .flat_map(|files| files.keys())
            .collect();

        let mut total = 4u64;
        for file in &paths {
            total += 4 + file.0.len() as u64;
        }

        total += 8; // the number of values
        for files in self.postings.values() {
            total += 8 + 4; // the hashed value, and how many files hold it
            for groups in files.values() {
                total += 4 + 4; // the file, and how many groups hold it
                for rows in groups.values() {
                    total += 4 + 4 + 8 * rows.len() as u64; // group, count, rows
                }
            }
        }
        total
    }

    /// How many bytes this bitmap was declared to occupy.
    pub fn bytes_estimate(&self) -> u64 {
        self.bytes
    }
}

impl Kind for Bitmap {
    fn name(&self) -> &'static str {
        "bitmap"
    }

    /// Prunes to the rows that can hold any of the queried values, with the
    /// same contract [`Index`](super::Index) keeps: only equality predicates on
    /// the indexed field are probed, alternatives on one field union, and a
    /// value with no postings prunes to nothing rather than failing.
    fn matches(&self, query: &Query) -> Option<Rewrite> {
        let values = query.equalities(self.field);
        if values.is_empty() {
            return None;
        }
        let mut files: BTreeMap<FileId, BTreeMap<u32, BTreeSet<u64>>> = BTreeMap::new();
        for value in values {
            let Some(posting) = self.postings.get(&value) else {
                continue;
            };
            for (file, groups) in posting {
                let entry = files.entry(file.clone()).or_default();
                for (group, rows) in groups {
                    entry
                        .entry(*group)
                        .or_default()
                        .extend(rows.iter().cloned());
                }
            }
        }
        Some(Rewrite::Prune {
            files: files
                .into_iter()
                .map(|(file, rows)| (file, Scope::Rows(rows)))
                .collect(),
        })
    }

    fn cost(&self, prices: &PriceTable) -> Cost {
        super::price_local(prices, self.bytes)
    }

    /// Row positions cannot be extended without reading the new files, so
    /// any change reports [`Refreshed::NeedsRebuild`] — the same posture
    /// [`Index`](super::Index) takes, and as little a problem: a stale bitmap
    /// stays usable because the rule reads the added files anyway.
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
    use crate::derived::{Derived, DerivedId, PolicyFingerprint, Predicate, Source};
    use crate::snapshot::{Snapshot, SnapshotGraph, SnapshotId, TableId};

    const POLICY: PolicyFingerprint = PolicyFingerprint(1);
    const STATUS: FieldId = 9;

    fn f(name: &str) -> FileId {
        FileId(name.to_owned())
    }

    /// status 0 at (a, group 0, row 1) and (b, group 0, row 0);
    /// status 1 at (b, group 0, rows 0 and 2).
    fn bitmap() -> Bitmap {
        Bitmap::new(STATUS)
            .with(0, f("a"), 0, 1)
            .with(0, f("b"), 0, 0)
            .with(1, f("b"), 0, 0)
            .with(1, f("b"), 0, 2)
            .with_bytes(256)
    }

    fn query_with(predicates: Vec<Predicate>) -> Query {
        Query {
            table: TableId("events".into()),
            snapshot: SnapshotId(810),
            policy: POLICY,
            plan_hash: 1,
            plan: None,
            projected: BTreeSet::from([STATUS]),
            predicates,
            aggregate: None,
            nearest: None,
            join: None,
            approximate: false,
            stale: false,
        }
    }

    #[test]
    fn an_equality_prunes_to_the_rows_holding_that_value() {
        let got = bitmap().matches(&query_with(vec![Predicate::Eq {
            field: STATUS,
            value: 0,
        }]));
        assert_eq!(
            got,
            Some(Rewrite::Prune {
                files: BTreeMap::from([
                    (
                        f("a"),
                        Scope::Rows(BTreeMap::from([(0, BTreeSet::from([1]))]))
                    ),
                    (
                        f("b"),
                        Scope::Rows(BTreeMap::from([(0, BTreeSet::from([0]))]))
                    ),
                ])
            })
        );
    }

    #[test]
    fn alternatives_on_one_field_union_their_rows() {
        let got = bitmap().matches(&query_with(vec![
            Predicate::Eq {
                field: STATUS,
                value: 0,
            },
            Predicate::Eq {
                field: STATUS,
                value: 1,
            },
        ]));
        assert_eq!(
            got,
            Some(Rewrite::Prune {
                files: BTreeMap::from([
                    (
                        f("a"),
                        Scope::Rows(BTreeMap::from([(0, BTreeSet::from([1]))]))
                    ),
                    (
                        f("b"),
                        Scope::Rows(BTreeMap::from([(0, BTreeSet::from([0, 2]))]))
                    ),
                ])
            }),
            "rows for either value, per group"
        );
    }

    #[test]
    fn an_unmodelled_predicate_does_not_match() {
        assert_eq!(
            bitmap().matches(&query_with(vec![Predicate::Opaque { field: STATUS }])),
            None
        );
    }

    #[test]
    fn a_stale_bitmap_is_usable_with_the_files_added_since() {
        use crate::derived::Decision;

        let g = SnapshotGraph::new()
            .with(Snapshot::root(SnapshotId(810)).with_clean_file(f("a")))
            .with(
                Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("new")),
            );
        let mut q = query_with(vec![Predicate::Eq {
            field: STATUS,
            value: 0,
        }]);
        q.snapshot = SnapshotId(811);

        let decision = Derived::new(
            DerivedId("bm".into()),
            Source {
                table: TableId("events".into()),
                snapshot: SnapshotId(810),
            },
            POLICY,
            256,
            Box::new(bitmap()),
        )
        .may_serve(&q, &g);
        match decision {
            Decision::UseWith { also_scan, .. } => {
                assert_eq!(also_scan, BTreeSet::from([f("new")]));
            }
            other => panic!("expected residual reads, got {other:?}"),
        }
    }
}
