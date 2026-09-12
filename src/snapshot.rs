//! Table identity, snapshots, and what changed between them.
//!
//! Two questions the rule in [`crate::derived`] needs answered, and both are
//! easy to get subtly wrong:
//!
//! 1. **Lineage.** Is the snapshot being queried a descendant of the one some
//!    derived state was built from? A rollback produces a *sibling*, not a
//!    descendant, so derived state built on the abandoned branch must not be
//!    used even though its snapshot id is older.
//!
//! 2. **What changed.** A commit that only adds delete files changes which
//!    rows are live while adding and removing no data file at all. Diffing
//!    file sets alone misses it, and the result is wrong answers rather than a
//!    missed optimization.

use std::collections::{BTreeMap, BTreeSet};

/// Identifies a table, independent of its name.
///
/// Keyed on identity rather than name so that renaming a table does not
/// orphan the derived state built from it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableId(pub String);

/// Identifies a snapshot within a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotId(pub i64);

/// Identifies a data file within a table.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub String);

/// An opaque fingerprint of the deletes that apply to one data file.
///
/// Quarry does not model delete files themselves; it only needs to know
/// whether the set applying to a data file has *changed*. Any value that
/// changes when the deletes change will do — in a real catalog this is derived
/// from the delete-file list or deletion-vector state for the file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeleteState(pub u64);

impl DeleteState {
    /// No rows in this file are deleted.
    pub const NONE: DeleteState = DeleteState(0);
}

/// One committed state of a table: which data files are live, and what is
/// deleted from each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    id: SnapshotId,
    parent: Option<SnapshotId>,
    files: BTreeMap<FileId, DeleteState>,
}

impl Snapshot {
    /// A snapshot with no parent: the first commit, or the root of a branch.
    pub fn root(id: SnapshotId) -> Self {
        Snapshot {
            id,
            parent: None,
            files: BTreeMap::new(),
        }
    }

    /// A snapshot descending from `parent`.
    pub fn child_of(id: SnapshotId, parent: SnapshotId) -> Self {
        Snapshot {
            id,
            parent: Some(parent),
            files: BTreeMap::new(),
        }
    }

    /// Add a live data file with the given deletes applied.
    pub fn with_file(mut self, file: FileId, deletes: DeleteState) -> Self {
        self.files.insert(file, deletes);
        self
    }

    /// Add a live data file with no deletes applied.
    pub fn with_clean_file(self, file: FileId) -> Self {
        self.with_file(file, DeleteState::NONE)
    }

    /// This snapshot's id.
    pub fn id(&self) -> SnapshotId {
        self.id
    }

    /// The snapshot this one descends from, if any.
    pub fn parent(&self) -> Option<SnapshotId> {
        self.parent
    }

    /// The live data files and their delete state.
    pub fn files(&self) -> &BTreeMap<FileId, DeleteState> {
        &self.files
    }
}

/// What changed between two snapshots, classified by whether derived state
/// built from the earlier one can be repaired by reading more data.
///
/// The distinction is the whole point of this type:
///
/// - [`Diff::added`] is **additive**. Derived state that missed these files is
///   still correct as far as it goes, so scanning them alongside it produces
///   the right answer.
/// - [`Diff::removed`] and [`Diff::redeleted`] are **subtractive**. Derived
///   state may contain rows that are no longer live, and no amount of extra
///   reading removes them. Such derived state cannot be used, only rebuilt.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Diff {
    /// Files live in the later snapshot but not the earlier one.
    pub added: BTreeSet<FileId>,
    /// Files live in the earlier snapshot but not the later one.
    pub removed: BTreeSet<FileId>,
    /// Files live in both, whose deletes changed.
    pub redeleted: BTreeSet<FileId>,
}

impl Diff {
    /// Whether nothing changed at all.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.redeleted.is_empty()
    }

    /// Whether every change only *adds* rows.
    ///
    /// When true, derived state built from the earlier snapshot can serve a
    /// query at the later one by being unioned with a scan of
    /// [`Diff::added`]. When false, it cannot be used at all.
    pub fn is_purely_additive(&self) -> bool {
        self.removed.is_empty() && self.redeleted.is_empty()
    }
}

/// The snapshots of one table and how they descend from each other.
#[derive(Clone, Debug, Default)]
pub struct SnapshotGraph {
    snapshots: BTreeMap<SnapshotId, Snapshot>,
}

impl SnapshotGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a snapshot, replacing any existing one with the same id.
    pub fn insert(&mut self, snapshot: Snapshot) {
        self.snapshots.insert(snapshot.id(), snapshot);
    }

    /// Builder form of [`SnapshotGraph::insert`].
    pub fn with(mut self, snapshot: Snapshot) -> Self {
        self.insert(snapshot);
        self
    }

    /// Look up a snapshot.
    pub fn get(&self, id: SnapshotId) -> Option<&Snapshot> {
        self.snapshots.get(&id)
    }

    /// Whether `candidate` is `ancestor` itself, or descends from it.
    ///
    /// Walks parent links upward from `candidate`, so a rolled-back branch is
    /// correctly *not* an ancestor of the branch that replaced it. Returns
    /// `false` if either snapshot is unknown, or if parent links form a cycle,
    /// rather than looping or panicking: an unverifiable lineage must not be
    /// reported as valid.
    pub fn is_descendant_or_self(&self, candidate: SnapshotId, ancestor: SnapshotId) -> bool {
        if !self.snapshots.contains_key(&candidate) || !self.snapshots.contains_key(&ancestor) {
            return false;
        }
        let mut at = Some(candidate);
        let mut steps = 0;
        while let Some(id) = at {
            if id == ancestor {
                return true;
            }
            steps += 1;
            if steps > self.snapshots.len() {
                return false; // cycle
            }
            at = self.snapshots.get(&id).and_then(Snapshot::parent);
        }
        false
    }

    /// Classify what changed between two snapshots.
    ///
    /// Compares live file sets *and* the delete state of files present in
    /// both, so a commit that only adds delete files is reported in
    /// [`Diff::redeleted`]. Returns `None` if either snapshot is unknown.
    pub fn diff(&self, from: SnapshotId, to: SnapshotId) -> Option<Diff> {
        let before = self.get(from)?.files();
        let after = self.get(to)?.files();

        let mut diff = Diff::default();
        for (file, deletes) in after {
            match before.get(file) {
                None => {
                    diff.added.insert(file.clone());
                }
                Some(was) if was != deletes => {
                    diff.redeleted.insert(file.clone());
                }
                Some(_) => {}
            }
        }
        for file in before.keys() {
            if !after.contains_key(file) {
                diff.removed.insert(file.clone());
            }
        }
        Some(diff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(name: &str) -> FileId {
        FileId(name.to_owned())
    }

    fn s(id: i64) -> SnapshotId {
        SnapshotId(id)
    }

    /// 810 -> 811 -> 812, with one file added at each step.
    fn linear() -> SnapshotGraph {
        SnapshotGraph::new()
            .with(Snapshot::root(s(810)).with_clean_file(f("a")))
            .with(
                Snapshot::child_of(s(811), s(810))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b")),
            )
            .with(
                Snapshot::child_of(s(812), s(811))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b"))
                    .with_clean_file(f("c")),
            )
    }

    #[test]
    fn a_snapshot_is_its_own_descendant() {
        let g = linear();
        assert!(g.is_descendant_or_self(s(812), s(812)));
    }

    #[test]
    fn descent_follows_parent_links() {
        let g = linear();
        assert!(g.is_descendant_or_self(s(812), s(810)));
        assert!(g.is_descendant_or_self(s(812), s(811)));
        assert!(!g.is_descendant_or_self(s(810), s(812)));
    }

    #[test]
    fn a_rolled_back_branch_is_not_an_ancestor() {
        // 811 is abandoned; 812 is committed against 810 instead.
        let g = SnapshotGraph::new()
            .with(Snapshot::root(s(810)))
            .with(Snapshot::child_of(s(811), s(810)))
            .with(Snapshot::child_of(s(812), s(810)));

        assert!(g.is_descendant_or_self(s(812), s(810)));
        assert!(
            !g.is_descendant_or_self(s(812), s(811)),
            "derived state built on an abandoned branch must not be reused"
        );
        assert!(!g.is_descendant_or_self(s(811), s(812)));
    }

    #[test]
    fn unknown_snapshots_are_never_descendants() {
        let g = linear();
        assert!(!g.is_descendant_or_self(s(999), s(810)));
        assert!(!g.is_descendant_or_self(s(810), s(999)));
    }

    #[test]
    fn a_parent_cycle_does_not_hang() {
        let mut g = SnapshotGraph::new();
        g.insert(Snapshot::child_of(s(1), s(2)));
        g.insert(Snapshot::child_of(s(2), s(1)));
        assert!(!g.is_descendant_or_self(s(1), s(3)));
    }

    #[test]
    fn appending_files_is_purely_additive() {
        let g = linear();
        let diff = g.diff(s(810), s(812)).expect("known snapshots");
        assert_eq!(diff.added, BTreeSet::from([f("b"), f("c")]));
        assert!(diff.removed.is_empty());
        assert!(diff.redeleted.is_empty());
        assert!(diff.is_purely_additive());
    }

    #[test]
    fn a_delete_only_commit_is_detected() {
        // No data file is added or removed: only the deletes on "a" change.
        let g = SnapshotGraph::new()
            .with(Snapshot::root(s(810)).with_clean_file(f("a")))
            .with(Snapshot::child_of(s(811), s(810)).with_file(f("a"), DeleteState(7)));

        let diff = g.diff(s(810), s(811)).expect("known snapshots");
        assert_eq!(
            diff.redeleted,
            BTreeSet::from([f("a")]),
            "a commit that only adds delete files changes which rows are live"
        );
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert!(
            !diff.is_purely_additive(),
            "derived state cannot be repaired by reading more; rows must be removed"
        );
    }

    #[test]
    fn compaction_removes_files_and_is_not_additive() {
        let g = SnapshotGraph::new()
            .with(
                Snapshot::root(s(810))
                    .with_clean_file(f("small-1"))
                    .with_clean_file(f("small-2")),
            )
            .with(Snapshot::child_of(s(811), s(810)).with_clean_file(f("merged")));

        let diff = g.diff(s(810), s(811)).expect("known snapshots");
        assert_eq!(diff.added, BTreeSet::from([f("merged")]));
        assert_eq!(diff.removed, BTreeSet::from([f("small-1"), f("small-2")]));
        assert!(!diff.is_purely_additive());
    }

    #[test]
    fn diffing_a_snapshot_against_itself_is_empty() {
        let g = linear();
        let diff = g.diff(s(812), s(812)).expect("known snapshots");
        assert!(diff.is_empty());
        assert!(diff.is_purely_additive());
    }

    #[test]
    fn diffing_an_unknown_snapshot_yields_nothing() {
        let g = linear();
        assert!(g.diff(s(810), s(999)).is_none());
        assert!(g.diff(s(999), s(810)).is_none());
    }
}
