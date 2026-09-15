//! Derived state, and the one rule that decides when it may replace a scan.
//!
//! Derived state is bytes computed from a table at a known snapshot that can
//! answer some queries more cheaply than the table can: a cached result, an
//! index, a projection, a pre-aggregate. All of them are the same thing with
//! different [`Kind`]s, so all of them are admitted by one function,
//! [`Derived::may_serve`].
//!
//! Checking admissibility in exactly one place is what makes a wrong answer
//! prevented by construction rather than by care. A new kind implements three
//! methods and inherits the correctness argument.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::cost::{Cost, PriceTable};
use crate::snapshot::{Diff, FileId, SnapshotGraph, SnapshotId, TableId};

/// Identifies one piece of derived state.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DerivedId(pub String);

/// A hash of the row filters, column masks, and grants that apply to a
/// principal reading the source tables.
///
/// Part of a piece of derived state's identity, because two principals issuing
/// the same query may be entitled to different rows. Reusing one principal's
/// derived state for another with a different effective policy would leak
/// them. Different policy means different derived state, at the cost of a
/// lower hit rate — which is the correct trade, not one to tune away.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PolicyFingerprint(pub u64);

impl PolicyFingerprint {
    /// Hash the components of a principal's effective policy — row filters,
    /// column masks, grants — in a canonical order the caller chooses.
    ///
    /// [`StableHasher`](crate::stable_hash::StableHasher), not
    /// `DefaultHasher`: the fingerprint is written into persisted blobs, and
    /// a hash that changes across restarts or toolchains makes stored derived
    /// state silently match nothing.
    pub fn of(parts: &[&str]) -> Self {
        PolicyFingerprint(crate::stable_hash::StableHasher::of(&parts))
    }
}

/// A field within a table, identified the way Iceberg identifies it.
///
/// Field *ids*, never names: renaming a column must not silently invalidate or
/// mismatch the derived state built from it.
pub type FieldId = u32;

/// A restriction a query places on a field.
///
/// Only equality is modelled, because only equality is currently *probed* by
/// an index. Everything else is [`Predicate::Opaque`], which still marks the
/// field as filtered but cannot be used to prune. Adding ranges later means
/// adding a variant, not changing any signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    /// `field = value`, where `value` is a hash of the literal.
    ///
    /// A hash rather than a value type: an equality index is probed by hash
    /// anyway, and it keeps a value model out of the core for now.
    Eq {
        /// The field restricted.
        field: FieldId,
        /// Hash of the literal compared against.
        value: u64,
    },
    /// A restriction on `field` whose form is not modelled.
    Opaque {
        /// The field restricted.
        field: FieldId,
    },
}

impl Predicate {
    /// The field this restricts.
    pub fn field(&self) -> FieldId {
        match self {
            Predicate::Eq { field, .. } | Predicate::Opaque { field } => *field,
        }
    }
}

/// An exact description of what a query computes.
///
/// # Why this is not built from [`Predicate`]
///
/// `Predicate` is deliberately lossy. It exists to decide which files *might*
/// contain matching rows, where over-approximating is safe, and it throws away
/// everything not needed for that:
///
/// ```text
/// Opaque { field }        a > 5 and a < 3 are indistinguishable
/// Eq { value: u64 }       the literal is a hash, so colliding literals
///                         are indistinguishable
/// only the first column   a complex filter over two columns records one
/// unknown columns         drop out of the projection entirely
/// ```
///
/// Every one of those is harmless for pruning and fatal for deciding that two
/// queries are the *same* query — which is what a substituting kind does
/// before handing one query the other's stored answer. So plan identity is
/// carried separately, exactly, and is absent rather than approximate when
/// the engine cannot render it faithfully.
/// One filter, faithfully rendered and tagged with the field it restricts.
///
/// `text` is the identity — a rendering faithful enough that two different
/// predicates never share one. `field` is what coverage needs: a cube can only
/// serve a query whose extra restrictions sit on fields it grouped by, and
/// that check needs the field, not the text. A filter whose field cannot be
/// determined is `None` — and uncovered by everything.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Filter {
    /// The field this restricts, when it is known.
    pub field: Option<FieldId>,
    /// The faithful rendering.
    pub text: String,
}

/// What a query computes, exactly — or as exactly as the engine can render
/// it. The identity a substitution is allowed to match on.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Plan {
    /// Fields the query reads.
    pub projected: BTreeSet<FieldId>,
    /// One faithful rendering per filter, sorted.
    ///
    /// Sorted so that `a = 1 AND b = 2` and `b = 2 AND a = 1` are one plan;
    /// a conjunction has no meaningful order.
    pub filters: Vec<Filter>,
}

impl From<String> for Filter {
    /// A filter whose field is unknown restricts nothing a cube can check, so
    /// it is `None` — and uncovered by everything.
    fn from(text: String) -> Self {
        Filter { field: None, text }
    }
}

impl From<&str> for Filter {
    fn from(text: &str) -> Self {
        Filter {
            field: None,
            text: text.to_owned(),
        }
    }
}

/// One side of `column op literal`, recovered from a [`Filter`]'s text.
///
/// The canonical rendering writes `column op Literal(...)`, and this parses
/// only exactly that shape — anything else, including a column name that
/// happens to contain an operator, yields `None` and no implication is
/// claimed. A missed implication costs a declined reuse; a wrong one costs
/// a wrong answer, so the parser errs toward saying nothing.
#[derive(Clone, Debug, PartialEq)]
enum Bound {
    /// `column = literal`.
    Eq(Literal),
    /// `column >= literal` or `column > literal`; the bool is "inclusive".
    Lower(Literal, bool),
    /// `column <= literal` or `column < literal`; same.
    Upper(Literal, bool),
}

/// A literal whose ordering is known.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
enum Literal {
    /// Any ordered numeric — integers, floats, dates, and timestamps all
    /// compare on their inner value for the bounds that matter.
    Num(f64),
    /// `Utf8` and friends, compared lexicographically.
    Str(String),
    /// `Boolean`.
    Bool(bool),
}

fn bound_of(filter: &Filter) -> Option<Bound> {
    // `column op literal`: the operator is the first ` op ` in the text,
    // and the tail must be a `Variant(inner)` scalar rendering.
    for (needle, make) in [
        (" >= ", |l| Some(Bound::Lower(l, true))),
        (" <= ", |l| Some(Bound::Upper(l, true))),
        (" > ", |l| Some(Bound::Lower(l, false))),
        (" < ", |l| Some(Bound::Upper(l, false))),
        (" = ", |l| Some(Bound::Eq(l))),
        (" != ", |_| None),
    ] as [(&str, fn(Literal) -> Option<Bound>); 6]
    {
        let Some(at) = filter.text.find(needle) else {
            continue;
        };
        // `!=` is found as ` = `'s neighbour; the explicit arms above keep
        // the equality arm from splitting `a != 1` into `a !` and `= 1`.
        if needle == " = " && filter.text[..at].ends_with('!') {
            return None;
        }
        // The left must be a bare column — `a + 1 = 5` is not `a = 5`, and
        // claiming it is would imply bounds it does not satisfy.
        let column = filter.text[..at].trim();
        if column.is_empty()
            || !column
                .chars()
                .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '$'))
        {
            return None;
        }
        let literal = literal_of(filter.text[at + needle.len()..].trim())?;
        return make(literal);
    }
    None
}

/// `Variant(inner)` — the scalar's `Debug` shape, with the inner parsed for
/// the variants whose ordering is meaningful.
fn literal_of(text: &str) -> Option<Literal> {
    let open = text.find('(')?;
    let variant = &text[..open];
    let inner = text[open + 1..].strip_suffix(')')?;

    // Timestamp renderings carry a timezone after the value; only the
    // value orders.
    let first = inner.split(',').next()?.trim();
    match variant {
        "Int8"
        | "Int16"
        | "Int32"
        | "Int64"
        | "UInt8"
        | "UInt16"
        | "UInt32"
        | "UInt64"
        | "Float16"
        | "Float32"
        | "Float64"
        | "Date32"
        | "Date64"
        | "Time32Second"
        | "Time32Millisecond"
        | "Time64Microsecond"
        | "Time64Nanosecond"
        | "TimestampSecond"
        | "TimestampMillisecond"
        | "TimestampMicrosecond"
        | "TimestampNanosecond" => first.parse().ok().map(Literal::Num),
        "Utf8" | "LargeUtf8" | "Utf8View" => {
            let quoted = inner.strip_prefix('"')?.strip_suffix('"')?;
            Some(Literal::Str(quoted.to_owned()))
        }
        "Boolean" => first.parse().ok().map(Literal::Bool),
        _ => None,
    }
}

impl Filter {
    /// Whether this filter — a query's — implies `baked`, a stored one.
    ///
    /// Identity counts as implication: a filter implies itself. Beyond
    /// that, only the orderable `column op literal` shapes reason — `x >= 3`
    /// implies `x >= 2` and `x = 4` implies `x >= 2` — and fields must
    /// agree, because `a >= 3` says nothing about `b >= 2`.
    pub fn implies(&self, baked: &Filter) -> bool {
        if self == baked {
            return true;
        }
        if self.field.is_none() || self.field != baked.field {
            return false;
        }
        let (Some(want), Some(have)) = (bound_of(baked), bound_of(self)) else {
            return false;
        };
        match (have, want) {
            // An equality implies whatever bound its value satisfies.
            (Bound::Eq(v), Bound::Eq(w)) => v == w,
            (Bound::Eq(v), Bound::Lower(w, inclusive)) => v > w || (inclusive && v == w),
            (Bound::Eq(v), Bound::Upper(w, inclusive)) => v < w || (inclusive && v == w),
            // A lower bound implies a weaker lower bound; the edge case is
            // the strictness — `x >= 3` does not imply `x > 3`.
            (Bound::Lower(v, v_incl), Bound::Lower(w, w_incl)) => {
                v > w || (v == w && (!v_incl || w_incl))
            }
            (Bound::Upper(v, v_incl), Bound::Upper(w, w_incl)) => {
                v < w || (v == w && (!v_incl || w_incl))
            }
            _ => false,
        }
    }
}

impl Plan {
    /// A plan reading `projected` under `filters`, canonicalised.
    pub fn new(
        projected: BTreeSet<FieldId>,
        filters: impl IntoIterator<Item = impl Into<Filter>>,
    ) -> Self {
        let mut filters: Vec<Filter> = filters.into_iter().map(Into::into).collect();
        filters.sort();
        Plan { projected, filters }
    }
}

/// What a query aggregates: the group keys and the measures over them.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Aggregate {
    /// Fields the query groups by.
    pub group_by: BTreeSet<FieldId>,
    /// Aggregations the query computes.
    pub measures: BTreeSet<Measure>,
}

/// One aggregation over one field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Measure {
    /// The aggregation.
    pub func: AggFunc,
    /// The field aggregated. `None` only for `Count` over `*`.
    pub field: Option<FieldId>,
}

/// The aggregations a cube can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AggFunc {
    /// `COUNT`.
    Count,
    /// `SUM`.
    Sum,
    /// `MIN`.
    Min,
    /// `MAX`.
    Max,
}

impl AggFunc {
    /// The function that combines this one's partials.
    ///
    /// Stored counts roll up by summing; everything else combines through
    /// itself.
    pub fn rollup(self) -> AggFunc {
        match self {
            AggFunc::Count => AggFunc::Sum,
            other => other,
        }
    }
}

impl Aggregate {
    /// Whether a stored cube `cube` can answer this query by re-aggregating.
    ///
    /// Two conditions, each exact:
    ///
    /// ```text
    /// keys       the query groups by a subset of the cube's keys, so the
    ///            cube's rows can be rolled up to the coarser grain
    /// measures   every measure the query wants is computable from the
    ///            measures the cube holds
    /// ```
    ///
    /// Filter coverage is the caller's: [`Plan`] comparisons happen where the
    /// two plans are both in hand.
    ///
    /// `approximate` is the session's opt-in: with it, a stored estimate may
    /// answer an exact ask ([`Measure::computable_from`]).
    pub fn covered_by(&self, cube: &Aggregate, approximate: bool) -> bool {
        self.group_by.is_subset(&cube.group_by)
            && self.measures.iter().all(|m| {
                cube.measures
                    .iter()
                    .any(|c| m.computable_from(*c, approximate))
            })
    }
}

impl Measure {
    /// Whether this measure can be computed from a stored one.
    ///
    /// Every function rolls up through itself — partial sums sum, partial
    /// minima minimise. `Avg` is deliberately absent from [`AggFunc`]: a cube
    /// that wants to answer averages stores `Sum` and `Count` and divides.
    ///
    /// `approximate` admits the asymmetric cases: an exact ask answered by a
    /// stored estimate, which only an opted-in session may ask.
    fn computable_from(&self, stored: Measure, _approximate: bool) -> bool {
        self.func == stored.func && self.field == stored.field
    }
}

/// What a query needs, reduced to what the rule and the kinds have to reason
/// about.
///
/// Carries identity ([`Query::plan`]), shape (`projected`, `predicates`), and
/// the context that admissibility depends on (`table`, `snapshot`, `policy`).
/// Identity and shape are separate on purpose: shape may over-approximate,
/// identity may not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    /// The table being read.
    pub table: TableId,
    /// The snapshot being read.
    pub snapshot: SnapshotId,
    /// The effective policy of the principal issuing the query.
    pub policy: PolicyFingerprint,
    /// A short name for the plan, for keying stored bytes.
    ///
    /// **Not** an identity. Two different plans can share a 64-bit hash, and
    /// treating a hash match as a plan match would hand one query another's
    /// answer. Use [`Query::plan`] to decide sameness; use this only to name
    /// something whose identity has already been established.
    pub plan_hash: u64,
    /// What this query computes, when it can be described exactly.
    ///
    /// `None` means the engine could not render the plan faithfully, so no
    /// substituting derived state may claim to answer it. Pruning is
    /// unaffected: it never needs to know that two queries are the same.
    pub plan: Option<Plan>,
    /// Fields the query reads.
    pub projected: BTreeSet<FieldId>,
    /// Restrictions the query places on fields.
    pub predicates: Vec<Predicate>,
    /// What the query aggregates, if it does.
    ///
    /// Never present inside `TableProvider::scan` — the group-by sits above
    /// the scan in the logical plan — so this is populated only by a caller
    /// that can see the whole plan, like `Session::sql`. Its absence in a
    /// scan-level query means "no aggregate was visible", which a cube treats
    /// as not-a-match rather than as "plain scan".
    pub aggregate: Option<Aggregate>,
    /// Whether the session accepts an approximate answer to an exact ask.
    ///
    /// `false` means a stored estimate may serve only a query that asked for
    /// one. `true` — `SET quarry.approximate` — lets stored approximate
    /// state answer exact expressions too, like `count(distinct)`.
    pub approximate: bool,
}

impl Query {
    /// Every field this query restricts, however it restricts it.
    pub fn filtered_fields(&self) -> BTreeSet<FieldId> {
        self.predicates.iter().map(Predicate::field).collect()
    }

    /// The hashes this query compares `field` against with equality.
    pub fn equalities(&self, field: FieldId) -> Vec<u64> {
        self.predicates
            .iter()
            .filter_map(|p| match p {
                Predicate::Eq { field: f, value } if *f == field => Some(*value),
                _ => None,
            })
            .collect()
    }
}

/// How much of one file a [`Rewrite::Prune`] admits.
///
/// A kind that knows files only returns [`Scope::Whole`]; a kind that knows
/// which parts of a file hold a value names them instead. The granularity is
/// the file's own row groups and, finer, rows within a group — for an
/// in-memory file, its batches play the row-group role.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Scope {
    /// Every row group.
    #[default]
    Whole,
    /// The named row groups only.
    ///
    /// A group not named cannot hold a row satisfying the predicate this
    /// scope came from.
    Groups(BTreeSet<u32>),
    /// Named rows within named groups: group id → row offsets inside it.
    ///
    /// A group not named admits nothing. Group-local offsets, not file
    /// ordinals, so the scope composes with [`Scope::Groups`] without
    /// needing the file's footer.
    Rows(BTreeMap<u32, BTreeSet<u64>>),
}

impl Scope {
    /// What both scopes admit — how two prunes on one file compose.
    ///
    /// `None` means nothing survives: the file cannot satisfy the conjunct.
    pub fn intersect(&self, other: &Scope) -> Option<Scope> {
        match (self, other) {
            (Scope::Whole, scope) | (scope, Scope::Whole) => Some(scope.clone()),
            (Scope::Groups(a), Scope::Groups(b)) => {
                let both: BTreeSet<u32> = a.intersection(b).cloned().collect();
                (!both.is_empty()).then_some(Scope::Groups(both))
            }
            (Scope::Groups(groups), Scope::Rows(rows))
            | (Scope::Rows(rows), Scope::Groups(groups)) => {
                let both: BTreeMap<u32, BTreeSet<u64>> = rows
                    .iter()
                    .filter(|(group, _)| groups.contains(group))
                    .map(|(group, rows)| (*group, rows.clone()))
                    .collect();
                (!both.is_empty()).then_some(Scope::Rows(both))
            }
            (Scope::Rows(a), Scope::Rows(b)) => {
                let both: BTreeMap<u32, BTreeSet<u64>> = a
                    .iter()
                    .filter_map(|(group, rows)| {
                        let shared: BTreeSet<u64> =
                            rows.intersection(b.get(group)?).cloned().collect();
                        (!shared.is_empty()).then_some((*group, shared))
                    })
                    .collect();
                (!both.is_empty()).then_some(Scope::Rows(both))
            }
        }
    }
}

/// How derived state changes a plan.
///
/// The distinction carries the design's central safety property, so it is a
/// type rather than a convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rewrite {
    /// Narrow which files must be read, and how much of each.
    ///
    /// The engine still reads real data and applies deletes, so naming too
    /// many files — or too many row groups — is merely slow. Naming too
    /// *few* would lose rows, which is why files added since the derived
    /// state was built are unioned back in by [`Derived::may_serve`].
    ///
    /// When returned by [`Derived::may_serve`], `files` is guaranteed to
    /// contain only files live at the queried snapshot. A [`Kind`] computing
    /// one from older metadata need not check that itself.
    Prune {
        /// Files that may contain matching rows, and how much of each.
        files: BTreeMap<FileId, Scope>,
    },
    /// Replace the data source with the derived state.
    ///
    /// Nothing downstream re-reads the table, so staleness is not slow, it is
    /// wrong.
    Substitute {
        /// Whether the derived rows are table-shaped.
        ///
        /// The rule repairs additive staleness by reading the derived state
        /// *and* the files added since. That only produces the right answer
        /// when the derived state holds rows of the same shape as the table,
        /// so the two can simply be concatenated.
        ///
        /// An aggregate does not qualify: unioning a pre-computed `count(*)`
        /// with raw rows is nonsense. Combining those needs a merge step the
        /// engine does not have yet, so aggregated derived state is only
        /// admitted when there is no residual at all.
        unionable: bool,
        /// The grain the stored rows are aggregated at, if they are.
        ///
        /// `Some` means the derived state holds partial aggregates, not raw
        /// rows: whoever serves the rewrite must re-aggregate to the query's
        /// grain, and additive residual cannot be unioned — which is why such
        /// state is always `unionable: false`.
        rollup: Option<Aggregate>,
    },
}

impl Rewrite {
    /// A pruning rewrite admitting `files` whole.
    ///
    /// Most kinds know files only; a kind that also knows row groups writes
    /// the [`BTreeMap`] itself.
    pub fn prune(files: impl IntoIterator<Item = FileId>) -> Rewrite {
        Rewrite::Prune {
            files: files.into_iter().map(|f| (f, Scope::Whole)).collect(),
        }
    }

    /// Whether this rewrite replaces the data source rather than narrowing it.
    pub fn is_substituting(&self) -> bool {
        matches!(self, Rewrite::Substitute { .. })
    }

    /// Whether a residual scan may simply be read alongside this rewrite.
    pub fn can_union_residual(&self) -> bool {
        match self {
            Rewrite::Prune { .. } => true,
            Rewrite::Substitute { unionable, .. } => *unionable,
        }
    }
}

/// Why a piece of derived state was not admitted.
///
/// Kept specific so that `EXPLAIN` can say why a query was slower than the
/// user expected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Built from a different table.
    WrongTable,
    /// The kind cannot answer this query.
    NoMatch,
    /// The queried snapshot does not descend from the one it was built from,
    /// so it may have been built on an abandoned branch.
    NotDescendant,
    /// The queried snapshot, or the one it was built from, is unknown.
    UnknownSnapshot,
    /// Built under a different effective policy.
    PolicyMismatch,
    /// Rows have been removed since it was built, so it holds rows that are
    /// no longer live. Unusable, not repairable.
    SubtractiveChange,
    /// Rows have been added since it was built, and its own rows cannot simply
    /// be read alongside them — an aggregate would need merging, not
    /// concatenating.
    ResidualNotUnionable,
}

/// The rule's verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Apply the rewrite; nothing else is needed.
    Use(Rewrite),
    /// Apply the rewrite, and also scan these files, which were added after
    /// the derived state was built.
    UseWith {
        /// The rewrite to apply.
        rewrite: Rewrite,
        /// Files to read alongside it.
        also_scan: BTreeSet<FileId>,
    },
    /// Scan the table.
    Reject(Reason),
}

impl Decision {
    /// Whether the derived state may be used at all.
    pub fn is_admitted(&self) -> bool {
        !matches!(self, Decision::Reject(_))
    }
}

/// Whether a refresh brought derived state up to date.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refreshed {
    /// Up to date with the target snapshot.
    UpToDate,
    /// Cannot be updated in place; rebuild from the source table.
    NeedsRebuild,
}

/// A kind of derived state.
///
/// Three methods. Adding a kind should touch no other file: the registry, the
/// rule, and `EXPLAIN` all work in terms of this trait.
///
/// `Send + Sync` because a registry is shared across the threads a query
/// engine plans and executes on. Requiring it here rather than wrapping
/// derived state in a lock keeps contention out of the planning path, which
/// every query goes through.
pub trait Kind: fmt::Debug + Send + Sync {
    /// A short name, for `EXPLAIN`.
    fn name(&self) -> &'static str;

    /// Whether and how this can answer `query`.
    ///
    /// Concerned only with *shape* — the columns, predicates, or plan identity
    /// it can serve. Snapshot lineage, policy, and staleness are the rule's
    /// job, not the kind's, so that no kind can forget them.
    fn matches(&self, query: &Query) -> Option<Rewrite>;

    /// What it costs to answer a query *using* this.
    ///
    /// The cost of *keeping* it is `bytes` times a retention rate, which the
    /// registry computes; a kind does not need to know it. Selection compares
    /// use-costs, retirement compares keeping-costs against realized benefit,
    /// and both price in the same currency.
    fn cost(&self, prices: &PriceTable) -> Cost;

    /// Bring this up to date across `diff`.
    fn refresh(&mut self, diff: &Diff) -> Refreshed;
}

/// Where a piece of derived state came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    /// The table it was computed from.
    pub table: TableId,
    /// The snapshot it was computed from.
    pub snapshot: SnapshotId,
}

/// Bytes computed from a table at a known snapshot, which can answer some
/// queries more cheaply than the table can.
///
/// Always disposable. Losing all derived state costs performance and never
/// correctness, which is what makes every optimization in the engine safe to
/// attempt.
#[derive(Debug)]
pub struct Derived {
    /// Its identity.
    pub id: DerivedId,
    /// What it was computed from.
    pub source: Source,
    /// The policy it was computed under.
    pub policy: PolicyFingerprint,
    /// How many bytes it occupies.
    pub bytes: u64,
    kind: Box<dyn Kind>,
    uses: u64,
}

impl Derived {
    /// Register a piece of derived state.
    pub fn new(
        id: DerivedId,
        source: Source,
        policy: PolicyFingerprint,
        bytes: u64,
        kind: Box<dyn Kind>,
    ) -> Self {
        Derived {
            id,
            source,
            policy,
            bytes,
            kind,
            uses: 0,
        }
    }

    /// Its kind.
    pub fn kind(&self) -> &dyn Kind {
        self.kind.as_ref()
    }

    /// How many times it has served a query.
    pub fn uses(&self) -> u64 {
        self.uses
    }

    /// Note that it served a query. Drives retirement.
    pub fn record_use(&mut self) {
        self.uses = self.uses.saturating_add(1);
    }

    /// What it costs to answer a query using this.
    pub fn cost(&self, prices: &PriceTable) -> Cost {
        self.kind.cost(prices)
    }

    /// Bring it up to date across `diff`.
    pub fn refresh(&mut self, diff: &Diff) -> Refreshed {
        self.kind.refresh(diff)
    }

    /// **The one rule.** Whether this may serve `query`, and how.
    ///
    /// The four conditions, in order, so that a rejection names the first
    /// thing that failed:
    ///
    /// 1. `MATCH` — the kind can answer the query's shape
    /// 2. `LINEAGE` — the queried snapshot descends from the built-from one
    /// 3. `POLICY` — the principal's effective policy matches
    /// 4. `RESIDUAL` — what changed since it was built is tolerable
    ///
    /// Condition 4 differs by rewrite, and this is the design's central
    /// asymmetry:
    ///
    /// - [`Rewrite::Prune`] tolerates *any* change. Files added since must be
    ///   scanned too, or rows would be lost; files removed or re-deleted are
    ///   harmless, because the engine reads only live files and applies
    ///   deletes as it goes. Over-selection is conservative.
    /// - [`Rewrite::Substitute`] tolerates only *additive* change. Once rows
    ///   have been removed, the derived state holds rows that are no longer
    ///   live and no amount of extra reading removes them. And when rows have
    ///   been *added*, only table-shaped derived state can be read alongside
    ///   them; an aggregate would need merging rather than concatenating, so
    ///   it is admitted only when there is no residual at all.
    pub fn may_serve(&self, query: &Query, graph: &SnapshotGraph) -> Decision {
        if self.source.table != query.table {
            return Decision::Reject(Reason::WrongTable);
        }

        let Some(rewrite) = self.kind.matches(query) else {
            return Decision::Reject(Reason::NoMatch);
        };

        let (Some(_), Some(at)) = (graph.get(self.source.snapshot), graph.get(query.snapshot))
        else {
            return Decision::Reject(Reason::UnknownSnapshot);
        };
        if !graph.is_descendant_or_self(query.snapshot, self.source.snapshot) {
            return Decision::Reject(Reason::NotDescendant);
        }

        if self.policy != query.policy {
            return Decision::Reject(Reason::PolicyMismatch);
        }

        // A pruning rewrite was computed against an older file set, so it can
        // name files the table no longer references. Drop them here rather
        // than trusting every consumer to intersect with the live set:
        // reading a file the table has dropped would resurrect deleted rows.
        let rewrite = match rewrite {
            Rewrite::Prune { files } => Rewrite::Prune {
                files: files
                    .into_iter()
                    .filter(|(file, _)| at.files().contains_key(file))
                    .collect(),
            },
            substituting => substituting,
        };

        let Some(diff) = graph.diff(self.source.snapshot, query.snapshot) else {
            return Decision::Reject(Reason::UnknownSnapshot);
        };

        if rewrite.is_substituting() && !diff.is_purely_additive() {
            return Decision::Reject(Reason::SubtractiveChange);
        }

        if diff.added.is_empty() {
            return Decision::Use(rewrite);
        }
        if !rewrite.can_union_residual() {
            return Decision::Reject(Reason::ResidualNotUnionable);
        }
        Decision::UseWith {
            rewrite,
            also_scan: diff.added,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{DeleteState, Snapshot};

    fn rows(entries: &[(u32, &[u64])]) -> Scope {
        Scope::Rows(
            entries
                .iter()
                .map(|(group, rows)| (*group, rows.iter().cloned().collect()))
                .collect(),
        )
    }

    #[test]
    fn whole_intersected_with_anything_is_that_thing() {
        let rows = rows(&[(0, &[1, 2])]);
        assert_eq!(
            Scope::Whole.intersect(&rows),
            Some(rows.clone()),
            "Whole admits everything, so the finer scope decides"
        );
        assert_eq!(rows.intersect(&Scope::Whole), Some(rows));
    }

    #[test]
    fn group_scopes_restrict_row_scopes_without_offsets() {
        let scoped = rows(&[(0, &[1, 2]), (1, &[5])]);
        assert_eq!(
            Scope::Groups(BTreeSet::from([0])).intersect(&scoped),
            Some(rows(&[(0, &[1, 2])])),
            "group 1 falls away — it needs no row offsets to do so"
        );
        assert_eq!(
            Scope::Groups(BTreeSet::from([9])).intersect(&scoped),
            None,
            "no admitted group survives"
        );
    }

    #[test]
    fn row_scopes_intersect_group_by_group() {
        let a = rows(&[(0, &[1, 2]), (1, &[7])]);
        let b = rows(&[(0, &[2, 3]), (2, &[9])]);
        assert_eq!(
            a.intersect(&b),
            Some(rows(&[(0, &[2])])),
            "only row 2 of group 0 is admitted by both"
        );
    }

    #[test]
    fn an_empty_intersection_means_the_file_drops() {
        let a = rows(&[(0, &[1])]);
        let b = rows(&[(0, &[2])]);
        assert_eq!(a.intersect(&b), None);
    }

    fn filt(field: u32, text: &str) -> Filter {
        Filter {
            field: Some(field),
            text: text.into(),
        }
    }

    #[test]
    fn a_filter_implies_itself() {
        let f = filt(1, "ts >= Int64(3)");
        assert!(f.implies(&f));
    }

    #[test]
    fn a_narrower_range_implies_a_wider_one() {
        let query = filt(1, "ts >= Int64(3)");
        let baked = filt(1, "ts >= Int64(1)");
        assert!(query.implies(&baked));
        assert!(
            !baked.implies(&query),
            "the wider bound proves nothing narrower"
        );
    }

    #[test]
    fn an_equality_implies_the_bounds_it_satisfies() {
        let eq = filt(1, "status = Int64(3)");
        assert!(eq.implies(&filt(1, "status >= Int64(3)")));
        assert!(eq.implies(&filt(1, "status <= Int64(3)")));
        assert!(!eq.implies(&filt(1, "status > Int64(3)")));
        assert!(!eq.implies(&filt(1, "status < Int64(3)")));
    }

    #[test]
    fn strictness_is_respected_at_the_edge() {
        // `x >= 3` does not imply `x > 3`; `x > 3` does imply `x >= 3`.
        assert!(!filt(1, "x >= Int64(3)").implies(&filt(1, "x > Int64(3)")));
        assert!(filt(1, "x > Int64(3)").implies(&filt(1, "x >= Int64(3)")));
    }

    #[test]
    fn different_fields_imply_nothing() {
        assert!(!filt(1, "a >= Int64(3)").implies(&filt(2, "b >= Int64(1)")));
    }

    #[test]
    fn unparseable_filters_imply_nothing() {
        // A filter outside `column op literal` — say `a + 1 = 5` rendered
        // whole — gets no implication; matching stays exact-text only.
        assert!(!filt(1, "a + Int64(1) = Int64(5)").implies(&filt(1, "a >= Int64(1)")));
    }

    #[test]
    fn timestamp_and_string_bounds_order() {
        let baked = filt(1, "ts >= TimestampNanosecond(100, None)");
        let query = filt(1, "ts >= TimestampNanosecond(200, None)");
        assert!(query.implies(&baked));

        let baked = filt(1, "name >= Utf8(\"b\")");
        let query = filt(1, "name >= Utf8(\"c\")");
        assert!(query.implies(&baked));
    }

    const POLICY: PolicyFingerprint = PolicyFingerprint(1);
    const OTHER_POLICY: PolicyFingerprint = PolicyFingerprint(2);

    fn t() -> TableId {
        TableId("events".into())
    }

    fn f(name: &str) -> FileId {
        FileId(name.to_owned())
    }

    fn s(id: i64) -> SnapshotId {
        SnapshotId(id)
    }

    /// Prunes to a fixed file set when the query filters on field 4.
    #[derive(Debug)]
    struct FakeIndex {
        files: BTreeSet<FileId>,
    }

    impl Kind for FakeIndex {
        fn name(&self) -> &'static str {
            "index"
        }
        fn matches(&self, query: &Query) -> Option<Rewrite> {
            query
                .filtered_fields()
                .contains(&4)
                .then(|| Rewrite::prune(self.files.iter().cloned()))
        }
        fn cost(&self, _prices: &PriceTable) -> Cost {
            Cost::ZERO
        }
        fn refresh(&mut self, _diff: &Diff) -> Refreshed {
            Refreshed::UpToDate
        }
    }

    /// Substitutes for one exact plan.
    #[derive(Debug)]
    struct FakeResult {
        plan_hash: u64,
    }

    impl Kind for FakeResult {
        fn name(&self) -> &'static str {
            "result"
        }
        fn matches(&self, query: &Query) -> Option<Rewrite> {
            (query.plan_hash == self.plan_hash).then_some(Rewrite::Substitute {
                unionable: true,
                rollup: None,
            })
        }
        fn cost(&self, _prices: &PriceTable) -> Cost {
            Cost::ZERO
        }
        fn refresh(&mut self, _diff: &Diff) -> Refreshed {
            Refreshed::NeedsRebuild
        }
    }

    fn index_at(snapshot: SnapshotId, files: &[&str]) -> Derived {
        Derived::new(
            DerivedId("idx".into()),
            Source {
                table: t(),
                snapshot,
            },
            POLICY,
            1024,
            Box::new(FakeIndex {
                files: files.iter().map(|n| f(n)).collect(),
            }),
        )
    }

    fn result_at(snapshot: SnapshotId, plan_hash: u64) -> Derived {
        Derived::new(
            DerivedId("res".into()),
            Source {
                table: t(),
                snapshot,
            },
            POLICY,
            64,
            Box::new(FakeResult { plan_hash }),
        )
    }

    fn query_at(snapshot: SnapshotId) -> Query {
        Query {
            table: t(),
            snapshot,
            policy: POLICY,
            plan_hash: 99,
            plan: None,
            projected: BTreeSet::from([4, 7]),
            predicates: vec![Predicate::Eq {
                field: 4,
                value: 0xABC,
            }],
            aggregate: None,
            approximate: false,
        }
    }

    /// 810 -> 811 (appends "b") -> 812 (appends "c")
    fn appends() -> SnapshotGraph {
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
    fn current_derived_state_is_used_as_is() {
        let g = appends();
        let d = index_at(s(812), &["a"]);
        assert_eq!(
            d.may_serve(&query_at(s(812)), &g),
            Decision::Use(Rewrite::Prune {
                files: BTreeMap::from([(f("a"), Scope::Whole)])
            })
        );
    }

    #[test]
    fn a_stale_index_is_used_with_the_added_files() {
        // Pruning tolerates staleness, but the files added since must be
        // scanned too or their rows would be lost.
        let g = appends();
        let d = index_at(s(810), &["a"]);
        assert_eq!(
            d.may_serve(&query_at(s(812)), &g),
            Decision::UseWith {
                rewrite: Rewrite::Prune {
                    files: BTreeMap::from([(f("a"), Scope::Whole)])
                },
                also_scan: BTreeSet::from([f("b"), f("c")]),
            }
        );
    }

    #[test]
    fn a_stale_result_is_used_with_the_added_files() {
        let g = appends();
        let d = result_at(s(811), 99);
        let decision = d.may_serve(&query_at(s(812)), &g);
        assert_eq!(
            decision,
            Decision::UseWith {
                rewrite: Rewrite::Substitute {
                    unionable: true,
                    rollup: None,
                },
                also_scan: BTreeSet::from([f("c")]),
            }
        );
    }

    #[test]
    fn an_aggregated_result_is_refused_once_rows_are_added() {
        // Reading a pre-computed count(*) alongside newly added raw rows is
        // nonsense: combining them needs a merge, not a concatenation.
        #[derive(Debug)]
        struct FakeAggregate;

        impl Kind for FakeAggregate {
            fn name(&self) -> &'static str {
                "aggregate"
            }
            fn matches(&self, _query: &Query) -> Option<Rewrite> {
                Some(Rewrite::Substitute {
                    unionable: false,
                    rollup: None,
                })
            }
            fn cost(&self, _prices: &PriceTable) -> Cost {
                Cost::ZERO
            }
            fn refresh(&mut self, _diff: &Diff) -> Refreshed {
                Refreshed::NeedsRebuild
            }
        }

        let aggregate = Derived::new(
            DerivedId("cube".into()),
            Source {
                table: t(),
                snapshot: s(810),
            },
            POLICY,
            32,
            Box::new(FakeAggregate),
        );

        let g = appends();

        // Nothing added yet: usable.
        assert_eq!(
            aggregate.may_serve(&query_at(s(810)), &g),
            Decision::Use(Rewrite::Substitute {
                unionable: false,
                rollup: None,
            })
        );

        // A file has been appended: refused, where a table-shaped result
        // would have been admitted with a residual scan.
        assert_eq!(
            aggregate.may_serve(&query_at(s(811)), &g),
            Decision::Reject(Reason::ResidualNotUnionable)
        );
        assert!(
            result_at(s(810), 99)
                .may_serve(&query_at(s(811)), &g)
                .is_admitted(),
            "a table-shaped result tolerates the same append"
        );
    }

    #[test]
    fn a_delete_disqualifies_a_substituting_kind_but_not_a_pruning_one() {
        // 811 deletes rows from "a" without adding or removing any file.
        let g = SnapshotGraph::new()
            .with(Snapshot::root(s(810)).with_clean_file(f("a")))
            .with(Snapshot::child_of(s(811), s(810)).with_file(f("a"), DeleteState(7)));

        let result = result_at(s(810), 99);
        assert_eq!(
            result.may_serve(&query_at(s(811)), &g),
            Decision::Reject(Reason::SubtractiveChange),
            "a cached result would return rows that are no longer live"
        );

        let index = index_at(s(810), &["a"]);
        assert_eq!(
            index.may_serve(&query_at(s(811)), &g),
            Decision::Use(Rewrite::Prune {
                files: BTreeMap::from([(f("a"), Scope::Whole)])
            }),
            "pruning is safe: the engine still reads the file and applies deletes"
        );
    }

    #[test]
    fn a_prune_set_never_names_a_file_the_table_has_dropped() {
        // Compaction replaces a and b with merged. The index still points at
        // a, which no longer exists: reading it would resurrect its rows.
        let g = SnapshotGraph::new()
            .with(
                Snapshot::root(s(810))
                    .with_clean_file(f("a"))
                    .with_clean_file(f("b")),
            )
            .with(Snapshot::child_of(s(811), s(810)).with_clean_file(f("merged")));

        assert_eq!(
            index_at(s(810), &["a"]).may_serve(&query_at(s(811)), &g),
            Decision::UseWith {
                rewrite: Rewrite::Prune {
                    files: BTreeMap::new()
                },
                also_scan: BTreeSet::from([f("merged")]),
            },
            "the dropped file is filtered out; the new one is scanned"
        );
    }

    #[test]
    fn a_prune_set_keeps_files_that_are_still_live() {
        let g = appends();
        assert_eq!(
            index_at(s(810), &["a"]).may_serve(&query_at(s(811)), &g),
            Decision::UseWith {
                rewrite: Rewrite::Prune {
                    files: BTreeMap::from([(f("a"), Scope::Whole)])
                },
                also_scan: BTreeSet::from([f("b")]),
            }
        );
    }

    #[test]
    fn compaction_disqualifies_a_substituting_kind() {
        let g = SnapshotGraph::new()
            .with(
                Snapshot::root(s(810))
                    .with_clean_file(f("small-1"))
                    .with_clean_file(f("small-2")),
            )
            .with(Snapshot::child_of(s(811), s(810)).with_clean_file(f("merged")));

        assert_eq!(
            result_at(s(810), 99).may_serve(&query_at(s(811)), &g),
            Decision::Reject(Reason::SubtractiveChange)
        );
    }

    #[test]
    fn derived_state_from_an_abandoned_branch_is_refused() {
        let g = SnapshotGraph::new()
            .with(Snapshot::root(s(810)).with_clean_file(f("a")))
            .with(Snapshot::child_of(s(811), s(810)).with_clean_file(f("a")))
            .with(Snapshot::child_of(s(812), s(810)).with_clean_file(f("a")));

        assert_eq!(
            index_at(s(811), &["a"]).may_serve(&query_at(s(812)), &g),
            Decision::Reject(Reason::NotDescendant)
        );
    }

    #[test]
    fn a_newer_query_snapshot_is_required() {
        let g = appends();
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&query_at(s(810)), &g),
            Decision::Reject(Reason::NotDescendant)
        );
    }

    #[test]
    fn a_policy_fingerprint_is_stable_and_sensitive() {
        let a = PolicyFingerprint::of(&["tenant=1", "mask:email"]);
        let again = PolicyFingerprint::of(&["tenant=1", "mask:email"]);
        let other = PolicyFingerprint::of(&["tenant=2", "mask:email"]);

        assert_eq!(a, again, "the same policy must hash identically");
        assert_ne!(a, other, "a different filter is a different policy");
    }

    #[test]
    fn a_policy_mismatch_is_refused() {
        let g = appends();
        let mut q = query_at(s(812));
        q.policy = OTHER_POLICY;
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&q, &g),
            Decision::Reject(Reason::PolicyMismatch)
        );
    }

    #[test]
    fn a_different_table_is_refused() {
        let g = appends();
        let mut q = query_at(s(812));
        q.table = TableId("other".into());
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&q, &g),
            Decision::Reject(Reason::WrongTable)
        );
    }

    #[test]
    fn a_kind_that_cannot_answer_is_refused() {
        let g = appends();
        let mut q = query_at(s(812));
        q.predicates.clear(); // the fake index needs a predicate on field 4
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&q, &g),
            Decision::Reject(Reason::NoMatch)
        );

        let mut q = query_at(s(812));
        q.plan_hash = 12345;
        assert_eq!(
            result_at(s(812), 99).may_serve(&q, &g),
            Decision::Reject(Reason::NoMatch)
        );
    }

    #[test]
    fn an_unknown_snapshot_is_refused() {
        let g = appends();
        assert_eq!(
            index_at(s(777), &["a"]).may_serve(&query_at(s(812)), &g),
            Decision::Reject(Reason::UnknownSnapshot)
        );
        assert_eq!(
            index_at(s(812), &["a"]).may_serve(&query_at(s(777)), &g),
            Decision::Reject(Reason::UnknownSnapshot)
        );
    }

    #[test]
    fn a_query_groups_finer_than_a_cube_cannot_rollup() {
        let cube = Aggregate {
            group_by: BTreeSet::from([DAY]),
            measures: BTreeSet::from([count_star()]),
        };
        // A query grouping by (day, tenant) is finer than the cube's grain:
        // the cube's rows have already collapsed tenant.
        let finer = Aggregate {
            group_by: BTreeSet::from([DAY, TENANT]),
            measures: BTreeSet::from([count_star()]),
        };
        assert!(!finer.covered_by(&cube, false));
    }

    #[test]
    fn a_query_groups_coarser_than_a_cube_rolls_up() {
        let cube = Aggregate {
            group_by: BTreeSet::from([DAY, TENANT]),
            measures: BTreeSet::from([count_star(), sum_bytes()]),
        };
        let coarser = Aggregate {
            group_by: BTreeSet::from([DAY]),
            measures: BTreeSet::from([count_star(), sum_bytes()]),
        };
        assert!(coarser.covered_by(&cube, false));
    }

    #[test]
    fn a_measure_rolls_up_only_through_itself() {
        let cube = Aggregate {
            group_by: BTreeSet::from([DAY]),
            measures: BTreeSet::from([count_star()]),
        };
        let wants_sum = Aggregate {
            group_by: BTreeSet::from([DAY]),
            measures: BTreeSet::from([sum_bytes()]),
        };
        // COUNT is no basis for SUM.
        assert!(!wants_sum.covered_by(&cube, false));
        let wants_count_of_bytes = Aggregate {
            group_by: BTreeSet::from([DAY]),
            measures: BTreeSet::from([Measure {
                func: AggFunc::Count,
                field: Some(BYTES),
            }]),
        };
        // COUNT(*) counts rows; COUNT(bytes) counts non-null bytes. A cube
        // holding one cannot serve the other.
        assert!(!wants_count_of_bytes.covered_by(&cube, false));
    }

    const DAY: FieldId = 1;
    const TENANT: FieldId = 2;
    const BYTES: FieldId = 3;

    fn count_star() -> Measure {
        Measure {
            func: AggFunc::Count,
            field: None,
        }
    }

    fn sum_bytes() -> Measure {
        Measure {
            func: AggFunc::Sum,
            field: Some(BYTES),
        }
    }

    #[test]
    fn use_counting_saturates() {
        let mut d = index_at(s(812), &["a"]);
        assert_eq!(d.uses(), 0);
        d.record_use();
        assert_eq!(d.uses(), 1);
        d.uses = u64::MAX;
        d.record_use();
        assert_eq!(d.uses(), u64::MAX);
    }
}
