//! Serving joins from a cached build-side relation.
//!
//! `TableProvider::scan` never sees the `Join` node — it sits above the scan
//! in the logical plan — so a join hash cannot be served the way an index or
//! a result cache is. This is the same plan-level treatment
//! [`cube`](super::cube) gives aggregates and [`ann`](super::ann) gives
//! top-k: find the shape, let the table describe it exactly, and let
//! [`Derived::may_serve`](crate::derived::Derived::may_serve) decide.
//!
//! # Only the build side is replaced
//!
//! The rewrite swaps the build-side `TableScan`'s *source* and changes
//! nothing else — not the join type, not the join keys, not the filter, and
//! above all not the probe side:
//!
//! ```text
//! Join: orders.id = lineitems.order_id     Join: orders.id = lineitems.order_id
//!   TableScan: orders (build)                TableScan: orders (stored rows)
//!   TableScan: lineitems (probe)              TableScan: lineitems (probe)
//! ```
//!
//! That is what makes this exact: the join runs as written, over rows with
//! the same schema. DataFusion's `HashJoinExec` builds its hash table from
//! the substituted rows — the saving is that the build side is local instead
//! of in object storage.
//!
//! # Which side is the build side
//!
//! DataFusion's optimizer picks the build side at physical planning time,
//! after this rewrite runs. The logical `Join` has no build/probe
//! distinction, so this rewrite treats the *left* side as the build side —
//! the convention SQL writers follow when they put the smaller table first.
//! If the optimizer later swaps sides, the stored rows still serve: a
//! `JoinHash` matches on `(table, keys, columns)`, not on which side it is.
//!
//! # What an append does
//!
//! The stored rows are a complete build side for their snapshot — they
//! cannot be unioned with appended rows without missing the new files'
//! rows entirely. [`JoinHash`](crate::kinds::JoinHash) therefore declares
//! itself non-unionable, so the rule rejects an appended-to table by name
//! and the query reads the table — correct, and slower until the
//! optimizer's refresh step rebuilds the hash.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::datasource::{DefaultTableSource, MemTable, provider_as_source};
use datafusion::error::Result as DfResult;
use datafusion::logical_expr::{Expr, JoinType, LogicalPlan, TableScan};

use crate::derived::{Decision, DerivedId, FieldId, Query, Rewrite};
use crate::workload::JoinAsk;

use super::{QuarryTable, ScanReport, Staleness};

/// What a join subtree asks of the table at its build side.
pub(crate) struct Ask<'a> {
    /// The build-side table.
    pub table: &'a QuarryTable,
    /// The scan the ask was read from — matched by identity at swap time so
    /// a second scan of the same table elsewhere in the plan is not
    /// mistaken for the build side.
    pub scan: &'a TableScan,
    /// The join key fields on the build side.
    pub keys: BTreeSet<FieldId>,
    /// All columns the join reads from the build side: join keys plus
    /// whatever the projection above carries through.
    pub columns: BTreeSet<FieldId>,
}

/// The first join ask anywhere in `plan`, or none.
///
/// Walks the plan looking for a `Join` whose left side is a `TableScan` of a
/// `QuarryTable`. Only inner and left joins are recognised: a right or full
/// join makes the *right* side the build side in DataFusion's physical plan,
/// and a cross join has no keys to hash on.
///
/// The columns a hash must cover are computed against `plan` itself rather
/// than the scan's `projection` — at logical-plan time the projection is
/// un-narrowed, so "everything above the scan" is where the real need is.
pub(crate) fn first_ask(plan: &LogicalPlan) -> Option<Ask<'_>> {
    ask_of(plan, plan).or_else(|| {
        plan.inputs()
            .iter()
            .find_map(|input| first_ask_at(input, plan))
    })
}

/// [`first_ask`] at a subtree, keeping `root` for the reference walk.
fn first_ask_at<'a>(plan: &'a LogicalPlan, root: &LogicalPlan) -> Option<Ask<'a>> {
    ask_of(plan, root).or_else(|| {
        plan.inputs()
            .iter()
            .find_map(|input| first_ask_at(input, root))
    })
}

/// The join ask at `plan`, if it is a `Join` this rewrite can serve.
fn ask_of<'a>(plan: &'a LogicalPlan, root: &LogicalPlan) -> Option<Ask<'a>> {
    let LogicalPlan::Join(join) = plan else {
        return None;
    };
    // Only inner and left joins: the left side is the build side. A right
    // join would make the right side the build side; a full join builds both.
    // Cross joins have no equijoin keys.
    if !matches!(join.join_type, JoinType::Inner | JoinType::Left) {
        return None;
    }
    let scan = table_scan(&join.left)?;
    let table = quarry_table(scan)?;
    // The join keys on the build (left) side, from both the equijoin `on`
    // pairs and the `filter` field. DataFusion's SQL planner puts a single
    // `ON a.x = b.x` into `filter` rather than `on`, so both must be checked.
    let mut keys: BTreeSet<FieldId> = join
        .on
        .iter()
        .filter_map(|(left, _)| column_field(table, left))
        .collect();
    if let Some(filter) = &join.filter {
        for (left, _) in equalities(filter, table) {
            keys.insert(left);
        }
    }
    if keys.is_empty() {
        return None;
    }
    // Every `on` pair must have its left side as a bare column of this table.
    // A key that is not means the join is not an equijoin we can accelerate.
    let on_keys: BTreeSet<FieldId> = join
        .on
        .iter()
        .filter_map(|(left, _)| column_field(table, left))
        .collect();
    if on_keys.len() != join.on.len() {
        return None;
    }
    // All columns the plan reads from the build side: join keys, the scan's
    // own filters, and whatever the nodes above reference through the scan's
    // qualifiers. The scan's `projection` is no help — it is un-narrowed at
    // this stage — so the whole plan is walked instead.
    let qualifiers = qualifiers(&join.left);
    let mut columns = referenced_fields(root, &qualifiers, table);
    columns.extend(keys.iter().copied());
    Some(Ask {
        table,
        scan,
        keys,
        columns,
    })
}

/// The names a scan answers to: the aliases wrapping it and its own table
/// name, collected on the descent from `plan`.
fn qualifiers(plan: &LogicalPlan) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    fn walk(plan: &LogicalPlan, out: &mut BTreeSet<String>) {
        match plan {
            LogicalPlan::SubqueryAlias(alias) => {
                out.insert(alias.alias.to_string());
                walk(&alias.input, out);
            }
            LogicalPlan::Projection(projection) => walk(&projection.input, out),
            LogicalPlan::TableScan(scan) => {
                out.insert(scan.table_name.to_string());
            }
            _ => {}
        }
    }
    walk(plan, &mut out);
    out
}

/// Every field of `table` that a column expression in `plan` refers to.
///
/// A qualified column attributes to this table when its relation names one
/// of `qualifiers`; an unqualified one when its name is a field — the plan
/// already analysed, so a bare name can only have meant one column, and
/// guessing more than needed is safe: it demands coverage, it cannot
/// produce a wrong answer.
fn referenced_fields(
    plan: &LogicalPlan,
    qualifiers: &BTreeSet<String>,
    table: &QuarryTable,
) -> BTreeSet<FieldId> {
    let mut fields = BTreeSet::new();
    let mut stack = vec![plan];
    while let Some(node) = stack.pop() {
        for expr in node.expressions() {
            let _ = expr.apply(|e| {
                if let Expr::Column(col) = e {
                    let ours = match &col.relation {
                        Some(relation) => qualifiers.contains(&relation.to_string()),
                        None => table.field_id_of(col.name()).is_some(),
                    };
                    if ours {
                        if let Some(field) = table.field_id_of(col.name()) {
                            fields.insert(field);
                        }
                    }
                }
                Ok(TreeNodeRecursion::Continue)
            });
        }
        stack.extend(node.inputs());
    }
    fields
}

/// The `TableScan` at `plan`, descending through row-preserving nodes that
/// DataFusion inserts for aliases and projections.
fn table_scan(plan: &LogicalPlan) -> Option<&TableScan> {
    match plan {
        LogicalPlan::TableScan(scan) => Some(scan),
        LogicalPlan::SubqueryAlias(alias) => table_scan(&alias.input),
        LogicalPlan::Projection(projection) => table_scan(&projection.input),
        _ => None,
    }
}

/// Whether two scans are the same node seen through a clone.
///
/// `TableScan` has no `PartialEq`, so the comparison is by hand: the source
/// `Arc` is shared across the clone `transform_up` walks, and the rest of
/// the fields are compared by value.
fn same_scan(a: &TableScan, b: &TableScan) -> bool {
    Arc::ptr_eq(&a.source, &b.source)
        && a.table_name == b.table_name
        && a.projection == b.projection
        && a.filters == b.filters
        && a.fetch == b.fetch
}

/// The [`QuarryTable`] a scan reads, if it reads one.
fn quarry_table(scan: &TableScan) -> Option<&QuarryTable> {
    scan.source
        .as_any()
        .downcast_ref::<DefaultTableSource>()?
        .table_provider
        .as_any()
        .downcast_ref::<QuarryTable>()
}

/// The field id a column expression names on `table`, if it is a bare
/// column of this table.
fn column_field(table: &QuarryTable, expr: &Expr) -> Option<FieldId> {
    let Expr::Column(col) = expr else {
        return None;
    };
    table.field_id_of(col.name())
}

/// The equality conjuncts in `expr` that compare a column of `table` to a
/// column of another table.
///
/// DataFusion's SQL planner puts `ON a.x = b.x` into the join's `filter`
/// field rather than `on` for some forms, so the keys must be extracted from
/// both. Each returned pair is `(field on this table, the other expression)`.
fn equalities(expr: &Expr, table: &QuarryTable) -> Vec<(FieldId, Expr)> {
    let mut out = Vec::new();
    for conjunct in split_and(expr) {
        if let Expr::BinaryExpr(bin) = conjunct {
            if let datafusion::logical_expr::Operator::Eq = bin.op {
                if let Some(f) = column_field(table, &bin.left) {
                    out.push((f, (*bin.right).clone()));
                } else if let Some(f) = column_field(table, &bin.right) {
                    out.push((f, (*bin.left).clone()));
                }
            }
        }
    }
    out
}

/// Split a conjunctive expression into its top-level `AND` parts.
fn split_and(expr: &Expr) -> Vec<&Expr> {
    fn walk<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
        match expr {
            Expr::BinaryExpr(bin) if bin.op == datafusion::logical_expr::Operator::And => {
                walk(&bin.left, out);
                walk(&bin.right, out);
            }
            _ => out.push(expr),
        }
    }
    let mut out = Vec::new();
    walk(expr, &mut out);
    out
}

/// The [`Query`] an [`Ask`] reduces to — the identity a join hash matches.
pub(crate) fn query_of(ask: &Ask, approximate: bool, stale: bool) -> Option<Query> {
    ask.table
        .join_query(&ask.keys, &ask.columns, &[], approximate, stale)
}

/// Rewrite `plan` to read a join hash wherever one is admissible.
///
/// Returns the rewritten plan and the id of what served it, for reporting.
pub(crate) fn rewrite(
    plan: &LogicalPlan,
    approximate: bool,
    stale: bool,
) -> DfResult<(LogicalPlan, Option<DerivedId>)> {
    // Recognised at the `Join`, but rewritten at the leaf — so the shape is
    // checked once and then the source is swapped where it lives.
    let Some(ask) = first_ask(plan) else {
        return Ok((plan.clone(), None));
    };
    let Some(source) = stored_source(&ask, approximate, stale) else {
        return Ok((plan.clone(), None));
    };
    let (source, id, stored_schema, report) = source;
    let table_schema = datafusion::catalog::TableProvider::schema(ask.table);

    let mut swapped = false;
    let rewritten = plan
        .clone()
        .transform_up(|node| {
            let LogicalPlan::TableScan(scan) = &node else {
                return Ok(Transformed::no(node));
            };
            // The ask's own scan, matched by structural identity — the source
            // `Arc` survives the clone that `transform_up` walks, and a
            // second scan of the same table elsewhere in the plan differs in
            // name, projection, filters, or fetch.
            if swapped || !same_scan(scan, ask.scan) {
                return Ok(Transformed::no(node));
            }
            // The scan's projection indexes the table's schema; the stored
            // rows may hold only the covered columns. Remap by name — a
            // projected column the hash does not cover means it cannot serve
            // after all, and the scan is left alone.
            let projection = match &scan.projection {
                Some(indices) => {
                    let mut mapped = Vec::with_capacity(indices.len());
                    for &index in indices {
                        let Ok(position) = stored_schema.index_of(table_schema.field(index).name())
                        else {
                            return Ok(Transformed::no(node));
                        };
                        mapped.push(position);
                    }
                    Some(mapped)
                }
                None => None,
            };
            // `try_new` recomputes the projected schema against the
            // substitute's; every other field stays the scan's own — the
            // pushed filters and the fetch included.
            let Ok(replaced) = TableScan::try_new(
                scan.table_name.clone(),
                Arc::clone(&source),
                projection,
                scan.filters.clone(),
                scan.fetch,
            ) else {
                return Ok(Transformed::no(node));
            };
            swapped = true;
            Ok(Transformed::yes(LogicalPlan::TableScan(replaced)))
        })
        .map(|t| t.data)?;

    // Reported only once the swap took: a scan that stayed a table scan is
    // not a substitution however admissible the candidate was.
    if swapped {
        ask.table.note_scan(report);
    }
    Ok((rewritten, swapped.then_some(id)))
}

/// The stored rows a join hash would serve `ask` from, as a table source
/// over the schema the rows were stored with, plus the report the swap
/// would produce.
///
/// `None` whenever the rule declines or no usable rows are registered —
/// either of which leaves the query reading the table.
fn stored_source(
    ask: &Ask,
    approximate: bool,
    stale: bool,
) -> Option<(
    Arc<dyn datafusion::logical_expr::TableSource>,
    DerivedId,
    SchemaRef,
    ScanReport,
)> {
    let table = ask.table;
    let query = query_of(ask, approximate, stale)?;
    // The candidate borrows the registry; the guard stays alive across it.
    let registry = table.registry().read().ok()?;
    let candidate = registry.best(&query, table.graph(), table.prices())?;
    let id = candidate.derived.id.clone();
    let built_at = candidate.derived.source.snapshot;

    // Only a row-shaped substitute serves this. `Use` reaches here when the
    // snapshot is unchanged; `UseStale` when the session opted in and only
    // appends happened since the build.
    let staleness = match &candidate.decision {
        Decision::Use(Rewrite::Substitute { rollup: None, .. }) => None,
        Decision::UseStale {
            rewrite: Rewrite::Substitute { rollup: None, .. },
            missed,
        } => Some(Staleness {
            built_at,
            missed_files: missed.clone(),
            missed_bytes: table.bytes_of(missed),
        }),
        _ => return None,
    };

    // Persisted rows first: they stream through the store rather than
    // sitting in memory beside the hash table that consumes them.
    let (source, stored_schema) = match table.row_file_source(&id) {
        Some(pair) => pair,
        None => {
            let stored = table.stored(&id)?;
            if stored.is_empty() {
                return None;
            }
            let schema = stored[0].schema();
            let mem = MemTable::try_new(Arc::clone(&schema), vec![stored]).ok()?;
            (provider_as_source(Arc::new(mem)), schema)
        }
    };

    Some((
        source,
        id.clone(),
        stored_schema,
        report(table, &query, &id, ask, staleness),
    ))
}

/// What the serving path reports.
///
/// `approximate` is deliberately false: a join hash holds every row of the
/// build side, so the join is exact.
fn report(
    table: &QuarryTable,
    query: &Query,
    used: &DerivedId,
    ask: &Ask<'_>,
    stale: Option<Staleness>,
) -> ScanReport {
    ScanReport {
        files_read: Default::default(),
        filters: Vec::new(),
        scopes: Default::default(),
        used: vec![used.0.clone()],
        also_scanned: Default::default(),
        substituted: true,
        approximate: false,
        stale,
        plan_hash: query.plan_hash,
        bytes_if_full_scan: table.live_bytes(),
        fingerprint: crate::workload::Fingerprint::of(query),
        plan: query.plan.clone(),
        aggregate: None,
        nearest: None,
        join: Some(crate::workload::JoinAsk {
            table: table.table_id().clone(),
            keys: ask.keys.clone(),
            columns: ask.columns.clone(),
        }),
        text: Vec::new(),
    }
}

/// The [`JoinAsk`] an [`Ask`] reduces to — for workload observation.
pub(crate) fn ask_of_workload(ask: &Ask) -> JoinAsk {
    JoinAsk {
        table: ask.table.table_id().clone(),
        keys: ask.keys.clone(),
        columns: ask.columns.clone(),
    }
}
