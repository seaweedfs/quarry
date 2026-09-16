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
//! The stored rows are sorted by the join key, so they cannot simply be
//! concatenated with appended rows — the sort order would be wrong.
//! [`JoinHash`](crate::kinds::JoinHash) therefore declares itself
//! non-unionable, so the rule rejects an appended-to table by name and the
//! query reads the table — correct, and slower until the optimizer's
//! refresh step rebuilds the hash.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::datasource::{DefaultTableSource, MemTable, provider_as_source};
use datafusion::error::Result as DfResult;
use datafusion::logical_expr::{Expr, JoinType, LogicalPlan, TableScan};

use crate::derived::{Decision, DerivedId, FieldId, Query, Rewrite};
use crate::workload::JoinAsk;

use super::{QuarryTable, ScanReport};

/// What a join subtree asks of the table at its build side.
pub(crate) struct Ask<'a> {
    /// The build-side table.
    pub table: &'a QuarryTable,
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
pub(crate) fn first_ask(plan: &LogicalPlan) -> Option<Ask<'_>> {
    ask_of(plan).or_else(|| plan.inputs().iter().find_map(|input| first_ask(input)))
}

/// The join ask at `plan`, if it is a `Join` this rewrite can serve.
fn ask_of(plan: &LogicalPlan) -> Option<Ask<'_>> {
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
    // All columns the join reads from the build side: the scan's projected
    // columns, which DataFusion has already narrowed to what the join needs.
    let table_schema = datafusion::catalog::TableProvider::schema(table);
    let columns: BTreeSet<FieldId> = scan
        .projection
        .as_ref()
        .map(|proj| {
            proj.iter()
                .filter_map(|&i| {
                    let name = table_schema.field(i).name();
                    table.field_id_of(name)
                })
                .collect()
        })
        .unwrap_or_else(|| {
            // No projection: all columns.
            table_schema
                .fields()
                .iter()
                .filter_map(|f| table.field_id_of(f.name()))
                .collect()
        });
    // The keys must be among the columns.
    if !keys.is_subset(&columns) {
        return None;
    }
    Some(Ask {
        table,
        keys,
        columns,
    })
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
    let Some(source) = first_ask(plan).and_then(|ask| stored_source(&ask, approximate, stale))
    else {
        return Ok((plan.clone(), None));
    };
    let (source, id) = source;

    let mut swapped = false;
    let rewritten = plan
        .clone()
        .transform_up(|node| {
            let LogicalPlan::TableScan(scan) = &node else {
                return Ok(Transformed::no(node));
            };
            if swapped || quarry_table(scan).is_none() {
                return Ok(Transformed::no(node));
            }
            // Every field but the source, so the projection, the pushed
            // filters, and above all the projected schema are the scan's own.
            let replaced = TableScan {
                source: Arc::clone(&source),
                ..scan.clone()
            };
            swapped = true;
            Ok(Transformed::yes(LogicalPlan::TableScan(replaced)))
        })
        .map(|t| t.data)?;

    Ok((rewritten, swapped.then_some(id)))
}

/// The stored rows a join hash would serve `ask` from, as a table source.
///
/// `None` whenever the rule declines, the rows are absent, or they are not
/// table-shaped — every one of which leaves the query reading the table.
fn stored_source(
    ask: &Ask,
    approximate: bool,
    stale: bool,
) -> Option<(Arc<dyn datafusion::logical_expr::TableSource>, DerivedId)> {
    let table = ask.table;
    let query = query_of(ask, approximate, stale)?;
    let (id, decision) = table
        .registry()
        .read()
        .ok()?
        .best(&query, table.graph(), table.prices())
        .map(|candidate| (candidate.derived.id.clone(), candidate.decision.clone()))?;

    // Only a row-shaped substitute serves this. `Use` alone reaches here in
    // practice — `JoinHash` declares itself non-unionable, so the rule
    // rejects an appended-to table by name.
    let Decision::Use(Rewrite::Substitute { rollup: None, .. }) = decision else {
        return None;
    };

    let stored = table.stored(&id)?;
    if stored.is_empty() {
        return None;
    }
    // Table-shaped, or the join's column references would not resolve.
    let schema = datafusion::catalog::TableProvider::schema(table);
    if stored[0].schema() != schema {
        return None;
    }

    table.note_scan(report(table, &query, &id, ask));
    let mem = MemTable::try_new(schema, vec![stored]).ok()?;
    Some((provider_as_source(Arc::new(mem)), id))
}

/// What the serving path reports.
///
/// `approximate` is deliberately false: a join hash holds every row of the
/// build side, so the join is exact.
fn report(table: &QuarryTable, query: &Query, used: &DerivedId, ask: &Ask<'_>) -> ScanReport {
    ScanReport {
        files_read: Default::default(),
        filters: Vec::new(),
        scopes: Default::default(),
        used: vec![used.0.clone()],
        also_scanned: Default::default(),
        substituted: true,
        approximate: false,
        stale: None,
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
