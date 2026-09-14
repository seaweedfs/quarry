//! Serving aggregate queries from cubes.
//!
//! `TableProvider::scan` never sees the group-by — aggregation sits above the
//! scan in the logical plan — so a cube cannot be served the way an index or a
//! result cache is. This is the plan-level half of the same rule: find
//! `Aggregate` over `TableScan`, let the table describe the shape exactly,
//! and let [`Derived::may_serve`](crate::derived::Derived::may_serve) decide.
//!
//! The rewrite replaces the whole `Aggregate -> Filter* -> TableScan` subtree
//! with the same structure over the cube's stored partials, and names the
//! cube's scan after the table it stands in for, so every expression the
//! plan above it wrote — `events.day`, `sum(events.bytes)` — still resolves:
//!
//! ```text
//! Aggregate(sum(events.bytes), by day)       Aggregate(sum("sum(bytes)"),
//!   Filter(day >= '2025-01-01')        ->      by events.day)
//!     TableScan(events)                        Filter(events.day >= ...)
//!                                                TableScan(cube as events)
//! ```
//!
//! Keys and filters pass through verbatim: a stored key column keeps the
//! table's own name. Only a measure changes — its argument becomes the stored
//! partial column, its function the rollup of itself (a stored `count` is
//! summed).

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::common::TableReference;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::datasource::{DefaultTableSource, MemTable, provider_as_source};
use datafusion::error::Result as DfResult;
use datafusion::functions_aggregate::expr_fn;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Operator};

use crate::derived::{AggFunc, Decision, DerivedId, Measure, Query, Rewrite};
use crate::workload::AggregateAsk;

use super::{QuarryTable, Rollup, ScanReport};

/// What an `Aggregate` node asks of the table it sits above.
pub(crate) struct Ask<'a> {
    pub table: &'a QuarryTable,
    /// The name the plan knows the table by — the rewritten scan keeps it so
    /// qualified references above still resolve.
    pub name: TableReference,
    pub group: &'a [Expr],
    pub aggr: &'a [Expr],
    pub filters: Vec<Expr>,
}

/// The aggregate over a [`QuarryTable`] in `plan`, if it is shaped like one a
/// cube could serve.
///
/// Recognised: `Aggregate` over any number of `Filter`s over `TableScan`.
/// Anything in between — a join, a projection, a second table — means `None`,
/// and the plan runs as written.
pub(crate) fn ask_of(plan: &LogicalPlan) -> Option<Ask<'_>> {
    let LogicalPlan::Aggregate(aggregate) = plan else {
        return None;
    };
    let mut filters = Vec::new();
    let mut input = aggregate.input.as_ref();
    loop {
        match input {
            LogicalPlan::Filter(filter) => {
                split_and(&filter.predicate, &mut filters);
                input = &filter.input;
            }
            LogicalPlan::TableScan(scan) => {
                for expr in &scan.filters {
                    split_and(expr, &mut filters);
                }
                let table = scan
                    .source
                    .as_any()
                    .downcast_ref::<DefaultTableSource>()?
                    .table_provider
                    .as_any()
                    .downcast_ref::<QuarryTable>()?;
                return Some(Ask {
                    table,
                    name: scan.table_name.clone(),
                    group: &aggregate.group_expr,
                    aggr: &aggregate.aggr_expr,
                    filters,
                });
            }
            _ => return None,
        }
    }
}

/// `a AND b AND c` is three restrictions, not one; a cube answers for each
/// conjunct separately.
fn split_and(expr: &Expr, into: &mut Vec<Expr>) {
    if let Expr::BinaryExpr(binary) = expr {
        if binary.op == Operator::And {
            split_and(&binary.left, into);
            split_and(&binary.right, into);
            return;
        }
    }
    into.push(expr.clone());
}

/// The first servable aggregate anywhere in `plan`, or none.
///
/// `ask_of` matches only a root `Aggregate`; reporting wants to find the ask
/// whether or not a `Projection` sits on top of it.
pub(crate) fn first_ask(plan: &LogicalPlan) -> Option<Ask<'_>> {
    if let Some(ask) = ask_of(plan) {
        return Some(ask);
    }
    plan.inputs().iter().find_map(|input| first_ask(input))
}

/// The [`Query`] an [`Ask`] reduces to — the identity a cube matches on.
pub(crate) fn query_of(ask: &Ask) -> Option<Query> {
    ask.table.aggregate_query(ask.group, ask.aggr, &ask.filters)
}

/// Rewrite `plan` to read a cube wherever one is admissible.
///
/// Returns the rewritten plan and the id of what served it, for reporting.
/// `None` means no aggregate in the plan matched a cube, for any of the
/// reasons [`ask_of`] or the rule can produce.
pub(crate) fn rewrite(plan: &LogicalPlan) -> DfResult<(LogicalPlan, Option<DerivedId>)> {
    let mut served = None;
    let rewritten = plan
        .clone()
        .transform_up(|node| {
            if served.is_some() {
                return Ok(Transformed::no(node));
            }
            let Some(ask) = ask_of(&node) else {
                return Ok(Transformed::no(node));
            };
            match substitute(&ask) {
                Some(Ok((subtree, id))) => {
                    served = Some(id);
                    Ok(Transformed::yes(subtree))
                }
                Some(Err(e)) => Err(e),
                None => Ok(Transformed::no(node)),
            }
        })
        .map(|t| t.data)?;
    Ok((rewritten, served))
}

/// The rewritten subtree for `ask`, if the rule admits a cube for it.
fn substitute(ask: &Ask) -> Option<DfResult<(LogicalPlan, DerivedId)>> {
    let table = ask.table;
    let query = query_of(ask)?;
    let (id, decision) = table
        .registry()
        .read()
        .ok()?
        .best(&query, table.graph(), table.prices())
        .map(|candidate| (candidate.derived.id.clone(), candidate.decision.clone()))?;
    let Decision::Use(Rewrite::Substitute {
        rollup: Some(spec), ..
    }) = decision
    else {
        return None;
    };
    // Record before the borrow escapes: the report lands on the table while
    // `ask`'s references are still in scope.
    table.note_scan(report(table, &query, &id, &filter_sql(ask)));

    let fields = spec.group_by.len() + spec.measures.iter().filter(|m| m.field.is_some()).count();
    let names: BTreeMap<_, _> = spec
        .group_by
        .iter()
        .copied()
        .chain(spec.measures.iter().filter_map(|m| m.field))
        .filter_map(|field| Some((field, table.column_of(field)?.to_owned())))
        .collect();
    if names.len() != fields {
        return None;
    }
    let rollup = Rollup::new(spec, names);
    let stored = table.stored(&id)?.to_vec();
    if stored.is_empty() {
        return None;
    }
    let mem = match MemTable::try_new(stored[0].schema(), vec![stored]) {
        Ok(mem) => mem,
        Err(e) => return Some(Err(e)),
    };

    // Only measures change. Each asked measure becomes the rollup of its
    // stored partials, aliased to the name the original expression carried so
    // the plan above resolves the same field it asked for.
    let measures = ask
        .aggr
        .iter()
        .map(|expr| {
            let measure = table.measure(expr)?;
            let stored = rollup.measure_name(&measure)?;
            let name = expr.name_for_alias().ok()?;
            Some(reaggregate(&measure, &stored).alias(name))
        })
        .collect::<Option<Vec<_>>>()?;

    let build = (|| -> DfResult<LogicalPlan> {
        let mut builder =
            LogicalPlanBuilder::scan(ask.name.clone(), provider_as_source(Arc::new(mem)), None)?;
        for filter in &ask.filters {
            builder = builder.filter(filter.clone())?;
        }
        builder.aggregate(ask.group.to_vec(), measures)?.build()
    })();

    Some(build.map(|plan| (plan, id)))
}

/// The re-aggregation of a stored partial: a stored `count` is summed,
/// everything else combines through its own function.
fn reaggregate(measure: &Measure, stored: &str) -> Expr {
    let column = Expr::Column(datafusion::common::Column::new_unqualified(stored));
    match measure.func.rollup() {
        AggFunc::Sum => expr_fn::sum(column),
        AggFunc::Min => expr_fn::min(column),
        AggFunc::Max => expr_fn::max(column),
        // `AggFunc::rollup` never yields Count: a count combines by summing.
        AggFunc::Count => unreachable!("a count's rollup is a sum"),
    }
}

/// The filters back as SQL — what a cube build would bake in.
pub(crate) fn filter_sql(ask: &Ask) -> Vec<String> {
    ask.filters
        .iter()
        .filter_map(|expr| datafusion::sql::unparser::expr_to_sql(expr).ok())
        .map(|sql| sql.to_string())
        .collect()
}

/// What the serving path reports: the same [`ScanReport`] a scan leaves, with
/// no files read because none were.
pub(crate) fn report(
    table: &QuarryTable,
    query: &Query,
    used: &DerivedId,
    filter_sql: &[String],
) -> ScanReport {
    ScanReport {
        files_read: Default::default(),
        used: vec![used.0.clone()],
        also_scanned: Default::default(),
        substituted: true,
        plan_hash: query.plan_hash,
        bytes_if_full_scan: table.live_bytes(),
        fingerprint: crate::workload::Fingerprint::of(query),
        plan: query.plan.clone(),
        aggregate: query
            .aggregate
            .clone()
            .zip(query.plan.clone())
            .map(|(spec, plan)| AggregateAsk {
                table: table.table_id().clone(),
                plan,
                spec,
                filter_sql: filter_sql.to_vec(),
            }),
    }
}
