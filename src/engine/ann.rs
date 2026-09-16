//! Serving top-k nearest-neighbour queries from a vector index.
//!
//! `TableProvider::scan` never sees the `ORDER BY` or the `LIMIT` — both sit
//! above the scan in the logical plan — so a vector index cannot be served
//! the way an index or a result cache is. This is the same plan-level
//! treatment [`cube`](super::cube) gives aggregates: find the shape, let the
//! table describe it exactly, and let
//! [`Derived::may_serve`](crate::derived::Derived::may_serve) decide.
//!
//! # Only the leaf is replaced
//!
//! The rewrite swaps the `TableScan`'s *source* and changes nothing else —
//! not the projections DataFusion inserts, not the filters, and above all not
//! the `Sort` or the `Limit`:
//!
//! ```text
//! Limit: fetch=3                        Limit: fetch=3
//!   Projection: id                        Projection: id
//!     Sort: distance(embedding, [..])  ->    Sort: distance(embedding, [..])
//!       Projection: id, embedding              Projection: id, embedding
//!         TableScan: docs                        TableScan: docs (stored rows)
//! ```
//!
//! That is what makes this exact, and it is a stronger argument than checking
//! the shape carefully would be. The distances that decide the answer are
//! recomputed by the same expression the query wrote, over rows with the same
//! schema; the plan cannot disagree with itself because it is the same plan.
//! A flat index holds every row, so the substituted relation *is* the table —
//! the saving is that the rows are local instead of in object storage.
//!
//! An approximate index — one storing candidates rather than every row —
//! would have to set [`ScanReport::approximate`](super::ScanReport) and
//! require the `quarry.approximate` opt-in, because candidates can miss a
//! true neighbour. This one cannot.
//!
//! # What an append does
//!
//! Nothing here repairs staleness, and a top-k is why: the k nearest of
//! (stored ∪ appended) is not the k nearest of each side, so reading the
//! residual alongside would need a re-ranking merge rather than a union.
//! [`VectorIndex`](crate::kinds::VectorIndex) therefore declares itself
//! non-unionable, so the rule rejects an appended-to table by name and the
//! query reads the table — correct, and slower until the optimizer's refresh
//! step rebuilds the index. The rebuild lands on the same id, which is keyed
//! on what the index covers rather than on the snapshot, so it replaces
//! rather than accumulates.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::datasource::{DefaultTableSource, MemTable, provider_as_source};
use datafusion::error::Result as DfResult;
use datafusion::logical_expr::{Expr, FetchType, LogicalPlan, SkipType, SortExpr, TableScan};

use crate::derived::{Decision, DerivedId, Metric, Nearest, Query, Rewrite};

use super::{QuarryTable, ScanReport};

/// What a top-k subtree asks of the table at its leaf.
pub(crate) struct Ask<'a> {
    pub table: &'a QuarryTable,
    /// The ask, reduced to what a vector index matches on.
    pub nearest: Nearest,
    /// Filters between the sort and the scan, for the ask's identity and the
    /// report. They stay in the plan — nothing replays them.
    pub filters: Vec<Expr>,
}

/// The top-k nearest-neighbour search in `plan`, if it is shaped like one a
/// vector index could serve.
///
/// Recognised: `Limit` over `Sort` over a `TableScan`, with row-preserving
/// nodes allowed in between — DataFusion inserts a `Projection` on each side
/// of the sort for `ORDER BY <expression>`. The sort's *leading* key must be
/// an ascending distance function on a `FixedSizeList` column of that table.
///
/// # What each refusal protects
///
/// ```text
/// ascending only    DESC asks for the *farthest* rows, which a nearest-
///                   neighbour index has no claim to
/// no offset         OFFSET k reaches past the k-th row, so a candidate-
///                   returning index could not serve it; refused here so
///                   the shape stays one thing rather than two
/// a bare column     ORDER BY distance(f(v), q) sorts on something the
///                   index was not built over
/// row-preserving    a join or an aggregate below the sort means the rows
///                   being ranked are not this table's
/// ```
///
/// # Why trailing sort keys are allowed
///
/// `ORDER BY distance(v, q), id` — a deterministic tie-break, and the usual
/// way to make a top-k reproducible. Only the leading key decides which index
/// applies; the rest are safe for the reason the whole rewrite is safe, which
/// is that the `Sort` node is left exactly as written and runs over rows of
/// the same schema. Requiring a single key would refuse the common form for
/// no correctness gain.
pub(crate) fn ask_of(plan: &LogicalPlan) -> Option<Ask<'_>> {
    let LogicalPlan::Limit(limit) = plan else {
        return None;
    };
    // Both must be plain literals, or the k is not known at plan time — and
    // any offset reaches past the k-th row, which a top-k has no claim to.
    // DataFusion's own accessors rather than a hand match: they normalise
    // `OFFSET NULL` to zero and `LIMIT NULL` to no limit.
    match limit.get_skip_type() {
        Ok(SkipType::Literal(0)) => {}
        _ => return None,
    }
    let k = match limit.get_fetch_type() {
        Ok(FetchType::Literal(Some(k))) => k as u64,
        _ => return None,
    };

    let LogicalPlan::Sort(sort) = descend(&limit.input)? else {
        return None;
    };
    let [
        SortExpr {
            expr,
            asc: true,
            nulls_first: _,
        },
        ..,
    ] = sort.expr.as_slice()
    else {
        return None;
    };

    // A `Sort` may carry its own fetch; a smaller one bounds the answer more
    // tightly than the limit above, and taking the larger would claim to
    // serve rows the query did not ask for.
    let k = sort.fetch.map_or(k, |fetch| k.min(fetch as u64));

    let (column, metric) = distance_of(expr)?;

    let mut filters = Vec::new();
    let scan = scan_under(&sort.input, &mut filters)?;
    let table = quarry_table(scan)?;

    // The dimension comes from the table's own schema, not the query's
    // literal: it is what an index would have been built over.
    let schema = datafusion::catalog::TableProvider::schema(table);
    let position = schema.index_of(&column).ok()?;
    let dimension = super::distance::dimension_of(schema.field(position).data_type())?;
    let field = table.field_id_of(&column)?;

    Some(Ask {
        table,
        nearest: Nearest {
            field,
            metric,
            dimension,
            k,
        },
        filters,
    })
}

/// Through nodes that pass their input's rows along unchanged.
///
/// `ORDER BY <expression>` makes DataFusion project the sort key alongside the
/// output columns, so a `Projection` sits between the `Limit` and the `Sort`
/// and another between the `Sort` and the scan. Neither adds or removes a
/// row, which is all this needs to be true.
fn descend(plan: &LogicalPlan) -> Option<&LogicalPlan> {
    match plan {
        LogicalPlan::Projection(projection) => descend(&projection.input),
        LogicalPlan::SubqueryAlias(alias) => descend(&alias.input),
        other => Some(other),
    }
}

/// The `TableScan` under `plan`, collecting the filters passed on the way.
///
/// Only row-preserving nodes are walked through. A `Join`, `Aggregate`,
/// `Union`, `Distinct` or `Window` returns `None`: the rows being ranked
/// would not be the table's own, so no index over the table can serve them.
fn scan_under<'a>(plan: &'a LogicalPlan, filters: &mut Vec<Expr>) -> Option<&'a TableScan> {
    match plan {
        LogicalPlan::Filter(filter) => {
            super::cube::split_and(&filter.predicate, filters);
            scan_under(&filter.input, filters)
        }
        LogicalPlan::Projection(projection) => scan_under(&projection.input, filters),
        LogicalPlan::SubqueryAlias(alias) => scan_under(&alias.input, filters),
        LogicalPlan::TableScan(scan) => {
            for expr in &scan.filters {
                super::cube::split_and(expr, filters);
            }
            Some(scan)
        }
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

/// The first top-k ask anywhere in `plan`, or none.
///
/// `ask_of` matches only at a `Limit`; reporting wants the ask wherever it
/// sits, since a `Projection` usually sits on top of it.
pub(crate) fn first_ask(plan: &LogicalPlan) -> Option<Ask<'_>> {
    if let Some(ask) = ask_of(plan) {
        return Some(ask);
    }
    plan.inputs().iter().find_map(|input| first_ask(input))
}

/// The column and metric a distance expression names, if it is one.
///
/// The first argument must be a bare column and the second must *not* be —
/// `distance(a, b)` between two columns is a row-wise computation, not a
/// probe against one query vector, and no index answers it.
fn distance_of(expr: &Expr) -> Option<(String, Metric)> {
    let Expr::ScalarFunction(function) = expr else {
        return None;
    };
    let metric = super::distance::metric_of(function.func.name())?;
    let [Expr::Column(column), query] = function.args.as_slice() else {
        return None;
    };
    if matches!(query, Expr::Column(_)) {
        return None;
    }
    Some((column.name().to_owned(), metric))
}

/// The [`Query`] an [`Ask`] reduces to — the identity a vector index matches.
pub(crate) fn query_of(ask: &Ask, approximate: bool) -> Option<Query> {
    ask.table
        .nearest_query(&ask.nearest, &ask.filters, approximate)
}

/// Rewrite `plan` to read a vector index wherever one is admissible.
///
/// Returns the rewritten plan and the id of what served it, for reporting.
pub(crate) fn rewrite(
    plan: &LogicalPlan,
    approximate: bool,
) -> DfResult<(LogicalPlan, Option<DerivedId>)> {
    // Recognised at the `Limit`, but rewritten at the leaf — so the shape is
    // checked once and then the source is swapped where it lives.
    let Some(source) = first_ask(plan).and_then(|ask| stored_source(&ask, approximate)) else {
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
            // A plan built on the old schema keeps resolving against the new.
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

/// The stored rows a vector index would serve `ask` from, as a table source.
///
/// `None` whenever the rule declines, the rows are absent, or they are not
/// table-shaped — every one of which leaves the query reading the table.
fn stored_source(
    ask: &Ask,
    approximate: bool,
) -> Option<(Arc<dyn datafusion::logical_expr::TableSource>, DerivedId)> {
    let table = ask.table;
    let query = query_of(ask, approximate)?;
    let (id, decision) = table
        .registry()
        .read()
        .ok()?
        .best(&query, table.graph(), table.prices())
        .map(|candidate| (candidate.derived.id.clone(), candidate.decision.clone()))?;

    // A rollup holds partial aggregates, not rows a distance can be computed
    // against; only a row-shaped substitute serves this. `Use` alone reaches
    // here in practice — `VectorIndex` declares itself non-unionable, so the
    // rule rejects an appended-to table by name rather than leaving this to
    // decline it quietly.
    let Decision::Use(Rewrite::Substitute { rollup: None, .. }) = decision else {
        return None;
    };

    let stored = table.stored(&id)?;
    if stored.is_empty() {
        return None;
    }
    // Table-shaped, or the sort's column reference and the projections above
    // would not resolve against them.
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
/// `approximate` is deliberately false: a flat index holds every row and the
/// sort above re-ranks them, so the answer is exact. The kind that stores
/// candidates instead is the one that must set it.
pub(crate) fn report(
    table: &QuarryTable,
    query: &Query,
    used: &DerivedId,
    ask: &Ask<'_>,
) -> ScanReport {
    ScanReport {
        files_read: Default::default(),
        filters: ask
            .filters
            .iter()
            .filter_map(|expr| {
                let sql = datafusion::sql::unparser::expr_to_sql(expr)
                    .ok()?
                    .to_string();
                Some(crate::workload::FilterAsk {
                    table: table.table_id().clone(),
                    filter: table.canonical_filter(expr),
                    sql,
                })
            })
            .collect(),
        scopes: Default::default(),
        used: vec![used.0.clone()],
        also_scanned: Default::default(),
        substituted: true,
        approximate: false,
        plan_hash: query.plan_hash,
        bytes_if_full_scan: table.live_bytes(),
        fingerprint: crate::workload::Fingerprint::of(query),
        plan: query.plan.clone(),
        aggregate: None,
        // A top-k is not a text match; nothing here is one.
        text: Vec::new(),
        nearest: Some(crate::workload::NearestAsk {
            table: table.table_id().clone(),
            nearest: ask.nearest.clone(),
        }),
        join: None,
    }
}
