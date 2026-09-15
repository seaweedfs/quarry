//! A projection: a subset of the table's columns, stored as rows.
//!
//! The taxonomy's level-4 promoted accelerator — frequently accessed
//! columns kept compact and served in place of the scan. Unlike
//! `MaterializedResult` it matches on shape rather than plan identity: any
//! query touching only the fields it covers may use it.
//!
//! The rows themselves live in the table's materialized store, supplied
//! through `QuarryTable::with_projection`; this kind only says which
//! queries they may answer.

use std::collections::BTreeSet;

use crate::cost::{Cost, PriceTable};
use crate::derived::{FieldId, Kind, Query, Refreshed, Rewrite};
use crate::snapshot::Diff;

/// Stored columns and the queries they can serve.
#[derive(Debug)]
pub struct Projection {
    /// The fields the stored columns cover.
    fields: BTreeSet<FieldId>,
    /// Stored bytes, for pricing.
    bytes: u64,
}

impl Projection {
    /// A projection covering `fields`, `bytes` on storage.
    pub fn covering(fields: impl IntoIterator<Item = FieldId>, bytes: u64) -> Self {
        Projection {
            fields: fields.into_iter().collect(),
            bytes,
        }
    }
}

impl Kind for Projection {
    fn name(&self) -> &'static str {
        "projection"
    }

    /// The query may read nothing the projection does not hold: its scan
    /// columns, its filter columns, and any aggregate's keys and measures.
    /// Predicates need no shape check — they evaluate above the scan on
    /// columns the scan already had to supply, so `projected` covers them.
    fn matches(&self, query: &Query) -> Option<Rewrite> {
        let mut needed = query.projected.clone();
        needed.extend(query.filtered_fields());
        if let Some(aggregate) = &query.aggregate {
            needed.extend(aggregate.group_by.iter().cloned());
            needed.extend(aggregate.measures.iter().filter_map(|m| m.field));
        }
        needed
            .is_subset(&self.fields)
            .then_some(Rewrite::Substitute {
                unionable: true,
                rollup: None,
            })
    }

    fn cost(&self, prices: &PriceTable) -> Cost {
        crate::kinds::price_local(prices, self.bytes)
    }

    /// Stored rows cannot be brought forward in place; the projection has
    /// to be rebuilt.
    fn refresh(&mut self, diff: &Diff) -> Refreshed {
        if diff.is_empty() {
            Refreshed::UpToDate
        } else {
            Refreshed::NeedsRebuild
        }
    }
}
