//! The kinds of derived state.
//!
//! Each kind implements [`Kind`](crate::derived::Kind)'s four methods and
//! nothing else. None of them decides whether it *may* be used — that is
//! [`Derived::may_serve`](crate::derived::Derived::may_serve)'s job — so a
//! kind cannot forget lineage, policy, or staleness.
//!
//! Kinds are added easiest-first, so that the trait is proven by a trivial
//! implementation before a demanding one arrives.

use crate::cost::{Cost, PriceTable, Tier};
use crate::place::Distance;

pub mod bitmap;
mod filter_set;
pub mod index;
pub mod projection;
pub mod result_cache;
pub mod vector;

pub use bitmap::Bitmap;
pub use filter_set::FilterSet;
pub use index::Index;
pub use projection::Projection;
pub use result_cache::ResultCache;
pub use vector::VectorIndex;

/// Price reading `bytes` of derived state.
///
/// Derived state lives on storage the engine owns and keeps warm, so it is
/// priced hot and local. If that ever stops being true — derived state spilled
/// to a remote tier, say — this is the one place that changes.
pub fn price_local(prices: &PriceTable, bytes: u64) -> Cost {
    prices.price(bytes, Tier::Hot, Distance::Local, 0.0)
}
