//! One currency for every decision.
//!
//! Plan choice, worker placement, whether to build derived state and whether
//! to retire it are all the same comparison, so they all price in the same
//! units. Keeping bytes and cpu-seconds alongside the money total matters
//! because they carry different confidence: bytes are read from immutable
//! metadata and are close to exact, while cpu-seconds are an estimate. A
//! caller that needs to know how much to trust a number needs both.

use crate::place::Distance;

/// Which storage tier bytes live in.
///
/// The distinction is economic rather than physical: cold bytes are billed on
/// retrieval, so avoiding them converts directly into money in a way that
/// avoiding hot bytes does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Immediately readable, priced per byte moved.
    Hot,
    /// Billed on retrieval, and slower.
    Cold,
}

/// What some work costs.
///
/// `usd` is the total the [`PriceTable`] arrived at; `bytes` and `cpu_seconds`
/// are kept so that a caller can see what drove it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    /// Bytes moved.
    pub bytes: u64,
    /// Cpu-seconds spent.
    pub cpu_seconds: f64,
    /// Total price in US dollars.
    pub usd: f64,
}

impl Cost {
    /// A cost of nothing: the identity for [`Add`](std::ops::Add).
    pub const ZERO: Cost = Cost {
        bytes: 0,
        cpu_seconds: 0.0,
        usd: 0.0,
    };
}

/// Combining costs is what makes plans composable: pricing a whole plan and
/// summing the prices of its parts give the same answer.
impl std::ops::Add for Cost {
    type Output = Cost;

    fn add(self, other: Cost) -> Cost {
        Cost {
            bytes: self.bytes.saturating_add(other.bytes),
            cpu_seconds: self.cpu_seconds + other.cpu_seconds,
            usd: self.usd + other.usd,
        }
    }
}

impl std::iter::Sum for Cost {
    fn sum<I: Iterator<Item = Cost>>(iter: I) -> Cost {
        iter.fold(Cost::ZERO, |a, b| a + b)
    }
}

/// What bytes and cpu-seconds cost here.
///
/// Prices are configuration, not constants: retrieval and egress pricing is
/// per-provider, tiered, and changes.
///
/// The distance multipliers deliberately blend two things that are really
/// independent — bandwidth, which drives latency, and price, which drives the
/// bill. An intra-rack transfer is slower than a node-local one but usually
/// free; a cross-cloud transfer is slower *and* billed. Collapsing them into
/// one "relative expense" per distance keeps the model to a single number and
/// is enough to order choices correctly. Splitting them is a two-field change
/// here and nowhere else, and is worth doing once a deployment cares about the
/// difference between "slow" and "expensive".
#[derive(Clone, Copy, Debug)]
pub struct PriceTable {
    /// Price of one hot, local byte.
    pub hot_byte_usd: f64,
    /// Multiplier applied to cold bytes, relative to hot.
    pub cold_multiplier: f64,
    /// Multiplier applied for [`Distance::Near`].
    pub near_multiplier: f64,
    /// Multiplier applied for [`Distance::Far`].
    pub far_multiplier: f64,
    /// Price of one cpu-second.
    pub cpu_second_usd: f64,
    /// Price of *keeping* one byte for one day.
    ///
    /// Separate from the price of moving a byte, and in different units: a
    /// rate rather than a one-off. Retirement compares what a piece of derived
    /// state has saved against what keeping it costs over some horizon, and
    /// those two cannot be compared without this.
    pub byte_day_usd: f64,
}

impl PriceTable {
    /// Price of a single byte at a given tier and distance.
    pub fn byte_usd(&self, tier: Tier, distance: Distance) -> f64 {
        let tier_multiplier = match tier {
            Tier::Hot => 1.0,
            Tier::Cold => self.cold_multiplier,
        };
        let distance_multiplier = match distance {
            Distance::Local => 1.0,
            Distance::Near => self.near_multiplier,
            Distance::Far => self.far_multiplier,
        };
        self.hot_byte_usd * tier_multiplier * distance_multiplier
    }

    /// Price keeping `bytes` for `days`.
    pub fn retention_usd(&self, bytes: u64, days: f64) -> f64 {
        bytes as f64 * self.byte_day_usd * days
    }

    /// Price reading `bytes` from `tier` at `distance`, spending
    /// `cpu_seconds` doing it.
    pub fn price(&self, bytes: u64, tier: Tier, distance: Distance, cpu_seconds: f64) -> Cost {
        Cost {
            bytes,
            cpu_seconds,
            usd: bytes as f64 * self.byte_usd(tier, distance) + cpu_seconds * self.cpu_second_usd,
        }
    }
}

impl Default for PriceTable {
    /// Prices in the shape of a public cloud: cold retrieval and crossing a
    /// boundary both cost real money, and local hot reads are nearly free.
    ///
    /// The absolute values matter less than the ratios, which are what order
    /// the optimizer's choices. Every distance is strictly more expensive than
    /// the one inside it, including `Near`, so that a scheduler with a choice
    /// between a node-local and a rack-local replica prefers the closer one.
    fn default() -> Self {
        PriceTable {
            hot_byte_usd: 1e-11,
            cold_multiplier: 1_000.0,
            near_multiplier: 2.0,
            far_multiplier: 100.0,
            cpu_second_usd: 1e-5,
            byte_day_usd: 7e-13,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn cold_and_far_bytes_cost_more_than_hot_local_ones() {
        let p = PriceTable::default();
        let hot_local = p.byte_usd(Tier::Hot, Distance::Local);
        let cold_local = p.byte_usd(Tier::Cold, Distance::Local);
        let hot_far = p.byte_usd(Tier::Hot, Distance::Far);
        let cold_far = p.byte_usd(Tier::Cold, Distance::Far);

        assert!(cold_local > hot_local, "cold must cost more than hot");
        assert!(hot_far > hot_local, "far must cost more than local");
        assert!(cold_far > cold_local);
        assert!(cold_far > hot_far);
    }

    #[test]
    fn distance_prices_strictly_increase_by_default() {
        // Strict, so that a scheduler choosing between a node-local and a
        // rack-local replica has a reason to prefer the closer one.
        let p = PriceTable::default();
        let at = |d| p.byte_usd(Tier::Hot, d);
        assert!(at(Distance::Local) < at(Distance::Near));
        assert!(at(Distance::Near) < at(Distance::Far));
    }

    #[test]
    fn an_on_prem_table_may_price_every_distance_the_same() {
        // Nothing requires the ordering to be strict; a deployment with no
        // egress billing and a flat fabric can say so.
        let flat = PriceTable {
            near_multiplier: 1.0,
            far_multiplier: 1.0,
            ..PriceTable::default()
        };
        let at = |d| flat.byte_usd(Tier::Hot, d);
        assert_eq!(at(Distance::Local), at(Distance::Far));
    }

    #[test]
    fn summing_parts_equals_costing_the_whole() {
        let p = PriceTable::default();
        let whole = p.price(3 * GB, Tier::Hot, Distance::Local, 3.0);
        let parts: Cost = (0..3)
            .map(|_| p.price(GB, Tier::Hot, Distance::Local, 1.0))
            .sum();

        assert_eq!(parts.bytes, whole.bytes);
        assert!((parts.cpu_seconds - whole.cpu_seconds).abs() < f64::EPSILON);
        assert!(
            (parts.usd - whole.usd).abs() < 1e-12,
            "parts {} != whole {}",
            parts.usd,
            whole.usd
        );
    }

    #[test]
    fn zero_is_the_additive_identity() {
        let p = PriceTable::default();
        let c = p.price(GB, Tier::Cold, Distance::Far, 2.5);
        assert_eq!(c + Cost::ZERO, c);
        assert_eq!(Cost::ZERO + c, c);
    }

    #[test]
    fn byte_counts_saturate_rather_than_wrap() {
        let huge = Cost {
            bytes: u64::MAX,
            ..Cost::ZERO
        };
        let one = Cost {
            bytes: 1,
            ..Cost::ZERO
        };
        assert_eq!((huge + one).bytes, u64::MAX);
    }

    #[test]
    fn free_prices_still_order_by_nothing() {
        // An on-prem table with no monetary cost prices everything at zero;
        // callers must then order by bytes, not by usd.
        let free = PriceTable {
            hot_byte_usd: 0.0,
            cpu_second_usd: 0.0,
            ..PriceTable::default()
        };
        let a = free.price(GB, Tier::Cold, Distance::Far, 10.0);
        let b = free.price(1, Tier::Hot, Distance::Local, 0.0);
        assert_eq!(a.usd, b.usd);
        assert!(a.bytes > b.bytes);
    }
}
