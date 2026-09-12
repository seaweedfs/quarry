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
    /// A cost of nothing: the identity for [`Cost::add`].
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
/// per-provider, tiered, and changes. An on-premises deployment sets the
/// distance prices to zero and still gets useful *relative* ordering, because
/// the latency asymmetry remains encoded in the multipliers.
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
    /// Prices in the shape of a public cloud: cold retrieval and cross-boundary
    /// traffic both cost real money, and local hot reads are nearly free.
    ///
    /// The absolute values matter less than the ratios, which are what order
    /// the optimizer's choices.
    fn default() -> Self {
        PriceTable {
            hot_byte_usd: 1e-11,
            cold_multiplier: 1_000.0,
            near_multiplier: 1.0,
            far_multiplier: 100.0,
            cpu_second_usd: 1e-5,
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
    fn distance_prices_are_monotonic() {
        let p = PriceTable::default();
        let at = |d| p.byte_usd(Tier::Hot, d);
        assert!(at(Distance::Local) <= at(Distance::Near));
        assert!(at(Distance::Near) <= at(Distance::Far));
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
