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
    /// A table derived from the numbers on a storage bill.
    ///
    /// Every other constructor here is this one with published rates filled
    /// in. Taking the inputs a user can actually read off an invoice means
    /// they never have to understand the multipliers to replace them.
    ///
    /// `average_read_bytes` is how a per-request charge becomes a per-byte
    /// one, and it matters: the same GET fee spread over a 4 KB footer read is
    /// 250 times the per-byte cost of spreading it over a 1 MB column chunk.
    pub fn from_rates(
        storage_usd_per_gb_month: f64,
        get_usd_per_1k_requests: f64,
        average_read_bytes: f64,
        egress_usd_per_gb: f64,
        cold_retrieval_usd_per_gb: f64,
    ) -> Self {
        const GB: f64 = 1e9;
        let per_request = get_usd_per_1k_requests / 1_000.0;
        let hot_local = per_request / average_read_bytes.max(1.0);

        PriceTable {
            hot_byte_usd: hot_local,
            cold_multiplier: (cold_retrieval_usd_per_gb / GB + hot_local) / hot_local,
            // Transfer within a region is not billed, so a byte from the next
            // rack costs exactly what a local one does. See the note on
            // `near_multiplier`.
            near_multiplier: 1.0,
            far_multiplier: (egress_usd_per_gb / GB + hot_local) / hot_local,
            cpu_second_usd: 1.1e-5,
            byte_day_usd: storage_usd_per_gb_month / (GB * 30.0),
        }
    }

    /// AWS S3 Standard, read by compute in the same region.
    ///
    /// Published US East (N. Virginia) rates, checked January 2026:
    ///
    /// ```text
    /// storage            $0.023 per GB-month
    /// GET requests       $0.0004 per 1,000
    /// transfer to EC2    free, same region, any availability zone
    /// cold retrieval     $0.01 per GB  (Standard-IA / One Zone-IA)
    /// ```
    ///
    /// Rates move; this is a starting point, not a promise. `from_rates` is
    /// there to be given current ones.
    pub fn aws_s3_same_region() -> Self {
        PriceTable::from_rates(0.023, 0.0004, 1e6, 0.0, 0.01)
    }

    /// AWS S3 Standard, read across regions at about $0.02 per GB.
    pub fn aws_s3_cross_region() -> Self {
        PriceTable::from_rates(0.023, 0.0004, 1e6, 0.02, 0.01)
    }

    /// AWS S3 Standard, served to the public internet at $0.09 per GB.
    ///
    /// The most expensive common case by a wide margin, and the one where
    /// pruning pays for itself fastest.
    pub fn aws_s3_internet() -> Self {
        PriceTable::from_rates(0.023, 0.0004, 1e6, 0.09, 0.01)
    }

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
    /// AWS S3 read in the same region.
    ///
    /// Derived rather than invented, which the previous default was not. The
    /// guesses it replaces, against rates published in January 2026:
    ///
    /// ```text
    ///                   guessed    derived
    /// hot_byte_usd       1e-11     4.0e-13     25x too high
    /// cold_multiplier    1000      26          38x too high
    /// near_multiplier    2.0       1.0         in-region transfer is free
    /// far_multiplier     100       51          about right
    /// byte_day_usd       7e-13     7.67e-13    about right
    /// ```
    ///
    /// The `cold_multiplier` error was the one worth catching. At 1000x, any
    /// decision priced in money would refuse to read cold data under
    /// practically any circumstances; the real ratio is about 26, which is a
    /// reason to prefer hot data rather than a reason to never touch cold.
    fn default() -> Self {
        PriceTable::aws_s3_same_region()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn cold_bytes_cost_more_than_hot_ones() {
        let p = PriceTable::default();
        assert!(
            p.byte_usd(Tier::Cold, Distance::Local) > p.byte_usd(Tier::Hot, Distance::Local),
            "a retrieval fee is a real charge"
        );
    }

    /// Distance is free within a region, and the design assumed otherwise.
    ///
    /// This test used to assert that every distance costs strictly more than
    /// the one inside it, on the reasoning that a scheduler needs a reason to
    /// prefer a closer replica. Calibration against published rates showed the
    /// premise is false for the dominant deployment: AWS bills nothing for
    /// transfer from S3 to compute in the same region, whatever availability
    /// zone either is in.
    ///
    /// So `Local`, `Near` and `Far` are all the same price by default, and the
    /// reason to prefer local data there is **latency**, which `Cost` does not
    /// model. That is a real gap, and pricing distance as though money were the
    /// reason would have hidden it behind a number nobody could defend.
    #[test]
    fn distance_is_free_within_a_region() {
        let p = PriceTable::default();
        let at = |d| p.byte_usd(Tier::Hot, d);
        assert_eq!(at(Distance::Local), at(Distance::Near));
        assert_eq!(at(Distance::Near), at(Distance::Far));
    }

    #[test]
    fn distance_costs_money_once_bytes_leave_the_region() {
        for p in [
            PriceTable::aws_s3_cross_region(),
            PriceTable::aws_s3_internet(),
        ] {
            let at = |d| p.byte_usd(Tier::Hot, d);
            assert!(
                at(Distance::Far) > at(Distance::Local),
                "egress is billed, so far must cost more"
            );
        }

        // And the internet is far dearer than another region, which is what
        // makes serving queries out of the wrong place expensive.
        assert!(
            PriceTable::aws_s3_internet().far_multiplier
                > PriceTable::aws_s3_cross_region().far_multiplier
        );
    }

    #[test]
    fn a_deployment_can_price_distance_however_it_likes() {
        // On-premises, cross-rack traffic contends for a shared uplink even
        // though nobody sends an invoice for it. A deployment that wants
        // locality to weigh on cost can say so, and nothing here prevents it.
        let contended = PriceTable {
            near_multiplier: 2.0,
            far_multiplier: 10.0,
            ..PriceTable::default()
        };
        let at = |d| contended.byte_usd(Tier::Hot, d);
        assert!(at(Distance::Local) < at(Distance::Near));
        assert!(at(Distance::Near) < at(Distance::Far));
    }

    /// The rates are checkable against the sources they came from.
    #[test]
    fn the_published_rates_are_what_was_derived_from() {
        let p = PriceTable::aws_s3_same_region();

        // $0.023 per GB-month over thirty days.
        assert!((p.byte_day_usd - 0.023 / (1e9 * 30.0)).abs() < 1e-18);
        // $0.0004 per 1,000 GETs, spread over a 1 MB read.
        assert!((p.hot_byte_usd - 4e-13).abs() < 1e-16);
        // $0.01 per GB retrieval on top of that.
        assert!(
            (p.cold_multiplier - 26.0).abs() < 0.5,
            "cold_multiplier {}",
            p.cold_multiplier
        );

        // Cross-region egress at $0.02 per GB.
        assert!(
            (PriceTable::aws_s3_cross_region().far_multiplier - 51.0).abs() < 1.0,
            "far_multiplier {}",
            PriceTable::aws_s3_cross_region().far_multiplier
        );
    }

    #[test]
    fn a_smaller_average_read_makes_every_byte_dearer() {
        // A per-request fee spread over a 4 KB footer read costs far more per
        // byte than the same fee over a 1 MB column chunk. Worth exposing,
        // since it is the one input a caller is likely to get wrong.
        let big = PriceTable::from_rates(0.023, 0.0004, 1e6, 0.0, 0.01);
        let small = PriceTable::from_rates(0.023, 0.0004, 4e3, 0.0, 0.01);
        assert!(small.hot_byte_usd > big.hot_byte_usd * 100.0);
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
