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
/// `usd` is the total the [`PriceTable`] arrived at; the rest are kept so that
/// a caller can see what drove it, and because they carry different confidence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    /// Bytes moved.
    pub bytes: u64,
    /// Cpu-seconds spent computing.
    pub cpu_seconds: f64,
    /// Seconds spent waiting for bytes to arrive.
    ///
    /// Separate from `cpu_seconds` because it is not work: a worker blocked on
    /// a round trip is idle, and the two have very different remedies. It is
    /// priced at the same rate all the same, which is the point — see
    /// [`PriceTable::wait_seconds`].
    pub wait_seconds: f64,
    /// Total price in US dollars.
    pub usd: f64,
}

impl Cost {
    /// A cost of nothing: the identity for [`Add`](std::ops::Add).
    pub const ZERO: Cost = Cost {
        bytes: 0,
        cpu_seconds: 0.0,
        wait_seconds: 0.0,
        usd: 0.0,
    };
}

/// Combining costs is what makes plans composable.
///
/// Bytes and seconds add exactly. Money does *not* reproduce the price of the
/// whole from the prices of its parts, and should not: three reads pay three
/// round trips where one read pays one. See
/// `splitting_a_read_costs_extra_round_trips`.
impl std::ops::Add for Cost {
    type Output = Cost;

    fn add(self, other: Cost) -> Cost {
        Cost {
            bytes: self.bytes.saturating_add(other.bytes),
            cpu_seconds: self.cpu_seconds + other.cpu_seconds,
            wait_seconds: self.wait_seconds + other.wait_seconds,
            usd: self.usd + other.usd,
        }
    }
}

impl std::iter::Sum for Cost {
    fn sum<I: Iterator<Item = Cost>>(iter: I) -> Cost {
        iter.fold(Cost::ZERO, |a, b| a + b)
    }
}

/// How quickly bytes arrive from a given distance.
///
/// Two numbers because a transfer has two parts that scale differently: a
/// round trip that a large read amortises away, and a rate that it does not.
/// Reading one megabyte from S3 spends more than half its time waiting; the
/// same round trip against a hundred megabytes is noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    /// Seconds before the first byte arrives.
    pub first_byte_seconds: f64,
    /// Bytes per second once they are flowing.
    pub bytes_per_second: f64,
}

impl Link {
    /// How long `bytes` take to arrive.
    pub fn seconds(&self, bytes: u64) -> f64 {
        self.first_byte_seconds + bytes as f64 / self.bytes_per_second.max(f64::MIN_POSITIVE)
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
    /// How fast bytes arrive from the same node — a cache hit, or colocated
    /// storage.
    pub local_link: Link,
    /// How fast bytes arrive from the same region.
    pub near_link: Link,
    /// How fast bytes arrive from another region, or over the internet.
    pub far_link: Link,
    /// How many reads this deployment has in flight at once.
    ///
    /// Waiting is the one part of a cost that does *not* add up over reads: a
    /// scan issuing sixteen requests in parallel waits once, not sixteen
    /// times. [`crate::budget::Meter`] divides accumulated waiting by this.
    ///
    /// # Why the default is one
    ///
    /// There is no conservative default, because the two consumers want
    /// opposite errors. A budget is safest assuming *no* overlap, since
    /// understating a cost lets a query overspend; a build decision is safest
    /// assuming *full* overlap, since overstating a saving is what builds
    /// indexes that do not pay. Guessing therefore cannot be safe in both, so
    /// the default is the literal truth for a caller with no engine — one read
    /// at a time — and [`crate::engine::Quarry`] replaces it with DataFusion's
    /// own `target_partitions` rather than a guess.
    pub concurrent_reads: f64,
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
            // Measured figures rather than derived ones, since latency is not
            // on a price list. See `aws_s3_same_region`.
            local_link: Link {
                first_byte_seconds: 100e-6,
                bytes_per_second: 2e9,
            },
            near_link: Link {
                first_byte_seconds: 20e-3,
                bytes_per_second: 90e6,
            },
            far_link: Link {
                first_byte_seconds: 80e-3,
                bytes_per_second: 40e6,
            },
            concurrent_reads: 1.0,
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
    /// Latency and throughput are measured, not published, and come from
    /// widely reproduced benchmarks:
    ///
    /// ```text
    /// same region     ~20 ms to first byte, ~90 MB/s on one stream
    /// same node       ~100 us, ~2 GB/s   (a cache hit, or local NVMe)
    /// another region  ~80 ms, ~40 MB/s   (inter-region round trip)
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

    /// The same table with a different number of reads in flight.
    pub fn with_concurrent_reads(mut self, reads: f64) -> Self {
        self.concurrent_reads = reads.max(1.0);
        self
    }

    /// How bytes arrive from `distance`.
    pub fn link(&self, distance: Distance) -> Link {
        match distance {
            Distance::Local => self.local_link,
            Distance::Near => self.near_link,
            Distance::Far => self.far_link,
        }
    }

    /// How long a worker waits for `bytes` to arrive from `distance`.
    ///
    /// # Why waiting is priced at all
    ///
    /// Calibration turned up an awkward fact: AWS bills nothing for transfer
    /// from S3 to compute in the same region, so a byte from the next rack and
    /// a byte from this node cost the same money. Distance was therefore
    /// invisible to a cost model made only of money, and the locality
    /// machinery had nothing to weigh.
    ///
    /// Waiting is not free, though — it is paid for in the worker that sits
    /// idle. Charging that time at [`PriceTable::cpu_second_usd`] puts
    /// distance back into the one currency without inventing a transfer fee
    /// nobody would be billed for.
    ///
    /// Cold storage adds its own delay, which is where the difference is
    /// stark: a retrieval measured in hours is not a slower read, it is a
    /// different kind of operation.
    pub fn wait_seconds(&self, bytes: u64, tier: Tier, distance: Distance) -> f64 {
        let waiting = self.link(distance).seconds(bytes);
        match tier {
            Tier::Hot => waiting,
            Tier::Cold => waiting * self.cold_multiplier,
        }
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
        let wait_seconds = if bytes == 0 {
            0.0
        } else {
            self.wait_seconds(bytes, tier, distance)
        };
        Cost {
            bytes,
            cpu_seconds,
            wait_seconds,
            usd: bytes as f64 * self.byte_usd(tier, distance)
                + (cpu_seconds + wait_seconds) * self.cpu_second_usd,
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

    /// Bytes and cpu-seconds compose exactly; money does not, and should not.
    ///
    /// This used to assert that pricing a whole read equals summing its parts.
    /// Once waiting is priced that is false, and falsely: three reads pay three
    /// round trips where one pays one. The difference is the reason large reads
    /// are preferred to small ones, so a model in which it vanished would be
    /// unable to express the most basic advice about object storage.
    #[test]
    fn splitting_a_read_costs_extra_round_trips() {
        let p = PriceTable::default();
        let whole = p.price(3 * GB, Tier::Hot, Distance::Near, 3.0);
        let parts: Cost = (0..3)
            .map(|_| p.price(GB, Tier::Hot, Distance::Near, 1.0))
            .sum();

        // The countable parts still add up.
        assert_eq!(parts.bytes, whole.bytes);
        assert!((parts.cpu_seconds - whole.cpu_seconds).abs() < f64::EPSILON);

        // The waiting does not: two extra first-byte latencies.
        let extra = parts.wait_seconds - whole.wait_seconds;
        let round_trip = p.near_link.first_byte_seconds;
        assert!(
            (extra - 2.0 * round_trip).abs() < 1e-9,
            "expected two extra round trips, got {extra}s"
        );
        assert!(parts.usd > whole.usd, "and they are paid for");
    }

    #[test]
    fn a_round_trip_dominates_a_small_read_and_vanishes_in_a_large_one() {
        // The lesson the two-part Link exists to express.
        let p = PriceTable::default();
        let small = p.price(1_000_000, Tier::Hot, Distance::Near, 0.0);
        let large = p.price(1_000_000_000, Tier::Hot, Distance::Near, 0.0);

        let round_trip = p.near_link.first_byte_seconds;
        assert!(
            round_trip / small.wait_seconds > 0.5,
            "a megabyte should spend most of its time waiting"
        );
        assert!(
            round_trip / large.wait_seconds < 0.01,
            "a gigabyte should barely notice"
        );
    }

    /// The gap calibration exposed, now closed.
    ///
    /// Transfer within a region is not billed, so `byte_usd` cannot tell
    /// `Local` from `Near`. Waiting can, and does.
    #[test]
    fn distance_costs_time_even_where_it_costs_no_money() {
        let p = PriceTable::default();
        assert_eq!(
            p.byte_usd(Tier::Hot, Distance::Local),
            p.byte_usd(Tier::Hot, Distance::Near),
            "money still cannot tell them apart"
        );

        let local = p.price(GB, Tier::Hot, Distance::Local, 0.0);
        let near = p.price(GB, Tier::Hot, Distance::Near, 0.0);
        let far = p.price(GB, Tier::Hot, Distance::Far, 0.0);

        assert!(local.wait_seconds < near.wait_seconds);
        assert!(near.wait_seconds < far.wait_seconds);
        assert!(
            local.usd < near.usd && near.usd < far.usd,
            "and the total now orders them: {} {} {}",
            local.usd,
            near.usd,
            far.usd
        );
    }

    #[test]
    fn cold_storage_is_slow_as_well_as_dear() {
        let p = PriceTable::default();
        let hot = p.price(GB, Tier::Hot, Distance::Near, 0.0);
        let cold = p.price(GB, Tier::Cold, Distance::Near, 0.0);
        assert!(cold.wait_seconds > hot.wait_seconds * 10.0);
    }

    #[test]
    fn nothing_read_means_nothing_waited_for() {
        // A cpu-only cost must not be charged a round trip it never made.
        let p = PriceTable::default();
        let compute = p.price(0, Tier::Hot, Distance::Far, 1.0);
        assert_eq!(compute.wait_seconds, 0.0);
        assert!((compute.usd - p.cpu_second_usd).abs() < 1e-18);
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
