//! Ceilings that stop execution, as opposed to estimates that inform it.
//!
//! The distinction is the whole reason this module exists:
//!
//! ```text
//! ESTIMATION  informs plan choice.          May be wrong.
//! ENFORCEMENT stops execution at the limit. Must not be.
//! ```
//!
//! A `max_bytes` that only feeds the planner is decoration: cardinality
//! estimates are routinely wrong by orders of magnitude, so a query that was
//! predicted to read 1 GB may read 400. Enforcement is what makes a budget a
//! promise, and it has to be metered against bytes actually read rather than
//! checked once before starting.

use crate::cost::{Cost, PriceTable, Tier};
use crate::place::Distance;

/// What a query is allowed to spend.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Budget {
    /// Most bytes that may be read, if limited.
    pub max_bytes: Option<u64>,
    /// Most money that may be spent, if limited.
    pub max_usd: Option<f64>,
}

impl Budget {
    /// No ceiling.
    pub const UNLIMITED: Budget = Budget {
        max_bytes: None,
        max_usd: None,
    };

    /// A byte ceiling.
    pub fn bytes(max_bytes: u64) -> Self {
        Budget {
            max_bytes: Some(max_bytes),
            ..Budget::UNLIMITED
        }
    }

    /// A money ceiling.
    pub fn usd(max_usd: f64) -> Self {
        Budget {
            max_usd: Some(max_usd),
            ..Budget::UNLIMITED
        }
    }

    /// Whether `spent` is still within this budget.
    pub fn allows(&self, spent: &Cost) -> bool {
        let within_bytes = self.max_bytes.is_none_or(|max| spent.bytes <= max);
        let within_usd = self.max_usd.is_none_or(|max| spent.usd <= max);
        within_bytes && within_usd
    }

    /// Which ceiling `spent` breaches, if any.
    ///
    /// Bytes are reported first when both are breached: it is the ceiling a
    /// caller set deliberately, and the more comprehensible of the two.
    pub fn breach(&self, spent: &Cost) -> Option<Exceeded> {
        if let Some(max) = self.max_bytes {
            if spent.bytes > max {
                return Some(Exceeded::Bytes {
                    limit: max,
                    spent: spent.bytes,
                });
            }
        }
        if let Some(max) = self.max_usd {
            if spent.usd > max {
                return Some(Exceeded::Usd {
                    limit: max,
                    spent: spent.usd,
                });
            }
        }
        None
    }
}

/// Which ceiling was breached.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Exceeded {
    /// The byte ceiling.
    Bytes {
        /// The ceiling.
        limit: u64,
        /// What had been read when it was breached.
        spent: u64,
    },
    /// The money ceiling.
    Usd {
        /// The ceiling.
        limit: f64,
        /// What had been spent when it was breached.
        spent: f64,
    },
}

/// How a query ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Outcome {
    /// Ran to completion within budget.
    Complete,
    /// Stopped at a ceiling.
    Aborted(Exceeded),
}

impl Outcome {
    /// Whether the query finished.
    pub fn is_complete(&self) -> bool {
        matches!(self, Outcome::Complete)
    }
}

/// Meters what a query spends and refuses reads past its ceiling.
///
/// Charged per read as it happens, not once up front, which is what lets a
/// query stop *mid-scan* rather than after overspending. A caller that
/// ignores the return value of [`Meter::charge`] has an estimate, not a
/// budget.
#[derive(Clone, Debug)]
pub struct Meter {
    budget: Budget,
    prices: PriceTable,
    /// Accumulated as though every read waited its turn; the overlap is
    /// applied in [`Meter::spent`], because it depends on how many reads there
    /// turned out to be.
    serial: Cost,
    reads: u64,
    stopped: Option<Exceeded>,
}

impl Meter {
    /// A meter enforcing `budget` at `prices`.
    pub fn new(budget: Budget, prices: PriceTable) -> Self {
        Meter {
            budget,
            prices,
            serial: Cost::ZERO,
            reads: 0,
            stopped: None,
        }
    }

    /// Account for a read, and say whether the query may continue.
    ///
    /// The read is charged *before* the ceiling is checked, so the reported
    /// spend reflects what was actually consumed rather than the last amount
    /// that happened to fit. Once stopped, a meter stays stopped and charges
    /// nothing further: a caller that keeps going gets the same refusal
    /// rather than a new one that hides the original cause.
    pub fn charge(&mut self, bytes: u64, tier: Tier, distance: Distance) -> Permit {
        // A request happened even when it returned no bytes: a GET of an
        // empty range is still billed, and still waited for.
        if bytes == 0 {
            return self.charge_meta(tier, distance);
        }
        self.account(self.prices.price(bytes, tier, distance, 0.0))
    }

    /// Account for a metadata request: a HEAD, or any call that returns
    /// properties rather than a payload.
    ///
    /// Billed at the GET rate on AWS and round-trips like one, so it costs a
    /// request fee plus one first-byte wait. It has no bytes — pricing it as
    /// `price(0, ..)` would charge nothing, which is exactly the unmetered
    /// surface this exists to close.
    pub fn charge_meta(&mut self, tier: Tier, distance: Distance) -> Permit {
        self.request(tier, distance, self.prices.request_usd)
    }

    /// Account for a listing request.
    ///
    /// AWS bills LIST at the PUT rate, twelve and a half times a GET. A call
    /// that paginates invisibly is charged once per page *the caller can see*
    /// — the [`ObjectStore`](object_store::ObjectStore) `list` stream hides
    /// its pages, so a long listing is undercharged rather than uncharged.
    pub fn charge_list(&mut self, tier: Tier, distance: Distance) -> Permit {
        self.request(tier, distance, self.prices.list_usd)
    }

    fn request(&mut self, tier: Tier, distance: Distance, fee: f64) -> Permit {
        let wait = self.prices.wait_seconds(0, tier, distance);
        self.account(Cost {
            wait_seconds: wait,
            usd: fee + wait * self.prices.cpu_second_usd,
            ..Cost::ZERO
        })
    }

    fn account(&mut self, cost: Cost) -> Permit {
        if let Some(exceeded) = self.stopped {
            return Permit::Stop(exceeded);
        }
        self.serial = self.serial + cost;
        self.reads += 1;
        let spent = self.spent();
        match self.budget.breach(&spent) {
            None => Permit::Continue,
            Some(exceeded) => {
                self.stopped = Some(exceeded);
                Permit::Stop(exceeded)
            }
        }
    }

    /// What has been spent so far.
    ///
    /// Waiting is discounted by however many reads overlapped, which is why
    /// this is computed rather than accumulated: `n` reads over `f` parallel
    /// streams wait `n/f` times, and `n` is not known until the scan ends.
    /// Dividing by `min(f, n)` keeps a single read paying its full round trip
    /// while a wide scan pays once per wave.
    pub fn spent(&self) -> Cost {
        let overlap = self
            .prices
            .concurrent_reads
            .min(self.reads.max(1) as f64)
            .max(1.0);
        let waited = self.serial.wait_seconds / overlap;
        Cost {
            wait_seconds: waited,
            usd: self.serial.usd - (self.serial.wait_seconds - waited) * self.prices.cpu_second_usd,
            ..self.serial
        }
    }

    /// How the query ended, as things stand.
    pub fn outcome(&self) -> Outcome {
        match self.stopped {
            None => Outcome::Complete,
            Some(exceeded) => Outcome::Aborted(exceeded),
        }
    }
}

/// Whether a query may keep reading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Permit {
    /// Carry on.
    Continue,
    /// Stop: this ceiling has been breached.
    Stop(Exceeded),
}

impl Permit {
    /// Whether reading may continue.
    pub fn is_continue(&self) -> bool {
        matches!(self, Permit::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1_000_000;

    fn meter(budget: Budget) -> Meter {
        Meter::new(budget, PriceTable::default())
    }

    #[test]
    fn an_unlimited_budget_permits_everything() {
        let mut m = meter(Budget::UNLIMITED);
        assert!(m.charge(u64::MAX, Tier::Cold, Distance::Far).is_continue());
        assert!(m.outcome().is_complete());
    }

    #[test]
    fn a_byte_ceiling_stops_mid_scan() {
        // Ten reads of 1 MB against a 4 MB ceiling: the fifth must fail, not
        // the tenth, and not a pre-flight check before the first.
        let mut m = meter(Budget::bytes(4 * MB));
        let mut permitted = 0;
        for _ in 0..10 {
            if m.charge(MB, Tier::Hot, Distance::Local).is_continue() {
                permitted += 1;
            } else {
                break;
            }
        }
        assert_eq!(permitted, 4);
        assert_eq!(
            m.outcome(),
            Outcome::Aborted(Exceeded::Bytes {
                limit: 4 * MB,
                spent: 5 * MB,
            })
        );
    }

    #[test]
    fn reaching_the_ceiling_exactly_is_allowed() {
        let mut m = meter(Budget::bytes(MB));
        assert!(m.charge(MB, Tier::Hot, Distance::Local).is_continue());
        assert!(m.outcome().is_complete());
    }

    #[test]
    fn a_money_ceiling_stops_reads_that_bytes_alone_would_allow() {
        // Same byte count, but cold and far, so the money ceiling binds.
        let prices = PriceTable::default();
        let budget = Budget::usd(prices.byte_usd(Tier::Hot, Distance::Local) * 2.0 * MB as f64);

        let mut hot = Meter::new(budget, prices);
        assert!(hot.charge(MB, Tier::Hot, Distance::Local).is_continue());

        let mut cold = Meter::new(budget, prices);
        let permit = cold.charge(MB, Tier::Cold, Distance::Far);
        assert!(!permit.is_continue());
        assert!(matches!(
            cold.outcome(),
            Outcome::Aborted(Exceeded::Usd { .. })
        ));
    }

    #[test]
    fn spend_reflects_what_was_consumed_not_what_fit() {
        // The breaching read is charged, so the report says 5 MB were read
        // against a 4 MB ceiling rather than pretending only 4 were.
        let mut m = meter(Budget::bytes(4 * MB));
        m.charge(5 * MB, Tier::Hot, Distance::Local);
        assert_eq!(m.spent().bytes, 5 * MB);
    }

    #[test]
    fn a_stopped_meter_stays_stopped_and_charges_nothing_more() {
        let mut m = meter(Budget::bytes(MB));
        let first = m.charge(2 * MB, Tier::Hot, Distance::Local);
        let spent = m.spent();

        let second = m.charge(100 * MB, Tier::Hot, Distance::Local);
        assert_eq!(first, second, "the original cause must not be overwritten");
        assert_eq!(m.spent(), spent, "no further charge after stopping");
    }

    #[test]
    fn bytes_are_reported_before_money_when_both_breach() {
        let budget = Budget {
            max_bytes: Some(0),
            max_usd: Some(0.0),
        };
        let mut m = meter(budget);
        let permit = m.charge(MB, Tier::Cold, Distance::Far);
        assert!(matches!(permit, Permit::Stop(Exceeded::Bytes { .. })));
    }

    #[test]
    fn allows_agrees_with_breach() {
        let budget = Budget::bytes(10);
        for bytes in [0u64, 5, 10, 11, 100] {
            let spent = Cost {
                bytes,
                ..Cost::ZERO
            };
            assert_eq!(
                budget.allows(&spent),
                budget.breach(&spent).is_none(),
                "disagreement at {bytes} bytes"
            );
        }
    }

    #[test]
    fn a_zero_byte_read_is_still_a_request() {
        // An empty GET moves no bytes, so the byte ceiling has nothing to
        // say — but the request happened, and it is billed and waited for.
        let mut m = meter(Budget::bytes(0));
        assert!(m.charge(0, Tier::Hot, Distance::Local).is_continue());

        let prices = PriceTable::default();
        let mut m = meter(Budget::usd(prices.request_usd * 1.5));
        assert!(m.charge(0, Tier::Hot, Distance::Local).is_continue());
        assert!(
            !m.charge(0, Tier::Hot, Distance::Local).is_continue(),
            "two requests cost more than one, even fetching nothing"
        );
    }

    /// Metadata calls were unmetered: `head` routes through `get_opts` with an
    /// empty range, priced `bytes=0`, charged nothing.
    ///
    /// The ground truth is computable: a HEAD costs its request fee plus one
    /// first-byte wait, a LIST twelve and a half times the fee.
    #[test]
    fn a_head_and_a_listing_are_charged_what_they_cost() {
        let prices = PriceTable::default();
        let wait = prices.wait_seconds(0, Tier::Hot, Distance::Far);

        let mut m = meter(Budget::UNLIMITED);
        m.charge_meta(Tier::Hot, Distance::Far);
        let head = m.spent();
        let expected = prices.request_usd + wait * prices.cpu_second_usd;
        assert!(
            (head.usd - expected).abs() < 1e-18,
            "a HEAD costs {expected}, charged {}",
            head.usd
        );
        assert!(head.wait_seconds > 0.0, "and it waited a round trip");

        let mut m = meter(Budget::UNLIMITED);
        m.charge_list(Tier::Hot, Distance::Far);
        let list = m.spent();
        let expected = prices.list_usd + wait * prices.cpu_second_usd;
        assert!(
            (list.usd - expected).abs() < 1e-18,
            "a LIST costs {expected}, charged {}",
            list.usd
        );
        assert!(list.usd > head.usd, "AWS bills LIST above GET");
    }

    #[test]
    fn metadata_requests_also_overlap() {
        // Sixteen parallel HEADs wait once per wave like anything else.
        let prices = PriceTable::default().with_concurrent_reads(4.0);
        let mut m = Meter::new(Budget::UNLIMITED, prices);
        for _ in 0..16 {
            m.charge_meta(Tier::Hot, Distance::Near);
        }
        let one = prices.wait_seconds(0, Tier::Hot, Distance::Near);
        assert!((m.spent().wait_seconds - 4.0 * one).abs() < 1e-12);
    }

    /// The overlap model, against arithmetic done here rather than by it.
    ///
    /// Sixteen reads over four streams wait four times. This is the term that
    /// was wrong by the engine's parallelism before `concurrent_reads`
    /// existed, and on a sixteen-core machine that is not a rounding error.
    #[test]
    fn reads_that_overlap_wait_once_per_wave() {
        let prices = PriceTable::default().with_concurrent_reads(4.0);
        let one_read = prices.wait_seconds(1_000, Tier::Hot, Distance::Near);

        let mut meter = Meter::new(Budget::UNLIMITED, prices);
        for _ in 0..16 {
            assert_eq!(
                meter.charge(1_000, Tier::Hot, Distance::Near),
                Permit::Continue
            );
        }

        let waited = meter.spent().wait_seconds;
        assert!(
            (waited - 4.0 * one_read).abs() < 1e-12,
            "16 reads over 4 streams should wait 4 times: {waited} vs {}",
            4.0 * one_read
        );
    }

    #[test]
    fn a_single_read_still_waits_in_full() {
        // The case that rules out simply dividing every read by the
        // parallelism: one read overlaps with nothing.
        let prices = PriceTable::default().with_concurrent_reads(16.0);
        let mut meter = Meter::new(Budget::UNLIMITED, prices);
        meter.charge(1_000, Tier::Hot, Distance::Far);

        let expected = prices.wait_seconds(1_000, Tier::Hot, Distance::Far);
        assert!((meter.spent().wait_seconds - expected).abs() < 1e-12);
    }

    #[test]
    fn overlap_is_reflected_in_the_money_too() {
        // Not just the reported seconds: the discount has to reach the total,
        // or a budget would still be spent at the serial rate.
        let serial = Meter::new(Budget::UNLIMITED, PriceTable::default());
        let parallel = Meter::new(
            Budget::UNLIMITED,
            PriceTable::default().with_concurrent_reads(8.0),
        );
        let spend = |mut m: Meter| {
            for _ in 0..8 {
                m.charge(1_000, Tier::Hot, Distance::Near);
            }
            m.spent()
        };
        let (serial, parallel) = (spend(serial), spend(parallel));

        assert_eq!(serial.bytes, parallel.bytes, "bytes are unaffected");
        assert!(parallel.usd < serial.usd);
    }

    #[test]
    fn overlapping_reads_let_a_budget_go_further() {
        // The consequence that matters. Waiting is charged, so overstating it
        // aborts queries that were in fact affordable.
        let serial_prices = PriceTable::default();
        let eight_reads = |prices: PriceTable, n: usize| {
            let mut m = Meter::new(Budget::UNLIMITED, prices);
            for _ in 0..n {
                m.charge(1_000, Tier::Hot, Distance::Near);
            }
            m.spent().usd
        };

        // A budget priced for exactly eight serial reads.
        let budget = Budget::usd(eight_reads(serial_prices, 8));

        let mut serial = Meter::new(budget, serial_prices);
        for _ in 0..8 {
            assert_eq!(
                serial.charge(1_000, Tier::Hot, Distance::Near),
                Permit::Continue
            );
        }
        assert!(
            matches!(
                serial.charge(1_000, Tier::Hot, Distance::Near),
                Permit::Stop(_)
            ),
            "the ninth serial read should not fit"
        );

        // Overlapped sixteen ways, the same money buys more than eight.
        let mut parallel = Meter::new(budget, serial_prices.with_concurrent_reads(16.0));
        for i in 0..16 {
            assert_eq!(
                parallel.charge(1_000, Tier::Hot, Distance::Near),
                Permit::Continue,
                "read {i} should fit when reads overlap"
            );
        }
    }
}
