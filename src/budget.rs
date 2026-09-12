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
    spent: Cost,
    stopped: Option<Exceeded>,
}

impl Meter {
    /// A meter enforcing `budget` at `prices`.
    pub fn new(budget: Budget, prices: PriceTable) -> Self {
        Meter {
            budget,
            prices,
            spent: Cost::ZERO,
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
        if let Some(exceeded) = self.stopped {
            return Permit::Stop(exceeded);
        }
        self.spent = self.spent + self.prices.price(bytes, tier, distance, 0.0);
        match self.budget.breach(&self.spent) {
            None => Permit::Continue,
            Some(exceeded) => {
                self.stopped = Some(exceeded);
                Permit::Stop(exceeded)
            }
        }
    }

    /// Account for cpu time.
    pub fn charge_cpu(&mut self, seconds: f64) -> Permit {
        if let Some(exceeded) = self.stopped {
            return Permit::Stop(exceeded);
        }
        self.spent = self.spent + self.prices.price(0, Tier::Hot, Distance::Local, seconds);
        match self.budget.breach(&self.spent) {
            None => Permit::Continue,
            Some(exceeded) => {
                self.stopped = Some(exceeded);
                Permit::Stop(exceeded)
            }
        }
    }

    /// What has been spent so far.
    pub fn spent(&self) -> Cost {
        self.spent
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
    fn cpu_time_is_charged_against_the_money_ceiling() {
        let prices = PriceTable::default();
        let mut m = Meter::new(Budget::usd(prices.cpu_second_usd * 1.5), prices);
        assert!(m.charge_cpu(1.0).is_continue());
        assert!(!m.charge_cpu(1.0).is_continue());
        assert!(matches!(
            m.outcome(),
            Outcome::Aborted(Exceeded::Usd { .. })
        ));
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
    fn a_zero_byte_read_never_breaches() {
        let mut m = meter(Budget::bytes(0));
        assert!(m.charge(0, Tier::Hot, Distance::Local).is_continue());
    }
}
