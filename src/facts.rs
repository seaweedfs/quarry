//! What a storage backend can tell us beyond returning bytes.
//!
//! This is the capability boundary the design cares most about. A backend that
//! knows where its objects live, and which of them are cold, lets the
//! optimizer make decisions a byte-counting one cannot. A backend that knows
//! none of that must still work, and must not be a special case.
//!
//! So every question is optional, and the answers are resolved into concrete
//! values in exactly one place ([`resolve`]). No caller branches on which
//! backend it has, and nothing in the engine names one.
//!
//! Deliberately dependency-free, and in the core rather than under `engine`:
//! it is a boundary, not an implementation detail. Objects are named by the
//! path string the storage layer uses, which is not necessarily the same string
//! a catalog records for a data file.

use std::collections::BTreeMap;
use std::fmt;

use crate::cost::Tier;
use crate::place::{Distance, Place};
use crate::snapshot::Commits;

/// What a backend can say about its objects.
///
/// All methods return `None` for "cannot say". See [`resolve`] for what is
/// assumed then, and why the two dimensions do *not* default the same way.
pub trait StorageFacts: fmt::Debug + Send + Sync {
    /// Which tier `object` is stored in.
    fn tier(&self, object: &str) -> Option<Tier>;

    /// How far `object` is from a reader at `reader`.
    fn distance(&self, object: &str, reader: &Place) -> Option<Distance>;

    /// The commit log this backend pushes into, if it can.
    ///
    /// A backend that sees commits — a catalog, a volume server — hands the
    /// optimizer the same [`Commits`] it notes them on, and a round hears the
    /// table moved without asking. `None` means poll: the table's own
    /// snapshot is the fallback and nothing is lost.
    fn commits(&self) -> Option<Commits> {
        None
    }
}

/// What the engine concluded about an object, after defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolved {
    /// The tier to price against.
    pub tier: Tier,
    /// The distance to price against.
    pub distance: Distance,
    /// Whether the backend actually answered, or defaults were used.
    pub assumed: bool,
}

/// Turn what a backend will say into what the engine will use.
///
/// The two defaults are deliberately not symmetric, and getting this wrong
/// would be expensive in opposite directions:
///
/// ```text
/// unknown distance → Far
///     If a backend cannot say where an object is, we must not claim
///     locality. Far is also usually the truth: object storage is remote.
///
/// unknown tier → Hot
///     NOT the worst case, on purpose. A backend that cannot report tiers
///     almost certainly does not have them — plain S3 is uniformly hot.
///     Assuming Cold would multiply every price by the cold factor, which
///     overprices every plain-S3 read and would abort budgeted queries
///     that should have succeeded.
/// ```
///
/// The design said to "assume the worst" for both. That is right for distance
/// and wrong for tier: pessimism about a dimension the backend does not *have*
/// is not caution, it is a made-up cost.
pub fn resolve(facts: &dyn StorageFacts, object: &str, reader: &Place) -> Resolved {
    let tier = facts.tier(object);
    let distance = facts.distance(object, reader);
    Resolved {
        tier: tier.unwrap_or(Tier::Hot),
        distance: distance.unwrap_or(Distance::Far),
        assumed: tier.is_none() || distance.is_none(),
    }
}

/// A backend that reports nothing: plain S3, GCS, Azure Blob.
///
/// Everything resolves to hot and [`Distance::Far`], so placement degrades to
/// load balancing and the cost model still works. The same code path as a
/// backend that answers everything — no special case.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpaqueStorage;

impl StorageFacts for OpaqueStorage {
    fn tier(&self, _object: &str) -> Option<Tier> {
        None
    }

    fn distance(&self, _object: &str, _reader: &Place) -> Option<Distance> {
        None
    }
}

/// Where a set of objects live, and how cold they are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Which tier these objects are stored in.
    pub tier: Tier,
    /// Where these objects physically are.
    pub at: Place,
}

impl Placement {
    /// Hot objects at `at`.
    pub fn hot(at: Place) -> Self {
        Placement {
            tier: Tier::Hot,
            at,
        }
    }

    /// Cold objects at `at`.
    pub fn cold(at: Place) -> Self {
        Placement {
            tier: Tier::Cold,
            at,
        }
    }
}

/// A backend that reports placement per path prefix.
///
/// Stands in for a store that knows its own topology — SeaweedFS tracks a data
/// center, rack and node for every volume, and a tier for every one that has
/// been moved to cloud storage. Longest matching prefix wins, so a whole
/// bucket can be declared and individual paths overridden.
#[derive(Clone, Debug, Default)]
pub struct PlacedStorage {
    by_prefix: BTreeMap<String, Placement>,
    fallback: Option<Placement>,
}

impl PlacedStorage {
    /// A backend that can place nothing yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare where objects under `prefix` live.
    pub fn with_prefix(mut self, prefix: impl Into<String>, placement: Placement) -> Self {
        self.by_prefix.insert(prefix.into(), placement);
        self
    }

    /// Declare where everything else lives.
    ///
    /// Without this, an object matching no prefix is reported as unknown —
    /// which is honest, and resolves to hot and far.
    pub fn with_fallback(mut self, placement: Placement) -> Self {
        self.fallback = Some(placement);
        self
    }

    fn placement_of(&self, object: &str) -> Option<&Placement> {
        self.by_prefix
            .iter()
            .filter(|(prefix, _)| object.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, placement)| placement)
            .or(self.fallback.as_ref())
    }
}

impl StorageFacts for PlacedStorage {
    fn tier(&self, object: &str) -> Option<Tier> {
        Some(self.placement_of(object)?.tier)
    }

    fn distance(&self, object: &str, reader: &Place) -> Option<Distance> {
        Some(reader.distance(&self.placement_of(object)?.at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn here() -> Place {
        Place::parse("/onprem/dc1/rack2/node7")
    }

    #[test]
    fn a_backend_with_nothing_to_push_offers_no_log() {
        assert!(OpaqueStorage.commits().is_none());
        assert!(PlacedStorage::new().commits().is_none());
    }

    #[test]
    fn a_backend_that_hears_commits_shares_its_log() {
        use crate::snapshot::{Commits, SnapshotId, TableId};

        #[derive(Debug)]
        struct HearsCommits(Commits);
        impl StorageFacts for HearsCommits {
            fn tier(&self, _: &str) -> Option<Tier> {
                None
            }
            fn distance(&self, _: &str, _: &Place) -> Option<Distance> {
                None
            }
            fn commits(&self) -> Option<Commits> {
                Some(self.0.clone())
            }
        }

        let facts = HearsCommits(Commits::new());
        let shared = facts.commits().expect("a log to share");
        shared.note(TableId("events".into()), SnapshotId(7));

        // The same log the optimizer would hold hears what the backend noted.
        assert_eq!(facts.0.drain()[&TableId("events".into())], SnapshotId(7));
    }

    #[test]
    fn a_backend_that_knows_nothing_resolves_to_hot_and_far() {
        let resolved = resolve(&OpaqueStorage, "bucket/data/a.parquet", &here());
        assert_eq!(
            resolved,
            Resolved {
                tier: Tier::Hot,
                distance: Distance::Far,
                assumed: true,
            }
        );
    }

    #[test]
    fn unknown_tier_is_hot_rather_than_cold() {
        // A backend without tiers must not be priced as if everything were
        // cold; that would overprice every read by the cold multiplier.
        #[derive(Debug)]
        struct KnowsPlaceOnly;
        impl StorageFacts for KnowsPlaceOnly {
            fn tier(&self, _object: &str) -> Option<Tier> {
                None
            }
            fn distance(&self, _object: &str, _reader: &Place) -> Option<Distance> {
                Some(Distance::Local)
            }
        }

        let resolved = resolve(&KnowsPlaceOnly, "a", &here());
        assert_eq!(resolved.tier, Tier::Hot);
        assert_eq!(resolved.distance, Distance::Local);
        assert!(resolved.assumed, "the tier was assumed");
    }

    #[test]
    fn a_fully_answering_backend_is_not_marked_assumed() {
        let facts = PlacedStorage::new().with_fallback(Placement::hot(here()));
        let resolved = resolve(&facts, "a", &here());
        assert!(!resolved.assumed);
        assert_eq!(resolved.distance, Distance::Local);
    }

    #[test]
    fn placement_makes_a_local_object_local() {
        let facts = PlacedStorage::new()
            .with_prefix("hot/", Placement::hot(here()))
            .with_prefix(
                "far/",
                Placement::hot(Place::parse("/aws/us-west-2/az1/i-1")),
            );

        assert_eq!(
            resolve(&facts, "hot/a.parquet", &here()).distance,
            Distance::Local
        );
        assert_eq!(
            resolve(&facts, "far/a.parquet", &here()).distance,
            Distance::Far
        );
    }

    #[test]
    fn the_longest_matching_prefix_wins() {
        let facts = PlacedStorage::new()
            .with_prefix("bucket/", Placement::hot(Place::parse("/aws/us-west-2")))
            .with_prefix(
                "bucket/archive/",
                Placement::cold(Place::parse("/aws/us-west-2")),
            );

        assert_eq!(resolve(&facts, "bucket/a", &here()).tier, Tier::Hot);
        assert_eq!(
            resolve(&facts, "bucket/archive/a", &here()).tier,
            Tier::Cold,
            "the more specific declaration must win"
        );
    }

    #[test]
    fn an_unmatched_object_falls_back_or_is_unknown() {
        let strict = PlacedStorage::new().with_prefix("known/", Placement::hot(here()));
        assert!(strict.tier("other/a").is_none());
        assert!(resolve(&strict, "other/a", &here()).assumed);

        let lenient = strict.clone().with_fallback(Placement::cold(here()));
        assert_eq!(lenient.tier("other/a"), Some(Tier::Cold));
    }

    #[test]
    fn a_reader_elsewhere_sees_the_same_object_as_farther() {
        let facts = PlacedStorage::new().with_fallback(Placement::hot(here()));

        assert_eq!(resolve(&facts, "a", &here()).distance, Distance::Local);
        assert_eq!(
            resolve(&facts, "a", &Place::parse("/onprem/dc1/rack2/node8")).distance,
            Distance::Near
        );
        assert_eq!(
            resolve(&facts, "a", &Place::parse("/onprem/dc9/rack1/node1")).distance,
            Distance::Far
        );
        assert_eq!(
            resolve(&facts, "a", &Place::unknown()).distance,
            Distance::Far,
            "a reader that cannot locate itself is far from everything"
        );
    }

    #[test]
    fn a_cold_object_prices_above_a_hot_one() {
        use crate::cost::PriceTable;

        let facts = PlacedStorage::new()
            .with_prefix("hot/", Placement::hot(here()))
            .with_prefix("cold/", Placement::cold(here()));
        let prices = PriceTable::default();

        let hot = resolve(&facts, "hot/a", &here());
        let cold = resolve(&facts, "cold/a", &here());
        assert!(
            prices.byte_usd(cold.tier, cold.distance) > prices.byte_usd(hot.tier, hot.distance)
        );
    }
}
