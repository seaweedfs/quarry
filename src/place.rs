//! Where things are, and how far apart.
//!
//! Every placement question in the engine — which worker runs a scan task,
//! which replica it reads, whether a shuffle crosses an expensive boundary —
//! is the same question at a different scale: *how far, and what does that
//! distance cost?*
//!
//! A [`Place`] is a path down a locality hierarchy. [`Distance`] is derived
//! from how much of that path two places share, so the engine never needs to
//! know what any particular level *means*. A deployment with two levels and
//! one with five both work, which is what keeps this vendor-neutral.

use std::fmt;

/// Where something is, as a path down a locality hierarchy.
///
/// The levels are deliberately unnamed. `/onprem/dc1/rack2/node7` and
/// `/aws/us-west-2/usw2-az1/i-0abc123` are both valid and comparable; the
/// engine only ever asks how much of the path two places share.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Place {
    /// Position is not known.
    ///
    /// [`Distance::Far`] from everything, *including itself*: two workers that
    /// cannot say where they are might be anywhere, and assuming otherwise
    /// would let the scheduler make confident bad decisions. This is why
    /// unknown is a variant rather than a `/unknown/<host>` path — a path
    /// would compare equal to itself and report [`Distance::Local`].
    Unknown,
    /// A known position, outermost level first.
    At(Vec<String>),
}

impl Place {
    /// A place whose position is not known.
    pub fn unknown() -> Self {
        Place::Unknown
    }

    /// Parse a slash-separated path such as `/onprem/dc1/rack2/node7`.
    ///
    /// Leading, trailing, and repeated separators are ignored. A path with no
    /// segments is [`Place::Unknown`], since it locates nothing.
    pub fn parse(path: &str) -> Self {
        let segments: Vec<String> = path
            .split('/')
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        if segments.is_empty() {
            Place::Unknown
        } else {
            Place::At(segments)
        }
    }

    /// The path segments, or `None` if the position is not known.
    pub fn segments(&self) -> Option<&[String]> {
        match self {
            Place::Unknown => None,
            Place::At(s) => Some(s),
        }
    }

    /// How far this place is from `other`.
    ///
    /// Let `shared` be the number of leading segments the two paths have in
    /// common and `depth` the deeper of the two:
    ///
    /// - `shared == depth` — the same position: [`Distance::Local`]
    /// - `shared == depth - 1` — siblings, or a parent and its child:
    ///   [`Distance::Near`]
    /// - otherwise: [`Distance::Far`]
    ///
    /// Either place being [`Place::Unknown`] yields [`Distance::Far`].
    pub fn distance(&self, other: &Place) -> Distance {
        let (a, b) = match (self.segments(), other.segments()) {
            (Some(a), Some(b)) => (a, b),
            _ => return Distance::Far,
        };
        let shared = a.iter().zip(b).take_while(|(x, y)| x == y).count();
        let depth = a.len().max(b.len());
        if shared == depth {
            Distance::Local
        } else if shared + 1 == depth {
            Distance::Near
        } else {
            Distance::Far
        }
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Place::Unknown => f.write_str("<unknown>"),
            Place::At(segments) => {
                for segment in segments {
                    write!(f, "/{segment}")?;
                }
                Ok(())
            }
        }
    }
}

/// How far apart two [`Place`]s are.
///
/// Three levels is enough to price placement today. The path representation in
/// [`Place`] means that distinguishing rack from data center, or adding a
/// cross-cloud boundary, later changes a price table rather than this type.
///
/// Ordered nearest first, so `min` picks the better option.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Distance {
    /// The same position. Reads are effectively free.
    Local,
    /// One level apart: siblings, or a parent and its child.
    Near,
    /// Anywhere else, or an unknown position. Assume the worst.
    Far,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(path: &str) -> Place {
        Place::parse(path)
    }

    #[test]
    fn identical_places_are_local() {
        assert_eq!(
            p("/onprem/dc1/rack2/node7").distance(&p("/onprem/dc1/rack2/node7")),
            Distance::Local
        );
    }

    #[test]
    fn siblings_are_near() {
        assert_eq!(
            p("/onprem/dc1/rack2/node7").distance(&p("/onprem/dc1/rack2/node8")),
            Distance::Near
        );
    }

    #[test]
    fn a_parent_and_its_child_are_near() {
        assert_eq!(
            p("/onprem/dc1/rack2").distance(&p("/onprem/dc1/rack2/node7")),
            Distance::Near
        );
    }

    #[test]
    fn different_racks_are_far() {
        assert_eq!(
            p("/onprem/dc1/rack2/node7").distance(&p("/onprem/dc1/rack9/node7")),
            Distance::Far
        );
    }

    #[test]
    fn different_clouds_are_far() {
        assert_eq!(
            p("/aws/us-west-2/usw2-az1/i-1").distance(&p("/gcp/us-central1/a/n-1")),
            Distance::Far
        );
    }

    #[test]
    fn unknown_is_far_from_everything_including_itself() {
        let here = p("/onprem/dc1/rack2/node7");
        assert_eq!(Place::unknown().distance(&here), Distance::Far);
        assert_eq!(here.distance(&Place::unknown()), Distance::Far);
        assert_eq!(
            Place::unknown().distance(&Place::unknown()),
            Distance::Far,
            "two workers that cannot locate themselves might be anywhere"
        );
    }

    #[test]
    fn distance_is_symmetric() {
        let places = [
            p("/a"),
            p("/a/b"),
            p("/a/b/c"),
            p("/a/b/d"),
            p("/z/y/x"),
            Place::unknown(),
        ];
        for x in &places {
            for y in &places {
                assert_eq!(
                    x.distance(y),
                    y.distance(x),
                    "distance {x} <-> {y} is not symmetric"
                );
            }
        }
    }

    #[test]
    fn nearest_sorts_first() {
        let mut ds = [Distance::Far, Distance::Local, Distance::Near];
        ds.sort();
        assert_eq!(ds, [Distance::Local, Distance::Near, Distance::Far]);
    }

    #[test]
    fn parsing_ignores_separator_noise() {
        assert_eq!(p("/a/b/"), p("a//b"));
        assert_eq!(p("/a/b/"), Place::At(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn a_path_that_locates_nothing_is_unknown() {
        assert_eq!(p(""), Place::Unknown);
        assert_eq!(p("/"), Place::Unknown);
    }

    #[test]
    fn display_round_trips() {
        assert_eq!(p("/onprem/dc1/rack2").to_string(), "/onprem/dc1/rack2");
        assert_eq!(Place::unknown().to_string(), "<unknown>");
    }
}
