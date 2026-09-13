//! Whether an index would tell us anything the file format does not.
//!
//! The measurement in `examples/measure.rs` found the loop's worst flaw: it
//! would build an index that saves 95% of a scan and two that save nothing,
//! with equal enthusiasm. The mechanism was fine; the judgement was absent.
//!
//! # Why selectivity alone cannot decide it
//!
//! ```text
//!              files per value   bytes saved
//! CLUSTERED         1.0            1.1%
//! SELECTIVE         1.1           95.0%
//! ```
//!
//! Nearly identical selectivity, utterly different value. What separates them
//! is that on clustered data **Parquet's own row-group statistics already
//! prune** — the index skips files the reader was barely touching anyway.
//!
//! So the question is not "how selective is this field" but "how much better
//! than the per-file min/max ranges can an index do". Both sides of that are
//! computable before building anything: Iceberg manifests carry
//! `lower_bounds` and `upper_bounds` per file per field.
//!
//! # The estimate
//!
//! For a point lookup, the files a reader must open using ranges alone is the
//! number of ranges covering the sought value. Averaged over the domain that
//! is
//!
//! ```text
//! files_by_bounds = sum of range widths / width of their union
//! ```
//!
//! which is 1 when files partition the domain and *N* when every file spans
//! all of it. An exact index can do no better than the files a value truly
//! occupies, which for values scattered over files is about the number of rows
//! sharing that value, capped by what the ranges already give:
//!
//! ```text
//! files_by_index = min(files_by_bounds, rows per value)
//! ```
//!
//! The difference is what an index buys. Applied to the three measured
//! regimes it gives ~0%, ~0% and ~94%, against realized savings of 1.1%, 0.0%
//! and 95.0%.
//!
//! # What it needs, and does not have for free
//!
//! Rows per value requires the number of distinct values, which Iceberg does
//! not record by default. It has to be estimated — from a sample, from a
//! Puffin sketch if one exists, or supplied by a caller who knows. This module
//! takes it as an input rather than inventing it, so the requirement is
//! visible instead of buried in a constant.

/// How one field's values sit across a table's files.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    /// Files in the table.
    pub files: u64,
    /// Files a point lookup touches using per-file ranges alone.
    ///
    /// What the file format achieves for nothing. Between 1 and `files`.
    pub files_by_bounds: f64,
    /// Files a point lookup touches with an exact index.
    ///
    /// Never more than [`Spread::files_by_bounds`]: an index that did worse
    /// than the ranges would simply not be consulted.
    pub files_by_index: f64,
}

impl Spread {
    /// Estimate from per-file value ranges and how many rows share a value.
    ///
    /// `ranges` is one `(low, high)` per file, as `f64` so that integers,
    /// dates and timestamps can all be mapped onto it by the caller. Files
    /// with no recorded range are excluded from the ranges but must still be
    /// counted in `files`, since a file whose contents are unknown has to be
    /// read.
    pub fn from_bounds(files: u64, ranges: &[(f64, f64)], rows_per_value: f64) -> Self {
        if files == 0 {
            return Spread {
                files: 0,
                files_by_bounds: 0.0,
                files_by_index: 0.0,
            };
        }

        let unranged = files.saturating_sub(ranges.len() as u64) as f64;
        let low = ranges.iter().map(|(low, _)| *low).fold(f64::MAX, f64::min);
        let high = ranges
            .iter()
            .map(|(_, high)| *high)
            .fold(f64::MIN, f64::max);
        let span = high - low;

        // Every range covering the whole domain, or no ranges at all, means
        // the bounds tell us nothing: assume every file must be opened.
        let by_bounds = if ranges.is_empty() || span <= 0.0 {
            files as f64
        } else {
            let covered: f64 = ranges.iter().map(|(low, high)| (high - low) / span).sum();
            // A file with no range is a file that always has to be read.
            (covered + unranged).clamp(1.0, files as f64)
        };

        Spread {
            files,
            files_by_bounds: by_bounds,
            files_by_index: rows_per_value.clamp(0.0, by_bounds),
        }
    }

    /// The fraction of a scan an index removes that the ranges do not.
    ///
    /// Zero when the format already prunes as well as an index could, which is
    /// the case the loop used to build for anyway.
    pub fn index_advantage(&self) -> f64 {
        if self.files == 0 || self.files_by_bounds <= 0.0 {
            return 0.0;
        }
        ((self.files_by_bounds - self.files_by_index) / self.files as f64).clamp(0.0, 1.0)
    }

    /// The same as a percentage, for comparing against a policy threshold.
    pub fn index_advantage_pct(&self) -> f64 {
        self.index_advantage() * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 20 files partitioning 5,000 tenants, 200 rows per tenant.
    ///
    /// The regime `examples/measure.rs` calls CLUSTERED, where the index
    /// pruned 19 of 20 files and saved 1.1% of the bytes.
    fn clustered() -> Spread {
        let ranges: Vec<(f64, f64)> = (0..20)
            .map(|file| {
                let low = file as f64 * 250.0;
                (low, low + 249.0)
            })
            .collect();
        Spread::from_bounds(20, &ranges, 200.0)
    }

    /// 20 files each spanning all 5,000 tenants, 200 rows per tenant.
    fn scattered() -> Spread {
        let ranges: Vec<(f64, f64)> = (0..20).map(|_| (0.0, 4_999.0)).collect();
        Spread::from_bounds(20, &ranges, 200.0)
    }

    /// 20 files each spanning 5,000,000 ids, about 1.1 rows per id.
    fn selective() -> Spread {
        let ranges: Vec<(f64, f64)> = (0..20).map(|_| (0.0, 4_999_999.0)).collect();
        Spread::from_bounds(20, &ranges, 1.1)
    }

    #[test]
    fn disjoint_ranges_leave_an_index_nothing_to_add() {
        let spread = clustered();
        assert!(
            spread.files_by_bounds < 1.5,
            "ranges alone should already reach one file, got {}",
            spread.files_by_bounds
        );
        assert!(
            spread.index_advantage_pct() < 5.0,
            "an index cannot improve on a partitioned domain, claimed {}%",
            spread.index_advantage_pct()
        );
    }

    #[test]
    fn values_in_every_file_leave_an_index_nothing_to_add() {
        let spread = scattered();
        assert_eq!(spread.files_by_bounds, 20.0, "ranges prune nothing");
        assert_eq!(
            spread.files_by_index, 20.0,
            "and neither does an index, since every file holds every value"
        );
        assert_eq!(spread.index_advantage_pct(), 0.0);
    }

    #[test]
    fn rare_values_under_overlapping_ranges_are_what_an_index_is_for() {
        let spread = selective();
        assert_eq!(spread.files_by_bounds, 20.0, "ranges prune nothing");
        assert!(spread.files_by_index < 2.0);
        assert!(
            spread.index_advantage_pct() > 90.0,
            "expected a large advantage, got {}%",
            spread.index_advantage_pct()
        );
    }

    /// The whole point: the three are ranked the way the measurement ranked
    /// them, which `Proposal::ceiling_usd` failed to do.
    #[test]
    fn the_three_measured_regimes_are_ordered_correctly() {
        let advantages = [
            clustered().index_advantage_pct(),
            scattered().index_advantage_pct(),
            selective().index_advantage_pct(),
        ];
        assert!(
            advantages[2] > advantages[0] && advantages[2] > advantages[1],
            "only the selective regime should look worth building: {advantages:?}"
        );
        assert!(
            advantages[2] > 10.0 * advantages[0].max(advantages[1]).max(0.01),
            "and by a wide margin: {advantages:?}"
        );
    }

    #[test]
    fn the_estimate_matches_what_was_measured() {
        // Realized savings were 1.1%, 0.0% and 95.0%.
        assert!((selective().index_advantage_pct() - 95.0).abs() < 6.0);
        assert!(clustered().index_advantage_pct() < 5.0);
        assert!(scattered().index_advantage_pct() < 5.0);
    }

    #[test]
    fn a_file_with_no_recorded_range_must_still_be_read() {
        // Nineteen files partition the domain, one reports nothing. The
        // unknown file has to be opened for every lookup, so the bounds
        // estimate cannot claim one file.
        let ranges: Vec<(f64, f64)> = (0..19)
            .map(|file| {
                let low = file as f64 * 250.0;
                (low, low + 249.0)
            })
            .collect();
        let spread = Spread::from_bounds(20, &ranges, 200.0);
        assert!(
            spread.files_by_bounds > 1.5,
            "an unranged file is always read, got {}",
            spread.files_by_bounds
        );
    }

    #[test]
    fn no_ranges_at_all_assumes_the_worst() {
        let spread = Spread::from_bounds(20, &[], 1.0);
        assert_eq!(spread.files_by_bounds, 20.0);
        // An index still helps here, which is right: knowing nothing about
        // layout is not evidence that an index is useless.
        assert!(spread.index_advantage_pct() > 90.0);
    }

    #[test]
    fn an_empty_table_claims_nothing() {
        let spread = Spread::from_bounds(0, &[], 1.0);
        assert_eq!(spread.index_advantage(), 0.0);
    }

    #[test]
    fn a_single_valued_domain_cannot_be_pruned_by_range() {
        // Every file reports the same single value, so ranges cannot separate
        // them and an index cannot either.
        let ranges: Vec<(f64, f64)> = (0..4).map(|_| (7.0, 7.0)).collect();
        let spread = Spread::from_bounds(4, &ranges, 100.0);
        assert_eq!(spread.files_by_bounds, 4.0);
        assert_eq!(spread.index_advantage(), 0.0);
    }

    #[test]
    fn an_index_is_never_credited_with_doing_worse_than_the_ranges() {
        // Rows per value far above what the ranges already achieve.
        let ranges: Vec<(f64, f64)> = (0..10)
            .map(|file| (file as f64 * 10.0, file as f64 * 10.0 + 9.0))
            .collect();
        let spread = Spread::from_bounds(10, &ranges, 1_000.0);
        assert!(spread.files_by_index <= spread.files_by_bounds);
        assert_eq!(spread.index_advantage(), 0.0);
    }
}
