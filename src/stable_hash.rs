//! A hash that means the same thing next year.
//!
//! [`std::hash::DefaultHasher`] is explicitly not stable across Rust releases,
//! which is fine for a value that lives in one process and fatal for one
//! written down. An index keyed on an unstable hash survives a compiler
//! upgrade as an index that matches **nothing** — a miss, not an error, so the
//! system quietly gets slower and the optimizer rebuilds everything with no
//! signal that anything broke.
//!
//! # What a collision costs, which is not the same everywhere
//!
//! ```text
//! pruning index   SAFE. Two values hashing alike means the index reports
//!                 files holding the other value too, so the scan reads
//!                 more than it needed. Over-selection; the predicate is
//!                 re-applied to rows regardless.
//!
//! plan identity   NOT SAFE. Two different queries hashing alike means one
//!                 is served the other's cached answer. See the warning on
//!                 [`StableHasher`]; this is a real, pre-existing gap and
//!                 not one a stable hash alone fixes.
//! ```
//!
//! # Versioning, because stability is not the same as permanence
//!
//! A stable algorithm still leaves one hazard: the *bytes fed to it*. Values
//! are hashed through their [`Hash`] implementations, which belong to `std`
//! and `arrow`, and either may change what it writes. So persisted derived
//! state carries [`HASH_VERSION`], and state stamped with a different version
//! is refused rather than probed. Invalidation becomes loud and rare instead
//! of silent.

use std::hash::{Hash, Hasher};

/// The version of everything that affects a persisted hash.
///
/// Bump this when the algorithm below changes, when the bytes fed to it
/// change, or when a dependency whose `Hash` implementation is used is
/// upgraded across a version that could reorder or reshape what it writes.
///
/// Refusing state with a different version costs a rebuild. Not refusing it
/// costs an index that silently matches nothing.
pub const HASH_VERSION: u32 = 1;

/// FNV-1a offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A fixed hash: FNV-1a over the bytes, then a SplitMix64 finalizer.
///
/// Both halves are published constants, so this produces the same value on
/// every platform and every compiler release, forever. FNV-1a alone has poor
/// avalanche — single-bit input changes barely move the high bits, which
/// clusters index buckets — and the finalizer fixes that for a handful of
/// arithmetic operations.
///
/// Not a cryptographic hash and not collision-resistant against an adversary
/// who picks inputs. That is acceptable for choosing which files to read,
/// where a collision costs extra I/O, and is **not** acceptable for deciding
/// that two query plans are the same.
#[derive(Clone, Debug)]
pub struct StableHasher {
    state: u64,
}

impl Default for StableHasher {
    fn default() -> Self {
        StableHasher { state: FNV_OFFSET }
    }
}

impl StableHasher {
    /// A fresh hasher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Hash one value and finish, which is the common case.
    pub fn of<T: Hash + ?Sized>(value: &T) -> u64 {
        let mut hasher = StableHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }
}

impl Hasher for StableHasher {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= *byte as u64;
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    // Every integer method is overridden, and that is not pedantry. The
    // default implementations write `to_ne_bytes()` — *native* endian — so a
    // hash written on x86 would not match one computed on a big-endian
    // machine. And `write_usize` defaults to the platform's pointer width, so
    // the same value would hash differently on 32- and 64-bit builds. Both
    // would have produced indexes that work until someone changes hardware.

    fn write_u8(&mut self, n: u8) {
        self.write(&[n]);
    }

    fn write_u16(&mut self, n: u16) {
        self.write(&n.to_le_bytes());
    }

    fn write_u32(&mut self, n: u32) {
        self.write(&n.to_le_bytes());
    }

    fn write_u64(&mut self, n: u64) {
        self.write(&n.to_le_bytes());
    }

    fn write_u128(&mut self, n: u128) {
        self.write(&n.to_le_bytes());
    }

    /// Widened to 64 bits, so a 32-bit build agrees with a 64-bit one.
    fn write_usize(&mut self, n: usize) {
        self.write(&(n as u64).to_le_bytes());
    }

    fn write_i8(&mut self, n: i8) {
        self.write_u8(n as u8);
    }

    fn write_i16(&mut self, n: i16) {
        self.write_u16(n as u16);
    }

    fn write_i32(&mut self, n: i32) {
        self.write_u32(n as u32);
    }

    fn write_i64(&mut self, n: i64) {
        self.write_u64(n as u64);
    }

    fn write_i128(&mut self, n: i128) {
        self.write_u128(n as u128);
    }

    fn write_isize(&mut self, n: isize) {
        self.write_usize(n as usize);
    }

    // `write_str` is deliberately *not* overridden: overriding it needs the
    // unstable `hasher_prefixfree_extras` feature. Its default writes the
    // string's bytes followed by `write_u8(0xff)`, which routes through the
    // override above, so strings are delimited — `("ab", "c")` and
    // `("a", "bc")` do not collide.
    //
    // That default is std's to change, which is the clearest example of why
    // `HASH_VERSION` exists: the algorithm here is fixed forever, but the
    // bytes handed to it are not entirely ours to fix.

    fn finish(&self) -> u64 {
        // SplitMix64's finalizer. Spreads FNV's weak high bits across the
        // whole word, so neighbouring inputs land in unrelated buckets.
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned so a change to the algorithm cannot pass unnoticed.
    ///
    /// These were computed by an independent implementation of FNV-1a plus
    /// the SplitMix64 finalizer, not copied from this code's output — a test
    /// that records whatever the implementation currently does would pass for
    /// a broken implementation too.
    ///
    /// If they move, every persisted index is invalid and [`HASH_VERSION`]
    /// must be bumped in the same change. That is the whole point of writing
    /// them down.
    #[test]
    fn the_algorithm_is_pinned_to_known_values() {
        assert_eq!(StableHasher::of(""), 5_705_415_379_283_624_141);
        assert_eq!(StableHasher::of("a"), 1_819_190_507_042_467_253);
        assert_eq!(StableHasher::of("hello"), 10_509_149_385_209_626_834);
        assert_eq!(StableHasher::of(&0u64), 9_313_164_154_874_788_883);
        assert_eq!(StableHasher::of(&1i64), 6_676_229_981_160_887_125);
    }

    #[test]
    fn a_delimiter_keeps_concatenations_apart() {
        assert_ne!(
            StableHasher::of(&("ab", "c")),
            StableHasher::of(&("a", "bc")),
            "without a terminator these are the same byte stream"
        );
    }

    #[test]
    fn integers_hash_the_same_width_regardless_of_platform() {
        // `usize` is widened to 64 bits, so these agree on any target.
        assert_eq!(StableHasher::of(&7usize), StableHasher::of(&7u64));
        assert_eq!(StableHasher::of(&(-7isize)), StableHasher::of(&(-7i64)));
    }

    #[test]
    fn the_same_input_always_hashes_the_same() {
        for value in ["", "a", "some longer string", "\u{1f600}"] {
            assert_eq!(StableHasher::of(value), StableHasher::of(value));
        }
    }

    #[test]
    fn different_inputs_hash_differently() {
        let one = StableHasher::of(&1i64);
        let two = StableHasher::of(&2i64);
        assert_ne!(one, two);
    }

    #[test]
    fn neighbouring_inputs_do_not_cluster() {
        // FNV-1a alone barely moves the high bits between 1 and 2. The
        // finalizer is what makes this hold, and index selectivity depends
        // on it: clustered buckets mean an index that prunes nothing.
        let high_bits: Vec<u64> = (0i64..64).map(|n| StableHasher::of(&n) >> 56).collect();
        let distinct: std::collections::BTreeSet<u64> = high_bits.iter().copied().collect();
        assert!(
            distinct.len() > 32,
            "only {} distinct high bytes across 64 neighbours",
            distinct.len()
        );
    }

    #[test]
    fn hashing_in_pieces_matches_hashing_at_once() {
        let mut split = StableHasher::new();
        split.write(b"hel");
        split.write(b"lo");

        let mut whole = StableHasher::new();
        whole.write(b"hello");

        assert_eq!(split.finish(), whole.finish());
    }

    #[test]
    fn no_collisions_across_a_realistic_key_space() {
        // Index keys are usually identifiers. A collision here would cost
        // over-selection rather than a wrong answer, but it should still be
        // rare enough not to appear in a million values.
        let hashes: std::collections::HashSet<u64> =
            (0i64..1_000_000).map(|n| StableHasher::of(&n)).collect();
        assert_eq!(hashes.len(), 1_000_000);
    }
}
