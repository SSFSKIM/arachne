//! How often the depth read along a curve and the plane registry run their parts, and how long the
//! registry takes, for measuring what they cost: counted only in a build with the `counters`
//! feature, which no release turns on. Every count is shared by the threads a draw runs on.

use std::sync::atomic::{AtomicU64, Ordering};

/// Searches of a curve for the segments near a pixel (`CurveDepth::near`).
pub static NEAR: AtomicU64 = AtomicU64::new(0);
/// Segment bounds those searches test.
pub static BOUNDS_TESTED: AtomicU64 = AtomicU64::new(0);
/// Keys read along a curve at a point (`Centreline::key_under`).
pub static KEY_UNDER: AtomicU64 = AtomicU64::new(0);
/// Segments those reads project the point onto.
pub static FEET: AtomicU64 = AtomicU64::new(0);
/// The most segments one of those reads projected the point onto.
pub static MOST_FEET: AtomicU64 = AtomicU64::new(0);
/// Box gaps measured (`Centreline::gap2`).
pub static GAPS: AtomicU64 = AtomicU64::new(0);
/// Keys read on one segment (`key_on`).
pub static KEY_ON: AtomicU64 = AtomicU64::new(0);
/// Records with no plane of their own that lay on more identities than a fragment carries
/// (`depth::MOST_IDENTITIES`), and so kept the first ones (`depth::Registry::settle`).
pub static CAPPED_SETS: AtomicU64 = AtomicU64::new(0);
/// Mesh triangles whose corners lie on a line, offered with no plane of their own
/// (`depth::Registry::offer_triangle`).
pub static DEGENERATE_TRIANGLES: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones no flat record's identity holds (`depth::Registry::attached`).
pub static UNATTACHED_TRIANGLES: AtomicU64 = AtomicU64::new(0);
/// The identities the plane registry's partition of flat records made (`depth::Registry::settle`).
pub static FLAT_IDENTITIES: AtomicU64 = AtomicU64::new(0);
/// Linear programs the partition ran (`depth::Registry::fit`).
pub static PARTITION_PROGRAMS: AtomicU64 = AtomicU64::new(0);
/// Its fits that ran over every member of an identity and found no plane holding them.
pub static PARTITION_FAILED_FITS: AtomicU64 = AtomicU64::new(0);
/// Records with no plane of their own the registry's attach pass offered to the partition's
/// identities (`depth::Registry::settle`).
pub static ATTACH_RECORDS: AtomicU64 = AtomicU64::new(0);
/// Pairs of such a record and a flat identity the attach pass tried.
pub static ATTACH_PAIRS: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones its first point's distance from the identity's plane did not part, which
/// were tested at every point (`depth::Registry::contains`).
pub static ATTACH_NEAR: AtomicU64 = AtomicU64::new(0);
/// Linear programs the attach pass ran.
pub static ATTACH_PROGRAMS: AtomicU64 = AtomicU64::new(0);
/// Its fits that ran over every member of an identity and found no plane holding them.
pub static ATTACH_FAILED_FITS: AtomicU64 = AtomicU64::new(0);
/// Identities of records with no plane of their own that attached to none and share an axis
/// plane (`depth::Registry::settle`).
pub static AXIS_RUN_IDENTITIES: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds the registry spent settling (`depth::Registry::settle`), in its partition of flat
/// records, and in its attach pass.
pub static SETTLE_NS: AtomicU64 = AtomicU64::new(0);
pub static PARTITION_NS: AtomicU64 = AtomicU64::new(0);
pub static ATTACH_NS: AtomicU64 = AtomicU64::new(0);

const ALL: [(&str, &AtomicU64); 22] = [
    ("near", &NEAR),
    ("bounds_tested", &BOUNDS_TESTED),
    ("key_under", &KEY_UNDER),
    ("feet", &FEET),
    ("most_feet", &MOST_FEET),
    ("key_on", &KEY_ON),
    ("gaps", &GAPS),
    ("capped_sets", &CAPPED_SETS),
    ("degenerate_triangles", &DEGENERATE_TRIANGLES),
    ("unattached_triangles", &UNATTACHED_TRIANGLES),
    ("flat_identities", &FLAT_IDENTITIES),
    ("partition_programs", &PARTITION_PROGRAMS),
    ("partition_failed_fits", &PARTITION_FAILED_FITS),
    ("attach_records", &ATTACH_RECORDS),
    ("attach_pairs", &ATTACH_PAIRS),
    ("attach_near", &ATTACH_NEAR),
    ("attach_programs", &ATTACH_PROGRAMS),
    ("attach_failed_fits", &ATTACH_FAILED_FITS),
    ("axis_run_identities", &AXIS_RUN_IDENTITIES),
    ("settle_ns", &SETTLE_NS),
    ("partition_ns", &PARTITION_NS),
    ("attach_ns", &ATTACH_NS),
];

/// Adds `n` to `counter`.
pub fn add(counter: &AtomicU64, n: usize) {
    counter.fetch_add(n as u64, Ordering::Relaxed);
}

/// Adds the nanoseconds since `start` to `counter`.
pub fn since(counter: &AtomicU64, start: std::time::Instant) {
    let ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    counter.fetch_add(ns, Ordering::Relaxed);
}

/// Raises `counter` to `n` if it is lower.
pub fn most(counter: &AtomicU64, n: usize) {
    counter.fetch_max(n as u64, Ordering::Relaxed);
}

/// Every count by name, each set back to zero.
pub fn take() -> Vec<(&'static str, u64)> {
    ALL.iter()
        .map(|(name, c)| (*name, c.swap(0, Ordering::Relaxed)))
        .collect()
}
