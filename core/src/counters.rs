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
/// Records it attached with a point outside an identity's members' hull, through the bound's
/// growth beyond it (`depth::Reach::bound`).
pub static ATTACH_THROUGH_ALLOWANCE: AtomicU64 = AtomicU64::new(0);
/// Pairs it refused because the record does not enter the identity's members' hull
/// (`depth::Registry::enters`) where the bound without its growth outside the hull and one plane
/// holding the record and the members within their tolerances would have attached it.
pub static ATTACH_UNENTERED: AtomicU64 = AtomicU64::new(0);
/// Identities of records with no plane of their own that attached to none and share an axis
/// plane (`depth::Registry::settle`).
pub static AXIS_RUN_IDENTITIES: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds the registry spent settling (`depth::Registry::settle`), in its partition of flat
/// records, and in its attach pass.
pub static SETTLE_NS: AtomicU64 = AtomicU64::new(0);
pub static PARTITION_NS: AtomicU64 = AtomicU64::new(0);
pub static ATTACH_NS: AtomicU64 = AtomicU64::new(0);
/// Pairs of fragments on flat records of different identities whose keys lay within the tie
/// term, tested for being near-parallel (`order::near_parallel`), counted once per test.
pub static NEAR_PARALLEL_TESTED: AtomicU64 = AtomicU64::new(0);
/// Of those tests, the ones that ran the geometric test, finding no verdict the pair's records kept
/// (`depth::tested`).
pub static NEAR_PARALLEL_RUN: AtomicU64 = AtomicU64::new(0);
/// The records' points those runs read, both records' at each.
pub static NEAR_PARALLEL_POINTS: AtomicU64 = AtomicU64::new(0);
/// Of those tests, the ones that found the pair near-parallel.
pub static NEAR_PARALLEL_PAIRS: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones whose records' dihedral exceeds 1e-2 radians.
pub static NEAR_PARALLEL_WIDE_PAIRS: AtomicU64 = AtomicU64::new(0);
/// Sub-samples two fragments covered and tied only because their records are near-parallel,
/// counted once per resolved pixel (`composite::TransparentPass::resolve`).
pub static NEAR_PARALLEL_SAMPLES: AtomicU64 = AtomicU64::new(0);
/// Resolved pixels with at least one such sub-sample.
pub static NEAR_PARALLEL_PX: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones where a pair tied so has a dihedral beyond 1e-2 radians.
pub static NEAR_PARALLEL_WIDE_PX: AtomicU64 = AtomicU64::new(0);

const ALL: [(&str, &AtomicU64); 32] = [
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
    ("attach_through_allowance", &ATTACH_THROUGH_ALLOWANCE),
    ("attach_unentered", &ATTACH_UNENTERED),
    ("axis_run_identities", &AXIS_RUN_IDENTITIES),
    ("settle_ns", &SETTLE_NS),
    ("partition_ns", &PARTITION_NS),
    ("attach_ns", &ATTACH_NS),
    ("near_parallel_tested", &NEAR_PARALLEL_TESTED),
    ("near_parallel_run", &NEAR_PARALLEL_RUN),
    ("near_parallel_points", &NEAR_PARALLEL_POINTS),
    ("near_parallel_pairs", &NEAR_PARALLEL_PAIRS),
    ("near_parallel_wide_pairs", &NEAR_PARALLEL_WIDE_PAIRS),
    ("near_parallel_samples", &NEAR_PARALLEL_SAMPLES),
    ("near_parallel_px", &NEAR_PARALLEL_PX),
    ("near_parallel_wide_px", &NEAR_PARALLEL_WIDE_PX),
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
