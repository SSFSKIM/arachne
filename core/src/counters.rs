//! How often the depth read along a curve runs its parts, for measuring what it costs: counted
//! only in a build with the `counters` feature, which no release turns on. Every count is shared
//! by the threads a draw runs on.

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

const ALL: [(&str, &AtomicU64); 10] = [
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
];

/// Adds `n` to `counter`.
pub fn add(counter: &AtomicU64, n: usize) {
    counter.fetch_add(n as u64, Ordering::Relaxed);
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
