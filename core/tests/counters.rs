//! The depth read along a curve does only the work that can change what it reads: it keys only
//! the segments it may pick. Counted in a build with the `counters` feature (`cargo test
//! --features counters --test counters`).
#![cfg(feature = "counters")]

use std::sync::Mutex;

use arachne_core::counters;
use arachne_core::depth::Centreline;

/// The counts are shared by every thread of the process: one test reads them at a time.
static COUNTS: Mutex<()> = Mutex::new(());

fn count(name: &str, counts: &[(&str, u64)]) -> u64 {
    counts.iter().find(|(n, _)| *n == name).unwrap().1
}

/// A zigzag of `n` segments along y = 0, each a unit long, keyed by how far along it lies.
fn zigzag(n: usize) -> Centreline {
    let points: Vec<([f64; 2], f64)> = (0..=n)
        .map(|i| ([i as f64, if i % 2 == 0 { 0.0 } else { 0.5 }], i as f64 * 0.01))
        .collect();
    Centreline::new(&points, &[0])
}

#[test]
fn a_key_along_a_curve_is_read_only_on_the_segments_it_may_be_read_on() {
    let _one = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    let line = zigzag(50);
    let all: Vec<u32> = (0..50).collect();
    // Far above the curve: no segment's body holds the point, so only the nearest is keyed.
    counters::take();
    line.key_under([25.0, 40.0], &all, 0.25);
    let counts = counters::take();
    assert_eq!(count("key_under", &counts), 1);
    assert_eq!(count("feet", &counts), 50);
    assert_eq!(count("key_on", &counts), 1, "{counts:?}");
    // On the curve at a vertex: the two segments meeting there hold it in their bodies (and the
    // ones beside them do not), so those two are keyed and the nearest needs no key of its own.
    line.key_under([10.0, 0.0], &all, 0.25);
    let counts = counters::take();
    assert_eq!(count("key_on", &counts), 2, "{counts:?}");
}
