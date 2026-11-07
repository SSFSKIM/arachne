//! The depth read along a curve does only the work that can change what it reads: it keys only
//! the segments it may pick, and searches a long curve only near the pixel it reads. Counted in a
//! build with the `counters` feature (`cargo test --features counters --test counters`).
#![cfg(feature = "counters")]

mod common;

use std::sync::Mutex;

use arachne_core::counters;
use arachne_core::depth::Centreline;
use arachne_core::stream::Stream;
use arachne_core::xf::*;
use common::three_d_view;

/// The counts are shared by every thread of the process: one test reads them at a time.
static COUNTS: Mutex<()> = Mutex::new(());

fn count(name: &str, counts: &[(&str, u64)]) -> u64 {
    counts.iter().find(|(n, _)| *n == name).unwrap().1
}

/// A zigzag of `n` segments along y = 0, each a unit long, keyed by how far along it lies.
fn zigzag(n: usize) -> Centreline {
    let points: Vec<([f64; 2], f64)> = (0..=n)
        .map(|i| {
            (
                [i as f64, if i % 2 == 0 { 0.0 } else { 0.5 }],
                i as f64 * 0.01,
            )
        })
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

/// Straight segments through `points` in CE's cubic layout.
fn lines(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

/// Segments in the loop [`loop_counts`] draws.
const N: usize = 2000;

/// The counts of one draw of a closed, stroked loop of [`N`] segments across a production-sized
/// frame, filled where `fill`.
fn loop_counts(fill: bool) -> Vec<(&'static str, u64)> {
    // A closed, filled and stroked loop of 2000 segments across a production-sized frame, upright
    // before the camera, its points on no one plane (a wave in depth around it), so its stroke
    // and fill are keyed along it, its fill far inside from every segment.
    let (w, h) = (854, 480);
    let mut points: Vec<[f64; 3]> = (0..N)
        .map(|i| {
            let t = i as f64 / N as f64 * std::f64::consts::TAU;
            [
                5.5 * libm::cos(t),
                0.8 * libm::sin(9.0 * t),
                3.0 * libm::sin(t),
            ]
        })
        .collect();
    points.push(points[0]);
    let shape = Shape::new(ShapeData::Path {
        points: lines(&points),
    });
    let mut r = Record::new(RecordKind::Path, 1, 101);
    r.geometry.shape_a = Some(shape.key());
    r.paint.fill = if fill {
        Paint::One([0.2, 0.4, 0.9, 0.6])
    } else {
        Paint::None
    };
    r.paint.stroke = Paint::One([1.0, 1.0, 1.0, 1.0]);
    r.paint.stroke_width = 4.0;
    r.depth.space = Space::World;
    let d = arachne_core::probes::first_delta(three_d_view(w, h), &[(shape, r)], &[1]);
    let mut s = Stream::new();
    s.apply(&d).expect("the delta applies");
    counters::take();
    s.draw(w, h, 1).unwrap();
    counters::take()
}

#[test]
fn a_long_curve_is_searched_only_near_the_pixel_read() {
    let _one = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    let counts = loop_counts(true);
    let (near, tested) = (count("near", &counts), count("bounds_tested", &counts));
    assert!(near > 0 && count("key_under", &counts) > 0, "{counts:?}");
    // A search tests the bounds near the pixel, and a read projects onto the segments near it,
    // even inside the fill, far from every segment: never the whole curve's.
    assert!(
        tested / near < N as u64 / 4,
        "{} bounds tested a search: {counts:?}",
        tested / near
    );
    assert!(
        count("most_feet", &counts) < N as u64 / 4,
        "a read projected onto {} segments: {counts:?}",
        count("most_feet", &counts)
    );
}

#[test]
fn a_pixel_s_reads_project_only_onto_the_segments_that_may_reach_it() {
    let _one = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    let counts = loop_counts(true);
    let (reads, feet) = (count("key_under", &counts), count("feet", &counts));
    assert!(reads > 0, "{counts:?}");
    // A pixel's points are read among the few segments within reach of them, not among every
    // segment whose bounds hold its centre.
    assert!(
        feet / reads < N as u64 / 64,
        "{} segments projected onto a read: {counts:?}",
        feet / reads
    );
}

#[test]
fn a_pixel_s_candidates_are_found_near_it_not_among_every_segment_whose_bounds_hold_it() {
    let _one = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    // A stroke alone: every pixel read lies on it, within a few pixels of its nearest segment.
    let counts = loop_counts(false);
    let (reads, gaps) = (count("key_under", &counts), count("gaps", &counts));
    assert!(reads > 0, "{counts:?}");
    // The segments a pixel's reads are made among are found from the few near it: the bounds
    // measured stay few, not the hundreds whose widened bounds hold the pixel's centre.
    assert!(
        gaps / reads < N as u64 / 16,
        "{} bounds measured a read: {counts:?}",
        gaps / reads
    );
}
