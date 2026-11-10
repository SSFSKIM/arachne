//! `join_mask@1` (contract §8) through the oracle's decisions: which join a carried decision lands
//! on, and what each edit adds or takes away from kurbo's stroke.

mod common;

use arachne_core::xf::Joint;
use common::*;

/// The total stroke area of one path record through `points` (pixel positions), `width` px wide,
/// under `joint`, carrying `joins`.
fn total(points: &[[f64; 3]], width: f64, joint: Joint, joins: Option<Vec<u8>>) -> f64 {
    let v = view(64, 64);
    let s = path_shape(points.to_vec());
    let mut r = path(0, s.key(), false, Some(width));
    r.joins = joins;
    r.paint.joint = joint;
    let d = decide(&[delta(1, Some(v), vec![s], vec![r])]);
    d.covers.iter().map(|c| c.pixel_area).sum::<f64>()
}

/// A sharp V, its apex at `(0, -0.5)` in frame units, its arms 6 long and `2·atan(1/12)` apart:
/// past the miter limit, so kurbo bevels it.
fn sharp_v() -> Vec<[f64; 3]> {
    let at = |x: f64, y: f64| [x, y, 0.0];
    polyline(&[at(-6.0, 0.0), at(0.0, -0.5), at(-6.0, -1.0)])
}

/// The tip a carried miter adds beyond kurbo's bevel at the V: the triangle from the bevel's
/// two outer corners to the tip on the bisector at the limit's length, `10 · w / 2`.
fn tip_area(width: f64) -> f64 {
    let half = libm::atan(0.5 / 6.0);
    let hw = width / 2.0;
    let base = 2.0 * hw * libm::cos(half);
    let height = 10.0 * hw - hw * libm::sin(half);
    0.5 * base * height
}

#[test]
fn a_carried_miter_where_kurbo_bevels_adds_the_tip_at_the_limits_length() {
    let pts = sharp_v();
    let w = 0.5;
    let kurbo = total(&pts, w, Joint::Miter, None);
    assert_eq!(
        kurbo,
        total(&pts, w, Joint::Bevel, None),
        "kurbo bevels the V"
    );
    let carried = total(&pts, w, Joint::Miter, Some(vec![0, 0b11]));
    assert!(
        (carried - kurbo - tip_area(w)).abs() < 1e-9,
        "{carried} - {kurbo} against {}",
        tip_area(w)
    );
    // A carried bevel where kurbo bevels changes nothing.
    assert_eq!(total(&pts, w, Joint::Miter, Some(vec![0, 0b01])), kurbo);
}

#[test]
fn each_occurrence_of_one_join_takes_its_own_decision() {
    // The V twice, as two subpaths: one join key, two occurrences, each edited within its own
    // subpath's contours. The union shows the tip wherever either occurrence carries a miter.
    let mut pts = sharp_v();
    pts.extend(sharp_v());
    let w = 0.5;
    let kurbo = total(&pts, w, Joint::Miter, None);
    let with = |bytes: [u8; 4]| total(&pts, w, Joint::Miter, Some(bytes.to_vec())) - kurbo;
    let tip = tip_area(w);
    for (bytes, tipped) in [
        ([0, 0b01, 0, 0b11], true),
        ([0, 0b11, 0, 0b01], true),
        ([0, 0, 0, 0b11], true),
        ([0, 0b01, 0, 0b01], false),
        ([0, 0b01, 0, 0], false),
    ] {
        let want = if tipped { tip } else { 0.0 };
        assert!(
            (with(bytes) - want).abs() < 1e-9,
            "{bytes:?}: {}",
            with(bytes)
        );
    }
}

#[test]
fn a_join_on_parallel_tangents_is_never_edited() {
    // There and back: the turn reverses, kurbo bevels it (no miter exists), and a carried miter
    // there stays bevelled.
    let at = |x: f64, y: f64| [x, y, 0.0];
    let pts = polyline(&[at(-8.0, 0.5), at(8.0, 0.5), at(-4.0, 0.5)]);
    let kurbo = total(&pts, 2.0, Joint::Miter, None);
    assert_eq!(total(&pts, 2.0, Joint::Miter, Some(vec![0, 0b11])), kurbo);
}
