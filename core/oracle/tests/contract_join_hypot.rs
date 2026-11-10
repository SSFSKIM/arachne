//! The core held to §8's mitre test, `2 h < (h + ab · cd) limit²` with `h = hypot(ab × cd, ab ·
//! cd)`, at corners where the `hypot` chosen decides it: libm's (`libm::hypot`) against the
//! platform's (`f64::hypot`).
//!
//! Each corner arrives along `ab = (4, 0)` at the origin and leaves along `cd`, a straight cubic
//! whose first control point is `cd` itself, so kurbo's tangents are exactly `ab` and `cd` in
//! frame units and on screen alike (each coordinate lies on a `2^-47` grid, which the view's
//! translation keeps exact). Each `cd` lies within an ulp of `h` of the boundary, where libm's
//! `hypot` and the platform's round to different doubles on the opposite sides of it. The core
//! mitres a corner where its frame stroked with a miter joint differs from its frame bevelled.

mod common;

use arachne_core::stream::Stream;
use arachne_core::xf::Joint;
use common::{delta, path, path_shape, view};

/// Corners `cd` as the bits of `(x, y)`, and whether libm's `hypot` mitres there. The first four
/// separate libm's decision from both glibc's and macOS's; the fifth from macOS's alone, the sixth
/// from glibc's alone.
const CORNERS: [(u64, u64, bool); 6] = [
    (0xbff06e57e2a33de0, 0x3fcab11353bd4d00, true),
    (0xbff76bb001212980, 0x3fd305e7b51d0000, true),
    (0xbffb0b11a78c7620, 0x3fd5f7280cbb8d80, false),
    (0xbffa3c01e323bf80, 0x3fd54ef94e0c4980, false),
    (0xc0017f3b35a0e940, 0x3fdc6c5fa74d0100, true),
    (0xbfe36153cdef3040, 0x3fbf7b861f0bd200, false),
];

const AB: [f64; 2] = [4.0, 0.0];

/// §8's mitre test for the corner `AB` into `cd`, with `hypot` the one given.
fn mitres(cd: [f64; 2], hypot: fn(f64, f64) -> f64) -> bool {
    let cross = AB[0] * cd[1] - AB[1] * cd[0];
    let dot = AB[0] * cd[0] + AB[1] * cd[1];
    let h = hypot(cross, dot);
    2.0 * h < (h + dot) * libm::pow(10.0, 2.0)
}

#[allow(clippy::disallowed_methods)] // the platform's `hypot`, compared, never used to decide
fn platform_hypot(a: f64, b: f64) -> f64 {
    a.hypot(b)
}

/// The core's frame of the corner stroked 1 unit wide with `joint`, over 64 x 64 px.
fn frame(cd: [f64; 2], joint: Joint) -> Vec<u8> {
    let [x, y] = cd;
    let shape = path_shape(vec![
        [-12.0, 0.0, 0.0],
        [-8.0, 0.0, 0.0],
        [-4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [x, y, 0.0],
        [2.0 * x, 2.0 * y, 0.0],
        [3.0 * x, 3.0 * y, 0.0],
    ]);
    let mut r = path(0, shape.key(), false, Some(1.0));
    r.paint.joint = joint;
    let mut s = Stream::new();
    s.apply(&delta(1, Some(view(64, 64)), vec![shape], vec![r]))
        .expect("the corner applies");
    s.render(64, 64, 1).expect("the corner renders").rgba
}

fn core_mitres(cd: [f64; 2]) -> bool {
    frame(cd, Joint::Miter) != frame(cd, Joint::Bevel)
}

#[test]
fn a_miter_and_a_bevel_are_told_apart_by_the_frame() {
    // A 45° corner mitres far inside the limit; one of about 2.9° bevels far outside it.
    assert!(core_mitres([-1.0, 1.0]));
    assert!(!core_mitres([-1.0, 0.05]));
}

#[test]
fn the_core_decides_the_miter_with_libms_hypot() {
    let mut separating = 0;
    for (x, y, by_libm) in CORNERS {
        let cd = [f64::from_bits(x), f64::from_bits(y)];
        assert_eq!(mitres(cd, libm::hypot), by_libm, "{cd:?}: libm's decision");
        let by_platform = mitres(cd, platform_hypot);
        separating += usize::from(by_platform != by_libm);
        assert_eq!(
            core_mitres(cd),
            by_libm,
            "{cd:?}: the core's decision, against libm's {by_libm} and the platform's {by_platform}"
        );
    }
    assert!(
        separating > 0,
        "no corner separates libm's hypot from this platform's"
    );
}
