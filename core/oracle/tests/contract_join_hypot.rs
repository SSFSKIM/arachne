//! The core held to §8's mitre test, `2 h < (h + ab · cd) limit²` with `h = hypot(ab × cd, ab ·
//! cd)`, at corners where the `hypot` chosen decides it: libm's (`libm::hypot`) against the
//! platform's (`f64::hypot`).
//!
//! Each corner arrives along `ab = (4, 0)` at the origin and leaves along `cd`, a straight cubic
//! whose first control point is `cd` itself, so kurbo's tangents are exactly `ab` and `cd` in
//! frame units and on screen alike (each coordinate lies on a `2^-47` grid, which the view's
//! translation keeps exact). Each `cd` lies within an ulp of `h` of the boundary, where libm's
//! `hypot` and the platform's can round to different doubles on opposite sides of it. Which
//! corners separate the two depends on the platform's `hypot`, which differs between machines, so
//! the test takes pinned corners and searches for more on the machine it runs on. The core mitres
//! a corner where its frame stroked with a miter joint differs from its frame bevelled.

mod common;

use arachne_core::stream::Stream;
use arachne_core::xf::Joint;
use common::{delta, path, path_shape, view};

/// Corners `cd` as the bits of `(x, y)`, and whether libm's `hypot` mitres there. Against one
/// glibc 2.39 and one macOS, the first four separate libm's decision from both platforms'; the
/// fifth from macOS's alone, the sixth from glibc's alone.
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

/// Up to `n` corners near the boundary, on the `2^-47` grid, where libm's decision and this
/// platform's differ: for each of a fixed sequence of lengths `r` in `[0.5, 2.5]`, the boundary's
/// `cd = (-0.98 r, 0.198997487421324 r)`, its `x` stepped through 60 grid steps each way.
fn separating_here(n: usize) -> Vec<[f64; 2]> {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let g = libm::ldexp(1.0, -47);
    let on_grid = |v: f64| libm::round(v / g) * g;
    let mut found = Vec::new();
    for _ in 0..20_000 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let r = 0.5 + 2.0 * ((seed >> 11) as f64 / (1u64 << 53) as f64);
        let (x0, y) = (on_grid(-0.98 * r), on_grid(0.198_997_487_421_324 * r));
        for k in -60i32..=60 {
            let cd = [x0 + f64::from(k) * g, y];
            if mitres(cd, libm::hypot) != mitres(cd, platform_hypot) {
                found.push(cd);
                if found.len() == n {
                    return found;
                }
            }
        }
    }
    found
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

fn holds_to_libm(cd: [f64; 2]) -> bool {
    let (by_libm, by_platform) = (mitres(cd, libm::hypot), mitres(cd, platform_hypot));
    assert_eq!(
        core_mitres(cd),
        by_libm,
        "{cd:?}: the core's decision, against libm's {by_libm} and the platform's {by_platform}"
    );
    by_libm != by_platform
}

#[test]
fn a_miter_and_a_bevel_are_told_apart_by_the_frame() {
    // A 45° corner mitres far inside the limit; one of about 2.9° bevels far outside it.
    assert!(core_mitres([-1.0, 1.0]));
    assert!(!core_mitres([-1.0, 0.05]));
}

#[test]
fn the_pinned_corners_are_decided_with_libms_hypot() {
    for (x, y, by_libm) in CORNERS {
        let cd = [f64::from_bits(x), f64::from_bits(y)];
        assert_eq!(mitres(cd, libm::hypot), by_libm, "{cd:?}: libm's decision");
        holds_to_libm(cd);
    }
}

#[test]
fn corners_separating_libm_from_this_platform_are_decided_with_libms() {
    let found = separating_here(8);
    assert!(
        !found.is_empty(),
        "no corner searched separates libm's hypot from this platform's"
    );
    for cd in found {
        assert!(holds_to_libm(cd));
    }
}
