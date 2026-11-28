//! The scanline coverage against each pixel's exact area, where two edges start at one point or
//! a hair apart (a glyph's flattened outline can hold two points one ulp apart at a peak), and
//! where two edges cross inside a band.

use arachne_core::composite::Rect;
use arachne_core::scanline::{Scratch, cell_areas, cover, edges_of};

/// The peak, as a glyph's outline gave it, and the point one ulp to its right.
const PEAK_Y: f64 = 316.14709;
const PEAK_X: f64 = 746.0833153792969;

/// How far a pixel's coverage, or its cells' mean, may lie from its exact area: 1e-9 of a pixel,
/// the area within which the coverage snaps to 0 or 1. The three are each exact up to rounding,
/// far below it.
const TOLERANCE: f64 = 1e-9;

fn ulp_right(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1)
}

/// The signed area of `ring` inside the unit pixel at `(x, y)`: the ring clipped to the pixel's
/// square, then the shoelace sum.
fn clipped_area(ring: &[[f64; 2]], x: f64, y: f64) -> f64 {
    let mut poly = ring.to_vec();
    // Each side of the square as (axis, bound, keep below the bound).
    for (axis, bound, below) in [
        (0, x, false),
        (0, x + 1.0, true),
        (1, y, false),
        (1, y + 1.0, true),
    ] {
        let inside = |p: &[f64; 2]| {
            if below {
                p[axis] <= bound
            } else {
                p[axis] >= bound
            }
        };
        let mut out = Vec::new();
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            if inside(&a) {
                out.push(a);
            }
            if inside(&a) != inside(&b) {
                let t = (bound - a[axis]) / (b[axis] - a[axis]);
                out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            }
        }
        poly = out;
        if poly.is_empty() {
            return 0.0;
        }
    }
    0.5 * (0..poly.len())
        .map(|i| {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
}

/// Every pixel of `rect` where `ring`'s coverage, or its cells' mean, is not its exact area
/// (`exact` per pixel): `(x, y, coverage, cells' mean, exact)`.
fn wrong_pixels(
    ring: &[[f64; 2]],
    rect: Rect,
    exact: impl Fn(f64, f64) -> f64,
) -> Vec<(u32, u32, f64, f64, f64)> {
    let edges = edges_of(&[ring.to_vec()]);
    let all: Vec<u32> = (0..edges.len() as u32).collect();
    let mut area = vec![0.0; ((rect.x1 - rect.x0) * (rect.y1 - rect.y0)) as usize];
    let at = |x: u32, y: u32| ((y - rect.y0) * (rect.x1 - rect.x0) + x - rect.x0) as usize;
    cover(
        &edges,
        &all,
        rect,
        &mut Scratch::default(),
        |y, x0, x1, a| {
            for x in x0..x1 {
                area[at(x, y)] = a;
            }
        },
    );
    let mut wrong = Vec::new();
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            let want = exact(x as f64, y as f64);
            let got = area[at(x, y)];
            let mean = cell_areas(&edges, &all, x, y).iter().flatten().sum::<f64>() / 16.0;
            if (got - want).abs() > TOLERANCE || (mean - want).abs() > TOLERANCE {
                wrong.push((x, y, got, mean, want));
            }
        }
    }
    wrong
}

const AROUND_PEAK: Rect = Rect {
    x0: 742,
    y0: 314,
    x1: 751,
    y1: 319,
};

/// A rectangle with a notch cut up from its bottom side to a peak, its two sides meeting the peak
/// at `left` and `right` (the side that reaches the bottom at the left starts at `right`, the one
/// that reaches it at the right at `left`), walked one way or the other.
fn notched(left: f64, right: f64, reversed: bool) -> Vec<[f64; 2]> {
    let mut ring = vec![
        [743.5, 315.5],
        [748.7, 315.5],
        [748.7, 316.6],
        [748.0, 316.6],
        [left, PEAK_Y],
        [right, PEAK_Y],
        [744.2, 316.6],
        [743.5, 316.6],
    ];
    if reversed {
        ring.reverse();
    }
    ring
}

/// The notch's exact area in pixel `(x, y)`: the rectangle's less the notch triangle's.
fn notched_exact(x: f64, y: f64) -> f64 {
    let rectangle = [
        [743.5, 315.5],
        [748.7, 315.5],
        [748.7, 316.6],
        [743.5, 316.6],
    ];
    let notch = [[748.0, 316.6], [PEAK_X, PEAK_Y], [744.2, 316.6]];
    clipped_area(&rectangle, x, y).abs() - clipped_area(&notch, x, y).abs()
}

#[test]
fn edges_from_one_peak_point_cover_the_exact_area() {
    for reversed in [false, true] {
        let wrong = wrong_pixels(
            &notched(PEAK_X, PEAK_X, reversed),
            AROUND_PEAK,
            notched_exact,
        );
        assert!(wrong.is_empty(), "reversed {reversed}: {wrong:?}");
    }
}

#[test]
fn edges_from_peak_points_one_ulp_apart_cover_the_exact_area() {
    // The two points one ulp apart in either order, so either side starts left of the other,
    // and the ring walked both ways, so either side adds +1 or -1: the sides' order at the top
    // is not their order below it, and they meet above the band.
    let mut wrong = Vec::new();
    for (left, right) in [(PEAK_X, ulp_right(PEAK_X)), (ulp_right(PEAK_X), PEAK_X)] {
        for reversed in [false, true] {
            let found = wrong_pixels(&notched(left, right, reversed), AROUND_PEAK, notched_exact);
            if !found.is_empty() {
                wrong.push((left, right, reversed, found));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn edges_that_cross_inside_a_band_cover_the_exact_area() {
    // A bowtie whose two diagonals cross at (3.95, 3.4), inside row 3: the nonzero rule fills
    // both of its triangles.
    let ring = vec![[1.2, 1.3], [6.7, 5.5], [6.7, 1.3], [1.2, 5.5]];
    let cross = [3.95, 3.4];
    let left = [[1.2, 1.3], cross, [1.2, 5.5]];
    let right = [[6.7, 1.3], [6.7, 5.5], cross];
    let exact = |x: f64, y: f64| clipped_area(&left, x, y).abs() + clipped_area(&right, x, y).abs();
    let rect = Rect {
        x0: 0,
        y0: 0,
        x1: 8,
        y1: 7,
    };
    let wrong = wrong_pixels(&ring, rect, exact);
    assert!(wrong.is_empty(), "{wrong:?}");
}

/// A small xorshift generator: the same rings on every machine.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// `x` moved by `k` ulps (either sign).
fn ulps(x: f64, k: i64) -> f64 {
    f64::from_bits((x.to_bits() as i64 + k) as u64)
}

/// A star-shaped ring around a point far from the origin (where an ulp is about 1e-13 px), each
/// corner followed, at random, by a near-tie: a point a few ulps left or right of it, the same
/// point again, or a micro-edge a few ulps across and one ulp down; walked either way.
fn ring_with_near_ties(rng: &mut Rng) -> Vec<[f64; 2]> {
    let centre = [706.37, 306.21];
    // Four corners or more, each at its own share of the turn: no gap reaches half a turn, so the
    // centre is inside and the ring is simple (its nonzero coverage is its clipped area).
    let n = 4 + rng.below(7);
    let mut ring = Vec::new();
    for k in 0..n {
        let a = (k as f64 + 0.8 * rng.unit()) / n as f64 * std::f64::consts::TAU;
        let r = 0.5 + 4.5 * rng.unit();
        let p = [centre[0] + r * libm::cos(a), centre[1] + r * libm::sin(a)];
        ring.push(p);
        let side = if rng.below(2) == 0 { -1 } else { 1 };
        match rng.below(4) {
            0 => ring.push([ulps(p[0], side * (1 + rng.below(3) as i64)), p[1]]),
            1 => ring.push(p),
            2 => ring.push([
                ulps(p[0], side * (1 + rng.below(1000) as i64)),
                ulps(p[1], 1),
            ]),
            _ => {}
        }
    }
    if rng.below(2) == 0 {
        ring.reverse();
    }
    ring
}

#[test]
fn rings_with_near_ties_cover_the_exact_area_and_their_cells_agree() {
    let rect = Rect {
        x0: 700,
        y0: 300,
        x1: 713,
        y1: 313,
    };
    let mut failed = Vec::new();
    for seed in 1..=2000u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let ring = ring_with_near_ties(&mut rng);
        let wrong = wrong_pixels(&ring, rect, |x, y| clipped_area(&ring, x, y).abs());
        if !wrong.is_empty() {
            failed.push((seed, wrong.len(), wrong[0]));
        }
    }
    assert!(
        failed.is_empty(),
        "{} of 2000 seeds cover a pixel off its exact area or its cells' mean, first {:?}",
        failed.len(),
        &failed[..failed.len().min(5)]
    );
}
