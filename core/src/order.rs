//! The order fragments composite in at a pixel. Depth decides the order of two fragments only
//! where both cover, and only where their depths are distinguishable within what the core knows
//! of them: at a sub-sample both cover, the nearer is in front unless the two depths lie within
//! their own error bounds of each other ([`tied`]); everywhere else, and at such a tie, they keep
//! the stream's paint order (the later drawn in front), which is the order Cairo paints them in.
//!
//! A depth's error bound ([`crate::composite::depth_error`]) is how far it may stand from the
//! exact depth of the core's own inputs for it (the snapped points and the normals carried): the
//! plane's [`crate::composite::Plane::err`], set by its constructor from its inputs, and the
//! rounding of reading the plane at the point, `4u (|a| + |b x| + |c y|)` (`u = 2^-53`; it grows
//! with the key's gradient times the pixel's coordinate, as the planes are anchored at the
//! frame's origin). The constructors count:
//!
//! - a face's plane on screen (`Projection::key_plane`): its world plane rounded to
//!   [`crate::depth::PLANE_GRID`] (each normal component and the offset within half a step), and
//!   its own arithmetic, chiefly the denominator, the plane's distance from the eye, which cancels
//!   as the plane nears the eye; each over the screen box its fragments lie in;
//! - the plane through a triangle's keyed corners (`Plane::through_with_errors`): the corners'
//!   keys' errors, and the determinant's cancellation, `u L / h` for a thin triangle;
//! - the plane along a curve's segment (`Centreline::plane_at`): its ends' keys' errors carried
//!   across the box it is read in, and the gradient's rounding;
//! - a point's key (`Projection::key_error`): the rotation's and the perspective's rounding, which
//!   grows as the point nears the eye.
//!
//! The bound does not count the snap of the corners to the input grid: every representation of a
//! shared edge or face is made from the same snapped corners, so the snap is common input, and
//! counting it would tie geometry that is really near-coincident. Nor does it count a model's
//! departure from the geometry (a segment's plane is flat across the segment; a face's plane is
//! the face's): two such depths that differ are two depths, and depth orders them.
//! One order for the pixel (what the coverage composite and the contributor report's leaf order
//! read) takes each pair's order from the sub-samples both cover ([`pair`]); one order per
//! sub-sample (what the sample resolve composites) takes each pair's order there. Both are ranked
//! from those pairwise orders by one rule ([`rank`]), so a cycle among them is broken the same way
//! wherever it is read.

use crate::composite::{Fragment, SAMPLES};
use std::cmp::Ordering;

/// A fragment as the order reads it at a pixel: its coverage and depth at each sub-sample, and how
/// far that depth may stand from the exact one.
#[derive(Clone, Copy)]
pub struct Seen<'a> {
    pub fragment: &'a Fragment,
    pub coverage: &'a [u8; SAMPLES],
    pub depth: &'a [f64; SAMPLES],
    pub err: f64,
}

/// How a pair's order at a pixel was decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Decided {
    /// No sub-sample both cover: paint order.
    Paint,
    /// Every sub-sample both cover holds them at one depth within their errors: paint order.
    Tie,
    /// Every sub-sample both cover and that is not a tie puts the same one in front.
    Depth,
    /// The sub-samples both cover disagree (they cross inside the pixel): the one in front at more
    /// of them, paint order where as many put each in front.
    Majority,
}

impl Decided {
    /// Whether depth decided the pair ([`Decided::Depth`], [`Decided::Majority`]): a strong
    /// order, which [`rank`] keeps before any weak one (paint order, where no sub-sample is shared
    /// or depth ties).
    pub fn is_strong(self) -> bool {
        matches!(self, Decided::Depth | Decided::Majority)
    }
}

/// Whether depths `a` (within `ea` of its exact value) and `b` (within `eb`) may be one depth.
/// Depths that are not finite (`-inf`, in front of the scene; `+inf`) tie only with themselves,
/// before any subtraction.
pub fn tied(a: f64, ea: f64, b: f64, eb: f64) -> bool {
    a == b || a.is_finite() && b.is_finite() && (a - b).abs() <= ea + eb
}

/// `a` against `b` in paint order: [`Ordering::Less`] when `a` is drawn later (in front).
pub fn painted(a: &Fragment, b: &Fragment) -> Ordering {
    b.order.cmp(&a.order).then(b.sub.cmp(&a.sub))
}

/// `a` (depth `da` within `ea`) against `b` (depth `db` within `eb`) at a sub-sample both cover:
/// [`Ordering::Less`] when `a` is in front. The nearer, unless they tie; then paint order.
pub fn at_sample(a: &Fragment, da: f64, ea: f64, b: &Fragment, db: f64, eb: f64) -> Ordering {
    if tied(da, ea, db, eb) {
        painted(a, b)
    } else {
        da.total_cmp(&db)
    }
}

/// Why a sub-sample's rank always finds an order: its strong orders are depth's at one point,
/// each pair's depths farther apart than their two bounds, which no cycle can be.
pub const ONE_SAMPLE: &str =
    "a sub-sample's strong orders are depth's at one point, which do not cycle";

/// Whether [`at_sample`]'s order of `a` and `b` is strong, depth deciding it (see [`rank`]); the
/// sub-sample's counterpart of [`Decided::is_strong`]. A rule that keeps paint order and is to hold
/// against depth joins here and there.
pub fn strong_at(_a: &Fragment, da: f64, ea: f64, _b: &Fragment, db: f64, eb: f64) -> bool {
    !tied(da, ea, db, eb)
}

/// `a` against `b` at the pixel, from their coverage and depth at each sub-sample, and how that was
/// decided: [`Ordering::Less`] when `a` is in front.
// The sub-sample `k` indexes both fragments' coverage and depth rows.
#[allow(clippy::needless_range_loop)]
pub fn pair(a: Seen, b: Seen) -> (Ordering, Decided) {
    let (mut shared, mut a_front, mut b_front) = (0, 0, 0);
    let (ca, da, cb, db) = (a.coverage, a.depth, b.coverage, b.depth);
    for k in 0..SAMPLES {
        if ca[k] == 0 || cb[k] == 0 {
            continue;
        }
        shared += 1;
        if !tied(da[k], a.err, db[k], b.err) {
            if da[k] < db[k] {
                a_front += 1;
            } else {
                b_front += 1;
            }
        }
    }
    let paint = painted(a.fragment, b.fragment);
    match (shared, a_front, b_front) {
        (0, _, _) => (paint, Decided::Paint),
        (_, 0, 0) => (paint, Decided::Tie),
        (_, _, 0) => (Ordering::Less, Decided::Depth),
        (_, 0, _) => (Ordering::Greater, Decided::Depth),
        _ => (
            a_front.cmp(&b_front).reverse().then(paint),
            Decided::Majority,
        ),
    }
}

/// How [`rank`] found its order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Ranked {
    /// The pairwise orders were one order.
    One = 0,
    /// They formed a cycle through a weak order, which yielded.
    Weak = 1,
    /// The strong orders alone formed a cycle: no one order keeps them ([`rank`] stops there,
    /// [`rank_through_cycles`] breaks it by wins).
    Strong = 2,
}

/// `items` ranked front first into `out`, from `front(a, b)` (whether `a` is in front of `b`, for
/// every pair) and `strong(a, b)` (whether depth decided that pair: [`Decided::is_strong`] at the
/// pixel, [`strong_at`] at a sub-sample; the others are weak, paint order where no sub-sample is
/// shared or where depth ties). Where the pairwise orders are
/// one order, that order. Where they are not, the weak orders yield first: every strong order is
/// kept, and among the items no strong order left puts behind another, the later painted
/// (`painted`) goes first, which keeps every weak order it can, a weak order being paint order.
/// So the rank is a function of the pairwise orders, their strength and paint order alone. Where
/// the strong orders alone form a cycle there is no such order: [`Ranked::Strong`], with `out`
/// holding only the front ranked before the cycle. (A leaf's fill and its own stroke, kept in
/// self-order, are to join the strong orders, through those two.)
pub fn rank(
    items: &[usize],
    front: impl Fn(usize, usize) -> bool,
    strong: impl Fn(usize, usize) -> bool,
    painted: impl Fn(usize, usize) -> Ordering,
    out: &mut Vec<usize>,
) -> Ranked {
    ranked(items, front, strong, painted, out, false)
}

/// [`rank`], and where the strong orders alone form a cycle, the item in front of the most of those
/// left goes first, the later painted where as many. Only the contributor report reads this, to
/// list a pixel's leaves in one order; no draw path may read it. The resolve composites such a
/// pixel at each sub-sample, in that sub-sample's own order, and reads no order of the pixel.
pub fn rank_through_cycles(
    items: &[usize],
    front: impl Fn(usize, usize) -> bool,
    strong: impl Fn(usize, usize) -> bool,
    painted: impl Fn(usize, usize) -> Ordering,
    out: &mut Vec<usize>,
) -> Ranked {
    ranked(items, front, strong, painted, out, true)
}

fn ranked(
    items: &[usize],
    front: impl Fn(usize, usize) -> bool,
    strong: impl Fn(usize, usize) -> bool,
    painted: impl Fn(usize, usize) -> Ordering,
    out: &mut Vec<usize>,
    through_cycles: bool,
) -> Ranked {
    // One order: then each is in front of all after it, and it is the only order that keeps every
    // pair, so inserting each item before the first it is in front of finds it. Checked, since the
    // pairwise orders need not be one order. It runs for every pixel and every sub-sample of one,
    // over a few fragments, so it inserts in place.
    out.clear();
    for &i in items {
        let at = out.iter().position(|&j| front(i, j)).unwrap_or(out.len());
        out.insert(at, i);
    }
    let one = (0..out.len()).all(|p| (p + 1..out.len()).all(|q| front(out[p], out[q])));
    if one {
        return Ranked::One;
    }
    let wins = |i: usize, among: &[usize]| among.iter().filter(|&&j| j != i && front(i, j)).count();
    let mut how = Ranked::Weak;
    let mut left = items.to_vec();
    out.clear();
    while !left.is_empty() {
        let free = left
            .iter()
            .copied()
            .filter(|&i| !left.iter().any(|&j| j != i && strong(j, i) && front(j, i)));
        let pick = match free.min_by(|&a, &b| painted(a, b)) {
            Some(i) => i,
            None if through_cycles => {
                how = Ranked::Strong;
                left.iter()
                    .copied()
                    .min_by(|&a, &b| wins(b, &left).cmp(&wins(a, &left)).then(painted(a, b)))
                    .expect("something is left")
            }
            None => return Ranked::Strong,
        };
        out.push(pick);
        left.retain(|&i| i != pick);
    }
    how
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::{
        Coverage, Plane, Resolve, SAMPLE_OFFSETS, Scratch, blend, depth_error, resolve_pixel,
        sample_point,
    };

    fn at_depth(depth: Plane, colour: [u8; 4], coverage: Coverage, order: u32) -> Fragment {
        Fragment::new(depth, colour, coverage, order, 0)
    }

    /// The pixel the resolve makes of `frags` over black, fragment `i` covering the sub-samples
    /// `rows[i]` gives it.
    fn resolve(frags: &[Fragment], rows: &[[u8; SAMPLES]]) -> (Resolve, [u8; 4]) {
        let mut s = Scratch::default();
        s.fragments.extend_from_slice(frags);
        resolve_pixel(&mut s, [0, 0, 0, 255], 0, 0, |i, _| rows[i])
    }

    /// `a` against `b` at pixel (0, 0), as the resolve reads them there.
    fn ordered(
        a: &Fragment,
        ca: &[u8; SAMPLES],
        b: &Fragment,
        cb: &[u8; SAMPLES],
    ) -> (Ordering, Decided) {
        let depth = |f: &Fragment| crate::composite::sample_depths(f, 0, 0);
        let (da, db) = (depth(a), depth(b));
        let seen = |f, coverage, depth| Seen {
            fragment: f,
            coverage,
            depth,
            err: depth_error(f, 0, 0),
        };
        pair(seen(a, ca, &da), seen(b, cb, &db))
    }

    fn only(samples: &[usize]) -> [u8; SAMPLES] {
        let mut row = [0; SAMPLES];
        for &s in samples {
            row[s] = 255;
        }
        row
    }

    const RED: [u8; 4] = [200, 0, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 200, 255];
    const FULL: [u8; SAMPLES] = [255; SAMPLES];

    #[test]
    fn depths_that_are_not_finite_tie_only_with_themselves_before_any_subtraction() {
        // `-inf - -inf` is NaN, which no bound holds: equal keys tie before a gap is taken.
        let inf = f64::INFINITY;
        assert!(tied(-inf, 0.0, -inf, 0.0));
        assert!(tied(inf, 0.0, inf, 0.0));
        assert!(!tied(-inf, 1.0, -1.0, 1.0));
        assert!(!tied(-inf, inf, inf, inf));
        let first = at_depth(Plane::constant(-inf), RED, Coverage::Mask(u16::MAX), 1);
        let later = at_depth(Plane::constant(-inf), BLUE, Coverage::Mask(u16::MAX), 2);
        assert_eq!(
            ordered(&first, &FULL, &later, &FULL),
            (Ordering::Greater, Decided::Tie)
        );
    }

    #[test]
    fn a_stroke_one_ulp_nearer_where_it_does_not_cover_keeps_paint_order() {
        // As in a camera orbit's frame (AmbientCameraMove, pixel 397,161): a stroke drawn first,
        // its key one ulp nearer than a later stroke's, covering part of the pixel; the later
        // stroke covering it whole. Where both cover they are one depth within what reading them
        // rounds, so the later is in front, as Cairo paints them, and the earlier shows nowhere.
        let near = Plane::constant(-1.0526315789473686);
        let far = Plane::constant(-1.0526315789473684);
        let first = at_depth(near, RED, Coverage::Area { area: 0.25, id: 0 }, 1);
        let later = at_depth(far, BLUE, Coverage::Area { area: 1.0, id: 0 }, 5);
        let rows = [only(&[1, 5, 9, 13]), FULL];
        assert_eq!(
            ordered(&first, &rows[0], &later, &FULL),
            (Ordering::Greater, Decided::Tie)
        );
        assert_eq!(resolve(&[first, later], &rows), (Resolve::Fast, BLUE));
    }

    #[test]
    fn a_cap_and_its_outline_one_rounding_apart_composite_in_paint_order() {
        // A cone's cap (a mesh) and the outline drawn after it, their keys 1e-16 apart: the
        // outline over the cap, though its key is the farther.
        let cap = at_depth(Plane::constant(-0.8), RED, Coverage::Mask(u16::MAX), 1);
        let outline = Plane::constant(-0.8 + 1e-16);
        let outline = at_depth(outline, BLUE, Coverage::Area { area: 0.5, id: 0 }, 2);
        let half = only(&[0, 2, 4, 6, 8, 10, 12, 14]);
        let (class, got) = resolve(&[cap, outline], &[FULL, half]);
        assert_eq!(class, Resolve::Coverage);
        assert_eq!(got, blend(BLUE, 128, RED));
    }

    #[test]
    fn a_pair_just_beyond_their_error_bounds_is_ordered_by_depth() {
        // A cap and an outline drawn after it: within the two bounds they tie and the outline is
        // over the cap; just beyond them, the nearer cap hides it. Once with the cap known within
        // 1e-9, and once with both exact, whose bounds are their reading's rounding alone (about
        // 1e-15 here): there a gap far below a tolerance taken relative to the depth (2^-46 of it
        // is 1e-14) is still told apart, as the core's own arithmetic tells it apart.
        let d = -0.8;
        let half = only(&[0, 2, 4, 6, 8, 10, 12, 14]);
        let outline = |gap: f64| {
            at_depth(
                Plane::constant(d + gap),
                BLUE,
                Coverage::Area { area: 0.5, id: 0 },
                2,
            )
        };
        for err in [1e-9, 0.0] {
            let cap = Plane {
                err,
                ..Plane::constant(d)
            };
            let cap = at_depth(cap, RED, Coverage::Mask(u16::MAX), 1);
            let bound = depth_error(&cap, 0, 0) + depth_error(&outline(0.0), 0, 0);
            assert!(err > 0.0 || bound < 1e-14, "{bound:e}");
            assert_eq!(
                resolve(&[cap, outline(0.9 * bound)], &[FULL, half]).1,
                blend(BLUE, 128, RED),
                "within the bounds, err {err:e}"
            );
            assert_eq!(
                resolve(&[cap, outline(1.1 * bound)], &[FULL, half]),
                (Resolve::Fast, RED),
                "beyond the bounds, err {err:e}"
            );
        }
    }

    #[test]
    fn a_pair_crossing_inside_the_pixel_takes_the_order_most_shared_sub_samples_give() {
        // `a` is in front left of x = 0.6, `b` (constant) right of it; `b` covers one sub-sample
        // left of the crossing and the rest of what it covers right of it.
        let a = at_depth(Plane::new(-0.6, 1.0, 0.0), RED, Coverage::Mask(0), 1);
        let b = at_depth(Plane::constant(0.0), BLUE, Coverage::Mask(0), 2);
        let x = |k: usize| sample_point(0, 0, k)[0];
        let mut covered: Vec<usize> = (0..SAMPLES).filter(|k| x(*k) > 0.6).collect();
        let right = covered.len();
        assert!(right > 1);
        covered.push((0..SAMPLES).find(|k| x(*k) < 0.6).unwrap());
        let rb = only(&covered);
        assert_eq!(
            ordered(&a, &FULL, &b, &rb),
            (Ordering::Greater, Decided::Majority)
        );
        // As many each way: paint order, the later in front.
        let even = only(&[covered[0], covered[right]]);
        assert_eq!(
            ordered(&a, &FULL, &b, &even),
            (Ordering::Greater, Decided::Majority)
        );
        assert_eq!(
            ordered(&b, &even, &a, &FULL),
            (Ordering::Less, Decided::Majority)
        );
    }

    #[test]
    fn the_report_breaks_a_cycle_of_strong_orders_by_wins_then_paint_order() {
        // 0 over 1, 1 over 2, 2 over 0, each by depth: one win each, so paint order (2 drawn last)
        // places the first, and the rest follow the pairs. The rank itself takes no order.
        let front = |a: usize, b: usize| matches!((a, b), (0, 1) | (1, 2) | (2, 0));
        let painted = |a: usize, b: usize| b.cmp(&a);
        let mut out = Vec::new();
        assert_eq!(
            rank_through_cycles(&[0, 1, 2], front, |_, _| true, painted, &mut out),
            Ranked::Strong
        );
        assert_eq!(out, vec![2, 0, 1]);
        // Whatever order the items come in.
        rank_through_cycles(&[1, 2, 0], front, |_, _| true, painted, &mut out);
        assert_eq!(out, vec![2, 0, 1]);
        // `rank` takes no order there.
        assert_eq!(
            rank(&[0, 1, 2], front, |_, _| true, painted, &mut out),
            Ranked::Strong
        );
        assert!(out.is_empty());
    }

    #[test]
    fn a_cycle_through_a_weak_order_keeps_every_strong_order() {
        // 0 over 1 and 1 over 2 by depth, 2 over 0 by paint order (2 drawn last): the paint order
        // yields, though the wins alone would put 2 first and 1 over 2 would be lost.
        let front = |a: usize, b: usize| matches!((a, b), (0, 1) | (1, 2) | (2, 0));
        let strong = |a: usize, b: usize| !matches!((a, b), (2, 0) | (0, 2));
        let painted = |a: usize, b: usize| b.cmp(&a);
        let mut out = Vec::new();
        for items in [[0, 1, 2], [2, 1, 0], [1, 2, 0]] {
            assert_eq!(rank(&items, front, strong, painted, &mut out), Ranked::Weak);
            assert_eq!(out, vec![0, 1, 2], "from {items:?}");
        }
        // Among the items no strong order puts behind another, paint order chooses: 3, drawn last
        // and over the others by paint order alone, goes first.
        let front4 = |a: usize, b: usize| front(a, b) || a == 3;
        let strong4 = |a: usize, b: usize| a != 3 && b != 3 && strong(a, b);
        rank(&[0, 1, 2, 3], front4, strong4, painted, &mut out);
        assert_eq!(out, vec![3, 0, 1, 2]);
        // Acyclic, the order the pairs give, whatever their strength.
        let line = |a: usize, b: usize| a < b;
        assert_eq!(
            rank(&[2, 0, 1], line, |_, _| false, painted, &mut out),
            Ranked::One
        );
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn the_cull_keeps_what_the_rank_shows_and_drops_what_it_hides() {
        // Three layers over the whole pixel at depths 0 (`near`, translucent, drawn first), 1
        // (`wall`, opaque) and 2 (`far`, translucent, drawn last), each within 0.6 of its exact
        // depth: `near` and `far` are told apart, the others tie, so paint order puts the wall over
        // `near` and `far` over the wall, a cycle through two weak orders. The strong order keeps
        // `near` over `far`; the wall goes first, hiding both at every sub-sample. Judged pair by
        // pair against the wall, `far` (over it by paint order) would be kept and composited over
        // it, which the rank does not.
        let layer = |d: f64, colour, order| {
            at_depth(
                Plane {
                    err: 0.6,
                    ..Plane::constant(d)
                },
                colour,
                Coverage::Mask(u16::MAX),
                order,
            )
        };
        let near = layer(0.0, [0, 200, 0, 128], 1);
        let wall = layer(1.0, RED, 2);
        let far = layer(2.0, [0, 0, 200, 128], 3);
        let frags = [near, wall, far];
        assert_eq!(resolve(&frags, &[FULL, FULL, FULL]), (Resolve::Fast, RED));
        // What the rank puts in front of the wall is kept: with `near` drawn last and `far` first,
        // paint order agrees with depth, and `near` shows over the wall, `far` nowhere.
        let near = layer(0.0, [0, 200, 0, 128], 4);
        let far = layer(2.0, BLUE, 1);
        let (_, got) = resolve(&[near, wall, far], &[FULL, FULL, FULL]);
        assert_eq!(got, blend([0, 200, 0, 128], 255, RED));
    }

    #[test]
    fn only_a_layer_covering_the_pixel_whole_hides_what_is_behind_it() {
        // An opaque stroke over half the pixel, in front at every sub-sample a translucent one
        // behind it covers, both over an opaque wall. The stroke hides the other at the sub-samples
        // but not in area, which shows beside it, as Cairo paints them: kept, it composites under
        // the stroke with its own area.
        let half = only(&[0, 2, 4, 6, 8, 10, 12, 14]);
        let area = Coverage::Area { area: 0.5, id: 0 };
        let stroke = at_depth(Plane::constant(0.0), BLUE, area, 3);
        let under = at_depth(Plane::constant(1.0), [0, 200, 0, 128], area, 2);
        let wall = at_depth(Plane::constant(2.0), RED, Coverage::Mask(u16::MAX), 1);
        assert_eq!(
            resolve(&[stroke, under, wall], &[half, half, FULL]),
            (
                Resolve::Coverage,
                blend(BLUE, 128, blend([0, 200, 0, 128], 128, RED))
            )
        );
    }

    #[test]
    fn a_pixel_whose_strong_orders_cycle_composites_each_sub_sample_in_its_own_order() {
        // As at a polyhedron's vertex (DodecahedronScene, pixel 357,191): three faces whose pairs
        // are decided over different sub-samples, `a` over `b` by depth where they share the middle
        // third, `c` over `a` by depth on the left, `b` over `c` by the majority of the right and
        // one sub-sample left of centre, so no order keeps them; behind an opaque wall and a
        // translucent layer, both over the whole pixel. No order of the pixel is taken: each
        // sub-sample composites in its own, the wall hiding the faces there.
        let u = |k: usize| SAMPLE_OFFSETS[k][0] as f64 + 0.5;
        let left: Vec<usize> = (0..SAMPLES).filter(|k| u(*k) <= -2.5).collect();
        let middle: Vec<usize> = (0..SAMPLES).filter(|k| u(*k).abs() <= 1.5).collect();
        let right: Vec<usize> = (0..SAMPLES).filter(|k| u(*k) >= 2.5).collect();
        let at = |ks: &[&[usize]]| only(&ks.concat());
        let lean = SAMPLE_OFFSETS.iter().position(|o| o[0] == -2).unwrap(); // u = -1.5
        let (ra, rb, rc) = (
            at(&[&left, &middle]),
            at(&[&middle, &right]),
            at(&[&right, &left, &[lean]]),
        );
        let face = |plane, order| at_depth(plane, RED, Coverage::Mask(0), order);
        // `c` at 10.5 + u / 2: nearer than `a` (10) left of u = -1, farther than `b` (11) right
        // of u = 1.
        let (a, b, c) = (
            face(Plane::constant(10.0), 1),
            face(Plane::constant(11.0), 2),
            face(Plane::new(6.5, 8.0, 0.0), 3),
        );
        assert_eq!(ordered(&a, &ra, &b, &rb), (Ordering::Less, Decided::Depth));
        assert_eq!(
            ordered(&b, &rb, &c, &rc),
            (Ordering::Less, Decided::Majority)
        );
        assert_eq!(ordered(&c, &rc, &a, &ra), (Ordering::Less, Decided::Depth));
        let wall = at_depth(Plane::constant(5.0), BLUE, Coverage::Mask(u16::MAX), 4);
        let glass = [0, 200, 0, 128];
        let front = at_depth(Plane::constant(0.0), glass, Coverage::Mask(u16::MAX), 5);
        assert_eq!(
            resolve(&[a, b, c, wall, front], &[ra, rb, rc, FULL, FULL]),
            (Resolve::Sample, blend(glass, 255, BLUE))
        );
    }
}
