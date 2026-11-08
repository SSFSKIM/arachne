//! The order fragments composite in at a pixel. Depth decides the order of two fragments only
//! where both cover, and only where their depths are distinguishable within what the core knows
//! of them: at a sub-sample both cover, the nearer is in front unless the two depths lie within
//! their own error bounds of each other ([`tied`]), or, for two fragments on one registered plane
//! ([`one_plane`]), within those bounds and the snap's departure of each ([`coplanar_tied`]);
//! everywhere else, at such a tie and between one path's fill and its own strokes
//! ([`own_layers`]), they keep the stream's paint order (the later drawn in front), which is the
//! order Cairo paints them in.
//!
//! A depth's error bound (a fragment's `Depths::err` at a pixel) is how far it may stand from the
//! exact depth of the core's own inputs for it (the snapped points and the normals carried). On a
//! plane it is the plane's [`crate::composite::Plane::err`], set by its constructor from its
//! inputs, and the rounding of reading the plane at the point, `4u (|a| + |b x| + |c y|)`
//! (`u = 2^-53`; it grows with the key's gradient times the pixel's coordinate, as the planes are
//! anchored at the frame's origin). The constructors count:
//!
//! - a face's plane on screen (`Projection::key_plane`): its world plane rounded to
//!   [`crate::depth::PLANE_GRID`] (each normal component and the offset within half a step), and
//!   its own arithmetic, chiefly the denominator, the plane's distance from the eye, which cancels
//!   as the plane nears the eye; each over the screen box its fragments lie in;
//! - the plane through a triangle's keyed corners (`Plane::through_with_errors`): the corners'
//!   keys' errors, and the determinant's cancellation, `u L / h` for a thin triangle;
//! - a curve's key at a point (`Centreline::key_under`, a curve's depth where no one plane gives
//!   it): its segment's ends' keys' errors, which a key between them carries at most once, and the
//!   rounding of the key along the segment; the front-most of several segments, the largest of
//!   their errors;
//! - a point's key (`Projection::key_error`): the rotation's and the perspective's rounding, which
//!   grows as the point nears the eye.
//!
//! The bound does not count the snap of the corners to the input grid: every representation of a
//! shared edge or face is made from the same snapped corners, so the snap is common input, and
//! counting it would tie geometry that is really near-coincident. Nor does it count a model's
//! departure from the geometry (a segment's plane is flat across the segment; a face's plane is
//! the face's), but for one: a face on no one plane is keyed on the plane nearest it, and its key
//! carries its drawn outline's departure from that plane (`path::nearest_plane_departure`), so it
//! ties with what that plane cannot tell it from. Two other depths that differ are two depths,
//! and depth orders them.
//!
//! But for two fragments on one registered plane: there the snap is not common input. Each is
//! keyed on the plane fitted to its own snapped points, which stand off the plane content put them
//! on by up to the snap's reach, so two faces content made coplanar part by that much, and two
//! parallel faces a few grid steps apart can part by no more. Their identity says they may be one
//! plane; their keys say whether they are within what the snap can do to one plane. Each such
//! fragment carries the snap's departure for its own key ([`Fragment::snap`]), and the two tie
//! where their depths lie within their error bounds and those departures of each other
//! ([`coplanar_tied`]); elsewhere depth orders them.
//!
//! Two fragments of flat records on no one identity whose own planes the snapped input cannot
//! tell apart in orientation ([`near_parallel`]) tie within the same term: there the input cannot
//! decide their order either, and the snap's noise, which changes from frame to frame, would.
//!
//! [`Fragment::snap`]: crate::composite::Fragment::snap
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
    pub fragment: &'a Fragment<'a>,
    pub coverage: &'a [u8; SAMPLES],
    pub depth: &'a [f64; SAMPLES],
    pub err: f64,
}

/// How a pair's order at a pixel was decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Decided {
    /// No sub-sample both cover, or one path's fill against its own stroke ([`own_layers`]):
    /// paint order.
    Paint,
    /// Every sub-sample both cover holds them at one depth within their errors, or, on one
    /// registered plane, within their errors and the snap's departures ([`coplanar_tied`]): paint
    /// order.
    Tie,
    /// Every sub-sample both cover and that is not a tie puts the same one in front.
    Depth,
    /// The sub-samples both cover disagree (they cross inside the pixel): the one in front at more
    /// of them, paint order where as many put each in front.
    Majority,
}

impl Decided {
    /// Whether depth decided the pair ([`Decided::Depth`], [`Decided::Majority`]): a strong
    /// order, which [`rank`] keeps before any weak one. Paint order (where no sub-sample is shared,
    /// depth ties, the two lie on one plane, or a path's fill meets its own stroke) is weak: a
    /// convention for a pair whose inputs say nothing of the depth between them, which yields to
    /// depth where they disagree.
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

/// Whether `a` and `b` are one path's fill and one of its own strokes (its stroke or its
/// background stroke), which keep paint order whatever their depths, as CE paints them: a stroke's
/// body runs across its fill's edge, and where the fill is keyed on a plane seen nearly edge-on and
/// the stroke on the curve, the two depths part there, though the stroke always paints over its
/// fill. Two strokes keep depth, of one path or of one leaf: a curve crossing itself in space must.
pub fn own_layers(a: &Fragment, b: &Fragment) -> bool {
    a.path_layer
        && b.path_layer
        && a.source() == b.source()
        && a.order == b.order
        && a.sub / 3 == b.sub / 3
        && a.sub != b.sub
        && (a.sub % 3 == 1 || b.sub % 3 == 1)
}

/// Whether `a` and `b` lie on one registered plane: their identities ([`Fragment::plane`]) meet,
/// whichever producer drew each, so one plane holds both within their tolerances and they may be
/// one plane. Whether they are, where both cover, their keys decide ([`coplanar_tied`]).
///
/// [`Fragment::plane`]: crate::composite::Fragment::plane
pub fn one_plane(a: &Fragment, b: &Fragment) -> bool {
    a.plane.iter().any(|p| *p != 0 && b.plane.contains(p))
}

/// Whether `a` and `b` are fragments of two flat records ([`Fragment::record`]) whose own planes
/// the snapped input cannot tell apart in orientation: their dihedral is within how far each
/// record's snap lets its own plane turn about the dihedral's axis from the plane content put its
/// points on (`depth::near_parallel`, which derives it). Pairwise and symmetric: it gives no
/// identity and does not chain. A fragment of a record with no plane of its own (a line, a curve),
/// of a pre-projected path, or of one the registry was not offered is never near-parallel. The
/// test runs once per pair of records in a frame (`depth::tested`).
///
/// [`Fragment::record`]: crate::composite::Fragment::record
pub fn near_parallel(a: &Fragment, b: &Fragment) -> bool {
    let (Some(x), Some(y)) = (a.record, b.record) else {
        return false;
    };
    let verdict = crate::depth::tested(x.0, y.0);
    count!(NEAR_PARALLEL_TESTED);
    #[cfg(feature = "counters")]
    if verdict.fresh {
        count!(NEAR_PARALLEL_RUN);
        count!(NEAR_PARALLEL_POINTS, x.0.points() + y.0.points());
    }
    #[cfg(feature = "counters")]
    if verdict.near {
        count!(NEAR_PARALLEL_PAIRS);
        if verdict.wide {
            count!(NEAR_PARALLEL_WIDE_PAIRS);
        }
    }
    verdict.near
}

/// Whether `a` (depth `da` within `ea`) and `b` (depth `db` within `eb`), at a sub-sample both
/// cover, lie on one registered plane ([`one_plane`]) or are near-parallel ([`near_parallel`]),
/// and are one depth there: within their error bounds and the snap's departure of each of each
/// other, a share ([`Fragment::snap`]) of each one's key there (with its bound: the key moves by
/// its own share of itself). Two faces content made coplanar always are; two parallel faces
/// farther apart than the snap can move one plane are not, and depth orders them, though the
/// registry gave them one identity, or found them near-parallel.
///
/// A fragment whose key nothing bounds (an infinite `snap`: a path whose points reached the core
/// already projected, so no world plane bounds its key) ties with anything on its plane, whatever
/// the depths: its key cannot tell it from that plane. A layer of a path on a registered plane (a
/// stroke, or a fill beyond its outline) is read here on its path's plane at the sub-sample, with
/// its path's share, against what shares that plane ([`crate::composite::Depths::against`]). Two
/// records with no plane of their own on flat records' identities, which may meet beyond the flat
/// records' hull, take against each other the shares their whole departures give
/// ([`Fragment::wide`]).
///
/// Two strokes of paths with no plane of their own on no one identity ([`lines`]) are one depth
/// within their error bounds and each one's own share of its key ([`lines_term`]): two lines
/// content put crossing, which the snap moves apart along each sight line by no more.
///
/// It is `one_plane(a, b) && within_term(..)`, else `lines(a, b) && lines_term(..)`, else
/// `near_parallel(a, b) && within_term(..)`: the three cases are disjoint (a flat record's
/// fragment is no stroke of a path with no plane of its own), and the near-parallel test, the only
/// one that reads more than the two fragments' own numbers, runs last: only where the two share no
/// identity, are both fragments of flat records, and their keys lie within the term.
///
/// [`Fragment::snap`]: crate::composite::Fragment::snap
/// [`Fragment::wide`]: crate::composite::Fragment::wide
pub fn coplanar_tied(a: &Fragment, da: f64, ea: f64, b: &Fragment, db: f64, eb: f64) -> bool {
    if one_plane(a, b) {
        within_term(a, da, ea, b, db, eb)
    } else if lines(a, b) {
        lines_term((da, ea, a.own), (db, eb, b.own))
    } else {
        a.record.is_some()
            && b.record.is_some()
            && within_term(a, da, ea, b, db, eb)
            && near_parallel(a, b)
    }
}

/// Whether `a` and `b` are strokes (or background strokes) of paths with no plane of their own
/// ([`Fragment::own`]), attached to identities or not, on no one identity: two lines or curves
/// whose keys along their curves the snap moves by at most each one's own share of itself.
///
/// [`Fragment::own`]: crate::composite::Fragment::own
pub fn lines(a: &Fragment, b: &Fragment) -> bool {
    a.own >= 0.0 && b.own >= 0.0 && !one_plane(a, b)
}

/// Whether two keys, each `(d, e, s)`: the key `d` within `e` and its share `s` of itself
/// ([`Fragment::own`]), lie within their error bounds and each one's share of its key (with its
/// bound) of each other: [`lines`]' term, `|da - db| <= ea + (|da| + ea) sa + eb + (|db| + eb) sb`.
///
/// [`Fragment::own`]: crate::composite::Fragment::own
pub fn lines_term((da, ea, sa): (f64, f64, f64), (db, eb, sb): (f64, f64, f64)) -> bool {
    let reach = |d: f64, e: f64, share: f64| e + (d.abs() + e) * share;
    tied(da, reach(da, ea, sa), db, reach(db, eb, sb))
}

/// Whether `a`'s key `da` (within `ea`) and `b`'s `db` (within `eb`) lie within their error bounds
/// and each one's share of its key of each other, or either key is unbounded (an infinite share):
/// [`coplanar_tied`]'s term. The share is [`Fragment::snap`], or [`Fragment::wide`] where both
/// carry one (two records with no plane of their own on flat records' identities).
///
/// [`Fragment::snap`]: crate::composite::Fragment::snap
/// [`Fragment::wide`]: crate::composite::Fragment::wide
pub(crate) fn within_term(a: &Fragment, da: f64, ea: f64, b: &Fragment, db: f64, eb: f64) -> bool {
    let reach = |d: f64, e: f64, share: f64| e + (d.abs() + e) * share;
    let share = |f: &Fragment, other: &Fragment| {
        if f.wide >= 0.0 && other.wide >= 0.0 {
            f.wide
        } else {
            f.snap
        }
    };
    let (sa, sb) = (share(a, b), share(b, a));
    !(sa.is_finite() && sb.is_finite()) || tied(da, reach(da, ea, sa), db, reach(db, eb, sb))
}

/// `a` (depth `da` within `ea`) against `b` (depth `db` within `eb`) at a sub-sample both cover:
/// [`Ordering::Less`] when `a` is in front. The nearer, unless they tie, are one depth on one
/// registered plane ([`coplanar_tied`]) or are one path's fill and its own stroke
/// ([`own_layers`]); then paint order.
pub fn at_sample(a: &Fragment, da: f64, ea: f64, b: &Fragment, db: f64, eb: f64) -> Ordering {
    ordered(a, da, b, db, weak_at(a, da, ea, b, db, eb))
}

/// Whether [`at_sample`]'s order of `a` and `b` at a sub-sample both cover is paint order, weak:
/// one path's fill and its own stroke, one depth on one registered plane, or tied. What
/// [`at_sample`] and [`strong_at`] read; the same both ways, as each of the three is.
fn weak_at(a: &Fragment, da: f64, ea: f64, b: &Fragment, db: f64, eb: f64) -> bool {
    own_layers(a, b) || coplanar_tied(a, da, ea, b, db, eb) || tied(da, ea, db, eb)
}

/// [`at_sample`]'s order of `a` (at depth `da`) against `b` (at `db`), given whether it is weak
/// ([`weak_at`]): paint order if so, else the nearer in front.
fn ordered(a: &Fragment, da: f64, b: &Fragment, db: f64, weak: bool) -> Ordering {
    if weak {
        painted(a, b)
    } else {
        da.total_cmp(&db)
    }
}

/// `items` ranked front first into `out` at one sub-sample, from [`at_sample`]'s orders
/// (`front`) and their strength (`strong`, [`strong_at`]), as [`rank`] ranks them. A fragment is
/// read at one depth against all others but one: a stroke of a path on a registered plane is read
/// on its path's plane against what shares that plane, on its curve against the rest
/// ([`crate::composite::Depths::against`]). So its strong orders at one point need not be one
/// depth's, and can cycle: a stroke `s` behind a fill `f` of its plane by the plane's key, `f`
/// behind a fill `g` of another, and `g` behind `s` by the curve's. Such a cycle is broken as
/// [`rank_through_cycles`] breaks one, the fragment in front of the most of those left first, the
/// later painted where as many: a function of the pairwise orders, their strength and paint order
/// alone, so every sub-sample of every frame ranks one way. Its [`Ranked`] says which happened.
pub fn rank_sample(
    items: &[usize],
    front: impl Fn(usize, usize) -> bool,
    strong: impl Fn(usize, usize) -> bool,
    painted: impl Fn(usize, usize) -> Ordering,
    out: &mut Vec<usize>,
) -> Ranked {
    ranked(items, front, strong, painted, out, true)
}

/// Whether [`at_sample`]'s order of `a` and `b` is strong, depth deciding it (see [`rank`]); the
/// sub-sample's counterpart of [`Decided::is_strong`]. Paint order at a tie, at one depth on one
/// registered plane ([`coplanar_tied`]) and between a path's fill and its own stroke
/// ([`own_layers`]) is weak. So at one sub-sample the strong orders are strict orders of finite
/// depths, which never cycle.
pub fn strong_at(a: &Fragment, da: f64, ea: f64, b: &Fragment, db: f64, eb: f64) -> bool {
    !weak_at(a, da, ea, b, db, eb)
}

// The masks below hold one bit per sub-sample: a sample count past their width fails the build
// rather than truncate them.
const _: () = assert!(SAMPLES <= u16::BITS as usize);

/// One way of a pair's orders at the sub-samples both cover, as masks over the sub-samples (bit
/// `k` for sub-sample `k`, clear wherever either does not cover it): where [`at_sample`] puts the
/// first in front ([`Ordering::Less`]), where behind ([`Ordering::Greater`]), and where
/// [`strong_at`] holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtSamples {
    pub front: u16,
    pub behind: u16,
    pub strong: u16,
}

/// A pair's orders at each sub-sample both cover, each read once ([`shared`]): what [`pair`]
/// decides the pixel's order from, and [`at_sample`]'s and [`strong_at`]'s orders there both
/// ways, which the pixel's sub-sample ranks read rather than evaluate again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Shared {
    /// The sub-samples both cover.
    pub covered: u16,
    /// Where the two are not one depth ([`tied`], [`coplanar_tied`]) and `a` is the nearer, and
    /// where `b` is: [`pair`]'s counts.
    pub a_nearer: u16,
    pub b_nearer: u16,
    /// `a` against `b`, and `b` against `a`.
    pub ab: AtSamples,
    pub ba: AtSamples,
}

/// The orders of `a` and `b` at each sub-sample both cover ([`Shared`]), with one evaluation of
/// their tie per sub-sample: [`pair`]'s test of `a` against `b`, which with [`own_layers`] is
/// [`weak_at`]'s of `a` against `b` and of `b` against `a`, since [`own_layers`],
/// [`coplanar_tied`] and [`tied`] are each symmetric in their two sides.
// The sub-sample `k` indexes both fragments' coverage and depth rows.
#[allow(clippy::needless_range_loop)]
pub fn shared(a: Seen, b: Seen) -> Shared {
    let mut s = Shared::default();
    let (ca, da, cb, db) = (a.coverage, a.depth, b.coverage, b.depth);
    let (fa, fb) = (a.fragment, b.fragment);
    let own = own_layers(fa, fb);
    for k in 0..SAMPLES {
        if ca[k] == 0 || cb[k] == 0 {
            continue;
        }
        let bit = 1u16 << k;
        s.covered |= bit;
        let tie =
            tied(da[k], a.err, db[k], b.err) || coplanar_tied(fa, da[k], a.err, fb, db[k], b.err);
        if !tie {
            if da[k] < db[k] {
                s.a_nearer |= bit;
            } else {
                s.b_nearer |= bit;
            }
        }
        let weak = own || tie;
        for (at, o) in [
            (&mut s.ab, ordered(fa, da[k], fb, db[k], weak)),
            (&mut s.ba, ordered(fb, db[k], fa, da[k], weak)),
        ] {
            match o {
                Ordering::Less => at.front |= bit,
                Ordering::Greater => at.behind |= bit,
                Ordering::Equal => {}
            }
            if !weak {
                at.strong |= bit;
            }
        }
    }
    s
}

/// `a` against `b` at the pixel, from their coverage and depth at each sub-sample, and how that was
/// decided: [`Ordering::Less`] when `a` is in front.
pub fn pair(a: Seen, b: Seen) -> (Ordering, Decided) {
    shared(a, b).decided(a.fragment, b.fragment)
}

impl Shared {
    /// [`pair`]'s order of `a` against `b`, the fragments these are the orders of, and how it was
    /// decided.
    pub fn decided(&self, a: &Fragment, b: &Fragment) -> (Ordering, Decided) {
        let (shared, a_front, b_front) = (
            self.covered.count_ones(),
            self.a_nearer.count_ones(),
            self.b_nearer.count_ones(),
        );
        let paint = painted(a, b);
        if shared > 0 && own_layers(a, b) {
            return (paint, Decided::Paint);
        }
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
/// holding only the front ranked before the cycle. The strong orders are depth's
/// ([`Decided::Depth`], [`Decided::Majority`]) alone; a path's fill against its own stroke
/// ([`own_layers`]) is paint order, weak, and yields to depth in a cycle.
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
        Coverage, Depths, Plane, Resolve, SAMPLE_OFFSETS, Scratch, blend, resolve_pixel,
        sample_point,
    };

    fn at_depth(
        depth: Plane,
        colour: [u8; 4],
        coverage: Coverage,
        order: u32,
    ) -> Fragment<'static> {
        Fragment::new(depth, colour, coverage, order, 0)
    }

    /// The pixel the resolve makes of `frags` over black, fragment `i` covering the sub-samples
    /// `rows[i]` gives it.
    fn resolve(frags: &[Fragment], rows: &[[u8; SAMPLES]]) -> (Resolve, [u8; 4]) {
        let mut s = Scratch::default();
        s.fragments.extend_from_slice(frags);
        resolve_pixel(
            &mut s,
            [0, 0, 0, 255],
            |i, _| rows[i],
            |_, f| Depths::of_plane(f, 0, 0),
        )
    }

    /// `a` against `b` at pixel (0, 0), as the resolve reads them there.
    fn ordered(
        a: &Fragment,
        ca: &[u8; SAMPLES],
        b: &Fragment,
        cb: &[u8; SAMPLES],
    ) -> (Ordering, Decided) {
        let (da, db) = (Depths::of_plane(a, 0, 0), Depths::of_plane(b, 0, 0));
        fn seen<'a>(f: &'a Fragment, coverage: &'a [u8; SAMPLES], d: &'a Depths) -> Seen<'a> {
            Seen {
                fragment: f,
                coverage,
                depth: &d.samples,
                err: d.err,
            }
        }
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
    fn on_one_plane_a_zero_share_parts_beyond_the_errors_and_an_unbounded_one_ties() {
        // In a frame whose points were not snapped (`grid` 0) the snap moved nothing: only the
        // core's own placing rounds them. One formula at every grid: half a step times `|n|_1`
        // at the writer's.
        let n = [0.0, 0.6, 0.8];
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let none = [crate::xf::IDENTITY, crate::xf::IDENTITY];
        let unsnapped = crate::depth::snap_departure(0.0, n, &none, &points);
        assert!((0.0..1e-14).contains(&unsnapped), "{unsnapped}");
        let g = 1.0 / 16384.0;
        let snapped = crate::depth::snap_departure(g, n, &none, &points);
        assert!((snapped - g / 2.0 * 1.4).abs() < 1e-14, "{snapped}");
        // A fill drawn first, 1e-9 nearer than one drawn later, both keyed within 1e-12, on one
        // registered plane with a zero share.
        let on_plane = |key: f64, order: u32| {
            let depth = Plane {
                err: 1e-12,
                ..Plane::constant(key)
            };
            let mut f = at_depth(depth, RED, Coverage::Mask(u16::MAX), order);
            (f.plane, f.snap) = ([7, 0, 0, 0], 0.0);
            f
        };
        let first = on_plane(-1.0, 1);
        let mut later = on_plane(-1.0 + 1e-9, 2);
        assert_eq!(
            ordered(&first, &FULL, &later, &FULL),
            (Ordering::Less, Decided::Depth)
        );
        // The later one unbounded (a path the camera placed itself): paint order.
        later.snap = f64::INFINITY;
        assert_eq!(
            ordered(&first, &FULL, &later, &FULL),
            (Ordering::Greater, Decided::Tie)
        );
    }

    /// A line on a registered plane drawn first, 1e-10 of the key nearer than a degenerate triangle
    /// and a flat face drawn after it on that plane at one sub-sample: the line, with no plane of
    /// its own, may stand 1e-9 of its key off the plane wherever it lies but 1e-11 where it meets
    /// the flat records on it; the triangle, with none either, 1e-12 both ways. Against the
    /// triangle, which may meet it beyond the flat records' hull, the line takes its whole share
    /// and they tie in paint order; against the face, the share within the hull, and depth orders
    /// them.
    #[test]
    fn a_record_with_no_plane_of_its_own_takes_its_whole_share_only_against_another() {
        let on_plane = |key: f64, order: u32, snap: f64, wide: f64| {
            let depth = Plane {
                err: 1e-15,
                ..Plane::constant(key)
            };
            let mut f = at_depth(depth, RED, Coverage::Mask(u16::MAX), order);
            (f.plane, f.snap, f.wide) = ([7, 0, 0, 0], snap, wide);
            f
        };
        let line = on_plane(-1.0 - 1e-10, 1, 1e-11, 1e-9);
        let triangle = on_plane(-1.0, 2, 1e-12, 1e-12);
        let face = on_plane(-1.0, 2, 1e-12, f64::NEG_INFINITY);
        assert_eq!(
            ordered(&line, &FULL, &triangle, &FULL),
            (Ordering::Greater, Decided::Tie)
        );
        assert_eq!(
            ordered(&line, &FULL, &face, &FULL),
            (Ordering::Less, Decided::Depth)
        );
    }

    /// A stroke `s` read on its path's plane against a fill `f1` of that plane, and on its curve
    /// against a fill `g` of another, at one sub-sample: by the plane's key `f1` is in front of `s`,
    /// `g` in front of `f1`, and by the curve's `s` in front of `g`, each beyond their bounds. The
    /// strong orders cycle at one point; the sub-sample's rank settles it as `rank_through_cycles`
    /// does (each in front of one: the later painted first, `s`), then keeps every order it can
    /// (`g` before `f1`): one order for every sub-sample, whatever order the fragments come in.
    #[test]
    fn a_stroke_read_two_ways_settles_a_cycle_at_one_sub_sample_by_wins_then_paint_order() {
        let fill = |key: f64, order: u32, plane: u32, colour: [u8; 4]| {
            let mut f = at_depth(
                Plane::constant(key),
                colour,
                Coverage::Mask(u16::MAX),
                order,
            );
            (f.plane, f.snap, f.path_layer, f.sub) = ([plane, 0, 0, 0], 0.0, true, 1);
            f
        };
        let f1 = fill(-1.0, 1, 7, RED);
        let g = fill(-1.2, 2, 9, RED);
        let mut s = fill(-1.4, 3, 7, BLUE);
        s.sub = 2;
        let own = |f: &Fragment| Depths::of_plane(f, 0, 0);
        let mut on = own(&s);
        on.on_plane = Some(([-0.8; SAMPLES], 0.0));
        let frags = [f1, g, s];
        let depths = [own(&f1), own(&g), on];
        let read = |a: usize, b: usize| depths[a].against(&frags[a], &frags[b]);
        let front = |a: usize, b: usize| {
            let ((da, ea), (db, eb)) = (read(a, b), read(b, a));
            a != b && at_sample(&frags[a], da[0], ea, &frags[b], db[0], eb).is_lt()
        };
        let strong = |a: usize, b: usize| {
            let ((da, ea), (db, eb)) = (read(a, b), read(b, a));
            strong_at(&frags[a], da[0], ea, &frags[b], db[0], eb)
        };
        assert!(front(0, 2) && front(1, 0) && front(2, 1), "the cycle");
        assert!(strong(0, 2) && strong(1, 0) && strong(2, 1));
        let painted = |a: usize, b: usize| painted(&frags[a], &frags[b]);
        let mut out = Vec::new();
        for items in [[0, 1, 2], [2, 1, 0], [1, 2, 0]] {
            let how = rank_sample(&items, front, strong, painted, &mut out);
            assert_eq!(
                (how, &out[..]),
                (Ranked::Strong, &[2, 1, 0][..]),
                "{items:?}"
            );
        }
        // The resolve composites it so: the stroke in front over the whole pixel.
        let mut scratch = crate::composite::Scratch::default();
        scratch.fragments.extend_from_slice(&frags);
        let depth = |i: usize, _: &Fragment| depths[i].clone();
        let (_, pixel) = resolve_pixel(&mut scratch, [0, 0, 0, 255], |_, _| FULL, depth);
        assert_eq!(pixel, BLUE);
    }

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
            let bound =
                Depths::of_plane(&cap, 0, 0).err + Depths::of_plane(&outline(0.0), 0, 0).err;
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

    #[test]
    fn a_path_s_fill_and_its_own_strokes_keep_paint_order_and_two_strokes_keep_depth() {
        // The layers of one path (record drawn 3rd, its path 0: sub-orders 0, 1, 2) and the stroke
        // of the record's path 1 (sub-order 5), each at its own depth, nearer being smaller.
        let layer = |depth: f64, colour: [u8; 4], sub: u32| {
            let mut f = Fragment::new(
                Plane::constant(depth),
                colour,
                Coverage::Mask(u16::MAX),
                3,
                sub,
            );
            f.path_layer = true;
            f
        };
        let green = [0, 200, 0, 255];
        let background = layer(-1.2, green, 0);
        let fill = layer(-1.0, RED, 1);
        let stroke = layer(-0.9, BLUE, 2);
        let other_stroke = layer(-1.1, green, 5);
        // The stroke over its fill, the fill over its background stroke, though each is the
        // farther: paint order, as CE paints them.
        assert_eq!(resolve(&[fill, stroke], &[FULL, FULL]).1, BLUE);
        assert_eq!(resolve(&[stroke, fill], &[FULL, FULL]).1, BLUE);
        assert_eq!(resolve(&[background, fill], &[FULL, FULL]).1, RED);
        assert_eq!(
            ordered(&fill, &FULL, &stroke, &FULL),
            (Ordering::Greater, Decided::Paint)
        );
        // Two strokes, of one path or of two of the record's paths, and a fill against another
        // path's stroke: depth.
        assert_eq!(resolve(&[background, stroke], &[FULL, FULL]).1, green);
        assert_eq!(resolve(&[stroke, other_stroke], &[FULL, FULL]).1, green);
        assert_eq!(resolve(&[fill, other_stroke], &[FULL, FULL]).1, green);
        // A mesh's triangles at those sub-orders are no path's layers: depth.
        let triangle = |depth: f64, colour: [u8; 4], sub: u32| {
            Fragment::new(
                Plane::constant(depth),
                colour,
                Coverage::Mask(u16::MAX),
                3,
                sub,
            )
        };
        assert_eq!(
            resolve(
                &[triangle(-1.0, RED, 1), triangle(-0.9, BLUE, 2)],
                &[FULL, FULL]
            )
            .1,
            RED
        );
    }

    #[test]
    fn inside_its_fill_a_stroke_is_at_least_as_near_as_its_fill() {
        // A path's fill, and its stroke drawn after it whose own depth (the wire's) is behind `x`
        // while the fill is in front of `x`. At every sub-sample the fill covers, the stroke takes
        // the nearer of its own depth and the fill's: one depth with the fill, over it by paint
        // order (weak), and in front of `x` by depth. So the stroke, the fill, `x`, front to back,
        // whatever order they arrive in.
        let layer = |depth: f64, colour: [u8; 4], sub: u32| {
            let mut f = Fragment::new(
                Plane::constant(depth),
                colour,
                Coverage::Mask(u16::MAX),
                3,
                sub,
            );
            f.path_layer = true;
            f
        };
        let (fill_c, x_c, stroke_c) = ([100, 0, 0, 128], [0, 100, 0, 128], [0, 0, 100, 128]);
        let fill = layer(-1.0, fill_c, 1);
        let stroke = layer(-0.8, stroke_c, 2);
        let x = at_depth(Plane::constant(-0.9), x_c, Coverage::Mask(u16::MAX), 2);
        let want = blend(
            stroke_c,
            255,
            blend(fill_c, 255, blend(x_c, 255, [0, 0, 0, 255])),
        );
        for frags in [[fill, stroke, x], [x, stroke, fill], [stroke, x, fill]] {
            assert_eq!(resolve(&frags, &[FULL; 3]).1, want, "{frags:?}");
        }
    }

    #[test]
    fn outside_its_fill_a_stroke_keeps_the_wire_s_depth() {
        // The stroke of `inside_its_fill_a_stroke_is_at_least_as_near_as_its_fill` where its fill
        // does not reach (its outer half): the fill covers half the sub-samples, the stroke and `x`
        // the other half. There the stroke keeps its own depth, behind `x`: `x`, then the stroke;
        // its depth and bound are its own at every sub-sample.
        let layer = |depth: f64, sub: u32| {
            let mut f = Fragment::new(
                Plane::constant(depth),
                RED,
                Coverage::Mask(u16::MAX),
                3,
                sub,
            );
            f.path_layer = true;
            f
        };
        let (fill, stroke) = (layer(-1.0, 1), layer(-0.8, 2));
        let x = at_depth(Plane::constant(-0.9), BLUE, Coverage::Mask(u16::MAX), 2);
        let frags = [fill, stroke, x];
        let (half, rest): (Vec<usize>, Vec<usize>) = (0..SAMPLES).partition(|k| k % 2 == 0);
        let rows = [only(&half), only(&rest), only(&rest)];
        let own: Vec<Depths> = frags.iter().map(|f| Depths::of_plane(f, 0, 0)).collect();
        let mut depths = own.clone();
        crate::composite::strokes_on_own_fills(&frags, &mut depths, |j| rows[j]);
        assert_eq!(depths, own);
        for &k in &rest {
            let (s, x_at) = (&depths[1], &depths[2]);
            assert_eq!(
                at_sample(&stroke, s.samples[k], s.err, &x, x_at.samples[k], x_at.err),
                Ordering::Greater
            );
        }
        // Where the fill covers too, the stroke takes the fill's depth and bound there.
        let rows = [FULL, FULL, FULL];
        crate::composite::strokes_on_own_fills(&frags, &mut depths, |j| rows[j]);
        assert_eq!(depths[1].samples, own[0].samples);
        assert_eq!(depths[1].err, own[0].err);
    }

    #[test]
    fn a_stroke_nearer_than_its_fill_within_their_bounds_is_not_behind_what_the_fill_is_not() {
        // The stroke's own depth nearer than its fill's, but within their bounds of it, the fill's
        // the wider (a plane seen nearly edge-on); `x` nearer than both, beyond the stroke's own
        // bound but within the fill's. Which of the two depths is the nearer is not known, so the
        // stroke on its fill carries both bounds: it ties with `x` as the fill does, and paint
        // order puts it over its fill, as CE draws it.
        let layer = |depth: f64, err: f64, colour: [u8; 4], sub: u32| {
            let plane = Plane {
                err,
                ..Plane::constant(depth)
            };
            let mut f = Fragment::new(plane, colour, Coverage::Mask(u16::MAX), 3, sub);
            f.path_layer = true;
            f
        };
        let (fill_c, x_c, stroke_c) = ([100, 0, 0, 128], [0, 100, 0, 128], [0, 0, 100, 128]);
        let fill = layer(0.52, 0.1, fill_c, 1);
        let stroke = layer(0.5, 0.01, stroke_c, 2);
        let x = at_depth(
            Plane {
                err: 0.01,
                ..Plane::constant(0.45)
            },
            x_c,
            Coverage::Mask(u16::MAX),
            2,
        );
        let want = blend(
            stroke_c,
            255,
            blend(fill_c, 255, blend(x_c, 255, [0, 0, 0, 255])),
        );
        for frags in [[fill, stroke, x], [x, stroke, fill], [stroke, x, fill]] {
            assert_eq!(resolve(&frags, &[FULL; 3]).1, want, "{frags:?}");
        }
    }

    #[test]
    fn at_one_sub_sample_the_strong_orders_never_cycle() {
        // Fragments of a few paths' layers and of meshes, on registered planes or none, at depths
        // that tie, part within their errors, or part beyond them, ranked as one sub-sample ranks
        // them: depth's strict orders alone are strong, so they never cycle.
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let mut out = Vec::new();
        for _ in 0..20_000 {
            let n = 3 + next(4) as usize;
            let frags: Vec<(Fragment, f64, f64)> = (0..n)
                .map(|_| {
                    let mut f = Fragment::new(
                        Plane::constant(0.0),
                        RED,
                        Coverage::Mask(u16::MAX),
                        next(2) as u32,
                        next(6) as u32,
                    );
                    f.path_layer = next(3) != 0;
                    f.plane = [next(3) as u32, 0, 0, 0];
                    (f, next(8) as f64 * 0.01, next(3) as f64 * 0.01)
                })
                .collect();
            let items: Vec<usize> = (0..n).collect();
            let at = |a: usize, b: usize| {
                let ((fa, da, ea), (fb, db, eb)) = (&frags[a], &frags[b]);
                (fa, *da, *ea, fb, *db, *eb)
            };
            let how = rank(
                &items,
                |a, b| {
                    let (fa, da, ea, fb, db, eb) = at(a, b);
                    a != b && at_sample(fa, da, ea, fb, db, eb) == Ordering::Less
                },
                |a, b| {
                    let (fa, da, ea, fb, db, eb) = at(a, b);
                    strong_at(fa, da, ea, fb, db, eb)
                },
                |a, b| painted(&frags[a].0, &frags[b].0),
                &mut out,
            );
            assert_ne!(how, Ranked::Strong, "{frags:?}");
        }
    }
}
