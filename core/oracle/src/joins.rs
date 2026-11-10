//! `join_mask@1` (contract §8): Cairo's join decisions, carried on the wire as XF 1.4's `joins`,
//! applied to kurbo's stroke of a path.
//!
//! Three readings, each from the contract's text: the decisions the record carries, read by
//! walking the outline's cubics ([`carried`]); the joins kurbo's stroker makes, found by walking
//! the path handed to it as the stroker walks it ([`made`]); and the edits where a matched
//! decision differs from kurbo's own ([`edit`]). kurbo is the shared stroker: what this module
//! knows of it is what the contract states, the stroker's walk over the elements it is handed and
//! the vertices it writes at a join, and a test holds that statement to kurbo's own output.

use std::collections::HashMap;

use kurbo::{BezPath, PathEl, Point, Vec2};

use crate::flatten::{MITER_LIMIT, closes, subpaths};

/// A tangent of squared length at most this is too short to use: kurbo's stroker takes the next
/// control point instead.
const SHORT_TANGENT: f64 = 1e-12;

/// A join's key: the bit patterns of its two elements' eight points (`x` then `y` of each, the four
/// points of the element it leaves and then the four of the one it enters) and whether it is a
/// subpath's closing join.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JoinKey {
    bits: [u64; 16],
    closing: bool,
}

impl JoinKey {
    fn new(from: &[Point; 4], into: &[Point; 4], closing: bool) -> JoinKey {
        let mut bits = [0u64; 16];
        for (i, p) in from.iter().chain(into).enumerate() {
            bits[2 * i] = p.x.to_bits();
            bits[2 * i + 1] = p.y.to_bits();
        }
        JoinKey { bits, closing }
    }

    pub fn closing(&self) -> bool {
        self.closing
    }
}

/// What one occurrence of a join carries: `Some(true)` where Cairo mitres, `Some(false)` where it
/// bevels, `None` where the record decides nothing.
pub type Decision = Option<bool>;

/// The carried decisions: per key, one per occurrence, in the order the outline's walk meets them.
pub type Carried = HashMap<JoinKey, Vec<Decision>>;

/// One `joins` byte read: bit 0 clear, or no byte, decides nothing; else bit 1 says miter.
fn decision(byte: Option<u8>) -> Decision {
    let b = byte?;
    (b & 1 != 0).then_some(b & 2 != 0)
}

fn pt(p: [f64; 3]) -> Point {
    Point::new(p[0], p[1])
}

/// The decisions the record's `joins` carry for the outline of `shown` (§8, "The carried
/// decisions"). Per subpath of §4's outline, its cubics walked in order from `p0`, the end of the
/// last cubic kept: a cubic whose other three points all equal `p0` is skipped; each cubic kept
/// after the first makes the join into it, decided by its own byte; a closed subpath then makes
/// its closing join, from its last cubic kept into its first, decided by the byte of its first
/// cubic in the points.
pub fn carried(shown: &[[f64; 3]], joins: &[u8]) -> Carried {
    let mut out = Carried::new();
    let mut occur = |key: JoinKey, d: Decision| out.entry(key).or_default().push(d);
    for (s, e) in subpaths(shown) {
        let closed = closes(shown, s, e);
        let (first_cubic, last_cubic) = (s / 4, e / 4 - 1);
        let mut p0 = pt(shown[s]);
        let mut first: Option<[Point; 4]> = None;
        let mut last: Option<[Point; 4]> = None;
        for c in first_cubic..=last_cubic {
            let i = 4 * c;
            let c3 = if closed && c == last_cubic {
                pt(shown[s])
            } else {
                pt(shown[i + 3])
            };
            let cubic = [p0, pt(shown[i + 1]), pt(shown[i + 2]), c3];
            if cubic[1..].iter().all(|q| *q == p0) {
                continue;
            }
            match last {
                Some(prev) => occur(
                    JoinKey::new(&prev, &cubic, false),
                    decision(joins.get(c).copied()),
                ),
                None => first = Some(cubic),
            }
            last = Some(cubic);
            p0 = c3;
        }
        if let (true, Some(first), Some(last)) = (closed, first, last) {
            occur(
                JoinKey::new(&last, &first, true),
                decision(joins.get(first_cubic).copied()),
            );
        }
    }
    out
}

/// One join kurbo's stroker makes: at `p0`, from the arriving tangent `ab` into the leaving `cd`.
#[derive(Clone, Debug, PartialEq)]
pub struct Made {
    pub p0: Point,
    pub ab: Vec2,
    pub cd: Vec2,
    /// Whether kurbo mitres it (under its miter style).
    pub mitres: bool,
    /// Its key, where both its elements carry one.
    pub key: Option<JoinKey>,
    /// Which stroked subpath it is in, counting only subpaths kurbo strokes.
    pub subpath: usize,
}

/// The joins kurbo's stroker makes on `path` and, per stroked subpath, how many contours kurbo's
/// outline holds for it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Walk {
    pub joins: Vec<Made>,
    pub contours: Vec<usize>,
}

/// A cubic's tangents at its ends, with kurbo's fallbacks for a short one.
fn cubic_tangents(p: [Point; 4]) -> (Vec2, Vec2) {
    let long = |v: Vec2| v.hypot2() > SHORT_TANGENT;
    let start = [p[1] - p[0], p[2] - p[0]]
        .into_iter()
        .find(|v| long(*v))
        .unwrap_or(p[3] - p[0]);
    let end = [p[3] - p[2], p[3] - p[1]]
        .into_iter()
        .find(|v| long(*v))
        .unwrap_or(p[3] - p[0]);
    (start, end)
}

/// A quadratic's tangents at its ends, with kurbo's fallback for a short one.
fn quad_tangents(p0: Point, c1: Point, c2: Point) -> (Vec2, Vec2) {
    let long = |v: Vec2| v.hypot2() > SHORT_TANGENT;
    let start = Some(c1 - p0).filter(|v| long(*v)).unwrap_or(c2 - p0);
    let end = Some(c2 - c1).filter(|v| long(*v)).unwrap_or(c2 - p0);
    (start, end)
}

/// The stroker's state over one subpath of the walk.
struct Stroker {
    join_thresh: f64,
    limit2: f64,
    out: Walk,
    /// Whether the current subpath has begun (kurbo's forward path is not empty).
    begun: bool,
    start_pt: Point,
    last_pt: Point,
    start_tan: Vec2,
    last_tan: Vec2,
    first_key: Option<[Point; 4]>,
    last_key: Option<[Point; 4]>,
}

impl Stroker {
    /// An element arriving with start tangent `tan` and key points `key`: the subpath's first, or
    /// the join into it from the element before it.
    fn arrive(&mut self, tan: Vec2, key: Option<[Point; 4]>, closing: bool) {
        if !self.begun {
            self.begun = true;
            self.start_tan = tan;
            self.first_key = key;
            self.out.contours.push(0);
            return;
        }
        let (ab, cd) = (self.last_tan, tan);
        let cross = ab.cross(cd);
        let dot = ab.dot(cd);
        let h = libm::hypot(cross, dot);
        if dot <= 0.0 || cross.abs() >= h * self.join_thresh {
            self.out.joins.push(Made {
                p0: self.last_pt,
                ab,
                cd,
                mitres: 2.0 * h < (h + dot) * self.limit2,
                key: match (self.last_key, key) {
                    (Some(a), Some(b)) => Some(JoinKey::new(&a, &b, closing)),
                    _ => None,
                },
                subpath: self.out.contours.len() - 1,
            });
        }
    }

    /// The element done: it ends at `end` with tangent `tan`.
    fn depart(&mut self, end: Point, tan: Vec2, key: Option<[Point; 4]>) {
        self.last_pt = end;
        self.last_tan = tan;
        self.last_key = key;
    }

    /// The subpath ends, open (one contour) or closed (two).
    fn finish(&mut self, contours: usize) {
        if self.begun {
            *self.out.contours.last_mut().expect("a begun subpath") = contours;
            self.begun = false;
        }
    }
}

/// The joins kurbo's `stroke` makes walking `path` at `tolerance` with a stroke `width` wide
/// (§8, "The joins kurbo makes"): `join_thresh = 2 · tolerance / width`, `limit² = pow(10, 2)`.
pub fn made(path: &[PathEl], tolerance: f64, width: f64) -> Walk {
    let mut k = Stroker {
        join_thresh: 2.0 * tolerance / width,
        limit2: libm::pow(MITER_LIMIT, 2.0),
        out: Walk::default(),
        begun: false,
        start_pt: Point::ZERO,
        last_pt: Point::ZERO,
        start_tan: Vec2::ZERO,
        last_tan: Vec2::ZERO,
        first_key: None,
        last_key: None,
    };
    for el in path {
        let p0 = k.last_pt;
        match *el {
            PathEl::MoveTo(p) => {
                k.finish(1);
                k.start_pt = p;
                k.last_pt = p;
            }
            PathEl::LineTo(p1) => {
                if p1 != p0 {
                    let tan = p1 - p0;
                    let key = Some([p0, p0, p0, p1]);
                    k.arrive(tan, key, false);
                    k.depart(p1, tan, key);
                }
            }
            PathEl::QuadTo(c1, c2) => {
                if c1 != p0 || c2 != p0 {
                    let (t0, t1) = quad_tangents(p0, c1, c2);
                    k.arrive(t0, None, false);
                    k.depart(c2, t1, None);
                }
            }
            PathEl::CurveTo(c1, c2, c3) => {
                if c1 != p0 || c2 != p0 || c3 != p0 {
                    let pts = [p0, c1, c2, c3];
                    let (t0, t1) = cubic_tangents(pts);
                    k.arrive(t0, Some(pts), false);
                    k.depart(c3, t1, Some(pts));
                }
            }
            PathEl::ClosePath => {
                if p0 != k.start_pt {
                    let tan = k.start_pt - p0;
                    k.arrive(tan, None, false);
                    k.depart(k.start_pt, tan, None);
                }
                if k.begun {
                    let (tan, key) = (k.start_tan, k.first_key);
                    k.arrive(tan, key, true);
                }
                k.finish(2);
            }
        }
    }
    k.finish(1);
    k.out
}

/// The decision each join kurbo makes takes (§8, "Matching"): in order, a join with a key takes
/// the next unused decision of its key's list. A join with no key, no list, or a list used up
/// takes none.
pub fn matched(joins: &[Made], carried: &Carried) -> Vec<Decision> {
    let mut used: HashMap<JoinKey, usize> = HashMap::new();
    joins
        .iter()
        .map(|j| {
            let key = j.key?;
            let list = carried.get(&key)?;
            let next = used.entry(key).or_insert(0);
            let d = *list.get(*next)?;
            *next += 1;
            d
        })
        .collect()
}

/// kurbo's offset of the tangent `t` at half the stroke's width.
fn norm(width: f64, t: Vec2) -> Vec2 {
    (0.5 * width / t.hypot()) * Vec2::new(-t.y, t.x)
}

/// The one element of `els` that is a line to `to`.
fn one_line_to(els: &[PathEl], to: Point) -> Option<usize> {
    one(els, |i| els[i] == PathEl::LineTo(to))
}

/// The one element of `els` that is a line whose element before it ends at `after`.
fn one_line_after(els: &[PathEl], after: Point) -> Option<usize> {
    one(els, |i| {
        matches!(els[i], PathEl::LineTo(_)) && i > 0 && els[i - 1].end_point() == Some(after)
    })
}

fn one(els: &[PathEl], hit: impl Fn(usize) -> bool) -> Option<usize> {
    let mut found = (0..els.len()).filter(|&i| hit(i));
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// Edits one subpath's contours `els` at the join `j`, whose carried decision `mitre` differs from
/// kurbo's own. A join on parallel tangents, or whose lines are not each found exactly once, is
/// left as kurbo drew it.
fn edit_one(els: &mut Vec<PathEl>, j: &Made, mitre: bool, width: f64) {
    let c = j.ab.cross(j.cd);
    if c == 0.0 {
        return;
    }
    let p0 = j.p0;
    let (n_a, n) = (norm(width, j.ab), norm(width, j.cd));
    let s = if c > 0.0 { -1.0 } else { 1.0 };
    let (this, last) = (p0 + s * n, p0 + s * n_a);
    let m = this - j.cd * (j.ab.cross(this - last) / c);
    if !mitre {
        // A carried bevel where kurbo mitred: drop its line to the miter point and its line back
        // to the join's point.
        let (Some(a), Some(b)) = (one_line_to(els, m), one_line_to(els, p0)) else {
            return;
        };
        els.remove(a.max(b));
        els.remove(a.min(b));
        return;
    }
    // A carried miter where kurbo bevelled: a tip on the bisector at the limit's length, before
    // the outer side's bevel line, and the join's point before the inner side's.
    let tip = p0 + (m - p0) * (MITER_LIMIT * 0.5 * width / (m - p0).hypot());
    let (Some(forward), Some(backward)) = (one_line_to(els, p0 - n), one_line_after(els, p0 + n))
    else {
        return;
    };
    let (outer, inner) = if c > 0.0 {
        (forward, backward)
    } else {
        (backward, forward)
    };
    let mut at = [(outer, tip), (inner, p0)];
    at.sort_by_key(|(place, _)| std::cmp::Reverse(*place));
    for (place, p) in at {
        els.insert(place, PathEl::LineTo(p));
    }
}

/// `stroked`, kurbo's stroke of `path` with a stroke `width` wide under its miter style at
/// `tolerance`, edited at every join whose matched carried decision differs from kurbo's
/// (§8, "The edits"). Each edit is searched for within its subpath's contours alone. Where the
/// stroke's contours are not the ones the walk accounts for, nothing is edited.
pub fn edit(
    stroked: BezPath,
    path: &[PathEl],
    tolerance: f64,
    width: f64,
    carried: &Carried,
) -> BezPath {
    let walk = made(path, tolerance, width);
    let mut subpaths: Vec<Vec<PathEl>> = Vec::new();
    let mut counts = walk.contours.iter().copied();
    let mut left = 0usize;
    for el in stroked.elements() {
        if let PathEl::MoveTo(_) = el {
            if left == 0 {
                let Some(n) = counts.next() else {
                    return stroked;
                };
                left = n;
                subpaths.push(Vec::new());
            }
            left -= 1;
        }
        let Some(current) = subpaths.last_mut() else {
            return stroked;
        };
        current.push(*el);
    }
    if left != 0 || counts.next().is_some() {
        return stroked;
    }
    for (j, d) in walk.joins.iter().zip(matched(&walk.joins, carried)) {
        if let Some(mitre) = d
            && mitre != j.mitres
        {
            edit_one(&mut subpaths[j.subpath], j, mitre, width);
        }
    }
    BezPath::from_vec(subpaths.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubic_line(a: [f64; 2], b: [f64; 2]) -> [[f64; 3]; 4] {
        let at = |k: f64| {
            [
                (a[0] * (3.0 - k) + b[0] * k) / 3.0,
                (a[1] * (3.0 - k) + b[1] * k) / 3.0,
                0.0,
            ]
        };
        [at(0.0), at(1.0), at(2.0), at(3.0)]
    }

    fn polyline(points: &[[f64; 2]]) -> Vec<[f64; 3]> {
        points
            .windows(2)
            .flat_map(|w| cubic_line(w[0], w[1]))
            .collect()
    }

    fn outline(shown: &[[f64; 3]]) -> BezPath {
        crate::flatten::outline(shown)
    }

    fn stroke(path: &BezPath, width: f64) -> BezPath {
        let style = kurbo::Stroke::new(width)
            .with_join(kurbo::Join::Miter)
            .with_caps(kurbo::Cap::Butt)
            .with_miter_limit(MITER_LIMIT);
        kurbo::stroke(
            path.elements().iter().copied(),
            &style,
            &kurbo::StrokeOpts::default(),
            0.1,
        )
    }

    const SQUARE: [[f64; 2]; 5] = [
        [-4.0, -4.0],
        [4.0, -4.0],
        [4.0, 4.0],
        [-4.0, 4.0],
        [-4.0, -4.0],
    ];

    #[test]
    fn a_closed_subpaths_closing_join_takes_its_first_cubics_byte() {
        let shown = polyline(&SQUARE);
        let c = carried(&shown, &[0b01, 0b11, 0b11, 0b11]);
        assert_eq!(c.len(), 4);
        let closing: Vec<_> = c.iter().filter(|(k, _)| k.closing()).collect();
        assert_eq!(closing.len(), 1);
        assert_eq!(closing[0].1, &vec![Some(false)]);
        assert!(
            c.iter()
                .filter(|(k, _)| !k.closing())
                .all(|(_, d)| d == &vec![Some(true)])
        );
    }

    #[test]
    fn a_skipped_first_cubic_still_decides_the_closing_join() {
        // A first cubic all at the subpath's start is skipped: the closing join runs from the last
        // cubic into the second, and still reads the first cubic's byte.
        let mut shown = vec![[-4.0, -4.0, 0.0]; 4];
        shown.extend(polyline(&SQUARE));
        let c = carried(&shown, &[0b11, 0b01, 0b01, 0b01, 0b01]);
        let closing: Vec<_> = c.iter().filter(|(k, _)| k.closing()).collect();
        assert_eq!(closing.len(), 1);
        assert_eq!(closing[0].1, &vec![Some(true)]);
        // Three ordinary joins (into the third, fourth and fifth cubics): no join into the second,
        // which is the first kept.
        assert_eq!(c.values().filter(|d| *d == &vec![Some(false)]).count(), 3);
    }

    #[test]
    fn one_keys_occurrences_are_listed_in_walk_order_undecided_ones_kept() {
        let (p, q, r) = ([-4.0, -3.0], [4.0, -3.0], [0.0, 3.0]);
        let shown = polyline(&[p, q, r, p, q, r]);
        let c = carried(&shown, &[0, 0b01, 0, 0, 0b11]);
        let lists: Vec<_> = c.values().cloned().collect();
        assert_eq!(c.len(), 3, "{lists:?}");
        assert!(lists.contains(&vec![Some(false), Some(true)]));
        assert_eq!(lists.iter().filter(|l| *l == &vec![None]).count(), 2);
    }

    #[test]
    fn a_byte_with_bit_0_clear_or_missing_decides_nothing() {
        let shown = polyline(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 8.0]]);
        let c = carried(&shown, &[0b11, 0b10]);
        let mut lists: Vec<_> = c.values().cloned().collect();
        lists.sort();
        assert_eq!(lists, vec![vec![None], vec![None]]);
    }

    #[test]
    fn the_walk_finds_the_joins_kurbo_makes_and_its_contours() {
        let path = outline(&polyline(&SQUARE));
        let w = made(path.elements(), 0.1, 1.0);
        assert_eq!(w.contours, vec![2]);
        assert_eq!(w.joins.len(), 4);
        assert!(w.joins.iter().all(|j| j.mitres && j.key.is_some()));
        assert_eq!(
            w.joins.iter().filter(|j| j.key.unwrap().closing()).count(),
            1
        );
        assert_eq!(w.joins[3].p0, Point::new(-4.0, -4.0));
        // kurbo's outline holds one move per contour.
        let moves = stroke(&path, 1.0)
            .elements()
            .iter()
            .filter(|e| matches!(e, PathEl::MoveTo(_)))
            .count();
        assert_eq!(moves, 2);
    }

    #[test]
    fn a_slight_turn_makes_no_join_and_a_reversal_always_does() {
        // join_thresh = 2 · 0.1 / 1 = 0.2: a turn of sin under 0.2 is suppressed.
        let slight = BezPath::from_vec(vec![
            PathEl::MoveTo(Point::new(0.0, 0.0)),
            PathEl::LineTo(Point::new(10.0, 0.0)),
            PathEl::LineTo(Point::new(20.0, 1.0)),
        ]);
        assert!(made(slight.elements(), 0.1, 1.0).joins.is_empty());
        let sharp = BezPath::from_vec(vec![
            PathEl::MoveTo(Point::new(0.0, 0.0)),
            PathEl::LineTo(Point::new(10.0, 0.0)),
            PathEl::LineTo(Point::new(20.0, 3.0)),
        ]);
        assert_eq!(made(sharp.elements(), 0.1, 1.0).joins.len(), 1);
        let back = BezPath::from_vec(vec![
            PathEl::MoveTo(Point::new(0.0, 0.0)),
            PathEl::LineTo(Point::new(10.0, 0.0)),
            PathEl::LineTo(Point::new(5.0, 0.0)),
        ]);
        let w = made(back.elements(), 0.1, 1.0);
        assert_eq!(w.joins.len(), 1);
        assert!(!w.joins[0].mitres, "a reversal never mitres");
    }

    #[test]
    fn a_short_tangent_falls_back_to_the_next_control_point() {
        let p0 = Point::new(0.0, 0.0);
        let (t0, t1) = cubic_tangents([p0, p0, Point::new(1.0, 1.0), Point::new(2.0, 0.0)]);
        assert_eq!(t0, Vec2::new(1.0, 1.0));
        assert_eq!(t1, Vec2::new(1.0, -1.0));
        let q = Point::new(2.0, 0.0);
        let (t0, t1) = cubic_tangents([p0, p0, q, q]);
        assert_eq!((t0, t1), (Vec2::new(2.0, 0.0), Vec2::new(2.0, 0.0)));
        let tiny = Point::new(1e-7, 0.0);
        let (t0, _) = cubic_tangents([p0, tiny, Point::new(1.0, 1.0), q]);
        assert_eq!(
            t0,
            Vec2::new(1.0, 1.0),
            "a tangent of squared length 1e-14 is too short"
        );
    }

    #[test]
    fn a_closing_line_carries_no_key() {
        // An open-ended closed subpath: kurbo strokes a closing line, whose joins have no key.
        let path = BezPath::from_vec(vec![
            PathEl::MoveTo(Point::new(0.0, 0.0)),
            PathEl::CurveTo(
                Point::new(3.0, 0.0),
                Point::new(6.0, 0.0),
                Point::new(9.0, 0.0),
            ),
            PathEl::CurveTo(
                Point::new(9.0, 3.0),
                Point::new(9.0, 6.0),
                Point::new(9.0, 9.0),
            ),
            PathEl::ClosePath,
        ]);
        let w = made(path.elements(), 0.1, 1.0);
        assert_eq!(w.joins.len(), 3);
        assert!(w.joins[0].key.is_some());
        assert!(w.joins[1].key.is_none() && w.joins[2].key.is_none());
    }

    /// The walk's model of kurbo against kurbo itself: at every join it says kurbo mitres, kurbo's
    /// outline holds the miter point it computes, as a line, exactly once in that subpath; at
    /// every bevel, the line to `p0 − n`.
    #[test]
    fn the_walks_vertices_are_kurbos_own() {
        let shapes: [&[[f64; 2]]; 3] = [
            &[
                [0.0, 0.0],
                [7.0, 1.0],
                [3.0, 9.0],
                [11.0, 13.0],
                [10.0, 0.5],
            ],
            &[[0.0, 0.0], [10.0, 0.0], [0.0, 1.0], [10.0, 2.5]],
            &SQUARE,
        ];
        for pts in shapes {
            let path = outline(&polyline(pts));
            let width = 1.25;
            let stroked = stroke(&path, width);
            let w = made(path.elements(), 0.1, width);
            assert!(!w.joins.is_empty());
            for j in &w.joins {
                let c = j.ab.cross(j.cd);
                let (n_a, n) = (norm(width, j.ab), norm(width, j.cd));
                let s = if c > 0.0 { -1.0 } else { 1.0 };
                let (this, last) = (j.p0 + s * n, j.p0 + s * n_a);
                let m = this - j.cd * (j.ab.cross(this - last) / c);
                let els = stroked.elements();
                if j.mitres {
                    assert!(
                        one_line_to(els, m).is_some(),
                        "{pts:?}: miter at {:?}",
                        j.p0
                    );
                    assert!(
                        one_line_to(els, j.p0).is_some(),
                        "{pts:?}: inner at {:?}",
                        j.p0
                    );
                } else {
                    assert!(
                        one_line_to(els, j.p0 - n).is_some(),
                        "{pts:?}: bevel at {:?}",
                        j.p0
                    );
                    assert!(
                        one_line_after(els, j.p0 + n).is_some(),
                        "{pts:?}: back at {:?}",
                        j.p0
                    );
                }
            }
        }
    }

    fn made_join(key: u64, mitres: bool) -> Made {
        let p = Point::new(key as f64, 0.0);
        Made {
            p0: p,
            ab: Vec2::new(1.0, 0.0),
            cd: Vec2::new(0.0, 1.0),
            mitres,
            key: Some(JoinKey::new(&[p; 4], &[p; 4], false)),
            subpath: 0,
        }
    }

    #[test]
    fn each_join_kurbo_makes_takes_the_next_unused_decision_of_its_key() {
        let (a, b) = (made_join(1, true), made_join(2, true));
        let mut c = Carried::new();
        c.insert(a.key.unwrap(), vec![Some(false), None, Some(true)]);
        c.insert(b.key.unwrap(), vec![Some(true)]);
        let none = Made {
            key: None,
            ..a.clone()
        };
        let joins = [
            a.clone(),
            b.clone(),
            none,
            a.clone(),
            b.clone(),
            a.clone(),
            a,
        ];
        assert_eq!(
            matched(&joins, &c),
            vec![Some(false), Some(true), None, None, None, Some(true), None]
        );
    }

    #[test]
    fn a_decision_unmatched_by_kurbo_is_never_spent() {
        // The walk met the key three times; kurbo made it once: the first decision applies.
        let a = made_join(1, false);
        let mut c = Carried::new();
        c.insert(a.key.unwrap(), vec![Some(true), Some(false), Some(false)]);
        assert_eq!(matched(&[a], &c), vec![Some(true)]);
    }
}
