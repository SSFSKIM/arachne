//! A stroke's miter joins decided as Cairo decides them, where the record says how (XF 1.4's
//! `joins`). kurbo decides a join from the tangents of the points it is handed, which are snapped;
//! Cairo, from slopes between CE's own points rounded to its 24.8 fixed point. Where a control arm
//! is a few 1/256 px long, or the turn is near the miter limit, the two can differ. The writer
//! decides each join on CE's points as Cairo does and carries the decision; the stroke is kurbo's
//! own (`kurbo::stroke`, unchanged), edited only at a join whose carried decision differs from
//! kurbo's, on kurbo's own vertices there:
//!
//! - carried bevel where kurbo mitred: its miter's two vertices (the tip on the outer side, the
//!   join point on the inner) are dropped, which leaves kurbo's own bevel outline;
//! - carried miter where kurbo bevelled: the two are inserted as kurbo inserts them, the tip on the
//!   bisector at the miter limit's length (kurbo's tangents put it past the limit there; Cairo's
//!   own tip, from its rounded slopes, is within it). Where kurbo's tangents are parallel no tip
//!   can be placed: the bevel stays.
//!
//! The vertices are computed with kurbo's own arithmetic (its tangents, the offset `norm`, the
//! miter point), so they are found in its outline by value; a join whose vertices are not found is
//! left as kurbo drew it.

use std::collections::HashMap;

use kurbo::{BezPath, CubicBez, PathEl, Point, Vec2};

/// A join of a stroked path, by its two elements' points' bits (each its start, its control
/// points and its end; a line's controls are its start): the one it leaves and the one it enters,
/// and whether it is a subpath's closing join.
pub type JoinKey = ([u64; 16], bool);

pub fn key(from: [Point; 4], into: [Point; 4], closing: bool) -> JoinKey {
    let mut bits = [0u64; 16];
    for (k, p) in from.iter().chain(into.iter()).enumerate() {
        (bits[2 * k], bits[2 * k + 1]) = (p.x.to_bits(), p.y.to_bits());
    }
    (bits, closing)
}

/// How the edits went at one stroke: counted for the diagnostics and the tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Edits {
    /// Miters dropped (carried bevel where kurbo mitred).
    pub dropped: u32,
    /// Miters inserted (carried miter where kurbo bevelled), each at the limit's length.
    pub inserted: u32,
    /// Carried miters left bevelled: kurbo's tangents parallel there.
    pub parallel: u32,
    /// Joins whose vertices were not found in kurbo's outline: left as drawn.
    pub unmatched: u32,
}

/// One join kurbo makes: at `p0`, from `ab` (the tangent arriving) to `cd` (leaving), and whether
/// kurbo mitres it.
struct Join {
    key: JoinKey,
    /// The stroked subpath it is in: kurbo's outline holds each one's contours in turn.
    sub: usize,
    p0: Point,
    ab: Vec2,
    cd: Vec2,
    mitres: bool,
}

/// kurbo's stroker's tangents of a cubic at its ends (`PathSeg::tangents`).
fn tangents(c: CubicBez) -> (Vec2, Vec2) {
    const EPS: f64 = 1e-12;
    let d01 = c.p1 - c.p0;
    let d0 = if d01.hypot2() > EPS {
        d01
    } else {
        let d02 = c.p2 - c.p0;
        if d02.hypot2() > EPS { d02 } else { c.p3 - c.p0 }
    };
    let d23 = c.p3 - c.p2;
    let d1 = if d23.hypot2() > EPS {
        d23
    } else {
        let d13 = c.p3 - c.p1;
        if d13.hypot2() > EPS { d13 } else { c.p3 - c.p0 }
    };
    (d0, d1)
}

/// The miter joins kurbo makes stroking `path` in `style` at `tolerance`, as its stroker walks the
/// path (`kurbo::stroke`): each element's join with the one before, a closed subpath's closing
/// join, and none where the turn is too slight for kurbo to join at all. Quadratic elements and a
/// closing line are walked but not keyed: no carried decision reaches them.
fn kurbo_joins(path: &[PathEl], style: &kurbo::Stroke, tolerance: f64) -> (Vec<Join>, Vec<usize>) {
    let join_thresh = 2.0 * tolerance / style.width;
    let limit2 = libm::pow(style.miter_limit, 2.0);
    let mut out = Vec::new();
    // Per stroked subpath, the contours kurbo's outline gives it: one, two where it closes.
    let mut contours: Vec<usize> = Vec::new();
    let (mut start, mut p0) = (Point::ZERO, Point::ZERO);
    let (mut last_tan, mut start_tan) = (Vec2::ZERO, Vec2::ZERO);
    // The first and the last element stroked in the subpath, by their points; `None` for one no
    // carried decision reaches (a quadratic).
    let (mut first, mut last): (Option<[Point; 4]>, Option<[Point; 4]>) = (None, None);
    let mut started = false;
    let join =
        |out: &mut Vec<Join>, key: Option<JoinKey>, sub: usize, p0: Point, ab: Vec2, cd: Vec2| {
            let cross = ab.cross(cd);
            let dot = ab.dot(cd);
            let hypot = libm::hypot(cross, dot);
            if let Some(key) = key
                && (dot <= 0.0 || cross.abs() >= hypot * join_thresh)
            {
                let mitres = 2.0 * hypot < (hypot + dot) * limit2;
                out.push(Join {
                    key,
                    sub,
                    p0,
                    ab,
                    cd,
                    mitres,
                });
            }
        };
    let keyed = |from: Option<[Point; 4]>, into: Option<[Point; 4]>, closing: bool| {
        Some(key(from?, into?, closing))
    };
    for el in path {
        // The element kurbo strokes here, its tangents at its ends and its points.
        let (tan0, tan1, pts, end) = match *el {
            PathEl::MoveTo(p) => {
                (start, p0, started, first, last) = (p, p, false, None, None);
                continue;
            }
            PathEl::LineTo(p1) => {
                if p1 == p0 {
                    continue;
                }
                let t = p1 - p0;
                (t, t, Some([p0, p0, p0, p1]), p1)
            }
            PathEl::QuadTo(p1, p2) => {
                if p1 == p0 && p2 == p0 {
                    continue;
                }
                // kurbo's stroker's tangents of a quadratic (`PathSeg::tangents`).
                let (d01, d12) = (p1 - p0, p2 - p1);
                let tan0 = if d01.hypot2() > 1e-12 { d01 } else { p2 - p0 };
                let tan1 = if d12.hypot2() > 1e-12 { d12 } else { p2 - p0 };
                (tan0, tan1, None, p2)
            }
            PathEl::CurveTo(p1, p2, p3) => {
                if p1 == p0 && p2 == p0 && p3 == p0 {
                    continue;
                }
                let (tan0, tan1) = tangents(CubicBez::new(p0, p1, p2, p3));
                (tan0, tan1, Some([p0, p1, p2, p3]), p3)
            }
            PathEl::ClosePath => {
                if started {
                    let sub = contours.len() - 1;
                    if p0 != start {
                        // kurbo strokes a closing line first: no carried decision reaches it.
                        let t = start - p0;
                        join(&mut out, None, sub, p0, last_tan, t);
                        (last_tan, last) = (t, None);
                    }
                    join(
                        &mut out,
                        keyed(last, first, true),
                        sub,
                        start,
                        last_tan,
                        start_tan,
                    );
                    contours[sub] = 2;
                }
                (p0, started) = (start, false);
                continue;
            }
        };
        if started {
            let sub = contours.len() - 1;
            join(&mut out, keyed(last, pts, false), sub, p0, last_tan, tan0);
        } else {
            (started, start_tan, first) = (true, tan0, pts);
            contours.push(1);
        }
        (last_tan, last, p0) = (tan1, pts, end);
    }
    (out, contours)
}

/// kurbo's offset of the tangent `t` at half the stroke's width (its stroker's `norm`).
fn norm(style: &kurbo::Stroke, t: Vec2) -> Vec2 {
    let scale = 0.5 * style.width / t.hypot();
    scale * Vec2::new(-t.y, t.x)
}

/// Where in a stroke's elements a vertex goes.
type Find<'a> = &'a dyn Fn(&[PathEl]) -> Option<usize>;

/// The position of the line to `to` among `els[within]`, and only one.
fn only_line_to(els: &[PathEl], within: std::ops::Range<usize>, to: Point) -> Option<usize> {
    let mut found = within.filter(|&i| matches!(els[i], PathEl::LineTo(p) if p == to));
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// The position of the line among `els[within]` that follows an element ending at `after`, and
/// only one.
fn line_after(els: &[PathEl], within: std::ops::Range<usize>, after: Point) -> Option<usize> {
    let mut found = within
        .skip(1)
        .filter(|&i| matches!(els[i], PathEl::LineTo(_)) && els[i - 1].end_point() == Some(after));
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// The carried decisions, by join ([`key`]): for joins whose elements are the same points, one per
/// occurrence, in the order the path holds them; `None` for an occurrence decided nothing, left as
/// kurbo strokes it.
pub type Decided = HashMap<JoinKey, Vec<Option<bool>>>;

/// `stroked`, kurbo's stroke of `path` in `style` at `tolerance` (`kurbo::stroke`), with each of its
/// miter joins whose carried decision (`decided`) differs from kurbo's edited to it.
pub fn apply(
    stroked: BezPath,
    path: &[PathEl],
    style: &kurbo::Stroke,
    tolerance: f64,
    decided: &Decided,
) -> (BezPath, Edits) {
    let mut edits = Edits::default();
    if decided.is_empty() || style.join != kurbo::Join::Miter {
        return (stroked, edits);
    }
    let mut els = stroked.into_elements();
    let half = 0.5 * style.width;
    let (joins, contours) = kurbo_joins(path, style, tolerance);
    let mut seen: HashMap<JoinKey, usize> = HashMap::new();
    for j in joins {
        let occurrence = seen.entry(j.key).or_default();
        let Some(&carried) = decided.get(&j.key).and_then(|d| d.get(*occurrence)) else {
            continue;
        };
        *occurrence += 1;
        let Some(mitres) = carried.filter(|&m| m != j.mitres) else {
            continue;
        };
        // The join's own subpath in kurbo's outline: its contours, each begun by a move.
        let moves: Vec<usize> = (0..els.len())
            .filter(|&i| matches!(els[i], PathEl::MoveTo(_)))
            .collect();
        let first: usize = contours[..j.sub].iter().sum();
        let (Some(&lo), hi) = (
            moves.get(first),
            moves
                .get(first + contours[j.sub])
                .copied()
                .unwrap_or(els.len()),
        ) else {
            edits.unmatched += 1;
            continue;
        };
        let cross = j.ab.cross(j.cd);
        if cross == 0.0 {
            // kurbo places no miter on parallel tangents, mitring or not.
            if mitres {
                edits.parallel += 1;
            }
            continue;
        }
        let (n_last, n) = (norm(style, j.ab), norm(style, j.cd));
        // The outer side (kurbo's forward path where the turn is positive) and its offsets.
        let s = if cross > 0.0 { -1.0 } else { 1.0 };
        let fp_last = j.p0 + s * n_last;
        let fp_this = j.p0 + s * n;
        let h = j.ab.cross(fp_this - fp_last) / cross;
        let miter_pt = fp_this - j.cd * h;
        // Where each side's miter vertex goes in kurbo's outline: on the forward path (the side
        // `p0 - n`, as written), before its bevel's line to `p0 - n`; on the backward one (the
        // side `p0 + n`, written reversed), before the line that follows its end at `p0 + n`.
        let forward = |els: &[PathEl]| only_line_to(els, lo..hi, j.p0 - n);
        let backward = |els: &[PathEl]| line_after(els, lo..hi, j.p0 + n);
        let (outer_at, inner_at): (Find, Find) = if cross > 0.0 {
            (&forward, &backward)
        } else {
            (&backward, &forward)
        };
        if j.mitres {
            // Carried bevel: drop kurbo's two miter vertices.
            let (Some(a), Some(b)) = (
                only_line_to(&els, lo..hi, miter_pt),
                only_line_to(&els, lo..hi, j.p0),
            ) else {
                edits.unmatched += 1;
                continue;
            };
            for i in if a > b { [a, b] } else { [b, a] } {
                els.remove(i);
            }
            edits.dropped += 1;
        } else {
            // Carried miter: kurbo's two miter vertices, the tip at the limit's length.
            let reach = miter_pt - j.p0;
            let tip = j.p0 + reach * (style.miter_limit * half / reach.hypot());
            let (Some(a), Some(b)) = (outer_at(&els), inner_at(&els)) else {
                edits.unmatched += 1;
                continue;
            };
            for (i, p) in if a > b {
                [(a, tip), (b, j.p0)]
            } else {
                [(b, j.p0), (a, tip)]
            } {
                els.insert(i, PathEl::LineTo(p));
            }
            edits.inserted += 1;
        }
    }
    (BezPath::from_vec(els), edits)
}
