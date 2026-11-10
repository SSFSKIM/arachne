//! A path layer's outline, stroke and rings (contract §4 and §8), through the shared kurbo.
//!
//! The outline is made in frame units from the path's shown points; the fill is the outline
//! mapped to pixels and flattened at 0.1 px; a stroke is kurbo's `stroke` of the outline in frame
//! units at `0.1 / s`, edited at the joins the record carries, then mapped and flattened as the
//! fill is. kurbo is a contract term: what this module adds around it is written from the text.

use std::collections::HashMap;

use arachne_core::xf::{Cap, Dash, Joint};
use kurbo::{BezPath, CubicBez, ParamCurve, ParamCurveArclen, PathEl, PathSeg, Point, Shape, Vec2};

use crate::camera::FrameMap;
use crate::draw::PathItem;

/// Cairo's default tolerance, in pixels.
pub const TOLERANCE: f64 = 0.1;
/// Cairo's miter limit.
pub const MITER_LIMIT: f64 = 10.0;
/// How far beyond the frame, in pixels, the guard band lies.
pub const GUARD: f64 = 65536.0;
/// Cairo's least length: a dash whose gap is shorter strokes solid.
pub const SOLID_DASH_GAP: f64 = 1.0 / 512.0;
/// The accuracy a guarded dash's pieces are measured to.
pub const DASH_ACCURACY: f64 = 1e-6;

/// CE's tolerance for two coordinates being one: `|a − b| <= 1e-6 + 1e-5 |b|`.
pub fn ce_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 + 1e-5 * b.abs()
}

/// CE's subpaths of `shown` (cubic layout), as `[start, end)` point ranges of at least four
/// points: split where a cubic's end and the next cubic's start differ in `x` or `y`.
pub fn subpaths(shown: &[[f64; 3]]) -> Vec<(usize, usize)> {
    let n = shown.len() - shown.len() % 4;
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 4;
    while i < n {
        let (a, b) = (shown[i - 1], shown[i]);
        if !(ce_close(a[0], b[0]) && ce_close(a[1], b[1])) {
            out.push((start, i));
            start = i;
        }
        i += 4;
    }
    out.push((start, n));
    out.retain(|(s, e)| e - s >= 4);
    out
}

/// Whether the subpath `[s, e)` of `shown` closes: its first and last points are one by
/// [`ce_close`] in `x` and `y`.
pub fn closes(shown: &[[f64; 3]], s: usize, e: usize) -> bool {
    let (a, b) = (shown[s], shown[e - 1]);
    ce_close(a[0], b[0]) && ce_close(a[1], b[1])
}

fn pt(p: [f64; 3]) -> Point {
    Point::new(p[0], p[1])
}

/// The outline in frame units: per subpath a move to its first point and a curve per cubic; a
/// closed subpath ends exactly at its first point and closes.
pub fn outline(shown: &[[f64; 3]]) -> BezPath {
    let mut out = BezPath::new();
    for (s, e) in subpaths(shown) {
        let closed = closes(shown, s, e);
        out.move_to(pt(shown[s]));
        let mut i = s;
        while i + 4 <= e {
            let end = if closed && i + 4 == e {
                shown[s]
            } else {
                shown[i + 3]
            };
            out.curve_to(pt(shown[i + 1]), pt(shown[i + 2]), pt(end));
            i += 4;
        }
        if closed {
            out.close_path();
        }
    }
    out
}

/// The guard's area (`guard_chords@1`), in frame units, or `None` where the guard does not apply:
/// the frame's box widened by `reach + 1`, where `reach` is the widest stroke's width (frame
/// units) times the miter limit, applying when a point of the outline lies within `reach` of the
/// guard band's edge or beyond it.
pub fn guard_area(user: &BezPath, map: &FrameMap, size: [u32; 2], widest: f64) -> Option<[f64; 4]> {
    let to_user = |px: f64, py: f64| [(px - map.x0) / map.xx, (py - map.y0) / map.yy];
    let reach = widest * MITER_LIMIT;
    let (w, h) = (size[0] as f64, size[1] as f64);
    let [ax, ay] = to_user(-GUARD, -GUARD);
    let [bx, by] = to_user(w + GUARD, h + GUARD);
    let band = [ax.min(bx), ay.min(by), ax.max(bx), ay.max(by)];
    let inside = user.elements().iter().all(|el| {
        let pts: &[Point] = match el {
            PathEl::MoveTo(p) | PathEl::LineTo(p) => std::slice::from_ref(p),
            PathEl::QuadTo(a, b) => &[*a, *b][..],
            PathEl::CurveTo(a, b, c) => &[*a, *b, *c][..],
            PathEl::ClosePath => &[],
        };
        pts.iter().all(|p| {
            p.x >= band[0] + reach
                && p.x <= band[2] - reach
                && p.y >= band[1] + reach
                && p.y <= band[3] - reach
        })
    });
    if inside {
        return None;
    }
    let [fx0, fy0] = to_user(0.0, 0.0);
    let [fx1, fy1] = to_user(w, h);
    Some([
        fx0.min(fx1) - reach - 1.0,
        fy0.min(fy1) - reach - 1.0,
        fx0.max(fx1) + reach + 1.0,
        fy0.max(fy1) + reach + 1.0,
    ])
}

/// Whether every point lies beyond one side of `area`.
fn misses(pts: &[Point], area: [f64; 4]) -> bool {
    pts.iter().all(|p| p.x < area[0])
        || pts.iter().all(|p| p.x > area[2])
        || pts.iter().all(|p| p.y < area[1])
        || pts.iter().all(|p| p.y > area[3])
}

/// A cubic's pieces under the guard, each handed on with whether it misses the area.
fn pieces(c: CubicBez, area: [f64; 4], out: &mut dyn FnMut(CubicBez, bool)) {
    fn split(
        c: CubicBez,
        depth: u32,
        area: [f64; 4],
        size: f64,
        out: &mut dyn FnMut(CubicBez, bool),
    ) {
        if misses(&[c.p0, c.p1, c.p2, c.p3], area) {
            out(c, true);
            return;
        }
        let r = c.bounding_box();
        if depth == 0 || r.width().max(r.height()) <= 64.0 * size {
            out(c, false);
            return;
        }
        let (a, b) = c.subdivide();
        split(a, depth - 1, area, size, out);
        split(b, depth - 1, area, size, out);
    }
    let size = (area[2] - area[0]).max(area[3] - area[1]);
    split(c, 40, area, size, out);
}

/// The outline under the guard: each piece that misses the area drawn as a line to its end.
pub fn guarded(path: &BezPath, area: [f64; 4]) -> BezPath {
    let mut out = BezPath::new();
    let mut last = Point::ZERO;
    let curve = |c: CubicBez, out: &mut BezPath| {
        pieces(c, area, &mut |piece, missed| {
            if missed {
                out.line_to(piece.p3);
            } else {
                out.curve_to(piece.p1, piece.p2, piece.p3);
            }
        })
    };
    for el in path.elements() {
        match *el {
            PathEl::MoveTo(p) => {
                out.move_to(p);
                last = p;
            }
            PathEl::LineTo(p) => {
                out.line_to(p);
                last = p;
            }
            PathEl::QuadTo(a, b) => {
                curve(kurbo::QuadBez::new(last, a, b).raise(), &mut out);
                last = b;
            }
            PathEl::CurveTo(a, b, c) => {
                curve(CubicBez::new(last, a, b, c), &mut out);
                last = c;
            }
            PathEl::ClosePath => out.close_path(),
        }
    }
    out
}

/// The dash a record strokes with (`dash_start_least_length@1`): `None` where it strokes solid
/// (no dash, a duty of 1, or a gap under Cairo's least length), else its two entries.
pub fn dash_entries(dash: Option<Dash>) -> Option<[f64; 2]> {
    let d = dash?;
    let entries = [d.period * d.duty, d.period * (1.0 - d.duty)];
    let total = entries[0] + entries[1];
    (d.duty < 1.0 && total - entries[0] >= SOLID_DASH_GAP).then_some(entries)
}

/// Whether Cairo's dashed stroker starts a subpath in an `on` dash: the offset `phase · period`
/// taken modulo the entries' total, then stepped past every entry it reaches or passes.
pub fn dash_starts_on(d: &Dash, entries: [f64; 2]) -> bool {
    let total = entries[0] + entries[1];
    let mut offset = (d.phase * d.period) % total;
    if offset < 0.0 {
        offset += total;
    }
    let (mut on, mut i) = (true, 0);
    while offset >= entries[i] {
        offset -= entries[i];
        on = !on;
        i = 1 - i;
    }
    on
}

fn dash(path: &BezPath, offset: f64, pattern: &[f64; 2]) -> Vec<PathEl> {
    kurbo::dash(path.elements().iter().copied(), offset, pattern).collect()
}

fn seg_el(seg: &PathSeg) -> PathEl {
    match *seg {
        PathSeg::Line(l) => PathEl::LineTo(l.p1),
        PathSeg::Quad(q) => PathEl::QuadTo(q.p1, q.p2),
        PathSeg::Cubic(c) => PathEl::CurveTo(c.p1, c.p2, c.p3),
    }
}

/// The dashes of the unguarded outline `path`, each subpath starting `offset` into `pattern`
/// (`dash_start_least_length@1`): kurbo's `dash` of the whole outline without the guard; with
/// it, each run of pieces that may reach the area dashed on its own at the length the pieces
/// before it put it, a closed subpath's seam joined.
pub fn dashed(path: &BezPath, offset: f64, pattern: &[f64; 2], area: Option<[f64; 4]>) -> BezPath {
    let Some(area) = area else {
        return BezPath::from_vec(dash(path, offset, pattern));
    };
    let mut out = Vec::new();
    let mut segs: Vec<(PathSeg, bool)> = Vec::new();
    let (mut start, mut last) = (Point::ZERO, Point::ZERO);
    for el in path.elements() {
        match *el {
            PathEl::MoveTo(p) => {
                dash_subpath(&segs, false, offset, pattern, &mut out);
                segs.clear();
                (start, last) = (p, p);
            }
            PathEl::LineTo(p) => {
                segs.push((
                    PathSeg::Line(kurbo::Line::new(last, p)),
                    misses(&[last, p], area),
                ));
                last = p;
            }
            PathEl::QuadTo(a, b) => {
                let c = kurbo::QuadBez::new(last, a, b).raise();
                pieces(c, area, &mut |c, missed| {
                    segs.push((PathSeg::Cubic(c), missed))
                });
                last = b;
            }
            PathEl::CurveTo(a, b, c) => {
                let c = CubicBez::new(last, a, b, c);
                pieces(c, area, &mut |c, missed| {
                    segs.push((PathSeg::Cubic(c), missed))
                });
                last = c.p3;
            }
            PathEl::ClosePath => {
                if last != start {
                    let line = kurbo::Line::new(last, start);
                    segs.push((PathSeg::Line(line), misses(&[last, start], area)));
                }
                dash_subpath(&segs, true, offset, pattern, &mut out);
                segs.clear();
                last = start;
            }
        }
    }
    dash_subpath(&segs, false, offset, pattern, &mut out);
    BezPath::from_vec(out)
}

/// One subpath's dashes under the guard into `out`.
fn dash_subpath(
    segs: &[(PathSeg, bool)],
    closed: bool,
    offset: f64,
    pattern: &[f64; 2],
    out: &mut Vec<PathEl>,
) {
    let Some(&(first, _)) = segs.first() else {
        return;
    };
    let run_path = |run: &[(PathSeg, bool)], close: bool| {
        let mut path = BezPath::new();
        path.move_to(run[0].0.start());
        for (seg, _) in run {
            path.push(seg_el(seg));
        }
        if close {
            path.close_path();
        }
        path
    };
    if segs.iter().all(|(_, missed)| !missed) {
        out.extend(dash(&run_path(segs, closed), offset, pattern));
        return;
    }
    let mut runs: Vec<(f64, &[(PathSeg, bool)])> = Vec::new();
    let (mut travelled, mut open) = (0.0, None);
    for (i, (seg, missed)) in segs.iter().enumerate() {
        match (missed, open) {
            (true, Some((at, from))) => {
                runs.push((at, &segs[from..i]));
                open = None;
            }
            (false, None) => open = Some((travelled, i)),
            _ => {}
        }
        travelled += seg.arclen(DASH_ACCURACY);
    }
    if let Some((at, from)) = open {
        runs.push((at, &segs[from..]));
    }
    let mut dashes: Vec<Vec<Vec<PathEl>>> = runs
        .iter()
        .map(|(at, run)| {
            let mut each: Vec<Vec<PathEl>> = Vec::new();
            for el in dash(&run_path(run, false), offset + at, pattern) {
                match (el, each.last_mut()) {
                    (PathEl::MoveTo(_), _) | (_, None) => each.push(vec![el]),
                    (_, Some(current)) => current.push(el),
                }
            }
            each
        })
        .collect();
    let start = first.start();
    let near = |p: Point| (p - start).hypot() <= 1e-9 * (1.0 + start.x.abs().max(start.y.abs()));
    if closed && runs.len() > 1 && !segs[0].1 && !segs[segs.len() - 1].1 {
        let last = runs.len() - 1;
        let opening = dashes[0].iter().position(|d| d[0] == PathEl::MoveTo(start));
        let closing = dashes[last]
            .iter()
            .position(|d| d.len() > 1 && d.last().and_then(|el| el.end_point()).is_some_and(near));
        if let (Some(i), Some(j)) = (opening, closing) {
            let head = dashes[0].remove(i);
            dashes[last][j].extend(head.into_iter().skip(1));
        }
    }
    out.extend(dashes.into_iter().flatten().flatten());
}

/// A join of a stroked path, by its two cubics' points' bits (each its start, its handles and its
/// end), and whether it is a subpath's closing join.
pub type JoinKey = ([u64; 16], bool);

fn join_key(from: [Point; 4], into: [Point; 4], closing: bool) -> JoinKey {
    let mut bits = [0u64; 16];
    for (k, p) in from.iter().chain(into.iter()).enumerate() {
        bits[2 * k] = p.x.to_bits();
        bits[2 * k + 1] = p.y.to_bits();
    }
    (bits, closing)
}

/// The carried decisions by join: per key, one per occurrence in the path's order, `Some(true)`
/// where Cairo mitres, `Some(false)` where it bevels, `None` where the record decides nothing.
pub type Decided = HashMap<JoinKey, Vec<Option<bool>>>;

/// The decisions the record's `joins` carry for the outline of `shown` (`join_mask@1`): the join
/// into each cubic from the last cubic before it that is not all one point, and for a closed
/// subpath its closing join, the first cubic's byte.
pub fn decided_joins(shown: &[[f64; 3]], joins: &[u8]) -> Decided {
    let mut out: Decided = HashMap::new();
    let byte = |i: usize| {
        joins
            .get(i / 4)
            .copied()
            .filter(|b| b & 1 != 0)
            .map(|b| b & 2 != 0)
    };
    for (s, e) in subpaths(shown) {
        let closed = closes(shown, s, e);
        let mut p0 = pt(shown[s]);
        let (mut first, mut last): (Option<[Point; 4]>, Option<[Point; 4]>) = (None, None);
        let mut i = s;
        while i + 4 <= e {
            let end = if closed && i + 4 == e {
                shown[s]
            } else {
                shown[i + 3]
            };
            let (p1, p2, p3) = (pt(shown[i + 1]), pt(shown[i + 2]), pt(end));
            i += 4;
            if p1 == p0 && p2 == p0 && p3 == p0 {
                continue;
            }
            let this = [p0, p1, p2, p3];
            match last {
                Some(from) => out
                    .entry(join_key(from, this, false))
                    .or_default()
                    .push(byte(i - 4)),
                None => first = Some(this),
            }
            (last, p0) = (Some(this), p3);
        }
        if let (true, Some(from), Some(into)) = (closed, last, first) {
            out.entry(join_key(from, into, true))
                .or_default()
                .push(byte(s));
        }
    }
    out
}

/// One join kurbo's stroker makes: at `p0`, from the tangent `ab` into `cd`.
struct KurboJoin {
    key: JoinKey,
    /// Which stroked subpath it is in.
    sub: usize,
    p0: Point,
    ab: Vec2,
    cd: Vec2,
    mitres: bool,
}

/// kurbo's stroker's tangents of a cubic at its ends.
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

/// The joins kurbo's stroker makes walking `path` in `style` at `tolerance`, and per stroked
/// subpath how many contours its outline holds (two where it closes).
fn kurbo_joins(
    path: &[PathEl],
    style: &kurbo::Stroke,
    tolerance: f64,
) -> (Vec<KurboJoin>, Vec<usize>) {
    let join_thresh = 2.0 * tolerance / style.width;
    let limit2 = style.miter_limit * style.miter_limit;
    let mut out = Vec::new();
    let mut contours: Vec<usize> = Vec::new();
    let (mut start, mut p0) = (Point::ZERO, Point::ZERO);
    let (mut last_tan, mut start_tan) = (Vec2::ZERO, Vec2::ZERO);
    let (mut first, mut last): (Option<[Point; 4]>, Option<[Point; 4]>) = (None, None);
    let mut started = false;
    let push = |out: &mut Vec<KurboJoin>,
                key: Option<JoinKey>,
                sub: usize,
                p0: Point,
                ab: Vec2,
                cd: Vec2| {
        let cross = ab.cross(cd);
        let dot = ab.dot(cd);
        let hypot = libm::hypot(cross, dot);
        if let Some(key) = key
            && (dot <= 0.0 || cross.abs() >= hypot * join_thresh)
        {
            out.push(KurboJoin {
                key,
                sub,
                p0,
                ab,
                cd,
                mitres: 2.0 * hypot < (hypot + dot) * limit2,
            });
        }
    };
    for el in path {
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
                        // kurbo strokes a closing line first; no carried decision reaches it.
                        let t = start - p0;
                        push(&mut out, None, sub, p0, last_tan, t);
                        (last_tan, last) = (t, None);
                    }
                    let key = match (last, first) {
                        (Some(a), Some(b)) => Some(join_key(a, b, true)),
                        _ => None,
                    };
                    push(&mut out, key, sub, start, last_tan, start_tan);
                    contours[sub] = 2;
                }
                (p0, started) = (start, false);
                continue;
            }
        };
        if started {
            let sub = contours.len() - 1;
            let key = match (last, pts) {
                (Some(a), Some(b)) => Some(join_key(a, b, false)),
                _ => None,
            };
            push(&mut out, key, sub, p0, last_tan, tan0);
        } else {
            (started, start_tan, first) = (true, tan0, pts);
            contours.push(1);
        }
        (last_tan, last, p0) = (tan1, pts, end);
    }
    (out, contours)
}

/// kurbo's offset of the tangent `t` at half the stroke's width.
fn norm(width: f64, t: Vec2) -> Vec2 {
    let scale = 0.5 * width / t.hypot();
    scale * Vec2::new(-t.y, t.x)
}

/// The one line to `to` among `els[within]`.
fn only_line_to(els: &[PathEl], within: std::ops::Range<usize>, to: Point) -> Option<usize> {
    let mut found = within.filter(|&i| matches!(els[i], PathEl::LineTo(p) if p == to));
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// The one line among `els[within]` that follows an element ending at `after`.
fn line_after(els: &[PathEl], within: std::ops::Range<usize>, after: Point) -> Option<usize> {
    let mut found = within
        .skip(1)
        .filter(|&i| matches!(els[i], PathEl::LineTo(_)) && els[i - 1].end_point() == Some(after));
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// `stroked`, kurbo's stroke of `path`, edited at each miter join whose carried decision differs
/// from kurbo's (`join_mask@1`): a carried bevel drops kurbo's two miter vertices; a carried miter
/// inserts them, the tip on the bisector at the miter limit's length. A join on parallel tangents,
/// or whose vertices are not found in kurbo's outline, is left as drawn.
pub fn edit_joins(
    stroked: BezPath,
    path: &[PathEl],
    style: &kurbo::Stroke,
    tolerance: f64,
    decided: &Decided,
) -> BezPath {
    if decided.is_empty() || style.join != kurbo::Join::Miter {
        return stroked;
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
        let moves: Vec<usize> = (0..els.len())
            .filter(|&i| matches!(els[i], PathEl::MoveTo(_)))
            .collect();
        let first: usize = contours[..j.sub].iter().sum();
        let Some(&lo) = moves.get(first) else {
            continue;
        };
        let hi = moves
            .get(first + contours[j.sub])
            .copied()
            .unwrap_or(els.len());
        let cross = j.ab.cross(j.cd);
        if cross == 0.0 {
            continue;
        }
        let (n_last, n) = (norm(style.width, j.ab), norm(style.width, j.cd));
        let s = if cross > 0.0 { -1.0 } else { 1.0 };
        let fp_last = j.p0 + s * n_last;
        let fp_this = j.p0 + s * n;
        let h = j.ab.cross(fp_this - fp_last) / cross;
        let miter_pt = fp_this - j.cd * h;
        if j.mitres {
            // Carried bevel where kurbo mitred: its tip and its inner vertex go.
            let (Some(a), Some(b)) = (
                only_line_to(&els, lo..hi, miter_pt),
                only_line_to(&els, lo..hi, j.p0),
            ) else {
                continue;
            };
            for i in if a > b { [a, b] } else { [b, a] } {
                els.remove(i);
            }
        } else if mitres {
            // Carried miter where kurbo bevelled: the tip at the limit's length on the outer side
            // (kurbo's forward path where the turn is positive), the join point on the inner.
            let reach = miter_pt - j.p0;
            let tip = j.p0 + reach * (style.miter_limit * half / reach.hypot());
            let forward = only_line_to(&els, lo..hi, j.p0 - n);
            let backward = line_after(&els, lo..hi, j.p0 + n);
            let (outer, inner) = if cross > 0.0 {
                (forward, backward)
            } else {
                (backward, forward)
            };
            let (Some(a), Some(b)) = (outer, inner) else {
                continue;
            };
            for (i, p) in if a > b {
                [(a, tip), (b, j.p0)]
            } else {
                [(b, j.p0), (a, tip)]
            } {
                els.insert(i, PathEl::LineTo(p));
            }
        }
    }
    BezPath::from_vec(els)
}

fn kurbo_join(joint: Joint) -> kurbo::Join {
    match joint {
        Joint::Auto | Joint::Miter => kurbo::Join::Miter,
        Joint::Round => kurbo::Join::Round,
        Joint::Bevel => kurbo::Join::Bevel,
    }
}

fn kurbo_cap(cap: Cap) -> kurbo::Cap {
    match cap {
        Cap::Auto | Cap::Butt => kurbo::Cap::Butt,
        Cap::Round => kurbo::Cap::Round,
        Cap::Square => kurbo::Cap::Square,
    }
}

/// A path through the frame mapping, point by point.
pub fn mapped(path: &BezPath, map: &FrameMap) -> BezPath {
    let m = |p: Point| {
        let [x, y] = map.map(p.x, p.y);
        Point::new(x, y)
    };
    let els = path.elements().iter().map(|el| match *el {
        PathEl::MoveTo(p) => PathEl::MoveTo(m(p)),
        PathEl::LineTo(p) => PathEl::LineTo(m(p)),
        PathEl::QuadTo(a, b) => PathEl::QuadTo(m(a), m(b)),
        PathEl::CurveTo(a, b, c) => PathEl::CurveTo(m(a), m(b), m(c)),
        PathEl::ClosePath => PathEl::ClosePath,
    });
    BezPath::from_vec(els.collect())
}

/// A ring: a closed polygon of pixel points.
pub type Ring = Vec<[f64; 2]>;

/// `path` (in pixels) flattened by kurbo at [`TOLERANCE`] into rings: each run from a move to a
/// close, or to the next move; a ring of fewer than three points is dropped.
pub fn rings(path: &BezPath) -> Vec<Ring> {
    let mut rings = Vec::new();
    let mut ring: Ring = Vec::new();
    kurbo::flatten(path.elements().iter().copied(), TOLERANCE, |el| match el {
        PathEl::MoveTo(p) => {
            if ring.len() > 2 {
                rings.push(std::mem::take(&mut ring));
            }
            ring.clear();
            ring.push([p.x, p.y]);
        }
        PathEl::LineTo(p) => ring.push([p.x, p.y]),
        PathEl::ClosePath => {
            if ring.len() > 2 {
                rings.push(std::mem::take(&mut ring));
            }
            ring.clear();
        }
        PathEl::QuadTo(..) | PathEl::CurveTo(..) => {}
    });
    if ring.len() > 2 {
        rings.push(ring);
    }
    rings
}

/// Whether a face strokes as a dot or not at all (`round_cap_dot@1`): `Some(true)` for a dot,
/// `Some(false)` for nothing, `None` for kurbo's stroke.
fn dot_rule(item: &PathItem, entries: Option<[f64; 2]>) -> Option<bool> {
    let s = &item.shown;
    if !item.face || s.iter().any(|p| p[0] != s[0][0] || p[1] != s[0][1]) {
        return None;
    }
    let starts_on = match (item.dash, entries) {
        (Some(d), Some(e)) => dash_starts_on(&d, e),
        _ => true,
    };
    Some(item.cap == Cap::Round && starts_on)
}

/// Each layer's rings in pixels (background stroke, fill, stroke), `None` for a layer not drawn.
pub fn layer_rings(item: &PathItem, size: [u32; 2]) -> [Option<Vec<Ring>>; 3] {
    let map = &item.map;
    let user = outline(&item.shown);
    let widest = item.layers[0]
        .unwrap_or(0.0)
        .max(item.layers[2].unwrap_or(0.0));
    let area = guard_area(&user, map, size, widest);
    let guarded_outline = match area {
        Some(area) => guarded(&user, area),
        None => user.clone(),
    };
    let scale = map.scale();
    let tolerance = TOLERANCE / scale;
    let entries = dash_entries(item.dash);
    let decided = item.joins.as_ref().map(|j| decided_joins(&item.shown, j));
    let dashes = match (item.dash, entries) {
        (Some(d), Some(pattern)) => Some(dashed(&user, d.phase * d.period, &pattern, area)),
        _ => None,
    };
    let dot = dot_rule(item, entries);
    std::array::from_fn(|k| {
        let w = item.layers[k]?;
        if k == 1 {
            return Some(rings(&mapped(&guarded_outline, map)));
        }
        let stroked = match dot {
            Some(false) => return Some(Vec::new()),
            Some(true) => {
                let c = item.shown[0];
                kurbo::Circle::new(Point::new(c[0], c[1]), w / 2.0).to_path(tolerance)
            }
            None => {
                let style = kurbo::Stroke::new(w)
                    .with_join(kurbo_join(item.joint))
                    .with_caps(kurbo_cap(item.cap))
                    .with_miter_limit(MITER_LIMIT);
                let elements = dashes.as_ref().unwrap_or(&guarded_outline).elements();
                let stroked = kurbo::stroke(
                    elements.iter().copied(),
                    &style,
                    &kurbo::StrokeOpts::default(),
                    tolerance,
                );
                match &decided {
                    Some(decided) => edit_joins(stroked, elements, &style, tolerance, decided),
                    None => stroked,
                }
            }
        };
        Some(rings(&mapped(&stroked, map)))
    })
}
