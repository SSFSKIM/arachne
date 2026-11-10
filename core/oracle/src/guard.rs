//! The guard (`guard_chords@1`, contract §4) and the dashes laid under it
//! (`dash_start_least_length@1`, contract §8), written from the contract's text.
//!
//! Far beyond the frame a path's curves are cut into pieces, and a piece whose control points all
//! lie beyond one side of the guard's area is drawn as a chord. A piece and its chord both lie in
//! its control hull, which misses the area, so nothing inside the area changes. A dashed stroke is
//! laid on the unguarded outline: under the guard, each run of pieces that do not miss is dashed
//! on its own at the length the pieces before it put it, and a closed subpath's seam is joined.

use kurbo::{
    BezPath, CubicBez, Line, ParamCurve, ParamCurveArclen, ParamCurveExtrema, PathEl, PathSeg,
    Point, QuadBez,
};

use crate::camera::FrameMap;
use crate::flatten::MITER_LIMIT;

/// How far beyond the frame, in pixels, the guard band lies.
pub const GUARD: f64 = 65536.0;
/// The accuracy a guarded dash's pieces are measured to.
pub const DASH_ACCURACY: f64 = 1e-6;
/// The split's depth limit.
const SPLIT_DEPTH: u32 = 40;
/// A piece is kept whole once its bounding box's larger side is at most this many times the
/// area's larger side.
const LONGEST_PIECE: f64 = 64.0;

/// A box `[x0, y0, x1, y1]` in frame units.
pub type Area = [f64; 4];

/// The frame units of the pixel box `[x0, y0, x1, y1]` under `map`.
fn unmapped(map: &FrameMap, b: [f64; 4]) -> Area {
    let (xa, xb) = ((b[0] - map.x0) / map.xx, (b[2] - map.x0) / map.xx);
    let (ya, yb) = ((b[1] - map.y0) / map.yy, (b[3] - map.y0) / map.yy);
    [xa.min(xb), ya.min(yb), xa.max(xb), ya.max(yb)]
}

fn points(el: &PathEl) -> impl Iterator<Item = Point> {
    let (n, ps) = match *el {
        PathEl::MoveTo(p) | PathEl::LineTo(p) => (1, [p; 3]),
        PathEl::QuadTo(a, b) => (2, [a, b, b]),
        PathEl::CurveTo(a, b, c) => (3, [a, b, c]),
        PathEl::ClosePath => (0, [Point::ZERO; 3]),
    };
    ps.into_iter().take(n)
}

/// The guard's area for the outline `outline` (frame units) of a record whose widest stroke is
/// `widest` frame units wide, drawn by `map` on a frame of `size` pixels; `None` where the guard
/// does not apply. With `reach` the widest width times the miter limit, the guard applies when a
/// point of the outline (a move's, a line's, or any of a curve's points) lies within `reach` of the
/// guard band's edge or beyond it, the band being the frame widened by [`GUARD`] pixels on every
/// side. The area is then the frame's box widened by `reach + 1`.
pub fn area(outline: &BezPath, map: &FrameMap, size: [u32; 2], widest: f64) -> Option<Area> {
    let (w, h) = (size[0] as f64, size[1] as f64);
    let frame = unmapped(map, [0.0, 0.0, w, h]);
    let band = unmapped(map, [-GUARD, -GUARD, w + GUARD, h + GUARD]);
    let reach = widest * MITER_LIMIT;
    let clear = |p: Point| {
        p.x > band[0] + reach
            && p.x < band[2] - reach
            && p.y > band[1] + reach
            && p.y < band[3] - reach
    };
    let applies = outline
        .elements()
        .iter()
        .any(|el| points(el).any(|p| !clear(p)));
    let out = reach + 1.0;
    applies.then_some([
        frame[0] - out,
        frame[1] - out,
        frame[2] + out,
        frame[3] + out,
    ])
}

/// Whether every one of `pts` lies beyond one side of `area`.
fn misses(pts: &[Point], area: Area) -> bool {
    pts.iter().all(|p| p.x < area[0])
        || pts.iter().all(|p| p.x > area[2])
        || pts.iter().all(|p| p.y < area[1])
        || pts.iter().all(|p| p.y > area[3])
}

/// The cubic `c` cut into pieces under the guard, each handed to `out` in order with whether it
/// misses the area: one that misses is kept as is; so is one whose bounding box's larger side is
/// at most 64 times the area's, or one at depth 0; any other is halved, each half cut again with
/// the limit one lower.
fn pieces(c: CubicBez, area: Area, out: &mut impl FnMut(CubicBez, bool)) {
    let size = (area[2] - area[0]).max(area[3] - area[1]);
    split(c, area, LONGEST_PIECE * size, SPLIT_DEPTH, out);
}

fn split(c: CubicBez, area: Area, longest: f64, limit: u32, out: &mut impl FnMut(CubicBez, bool)) {
    if misses(&[c.p0, c.p1, c.p2, c.p3], area) {
        out(c, true);
        return;
    }
    let b = ParamCurveExtrema::bounding_box(&c);
    if b.width().max(b.height()) <= longest || limit == 0 {
        out(c, false);
        return;
    }
    let (l, r) = c.subdivide();
    split(l, area, longest, limit - 1, out);
    split(r, area, longest, limit - 1, out);
}

/// The outline `path` under the guard of `area`: every move, line and close kept, every curve (a
/// quadratic raised to a cubic first) replaced by its pieces in place, a piece that misses the
/// area drawn as a line to its end and any other as a curve.
pub fn guarded(path: &BezPath, area: Area) -> BezPath {
    let mut out = BezPath::new();
    let (mut start, mut last) = (Point::ZERO, Point::ZERO);
    let cubic = |out: &mut BezPath, c: CubicBez| {
        pieces(c, area, &mut |p, miss| {
            if miss {
                out.line_to(p.p3);
            } else {
                out.curve_to(p.p1, p.p2, p.p3);
            }
        })
    };
    for el in path.elements() {
        match *el {
            PathEl::MoveTo(p) => {
                out.move_to(p);
                (start, last) = (p, p);
            }
            PathEl::LineTo(p) => {
                out.line_to(p);
                last = p;
            }
            PathEl::QuadTo(a, b) => {
                cubic(&mut out, QuadBez::new(last, a, b).raise());
                last = b;
            }
            PathEl::CurveTo(a, b, c) => {
                cubic(&mut out, CubicBez::new(last, a, b, c));
                last = c;
            }
            PathEl::ClosePath => {
                out.close_path();
                last = start;
            }
        }
    }
    out
}

/// One subpath of an outline: its start, its segments, and whether it closes.
struct Subpath {
    start: Point,
    segs: Vec<PathSeg>,
    closed: bool,
}

fn subpaths(path: &BezPath) -> Vec<Subpath> {
    let mut out: Vec<Subpath> = Vec::new();
    let mut last = Point::ZERO;
    for el in path.elements() {
        if !matches!(el, PathEl::MoveTo(_)) && out.last().is_none_or(|s| s.closed) {
            out.push(Subpath {
                start: last,
                segs: Vec::new(),
                closed: false,
            });
        }
        match *el {
            PathEl::MoveTo(p) => {
                out.push(Subpath {
                    start: p,
                    segs: Vec::new(),
                    closed: false,
                });
                last = p;
            }
            PathEl::ClosePath => {
                let s = out.last_mut().expect("a subpath");
                s.closed = true;
                last = s.start;
            }
            _ => {
                let s = out.last_mut().expect("a subpath");
                let seg = match *el {
                    PathEl::LineTo(p) => PathSeg::Line(Line::new(last, p)),
                    PathEl::QuadTo(a, b) => PathSeg::Quad(QuadBez::new(last, a, b)),
                    PathEl::CurveTo(a, b, c) => PathSeg::Cubic(CubicBez::new(last, a, b, c)),
                    PathEl::MoveTo(_) | PathEl::ClosePath => unreachable!(),
                };
                last = seg.end();
                s.segs.push(seg);
            }
        }
    }
    out
}

fn seg_el(seg: &PathSeg) -> PathEl {
    match *seg {
        PathSeg::Line(l) => PathEl::LineTo(l.p1),
        PathSeg::Quad(q) => PathEl::QuadTo(q.p1, q.p2),
        PathSeg::Cubic(c) => PathEl::CurveTo(c.p1, c.p2, c.p3),
    }
}

/// kurbo's `dash` of `els` at `offset`, cut into its dashes (each begun by a move).
fn dashes_of(els: Vec<PathEl>, offset: f64, pattern: &[f64; 2]) -> Vec<Vec<PathEl>> {
    let mut out: Vec<Vec<PathEl>> = Vec::new();
    for el in kurbo::dash(els.into_iter(), offset, &pattern[..]) {
        match (el, out.last_mut()) {
            (PathEl::MoveTo(_), _) | (_, None) => out.push(vec![el]),
            (_, Some(d)) => d.push(el),
        }
    }
    out
}

/// One subpath's pieces under the guard, each with whether it misses the area; a closed
/// subpath's closing line is a piece where its ends differ.
fn subpath_pieces(sub: &Subpath, area: Area) -> Vec<(PathSeg, bool)> {
    let mut out = Vec::new();
    for seg in &sub.segs {
        match *seg {
            PathSeg::Line(l) => out.push((*seg, misses(&[l.p0, l.p1], area))),
            PathSeg::Quad(q) => pieces(q.raise(), area, &mut |c, m| {
                out.push((PathSeg::Cubic(c), m))
            }),
            PathSeg::Cubic(c) => pieces(c, area, &mut |c, m| out.push((PathSeg::Cubic(c), m))),
        }
    }
    let end = sub.segs.last().map_or(sub.start, |s| s.end());
    if sub.closed && end != sub.start {
        let l = Line::new(end, sub.start);
        out.push((PathSeg::Line(l), misses(&[l.p0, l.p1], area)));
    }
    out
}

/// The dashes of the unguarded outline `path`, each subpath starting `offset` into `pattern`
/// (`dash_start_least_length@1`). Without the guard, kurbo's `dash` of the whole outline. Under
/// the guard of `area`, per subpath: one none of whose pieces misses is dashed whole, as one path
/// of its pieces that keeps its close; otherwise each run of pieces that do not miss is dashed on
/// its own, at `offset` plus the summed length of every piece before it, and the runs' dashes are
/// emitted in run order. A closed subpath with several runs, whose first and last pieces both do
/// not miss, has its first run's dash that starts at the subpath's start joined onto the end of
/// the last run's dash that ends within `1e-9 (1 + max(|x|, |y|))` of that start.
pub fn dashed(path: &BezPath, offset: f64, pattern: &[f64; 2], area: Option<Area>) -> BezPath {
    let Some(area) = area else {
        return BezPath::from_vec(
            kurbo::dash(path.elements().iter().copied(), offset, &pattern[..]).collect(),
        );
    };
    let mut out: Vec<PathEl> = Vec::new();
    for sub in subpaths(path) {
        let pieces = subpath_pieces(&sub, area);
        if pieces.is_empty() {
            continue;
        }
        if pieces.iter().all(|(_, miss)| !miss) {
            let mut els = vec![PathEl::MoveTo(sub.start)];
            els.extend(pieces.iter().map(|(seg, _)| seg_el(seg)));
            if sub.closed {
                els.push(PathEl::ClosePath);
            }
            out.extend(kurbo::dash(els.into_iter(), offset, &pattern[..]));
            continue;
        }
        let mut runs: Vec<Vec<Vec<PathEl>>> = Vec::new();
        let mut at = 0.0;
        let mut i = 0;
        while i < pieces.len() {
            if pieces[i].1 {
                at += pieces[i].0.arclen(DASH_ACCURACY);
                i += 1;
                continue;
            }
            let run_at = at;
            let mut els = vec![PathEl::MoveTo(pieces[i].0.start())];
            while i < pieces.len() && !pieces[i].1 {
                els.push(seg_el(&pieces[i].0));
                at += pieces[i].0.arclen(DASH_ACCURACY);
                i += 1;
            }
            runs.push(dashes_of(els, offset + run_at, pattern));
        }
        let seam = sub.closed && runs.len() > 1 && !pieces[0].1 && !pieces[pieces.len() - 1].1;
        if seam {
            let s = sub.start;
            let tol = 1e-9 * (1.0 + s.x.abs().max(s.y.abs()));
            let near = |p: Point| (p.x - s.x).abs() <= tol && (p.y - s.y).abs() <= tol;
            let last = runs.len() - 1;
            let opening = runs[0].iter().position(|d| d[0] == PathEl::MoveTo(s));
            let closing = runs[last]
                .iter()
                .rposition(|d| d.last().and_then(PathEl::end_point).is_some_and(near));
            if let (Some(a), Some(b)) = (opening, closing) {
                let first = runs[0].remove(a);
                runs[last][b].extend(first.into_iter().skip(1));
            }
        }
        out.extend(runs.into_iter().flatten().flatten());
    }
    BezPath::from_vec(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(a: [f64; 2], b: [f64; 2]) -> CubicBez {
        let at = |k: f64| {
            Point::new(
                (a[0] * (3.0 - k) + b[0] * k) / 3.0,
                (a[1] * (3.0 - k) + b[1] * k) / 3.0,
            )
        };
        CubicBez::new(at(0.0), at(1.0), at(2.0), at(3.0))
    }

    fn closed(corners: &[[f64; 2]]) -> BezPath {
        let mut p = BezPath::new();
        p.move_to(Point::new(corners[0][0], corners[0][1]));
        for (i, a) in corners.iter().enumerate() {
            let c = line(*a, corners[(i + 1) % corners.len()]);
            p.curve_to(c.p1, c.p2, c.p3);
        }
        p.close_path();
        p
    }

    /// The flat 16 x 16 px view whose frame units are pixels about the centre, `y` up.
    const MAP: FrameMap = FrameMap {
        xx: 1.0,
        yy: -1.0,
        x0: 8.0,
        y0: 8.0,
    };

    const FAR: [[f64; 2]; 5] = [
        [-4.0, -4.0],
        [4.0, -4.0],
        [100000.0, -4.0],
        [100000.0, 4.0],
        [-4.0, 4.0],
    ];

    #[test]
    fn the_guard_applies_within_reach_of_its_band_and_its_area_is_the_frame_widened() {
        // The band's right edge is at x = 8 + 65536; a stroke 1 wide reaches 10.
        let edge = 8.0 + GUARD;
        let at = |x: f64| closed(&[[-4.0, -4.0], [x, -4.0], [x, 4.0]]);
        assert_eq!(area(&at(edge - 10.5), &MAP, [16, 16], 1.0), None);
        assert_eq!(
            area(&at(edge - 10.0), &MAP, [16, 16], 1.0),
            Some([-19.0, -19.0, 19.0, 19.0])
        );
        assert!(area(&at(edge + 5.0), &MAP, [16, 16], 1.0).is_some());
        // A curve's control points count as the outline's points.
        let mut p = BezPath::new();
        p.move_to((0.0, 0.0));
        p.curve_to((edge, 0.0), (0.0, 1.0), (0.0, 2.0));
        assert!(area(&p, &MAP, [16, 16], 0.0).is_some());
    }

    #[test]
    fn far_pieces_are_chords_and_every_move_and_close_is_kept() {
        let path = closed(&FAR);
        let a = area(&path, &MAP, [16, 16], 1.0).expect("the guard applies");
        let g = guarded(&path, a);
        let els = g.elements();
        assert_eq!(els.first(), Some(&PathEl::MoveTo(Point::new(-4.0, -4.0))));
        assert_eq!(els.last(), Some(&PathEl::ClosePath));
        assert_eq!(
            els.iter()
                .filter(|e| matches!(e, PathEl::MoveTo(_)))
                .count(),
            1
        );
        assert_eq!(
            els.iter()
                .filter(|e| matches!(e, PathEl::ClosePath))
                .count(),
            1
        );
        // The edges near the frame stay whole curves; the far ones become chords.
        assert!(
            els[1] == path.elements()[1],
            "the first edge is kept as it was"
        );
        assert_eq!(els[els.len() - 2], path.elements()[5], "so is the last");
        assert!(els.iter().any(|e| matches!(e, PathEl::LineTo(_))));
        // Every element ends on the outline: the chords stand in place.
        for e in els {
            if let Some(p) = e.end_point() {
                assert!(
                    p.y == -4.0 || p.y == 4.0 || p.x == -4.0 || p.x == 100000.0,
                    "{p:?}"
                );
            }
        }
    }

    #[test]
    fn a_piece_misses_only_with_every_control_point_beyond_one_side() {
        let a = [-1.0, -1.0, 1.0, 1.0];
        let p = |x: f64, y: f64| Point::new(x, y);
        assert!(misses(&[p(2.0, 0.0), p(3.0, 5.0)], a));
        assert!(
            !misses(&[p(2.0, 0.0), p(1.0, 5.0)], a),
            "on the side is not beyond it"
        );
        assert!(
            !misses(&[p(2.0, 0.0), p(0.0, 2.0)], a),
            "beyond two sides is not beyond one"
        );
    }

    #[test]
    fn a_subpath_none_of_whose_pieces_misses_is_dashed_whole_across_its_seam() {
        // A square near the frame, dashed on at its start and its end: kurbo joins the last dash
        // onto the first, so no dash begins at the start.
        let near = closed(&[[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]]);
        let far = closed(&FAR);
        let mut both = near.clone();
        both.extend(far.elements().iter().copied());
        let a = area(&both, &MAP, [16, 16], 1.0).expect("the far subpath brings the guard");
        let pattern = [1.5, 0.5];
        let with = dashed(&near, 0.5, &pattern, Some(a));
        let without = dashed(&near, 0.5, &pattern, None);
        assert_eq!(with.elements(), without.elements());
        assert!(
            !with
                .elements()
                .contains(&PathEl::MoveTo(Point::new(-4.0, -4.0)))
        );
    }

    #[test]
    fn a_guarded_seam_joins_the_last_runs_dash_onto_the_first() {
        let path = closed(&FAR);
        let a = area(&path, &MAP, [16, 16], 1.0).expect("the guard applies");
        let d = dashed(&path, 0.5, &[1.5, 0.5], Some(a));
        let start = Point::new(-4.0, -4.0);
        let els = d.elements();
        // No dash begins at the start: the one that opened there now runs on from the last.
        assert!(!els.contains(&PathEl::MoveTo(start)));
        let through = els
            .windows(2)
            .position(|w| w[0].end_point() == Some(start) && !matches!(w[1], PathEl::MoveTo(_)));
        let i = through.expect("a dash passes through the start");
        assert!(
            matches!(els[i], PathEl::CurveTo(..)),
            "it arrives along the last edge"
        );
        assert_eq!(
            els[i + 1].end_point().map(|p| p.y),
            Some(-4.0),
            "and leaves along the first"
        );
        // Pieces that miss are measured, never dashed: no dash reaches the far side, only the
        // pieces next to the area that do not miss it.
        let reach = LONGEST_PIECE * (a[2] - a[0]);
        for e in els {
            if let Some(p) = e.end_point() {
                assert!(p.x <= a[2] + reach, "{p:?}");
            }
        }
    }

    #[test]
    fn each_run_is_dashed_at_the_length_the_pieces_before_it_put_it() {
        // Out along y = 0 to far beyond the frame and back along y = 1. The returning run is the
        // last piece of the third edge; it is dashed as kurbo dashes it alone at the phase plus
        // the length of everything before it: the first two edges and the third's missing pieces.
        let mut path = BezPath::new();
        path.move_to((-4.0, 0.0));
        for c in [
            line([-4.0, 0.0], [200000.0, 0.0]),
            line([200000.0, 0.0], [200000.0, 1.0]),
            line([200000.0, 1.0], [-4.0, 1.0]),
        ] {
            path.curve_to(c.p1, c.p2, c.p3);
        }
        let a = area(&path, &MAP, [16, 16], 1.0).expect("the guard applies");
        let g = guarded(&path, a);
        let els = g.elements();
        let PathEl::CurveTo(c1, c2, c3) = els[els.len() - 1] else {
            panic!("the run is a curve");
        };
        let x0 = els[els.len() - 2].end_point().unwrap();
        assert!(
            x0.x > a[2]
                && els[2..els.len() - 1]
                    .iter()
                    .all(|e| matches!(e, PathEl::LineTo(_)))
        );
        let before = 200004.0 + 1.0 + (200000.0 - x0.x);
        let pattern = [1.5, 0.5];
        let run = [PathEl::MoveTo(x0), PathEl::CurveTo(c1, c2, c3)];
        let want: Vec<PathEl> = kurbo::dash(run.into_iter(), 0.25 + before, &pattern[..]).collect();
        let got: Vec<PathEl> = dashed(&path, 0.25, &pattern, Some(a))
            .elements()
            .iter()
            .copied()
            .filter(|e| e.end_point().is_some_and(|p| p.y > 0.5))
            .collect();
        // The first edge's run ends at its last piece that does not miss; its dashes are at
        // y = 0. Those at y = 1 are the returning run's alone.
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            let (g, w) = (g.end_point().unwrap(), w.end_point().unwrap());
            assert!((g - w).hypot() < 1e-6, "{g:?} against {w:?}");
        }
        // Restarting the pattern at the run would move its dashes by the length before it,
        // modulo the period: by 0.47 here.
        assert!(libm::fmod(before, 2.0) > 0.4 && libm::fmod(before, 2.0) < 0.5);
    }
}
