//! The path rasterizer: paths (CE's cubic Bézier outlines, glyphs included) lowered to
//! coverage layers, and those rasterized into fragments carrying their exact area, with their masks
//! at the sub-samples on demand.
//!
//! - Placement: a path's points through its placement, then as CE places a mobject's points for
//!   display (`project::pre_display`), as for meshes.
//! - Outline: CE's subpath rule on the placed points (`VMobject.get_subpath_split_indices_from_points`
//!   in two dimensions, and a subpath closed when its first and last points are equal by
//!   `consider_points_equals_2d`), each subpath a move and its cubics, in the frame's units, as CE
//!   hands them to Cairo.
//! - Paint, in CE's order: the background stroke, the fill, the stroke, as three layers at one depth
//!   tying by that order. The fill is nonzero (Cairo's default, which CE never changes); strokes are
//!   expanded by kurbo in the frame's units, as Cairo strokes in user space: CE's width times the
//!   view's `cairo_line_width_multiple`, its joint (auto is a miter) and cap (auto is butt), Cairo's
//!   miter limit of 10, and the record's dash. Each layer is flattened on the pixel plane to Cairo's
//!   tolerance of 0.1 px, so flattening follows the projected size.
//! - Colour: one colour, or, for a paint of two or more, a linear gradient between the record's
//!   ends placed for display, coloured as Cairo's pattern and pixman's fetch colour it
//!   ([`crate::gradient`]), each pixel from the start of the composite run Cairo would paint it in
//!   ([`composite_runs`]). A path CE's lighting shades (`shade_in_3d` under a camera that shades)
//!   is lit into two stops at its start and end corners (`shade::ce_face_stops`), its gradient
//!   between those corners.
//! - Depth: a path in world space under a 3D camera lies on its plane when it has one seen clearly
//!   (`depth::WorldPlane`, made canonical, so text and axes on one plane, and a mesh on it, tie
//!   exactly); otherwise (a curve in space, a plane seen nearly edge-on) each pixel takes the depth
//!   of the path's own curve (`depth::Centreline`: the front-most part that reaches the pixel, else
//!   the nearest). A fixed-orientation path is a camera-facing
//!   plane at its centre's depth; anything else is in front of the scene, tying by draw order.
//! - Pre-projected (XF 1.1's tail): the points, paints and gradient ends a camera of the scene's
//!   own produced, drawn as they are, never projected or lit again. Depth follows the rules above
//!   in that camera's axes, the points put back where its projection took them from
//!   (`Projection::in_camera_axes`, `unproject`), so the view's angles, which that camera may not
//!   have drawn with, never reach it.
//! - Coverage: each pixel's exact nonzero area (`scanline::cover`); a pixel the compositor resolves
//!   sample by sample asks for the layer's winding at the sub-samples (`scanline::mask`).

use crate::composite::{
    Coverage, Fragment, OpaquePass, Plane, Producer, Rect, SAMPLES, TransparentPass, premultiply,
    sample_cell,
};
use crate::depth::{Centreline, WorldPlane};
use crate::gradient::{self, Linear};
use crate::project::{FrameMap, Placement, Projection, placement, pre_display, produced_key};
use crate::scanline::{self, Edge, Scratch};
use crate::tiles::{TILE, tile_index, tiles_across};
use crate::xf::*;
use kurbo::{BezPath, ParamCurve as _, ParamCurveArclen as _, PathEl, PathSeg, Point, Shape as _};

/// Cairo's default tolerance for flattening curves, in pixels.
pub const TOLERANCE: f64 = 0.1;

/// Cairo's default miter limit, which CE never changes (kurbo's default is 4).
pub const MITER_LIMIT: f64 = 10.0;

/// CE's default `tolerance_for_point_equality`, which XF does not carry.
const POINT_EQUALITY: f64 = 1e-6;

/// A path seen within this cosine of edge-on (about 3 degrees) takes its depth from its curve, not
/// from its plane: a stroke's width is on screen, and across it a grazing plane's depth would run
/// far in front of and behind the curve.
const MIN_PLANE_COS: f64 = 0.05;

/// How far outside the frame, in pixels, a path may reach before its curves outside the frame are
/// replaced by their chords.
const GUARD: f64 = 65536.0;

/// One path to draw: a shape's points through a placement, painted by a record's paint block, with
/// each paint resolved to its colour rows (straight RGBA, before the group opacity: one colour, or
/// a gradient's stops) or none.
pub struct PathDraw<'a> {
    pub record: &'a Record,
    /// The shape's points, or the points a reveal or a blend made of it
    /// ([`crate::geometry::path_points`], placed already).
    pub points: std::borrow::Cow<'a, [[f64; 3]]>,
    pub place: Affine,
    pub fill: Option<Vec<[f64; 4]>>,
    pub stroke: Option<Vec<[f64; 4]>>,
    pub background_stroke: Option<Vec<[f64; 4]>>,
    /// A gradient's ends in world space, as `get_gradient_start_and_end_points` gives them: where
    /// a paint of two or more rows runs between, unless CE's lighting shades the path (whose
    /// gradient runs between its start and end corners).
    pub ends: Option<[[f64; 3]; 2]>,
    /// A face of a surface (a face group's): an outline on no one plane takes the plane nearest it
    /// ([`WorldPlane::nearest`]), not its curve's depth.
    pub surface: bool,
    /// The points are where a camera of the scene's own placed them (a pre-projected record's
    /// tail), and the paints and `ends` are what it produced: nothing is projected or lit again,
    /// and depth is keyed in that camera's axes ([`crate::project::produced_key`]).
    pub pre_projected: bool,
    /// The stroke widths drawn, in CE's units: the record's, or a paint blend's mix of them.
    pub stroke_width: f64,
    pub background_stroke_width: f64,
    pub order: u32,
    /// The first of the three sub-orders the path's layers take (background stroke, fill, stroke).
    pub sub: u32,
}

/// Where a layer's fragments sit in depth.
#[derive(Clone, Copy, Debug)]
enum LayerDepth {
    Plane(Plane),
    /// The depth along a curve: an index into `Paths::centrelines`.
    Curve(u32),
}

/// What a layer paints its pixels with.
enum LayerPaint {
    /// One premultiplied colour.
    Solid([u8; 4]),
    /// A linear gradient, and per row of the layer's box from its top, Cairo's composite runs
    /// (`x0..x1`, sorted): a pixel's colour is fetched from the start of its run.
    Linear(Box<Linear>, Vec<Vec<(u32, u32)>>),
}

/// A layer's paint before its pixels are known: the gradient is fixed to the drawing's extents.
enum PaintSpec {
    Solid([u8; 4]),
    /// Stops (straight RGBA, group opacity applied) between two ends in the frame's units.
    Gradient(Vec<[f64; 4]>, [f64; 2], [f64; 2]),
}

/// Cairo's run length for a gradient source (`inplace_renderer_init`): a run of full coverage at
/// least this long is composited on its own.
const RUN_LENGTH: u32 = 256;

/// The composite runs Cairo's in-place span renderer (`_inplace_spans`) makes of one row whose
/// covered spans are `spans` (`x0, x1, full`, left to right; the gaps between them hold no
/// coverage): a long full span on its own, the rest gathered from the first pixel on, cut at a
/// gap once what was gathered is longer than a run.
fn composite_runs(spans: &[(u32, u32, bool)]) -> Vec<(u32, u32)> {
    let mut runs = Vec::new();
    let Some(&(first, _, _)) = spans.first() else {
        return runs;
    };
    let (mut x0, mut x1) = (first, first);
    let mut at = first;
    let mut row = Vec::with_capacity(2 * spans.len());
    for &(a, b, full) in spans {
        if a > at {
            row.push((at, a, Some(false)));
        }
        row.push((a, b, if full { None } else { Some(true) }));
        at = b;
    }
    // Each span: (x0, x1, None for full, Some(covered) otherwise).
    for (a, b, coverage) in row {
        if b - a > 1 {
            if b - a >= RUN_LENGTH && coverage.is_none() {
                if x1 != x0 {
                    runs.push((x0, x1));
                }
                runs.push((a, b));
                x0 = b;
            } else if coverage == Some(false) && x1 - x0 > RUN_LENGTH {
                runs.push((x0, x1));
                x0 = b;
            }
        }
        x1 = b;
    }
    if x1 != x0 {
        runs.push((x0, x1));
    }
    runs
}

/// Cairo's composite runs for every row of a layer's box ([`composite_runs`]), from the layer's
/// own coverage.
fn layer_runs(layer: &Layer) -> Vec<Vec<(u32, u32)>> {
    let mut scratch = Scratch::default();
    (layer.bbox[1]..layer.bbox[3])
        .map(|y| {
            let mut spans: Vec<(u32, u32, bool)> = Vec::new();
            let row = Rect {
                x0: layer.bbox[0],
                y0: y,
                x1: layer.bbox[2],
                y1: y + 1,
            };
            scanline::cover(
                &layer.edges,
                layer.candidates(y),
                row,
                &mut scratch,
                |_, x0, x1, area| {
                    let full = area >= 1.0;
                    match spans.last_mut() {
                        Some(last) if full && last.2 && last.1 == x0 => last.1 = x1,
                        _ => spans.push((x0, x1, full)),
                    }
                },
            );
            composite_runs(&spans)
        })
        .collect()
}

/// One coverage layer: a fill or a stroke's outline, with its paint and place.
struct Layer {
    edges: Vec<Edge>,
    /// The first tile row the layer touches, and per tile row from it, the range of its edges
    /// (indices into `band_edges`) overlapping that row.
    band0: u32,
    bands: Vec<(u32, u32)>,
    band_edges: Vec<u32>,
    /// Pixels `x0..x1` by `y0..y1`, inside the frame.
    bbox: [u32; 4],
    paint: LayerPaint,
    /// Every pixel of its paint is opaque.
    opaque: bool,
    depth: LayerDepth,
    order: u32,
    sub: u32,
}

impl Layer {
    /// The premultiplied colour the layer paints at `(x, y)`.
    fn colour(&self, x: u32, y: u32) -> [u8; 4] {
        match &self.paint {
            LayerPaint::Solid(colour) => *colour,
            LayerPaint::Linear(gradient, rows) => {
                let runs = &rows[(y - self.bbox[1]) as usize];
                let i = runs.partition_point(|r| r.0 <= x);
                let start = if i == 0 { x } else { runs[i - 1].0 };
                gradient.at(x as i64, y as i64, start as i64)
            }
        }
    }

    fn candidates(&self, y: u32) -> &[u32] {
        let band = y / TILE;
        match band.checked_sub(self.band0) {
            Some(b) if (b as usize) < self.bands.len() => {
                let (lo, hi) = self.bands[b as usize];
                &self.band_edges[lo as usize..hi as usize]
            }
            _ => &[],
        }
    }
}

/// A curve's depth, the bounds of its segments widened by how far its pixels may reach from it,
/// and how far a pixel takes the front-most segment that reaches it.
struct CurveDepth {
    line: Centreline,
    bounds: Vec<[f64; 4]>,
    reach: f64,
}

/// Paths lowered to coverage layers, binned by tile: a producer of area fragments.
pub struct Paths {
    layers: Vec<Layer>,
    centrelines: Vec<CurveDepth>,
    bins: Vec<Vec<u32>>,
    width: u32,
}

/// numpy's `isclose(a, b)` with CE's tolerance, for one coordinate.
fn close(a: f64, b: f64) -> bool {
    a == b || ((a - b).abs() <= POINT_EQUALITY + 1e-5 * b.abs() && (a - b).is_finite())
}

/// CE's subpath ranges `start..end` over the placed points, compared in two dimensions.
fn subpaths(points: &[[f64; 3]]) -> Vec<(usize, usize)> {
    let n = points.len();
    if n < 4 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    for b in (4..n).step_by(4) {
        let (end, next) = (points[b - 1], points[b]);
        if !(close(end[0], next[0]) && close(end[1], next[1])) {
            out.push((start, b));
            start = b;
        }
    }
    out.push((start, n));
    out
}

/// The outline CE hands Cairo, in the frame's units: per subpath of four points or more, a move to
/// its first point, its cubics, and a close when its ends are equal.
fn outline(shown: &[[f64; 3]]) -> BezPath {
    let mut path = BezPath::new();
    let pt = |p: [f64; 3]| Point::new(p[0], p[1]);
    for (start, end) in subpaths(shown) {
        if end - start < 4 {
            continue;
        }
        let (a, b) = (shown[start], shown[end - 1]);
        // `consider_points_equals_2d(first, last)`: tolerance relative to the last point.
        let closed = close(a[0], b[0]) && close(a[1], b[1]);
        path.move_to(pt(a));
        for i in (start..=end - 4).step_by(4) {
            // A closed subpath ends exactly at its start. Kept, the gap CE's tolerance allows
            // would add a closing segment of arbitrary direction and joins to it; Cairo, whose
            // points sit on a 1/256-pixel grid, drops it as degenerate at common sizes.
            let last = if closed && i + 4 == end {
                a
            } else {
                shown[i + 3]
            };
            path.curve_to(pt(shown[i + 1]), pt(shown[i + 2]), pt(last));
        }
        if closed {
            path.close_path();
        }
    }
    path
}

/// A path through the frame mapping (an affine map of each point).
fn mapped(path: &BezPath, map: &FrameMap) -> BezPath {
    let m = |p: Point| {
        let [x, y] = map.apply([p.x, p.y, 0.0]);
        Point::new(x, y)
    };
    path.elements()
        .iter()
        .map(|el| match *el {
            PathEl::MoveTo(p) => PathEl::MoveTo(m(p)),
            PathEl::LineTo(p) => PathEl::LineTo(m(p)),
            PathEl::QuadTo(a, b) => PathEl::QuadTo(m(a), m(b)),
            PathEl::CurveTo(a, b, c) => PathEl::CurveTo(m(a), m(b), m(c)),
            PathEl::ClosePath => PathEl::ClosePath,
        })
        .collect()
}

/// Whether a curve's control points all lie on one side of `area` (`[x0, y0, x1, y1]`) and
/// beyond it: the curve lies inside their hull, so it misses the area.
fn misses(pts: &[Point], area: [f64; 4]) -> bool {
    pts.iter().all(|p| p.x < area[0])
        || pts.iter().all(|p| p.x > area[2])
        || pts.iter().all(|p| p.y < area[1])
        || pts.iter().all(|p| p.y > area[3])
}

/// A curve as pieces in order, each handed to `piece` with whether it misses `area`: a curve whose
/// hull meets the area but is far larger is split until its pieces are near the area's size or
/// miss it.
fn cull_pieces(c: kurbo::CubicBez, area: [f64; 4], piece: &mut dyn FnMut(kurbo::CubicBez, bool)) {
    fn split(
        c: kurbo::CubicBez,
        depth: u32,
        area: [f64; 4],
        size: f64,
        piece: &mut dyn FnMut(kurbo::CubicBez, bool),
    ) {
        if misses(&[c.p0, c.p1, c.p2, c.p3], area) {
            piece(c, true);
            return;
        }
        let r = c.bounding_box();
        if depth == 0 || r.width().max(r.height()) <= 64.0 * size {
            piece(c, false);
            return;
        }
        let (a, b) = c.subdivide();
        split(a, depth - 1, area, size, piece);
        split(b, depth - 1, area, size, piece);
    }
    let size = (area[2] - area[0]).max(area[3] - area[1]);
    split(c, 40, area, size, piece);
}

/// A path whose curves outside `area` (`[x0, y0, x1, y1]`) are replaced by their chords. A curve
/// lies inside its control points' hull, and so does its chord, so where the hull misses the area,
/// nothing inside it changes ([`cull_pieces`]).
fn culled(path: &BezPath, area: [f64; 4]) -> BezPath {
    let mut out = BezPath::new();
    let mut last = Point::ZERO;
    let curve = |c: kurbo::CubicBez, out: &mut BezPath| {
        cull_pieces(c, area, &mut |piece, missed| {
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
                curve(kurbo::CubicBez::new(last, a, b, c), &mut out);
                last = c;
            }
            PathEl::ClosePath => out.close_path(),
        }
    }
    out
}

/// kurbo's accuracy for the arc lengths its dashing measures.
const DASH_ACCURACY: f64 = 1e-6;

/// A subpath's pieces, each with whether it misses the guard area.
type Pieces = Vec<(PathSeg, bool)>;

/// The dashes of a path as kurbo lays them (each subpath starting `offset` into `pattern`, a closed
/// subpath's last dash joined to its first when both are on), keeping, when `area` is given, only
/// what can reach it. A piece of a curve that misses the area ([`cull_pieces`]) is measured, not
/// dashed: the dashes after it start where the whole curve's length puts them, never where a
/// shortcut around it would.
fn dashed(path: &BezPath, offset: f64, pattern: &[f64; 2], area: Option<[f64; 4]>) -> BezPath {
    let Some(area) = area else {
        return BezPath::from_vec(dash(path, offset, pattern));
    };
    let mut out = Vec::new();
    let mut pieces = Pieces::new();
    let (mut start, mut last) = (Point::ZERO, Point::ZERO);
    for el in path.elements() {
        match *el {
            PathEl::MoveTo(p) => {
                dash_subpath(&pieces, false, offset, pattern, &mut out);
                pieces.clear();
                (start, last) = (p, p);
            }
            PathEl::LineTo(p) => {
                let line = kurbo::Line::new(last, p);
                pieces.push((PathSeg::Line(line), misses(&[last, p], area)));
                last = p;
            }
            PathEl::QuadTo(a, b) => {
                let c = kurbo::QuadBez::new(last, a, b).raise();
                cull_pieces(c, area, &mut |c, missed| {
                    pieces.push((PathSeg::Cubic(c), missed))
                });
                last = b;
            }
            PathEl::CurveTo(a, b, c) => {
                let c = kurbo::CubicBez::new(last, a, b, c);
                cull_pieces(c, area, &mut |c, missed| {
                    pieces.push((PathSeg::Cubic(c), missed))
                });
                last = c.p3;
            }
            PathEl::ClosePath => {
                if last != start {
                    let line = kurbo::Line::new(last, start);
                    pieces.push((PathSeg::Line(line), misses(&[last, start], area)));
                }
                dash_subpath(&pieces, true, offset, pattern, &mut out);
                pieces.clear();
                last = start;
            }
        }
    }
    dash_subpath(&pieces, false, offset, pattern, &mut out);
    BezPath::from_vec(out)
}

fn dash(path: &BezPath, offset: f64, pattern: &[f64; 2]) -> Vec<PathEl> {
    kurbo::dash(path.elements().iter().copied(), offset, pattern).collect()
}

/// One subpath's dashes into `out`, its pieces that miss the guard area measured and dropped.
fn dash_subpath(
    pieces: &[(PathSeg, bool)],
    closed: bool,
    offset: f64,
    pattern: &[f64; 2],
    out: &mut Vec<PathEl>,
) {
    let Some(&(first, _)) = pieces.first() else {
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
    if pieces.iter().all(|(_, missed)| !missed) {
        out.extend(dash(&run_path(pieces, closed), offset, pattern));
        return;
    }
    // Runs of pieces that may reach the area, each with how far into the subpath it starts.
    let mut runs: Vec<(f64, &[(PathSeg, bool)])> = Vec::new();
    let (mut travelled, mut open) = (0.0, None);
    for (i, (seg, missed)) in pieces.iter().enumerate() {
        match (missed, open) {
            (true, Some((at, from))) => {
                runs.push((at, &pieces[from..i]));
                open = None;
            }
            (false, None) => open = Some((travelled, i)),
            _ => {}
        }
        travelled += seg.arclen(DASH_ACCURACY);
    }
    if let Some((at, from)) = open {
        runs.push((at, &pieces[from..]));
    }
    // Each run's dashes, one subpath each (in the order kurbo gives them, its first dash last).
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
    // A closed subpath whose start lies in a run, its last run ending there and its first
    // beginning there: as kurbo does, a dash running to the end continues into a dash starting at
    // the start.
    let start = first.start();
    let near = |p: Point| (p - start).hypot() <= 1e-9 * (1.0 + start.x.abs().max(start.y.abs()));
    if closed && runs.len() > 1 && !pieces[0].1 && !pieces[pieces.len() - 1].1 {
        let opening = dashes[0].iter().position(|d| d[0] == PathEl::MoveTo(start));
        let closing = dashes[runs.len() - 1]
            .iter()
            .position(|d| d.len() > 1 && d.last().and_then(|el| el.end_point()).is_some_and(near));
        if let (Some(i), Some(j)) = (opening, closing) {
            let head = dashes[0].remove(i);
            dashes[runs.len() - 1][j].extend(head.into_iter().skip(1));
        }
    }
    out.extend(dashes.into_iter().flatten().flatten());
}

/// A segment as the path element that draws it from its start.
fn seg_el(seg: &PathSeg) -> PathEl {
    match *seg {
        PathSeg::Line(l) => PathEl::LineTo(l.p1),
        PathSeg::Quad(q) => PathEl::QuadTo(q.p1, q.p2),
        PathSeg::Cubic(c) => PathEl::CurveTo(c.p1, c.p2, c.p3),
    }
}

/// A path flattened to closed rings of pixel points.
fn rings(path: &BezPath) -> Vec<Vec<[f64; 2]>> {
    let mut rings = Vec::new();
    let mut ring: Vec<[f64; 2]> = Vec::new();
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
        PathEl::QuadTo(..) | PathEl::CurveTo(..) => unreachable!("flatten emits lines"),
    });
    if ring.len() > 2 {
        rings.push(ring);
    }
    rings
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

/// The screen polyline of a path's cubics with the depth key carried along each (the cubic of its
/// control points' keys, at the same parameter).
fn curve_depth(shown: &[[f64; 3]], map: &FrameMap, projection: &Projection) -> Centreline {
    let mut points = Vec::new();
    let mut breaks = Vec::new();
    let screen = |p: [f64; 3]| (map.apply(p), -projection.factor(p[2]));
    for (start, end) in subpaths(shown) {
        if end - start < 4 {
            continue;
        }
        breaks.push(points.len());
        points.push(screen(shown[start]));
        for i in (start..=end - 4).step_by(4) {
            let c = [i, i + 1, i + 2, i + 3].map(|k| screen(shown[k]));
            let length: f64 = (0..3)
                .map(|k| {
                    let (a, b) = (c[k].0, c[k + 1].0);
                    ((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])).sqrt()
                })
                .sum();
            let n = if length.is_finite() {
                (length / 2.0).ceil().clamp(1.0, 64.0) as usize
            } else {
                1
            };
            for s in 1..=n {
                let t = s as f64 / n as f64;
                let u = 1.0 - t;
                let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                let at = |f: &dyn Fn(usize) -> f64| {
                    w[0] * f(0) + w[1] * f(1) + w[2] * f(2) + w[3] * f(3)
                };
                points.push(([at(&|k| c[k].0[0]), at(&|k| c[k].0[1])], at(&|k| c[k].1)));
            }
        }
    }
    Centreline::new(&points, &breaks)
}

/// Lowering state shared by every path of a frame.
pub struct Lowering<'v> {
    view: &'v View,
    map: FrameMap,
    projection: Option<Projection>,
    /// The projection in the camera's axes, for pre-projected paths.
    axes: Option<Projection>,
    width: u32,
    height: u32,
    paths: Paths,
}

impl<'v> Lowering<'v> {
    /// Starts lowering for a frame of `width x height` pixels drawn from `view`.
    pub fn new(view: &'v View, width: u32, height: u32) -> Lowering<'v> {
        // Paths and path instances map through CE's Cairo context, around its frozen centre.
        let map = FrameMap::of_kind(view, RecordKind::Path, width, height);
        let projection = match &view.camera {
            Camera::ThreeD(c) => Some(Projection::new(c, view.frame_center)),
            _ => None,
        };
        Lowering {
            view,
            map,
            axes: projection.as_ref().map(Projection::in_camera_axes),
            projection,
            width,
            height,
            paths: Paths {
                layers: Vec::new(),
                centrelines: Vec::new(),
                bins: vec![Vec::new(); (tiles_across(width) * height.div_ceil(TILE)) as usize],
                width,
            },
        }
    }

    /// Lowers one path into up to three layers.
    pub fn path(&mut self, draw: &PathDraw<'_>) {
        let rec = draw.record;
        // A pre-projected path's points are its camera's placed points already (placed by
        // the identity).
        let world: Vec<[f64; 3]> = draw.points.iter().map(|p| place(&draw.place, *p)).collect();
        if world.len() < 4 || world.iter().flatten().any(|v| !v.is_finite()) {
            return; // CE draws a mobject with a non-finite point as one point: nothing.
        }
        let (placement, keyed) = if draw.pre_projected {
            let fixed = rec.depth.fixed_orientation_centre.is_some();
            let key = produced_key(
                self.axes.as_ref(),
                &self.map,
                rec.depth.space,
                fixed,
                &world,
            );
            (Placement::Flat, key)
        } else {
            placement(
                self.projection.as_ref(),
                &self.map,
                rec.depth.space,
                rec.depth.fixed_orientation_centre,
            )
        };
        let shown = pre_display(&world, self.projection.as_ref(), placement);
        let user = outline(&shown);
        if user.elements().is_empty() {
            return;
        }
        let depth = match keyed {
            Some(plane) => LayerDepth::Plane(plane),
            None => {
                // How far the path's pixels reach from its curve: half the widest stroke and the
                // pixel's antialiasing beyond it, a miter's length at most (for binning).
                let width = draw.stroke_width.max(draw.background_stroke_width);
                let half = width * self.view.cairo_line_width_multiple * self.map.xx.abs() / 2.0;
                let reach = half + 1.0;
                let bins = (half * MITER_LIMIT).max(TILE as f64) + 2.0;
                if draw.pre_projected {
                    // Placed by the camera already: keyed where it puts the points back in its axes.
                    let axes = self.axes.clone().expect("a pre-projected path keyed in 3D");
                    let back: Vec<[f64; 3]> = shown.iter().map(|q| axes.unproject(*q)).collect();
                    self.world_depth(&axes, &back, &shown, reach, bins, false)
                } else {
                    let p = self
                        .projection
                        .clone()
                        .expect("world depth needs the 3D camera");
                    self.world_depth(&p, &world, &shown, reach, bins, draw.surface)
                }
            }
        };
        let alpha = |c: [f64; 4]| [c[0], c[1], c[2], c[3] * rec.opacity];
        let scale = self.map.xx.abs().max(self.map.yy.abs());
        let area = self.guard(&user, draw.stroke_width.max(draw.background_stroke_width));
        let guarded = match area {
            Some(area) => culled(&user, area),
            None => user.clone(),
        };
        // CE's lighting (`ThreeDCamera.modified_rgbas`): a shaded path's paint becomes two stops,
        // lit at its start and end corners, between which its gradient runs.
        let corners = [0, (world.len() - 1) / 6 * 3];
        let lit = match (&self.view.camera, rec.depth.shade_in_3d) {
            _ if draw.pre_projected => None,
            (Camera::ThreeD(c), true) if c.should_apply_shading => Some(c.light_source),
            _ => None,
        };
        let ends = match (lit, draw.ends) {
            (Some(_), _) => Some(corners.map(|i| shown[i])),
            (None, Some(ends)) if draw.pre_projected => Some(ends),
            (None, Some(ends)) => {
                let shown_ends = pre_display(&ends, self.projection.as_ref(), placement);
                Some([shown_ends[0], shown_ends[1]])
            }
            (None, None) => None,
        };
        for (k, paint, width) in [
            (0, &draw.background_stroke, draw.background_stroke_width),
            (1, &draw.fill, 0.0),
            (2, &draw.stroke, draw.stroke_width),
        ] {
            let Some(rows) = paint else { continue };
            let rows = match lit {
                Some(light) => crate::shade::ce_face_stops(&world, rows, light).to_vec(),
                None => rows.clone(),
            };
            let spec = match (rows.as_slice(), ends) {
                ([one], _) => PaintSpec::Solid(premultiply(alpha(*one))),
                (stops, Some([p1, p2])) if stops.len() > 1 => PaintSpec::Gradient(
                    stops.iter().map(|c| alpha(*c)).collect(),
                    [p1[0], p1[1]],
                    [p2[0], p2[1]],
                ),
                // Rows without ends are refused before drawing.
                _ => continue,
            };
            if let PaintSpec::Solid(colour) = spec
                && colour[3] == 0
            {
                continue;
            }
            let outline_px = if k == 1 {
                mapped(&guarded, &self.map)
            } else {
                let w = width * self.view.cairo_line_width_multiple;
                if w.is_nan() || w <= 0.0 {
                    continue;
                }
                let style = kurbo::Stroke::new(w)
                    .with_join(kurbo_join(rec.paint.joint))
                    .with_caps(kurbo_cap(rec.paint.cap))
                    .with_miter_limit(MITER_LIMIT);
                // Dashes are laid on the whole outline, then culled, so culling never moves them.
                let dashes = rec.dash.filter(|d| d.duty < 1.0).map(|d| {
                    let pattern = [d.period * d.duty, d.period * (1.0 - d.duty)];
                    dashed(&user, d.phase * d.period, &pattern, area)
                });
                let stroked = kurbo::stroke(
                    dashes
                        .as_ref()
                        .unwrap_or(&guarded)
                        .elements()
                        .iter()
                        .copied(),
                    &style,
                    &kurbo::StrokeOpts::default(),
                    TOLERANCE / scale,
                );
                mapped(&stroked, &self.map)
            };
            self.push(
                scanline::edges_of(&rings(&outline_px)),
                spec,
                depth,
                draw.order,
                draw.sub + k,
            );
        }
    }

    /// When the outline reaches beyond the frame's guard band, the area its curves outside may be
    /// simplified away beyond ([`culled`], [`dashed`]): the frame widened by a stroke's reach beyond
    /// its curve and a pixel. `None` when the outline stays inside the band.
    fn guard(&self, user: &BezPath, widest: f64) -> Option<[f64; 4]> {
        let m = &self.map;
        let to_user = |px: f64, py: f64| [(px - m.x0) / m.xx, (py - m.y0) / m.yy];
        let reach = widest * self.view.cairo_line_width_multiple * MITER_LIMIT;
        let [ax, ay] = to_user(-GUARD, -GUARD);
        let [bx, by] = to_user(self.width as f64 + GUARD, self.height as f64 + GUARD);
        let area = [ax.min(bx), ay.min(by), ax.max(bx), ay.max(by)];
        let inside = user.elements().iter().all(|el| {
            let pts: &[Point] = match el {
                PathEl::MoveTo(p) | PathEl::LineTo(p) => std::slice::from_ref(p),
                PathEl::CurveTo(a, b, c) => &[*a, *b, *c][..],
                _ => &[],
            };
            pts.iter().all(|p| {
                p.x >= area[0] + reach
                    && p.x <= area[2] - reach
                    && p.y >= area[1] + reach
                    && p.y <= area[3] - reach
            })
        });
        if inside {
            return None;
        }
        let [fx0, fy0] = to_user(0.0, 0.0);
        let [fx1, fy1] = to_user(self.width as f64, self.height as f64);
        Some([
            fx0.min(fx1) - reach - 1.0,
            fy0.min(fy1) - reach - 1.0,
            fx0.max(fx1) + reach + 1.0,
            fy0.max(fy1) + reach + 1.0,
        ])
    }

    /// The depth of a path placed in world space under a 3D camera's projection `p`, `world` its
    /// points before `p` placed them at `shown`.
    fn world_depth(
        &mut self,
        p: &Projection,
        world: &[[f64; 3]],
        shown: &[[f64; 3]],
        reach: f64,
        bins: f64,
        surface: bool,
    ) -> LayerDepth {
        if world.iter().all(|q| p.keys_planar_at(*q))
            && let Some(plane) = if surface {
                WorldPlane::nearest(world)
            } else {
                WorldPlane::of_points(world)
            }
            && let Some(key) = p.key_plane(&plane, &self.map, world[0], MIN_PLANE_COS)
        {
            return LayerDepth::Plane(key);
        }
        let line = curve_depth(shown, &self.map, p);
        if line.is_empty() {
            return LayerDepth::Plane(Plane::constant(f64::NEG_INFINITY));
        }
        let bounds = line.segment_bounds(bins).collect();
        self.paths.centrelines.push(CurveDepth {
            line,
            bounds,
            reach,
        });
        LayerDepth::Curve(self.paths.centrelines.len() as u32 - 1)
    }

    fn push(&mut self, edges: Vec<Edge>, spec: PaintSpec, depth: LayerDepth, order: u32, sub: u32) {
        if edges.is_empty() {
            return;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for e in &edges {
            x0 = x0.min(e.x_min());
            x1 = x1.max(e.x_max());
            y0 = y0.min(e.y0);
            y1 = y1.max(e.y1);
        }
        let clamp = |v: f64, limit: u32| v.clamp(0.0, limit as f64) as u32;
        let bbox = [
            clamp(x0.floor(), self.width),
            clamp(y0.floor(), self.height),
            clamp(x1.ceil() + 1.0, self.width),
            clamp(y1.ceil(), self.height),
        ];
        if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
            return;
        }
        let band0 = bbox[1] / TILE;
        let band_end = (bbox[3] - 1) / TILE;
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); (band_end - band0 + 1) as usize];
        for (i, e) in edges.iter().enumerate() {
            let first = (e.y0.max(bbox[1] as f64) as u32 / TILE).max(band0);
            let last = ((e.y1.min(bbox[3] as f64) - 1e-9).max(0.0) as u32 / TILE).min(band_end);
            for band in first..=last {
                lists[(band - band0) as usize].push(i as u32);
            }
        }
        let mut bands = Vec::with_capacity(lists.len());
        let mut band_edges = Vec::new();
        for list in lists {
            let start = band_edges.len() as u32;
            band_edges.extend(list);
            bands.push((start, band_edges.len() as u32));
        }
        let solid = |colour: [u8; 4]| Layer {
            edges: Vec::new(),
            band0,
            bands: Vec::new(),
            band_edges: Vec::new(),
            bbox,
            paint: LayerPaint::Solid(colour),
            opaque: colour[3] == 255,
            depth,
            order,
            sub,
        };
        let mut layer = match spec {
            PaintSpec::Solid(colour) => solid(colour),
            PaintSpec::Gradient(stops, p1, p2) => {
                // The composite's bounded extents: the outline's box in Cairo's 24.8 fixed point,
                // rounded out, cut to the frame. Their middle fixes pixman's translation.
                let down = |v: f64| gradient::fixed_24_8(v) >> 8;
                let up = |v: f64| (gradient::fixed_24_8(v) + 255) >> 8;
                let (ex0, ey0) = (down(x0).max(0), down(y0).max(0));
                let (ex1, ey1) = (
                    up(x1).min(self.width as i32),
                    up(y1).min(self.height as i32),
                );
                let extents = [ex0, ey0, (ex1 - ex0).max(0), (ey1 - ey0).max(0)];
                let m = &self.map;
                let ctm = gradient::Matrix {
                    xx: m.xx,
                    yx: 0.0,
                    xy: 0.0,
                    yy: m.yy,
                    x0: m.x0,
                    y0: m.y0,
                };
                match gradient::source(&ctm, p1, p2, &stops, extents) {
                    gradient::Source::Solid(colour) if colour[3] == 0 => return,
                    gradient::Source::Solid(colour) => solid(colour),
                    gradient::Source::Linear(g) => {
                        let opaque = g.opaque;
                        Layer {
                            paint: LayerPaint::Linear(Box::new(g), Vec::new()),
                            opaque,
                            ..solid([0; 4])
                        }
                    }
                }
            }
        };
        layer.edges = edges;
        layer.bands = bands;
        layer.band_edges = band_edges;
        if matches!(layer.paint, LayerPaint::Linear(..)) {
            let runs = layer_runs(&layer);
            if let LayerPaint::Linear(_, rows) = &mut layer.paint {
                *rows = runs;
            }
        }
        let index = self.paths.layers.len() as u32;
        for ty in band0..=band_end {
            for tx in bbox[0] / TILE..=(bbox[2] - 1) / TILE {
                self.paths.bins[tile_index(self.width, tx * TILE, ty * TILE)].push(index);
            }
        }
        self.paths.layers.push(layer);
    }

    /// The lowered paths, ready to draw.
    pub fn finish(self) -> Paths {
        self.paths
    }
}

impl Paths {
    /// Whether nothing was lowered.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    fn depth_at(&self, depth: LayerDepth, x: u32, y: u32, near: &mut Vec<u32>) -> Plane {
        match depth {
            LayerDepth::Plane(p) => p,
            LayerDepth::Curve(i) => {
                let curve = &self.centrelines[i as usize];
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                near.clear();
                near.extend(
                    curve
                        .bounds
                        .iter()
                        .enumerate()
                        .filter(|(_, b)| px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3])
                        .map(|(k, _)| k as u32),
                );
                curve.line.plane_at(x, y, near, curve.reach)
            }
        }
    }

    /// Rasterizes the opaque or the other layers of the tile at `rect`.
    fn raster(&self, rect: Rect, opaque: bool, mut add: impl FnMut(u32, u32, Fragment)) {
        let mut scratch = Scratch::default();
        let mut near = Vec::new();
        for &i in &self.bins[tile_index(self.width, rect.x0, rect.y0)] {
            let layer = &self.layers[i as usize];
            if layer.opaque != opaque {
                continue;
            }
            let area = Rect {
                x0: rect.x0.max(layer.bbox[0]),
                y0: rect.y0.max(layer.bbox[1]),
                x1: rect.x1.min(layer.bbox[2]),
                y1: rect.y1.min(layer.bbox[3]),
            };
            if area.x0 >= area.x1 || area.y0 >= area.y1 {
                continue;
            }
            let candidates = layer.candidates(area.y0);
            scanline::cover(
                &layer.edges,
                candidates,
                area,
                &mut scratch,
                |y, x0, x1, a| {
                    for x in x0..x1 {
                        let depth = self.depth_at(layer.depth, x, y, &mut near);
                        add(
                            x,
                            y,
                            Fragment::new(
                                depth,
                                layer.colour(x, y),
                                Coverage::Area {
                                    area: a as f32,
                                    id: i,
                                },
                                layer.order,
                                layer.sub,
                            ),
                        );
                    }
                },
            );
        }
    }
}

impl Producer for Paths {
    fn opaque(&self, pass: &mut OpaquePass<'_>) {
        self.raster(pass.rect(), true, |x, y, f| pass.add(x, y, f));
    }

    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        self.raster(pass.rect(), false, |x, y, f| pass.add(x, y, f));
    }

    fn mask(&self, id: u32, x: u32, y: u32) -> u16 {
        let layer = &self.layers[id as usize];
        scanline::mask(&layer.edges, layer.candidates(y), x, y)
    }

    fn cell_areas(&self, id: u32, x: u32, y: u32) -> [f32; SAMPLES] {
        let layer = &self.layers[id as usize];
        let cells = scanline::cell_areas(&layer.edges, layer.candidates(y), x, y);
        std::array::from_fn(|s| {
            let [column, row] = sample_cell(s);
            cells[row][column] as f32
        })
    }
}
