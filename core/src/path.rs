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
//!   between the record's ends (CE's: its corners, unless its mobject gives others), or its
//!   corners when it has none.
//! - Depth: a path in world space under a 3D camera fills at the depth of its plane when it has one
//!   the camera keys finitely (`depth::WorldPlane`, made canonical, so text and axes on one plane,
//!   and a mesh on it, tie exactly), however nearly edge-on; it strokes, and a path without such a
//!   plane (a curve in space) fills too, at the depth of the part of its own curve under each
//!   sub-sample (`depth::Centreline::key_under`: the front-most part whose stroke body covers it,
//!   else the nearest point). A plane is the fill's depth only: outside the outline, where a stroke
//!   reaches, it is not the face. Against what lies on its own registered plane, whichever
//!   producer drew it, a path's layers keep paint order where their keys lie within the snap's
//!   departure of one plane (`order::coplanar_tied`), so things on one plane tie; there each layer
//!   is read on its path's plane at the sub-sample (`Producer::plane_depths`,
//!   `composite::Depths::against`), so a stroke is where its face is, its body off the curve on
//!   that plane, and a fill beyond its outline where the sub-sample lies, and against all else a
//!   stroke on its curve and a fill at its outline (`fringe_depths`); and a path's fill
//!   and its own strokes keep paint order between them (`order::own_layers`), so a stroke shows
//!   over its fill wherever their depths part. A
//!   fixed-orientation path is a camera-facing plane at its centre's depth; anything else is in
//!   front of the scene, tying by draw order.
//! - Pre-projected (XF 1.1's tail): the points, paints and gradient ends a camera of the scene's
//!   own produced, drawn as they are, never projected or lit again. Depth follows the rules above
//!   in that camera's axes, the points put back where its projection took them from
//!   (`Projection::in_camera_axes`, `unproject`), so the view's angles, which that camera may not
//!   have drawn with, never reach it.
//! - Coverage: each pixel's exact nonzero area (`scanline::cover`); a pixel the compositor resolves
//!   sample by sample asks for the layer's winding at the sub-samples (`scanline::mask`).

use crate::composite::{
    Coverage, Fragment, OpaquePass, Plane, Producer, Rect, SAMPLES, TransparentPass, premultiply,
    sample_cell, sample_point,
};
use crate::depth::{Centreline, Planarity, Registry, WorldPlane};
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
    /// The placements `place` is made of, outer first, as the stream carries them (the identity
    /// for none): an instance's placement and its record's. What the snap of each may have moved
    /// the points by bounds how far a path content made coplanar with another may key apart from it
    /// ([`crate::depth::snap_departure`]).
    pub places: [Affine; 2],
    pub fill: Option<Vec<[f64; 4]>>,
    pub stroke: Option<Vec<[f64; 4]>>,
    pub background_stroke: Option<Vec<[f64; 4]>>,
    /// A gradient's ends in world space, as `get_gradient_start_and_end_points` gives them: where
    /// a paint of two or more rows runs between, and the two stops CE's lighting makes of a shaded
    /// path's paint (which, without them, run between its start and end corners).
    pub ends: Option<[[f64; 3]; 2]>,
    /// A face of a surface (a face group's): an outline on no one plane takes the plane nearest it
    /// ([`WorldPlane::nearest`]), not its curve's depth.
    pub surface: bool,
    /// The points are where a camera of the scene's own placed them (a pre-projected record's
    /// tail), and the paints and `ends` are what it produced: nothing is projected or lit again,
    /// and depth is keyed in that camera's axes ([`crate::project::produced_key`]).
    pub pre_projected: bool,
    /// Where that camera placed the record's fixed-orientation centre, when it has one
    /// ([`crate::project::produced_pivot`]).
    pub pivot: Option<[f64; 3]>,
    /// The stroke widths drawn, in CE's units: the record's, or a paint blend's mix of them.
    pub stroke_width: f64,
    pub background_stroke_width: f64,
    pub order: u32,
    /// The first of the three sub-orders the path's layers take (background stroke, fill, stroke).
    pub sub: u32,
    /// Which of the paths its record draws this is: its place in the record's `planes`.
    pub which: usize,
}

impl PathDraw<'_> {
    /// The normals its lit corners are lit by, as its record's writer decided before the snap, if
    /// it did.
    fn decided_normals(&self) -> Option<[[f64; 3]; 2]> {
        let normals = self.record.normals.as_ref()?;
        Some([
            *normals.get(2 * self.which)?,
            *normals.get(2 * self.which + 1)?,
        ])
    }

    /// How the path lies, as its record's writer decided before the snap, if it did.
    fn decided(&self) -> Option<Planarity> {
        let code = *self.record.planes.as_ref()?.get(self.which)?;
        Planarity::of_code(code)
    }
}

/// A shaded path's two stops per layer (background stroke, fill, stroke), `None` for a layer
/// without paint ([`Lowering::face_stops`]).
pub type FaceStops = [Option<[[f64; 4]; 2]>; 3];

/// `draw`'s points placed in the world, or `None` when the drawer draws nothing of it: fewer than
/// four points, or one not finite (CE draws a mobject with a non-finite point as one point:
/// nothing). A pre-projected path's points are its camera's placed points already (placed by the
/// identity).
fn drawn_world(draw: &PathDraw<'_>) -> Option<Vec<[f64; 3]>> {
    let world: Vec<[f64; 3]> = draw.points.iter().map(|p| place(&draw.place, *p)).collect();
    if world.len() < 4 || world.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    Some(world)
}

/// Where a layer's fragments sit in depth.
#[derive(Clone, Copy, Debug)]
enum LayerDepth {
    Plane(Plane),
    /// The depth along a curve: an index into `Paths::centrelines`, and the half width in pixels of
    /// the layer's stroke (`0` for a fill): how far from the curve the layer's body reaches.
    Curve(u32, f64),
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
    /// The outline's horizontal segments, which `edges` leaves out ([`scanline::edges_of`]), as
    /// their height and their two ends' x: with `edges`, the whole outline, which a layer keyed on
    /// a sloped plane reads its fringe's depth on ([`fringe_depths`]). Empty for others.
    flats: Vec<[f64; 3]>,
    /// Pixels `x0..x1` by `y0..y1`, inside the frame.
    bbox: [u32; 4],
    paint: LayerPaint,
    /// Every pixel of its paint is opaque.
    opaque: bool,
    depth: LayerDepth,
    /// The registered plane it lies on ([`Fragment::plane`]), `0` for none.
    plane: u32,
    /// How far its key may stand from the plane content put it on ([`Fragment::snap`]): finite
    /// for a layer of a path on a registered plane.
    snap: f64,
    /// For a stroke of a path keyed on a registered plane, its path's key there: what it is read on
    /// against a fragment on that plane ([`Producer::plane_depths`]).
    on_plane: Option<Plane>,
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
        self.band_candidates(y / TILE)
    }

    /// The edges overlapping tile row `band`.
    fn band_candidates(&self, band: u32) -> &[u32] {
        match band.checked_sub(self.band0) {
            Some(b) if (b as usize) < self.bands.len() => {
                let (lo, hi) = self.bands[b as usize];
                &self.band_edges[lo as usize..hi as usize]
            }
            _ => &[],
        }
    }

    /// The point of the layer's outline nearest `p`, a point of pixel `(_, y)`, among its edges
    /// overlapping the rows from `y - 2` to `y + 2` and its horizontal segments; `p` itself when
    /// none is there. Where the outline crosses the pixel, no point of the pixel is farther from
    /// it than the pixel's diagonal (1.42 px), so the nearest of the whole outline is among those.
    fn nearest_on_outline(&self, p: [f64; 2], y: u32) -> [f64; 2] {
        // The nearest point so far, its squared distance and its distance.
        let mut best = (p, f64::INFINITY, f64::INFINITY);
        let mut segment = |a: [f64; 2], b: [f64; 2]| {
            let reach = best.2;
            if a[1].max(b[1]) < p[1] - reach
                || a[1].min(b[1]) > p[1] + reach
                || a[0].max(b[0]) < p[0] - reach
                || a[0].min(b[0]) > p[0] + reach
            {
                return;
            }
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let length = dx * dx + dy * dy;
            let t = if length > 0.0 {
                (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = [a[0] + t * dx, a[1] + t * dy];
            let (ex, ey) = (q[0] - p[0], q[1] - p[1]);
            let d2 = ex * ex + ey * ey;
            if d2 < best.1 {
                best = (q, d2, d2.sqrt());
            }
        };
        for band in y.saturating_sub(2) / TILE..=(y + 2) / TILE {
            for &i in self.band_candidates(band) {
                let e = &self.edges[i as usize];
                segment([e.x0, e.y0], [e.x1, e.y1]);
            }
        }
        for &[fy, x0, x1] in &self.flats {
            segment([x0, fy], [x1, fy]);
        }
        best.0
    }
}

/// The horizontal segments of closed polygons' outlines (each ring closes back to its first
/// point), as their height and their two ends' x, least first: the ones [`scanline::edges_of`]
/// leaves out of the same rings, which it leaves out whole as this does.
fn flats_of(rings: &[Vec<[f64; 2]>]) -> Vec<[f64; 3]> {
    let mut flats = Vec::new();
    for ring in rings {
        if ring.len() < 2 || ring.iter().flatten().any(|v| !v.is_finite()) {
            continue;
        }
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            if a[1] == b[1] {
                flats.push([a[1], a[0].min(b[0]), a[0].max(b[0])]);
            }
        }
    }
    flats
}

/// A curve's depth, the bounds of its segments widened by `margin`, how far its pixels may reach
/// from it, and which segments' bounds meet each tile of the frame.
struct CurveDepth {
    line: Centreline,
    bounds: Vec<[f64; 4]>,
    margin: f64,
    /// The frame's tiles across and down.
    across: u32,
    down: u32,
    /// The segments whose bounds meet tile `t` ([`tile_index`]), in order, are
    /// `tiled[starts[t]..starts[t + 1]]`.
    starts: Vec<u32>,
    tiled: Vec<u32>,
    /// The segments whose bounds meet no tile of the frame, in order.
    beyond: Vec<u32>,
    /// The segments filed by the cells of a fine grid over their own bounds.
    fine: Fine,
}

/// The cell size, in pixels, of a curve's fine grid ([`Fine`]).
const FINE: f64 = 8.0;

/// A curve's segments filed, in order, under the cells of a grid `FINE` pixels square over their
/// own screen bounds (not widened), clipped to `region`, the frame widened by the margin and a
/// pixel: where a pixel's read finds the few segments within its reach without testing every
/// segment whose widened bounds hold it ([`CurveDepth::near_cut`]).
#[derive(Default)]
struct Fine {
    origin: [f64; 2],
    /// Cells across and down; none where no segment meets the region.
    cols: i64,
    rows: i64,
    /// The segments filed under cell `(c, r)`, in order, are
    /// `filed[starts[r * cols + c]..starts[r * cols + c + 1]]`.
    starts: Vec<u32>,
    filed: Vec<u32>,
}

impl Fine {
    fn new(line: &Centreline, region: [f64; 4]) -> Fine {
        let mut all = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for i in 0..line.len() {
            let b = line.segment_box(i);
            all = [
                all[0].min(b[0]),
                all[1].min(b[1]),
                all[2].max(b[2]),
                all[3].max(b[3]),
            ];
        }
        let (lo, hi) = (
            [all[0].max(region[0]), all[1].max(region[1])],
            [all[2].min(region[2]), all[3].min(region[3])],
        );
        if !(lo[0] <= hi[0] && lo[1] <= hi[1]) {
            return Fine::default();
        }
        let origin = [lo[0].floor(), lo[1].floor()];
        let at = |v: f64, axis: usize| ((v - origin[axis]) / FINE).floor() as i64;
        let (cols, rows) = (at(hi[0], 0) + 1, at(hi[1], 1) + 1);
        // The cells a segment's bounds meet within the region, none for one beyond it.
        let spans: Vec<Option<[i64; 4]>> = (0..line.len())
            .map(|i| {
                let b = line.segment_box(i);
                (b[2] >= lo[0] && b[0] <= hi[0] && b[3] >= lo[1] && b[1] <= hi[1]).then(|| {
                    [
                        at(b[0].max(lo[0]), 0).clamp(0, cols - 1),
                        at(b[2].min(hi[0]), 0).clamp(0, cols - 1),
                        at(b[1].max(lo[1]), 1).clamp(0, rows - 1),
                        at(b[3].min(hi[1]), 1).clamp(0, rows - 1),
                    ]
                })
            })
            .collect();
        let cells = |s: [i64; 4]| {
            (s[2]..=s[3]).flat_map(move |r| (s[0]..=s[1]).map(move |c| (r * cols + c) as usize))
        };
        let mut starts = vec![0u32; (cols * rows) as usize + 1];
        for t in spans.iter().flatten().flat_map(|s| cells(*s)) {
            starts[t + 1] += 1;
        }
        for t in 1..starts.len() {
            starts[t] += starts[t - 1];
        }
        let mut next = starts.clone();
        let mut filed = vec![0u32; starts[starts.len() - 1] as usize];
        for (k, s) in spans.iter().enumerate() {
            for t in s.iter().flat_map(|s| cells(*s)) {
                filed[next[t] as usize] = k as u32;
                next[t] += 1;
            }
        }
        Fine {
            origin,
            cols,
            rows,
            starts,
            filed,
        }
    }

    /// The cell holding coordinate `v` along `axis`, unclamped.
    fn at(&self, v: f64, axis: usize) -> i64 {
        ((v - self.origin[axis]) / FINE).floor() as i64
    }

    /// The segments filed under cell `(c, r)`, which lies in the grid.
    fn cell(&self, c: i64, r: i64) -> &[u32] {
        let t = (r * self.cols + c) as usize;
        &self.filed[self.starts[t] as usize..self.starts[t + 1] as usize]
    }
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

/// How far `key`, the depth key across the plane nearest a path's points (they lie on no one
/// plane: [`Planarity::Nearest`]), may stand from the depth keys of the path's drawn outline
/// wherever its fill is read: inside its outline, and at the outline's points nearest its fringe.
/// `shown` are the path's points as the camera placed them, under `p` and the frame mapping `map`.
///
/// Along each cubic the outline is keyed as its curve is ([`curve_depth`]): position and key are
/// both the cubic through its control points' positions and keys. `key` is affine on the screen,
/// so read along the cubic it is the cubic through its reads at the control points, and its
/// departure from the outline's key is the cubic through the control points' departures: within
/// their largest (a Bernstein polynomial lies within its coefficients), between the anchors as at
/// them. A closed subpath's last control point is drawn at its first point's position
/// ([`outline`]) and departs from there too. The fill's outline is that curve flattened to within
/// [`TOLERANCE`] pixels, so each corner of its polygon lies that near a point of the curve, where
/// `key` moves by at most its slope times that; and every point the fill is read at (a
/// sub-sample inside the polygon, or the polygon's point nearest one outside it) lies in the
/// polygon's hull, where an affine key lies between its values at the corners. So each read lies
/// within this bound of the outline's range of keys. Each departure's own rounding (the key read
/// at the point, the point's key, their difference) is added.
fn nearest_plane_departure(key: &Plane, shown: &[[f64; 3]], map: &FrameMap, p: &Projection) -> f64 {
    use crate::composite::U;
    // The departure of `key` at the position of `at` from the key of `keyed`.
    let departure = |at: [f64; 3], keyed: [f64; 3]| {
        let [x, y] = map.apply(at);
        let k = -p.factor(keyed[2]);
        let gap = (key.at(x, y) - k).abs();
        gap + key.read_error(x, y) + p.key_error(keyed) + 4.0 * U * k.abs() + U * gap
    };
    let mut worst = 0.0f64;
    for (start, end) in subpaths(shown) {
        if end - start < 4 {
            continue;
        }
        for q in &shown[start..end] {
            worst = worst.max(departure(*q, *q));
        }
        let (a, b) = (shown[start], shown[end - 1]);
        if close(a[0], b[0]) && close(a[1], b[1]) {
            worst = worst.max(departure(a, b));
        }
    }
    worst + libm::hypot(key.b, key.c) * TOLERANCE
}

/// The screen polyline of a path's cubics with the depth key carried along each (the cubic of its
/// control points' keys, at the same parameter), each key with its error: the greatest of its
/// control points' ([`Projection::key_error`]) and the rounding of the cubic's weighted sum.
fn curve_depth(shown: &[[f64; 3]], map: &FrameMap, projection: &Projection) -> Centreline {
    let mut points = Vec::new();
    let mut errors = Vec::new();
    let mut breaks = Vec::new();
    let screen = |p: [f64; 3]| (map.apply(p), -projection.factor(p[2]));
    for (start, end) in subpaths(shown) {
        if end - start < 4 {
            continue;
        }
        breaks.push(points.len());
        points.push(screen(shown[start]));
        errors.push(projection.key_error(shown[start]));
        for i in (start..=end - 4).step_by(4) {
            let c = [i, i + 1, i + 2, i + 3].map(|k| screen(shown[k]));
            let err = [i, i + 1, i + 2, i + 3]
                .iter()
                .map(|k| projection.key_error(shown[*k]))
                .fold(0.0, f64::max)
                + 4.0 * crate::composite::U * c.iter().map(|p| p.1.abs()).fold(0.0, f64::max);
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
                errors.push(err);
            }
        }
    }
    Centreline::with_errors(&points, &errors, &breaks)
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

    /// How `draw`'s points lie, decided from the points as given ([`WorldPlane::decide`]): the
    /// points its depth is keyed from, its placed points, or a pre-projected path's put back in
    /// its camera's axes. `Curve` for a path [`Lowering::path`] does not draw.
    pub fn planarity(&self, draw: &PathDraw<'_>) -> Planarity {
        let world: Vec<[f64; 3]> = draw.points.iter().map(|p| place(&draw.place, *p)).collect();
        if world.len() < 4 || world.iter().flatten().any(|v| !v.is_finite()) {
            return Planarity::Curve;
        }
        if draw.pre_projected {
            let Some(axes) = self.axes.as_ref() else {
                return Planarity::Curve;
            };
            let shown = pre_display(&world, self.projection.as_ref(), Placement::Flat);
            let back: Vec<[f64; 3]> = shown.iter().map(|q| axes.unproject(*q)).collect();
            WorldPlane::decide(&back, false)
        } else {
            WorldPlane::decide(&world, draw.surface)
        }
    }

    /// The normals `draw`'s lit corners are lit by, decided from its points as a writer that snaps
    /// them to `grid` carries them ([`crate::shade::decided_normals`]); `None` for a path CE's
    /// lighting does not shade (not `shade_in_3d`, a camera that does not shade, or pre-projected).
    pub fn normals(&self, draw: &PathDraw<'_>, grid: f64) -> Option<[[f64; 3]; 2]> {
        let shades = matches!(&self.view.camera, Camera::ThreeD(c) if c.should_apply_shading);
        if draw.pre_projected || !shades || !draw.record.depth.shade_in_3d {
            return None;
        }
        let world: Vec<[f64; 3]> = draw.points.iter().map(|p| place(&draw.place, *p)).collect();
        Some(crate::shade::decided_normals(&world, grid))
    }

    /// Which of `draw`'s two lit corners the drawer lights by its face's Newell normal: those whose
    /// normal the record's writer decided degenerate (the zero vector) where the Newell normal does
    /// not cancel ([`crate::shade::ce_face_stops_decided`]). `None` for a path CE's lighting does
    /// not shade; both `false` for one whose record carries no normals.
    pub fn newell_corners(&self, draw: &PathDraw<'_>) -> Option<[bool; 2]> {
        let shades = matches!(&self.view.camera, Camera::ThreeD(c) if c.should_apply_shading);
        if draw.pre_projected || !shades || !draw.record.depth.shade_in_3d {
            return None;
        }
        let Some(decided) = draw.decided_normals() else {
            return Some([false; 2]);
        };
        let world: Vec<[f64; 3]> = draw.points.iter().map(|p| place(&draw.place, *p)).collect();
        let cancels = match world.first() {
            Some(first) => {
                let n = crate::depth::newell(&world, *first);
                n[0] * n[0] + n[1] * n[1] + n[2] * n[2] == 0.0
            }
            None => true,
        };
        Some(decided.map(|n| n == [0.0; 3] && !cancels))
    }

    /// The two stops each layer of `draw` (background stroke, fill, stroke) is painted with, as
    /// [`Lowering::path`] paints them ([`Lowering::layer_rows`]): lit at its start and end
    /// corners, straight RGBA, the record's group opacity applied, unclamped; `None` for a layer
    /// without paint. `None` for a path CE's lighting does not shade (not `shade_in_3d`, a camera
    /// that does not shade, or pre-projected) and for one the drawer draws nothing of.
    pub fn face_stops(&self, draw: &PathDraw<'_>) -> Option<FaceStops> {
        let light = self.light(draw)?;
        let world = drawn_world(draw)?;
        let layers = self.layer_rows(draw, &world, Some(light));
        Some(layers.map(|rows| rows.and_then(|r| r.try_into().ok())))
    }

    /// The points `draw`'s depth is keyed from, and whether they are placed in the world by the
    /// 3D camera's projection (not fixed in frame, fixed in orientation, or pre-projected);
    /// `None` for a path [`Lowering::path`] does not draw or keys whole.
    fn depth_points(&self, draw: &PathDraw<'_>) -> Option<(Vec<[f64; 3]>, bool)> {
        let world: Vec<[f64; 3]> = draw.points.iter().map(|p| place(&draw.place, *p)).collect();
        if world.len() < 4 || world.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        if draw.pre_projected {
            let axes = self.axes.as_ref()?;
            let keyed = produced_key(Some(axes), &self.map, draw.record.depth.space, draw.pivot);
            let shown = pre_display(&world, self.projection.as_ref(), Placement::Flat);
            let back = shown.iter().map(|q| axes.unproject(*q)).collect();
            return keyed.is_none().then_some((back, false));
        }
        let (placed, keyed) = placement(
            self.projection.as_ref(),
            &self.map,
            draw.record.depth.space,
            draw.record.depth.fixed_orientation_centre,
        );
        (keyed.is_none() && matches!(placed, Placement::Projected)).then_some((world, true))
    }

    /// The plane each of `draws` (in draw order) is keyed on, its identity in the frame's
    /// `registry` (`0`: not registered) and how the path's own points lie, `None` for its curve's
    /// depth.
    ///
    /// A flat path (its record's `planes`, or decided from its points) is fitted its plane, then
    /// merged in the frame's [`Registry`] in draw order, after its meshes' triangles: a path on a
    /// plane registered before it takes that plane's identity, else registers its own, and is
    /// keyed on its own plane either way. Then a path with no plane of
    /// its own (a line, a curve) whose points all lie on a registered plane takes the first such
    /// plane, whatever the order. So things on one plane tie by draw order, though each plane fitted
    /// to snapped points differs by the snap's noise. A surface's face on no one plane, and a
    /// pre-projected path, keep the plane fitted to their own points.
    pub fn coplanar(
        &self,
        draws: &[PathDraw<'_>],
        registry: &mut Registry,
    ) -> Vec<Option<(WorldPlane, u32, Planarity)>> {
        let mut found = vec![None; draws.len()];
        let mut curves = Vec::new();
        for (i, draw) in draws.iter().enumerate() {
            let Some((points, in_world)) = self.depth_points(draw) else {
                continue;
            };
            let planarity = draw
                .decided()
                .unwrap_or_else(|| WorldPlane::decide(&points, draw.surface && in_world));
            // A path any of whose points the camera does not key on a plane keeps its curve's
            // depth ([`Lowering::world_depth`]): it neither registers a plane nor takes one.
            if in_world
                && let Some(p) = self.projection.as_ref()
                && !points.iter().all(|q| p.keys_planar_at(*q))
            {
                continue;
            }
            let own = WorldPlane::fit(&points, planarity);
            found[i] = match (own, planarity, in_world) {
                (Some(own), Planarity::Newell | Planarity::Widest | Planarity::Axis(_), true) => {
                    Some((own, registry.merge(own, &points), planarity))
                }
                (None, Planarity::Curve, true) => {
                    curves.push((i, points));
                    None
                }
                (own, _, _) => own.map(|own| (own, 0, planarity)),
            };
        }
        for (i, points) in curves {
            found[i] = registry
                .containing(&points)
                .map(|(plane, id)| (plane, id, Planarity::Curve));
        }
        found
    }

    /// Lowers one path into up to three layers, `plane` the plane it is keyed on
    /// ([`Lowering::coplanar`]), or `None` for its curve's depth.
    pub fn path(&mut self, draw: &PathDraw<'_>, plane: Option<(WorldPlane, u32, Planarity)>) {
        let rec = draw.record;
        let Some(world) = drawn_world(draw) else {
            return;
        };
        let (placement, keyed) = if draw.pre_projected {
            let key = produced_key(self.axes.as_ref(), &self.map, rec.depth.space, draw.pivot);
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
        // The strokes' depth and the fill's.
        let (strokes, fill, plane, fill_snap) = match keyed {
            Some(plane) => (
                LayerDepth::Plane(plane),
                LayerDepth::Plane(plane),
                0,
                f64::INFINITY,
            ),
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
                    self.world_depth(&axes, &back, &shown, reach, bins, plane, None)
                } else {
                    let p = self
                        .projection
                        .clone()
                        .expect("world depth needs the 3D camera");
                    let placed = (&draw.places, &draw.points[..]);
                    self.world_depth(&p, &world, &shown, reach, bins, plane, Some(placed))
                }
            }
        };
        let scale = self.map.xx.abs().max(self.map.yy.abs());
        let area = self.guard(&user, draw.stroke_width.max(draw.background_stroke_width));
        let guarded = match area {
            Some(area) => culled(&user, area),
            None => user.clone(),
        };
        // CE's lighting (`ThreeDCamera.modified_rgbas`): a shaded path's paint becomes two stops,
        // lit at its start and end corners. Its gradient runs, as any path's does, between the
        // ends the mobject gives (`get_gradient_start_and_end_points`, the record's): for a shaded
        // path its corners, unless it gives others, as `ThreeDAxes` gives every piece of an axis
        // the axis's own ends. A shaded path with no ends recorded (a face of a surface, an
        // instance) runs between its corners.
        let corners = [0, (world.len() - 1) / 6 * 3];
        let lit = self.light(draw);
        let ends = match (draw.ends, lit) {
            (Some(ends), _) if draw.pre_projected => Some(ends),
            (Some(ends), _) => {
                let shown_ends = pre_display(&ends, self.projection.as_ref(), placement);
                Some([shown_ends[0], shown_ends[1]])
            }
            (None, Some(_)) => Some(corners.map(|i| shown[i])),
            (None, None) => None,
        };
        let [background_rows, fill_rows, stroke_rows] = self.layer_rows(draw, &world, lit);
        for (k, rows, width) in [
            (0, background_rows, draw.background_stroke_width),
            (1, fill_rows, 0.0),
            (2, stroke_rows, draw.stroke_width),
        ] {
            let Some(rows) = rows else { continue };
            let spec = match (rows.as_slice(), ends) {
                ([one], _) => PaintSpec::Solid(premultiply(*one)),
                (stops, Some([p1, p2])) if stops.len() > 1 => {
                    PaintSpec::Gradient(stops.to_vec(), [p1[0], p1[1]], [p2[0], p2[1]])
                }
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
            // A stroke's body reaches half its width from the curve; a fill has none.
            let depth = match if k == 1 { fill } else { strokes } {
                LayerDepth::Curve(i, _) => LayerDepth::Curve(
                    i,
                    width * self.view.cairo_line_width_multiple * self.map.xx.abs() / 2.0,
                ),
                plane => plane,
            };
            // A layer of a path on a registered plane is read on its path's plane at each
            // sub-sample, with its path's term, against what shares that plane
            // ([`Producer::plane_depths`]): a stroke, keyed on its curve, and the fill, whose own key
            // beyond its outline is the plane's at the outline's nearest point ([`fringe_depths`]).
            let on_plane = match fill {
                LayerDepth::Plane(key) if plane != 0 => Some(key),
                _ => None,
            };
            let snap = if k == 1 || on_plane.is_some() {
                fill_snap
            } else {
                f64::INFINITY
            };
            self.push(
                &rings(&outline_px),
                spec,
                depth,
                (plane, snap, on_plane),
                draw.order,
                draw.sub + k,
            );
        }
    }

    /// The light CE's lighting (`ThreeDCamera.modified_rgbas`) shades `draw` from: its 3D
    /// camera's light source, when the path is `shade_in_3d` and the camera shades; `None` for
    /// any other path, and for a pre-projected one, whose camera lit its paints already.
    fn light(&self, draw: &PathDraw<'_>) -> Option<[f64; 3]> {
        match (&self.view.camera, draw.record.depth.shade_in_3d) {
            _ if draw.pre_projected => None,
            (Camera::ThreeD(c), true) if c.should_apply_shading => Some(c.light_source),
            _ => None,
        }
    }

    /// The rows each layer of `draw` (background stroke, fill, stroke) is painted with, straight
    /// RGBA with the record's group opacity applied; `None` for a layer without paint. Lit from
    /// `light` ([`Lowering::light`]), a layer's rows are CE's two stops, lit at the start and end
    /// corners of `world` (its placed points) by the normals its writer decided when it carries
    /// them ([`crate::shade::ce_face_stops_decided`]), else by CE's own
    /// ([`crate::shade::ce_face_stops`]). The drawer ([`Lowering::path`]) and
    /// [`Lowering::face_stops`] both take a path's paint from here.
    fn layer_rows(
        &self,
        draw: &PathDraw<'_>,
        world: &[[f64; 3]],
        light: Option<[f64; 3]>,
    ) -> [Option<Vec<[f64; 4]>>; 3] {
        let opacity = draw.record.opacity;
        [&draw.background_stroke, &draw.fill, &draw.stroke].map(|paint| {
            let rows = paint.as_ref()?;
            let rows = match light {
                Some(light) => match draw.decided_normals() {
                    Some(normals) => {
                        crate::shade::ce_face_stops_decided(world, rows, light, normals).to_vec()
                    }
                    None => crate::shade::ce_face_stops(world, rows, light).to_vec(),
                },
                None => rows.clone(),
            };
            Some(
                rows.into_iter()
                    .map(|c| [c[0], c[1], c[2], c[3] * opacity])
                    .collect(),
            )
        })
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

    /// The depths of a path placed in world space under a 3D camera's projection `p`, `world` its
    /// points before `p` placed them at `shown`, for its strokes and for its fill, and the plane
    /// they lie on: the strokes along its curve; the fill on `plane` ([`Lowering::coplanar`])
    /// wherever the camera keys the plane finitely, however nearly edge-on, and then the plane's
    /// identity among the frame's registered planes ([`Fragment::plane`]), else along its curve
    /// too, on no plane (`0`). A fill on the plane nearest points on no one plane
    /// ([`Planarity::Nearest`]) carries in its key's bound how far its drawn outline's keys depart
    /// from that plane's ([`nearest_plane_departure`]). A fill with an identity carries how far its
    /// key may stand from the plane content put it on ([`Fragment::snap`]): for a path content made
    /// flat (its planarity, as the writer decided it before the snap: Newell's, the widest
    /// triangle's or an axis plane) the snap's departure, from `placed` (its placements and its
    /// shape's points: [`crate::depth::snap_departure`]; in a frame not snapped, the rounding of the
    /// core's own placing alone), and its points' departure from its plane
    /// ([`crate::depth::departure`]): within its points' hull its plane stands off the plane content
    /// put them on by no more than their sum, as the gap between two planes is affine there. The
    /// rounding of its plane's coefficients is in its key's own bound, as it happened
    /// ([`Projection::key_plane`]). A path content did not make flat (keyed on a plane its points
    /// lie near) has no plane content put it on: zero. A path the camera placed itself (`placed`
    /// none) has no bound.
    #[allow(clippy::too_many_arguments)]
    fn world_depth(
        &mut self,
        p: &Projection,
        world: &[[f64; 3]],
        shown: &[[f64; 3]],
        reach: f64,
        bins: f64,
        plane: Option<(WorldPlane, u32, Planarity)>,
        placed: Option<(&[Affine; 2], &[[f64; 3]])>,
    ) -> (LayerDepth, LayerDepth, u32, f64) {
        let grid = crate::depth::input_grid(self.view);
        let fill = plane
            .filter(|_| world.iter().all(|q| p.keys_planar_at(*q)))
            .and_then(|(plane, id, planarity)| {
                let region = self.region(shown, reach);
                let mut key = p.key_plane(&plane, &self.map, world[0], 0.0, region)?;
                if planarity == Planarity::Nearest {
                    key.err += nearest_plane_departure(&key, shown, &self.map, p);
                }
                let snap = match placed {
                    Some((places, points)) if id != 0 => {
                        let off = match planarity {
                            Planarity::Newell | Planarity::Widest | Planarity::Axis(_) => {
                                let normal = plane.unrounded.normal;
                                crate::depth::snap_departure(grid, normal, places, points)
                                    + crate::depth::departure(&plane, world)
                            }
                            _ => 0.0,
                        };
                        p.key_reach(&plane, &self.map, off)
                    }
                    _ => f64::INFINITY,
                };
                Some((key, id, snap))
            });
        let registered = fill.map_or(0, |(_, id, _)| id);
        let snap = fill.map_or(f64::INFINITY, |(_, _, snap)| snap);
        let line = curve_depth(shown, &self.map, p);
        let curve = if line.is_empty() {
            LayerDepth::Plane(Plane::constant(f64::NEG_INFINITY))
        } else {
            let depth = CurveDepth::new(line, bins, self.width, self.height);
            self.paths.centrelines.push(depth);
            LayerDepth::Curve(self.paths.centrelines.len() as u32 - 1, 0.0)
        };
        (
            curve,
            fill.map_or(curve, |(key, _, _)| LayerDepth::Plane(key)),
            registered,
            snap,
        )
    }

    /// The screen box the path at `shown` paints in, a stroke reaching `reach` past it, and the
    /// pixel its sub-samples are read in around that.
    fn region(&self, shown: &[[f64; 3]], reach: f64) -> [f64; 4] {
        let mut b = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for q in shown {
            let [x, y] = self.map.apply(*q);
            b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
        }
        let m = reach + 1.0;
        [b[0] - m, b[1] - m, b[2] + m, b[3] + m]
    }

    /// Adds the layer whose outline is the closed polygons `rings`, on the pixel plane, on the
    /// registered plane `on.0` with its key's snap departure `on.1` ([`Fragment::snap`]).
    fn push(
        &mut self,
        rings: &[Vec<[f64; 2]>],
        spec: PaintSpec,
        depth: LayerDepth,
        on: (u32, f64, Option<Plane>),
        order: u32,
        sub: u32,
    ) {
        let (plane, snap, on_plane) = on;
        let edges = scanline::edges_of(rings);
        if edges.is_empty() {
            return;
        }
        // A layer keyed on a sloped plane reads its fringe's depth on its whole outline.
        let flats = match depth {
            LayerDepth::Plane(p) if p.b != 0.0 || p.c != 0.0 => flats_of(rings),
            _ => Vec::new(),
        };
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
            flats: Vec::new(),
            bbox,
            paint: LayerPaint::Solid(colour),
            opaque: colour[3] == 255,
            depth,
            plane,
            snap,
            on_plane,
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
        layer.flats = flats;
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
            LayerDepth::Curve(i, body) => {
                let curve = &self.centrelines[i as usize];
                let centre = [x as f64 + 0.5, y as f64 + 0.5];
                if !curve.near_cut(centre, &[centre], body, near) {
                    curve.candidates(x, y, body, near);
                    curve.line.cut(centre, &[centre], near, body);
                }
                let (key, err) = curve.line.key_under(centre, near, body);
                Plane {
                    err,
                    ..Plane::constant(key)
                }
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
                        let mut f = Fragment::new(
                            depth,
                            layer.colour(x, y),
                            Coverage::Area {
                                area: a as f32,
                                id: i,
                            },
                            layer.order,
                            layer.sub,
                        );
                        f.path_layer = true;
                        f.plane = layer.plane;
                        f.snap = layer.snap;
                        add(x, y, f);
                    }
                },
            );
        }
    }
}

impl CurveDepth {
    /// `line`'s depth on a frame of `width` by `height` pixels, its segments' bounds widened by
    /// `margin` and filed, in order, under every tile holding a pixel centre they may hold.
    fn new(line: Centreline, margin: f64, width: u32, height: u32) -> CurveDepth {
        let bounds: Vec<[f64; 4]> = line.segment_bounds(margin).collect();
        let (across, down) = (tiles_across(width), height.div_ceil(TILE));
        // The first and last of `n` tiles along an axis meeting `lo..=hi` there, none beyond them.
        let span = |lo: f64, hi: f64, n: u32| {
            let first = (lo.max(0.0) / TILE as f64).floor() as u32;
            let last = ((hi / TILE as f64).floor() as u32).min(n.saturating_sub(1));
            (hi >= 0.0 && first <= last && n > 0).then_some((first, last))
        };
        let spans: Vec<Option<[u32; 4]>> = bounds
            .iter()
            .map(|b| {
                let (c0, c1) = span(b[0], b[2], across)?;
                let (r0, r1) = span(b[1], b[3], down)?;
                Some([c0, c1, r0, r1])
            })
            .collect();
        let tiles = |s: [u32; 4]| {
            (s[2]..=s[3]).flat_map(move |r| (s[0]..=s[1]).map(move |c| (r * across + c) as usize))
        };
        let mut starts = vec![0u32; (across * down) as usize + 1];
        for t in spans.iter().flatten().flat_map(|s| tiles(*s)) {
            starts[t + 1] += 1;
        }
        for t in 1..starts.len() {
            starts[t] += starts[t - 1];
        }
        let mut next = starts.clone();
        let mut tiled = vec![0u32; starts[starts.len() - 1] as usize];
        let mut beyond = Vec::new();
        for (k, s) in spans.iter().enumerate() {
            match s {
                Some(s) => {
                    for t in tiles(*s) {
                        tiled[next[t] as usize] = k as u32;
                        next[t] += 1;
                    }
                }
                None => beyond.push(k as u32),
            }
        }
        let reach = margin + 1.0;
        let region = [-reach, -reach, width as f64 + reach, height as f64 + reach];
        let fine = Fine::new(&line, region);
        CurveDepth {
            line,
            bounds,
            margin,
            across,
            down,
            starts,
            tiled,
            beyond,
            fine,
        }
    }

    /// The candidates [`Self::candidates`] then [`Centreline::cut`] leave for a read at pixel
    /// `(x, y)` among `points` (all within the pixel, its centre `centre` among them) of a stroke
    /// whose body reaches `body`, in order, found from the fine grid instead, into `kept`; false,
    /// `kept` as it was, where the grid cannot show they are the same (the read takes that path).
    ///
    /// The centre's nearest segment is found ring by ring of cells around the centre's, a segment
    /// projected onto only where its bounds lie within the nearest found so far
    /// ([`Centreline::cut`]'s rule), until every segment not met lies further than the nearest
    /// found: one filed only from ring `ring` out lies over `ring - 1` cells from the centre,
    /// and a part beyond the region over the margin and a pixel. Two equalities make the grid's
    /// answer the cut's, each holding with `room` for rounding ([`Centreline::rounding`]):
    /// - The least distance found equals the least over the candidates where it lies under the
    ///   margin less `room`: a segment that near the centre has widened bounds holding the centre,
    ///   so it is a candidate, and every candidate is a segment, so none is nearer.
    /// - The segments kept, those whose bounds lie within the cut's limit of the points' box, are
    ///   all candidates where the limit and half a pixel lie under the margin less `room` (every
    ///   point lies within half a pixel of the centre on each axis), and each lies within the
    ///   region, so it is filed under a cell the box widened by the limit meets: those cells'
    ///   segments, in order and cut by the same test, are the cut's.
    fn near_cut(
        &self,
        centre: [f64; 2],
        points: &[[f64; 2]],
        body: f64,
        kept: &mut Vec<u32>,
    ) -> bool {
        let fine = &self.fine;
        if fine.cols == 0 {
            return false;
        }
        let at = [centre[0], centre[1], centre[0], centre[1]];
        let room = self
            .line
            .rounding(centre[0].abs().max(centre[1].abs()) + self.margin + 2.0);
        let beyond = self.margin + 1.0;
        let (kx, ky) = (fine.at(centre[0], 0), fine.at(centre[1], 1));
        let mut nearest = f64::INFINITY;
        for ring in 0i64.. {
            // The rings met so far hold every point within `ring - 1` cells of the centre.
            let met = ((ring - 1).max(0) as f64 * FINE).min(beyond);
            if nearest.sqrt() + room < met {
                break;
            }
            if (ring - 1) as f64 * FINE > beyond {
                return false;
            }
            for r in (ky - ring).max(0)..=(ky + ring).min(fine.rows - 1) {
                for c in (kx - ring).max(0)..=(kx + ring).min(fine.cols - 1) {
                    if (r - ky).abs().max((c - kx).abs()) != ring {
                        continue;
                    }
                    for &k in fine.cell(c, r) {
                        let within = nearest.sqrt() + room;
                        if self.line.gap2(k as usize, at) <= within * within {
                            nearest = nearest.min(self.line.distance2(k as usize, centre));
                        }
                    }
                }
            }
            if kx - ring <= 0
                && ky - ring <= 0
                && kx + ring >= fine.cols - 1
                && ky + ring >= fine.rows - 1
            {
                break;
            }
        }
        let (area, limit) = self.line.reach(centre, points, nearest, body);
        // Answered only where it lies under the margin; never where it does not, or is not a number.
        if (limit + 0.5 + room).partial_cmp(&self.margin) != Some(std::cmp::Ordering::Less) {
            return false;
        }
        let wide = limit + room;
        let rows =
            fine.at(area[1] - wide, 1).max(0)..=fine.at(area[3] + wide, 1).min(fine.rows - 1);
        let cols =
            fine.at(area[0] - wide, 0).max(0)..=fine.at(area[2] + wide, 0).min(fine.cols - 1);
        kept.clear();
        for r in rows {
            for c in cols.clone() {
                kept.extend_from_slice(fine.cell(c, r));
            }
        }
        kept.sort_unstable();
        kept.dedup();
        kept.retain(|&k| self.line.gap2(k as usize, area) <= limit * limit);
        true
    }

    /// The segments whose bounds hold pixel `(x, y)`'s centre, in order, into `near`: of those
    /// filed under its tile, as every other's bounds miss the tile.
    fn near(&self, x: u32, y: u32, near: &mut Vec<u32>) {
        count!(NEAR);
        let t = (y / TILE * self.across + x / TILE) as usize;
        let filed = &self.tiled[self.starts[t] as usize..self.starts[t + 1] as usize];
        count!(BOUNDS_TESTED, filed.len());
        let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
        near.clear();
        near.extend(filed.iter().copied().filter(|&k| {
            let b = &self.bounds[k as usize];
            px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3]
        }));
    }

    /// The segments a read at pixel `(x, y)` (at its centre or its sub-samples) of a stroke whose
    /// body reaches `body` picks among, in order, into `near`: those whose bounds hold the pixel's
    /// centre ([`Self::near`]); where none does, enough of the others that a read among them
    /// picks as a read among every segment does ([`Self::around`]).
    fn candidates(&self, x: u32, y: u32, body: f64, near: &mut Vec<u32>) {
        self.near(x, y, near);
        if near.is_empty() {
            self.around(x, y, body, near);
        }
    }

    /// The segments that may lie nearest a point of pixel `(x, y)` or hold one within `body` of
    /// them, in order, into `out` (empty), where no segment's bounds hold the pixel's centre: those
    /// beyond the frame's tiles and those filed under the tiles ring by ring around the pixel's,
    /// until every segment not found lies further from every point of the pixel than `reach`, the
    /// further of `body` and the centre's nearest found and how far a point of the pixel lies from
    /// the centre (under `0.75`); then of those, the ones whose bounds lie within `reach` of the
    /// pixel. A segment not found has bounds widened by the margin that meet no tile within `ring`
    /// tiles of the pixel's, so it lies over `ring` tiles and the margin, less half a pixel, from
    /// every point of the pixel. A segment beyond `reach` of every point is neither the nearest of
    /// any (the centre's nearest lies within `reach`) nor holds one in its body, whatever the order
    /// of the rest, so a read among these picks as one among every segment does; each comparison
    /// holds with room for its rounding ([`Centreline::rounding`]).
    fn around(&self, x: u32, y: u32, body: f64, out: &mut Vec<u32>) {
        let centre = [x as f64 + 0.5, y as f64 + 0.5];
        let at = [centre[0], centre[1], centre[0], centre[1]];
        let room = self.line.rounding(centre[0].max(centre[1]));
        let mut nearest = f64::INFINITY;
        // A segment whose bounds lie further from the centre than the nearest found cannot be
        // nearer ([`Centreline::cut`]).
        let find = |k: u32, nearest: &mut f64| {
            let within = nearest.sqrt() + room;
            if self.line.gap2(k as usize, at) <= within * within {
                *nearest = nearest.min(self.line.distance2(k as usize, centre));
            }
        };
        for &k in &self.beyond {
            find(k, &mut nearest);
            out.push(k);
        }
        let (tx, ty) = ((x / TILE) as i64, (y / TILE) as i64);
        let (across, down) = (self.across as i64, self.down as i64);
        let reach = |nearest: f64| (nearest.sqrt() + 0.75).max(body);
        for ring in 0i64.. {
            for row in (ty - ring).max(0)..=(ty + ring).min(down - 1) {
                for column in (tx - ring).max(0)..=(tx + ring).min(across - 1) {
                    if (row - ty).abs().max((column - tx).abs()) != ring {
                        continue;
                    }
                    let t = (row * across + column) as usize;
                    for &k in &self.tiled[self.starts[t] as usize..self.starts[t + 1] as usize] {
                        find(k, &mut nearest);
                        out.push(k);
                    }
                }
            }
            let every = tx - ring <= 0
                && ty - ring <= 0
                && tx + ring >= across - 1
                && ty + ring >= down - 1;
            let clear = (ring * TILE as i64) as f64 + self.margin - 0.5;
            let reach = reach(nearest);
            if every || clear > reach + self.line.rounding(clear + reach) {
                break;
            }
        }
        out.sort_unstable();
        out.dedup();
        let reach = reach(nearest);
        let pixel = [x as f64, y as f64, x as f64 + 1.0, y as f64 + 1.0];
        let limit = reach + self.line.rounding(reach + pixel[2].max(pixel[3]));
        out.retain(|&k| self.line.gap2(k as usize, pixel) <= limit * limit);
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
    fn edges(&self, id: u32) -> Option<&[Edge]> {
        Some(&self.layers[id as usize].edges)
    }

    fn point_depths(
        &self,
        id: u32,
        x: u32,
        y: u32,
        points: &[[f64; 2]],
        out: &mut [f64],
    ) -> Option<f64> {
        let layer = &self.layers[id as usize];
        let (i, body) = match layer.depth {
            LayerDepth::Curve(i, body) => (i, body),
            LayerDepth::Plane(plane) => return fringe_depths(layer, plane, x, y, points, out),
        };
        let curve = &self.centrelines[i as usize];
        let mut near = Vec::new();
        let centre = [x as f64 + 0.5, y as f64 + 0.5];
        if curve.near_cut(centre, points, body, &mut near) {
            return Some(curve.line.keys_among(points, &near, body, out));
        }
        curve.candidates(x, y, body, &mut near);
        Some(curve.line.keys_under(centre, points, &mut near, body, out))
    }

    fn plane_depths(&self, id: u32, x: u32, y: u32, out: &mut [f64; SAMPLES]) -> Option<f64> {
        let plane = self.layers[id as usize].on_plane?;
        for (k, d) in out.iter_mut().enumerate() {
            let [px, py] = sample_point(x, y, k);
            *d = plane.at(px, py);
        }
        Some(plane.err + plane.read_error(x as f64 + 1.0, y as f64 + 1.0))
    }
}

/// A layer keyed on `plane`, at pixel `(x, y)`'s sub-samples (the first [`SAMPLES`] of `points`,
/// in order) and its centre: at a sub-sample whose point lies outside the layer's outline (the
/// exact point-in-outline test of [`scanline::mask`], whatever area of its cell the layer covers),
/// the plane read at the outline's point nearest it, the depth of the layer's edge there; at the
/// others, the plane at the point. A plane read beyond its outline runs on past every depth of
/// the layer, the farther the steeper it is: by a pixel's fraction on a face seen at a slant, past
/// a neighbour sharing its edge on one seen nearly edge-on. The error bound is the plane's own and
/// the rounding of reading it at any point read. `None`, the plane at every point, where it is flat
/// or covers every sub-sample.
fn fringe_depths(
    layer: &Layer,
    plane: Plane,
    x: u32,
    y: u32,
    points: &[[f64; 2]],
    out: &mut [f64],
) -> Option<f64> {
    if plane.b == 0.0 && plane.c == 0.0 {
        return None;
    }
    let inside = scanline::mask(&layer.edges, layer.candidates(y), x, y);
    if inside == u16::MAX {
        return None;
    }
    let mut rounding = plane.read_error(x as f64 + 1.0, y as f64 + 1.0);
    for (k, (p, d)) in points.iter().zip(out.iter_mut()).enumerate() {
        let read = if k < SAMPLES && inside & (1 << k) == 0 {
            layer.nearest_on_outline(*p, y)
        } else {
            *p
        };
        *d = plane.at(read[0], read[1]);
        rounding = rounding.max(plane.read_error(read[0], read[1]));
    }
    Some(plane.err + rounding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::{SAMPLES, sample_point};

    /// A small seeded generator (xorshift64*), so the cases are the same on every run.
    struct Seeded(u64);

    impl Seeded {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn pick<T: Copy>(&mut self, from: &[T]) -> T {
            from[self.below(from.len() as u64) as usize]
        }
    }

    /// The candidates a read at pixel `(x, y)` among `points` keeps without the fine grid.
    fn by_cut(depth: &CurveDepth, x: u32, y: u32, points: &[[f64; 2]], body: f64) -> Vec<u32> {
        let mut kept = Vec::new();
        depth.candidates(x, y, body, &mut kept);
        depth
            .line
            .cut([x as f64 + 0.5, y as f64 + 0.5], points, &mut kept, body);
        kept
    }

    fn points(x: u32, y: u32, all: bool) -> Vec<[f64; 2]> {
        let centre = [x as f64 + 0.5, y as f64 + 0.5];
        if !all {
            return vec![centre];
        }
        let mut points: Vec<[f64; 2]> = (0..SAMPLES).map(|k| sample_point(x, y, k)).collect();
        points.push(centre);
        points
    }

    #[test]
    fn the_fine_grid_keeps_what_the_cut_keeps() {
        let mut s = Seeded(0x00f1_2026_1107);
        let (mut cases, mut found, mut fell) = (0u64, 0u64, 0u64);
        while cases < 20_000 {
            let (w, h) = s.pick(&[(64u32, 48u32), (130, 70), (200, 40), (9, 300)]);
            let margin = s.pick(&[66.0, 70.5, 130.0, 12.0, 3.0, 0.75]);
            let step = s.pick(&[0.5, 1.0, 3.0, 12.0, 40.0]);
            let n = 1 + s.below(30) as usize;
            let mut at = [
                s.below(w as u64 + 160) as f64 - 80.0,
                s.below(h as u64 + 160) as f64 - 80.0,
            ];
            let mut line: Vec<([f64; 2], f64)> = Vec::new();
            for _ in 0..n {
                if s.below(6) != 0 {
                    at = [
                        at[0] + step * (s.below(9) as f64 - 4.0) / 4.0,
                        at[1] + step * (s.below(9) as f64 - 4.0) / 4.0,
                    ];
                }
                line.push((at, s.pick(&[-0.5, 0.0, 0.25, 1.0])));
            }
            let breaks: Vec<usize> = (1..n).filter(|_| s.below(7) == 0).collect();
            let line = Centreline::new(&line, &breaks);
            if line.is_empty() {
                continue;
            }
            cases += 1;
            let depth = CurveDepth::new(line, margin, w, h);
            let (x, y) = (s.below(w as u64) as u32, s.below(h as u64) as u32);
            let body = s.pick(&[0.0, 0.1, 0.5, 1.0, 2.5]);
            let points = points(x, y, s.below(2) == 0);
            let centre = [x as f64 + 0.5, y as f64 + 0.5];
            let mut kept = vec![u32::MAX];
            if depth.near_cut(centre, &points, body, &mut kept) {
                found += 1;
                assert_eq!(kept, by_cut(&depth, x, y, &points, body), "case {cases}");
            } else {
                fell += 1;
                assert_eq!(
                    kept,
                    vec![u32::MAX],
                    "case {cases}: a fallback touched its candidates"
                );
            }
        }
        // Both ways are taken often (half the margins are too narrow for the grid to answer at all).
        assert!(
            found > cases / 8 && fell > cases / 50,
            "{found} found, {fell} fell back"
        );
    }

    #[test]
    fn at_the_edge_of_the_grid_s_reach_its_answer_is_still_the_cut_s() {
        // One segment level with the pixel's centre, a distance below it that runs from where the
        // grid answers to where it cannot; at the last distance it answers and the first it does
        // not, a bit apart, it keeps what the cut keeps (or leaves the read to the cut).
        let (x, y, margin) = (100u32, 100u32, 66.0);
        let centre = [100.5, 100.5];
        let depth_at = |below: f64| {
            let line = Centreline::new(
                &[
                    ([90.0, centre[1] + below], 0.5),
                    ([111.0, centre[1] + below], 0.25),
                    ([180.0, centre[1] + below], 1.0),
                ],
                &[],
            );
            CurveDepth::new(line, margin, 200, 260)
        };
        for all in [false, true] {
            let points = points(x, y, all);
            for body in [0.0, 1.0] {
                let answers = |below: f64| {
                    let mut kept = Vec::new();
                    depth_at(below).near_cut(centre, &points, body, &mut kept)
                };
                let (mut yes, mut no) = (10.0f64, 80.0f64);
                assert!(answers(yes) && !answers(no));
                while no.to_bits() - yes.to_bits() > 1 {
                    let mid = f64::from_bits((yes.to_bits() + no.to_bits()) / 2);
                    if answers(mid) { yes = mid } else { no = mid }
                }
                assert!(yes > margin - 3.0 && yes < margin, "the edge at {yes}");
                for below in [
                    f64::from_bits(yes.to_bits() - 1),
                    yes,
                    no,
                    f64::from_bits(no.to_bits() + 1),
                ] {
                    let depth = depth_at(below);
                    let mut kept = Vec::new();
                    if depth.near_cut(centre, &points, body, &mut kept) {
                        assert_eq!(kept, by_cut(&depth, x, y, &points, body), "at {below}");
                    }
                }
            }
        }
    }
}
