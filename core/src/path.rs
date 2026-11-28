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
    Coverage, FlatRecord, Fragment, OpaquePass, Plane, Producer, Rect, SAMPLES, TransparentPass,
    clear, premultiply, sample_cell, sample_point,
};
use crate::depth::{Boundary, Centreline, Flat, Planarity, Registry, WorldPlane};
use crate::gradient::{self, Linear};
use crate::join_mask;
use crate::project::{FrameMap, Placement, Projection, placement, pre_display, produced_key};
use crate::scanline::{self, Edge, Scratch};
use crate::tiles::{TILE, tile_index, tiles_across};
use crate::xf::*;
use kurbo::{BezPath, ParamCurve as _, ParamCurveArclen as _, PathEl, PathSeg, Point, Shape as _};
use std::collections::HashMap;
use std::sync::Arc;

/// Cairo's default tolerance for flattening curves, in pixels.
pub const TOLERANCE: f64 = 0.1;

/// The share of a stroke's half width its sides may stray by, where that is tighter than
/// [`TOLERANCE`].
const HALF_WIDTH_SHARE: f64 = 0.25;

/// The tolerance a stroke of `width` pixels is stroked at, in pixels. Cairo flattens the centreline
/// at [`TOLERANCE`] and offsets each point it reaches by the exact half width, so its band keeps its
/// width however thin; kurbo fits each side's offset to within the tolerance it is given, which at
/// a half width under [`TOLERANCE`] lets both sides settle near the centreline and the band
/// collapse. So a thin stroke's sides are fitted to a share of its half width instead.
fn stroke_tolerance(width: f64) -> f64 {
    TOLERANCE.min(HALF_WIDTH_SHARE * width / 2.0)
}

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

/// Cairo's least length (`CAIRO_FIXED_ERROR_DOUBLE`, 1/512): `_cairo_gstate_set_dash` makes a
/// dash whose gaps total less than this solid.
const SOLID_DASH_GAP: f64 = 1.0 / 512.0;

/// A dash's two entries, on and off, as they are handed to Cairo (`set_dash`).
fn dash_entries(d: &Dash) -> [f64; 2] {
    [d.period * d.duty, d.period * (1.0 - d.duty)]
}

/// The dash `rec` is stroked with, as Cairo takes it: none where it has none, its duty is 1, or
/// its gap is shorter than Cairo's least length ([`SOLID_DASH_GAP`]), which Cairo strokes solid.
/// The gap is Cairo's own sums (`_cairo_gstate_set_dash`): the entries' total less the on length.
fn drawn_dash(rec: &Record) -> Option<Dash> {
    rec.dash.filter(|d| {
        let [on, off] = dash_entries(d);
        let total = on + off;
        d.duty < 1.0 && total - on >= SOLID_DASH_GAP
    })
}

/// Whether Cairo's dashed stroker starts a subpath in a dash that is on (`dash_starts_on`): the
/// offset taken within the entries' total (`_cairo_gstate_set_dash`), then stepped past each
/// entry it reaches or passes (`_cairo_stroker_dash_start`).
fn dash_starts_on(d: &Dash) -> bool {
    let entries = dash_entries(d);
    let total = entries[0] + entries[1];
    let mut offset = (d.phase * d.period) % total;
    if offset < 0.0 {
        offset += total;
    }
    let (mut on, mut i) = (true, 0);
    while offset >= entries[i] {
        offset -= entries[i];
        on = !on;
        i = (i + 1) % 2;
    }
    on
}

/// Whether `draw`'s strokes are a round cap's dot, as Cairo strokes a closed subpath of no
/// length: a face of a surface (one not yet drawn, `Create`, is its points at one spot) whose
/// points `shown` are at one spot as the camera shows them, under a round cap (a butt or square
/// one strokes nothing of it), undashed or dashed on at the subpath's start. Cairo's dashed
/// stroker starts each subpath at the dash's offset, stepping past each dash the offset reaches
/// or passes, and dots the subpath only when the dash it lands in is on (`dash_starts_on`).
fn strokes_as_dot(draw: &PathDraw<'_>, shown: &[[f64; 3]]) -> bool {
    let rec = draw.record;
    let on_at_start = match drawn_dash(rec) {
        None => true,
        Some(d) => dash_starts_on(&d),
    };
    draw.surface
        && rec.paint.cap == Cap::Round
        && on_at_start
        && shown.iter().all(|p| p[..2] == shown[0][..2])
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

impl PaintSpec {
    /// Whether Cairo paints nothing of it ([`clear`]): a solid colour's, or the one colour a
    /// gradient collapses to ([`gradient::collapsed`]: clear, or meeting ends' average).
    fn clear(&self) -> bool {
        match self {
            PaintSpec::Solid(colour) => clear(*colour),
            PaintSpec::Gradient(stops, p1, p2) => {
                gradient::collapsed(*p1, *p2, stops).is_some_and(clear)
            }
        }
    }
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

/// Where a layer lies ([`Lowering::push`]): the identities of the registered planes it lies on,
/// its key's shares ([`Fragment::snap`] and [`Fragment::wide`]), its keys on those planes, the
/// registry's handle of its flat record ([`Fragment::record`]), and its own share of its key
/// with how far on screen the snap may move its curve's points ([`Fragment::own`],
/// [`Layer::shift`]).
type On = (
    [u32; 4],
    [f64; 2],
    [Option<Plane>; 4],
    Option<u32>,
    [f64; 2],
);

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
    /// a sloped plane reads its fringe's depth on ([`fringe_depths`]) and the labels' planner
    /// tests a point on ([`LayerView::flats`]), whatever the layer is keyed on.
    flats: Vec<[f64; 3]>,
    /// Pixels `x0..x1` by `y0..y1`, inside the frame.
    bbox: [u32; 4],
    paint: LayerPaint,
    /// Every pixel of its paint is opaque.
    opaque: bool,
    depth: LayerDepth,
    /// The identities of the registered planes it lies on ([`Fragment::plane`]), all zero for
    /// none.
    plane: [u32; 4],
    /// How far its key may stand from the plane content put it on ([`Fragment::snap`]): finite
    /// for a layer of a path on a registered plane.
    snap: f64,
    /// Its share against what else has no plane of its own ([`Fragment::wide`]).
    wide: f64,
    /// For a stroke of a path with no plane of its own, its path's share of the snap's move of its
    /// keys at fixed points of its curve ([`Fragment::own`] adds, pixel by pixel, how far its reads
    /// may slide along it: [`Paths::own_at`]).
    own: f64,
    /// For such a stroke, how far on screen, in pixels, the snap may move each point its curve is
    /// drawn through ([`screen_shift`]); zero for any other layer.
    shift: f64,
    /// For such a stroke, drawn after its path's fill and keyed on the curve that fill is read on
    /// (with no body): where both cover a sub-sample, its key there is the nearer of its read and
    /// the fill's ([`crate::composite::strokes_on_own_fills`]), which its own share bounds too.
    on_fill: bool,
    /// For a layer of a path keyed on a registered plane, its path's key on each identity's plane,
    /// in the order of `plane`: what it is read on against a fragment sharing that identity
    /// ([`Producer::plane_depths`]).
    on_plane: [Option<Plane>; 4],
    /// For a layer of a flat path keyed with a finite `snap`, its record's handle in the frame's
    /// registry: the record its fragments carry ([`Fragment::record`]).
    record: Option<u32>,
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

/// One layer as the labels' planner reads it ([`Paths::layers`]).
pub(crate) struct LayerView<'a> {
    pub order: u32,
    pub sub: u32,
    /// Its pixels, `x0..x1` by `y0..y1`, inside the frame.
    pub bbox: [u32; 4],
    /// Its outline's edges, but for its horizontal segments, in device pixels.
    pub edges: &'a [Edge],
    /// Its outline's horizontal segments, as their height and their two ends' x, least first:
    /// with `edges`, the whole outline.
    pub flats: &'a [[f64; 3]],
    /// The plane it is keyed on, where it is keyed on one.
    pub plane: Option<Plane>,
    layer: &'a Layer,
}

impl LayerView<'_> {
    /// The premultiplied colour its fragments carry at pixel `(x, y)`, as the layer paints it there:
    /// at the pixel of its box nearest for one outside the box; clear for a box with no pixel.
    pub fn colour(&self, x: u32, y: u32) -> [u8; 4] {
        let b = self.bbox;
        if b[0] >= b[2] || b[1] >= b[3] {
            return [0; 4];
        }
        self.layer
            .colour(x.clamp(b[0], b[2] - 1), y.clamp(b[1], b[3] - 1))
    }
}

/// Paths lowered to coverage layers, binned by tile: a producer of area fragments.
pub struct Paths {
    layers: Vec<Layer>,
    centrelines: Vec<CurveDepth>,
    bins: Vec<Vec<u32>>,
    width: u32,
    /// The frame's registry, settled, which holds the flat paths' records their layers' fragments
    /// carry ([`Fragment::record`]); none for paths lowered without one.
    records: Option<Arc<Registry>>,
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

/// The decisions a record's `joins` (XF 1.4: one byte per cubic of its points) carries for the
/// joins of [`outline`]`(shown)` as kurbo's stroker makes them, by join ([`join_mask::key`]: the
/// cubic it leaves and the one it enters), one per occurrence: the join into each cubic mitred
/// where bit 1 is set, where bit 0 is; undecided (`None`) where bit 0 is clear (Cairo makes no
/// join there) or the record holds no byte for it, which keeps its place among the occurrences
/// of its points; the first cubic's byte of a closed subpath, its closing join. A cubic whose
/// points are one the stroker skips: the join is from the cubic before it, and the closing join
/// into the first it strokes.
pub fn decided_joins(shown: &[[f64; 3]], joins: &[u8]) -> join_mask::Decided {
    let mut out = HashMap::new();
    let pt = |p: [f64; 3]| Point::new(p[0], p[1]);
    let byte = |i: usize| {
        joins
            .get(i / 4)
            .copied()
            .filter(|b| b & 1 != 0)
            .map(|b| b & 2 != 0)
    };
    for (start, end) in subpaths(shown) {
        if end - start < 4 {
            continue;
        }
        let (a, b) = (shown[start], shown[end - 1]);
        let closed = close(a[0], b[0]) && close(a[1], b[1]);
        let mut p0 = pt(a);
        let (mut first, mut last): (Option<[Point; 4]>, Option<[Point; 4]>) = (None, None);
        for i in (start..=end - 4).step_by(4) {
            // As `outline` writes it: a closed subpath ends exactly at its start.
            let end_point = if closed && i + 4 == end {
                a
            } else {
                shown[i + 3]
            };
            let (p1, p2, p3) = (pt(shown[i + 1]), pt(shown[i + 2]), pt(end_point));
            if p1 == p0 && p2 == p0 && p3 == p0 {
                continue;
            }
            let this = [p0, p1, p2, p3];
            match last {
                Some(from) => {
                    let key = join_mask::key(from, this, false);
                    out.entry(key).or_insert_with(Vec::new).push(byte(i));
                }
                None => first = Some(this),
            }
            (last, p0) = (Some(this), p3);
        }
        if let (true, Some(from), Some(into)) = (closed, last, first) {
            let key = join_mask::key(from, into, true);
            out.entry(key).or_insert_with(Vec::new).push(byte(start));
        }
    }
    out
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
/// control points' ([`Projection::key_error`]) and the rounding of the cubic's weighted sum. Each
/// cubic's first piece after another of the list is a joint ([`Boundary`]), and each cubic's cut
/// into pieces is recorded ([`Flat`]).
fn curve_depth(shown: &[[f64; 3]], map: &FrameMap, projection: &Projection) -> Centreline {
    let mut points = Vec::new();
    let mut errors = Vec::new();
    let mut turns = Vec::new();
    let mut breaks = Vec::new();
    let mut joints = Vec::new();
    let mut pieces = Vec::new();
    let mut flats = Vec::new();
    let screen = |p: [f64; 3]| (map.apply(p), -projection.factor(p[2]));
    // The joint of the cubic at `i` after another, its first piece ending at point `end`: the
    // previous cubic's end and this one's first point as drawn, CE's tolerance at the latter through
    // the frame's scale on each axis, and whether the drawn frame splits the two.
    let joint = |i: usize, end: usize, split: bool| {
        let ((e, ke), (s, ks)) = (screen(shown[i - 1]), screen(shown[i]));
        let tol = |k: usize, scale: f64| (POINT_EQUALITY + 1e-5 * shown[i][k].abs()) * scale.abs();
        Boundary {
            end,
            gap: libm::hypot(s[0] - e[0], s[1] - e[1]),
            tol: libm::hypot(tol(0, map.xx), tol(1, map.yy)) * (1.0 + 1e-9),
            key: (ks - ke).abs(),
            split,
        }
    };
    for (start, end) in subpaths(shown) {
        if end - start < 4 {
            continue;
        }
        breaks.push(points.len());
        points.push(screen(shown[start]));
        errors.push(projection.key_error(shown[start]));
        turns.push(2.0);
        pieces.push(u32::MAX);
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
            flats.push(Flat::of(c, length));
            let weights = |t: f64| {
                let u = 1.0 - t;
                [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t]
            };
            for s in 1..=n {
                let t = s as f64 / n as f64;
                let w = weights(t);
                let at = |f: &dyn Fn(usize) -> f64| {
                    w[0] * f(0) + w[1] * f(1) + w[2] * f(2) + w[3] * f(3)
                };
                points.push(([at(&|k| c[k].0[0]), at(&|k| c[k].0[1])], at(&|k| c[k].1)));
                errors.push(err);
                pieces.push(flats.len() as u32 - 1);
                if s == 1 && i >= 4 {
                    joints.push(joint(i, points.len() - 1, i == start));
                }
                // The piece's two ends move apart by at most the sum of their weights' changes
                // times how far a control point moves (at most `6 dt` of the parameter's `dt`); a
                // cubic's first piece after another of its subpath starts at the other's end, a
                // point of its own, so its ends move apart on their own.
                let before = weights((s - 1) as f64 / n as f64);
                let turn: f64 = (0..4).map(|k| (w[k] - before[k]).abs()).sum();
                turns.push(if s == 1 && i > start {
                    2.0
                } else {
                    (turn * (1.0 + 16.0 * crate::composite::U)).min(2.0)
                });
            }
        }
    }
    Centreline::with_flats(&points, &errors, &turns, &breaks, &joints, &pieces, &flats)
}

/// How far on screen, in pixels, the snap may move any point a curve through `shown` (as the
/// camera placed them, under `p` and the frame mapping `map`) is drawn through, where it moves
/// each of its inputs by at most `across` along the axes that become the screen's (the camera's
/// first two for a `projected` path, else the points' own x and y) and their factor by at most
/// `share` of itself: a projected point shows at `q f zoom` for its position `q` before the
/// perspective, which moves to `(q + e) f' zoom` with `|f' - f| <= share f`, so by at most
/// `f zoom (1 + share) |e| + |shown| share` along each axis; any other point shows where it is.
/// Each is mapped by `map`'s scale on each axis; the most of them.
fn screen_shift(
    shown: &[[f64; 3]],
    p: &Projection,
    map: &FrameMap,
    projected: bool,
    across: [f64; 2],
    share: f64,
) -> f64 {
    if !share.is_finite() {
        return f64::INFINITY;
    }
    shown
        .iter()
        .map(|q| {
            let [x, y] = if projected {
                let f = (p.factor(q[2]) * p.zoom()).abs();
                [0, 1].map(|k| f * (1.0 + share) * across[k] + q[k].abs() * share)
            } else {
                across
            };
            libm::hypot(map.xx.abs() * x, map.yy.abs() * y)
        })
        .fold(0.0, f64::max)
        * (1.0 + 64.0 * crate::composite::U)
}

/// Where a path's depth is keyed ([`Lowering::coplanar`]).
#[derive(Clone, Debug)]
pub struct Keyed {
    /// The plane its fill is keyed on against what shares none of its identities: a flat path's
    /// own; `None` for its curve's depth, as for a path with no plane of its own, whose points lie
    /// on a line ([`Lowering::coplanar`]).
    pub plane: Option<WorldPlane>,
    /// Its identities ([`Fragment::plane`]), all zero for none.
    pub ids: [u32; 4],
    /// How its points lie.
    pub planarity: Planarity,
    /// For a path with no plane of its own on identities, each one's plane, in the order of `ids`:
    /// what its layers are read on against what shares that identity. Empty for any other.
    pub shared: Vec<WorldPlane>,
    /// For a path with no plane of its own attached to flat records' identities, for each of
    /// `shared`, how far its points may stand off that plane, as its shares of a key read there
    /// take it: where it meets the flat records on it ([`Registry::off_members`]), and wherever it
    /// meets another record with no plane of its own ([`Registry::off_identity`]). Empty for any
    /// other: one on axis planes alone lies on each as its writer coded it or exactly.
    pub offs: Vec<[f64; 2]>,
    /// For a flat path (offered to the registry on its own plane), its handle there: the record
    /// its layers' fragments carry ([`Fragment::record`]). None for any other.
    pub record: Option<u32>,
    /// Offered to the registry with no plane of its own, attached or not: its strokes carry their
    /// own share of their keys ([`Fragment::own`]).
    pub planeless: bool,
}

impl Keyed {
    /// A path keyed on `plane`, on no identity.
    fn alone(plane: WorldPlane, planarity: Planarity) -> Keyed {
        Keyed {
            plane: Some(plane),
            ids: [0; 4],
            planarity,
            shared: Vec::new(),
            offs: Vec::new(),
            record: None,
            planeless: false,
        }
    }
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
                records: None,
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

    /// Where each of `draws` (in draw order) is keyed ([`Keyed`]): `None` for a path read by its
    /// curve's depth that the registry was not offered, and for one Cairo paints nothing of
    /// ([`Lowering::paints`]), which is not offered: the identities are a function of the set
    /// offered, which what draws nothing must not move.
    ///
    /// Every path in the world is offered to the frame's [`Registry`], with its meshes' triangles,
    /// then the registry settles and each takes its identities, whatever the draw order: a flat
    /// path (its record's `planes`, or decided from its points), fitted its plane, whose points do
    /// not lie on a line, takes part in the partition and is keyed on its own plane; a path with
    /// no plane of its own (points on a line, however coded; a curve) attaches to every identity on
    /// whose plane it lies and is read on each one's plane against what shares it, and against
    /// anything else by its curve's depth, its fill as its strokes (on no flat record's identity,
    /// it shares one with the others on each axis plane it lies on, coded or exactly, and is read
    /// on each such plane against them alike). Its points have no area for a plane to be a face
    /// on, so no plane fitted to them keys it: read at their outline, such a plane departs from
    /// their depth by the fit's departure over the cosine of a near-grazing angle. So things on
    /// one plane tie by draw order, though each plane fitted to snapped points differs by the
    /// snap's noise, and attaching moves no key read against anything else. A surface's face on no
    /// one plane, and a pre-projected path, keep the plane fitted to their own points.
    pub fn coplanar(&self, draws: &[PathDraw<'_>], registry: &mut Registry) -> Vec<Option<Keyed>> {
        let mut found = vec![None; draws.len()];
        let mut offered = Vec::new();
        for (i, draw) in draws.iter().enumerate() {
            if !self.paints(draw) {
                continue;
            }
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
            if !in_world || planarity == Planarity::Nearest {
                found[i] =
                    WorldPlane::fit(&points, planarity).map(|plane| Keyed::alone(plane, planarity));
                continue;
            }
            let (handle, flat, own) =
                registry.offer_path(&points, planarity, &draw.places, &draw.points[..]);
            offered.push((i, handle, flat, own, planarity));
        }
        registry.settle();
        for (i, handle, flat, own, planarity) in offered {
            let ids = registry.identities(handle);
            found[i] = if !flat && registry.attached(handle) {
                let on = ids.iter().take_while(|id| **id != 0);
                Some(Keyed {
                    plane: None,
                    ids,
                    planarity,
                    shared: on.clone().map(|id| registry.plane(*id)).collect(),
                    offs: on
                        .map(|id| {
                            let off = |f: fn(&Registry, u32, u32) -> f64| f(registry, handle, *id);
                            [off(Registry::off_members), off(Registry::off_identity)]
                        })
                        .collect(),
                    record: None,
                    planeless: true,
                })
            } else if !flat {
                let on = ids.iter().take_while(|id| **id != 0);
                Some(Keyed {
                    plane: None,
                    ids,
                    planarity,
                    shared: on.map(|id| registry.plane(*id)).collect(),
                    offs: Vec::new(),
                    record: None,
                    planeless: true,
                })
            } else {
                own.map(|plane| Keyed {
                    ids,
                    record: Some(handle),
                    ..Keyed::alone(plane, planarity)
                })
            };
        }
        found
    }

    /// Lowers one path into up to three layers, `plane` where it is keyed ([`Lowering::coplanar`]),
    /// or `None` for its curve's depth off the registry.
    /// How `draw`'s points are placed for display and keyed for depth: a pre-projected path's as
    /// its camera produced them, any other's by this frame's camera.
    fn placed(&self, draw: &PathDraw<'_>) -> (Placement, Option<Plane>) {
        let rec = draw.record;
        if draw.pre_projected {
            let key = produced_key(self.axes.as_ref(), &self.map, rec.depth.space, draw.pivot);
            (Placement::Flat, key)
        } else {
            placement(
                self.projection.as_ref(),
                &self.map,
                rec.depth.space,
                rec.depth.fixed_orientation_centre,
            )
        }
    }

    /// What each layer of `draw` (background stroke, fill, stroke) paints, as [`Lowering::path`]
    /// draws it: its paint and its width in CE's units, `None` for a layer it draws nothing of (no
    /// paint, a gradient without ends, a paint Cairo paints nothing of ([`PaintSpec::clear`]), a
    /// stroke of no width); and whether the fill has paint at all. `shown` is `world` as
    /// `placement` shows it. CE's lighting (`ThreeDCamera.modified_rgbas`): a shaded path's paint
    /// becomes two stops, lit at its start and end corners. Its gradient runs, as any path's does,
    /// between the ends the mobject gives (`get_gradient_start_and_end_points`, the record's): for
    /// a shaded path its corners, unless it gives others, as `ThreeDAxes` gives every piece of an
    /// axis the axis's own ends. A shaded path with no ends recorded (a face of a surface, an
    /// instance) runs between its corners.
    fn layers(
        &self,
        draw: &PathDraw<'_>,
        world: &[[f64; 3]],
        shown: &[[f64; 3]],
        placement: Placement,
    ) -> ([Option<(PaintSpec, f64)>; 3], bool) {
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
        let [background_rows, fill_rows, stroke_rows] = self.layer_rows(draw, world, lit);
        let fill_drawn = fill_rows.is_some();
        let multiple = self.view.cairo_line_width_multiple;
        let layer = |k: usize, rows: Option<Vec<[f64; 4]>>, width: f64| {
            let rows = rows?;
            let spec = match (rows.as_slice(), ends) {
                ([one], _) => PaintSpec::Solid(premultiply(*one)),
                (stops, Some([p1, p2])) if stops.len() > 1 => {
                    PaintSpec::Gradient(stops.to_vec(), [p1[0], p1[1]], [p2[0], p2[1]])
                }
                // Rows without ends are refused before drawing.
                _ => return None,
            };
            if spec.clear() {
                return None;
            }
            let w = width * multiple;
            if k != 1 && (w.is_nan() || w <= 0.0) {
                return None;
            }
            Some((spec, width))
        };
        (
            [
                layer(0, background_rows, draw.background_stroke_width),
                layer(1, fill_rows, 0.0),
                layer(2, stroke_rows, draw.stroke_width),
            ],
            fill_drawn,
        )
    }

    /// Whether [`Lowering::path`] strokes `draw` as a round cap's dot ([`strokes_as_dot`]): a
    /// stroke of it is drawn ([`Lowering::layers`]) and its points as this frame's camera shows
    /// them are at one spot. Nothing is drawn.
    pub fn round_cap_dot(&self, draw: &PathDraw<'_>) -> bool {
        let Some(world) = drawn_world(draw) else {
            return false;
        };
        let placement = self.placed(draw).0;
        let shown = pre_display(&world, self.projection.as_ref(), placement);
        if outline(&shown).elements().is_empty() || !strokes_as_dot(draw, &shown) {
            return false;
        }
        let ([background, _, stroke], _) = self.layers(draw, &world, &shown, placement);
        background.is_some() || stroke.is_some()
    }

    /// Whether [`Lowering::path`] draws a layer of `draw` ([`Lowering::layers`]): whether Cairo
    /// paints anything of it. Nothing is drawn.
    fn paints(&self, draw: &PathDraw<'_>) -> bool {
        let Some(world) = drawn_world(draw) else {
            return false;
        };
        let placement = self.placed(draw).0;
        let shown = pre_display(&world, self.projection.as_ref(), placement);
        let (layers, _) = self.layers(draw, &world, &shown, placement);
        layers.iter().any(Option::is_some)
    }

    pub fn path(&mut self, draw: &PathDraw<'_>, plane: Option<Keyed>) {
        let rec = draw.record;
        let Some(world) = drawn_world(draw) else {
            return;
        };
        let (placement, keyed) = self.placed(draw);
        let shown = pre_display(&world, self.projection.as_ref(), placement);
        let user = outline(&shown);
        if user.elements().is_empty() {
            return;
        }
        let decided = rec.joins.as_deref().map(|j| decided_joins(&shown, j));
        // The strokes' depth and the fill's.
        let (strokes, fill, ids, fill_snap, plane_rows) = match keyed {
            Some(plane) => (
                LayerDepth::Plane(plane),
                LayerDepth::Plane(plane),
                [0; 4],
                [f64::INFINITY, f64::NEG_INFINITY],
                [None; 4],
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
                    self.world_depth(&axes, &back, &shown, reach, bins, plane.as_ref(), None)
                } else {
                    let p = self
                        .projection
                        .clone()
                        .expect("world depth needs the 3D camera");
                    let placed = (&draw.places, &draw.points[..]);
                    let on = plane.as_ref();
                    self.world_depth(&p, &world, &shown, reach, bins, on, Some(placed))
                }
            }
        };
        // A flat path's record, carried by each of its layers keyed with its term.
        let record = plane.as_ref().and_then(|on| on.record);
        // A path with no plane of its own: how far its snap may move the keys at fixed points of
        // its curve, as a share of them, and how far on screen it may move the points the curve
        // is drawn through, which its strokes carry ([`Fragment::own`], [`Layer::shift`]). Its key
        // is `-factor` of the depth before the perspective, which the snap moves by at most its
        // departure along the axis that depth is read along (the camera's sight axis for a
        // projected path; the points' own z for one a camera of the scene's own placed, or one at
        // a fixed orientation, whose points are shifted, not turned); on screen it moves them by
        // at most its departures along the other two.
        let own = match (&plane, self.projection.as_ref()) {
            (Some(on), Some(p)) if on.planeless => {
                let grid = crate::depth::input_grid(self.view);
                let projected = matches!(placement, Placement::Projected);
                let axes = if projected {
                    p.rotation()
                } else {
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                };
                let along = |k: usize| {
                    crate::depth::snap_departure(grid, axes[k], &draw.places, &draw.points)
                };
                let share = p.depth_share(&shown, along(2));
                let across = [along(0), along(1)];
                [
                    share,
                    screen_shift(&shown, p, &self.map, projected, across, share),
                ]
            }
            _ => [f64::NEG_INFINITY, 0.0],
        };
        let scale = self.map.xx.abs().max(self.map.yy.abs());
        let area = self.guard(&user, draw.stroke_width.max(draw.background_stroke_width));
        let guarded = match area {
            Some(area) => culled(&user, area),
            None => user.clone(),
        };
        let (layers, fill_drawn) = self.layers(draw, &world, &shown, placement);
        for (k, layer) in layers.into_iter().enumerate() {
            let Some((spec, width)) = layer else { continue };
            let k = k as u32;
            let outline_px = if k == 1 {
                mapped(&guarded, &self.map)
            } else {
                let w = width * self.view.cairo_line_width_multiple;
                let style = kurbo::Stroke::new(w)
                    .with_join(kurbo_join(rec.paint.joint))
                    .with_caps(kurbo_cap(rec.paint.cap))
                    .with_miter_limit(MITER_LIMIT);
                // Dashes are laid on the whole outline, then culled, so culling never moves them.
                let dashes = drawn_dash(rec).map(|d| {
                    let pattern = dash_entries(&d);
                    dashed(&user, d.phase * d.period, &pattern, area)
                });
                let elements = dashes.as_ref().unwrap_or(&guarded).elements();
                let stroked = if strokes_as_dot(draw, &shown) {
                    kurbo::Circle::new(Point::new(shown[0][0], shown[0][1]), w / 2.0)
                        .to_path(TOLERANCE / scale)
                } else {
                    // kurbo decides which turns it joins by the tolerance it strokes at, so the
                    // joins are read at the same one.
                    let tolerance = stroke_tolerance(w * scale) / scale;
                    let stroked = kurbo::stroke(
                        elements.iter().copied(),
                        &style,
                        &kurbo::StrokeOpts::default(),
                        tolerance,
                    );
                    // Joins carried as Cairo decided them (XF 1.4): kurbo's stroke, edited at those
                    // it decided otherwise.
                    match &decided {
                        Some(decided) => {
                            let (edited, _edits) =
                                join_mask::apply(stroked, elements, &style, tolerance, decided);
                            count!(JOINS_DROPPED, _edits.dropped as usize);
                            count!(JOINS_INSERTED, _edits.inserted as usize);
                            count!(JOINS_PARALLEL, _edits.parallel as usize);
                            count!(JOINS_UNMATCHED, _edits.unmatched as usize);
                            edited
                        }
                        None => stroked,
                    }
                };
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
            // sub-sample (for a path with no plane of its own, on each identity's), with its
            // path's term, against what shares that plane ([`Producer::plane_depths`]): a stroke,
            // keyed on its curve, and the fill, whose own key beyond its outline is the plane's at
            // the outline's nearest point ([`fringe_depths`]), or for a path with no plane of its
            // own, its key along its curve.
            let on_plane = if ids[0] != 0 { plane_rows } else { [None; 4] };
            let snap = if k == 1 || on_plane.iter().any(Option::is_some) {
                fill_snap
            } else {
                [f64::INFINITY, f64::NEG_INFINITY]
            };
            let record = record.filter(|_| snap[0].is_finite());
            let own = if k == 1 {
                [f64::NEG_INFINITY, 0.0]
            } else {
                own
            };
            self.push(
                &rings(&outline_px),
                spec,
                depth,
                (ids, snap, on_plane, record, own),
                draw.order,
                draw.sub + k,
                k == 2 && fill_drawn && matches!(fill, LayerDepth::Curve(..)),
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
    /// ([`Projection::key_plane`]). A flat path content did not make flat (keyed on a plane its
    /// points lie near) has no plane content put it on: zero. A path with no plane of its own on
    /// identities ([`Keyed::shared`]) is read on each one's plane against what shares it: for
    /// each, how far its points may stand off that plane ([`Keyed::offs`]) bounds how far it may
    /// stand from that plane, and it carries the most of those; its fill keyed as on none, along
    /// its curve ([`Keyed::plane`]); on no identity, its fill keyed so,
    /// where the camera keys the first identity's plane nowhere (the eye in it). One on axis
    /// planes alone (no `offs`: on no flat record's identity) is read on each of those the camera
    /// keys, carrying the most of the snap's departures along them and its points' from them, and
    /// on none of the others. A path the camera placed itself (`placed` none) has no bound.
    ///
    /// With the depths: the identities, and the rows each layer is read on against what shares
    /// each ([`Producer::plane_depths`]): a flat path's own key for its one identity, else each
    /// identity's plane keyed, as any registered plane is ([`Projection::key_plane`]).
    #[allow(clippy::too_many_arguments)]
    fn world_depth(
        &mut self,
        p: &Projection,
        world: &[[f64; 3]],
        shown: &[[f64; 3]],
        reach: f64,
        bins: f64,
        plane: Option<&Keyed>,
        placed: Option<(&[Affine; 2], &[[f64; 3]])>,
    ) -> (
        LayerDepth,
        LayerDepth,
        [u32; 4],
        [f64; 2],
        [Option<Plane>; 4],
    ) {
        let grid = crate::depth::input_grid(self.view);
        let fill = plane
            .filter(|_| world.iter().all(|q| p.keys_planar_at(*q)))
            .and_then(|on| {
                let region = self.region(shown, reach);
                // Its own key, read against what shares none of its identities: on its own plane,
                // or none (its curve's depth), as for a path with no plane of its own.
                let key = on.plane.and_then(|plane| {
                    let mut key = p.key_plane(&plane, &self.map, world[0], 0.0, region)?;
                    if on.planarity == Planarity::Nearest {
                        key.err += nearest_plane_departure(&key, shown, &self.map, p);
                    }
                    Some(key)
                });
                let mut rows = [None; 4];
                let (mut ids, mut planes) = (on.ids, Vec::new());
                if on.shared.is_empty() {
                    rows[0] = Some(key?);
                } else if on.offs.is_empty() {
                    // On axis planes alone: read on each one the camera keys against what shares
                    // it, and on none of the others (the eye in it), as against what shares none.
                    ids = [0; 4];
                    for (id, shared) in on.ids.iter().zip(&on.shared) {
                        if let Some(row) = p.key_plane(shared, &self.map, world[0], 0.0, region) {
                            (ids[planes.len()], rows[planes.len()]) = (*id, Some(row));
                            planes.push(*shared);
                        }
                    }
                    if planes.is_empty() {
                        return Some((key, [0; 4], [f64::INFINITY, f64::NEG_INFINITY], [None; 4]));
                    }
                } else {
                    for (row, shared) in rows.iter_mut().zip(&on.shared) {
                        *row = p.key_plane(shared, &self.map, world[0], 0.0, region);
                    }
                    // Where the camera keys its first identity's plane nowhere (the eye in it),
                    // it is read on none: its own key alone, as against what shares none.
                    if rows[0].is_none() {
                        return Some((key, [0; 4], [f64::INFINITY, f64::NEG_INFINITY], [None; 4]));
                    }
                }
                let departing = |plane: &WorldPlane, places: &[Affine; 2], points: &[[f64; 3]]| {
                    let normal = plane.unrounded.normal;
                    crate::depth::snap_departure(grid, normal, places, points)
                        + crate::depth::departure(plane, world)
                };
                let snap = match placed {
                    // On axis planes alone: the snap's departure along each, the most of them.
                    Some((places, points)) if !planes.is_empty() => {
                        let most = planes
                            .iter()
                            .map(|plane| {
                                p.key_reach(plane, &self.map, departing(plane, places, points))
                            })
                            .fold(0.0, f64::max);
                        [most, f64::NEG_INFINITY]
                    }
                    Some(_) if on.ids[0] != 0 && !on.offs.is_empty() => {
                        let most = |k: usize| {
                            on.shared
                                .iter()
                                .zip(&on.offs)
                                .map(|(shared, off)| p.key_reach(shared, &self.map, off[k]))
                                .fold(0.0, f64::max)
                        };
                        [most(0), most(1)]
                    }
                    Some((places, points)) if on.ids[0] != 0 => {
                        let plane = on.plane?;
                        let off = match on.planarity {
                            Planarity::Newell | Planarity::Widest | Planarity::Axis(_) => {
                                departing(&plane, places, points)
                            }
                            _ => 0.0,
                        };
                        [p.key_reach(&plane, &self.map, off), f64::NEG_INFINITY]
                    }
                    _ => [f64::INFINITY, f64::NEG_INFINITY],
                };
                Some((key, ids, snap, rows))
            });
        let registered = fill.map_or([0; 4], |(_, ids, _, _)| ids);
        let snap = fill.map_or([f64::INFINITY, f64::NEG_INFINITY], |(_, _, snap, _)| snap);
        let rows = fill.map_or([None; 4], |(_, _, _, rows)| rows);
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
            fill.and_then(|(key, _, _, _)| key)
                .map_or(curve, LayerDepth::Plane),
            registered,
            snap,
            rows,
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
    /// registered planes `on.0` with its key's shares `on.1` ([`Fragment::snap`] and
    /// [`Fragment::wide`]), its keys on their planes `on.2`, the registry's handle of its flat
    /// record `on.3` ([`Fragment::record`]) and its own share of its key with its curve's shift on
    /// screen `on.4` ([`Fragment::own`], [`Layer::shift`]); `on_fill` for a stroke drawn after its
    /// path's fill and keyed on the curve that fill is read on ([`Layer::on_fill`]).
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        rings: &[Vec<[f64; 2]>],
        spec: PaintSpec,
        depth: LayerDepth,
        on: On,
        order: u32,
        sub: u32,
        on_fill: bool,
    ) {
        let (plane, [snap, wide], on_plane, record, [own, shift]) = on;
        let edges = scanline::edges_of(rings);
        if edges.is_empty() {
            return;
        }
        // Its whole outline, which a layer keyed on a sloped plane reads its fringe's depth on and
        // the labels' planner tests a centre on, on whatever the layer is keyed.
        let flats = flats_of(rings);
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
            wide,
            own,
            shift,
            on_fill,
            on_plane,
            record,
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
                // Never clear: [`Lowering::layers`] drops a clear one ([`PaintSpec::clear`]).
                match gradient::source(&ctm, p1, p2, &stops, extents) {
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

    /// The lowered paths, ready to draw, their flat paths' layers carrying their records as
    /// `registry`, the frame's, settled, holds them ([`Fragment::record`]).
    pub fn finish(self, registry: &Arc<Registry>) -> Paths {
        Paths {
            records: Some(Arc::clone(registry)),
            ..self.paths
        }
    }
}

impl Paths {
    /// Whether nothing was lowered.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// What the labels' planner reads of each layer (`label_plan.rs`).
    pub(crate) fn layers(&self) -> impl Iterator<Item = LayerView<'_>> {
        self.layers.iter().map(|l| LayerView {
            order: l.order,
            sub: l.sub,
            bbox: l.bbox,
            edges: &l.edges,
            flats: &l.flats,
            plane: match l.depth {
                LayerDepth::Plane(p) => Some(p),
                LayerDepth::Curve(..) => None,
            },
            layer: l,
        })
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

    /// Layer `layer`'s own share of its key at pixel `(x, y)` ([`Fragment::own`]): for a stroke of
    /// a path with no plane of its own keyed on its curve, its path's share of the snap's move of
    /// its keys at fixed points of the curve, and how far, as a share of the least of them, any of
    /// the pixel's reads may move beyond that where the snap moves the curve's points on screen
    /// by its shift ([`Centreline::slides_among`]); the layer's own for any other.
    fn own_at(&self, layer: &Layer, x: u32, y: u32) -> f64 {
        let LayerDepth::Curve(i, body) = layer.depth else {
            return layer.own;
        };
        if !(0.0..f64::INFINITY).contains(&layer.own) {
            return layer.own;
        }
        let curve = &self.centrelines[i as usize];
        let centre = [x as f64 + 0.5, y as f64 + 0.5];
        // The pixel's reads: its sub-samples, then its centre, as `Depths::of` makes them.
        let points: [[f64; 2]; SAMPLES + 1] = std::array::from_fn(|k| {
            if k < SAMPLES {
                sample_point(x, y, k)
            } else {
                centre
            }
        });
        // The segments the pixel's reads are made among for a body `wide`, cut as `point_depths`
        // cuts them: with `wide` past the body by the shift, every segment that may hold a point;
        // past the centre's nearest by the spread and the shift twice, every one near a point's
        // nearest.
        let among = |wide: f64| {
            let mut near = Vec::new();
            if !curve.near_cut(centre, &points, wide, &mut near) {
                curve.candidates(x, y, wide, &mut near);
                curve.line.cut(centre, &points, &mut near, wide);
            }
            near
        };
        let room = curve.line.rounding(centre[0].max(centre[1]) + body + 2.0);
        let reach = layer.shift + curve.line.slack_bound(layer.shift) + room;
        let mut near = among(body + 2.0 * reach);
        let nearest = near
            .iter()
            .map(|&s| curve.line.distance2(s as usize, centre))
            .fold(f64::INFINITY, f64::min)
            .sqrt();
        let wide = nearest + 0.75 + 3.0 * reach;
        if wide > body + 2.0 * reach {
            near = among(wide);
        }
        let (most, least) = curve.line.slides_among(
            &points,
            centre,
            &near,
            body,
            layer.shift,
            layer.own,
            layer.on_fill,
        );
        if least > 0.0 {
            layer.own + most / least * (1.0 + 4.0 * crate::composite::U)
        } else {
            f64::INFINITY
        }
    }

    /// Rasterizes the opaque or the other layers of the tile at `rect`.
    fn raster<'a>(&'a self, rect: Rect, opaque: bool, mut add: impl FnMut(u32, u32, Fragment<'a>)) {
        let mut scratch = Scratch::default();
        let mut near = Vec::new();
        let records = self.records.as_deref();
        for &i in &self.bins[tile_index(self.width, rect.x0, rect.y0)] {
            let layer = &self.layers[i as usize];
            if layer.opaque != opaque {
                continue;
            }
            let record = layer.record.and_then(|handle| records?.flat_record(handle));
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
                        f.record = record.map(FlatRecord);
                        f.wide = layer.wide;
                        f.own = self.own_at(layer, x, y);
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
    /// centre ([`Self::near`]) where they hold every segment such a read may take; else enough of
    /// the others that a read among them picks as a read among every segment does
    /// ([`Self::around`]).
    ///
    /// A segment's bounds lie no farther from a point than the segment, on each axis, so the
    /// bounds widened by the margin hold the centre for every segment within the margin of it, on
    /// each axis: [`Self::near`] holds them all. A read at a point of the pixel takes a segment
    /// holding the point within `body`, or its first nearest, which lies no farther from the point
    /// than the centre's nearest does, within the centre's nearest distance and how far the point
    /// lies from the centre (under 0.75): such a segment lies within
    /// `max(body, nearest + 0.75)` of a point of the pixel, the reach [`Self::around`] keeps, so
    /// within that and half a pixel of the centre on each axis. The nearest among those found is no
    /// nearer than the centre's own. So where that reach and half a pixel lie under the margin,
    /// with room for their rounding, the segments found hold every one a read may take, in order.
    fn candidates(&self, x: u32, y: u32, body: f64, near: &mut Vec<u32>) {
        self.near(x, y, near);
        if !near.is_empty() {
            let centre = [x as f64 + 0.5, y as f64 + 0.5];
            let nearest = near
                .iter()
                .map(|&k| self.line.distance2(k as usize, centre))
                .fold(f64::INFINITY, f64::min)
                .sqrt();
            let reach = (nearest + 0.75).max(body) + 0.5;
            let room = self.line.rounding(reach + centre[0].max(centre[1]));
            // Answered only under the margin; never at or over it, or where it is not a number.
            if (reach + room).partial_cmp(&self.margin) == Some(std::cmp::Ordering::Less) {
                return;
            }
            near.clear();
        }
        self.around(x, y, body, near);
    }

    /// The segments that may lie nearest a point of pixel `(x, y)` or hold one within `body` of
    /// them, in order, into `out` (empty), where the segments whose bounds hold the pixel's centre
    /// may not be all of them ([`Self::candidates`]): those
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
    fn opaque<'a>(&'a self, pass: &mut OpaquePass<'a>) {
        self.raster(pass.rect(), true, |x, y, f| pass.add(x, y, f));
    }

    fn transparent<'a>(&'a self, pass: &mut TransparentPass<'a>) {
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

    fn plane_depths(
        &self,
        id: u32,
        identity: u32,
        x: u32,
        y: u32,
        out: &mut [f64; SAMPLES],
    ) -> Option<f64> {
        let layer = &self.layers[id as usize];
        let k = layer.plane.iter().position(|p| *p == identity)?;
        let plane = layer.on_plane[k]?;
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

    /// A camera looking down z, its eye on the z axis; a face on the plane `x = 0`, which holds
    /// the eye, so the camera keys that plane nowhere; and a sliver a quarter grid step wide lying
    /// on it, filled, its points on a line to within the grid and coded on the axis plane of z:
    /// the sliver, with no plane of its own, lies on the face's identity, and its fill is keyed
    /// along its curve and on no identity, as it is with the face not drawn (then on its axis
    /// plane's identity alone).
    #[test]
    fn a_planeless_sliver_on_an_identity_whose_plane_holds_the_eye_is_keyed_along_its_curve() {
        let grid = 1.0 / 16384.0;
        let mut view = crate::probes::view(854, 480, 0.0, -std::f64::consts::FRAC_PI_2);
        view.input_grid = grid;
        let ring = |c: &[[f64; 3]]| -> Vec<[f64; 3]> {
            let mut out = Vec::new();
            for (i, a) in c.iter().enumerate() {
                let b = c[(i + 1) % c.len()];
                let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
                out.extend([*a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
            }
            out
        };
        let face = ring(&[
            [0.0, -0.3, -0.3],
            [0.0, 0.3, -0.3],
            [0.0, 0.3, 0.3],
            [0.0, -0.3, 0.3],
        ]);
        let w = grid / 4.0;
        let sliver = ring(&[
            [0.0, -0.2, 0.1],
            [w, -0.2, 0.1],
            [w, 0.2, 0.1],
            [0.0, 0.2, 0.1],
        ]);
        let record = |slot: u32, planes: Planarity| {
            let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
            r.paint.fill = Paint::One([0.2, 0.4, 0.8, 1.0]);
            r.depth.space = Space::World;
            r.planes = Some(vec![planes.code()]);
            r
        };
        let (face_record, sliver_record) =
            (record(1, Planarity::Newell), record(2, Planarity::Axis(2)));
        fn draw<'a>(record: &'a Record, points: &[[f64; 3]], order: u32) -> PathDraw<'a> {
            PathDraw {
                record,
                points: std::borrow::Cow::Owned(points.to_vec()),
                place: IDENTITY,
                places: [IDENTITY, IDENTITY],
                fill: Some(vec![[0.2, 0.4, 0.8, 1.0]]),
                stroke: None,
                background_stroke: None,
                ends: None,
                surface: false,
                pre_projected: false,
                pivot: None,
                stroke_width: 0.0,
                background_stroke_width: 0.0,
                order,
                sub: 0,
                which: 0,
            }
        }
        // The sliver's identities as keyed, its fill layer's key and identities, and whether the
        // camera keys its first identity's plane.
        let lowered = |with_face: bool| {
            let mut draws = Vec::new();
            if with_face {
                draws.push(draw(&face_record, &face, 0));
            }
            draws.push(draw(&sliver_record, &sliver, 1));
            let mut registry = Registry::new(&view);
            let mut lowering = Lowering::new(&view, 854, 480);
            let keyed = lowering.coplanar(&draws, &mut registry);
            let on = keyed
                .last()
                .cloned()
                .flatten()
                .expect("the sliver is keyed");
            let p = lowering.projection.clone().expect("a 3D camera");
            let region = [0.0, 0.0, 854.0, 480.0];
            let keys = with_face
                && p.key_plane(
                    &registry.plane(on.ids[0]),
                    &lowering.map,
                    sliver[0],
                    0.0,
                    region,
                )
                .is_some();
            for (d, k) in draws.iter().zip(keyed) {
                lowering.path(d, k);
            }
            let fill = lowering
                .paths
                .layers
                .iter()
                .find(|l| l.order == 1 && l.sub == 1)
                .map(|l| match l.depth {
                    LayerDepth::Plane(key) => (Some(key), l.plane),
                    LayerDepth::Curve(..) => (None, l.plane),
                })
                .expect("the sliver's fill");
            (on, fill, keys)
        };
        let (on, (key, ids), keys) = lowered(true);
        assert!(
            on.ids[0] != 0 && on.shared.len() == 1,
            "the sliver lies on {:?}",
            on.ids
        );
        assert!(!keys, "the camera keys the face's plane");
        let (alone, (key_alone, _), _) = lowered(false);
        assert!(
            alone.offs.is_empty() && alone.shared.len() == 1,
            "with the face not drawn, on its axis plane alone: {:?}",
            alone.ids
        );
        assert!(
            key.is_none() && key_alone.is_none(),
            "{key:?} against {key_alone:?}"
        );
        assert_eq!(ids, [0; 4]);
    }

    /// The camera of the row above, its face on the plane `x = 0`, which holds the eye, and a
    /// smaller face, filled and coded Newell, turned off that plane about y by 5e-5 (its points
    /// within a quarter grid step of it): a face with area, flat, on the first one's identity,
    /// whose own plane the camera keys, nearly edge-on. The first face is the larger so that it
    /// founds the identity on its plane (the registry founds identities from the least spread
    /// first). The turned face's fill keeps its own key, its own plane's, on that identity, whose
    /// plane the camera keys nowhere, and as it does with the first face not drawn.
    #[test]
    fn a_flat_fill_on_an_identity_whose_plane_holds_the_eye_keeps_its_own_key() {
        let grid = 1.0 / 16384.0;
        let mut view = crate::probes::view(854, 480, 0.0, -std::f64::consts::FRAC_PI_2);
        view.input_grid = grid;
        let ring = |c: &[[f64; 3]]| -> Vec<[f64; 3]> {
            let mut out = Vec::new();
            for (i, a) in c.iter().enumerate() {
                let b = c[(i + 1) % c.len()];
                let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
                out.extend([*a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
            }
            out
        };
        let face = ring(&[
            [0.0, -0.4, -0.4],
            [0.0, 0.4, -0.4],
            [0.0, 0.4, 0.4],
            [0.0, -0.4, 0.4],
        ]);
        let off = 0.3 * 5e-5;
        let turned = ring(&[
            [-off, -0.3, -0.3],
            [-off, 0.3, -0.3],
            [off, 0.3, 0.3],
            [off, -0.3, 0.3],
        ]);
        let record = |slot: u32| {
            let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
            r.paint.fill = Paint::One([0.2, 0.4, 0.8, 1.0]);
            r.depth.space = Space::World;
            r.planes = Some(vec![Planarity::Newell.code()]);
            r
        };
        let (face_record, turned_record) = (record(1), record(2));
        fn draw<'a>(record: &'a Record, points: &[[f64; 3]], order: u32) -> PathDraw<'a> {
            PathDraw {
                record,
                points: std::borrow::Cow::Owned(points.to_vec()),
                place: IDENTITY,
                places: [IDENTITY, IDENTITY],
                fill: Some(vec![[0.2, 0.4, 0.8, 1.0]]),
                stroke: None,
                background_stroke: None,
                ends: None,
                surface: false,
                pre_projected: false,
                pivot: None,
                stroke_width: 0.0,
                background_stroke_width: 0.0,
                order,
                sub: 0,
                which: 0,
            }
        }
        // Each face as keyed, the turned face's fill layer's key and identities, and whether the
        // camera keys the plane of its first identity.
        let lowered = |with_face: bool| {
            let mut draws = Vec::new();
            if with_face {
                draws.push(draw(&face_record, &face, 0));
            }
            draws.push(draw(&turned_record, &turned, 1));
            let mut registry = Registry::new(&view);
            let mut lowering = Lowering::new(&view, 854, 480);
            let keyed = lowering.coplanar(&draws, &mut registry);
            let on: Vec<Keyed> = keyed.iter().cloned().map(|k| k.expect("keyed")).collect();
            let p = lowering.projection.clone().expect("a 3D camera");
            let region = [0.0, 0.0, 854.0, 480.0];
            let last = on.last().expect("the turned face");
            let keys = last.ids[0] != 0
                && p.key_plane(
                    &registry.plane(last.ids[0]),
                    &lowering.map,
                    turned[0],
                    0.0,
                    region,
                )
                .is_some();
            let own = last
                .plane
                .and_then(|plane| p.key_plane(&plane, &lowering.map, turned[0], 0.0, region));
            for (d, k) in draws.iter().zip(keyed) {
                lowering.path(d, k);
            }
            let fill = lowering
                .paths
                .layers
                .iter()
                .find(|l| l.order == 1 && l.sub == 1)
                .map(|l| match l.depth {
                    LayerDepth::Plane(key) => (Some(key), l.plane),
                    LayerDepth::Curve(..) => (None, l.plane),
                })
                .expect("the turned face's fill");
            (on, fill, keys, own)
        };
        let (on, (key, ids), keys, own) = lowered(true);
        assert!(
            on.iter().all(|k| !k.planeless && k.record.is_some()),
            "both faces flat"
        );
        assert!(
            on[1].ids[0] != 0 && on[1].ids[0] == on[0].ids[0],
            "the turned face lies on {:?}, the face on {:?}",
            on[1].ids,
            on[0].ids
        );
        assert!(!keys, "the camera keys the identity's plane");
        assert!(
            own.is_some(),
            "the camera keys the turned face's own plane nowhere"
        );
        let (_, (key_alone, _), _, _) = lowered(false);
        let terms = |k: Option<Plane>| k.map(|k| [k.a, k.b, k.c]);
        assert!(
            key.is_some() && terms(key) == terms(own) && key == key_alone,
            "{key:?} against its own {own:?} and alone {key_alone:?}"
        );
        assert_eq!(ids[0], on[1].ids[0]);
    }

    /// The camera looking down z: a point at depth `z` keyed `-20 / (20 - z)`.
    fn down_z() -> Projection {
        Projection::new(
            &ThreeDCamera {
                phi: 0.0,
                theta: -std::f64::consts::FRAC_PI_2,
                gamma: 0.0,
                focal_distance: 20.0,
                zoom: 1.0,
                exponential_projection: false,
                light_source: [0.0; 3],
                shading_factor: 0.2,
                should_apply_shading: true,
            },
            [0.0; 3],
        )
    }

    /// The straight cubic from `a` to `b` in CE's layout, its handles at its thirds, depth alike.
    fn straight(a: [f64; 3], b: [f64; 3]) -> [[f64; 3]; 4] {
        std::array::from_fn(|k| {
            let t = k as f64 / 3.0;
            std::array::from_fn(|c| a[c] + (b[c] - a[c]) * t)
        })
    }

    /// The points of a grid from `(x0, y0)` to `(x1, y1)`, `step` apart.
    fn grid(x0: f64, x1: f64, y0: f64, y1: f64, step: f64) -> Vec<[f64; 2]> {
        let nx = ((x1 - x0) / step).round() as usize;
        let ny = ((y1 - y0) / step).round() as usize;
        (0..=nx)
            .flat_map(|i| (0..=ny).map(move |j| [x0 + i as f64 * step, y0 + j as f64 * step]))
            .collect()
    }

    /// Over the points `at`, the worst move over reach of the key read on the centreline
    /// `curve_depth` makes of `drawn` (through `map`, keyed by [`down_z`]) to the one it makes of
    /// each of `contents` (each point within `shift` of `drawn`'s on screen, keyed within `share`),
    /// against the bound read on the drawn one at `body`; with `fill`, every way a stroke on its
    /// own fill allows (its read, or the nearer of it and the fill's, on each side). The ratio,
    /// where, the move and the reach.
    #[allow(clippy::too_many_arguments)]
    fn worst_over_reach(
        drawn: &[[f64; 3]],
        contents: &[Vec<[f64; 3]>],
        map: &FrameMap,
        at: &[[f64; 2]],
        body: f64,
        shift: f64,
        share: f64,
        fill: bool,
    ) -> (f64, [f64; 2], f64, f64) {
        let p = down_z();
        for c in contents {
            assert_eq!(c.len(), drawn.len());
            for (q, d) in c.iter().zip(drawn) {
                let (a, b) = (map.apply(*q), map.apply(*d));
                let (kq, kd) = (p.factor(q[2]), p.factor(d[2]));
                assert!(
                    libm::hypot(a[0] - b[0], a[1] - b[1]) <= shift
                        && (kq - kd).abs() <= share * kd.abs()
                );
            }
        }
        let line = curve_depth(drawn, map, &p);
        let lines: Vec<Centreline> = contents.iter().map(|c| curve_depth(c, map, &p)).collect();
        let resolved = |l: &Centreline, q: [f64; 2]| {
            let (s, es) = l.key_under(q, &[], body);
            if fill {
                let (f, ef) = l.key_under(q, &[], 0.0);
                (vec![s, s.min(f)], es.max(ef))
            } else {
                (vec![s], es)
            }
        };
        let mut worst = (0.0f64, [0.0; 2], 0.0, 0.0);
        for &q in at {
            let (before, e0) = resolved(&line, q);
            let (slide, magnitude) = line.slide_under(q, &[], body, shift, share, fill);
            let own = share + slide / magnitude * (1.0 + 4.0 * crate::composite::U);
            for l in &lines {
                let (after, e1) = resolved(l, q);
                for &k0 in &before {
                    for &k1 in &after {
                        let reach = e0 + e1 + (k0.abs() + e0) * own;
                        let moved = (k1 - k0).abs();
                        if moved / reach > worst.0 {
                            worst = (moved / reach, q, moved, reach);
                        }
                    }
                }
            }
        }
        worst
    }

    /// Two straight cubics on one line, keyed along it, their shared anchor snapped to one point
    /// while the content holds the first cubic's end and the second's first point up to twice the
    /// shift apart, within CE's subpath tolerance: the content draws the second cubic's first
    /// piece from the first one's end, a point of its own, so the piece's ends move apart by up to
    /// twice the shift whatever its weights (its end moves with its cubic's points, its start with
    /// the other cubic's). The second cubic is long, so its first piece's weights hold its ends
    /// together closely: 36 px, its piece count on a step the content crosses, and 36.9 px, on
    /// none. Read across that piece, a stroke 6 px wide.
    #[test]
    fn at_a_cubic_boundary_snapped_to_one_point_the_next_piece_s_ends_move_apart_on_their_own() {
        let map = FrameMap::cairo(1280, 720, 1280.0 / 90.0, 8.0, [0.0; 3]);
        let delta = 1.8e-5;
        let shift = delta * map.xx * (1.0 + 1e-6);
        let at = |y: f64| [6.0, y, 5.0 * (y + 0.3)];
        let mut failed = Vec::new();
        for end in [0.4, 0.41] {
            let drawn = [straight(at(-0.3), at(0.0)), straight(at(0.0), at(end))].concat();
            let moved = |e: [f64; 2], s: [f64; 2]| {
                let mut c = drawn.clone();
                c[3] = [c[3][0] + e[0], c[3][1] + e[1], c[3][2]];
                c[4] = [c[4][0] + s[0], c[4][1] + s[1], c[4][2]];
                c
            };
            let mut contents = vec![
                moved([delta, 0.0], [-delta, 0.0]),
                moved([-delta, 0.0], [delta, 0.0]),
            ];
            assert!(contents.iter().all(|c| subpaths(c).len() == 1));
            let mut seeded = Seeded(0x6a09_e667_f3bc_c908);
            for _ in 0..16 {
                let mut turn =
                    || (seeded.next() >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
                let a = turn();
                let b = turn();
                contents.push(moved(
                    [delta * libm::cos(a), delta * libm::sin(a)],
                    [delta * libm::cos(b), delta * libm::sin(b)],
                ));
            }
            let v = map.apply(at(0.0));
            let points = grid(v[0] - 3.0, v[0] + 3.0, v[1] - 2.5, v[1] + 0.5, 0.1);
            let (ratio, p, moved_by, reach) =
                worst_over_reach(&drawn, &contents, &map, &points, 3.0, shift, 0.0, false);
            let px = end * 90.0;
            eprintln!(
                "TIGHT cubic boundary snapped to one point, {px:.1} px: {ratio:.3} at worst over {} \
                 points, at {p:?}",
                points.len()
            );
            if ratio > 1.0 {
                failed.push(format!(
                    "{px:.1} px, at {p:?}: the key moved by {moved_by:e} against its reach {reach:e}"
                ));
            }
        }
        assert!(failed.is_empty(), "snapped to one point: {failed:?}");
    }

    /// The worst move over reach, over a stroke 6 px wide read across the straight cubic `drawn`
    /// (on screen along y), of the contents with its ends moved apart and together along it by
    /// `delta` and of 16 seeded moves of every point by `delta`.
    fn step_row(drawn: &[[f64; 3]], delta: f64, seed: u64) -> (f64, [f64; 2], f64, f64) {
        step_row_with(drawn, delta, seed, Vec::new())
    }

    /// [`step_row`] with the contents `more` beside its own.
    fn step_row_with(
        drawn: &[[f64; 3]],
        delta: f64,
        seed: u64,
        mut contents: Vec<Vec<[f64; 3]>>,
    ) -> (f64, [f64; 2], f64, f64) {
        let map = FrameMap::cairo(1280, 720, 1280.0 / 90.0, 8.0, [0.0; 3]);
        let shift = delta * map.xx * (1.0 + 1e-6);
        for s in [-1.0, 1.0] {
            let mut c = drawn.to_vec();
            c[0][1] -= s * delta;
            c[3][1] += s * delta;
            contents.push(c);
        }
        let mut seeded = Seeded(seed);
        for _ in 0..16 {
            let mut c = drawn.to_vec();
            for q in &mut c {
                let a = (seeded.next() >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
                q[0] += delta * libm::cos(a);
                q[1] += delta * libm::sin(a);
            }
            contents.push(c);
        }
        let points = across(drawn, &map);
        worst_over_reach(drawn, &contents, &map, &points, 3.0, shift, 0.0, false)
    }

    /// The points a stroke 6 px wide is read at across `drawn` (on screen along y): 0.1 px apart,
    /// 3 px either side of its first point and 0.5 px past its points' ends.
    fn across(drawn: &[[f64; 3]], map: &FrameMap) -> Vec<[f64; 2]> {
        let ys: Vec<f64> = drawn.iter().map(|q| map.apply(*q)[1]).collect();
        let x = map.apply(drawn[0])[0];
        let (y0, y1) = ys
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, m), y| {
                (l.min(*y), m.max(*y))
            });
        grid(x - 3.0, x + 3.0, y0 - 0.5, y1 + 0.5, 0.1)
    }

    /// A cubic straight on screen in a plane through the eye, 5 to 15 deep before a camera 20 away
    /// (keyed `-4/3` to `-4`) over 36 px: its key bends far off its chord, and its piece count
    /// sits on a step (at 36 px, and 1.5 shifts short of it) that the content crosses, cutting it
    /// into one piece more or fewer. Within the reach its pieces take from the cut a step away.
    #[test]
    fn at_a_piece_count_step_a_depth_curved_cubic_s_read_moves_within_its_reach() {
        let delta = 1.8e-5;
        let mut failed = Vec::new();
        for short in [0.0, 1.5] {
            let drawn = straight([6.0, 0.0, 5.0], [6.0, 0.4 - short * delta, 15.0]);
            let (ratio, p, moved_by, reach) = step_row(&drawn, delta, 0xbb67_ae85_84ca_a73b);
            eprintln!(
                "TIGHT piece-count step, depth-curved cubic {short} shifts short: {ratio:.3} at \
                 worst, at {p:?}"
            );
            if ratio > 1.0 {
                failed.push(format!(
                    "{short} shifts short, at {p:?}: the key moved by {moved_by:e} against its \
                     reach {reach:e}"
                ));
            }
        }
        assert!(failed.is_empty(), "{failed:?}");
    }

    /// The depth-curved cubic of the row above folding back on its chord, its control points 0,
    /// 24, -4 and 20 px along it (76 px of control polygon, on a step): no single place along the
    /// chord holds one point of each cut, so its pieces take the band.
    #[test]
    fn at_a_piece_count_step_a_cubic_folding_back_on_its_chord_takes_the_band() {
        let delta = 1.8e-5;
        let at = |px: f64, z: f64| [6.0, px / 90.0, z];
        let drawn = [
            at(0.0, 5.0),
            at(24.0, 5.0 + 10.0 / 3.0),
            at(-4.0, 5.0 + 20.0 / 3.0),
            at(20.0, 15.0),
        ];
        let (ratio, p, moved_by, reach) = step_row(&drawn, delta, 0x3c6e_f372_fe94_f82b);
        eprintln!("TIGHT piece-count step, folding back: {ratio:.3} at worst, at {p:?}");
        assert!(
            ratio <= 1.0,
            "at {p:?}: the key moved by {moved_by:e} against its reach {reach:e}"
        );
    }

    /// The folding cubic of the row above 5.5 shifts short of its step, its points also moved
    /// back and forth along the chord in turn: each leg then lengthens by twice the shift, the
    /// polygon by six shifts, across the step. The step is charged within that window.
    #[test]
    fn at_a_piece_count_step_a_cubic_folding_back_5_5_shifts_short_takes_the_band() {
        let delta = 1.8e-5;
        let at = |px: f64, z: f64| [6.0, px / 90.0, z];
        let drawn = [
            at(0.0, 5.0),
            at(24.0, 5.0 + 10.0 / 3.0),
            at(-4.0, 5.0 + 20.0 / 3.0),
            [6.0, 20.0 / 90.0 - 5.5 * delta, 15.0],
        ];
        let turns = [-1.0, 1.0].map(|s| {
            let mut c = drawn.to_vec();
            for (j, q) in c.iter_mut().enumerate() {
                let sign = if j % 2 == 0 { -s } else { s };
                q[1] += sign * delta;
            }
            c
        });
        let (ratio, p, moved_by, reach) =
            step_row_with(&drawn, delta, 0xa54f_f53a_5f1d_36f1, turns.to_vec());
        eprintln!(
            "TIGHT piece-count step, folding back 5.5 shifts short: {ratio:.3} at worst, at {p:?}"
        );
        assert!(
            ratio <= 1.0,
            "at {p:?}: the key moved by {moved_by:e} against its reach {reach:e}"
        );
    }

    /// The depth-curved cubic at its step, its interior control points 2.5 shifts off its chord
    /// on screen: within the gate (`(2 + 2 |u'|) shift`, about 3.3 shifts here), so its pieces
    /// still take the step.
    #[test]
    fn at_a_piece_count_step_a_depth_curved_cubic_bent_within_the_gate_takes_the_step() {
        let delta = 1.8e-5;
        let mut drawn = straight([6.0, 0.0, 5.0], [6.0, 0.4, 15.0]);
        for q in &mut drawn[1..3] {
            q[0] += 2.5 * delta;
        }
        let (ratio, p, moved_by, reach) = step_row(&drawn, delta, 0x510e_527f_ade6_82d1);
        eprintln!("TIGHT piece-count step, bent 2.5 shifts: {ratio:.3} at worst, at {p:?}");
        assert!(
            ratio <= 1.0,
            "at {p:?}: the key moved by {moved_by:e} against its reach {reach:e}"
        );
    }

    /// A cubic whose chord may turn freely (0.0009 px, under four shifts): its interior control
    /// points 1 px off its ends, its control polygon 1.9991 px, by the step from one piece to two,
    /// keyed `-4/3` at its first point and on the line through its ends in place along the chord
    /// and key, 0.3 nearer the eye at its interior points. Drawn as one piece, it may be cut into
    /// two whose shared point lies at the far side of the loop: the read takes the span, though
    /// its band is none.
    #[test]
    fn at_a_piece_count_step_a_cubic_whose_chord_may_turn_freely_takes_its_span() {
        let delta = 1.8e-5;
        let at = |px: f64, key: f64| [6.0, px / 90.0, depth_of(key)];
        let (k0, chord) = (-4.0 / 3.0, 0.0009);
        let drawn = [
            at(0.0, k0),
            at(1.0, k0 - 0.3),
            at(1.0, k0 - 0.3),
            at(chord, k0 - 0.3 * chord),
        ];
        let mut out = Vec::new();
        for s in [-1.0, 1.0] {
            for j in [1, 2] {
                let mut c = drawn.to_vec();
                c[j][1] += s * delta;
                out.push(c);
            }
        }
        let (ratio, p, moved_by, reach) = step_row_with(&drawn, delta, 0x9b05_688c_2b3e_6c1f, out);
        eprintln!("TIGHT piece-count step, free chord: {ratio:.3} at worst, at {p:?}");
        assert!(
            ratio <= 1.0,
            "at {p:?}: the key moved by {moved_by:e} against its reach {reach:e}"
        );
    }

    /// The depth [`down_z`] keys `key` (negative).
    fn depth_of(key: f64) -> f64 {
        20.0 + 20.0 / key
    }

    /// A cubic `px` long on screen along y, its control points at its thirds and keyed linearly
    /// along it from `-1.2` to `-1.3`: collinear in place and key, as a cubic straight in the world
    /// before the eye draws under the default projection.
    fn collinear(px: f64) -> [[f64; 3]; 4] {
        std::array::from_fn(|j| {
            let t = j as f64 / 3.0;
            [6.0, px / 90.0 * t, depth_of(-1.2 - 0.1 * t)]
        })
    }

    /// The collinear cubic `px` long, by a step of its piece count, against its contents: its ends
    /// moved apart and together along it by `delta`, with its interior keys moved by `share` each
    /// way, its interior points by `delta` along it and across it each way; and 16 seeded moves of
    /// every point by `delta` and every key within `share`. A stroke 6 px wide read across it
    /// moves within its reach, and at each point of its middle its own share is at most 1.1 times
    /// the one it takes there `away` px long, off every step.
    fn collinear_row(px: f64, away: f64) {
        let map = FrameMap::cairo(1280, 720, 1280.0 / 90.0, 8.0, [0.0; 3]);
        let (delta, share, body) = (6e-5, 2e-6, 3.0);
        let shift = delta * map.xx * (1.0 + 1e-6);
        let key = |q: [f64; 3]| -down_z().factor(q[2]);
        let rekey = |q: &mut [f64; 3], s: f64| {
            q[2] = depth_of(key(*q) * (1.0 + s * share * (1.0 - 1e-6)));
        };
        let drawn = collinear(px);
        let signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
        let mut contents = Vec::new();
        for e in [-1.0, 1.0] {
            let mut ends = drawn.to_vec();
            ends[0][1] -= e * delta;
            ends[3][1] += e * delta;
            contents.push(ends.clone());
            for (s1, s2) in signs {
                let mut c = ends.clone();
                rekey(&mut c[1], s1);
                rekey(&mut c[2], s2);
                contents.push(c);
                for axis in [0, 1] {
                    let mut c = ends.clone();
                    c[1][axis] += s1 * delta;
                    c[2][axis] += s2 * delta;
                    contents.push(c);
                }
            }
        }
        let mut seeded = Seeded(0x1f83_d9ab_fb41_bd6b);
        for _ in 0..16 {
            let mut c = drawn.to_vec();
            for q in &mut c {
                let a = (seeded.next() >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
                q[0] += delta * libm::cos(a);
                q[1] += delta * libm::sin(a);
                rekey(q, (seeded.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0);
            }
            contents.push(c);
        }
        // The contents cross the step.
        let count = |c: &[[f64; 3]]| {
            let length: f64 = c
                .windows(2)
                .map(|w| {
                    let (a, b) = (map.apply(w[0]), map.apply(w[1]));
                    libm::hypot(b[0] - a[0], b[1] - a[1])
                })
                .sum();
            (length / 2.0).ceil()
        };
        assert!(contents.iter().any(|c| count(c) != count(&drawn[..])));
        let points = across(&drawn, &map);
        let (ratio, p, moved_by, reach) =
            worst_over_reach(&drawn, &contents, &map, &points, body, shift, share, false);
        // The own share at each point of the stroke's middle: a quarter of it from either end, 0.05
        // px or more from every piece's end at either length, and 0.5 px or more inside the body.
        let [x, top] = map.apply(drawn[0]);
        let bottom = map.apply(drawn[3])[1];
        let ends = |length: f64| {
            let n = (length / 2.0).ceil();
            (0..=n as usize).map(move |s| top - length * s as f64 / n)
        };
        let middle: Vec<[f64; 2]> = points
            .iter()
            .filter(|q| {
                (q[0] - x).abs() <= body - 0.5
                    && q[1] <= top - px / 4.0
                    && q[1] >= bottom + px / 4.0
                    && ends(px).chain(ends(away)).all(|y| (q[1] - y).abs() >= 0.05)
            })
            .copied()
            .collect();
        let own = |line: &Centreline, q: [f64; 2]| {
            let (slide, magnitude) = line.slide_under(q, &[], body, shift, share, false);
            share + slide / magnitude
        };
        let (on, off) = (
            curve_depth(&drawn, &map, &down_z()),
            curve_depth(&collinear(away), &map, &down_z()),
        );
        let (widened, at) = middle
            .iter()
            .map(|q| (own(&on, *q) / own(&off, *q), *q))
            .fold((0.0f64, [0.0; 2]), |w, r| if r.0 > w.0 { r } else { w });
        eprintln!(
            "TIGHT piece-count step, collinear {px} px: {ratio:.3} at worst, at {p:?}; own share \
             {widened:.3} times its own off the step at worst, at {at:?}"
        );
        assert!(
            ratio <= 1.0,
            "at {p:?}: the key moved by {moved_by:e} against its reach {reach:e}"
        );
        assert!(
            !middle.is_empty() && widened <= 1.1,
            "own share {widened:.3} times its own off the step, at {at:?}"
        );
    }

    /// The collinear cubic 28.0099 px long, 0.0099 px past the step at 28 px: a world-straight
    /// axis before the eye draws one (the ThreeDParametricSpring example's x axis does), and a
    /// step charge there widened its own share past the curve in front of it.
    #[test]
    fn at_a_piece_count_step_a_cubic_collinear_in_place_and_key_keeps_its_own_share() {
        collinear_row(28.0099, 28.04);
    }

    /// The collinear row at the step from one piece to two, 2.0099 px long, where the two counts'
    /// cuts depart most.
    #[test]
    fn at_the_first_piece_count_step_a_cubic_collinear_in_place_and_key_keeps_its_own_share() {
        collinear_row(2.0099, 2.04);
    }

    /// Two straight cubics on one line whose shared anchor's two points, the first cubic's end `E`
    /// and the second's first point `S`, stand 0.9e-5 |S| apart in one frame and 1.5e-5 |S| in
    /// the other, across CE's subpath tolerance (`1e-6 + 1e-5 |S|` on each axis): one frame joins
    /// the two, drawing the second cubic's first piece from `E`, and the other splits them,
    /// drawing it from `S`. Drawn joined and content split, and drawn split and content joined:
    /// the piece's start moves by the shift and the gap the joined frame closes. The second cubic
    /// is short (two pieces), so its first piece is too.
    #[test]
    fn at_a_cubic_boundary_across_the_subpath_tolerance_the_piece_starts_where_either_frame_draws_it()
     {
        let map = FrameMap::cairo(1280, 720, 1280.0 / 90.0, 8.0, [0.0; 3]);
        let b = 6.0;
        let delta = 0.3e-5 * b;
        let shift = delta * map.xx * (1.0 + 1e-6);
        let at = |x: f64, y: f64| [x, y, 5.0 * (y + 0.3)];
        let mut failed = Vec::new();
        for (name, drawn_gap, content_gap) in [
            ("joined, content split", 0.9e-5, 1.5e-5),
            ("split, content joined", 1.5e-5, 0.9e-5),
        ] {
            let drawn = [
                straight(at(b, -0.3), at(b - drawn_gap * b, 0.0)),
                straight(at(b, 0.0), at(b, 0.025)),
            ]
            .concat();
            let apart = if content_gap > drawn_gap { 1.0 } else { -1.0 };
            let mut content = drawn.clone();
            content[3][0] -= apart * delta;
            content[4][0] += apart * delta;
            let joined = |c: &[[f64; 3]]| subpaths(c).len() == 1;
            assert_eq!(
                (joined(&drawn), joined(&content)),
                (drawn_gap < content_gap, drawn_gap > content_gap),
                "{name}"
            );
            let v = map.apply(at(b, 0.0));
            let points = grid(v[0] - 3.0, v[0] + 3.0, v[1] - 2.0, v[1] + 0.5, 0.05);
            let (ratio, p, moved_by, reach) =
                worst_over_reach(&drawn, &[content], &map, &points, 3.0, shift, 0.0, false);
            eprintln!(
                "TIGHT cubic boundary {name}: {ratio:.3} at worst over {} points, at {p:?}",
                points.len()
            );
            failed.extend((ratio > 1.0).then(|| {
                format!(
                    "{name}, at {p:?}: the key moved by {moved_by:e} against its reach {reach:e}"
                )
            }));
        }
        assert!(failed.is_empty(), "{failed:?}");
    }

    /// The corner of the depth tests' square (from (10, 13) down to (10, 10) and on to (13, 10),
    /// keyed as there), its two sides two straight cubics: the first cubic's end and the second's
    /// first point are two points of content, which the snap put together, and which the content
    /// holds farther apart than CE's subpath tolerance, splitting them. Read as a stroke on its own
    /// fill by the corner's outside, against the content of the case a partner across the
    /// boundary leaves open (the second cubic's first point moved toward the point read by the
    /// shift, the first cubic away along x by as much), the first cubic's end a third of the shift
    /// off, and seeded moves.
    #[test]
    fn at_a_cubic_boundary_a_filled_stroke_s_resolved_key_lies_within_its_reach() {
        let map = FrameMap {
            xx: 1.0,
            yy: 1.0,
            x0: 0.0,
            y0: 0.0,
        };
        let z = |k: f64| 20.0 + 20.0 / k;
        let (c, v, a) = (
            [10.0, 13.0, z(-1.0287)],
            [10.0, 10.0, z(-1.0300)],
            [13.0, 10.0, z(-1.0313)],
        );
        let drawn = [straight(c, v), straight(v, a)].concat();
        let (shift, share, body) = (4.2e-3, 2.9e-6, 0.15);
        let rim = shift * (1.0 - 1e-9);
        let target = [9.9, 10.04];
        let toward = {
            let (dx, dy) = (target[0] - v[0], target[1] - v[1]);
            let l = libm::hypot(dx, dy);
            [rim * dx / l, rim * dy / l]
        };
        let moved = |m: &dyn Fn(usize) -> [f64; 2]| -> Vec<[f64; 3]> {
            drawn
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let d = m(i);
                    [p[0] + d[0], p[1] + d[1], p[2]]
                })
                .collect()
        };
        let mut contents = vec![
            moved(&|i| match i {
                4 => toward,
                0..4 => [rim, 0.0],
                _ => [0.0, 0.0],
            }),
            moved(&|i| if i == 3 { [rim / 3.0, 0.0] } else { [0.0, 0.0] }),
        ];
        let mut seeded = Seeded(0x0c10_5e0f_f1e1_d5a1);
        for _ in 0..16 {
            let m: Vec<[f64; 2]> = (0..drawn.len())
                .map(|_| {
                    let a =
                        (seeded.next() >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
                    [rim * libm::cos(a), rim * libm::sin(a)]
                })
                .collect();
            contents.push(moved(&|i| m[i]));
        }
        assert!(
            subpaths(&drawn).len() == 1 && contents.iter().take(2).all(|c| subpaths(c).len() == 2)
        );
        let mut points = grid(9.85, 9.95, 10.0, 10.08, 0.005);
        points.push(target);
        let (ratio, p, moved_by, reach) =
            worst_over_reach(&drawn, &contents, &map, &points, body, shift, share, true);
        eprintln!(
            "TIGHT cubic boundary filled corner: {ratio:.3} at worst over {} points, at {p:?}",
            points.len()
        );
        assert!(
            ratio <= 1.0,
            "filled corner, at {p:?}: the resolved key moved by {moved_by:e} against its reach {reach:e}"
        );
    }

    /// The drawn-joined boundary of the tolerance row, its second cubic keyed apart from the first
    /// one's end, 0.05 nearer the eye (CE's tolerance does not read depth): the content, splitting
    /// the two, starts the piece at its own first point and key, so the start's key moves by the
    /// keys' distance.
    #[test]
    fn at_a_cubic_boundary_keyed_apart_the_piece_s_start_key_moves_by_the_keys_distance() {
        let map = FrameMap::cairo(1280, 720, 1280.0 / 90.0, 8.0, [0.0; 3]);
        let b = 6.0;
        let delta = 0.3e-5 * b;
        let shift = delta * map.xx * (1.0 + 1e-6);
        let at = |x: f64, y: f64, dz: f64| [x, y, 5.0 * (y + 0.3) + dz];
        let drawn = [
            straight(at(b, -0.3, 0.0), at(b - 0.9e-5 * b, 0.0, 0.0)),
            straight(at(b, 0.0, 0.05), at(b, 0.025, 0.05)),
        ]
        .concat();
        let mut content = drawn.clone();
        content[3][0] -= delta;
        content[4][0] += delta;
        assert!(subpaths(&drawn).len() == 1 && subpaths(&content).len() == 2);
        let v = map.apply(at(b, 0.0, 0.0));
        let points = grid(v[0] - 3.0, v[0] + 3.0, v[1] - 2.0, v[1] + 0.5, 0.05);
        let (ratio, p, moved_by, reach) =
            worst_over_reach(&drawn, &[content], &map, &points, 3.0, shift, 0.0, false);
        eprintln!(
            "TIGHT cubic boundary keyed apart: {ratio:.3} at worst over {} points, at {p:?}",
            points.len()
        );
        assert!(
            ratio <= 1.0,
            "keyed apart, at {p:?}: the key moved by {moved_by:e} against its reach {reach:e}"
        );
    }

    /// The drawn-joined boundary of the row above (the first cubic's end drawn 0.9e-5 |S| from the
    /// second's first point, the content splitting them), with a far part of the curve: a point
    /// outside the stroke's body whose nearest is that part, keyed far from the line, and whose
    /// distance from the boundary piece lies past the body by more than the shift and less than
    /// the shift and the gap. The content, starting the piece at `S`, brings it within the body,
    /// and the piece takes the read: only a test of which segments may hold a point that takes the
    /// gap beside the shift has the piece among them.
    #[test]
    fn at_a_cubic_boundary_a_piece_the_gap_brings_within_the_body_takes_the_read() {
        let map = FrameMap::cairo(1280, 720, 1280.0 / 90.0, 8.0, [0.0; 3]);
        let b = 6.0;
        let delta = 0.3e-5 * b;
        let shift = delta * map.xx * (1.0 + 1e-6);
        let body = 3.0;
        let at = |x: f64, y: f64| [x, y, 5.0 * (y + 0.3)];
        let line = [
            straight(at(b, -0.3), at(b - 0.9e-5 * b, 0.0)),
            straight(at(b, 0.0), at(b, 0.025)),
        ]
        .concat();
        let p = down_z();
        // The boundary piece is segment 14: the first cubic is 27 px long, 14 pieces.
        let alone = curve_depth(&line, &map, &p);
        let y = 359.7;
        let distance = |x: f64| alone.distance2(14, [x, y]).sqrt();
        let (mut lo, mut hi) = (1180.0, 1190.0);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if distance(mid) < body + 4e-3 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let q = [lo, y];
        // The far part: a straight cubic along y, 3.0004 px beyond the point, keyed -1.2.
        let far = (q[0] + body + 4e-4 - map.x0) / map.xx;
        let zf = 20.0 + 20.0 / -1.2;
        let drawn = [
            line.clone(),
            straight([far, 0.06, zf], [far, -0.06, zf]).to_vec(),
        ]
        .concat();
        let mut content = drawn.clone();
        content[3][0] -= delta;
        content[4][0] += delta;
        let (before, after) = (
            curve_depth(&drawn, &map, &p),
            curve_depth(&content, &map, &p),
        );
        let (k0, k1) = (
            before.key_under(q, &[], body).0,
            after.key_under(q, &[], body).0,
        );
        assert!(
            (k0 + 1.2).abs() < 1e-9 && (k1 - k0).abs() > 0.1,
            "the far part keys the drawn read ({k0}), the piece the content's ({k1})"
        );
        let (ratio, _, moved_by, reach) =
            worst_over_reach(&drawn, &[content], &map, &[q], body, shift, 0.0, false);
        eprintln!("TIGHT cubic boundary piece brought within the body: {ratio:.3} at {q:?}");
        assert!(
            ratio <= 1.0,
            "at {q:?}: the key moved by {moved_by:e} against its reach {reach:e}"
        );
    }
}

/// Reads through the frame's candidate paths (`Paths::depth_at` for a pixel's centre,
/// `Paths::point_depths` for its sub-samples), against the same read among every segment of the
/// curve, which needs no candidate set to be complete.
#[cfg(test)]
mod brute_force_reads {
    use super::*;
    use crate::composite::{SAMPLES, sample_point};

    type Key = dyn Fn([f64; 2]) -> f64;

    /// A straight CE edge from the last point of `out` to `b` (handles at its thirds), flattened as
    /// `curve_depth` flattens it: `ceil(len / 2)` pieces, 1 to 64, at even steps of its parameter.
    fn edge(out: &mut Vec<([f64; 2], f64)>, b: [f64; 2], key: &Key) {
        let a = out.last().expect("a start").0;
        let len = ((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])).sqrt();
        let n = (len / 2.0).ceil().clamp(1.0, 64.0) as usize;
        let c = [
            a,
            [a[0] + (b[0] - a[0]) / 3.0, a[1] + (b[1] - a[1]) / 3.0],
            [
                a[0] + (b[0] - a[0]) * 2.0 / 3.0,
                a[1] + (b[1] - a[1]) * 2.0 / 3.0,
            ],
            b,
        ];
        for s in 1..=n {
            let t = s as f64 / n as f64;
            let u = 1.0 - t;
            let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
            let p =
                [0, 1].map(|k| w[0] * c[0][k] + w[1] * c[1][k] + w[2] * c[2][k] + w[3] * c[3][k]);
            out.push((p, key(p)));
        }
    }

    fn polyline(corners: &[[f64; 2]], closed: bool, key: &Key) -> Centreline {
        let mut out = vec![(corners[0], key(corners[0]))];
        for c in &corners[1..] {
            edge(&mut out, *c, key);
        }
        if closed {
            edge(&mut out, corners[0], key);
        }
        Centreline::new(&out, &[])
    }

    fn inside(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
        let mut c = false;
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
            {
                c = !c;
            }
        }
        c
    }

    /// The first nearest of `among` (all segments when empty) to `p`, as `find` picks it, and its
    /// distance.
    fn nearest(line: &Centreline, among: &[u32], p: [f64; 2]) -> (usize, f64) {
        let all: Vec<u32> = (0..line.len() as u32).collect();
        let among = if among.is_empty() { &all[..] } else { among };
        let mut best = (usize::MAX, f64::INFINITY);
        for &i in among {
            let d2 = line.distance2(i as usize, p);
            if d2 < best.1 {
                best = (i as usize, d2);
            }
        }
        (best.0, best.1.sqrt())
    }

    #[derive(Default)]
    struct Tally {
        centres: u64,
        subs: u64,
        misses: Vec<String>,
        missed: u64,
        worst: f64,
    }

    impl Tally {
        fn miss(&mut self, what: String, dk: f64) {
            self.missed += 1;
            self.worst = self.worst.max(dk);
            if self.misses.len() < 12 {
                self.misses.push(what);
            }
        }

        fn add(&mut self, t: Tally) {
            self.centres += t.centres;
            self.subs += t.subs;
            self.missed += t.missed;
            self.worst = self.worst.max(t.worst);
            if self.misses.len() < 12 {
                self.misses.extend(t.misses.into_iter().take(3));
            }
        }
    }

    /// Pixel `(x, y)` of `depth` at body `body`: its centre as `depth_at` reads it and its
    /// sub-samples as `point_depths` reads them, each against the read among every segment.
    fn check(depth: &CurveDepth, x: u32, y: u32, body: f64, subs: bool, case: &str, t: &mut Tally) {
        let line = &depth.line;
        let centre = [x as f64 + 0.5, y as f64 + 0.5];
        // As `Paths::depth_at`.
        let mut near = Vec::new();
        let by_grid = depth.near_cut(centre, &[centre], body, &mut near);
        if !by_grid {
            depth.candidates(x, y, body, &mut near);
            line.cut(centre, &[centre], &mut near, body);
        }
        let got = line.key_under(centre, &near, body);
        let want = line.key_under(centre, &[], body);
        t.centres += 1;
        if got.0.to_bits() != want.0.to_bits() || got.1.to_bits() != want.1.to_bits() {
            let (fi, fd) = nearest(line, &near, centre);
            let (ti, td) = nearest(line, &[], centre);
            t.miss(
                format!(
                    "{case} centre ({x}, {y}) [{}]: key {:e} against {:e} (diff {:e}); nearest \
                     candidate {fi} at {fd:.3} px, true nearest {ti} at {td:.3} px; {} candidates",
                    if by_grid {
                        "fine grid"
                    } else {
                        "candidates+cut"
                    },
                    got.0,
                    want.0,
                    (got.0 - want.0).abs(),
                    near.len()
                ),
                (got.0 - want.0).abs(),
            );
        }
        if !subs {
            return;
        }
        // As `Paths::point_depths`.
        let mut points: Vec<[f64; 2]> = (0..SAMPLES).map(|k| sample_point(x, y, k)).collect();
        points.push(centre);
        let mut near = Vec::new();
        let mut out = vec![0.0; points.len()];
        let by_grid = depth.near_cut(centre, &points, body, &mut near);
        if by_grid {
            line.keys_among(&points, &near, body, &mut out);
        } else {
            depth.candidates(x, y, body, &mut near);
            line.keys_under(centre, &points, &mut near, body, &mut out);
        }
        for (k, p) in points.iter().enumerate() {
            let want = line.key_under(*p, &[], body).0;
            t.subs += 1;
            if out[k].to_bits() != want.to_bits() {
                let (fi, fd) = nearest(line, &near, *p);
                let (ti, td) = nearest(line, &[], *p);
                t.miss(
                    format!(
                        "{case} ({x}, {y}) sub {k} at {p:?} [{}]: key {:e} against {:e} (diff \
                         {:e}); nearest candidate {fi} at {fd:.3} px, true nearest {ti} at {td:.3} \
                         px; {} candidates",
                        if by_grid {
                            "fine grid"
                        } else {
                            "candidates+cut"
                        },
                        out[k],
                        want,
                        (out[k] - want).abs(),
                        near.len()
                    ),
                    (out[k] - want).abs(),
                );
            }
        }
    }

    fn report(name: &str, t: &Tally) {
        eprintln!(
            "BRUTE {name}: {} centres, {} sub-samples, {} misses, worst key diff {:e}",
            t.centres, t.subs, t.missed, t.worst
        );
        for m in &t.misses {
            eprintln!("  {m}");
        }
    }

    /// Deep inside large filled outlines of a path with no plane of its own (its fill read on its
    /// curve with body 0, its margin the bins of a path with no stroke: 18 px), every pixel inside
    /// on a stride of 3, centres everywhere and sub-samples at every fifth.
    #[test]
    #[ignore = "exhaustive: some 9M reads, about half a minute in release; run with --ignored"]
    fn deep_inside_a_large_planeless_fill_the_read_is_the_true_nearest_s() {
        let (w, h) = (1280u32, 960u32);
        let key = |p: [f64; 2]| -(1.0 + 1e-4 * (0.37 * p[0] + 0.61 * p[1]) + 1e-8 * p[0] * p[1]);
        let rot = |c: [f64; 2], r: f64, deg: f64, n: usize| -> Vec<[f64; 2]> {
            (0..n)
                .map(|i| {
                    let a = (deg + 360.0 * i as f64 / n as f64).to_radians();
                    [c[0] + r * libm::cos(a), c[1] + r * libm::sin(a)]
                })
                .collect()
        };
        let mut shapes: Vec<(String, Vec<[f64; 2]>)> = Vec::new();
        for s in [200.0, 400.0, 800.0, 1600.0, 3200.0] {
            shapes.push((
                format!("right triangle {s}"),
                vec![[40.25, 30.5], [40.25 + s, 30.5], [40.25, 30.5 + s]],
            ));
            shapes.push((
                format!("right triangle flipped {s}"),
                vec![
                    [1240.75, 930.25],
                    [1240.75 - s, 930.25],
                    [1240.75, 930.25 - s],
                ],
            ));
            for (n, deg) in [(3usize, 7.0), (4, 20.0), (5, 3.0), (6, 11.0), (32, 1.0)] {
                shapes.push((
                    format!("{n}-gon r {} turned {deg}", s / 2.0),
                    rot([640.3, 480.7], s / 2.0, deg, n),
                ));
            }
        }
        let mut total = Tally::default();
        for (name, poly) in &shapes {
            let line = polyline(poly, true, &key);
            let depth = CurveDepth::new(line, 18.0, w, h);
            let mut t = Tally::default();
            for y in (0..h).step_by(3) {
                for x in (0..w).step_by(3) {
                    if !inside(poly, [x as f64 + 0.5, y as f64 + 0.5]) {
                        continue;
                    }
                    let subs = (x / 3 + y / 3) % 5 == 0;
                    check(&depth, x, y, 0.0, subs, name, &mut t);
                }
            }
            report(name, &t);
            total.add(t);
        }
        report("deep fills, all shapes", &total);
        assert_eq!(total.missed, 0, "reads that are not the true nearest's");
    }

    /// At and around the miter tips of two straight edges meeting at 10, 12, 15, 20 and 30 degrees
    /// (10 is past Cairo's miter limit, a bevel), under four turns, at stroke half widths from 0.5
    /// to 20 px, the margin the path's bins give it (`(half * MITER_LIMIT).max(TILE) + 2`): every
    /// pixel on the tip's side the stroke rasterizes (one of its points within an edge's body,
    /// butt-ended at the joint, or the join: the miter up to its tip within the limit, the bevel
    /// past it), centres and sub-samples, read at the stroke's body.
    #[test]
    #[ignore = "exhaustive: some 3M reads; run with --ignored"]
    fn at_miter_tips_the_read_is_the_true_nearest_s() {
        let (w, h) = (1400u32, 1400u32);
        let key = |p: [f64; 2]| -(2.0 + 3e-4 * (0.53 * p[0] - 0.29 * p[1]));
        let v = [700.3, 700.7];
        let mut total = Tally::default();
        for theta in [10.0f64, 12.0, 15.0, 20.0, 30.0] {
            for turn in [0.0f64, 17.0, 45.0, 73.0] {
                for half in [0.5f64, 1.2, 2.0, 4.0, 10.0, 20.0] {
                    let margin = (half * MITER_LIMIT).max(TILE as f64) + 2.0;
                    let len = 300.0;
                    // The tip points along `turn`; the edges come into the joint from the far side
                    // at +-theta/2 about the bisector.
                    let at = |deg: f64| {
                        let a = deg.to_radians();
                        [v[0] + len * libm::cos(a), v[1] + len * libm::sin(a)]
                    };
                    let (a, b) = (
                        at(turn + theta / 2.0 + 180.0),
                        at(turn - theta / 2.0 + 180.0),
                    );
                    let line = polyline(&[a, v, b], false, &key);
                    let depth = CurveDepth::new(line, margin, w, h);
                    // Cairo's join: a miter where its length over the stroke's width,
                    // 1 / sin(theta / 2), is within the limit, its tip that times `half` from the
                    // joint along the bisector; past it a bevel, the line between the outer
                    // corners of the two edges' butt ends at the joint.
                    let miter = 1.0 / libm::sin((theta / 2.0).to_radians());
                    let bevel = miter > MITER_LIMIT;
                    let tip = half * miter;
                    let r = if bevel { half } else { tip } + 2.0;
                    let (dx, dy) = (libm::cos(turn.to_radians()), libm::sin(turn.to_radians()));
                    let unit = |s: [f64; 2], e: [f64; 2]| {
                        let l = libm::hypot(e[0] - s[0], e[1] - s[1]);
                        [(e[0] - s[0]) / l, (e[1] - s[1]) / l]
                    };
                    let (u1, u2) = (unit(a, v), unit(v, b));
                    // The outer side of the joint: right of the path where it turns left.
                    let out = if u1[0] * u2[1] - u1[1] * u2[0] > 0.0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let corner = |u: [f64; 2]| [v[0] + out * half * u[1], v[1] - out * half * u[0]];
                    let join = if bevel {
                        vec![v, corner(u1), corner(u2)]
                    } else {
                        vec![
                            v,
                            corner(u1),
                            [v[0] + tip * dx, v[1] + tip * dy],
                            corner(u2),
                        ]
                    };
                    // An edge's body: within `half` of it, its foot on it.
                    let on_edge = |s: [f64; 2], e: [f64; 2], q: [f64; 2]| {
                        let u = unit(s, e);
                        let l = libm::hypot(e[0] - s[0], e[1] - s[1]);
                        let along = (q[0] - s[0]) * u[0] + (q[1] - s[1]) * u[1];
                        let across = (q[1] - s[1]) * u[0] - (q[0] - s[0]) * u[1];
                        (0.0..=l).contains(&along) && across.abs() <= half
                    };
                    let rasterized =
                        |q: [f64; 2]| on_edge(a, v, q) || on_edge(v, b, q) || inside(&join, q);
                    let mut t = Tally::default();
                    let name = format!("joint {theta} deg turned {turn} half {half}");
                    let (x0, x1) = ((v[0] - r - 2.0) as u32, (v[0] + r + 2.0) as u32);
                    let (y0, y1) = ((v[1] - r - 2.0) as u32, (v[1] + r + 2.0) as u32);
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let p = [x as f64 + 0.5 - v[0], y as f64 + 0.5 - v[1]];
                            // The tip's side of the joint, where the stroke rasterizes the pixel.
                            let points = (0..SAMPLES)
                                .map(|k| sample_point(x, y, k))
                                .chain([[x as f64 + 0.5, y as f64 + 0.5]]);
                            if p[0] * dx + p[1] * dy < -1.5 || !points.into_iter().any(rasterized) {
                                continue;
                            }
                            check(&depth, x, y, half, true, &name, &mut t);
                        }
                    }
                    if t.missed > 0 {
                        report(&name, &t);
                    }
                    total.add(t);
                }
            }
        }
        report("miter tips, all joints", &total);
        assert_eq!(total.missed, 0, "reads that are not the true nearest's");
    }

    /// Inside filled outlines of short pieces, of a path with no plane of its own (its fill read on
    /// its curve with body 0, its margin the bins of a path with no stroke: 18 px), seeded: 16
    /// polygons of 12 to 48 sides, 60 to 160 px in radius, turned and placed at random, their keys
    /// sloped at random; at 400 pixels each, drawn at random inside, the centre as `depth_at` reads
    /// it and the sub-samples as `point_depths` reads them, each against the read among every
    /// segment. Where the widened bounds of a piece hold a pixel whose nearest piece lies beyond
    /// the margin, the bounds alone are not all a read may take.
    #[test]
    fn inside_seeded_planeless_fills_of_short_pieces_the_read_is_the_true_nearest_s() {
        let (w, h) = (1280u32, 960u32);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut total = Tally::default();
        for trial in 0..16 {
            let n = 12 + (next() * 37.0) as usize;
            let r = 60.0 + next() * 100.0;
            let c = [400.0 + next() * 480.0, 300.0 + next() * 360.0];
            let turn = next() * 360.0;
            let slope = [next() - 0.5, next() - 0.5];
            let key = move |p: [f64; 2]| -(1.0 + 1e-4 * (slope[0] * p[0] + slope[1] * p[1]));
            let poly: Vec<[f64; 2]> = (0..n)
                .map(|i| {
                    let a = (turn + 360.0 * i as f64 / n as f64).to_radians();
                    [c[0] + r * libm::cos(a), c[1] + r * libm::sin(a)]
                })
                .collect();
            let depth = CurveDepth::new(polyline(&poly, true, &key), 18.0, w, h);
            let name = format!("trial {trial}: {n}-gon r {r:.1}");
            let mut t = Tally::default();
            let mut read = 0;
            while read < 400 {
                let (x, y) = (
                    (c[0] + (2.0 * next() - 1.0) * r) as u32,
                    (c[1] + (2.0 * next() - 1.0) * r) as u32,
                );
                if inside(&poly, [x as f64 + 0.5, y as f64 + 0.5]) {
                    check(&depth, x, y, 0.0, true, &name, &mut t);
                    read += 1;
                }
            }
            total.add(t);
        }
        report("seeded short-piece fills", &total);
        assert!(total.subs >= 16 * 400 * SAMPLES as u64);
        assert_eq!(total.missed, 0, "reads that are not the true nearest's");
    }

    /// Where the widened bounds of the pixel centre's nearest hold the centre, the pixel's
    /// sub-samples reach half a pixel past it on each axis: a fill's curve with a long side 17.2 px
    /// to the left of the centre (its nearest, its bounds holding it) and a short one 18.05 px to
    /// its right (its bounds, widened by the 18 px margin, missing it by 0.05 px), keyed apart.
    /// A sub-sample at the pixel's right edge lies nearer the right side, 17.58 against 17.67 px:
    /// within the centre's nearest and 0.75, and that and the half pixel beyond the margin.
    #[test]
    fn near_a_pixel_s_edge_a_read_takes_a_segment_whose_widened_bounds_miss_its_centre() {
        let (w, h) = (256u32, 256u32);
        let c = [100.5, 100.5];
        let line = Centreline::new(
            &[
                ([c[0] - 17.2, 50.0], -1.01),
                ([c[0] - 17.2, 150.0], -1.01),
                ([c[0] + 18.05, 95.0], -1.0),
                ([c[0] + 18.05, 105.0], -1.0),
            ],
            &[2],
        );
        let depth = CurveDepth::new(line, 18.0, w, h);
        let mut t = Tally::default();
        check(
            &depth,
            100,
            100,
            0.0,
            true,
            "two sides about the margin",
            &mut t,
        );
        report("two sides about the margin", &t);
        assert_eq!(t.missed, 0, "reads that are not the true nearest's");
    }
}
