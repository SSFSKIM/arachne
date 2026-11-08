//! The frame buffer, and the one place colour is blended and fragments become frame bytes.
//!
//! Frames have the layout of CE's `camera.pixel_array`: `height * width * 4` bytes, row-major, in
//! R, G, B, A order (CE hands Cairo its colours with red and blue swapped, so Cairo's native ARGB32
//! words land in memory as R, G, B, A), with paint premultiplied as Cairo stores it and the
//! background as CE writes it.
//!
//! The colour contract:
//! - 2D paint is blended in sRGB-encoded space with the arithmetic of Cairo's ARGB32 image surface
//!   ([`premultiply`], [`blend`]);
//! - lit colour is computed in linear light (`shade.rs`) and encoded ([`srgb_encode`]) before it
//!   is blended;
//! - emitted light is kept in its own linear buffer per pixel and tone-mapped onto the frame once,
//!   at the end ([`add_emission`]);
//! - the opaque pass runs first, and its depth is readable ([`TransparentPass::opaque_depth`])
//!   before the transparent pass resolves;
//! - the transparent pass takes fragments from any producer through one call
//!   ([`TransparentPass::add`]).
//!
//! Each pixel resolves its fragments by depth, ties by draw order (later on top), on a fixed
//! pattern of [`SAMPLES`] sub-samples. A pixel whose fragments keep one depth order at every
//! sub-sample and hold at most one partially covering mask, and no partial mask beside a path's
//! partial area, takes the **coverage resolve**: its fragments composite back to front with their
//! coverage (a path's exact area, a mask's count), as Cairo composites paint. A depth crossing
//! inside the pixel, two or more partially covering masks, or a partial mask with a path's partial
//! area (a surface's silhouette under a path's edge, which the coverage resolve would composite as
//! independent shapes, and whose resolve would otherwise flip as crossings come and go) takes the
//! **sample resolve**: each sub-sample composites, in its own depth order, the
//! fragments covering it, and the pixel averages its sub-samples. There a mask fragment covers a
//! sub-sample or does not, and a fragment with exact area composites with its exact area inside
//! the sub-sample's cell ([`Producer::cell_areas`], [`sample_cell`]), so a path's edge lands where
//! the coverage resolve puts it, up to rounding, whichever resolve its pixel takes. A pixel whose
//! front fragment is opaque and covers it, with nothing in front, takes the **fast** path.
//! Fragments that every sub-sample's order puts behind an opaque fragment covering the pixel are
//! dropped first, read from those orders ([`crate::order::rank`]), not pair by pair. A
//! mask is never made from an alpha: a fragment with exact area gives its cells' areas only when
//! its pixel takes the sample resolve, and its point-sampled mask ([`Producer::mask`]) only when a
//! producer reads the opaque depth.
//!
//! The resolves count **units**, not fragments: a run, in depth order, of one surface's mask
//! fragments whose masks do not overlap (a mesh's triangles meeting at an internal edge) is one
//! unit, with their union for its mask and their colour averaged over the sub-samples each covers;
//! every other fragment is a unit of its own. So a surface's internal edges take the coverage
//! resolve, and an opaque surface tiling the pixel hides what is behind it at every sub-sample.
//! The average is exact only while every other layer covers the whole pixel: a unit of several
//! colours beside a partial mask or a path's partial area (an edge of something else crossing a
//! lit surface's internal edge) counts as a partial mask, so the sample resolve keeps which of
//! its colours the partial layer covers.

use std::cmp::Ordering;

/// Sub-samples per pixel.
pub const SAMPLES: usize = 16;

/// A mask covering every sub-sample.
pub const FULL: u16 = 0xFFFF;

/// Sub-pixel units per pixel in the fixed-point positions rasterizers snap to.
pub const SUBPIXEL: i64 = 256;

/// The sub-sample pattern, offsets in sixteenths of a pixel on a grid of 16 x 16 cells. It is
/// stratified: each cell of the pixel's 4 x 4 grid holds one sample ([`sample_cell`]), and each of
/// the 16 rows and 16 columns holds one (n-rooks), so an axis-aligned edge gets 17 coverage levels.
/// Within that, the samples are shuffled to the arrangement a search found with the least worst
/// error of a half-plane's coverage over every edge angle and offset (0.115 of the pixel, where
/// Direct3D's standard pattern has 0.148), among those giving at least 13 levels to edges at 45
/// and 135 degrees (it gives 14 and 15); the unshuffled stratified pattern gives those edges 8.
/// Sample `s` sits at the centre of cell `SAMPLE_OFFSETS[s]` from the pixel's centre,
/// `(offset + 1/2) / 16`, so the samples' centroid is the pixel's centre.
///
/// A GPU backend computes masks in a compute stage at these positions, not with hardware
/// multisampling: WebGPU guarantees only 1 and 4 samples per pixel.
pub const SAMPLE_OFFSETS: [[i8; 2]; SAMPLES] = [
    [-8, -7],
    [-1, -8],
    [3, -6],
    [5, -5],
    [-6, -1],
    [-4, -4],
    [0, -3],
    [6, -2],
    [-7, 2],
    [-2, 0],
    [2, 3],
    [4, 1],
    [-5, 5],
    [-3, 4],
    [1, 7],
    [7, 6],
];

/// The cell of the pixel's 4 x 4 grid that holds sub-sample `s`, as its column and row (`0..4`
/// each, left to right and top to bottom). Each cell holds one sub-sample.
pub fn sample_cell(s: usize) -> [usize; 2] {
    SAMPLE_OFFSETS[s].map(|o| (o as i16 + 8) as usize / 4)
}

/// Sub-sample `s` of pixel `(x, y)` in pixel coordinates (pixel `(x, y)` spans `[x, x + 1)` by
/// `[y, y + 1)`).
pub fn sample_point(x: u32, y: u32, s: usize) -> [f64; 2] {
    let [dx, dy] = SAMPLE_OFFSETS[s];
    [
        x as f64 + 0.5 + (dx as f64 + 0.5) / 16.0,
        y as f64 + 0.5 + (dy as f64 + 0.5) / 16.0,
    ]
}

/// Sub-sample `s`'s position inside its pixel in sub-pixel units ([`SUBPIXEL`] per pixel); the
/// same point as [`sample_point`], exactly.
pub fn sample_fixed(s: usize) -> [i64; 2] {
    let [dx, dy] = SAMPLE_OFFSETS[s];
    let cell = SUBPIXEL / 16;
    [
        SUBPIXEL / 2 + dx as i64 * cell + cell / 2,
        SUBPIXEL / 2 + dy as i64 * cell + cell / 2,
    ]
}

/// The centroid of the sub-samples `mask` covers in pixel `(x, y)`: where a fragment's colour is
/// read.
pub fn mask_centroid(x: u32, y: u32, mask: u16) -> [f64; 2] {
    if mask == FULL {
        return [x as f64 + 0.5, y as f64 + 0.5];
    }
    let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
    for s in 0..SAMPLES {
        if mask & (1 << s) != 0 {
            let [px, py] = sample_point(x, y, s);
            sx += px;
            sy += py;
            n += 1.0;
        }
    }
    [sx / n, sy / n]
}

/// A value over the pixel plane, `a + b * x + c * y` in pixel coordinates: a depth (smaller is
/// nearer) or a channel of a colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    /// For a depth, how far its coefficients may put it, anywhere its fragments are read, from the
    /// exact depth of the inputs it was made from (the snapped points and the normals carried),
    /// besides the rounding of reading it at a point ([`Plane::read_error`]): the representation
    /// its constructor imposes and that constructor's own arithmetic, and for a face keyed on the
    /// plane nearest it, its outline's departure from that plane ([`crate::order`]).
    pub err: f64,
}

/// The unit roundoff of `f64`.
pub(crate) const U: f64 = f64::EPSILON / 2.0;

impl Plane {
    /// `a + b * x + c * y`, exact.
    pub const fn new(a: f64, b: f64, c: f64) -> Plane {
        Plane { a, b, c, err: 0.0 }
    }

    /// One depth everywhere.
    pub const fn constant(depth: f64) -> Plane {
        Plane::new(depth, 0.0, 0.0)
    }

    /// The plane through three `(x, y, depth)` points. It is constant where they share one depth
    /// (the mean where they are collinear, the nearest where a depth is not finite).
    pub fn through(points: [[f64; 3]; 3]) -> Plane {
        Plane::through_with_errors(points, [0.0; 3])
    }

    /// [`Plane::through`] for depths known within `errors`, its [`Plane::err`] bounding it inside
    /// the triangle (where a mask covers a sub-sample). Inside, the corners' own errors carry over
    /// at most their largest (the plane there is a convex combination of the corners). The
    /// determinant and the numerators cancel as the triangle thins: each rounds by about `4u` of
    /// its two products' magnitudes, so the gradient errs by its numerators' error and its own
    /// share of the determinant's over `|det|`, and moves the plane by at most that error times an
    /// edge from the first corner, `u L / h` of the depth's change along the triangle for a
    /// triangle `L` long and `h` high. `a` rounds once more, by `3u` of its terms.
    pub fn through_with_errors(points: [[f64; 3]; 3], errors: [f64; 3]) -> Plane {
        let [[x0, y0, d0], [x1, y1, d1], [x2, y2, d2]] = points;
        let input = errors[0].max(errors[1]).max(errors[2]);
        if d0 == d1 && d1 == d2 {
            return Plane {
                err: input,
                ..Plane::constant(d0)
            };
        }
        if !(d0.is_finite() && d1.is_finite() && d2.is_finite()) {
            return Plane::constant(d0.min(d1).min(d2));
        }
        let (e1x, e1y, e2x, e2y) = (x1 - x0, y1 - y0, x2 - x0, y2 - y0);
        let det = e1x * e2y - e2x * e1y;
        if det == 0.0 || !det.is_finite() {
            let mean = (d0 + d1 + d2) / 3.0;
            let spread = [d0, d1, d2]
                .iter()
                .map(|d| (d - mean).abs())
                .fold(0.0, f64::max);
            return Plane {
                err: spread + input + 4.0 * U * mean.abs(),
                ..Plane::constant(mean)
            };
        }
        let (g1, g2) = (d1 - d0, d2 - d0);
        let b = (g1 * e2y - g2 * e1y) / det;
        let c = (e1x * g2 - e2x * g1) / det;
        let a = d0 - b * x0 - c * y0;
        let d_det = 4.0 * U * ((e1x * e2y).abs() + (e2x * e1y).abs());
        let d_b = (4.0 * U * ((g1 * e2y).abs() + (g2 * e1y).abs()) + b.abs() * d_det) / det.abs()
            + U * b.abs();
        let d_c = (4.0 * U * ((e1x * g2).abs() + (e2x * g1).abs()) + c.abs() * d_det) / det.abs()
            + U * c.abs();
        let spread = (d_b * e1x.abs() + d_c * e1y.abs()).max(d_b * e2x.abs() + d_c * e2y.abs());
        let rounding = 3.0 * U * (d0.abs() + (b * x0).abs() + (c * y0).abs());
        Plane {
            a,
            b,
            c,
            err: input + spread + rounding,
        }
    }

    /// How far reading the plane at `(x, y)` may round, `4u` of the magnitudes it sums.
    pub fn read_error(&self, x: f64, y: f64) -> f64 {
        4.0 * U * (self.a.abs() + (self.b * x).abs() + (self.c * y).abs())
    }

    /// The depth at a point (`-0.0` read as `0.0`, so equal depths compare equal).
    pub fn at(&self, x: f64, y: f64) -> f64 {
        self.a + self.b * x + self.c * y + 0.0
    }
}

/// How much of a pixel a fragment covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coverage {
    /// A mask over the sub-samples (bit `s` is sub-sample `s`). Meshes give masks.
    Mask(u16),
    /// An exact area in `0..=1`; the producer gives the mask at the same sub-samples on demand,
    /// through [`Producer::mask`] with this `id`. Paths give areas.
    Area { area: f32, id: u32 },
}

/// One drawn object's share of one pixel, `'f` the frame's records it may carry one of
/// ([`Fragment::record`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fragment<'f> {
    pub depth: Plane,
    /// Premultiplied, in the frame's byte order.
    pub colour: [u8; 4],
    pub coverage: Coverage,
    /// The place in the draw order; of two fragments at one depth, the later is on top.
    pub order: u32,
    /// The part within one drawn object (a triangle of a mesh), breaking ties the same way.
    pub sub: u32,
    /// The surface it belongs to, when not 0: one mesh drawn, whose triangles share edges exactly
    /// and never cover one sub-sample twice where they tile the pixel (see [`Fragment::on_surface`]).
    pub surface: u32,
    /// Whether a path's layer painted it: then `sub` is `3 i + k` for its record's `i`th path and
    /// the layer `k` (0 its background stroke, 1 its fill, 2 its stroke), which the order reads
    /// ([`crate::order::own_layers`]).
    pub path_layer: bool,
    /// The identities of the planes a path's layer or a mesh's triangle lies on, ascending, the
    /// rest zero (none: all zero): of its frame's registered planes (`depth::Registry`), which the
    /// camera keys finitely. A flat record lies on one; a record with no plane of its own (a line,
    /// a curve) on every one whose plane holds it, up to four (`depth::MOST_IDENTITIES`), so a line
    /// along a cube's edge lies on both faces'. Two fragments sharing one, whichever producer drew
    /// each, keep paint order where their keys are one depth within what the snap can do to one
    /// plane ([`crate::order::coplanar_tied`]).
    pub plane: [u32; 4],
    /// How far its key may stand, as a share of the key wherever it is read, from the key across
    /// the plane content put its points on, for the snap of those points to the input grid and its
    /// own plane's fit to them (`Projection::key_reach` of `depth::snap_departure` and
    /// `depth::departure`; in a frame not snapped, the rounding of the core's own placing of them
    /// alone): what two fragments on one registered plane may part by and still be one plane
    /// ([`crate::order::coplanar_tied`]). For a record with no plane of its own read on the planes
    /// of the identities it lies on, the most over them of how far its points may stand off each
    /// one's plane where it meets the flat records on it (`depth::Registry::off_members`). Zero
    /// for a flat path content did not make flat (its planarity as the writer decided it): no
    /// plane content put it on. A stroke on a registered plane carries its path's; each layer of
    /// such a path is read on its plane against what shares it ([`Depths::against`]). Infinite
    /// where nothing bounds it (a path whose points reached the core already projected), and then
    /// it ties with anything on its plane.
    pub snap: f64,
    /// The flat record it is a fragment of, as its frame's registry holds it (`depth::Registry`):
    /// a path content made flat, or a mesh triangle, offered on its own plane and keyed with a
    /// finite [`Fragment::snap`]; `None` for any other (a record with no plane of its own, a
    /// pre-projected path, one the registry was not offered). Two fragments of flat records on no
    /// one identity, whose own planes the snapped input cannot tell apart in orientation, tie
    /// within the same term as two on one identity ([`crate::order::near_parallel`]).
    pub record: Option<FlatRecord<'f>>,
    /// For a record with no plane of its own on flat records' identities, its share against
    /// another such record ([`crate::order::coplanar_tied`]), which may meet it beyond the flat
    /// records' hull: the most over its identities of its snap's departure along each one's
    /// normal and its points' departure from it (`depth::Registry::off_identity`). Negative (minus
    /// infinity) for any other.
    pub wide: f64,
    /// For a stroke (or background stroke) of a path with no plane of its own, on identities or
    /// not: how far, as a share of itself, its key at each of this pixel's sub-samples may stand
    /// from the key read there along its curve through its points as content put them, for their
    /// snap to the input grid: the snap's move of the key at fixed points of the curve (its
    /// departure along the sight axis, `Projection::depth_share` of `depth::snap_departure`), and
    /// how far the read may slide along the curve and between its pieces where the snap moves them
    /// on screen (`depth::Centreline::slides_among`, over the least of the reads). Two such strokes
    /// sharing no identity tie within it ([`crate::order::lines`]). Negative (minus infinity) for
    /// any other.
    pub own: f64,
    /// The producer that added it (set by the pass).
    source: u16,
}

/// A fragment's flat record ([`Fragment::record`]): one record of its frame's registry, which two
/// fragments share when they are fragments of the same record, whatever its content.
#[derive(Clone, Copy, Debug)]
pub struct FlatRecord<'f>(pub &'f crate::depth::Offer);

impl PartialEq for FlatRecord<'_> {
    fn eq(&self, other: &FlatRecord<'_>) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl Fragment<'_> {
    /// The producer that added it, as an index into the pass's producers.
    pub(crate) fn source(&self) -> u16 {
        self.source
    }

    pub fn new(depth: Plane, colour: [u8; 4], coverage: Coverage, order: u32, sub: u32) -> Self {
        Fragment {
            depth,
            colour,
            coverage,
            order,
            sub,
            surface: 0,
            path_layer: false,
            plane: [0; 4],
            snap: f64::INFINITY,
            record: None,
            wide: f64::NEG_INFINITY,
            own: f64::NEG_INFINITY,
            source: 0,
        }
    }

    /// The fragment as part of surface `surface` of its producer. Mask fragments of one surface
    /// whose masks do not overlap, and that keep one place in the pixel's depth order against every
    /// other fragment, are merged into one fragment (their masks' union, their colours averaged by
    /// sub-sample count), so the internal edges of a surface take the coverage resolve. For meshes
    /// alone, and for a path over a surface that tiles the pixel, that is what the sample resolve
    /// gives, up to rounding.
    pub fn on_surface(mut self, surface: u32) -> Self {
        self.surface = surface;
        self
    }

    fn is_opaque(&self) -> bool {
        self.colour[3] == 255
    }

    fn is_full(&self) -> bool {
        match self.coverage {
            Coverage::Mask(m) => m == FULL,
            Coverage::Area { area, .. } => area >= 1.0,
        }
    }

    fn coverage8(&self) -> u8 {
        match self.coverage {
            Coverage::Mask(m) => mask_coverage(m),
            Coverage::Area { area, .. } => area_coverage(area),
        }
    }
}

/// Light a producer emits into one pixel, in linear light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emission {
    /// Linear RGB radiance, unbounded and not negative.
    pub radiance: [f64; 3],
    pub mask: u16,
    pub depth: Plane,
}

/// A colour as Cairo stores a solid source: straight RGBA clamped to `0..=1`, premultiplied to 16
/// bits with rounding, then truncated to 8 bits.
pub fn premultiply(rgba: [f64; 4]) -> [u8; 4] {
    let [r, g, b, a] = rgba.map(|c| c.clamp(0.0, 1.0));
    let short = |d: f64| ((d * 65535.0 + 0.5) as u32 >> 8) as u8;
    [short(r * a), short(g * a), short(b * a), short(a)]
}

/// `a * b / 255`, rounded as pixman rounds.
fn mul(a: u8, b: u8) -> u8 {
    let t = a as u32 * b as u32 + 0x80;
    (((t >> 8) + t) >> 8) as u8
}

/// `a * b / 255`, rounded as Cairo's span compositor rounds.
fn mul_7f(a: u8, b: u8) -> u8 {
    let t = a as u32 * b as u32 + 0x7f;
    ((t + (t >> 8)) >> 8) as u8
}

/// One premultiplied source over a pixel with an 8-bit coverage, as Cairo's ARGB32 image surface
/// composites a fill: an opaque source is interpolated toward the pixel by the coverage, any other
/// is multiplied by the coverage and laid over, saturating.
pub fn blend(src: [u8; 4], coverage: u8, dst: [u8; 4]) -> [u8; 4] {
    if src[3] == 255 {
        if coverage == 255 {
            return src;
        }
        let keep = 255 - coverage;
        return std::array::from_fn(|k| {
            (mul_7f(src[k], coverage) as u32 + mul_7f(dst[k], keep) as u32).min(255) as u8
        });
    }
    let src = if coverage == 255 {
        src
    } else {
        src.map(|c| mul(c, coverage))
    };
    let keep = 255 - src[3];
    std::array::from_fn(|k| (src[k] as u32 + mul(dst[k], keep) as u32).min(255) as u8)
}

/// An exact area as an 8-bit coverage.
pub fn area_coverage(area: f32) -> u8 {
    ((area as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// How far, per byte, the coverage resolve may land from what the sample resolve would give the
/// same pixel (blend at each sub-sample, then average), for a pixel holding one partial mask of
/// one colour over full layers and under at most one translucent full layer: the mask's count is
/// rounded to an 8-bit coverage and blended once, where the sample resolve blends at each covered
/// sub-sample and rounds the average, and the layer in front rounds each side again.
/// `tests/coverage_bound.rs` enumerates that scope exhaustively through [`blend`] and
/// [`mask_coverage`] (every partial colour channel and alpha, every byte beneath, every count,
/// every front alpha): the widest gap is exactly 3, and 2 with nothing in front, the values
/// [`coverage_to_sample_bound`] derives for it. Every other pixel on the coverage resolve (a
/// path's partial area, a surface's masks in several colours, more layers in front) takes that
/// function's bound for its stack.
pub const COVERAGE_TO_SAMPLE_LSB: u8 = 3;

/// The partial layer of a pixel on the coverage resolve, for [`coverage_to_sample_bound`]. A pixel
/// holds at most one: two partial masks, or a partial mask and a path's partial area, take the
/// sample resolve. Two or more paths' partial areas take the coverage resolve, as independent
/// coverages (as Cairo blends them), and on the sample resolve as their overlap in each cell: they
/// differ there by how the shapes overlap, not by rounding, and no bound here covers them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partial {
    /// One mask in one colour.
    Mask,
    /// One surface's masks in several colours, one unit: its colour their average.
    MergedMask,
    /// One path's exact area.
    Path,
}

/// `mul` lies within 127/255 of `a * b / 255`, `mul_7f` within 128/255 (every pair of bytes;
/// the tests below).
const MUL_ERROR: f64 = 127.0 / 255.0;
const MUL_7F_ERROR: f64 = 128.0 / 255.0;

/// How far a path's exact area may lie from the mean of its cells' areas ([`Producer::cell_areas`]):
/// both are exact in f64 up to their sums' rounding, then stored as f32. `tests/coverage_bound.rs`
/// holds the rasterizer to it.
pub const AREA_DRIFT: f64 = 1e-6;

/// How far, per byte, the coverage resolve of a pixel holding `partial` over full layers, under
/// `fronts` translucent full layers, may land from the sample resolve of the same pixel. Colours
/// are premultiplied (no channel above its alpha), as every colour the core blends is, so no blend
/// below saturates.
///
/// Per channel, let `B(c)` be the exact blend of the partial layer at coverage `c` over `d`, what
/// lies beneath it (the same at every sub-sample, and the same bytes on both resolves): `B` is
/// linear in `c`, its slope at most 1 per coverage level. The coverage resolve blends once at its
/// coverage `c`; the sample resolve blends at each sub-sample's coverage `c_k` and averages, which
/// is `B` at the mean of the `c_k` plus the mean of their roundings. Before the sample resolve
/// rounds its average, the two differ by at most
///
/// `g = |c - mean c_k| + e(c) + mean e(c_k) + m + fronts * 2 * MUL_ERROR`, where
/// - `|c - mean c_k|` is at most 1/2 for a mask (its count rounded to a byte, the `c_k` all or
///   nothing), and `1 + 255 * AREA_DRIFT` for a path (its area and each cell's rounded to a byte);
/// - `e` is one blend's rounding: at a partial coverage the larger of an opaque source's
///   `2 * MUL_7F_ERROR` and a translucent one's `3 * MUL_ERROR` (the source and its alpha scaled
///   by the coverage, what lies beneath by what is left), at full coverage `MUL_ERROR`, at none 0;
/// - `m`, for a merged mask, is under 1: its colour and alpha are each the nearest byte to their
///   average, half a level each, scaled by the coverage;
/// - each translucent full layer in front (its colour plus `mul(x, 255 - alpha)`, on both sides)
///   scales the difference by less than 1 and rounds once on each side.
///
/// The sample resolve rounds its average to the nearest byte, so the bytes differ by at most
/// `floor(g + 1/2)`: 2 + `fronts` for a mask, 3 + `fronts` for a merged mask, 4 + `fronts` for a
/// path. `tests/coverage_bound.rs` holds random and adversarial stacks of each to it through the
/// resolves themselves.
pub fn coverage_to_sample_bound(partial: Partial, fronts: u32) -> u8 {
    let partial_blend = (2.0 * MUL_7F_ERROR).max(3.0 * MUL_ERROR);
    let g = match partial {
        Partial::Mask => 0.5 + partial_blend + MUL_ERROR,
        Partial::MergedMask => 0.5 + partial_blend + MUL_ERROR + 1.0,
        Partial::Path => 1.0 + 255.0 * AREA_DRIFT + 2.0 * partial_blend,
    } + fronts as f64 * 2.0 * MUL_ERROR;
    (g + 0.5).floor() as u8
}

/// A mask's count as an 8-bit coverage.
pub fn mask_coverage(mask: u16) -> u8 {
    ((mask.count_ones() * 255 + 8) / 16) as u8
}

/// sRGB's transfer function, linear light to encoded.
pub fn srgb_encode(linear: f64) -> f64 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * libm::pow(linear, 1.0 / 2.4) - 0.055
    }
}

/// sRGB's transfer function, encoded to linear light.
pub fn srgb_decode(encoded: f64) -> f64 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        libm::pow((encoded + 0.055) / 1.055, 2.4)
    }
}

/// A pixel with the light emitted into it, tone-mapped once: in linear light each channel moves
/// toward white by `1 - exp(-E)`, so no amount of it passes white. A channel with no emission
/// keeps its byte, and alpha rises only to the channels that emission raised, so a pixel without
/// emission keeps its bytes whatever they are (a translucent background's are not premultiplied).
pub fn add_emission(pixel: [u8; 4], emission: [f64; 3]) -> [u8; 4] {
    let mut out = pixel;
    for k in 0..3 {
        if emission[k] != 0.0 {
            let l = srgb_decode(pixel[k] as f64 / 255.0);
            let lit = l + (1.0 - l) * (1.0 - libm::exp(-emission[k]));
            out[k] = (srgb_encode(lit.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
            if out[k] != pixel[k] {
                out[3] = out[3].max(out[k]);
            }
        }
    }
    out
}

/// Something that draws into the passes: a rasterizer of meshes, of paths, or anything else.
///
/// Each method must be a pure function of what the producer was built from (the stream's state)
/// and the pass's rectangle: no state shared across calls, no dependence on which rectangles were
/// drawn before. Tiles are drawn in any order on any thread, and a frame's bytes are the same on
/// any number of threads only because of this.
pub trait Producer: Sync {
    /// Adds the producer's opaque fragments in the pass's rectangle. A fragment may carry a record
    /// the producer holds ([`Fragment::record`]) for as long as the pass lives.
    fn opaque<'a>(&'a self, _pass: &mut OpaquePass<'a>) {}

    /// Adds the producer's other fragments and emission in the pass's rectangle; the opaque depth
    /// is readable here.
    fn transparent<'a>(&'a self, _pass: &mut TransparentPass<'a>) {}

    /// The mask, at the sub-samples, of the fragment this producer added to pixel `(x, y)` with
    /// exact area and this `id`: point-sampled at [`sample_point`]s. Read for the opaque depth.
    fn mask(&self, id: u32, x: u32, y: u32) -> u16 {
        unreachable!("a producer that adds exact areas gives their masks ({id} at {x}, {y})")
    }

    /// The exact area, in `0..=1` of the cell, of the fragment this producer added to pixel
    /// `(x, y)` with exact area and this `id`, inside each sub-sample's cell ([`sample_cell`]):
    /// what it composites with at each sub-sample of the sample resolve. Their mean is its area.
    fn cell_areas(&self, id: u32, x: u32, y: u32) -> [f32; SAMPLES] {
        unreachable!("a producer that adds exact areas gives their cells' areas ({id} at {x}, {y})")
    }

    /// The edges the rasterizer reads for the fragment this producer added with exact area and
    /// this `id` (the ones [`Self::mask`] and [`Self::cell_areas`] read), in device pixels; None
    /// from a producer of masks.
    fn edges(&self, _id: u32) -> Option<&[crate::scanline::Edge]> {
        None
    }

    /// The depth at each of `points` (into `out`: the pixel's sub-samples in order, then its
    /// centre) of the fragment this producer added to pixel `(x, y)` with exact area and this
    /// `id`, where its plane read at each point does not give it, and how far any of them may
    /// stand from the exact depth of its inputs: along a curve where its depth runs along one
    /// (`depth::Centreline::key_under`), and for a sloped plane at its fringe, where a sub-sample's
    /// point lies outside its outline, the plane at the outline's nearest point. `None`, and `out`
    /// untouched, where its plane at each point gives it.
    fn point_depths(
        &self,
        _id: u32,
        _x: u32,
        _y: u32,
        _points: &[[f64; 2]],
        _out: &mut [f64],
    ) -> Option<f64> {
        None
    }

    /// For the fragment this producer added to pixel `(x, y)` with exact area and this `id`, a
    /// layer of a path on a registered plane (its fill, or a stroke keyed on its curve), and
    /// `identity`, one of the identities it lies on ([`Fragment::plane`]): its key on that
    /// identity's plane at each of the pixel's sub-samples ([`sample_point`]), wherever the
    /// sub-sample lies, into `out`, and that key's error bound there. A flat path's key there is
    /// its own plane's; a path with no plane of its own is keyed on each identity's plane. What it
    /// is read on against a fragment sharing that identity ([`Depths::against`]). `None`, and `out`
    /// untouched, for any other fragment.
    fn plane_depths(
        &self,
        _id: u32,
        _identity: u32,
        _x: u32,
        _y: u32,
        _out: &mut [f64; SAMPLES],
    ) -> Option<f64> {
        None
    }
}

/// A fragment's key at each sub-sample of a pixel and that key's bound: one of [`Depths`]'s plane
/// rows.
pub(crate) type Row = ([f64; SAMPLES], f64);

/// A fragment's depth at each sub-sample of a pixel and at its centre, where the resolve and the
/// contributor report read it (`-0.0` read as `0.0`, as [`Plane::at`] reads it), and how far any
/// of them may stand from the exact depth of the fragment's inputs ([`crate::order`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Depths {
    pub(crate) samples: [f64; SAMPLES],
    pub(crate) centre: f64,
    pub(crate) err: f64,
    /// For a layer of a path on a registered plane, its key on the plane of the first identity it
    /// lies on ([`Fragment::plane`]) at the sub-samples and that key's bound
    /// ([`Producer::plane_depths`]): what it is read on against a fragment sharing that identity
    /// ([`Depths::against`]).
    pub(crate) on_plane: Option<Row>,
    /// Its rows on the planes of the identities it lies on after the first, in their order: empty
    /// but for a record with no plane of its own lying on more than one.
    pub(crate) more: Vec<Option<Row>>,
}

impl Depths {
    /// The depths `a` (whose depths these are) is read at against `b`, and their bound: where `b`
    /// shares an identity with it ([`crate::order::one_plane`]), its row on the first identity of
    /// its own that `b` lies on too, where it has one, so a path's layer and what shares its plane
    /// are held to one plane's keys at one point: a stroke's body reaching off its curve on that
    /// plane, a fill's sub-sample beyond its outline read where it lies, not at the outline's
    /// nearest point; else its own. Against anything else a stroke keeps the key along its curve,
    /// and a fill its key at its outline, which no plane extrapolates.
    pub(crate) fn against(&self, a: &Fragment, b: &Fragment) -> (&[f64; SAMPLES], f64) {
        let row = if crate::order::one_plane(a, b) {
            match shared_identity(&a.plane, &b.plane) {
                Some(0) | None => self.on_plane.as_ref(),
                Some(k) => self.more.get(k - 1).and_then(Option::as_ref),
            }
        } else {
            None
        };
        match row {
            Some((samples, err)) => (samples, *err),
            None => (&self.samples, self.err),
        }
    }

    /// Fragment `f`'s at pixel `(x, y)`: as its producer keys it where its plane read at each
    /// point does not give it ([`Producer::point_depths`]), else on its plane; with its rows on
    /// the planes of the identities it lies on ([`Producer::plane_depths`]).
    pub(crate) fn of(producers: &[&dyn Producer], f: &Fragment, x: u32, y: u32) -> Depths {
        if let Coverage::Area { id, .. } = f.coverage {
            let mut points = [[x as f64 + 0.5, y as f64 + 0.5]; SAMPLES + 1];
            for (k, p) in points.iter_mut().take(SAMPLES).enumerate() {
                *p = sample_point(x, y, k);
            }
            let producer = producers[f.source as usize];
            let row = |identity: u32| {
                let mut on = [0.0; SAMPLES];
                producer
                    .plane_depths(id, identity, x, y, &mut on)
                    .map(|err| (on.map(|v| v + 0.0), err))
            };
            let on_plane = (f.plane[0] != 0).then(|| row(f.plane[0])).flatten();
            let more: Vec<Option<Row>> = f.plane[1..]
                .iter()
                .take_while(|identity| **identity != 0)
                .map(|identity| row(*identity))
                .collect();
            let mut out = [0.0; SAMPLES + 1];
            if let Some(err) = producer.point_depths(id, x, y, &points, &mut out) {
                return Depths {
                    samples: std::array::from_fn(|k| out[k] + 0.0),
                    centre: out[SAMPLES] + 0.0,
                    err,
                    on_plane,
                    more,
                };
            }
            return Depths {
                on_plane,
                more,
                ..Depths::of_plane(f, x, y)
            };
        }
        Depths::of_plane(f, x, y)
    }

    /// Every plane row it has ([`Depths::on_plane`], [`Depths::more`]).
    pub(crate) fn rows(&self) -> impl Iterator<Item = &Row> {
        self.on_plane.iter().chain(self.more.iter().flatten())
    }

    /// Fragment `f`'s at pixel `(x, y)` on its plane.
    pub(crate) fn of_plane(f: &Fragment, x: u32, y: u32) -> Depths {
        let samples = if f.depth.b == 0.0 && f.depth.c == 0.0 {
            [f.depth.a + 0.0; SAMPLES]
        } else {
            std::array::from_fn(|k| {
                let [px, py] = sample_point(x, y, k);
                f.depth.at(px, py)
            })
        };
        Depths {
            samples,
            centre: f.depth.at(x as f64 + 0.5, y as f64 + 0.5),
            // The plane's own error and the rounding of reading it anywhere in the pixel.
            err: f.depth.err + f.depth.read_error(x as f64 + 1.0, y as f64 + 1.0),
            on_plane: None,
            more: Vec::new(),
        }
    }
}

/// Where in identity set `a` ([`Fragment::plane`]) lies the first identity set `b` holds too.
pub(crate) fn shared_identity(a: &[u32; 4], b: &[u32; 4]) -> Option<usize> {
    a.iter().position(|p| *p != 0 && b.contains(p))
}

/// A path's strokes drawn on its own fill, at pixel depths `depths` of `frags`: at each sub-sample
/// the stroke and the fill both cover (`coverage(i)`, fragment `i`'s coverage at each sub-sample),
/// a stroke drawn after the fill takes the nearer of its own depth (the wire's, along its curve) and the fill's
/// there, whatever the fill is keyed on, as CE draws a stroke over its fill: wherever the fill is,
/// the stroke is at least as near. Elsewhere (its outer half, a path without a fill) it keeps its
/// own. Its error bound is the largest of those of the depths it takes, both depths' where they
/// lie within their bounds of each other (either may be the nearer); its centre keeps its own,
/// which no order reads. A background stroke, drawn under its fill, keeps its own depth. What
/// stands in front of the fill hides both still, and a depth taken is a finite key with its own
/// bound, so the strong orders stay depth's strict ones and still never cycle at one sub-sample.
/// With those bounds, whatever depth puts in front of the stroke where both cover it also puts in
/// front of the fill, so no rank puts the fill in front of its stroke there.
pub(crate) fn strokes_on_own_fills(
    frags: &[Fragment],
    depths: &mut [Depths],
    coverage: impl Fn(usize) -> [u8; SAMPLES],
) {
    for (j, fill) in frags.iter().enumerate() {
        if !fill.path_layer || fill.sub % 3 != 1 {
            continue;
        }
        let mut filled = None;
        for (i, stroke) in frags.iter().enumerate() {
            if stroke.sub < fill.sub || !crate::order::own_layers(stroke, fill) {
                continue;
            }
            let filled = *filled.get_or_insert_with(|| coverage(j));
            let stroked = coverage(i);
            let (on, on_err) = (depths[j].samples, depths[j].err);
            let (own, own_err) = (depths[i].samples, depths[i].err);
            let mut err = 0.0f64;
            for k in 0..SAMPLES {
                if filled[k] == 0 || stroked[k] == 0 {
                    err = err.max(own_err);
                    continue;
                }
                let (f, s) = (on[k], own[k]);
                if f < s {
                    depths[i].samples[k] = f;
                }
                // The bound of the nearer of two depths: the chosen one's where they part beyond
                // their bounds, else either's, as either may be the nearer.
                err = err.max(if crate::order::tied(f, on_err, s, own_err) {
                    on_err.max(own_err)
                } else if f < s {
                    on_err
                } else {
                    own_err
                });
            }
            depths[i].err = err;
        }
    }
}

/// A rectangle of pixels, `x0..x1` by `y0..y1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl Rect {
    pub fn width(&self) -> u32 {
        self.x1 - self.x0
    }

    /// Pixels in the rectangle.
    pub fn pixels(&self) -> usize {
        self.width() as usize * (self.y1 - self.y0) as usize
    }

    fn index(&self, x: u32, y: u32) -> usize {
        assert!(
            (self.x0..self.x1).contains(&x) && (self.y0..self.y1).contains(&y),
            "pixel ({x}, {y}) is outside the pass's rectangle {self:?}"
        );
        (y - self.y0) as usize * self.width() as usize + (x - self.x0) as usize
    }
}

/// How a pixel was resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Resolve {
    /// No fragment: the background.
    Background = 0,
    /// One opaque fragment covering the pixel, nothing in front of it.
    Fast = 1,
    Coverage = 2,
    Sample = 3,
}

/// Pixels per resolve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResolveCounts {
    pub background: u64,
    pub fast: u64,
    pub coverage: u64,
    pub sample: u64,
}

impl ResolveCounts {
    fn count(&mut self, r: Resolve) {
        match r {
            Resolve::Background => self.background += 1,
            Resolve::Fast => self.fast += 1,
            Resolve::Coverage => self.coverage += 1,
            Resolve::Sample => self.sample += 1,
        }
    }

    pub fn add(&mut self, other: &ResolveCounts) {
        self.background += other.background;
        self.fast += other.fast;
        self.coverage += other.coverage;
        self.sample += other.sample;
    }
}

/// Fragments grouped by pixel, in the order they were added.
struct ByPixel<'f> {
    starts: Vec<u32>,
    items: Vec<Fragment<'f>>,
}

impl<'f> ByPixel<'f> {
    fn new(list: Vec<(u32, Fragment<'f>)>, pixels: usize) -> ByPixel<'f> {
        let mut starts = vec![0u32; pixels + 1];
        for (i, _) in &list {
            starts[*i as usize + 1] += 1;
        }
        for i in 0..pixels {
            starts[i + 1] += starts[i];
        }
        let mut next = starts.clone();
        let blank = Fragment::new(Plane::constant(0.0), [0; 4], Coverage::Mask(0), 0, 0);
        let mut items = vec![blank; list.len()];
        for (i, f) in list {
            let at = &mut next[i as usize];
            items[*at as usize] = f;
            *at += 1;
        }
        ByPixel { starts, items }
    }

    fn at(&self, pixel: usize) -> &[Fragment<'f>] {
        &self.items[self.starts[pixel] as usize..self.starts[pixel + 1] as usize]
    }
}

/// The opaque pass of one rectangle of the frame.
pub struct OpaquePass<'a> {
    rect: Rect,
    producers: &'a [&'a dyn Producer],
    pub(crate) source: u16,
    fragments: Vec<(u32, Fragment<'a>)>,
}

impl<'a> OpaquePass<'a> {
    pub(crate) fn new(rect: Rect, producers: &'a [&'a dyn Producer]) -> OpaquePass<'a> {
        OpaquePass {
            rect,
            producers,
            source: 0,
            fragments: Vec::new(),
        }
    }

    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// Adds an opaque fragment to pixel `(x, y)`. (A fragment that is not opaque is composited
    /// like any other but adds nothing to the opaque depth.)
    pub fn add(&mut self, x: u32, y: u32, mut fragment: Fragment<'a>) {
        fragment.source = self.source;
        let i = self.rect.index(x, y) as u32;
        self.fragments.push((i, fragment));
    }

    /// Ends the opaque pass: its depth becomes readable.
    pub(crate) fn finish(self) -> TransparentPass<'a> {
        let pixels = self.rect.pixels();
        TransparentPass {
            rect: self.rect,
            producers: self.producers,
            source: 0,
            opaque: ByPixel::new(self.fragments, pixels),
            fragments: Vec::new(),
            emission: Vec::new(),
        }
    }
}

/// The transparent pass of one rectangle, after its opaque pass.
pub struct TransparentPass<'a> {
    rect: Rect,
    producers: &'a [&'a dyn Producer],
    pub(crate) source: u16,
    opaque: ByPixel<'a>,
    fragments: Vec<(u32, Fragment<'a>)>,
    /// Linear RGB emitted into each pixel, made on the first emission.
    emission: Vec<[f64; 3]>,
}

/// Drawn pixels, a tile's or a frame's: their bytes, each pixel's [`Resolve`] code, row by row,
/// and the count of pixels per resolve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendered {
    pub rgba: Vec<u8>,
    pub resolves: Vec<u8>,
    pub counts: ResolveCounts,
}

impl<'a> TransparentPass<'a> {
    pub fn rect(&self) -> Rect {
        self.rect
    }

    fn mask_of(&self, f: &Fragment, x: u32, y: u32) -> u16 {
        match f.coverage {
            Coverage::Mask(m) => m,
            Coverage::Area { area, .. } if area >= 1.0 => FULL,
            Coverage::Area { id, .. } => self.producers[f.source as usize].mask(id, x, y),
        }
    }

    /// A fragment's 8-bit coverage at each sub-sample of the sample resolve: all or nothing for a
    /// mask, the exact area inside each sub-sample's cell for an area.
    fn sample_coverage(&self, f: &Fragment, x: u32, y: u32) -> [u8; SAMPLES] {
        match f.coverage {
            Coverage::Mask(m) => std::array::from_fn(|s| if m & (1 << s) != 0 { 255 } else { 0 }),
            Coverage::Area { area, .. } if area >= 1.0 => [255; SAMPLES],
            Coverage::Area { id, .. } => self.producers[f.source as usize]
                .cell_areas(id, x, y)
                .map(area_coverage),
        }
    }

    /// The opaque pass's depth at pixel `(x, y)`'s sub-samples: the nearest opaque fragment
    /// covering each, `+inf` where none does.
    pub fn opaque_depth(&self, x: u32, y: u32) -> [f64; SAMPLES] {
        let mut depth = [f64::INFINITY; SAMPLES];
        for f in self.opaque.at(self.rect.index(x, y)) {
            if !f.is_opaque() {
                continue;
            }
            let mask = self.mask_of(f, x, y);
            let at = Depths::of(self.producers, f, x, y).samples;
            for (s, d) in depth.iter_mut().enumerate() {
                if mask & (1 << s) != 0 {
                    *d = d.min(at[s]);
                }
            }
        }
        depth
    }

    /// Adds a fragment to pixel `(x, y)`: the one call through which every producer feeds the
    /// transparent pass.
    pub fn add(&mut self, x: u32, y: u32, mut fragment: Fragment<'a>) {
        fragment.source = self.source;
        let i = self.rect.index(x, y) as u32;
        self.fragments.push((i, fragment));
    }

    /// Adds light emitted into pixel `(x, y)`, at the sub-samples of its mask where it is nearer
    /// than the opaque depth.
    pub fn add_emission(&mut self, x: u32, y: u32, emission: Emission) {
        let opaque = self.opaque_depth(x, y);
        let visible = (0..SAMPLES)
            .filter(|s| {
                let [px, py] = sample_point(x, y, *s);
                emission.mask & (1 << s) != 0 && emission.depth.at(px, py) < opaque[*s]
            })
            .count();
        if visible == 0 {
            return;
        }
        if self.emission.is_empty() {
            self.emission = vec![[0.0; 3]; self.rect.pixels()];
        }
        let share = visible as f64 / SAMPLES as f64;
        let e = &mut self.emission[self.rect.index(x, y)];
        for (e, r) in e.iter_mut().zip(emission.radiance) {
            *e += r * share;
        }
    }

    /// Hands `visit` each pixel of the rectangle inside `keep` that holds a fragment, with its
    /// fragments (the opaque pass's, then the transparent pass's, as [`Self::resolve`] takes
    /// them), each one's 8-bit coverage at every sub-sample, as the sample resolve composites
    /// it, and its depths as the passes give them (the resolve orders by them once a stroke takes
    /// its own fill's depth: [`strokes_on_own_fills`]): the contributor report's read of the passes
    /// (`contributors.rs`). Nothing is resolved.
    pub(crate) fn visit_pixels(
        mut self,
        keep: Rect,
        mut visit: impl FnMut(u32, u32, &[Fragment], &[[u8; SAMPLES]], &[Depths]),
    ) {
        let pixels = self.rect.pixels();
        let transparent = ByPixel::new(std::mem::take(&mut self.fragments), pixels);
        let w = self.rect.width();
        let mut fragments = Vec::new();
        let mut coverage = Vec::new();
        let mut depths = Vec::new();
        for i in 0..pixels {
            let (x, y) = (self.rect.x0 + i as u32 % w, self.rect.y0 + i as u32 / w);
            if x < keep.x0 || x >= keep.x1 || y < keep.y0 || y >= keep.y1 {
                continue;
            }
            fragments.clear();
            fragments.extend_from_slice(self.opaque.at(i));
            fragments.extend_from_slice(transparent.at(i));
            if fragments.is_empty() {
                continue;
            }
            coverage.clear();
            coverage.extend(fragments.iter().map(|f| self.sample_coverage(f, x, y)));
            depths.clear();
            depths.extend(
                fragments
                    .iter()
                    .map(|f| Depths::of(self.producers, f, x, y)),
            );
            visit(x, y, &fragments, &coverage, &depths);
        }
    }

    /// Resolves every pixel of the rectangle over its background (the rectangle's bytes, row by
    /// row), then tone-maps the emitted light onto it.
    pub(crate) fn resolve(mut self, background: &[u8]) -> Rendered {
        let pixels = self.rect.pixels();
        assert_eq!(background.len(), pixels * 4);
        let transparent = ByPixel::new(std::mem::take(&mut self.fragments), pixels);
        let mut out = Rendered {
            rgba: background.to_vec(),
            resolves: vec![Resolve::Background as u8; pixels],
            counts: ResolveCounts::default(),
        };
        let mut scratch = Scratch::default();
        let w = self.rect.width();
        for i in 0..pixels {
            let (x, y) = (self.rect.x0 + i as u32 % w, self.rect.y0 + i as u32 / w);
            scratch.fragments.clear();
            scratch.fragments.extend_from_slice(self.opaque.at(i));
            scratch.fragments.extend_from_slice(transparent.at(i));
            let bg: [u8; 4] = background[i * 4..i * 4 + 4].try_into().unwrap();
            let (class, mut pixel) = if scratch.fragments.is_empty() {
                (Resolve::Background, bg)
            } else {
                resolve_pixel(
                    &mut scratch,
                    bg,
                    |_, f| self.sample_coverage(f, x, y),
                    |_, f| Depths::of(self.producers, f, x, y),
                )
            };
            match self.emission.get(i) {
                Some(e) if *e != [0.0; 3] => pixel = add_emission(pixel, *e),
                _ => {}
            }
            #[cfg(feature = "counters")]
            self.count_near_parallel(&scratch.fragments, x, y);
            out.rgba[i * 4..i * 4 + 4].copy_from_slice(&pixel);
            out.resolves[i] = class as u8;
            out.counts.count(class);
        }
        out
    }

    /// Counts, at pixel `(x, y)` holding `frags`, the sub-samples where two fragments are one
    /// depth only by the near-parallel rule (`order::near_parallel`: both cover the sub-sample, they
    /// share no identity, their keys as the pixel's order reads them lie within the term and not
    /// within their error bounds), once per pair and sub-sample, and the pixel once if any; and the
    /// pixel once more if such a pair's dihedral passes `depth::WIDE` (`counters` builds only:
    /// it reads the pixel again, apart from its resolve).
    #[cfg(feature = "counters")]
    fn count_near_parallel(&self, frags: &[Fragment], x: u32, y: u32) {
        use crate::order::{one_plane, own_layers, tied, within_term};
        let n = frags.len();
        // The pairs the rule may tie: both of flat records, on no one identity.
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
            .filter(|&(a, b)| {
                let (fa, fb) = (&frags[a], &frags[b]);
                fa.record.is_some()
                    && fb.record.is_some()
                    && !one_plane(fa, fb)
                    && !own_layers(fa, fb)
            })
            .collect();
        if pairs.is_empty() {
            return;
        }
        let cov: Vec<[u8; SAMPLES]> = frags
            .iter()
            .map(|f| self.sample_coverage(f, x, y))
            .collect();
        let mut rows: Vec<Depths> = frags
            .iter()
            .map(|f| Depths::of(self.producers, f, x, y))
            .collect();
        strokes_on_own_fills(frags, &mut rows, |i| cov[i]);
        let (mut samples, mut wide) = (0usize, false);
        for (a, b) in pairs {
            let (fa, fb) = (&frags[a], &frags[b]);
            let (Some(ra), Some(rb)) = (fa.record, fb.record) else {
                continue;
            };
            let tilt = crate::depth::near_parallel(ra.0, rb.0);
            if !tilt.near {
                continue;
            }
            let ((da, ea), (db, eb)) = (rows[a].against(fa, fb), rows[b].against(fb, fa));
            let alone = (0..SAMPLES)
                .filter(|&k| {
                    cov[a][k] != 0
                        && cov[b][k] != 0
                        && !tied(da[k], ea, db[k], eb)
                        && within_term(fa, da[k], ea, fb, db[k], eb)
                })
                .count();
            samples += alone;
            wide |= alone > 0 && tilt.dihedral > crate::depth::WIDE;
        }
        if samples > 0 {
            count!(NEAR_PARALLEL_SAMPLES, samples);
            count!(NEAR_PARALLEL_PX);
            if wide {
                count!(NEAR_PARALLEL_WIDE_PX);
            }
        }
    }
}

/// Buffers reused from pixel to pixel.
#[derive(Default)]
pub(crate) struct Scratch<'f> {
    pub(crate) fragments: Vec<Fragment<'f>>,
    /// Each fragment's depths, its strokes on its own fills ([`strokes_on_own_fills`]).
    rows: Vec<Depths>,
    order: Vec<usize>,
    /// Runs `start..end` of `order` that composite as one fragment.
    units: Vec<(usize, usize)>,
    /// Each fragment's coverage at each sub-sample, for the sample resolve.
    coverage: Vec<[u8; SAMPLES]>,
    covering: Vec<usize>,
    /// `fronts[a * n + b]`: fragment `a` is in front of fragment `b` at the pixel.
    fronts: Vec<bool>,
    /// `strong[a * n + b]`: depth decided that pair ([`crate::order::Decided::is_strong`]).
    strong: Vec<bool>,
    /// `at[a * n + b]`: fragment `a` against fragment `b` at each sub-sample both cover
    /// ([`crate::order::shared`]), what the sub-sample orders read; empty for a flat pixel.
    at: Vec<crate::order::AtSamples>,
    samples: SampleRanks,
    /// Evaluate each sub-sample order where it is read, as before [`Scratch::at`] held them: the
    /// tests' reference for the orders read from it.
    #[cfg(test)]
    unhoisted: bool,
}

/// Each sub-sample's fragments in the order the sample resolve composites them, and which of them
/// show: what some sub-sample's rank puts in front of every fragment that hides the pixel. The
/// culls read it, so what they drop is what the ranks hide, and the sample resolve composites from
/// the same ranks.
#[derive(Default)]
struct SampleRanks {
    /// `ranks[ends[k - 1]..ends[k]]`: sub-sample `k`'s fragments, front first.
    ranks: Vec<usize>,
    ends: Vec<usize>,
    /// `shown[i]`: some sub-sample shows fragment `i`.
    shown: Vec<bool>,
    covering: Vec<usize>,
    ranked: Vec<usize>,
}

impl SampleRanks {
    fn is_ranked(&self) -> bool {
        !self.ends.is_empty()
    }

    fn at(&self, k: usize) -> &[usize] {
        let start = if k == 0 { 0 } else { self.ends[k - 1] };
        &self.ranks[start..self.ends[k]]
    }

    /// Rank `items` at each sub-sample from `front(a, b, k)` and `strong(a, b, k)`, as
    /// [`crate::order::rank_sample`] does.
    // The sub-sample `k` indexes each fragment's coverage row.
    #[allow(clippy::needless_range_loop)]
    fn rank(
        &mut self,
        cov: &[[u8; SAMPLES]],
        items: &[usize],
        front: impl Fn(usize, usize, usize) -> bool,
        strong: impl Fn(usize, usize, usize) -> bool,
        painted: impl Fn(usize, usize) -> Ordering,
    ) {
        self.ranks.clear();
        self.ends.clear();
        for k in 0..SAMPLES {
            self.covering.clear();
            self.covering
                .extend(items.iter().copied().filter(|i| cov[*i][k] != 0));
            crate::order::rank_sample(
                &self.covering,
                |a, b| front(a, b, k),
                |a, b| strong(a, b, k),
                &painted,
                &mut self.ranked,
            );
            self.ranks.extend_from_slice(&self.ranked);
            self.ends.push(self.ranks.len());
        }
    }

    /// The pixel the sub-samples make over `bg`, each compositing the fragments covering it in its
    /// own order, as ranked: their mean.
    // The sub-sample `k` indexes each fragment's coverage row.
    #[allow(clippy::needless_range_loop)]
    fn composite(&self, frags: &[Fragment], cov: &[[u8; SAMPLES]], bg: [u8; 4]) -> [u8; 4] {
        let mut sum = [0u32; 4];
        for k in 0..SAMPLES {
            let mut px = bg;
            for i in self.at(k).iter().rev() {
                px = blend(frags[*i].colour, cov[*i][k], px);
            }
            for (s, c) in sum.iter_mut().zip(px) {
                *s += c as u32;
            }
        }
        let n = SAMPLES as u32;
        sum.map(|v| ((v + n / 2) / n) as u8)
    }

    /// Which of `n` fragments some sub-sample shows: those its rank puts in front of every
    /// fragment `hides`, whose layer, being opaque and covering the pixel whole, leaves nothing
    /// behind it to show in area as well as at the sub-sample.
    fn show(&mut self, n: usize, hides: impl Fn(usize) -> bool) {
        self.shown.clear();
        self.shown.resize(n, false);
        for k in 0..SAMPLES {
            let start = if k == 0 { 0 } else { self.ends[k - 1] };
            for &i in &self.ranks[start..self.ends[k]] {
                self.shown[i] = true;
                if hides(i) {
                    break;
                }
            }
        }
    }
}

/// The pixel that `s.fragments` composite over `bg`, and the resolve that made it.
/// `sample_coverage(i, f)` is fragment `i`'s coverage at each sub-sample, `sample_depth(i, f)` its
/// depths there with their error bound, before a stroke takes its own fill's depth where nearer
/// ([`strokes_on_own_fills`]). The fragments composite in the order [`crate::order`]
/// gives them: by depth only where two cover one sub-sample, by paint order everywhere else.
/// The contributor report calls it on one leaf's fragments alone, so what a leaf draws is what
/// this resolve makes of it.
// The sub-sample `k` indexes each fragment's coverage and depth rows.
#[allow(clippy::needless_range_loop)]
pub(crate) fn resolve_pixel(
    s: &mut Scratch,
    bg: [u8; 4],
    sample_coverage: impl Fn(usize, &Fragment) -> [u8; SAMPLES],
    sample_depth: impl Fn(usize, &Fragment) -> Depths,
) -> (Resolve, [u8; 4]) {
    use crate::order;
    let frags = &s.fragments;
    if let [f] = frags.as_slice() {
        if f.is_opaque() && f.is_full() {
            return (Resolve::Fast, f.colour);
        }
        return (Resolve::Coverage, blend(f.colour, f.coverage8(), bg));
    }
    s.rows.clear();
    s.rows
        .extend(frags.iter().enumerate().map(|(i, f)| sample_depth(i, f)));
    strokes_on_own_fills(frags, &mut s.rows, |i| sample_coverage(i, &frags[i]));
    // One depth row for all (a flat scene): paint order decides every pair whatever they cover,
    // so the coverage rows wait for the sample resolve.
    let first = s.rows[0].samples;
    let flat = s
        .rows
        .iter()
        .all(|d| d.samples == first && d.rows().all(|(on, _)| *on == first));
    s.coverage.clear();
    if !flat {
        s.coverage
            .extend(frags.iter().enumerate().map(|(i, f)| sample_coverage(i, f)));
    }
    let (rows, cov) = (&s.rows, &s.coverage);
    let painted = |a: usize, b: usize| order::painted(&frags[a], &frags[b]);
    // `a`'s depths and their bound against `b` ([`Depths::against`]).
    let read = |a: usize, b: usize| rows[a].against(&frags[a], &frags[b]);
    // `a` against `b` at sub-sample `k`, where both cover it, evaluated there.
    let at = |a: usize, b: usize, k: usize| {
        let ((da, ea), (db, eb)) = (read(a, b), read(b, a));
        order::at_sample(&frags[a], da[k], ea, &frags[b], db[k], eb)
    };
    s.samples.ends.clear();
    let seen = |i: usize, against: usize| {
        let (depth, err) = read(i, against);
        order::Seen {
            fragment: &frags[i],
            coverage: &cov[i],
            depth,
            err,
        }
    };
    // Each pair's order, once: `order::pair` of `b` and `a` is that of `a` and `b` reversed. From
    // the same reads, each pair's orders at every sub-sample both cover, both ways
    // (`order::shared`), which the sub-sample orders below read rather than evaluate again.
    let n = frags.len();
    s.fronts.clear();
    s.fronts.resize(n * n, false);
    s.strong.clear();
    s.strong.resize(n * n, false);
    s.at.clear();
    if !flat {
        s.at.resize(n * n, order::AtSamples::default());
    }
    for a in 0..n {
        for b in a + 1..n {
            let (o, how) = if flat {
                (painted(a, b), order::Decided::Paint)
            } else {
                let shared = order::shared(seen(a, b), seen(b, a));
                (s.at[a * n + b], s.at[b * n + a]) = (shared.ab, shared.ba);
                shared.decided(&frags[a], &frags[b])
            };
            s.fronts[a * n + b] = o == Ordering::Less;
            s.fronts[b * n + a] = o == Ordering::Greater;
            s.strong[a * n + b] = how.is_strong();
            s.strong[b * n + a] = how.is_strong();
        }
    }
    let (fronts, strong) = (&s.fronts, &s.strong);
    // The sub-sample orders at a sub-sample both cover, as `SampleRanks::rank` and the units read
    // them: `a` in front of `b`, behind it, and depth deciding. Read from `s.at`; a flat pixel has
    // none, and evaluates them.
    #[cfg(test)]
    let unhoisted = s.unhoisted;
    #[cfg(not(test))]
    let unhoisted = false;
    let table = (!flat && !unhoisted).then_some(&s.at);
    let bit = |mask: u16, k: usize| mask >> k & 1 != 0;
    let front_at = |a: usize, b: usize, k: usize| {
        a != b
            && match table {
                Some(t) => bit(t[a * n + b].front, k),
                None => at(a, b, k) == Ordering::Less,
            }
    };
    let behind_at = |a: usize, b: usize, k: usize| match table {
        Some(t) => bit(t[a * n + b].behind, k),
        None => at(a, b, k) == Ordering::Greater,
    };
    let strong_at = |a: usize, b: usize, k: usize| match table {
        Some(t) => bit(t[a * n + b].strong, k),
        None => {
            let ((da, ea), (db, eb)) = (read(a, b), read(b, a));
            order::strong_at(&frags[a], da[k], ea, &frags[b], db[k], eb)
        }
    };
    s.covering.clear();
    s.covering.extend(0..n);
    let how = order::rank(
        &s.covering,
        |a, b| fronts[a * n + b],
        |a, b| strong[a * n + b],
        painted,
        &mut s.order,
    );
    // Where the strong orders cycle the pixel has no order: each sub-sample composites in its own,
    // where a layer covering the pixel whole hides what that order puts behind it.
    if how == order::Ranked::Strong {
        s.samples
            .rank(cov, &s.covering, front_at, strong_at, painted);
        return (Resolve::Sample, s.samples.composite(frags, cov, bg));
    }

    // Where an opaque fragment covers the pixel whole, drop what the sub-samples' ranks hide: a
    // fragment that no sub-sample's rank puts in front of every such fragment. One that covers no
    // sub-sample is kept where the pixel's rank puts it in front of the first. Flat, every
    // sub-sample's order is paint order, the pixel's: what that puts behind the first is hidden.
    let covers_whole = |i: usize| frags[i].is_opaque() && frags[i].is_full();
    if let Some(p) = s.order.iter().position(|i| covers_whole(*i)) {
        if flat {
            s.order.truncate(p + 1);
        } else {
            s.samples.rank(cov, &s.order, front_at, strong_at, painted);
            s.samples.show(n, covers_whole);
            let shown = &s.samples.shown;
            let mut kept = 0;
            for q in 0..s.order.len() {
                let f = s.order[q];
                let covers = cov[f].iter().any(|c| *c != 0);
                if shown[f] || !covers && q < p {
                    s.order[kept] = f;
                    kept += 1;
                }
            }
            s.order.truncate(kept);
        }
        if let &[o] = s.order.as_slice()
            && covers_whole(o)
        {
            return (Resolve::Fast, frags[o].colour);
        }
    }

    // Units: runs of the order that composite as one fragment. A run of one surface's mask
    // fragments whose masks do not overlap is one unit (a surface tiling the pixel), the others
    // one each.
    s.units.clear();
    let mut union = 0u16;
    for q in 0..s.order.len() {
        let f = &frags[s.order[q]];
        let joins = match (s.units.last(), f.coverage) {
            (Some(&(start, _)), Coverage::Mask(m)) => {
                let first = &frags[s.order[start]];
                f.surface != 0
                    && f.surface == first.surface
                    && f.source == first.source
                    && matches!(first.coverage, Coverage::Mask(_))
                    && union & m == 0
            }
            _ => false,
        };
        let m = match f.coverage {
            Coverage::Mask(m) => m,
            Coverage::Area { .. } => 0,
        };
        if joins {
            s.units.last_mut().unwrap().1 = q + 1;
            union |= m;
        } else {
            s.units.push((q, q + 1));
            union = m;
        }
    }
    let order = &s.order;
    // Whether every fragment of unit `u` is in front of every fragment of unit `v` at sub-sample
    // k, where both cover it.
    let before = |u: (usize, usize), v: (usize, usize), k: usize| {
        order[u.0..u.1].iter().all(|a| {
            order[v.0..v.1]
                .iter()
                .all(|b| !flat && (cov[*a][k] == 0 || cov[*b][k] == 0) || !behind_at(*a, *b, k))
        })
    };
    let unit_mask = |u: (usize, usize)| {
        order[u.0..u.1]
            .iter()
            .try_fold(0u16, |m, i| match frags[*i].coverage {
                Coverage::Mask(f) => Some(m | f),
                Coverage::Area { .. } => None,
            })
    };
    let opaque_full = |u: (usize, usize)| {
        order[u.0..u.1].iter().all(|i| frags[*i].is_opaque()) && unit_mask(u) == Some(FULL)
    };
    // A merged opaque unit covering the pixel (a surface tiling it): the units after it go where the
    // sub-samples' ranks hide every fragment of theirs behind it, or behind an opaque fragment
    // covering the pixel whole (flat, paint order hides them all).
    let mut units: &[(usize, usize)] = &s.units;
    if let Some(p) = units.iter().position(|u| u.1 - u.0 > 1 && opaque_full(*u)) {
        let hidden = flat || {
            if !s.samples.is_ranked() {
                s.samples.rank(cov, order, front_at, strong_at, painted);
            }
            let tile = &order[units[p].0..units[p].1];
            s.samples.show(n, |i| covers_whole(i) || tile.contains(&i));
            let shown = &s.samples.shown;
            units[p + 1..]
                .iter()
                .all(|w| order[w.0..w.1].iter().all(|i| !shown[*i]))
        };
        if hidden {
            units = &units[..=p];
        }
    }
    let partial = |u: &(usize, usize)| unit_mask(*u).is_some_and(|m| m != FULL);
    let mut partial_masks = units.iter().filter(|u| partial(u)).count();
    // Masks come from surfaces, so paths alone never reach the sample resolve this way.
    let partial_area = units
        .iter()
        .any(|u| matches!(frags[order[u.0]].coverage, Coverage::Area { area, .. } if area < 1.0));
    // A surface tiling the pixel in several colours composites as their average, which is exact
    // only while everything else covers the whole pixel: beside a partial layer, which covers some
    // of its colours and not others, it counts as partial itself.
    if partial_masks > 0 || partial_area {
        partial_masks += units
            .iter()
            .filter(|u| {
                let first = frags[order[u.0]].colour;
                !partial(u) && order[u.0..u.1].iter().any(|i| frags[*i].colour != first)
            })
            .count();
    }
    let one_order = (partial_masks == 0 || partial_masks == 1 && !partial_area)
        && (0..SAMPLES).all(|k| {
            units
                .iter()
                .enumerate()
                .all(|(p, u)| units[p + 1..].iter().all(|v| before(*u, *v, k)))
        });
    if one_order {
        let mut px = bg;
        for u in units.iter().rev() {
            let (colour, coverage) = if u.1 - u.0 == 1 {
                let f = &frags[order[u.0]];
                (f.colour, f.coverage8())
            } else {
                // The members' colours averaged over the sub-samples each covers.
                let mut sum = [0u32; 4];
                let mut count = 0u32;
                for i in &order[u.0..u.1] {
                    let f = &frags[*i];
                    let n = match f.coverage {
                        Coverage::Mask(m) => m.count_ones(),
                        Coverage::Area { .. } => unreachable!("units merge masks"),
                    };
                    count += n;
                    for (s, c) in sum.iter_mut().zip(f.colour) {
                        *s += c as u32 * n;
                    }
                }
                let union = unit_mask(*u).expect("units merge masks");
                (
                    sum.map(|v| ((v + count / 2) / count) as u8),
                    mask_coverage(union),
                )
            };
            px = blend(colour, coverage, px);
        }
        return (Resolve::Coverage, px);
    }

    if flat {
        s.coverage.resize(frags.len(), [0; SAMPLES]);
        for i in &s.order {
            s.coverage[*i] = sample_coverage(*i, &frags[*i]);
        }
    }
    let cov = &s.coverage;
    // As the culls ranked them, which may hold what they dropped: behind a fragment opaque and
    // covering the sub-sample whole, which blends over it exactly.
    if !s.samples.is_ranked() {
        s.samples.rank(cov, &s.order, front_at, strong_at, painted);
    }
    (Resolve::Sample, s.samples.composite(frags, cov, bg))
}

/// A frame being drawn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl Frame {
    /// A frame of one colour.
    pub fn filled(width: u32, height: u32, pixel: [u8; 4]) -> Frame {
        let n = width as usize * height as usize;
        let mut rgba = Vec::with_capacity(n * 4);
        for _ in 0..n {
            rgba.extend_from_slice(&pixel);
        }
        Frame {
            width,
            height,
            rgba,
        }
    }

    /// The top-left `width x height` of an image (straight RGBA, row-major), as CE crops a
    /// background image (`np.array(image)[:height, :width]`); `None` when the image is smaller.
    pub fn cropped(
        width: u32,
        height: u32,
        image_width: u32,
        image_height: u32,
        image: &[u8],
    ) -> Option<Frame> {
        if image_width < width || image_height < height {
            return None;
        }
        let row = width as usize * 4;
        let stride = image_width as usize * 4;
        let mut rgba = Vec::with_capacity(row * height as usize);
        for y in 0..height as usize {
            rgba.extend_from_slice(&image[y * stride..y * stride + row]);
        }
        Some(Frame {
            width,
            height,
            rgba,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The frame's bytes.
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// The frame's bytes.
    pub fn into_rgba(self) -> Vec<u8> {
        self.rgba
    }
}

/// A background colour's bytes as CE writes them: `(rgba * 255).astype(int)` stored into uint8,
/// truncated toward zero and wrapped as NumPy's cast wraps, not premultiplied.
pub fn background_pixel(rgba: [f64; 4]) -> [u8; 4] {
    rgba.map(|c| (c * 255.0) as i64 as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_byte_products_round_within_their_stated_errors() {
        // Every pair of bytes: the widest `|255 mul(a, b) - a b|`, which is 255 times the error.
        let widest = |f: fn(u8, u8) -> u8| {
            (0..=255u8)
                .flat_map(|a| (0..=255u8).map(move |b| (a, b)))
                .map(|(a, b)| (255 * f(a, b) as i64 - a as i64 * b as i64).abs())
                .max()
                .unwrap()
        };
        assert_eq!(widest(mul) as f64 / 255.0, MUL_ERROR);
        assert_eq!(widest(mul_7f) as f64 / 255.0, MUL_7F_ERROR);
    }

    #[test]
    fn a_stroke_is_read_on_its_path_s_plane_only_against_what_shares_that_plane() {
        let on = |key: f64, plane: [u32; 4]| {
            let mut f = Fragment::new(Plane::constant(key), [0; 4], Coverage::Mask(u16::MAX), 1, 0);
            f.plane = plane;
            f
        };
        let set = |p: u32| [p, 0, 0, 0];
        let (stroke, same, other, none) = (
            on(-1.0, set(7)),
            on(-2.0, set(7)),
            on(-2.0, set(9)),
            on(-2.0, set(0)),
        );
        let mut d = Depths::of_plane(&stroke, 0, 0);
        let own = (d.samples, d.err);
        d.on_plane = Some(([-0.5; SAMPLES], 1e-9));
        let read = |d: &Depths, a: &Fragment, b: &Fragment| {
            let (s, e) = d.against(a, b);
            (*s, e)
        };
        assert_eq!(read(&d, &stroke, &same), ([-0.5; SAMPLES], 1e-9));
        assert_eq!(read(&d, &stroke, &other), own);
        assert_eq!(read(&d, &stroke, &none), own);
        // A line on two identities is read on the plane of the one it shares.
        let line = on(-1.0, [3, 7, 0, 0]);
        d.more = vec![Some(([-0.6; SAMPLES], 2e-9))];
        assert_eq!(read(&d, &line, &same), ([-0.6; SAMPLES], 2e-9));
        assert_eq!(read(&d, &line, &on(-2.0, set(3))), ([-0.5; SAMPLES], 1e-9));
        assert_eq!(read(&d, &line, &other), own);
        // A fragment with no plane row reads its own against anything.
        d.on_plane = None;
        d.more.clear();
        assert_eq!(read(&d, &stroke, &same), own);
        assert_eq!(read(&d, &line, &same), own);
    }

    /// A pseudo-random pixel: `n` fragments of paths' layers and meshes, on registered planes or
    /// none, sloped or flat, some read on a plane row against what shares it (so a fragment's
    /// depths against one differ from its depths against another), at depths and bounds that tie,
    /// part within their terms or part beyond them, translucent or opaque, covering the pixel
    /// whole or a few sub-samples; with each fragment's coverage at each sub-sample and depths.
    fn random_pixel(
        next: &mut impl FnMut(u64) -> u64,
    ) -> (Vec<Fragment<'static>>, Vec<[u8; SAMPLES]>, Vec<Depths>) {
        let n = 2 + next(6) as usize;
        let (mut frags, mut rows, mut depths) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let slope = [0.0, 0.05, -0.05, 0.3][next(4) as usize];
            let key = next(6) as f64 * 0.04;
            let plane = Plane {
                err: [0.0, 0.01, 0.05][next(3) as usize],
                ..Plane::new(key - slope * 0.5, slope, [0.0, 0.02][next(2) as usize])
            };
            let mask = match next(3) {
                0 => FULL,
                _ => (next(1 << 16) as u16) | 1 << next(16),
            };
            let coverage = if next(2) == 0 {
                Coverage::Mask(mask)
            } else {
                let area = if mask == FULL { 1.0 } else { 0.5 };
                Coverage::Area { area, id: 0 }
            };
            let colour = if next(3) == 0 {
                [0, 60, 120, 128]
            } else {
                [next(256) as u8, 40, 200, 255]
            };
            let mut f = Fragment::new(plane, colour, coverage, next(4) as u32, next(6) as u32);
            f.path_layer = next(3) != 0;
            f.plane = [[0, 0, 0, 0], [7, 0, 0, 0], [9, 0, 0, 0], [3, 7, 0, 0]][next(4) as usize];
            f.snap = [0.0, 0.05, f64::INFINITY][next(3) as usize];
            if f.plane[0] == 3 || next(4) == 0 {
                f.own = [0.0, 0.1][next(2) as usize];
            }
            if next(4) == 0 {
                f.wide = 0.1;
            }
            let mut d = Depths::of_plane(&f, 0, 0);
            if f.path_layer && f.plane[0] != 0 && next(2) == 0 {
                let off = [-0.1, 0.0, 0.1][next(3) as usize];
                d.on_plane = Some((d.samples.map(|v| v + off), 0.01));
                if f.plane[1] != 0 {
                    d.more = vec![Some((d.samples.map(|v| v - off), 0.0))];
                }
            }
            let row: [u8; SAMPLES] =
                std::array::from_fn(|k| if mask >> k & 1 != 0 { 255 } else { 0 });
            frags.push(f);
            rows.push(row);
            depths.push(d);
        }
        (frags, rows, depths)
    }

    /// The sub-sample orders a pixel reads from its pairs' (`Scratch::at`, filled from
    /// `order::shared`) are those it evaluates where it reads them: on pseudo-random pixels, the
    /// resolve gives one pixel both ways, ranks every sub-sample one way, keeps the same order and
    /// shows the same fragments; among them pixels whose strong orders cycle, pixels a layer
    /// covering them whole culls through the sub-sample ranks, and sub-samples whose rank breaks
    /// a cycle through a weak order.
    #[test]
    fn the_sub_sample_orders_read_from_the_pairs_are_those_evaluated_where_read() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let (mut cycles, mut culls, mut weak_subs, mut sampled) = (0, 0, 0, 0);
        for _ in 0..40_000 {
            let (frags, rows, depths) = random_pixel(&mut next);
            let run = |unhoisted: bool| {
                let mut s = Scratch {
                    unhoisted,
                    ..Scratch::default()
                };
                s.fragments.extend_from_slice(&frags);
                let got = resolve_pixel(
                    &mut s,
                    [10, 20, 30, 255],
                    |i, _| rows[i],
                    |i, _| depths[i].clone(),
                );
                let ranks = (s.samples.ranks.clone(), s.samples.ends.clone());
                (got, s.order.clone(), ranks, s.samples.shown.clone())
            };
            let hoisted = run(false);
            assert_eq!(hoisted, run(true), "{frags:?} {rows:?} {depths:?}");
            sampled += (hoisted.0.0 == Resolve::Sample) as usize;
            // Which of the paths that read the sub-sample orders the pixel took.
            let n = frags.len();
            let mut d = depths.clone();
            strokes_on_own_fills(&frags, &mut d, |i| rows[i]);
            let read = |a: usize, b: usize| d[a].against(&frags[a], &frags[b]);
            let seen = |i: usize, against: usize| {
                let (depth, err) = read(i, against);
                crate::order::Seen {
                    fragment: &frags[i],
                    coverage: &rows[i],
                    depth,
                    err,
                }
            };
            let pair = |a: usize, b: usize| crate::order::pair(seen(a, b), seen(b, a));
            let painted = |a: usize, b: usize| crate::order::painted(&frags[a], &frags[b]);
            let mut out = Vec::new();
            let items: Vec<usize> = (0..n).collect();
            let how = crate::order::rank(
                &items,
                |a, b| a != b && pair(a, b).0 == Ordering::Less,
                |a, b| pair(a, b).1.is_strong(),
                painted,
                &mut out,
            );
            let flat = d
                .iter()
                .all(|r| r.samples == d[0].samples && r.rows().all(|(on, _)| *on == d[0].samples));
            if flat {
                continue;
            }
            if how == crate::order::Ranked::Strong {
                cycles += 1;
            } else if out
                .iter()
                .any(|i| frags[*i].is_opaque() && frags[*i].is_full())
            {
                culls += 1;
            }
            for k in 0..SAMPLES {
                let covering: Vec<usize> = (0..n).filter(|i| rows[*i][k] != 0).collect();
                let at = |a: usize, b: usize| {
                    let ((da, ea), (db, eb)) = (read(a, b), read(b, a));
                    (&frags[a], da[k], ea, &frags[b], db[k], eb)
                };
                let how = crate::order::rank_sample(
                    &covering,
                    |a, b| {
                        let (fa, da, ea, fb, db, eb) = at(a, b);
                        a != b && crate::order::at_sample(fa, da, ea, fb, db, eb).is_lt()
                    },
                    |a, b| {
                        let (fa, da, ea, fb, db, eb) = at(a, b);
                        crate::order::strong_at(fa, da, ea, fb, db, eb)
                    },
                    painted,
                    &mut out,
                );
                weak_subs += (how == crate::order::Ranked::Weak) as usize;
            }
        }
        assert!(
            cycles > 100 && culls > 100 && weak_subs > 100 && sampled > 100,
            "{cycles} {culls} {weak_subs} {sampled}"
        );
    }

    /// `order::shared`'s masks are `order::at_sample`'s and `order::strong_at`'s orders both ways
    /// at each sub-sample both cover, and nothing elsewhere; and `order::pair` from them decides
    /// as counting each shared sub-sample's tie in turn does.
    #[test]
    fn a_pair_s_shared_orders_are_its_orders_at_each_sub_sample_both_ways() {
        use crate::order::{
            Decided, Seen, at_sample, coplanar_tied, own_layers, painted, strong_at,
        };
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let mut decided = [0usize; 4];
        for _ in 0..20_000 {
            let (frags, rows, depths) = random_pixel(&mut next);
            for a in 0..frags.len() {
                for b in 0..frags.len() {
                    if a == b {
                        continue;
                    }
                    let (fa, fb) = (&frags[a], &frags[b]);
                    let ((da, ea), (db, eb)) =
                        (depths[a].against(fa, fb), depths[b].against(fb, fa));
                    let sa = Seen {
                        fragment: fa,
                        coverage: &rows[a],
                        depth: da,
                        err: ea,
                    };
                    let sb = Seen {
                        fragment: fb,
                        coverage: &rows[b],
                        depth: db,
                        err: eb,
                    };
                    let s = crate::order::shared(sa, sb);
                    let (mut shared, mut a_front, mut b_front) = (0, 0, 0);
                    for k in 0..SAMPLES {
                        let bit = |m: u16| m >> k & 1 != 0;
                        let both = rows[a][k] != 0 && rows[b][k] != 0;
                        assert_eq!(bit(s.covered), both);
                        let (ab, ba) = if both {
                            (
                                at_sample(fa, da[k], ea, fb, db[k], eb),
                                at_sample(fb, db[k], eb, fa, da[k], ea),
                            )
                        } else {
                            (Ordering::Equal, Ordering::Equal)
                        };
                        let strong = both && strong_at(fa, da[k], ea, fb, db[k], eb);
                        assert_eq!(strong, both && strong_at(fb, db[k], eb, fa, da[k], ea));
                        for (at, o) in [(s.ab, ab), (s.ba, ba)] {
                            assert_eq!(
                                (bit(at.front), bit(at.behind), bit(at.strong)),
                                (o.is_lt(), o.is_gt(), strong),
                                "{a} {b} {k}"
                            );
                        }
                        // `pair` as it counted them.
                        if both {
                            shared += 1;
                            if !crate::order::tied(da[k], ea, db[k], eb)
                                && !coplanar_tied(fa, da[k], ea, fb, db[k], eb)
                            {
                                if da[k] < db[k] {
                                    a_front += 1;
                                } else {
                                    b_front += 1;
                                }
                            }
                        }
                    }
                    let paint = painted(fa, fb);
                    let want = if shared > 0 && own_layers(fa, fb) {
                        (paint, Decided::Paint)
                    } else {
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
                    };
                    assert_eq!(crate::order::pair(sa, sb), want);
                    decided[want.1 as usize] += 1;
                }
            }
        }
        assert!(decided.iter().all(|c| *c > 1000), "{decided:?}");
    }

    #[test]
    fn the_derived_bounds_grow_by_one_per_layer_in_front() {
        assert_eq!(
            coverage_to_sample_bound(Partial::Mask, 1),
            COVERAGE_TO_SAMPLE_LSB
        );
        for fronts in 0..8 {
            assert_eq!(
                coverage_to_sample_bound(Partial::Mask, fronts),
                2 + fronts as u8
            );
            assert_eq!(
                coverage_to_sample_bound(Partial::MergedMask, fronts),
                3 + fronts as u8
            );
            assert_eq!(
                coverage_to_sample_bound(Partial::Path, fronts),
                4 + fronts as u8
            );
        }
    }
}
