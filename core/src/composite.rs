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
//! Fragments hidden at every sub-sample behind an opaque covering fragment are dropped first. A
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
}

impl Plane {
    /// One depth everywhere.
    pub const fn constant(depth: f64) -> Plane {
        Plane {
            a: depth,
            b: 0.0,
            c: 0.0,
        }
    }

    /// The plane through three `(x, y, depth)` points. It is constant where they share one depth
    /// (the mean where they are collinear, the nearest where a depth is not finite).
    pub fn through(points: [[f64; 3]; 3]) -> Plane {
        let [[x0, y0, d0], [x1, y1, d1], [x2, y2, d2]] = points;
        if d0 == d1 && d1 == d2 {
            return Plane::constant(d0);
        }
        if !(d0.is_finite() && d1.is_finite() && d2.is_finite()) {
            return Plane::constant(d0.min(d1).min(d2));
        }
        let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if det == 0.0 || !det.is_finite() {
            return Plane::constant((d0 + d1 + d2) / 3.0);
        }
        let b = ((d1 - d0) * (y2 - y0) - (d2 - d0) * (y1 - y0)) / det;
        let c = ((x1 - x0) * (d2 - d0) - (x2 - x0) * (d1 - d0)) / det;
        Plane {
            a: d0 - b * x0 - c * y0,
            b,
            c,
        }
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

/// One drawn object's share of one pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fragment {
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
    /// The producer that added it (set by the pass).
    source: u16,
}

impl Fragment {
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
    /// Adds the producer's opaque fragments in the pass's rectangle.
    fn opaque(&self, _pass: &mut OpaquePass<'_>) {}

    /// Adds the producer's other fragments and emission in the pass's rectangle; the opaque depth
    /// is readable here.
    fn transparent(&self, _pass: &mut TransparentPass<'_>) {}

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
struct ByPixel {
    starts: Vec<u32>,
    items: Vec<Fragment>,
}

impl ByPixel {
    fn new(list: Vec<(u32, Fragment)>, pixels: usize) -> ByPixel {
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

    fn at(&self, pixel: usize) -> &[Fragment] {
        &self.items[self.starts[pixel] as usize..self.starts[pixel + 1] as usize]
    }
}

/// The opaque pass of one rectangle of the frame.
pub struct OpaquePass<'a> {
    rect: Rect,
    producers: &'a [&'a dyn Producer],
    pub(crate) source: u16,
    fragments: Vec<(u32, Fragment)>,
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
    pub fn add(&mut self, x: u32, y: u32, mut fragment: Fragment) {
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
    opaque: ByPixel,
    fragments: Vec<(u32, Fragment)>,
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

impl TransparentPass<'_> {
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
            for (s, d) in depth.iter_mut().enumerate() {
                if mask & (1 << s) != 0 {
                    let [px, py] = sample_point(x, y, s);
                    *d = d.min(f.depth.at(px, py));
                }
            }
        }
        depth
    }

    /// Adds a fragment to pixel `(x, y)`: the one call through which every producer feeds the
    /// transparent pass.
    pub fn add(&mut self, x: u32, y: u32, mut fragment: Fragment) {
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
    /// them) and each one's 8-bit coverage at every sub-sample, as the sample resolve composites
    /// it: the contributor report's read of the passes (`contributors.rs`). Nothing is resolved.
    pub(crate) fn visit_pixels(
        mut self,
        keep: Rect,
        mut visit: impl FnMut(u32, u32, &[Fragment], &[[u8; SAMPLES]]),
    ) {
        let pixels = self.rect.pixels();
        let transparent = ByPixel::new(std::mem::take(&mut self.fragments), pixels);
        let w = self.rect.width();
        let mut fragments = Vec::new();
        let mut coverage = Vec::new();
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
            visit(x, y, &fragments, &coverage);
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
                resolve_pixel(&mut scratch, bg, x, y, |_, f| self.sample_coverage(f, x, y))
            };
            match self.emission.get(i) {
                Some(e) if *e != [0.0; 3] => pixel = add_emission(pixel, *e),
                _ => {}
            }
            out.rgba[i * 4..i * 4 + 4].copy_from_slice(&pixel);
            out.resolves[i] = class as u8;
            out.counts.count(class);
        }
        out
    }
}

/// Buffers reused from pixel to pixel.
#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) fragments: Vec<Fragment>,
    depths: Vec<[f64; SAMPLES]>,
    order: Vec<usize>,
    /// Runs `start..end` of `order` that composite as one fragment.
    units: Vec<(usize, usize)>,
    /// Each fragment's coverage at each sub-sample, for the sample resolve.
    coverage: Vec<[u8; SAMPLES]>,
    covering: Vec<usize>,
}

/// Whether fragment `a` (depth `da`) is in front of fragment `b` (depth `db`) at one sub-sample:
/// nearer first, then later in the draw order, then the later part.
pub(crate) fn front_to_back(a: &Fragment, da: f64, b: &Fragment, db: f64) -> Ordering {
    da.total_cmp(&db)
        .then(b.order.cmp(&a.order))
        .then(b.sub.cmp(&a.sub))
}

/// The pixel `(x, y)` that `s.fragments` composite over `bg`, and the resolve that made it.
/// `sample_coverage(i, f)` is fragment `i`'s coverage at each sub-sample, for the sample resolve.
/// The contributor report calls it on one leaf's fragments alone, so what a leaf draws is what
/// this resolve makes of it.
pub(crate) fn resolve_pixel(
    s: &mut Scratch,
    bg: [u8; 4],
    x: u32,
    y: u32,
    sample_coverage: impl Fn(usize, &Fragment) -> [u8; SAMPLES],
) -> (Resolve, [u8; 4]) {
    let frags = &s.fragments;
    if let [f] = frags.as_slice() {
        if f.is_opaque() && f.is_full() {
            return (Resolve::Fast, f.colour);
        }
        return (Resolve::Coverage, blend(f.colour, f.coverage8(), bg));
    }
    let points: [[f64; 2]; SAMPLES] = std::array::from_fn(|k| sample_point(x, y, k));
    s.depths.clear();
    s.depths.extend(frags.iter().map(|f| {
        if f.depth.b == 0.0 && f.depth.c == 0.0 {
            [f.depth.a + 0.0; SAMPLES]
        } else {
            points.map(|[px, py]| f.depth.at(px, py))
        }
    }));
    let depths = &s.depths;
    let cmp_at = |i: usize, j: usize, k: usize| {
        front_to_back(&frags[i], depths[i][k], &frags[j], depths[j][k])
    };
    s.order.clear();
    s.order.extend(0..frags.len());
    s.order.sort_by(|i, j| cmp_at(*i, *j, 0));

    // Drop what an opaque covering fragment hides at every sub-sample.
    if let Some(p) = s
        .order
        .iter()
        .position(|i| frags[*i].is_opaque() && frags[*i].is_full())
    {
        let o = s.order[p];
        let mut kept = p + 1;
        for q in p + 1..s.order.len() {
            let f = s.order[q];
            if (0..SAMPLES).any(|k| cmp_at(f, o, k) == Ordering::Less) {
                s.order[kept] = f;
                kept += 1;
            }
        }
        s.order.truncate(kept);
        if kept == 1 {
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
    // Whether every fragment of unit `u` is in front of every fragment of unit `v` at sub-sample k.
    let before = |u: (usize, usize), v: (usize, usize), k: usize| {
        order[u.0..u.1].iter().all(|a| {
            order[v.0..v.1]
                .iter()
                .all(|b| cmp_at(*a, *b, k) != Ordering::Greater)
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
    // A merged opaque unit covering the pixel hides what lies behind it at every sub-sample.
    let mut units: &[(usize, usize)] = &s.units;
    if let Some(p) = units.iter().position(|u| u.1 - u.0 > 1 && opaque_full(*u))
        && units[p + 1..]
            .iter()
            .all(|w| (0..SAMPLES).all(|k| before(units[p], *w, k)))
    {
        units = &units[..=p];
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
        && (0..SAMPLES).all(|k| units.windows(2).all(|w| before(w[0], w[1], k)));
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

    s.coverage.clear();
    s.coverage.resize(frags.len(), [0; SAMPLES]);
    for i in &s.order {
        s.coverage[*i] = sample_coverage(*i, &frags[*i]);
    }
    let mut sum = [0u32; 4];
    for k in 0..SAMPLES {
        s.covering.clear();
        s.covering
            .extend(s.order.iter().copied().filter(|i| s.coverage[*i][k] != 0));
        s.covering.sort_by(|i, j| cmp_at(*i, *j, k));
        let mut px = bg;
        for i in s.covering.iter().rev() {
            px = blend(frags[*i].colour, s.coverage[*i][k], px);
        }
        for c in 0..4 {
            sum[c] += px[c] as u32;
        }
    }
    let n = SAMPLES as u32;
    (Resolve::Sample, sum.map(|v| ((v + n / 2) / n) as u8))
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
