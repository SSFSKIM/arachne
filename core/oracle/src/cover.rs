//! Coverage at every sample (contract §1's sample pattern, §3 and §4).
//!
//! - A mesh triangle is judged against its nominal, unrounded corners with the margin
//!   `m = ε + 1/256` px: `Yes` farther than `m` inside every edge, absent farther than `m` outside,
//!   `Either` between.
//! - A path layer's cells take their exact nonzero-winding area of the layer's rings by the
//!   oracle's own trapezoid integration, admitted within `μ = α + 2^-24 + μ_ε`; a cell is `Yes`
//!   where its whole interval rounds to a nonzero byte, absent where none of it does.
//! - A point block's pixels are `Yes`, or `Either` where the point's truncation may move.

use crate::out::Cover;

/// The 16 sample offsets in sixteenths of a pixel from its centre (contract §1).
pub const SAMPLE_OFFSETS: [[i8; 2]; 16] = [
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

/// Sample `s` of pixel `(x, y)`: `(x + 1/2 + (dx + 1/2) / 16, y + 1/2 + (dy + 1/2) / 16)`.
pub fn sample_point(x: i64, y: i64, s: usize) -> [f64; 2] {
    let [dx, dy] = SAMPLE_OFFSETS[s];
    [
        x as f64 + 0.5 + (dx as f64 + 0.5) / 16.0,
        y as f64 + 0.5 + (dy as f64 + 0.5) / 16.0,
    ]
}

/// The cell `(column, row)` of the pixel's 4 x 4 grid that sample `s` lies in.
pub fn sample_cell(s: usize) -> (usize, usize) {
    let [dx, dy] = SAMPLE_OFFSETS[s];
    (((dx + 8) / 4) as usize, ((dy + 8) / 4) as usize)
}

/// A coverage byte: `floor(clamp(v, 0, 1) · 255 + 1/2)`.
pub fn byte(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
}

/// A mask's coverage byte: `(bits · 255 + 8) div 16`.
pub fn mask_byte(bits: u32) -> u8 {
    ((bits * 255 + 8) / 16) as u8
}

/// One fragment's verdicts at one pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelCover {
    pub x: u32,
    pub y: u32,
    pub cover: [u8; 16],
    pub cell_lo: [u8; 16],
    pub cell_hi: [u8; 16],
    pub pixel_lo: u8,
    pub pixel_hi: u8,
    pub pixel_area: f64,
    pub cell_area: [f64; 16],
}

impl PixelCover {
    /// A mask's row from its sample verdicts.
    fn mask(x: u32, y: u32, cover: [u8; 16]) -> PixelCover {
        let yes = cover.iter().filter(|&&c| c == Cover::Yes as u8).count() as u32;
        let any = cover.iter().filter(|&&c| c != Cover::Absent as u8).count() as u32;
        PixelCover {
            x,
            y,
            cover,
            cell_lo: cover.map(|c| if c == Cover::Yes as u8 { 255 } else { 0 }),
            cell_hi: cover.map(|c| if c == Cover::Absent as u8 { 0 } else { 255 }),
            pixel_lo: mask_byte(yes),
            pixel_hi: mask_byte(any),
            pixel_area: f64::NAN,
            cell_area: [f64::NAN; 16],
        }
    }
}

/// The pixel window `[x0, y0, x1, y1)`.
pub type Window = [u32; 4];

/// The window's pixel rows and columns a box `[x0, y0, x1, y1]` (pixels, any reals) reaches.
fn pixels_of(b: [f64; 4], w: Window) -> Option<(std::ops::Range<i64>, std::ops::Range<i64>)> {
    if b.iter().any(|v| v.is_nan()) {
        return None;
    }
    let clamp = |v: f64, lo: u32, hi: u32| v.clamp(lo as f64, hi as f64) as i64;
    let xs = clamp(b[0].floor(), w[0], w[2])..clamp(b[2].floor() + 1.0, w[0], w[2]);
    let ys = clamp(b[1].floor(), w[1], w[3])..clamp(b[3].floor() + 1.0, w[1], w[3]);
    (!xs.is_empty() && !ys.is_empty()).then_some((xs, ys))
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// The unit roundoff of binary64.
const U: f64 = f64::EPSILON / 2.0;

/// The guard: a triangle reaching past `±GUARD` px is clipped there (contract §3).
const GUARD: f64 = 65536.0;

/// An edge's nominal line, read at a sample with a bound on the reading's own rounding, so that a
/// verdict never rests on an error the arithmetic made. Its direction comes from the corners
/// scaled by a quarter (exact), so corners anywhere in binary64's range neither overflow nor lose
/// the edge.
struct Line {
    a: [f64; 2],
    b: [f64; 2],
    /// The unit direction `a → b`; `None` where the corners coincide.
    u: Option<[f64; 2]>,
    /// The edge's length.
    len: f64,
}

/// A reading and the most its rounding may have moved it.
#[derive(Clone, Copy)]
struct Bounded {
    v: f64,
    err: f64,
}

impl Line {
    fn new(a: [f64; 2], b: [f64; 2]) -> Line {
        let q = [0.25 * b[0] - 0.25 * a[0], 0.25 * b[1] - 0.25 * a[1]];
        let l = libm::hypot(q[0], q[1]);
        let u = (l > 0.0 && l.is_finite()).then(|| [q[0] / l, q[1] / l]);
        Line {
            a,
            b,
            u,
            len: 4.0 * l,
        }
    }

    /// The signed distance of `p` from the line (positive on one side, the same for every edge of
    /// one winding) and its position along the edge from `a`. The unit direction is within `4u`
    /// per coordinate of the exact one and `p − a` within `u` relative, so each reading is within
    /// `10u (|p − a|₀ + |p − a|₁)` of its exact value.
    fn read(&self, p: [f64; 2]) -> Option<(Bounded, Bounded)> {
        let u = self.u?;
        let pa = sub(p, self.a);
        let err = 10.0 * U * (pa[0].abs() + pa[1].abs());
        let side = Bounded {
            v: u[0] * pa[1] - u[1] * pa[0],
            err,
        };
        let along = Bounded {
            v: u[0] * pa[0] + u[1] * pa[1],
            err,
        };
        Some((side, along))
    }

    /// A lower bound on the distance from `p` to the segment.
    fn distance_below(&self, p: [f64; 2]) -> f64 {
        let to = |q: [f64; 2]| libm::hypot(p[0] - q[0], p[1] - q[1]) * (1.0 - 4.0 * U);
        let ends = to(self.a).min(to(self.b));
        match self.read(p) {
            Some((side, along))
                if along.v + along.err >= 0.0 && along.v - along.err <= self.len =>
            {
                ends.min(side.v.abs() - side.err)
            }
            _ => ends,
        }
    }
}

/// The winding of the corners `a, b, c` where their reading settles it: the sign of
/// `(b − a) × (c − a)`, read on the corners scaled by the power of two that brings the largest
/// coordinate into `[1/2, 1)`, so that no difference or product can overflow whatever the corners
/// are, with a bound on the reading's rounding (and on the scaling's, where it makes a coordinate
/// subnormal). `None` where the bound does not settle the sign: the corners are collinear, or too
/// nearly so for binary64 to tell.
fn winding(corners: [[f64; 2]; 3]) -> Option<f64> {
    let big = corners.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    if big == 0.0 || !big.is_finite() {
        return None;
    }
    let s = libm::ldexp(1.0, -libm::frexp(big).1);
    let q = corners.map(|p| p.map(|v| v * s));
    let (d1, d2) = (sub(q[1], q[0]), sub(q[2], q[0]));
    let (t1, t2) = (d1[0] * d2[1], d1[1] * d2[0]);
    let det = t1 - t2;
    let err = 4.0 * U * (t1.abs() + t2.abs()) + 16.0 * f64::from_bits(1);
    (det.abs() > err).then_some(det.signum())
}

/// An upper bound on how far a point inside the triangle `corners` can lie from its edges: its
/// inradius, `2A / P <= |(b − a) × (c − a)| / (2 L)` with `L` its longest edge (its perimeter is at
/// least `2 L`). Read on the corners scaled as `winding` scales them, so nothing overflows; the
/// scaled bound is scaled back up, to infinity where binary64 cannot hold it, which bounds
/// nothing. A triangle with no area has none, and this is what bounds its interior by its edges
/// whatever its winding reads.
fn inradius_above(corners: [[f64; 2]; 3]) -> f64 {
    let big = corners.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    if big == 0.0 {
        return 0.0;
    }
    if !big.is_finite() {
        return f64::INFINITY;
    }
    let e = libm::frexp(big).1;
    let s = libm::ldexp(1.0, -e);
    let q = corners.map(|p| p.map(|v| v * s));
    let (d1, d2) = (sub(q[1], q[0]), sub(q[2], q[0]));
    let (t1, t2) = (d1[0] * d2[1], d1[1] * d2[0]);
    let det = (t1 - t2).abs() + 4.0 * U * (t1.abs() + t2.abs()) + 16.0 * f64::from_bits(1);
    // Each scaled coordinate is within one subnormal unit of the exact one, so an edge within
    // four of its reading.
    let longest = (0..3)
        .map(|i| {
            let d = sub(q[(i + 1) % 3], q[i]);
            libm::hypot(d[0], d[1]) * (1.0 - 4.0 * U)
        })
        .fold(0.0f64, f64::max)
        - 4.0 * f64::from_bits(1);
    if longest <= 0.0 {
        // The corners lie within a few subnormal units of one another, scaled: the triangle
        // within that of each corner.
        return libm::ldexp(8.0 * f64::from_bits(1), e);
    }
    libm::ldexp(det / (2.0 * longest) * (1.0 + 4.0 * U), e)
}

/// A mesh triangle's verdicts at every sample of the window (contract §3, oracle reading): its
/// nominal corners in pixels and each corner's screen shift `ε` in pixels, all finite (a triangle
/// with an infinite one is unbounded, which the caller writes).
///
/// A distance counts beyond the margin only when it is beyond it by more than its reading's error
/// bound, so corners anywhere in binary64's range keep their verdicts. The guard clip acts as four
/// more edges: a sample farther than `1/256` px outside `±65536` is absent, one within it `Either`.
/// The clipped corners lie on the nominal edges, and their rounding to the 1/256 px grid is within
/// the margin as the triangle's own corners' is.
pub fn triangle(corners: [[f64; 2]; 3], eps: [f64; 3], window: Window) -> Vec<PixelCover> {
    let margins: [f64; 3] = std::array::from_fn(|e| eps[e].max(eps[(e + 1) % 3]) + 1.0 / 256.0);
    let reach = margins.iter().copied().fold(0.0, f64::max);
    let lo = |k: usize| corners.iter().map(|c| c[k]).fold(f64::INFINITY, f64::min);
    let hi = |k: usize| {
        corners
            .iter()
            .map(|c| c[k])
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let clip = |v: f64| v.clamp(-GUARD - reach, GUARD + reach);
    let Some((xs, ys)) = pixels_of(
        [
            clip(lo(0) - reach),
            clip(lo(1) - reach),
            clip(hi(0) + reach),
            clip(hi(1) + reach),
        ],
        window,
    ) else {
        return Vec::new();
    };
    let lines: [Line; 3] = std::array::from_fn(|e| Line::new(corners[e], corners[(e + 1) % 3]));
    let winding = winding(corners);
    let inside = inradius_above(corners);
    let guard = |p: [f64; 2]| -> Cover {
        let out = (p[0].abs() - GUARD).max(p[1].abs() - GUARD);
        if out > 1.0 / 256.0 {
            Cover::Absent
        } else if out >= -1.0 / 256.0 {
            Cover::Either
        } else {
            Cover::Yes
        }
    };
    let verdict = |p: [f64; 2]| -> Cover {
        let near = lines
            .iter()
            .map(|l| l.distance_below(p))
            .fold(f64::INFINITY, f64::min);
        // Farther from every edge than any point inside can lie, and than the margin: outside by
        // more than the margin, whatever the winding. Where the triangle has no area (its corners
        // coincide or lie on one line) this is what says absent off its segments.
        if near > reach && near > inside {
            return Cover::Absent;
        }
        // The verdict were the triangle wound by `sign`: each edge's distance, positive inside (an
        // edge whose corners coincide constrains nothing).
        let wound = |sign: f64| -> Cover {
            let d: [Option<Bounded>; 3] = std::array::from_fn(|e| {
                lines[e].read(p).map(|(side, _)| Bounded {
                    v: sign * side.v,
                    err: side.err,
                })
            });
            let beyond =
                |e: usize, f: &dyn Fn(Bounded, f64) -> bool| d[e].is_some_and(|d| f(d, margins[e]));
            if (0..3).all(|e| beyond(e, &|d, m| d.v - d.err > m)) {
                Cover::Yes
            } else if (0..3).any(|e| beyond(e, &|d, m| d.v + d.err < -m))
                || ((0..3).any(|e| beyond(e, &|d, _| d.v + d.err < 0.0)) && near > reach)
            {
                Cover::Absent
            } else {
                Cover::Either
            }
        };
        let own = match winding {
            Some(sign) => wound(sign),
            // A winding the corners do not settle: collinear, or numerically unresolved. A sample
            // is absent only where it is absent under both windings, and never sure.
            None => {
                if wound(1.0) == Cover::Absent && wound(-1.0) == Cover::Absent {
                    Cover::Absent
                } else {
                    Cover::Either
                }
            }
        };
        match (own, guard(p)) {
            (Cover::Absent, _) | (_, Cover::Absent) => Cover::Absent,
            (Cover::Yes, Cover::Yes) => Cover::Yes,
            _ => Cover::Either,
        }
    };
    let mut out = Vec::new();
    for y in ys {
        for x in xs.clone() {
            let cover: [u8; 16] = std::array::from_fn(|s| verdict(sample_point(x, y, s)) as u8);
            if cover.iter().all(|&c| c == Cover::Absent as u8) {
                continue;
            }
            out.push(PixelCover::mask(x as u32, y as u32, cover));
        }
    }
    out
}

/// A point block's pixels (contract §3, `point_blocks@1`): for a point at pixel position `at`
/// with screen shift `eps`, block side `t`, each candidate truncation `(X, Y)` and the pixels its
/// block covers with their offsets `(o, o')` from it. A pixel is `Yes` where the truncation cannot
/// move, else `Either` under each candidate.
pub struct BlockPixel {
    pub cover: PixelCover,
    pub offset: (i64, i64),
}

/// The pixels a block covers in the frame `size`, per candidate truncation, for a finite `eps`
/// (a point with an infinite one is unbounded, which the caller writes).
pub fn block(at: [f64; 2], eps: f64, side: i64, size: [u32; 2], window: Window) -> Vec<BlockPixel> {
    let trunc = |v: f64| v.trunc() as i64;
    let lo = (-side as f64 / 2.0).floor() as i64 + 1;
    let (w0, h0) = (window[0] as i64, window[1] as i64);
    let (w1, h1) = (
        (window[2].min(size[0])) as i64,
        (window[3].min(size[1])) as i64,
    );
    let (xs, ys) = (
        trunc(at[0] - eps)..=trunc(at[0] + eps),
        trunc(at[1] - eps)..=trunc(at[1] + eps),
    );
    let moves = xs.start() != xs.end() || ys.start() != ys.end();
    let code = if moves { Cover::Either } else { Cover::Yes };
    // Only the candidates whose block reaches the window: a wide `eps` lists no more than it shows.
    let reaching = |r: std::ops::RangeInclusive<i64>, w0: i64, w1: i64| {
        *r.start()
            .max(&(w0.saturating_sub(lo).saturating_sub(side) + 1))
            ..=*r.end().min(&(w1.saturating_sub(lo).saturating_sub(1)))
    };
    let (xs, ys) = (reaching(xs, w0, w1), reaching(ys, h0, h1));
    let mut out = Vec::new();
    for bx in xs {
        for by in ys.clone() {
            let px = (bx + lo).max(w0)..(bx.saturating_add(lo).saturating_add(side)).min(w1);
            let py = (by + lo).max(h0)..(by.saturating_add(lo).saturating_add(side)).min(h1);
            for x in px {
                for y in py.clone() {
                    out.push(BlockPixel {
                        cover: PixelCover::mask(x as u32, y as u32, [code as u8; 16]),
                        offset: (x - bx, y - by),
                    });
                }
            }
        }
    }
    out
}

/// A ring edge in pixels.
#[derive(Clone, Copy, Debug)]
struct Edge {
    a: [f64; 2],
    b: [f64; 2],
}

impl Edge {
    fn ymin(&self) -> f64 {
        self.a[1].min(self.b[1])
    }
    fn ymax(&self) -> f64 {
        self.a[1].max(self.b[1])
    }
    /// Its `x` at `y`, for a non-horizontal edge.
    fn x_at(&self, y: f64) -> f64 {
        let t = (y - self.a[1]) / (self.b[1] - self.a[1]);
        self.a[0] + (self.b[0] - self.a[0]) * t
    }
}

/// `∫ max(0, X − x(y)) dy` over a slab of height `h` where `x` runs linearly from `xa` to `xb`.
fn left_of(xa: f64, xb: f64, h: f64, x: f64) -> f64 {
    let (lo, hi) = (xa.min(xb), xa.max(xb));
    if x <= lo {
        0.0
    } else if x >= hi {
        h * (x - (xa + xb) / 2.0)
    } else {
        0.5 * h * (x - lo) * (x - lo) / (hi - lo)
    }
}

/// The part of the segment `a → b` inside the box `[x0, y0, x1, y1]`: its length, or `None` where
/// it misses (Liang–Barsky).
fn clip_length(a: [f64; 2], b: [f64; 2], bx: [f64; 4]) -> Option<f64> {
    let d = sub(b, a);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [
        (-d[0], a[0] - bx[0]),
        (d[0], bx[2] - a[0]),
        (-d[1], a[1] - bx[1]),
        (d[1], bx[3] - a[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 <= t1).then(|| (t1 - t0) * libm::hypot(d[0], d[1]))
}

fn inside(p: [f64; 2], bx: [f64; 4]) -> bool {
    p[0] >= bx[0] && p[0] <= bx[2] && p[1] >= bx[1] && p[1] <= bx[3]
}

/// The admitted interval's half-width for areas measured over a square of side `side` px whose
/// row band meets `n_e` ring edges, the rings `L` px long and in `m` pieces within it widened by
/// `eps` (contract §4: `μ = α + 2^-24 + μ_ε`, `μ_ε` divided by the square's area).
fn mu(n_e: u32, eps: f64, l: f64, m: u32, area: f64) -> f64 {
    let alpha = (1.0 + n_e as f64) / 1_073_741_824.0;
    let storage = 1.0 / 16_777_216.0;
    let moved = (2.0 * eps * l + std::f64::consts::PI * eps * eps * m as f64) / area;
    alpha + storage + moved
}

/// A sum within `1e-9` of 0 is 0, one within `1e-9` of 1 is 1.
fn snapped(a: f64) -> f64 {
    if a.abs() <= 1e-9 {
        0.0
    } else if (a - 1.0).abs() <= 1e-9 {
        1.0
    } else {
        a
    }
}

/// Lengths and pieces of the rings within boxes widened by `eps`, accumulated per box.
struct Reach {
    l: Vec<f64>,
    m: Vec<u32>,
}

/// The boxes, each with its index, that an edge may reach.
type Boxes<'a> = dyn Fn(&Edge) -> Vec<(usize, [f64; 4])> + 'a;

/// For each ring, the boxes `boxes(edge)` names around each edge: the length of the edge inside
/// each, and a piece counted where the edge enters it (its start outside) or, for a ring wholly
/// inside, once.
fn reach(rings: &[Vec<[f64; 2]>], n: usize, boxes: &Boxes) -> Reach {
    let mut r = Reach {
        l: vec![0.0; n],
        m: vec![0; n],
    };
    for ring in rings {
        let mut touched: std::collections::HashMap<usize, bool> = Default::default();
        for i in 0..ring.len() {
            let e = Edge {
                a: ring[i],
                b: ring[(i + 1) % ring.len()],
            };
            for (k, bx) in boxes(&e) {
                if let Some(len) = clip_length(e.a, e.b, bx) {
                    r.l[k] += len;
                    let entered = !inside(e.a, bx);
                    if entered {
                        r.m[k] += 1;
                    }
                    *touched.entry(k).or_default() |= entered;
                }
            }
        }
        for (k, entered) in touched {
            if !entered {
                r.m[k] += 1;
            }
        }
    }
    r
}

/// A path layer's verdicts over the window (contract §4, oracle reading): its rings in pixels and
/// its screen shift `eps`, all finite (a layer with an infinite one is unbounded, which the caller
/// writes).
pub fn path_layer(rings: &[Vec<[f64; 2]>], eps: f64, window: Window) -> Vec<PixelCover> {
    let all: Vec<Edge> = rings
        .iter()
        .flat_map(|r| {
            (0..r.len()).map(move |i| Edge {
                a: r[i],
                b: r[(i + 1) % r.len()],
            })
        })
        .collect();
    if all.is_empty() {
        return Vec::new();
    }
    let lo = |k: usize| {
        all.iter()
            .map(|e| e.a[k].min(e.b[k]))
            .fold(f64::INFINITY, f64::min)
    };
    let hi = |k: usize| {
        all.iter()
            .map(|e| e.a[k].max(e.b[k]))
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let pad = eps + 1.0;
    let Some((xs, ys)) = pixels_of([lo(0) - pad, lo(1) - pad, hi(0) + pad, hi(1) + pad], window)
    else {
        return Vec::new();
    };
    let areas: Vec<Edge> = all.iter().copied().filter(|e| e.a[1] != e.b[1]).collect();
    let (x0, nx) = (xs.start, (xs.end - xs.start) as usize);
    let mut out = Vec::new();
    for py in ys {
        let (top, bottom) = (py as f64, py as f64 + 1.0);
        let row_edges: Vec<Edge> = areas
            .iter()
            .copied()
            .filter(|e| e.ymax() > top && e.ymin() < bottom)
            .collect();
        // Cell areas: [cell row][cell column across the window's columns].
        let mut cells = vec![vec![0.0f64; 4 * nx]; 4];
        for (j, row) in cells.iter_mut().enumerate() {
            let (ya, yb) = (4.0 * top + j as f64, 4.0 * top + j as f64 + 1.0);
            cell_row(&row_edges, ya, yb, 4 * x0, row);
        }
        // Edge counts per cell row band and for the pixel's band.
        let n_e: [u32; 5] = std::array::from_fn(|j| {
            let (a, b) = if j < 4 {
                (top + j as f64 / 4.0, top + (j + 1) as f64 / 4.0)
            } else {
                (top, bottom)
            };
            all.iter()
                .filter(|e| e.ymax() >= a && e.ymin() <= b)
                .count() as u32
        });
        let (cell_reach, pixel_reach) = if eps > 0.0 {
            let cell_boxes = |e: &Edge| -> Vec<(usize, [f64; 4])> {
                let mut v = Vec::new();
                for j in 0..4 {
                    let by = [top + j as f64 / 4.0 - eps, top + (j + 1) as f64 / 4.0 + eps];
                    if e.ymax() < by[0] || e.ymin() > by[1] {
                        continue;
                    }
                    let (xa, xb) = (e.a[0].min(e.b[0]) - eps, e.a[0].max(e.b[0]) + eps);
                    let c0 = ((4.0 * xa).floor() as i64).max(4 * x0);
                    let c1 = ((4.0 * xb).floor() as i64 + 1).min(4 * (x0 + nx as i64));
                    for c in c0..c1 {
                        let bx = [
                            c as f64 / 4.0 - eps,
                            by[0],
                            (c + 1) as f64 / 4.0 + eps,
                            by[1],
                        ];
                        v.push((j * 4 * nx + (c - 4 * x0) as usize, bx));
                    }
                }
                v
            };
            let pixel_boxes = |e: &Edge| -> Vec<(usize, [f64; 4])> {
                let by = [top - eps, bottom + eps];
                if e.ymax() < by[0] || e.ymin() > by[1] {
                    return Vec::new();
                }
                let (xa, xb) = (e.a[0].min(e.b[0]) - eps, e.a[0].max(e.b[0]) + eps);
                let c0 = (xa.floor() as i64).max(x0);
                let c1 = (xb.floor() as i64 + 1).min(x0 + nx as i64);
                (c0..c1)
                    .map(|c| {
                        (
                            (c - x0) as usize,
                            [c as f64 - eps, by[0], c as f64 + 1.0 + eps, by[1]],
                        )
                    })
                    .collect()
            };
            (
                Some(reach(rings, 16 * nx, &cell_boxes)),
                Some(reach(rings, nx, &pixel_boxes)),
            )
        } else {
            (None, None)
        };
        for px in 0..nx {
            let mut sum = 0.0;
            let cell_area: [f64; 16] = std::array::from_fn(|s| {
                let (cc, cr) = sample_cell(s);
                snapped(cells[cr][4 * px + cc])
            });
            for row in &cells {
                for c in 0..4 {
                    sum += row[4 * px + c];
                }
            }
            let pixel_area = snapped(sum / 16.0);
            let mut cover = [0u8; 16];
            let (mut cell_lo, mut cell_hi) = ([0u8; 16], [0u8; 16]);
            for s in 0..16 {
                let (cc, cr) = sample_cell(s);
                let k = cr * 4 * nx + 4 * px + cc;
                let m = match &cell_reach {
                    Some(r) => mu(n_e[cr], eps, r.l[k], r.m[k], 1.0 / 16.0),
                    None => mu(n_e[cr], 0.0, 0.0, 0, 1.0),
                };
                let a = cell_area[s];
                (cell_lo[s], cell_hi[s]) = (byte(a - m), byte(a + m));
                cover[s] = if cell_lo[s] >= 1 {
                    Cover::Yes
                } else if cell_hi[s] == 0 {
                    Cover::Absent
                } else {
                    Cover::Either
                } as u8;
            }
            let mp = match &pixel_reach {
                Some(r) => mu(n_e[4], eps, r.l[px], r.m[px], 1.0),
                None => mu(n_e[4], 0.0, 0.0, 0, 1.0),
            };
            let (pixel_lo, pixel_hi) = (byte(pixel_area - mp), byte(pixel_area + mp));
            if pixel_hi == 0 && cover.iter().all(|&c| c == Cover::Absent as u8) {
                continue;
            }
            out.push(PixelCover {
                x: (x0 + px as i64) as u32,
                y: py as u32,
                cover,
                cell_lo,
                cell_hi,
                pixel_lo,
                pixel_hi,
                pixel_area,
                cell_area,
            });
        }
    }
    out
}

/// One cell row's exact nonzero-winding areas, in cell units (the pixel plane magnified 4 times):
/// the row `[ya, yb)`, its cells from column `c0`, the row's non-horizontal edges in pixels.
fn cell_row(edges: &[Edge], ya: f64, yb: f64, c0: i64, row: &mut [f64]) {
    // The edges in cell units, clipped to the row: (x at the row's span ends, span, winding).
    struct Span {
        y0: f64,
        y1: f64,
        x0: f64,
        x1: f64,
        dir: i32,
    }
    let mut spans = Vec::new();
    for e in edges {
        let (a, b) = ([4.0 * e.a[0], 4.0 * e.a[1]], [4.0 * e.b[0], 4.0 * e.b[1]]);
        let (lo, hi) = (a[1].min(b[1]), a[1].max(b[1]));
        let (y0, y1) = (lo.max(ya), hi.min(yb));
        if y1 <= y0 {
            continue;
        }
        let e4 = Edge { a, b };
        let at = |y: f64| {
            if y == a[1] {
                a[0]
            } else if y == b[1] {
                b[0]
            } else {
                e4.x_at(y)
            }
        };
        spans.push(Span {
            y0,
            y1,
            x0: at(y0),
            x1: at(y1),
            dir: if b[1] > a[1] { 1 } else { -1 },
        });
    }
    if spans.is_empty() {
        return;
    }
    let x_of = |s: &Span, y: f64| {
        if y == s.y0 {
            s.x0
        } else if y == s.y1 {
            s.x1
        } else {
            s.x0 + (s.x1 - s.x0) * ((y - s.y0) / (s.y1 - s.y0))
        }
    };
    let mut cuts: Vec<f64> = vec![ya, yb];
    for s in &spans {
        cuts.push(s.y0);
        cuts.push(s.y1);
    }
    for i in 0..spans.len() {
        for k in i + 1..spans.len() {
            let (s, t) = (&spans[i], &spans[k]);
            let (y0, y1) = (s.y0.max(t.y0), s.y1.min(t.y1));
            if y1 <= y0 {
                continue;
            }
            let d0 = x_of(s, y0) - x_of(t, y0);
            let d1 = x_of(s, y1) - x_of(t, y1);
            if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
                cuts.push(y0 + (y1 - y0) * (d0 / (d0 - d1)));
            }
        }
    }
    cuts.retain(|y| *y >= ya && *y <= yb);
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let width = row.len() as i64;
    for w in cuts.windows(2) {
        let (s0, s1) = (w[0], w[1]);
        let h = s1 - s0;
        if h <= 0.0 {
            continue;
        }
        let mid = (s0 + s1) / 2.0;
        let mut active: Vec<(f64, f64, f64, i32)> = spans
            .iter()
            .filter(|s| s.y0 <= s0 && s.y1 >= s1)
            .map(|s| (x_of(s, mid), x_of(s, s0), x_of(s, s1), s.dir))
            .collect();
        active.sort_by(|p, q| p.0.total_cmp(&q.0));
        let mut winding = 0;
        for i in 0..active.len() {
            winding += active[i].3;
            if winding == 0 || i + 1 == active.len() {
                continue;
            }
            let (l, r) = (active[i], active[i + 1]);
            let (la, lb, ra, rb) = (l.1, l.2, r.1, r.2);
            let lmin = la.min(lb);
            let lmax = la.max(lb);
            let rmin = ra.min(rb);
            let rmax = ra.max(rb);
            let first = (lmin.floor() as i64).max(c0);
            let last = (rmax.ceil() as i64).min(c0 + width);
            let within = |x: f64| left_of(la, lb, h, x) - left_of(ra, rb, h, x);
            for c in first..last {
                let (cx0, cx1) = (c as f64, c as f64 + 1.0);
                let a = if cx0 >= lmax && cx1 <= rmin {
                    h
                } else {
                    within(cx1) - within(cx0)
                };
                row[(c - c0) as usize] += a;
            }
        }
    }
}
