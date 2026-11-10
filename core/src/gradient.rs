//! Linear gradients as CE's frames paint them: a path filled or stroked through
//! `cairo.LinearGradient` on Cairo 1.18's image surface, whose colours pixman 0.42 computes.
//!
//! The chain, each step in its library's own arithmetic:
//!
//! - **Cairo's pattern.** The two ends in the frame's units, CE's stops at `np.linspace(0, 1, n)`,
//!   each colour clamped to `0..=1` and kept as 16-bit shorts, not premultiplied. A gradient whose
//!   stops all hold one colour, or whose ends meet, is painted as one solid colour
//!   (`_cairo_gradient_pattern_is_solid`). Its matrix from pixels to the pattern is the inverse
//!   of the context's ([`crate::project::FrameMap`]), inverted as `cairo_matrix_invert` inverts.
//! - **To pixman** (`_pixman_image_for_gradient`): the ends and stop offsets in 16.16 fixed point,
//!   rounded as Cairo rounds (`_cairo_fixed_16_16_from_double`); the matrix split into an integer
//!   offset and a fixed-point transform, whose translation is corrected to agree with Cairo's
//!   matrix at the middle of the drawn extents (`_cairo_matrix_to_pixman_matrix_offset`).
//! - **pixman's fetch** (`linear_get_scanline`): each pixel's centre through the transform in
//!   fixed point, its parameter along the gradient in double, truncated to 16.16, then stepped
//!   along a composite run from the run's first pixel (`t + trunc(inc * i)`), which is why a
//!   pixel's value depends on where its run starts ([`Linear::run_start`]).
//! - **The walker** (`pixman_gradient_walker_pixel_32`, pad extend): the colour between two
//!   stops in single precision, premultiplied, rounded to 8 bits.
//!
//! Against Cairo's own renders of random gradients (two and three stops, opaque and not, frames
//! of 192 to 1920 pixels) this agrees on all but about 1 in 60,000 interior pixels, those by one
//! level.

/// `CAIRO_MAGIC_NUMBER_FIXED_16_16`: 1.5 * 2^36, whose sum with a double holds it rounded to
/// 16.16 fixed point in its low 32 bits.
const MAGIC: f64 = 103079215104.0;

/// `_cairo_fixed_16_16_from_double`.
pub fn fixed_16_16(d: f64) -> i32 {
    (d + MAGIC).to_bits() as u32 as i32
}

/// `_cairo_fixed_from_double`: Cairo's 24.8 fixed point, rounded as its magic number rounds.
pub fn fixed_24_8(d: f64) -> i32 {
    const MAGIC_24_8: f64 = 26388279066624.0; // 1.5 * 2^44
    (d + MAGIC_24_8).to_bits() as u32 as i32
}

/// A Cairo matrix: `x' = xx * x + xy * y + x0`, `y' = yx * x + yy * y + y0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix {
    pub xx: f64,
    pub yx: f64,
    pub xy: f64,
    pub yy: f64,
    pub x0: f64,
    pub y0: f64,
}

impl Matrix {
    /// `cairo_matrix_multiply(result, a, b)`: `a` then `b`.
    fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
        Matrix {
            xx: a.xx * b.xx + a.yx * b.xy,
            yx: a.xx * b.yx + a.yx * b.yy,
            xy: a.xy * b.xx + a.yy * b.xy,
            yy: a.xy * b.yx + a.yy * b.yy,
            x0: a.x0 * b.xx + a.y0 * b.xy + b.x0,
            y0: a.x0 * b.yx + a.y0 * b.yy + b.y0,
        }
    }

    /// `cairo_matrix_translate`.
    fn translated(&self, tx: f64, ty: f64) -> Matrix {
        let t = Matrix {
            xx: 1.0,
            yx: 0.0,
            xy: 0.0,
            yy: 1.0,
            x0: tx,
            y0: ty,
        };
        Matrix::multiply(&t, self)
    }

    /// `cairo_matrix_invert`, `None` where it fails.
    pub fn inverted(&self) -> Option<Matrix> {
        let mut m = *self;
        if m.xy == 0.0 && m.yx == 0.0 {
            m.x0 = -m.x0;
            m.y0 = -m.y0;
            if m.xx != 1.0 {
                if m.xx == 0.0 {
                    return None;
                }
                m.xx = 1.0 / m.xx;
                m.x0 *= m.xx;
            }
            if m.yy != 1.0 {
                if m.yy == 0.0 {
                    return None;
                }
                m.yy = 1.0 / m.yy;
                m.y0 *= m.yy;
            }
            return Some(m);
        }
        let det = self.determinant();
        if !det.is_finite() || det == 0.0 {
            return None;
        }
        let (a, b, c, d, tx, ty) = (m.xx, m.yx, m.xy, m.yy, m.x0, m.y0);
        let s = 1.0 / det;
        Some(Matrix {
            xx: d * s,
            yx: -b * s,
            xy: -c * s,
            yy: a * s,
            x0: (c * ty - d * tx) * s,
            y0: (b * tx - a * ty) * s,
        })
    }

    fn determinant(&self) -> f64 {
        self.xx * self.yy - self.yx * self.xy
    }

    /// `cairo_matrix_transform_distance`.
    fn distance(&self, x: f64, y: f64) -> (f64, f64) {
        (self.xx * x + self.xy * y, self.yx * x + self.yy * y)
    }

    /// `cairo_matrix_transform_point`.
    fn point(&self, x: f64, y: f64) -> (f64, f64) {
        let (x, y) = self.distance(x, y);
        (x + self.x0, y + self.y0)
    }
}

/// A pixman transform: 16.16 fixed point.
type Transform = [[i64; 3]; 3];

/// `pixman_transform_point_31_16_3d`: the 48.16 vector `v` through `t`, each product's low half
/// rounded once.
fn point_3d(t: &Transform, v: [i64; 3]) -> [i64; 3] {
    std::array::from_fn(|i| {
        let hi: i64 = (0..3).map(|j| t[i][j] * (v[j] >> 16)).sum();
        let lo: i64 = (0..3).map(|j| t[i][j] * (v[j] & 0xFFFF)).sum();
        hi + ((lo + 0x8000) >> 16)
    })
}

/// `_cairo_matrix_to_pixman_matrix_offset` with no offset given: the transform and the integer
/// offset `(ix, iy)` added to a pixel before it.
fn to_pixman(matrix: &Matrix, xc: f64, yc: f64) -> (Transform, i64, i64) {
    let identity: Transform = [[1 << 16, 0, 0], [0, 1 << 16, 0], [0, 0, 1 << 16]];
    // `_cairo_matrix_is_pixman_translation`.
    if matrix.xx == 1.0 && matrix.yx == 0.0 && matrix.xy == 0.0 && matrix.yy == 1.0 {
        if matrix.x0 == 0.0 && matrix.y0 == 0.0 {
            return (identity, 0, 0);
        }
        let (tx, ty) = (matrix.x0, matrix.y0);
        if tx == tx.floor() && ty == ty.floor() && tx.abs() <= 32767.0 && ty.abs() <= 32767.0 {
            return (identity, tx as i64, ty as i64);
        }
    }
    let mut m = matrix.translated(0.0, 0.0);
    let (ix, iy);
    if m.x0 != 0.0 || m.y0 != 0.0 {
        // Split the translation between the transform and the offset, evenly in bits.
        let (mut tx, mut ty) = (m.x0, m.y0);
        let mut norm = tx.abs().max(ty.abs());
        for i in [-1.0, 1.0] {
            for j in [-1.0, 1.0] {
                let mut den = (m.xx + i) * (m.yy + j) - m.xy * m.yx;
                if den.abs() < f64::EPSILON {
                    continue;
                }
                let mut x = m.y0 * m.xy - m.x0 * (m.yy + j);
                let mut y = m.x0 * m.yx - m.y0 * (m.xx + i);
                den = 1.0 / den;
                x *= den;
                y *= den;
                let new_norm = x.abs().max(y.abs());
                if norm > new_norm {
                    norm = new_norm;
                    tx = x;
                    ty = y;
                }
            }
        }
        let (tx, ty) = (tx.floor(), ty.floor());
        ix = -tx as i64;
        iy = -ty as i64;
        m = m.translated(tx, ty);
    } else {
        ix = 0;
        iy = 0;
    }
    let f = |d: f64| fixed_16_16(d) as i64;
    let mut t: Transform = [
        [f(m.xx), f(m.xy), f(m.x0)],
        [f(m.yx), f(m.yy), f(m.y0)],
        [0, 0, 1 << 16],
    ];
    // `_cairo_matrix_has_unity_scale`.
    const EPS: f64 = 1.0 / 256.0;
    let det = m.determinant();
    if (det * det - 1.0).abs() < EPS
        && ((m.xy.abs() < EPS && m.yx.abs() < EPS) || (m.xx.abs() < EPS && m.yy.abs() < EPS))
    {
        return (t, ix, iy);
    }
    let Some(inv) = m.inverted() else {
        return (t, ix, iy);
    };
    // Fix the point (xc, yc): adjust pixman's translation to agree with Cairo's matrix there.
    for _ in 0..5 {
        let v = point_3d(&t, [f(xc), f(yc), 1 << 16]);
        let (x, y) = inv.point(v[0] as f64 / 65536.0, v[1] as f64 / 65536.0);
        let (x, y) = m.distance(x - xc, y - yc);
        let (dx, dy) = (f(x), f(y));
        t[0][2] -= dx;
        t[1][2] -= dy;
        if dx == 0 && dy == 0 {
            break;
        }
    }
    (t, ix, iy)
}

/// What a gradient paints: one colour, or a linear gradient.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// Premultiplied RGBA, as Cairo stores a solid source; `[0; 4]` for a clear gradient, which
    /// Cairo draws nothing from.
    Solid([u8; 4]),
    Linear(Linear),
}

/// A linear gradient as pixman evaluates it, its colours in RGBA byte order.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    transform: Transform,
    offset: [i64; 2],
    p1: [i64; 2],
    p2: [i64; 2],
    /// Positions in 16.16 and straight colours as 16-bit shorts.
    stops: Vec<(i64, [u16; 4])>,
    /// Every stop opaque: every pixel's alpha is 255.
    pub opaque: bool,
}

/// `_cairo_color_double_to_short`.
fn short(d: f64) -> u16 {
    (d * 65535.0 + 0.5) as u16
}

/// The one colour [`source`] makes of `rgbas` between `p1` and `p2` where it makes one,
/// premultiplied as Cairo stores a solid source; `None` where it makes a linear gradient. Cairo
/// paints nothing of a gradient whose one colour is clear ([`crate::composite::clear`]).
pub fn collapsed(p1: [f64; 2], p2: [f64; 2], rgbas: &[[f64; 4]]) -> Option<[u8; 4]> {
    assert!(rgbas.len() >= 2, "a gradient has two stops or more");
    let clamped: Vec<[f64; 4]> = rgbas.iter().map(|c| c.map(|v| v.clamp(0.0, 1.0))).collect();
    let shorts: Vec<[u16; 4]> = clamped.iter().map(|c| c.map(short)).collect();
    // Clear: Cairo draws nothing from it under OVER (`_cairo_pattern_is_clear`). A linear
    // gradient under pad, its ends meeting or not, is clear when every stop's alpha short is at
    // most 0xff (`_gradient_is_clear`, `CAIRO_ALPHA_SHORT_IS_CLEAR`), every stop clear as a solid
    // source. Drawn as a gradient, it would paint up to alpha 1, and lay a layer that takes part
    // in depth order everywhere.
    if clamped
        .iter()
        .all(|c| crate::composite::clear(crate::composite::premultiply(*c)))
    {
        return Some([0; 4]);
    }
    if (p1[0] - p2[0]).abs() < f64::EPSILON && (p1[1] - p2[1]).abs() < f64::EPSILON {
        // Degenerate under pad: the average of the first and last stops.
        let (a, z) = (clamped[0], clamped[clamped.len() - 1]);
        let avg: [f64; 4] = std::array::from_fn(|k| (1.0 * a[k] + 1.0 * z[k]) * 0.5);
        return Some(crate::composite::premultiply(avg));
    }
    if shorts.iter().all(|s| *s == shorts[0]) {
        return Some(crate::composite::premultiply(clamped[0]));
    }
    None
}

/// The source CE's `set_cairo_context_color` makes of `rgbas` (two or more rows, straight RGBA)
/// between `p1` and `p2` (the ends' frame coordinates as placed for display), under the
/// context's matrix `ctm`, for a drawing whose device extents are `extents` (`[x, y, width,
/// height]`, integer pixels, cut to the frame: the composite's bounded rectangle, whose middle
/// fixes pixman's translation).
pub fn source(
    ctm: &Matrix,
    p1: [f64; 2],
    p2: [f64; 2],
    rgbas: &[[f64; 4]],
    extents: [i32; 4],
) -> Source {
    if let Some(colour) = collapsed(p1, p2, rgbas) {
        return Source::Solid(colour);
    }
    let clamped: Vec<[f64; 4]> = rgbas.iter().map(|c| c.map(|v| v.clamp(0.0, 1.0))).collect();
    let shorts: Vec<[u16; 4]> = clamped.iter().map(|c| c.map(short)).collect();
    let n = rgbas.len();
    // `np.linspace(0, 1, n)`: `i * step`, the last exactly 1.
    let step = 1.0 / (n - 1) as f64;
    let stops = shorts
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let at = if i == n - 1 { 1.0 } else { i as f64 * step };
            (fixed_16_16(at) as i64, *s)
        })
        .collect();
    let pattern = ctm.inverted().expect("CE's context matrix is invertible");
    let [x, y, w, h] = extents;
    let (transform, ix, iy) = to_pixman(
        &pattern,
        x as f64 + w as f64 / 2.0,
        y as f64 + h as f64 / 2.0,
    );
    let f = |d: f64| fixed_16_16(d) as i64;
    Source::Linear(Linear {
        transform,
        offset: [ix, iy],
        p1: [f(p1[0]), f(p1[1])],
        p2: [f(p2[0]), f(p2[1])],
        stops,
        opaque: shorts.iter().all(|s| s[3] == u16::MAX),
    })
}

impl Linear {
    /// The parameter at pixel `(x, y)` (16.16, truncated) and its step per pixel along a row.
    fn parameter(&self, x: i64, y: i64) -> (i64, f64) {
        let t = &self.transform;
        let v = point_3d(
            t,
            [
                ((x + self.offset[0]) << 16) + 0x8000,
                ((y + self.offset[1]) << 16) + 0x8000,
                1 << 16,
            ],
        );
        // pixman's products are 64-bit, but Cairo fits a gradient's ends within half the 16.16
        // range before they reach it (`_cairo_gradient_pattern_fit_to_range`); ends anywhere in
        // the range need 128 bits here, where they are taken as they are.
        let wide = |v: i64| v as i128;
        let (dx, dy) = (
            wide(self.p2[0]) - wide(self.p1[0]),
            wide(self.p2[1]) - wide(self.p1[1]),
        );
        let l = dx * dx + dy * dy;
        let invden = 65536.0 * 65536.0 / (l as f64 * v[2] as f64);
        let v2 = v[2] as f64 * (1.0 / 65536.0);
        let along = (dx * wide(v[0]) + dy * wide(v[1])) as f64
            - (dx * wide(self.p1[0]) + dy * wide(self.p1[1])) as f64 * v2;
        let inc = (dx * wide(t[0][0]) + dy * wide(t[1][0])) as f64 * invden;
        ((along * invden) as i64, inc)
    }

    /// The premultiplied RGBA pixman paints at `x` on row `y`, in a composite run that starts at
    /// `start` (`start <= x`).
    pub fn at(&self, x: i64, y: i64, start: i64) -> [u8; 4] {
        let (t, inc) = self.parameter(start, y);
        let i = x - start;
        walker(&self.stops, t + (inc * i as f64) as i64)
    }
}

/// `pixman_gradient_walker_pixel_32` under pad extend: the premultiplied colour at `pos` (16.16).
fn walker(stops: &[(i64, [u16; 4])], pos: i64) -> [u8; 4] {
    const MIN: i64 = i32::MIN as i64;
    const MAX: i64 = i32::MAX as i64;
    let n = stops.iter().take_while(|s| pos >= s.0).count();
    let (left_x, left_c, right_x, right_c) = if n == 0 {
        (MIN, stops[0].1, stops[0].0, stops[0].1)
    } else if n == stops.len() {
        let last = stops[n - 1];
        (last.0, last.1, MAX, last.1)
    } else {
        (stops[n - 1].0, stops[n - 1].1, stops[n].0, stops[n].1)
    };
    let k = 1.0f32 / 257.0f32;
    // Colours in [r, g, b, a] order; pixman's walker reads alpha first.
    let l = left_c.map(|c| c as f32 * k);
    let r = right_c.map(|c| c as f32 * k);
    let lx = left_x as f32 * (1.0f32 / 65536.0f32);
    let rx = right_x as f32 * (1.0f32 / 65536.0f32);
    let inv255 = 1.0f32 / 255.0f32;
    let (s, b): ([f32; 4], [f32; 4]) = if (-f32::MIN_POSITIVE < rx - lx
        && rx - lx < f32::MIN_POSITIVE)
        || left_x == MIN
        || right_x == MAX
    {
        ([0.0; 4], std::array::from_fn(|k| (l[k] + r[k]) / 510.0f32))
    } else {
        let w = 1.0f32 / (rx - lx);
        (
            std::array::from_fn(|k| (r[k] - l[k]) * w * inv255),
            std::array::from_fn(|k| (l[k] * rx - r[k] * lx) * w * inv255),
        )
    };
    let y = pos as f32 * (1.0f32 / 65536.0f32);
    let a = 255.0f32 * (s[3] * y + b[3]);
    let channel = |k: usize| (a * (s[k] * y + b[k]) + 0.5f32) as u32 as u8;
    [
        channel(0),
        channel(1),
        channel(2),
        (a + 0.5f32) as u32 as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_rounds_to_nearest_even_as_cairo_s_magic_number_does() {
        assert_eq!(fixed_16_16(1.0), 65536);
        assert_eq!(fixed_16_16(-1.0), -65536);
        assert_eq!(fixed_16_16(0.5 / 65536.0), 0);
        assert_eq!(fixed_16_16(1.5 / 65536.0), 2);
        assert_eq!(fixed_16_16(0.3), 19661);
    }

    #[test]
    fn a_scaling_matrix_inverts_by_its_fast_path() {
        let m = Matrix {
            xx: 135.0,
            yx: 0.0,
            xy: 0.0,
            yy: -135.0,
            x0: 960.0,
            y0: 540.0,
        };
        let inv = m.inverted().unwrap();
        assert_eq!(inv.xx, 1.0 / 135.0);
        assert_eq!(inv.x0, -960.0 * (1.0 / 135.0));
        let (x, y) = inv.point(960.0, 540.0);
        assert!(x.abs() < 1e-12 && y.abs() < 1e-12);
    }

    #[test]
    fn equal_stops_and_meeting_ends_paint_one_colour() {
        let ctm = Matrix {
            xx: 13.5,
            yx: 0.0,
            xy: 0.0,
            yy: -13.5,
            x0: 96.0,
            y0: 54.0,
        };
        let red = [1.0, 0.0, 0.0, 1.0];
        let ext = [0, 0, 192, 108];
        assert_eq!(
            source(&ctm, [0.0, 0.0], [1.0, 0.0], &[red, red], ext),
            Source::Solid([255, 0, 0, 255])
        );
        // Clamped first: two colours past 1 are one colour.
        assert_eq!(
            source(
                &ctm,
                [0.0, 0.0],
                [1.0, 0.0],
                &[[1.5, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
                ext
            ),
            Source::Solid([255, 0, 0, 255])
        );
        let Source::Solid(avg) = source(
            &ctm,
            [1.0, 1.0],
            [1.0, 1.0],
            &[red, [0.0, 0.0, 1.0, 1.0]],
            ext,
        ) else {
            panic!("a gradient whose ends meet is solid")
        };
        assert_eq!(avg, [128, 0, 128, 255]);
    }

    /// A gradient whose ends lie far outside the frame, up to the ends of the 16.16 range, still
    /// paints its middle colour at the middle: the products along it do not overflow.
    #[test]
    fn a_gradient_with_ends_across_the_whole_fixed_point_range_paints_its_middle() {
        let ctm = Matrix {
            xx: 13.5,
            yx: 0.0,
            xy: 0.0,
            yy: -13.5,
            x0: 96.0,
            y0: 54.0,
        };
        let (red, blue) = ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]);
        let ext = [0, 0, 192, 108];
        let edge = 32767.0 + 65535.0 / 65536.0;
        for (p1, p2) in [
            ([-25000.0, 0.0], [25000.0, 0.0]),
            ([0.0, -25000.0], [0.0, 25000.0]),
            ([-25000.0, -25000.0], [25000.0, 25000.0]),
            ([-edge, 0.0], [edge, 0.0]),
            ([-32768.0, -32768.0], [edge, edge]),
        ] {
            let Source::Linear(g) = source(&ctm, p1, p2, &[red, blue], ext) else {
                panic!("two colours between two ends are a gradient")
            };
            let [r, gr, b, a] = g.at(96, 54, 0);
            assert!(
                r.abs_diff(128) <= 1 && gr == 0 && b.abs_diff(128) <= 1 && a == 255,
                "{p1:?}..{p2:?}: {:?}",
                [r, gr, b, a]
            );
        }
    }

    #[test]
    fn the_walker_pads_past_the_ends_and_premultiplies() {
        let stops = vec![(0, [65535, 0, 0, 65535]), (65536, [0, 0, 65535, 32768])];
        assert_eq!(walker(&stops, -100), [255, 0, 0, 255]);
        let end = walker(&stops, 1 << 20);
        assert_eq!(end[3], 128);
        assert_eq!(end[2], 128);
    }
}
