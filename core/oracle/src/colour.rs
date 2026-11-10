//! Each fragment's premultiplied bytes, from XF (contract §9, "Colour, from XF"): a solid through
//! `cairo_premultiply@1`, a point block's bytes as CE writes them, and a lit mesh triangle's
//! corners (`ce_shading@1`) interpolated on the pixel plane. A layer of two or more rows, and a
//! shaded path, paint a gradient; the oracle does not yet compute gradients
//! (`cairo_linear_gradient@1`), and says so.

use arachne_core::xf::{Affine, LightModel};

use crate::camera::{FrameMap, Projection};
use crate::depth::three_corner_plane;

/// A fragment's colour as the oracle states it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Colour {
    /// One premultiplied colour at every pixel.
    Solid([u8; 4]),
    /// A colour per pixel (a lit mesh triangle), given per cover row.
    PerPixel,
    /// A gradient (`cairo_linear_gradient@1`), not computed by this oracle.
    Gradient,
}

fn clamp01(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

/// `trunc(v · 65535 + 0.5) >> 8`, for `v` in `[0, 1]`.
fn short_byte(v: f64) -> u8 {
    (((v * 65535.0 + 0.5) as u32) >> 8) as u8
}

/// `cairo_premultiply@1`: straight `c` under group opacity `opacity`.
pub fn premultiply(c: [f64; 4], opacity: f64) -> [u8; 4] {
    let a = clamp01(c[3] * opacity);
    [
        short_byte(clamp01(c[0]) * a),
        short_byte(clamp01(c[1]) * a),
        short_byte(clamp01(c[2]) * a),
        short_byte(a),
    ]
}

/// A point block's bytes (contract §9, "Point bytes"); `None` where its alpha byte is 0.
pub fn point_bytes(c: [f64; 4], opacity: f64) -> Option<[u8; 4]> {
    let byte = |v: f64| (255.0 * v).clamp(0.0, 255.0) as u32;
    let a = byte(c[3] * opacity);
    if a == 0 {
        return None;
    }
    let ch = |v: f64| {
        let b = byte(v);
        if a < 255 { (b * a + 127) / 255 } else { b }
    };
    Some([ch(c[0]) as u8, ch(c[1]) as u8, ch(c[2]) as u8, a as u8])
}

fn norm(v: [f64; 3]) -> f64 {
    libm::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = norm(v);
    if l == 0.0 { v } else { v.map(|c| c / l) }
}

/// A carried normal through a placement's cofactors, then to unit length (left as is at length 0).
pub fn place_normal(m: &Affine, n: [f64; 3]) -> [f64; 3] {
    let l = |i: usize, j: usize| m[i % 3][j % 3];
    let cof =
        |i: usize, j: usize| l(i + 1, j + 1) * l(i + 2, j + 2) - l(i + 1, j + 2) * l(i + 2, j + 1);
    unit(std::array::from_fn(|i| {
        (cof(i, 0) * n[0] + cof(i, 1) * n[1]) + cof(i, 2) * n[2]
    }))
}

/// CE's unit normal of two edge vectors (contract §9, the corner normal from its second step).
pub fn ce_unit_normal(v1: [f64; 3], v2: [f64; 3]) -> [f64; 3] {
    let scaled = |v: [f64; 3]| {
        let m = v[0].abs().max(v[1].abs()).max(v[2].abs());
        if m == 0.0 { v } else { v.map(|c| c / m) }
    };
    let (a, b) = (scaled(v1), scaled(v2));
    let zero = |v: [f64; 3]| v == [0.0; 3];
    let n = if zero(a) && zero(b) {
        [0.0, -1.0, 0.0]
    } else {
        let u = if zero(a) { b } else { a };
        let c = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let cl = norm(c);
        if !zero(a) && !zero(b) && cl > 1e-6 {
            c.map(|v| v / cl)
        } else if u[0].abs() < 1e-6 && u[1].abs() < 1e-6 {
            [0.0, -1.0, 0.0]
        } else {
            unit([-u[0] * u[2], -u[1] * u[2], u[0] * u[0] + u[1] * u[1]])
        }
    };
    if zero(n) { [0.0, 1.0, 0.0] } else { n }
}

/// CE's shading of an encoded colour at a world point with unit normal `n` and light at `light`:
/// `δ = 0.5 (n · to_sun)³`, halved where negative, added to each channel (unclamped).
pub fn ce_shade(rgb: [f64; 3], n: [f64; 3], point: [f64; 3], light: [f64; 3]) -> [f64; 3] {
    let to_sun = unit([
        light[0] - point[0],
        light[1] - point[1],
        light[2] - point[2],
    ]);
    let f = dot(n, to_sun);
    let mut delta = 0.5 * (f * f * f);
    if delta < 0.0 {
        delta *= 0.5;
    }
    rgb.map(|c| c + delta)
}

fn srgb_decode(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        libm::pow((c + 0.055) / 1.055, 2.4)
    }
}

fn srgb_encode(v: f64) -> f64 {
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * libm::pow(v, 1.0 / 2.4) - 0.055
    }
}

/// The lighting a mesh corner takes.
pub struct Light<'a> {
    pub model: &'a LightModel,
    pub source: [f64; 3],
    /// The eye, `frame_center + d R_2`.
    pub eye: [f64; 3],
}

impl Light<'_> {
    /// A corner's lit colour at its placed position: encoded under CE's model, linear under soft
    /// diffuse. `None` for a model this oracle does not know (refused before colour).
    fn corner(&self, rgb: [f64; 3], n: [f64; 3], at: [f64; 3]) -> Option<[f64; 3]> {
        match self.model {
            LightModel::Ce => Some(ce_shade(rgb, n, at, self.source).map(clamp01)),
            LightModel::SoftDiffuse { ambient, wrap } => {
                let lin = rgb.map(|c| srgb_decode(clamp01(c)));
                let to_eye = [
                    self.eye[0] - at[0],
                    self.eye[1] - at[1],
                    self.eye[2] - at[2],
                ];
                let n = if dot(n, to_eye) < 0.0 {
                    n.map(|v| -v)
                } else {
                    n
                };
                let l = unit([
                    self.source[0] - at[0],
                    self.source[1] - at[1],
                    self.source[2] - at[2],
                ]);
                let k = ambient + (1.0 - ambient) * ((dot(n, l) + wrap) / (1.0 + wrap)).max(0.0);
                Some(lin.map(|c| c * k))
            }
            LightModel::Other { .. } => None,
        }
    }

    pub fn linear(&self) -> bool {
        matches!(self.model, LightModel::SoftDiffuse { .. })
    }
}

/// A lit mesh triangle's three corner colours: straight `fill`, placed corners `placed`, carried
/// normals where it has them.
pub fn lit_corners(
    light: &Light,
    fill: [f64; 4],
    placed: &[[f64; 3]; 3],
    normals: Option<&[[f64; 3]; 3]>,
) -> Option<[[f64; 3]; 3]> {
    let own = || {
        let c = placed;
        ce_unit_normal(
            [c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]],
            [c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]],
        )
    };
    let rgb = [fill[0], fill[1], fill[2]];
    let mut out = [[0.0; 3]; 3];
    for k in 0..3 {
        let n = normals.map_or_else(own, |n| n[k]);
        out[k] = light.corner(rgb, n, placed[k])?;
    }
    Some(out)
}

/// A lit triangle's bytes at a pixel: the corners interpolated as planes on the pixel plane
/// through its mapped corners `at`, read at each of `points` (the samples the fragment may cover);
/// the least and most bytes over them. Exact where `points` are the covered samples' centroid.
pub fn lit_bytes(
    linear: bool,
    corners: &[[f64; 3]; 3],
    at: [[f64; 2]; 3],
    alpha: f64,
    points: &[[f64; 2]],
) -> ([u8; 4], [u8; 4]) {
    let planes: [_; 3] = std::array::from_fn(|ch| {
        three_corner_plane(
            at,
            [corners[0][ch], corners[1][ch], corners[2][ch]],
            [0.0; 3],
        )
        .0
    });
    let bytes = |p: [f64; 2]| {
        let v: [f64; 3] = std::array::from_fn(|ch| {
            let v = clamp01(planes[ch].at(p[0], p[1]));
            if linear { srgb_encode(v) } else { v }
        });
        premultiply([v[0], v[1], v[2], alpha], 1.0)
    };
    let mut lo = [255u8; 4];
    let mut hi = [0u8; 4];
    for p in points {
        let b = bytes(*p);
        for k in 0..4 {
            lo[k] = lo[k].min(b[k]);
            hi[k] = hi[k].max(b[k]);
        }
    }
    (lo, hi)
}

/// The eye of a projection, `c + d R_2`.
pub fn eye(pr: &Projection) -> [f64; 3] {
    std::array::from_fn(|i| pr.c[i] + pr.d * pr.r[2][i])
}

/// A shown corner on the pixel plane.
pub fn mapped(map: &FrameMap, q: [f64; 3]) -> [f64; 2] {
    map.map(q[0], q[1])
}
