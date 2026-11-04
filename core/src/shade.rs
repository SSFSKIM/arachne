//! The one place lighting lives.
//!
//! - CE's model, as `ThreeDCamera.modified_rgbas` applies it: each shaded face gets two colour
//!   stops, `get_shaded_rgb` at its start and end corners, which Cairo then interpolates as a
//!   linear gradient between those corners. It adds light to the encoded colour, unclamped (Cairo
//!   clamps each stop), and ignores the camera's `shading_factor`, as CE does.
//! - Soft diffuse light, in linear light: diffuse light that wraps past the terminator, over an
//!   ambient floor, lit on the side the eye sees. Its parameters are the view's.
//!
//! The arithmetic is plain IEEE operations in CE's order; NumPy takes its norms and dot products
//! through BLAS and `** 3` from the platform, so results may differ from CE's in the last bits.

type V3 = [f64; 3];

/// CE's `UP`.
const UP: V3 = [0.0, 1.0, 0.0];
/// CE's `DOWN`.
const DOWN: V3 = [0.0, -1.0, 0.0];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `space_ops.normalize`: a zero vector stays zero.
fn normalize(v: V3) -> V3 {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    } else {
        [0.0; 3]
    }
}

/// `space_ops.get_unit_normal(v1, v2)`, with its tolerance of `1e-6` and its fallbacks.
pub fn ce_unit_normal(v1: V3, v2: V3) -> V3 {
    const TOL: f64 = 1e-6;
    let max_abs = |v: V3| v[0].abs().max(v[1].abs()).max(v[2].abs());
    let (div1, div2) = (max_abs(v1), max_abs(v2));
    let u = if div1 == 0.0 {
        if div2 == 0.0 {
            return DOWN;
        }
        scale_div(v2, div2)
    } else if div2 == 0.0 {
        scale_div(v1, div1)
    } else {
        let (u1, u2) = (scale_div(v1, div1), scale_div(v2, div2));
        let cp = [
            u1[1] * u2[2] - u1[2] * u2[1],
            u1[2] * u2[0] - u1[0] * u2[2],
            u1[0] * u2[1] - u1[1] * u2[0],
        ];
        let cp_norm = dot(cp, cp).sqrt();
        if cp_norm > TOL {
            return scale_div(cp, cp_norm);
        }
        u1
    };
    if u[0].abs() < TOL && u[1].abs() < TOL {
        return DOWN;
    }
    let cp = [-u[0] * u[2], -u[1] * u[2], u[0] * u[0] + u[1] * u[1]];
    scale_div(cp, dot(cp, cp).sqrt())
}

fn scale_div(v: V3, d: f64) -> V3 {
    [v[0] / d, v[1] / d, v[2] / d]
}

/// `color.get_shaded_rgb`: light or shadow added to a colour at `point` facing `unit_normal`.
pub fn ce_shaded_rgb(rgb: V3, point: V3, unit_normal: V3, light: V3) -> V3 {
    let to_sun = normalize(sub(light, point));
    let d = dot(unit_normal, to_sun);
    let mut add = 0.5 * (d * d * d);
    if add < 0.0 {
        add *= 0.5;
    }
    [rgb[0] + add, rgb[1] + add, rgb[2] + add]
}

/// `three_d_utils.get_3d_vmob_unit_normal` for the point at `i` of a face's points.
fn face_unit_normal(points: &[V3], i: usize) -> V3 {
    let n = points.len();
    // `len(vmob.get_anchors())`: start and end anchors zipped, or the one point.
    let anchors = if n == 1 {
        1
    } else {
        let starts = n.div_ceil(4);
        let ends = if n >= 4 { (n - 4) / 4 + 1 } else { 0 };
        2 * starts.min(ends)
    };
    if anchors <= 2 {
        return UP;
    }
    let im3 = if i > 2 { i - 3 } else { n - 4 };
    let ip3 = if i < n - 3 { i + 3 } else { 3 };
    let normal = ce_unit_normal(sub(points[ip3], points[i]), sub(points[im3], points[i]));
    if dot(normal, normal) == 0.0 {
        return UP;
    }
    normal
}

/// `ThreeDCamera.modified_rgbas` for a face CE shades (`shade_in_3d`, points present, shading
/// on): its first two colour stops (one repeated when it has one), lit at its start corner (the
/// first point) and its end corner (point `((n - 1) // 6) * 3`).
pub fn ce_face_stops(points: &[V3], rgbas: &[[f64; 4]], light: V3) -> [[f64; 4]; 2] {
    let mut stops = match rgbas {
        [one] => [*one, *one],
        [a, b, ..] => [*a, *b],
        [] => panic!("a face has at least one colour"),
    };
    if points.is_empty() {
        return stops;
    }
    let end = (points.len() - 1) / 6 * 3;
    for (stop, corner) in stops.iter_mut().zip([0, end]) {
        let rgb = ce_shaded_rgb(
            [stop[0], stop[1], stop[2]],
            points[corner],
            face_unit_normal(points, corner),
            light,
        );
        stop[..3].copy_from_slice(&rgb);
    }
    stops
}

/// Soft diffuse light (the view's light model 1). Light is white; colours are linear.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftDiffuse {
    /// The share of light a surface facing away from the light keeps, in `0..=1`.
    pub ambient: f64,
    /// How far the light wraps past the terminator, at least 0 (0 is Lambert's cosine).
    pub wrap: f64,
}

impl SoftDiffuse {
    /// A linear colour at `point` facing `normal` (a unit vector), lit from a light at `light`
    /// and seen from `eye`; the normal is turned toward the eye first.
    pub fn lit(&self, linear: V3, point: V3, normal: V3, light: V3, eye: V3) -> V3 {
        let n = if dot(normal, sub(eye, point)) < 0.0 {
            scale(normal, -1.0)
        } else {
            normal
        };
        let to_light = normalize(sub(light, point));
        let diffuse = ((dot(n, to_light) + self.wrap) / (1.0 + self.wrap)).max(0.0);
        let share = self.ambient + (1.0 - self.ambient) * diffuse;
        scale(linear, share)
    }
}
