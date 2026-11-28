//! The one place lighting lives.
//!
//! - CE's model, as `ThreeDCamera.modified_rgbas` applies it: each shaded face gets two colour
//!   stops, `get_shaded_rgb` at its start and end corners, which Cairo then interpolates as a
//!   linear gradient between the face's gradient ends (those corners, unless its mobject gives
//!   others). It adds light to the encoded colour, unclamped (Cairo clamps each stop), and ignores
//!   the camera's `shading_factor`, as CE does.
//! - Soft diffuse light, in linear light: diffuse light that wraps past the terminator, over an
//!   ambient floor, lit on the side the eye sees. Its parameters are the view's ([`SOFT_DIFFUSE`]
//!   by default). A surface's face is lit by its four corner normals, smooth across its grid
//!   ([`crate::normals`]), its share of light interpolated over the face ([`FaceShares`]); any
//!   other path at its centroid ([`soft_face_stops`]), one colour where it lies in one plane.
//! - A lit mesh triangle ([`Light::triangle`]): each corner lit by the view's model, then the
//!   corners' colours interpolated in screen space (Gouraud), read at the centroid of the
//!   sub-samples a fragment covers ([`TriangleColour`]). A GPU path mirrors this module.
//!
//! The arithmetic is plain IEEE operations in CE's order; NumPy takes its norms and dot products
//! through BLAS and `** 3` from the platform, so results may differ from CE's in the last bits.

use crate::composite::{Plane, mask_centroid, premultiply, srgb_decode, srgb_encode};
use crate::xf::{LightModel, ThreeDCamera};

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

/// The largest absolute coordinate of `v`.
fn max_abs(v: V3) -> f64 {
    v[0].abs().max(v[1].abs()).max(v[2].abs())
}

/// `len(vmob.get_anchors())` for `n` points: start and end anchors zipped, or the one point.
fn anchors(n: usize) -> usize {
    if n == 1 {
        return 1;
    }
    let starts = n.div_ceil(4);
    let ends = if n >= 4 { (n - 4) / 4 + 1 } else { 0 };
    2 * starts.min(ends)
}

/// The two vectors CE's normal at point `i` of a face's points is found from: to the points
/// three on and three back, wrapping as `get_3d_vmob_unit_normal` does.
fn corner_vectors(points: &[V3], i: usize) -> (V3, V3) {
    let n = points.len();
    let im3 = if i > 2 { i - 3 } else { n - 4 };
    let ip3 = if i < n - 3 { i + 3 } else { 3 };
    (sub(points[ip3], points[i]), sub(points[im3], points[i]))
}

/// The corners CE lights a face at: its first point and point `((n - 1) // 6) * 3`.
fn lit_corners(n: usize) -> [usize; 2] {
    [0, (n - 1) / 6 * 3]
}

/// The normals a writer that snaps a shaded face's points to `grid` carries for its two lit
/// corners (XF 1.2 `normals`), decided on the points as made: CE's normal at each
/// ([`ce_face_stops`]), or the zero vector at a degenerate corner, which the drawer lights by the
/// face's own normal ([`ce_face_stops_decided`]).
///
/// The degenerate-corner rule: a corner is degenerate when a point its normal is found from lies
/// off it by less than one grid step in every coordinate, coincident points included. CE's normal
/// there is its fixed fallback for coincident points, or the direction of the float noise that
/// keeps them apart (a sphere's pole, where `cos(pi / 2)` leaves its points about 1e-16 apart);
/// which of the two a pole gets is the rounding of the machine that made it, and in the points
/// drawn both are one snapped point. The boundary sits at one grid step, far from content on
/// both sides: poles and apexes at 0 or 1e-16, a smooth curve's points a hundred steps apart and
/// more.
pub fn decided_normals(points: &[V3], grid: f64) -> [V3; 2] {
    if points.is_empty() {
        return [UP; 2];
    }
    lit_corners(points.len()).map(|i| {
        if anchors(points.len()) > 2 {
            let (v1, v2) = corner_vectors(points, i);
            let short = |v: V3| max_abs(v) < grid;
            if short(v1) || short(v2) {
                return [0.0; 3];
            }
        }
        face_unit_normal(points, i)
    })
}

/// [`ce_face_stops`] with the corners' normals its writer decided ([`decided_normals`]): each as
/// carried, or at a degenerate corner (the zero vector) the face's Newell normal over `points`,
/// oriented as CE's corner normal is on a face it winds about; where that cancels too, CE's own
/// normal from `points`.
pub fn ce_face_stops_decided(
    points: &[V3],
    rgbas: &[[f64; 4]],
    light: V3,
    decided: [V3; 2],
) -> [[f64; 4]; 2] {
    face_stops(points, rgbas, light, |k, corner| {
        if decided[k] != [0.0; 3] {
            return decided[k];
        }
        let newell = crate::depth::newell(points, points[0]);
        if dot(newell, newell) > 0.0 {
            normalize(newell)
        } else {
            face_unit_normal(points, corner)
        }
    })
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
    if anchors(points.len()) <= 2 {
        return UP;
    }
    let (v1, v2) = corner_vectors(points, i);
    let normal = ce_unit_normal(v1, v2);
    if dot(normal, normal) == 0.0 {
        return UP;
    }
    normal
}

/// `ThreeDCamera.modified_rgbas` for a face CE shades (`shade_in_3d`, points present, shading
/// on): its first two colour stops (one repeated when it has one), lit at its start corner (the
/// first point) and its end corner (point `((n - 1) // 6) * 3`).
pub fn ce_face_stops(points: &[V3], rgbas: &[[f64; 4]], light: V3) -> [[f64; 4]; 2] {
    face_stops(points, rgbas, light, |_, corner| {
        face_unit_normal(points, corner)
    })
}

/// A face's two stops, lit at its corners by `normal(k, corner)` for the `k`th lit corner.
fn face_stops(
    points: &[V3],
    rgbas: &[[f64; 4]],
    light: V3,
    normal: impl Fn(usize, usize) -> V3,
) -> [[f64; 4]; 2] {
    let mut stops = match rgbas {
        [one] => [*one, *one],
        [a, b, ..] => [*a, *b],
        [] => panic!("a face has at least one colour"),
    };
    if points.is_empty() {
        return stops;
    }
    for (k, (stop, corner)) in stops.iter_mut().zip(lit_corners(points.len())).enumerate() {
        let rgb = ce_shaded_rgb(
            [stop[0], stop[1], stop[2]],
            points[corner],
            normal(k, corner),
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

/// The soft light a 3D scene is drawn in by default (Arachne's `core_light=soft`), and the one
/// place its parameters are set: a stylized look, soft and even. A surface facing the light shows
/// its own colour, never lighter; one facing away keeps half its light, so its colour stays
/// legible; the wrap of 1 (half Lambert) carries the light all the way round, so only a surface
/// facing straight away falls to the ambient share and a curved surface's shading runs long and
/// gentle. Nothing glows. A render may set others (`core_light_ambient`, `core_light_wrap`).
pub const SOFT_DIFFUSE: SoftDiffuse = SoftDiffuse {
    ambient: 0.5,
    wrap: 1.0,
};

impl SoftDiffuse {
    /// The share of light kept by a surface whose unit normal makes cosine `cos` with the
    /// direction to the light: the ambient floor, and above it the diffuse light wrapped past the
    /// terminator, `(cos + wrap) / (1 + wrap)`, none beyond `cos = -wrap`. Facing the light it is
    /// 1.
    pub fn share(&self, cos: f64) -> f64 {
        let diffuse = ((cos + self.wrap) / (1.0 + self.wrap)).max(0.0);
        self.ambient + (1.0 - self.ambient) * diffuse
    }

    /// The share of light at `point` facing `normal` (a unit vector), lit from a light at `light`.
    pub fn share_at(&self, point: V3, normal: V3, light: V3) -> f64 {
        self.share(dot(normal, normalize(sub(light, point))))
    }

    /// A linear colour at `point` facing `normal` (a unit vector), lit from a light at `light`
    /// and seen from `eye`; the normal is turned toward the eye first.
    pub fn lit(&self, linear: V3, point: V3, normal: V3, light: V3, eye: V3) -> V3 {
        scale(
            linear,
            self.share_at(point, toward(normal, sub(eye, point)), light),
        )
    }
}

/// `normal` turned to face `view` (the direction from the surface to the eye): lit on the side the
/// eye sees, whatever way the surface winds.
fn toward(normal: V3, view: V3) -> V3 {
    if dot(normal, view) < 0.0 {
        scale(normal, -1.0)
    } else {
        normal
    }
}

/// An encoded straight colour `rgba` lit by the share of light `share`, in linear light: decoded,
/// scaled, encoded again ([`encode`]); alpha as it is.
pub fn soft_colour(rgba: [f64; 4], share: f64) -> [f64; 4] {
    let lit = |c: f64| encode(srgb_decode(c.clamp(0.0, 1.0)) * share);
    [lit(rgba[0]), lit(rgba[1]), lit(rgba[2]), rgba[3]]
}

/// The steps of [`encode`]'s table over linear `0..=1`.
const ENCODE_STEPS: usize = 4096;

/// sRGB's encoding of a linear value clamped to `0..=1`, read from a table of [`srgb_encode`] at
/// [`ENCODE_STEPS`] steps, linearly between them: within 2e-5 of it (far under one level of 255),
/// and cheap enough to colour every pixel of a lit surface. The table is the same on every
/// machine, made by `libm`.
pub fn encode(linear: f64) -> f64 {
    static TABLE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        (0..=ENCODE_STEPS)
            .map(|i| srgb_encode(i as f64 / ENCODE_STEPS as f64))
            .collect()
    });
    let v = if linear.is_nan() {
        0.0
    } else {
        linear.clamp(0.0, 1.0)
    };
    let at = v * ENCODE_STEPS as f64;
    let i = (at as usize).min(ENCODE_STEPS - 1);
    let frac = at - i as f64;
    table[i] + (table[i + 1] - table[i]) * frac
}

/// The normal of the plane `points` lie on, made unit ([`WorldPlane::of_points`]: Newell's, the
/// widest triangle's where an outline crossing itself cancels that, or an axis), or `None` for
/// points on no one plane.
pub fn flat_normal(points: &[V3]) -> Option<V3> {
    let plane = crate::depth::WorldPlane::of_points(points)?;
    let n = normalize(plane.normal);
    (n != [0.0; 3]).then_some(n)
}

/// The normals the soft light lights a path outside a surface's grid by, at its two lit corners
/// (XF 1.2 `normals`), as a writer that snaps its points to `grid` carries them, decided on the
/// points as made: a flat path's plane normal at both, so it takes one colour whatever the turn
/// at its corners (shaded `Text`, whose glyphs each close with a line of no length, included);
/// for a path on no one plane, CE's corner normals as [`decided_normals`] decides them.
pub fn soft_decided_normals(points: &[V3], grid: f64) -> [V3; 2] {
    match flat_normal(points) {
        Some(n) => [n, n],
        None => decided_normals(points, grid),
    }
}

/// The soft light's two stops for a path outside a surface's grid: CE's two stops of its paint
/// (its first two rows, one repeated), each lit in linear light ([`soft_colour`]) at the centroid
/// of `points` by the normal its writer decided for that corner (`decided`, from
/// [`soft_decided_normals`]), turned toward `eye`; at a degenerate corner (the zero vector) by the
/// path's Newell normal, or where that cancels too, CE's own normal there. A flat path's two
/// stops are one colour.
pub fn soft_face_stops(
    points: &[V3],
    rgbas: &[[f64; 4]],
    model: &SoftDiffuse,
    light: V3,
    eye: V3,
    decided: [V3; 2],
) -> [[f64; 4]; 2] {
    let mut stops = match rgbas {
        [one] => [*one, *one],
        [a, b, ..] => [*a, *b],
        [] => panic!("a face has at least one colour"),
    };
    let Some(&first) = points.first() else {
        return stops;
    };
    let centroid = centroid(points);
    for (k, (stop, corner)) in stops.iter_mut().zip(lit_corners(points.len())).enumerate() {
        let normal = if decided[k] != [0.0; 3] {
            decided[k]
        } else {
            let newell = crate::depth::newell(points, first);
            if dot(newell, newell) > 0.0 {
                normalize(newell)
            } else {
                face_unit_normal(points, corner)
            }
        };
        let n = toward(normal, sub(eye, centroid));
        *stop = soft_colour(*stop, model.share_at(centroid, n, light));
    }
    stops
}

/// The one share of light the soft light gives a path lit as one piece: at the centroid of
/// `points` facing `normal` turned toward `eye`.
pub fn flat_share(model: &SoftDiffuse, points: &[V3], normal: V3, light: V3, eye: V3) -> f64 {
    let centroid = centroid(points);
    model.share_at(centroid, toward(normal, sub(eye, centroid)), light)
}

/// The centroid of `points`, summed about the first so it does not depend on where they are; the
/// origin for none.
fn centroid(points: &[V3]) -> V3 {
    let Some(&first) = points.first() else {
        return [0.0; 3];
    };
    let mut sum = [0.0; 3];
    for p in points {
        sum = [
            sum[0] + (p[0] - first[0]),
            sum[1] + (p[1] - first[1]),
            sum[2] + (p[2] - first[2]),
        ];
    }
    let count = points.len() as f64;
    std::array::from_fn(|k| first[k] + sum[k] / count)
}

/// The soft light across one face of a surface's grid, on the pixel plane: the share of light at
/// each of its four corners, interpolated over its two triangles, corners (0, 1, 2) and
/// (0, 2, 3) (Gouraud shading of the share; a layer colours a pixel by its own colour lit by the
/// share there, [`soft_colour`]). A face's corners share their shares with the faces meeting
/// there, so the light runs on across its edges. A point beyond the face (the centre of a pixel
/// its edge crosses) reads the share at the point of its triangle its barycentric weights, the
/// negative ones dropped, name: never beyond the corners' shares, however thin the triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceShares {
    triangles: [Triangle; 2],
    /// The diagonal from corner 0 to corner 2 on the pixel plane, and which side of it corner 1
    /// lies on (`0.0` where that triangle has no area, so the other is read everywhere; infinite
    /// where the other has none).
    origin: [f64; 2],
    along: [f64; 2],
    side: f64,
}

/// A triangle of a face on the pixel plane: its first corner, its edges from it, the inverse of
/// their cross product (`0.0` for none), and its corners' shares.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Triangle {
    a: [f64; 2],
    e1: [f64; 2],
    e2: [f64; 2],
    inverse: f64,
    shares: [f64; 3],
}

impl Triangle {
    fn new(px: [[f64; 2]; 3], shares: [f64; 3]) -> Triangle {
        let e1 = [px[1][0] - px[0][0], px[1][1] - px[0][1]];
        let e2 = [px[2][0] - px[0][0], px[2][1] - px[0][1]];
        let det = e1[0] * e2[1] - e1[1] * e2[0];
        let inverse = if det != 0.0 && det.is_finite() {
            1.0 / det
        } else {
            0.0
        };
        Triangle {
            a: px[0],
            e1,
            e2,
            inverse,
            shares,
        }
    }

    fn at(&self, x: f64, y: f64) -> f64 {
        let [s0, s1, s2] = self.shares;
        if self.inverse == 0.0 {
            return (s0 + s1 + s2) / 3.0;
        }
        let d = [x - self.a[0], y - self.a[1]];
        let b = (d[0] * self.e2[1] - d[1] * self.e2[0]) * self.inverse;
        let c = (self.e1[0] * d[1] - self.e1[1] * d[0]) * self.inverse;
        let weights = [1.0 - b - c, b, c].map(|w: f64| w.max(0.0));
        let sum = weights[0] + weights[1] + weights[2];
        (weights[0] * s0 + weights[1] * s1 + weights[2] * s2) / sum
    }
}

impl FaceShares {
    /// The face whose corners lie at `px` on the pixel plane with shares `shares`.
    pub fn new(px: [[f64; 2]; 4], shares: [f64; 4]) -> FaceShares {
        let along = [px[2][0] - px[0][0], px[2][1] - px[0][1]];
        let side_of =
            |k: usize| along[0] * (px[k][1] - px[0][1]) - along[1] * (px[k][0] - px[0][0]);
        let (one, three) = (side_of(1), side_of(3));
        let side = if one == 0.0 {
            0.0
        } else if three == 0.0 {
            f64::INFINITY.copysign(one)
        } else {
            one
        };
        FaceShares {
            triangles: [
                Triangle::new([px[0], px[1], px[2]], [shares[0], shares[1], shares[2]]),
                Triangle::new([px[0], px[2], px[3]], [shares[0], shares[2], shares[3]]),
            ],
            origin: px[0],
            along,
            side,
        }
    }

    /// The one share of a face whose corners all have it.
    pub fn constant(shares: [f64; 4]) -> Option<f64> {
        shares[1..].iter().all(|s| *s == shares[0]).then_some(shares[0])
    }

    /// The share at `(x, y)` on the pixel plane: read on the triangle the point lies on the side
    /// of the diagonal of.
    pub fn at(&self, x: f64, y: f64) -> f64 {
        let s = self.along[0] * (y - self.origin[1]) - self.along[1] * (x - self.origin[0]);
        let first = self.side != 0.0 && (self.side.is_infinite() || s * self.side >= 0.0);
        self.triangles[usize::from(!first)].at(x, y)
    }
}

/// The share of light at each of a face's four corners under the soft light: corner `k` at
/// `corners[k]` (in the world) facing `normals[k]`, the face's normals turned together toward
/// `eye` by their sum (so a face is lit on the side the eye sees as one piece).
pub fn face_corner_shares(
    model: &SoftDiffuse,
    corners: [V3; 4],
    normals: [V3; 4],
    light: V3,
    eye: V3,
) -> [f64; 4] {
    let facing = normals
        .iter()
        .fold([0.0; 3], |s, n| [s[0] + n[0], s[1] + n[1], s[2] + n[2]]);
    let centre = std::array::from_fn(|k| {
        (corners[0][k] + corners[1][k] + corners[2][k] + corners[3][k]) / 4.0
    });
    let flip = dot(facing, sub(eye, centre)) < 0.0;
    std::array::from_fn(|k| {
        let n = if flip {
            scale(normals[k], -1.0)
        } else {
            normals[k]
        };
        model.share_at(corners[k], n, light)
    })
}

/// A view's light model resolved against its 3D camera: what lights a shaded mesh's corners.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
    /// CE's model, lit from the camera's light source.
    Ce { source: V3 },
    /// Soft diffuse light from the camera's light source, seen from the camera's eye.
    Soft {
        model: SoftDiffuse,
        source: V3,
        eye: V3,
    },
}

impl Light {
    /// The light for a view's `model` under `camera`, whose projection rotates by `rotation`
    /// about `frame_center`. The eye is where CE's projection puts it: `focal_distance` along the
    /// rotation's third row from the frame's centre. An unknown model is refused before drawing.
    pub fn new(
        model: &LightModel,
        camera: &ThreeDCamera,
        frame_center: V3,
        rotation: [[f64; 3]; 3],
    ) -> Light {
        let source = camera.light_source;
        match model {
            LightModel::Ce => Light::Ce { source },
            LightModel::SoftDiffuse { ambient, wrap } => Light::Soft {
                model: SoftDiffuse {
                    ambient: *ambient,
                    wrap: *wrap,
                },
                source,
                eye: std::array::from_fn(|k| {
                    frame_center[k] + camera.focal_distance * rotation[2][k]
                }),
            },
            LightModel::Other { .. } => unreachable!("an unknown light model is refused"),
        }
    }

    /// A triangle's colour from its fill (encoded straight colour `rgb`, `alpha`), lit at its
    /// world-space corners `points` facing `normals` (each corner's own, or CE's normal of the
    /// face when the mesh has none), placed at `px` on the pixel plane. CE's model lights encoded
    /// colour, each corner clamped to `0..=1` as Cairo clamps a gradient's stops; soft diffuse
    /// lights the decoded colour in linear light.
    pub fn triangle(
        &self,
        rgb: V3,
        alpha: f64,
        points: [V3; 3],
        normals: Option<[V3; 3]>,
        px: [[f64; 2]; 3],
    ) -> TriangleColour {
        let n = normals.unwrap_or_else(|| {
            [ce_unit_normal(sub(points[1], points[0]), sub(points[2], points[0])); 3]
        });
        match self {
            Light::Ce { source } => {
                let values = [0, 1, 2].map(|k| {
                    ce_shaded_rgb(rgb, points[k], n[k], *source).map(|v| v.clamp(0.0, 1.0))
                });
                TriangleColour::gouraud(px, values, alpha, false)
            }
            Light::Soft { model, source, eye } => {
                let base = rgb.map(|v| srgb_decode(v.clamp(0.0, 1.0)));
                let values = [0, 1, 2].map(|k| model.lit(base, points[k], n[k], *source, *eye));
                TriangleColour::gouraud(px, values, alpha, true)
            }
        }
    }
}

/// A triangle's colour across the pixel plane.
///
/// Lit colour is interpolated in screen space (Gouraud shading, not perspective-corrected): each
/// channel is the plane through the corners' values on the pixel plane, as Cairo interpolates the
/// two stops of a face CE shades. A fragment reads it at the centroid of the sub-samples it
/// covers, clamps it to `0..=1`, encodes it when it is linear, and premultiplies it with the
/// triangle's alpha.
#[derive(Clone, Copy, Debug)]
pub enum TriangleColour {
    /// One premultiplied colour.
    Flat([u8; 4]),
    /// Encoded straight colour per channel, and the alpha.
    Encoded([Plane; 3], f64),
    /// Linear colour per channel, and the alpha.
    Linear([Plane; 3], f64),
}

impl TriangleColour {
    /// An unlit triangle: its fill, encoded straight colour `rgb` with `alpha`.
    pub fn flat(rgb: V3, alpha: f64) -> TriangleColour {
        TriangleColour::Flat(premultiply([rgb[0], rgb[1], rgb[2], alpha]))
    }

    /// The corners' colours `values` (linear or encoded) at `px`, interpolated.
    fn gouraud(px: [[f64; 2]; 3], values: [V3; 3], alpha: f64, linear: bool) -> TriangleColour {
        if values[0] == values[1] && values[1] == values[2] {
            let c = values[0].map(|v| encoded(v, linear));
            return TriangleColour::flat(c, alpha);
        }
        let planes = [0, 1, 2]
            .map(|k| Plane::through([0, 1, 2].map(|i| [px[i][0], px[i][1], values[i][k]])));
        if linear {
            TriangleColour::Linear(planes, alpha)
        } else {
            TriangleColour::Encoded(planes, alpha)
        }
    }

    /// The premultiplied colour of a fragment covering `mask` in pixel `(x, y)`.
    pub fn at(&self, x: u32, y: u32, mask: u16) -> [u8; 4] {
        let (planes, alpha, linear) = match self {
            TriangleColour::Flat(c) => return *c,
            TriangleColour::Encoded(p, a) => (p, *a, false),
            TriangleColour::Linear(p, a) => (p, *a, true),
        };
        let [cx, cy] = mask_centroid(x, y, mask);
        let rgb = planes.map(|p| encoded(p.at(cx, cy), linear));
        premultiply([rgb[0], rgb[1], rgb[2], alpha])
    }
}

/// A channel clamped to `0..=1`, encoded when it is linear.
fn encoded(v: f64, linear: bool) -> f64 {
    let v = v.clamp(0.0, 1.0);
    if linear { srgb_encode(v) } else { v }
}
