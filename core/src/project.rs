//! CE's camera maths, as `ThreeDCamera` and `Camera` compute them.
//!
//! - The 3D projection (`ThreeDCamera.project_points`): subtract the frame centre, rotate by
//!   `generate_rotation_matrix`, then scale x and y by `focal / (focal - z)` (or the exponential
//!   option's factor), times the zoom; a point at or past the eye's plane takes CE's factor of 10^6.
//!   z is kept, rotated.
//! - What a mobject's points become before display (`transform_points_pre_display`): a mobject with
//!   any non-finite point is drawn as one point at the origin; fixed-in-frame points are kept;
//!   fixed-orientation points move by their centre's projection offset; the rest are projected.
//! - The frame mappings: the Cairo context's matrix that paths are drawn through
//!   (`Camera.get_cairo_context`), and `points_to_subpixel_coords` for point clouds. Both subtract
//!   the frame centre, so a 3D camera's centre is subtracted twice, once before projecting and once
//!   here, as CE does.
//!
//! The arithmetic is written in CE's order with plain IEEE operations and the pure-Rust libm, so
//! it is the same on every machine. NumPy multiplies through BLAS and takes sin, cos and exp from
//! the platform, so the results may differ from CE's in the last bits.

use crate::composite::Plane;
use crate::depth::{Direction, WorldPlane};
use crate::xf::{Space, ThreeDCamera};

/// `arachne.constants.DEGREES`.
const DEGREES: f64 = std::f64::consts::TAU / 360.0;

type Matrix = [[f64; 3]; 3];

fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// `space_ops.rotation_about_z`.
fn rotation_about_z(angle: f64) -> Matrix {
    let (c, s) = (libm::cos(angle), libm::sin(angle));
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// `space_ops.rotation_matrix(angle, axis)`: SciPy's `Rotation.from_rotvec(angle * axis)`, then
/// `as_matrix`, for a unit axis.
fn rotation_about_axis(angle: f64, axis: [f64; 3]) -> Matrix {
    let v = [angle * axis[0], angle * axis[1], angle * axis[2]];
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let scale = if norm <= 1e-3 {
        let n2 = norm * norm;
        0.5 - n2 / 48.0 + n2 * n2 / 3840.0
    } else {
        libm::sin(norm / 2.0) / norm
    };
    let (x, y, z, w) = (
        scale * v[0],
        scale * v[1],
        scale * v[2],
        libm::cos(norm / 2.0),
    );
    let (x2, y2, z2, w2) = (x * x, y * y, z * z, w * w);
    let (xy, zw, xz, yw, yz, xw) = (x * y, z * w, x * z, y * w, y * z, x * w);
    [
        [x2 - y2 - z2 + w2, 2.0 * (xy - zw), 2.0 * (xz + yw)],
        [2.0 * (xy + zw), -x2 + y2 - z2 + w2, 2.0 * (yz - xw)],
        [2.0 * (xz - yw), 2.0 * (yz + xw), -x2 - y2 + z2 + w2],
    ]
}

/// `ThreeDCamera.generate_rotation_matrix`: about z by `-theta - 90°`, then about x by `-phi`,
/// then about z by `gamma`.
pub fn rotation_matrix(phi: f64, theta: f64, gamma: f64) -> Matrix {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut result = identity;
    for m in [
        rotation_about_z(-theta - 90.0 * DEGREES),
        rotation_about_axis(-phi, [1.0, 0.0, 0.0]),
        rotation_about_z(gamma),
    ] {
        result = matmul(&m, &result);
    }
    result
}

/// A 3D camera's projection.
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    rotation: Matrix,
    frame_center: [f64; 3],
    focal_distance: f64,
    zoom: f64,
    exponential: bool,
}

impl Projection {
    pub fn new(camera: &ThreeDCamera, frame_center: [f64; 3]) -> Projection {
        Projection {
            rotation: rotation_matrix(camera.phi, camera.theta, camera.gamma),
            frame_center,
            focal_distance: camera.focal_distance,
            zoom: camera.zoom,
            exponential: camera.exponential_projection,
        }
    }

    pub fn rotation(&self) -> Matrix {
        self.rotation
    }

    /// One point through `ThreeDCamera.project_points`.
    pub fn project(&self, p: [f64; 3]) -> [f64; 3] {
        let [x, y, z] = self.rotated(p);
        let scale = self.factor(z) * self.zoom;
        [x * scale, y * scale, z]
    }

    /// A point about the frame centre, rotated into the camera's axes (before perspective).
    fn rotated(&self, p: [f64; 3]) -> [f64; 3] {
        let c = self.frame_center;
        let q = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let r = &self.rotation;
        let row = |k: usize| q[0] * r[k][0] + q[1] * r[k][1] + q[2] * r[k][2];
        [row(0), row(1), row(2)]
    }

    /// The perspective factor `project` scales a point by at a projected depth `z` (before zoom),
    /// as `ThreeDCamera.project_points` takes it. The rasterizer's depth key is this, negated, so
    /// position and depth come from one formula.
    pub fn factor(&self, z: f64) -> f64 {
        let d = self.focal_distance;
        if self.exponential {
            if z < 0.0 {
                d / (d - z)
            } else {
                libm::exp(z / d)
            }
        } else if d - z < 0.0 {
            1e6
        } else {
            d / (d - z)
        }
    }

    /// Whether `factor` is `d / (d - z)` at a world point: in front of the eye, under CE's default
    /// projection. Only there is the depth key affine across a plane on screen (`key_plane`).
    pub fn keys_planar_at(&self, p: [f64; 3]) -> bool {
        !self.exponential && self.focal_distance - self.rotated(p)[2] > 0.0
    }

    /// The depth key (`-factor`) across a world plane, as a plane on the pixel plane under the
    /// frame mapping `map`; `None` under the exponential option or where the plane is seen within
    /// `min_cos` of edge-on (the cosine between its normal and the sight line to `at`, a point on
    /// it). The caller makes sure every point it keys this way is in front of the eye
    /// ([`Projection::keys_planar_at`]).
    ///
    /// With `q = R (p - centre)`, the plane `n . p = e` is `m . q = k` for `m = R n` and
    /// `k = e - n . centre`. Under `factor = d / (d - q_z)`, a point projects to
    /// `X = q_x f zoom`, `Y = q_y f zoom` with `q_z = d - d / f`, so on the plane
    /// `f = (m_x X / zoom + m_y Y / zoom - m_z d) / (k - m_z d)`: affine in `X` and `Y`, and so in
    /// pixels. The same plane always gives the same numbers, so things on one plane tie exactly.
    pub fn key_plane(
        &self,
        plane: &WorldPlane,
        map: &FrameMap,
        at: [f64; 3],
        min_cos: f64,
    ) -> Option<crate::composite::Plane> {
        if self.exponential || self.zoom == 0.0 {
            return None;
        }
        let basis = self.key_basis(plane.normal, map);
        self.key_plane_on(&basis, plane.offset, map, self.rotated(at), min_cos)
    }

    /// The part of [`Projection::key_plane`] that depends on the plane's normal alone.
    fn key_basis(&self, n: [f64; 3], map: &FrameMap) -> KeyBasis {
        let (c, r) = (self.frame_center, &self.rotation);
        let m = [0, 1, 2].map(|i| r[i][0] * n[0] + r[i][1] * n[1] + r[i][2] * n[2]);
        KeyBasis {
            m,
            m_len: (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt(),
            n_dot_centre: n[0] * c[0] + n[1] * c[1] + n[2] * c[2],
            gx: m[0] / (self.zoom * map.xx),
            gy: m[1] / (self.zoom * map.yy),
        }
    }

    /// The rest of [`Projection::key_plane`]: the plane with normal basis `basis` at `offset`,
    /// seen from `q`, the rotated point `at`.
    fn key_plane_on(
        &self,
        basis: &KeyBasis,
        offset: f64,
        map: &FrameMap,
        q: [f64; 3],
        min_cos: f64,
    ) -> Option<crate::composite::Plane> {
        let (m, d) = (basis.m, self.focal_distance);
        let k = offset - basis.n_dot_centre;
        let denominator = k - m[2] * d;
        let sight = [q[0], q[1], q[2] - d];
        let sight_len = (sight[0] * sight[0] + sight[1] * sight[1] + sight[2] * sight[2]).sqrt();
        let cos = denominator.abs() / (basis.m_len * sight_len);
        if cos.is_nan() || cos < min_cos || denominator == 0.0 {
            return None;
        }
        let (gx, gy) = (basis.gx, basis.gy);
        let plane = crate::composite::Plane {
            a: -(-gx * map.x0 - gy * map.y0 - m[2] * d) / denominator,
            b: -gx / denominator,
            c: -gy / denominator,
        };
        [plane.a, plane.b, plane.c]
            .iter()
            .all(|v| v.is_finite())
            .then_some(plane)
    }

    /// The depth key across the plane through `centre` that faces the camera, as a plane on the
    /// pixel plane: where a fixed-orientation mobject sits. It is made canonical as a world plane is
    /// ([`WorldPlane`]), so a surface facing the camera through the same point gets the same numbers
    /// and the two tie by draw order. Where a plane cannot key depth (the exponential option, a
    /// centre at or behind the eye), it is the centre's own key everywhere.
    pub fn facing_plane(&self, centre: [f64; 3], map: &FrameMap) -> Plane {
        self.facing_planes(map).at(centre)
    }

    /// [`Projection::facing_plane`] for many centres under the frame mapping `map`.
    pub fn facing_planes<'a>(&'a self, map: &'a FrameMap) -> FacingPlanes<'a> {
        // The plane `q_z = const` in the camera's axes: its world normal is the rotation's last row.
        let shared = if self.exponential || self.zoom == 0.0 {
            None
        } else {
            Direction::of(self.rotation[2]).map(|dir| (dir, self.key_basis(dir.normal(), map)))
        };
        FacingPlanes {
            projection: self,
            map,
            shared,
        }
    }
}

/// What [`Projection::key_plane`] works out from a plane's normal alone.
#[derive(Clone, Copy, Debug)]
struct KeyBasis {
    /// The normal in the camera's axes.
    m: [f64; 3],
    m_len: f64,
    /// The normal against the frame centre.
    n_dot_centre: f64,
    /// The key's gradient across the frame, before the denominator.
    gx: f64,
    gy: f64,
}

/// The camera-facing planes of one camera and frame mapping ([`Projection::facing_planes`]), with
/// what every centre shares worked out once: a point cloud keys each point by one.
pub struct FacingPlanes<'a> {
    projection: &'a Projection,
    map: &'a FrameMap,
    shared: Option<(Direction, KeyBasis)>,
}

impl FacingPlanes<'_> {
    /// [`Projection::facing_plane`] at `centre`.
    pub fn at(&self, centre: [f64; 3]) -> Plane {
        let p = self.projection;
        let own = || Plane::constant(-p.factor(p.project(centre)[2]));
        let q = p.rotated(centre);
        // `keys_planar_at`, from the rotated centre this needs anyway.
        let planar = !p.exponential && p.focal_distance - q[2] > 0.0;
        if !planar {
            return own();
        }
        let Some((dir, basis)) = &self.shared else {
            return own();
        };
        dir.through(centre)
            .and_then(|plane| p.key_plane_on(basis, plane.offset, self.map, q, 0.0))
            .unwrap_or_else(own)
    }
}

/// How a mobject's points are placed before display.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Placement {
    /// A 2D camera: points are kept.
    Flat,
    /// A 3D camera: points are projected.
    Projected,
    /// Fixed in frame (`add_fixed_in_frame_mobjects`): points are kept.
    FixedInFrame,
    /// Fixed orientation (`add_fixed_orientation_mobjects`) about this centre: points move by the
    /// centre's projection offset.
    FixedOrientation([f64; 3]),
}

/// How a record in `space` is placed for display under a camera whose projection, if it is a 3D
/// camera, is `projection`; and the depth key it takes as a whole, when it takes one: the plane a
/// fixed orientation faces the camera on ([`Projection::facing_plane`], through the frame mapping
/// `map`), or in front of all depth (overlay records under a 3D camera, and everything under a
/// flat one, where records tie and paint in draw order). Projected points take each the plane a
/// fixed orientation centred on them faces the camera on.
pub fn placement(
    projection: Option<&Projection>,
    map: &FrameMap,
    space: Space,
    fixed_orientation_centre: Option<[f64; 3]>,
) -> (Placement, Option<Plane>) {
    let in_front = Some(Plane::constant(f64::NEG_INFINITY));
    match (projection, space) {
        (Some(p), Space::World) => match fixed_orientation_centre {
            Some(centre) => (
                Placement::FixedOrientation(centre),
                Some(p.facing_plane(centre, map)),
            ),
            None => (Placement::Projected, None),
        },
        (Some(_), Space::Overlay) => (Placement::FixedInFrame, in_front),
        _ => (Placement::Flat, in_front),
    }
}

/// `transform_points_pre_display`. A 3D placement needs the projection.
pub fn pre_display(
    points: &[[f64; 3]],
    projection: Option<&Projection>,
    placement: Placement,
) -> Vec<[f64; 3]> {
    let all_finite = points.iter().flatten().all(|v| v.is_finite());
    let points: &[[f64; 3]] = if all_finite { points } else { &[[0.0; 3]] };
    let projection = || projection.expect("a 3D placement needs the camera's projection");
    match placement {
        Placement::Flat | Placement::FixedInFrame => points.to_vec(),
        Placement::Projected => {
            let p = projection();
            points.iter().map(|q| p.project(*q)).collect()
        }
        Placement::FixedOrientation(centre) => {
            let moved = projection().project(centre);
            let offset = [
                moved[0] - centre[0],
                moved[1] - centre[1],
                moved[2] - centre[2],
            ];
            points
                .iter()
                .map(|q| [q[0] + offset[0], q[1] + offset[1], q[2] + offset[2]])
                .collect()
        }
    }
}

/// The matrix of CE's Cairo context, from frame coordinates to pixels:
/// `x' = xx * x + x0`, `y' = yy * y + y0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameMap {
    pub xx: f64,
    pub yy: f64,
    pub x0: f64,
    pub y0: f64,
}

impl FrameMap {
    /// `Camera.get_cairo_context`'s matrix.
    pub fn cairo(
        pixel_width: u32,
        pixel_height: u32,
        frame_width: f64,
        frame_height: f64,
        frame_center: [f64; 3],
    ) -> FrameMap {
        let (pw, ph) = (pixel_width as f64, pixel_height as f64);
        FrameMap {
            xx: pw / frame_width,
            yy: -(ph / frame_height),
            x0: (pw / 2.0) - frame_center[0] * (pw / frame_width),
            y0: (ph / 2.0) + frame_center[1] * (ph / frame_height),
        }
    }

    /// A point through the matrix as Cairo transforms it (`cairo_matrix_transform_point`).
    pub fn apply(&self, p: [f64; 3]) -> [f64; 2] {
        [
            (self.xx * p[0] + 0.0 * p[1]) + self.x0,
            (0.0 * p[0] + self.yy * p[1]) + self.y0,
        ]
    }
}

/// `Camera.points_to_subpixel_coords` for points already through `pre_display`.
pub fn subpixel_coords(
    points: &[[f64; 3]],
    pixel_width: u32,
    pixel_height: u32,
    frame_width: f64,
    frame_height: f64,
    frame_center: [f64; 3],
) -> Vec<[f64; 2]> {
    let (pw, ph) = (pixel_width as f64, pixel_height as f64);
    let width_mult = pw / frame_width;
    let width_add = pw / 2.0;
    let height_mult = -(ph / frame_height);
    let height_add = ph / 2.0;
    points
        .iter()
        .map(|p| {
            [
                (p[0] - frame_center[0]) * width_mult + width_add,
                (p[1] - frame_center[1]) * height_mult + height_add,
            ]
        })
        .collect()
}
