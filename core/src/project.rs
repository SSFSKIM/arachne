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

use crate::xf::ThreeDCamera;

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
        let c = self.frame_center;
        let q = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let r = &self.rotation;
        let row = |k: usize| q[0] * r[k][0] + q[1] * r[k][1] + q[2] * r[k][2];
        let (x, y, z) = (row(0), row(1), row(2));
        let d = self.focal_distance;
        let factor = if self.exponential {
            if z < 0.0 {
                d / (d - z)
            } else {
                libm::exp(z / d)
            }
        } else if d - z < 0.0 {
            1e6
        } else {
            d / (d - z)
        };
        let scale = factor * self.zoom;
        [x * scale, y * scale, z]
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
