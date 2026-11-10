//! The camera, placement kinds and the frame mapping (contract §2).
//!
//! Every formula is evaluated in the contract's order of operations, with libm's `sin`, `cos` and
//! `exp`, and no fused multiply-add, so a projected or mapped point equals the core's to the last
//! bit: the shared kurbo then flattens identical input.

use arachne_core::xf::{Camera, Record, Space, ThreeDCamera, View};

/// A degree: `τ / 360`.
const DEG: f64 = std::f64::consts::TAU / 360.0;

/// CE's behind-the-eye factor.
pub const BEHIND_EYE: f64 = 1e6;

pub type Mat3 = [[f64; 3]; 3];

/// `Rz(a)`.
pub fn rotation_about_z(a: f64) -> Mat3 {
    let (c, s) = (libm::cos(a), libm::sin(a));
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// `Rx(a)`, as a rotation by `a` about the axis `(1, 0, 0)` written through its quaternion.
pub fn rotation_about_x(a: f64) -> Mat3 {
    let v = [a * 1.0, a * 0.0, a * 0.0];
    let n = ((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]).sqrt();
    let kappa = if n <= 1e-3 {
        let n2 = n * n;
        (0.5 - n2 / 48.0) + (n2 * n2) / 3840.0
    } else {
        libm::sin(n / 2.0) / n
    };
    let (x, y, z, w) = (kappa * v[0], kappa * v[1], kappa * v[2], libm::cos(n / 2.0));
    let (x2, y2, z2, w2) = (x * x, y * y, z * z, w * w);
    let (xy, zw, xz, yw, yz, xw) = (x * y, z * w, x * z, y * w, y * z, x * w);
    [
        [((x2 - y2) - z2) + w2, 2.0 * (xy - zw), 2.0 * (xz + yw)],
        [2.0 * (xy + zw), ((-x2 + y2) - z2) + w2, 2.0 * (yz - xw)],
        [2.0 * (xz - yw), 2.0 * (yz + xw), ((-x2 - y2) + z2) + w2],
    ]
}

/// `M · N`, each entry `(M[i][0]·N[0][j] + M[i][1]·N[1][j]) + M[i][2]·N[2][j]`.
pub fn matmul(m: &Mat3, n: &Mat3) -> Mat3 {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (m[i][0] * n[0][j] + m[i][1] * n[1][j]) + m[i][2] * n[2][j])
    })
}

pub const IDENTITY3: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A 3D camera's projection about its frame centre.
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    /// The rotation `R`.
    pub r: Mat3,
    /// The frame centre `c` the projection subtracts.
    pub c: [f64; 3],
    /// The focal distance `d`.
    pub d: f64,
    /// The zoom `z0`.
    pub zoom: f64,
    pub exponential: bool,
}

impl Projection {
    /// The projection of `cam` about `frame_center`.
    pub fn new(cam: &ThreeDCamera, frame_center: [f64; 3]) -> Projection {
        let mut r = IDENTITY3;
        for f in [
            rotation_about_z(-cam.theta - 90.0 * DEG),
            rotation_about_x(-cam.phi),
            rotation_about_z(cam.gamma),
        ] {
            r = matmul(&f, &r);
        }
        Projection {
            r,
            c: frame_center,
            d: cam.focal_distance,
            zoom: cam.zoom,
            exponential: cam.exponential_projection,
        }
    }

    /// The same camera in its own axes: `R = I`, `c = 0`.
    pub fn in_camera_axes(&self) -> Projection {
        Projection {
            r: IDENTITY3,
            c: [0.0; 3],
            ..self.clone()
        }
    }

    pub fn rotated(&self, p: [f64; 3]) -> [f64; 3] {
        let q = [p[0] - self.c[0], p[1] - self.c[1], p[2] - self.c[2]];
        std::array::from_fn(|k| (q[0] * self.r[k][0] + q[1] * self.r[k][1]) + q[2] * self.r[k][2])
    }

    /// The perspective factor at rotated depth `z`.
    pub fn factor(&self, z: f64) -> f64 {
        let d = self.d;
        if self.exponential {
            if z < 0.0 {
                d / (d - z)
            } else {
                libm::exp(z / d)
            }
        } else if d - z < 0.0 {
            BEHIND_EYE
        } else {
            d / (d - z)
        }
    }

    pub fn project(&self, p: [f64; 3]) -> [f64; 3] {
        let [x, y, z] = self.rotated(p);
        let s = self.factor(z) * self.zoom;
        [x * s, y * s, z]
    }

    pub fn unproject(&self, p: [f64; 3]) -> [f64; 3] {
        let s = self.factor(p[2]) * self.zoom;
        [p[0] / s, p[1] / s, p[2]]
    }

    /// The depth key of a projected point whose rotated depth is `z`: the factor negated.
    pub fn key(&self, z: f64) -> f64 {
        -self.factor(z)
    }

    /// Whether a point at rotated depth `z` is keyed planar: only there is a key affine across a
    /// plane on screen.
    pub fn keyed_planar(&self, z: f64) -> bool {
        !self.exponential && self.d - z > 0.0
    }
}

/// How a record's placed points become shown points (contract §2, "Before display").
#[derive(Clone, Debug, PartialEq)]
pub enum Placement {
    /// Kept as placed: a `screen` record, every record under a flat or moving camera, and a
    /// pre-projected record's tail. Key `−∞` unless pre-projected.
    Kept,
    /// Through the projection.
    Projected,
    /// Moved by `project(C) − C` for its centre `C`.
    FixedOrientation { centre: [f64; 3], offset: [f64; 3] },
    /// The overlay's (`overlay` space under a 3D camera): never drawn by the core.
    FixedInFrame,
}

impl Placement {
    /// The placement kind of `rec` under `view`'s camera.
    pub fn of(rec: &Record, view: &View) -> Placement {
        let Camera::ThreeD(cam) = &view.camera else {
            return Placement::Kept;
        };
        if rec.pre_projected.is_some() {
            return Placement::Kept;
        }
        match rec.depth.space {
            Space::Screen => Placement::Kept,
            Space::Overlay => Placement::FixedInFrame,
            Space::World => match rec.depth.fixed_orientation_centre {
                Some(centre) => {
                    let p = Projection::new(cam, view.frame_center).project(centre);
                    Placement::FixedOrientation {
                        centre,
                        offset: [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]],
                    }
                }
                None => Placement::Projected,
            },
        }
    }

    /// A placed point shown, under `projection` where the camera is 3D.
    pub fn show(&self, projection: Option<&Projection>, p: [f64; 3]) -> [f64; 3] {
        match self {
            Placement::Projected => projection.map_or(p, |pr| pr.project(p)),
            Placement::FixedOrientation { offset, .. } => {
                [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]]
            }
            Placement::Kept | Placement::FixedInFrame => p,
        }
    }
}

/// CE's Cairo context matrix: frame units to pixels about a centre `m`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameMap {
    pub xx: f64,
    pub yy: f64,
    pub x0: f64,
    pub y0: f64,
}

impl FrameMap {
    /// The map about the centre `m`.
    pub fn about(view: &View, m: [f64; 3]) -> FrameMap {
        let (w, h) = (view.pixel_width as f64, view.pixel_height as f64);
        let (fw, fh) = (view.frame_width, view.frame_height);
        FrameMap {
            xx: w / fw,
            yy: -(h / fh),
            x0: (w / 2.0) - m[0] * (w / fw),
            y0: (h / 2.0) + m[1] * (h / fh),
        }
    }

    /// The map paths, faces, meshes and instances take: about the view's `mapping_centre`.
    pub fn of_paths(view: &View) -> FrameMap {
        FrameMap::about(view, view.mapping_centre)
    }

    /// `((xx·p0 + 0·p1) + x0, (0·p0 + yy·p1) + y0)`.
    pub fn map(&self, x: f64, y: f64) -> [f64; 2] {
        [
            (self.xx * x + 0.0 * y) + self.x0,
            (0.0 * x + self.yy * y) + self.y0,
        ]
    }

    /// The larger of the map's two scales, `s = max(|xx|, |yy|)`.
    pub fn scale(&self) -> f64 {
        self.xx.abs().max(self.yy.abs())
    }
}

/// How point clouds and images map: about `frame_center`,
/// `((p0 − c0)·(W / fw) + W / 2, (p1 − c1)·(−(H / fh)) + H / 2)`.
pub fn map_point_cloud(view: &View, p: [f64; 3]) -> [f64; 2] {
    let (w, h) = (view.pixel_width as f64, view.pixel_height as f64);
    let c = view.frame_center;
    [
        (p[0] - c[0]) * (w / view.frame_width) + w / 2.0,
        (p[1] - c[1]) * (-(h / view.frame_height)) + h / 2.0,
    ]
}

/// The view's projection, when its camera is 3D.
pub fn projection(view: &View) -> Option<Projection> {
    match &view.camera {
        Camera::ThreeD(cam) => Some(Projection::new(cam, view.frame_center)),
        _ => None,
    }
}
