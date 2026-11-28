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
//!   (`Camera.get_cairo_context`), around the centre CE froze into it at the array's first path draw
//!   (the view's `mapping_centre`), and `points_to_subpixel_coords` for point clouds and images,
//!   around the frame's centre as it is. Both subtract a centre, so a 3D camera's centre is
//!   subtracted twice, once before projecting and once here, as CE does.
//!
//! The arithmetic is written in CE's order with plain IEEE operations and the pure-Rust libm, so
//! it is the same on every machine. NumPy multiplies through BLAS and takes sin, cos and exp from
//! the platform, so the results may differ from CE's in the last bits.

use crate::composite::Plane;
use crate::depth::{Direction, WorldPlane};
use crate::xf::{Record, RecordKind, Space, ThreeDCamera, View, place};

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

    /// The camera's zoom, which scales every projected point's position after the perspective.
    pub fn zoom(&self) -> f64 {
        self.zoom
    }

    /// This projection seen in the camera's own axes: no rotation and no centre, only the
    /// perspective and the zoom. A pre-projected record's points are already in those axes (as
    /// `project` leaves them), so their depth is keyed through this, never through the view's
    /// rotation, which a camera of the scene's own may not have drawn with.
    pub fn in_camera_axes(&self) -> Projection {
        Projection {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            frame_center: [0.0; 3],
            ..self.clone()
        }
    }

    /// The point in the camera's axes that `project` places at `shown`: `shown`'s depth, and its
    /// position before the perspective factor and the zoom.
    pub fn unproject(&self, shown: [f64; 3]) -> [f64; 3] {
        let scale = self.factor(shown[2]) * self.zoom;
        [shown[0] / scale, shown[1] / scale, shown[2]]
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

    /// How far the depth key `-factor(z)` of a point `project` placed at `shown` may stand from its
    /// exact value: the rotation rounds `q = R (p - centre)` by about `3u |q|` (taken with the
    /// centre's size), and `factor = d / (d - z)` moves by `f^2 / d` per unit of `z` and rounds by
    /// `2u` of itself, so the error grows as the point nears the eye.
    pub fn key_error(&self, shown: [f64; 3]) -> f64 {
        use crate::composite::U;
        let d = self.focal_distance;
        let f = self.factor(shown[2]);
        let c = self.frame_center;
        let reach = shown[2].abs()
            + libm::hypot(shown[0], shown[1]) / (f * self.zoom).abs()
            + (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        U * (2.0 * f.abs() + 4.0 * f * f / d.abs() * reach)
    }

    /// How far, as a share of itself, the depth key `-factor(z)` of a curve through points
    /// `project` placed at `shown` may move where each of those points moves by up to `moved` in
    /// the world ([`depth_share`]).
    pub fn depth_share(&self, shown: &[[f64; 3]], moved: f64) -> f64 {
        depth_share(
            self.focal_distance,
            self.exponential,
            shown.iter().map(|q| q[2]),
            moved,
        )
    }

    /// The depth key (`-factor`) across a world plane, as a plane on the pixel plane under the
    /// frame mapping `map`; `None` under the exponential option or where the plane is seen within
    /// `min_cos` of edge-on (the cosine between its normal and the sight line to `at`, a point on
    /// it). The caller makes sure every point it keys this way is in front of the eye
    /// ([`Projection::keys_planar_at`]). `region` (`[x0, y0, x1, y1]` in pixels) is where the
    /// plane is read: its error bound ([`Plane::err`]) holds there.
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
        region: [f64; 4],
    ) -> Option<crate::composite::Plane> {
        if self.exponential || self.zoom == 0.0 {
            return None;
        }
        let basis = self.key_basis(plane.normal, map);
        let unrounded = self.key_basis(plane.unrounded.normal, map);
        self.key_plane_on(
            (&basis, &unrounded),
            plane,
            map,
            self.rotated(at),
            min_cos,
            region,
        )
    }

    /// How far, as a share of itself, the depth key across `plane` ([`Projection::key_plane`]) may
    /// move at any pixel where the plane stands up to `distance` off it along its normal: what a
    /// world departure moves the key by. On the plane the key is `f = N / D`, `D` the plane's
    /// distance from the eye along its unit normal and `N` fixed by the pixel and the normal; moved
    /// by `distance` along the normal, `D` moves by as much and `f` by at most
    /// `|f| distance / (|D| - distance)`: the share returned is `distance / (|D| - distance)`. (In
    /// view depth that is `distance / |n . r|` for the sight line `r` with a unit z: the key moves
    /// by `f^2 / d` per unit of depth.) A plane tilted off it
    /// by no more than `distance` moves each pixel's key by as much, but for the tilt's change of
    /// the sight line's angle to the plane, of the order of the departure over the plane's size.
    /// Infinite where the moved plane may reach the eye.
    pub fn key_reach(&self, plane: &WorldPlane, map: &FrameMap, distance: f64) -> f64 {
        let exact = plane.unrounded;
        let basis = self.key_basis(exact.normal, map);
        let away =
            (exact.offset - basis.n_dot_centre - basis.m[2] * self.focal_distance).abs() - distance;
        if away.is_nan() || away <= 0.0 || !distance.is_finite() {
            return f64::INFINITY;
        }
        distance / away
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

    /// The rest of [`Projection::key_plane`]: `plane`, whose normal and unrounded normal have the
    /// bases `bases`, seen from `q`, the rotated point `at`, read in `region`.
    fn key_plane_on(
        &self,
        bases: (&KeyBasis, &KeyBasis),
        plane: &WorldPlane,
        map: &FrameMap,
        q: [f64; 3],
        min_cos: f64,
        region: [f64; 4],
    ) -> Option<crate::composite::Plane> {
        let (basis, unrounded) = bases;
        let d = self.focal_distance;
        let denominator = plane.offset - basis.n_dot_centre - basis.m[2] * d;
        let sight = [q[0], q[1], q[2] - d];
        let sight_len = (sight[0] * sight[0] + sight[1] * sight[1] + sight[2] * sight[2]).sqrt();
        let cos = denominator.abs() / (basis.m_len * sight_len);
        if cos.is_nan() || cos < min_cos || denominator == 0.0 {
            return None;
        }
        let exact = plane.unrounded;
        let exact_denominator = exact.offset - unrounded.n_dot_centre - unrounded.m[2] * d;
        if exact_denominator.is_nan() || exact_denominator.abs() <= exact.offset_err {
            return None; // The plane before rounding may pass through the eye.
        }
        let key = |b: &KeyBasis, denominator: f64| {
            [
                -(-b.gx * map.x0 - b.gy * map.y0 - b.m[2] * d) / denominator,
                -b.gx / denominator,
                -b.gy / denominator,
            ]
        };
        let [a, b, c] = key(basis, denominator);
        let before = key(unrounded, exact_denominator);
        let plane = crate::composite::Plane {
            a,
            b,
            c,
            err: self.key_plane_error(
                (unrounded, exact.offset, exact_denominator),
                [[a, b, c], before],
                exact.offset_err,
                map,
                region,
            ),
        };
        [plane.a, plane.b, plane.c, plane.err]
            .iter()
            .all(|v| v.is_finite())
            .then_some(plane)
    }

    /// How far [`Projection::key_plane_on`]'s plane may stand, in `region`, from the key across the
    /// exact plane its world plane was rounded from (its unrounded normal, and the exact offset
    /// along it). The key is `f = N / D`: `N = m . s` for `s = (-(X - x0) / (zoom xx),
    /// -(Y - y0) / (zoom yy), d)` (the pixel in scene units, and the focal distance) and
    /// `D = e - n . eye`, the plane's distance from the eye. `unrounded` is the unrounded plane's
    /// basis, offset and `D`; `keys` the rounded plane's key plane and the unrounded one's, as
    /// computed.
    ///
    /// The rounding to the grid is carried as it happened, not as its worst half step: both planes
    /// are keyed, and the two keys differ by an affine function of the pixel, largest at a corner
    /// of the region, read there (each read rounding by `4u` of its terms). The unrounded plane's
    /// key arithmetic adds, at a corner, `(dN + |f| dD) / |D| + 4u |f|`, where the rotation rounds
    /// `m` by `8u |m|` (so `N` by that times `|s|`, plus `4u` of `N`'s terms) and `D` by `4u` of
    /// its terms plus that times `|eye|`. Its offset lies within `offset_err` of the exact one,
    /// which moves `D` by as much: `|f| offset_err / (|D| - offset_err)`. Every term is convex
    /// across the region, so the sum is largest at a corner. Without bound as the plane nears the
    /// eye.
    fn key_plane_error(
        &self,
        unrounded: (&KeyBasis, f64, f64),
        keys: [[f64; 3]; 2],
        offset_err: f64,
        map: &FrameMap,
        region: [f64; 4],
    ) -> f64 {
        use crate::composite::U;
        let (d, c, r) = (self.focal_distance, self.frame_center, &self.rotation);
        let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let eye = [0, 1, 2].map(|i| c[i] + d * r[2][i]);
        let mut worst: f64 = 0.0;
        for x in [region[0], region[2]] {
            for y in [region[1], region[3]] {
                let s = [
                    (x - map.x0) / (self.zoom * map.xx),
                    (y - map.y0) / (self.zoom * map.yy),
                    d,
                ];
                // Each plane's own arithmetic, at this corner, and its key there.
                let arithmetic = |(basis, offset, denominator): (&KeyBasis, f64, f64)| {
                    let m = basis.m;
                    let (tx, ty) = (basis.gx * (x - map.x0), basis.gy * (y - map.y0));
                    let dn = 8.0 * U * basis.m_len;
                    let de = 4.0 * U * (offset.abs() + basis.m_len * norm(c));
                    let dd = de + dn * norm(eye) + 4.0 * U * (m[2] * d).abs();
                    let f = (m[2] * d - tx - ty).abs() / denominator.abs();
                    let dnum = dn * norm(s) + 4.0 * U * (tx.abs() + ty.abs() + (m[2] * d).abs());
                    ((dnum + f * dd) / denominator.abs() + 4.0 * U * f, f)
                };
                let (exact, f) = arithmetic(unrounded);
                let shift = f * offset_err / (unrounded.2.abs() - offset_err);
                let read = |k: [f64; 3]| {
                    (
                        k[0] + k[1] * x + k[2] * y,
                        k[0].abs() + (k[1] * x).abs() + (k[2] * y).abs(),
                    )
                };
                let ((at, terms), (before, before_terms)) = (read(keys[0]), read(keys[1]));
                let gap = (at - before).abs() + 4.0 * U * (terms + before_terms);
                worst = worst.max(gap + exact + shift);
            }
        }
        worst
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
            Direction::of(self.rotation[2]).map(|dir| {
                let bases = (
                    self.key_basis(dir.normal(), map),
                    self.key_basis(dir.unit(), map),
                );
                (dir, bases)
            })
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
    shared: Option<(Direction, (KeyBasis, KeyBasis))>,
}

impl FacingPlanes<'_> {
    /// [`Projection::facing_plane`] at `centre`.
    pub fn at(&self, centre: [f64; 3]) -> Plane {
        let p = self.projection;
        let own = || {
            let shown = p.project(centre);
            Plane {
                err: p.key_error(shown),
                ..Plane::constant(-p.factor(shown[2]))
            }
        };
        let q = p.rotated(centre);
        // `keys_planar_at`, from the rotated centre this needs anyway.
        let planar = !p.exponential && p.focal_distance - q[2] > 0.0;
        if !planar {
            return own();
        }
        let Some((dir, (basis, unrounded))) = &self.shared else {
            return own();
        };
        // Read anywhere on the frame, taken generously as the box twice the frame mapping's centre
        // about it.
        let (x0, y0) = (self.map.x0.abs(), self.map.y0.abs());
        let frame = [-x0, -y0, 3.0 * x0, 3.0 * y0];
        dir.through(centre)
            .and_then(|plane| p.key_plane_on((basis, unrounded), &plane, self.map, q, 0.0, frame))
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

/// The depth key a pre-projected record in `space` takes as a whole, when it takes one, `axes`
/// the projection in its camera's axes ([`Projection::in_camera_axes`]) under a 3D camera: as
/// [`placement`] keys a record placed by the engine, a fixed orientation facing the camera at its
/// centre where that camera placed it (`pivot`, [`produced_pivot`]). `None` for points keyed each
/// on its own: through `axes`, from where it puts them back ([`Projection::unproject`]).
pub fn produced_key(
    axes: Option<&Projection>,
    map: &FrameMap,
    space: Space,
    pivot: Option<[f64; 3]>,
) -> Option<Plane> {
    match (axes, space) {
        (Some(p), Space::World) => pivot.map(|pivot| p.facing_plane(pivot, map)),
        _ => Some(Plane::constant(f64::NEG_INFINITY)),
    }
}

/// Where a camera of the scene's own placed a pre-projected record's fixed-orientation centre, in
/// its axes, when the record has one: CE moves every point of a fixed orientation by its centre's
/// projection offset, so the centre moved as the record's first point did, from `own` (placed by
/// the record's placement) to `shown`. The centre is the record's own, not its points' middle: a
/// family's members turn about their parent's centre, and a centre function may name any point.
pub fn produced_pivot(record: &Record, own: &[[f64; 3]], shown: &[[f64; 3]]) -> Option<[f64; 3]> {
    let centre = record.depth.fixed_orientation_centre?;
    let own = place(&record.geometry.place_a, *own.first()?);
    let shown = *shown.first()?;
    Some(std::array::from_fn(|k| centre[k] + (shown[k] - own[k])))
}

/// How far, as a share of itself, the depth key `-factor(z)` of a curve through points at
/// projected depths `depths` (before the perspective, as `Projection::project` gives them) may move
/// where each of those points moves by up to `moved` in the world, under a camera at
/// `focal_distance`: `z`, rotated, moves by as much, and `factor = d / (d - z)` by at most
/// `factor moved / (d - z - moved)` of itself (at `z + moved`); a key along the curve is a
/// combination of its points' keys with weights summing to one, all of one sign, so it moves by
/// at most the most of their shares. Under the exponential option `factor` grows by at most
/// `1 / d` of itself per unit of `z`, so by at most `moved / (d - moved)`. Infinite where a moved
/// point may reach the eye.
pub fn depth_share(
    focal_distance: f64,
    exponential: bool,
    depths: impl IntoIterator<Item = f64>,
    moved: f64,
) -> f64 {
    let share = |room: f64| {
        if room - moved > 0.0 {
            moved / (room - moved)
        } else {
            f64::INFINITY
        }
    };
    if exponential {
        return share(focal_distance);
    }
    depths
        .into_iter()
        .map(|z| share(focal_distance - z))
        .fold(0.0, f64::max)
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

    /// The mapping a record of `kind` is drawn through in a frame of `width x height` pixels of
    /// `view`, as CE maps it: a path (a `path` record, a face group's faces, a path or mesh
    /// instance, a mesh) through the frame array's Cairo context, whose matrix holds the
    /// `mapping_centre` CE froze into it; a point cloud or an image by `points_to_subpixel_coords`,
    /// around the frame's centre as it is. Both are this matrix; only the centre differs.
    pub fn of_kind(view: &View, kind: RecordKind, width: u32, height: u32) -> FrameMap {
        FrameMap::cairo(
            width,
            height,
            view.frame_width,
            view.frame_height,
            mapping_centre_of(view, kind),
        )
    }

    /// A point through the matrix as Cairo transforms it (`cairo_matrix_transform_point`).
    pub fn apply(&self, p: [f64; 3]) -> [f64; 2] {
        [
            (self.xx * p[0] + 0.0 * p[1]) + self.x0,
            (0.0 * p[0] + self.yy * p[1]) + self.y0,
        ]
    }
}

/// The centre a record of `kind` maps to pixels around ([`FrameMap::of_kind`]).
pub fn mapping_centre_of(view: &View, kind: RecordKind) -> [f64; 3] {
    match kind {
        RecordKind::Points | RecordKind::Image => view.frame_center,
        _ => view.mapping_centre,
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
