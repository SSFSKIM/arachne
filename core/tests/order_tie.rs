//! The error bounds two depths tie within (`order::tied`) against the core's own key arithmetic:
//! one depth computed each way the core computes it, on random faces and near-degenerate ones
//! (thin triangles, nearly parallel edges, corners near the eye, faces nearly edge-on), must lie
//! within the two depths' summed bounds.
//!
//! On a plane the depth key is affine on screen (`Projection::key_plane`), so where the core keys
//! one point of a face several ways they agree but for rounding and the face plane's rounding to
//! the grid: the face's plane (`key_plane`, of the plane the corners lie on, rounded), the plane
//! through a triangle's keyed corners (`Plane::through_with_errors`), and the plane along an
//! outline's segment between keyed vertices (`Centreline::plane_at`), read at a point on the
//! segment; and two segments meeting at a vertex, read at the vertex.

use arachne_core::composite::Plane;
use arachne_core::depth::{Centreline, WorldPlane};
use arachne_core::order::tied;
use arachne_core::project::{FrameMap, Projection};
use arachne_core::xf::ThreeDCamera;

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// `lo..hi` spread over its orders of magnitude.
    fn log_range(&mut self, lo: f64, hi: f64) -> f64 {
        libm::exp(libm::log(lo) + (libm::log(hi) - libm::log(lo)) * self.unit())
    }
}

type V = [f64; 3];

fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V, s: f64) -> V {
    a.map(|v| v * s)
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: V) -> V {
    scale(a, 1.0 / dot(a, a).sqrt())
}

struct Scene {
    camera: ThreeDCamera,
    projection: Projection,
    map: FrameMap,
    /// The eye in world space.
    eye: V,
}

impl Scene {
    fn random(rng: &mut Rng) -> Scene {
        let camera = ThreeDCamera {
            phi: rng.range(0.0, std::f64::consts::PI),
            theta: rng.range(-std::f64::consts::PI, std::f64::consts::PI),
            gamma: rng.range(-0.3, 0.3),
            focal_distance: rng.log_range(2.0, 60.0),
            zoom: rng.log_range(0.3, 3.0),
            exponential_projection: false,
            light_source: [-7.0, -9.0, 10.0],
            shading_factor: 0.2,
            should_apply_shading: true,
        };
        let centre = [
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
            rng.range(-1.0, 1.0),
        ];
        let (w, h) = [(854, 480), (1920, 1080), (3840, 2160)][(rng.unit() * 3.0) as usize % 3];
        let projection = Projection::new(&camera, centre);
        // The eye sits at the focal distance along the camera's view axis (the rotation's last
        // row, in world space) from the frame centre.
        let r = projection.rotation();
        let eye = add(centre, scale(r[2], camera.focal_distance));
        Scene {
            camera,
            projection,
            map: FrameMap::cairo(w, h, 14.222222222222221, 8.0, centre),
            eye,
        }
    }

    fn key(&self, q: V) -> f64 {
        -self.projection.factor(self.projection.project(q)[2])
    }

    fn screen(&self, q: V) -> [f64; 2] {
        self.map.apply(self.projection.project(q))
    }

    /// A point on screen, its key and that key's error bound, as the core keys a point.
    fn keyed(&self, q: V) -> ([f64; 2], f64, f64) {
        let shown = self.projection.project(q);
        (
            self.screen(q),
            self.key(q),
            self.projection.key_error(shown),
        )
    }
}

/// A face of three corners on an exact plane, and the canonical plane the core rounds it to
/// (each unit normal component and the offset to the nearest grid step).
struct Face {
    plane: WorldPlane,
    corners: [V; 3],
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Ordinary,
    Thin,
    NearEye,
    EdgeOn,
}

fn on(normal: V, offset: f64, p: V) -> V {
    sub(
        p,
        scale(normal, (dot(normal, p) - offset) / dot(normal, normal)),
    )
}

fn face(s: &Scene, rng: &mut Rng, kind: Kind) -> Option<Face> {
    let r = s.projection.rotation();
    let (right, up, back) = (r[0], r[1], r[2]);
    // A centre in view, at a depth along the view axis; near the eye for `NearEye`.
    let d = s.camera.focal_distance;
    let depth = match kind {
        Kind::NearEye => d - rng.log_range(1e-3, 0.2 * d),
        _ => rng.range(-0.5 * d, 0.8 * d),
    };
    let spread = 3.0 / s.camera.zoom;
    let centre = add(
        s.eye,
        add(
            scale(back, depth - d),
            add(
                scale(right, rng.range(-spread, spread)),
                scale(up, rng.range(-spread, spread)),
            ),
        ),
    );
    let sight = unit(sub(centre, s.eye));
    let random = unit([
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    ]);
    let normal = match kind {
        // Within about 1e-9..1e-2 of perpendicular to the sight line.
        Kind::EdgeOn => {
            let across = unit(cross(sight, random));
            unit(add(across, scale(sight, rng.log_range(1e-9, 1e-2))))
        }
        _ => random,
    };
    // The face's plane is `normal . p = offset` exactly; the core's is it rounded to the grid.
    let offset = dot(normal, centre);
    let plane = WorldPlane::rounded(normal, offset, 0.0);
    if dot(plane.normal, plane.normal) == 0.0 {
        return None;
    }
    // The face's own axes: across the normal from a second direction (the first may be the normal).
    let aside = [
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    ];
    let u = unit(cross(normal, aside));
    let w = unit(cross(normal, u));
    let size = rng.log_range(0.01, 4.0)
        * if matches!(kind, Kind::NearEye) {
            0.01
        } else {
            1.0
        };
    let at = |a: f64, b: f64| {
        on(
            normal,
            offset,
            add(centre, add(scale(u, a * size), scale(w, b * size))),
        )
    };
    let corners = match kind {
        // The third corner within 1e-9..1e-2 of the size off the line through the others.
        Kind::Thin => {
            let t = rng.range(-0.5, 1.5);
            [at(0.0, 0.0), at(1.0, 0.0), at(t, rng.log_range(1e-9, 1e-2))]
        }
        _ => [
            at(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)),
            at(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)),
            at(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)),
        ],
    };
    // In front of the eye, as the core keys planes only there.
    if !corners.iter().all(|c| s.projection.keys_planar_at(*c)) {
        return None;
    }
    Some(Face { plane, corners })
}

/// The worst gap seen for one pair of ways as a share of the two depths' summed error bounds
/// (above 1: outside them), and where.
#[derive(Default)]
struct Worst {
    share: f64,
    seen: usize,
    case: String,
}

impl Worst {
    fn see(&mut self, (a, ea): (f64, f64), (b, eb): (f64, f64), case: impl Fn() -> String) {
        self.seen += 1;
        if a == b {
            return;
        }
        let share = (a - b).abs() / (ea + eb);
        if share > self.share || share.is_nan() {
            self.share = share;
            self.case = case();
        }
    }
}

/// A plane read at `x`, with its error bound there.
fn read(p: &Plane, x: [f64; 2]) -> (f64, f64) {
    (p.at(x[0], x[1]), p.err + p.read_error(x[0], x[1]))
}

/// The plane along the segment between keyed points `a` and `b` that the core reads at `at`.
fn segment(a: ([f64; 2], f64, f64), b: ([f64; 2], f64, f64), at: [f64; 2]) -> Plane {
    let line = Centreline::with_errors(&[(a.0, a.1), (b.0, b.1)], &[a.2, b.2], &[]);
    line.plane_at(
        at[0].floor() as u32,
        at[1].floor() as u32,
        &[],
        f64::INFINITY,
    )
}

/// The screen box of keyed points, a pixel around it.
fn region(k: &[([f64; 2], f64, f64)]) -> [f64; 4] {
    let (mut b0, mut b1, mut b2, mut b3) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (p, _, _) in k {
        (b0, b1, b2, b3) = (b0.min(p[0]), b1.min(p[1]), b2.max(p[0]), b3.max(p[1]));
    }
    [b0 - 1.0, b1 - 1.0, b2 + 1.0, b3 + 1.0]
}

#[test]
fn one_depth_computed_each_way_the_core_keys_it_lies_within_their_error_bounds() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut worst: Vec<(&str, Kind, Worst)> = Vec::new();
    for kind in [Kind::Ordinary, Kind::Thin, Kind::NearEye, Kind::EdgeOn] {
        let (mut through, mut along, mut vertex, mut reversed) = (
            Worst::default(),
            Worst::default(),
            Worst::default(),
            Worst::default(),
        );
        let (mut faces, mut tries) = (0, 0);
        while faces < 3000 {
            tries += 1;
            assert!(
                tries < 1_000_000,
                "{kind:?}: {faces} faces in {tries} tries"
            );
            let s = Scene::random(&mut rng);
            let Some(f) = face(&s, &mut rng, kind) else {
                continue;
            };
            let k = f.corners.map(|c| s.keyed(c));
            // Each pixel a point lands in must be on the frame's pixel grid for `plane_at`.
            if k.iter()
                .any(|(p, _, _)| !(p[0] >= 0.0 && p[1] >= 0.0 && p[0] < 4e4 && p[1] < 4e4))
            {
                continue;
            }
            let Some(kp) = s
                .projection
                .key_plane(&f.plane, &s.map, f.corners[0], 1e-9, region(&k))
            else {
                continue;
            };
            faces += 1;
            let describe = |what: &str, q: V, a: (f64, f64), b: (f64, f64)| {
                format!(
                    "{what}: camera {:?}, plane {:?}, corners {:?}, point {q:?} at {:?}: {a:?} vs {b:?}",
                    s.camera,
                    f.plane,
                    f.corners,
                    s.screen(q)
                )
            };
            let tri = Plane::through_with_errors(
                k.map(|(p, d, _)| [p[0], p[1], d]),
                k.map(|(_, _, e)| e),
            );
            for _ in 0..8 {
                // A point inside the face (barycentric), some near an edge.
                let (mut l1, mut l2) = (rng.unit(), rng.unit());
                if l1 + l2 > 1.0 {
                    (l1, l2) = (1.0 - l1, 1.0 - l2);
                }
                let q = add(
                    f.corners[0],
                    add(
                        scale(sub(f.corners[1], f.corners[0]), l1),
                        scale(sub(f.corners[2], f.corners[0]), l2),
                    ),
                );
                let x = s.screen(q);
                let (a, b) = (read(&kp, x), read(&tri, x));
                through.see(a, b, || describe("key_plane against through", q, a, b));
                // A point on an edge, keyed along the edge's segment and on the face's plane.
                let (i, j) = [(0, 1), (1, 2), (2, 0)][(rng.unit() * 3.0) as usize % 3];
                let t = rng.unit();
                let q = add(f.corners[i], scale(sub(f.corners[j], f.corners[i]), t));
                let x = s.screen(q);
                let (a, b) = (read(&kp, x), read(&segment(k[i], k[j], x), x));
                along.see(a, b, || describe("key_plane against a segment", q, a, b));
                // The same edge as the neighbouring face's outline traverses it, the other way.
                let c = read(&segment(k[j], k[i], x), x);
                reversed.see(b, c, || {
                    describe("a segment against itself reversed", q, b, c)
                });
            }
            // Two segments meeting at each corner, read at the corner.
            for i in 0..3 {
                let (prev, next) = ((i + 2) % 3, (i + 1) % 3);
                let x = k[i].0;
                let (a, b) = (
                    read(&segment(k[prev], k[i], x), x),
                    read(&segment(k[i], k[next], x), x),
                );
                vertex.see(a, b, || {
                    describe("two segments at a vertex", f.corners[i], a, b)
                });
            }
        }
        worst.push(("key_plane against through", kind, through));
        worst.push(("key_plane against a segment", kind, along));
        worst.push(("a segment against itself reversed", kind, reversed));
        worst.push(("two segments at a vertex", kind, vertex));
    }
    let mut failed = Vec::new();
    for (what, kind, w) in &worst {
        eprintln!(
            "{what:36} {kind:?}: worst gap {:.3} of the two bounds over {} points",
            w.share, w.seen
        );
        if w.share.is_nan() || w.share > 1.0 {
            failed.push(w.case.clone());
        }
    }
    assert!(
        failed.is_empty(),
        "gaps beyond the bounds:\n{}",
        failed.join("\n")
    );
}

#[test]
fn a_face_and_its_own_outline_tie_where_the_outline_runs() {
    // The face's plane is its world plane rounded to the grid; its outline keys its own corners.
    // Read on the outline, the two differ by that rounding, which the face's bound counts: they
    // tie, and paint order decides between them.
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (mut seen, mut tries) = (0, 0);
    while seen < 2000 {
        tries += 1;
        assert!(tries < 1_000_000, "{seen} faces in {tries} tries");
        let s = Scene::random(&mut rng);
        let Some(f) = face(&s, &mut rng, Kind::Ordinary) else {
            continue;
        };
        let k = f.corners.map(|c| s.keyed(c));
        if k.iter()
            .any(|(p, _, _)| !(p[0] >= 0.0 && p[1] >= 0.0 && p[0] < 4e4 && p[1] < 4e4))
        {
            continue;
        }
        let Some(kp) = s
            .projection
            .key_plane(&f.plane, &s.map, f.corners[0], 0.05, region(&k))
        else {
            continue;
        };
        seen += 1;
        let t = rng.unit();
        let q = add(f.corners[0], scale(sub(f.corners[1], f.corners[0]), t));
        let x = s.screen(q);
        let (a, b) = (read(&kp, x), read(&segment(k[0], k[1], x), x));
        assert!(
            tied(a.0, a.1, b.0, b.1),
            "plane {:?}, corners {:?}, at {x:?}: {a:?} against {b:?}",
            f.plane,
            f.corners
        );
    }
}
