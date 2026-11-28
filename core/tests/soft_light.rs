//! The soft light, the core's default light for 3D scenes: its terms (an ambient floor, a wrap
//! past the terminator, linear light), the smooth normals it lights a surface's grid by (welded at
//! a closed seam, joined at a pole, an open sheet's edges kept apart), shaded text lit flat, and
//! the XF 1.6 field that carries the corner normals. Under CE's light nothing of it is decided or
//! written.

mod common;

use arachne_core::composite::{srgb_decode, srgb_encode};
use arachne_core::normals::corner_normals;
use arachne_core::shade::{
    FaceShares, SOFT_DIFFUSE, SoftDiffuse, decided_normals, encode, face_corner_shares,
    soft_colour, soft_decided_normals, soft_face_stops,
};
use arachne_core::stream::Stream;
use arachne_core::xf::wire::{decode_delta, encode_delta};
use arachne_core::xf::*;

type V3 = [f64; 3];

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(v: V3) -> V3 {
    let n = dot(v, v).sqrt();
    v.map(|c| c / n)
}

fn close(a: V3, b: V3, tol: f64) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
}

/// Whether `a` lies along `b`, either way: the soft light turns a normal to the eye, so which
/// way a surface's parametrization winds does not matter.
fn along(a: V3, b: V3, tol: f64) -> bool {
    close(a, b, tol) || close(a, b.map(|v| -v), tol)
}

/// CE's `np.linspace(lo, hi, n + 1)`.
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let step = (hi - lo) / n as f64;
    (0..=n)
        .map(|i| if i == n { hi } else { lo + i as f64 * step })
        .collect()
}

/// A surface's faces as CE's `Surface` makes them: for each `u` index, each `v` index, the corners
/// `(u1, v1), (u2, v1), (u2, v2), (u1, v2)` closed again, as four straight cubics in uv space,
/// every point then mapped by `f`. Returns the points and each face's indices.
fn uv_surface(
    nu: usize,
    nv: usize,
    u: (f64, f64),
    v: (f64, f64),
    f: impl Fn(f64, f64) -> V3,
) -> (Vec<V3>, Vec<[u32; 2]>) {
    let (us, vs) = (linspace(u.0, u.1, nu), linspace(v.0, v.1, nv));
    let (mut points, mut uv) = (Vec::new(), Vec::new());
    for i in 0..nu {
        for j in 0..nv {
            let c = [
                [us[i], vs[j]],
                [us[i + 1], vs[j]],
                [us[i + 1], vs[j + 1]],
                [us[i], vs[j + 1]],
                [us[i], vs[j]],
            ];
            for w in c.windows(2) {
                let (a, b) = (w[0], w[1]);
                for t in [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0] {
                    let p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
                    points.push(f(p[0], p[1]));
                }
            }
            uv.push([i as u32, j as u32]);
        }
    }
    (points, uv)
}

fn faces(uv: Vec<[u32; 2]>, grid: [u32; 2], fill: Key) -> FaceGroup {
    FaceGroup {
        uv,
        grid,
        fill,
        stroke: None,
        face_ends: Vec::new(),
        face_widths: Vec::new(),
    }
}

/// CE's `Sphere`: u around, v from the south pole to the north.
fn sphere(nu: usize, nv: usize, r: f64) -> (Vec<V3>, Vec<[u32; 2]>) {
    uv_surface(nu, nv, (0.0, TAU), (0.0, PI), |u, v| {
        [
            r * libm::cos(u) * libm::sin(v),
            r * libm::sin(u) * libm::sin(v),
            -r * libm::cos(v),
        ]
    })
}

/// The corner normal at corner `k` of face `f`.
fn at(normals: &[V3], f: usize, k: usize) -> V3 {
    normals[4 * f + k]
}

/// The face index of `(i, j)` in CE's order.
fn face(i: usize, j: usize, nv: usize) -> usize {
    i * nv + j
}

#[test]
fn the_soft_lights_terms_are_an_ambient_floor_a_wrap_and_linear_light() {
    let soft = SoftDiffuse {
        ambient: 0.3,
        wrap: 0.5,
    };
    // Facing the light, a surface shows its own colour: the share is 1.
    assert_eq!(soft.share(1.0), 1.0);
    // At the terminator the light wraps past it: (0 + wrap) / (1 + wrap) above the floor.
    assert!((soft.share(0.0) - (0.3 + 0.7 * (0.5 / 1.5))).abs() < 1e-15);
    // From `cos = -wrap` on, and facing away, the ambient floor alone.
    assert_eq!(soft.share(-0.5), 0.3);
    assert_eq!(soft.share(-1.0), 0.3);
    // The share rises with the cosine.
    let shares: Vec<f64> = (-10..=10).map(|i| soft.share(i as f64 / 10.0)).collect();
    assert!(shares.windows(2).all(|w| w[0] <= w[1]), "{shares:?}");
    // Light scales the colour in linear light: half the light on a mid grey is not half its code.
    let grey = soft_colour([0.5, 0.5, 0.5, 0.8], 0.5);
    let want = srgb_encode(srgb_decode(0.5) * 0.5);
    assert!((grey[0] - want).abs() < 2e-5, "{grey:?}");
    assert!((grey[0] - 0.25).abs() > 0.1, "{grey:?}");
    assert_eq!(grey[3], 0.8, "alpha is not lit");
    // Fully lit, a colour is its own.
    let own = soft_colour([0.2, 0.6, 0.9, 1.0], 1.0);
    for (k, c) in [0.2, 0.6, 0.9].into_iter().enumerate() {
        assert!((own[k] - c).abs() < 2e-5, "{own:?}");
    }
    // The table the encoding is read from stays within 2e-5 of sRGB's curve.
    for i in 0..=20_000 {
        let v = i as f64 / 20_000.0;
        assert!((encode(v) - srgb_encode(v)).abs() < 2e-5, "at {v}");
    }
    // The default light is a soft one: a floor that keeps colour legible, a wide wrap.
    const { assert!(SOFT_DIFFUSE.ambient > 0.0 && SOFT_DIFFUSE.ambient < 1.0) };
    const { assert!(SOFT_DIFFUSE.wrap > 0.0) };
}

#[test]
fn an_open_patch_keeps_its_edges_apart() {
    // Half a cylinder: its first and last columns lie two units apart, so they never weld.
    let (nu, nv) = (12, 3);
    let (points, uv) = uv_surface(nu, nv, (0.0, PI), (0.0, 1.0), |u, v| {
        [libm::cos(u), v, libm::sin(u)]
    });
    let n = corner_normals(&points, &faces(uv, [nu as u32, nv as u32], Key([0; 16])));
    let du = PI / nu as f64;
    for j in 0..nv {
        // Each edge vertex takes the normal of the faces on its own side alone: half a step in.
        let first = at(&n, face(0, j, nv), 0);
        let last = at(&n, face(nu - 1, j, nv), 1);
        let inward = |a: f64| [libm::cos(a), 0.0, libm::sin(a)];
        assert!(along(first, inward(du / 2.0), 1e-12), "{first:?}");
        assert!(along(last, inward(PI - du / 2.0), 1e-12), "{last:?}");
    }
    // An interior vertex is the radial direction there, shared by the four faces meeting on it.
    let mid = [libm::cos(du * 5.0), 0.0, libm::sin(du * 5.0)];
    let corners = [
        at(&n, face(4, 1, nv), 2),
        at(&n, face(5, 1, nv), 3),
        at(&n, face(4, 2, nv), 1),
        at(&n, face(5, 2, nv), 0),
    ];
    for c in corners {
        assert_eq!(c, corners[0], "one vertex, one normal");
        assert!(along(c, mid, 1e-12), "{c:?}");
    }
}

#[test]
fn a_closed_seam_whose_corners_differ_in_their_last_bits_welds() {
    let (nu, nv) = (16, 4);
    let (points, uv) = uv_surface(nu, nv, (0.0, TAU), (-1.0, 1.0), |u, v| {
        [libm::cos(u), libm::sin(u), v]
    });
    // The seam's two columns are not the same points: `sin(TAU)` is not 0.
    let (start, end) = (points[0], points[16 * face(nu - 1, 0, nv) + 4]);
    assert_ne!(start, end);
    assert!(close(start, end, 1e-15));
    let n = corner_normals(&points, &faces(uv, [nu as u32, nv as u32], Key([0; 16])));
    for j in 0..nv {
        let first = at(&n, face(0, j, nv), 0);
        let last = at(&n, face(nu - 1, j, nv), 1);
        assert_eq!(first, last, "the seam is one vertex");
        assert!(along(first, [1.0, 0.0, 0.0], 1e-12), "{first:?}");
    }
}

#[test]
fn a_spheres_poles_are_one_vertex_each_lit_along_its_axis() {
    let (nu, nv) = (24, 12);
    let (points, uv) = sphere(nu, nv, 1.5);
    let n = corner_normals(&points, &faces(uv, [nu as u32, nv as u32], Key([0; 16])));
    // Every face around the south pole meets it with corners 0 and 1, the north with 2 and 3:
    // one normal for all of them, the pole's own direction, so no face around it stands out.
    let south = at(&n, face(0, 0, nv), 0);
    let north = at(&n, face(0, nv - 1, nv), 2);
    assert!(along(south, [0.0, 0.0, -1.0], 1e-12), "{south:?}");
    assert!(along(north, [0.0, 0.0, 1.0], 1e-12), "{north:?}");
    for i in 0..nu {
        for k in [0, 1] {
            assert_eq!(at(&n, face(i, 0, nv), k), south, "face {i} corner {k}");
        }
        for k in [2, 3] {
            assert_eq!(at(&n, face(i, nv - 1, nv), k), north, "face {i} corner {k}");
        }
    }
    // Elsewhere the normal is the sphere's own to within a part of the grid's step, the seam
    // included.
    for (f, p) in points.chunks(16).enumerate() {
        for k in 0..4 {
            let c = p[4 * k];
            if c[2].abs() < 1.49 {
                let (want, got) = (unit(c), at(&n, f, k));
                assert!(dot(got, want).abs() > 0.995, "face {f} corner {k}: {got:?}");
            }
        }
    }
    // Lit, the faces around a pole share its share of light: the pole is one point of colour.
    let soft = SOFT_DIFFUSE;
    let (light, eye) = ([-7.0, -9.0, 10.0], [0.0, 0.0, 20.0]);
    let pole_share = |f: usize, k: usize| {
        let p = &points[16 * f..16 * f + 16];
        let corners = [p[0], p[4], p[8], p[12]];
        let normals = [at(&n, f, 0), at(&n, f, 1), at(&n, f, 2), at(&n, f, 3)];
        face_corner_shares(&soft, corners, normals, light, eye)[k]
    };
    let first = pole_share(face(0, nv - 1, nv), 2);
    for i in 0..nu {
        for k in [2, 3] {
            let s = pole_share(face(i, nv - 1, nv), k);
            assert!((s - first).abs() < 1e-12, "face {i}: {s} against {first}");
        }
    }
}

#[test]
fn a_cones_apex_is_one_vertex() {
    // CE's `Cone`: u along the slant from the apex, v around.
    let (nu, nv) = (6, 20);
    let slant = 2.0f64.sqrt();
    let (points, uv) = uv_surface(nu, nv, (0.0, slant), (0.0, TAU), |u, v| {
        let r = u / slant;
        [r * libm::cos(v), r * libm::sin(v), 1.0 - u / slant]
    });
    let n = corner_normals(&points, &faces(uv, [nu as u32, nv as u32], Key([0; 16])));
    let apex = at(&n, face(0, 0, nv), 0);
    for j in 0..nv {
        for k in [0, 3] {
            assert_eq!(at(&n, face(0, j, nv), k), apex, "face {j} corner {k}");
        }
    }
    // The faces around the apex face up and out alike: their sum points up the axis.
    assert!(along(apex, [0.0, 0.0, 1.0], 1e-12), "{apex:?}");
}

#[test]
fn a_corner_moved_off_its_vertex_keeps_its_faces_own_normal() {
    let (nu, nv) = (4, 4);
    let (mut points, uv) = uv_surface(nu, nv, (0.0, 1.0), (0.0, 1.0), |u, v| {
        [u, v, 0.3 * u * u]
    });
    // Lift one face clear of its neighbours.
    let lifted = face(2, 2, nv);
    for p in &mut points[16 * lifted..16 * lifted + 16] {
        p[2] += 0.5;
    }
    let n = corner_normals(&points, &faces(uv, [nu as u32, nv as u32], Key([0; 16])));
    let p = &points[16 * lifted..16 * lifted + 16];
    let own = unit({
        let (a, b) = (
            [p[8][0] - p[0][0], p[8][1] - p[0][1], p[8][2] - p[0][2]],
            [p[12][0] - p[4][0], p[12][1] - p[4][1], p[12][2] - p[4][2]],
        );
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    });
    for k in 0..4 {
        assert!(along(at(&n, lifted, k), own, 1e-12), "corner {k}");
    }
}

/// A glyph's outline as `Text` makes it: each contour a run of straight cubics closed at its first
/// point, then a cubic of no length from its last point to its first ("make sure last curve is
/// closed").
fn glyph(contours: &[Vec<[f64; 2]>]) -> Vec<V3> {
    let mut points = Vec::new();
    for corners in contours {
        let mut ring: Vec<[f64; 2]> = corners.clone();
        ring.push(corners[0]);
        for w in ring.windows(2) {
            let (a, b) = (w[0], w[1]);
            for t in [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0] {
                points.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.0]);
            }
        }
        let first = [corners[0][0], corners[0][1], 0.0];
        points.extend([first; 4]);
    }
    points
}

/// A glyph tipped back in space, so its plane faces partly toward the light and partly away.
fn tipped(points: &[V3]) -> Vec<V3> {
    let (c, s) = (libm::cos(1.1), libm::sin(1.1));
    points
        .iter()
        .map(|p| [p[0], c * p[1] - s * p[2] + 0.2, s * p[1] + c * p[2]])
        .collect()
}

/// The tests' 3D view under the default soft light, over a clear background (so a pixel the
/// drawing covers whole is opaque, and its edge is not).
fn soft_view() -> View {
    let mut view = common::three_d_view(192, 108);
    view.light = LightModel::SoftDiffuse {
        ambient: SOFT_DIFFUSE.ambient,
        wrap: SOFT_DIFFUSE.wrap,
    };
    view.background = Background::Color([0.0; 4]);
    view
}

fn glyph_stream(view: View, points: Vec<V3>, normals: bool) -> Stream {
    let shape = Shape::new(ShapeData::Path { points });
    let mut r = Record::new(RecordKind::Path, 1, 101);
    r.geometry.shape_a = Some(shape.key());
    r.depth.space = Space::World;
    r.depth.shade_in_3d = true;
    r.paint.fill = Paint::One([0.9, 0.3, 0.2, 1.0]);
    let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
    d.view = Some(view);
    d.upload = vec![shape];
    d.set = vec![r];
    d.order = Some(vec![OrderEntry {
        layer: Layer::Moving,
        slot: 1,
        part: 0,
    }]);
    let mut s = Stream::new();
    s.apply(&d).expect("a valid scene");
    if normals {
        let decided = s.normals(&[1], 0.0).unwrap()[&1].clone();
        let mut again = d.clone();
        again.view = None;
        again.upload.clear();
        again.set[0].normals = Some(decided);
        s.apply(&again).expect("a valid delta");
    }
    s
}

/// A frame's pixels, drawn on one thread.
fn pixels(s: &Stream, width: u32, height: u32) -> Vec<[u8; 4]> {
    let rgba = s.draw(width, height, 1).expect("drawn");
    rgba.chunks(4).map(|p| [p[0], p[1], p[2], p[3]]).collect()
}

/// The colours a frame shows other than its background.
fn colours(s: &Stream, width: u32, height: u32) -> Vec<[u8; 4]> {
    let mut found: Vec<[u8; 4]> = pixels(s, width, height)
        .into_iter()
        .filter(|p| *p != [0, 0, 0, 0])
        .collect();
    found.sort();
    found.dedup();
    found
}

/// Shaded text is lit as its plane, turned to the eye, whatever each glyph's first corner turns
/// and whatever way it winds: one colour over every glyph, a single contour or several.
#[test]
fn shaded_text_is_lit_flat_as_its_plane_turned_to_the_eye() {
    let single = vec![vec![
        [-1.0, -1.0],
        [-0.4, -1.0],
        [-0.4, 0.6],
        [0.8, 0.6],
        [0.8, 1.0],
        [-1.0, 1.0],
    ]];
    let multiple = vec![
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        vec![[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
    ];
    for (name, contours) in [("single", single), ("multiple", multiple)] {
        let reversed: Vec<Vec<[f64; 2]>> = contours
            .iter()
            .map(|c| c.iter().rev().copied().collect())
            .collect();
        let (a, b) = (tipped(&glyph(&contours)), tipped(&glyph(&reversed)));
        // The writer carries the plane's normal at both lit corners.
        let [n0, n1] = soft_decided_normals(&a, 0.0);
        assert_eq!(n0, n1, "{name}");
        let plane = unit([0.0, -libm::sin(1.1), libm::cos(1.1)]);
        assert!((dot(n0, plane).abs() - 1.0).abs() < 1e-12, "{name}: {n0:?}");
        // The two stops it is painted with are one colour, the plane lit facing the eye.
        let view = soft_view();
        let Camera::ThreeD(c) = &view.camera else {
            unreachable!()
        };
        let light = c.light_source;
        let eye = [0.0, 0.0, 30.0];
        let stops = soft_face_stops(&a, &[[0.9, 0.3, 0.2, 1.0]], &SOFT_DIFFUSE, light, eye, [n0, n1]);
        assert_eq!(stops[0], stops[1], "{name}");
        // Drawn, every pixel of the glyph's interior is one colour, and the glyph wound the
        // other way draws the same frame.
        for normals in [true, false] {
            let one = colours(&glyph_stream(view.clone(), a.clone(), normals), 192, 108);
            let other = colours(&glyph_stream(view.clone(), b.clone(), normals), 192, 108);
            let opaque: Vec<_> = one.iter().filter(|p| p[3] == 255).collect();
            assert_eq!(opaque.len(), 1, "{name}: {opaque:?}");
            assert_eq!(one, other, "{name}: one frame whatever the winding");
        }
    }
}

/// A face group CE's light would refuse no longer is: the soft light draws it, and a sphere's
/// faces run smoothly into one another, no face standing apart from its neighbours. Its faces are
/// stroked in their own colour, the stroke lit as its face is, so the frame shows the light alone:
/// unstroked, the pixels a seam between two faces crosses composite both faces by their areas
/// over what lies behind, as Cairo's do.
#[test]
fn a_soft_lit_sphere_is_smooth_across_its_faces() {
    let (nu, nv) = (32, 16);
    let (points, uv) = sphere(nu, nv, 3.5);
    let shape = Shape::new(ShapeData::Path { points });
    let fills = Stops::new(vec![[0.2, 0.5, 0.9, 1.0]; nu * nv]);
    let mut r = Record::new(RecordKind::FaceGroup, 1, 101);
    r.geometry.shape_a = Some(shape.key());
    r.depth.space = Space::World;
    r.depth.shade_in_3d = true;
    r.paint.stroke_width = 15.0;
    let mut group = faces(uv, [nu as u32, nv as u32], fills.key());
    group.stroke = Some(fills.key());
    r.body = Body::FaceGroup(group);
    let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
    d.view = Some(soft_view());
    d.upload = vec![shape];
    d.upload_paints = vec![fills];
    d.order = Some(
        (0..(nu * nv) as u32)
            .map(|part| OrderEntry {
                layer: Layer::Moving,
                slot: 1,
                part,
            })
            .collect(),
    );
    d.set = vec![r];
    let mut s = Stream::new();
    s.apply(&d).expect("a valid scene");
    let (w, h) = (192, 108);
    let px = pixels(&s, w, h);
    let opaque = |x: u32, y: u32| px[(y * w + x) as usize][3] == 255;
    // Pixels well inside the sphere: its silhouette, where a face turned from the eye shows at
    // the rim, is not the light's to judge.
    const IN: u32 = 5;
    let inside = |x: u32, y: u32| {
        (x - IN..=x + IN).all(|i| (y - IN..=y + IN).all(|j| opaque(i, j)))
    };
    let (mut pairs, mut worst) = (0, 0);
    for y in IN..h - IN {
        for x in IN..w - IN - 1 {
            if !(inside(x, y) && inside(x + 1, y)) {
                continue;
            }
            let (a, b) = (px[(y * w + x) as usize], px[(y * w + x + 1) as usize]);
            let step = (0..3)
                .map(|k| (a[k] as i32 - b[k] as i32).abs())
                .max()
                .unwrap();
            worst = worst.max(step);
            pairs += 1;
        }
    }
    assert!(pairs > 2000, "{pairs} pairs, the worst a step of {worst}");
    assert!(worst <= 3, "a step of {worst} levels between neighbouring pixels");
    // The carried normals draw the frame the stream found itself.
    let found = s.corner_normals(&[1]).unwrap()[&1].clone();
    let mut carried = d.clone();
    carried.set[0].corner_normals = Some(found);
    let mut t = Stream::new();
    t.apply(&carried).expect("a valid scene");
    assert_eq!(pixels(&t, w, h), px);
}

/// The face's share of light runs across it, corner to corner, and on into its neighbours: on a
/// shared edge both faces read the same share.
#[test]
fn neighbouring_faces_share_the_light_on_their_shared_edge() {
    let a = FaceShares::new(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        [0.4, 0.6, 0.9, 0.5],
    );
    let b = FaceShares::new(
        [[10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0]],
        [0.6, 0.7, 0.8, 0.9],
    );
    for t in 0..=10 {
        let y = t as f64;
        assert!((a.at(10.0, y) - b.at(10.0, y)).abs() < 1e-12, "at y {y}");
    }
    assert!((a.at(0.0, 0.0) - 0.4).abs() < 1e-12);
    assert!((a.at(10.0, 10.0) - 0.9).abs() < 1e-12);
    assert_eq!(FaceShares::constant([0.5; 4]), Some(0.5));
    assert_eq!(FaceShares::constant([0.5, 0.5, 0.5, 0.6]), None);
}

fn sphere_delta(view: View) -> Delta {
    let (nu, nv) = (8, 4);
    let (points, uv) = sphere(nu, nv, 1.0);
    let shape = Shape::new(ShapeData::Path { points });
    let fills = Stops::new(vec![[0.2, 0.5, 0.9, 1.0]; nu * nv]);
    let mut r = Record::new(RecordKind::FaceGroup, 1, 101);
    r.geometry.shape_a = Some(shape.key());
    r.depth.space = Space::World;
    r.depth.shade_in_3d = true;
    r.body = Body::FaceGroup(faces(uv, [nu as u32, nv as u32], fills.key()));
    let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
    d.view = Some(view);
    d.upload = vec![shape];
    d.upload_paints = vec![fills];
    d.set = vec![r];
    d.order = Some(
        (0..(nu * nv) as u32)
            .map(|part| OrderEntry {
                layer: Layer::Moving,
                slot: 1,
                part,
            })
            .collect(),
    );
    d
}

/// Under CE's light the stream decides CE's normals and no corner normals, so a writer carries
/// nothing XF 1.6 holds, and its deltas stay at the minor they were.
#[test]
fn under_ces_light_nothing_of_the_soft_light_is_decided_or_written() {
    let d = sphere_delta(common::three_d_view(192, 108));
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    assert!(s.corner_normals(&[1]).unwrap().is_empty());
    let normals = s.normals(&[1], 1e-6).unwrap()[&1].clone();
    let ShapeData::Path { points } = d.upload[0].data() else {
        unreachable!()
    };
    let want: Vec<V3> = points
        .chunks(16)
        .flat_map(|face| decided_normals(face, 1e-6))
        .collect();
    assert_eq!(normals, want);
    let mut carried = d.clone();
    carried.set[0].normals = Some(normals);
    let bytes = encode_delta(&carried);
    assert_eq!(decode_delta(&bytes).unwrap().version, (1, 2));

    // Under the soft light a face group carries corner normals, not CE's.
    let mut s = Stream::new();
    s.apply(&sphere_delta(soft_view())).unwrap();
    assert!(s.normals(&[1], 1e-6).unwrap().is_empty());
    assert_eq!(s.corner_normals(&[1]).unwrap()[&1].len(), 4 * 32);
}

/// XF 1.6 carries a face group's corner normals in a section after XF 1.5's, written empty where
/// only XF 1.6's is carried: an XF 1.5 reader, which stops after its own section, reads the delta
/// as it was without them.
#[test]
fn corner_normals_are_xf_1_6_and_an_older_reader_skips_them() {
    let d = sphere_delta(soft_view());
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    let normals = s.corner_normals(&[1]).unwrap()[&1].clone();
    let mut carried = d.clone();
    carried.set[0].corner_normals = Some(normals.clone());
    let bytes = encode_delta(&carried);
    let back = decode_delta(&bytes).unwrap();
    assert_eq!(back.version, (1, 6));
    assert_eq!(back.set[0].corner_normals.as_deref(), Some(&normals[..]));
    // Without its section the delta is the one that carries none (at the version it states).
    let section = 4 + 4 + 4 + 24 * normals.len();
    let older = decode_delta(&bytes[..bytes.len() - section]).unwrap();
    let mut without = back.clone();
    without.set[0].corner_normals = None;
    assert_eq!(older.set, without.set);
    assert_eq!(encode_delta(&d).len(), bytes.len() - section - 8);
    // Re-encoded, the delta is the same bytes.
    assert_eq!(encode_delta(&back), bytes);
}

#[test]
fn corner_normals_are_refused_where_they_do_not_describe_the_faces() {
    let d = sphere_delta(soft_view());
    // Four per face.
    let mut short = d.clone();
    short.set[0].corner_normals = Some(vec![[0.0, 0.0, 1.0]; 4 * 32 - 1]);
    let err = Stream::new().apply(&short).unwrap_err().to_string();
    assert!(err.contains("corner normals"), "{err}");
    // Finite.
    let mut infinite = d.clone();
    let mut normals = vec![[0.0, 0.0, 1.0]; 4 * 32];
    normals[3] = [f64::NAN, 0.0, 0.0];
    infinite.set[0].corner_normals = Some(normals);
    assert!(Stream::new().apply(&infinite).is_err());
    // On a face group alone.
    let mut path = d.clone();
    path.set[0].kind = RecordKind::Path;
    path.set[0].body = Body::None;
    path.set[0].corner_normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
    let err = Stream::new().apply(&path).unwrap_err().to_string();
    assert!(err.contains("corner normals on a path"), "{err}");
}

#[test]
fn the_corner_normals_section_names_each_slot_the_delta_sets_once() {
    let d = sphere_delta(soft_view());
    let mut carried = d.clone();
    carried.set[0].corner_normals = Some(vec![[0.0, 0.0, 1.0]; 4 * 32]);
    let bytes = encode_delta(&carried);
    let rows_at = bytes.len() - (4 + 4 + 4 + 24 * 4 * 32);
    // A row for a slot the delta does not set.
    let mut other = bytes.clone();
    other[rows_at + 4..rows_at + 8].copy_from_slice(&7u32.to_le_bytes());
    let err = decode_delta(&other).unwrap_err().to_string();
    assert!(err.contains("slot_not_set"), "{err}");
    // A row of no normals.
    let mut empty = bytes[..rows_at + 8].to_vec();
    empty.extend(0u32.to_le_bytes());
    let err = decode_delta(&empty).unwrap_err().to_string();
    assert!(err.contains("row_empty"), "{err}");
    // A slot named twice.
    let row = bytes[rows_at + 4..].to_vec();
    let mut twice = bytes[..rows_at].to_vec();
    twice.extend(2u32.to_le_bytes());
    twice.extend(&row);
    twice.extend(&row);
    let err = decode_delta(&twice).unwrap_err().to_string();
    assert!(err.contains("slot_named_twice"), "{err}");
}
