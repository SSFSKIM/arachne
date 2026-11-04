//! Engine-authored mesh scenes, and oracles that judge drawn pixels against exact geometry.

use super::{flat_view, three_d_view};
use arachne_core::project::{FrameMap, Placement, Projection, pre_display};
use arachne_core::stream::Stream;
use arachne_core::xf::*;

/// A mesh over the unit square of (u, v) with `nu x nv` cells, two triangles each, its vertices
/// placed by `f`.
pub fn grid_mesh(nu: usize, nv: usize, f: impl Fn(f64, f64) -> [f64; 3]) -> Shape {
    let mut positions = Vec::new();
    for j in 0..=nv {
        for i in 0..=nu {
            positions.push(f(i as f64 / nu as f64, j as f64 / nv as f64));
        }
    }
    let at = |i: usize, j: usize| (j * (nu + 1) + i) as u32;
    let mut triangles = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            triangles.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
            triangles.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    Shape::new(ShapeData::Mesh {
        positions,
        triangles,
        normals: None,
    })
}

/// A flat quad from its four corners (two triangles, split along corners 0-2).
pub fn quad(corners: [[f64; 3]; 4]) -> Shape {
    Shape::new(ShapeData::Mesh {
        positions: corners.to_vec(),
        triangles: vec![[0, 1, 2], [0, 2, 3]],
        normals: None,
    })
}

/// A UV sphere with per-vertex normals.
pub fn sphere(centre: [f64; 3], radius: f64, rings: usize, segments: usize) -> Shape {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for j in 0..=rings {
        let phi = std::f64::consts::PI * j as f64 / rings as f64;
        for i in 0..=segments {
            let theta = std::f64::consts::TAU * i as f64 / segments as f64;
            let n = [
                libm::sin(phi) * libm::cos(theta),
                libm::sin(phi) * libm::sin(theta),
                libm::cos(phi),
            ];
            positions.push([
                centre[0] + radius * n[0],
                centre[1] + radius * n[1],
                centre[2] + radius * n[2],
            ]);
            normals.push(n);
        }
    }
    let at = |i: usize, j: usize| (j * (segments + 1) + i) as u32;
    let mut triangles = Vec::new();
    for j in 0..rings {
        for i in 0..segments {
            triangles.push([at(i, j), at(i, j + 1), at(i + 1, j + 1)]);
            triangles.push([at(i, j), at(i + 1, j + 1), at(i + 1, j)]);
        }
    }
    Shape::new(ShapeData::Mesh {
        positions,
        triangles,
        normals: Some(normals),
    })
}

/// A mesh record filled with one colour, placed in `space`.
pub fn mesh_record(slot: u32, shape: &Shape, rgba: [f64; 4], space: Space) -> Record {
    let mut r = Record::new(RecordKind::Mesh, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key);
    r.paint.fill = Paint::One(rgba);
    r.depth.space = space;
    r
}

/// A stream holding `view` and the records, drawn in the order of `order` (slots).
pub fn stream_of(view: View, items: &[(Shape, Record)], order: &[u32]) -> Stream {
    let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
    d.view = Some(view);
    for (shape, record) in items {
        if !d.upload.iter().any(|s| s.key == shape.key) {
            d.upload.push(shape.clone());
        }
        d.set.push(record.clone());
    }
    d.order = Some(
        order
            .iter()
            .map(|slot| OrderEntry {
                layer: Layer::Moving,
                slot: *slot,
                part: 0,
            })
            .collect(),
    );
    let mut s = Stream::new();
    s.apply(&d).expect("a valid scene");
    s
}

/// Scenes the identity example and the determinism tests draw: a lit sphere crossed by a
/// translucent sheet, a translucent sphere in soft light, two crossing quads over a rippled sheet,
/// and overlapping translucent meshes under a flat camera.
pub fn scenes(width: u32, height: u32) -> Vec<(&'static str, Stream)> {
    let ball = sphere([0.3, -0.2, 0.4], 2.1, 14, 28);
    let sheet = grid_mesh(9, 7, |u, v| {
        [
            -4.0 + 8.0 * u,
            -3.0 + 6.0 * v,
            0.3 + 0.8 * (u - 0.5) + 0.4 * (v - 0.5),
        ]
    });
    let ripple = grid_mesh(30, 20, |u, v| {
        let (x, y) = (-5.0 + 10.0 * u, -3.5 + 7.0 * v);
        [x, y, -1.5 + 0.35 * libm::sin(1.7 * x) * libm::cos(1.3 * y)]
    });
    let flat_a = quad([
        [-1.5, -1.5, 0.0],
        [1.5, -1.5, 0.0],
        [1.5, 1.5, 0.0],
        [-1.5, 1.5, 0.0],
    ]);
    let flat_b = quad([
        [0.2, -2.0, 0.0],
        [3.0, -1.0, 0.0],
        [2.5, 2.2, 0.0],
        [-0.5, 1.0, 0.0],
    ]);
    let crossing = quad([
        [-3.0, -3.0, -1.5],
        [3.0, -3.0, -1.5],
        [3.0, 3.0, 1.5],
        [-3.0, 3.0, 1.5],
    ]);
    let level = quad([
        [-3.0, -3.0, 0.0],
        [3.0, -3.0, 0.0],
        [3.0, 3.0, 0.0],
        [-3.0, 3.0, 0.0],
    ]);

    let lit = |mut r: Record| {
        r.depth.shade_in_3d = true;
        r
    };
    let mut soft = three_d_view(width, height);
    soft.light = LightModel::SoftDiffuse {
        ambient: 0.25,
        wrap: 0.5,
    };
    soft.background = Background::Color([0.95, 0.95, 0.9, 1.0]);
    vec![
        (
            "sphere-ce",
            stream_of(
                three_d_view(width, height),
                &[
                    (
                        ball.clone(),
                        lit(mesh_record(1, &ball, [0.2, 0.5, 0.9, 1.0], Space::World)),
                    ),
                    (
                        sheet.clone(),
                        mesh_record(2, &sheet, [0.9, 0.6, 0.1, 0.5], Space::World),
                    ),
                ],
                &[2, 1],
            ),
        ),
        (
            "sphere-soft",
            stream_of(
                soft,
                &[(
                    ball.clone(),
                    lit(mesh_record(1, &ball, [0.8, 0.3, 0.3, 0.6], Space::World)),
                )],
                &[1],
            ),
        ),
        (
            "crossing",
            stream_of(
                three_d_view(width, height),
                &[
                    (
                        ripple.clone(),
                        mesh_record(1, &ripple, [0.3, 0.8, 0.4, 1.0], Space::World),
                    ),
                    (
                        crossing.clone(),
                        mesh_record(2, &crossing, [0.9, 0.2, 0.2, 1.0], Space::World),
                    ),
                    (
                        level.clone(),
                        mesh_record(3, &level, [0.2, 0.3, 0.9, 0.7], Space::World),
                    ),
                ],
                &[3, 2, 1],
            ),
        ),
        (
            "flat",
            stream_of(
                flat_view(width, height, [0.1, 0.1, 0.1, 1.0]),
                &[
                    (
                        flat_a.clone(),
                        mesh_record(1, &flat_a, [0.9, 0.9, 0.2, 1.0], Space::Screen),
                    ),
                    (
                        flat_b.clone(),
                        mesh_record(2, &flat_b, [0.2, 0.7, 0.9, 0.5], Space::Screen),
                    ),
                ],
                &[1, 2],
            ),
        ),
    ]
}

/// Where a view puts a world point on the pixel plane, and its depth key there (the projection's
/// factor, negated: smaller is nearer, and affine across a projected triangle).
pub fn to_pixels(view: &View, width: u32, height: u32, p: [f64; 3]) -> ([f64; 2], f64) {
    let map = FrameMap::cairo(
        width,
        height,
        view.frame_width,
        view.frame_height,
        view.frame_center,
    );
    match &view.camera {
        Camera::ThreeD(c) => {
            let projection = Projection::new(c, view.frame_center);
            let shown = pre_display(&[p], Some(&projection), Placement::Projected)[0];
            let d = c.focal_distance;
            let factor = if d - shown[2] < 0.0 {
                1e6
            } else {
                d / (d - shown[2])
            };
            (map.apply(shown), -factor)
        }
        _ => (map.apply(p), 0.0),
    }
}

/// The part of a convex polygon (pixel coordinates) with `a * x + b * y <= c`.
pub fn clip(poly: &[[f64; 2]], a: f64, b: f64, c: f64) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    for i in 0..poly.len() {
        let p = poly[i];
        let q = poly[(i + 1) % poly.len()];
        let (fp, fq) = (a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c);
        if fp <= 0.0 {
            out.push(p);
        }
        if (fp < 0.0 && fq > 0.0) || (fp > 0.0 && fq < 0.0) {
            let t = fp / (fp - fq);
            out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
        }
    }
    out
}

/// A polygon's area (either winding).
pub fn area(poly: &[[f64; 2]]) -> f64 {
    let mut s = 0.0;
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        s += p[0] * q[1] - q[0] * p[1];
    }
    (s / 2.0).abs()
}

/// The exact area of a convex polygon inside pixel `(x, y)`.
pub fn area_in_pixel(poly: &[[f64; 2]], x: u32, y: u32) -> f64 {
    let (x, y) = (x as f64, y as f64);
    let mut p = clip(poly, -1.0, 0.0, -x);
    p = clip(&p, 1.0, 0.0, x + 1.0);
    p = clip(&p, 0.0, -1.0, -y);
    p = clip(&p, 0.0, 1.0, y + 1.0);
    if p.len() < 3 { 0.0 } else { area(&p) }
}

/// Whether a point is inside a triangle (either winding), and its barycentric weights.
pub fn barycentric(t: [[f64; 2]; 3], p: [f64; 2]) -> Option<[f64; 3]> {
    let d = (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1]);
    if d == 0.0 {
        return None;
    }
    let w1 = ((p[0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (p[1] - t[0][1])) / d;
    let w2 = ((t[1][0] - t[0][0]) * (p[1] - t[0][1]) - (p[0] - t[0][0]) * (t[1][1] - t[0][1])) / d;
    let w0 = 1.0 - w1 - w2;
    (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0).then_some([w0, w1, w2])
}

/// The distance from a point to a segment.
pub fn distance_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
    };
    let (ex, ey) = (a[0] + dx * t - p[0], a[1] + dy * t - p[1]);
    (ex * ex + ey * ey).sqrt()
}

/// One opaque mesh's projected triangles: pixel positions and depth keys per corner.
pub struct Projected {
    pub triangles: Vec<([[f64; 2]; 3], [f64; 3])>,
}

pub fn project_mesh(view: &View, width: u32, height: u32, shape: &Shape) -> Projected {
    let ShapeData::Mesh {
        positions,
        triangles,
        ..
    } = &shape.data
    else {
        panic!("a mesh shape")
    };
    let corners: Vec<([f64; 2], f64)> = positions
        .iter()
        .map(|p| to_pixels(view, width, height, *p))
        .collect();
    Projected {
        triangles: triangles
            .iter()
            .map(|t| {
                let c = t.map(|i| corners[i as usize]);
                ([c[0].0, c[1].0, c[2].0], [c[0].1, c[1].1, c[2].1])
            })
            .collect(),
    }
}

/// Which of `meshes` is nearest at a point (later in `meshes` on a tie), if any covers it.
pub fn front_at(meshes: &[(Projected, [u8; 4])], p: [f64; 2]) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for (m, (mesh, _)) in meshes.iter().enumerate() {
        for (t, keys) in &mesh.triangles {
            if let Some(w) = barycentric(*t, p) {
                let key = w[0] * keys[0] + w[1] * keys[1] + w[2] * keys[2];
                if best.is_none_or(|(k, _)| key <= k) {
                    best = Some((key, m));
                }
            }
        }
    }
    best.map(|(_, m)| m)
}

/// `n x n` points spread evenly over the box `x0..x1` by `y0..y1`.
pub fn grid_points(x0: f64, y0: f64, x1: f64, y1: f64, n: usize) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            out.push([
                x0 + (x1 - x0) * (i as f64 + 0.5) / n as f64,
                y0 + (y1 - y0) * (j as f64 + 0.5) / n as f64,
            ]);
        }
    }
    out
}

/// An oracle for opaque meshes: per pixel, `n x n` evenly spread samples each take the colour of
/// the nearest mesh covering them, or the background; the pixel is their mean, per channel, in
/// 0..255.
pub fn supersample_opaque(
    meshes: &[(Projected, [u8; 4])],
    background: [u8; 4],
    x: u32,
    y: u32,
    n: usize,
) -> [f64; 4] {
    let (x, y) = (x as f64, y as f64);
    let mut sum = [0.0; 4];
    for p in grid_points(x, y, x + 1.0, y + 1.0, n) {
        let c = front_at(meshes, p).map_or(background, |m| meshes[m].1);
        for k in 0..4 {
            sum[k] += c[k] as f64;
        }
    }
    sum.map(|s| s / (n * n) as f64)
}

/// The pixels whose centres lie inside a polygon (pixel coordinates) and more than `margin`
/// pixels from its outline.
pub fn pixels_inside(
    outline: &[[f64; 2]],
    margin: f64,
    width: u32,
    height: u32,
) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let c = [x as f64 + 0.5, y as f64 + 0.5];
            let n = outline.len();
            let far =
                (0..n).all(|i| distance_to_segment(c, outline[i], outline[(i + 1) % n]) > margin);
            let mut inside = false;
            for i in 0..n {
                let (a, b) = (outline[i], outline[(i + 1) % n]);
                if (a[1] > c[1]) != (b[1] > c[1])
                    && c[0] < a[0] + (c[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
                {
                    inside = !inside;
                }
            }
            if far && inside {
                out.push((x, y));
            }
        }
    }
    out
}

/// A view's pixel outline of world points.
pub fn outline(view: &View, width: u32, height: u32, points: &[[f64; 3]]) -> Vec<[f64; 2]> {
    points
        .iter()
        .map(|p| to_pixels(view, width, height, *p).0)
        .collect()
}
