//! The triangle rasterizer: `mesh` records lowered to screen triangles, and those rasterized into
//! mask fragments with depth.
//!
//! - Placement: a record's points through its placement, then as CE places a mobject's points for
//!   display (`project::pre_display`): projected under a 3D camera when the record is in world
//!   space (moved by its centre's projection when it has a fixed orientation), kept in frame
//!   otherwise; then through CE's frame-to-pixel mapping.
//! - Depth: in world space under a 3D camera, the projection's factor, negated, which is affine
//!   across a projected triangle (for CE's default projection; under the exponential option, or
//!   with a corner at or behind the eye, the plane through the corners approximates it), so a plane
//!   per triangle is exact at every sub-sample; a fixed orientation record is a camera-facing plane
//!   at its centre's depth. Anything else is in front of the scene, and ties go by draw order, as
//!   under a flat camera, where everything ties.
//! - Colour: the fill, by the record's group opacity; lit when the record is `shade_in_3d` and the
//!   3D camera applies shading, as `shade` lights a triangle's corners and interpolates them.
//! - Coverage: corners snapped to [`SUBPIXEL`] units, sub-samples tested against exact integer edge
//!   functions with a tie rule that gives a sample on an edge shared by two triangles to exactly
//!   one of them, so a surface's internal edges neither gap nor overlap. Triangles reaching past a
//!   guard band far outside the frame are clipped to it first, with clip points computed the same
//!   way from either side of a shared edge.

use crate::composite::{
    Coverage, FULL, Fragment, OpaquePass, Plane, Producer, Rect, SAMPLES, SUBPIXEL,
    TransparentPass, premultiply, sample_fixed,
};
use crate::project::{FrameMap, Placement, Projection, pre_display};
use crate::shade::{Light, TriangleColour};
use crate::tiles::{TILE, tile_index, tiles_across};
use crate::xf::*;

/// How far outside the frame, in pixels, a triangle may reach before it is clipped.
const GUARD: f64 = 65536.0;

/// One mesh record to draw, with its shape's arrays and its place in the draw order.
pub struct MeshDraw<'a> {
    pub record: &'a Record,
    pub positions: &'a [[f64; 3]],
    pub triangles: &'a [[u32; 3]],
    pub normals: Option<&'a [[f64; 3]]>,
    pub order: u32,
}

/// A triangle on the pixel plane, ready to rasterize.
#[derive(Clone, Debug)]
struct Tri {
    /// Edge functions `e(p) = a * p.y + b * p.x + c` in sub-pixel units, positive inside, and the
    /// bias that excludes a sample exactly on the edge unless the tie rule gives it this triangle.
    edges: [[i64; 4]; 3],
    /// Pixels `x0..x1` by `y0..y1`, inside the frame.
    bbox: [u32; 4],
    depth: Plane,
    colour: TriangleColour,
    opaque: bool,
    order: u32,
    sub: u32,
}

/// Mesh records lowered to triangles, binned by tile: a producer of mask fragments.
pub struct Meshes {
    tris: Vec<Tri>,
    bins: Vec<Vec<u32>>,
    width: u32,
}

fn transform(m: &Affine, p: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3])
}

/// A normal through a placement: by the cofactors of its linear part, normalized.
fn transform_normal(m: &Affine, n: [f64; 3]) -> [f64; 3] {
    let c = |i: usize, j: usize| {
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
        m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]
    };
    let out: [f64; 3] = std::array::from_fn(|i| c(i, 0) * n[0] + c(i, 1) * n[1] + c(i, 2) * n[2]);
    let norm = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt();
    if norm > 0.0 {
        out.map(|v| v / norm)
    } else {
        out
    }
}

/// Lowers mesh records to triangles for a frame of `width x height` pixels.
pub fn lower(view: &View, width: u32, height: u32, draws: &[MeshDraw<'_>]) -> Meshes {
    let map = FrameMap::cairo(
        width,
        height,
        view.frame_width,
        view.frame_height,
        view.frame_center,
    );
    let camera = match &view.camera {
        Camera::ThreeD(c) => Some(c),
        _ => None,
    };
    let projection = camera.map(|c| Projection::new(c, view.frame_center));
    let mut meshes = Meshes {
        tris: Vec::new(),
        bins: vec![Vec::new(); (tiles_across(width) * height.div_ceil(TILE)) as usize],
        width,
    };
    for draw in draws {
        let rec = draw.record;
        let Paint::One(fill) = rec.paint.fill else {
            continue; // No fill: nothing to draw (other paints are refused before drawing).
        };
        let world: Vec<[f64; 3]> = draw
            .positions
            .iter()
            .map(|p| transform(&rec.geometry.place_a, *p))
            .collect();
        if world.iter().flatten().any(|v| !v.is_finite()) {
            continue; // CE draws such a mobject as one point: nothing.
        }
        let (placement, keyed) = match (camera, rec.depth.space) {
            (Some(_), Space::World) => match rec.depth.fixed_orientation_centre {
                Some(centre) => {
                    let p = projection.as_ref().unwrap();
                    (
                        Placement::FixedOrientation(centre),
                        Some(-p.factor(p.project(centre)[2])),
                    )
                }
                None => (Placement::Projected, None),
            },
            (Some(_), Space::Overlay) => (Placement::FixedInFrame, Some(f64::NEG_INFINITY)),
            _ => (Placement::Flat, Some(f64::NEG_INFINITY)),
        };
        let shown = pre_display(&world, projection.as_ref(), placement);
        let corners: Vec<([f64; 2], f64)> = shown
            .iter()
            .map(|p| {
                let key = keyed.unwrap_or_else(|| -projection.as_ref().unwrap().factor(p[2]));
                (map.apply(*p), key)
            })
            .collect();
        let alpha = fill[3] * rec.opacity;
        let rgb = [fill[0], fill[1], fill[2]];
        let light = match camera {
            Some(c) if rec.depth.shade_in_3d && c.should_apply_shading => Some(Light::new(
                &view.light,
                c,
                view.frame_center,
                projection.as_ref().unwrap().rotation(),
            )),
            _ => None,
        };
        let normals: Option<Vec<[f64; 3]>> = draw.normals.map(|ns| {
            ns.iter()
                .map(|n| transform_normal(&rec.geometry.place_a, *n))
                .collect()
        });
        let opaque = premultiply([0.0, 0.0, 0.0, alpha])[3] == 255;
        for (sub, t) in draw.triangles.iter().enumerate() {
            let idx = t.map(|i| i as usize);
            let px = idx.map(|i| corners[i].0);
            let depth =
                Plane::through(idx.map(|i| [corners[i].0[0], corners[i].0[1], corners[i].1]));
            let colour = match &light {
                None => TriangleColour::flat(rgb, alpha),
                Some(light) => light.triangle(
                    rgb,
                    alpha,
                    idx.map(|i| world[i]),
                    normals.as_ref().map(|ns| idx.map(|i| ns[i])),
                    px,
                ),
            };
            meshes.push(px, depth, colour, opaque, draw.order, sub as u32, height);
        }
    }
    meshes
}

/// The intersection of segment `p`-`q` with the line where coordinate `axis` is `at`, computed
/// from the segment's endpoints in one order whichever way it is walked.
fn cut(p: [f64; 2], q: [f64; 2], axis: usize, at: f64) -> [f64; 2] {
    let (a, b) = if (p[0], p[1]) <= (q[0], q[1]) {
        (p, q)
    } else {
        (q, p)
    };
    let t = (at - a[axis]) / (b[axis] - a[axis]);
    let mut out = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    out[axis] = at;
    out
}

/// A polygon clipped to the guard band.
fn clip_to_guard(poly: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    let mut poly = poly;
    for (axis, at, keep_below) in [
        (0, -GUARD, false),
        (0, GUARD, true),
        (1, -GUARD, false),
        (1, GUARD, true),
    ] {
        let inside = |p: [f64; 2]| {
            if keep_below {
                p[axis] <= at
            } else {
                p[axis] >= at
            }
        };
        let mut out = Vec::with_capacity(poly.len() + 2);
        for i in 0..poly.len() {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            if inside(p) {
                out.push(p);
            }
            if inside(p) != inside(q) {
                out.push(cut(p, q, axis, at));
            }
        }
        poly = out;
        if poly.is_empty() {
            break;
        }
    }
    poly
}

fn snap(p: [f64; 2]) -> [i64; 2] {
    p.map(|v| (v * SUBPIXEL as f64).round() as i64)
}

impl Meshes {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        px: [[f64; 2]; 3],
        depth: Plane,
        colour: TriangleColour,
        opaque: bool,
        order: u32,
        sub: u32,
        height: u32,
    ) {
        if px.iter().flatten().any(|v| !v.is_finite()) {
            return;
        }
        let inside_guard = px.iter().flatten().all(|v| v.abs() <= GUARD);
        let pieces = if inside_guard {
            vec![px]
        } else {
            let poly = clip_to_guard(px.to_vec());
            (1..poly.len().saturating_sub(1))
                .map(|i| [poly[0], poly[i], poly[i + 1]])
                .collect()
        };
        for piece in pieces {
            let mut v = piece.map(snap);
            let cross = |v: &[[i64; 2]; 3]| {
                (v[1][0] - v[0][0]) * (v[2][1] - v[0][1])
                    - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0])
            };
            match cross(&v).signum() {
                0 => continue,
                -1 => v.swap(1, 2),
                _ => {}
            }
            let edges = [0, 1, 2].map(|i| {
                let (a, b) = (v[i], v[(i + 1) % 3]);
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                // e(p) = dx * (p.y - a.y) - dy * (p.x - a.x); a sample on the edge belongs to the
                // triangle when the edge runs up, or right along a row.
                let owns_ties = dy < 0 || (dy == 0 && dx > 0);
                [
                    dx,
                    -dy,
                    dy * a[0] - dx * a[1],
                    if owns_ties { 0 } else { -1 },
                ]
            });
            let lo = |k: usize| v.iter().map(|p| p[k]).min().unwrap();
            let hi = |k: usize| v.iter().map(|p| p[k]).max().unwrap();
            let clamp_px =
                |s: i64, limit: u32| s.div_euclid(SUBPIXEL).clamp(0, limit as i64) as u32;
            let bbox = [
                clamp_px(lo(0), self.width),
                clamp_px(lo(1), height),
                clamp_px(hi(0) + SUBPIXEL, self.width),
                clamp_px(hi(1) + SUBPIXEL, height),
            ];
            if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
                continue;
            }
            let index = self.tris.len() as u32;
            self.tris.push(Tri {
                edges,
                bbox,
                depth,
                colour,
                opaque,
                order,
                sub,
            });
            for ty in bbox[1] / TILE..=(bbox[3] - 1) / TILE {
                for tx in bbox[0] / TILE..=(bbox[2] - 1) / TILE {
                    self.bins[tile_index(self.width, tx * TILE, ty * TILE)].push(index);
                }
            }
        }
    }

    /// Rasterizes the opaque or the other triangles of the tile at `rect`.
    fn raster(&self, rect: Rect, opaque: bool, mut add: impl FnMut(u32, u32, Fragment)) {
        let offsets: [[i64; 2]; SAMPLES] = std::array::from_fn(sample_fixed);
        for &i in &self.bins[tile_index(self.width, rect.x0, rect.y0)] {
            let t = &self.tris[i as usize];
            if t.opaque != opaque {
                continue;
            }
            // Per edge: the offset of each sub-sample from the pixel's corner, and their range.
            let off: [[i64; SAMPLES]; 3] = t
                .edges
                .map(|e| offsets.map(|[sx, sy]| e[0] * sy + e[1] * sx));
            let range = off.map(|o| (*o.iter().min().unwrap(), *o.iter().max().unwrap()));
            for y in t.bbox[1].max(rect.y0)..t.bbox[3].min(rect.y1) {
                for x in t.bbox[0].max(rect.x0)..t.bbox[2].min(rect.x1) {
                    let (cx, cy) = (x as i64 * SUBPIXEL, y as i64 * SUBPIXEL);
                    let corner = t.edges.map(|e| e[0] * cy + e[1] * cx + e[2] + e[3]);
                    if (0..3).any(|k| corner[k] + range[k].1 < 0) {
                        continue;
                    }
                    let mask = if (0..3).all(|k| corner[k] + range[k].0 >= 0) {
                        FULL
                    } else {
                        (0..SAMPLES).fold(0u16, |m, s| {
                            if (0..3).all(|k| corner[k] + off[k][s] >= 0) {
                                m | 1 << s
                            } else {
                                m
                            }
                        })
                    };
                    if mask == 0 {
                        continue;
                    }
                    let colour = t.colour.at(x, y, mask);
                    add(
                        x,
                        y,
                        Fragment::new(t.depth, colour, Coverage::Mask(mask), t.order, t.sub),
                    );
                }
            }
        }
    }
}

impl Producer for Meshes {
    fn opaque(&self, pass: &mut OpaquePass<'_>) {
        self.raster(pass.rect(), true, |x, y, f| pass.add(x, y, f));
    }

    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        self.raster(pass.rect(), false, |x, y, f| pass.add(x, y, f));
    }
}
