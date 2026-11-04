//! The class probes: engine-authored scenes, as XF streams, in which a painter's order draws
//! pixels wrong, each with the pixels where it does, what painter order gives there and what the
//! core must.
//! They are the parity kit's fixtures (`python -m arachne_core.parity` draws them against the
//! kit's reference frames); `tests/probes.rs` asserts them.
//!
//! CE's 3D camera sorts the faces of 3D objects by the depth of their centres, draws them in that
//! order, then draws every ordinary object (curves, axes, text) on top of all of them. A probe's
//! order is its scene's draw order (the order of `add`), which is not always CE's: where it puts an
//! ordinary object before a surface (`fixed_label_over_surface`, `depth_rule_both_ways`), CE still
//! draws the object after the surface's faces. Each check names a pixel painter order draws wrong
//! (or, where the probe also shows the core keeping what painter order draws right, that pixel too)
//! with painter order's bytes there (`Check::ce`): for sheets, the sheets covering the pixel
//! composited in the probe's order; an ordinary object, on top. These are the painter's model, not
//! a CE render: CE sorts each face by its centre's depth, which can order two crossing sheets'
//! faces differently pixel by pixel. Checking them against real CE renders is a later milestone's.
//!
//! The expected pixels come from an oracle of their own: each flat sheet's depth at a pixel by
//! casting the pixel's sight line onto the sheet's plane, the sheets composited in depth order. A
//! check sits where nothing crosses its pixel, so one order holds across it.

use crate::composite::{blend, premultiply};
use crate::project::{FrameMap, Placement, Projection, pre_display};
use crate::xf::*;

/// A probe's frame size.
pub const WIDTH: u32 = 320;
pub const HEIGHT: u32 = 180;

/// What the core must draw at a check's pixel.
#[derive(Clone, Debug, PartialEq)]
pub enum Expect {
    /// These bytes.
    Bytes([u8; 4]),
    /// What the frame shows there without the record in this slot (the record is hidden).
    Hidden(u32),
}

/// A pixel where CE's painter order draws wrong (or right, where `ce` is what the core draws too).
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    /// The frame: the stream after `deltas[..=frame]`.
    pub frame: usize,
    pub x: u32,
    pub y: u32,
    pub core: Expect,
    /// The bytes painter order gives there (the probe's order, ordinary objects on top): the
    /// painter's model, not a CE render.
    pub ce: [u8; 4],
    pub what: &'static str,
}

/// One probe: a stream and its checks.
#[derive(Clone, Debug)]
pub struct Probe {
    pub name: &'static str,
    pub width: u32,
    pub height: u32,
    pub deltas: Vec<Delta>,
    pub checks: Vec<Check>,
}

/// Every probe, in a fixed order: the eight class probes, the seam probe, then the fixture of the
/// paint kinds the probes do not draw.
pub fn all() -> Vec<Probe> {
    vec![
        intersecting_surfaces(),
        curve_behind_surface(),
        translucent_over_opaque(),
        fixed_label_over_surface(),
        depth_rule_both_ways(),
        stroke_crossing_surface(),
        edge_on_surface(),
        orbit(8, WIDTH, HEIGHT),
        seam(),
        paint_kinds(),
    ]
}

// ---------------------------------------------------------------------------------------------
// Building scenes.

/// A 3D view of `width x height` pixels (CE's frame width; the height fitted to the aspect ratio),
/// with CE's focal distance and light, looking from polar angle `phi` and azimuth `theta`.
pub fn view(width: u32, height: u32, phi: f64, theta: f64) -> View {
    let frame_width = 14.222222222222221;
    View {
        pixel_width: width,
        pixel_height: height,
        frame_width,
        frame_height: frame_width / (width as f64 / height as f64),
        frame_center: [0.0; 3],
        mapping_centre: [0.0; 3],
        background: Background::Color([0.0, 0.0, 0.0, 1.0]),
        camera: Camera::ThreeD(ThreeDCamera {
            phi,
            theta,
            gamma: 0.0,
            focal_distance: 20.0,
            zoom: 1.0,
            exponential_projection: false,
            light_source: [-7.0, -9.0, 10.0],
            shading_factor: 0.2,
            should_apply_shading: true,
        }),
        cairo_line_width_multiple: 0.01,
        use_z_index: true,
        light: LightModel::Ce,
    }
}

const DEG: f64 = std::f64::consts::PI / 180.0;

/// A quad's surface as a mesh: its bilinear grid of `nu x nv` cells, two triangles each.
pub fn sheet(corners: [[f64; 3]; 4], nu: usize, nv: usize) -> Shape {
    let lerp = |a: [f64; 3], b: [f64; 3], t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
    grid(nu, nv, |u, v| {
        lerp(
            lerp(corners[0], corners[1], u),
            lerp(corners[3], corners[2], u),
            v,
        )
    })
}

/// A mesh over the unit square's `nu x nv` grid through `f`.
pub fn grid(nu: usize, nv: usize, f: impl Fn(f64, f64) -> [f64; 3]) -> Shape {
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

/// A UV sphere whose seam column and poles share their vertices, so it is watertight.
pub fn sphere(centre: [f64; 3], radius: f64, rings: usize, segments: usize) -> Shape {
    let tau = std::f64::consts::TAU;
    let point = |i: usize, j: usize| {
        let (theta, phi) = (
            tau * i as f64 / segments as f64,
            std::f64::consts::PI * j as f64 / rings as f64,
        );
        [
            centre[0] + radius * libm::sin(phi) * libm::cos(theta),
            centre[1] + radius * libm::sin(phi) * libm::sin(theta),
            centre[2] - radius * libm::cos(phi),
        ]
    };
    let mut positions = vec![point(0, 0)];
    for j in 1..rings {
        for i in 0..segments {
            positions.push(point(i, j));
        }
    }
    positions.push(point(0, rings));
    let top = positions.len() as u32 - 1;
    let at = |i: usize, j: usize| (1 + (j - 1) * segments + i % segments) as u32;
    let mut triangles = Vec::new();
    for i in 0..segments {
        triangles.push([0, at(i + 1, 1), at(i, 1)]);
        triangles.push([top, at(i, rings - 1), at(i + 1, rings - 1)]);
    }
    for j in 1..rings - 1 {
        for i in 0..segments {
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

/// A curve through `points` in CE's cubic layout, one straight cubic per step.
pub fn polyline(points: &[[f64; 3]]) -> Shape {
    let mut out = Vec::new();
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    Shape::new(ShapeData::Path { points: out })
}

/// A closed curve through `f` at `n` steps over `0..1`.
pub fn closed(n: usize, f: impl Fn(f64) -> [f64; 3]) -> Shape {
    let points: Vec<[f64; 3]> = (0..=n).map(|i| f((i % n) as f64 / n as f64)).collect();
    polyline(&points)
}

/// A mesh record in world space, filled with one colour.
pub fn mesh_record(slot: u32, shape: &Shape, rgba: [f64; 4], shade: bool) -> Record {
    let mut r = Record::new(RecordKind::Mesh, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
    r.paint.fill = Paint::One(rgba);
    r.depth.space = Space::World;
    r.depth.shade_in_3d = shade;
    r
}

/// A stroked path record in world space (CE's units of width).
pub fn stroke_record(slot: u32, shape: &Shape, rgba: [f64; 4], width: f64) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
    r.paint.stroke = Paint::One(rgba);
    r.paint.stroke_width = width;
    r.depth.space = Space::World;
    r
}

/// A filled path record.
pub fn fill_record(slot: u32, shape: &Shape, rgba: [f64; 4], space: Space) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
    r.paint.fill = Paint::One(rgba);
    r.depth.space = space;
    r
}

/// The first delta of a stream: the view, every shape, every record, the order (slots in order).
pub fn first_delta(view: View, items: &[(Shape, Record)], order: &[u32]) -> Delta {
    let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
    d.view = Some(view);
    for (shape, record) in items {
        if !d.upload.iter().any(|s| s.key() == shape.key()) {
            d.upload.push(shape.clone());
        }
        d.objects.push(ObjectInfo {
            object: record.object,
            stable_id: format!("probe-{}", record.slot),
            source: "probes.rs:0".into(),
        });
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
    d
}

// ---------------------------------------------------------------------------------------------
// The oracle.

fn camera_of(view: &View) -> &ThreeDCamera {
    match &view.camera {
        Camera::ThreeD(c) => c,
        _ => unreachable!("probes are 3D"),
    }
}

fn map_of(view: &View, width: u32, height: u32) -> FrameMap {
    FrameMap::cairo(
        width,
        height,
        view.frame_width,
        view.frame_height,
        view.frame_center,
    )
}

/// Where a world point lands on the pixel plane, projected.
pub fn screen_of(view: &View, width: u32, height: u32, p: [f64; 3]) -> [f64; 2] {
    let projection = Projection::new(camera_of(view), view.frame_center);
    let shown = pre_display(&[p], Some(&projection), Placement::Projected)[0];
    map_of(view, width, height).apply(shown)
}

/// The pixel a world point lands on (`None` off the frame).
pub fn pixel_of(view: &View, width: u32, height: u32, p: [f64; 3]) -> Option<(u32, u32)> {
    let [x, y] = screen_of(view, width, height, p);
    (x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64).then_some((x as u32, y as u32))
}

/// A flat convex sheet: its corners in order, its colour (straight RGBA).
#[derive(Clone, Copy, Debug)]
pub struct Flat {
    pub corners: [[f64; 3]; 4],
    pub rgba: [f64; 4],
}

/// The depth key (smaller is nearer) where the sight line through screen point `(sx, sy)` meets a
/// sheet's plane, when that point is inside the sheet on screen: by casting the line onto the
/// plane, not through the core's depth planes.
pub fn sight_key(
    view: &View,
    width: u32,
    height: u32,
    flat: &Flat,
    sx: f64,
    sy: f64,
) -> Option<f64> {
    let c = camera_of(view);
    let projection = Projection::new(c, view.frame_center);
    let map = map_of(view, width, height);
    let screen: Vec<[f64; 2]> = flat
        .corners
        .iter()
        .map(|p| screen_of(view, width, height, *p))
        .collect();
    let side = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (sy - a[1]) - (b[1] - a[1]) * (sx - a[0]);
    let sides: Vec<f64> = (0..4)
        .map(|i| side(screen[i], screen[(i + 1) % 4]))
        .collect();
    if !(sides.iter().all(|s| *s > 0.0) || sides.iter().all(|s| *s < 0.0)) {
        return None;
    }
    // In the camera's axes the eye is at (0, 0, d), and the frame point (X, Y) sees the points
    // x = X (d - z) / (d zoom), y = Y (d - z) / (d zoom).
    let (fx, fy) = ((sx - map.x0) / map.xx, (sy - map.y0) / map.yy);
    let r = projection.rotation();
    let d = c.focal_distance;
    let p0 = flat.corners[0];
    let u: [f64; 3] = std::array::from_fn(|k| flat.corners[1][k] - p0[k]);
    let v: [f64; 3] = std::array::from_fn(|k| flat.corners[3][k] - p0[k]);
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let rot = |w: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|i| r[i][0] * w[0] + r[i][1] * w[1] + r[i][2] * w[2])
    };
    let m = rot(n);
    let q0 = rot(std::array::from_fn(|k| p0[k] - view.frame_center[k]));
    let k = m[0] * q0[0] + m[1] * q0[1] + m[2] * q0[2];
    let (a, b) = (fx / (d * c.zoom), fy / (d * c.zoom));
    // m . (a (d - z), b (d - z), z) = k, solved for z.
    let z = (k - (m[0] * a + m[1] * b) * d) / (m[2] - m[0] * a - m[1] * b);
    Some(-d / (d - z))
}

/// The bytes the sheets give pixel `(x, y)` over `bg`, composited back to front in depth order
/// (ties by their order in `flats`, the later on top), when one order of one set of sheets holds
/// over the whole pixel (checked at its centre and near its corners); `None` otherwise.
pub fn composite(
    view: &View,
    width: u32,
    height: u32,
    flats: &[Flat],
    x: u32,
    y: u32,
    bg: [u8; 4],
) -> Option<[u8; 4]> {
    let offsets = [
        (0.5, 0.5),
        (0.02, 0.02),
        (0.98, 0.02),
        (0.02, 0.98),
        (0.98, 0.98),
    ];
    let order_at = |dx: f64, dy: f64| {
        let mut hits: Vec<(f64, usize)> = flats
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                sight_key(view, width, height, f, x as f64 + dx, y as f64 + dy).map(|k| (k, i))
            })
            .collect();
        hits.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
        hits.into_iter().map(|(_, i)| i).collect::<Vec<_>>()
    };
    let order = order_at(offsets[0].0, offsets[0].1);
    if offsets[1..]
        .iter()
        .any(|(dx, dy)| order_at(*dx, *dy) != order)
    {
        return None;
    }
    let mut px = bg;
    for i in order.iter().rev() {
        px = blend(premultiply(flats[*i].rgba), 255, px);
    }
    Some(px)
}

/// The bytes the probe's painter order gives a clean pixel (see [`composite`]): the sheets
/// covering it composited in their order in `flats`, the later on top, whatever their depth. Not
/// CE's order, which sorts each face by its centre's depth.
pub fn painted(
    view: &View,
    width: u32,
    height: u32,
    flats: &[Flat],
    x: u32,
    y: u32,
    bg: [u8; 4],
) -> Option<[u8; 4]> {
    composite(view, width, height, flats, x, y, bg)?;
    let mut px = bg;
    for f in flats {
        if sight_key(view, width, height, f, x as f64 + 0.5, y as f64 + 0.5).is_some() {
            px = blend(premultiply(f.rgba), 255, px);
        }
    }
    Some(px)
}

/// The pixel a world point lands on, which must be clean for the sheets, the sheets' bytes there
/// over black in depth order, and in CE's painter order.
fn clean(view: &View, flats: &[Flat], p: [f64; 3]) -> (u32, u32, [u8; 4], [u8; 4]) {
    let (x, y) = pixel_of(view, WIDTH, HEIGHT, p).expect("a probe's point is on the frame");
    let bg = [0, 0, 0, 255];
    let bytes = composite(view, WIDTH, HEIGHT, flats, x, y, bg)
        .unwrap_or_else(|| panic!("the pixel of {p:?} ({x}, {y}) is not clean"));
    let ce = painted(view, WIDTH, HEIGHT, flats, x, y, bg).expect("a clean pixel");
    (x, y, bytes, ce)
}

// ---------------------------------------------------------------------------------------------
// The probes.

const RED: [f64; 4] = [0.85, 0.2, 0.15, 1.0];
const BLUE: [f64; 4] = [0.15, 0.3, 0.85, 1.0];
const GREEN: [f64; 4] = [0.2, 0.6, 0.3, 1.0];
const GREY: [f64; 4] = [0.5, 0.5, 0.5, 1.0];
const YELLOW: [f64; 4] = [1.0, 0.85, 0.2, 1.0];
const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
const ORANGE: [f64; 4] = [0.95, 0.5, 0.1, 1.0];

fn level(z: f64, hx: f64, hy: f64) -> [[f64; 3]; 4] {
    [[-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z]]
}

fn check(x: u32, y: u32, core: [u8; 4], ce: [u8; 4], what: &'static str) -> Check {
    Check {
        frame: 0,
        x,
        y,
        core: Expect::Bytes(core),
        ce,
        what,
    }
}

/// Two opaque sheets crossing: a level one, and one tilted through it. Painter order draws the
/// later of the two over the whole of the other.
pub fn intersecting_surfaces() -> Probe {
    let view = view(WIDTH, HEIGHT, 62.0 * DEG, -50.0 * DEG);
    let lev = level(0.0, 3.0, 2.0);
    let tilt = [
        [-3.0, -2.0, -1.5],
        [3.0, -2.0, 1.5],
        [3.0, 2.0, 1.5],
        [-3.0, 2.0, -1.5],
    ];
    let (a, b) = (sheet(lev, 6, 4), sheet(tilt, 6, 4));
    let items = [
        (a.clone(), mesh_record(1, &a, RED, false)),
        (b.clone(), mesh_record(2, &b, BLUE, false)),
    ];
    let flats = [
        Flat {
            corners: lev,
            rgba: RED,
        },
        Flat {
            corners: tilt,
            rgba: BLUE,
        },
    ];
    // A point of the tilted sheet below the level one: both cover its pixel.
    let (x, y, low, low_ce) = clean(&view, &flats, [-0.5, 0.0, -0.25]);
    let (hx, hy, high, high_ce) = clean(&view, &flats, [2.0, 0.5, 1.0]);
    Probe {
        name: "intersecting_surfaces",
        width: WIDTH,
        height: HEIGHT,
        deltas: vec![first_delta(view, &items, &[1, 2])],
        checks: vec![
            check(
                x,
                y,
                low,
                low_ce,
                "the level sheet in front where the tilted one dips below it",
            ),
            check(
                hx,
                hy,
                high,
                high_ce,
                "the tilted sheet in front where it rises above (CE right)",
            ),
        ],
    }
}

/// A curve in a vertical plane, its lower arc under an opaque level sheet. CE draws the whole curve
/// on top.
pub fn curve_behind_surface() -> Probe {
    let view = view(WIDTH, HEIGHT, 60.0 * DEG, -50.0 * DEG);
    let lev = level(0.0, 4.0, 3.0);
    let floor = sheet(lev, 8, 6);
    let tau = std::f64::consts::TAU;
    let ring = |t: f64| [2.2 * libm::cos(tau * t), 0.0, 1.2 * libm::sin(tau * t)];
    let curve = closed(96, ring);
    let items = [
        (floor.clone(), mesh_record(1, &floor, GREEN, false)),
        (curve.clone(), stroke_record(2, &curve, YELLOW, 24.0)),
    ];
    let flats = [Flat {
        corners: lev,
        rgba: GREEN,
    }];
    let (x, y, below, _) = clean(&view, &flats, ring(0.75));
    let (ux, uy, _, _) = clean(&view, &flats, ring(0.25));
    Probe {
        name: "curve_behind_surface",
        width: WIDTH,
        height: HEIGHT,
        deltas: vec![first_delta(view, &items, &[1, 2])],
        checks: vec![
            check(
                x,
                y,
                below,
                premultiply(YELLOW),
                "the curve's lower arc, under the sheet",
            ),
            check(
                ux,
                uy,
                premultiply(YELLOW),
                premultiply(YELLOW),
                "the curve's upper arc, above the sheet (CE right)",
            ),
        ],
    }
}

/// Translucent sheets over an opaque one at several depths: one crosses the opaque sheet, one hangs
/// above both. Painter order draws the crossing sheet over the opaque one everywhere.
pub fn translucent_over_opaque() -> Probe {
    let view = view(WIDTH, HEIGHT, 58.0 * DEG, -40.0 * DEG);
    let ground = level(0.0, 3.5, 2.5);
    let tilt = [
        [-3.0, -2.0, -1.2],
        [3.0, -2.0, 1.2],
        [3.0, 2.0, 1.2],
        [-3.0, 2.0, -1.2],
    ];
    let high = level(1.6, 1.4, 1.0);
    let red = [0.9, 0.15, 0.1, 0.5];
    let blue = [0.1, 0.3, 0.95, 0.4];
    let (g, t, h) = (sheet(ground, 7, 5), sheet(tilt, 6, 4), sheet(high, 4, 3));
    let items = [
        (g.clone(), mesh_record(1, &g, GREY, false)),
        (t.clone(), mesh_record(2, &t, red, false)),
        (h.clone(), mesh_record(3, &h, blue, false)),
    ];
    let flats = [
        Flat {
            corners: ground,
            rgba: GREY,
        },
        Flat {
            corners: tilt,
            rgba: red,
        },
        Flat {
            corners: high,
            rgba: blue,
        },
    ];
    // A point of the translucent sheet below the opaque one: both cover its pixel.
    let (x, y, dipped, dipped_ce) = clean(&view, &flats, [-1.8, -1.0, -0.72]);
    let (rx, ry, risen, risen_ce) = clean(&view, &flats, [2.5, -1.5, 1.0]);
    let (hx, hy, stacked, stacked_ce) = clean(&view, &flats, [0.3, 0.2, 1.6]);
    Probe {
        name: "translucent_over_opaque",
        width: WIDTH,
        height: HEIGHT,
        deltas: vec![first_delta(view, &items, &[1, 2, 3])],
        checks: vec![
            check(
                x,
                y,
                dipped,
                dipped_ce,
                "the opaque sheet where the translucent one dips below it",
            ),
            check(
                rx,
                ry,
                risen,
                risen_ce,
                "the translucent sheet over the opaque one where it rises (CE right)",
            ),
            check(
                hx,
                hy,
                stacked,
                stacked_ce,
                "the high sheet over both, in depth order (CE right)",
            ),
        ],
    }
}

/// A square outline in CE's cubic layout about `(cx, cy, cz)`, in its own xy-plane.
fn square(cx: f64, cy: f64, cz: f64, h: f64) -> Shape {
    polyline(&[
        [cx - h, cy - h, cz],
        [cx + h, cy - h, cz],
        [cx + h, cy + h, cz],
        [cx - h, cy + h, cz],
        [cx - h, cy - h, cz],
    ])
}

/// A label fixed in frame over a surface, and a fixed-orientation label below the surface. CE draws
/// both on top; the core keeps the fixed label on top and hides the other.
pub fn fixed_label_over_surface() -> Probe {
    let view = view(WIDTH, HEIGHT, 55.0 * DEG, -60.0 * DEG);
    let lev = level(0.0, 4.0, 3.0);
    let floor = sheet(lev, 8, 6);
    // The fixed label is in frame coordinates, over the middle of the frame.
    let label = square(0.0, -0.5, 0.0, 0.6);
    let mut fixed = fill_record(1, &label, WHITE, Space::Overlay);
    fixed.paint.stroke = Paint::One([0.1, 0.1, 0.1, 1.0]);
    fixed.paint.stroke_width = 8.0;
    // The fixed-orientation label's centre is under the floor; its points are about the centre.
    let centre = [0.8, 0.6, -0.9];
    let tag = square(centre[0], centre[1], centre[2], 0.5);
    let mut oriented = fill_record(3, &tag, ORANGE, Space::World);
    oriented.depth.fixed_orientation_centre = Some(centre);
    let items = [
        (label.clone(), fixed),
        (floor.clone(), mesh_record(2, &floor, GREEN, false)),
        (tag.clone(), oriented),
    ];
    let flats = [Flat {
        corners: lev,
        rgba: GREEN,
    }];
    let [lx, ly] = map_of(&view, WIDTH, HEIGHT).apply([0.0, -0.5, 0.0]);
    let (tx, ty, under, _) = clean(&view, &flats, centre);
    Probe {
        name: "fixed_label_over_surface",
        width: WIDTH,
        height: HEIGHT,
        deltas: vec![first_delta(view, &items, &[1, 2, 3])],
        checks: vec![
            check(
                lx as u32,
                ly as u32,
                premultiply(WHITE),
                premultiply(WHITE),
                "the fixed label on top, though drawn before the surface (CE right)",
            ),
            check(
                tx,
                ty,
                under,
                premultiply(ORANGE),
                "the fixed-orientation label, below the surface",
            ),
        ],
    }
}

/// Ordinary objects on both sides of a surface: one behind it, drawn after it with a higher
/// z-index; one in front of it, drawn before it with a lower one. Depth decides both ways.
pub fn depth_rule_both_ways() -> Probe {
    let view = view(WIDTH, HEIGHT, 65.0 * DEG, -70.0 * DEG);
    let wall = [
        [-3.0, 0.0, -1.5],
        [3.0, 0.0, -1.5],
        [3.0, 0.0, 2.0],
        [-3.0, 0.0, 2.0],
    ];
    let w = sheet(wall, 6, 4);
    let purple = [0.5, 0.2, 0.7, 1.0];
    let cyan = [0.1, 0.85, 0.9, 1.0];
    let behind = polyline(&[[-2.0, 1.5, 0.3], [2.0, 1.5, 0.3]]);
    let front = polyline(&[[-2.0, -1.5, 0.9], [2.0, -1.5, 0.9]]);
    let mut b = stroke_record(3, &behind, cyan, 30.0);
    b.depth.z_index = 2.0;
    let mut f = stroke_record(1, &front, YELLOW, 30.0);
    f.depth.z_index = -1.0;
    let items = [
        (front.clone(), f),
        (w.clone(), mesh_record(2, &w, purple, false)),
        (behind.clone(), b),
    ];
    let flats = [Flat {
        corners: wall,
        rgba: purple,
    }];
    let (bx, by, hidden, _) = clean(&view, &flats, [0.0, 1.5, 0.3]);
    let (fx, fy, _, _) = clean(&view, &flats, [0.0, -1.5, 0.9]);
    Probe {
        name: "depth_rule_both_ways",
        width: WIDTH,
        height: HEIGHT,
        deltas: vec![first_delta(view, &items, &[1, 2, 3])],
        checks: vec![
            check(
                bx,
                by,
                hidden,
                premultiply(cyan),
                "a line behind the surface, drawn after it",
            ),
            check(
                fx,
                fy,
                premultiply(YELLOW),
                premultiply(YELLOW),
                "a line in front of the surface, drawn before it (CE right)",
            ),
        ],
    }
}

/// A thick straight stroke rising through an opaque level sheet. CE draws all of it on top; the
/// core hides the part below, the edge across the stroke antialiased.
pub fn stroke_crossing_surface() -> Probe {
    let view = view(WIDTH, HEIGHT, 60.0 * DEG, -35.0 * DEG);
    let lev = level(0.0, 4.0, 3.0);
    let floor = sheet(lev, 8, 6);
    let navy = [0.1, 0.15, 0.45, 1.0];
    let line = polyline(&[[-3.0, 0.5, -1.0], [3.0, 0.5, 1.0]]);
    let items = [
        (floor.clone(), mesh_record(1, &floor, navy, false)),
        (line.clone(), stroke_record(2, &line, ORANGE, 40.0)),
    ];
    let flats = [Flat {
        corners: lev,
        rgba: navy,
    }];
    let (x, y, hidden, _) = clean(&view, &flats, [-2.0, 0.5, -2.0 / 3.0]);
    let (ux, uy, _, _) = clean(&view, &flats, [2.0, 0.5, 2.0 / 3.0]);
    Probe {
        name: "stroke_crossing_surface",
        width: WIDTH,
        height: HEIGHT,
        deltas: vec![first_delta(view, &items, &[1, 2])],
        checks: vec![
            check(
                x,
                y,
                hidden,
                premultiply(ORANGE),
                "the stroke below the sheet",
            ),
            check(
                ux,
                uy,
                premultiply(ORANGE),
                premultiply(ORANGE),
                "the stroke above the sheet (CE right)",
            ),
        ],
    }
}

/// A sheet seen exactly edge-on, a ring tilted a degree out of its plane, and a wall in front of
/// part of the ring. The sheet covers nothing; the ring, nearly edge-on too, its near and far arcs
/// a pixel either side of one line with the stroke wide enough to cover both, shows its near arc,
/// hidden by the wall only where that arc is behind it. CE draws the whole ring on top.
pub fn edge_on_surface() -> Probe {
    let view = view(WIDTH, HEIGHT, 90.0 * DEG, -90.0 * DEG);
    let lev = level(0.0, 3.0, 3.0);
    let floor = sheet(lev, 6, 6);
    let wall = [
        [-3.5, -1.0, -2.0],
        [3.5, -1.0, -2.0],
        [3.5, -1.0, 2.0],
        [-3.5, -1.0, 2.0],
    ];
    let w = sheet(wall, 7, 4);
    let teal = [0.1, 0.5, 0.5, 1.0];
    let tau = std::f64::consts::TAU;
    let tilt = DEG;
    let ring = |t: f64| {
        let s = 2.5 * libm::sin(tau * t);
        [
            2.5 * libm::cos(tau * t),
            s * libm::cos(tilt),
            s * libm::sin(tilt),
        ]
    };
    let curve = closed(120, ring);
    let items = [
        (floor.clone(), mesh_record(1, &floor, RED, false)),
        (w.clone(), mesh_record(2, &w, teal, false)),
        (curve.clone(), stroke_record(3, &curve, YELLOW, 30.0)),
    ];
    let flats = [Flat {
        corners: wall,
        rgba: teal,
    }];
    // At x = 2.4 the ring's near arc (y about -0.7) is behind the wall at y = -1; at x = 0 it
    // (y = -2.5) is in front of it, and the far arc (y = 2.5) behind it: on the pixel row the far
    // arc's centreline crosses, nearer the far arc than the near one, the near arc still shows.
    let (hx, hy, hidden, _) = clean(&view, &flats, [2.4, -1.0, 0.0]);
    let (vx, vy) = pixel_of(&view, WIDTH, HEIGHT, ring(0.25)).expect("the far arc is on the frame");
    let (ax, ay, above, _) = clean(&view, &flats, [-1.0, -1.0, 0.8]);
    Probe {
        name: "edge_on_surface",
        width: WIDTH,
        height: HEIGHT,
        deltas: vec![first_delta(view, &items, &[1, 2, 3])],
        checks: vec![
            check(
                hx,
                hy,
                hidden,
                premultiply(YELLOW),
                "the ring where its near arc is behind the wall",
            ),
            check(
                vx,
                vy,
                premultiply(YELLOW),
                premultiply(YELLOW),
                "the ring's near arc in front of the wall, on the far arc's row (CE right)",
            ),
            check(
                ax,
                ay,
                above,
                above,
                "the wall above the edge-on sheet, which covers nothing (CE right)",
            ),
        ],
    }
}

/// A lit sphere ringed by a curve and a title fixed in frame, the camera orbiting once over
/// `frames` frames (each later frame a delta of the view alone). In every frame the ring's far side
/// is behind the sphere; CE draws it on top.
pub fn orbit(frames: usize, width: u32, height: u32) -> Probe {
    let phi = 65.0 * DEG;
    let theta0 = -90.0 * DEG;
    let base = view(width, height, phi, theta0);
    let ball = sphere([0.0; 3], 1.5, 16, 30);
    let tau = std::f64::consts::TAU;
    let ring_z = 0.3;
    let ring = |t: f64| [2.4 * libm::cos(tau * t), 2.4 * libm::sin(tau * t), ring_z];
    let curve = closed(128, ring);
    let title = polyline(&[
        [-6.5, 3.3, 0.0],
        [-3.5, 3.3, 0.0],
        [-3.5, 3.7, 0.0],
        [-6.5, 3.7, 0.0],
        [-6.5, 3.3, 0.0],
    ]);
    let items = [
        (ball.clone(), mesh_record(1, &ball, ORANGE, true)),
        (curve.clone(), stroke_record(2, &curve, WHITE, 24.0)),
        (title.clone(), fill_record(3, &title, WHITE, Space::Overlay)),
    ];
    let mut deltas = vec![first_delta(base.clone(), &items, &[1, 2, 3])];
    let mut checks = Vec::new();
    for f in 0..frames {
        let theta = theta0 + tau * f as f64 / frames as f64;
        let mut v = base.clone();
        if let Camera::ThreeD(c) = &mut v.camera {
            c.theta = theta;
        }
        if f > 0 {
            let mut d = Delta::new(
                Time {
                    num: f as i64,
                    den: 15,
                },
                f as u64,
            );
            d.view = Some(v.clone());
            deltas.push(d);
        }
        // The camera looks from azimuth theta, so the ring's far side is opposite it.
        let far = [-2.4 * libm::cos(theta), -2.4 * libm::sin(theta), ring_z];
        if let Some((x, y)) = pixel_of(&v, width, height, far) {
            checks.push(Check {
                frame: f,
                x,
                y,
                core: Expect::Hidden(2),
                ce: premultiply(WHITE),
                what: "the ring's far side, behind the sphere",
            });
        }
    }
    Probe {
        name: "orbit",
        width,
        height,
        deltas,
        checks,
    }
}

/// A rippled sheet whose interior vertices are jittered, so its internal edges run at many angles,
/// and its boundary as a ring of points.
pub fn rippled_sheet() -> (Shape, Vec<[f64; 3]>) {
    let place = |u: f64, v: f64| {
        let interior = u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0;
        let jitter = if interior {
            0.015 * libm::sin(91.0 * u + 37.0 * v)
        } else {
            0.0
        };
        [
            -4.0 + 8.0 * (u + jitter),
            -2.5 + 5.0 * (v - jitter),
            0.4 * libm::sin(3.0 * u) * libm::cos(2.0 * v),
        ]
    };
    let (nu, nv) = (17, 11);
    let shape = grid(nu, nv, place);
    let mut ring = Vec::new();
    ring.extend((0..=nu).map(|i| place(i as f64 / nu as f64, 0.0)));
    ring.extend((1..=nv).map(|j| place(1.0, j as f64 / nv as f64)));
    ring.extend((0..nu).rev().map(|i| place(i as f64 / nu as f64, 1.0)));
    ring.extend((1..nv).rev().map(|j| place(0.0, j as f64 / nv as f64)));
    (shape, ring)
}

/// The seam probe's sheet colour in each of its frames: opaque, then translucent.
pub const SEAM_COLOURS: [[f64; 4]; 2] = [[0.2, 0.6, 0.9, 1.0], [0.2, 0.6, 0.9, 0.5]];

/// The seam probe: a rippled sheet over a white background, opaque in its first frame and
/// translucent in its second. The background must never show through its internal edges
/// (`tests/probes.rs` checks its whole interior).
pub fn seam() -> Probe {
    let (shape, _) = rippled_sheet();
    let mut v = view(WIDTH, HEIGHT, 55.0 * DEG, -60.0 * DEG);
    v.background = Background::Color([1.0, 1.0, 1.0, 1.0]);
    if let Camera::ThreeD(c) = &mut v.camera {
        c.zoom = 1.3;
    }
    let first = first_delta(
        v,
        &[(
            shape.clone(),
            mesh_record(1, &shape, SEAM_COLOURS[0], false),
        )],
        &[1],
    );
    let mut second = Delta::new(Time { num: 1, den: 15 }, 1);
    second.set = vec![mesh_record(1, &shape, SEAM_COLOURS[1], false)];
    Probe {
        name: "seam",
        width: WIDTH,
        height: HEIGHT,
        deltas: vec![first, second],
        checks: Vec::new(),
    }
}

/// The fixture's point colours, its instances' fill rows and its two strokes' colours.
pub const PAINT_KINDS_POINTS: [[f64; 4]; 3] = [
    [1.0, 0.85, 0.2, 1.0],
    [0.2, 0.9, 0.9, 1.0],
    [0.95, 0.3, 0.6, 1.0],
];
pub const PAINT_KINDS_FILLS: [[f64; 4]; 3] = [
    [0.9, 0.2, 0.15, 1.0],
    [0.2, 0.7, 0.3, 1.0],
    [0.25, 0.35, 0.95, 1.0],
];
pub const PAINT_KINDS_STROKES: [[f64; 4]; 2] = [[1.0, 1.0, 1.0, 1.0], [0.95, 0.6, 0.1, 1.0]];

/// Not a class probe: the paint kinds the probes do not draw, over a level sheet under the 3D
/// camera, so the identity gate, the thread matrix and the parity kit cover them. A dashed zigzag
/// with round caps and joins; a zigzag with square caps and bevel joins; an `instances` record of a
/// filled, stroked square, its members overlapping; a `points` record of blocks four pixels a side,
/// some overlapping, some below the sheet.
pub fn paint_kinds() -> Probe {
    let view = view(WIDTH, HEIGHT, 55.0 * DEG, -60.0 * DEG);
    let lev = level(0.0, 4.0, 3.0);
    let floor = sheet(lev, 4, 3);
    let zig = |y: f64| {
        polyline(&[
            [-3.0, y, 0.3],
            [-1.5, y + 0.6, 0.3],
            [0.0, y, 0.3],
            [1.5, y + 0.6, 0.3],
            [3.0, y, 0.3],
        ])
    };
    let (dashed, beveled) = (zig(-2.2), zig(1.6));
    let mut a = stroke_record(2, &dashed, PAINT_KINDS_STROKES[0], 20.0);
    a.paint.cap = Cap::Round;
    a.paint.joint = Joint::Round;
    a.dash = Some(Dash {
        period: 0.6,
        duty: 0.6,
        phase: 0.25,
    });
    let mut b = stroke_record(3, &beveled, PAINT_KINDS_STROKES[1], 24.0);
    b.paint.cap = Cap::Square;
    b.paint.joint = Joint::Bevel;
    let unit = polyline(&[
        [-0.4, -0.4, 0.0],
        [0.4, -0.4, 0.0],
        [0.4, 0.4, 0.0],
        [-0.4, 0.4, 0.0],
        [-0.4, -0.4, 0.0],
    ]);
    let fills = Stops::new(PAINT_KINDS_FILLS.to_vec());
    let strokes = Stops::new(vec![PAINT_KINDS_STROKES[0]; 3]);
    let mut c = fill_record(4, &unit, WHITE, Space::World);
    c.kind = RecordKind::Instances;
    c.paint.stroke = Paint::One(WHITE);
    c.paint.stroke_width = 6.0;
    let shift = |dx: f64, dy: f64, dz: f64| -> Affine {
        [
            [1.0, 0.0, 0.0, dx],
            [0.0, 1.0, 0.0, dy],
            [0.0, 0.0, 1.0, dz],
        ]
    };
    c.body = Body::Instances(Instances {
        placements: vec![
            shift(-1.6, -0.3, 0.5),
            shift(-1.1, 0.0, 0.7),
            shift(-0.6, 0.3, 0.9),
        ],
        fill: fills.key(),
        stroke: strokes.key(),
        objects: vec![401, 402, 403],
    });
    // Twelve points along a line, every fourth below the sheet, neighbours' blocks overlapping.
    let cloud_points: Vec<[f64; 3]> = (0..12)
        .map(|i| {
            let z = if i % 4 == 0 { -0.4 } else { 0.6 };
            [1.0 + 0.12 * i as f64, -0.5 + 0.03 * (i % 3) as f64, z]
        })
        .collect();
    let colours = Stops::new((0..12).map(|i| PAINT_KINDS_POINTS[i % 3]).collect());
    let cloud = Shape::new(ShapeData::Path {
        points: cloud_points,
    });
    let mut d = Record::new(RecordKind::Points, 5, 105);
    d.geometry.shape_a = Some(cloud.key());
    d.depth.space = Space::World;
    d.body = Body::Points(Points {
        colors: colours.key(),
        radius: 4.0,
    });
    let items = [
        (floor.clone(), mesh_record(1, &floor, GREY, false)),
        (dashed.clone(), a),
        (beveled.clone(), b),
        (unit.clone(), c),
        (cloud.clone(), d),
    ];
    let mut delta = first_delta(view, &items, &[1, 2, 3, 4, 5]);
    delta.upload_paints = vec![fills, strokes, colours];
    Probe {
        name: "paint_kinds",
        width: WIDTH,
        height: HEIGHT,
        deltas: vec![delta],
        checks: Vec::new(),
    }
}
