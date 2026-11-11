//! The scenes of the contract's §12 (Fixtures, The views): its two views and their variants, its
//! lines and polygons written as §12 writes them, records of each kind it uses, and XF 1.5's
//! source section written from `XF.md`'s binary form, all from the XF types alone.
#![allow(dead_code)]

use arachne_core::xf::{
    Background, Body, Camera, Delta, FEATURE_PRE_PROJECTED, FaceGroup, Layer, LightModel,
    OrderEntry, Paint, Points, PreProjected, Record, RecordKind, Shape, ShapeData, Space, Stops,
    ThreeDCamera, Time, View, wire,
};

/// §12's input grid.
pub const GRID: f64 = 1.0 / 1_048_576.0;

/// View F: 16 x 16 px over a 16 x 16 unit frame about the origin, `xx = 1`, `yy = −1`,
/// `x0 = y0 = 8`, a flat camera, a black background.
pub fn view_f() -> View {
    View {
        pixel_width: 16,
        pixel_height: 16,
        frame_width: 16.0,
        frame_height: 16.0,
        frame_center: [0.0; 3],
        mapping_centre: [0.0; 3],
        background: Background::Color([0.0, 0.0, 0.0, 1.0]),
        camera: Camera::Flat,
        cairo_line_width_multiple: 1.0,
        use_z_index: false,
        light: LightModel::Ce,
        input_grid: GRID,
    }
}

fn camera(focal_distance: f64) -> Camera {
    Camera::ThreeD(ThreeDCamera {
        phi: 0.0,
        theta: -std::f64::consts::FRAC_PI_2,
        gamma: 0.0,
        focal_distance,
        zoom: 1.0,
        exponential_projection: false,
        light_source: [-7.0, -9.0, 10.0],
        shading_factor: 0.2,
        should_apply_shading: false,
    })
}

/// View C: 64 x 64 px over a 16 x 16 unit frame, the 3D camera with the identity rotation, focal
/// distance 20, zoom 1: `xx = 4`, `yy = −4`, `x0 = y0 = 32`.
pub fn view_c() -> View {
    View {
        pixel_width: 64,
        pixel_height: 64,
        camera: camera(20.0),
        ..view_f()
    }
}

/// W26's wider view C: 128 x 64 px over a 32 x 16 unit frame (`x0 = 64`).
pub fn view_c_wide() -> View {
    View {
        pixel_width: 128,
        frame_width: 32.0,
        ..view_c()
    }
}

/// View O: view C's camera at focal distance `10^6` over W26's frame, in effect orthographic.
pub fn view_o() -> View {
    View {
        camera: camera(1e6),
        ..view_c_wide()
    }
}

/// A line from `p` to `q` as §12 writes it: one cubic with controls the nearest binary64 values to
/// `(2p + q) / 3` and `(p + 2q) / 3`.
pub fn line(p: [f64; 3], q: [f64; 3]) -> [[f64; 3]; 4] {
    let third = |a: f64, b: f64| (2.0 * a + b) / 3.0;
    [
        p,
        std::array::from_fn(|k| third(p[k], q[k])),
        std::array::from_fn(|k| third(q[k], p[k])),
        q,
    ]
}

/// An open polyline of §12's lines.
pub fn open(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    points.windows(2).flat_map(|w| line(w[0], w[1])).collect()
}

/// A polygon of §12's lines, its subpath closed.
pub fn polygon(corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut closed = corners.to_vec();
    closed.push(corners[0]);
    open(&closed)
}

/// The axis-aligned square or rectangle `[x0, x1] × [y0, y1]` at `z`, counter-clockwise.
pub fn rect(x: [f64; 2], y: [f64; 2], z: f64) -> Vec<[f64; 3]> {
    polygon(&[
        [x[0], y[0], z],
        [x[1], y[0], z],
        [x[1], y[1], z],
        [x[0], y[1], z],
    ])
}

/// A frame's records and shapes, set in one order.
#[derive(Clone, Debug, Default)]
pub struct Build {
    pub shapes: Vec<Shape>,
    pub records: Vec<Record>,
    pub paints: Vec<Stops>,
    world: bool,
}

pub const OPAQUE_RED: [f64; 4] = [1.0, 0.0, 0.0, 1.0];
pub const OPAQUE_GREEN: [f64; 4] = [0.0, 1.0, 0.0, 1.0];
pub const OPAQUE_BLUE: [f64; 4] = [0.0, 0.0, 1.0, 1.0];
pub const WHITE: [f64; 4] = [1.0; 4];
pub const BLACK: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

impl Build {
    /// Records in world space under a 3D camera (else in screen space, for view F).
    pub fn new(world: bool) -> Build {
        Build {
            world,
            ..Build::default()
        }
    }

    fn record(&mut self, kind: RecordKind, data: ShapeData) -> &mut Record {
        let slot = self.records.len() as u32;
        let shape = Shape::new(data);
        let mut r = Record::new(kind, slot, 100 + slot);
        r.geometry.shape_a = Some(shape.key());
        r.depth.space = if self.world {
            Space::World
        } else {
            Space::Screen
        };
        self.shapes.push(shape);
        self.records.push(r);
        self.records.last_mut().unwrap()
    }

    /// A path record of `points`, unpainted: set its paints on the record returned.
    pub fn path(&mut self, points: Vec<[f64; 3]>) -> &mut Record {
        self.record(RecordKind::Path, ShapeData::Path { points })
    }

    /// A path filled `rgba`.
    pub fn fill(&mut self, points: Vec<[f64; 3]>, rgba: [f64; 4]) -> &mut Record {
        let r = self.path(points);
        r.paint.fill = Paint::One(rgba);
        r
    }

    /// A path stroked `rgba` at `width` frame units.
    pub fn stroke(&mut self, points: Vec<[f64; 3]>, rgba: [f64; 4], width: f64) -> &mut Record {
        let r = self.path(points);
        r.paint.stroke = Paint::One(rgba);
        r.paint.stroke_width = width;
        r
    }

    /// A camera-facing glyph: a path filled `rgba`, kept facing the camera about `centre`.
    pub fn glyph(
        &mut self,
        points: Vec<[f64; 3]>,
        rgba: [f64; 4],
        centre: [f64; 3],
    ) -> &mut Record {
        let r = self.fill(points, rgba);
        r.depth.fixed_orientation_centre = Some(centre);
        r
    }

    /// W5's label glyph, the square `[−1.5, 1.5]²` about `centre`, fill white.
    pub fn label(&mut self, centre: [f64; 3]) -> &mut Record {
        let [cx, cy, cz] = centre;
        self.glyph(
            rect([cx - 1.5, cx + 1.5], [cy - 1.5, cy + 1.5], cz),
            WHITE,
            centre,
        )
    }

    /// A mesh of `triangles` over `positions`, filled `rgba`.
    pub fn mesh(
        &mut self,
        positions: Vec<[f64; 3]>,
        triangles: Vec<[u32; 3]>,
        rgba: [f64; 4],
    ) -> &mut Record {
        let r = self.record(
            RecordKind::Mesh,
            ShapeData::Mesh {
                positions,
                triangles,
                normals: None,
            },
        );
        r.paint.fill = Paint::One(rgba);
        r
    }

    /// A face group: one record drawing each of `faces` (closed polygons) filled `rgba`, one order
    /// entry per face.
    pub fn face_group(&mut self, faces: &[Vec<[f64; 3]>], rgba: [f64; 4]) -> &mut Record {
        let fill = Stops::new(vec![rgba; faces.len()]);
        let key = fill.key();
        self.paints.push(fill);
        let mut face_ends = Vec::new();
        for f in faces {
            face_ends.push(face_ends.last().copied().unwrap_or(0) + f.len() as u32);
        }
        let points = faces.concat();
        let n = faces.len() as u32;
        let r = self.record(RecordKind::FaceGroup, ShapeData::Path { points });
        r.body = Body::FaceGroup(FaceGroup {
            uv: (0..n).map(|i| [i, 0]).collect(),
            grid: [n, 1],
            fill: key,
            stroke: None,
            face_ends,
            face_widths: Vec::new(),
        });
        r
    }

    /// A points record: one block per position, coloured `rgba`, of CE's stroke width `radius`.
    pub fn points(&mut self, positions: Vec<[f64; 3]>, rgba: [f64; 4], radius: f64) -> &mut Record {
        let colors = Stops::new(vec![rgba; positions.len()]);
        let key = colors.key();
        self.paints.push(colors);
        let r = self.record(RecordKind::Points, ShapeData::Path { points: positions });
        r.body = Body::Points(Points {
            colors: key,
            radius,
        });
        r
    }

    /// A pre-projected path: its tail's points as the scene camera produced them (frame units),
    /// filled `rgba`.
    pub fn pre_projected(&mut self, points: Vec<[f64; 3]>, rgba: [f64; 4]) -> &mut Record {
        let r = self.path(points);
        r.features |= FEATURE_PRE_PROJECTED;
        r.pre_projected = Some(PreProjected {
            points: r.geometry.shape_a.unwrap(),
            gradient: None,
            fill: Paint::One(rgba),
            stroke: Paint::None,
            background_stroke: Paint::None,
        });
        r
    }

    /// The delta at `seq` setting `view`, every record, and the order they were added in.
    pub fn delta(&self, seq: u64, view: View) -> Delta {
        let mut d = Delta::new(
            Time {
                num: seq as i64,
                den: 1,
            },
            seq,
        );
        d.version = (1, 4);
        d.view = Some(view);
        d.order = Some(
            self.records
                .iter()
                .flat_map(|r| {
                    let parts = match &r.body {
                        Body::FaceGroup(f) => f.uv.len() as u32,
                        _ => 1,
                    };
                    (0..parts).map(|part| OrderEntry {
                        layer: Layer::Moving,
                        slot: r.slot,
                        part,
                    })
                })
                .collect(),
        );
        d.upload = self.shapes.clone();
        d.upload_paints = self.paints.clone();
        d.set = self.records.clone();
        d
    }
}

/// One row of XF 1.5's source section: a slot, its facing group, its paths of `(token, class
/// name)` nodes, owner first.
#[derive(Clone, Debug)]
pub struct Row {
    pub slot: u32,
    pub group: u32,
    pub paths: Vec<Vec<(u32, String)>>,
}

/// A row from string class names.
pub fn row(slot: u32, group: u32, paths: &[&[(u32, &str)]]) -> Row {
    Row {
        slot,
        group,
        paths: paths
            .iter()
            .map(|p| p.iter().map(|(t, c)| (*t, c.to_string())).collect())
            .collect(),
    }
}

/// A source section as bytes: its class names, each once in the order the rows first use them
/// (or `names` as given, where a witness writes them itself), then its rows; each class written as
/// its index into the names.
pub fn section(rows: &[Row], names: Option<&[&str]>) -> Vec<u8> {
    let mut list: Vec<String> =
        names.map_or_else(Vec::new, |n| n.iter().map(|s| s.to_string()).collect());
    if names.is_none() {
        for r in rows {
            for p in &r.paths {
                for (_, c) in p {
                    if !list.contains(c) {
                        list.push(c.clone());
                    }
                }
            }
        }
    }
    let index = |c: &str| {
        list.iter()
            .position(|n| n == c)
            .map_or(u32::MAX, |i| i as u32)
    };
    let mut b = Vec::new();
    let u32le = |b: &mut Vec<u8>, v: u32| b.extend(v.to_le_bytes());
    u32le(&mut b, list.len() as u32);
    for n in &list {
        u32le(&mut b, n.len() as u32);
        b.extend(n.as_bytes());
    }
    u32le(&mut b, rows.len() as u32);
    for r in rows {
        u32le(&mut b, r.slot);
        u32le(&mut b, r.group);
        u32le(&mut b, r.paths.len() as u32);
        for p in &r.paths {
            u32le(&mut b, p.len() as u32);
            for (t, c) in p {
                u32le(&mut b, *t);
                u32le(&mut b, index(c));
            }
        }
    }
    b
}

/// A row as written: its slot, its facing group, its paths of `(token, class index)`.
pub type RawRow = (u32, u32, Vec<Vec<(u32, u32)>>);

/// A source section written as given: its class names, then rows of `(slot, facing group, paths
/// of (token, class index))`, for a witness that writes what a writer must not.
pub fn raw_section(names: &[&str], rows: &[RawRow]) -> Vec<u8> {
    let mut b = Vec::new();
    let u32le = |b: &mut Vec<u8>, v: u32| b.extend(v.to_le_bytes());
    u32le(&mut b, names.len() as u32);
    for n in names {
        u32le(&mut b, n.len() as u32);
        b.extend(n.as_bytes());
    }
    u32le(&mut b, rows.len() as u32);
    for (slot, group, paths) in rows {
        u32le(&mut b, *slot);
        u32le(&mut b, *group);
        u32le(&mut b, paths.len() as u32);
        for p in paths {
            u32le(&mut b, p.len() as u32);
            for (t, c) in p {
                u32le(&mut b, *t);
                u32le(&mut b, *c);
            }
        }
    }
    b
}

/// A delta's bytes as XF 1.5 writes them: the XF 1.4 bytes, minor 5 in its header, then the
/// source section after the order (`XF.md` at the label head, Binary form).
pub fn delta15(d: &Delta, section: &[u8]) -> Vec<u8> {
    let mut b = wire::encode_delta(d);
    b[2..4].copy_from_slice(&5u16.to_le_bytes());
    b.extend(section);
    b
}

/// A file of deltas' bytes at XF 1.5: the magic, the version, the count, each delta an item.
pub fn file15(deltas: &[Vec<u8>]) -> Vec<u8> {
    let mut b = b"ARXF".to_vec();
    b.extend(1u16.to_le_bytes());
    b.extend(5u16.to_le_bytes());
    b.extend((deltas.len() as u32).to_le_bytes());
    for d in deltas {
        b.extend((d.len() as u32).to_le_bytes());
        b.extend(d);
    }
    b
}
