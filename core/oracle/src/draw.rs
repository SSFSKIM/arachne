//! What each entry of the draw order draws (contract §2, §3, §4, §8, §9's `paint_rows@1`): a
//! mesh's triangles, a path's layers or a points record's blocks, each with the identity its
//! fragments carry and the inputs the screen shift reads.
//!
//! Only what decides coverage is here: points, placements, widths, styles, and whether a layer
//! paints at all (`alpha_round@1`). Colour beyond that is not.

use std::collections::HashMap;

use arachne_core::xf::{
    Affine, Body, Camera, Cap, Dash, IDENTITY, Joint, Key, Paint, PaintBlock, Record, RecordKind,
    Shape, ShapeData, Stops, View, compose, place,
};

use crate::camera::{FrameMap, Placement, Projection, map_point_cloud, projection};

/// A paint's straight RGBA rows, or `None` for no paint.
pub fn rows(paint: &Paint, tables: &HashMap<Key, Stops>) -> Option<Vec<[f64; 4]>> {
    match paint {
        Paint::None => None,
        Paint::One(c) => Some(vec![*c]),
        Paint::Two(cs) => Some(cs.to_vec()),
        Paint::Table(key) => tables.get(key).map(|t| t.rows().to_vec()),
    }
}

/// The paints and widths a record draws with (`paint_rows@1`).
#[derive(Clone, Debug, PartialEq)]
pub struct DrawnPaint {
    pub fill: Option<Vec<[f64; 4]>>,
    pub stroke: Option<Vec<[f64; 4]>>,
    pub background_stroke: Option<Vec<[f64; 4]>>,
    pub stroke_width: f64,
    pub background_stroke_width: f64,
}

/// A record's paint block as drawn: its own, or under a paint blend its own mixed toward
/// `paint_b` by `mix`. `Err` where numpy would not broadcast the row counts.
pub fn drawn_paint(
    block: &PaintBlock,
    tables: &HashMap<Key, Stops>,
) -> Result<DrawnPaint, &'static str> {
    let own = |p: &Paint| rows(p, tables);
    let Some(b) = &block.paint_b else {
        return Ok(DrawnPaint {
            fill: own(&block.fill),
            stroke: own(&block.stroke),
            background_stroke: own(&block.background_stroke),
            stroke_width: block.stroke_width,
            background_stroke_width: block.background_stroke_width,
        });
    };
    let mix = block.mix;
    let width = |a: f64, z: f64| {
        if mix == 1.0 {
            z
        } else {
            (1.0 - mix) * a + mix * z
        }
    };
    let blend = |a: &Paint, z: &Paint| -> Result<Option<Vec<[f64; 4]>>, &'static str> {
        let (ra, rz) = (own(a), own(z));
        if mix == 1.0 {
            return Ok(rz);
        }
        let (ra, rz) = (ra.unwrap_or_default(), rz.unwrap_or_default());
        let n = match (ra.len(), rz.len()) {
            (0, _) | (_, 0) => 0,
            (1, n) | (n, 1) => n,
            (m, n) if m == n => n,
            _ => return Err("a paint blend of row counts numpy does not broadcast"),
        };
        let row = |r: &[[f64; 4]], i: usize| r[if r.len() == 1 { 0 } else { i }];
        let out: Vec<[f64; 4]> = (0..n)
            .map(|i| {
                let (p, q) = (row(&ra, i), row(&rz, i));
                std::array::from_fn(|k| (1.0 - mix) * p[k] + mix * q[k])
            })
            .collect();
        Ok((!out.is_empty()).then_some(out))
    };
    Ok(DrawnPaint {
        fill: blend(&block.fill, &b.fill)?,
        stroke: blend(&block.stroke, &b.stroke)?,
        background_stroke: blend(&block.background_stroke, &b.background_stroke)?,
        stroke_width: width(block.stroke_width, b.stroke_width),
        background_stroke_width: width(block.background_stroke_width, b.background_stroke_width),
    })
}

/// A stop's alpha as Cairo keeps it, a 16-bit short: `trunc(d · 65535 + 0.5)` of the alpha
/// clamped to `[0, 1]` (NaN reads 0).
pub fn alpha_short(d: f64) -> u32 {
    (d.clamp(0.0, 1.0) * 65535.0 + 0.5) as u32
}

/// `alpha_round@1`: the alpha byte Cairo writes, `alpha_short(d) >> 8`.
pub fn alpha_byte(d: f64) -> u32 {
    alpha_short(d) >> 8
}

/// Whether a layer painted with `rows` at group opacity `opacity` paints anything
/// (`alpha_round@1`): a solid whose alpha byte is not 0, or a gradient one of whose stops' shorts
/// is above `0xff`. A shaded layer's stops are its first two rows (one row lit twice).
pub fn paints(rows: &[[f64; 4]], opacity: f64, shaded: bool) -> bool {
    let considered = if shaded {
        &rows[..rows.len().min(2)]
    } else {
        rows
    };
    considered
        .iter()
        .any(|r| alpha_short(r[3] * opacity) > 0xff)
}

/// `pointwise_become_partial`'s cut of `points` (CE's cubic layout) to the window `(lo, hi)`
/// (contract §2).
pub fn revealed(points: &[[f64; 3]], lo: f64, hi: f64) -> Vec<[f64; 3]> {
    if lo <= 0.0 && hi >= 1.0 {
        return points.to_vec();
    }
    let n = points.len() / 4;
    if n == 0 {
        return Vec::new();
    }
    let at = |v: f64| -> (usize, f64) {
        if v >= 1.0 {
            (n - 1, 1.0)
        } else if v <= 0.0 {
            (0, 0.0)
        } else {
            let x = n as f64 * v;
            ((x as usize).min(n - 1), x.rem_euclid(1.0))
        }
    };
    let cubic = |i: usize| -> [[f64; 3]; 4] { std::array::from_fn(|k| points[4 * i + k]) };
    let (i_lo, r_lo) = at(lo);
    let (i_hi, r_hi) = at(hi);
    let mut out = Vec::new();
    if i_lo == i_hi {
        out.extend(partial(cubic(i_lo), r_lo, r_hi));
    } else {
        out.extend(partial(cubic(i_lo), r_lo, 1.0));
        for i in i_lo + 1..i_hi {
            out.extend(cubic(i));
        }
        out.extend(partial(cubic(i_hi), 0.0, r_hi));
    }
    out
}

fn lerp(p: [f64; 3], q: [f64; 3], t: f64) -> [f64; 3] {
    std::array::from_fn(|k| p[k] + (q[k] - p[k]) * t)
}

/// The points de Casteljau's split of `c` at `t` makes: `ab, abc, bcd, cd`.
fn split(c: [[f64; 3]; 4], t: f64) -> [[f64; 3]; 4] {
    let ab = lerp(c[0], c[1], t);
    let bc = lerp(c[1], c[2], t);
    let cd = lerp(c[2], c[3], t);
    let abc = lerp(ab, bc, t);
    let bcd = lerp(bc, cd, t);
    [ab, abc, bcd, cd]
}

fn tail(c: [[f64; 3]; 4], t: f64) -> [[f64; 3]; 4] {
    let [_, abc, bcd, cd] = split(c, t);
    [lerp(abc, bcd, t), bcd, cd, c[3]]
}

fn head(c: [[f64; 3]; 4], t: f64) -> [[f64; 3]; 4] {
    let [ab, abc, bcd, _] = split(c, t);
    [c[0], ab, abc, lerp(abc, bcd, t)]
}

fn partial(c: [[f64; 3]; 4], a: f64, b: f64) -> [[f64; 3]; 4] {
    if a >= 1.0 {
        return [c[3]; 4];
    }
    head(tail(c, a), (b - a) / (1.0 - a))
}

fn path_shape<'a>(shapes: &'a HashMap<Key, Shape>, key: &Key) -> Option<&'a [[f64; 3]]> {
    match shapes.get(key).map(Shape::data) {
        Some(ShapeData::Path { points }) => Some(points),
        _ => None,
    }
}

/// A `path` record's placed points (contract §2, "A path record's drawn points"): its shape
/// placed, cut to its reveal window, blended with its second shape. `None` without a path shape,
/// or where the blend's two shapes differ in length.
pub fn path_points(rec: &Record, shapes: &HashMap<Key, Shape>) -> Option<Vec<[f64; 3]>> {
    let g = &rec.geometry;
    let a = path_shape(shapes, g.shape_a.as_ref()?)?;
    let mut points: Vec<[f64; 3]> = a.iter().map(|p| place(&g.place_a, *p)).collect();
    if let Some((lo, hi)) = g.reveal {
        points = revealed(&points, lo, hi);
    }
    if let Some(blend) = &g.blend {
        let b = path_shape(shapes, &blend.shape_b)?;
        if b.len() != points.len() {
            return None;
        }
        let t = blend.t;
        points = points
            .iter()
            .zip(b)
            .map(|(p, q)| {
                let q = place(&blend.place_b, *q);
                std::array::from_fn(|k| (1.0 - t) * p[k] + t * q[k])
            })
            .collect();
    }
    Some(points)
}

/// How many points a `path` record's shape keeps after its reveal window (`None` without one).
pub fn cut_len<'a>(rec: &Record, shape: &dyn Fn(&Key) -> Option<&'a [[f64; 3]]>) -> Option<usize> {
    let a = shape(rec.geometry.shape_a.as_ref()?)?;
    Some(match rec.geometry.reveal {
        Some((lo, hi)) => revealed(a, lo, hi).len(),
        None => a.len(),
    })
}

/// A point block's side: `t = trunc(1 + (radius − 1))`, no block where `t < 1`.
pub fn block_side(radius: f64) -> i64 {
    let t = (1.0 + (radius - 1.0)).trunc();
    if t >= 1.0 && t.is_finite() {
        t.min(i64::MAX as f64) as i64
    } else {
        0
    }
}

/// The layer a fragment paints, as `ORACLE_FORMAT` codes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Layer {
    BackgroundStroke = 0,
    Fill = 1,
    Stroke = 2,
    Mesh = 3,
    Points = 4,
}

impl Layer {
    pub fn name(self) -> &'static str {
        match self {
            Layer::BackgroundStroke => "background_stroke",
            Layer::Fill => "fill",
            Layer::Stroke => "stroke",
            Layer::Mesh => "mesh",
            Layer::Points => "points",
        }
    }
}

/// The points one placing puts in the world, for the screen shift: a shape's points under the
/// record's placements, outer first (contract §6, "The snap's departure"), and the coefficient
/// the drawn points take them with: 1, or a shape blend's `1 − t` and `t`.
#[derive(Clone, Debug, PartialEq)]
pub struct Placing {
    pub places: [Affine; 2],
    pub points: Vec<[f64; 3]>,
    pub weight: f64,
}

/// What every drawn thing of one record shares: where it is in the order and how it is shown.
#[derive(Clone, Debug)]
pub struct Common<'a> {
    pub entry: u32,
    pub record: &'a Record,
    pub object: u32,
    pub part: u32,
    pub placement: Placement,
    /// The projection, where the camera is 3D.
    pub projection: Option<Projection>,
    /// The placings whose departure bounds its screen shift.
    pub placings: Vec<Placing>,
    /// A pre-projected record's pivot (contract §2), where it carries a centre.
    pub pivot: Option<Pivot>,
}

/// A pre-projected record's pivot (contract §2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pivot {
    /// The carried centre moved as the record's first point moved to its tail's first point.
    At([f64; 3]),
    /// The record or its tail has no first point to move it by: no key is stated
    /// (`pivot_unknown`).
    Unknown,
}

/// One path drawn: a `path` record, a face of a face group, or a path instance's member.
#[derive(Clone, Debug)]
pub struct PathItem<'a> {
    pub common: Common<'a>,
    /// `i` of its layers' `sub = 3 i + k`.
    pub index: u32,
    /// Its shown points, in frame units.
    pub shown: Vec<[f64; 3]>,
    /// Its drawn points before display: placed, cut to its reveal window and blended (a
    /// pre-projected tail's points as produced). Depth reads them (contract §5).
    pub world: Vec<[f64; 3]>,
    /// Per layer (background stroke, fill, stroke): drawn or not, and the stroke's width in frame
    /// units (0 for the fill).
    pub layers: [Option<f64>; 3],
    pub joint: Joint,
    pub cap: Cap,
    pub dash: Option<Dash>,
    /// The record's XF 1.4 joins, for a path record.
    pub joins: Option<Vec<u8>>,
    /// A face of a face group (`round_cap_dot@1` reads it).
    pub face: bool,
    /// Per layer, its straight RGBA rows (`paint_rows@1`), for colour (§9).
    pub rows: [Option<Vec<[f64; 4]>>; 3],
    /// Whether the camera shades it (`ce_shading@1`).
    pub shaded: bool,
    pub map: FrameMap,
}

/// One mesh drawn: a `mesh` record or a mesh instance's member.
#[derive(Clone, Debug)]
pub struct MeshItem<'a> {
    pub common: Common<'a>,
    /// Its triangles' `sub` start: `t i` for member `i` of a `t`-triangle prototype.
    pub sub: u32,
    /// Per triangle, its shown corners.
    pub triangles: Vec<[[f64; 3]; 3]>,
    /// Per triangle, its placed corners (before display).
    pub placed: Vec<[[f64; 3]; 3]>,
    /// Its fill, straight RGBA (a member's row for an instance).
    pub fill: [f64; 4],
    /// Per triangle, its corners' carried normals as placed (`ce_shading@1`), where it carries
    /// them.
    pub normals: Option<Vec<[[f64; 3]; 3]>>,
    /// Whether the camera shades it (`ce_shading@1`).
    pub shaded: bool,
    pub map: FrameMap,
}

/// One point of a `points` record that draws a block.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockPoint {
    pub index: u32,
    /// Its placed position, before display.
    pub placed: [f64; 3],
    /// Its shown position, in frame units.
    pub shown: [f64; 3],
    /// Its pixel position (not yet truncated).
    pub at: [f64; 2],
}

/// A `points` record's points that draw blocks.
#[derive(Clone, Debug)]
pub struct PointsItem<'a> {
    pub common: Common<'a>,
    /// Per point of the record (by `index`), its straight RGBA.
    pub colours: Vec<[f64; 4]>,
    /// Point count `n` of `sub = offset index · n + i`.
    pub count: u32,
    pub side: i64,
    pub points: Vec<BlockPoint>,
}

#[derive(Clone, Debug)]
pub enum Item<'a> {
    Path(PathItem<'a>),
    Mesh(MeshItem<'a>),
    Points(PointsItem<'a>),
}

/// Which drawn things [`items_of`] yields: those that paint, or only those dropped for painting
/// nothing (`alpha_round@1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Visible,
    /// A path all of whose layers paint nothing, as one fill of width 0, and a mesh whose alpha
    /// byte is 0: what the registry is offered beside the painting records in the counterfactual
    /// sets of `invisible_registry_offer` (contract §5), and the mesh triangles the contract names
    /// `zero_alpha_fragment` (§1).
    Invisible,
}

/// Everything the frame's order draws, in order (overlay entries excluded). The frame is assumed
/// not refused ([`crate::stream::State::refusals`] empty).
pub fn items<'a>(
    view: &'a View,
    records: &'a std::collections::BTreeMap<u32, Record>,
    order: &'a [arachne_core::xf::OrderEntry],
    shapes: &'a HashMap<Key, Shape>,
    tables: &'a HashMap<Key, Stops>,
) -> Vec<Item<'a>> {
    items_of(view, records, order, shapes, tables, Which::Visible)
}

/// [`items`], or what it drops for painting nothing ([`Which`]).
pub fn items_of<'a>(
    view: &'a View,
    records: &'a std::collections::BTreeMap<u32, Record>,
    order: &'a [arachne_core::xf::OrderEntry],
    shapes: &'a HashMap<Key, Shape>,
    tables: &'a HashMap<Key, Stops>,
    which: Which,
) -> Vec<Item<'a>> {
    let invisible = which == Which::Invisible;
    let mut out = Vec::new();
    let pr = projection(view);
    let shading = matches!(&view.camera, Camera::ThreeD(c) if c.should_apply_shading);
    for (entry, e) in order.iter().enumerate() {
        let rec = &records[&e.slot];
        if crate::stream::is_overlay(rec) {
            continue;
        }
        let placement = Placement::of(rec, view);
        let pivot = rec.pre_projected.as_ref().and_then(|tail| {
            let c = rec.depth.fixed_orientation_centre?;
            let moved = path_shape(shapes, &tail.points)
                .and_then(|t| t.first().copied())
                .zip(path_points(rec, shapes).and_then(|p| p.first().copied()));
            Some(match moved {
                Some((to, from)) => Pivot::At(std::array::from_fn(|k| c[k] + (to[k] - from[k]))),
                None => Pivot::Unknown,
            })
        });
        let common = |object: u32, placings: Vec<Placing>| Common {
            entry: entry as u32,
            record: rec,
            object,
            part: e.part,
            placement: placement.clone(),
            projection: pr.clone(),
            placings,
            pivot,
        };
        let shaded = rec.pre_projected.is_none() && shading && rec.depth.shade_in_3d;
        let prototype = rec
            .geometry
            .shape_a
            .and_then(|k| shapes.get(&k))
            .map(Shape::data);
        match (&rec.kind, &rec.body) {
            (RecordKind::Path, _) => {
                let (world, paint, placings) = match &rec.pre_projected {
                    Some(tail) => {
                        let Some(pts) = path_shape(shapes, &tail.points) else {
                            continue;
                        };
                        let paint = DrawnPaint {
                            fill: rows(&tail.fill, tables),
                            stroke: rows(&tail.stroke, tables),
                            background_stroke: rows(&tail.background_stroke, tables),
                            stroke_width: rec.paint.stroke_width,
                            background_stroke_width: rec.paint.background_stroke_width,
                        };
                        let placing = Placing {
                            places: [IDENTITY, IDENTITY],
                            points: pts.to_vec(),
                            weight: 1.0,
                        };
                        (pts.to_vec(), paint, vec![placing])
                    }
                    None => {
                        let Some(world) = path_points(rec, shapes) else {
                            continue;
                        };
                        let Ok(paint) = drawn_paint(&rec.paint, tables) else {
                            continue;
                        };
                        let g = &rec.geometry;
                        // A blend takes each point as `(1 − t) p + t q`, `t` unbounded: each
                        // placing moves the drawn point by its own departure times `|coefficient|`.
                        let t = g.blend.as_ref().map(|b| b.t);
                        let mut placings = Vec::new();
                        if let Some(a) = g.shape_a.as_ref().and_then(|k| path_shape(shapes, k)) {
                            placings.push(Placing {
                                places: [IDENTITY, g.place_a],
                                points: a.to_vec(),
                                weight: t.map_or(1.0, |t| (1.0 - t).abs()),
                            });
                        }
                        if let Some(b) = g.blend.as_ref()
                            && let Some(q) = path_shape(shapes, &b.shape_b)
                        {
                            placings.push(Placing {
                                places: [IDENTITY, b.place_b],
                                points: q.to_vec(),
                                weight: b.t.abs(),
                            });
                        }
                        (world, paint, placings)
                    }
                };
                if let Some(item) = path_item(
                    view,
                    common(rec.object, placings),
                    0,
                    world,
                    &paint,
                    rec.joins.clone(),
                    false,
                    shaded,
                    which,
                ) {
                    out.push(Item::Path(item));
                }
            }
            (RecordKind::FaceGroup, Body::FaceGroup(faces)) => {
                let Some(ShapeData::Path { points }) = prototype else {
                    continue;
                };
                let i = e.part as usize;
                let run = &points[faces.run(points.len(), i)];
                let world: Vec<[f64; 3]> = run
                    .iter()
                    .map(|p| place(&rec.geometry.place_a, *p))
                    .collect();
                let paint = DrawnPaint {
                    fill: tables.get(&faces.fill).map(|t| vec![t.rows()[i]]),
                    stroke: faces
                        .stroke
                        .and_then(|k| tables.get(&k))
                        .map(|t| vec![t.rows()[i]]),
                    background_stroke: rows(&rec.paint.background_stroke, tables),
                    stroke_width: faces.width(rec.paint.stroke_width, i),
                    background_stroke_width: rec.paint.background_stroke_width,
                };
                let placing = Placing {
                    places: [IDENTITY, rec.geometry.place_a],
                    points: run.to_vec(),
                    weight: 1.0,
                };
                if let Some(item) = path_item(
                    view,
                    common(rec.object, vec![placing]),
                    0,
                    world,
                    &paint,
                    None,
                    true,
                    shaded,
                    which,
                ) {
                    out.push(Item::Path(item));
                }
            }
            (RecordKind::Instances, Body::Instances(body)) => match prototype {
                Some(ShapeData::Path { points }) => {
                    let one = |p: &Paint| matches!(p, Paint::One(_));
                    let (Some(fill), Some(stroke)) =
                        (tables.get(&body.fill), tables.get(&body.stroke))
                    else {
                        continue;
                    };
                    for (i, member) in body.placements.iter().enumerate() {
                        let m = compose(member, &rec.geometry.place_a);
                        let world: Vec<[f64; 3]> = points.iter().map(|p| place(&m, *p)).collect();
                        let paint = DrawnPaint {
                            fill: one(&rec.paint.fill).then(|| vec![fill.rows()[i]]),
                            stroke: one(&rec.paint.stroke).then(|| vec![stroke.rows()[i]]),
                            background_stroke: match rec.paint.background_stroke {
                                Paint::One(c) => Some(vec![c]),
                                _ => None,
                            },
                            stroke_width: rec.paint.stroke_width,
                            background_stroke_width: rec.paint.background_stroke_width,
                        };
                        let placing = Placing {
                            places: [*member, rec.geometry.place_a],
                            points: points.clone(),
                            weight: 1.0,
                        };
                        let object = body.objects.get(i).copied().unwrap_or(rec.object);
                        if let Some(item) = path_item(
                            view,
                            common(object, vec![placing]),
                            i as u32,
                            world,
                            &paint,
                            None,
                            false,
                            shaded,
                            which,
                        ) {
                            out.push(Item::Path(item));
                        }
                    }
                }
                Some(ShapeData::Mesh {
                    positions,
                    triangles,
                    normals,
                    ..
                }) => {
                    if !matches!(rec.paint.fill, Paint::One(_)) {
                        continue;
                    }
                    let Some(fill) = tables.get(&body.fill) else {
                        continue;
                    };
                    let per = triangles.len() as u32;
                    for (i, member) in body.placements.iter().enumerate() {
                        let row = fill.rows()[i];
                        let alpha = row[3] * rec.opacity;
                        if (alpha_byte(alpha) == 0) != invisible {
                            continue;
                        }
                        let m = compose(member, &rec.geometry.place_a);
                        let placing = Placing {
                            places: [*member, rec.geometry.place_a],
                            points: positions.clone(),
                            weight: 1.0,
                        };
                        let object = body.objects.get(i).copied().unwrap_or(rec.object);
                        if let Some(item) = mesh_item(
                            view,
                            common(object, vec![placing]),
                            per * i as u32,
                            positions,
                            triangles,
                            normals.as_ref(),
                            &m,
                            row,
                            shaded,
                        ) {
                            out.push(Item::Mesh(item));
                        }
                    }
                }
                _ => {}
            },
            (RecordKind::Mesh, _) => {
                let Some(ShapeData::Mesh {
                    positions,
                    triangles,
                    normals,
                    ..
                }) = prototype
                else {
                    continue;
                };
                let fill = rows(&rec.paint.fill, tables);
                let Some(colour) = fill.as_ref().and_then(|r| r.first()) else {
                    continue;
                };
                if (alpha_byte(colour[3] * rec.opacity) == 0) != invisible {
                    continue;
                }
                let placing = Placing {
                    places: [IDENTITY, rec.geometry.place_a],
                    points: positions.clone(),
                    weight: 1.0,
                };
                if let Some(item) = mesh_item(
                    view,
                    common(rec.object, vec![placing]),
                    0,
                    positions,
                    triangles,
                    normals.as_ref(),
                    &rec.geometry.place_a,
                    *colour,
                    shaded,
                ) {
                    out.push(Item::Mesh(item));
                }
            }
            (RecordKind::Points, Body::Points(body)) => {
                let Some(ShapeData::Path { points: own }) = prototype else {
                    continue;
                };
                let (positions, places) = match &rec.pre_projected {
                    Some(tail) => match path_shape(shapes, &tail.points) {
                        Some(p) => (p, [IDENTITY, IDENTITY]),
                        None => continue,
                    },
                    None => (own.as_slice(), [IDENTITY, rec.geometry.place_a]),
                };
                let Some(colours) = tables.get(&body.colors) else {
                    continue;
                };
                let placing = Placing {
                    places,
                    points: positions.to_vec(),
                    weight: 1.0,
                };
                let c = common(rec.object, vec![placing]);
                let side = block_side(body.radius);
                let mut pts = Vec::new();
                for (i, p) in positions.iter().enumerate() {
                    let Some(colour) = colours.rows().get(i) else {
                        continue;
                    };
                    // CE's own truncated alpha byte: no block where it is 0.
                    if (255.0 * (colour[3] * rec.opacity)) as i64 <= 0 {
                        continue;
                    }
                    let placed = place(&places[1], *p);
                    if placed.iter().any(|v| !v.is_finite()) {
                        continue;
                    }
                    let shown = c.placement.show(c.projection.as_ref(), placed);
                    let at = map_point_cloud(view, shown);
                    let key_nan = matches!(c.placement, Placement::Projected)
                        && c.projection
                            .as_ref()
                            .is_some_and(|pr| pr.key(shown[2]).is_nan());
                    if at.iter().any(|v| !v.is_finite() || v.abs() > 4e18) || key_nan {
                        continue;
                    }
                    pts.push(BlockPoint {
                        index: i as u32,
                        placed,
                        shown,
                        at,
                    });
                }
                if side >= 1 && !pts.is_empty() {
                    out.push(Item::Points(PointsItem {
                        common: c,
                        colours: colours.rows().to_vec(),
                        count: positions.len() as u32,
                        side,
                        points: pts,
                    }));
                }
            }
            _ => {}
        }
    }
    out
}

/// One path drawn, or `None` where it draws nothing: fewer than four points, a non-finite placed
/// point, or no layer that paints.
#[allow(clippy::too_many_arguments)]
fn path_item<'a>(
    view: &View,
    common: Common<'a>,
    index: u32,
    world: Vec<[f64; 3]>,
    paint: &DrawnPaint,
    joins: Option<Vec<u8>>,
    face: bool,
    shaded: bool,
    which: Which,
) -> Option<PathItem<'a>> {
    if world.len() < 4 || world.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let rec = common.record;
    let multiple = view.cairo_line_width_multiple;
    let layer = |rows: &Option<Vec<[f64; 4]>>, width: Option<f64>| -> Option<f64> {
        let rows = rows.as_ref()?;
        if !paints(rows, rec.opacity, shaded) {
            return None;
        }
        match width {
            None => Some(0.0),
            Some(w) => {
                let w = w * multiple;
                (w > 0.0).then_some(w)
            }
        }
    };
    let mut layers = [
        layer(
            &paint.background_stroke,
            Some(paint.background_stroke_width),
        ),
        layer(&paint.fill, None),
        layer(&paint.stroke, Some(paint.stroke_width)),
    ];
    let paints_nothing = layers.iter().all(Option::is_none);
    if paints_nothing != (which == Which::Invisible) {
        return None;
    }
    if paints_nothing {
        layers = [None, Some(0.0), None];
    }
    let shown = world
        .iter()
        .map(|p| common.placement.show(common.projection.as_ref(), *p))
        .collect();
    Some(PathItem {
        index,
        shown,
        world,
        layers,
        joint: rec.paint.joint,
        cap: rec.paint.cap,
        dash: rec.dash,
        joins,
        face,
        rows: [
            paint.background_stroke.clone(),
            paint.fill.clone(),
            paint.stroke.clone(),
        ],
        shaded,
        map: FrameMap::of_paths(view),
        common,
    })
}

/// One mesh drawn under the placement `m`, or `None` where a placed vertex is not finite.
#[allow(clippy::too_many_arguments)]
fn mesh_item<'a>(
    view: &View,
    common: Common<'a>,
    sub: u32,
    positions: &[[f64; 3]],
    triangles: &[[u32; 3]],
    normals: Option<&Vec<[f64; 3]>>,
    m: &Affine,
    fill: [f64; 4],
    shaded: bool,
) -> Option<MeshItem<'a>> {
    let placed: Vec<[f64; 3]> = positions.iter().map(|p| place(m, *p)).collect();
    if placed.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let shown: Vec<[f64; 3]> = placed
        .iter()
        .map(|p| common.placement.show(common.projection.as_ref(), *p))
        .collect();
    let corners = |t: &[u32; 3], of: &[[f64; 3]]| t.map(|v| of[v as usize]);
    let placed_corners = triangles.iter().map(|t| corners(t, &placed)).collect();
    // A carried normal goes through the placement's cofactors, then to unit length (contract §9).
    let normals = normals.filter(|n| n.len() == positions.len()).map(|n| {
        let turned: Vec<[f64; 3]> = n
            .iter()
            .map(|v| crate::colour::place_normal(m, *v))
            .collect();
        triangles.iter().map(|t| corners(t, &turned)).collect()
    });
    let triangles = triangles.iter().map(|t| corners(t, &shown)).collect();
    Some(MeshItem {
        sub,
        triangles,
        placed: placed_corners,
        fill,
        normals,
        shaded,
        map: FrameMap::of_paths(view),
        common,
    })
}
