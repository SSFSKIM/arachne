//! The instance path: `instances` and `points` records drawn as their members, through the same
//! rasterizers as the records they stand for.
//!
//! - An `instances` record's instance `i` is its prototype (the record's shape and paint block)
//!   placed by `placements[i]` after the record's own placement, its fill and stroke colours row
//!   `i` of the record's tables where the prototype fills or strokes, at the record's place in the
//!   draw order, tying above instance `i - 1` as the run's `i`-th leaf did.
//! - A `points` record draws its points as CE's blocks (`points.rs`), in the same instance path:
//!   point `i` takes colour row `i`.
//!
//! Members are drawn one by one: correct, not yet fast.

use std::borrow::Cow;

use crate::geometry::{DrawnPaint, rows};
use crate::mesh::MeshDraw;
use crate::path::PathDraw;
use crate::points::PointsDraw;
use crate::project::produced_pivot;
use crate::xf::*;

/// A `points` record's points, drawn as CE's blocks: `positions` its shape's, or its tail's points
/// when it is pre-projected.
pub fn points<'a>(
    record: &'a Record,
    own: &[[f64; 3]],
    positions: &'a [[f64; 3]],
    colours: &'a Stops,
    order: u32,
) -> PointsDraw<'a> {
    let pre_projected = record.pre_projected.is_some();
    PointsDraw {
        record,
        positions,
        colours: colours.rows(),
        pre_projected,
        pivot: pre_projected
            .then(|| produced_pivot(record, own, positions))
            .flatten(),
        order,
    }
}

/// The colour a paint resolves to, when it is one colour (other paints are refused before
/// drawing).
pub fn one(paint: &Paint) -> Option<[f64; 4]> {
    match paint {
        Paint::One(c) => Some(*c),
        _ => None,
    }
}

/// The path a `path` record draws: `points` placed by `place` (the shape and the record's
/// placement, or what a reveal or a blend made of it, placed already), painted by `paint` (the
/// record's paint, or a paint blend's mix: [`crate::geometry::drawn_paint`]), its gradient between
/// the record's ends.
pub fn path_record<'a>(
    record: &'a Record,
    points: Cow<'a, [[f64; 3]]>,
    place: Affine,
    paint: DrawnPaint,
    order: u32,
) -> PathDraw<'a> {
    PathDraw {
        record,
        points,
        place,
        places: [IDENTITY, place],
        fill: paint.fill,
        stroke: paint.stroke,
        background_stroke: paint.background_stroke,
        ends: record.paint.gradient,
        surface: false,
        pre_projected: false,
        pivot: None,
        stroke_width: paint.stroke_width,
        background_stroke_width: paint.background_stroke_width,
        order,
        sub: 0,
        which: 0,
    }
}

/// The path a pre-projected `path` record draws: the points its camera placed (`placed`, its tail's
/// shape) painted in the colours that camera painted with, its gradient between the ends it placed
/// (`tail`), at the record's stroke widths.
pub fn produced_path<'a, 't>(
    record: &'a Record,
    own: &[[f64; 3]],
    placed: &'a [[f64; 3]],
    tail: &PreProjected,
    table: impl Fn(&Key) -> &'t Stops + Copy,
    order: u32,
) -> PathDraw<'a> {
    let p = &record.paint;
    PathDraw {
        record,
        points: Cow::Borrowed(placed),
        place: IDENTITY,
        places: [IDENTITY, IDENTITY],
        fill: rows(&tail.fill, table),
        stroke: rows(&tail.stroke, table),
        background_stroke: rows(&tail.background_stroke, table),
        ends: tail.gradient,
        surface: false,
        pre_projected: true,
        pivot: produced_pivot(record, own, placed),
        stroke_width: p.stroke_width,
        background_stroke_width: p.background_stroke_width,
        order,
        sub: 0,
        which: 0,
    }
}

/// An `instances` record's members whose prototype is a path.
pub fn path_members<'a>(
    record: &'a Record,
    points: &'a [[f64; 3]],
    body: &'a Instances,
    fill: &'a Stops,
    stroke: &'a Stops,
    order: u32,
    out: &mut Vec<PathDraw<'a>>,
) {
    let p = &record.paint;
    for (i, placement) in body.placements.iter().enumerate() {
        out.push(PathDraw {
            record,
            points: Cow::Borrowed(points),
            place: compose(placement, &record.geometry.place_a),
            places: [*placement, record.geometry.place_a],
            fill: one(&p.fill).map(|_| vec![fill.rows()[i]]),
            stroke: one(&p.stroke).map(|_| vec![stroke.rows()[i]]),
            background_stroke: one(&p.background_stroke).map(|c| vec![c]),
            ends: None,
            surface: false,
            pre_projected: false,
            pivot: None,
            stroke_width: p.stroke_width,
            background_stroke_width: p.background_stroke_width,
            order,
            sub: 3 * i as u32,
            which: i,
        });
    }
}

/// Face `part` of a `face_group` record, as CE draws each face of a surface: the record's points
/// cut into one equal run per face, face `part`'s run through the record's placement, filled and
/// stroked by row `part` of the face tables (found by `table`), with the record's background
/// stroke and widths; a face of a surface for depth ([`PathDraw::surface`]).
pub fn face<'a, 't>(
    record: &'a Record,
    points: &'a [[f64; 3]],
    faces: &FaceGroup,
    part: u32,
    table: impl Fn(&Key) -> &'t Stops + Copy,
    order: u32,
) -> PathDraw<'a> {
    let size = points.len() / faces.uv.len();
    let i = part as usize;
    let p = &record.paint;
    PathDraw {
        record,
        points: Cow::Borrowed(&points[i * size..(i + 1) * size]),
        place: record.geometry.place_a,
        places: [IDENTITY, record.geometry.place_a],
        fill: Some(vec![table(&faces.fill).rows()[i]]),
        stroke: faces.stroke.map(|key| vec![table(&key).rows()[i]]),
        background_stroke: rows(&p.background_stroke, table),
        ends: None,
        surface: true,
        pre_projected: false,
        pivot: None,
        stroke_width: p.stroke_width,
        background_stroke_width: p.background_stroke_width,
        order,
        sub: 0,
        which: i,
    }
}

/// An `instances` record's members whose prototype is a mesh.
#[allow(clippy::too_many_arguments)]
pub fn mesh_members<'a>(
    record: &'a Record,
    positions: &'a [[f64; 3]],
    triangles: &'a [[u32; 3]],
    normals: Option<&'a [[f64; 3]]>,
    body: &'a Instances,
    fill: &'a Stops,
    order: u32,
    out: &mut Vec<MeshDraw<'a>>,
) {
    if one(&record.paint.fill).is_none() {
        return;
    }
    let per = triangles.len() as u32;
    for (i, placement) in body.placements.iter().enumerate() {
        out.push(MeshDraw {
            record,
            positions,
            triangles,
            normals,
            place: compose(placement, &record.geometry.place_a),
            places: [*placement, record.geometry.place_a],
            fill: fill.rows()[i],
            order,
            sub: per * i as u32,
        });
    }
}
