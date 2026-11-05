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

use crate::mesh::MeshDraw;
use crate::path::PathDraw;
use crate::points::PointsDraw;
use crate::xf::*;

/// A `points` record's points, drawn as CE's blocks.
pub fn points<'a>(
    record: &'a Record,
    positions: &'a [[f64; 3]],
    colours: &'a Stops,
    order: u32,
) -> PointsDraw<'a> {
    PointsDraw {
        record,
        positions,
        colours: colours.rows(),
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
/// record's paint, or a paint blend's mix: [`crate::geometry::drawn_paint`]).
pub fn path_record<'a>(
    record: &'a Record,
    points: Cow<'a, [[f64; 3]]>,
    place: Affine,
    paint: &PaintBlock,
    order: u32,
) -> PathDraw<'a> {
    PathDraw {
        record,
        points,
        place,
        fill: one(&paint.fill),
        stroke: one(&paint.stroke),
        background_stroke: one(&paint.background_stroke),
        stroke_width: paint.stroke_width,
        background_stroke_width: paint.background_stroke_width,
        order,
        sub: 0,
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
            fill: one(&p.fill).map(|_| fill.rows()[i]),
            stroke: one(&p.stroke).map(|_| stroke.rows()[i]),
            background_stroke: one(&p.background_stroke),
            stroke_width: p.stroke_width,
            background_stroke_width: p.background_stroke_width,
            order,
            sub: 3 * i as u32,
        });
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
            fill: fill.rows()[i],
            order,
            sub: per * i as u32,
        });
    }
}
