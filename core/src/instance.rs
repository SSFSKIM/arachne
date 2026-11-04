//! The instance path: `instances` and `points` records drawn as their members, through the same
//! rasterizers as the records they stand for.
//!
//! - An `instances` record's instance `i` is its prototype (the record's shape and paint block)
//!   placed by `placements[i]` after the record's own placement, its fill and stroke colours row
//!   `i` of the record's tables where the prototype fills or strokes, at the record's place in the
//!   draw order, tying above instance `i - 1` as the run's `i`-th leaf did.
//! - A `points` record's point `i` is a disc of its colour row `i`, at the point's place, tying
//!   above point `i - 1`.
//!
//! Members are drawn one by one: correct, not yet fast.

use crate::mesh::MeshDraw;
use crate::path::{DiscDraw, PathDraw};
use crate::xf::*;

/// The colour a paint resolves to, when it is one colour (other paints are refused before
/// drawing).
pub fn one(paint: &Paint) -> Option<[f64; 4]> {
    match paint {
        Paint::One(c) => Some(*c),
        _ => None,
    }
}

/// The path a `path` record draws.
pub fn path_record<'a>(record: &'a Record, points: &'a [[f64; 3]], order: u32) -> PathDraw<'a> {
    PathDraw {
        record,
        points,
        place: record.geometry.place_a,
        fill: one(&record.paint.fill),
        stroke: one(&record.paint.stroke),
        background_stroke: one(&record.paint.background_stroke),
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
            points,
            place: compose(placement, &record.geometry.place_a),
            fill: one(&p.fill).map(|_| fill.rows[i]),
            stroke: one(&p.stroke).map(|_| stroke.rows[i]),
            background_stroke: one(&p.background_stroke),
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
            fill: fill.rows[i],
            order,
            sub: per * i as u32,
        });
    }
}

/// A `points` record's discs.
pub fn discs<'a>(
    record: &'a Record,
    positions: &'a [[f64; 3]],
    colours: &'a Stops,
    order: u32,
    out: &mut Vec<DiscDraw<'a>>,
) {
    for (i, point) in positions.iter().enumerate() {
        out.push(DiscDraw {
            record,
            point: *point,
            colour: colours.rows[i],
            order,
            sub: i as u32,
        });
    }
}
