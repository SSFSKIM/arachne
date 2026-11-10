//! Which surface sheets a camera-facing label goes behind: one decision per label and sheet, by
//! the sheet's depth on the label's centre ray where it covers the centre, else over the whole
//! overlap; open where the error bounds, a fold or a second sheet leave it undecided.
//!
//! The camera looks straight down the z axis (no rotation), so a point's screen position is its
//! x and y scaled by the perspective, and a larger z is nearer.

mod common;

use arachne_core::label_plan::{Open, Outcome, Pair};
use arachne_core::stream::Stream;
use arachne_core::xf::*;
use common::three_d_view;

const W: u32 = 160;
const H: u32 = 90;
const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
const BLUE: [f64; 4] = [0.2, 0.3, 0.9, 1.0];
const SQUARE: &str = "arachne.mobject.geometry.polygram.Square";
const CUBE: &str = "arachne.mobject.three_d.three_dimensions.Cube";
const SURFACE: &str = "arachne.mobject.three_d.three_dimensions.Surface";
const FACE: &str = "arachne.mobject.three_d.three_dimensions.ThreeDVMobject";
const LINE: &str = "arachne.mobject.geometry.line.Line";
const GLYPH: &str = "arachne.mobject.svg.svg_mobject.VMobjectFromSVGPath";
const TEX: &str = "arachne.mobject.text.tex_mobject.MathTex";

fn view() -> View {
    let mut v = three_d_view(W, H);
    v.camera = Camera::ThreeD(ThreeDCamera {
        phi: 0.0,
        theta: -std::f64::consts::FRAC_PI_2,
        gamma: 0.0,
        focal_distance: 20.0,
        zoom: 1.0,
        exponential_projection: false,
        light_source: [-7.0, -9.0, 10.0],
        shading_factor: 0.2,
        should_apply_shading: false,
    });
    v.frame_center = [0.0; 3];
    v.mapping_centre = [0.0; 3];
    v
}

/// A closed polygon through `corners` in CE's cubic layout.
fn polygon(corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for i in 0..corners.len() {
        let (a, b) = (corners[i], corners[(i + 1) % corners.len()]);
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

fn node(token: u32, class: &str) -> SourceNode {
    SourceNode {
        token,
        class: class.into(),
    }
}

/// A filled polygon in the world drawn from `source`.
fn face(
    slot: u32,
    corners: &[[f64; 3]],
    rgba: [f64; 4],
    source: &[(u32, &str)],
) -> (Shape, Record) {
    let shape = Shape::new(ShapeData::Path {
        points: polygon(corners),
    });
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
    r.paint.fill = Paint::One(rgba);
    r.depth.space = Space::World;
    r.source = vec![source.iter().map(|(t, c)| node(*t, c)).collect()];
    (shape, r)
}

/// The rectangle `x0..x1` by `y0..y1`, wound counter-clockwise in the world (`front`) or
/// clockwise, each corner at the depth `z` gives it.
fn rect(x: [f64; 2], y: [f64; 2], front: bool, z: impl Fn(f64, f64) -> f64) -> Vec<[f64; 3]> {
    let mut c = vec![
        [x[0], y[0], z(x[0], y[0])],
        [x[1], y[0], z(x[1], y[0])],
        [x[1], y[1], z(x[1], y[1])],
        [x[0], y[1], z(x[0], y[1])],
    ];
    if !front {
        c.reverse();
    }
    c
}

/// A glyph of facing group `group`: a white rectangle of half extents `half` about `at`, at the
/// fixed orientation of `centre`.
fn glyph(slot: u32, group: u32, centre: [f64; 3], at: [f64; 2], half: [f64; 2]) -> (Shape, Record) {
    let z = centre[2];
    let corners = rect(
        [at[0] - half[0], at[0] + half[0]],
        [at[1] - half[1], at[1] + half[1]],
        true,
        |_, _| z,
    );
    let (shape, mut r) = face(
        slot,
        &corners,
        WHITE,
        &[(50 + slot, GLYPH), (90 + group, TEX)],
    );
    r.depth.fixed_orientation_centre = Some(centre);
    r.facing_group = group;
    (shape, r)
}

/// A one-glyph label of group 7 at `centre`, `half` its glyph's half extents.
fn label(centre: [f64; 3], half: [f64; 2]) -> (Shape, Record) {
    glyph(1, 7, centre, [centre[0], centre[1]], half)
}

fn stream(items: &[(Shape, Record)]) -> Stream {
    let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
    let d = arachne_core::probes::first_delta(view(), items, &order);
    let mut s = Stream::new();
    s.apply(&d).expect("the delta applies");
    s
}

fn pairs(items: &[(Shape, Record)]) -> Vec<Pair> {
    stream(items)
        .label_plan(W, H)
        .expect("the frame draws")
        .expect("the frame draws a label")
        .pairs
}

/// The one pair the label makes, with its outcome and whether the sheet covers the centre.
fn one(items: &[(Shape, Record)]) -> (Outcome, bool) {
    let found = pairs(items);
    assert_eq!(found.len(), 1, "{found:?}");
    (found[0].outcome, found[0].centre)
}

fn square_at(slot: u32, z: f64, front: bool, owner: u32) -> (Shape, Record) {
    let corners = rect([-1.0, 1.0], [-1.0, 1.0], front, |_, _| z);
    face(slot, &corners, BLUE, &[(10 + slot, SQUARE), (owner, CUBE)])
}

#[test]
fn a_frame_without_a_label_has_no_plan() {
    let s = stream(&[square_at(2, 1.0, true, 1)]);
    assert!(s.label_plan(W, H).unwrap().is_none());
}

#[test]
fn a_nearer_sheet_covering_the_centre_qualifies_and_a_farther_one_does_not() {
    let near = [
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        square_at(2, 1.0, true, 1),
    ];
    assert_eq!(one(&near), (Outcome::Qualifies, true));
    let far = [
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        square_at(2, -1.0, true, 1),
    ];
    assert_eq!(one(&far), (Outcome::NotNearer, true));
}

/// "A translucent cube ... a label at 8 goes behind both qualifying sheets ... A label at 4 goes
/// behind the front sheet and over the rear sheet."
#[test]
fn a_label_inside_a_translucent_cube_is_behind_its_front_alone_and_one_behind_it_behind_both() {
    let translucent = [0.2, 0.3, 0.9, 0.5];
    let cube = |slot: u32, z: f64, front: bool| {
        let corners = rect([-1.0, 1.0], [-1.0, 1.0], front, |_, _| z);
        face(
            slot,
            &corners,
            translucent,
            &[(10 + slot, SQUARE), (1, CUBE)],
        )
    };
    let outcomes = |z: f64| -> Vec<(Outcome, bool)> {
        let found = pairs(&[
            label([0.0, 0.0, z], [0.5, 0.3]),
            cube(2, 1.0, true),
            cube(3, -1.0, false),
        ]);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_ne!(found[0].sheet, found[1].sheet);
        found.iter().map(|p| (p.outcome, p.centre)).collect()
    };
    let inside = outcomes(0.0);
    assert!(inside.contains(&(Outcome::Qualifies, true)), "{inside:?}");
    assert!(inside.contains(&(Outcome::NotNearer, true)), "{inside:?}");
    assert_eq!(outcomes(-2.0), vec![(Outcome::Qualifies, true); 2]);
}

/// "L is at depth 10, centre x=0; S exists only at x=0.5…1 with depth 9+4x. Its infinite plane is
/// nearer at the centre, but all actual overlap is farther."
#[test]
fn a_sheet_missing_the_centre_is_decided_over_its_actual_overlap_not_its_plane() {
    // Depth `9 + 4x` as a distance, here `z = -(0.9 + 0.4 x)` against the label's `-1`.
    let sheet = rect([0.5, 1.0], [-0.5, 0.5], true, |x, _| -(0.9 + 0.4 * x));
    let items = [
        label([0.0, 0.0, -1.0], [1.2, 0.3]),
        face(2, &sheet, BLUE, &[(12, SQUARE)]),
    ];
    assert_eq!(one(&items), (Outcome::NotNearer, false));
}

/// "A genuine straddle is not shown nearer and does not qualify ... State the cost: a sheet mostly
/// in front, but with a farther sliver, lets the label show through it."
#[test]
fn a_sheet_crossing_the_label_s_depth_away_from_the_centre_does_not_qualify() {
    let sheet = rect([0.2, 1.0], [-0.5, 0.5], true, |x, _| 0.5 - x * 0.7);
    let items = [
        label([0.0, 0.0, 0.0], [1.2, 0.3]),
        face(2, &sheet, BLUE, &[(12, SQUARE)]),
    ];
    assert_eq!(one(&items), (Outcome::NotNearer, false));
}

/// A sheet at the label's own depth, missing its centre: the bounds cannot show it nearer or
/// farther, so the pair is open by name, and the label stays in front.
#[test]
fn a_sheet_the_bounds_cannot_place_is_an_open_interval() {
    let sheet = rect([0.3, 1.0], [-0.5, 0.5], true, |_, _| 0.0);
    let items = [
        label([0.0, 0.0, 0.0], [1.2, 0.3]),
        face(2, &sheet, BLUE, &[(12, SQUARE)]),
    ];
    assert_eq!(one(&items), (Outcome::Open(Open::Interval), false));
}

/// "Shown strictly nearer qualifies the whole pair, even if S crosses behind the label depth
/// elsewhere in their overlap."
#[test]
fn a_sheet_nearer_on_the_centre_ray_qualifies_though_it_crosses_the_label_s_depth() {
    let sheet = rect([-1.5, 1.5], [-1.0, 1.0], true, |x, _| 0.3 - 0.4 * x);
    let items = [
        label([0.0, 0.0, 0.0], [1.4, 0.3]),
        face(2, &sheet, BLUE, &[(12, SQUARE)]),
    ];
    assert_eq!(one(&items), (Outcome::Qualifies, true));
}

/// The centre may lie outside the label's glyphs: the sheet covering it there still decides on
/// that ray.
#[test]
fn the_centre_ray_decides_where_no_glyph_covers_the_centre() {
    let items = [
        glyph(1, 7, [0.0, 0.0, 0.0], [-0.8, 0.0], [0.3, 0.3]),
        glyph(3, 7, [0.0, 0.0, 0.0], [0.8, 0.0], [0.3, 0.3]),
        square_at(2, 1.0, true, 1),
    ];
    assert_eq!(one(&items), (Outcome::Qualifies, true));
}

#[test]
fn a_wholly_transparent_sheet_and_a_line_are_no_sheets() {
    let clear = rect([-1.0, 1.0], [-1.0, 1.0], true, |_, _| 1.0);
    let items = [
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        face(2, &clear, [0.2, 0.3, 0.9, 0.0], &[(12, SQUARE)]),
        face(3, &clear, BLUE, &[(13, LINE)]),
    ];
    assert!(pairs(&items).is_empty());
}

/// "Never decide per triangulation face": a sheet cut into more faces is the same sheet.
#[test]
fn a_sheet_cut_into_faces_is_one_sheet() {
    let half = |x: [f64; 2]| rect(x, [-1.0, 1.0], true, |_, _| 1.0);
    let items = [
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        face(2, &half([-1.0, 0.0]), BLUE, &[(12, FACE), (1, SURFACE)]),
        face(3, &half([0.0, 1.0]), BLUE, &[(13, FACE), (1, SURFACE)]),
    ];
    assert_eq!(one(&items), (Outcome::Qualifies, true));
}

/// A surface folded over itself on screen: two faces of one sheet covering the label's cells
/// twice.
#[test]
fn a_sheet_folded_over_itself_is_open() {
    let a = rect([-1.0, 1.0], [-1.0, 1.0], true, |_, _| 1.0);
    // Sharing two corners with `a` and wound its way, on its plane, inside it.
    let b = [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 0.8, 1.0]];
    let items = [
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        face(2, &a, BLUE, &[(12, FACE), (1, SURFACE)]),
        face(3, &b, BLUE, &[(13, FACE), (1, SURFACE)]),
    ];
    assert_eq!(one(&items).0, Outcome::Open(Open::Folded));
}

/// Two sheets of one surface facing one way that both meet the label: neither is decided.
#[test]
fn two_sheets_of_one_surface_facing_one_way_are_open() {
    let left = rect([-1.5, -0.2], [-1.0, 1.0], true, |_, _| 1.0);
    let right = rect([0.2, 1.5], [-1.0, 1.0], true, |_, _| 1.0);
    let items = [
        label([0.0, 0.0, 0.0], [1.0, 0.3]),
        face(2, &left, BLUE, &[(12, FACE), (1, SURFACE)]),
        face(3, &right, BLUE, &[(13, FACE), (1, SURFACE)]),
    ];
    let found = pairs(&items);
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(
        found
            .iter()
            .all(|p| p.outcome == Outcome::Open(Open::Disconnected))
    );
}
