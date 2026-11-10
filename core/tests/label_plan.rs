//! Which surface sheets a camera-facing label goes behind: one decision per label and sheet, by
//! the sheet's depth on the label's centre ray where it covers the centre, else over the whole
//! overlap; open where the error bounds, a fold or a second sheet leave it undecided.
//!
//! The camera looks straight down the z axis (no rotation), so a point's screen position is its
//! x and y scaled by the perspective, and a larger z is nearer.

mod common;

use arachne_core::composite::{background_pixel, blend, premultiply};
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

// Placement: what the frame draws once the plan is made. Pixel (80, 45) is the label's centre's,
// inside every glyph here; pixel (93, 45) lies 1.2 units to its right.

fn px(frame: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = (y * W + x) as usize * 4;
    frame[i..i + 4].try_into().unwrap()
}

fn drawn(items: &[(Shape, Record)]) -> Vec<u8> {
    stream(items).draw(W, H, 1).expect("the frame draws")
}

fn background() -> [u8; 4] {
    background_pixel([0.0, 0.0, 0.0, 1.0])
}

/// `layers`, front first, each covering the pixel whole, over the background.
fn stacked(layers: &[[f64; 4]]) -> [u8; 4] {
    layers
        .iter()
        .rev()
        .fold(background(), |under, c| blend(premultiply(*c), 255, under))
}

/// A line of class `Line` along y = 0 at depth `z`, its stroke 4.5 pixels wide.
fn line_at(slot: u32, z: f64, rgba: [f64; 4]) -> (Shape, Record) {
    let shape = Shape::new(ShapeData::Path {
        points: polygon(&[[-2.0, 0.0, z], [2.0, 0.0, z]])[..4].to_vec(),
    });
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
    r.paint.stroke = Paint::One(rgba);
    r.paint.stroke_width = 40.0;
    r.depth.space = Space::World;
    r.source = vec![vec![node(10 + slot, LINE)]];
    (shape, r)
}

#[test]
fn a_label_behind_a_nearer_sheet_is_hidden_where_the_sheet_covers_it() {
    let behind = drawn(&[
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        square_at(2, 1.0, true, 1),
    ]);
    assert_eq!(px(&behind, 80, 45), stacked(&[BLUE]));
    let front = drawn(&[
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        square_at(2, -1.0, true, 1),
    ]);
    assert_eq!(px(&front, 80, 45), stacked(&[WHITE]));
}

/// "No depth-induced split": a sheet nearer on the centre ray hides the whole label, also where it
/// passes behind the label's depth, which the plate model drew the label in front of.
#[test]
fn a_qualifying_sheet_hides_the_label_also_where_it_crosses_behind_its_depth() {
    let sheet = rect([-1.5, 1.5], [-1.0, 1.0], true, |x, _| 0.3 - 0.4 * x);
    let frame = drawn(&[
        label([0.0, 0.0, 0.0], [1.4, 0.3]),
        face(2, &sheet, BLUE, &[(12, SQUARE)]),
    ]);
    assert_eq!(px(&frame, 80, 45), stacked(&[BLUE]));
    assert_eq!(px(&frame, 93, 45), stacked(&[BLUE]));
}

/// "Distinguish a label inside a translucent cube from one behind it."
#[test]
fn a_label_inside_a_translucent_cube_draws_between_its_sheets_and_one_behind_it_under_both() {
    let glass = [0.2, 0.3, 0.9, 0.5];
    let red = [0.9, 0.2, 0.1, 0.5];
    let cube = |z: f64| {
        drawn(&[
            label([0.0, 0.0, z], [0.5, 0.3]),
            face(
                2,
                &rect([-1.0, 1.0], [-1.0, 1.0], true, |_, _| 1.0),
                glass,
                &[(12, SQUARE), (1, CUBE)],
            ),
            face(
                3,
                &rect([-1.0, 1.0], [-1.0, 1.0], false, |_, _| -1.0),
                red,
                &[(13, SQUARE), (1, CUBE)],
            ),
        ])
    };
    assert_eq!(px(&cube(0.0), 80, 45), stacked(&[glass, WHITE]));
    let translucent = [1.0, 1.0, 1.0, 0.6];
    let behind = drawn(&[
        {
            let (s, mut r) = label([0.0, 0.0, -2.0], [0.5, 0.3]);
            r.paint.fill = Paint::One(translucent);
            (s, r)
        },
        face(
            2,
            &rect([-1.0, 1.0], [-1.0, 1.0], true, |_, _| 1.0),
            glass,
            &[(12, SQUARE), (1, CUBE)],
        ),
        face(
            3,
            &rect([-1.0, 1.0], [-1.0, 1.0], false, |_, _| -1.0),
            red,
            &[(13, SQUARE), (1, CUBE)],
        ),
    ]);
    assert_eq!(px(&behind, 80, 45), stacked(&[glass, red, translucent]));
}

/// "An unbacked label wrongly hidden by a line must fail": lines never push a label back.
#[test]
fn a_nearer_line_never_hides_a_label_no_sheet_backs() {
    let orange = [1.0, 0.5, 0.0, 1.0];
    let frame = drawn(&[label([0.0, 0.0, 0.0], [0.5, 0.3]), line_at(2, 1.0, orange)]);
    assert_eq!(px(&frame, 80, 45), stacked(&[WHITE]));
}

/// "A surface-backed label incorrectly drawn over a line above that surface must fail": the line
/// over the sheet the label goes behind covers the label; a line under that sheet does not.
#[test]
fn a_label_behind_a_sheet_takes_the_sheet_s_place_among_lines() {
    let orange = [1.0, 0.5, 0.0, 1.0];
    let glass = [0.2, 0.3, 0.9, 0.5];
    let plane = face(
        2,
        &rect([-1.0, 1.0], [-1.0, 1.0], true, |_, _| 1.0),
        glass,
        &[(12, SQUARE)],
    );
    let over = drawn(&[
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        plane.clone(),
        line_at(3, 2.0, orange),
    ]);
    assert_eq!(px(&over, 80, 45), stacked(&[orange]));
    let under = drawn(&[
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        plane,
        line_at(3, 0.5, orange),
    ]);
    assert_eq!(px(&under, 80, 45), stacked(&[glass, WHITE]));
}

/// "L1 behind S and L2 in front of S override their former CE ordering": where the two labels
/// overlap on the sheet, the one the sheet backs is behind it though drawn later.
#[test]
fn placement_overrides_the_labels_draw_order() {
    let yellow = [1.0, 0.9, 0.1, 1.0];
    let (shape, mut l2) = glyph(1, 8, [0.0, 0.0, 2.0], [0.0, 0.0], [0.5, 0.3]);
    l2.paint.fill = Paint::One(yellow);
    let l1 = glyph(3, 7, [0.0, 0.0, 0.0], [0.0, 0.0], [0.5, 0.3]);
    let sheet = face(
        2,
        &rect([-1.0, 1.0], [-1.0, 1.0], true, |_, _| 1.0),
        BLUE,
        &[(12, SQUARE)],
    );
    // L2 is nearer than the sheet: it stays in front of the world, L1 behind the sheet, though
    // L1 is drawn after it.
    let frame = drawn(&[(shape, l2), sheet, l1]);
    assert_eq!(px(&frame, 80, 45), stacked(&[yellow]));
}

#[test]
fn a_placed_frame_is_the_same_on_any_number_of_threads() {
    let sheet = rect([-1.5, 1.5], [-1.0, 1.0], true, |x, _| 0.3 - 0.4 * x);
    let s = stream(&[
        label([0.0, 0.0, 0.0], [1.4, 0.3]),
        face(2, &sheet, [0.2, 0.3, 0.9, 0.5], &[(12, SQUARE)]),
        line_at(3, 2.0, [1.0, 0.5, 0.0, 1.0]),
    ]);
    let one = s.draw(W, H, 1).unwrap();
    for threads in [2, 4, 7] {
        assert_eq!(s.draw(W, H, threads).unwrap(), one, "{threads} threads");
    }
}

/// "SOURCE-only changes must preserve every RGBA byte, including label scenes with the new policy
/// disabled": placement off, a label scene draws what it draws without its facing groups.
#[test]
fn placement_off_draws_the_frame_without_facing_groups() {
    let sheet = rect([-1.5, 1.5], [-1.0, 1.0], true, |x, _| 0.3 - 0.4 * x);
    let items = |grouped: bool| {
        let (shape, mut r) = label([0.0, 0.0, 0.0], [1.4, 0.3]);
        if !grouped {
            r.facing_group = 0;
        }
        [(shape, r), face(2, &sheet, BLUE, &[(12, SQUARE)])]
    };
    let mut off = stream(&items(true));
    off.set_label_placement(false);
    assert!(off.label_plan(W, H).unwrap().is_none());
    let plain = stream(&items(false)).draw(W, H, 1).unwrap();
    assert_eq!(off.draw(W, H, 1).unwrap(), plain);
    assert_ne!(stream(&items(true)).draw(W, H, 1).unwrap(), plain);
}

/// "Both resolves, contributors and diagnostics must use the same insertion policy": the
/// contributor report lists the label under the sheet it goes behind, over one it does not, and
/// over a nearer line.
#[test]
fn the_contributor_report_reads_the_placement_the_frame_is_drawn_by() {
    let top = |items: &[(Shape, Record)]| {
        let found = stream(items).contributors(80, 45, 81, 46).unwrap();
        let pixel = &found[&(80, 45)];
        pixel.leaves.last().expect("a leaf").object
    };
    let label_object = 101;
    let behind = [
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        square_at(2, 1.0, true, 1),
    ];
    assert_eq!(top(&behind), 102);
    let front = [
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        square_at(2, -1.0, true, 1),
    ];
    assert_eq!(top(&front), label_object);
    let line = [
        label([0.0, 0.0, 0.0], [0.5, 0.3]),
        line_at(2, 1.0, [1.0, 0.5, 0.0, 1.0]),
    ];
    assert_eq!(top(&line), label_object);
}

// The overlap is the order points both hold (main's ruling on per-cell overlap). Pixel column 100
// holds order points at x = 100 + (2k + 1) / 32 for k = 0..15, one in each row; a vertical strip
// between two of them holds none. A camera this far away draws almost orthographically, so a
// world x lands where its frame mapping puts it.

/// The camera of [`view`], `focal_distance` away.
fn far_view() -> View {
    let mut v = view();
    if let Camera::ThreeD(c) = &mut v.camera {
        c.focal_distance = 1e6;
    }
    v
}

/// The world x the frame mapping puts at pixel x `px`.
fn world_x(px: f64) -> f64 {
    let v = view();
    (px - W as f64 / 2.0) * v.frame_width / W as f64
}

/// The plan and the frame, under the far camera, of a wide white label at depth 0, a blue sheet
/// over pixels 90..`end` at the label's depth where x is the pixel x `cross` (nearer to its left,
/// farther to its right), and `more`.
fn sliver_with(cross: f64, end: f64, more: &[(Shape, Record)]) -> (Vec<Pair>, Vec<u8>) {
    let (c, e) = (world_x(cross), world_x(end));
    let sheet = rect([world_x(90.0), e], [-1.0, 1.0], true, |x, _| 0.5 * (c - x));
    let mut items = vec![
        label([0.0, 0.0, 0.0], [3.0, 0.3]),
        face(2, &sheet, BLUE, &[(12, SQUARE)]),
    ];
    items.extend_from_slice(more);
    let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
    let d = arachne_core::probes::first_delta(far_view(), &items, &order);
    let mut s = Stream::new();
    s.apply(&d).expect("the delta applies");
    let pairs = s.label_plan(W, H).unwrap().expect("a label").pairs;
    (pairs, s.draw(W, H, 1).expect("the frame draws"))
}

fn sliver(cross: f64, end: f64) -> (Vec<Pair>, Vec<u8>) {
    sliver_with(cross, end, &[])
}

/// Whether `got` is `under` with `over` composited over `share` of it, each colour channel to 2
/// levels: a pixel whose sub-samples composite `over` on that share, `under` on the rest.
fn mixed(got: [u8; 4], under: [u8; 4], over: [u8; 4], share: f64) -> bool {
    (0..3).all(|c| {
        let want = under[c] as f64 * (1.0 - share) + over[c] as f64 * share;
        (got[c] as f64 - want).abs() <= 2.0
    })
}

// In pixel (100, 45), sub-samples 0 and 8 (offsets [-8, -7] and [-7, 2]) have their order points
// at x = 100.03125 and 100.09375; the four of the pixel's first cell column (x 100 to 100.25) at
// x up to 100.21875; those of its second column at x 100.28125 and beyond. A qualifying sheet
// anchors the label's insertion at a sub-sample only where it holds its order point: where it
// covers some of the cell but not the point, the label is in front.

/// "A farther sliver narrower than the sample spacing and lying between order points qualifies."
/// The sheet ends at 100.15: it holds two order points of pixel 100, whose sub-samples composite
/// the sheet's 0.6 of their cell over the label; at the rest the label is in front.
#[test]
fn a_farther_sliver_between_order_points_leaves_the_sheet_qualifying() {
    let (found, frame) = sliver(100.10, 100.15);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(
        (found[0].outcome, found[0].centre),
        (Outcome::Qualifies, false)
    );
    let p = px(&frame, 100, 45);
    let (white, blue) = (stacked(&[WHITE]), stacked(&[BLUE]));
    assert!(mixed(p, white, blue, 2.0 / 16.0 * 0.6), "{p:?}");
}

/// "A farther sliver covering one order point does not qualify": the label is in front.
#[test]
fn a_farther_sliver_holding_an_order_point_keeps_the_label_in_front() {
    let (found, frame) = sliver(100.08, 100.12);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(
        (found[0].outcome, found[0].centre),
        (Outcome::NotNearer, false)
    );
    assert_eq!(px(&frame, 100, 45), stacked(&[WHITE]));
}

/// A farther sliver across the boundary of two sub-sample cells (x = 100.25), holding neither
/// cell's order point: no point of the overlap is farther. The sheet holds the four order points
/// of pixel 100's first cell column, covering their cells whole; it covers 0.1 of the second
/// column's cells and none of their points, and there the label is in front.
#[test]
fn a_farther_sliver_across_a_cell_boundary_holding_no_order_point_is_no_overlap() {
    let (found, frame) = sliver(100.225, 100.275);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(
        (found[0].outcome, found[0].centre),
        (Outcome::Qualifies, false)
    );
    let p = px(&frame, 100, 45);
    assert!(
        mixed(p, stacked(&[WHITE]), stacked(&[BLUE]), 4.0 / 16.0),
        "{p:?}"
    );
}

/// Insertion reads qualification's predicate: a label goes behind a qualifying sheet at a
/// sub-sample only where the sheet holds its order point. With an opaque line in front of the
/// sheet, the line hides the label at the four sub-samples whose points the sheet holds, and
/// nowhere else: not at the four whose cells the sheet covers a tenth of.
#[test]
fn a_label_goes_behind_a_qualifying_sheet_only_where_the_sheet_holds_the_order_point() {
    let black = [0.0, 0.0, 0.0, 1.0];
    let (found, frame) = sliver_with(100.225, 100.275, &[line_at(3, 1.0, black)]);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].outcome, Outcome::Qualifies);
    let p = px(&frame, 100, 45);
    assert!(
        mixed(p, stacked(&[WHITE]), stacked(&[black]), 4.0 / 16.0),
        "{p:?}"
    );
}

/// A centre on a sheet's horizontal edge lies on the sheet, as one on any edge does: the sheet's
/// depth there decides, and here it is nearer.
#[test]
fn a_centre_on_a_sheet_s_horizontal_edge_is_held() {
    let sheet = rect([-1.5, 1.5], [0.0, 1.0], true, |x, _| 0.3 - 0.4 * x);
    let items = [
        label([0.0, 0.0, 0.0], [1.4, 0.3]),
        face(2, &sheet, BLUE, &[(12, SQUARE)]),
    ];
    assert_eq!(one(&items), (Outcome::Qualifies, true));
}

/// A record kept facing the camera whose facing group a held row set to 0 belongs to a label not
/// known (XF.md's "Facing groups"): never a sheet, whatever its source, so it hides no label; the
/// pair is open, the label staying in front.
#[test]
fn a_camera_facing_record_outside_every_label_is_no_sheet() {
    let (shape, mut nearer) = glyph(2, 8, [0.0, 0.0, 1.0], [0.0, 0.0], [0.6, 0.4]);
    nearer.paint.fill = Paint::One(BLUE);
    let items = [label([0.0, 0.0, 0.0], [0.5, 0.3]), (shape, nearer)];
    let mut s = stream(&items);
    let mut d = Delta::new(Time { num: 1, den: 15 }, 1);
    d.source = vec![SourceRow {
        slot: 2,
        facing_group: 0,
        source: vec![vec![node(12, SQUARE)]],
    }];
    s.apply(&d).expect("the row applies");
    let found = s.label_plan(W, H).unwrap().expect("a label").pairs;
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].outcome, Outcome::Open(Open::Label));
    let frame = s.draw(W, H, 1).expect("the frame draws");
    assert_eq!(px(&frame, 80, 45), stacked(&[WHITE]));
}

/// A label drawn as a mesh meets a sheet wherever its geometry does, not only where it has paths:
/// open against it, as against every sheet it meets.
#[test]
fn a_label_drawn_as_a_mesh_is_open_against_the_sheet_it_meets() {
    let shape = common::meshes::quad([
        [-0.5, -0.3, 0.0],
        [0.5, -0.3, 0.0],
        [0.5, 0.3, 0.0],
        [-0.5, 0.3, 0.0],
    ]);
    let mut r = common::meshes::mesh_record(1, &shape, WHITE, Space::World);
    r.facing_group = 7;
    r.source = vec![vec![node(51, GLYPH), node(97, TEX)]];
    let found = pairs(&[(shape, r), square_at(2, 1.0, true, 1)]);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].outcome, Outcome::Open(Open::Label));
}

/// A sheet of `rgba` at the ends `from` (alpha `1`) to `to` (alpha `0`): a linear gradient,
/// transparent from `to` on.
fn fading(slot: u32, corners: &[[f64; 3]], from: [f64; 3], to: [f64; 3]) -> (Shape, Record) {
    let (shape, mut r) = face(slot, corners, BLUE, &[(10 + slot, SQUARE)]);
    let mut clear = BLUE;
    clear[3] = 0.0;
    r.paint.fill = Paint::Two([BLUE, clear]);
    r.paint.gradient = Some([from, to]);
    (shape, r)
}

/// A sheet qualifying where its gradient is opaque backs no label where its colour is clear: there
/// the label is in front, as where no sheet is, so a nearer line does not hide it. Where the sheet
/// is opaque the line covers the label behind it.
#[test]
fn a_sheet_backs_a_label_only_where_its_fill_has_positive_alpha() {
    let orange = [1.0, 0.5, 0.0, 1.0];
    let sheet = fading(
        2,
        &rect([-1.5, 1.5], [-1.0, 1.0], true, |_, _| 1.0),
        [-1.5, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    );
    let items = [
        label([-0.5, 0.0, 0.0], [1.4, 0.3]),
        sheet,
        line_at(3, 2.0, orange),
    ];
    assert_eq!(one(&items), (Outcome::Qualifies, true));
    let frame = drawn(&items);
    assert_eq!(px(&frame, 69, 45), stacked(&[orange]));
    assert_eq!(px(&frame, 86, 45), stacked(&[WHITE]));
}
