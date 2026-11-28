//! Pre-projected records (XF 1.1's tail) on the core: a path or a points record drawn where a
//! scene's own camera placed it and in the colours that camera painted, nothing projected or lit
//! again, its depth keyed from the camera's own z; never moved by the view's rotation.

mod common;

use arachne_core::project::{Placement, Projection, pre_display};
use arachne_core::shade::ce_face_stops;
use arachne_core::stream::{DrawError, Stream};
use arachne_core::xf::*;

const W: u32 = 160;
const H: u32 = 90;
const CAP: [f64; 4] = [0.0, 1.0, 1.0, 1.0];

fn path(points: Vec<[f64; 3]>) -> Shape {
    Shape::new(ShapeData::Path { points })
}

/// A closed polygon's outline in CE's cubic layout.
fn ring(corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for i in 0..corners.len() {
        let (a, b) = (corners[i], corners[(i + 1) % corners.len()]);
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

/// A curve in space, not on one plane: a quarter of a helix as cubics.
fn helix() -> Vec<[f64; 3]> {
    let at = |t: f64| {
        let a = t * std::f64::consts::FRAC_PI_2;
        [
            2.2 * libm::cos(a) - 1.0,
            2.2 * libm::sin(a) - 1.0,
            -0.6 + 0.4 * t,
        ]
    };
    let mut out = Vec::new();
    let n = 6;
    for i in 0..n {
        let (t0, t1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
        out.extend([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0].map(|s| at(t0 + (t1 - t0) * s)));
    }
    out
}

fn projection(view: &View) -> Projection {
    let Camera::ThreeD(c) = &view.camera else {
        unreachable!("a 3D view")
    };
    Projection::new(c, view.frame_center)
}

fn light(view: &View) -> [f64; 3] {
    let Camera::ThreeD(c) = &view.camera else {
        unreachable!("a 3D view")
    };
    c.light_source
}

fn record(kind: RecordKind, slot: u32, shape: &Shape) -> Record {
    let mut r = Record::new(kind, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
    r.depth.space = Space::World;
    r
}

/// The scene drawn: a floor below, a lit slanted square above it, a curve between the two, a square
/// in a fixed orientation above the floor and a cap above the middle of the lit square, each pair
/// overlapping on screen at depths well apart.
struct Scene {
    floor: (Shape, Record),
    cap: (Shape, Record),
    square: (Shape, Record),
    curve: (Shape, Record),
    facing: (Shape, Record),
}

fn scene() -> Scene {
    let floor_shape = Shape::new(ShapeData::Mesh {
        positions: vec![
            [-4.0, -4.0, -1.5],
            [4.0, -4.0, -1.5],
            [4.0, 4.0, -1.5],
            [-4.0, 4.0, -1.5],
        ],
        triangles: vec![[0, 1, 2], [0, 2, 3]],
        normals: None,
    });
    let mut floor = record(RecordKind::Mesh, 1, &floor_shape);
    floor.paint.fill = Paint::One([0.1, 0.2, 0.7, 1.0]);

    // The middle of the square, moved two units toward the camera (polar angle 70 degrees,
    // azimuth -45): over the square on screen, in front of it.
    let toward = [0.664, -0.664, 0.342].map(|v| 2.0 * v);
    let cap_shape = Shape::new(ShapeData::Mesh {
        positions: [[-0.6, -0.6], [0.2, -0.6], [0.2, 0.2], [-0.6, 0.2]]
            .map(|[x, y]| [x + toward[0], y + toward[1], 0.65 + toward[2]])
            .to_vec(),
        triangles: vec![[0, 1, 2], [0, 2, 3]],
        normals: None,
    });
    let mut cap = record(RecordKind::Mesh, 5, &cap_shape);
    cap.paint.fill = Paint::One(CAP);

    let square_shape = path(ring(&[
        [-1.5, -1.5, 0.5],
        [1.0, -1.5, 0.8],
        [1.0, 1.0, 0.8],
        [-1.5, 1.0, 0.5],
    ]));
    let mut square = record(RecordKind::Path, 2, &square_shape);
    square.depth.shade_in_3d = true;
    square.paint.fill = Paint::One([0.9, 0.5, 0.2, 1.0]);
    square.paint.stroke = Paint::One([1.0, 1.0, 1.0, 1.0]);
    square.paint.stroke_width = 4.0;

    let curve_shape = path(helix());
    let mut curve = record(RecordKind::Path, 3, &curve_shape);
    curve.paint.stroke = Paint::One([0.2, 1.0, 0.3, 1.0]);
    curve.paint.stroke_width = 6.0;

    let facing_shape = path(ring(&[
        [2.0, 1.5, 0.3],
        [3.0, 1.5, 0.3],
        [3.0, 1.5, 1.3],
        [2.0, 1.5, 1.3],
    ]));
    let mut facing = record(RecordKind::Path, 4, &facing_shape);
    facing.depth.fixed_orientation_centre = Some([2.5, 1.5, 0.8]);
    facing.paint.fill = Paint::One([1.0, 0.9, 0.1, 1.0]);

    Scene {
        floor: (floor_shape, floor),
        cap: (cap_shape, cap),
        square: (square_shape, square),
        curve: (curve_shape, curve),
        facing: (facing_shape, facing),
    }
}

/// The record a camera of the scene's own produced for `rec`: its points where `placement` puts
/// them, its paints lit as CE lights a shaded path; the record's own shape and colours replaced by
/// ones nothing should draw, so only the tail can draw what the frame shows. A fixed orientation
/// keeps its own shape: its first point against where the camera placed it says where the camera
/// moved its centre.
fn produced(
    view: &View,
    (shape, rec): &(Shape, Record),
    placement: Placement,
    decoy: &Shape,
    uploads: &mut Vec<Shape>,
) -> Record {
    let ShapeData::Path { points } = shape.data() else {
        unreachable!("a path")
    };
    let shown = pre_display(points, Some(&projection(view)), placement);
    let lit = |paint: &Paint| match paint {
        Paint::One(c) if rec.depth.shade_in_3d => {
            Paint::Two(ce_face_stops(points, &[*c], light(view)))
        }
        other => other.clone(),
    };
    let corners = [shown[0], shown[(shown.len() - 1) / 6 * 3]];
    let tail = path(shown);
    let mut r = rec.clone();
    r.features = FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: tail.key(),
        gradient: rec.depth.shade_in_3d.then_some(corners),
        fill: lit(&rec.paint.fill),
        stroke: lit(&rec.paint.stroke),
        background_stroke: lit(&rec.paint.background_stroke),
    });
    if !matches!(placement, Placement::FixedOrientation(_)) {
        r.geometry.shape_a = Some(decoy.key());
    }
    r.paint.fill = Paint::One([1.0, 0.0, 1.0, 1.0]);
    r.paint.stroke = Paint::One([1.0, 0.0, 1.0, 1.0]);
    uploads.push(tail);
    r
}

fn draw(view: &View, items: &[(Shape, Record)], extra: Vec<Shape>) -> Result<Vec<u8>, DrawError> {
    let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
    let mut d = arachne_core::probes::first_delta(view.clone(), items, &order);
    d.upload.extend(extra);
    let mut s = Stream::new();
    s.apply(&d).expect("the delta applies");
    s.draw(W, H, 1)
}

/// The scene's paths, pre-projected as the engine's own camera would place and light them.
fn pre_projected_scene(view: &View, s: &Scene) -> (Vec<(Shape, Record)>, Vec<Shape>) {
    let decoy = path(ring(&[[5.0, 3.0, 0.0], [6.0, 3.0, 0.0], [6.0, 3.5, 0.0]]));
    let mut extra = Vec::new();
    let square = produced(view, &s.square, Placement::Projected, &decoy, &mut extra);
    let curve = produced(view, &s.curve, Placement::Projected, &decoy, &mut extra);
    let centre = s.facing.1.depth.fixed_orientation_centre.unwrap();
    let facing = produced(
        view,
        &s.facing,
        Placement::FixedOrientation(centre),
        &decoy,
        &mut extra,
    );
    let items = vec![
        (decoy.clone(), square),
        (decoy, curve),
        (s.facing.0.clone(), facing),
    ];
    (items, extra)
}

#[test]
fn a_pre_projected_path_draws_what_its_camera_placed_and_painted_at_that_camera_s_depth() {
    // Drawn in the order that puts depth to work: the cap first, so the square it hides part of
    // must lose by depth; the floor last, so everything over it must win by depth; the square
    // before the curve it hides part of.
    let view = common::three_d_view(W, H);
    let s = scene();
    let engine = draw(
        &view,
        &[
            s.cap.clone(),
            s.square.clone(),
            s.curve.clone(),
            s.facing.clone(),
            s.floor.clone(),
        ],
        Vec::new(),
    )
    .unwrap();
    let (mut items, extra) = pre_projected_scene(&view, &s);
    items.insert(0, s.cap.clone());
    items.push(s.floor.clone());
    let produced = draw(&view, &items, extra).unwrap();
    let differ = (0..(W * H) as usize)
        .filter(|i| engine[4 * i..4 * i + 4] != produced[4 * i..4 * i + 4])
        .count();
    assert_eq!(differ, 0, "{differ} pixels differ");
    // The scene draws what it says: the lit square's gradient, the curve and the fixed square.
    let colours: std::collections::HashSet<&[u8]> = engine.chunks(4).collect();
    assert!(colours.len() > 20, "{} colours", colours.len());
    // The cap is lit as CE lights a mesh: its pixels are cyan, not CAP's bytes.
    let cap = engine
        .chunks(4)
        .filter(|p| p[0] < 40 && p[1] > 120 && p[1] == p[2])
        .count();
    assert!(cap >= 10, "{cap} pixels of the cap");
}

/// A fixed orientation faces the camera at its pivot, wherever its own points' middle is: a member
/// of a family fixed about its parent's centre (or about a custom centre) sits at the same depth
/// whichever camera placed it.
#[test]
fn a_pre_projected_fixed_orientation_is_keyed_at_its_pivot_not_its_points_middle() {
    let view = common::three_d_view(W, H);
    let s = scene();
    // A child square over the floor, fixed about its parent's centre under the floor, where the
    // floor hides that centre on screen: the floor hides the child too.
    let pivot = [0.0, 0.0, -2.5];
    let child_shape = path(ring(&[
        [-0.3, -0.3, 1.0],
        [0.3, -0.3, 1.0],
        [0.3, 0.3, 1.0],
        [-0.3, 0.3, 1.0],
    ]));
    let mut child = record(RecordKind::Path, 6, &child_shape);
    child.depth.fixed_orientation_centre = Some(pivot);
    child.paint.fill = Paint::One([1.0, 0.9, 0.1, 1.0]);
    let child = (child_shape, child);
    let engine = draw(&view, &[child.clone(), s.floor.clone()], Vec::new()).unwrap();
    let floor = draw(&view, std::slice::from_ref(&s.floor), Vec::new()).unwrap();
    assert_eq!(engine, floor, "the floor hides the child on this camera");
    let decoy = path(ring(&[[5.0, 3.0, 0.0], [6.0, 3.0, 0.0], [6.0, 3.5, 0.0]]));
    let mut extra = Vec::new();
    let placed = produced(
        &view,
        &child,
        Placement::FixedOrientation(pivot),
        &decoy,
        &mut extra,
    );
    let produced = draw(&view, &[(child.0.clone(), placed), s.floor.clone()], extra).unwrap();
    let differ = (0..(W * H) as usize)
        .filter(|i| engine[4 * i..4 * i + 4] != produced[4 * i..4 * i + 4])
        .count();
    assert_eq!(differ, 0, "{differ} pixels differ");
}

#[test]
fn a_pre_projected_record_never_moves_with_the_view_s_rotation() {
    // The view's angles are what the engine's camera holds; a camera of the scene's own may have
    // drawn from others (an overridden `get_phi`), and its records say where it placed them.
    let view = common::three_d_view(W, H);
    let s = scene();
    let (items, extra) = pre_projected_scene(&view, &s);
    let first = draw(&view, &items, extra.clone()).unwrap();
    let mut turned = view.clone();
    let Camera::ThreeD(c) = &mut turned.camera else {
        unreachable!()
    };
    c.phi += 0.4;
    c.theta -= 0.7;
    c.gamma += 0.2;
    assert_eq!(first, draw(&turned, &items, extra).unwrap());
}

#[test]
fn a_pre_projected_points_record_draws_its_points_where_its_camera_placed_them() {
    let view = common::three_d_view(W, H);
    let s = scene();
    let colours = Stops::new(vec![[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]);
    // One point above the floor, one below it.
    let world = vec![[0.5, 0.3, -0.5], [-0.4, 0.6, -2.5]];
    let cloud = path(world.clone());
    let mut dots = record(RecordKind::Points, 2, &cloud);
    dots.body = Body::Points(Points {
        colors: colours.key(),
        radius: 5.0,
    });
    let frame = |dots: Record, extra: Vec<Shape>| {
        let items = [s.floor.clone(), (cloud.clone(), dots)];
        let order = [2, 1];
        let mut d = arachne_core::probes::first_delta(view.clone(), &items, &order);
        d.upload.extend(extra);
        d.upload_paints = vec![colours.clone()];
        let mut st = Stream::new();
        st.apply(&d).unwrap();
        st.draw(W, H, 1).unwrap()
    };
    let engine = frame(dots.clone(), Vec::new());
    let tail = path(pre_display(
        &world,
        Some(&projection(&view)),
        Placement::Projected,
    ));
    let decoy = path(vec![[5.0, 3.0, 0.0], [-5.0, -3.0, 0.0]]);
    let mut pre = dots;
    pre.geometry.shape_a = Some(decoy.key());
    pre.features = FEATURE_PRE_PROJECTED;
    pre.pre_projected = Some(PreProjected {
        points: tail.key(),
        gradient: None,
        fill: Paint::None,
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let produced = frame(pre, vec![tail, decoy]);
    assert_eq!(engine, produced);
    let yellow = engine
        .chunks(4)
        .filter(|p| p == &[255, 255, 0, 255])
        .count();
    let red = engine.chunks(4).filter(|p| p == &[255, 0, 0, 255]).count();
    assert_eq!(
        (yellow, red),
        (25, 0),
        "the point above shows, the one below does not"
    );
}

fn refused(view: &View, items: &[(Shape, Record)], extra: Vec<Shape>) -> String {
    match draw(view, items, extra) {
        Err(DrawError::Unsupported(u)) => u.reason,
        other => panic!("not refused: {:?}", other.map(|_| ())),
    }
}

#[test]
fn a_pre_projected_gradient_without_its_placed_ends_is_refused() {
    let view = common::three_d_view(W, H);
    let s = scene();
    let (items, extra) = pre_projected_scene(&view, &s);
    let mut square = items[0].clone();
    square.1.pre_projected.as_mut().unwrap().gradient = None;
    let reason = refused(&view, &[square], extra);
    assert!(reason.contains("gradient"), "{reason}");
}

#[test]
fn a_pre_projected_points_record_placing_other_than_one_point_per_colour_is_refused() {
    let view = common::three_d_view(W, H);
    let colours = Stops::new(vec![[1.0, 1.0, 0.0, 1.0]]);
    let cloud = path(vec![[0.0, 0.0, 0.0]]);
    let tail = path(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let mut dots = record(RecordKind::Points, 1, &cloud);
    dots.body = Body::Points(Points {
        colors: colours.key(),
        radius: 3.0,
    });
    dots.features = FEATURE_PRE_PROJECTED;
    dots.pre_projected = Some(PreProjected {
        points: tail.key(),
        gradient: None,
        fill: Paint::None,
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let mut d = arachne_core::probes::first_delta(view, &[(cloud, dots)], &[1]);
    d.upload.push(tail);
    d.upload_paints = vec![colours];
    let mut st = Stream::new();
    st.apply(&d).unwrap();
    let Err(DrawError::Unsupported(u)) = st.draw(W, H, 1) else {
        panic!("not refused")
    };
    assert!(u.reason.contains("one point per colour"), "{}", u.reason);
}

#[test]
fn a_pre_projected_path_is_painted_as_produced_under_any_light_model() {
    // The camera that produced the colours lit them already: the view's light model is not asked.
    let mut view = common::three_d_view(W, H);
    view.light = LightModel::SoftDiffuse {
        ambient: 0.3,
        wrap: 0.2,
    };
    let s = scene();
    let (items, extra) = pre_projected_scene(&view, &s);
    assert!(draw(&view, &items, extra).is_ok());
}
