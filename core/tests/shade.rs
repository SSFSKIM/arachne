//! Lighting: CE's per-face model against values Arachne's own ThreeDCamera computed
//! (`tests/data/shading.txt`, written by `tests/data/gen_shading.py`), and the soft diffuse model,
//! which lights in linear light; and the stops a stream's drawer paints its shaded paths with,
//! read without drawing (`Stream::face_stops`).

mod common;

use arachne_core::shade::{
    SoftDiffuse, ce_face_stops, ce_face_stops_decided, ce_shaded_rgb, decided_normals,
};
use arachne_core::stream::Stream;
use arachne_core::xf::*;

/// NumPy takes `** 3` and the norms from the platform and BLAS; the core multiplies and takes the
/// square root itself, so the last bits may differ.
const REL: f64 = 1e-12;

fn nums(tokens: &[&str]) -> Vec<f64> {
    tokens
        .iter()
        .map(|t| t.parse::<f64>().unwrap_or_else(|_| panic!("number: {t}")))
        .collect()
}

#[test]
fn ces_two_stops_match_arachnes_three_d_camera() {
    let text = include_str!("data/shading.txt");
    let mut faces = 0;
    for line in text.lines().filter(|l| l.starts_with("face ")) {
        let t: Vec<&str> = line.split_whitespace().collect();
        let n: usize = t[1].parse().unwrap();
        let points: Vec<[f64; 3]> = nums(&t[2..2 + 3 * n])
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let k: usize = t[2 + 3 * n].parse().unwrap();
        let at = 3 + 3 * n;
        let rgbas: Vec<[f64; 4]> = nums(&t[at..at + 4 * k])
            .chunks(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let light = nums(&t[at + 4 * k..at + 4 * k + 3]);
        assert_eq!(t[at + 4 * k + 3], "->");
        let want = nums(&t[at + 4 * k + 4..]);
        assert_eq!(want.len(), 8, "{line}");
        let got = ce_face_stops(&points, &rgbas, [light[0], light[1], light[2]]);
        for (g, w) in got.iter().flatten().zip(&want) {
            let err = (g - w).abs() / w.abs().max(1.0);
            assert!(err <= REL, "face {faces}: got {got:?}, want {want:?}");
        }
        faces += 1;
    }
    assert!(faces >= 48, "{faces} faces");
}

#[test]
fn soft_diffuse_lights_in_linear_light() {
    let soft = SoftDiffuse {
        ambient: 0.2,
        wrap: 0.5,
    };
    let point = [0.0, 0.0, 0.0];
    let eye = [0.0, 0.0, 10.0];
    let up = [0.0, 0.0, 1.0];
    let base = [0.5, 0.25, 0.1];
    // Facing the light: full light.
    let lit = soft.lit(base, point, up, [0.0, 0.0, 100.0], eye);
    for k in 0..3 {
        assert!((lit[k] - base[k]).abs() < 1e-15);
    }
    // At the terminator the light wraps: (0 + wrap) / (1 + wrap) of the light past the ambient.
    let side = soft.lit(base, point, up, [100.0, 0.0, 0.0], eye);
    let share = 0.2 + 0.8 * (0.5 / 1.5);
    for k in 0..3 {
        assert!((side[k] - base[k] * share).abs() < 1e-15, "{side:?}");
    }
    // Facing away from the light: the ambient floor.
    let away = soft.lit(base, point, up, [0.0, 0.0, -100.0], eye);
    for k in 0..3 {
        assert!((away[k] - base[k] * 0.2).abs() < 1e-15, "{away:?}");
    }
    // A normal turned away from the eye is lit as the side the eye sees.
    let down = [0.0, 0.0, -1.0];
    assert_eq!(soft.lit(base, point, down, [0.0, 0.0, 100.0], eye), lit);
}

/// A grid the engine snaps geometry to: 2^-14, a frame 14.2 wide on 854 pixels.
const GRID: f64 = 1.0 / 16384.0;

fn snapped(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    points
        .iter()
        .map(|p| p.map(|v| (v / GRID).round_ties_even() * GRID + 0.0))
        .collect()
}

/// A helix as `ParametricFunction` lays it (cubics through samples `step` apart, handles a third
/// of the way along the chord's tangent), around `turns` turns.
fn helix(step: f64, samples: usize) -> Vec<[f64; 3]> {
    let at = |u: f64| [1.2 * libm::cos(u), 1.2 * libm::sin(u), 0.05 * u];
    let tangent = |u: f64| [-1.2 * libm::sin(u), 1.2 * libm::cos(u), 0.05];
    let mut out = Vec::new();
    for i in 0..samples {
        let (u0, u1) = (-3.0 + step * i as f64, -3.0 + step * (i + 1) as f64);
        let (a, b, ta, tb) = (at(u0), at(u1), tangent(u0), tangent(u1));
        out.push(a);
        out.push(std::array::from_fn(|k| a[k] + ta[k] * step / 3.0));
        out.push(std::array::from_fn(|k| b[k] - tb[k] * step / 3.0));
        out.push(b);
    }
    out
}

const LIGHT: [f64; 3] = [-7.0, -9.0, 10.0];
const RED: [[f64; 4]; 1] = [[0.8, 0.2, 0.15, 1.0]];

/// A smooth curve's lit corners, decided on its points as made, light its snapped points as its
/// points as made are lit: CE's normal at a corner of a curve sampled finely is the cross product
/// of two nearly opposite chords, which the snap turns by tens of degrees.
#[test]
fn a_smooth_curve_is_lit_by_the_normals_of_its_points_as_made() {
    let raw = helix(0.01, 400);
    let drawn = snapped(&raw);
    let decided = decided_normals(&raw, GRID);
    assert!(decided.iter().all(|n| *n != [0.0; 3]), "{decided:?}");
    let want = ce_face_stops(&raw, &RED, LIGHT);
    let got = ce_face_stops_decided(&drawn, &RED, LIGHT, decided);
    let unguided = ce_face_stops(&drawn, &RED, LIGHT);
    let off = |a: &[[f64; 4]; 2]| {
        a.iter()
            .flatten()
            .zip(want.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    };
    // The light's direction moves with the snapped corner only.
    assert!(off(&got) < 1e-5, "decided: {got:?} against {want:?}");
    assert!(
        off(&unguided) > 0.05,
        "the snap no longer moves it: {unguided:?}"
    );
}

/// A sphere's pole face as `Surface` lays it: two corners at the pole, `cos(pi / 2)` apart (about
/// 1e-16), and two on the next latitude.
fn pole_face() -> Vec<[f64; 3]> {
    let point = |u: f64, v: f64| {
        [
            1.5 * libm::cos(u) * libm::cos(v),
            1.5 * libm::cos(u) * libm::sin(v),
            1.5 * libm::sin(u),
        ]
    };
    let (top, below) = (
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_2 - 0.21,
    );
    let corners = [
        point(below, 0.0),
        point(top, 0.0),
        point(top, 0.2),
        point(below, 0.2),
    ];
    let mut out = Vec::new();
    for i in 0..4 {
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        let at = |t: f64| -> [f64; 3] { std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t) };
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

/// At a corner whose neighbour lies a float's noise off it, CE's normal is the noise's direction:
/// the corner is decided degenerate (the zero vector) and lit by the face's Newell normal over
/// the points drawn, oriented as CE's corner normal on the face's other corners.
#[test]
fn a_corner_a_floats_noise_from_its_neighbour_is_lit_by_the_faces_normal() {
    let raw = pole_face();
    let drawn = snapped(&raw);
    let decided = decided_normals(&raw, GRID);
    let end = (raw.len() - 1) / 6 * 3;
    // Corner 0 is on the latitude below; the end corner (point 6, a handle of the edge along the
    // pole) is within the noise of the pole's other corner.
    assert_ne!(decided[0], [0.0; 3], "{decided:?}");
    assert_eq!(decided[1], [0.0; 3], "point {end}: {decided:?}");
    let got = ce_face_stops_decided(&drawn, &RED, LIGHT, decided);
    // The face's normal: Newell's over the drawn points, normalised; it points the way CE's
    // normal at the well-found corner does.
    let mut n = [0.0f64; 3];
    for i in 0..drawn.len() {
        let (a, b) = (drawn[i], drawn[(i + 1) % drawn.len()]);
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let n = n.map(|v| v / len);
    assert!(n[0] * decided[0][0] + n[1] * decided[0][1] + n[2] * decided[0][2] > 0.9);
    let rgb = ce_shaded_rgb([0.8, 0.2, 0.15], drawn[end], n, LIGHT);
    for k in 0..3 {
        assert!(
            (got[1][k] - rgb[k]).abs() < 1e-12,
            "{got:?} against {rgb:?}"
        );
    }
    // Not snapped (grid 0): never degenerate.
    assert!(decided_normals(&raw, 0.0).iter().all(|n| *n != [0.0; 3]));
}

/// A pole whose points coincide exactly lies on the same side of the boundary as one whose
/// points a float's noise keeps apart: both corners are degenerate and lit by the face's normal,
/// so a pole lights the same whichever way the machine that made it rounded.
#[test]
fn a_pole_of_coincident_points_is_lit_as_one_a_floats_noise_apart() {
    let noise = pole_face();
    let mut exact = noise.clone();
    for p in &mut exact {
        if p[2] > 1.49 {
            *p = [0.0, 0.0, 1.5];
        }
    }
    assert_ne!(exact, noise);
    let (drawn, drawn_exact) = (snapped(&noise), snapped(&exact));
    assert_eq!(drawn, drawn_exact, "the snap takes both to one pole");
    // The end corner is the pole's; corner 0, on the latitude below, is found from chords that
    // reach the pole, so it moves by the noise alone.
    let (decided, decided_exact) = (decided_normals(&noise, GRID), decided_normals(&exact, GRID));
    assert_eq!(decided[1], [0.0; 3], "{decided:?}");
    assert_eq!(decided_exact[1], [0.0; 3], "{decided_exact:?}");
    let lit = ce_face_stops_decided(&drawn, &RED, LIGHT, decided);
    let lit_exact = ce_face_stops_decided(&drawn_exact, &RED, LIGHT, decided_exact);
    for (a, b) in lit.iter().flatten().zip(lit_exact.iter().flatten()) {
        assert!((a - b).abs() < 1e-12, "{lit:?} against {lit_exact:?}");
    }
}

const STROKE: [[f64; 4]; 2] = [[0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, 0.8]];
const OPACITY: f64 = 0.5;

/// The test's 3D view (lit from `LIGHT`), shading or not.
fn shading_view(shading: bool) -> View {
    let mut view = common::three_d_view(64, 36);
    let Camera::ThreeD(c) = &mut view.camera else {
        unreachable!()
    };
    assert_eq!(c.light_source, LIGHT);
    c.should_apply_shading = shading;
    view
}

/// A shaded path of `points` in world space: a one-row fill, a two-row stroke, group opacity
/// `OPACITY`, no background stroke, carrying `normals` when given.
fn shaded_path(points: &[[f64; 3]], normals: Option<[[f64; 3]; 2]>) -> (Shape, Record) {
    let shape = Shape::new(ShapeData::Path {
        points: points.to_vec(),
    });
    let mut r = Record::new(RecordKind::Path, 1, 101);
    r.geometry.shape_a = Some(shape.key());
    r.depth.space = Space::World;
    r.depth.shade_in_3d = true;
    r.paint.fill = Paint::One(RED[0]);
    r.paint.stroke = Paint::Two(STROKE);
    r.paint.stroke_width = 2.0;
    r.opacity = OPACITY;
    r.normals = normals.map(|n| n.to_vec());
    (shape, r)
}

/// A stream holding `view` and `record`, every part of it in the order.
fn stream_with(view: View, shapes: Vec<Shape>, paints: Vec<Stops>, record: Record) -> Stream {
    let parts = match &record.body {
        Body::FaceGroup(f) => f.uv.len() as u32,
        _ => 1,
    };
    let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
    d.view = Some(view);
    d.upload = shapes;
    d.upload_paints = paints;
    d.order = Some(
        (0..parts)
            .map(|part| OrderEntry {
                layer: Layer::Moving,
                slot: record.slot,
                part,
            })
            .collect(),
    );
    d.set = vec![record];
    let mut s = Stream::new();
    s.apply(&d).expect("a valid scene");
    s
}

fn with_opacity(stops: [[f64; 4]; 2]) -> [[f64; 4]; 2] {
    stops.map(|c| [c[0], c[1], c[2], c[3] * OPACITY])
}

/// A face's Newell normal over its points, normalised.
fn newell_unit(points: &[[f64; 3]]) -> [f64; 3] {
    let mut n = [0.0f64; 3];
    for i in 0..points.len() {
        let (a, b) = (points[i], points[(i + 1) % points.len()]);
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n.map(|v| v / len)
}

/// Each layer's two stops are the ones its writer's normals light it by (`ce_face_stops_decided`
/// on the points drawn), the record's opacity applied; a layer without paint has none. The pole
/// face's end corner is decided degenerate, so it is lit by the face's Newell normal, the corner
/// `newell_corners` names.
#[test]
fn a_shaded_paths_stops_are_its_decided_stops_with_its_opacity() {
    let raw = pole_face();
    let drawn = snapped(&raw);
    let decided = decided_normals(&raw, GRID);
    let (shape, record) = shaded_path(&drawn, Some(decided));
    let s = stream_with(shading_view(true), vec![shape], Vec::new(), record);
    let found = s.face_stops(&[1, 2]).unwrap();
    let fill = with_opacity(ce_face_stops_decided(&drawn, &RED, LIGHT, decided));
    let stroke = with_opacity(ce_face_stops_decided(&drawn, &STROKE, LIGHT, decided));
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[&1], vec![Some([None, Some(fill), Some(stroke)])]);

    // The Newell corner: the end stop is lit by the face's normal, not by CE's at that corner.
    assert_eq!(s.newell_corners(&[1]).unwrap()[&1], vec![[false, true]]);
    let end = (drawn.len() - 1) / 6 * 3;
    let newell = ce_shaded_rgb([0.8, 0.2, 0.15], drawn[end], newell_unit(&drawn), LIGHT);
    for k in 0..3 {
        assert!(
            (fill[1][k] - newell[k]).abs() < 1e-12,
            "{fill:?} against {newell:?}"
        );
    }
    let ce = ce_face_stops(&drawn, &RED, LIGHT);
    assert!(
        (ce[1][0] - fill[1][0]).abs() > 1e-3,
        "{ce:?} against {fill:?}"
    );
    // The start corner is lit by its normal as decided.
    let start = ce_shaded_rgb([0.8, 0.2, 0.15], drawn[0], decided[0], LIGHT);
    assert_eq!(fill[0][..3], start, "its decided normal");
}

/// Without its writer's normals a path is lit by CE's normals at its points as drawn.
#[test]
fn a_path_without_normals_is_lit_by_ces_normals_at_its_points() {
    let drawn = snapped(&helix(0.01, 40));
    let (shape, record) = shaded_path(&drawn, None);
    let s = stream_with(shading_view(true), vec![shape], Vec::new(), record);
    let fill = with_opacity(ce_face_stops(&drawn, &RED, LIGHT));
    let stroke = with_opacity(ce_face_stops(&drawn, &STROKE, LIGHT));
    assert_eq!(
        s.face_stops(&[1]).unwrap()[&1],
        vec![Some([None, Some(fill), Some(stroke)])]
    );
}

/// A face group's faces, in its order of `planes`, each lit with its own row.
#[test]
fn a_face_groups_faces_have_their_own_stops_in_order() {
    let left = common::square_points(0.5);
    let right: Vec<[f64; 3]> = left.iter().map(|p| [p[0] + 1.0, p[1], 0.3]).collect();
    let shape = Shape::new(ShapeData::Path {
        points: [left.clone(), right.clone()].concat(),
    });
    let fills = Stops::new(vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.75]]);
    let mut r = Record::new(RecordKind::FaceGroup, 1, 101);
    r.geometry.shape_a = Some(shape.key());
    r.depth.space = Space::World;
    r.depth.shade_in_3d = true;
    r.body = Body::FaceGroup(FaceGroup {
        uv: vec![[0, 0], [1, 0]],
        grid: [2, 1],
        fill: fills.key(),
        stroke: None,
        face_ends: Vec::new(),
        face_widths: Vec::new(),
    });
    let s = stream_with(shading_view(true), vec![shape], vec![fills.clone()], r);
    let want: Vec<_> = [(&left, 0), (&right, 1)]
        .into_iter()
        .map(|(face, row)| {
            let stops = ce_face_stops(face, &fills.rows()[row..row + 1], LIGHT);
            Some([None, Some(stops), None])
        })
        .collect();
    assert_eq!(s.face_stops(&[1]).unwrap()[&1], want);
}

/// A path CE's lighting does not shade has no stops: not `shade_in_3d`, a camera that does not
/// shade, a flat camera, or a pre-projected path (its own camera lit it).
#[test]
fn an_unshaded_or_pre_projected_path_has_no_entry() {
    let drawn = snapped(&pole_face());
    let (shape, r) = shaded_path(&drawn, None);
    let mut plain = r.clone();
    plain.depth.shade_in_3d = false;
    let unshaded = stream_with(shading_view(true), vec![shape.clone()], Vec::new(), plain);
    assert!(unshaded.face_stops(&[1]).unwrap().is_empty());
    let off = stream_with(
        shading_view(false),
        vec![shape.clone()],
        Vec::new(),
        r.clone(),
    );
    assert!(off.face_stops(&[1]).unwrap().is_empty());
    let flat = common::flat_view(64, 36, [0.0, 0.0, 0.0, 1.0]);
    let flat = stream_with(flat, vec![shape.clone()], Vec::new(), r.clone());
    assert!(flat.face_stops(&[1]).unwrap().is_empty());

    let placed = Shape::new(ShapeData::Path {
        points: drawn
            .iter()
            .map(|p| [p[0] * 0.5, p[1] * 0.5, 0.0])
            .collect(),
    });
    let mut produced = r;
    produced.features = FEATURE_PRE_PROJECTED;
    produced.pre_projected = Some(PreProjected {
        points: placed.key(),
        gradient: None,
        fill: Paint::One(RED[0]),
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let pre = stream_with(
        shading_view(true),
        vec![shape, placed],
        Vec::new(),
        produced,
    );
    assert!(pre.face_stops(&[1]).unwrap().is_empty());
    // A slot the stream holds no record for has none either.
    assert!(pre.face_stops(&[7]).unwrap().is_empty());
}
