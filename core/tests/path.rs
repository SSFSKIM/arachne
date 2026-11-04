//! Paths and points drawn through a Stream: fills by the nonzero rule at their exact area, glyph
//! holes, pixel-aligned rectangles whole, strokes at CE's width with Cairo's miter limit, CE's
//! subpath split and close rule, dashes, paths far beyond the frame, coplanar paths and meshes
//! tying exactly, discs of CE's thickness, instances drawing what their members drawn as records
//! draw, what paths do not draw yet refused, and the scanline's masks against point winding.

mod common;

use arachne_core::composite::{SAMPLES, area_coverage, blend, premultiply, sample_point};
use arachne_core::project::{FrameMap, subpixel_coords};
use arachne_core::scanline;
use arachne_core::stream::{DrawError, Stream};
use arachne_core::xf::*;
use common::meshes::{area_in_pixel, pixels_inside};
use common::{flat_view, three_d_view};

const W: u32 = 160;
const H: u32 = 90;
const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
const BLACK: [u8; 4] = [0, 0, 0, 255];

fn px(frame: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = (y * W + x) as usize * 4;
    frame[i..i + 4].try_into().unwrap()
}

fn map(view: &View) -> FrameMap {
    FrameMap::cairo(W, H, view.frame_width, view.frame_height, view.frame_center)
}

/// A pixel-plane point back in frame units (flat view, z = 0).
fn frame_of(view: &View, p: [f64; 2]) -> [f64; 3] {
    let m = map(view);
    [(p[0] - m.x0) / m.xx, (p[1] - m.y0) / m.yy, 0.0]
}

/// Straight segments through `points` in CE's cubic layout.
fn lines(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

/// A closed polygon's outline in CE's cubic layout (its first point repeated at the end).
fn ring(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut closed = points.to_vec();
    closed.push(points[0]);
    lines(&closed)
}

fn path(points: Vec<[f64; 3]>) -> Shape {
    Shape::new(ShapeData::Path { points })
}

fn fill(slot: u32, shape: &Shape, rgba: [f64; 4]) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key);
    r.paint.fill = Paint::One(rgba);
    r
}

fn stroke(slot: u32, shape: &Shape, rgba: [f64; 4], width: f64) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key);
    r.paint.stroke = Paint::One(rgba);
    r.paint.stroke_width = width;
    r
}

fn stream(view: View, items: &[(Shape, Record)], paints: Vec<Stops>) -> Stream {
    let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
    let mut d = arachne_core::probes::first_delta(view, items, &order);
    d.upload_paints = paints;
    let mut s = Stream::new();
    s.apply(&d).expect("the delta applies");
    s
}

fn draw(view: View, items: &[(Shape, Record)]) -> Vec<u8> {
    stream(view, items, Vec::new()).draw(W, H, 1).unwrap()
}

/// White over black at an exact area, as the coverage resolve gives it.
fn white_at(area: f64) -> [u8; 4] {
    blend([255; 4], area_coverage(area as f32), BLACK)
}

/// Every pixel against white over black at `area(x, y)`: within one step per byte (the oracle's
/// area and the core's round differently in their last bits), and exact at most pixels.
fn assert_areas(frame: &[u8], area: impl Fn(u32, u32) -> f64, what: &str) {
    let (mut partial, mut exact) = (0, 0);
    for y in 0..H {
        for x in 0..W {
            let a = area(x, y);
            let (got, want) = (px(frame, x, y), white_at(a));
            for k in 0..3 {
                assert!(
                    (got[k] as i32 - want[k] as i32).abs() <= 1,
                    "{what}: pixel {x},{y}: {got:?}, area {a}"
                );
            }
            if a > 1e-6 && a < 1.0 - 1e-6 {
                partial += 1;
                exact += (got == want) as u32;
            }
        }
    }
    assert!(partial > 20, "{what}: {partial} partial pixels");
    assert!(
        10 * exact >= 9 * partial,
        "{what}: {exact} of {partial} exact"
    );
}

#[test]
fn a_fill_covers_each_pixel_by_its_exact_area() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let corners = [
        [-3.1, -2.2, 0.0],
        [2.7, -2.9, 0.0],
        [3.6, 0.4, 0.0],
        [0.3, 3.1, 0.0],
        [-3.3, 1.2, 0.0],
    ];
    let shape = path(ring(&corners));
    let frame = draw(view.clone(), &[(shape.clone(), fill(1, &shape, WHITE))]);
    let m = map(&view);
    let poly: Vec<[f64; 2]> = corners.iter().map(|p| m.apply(*p)).collect();
    assert_areas(&frame, |x, y| area_in_pixel(&poly, x, y), "pentagon");
}

/// A star drawn as one self-crossing outline: the nonzero rule fills its middle too.
#[test]
fn a_self_crossing_outline_fills_by_the_nonzero_rule() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let tau = std::f64::consts::TAU;
    let r = 3.5;
    let at = |k: f64, radius: f64| {
        let a = tau * k / 10.0 + 0.3;
        [radius * libm::cos(a), radius * libm::sin(a), 0.0]
    };
    let star: Vec<[f64; 3]> = (0..5).map(|i| at(4.0 * i as f64, r)).collect();
    let shape = path(ring(&star));
    let frame = draw(view.clone(), &[(shape.clone(), fill(1, &shape, WHITE))]);
    // What nonzero fills: the star's outline, its inner corners where its edges cross.
    let inner = r * libm::cos(tau / 5.0) / libm::cos(tau / 10.0);
    let m = map(&view);
    let outline: Vec<[f64; 2]> = (0..10)
        .map(|k| m.apply(at(k as f64, if k % 2 == 0 { r } else { inner })))
        .collect();
    assert_areas(&frame, |x, y| area_in_pixel(&outline, x, y), "star");
}

/// An "O" as a glyph's outlines are: an outer contour and an inner one wound the other way, in one
/// path. Wound the same way, nonzero fills the inside.
#[test]
fn a_glyphs_inner_contour_wound_the_other_way_is_a_hole() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let tau = std::f64::consts::TAU;
    let circle = |r: f64, reversed: bool| -> Vec<[f64; 3]> {
        (0..48)
            .map(|i| {
                let t = tau * i as f64 / 48.0 * if reversed { -1.0 } else { 1.0 };
                [0.4 + r * libm::cos(t), -0.3 + r * libm::sin(t), 0.0]
            })
            .collect()
    };
    let m = map(&view);
    let poly = |pts: &[[f64; 3]]| pts.iter().map(|p| m.apply(*p)).collect::<Vec<_>>();
    let (outer, hole, same) = (circle(3.0, false), circle(1.6, true), circle(1.6, false));
    for (inner, is_hole) in [(&hole, true), (&same, false)] {
        let mut points = ring(&outer);
        points.extend(ring(inner));
        let shape = path(points);
        let frame = draw(view.clone(), &[(shape.clone(), fill(1, &shape, WHITE))]);
        let (o, i) = (poly(&outer), poly(inner));
        assert_areas(
            &frame,
            |x, y| {
                let a = area_in_pixel(&o, x, y);
                if is_hole {
                    a - area_in_pixel(&i, x, y)
                } else {
                    a
                }
            },
            if is_hole { "hole" } else { "filled" },
        );
    }
}

#[test]
fn a_pixel_aligned_rectangle_covers_whole_pixels_exactly() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let (x0, y0, x1, y1) = (23.0, 11.0, 131.0, 67.0);
    let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|p| frame_of(&view, p));
    let shape = path(ring(&corners));
    let frame = draw(view, &[(shape.clone(), fill(1, &shape, WHITE))]);
    for y in 0..H {
        for x in 0..W {
            let inside = (x as f64) >= x0 && (x as f64) < x1 && (y as f64) >= y0 && (y as f64) < y1;
            let want = if inside { [255; 4] } else { BLACK };
            assert_eq!(px(&frame, x, y), want, "pixel {x},{y}");
        }
    }
}

/// A butt-capped straight stroke is a rectangle CE's width times the view's line width multiple
/// across, in the frame's units.
#[test]
fn a_stroke_covers_its_outline_at_ces_width() {
    let mut view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    view.cairo_line_width_multiple = 0.013;
    let m = map(&view);
    for (a, b) in [
        ([-3.0, 0.37, 0.0], [3.2, 0.37, 0.0]),
        ([-2.6, -2.1, 0.0], [2.9, 1.7, 0.0]),
    ] {
        let shape = path(lines(&[a, b]));
        let frame = draw(
            view.clone(),
            &[(shape.clone(), stroke(1, &shape, WHITE, 60.0))],
        );
        let half = 60.0 * 0.013 / 2.0;
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        let n = [-dy / len * half, dx / len * half];
        let rect: Vec<[f64; 2]> = [
            [a[0] + n[0], a[1] + n[1], 0.0],
            [b[0] + n[0], b[1] + n[1], 0.0],
            [b[0] - n[0], b[1] - n[1], 0.0],
            [a[0] - n[0], a[1] - n[1], 0.0],
        ]
        .iter()
        .map(|p| m.apply(*p))
        .collect();
        assert_areas(&frame, |x, y| area_in_pixel(&rect, x, y), "stroke");
    }
}

/// A joint whose miter is 5.8 times the stroke's half width: Cairo's limit of 10 mitres it (kurbo's
/// own default of 4 would bevel it); a sharper one, past the limit, is bevelled.
#[test]
fn a_sharp_joint_is_mitred_up_to_cairos_limit() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let m = map(&view);
    let half = 40.0 * 0.01 / 2.0;
    for (angle, mitred) in [(20.0f64, true), (10.0, false)] {
        let h = angle.to_radians() / 2.0;
        let v = [0.0, -1.5, 0.0];
        let arm = |s: f64| {
            [
                v[0] + s * 3.0 * libm::sin(h),
                v[1] + 3.0 * libm::cos(h),
                0.0,
            ]
        };
        let shape = path(lines(&[arm(-1.0), v, arm(1.0)]));
        let frame = draw(
            view.clone(),
            &[(shape.clone(), stroke(1, &shape, WHITE, 40.0))],
        );
        // Halfway from the joint to where a miter's tip is.
        let tip = half / libm::sin(h);
        let [x, y] = m.apply([v[0], v[1] - 0.5 * tip.min(1.2), 0.0]);
        let got = px(&frame, x as u32, y as u32);
        assert_eq!(got != BLACK, mitred, "{angle} degrees: {got:?}");
    }
}

/// CE splits a path where a cubic's end and the next one's start differ, and closes a subpath whose
/// first and last points are equal within its tolerance (relative to the last).
#[test]
fn subpaths_split_and_close_by_ces_rule() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let m = map(&view);
    let square = |cx: f64| {
        [
            [cx - 1.0, 1.0, 0.0],
            [cx + 1.0, 1.0, 0.0],
            [cx + 1.0, -1.0, 0.0],
            [cx - 1.0, -1.0, 0.0],
        ]
    };
    // Two squares in one path: nothing joins them.
    let mut points = ring(&square(-2.5));
    points.extend(ring(&square(2.5)));
    let shape = path(points);
    let frame = draw(
        view.clone(),
        &[(shape.clone(), stroke(1, &shape, WHITE, 30.0))],
    );
    let [x, y] = m.apply([0.0, 0.0, 0.0]);
    assert_eq!(px(&frame, x as u32, y as u32), BLACK, "between the squares");
    let [x, y] = m.apply([-3.5, 0.0, 0.0]);
    assert_eq!(
        px(&frame, x as u32, y as u32),
        [255; 4],
        "on the left square"
    );

    // One square whose last point is moved: closed within the tolerance (its start corner is
    // mitred), open beyond it (butt caps leave the corner's outside empty).
    let half = 60.0 * 0.01 / 2.0;
    let corner = square(0.0)[0];
    let outside = [corner[0] - 0.5 * half, corner[1] + 0.5 * half, 0.0];
    for (shift, closed) in [(0.0, true), (1e-7, true), (0.05, false)] {
        let mut points = ring(&square(0.0));
        let last = points.len() - 1;
        points[last][0] += shift;
        let shape = path(points);
        let frame = draw(
            view.clone(),
            &[(shape.clone(), stroke(1, &shape, WHITE, 60.0))],
        );
        let [x, y] = m.apply(outside);
        let want = if closed { [255; 4] } else { BLACK };
        assert_eq!(px(&frame, x as u32, y as u32), want, "shifted by {shift}");
    }
}

/// A dash's period is in the frame's units, its duty the share drawn, its phase the share of a
/// period the pattern starts into.
#[test]
fn a_dashed_stroke_follows_its_period_duty_and_phase() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let m = map(&view);
    let (x0, x1) = (-4.0, 4.0);
    for (period, duty, phase) in [(1.0, 0.5, 0.0), (1.2, 0.25, 0.0), (1.0, 0.5, 0.25)] {
        let shape = path(lines(&[[x0, 0.0, 0.0], [x1, 0.0, 0.0]]));
        let mut r = stroke(1, &shape, WHITE, 40.0);
        r.dash = Some(Dash {
            period,
            duty,
            phase,
        });
        let frame = draw(view.clone(), &[(shape.clone(), r)]);
        let mut checked = 0;
        for x in 0..W {
            let [fx, _, _] = frame_of(&view, [x as f64 + 0.5, 0.0]);
            if fx < x0 + 0.1 || fx > x1 - 0.1 {
                continue;
            }
            let s = (fx - x0 + phase * period).rem_euclid(period);
            let edge = s.min((s - duty * period).abs()).min(period - s);
            if edge < 0.15 {
                continue; // Within a pixel and a half of a dash's end.
            }
            let [_, y] = m.apply([fx, 0.0, 0.0]);
            let on = s < duty * period;
            let got = px(&frame, x, y as u32);
            assert_eq!(
                got == [255; 4],
                on,
                "{period} {duty} {phase}: x {x}: {got:?}"
            );
            assert!(got == [255; 4] || got == BLACK, "x {x}: {got:?}");
            checked += 1;
        }
        assert!(checked > 25, "{checked} pixels checked");
    }
}

/// A fill and a stroke whose points lie a million frame widths out: drawn where they cross the
/// frame, quickly.
#[test]
fn a_path_far_beyond_the_frame_draws_where_it_crosses_it() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let far = 1.0e7;
    let big = path(ring(&[
        [-far, -far, 0.0],
        [far, -far, 0.0],
        [far, far, 0.0],
        [-far, far, 0.0],
    ]));
    let frame = draw(view.clone(), &[(big.clone(), fill(1, &big, WHITE))]);
    assert!(
        frame.chunks(4).all(|p| p == [255; 4]),
        "the fill covers the frame"
    );

    let line = path(lines(&[[-far, 0.25, 0.0], [far, 0.25, 0.0]]));
    let frame = draw(
        view.clone(),
        &[(line.clone(), stroke(1, &line, WHITE, 60.0))],
    );
    let m = map(&view);
    let (top, bottom) = (m.apply([0.0, 0.55, 0.0])[1], m.apply([0.0, -0.05, 0.0])[1]);
    for y in 0..H {
        let (a, b) = (y as f64, y as f64 + 1.0);
        let area = (b.min(bottom) - a.max(top)).max(0.0);
        for x in [0, W / 2, W - 1] {
            let (got, want) = (px(&frame, x, y), white_at(area));
            assert!(
                (0..3).all(|k| (got[k] as i32 - want[k] as i32).abs() <= 1),
                "pixel {x},{y}: {got:?} against {want:?}"
            );
        }
    }
}

/// A 3D view from 40 degrees off the vertical.
fn looking_down() -> View {
    arachne_core::probes::view(W, H, 40f64.to_radians(), -60f64.to_radians())
}

fn on_plane(x: f64, y: f64) -> [f64; 3] {
    [x, y, 0.37 * x - 0.21 * y + 0.13]
}

/// A mesh on a slanted plane, triangulated two ways, and a filled path on the same plane: the
/// later on top wherever they overlap, in either order, and either triangulation the same bytes.
#[test]
fn a_path_on_a_meshs_plane_ties_with_it_by_draw_order() {
    let view = looking_down();
    let quad = [(-3.3, -2.7), (3.1, -2.9), (3.4, 2.6), (-2.9, 2.8)].map(|(x, y)| on_plane(x, y));
    let mesh = |triangles: Vec<[u32; 3]>| {
        Shape::new(ShapeData::Mesh {
            positions: quad.to_vec(),
            triangles,
            normals: None,
        })
    };
    let meshes = [
        mesh(vec![[0, 1, 2], [0, 2, 3]]),
        mesh(vec![[0, 1, 3], [1, 2, 3]]),
    ];
    let label = [(-2.1, -1.7), (2.3, -1.9), (2.2, 1.8), (-1.9, 1.6)].map(|(x, y)| on_plane(x, y));
    let label_shape = path(ring(&label));
    let (green, white) = ([0.1, 0.6, 0.2, 1.0], WHITE);
    let mut label_record = fill(2, &label_shape, white);
    label_record.depth.space = Space::World;
    let label_px: Vec<[f64; 2]> = label
        .iter()
        .map(|p| arachne_core::probes::screen_of(&view, W, H, *p))
        .collect();
    let inside = pixels_inside(&label_px, 1.0, W, H);
    assert!(
        inside.len() > 300,
        "{} pixels inside the label",
        inside.len()
    );
    let mut frames = Vec::new();
    for shape in &meshes {
        let mut mesh_record = Record::new(RecordKind::Mesh, 1, 101);
        mesh_record.geometry.shape_a = Some(shape.key);
        mesh_record.paint.fill = Paint::One(green);
        mesh_record.depth.space = Space::World;
        for path_on_top in [true, false] {
            let mut items = vec![
                (shape.clone(), mesh_record.clone()),
                (label_shape.clone(), label_record.clone()),
            ];
            if !path_on_top {
                items.reverse();
            }
            let frame = draw(view.clone(), &items);
            let want = premultiply(if path_on_top { white } else { green });
            for (x, y) in &inside {
                assert_eq!(
                    px(&frame, *x, *y),
                    want,
                    "path on top {path_on_top}: {x},{y}"
                );
            }
            frames.push(frame);
        }
    }
    assert!(
        frames[0] == frames[2],
        "the path on top: the triangulations differ"
    );
    assert!(
        frames[1] == frames[3],
        "the mesh on top: the triangulations differ"
    );
}

fn points_record(slot: u32, shape: &Shape, colours: &Stops, radius: f64) -> Record {
    let mut r = Record::new(RecordKind::Points, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key);
    r.body = Body::Points(Points {
        colors: colours.key,
        radius,
    });
    r
}

/// CE draws a point as a block `int(thickness)` pixels across at its subpixel place; the core
/// draws a disc that wide there, in its own colour.
#[test]
fn a_point_is_a_disc_of_ces_thickness_at_its_place() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let places = vec![[-2.03, 0.71, 0.0], [2.44, -1.18, 0.0]];
    let colours = Stops::new(vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]);
    let shape = path(places.clone());
    let centres = subpixel_coords(
        &places,
        W,
        H,
        view.frame_width,
        view.frame_height,
        view.frame_center,
    );
    for (radius, diameter) in [(6.7, 6.0), (11.0, 11.0), (0.9, 0.0)] {
        let s = stream(
            view.clone(),
            &[(shape.clone(), points_record(1, &shape, &colours, radius))],
            vec![colours.clone()],
        );
        let frame = s.draw(W, H, 1).unwrap();
        for (i, c) in centres.iter().enumerate() {
            let channel = i; // red, then green
            let (mut area, mut mx, mut my) = (0.0, 0.0, 0.0);
            for y in 0..H {
                for x in 0..W {
                    let (dx, dy) = (x as f64 + 0.5 - c[0], y as f64 + 0.5 - c[1]);
                    if dx * dx + dy * dy > 16.0 * 16.0 {
                        continue;
                    }
                    let a = px(&frame, x, y)[channel] as f64 / 255.0;
                    area += a;
                    mx += a * (x as f64 + 0.5);
                    my += a * (y as f64 + 0.5);
                }
            }
            let want = std::f64::consts::PI * diameter * diameter / 4.0;
            assert!(
                (area - want).abs() < 0.02 * want + 0.5,
                "r {radius}: area {area} of {want}"
            );
            if diameter > 0.0 {
                let (cx, cy) = (mx / area, my / area);
                assert!(
                    (cx - c[0]).abs() < 0.05 && (cy - c[1]).abs() < 0.05,
                    "centre {cx},{cy}"
                );
                let full = px(&frame, c[0] as u32, c[1] as u32);
                assert_eq!(full[channel], 255, "r {radius}: the disc's middle");
            }
        }
    }
}

/// Under the 3D camera a point sits at its own depth: hidden behind a level sheet from below it,
/// shown in front of it from above.
#[test]
fn a_point_is_drawn_at_its_depth() {
    let view = three_d_view(W, H);
    let sheet = Shape::new(ShapeData::Mesh {
        positions: vec![
            [-3.0, -3.0, 0.0],
            [3.0, -3.0, 0.0],
            [3.0, 3.0, 0.0],
            [-3.0, 3.0, 0.0],
        ],
        triangles: vec![[0, 1, 2], [0, 2, 3]],
        normals: None,
    });
    let mut floor = Record::new(RecordKind::Mesh, 1, 101);
    floor.geometry.shape_a = Some(sheet.key);
    floor.paint.fill = Paint::One([0.1, 0.2, 0.7, 1.0]);
    floor.depth.space = Space::World;
    let colours = Stops::new(vec![[1.0, 1.0, 0.0, 1.0]]);
    for (z, shown) in [(-1.0, false), (1.0, true)] {
        let cloud = path(vec![[0.5, 0.3, z]]);
        let mut dot = points_record(2, &cloud, &colours, 9.0);
        dot.depth.space = Space::World;
        let s = stream(
            view.clone(),
            &[(sheet.clone(), floor.clone()), (cloud.clone(), dot)],
            vec![colours.clone()],
        );
        let frame = s.draw(W, H, 1).unwrap();
        let [x, y] = arachne_core::probes::screen_of(&view, W, H, [0.5, 0.3, z]);
        let got = px(&frame, x as u32, y as u32);
        let want = if shown {
            premultiply([1.0, 1.0, 0.0, 1.0])
        } else {
            premultiply([0.1, 0.2, 0.7, 1.0])
        };
        assert_eq!(got, want, "z {z}");
    }
}

fn shift(dx: f64, dy: f64, dz: f64) -> Affine {
    [
        [1.0, 0.0, 0.0, dx],
        [0.0, 1.0, 0.0, dy],
        [0.0, 0.0, 1.0, dz],
    ]
}

/// An instances record of a filled, stroked path, its members overlapping, draws the bytes its
/// members draw as path records one after another; likewise a mesh prototype under the 3D camera.
#[test]
fn an_instances_record_draws_what_its_members_draw_as_records() {
    let squash = [
        [0.8, 0.3, 0.0, 0.1],
        [-0.2, 0.9, 0.0, -0.2],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let placements = vec![
        shift(-1.0, 0.0, 0.0),
        shift(-0.2, 0.4, 0.3),
        shift(0.6, 0.8, -0.4),
    ];
    let fills = Stops::new(vec![
        [0.9, 0.1, 0.1, 1.0],
        [0.1, 0.8, 0.2, 0.6],
        [0.2, 0.3, 0.9, 1.0],
    ]);
    let strokes = Stops::new(vec![
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.9, 0.2, 1.0],
        [0.2, 0.9, 0.9, 0.5],
    ]);
    let square = path(ring(&[
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ]));
    let quad = common::meshes::quad([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.2],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, -0.3],
    ]);
    for (view, prototype) in [
        (flat_view(W, H, [0.0, 0.0, 0.0, 1.0]), &square),
        (looking_down(), &square),
        (looking_down(), &quad),
    ] {
        let is_path = matches!(prototype.data, ShapeData::Path { .. });
        let mut member = Record::new(
            if is_path {
                RecordKind::Path
            } else {
                RecordKind::Mesh
            },
            1,
            101,
        );
        member.geometry.shape_a = Some(prototype.key);
        member.geometry.place_a = squash;
        member.paint.fill = Paint::One(WHITE);
        if is_path {
            member.paint.stroke = Paint::One(WHITE);
            member.paint.stroke_width = 30.0;
            member.paint.background_stroke = Paint::One([0.0, 0.0, 0.0, 1.0]);
            member.paint.background_stroke_width = 60.0;
        }
        member.depth.space = Space::World;

        let mut batch = member.clone();
        batch.kind = RecordKind::Instances;
        batch.body = Body::Instances(Instances {
            placements: placements.clone(),
            fill: fills.key,
            stroke: strokes.key,
            objects: vec![201, 202, 203],
        });
        let batched = stream(
            view.clone(),
            &[(prototype.clone(), batch)],
            vec![fills.clone(), strokes.clone()],
        )
        .draw(W, H, 1)
        .unwrap();

        let members: Vec<(Shape, Record)> = (0..3)
            .map(|i| {
                let mut r = member.clone();
                r.slot = 1 + i as u32;
                r.object = 201 + i as u32;
                r.geometry.place_a = compose(&placements[i], &squash);
                r.paint.fill = Paint::One(fills.rows[i]);
                if is_path {
                    r.paint.stroke = Paint::One(strokes.rows[i]);
                }
                (prototype.clone(), r)
            })
            .collect();
        let one_by_one = draw(view.clone(), &members);
        assert!(batched == one_by_one, "{:?} prototype", prototype.data);
        let background = px(&one_by_one, 0, 0);
        let drawn = one_by_one.chunks(4).filter(|p| *p != background).count();
        assert!(drawn > 150, "{drawn} pixels drawn");
    }
}

#[test]
fn what_paths_and_points_do_not_draw_yet_is_refused_whole() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let shape = path(ring(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]));
    let other = path(ring(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]));
    // A per-point table: a row for each of the shape's 12 points.
    let table = Stops::new(vec![WHITE; 12]);
    let base = {
        let mut r = fill(1, &shape, WHITE);
        r.paint.stroke = Paint::One(WHITE);
        r.paint.stroke_width = 4.0;
        r
    };
    type Change = fn(&mut Record, Key);
    let cases: [(&str, Change); 10] = [
        ("blend", |r, k| {
            r.geometry.blend = Some(Blend {
                shape_b: k,
                place_b: shift(0.0, 0.0, 0.0),
                t: 0.5,
            })
        }),
        ("reveal", |r, _| r.geometry.reveal = Some((0.0, 0.5))),
        ("paint blend", |r, _| {
            r.paint.paint_b = Some(PaintB {
                fill: Paint::One(WHITE),
                stroke: Paint::None,
                background_stroke: Paint::None,
                stroke_width: 0.0,
                background_stroke_width: 0.0,
            })
        }),
        ("clip", |r, k| r.clip = Some((k, shift(0.0, 0.0, 0.0)))),
        ("gradient", |r, _| {
            r.paint.stroke = Paint::Two([WHITE, [0.0, 0.0, 0.0, 1.0]])
        }),
        ("gradient", |r, _| {
            r.paint.fill = Paint::Table(Stops::new(vec![WHITE; 12]).key)
        }),
        ("dash", |r, _| {
            r.dash = Some(Dash {
                period: 0.0,
                duty: 0.5,
                phase: 0.0,
            })
        }),
        ("dash", |r, _| {
            r.dash = Some(Dash {
                period: 1.0,
                duty: 0.0,
                phase: 0.0,
            })
        }),
        ("dash", |r, _| {
            r.dash = Some(Dash {
                period: 1.0,
                duty: 1.5,
                phase: 0.0,
            })
        }),
        ("dash", |r, _| {
            r.dash = Some(Dash {
                period: 1.0,
                duty: 0.5,
                phase: f64::NAN,
            })
        }),
    ];
    for (what, change) in cases {
        let mut r = base.clone();
        change(&mut r, other.key);
        let mut d = arachne_core::probes::first_delta(
            view.clone(),
            &[(shape.clone(), r), (other.clone(), fill(2, &other, WHITE))],
            &[1],
        );
        d.upload_paints = vec![table.clone()];
        let mut s = Stream::new();
        s.apply(&d).unwrap();
        match s.draw(W, H, 1) {
            Err(DrawError::Unsupported(u)) => {
                assert_eq!((u.kind.as_str(), u.slot), ("path", Some(1)), "{what}");
                assert!(u.reason.contains(what), "{what}: {}", u.reason);
            }
            other => panic!("{what}: {:?}", other.map(|f| f.len())),
        }
    }
    // A gradient on a stroke of no width is never drawn, so not refused.
    let mut r = base.clone();
    r.paint.stroke = Paint::Two([WHITE, WHITE]);
    r.paint.stroke_width = 0.0;
    assert!(draw_one(&view, &shape, r).is_ok());

    let colours = Stops::new(vec![WHITE; 12]);
    let mut dots = points_record(1, &shape, &colours, 4.0);
    dots.geometry.reveal = Some((0.0, 0.5));
    let mut d = arachne_core::probes::first_delta(view, &[(shape.clone(), dots)], &[1]);
    d.upload_paints = vec![colours];
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    let Err(DrawError::Unsupported(u)) = s.draw(W, H, 1) else {
        panic!("a points record with a reveal draws")
    };
    assert_eq!(u.kind, "points");
    assert!(u.reason.contains("reveal"), "{}", u.reason);
}

fn draw_one(view: &View, shape: &Shape, r: Record) -> Result<Vec<u8>, DrawError> {
    let d = arachne_core::probes::first_delta(view.clone(), &[(shape.clone(), r)], &[1]);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    s.draw(W, H, 1)
}

/// The nonzero winding of polygons at a point (a ray to the left).
fn winding(rings: &[Vec<[f64; 2]>], p: [f64; 2]) -> i32 {
    let mut w = 0;
    for ring in rings {
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            let (lo, hi, dir) = if a[1] < b[1] { (a, b, 1) } else { (b, a, -1) };
            if lo[1] <= p[1] && p[1] < hi[1] {
                let x = lo[0] + (p[1] - lo[1]) / (hi[1] - lo[1]) * (hi[0] - lo[0]);
                if x < p[0] {
                    w += dir;
                }
            }
        }
    }
    w
}

/// Overlapping, self-crossing rings: each pixel's mask is the winding at its sub-samples, and its
/// coverage the area where the winding is not zero (against 128 x 128 samples).
#[test]
fn the_scanlines_masks_and_areas_follow_the_nonzero_winding() {
    let tau = std::f64::consts::TAU;
    let star: Vec<[f64; 2]> = (0..7)
        .map(|i| {
            let a = tau * (3 * i) as f64 / 7.0 + 0.2;
            [20.3 + 13.7 * libm::cos(a), 18.1 + 13.9 * libm::sin(a)]
        })
        .collect();
    let box_ = vec![[24.6, 9.2], [41.3, 12.8], [37.9, 33.4], [21.1, 30.7]];
    let rings = vec![star, box_];
    let edges = scanline::edges_of(&rings);
    let all: Vec<u32> = (0..edges.len() as u32).collect();
    let rect = arachne_core::composite::Rect {
        x0: 0,
        y0: 0,
        x1: 48,
        y1: 40,
    };
    let mut areas = vec![0.0; 48 * 40];
    scanline::cover(
        &edges,
        &all,
        rect,
        &mut scanline::Scratch::default(),
        |y, x0, x1, a| {
            for x in x0..x1 {
                areas[(y * 48 + x) as usize] = a;
            }
        },
    );
    let mut worst: f64 = 0.0;
    for y in 0..40 {
        for x in 0..48 {
            let mask = scanline::mask(&edges, &all, x, y);
            for s in 0..SAMPLES {
                let on = winding(&rings, sample_point(x, y, s)) != 0;
                assert_eq!(mask >> s & 1 == 1, on, "pixel {x},{y} sample {s}");
            }
            let n = 128;
            let inside = (0..n * n)
                .filter(|k| {
                    let p = [
                        x as f64 + ((k % n) as f64 + 0.5) / n as f64,
                        y as f64 + ((k / n) as f64 + 0.5) / n as f64,
                    ];
                    winding(&rings, p) != 0
                })
                .count();
            let err = (areas[(y * 48 + x) as usize] - inside as f64 / (n * n) as f64).abs();
            worst = worst.max(err);
        }
    }
    // 128 samples a side resolve an edge to within about 1/128 of the pixel.
    assert!(worst < 0.02, "worst area error {worst}");
}
