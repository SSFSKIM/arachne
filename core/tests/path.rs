//! Paths and points drawn through a Stream: fills by the nonzero rule at their exact area, glyph
//! holes, pixel-aligned rectangles whole, strokes at CE's width with Cairo's miter limit, CE's
//! subpath split and close rule, dashes, paths far beyond the frame, coplanar paths and meshes
//! tying exactly, points as CE's square blocks, instances drawing what their members drawn as
//! records draw, what paths do not draw yet refused, and the scanline's masks against point
//! winding.

mod common;

use arachne_core::composite::{SAMPLES, area_coverage, blend, premultiply, sample_point};
use arachne_core::depth::{Planarity, WorldPlane};
use arachne_core::project::{FrameMap, Placement, Projection, pre_display, subpixel_coords};
use arachne_core::scanline;
use arachne_core::stream::{DrawError, Stream};
use arachne_core::xf::*;
use common::meshes::{area_in_pixel, pixels_inside};
use common::{flat_view, three_d_view};

const W: u32 = 160;
const H: u32 = 90;
const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
const BLACK: [u8; 4] = [0, 0, 0, 255];
/// The grid the engine snaps geometry to at a 1920-pixel frame of CE's width: positions the core
/// is handed lie on it, and its plane fits are told so.
const G: f64 = 1.0 / 32768.0;

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
    r.geometry.shape_a = Some(shape.key());
    r.paint.fill = Paint::One(rgba);
    r
}

fn stroke(slot: u32, shape: &Shape, rgba: [f64; 4], width: f64) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
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

/// A dashed stroke that leaves the frame far beyond its guard band, curves there and comes back:
/// the dashes it shows on its way back are where the whole curve's length puts them, as if nothing
/// outside the frame had been simplified.
#[test]
fn a_dashed_stroke_returning_from_far_beyond_the_frame_keeps_its_dash_phase() {
    use kurbo::{CubicBez, ParamCurveArclen, Point};
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let m = map(&view);
    let (x0, far, period, duty) = (-4.0, 1.0e5, 1.0, 0.5);
    // A bend whose length differs from its chord's (2) by about half a period, so drawing the
    // chord instead would swap the returning stroke's dashes and gaps.
    let bend = |d: f64| {
        [
            [far, 1.0],
            [far + d, 1.0 + d],
            [far + d, -1.0 - d],
            [far, -1.0],
        ]
    };
    let arc = |d: f64| {
        let p = bend(d).map(|[x, y]| Point::new(x, y));
        CubicBez::new(p[0], p[1], p[2], p[3]).arclen(1e-9)
    };
    let d = (1..200)
        .map(|k| k as f64 * 0.01)
        .find(|d| ((arc(*d) - 2.0).rem_euclid(period) - 0.5 * period).abs() < 0.1)
        .expect("a bend about half a period longer than its chord");
    let mut points = lines(&[[x0, 1.0, 0.0], [far, 1.0, 0.0]]);
    points.extend(bend(d).map(|[x, y]| [x, y, 0.0]));
    points.extend(lines(&[[far, -1.0, 0.0], [x0, -1.0, 0.0]]));
    let shape = path(points);
    let mut r = stroke(1, &shape, WHITE, 40.0);
    r.dash = Some(Dash {
        period,
        duty,
        phase: 0.0,
    });
    let frame = draw(view.clone(), &[(shape.clone(), r)]);
    let mut checked = [0, 0];
    let bend_length = arc(d);
    // The stroke's length from its start to `fx` on the way out (y = 1) and on the way back.
    for (k, y_frame) in [(0, 1.0), (1, -1.0)] {
        for x in 0..W {
            let [fx, _, _] = frame_of(&view, [x as f64 + 0.5, 0.0]);
            if fx < x0 + 0.1 {
                continue;
            }
            let travelled = if k == 0 {
                fx - x0
            } else {
                (far - x0) + bend_length + (far - fx)
            };
            let s = travelled.rem_euclid(period);
            let edge = s.min((s - duty * period).abs()).min(period - s);
            if edge < 0.15 {
                continue; // Within a pixel and a half of a dash's end.
            }
            let [_, y] = m.apply([fx, y_frame, 0.0]);
            let got = px(&frame, x, y as u32);
            assert_eq!(
                got == [255; 4],
                s < duty * period,
                "stroke at y {y_frame}: x {x}: {got:?}"
            );
            assert!(got == [255; 4] || got == BLACK, "x {x}: {got:?}");
            checked[k] += 1;
        }
    }
    assert!(
        checked.iter().all(|c| *c > 25),
        "{checked:?} pixels checked"
    );
}

/// A closed dashed outline that starts at a corner in the frame, runs out, bends and comes back to
/// close at its start, its dash on at both ends: drawn with the bend inside the guard band (laid by
/// the stroker on the whole outline) and far beyond it (a whole number of periods farther out and
/// back, so the same dashes are on screen), the frame is the same, the closing dash joined to the
/// first at the corner in both.
#[test]
fn a_closed_dashed_outline_beyond_the_frame_draws_as_the_whole_outline_does() {
    use kurbo::{CubicBez, ParamCurveArclen, Point};
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let (period, duty) = (1.0, 0.8);
    let bend = |far: f64, d: f64| {
        [
            [-far, 0.0],
            [-far - d, 0.0 - d],
            [-far - d, -2.0 + d],
            [-far, -2.0],
        ]
    };
    let arc = |d: f64| {
        let p = bend(0.0, d).map(|[x, y]| Point::new(x, y));
        CubicBez::new(p[0], p[1], p[2], p[3]).arclen(1e-9)
    };
    // A bend whose length differs from its chord's (2) by about half a period.
    let d = (1..200)
        .map(|k| k as f64 * 0.01)
        .find(|d| ((arc(*d) - 2.0).rem_euclid(period) - 0.5 * period).abs() < 0.1)
        .expect("a bend about half a period longer than its chord");
    let (near, far) = (1000.0, 1.0e5);
    // The whole outline's length, near; far adds whole periods. The phase puts both ends on a dash.
    let length = 2.0 * near + arc(d) + 2.0;
    let on = |at: f64| (0.1..=duty - 0.1).contains(&(at.rem_euclid(period) / period));
    let phase = (1..100)
        .map(|k| k as f64 * 0.01 * period)
        .find(|x| on(*x) && on(x + length))
        .expect("a phase with a dash on at both ends");
    let outline = |far: f64| {
        let mut points = lines(&[[0.0, 0.0, 0.0], [-far, 0.0, 0.0]]);
        points.extend(bend(far, d).map(|[x, y]| [x, y, 0.0]));
        points.extend(lines(&[
            [-far, -2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]));
        let shape = path(points);
        let mut r = stroke(1, &shape, WHITE, 40.0);
        r.dash = Some(Dash {
            period,
            duty,
            phase: phase / period,
        });
        draw(view.clone(), &[(shape, r)])
    };
    let (inside, beyond) = (outline(near), outline(far));
    let lit = inside.chunks(4).filter(|p| p[0] > 128).count();
    assert!(lit > 100, "{lit} lit pixels");
    // The far outline's curves are cut into other pieces before they are stroked, and the stroker
    // fits each piece's sides within its tolerance of 0.1 px: edges may move that far, never a dash.
    let bound = (arachne_core::path::TOLERANCE * 255.0).ceil() as i32;
    for (i, (a, b)) in inside.iter().zip(&beyond).enumerate() {
        let (x, y) = ((i / 4) as u32 % W, (i / 4) as u32 / W);
        assert!(
            (*a as i32 - *b as i32).abs() <= bound,
            "pixel {x},{y}: {a} within the guard band, {b} beyond it"
        );
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

/// An outline crossing itself on a plane (a bowtie on z = x + y), whose Newell normal cancels to
/// zero, keeps its plane, the one a triangle of its points gives; open or closed. Points on a line
/// still have none.
#[test]
fn a_planar_outline_crossing_itself_keeps_its_plane() {
    let bowtie = [
        [-1.0, -1.0, -2.0],
        [1.0, 1.0, 2.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ];
    let want = WorldPlane::of_triangle([bowtie[0], bowtie[1], bowtie[2]], 0.0).unwrap();
    let mut closed = bowtie.to_vec();
    closed.push(bowtie[0]);
    for points in [bowtie.to_vec(), closed, lines(&bowtie)] {
        assert_eq!(WorldPlane::of_points(&points), Some(want), "{points:?}");
    }
    let skew = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.5],
    ];
    assert_eq!(WorldPlane::of_points(&skew), None, "not on one plane");
    let line = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]];
    assert_eq!(WorldPlane::of_points(&line), None, "on a line");
}

/// A surface's face, curved a little, takes the plane nearest it: the one a planar face would
/// have when it is flattened onto its mean, and its own when it is planar. Points on a line have
/// none.
#[test]
fn a_curved_face_takes_the_plane_nearest_it() {
    let flat = [
        [0.0, 0.0, 0.25],
        [1.0, 0.0, 0.25],
        [1.0, 1.0, 0.25],
        [0.0, 1.0, 0.25],
    ];
    let bent = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.5],
    ];
    assert_eq!(WorldPlane::of_points(&bent), None);
    assert_eq!(WorldPlane::nearest(&bent), WorldPlane::of_points(&flat));
    assert_eq!(WorldPlane::nearest(&flat), WorldPlane::of_points(&flat));
    let line = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]];
    assert_eq!(WorldPlane::nearest(&line), None);
}

/// A line whose points share a coordinate takes that coordinate's axis plane, and so does the
/// same line placed by arithmetic that lands a point an ulp off the shared value, either way:
/// machines whose maths differ in the last bit key it alike. A coordinate apart by far more than
/// rounding is not shared.
#[test]
fn a_line_sharing_a_coordinate_to_the_last_bit_takes_its_axis_plane() {
    let ulp_up = |v: f64| f64::from_bits(v.to_bits() + 1);
    let ulp_down = |v: f64| f64::from_bits(v.to_bits() - 1);
    for shared in [1.0, -1.0, 2.0, 0.37] {
        // A cubic from (1, shared, 1) to (-1, shared, 2), as an edge between two vertices.
        let exact: Vec<[f64; 3]> = (0..4)
            .map(|i| {
                let t = i as f64 / 3.0;
                [1.0 - 2.0 * t, shared, 1.0 + t]
            })
            .collect();
        let want = WorldPlane::of_points(&exact);
        assert_eq!(
            want.map(|p| p.normal),
            Some([0.0, 1.0, 0.0]),
            "the exact line at y = {shared}"
        );
        for nudge in [ulp_up, ulp_down] {
            for at in [0, 1, 3] {
                let mut off = exact.clone();
                off[at][1] = nudge(shared);
                assert_eq!(
                    WorldPlane::of_points(&off),
                    want,
                    "y = {shared}, point {at}"
                );
            }
        }
        let mut apart = exact.clone();
        apart[3][1] = shared + 1e-6;
        assert_ne!(WorldPlane::of_points(&apart), want, "apart by 1e-6");
    }
    // At zero an ulp is a subnormal-sized step, and the line still keys as the plane y = 0.
    let zero: Vec<[f64; 3]> = vec![[1.0, 0.0, 1.0], [-1.0, -f64::MIN_POSITIVE, 2.0]];
    assert_eq!(
        WorldPlane::of_points(&zero).map(|p| (p.normal, p.offset)),
        Some(([0.0, 1.0, 0.0], 0.0))
    );
}

/// Points on a line have no plane, exact or nearest, wherever arithmetic lands their last bits: a
/// one-ulp nudge must not give them a normal made of rounding noise, which would key a surface's
/// edge as an extrapolated plane on one machine and as a centreline on another. A triangle on a
/// line has none either.
#[test]
fn points_an_ulp_off_a_line_have_no_plane() {
    let ulp_up = |v: f64| f64::from_bits(v.to_bits() + 1);
    let ulp_down = |v: f64| f64::from_bits(v.to_bits() - 1);
    for nudge in [ulp_up, ulp_down] {
        let line = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, nudge(6.0)]];
        assert_eq!(WorldPlane::of_points(&line), None, "{line:?}");
        assert_eq!(WorldPlane::nearest(&line), None, "nearest {line:?}");
        assert_eq!(
            WorldPlane::of_triangle(line, 0.0),
            None,
            "triangle {line:?}"
        );
    }
}

/// A position as the engine hands it to the core: snapped to `grid`.
fn snapped_to(p: [f64; 3], grid: f64) -> [f64; 3] {
    p.map(|v| (v / grid).round_ties_even() * grid + 0.0)
}

/// A position snapped to the grid [`G`].
fn snapped(p: [f64; 3]) -> [f64; 3] {
    snapped_to(p, G)
}

/// A tilted plane's unit vectors, none along an axis.
fn tilted() -> ([f64; 3], [f64; 3]) {
    let u = [0.6, 0.64, 0.48];
    let raw: [f64; 3] = [-0.8, 0.48, 0.36];
    let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
    (u, raw.map(|x| x / norm))
}

/// A triangle the snap leaves a sliver a grid step or two wide has no world plane at that grid
/// (its normal is the snap's noise), and its depth comes from its own corners on screen, exact
/// for a triangle. The decision moves its depth only, never what it covers: drawn alone, it
/// draws the same pixels as when a plane is found for it.
#[test]
fn a_triangle_the_snap_makes_degenerate_draws_what_it_covers() {
    let view = looking_down();
    let (u, v) = tilted();
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let at = |s: f64, off: f64| -> [f64; 3] {
        snapped(std::array::from_fn(|k| {
            0.3 + s * u[k] + off * (0.6 * v[k] + 0.8 * n[k])
        }))
    };
    for length in [0.5, 2.0, 5.0] {
        let triangle = [at(-length, 0.0), at(0.4 * length, 1.5 * G), at(length, 0.0)];
        assert_eq!(
            WorldPlane::of_triangle(triangle, G),
            None,
            "length {length}"
        );
        let shape = Shape::new(ShapeData::Mesh {
            positions: triangle.to_vec(),
            triangles: vec![[0, 1, 2]],
            normals: None,
        });
        let mut record = Record::new(RecordKind::Mesh, 1, 101);
        record.geometry.shape_a = Some(shape.key());
        record.paint.fill = Paint::One(WHITE);
        record.depth.space = Space::World;
        assert!(
            WorldPlane::of_triangle(triangle, 0.0).is_some(),
            "length {length}"
        );
        let mut grid_view = view.clone();
        grid_view.input_grid = G;
        let items = [(shape, record)];
        assert!(
            draw(grid_view, &items) == draw(view.clone(), &items),
            "length {length}: the sliver draws other pixels"
        );
    }
}

/// SplitMix64: a seeded stream of numbers, the same on every machine.
struct Seeded(u64);

impl Seeded {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn between(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// A rotation drawn uniformly: the images of the x and y axes, and of z, its plane's normal.
    fn rotation(&mut self) -> [[f64; 3]; 3] {
        let q = loop {
            let q: [f64; 4] = std::array::from_fn(|_| self.between(-1.0, 1.0));
            let norm = q.iter().map(|v| v * v).sum::<f64>();
            if norm > 0.01 && norm <= 1.0 {
                break q.map(|v| v / norm.sqrt());
            }
        };
        let [w, x, y, z] = q;
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                2.0 * (x * z - w * y),
            ],
            [
                2.0 * (x * y - w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + w * x),
            ],
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    /// `v` moved 1 to 4 ulp, up or down.
    fn nudged(&mut self, v: f64) -> f64 {
        let steps = 1 + self.next() % 4;
        let up = self.next().is_multiple_of(2);
        (0..steps).fold(v, |v, _| if up { v.next_up() } else { v.next_down() })
    }
}

/// A face's outline in its own plane, at unit size: a Platonic solid's face (a regular triangle,
/// square or pentagon), a rectangle up to four times as long as it is wide, or a convex polygon of
/// up to twelve corners at uneven radii.
fn outline(seeded: &mut Seeded) -> Vec<[f64; 2]> {
    let regular = |n: u32, turn: f64| -> Vec<[f64; 2]> {
        (0..n)
            .map(|i| {
                let a = turn + std::f64::consts::TAU * f64::from(i) / f64::from(n);
                [libm::cos(a), libm::sin(a)]
            })
            .collect()
    };
    match seeded.next() % 5 {
        0 => regular(3, seeded.between(0.0, 1.0)),
        1 => regular(4, seeded.between(0.0, 1.0)),
        2 => regular(5, seeded.between(0.0, 1.0)),
        3 => {
            let (w, h) = (1.0, seeded.between(0.25, 1.0));
            vec![[-w, -h], [w, -h], [w, h], [-w, h]]
        }
        _ => {
            let n = 4 + (seeded.next() % 9) as u32;
            regular(n, seeded.between(0.0, 1.0))
                .into_iter()
                .map(|[x, y]| {
                    let r = seeded.between(0.7, 1.0);
                    [r * x, r * y]
                })
                .collect()
        }
    }
}

/// A seeded face: its outline at a random rotation, size from 0.01 to 100 and offset up to 50
/// from the origin, one corner (of four or more) moved off its plane by `bend` times its size.
fn seeded_face(seeded: &mut Seeded, bend: f64) -> Vec<[f64; 3]> {
    let [u, v, n] = seeded.rotation();
    let size = libm::pow(10.0, seeded.between(-2.0, 2.0));
    let centre: [f64; 3] = std::array::from_fn(|_| seeded.between(-50.0, 50.0));
    let shape = outline(seeded);
    let bent_at = if shape.len() > 3 {
        (seeded.next() as usize) % shape.len()
    } else {
        usize::MAX
    };
    shape
        .iter()
        .enumerate()
        .map(|(i, [x, y])| {
            let off = if i == bent_at { bend * size } else { 0.0 };
            std::array::from_fn(|k| centre[k] + size * (x * u[k] + y * v[k]) + off * n[k])
        })
        .collect()
}

/// A face decided flat before the snap keeps a fitted plane after it, at every grid the engine
/// snaps to; a face bent off its plane at one corner by a hundred grid steps is decided a curve,
/// and fits none.
#[test]
fn a_face_decided_flat_keeps_a_fitted_plane_under_the_snap() {
    let mut seeded = Seeded(0x5EED_F00D);
    for grid in [1.0 / 1_048_576.0, 1.0 / 65_536.0, G, 1.0 / 8_192.0] {
        for case in 0..3000 {
            let face = seeded_face(&mut seeded, 0.0);
            let snap = |points: &[[f64; 3]]| -> Vec<[f64; 3]> {
                points.iter().map(|p| snapped_to(*p, grid)).collect()
            };
            let what = format!("grid {grid}, case {case}: {face:?}");
            let decided = WorldPlane::decide(&face, false);
            assert!(
                matches!(decided, Planarity::Newell | Planarity::Widest),
                "flat, {what}: {decided:?}"
            );
            assert!(
                WorldPlane::fit(&snap(&face), decided).is_some(),
                "fit, {what}"
            );
            if face.len() > 3 {
                let mut bent = face.clone();
                let n = WorldPlane::fit(&face, decided).unwrap().normal;
                for k in 0..3 {
                    bent[1][k] += 100.0 * grid * n[k];
                }
                assert_eq!(
                    WorldPlane::decide(&bent, false),
                    Planarity::Curve,
                    "bent, {what}"
                );
            }
        }
    }
}

/// A face off its plane at one corner by 1e-6 to 1e-3 of its size is decided a curve, and a
/// flat face flat, whatever the last bits of its points: moved 1 to 4 ulp, each keeps its
/// decision. (A face within ulps of the bound, 1e-9 of its extent off its plane, can still be
/// decided apart on two machines: content reaches that only by coincidence.)
#[test]
fn a_planarity_decision_does_not_move_with_the_last_bits() {
    let mut seeded = Seeded(0xD1CE);
    for case in 0..4000 {
        let bend = match case % 4 {
            0 => 0.0,
            _ => libm::pow(10.0, seeded.between(-6.0, -3.0)),
        };
        let face = seeded_face(&mut seeded, bend);
        let decided = WorldPlane::decide(&face, false);
        let want = if bend == 0.0 || face.len() == 3 {
            matches!(decided, Planarity::Newell | Planarity::Widest)
        } else {
            decided == Planarity::Curve
        };
        assert!(want, "case {case}, bend {bend}: {decided:?} {face:?}");
        for _ in 0..4 {
            let nudged: Vec<[f64; 3]> = face.iter().map(|p| p.map(|v| seeded.nudged(v))).collect();
            assert_eq!(
                WorldPlane::decide(&nudged, false),
                decided,
                "case {case}, bend {bend}: {nudged:?}"
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
        mesh_record.geometry.shape_a = Some(shape.key());
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

/// The grid the engine snaps a camera's geometry to: `2^(e - 9)`, `e` the binary exponent of the
/// scene units one of its pixels spans.
fn camera_grid(view: &View) -> f64 {
    let (_, e) = libm::frexp(view.frame_width / f64::from(view.pixel_width));
    libm::ldexp(1.0, e - 9)
}

/// A path through `points` in world space as the engine hands it to the core: its points snapped
/// to `grid`, its planarity decided on them as content made them.
fn on_grid(record: impl Fn(&Shape) -> Record, points: &[[f64; 3]], grid: f64) -> (Shape, Record) {
    let shape = path(points.iter().map(|p| snapped_to(*p, grid)).collect());
    let mut r = record(&shape);
    r.depth.space = Space::World;
    r.planes = Some(vec![WorldPlane::decide(points, false).code()]);
    (shape, r)
}

/// A square on the tilted plane through `centre`: `half` its half side, `turn` its angle, `at`
/// its centre in the plane's coordinates.
fn tilted_square(centre: [f64; 3], half: f64, turn: f64, at: [f64; 2]) -> Vec<[f64; 3]> {
    let (u, v) = tilted();
    (0..4)
        .map(|i| {
            let a = turn + std::f64::consts::FRAC_PI_2 * f64::from(i);
            let (x, y) = (at[0] + half * libm::cos(a), at[1] + half * libm::sin(a));
            std::array::from_fn(|k| centre[k] + x * u[k] + y * v[k])
        })
        .collect()
}

/// The view [`looking_down`] with the grid the engine snaps its geometry to.
fn looking_down_on_grid() -> View {
    let mut view = looking_down();
    view.input_grid = camera_grid(&view);
    view
}

/// The pixels well inside a world polygon on screen.
fn inside_on_screen(view: &View, corners: &[[f64; 3]]) -> Vec<(u32, u32)> {
    let shown: Vec<[f64; 2]> = corners
        .iter()
        .map(|p| arachne_core::probes::screen_of(view, W, H, *p))
        .collect();
    pixels_inside(&shown, 1.0, W, H)
}

const CENTRES: [[f64; 3]; 3] = [
    [0.31, -0.42, 0.23],
    [-1.13, 0.61, -0.52],
    [0.83, 0.94, 0.41],
];

/// A small square inside a larger one on one tilted plane, turned against it, each snapped to the
/// camera's grid as the engine hands them to the core: the later on top wherever they overlap, in
/// either order. Fitted alone, the planes of the two snapped squares differ by the snap's noise
/// and cross; on one plane, they tie by draw order.
#[test]
fn a_small_square_on_a_turned_square_ties_with_it_by_draw_order() {
    let view = looking_down_on_grid();
    let grid = view.input_grid;
    let (green, white) = ([0.1, 0.6, 0.2, 1.0], WHITE);
    let mut wrong = Vec::new();
    for centre in CENTRES {
        let large = tilted_square(centre, 2.5, 0.4, [0.0, 0.0]);
        for (half, at) in [(0.5, [0.7, -0.3]), (0.45, [-0.9, 0.6]), (1.0, [0.0, 0.0])] {
            let small = tilted_square(centre, half, 1.1, at);
            let inside = inside_on_screen(&view, &small);
            assert!(
                inside.len() > 10,
                "{centre:?} {half}: {} pixels",
                inside.len()
            );
            for small_on_top in [true, false] {
                let mut items = vec![
                    on_grid(|s| fill(1, s, green), &ring(&large), grid),
                    on_grid(|s| fill(2, s, white), &ring(&small), grid),
                ];
                if !small_on_top {
                    items.reverse();
                }
                let frame = draw(view.clone(), &items);
                let want = premultiply(if small_on_top { white } else { green });
                let off = inside.iter().filter(|(x, y)| px(&frame, *x, *y) != want);
                let off = off.count();
                if off > 0 {
                    wrong.push(format!(
                        "{centre:?} half {half}, small on top {small_on_top}: {off} of {} pixels",
                        inside.len()
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A line drawn on a tilted square's plane, each snapped to the camera's grid as the engine hands
/// them to the core. The line has no plane of its own; it is keyed on the plane of the square it
/// lies on, so they tie by draw order. Drawn after the square, the line is its own pixels over the
/// square's; drawn before, the square hides it.
#[test]
fn a_line_on_a_tilted_plane_ties_with_it_by_draw_order() {
    let view = looking_down_on_grid();
    let grid = view.input_grid;
    let (u, v) = tilted();
    let green = [0.1, 0.6, 0.2, 1.0];
    let mut wrong = Vec::new();
    for centre in CENTRES {
        let corners = tilted_square(centre, 2.5, 0.4, [0.0, 0.0]);
        let inside = inside_on_screen(&view, &corners);
        let square = on_grid(|s| fill(1, s, green), &ring(&corners), grid);
        let square_alone = draw(view.clone(), std::slice::from_ref(&square));
        for (a, b) in [([-1.7, -0.9], [1.9, 0.6]), ([-0.4, 1.6], [0.5, -1.8])] {
            let at = |q: [f64; 2]| -> [f64; 3] {
                std::array::from_fn(|k| centre[k] + q[0] * u[k] + q[1] * v[k])
            };
            let line = on_grid(|s| stroke(2, s, WHITE, 24.0), &lines(&[at(a), at(b)]), grid);
            assert_eq!(line.1.planes, Some(vec![Planarity::Curve.code()]));
            let line_alone = draw(view.clone(), std::slice::from_ref(&line));
            let lit = inside
                .iter()
                .filter(|(x, y)| px(&line_alone, *x, *y)[0] > 128);
            assert!(lit.count() > 10, "{centre:?} {a:?}: the line is drawn");
            let on_top = draw(view.clone(), &[square.clone(), line.clone()]);
            let beneath = draw(view.clone(), &[line.clone(), square.clone()]);
            let (mut off_top, mut off_beneath) = (0, 0);
            for (x, y) in &inside {
                let cover = f64::from(px(&line_alone, *x, *y)[0]) / 255.0;
                let under = px(&square_alone, *x, *y);
                let got = px(&on_top, *x, *y);
                let over = (0..3).any(|k| {
                    let want = 255.0 * cover + f64::from(under[k]) * (1.0 - cover);
                    (f64::from(got[k]) - want).abs() > 1.0
                });
                off_top += usize::from(over);
                off_beneath += usize::from(px(&beneath, *x, *y) != under);
            }
            if off_top + off_beneath > 0 {
                wrong.push(format!(
                    "{centre:?} {a:?}: {off_top} pixels off with the line on top, \
                     {off_beneath} with the square on top, of {}",
                    inside.len()
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn points_record(slot: u32, shape: &Shape, colours: &Stops, radius: f64) -> Record {
    let mut r = Record::new(RecordKind::Points, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
    r.body = Body::Points(Points {
        colors: colours.key(),
        radius,
    });
    r
}

/// The frame point whose subpixel coordinates (flat view, CE's `points_to_subpixel_coords`) are
/// `(sx, sy)`.
fn at_subpixel(view: &View, sx: f64, sy: f64) -> [f64; 3] {
    let (pw, ph) = (W as f64, H as f64);
    [
        (sx - pw / 2.0) / (pw / view.frame_width) + view.frame_center[0],
        (sy - ph / 2.0) / -(ph / view.frame_height) + view.frame_center[1],
        0.0,
    ]
}

/// Draws a points record over black (flat view), its points placed at subpixel coordinates.
fn draw_points(subpixel: &[[f64; 2]], colours: &[[f64; 4]], radius: f64) -> Vec<u8> {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let places: Vec<[f64; 3]> = subpixel
        .iter()
        .map(|p| at_subpixel(&view, p[0], p[1]))
        .collect();
    let table = Stops::new(colours.to_vec());
    let shape = path(places);
    stream(
        view,
        &[(shape.clone(), points_record(1, &shape, &table, radius))],
        vec![table.clone()],
    )
    .draw(W, H, 1)
    .unwrap()
}

/// The pixels CE's block covers: about the truncated pixel, `range(-t//2 + 1, t//2 + 1)` on each
/// axis, on the frame.
fn block(at: [i64; 2], t: i64) -> Vec<(u32, u32)> {
    let lo = (-t).div_euclid(2) + 1;
    let mut out = Vec::new();
    for x in at[0] + lo..at[0] + lo + t {
        for y in at[1] + lo..at[1] + lo + t {
            if (0..W as i64).contains(&x) && (0..H as i64).contains(&y) {
                out.push((x as u32, y as u32));
            }
        }
    }
    out
}

/// Every pixel: the colour inside `pixels`, black elsewhere.
fn assert_exactly(frame: &[u8], pixels: &[(u32, u32)], colour: [u8; 4], what: &str) {
    for y in 0..H {
        for x in 0..W {
            let want = if pixels.contains(&(x, y)) {
                colour
            } else {
                BLACK
            };
            assert_eq!(px(frame, x, y), want, "{what}: pixel {x},{y}");
        }
    }
}

/// CE draws a point as a square block of `int(stroke width)` pixels a side about the pixel its
/// subpixel place truncates to, `trunc(255 c)` in every channel.
#[test]
fn a_point_is_ces_square_block_of_whole_pixels() {
    let colour = [0.7, 0.35, 0.123, 1.0];
    let bytes = [178, 89, 31, 255];
    for t in 1..=4 {
        // A stroke width with a fraction: CE's `int` truncates it.
        let frame = draw_points(&[[37.6, 20.3]], &[colour], t as f64 + 0.7);
        let pixels = block([37, 20], t);
        assert_eq!(pixels.len() as i64, t * t);
        assert_exactly(&frame, &pixels, bytes, &format!("t {t}"));
    }
    // Offsets for an even side reach one further on the high side: t = 4 is -1..=2.
    assert_eq!(block([10, 10], 4)[0], (9, 9));
    assert_eq!(*block([10, 10], 4).last().unwrap(), (12, 12));
    for radius in [0.0, 0.9, -3.0] {
        let frame = draw_points(&[[37.6, 20.3]], &[colour], radius);
        assert_exactly(&frame, &[], bytes, &format!("radius {radius}"));
    }
}

/// `astype(int64)` truncates toward zero: a point at x = -0.6 lands on pixel 0, not -1; a block
/// partly off the frame keeps its pixels on it.
#[test]
fn a_points_pixel_truncates_toward_zero_and_its_block_is_clipped() {
    let colour = [1.0, 1.0, 1.0, 1.0];
    let frame = draw_points(&[[-0.6, 10.2]], &[colour], 3.0);
    assert_exactly(&frame, &block([0, 10], 3), [255; 4], "x = -0.6");
    assert_eq!(
        block([0, 10], 3).len(),
        6,
        "a column of three is off the frame"
    );
    let corner = [W as f64 - 0.7, H as f64 - 0.6];
    let frame = draw_points(&[corner], &[colour], 4.0);
    let pixels = block([W as i64 - 1, H as i64 - 1], 4);
    assert_eq!(pixels.len(), 4);
    assert_exactly(&frame, &pixels, [255; 4], "the corner");
}

/// Where blocks overlap, CE's later write wins, and CE writes every point at its first offset,
/// then every point at its next: so point 0's block, offset (1, 0) from its pixel, wins over point
/// 1's offset (0, 0), though point 1 comes later.
#[test]
fn overlapping_blocks_keep_ces_offset_major_write_order() {
    let (red, green) = ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
    let frame = draw_points(&[[50.5, 30.5], [51.5, 30.5]], &[red, green], 2.0);
    let r = [255, 0, 0, 255];
    let g = [0, 255, 0, 255];
    for (x, y, want) in [
        (50, 30, r),
        (51, 30, r),
        (52, 30, g),
        (50, 31, r),
        (51, 31, r),
        (52, 31, g),
    ] {
        assert_eq!(px(&frame, x, y), want, "pixel {x},{y}");
    }
}

#[test]
fn a_points_record_of_more_block_writes_than_sub_orders_is_refused() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let table = Stops::new(vec![WHITE]);
    let shape = path(vec![[0.0, 0.0, 0.0]]);
    // 65537^2 > 2^32 writes for one point; 65536^2 = 2^32 is the most there are sub-orders for.
    for (radius, refused) in [(65536.0, false), (65537.0, true)] {
        let s = stream(
            view.clone(),
            &[(shape.clone(), points_record(1, &shape, &table, radius))],
            vec![table.clone()],
        );
        match s.draw(W, H, 1) {
            Err(DrawError::Unsupported(u)) => {
                assert!(refused, "radius {radius}: {}", u.reason);
                assert!(u.reason.contains("block pixels"), "{}", u.reason);
            }
            Ok(frame) => {
                assert!(!refused, "radius {radius} drew");
                assert!(
                    frame.chunks(4).all(|p| p == [255; 4]),
                    "the block covers the frame"
                );
            }
            Err(e) => panic!("{e:?}"),
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
    floor.geometry.shape_a = Some(sheet.key());
    floor.paint.fill = Paint::One([0.1, 0.2, 0.7, 1.0]);
    floor.depth.space = Space::World;
    let colours = Stops::new(vec![[1.0, 1.0, 0.0, 1.0]]);
    let Camera::ThreeD(c) = &view.camera else {
        unreachable!()
    };
    let projection = Projection::new(c, view.frame_center);
    for (z, shown) in [(-1.0, false), (1.0, true)] {
        let p = [0.5, 0.3, z];
        let cloud = path(vec![p]);
        let mut dot = points_record(2, &cloud, &colours, 3.0);
        dot.depth.space = Space::World;
        let s = stream(
            view.clone(),
            &[(sheet.clone(), floor.clone()), (cloud.clone(), dot)],
            vec![colours.clone()],
        );
        let frame = s.draw(W, H, 1).unwrap();
        let placed = pre_display(&[p], Some(&projection), Placement::Projected);
        let [sx, sy] = subpixel_coords(
            &placed,
            W,
            H,
            view.frame_width,
            view.frame_height,
            view.frame_center,
        )[0];
        let got = px(&frame, sx as u32, sy as u32);
        let want = if shown {
            [255, 255, 0, 255]
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
        let is_path = matches!(prototype.data(), ShapeData::Path { .. });
        let mut member = Record::new(
            if is_path {
                RecordKind::Path
            } else {
                RecordKind::Mesh
            },
            1,
            101,
        );
        member.geometry.shape_a = Some(prototype.key());
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
            fill: fills.key(),
            stroke: strokes.key(),
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
                r.paint.fill = Paint::One(fills.rows()[i]);
                if is_path {
                    r.paint.stroke = Paint::One(strokes.rows()[i]);
                }
                (prototype.clone(), r)
            })
            .collect();
        let one_by_one = draw(view.clone(), &members);
        assert!(batched == one_by_one, "{:?} prototype", prototype.data());
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
    // A path's blend, reveal and paint blend draw (tests/stream.rs); a points record's reveal is
    // refused below.
    type Change = fn(&mut Record, Key);
    let cases: [(&str, Change); 7] = [
        ("clip", |r, k| r.clip = Some((k, shift(0.0, 0.0, 0.0)))),
        ("gradient", |r, _| {
            r.paint.stroke = Paint::Two([WHITE, [0.0, 0.0, 0.0, 1.0]])
        }),
        ("gradient", |r, _| {
            r.paint.fill = Paint::Table(Stops::new(vec![WHITE; 12]).key())
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
        change(&mut r, other.key());
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
