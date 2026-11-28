//! A stroke's miter joins as the record carries them (XF 1.4's `joins`): kurbo's own stroke, edited
//! only where the carried decision differs from kurbo's: a carried bevel leaves kurbo's own bevel
//! outline, a carried miter puts kurbo's miter in with its tip at the limit's length; agreeing
//! decisions change no bit. Drawn: a join kurbo mitres on tangents its snapped points give, which
//! Cairo bevels, carried as a bevel, is bevelled; joins on each side of the limit carried as kurbo
//! decides them draw the same bytes as without the field.

mod common;

use std::collections::HashMap;

use arachne_core::join_mask::{self, Decided, Edits};
use arachne_core::project::FrameMap;
use arachne_core::stream::Stream;
use arachne_core::xf::*;
use common::flat_view;
use kurbo::{BezPath, Join, PathEl, Point, Stroke, Vec2};

const W: u32 = 160;
const H: u32 = 90;
const LIMIT: f64 = 10.0;
const TOL: f64 = 0.01;

fn style(width: f64, join: Join) -> Stroke {
    Stroke::new(width).with_join(join).with_miter_limit(LIMIT)
}

fn stroked(path: &BezPath, s: &Stroke) -> BezPath {
    kurbo::stroke(path.iter(), s, &kurbo::StrokeOpts::default(), TOL)
}

/// Two cubics meeting at `p0`, arriving along `ab` and leaving along `cd` (unit-ish directions),
/// each `len` long with arms `arm` long at the join; closed into a triangle when `closed`.
fn corner(p0: Point, ab: Vec2, cd: Vec2, len: f64, arm: f64, closed: bool) -> BezPath {
    let start = p0 - ab * len;
    let end = p0 + cd * len;
    let mut path = BezPath::new();
    path.move_to(start);
    path.curve_to(start + ab * (len / 3.0), p0 - ab * arm, p0);
    path.curve_to(p0 + cd * arm, end - cd * (len / 3.0), end);
    if closed {
        let back = start - end;
        path.curve_to(end + back / 3.0, end + back * (2.0 / 3.0), start);
        path.close_path();
    }
    path
}

/// The cubics of `path`, each by its four points, in order.
fn cubics(path: &BezPath) -> Vec<[Point; 4]> {
    let mut out = Vec::new();
    let mut p0 = Point::ZERO;
    for el in path.elements() {
        match *el {
            PathEl::MoveTo(p) => p0 = p,
            PathEl::CurveTo(a, b, c) => {
                out.push([p0, a, b, c]);
                p0 = c;
            }
            _ => {}
        }
    }
    out
}

/// The carried decision `mitres` for the join from the first cubic of `path` into the second.
fn deciding(path: &BezPath, mitres: bool) -> Decided {
    let c = cubics(path);
    HashMap::from([(join_mask::key(c[0], c[1], false), vec![Some(mitres)])])
}

fn direction(degrees: f64) -> Vec2 {
    let r = degrees.to_radians();
    Vec2::new(libm::cos(r), libm::sin(r))
}

/// Every join of random corners kurbo mitres, carried as a bevel: the outline is kurbo's own bevel
/// stroke, element for element and to the bit; so kurbo's miter added nothing else anywhere.
#[test]
fn a_carried_bevel_where_kurbo_mitres_leaves_kurbo_s_own_bevel() {
    let mut seen = 0;
    for i in 0..400 {
        let f = i as f64;
        let turn = 20.0 + (f * 37.0) % 148.0;
        let heading = (f * 53.0) % 360.0;
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let (ab, cd) = (direction(heading), direction(heading + sign * turn));
        let arm = [3.0, 0.4, 0.01][i % 3];
        let path = corner(Point::new(f * 0.37, -f * 0.21), ab, cd, 9.0, arm, false);
        let width = [0.3, 1.2, 2.4][i % 3];
        let mitred = stroked(&path, &style(width, Join::Miter));
        let bevelled = stroked(&path, &style(width, Join::Bevel));
        if mitred == bevelled {
            continue;
        }
        let (got, edits) = join_mask::apply(
            mitred,
            path.elements(),
            &style(width, Join::Miter),
            TOL,
            &deciding(&path, false),
        );
        assert_eq!(
            edits,
            Edits {
                dropped: 1,
                ..Edits::default()
            },
            "corner {i}"
        );
        assert_eq!(got.elements(), bevelled.elements(), "corner {i}");
        seen += 1;
    }
    assert!(seen > 300, "{seen}");
}

/// A join past the limit, which kurbo bevels, carried as a miter: kurbo's bevel outline with its
/// miter's two vertices put in, the tip on the bisector of the outer offsets at the limit's length
/// from the join, the inner one at the join itself.
#[test]
fn a_carried_miter_where_kurbo_bevels_puts_the_tip_at_the_limit_on_the_bisector() {
    let p0 = Point::new(4.0, 2.0);
    for (turn, sign) in [(172.0, 1.0), (172.0, -1.0), (179.0, 1.0)] {
        let (ab, cd) = (direction(10.0), direction(10.0 + sign * turn));
        let path = corner(p0, ab, cd, 9.0, 2.0, false);
        let s = style(1.0, Join::Miter);
        let bevelled = stroked(&path, &style(1.0, Join::Bevel));
        let drawn = stroked(&path, &s);
        assert_eq!(drawn, bevelled, "kurbo bevels past the limit");
        let (got, edits) =
            join_mask::apply(drawn, path.elements(), &s, TOL, &deciding(&path, true));
        assert_eq!(
            edits,
            Edits {
                inserted: 1,
                ..Edits::default()
            }
        );
        let added: Vec<Point> = got
            .elements()
            .iter()
            .filter(|e| !bevelled.elements().contains(e))
            .map(|e| e.end_point().unwrap())
            .collect();
        assert_eq!(added.len(), 2, "{turn} {sign}");
        assert!(added.contains(&p0));
        let tip = added.into_iter().find(|p| *p != p0).unwrap();
        let reach = tip - p0;
        assert!(
            (reach.hypot() - LIMIT * 0.5).abs() < 1e-12,
            "{turn}: {}",
            reach.hypot()
        );
        // The outer bisector: away from both arms, halfway between them.
        let outward = -(ab.normalize() - cd.normalize()) * -1.0;
        let bisector = (ab.normalize() - cd.normalize()).normalize();
        assert!(reach.normalize().cross(bisector).abs() < 1e-9, "{turn}");
        assert!(reach.dot(outward) > 0.0, "{turn}: the outer side");
        // Taken out again, they leave kurbo's bevel.
        let back: Vec<PathEl> = got
            .elements()
            .iter()
            .copied()
            .filter(|e| bevelled.elements().contains(e))
            .collect();
        assert_eq!(back, bevelled.elements());
    }
}

/// Carried decisions that agree with kurbo's, at a join within the limit (mitred) and one past it
/// (bevelled), on long arms and on arms under 0.017 units, and at a closing join: no bit moves.
#[test]
fn carried_decisions_kurbo_agrees_with_change_nothing() {
    for (turn, arm, closed) in [
        (120.0, 3.0, false),
        (168.0, 3.0, false),
        (169.0, 3.0, false),
        (150.0, 0.012, false),
        (175.0, 0.012, false),
        (60.0, 3.0, true),
    ] {
        let (ab, cd) = (direction(0.0), direction(turn));
        let path = corner(Point::new(1.0, 1.0), ab, cd, 9.0, arm, closed);
        let s = style(1.0, Join::Miter);
        let drawn = stroked(&path, &s);
        let kurbo_mitres = drawn != stroked(&path, &style(1.0, Join::Bevel));
        let mut decided = deciding(&path, kurbo_mitres);
        if closed {
            let c = cubics(&path);
            decided.insert(join_mask::key(c[2], c[0], true), vec![Some(true)]);
        }
        let (got, edits) = join_mask::apply(drawn.clone(), path.elements(), &s, TOL, &decided);
        assert_eq!(edits, Edits::default(), "{turn} {arm}");
        assert_eq!(got.elements(), drawn.elements(), "{turn} {arm}");
    }
}

/// A path holding one corner twice, as two subpaths of the same points: each copy takes its own
/// carried decision (one per occurrence, in order), edited in its own contour; bevelled both
/// times, the outline is kurbo's own bevel of both; bevelled then mitred, kurbo's bevel of the
/// first then its miter of the second.
#[test]
fn each_copy_of_a_repeated_subpath_takes_its_own_decision_in_its_own_contour() {
    let one = corner(
        Point::new(2.0, 1.0),
        direction(0.0),
        direction(150.0),
        9.0,
        3.0,
        false,
    );
    let mut twice = one.clone();
    for el in one.elements() {
        twice.push(*el);
    }
    let s = style(1.0, Join::Miter);
    let c = cubics(&one);
    let bevel = stroked(&one, &style(1.0, Join::Bevel));
    let miter = stroked(&one, &s);
    for (decisions, dropped, expected) in [
        (vec![Some(false), Some(false)], 2, [&bevel, &bevel]),
        (vec![Some(false), Some(true)], 1, [&bevel, &miter]),
    ] {
        let decided = HashMap::from([(join_mask::key(c[0], c[1], false), decisions)]);
        let (got, edits) =
            join_mask::apply(stroked(&twice, &s), twice.elements(), &s, TOL, &decided);
        assert_eq!(
            edits,
            Edits {
                dropped,
                ..Edits::default()
            }
        );
        let whole: Vec<PathEl> = expected
            .iter()
            .flat_map(|p| p.elements().to_vec())
            .collect();
        assert_eq!(got.elements(), whole);
    }
}

/// A join Cairo does not make (its byte without `JOINED`) keeps its place among the occurrences of
/// its points: decided nothing, it is left as kurbo strokes it, and the next occurrence takes its
/// own decision. Two copies of one corner, Cairo joining only the second and bevelling it: kurbo's
/// miter at the first, its bevel at the second.
#[test]
fn a_join_cairo_does_not_make_keeps_its_place_among_its_points_occurrences() {
    let one = corner(
        Point::new(2.0, 1.0),
        direction(0.0),
        direction(150.0),
        9.0,
        3.0,
        false,
    );
    let mut twice = one.clone();
    for el in one.elements() {
        twice.push(*el);
    }
    let shown: Vec<[f64; 3]> = cubics(&twice)
        .iter()
        .flat_map(|c| c.map(|p| [p.x, p.y, 0.0]))
        .collect();
    let decided = arachne_core::path::decided_joins(&shown, &[0, 0, 0, 1]);
    let c = cubics(&one);
    assert_eq!(
        decided,
        HashMap::from([(join_mask::key(c[0], c[1], false), vec![None, Some(false)])])
    );
    let s = style(1.0, Join::Miter);
    let (got, edits) = join_mask::apply(stroked(&twice, &s), twice.elements(), &s, TOL, &decided);
    assert_eq!(
        edits,
        Edits {
            dropped: 1,
            ..Edits::default()
        }
    );
    let bevel = stroked(&one, &style(1.0, Join::Bevel));
    let miter = stroked(&one, &s);
    let whole: Vec<PathEl> = [&miter, &bevel]
        .iter()
        .flat_map(|p| p.elements().to_vec())
        .collect();
    assert_eq!(got.elements(), whole);
}

// --- drawn through a Stream ---

fn map(view: &View) -> FrameMap {
    FrameMap::cairo(W, H, view.frame_width, view.frame_height, view.frame_center)
}

fn frame_of(view: &View, p: [f64; 2]) -> [f64; 3] {
    let m = map(view);
    [(p[0] - m.x0) / m.xx, (p[1] - m.y0) / m.yy, 0.0]
}

fn draw(view: &View, points: Vec<[f64; 3]>, width: f64, joins: Option<Vec<u8>>) -> Vec<u8> {
    draw_jointed(view, points, width, joins, Joint::Auto)
}

fn draw_jointed(
    view: &View,
    points: Vec<[f64; 3]>,
    width: f64,
    joins: Option<Vec<u8>>,
    joint: Joint,
) -> Vec<u8> {
    let shape = Shape::new(ShapeData::Path { points });
    let mut r = Record::new(RecordKind::Path, 1, 101);
    r.geometry.shape_a = Some(shape.key());
    r.paint.stroke = Paint::One([1.0, 1.0, 1.0, 1.0]);
    r.paint.stroke_width = width;
    r.paint.joint = joint;
    r.joins = joins;
    let d = arachne_core::probes::first_delta(view.clone(), &[(shape, r)], &[1]);
    let mut s = Stream::new();
    s.apply(&d).expect("the delta applies");
    s.draw(W, H, 1).unwrap()
}

/// Two cubics in pixels, as CE's points (frame units): arriving at `j` from the left along `a`,
/// leaving back along `b` (arms in pixels), each 30 px long.
fn reversal(view: &View, j: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Vec<[f64; 3]> {
    let start = [j[0] - 30.0, j[1]];
    let end = [j[0] - 30.0, j[1] + 3.0];
    [
        start,
        [start[0] + 10.0, start[1]],
        [j[0] - a[0], j[1] - a[1]],
        j,
        j,
        [j[0] + b[0], j[1] + b[1]],
        [end[0] + 10.0, end[1]],
        end,
    ]
    .iter()
    .map(|p| frame_of(view, *p))
    .collect()
}

fn lit_right_of(frame: &[u8], x0: u32) -> usize {
    (0..H)
        .flat_map(|y| (x0..W).map(move |x| (x, y)))
        .filter(|&(x, y)| frame[((y * W + x) * 4) as usize] != 0)
        .count()
}

/// A reversal on arms of a few 1/256 px whose tangents, as handed, turn 163 degrees: kurbo mitres,
/// drawing a spike past the join; carried as Cairo's bevel (its rounded slopes turn past the
/// limit), nothing is drawn beyond the join's half width.
#[test]
fn a_reversal_cairo_bevels_carried_as_a_bevel_draws_no_spike() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let j = [100.3, 40.4];
    // Arriving along (6, -1)/256 px, leaving along (-2, 1)/256: 162.9 degrees.
    let points = reversal(
        &view,
        j,
        [6.0 / 256.0, -1.0 / 256.0],
        [-2.0 / 256.0, 1.0 / 256.0],
    );
    let width = 2.4 / (map(&view).xx * view.cairo_line_width_multiple);
    let x0 = (j[0] + 2.0) as u32;
    let mitred = draw(&view, points.clone(), width, None);
    assert!(lit_right_of(&mitred, x0) > 0, "kurbo mitres it");
    let bevelled = draw(&view, points, width, Some(vec![0, 1]));
    assert_eq!(lit_right_of(&bevelled, x0), 0);
}

/// Joins just within the limit (168 degrees, mitred) and just past it (169, bevelled), on long arms
/// the snap cannot turn: carried as kurbo decides them, the frame's bytes are those without the
/// field.
#[test]
fn joins_on_each_side_of_the_limit_carried_as_decided_draw_the_same_bytes() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    for (turn, mitres) in [(168.0f64, true), (169.0, false)] {
        let r = turn.to_radians();
        let j = [100.0, 45.0];
        let b = [10.0 * libm::cos(r), -10.0 * libm::sin(r)];
        let points = reversal(&view, j, [10.0, 0.0], b);
        let width = 1.0 / (map(&view).xx * view.cairo_line_width_multiple);
        let plain = draw(&view, points.clone(), width, None);
        let byte = 1 | if mitres { 2 } else { 0 };
        let carried = draw(&view, points, width, Some(vec![0, byte]));
        assert!(plain == carried, "{turn} degrees");
    }
}

/// A closed triangle whose points begin with a cubic of one point (a curve the stroker skips),
/// its closing corner sharp enough that kurbo mitres it: carried as a bevel in the first cubic's
/// byte, the closing join is bevelled; without the field, its miter is drawn.
#[test]
fn a_closing_join_after_a_skipped_cubic_takes_its_carried_decision() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    // In pixels: the closing corner at a, its sides 14 degrees apart, opening to the left.
    let (a, b, c) = ([110.0, 45.0], [40.0, 36.4], [40.0, 53.6]);
    let line = |p: [f64; 2], q: [f64; 2]| {
        let at = |t: f64| [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
        [p, at(1.0 / 3.0), at(2.0 / 3.0), q]
    };
    let mut points = vec![a, a, a, a];
    points.extend(line(a, b));
    points.extend(line(b, c));
    points.extend(line(c, a));
    let points: Vec<[f64; 3]> = points.iter().map(|p| frame_of(&view, *p)).collect();
    let width = 1.0 / (map(&view).xx * view.cairo_line_width_multiple);
    let x0 = (a[0] + 1.0) as u32;
    let mitred = draw(&view, points.clone(), width, None);
    assert!(
        lit_right_of(&mitred, x0) > 0,
        "kurbo mitres the closing join"
    );
    let bevelled = draw(&view, points, width, Some(vec![1, 3, 3, 3]));
    assert_eq!(lit_right_of(&bevelled, x0), 0);
}

/// Two straight cubics in pixels, as CE's points: arriving at `j` along +x and leaving it turned
/// by `turn` degrees, each 30 px long.
fn bent(view: &View, j: [f64; 2], turn: f64) -> Vec<[f64; 3]> {
    let r = turn.to_radians();
    let start = [j[0] - 30.0, j[1]];
    let end = [j[0] + 30.0 * libm::cos(r), j[1] - 30.0 * libm::sin(r)];
    let line = |p: [f64; 2], q: [f64; 2]| {
        let at = |t: f64| [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
        [p, at(1.0 / 3.0), at(2.0 / 3.0), q]
    };
    let mut points = line(start, j).to_vec();
    points.extend(line(j, end));
    points.iter().map(|p| frame_of(view, *p)).collect()
}

/// A stroke under 0.8 px is stroked at a quarter of its half width, not at the flattening
/// tolerance, and kurbo leaves a turn unjoined by the tolerance it strokes at (a turn whose sine is
/// under twice the tolerance over the width): the joins are read at that same tolerance. A turn
/// kurbo joins at it, carried as a bevel, draws the bevel-jointed stroke's bytes (read at the
/// flattening tolerance, a turn under 90 degrees on a 0.2 px stroke is taken for one kurbo did not
/// join, and its miter stays); and joins on each side of the limit, carried as decided, draw the
/// bytes without the field.
#[test]
fn a_thin_stroke_s_joins_are_read_at_the_tolerance_it_is_stroked_at() {
    let view = flat_view(W, H, [0.0, 0.0, 0.0, 1.0]);
    let per_px = map(&view).xx * view.cairo_line_width_multiple;
    let mut shows = 0;
    for w_px in [0.2, 0.3, 0.5, 0.7] {
        let width = w_px / per_px;
        for turn in [20.0, 40.0, 60.0, 85.0] {
            for j in [[100.3, 45.2], [80.55, 40.7], [60.81, 50.35]] {
                let points = bent(&view, j, turn);
                let bevel = draw_jointed(&view, points.clone(), width, None, Joint::Bevel);
                let carried = draw(&view, points.clone(), width, Some(vec![0, 1]));
                assert!(
                    carried == bevel,
                    "{w_px} px, {turn} degrees at {j:?}: a carried bevel is not the bevel"
                );
                shows += (draw(&view, points, width, None) != bevel) as u32;
            }
        }
        let j = [100.3, 45.2];
        for (turn, mitres) in [(168.0f64, true), (169.0, false)] {
            let points = bent(&view, j, turn);
            let plain = draw(&view, points.clone(), width, None);
            let byte = 1 | if mitres { 2 } else { 0 };
            let carried = draw(&view, points, width, Some(vec![0, byte]));
            assert!(plain == carried, "{w_px} px, {turn} degrees");
        }
    }
    assert!(shows > 0, "no miter of these differs from its bevel");
}
