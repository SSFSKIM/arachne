//! The dash rules of contract §8 through the oracle's decisions, at the cases the text settles:
//! `dash_start_least_length@1`'s solid rule (a duty of 1, and a gap clearly under or clearly over
//! Cairo's least length, `1/512`) and `round_cap_dot@1`'s dash start (an offset clearly inside an
//! on entry or an off one, with a positive phase). The boundaries the text leaves open (a gap of
//! exactly `1/512`, an offset exactly on an entry, a negative phase) are not asserted here.

mod common;

use arachne_core::xf::{Body, Cap, Dash, FaceGroup, Paint, Record, RecordKind, Space, Stops, View};
use common::*;

/// A horizontal open line across a 16 x 16 frame, 1 px wide, butt-capped, dashed by `dash`.
fn line(dash: Option<Dash>) -> Vec<String> {
    let v = view(16, 16);
    let s = path_shape(polyline(&[frame(&v, [2.0, 8.5]), frame(&v, [14.0, 8.5])]));
    let mut r = path(0, s.key(), false, Some(1.0));
    r.dash = dash;
    cover_lines(&decide(&[delta(1, Some(v), vec![s], vec![r])]))
}

fn line_area(dash: Option<Dash>) -> f64 {
    let v = view(16, 16);
    let s = path_shape(polyline(&[frame(&v, [2.0, 8.5]), frame(&v, [14.0, 8.5])]));
    let mut r = path(0, s.key(), false, Some(1.0));
    r.dash = dash;
    let d = decide(&[delta(1, Some(v), vec![s], vec![r])]);
    d.covers.iter().map(|c| c.pixel_area).sum()
}

fn dash(period: f64, duty: f64, phase: f64) -> Option<Dash> {
    Some(Dash {
        period,
        duty,
        phase,
    })
}

#[test]
fn a_duty_of_one_strokes_solid() {
    assert_eq!(line(dash(2.0, 1.0, 0.0)), line(None));
    assert_eq!(line(dash(2.0, 1.0, 0.3)), line(None));
}

#[test]
fn a_gap_clearly_under_the_least_length_strokes_solid() {
    // Gap 2 · 0.00025 = 0.0005, a quarter of 1/512.
    assert_eq!(line(dash(2.0, 0.99975, 0.0)), line(None));
    assert_eq!(line(dash(2.0, 0.99975, 0.4)), line(None));
}

#[test]
fn a_gap_clearly_over_the_least_length_is_dashed() {
    // Gap 2 · 0.01 = 0.02, ten times 1/512: six gaps along the 12 px line take 0.12 px² out.
    let solid = line_area(None);
    let dashed = line_area(dash(2.0, 0.99, 0.0));
    assert!(
        solid - dashed > 0.1 && solid - dashed < 0.14,
        "{solid} - {dashed}"
    );
    // A plain half-duty dash: half the line, to the accuracy kurbo places dash ends at.
    let half = line_area(dash(2.0, 0.5, 0.0));
    assert!(
        (half - solid / 2.0).abs() < 1e-5,
        "{half} against {solid} / 2"
    );
}

/// One face of a face group, all four shown points at the frame's centre, stroked white `width`
/// px wide under `cap`, dashed by `dash`: the area its stroke covers, or `None` where it makes no
/// stroke fragment.
fn spot(cap: Cap, width: f64, dash: Option<Dash>) -> Option<f64> {
    let v: View = view(16, 16);
    let c = frame(&v, [8.0, 8.0]);
    let s = path_shape(vec![c; 4]);
    // One white row serves as the face's fill and stroke table.
    let white = Stops::new(vec![[1.0, 1.0, 1.0, 1.0]]);
    let mut r = Record::new(RecordKind::FaceGroup, 0, 100);
    r.geometry.shape_a = Some(s.key());
    r.body = Body::FaceGroup(FaceGroup {
        uv: vec![[0, 0]],
        grid: [1, 1],
        fill: white.key(),
        stroke: Some(white.key()),
        face_ends: Vec::new(),
        face_widths: Vec::new(),
    });
    r.paint.stroke = Paint::One([1.0, 1.0, 1.0, 1.0]);
    r.paint.stroke_width = width;
    r.paint.cap = cap;
    r.dash = dash;
    r.depth.space = Space::Screen;
    let mut d = delta(1, Some(v), vec![s], vec![r]);
    d.upload_paints = vec![white];
    let out = decide(&[d]);
    let rows = rows_of(&out, 2);
    (!rows.is_empty()).then(|| rows.iter().map(|c| c.pixel_area).sum())
}

/// A dot of radius `WIDE / 2` px: its area at most the circle's, and short of it by no more than a
/// band of 0.2 px along its rim (`Circle::to_path` and the flattening, each at 0.1 px).
fn is_dot(area: Option<f64>) -> bool {
    let r = WIDE / 2.0;
    let circle = std::f64::consts::PI * r * r;
    area.is_some_and(|a| a <= circle && a >= circle - 2.0 * std::f64::consts::PI * r * 0.2)
}

/// The dot's stroke width, in px.
const WIDE: f64 = 8.0;

#[test]
fn a_face_at_one_spot_under_a_round_cap_strokes_a_dot() {
    assert!(
        is_dot(spot(Cap::Round, WIDE, None)),
        "{:?}",
        spot(Cap::Round, WIDE, None)
    );
    assert_eq!(spot(Cap::Butt, WIDE, None), None);
    assert_eq!(spot(Cap::Square, WIDE, None), None);
}

#[test]
fn a_dashed_dot_is_drawn_only_where_the_dash_starts_on() {
    // Entries [1, 1]: the start `phase · period` taken modulo 2.
    for (phase, on) in [(0.25, true), (0.75, false), (1.25, true), (1.6, false)] {
        let a = spot(Cap::Round, WIDE, dash(2.0, 0.5, phase));
        if on {
            assert!(is_dot(a), "phase {phase}: {a:?}");
        } else {
            assert_eq!(a, None, "phase {phase}");
        }
    }
    // Entries [1.5, 0.5]: 1.2 is inside the on entry, 1.8 inside the off one.
    assert!(is_dot(spot(Cap::Round, WIDE, dash(2.0, 0.75, 0.6))));
    assert_eq!(spot(Cap::Round, WIDE, dash(2.0, 0.75, 0.9)), None);
}
