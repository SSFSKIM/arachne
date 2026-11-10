//! The dash rules of contract §8: `dash_start_least_length@1`'s entries and solid rule (a duty of
//! 1, a gap under Cairo's least length `1/512`, the gap taken as the total less the on length, a
//! gap of exactly `1/512` dashed) and `round_cap_dot@1`'s dash start (the offset taken by `fmod`
//! modulo the entries' total and raised by it when negative, then stepped past every entry it
//! reaches, an offset equal to an entry included). Each is asserted on the two rules' functions at
//! its boundary, and through the oracle's decisions.

mod common;

use arachne_core::xf::{Body, Cap, Dash, FaceGroup, Paint, Record, RecordKind, Space, Stops, View};
use arachne_oracle::flatten::{dash_entries, dash_starts_on};
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

/// Cairo's least length, `1/512`.
const LEAST: f64 = 1.0 / 512.0;

fn entries(period: f64, duty: f64) -> Option<[u64; 2]> {
    dash_entries(dash(period, duty, 0.0)).map(|e| e.map(f64::to_bits))
}

fn bits(e: [f64; 2]) -> Option<[u64; 2]> {
    Some(e.map(f64::to_bits))
}

#[test]
fn the_entries_are_period_times_duty_and_period_times_one_less_the_duty() {
    // 0.1 · (1 − 0.7) = 0.030000000000000006; the period less the on entry would be
    // 0.030000000000000013.
    assert_eq!(
        entries(0.1, 0.7),
        bits([0.06999999999999999, 0.030000000000000006])
    );
    assert_eq!(entries(2.0, 0.75), bits([1.5, 0.5]));
}

#[test]
fn no_dash_and_a_duty_of_one_have_no_entries() {
    assert_eq!(dash_entries(None), None);
    assert_eq!(entries(2.0, 1.0), None);
    assert_eq!(entries(1e-3, 1.0), None);
}

#[test]
fn a_gap_of_exactly_the_least_length_is_dashed() {
    // a = 511/512, b = 1/512, a + b = 1, gap 1 − 511/512 = 1/512 exactly.
    assert_eq!(entries(1.0, 1.0 - LEAST), bits([1.0 - LEAST, LEAST]));
    // The same at period 4: entries 4 · 511/512 and 4 · 1/512, gap 4/512.
    assert_eq!(
        entries(4.0, 1.0 - LEAST),
        bits([4.0 - 4.0 * LEAST, 4.0 * LEAST])
    );
    // Period 1/512 at duty 1/2: gap 1/1024, under.
    assert_eq!(entries(LEAST, 0.5), None);
    // Period 2/512 at duty 1/2: gap exactly 1/512.
    assert_eq!(entries(2.0 * LEAST, 0.5), bits([LEAST, LEAST]));
}

#[test]
fn a_gap_just_under_the_least_length_is_solid() {
    // The duty one unit of roundoff above 511/512: the gap is 1/512 − 2^-53.
    let duty = 1.0 - LEAST + f64::EPSILON / 2.0;
    assert!(duty > 1.0 - LEAST);
    assert_eq!(entries(1.0, duty), None);
    // Period 2/512 less one unit in its last place, at duty 1/2: the gap is just under 1/512.
    let period = f64::from_bits((2.0 * LEAST).to_bits() - 1);
    assert_eq!(entries(period, 0.5), None);
}

#[test]
fn the_gap_is_the_total_less_the_on_length_not_the_off_entry() {
    // Period 3 · 2^42 at duty 1 − 2^-53: the off entry b = 0.00146484375 is under 1/512, but
    // a + b rounds up to the period and (a + b) − a is exactly 1/512, so the dash is dashed, and
    // its entries stay a and b.
    let period = 3.0 * (1u64 << 42) as f64;
    let duty = 1.0 - f64::EPSILON / 2.0;
    let e = dash_entries(dash(period, duty, 0.0)).expect("a gap of 1/512 is dashed");
    assert_eq!(bits(e), bits([13194139533311.998, 0.00146484375]));
    assert!(e[1] < LEAST);
    assert_eq!((e[0] + e[1]) - e[0], LEAST);
}

/// Whether the dash `period, duty, phase` starts its subpath on.
fn starts_on(period: f64, duty: f64, phase: f64) -> bool {
    let d = Dash {
        period,
        duty,
        phase,
    };
    dash_starts_on(&d, dash_entries(Some(d)).expect("a dashed dash"))
}

#[test]
fn an_offset_inside_an_entry_keeps_its_side() {
    // Entries [1, 1] and [1.5, 0.5]: the offset is phase · period.
    assert!(starts_on(2.0, 0.5, 0.0));
    assert!(starts_on(2.0, 0.5, 0.25));
    assert!(!starts_on(2.0, 0.5, 0.75));
    assert!(starts_on(2.0, 0.75, 0.6));
    assert!(!starts_on(2.0, 0.75, 0.9));
    // Past a whole period: 3.2 modulo 2 is 1.2, in the off entry.
    assert!(!starts_on(2.0, 0.5, 1.6));
    assert!(starts_on(2.0, 0.5, 1.25));
}

#[test]
fn an_offset_equal_to_an_entry_steps_past_it() {
    // o = a: off.
    assert!(!starts_on(2.0, 0.5, 0.5));
    assert!(!starts_on(2.0, 0.75, 0.75));
    assert!(!starts_on(2.0, 0.25, 0.25));
    // o = a + b, the total: fmod makes it 0, on.
    assert!(starts_on(2.0, 0.5, 1.0));
    assert!(starts_on(2.0, 0.75, 2.0));
    // Just short of a: still on.
    assert!(starts_on(2.0, 0.5, f64::from_bits(0.5f64.to_bits() - 1)));
}

#[test]
fn a_negative_phase_is_taken_by_fmod_then_raised_by_the_total() {
    // Entries [1.5, 0.5], total 2.
    // −0.2: fmod −0.2, plus 2 is 1.8, in the off entry.
    assert!(!starts_on(2.0, 0.75, -0.1));
    // −1: plus 2 is 1, in the on entry.
    assert!(starts_on(2.0, 0.75, -0.5));
    // −1.2: plus 2 is 0.8, on.
    assert!(starts_on(2.0, 0.75, -0.6));
    // −0.5: plus 2 is 1.5, exactly the on entry, stepped past: off.
    assert!(!starts_on(2.0, 0.75, -0.25));
    // −2.2: fmod is −0.20000000000000018, plus 2 is 1.7999999999999998, off (adding the total
    // without fmod would leave −0.2, read as 0: on).
    assert!(!starts_on(2.0, 0.75, -1.1));
    // −4.6: fmod −0.6000000000000005, plus 2 is 1.4000000000000004, on.
    assert!(starts_on(2.0, 0.75, -2.3));
}

#[test]
fn a_negative_offset_of_whole_periods_is_zero_and_starts_on() {
    // fmod(−2, 2) is −0, not below 0; at most 0, so 0: on.
    assert!(starts_on(2.0, 0.5, -1.0));
    assert!(starts_on(2.0, 0.75, -2.0));
}

#[test]
fn a_negative_offset_that_rounds_up_to_the_total_steps_past_both_entries() {
    // −1e-20 plus 2 is 2 exactly: past a (1) and b (1), back to a with 0 left: on.
    assert!(starts_on(2.0, 0.5, -5e-21));
    // Entries [1.5, 0.5]: the same, on.
    assert!(starts_on(2.0, 0.75, -5e-21));
}

#[test]
fn the_offset_is_taken_modulo_the_entries_total_not_the_period() {
    // Here a + b is 7.346672902020838, one unit above the period 7.3466729020208374: at phase 1
    // the offset is the period, short of the total, so it steps past a and stops in b: off.
    let (period, duty) = (7.3466729020208374, 0.8984617768996097);
    let e = dash_entries(dash(period, duty, 0.0)).expect("dashed");
    assert!(e[0] + e[1] > period);
    assert!(!starts_on(period, duty, 1.0));
}

#[test]
fn an_on_entry_of_zero_starts_on_only_at_offset_zero() {
    // Period 1/4 at duty 2^-1074: a rounds to 0, b is 1/4, the gap 1/4 is dashed.
    let duty = f64::from_bits(1);
    assert_eq!(entries(0.25, duty), bits([0.0, 0.25]));
    // o = 0 is not above 0: the start stays on, at the zero-length on entry.
    assert!(starts_on(0.25, duty, 0.0));
    // o = 0.1 steps past the empty on entry into the off one.
    assert!(!starts_on(0.25, duty, 0.4));
}

#[test]
fn a_gap_of_exactly_the_least_length_strokes_dashed_and_one_just_under_strokes_solid() {
    // Period 1, twelve gaps of 1/512 along the 12 px line: 12/512 ≈ 0.0234 px² out.
    let solid = line_area(None);
    let dashed = line_area(dash(1.0, 1.0 - LEAST, 0.0));
    assert!(
        solid - dashed > 0.02 && solid - dashed < 0.027,
        "{solid} - {dashed}"
    );
    assert_eq!(
        line(dash(1.0, 1.0 - LEAST + f64::EPSILON / 2.0, 0.0)),
        line(None)
    );
}

#[test]
fn a_dashed_dot_at_an_entry_boundary_or_a_negative_phase_follows_cairos_start() {
    // Offset 1 = a of [1, 1]: off, no dot.
    assert_eq!(spot(Cap::Round, WIDE, dash(2.0, 0.5, 0.5)), None);
    // Offset 2, the total: 0, a dot.
    assert!(is_dot(spot(Cap::Round, WIDE, dash(2.0, 0.5, 1.0))));
    // Entries [1.5, 0.5]: −0.2 is 1.8, off; −1 is 1, on; −2 is −0, on.
    assert_eq!(spot(Cap::Round, WIDE, dash(2.0, 0.75, -0.1)), None);
    assert!(is_dot(spot(Cap::Round, WIDE, dash(2.0, 0.75, -0.5))));
    assert!(is_dot(spot(Cap::Round, WIDE, dash(2.0, 0.75, -1.0))));
}
