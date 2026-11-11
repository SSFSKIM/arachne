//! The edges a pair admits at a sample where the contract fixes them in closed form (§5, §6, §7),
//! each in a scene of view C's frame (64 x 64 px over 16 x 16 units, 4 px a unit, centre at pixel
//! 32) under the identity rotation at the focal distance each case names.

#[path = "common/scenes.rs"]
mod scenes;

use arachne_core::xf::{Paint, View};
use arachne_oracle::cover::sample_point;
use arachne_oracle::decide::decide_at;
use arachne_oracle::out::{Decisions, PairRow};
use scenes::*;

fn view(focal_distance: f64) -> View {
    View {
        camera: camera(focal_distance),
        ..view_c()
    }
}

fn decide(b: &Build, v: View) -> Decisions {
    let d = decide_at(&[b.delta(1, v)], 1, None).unwrap();
    assert_eq!(d.header.status, 0, "{:?}", d.refused);
    d
}

fn id_of(d: &Decisions, slot: u32, layer: u8) -> u32 {
    d.fragments
        .iter()
        .find(|f| f.slot == slot && f.layer == layer)
        .unwrap_or_else(|| panic!("no fragment of slot {slot} layer {layer}"))
        .id
}

/// The pair rows of fragments `a` and `b` (in that order) over the frame.
fn pairs_of(d: &Decisions, a: u32, b: u32) -> Vec<&PairRow> {
    let rows: Vec<_> = d.pairs.iter().filter(|r| (r.a, r.b) == (a, b)).collect();
    assert!(!rows.is_empty(), "no pair of {a} and {b}");
    rows
}

#[test]
fn an_axis_identity_whose_row_does_not_form_is_not_kept() {
    // §5: a path with no plane of its own on axis planes alone keeps only the axis identities
    // whose rows form. Two lines in y = 0 at focal distance 10, `B` on z = 1 + x / 1000 painted
    // first, `A` on z = 0 after: y = 0 holds the eye, so neither keeps it, and each reads its own
    // key. `B` is nearer (about −10/9 against −1), by far more than any bound.
    let mut b = Build::new(true);
    b.stroke(
        open(&[[-3.05, 0.0, 1.0 - 0.00305], [3.05, 0.0, 1.0 + 0.00305]]),
        OPAQUE_GREEN,
        0.5,
    );
    b.stroke(
        open(&[[-3.05, 0.0, 0.0], [3.05, 0.0, 0.0]]),
        OPAQUE_RED,
        0.5,
    );
    let d = decide(&b, view(10.0));
    let (lb, la) = (id_of(&d, 0, 2), id_of(&d, 1, 2));
    for r in pairs_of(&d, lb, la) {
        assert_eq!(
            r.edges, 2,
            "({}, {}) sample {}: label {}",
            r.x, r.y, r.sample, r.label
        );
    }
}

#[test]
fn a_flat_record_that_keeps_no_identity_still_witnesses_a_shared_one() {
    // §7, may share (2): some flat registry-offered record `M` passes the width test with each of
    // two records with no plane of their own. §5: a flat triangle keyed on screen carries no
    // identity, though it took part in the partition. At focal distance 100, a triangle on x = 1
    // with first corner (1, 10^10, 0) keys no plane of its own; two lines on x = 1 pass the width
    // test with it, so the pair may share its identity, and every edge is admitted.
    let mut b = Build::new(true);
    b.mesh(
        vec![[1.0, 1e10, 0.0], [1.0, -1e10, 0.0], [1.0, 0.0, 50.0]],
        vec![[0, 1, 2]],
        OPAQUE_BLUE,
    );
    b.stroke(
        open(&[[1.0, -3.05, 0.0], [1.0, 3.05, 0.0]]),
        OPAQUE_GREEN,
        0.5,
    );
    b.stroke(
        open(&[[1.0, -3.05, 0.5], [1.0, 3.05, 0.5]]),
        OPAQUE_RED,
        0.5,
    );
    let d = decide(&b, view(100.0));
    let (p, q) = (id_of(&d, 1, 2), id_of(&d, 2, 2));
    for r in pairs_of(&d, p, q) {
        assert_eq!(
            r.edges, 7,
            "({}, {}) sample {}: label {}",
            r.x, r.y, r.sample, r.label
        );
    }
}

#[test]
fn a_flat_record_that_keeps_no_identity_still_takes_a_line_out_of_the_axis_rule_alone() {
    // §7, may share (3): the axis rule alone holds where no flat record passes (1) with either
    // record. A triangle on z = 0 with first corner (1, 10^12, 0), wide enough not to lie on a
    // line, keys no plane of its own at
    // focal distance 100; line `A` on x = 1, z = 0 passes (1) with it, line `B` on x = 1, z = 0.5
    // does not. They share x = 1 by the axis rule, but not alone: `A` may read its own key, and
    // `B`, nearer, may be strongly in front.
    let mut b = Build::new(true);
    b.mesh(
        vec![[1.0, 1e12, 0.0], [1.0, -1e12, 0.0], [20_000.0, 0.0, 0.0]],
        vec![[0, 1, 2]],
        OPAQUE_BLUE,
    );
    b.stroke(
        open(&[[1.0, -3.05, 0.0], [1.0, 3.05, 0.0]]),
        OPAQUE_GREEN,
        0.5,
    );
    b.stroke(
        open(&[[1.0, -3.05, 0.5], [1.0, 3.05, 0.5]]),
        OPAQUE_RED,
        0.5,
    );
    let d = decide(&b, view(100.0));
    let (p, q) = (id_of(&d, 1, 2), id_of(&d, 2, 2));
    for r in pairs_of(&d, p, q) {
        assert_eq!(
            r.edges & 4,
            4,
            "({}, {}) sample {}: edges {} label {}",
            r.x,
            r.y,
            r.sample,
            r.edges,
            r.label
        );
    }
}

#[test]
fn a_record_with_no_plane_of_its_own_is_no_flat_witness_for_itself() {
    // §7: only flat records found identities, so two records with no plane of their own share one
    // only through a flat record or by the axis rule. One straight line on z = 0, its background
    // stroke and its stroke, with no flat record in the frame: two strokes of one path keep depth
    // (`own_layers@1`), and on their axis plane the two rows are one computation, an `ExactTie`.
    let mut b = Build::new(true);
    let r = b.stroke(
        open(&[[-3.05, 0.0, 0.0], [3.05, 0.0, 0.0]]),
        OPAQUE_GREEN,
        0.5,
    );
    r.paint.background_stroke = Paint::One(OPAQUE_RED);
    r.paint.background_stroke_width = 1.0;
    let d = decide(&b, view(20.0));
    let (bg, fg) = (id_of(&d, 0, 0), id_of(&d, 0, 2));
    for r in pairs_of(&d, bg, fg) {
        assert_eq!(
            (r.edges, r.label),
            (1, 4),
            "({}, {}) sample {}",
            r.x,
            r.y,
            r.sample
        );
    }
}

#[test]
fn a_stroke_adjusted_on_its_own_fill_keeps_the_hull_of_both_intervals() {
    // §6 `band_input@1`: a stroke adjusted on its own fill takes the hull of the two keys'
    // intervals, disjoint or not. A square sloping towards the eye from its edge on x = 0 (z = 0)
    // to x = −1.5 (z = 10), stroked 2 wide: just inside the near edge the stroke reads its
    // centreline there, keyed −1, and the fill a nearer key, the two intervals apart. Its edges'
    // lengths on screen keep away from a cubic count step (`curve_recut`).
    let mut b = Build::new(true);
    let r = b.fill(
        polygon(&[
            [0.0, -2.05, 0.0],
            [0.0, 2.05, 0.0],
            [-1.5, 2.05, 10.0],
            [-1.5, -2.05, 10.0],
        ]),
        OPAQUE_BLUE,
    );
    r.paint.stroke = Paint::One(WHITE);
    r.paint.stroke_width = 2.0;
    let d = decide(&b, view(20.0));
    let (fill, stroke) = (id_of(&d, 0, 1), id_of(&d, 0, 2));
    let row = |f: u32, x: u32, y: u32| {
        d.bands
            .iter()
            .find(|b| (b.fragment, b.x, b.y, b.which) == (f, x, y, 0))
    };
    let mut apart = 0;
    for f in d
        .bands
        .iter()
        .filter(|b| b.fragment == fill && b.which == 0)
    {
        let Some(s) = row(stroke, f.x, f.y) else {
            continue;
        };
        for k in 0..16 {
            // Within a pixel of the near edge (shown x = 0), far from the far one (shown x = −3) and
            // the ends.
            let p = sample_point(f.x as i64, f.y as i64, k);
            let (x, y) = ((p[0] - 32.0) / 4.0, -(p[1] - 32.0) / 4.0);
            if f.status[k] != 1 || s.status[k] != 1 || !(x > -0.25 && x < 0.0 && y.abs() < 0.5) {
                continue;
            }
            let fill_hi = f.o[k] + f.s_plus[k] + 3.0 * f.r[k];
            // The stroke's own key on its near edge is −1: where the fill's interval lies wholly
            // nearer, the hull reaches −1.
            if fill_hi >= -1.0 - 1e-6 {
                continue;
            }
            let stroke_hi = s.o[k] + s.s_plus[k] + 3.0 * s.r[k];
            assert!(
                stroke_hi >= -1.0 - 1e-9,
                "({}, {}) sample {k}: the stroke's interval ends at {stroke_hi}, the fill's at \
                 {fill_hi}",
                f.x,
                f.y
            );
            apart += 1;
        }
    }
    assert!(apart > 0, "no sample where the fill's interval is apart");
}
