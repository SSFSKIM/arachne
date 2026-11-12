//! The edges a pair admits at a sample where the contract fixes them in closed form (§5, §6, §7),
//! each in a scene of view C's frame (64 x 64 px over 16 x 16 units, 4 px a unit, centre at pixel
//! 32) under the identity rotation at the focal distance each case names.

#[path = "common/scenes.rs"]
mod scenes;

use arachne_core::xf::{Camera, Paint, View};
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
    // Records of one shape upload it once.
    let mut delta = b.delta(1, v);
    let mut seen = std::collections::HashSet::new();
    delta.upload.retain(|s| seen.insert(s.key()));
    let d = decide_at(&[delta], 1, None).unwrap();
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
fn a_stroke_adjusted_on_its_own_fill_ends_at_the_nearer_far_end_where_both_cover() {
    // §6 `band_input@3`: a stroke adjusted on its own fill, where both cover certainly, runs from
    // the nearer of the two intervals' near ends to the nearer of their far ends; where the fill's
    // cover is `Either`, to the stroke's own far end. A square sloping towards the eye from its
    // edge on x = 0 (z = 0) to x = −1.5 (z = 10), stroked 2 wide: just inside the near edge the
    // stroke reads its centreline there, keyed −1, and the fill a nearer key, the two intervals
    // apart. Its edges' lengths on screen keep away from a cubic count step (`curve_recut`).
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
    let cover = |f: u32, x: u32, y: u32| {
        d.covers
            .iter()
            .find(|c| (c.fragment, c.x, c.y) == (f, x, y))
            .map_or([0; 16], |c| c.cover)
    };
    let mut certain = 0;
    for f in d
        .bands
        .iter()
        .filter(|b| b.fragment == fill && b.which == 0)
    {
        let Some(s) = row(stroke, f.x, f.y) else {
            continue;
        };
        let (cs, cf) = (cover(stroke, f.x, f.y), cover(fill, f.x, f.y));
        for k in 0..16 {
            // Within a pixel of the near edge (shown x = 0), far from the far one (shown x = −3) and
            // the ends.
            let p = sample_point(f.x as i64, f.y as i64, k);
            let (x, y) = ((p[0] - 32.0) / 4.0, -(p[1] - 32.0) / 4.0);
            if f.status[k] != 1 || s.status[k] != 1 || !(x > -0.25 && x < 0.0 && y.abs() < 0.5) {
                continue;
            }
            let fill_hi = f.o[k] + f.s_plus[k] + 3.0 * f.r[k];
            // The stroke's own key on its near edge is −1, and the fill's interval lies wholly
            // nearer.
            if fill_hi >= -1.0 - 1e-6 || cs[k] == 0 {
                continue;
            }
            let stroke_hi = s.o[k] + s.s_plus[k] + 3.0 * s.r[k];
            let at = format!(
                "({}, {}) sample {k}: the stroke's interval ends at {stroke_hi}, the fill's at {fill_hi}",
                f.x, f.y
            );
            match cf[k] {
                1 => {
                    // It never reaches the stroke's own key, which the nearer of the two never
                    // takes.
                    assert!(stroke_hi < -1.0 - 1e-6, "{at}");
                    certain += 1;
                }
                _ => {
                    assert!(stroke_hi >= -1.0 - 1e-9, "{at}");
                }
            }
        }
    }
    assert!(
        certain > 0,
        "no sample where both cover and the fill's interval is apart"
    );
}

#[test]
fn over_an_either_fill_an_adjusted_stroke_s_own_key_is_written_unadjusted() {
    // §5 `stroke_on_own_fill@2`: where the fill's cover is `Either`, the core may keep the
    // stroke's own key, which `BAND` writes unadjusted on the stroke's `which` 2 row there, and
    // nowhere else. A square on z = x / 2 (its plane, the fill's key, nearer than the stroke's
    // curve on one side), stroked 1 wide, on a grid of 0.05: its edge samples are `Either`.
    let mut b = Build::new(true);
    let r = b.fill(
        polygon(&[
            [-3.0, -3.0, -1.5],
            [3.0, -3.0, 1.5],
            [3.0, 3.0, 1.5],
            [-3.0, 3.0, -1.5],
        ]),
        OPAQUE_BLUE,
    );
    r.paint.stroke = Paint::One(WHITE);
    r.paint.stroke_width = 1.0;
    let d = decide(
        &b,
        View {
            input_grid: 0.05,
            ..view(20.0)
        },
    );
    let (fill, stroke) = (id_of(&d, 0, 1), id_of(&d, 0, 2));
    let cover = |f: u32, x: u32, y: u32| {
        d.covers
            .iter()
            .find(|c| (c.fragment, c.x, c.y) == (f, x, y))
            .map_or([0; 16], |c| c.cover)
    };
    let mut written = 0;
    for row in d
        .bands
        .iter()
        .filter(|b| b.fragment == stroke && b.which == 0)
    {
        let (cs, cf) = (cover(stroke, row.x, row.y), cover(fill, row.x, row.y));
        let two = d
            .bands
            .iter()
            .find(|b| (b.fragment, b.x, b.y, b.which) == (stroke, row.x, row.y, 2));
        for k in 0..16 {
            let either = cs[k] != 0 && cf[k] == 2;
            let at = format!("({}, {}) sample {k}", row.x, row.y);
            match two {
                Some(t) if either => {
                    assert_ne!(t.status[k], 0, "{at}");
                    written += 1;
                }
                Some(t) => assert_eq!(t.status[k], 0, "{at}"),
                None => assert!(!either, "{at}: no unadjusted row"),
            }
        }
    }
    assert!(written > 0, "no sample over an Either fill");
}

#[test]
fn an_adjusted_stroke_unable_at_a_pixel_is_unable_against_others_and_judged_against_its_fill() {
    // §5 `stroke_on_own_fill@2`: an adjusted stroke's bound is the pixel's largest taken, so where
    // one adjusted read is `curve_recut` (§6, per read) every sample of it at the pixel is unable,
    // and a pair reading its key there is `truth_unable`; `own_layers@1` with its fill reads no
    // key and stays judged. A rectangle on z = 0, 18 px wide (each edge's control polygon on a cubic
    // count step: its reads `curve_recut`) and 17.2 px tall (off one), stroked 4 px wide, over a
    // triangle behind it on z = −1. At its corners the stroke's samples read either edge.
    let mut b = Build::new(true);
    b.mesh(
        vec![[-6.0, -6.0, -1.0], [6.0, -6.0, -1.0], [0.0, 6.0, -1.0]],
        vec![[0, 1, 2]],
        OPAQUE_GREEN,
    );
    let r = b.fill(rect([-2.25, 2.25], [-2.15, 2.15], 0.0), OPAQUE_BLUE);
    r.paint.stroke = Paint::One(WHITE);
    r.paint.stroke_width = 1.0;
    let d = decide(&b, view(20.0));
    let (mesh, fill, stroke) = (id_of(&d, 0, 3), id_of(&d, 1, 1), id_of(&d, 1, 2));
    let cover = |f: u32, x: u32, y: u32| {
        d.covers
            .iter()
            .find(|c| (c.fragment, c.x, c.y) == (f, x, y))
            .map_or([0; 16], |c| c.cover)
    };
    let mut witnessed = 0;
    for row in d
        .bands
        .iter()
        .filter(|b| b.fragment == stroke && b.which == 0)
    {
        let (x, y) = (row.x, row.y);
        let (cs, cf) = (cover(stroke, x, y), cover(fill, x, y));
        let unable_adjusted = (0..16).any(|k| cs[k] != 0 && cf[k] != 0 && row.status[k] == 3);
        if !unable_adjusted {
            continue;
        }
        // Every sample of the stroke at the pixel is unable, its fill's or not.
        for k in (0..16).filter(|&k| cs[k] != 0) {
            assert_eq!(row.status[k], 3, "({x}, {y}) sample {k}");
        }
        if (0..16).any(|k| cs[k] != 0 && cf[k] == 0) {
            witnessed += 1;
        }
        for p in d.pairs.iter().filter(|p| (p.x, p.y) == (x, y)) {
            let k = p.sample as usize;
            if (p.a, p.b) == (mesh, stroke) || (p.a, p.b) == (stroke, mesh) {
                assert_eq!(
                    p.label, 0,
                    "({x}, {y}) sample {k}: the stroke against the mesh"
                );
            }
            if (p.a, p.b) == (fill, stroke) || (p.a, p.b) == (stroke, fill) {
                assert_eq!(
                    p.label, 5,
                    "({x}, {y}) sample {k}: the stroke against its fill"
                );
            }
        }
    }
    assert!(
        witnessed > 0,
        "no pixel where an unable adjusted read reaches another sample"
    );
}

#[test]
fn an_adjusted_stroke_is_unable_at_a_pixel_where_one_of_its_unadjusted_reads_is() {
    // §5 `stroke_on_own_fill@2`: the pixel's bound is the largest over the samples of the bound
    // of the key the stroke took, its own where the fill does not cover; one `curve_recut` read
    // there (§6) leaves that largest unstated at every sample of the stroke at the pixel. One
    // record, filled and stroked 4 px wide: a square 17.2 px across (off a count step), and an open
    // line 18 px long (its cubic on a step: its reads `curve_recut`) 2.25 px above the square's
    // top edge. In the row of pixels across that edge, the line's body reaches the samples above
    // it, outside the fill, and the samples below, inside the fill and the square's stroke, lie
    // farther than its body from it.
    let mut b = Build::new(true);
    let mut points = rect([-2.15, 2.15], [-2.15, 2.15], 0.0);
    points.extend(open(&[[-2.25, 2.7125, 0.0], [2.25, 2.7125, 0.0]]));
    let r = b.fill(points, OPAQUE_BLUE);
    r.paint.stroke = Paint::One(WHITE);
    r.paint.stroke_width = 1.0;
    let d = decide(&b, view(20.0));
    let (fill, stroke) = (id_of(&d, 0, 1), id_of(&d, 0, 2));
    let cover = |f: u32, x: u32, y: u32| {
        d.covers
            .iter()
            .find(|c| (c.fragment, c.x, c.y) == (f, x, y))
            .map_or([0; 16], |c| c.cover)
    };
    let mut mixed = 0;
    for row in d
        .bands
        .iter()
        .filter(|b| b.fragment == stroke && b.which == 0)
    {
        let (x, y) = (row.x, row.y);
        let (cs, cf) = (cover(stroke, x, y), cover(fill, x, y));
        let adjusted = (0..16).any(|k| cs[k] != 0 && cf[k] != 0);
        let unadjusted_unable = (0..16).any(|k| cs[k] != 0 && cf[k] == 0 && row.status[k] == 3);
        if !(adjusted && unadjusted_unable) {
            continue;
        }
        for k in (0..16).filter(|&k| cs[k] != 0) {
            assert_eq!(row.status[k], 3, "({x}, {y}) sample {k}");
        }
        mixed += 1;
    }
    assert!(
        mixed > 0,
        "no adjusted stroke pixel with an unable read where the fill does not cover"
    );
}

#[test]
fn two_flat_triangles_that_may_share_tie_within_term_though_neither_has_a_row() {
    // §7, two flat records that may share: in the shared scenario each reads its row, a triangle
    // its own key (it has none, §5), and the term is `within_term` with the snap shares. At focal
    // distance 10 and grid 0.001, `B` [(−1, −0.05, 0), (1, −0.05, 0), (0, 0.05, 0.003)] painted
    // first and `A` the same with its third corner on z = 0: they pass the width test, are not
    // near-parallel, and at pixel (31, 31) sample 14 their own intervals do not meet while the
    // shared term holds. Weak and `B` in front are admitted.
    let mut b = Build::new(true);
    b.mesh(
        vec![[-1.0, -0.05, 0.0], [1.0, -0.05, 0.0], [0.0, 0.05, 0.003]],
        vec![[0, 1, 2]],
        OPAQUE_BLUE,
    );
    b.mesh(
        vec![[-1.0, -0.05, 0.0], [1.0, -0.05, 0.0], [0.0, 0.05, 0.0]],
        vec![[0, 1, 2]],
        OPAQUE_RED,
    );
    let d = decide(
        &b,
        View {
            input_grid: 0.001,
            ..view(10.0)
        },
    );
    let (tb, ta) = (id_of(&d, 0, 3), id_of(&d, 1, 3));
    let at = pairs_of(&d, tb, ta)
        .into_iter()
        .find(|r| (r.x, r.y, r.sample) == (31, 31, 14))
        .expect("no pair at (31, 31) sample 14");
    assert_eq!((at.edges, at.label), (3, 6));
}

/// Two point blocks of radius 2 at (0.125, −0.125, 0), red then blue, with `between` drawn
/// between them, at focal distance 10 and grid 0: the pair of the two blocks at pixel (32, 32).
fn two_co_placed_blocks(between: impl FnOnce(&mut Build)) -> Vec<(u8, u8)> {
    two_blocks([0.125, -0.125, 0.0], [0.125, -0.125, 0.0], false, between)
}

/// Point blocks of radius 2 at `p` (red) and `q` (blue), with `between` drawn between them, at
/// focal distance 10 and grid 0, under the exponential projection where `exponential`: the pair
/// of the two blocks at pixel (32, 32).
fn two_blocks(
    p: [f64; 3],
    q: [f64; 3],
    exponential: bool,
    between: impl FnOnce(&mut Build),
) -> Vec<(u8, u8)> {
    let mut b = Build::new(true);
    b.points(vec![p], OPAQUE_RED, 2.0);
    between(&mut b);
    let last = b.records.len() as u32;
    b.points(vec![q], OPAQUE_BLUE, 2.0);
    let mut v = View {
        input_grid: 0.0,
        ..view(10.0)
    };
    if let Camera::ThreeD(c) = &mut v.camera {
        c.exponential_projection = exponential;
    }
    let d = decide(&b, v);
    let (p, q) = (id_of(&d, 0, 4), id_of(&d, last, 4));
    pairs_of(&d, p, q)
        .into_iter()
        .filter(|r| (r.x, r.y) == (32, 32))
        .map(|r| (r.edges, r.label))
        .collect()
}

#[test]
fn two_keys_of_one_computation_in_every_scenario_are_an_exact_tie() {
    // §7 `ExactTie`: "any other pair whose keys are one computation in every scenario". Two
    // projected point blocks at one placed position are keyed on one facing plane, their only
    // scenario: weak alone.
    let alone = two_co_placed_blocks(|_| {});
    assert!(!alone.is_empty());
    assert!(alone.iter().all(|e| *e == (1, 4)), "{alone:?}");
    // With a triangle on z = −5 painted between them, on no plane of theirs.
    let apart = two_co_placed_blocks(|b| {
        b.mesh(
            vec![[-2.0, -2.0, -5.0], [2.0, -2.0, -5.0], [0.0, 2.0, -5.0]],
            vec![[0, 1, 2]],
            OPAQUE_GREEN,
        );
    });
    assert!(!apart.is_empty());
    assert!(apart.iter().all(|e| *e == (1, 4)), "{apart:?}");
}

#[test]
fn two_facing_keys_of_one_centre_are_an_exact_tie_where_the_facing_plane_does_not_key() {
    // §7 `ExactTie`, as above, under the exponential projection: the facing plane does not key
    // (§5), and each block is keyed by its centre's own key, −exp(0) = −1, from one input. One
    // computation: weak alone.
    let p = [0.125, -0.125, 0.0];
    let one = two_blocks(p, p, true, |_| {});
    assert!(!one.is_empty());
    assert!(one.iter().all(|e| *e == (1, 4)), "{one:?}");
    // Two centres apart in one pixel on z = 0 key −1 too, the same value from other inputs: a tie
    // by band, not one computation.
    let apart = two_blocks(p, [0.1875, -0.125, 0.0], true, |_| {});
    assert!(!apart.is_empty());
    assert!(apart.iter().all(|e| *e == (7, 3)), "{apart:?}");
}

/// The pair rows, as `(edges, label)`, of two strokes of the polyline through `(−2, 0, 0)`,
/// `(0, 0, 1)`, `(2, 1, −1)` and `(2, −1, 2)`, on no plane, red of width 0.5 then blue of width
/// `blue_width`, in view C.
fn two_strokes_of_one_polyline(blue_width: f64) -> Vec<(u8, u8)> {
    let pts = open(&[
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 1.0, -1.0],
        [2.0, -1.0, 2.0],
    ]);
    let mut b = Build::new(true);
    b.stroke(pts.clone(), OPAQUE_RED, 0.5);
    b.stroke(pts, OPAQUE_BLUE, blue_width);
    let d = decide(&b, view_c());
    let (p, q) = (id_of(&d, 0, 2), id_of(&d, 1, 2));
    pairs_of(&d, p, q)
        .into_iter()
        .map(|r| (r.edges, r.label))
        .collect()
}

#[test]
fn two_strokes_of_one_curve_read_at_one_point_are_an_exact_tie() {
    // §7 `ExactTie`: "any other pair whose keys are one computation in every scenario". Two
    // strokes of one polyline on no plane, of one width, are keyed along one centreline built
    // from one input (§5, `curve_key@1`) and read at one sample: one computation, weak alone.
    // Where a read is `curve_recut` (§6, per read) the pair stays unable.
    let one = two_strokes_of_one_polyline(0.5);
    let off: Vec<_> = one
        .iter()
        .filter(|e| **e != (1, 4) && **e != (7, 0))
        .collect();
    assert!(one.contains(&(1, 4)), "{one:?}");
    assert!(
        off.is_empty(),
        "{} of {}: {:?}",
        off.len(),
        one.len(),
        &off[..off.len().min(4)]
    );
    // Of another width, the read takes another body: not one computation.
    let wider = two_strokes_of_one_polyline(1.0);
    assert!(!wider.is_empty());
    assert!(wider.iter().all(|e| e.1 != 4), "{wider:?}");
}

/// View C with its rotation the identity exactly (`theta` −90° as the projection's degree), so a
/// pre-projected record's camera axes are the scene's.
fn view_c_exact() -> View {
    let mut v = view_c();
    if let Camera::ThreeD(c) = &mut v.camera {
        c.theta = -90.0 * (std::f64::consts::TAU / 360.0);
    }
    v
}

/// The pair rows, as `(edges, label)`, of two world-space records stroking the polyline through
/// `(−3, 0, 0)`, `(0, 0, 1.5)`, `(3, 1.5, −1.5)` and `(3, −1.5, 3)` (its controls exact) in
/// `view_c_exact`, red then blue, each stroke of width 0.5 over a background stroke of width 1,
/// after `edit`; read on the red's layer `la` and the blue's `lb` (0 background stroke, 2
/// stroke). `pre`: both pre-projected, their tails the polyline. Asserts the two reads key alike:
/// every pair-sample at which both are bounded reads one key, and some are.
fn two_curve_records(pre: bool, (la, lb): (u8, u8), edit: impl Fn(&mut Build)) -> Vec<(u8, u8)> {
    two_curve_reads(pre, (la, lb), edit, true)
}

/// [`two_curve_records`], asserting the two reads key alike only where `alike`.
fn two_curve_reads(
    pre: bool,
    (la, lb): (u8, u8),
    edit: impl Fn(&mut Build),
    alike: bool,
) -> Vec<(u8, u8)> {
    let pts = open(&[
        [-3.0, 0.0, 0.0],
        [0.0, 0.0, 1.5],
        [3.0, 1.5, -1.5],
        [3.0, -1.5, 3.0],
    ]);
    curve_pair(pts, pre, (la, lb), edit, alike)
}

/// [`two_curve_reads`] of the curve through the cubic control points `pts`.
fn curve_pair(
    pts: Vec<[f64; 3]>,
    pre: bool,
    (la, lb): (u8, u8),
    edit: impl Fn(&mut Build),
    alike: bool,
) -> Vec<(u8, u8)> {
    let mut b = Build::new(true);
    for rgba in [OPAQUE_RED, OPAQUE_BLUE] {
        let r = if pre {
            b.pre_projected(pts.clone(), rgba)
        } else {
            b.path(pts.clone())
        };
        if let Some(tail) = &mut r.pre_projected {
            tail.fill = Paint::None;
            tail.stroke = Paint::One(rgba);
            tail.background_stroke = Paint::One(BLACK);
        }
        r.paint.stroke = Paint::One(rgba);
        r.paint.background_stroke = Paint::One(BLACK);
        r.paint.stroke_width = 0.5;
        r.paint.background_stroke_width = 1.0;
    }
    edit(&mut b);
    let d = decide(&b, view_c_exact());
    let (p, q) = (id_of(&d, 0, la), id_of(&d, 1, lb));
    let rows = pairs_of(&d, p, q);
    let own = |f: u32, x: u32, y: u32| {
        d.bands
            .iter()
            .find(|r| (r.fragment, r.x, r.y, r.which) == (f, x, y, 0))
            .unwrap()
    };
    let mut bounded = 0;
    for r in &rows {
        let s = r.sample as usize;
        let (a, b) = (own(p, r.x, r.y), own(q, r.x, r.y));
        if alike && a.status[s] == 1 && b.status[s] == 1 {
            bounded += 1;
            assert_eq!(a.o[s].to_bits(), b.o[s].to_bits(), "({}, {}) {s}", r.x, r.y);
        }
    }
    assert!(
        !alike || bounded > 0,
        "no pair-sample with both reads bounded"
    );
    rows.into_iter().map(|r| (r.edges, r.label)).collect()
}

/// Asserts `rows` hold pair-samples and none an `ExactTie`.
fn no_exact_tie(what: &str, rows: &[(u8, u8)]) {
    assert!(!rows.is_empty(), "{what}: no pair-sample");
    assert!(
        rows.iter().all(|e| e.1 != 4),
        "{what}: {:?}",
        rows.iter().filter(|e| e.1 == 4).take(4).collect::<Vec<_>>()
    );
}

/// Asserts the records as made, read on their strokes, hold an `ExactTie`, and once `edit`
/// makes one record input differ (the reads keying alike still), none.
fn not_one_computation(pre: bool, what: &str, edit: impl Fn(&mut Build)) {
    let control = two_curve_records(pre, (2, 2), |_| {});
    assert!(control.contains(&(1, 4)), "{what}: control");
    no_exact_tie(what, &two_curve_records(pre, (2, 2), edit));
}

#[test]
fn equal_shown_points_placed_from_another_shape_are_not_one_computation() {
    // The core keys a curve from its record's inputs, which it places itself: a shape moved by one
    // placement and a shape moved by another, shown at equal points, are two computations.
    not_one_computation(false, "placings", |b| {
        let moved = open(&[
            [-4.0, 0.0, 0.0],
            [-1.0, 0.0, 1.5],
            [2.0, 1.5, -1.5],
            [2.0, -1.5, 3.0],
        ]);
        let shape =
            arachne_core::xf::Shape::new(arachne_core::xf::ShapeData::Path { points: moved });
        let r = &mut b.records[1];
        r.geometry.shape_a = Some(shape.key());
        r.geometry.place_a = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        b.shapes.push(shape);
    });
}

#[test]
fn equal_shown_points_under_a_full_reveal_window_are_not_one_computation() {
    not_one_computation(false, "reveal", |b| {
        b.records[1].geometry.reveal = Some((0.0, 1.0));
    });
}

#[test]
fn equal_shown_points_blended_at_another_t_are_not_one_computation() {
    // Both blended toward their own shape, so the points do not move; `t` differs.
    not_one_computation(false, "blend t", |b| {
        for (r, t) in b.records.iter_mut().zip([0.25, 0.5]) {
            r.geometry.blend = Some(arachne_core::xf::Blend {
                shape_b: r.geometry.shape_a.unwrap(),
                place_b: arachne_core::xf::IDENTITY,
                t,
            });
        }
    });
}

#[test]
fn equal_bodies_of_other_stroke_widths_are_not_one_computation() {
    // The red's stroke against the blue's background stroke, both 0.5 wide once the blue's two
    // widths are swapped: one body, other width inputs.
    let control = two_curve_records(false, (2, 2), |_| {});
    assert!(control.contains(&(1, 4)), "control");
    let swapped = two_curve_records(false, (2, 0), |b| {
        b.records[1].paint.stroke_width = 1.0;
        b.records[1].paint.background_stroke_width = 0.5;
    });
    no_exact_tie("widths", &swapped);
    // A paint blend whose widths are the record's own: one body, other width inputs.
    let blended = two_curve_records(false, (2, 2), |b| {
        let p = &mut b.records[1].paint;
        p.paint_b = Some(arachne_core::xf::PaintB {
            fill: Paint::None,
            stroke: p.stroke.clone(),
            background_stroke: p.background_stroke.clone(),
            stroke_width: 0.5,
            background_stroke_width: 1.0,
        });
        p.mix = 0.5;
    });
    no_exact_tie("paint blend", &blended);
    // The width inputs equal, the two layers' bodies differ (0.5 against 1), and so may the
    // segment a read holds.
    no_exact_tie("body", &two_curve_reads(false, (2, 0), |_| {}, false));
}

#[test]
fn equal_tails_of_pre_projected_records_of_other_shapes_are_not_one_computation() {
    // A pre-projected record is shown at its tail's points; the core reads its own shape too.
    not_one_computation(true, "pre-projected shape", |b| {
        let other = arachne_core::xf::Shape::new(arachne_core::xf::ShapeData::Path {
            points: open(&[[0.0; 3], [1.0, 0.0, 0.0]]),
        });
        b.records[1].geometry.shape_a = Some(other.key());
        b.shapes.push(other);
    });
}

#[test]
fn equal_shown_points_of_another_depth_space_are_not_one_computation() {
    // Pre-projected records are kept as produced in either space.
    not_one_computation(true, "depth space", |b| {
        b.records[1].depth.space = arachne_core::xf::Space::Screen;
    });
}

/// Record `i` of `b` made a face group of one face, its run the record's points, stroked `rgba`
/// from its own per-face table and filled with nothing (alpha 0, so no stroke read is adjusted on
/// its own fill), with `face_ends` and `face_widths` as given.
fn as_face_group(b: &mut Build, i: usize, rgba: [f64; 4], ends: Vec<u32>, widths: Vec<f64>) {
    let stroke = arachne_core::xf::Stops::new(vec![rgba]);
    let fill = arachne_core::xf::Stops::new(vec![[rgba[0], rgba[1], rgba[2], 0.0]]);
    let (key, fill_key) = (stroke.key(), fill.key());
    b.paints.push(stroke);
    b.paints.push(fill);
    let r = &mut b.records[i];
    r.kind = arachne_core::xf::RecordKind::FaceGroup;
    r.body = arachne_core::xf::Body::FaceGroup(arachne_core::xf::FaceGroup {
        uv: vec![[0, 0]],
        grid: [1, 1],
        fill: fill_key,
        stroke: Some(key),
        face_ends: ends,
        face_widths: widths,
    });
}

#[test]
fn a_path_and_a_face_group_of_one_face_at_equal_points_are_not_one_computation() {
    // XF 1.3: a face group with no face ends cuts its points into one equal run per face, so one
    // face of no face ends runs over every point, as a path does: the kind alone differs.
    not_one_computation(false, "kind", |b| {
        as_face_group(b, 1, OPAQUE_BLUE, Vec::new(), Vec::new())
    });
}

#[test]
fn face_groups_of_one_run_under_other_face_lists_are_not_one_computation() {
    // One face over all 12 points with no face ends, or with the one face end 12; and of no face
    // widths, or with the face width the record's own.
    let both = |ends: Vec<u32>, widths: Vec<f64>| {
        move |b: &mut Build| {
            as_face_group(b, 0, OPAQUE_RED, Vec::new(), Vec::new());
            as_face_group(b, 1, OPAQUE_BLUE, ends.clone(), widths.clone());
        }
    };
    let control = two_curve_records(false, (2, 2), both(Vec::new(), Vec::new()));
    assert!(control.contains(&(1, 4)), "control");
    no_exact_tie(
        "face ends",
        &two_curve_records(false, (2, 2), both(vec![12], Vec::new())),
    );
    no_exact_tie(
        "face widths",
        &two_curve_records(false, (2, 2), both(Vec::new(), vec![0.5])),
    );
}

#[test]
fn a_pre_projected_record_and_a_projected_one_at_equal_points_are_not_one_computation() {
    // Each point on `z = 0` or on the axis `x = y = 0` projects to itself in `view_c_exact`
    // (factor 1 at `z = 0`), so a world-space record of these points and a pre-projected record
    // whose tail holds them are shown, placed and keyed alike: only the tail differs. Not planar:
    // `(0, 0, 1)` lies off `z = 0`.
    let pts = vec![
        [-1.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.75, 0.25, 0.0],
        [0.25, 0.75, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let control = curve_pair(pts.clone(), false, (2, 2), |_| {}, true);
    assert!(control.contains(&(1, 4)), "control");
    let tail = curve_pair(
        pts.clone(),
        false,
        (2, 2),
        |b| {
            let r = &mut b.records[1];
            r.features |= arachne_core::xf::FEATURE_PRE_PROJECTED;
            r.pre_projected = Some(arachne_core::xf::PreProjected {
                points: r.geometry.shape_a.unwrap(),
                gradient: None,
                fill: Paint::None,
                stroke: r.paint.stroke.clone(),
                background_stroke: r.paint.background_stroke.clone(),
            });
        },
        true,
    );
    no_exact_tie("pre-projected", &tail);
}

#[test]
fn a_record_kept_facing_its_centre_never_reaches_a_curve_key() {
    // §5: a world-space record with a fixed-orientation centre is keyed by its facing plane, a
    // pre-projected one with a centre by its pivot's facing plane, and a screen-space one keys
    // −∞, whose pairs are an `ExactTie` by rule (§7: "both keys are `−∞`"; §6: "Two `−∞` keys
    // are equal, and the label is `ExactTie`"). No curve read carries a centre, so `CurveKey`
    // compares it for completeness alone; a change that keys a centred record along its curve
    // fails here.
    use arachne_oracle::keys::Model;
    let pts = open(&[
        [-3.0, 0.0, 0.0],
        [0.0, 0.0, 1.5],
        [3.0, 1.5, -1.5],
        [3.0, -1.5, 3.0],
    ]);
    for (world, pre) in [(true, false), (true, true), (false, false)] {
        let mut b = Build::new(world);
        let r = if pre {
            b.pre_projected(pts.clone(), OPAQUE_RED)
        } else {
            b.path(pts.clone())
        };
        if let Some(tail) = &mut r.pre_projected {
            tail.stroke = Paint::One(OPAQUE_RED);
        }
        r.paint.stroke = Paint::One(OPAQUE_RED);
        r.paint.stroke_width = 0.5;
        r.depth.fixed_orientation_centre = Some([0.0, 0.0, 1.0]);
        let v = view_c_exact();
        let delta = b.delta(1, v.clone());
        let mut state = arachne_oracle::stream::State::default();
        state.apply(&delta).unwrap();
        let items = arachne_oracle::draw::items(
            &v,
            &state.records,
            &state.order,
            &state.shapes,
            &state.paints,
        );
        let mut paths = 0;
        for it in &items {
            if let arachne_oracle::draw::Item::Path(p) = it {
                paths += 1;
                let depth = arachne_oracle::keys::path_depth(&v, p, &[None, None, None]);
                for m in &depth.layers {
                    assert!(
                        !matches!(m, Model::Curve { .. }),
                        "world {world} pre {pre}: {m:?}"
                    );
                }
                if !world {
                    assert!(depth.layers.iter().all(|m| matches!(m, Model::NegInf)));
                }
            }
        }
        assert_eq!(paths, 1, "world {world} pre {pre}");
    }
}

// `CurveKey`'s inputs: each above has a scene and a plant (`curve-key-*`), except these, each
// compared for completeness; no pair differs in it alone:
// - `projection`: a frame's path items share one projection (draw.rs `items_of`,
//   `projection(view)`), a pre-projected record's taken in its camera's axes (keys.rs
//   `path_depth`, `in_camera_axes`): it differs only with the pre-projected tail, compared.
// - `map`: every path item maps through its view's (draw.rs `path_item`,
//   `FrameMap::of_paths(view)`), and a frame's pairs are formed within one frame's decision
//   under its one view (pairs.rs `frame`, over that decision's covers).
// - `line_width_multiple`: the view's `cairo_line_width_multiple`, one per frame, as above.
// - `shown`: a path item's shown points are its world points shown by its placement (draw.rs
//   `path_item`, `Placement::show`); its world points are its placings' points under their
//   placements and blend, cut to its reveal window (draw.rs `path_points`; a face's run, an
//   instance member's placement); its placement is fixed by its space, centre and tail
//   (camera.rs `Placement::of`). Equal compared inputs give equal shown points.
// - `fixed_orientation_centre`: no curve read carries one
//   (`a_record_kept_facing_its_centre_never_reaches_a_curve_key`, its plant
//   `curve-key-centre-pin`).

#[test]
fn an_axis_only_pair_reads_no_own_key() {
    // §7: two records with no plane of their own sharing an axis plane by the axis rule alone
    // read their rows on it, one computation: an `ExactTie`, which reads neither own key. Two
    // strokes of one cubic on z = 0 at focal distance 10, its control polygon 18 px long, on a
    // count step: their own reads are `curve_recut` (§6, per read), and the pair needs none.
    let cubic = vec![
        [-2.25, 0.0, 0.0],
        [-0.75, 0.0, 0.0],
        [0.75, 0.0, 0.0],
        [2.25, 0.0, 0.0],
    ];
    let mut b = Build::new(true);
    b.stroke(cubic.clone(), OPAQUE_GREEN, 0.5);
    b.stroke(cubic, OPAQUE_RED, 0.5);
    let d = decide(
        &b,
        View {
            input_grid: 0.0,
            ..view(10.0)
        },
    );
    let (p, q) = (id_of(&d, 0, 2), id_of(&d, 1, 2));
    let rows = pairs_of(&d, p, q);
    let off: Vec<_> = rows
        .iter()
        .filter(|r| (r.edges, r.label) != (1, 4))
        .map(|r| (r.x, r.y, r.sample, r.edges, r.label))
        .collect();
    assert!(
        off.is_empty(),
        "{} of {}: {:?}",
        off.len(),
        rows.len(),
        &off[..off.len().min(4)]
    );
}

#[test]
fn mesh_triangles_on_one_axis_plane_are_read_on_their_own_keys() {
    // §5, `shared_plane_read@1`: "Mesh triangles and point blocks have no rows; they are always
    // read on their own." §7: two records with no plane of their own sharing an axis plane are an
    // `ExactTie` "wherever both rows are read there", and two triangles read none. Two thin
    // triangles on a line, their corners sharing `y = 0` within the grid 0.05, first points
    // equal on it, one about `z = 1` and one about `z = 0`: where their own intervals are apart,
    // the nearer is in front by depth, never by paint order.
    let mut b = Build::new(true);
    for z in [1.0, 0.0] {
        b.mesh(
            vec![[-3.0, 0.0, z - 0.1], [3.0, 0.0, z + 0.1], [0.0, 0.02, z]],
            vec![[0, 1, 2]],
            OPAQUE_BLUE,
        );
    }
    let d = decide(
        &b,
        View {
            input_grid: 0.05,
            ..view_c()
        },
    );
    let (p, q) = (id_of(&d, 0, 3), id_of(&d, 1, 3));
    let own = |f: u32, x: u32, y: u32| {
        d.bands
            .iter()
            .find(|r| (r.fragment, r.x, r.y, r.which) == (f, x, y, 0))
            .unwrap()
    };
    let mut apart = 0;
    for r in pairs_of(&d, p, q) {
        let s = r.sample as usize;
        let (a, b) = (own(p, r.x, r.y), own(q, r.x, r.y));
        if a.status[s] != 1 || b.status[s] != 1 {
            continue;
        }
        let a_hi = a.o[s] + a.s_plus[s] + 3.0 * a.r[s];
        let b_lo = b.o[s] - b.s_minus[s] - 3.0 * b.r[s];
        if a_hi < b_lo {
            apart += 1;
            assert_eq!((r.edges, r.label), (2, 1), "({}, {}) sample {s}", r.x, r.y);
        }
    }
    assert!(apart > 0, "no sample where the own intervals are apart");
}

/// Two mesh triangles across view C's frame, each through `(−6, −6)`, `(6, −6)` and `(0, 6)` and
/// rising along `x` by `slope`, at grid 0: their plane's conforming roundings, and the pair rows.
fn two_sloped_triangles(slopes: [f64; 2]) -> ([usize; 2], Vec<(u8, u8)>) {
    let mut b = Build::new(true);
    for slope in slopes {
        b.mesh(
            vec![
                [-6.0, -6.0, -6.0 * slope],
                [6.0, -6.0, 6.0 * slope],
                [0.0, 6.0, 0.0],
            ],
            vec![[0, 1, 2]],
            OPAQUE_BLUE,
        );
    }
    let v = View {
        input_grid: 0.0,
        ..view_c()
    };
    let mut delta = b.delta(1, v.clone());
    let mut seen = std::collections::HashSet::new();
    delta.upload.retain(|s| seen.insert(s.key()));
    let mut state = arachne_oracle::stream::State::default();
    state.apply(&delta).unwrap();
    let roundings: Vec<usize> = arachne_oracle::draw::items(
        &v,
        &state.records,
        &state.order,
        &state.shapes,
        &state.paints,
    )
    .iter()
    .filter_map(|it| match it {
        arachne_oracle::draw::Item::Mesh(m) => {
            let (_, offered) = arachne_oracle::keys::triangle_depth(&v, m, 0);
            Some(offered?.flat?.conforming().len())
        }
        _ => None,
    })
    .collect();
    let d = decide_at(&[delta], 1, None).unwrap();
    assert_eq!(d.header.status, 0, "{:?}", d.refused);
    let (p, q) = (id_of(&d, 0, 3), id_of(&d, 1, 3));
    let rows = pairs_of(&d, p, q)
        .into_iter()
        .map(|r| (r.edges, r.label))
        .collect();
    ([roundings[0], roundings[1]], rows)
}

#[test]
fn equal_screen_keys_of_planes_whose_rounding_is_not_settled_are_not_one_computation() {
    // §7 `ExactTie`: the keys are one computation "in every scenario". A plane's normal and offset
    // are rounded to the plane grid, and where a value lies within the half-step's margin either
    // rounding conforms. Two triangles a part in a billion and a part in a million above a half
    // step of slope: the first plane may round either way, the second one way, and the two read
    // one screen key as the oracle rounds them. Not one computation: no row is an `ExactTie`.
    let half = 0.5 / 1_048_576.0;
    let (roundings, rows) = two_sloped_triangles([half * (1.0 + 1e-9), half * (1.0 + 1e-6)]);
    assert_eq!(roundings, [2, 1]);
    assert!(!rows.is_empty());
    assert!(
        rows.iter().all(|e| e.1 != 4),
        "{:?}",
        &rows[..rows.len().min(4)]
    );
    // Two triangles of one settled plane are one computation still: weak alone.
    let (roundings, rows) = two_sloped_triangles([half * (1.0 + 1e-6); 2]);
    assert_eq!(roundings, [1, 1]);
    assert!(!rows.is_empty());
    assert!(
        rows.iter().all(|e| *e == (1, 4)),
        "{:?}",
        &rows[..rows.len().min(4)]
    );
}

#[test]
fn a_path_and_a_mesh_triangle_on_one_axis_plane_read_the_path_s_row_against_the_triangle_s_key() {
    // §5, `shared_plane_read@1`: against a fragment it shares an identity with, a path layer is
    // read on its row for that identity; "Mesh triangles and point blocks have no rows; they are
    // always read on their own." §7: two records with no plane of their own on one axis plane are
    // an `ExactTie` "wherever both rows are read there", and here one is not. A line along
    // `y = 0` on `z = 0`, stroked, and a thin triangle on a line within the grid 0.05 on `z = 0`,
    // first points equal on it. The stroke's row on `z = 0` is read off its centreline, outside a
    // hull no wider than `2 τ_P` (a line's), so it is unbounded (§6, `band_input@3`): the pair is
    // `truth_unable` at every sample, every edge admitted. Never weak alone.
    let mut b = Build::new(true);
    b.stroke(
        open(&[[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        OPAQUE_GREEN,
        0.5,
    );
    b.mesh(
        vec![[-3.0, -0.02, 0.0], [3.0, -0.02, 0.0], [0.0, 0.02, 0.0]],
        vec![[0, 1, 2]],
        OPAQUE_BLUE,
    );
    let d = decide(
        &b,
        View {
            input_grid: 0.05,
            ..view_c()
        },
    );
    let (p, q) = (id_of(&d, 0, 2), id_of(&d, 1, 3));
    let rows: Vec<_> = pairs_of(&d, p, q)
        .into_iter()
        .map(|r| (r.edges, r.label))
        .collect();
    assert!(!rows.is_empty());
    assert!(
        rows.iter().all(|e| *e == (7, 0)),
        "{:?}",
        &rows[..rows.len().min(4)]
    );
}

#[test]
fn a_path_s_row_on_a_shared_axis_plane_is_read_where_its_own_key_alone_would_order_the_pair() {
    // §5, `shared_plane_read@1`: against a fragment it shares an identity with, a path layer is
    // read on its row for that identity, never its own key. A line along `y` on `x = 1` at
    // `z = 0`, stroked 4 px wide (23 px long, clear of `curve_recut`), and a triangle on a line
    // across `y` and `z` on `x = 1` within the grid 0.01: neither has a plane of its own, and they
    // share `x = 1`, whose key changes along the screen's `x`. Off the centreline the stroke's own
    // key keeps its centreline's, apart from the triangle's there; its row on `x = 1` is read
    // outside a hull no wider than `2 τ_P` (a line's), so it is unbounded (§6, `band_input@3`).
    // Every pair-sample is `truth_unable`: none is ordered by the own key.
    let mut b = Build::new(true);
    b.stroke(
        open(&[[1.0, -2.875, 0.0], [1.0, 2.875, 0.0]]),
        OPAQUE_GREEN,
        1.0,
    );
    b.mesh(
        vec![[1.0, -1.0, -3.0], [1.0, 1.0, 3.0], [1.0, 0.008, 0.0]],
        vec![[0, 1, 2]],
        OPAQUE_BLUE,
    );
    let d = decide(
        &b,
        View {
            input_grid: 0.01,
            ..view_c()
        },
    );
    let (p, q) = (id_of(&d, 0, 2), id_of(&d, 1, 3));
    let rows = pairs_of(&d, p, q);
    let interval = |f: u32, x: u32, y: u32, s: usize| {
        let r = d
            .bands
            .iter()
            .find(|r| (r.fragment, r.x, r.y, r.which) == (f, x, y, 0))
            .unwrap();
        assert_eq!(r.status[s], 1, "fragment {f} at ({x}, {y}) sample {s}");
        (
            r.o[s] - r.s_minus[s] - 3.0 * r.r[s],
            r.o[s] + r.s_plus[s] + 3.0 * r.r[s],
        )
    };
    assert!(rows.len() >= 8, "{} pair-samples", rows.len());
    for r in &rows {
        let s = r.sample as usize;
        let (a, b) = (interval(p, r.x, r.y, s), interval(q, r.x, r.y, s));
        assert!(
            a.1 < b.0 || b.1 < a.0,
            "own intervals meet at ({}, {}) sample {s}",
            r.x,
            r.y
        );
        assert_eq!((r.edges, r.label), (7, 0), "({}, {}) sample {s}", r.x, r.y);
    }
}

/// The pair rows of two records stroking the polyline of `two_strokes_of_one_polyline` in view C,
/// red then blue, each of stroke width 0.5 over a background stroke of width 1, after `edit`, read
/// on the red's stroke and the blue's layer `lb`, as `(edges, label, recut)`: `recut` where either
/// own read is `curve_recut`.
fn recut_pair(lb: u8, edit: impl Fn(&mut Build)) -> Vec<(u8, u8, bool)> {
    let pts = open(&[
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 1.0, -1.0],
        [2.0, -1.0, 2.0],
    ]);
    let mut b = Build::new(true);
    for rgba in [OPAQUE_RED, OPAQUE_BLUE] {
        let r = b.stroke(pts.clone(), rgba, 0.5);
        r.paint.background_stroke = Paint::One(BLACK);
        r.paint.background_stroke_width = 1.0;
    }
    edit(&mut b);
    let d = decide(&b, view_c());
    let (p, q) = (id_of(&d, 0, 2), id_of(&d, 1, lb));
    let status = |f: u32, x: u32, y: u32, s: usize| {
        d.bands
            .iter()
            .find(|r| (r.fragment, r.x, r.y, r.which) == (f, x, y, 0))
            .unwrap()
            .status[s]
    };
    pairs_of(&d, p, q)
        .into_iter()
        .map(|r| {
            let s = r.sample as usize;
            let recut = status(p, r.x, r.y, s) == 3 || status(q, r.x, r.y, s) == 3;
            (r.edges, r.label, recut)
        })
        .collect()
}

#[test]
fn recut_reads_of_one_curve_computation_are_an_exact_tie_and_of_two_stay_unable() {
    // §7: "A read that is `curve_recut` (§6) is no bar to an `ExactTie`": two strokes of one
    // record input read at one sample are one computation, an `ExactTie` at every pair-sample,
    // those whose reads are recut included.
    let one = recut_pair(2, |_| {});
    assert!(one.iter().any(|e| e.2), "no recut read");
    assert!(
        one.iter().all(|e| (e.0, e.1) == (1, 4)),
        "{:?}",
        one.iter()
            .filter(|e| (e.0, e.1) != (1, 4))
            .take(4)
            .collect::<Vec<_>>()
    );
    // The red's stroke against the blue's background stroke once its widths are swapped: one
    // body, other width inputs, two computations. A recut read stays unable there.
    let two = recut_pair(0, |b| {
        b.records[1].paint.stroke_width = 1.0;
        b.records[1].paint.background_stroke_width = 0.5;
    });
    assert!(two.iter().any(|e| e.2), "no recut read");
    assert!(
        two.iter().all(|e| e.1 != 4 && (!e.2 || e.1 == 0)),
        "{:?}",
        two.iter()
            .filter(|e| e.1 == 4 || (e.2 && e.1 != 0))
            .take(4)
            .collect::<Vec<_>>()
    );
}

/// The decisions' axis band rows (FORMAT.md `AXBD`), as `(x, y, fragment, against, status)`: a
/// fragment's row on the axis plane it shares with `against`, the read §5 `shared_plane_read@1`
/// takes against it.
fn axis_rows(d: &Decisions) -> Vec<(u32, u32, u32, u32, [u8; 16])> {
    d.axis_bands
        .iter()
        .map(|a| (a.x, a.y, a.fragment, a.against, a.status))
        .collect()
}

#[test]
fn an_unbounded_axis_read_names_its_pair_band_unbounded() {
    // §6: a pair `truth_unable` at a sample is named by the first status among its reads there
    // that states no bound. The scene of
    // `a_path_s_row_on_a_shared_axis_plane_is_read_where_its_own_key_alone_would_order_the_pair`:
    // the stroke's row on `x = 1`, read against the triangle, is unbounded at every pair-sample.
    // That read reaches the facts as an axis row of the stroke against the triangle, status 2,
    // so the pair is named `band_unbounded`, not left unnamed.
    let mut b = Build::new(true);
    b.stroke(
        open(&[[1.0, -2.875, 0.0], [1.0, 2.875, 0.0]]),
        OPAQUE_GREEN,
        1.0,
    );
    b.mesh(
        vec![[1.0, -1.0, -3.0], [1.0, 1.0, 3.0], [1.0, 0.008, 0.0]],
        vec![[0, 1, 2]],
        OPAQUE_BLUE,
    );
    let d = decide(
        &b,
        View {
            input_grid: 0.01,
            ..view_c()
        },
    );
    let (p, q) = (id_of(&d, 0, 2), id_of(&d, 1, 3));
    let axis = axis_rows(&d);
    for r in pairs_of(&d, p, q) {
        let s = r.sample as usize;
        let row = axis
            .iter()
            .find(|a| (a.0, a.1, a.2, a.3) == (r.x, r.y, p, q));
        assert!(
            row.is_some_and(|a| a.4[s] == 2),
            "({}, {}) sample {s}: the stroke's axis read against the triangle {:?}",
            r.x,
            r.y,
            row.map(|a| a.4[s])
        );
    }
}

#[test]
fn a_face_group_s_curve_keys_share_one_face_list() {
    // A face group's face ends and widths are one list per record: every face's curve key holds
    // the record's one copy, not a copy of its own (a group of N faces would hold N lists of N).
    let n = 256;
    let faces: Vec<Vec<[f64; 3]>> = (0..n)
        .map(|i| {
            let x = -6.0 + 12.0 * (i as f64) / (n as f64);
            rect([x, x + 0.03], [-1.0, 1.0], 0.0)
        })
        .collect();
    let mut b = Build::new(true);
    let r = b.face_group(&faces, OPAQUE_BLUE);
    r.paint.stroke_width = 0.25;
    let stroke = arachne_core::xf::Stops::new(vec![OPAQUE_RED; n]);
    let key = stroke.key();
    if let arachne_core::xf::Body::FaceGroup(f) = &mut r.body {
        f.stroke = Some(key);
        f.face_widths = vec![0.25; n];
    }
    b.paints.push(stroke);
    let v = view_c_exact();
    let delta = b.delta(1, v.clone());
    let mut state = arachne_oracle::stream::State::default();
    state.apply(&delta).unwrap();
    let items = arachne_oracle::draw::items(
        &v,
        &state.records,
        &state.order,
        &state.shapes,
        &state.paints,
    );
    // Every face's depth is kept alive while the lists are compared, so no freed list's address is
    // reused by the next face's.
    let depths: Vec<_> = items
        .iter()
        .filter_map(|it| match it {
            arachne_oracle::draw::Item::Path(p) => {
                Some(arachne_oracle::keys::path_depth(&v, p, &[None, None, None]))
            }
            _ => None,
        })
        .collect();
    let mut lists = Vec::new();
    for depth in &depths {
        if let arachne_oracle::keys::Model::Curve { of, .. } = &depth.layers[2] {
            assert_eq!(of.faces.0.len(), n, "face ends");
            lists.push((of.faces.0.as_ptr(), of.faces.1.as_ptr()));
        }
    }
    assert_eq!(lists.len(), n, "a stroke curve key per face");
    assert!(
        lists.iter().all(|l| *l == lists[0]),
        "{} distinct face lists",
        lists.iter().collect::<std::collections::HashSet<_>>().len()
    );
}

#[test]
fn a_recut_read_with_an_unbounded_bound_is_band_unbounded() {
    // §6: where `R` is infinite the oracle states no bound, `curve_recut` included, so no pair is
    // lifted to an `ExactTie` over a read that states no bound. Two pre-projected, pivotless
    // strokes of the polyline of `recut_pair` under zoom 0: their screen geometry and shift stay
    // finite, but every key error off the origin is infinite (`hypot(x, y) / |f z0|`), so every
    // read off the origin, recut or not, has `R = ∞`.
    let pts = open(&[
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 1.0, -1.0],
        [2.0, -1.0, 2.0],
    ]);
    let mut b = Build::new(true);
    for rgba in [OPAQUE_RED, OPAQUE_BLUE] {
        let r = b.pre_projected(pts.clone(), rgba);
        if let Some(tail) = &mut r.pre_projected {
            tail.fill = Paint::None;
            tail.stroke = Paint::One(rgba);
        }
        r.paint.stroke_width = 0.5;
    }
    let mut v = view_c();
    if let Camera::ThreeD(c) = &mut v.camera {
        c.zoom = 0.0;
    }
    let d = decide(&b, v);
    let (p, q) = (id_of(&d, 0, 2), id_of(&d, 1, 2));
    let own: Vec<_> = d
        .bands
        .iter()
        .filter(|r| (r.fragment == p || r.fragment == q) && r.which == 0)
        .collect();
    let recut: Vec<_> = own
        .iter()
        .flat_map(|r| (0..16).filter(|&s| r.status[s] == 3).map(|s| (r.x, r.y, s)))
        .collect();
    assert!(
        recut.is_empty(),
        "{} recut reads with R infinite, first {:?}",
        recut.len(),
        recut.first()
    );
    let ties = pairs_of(&d, p, q).iter().filter(|r| r.label == 4).count();
    assert_eq!(ties, 0, "ExactTie pair-samples over unbounded reads");
}

#[test]
fn two_blends_of_opposite_sign_are_two_computations() {
    // §7, `curve_key@1`: a curve's computation is its record's inputs as recorded. Two records
    // blending their shape toward itself at `t = +2^-60` and `t = −2^-60` are shown alike, and
    // their blend weights `|1 − t|`, `|t|` are equal; their signed `t` is not.
    not_one_computation(false, "blend sign", |b| {
        let t60 = 1.0 / 1_152_921_504_606_846_976.0; // 2^-60
        for (i, t) in [(0, t60), (1, -t60)] {
            let r = &mut b.records[i];
            r.geometry.blend = Some(arachne_core::xf::Blend {
                shape_b: r.geometry.shape_a.unwrap(),
                place_b: arachne_core::xf::IDENTITY,
                t,
            });
        }
    });
}

#[test]
fn two_pre_projected_records_of_other_placements_are_two_computations() {
    // §7, `curve_key@1`: a pre-projected record is shown at its tail's points; the core still reads
    // its own placement. One tail, two `place_a`: two computations.
    not_one_computation(true, "pre-projected placement", |b| {
        b.records[1].geometry.place_a[0][3] = 1.0;
    });
}

#[test]
fn a_point_clearing_its_shift_by_less_than_the_margin_is_either() {
    // §3, point blocks, `point_blocks@3`: within `ε' = ε + η max(|at|, ε)` of an integer the
    // truncation's move is undecided. A 2D view of 4096 x 64 px over 4096 x 64 units about the
    // origin, input grid 1/256: a point at (1/256, 1/2) is shown at x = 2048.00390625, which
    // clears its screen shift by less than `η · 2048`. Its block's pixels are `Either`, never
    // `Yes`.
    let mut b = Build::new(false);
    b.points(vec![[1.0 / 256.0, 0.5, 0.0]], WHITE, 1.0);
    let d = decide(
        &b,
        View {
            pixel_width: 4096,
            pixel_height: 64,
            frame_width: 4096.0,
            frame_height: 64.0,
            input_grid: 1.0 / 256.0,
            ..view_f()
        },
    );
    let ids: Vec<u32> = d
        .fragments
        .iter()
        .filter(|f| f.slot == 0)
        .map(|f| f.id)
        .collect();
    let codes: Vec<u8> = d
        .covers
        .iter()
        .filter(|c| ids.contains(&c.fragment))
        .flat_map(|c| c.cover)
        .filter(|&c| c != 0)
        .collect();
    assert!(
        !codes.is_empty() && codes.iter().all(|&c| c == 2),
        "the point's block: {} covered samples, codes {:?}",
        codes.len(),
        codes.iter().collect::<std::collections::BTreeSet<_>>()
    );
}

#[test]
fn a_flat_path_stroked_on_its_own_fill_keeps_its_exact_bounds() {
    // §6: under a flat camera every key is `−∞` (`Model::NegInf`), exact, with `S− = S+ = 0`. A
    // path filled and stroked: where the stroke is read adjusted on its own fill, both reads are
    // `−∞`, and every read of the stroke states that key, bounded, with no widening: never an
    // infinite `S`, and never unable.
    let mut b = Build::new(false);
    let r = b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 0.0), OPAQUE_BLUE);
    r.paint.stroke = Paint::One(OPAQUE_RED);
    r.paint.stroke_width = 1.0;
    let d = decide(&b, view_f());
    let stroke = id_of(&d, 0, 2);
    let mut read = 0;
    for row in d.bands.iter().filter(|r| r.fragment == stroke) {
        for s in (0..16).filter(|&s| row.status[s] != 0) {
            read += 1;
            assert!(
                row.status[s] == 1
                    && row.o[s] == f64::NEG_INFINITY
                    && row.s_minus[s] == 0.0
                    && row.s_plus[s] == 0.0,
                "({}, {}) which {} sample {s}: status {} o {} S- {} S+ {}",
                row.x,
                row.y,
                row.which,
                row.status[s],
                row.o[s],
                row.s_minus[s],
                row.s_plus[s]
            );
        }
    }
    assert!(read > 0, "no read of the stroke");
}
