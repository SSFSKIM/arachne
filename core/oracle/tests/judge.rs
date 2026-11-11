//! The judge reads a report's order at a sample as an order of the fragments it reports present
//! there (contract §7): each once, none left out, none added; and names `label_plate_model` only
//! for the plate model's own order (`PAIR`'s `plate`), never for every order the tie sets admit,
//! nor where the keys establish no plate-model order.

#[path = "common/judge.rs"]
mod judge;
#[path = "common/scenes.rs"]
mod scenes;

use arachne_oracle::decide::decide_at;
use judge::{Entry, Reported, Verdict, judge};
use scenes::*;

#[test]
fn a_samples_order_is_a_permutation_of_its_present_fragments() {
    // Two opaque squares: `A` on z = 0, `B` nearer on z = 2, both covering pixel (32, 32) whole.
    let mut b = Build::new(true);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_RED);
    b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), OPAQUE_BLUE);
    let d = decide_at(&[b.delta(1, view_c())], 1, None).unwrap();
    let (x, y) = (32, 32);
    let entries: Vec<Entry> = [0, 1]
        .map(|slot| {
            let f = d.fragments.iter().find(|f| f.slot == slot).unwrap();
            let c = d
                .covers
                .iter()
                .find(|c| (c.x, c.y, c.fragment) == (x, y, f.id))
                .unwrap();
            assert_eq!(c.cover, [1; 16]);
            Entry {
                identity: (f.object, f.part, f.layer, f.sub),
                cells: c.cell_hi,
            }
        })
        .into();
    let report = |order: Vec<usize>| Reported {
        x,
        y,
        entries: entries.clone(),
        order: vec![order; 16],
    };
    assert_eq!(judge(&d, &report(vec![1, 0]), 0), Verdict::Agree);
    assert_eq!(
        judge(&d, &report(vec![0, 1]), 0),
        Verdict::Mismatch("mismatch")
    );
    for order in [vec![], vec![1], vec![1, 1, 0], vec![1, 0, 0]] {
        assert_eq!(
            judge(&d, &report(order.clone()), 0),
            Verdict::Mismatch("mismatch"),
            "order {order:?}"
        );
    }
}

#[test]
fn a_lone_fragment_whose_key_states_no_bound_is_unable_not_agreement() {
    // §6: a fragment with no key bounded at a sample is `truth_unable` there, a pair or none. One
    // fixed-orientation square, its centre at z = 10^300: its keys state no bound (`BAND` status
    // 2), and alone it makes no pair. Reported alone, it is not judged.
    let mut b = Build::new(true);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_RED)
        .depth
        .fixed_orientation_centre = Some([0.0, 0.0, 1e300]);
    let d = decide_at(&[b.delta(1, view_c())], 1, None).unwrap();
    let (x, y) = (32, 32);
    let f = &d.fragments[0];
    let c = d
        .covers
        .iter()
        .find(|c| (c.x, c.y, c.fragment) == (x, y, f.id))
        .unwrap();
    assert_eq!(c.cover, [1; 16]);
    let row = d
        .bands
        .iter()
        .find(|r| (r.x, r.y, r.fragment, r.which) == (x, y, f.id, 0))
        .unwrap();
    assert_eq!(row.status, [2; 16]);
    assert!(d.pairs.is_empty());
    let report = Reported {
        x,
        y,
        entries: vec![Entry {
            identity: (f.object, f.part, f.layer, f.sub),
            cells: c.cell_hi,
        }],
        order: vec![vec![0]; 16],
    };
    for s in 0..16 {
        assert_eq!(
            judge(&d, &report, s),
            Verdict::Unable("band_unbounded"),
            "sample {s}"
        );
    }
    // A `curve_recut` read is unable only where a pair reads it: W's `stroke_under_own_fill`
    // twin, judged by `own_layers@1` alone, stays judged (`contract_mutation`).
}

#[test]
fn a_fragment_whose_colour_is_unknown_is_still_judged_for_its_order() {
    // `FORMAT.md`, `COLR` kind 0: a shaded path's colour is not computed, its colour verdict
    // alone is open. Its presence and its order stay judged. Two opaque squares as in the first
    // test, the nearer one shaded in 3D under a shading camera (kind 0).
    let mut b = Build::new(true);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_RED);
    b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), OPAQUE_BLUE)
        .depth
        .shade_in_3d = true;
    let mut view = view_c();
    if let arachne_core::xf::Camera::ThreeD(c) = &mut view.camera {
        c.should_apply_shading = true;
    }
    let d = decide_at(&[b.delta(1, view)], 1, None).unwrap();
    let (x, y) = (32, 32);
    let ids: Vec<u32> = [0, 1]
        .map(|slot| d.fragments.iter().find(|f| f.slot == slot).unwrap().id)
        .into();
    assert_eq!(d.colours[ids[0] as usize].kind, 1);
    assert_eq!(d.colours[ids[1] as usize].kind, 0);
    let entries: Vec<Entry> = ids
        .iter()
        .map(|&id| {
            let f = &d.fragments[id as usize];
            let c = d
                .covers
                .iter()
                .find(|c| (c.x, c.y, c.fragment) == (x, y, id))
                .unwrap();
            Entry {
                identity: (f.object, f.part, f.layer, f.sub),
                cells: c.cell_hi,
            }
        })
        .collect();
    let report = |order: Vec<usize>| Reported {
        x,
        y,
        entries: entries.clone(),
        order: vec![order; 16],
    };
    for s in 0..16 {
        assert_eq!(
            judge(&d, &report(vec![1, 0]), s),
            Verdict::Agree,
            "sample {s}"
        );
        assert_eq!(
            judge(&d, &report(vec![0, 1]), s),
            Verdict::Mismatch("mismatch"),
            "sample {s}"
        );
    }
}

/// W18's label and a line at `z`, the label painted first or last, and the verdicts of the label
/// first and the line first at the samples of pixel (32, 32), with the pair's `edges` and `plate`
/// there.
fn label_and_line(z: f64, label_painted_first: bool) -> Vec<(Verdict, Verdict, u8, u8)> {
    let mut b = Build::new(true);
    let line = |b: &mut Build| {
        b.stroke(
            open(&[[-3.1875, 0.0, z], [3.1875, 0.0, z]]),
            OPAQUE_GREEN,
            0.5,
        );
    };
    if label_painted_first {
        b.label([0.0; 3]);
        line(&mut b);
    } else {
        line(&mut b);
        b.label([0.0; 3]);
    }
    let d = decide_at(&[b.delta(1, view_c())], 1, None).unwrap();
    let (x, y) = (32, 32);
    let ids: Vec<u32> = [0, 1]
        .map(|slot| d.fragments.iter().find(|f| f.slot == slot).unwrap().id)
        .into();
    let entries: Vec<Entry> = ids
        .iter()
        .map(|&id| {
            let f = &d.fragments[id as usize];
            let c = d
                .covers
                .iter()
                .find(|c| (c.x, c.y, c.fragment) == (x, y, id))
                .unwrap();
            assert_eq!(c.cover, [1; 16]);
            Entry {
                identity: (f.object, f.part, f.layer, f.sub),
                cells: c.cell_hi,
            }
        })
        .collect();
    let report = |order: Vec<usize>| Reported {
        x,
        y,
        entries: entries.clone(),
        order: vec![order; 16],
    };
    // Entries in slot order: the label's is the first where it is painted first.
    let (label_first, line_first) = if label_painted_first {
        (vec![0, 1], vec![1, 0])
    } else {
        (vec![1, 0], vec![0, 1])
    };
    (0..16)
        .map(|s| {
            let p = d
                .pairs
                .iter()
                .find(|p| (p.x, p.y, p.sample as usize) == (x, y, s))
                .unwrap();
            // `plate` in the label's terms: its bits with the label as `a`.
            let plate = if label_painted_first {
                p.plate
            } else {
                (p.plate & 1) | (p.plate & 2) << 1 | (p.plate & 4) >> 1
            };
            (
                judge(&d, &report(label_first.clone()), s),
                judge(&d, &report(line_first.clone()), s),
                p.edges,
                plate,
            )
        })
        .collect()
}

#[test]
fn only_the_plate_models_own_order_is_named_label_plate_model() {
    // W18: the line nearer, by far more than any bound: the plate model draws it over the label.
    for (label_first, line_first, edges, plate) in label_and_line(2.0, true) {
        assert_eq!(label_first, Verdict::Agree);
        assert_eq!(line_first, Verdict::Mismatch("label_plate_model"));
        assert_eq!((edges, plate), (4, 4));
    }
    // The line 2^-30 behind the label: inside the input snap, so `PAIR` admits the line first too,
    // but the plate model's keys put the label in front. The line first is no plate-model order.
    for (label_first, line_first, edges, plate) in label_and_line(-1.0 / 1_073_741_824.0, true) {
        assert_eq!(label_first, Verdict::Agree);
        assert_eq!(edges & 4, 4, "PAIR admits the line in front");
        assert_eq!(plate, 2, "the plate model puts the label in front");
        assert_eq!(line_first, Verdict::Mismatch("mismatch"));
    }
}

#[test]
fn a_tied_line_painted_after_the_label_is_drawn_over_it_by_the_plate_model() {
    // The line on the label's plane: both keys are exactly −1 at every sample, a tie the plate
    // model orders by paint order (§7). Painted after the label, the line is drawn over it.
    for (label_first, line_first, _, plate) in label_and_line(0.0, true) {
        assert_eq!(label_first, Verdict::Agree);
        assert_eq!(line_first, Verdict::Mismatch("label_plate_model"));
        assert_eq!(plate, 1, "a tie");
    }
}

#[test]
fn a_tied_line_painted_before_the_label_is_drawn_under_it_by_the_plate_model() {
    // As above, the line painted first: the plate model draws the label over it, and the line
    // first is no plate-model order.
    for (label_first, line_first, _, plate) in label_and_line(0.0, false) {
        assert_eq!(label_first, Verdict::Agree);
        assert_eq!(line_first, Verdict::Mismatch("mismatch"));
        assert_eq!(plate, 1, "a tie");
    }
}

#[test]
fn a_plate_relation_the_keys_do_not_establish_names_no_deviation() {
    // The line 2^-45 nearer than the label: its key is a few ulps nearer, well inside the keys'
    // ±3R. The plate model's order is neither a tie nor a separation there, so no order is named.
    for (label_first, line_first, edges, plate) in label_and_line(1.0 / 35_184_372_088_832.0, true)
    {
        assert_eq!(edges, 7);
        assert_eq!(label_first, Verdict::Agree);
        assert_eq!(line_first, Verdict::Mismatch("mismatch"));
        assert_eq!(plate, 0, "undetermined");
    }
}
