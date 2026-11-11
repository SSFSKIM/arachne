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
fn an_unbounded_fragment_is_unable_at_every_sample_whether_reported_or_not() {
    // FORMAT, `flags` bit 0: an unbounded fragment "is `truth_unable` at every sample of the
    // window, never absent"; it has no rows, so a reader treats it as present-unknown at every
    // pixel (§6). Two scenes make one: a square placed at the eye's depth, and a point block just
    // short of it. Reported with nothing, or with it, every sample of the window is unjudged.
    let mut square = Build::new(true);
    square
        .fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_RED)
        .geometry
        .place_a[2][3] = 19.9999999;
    let mut point = Build::new(true);
    point.points(vec![[0.0, 0.0, 20.0 - 1e-7]], WHITE, 1.0);
    for b in [square, point] {
        let d = decide_at(&[b.delta(1, view_c())], 1, None).unwrap();
        let f = &d.fragments[0];
        assert_eq!(
            f.flags & arachne_oracle::out::FLAG_UNBOUNDED,
            1,
            "{:?}",
            d.fragments
        );
        assert!(d.covers.is_empty());
        let identity = (f.object, f.part, f.layer, f.sub);
        for (x, y) in [(32, 32), (0, 0), (63, 63)] {
            let nothing = Reported {
                x,
                y,
                entries: vec![],
                order: vec![vec![]; 16],
            };
            let it = Reported {
                x,
                y,
                entries: vec![Entry {
                    identity,
                    cells: [255; 16],
                }],
                order: vec![vec![0]; 16],
            };
            for s in 0..16 {
                for report in [&nothing, &it] {
                    assert_eq!(
                        judge(&d, report, s),
                        Verdict::Unable("band_unbounded"),
                        "({x}, {y}) sample {s}, {} entries",
                        report.entries.len()
                    );
                }
            }
        }
    }
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

/// The report at `(32, 32)` of the fragments of `slots`, each with cells `cells` (its own cell
/// interval's high end where `None`), in that order front first at every sample.
fn report_of(
    d: &arachne_oracle::out::Decisions,
    slots: &[u32],
    cells: Option<[u8; 16]>,
) -> Reported {
    let entries: Vec<Entry> = slots
        .iter()
        .map(|&slot| {
            let f = d.fragments.iter().find(|f| f.slot == slot).unwrap();
            let c = d
                .covers
                .iter()
                .find(|c| (c.x, c.y, c.fragment) == (32, 32, f.id))
                .unwrap();
            Entry {
                identity: (f.object, f.part, f.layer, f.sub),
                cells: cells.unwrap_or(c.cell_hi),
            }
        })
        .collect();
    Reported {
        x: 32,
        y: 32,
        order: vec![(0..entries.len()).collect(); 16],
        entries,
    }
}

#[test]
fn an_either_fragment_left_out_of_the_report_is_absent_from_its_pairs() {
    // §7: the pairs present at a sample are each `Yes`, or `Either` and present in the core's
    // report. Two squares, `A` on z = 0 and `B` nearer on z = 2, and a thin line on z = 5 whose
    // body may or may not reach sample 0 of (32, 32): `Either` there, its reads `curve_recut`. A
    // report leaving the line out and putting `A` in front is judged on `A` and `B` alone.
    let mut b = Build::new(true);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_RED);
    b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), OPAQUE_BLUE);
    let sy = (0.125 - 0.25 / 510.0) / (4.0 * (20.0 / 15.0));
    b.stroke(
        open(&[[-3.0, sy, 5.0], [3.0, sy, 5.0]]),
        OPAQUE_GREEN,
        0.0625,
    );
    let d = decide_at(&[b.delta(1, view_c())], 1, None).unwrap();
    let line = d.fragments.iter().find(|f| f.slot == 2).unwrap().id;
    let at = d
        .covers
        .iter()
        .find(|c| (c.x, c.y, c.fragment) == (32, 32, line))
        .unwrap();
    assert_eq!(at.cover[0], 2, "the line is not Either at sample 0");
    assert!(
        d.pairs.iter().any(|p| (p.x, p.y, p.sample) == (32, 32, 0)
            && (p.a == line || p.b == line)
            && p.label == arachne_oracle::out::LABEL_UNABLE),
        "no unable pair of the line at sample 0"
    );
    let reversed = report_of(&d, &[0, 1], None);
    assert_eq!(judge(&d, &reversed, 0), Verdict::Mismatch("mismatch"));
    let mut right = reversed.clone();
    right.order = vec![vec![1, 0]; 16];
    assert_eq!(judge(&d, &right, 0), Verdict::Agree);
}

#[test]
fn an_either_fragment_left_out_of_the_report_opens_no_label_pair() {
    // As above for the label rule: a label about the origin behind `B`, a rectangle on z = 2, and
    // a square with no source on z = 0 whose edge may or may not reach sample 0 of (32, 32), an
    // open class there. A report leaving it out and putting the label in front of `B` is judged
    // on the label and `B` alone: not open.
    let mut b = Build::new(true);
    b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), OPAQUE_BLUE);
    b.label([0.0; 3]);
    b.fill(rect([0.062475, 2.0], [-3.0, 3.0], 0.0), OPAQUE_RED);
    let mut st = arachne_oracle::stream::State::new();
    let mut v = view_c();
    v.input_grid = 0.01;
    st.apply(&b.delta(1, v)).unwrap();
    st.rules15 = true;
    let rectangle = "arachne.mobject.geometry.polygram.Rectangle".to_string();
    st.meta.insert(
        0,
        arachne_oracle::stream::Meta {
            source: vec![vec![(0, rectangle)]],
            facing_group: 0,
        },
    );
    st.meta.insert(
        1,
        arachne_oracle::stream::Meta {
            source: vec![],
            facing_group: 1,
        },
    );
    let d = arachne_oracle::decide::decide(&st, (1, 5), 1, None);
    let square = d.fragments.iter().find(|f| f.slot == 2).unwrap().id;
    let at = d
        .covers
        .iter()
        .find(|c| (c.x, c.y, c.fragment) == (32, 32, square))
        .unwrap();
    assert_eq!(at.cover[0], 2, "the square is not Either at sample 0");
    let before = report_of(&d, &[1, 0], Some([255; 16]));
    let v = judge(&d, &before, 0);
    assert!(matches!(v, Verdict::Mismatch(_)), "{v:?}");
}

#[test]
fn a_label_among_many_kind_unknown_faces_is_judged_however_many_they_are() {
    // §10: in a stream read as XF 1.4 a face is kind unknown, read both as a surface that anchors
    // the label and as one that does not, and both orders are admitted; no number of such faces
    // is a limit. Thirteen face squares, each its own record, on z = 1 to 2.2 across the label
    // about the origin: each nearer square may anchor it, so it stands in front of all of them or
    // behind any one. Every such order agrees; the squares out of depth order do not.
    let mut b = Build::new(true);
    for k in 0..13 {
        b.face_group(
            &[rect([-3.0, 3.0], [-3.0, 3.0], 1.0 + 0.1 * k as f64)],
            OPAQUE_BLUE,
        );
    }
    b.label([0.0; 3]);
    // Records of one paint upload it once.
    let mut delta = b.delta(1, view_c());
    let mut seen = std::collections::HashSet::new();
    delta.upload_paints.retain(|p| seen.insert(p.key()));
    let d = decide_at(&[delta], 1, None).unwrap();
    assert_eq!(d.header.status, 0, "{:?}", d.refused);
    let unknown = d
        .labels
        .iter()
        .filter(|r| {
            (r.x, r.y, r.sample) == (32, 32, 0) && r.role == arachne_oracle::out::ROLE_KIND_UNKNOWN
        })
        .count();
    assert_eq!(unknown, 13);
    let rep = report_of(&d, &(0..14).collect::<Vec<u32>>(), None);
    // The squares nearest first, then the label at `place`.
    let with_label_at = |place: usize| {
        let mut order: Vec<usize> = (0..13).rev().collect();
        order.insert(place, 13);
        Reported {
            order: vec![order; 16],
            ..rep.clone()
        }
    };
    for place in [0, 1, 7, 13] {
        assert_eq!(
            judge(&d, &with_label_at(place), 0),
            Verdict::Agree,
            "place {place}"
        );
    }
    let mut out_of_order = with_label_at(0);
    out_of_order.order = vec![(0..14).collect(); 16];
    let v = judge(&d, &out_of_order, 0);
    assert!(matches!(v, Verdict::Mismatch(_)), "{v:?}");
}
