//! The judge reads a report's order at a sample as an order of the fragments it reports present
//! there (contract §7): each once, none left out, none added; and names `label_plate_model` only
//! for the plate model's own order (`PAIR`'s `plate`), never for every order the tie sets admit.

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

/// W18's label and a line at `z`, and the verdicts of the label first and the line first at the
/// samples of pixel (32, 32), with the pair's `edges` and `plate` there.
fn label_and_line(z: f64) -> Vec<(Verdict, Verdict, u8, u8)> {
    let mut b = Build::new(true);
    b.label([0.0; 3]);
    b.stroke(
        open(&[[-3.1875, 0.0, z], [3.1875, 0.0, z]]),
        OPAQUE_GREEN,
        0.5,
    );
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
    (0..16)
        .map(|s| {
            let p = d
                .pairs
                .iter()
                .find(|p| (p.x, p.y, p.sample as usize) == (x, y, s))
                .unwrap();
            (
                judge(&d, &report(vec![0, 1]), s),
                judge(&d, &report(vec![1, 0]), s),
                p.edges,
                p.plate,
            )
        })
        .collect()
}

#[test]
fn only_the_plate_models_own_order_is_named_label_plate_model() {
    // W18: the line nearer, by far more than any bound: the plate model draws it over the label.
    for (label_first, line_first, edges, plate) in label_and_line(2.0) {
        assert_eq!(label_first, Verdict::Agree);
        assert_eq!(line_first, Verdict::Mismatch("label_plate_model"));
        assert_eq!((edges, plate), (4, 4));
    }
    // The line 2^-30 behind the label: inside the input snap, so `PAIR` admits the line first too,
    // but the plate model's keys put the label in front. The line first is no plate-model order.
    for (label_first, line_first, edges, plate) in label_and_line(-1.0 / 1_073_741_824.0) {
        assert_eq!(label_first, Verdict::Agree);
        assert_eq!(edges & 4, 4, "PAIR admits the line in front");
        assert_eq!(plate, 2, "the plate model puts the label in front");
        assert_eq!(line_first, Verdict::Mismatch("mismatch"));
    }
}
