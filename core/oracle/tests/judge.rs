//! The judge reads a report's order at a sample as an order of the fragments it reports present
//! there (contract §7): each once, none left out, none added.

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
