//! `label_placement@2` makes a pair of a label and a sheet only where the sheet holds a counted
//! order point (contract §10, Terms: "A sliver of either between order points holds none"). An
//! ungrouped record kept facing the camera is an open sheet of its own, and is no exception: a
//! sliver of it between the label's order points makes no pair, so it opens no sample.

#[path = "common/scenes.rs"]
mod scenes;

use arachne_oracle::decide::decide_at_wire;
use arachne_oracle::out::{Decisions, ROLE_FRONT, ROLE_OPEN};
use arachne_oracle::source;
use scenes::*;

const POLYGON: &str = "arachne.mobject.geometry.polygram.Polygon";
const PAIR_CAUSE_LABEL: u8 = 5;

/// View C: W5's label `L` about the origin (slot 1, facing group 1), and in slot 0 an opaque
/// glyph `U` kept facing the camera with no facing group, over the screen columns `[x0, x1]` of
/// pixel `(32, 32)`'s row, its whole height. The `LABL` rows of `U`'s fill at that pixel.
fn ungrouped_rows(x0: f64, x1: f64) -> Vec<(u8, u8, u8)> {
    // View C at z = 0: x = (px − 32) / 4, y = −(py − 32) / 4.
    let (a, b) = ((x0 - 32.0) / 4.0, (x1 - 32.0) / 4.0);
    let mut s = Build::new(true);
    s.glyph(
        rect([a, b], [-0.25, 0.0], 0.0),
        OPAQUE_BLUE,
        [(a + b) / 2.0, -0.125, 0.0],
    );
    s.label([0.0; 3]);
    let rows = vec![
        row(0, 0, &[&[(1, POLYGON)]]),
        row(1, 1, &[&[(900, POLYGON)]]),
    ];
    let bytes = file15(&[delta15(&s.delta(1, view_c()), &section(&rows, None))]);
    let (_, wire) = source::read_file_bytes(&bytes).unwrap();
    let d: Decisions = decide_at_wire(&wire, 1, None).unwrap();
    assert_eq!(d.header.status, 0, "{:?}", d.refused);
    let u = d
        .fragments
        .iter()
        .find(|f| f.slot == 0 && f.layer == 1)
        .unwrap()
        .id;
    assert!(
        d.covers
            .iter()
            .any(|c| (c.x, c.y) == (32, 32) && c.fragment == u && c.cover.iter().any(|v| *v != 0)),
        "U covers none of (32, 32)'s cells"
    );
    d.labels
        .iter()
        .filter(|r| (r.x, r.y) == (32, 32) && r.fragment == u)
        .map(|r| (r.sample, r.role, r.pair_cause))
        .collect()
}

#[test]
fn an_ungrouped_sliver_between_order_points_opens_no_sample() {
    // [32.05, 32.06] lies between the order points' columns 32.031 and 32.094: U holds none.
    let rows = ungrouped_rows(32.05, 32.06);
    assert!(!rows.is_empty(), "no LABL rows for U");
    assert!(
        rows.iter()
            .all(|&(_, role, pc)| role == ROLE_FRONT && pc == 0),
        "{rows:?}"
    );
}

#[test]
fn an_ungrouped_sheet_holding_an_order_point_is_open() {
    // [32.0, 32.12] holds the order points of the first two columns: the pair is made, and open.
    let rows = ungrouped_rows(32.0, 32.12);
    assert!(!rows.is_empty(), "no LABL rows for U");
    assert!(
        rows.iter()
            .all(|&(_, role, pc)| role == ROLE_OPEN && pc == PAIR_CAUSE_LABEL),
        "{rows:?}"
    );
}
