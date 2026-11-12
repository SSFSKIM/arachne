//! `label_placement@2`'s sheets (contract §10, Terms): "A sheet is a connected part of one surface
//! object, read from its fills on screen", joined by winding and two shared corners, transitively.
//! A fill the frame does not reach is still one of the object's fills: it joins two fills it
//! shares corners with, though it holds no point.

#[path = "common/scenes.rs"]
mod scenes;

use arachne_oracle::decide::decide_at_wire;
use arachne_oracle::out::Decisions;
use arachne_oracle::source;
use scenes::*;

const CUBE: &str = "arachne.mobject.three_d.three_dimensions.Cube";
const POLYGON: &str = "arachne.mobject.geometry.polygram.Polygon";
const PAIR_CAUSE_DISCONNECTED: u8 = 3;

/// View C: two arms of one surface object (slots 0 and 1, `[−3, −1]` and `[1, 3]` by `[0, 9]` at
/// z = 2), with `bridge` drawn (slot 2) where given, then a label (slot 3) whose glyph spans
/// `[−4, 4]` by `[0, 1]` at z = 0 about the origin. Each arm's `LABL` pair causes, and whether the
/// bridge has a fragment.
fn arms(bridge: Option<Vec<[f64; 3]>>) -> ([Vec<u8>; 2], bool) {
    let mut s = Build::new(true);
    s.fill(rect([-3.0, -1.0], [0.0, 9.0], 2.0), OPAQUE_BLUE);
    s.fill(rect([1.0, 3.0], [0.0, 9.0], 2.0), OPAQUE_BLUE);
    let mut rows = vec![row(0, 0, &[&[(42, CUBE)]]), row(1, 0, &[&[(42, CUBE)]])];
    let label = if let Some(b) = bridge {
        s.fill(b, OPAQUE_BLUE);
        rows.push(row(2, 0, &[&[(42, CUBE)]]));
        3
    } else {
        2
    };
    s.glyph(rect([-4.0, 4.0], [0.0, 1.0], 0.0), WHITE, [0.0; 3]);
    rows.push(row(label, 1, &[&[(900, POLYGON)]]));
    let bytes = file15(&[delta15(&s.delta(1, view_c()), &section(&rows, None))]);
    let (_, wire) = source::read_file_bytes(&bytes).unwrap();
    let d: Decisions = decide_at_wire(&wire, 1, None).unwrap();
    assert_eq!(d.header.status, 0, "{:?}", d.refused);
    let causes = [0, 1].map(|slot| {
        let f = d
            .fragments
            .iter()
            .find(|f| f.slot == slot && f.layer == 1)
            .unwrap_or_else(|| panic!("no fill of slot {slot}"))
            .id;
        let c: Vec<u8> = d
            .labels
            .iter()
            .filter(|r| r.fragment == f)
            .map(|r| r.pair_cause)
            .collect();
        assert!(!c.is_empty(), "no LABL rows for slot {slot}");
        c
    });
    (
        causes,
        d.fragments.iter().any(|f| f.slot == 2 && label == 3),
    )
}

#[test]
fn a_fill_outside_the_frame_joins_the_sheets_it_shares_corners_with() {
    // Apart, the arms are two sheets of one object and winding that both count a point with the
    // label: `disconnected`, both pairs open.
    let disconnected = |c: &Vec<u8>| c.iter().filter(|&&c| c == PAIR_CAUSE_DISCONNECTED).count();
    let (apart, _) = arms(None);
    for c in &apart {
        assert_eq!(disconnected(c), c.len(), "of {} rows", c.len());
    }
    // A bridge across `[−3, 3]` by `[9, 11]`, above the frame, wound as they are and sharing the
    // arms' top corners, joins them into one sheet: not disconnected.
    let bridge = polygon(&[
        [-3.0, 9.0, 2.0],
        [-1.0, 9.0, 2.0],
        [1.0, 9.0, 2.0],
        [3.0, 9.0, 2.0],
        [3.0, 11.0, 2.0],
        [-3.0, 11.0, 2.0],
    ]);
    let (joined, drawn) = arms(Some(bridge));
    assert!(!drawn, "the bridge reaches the frame");
    for c in &joined {
        assert_eq!(disconnected(c), 0, "of {} rows", c.len());
    }
}

#[test]
fn an_off_frame_connector_holding_the_centre_decides_the_sheet() {
    // §10 rule 5: where a fill of the sheet holds the label's centre, the centre comparison
    // decides; a fill the frame does not reach is still one of the sheet's fills. The arms of
    // `arms` at z = 2, joined above the frame by a bridge sharing their top corners and tilting
    // back to z = −2 at y = 12; the label's glyph spans `[−4, 4]` by `[0, 1]` at z = 0 over both
    // arms, kept facing the camera about (0, 10.8, 0), off the frame, where the bridge stands
    // behind it (z ≈ −1.4). The bridge holds the centre and stands behind the label, so the
    // arms, nearer at every point the label overlaps, do not anchor it.
    let mut s = Build::new(true);
    s.fill(rect([-3.0, -1.0], [0.0, 9.0], 2.0), OPAQUE_BLUE);
    s.fill(rect([1.0, 3.0], [0.0, 9.0], 2.0), OPAQUE_BLUE);
    s.fill(
        polygon(&[
            [-3.0, 9.0, 2.0],
            [-1.0, 9.0, 2.0],
            [1.0, 9.0, 2.0],
            [3.0, 9.0, 2.0],
            [3.0, 12.0, -2.0],
            [-3.0, 12.0, -2.0],
        ]),
        OPAQUE_BLUE,
    );
    s.glyph(rect([-4.0, 4.0], [0.0, 1.0], 0.0), WHITE, [0.0, 10.8, 0.0]);
    let rows = vec![
        row(0, 0, &[&[(42, CUBE)]]),
        row(1, 0, &[&[(42, CUBE)]]),
        row(2, 0, &[&[(42, CUBE)]]),
        row(3, 1, &[&[(900, POLYGON)]]),
    ];
    let bytes = file15(&[delta15(&s.delta(1, view_c()), &section(&rows, None))]);
    let (_, wire) = source::read_file_bytes(&bytes).unwrap();
    let d: Decisions = decide_at_wire(&wire, 1, None).unwrap();
    assert_eq!(d.header.status, 0, "{:?}", d.refused);
    assert!(
        !d.fragments.iter().any(|f| f.slot == 2),
        "premise: the bridge reaches the frame"
    );
    for slot in [0, 1] {
        let f = d
            .fragments
            .iter()
            .find(|f| f.slot == slot && f.layer == 1)
            .unwrap_or_else(|| panic!("no fill of slot {slot}"))
            .id;
        let labl: Vec<_> = d.labels.iter().filter(|r| r.fragment == f).collect();
        assert!(!labl.is_empty(), "no LABL rows for slot {slot}");
        assert!(
            labl.iter().all(|r| r.pair_cause != PAIR_CAUSE_DISCONNECTED),
            "premise: slot {slot} disconnected from the bridge"
        );
        let anchored = labl
            .iter()
            .filter(|r| r.role == arachne_oracle::out::ROLE_ANCHOR)
            .count();
        assert_eq!(
            anchored,
            0,
            "slot {slot}: {anchored} of {} rows anchor the label by its points",
            labl.len()
        );
    }
}
