//! `label_placement@2` qualifies each pair once over the whole frame, before any tile (contract
//! §10): a window's `LABL` rows are the whole frame's rows at its pixels.

#[path = "common/scenes.rs"]
mod scenes;

use arachne_oracle::decide::decide_at;
use arachne_oracle::out::{Decisions, NONE_U32};
use scenes::*;

/// The `LABL` rows at the window's pixels, each fragment named by its identity, and each
/// fragment's label by identity.
#[allow(clippy::type_complexity)]
fn label_facts(
    d: &Decisions,
    w: [u32; 4],
) -> (
    Vec<(u32, u32, u32, (u32, u32, u8, u32), u8, u8, u8, u8)>,
    Vec<((u32, u32, u8, u32), u32)>,
) {
    let ident = |f: u32| {
        let f = &d.fragments[f as usize];
        (f.object, f.part, f.layer, f.sub)
    };
    let inside = |x: u32, y: u32| x >= w[0] && x < w[2] && y >= w[1] && y < w[3];
    let rows = d
        .labels
        .iter()
        .filter(|r| inside(r.x, r.y))
        .map(|r| {
            (
                r.x,
                r.y,
                r.label,
                ident(r.fragment),
                r.sample,
                r.role,
                r.pair_cause,
                r.class_cause,
            )
        })
        .collect();
    let mut labels: Vec<_> = d
        .fragments
        .iter()
        .filter(|f| {
            f.label != NONE_U32
                && d.covers
                    .iter()
                    .any(|c| c.fragment == f.id && inside(c.x, c.y))
        })
        .map(|f| (ident(f.id), f.label))
        .collect();
    labels.sort();
    (rows, labels)
}

#[test]
fn a_window_reads_the_whole_frames_qualification() {
    // W22: the surface straddles the label's depth; the settled-farther samples at (37, 32) keep
    // it from qualifying. A window holding only (35, 32), where it is nearer, must not qualify it.
    let mut b = Build::new(true);
    b.fill(
        polygon(&[
            [0.5, -1.5, 1.2],
            [2.5, -1.5, -2.0],
            [2.5, 1.5, -2.0],
            [0.5, 1.5, 1.2],
        ]),
        OPAQUE_BLUE,
    );
    b.label([0.0; 3]);
    let deltas = [b.delta(1, view_c())];
    let whole = decide_at(&deltas, 1, None).unwrap();
    for w in [[35, 32, 36, 33], [30, 28, 38, 36], [0, 0, 64, 64]] {
        let tile = decide_at(&deltas, 1, Some(w)).unwrap();
        let (rows, labels) = label_facts(&tile, w);
        assert!(!rows.is_empty(), "no label rows in {w:?}");
        assert_eq!((rows, labels), label_facts(&whole, w), "window {w:?}");
    }
}

#[test]
fn a_window_s_label_rows_name_the_fragments_present_where_they_stand() {
    // As above, with a point block under the label: each pixel of a block, under each truncation
    // candidate, is a fragment of its own, and candidates of one offset share one `sub` at other
    // pixels. A window's `LABL` row names the fragment present at its own pixel and sample, as the
    // whole frame's does.
    let mut b = Build::new(true);
    b.points(vec![[0.0; 3]], OPAQUE_BLUE, 2.0);
    b.label([0.0; 3]);
    let deltas = [b.delta(1, view_c())];
    let present = |d: &Decisions, w: [u32; 4]| -> (usize, usize) {
        let inside = |x: u32, y: u32| x >= w[0] && x < w[2] && y >= w[1] && y < w[3];
        let rows: Vec<_> = d.labels.iter().filter(|r| inside(r.x, r.y)).collect();
        let held = rows
            .iter()
            .filter(|r| {
                d.covers.iter().any(|c| {
                    (c.x, c.y, c.fragment) == (r.x, r.y, r.fragment)
                        && c.cover[r.sample as usize] != 0
                })
            })
            .count();
        (held, rows.len())
    };
    let w = [31, 31, 34, 34];
    let whole = decide_at(&deltas, 1, None).unwrap();
    let (held, rows) = present(&whole, w);
    assert!(rows > 0, "no label rows in {w:?}");
    assert_eq!(held, rows, "whole frame");
    let tile = decide_at(&deltas, 1, Some(w)).unwrap();
    assert_eq!(present(&tile, w), (rows, rows), "window {w:?}");
    assert_eq!(
        label_facts(&tile, w),
        label_facts(&whole, w),
        "window {w:?}"
    );
}
