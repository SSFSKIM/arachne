//! The facts the decisions file carries for the comparator beside each pair's edges (`FORMAT.md`):
//! each strong edge's provenance, the edges with the records that paint nothing offered, and the
//! zero-alpha mesh triangles. Each is shown both ways.

#[path = "common/scenes.rs"]
mod scenes;

use arachne_oracle::decide::decide_at;
use arachne_oracle::out::{Decisions, FLAG_ZERO_ALPHA, PairRow};
use scenes::*;

fn decide(b: &Build, view: arachne_core::xf::View) -> Decisions {
    decide_at(&[b.delta(1, view)], 1, None).unwrap()
}

fn id_of(d: &Decisions, slot: u32, layer: u8) -> u32 {
    d.fragments
        .iter()
        .find(|f| f.slot == slot && f.layer == layer)
        .unwrap_or_else(|| panic!("no fragment of slot {slot} layer {layer}"))
        .id
}

fn pairs_at(d: &Decisions, x: u32, y: u32, a: u32, b: u32) -> Vec<&PairRow> {
    let rows: Vec<_> = d
        .pairs
        .iter()
        .filter(|p| (p.x, p.y, p.a, p.b) == (x, y, a.min(b), a.max(b)))
        .collect();
    assert_eq!(rows.len(), 16, "pair rows of {a}, {b} at ({x}, {y})");
    rows
}

const FILL: u8 = 1;
const STROKE: u8 = 2;
const MESH: u8 = 3;
const ROW_GAP: f64 = 1.0 / 1_048_576.0;

/// `PAIR`'s provenance codes, `a` in front's then `b` in front's.
fn via(p: &PairRow) -> (u8, u8) {
    (p.provenance & 3, p.provenance >> 2)
}

#[test]
fn each_strong_edge_names_how_it_is_read() {
    // A flat square on z = 0 and a line lying in it: the line has no plane of its own and may
    // share the square's flat identity, so every edge is admitted, its strong ones through that
    // identity (`one_plane@1`): code 2.
    let mut b = Build::new(true);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_BLUE);
    b.stroke(open(&[[-3.1875, 0.0, 0.0], [3.1875, 0.0, 0.0]]), WHITE, 0.5);
    let d = decide(&b, view_c());
    for p in pairs_at(&d, 32, 31, id_of(&d, 0, FILL), id_of(&d, 1, STROKE)) {
        assert_eq!((p.label, p.edges, via(p)), (6, 7, (2, 2)), "{p:?}");
    }
    // Two flat squares a hair apart: they may share a plane, and the shared scenario reads each
    // one's row there (`shared_plane_read@1`): code 1.
    let mut b = Build::new(true);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_BLUE);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], ROW_GAP), OPAQUE_RED);
    let d = decide(&b, view_c());
    for p in pairs_at(&d, 32, 31, id_of(&d, 0, FILL), id_of(&d, 1, FILL)) {
        assert_eq!((p.label, p.edges, via(p)), (3, 7, (1, 1)), "{p:?}");
    }
    // A flat triangle and a flat square a hair above it: they may share a plane, where the
    // square reads its row and the triangle, which has none, its own key (§5): still a row read.
    let mut b = Build::new(true);
    b.mesh(
        vec![[-6.0, -6.0, 0.0], [6.0, -6.0, 0.0], [0.0, 6.0, 0.0]],
        vec![[0, 1, 2]],
        OPAQUE_RED,
    );
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], ROW_GAP), OPAQUE_BLUE);
    let d = decide(&b, view_c());
    for p in pairs_at(&d, 32, 31, id_of(&d, 0, MESH), id_of(&d, 1, FILL)) {
        assert_eq!((p.edges, via(p)), (7, (1, 1)), "{p:?}");
    }
    // W4's surfaces: no shared plane, so the strong edge is read from the own keys alone.
    let mut b = Build::new(true);
    b.mesh(
        vec![[-6.0, -6.0, 0.0], [6.0, -6.0, 0.0], [0.0, 6.0, 0.0]],
        vec![[0, 1, 2]],
        OPAQUE_RED,
    );
    b.fill(
        polygon(&[
            [-6.0, -6.0, -3.0],
            [6.0, -6.0, 3.0],
            [6.0, 6.0, 3.0],
            [-6.0, 6.0, -3.0],
        ]),
        OPAQUE_BLUE,
    );
    let d = decide(&b, view_c());
    let (t, q) = (id_of(&d, 0, MESH), id_of(&d, 1, FILL));
    for p in pairs_at(&d, 36, 32, t, q) {
        assert_eq!((p.label, p.edges, via(p)), (2, 4, (0, 0)), "{p:?}");
    }
}

/// W17's synthetic form: two half-transparent lines on `z = 0`, then a square that paints
/// nothing, there or not.
fn registry_scene(with_polygon: bool) -> Build {
    let mut b = Build::new(true);
    b.stroke(
        open(&[[-4.25, 0.0, 0.0], [2.125, 0.0, 0.0]]),
        [1.0, 0.0, 0.0, 0.5],
        0.5,
    );
    b.stroke(
        open(&[[-2.125, 0.0, 0.0], [4.25, 0.0, 0.0]]),
        [0.0, 0.0, 1.0, 0.5],
        0.5,
    );
    if with_polygon {
        let r = b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), [0.0, 1.0, 0.0, 0.0]);
        r.paint.stroke = arachne_core::xf::Paint::One([0.0, 1.0, 0.0, 0.0]);
        r.paint.stroke_width = 0.5;
    }
    b
}

#[test]
fn the_edges_with_the_invisible_records_offered_admit_what_their_flat_identity_allows() {
    let d = decide(&registry_scene(true), view_c());
    assert!(
        d.fragments.iter().all(|f| f.slot != 2),
        "the square paints nothing"
    );
    let (r, u) = (id_of(&d, 0, STROKE), id_of(&d, 1, STROKE));
    for p in pairs_at(&d, 32, 32, r, u) {
        // By the axis rule alone, an exact tie; offered the square's flat identity, every edge.
        assert_eq!((p.label, p.edges, p.offered), (4, 1, 7), "{p:?}");
    }
    let d = decide(&registry_scene(false), view_c());
    assert!(!d.pairs.is_empty());
    for p in &d.pairs {
        assert_eq!(p.offered, p.edges, "{p:?}");
    }
}

#[test]
fn a_mesh_triangle_whose_alpha_byte_is_0_is_written_flagged_with_no_rows() {
    for (alpha, zero) in [(0.003, true), (1.0, false)] {
        let mut b = Build::new(false);
        b.mesh(
            vec![[-6.0, -6.0, 0.0], [6.0, -6.0, 0.0], [0.0, 6.0, 0.0]],
            vec![[0, 1, 2]],
            [1.0, 0.0, 0.0, alpha],
        );
        let d = decide(&b, view_f());
        assert_eq!(d.fragments.len(), 1, "{:?}", d.fragments);
        let f = &d.fragments[0];
        assert_eq!(f.flags & FLAG_ZERO_ALPHA != 0, zero, "{f:?}");
        assert_eq!(d.covers.is_empty(), zero);
        assert_eq!(d.colours.len(), 1);
        if zero {
            assert_eq!(d.colours[0].rgba, [0; 4]);
            assert!(f.bbox[0] <= 8 && f.bbox[2] > 8, "{f:?}");
        }
    }
}
