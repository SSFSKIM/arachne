//! Keys and bounds the contract fixes in closed form (§2, §5, §6), each in a scene of view C's
//! frame (64 x 64 px over 16 x 16 units, 4 px a unit, centre at pixel 32) under the identity
//! rotation at the focal distance each case names.

#[path = "common/scenes.rs"]
mod scenes;

use arachne_core::xf::{
    Camera, FEATURE_PRE_PROJECTED, Paint, PreProjected, Shape, ShapeData, View,
};
use arachne_oracle::cover::sample_point;
use arachne_oracle::decide::decide_at;
use arachne_oracle::out::{BandRow, Decisions};
use scenes::*;

fn view(focal_distance: f64, grid: f64) -> View {
    View {
        camera: camera(focal_distance),
        input_grid: grid,
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

/// `fragment`'s own rows (`which` 0).
fn own_rows(d: &Decisions, fragment: u32) -> Vec<&BandRow> {
    d.bands
        .iter()
        .filter(|b| b.fragment == fragment && b.which == 0)
        .collect()
}

fn interval(b: &BandRow, s: usize) -> (f64, f64) {
    (
        b.o[s] - b.s_minus[s] - 3.0 * b.r[s],
        b.o[s] + b.s_plus[s] + 3.0 * b.r[s],
    )
}

/// The frame point (frame units) of a sample of pixel `(x, y)`.
fn shown(x: u32, y: u32, s: usize) -> [f64; 2] {
    let p = sample_point(x as i64, y as i64, s);
    [(p[0] - 32.0) / 4.0, -(p[1] - 32.0) / 4.0]
}

#[test]
fn a_pre_projected_fill_is_fitted_on_its_points_put_back_by_unproject() {
    // A tail with corners (1, −1, 0), (2, −1, 5), (2, 1, 5), (1, 1, 0) at focal distance 10:
    // `unproject` puts each on x = 1 in the camera's axes, where a screen point at shown x keys
    // −x. Fitted on the shown points instead, it is z = 5x − 5. Each edge `a → b` is written
    // `[a, a, b, b]`, so every control point puts back on x = 1.
    let corners = [
        [1.0, -1.0, 0.0],
        [2.0, -1.0, 5.0],
        [2.0, 1.0, 5.0],
        [1.0, 1.0, 0.0],
    ];
    let tail = (0..4)
        .flat_map(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            [a, a, b, b]
        })
        .collect();
    let mut b = Build::new(true);
    b.pre_projected(tail, OPAQUE_BLUE);
    let d = decide(&b, view(10.0, 0.0));
    let fill = id_of(&d, 0, 1);
    let mut read = 0;
    for row in own_rows(&d, fill) {
        for s in 0..16 {
            let [x, y] = shown(row.x, row.y, s);
            if row.status[s] != 1 || !(x > 1.01 && x < 1.99 && y.abs() < 0.99) {
                continue;
            }
            let (lo, hi) = interval(row, s);
            assert!(
                lo <= -x && -x <= hi,
                "({}, {}) sample {s} at shown x {x}: [{lo}, {hi}] misses {}",
                row.x,
                row.y,
                -x
            );
            read += 1;
        }
    }
    assert!(read > 100, "{read} samples read");
}

/// A mesh triangle at z = 5 over the origin and the point block `points` adds: the pair's `edges`,
/// the mesh as `a`, one at every sample both cover.
fn mesh_and_block(points: impl FnOnce(&mut Build)) -> u8 {
    let mut b = Build::new(true);
    b.mesh(
        vec![[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 5.0]],
        vec![[0, 1, 2]],
        OPAQUE_RED,
    );
    points(&mut b);
    let d = decide(&b, view(10.0, GRID));
    let (m, p) = (id_of(&d, 0, 3), id_of(&d, 1, 4));
    let pairs: Vec<_> = d.pairs.iter().filter(|r| (r.a, r.b) == (m, p)).collect();
    assert!(!pairs.is_empty(), "no pair of the mesh and the block");
    let edges = pairs[0].edges;
    assert!(pairs.iter().all(|r| r.edges == edges));
    edges
}

#[test]
fn a_fixed_orientation_point_block_is_keyed_on_its_facing_plane() {
    // §5 `facing_plane@1`: a fixed-orientation record's fragments, point blocks included, are
    // keyed on the facing plane through its centre. The block at z = 0 keys −1, the mesh −2: the
    // mesh is in front, strongly.
    let edges = mesh_and_block(|b| {
        b.points(vec![[0.125, -0.125, 0.0]], OPAQUE_GREEN, 2.0)
            .depth
            .fixed_orientation_centre = Some([0.125, -0.125, 0.0]);
    });
    assert_eq!(edges, 2, "the mesh in front alone");
}

#[test]
fn a_pre_projected_point_block_is_keyed_in_the_camera_s_axes() {
    // §2: a pre-projected record is keyed in the camera's own axes, each shown point put back by
    // `unproject`; §5: a point is keyed on the facing plane through its position. The tail's
    // point (0.25, −0.25, 5) puts back at (0.125, −0.125, 5), keyed −2. Against a mesh at z = 7.5
    // (−4), the mesh is in front.
    let mut b = Build::new(true);
    b.mesh(
        vec![[-1.0, -1.0, 7.5], [1.0, -1.0, 7.5], [0.0, 1.0, 7.5]],
        vec![[0, 1, 2]],
        OPAQUE_RED,
    );
    let r = b.points(vec![[0.25, -0.25, 5.0]], OPAQUE_GREEN, 2.0);
    r.features |= FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: r.geometry.shape_a.unwrap(),
        gradient: None,
        fill: Paint::None,
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let d = decide(&b, view(10.0, GRID));
    let (m, p) = (id_of(&d, 0, 3), id_of(&d, 1, 4));
    let at: Vec<_> = d
        .pairs
        .iter()
        .filter(|r| (r.a, r.b) == (m, p))
        .map(|r| r.edges)
        .collect();
    assert!(!at.is_empty(), "no pair of the mesh and the block");
    assert!(at.iter().all(|e| *e == 2), "{at:?}");
}

#[test]
fn a_pre_projected_pivot_moves_as_the_record_s_first_point_moved() {
    // §2: a fixed-orientation pivot is the pivot as produced, moved as the record's first point
    // moved; §5: a pre-projected record with a pivot is keyed on the facing plane through it, in
    // the camera's axes. The record's square lies on z = 0 about its pivot (0, 0, 0); its tail is
    // that square moved to z = 5. The pivot moves to (0, 0, 5) with it, keyed −2 at focal
    // distance 10, where the unmoved pivot keys −1.
    let mut b = Build::new(true);
    let tail = Shape::new(ShapeData::Path {
        points: rect([-1.0, 1.0], [-1.0, 1.0], 5.0),
    });
    let key = tail.key();
    b.shapes.push(tail);
    let r = b.fill(rect([-1.0, 1.0], [-1.0, 1.0], 0.0), OPAQUE_BLUE);
    r.depth.fixed_orientation_centre = Some([0.0; 3]);
    r.features |= FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: key,
        gradient: None,
        fill: Paint::One(OPAQUE_BLUE),
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let d = decide(&b, view(10.0, GRID));
    let fill = id_of(&d, 0, 1);
    let mut read = 0;
    for row in own_rows(&d, fill) {
        for s in 0..16 {
            if row.status[s] != 1 {
                continue;
            }
            let (lo, hi) = interval(row, s);
            assert!(
                lo <= -2.0 && -2.0 <= hi,
                "({}, {}) sample {s}: [{lo}, {hi}] misses −2",
                row.x,
                row.y
            );
            read += 1;
        }
    }
    assert!(read > 100, "{read} samples read");
}

#[test]
fn a_pre_projected_point_block_s_pivot_moves_as_its_first_point_moved() {
    // As above, for a point block: its position (0.125, −0.125, 0) about its pivot there, its tail's
    // point (0.25, −0.25, 5). The pivot moves to (0.25, −0.25, 5), keyed −2 at focal distance 10;
    // unmoved it keys −1. Against a mesh at z = 2.5 (−4/3), the block is in front.
    let mut b = Build::new(true);
    b.mesh(
        vec![[-1.0, -1.0, 2.5], [1.0, -1.0, 2.5], [0.0, 1.0, 2.5]],
        vec![[0, 1, 2]],
        OPAQUE_RED,
    );
    let tail = Shape::new(ShapeData::Path {
        points: vec![[0.25, -0.25, 5.0]],
    });
    let key = tail.key();
    b.shapes.push(tail);
    let r = b.points(vec![[0.125, -0.125, 0.0]], OPAQUE_GREEN, 2.0);
    r.depth.fixed_orientation_centre = Some([0.125, -0.125, 0.0]);
    r.features |= FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: key,
        gradient: None,
        fill: Paint::None,
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let d = decide(&b, view(10.0, GRID));
    let (m, p) = (id_of(&d, 0, 3), id_of(&d, 1, 4));
    let at: Vec<_> = d
        .pairs
        .iter()
        .filter(|r| (r.a, r.b) == (m, p))
        .map(|r| r.edges)
        .collect();
    assert!(!at.is_empty(), "no pair of the mesh and the block");
    assert!(at.iter().all(|e| *e == 4), "{at:?}");
}

/// Every read sample of `fragment`'s own rows: `(S⁻, S⁺)`.
fn input_bounds(d: &Decisions, fragment: u32) -> Vec<(f64, f64)> {
    own_rows(d, fragment)
        .iter()
        .flat_map(|row| {
            (0..16)
                .filter(|&s| row.status[s] == 1)
                .map(|s| (row.s_minus[s], row.s_plus[s]))
        })
        .collect()
}

#[test]
fn a_pre_projected_pivot_s_facing_key_takes_its_carried_centre_s_precision() {
    // §5: a pre-projected record with a pivot is keyed on the facing plane through it, in the
    // camera's axes; §6, on a facing plane, for a label: `S⁻ = S⁺ = (|o| + R) ds(Δ_c)`, with
    // `Δ_c = (g/2) |R_2|_1` for its centre as carried. Under the identity camera at focal distance
    // 20, a pivot at the origin keys −1 and `Δ_c = g/2`: `S ≈ (g/2) / (20 − g/2)`, the `R` term
    // far below it.
    let ds = (GRID / 2.0) / (20.0 - GRID / 2.0);
    let near = |(lo, hi): (f64, f64)| lo == hi && (lo - ds).abs() <= ds * 1e-6;
    // A path, its tail its own square.
    let mut b = Build::new(true);
    b.pre_projected(rect([-1.0, 1.0], [-1.0, 1.0], 0.0), OPAQUE_BLUE)
        .depth
        .fixed_orientation_centre = Some([0.0; 3]);
    let d = decide(&b, view(20.0, GRID));
    let path = input_bounds(&d, id_of(&d, 0, 1));
    assert!(path.len() > 100, "{} samples read", path.len());
    assert!(path.iter().all(|s| near(*s)), "{:?}", &path[..4]);
    // A point block, its tail its own point.
    let mut b = Build::new(true);
    let tail = Shape::new(ShapeData::Path {
        points: vec![[0.125, -0.125, 0.0]],
    });
    let key = tail.key();
    b.shapes.push(tail);
    let r = b.points(vec![[0.125, -0.125, 0.0]], OPAQUE_GREEN, 2.0);
    r.depth.fixed_orientation_centre = Some([0.125, -0.125, 0.0]);
    r.features |= FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: key,
        gradient: None,
        fill: Paint::None,
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let mut delta = b.delta(1, view(20.0, GRID));
    let mut seen = std::collections::HashSet::new();
    delta.upload.retain(|s| seen.insert(s.key()));
    let d = decide_at(&[delta], 1, None).unwrap();
    assert_eq!(d.header.status, 0, "{:?}", d.refused);
    let block = input_bounds(&d, id_of(&d, 0, 4));
    assert!(!block.is_empty());
    assert!(block.iter().all(|s| near(*s)), "{block:?}");
}

#[test]
fn a_pre_projected_pivot_with_no_first_point_to_move_by_states_no_bound() {
    // §2 moves a pivot as the record's first point moved; a record whose own shape is empty has
    // no first point, and its tail still draws. Its key is not stated: `pivot_unknown`.
    let mut b = Build::new(true);
    let tail = Shape::new(ShapeData::Path {
        points: rect([-1.0, 1.0], [-1.0, 1.0], 5.0),
    });
    let key = tail.key();
    b.shapes.push(tail);
    let r = b.fill(Vec::new(), OPAQUE_BLUE);
    r.depth.fixed_orientation_centre = Some([0.0; 3]);
    r.features |= FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: key,
        gradient: None,
        fill: Paint::One(OPAQUE_BLUE),
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let d = decide(&b, view(10.0, GRID));
    let fill = id_of(&d, 0, 1);
    let rows = own_rows(&d, fill);
    assert!(!rows.is_empty());
    for row in rows {
        for s in 0..16 {
            if row.status[s] != 0 {
                assert_eq!(row.status[s], 4, "({}, {}) sample {s}", row.x, row.y);
            }
        }
    }
}

/// The pair rows of record 0's fill and stroke where both are present: edges and label.
fn own_layer_pairs(d: &Decisions) -> Vec<(u8, u8)> {
    let (fill, stroke) = (id_of(d, 0, 1), id_of(d, 0, 2));
    d.pairs
        .iter()
        .filter(|r| (r.a, r.b) == (fill, stroke) || (r.a, r.b) == (stroke, fill))
        .map(|r| (r.edges, r.label))
        .collect()
}

#[test]
fn a_pivot_with_no_move_leaves_its_record_s_layers_unable_against_each_other() {
    // §2: the pivot case above with an opaque stroke 0.25 wide on its tail. Both layers state no
    // key (`pivot_unknown`, per fragment), so where they meet the pair is `truth_unable`: every
    // edge, label 0, though the two are one record's own layers.
    let mut b = Build::new(true);
    let tail = Shape::new(ShapeData::Path {
        points: rect([-1.0, 1.0], [-1.0, 1.0], 5.0),
    });
    let key = tail.key();
    b.shapes.push(tail);
    let r = b.fill(Vec::new(), OPAQUE_BLUE);
    r.paint.stroke = Paint::One(WHITE);
    r.paint.stroke_width = 0.25;
    r.depth.fixed_orientation_centre = Some([0.0; 3]);
    r.features |= FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: key,
        gradient: None,
        fill: Paint::One(OPAQUE_BLUE),
        stroke: Paint::One(WHITE),
        background_stroke: Paint::None,
    });
    let d = decide(&b, view(10.0, GRID));
    let at = own_layer_pairs(&d);
    assert!(!at.is_empty(), "no pair of the fill and its stroke");
    assert!(at.iter().all(|e| *e == (7, 0)), "{at:?}");
}

#[test]
fn a_record_whose_keys_have_no_bound_leaves_its_layers_unable_against_each_other() {
    // §6: `band_unbounded` is per fragment. A fixed-orientation square with an opaque stroke 0.25
    // wide, its centre at z = 10^300: the facing model through it states no bound, for either
    // layer, and where they meet the pair is `truth_unable`.
    let mut b = Build::new(true);
    let r = b.fill(rect([-1.0, 1.0], [-1.0, 1.0], 0.0), OPAQUE_BLUE);
    r.paint.stroke = Paint::One(WHITE);
    r.paint.stroke_width = 0.25;
    r.depth.fixed_orientation_centre = Some([0.0, 0.0, 1e300]);
    let d = decide(&b, view(10.0, GRID));
    let fill = id_of(&d, 0, 1);
    for row in own_rows(&d, fill) {
        for s in 0..16 {
            if row.status[s] != 0 {
                assert_eq!(row.status[s], 2, "({}, {}) sample {s}", row.x, row.y);
            }
        }
    }
    let at = own_layer_pairs(&d);
    assert!(!at.is_empty(), "no pair of the fill and its stroke");
    assert!(at.iter().all(|e| *e == (7, 0)), "{at:?}");
}

#[test]
fn a_row_read_outside_a_hull_no_wider_than_twice_its_tolerance_is_unbounded() {
    // §6 `band_input@2`: where a read lies outside H and w_P <= 2 τ_P, the bound is infinite. A
    // unit square on z = 0 placed through diag(1, 1, 1000) at grid 0.01 has width 1 and a normal
    // tolerance near 5. Its stroke, 0.25 wide, reaches 0.125 past its right edge (x = 0.5) onto a
    // second square on z = 0, against which it reads its row on their shared plane.
    let mut b = Build::new(true);
    let r = b.fill(rect([-0.5, 0.5], [-0.5, 0.5], 0.0), OPAQUE_BLUE);
    r.paint.stroke = Paint::One(WHITE);
    r.paint.stroke_width = 0.25;
    r.geometry.place_a[2][2] = 1000.0;
    b.fill(rect([0.5, 2.0], [-0.5, 0.5], 0.0), OPAQUE_RED);
    let d = decide(&b, view(1000.0, 0.01));
    let stroke = id_of(&d, 0, 2);
    let mut outside = 0;
    for row in d
        .bands
        .iter()
        .filter(|b| b.fragment == stroke && b.which == 1)
    {
        for s in 0..16 {
            // Past the edge by more than the in-plane rounding, inside the second square.
            let [x, y] = shown(row.x, row.y, s);
            if row.status[s] == 0 || x <= 0.5 + 1e-3 || y.abs() >= 0.5 {
                continue;
            }
            assert_eq!(
                row.status[s],
                2,
                "({}, {}) sample {s} at x {x}: [{:?}]",
                row.x,
                row.y,
                interval(row, s)
            );
            outside += 1;
        }
    }
    assert!(outside > 0, "no row read past the edge");
}

#[test]
fn a_facing_key_whose_bounds_overflow_states_no_bound() {
    // §6: where S or R is infinite, the oracle states no bound. A glyph kept facing the camera
    // about (0, 0, 400) under the exponential option at focal distance 1 keys −e^400, finite; its
    // error overflows.
    let mut b = Build::new(true);
    b.glyph(
        rect([-1.0, 1.0], [-1.0, 1.0], 400.0),
        OPAQUE_BLUE,
        [0.0, 0.0, 400.0],
    );
    let mut v = view(1.0, 0.0);
    if let Camera::ThreeD(c) = &mut v.camera {
        c.exponential_projection = true;
    }
    let d = decide(&b, v);
    let fill = id_of(&d, 0, 1);
    let rows = own_rows(&d, fill);
    assert!(!rows.is_empty());
    for row in rows {
        for s in 0..16 {
            assert_ne!(
                row.status[s], 1,
                "({}, {}) sample {s} bounded: o {} r {} s {} {}",
                row.x, row.y, row.o[s], row.r[s], row.s_minus[s], row.s_plus[s]
            );
        }
    }
}
