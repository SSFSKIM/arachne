//! Depth cases with a closed form (contract §5, §6, §7), in view C's camera (64 x 64, the identity
//! rotation, focal distance 20, zoom 1, 4 px per frame unit): two planes meeting on a known line,
//! a plane seen edge-on, a stroke crossing a surface.

mod common;

use arachne_core::xf::{
    Camera, Paint, Record, RecordKind, Shape, ShapeData, Space, ThreeDCamera, View,
};
use arachne_oracle::cover::sample_point;
use arachne_oracle::out::{Decisions, PairRow};
use common::*;

const D: f64 = 20.0;
const XX: f64 = 4.0;

fn view_c(grid: f64) -> View {
    let mut v = view(64, 64);
    v.frame_width = 16.0;
    v.frame_height = 16.0;
    v.camera = Camera::ThreeD(ThreeDCamera {
        phi: 0.0,
        theta: -std::f64::consts::FRAC_PI_2,
        gamma: 0.0,
        focal_distance: D,
        zoom: 1.0,
        exponential_projection: false,
        light_source: [-7.0, -9.0, 10.0],
        shading_factor: 0.2,
        should_apply_shading: false,
    });
    v.input_grid = grid;
    v
}

/// A world-space mesh of one triangle, filled white.
fn triangle(slot: u32, corners: [[f64; 3]; 3]) -> (Shape, Record) {
    let mesh = Shape::new(ShapeData::Mesh {
        positions: corners.to_vec(),
        triangles: vec![[0, 1, 2]],
        normals: None,
    });
    let mut m = Record::new(RecordKind::Mesh, slot, 100 + slot);
    m.geometry.shape_a = Some(mesh.key());
    m.paint.fill = Paint::One([1.0, 1.0, 1.0, 1.0]);
    m.depth.space = Space::World;
    (mesh, m)
}

fn decide_window(v: View, items: Vec<(Shape, Record)>, window: [u32; 4]) -> Decisions {
    let (shapes, records): (Vec<_>, Vec<_>) = items.into_iter().unzip();
    arachne_oracle::decide::decide_at(&[delta(1, Some(v), shapes, records)], 1, Some(window))
        .unwrap()
}

fn fragment_of(d: &Decisions, slot: u32) -> u32 {
    let f: Vec<_> = d.fragments.iter().filter(|f| f.slot == slot).collect();
    assert_eq!(f.len(), 1, "{:?}", d.fragments);
    f[0].id
}

fn cover_at(d: &Decisions, x: u32, y: u32, fragment: u32, s: u8) -> u8 {
    d.covers
        .iter()
        .find(|c| (c.x, c.y, c.fragment) == (x, y, fragment))
        .map_or(0, |c| c.cover[s as usize])
}

/// The pair rows of fragments `a` and `b` where both are `Yes`, with their sample's position.
fn pairs_both_yes(d: &Decisions, a: u32, b: u32) -> Vec<(&PairRow, [f64; 2])> {
    d.pairs
        .iter()
        .filter(|p| (p.a, p.b) == (a.min(b), a.max(b)))
        .filter(|p| {
            cover_at(d, p.x, p.y, p.a, p.sample) == 1 && cover_at(d, p.x, p.y, p.b, p.sample) == 1
        })
        .map(|p| (p, sample_point(p.x as i64, p.y as i64, p.sample as usize)))
        .collect()
}

const DEPTH_A: u8 = 1;
const DEPTH_B: u8 = 2;
const TIE_BAND: u8 = 3;

/// §6's `S` on a flat triangle's world plane `n · p = 0` (unplaced, the identity rotation, the
/// camera at the origin), read at key `o` inside its hull, from the contract's text:
/// `(|o| + R) kr(P, δ)` with `δ = τ_P(n_P) = (g/2) |n|_1` (exact corners depart by nothing),
/// `kr = δ / (|D| − δ)` and `D = −n_2 d`. `R` is far smaller and left out.
fn s_world_plane(n: [f64; 3], g: f64, o: f64) -> f64 {
    let len = libm::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    let n = n.map(|v| v / len);
    let delta = g / 2.0 * (n[0].abs() + n[1].abs() + n[2].abs());
    let big_d = (n[2] * D).abs();
    o.abs() * delta / (big_d - delta)
}

#[test]
fn two_planes_meeting_on_a_line_are_ordered_by_depth_off_a_tie_band_as_wide_as_s6_predicts() {
    // A on z = x / 2, B on z = 0: they meet on x = 0, screen column 32; A is nearer right of it.
    let g = 1.0 / 16.0;
    let a = [[-6.0, -6.0, -3.0], [6.0, -6.0, 3.0], [0.0, 6.0, 0.0]];
    let b = [[-8.0, -8.0, 0.0], [8.0, -8.0, 0.0], [0.0, 8.0, 0.0]];
    let d = decide_window(
        view_c(g),
        vec![triangle(0, a), triangle(1, b)],
        [20, 28, 44, 36],
    );
    let (fa, fb) = (fragment_of(&d, 0), fragment_of(&d, 1));
    assert!(fa < fb);
    let pairs = pairs_both_yes(&d, fa, fb);
    assert!(pairs.len() > 1000, "{}", pairs.len());
    let (mut tie_lo, mut tie_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut a_front, mut b_front) = (0, 0);
    for (p, at) in &pairs {
        match p.label {
            DEPTH_A => {
                assert!(at[0] > 32.0, "A in front left of the line at {at:?}");
                a_front += 1;
            }
            DEPTH_B => {
                assert!(at[0] < 32.0, "B in front right of the line at {at:?}");
                b_front += 1;
            }
            TIE_BAND => {
                tie_lo = tie_lo.min(at[0]);
                tie_hi = tie_hi.max(at[0]);
            }
            other => panic!("label {other} at {at:?}"),
        }
    }
    assert!(a_front > 100 && b_front > 100, "{a_front} {b_front}");
    // §6's prediction: the intervals meet where |k_A − k_B| <= S_A + S_B (R is far smaller), both
    // keys near −1 there. On screen near the line, k_A moves by (1/2) / D per frame unit, k_B not
    // at all.
    let slope = 0.5 / D / XX;
    let s = s_world_plane([-0.5, 0.0, 1.0], g, -1.0) + s_world_plane([0.0, 0.0, 1.0], g, -1.0);
    let predicted = 2.0 * s / slope;
    let measured = tie_hi - tie_lo;
    assert!(
        measured >= predicted / 2.0 && measured <= predicted * 2.0,
        "tie band {measured} px wide, predicted {predicted} px ({tie_lo}..{tie_hi})"
    );
    assert!(tie_lo < 32.0 && tie_hi > 32.0, "{tie_lo}..{tie_hi}");
}

#[test]
fn a_plane_seen_edge_on_is_never_certainly_covered() {
    // The plane x = 0 holds the eye: every corner draws on screen column 32.
    // A grid coarse enough that the snap may move the edge past the nearest samples (1/32 px).
    let g = 1.0 / 16.0;
    let edge = [[0.0, -4.0, -2.0], [0.0, 4.0, -2.0], [0.0, 0.0, 3.0]];
    let back = [[-8.0, -8.0, -4.0], [8.0, -8.0, -4.0], [0.0, 8.0, -4.0]];
    let d = decide_window(
        view_c(g),
        vec![triangle(0, edge), triangle(1, back)],
        [24, 16, 40, 48],
    );
    let fe = fragment_of(&d, 0);
    let rows: Vec<_> = d.covers.iter().filter(|c| c.fragment == fe).collect();
    assert!(!rows.is_empty());
    for r in rows {
        assert!(
            r.cover.iter().all(|c| *c != 1),
            "({}, {}) {:?}",
            r.x,
            r.y,
            r.cover
        );
    }
}

#[test]
fn a_stroke_crossing_a_surface_is_in_front_on_its_near_side_and_ties_only_at_the_crossing() {
    // The line runs from z = −2 to z = 2 across the plane z = 0, crossing it on screen column 32.
    let g = 1.0 / 1_048_576.0;
    let line = path_shape(polyline(&[[-6.0, 0.0, -2.0], [6.0, 0.0, 2.0]]));
    let mut stroke = path(0, line.key(), false, Some(0.5));
    stroke.depth.space = Space::World;
    let surface = [[-8.0, -8.0, 0.0], [8.0, -8.0, 0.0], [0.0, 8.0, 0.0]];
    let d = decide_window(
        view_c(g),
        vec![(line, stroke), triangle(1, surface)],
        [8, 28, 56, 36],
    );
    let (fs, fm) = (fragment_of(&d, 0), fragment_of(&d, 1));
    assert!(fs < fm);
    let pairs = pairs_both_yes(&d, fs, fm);
    assert!(pairs.len() > 100, "{}", pairs.len());
    let (mut near, mut far) = (0, 0);
    for (p, at) in &pairs {
        match p.label {
            DEPTH_A => {
                assert!(at[0] > 32.0, "the stroke in front at {at:?}");
                near += 1;
            }
            DEPTH_B => {
                assert!(at[0] < 32.0, "the surface in front at {at:?}");
                far += 1;
            }
            TIE_BAND => assert!(
                (at[0] - 32.0).abs() < 1.0,
                "a tie away from the crossing at {at:?}"
            ),
            other => panic!("label {other} at {at:?}"),
        }
    }
    assert!(near > 20 && far > 20, "{near} {far}");
}
