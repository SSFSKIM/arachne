//! A record Cairo paints nothing of takes no part in the frame's plane registry
//! (`depth::Registry`), whose identities are a function of the set of records offered: offered, it
//! could move what the visible records share, and so how their pairs are ordered, while Cairo's
//! frame is byte-identical with or without it.

mod common;

use std::collections::BTreeMap;

use arachne_core::composite::premultiply;
use arachne_core::diagnostics::Pixel;
use arachne_core::order::Decided;
use arachne_core::xf::*;
use common::meshes::{mesh_record, quad, stream_of};

const W: u32 = 160;
const H: u32 = 90;

fn view() -> View {
    let mut v = common::three_d_view(W, H);
    v.frame_center = [0.0; 3];
    v.mapping_centre = [0.0; 3];
    v.input_grid = 1.0 / 32768.0;
    if let Camera::ThreeD(c) = &mut v.camera {
        c.phi = 0.8;
        c.theta = -std::f64::consts::FRAC_PI_2;
        c.gamma = 0.0;
        c.should_apply_shading = false;
    }
    v
}

/// The polyline through `corners` in CE's cubic layout.
fn cubics(corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for w in corners.windows(2) {
        let (a, b) = (w[0], w[1]);
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

/// A path record in the world drawing `corners`, with no paint.
fn path(slot: u32, corners: &[[f64; 3]]) -> (Shape, Record) {
    let shape = Shape::new(ShapeData::Path {
        points: cubics(corners),
    });
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
    r.depth.space = Space::World;
    (shape, r)
}

fn stroked(slot: u32, corners: &[[f64; 3]], colour: [f64; 4]) -> (Shape, Record) {
    let (shape, mut r) = path(slot, corners);
    r.paint.stroke = Paint::One(colour);
    r.paint.stroke_width = 60.0;
    (shape, r)
}

/// Two translucent lines on z = 0, side by side: things on one plane, tied by draw order.
fn lines() -> Vec<(Shape, Record)> {
    vec![
        stroked(
            1,
            &[[-3.0, -0.1, 0.0], [3.0, -0.1, 0.0]],
            [1.0, 0.0, 0.0, 0.5],
        ),
        stroked(
            2,
            &[[-3.0, 0.1, 0.0], [3.0, 0.1, 0.0]],
            [0.0, 0.0, 1.0, 0.5],
        ),
    ]
}

/// A rectangle on z = 0 whose hull holds the first line and not the second, with no paint.
fn plate(slot: u32) -> (Shape, Record) {
    path(
        slot,
        &[
            [-4.0, -1.0, 0.0],
            [4.0, -1.0, 0.0],
            [4.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [-4.0, -1.0, 0.0],
        ],
    )
}

/// Everything a frame shows of its pixels: colour, resolve class, and every pixel's fragments
/// with each pair's order and how it was decided.
struct Shown {
    rgba: Vec<u8>,
    resolves: Vec<u8>,
    pixels: BTreeMap<(u32, u32), Pixel>,
}

fn shown(items: &[(Shape, Record)], order: &[u32]) -> Shown {
    let s = stream_of(view(), items, order);
    let r = s.render(W, H, 1).unwrap();
    Shown {
        rgba: r.rgba,
        resolves: r.resolves,
        pixels: s.diagnostic_resolves(0, 0, W, H, true).unwrap(),
    }
}

/// How `with` departs from `without`: the pixels whose colour, resolve class or pair decisions
/// differ.
fn departures(with: &Shown, without: &Shown) -> [usize; 3] {
    let recoloured = (0..(W * H) as usize)
        .filter(|i| with.rgba[i * 4..i * 4 + 4] != without.rgba[i * 4..i * 4 + 4])
        .count();
    let reclassed = (0..(W * H) as usize)
        .filter(|i| with.resolves[*i] != without.resolves[*i])
        .count();
    let reordered = with
        .pixels
        .iter()
        .filter(|(at, p)| {
            let q = &without.pixels[*at];
            p.pairs != q.pairs || p.sample_pairs != q.sample_pairs
        })
        .count();
    [recoloured, reclassed, reordered]
}

/// The largest alpha Cairo paints nothing of (its 16-bit alpha at most 0xff, its 8-bit alpha
/// zero) and the smallest it paints.
fn boundary() -> (f64, f64) {
    let dropped = 0.0038986800946059354_f64;
    let kept = dropped.next_up();
    assert_eq!(premultiply([0.0, 0.0, 0.0, dropped])[3], 0);
    assert_eq!(premultiply([0.0, 0.0, 0.0, kept])[3], 1);
    (dropped, kept)
}

type Painting = Box<dyn Fn(&mut Record)>;

/// The plate, painted by `paint`, drawn after the lines in either order: the frame with it, the
/// frame without it, and the lines' order.
fn with_and_without(paint: &Painting) -> Vec<(Shown, Shown, [u32; 2])> {
    let (shape, mut r) = plate(3);
    paint(&mut r);
    let mut out = Vec::new();
    for order in [[1, 2], [2, 1]] {
        let without = shown(&lines(), &order);
        let mut items = lines();
        items.push((shape.clone(), r.clone()));
        out.push((shown(&items, &[order[0], order[1], 3]), without, order));
    }
    out
}

fn fill(alpha: f64) -> Painting {
    Box::new(move |r: &mut Record| r.paint.fill = Paint::One([0.0, 1.0, 0.0, alpha]))
}

fn gradient(alphas: [f64; 2], ends: [[f64; 3]; 2]) -> Painting {
    Box::new(move |r: &mut Record| {
        r.paint.fill = Paint::Two([[0.0, 1.0, 0.0, alphas[0]], [0.0, 1.0, 0.0, alphas[1]]]);
        r.paint.gradient = Some(ends);
    })
}

const ACROSS: [[f64; 3]; 2] = [[-4.0, -1.0, 0.0], [4.0, 0.0, 0.0]];

#[test]
fn a_record_cairo_paints_nothing_of_changes_no_pixel_resolve_or_pair() {
    let (dropped, _) = boundary();
    let cases: Vec<(&str, Painting)> = vec![
        ("a fill of no alpha", fill(0.0)),
        (
            "a fill and a stroke of no alpha",
            Box::new(|r: &mut Record| {
                r.paint.fill = Paint::One([0.0, 1.0, 0.0, 0.0]);
                r.paint.stroke = Paint::One([0.0, 1.0, 0.0, 0.0]);
                r.paint.stroke_width = 60.0;
            }),
        ),
        ("a fill of the largest alpha dropped", fill(dropped)),
        (
            "a fill under the largest opacity dropped",
            Box::new(move |r: &mut Record| {
                r.paint.fill = Paint::One([0.0, 1.0, 0.0, 1.0]);
                r.opacity = dropped;
            }),
        ),
        ("a clear gradient", gradient([0.0, dropped], ACROSS)),
        // Its ends meet: Cairo paints the average of its first and last stops, of an alpha
        // (0.0035) dropped though its last stop's is not.
        (
            "a gradient whose ends meet, of a clear average",
            gradient([0.0, 0.007], [[0.0, -0.5, 0.0]; 2]),
        ),
        (
            "a fill of no alpha and a stroke of no width",
            Box::new(|r: &mut Record| {
                r.paint.fill = Paint::One([0.0, 1.0, 0.0, 0.0]);
                r.paint.stroke = Paint::One([0.0, 1.0, 0.0, 1.0]);
                r.paint.stroke_width = 0.0;
            }),
        ),
    ];
    let mut failures = Vec::new();
    for (name, paint) in &cases {
        for (with, without, order) in with_and_without(paint) {
            let [recoloured, reclassed, reordered] = departures(&with, &without);
            if recoloured + reclassed + reordered > 0 {
                failures.push(format!(
                    "{name}, lines drawn {order:?}: {recoloured} pixels change colour, \
                     {reclassed} resolve class, {reordered} pair decisions"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_smallest_alpha_cairo_paints_is_drawn() {
    let (_, kept) = boundary();
    let cases: Vec<(&str, Painting)> = vec![
        ("a fill", fill(kept)),
        ("a gradient", gradient([0.0, kept], ACROSS)),
    ];
    for (name, paint) in &cases {
        for (with, without, order) in with_and_without(paint) {
            let [recoloured, reclassed, _] = departures(&with, &without);
            let drawn = with
                .pixels
                .values()
                .any(|p| p.entries.iter().any(|e| e.leaf.0 == 103));
            assert!(
                drawn && recoloured + reclassed > 0,
                "{name} of alpha {kept}, lines drawn {order:?}: drew nothing \
                 ({recoloured} pixels change colour, {reclassed} resolve class)"
            );
        }
    }
}

/// The plate's stroke is drawn though its fill is not: it is keyed on its plane, which the line
/// crossing it shares, and they tie by draw order wherever both cover a pixel.
#[test]
fn a_visible_stroke_of_an_invisible_fill_still_takes_part() {
    let crossing = stroked(
        1,
        &[[-1.0, -2.0, 0.0], [-1.0, 1.0, 0.0]],
        [0.0, 0.0, 1.0, 0.5],
    );
    let (shape, mut r) = plate(2);
    r.paint.fill = Paint::One([0.0, 1.0, 0.0, 0.0]);
    r.paint.stroke = Paint::One([1.0, 0.0, 0.0, 0.5]);
    r.paint.stroke_width = 60.0;
    let items = vec![crossing, (shape, r)];
    let mut failures = Vec::new();
    for order in [[1, 2], [2, 1]] {
        let s = shown(&items, &order);
        let mut both = 0;
        for (at, p) in &s.pixels {
            let leaf = |slot: u32| p.entries.iter().position(|e| e.leaf.0 == 100 + slot);
            let (Some(line), Some(plate)) = (leaf(1), leaf(2)) else {
                continue;
            };
            both += 1;
            let later = if order[1] == 1 { line } else { plate };
            for pair in p.pairs.iter().chain(p.sample_pairs.iter().flatten()) {
                if [pair.a, pair.b] == [line.min(plate), line.max(plate)]
                    && (pair.how != Decided::Tie || pair.front != Some(later))
                {
                    failures.push(format!("drawn {order:?}, pixel {at:?}: {pair:?}"));
                }
            }
        }
        assert!(both > 0, "the line crosses the plate's stroke");
    }
    assert!(
        failures.is_empty(),
        "{} pairs not tied by draw order:\n{}",
        failures.len(),
        failures[..failures.len().min(8)].join("\n")
    );
}

/// A mesh Cairo paints nothing of (its alpha, by its fill or its record's opacity, clear), on z = 0
/// over one end of the first line, is offered to the registry no more than a path is: the lines'
/// pairs and pixels are the frame's without it wherever it lays no fragment of its own.
#[test]
fn a_mesh_cairo_paints_nothing_of_moves_no_pair_of_others() {
    let (dropped, _) = boundary();
    let corners = [
        [-4.0, -1.0, 0.0],
        [-2.0, -1.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-4.0, 0.0, 0.0],
    ];
    let shape = quad(corners);
    let mut failures = Vec::new();
    for (alpha, opacity) in [(0.0, 1.0), (1.0, 0.0), (dropped, 1.0), (1.0, dropped)] {
        let mut r = mesh_record(3, &shape, [0.0, 1.0, 0.0, alpha], Space::World);
        r.opacity = opacity;
        for order in [[1, 2], [2, 1]] {
            let without = shown(&lines(), &order);
            let mut items = lines();
            items.push((shape.clone(), r.clone()));
            let with = shown(&items, &[order[0], order[1], 3]);
            // Where the mesh lays a fragment, the fragment's own resolve is another matter.
            let own = |at: &(u32, u32)| with.pixels[at].entries.iter().any(|e| e.leaf.0 == 103);
            let moved = with
                .pixels
                .iter()
                .filter(|(at, p)| {
                    let i = (at.1 * W + at.0) as usize;
                    !own(at)
                        && (**p != without.pixels[*at]
                            || with.rgba[i * 4..i * 4 + 4] != without.rgba[i * 4..i * 4 + 4]
                            || with.resolves[i] != without.resolves[i])
                })
                .count();
            if moved > 0 {
                failures.push(format!(
                    "alpha {alpha} at opacity {opacity}, lines drawn {order:?}: {moved} pixels \
                     the mesh lays no fragment in change"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
