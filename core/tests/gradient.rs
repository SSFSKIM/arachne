//! The gradient paint against Cairo's own bytes (`tests/data/gradient.txt`, written by
//! `tests/data/gen_gradient.py`): pixel-aligned rectangles filled with CE's linear gradients, every
//! row one composite run from the rectangle's left edge. The paint alone, then a path record so
//! filled, drawn by a stream.

use arachne_core::gradient::{Matrix, Source, source};
use arachne_core::project::FrameMap;
use arachne_core::stream::Stream;
use arachne_core::xf::*;

const FRAME: [f64; 2] = [14.222222222222221, 8.0];

struct Case {
    size: [u32; 2],
    centre: [f64; 3],
    p1: [f64; 2],
    p2: [f64; 2],
    rect: [i32; 4],
    stops: Vec<[f64; 4]>,
    pixels: Vec<(i64, i64, [u8; 4])>,
}

impl Case {
    /// CE's context matrix (`Camera.make_cairo_context`).
    fn ctm(&self) -> Matrix {
        let m = FrameMap::cairo(self.size[0], self.size[1], FRAME[0], FRAME[1], self.centre);
        Matrix {
            xx: m.xx,
            yx: 0.0,
            xy: 0.0,
            yy: m.yy,
            x0: m.x0,
            y0: m.y0,
        }
    }
}

fn cases() -> Vec<Case> {
    let text = include_str!("data/gradient.txt");
    let mut out: Vec<Case> = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let words: Vec<&str> = line.split_whitespace().collect();
        let num = |i: usize| words[i].parse::<f64>().unwrap();
        match words[0] {
            "case" => out.push(Case {
                size: [words[1].parse().unwrap(), words[2].parse().unwrap()],
                centre: [num(3), num(4), 0.0],
                p1: [num(5), num(6)],
                p2: [num(7), num(8)],
                rect: std::array::from_fn(|k| words[9 + k].parse().unwrap()),
                stops: Vec::new(),
                pixels: Vec::new(),
            }),
            "stop" => out
                .last_mut()
                .unwrap()
                .stops
                .push(std::array::from_fn(|k| num(1 + k))),
            "px" => out.last_mut().unwrap().pixels.push((
                words[1].parse().unwrap(),
                words[2].parse().unwrap(),
                std::array::from_fn(|k| words[4 + k].parse().unwrap()),
            )),
            other => panic!("unknown line {other}"),
        }
    }
    assert_eq!(out.len(), 24);
    out
}

#[test]
fn a_gradient_fill_is_cairo_s_bytes() {
    let mut compared = 0;
    for (n, c) in cases().iter().enumerate() {
        let painted = source(&c.ctm(), c.p1, c.p2, &c.stops, c.rect);
        for &(x, y, want) in &c.pixels {
            let got = match &painted {
                Source::Solid(colour) => *colour,
                Source::Linear(g) => g.at(x, y, c.rect[0] as i64),
            };
            assert_eq!(got, want, "case {n}, pixel ({x}, {y})");
            compared += 1;
        }
    }
    assert_eq!(compared, 24 * 40);
}

/// A closed polygon in CE's cubic layout: each side a straight cubic.
fn polygon(corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
    (0..corners.len())
        .flat_map(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % corners.len()]);
            let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
            [a, at(1.0 / 3.0), at(2.0 / 3.0), b]
        })
        .collect()
}

#[test]
fn a_path_filled_with_a_gradient_is_drawn_with_cairo_s_bytes() {
    let mut compared = 0;
    for (n, c) in cases().iter().enumerate() {
        let inverse = c.ctm().inverted().unwrap();
        let [x, y, w, h] = c.rect.map(|v| v as f64);
        let user = |px: f64, py: f64| {
            let (ux, uy) = (
                inverse.xx * px + inverse.xy * py + inverse.x0,
                inverse.yx * px + inverse.yy * py + inverse.y0,
            );
            [ux, uy, 0.0]
        };
        let corners = [
            user(x, y),
            user(x + w, y),
            user(x + w, y + h),
            user(x, y + h),
        ];
        let shape = Shape::new(ShapeData::Path {
            points: polygon(&corners),
        });
        let table = Stops::new(c.stops.clone());
        let mut r = Record::new(RecordKind::Path, 1, 101);
        r.geometry.shape_a = Some(shape.key());
        r.paint.fill = match c.stops.as_slice() {
            [a, b] => Paint::Two([*a, *b]),
            _ => Paint::Table(table.key()),
        };
        r.paint.gradient = Some([[c.p1[0], c.p1[1], 0.0], [c.p2[0], c.p2[1], 0.0]]);
        let mut d = Delta::new(Time { num: 0, den: 15 }, 0);
        d.view = Some(View {
            pixel_width: c.size[0],
            pixel_height: c.size[1],
            frame_width: FRAME[0],
            frame_height: FRAME[1],
            frame_center: c.centre,
            mapping_centre: c.centre,
            background: Background::Color([0.0; 4]),
            camera: Camera::Flat,
            cairo_line_width_multiple: 0.01,
            use_z_index: true,
            light: LightModel::Ce,
            input_grid: 0.0,
        });
        d.upload = vec![shape];
        d.upload_paints = vec![table];
        d.set = vec![r];
        d.order = Some(vec![OrderEntry {
            layer: Layer::Moving,
            slot: 1,
            part: 0,
        }]);
        let mut s = Stream::new();
        s.apply(&d).unwrap();
        let frame = s.draw(c.size[0], c.size[1], 2).unwrap();
        for &(px, py, want) in &c.pixels {
            let at = 4 * (py as usize * c.size[0] as usize + px as usize);
            assert_eq!(frame[at..at + 4], want, "case {n}, pixel ({px}, {py})");
            compared += 1;
        }
    }
    assert_eq!(compared, 24 * 40);
}

/// Cairo paints nothing from a source it finds clear under OVER (`_cairo_surface_fill` and
/// `_cairo_surface_stroke` return at `nothing_to_do`). A linear gradient under pad, as CE builds
/// one, is clear when every stop's alpha, as a 16-bit short (`d * 65535 + 0.5`), is at most 0xff
/// (`_gradient_is_clear`, `CAIRO_ALPHA_SHORT_IS_CLEAR`); pixman would paint such stops at alpha 1.
/// These draw a square path record whose fill or stroke has two stops of different colours, its
/// gradient across the square, alone on a clear frame.
fn square_painted_with(stops: [[f64; 4]; 2], stroke: bool) -> Vec<u8> {
    let (w, h) = (192, 108);
    let shape = Shape::new(ShapeData::Path {
        points: polygon(&[
            [-3.0, -2.0, 0.0],
            [3.0, -2.0, 0.0],
            [3.0, 2.0, 0.0],
            [-3.0, 2.0, 0.0],
        ]),
    });
    let mut r = Record::new(RecordKind::Path, 1, 101);
    r.geometry.shape_a = Some(shape.key());
    if stroke {
        r.paint.stroke = Paint::Two(stops);
        r.paint.stroke_width = 20.0;
    } else {
        r.paint.fill = Paint::Two(stops);
    }
    r.paint.gradient = Some([[-3.0, -2.0, 0.0], [3.0, 2.0, 0.0]]);
    let mut d = Delta::new(Time { num: 0, den: 15 }, 0);
    d.view = Some(View {
        pixel_width: w,
        pixel_height: h,
        frame_width: FRAME[0],
        frame_height: FRAME[1],
        frame_center: [0.0; 3],
        mapping_centre: [0.0; 3],
        background: Background::Color([0.0; 4]),
        camera: Camera::Flat,
        cairo_line_width_multiple: 0.01,
        use_z_index: true,
        light: LightModel::Ce,
        input_grid: 0.0,
    });
    d.upload = vec![shape];
    d.set = vec![r];
    d.order = Some(vec![OrderEntry {
        layer: Layer::Moving,
        slot: 1,
        part: 0,
    }]);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    s.draw(w, h, 2).unwrap()
}

/// Two stops, red and blue, at straight alphas `a` and `b`.
fn red_to_blue(a: f64, b: f64) -> [[f64; 4]; 2] {
    [[1.0, 0.0, 0.0, a], [0.0, 0.0, 1.0, b]]
}

/// The pixels of `frame` that are not clear.
fn painted(frame: &[u8]) -> usize {
    frame.chunks(4).filter(|p| p != &[0; 4]).count()
}

/// The alpha whose short is `s`.
fn at_short(s: u16) -> f64 {
    s as f64 / 65535.0
}

#[test]
fn a_gradient_whose_every_stop_is_clear_paints_nothing() {
    assert_eq!((0.003548f64 * 65535.0 + 0.5) as u16, 233);
    let mut drawn = Vec::new();
    for (a, b) in [
        (0.003548, 0.003548),
        (at_short(0xff), at_short(0xff)),
        (0.0, at_short(0xff)),
        (at_short(0x81), at_short(0xff)),
    ] {
        for stroke in [false, true] {
            let n = painted(&square_painted_with(red_to_blue(a, b), stroke));
            if n > 0 {
                drawn.push(format!("alphas {a} and {b}, stroke {stroke}: {n} px"));
            }
        }
    }
    assert!(drawn.is_empty(), "{drawn:#?}");
}

#[test]
fn a_gradient_with_a_stop_past_clear_paints() {
    // Every stop just past clear, then one stop clear and one not: Cairo draws both.
    for (a, b) in [
        (at_short(0x100), at_short(0x100)),
        (at_short(0xff), at_short(0x100)),
        (0.0, 0.5),
    ] {
        for stroke in [false, true] {
            let n = painted(&square_painted_with(red_to_blue(a, b), stroke));
            assert!(n > 0, "alphas {a} and {b}, stroke {stroke}");
        }
    }
}

/// A 3D frame of an opaque square at z = 0 and, when `crossing` is given, a square through it
/// (on the plane z = x) filled with those two stops, drawn after it.
fn crossed_by(crossing: Option<[[f64; 4]; 2]>) -> Stream {
    let square = |corners: [[f64; 3]; 4]| {
        Shape::new(ShapeData::Path {
            points: polygon(&corners),
        })
    };
    let flat = square([
        [-2.0, -2.0, 0.0],
        [2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
        [-2.0, 2.0, 0.0],
    ]);
    let mut a = Record::new(RecordKind::Path, 1, 101);
    a.geometry.shape_a = Some(flat.key());
    a.paint.fill = Paint::One([0.2, 0.4, 0.9, 1.0]);
    a.depth.space = Space::World;
    let mut d = Delta::new(Time { num: 0, den: 15 }, 0);
    d.view = Some(View {
        pixel_width: 192,
        pixel_height: 108,
        frame_width: FRAME[0],
        frame_height: FRAME[1],
        frame_center: [0.0; 3],
        mapping_centre: [0.0; 3],
        background: Background::Color([0.0; 4]),
        camera: Camera::ThreeD(ThreeDCamera {
            phi: 1.2217304763960306,
            theta: -std::f64::consts::FRAC_PI_4,
            gamma: 0.1,
            focal_distance: 20.0,
            zoom: 1.0,
            exponential_projection: false,
            light_source: [-7.0, -9.0, 10.0],
            shading_factor: 0.2,
            should_apply_shading: true,
        }),
        cairo_line_width_multiple: 0.01,
        use_z_index: true,
        light: LightModel::Ce,
        input_grid: 0.0,
    });
    let mut order = vec![OrderEntry {
        layer: Layer::Moving,
        slot: 1,
        part: 0,
    }];
    d.upload = vec![flat];
    d.set = vec![a];
    if let Some(stops) = crossing {
        let tilted = square([
            [-1.5, -1.5, -1.5],
            [1.5, -1.5, 1.5],
            [1.5, 1.5, 1.5],
            [-1.5, 1.5, -1.5],
        ]);
        let mut b = Record::new(RecordKind::Path, 2, 102);
        b.geometry.shape_a = Some(tilted.key());
        b.paint.fill = Paint::Two(stops);
        b.paint.gradient = Some([[-1.5, 0.0, -1.5], [1.5, 0.0, 1.5]]);
        b.depth.space = Space::World;
        d.upload.push(tilted);
        d.set.push(b);
        order.push(OrderEntry {
            layer: Layer::Moving,
            slot: 2,
            part: 0,
        });
    }
    d.order = Some(order);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    s
}

#[test]
fn a_clear_gradient_through_another_leaf_leaves_its_pixels_resolved_as_without_it() {
    // Two stops whose colours differ, so they do not collapse to one solid colour. No layer of
    // theirs may take part in the depth order or the resolve, even where they would paint 0.
    let alone = crossed_by(None);
    let (w, h) = (192, 108);
    let want = alone.render(w, h, 2).unwrap();
    let want_leaves = alone.contributors(0, 0, w, h).unwrap();
    for (a, b) in [(0.0, 0.0), (at_short(0xff), at_short(0xff))] {
        let s = crossed_by(Some(red_to_blue(a, b)));
        let got = s.render(w, h, 2).unwrap();
        let leaves = s.contributors(0, 0, w, h).unwrap();
        let moved: Vec<_> = leaves
            .iter()
            .filter(|(at, p)| want_leaves.get(at) != Some(p))
            .map(|(at, p)| (*at, p.crossing))
            .collect();
        assert!(
            moved.is_empty(),
            "alphas {a} and {b}: {} pixels' contributors differ, {} of them crossings, first {:?}",
            moved.len(),
            moved.iter().filter(|m| m.1).count(),
            moved.first()
        );
        assert_eq!(leaves.len(), want_leaves.len(), "alphas {a} and {b}");
        assert!(
            got.resolves == want.resolves,
            "alphas {a} and {b}: resolves"
        );
        assert!(got.rgba == want.rgba, "alphas {a} and {b}: pixels");
    }
}
