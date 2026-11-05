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
