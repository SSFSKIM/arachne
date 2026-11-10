//! Streams for the oracle's tests, built from the XF types alone.
#![allow(dead_code)]

use arachne_core::xf::{
    Background, Camera, Delta, Key, Layer, LightModel, OrderEntry, Paint, Record, RecordKind,
    Shape, ShapeData, Space, Time, View,
};
use arachne_oracle::decide::decide_at;
use arachne_oracle::out::{CoverRow, Decisions};

/// A flat view of `w` x `h` pixels whose frame units are pixels, centred: `xx = 1`, `yy = -1`,
/// pixel `(px, py)` is frame point `(px - w/2, h/2 - py)`.
pub fn view(w: u32, h: u32) -> View {
    View {
        pixel_width: w,
        pixel_height: h,
        frame_width: w as f64,
        frame_height: h as f64,
        frame_center: [0.0; 3],
        mapping_centre: [0.0; 3],
        background: Background::Color([0.0, 0.0, 0.0, 1.0]),
        camera: Camera::Flat,
        cairo_line_width_multiple: 1.0,
        use_z_index: false,
        light: LightModel::Ce,
        input_grid: 0.0,
    }
}

/// A pixel position in the frame units of [`view`].
pub fn frame(v: &View, p: [f64; 2]) -> [f64; 3] {
    [
        p[0] - v.pixel_width as f64 / 2.0,
        v.pixel_height as f64 / 2.0 - p[1],
        0.0,
    ]
}

/// A polygon's outline through `corners` in CE's cubic layout, each side a straight cubic.
pub fn polygon(corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for (i, a) in corners.iter().enumerate() {
        let b = corners[(i + 1) % corners.len()];
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([*a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

/// An open polyline in CE's cubic layout.
pub fn polyline(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

/// A circle of radius `r` about `c`, as CE's eight cubic arcs.
pub fn circle(c: [f64; 3], r: f64) -> Vec<[f64; 3]> {
    let n = 8;
    let k = 4.0 / 3.0 * libm::tan(std::f64::consts::PI / (2.0 * n as f64));
    let mut out = Vec::new();
    for i in 0..n {
        let a0 = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let a1 = 2.0 * std::f64::consts::PI * (i + 1) as f64 / n as f64;
        let p = |a: f64| [c[0] + r * libm::cos(a), c[1] + r * libm::sin(a), 0.0];
        let t = |a: f64| [-libm::sin(a), libm::cos(a)];
        let (p0, p3) = (p(a0), p(a1));
        let (t0, t1) = (t(a0), t(a1));
        out.extend([
            p0,
            [p0[0] + k * r * t0[0], p0[1] + k * r * t0[1], 0.0],
            [p3[0] - k * r * t1[0], p3[1] - k * r * t1[1], 0.0],
            p3,
        ]);
    }
    out
}

pub fn path_shape(points: Vec<[f64; 3]>) -> Shape {
    Shape::new(ShapeData::Path { points })
}

/// A path record in slot `slot` drawing `shape`, filled white (or not), stroked white at `stroke`
/// (frame units, or not).
pub fn path(slot: u32, shape: Key, fill: bool, stroke: Option<f64>) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape);
    if fill {
        r.paint.fill = Paint::One([1.0, 1.0, 1.0, 1.0]);
    }
    if let Some(w) = stroke {
        r.paint.stroke = Paint::One([1.0, 1.0, 1.0, 1.0]);
        r.paint.stroke_width = w;
    }
    r.depth.space = Space::Screen;
    r
}

pub fn entry(slot: u32) -> OrderEntry {
    OrderEntry {
        layer: Layer::Moving,
        slot,
        part: 0,
    }
}

/// A delta at `seq` setting the view, uploading `shapes`, setting `records` and ordering them.
pub fn delta(seq: u64, view: Option<View>, shapes: Vec<Shape>, records: Vec<Record>) -> Delta {
    let mut d = Delta::new(
        Time {
            num: seq as i64,
            den: 1,
        },
        seq,
    );
    d.version = (1, 4);
    d.view = view;
    d.order = Some(records.iter().map(|r| entry(r.slot)).collect());
    d.upload = shapes;
    d.set = records;
    d
}

/// The decisions at the last delta of `deltas`.
pub fn decide(deltas: &[Delta]) -> Decisions {
    decide_at(deltas, deltas.last().unwrap().seq, None).unwrap()
}

/// The decisions' fragment and cover lines in canonical text: what two frames compare by.
pub fn cover_lines(d: &Decisions) -> Vec<String> {
    arachne_oracle::out::canonical(d)
        .lines()
        .filter(|l| l.starts_with("fragment ") || l.starts_with("cover "))
        .map(str::to_owned)
        .collect()
}

/// The cover rows of the fragment whose layer is `layer`, by pixel.
pub fn rows_of(d: &Decisions, layer: u8) -> Vec<&CoverRow> {
    let ids: Vec<u32> = d
        .fragments
        .iter()
        .filter(|f| f.layer == layer)
        .map(|f| f.id)
        .collect();
    d.covers
        .iter()
        .filter(|c| ids.contains(&c.fragment))
        .collect()
}
