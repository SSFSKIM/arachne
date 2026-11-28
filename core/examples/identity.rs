//! Writes what the core computes on a fixed set of inputs, as raw bytes, into a directory, so two
//! builds (a baseline x86-64 and an x86-64-v3 build) can be compared byte for byte
//! (`core/ci/identity.sh`): backgrounds, mesh scenes (lit, translucent, intersecting, flat) and
//! every frame of the class probes (`parity/probes.rs`: paths, strokes and meshes by depth), and
//! instances records (a run of arrows through parts and widths, lit spheres at many placements, under
//! each light model), with each pixel's resolve, drawn on one thread and on four, CE's projection over a grid of cameras
//! and points, and an XF file with its content keys.
//!
//! Usage: `cargo run --release --example identity -- <out-dir>`

#[path = "../tests/common/mod.rs"]
mod common;

use arachne_core::project::{FrameMap, Placement, Projection, pre_display};
use arachne_core::stream::Stream;
use arachne_core::xf::*;
use std::path::PathBuf;

fn frames() -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for (name, view) in [
        (
            "background-grey",
            common::flat_view(854, 480, [0.5, 0.5, 0.5, 1.0]),
        ),
        (
            "background-translucent",
            common::flat_view(320, 180, [0.2, 0.4, 0.6, 0.3]),
        ),
        ("background-3d", common::three_d_view(1920, 1080)),
    ] {
        let mut stream = Stream::new();
        let mut delta = Delta::new(Time { num: 0, den: 1 }, 0);
        delta.view = Some(view.clone());
        stream.apply(&delta).expect("a valid delta");
        let frame = stream
            .draw(view.pixel_width, view.pixel_height, 1)
            .expect("the core draws backgrounds");
        out.push((name.to_string(), frame));
    }
    let (w, h) = (480, 270);
    for (name, stream) in common::meshes::scenes(w, h) {
        let one = stream.render(w, h, 1).expect("the core draws meshes");
        let four = stream.render(w, h, 4).expect("the core draws meshes");
        assert!(one == four, "{name}: one thread and four differ");
        out.push((format!("mesh-{name}"), one.rgba));
        out.push((format!("mesh-{name}-resolves"), one.resolves));
    }
    for (name, stream) in instance_scenes() {
        let (w, h) = (480, 270);
        let one = stream.render(w, h, 1).expect("the core draws instances");
        let four = stream.render(w, h, 4).expect("the core draws instances");
        assert!(one == four, "{name}: one thread and four differ");
        out.push((format!("instances-{name}"), one.rgba));
        out.push((format!("instances-{name}-resolves"), one.resolves));
    }
    for probe in arachne_core::probes::all() {
        let mut stream = Stream::new();
        for (i, delta) in probe.deltas.iter().enumerate() {
            stream.apply(delta).expect("a probe's delta applies");
            let (w, h) = (probe.width, probe.height);
            let one = stream.render(w, h, 1).expect("the core draws the probes");
            let four = stream.render(w, h, 4).expect("the core draws the probes");
            assert!(
                one == four,
                "{} frame {i}: one thread and four differ",
                probe.name
            );
            out.push((format!("probe-{}-{i}", probe.name), one.rgba));
            out.push((format!("probe-{}-{i}-resolves", probe.name), one.resolves));
        }
    }
    out
}

/// Every camera of a grid projects every point of a grid, through the Cairo frame mapping.
/// Straight segments through `points` in CE's cubic layout.
fn lines(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

/// A turn by `angle` about z, a scale by `s`, then a shift.
fn placed(angle: f64, s: f64, shift: [f64; 3]) -> Affine {
    let (c, n) = (libm::cos(angle) * s, libm::sin(angle) * s);
    [
        [c, -n, 0.0, shift[0]],
        [n, c, 0.0, shift[1]],
        [0.0, 0.0, s, shift[2]],
    ]
}

/// The instances scenes, each under CE's light and soft diffuse light: 60 arrows (a shaft and a
/// tip a group, each shaft at its own width, overlapping, translucent in places) and 400 lit
/// spheres on orbits, both in world space under a 3D camera.
fn instance_scenes() -> Vec<(String, Stream)> {
    let shaft = Shape::new(ShapeData::Path {
        points: lines(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    });
    let tip = Shape::new(ShapeData::Path {
        points: lines(&[
            [1.0, -0.25, 0.0],
            [1.45, 0.0, 0.0],
            [1.0, 0.25, 0.0],
            [1.0, -0.25, 0.0],
        ]),
    });
    let ball = common::meshes::sphere([0.0, 0.0, 0.0], 1.0, 8, 16);
    let mut found = Vec::new();
    for (light, model) in [
        ("ce", LightModel::Ce),
        (
            "soft",
            LightModel::SoftDiffuse {
                ambient: 0.35,
                wrap: 0.4,
            },
        ),
    ] {
        let mut view = common::three_d_view(480, 270);
        view.light = model;
        let colour = |i: usize| {
            let t = i as f64 * 0.37;
            let alpha = if i % 3 == 1 { 0.55 } else { 1.0 };
            [
                0.5 + 0.5 * libm::sin(t),
                0.5 + 0.5 * libm::sin(t + 2.0),
                0.5 + 0.5 * libm::sin(t + 4.0),
                alpha,
            ]
        };
        // Arrows: member 2g is arrow g's shaft, 2g + 1 its tip.
        let arrows = 60;
        let mut placements = Vec::new();
        let mut widths = Vec::new();
        for g in 0..arrows {
            let t = g as f64;
            let place = placed(
                0.7 * t,
                0.6 + 0.02 * t,
                [-4.0 + 0.13 * t, 2.5 * libm::sin(0.4 * t), 0.0],
            );
            placements.extend([place, place]);
            widths.extend([1.0 + 0.1 * t, 3.0]);
        }
        let mut arrow = Record::new(RecordKind::Instances, 1, 1);
        arrow.geometry.shape_a = Some(shaft.key());
        arrow.paint.stroke = Paint::One([1.0; 4]);
        arrow.paint.stroke_width = 6.0;
        arrow.paint.joint = Joint::Miter;
        arrow.paint.cap = Cap::Square;
        arrow.depth.space = Space::World;
        arrow.features |= FEATURE_MEMBERS;
        let rows = Stops::new((0..placements.len()).map(colour).collect());
        arrow.body = Body::Instances(Instances {
            placements: placements.clone(),
            fill: rows.key(),
            stroke: rows.key(),
            objects: (0..placements.len() as u32).map(|i| 10 + i).collect(),
            parts: vec![Part {
                shape: tip.key(),
                paint: PaintBlock {
                    fill: Paint::One([1.0; 4]),
                    stroke: Paint::One([1.0; 4]),
                    stroke_width: 3.0,
                    background_stroke: Paint::One([0.1, 0.1, 0.4, 0.8]),
                    background_stroke_width: 7.0,
                    joint: Joint::Bevel,
                    cap: Cap::Round,
                    ..PaintBlock::default()
                },
            }],
            widths,
        });
        // Spheres: orbits about the centre at three heights.
        let spheres = 400;
        let orbit: Vec<Affine> = (0..spheres)
            .map(|i| {
                let t = i as f64;
                let r = 0.8 + 3.0 * (t / spheres as f64);
                let a = 2.39996 * t;
                let mut p = placed(0.0, 0.15, [r * libm::cos(a), r * libm::sin(a), 0.0]);
                p[2][3] = 0.4 * ((i % 3) as f64 - 1.0);
                p
            })
            .collect();
        let tones = Stops::new((0..spheres).map(|i| colour(i + 1)).collect());
        let mut lit = Record::new(RecordKind::Instances, 2, 2);
        lit.geometry.shape_a = Some(ball.key());
        lit.paint.fill = Paint::One([1.0; 4]);
        lit.depth.space = Space::World;
        lit.depth.shade_in_3d = true;
        lit.body = Body::Instances(Instances {
            placements: orbit,
            fill: tones.key(),
            stroke: tones.key(),
            objects: (0..spheres as u32).map(|i| 1000 + i).collect(),
            parts: Vec::new(),
            widths: Vec::new(),
        });
        for (name, shapes, record, table) in [
            ("arrows", vec![shaft.clone(), tip.clone()], arrow, rows),
            ("spheres", vec![ball.clone()], lit, tones),
        ] {
            let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
            d.view = Some(view.clone());
            d.upload = shapes;
            d.upload_paints = vec![table];
            d.set.push(record);
            d.order = Some(vec![OrderEntry {
                layer: Layer::Moving,
                slot: d.set[0].slot,
                part: 0,
            }]);
            let mut stream = Stream::new();
            stream.apply(&d).expect("an instances delta applies");
            found.push((format!("{name}-{light}"), stream));
        }
    }
    found
}

fn projection() -> Vec<u8> {
    let mut bytes = Vec::new();
    let angles = [-2.5, -1.0, -0.3, 0.0, 0.0004, 0.7, 1.3, 3.0];
    for (i, phi) in angles.iter().enumerate() {
        for (j, theta) in angles.iter().enumerate() {
            let camera = ThreeDCamera {
                phi: *phi,
                theta: *theta,
                gamma: angles[(i + j) % angles.len()],
                focal_distance: 4.0 + i as f64,
                zoom: 0.5 + j as f64 * 0.25,
                exponential_projection: (i + j) % 2 == 0,
                light_source: [-7.0, -9.0, 10.0],
                shading_factor: 0.2,
                should_apply_shading: true,
            };
            let centre = [0.25 * i as f64, -0.5, 0.125 * j as f64];
            let projection = Projection::new(&camera, centre);
            let map = FrameMap::cairo(1920, 1080, 14.222222222222221, 8.0, centre);
            for x in -4..=4 {
                for y in -3..=3 {
                    for z in -2..=6 {
                        let p = [x as f64 * 1.37, y as f64 * 0.91, z as f64 * 1.73];
                        let shown = pre_display(&[p], Some(&projection), Placement::Projected);
                        for v in map.apply(shown[0]).iter().chain(&[shown[0][2]]) {
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            }
        }
    }
    bytes
}

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect("usage: identity <out-dir>"));
    std::fs::create_dir_all(&out).expect("the output directory");
    for (name, frame) in frames() {
        std::fs::write(out.join(format!("{name}.rgba")), frame).expect("a frame file");
    }
    std::fs::write(out.join("projection.f64"), projection()).expect("the projection file");
    wire::write_file(&out.join("sample.xf"), &common::sample_deltas()).expect("the XF file");
}
