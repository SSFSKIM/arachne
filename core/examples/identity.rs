//! Writes what the core computes on a fixed set of inputs, as raw bytes, into a directory, so two
//! builds (a baseline x86-64 and an x86-64-v3 build) can be compared byte for byte
//! (`core/ci/identity.sh`): every frame the core draws today, CE's projection over a grid of
//! cameras and points, and an XF file with its content keys.
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
    out
}

/// Every camera of a grid projects every point of a grid, through the Cairo frame mapping.
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
