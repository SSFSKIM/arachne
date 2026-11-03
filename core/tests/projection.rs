//! CE's camera maths against values Arachne's own Camera and ThreeDCamera computed
//! (`tests/data/projection.txt`, written by `tests/data/gen_projection.py`).
//!
//! The core computes with plain IEEE arithmetic and the pure-Rust libm, while NumPy multiplies
//! through BLAS and SciPy and NumPy take sin, cos and exp from the platform, so the last bits may
//! differ. The bound is relative, far below a pixel. Set `PROJECTION_STATS=1` to print how many
//! values match bit for bit.

use arachne_core::project::{FrameMap, Placement, Projection, pre_display, subpixel_coords};
use arachne_core::xf::ThreeDCamera;
use std::cell::Cell;

const REL: f64 = 1e-13;

struct Stats {
    values: Cell<usize>,
    exact: Cell<usize>,
    worst: Cell<f64>,
}

impl Stats {
    fn check(&self, what: &str, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (g, w) in got.iter().zip(want) {
            self.values.set(self.values.get() + 1);
            if g.to_bits() == w.to_bits() || (*g == 0.0 && *w == 0.0) {
                self.exact.set(self.exact.get() + 1);
                continue;
            }
            let err = (g - w).abs() / w.abs().max(1.0);
            self.worst.set(self.worst.get().max(err));
            assert!(
                err <= REL,
                "{what}: got {g:?}, want {w:?} (relative {err:e})"
            );
        }
    }
}

fn nums(tokens: &[&str]) -> Vec<f64> {
    tokens
        .iter()
        .filter(|t| **t != "->")
        .map(|t| t.parse::<f64>().unwrap_or_else(|_| panic!("number: {t}")))
        .collect()
}

fn v3(v: &[f64]) -> [f64; 3] {
    [v[0], v[1], v[2]]
}

#[test]
fn matches_arachnes_cameras() {
    let text = include_str!("data/projection.txt");
    let stats = Stats {
        values: Cell::new(0),
        exact: Cell::new(0),
        worst: Cell::new(0.0),
    };
    let mut projection: Option<Projection> = None;
    let mut map: Option<FrameMap> = None;
    let mut frame = (0u32, 0u32, 0.0f64, 0.0f64, [0.0f64; 3]);
    let mut lines = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        lines += 1;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "camera" => {
                let v = nums(&tokens[2..]);
                let camera = ThreeDCamera {
                    phi: v[0],
                    theta: v[1],
                    gamma: v[2],
                    focal_distance: v[3],
                    zoom: v[4],
                    exponential_projection: v[5] != 0.0,
                    light_source: [-7.0, -9.0, 10.0],
                    shading_factor: 0.2,
                    should_apply_shading: true,
                };
                let centre = v3(&v[6..9]);
                projection = Some(Projection::new(&camera, centre));
                frame = (v[9] as u32, v[10] as u32, v[11], v[12], centre);
                map = Some(FrameMap::cairo(frame.0, frame.1, frame.2, frame.3, centre));
            }
            "flat" => {
                let v = nums(&tokens[1..]);
                let centre = v3(&v[0..3]);
                projection = None;
                frame = (v[3] as u32, v[4] as u32, v[5], v[6], centre);
                map = Some(FrameMap::cairo(frame.0, frame.1, frame.2, frame.3, centre));
            }
            "rotation" => {
                let r = projection.as_ref().unwrap().rotation();
                let got: Vec<f64> = r.iter().flatten().copied().collect();
                stats.check(line, &got, &nums(&tokens[1..]));
            }
            "matrix" => {
                // Cairo's (xx, yx, xy, yy, x0, y0).
                let m = map.as_ref().unwrap();
                stats.check(
                    line,
                    &[m.xx, 0.0, 0.0, m.yy, m.x0, m.y0],
                    &nums(&tokens[1..]),
                );
            }
            "project" => {
                let v = nums(&tokens[1..]);
                let got = projection.as_ref().unwrap().project(v3(&v[0..3]));
                stats.check(line, &got, &v[3..6]);
            }
            "device" => {
                let v = nums(&tokens[1..]);
                let placement = match projection {
                    Some(_) => Placement::Projected,
                    None => Placement::Flat,
                };
                let shown = pre_display(&[v3(&v[0..3])], projection.as_ref(), placement);
                let got = map.as_ref().unwrap().apply(shown[0]);
                stats.check(line, &got, &v[3..5]);
            }
            "subpixel" => {
                let v = nums(&tokens[1..]);
                let placement = match projection {
                    Some(_) => Placement::Projected,
                    None => Placement::Flat,
                };
                let shown = pre_display(&[v3(&v[0..3])], projection.as_ref(), placement);
                let got = subpixel_coords(&shown, frame.0, frame.1, frame.2, frame.3, frame.4);
                stats.check(line, &got[0], &v[3..5]);
            }
            "fixed_orientation" => {
                let v = nums(&tokens[1..]);
                let shown = pre_display(
                    &[v3(&v[3..6])],
                    projection.as_ref(),
                    Placement::FixedOrientation(v3(&v[0..3])),
                );
                stats.check(line, &shown[0], &v[6..9]);
            }
            "fixed_in_frame" => {
                let v = nums(&tokens[1..]);
                let shown = pre_display(
                    &[v3(&v[0..3])],
                    projection.as_ref(),
                    Placement::FixedInFrame,
                );
                stats.check(line, &shown[0], &v[3..6]);
            }
            "nonfinite" => {
                // [[nan, 0, 0], [1, 2, 3]] becomes CE's single origin point, then is projected.
                let shown = pre_display(
                    &[[f64::NAN, 0.0, 0.0], [1.0, 2.0, 3.0]],
                    projection.as_ref(),
                    Placement::Projected,
                );
                assert_eq!(shown.len(), 1, "{line}");
                stats.check(line, &shown[0], &nums(&tokens[1..]));
            }
            other => panic!("unknown line kind {other}"),
        }
    }
    assert!(lines > 400, "the fixture is complete ({lines} lines)");
    if std::env::var_os("PROJECTION_STATS").is_some() {
        eprintln!(
            "projection: {} values, {} bit-exact, worst relative error {:e}",
            stats.values.get(),
            stats.exact.get(),
            stats.worst.get()
        );
    }
}

#[test]
fn a_point_behind_the_eye_takes_ces_large_factor() {
    let camera = ThreeDCamera {
        phi: 0.0,
        theta: -std::f64::consts::FRAC_PI_2,
        gamma: 0.0,
        focal_distance: 5.0,
        zoom: 2.0,
        exponential_projection: false,
        light_source: [0.0; 3],
        shading_factor: 0.2,
        should_apply_shading: true,
    };
    let p = Projection::new(&camera, [0.0; 3]).project([1.0, -0.5, 6.0]);
    assert_eq!(p, [2e6, -1e6, 6.0]);
}
