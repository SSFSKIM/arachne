//! Switching resolves. As geometry moves, a pixel can take the coverage resolve in one frame and
//! the sample resolve in the next; where a path's edge crosses it, the switch must not show as a
//! jump. In the frame where such a pixel took the coverage resolve, its bytes are held to the
//! sample resolve of the same frame's geometry (the frame drawn again with
//! [`common::resolve::EverySample`]): that difference is the jump the switch itself causes, beyond
//! what the motion explains.

mod common;

use arachne_core::composite::{COVERAGE_TO_SAMPLE_LSB, Resolve};
use arachne_core::probes::{first_delta, mesh_record, polyline, sheet, stroke_record, view};
use arachne_core::stream::Stream;
use common::resolve::EverySample;

const W: u32 = 320;
const H: u32 = 180;
const FRAMES: usize = 31;

/// Stroke alpha and sheet alpha: both opaque, a translucent stroke, a translucent sheet.
const CASES: [(f64, f64); 3] = [(1.0, 1.0), (0.6, 1.0), (1.0, 0.7)];

/// A sheet spanning x from -4 to `right` and y from -3 to 3, tilting about the x axis from -6 to
/// +6 degrees over the run, and a stroke `width` px wide from (-3, `y`, -1) to (3, `y`, 1), so the
/// crossing slides along the stroke.
#[derive(Clone, Copy)]
struct Scene {
    right: f64,
    y: f64,
    width: f64,
}

/// The stroke through the sheet's middle.
const MIDDLE: Scene = Scene {
    right: 4.0,
    y: 0.5,
    width: 40.0,
};

fn scene(frame: usize, stroke_alpha: f64, sheet_alpha: f64, at: Scene) -> Stream {
    let deg = std::f64::consts::PI / 180.0;
    let a = (-6.0 + 0.4 * frame as f64) * deg;
    let (c, s) = (libm::cos(a), libm::sin(a));
    let corner = |x: f64, y: f64| [x, y * c, y * s];
    let floor = sheet(
        [
            corner(-4.0, -3.0),
            corner(at.right, -3.0),
            corner(at.right, 3.0),
            corner(-4.0, 3.0),
        ],
        8,
        6,
    );
    let line = polyline(&[[-3.0, at.y, -1.0], [3.0, at.y, 1.0]]);
    let items = [
        (
            floor.clone(),
            mesh_record(1, &floor, [0.1, 0.15, 0.45, sheet_alpha], false),
        ),
        (
            line.clone(),
            stroke_record(2, &line, [0.95, 0.5, 0.1, stroke_alpha], at.width),
        ),
    ];
    let mut stream = Stream::new();
    stream
        .apply(&first_delta(
            view(W, H, 60.0 * deg, -35.0 * deg),
            &items,
            &[1, 2],
        ))
        .unwrap();
    stream
}

/// Pixels that switch between the coverage and the sample resolve from one frame to the next, and
/// the largest jump the switch causes at one of them (frame, pixel), in any channel.
struct Switches {
    pixels: usize,
    worst: i32,
    at: (usize, u32, u32),
}

fn switches(stroke_alpha: f64, sheet_alpha: f64, at: Scene) -> Switches {
    let frames: Vec<_> = (0..FRAMES)
        .map(|f| {
            let s = scene(f, stroke_alpha, sheet_alpha, at);
            let drawn = s.render(W, H, 1).unwrap();
            let sampled = s.render_with(W, H, 1, &[&EverySample]).unwrap().rgba;
            (drawn, sampled)
        })
        .collect();
    let (coverage, sample) = (Resolve::Coverage as u8, Resolve::Sample as u8);
    let mut out = Switches {
        pixels: 0,
        worst: 0,
        at: (0, 0, 0),
    };
    for f in 1..FRAMES {
        for i in 0..(W * H) as usize {
            let (a, b) = (frames[f - 1].0.resolves[i], frames[f].0.resolves[i]);
            let side = match (a, b) {
                _ if a == coverage && b == sample => f - 1,
                _ if a == sample && b == coverage => f,
                _ => continue,
            };
            out.pixels += 1;
            let (drawn, sampled) = &frames[side];
            let jump = (0..4)
                .map(|k| (drawn.rgba[i * 4 + k] as i32 - sampled[i * 4 + k] as i32).abs())
                .max()
                .unwrap();
            if jump > out.worst {
                out.worst = jump;
                out.at = (side, i as u32 % W, i as u32 / W);
            }
        }
    }
    if std::env::var_os("CORE_STATS").is_some() {
        eprintln!(
            "stroke alpha {stroke_alpha}, sheet alpha {sheet_alpha}, sheet to x = {}, stroke at \
             y = {} {} px wide: {} switching pixels, worst jump {} (frame {}, pixel {},{})",
            at.right, at.y, at.width, out.pixels, out.worst, out.at.0, out.at.1, out.at.2
        );
    }
    out
}

#[test]
fn a_path_edge_lands_within_the_bound_whichever_resolve_its_pixel_takes() {
    for (stroke_alpha, sheet_alpha) in CASES {
        let s = switches(stroke_alpha, sheet_alpha, MIDDLE);
        assert!(s.pixels > 10, "{} switching pixels", s.pixels);
        assert!(
            s.worst <= COVERAGE_TO_SAMPLE_LSB as i32,
            "stroke alpha {stroke_alpha}, sheet alpha {sheet_alpha}: a jump of {} at frame {}, \
             pixel {},{}",
            s.worst,
            s.at.0,
            s.at.1,
            s.at.2
        );
    }
}

/// The largest jumps measured where the sheet's silhouette, the stroke's edge and the crossing
/// share pixels, recorded beside `COVERAGE_TO_SAMPLE_LSB`, not bounded by it: the coverage resolve
/// composites the silhouette's mask count and the path's area as if they were independent inside
/// the pixel (as Cairo composites two shapes), where the sample resolve sees where they overlap.
/// Across each other that costs little; along each other, a stroke on a surface's edge, it is the
/// whole difference between the edge showing the background through and not.
const ACROSS_JUMP: i32 = 2;
const ALONG_JUMP: i32 = 48;

fn worst_of(scenes: impl Iterator<Item = Scene>) -> i32 {
    let mut worst = 0;
    for at in scenes {
        for (stroke_alpha, sheet_alpha) in CASES {
            worst = worst.max(switches(stroke_alpha, sheet_alpha, at).worst);
        }
    }
    worst
}

#[test]
fn where_a_silhouette_a_path_edge_and_a_crossing_share_a_pixel_the_jump_is_recorded() {
    // Across: the sheet ends inside the crossing's sweep (x = 1.5 tan(a), within 0.16), the
    // stroke's edges crossing its silhouette.
    let across = worst_of((0..3).map(|k| Scene {
        right: -0.05 + 0.05 * k as f64,
        ..MIDDLE
    }));
    // Along: a thin stroke running beside the sheet's edge at y = 3 (the crossing sweeps along
    // it, x = 9 sin(a)), its edge on the silhouette at one of the offsets.
    let along = worst_of((0..4).map(|k| Scene {
        right: 4.0,
        y: 2.92 + 0.04 * k as f64,
        width: 6.0,
    }));
    if std::env::var_os("CORE_STATS").is_some() {
        eprintln!("silhouette, path edge and crossing: across {across}, along {along}");
    }
    assert!(
        across <= ACROSS_JUMP && along <= ALONG_JUMP,
        "jumps of {across} across and {along} along, past the {ACROSS_JUMP} and {ALONG_JUMP} \
         recorded"
    );
}
