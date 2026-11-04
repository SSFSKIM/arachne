//! The class probes (`parity/probes.rs`), asserted: at each check's pixel the core draws what the
//! probe's own oracle expects and not what CE's painter order draws; the seam probe's interior
//! shows no background through its internal edges, opaque or translucent; the stroke crossing a
//! surface is antialiased where they meet; every probe draws the same bytes on every run and on 1,
//! 2 and 4 threads; and the orbit reports its time per frame. Set `CORE_STATS=1` to print each
//! probe's share of pixels per resolve.

mod common;

use arachne_core::composite::{Rendered, Resolve, blend, premultiply};
use arachne_core::probes::{self, Check, Expect, Flat, Probe};
use arachne_core::stream::Stream;
use arachne_core::xf::*;
use common::meshes::{distance_to_segment, grid_points, pixels_inside};
use std::time::Instant;

fn pixel(frame: &Rendered, width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = (y * width + x) as usize * 4;
    frame.rgba[i..i + 4].try_into().unwrap()
}

/// Every frame of a probe, drawn on `threads` threads.
fn frames(probe: &Probe, threads: usize) -> Vec<Rendered> {
    let mut s = Stream::new();
    probe
        .deltas
        .iter()
        .map(|d| {
            s.apply(d).expect("a probe's delta applies");
            s.render(probe.width, probe.height, threads)
                .expect("the core draws every probe")
        })
        .collect()
}

/// A probe's frame drawn without the record in `slot` (removed from the order).
fn frame_without(probe: &Probe, frame: usize, slot: u32) -> Rendered {
    let mut s = Stream::new();
    for d in &probe.deltas[..=frame] {
        s.apply(d).unwrap();
    }
    let mut d = Delta::new(Time { num: 0, den: 1 }, 99);
    d.order = Some(
        s.order()
            .iter()
            .copied()
            .filter(|e| e.slot != slot)
            .collect(),
    );
    s.apply(&d).unwrap();
    s.render(probe.width, probe.height, 1).unwrap()
}

fn assert_check(probe: &Probe, drawn: &[Rendered], c: &Check) {
    let got = pixel(&drawn[c.frame], probe.width, c.x, c.y);
    let want = match c.core {
        Expect::Bytes(b) => b,
        Expect::Hidden(slot) => pixel(&frame_without(probe, c.frame, slot), probe.width, c.x, c.y),
    };
    assert_eq!(
        got, want,
        "{}: frame {} pixel {},{}: {}",
        probe.name, c.frame, c.x, c.y, c.what
    );
    let ce_right = c.what.ends_with("(CE right)");
    assert_eq!(
        want == c.ce,
        ce_right,
        "{}: {}: the core {:?}, CE {:?}",
        probe.name,
        c.what,
        want,
        c.ce
    );
}

fn run(probe: Probe) {
    let drawn = frames(&probe, 1);
    assert!(
        probe.checks.iter().any(|c| !c.what.ends_with("(CE right)")),
        "{}: a probe asserts a pixel CE draws wrong",
        probe.name
    );
    for c in &probe.checks {
        assert_check(&probe, &drawn, c);
    }
    if std::env::var_os("CORE_STATS").is_some() {
        for (i, f) in drawn.iter().enumerate() {
            let c = f.counts;
            println!(
                "{} frame {i}: background {} fast {} coverage {} sample {}",
                probe.name, c.background, c.fast, c.coverage, c.sample
            );
        }
    }
}

#[test]
fn intersecting_surfaces_each_show_where_they_are_in_front() {
    run(probes::intersecting_surfaces());
}

#[test]
fn a_curve_passing_behind_a_surface_is_hidden_there() {
    run(probes::curve_behind_surface());
}

#[test]
fn transparent_surfaces_over_an_opaque_one_composite_in_depth_order() {
    run(probes::translucent_over_opaque());
}

#[test]
fn a_label_fixed_in_frame_stays_on_top_and_a_placed_one_obeys_depth() {
    run(probes::fixed_label_over_surface());
}

#[test]
fn the_depth_rule_holds_in_both_directions() {
    run(probes::depth_rule_both_ways());
}

#[test]
fn a_surface_seen_edge_on_covers_nothing_and_a_ring_nearly_in_its_plane_shows_its_near_arc() {
    run(probes::edge_on_surface());
}

#[test]
fn an_orbiting_camera_keeps_the_ring_behind_the_sphere_in_every_frame() {
    let probe = probes::orbit(8, probes::WIDTH, probes::HEIGHT);
    assert_eq!(probe.checks.len(), 8, "every frame is checked");
    run(probe);
}

#[test]
fn the_orbit_reports_its_time_per_frame() {
    // Indicative only: whatever machine runs the tests. 854 x 480, one thread and two.
    let (w, h) = (854, 480);
    let probe = probes::orbit(24, w, h);
    for threads in [1, 2] {
        let mut s = Stream::new();
        let mut times = Vec::new();
        for d in &probe.deltas {
            s.apply(d).unwrap();
            let t = Instant::now();
            let frame = s.render(w, h, threads).unwrap();
            times.push(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(frame.rgba.len(), (w * h * 4) as usize);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "orbit at {w}x{h}, {threads} thread(s): median {:.1} ms per frame over {} frames",
            times[times.len() / 2],
            times.len()
        );
    }
}

/// The seam probe: inside the sheet's outline (away from it by more than a pixel and a half), every
/// pixel is the sheet's colour over the background, in both frames: no background through any
/// internal edge.
#[test]
fn no_background_shows_through_a_surfaces_internal_edges_opaque_or_transparent() {
    let probe = probes::seam();
    let drawn = frames(&probe, 1);
    let Some(view) = probe.deltas[0].view.clone() else {
        unreachable!()
    };
    let (_, ring) = probes::rippled_sheet();
    let outline: Vec<[f64; 2]> = ring
        .iter()
        .map(|p| probes::screen_of(&view, probe.width, probe.height, *p))
        .collect();
    let inside = pixels_inside(&outline, 1.5, probe.width, probe.height);
    assert!(inside.len() > 5000, "{} interior pixels", inside.len());
    for (frame, colour) in probes::SEAM_COLOURS.iter().enumerate() {
        let want = blend(premultiply(*colour), 255, [255, 255, 255, 255]);
        let mut merged = 0;
        for (x, y) in &inside {
            assert_eq!(
                pixel(&drawn[frame], probe.width, *x, *y),
                want,
                "frame {frame}: pixel {x},{y}"
            );
            let class = drawn[frame].resolves[(y * probe.width + x) as usize];
            assert_ne!(class, Resolve::Sample as u8, "frame {frame}: pixel {x},{y}");
            if class == Resolve::Coverage as u8 {
                merged += 1;
            }
        }
        // Internal edges cross many pixels; their triangles merge into one fragment there.
        assert!(
            merged > 500,
            "frame {frame}: {merged} pixels on internal edges"
        );
    }
}

/// Where the stroke meets the sheet, each pixel is what 64 x 64 samples of the two give it, within
/// the 16-sample pattern's own error, and most pixels there are neither colour alone.
#[test]
fn a_stroke_crossing_a_surface_is_antialiased_where_they_meet() {
    let probe = probes::stroke_crossing_surface();
    let drawn = &frames(&probe, 1)[0];
    let Some(view) = probe.deltas[0].view.clone() else {
        unreachable!()
    };
    let (w, h) = (probe.width, probe.height);
    let (navy, orange) = ([0.1, 0.15, 0.45, 1.0], [0.95, 0.5, 0.1, 1.0]);
    // The stroke: its centreline's ends on screen and its half width in pixels (CE's 40 units at
    // the line width multiple 0.01, through the frame's scale).
    let (a, b) = (
        probes::screen_of(&view, w, h, [-3.0, 0.5, -1.0]),
        probes::screen_of(&view, w, h, [3.0, 0.5, 1.0]),
    );
    let half = 40.0 * view.cairo_line_width_multiple * (w as f64 / view.frame_width) / 2.0;
    let floor = Flat {
        corners: [
            [-4.0, -3.0, 0.0],
            [4.0, -3.0, 0.0],
            [4.0, 3.0, 0.0],
            [-4.0, 3.0, 0.0],
        ],
        rgba: navy,
    };
    let plane = Flat {
        corners: [
            [-9.0, 0.5, -9.0],
            [9.0, 0.5, -9.0],
            [9.0, 0.5, 9.0],
            [-9.0, 0.5, 9.0],
        ],
        rgba: orange,
    };
    let on_stroke = |p: [f64; 2]| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy);
        (0.0..=1.0).contains(&t) && distance_to_segment(p, a, b) < half
    };
    let pure = [premultiply(orange), premultiply(navy)];
    let (mut crossing, mut mixed, mut worst, mut sum) = (0, 0, 0.0f64, 0.0);
    for y in 0..h {
        for x in 0..w {
            if drawn.resolves[(y * w + x) as usize] != Resolve::Sample as u8 {
                continue;
            }
            let mut acc = [0.0; 4];
            let points = grid_points(x as f64, y as f64, x as f64 + 1.0, y as f64 + 1.0, 64);
            for p in &points {
                let floor_key = probes::sight_key(&view, w, h, &floor, p[0], p[1]);
                let stroke_key = on_stroke(*p)
                    .then(|| probes::sight_key(&view, w, h, &plane, p[0], p[1]))
                    .flatten();
                let c = match (floor_key, stroke_key) {
                    (Some(f), Some(s)) if s < f => premultiply(orange),
                    (Some(_), _) => premultiply(navy),
                    (None, Some(_)) => premultiply(orange),
                    (None, None) => [0, 0, 0, 255],
                };
                for k in 0..4 {
                    acc[k] += c[k] as f64;
                }
            }
            let want = acc.map(|v| v / points.len() as f64);
            let got = pixel(drawn, w, x, y);
            let err = (0..3)
                .map(|k| (got[k] as f64 - want[k]).abs())
                .fold(0.0, f64::max);
            worst = worst.max(err);
            sum += err;
            crossing += 1;
            if !pure.contains(&got) {
                mixed += 1;
            }
        }
    }
    assert!(
        crossing >= 5,
        "{crossing} pixels where the stroke meets the sheet"
    );
    // A pixel's 16 samples can all fall on one side of the crossing; most do not.
    assert!(
        2 * mixed > crossing,
        "{mixed} of {crossing} pixels antialiased"
    );
    // A 16-sample pattern resolves an edge to a sixteenth of the pixel; the colours are far apart.
    let contrast = 255.0 * (0.95 - 0.1);
    assert!(worst <= contrast / 8.0, "worst error {worst}");
    assert!(
        sum / crossing as f64 <= contrast / 24.0,
        "mean error {}",
        sum / crossing as f64
    );
    for c in &probe.checks {
        assert_check(&probe, std::slice::from_ref(drawn), c);
    }
}

#[test]
fn every_probe_draws_the_same_bytes_on_every_run_and_any_number_of_threads() {
    for probe in probes::all() {
        let first = frames(&probe, 1);
        for run in 0..3 {
            for threads in [1, 2, 4] {
                assert!(
                    frames(&probe, threads) == first,
                    "{}: run {run} on {threads} threads differs",
                    probe.name
                );
            }
        }
    }
}
