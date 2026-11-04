//! Meshes drawn through a Stream: coverage against exact area, no seams along internal edges,
//! depth resolved at every sub-sample, transparency in depth order whatever the draw order, ties by
//! draw order, fixed orientation at its centre's depth, CE's light and soft light, the resolve
//! classes, and identical bytes on any number of threads. Set `CORE_STATS=1` to print the coverage error and each scene's share of pixels per
//! resolve.

mod common;

use arachne_core::composite::{
    Resolve, SAMPLES, blend, premultiply, sample_point, srgb_decode, srgb_encode,
};
use arachne_core::project::{FrameMap, Projection};
use arachne_core::shade::{ce_shaded_rgb, ce_unit_normal};
use arachne_core::stream::{DrawError, Stream, StreamError};
use arachne_core::xf::*;
use common::meshes::*;
use common::{flat_view, three_d_view};

const W: u32 = 160;
const H: u32 = 90;

fn px(frame: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = (y * W + x) as usize * 4;
    [frame[i], frame[i + 1], frame[i + 2], frame[i + 3]]
}

fn class(resolves: &[u8], x: u32, y: u32) -> u8 {
    resolves[(y * W + x) as usize]
}

fn opaque_black() -> View {
    flat_view(W, H, [0.0, 0.0, 0.0, 1.0])
}

/// The 3D test view, zoomed so test surfaces cover many pixels.
fn close_view() -> View {
    let mut view = three_d_view(W, H);
    if let Camera::ThreeD(c) = &mut view.camera {
        c.zoom = 1.6;
    }
    view
}

fn square_at(z: f64, half: f64) -> [[f64; 3]; 4] {
    [
        [-half, -half, z],
        [half, -half, z],
        [half, half, z],
        [-half, half, z],
    ]
}

#[test]
fn a_meshs_coverage_is_its_exact_area_within_the_sample_patterns_bound() {
    let corners = [[-5.3, -3.1, 0.0], [4.7, -1.2, 0.0], [-0.9, 3.4, 0.0]];
    let tri = Shape::new(ShapeData::Mesh {
        positions: corners.to_vec(),
        triangles: vec![[0, 1, 2]],
        normals: None,
    });
    let view = opaque_black();
    let s = stream_of(
        view.clone(),
        &[(
            tri.clone(),
            mesh_record(1, &tri, [1.0, 1.0, 1.0, 1.0], Space::Screen),
        )],
        &[1],
    );
    let frame = s.draw(W, H, 1).unwrap();
    let poly = outline(&view, W, H, &corners);
    let (mut edge_err, mut edges, mut worst) = (0.0, 0, 0.0f64);
    for y in 0..H {
        for x in 0..W {
            let area = area_in_pixel(&poly, x, y);
            let got = px(&frame, x, y);
            if area == 0.0 {
                assert_eq!(got, [0, 0, 0, 255], "outside at {x},{y}");
            } else if area == 1.0 {
                assert_eq!(got, [255, 255, 255, 255], "inside at {x},{y}");
            } else {
                let err = (got[0] as f64 / 255.0 - area).abs();
                worst = worst.max(err);
                edge_err += err;
                edges += 1;
            }
        }
    }
    let mean = edge_err / edges as f64;
    if std::env::var_os("CORE_STATS").is_some() {
        println!("coverage: {edges} edge pixels, mean error {mean:.4}, worst {worst:.4}");
    }
    assert!(edges > 200, "{edges} edge pixels");
    assert!(worst <= 0.14, "worst edge pixel off by {worst}");
    assert!(mean <= 0.03, "edge pixels off by {mean} on average");
}

const SHEET: (usize, usize) = (17, 11);

fn rippled_sheet() -> (Shape, Vec<[f64; 3]>) {
    let place = |u: f64, v: f64| {
        // Interior vertices jittered so internal edges run at many angles.
        let interior = u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0;
        let jitter = if interior {
            0.015 * libm::sin(91.0 * u + 37.0 * v)
        } else {
            0.0
        };
        [
            -4.0 + 8.0 * (u + jitter),
            -2.5 + 5.0 * (v - jitter),
            0.4 * libm::sin(3.0 * u) * libm::cos(2.0 * v),
        ]
    };
    let shape = grid_mesh(SHEET.0, SHEET.1, place);
    // The boundary, walked around.
    let (nu, nv) = SHEET;
    let mut ring = Vec::new();
    ring.extend((0..=nu).map(|i| place(i as f64 / nu as f64, 0.0)));
    ring.extend((1..=nv).map(|j| place(1.0, j as f64 / nv as f64)));
    ring.extend((0..nu).rev().map(|i| place(i as f64 / nu as f64, 1.0)));
    ring.extend((1..nv).rev().map(|j| place(0.0, j as f64 / nv as f64)));
    (shape, ring)
}

#[test]
fn no_background_shows_through_a_surfaces_internal_edges_opaque_or_transparent() {
    let (sheet, ring) = rippled_sheet();
    let mut view = close_view();
    view.background = Background::Color([1.0, 1.0, 1.0, 1.0]);
    let inside = pixels_inside(&outline(&view, W, H, &ring), 1.5, W, H);
    assert!(inside.len() > 2000, "{} interior pixels", inside.len());
    for alpha in [1.0, 0.5] {
        let colour = [0.2, 0.6, 0.9, alpha];
        let s = stream_of(
            view.clone(),
            &[(sheet.clone(), mesh_record(1, &sheet, colour, Space::World))],
            &[1],
        );
        let drawn = s.render(W, H, 1).unwrap();
        let want = blend(premultiply(colour), 255, [255, 255, 255, 255]);
        let mut along_edges = 0;
        for (x, y) in &inside {
            assert_eq!(
                px(&drawn.rgba, *x, *y),
                want,
                "alpha {alpha}: pixel {x},{y}"
            );
            let c = class(&drawn.resolves, *x, *y);
            if c == Resolve::Sample as u8 {
                along_edges += 1;
            } else if alpha == 1.0 {
                assert_eq!(c, Resolve::Fast as u8, "pixel {x},{y}");
            } else {
                assert_eq!(c, Resolve::Coverage as u8, "pixel {x},{y}");
            }
        }
        // Internal edges cross many pixels: they take the sample resolve, and still match.
        assert!(along_edges > 300, "{along_edges} edge pixels");
    }
}

#[test]
fn intersecting_surfaces_resolve_by_depth_at_every_sub_sample() {
    let level = quad(square_at(0.0, 3.0));
    let tilted = quad([
        [-3.0, -2.5, -1.2],
        [3.0, -2.5, 1.2],
        [3.0, 2.5, 1.2],
        [-3.0, 2.5, -1.2],
    ]);
    let view = close_view();
    let (red, blue) = ([0.9, 0.1, 0.1, 1.0], [0.1, 0.2, 0.9, 1.0]);
    let s = stream_of(
        view.clone(),
        &[
            (level.clone(), mesh_record(1, &level, red, Space::World)),
            (tilted.clone(), mesh_record(2, &tilted, blue, Space::World)),
        ],
        // Painter's order would put the tilted quad over the level one everywhere.
        &[1, 2],
    );
    let drawn = s.render(W, H, 1).unwrap();
    let meshes = [
        (project_mesh(&view, W, H, &level), premultiply(red)),
        (project_mesh(&view, W, H, &tilted), premultiply(blue)),
    ];
    let bg = [0, 0, 0, 255];
    let (mut fronts, mut crossings) = ([0, 0], 0);
    for y in 0..H {
        for x in 0..W {
            let want = supersample_opaque(&meshes, bg, x, y, 64);
            let got = px(&drawn.rgba, x, y);
            for k in 0..4 {
                assert!(
                    (got[k] as f64 - want[k]).abs() <= 0.14 * 255.0,
                    "pixel {x},{y}: got {got:?}, want {want:?}"
                );
            }
            // Over the pixel, a little widened: which surface is in front, and is each covering?
            let (xf, yf) = (x as f64, y as f64);
            let points = grid_points(xf - 0.05, yf - 0.05, xf + 1.05, yf + 1.05, 24);
            let at: Vec<Option<usize>> = points.iter().map(|p| front_at(&meshes, *p)).collect();
            let both_cover = points.iter().all(|p| {
                meshes.iter().all(|(m, _)| {
                    m.triangles
                        .iter()
                        .any(|(t, _)| barycentric(*t, *p).is_some())
                })
            });
            for m in 0..2 {
                if at.iter().all(|f| *f == Some(m)) {
                    fronts[m] += 1;
                    assert_eq!(got, meshes[m].1, "pixel {x},{y}");
                }
            }
            // Where the front surface differs between the pixel's own sub-samples, the depth order
            // is not one order across it: the sample resolve. (A crossing that passes between the
            // sub-samples leaves one order at all of them, and the coverage resolve agrees.)
            let at_samples: Vec<Option<usize>> = (0..SAMPLES)
                .map(|s| front_at(&meshes, sample_point(x, y, s)))
                .collect();
            if both_cover && at_samples.contains(&Some(0)) && at_samples.contains(&Some(1)) {
                crossings += 1;
                assert_eq!(
                    class(&drawn.resolves, x, y),
                    Resolve::Sample as u8,
                    "pixel {x},{y} straddles the intersection"
                );
            }
        }
    }
    // Each surface is in front somewhere: neither paints over the other whole.
    assert!(fronts[0] > 200 && fronts[1] > 200, "fronts {fronts:?}");
    assert!(crossings > 20, "{crossings} pixels on the intersection");
}

fn layers() -> Vec<(Shape, Record)> {
    [
        (-0.3, [0.9, 0.1, 0.1, 0.4]),
        (0.0, [0.1, 0.9, 0.1, 0.5]),
        (0.3, [0.1, 0.1, 0.9, 0.6]),
    ]
    .iter()
    .enumerate()
    .map(|(i, (z, colour))| {
        let shape = quad(square_at(*z, 2.0 + 0.1 * *z));
        let record = mesh_record(i as u32 + 1, &shape, *colour, Space::World);
        (shape, record)
    })
    .collect()
}

#[test]
fn translucent_surfaces_composite_in_depth_order_whatever_the_draw_order() {
    let view = close_view();
    let items = layers();
    let mut frames = Vec::new();
    for order in [
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1],
    ] {
        frames.push(
            stream_of(view.clone(), &items, &order)
                .draw(W, H, 1)
                .unwrap(),
        );
    }
    assert!(frames.iter().all(|f| *f == frames[0]));
    // A pixel inside all three: the camera (looking down from +z) sees z = 0.3 in front.
    let inside: Vec<Vec<(u32, u32)>> = [-0.3, 0.0, 0.3]
        .iter()
        .map(|z| {
            pixels_inside(
                &outline(&view, W, H, &square_at(*z, 2.0 + 0.1 * z)),
                2.0,
                W,
                H,
            )
        })
        .collect();
    let (x, y) = *inside[0]
        .iter()
        .find(|p| inside[1].contains(p) && inside[2].contains(p))
        .expect("a pixel inside all three");
    let colour = |i: usize| match &items[i].1.paint.fill {
        Paint::One(c) => premultiply(*c),
        _ => unreachable!(),
    };
    let back_to_front = blend(
        colour(2),
        255,
        blend(colour(1), 255, blend(colour(0), 255, [0, 0, 0, 255])),
    );
    assert_eq!(px(&frames[0], x, y), back_to_front);
}

#[test]
fn coplanar_meshes_tie_by_draw_order_the_later_on_top() {
    let view = close_view();
    let shape = quad(square_at(0.5, 2.0));
    let (red, green) = ([0.9, 0.1, 0.1, 1.0], [0.1, 0.9, 0.1, 1.0]);
    let items = [
        (shape.clone(), mesh_record(1, &shape, red, Space::World)),
        (shape.clone(), mesh_record(2, &shape, green, Space::World)),
    ];
    let inside = pixels_inside(&outline(&view, W, H, &square_at(0.5, 2.0)), 1.0, W, H);
    let red_last = stream_of(view.clone(), &items, &[2, 1])
        .draw(W, H, 1)
        .unwrap();
    let green_last = stream_of(view.clone(), &items, &[1, 2])
        .draw(W, H, 1)
        .unwrap();
    assert!(inside.len() > 300);
    for (x, y) in inside {
        assert_eq!(px(&red_last, x, y), premultiply(red));
        assert_eq!(px(&green_last, x, y), premultiply(green));
    }
    // Along the outline too, no sub-sample lets the earlier mesh through.
    let g = premultiply(green);
    for p in green_last.chunks(4) {
        assert!(p[0] <= g[0], "{p:?}");
    }
}

#[test]
fn the_inside_of_a_translucent_closed_surface_takes_the_coverage_resolve() {
    let octahedron = Shape::new(ShapeData::Mesh {
        positions: vec![
            [2.5, 0.0, 0.0],
            [-2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, -2.5, 0.0],
            [0.0, 0.0, 2.5],
            [0.0, 0.0, -2.5],
        ],
        triangles: vec![
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        normals: None,
    });
    let view = close_view();
    let colour = [0.3, 0.7, 0.2, 0.5];
    let s = stream_of(
        view.clone(),
        &[(
            octahedron.clone(),
            mesh_record(1, &octahedron, colour, Space::World),
        )],
        &[1],
    );
    let drawn = s.render(W, H, 1).unwrap();
    let projected = project_mesh(&view, W, H, &octahedron);
    let src = premultiply(colour);
    let front_over_back = blend(src, 255, blend(src, 255, [0, 0, 0, 255]));
    let mut checked = 0;
    for y in 0..H {
        for x in 0..W {
            let c = [x as f64 + 0.5, y as f64 + 0.5];
            let covering = projected
                .triangles
                .iter()
                .filter(|(t, _)| barycentric(*t, c).is_some())
                .count();
            let near_edge = projected
                .triangles
                .iter()
                .any(|(t, _)| (0..3).any(|i| distance_to_segment(c, t[i], t[(i + 1) % 3]) <= 1.0));
            if covering == 2 && !near_edge {
                checked += 1;
                assert_eq!(
                    class(&drawn.resolves, x, y),
                    Resolve::Coverage as u8,
                    "pixel {x},{y}: front and back faces cover it"
                );
                assert_eq!(px(&drawn.rgba, x, y), front_over_back, "pixel {x},{y}");
            }
        }
    }
    assert!(checked > 300, "{checked} interior pixels");
}

#[test]
fn the_resolve_counts_cover_every_pixel_once() {
    for (name, s) in scenes(W, H) {
        let drawn = s.render(W, H, 2).unwrap();
        let c = drawn.counts;
        if std::env::var_os("CORE_STATS").is_some() {
            let share = |n: u64| 100.0 * n as f64 / (W * H) as f64;
            println!(
                "{name}: background {:.1}%, fast {:.1}%, coverage {:.1}%, sample {:.1}%",
                share(c.background),
                share(c.fast),
                share(c.coverage),
                share(c.sample)
            );
        }
        assert_eq!(
            c.background + c.fast + c.coverage + c.sample,
            (W * H) as u64,
            "{name}"
        );
        let count = |r: Resolve| drawn.resolves.iter().filter(|v| **v == r as u8).count() as u64;
        assert_eq!(c.background, count(Resolve::Background), "{name}");
        assert_eq!(c.fast, count(Resolve::Fast), "{name}");
        assert_eq!(c.coverage, count(Resolve::Coverage), "{name}");
        assert_eq!(c.sample, count(Resolve::Sample), "{name}");
    }
}

#[test]
fn frames_are_byte_identical_across_runs_and_thread_counts() {
    // Sizes that do not divide into tiles, so edge tiles are partial.
    let (w, h) = (333, 197);
    for (name, s) in scenes(w, h) {
        let first = s.render(w, h, 1).unwrap();
        for threads in [1, 1, 2, 3, 4, 8] {
            let again = s.render(w, h, threads).unwrap();
            assert!(again.rgba == first.rgba, "{name} on {threads} threads");
            assert!(
                again.resolves == first.resolves,
                "{name} on {threads} threads"
            );
        }
    }
}

const TRIANGLE: [[f64; 3]; 3] = [[-4.0, -3.0, -0.5], [4.0, -2.0, 0.5], [-1.0, 3.5, 1.0]];
const PAINT: [f64; 4] = [0.4, 0.5, 0.6, 1.0];

fn single_triangle(view: &View, lit: bool) -> Stream {
    let tri = Shape::new(ShapeData::Mesh {
        positions: TRIANGLE.to_vec(),
        triangles: vec![[0, 1, 2]],
        normals: None,
    });
    let mut r = mesh_record(1, &tri, PAINT, Space::World);
    r.depth.shade_in_3d = lit;
    stream_of(view.clone(), &[(tri, r)], &[1])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[test]
fn ces_light_shades_a_mesh_at_its_corners_as_ce_shades_a_face() {
    let view = close_view();
    let Camera::ThreeD(camera) = &view.camera else {
        unreachable!()
    };
    let frame = single_triangle(&view, true).draw(W, H, 1).unwrap();
    // CE's colour at each corner; across the triangle the corners' colours interpolate on screen,
    // as Cairo interpolates CE's two-stop gradient (encoded values, straight alpha).
    let normal = ce_unit_normal(sub(TRIANGLE[1], TRIANGLE[0]), sub(TRIANGLE[2], TRIANGLE[0]));
    let rgb = [PAINT[0], PAINT[1], PAINT[2]];
    // Cairo clamps each stop to 0..1 before it interpolates.
    let shades = TRIANGLE
        .map(|c| ce_shaded_rgb(rgb, c, normal, camera.light_source).map(|v| v.clamp(0.0, 1.0)));
    let screen = outline(&view, W, H, &TRIANGLE);
    let tri = [screen[0], screen[1], screen[2]];
    let inside = pixels_inside(&screen, 1.0, W, H);
    assert!(inside.len() > 500);
    let mut varied = std::collections::HashSet::new();
    for (x, y) in inside {
        let w = barycentric(tri, [x as f64 + 0.5, y as f64 + 0.5]).unwrap();
        let c = [0, 1, 2].map(|k| w[0] * shades[0][k] + w[1] * shades[1][k] + w[2] * shades[2][k]);
        let want = premultiply([c[0], c[1], c[2], 1.0]);
        let got = px(&frame, x, y);
        for k in 0..3 {
            assert!(
                (got[k] as i32 - want[k] as i32).abs() <= 1,
                "pixel {x},{y}: got {got:?}, want {want:?}"
            );
        }
        varied.insert(got);
    }
    assert!(varied.len() > 10, "the light varies across the triangle");
    // Unlit, the same triangle is its paint.
    let flat = single_triangle(&view, false).draw(W, H, 1).unwrap();
    let centre = barycentric_point(&screen);
    assert_eq!(px(&flat, centre.0, centre.1), premultiply(PAINT));
    // And with the camera's shading switched off.
    let mut off = view.clone();
    if let Camera::ThreeD(c) = &mut off.camera {
        c.should_apply_shading = false;
    }
    assert_eq!(single_triangle(&off, true).draw(W, H, 1).unwrap(), flat);
}

fn barycentric_point(screen: &[[f64; 2]]) -> (u32, u32) {
    let n = screen.len() as f64;
    let x = screen.iter().map(|p| p[0]).sum::<f64>() / n;
    let y = screen.iter().map(|p| p[1]).sum::<f64>() / n;
    (x as u32, y as u32)
}

#[test]
fn soft_light_is_computed_in_linear_light_then_encoded() {
    let mut view = close_view();
    let Camera::ThreeD(camera) = &mut view.camera else {
        unreachable!()
    };
    // A light far off at 60 degrees from the quad's normal: n . l = 1/2 everywhere.
    camera.light_source = [1e9 * 0.8660254037844386, 0.0, 1e9 * 0.5];
    view.light = LightModel::SoftDiffuse {
        ambient: 0.0,
        wrap: 0.0,
    };
    let base = 0.8;
    let lit = srgb_encode(0.5 * srgb_decode(base));
    let want = premultiply([lit, lit, lit, 1.0]);
    let corners = square_at(0.0, 2.0);
    let inside = pixels_inside(&outline(&view, W, H, &corners), 1.0, W, H);
    assert!(inside.len() > 300);
    for normal_z in [1.0, -1.0] {
        let shape = Shape::new(ShapeData::Mesh {
            positions: corners.to_vec(),
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normals: Some(vec![[0.0, 0.0, normal_z]; 4]),
        });
        let mut r = mesh_record(1, &shape, [base, base, base, 1.0], Space::World);
        r.depth.shade_in_3d = true;
        let frame = stream_of(view.clone(), &[(shape, r)], &[1])
            .draw(W, H, 1)
            .unwrap();
        // A normal turned away from the eye is lit as the side the eye sees.
        for (x, y) in &inside {
            let got = px(&frame, *x, *y);
            assert!(
                (got[0] as i32 - want[0] as i32).abs() <= 1,
                "normal z {normal_z}, pixel {x},{y}: got {got:?}, want {want:?}"
            );
        }
    }
    // Halving the encoded value instead would give 0.4 * 255 = 102.
    assert!(want[0] > 140, "{want:?}");
}

#[test]
fn under_a_flat_camera_meshes_paint_in_draw_order_whatever_their_z() {
    let near = quad(square_at(3.0, 2.0));
    let far = quad([
        [-1.0, -1.0, -3.0],
        [3.0, -1.0, -3.0],
        [3.0, 3.0, -3.0],
        [-1.0, 3.0, -3.0],
    ]);
    let (a, b) = ([0.9, 0.2, 0.1, 0.7], [0.1, 0.3, 0.9, 0.7]);
    let items = [
        (near.clone(), mesh_record(1, &near, a, Space::World)),
        (far.clone(), mesh_record(2, &far, b, Space::World)),
    ];
    let bg = [0, 0, 0, 255];
    let view = opaque_black();
    let p = to_pixels(&view, W, H, [0.5, 0.5, 0.0]).0;
    let (x, y) = (p[0] as u32, p[1] as u32);
    let far_last = stream_of(view.clone(), &items, &[1, 2])
        .draw(W, H, 1)
        .unwrap();
    let near_last = stream_of(view.clone(), &items, &[2, 1])
        .draw(W, H, 1)
        .unwrap();
    let (pa, pb) = (premultiply(a), premultiply(b));
    assert_eq!(px(&far_last, x, y), blend(pb, 255, blend(pa, 255, bg)));
    assert_eq!(px(&near_last, x, y), blend(pa, 255, blend(pb, 255, bg)));
}

#[test]
fn group_opacity_scales_a_meshs_alpha() {
    let shape = quad(square_at(0.0, 2.0));
    let view = opaque_black();
    let mut half = mesh_record(1, &shape, [0.9, 0.5, 0.1, 1.0], Space::Screen);
    half.opacity = 0.5;
    let translucent = mesh_record(1, &shape, [0.9, 0.5, 0.1, 0.5], Space::Screen);
    assert_eq!(
        stream_of(view.clone(), &[(shape.clone(), half)], &[1])
            .draw(W, H, 1)
            .unwrap(),
        stream_of(view, &[(shape, translucent)], &[1])
            .draw(W, H, 1)
            .unwrap()
    );
}

#[test]
fn an_overlay_mesh_is_drawn_in_frame_and_in_front_of_the_scene() {
    let view = three_d_view(W, H);
    let label = quad(square_at(0.0, 1.0));
    let wall = quad(square_at(1.0, 6.0));
    let (white, grey) = ([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 1.0]);
    let s = stream_of(
        view.clone(),
        &[
            (label.clone(), mesh_record(1, &label, white, Space::Overlay)),
            (wall.clone(), mesh_record(2, &wall, grey, Space::World)),
        ],
        &[1, 2],
    );
    let frame = s.draw(W, H, 1).unwrap();
    // The label's points are not projected: frame coordinates through the frame mapping.
    let map = FrameMap::cairo(W, H, view.frame_width, view.frame_height, view.frame_center);
    let label_outline: Vec<[f64; 2]> = square_at(0.0, 1.0).iter().map(|p| map.apply(*p)).collect();
    let on_label = pixels_inside(&label_outline, 1.0, W, H);
    let wall_outline = outline(&view, W, H, &square_at(1.0, 6.0));
    let on_wall = pixels_inside(&wall_outline, 1.0, W, H);
    // The label covers the wall where both are: it is in front, though drawn first.
    let both: Vec<_> = on_label.iter().filter(|p| on_wall.contains(p)).collect();
    assert!(both.len() > 100, "{}", both.len());
    for (x, y) in both {
        assert_eq!(px(&frame, *x, *y), premultiply(white));
    }
    let (x, y) = *on_wall
        .iter()
        .find(|p| {
            !on_label
                .iter()
                .any(|q| q.0.abs_diff(p.0) < 3 && q.1.abs_diff(p.1) < 3)
        })
        .expect("a pixel of the wall alone");
    assert_eq!(px(&frame, x, y), premultiply(grey));
}

#[test]
fn a_mesh_far_larger_than_the_frame_covers_it_without_cracks() {
    // Its corners land millions of pixels away, and its internal edge crosses the frame.
    let huge = quad([
        [-1e5, -2e5, 0.0],
        [3e5, -1e5, 0.0],
        [2e5, 4e5, 0.0],
        [-3e5, 1e5, 0.0],
    ]);
    let s = stream_of(
        opaque_black(),
        &[(
            huge.clone(),
            mesh_record(1, &huge, [0.2, 0.4, 0.6, 1.0], Space::Screen),
        )],
        &[1],
    );
    let want = premultiply([0.2, 0.4, 0.6, 1.0]);
    let frame = s.draw(W, H, 3).unwrap();
    assert!(frame.chunks(4).all(|p| p == want));
}

#[test]
fn a_mesh_with_no_fill_draws_nothing() {
    let shape = quad(square_at(0.0, 2.0));
    let mut r = mesh_record(1, &shape, [1.0; 4], Space::Screen);
    r.paint.fill = Paint::None;
    let frame = stream_of(opaque_black(), &[(shape, r)], &[1])
        .draw(W, H, 1)
        .unwrap();
    assert!(frame.chunks(4).all(|p| p == [0, 0, 0, 255]));
}

#[test]
fn a_mesh_carrying_what_this_build_does_not_draw_is_refused_whole() {
    let shape = quad(square_at(0.0, 2.0));
    let other = quad(square_at(0.0, 1.0));
    let line = Shape::new(ShapeData::Path {
        points: common::square_points(1.0),
    });
    let (other_key, line_key) = (other.key, line.key);
    type Change = Box<dyn Fn(&mut Record)>;
    let cases: Vec<(&str, Change)> = vec![
        (
            "blend",
            Box::new(move |r: &mut Record| {
                r.geometry.blend = Some(Blend {
                    shape_b: other_key,
                    place_b: IDENTITY,
                    t: 0.5,
                })
            }),
        ),
        (
            "reveal",
            Box::new(|r: &mut Record| r.geometry.reveal = Some((0.0, 0.5))),
        ),
        (
            "stroke",
            Box::new(|r: &mut Record| {
                r.paint.stroke = Paint::One([1.0; 4]);
                r.paint.stroke_width = 2.0;
            }),
        ),
        (
            "gradient",
            Box::new(|r: &mut Record| r.paint.fill = Paint::Two([[1.0; 4], [0.0, 0.0, 0.0, 1.0]])),
        ),
        (
            "paint blend",
            Box::new(|r: &mut Record| {
                r.paint.paint_b = Some(PaintB {
                    fill: Paint::One([0.0, 0.0, 0.0, 1.0]),
                    stroke: Paint::None,
                    background_stroke: Paint::None,
                    stroke_width: 0.0,
                    background_stroke_width: 0.0,
                });
                r.paint.mix = 0.5;
            }),
        ),
        (
            "dash",
            Box::new(|r: &mut Record| {
                r.dash = Some(Dash {
                    period: 1.0,
                    duty: 0.5,
                    phase: 0.0,
                })
            }),
        ),
        (
            "clip",
            Box::new(move |r: &mut Record| r.clip = Some((line_key, IDENTITY))),
        ),
    ];
    for (what, change) in cases {
        let mut r = mesh_record(1, &shape, [1.0; 4], Space::Screen);
        change(&mut r);
        let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
        d.view = Some(opaque_black());
        d.upload = vec![shape.clone(), other.clone(), line.clone()];
        d.set = vec![r];
        d.order = Some(vec![OrderEntry {
            layer: Layer::Moving,
            slot: 1,
            part: 0,
        }]);
        let mut s = Stream::new();
        s.apply(&d).unwrap();
        match s.draw(W, H, 1) {
            Err(DrawError::Unsupported(u)) => {
                assert_eq!((u.kind.as_str(), u.slot), ("mesh", Some(1)), "{what}");
                assert!(u.reason.contains(what), "{what}: {}", u.reason);
            }
            other => panic!("{what}: {other:?}"),
        }
    }
}

#[test]
fn a_mesh_whose_triangles_name_missing_vertices_is_refused_by_apply() {
    let bad = Shape::new(ShapeData::Mesh {
        positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 3]],
        normals: None,
    });
    let short_normals = Shape::new(ShapeData::Mesh {
        positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 2]],
        normals: Some(vec![[0.0, 0.0, 1.0]]),
    });
    for shape in [bad, short_normals] {
        let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
        d.view = Some(opaque_black());
        d.upload = vec![shape];
        assert!(matches!(
            Stream::new().apply(&d),
            Err(StreamError::Invalid(_))
        ));
    }
}

#[test]
fn soft_light_needs_an_ambient_share_and_a_wrap_it_can_use() {
    for (ambient, wrap) in [(-0.1, 0.0), (1.5, 0.0), (0.2, -1.0), (f64::NAN, 0.0)] {
        let mut view = three_d_view(W, H);
        view.light = LightModel::SoftDiffuse { ambient, wrap };
        let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
        d.view = Some(view);
        assert!(
            matches!(Stream::new().apply(&d), Err(StreamError::Invalid(_))),
            "{ambient} {wrap}"
        );
    }
}

#[test]
fn a_fixed_orientation_mesh_is_a_camera_facing_plane_at_its_centres_depth() {
    // A label kept facing the camera about its centre, which lies on a level surface the camera
    // sees at a slant: the surface passes through the label's plane, in front of half the label
    // and behind the other half, whichever is drawn first.
    let view = close_view();
    let centre = [0.4, 0.3, 0.0];
    let label_corners = [
        [-0.8, -0.6, 0.0],
        [2.0, -0.6, 0.0],
        [2.0, 1.2, 0.0],
        [-0.8, 1.2, 0.0],
    ];
    let label = quad(label_corners);
    let level = quad(square_at(0.0, 4.0));
    let (white, green) = ([1.0, 1.0, 1.0, 1.0], [0.1, 0.6, 0.2, 1.0]);
    let mut label_record = mesh_record(1, &label, white, Space::World);
    label_record.depth.fixed_orientation_centre = Some(centre);
    let items = [
        (label.clone(), label_record),
        (level.clone(), mesh_record(2, &level, green, Space::World)),
    ];
    let drawn = stream_of(view.clone(), &items, &[1, 2])
        .render(W, H, 1)
        .unwrap();
    let reversed = stream_of(view.clone(), &items, &[2, 1])
        .render(W, H, 1)
        .unwrap();
    assert_eq!(drawn.rgba, reversed.rgba, "depth decides, not draw order");

    // The oracle: the label's corners moved by its centre's projection, unrotated, at the
    // centre's depth key everywhere.
    let Camera::ThreeD(c) = &view.camera else {
        unreachable!()
    };
    let projection = Projection::new(c, view.frame_center);
    let moved = projection.project(centre);
    let key = -projection.factor(moved[2]);
    let map = FrameMap::cairo(W, H, view.frame_width, view.frame_height, view.frame_center);
    let shown = label_corners.map(|p| {
        map.apply([
            p[0] + moved[0] - centre[0],
            p[1] + moved[1] - centre[1],
            0.0,
        ])
    });
    let label_projected = Projected {
        triangles: [[0, 1, 2], [0, 2, 3]]
            .iter()
            .map(|t: &[usize; 3]| (t.map(|i| shown[i]), [key; 3]))
            .collect(),
    };
    let meshes = [
        (label_projected, premultiply(white)),
        (project_mesh(&view, W, H, &level), premultiply(green)),
    ];
    let mut fronts = [0, 0];
    for y in 0..H {
        for x in 0..W {
            let (xf, yf) = (x as f64, y as f64);
            let points = grid_points(xf - 0.05, yf - 0.05, xf + 1.05, yf + 1.05, 24);
            let at: Vec<Option<usize>> = points.iter().map(|p| front_at(&meshes, *p)).collect();
            for m in 0..2 {
                if at.iter().all(|f| *f == Some(m)) {
                    fronts[m] += 1;
                    assert_eq!(px(&drawn.rgba, x, y), meshes[m].1, "pixel {x},{y}");
                }
            }
        }
    }
    // The label is in front somewhere and behind the surface somewhere.
    let label_area = area(&shown) as usize;
    let label_front = fronts[0];
    assert!(label_front > 100, "label in front at {label_front} pixels");
    assert!(
        label_front + 100 < label_area,
        "label in front at {label_front} of its {label_area} pixels"
    );
    assert!(fronts[1] > 1000, "surface in front at {} pixels", fronts[1]);
}
