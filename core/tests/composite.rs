//! The compositor and the colour contract, with synthetic producers: Cairo's ARGB32 blending of 2D
//! paint, exact area on the coverage resolve with masks asked for only when a pixel needs them,
//! the depth order read at the sub-samples, the opaque depth readable before the transparent pass,
//! fragments from any producer through one call, and emitted light kept linear and tone-mapped
//! once.

mod common;

use arachne_core::composite::*;
use arachne_core::mesh;
use arachne_core::tiles;
use arachne_core::xf::*;
use common::meshes::*;
use common::resolve::EverySample;
use common::three_d_view;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

fn rect_pixels(pass_rect: Rect, x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for y in y0.max(pass_rect.y0)..y1.min(pass_rect.y1) {
        for x in x0.max(pass_rect.x0)..x1.min(pass_rect.x1) {
            out.push((x, y));
        }
    }
    out
}

/// An opaque rectangle of pixels at one depth, covering the sub-samples of `mask`.
struct Wall {
    rect: [u32; 4],
    depth: f64,
    colour: [u8; 4],
    mask: u16,
    order: u32,
}

impl Producer for Wall {
    fn opaque(&self, pass: &mut OpaquePass<'_>) {
        let [x0, y0, x1, y1] = self.rect;
        for (x, y) in rect_pixels(pass.rect(), x0, y0, x1, y1) {
            let f = Fragment::new(
                Plane::constant(self.depth),
                self.colour,
                Coverage::Mask(self.mask),
                self.order,
                0,
            );
            pass.add(x, y, f);
        }
    }
}

/// A path-like producer: the half-plane `a * x + b * y <= c` in pixel coordinates, with its exact
/// area per pixel and, on demand, its mask point-sampled at the sub-samples and its exact area in
/// each sub-sample's cell (`asks` counts the demands).
struct HalfPlane {
    line: [f64; 3],
    colour: [u8; 4],
    depth: Plane,
    order: u32,
    asks: AtomicUsize,
}

impl HalfPlane {
    fn new(line: [f64; 3], colour: [u8; 4], depth: Plane) -> HalfPlane {
        HalfPlane {
            line,
            colour,
            depth,
            order: 1,
            asks: AtomicUsize::new(0),
        }
    }

    fn area(&self, x: u32, y: u32) -> f64 {
        self.area_in(x as f64, y as f64, 1.0)
    }

    /// The fraction of the square at `(x0, y0)` of side `side` inside the half-plane.
    fn area_in(&self, x0: f64, y0: f64, side: f64) -> f64 {
        let square = [
            [x0, y0],
            [x0 + side, y0],
            [x0 + side, y0 + side],
            [x0, y0 + side],
        ];
        area(&clip(&square, self.line[0], self.line[1], self.line[2])) / (side * side)
    }

    fn inside(&self, p: [f64; 2]) -> bool {
        self.line[0] * p[0] + self.line[1] * p[1] <= self.line[2]
    }
}

impl Producer for HalfPlane {
    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        let r = pass.rect();
        for (x, y) in rect_pixels(r, r.x0, r.y0, r.x1, r.y1) {
            let area = self.area(x, y);
            if area > 0.0 {
                let coverage = Coverage::Area {
                    area: area as f32,
                    id: 7,
                };
                pass.add(
                    x,
                    y,
                    Fragment::new(self.depth, self.colour, coverage, self.order, 0),
                );
            }
        }
    }

    fn mask(&self, id: u32, x: u32, y: u32) -> u16 {
        assert_eq!(id, 7);
        self.asks.fetch_add(1, Ordering::Relaxed);
        (0..SAMPLES).fold(0, |m, s| {
            if self.inside(sample_point(x, y, s)) {
                m | 1 << s
            } else {
                m
            }
        })
    }

    fn cell_areas(&self, id: u32, x: u32, y: u32) -> [f32; SAMPLES] {
        assert_eq!(id, 7);
        self.asks.fetch_add(1, Ordering::Relaxed);
        std::array::from_fn(|s| {
            let [column, row] = sample_cell(s);
            let (cx, cy) = (x as f64 + column as f64 / 4.0, y as f64 + row as f64 / 4.0);
            self.area_in(cx, cy, 0.25) as f32
        })
    }
}

/// One fragment in one pixel, through the transparent pass.
struct Single {
    at: (u32, u32),
    fragment: Fragment,
}

impl Producer for Single {
    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        let r = pass.rect();
        let (x, y) = self.at;
        if (r.x0..r.x1).contains(&x) && (r.y0..r.y1).contains(&y) {
            pass.add(x, y, self.fragment);
        }
    }
}

const FAR: f64 = 10.0;

/// A pixel, and the opaque depth read there.
type DepthRead = ((u32, u32), [f64; SAMPLES]);

/// A fake volume: fog from `near` up to the opaque surface (or `FAR`), read from the opaque depth,
/// handed to the transparent pass as one fragment per pixel at depth `near`.
struct Fog {
    near: f64,
    density: f64,
    rgb: [f64; 3],
    order: u32,
    reads: Mutex<Vec<DepthRead>>,
}

impl Fog {
    fn new(near: f64, density: f64, rgb: [f64; 3], order: u32) -> Fog {
        Fog {
            near,
            density,
            rgb,
            order,
            reads: Mutex::new(Vec::new()),
        }
    }

    fn colour_through(&self, thickness: f64) -> [u8; 4] {
        let alpha = 1.0 - libm::exp(-self.density * thickness);
        premultiply([self.rgb[0], self.rgb[1], self.rgb[2], alpha])
    }
}

impl Producer for Fog {
    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        let r = pass.rect();
        for (x, y) in rect_pixels(r, r.x0, r.y0, r.x1, r.y1) {
            let depth = pass.opaque_depth(x, y);
            self.reads.lock().unwrap().push(((x, y), depth));
            let thickness =
                depth.iter().map(|d| d.min(FAR) - self.near).sum::<f64>() / SAMPLES as f64;
            let f = Fragment::new(
                Plane::constant(self.near),
                self.colour_through(thickness),
                Coverage::Mask(FULL),
                self.order,
                0,
            );
            pass.add(x, y, f);
        }
    }
}

/// Light emitted over a rectangle at one depth.
struct Emitter {
    rect: [u32; 4],
    radiance: [f64; 3],
    depth: f64,
}

impl Producer for Emitter {
    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        let [x0, y0, x1, y1] = self.rect;
        for (x, y) in rect_pixels(pass.rect(), x0, y0, x1, y1) {
            pass.add_emission(
                x,
                y,
                Emission {
                    radiance: self.radiance,
                    mask: FULL,
                    depth: Plane::constant(self.depth),
                },
            );
        }
    }
}

const GREY: [u8; 4] = [128, 128, 128, 255];

fn render(w: u32, h: u32, producers: &[&dyn Producer], threads: usize) -> Rendered {
    tiles::render(Frame::filled(w, h, GREY), producers, threads)
}

fn at(r: &Rendered, w: u32, x: u32, y: u32) -> [u8; 4] {
    let i = (y * w + x) as usize * 4;
    [r.rgba[i], r.rgba[i + 1], r.rgba[i + 2], r.rgba[i + 3]]
}

fn class_at(r: &Rendered, w: u32, x: u32, y: u32) -> u8 {
    r.resolves[(y * w + x) as usize]
}

#[test]
fn two_d_paint_blends_with_cairos_argb32_arithmetic() {
    // blend.txt holds bytes Cairo 1.18's image surface wrote, filling as CE fills.
    let text = include_str!("data/blend.txt");
    let mut lines = 0;
    for line in text.lines().filter(|l| l.starts_with("blend ")) {
        let t: Vec<&str> = line.split_whitespace().collect();
        let f = |i: usize| t[i].parse::<f64>().unwrap();
        let b = |i: usize| t[i].parse::<u8>().unwrap();
        let rgba = [f(1), f(2), f(3), f(4)];
        let dst = [b(5), b(6), b(7), b(8)];
        let coverage = b(9);
        assert_eq!(t[10], "->");
        let want = [b(11), b(12), b(13), b(14)];
        assert_eq!(blend(premultiply(rgba), coverage, dst), want, "{line}");
        lines += 1;
    }
    assert!(lines > 600, "{lines} cases");
    // In encoded space: half-transparent white over black is half the code value, not the 188
    // that blending in linear light would give.
    assert_eq!(
        blend(premultiply([1.0, 1.0, 1.0, 0.5]), 255, [0, 0, 0, 255]),
        [128, 128, 128, 255]
    );
}

#[test]
fn the_sample_pattern_is_stratified_and_centred_on_the_pixel() {
    // One sample per cell of the 4 x 4 grid, each in the cell sample_cell names.
    let mut cells: Vec<[usize; 2]> = (0..SAMPLES).map(sample_cell).collect();
    for s in 0..SAMPLES {
        let [fx, fy] = sample_fixed(s);
        assert_eq!(sample_cell(s), [fx as usize / 64, fy as usize / 64]);
    }
    cells.sort();
    cells.dedup();
    assert_eq!(cells.len(), SAMPLES);
    // Edges at 45 and 135 degrees get at least 13 coverage levels: 12 distinct sums and
    // differences of the offsets.
    let distinct = |f: fn([i8; 2]) -> i8| {
        let mut v: Vec<i8> = SAMPLE_OFFSETS.iter().map(|o| f(*o)).collect();
        v.sort();
        v.dedup();
        v.len()
    };
    assert!(distinct(|[x, y]| x + y) >= 12);
    assert!(distinct(|[x, y]| x - y) >= 12);
    // One sample per row and per column of sixteenths, centred on the pixel.
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    let (mut cx, mut cy) = (0.0, 0.0);
    for s in 0..SAMPLES {
        let [x, y] = sample_point(10, 20, s);
        let [fx, fy] = sample_fixed(s);
        assert_eq!([x - 10.0, y - 20.0], [fx as f64 / 256.0, fy as f64 / 256.0]);
        xs.push(fx);
        ys.push(fy);
        cx += x - 10.0;
        cy += y - 20.0;
    }
    xs.sort();
    ys.sort();
    xs.dedup();
    ys.dedup();
    assert_eq!((xs.len(), ys.len()), (SAMPLES, SAMPLES));
    assert_eq!([cx / 16.0, cy / 16.0], [0.5, 0.5]);
}

#[test]
fn srgb_encoding_round_trips_every_byte() {
    for b in 0..=255u8 {
        let back = srgb_encode(srgb_decode(b as f64 / 255.0)) * 255.0;
        assert_eq!((back + 0.5) as u8, b);
    }
    assert!((srgb_decode(0.5) - 0.21404114048223255).abs() < 1e-12);
    assert!((srgb_encode(0.5) - 0.7353569830524495).abs() < 1e-12);
}

#[test]
fn a_paths_exact_area_composites_on_the_coverage_resolve_and_asks_for_no_mask() {
    let (w, h) = (37, 23);
    let red = premultiply([0.9, 0.1, 0.2, 0.8]);
    let path = HalfPlane::new([0.6, 1.0, 21.3], red, Plane::constant(0.0));
    let drawn = render(w, h, &[&path], 2);
    for y in 0..h {
        for x in 0..w {
            let area = path.area(x, y);
            let (want, class) = if area > 0.0 {
                (
                    blend(red, area_coverage(area as f32), GREY),
                    Resolve::Coverage,
                )
            } else {
                (GREY, Resolve::Background)
            };
            assert_eq!(at(&drawn, w, x, y), want, "pixel {x},{y}");
            assert_eq!(class_at(&drawn, w, x, y), class as u8, "pixel {x},{y}");
        }
    }
    assert_eq!(path.asks.load(Ordering::Relaxed), 0);
}

#[test]
fn a_path_crossing_a_surface_in_depth_takes_the_sample_resolve_with_its_cells_areas() {
    let (w, h) = (48, 32);
    let blue = premultiply([0.1, 0.2, 0.9, 1.0]);
    let red = premultiply([0.9, 0.1, 0.1, 0.9]);
    let wall = Wall {
        rect: [0, 0, w, h],
        depth: 0.0,
        colour: blue,
        mask: FULL,
        order: 0,
    };
    // The path is in front of the wall left of x = 20.3 and behind it to the right.
    let depth = Plane {
        a: -20.3 * 0.25,
        b: 0.25,
        c: 0.0,
    };
    let path = HalfPlane::new([1.0, 0.45, 30.0], red, depth);
    let drawn = render(w, h, &[&wall, &path], 1);
    let (mut samples, mut partial) = (0, 0);
    for y in 0..h {
        for x in 0..w {
            let area = path.area(x, y);
            let class = class_at(&drawn, w, x, y);
            let got = at(&drawn, w, x, y);
            if area == 0.0 || x >= 21 {
                // Nothing in front of the wall: its pixels take the fast path.
                assert_eq!(got, blue, "pixel {x},{y}");
                assert_eq!(class, Resolve::Fast as u8, "pixel {x},{y}");
            } else if x < 20 {
                assert_eq!(
                    got,
                    blend(red, area_coverage(area as f32), blue),
                    "pixel {x},{y}"
                );
                assert_eq!(class, Resolve::Coverage as u8, "pixel {x},{y}");
            } else {
                // The crossing column: each sub-sample takes what is in front of it there, the
                // path with its exact area inside the sub-sample's cell.
                assert_eq!(class, Resolve::Sample as u8, "pixel {x},{y}");
                samples += 1;
                if area < 1.0 {
                    partial += 1;
                }
                let mut sum = [0u32; 4];
                for s in 0..SAMPLES {
                    let [px, py] = sample_point(x, y, s);
                    let [column, row] = sample_cell(s);
                    let cell = path.area_in(
                        x as f64 + column as f64 / 4.0,
                        y as f64 + row as f64 / 4.0,
                        0.25,
                    );
                    let c = if depth.at(px, py) < 0.0 {
                        blend(red, area_coverage(cell as f32), blue)
                    } else {
                        blue
                    };
                    for k in 0..4 {
                        sum[k] += c[k] as u32;
                    }
                }
                assert_eq!(got, sum.map(|v| ((v + 8) / 16) as u8), "pixel {x},{y}");
            }
        }
    }
    assert!(samples > 10 && partial > 0, "{samples} {partial}");
    // Cells' areas were asked for once per sample-resolve pixel the path covers in part, and
    // nowhere else (where it covers the whole pixel, every cell is full).
    assert_eq!(path.asks.load(Ordering::Relaxed), partial);
}

#[test]
fn the_depth_order_is_read_at_the_sub_samples_not_between_them() {
    let blue = premultiply([0.1, 0.2, 0.9, 1.0]);
    let red = premultiply([0.9, 0.1, 0.1, 0.5]);
    let wall = Wall {
        rect: [0, 0, 8, 8],
        depth: 0.0,
        colour: blue,
        mask: FULL,
        order: 0,
    };
    // Crossing the wall's depth at x = 3.97: right of the pixel's last sub-sample column (3.9375),
    // so in front at every sub-sample; then at x = 3.5, in the middle of them.
    for (cross, class, want) in [
        (3.97, Resolve::Coverage, blend(red, 255, blue)),
        (3.5, Resolve::Sample, [0; 4]),
    ] {
        let near = Single {
            at: (3, 5),
            fragment: Fragment::new(
                Plane {
                    a: -cross,
                    b: 1.0,
                    c: 0.0,
                },
                red,
                Coverage::Mask(FULL),
                1,
                0,
            ),
        };
        let drawn = render(8, 8, &[&wall, &near], 1);
        assert_eq!(
            class_at(&drawn, 8, 3, 5),
            class as u8,
            "crossing at {cross}"
        );
        if class == Resolve::Coverage {
            assert_eq!(at(&drawn, 8, 3, 5), want);
        } else {
            // Half the sub-samples see the red fragment in front (on a tie, the later fragment).
            let half = (0..SAMPLES)
                .filter(|s| sample_point(3, 5, *s)[0] <= cross)
                .count();
            assert_eq!(half, 8);
            let front = blend(red, 255, blue);
            let mean = |k: usize| {
                ((front[k] as usize * half + blue[k] as usize * (SAMPLES - half) + 8) / SAMPLES)
                    as u8
            };
            assert_eq!(at(&drawn, 8, 3, 5), [mean(0), mean(1), mean(2), mean(3)]);
        }
    }
}

#[test]
fn the_opaque_depth_is_readable_before_the_transparent_pass_resolves() {
    let (w, h) = (40, 20);
    let wall_colour = premultiply([0.2, 0.7, 0.3, 1.0]);
    let wall = Wall {
        rect: [0, 0, 20, h],
        depth: 2.0,
        colour: wall_colour,
        mask: FULL,
        order: 0,
    };
    let fog = Fog::new(0.5, 0.4, [0.9, 0.9, 1.0], 1);
    // The producers' list order is not the passes' order: the opaque pass runs first.
    let drawn = render(w, h, &[&fog, &wall], 3);
    for y in 0..h {
        for x in 0..w {
            let want = if x < 20 {
                blend(fog.colour_through(1.5), 255, wall_colour)
            } else {
                blend(fog.colour_through(FAR - 0.5), 255, GREY)
            };
            assert_eq!(at(&drawn, w, x, y), want, "pixel {x},{y}");
        }
    }
    let reads = fog.reads.lock().unwrap();
    assert_eq!(reads.len(), (w * h) as usize);
    for ((x, _), depth) in reads.iter() {
        let want = if *x < 20 { 2.0 } else { f64::INFINITY };
        assert!(depth.iter().all(|d| *d == want), "{x}: {depth:?}");
    }
}

#[test]
fn a_meshs_opaque_depth_is_its_plane_at_each_sub_sample() {
    let (w, h) = (160, 90);
    let mut view = three_d_view(w, h);
    if let Camera::ThreeD(c) = &mut view.camera {
        c.zoom = 1.6;
    }
    let corners = [[-3.0, -2.0, -0.5], [3.0, -1.5, 1.0], [0.0, 3.0, 0.25]];
    let tri = Shape::new(ShapeData::Mesh {
        positions: corners.to_vec(),
        triangles: vec![[0, 1, 2]],
        normals: None,
    });
    let record = mesh_record(1, &tri, [0.5, 0.5, 0.5, 1.0], Space::World);
    let ShapeData::Mesh {
        positions,
        triangles,
        ..
    } = &tri.data()
    else {
        unreachable!()
    };
    let meshes = mesh::lower(
        &view,
        w,
        h,
        &[mesh::MeshDraw {
            record: &record,
            positions,
            triangles,
            normals: None,
            place: IDENTITY,
            fill: [0.5, 0.5, 0.5, 1.0],
            order: 0,
            sub: 0,
        }],
    );
    let fog = Fog::new(-1e9, 0.0, [1.0; 3], 1);
    render(w, h, &[&meshes, &fog], 2);
    let screen: Vec<([f64; 2], f64)> = corners.iter().map(|c| to_pixels(&view, w, h, *c)).collect();
    let tri_px = [screen[0].0, screen[1].0, screen[2].0];
    let inside = pixels_inside(&[tri_px[0], tri_px[1], tri_px[2]], 1.0, w, h);
    assert!(inside.len() > 200);
    let reads = fog.reads.lock().unwrap();
    for ((x, y), depth) in reads.iter() {
        if !inside.contains(&(*x, *y)) {
            continue;
        }
        for (s, d) in depth.iter().enumerate() {
            let p = sample_point(*x, *y, s);
            let b = barycentric(tri_px, p).unwrap();
            let want = b[0] * screen[0].1 + b[1] * screen[1].1 + b[2] * screen[2].1;
            assert!(
                (d - want).abs() < 1e-9,
                "pixel {x},{y} sample {s}: {d} vs {want}"
            );
        }
    }
}

#[test]
fn the_transparent_pass_takes_fragments_from_any_producer_in_depth_order() {
    let wall_colour = premultiply([0.3, 0.3, 0.3, 1.0]);
    let front = premultiply([0.9, 0.1, 0.1, 0.5]);
    let middle = premultiply([0.1, 0.9, 0.1, 0.4]);
    let wall = Wall {
        rect: [0, 0, 6, 6],
        depth: 3.0,
        colour: wall_colour,
        mask: FULL,
        order: 3,
    };
    let fog = Fog::new(1.0, 0.3, [0.8, 0.8, 0.9], 0);
    let near = Single {
        at: (2, 2),
        fragment: Fragment::new(Plane::constant(0.0), front, Coverage::Mask(FULL), 1, 0),
    };
    let between = Single {
        at: (2, 2),
        fragment: Fragment::new(Plane::constant(2.0), middle, Coverage::Mask(FULL), 2, 0),
    };
    let want = blend(
        front,
        255,
        blend(
            fog.colour_through(2.0),
            255,
            blend(middle, 255, wall_colour),
        ),
    );
    let lists: [[&dyn Producer; 4]; 3] = [
        [&wall, &fog, &near, &between],
        [&between, &near, &fog, &wall],
        [&near, &wall, &between, &fog],
    ];
    for producers in lists {
        let drawn = render(6, 6, &producers, 1);
        assert_eq!(at(&drawn, 6, 2, 2), want);
        assert_eq!(class_at(&drawn, 6, 2, 2), Resolve::Coverage as u8);
    }
}

#[test]
fn emitted_light_is_kept_linear_and_tone_mapped_once() {
    let (w, h) = (24, 8);
    let e = [0.8, 0.35, 0.1];
    let half = e.map(|v| v / 2.0);
    let one = Emitter {
        rect: [0, 0, 12, h],
        radiance: e,
        depth: 0.0,
    };
    let (a, b) = (
        Emitter {
            rect: [0, 0, 12, h],
            radiance: half,
            depth: 0.0,
        },
        Emitter {
            rect: [0, 0, 12, h],
            radiance: half,
            depth: 0.5,
        },
    );
    let single = render(w, h, &[&one], 1);
    let split = render(w, h, &[&a, &b], 1);
    let plain = render(w, h, &[], 1);
    // In linear light: each channel moves toward white by 1 - exp(-E), from the pixel's own value.
    let lit = |c: u8, e: f64| {
        let l = srgb_decode(c as f64 / 255.0);
        let out = l + (1.0 - l) * (1.0 - libm::exp(-e));
        (srgb_encode(out) * 255.0 + 0.5) as u8
    };
    let want = [lit(128, e[0]), lit(128, e[1]), lit(128, e[2]), 255];
    assert_eq!(add_emission(GREY, e), want);
    for y in 0..h {
        for x in 0..w {
            if x < 12 {
                assert_eq!(at(&single, w, x, y), want, "pixel {x},{y}");
            } else {
                // Where nothing emits, the frame is the frame drawn without emitters.
                assert_eq!(at(&single, w, x, y), at(&plain, w, x, y));
            }
        }
    }
    // Emission accumulates before the one tone map: two halves are one whole, byte for byte,
    // although tone-mapping each half onto the frame in turn would round differently.
    assert_eq!(split.rgba, single.rgba);
    let mut differs = false;
    for c in (0..=255u8).step_by(5) {
        let px = [c, c, c, 255];
        for v in [0.05, 0.3, 0.7, 1.9] {
            let twice = add_emission(add_emission(px, [v / 2.0; 3]), [v / 2.0; 3]);
            differs |= twice != add_emission(px, [v; 3]);
        }
    }
    assert!(differs, "the check above must be able to fail");
}

#[test]
fn emitted_light_is_hidden_where_the_opaque_depth_is_nearer() {
    let wall_colour = premultiply([0.2, 0.3, 0.5, 1.0]);
    let e = [0.6, 0.6, 0.6];
    // The wall covers sub-samples 0..8 at depth 1; behind it an emitter at depth 5, in front of it
    // one at depth 0.5.
    let wall = Wall {
        rect: [0, 0, 4, 4],
        depth: 1.0,
        colour: wall_colour,
        mask: 0x00FF,
        order: 0,
    };
    let behind = Emitter {
        rect: [0, 0, 2, 4],
        radiance: e,
        depth: 5.0,
    };
    let in_front = Emitter {
        rect: [2, 0, 4, 4],
        radiance: e,
        depth: 0.5,
    };
    let drawn = render(4, 4, &[&wall, &behind, &in_front], 1);
    let partial = blend(wall_colour, mask_coverage(0x00FF), GREY);
    // Behind: only the eight uncovered sub-samples let its light through.
    assert_eq!(
        at(&drawn, 4, 0, 1),
        add_emission(partial, e.map(|v| v / 2.0))
    );
    assert_eq!(at(&drawn, 4, 3, 2), add_emission(partial, e));
}

#[test]
fn synthetic_producers_draw_identically_on_any_number_of_threads() {
    let (w, h) = (211, 157);
    let wall = Wall {
        rect: [30, 20, 180, 140],
        depth: 2.0,
        colour: premultiply([0.2, 0.6, 0.3, 1.0]),
        mask: FULL,
        order: 0,
    };
    let path = HalfPlane::new(
        [0.7, 1.0, 150.0],
        premultiply([0.9, 0.2, 0.2, 0.7]),
        Plane {
            a: -1.0,
            b: 0.03,
            c: 0.002,
        },
    );
    let fog = Fog::new(0.25, 0.2, [0.7, 0.7, 0.8], 2);
    let glow = Emitter {
        rect: [60, 10, 120, 150],
        radiance: [0.4, 0.2, 0.9],
        depth: 1.0,
    };
    let producers: [&dyn Producer; 4] = [&wall, &path, &fog, &glow];
    let first = render(w, h, &producers, 1);
    for threads in [1, 2, 3, 4, 8] {
        let again = render(w, h, &producers, threads);
        assert!(again.rgba == first.rgba, "{threads} threads");
        assert!(again.resolves == first.resolves, "{threads} threads");
        assert_eq!(again.counts, first.counts);
    }
    let c = first.counts;
    assert_eq!(
        c.background + c.fast + c.coverage + c.sample,
        (w * h) as u64
    );
}

#[test]
fn a_pixel_without_emission_keeps_a_translucent_backgrounds_bytes() {
    // CE stores a translucent background unpremultiplied: its channels can exceed its alpha.
    let bg = [200, 100, 50, 128];
    let (w, h) = (80, 8);
    let red = Emitter {
        rect: [0, 0, 4, 8],
        radiance: [0.5, 0.0, 0.0],
        depth: 1.0,
    };
    let dark = Emitter {
        rect: [4, 0, 8, 8],
        radiance: [0.0; 3],
        depth: 1.0,
    };
    let drawn = tiles::render(Frame::filled(w, h, bg), &[&red, &dark], 1);
    let l = srgb_decode(200.0 / 255.0);
    let r = (srgb_encode(l + (1.0 - l) * (1.0 - libm::exp(-0.5))) * 255.0 + 0.5) as u8;
    for y in 0..h {
        for x in 0..w {
            let want = if x < 4 { [r, 100, 50, r.max(128)] } else { bg };
            assert_eq!(at(&drawn, w, x, y), want, "pixel {x},{y}");
        }
    }
}

/// A small deterministic generator, for sweeps.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Fragments stacked in one pixel, through the transparent pass.
struct Stack(Vec<Fragment>);

impl Producer for Stack {
    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        for f in &self.0 {
            pass.add(0, 0, *f);
        }
    }
}

#[test]
fn the_coverage_resolve_is_within_its_stated_bound_of_the_sample_resolve() {
    // A pixel with one partial mask takes the coverage resolve: its 8-bit coverage, rounded from
    // the mask's count, blended once. The sample resolve would blend at each sub-sample and
    // average. They differ by rounding only, within COVERAGE_TO_SAMPLE_LSB.
    let mut rng = Lcg(5);
    let mut colour = |opaque: bool| {
        let a = if opaque { 1.0 } else { rng.unit() };
        premultiply([rng.unit(), rng.unit(), rng.unit(), a])
    };
    let (mut cases, mut widest) = (0, 0);
    for round in 0..400 {
        for count in 1..SAMPLES {
            let mask = ((1u32 << count) - 1) as u16;
            let partial = colour(round % 2 == 0);
            let full = colour(round % 3 == 0);
            let front = colour(false);
            // Back to front: the partial mask alone, over a full layer, under a full layer, and
            // between two.
            let stacks: [Vec<(u16, [u8; 4])>; 4] = [
                vec![(mask, partial)],
                vec![(FULL, full), (mask, partial)],
                vec![(mask, partial), (FULL, front)],
                vec![(FULL, full), (mask, partial), (FULL, front)],
            ];
            for stack in stacks {
                let n = stack.len();
                let fragments: Vec<Fragment> = stack
                    .iter()
                    .enumerate()
                    .map(|(i, (m, c))| {
                        let depth = Plane::constant((n - i) as f64);
                        Fragment::new(depth, *c, Coverage::Mask(*m), i as u32, 0)
                    })
                    .collect();
                let drawn = tiles::render(Frame::filled(1, 1, GREY), &[&Stack(fragments)], 1);
                if drawn.resolves[0] != Resolve::Coverage as u8 {
                    // An opaque full layer in front hides the rest: nothing to compare.
                    assert!(n > 1, "a lone partial mask takes the coverage resolve");
                    continue;
                }
                let mut sum = [0u32; 4];
                for s in 0..SAMPLES {
                    let mut px = GREY;
                    for (m, c) in &stack {
                        if m & (1 << s) != 0 {
                            px = blend(*c, 255, px);
                        }
                    }
                    for k in 0..4 {
                        sum[k] += px[k] as u32;
                    }
                }
                let sample = sum.map(|v| (v + 8) / 16);
                for k in 0..4 {
                    let gap = (drawn.rgba[k] as i64 - sample[k] as i64).unsigned_abs();
                    widest = widest.max(gap);
                    assert!(
                        gap <= COVERAGE_TO_SAMPLE_LSB as u64,
                        "mask {mask:016b} stack {stack:?}: coverage {:?}, sample {sample:?}",
                        &drawn.rgba[..4]
                    );
                }
                cases += 1;
            }
        }
    }
    assert!(cases > 10_000, "{cases} cases");
    if std::env::var_os("CORE_STATS").is_some() {
        eprintln!("coverage vs sample resolve: widest gap {widest} over {cases} cases");
    }
}

#[test]
fn a_paths_exact_area_lands_within_the_bound_on_either_resolve() {
    // A path's edge in a pixel: the coverage resolve blends its exact area once; the sample
    // resolve blends, at each sub-sample, its exact area inside that sub-sample's cell, and
    // averages. Alone, over an opaque wall and under a translucent layer, every edge angle and
    // offset lands within COVERAGE_TO_SAMPLE_LSB of the other resolve.
    let (w, h) = (24, 24);
    let mut rng = Lcg(11);
    let (mut pixels, mut widest) = (0, 0);
    for round in 0..120 {
        let turn = rng.unit() * std::f64::consts::PI;
        let (a, b) = (libm::cos(turn), libm::sin(turn));
        let c = a * 12.0 + b * 12.0 + rng.unit() - 0.5;
        let alpha = if round % 2 == 0 { 1.0 } else { rng.unit() };
        let colour = premultiply([rng.unit(), rng.unit(), rng.unit(), alpha]);
        let path = HalfPlane::new([a, b, c], colour, Plane::constant(1.0));
        let wall = Wall {
            rect: [0, 0, w, h],
            depth: 2.0,
            colour: premultiply([rng.unit(), rng.unit(), rng.unit(), 1.0]),
            mask: FULL,
            order: 0,
        };
        let front = premultiply([rng.unit(), rng.unit(), rng.unit(), 0.5]);
        let over = Veil(front);
        let stacks: [Vec<&dyn Producer>; 3] = [vec![&path], vec![&wall, &path], vec![&path, &over]];
        for stack in stacks {
            let drawn = tiles::render(Frame::filled(w, h, GREY), &stack, 1);
            let mut with = stack.clone();
            with.push(&EverySample);
            let sampled = tiles::render(Frame::filled(w, h, GREY), &with, 1);
            for y in 0..h {
                for x in 0..w {
                    let area = path.area(x, y);
                    if area <= 0.0 || area >= 1.0 {
                        continue;
                    }
                    let i = (y * w + x) as usize * 4;
                    assert_eq!(drawn.resolves[i / 4], Resolve::Coverage as u8);
                    pixels += 1;
                    for k in 0..4 {
                        let gap = (drawn.rgba[i + k] as i64 - sampled.rgba[i + k] as i64).abs();
                        widest = widest.max(gap);
                        assert!(
                            gap <= COVERAGE_TO_SAMPLE_LSB as i64,
                            "pixel {x},{y}, line {:?}: coverage {:?}, sample {:?}",
                            path.line,
                            &drawn.rgba[i..i + 4],
                            &sampled.rgba[i..i + 4]
                        );
                    }
                }
            }
        }
    }
    assert!(pixels > 5_000, "{pixels} pixels");
    if std::env::var_os("CORE_STATS").is_some() {
        eprintln!(
            "a path's area, coverage vs sample resolve: widest gap {widest} over {pixels} pixels"
        );
    }
}

/// A translucent layer over every pixel, in front of everything.
struct Veil([u8; 4]);

impl Producer for Veil {
    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        let r = pass.rect();
        for (x, y) in rect_pixels(r, r.x0, r.y0, r.x1, r.y1) {
            let f = Fragment::new(Plane::constant(0.0), self.0, Coverage::Mask(FULL), 9, 0);
            pass.add(x, y, f);
        }
    }
}
