//! Coverage against closed forms (contract §3 and §4): cell areas of half-planes, a disc, strokes
//! of known width, a mesh edge's margin, point blocks, and a thin line dropped and restored; an
//! unbounded triangle, one with corners far out, the guard, and a blend beyond its shapes.

mod common;

use arachne_core::xf::{
    Body, Joint, Paint, Points, Record, RecordKind, Shape, ShapeData, Space, Stops,
};
use arachne_oracle::cover::{SAMPLE_OFFSETS, sample_cell, sample_point};
use arachne_oracle::out::{Cover, Decisions};
use common::*;

const YES: u8 = Cover::Yes as u8;
const EITHER: u8 = Cover::Either as u8;
const ABSENT: u8 = Cover::Absent as u8;

/// The area of the box `[x0, x1] x [y0, y1]` on the side `n · (p − q) <= 0` of a line: the box
/// clipped by the half-plane, by the shoelace formula.
fn box_in_half_plane(b: [f64; 4], n: [f64; 2], q: [f64; 2]) -> f64 {
    let poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
    let side = |p: [f64; 2]| n[0] * (p[0] - q[0]) + n[1] * (p[1] - q[1]);
    let mut out: Vec<[f64; 2]> = Vec::new();
    for i in 0..4 {
        let (a, c) = (poly[i], poly[(i + 1) % 4]);
        let (sa, sc) = (side(a), side(c));
        if sa <= 0.0 {
            out.push(a);
        }
        if (sa < 0.0 && sc > 0.0) || (sa > 0.0 && sc < 0.0) {
            let t = sa / (sa - sc);
            out.push([a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t]);
        }
    }
    let mut area = 0.0;
    for i in 0..out.len() {
        let (a, c) = (out[i], out[(i + 1) % out.len()]);
        area += a[0] * c[1] - a[1] * c[0];
    }
    area.abs() / 2.0
}

/// The box of sample `s`'s cell of pixel `(x, y)`.
fn cell_box(x: u32, y: u32, s: usize) -> [f64; 4] {
    let (c, r) = sample_cell(s);
    let (x0, y0) = (x as f64 + c as f64 / 4.0, y as f64 + r as f64 / 4.0);
    [x0, y0, x0 + 0.25, y0 + 0.25]
}

fn only_rows(
    d: &Decisions,
) -> std::collections::HashMap<(u32, u32), &arachne_oracle::out::CoverRow> {
    assert_eq!(d.fragments.len(), 1, "{:?}", d.fragments);
    d.covers.iter().map(|c| ((c.x, c.y), c)).collect()
}

#[test]
fn a_half_planes_cells_take_their_exact_areas() {
    let v = view(16, 16);
    for angle in [0.0f64, 30.0, 45.0, 90.0] {
        let a = angle.to_radians();
        let n = [libm::cos(a), libm::sin(a)];
        let t = [-n[1], n[0]];
        for k in 0..64 {
            let off = k as f64 / 64.0;
            let q = [8.0 + off * n[0], 8.0 + off * n[1]];
            let at = |u: f64, w: f64| [q[0] + u * t[0] - w * n[0], q[1] + u * t[1] - w * n[1]];
            let corners = [
                at(-300.0, 0.0),
                at(300.0, 0.0),
                at(300.0, 300.0),
                at(-300.0, 300.0),
            ];
            let s = path_shape(polygon(&corners.map(|p| frame(&v, p))));
            let d = decide(&[delta(
                1,
                Some(v.clone()),
                vec![s.clone()],
                vec![path(0, s.key(), true, None)],
            )]);
            let rows = only_rows(&d);
            for y in 0..16 {
                for x in 0..16 {
                    for s in 0..16 {
                        let want = 16.0 * box_in_half_plane(cell_box(x, y, s), n, q);
                        match rows.get(&(x, y)) {
                            Some(r) => {
                                let got = r.cell_area[s];
                                assert!(
                                    (got - want).abs() <= 1e-12,
                                    "angle {angle} offset {k}/64 pixel ({x}, {y}) sample {s}: \
                                     {got} against {want}"
                                );
                                let code = r.cover[s];
                                if want >= 1.0 / 510.0 + 1e-9 {
                                    assert_eq!(code, YES);
                                } else if want < 1.0 / 510.0 - 1e-9 {
                                    assert_eq!(code, ABSENT);
                                }
                            }
                            None => assert!(want < 1.0 / 510.0, "({x}, {y}) {s}: {want}"),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn a_discs_coverage_is_its_area_within_the_flattening_tolerance() {
    let v = view(64, 64);
    let r = 20.0;
    let c = [32.0, 32.0];
    let s = path_shape(circle(frame(&v, c), r));
    let d = decide(&[delta(
        1,
        Some(v),
        vec![s.clone()],
        vec![path(0, s.key(), true, None)],
    )]);
    let total: f64 = d.covers.iter().map(|c| c.pixel_area).sum();
    // Flattening at 0.1 px moves the outline by at most 0.1 px; eight cubic arcs stand within
    // 1e-5 of the radius.
    let bound = 2.0 * std::f64::consts::PI * r * (0.1 + 1e-5 * r);
    let want = std::f64::consts::PI * r * r;
    assert!(
        (total - want).abs() <= bound,
        "{total} against {want} within {bound}"
    );
    let rows = only_rows(&d);
    for y in 0..64u32 {
        for x in 0..64u32 {
            let corners = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
                .map(|o: [f64; 2]| libm::hypot(x as f64 + o[0] - c[0], y as f64 + o[1] - c[1]));
            let far = corners.iter().copied().fold(0.0, f64::max);
            let near = libm::hypot(
                (c[0] - c[0].clamp(x as f64, x as f64 + 1.0)).abs(),
                (c[1] - c[1].clamp(y as f64, y as f64 + 1.0)).abs(),
            );
            if far < r - 0.2 {
                let row = rows[&(x, y)];
                assert_eq!(row.pixel_area, 1.0);
                assert_eq!(row.cover, [YES; 16]);
            } else if near > r + 0.2 {
                assert!(!rows.contains_key(&(x, y)), "({x}, {y})");
            }
        }
    }
}

/// A horizontal stroke `w` px wide along pixel row position `y0`, across the frame and beyond.
fn band(w: f64, y0: f64) -> (Shape, Record, arachne_core::xf::View) {
    let v = view(16, 16);
    let s = path_shape(polyline(&[frame(&v, [-50.0, y0]), frame(&v, [66.0, y0])]));
    let r = path(0, s.key(), false, Some(w));
    (s, r, v)
}

#[test]
fn a_strokes_cells_take_the_exact_areas_of_its_rectangle() {
    for (w, y0) in [(1.0, 8.5), (1.0, 8.3), (0.3, 8.5), (0.3, 8.3), (0.3, 8.0)] {
        let (s, r, v) = band(w, y0);
        let d = decide(&[delta(1, Some(v), vec![s], vec![r])]);
        let rows = only_rows(&d);
        let (lo, hi) = (y0 - w / 2.0, y0 + w / 2.0);
        for y in 0..16 {
            for x in 0..16 {
                for s in 0..16 {
                    let b = cell_box(x, y, s);
                    let want = 16.0 * 0.25 * (hi.min(b[3]) - lo.max(b[1])).max(0.0);
                    let got = rows.get(&(x, y)).map_or(0.0, |r| r.cell_area[s]);
                    assert!(
                        (got - want).abs() <= 1e-12,
                        "{w} px at {y0}: pixel ({x}, {y}) sample {s}: {got} against {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_thin_line_dropped_leaves_no_fragment_and_restored_returns() {
    let (s, r, v) = band(0.3, 8.3);
    let first = delta(1, Some(v), vec![s], vec![r]);
    let mut dropped = delta(2, None, vec![], vec![]);
    dropped.order = Some(vec![]);
    let mut restored = delta(3, None, vec![], vec![]);
    restored.order = Some(vec![entry(0)]);
    let deltas = [first, dropped, restored];
    let at = |seq| arachne_oracle::decide::decide_at(&deltas, seq, None).unwrap();
    let present = at(1);
    assert!(present.covers.iter().any(|c| c.cover.contains(&YES)));
    let gone = at(2);
    assert!(gone.fragments.is_empty() && gone.covers.is_empty());
    assert_eq!(cover_lines(&at(3)), cover_lines(&present));
}

#[test]
fn a_self_overlapping_outline_fills_by_the_nonzero_rule() {
    // Two squares wound the same way in one path: where they overlap the winding number is 2,
    // which the nonzero rule fills once.
    let v = view(16, 16);
    let sq = |x0: f64, y0: f64, x1: f64, y1: f64| {
        polygon(&[
            frame(&v, [x0, y0]),
            frame(&v, [x1, y0]),
            frame(&v, [x1, y1]),
            frame(&v, [x0, y1]),
        ])
    };
    let mut points = sq(2.0, 2.0, 10.0, 10.0);
    points.extend(sq(6.0, 6.0, 14.0, 14.0));
    let s = path_shape(points);
    let d = decide(&[delta(
        1,
        Some(v),
        vec![s.clone()],
        vec![path(0, s.key(), true, None)],
    )]);
    let rows = only_rows(&d);
    assert_eq!(rows[&(7, 7)].pixel_area, 1.0);
    assert_eq!(rows[&(7, 7)].cover, [YES; 16]);
    let total: f64 = d.covers.iter().map(|c| c.pixel_area).sum();
    assert!((total - (64.0 + 64.0 - 16.0)).abs() < 1e-9, "{total}");
}

#[test]
fn a_record_outside_the_frame_yields_no_fragment() {
    let v = view(16, 16);
    let far = |x: f64| frame(&v, [x, 4.0]);
    let s = path_shape(polygon(&[far(100.0), far(110.0), frame(&v, [110.0, 9.0])]));
    let mesh = Shape::new(ShapeData::Mesh {
        positions: vec![far(-40.0), far(-20.0), frame(&v, [-30.0, 12.0])],
        triangles: vec![[0, 1, 2]],
        normals: None,
    });
    let mut m = Record::new(RecordKind::Mesh, 1, 101);
    m.geometry.shape_a = Some(mesh.key());
    m.paint.fill = Paint::One([1.0, 0.0, 0.0, 1.0]);
    let d = decide(&[delta(
        1,
        Some(v),
        vec![s.clone(), mesh],
        vec![path(0, s.key(), true, Some(2.0)), m],
    )]);
    assert!(
        d.fragments.is_empty() && d.covers.is_empty(),
        "{:?}",
        d.fragments
    );
}

/// A mesh of one large triangle whose lower edge lies along pixel row position `ye`, inside
/// above it, under the view's grid `grid`.
fn edge_mesh(ye: f64, grid: f64) -> Decisions {
    let mut v = view(16, 16);
    v.input_grid = grid;
    let mesh = Shape::new(ShapeData::Mesh {
        positions: vec![
            frame(&v, [-100.0, ye]),
            frame(&v, [200.0, ye]),
            frame(&v, [50.0, -400.0]),
        ],
        triangles: vec![[0, 1, 2]],
        normals: None,
    });
    let mut m = Record::new(RecordKind::Mesh, 0, 100);
    m.geometry.shape_a = Some(mesh.key());
    m.paint.fill = Paint::One([1.0, 1.0, 1.0, 1.0]);
    decide(&[delta(1, Some(v), vec![mesh], vec![m])])
}

#[test]
fn a_mesh_sample_is_yes_beyond_the_margin_inside_and_either_within_it() {
    // Sample 9 sits at y = 8.5 + 0.5/16 in pixel (x, 8); the edge lies 0.005 px below it.
    let ye = 8.5 + 0.5 / 16.0 + 0.005;
    let at = |d: &Decisions, x: u32, y: u32| {
        d.covers
            .iter()
            .find(|c| (c.x, c.y) == (x, y))
            .map(|c| c.cover)
    };
    let plain = edge_mesh(ye, 0.0);
    for s in 0..16 {
        let dist = ye - sample_point(5, 8, s)[1];
        let want = if dist > 1.0 / 256.0 + 1e-9 {
            YES
        } else if dist < -(1.0 / 256.0 + 1e-9) {
            ABSENT
        } else {
            EITHER
        };
        assert_eq!(at(&plain, 5, 8).unwrap()[s], want, "sample {s} at {dist}");
    }
    assert_eq!(at(&plain, 5, 8).unwrap()[9], YES);
    // 0.002 px inside is within the 1/256 px the corners' rounding may move an edge.
    let near = edge_mesh(8.5 + 0.5 / 16.0 + 0.002, 0.0);
    assert_eq!(at(&near, 5, 8).unwrap()[9], EITHER);
    assert_eq!(at(&plain, 5, 3).unwrap(), [YES; 16]);
    assert_eq!(at(&plain, 5, 10), None);
    // A grid of 0.01 lets each corner move by hypot(0.005, 0.005): the sample 0.005 px inside is
    // now within the margin, and the one a sixteenth inside is still beyond it.
    let snapped = edge_mesh(ye, 0.01);
    let cover = at(&snapped, 5, 8).unwrap();
    assert_eq!(cover[9], EITHER);
    assert_eq!(cover[4], YES);
    assert_eq!(cover[11], ABSENT);
    let row = snapped
        .covers
        .iter()
        .find(|c| (c.x, c.y) == (5, 8))
        .unwrap();
    let yes = cover.iter().filter(|&&c| c == YES).count() as u32;
    let any = cover.iter().filter(|&&c| c != ABSENT).count() as u32;
    assert_eq!(
        (row.pixel_lo, row.pixel_hi),
        (((yes * 255 + 8) / 16) as u8, ((any * 255 + 8) / 16) as u8)
    );
}

#[test]
fn an_op_whose_alpha_rounds_to_zero_paints_nothing() {
    let v = view(16, 16);
    let s = path_shape(polygon(&[
        frame(&v, [2.0, 2.0]),
        frame(&v, [10.0, 2.0]),
        frame(&v, [10.0, 10.0]),
    ]));
    let with_alpha = |a: f64| {
        let mut r = path(0, s.key(), true, None);
        r.paint.fill = Paint::One([1.0, 1.0, 1.0, a]);
        decide(&[delta(1, Some(v.clone()), vec![s.clone()], vec![r])])
    };
    assert!(with_alpha(255.4 / 65535.0).fragments.is_empty());
    assert!(!with_alpha(255.6 / 65535.0).fragments.is_empty());
    let mesh = Shape::new(ShapeData::Mesh {
        positions: vec![
            frame(&v, [2.0, 2.0]),
            frame(&v, [10.0, 2.0]),
            frame(&v, [10.0, 10.0]),
        ],
        triangles: vec![[0, 1, 2]],
        normals: None,
    });
    let mut m = Record::new(RecordKind::Mesh, 0, 100);
    m.geometry.shape_a = Some(mesh.key());
    m.paint.fill = Paint::One([1.0, 1.0, 1.0, 0.001]);
    assert!(
        decide(&[delta(1, Some(v), vec![mesh], vec![m])])
            .fragments
            .is_empty()
    );
}

fn points(at: [f64; 2], radius: f64, grid: f64) -> Decisions {
    let mut v = view(16, 16);
    v.input_grid = grid;
    let shape = path_shape(vec![frame(&v, at)]);
    let colours = Stops::new(vec![[1.0, 1.0, 1.0, 1.0]]);
    let mut r = Record::new(RecordKind::Points, 0, 100);
    r.geometry.shape_a = Some(shape.key());
    r.depth.space = Space::Screen;
    r.body = Body::Points(Points {
        colors: colours.key(),
        radius,
    });
    let mut d = delta(1, Some(v), vec![shape], vec![r]);
    d.upload_paints = vec![colours];
    decide(&[d])
}

#[test]
fn a_point_block_covers_whole_pixels_about_its_truncated_pixel() {
    let d = points([5.3, 7.6], 3.0, 0.0);
    let mut pixels: Vec<(u32, u32)> = d.covers.iter().map(|c| (c.x, c.y)).collect();
    pixels.sort();
    let want: Vec<(u32, u32)> = (4..7).flat_map(|x| (6..9).map(move |y| (x, y))).collect();
    assert_eq!(pixels, want);
    assert!(
        d.covers
            .iter()
            .all(|c| c.cover == [YES; 16] && c.pixel_lo == 255)
    );
    // Each pixel is its own fragment, numbered by its offset, offset-major.
    let mut subs: Vec<u32> = d.fragments.iter().map(|f| f.sub).collect();
    subs.sort();
    assert_eq!(subs, (0..9).collect::<Vec<_>>());
    // Near an integer, where the grid lets the truncation move, both blocks are Either.
    let moved = points([5.001, 7.6], 3.0, 0.01);
    let mut xs: Vec<u32> = moved.covers.iter().map(|c| c.x).collect();
    xs.sort();
    xs.dedup();
    assert_eq!(xs, vec![3, 4, 5, 6]);
    assert!(moved.covers.iter().all(|c| c.cover == [EITHER; 16]));
}

#[test]
fn a_carried_bevel_cuts_the_miter_kurbo_would_draw() {
    let v = view(64, 64);
    let s = path_shape(polyline(&[
        frame(&v, [20.0, 40.0]),
        frame(&v, [40.0, 40.0]),
        frame(&v, [40.0, 20.0]),
    ]));
    let total = |joins: Option<Vec<u8>>, joint: Joint| {
        let mut r = path(0, s.key(), false, Some(4.0));
        r.joins = joins;
        r.paint.joint = joint;
        let d = decide(&[delta(1, Some(v.clone()), vec![s.clone()], vec![r])]);
        d.covers.iter().map(|c| c.pixel_area).sum::<f64>()
    };
    let mitred = total(None, Joint::Auto);
    let bevel = total(None, Joint::Bevel);
    // The miter's triangle beyond the bevel at a right-angled turn of a 4 px stroke.
    assert!((mitred - bevel - 2.0).abs() < 1e-9, "{mitred} - {bevel}");
    // Bit 0 on the second cubic: Cairo joins into it; bit 1 clear: it bevels there. The edit
    // leaves kurbo's own bevel.
    let carried = total(Some(vec![0, 1]), Joint::Auto);
    assert!((carried - bevel).abs() < 1e-9, "{carried} against {bevel}");
    // A carried miter where kurbo mitres anyway changes nothing; a round joint is never edited.
    assert_eq!(total(Some(vec![0, 3]), Joint::Auto), mitred);
    assert_eq!(
        total(Some(vec![0, 1]), Joint::Round),
        total(None, Joint::Round)
    );
}

#[test]
fn the_sample_pattern_is_n_rooks_with_one_sample_per_cell() {
    let mut cols = [0; 16];
    let mut rows = [0; 16];
    let mut cells = [0; 16];
    for (s, [dx, dy]) in SAMPLE_OFFSETS.iter().enumerate() {
        cols[(dx + 8) as usize] += 1;
        rows[(dy + 8) as usize] += 1;
        let (c, r) = sample_cell(s);
        cells[r * 4 + c] += 1;
    }
    assert_eq!((cols, rows, cells), ([1; 16], [1; 16], [1; 16]));
    // The fixed-point position the contract states is the same point.
    for (s, &[dx, dy]) in SAMPLE_OFFSETS.iter().enumerate() {
        let p = sample_point(0, 0, s);
        assert_eq!(p[0] * 256.0, (128 + 16 * dx as i32 + 8) as f64);
        assert_eq!(p[1] * 256.0, (128 + 16 * dy as i32 + 8) as f64);
    }
}

/// A 3D view looking straight down the z axis (theta −90°, no tilt: the identity rotation) from
/// focal distance 20, its frame units pixels, under the grid `grid`.
fn view_3d(grid: f64) -> arachne_core::xf::View {
    let mut v = view(16, 16);
    v.camera = arachne_core::xf::Camera::ThreeD(arachne_core::xf::ThreeDCamera {
        phi: 0.0,
        theta: -std::f64::consts::FRAC_PI_2,
        gamma: 0.0,
        focal_distance: 20.0,
        zoom: 1.0,
        exponential_projection: false,
        light_source: [-7.0, -9.0, 10.0],
        shading_factor: 0.2,
        should_apply_shading: false,
    });
    v.input_grid = grid;
    v
}

/// A world-space mesh of the one triangle `corners`, filled white, decided over `window`.
fn world_triangle(
    v: arachne_core::xf::View,
    corners: [[f64; 3]; 3],
    window: Option<[u32; 4]>,
) -> Decisions {
    let mesh = Shape::new(ShapeData::Mesh {
        positions: corners.to_vec(),
        triangles: vec![[0, 1, 2]],
        normals: None,
    });
    let mut m = Record::new(RecordKind::Mesh, 0, 100);
    m.geometry.shape_a = Some(mesh.key());
    m.paint.fill = Paint::One([1.0, 1.0, 1.0, 1.0]);
    m.depth.space = Space::World;
    arachne_oracle::decide::decide_at(&[delta(1, Some(v), vec![mesh], vec![m])], 1, window).unwrap()
}

#[test]
fn a_triangle_through_the_eye_is_unable_over_the_window_never_absent() {
    // One corner at the eye's depth: the depth share is infinite, so the screen shift is, and the
    // triangle may stand anywhere. Its nominal corners all lie far right of and below the window.
    let corners = [
        [100.0, -100.0, 0.0],
        [120.0, -100.0, 0.0],
        [110.0, -120.0, 20.0],
    ];
    for (window, bbox) in [(None, [0, 0, 16, 16]), (Some([0, 0, 2, 2]), [0, 0, 2, 2])] {
        let d = world_triangle(view_3d(1.0 / 1_048_576.0), corners, window);
        assert_eq!(d.fragments.len(), 1, "{:?}", d.fragments);
        let f = &d.fragments[0];
        assert_eq!(f.flags & arachne_oracle::out::FLAG_UNBOUNDED, 1);
        assert!(f.epsilon.is_infinite());
        // Present but unjudged at every sample of the window: its box is the window, and it has
        // no rows that could say absent anywhere.
        assert_eq!(f.bbox, bbox);
        assert!(d.covers.is_empty());
    }
    // In front of the eye the same triangle is bounded and, off the window, has no fragment.
    let near = [
        [100.0, -100.0, 0.0],
        [120.0, -100.0, 0.0],
        [110.0, -120.0, 5.0],
    ];
    let d = world_triangle(view_3d(1.0 / 1_048_576.0), near, None);
    assert!(d.fragments.is_empty(), "{:?}", d.fragments);
}

#[test]
fn a_triangle_with_corners_far_out_keeps_its_verdicts() {
    // Its lower edge runs along y = 12; the third corner lies far above. Inside above the edge,
    // absent below it.
    let tri = |far: f64| {
        arachne_oracle::cover::triangle(
            [[-far, 12.0], [far, 12.0], [0.0, -far]],
            [0.0; 3],
            [0, 0, 16, 16],
        )
    };
    let at = |rows: &[arachne_oracle::cover::PixelCover], x: u32, y: u32| {
        rows.iter().find(|r| (r.x, r.y) == (x, y)).map(|r| r.cover)
    };
    // Corners 10^6 px out, past the guard: the readings are exact to far below the margin.
    let rows = tri(1e6);
    assert_eq!(at(&rows, 5, 3), Some([YES; 16]));
    // Pixel row 11's samples lie 1/32 px and more above the edge: inside by more than 1/256.
    assert_eq!(at(&rows, 9, 11), Some([YES; 16]));
    assert_eq!(at(&rows, 5, 12), None);
    // Corners 10^200 px out and up to binary64's largest: an edge read from a corner that far
    // carries that corner's rounding, so a verdict may be Either, but a sample inside is never
    // absent and one outside never Yes.
    for far in [1e200, 1e300, 1e307, 1e308, f64::MAX] {
        let rows = tri(far);
        for y in 0..16u32 {
            for x in 0..16u32 {
                let cover = at(&rows, x, y).unwrap_or([ABSENT; 16]);
                for (s, c) in cover.iter().enumerate() {
                    let inside = sample_point(x as i64, y as i64, s)[1] < 12.0;
                    let wrong = if inside { ABSENT } else { YES };
                    assert_ne!(*c, wrong, "far {far}, pixel ({x}, {y}) sample {s}");
                }
            }
        }
    }
    // Drawn, such a triangle's corners carry their own placing's rounding, `8u` of 10^200 px:
    // every sample of the window is within that of an edge, so none is absent and none is sure.
    let v = view(16, 16);
    let d = world_triangle(
        v.clone(),
        [
            frame(&v, [-1e200, 12.0]),
            frame(&v, [1e200, 12.0]),
            frame(&v, [0.0, -1e200]),
        ],
        None,
    );
    assert_eq!(d.covers.len(), 256);
    assert!(d.covers.iter().all(|c| c.cover == [EITHER; 16]));
}

#[test]
fn a_triangle_whose_corners_differ_past_binary64_keeps_its_winding() {
    // Corner differences overflow (2e308), so the winding is read on the corners scaled. The
    // window lies about 5e299 px inside every edge, past a margin of 8.9e292 px (an input rounding
    // of that order), so it is covered whole.
    let corners = [[-1e308, -1e300], [1e308, -1e300], [1e308, 2e300]];
    let margin = 8.9e292;
    for c in [corners, [corners[0], corners[2], corners[1]]] {
        let rows = arachne_oracle::cover::triangle(c, [margin; 3], [0, 0, 16, 16]);
        assert_eq!(rows.len(), 256);
        assert!(rows.iter().all(|r| r.cover == [YES; 16]));
    }
    // Collinear corners (a winding no reading settles) cover nothing for sure: absent away from
    // their line, Either on it, never Yes. The line runs through sample 9 of pixel row 8.
    let y9 = sample_point(0, 8, 9)[1];
    let line = arachne_oracle::cover::triangle(
        [[-100.0, y9], [0.0, y9], [100.0, y9]],
        [0.0; 3],
        [0, 0, 16, 16],
    );
    assert_eq!(line.len(), 16);
    for r in &line {
        assert_eq!(r.y, 8);
        let mut want = [ABSENT; 16];
        want[9] = EITHER;
        assert_eq!(r.cover, want);
    }
}

#[test]
fn the_guard_clips_a_triangle_at_65536_px() {
    let rows = arachne_oracle::cover::triangle(
        [[-1e9, 4.0], [1e9, 4.0], [0.0, -1e9]],
        [0.0; 3],
        [65_530, 0, 65_540, 2],
    );
    let at = |x: u32| rows.iter().find(|r| (r.x, r.y) == (x, 1)).map(|r| r.cover);
    assert_eq!(at(65_534), Some([YES; 16]));
    assert_eq!(at(65_535), Some([YES; 16]));
    // Past the guard by more than 1/256 px: clipped away.
    assert_eq!(at(65_536), None);
    assert_eq!(at(65_539), None);
}

#[test]
fn a_blend_beyond_its_shapes_moves_by_each_coefficient() {
    // `(1 − t) p + t q` with t = 2 is `−p + 2q`: under the grid each of p and q moves by up to
    // the grid's half, so the drawn point by three halves; at t = 1/2, by one.
    let grid = 1.0 / 1024.0;
    let eps_at = |t: f64| {
        let mut v = view(16, 16);
        v.input_grid = grid;
        let sq = |x0: f64| {
            path_shape(polygon(&[
                frame(&v, [x0, 4.0]),
                frame(&v, [x0 + 4.0, 4.0]),
                frame(&v, [x0 + 4.0, 8.0]),
                frame(&v, [x0, 8.0]),
            ]))
        };
        let (a, b) = (sq(4.0), sq(5.0));
        let mut r = path(0, a.key(), true, None);
        r.geometry.blend = Some(arachne_core::xf::Blend {
            shape_b: b.key(),
            place_b: arachne_core::xf::IDENTITY,
            t,
        });
        let d = decide(&[delta(1, Some(v), vec![a, b], vec![r])]);
        d.fragments[0].epsilon
    };
    let one = grid / 2.0 * std::f64::consts::SQRT_2;
    let (mid, out) = (eps_at(0.5), eps_at(2.0));
    assert!((mid - one).abs() < one * 1e-6, "{mid} against {one}");
    assert!(out >= 3.0 * one, "{out} against {}", 3.0 * one);
    assert!(
        (out - 3.0 * one).abs() < one * 1e-6,
        "{out} against {}",
        3.0 * one
    );
}
