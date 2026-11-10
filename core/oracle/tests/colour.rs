//! Colour from XF (contract §9, §1's `alpha_round@1`): solids through `cairo_premultiply@1`,
//! point bytes, CE's shading of a lit corner, and what paints at the alpha band's edges.

mod common;

use arachne_core::xf::{Paint, Shape, ShapeData};
use arachne_oracle::colour::{ce_shade, ce_unit_normal, place_normal, point_bytes, premultiply};
use arachne_oracle::out::{COLOUR_GRADIENT, COLOUR_SOLID};
use common::*;

/// `cairo_premultiply@1` written from the contract's text.
fn premultiply_by_text(c: [f64; 4], opacity: f64) -> [u8; 4] {
    let a = (c[3] * opacity).clamp(0.0, 1.0);
    let q = |v: f64| (((v * 65535.0 + 0.5) as u32) >> 8) as u8;
    [
        q(c[0].clamp(0.0, 1.0) * a),
        q(c[1].clamp(0.0, 1.0) * a),
        q(c[2].clamp(0.0, 1.0) * a),
        q(a),
    ]
}

#[test]
fn a_solid_is_premultiplied_as_cairo_quantises_it() {
    // The intersecting-surfaces probe's red sheet: the bytes its check asserts.
    assert_eq!(premultiply([0.85, 0.2, 0.15, 1.0], 1.0), [217, 51, 38, 255]);
    for (c, o) in [
        ([0.9, 0.15, 0.1, 0.5], 1.0),
        ([0.1, 0.3, 0.95, 0.4], 0.75),
        ([1.5, -0.2, 0.5, 1.0], 0.5),
        ([0.2, 0.6, 0.3, 2.0], 1.0),
    ] {
        assert_eq!(premultiply(c, o), premultiply_by_text(c, o), "{c:?} {o}");
    }
}

#[test]
fn a_point_block_s_bytes_are_ce_s() {
    assert_eq!(
        point_bytes([1.0, 0.5, 0.0, 1.0], 1.0),
        Some([255, 127, 0, 255])
    );
    // Alpha byte 127: each channel becomes (c a + 127) div 255.
    assert_eq!(
        point_bytes([1.0, 0.5, 0.2, 0.5], 1.0),
        Some([
            (255 * 127 + 127) / 255,
            (127 * 127 + 127) / 255,
            (51 * 127 + 127) / 255,
            127
        ])
        .map(|v| v.map(|c| c as u8))
    );
    // No block where the alpha byte is 0.
    assert_eq!(point_bytes([1.0, 1.0, 1.0, 0.003], 1.0), None);
}

#[test]
fn a_corner_is_shaded_as_ce_s_get_shaded_rgb_shades_it() {
    // CE's get_shaded_rgb: to_sun the unit vector toward the light, factor 0.5 (n · to_sun)^3,
    // halved where negative, added to every channel.
    let light: [f64; 3] = [-7.0, -9.0, 10.0];
    for (n, p) in [
        ([0.0, 0.0, 1.0], [1.0, 2.0, 0.0]),
        ([0.0, 0.0, -1.0], [1.0, 2.0, 0.0]),
        ([0.6, 0.0, 0.8], [-3.0, 0.5, 1.25]),
    ] {
        let d = [light[0] - p[0], light[1] - p[1], light[2] - p[2]];
        let l = libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        let dot = (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) / l;
        let mut f = 0.5 * (dot * dot * dot);
        if f < 0.0 {
            f *= 0.5;
        }
        let rgb = [0.95, 0.5, 0.1];
        let got = ce_shade(rgb, n, p, light);
        for k in 0..3 {
            assert!((got[k] - (rgb[k] + f)).abs() <= 1e-9, "{n:?} {k}");
        }
    }
}

#[test]
fn ce_s_unit_normal_follows_its_degenerate_cases() {
    assert_eq!(
        ce_unit_normal([2.0, 0.0, 0.0], [0.0, 3.0, 0.0]),
        [0.0, 0.0, 1.0]
    );
    // Both edges zero.
    assert_eq!(ce_unit_normal([0.0; 3], [0.0; 3]), [0.0, -1.0, 0.0]);
    // One zero: the other alone, along z, has no x or y: (0, −1, 0).
    assert_eq!(ce_unit_normal([0.0; 3], [0.0, 0.0, 5.0]), [0.0, -1.0, 0.0]);
    // Collinear edges: from u = (1, 0, 1), (−u_x u_z, −u_y u_z, u_x² + u_y²) normalised.
    let n = ce_unit_normal([1.0, 0.0, 1.0], [2.0, 0.0, 2.0]);
    let s = 1.0 / libm::sqrt(2.0);
    assert!(
        (n[0] + s).abs() < 1e-15 && n[1] == 0.0 && (n[2] - s).abs() < 1e-15,
        "{n:?}"
    );
}

#[test]
fn a_carried_normal_turns_through_the_placement_s_cofactors() {
    // A scale of 2 along x: a normal along (1, 1, 0) turns toward y.
    let m = [
        [2.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let n = place_normal(&m, [1.0, 1.0, 0.0]);
    let l = libm::sqrt(5.0);
    assert!(
        (n[0] - 1.0 / l).abs() < 1e-15 && (n[1] - 2.0 / l).abs() < 1e-15,
        "{n:?}"
    );
}

/// A 16 x 16 square path at fill `rgba`, decided.
fn square(rgba: [f64; 4], rows: Option<Paint>) -> arachne_oracle::out::Decisions {
    let v = view(16, 16);
    let s = path_shape(polygon(&[
        frame(&v, [4.0, 4.0]),
        frame(&v, [12.0, 4.0]),
        frame(&v, [12.0, 12.0]),
        frame(&v, [4.0, 12.0]),
    ]));
    let mut r = path(0, s.key(), true, None);
    if rows.is_some() {
        r.paint.gradient = Some([frame(&v, [4.0, 8.0]), frame(&v, [12.0, 8.0])]);
    }
    r.paint.fill = rows.unwrap_or(Paint::One(rgba));
    decide(&[delta(1, Some(v), vec![s], vec![r])])
}

#[test]
fn at_the_alpha_band_s_edges_only_an_alpha_byte_above_zero_paints() {
    // `alpha_round@1`: the byte is trunc(d · 65535 + 0.5) >> 8; an op whose byte is 0 makes no
    // fragment. Just below, at and just above 255.5 / 65535.
    let edge = 255.5 / 65535.0;
    for d in [edge * (1.0 - 1e-9), edge, edge * (1.0 + 1e-9)] {
        let byte = ((d * 65535.0 + 0.5) as u32) >> 8;
        let dec = square([1.0, 1.0, 1.0, d], None);
        assert_eq!(
            !dec.fragments.is_empty(),
            byte > 0,
            "alpha {d:e}, byte {byte}"
        );
        if byte > 0 {
            assert_eq!(dec.colours[0].kind, COLOUR_SOLID);
            assert_eq!(dec.colours[0].rgba[3], byte as u8);
        }
    }
}

#[test]
fn a_layer_of_two_rows_is_a_gradient_the_oracle_does_not_compute() {
    let dec = square(
        [0.0; 4],
        Some(Paint::Two([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])),
    );
    assert_eq!(dec.colours.len(), 1, "{:?}", dec.refused);
    assert_eq!(dec.colours[0].kind, COLOUR_GRADIENT);
}

#[test]
fn a_flat_mesh_triangle_is_its_fill_premultiplied() {
    let v = view(16, 16);
    let mesh = Shape::new(ShapeData::Mesh {
        positions: vec![
            frame(&v, [2.0, 2.0]),
            frame(&v, [14.0, 2.0]),
            frame(&v, [8.0, 14.0]),
        ],
        triangles: vec![[0, 1, 2]],
        normals: None,
    });
    let mut m = arachne_core::xf::Record::new(arachne_core::xf::RecordKind::Mesh, 0, 100);
    m.geometry.shape_a = Some(mesh.key());
    m.paint.fill = Paint::One([0.2, 0.6, 0.3, 0.5]);
    m.opacity = 0.5;
    let d = decide(&[delta(1, Some(v), vec![mesh], vec![m])]);
    assert_eq!(d.colours[0].kind, COLOUR_SOLID);
    assert_eq!(
        d.colours[0].rgba,
        premultiply_by_text([0.2, 0.6, 0.3, 0.5], 0.5)
    );
}
