//! A path record's geometry and paint made into what is drawn: its shape through its placement, a
//! reveal window cut from it (`VMobject.pointwise_become_partial`), a blend with its second shape
//! (`interpolate`), and a paint blend mixed as CE's style interpolation mixes it. The drawer and
//! the screen boxes both read a record through these, so the shape boxed is the shape drawn.

use crate::xf::*;

/// The points a path record draws, in world space: its shape `a` through its placement, then its
/// reveal window, then its blend with `b` (its second shape, found by the caller), each as CE
/// computes it. `None` when the record blends and `b` is missing or of another length (`apply`
/// refuses both, so a drawn record always has its points).
pub fn path_points(
    geometry: &Geometry,
    a: &[[f64; 3]],
    b: Option<&[[f64; 3]]>,
) -> Option<Vec<[f64; 3]>> {
    let mut points: Vec<[f64; 3]> = a.iter().map(|p| place(&geometry.place_a, *p)).collect();
    if let Some((lo, hi)) = geometry.reveal {
        points = revealed(&points, lo, hi);
    }
    if let Some(blend) = &geometry.blend {
        let b = b?;
        if b.len() != points.len() {
            return None;
        }
        points = points
            .iter()
            .zip(b)
            .map(|(p, q)| interpolate(*p, place(&blend.place_b, *q), blend.t))
            .collect();
    }
    Some(points)
}

/// `bezier.interpolate(p, q, t)`: `(1 - t) * p + t * q`, coordinate by coordinate.
pub fn interpolate(p: [f64; 3], q: [f64; 3], t: f64) -> [f64; 3] {
    std::array::from_fn(|k| (1.0 - t) * p[k] + t * q[k])
}

/// The paint block a record draws with: its own, or, under a paint blend, its own mixed toward
/// `paint_b` by `mix` as CE's style interpolation mixes them (each paint's rows and each width by
/// `interpolate`, `paint_b`'s own where `mix` is 1). An empty paint mixed with a painted one
/// stays empty, as numpy broadcasts no rows against one; one colour mixed with two takes two.
/// `Err` names what it cannot mix: a paint of a table of rows.
pub fn drawn_paint(paint: &PaintBlock) -> Result<PaintBlock, &'static str> {
    let Some(b) = &paint.paint_b else {
        return Ok(paint.clone());
    };
    let mix = paint.mix;
    let width = |a: f64, z: f64| {
        if mix == 1.0 {
            z
        } else {
            (1.0 - mix) * a + mix * z
        }
    };
    Ok(PaintBlock {
        fill: mixed(&paint.fill, &b.fill, mix)?,
        stroke: mixed(&paint.stroke, &b.stroke, mix)?,
        background_stroke: mixed(&paint.background_stroke, &b.background_stroke, mix)?,
        stroke_width: width(paint.stroke_width, b.stroke_width),
        background_stroke_width: width(paint.background_stroke_width, b.background_stroke_width),
        paint_b: None,
        mix: 0.0,
        ..paint.clone()
    })
}

fn mixed(a: &Paint, z: &Paint, mix: f64) -> Result<Paint, &'static str> {
    let rows = |p: &Paint| -> Result<Vec<[f64; 4]>, &'static str> {
        match p {
            Paint::None => Ok(Vec::new()),
            Paint::One(c) => Ok(vec![*c]),
            Paint::Two(cs) => Ok(cs.to_vec()),
            Paint::Table(_) => Err("a paint blend of a table of colours"),
        }
    };
    if mix == 1.0 {
        return Ok(z.clone());
    }
    let (ra, rz) = (rows(a)?, rows(z)?);
    // numpy's broadcasting of the two row arrays: one row stretches to the other's count.
    let n = match (ra.len(), rz.len()) {
        (0, _) | (_, 0) => 0,
        (1, n) | (n, 1) => n,
        (m, n) if m == n => n,
        _ => return Err("a paint blend of row counts numpy does not broadcast"),
    };
    let row = |r: &[[f64; 4]], i: usize| r[if r.len() == 1 { 0 } else { i }];
    let out: Vec<[f64; 4]> = (0..n)
        .map(|i| {
            let (p, q) = (row(&ra, i), row(&rz, i));
            std::array::from_fn(|k| (1.0 - mix) * p[k] + mix * q[k])
        })
        .collect();
    Ok(match out.as_slice() {
        [] => Paint::None,
        [c] => Paint::One(*c),
        [c, d] => Paint::Two([*c, *d]),
        _ => unreachable!("rows of at most two"),
    })
}

/// `VMobject.pointwise_become_partial(path, lo, hi)` of a path holding no points before: the part
/// of the cubic spline `points` between `lo` and `hi`.
pub fn revealed(points: &[[f64; 3]], lo: f64, hi: f64) -> Vec<[f64; 3]> {
    if lo <= 0.0 && hi >= 1.0 {
        return points.to_vec();
    }
    let curves = points.len() / 4;
    if curves == 0 {
        return Vec::new();
    }
    let (lower, lower_residue) = integer_interpolate(curves, lo);
    let (upper, upper_residue) = integer_interpolate(curves, hi);
    let curve = |i: usize| -> [[f64; 3]; 4] { std::array::from_fn(|k| points[4 * i + k]) };
    if lower == upper {
        return partial_cubic(curve(lower), lower_residue, upper_residue).to_vec();
    }
    let mut out = partial_cubic(curve(lower), lower_residue, 1.0).to_vec();
    out.extend_from_slice(&points[4 * (lower + 1)..4 * upper]);
    out.extend(partial_cubic(curve(upper), 0.0, upper_residue));
    out
}

/// `bezier.integer_interpolate(0, end, alpha)`.
fn integer_interpolate(end: usize, alpha: f64) -> (usize, f64) {
    if alpha >= 1.0 {
        return (end - 1, 1.0);
    }
    if alpha <= 0.0 {
        return (0, 0.0);
    }
    let scaled = end as f64 * alpha;
    (scaled as usize, scaled.rem_euclid(1.0))
}

/// The part of one cubic between `a` and `b` (`bezier.partial_bezier_points`), by de Casteljau.
fn partial_cubic(c: [[f64; 3]; 4], a: f64, b: f64) -> [[f64; 3]; 4] {
    let lerp = |p: [f64; 3], q: [f64; 3], t: f64| -> [f64; 3] {
        std::array::from_fn(|k| p[k] + (q[k] - p[k]) * t)
    };
    // The curve from `t` on.
    let tail = |c: [[f64; 3]; 4], t: f64| -> [[f64; 3]; 4] {
        let (ab, bc, cd) = (
            lerp(c[0], c[1], t),
            lerp(c[1], c[2], t),
            lerp(c[2], c[3], t),
        );
        let (abc, bcd) = (lerp(ab, bc, t), lerp(bc, cd, t));
        [lerp(abc, bcd, t), bcd, cd, c[3]]
    };
    // The curve up to `t`.
    let head = |c: [[f64; 3]; 4], t: f64| -> [[f64; 3]; 4] {
        let (ab, bc, cd) = (
            lerp(c[0], c[1], t),
            lerp(c[1], c[2], t),
            lerp(c[2], c[3], t),
        );
        let (abc, bcd) = (lerp(ab, bc, t), lerp(bc, cd, t));
        [c[0], ab, abc, lerp(abc, bcd, t)]
    };
    if a >= 1.0 {
        return [c[3]; 4];
    }
    let from_a = tail(c, a);
    head(from_a, (b - a) / (1.0 - a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
        corners
            .windows(2)
            .flat_map(|w| {
                let (p, q) = (w[0], w[1]);
                let at = |t: f64| std::array::from_fn(|k| p[k] + (q[k] - p[k]) * t);
                [p, at(1.0 / 3.0), at(2.0 / 3.0), q]
            })
            .collect()
    }

    #[test]
    fn a_reveal_cuts_whole_curves_and_parts_as_ce_does() {
        let points = line(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 3.0, 0.0]]);
        let part = revealed(&points, 0.25, 0.75);
        assert_eq!(part.len(), 8);
        let x = |p: [f64; 3]| (p[0] * 1e9).round() / 1e9;
        assert_eq!(x(part[0]), 1.5);
        assert_eq!(part[3], [3.0, 0.0, 0.0]);
        assert!((part[7][1] - 1.5).abs() < 1e-12);
        assert_eq!(revealed(&points, 0.0, 1.0), points);
        assert_eq!(revealed(&[], 0.2, 0.4), Vec::<[f64; 3]>::new());
    }

    #[test]
    fn a_blend_interpolates_point_by_point_after_the_placements() {
        let a = line(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]);
        let b = line(&[[0.0, 2.0, 0.0], [4.0, 2.0, 0.0]]);
        let mut place_b = IDENTITY;
        place_b[0][3] = 1.0; // b moved one unit right
        let geometry = Geometry {
            blend: Some(Blend {
                shape_b: Key([1; 16]),
                place_b,
                t: 0.25,
            }),
            ..Geometry::default()
        };
        let got = path_points(&geometry, &a, Some(&b)).unwrap();
        assert_eq!(got[0], [0.25, 0.5, 0.0]);
        assert_eq!(got[3], [4.25, 0.5, 0.0]);
        assert!(path_points(&geometry, &a, None).is_none());
        assert!(path_points(&geometry, &a, Some(&b[..2])).is_none());
    }

    #[test]
    fn a_paint_blend_mixes_as_ce_s_style_interpolation() {
        let paint = PaintBlock {
            fill: Paint::One([1.0, 0.0, 0.0, 1.0]),
            stroke: Paint::Two([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
            background_stroke: Paint::None,
            stroke_width: 4.0,
            background_stroke_width: 0.0,
            paint_b: Some(PaintB {
                fill: Paint::One([0.0, 0.0, 1.0, 0.5]),
                stroke: Paint::One([1.0, 0.0, 0.0, 1.0]),
                background_stroke: Paint::One([1.0, 1.0, 1.0, 1.0]),
                stroke_width: 2.0,
                background_stroke_width: 6.0,
            }),
            mix: 0.25,
            ..PaintBlock::default()
        };
        let got = drawn_paint(&paint).unwrap();
        assert_eq!(got.fill, Paint::One([0.75, 0.0, 0.25, 0.875]));
        assert_eq!(
            got.stroke,
            Paint::Two([[0.25, 0.0, 0.0, 1.0], [1.0, 0.75, 0.75, 1.0]])
        );
        // No rows broadcast against one: still none.
        assert_eq!(got.background_stroke, Paint::None);
        assert_eq!(got.stroke_width, 3.5);
        assert_eq!(got.background_stroke_width, 1.5);
        assert!(got.paint_b.is_none());
        // At 1 the end's paint is taken whole, as CE takes it.
        let end = drawn_paint(&PaintBlock {
            mix: 1.0,
            ..paint.clone()
        })
        .unwrap();
        assert_eq!(end.background_stroke, Paint::One([1.0, 1.0, 1.0, 1.0]));
        assert_eq!(end.stroke_width, 2.0);
        let table = PaintBlock {
            fill: Paint::Table(Key([2; 16])),
            ..paint
        };
        assert!(drawn_paint(&table).is_err());
    }
}
