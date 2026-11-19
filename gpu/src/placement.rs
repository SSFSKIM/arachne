//! Where a quad's texture lands: the map from a pixel to the texel it shows.

/// What four corners make of a quad.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Placement {
    /// The rows of the map from a pixel position (x, y, 1) to texel coordinates
    /// (u * width, v * height, w), scaled so that w > 0 inside the quad.
    Mapped([[f32; 4]; 3]),
    /// Corners on one line or at one point: the quad covers no pixel, as CE draws such an image.
    Empty,
    /// An outline no texture can be placed in: crossed, concave, or with three corners on one
    /// line. The frame is refused; this is why.
    Refused(&'static str),
}

/// The placement of a `width` x `height` texture at `corners`, in CE's order: upper left, upper
/// right, lower left, lower right, in pixels (x right, y down).
///
/// Only a strictly convex outline is placed. A projective map exists for any four corners no three
/// of which are on one line, but for a crossed or concave outline it sends part of the plane
/// through infinity inside the texture's square, and what it draws is not the outline.
pub(crate) fn place(corners: &[[f64; 2]; 4], width: u32, height: u32) -> Placement {
    let [ul, ur, ll, lr] = *corners;
    // The outline in order around the quad: upper left, upper right, lower right, lower left.
    let outline = [ul, ur, lr, ll];
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    // Every corner on the line through the two farthest apart, or all at one point.
    let (a, b) = (0..4)
        .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
        .max_by(|&(i, j), &(k, l)| {
            let d = |p: [f64; 2], q: [f64; 2]| {
                let (dx, dy) = (p[0] - q[0], p[1] - q[1]);
                dx * dx + dy * dy
            };
            d(outline[i], outline[j]).total_cmp(&d(outline[k], outline[l]))
        })
        .map(|(i, j)| (outline[i], outline[j]))
        .expect("four corners");
    if outline.iter().all(|&p| cross(a, b, p) == 0.0) {
        return Placement::Empty;
    }
    // The turn at each corner, going round: all of one sign for a convex outline.
    let turns: [f64; 4] =
        std::array::from_fn(|i| cross(outline[i], outline[(i + 1) % 4], outline[(i + 2) % 4]));
    if turns.contains(&0.0) {
        return Placement::Refused("a quad with three corners on one line");
    }
    if !(turns.iter().all(|&t| t > 0.0) || turns.iter().all(|&t| t < 0.0)) {
        return Placement::Refused("a quad whose outline is crossed or concave");
    }
    match texel_mapping(corners, width, height) {
        Some(rows) => Placement::Mapped(rows),
        None => Placement::Refused("a quad whose placement cannot be computed"),
    }
}

/// The projective map of the unit square onto the quad, inverted and scaled to texels; `None` when
/// the map does not exist or cannot be inverted.
fn texel_mapping(corners: &[[f64; 2]; 4], width: u32, height: u32) -> Option<[[f32; 4]; 3]> {
    let [ul, ur, ll, lr] = *corners;
    // (0,0) upper left, (1,0) upper right, (1,1) lower right, (0,1) lower left (Heckbert's
    // square-to-quad).
    let (x0, y0, x1, y1, x2, y2, x3, y3) = (ul[0], ul[1], ur[0], ur[1], lr[0], lr[1], ll[0], ll[1]);
    let sx = x0 - x1 + x2 - x3;
    let sy = y0 - y1 + y2 - y3;
    let (dx1, dx2, dy1, dy2) = (x1 - x2, x3 - x2, y1 - y2, y3 - y2);
    let den = dx1 * dy2 - dx2 * dy1;
    if den == 0.0 {
        return None;
    }
    let g = (sx * dy2 - dx2 * sy) / den;
    let h = (dx1 * sy - sx * dy1) / den;
    let forward = [
        [x1 - x0 + g * x1, x3 - x0 + h * x3, x0],
        [y1 - y0 + g * y1, y3 - y0 + h * y3, y0],
        [g, h, 1.0],
    ];
    let inverse = invert(&forward)?;
    let scale = [f64::from(width), f64::from(height), 1.0];
    let mut rows = [[0.0f64; 3]; 3];
    for (r, row) in rows.iter_mut().enumerate() {
        for (c, value) in row.iter_mut().enumerate() {
            *value = scale[r] * inverse[r][c];
        }
    }
    // Inside the quad w keeps one sign; make it positive there, at the corners' centre.
    let cx = (x0 + x1 + x2 + x3) / 4.0;
    let cy = (y0 + y1 + y2 + y3) / 4.0;
    let w = rows[2][0] * cx + rows[2][1] * cy + rows[2][2];
    if w == 0.0 || !w.is_finite() {
        return None;
    }
    let sign = w.signum();
    Some(rows.map(|[a, b, c]| [(sign * a) as f32, (sign * b) as f32, (sign * c) as f32, 0.0]))
}

fn invert(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let cofactor =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let adjugate = [
        [
            cofactor(1, 2, 1, 2),
            -cofactor(0, 2, 1, 2),
            cofactor(0, 1, 1, 2),
        ],
        [
            -cofactor(1, 2, 0, 2),
            cofactor(0, 2, 0, 2),
            -cofactor(0, 1, 0, 2),
        ],
        [
            cofactor(1, 2, 0, 1),
            -cofactor(0, 2, 0, 1),
            cofactor(0, 1, 0, 1),
        ],
    ];
    let det = m[0][0] * adjugate[0][0] + m[0][1] * adjugate[1][0] + m[0][2] * adjugate[2][0];
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let inverse = adjugate.map(|row| row.map(|v| v / det));
    inverse
        .iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placed(corners: [[f64; 2]; 4]) -> Placement {
        place(&corners, 16, 16)
    }

    #[test]
    fn a_convex_outline_is_placed() {
        let rectangle = [[10.0, 10.0], [50.0, 10.0], [10.0, 40.0], [50.0, 40.0]];
        let trapezoid = [[300.0, 40.0], [380.0, 40.0], [270.0, 140.0], [410.0, 140.0]];
        // The same rectangle mirrored, its outline running the other way round.
        let mirrored = [[50.0, 10.0], [10.0, 10.0], [50.0, 40.0], [10.0, 40.0]];
        for corners in [rectangle, trapezoid, mirrored] {
            assert!(
                matches!(placed(corners), Placement::Mapped(_)),
                "{corners:?}"
            );
        }
    }

    #[test]
    fn a_crossed_outline_is_refused() {
        // Upper right and lower right swapped: a bow-tie, whose signed area is zero though no
        // corner is on another's line.
        let bow_tie = [
            [100.0, 100.0],
            [200.0, 200.0],
            [100.0, 200.0],
            [200.0, 100.0],
        ];
        assert_eq!(
            placed(bow_tie),
            Placement::Refused("a quad whose outline is crossed or concave")
        );
    }

    #[test]
    fn a_concave_outline_is_refused() {
        // The lower right corner pulled inside: the projective map's denominator,
        // 1 - 1.5 u - 1.5 v, vanishes inside the texture's square.
        let dart = [
            [100.0, 100.0],
            [200.0, 100.0],
            [100.0, 200.0],
            [125.0, 125.0],
        ];
        assert_eq!(
            placed(dart),
            Placement::Refused("a quad whose outline is crossed or concave")
        );
    }

    #[test]
    fn three_corners_on_one_line_are_refused() {
        let collinear = [[10.0, 10.0], [50.0, 10.0], [10.0, 60.0], [90.0, 10.0]];
        assert_eq!(
            placed(collinear),
            Placement::Refused("a quad with three corners on one line")
        );
    }

    #[test]
    fn corners_on_one_line_or_at_one_point_cover_nothing() {
        let line = [[10.0, 10.0], [50.0, 10.0], [20.0, 10.0], [60.0, 10.0]];
        let diagonal = [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0], [2.0, 2.0]];
        let point = [[7.0, 7.0]; 4];
        for corners in [line, diagonal, point] {
            assert_eq!(placed(corners), Placement::Empty, "{corners:?}");
        }
    }
}
