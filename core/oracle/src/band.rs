//! The tie band's input terms (contract §6, `band_input@2`): the departure along any normal
//! `Δ⁺`, a record's tolerance, what a move does to a key on a plane (`kr`), the hull of a record's
//! points on its plane and a lower bound of two records' joint width (§7's may-share test).

use arachne_core::xf::IDENTITY;

use crate::depth::{WorldPlane, cross, norm};
use crate::draw::Placing;
use crate::shift::{U, departure_of};

/// `|v|₁`.
fn norm1(v: [f64; 3]) -> f64 {
    v[0].abs() + v[1].abs() + v[2].abs()
}

fn abs_place(m: &arachne_core::xf::Affine, p: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| {
        m[i][0].abs() * p[0] + m[i][1].abs() * p[1] + m[i][2].abs() * p[2] + m[i][3].abs()
    })
}

/// `Δ⁺`: the departure along any normal — `Δ` with each `|Cᵀn|₁` replaced by `√3 ‖C‖_F` and the
/// rounding term by the largest length of `|outer| (|inner| |p|)`, times `1 + 10^-12`, each
/// placing taken with its coefficient.
pub fn departure_plus(g: f64, placings: &[Placing]) -> f64 {
    let one = |placing: &Placing| -> f64 {
        let [outer, inner] = &placing.places;
        let mut total = 0.0;
        if g > 0.0 {
            // C, the linear part carried so far (the identity first).
            let mut c = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            let frob =
                |c: &[[f64; 3]; 3]| libm::sqrt(c.iter().flatten().map(|v| v * v).sum::<f64>());
            for (k, m) in [outer, inner].into_iter().enumerate() {
                if *m == IDENTITY {
                    continue;
                }
                let reach = placing
                    .points
                    .iter()
                    .map(|p| {
                        if k == 0 && *inner != IDENTITY {
                            norm1(arachne_core::xf::place(inner, *p))
                        } else {
                            norm1(*p)
                        }
                    })
                    .fold(0.0, crate::bound::max);
                total += libm::sqrt(3.0) * frob(&c) * (g / 2.0 + reach / 33_554_432.0);
                let l: [[f64; 3]; 3] = std::array::from_fn(|i| [m[i][0], m[i][1], m[i][2]]);
                c = std::array::from_fn(|i| {
                    std::array::from_fn(|j| (0..3).map(|q| c[i][q] * l[q][j]).sum())
                });
            }
            total += (g / 2.0) * libm::sqrt(3.0) * frob(&c);
        }
        let rounding = placing
            .points
            .iter()
            .map(|p| norm(abs_place(outer, abs_place(inner, p.map(f64::abs)))))
            .fold(0.0, crate::bound::max);
        (total + 8.0 * U * rounding) * (1.0 + 1e-12)
    };
    placings.iter().map(|p| p.weight * one(p)).sum()
}

/// A record's tolerance along its plane's unit normal, `τ(n_P) = Δ(n_P) + dep`.
pub fn tolerance(g: f64, plane: &WorldPlane, placings: &[Placing], dep: f64) -> f64 {
    departure_of(g, plane.unrounded_n, placings) + dep
}

/// `kr(P, δ) = δ / (|D_P| − δ)` times `(1 + 2η)`, infinite where `|D_P| − δ <= η (|D_P| + δ)`
/// (§6, the margin).
pub fn kr(big_d: f64, delta: f64) -> f64 {
    crate::bound::quotient(delta, big_d.abs() - delta, big_d.abs() + delta)
}

/// A record's points on its plane, in two coordinates of the plane, and their convex hull.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneHull {
    pub origin: [f64; 3],
    pub u: [f64; 3],
    pub v: [f64; 3],
    pub hull: Vec<[f64; 2]>,
    /// `w_P`, the hull's least width.
    pub width: f64,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

/// The convex hull of 2D points, counter-clockwise (Andrew's monotone chain).
pub fn hull2(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut pts: Vec<[f64; 2]> = points
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let crossz = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f64; 2]> = Vec::new();
    for q in pts.iter() {
        while hull.len() >= 2 && crossz(hull[hull.len() - 2], hull[hull.len() - 1], *q) <= 0.0 {
            hull.pop();
        }
        hull.push(*q);
    }
    let lower = hull.len() + 1;
    for q in pts.iter().rev().skip(1) {
        while hull.len() >= lower && crossz(hull[hull.len() - 2], hull[hull.len() - 1], *q) <= 0.0 {
            hull.pop();
        }
        hull.push(*q);
    }
    hull.pop();
    hull
}

impl PlaneHull {
    pub fn of(plane: &WorldPlane, points: &[[f64; 3]]) -> PlaneHull {
        let n = plane.unrounded_n;
        let pick = if n[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let u = cross(n, pick);
        let ul = norm(u);
        let u = u.map(|c| c / ul);
        let v = cross(n, u);
        let origin = points.first().copied().unwrap_or([0.0; 3]);
        let flat: Vec<[f64; 2]> = points
            .iter()
            .map(|p| {
                let d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
                [dot(d, u), dot(d, v)]
            })
            .collect();
        let hull = hull2(&flat);
        let width = least_width(&hull);
        PlaneHull {
            origin,
            u,
            v,
            hull,
            width,
        }
    }

    /// The distance of a world point from the hull, in the plane (0 inside).
    pub fn distance(&self, x: [f64; 3]) -> f64 {
        let d = [
            x[0] - self.origin[0],
            x[1] - self.origin[1],
            x[2] - self.origin[2],
        ];
        distance_to_polygon(&self.hull, [dot(d, self.u), dot(d, self.v)])
    }
}

/// A convex polygon's least width: the least, over its edges, of the farthest vertex from the
/// edge's line (0 for fewer than three vertices).
pub fn least_width(hull: &[[f64; 2]]) -> f64 {
    if hull.len() < 3 {
        return 0.0;
    }
    let mut best = f64::INFINITY;
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let e = [b[0] - a[0], b[1] - a[1]];
        let l = libm::hypot(e[0], e[1]);
        if l == 0.0 {
            continue;
        }
        let mut far: f64 = 0.0;
        for p in hull {
            let t = ((p[0] - a[0]) * e[1] - (p[1] - a[1]) * e[0]).abs() / l;
            // A term that is not a number states no width: the hull is as narrow as can be.
            if t.is_nan() {
                return 0.0;
            }
            far = far.max(t);
        }
        best = best.min(far);
    }
    if best.is_finite() { best } else { 0.0 }
}

/// The distance from `p` to a convex polygon (counter-clockwise), 0 inside.
pub fn distance_to_polygon(hull: &[[f64; 2]], p: [f64; 2]) -> f64 {
    if hull.is_empty() {
        return f64::INFINITY;
    }
    let crossz = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    if hull.len() >= 3
        && (0..hull.len()).all(|i| crossz(hull[i], hull[(i + 1) % hull.len()], p) >= 0.0)
    {
        return 0.0;
    }
    let seg = |a: [f64; 2], b: [f64; 2]| {
        let ab = [b[0] - a[0], b[1] - a[1]];
        let l2 = ab[0] * ab[0] + ab[1] * ab[1];
        let t = if l2 == 0.0 {
            0.0
        } else {
            (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0)
        };
        libm::hypot(p[0] - (a[0] + ab[0] * t), p[1] - (a[1] + ab[1] * t))
    };
    crate::bound::least(
        (0..hull.len()).map(|i| seg(hull[i], hull[(i + 1) % hull.len()])),
        f64::INFINITY,
    )
}

/// A lower bound of the least width of a set of 3D points: the least width of tetrahedra drawn
/// from it (each tetrahedron's least width is the least of its four vertex heights and its three
/// opposite-edge distances), the largest over the tetrahedra tried. Any lower bound only grows
/// the may-share set (contract §7).
pub fn width_lower_bound(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let all: Vec<[f64; 3]> = a.iter().chain(b).copied().collect();
    if all.len() < 4 {
        return 0.0;
    }
    let base = |set: &[[f64; 3]]| -> Option<[[f64; 3]; 3]> {
        let f = *set.first()?;
        let far = *set
            .iter()
            .max_by(|p, q| dist(**p, f).total_cmp(&dist(**q, f)))?;
        let third = *set
            .iter()
            .max_by(|p, q| line_distance(**p, f, far).total_cmp(&line_distance(**q, f, far)))?;
        Some([f, far, third])
    };
    let mut best: f64 = 0.0;
    for tri in [base(a), base(b), base(&all)].into_iter().flatten() {
        let n = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]));
        let nl = norm(n);
        if nl == 0.0 {
            continue;
        }
        let off = |p: [f64; 3]| dot(sub(p, tri[0]), n).abs() / nl;
        let apex = *all
            .iter()
            .max_by(|p, q| off(**p).total_cmp(&off(**q)))
            .expect("points");
        best = best.max(tetra_width([tri[0], tri[1], tri[2], apex]));
    }
    best
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

fn line_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let l = norm(ab);
    if l == 0.0 {
        return dist(p, a);
    }
    norm(cross(sub(p, a), ab)) / l
}

/// A tetrahedron's least width.
pub fn tetra_width(t: [[f64; 3]; 4]) -> f64 {
    let mut best = f64::INFINITY;
    for i in 0..4 {
        let f: Vec<[f64; 3]> = (0..4).filter(|j| *j != i).map(|j| t[j]).collect();
        let n = cross(sub(f[1], f[0]), sub(f[2], f[0]));
        let nl = norm(n);
        if nl > 0.0 {
            let h = dot(sub(t[i], f[0]), n).abs() / nl;
            if h.is_nan() {
                return 0.0;
            }
            best = best.min(h);
        } else {
            return 0.0;
        }
    }
    for (a, b, c, d) in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)] {
        let n = cross(sub(t[b], t[a]), sub(t[d], t[c]));
        let nl = norm(n);
        if nl > 0.0 {
            let h = dot(sub(t[c], t[a]), n).abs() / nl;
            if h.is_nan() {
                return 0.0;
            }
            best = best.min(h);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A convex hull at the scale where an edge's cross products overflow: the edge from
    /// `(1e200, 0)` to `(1, 1e200)` reads `∞ − ∞` at the far vertices.
    const OVERFLOWING: [[f64; 2]; 4] = [[0.0, 0.0], [1e200, 0.0], [1.0, 1e200], [0.0, 1.0]];

    #[test]
    fn a_key_reach_whose_room_is_within_its_rounding_is_unbounded() {
        // §6, the margin, at `kr`: the depth share's witness with the depth taken into the move.
        // `|D| = 38.6` against `δ = fl(z + 2^-49)`, `z` the depth `fl(31 s + 25 c)` for
        // `s = fl(0.6)`, `c = fl(0.8)`: exactly `|D| − (31 s + 25 c) − 2^-49 = −7u`, not positive
        // (checked in integers at the scale `2^53`), while the computed room `|D| − δ` is `64u`.
        let (s, c, big_d) = (0.6, 0.8, 38.6);
        let tiny = 1.0 / 562_949_953_421_312.0; // 2^-49
        let delta = (31.0 * s + 25.0 * c) + tiny;
        let scale = 9_007_199_254_740_992.0; // 2^53
        let int = |v: f64| {
            let n = v * scale;
            assert_eq!(n.fract(), 0.0, "{v} is not a multiple of 2^-53");
            n as i128
        };
        let exact = int(big_d) - (31 * int(s) + 25 * int(c)) - int(tiny);
        assert!(
            exact <= 0,
            "premise: the exact room {exact} u is not positive"
        );
        assert!(
            big_d - delta > 0.0,
            "premise: the computed room is positive"
        );
        let reach = kr(big_d, delta);
        assert!(reach.is_infinite(), "kr {reach} over a room of {exact} u");
    }

    #[test]
    fn a_hull_width_with_a_nan_term_is_zero() {
        // The P-NaN invariant: a NaN term of the least width states no width (0, so `θ` is
        // infinite and the read outside the hull states no bound), not the width of the terms
        // that stayed finite.
        assert_eq!(least_width(&OVERFLOWING), 0.0);
    }

    #[test]
    fn a_polygon_distance_with_a_nan_term_is_unbounded() {
        // The P-NaN invariant: a NaN segment distance states no distance (`∞`), not the least of
        // the segments that stayed finite.
        assert_eq!(
            distance_to_polygon(&OVERFLOWING, [-1.0, -1.0]),
            f64::INFINITY
        );
    }
}
