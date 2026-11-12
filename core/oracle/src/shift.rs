//! How far the input grid may move a record on screen: the snap's departure `Δ` and the screen
//! shift `ε` of contract §6, which §3's mesh margin and §4's presence band read.

use arachne_core::xf::{Affine, IDENTITY, View};

use crate::camera::{FrameMap, Placement, Projection};
use crate::draw::{Common, Placing};

/// The unit roundoff of binary64.
pub const U: f64 = f64::EPSILON / 2.0;

/// How far a placement's snapped linear entry may stand from its value: half of `2^-24`.
const LINEAR_SNAP: f64 = 1.0 / 33_554_432.0;

fn abs_place(m: &Affine, p: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| {
        m[i][0].abs() * p[0] + m[i][1].abs() * p[1] + m[i][2].abs() * p[2] + m[i][3].abs()
    })
}

fn norm1(v: [f64; 3]) -> f64 {
    v[0].abs() + v[1].abs() + v[2].abs()
}

/// `Δ(n)`: how far a placing's points may depart along the unit normal `n` under the grid `g`.
///
/// Through each placement that is not the identity, outer first, `|Cᵀn|₁ (g/2 + 2^-25 X_M)` for
/// `X_M` the largest `|x|₁` over the points it places, `Cᵀn` then carried through its linear
/// part; then `(g/2) |Cᵀn|₁` for the shape's points (none of these when `g = 0`); and in every case
/// `8u max_p |n| · (|outer| (|inner| |p|))` for the placing's own rounding.
pub fn departure(g: f64, n: [f64; 3], placing: &Placing) -> f64 {
    let [outer, inner] = &placing.places;
    let mut total = 0.0;
    if g > 0.0 {
        let mut v = n;
        for (k, m) in [outer, inner].into_iter().enumerate() {
            if *m == IDENTITY {
                continue;
            }
            // The points `m` places: the inner-placed points for the outer placement, the shape's
            // own for the inner.
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
            total += norm1(v) * (g / 2.0 + reach * LINEAR_SNAP);
            v = std::array::from_fn(|j| m[0][j] * v[0] + m[1][j] * v[1] + m[2][j] * v[2]);
        }
        total += (g / 2.0) * norm1(v);
    }
    let rounding = placing
        .points
        .iter()
        .map(|p| {
            let r = abs_place(outer, abs_place(inner, p.map(f64::abs)));
            n[0].abs() * r[0] + n[1].abs() * r[1] + n[2].abs() * r[2]
        })
        .fold(0.0, crate::bound::max);
    total + 8.0 * U * rounding
}

/// A record's departure: its placings' departures, each times the coefficient the drawn points
/// take it with. A blend's point `(1 − t) p + t q` moves by `|1 − t| Δ_p + |t| Δ_q` whatever `t`
/// is, `t = 2` included (`−p + 2q`, three departures). The sum also holds the blend's own
/// rounding, within `4u (|1 − t| |p| + |t| |q|)`: each placing's rounding term is `8u` of its
/// points, taken with the same coefficients.
pub fn departure_of(g: f64, n: [f64; 3], placings: &[Placing]) -> f64 {
    placings.iter().map(|p| p.weight * departure(g, n, p)).sum()
}

/// The depth share `ds(δ)` along the sight axis for points at rotated depths `zs`: the largest
/// `δ / (d − z − δ)` times `(1 + 2η)`, infinite where a denominator is within `η (|d| + |z| + δ)`
/// of zero; `δ / (d − δ)` under the exponential option, at the scale `|d| + δ` (§6, the margin).
pub fn depth_share(p: &Projection, delta: f64, zs: impl Iterator<Item = f64>) -> f64 {
    if p.exponential {
        return crate::bound::quotient(delta, p.d - delta, p.d.abs() + delta);
    }
    zs.map(|z| crate::bound::quotient(delta, p.d - z - delta, p.d.abs() + z.abs() + delta))
        .fold(0.0, crate::bound::max)
}

/// A record's screen shift, read per point: `ε(q) = hypot(|xx| x₀, |yy| y₀) (1 + 64u)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Shift {
    /// `a_0`, `a_1`: the departures along the camera's first two axes (the points' own axes for
    /// a record not projected).
    pub a: [f64; 2],
    /// The factor share `σ`.
    pub sigma: f64,
    /// For a projected record, the zoom and projection the factor is read through.
    pub projected: Option<Projection>,
    /// For a fixed-orientation record, how far its centre's snap may move its offset, per axis.
    pub carried: [f64; 2],
    pub xx: f64,
    pub yy: f64,
}

impl Shift {
    /// The shift of a drawn thing whose shown points are `shown` (their third coordinate the
    /// rotated depth where projected).
    pub fn of(common: &Common, view: &View, map: &FrameMap, shown: &[[f64; 3]]) -> Shift {
        let g = view.input_grid;
        let axes = |p: Option<&Projection>| p.map_or(crate::camera::IDENTITY3, |p| p.r);
        let projected = matches!(common.placement, Placement::Projected)
            .then(|| common.projection.clone())
            .flatten();
        let rows = axes(projected.as_ref());
        let along = |k: usize| departure_of(g, rows[k], &common.placings);
        let sigma = match &projected {
            Some(p) => depth_share(p, along(2), shown.iter().map(|q| q[2])),
            None => 0.0,
        };
        // A fixed-orientation record moves with its centre's offset `project(C) − C`; the centre,
        // carried snapped, moves by up to `g/2` per coordinate.
        let carried = match (&common.placement, &common.projection) {
            (Placement::FixedOrientation { centre, .. }, Some(p)) => {
                let q = p.project(*centre);
                let dc = |k: usize| (g / 2.0) * norm1(p.r[k]);
                let s = depth_share(p, dc(2), std::iter::once(q[2]));
                let f = (p.factor(q[2]) * p.zoom).abs();
                std::array::from_fn(|k| f * (1.0 + s) * dc(k) + q[k].abs() * s + g / 2.0)
            }
            _ => [0.0; 2],
        };
        Shift {
            a: [along(0), along(1)],
            sigma,
            projected,
            carried,
            xx: map.xx,
            yy: map.yy,
        }
    }

    /// `ε(q)` in pixels for a shown point `q`; infinite where the share is.
    pub fn at(&self, q: [f64; 3]) -> f64 {
        if !self.sigma.is_finite() {
            return f64::INFINITY;
        }
        let x: [f64; 2] = match &self.projected {
            Some(p) => {
                let f = (p.factor(q[2]) * p.zoom).abs();
                std::array::from_fn(|k| {
                    f * (1.0 + self.sigma) * self.a[k] + q[k].abs() * self.sigma
                })
            }
            None => std::array::from_fn(|k| self.a[k] + self.carried[k]),
        };
        libm::hypot(self.xx.abs() * x[0], self.yy.abs() * x[1]) * (1.0 + 64.0 * U)
    }

    /// The record's `ε`: the largest over its points.
    pub fn of_points(&self, shown: &[[f64; 3]]) -> f64 {
        shown
            .iter()
            .map(|q| self.at(*q))
            .fold(0.0, crate::bound::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_depth_share_whose_denominator_is_within_its_rounding_is_unbounded() {
        // §6: `ds(δ)` is infinite where a denominator is not positive, and that is the exact
        // `d − z − δ` of the record's point, not the one its depth rounds to; the margin `η`
        // reads a denominator within `η (|d| + |z| + δ)` of zero as not positive. A rotation about x
        // with third row `(0, s, c)`, `s = fl(0.6)`, `c = fl(0.8)`, no frame centre, `d = 38.6`;
        // the world point `(0, 31, 25)` has the exact depth `z* = 31 s + 25 c`, which rounds 55u
        // low, and `δ = 2^-49`. Exactly `d − z* − δ = −7u`, not positive (checked below in
        // integers at the scale `2^53`, where `s`, `c`, `d` and `δ` are all integers); the
        // computed one is `(d − z) − δ = 48u`, positive, so today's share is `1/3`.
        let (s, c) = (0.6, 0.8);
        let p = Projection {
            r: [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            c: [0.0; 3],
            d: 38.6,
            zoom: 1.0,
            exponential: false,
        };
        let delta = 1.0 / 562_949_953_421_312.0; // 2^-49
        let shown = [p.project([0.0, 31.0, 25.0])];
        let scale = 9_007_199_254_740_992.0; // 2^53
        let int = |v: f64| {
            let n = v * scale;
            assert_eq!(n.fract(), 0.0, "{v} is not a multiple of 2^-53");
            n as i128
        };
        let exact = int(p.d) - (31 * int(s) + 25 * int(c)) - int(delta);
        assert!(
            exact <= 0,
            "premise: the exact denominator {exact} u is not positive"
        );
        let computed = (p.d - shown[0][2]) - delta;
        assert!(
            computed > 0.0,
            "premise: the computed denominator {computed:e} is positive"
        );
        let share = depth_share(&p, delta, shown.iter().map(|q| q[2]));
        assert!(
            share.is_infinite(),
            "ds {share} over a denominator of {exact} u"
        );
    }

    #[test]
    fn a_depth_share_of_a_shift_rounded_low_still_bounds_the_exact_share() {
        // §6, the margin: an upper-bound quotient is taken times `(1 + 2η)`, which covers the
        // roundings of its inputs. At `d = 2^20` and a depth of 0, a shift `δ* = 1` computed
        // `8u` low as `δ = 1 − 8u` gives the denominator `fl(d − δ) = 2^20 − 1` and the quotient
        // `fl(δ / (2^20 − 1))`, below the exact share `1 / (2^20 − 1)`; the share taken holds it,
        // checked in integers as `share · (2^20 − 1) >= 1`.
        let p = Projection {
            r: crate::camera::IDENTITY3,
            c: [0.0; 3],
            d: 1_048_576.0, // 2^20
            zoom: 1.0,
            exponential: false,
        };
        let delta = 1.0 - 8.0 * U;
        let share = depth_share(&p, delta, std::iter::once(0.0));
        // `share = m · 2^e` exactly, with `m < 2^53` and `e < 0`.
        let bits = share.to_bits();
        let (m, e) = (
            (bits & ((1 << 52) - 1)) | (1 << 52),
            ((bits >> 52) & 0x7ff) as i32 - 1075,
        );
        assert!(share.is_finite() && share > 0.0 && e < 0, "{share}");
        assert!(
            (m as i128) * 1_048_575 >= 1i128 << (-e),
            "the share {share:e} is below 1/1048575"
        );
    }
}
