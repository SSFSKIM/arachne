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
                .fold(0.0, f64::max);
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
        .fold(0.0, f64::max);
    total + 8.0 * U * rounding
}

/// The largest departure over a record's placings.
pub fn departure_of(g: f64, n: [f64; 3], placings: &[Placing]) -> f64 {
    placings
        .iter()
        .map(|p| departure(g, n, p))
        .fold(0.0, f64::max)
}

/// The depth share `ds(δ)` along the sight axis for points at rotated depths `zs`: the largest
/// `δ / (d − z − δ)`, infinite where a denominator is not positive; `δ / (d − δ)` under the
/// exponential option.
pub fn depth_share(p: &Projection, delta: f64, zs: impl Iterator<Item = f64>) -> f64 {
    let share = |den: f64| {
        if den > 0.0 {
            delta / den
        } else {
            f64::INFINITY
        }
    };
    if p.exponential {
        return share(p.d - delta);
    }
    zs.map(|z| share(p.d - z - delta)).fold(0.0, f64::max)
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
        shown.iter().map(|q| self.at(*q)).fold(0.0, f64::max)
    }
}
