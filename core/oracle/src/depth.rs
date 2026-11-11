//! Depth keys from XF (contract §5): world planes, their canonical form and their key across the
//! pixel plane; the plane through three keyed corners; a projected point's key and its error;
//! how a path lies (its planarity). The band's terms are `band.rs`'s; curves are `centreline.rs`'s.

use crate::camera::{FrameMap, Projection};
use crate::shift::U;

/// `PLANE_GRID = 2^-20`: a canonical plane's coefficients are rounded to its multiples.
pub const PLANE_GRID: f64 = 1.0 / 1_048_576.0;

/// `round(v)`, half away from zero.
pub fn round_half_away(v: f64) -> f64 {
    libm::round(v)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

fn abs_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0].abs() * b[0].abs() + a[1].abs() * b[1].abs()) + a[2].abs() * b[2].abs()
}

pub fn norm(v: [f64; 3]) -> f64 {
    libm::sqrt(dot(v, v))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A world plane `n · p = e`, canonical: unit normal turned so its largest component (lower axis
/// on equal magnitudes) is positive, its offset through a point with its error bound, and both
/// rounded to `PLANE_GRID`; the unrounded plane is kept beside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPlane {
    pub n: [f64; 3],
    pub e: f64,
    pub unrounded_n: [f64; 3],
    pub unrounded_e: f64,
    /// The offset's error bound, `4u (|unit| · |p|)` (0 for an axis plane).
    pub e_err: f64,
    /// A `Nearest` face's `δe`, how far two conforming computations' offsets can differ (its
    /// centroid's order, contract §6); `None` for every other plane, whose offset is one value.
    pub offset_spread: Option<f64>,
    /// For an axis plane, its axis.
    pub axis: Option<usize>,
}

fn grid(v: f64) -> f64 {
    round_half_away(v / PLANE_GRID) * PLANE_GRID
}

impl WorldPlane {
    /// The canonical plane of normal `n` through `p`; `None` where `n` has no length.
    pub fn canonical(n: [f64; 3], p: [f64; 3]) -> Option<WorldPlane> {
        let len = norm(n);
        if !(len > 0.0 && len.is_finite()) {
            return None;
        }
        let mut unit = n.map(|v| v / len);
        let mut big = 0;
        for k in 1..3 {
            if unit[k].abs() > unit[big].abs() {
                big = k;
            }
        }
        if unit[big] < 0.0 {
            unit = unit.map(|v| -v);
        }
        let e = dot(unit, p);
        Some(WorldPlane {
            n: unit.map(grid),
            e: grid(e),
            unrounded_n: unit,
            unrounded_e: e,
            e_err: 4.0 * U * abs_dot(unit, p),
            offset_spread: None,
            axis: None,
        })
    }

    /// The axis plane of coordinate `k` through `p`.
    pub fn axis(k: usize, p: [f64; 3]) -> WorldPlane {
        let mut n = [0.0; 3];
        n[k] = 1.0;
        WorldPlane {
            n,
            e: grid(p[k]),
            unrounded_n: n,
            unrounded_e: p[k],
            e_err: 0.0,
            offset_spread: None,
            axis: Some(k),
        }
    }

    /// The rounded planes a conforming computation can produce: the oracle's, and for each
    /// coefficient within `2^-30` steps of a grid half-step, the other neighbour too. A `Nearest`
    /// face's offset takes `2^-30 + δe / 2^-20` steps, as the other computation's offset may lie
    /// up to `δe` from this one's; its normal is the same bits in both.
    pub fn conforming(&self) -> Vec<([f64; 3], f64)> {
        let window = 1.0 / 1_073_741_824.0;
        let e_window = window + self.offset_spread.unwrap_or(0.0) / PLANE_GRID;
        let options = |v: f64, window: f64| -> Vec<f64> {
            let q = v / PLANE_GRID;
            let mine = grid(v);
            let frac = q - libm::floor(q);
            if (frac - 0.5).abs() <= window {
                let other = if mine / PLANE_GRID == libm::floor(q) {
                    (libm::floor(q) + 1.0) * PLANE_GRID
                } else {
                    libm::floor(q) * PLANE_GRID
                };
                vec![mine, other]
            } else {
                vec![mine]
            }
        };
        if self.axis.is_some() {
            return options(self.unrounded_e, e_window)
                .into_iter()
                .map(|e| (self.n, e))
                .collect();
        }
        let mut out = vec![];
        for a in options(self.unrounded_n[0], window) {
            for b in options(self.unrounded_n[1], window) {
                for c in options(self.unrounded_n[2], window) {
                    for e in options(self.unrounded_e, e_window) {
                        out.push(([a, b, c], e));
                    }
                }
            }
        }
        out
    }
}

/// A key across the pixel plane, `a + b x + c y`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenKey {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl ScreenKey {
    pub fn constant(k: f64) -> ScreenKey {
        ScreenKey {
            a: k,
            b: 0.0,
            c: 0.0,
        }
    }

    /// The plane read at `(x, y)`: `a + b x + c y + 0.0`.
    pub fn at(&self, x: f64, y: f64) -> f64 {
        ((self.a + self.b * x) + self.c * y) + 0.0
    }

    /// The reading's rounding at a point, `4u (|a| + |b x| + |c y|)`.
    pub fn rho_at(&self, p: [f64; 2]) -> f64 {
        4.0 * U * (self.a.abs() + (self.b * p[0]).abs() + (self.c * p[1]).abs())
    }

    /// `ρ` at pixel `(px, py)`: the largest of the reading's rounding at its far corner and at
    /// every point the read takes (a fringe read's may lie outside the pixel).
    pub fn rho(&self, px: i64, py: i64, reads: &[[f64; 2]]) -> f64 {
        let far = self.rho_at([(px + 1) as f64, (py + 1) as f64]);
        reads.iter().map(|p| self.rho_at(*p)).fold(far, f64::max)
    }

    pub fn sloped(&self) -> bool {
        self.b != 0.0 || self.c != 0.0
    }
}

/// The pieces of a world plane's key on screen, from one rounded or unrounded plane.
#[derive(Clone, Copy, Debug)]
pub struct PlaneTerms {
    pub m: [f64; 3],
    pub big_d: f64,
    pub gx: f64,
    pub gy: f64,
}

fn plane_terms(pr: &Projection, map: &FrameMap, n: [f64; 3], e: f64) -> PlaneTerms {
    let r = &pr.r;
    let m = [dot(r[0], n), dot(r[1], n), dot(r[2], n)];
    let big_d = (e - dot(n, pr.c)) - m[2] * pr.d;
    PlaneTerms {
        m,
        big_d,
        gx: m[0] / (pr.zoom * map.xx),
        gy: m[1] / (pr.zoom * map.yy),
    }
}

fn key_of_terms(t: &PlaneTerms, pr: &Projection, map: &FrameMap) -> ScreenKey {
    let d = t.big_d;
    ScreenKey {
        a: -(((-t.gx * map.x0) - t.gy * map.y0) - t.m[2] * pr.d) / d,
        b: -t.gx / d,
        c: -t.gy / d,
    }
}

/// `key_plane`: a world plane's key across the pixel plane, or `None` where it does not form
/// (the exponential option, zoom 0, `D` 0 or within the offset's bound of 0, too near edge-on
/// against `min_cos` at the rotated point `q`, or a coefficient not finite).
pub fn key_plane(
    pr: &Projection,
    map: &FrameMap,
    p: &WorldPlane,
    q: [f64; 3],
    min_cos: f64,
) -> Option<ScreenKey> {
    if pr.exponential || pr.zoom == 0.0 {
        return None;
    }
    let t = plane_terms(pr, map, p.n, p.e);
    if t.big_d == 0.0 {
        return None;
    }
    let u = plane_terms(pr, map, p.unrounded_n, p.unrounded_e);
    if u.big_d.abs() <= p.e_err {
        return None;
    }
    let eye = [q[0], q[1], q[2] - pr.d];
    let cos = t.big_d.abs() / (norm(t.m) * norm(eye));
    if cos.is_nan() || cos < min_cos {
        return None;
    }
    let k = key_of_terms(&t, pr, map);
    (k.a.is_finite() && k.b.is_finite() && k.c.is_finite()).then_some(k)
}

/// The keys of the rounded planes a conforming computation can produce (`WorldPlane::conforming`),
/// the oracle's own first.
pub fn conforming_keys(pr: &Projection, map: &FrameMap, p: &WorldPlane) -> Vec<ScreenKey> {
    p.conforming()
        .into_iter()
        .map(|(n, e)| key_of_terms(&plane_terms(pr, map, n, e), pr, map))
        .collect()
}

/// `M` for a world plane keyed on screen (contract §6, `band_arith@3`): the largest over the
/// region's corners of `gap + 3A + C + shift`, all of the unrounded plane, with `A`, `C` and the
/// shift taken at `Dn`, the nearest the other computation's `D` can stand, and at `|e| + δe`.
/// Infinite where `Dn <= ε`. A read adds `ρ` at its pixel, a `Nearest` face's departure `O`, and
/// `ν` of its key, under `(1 + 64u)`.
pub fn plane_r_region(pr: &Projection, map: &FrameMap, p: &WorldPlane, region: [f64; 4]) -> f64 {
    let u = plane_terms(pr, map, p.unrounded_n, p.unrounded_e);
    let ku = key_of_terms(&u, pr, map);
    let rounded = conforming_keys(pr, map, p);
    let c = pr.c;
    let spread = p.offset_spread.unwrap_or(0.0);
    let m_len = norm(u.m);
    let dn = 8.0 * U * m_len;
    let de = 4.0 * U * ((p.unrounded_e.abs() + spread) + m_len * norm(c));
    let eye = [
        c[0] + pr.d * pr.r[2][0],
        c[1] + pr.d * pr.r[2][1],
        c[2] + pr.d * pr.r[2][2],
    ];
    let m2d = u.m[2] * pr.d;
    let dd = (de + dn * norm(eye)) + 4.0 * U * m2d.abs();
    let near = match p.offset_spread {
        Some(spread) => (u.big_d.abs() - spread) - 2.0 * dd,
        None => u.big_d.abs(),
    };
    let eps = p.e_err + 4.0 * U * spread;
    // A NaN `Dn` bounds nothing either.
    if near.is_nan() || near <= eps {
        return f64::INFINITY;
    }
    let mut worst: f64 = 0.0;
    for (x, y) in [
        (region[0], region[1]),
        (region[2], region[1]),
        (region[0], region[3]),
        (region[2], region[3]),
    ] {
        let terms = |k: &ScreenKey| k.a.abs() + (k.b * x).abs() + (k.c * y).abs();
        let gap = rounded
            .iter()
            .map(|kr| (kr.at(x, y) - ku.at(x, y)).abs() + 4.0 * U * (terms(kr) + terms(&ku)))
            .fold(0.0, f64::max);
        let s = [
            (x - map.x0) / (pr.zoom * map.xx),
            (y - map.y0) / (pr.zoom * map.yy),
            pr.d,
        ];
        let tx = u.gx * (x - map.x0);
        let ty = u.gy * (y - map.y0);
        let f = ((m2d - tx) - ty).abs() / near;
        let dnum = dn * norm(s) + 4.0 * U * (tx.abs() + ty.abs() + m2d.abs());
        let a_term = (dnum + f * dd) / near + 4.0 * U * f;
        let c_term = (f + dnum / near) * spread / near;
        let shift = f * eps / (near - eps);
        worst = worst.max(((gap + 3.0 * a_term) + c_term) + shift);
    }
    worst
}

/// `ν = 2^-44 (|o| + 1)`: last-bit differences between conforming evaluations.
pub fn nu(o: f64) -> f64 {
    (o.abs() + 1.0) / 17_592_186_044_416.0
}

/// A projected point's key error, `key_error = u (2 |f| + 4 f² / |d| r)` with
/// `r = |z| + hypot(x, y) / |f z0| + |c|`, for the point shown at `(x, y, z)`.
pub fn key_error(pr: &Projection, shown: [f64; 3]) -> f64 {
    let f = pr.factor(shown[2]);
    let r = shown[2].abs() + libm::hypot(shown[0], shown[1]) / (f * pr.zoom).abs() + norm(pr.c);
    U * (2.0 * f.abs() + 4.0 * f * f / pr.d.abs() * r)
}

/// The plane on screen through three keyed corners (`(x, y)` in pixels, `key`, its bound), with
/// its own bound before `ρ` (contract §6, "A plane through three keyed corners").
pub fn three_corner_plane(c: [[f64; 2]; 3], d: [f64; 3], err: [f64; 3]) -> (ScreenKey, f64) {
    let inb = err[0].max(err[1]).max(err[2]);
    if d[0] == d[1] && d[1] == d[2] {
        return (ScreenKey::constant(d[0]), inb);
    }
    if !(d[0].is_finite() && d[1].is_finite() && d[2].is_finite()) {
        return (ScreenKey::constant(d[0].min(d[1]).min(d[2])), 0.0);
    }
    let e1 = [c[1][0] - c[0][0], c[1][1] - c[0][1]];
    let e2 = [c[2][0] - c[0][0], c[2][1] - c[0][1]];
    let det = e1[0] * e2[1] - e2[0] * e1[1];
    if det == 0.0 || !det.is_finite() {
        let mean = ((d[0] + d[1]) + d[2]) / 3.0;
        let spread = d.iter().map(|v| (v - mean).abs()).fold(0.0, f64::max);
        return (
            ScreenKey::constant(mean),
            spread + inb + 4.0 * U * mean.abs(),
        );
    }
    let g1 = d[1] - d[0];
    let g2 = d[2] - d[0];
    let b = (g1 * e2[1] - g2 * e1[1]) / det;
    let cc = (e1[0] * g2 - e2[0] * g1) / det;
    let a = d[0] - b * c[0][0] - cc * c[0][1];
    let d_det = 4.0 * U * ((e1[0] * e2[1]).abs() + (e2[0] * e1[1]).abs());
    let d_b = (4.0 * U * ((g1 * e2[1]).abs() + (g2 * e1[1]).abs()) + b.abs() * d_det) / det.abs()
        + U * b.abs();
    let d_c = (4.0 * U * ((e1[0] * g2).abs() + (e2[0] * g1).abs()) + cc.abs() * d_det) / det.abs()
        + U * cc.abs();
    let bound = inb
        + (d_b * e1[0].abs() + d_c * e1[1].abs()).max(d_b * e2[0].abs() + d_c * e2[1].abs())
        + 3.0 * U * (d[0].abs() + (b * c[0][0]).abs() + (cc * c[0][1]).abs());
    (ScreenKey { a, b, c: cc }, bound)
}

/// How a path lies (contract §5): its XF 1.2 code, or decided from its points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Planarity {
    Curve,
    Newell,
    Widest,
    Nearest,
    Axis(usize),
}

impl Planarity {
    pub fn of_code(code: u8) -> Option<Planarity> {
        Some(match code {
            0 => Planarity::Curve,
            1 => Planarity::Newell,
            2 => Planarity::Widest,
            3 => Planarity::Nearest,
            4 => Planarity::Axis(0),
            5 => Planarity::Axis(1),
            6 => Planarity::Axis(2),
            _ => return None,
        })
    }
}

/// The points' widest extent along an axis.
pub fn extent(points: &[[f64; 3]]) -> f64 {
    (0..3)
        .map(|k| {
            let lo = points.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min);
            let hi = points
                .iter()
                .map(|p| p[k])
                .fold(f64::NEG_INFINITY, f64::max);
            hi - lo
        })
        .fold(0.0, f64::max)
}

/// The closed polygon's length.
pub fn perimeter(points: &[[f64; 3]]) -> f64 {
    let n = points.len();
    (0..n)
        .map(|i| norm(sub(points[(i + 1) % n], points[i])))
        .sum()
}

/// Newell's normal of the points relative to the first.
pub fn newell(points: &[[f64; 3]]) -> [f64; 3] {
    let f = points[0];
    let rel: Vec<[f64; 3]> = points.iter().map(|p| sub(*p, f)).collect();
    let mut n = [0.0; 3];
    for i in 0..rel.len() {
        let a = rel[i];
        let b = rel[(i + 1) % rel.len()];
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    n
}

/// The widest normal: `far × (p − f)` of largest length, `far` the point farthest from the first
/// (the last of equals).
pub fn widest(points: &[[f64; 3]]) -> [f64; 3] {
    let f = points[0];
    let mut far = [0.0; 3];
    let mut best = -1.0;
    for p in points {
        let v = sub(*p, f);
        let l = dot(v, v);
        if l >= best {
            best = l;
            far = v;
        }
    }
    let mut n = [0.0; 3];
    let mut best = -1.0;
    for p in points {
        let c = cross(far, sub(*p, f));
        let l = dot(c, c);
        if l >= best {
            best = l;
            n = c;
        }
    }
    n
}

/// Whether `n` points somewhere: `|n| > max(1e-9 · extent², 4 g sqrt(0.75) · perimeter)`.
pub fn points_somewhere(n: [f64; 3], extent: f64, g: f64, perimeter: f64) -> bool {
    norm(n) > (1e-9 * extent * extent).max(4.0 * g * libm::sqrt(0.75) * perimeter)
}

/// Whether the plane of normal `n` holds the points within `tol` of the first.
fn holds(n: [f64; 3], points: &[[f64; 3]], tol: f64) -> bool {
    let ext = extent(points);
    if !points_somewhere(n, ext, 0.0, 0.0) {
        return false;
    }
    let l = norm(n);
    let u = n.map(|v| v / l);
    let base = dot(u, points[0]);
    points.iter().all(|p| (dot(u, *p) - base).abs() <= tol)
}

/// Whether the points lie on a line: their widest normal points nowhere, with the record's own
/// grid and perimeter.
pub fn on_a_line(points: &[[f64; 3]], g: f64) -> bool {
    !points_somewhere(widest(points), extent(points), g, perimeter(points))
}

/// How a path of points `p` lies: its code where carried, else decided (contract §5); and its
/// plane, fitted to the points as drawn. `face` allows `Nearest`.
pub fn planarity(
    points: &[[f64; 3]],
    code: Option<u8>,
    face: bool,
) -> (Planarity, Option<WorldPlane>) {
    if points.is_empty() {
        return (Planarity::Curve, None);
    }
    let tol = 1e-9 * extent(points).max(1.0);
    let decided = code.and_then(Planarity::of_code).unwrap_or_else(|| {
        for k in [2, 1, 0] {
            if points.iter().all(|p| (p[k] - points[0][k]).abs() <= tol) {
                return Planarity::Axis(k);
            }
        }
        if holds(newell(points), points, tol) {
            return Planarity::Newell;
        }
        if holds(widest(points), points, tol) {
            return Planarity::Widest;
        }
        if face {
            Planarity::Nearest
        } else {
            Planarity::Curve
        }
    });
    let plane = fit(points, decided);
    (decided, plane)
}

/// The plane of a planarity fitted to the points as drawn.
pub fn fit(points: &[[f64; 3]], how: Planarity) -> Option<WorldPlane> {
    match how {
        Planarity::Curve => None,
        Planarity::Axis(k) => Some(WorldPlane::axis(k, points[0])),
        Planarity::Newell => WorldPlane::canonical(newell(points), points[0]),
        Planarity::Widest => WorldPlane::canonical(widest(points), points[0]),
        Planarity::Nearest => {
            let n = points.len() as f64;
            let mut c = [0.0; 3];
            for p in points {
                for k in 0..3 {
                    c[k] += p[k];
                }
            }
            let mut plane = WorldPlane::canonical(newell(points), c.map(|v| v / n))?;
            plane.offset_spread = Some(offset_spread(points, plane.unrounded_n));
            Some(plane)
        }
    }
}

/// `γ_j = j u / (1 − j u)`, infinite where `j u >= 1`.
fn gamma(j: usize) -> f64 {
    let ju = j as f64 * U;
    if ju >= 1.0 {
        f64::INFINITY
    } else {
        ju / (1.0 - ju)
    }
}

/// A `Nearest` face's `δe = 2 Σ_k |û_k| E_k`, with
/// `E_k = γ_{2N+66} (Σ_i |p_ik − p_0k| + Σ_i |p_ik|) / N` over its `N` points (contract §6): how
/// far two conforming computations' offsets can differ, their centroids' share and each offset's
/// own bound included.
fn offset_spread(points: &[[f64; 3]], unit: [f64; 3]) -> f64 {
    let n = points.len();
    let g = gamma(2 * n + 66);
    let e: [f64; 3] = std::array::from_fn(|k| {
        let first: f64 = points.iter().map(|q| (q[k] - points[0][k]).abs()).sum();
        let own: f64 = points.iter().map(|q| q[k].abs()).sum();
        g * (first + own) / n as f64
    });
    2.0 * ((unit[0].abs() * e[0] + unit[1].abs() * e[1]) + unit[2].abs() * e[2])
}

/// `dep_P`: the largest `|n_P · p − e_P| + 4u (|n_P| · |p| + |e_P|)` over the points, plus the
/// plane offset's bound, on the unrounded plane.
pub fn departure_from(p: &WorldPlane, points: &[[f64; 3]]) -> f64 {
    points
        .iter()
        .map(|q| {
            (dot(p.unrounded_n, *q) - p.unrounded_e).abs()
                + 4.0 * U * (abs_dot(p.unrounded_n, *q) + p.unrounded_e.abs())
        })
        .fold(0.0, f64::max)
        + p.e_err
}

/// The camera-facing plane through a world point `centre`: normal `R`'s third row (contract §5,
/// `facing_plane@1`).
pub fn facing_plane(pr: &Projection, centre: [f64; 3]) -> Option<WorldPlane> {
    WorldPlane::canonical(pr.r[2], centre)
}
