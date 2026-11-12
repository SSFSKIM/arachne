//! `band_arith@3` (contract §6, "`R`, the arithmetic bound, by key"): a world plane's `R` read
//! term by term from the contract's text, against the oracle's.

use arachne_oracle::camera::{
    FrameMap, IDENTITY3, Projection, matmul, rotation_about_x, rotation_about_z,
};
use arachne_oracle::depth::{
    PLANE_GRID, Planarity, ScreenKey, WorldPlane, fit, key_plane, nu, plane_r_region,
};
use arachne_oracle::keys::{Model, PlaneS, samples};
use arachne_oracle::shift::U;

fn map() -> FrameMap {
    FrameMap {
        xx: 4.0,
        yy: -4.0,
        x0: 32.0,
        y0: 32.0,
    }
}

fn straight(d: f64) -> Projection {
    Projection {
        r: IDENTITY3,
        c: [0.0; 3],
        d,
        zoom: 1.0,
        exponential: false,
    }
}

/// A camera turned off every axis, about a centre off the origin.
fn turned() -> Projection {
    Projection {
        r: matmul(&rotation_about_x(0.7), &rotation_about_z(0.3)),
        c: [0.2, -0.1, 0.4],
        d: 9.0,
        zoom: 1.1,
        exponential: false,
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    libm::sqrt(dot(v, v))
}

/// `γ_j = j u / (1 − j u)`.
fn gamma(j: usize) -> f64 {
    let ju = j as f64 * U;
    ju / (1.0 - ju)
}

/// A quad that bends off any one plane: a `Nearest` face's points.
fn bent_quad() -> Vec<[f64; 3]> {
    vec![
        [-1.0, -0.75, 0.25],
        [1.25, -1.0, 0.5],
        [1.0, 1.5, 0.375],
        [-0.75, 1.0, 0.125],
    ]
}

/// A plane band read at pixel `(px, py)` with no input share, its own key and `others` the
/// rounded planes `gap` admits.
fn plane_band(
    key: ScreenKey,
    others: &[ScreenKey],
    r0: f64,
    extra: f64,
    fringe: Option<Vec<Vec<[f64; 2]>>>,
    px: i64,
    py: i64,
) -> f64 {
    let m = Model::Plane {
        key,
        r0,
        extra,
        s: PlaneS::Facing { ds: 0.0 },
        fringe,
        rounded: None,
        admitted: std::iter::once(key).chain(others.iter().copied()).collect(),
    };
    let bands = m.pixel(px, py, false);
    let r = bands[0].as_ref().expect("a bounded read").r;
    assert!(
        bands.iter().all(|b| b.as_ref().unwrap().r == r),
        "one R serves the pixel"
    );
    r
}

fn rho_at(k: &ScreenKey, p: [f64; 2]) -> f64 {
    4.0 * U * (k.a.abs() + (k.b * p[0]).abs() + (k.c * p[1]).abs())
}

/// `R = (1 + 64u) ((M + ρ + O) + ν)`, in the order the formula gives: `O` is a term of the sum the
/// factor covers.
#[test]
fn a_plane_reads_r_as_its_terms_summed_in_order_under_one_factor() {
    let key = ScreenKey {
        a: -1.25,
        b: 1.0e-3,
        c: -2.0e-3,
    };
    let (m, o) = (1.0e-12, 0.75e-12);
    let (px, py) = (3, 5);
    let pts = samples(px, py);
    let rho = pts
        .iter()
        .map(|p| rho_at(&key, *p))
        .fold(rho_at(&key, [(px + 1) as f64, (py + 1) as f64]), f64::max);
    let omax = pts
        .iter()
        .map(|p| key.at(p[0], p[1]).abs())
        .fold(0.0, f64::max);
    let want = (1.0 + 64.0 * U) * (((m + rho) + o) + nu(omax));
    assert_eq!(plane_band(key, &[], m, o, None, px, py), want);
}

/// `ρ` is the largest of the reading's rounding at the pixel's far corner and at every point the
/// read takes: a fringe read takes the outline's nearest point, which may lie far outside the pixel.
#[test]
fn rho_is_the_largest_over_the_far_corner_and_every_point_the_read_takes() {
    let key = ScreenKey {
        a: -1.0,
        b: 1.0e-3,
        c: 1.0e-3,
    };
    let ring = vec![
        [1000.0, 1000.0],
        [1010.0, 1000.0],
        [1010.0, 1010.0],
        [1000.0, 1010.0],
    ];
    let r = plane_band(key, &[], 0.0, 0.0, Some(vec![ring]), 0, 0);
    // Every sample of pixel (0, 0) lies outside the ring, so each reads the corner (1000, 1000).
    let far_read = rho_at(&key, [1000.0, 1000.0]);
    assert!(
        far_read > rho_at(&key, [1.0, 1.0]),
        "premise: the read point's rounding is the larger"
    );
    let o = key.at(1000.0, 1000.0);
    assert!(
        r >= far_read + nu(o),
        "R {r:e} under the read point's ρ {far_read:e} and ν"
    );
}

/// A `Nearest` face's `E_k = γ_{2N+66} (Σ_i |p_ik − p_0k| + Σ_i |p_ik|) / N` and
/// `δe = 2 Σ_k |û_k| E_k`; every other plane's is 0.
#[test]
fn a_nearest_faces_offset_spread_is_gamma_2n_plus_66_of_its_points() {
    let pts = bent_quad();
    let p = fit(&pts, Planarity::Nearest).expect("the face's plane");
    let n = pts.len();
    let e: [f64; 3] = std::array::from_fn(|k| {
        let first: f64 = pts.iter().map(|q| (q[k] - pts[0][k]).abs()).sum();
        let own: f64 = pts.iter().map(|q| q[k].abs()).sum();
        gamma(2 * n + 66) * (first + own) / n as f64
    });
    let u = p.unrounded_n;
    let de = 2.0 * ((u[0].abs() * e[0] + u[1].abs() * e[1]) + u[2].abs() * e[2]);
    assert_eq!(p.offset_spread, Some(de));
    assert_eq!(fit(&pts, Planarity::Newell).unwrap().offset_spread, None);
}

/// A plane on the half grid step, `4 · 2^-30` steps past it: outside the shared rounding window,
/// inside a `Nearest` face's `2^-30 + δe / 2^-20`.
fn past_half_step(spread: Option<f64>) -> WorldPlane {
    let e = (7.5 + 4.0 / 1_073_741_824.0) * PLANE_GRID;
    WorldPlane {
        n: [0.0, 0.0, 1.0],
        e: libm::round(e / PLANE_GRID) * PLANE_GRID,
        unrounded_n: [0.0, 0.0, 1.0],
        unrounded_e: e,
        e_err: 0.0,
        offset_spread: spread,
        axis: None,
    }
}

/// The other computation's offset may lie up to `δe` from this one's, so its offset may round to
/// the other neighbour wherever this one's lies within `2^-30 + δe / 2^-20` steps of a half step.
#[test]
fn a_nearest_faces_offset_rounds_either_way_within_its_spread_of_a_half_step() {
    assert_eq!(
        past_half_step(None).conforming().len(),
        1,
        "premise: outside 2^-30"
    );
    let both = past_half_step(Some(16.0 * PLANE_GRID / 1_073_741_824.0)).conforming();
    let offsets: Vec<f64> = both.iter().map(|(_, e)| *e).collect();
    assert_eq!(offsets, vec![8.0 * PLANE_GRID, 7.0 * PLANE_GRID]);
}

/// A plane `z = e` before the eye, `|D|` past its offset's bound but within `δe + ε + 2 dD`.
fn near_the_eye(spread: Option<f64>) -> WorldPlane {
    let e = 10.3 + 1.0e-13;
    WorldPlane {
        n: [0.0, 0.0, 1.0],
        e: libm::round(e / PLANE_GRID) * PLANE_GRID,
        unrounded_n: [0.0, 0.0, 1.0],
        unrounded_e: e,
        e_err: 1.0e-15,
        offset_spread: spread,
        axis: None,
    }
}

/// `Dn = (|D| − δe) − 2 dD` is the nearest the other computation's `D` can stand; where
/// `Dn <= ε` the key may form but `R` is infinite.
#[test]
fn a_nearest_face_the_other_computation_may_hold_too_near_its_eye_is_unbounded() {
    let pr = Projection {
        c: [0.0, 0.0, 0.3],
        ..straight(10.0)
    };
    let region = [0.0, 0.0, 64.0, 64.0];
    assert!(
        plane_r_region(&pr, &map(), &near_the_eye(None), region).is_finite(),
        "premise: |D| alone stands past the offset's bound"
    );
    let r = plane_r_region(&pr, &map(), &near_the_eye(Some(1.0e-13)), region);
    assert_eq!(r, f64::INFINITY);
}

/// `M`, the largest at the region's corners of `gap + 3A + C + shift`, transcribed from §6.
fn m_of(pr: &Projection, map: &FrameMap, p: &WorldPlane, region: [f64; 4]) -> f64 {
    let q = pr.rotated([0.0; 3]);
    let kr = key_plane(pr, map, p, q, 0.0).expect("premise: the plane keys");
    let unrounded = WorldPlane {
        n: p.unrounded_n,
        e: p.unrounded_e,
        ..*p
    };
    let ku = key_plane(pr, map, &unrounded, q, 0.0).expect("premise: the unrounded plane keys");
    let u = p.unrounded_n;
    let de_ = p.offset_spread.unwrap_or(0.0);
    let m = [dot(pr.r[0], u), dot(pr.r[1], u), dot(pr.r[2], u)];
    let big_d = (p.unrounded_e - dot(u, pr.c)) - m[2] * pr.d;
    let (gx, gy) = (m[0] / (pr.zoom * map.xx), m[1] / (pr.zoom * map.yy));
    let dn = 8.0 * U * norm(m);
    let de = 4.0 * U * ((p.unrounded_e.abs() + de_) + norm(m) * norm(pr.c));
    let eye = std::array::from_fn(|k| pr.c[k] + pr.d * pr.r[2][k]);
    let m2d = m[2] * pr.d;
    let dd = (de + dn * norm(eye)) + 4.0 * U * m2d.abs();
    let near = match p.offset_spread {
        Some(spread) => (big_d.abs() - spread) - 2.0 * dd,
        None => big_d.abs(),
    };
    let eps = p.e_err + 4.0 * U * de_;
    let mut worst: f64 = 0.0;
    for (x, y) in [
        (region[0], region[1]),
        (region[2], region[1]),
        (region[0], region[3]),
        (region[2], region[3]),
    ] {
        let terms = |k: &ScreenKey| k.a.abs() + (k.b * x).abs() + (k.c * y).abs();
        let gap = (kr.at(x, y) - ku.at(x, y)).abs() + 4.0 * U * (terms(&kr) + terms(&ku));
        let s = [
            (x - map.x0) / (pr.zoom * map.xx),
            (y - map.y0) / (pr.zoom * map.yy),
            pr.d,
        ];
        let (tx, ty) = (gx * (x - map.x0), gy * (y - map.y0));
        let f = ((m2d - tx) - ty).abs() / near;
        let dnum = dn * norm(s) + 4.0 * U * (tx.abs() + ty.abs() + m2d.abs());
        let a = (dnum + f * dd) / near + 4.0 * U * f;
        let c = (f + dnum / near) * de_ / near;
        let shift = f * eps / (near - eps);
        worst = worst.max(((gap + 3.0 * a) + c) + shift);
    }
    worst
}

/// `M` carries `3A`, `C` and the shift, each at `Dn` and `|e| + δe`: on a `Nearest` face, and on a
/// plane of one computation, where `δe` is 0 and `Dn` is `|D|`.
#[test]
fn m_takes_three_a_c_and_the_shift_at_the_nearest_d_the_other_computation_can_hold() {
    // On the grid, with no offset bound of its own: `gap` is its reads' rounding alone and `C`
    // carries `M`, so `ε`'s `4u δe` reaches `M`'s last bits.
    let on_grid = WorldPlane {
        n: [0.0, 0.0, 1.0],
        e: 2.0,
        unrounded_n: [0.0, 0.0, 1.0],
        unrounded_e: 2.0,
        e_err: 0.0,
        offset_spread: Some(1.0e-9),
        axis: None,
    };
    let pr = turned();
    let region = [10.0, 12.0, 50.0, 54.0];
    let face = fit(&bent_quad(), Planarity::Nearest).unwrap();
    assert!(face.offset_spread.unwrap() > 0.0);
    assert_eq!(face.conforming().len(), 1, "premise: one rounded plane");
    assert_eq!(
        plane_r_region(&pr, &map(), &face, region),
        m_of(&pr, &map(), &face, region)
    );
    let flat = fit(&bent_quad()[..3], Planarity::Newell).unwrap();
    assert_eq!(flat.conforming().len(), 1, "premise: one rounded plane");
    assert_eq!(
        plane_r_region(&pr, &map(), &flat, region),
        m_of(&pr, &map(), &flat, region)
    );
    assert_eq!(on_grid.conforming().len(), 1, "premise: one rounded plane");
    assert_eq!(
        plane_r_region(&pr, &map(), &on_grid, region),
        m_of(&pr, &map(), &on_grid, region)
    );
    // `dn = 8u |m|` and `de`'s `|m| |c|` read `m`'s own length: a rotation of scale 2 parts it
    // from the normal's, which a rotation of unit scale matches to its last bit.
    let scaled = Projection {
        r: pr.r.map(|row| row.map(|v| 2.0 * v)),
        ..turned()
    };
    let one = WorldPlane {
        offset_spread: None,
        ..on_grid
    };
    assert_eq!(
        plane_r_region(&scaled, &map(), &one, region),
        m_of(&scaled, &map(), &one, region)
    );
}

/// A triangle's plane on screen reads `ρ` the same way: at every point it reads as well as the
/// far corner, which on a pixel left of and above the origin is the smaller.
#[test]
fn a_triangle_planes_rho_is_the_largest_over_the_far_corner_and_its_reads() {
    let key = ScreenKey {
        a: -1.0,
        b: 1.0e-3,
        c: 1.0e-3,
    };
    let (px, py) = (-1000, -1000);
    let tri = Model::Tri {
        key,
        bound: 0.0,
        sigma: 0.0,
        kmax: 0.0,
        eps: 0.0,
    };
    let r = tri.pixel(px, py, false)[0]
        .as_ref()
        .expect("a bounded read")
        .r;
    let pts = samples(px, py);
    let rho = pts.iter().map(|p| rho_at(&key, *p)).fold(0.0, f64::max);
    assert!(
        rho > rho_at(&key, [(px + 1) as f64, (py + 1) as f64]),
        "premise: a read's rounding is the larger"
    );
    let omax = pts
        .iter()
        .map(|p| key.at(p[0], p[1]).abs())
        .fold(0.0, f64::max);
    assert!(
        r >= rho + nu(omax),
        "R {r:e} under the reads' ρ {rho:e} and ν"
    );
}

/// On a world plane keyed on screen, `ρ` is also the largest over the rounded planes `gap`
/// admits, since the other computation may hold either: its reading rounding moves with the plane
/// it holds. The departure-gap test could not show this either: its keys are constant
/// (`b = c = 0`).
#[test]
fn a_world_planes_rho_is_the_largest_over_the_rounded_planes_it_admits() {
    let own = ScreenKey {
        a: -1.0,
        b: 1.0e-3,
        c: 1.0e-3,
    };
    let other = ScreenKey { b: 2.0e-3, ..own };
    let (px, py) = (40, 30);
    let pts = samples(px, py);
    let far = [(px + 1) as f64, (py + 1) as f64];
    let rho = |k: &ScreenKey| {
        pts.iter()
            .map(|p| rho_at(k, *p))
            .fold(rho_at(k, far), f64::max)
    };
    assert!(
        rho(&other) > rho(&own),
        "premise: the other plane's ρ is the larger"
    );
    let omax = pts
        .iter()
        .map(|p| own.at(p[0], p[1]).abs())
        .fold(0.0, f64::max);
    let r = plane_band(own, &[other], 0.0, 0.0, None, px, py);
    let want = (1.0 + 64.0 * U) * ((0.0 + rho(&other)) + nu(omax));
    assert!(
        r >= want,
        "R {r:e} under the other plane's ρ and ν, {want:e}"
    );
}

/// Whether `(1 + k u)(1 − γ_N) >= 1 + γ_N'` holds for every `0 < u <= 2^-24`, exactly. Cross
/// multiplied, the difference is `u q(u)` with `q(u) = (k − N − N') + (2 N N' − k (2N + N')) u
/// + 2 k N N' u²`, a parabola opening upwards: it holds where `q(0) > 0`, `q(2^-24) > 0` and, where
/// its vertex lies inside the interval, `q` is positive there (`4 q(0) a_2 > a_1²`).
fn factor_holds(k: i128, n: i128, n2: i128) -> bool {
    let (a0, a1, a2) = (k - n - n2, 2 * n * n2 - k * (2 * n + n2), 2 * k * n * n2);
    let m: i128 = 1 << 24;
    // q(2^-24) 2^48 = a0 2^48 + a1 2^24 + a2
    let at_top = a0 * m * m + a1 * m + a2;
    // The vertex −a1 / (2 a2) inside (0, 2^-24] where 0 < −a1 and −a1 m <= 2 a2.
    let vertex_inside = a1 < 0 && -a1 * m <= 2 * a2;
    a0 > 0 && at_top > 0 && (!vertex_inside || 4 * a0 * a2 > a1 * a1)
}

#[test]
fn the_departure_factor_is_the_derivation_s() {
    // §6 `band_arith@4`: `R = (1 + 128u) ((M + ρ + O) + ν)`. Counted by §6's rule from the code,
    // the order the formulas give reaches `R` through 35 roundings with the factor and any other
    // order through 35 without it (the derivation note): `(1 + 64u)` does not
    // cover them and `(1 + 128u)` does, exactly over `0 < u <= 2^-24`.
    assert!(!factor_holds(64, 35, 35), "64u covers 35 and 35");
    assert!(factor_holds(128, 35, 35), "128u does not cover 35 and 35");
    let key = ScreenKey {
        a: -1.25,
        b: 1.0e-3,
        c: -2.0e-3,
    };
    let (m, o) = (1.0e-12, 0.75e-12);
    let (px, py) = (3, 5);
    let pts = samples(px, py);
    let rho = pts
        .iter()
        .map(|p| rho_at(&key, *p))
        .fold(rho_at(&key, [(px + 1) as f64, (py + 1) as f64]), f64::max);
    let omax = pts
        .iter()
        .map(|p| key.at(p[0], p[1]).abs())
        .fold(0.0, f64::max);
    let want = (1.0 + 128.0 * U) * (((m + rho) + o) + nu(omax));
    assert_eq!(plane_band(key, &[], m, o, None, px, py), want);
}

#[test]
fn the_screen_shift_factor_is_its_count_s() {
    // §6 "Screen shift": `ε(q) = hypot(|xx| x_0, |yy| y_0) (1 + 256u)`. Counted by §6's rule from
    // the inputs (`Δ`, `σ`, `f` with them), ε reaches 63 roundings in the code's order and 69 in
    // another: `(1 + 64u)` and `(1 + 128u)` do not cover them, `(1 + 256u)` does.
    assert!(!factor_holds(64, 63, 69), "64u covers 63 and 69");
    assert!(!factor_holds(128, 63, 69), "128u covers 63 and 69");
    assert!(factor_holds(256, 63, 69), "256u does not cover 63 and 69");
    let shift = arachne_oracle::shift::Shift {
        a: [1.0, 2.0],
        sigma: 0.0,
        projected: None,
        carried: [0.0; 2],
        xx: 4.0,
        yy: -4.0,
    };
    let want = libm::hypot(4.0, 8.0) * (1.0 + 256.0 * U);
    assert_eq!(shift.at([0.0; 3]), want);
}

/// `γ_j = j u / (1 − j u)` at the unit roundoff `u`, infinite where `j u >= 1`, and whether a
/// `Nearest` face of `n` points lies in `E_k`'s domain, `(2N + 90) u < 1`. At d061a532 the oracle
/// has neither as a function of `u`, and keys `E_k` at `γ_{2N+66}`: these stand for its code as it
/// is (`depth.rs` `gamma`, `offset_spread`) until `depth::gamma_at` and `depth::e_k_domain` exist.
fn gamma_at(j: usize, u: f64) -> f64 {
    let ju = j as f64 * u;
    if ju >= 1.0 {
        f64::INFINITY
    } else {
        ju / (1.0 - ju)
    }
}
fn e_k_domain(n: usize, u: f64) -> bool {
    gamma_at(2 * n + 66, u).is_finite()
}

#[test]
fn the_e_k_domain_ends_where_its_gamma_does() {
    // §6: `E_k = γ_{2N+90}` with `n = N + 45`, and wherever `(2N + 90) u >= 1` the face states no
    // bound. This tests the domain predicate's arithmetic through its `u` seam, not a claim at
    // binary32: `u = 2^-24` only puts the edge at a face size a test can name. There the edge is
    // `N = (2^24 − 90) / 2`: `(2N + 90) u = 1`, no bound; one point fewer, a bound.
    let u = 1.0 / 16_777_216.0;
    let edge = (16_777_216 - 90) / 2;
    assert!(!e_k_domain(edge, u), "N = {edge} inside the domain");
    assert!(gamma_at(2 * edge + 90, u).is_infinite());
    assert!(
        e_k_domain(edge - 1, u),
        "N = {} outside the domain",
        edge - 1
    );
    assert!(gamma_at(2 * (edge - 1) + 90, u).is_finite());
}

#[test]
fn a_nan_term_leaves_the_departure_bound_unbounded() {
    // §6: a term of `M` that is NaN states no bound (P-NaN): `M` is not the largest of the corners
    // whose sum stayed a number. The axis plane z = 0 under the identity rotation at a positive
    // subnormal focal distance: at every corner off the centre `dN / Dn` overflows, so `A` is ∞
    // and `C = ∞ · 0` (no `Nearest` spread) is NaN.
    let pr = Projection {
        d: 5e-324,
        ..straight(1.0)
    };
    let plane = WorldPlane::axis(2, [0.0; 3]);
    let m = plane_r_region(&pr, &map(), &plane, [40.0, 40.0, 48.0, 48.0]);
    assert!(m.is_infinite(), "M {m:e}");
}
