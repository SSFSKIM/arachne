//! Free rotation in the 3D solver world, against the closed forms of Euler's equations: a box spun
//! about its middle axis flips over and back (the intermediate-axis theorem) at the times they
//! give, while one spun about its largest axis never flips; and a fast symmetric top on a ball
//! joint precesses about the vertical at `m g r / (I3 w3)` without falling, while the same top
//! unspun falls over. Each case prints its closed form, the measured value and its tolerance
//! (`cargo test -- --show-output`), and the raw numbers behind them.
//!
//! A body's principal moments are computed here from its mass as the world reports it and its
//! shape's half extents as stored: the world exposes no inertia, and a cuboid of uniform density
//! has `I_x = m (b^2 + c^2) / 3` and its turns, `a, b, c` its half extents.

mod common;

use std::f64::consts::{PI, TAU};

use arachne_sim::{Body, Integrator, Joint, JointKind, ShapeKind, World};
use common::*;

type V3 = [f64; 3];

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `v` turned by the unit quaternion `q = (x, y, z, w)`, or by its inverse.
fn turn(q: &[f64], v: V3, inverse: bool) -> V3 {
    let s = if inverse { -1.0 } else { 1.0 };
    let u = [s * q[0], s * q[1], s * q[2]];
    let t = cross(u, v).map(|x| 2.0 * x);
    let c = cross(u, t);
    [0, 1, 2].map(|k| v[k] + q[3] * t[k] + c[k])
}

/// A moving body of one cuboid of density 1, at `position`, turned by `rotation`, spinning at
/// `spin` (in the world's frame).
fn cuboid_body(
    w: &mut World,
    half_extents: &[f64],
    position: V3,
    rotation: [f64; 4],
    spin: V3,
) -> u32 {
    let b = w
        .add_body(Body {
            fixed: false,
            position: fs(&position),
            rotation: fs(&rotation),
            velocity: vec![0.0; 3],
            angular_velocity: fs(&spin),
            mass: 0.0,
            gravity_scale: 1.0,
        })
        .expect("a body");
    let kind = ShapeKind::Cuboid {
        half_extents: fs(half_extents),
    };
    shape(w, b, kind, 1.0, 0.0, 0.0);
    b
}

/// A cuboid's principal moments about its centre, along its own x, y and z, from its mass and its
/// half extents as stored.
fn moments(m: f64, half_extents: &[f64]) -> V3 {
    let h: Vec<f64> = half_extents.iter().map(|&x| f64::from(f(x))).collect();
    let sq = |k: usize| h[k] * h[k];
    [
        m * (sq(1) + sq(2)) / 3.0,
        m * (sq(0) + sq(2)) / 3.0,
        m * (sq(0) + sq(1)) / 3.0,
    ]
}

/// Body `b`'s angular velocity at every row, in its own frame.
fn body_spins(w: &World, b: u32) -> Vec<V3> {
    poses(w, b)
        .iter()
        .zip(velocities(w, b))
        .map(|(p, v)| turn(&p[3..7], [v[3], v[4], v[5]], true))
        .collect()
}

/// The box that tumbles, by its half extents: its principal moments are distinct,
/// `I_x < I_y < I_z`, so y is its middle axis.
const BOX: [f64; 3] = [0.4, 0.25, 0.1];

/// The box alone, without gravity, at rest but for `spin`, for `seconds`: the world, the box and
/// its principal moments.
fn tumble(spin: V3, seconds: f64) -> (World, u32, V3) {
    let mut w = world(3, Integrator::Solver);
    let b = cuboid_body(&mut w, &BOX, [0.0; 3], [0.0, 0.0, 0.0, 1.0], spin);
    w.advance_to(ticks(seconds)).expect("the tumble");
    let m = f64::from(w.mass(b).expect("a mass"));
    (w, b, moments(m, &BOX))
}

/// Twice the kinetic energy and the squared angular momentum of a spin `s` (body frame) of a body
/// of principal moments `i`: the invariants of its free rotation.
fn invariants(i: V3, s: V3) -> (f64, f64) {
    let l = [0, 1, 2].map(|k| i[k] * s[k]);
    (dot(l, s), dot(l, l))
}

/// The lines on a free body's invariants, at every one of the run's `rows`:
///
/// - Its angular momentum in the world's frame, `L = R I R^T w`, stays `L(0)` to 2% of `|L(0)|`.
///   The difference is a vector, so the line holds its direction too: 2% is 1.1 degrees of the
///   axis the body tumbles about, under what a viewer can see of a tumbling body. f32 rounding
///   over the run's substeps accounts for about 1e-5 of it; the rest is left to the solver's
///   step, which is first order in its substep (1/240 s here), and its gyroscopic pass.
/// - Its kinetic energy stays within the crate's energy line, 1% of `E(0)` (common::assert_energy).
fn assert_invariants(case: &str, w: &World, b: u32, i: V3, rows: usize) {
    let mut momenta: Vec<V3> = Vec::new();
    let mut energies: Vec<f64> = Vec::new();
    for (p, s) in poses(w, b).iter().zip(body_spins(w, b)) {
        let l = [0, 1, 2].map(|k| i[k] * s[k]);
        momenta.push(turn(&p[3..7], l, false));
        energies.push(0.5 * dot(l, s));
    }
    let l0 = momenta[0];
    let scale = dot(l0, l0).sqrt();
    let drift: Vec<f64> = momenta
        .iter()
        .map(|l| {
            let d = [0, 1, 2].map(|k| l[k] - l0[k]);
            dot(d, d).sqrt() / scale
        })
        .collect();
    let worst = largest("an angular momentum drift", &drift, rows);
    let last = momenta[momenta.len() - 1];
    eprintln!(
        "{case}: L(0) = {l0:.6?}, |L(0)| = {scale:.6}; L(end) = {last:.6?}; \
         max |L(t) - L(0)| / |L(0)| = {worst:.6}; E(0) = {:.6}, E(end) = {:.6}",
        energies[0],
        energies[energies.len() - 1]
    );
    let ok = worst < 0.02;
    report(
        &format!("{case} angular momentum"),
        &format!("L(t) = R I R^T w = L(0) in the world frame, |L(0)| = {scale:.6}"),
        "max |L(t) - L(0)| / |L(0)|",
        worst,
        "< 0.02 at every tick",
        ok,
    );
    assert_energy(
        case,
        "kinetic only, no potential",
        &energies,
        energies[0],
        rows,
    );
    assert!(
        ok,
        "{case}: the angular momentum moved by {worst:.5} of its size"
    );
}

/// The intermediate-axis theorem: spun about its middle axis with a little spin about its smallest,
/// the box turns over and back. Its polhode circles the smallest axis (`L^2 < 2 E I_y`), and
/// Euler's equations give its spin about the middle axis as a Jacobi elliptic function of
/// `lambda t`, at its largest at `t = 0` and crossing zero at `t = (2n + 1) K(k) / lambda`, with
/// `lambda^2 = (I_y - I_x)(2E I_z - L^2) / (I_x I_y I_z)` and
/// `k^2 = (I_z - I_y)(L^2 - 2E I_x) / ((I_y - I_x)(2E I_z - L^2))`. Its spins about the other two
/// axes are largest as it crosses, where `w_y = 0` and the invariants fix them.
#[test]
fn a_box_spun_about_its_middle_axis_flips_over_and_back_at_the_closed_form_times() {
    // 5 rad/s about the middle axis and 0.1 about the smallest: 5.001 rad/s, an eighth of the
    // world's default angular speed limit of 40 rad/s (the solver's own cap, a quarter turn per
    // tick, is 47.12 rad/s).
    let seconds = 8.0;
    let n = rows(seconds);
    let (w, b, i) = tumble([0.1, 5.0, 0.0], seconds);
    let [ix, iy, iz] = i;
    let spins = body_spins(&w, b);
    // The invariants as the world holds them at its first row.
    let (e2, l2) = invariants(i, spins[0]);
    let lambda = ((iy - ix) * (iz * e2 - l2) / (ix * iy * iz)).sqrt();
    let k = ((iz - iy) * (l2 - ix * e2) / ((iy - ix) * (iz * e2 - l2))).sqrt();
    let quarter = elliptic_k(k) / lambda;
    let middle: Vec<f64> = spins.iter().map(|s| s[1]).collect();
    complete("the spin about the middle axis", &middle, n);
    let flips = crossings(&middle);
    let along = |k: usize| -> Vec<f64> { spins.iter().map(|s| s[k].abs()).collect() };
    let (most_x, most_z) = (
        largest("|w_x|", &along(0), n),
        largest("|w_z|", &along(2), n),
    );
    let (bound_x, bound_z) = (
        ((iz * e2 - l2) / (ix * (iz - ix))).sqrt(),
        ((l2 - ix * e2) / (iz * (iz - ix))).sqrt(),
    );
    eprintln!(
        "middle-axis flip: moments {i:.6?}; w_y crosses zero at {flips:.4?} s, closed form \
         {:.4} s and every {:.4} s after; largest |w_x| {most_x:.4} (closed form {bound_x:.4}), \
         |w_z| {most_z:.4} (closed form {bound_z:.4}); w_y at the end {:.4}",
        quarter,
        2.0 * quarter,
        middle[n - 1]
    );
    assert!(
        flips.len() >= 2,
        "the box did not turn over and back in {seconds} s: its middle-axis spin crossed zero at \
         {flips:?}"
    );
    // 5%: a tenth of a second at the first flip, under what a viewer can time by eye.
    check(
        "middle-axis flip, first",
        "t1 = K(k) / lambda",
        quarter,
        flips[0],
        0.05,
    );
    check(
        "middle-axis flip, back",
        "t2 - t1 = 2 K(k) / lambda",
        2.0 * quarter,
        flips[1] - flips[0],
        0.05,
    );
    // Bounded, and by what the invariants allow: 2% of each.
    check(
        "middle-axis flip, largest |w_x|",
        "w_x,max = sqrt((2E I_z - L^2) / (I_x (I_z - I_x)))",
        bound_x,
        most_x,
        0.02,
    );
    check(
        "middle-axis flip, largest |w_z|",
        "w_z,max = sqrt((L^2 - 2E I_x) / (I_z (I_z - I_x)))",
        bound_z,
        most_z,
        0.02,
    );
    assert_invariants("middle-axis flip", &w, b, i, n);
}

/// The twin: the same box with the same small spin about its smallest axis, spun about its largest
/// instead, wobbles and never flips. Its polhode circles the largest axis (`L^2 > 2 E I_y`), and
/// its spin about it stays between its start and `sqrt((L^2 - 2E I_y) / (I_z (I_z - I_y)))`.
#[test]
fn a_box_spun_about_its_largest_axis_never_flips() {
    // 5 rad/s about the largest axis and 0.1 about the smallest: 5.001 rad/s, an eighth of the
    // world's default angular speed limit of 40 rad/s.
    let seconds = 8.0;
    let n = rows(seconds);
    let (w, b, i) = tumble([0.1, 0.0, 5.0], seconds);
    let [ix, iy, iz] = i;
    let spins = body_spins(&w, b);
    let (e2, l2) = invariants(i, spins[0]);
    let largest_axis: Vec<f64> = spins.iter().map(|s| s[2]).collect();
    complete("the spin about the largest axis", &largest_axis, n);
    let flips = crossings(&largest_axis);
    let lowest = largest_axis.iter().copied().fold(f64::INFINITY, f64::min);
    let floor = ((l2 - e2 * iy) / (iz * (iz - iy))).sqrt();
    // The wobble: the spins about the other two axes, whose amplitudes the invariants fix.
    let along = |k: usize| -> Vec<f64> { spins.iter().map(|s| s[k].abs()).collect() };
    let (most_x, most_y) = (
        largest("|w_x|", &along(0), n),
        largest("|w_y|", &along(1), n),
    );
    let (wobble_x, wobble_y) = (
        ((iz * e2 - l2) / (ix * (iz - ix))).sqrt(),
        ((iz * e2 - l2) / (iy * (iz - iy))).sqrt(),
    );
    eprintln!(
        "largest-axis spin: w_z crosses zero at {flips:?}; lowest w_z {lowest:.6} (closed form \
         {floor:.6}); largest |w_x| {most_x:.4} (closed form {wobble_x:.4}), |w_y| {most_y:.4} \
         (closed form {wobble_y:.4})"
    );
    let ok = flips.is_empty();
    report(
        "largest-axis spin, no flip",
        "w_z keeps its sign",
        "zero crossings of w_z over 8 s,",
        flips.len() as f64,
        "none",
        ok,
    );
    assert!(ok, "the box flipped about its largest axis at {flips:?} s");
    // Its lowest spin to 1% of what the invariants allow.
    check(
        "largest-axis spin, lowest w_z",
        "w_z,min = sqrt((L^2 - 2E I_y) / (I_z (I_z - I_y)))",
        floor,
        lowest,
        0.01,
    );
    assert_invariants("largest-axis spin", &w, b, i, n);
}

/// The top: a square plate, `PLATE` its half extents, so its moments about its own x and z are
/// equal (`I1`) and its symmetry axis is its y (`I3`).
const PLATE: [f64; 3] = [0.5, 0.05, 0.5];
/// From the pivot to the plate's centre, along its axis.
const ARM: f64 = 0.25;
/// The axis's tilt from the vertical as released.
const TILT: f64 = PI / 6.0;

/// The top on a ball joint at a fixed pivot at the origin, its centre `ARM` along its axis from
/// the pivot, the axis tilted `TILT` from the vertical toward +x, spinning at `spin` about its
/// axis, under gravity along -y, for `seconds`. Returns the world and the top.
fn top(spin: f64, seconds: f64) -> (World, u32) {
    let mut w = world(3, Integrator::Solver);
    gravity(&mut w, &[0.0, -G, 0.0]);
    let zero = [0.0; 3];
    let pivot = body(&mut w, true, &zero, &zero, 0.0);
    let axis = [libm::sin(TILT), libm::cos(TILT), 0.0];
    // A turn about z by -TILT takes the plate's y onto the axis.
    let half = -TILT / 2.0;
    let rotation = [0.0, 0.0, libm::sin(half), libm::cos(half)];
    let top = cuboid_body(
        &mut w,
        &PLATE,
        axis.map(|x| ARM * x),
        rotation,
        axis.map(|x| spin * x),
    );
    w.add_joint(Joint {
        kind: JointKind::Ball,
        body1: pivot,
        body2: top,
        anchor1: vec![0.0; 3],
        anchor2: fs(&[0.0, -ARM, 0.0]),
        axis: vec![],
        limits: None,
        contacts: false,
    })
    .expect("the ball joint");
    w.advance_to(ticks(seconds)).expect("the top");
    (w, top)
}

/// The top's symmetry axis at every row, in the world's frame.
fn top_axes(w: &World, top: u32) -> Vec<V3> {
    poses(w, top)
        .iter()
        .map(|p| turn(&p[3..7], [0.0, 1.0, 0.0], false))
        .collect()
}

/// Each axis's tilt from the vertical.
fn tilts(axes: &[V3]) -> Vec<f64> {
    axes.iter()
        .map(|a| libm::acos(a[1].clamp(-1.0, 1.0)))
        .collect()
}

/// Each axis's azimuth as a turn about +y (from +x toward -z), unwrapped across rows.
fn azimuths(axes: &[V3]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(axes.len());
    for a in axes {
        let raw = libm::atan2(-a[2], a[0]);
        let next = match out.last() {
            None => raw,
            Some(&previous) => {
                let d = raw - previous;
                previous + d - TAU * (d / TAU).round()
            }
        };
        out.push(next);
    }
    out
}

/// The top's mass, its moments `I1` (about a transverse axis through its centre) and `I3`, and
/// its arm, as stored.
fn top_constants(w: &World, top: u32) -> (f64, f64, f64, f64) {
    let m = f64::from(w.mass(top).expect("a mass"));
    let [i1, i3, _] = moments(m, &PLATE);
    (m, i1, i3, f64::from(f(ARM)))
}

/// A fast top precesses: its spin's angular momentum `I3 w3` along its axis, turned by gravity's
/// torque `m g r sin(theta)` about the pivot, sweeps the axis about the vertical at
/// `Omega = m g r / (I3 w3)` in the fast-top limit `(I3 w3)^2 >> 4 m g r I1'`, `I1' = I1 + m r^2`
/// its transverse moment about the pivot. Released without precessing, it nutates about that
/// motion at `I3 w3 / I1'` (a period of 0.18 s here), over a band of about
/// `2 m g r I1' sin(theta0) / (I3 w3)^2` (0.8 degrees), and does not fall.
#[test]
fn a_fast_top_precesses_at_the_closed_form_rate_without_falling() {
    // 30 rad/s about its axis, three quarters of the world's default angular speed limit of
    // 40 rad/s (the solver's own cap is 47.12 rad/s); its precession adds 0.5 rad/s to it.
    let seconds = 8.0;
    let n = rows(seconds);
    let (w, top) = top(30.0, seconds);
    let (m, i1, i3, r) = top_constants(&w, top);
    let g = f64::from(f(G));
    let axes = top_axes(&w, top);
    let vels = velocities(&w, top);
    let v0 = &vels[0];
    // The spin about the axis as the world holds it.
    let w3 = dot([v0[3], v0[4], v0[5]], axes[0]);
    let expected = m * g * r / (i3 * w3);
    let pivot_moment = i1 + m * r * r;
    let fast = (i3 * w3) * (i3 * w3) / (4.0 * m * g * r * pivot_moment);
    let nutation_period = TAU * pivot_moment / (i3 * w3);
    let tilt = tilts(&axes);
    complete("the top's tilt", &tilt, n);
    let band = 2.0 * m * g * r * pivot_moment * libm::sin(tilt[0]) / ((i3 * w3) * (i3 * w3));
    // The window: from 0.5 s, past the first nutations (whose half period is 0.09 s), to the end;
    // the nutation's ripple on the azimuth is under a hundredth of what the axis sweeps in it.
    let start = ticks(0.5);
    let psi = azimuths(&axes);
    complete("the top's azimuth", &psi, n);
    let rate = (psi[n - 1] - psi[start]) / ((n - 1 - start) as f64 * TICK);
    let away: Vec<f64> = tilt.iter().map(|t| (t - tilt[0]).abs()).collect();
    let worst = largest("the top's change of tilt", &away, n);
    let (low, high) = (
        tilt.iter().copied().fold(f64::INFINITY, f64::min),
        tilt.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    let last = &vels[n - 1];
    let w3_end = dot([last[3], last[4], last[5]], axes[n - 1]);
    eprintln!(
        "top: m {m:.6}, I1 {i1:.6}, I3 {i3:.6}, r {r:.6}, w3 {w3:.6} (at the end {w3_end:.6}); \
         (I3 w3)^2 / (4 m g r I1') = {fast:.2}; precession measured {rate:.6} rad/s over \
         {:.2}-{seconds:.2} s, expected {expected:.6}; nutation period {nutation_period:.4} s, \
         band {:.3} deg; tilt from {:.3} deg, between {:.3} and {:.3} deg",
        start as f64 * TICK,
        band.to_degrees(),
        tilt[0].to_degrees(),
        low.to_degrees(),
        high.to_degrees()
    );
    check(
        "top precession",
        "Omega = m g r / (I3 w3)",
        expected,
        rate,
        0.1,
    );
    // It does not fall, nor rise: its tilt stays within 5 degrees of its release, six times the
    // nutation's band, and a sixth of the 30 degrees its twin falls through to pass 60.
    let ok = worst < 5.0_f64.to_radians();
    report(
        "top tilt",
        "the tilt stays in its nutation band, about 0.8 degrees",
        "max |theta(t) - theta(0)|, degrees",
        worst.to_degrees(),
        "< 5 at every tick",
        ok,
    );
    assert!(
        ok,
        "the top's tilt moved {:.3} degrees from its release",
        worst.to_degrees()
    );
}

/// The time a body released at rest at `theta0` from the upright takes to reach `theta1`, falling
/// as an inverted physical pendulum, `theta'' = w0^2 sin(theta)`: the integral of
/// `d theta / (w0 sqrt(2 (cos(theta0) - cos(theta))))`, with `theta = theta0 + u^2` to take away
/// the singularity at the start, by the midpoint rule.
fn fall_time(theta0: f64, theta1: f64, w0: f64) -> f64 {
    let steps: u32 = 10_000;
    let du = (theta1 - theta0).sqrt() / f64::from(steps);
    let sum: f64 = (0..steps)
        .map(|k| {
            let u = (f64::from(k) + 0.5) * du;
            let theta = theta0 + u * u;
            2.0 * u / (w0 * (2.0 * (libm::cos(theta0) - libm::cos(theta))).sqrt())
        })
        .sum();
    sum * du
}

/// The twin: the same top, unspun, is an inverted physical pendulum about its pivot,
/// `w0^2 = m g r / I1'`, and falls over: its tilt passes 60 degrees within a second, at the time
/// that pendulum gives.
#[test]
fn the_same_top_unspun_falls_over_at_the_closed_form_time() {
    // Unspun: 0 rad/s as released, about 3.5 rad/s as it passes 60 degrees and under 8 rad/s at
    // the bottom of its fall, a fifth of the world's default angular speed limit of 40 rad/s.
    let seconds = 1.0;
    let n = rows(seconds);
    let (w, top) = top(0.0, seconds);
    let (m, i1, _, r) = top_constants(&w, top);
    let g = f64::from(f(G));
    let tilt = tilts(&top_axes(&w, top));
    complete("the top's tilt", &tilt, n);
    let over = PI / 3.0;
    let past: Vec<f64> = tilt.iter().map(|t| t - over).collect();
    let falls = crossings(&past);
    let w0 = (m * g * r / (i1 + m * r * r)).sqrt();
    let expected = fall_time(tilt[0], over, w0);
    eprintln!(
        "unspun top: tilt from {:.3} deg, at the end {:.3} deg; past 60 deg at {falls:.4?} s, \
         closed form {expected:.4} s",
        tilt[0].to_degrees(),
        tilt[n - 1].to_degrees()
    );
    let ok = !falls.is_empty() && falls[0] < seconds;
    report(
        "unspun top falls",
        "its tilt passes 60 degrees",
        "the first time it does, s,",
        falls.first().copied().unwrap_or(f64::NAN),
        "within 1 s",
        ok,
    );
    assert!(
        ok,
        "the unspun top did not fall past 60 degrees in {seconds} s"
    );
    check(
        "unspun top fall time",
        "t = integral of d theta / (w0 sqrt(2 (cos theta0 - cos theta))), w0^2 = m g r / I1'",
        expected,
        falls[0],
        0.05,
    );
}
