//! What the closed-form tests share: the one conversion of a number to the stored type, reads of a
//! world's rows, the measurements, and the report line each case prints.

#![allow(dead_code)]

use arachne_sim::{Body, Force, Integrator, Shape, ShapeKind, World};

pub const TICK: f64 = 1.0 / 60.0;
pub const G: f64 = 9.8;

/// The one conversion of a number to the stored type, round to nearest even.
pub fn f(x: f64) -> f32 {
    x as f32
}

pub fn fs(xs: &[f64]) -> Vec<f32> {
    xs.iter().map(|&x| f(x)).collect()
}

pub fn world(dimension: usize, integrator: Integrator) -> World {
    World::new(dimension, integrator, false).expect("a world")
}

pub fn gravity(w: &mut World, g: &[f64]) {
    w.add_force(Force::Gravity {
        acceleration: fs(g),
    })
    .expect("gravity");
}

/// A body at rest unless `velocity` is given, of point mass `mass`.
pub fn body(w: &mut World, fixed: bool, position: &[f64], velocity: &[f64], mass: f64) -> u32 {
    let d = w.dimension();
    let mut b = Body {
        fixed,
        position: fs(position),
        rotation: if d == 2 {
            vec![0.0]
        } else {
            vec![0.0, 0.0, 0.0, 1.0]
        },
        velocity: fs(velocity),
        angular_velocity: vec![0.0; if d == 2 { 1 } else { 3 }],
        mass: f(mass),
        gravity_scale: 1.0,
    };
    if b.velocity.is_empty() {
        b.velocity = vec![0.0; d];
    }
    w.add_body(b).expect("a body")
}

/// A shape at the body's origin.
pub fn shape(
    w: &mut World,
    body: u32,
    kind: ShapeKind,
    density: f64,
    friction: f64,
    restitution: f64,
) {
    let d = w.dimension();
    w.add_shape(Shape {
        body,
        kind,
        position: vec![0.0; d],
        rotation: if d == 2 {
            vec![0.0]
        } else {
            vec![0.0, 0.0, 0.0, 1.0]
        },
        density: f(density),
        friction: f(friction),
        restitution: f(restitution),
    })
    .expect("a shape");
}

/// Ticks for `seconds` of simulation.
pub fn ticks(seconds: f64) -> usize {
    (seconds * 60.0).round() as usize
}

/// Body `b`'s pose at every row, as f64.
pub fn poses(w: &World, b: u32) -> Vec<Vec<f64>> {
    rows_of(w.pose(0, w.rows()), w.bodies(), w.pose_width(), b)
}

pub fn velocities(w: &World, b: u32) -> Vec<Vec<f64>> {
    rows_of(w.velocity(0, w.rows()), w.bodies(), w.velocity_width(), b)
}

fn rows_of(values: &[f32], bodies: usize, width: usize, b: u32) -> Vec<Vec<f64>> {
    values
        .chunks_exact(bodies * width)
        .map(|row| {
            let at = b as usize * width;
            row[at..at + width].iter().map(|&x| f64::from(x)).collect()
        })
        .collect()
}

/// The instants at which `series` (one value per row) crosses zero, either way, by linear
/// interpolation between rows.
pub fn crossings(series: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 1..series.len() {
        let (a, b) = (series[i - 1], series[i]);
        if (a < 0.0 && b >= 0.0) || (a >= 0.0 && b < 0.0) {
            out.push((i as f64 - 1.0 + a / (a - b)) * TICK);
        }
    }
    out
}

/// The mean period: twice the mean time between the first and last crossing of zero.
pub fn period(series: &[f64]) -> f64 {
    let c = crossings(series);
    assert!(c.len() >= 3, "fewer than three crossings: {c:?}");
    2.0 * (c[c.len() - 1] - c[0]) / (c.len() - 1) as f64
}

/// The complete elliptic integral of the first kind, K(k), by the arithmetic-geometric mean.
pub fn elliptic_k(k: f64) -> f64 {
    let (mut a, mut b) = (1.0, (1.0 - k * k).sqrt());
    for _ in 0..40 {
        let (an, bn) = ((a + b) / 2.0, (a * b).sqrt());
        a = an;
        b = bn;
    }
    std::f64::consts::PI / (2.0 * a)
}

/// The largest `|E(t) - E(0)| / scale` over every row.
pub fn energy_drift(energies: &[f64], scale: f64) -> f64 {
    assert!(scale > 0.0, "an energy scale is positive");
    energies
        .iter()
        .map(|e| (e - energies[0]).abs() / scale)
        .fold(0.0, f64::max)
}

/// The energy line: the drift stays under 1% of the case's scale at every row. Prints the
/// case's line, then fails if the line is not met.
pub fn assert_energy(case: &str, convention: &str, energies: &[f64], scale: f64) {
    let drift = energy_drift(energies, scale);
    report(
        &format!("{case} energy"),
        &format!("E(t) = E(0), {convention}, scale S = {scale:.6}"),
        "max |E(t) - E(0)| / S",
        drift,
        "< 0.01 at every tick",
        drift < 0.01,
    );
    assert!(drift < 0.01, "{case}: energy drift {drift:.5} of its scale");
}

/// Prints one case: its closed form, what was measured, the tolerance and whether it holds; then
/// fails if it does not.
pub fn check(case: &str, closed_form: &str, expected: f64, measured: f64, tolerance: f64) {
    let relative = (measured - expected).abs() / expected.abs();
    report(
        case,
        &format!("{closed_form} = {expected:.6}"),
        &format!("measured {measured:.6}, relative error"),
        relative,
        &format!("< {tolerance}"),
        relative < tolerance,
    );
    assert!(
        relative < tolerance,
        "{case}: measured {measured} against {expected}, relative error {relative:.5}"
    );
}

pub fn report(
    case: &str,
    closed_form: &str,
    measured: &str,
    value: f64,
    tolerance: &str,
    ok: bool,
) {
    println!(
        "case {case}: closed form {closed_form}; {measured} {value:.6}; tolerance {tolerance}; {}",
        if ok { "holds" } else { "FAILS" }
    );
}

pub fn ball(radius: f64) -> ShapeKind {
    ShapeKind::Ball { radius: f(radius) }
}
