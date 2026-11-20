//! Worlds whose motion has a closed form, at the scale a lesson uses (lengths of 1 to 4 units,
//! periods of 1 to 4 s, a gravity of 9.8): each case prints its closed form, the measured value
//! and its tolerance (`cargo test -- --show-output`). The conservative cases also keep their
//! energy within 1% of a stated positive scale at every tick of 10 s.

mod common;

use arachne_sim::{Body, Force, Integrator, Joint, JointKind, ShapeKind, World};
use common::*;

/// A swing: a disc (2D) or sphere (3D) of radius `r` on a massless rigid rod of length `l` from a
/// fixed pivot at the origin, released at rest `theta0` from the bottom. Returns the world and the
/// bob.
fn swing(dimension: usize, l: f64, r: f64, theta0: f64) -> (World, u32) {
    let mut w = world(dimension, Integrator::Solver);
    let (s, c) = (libm::sin(theta0), libm::cos(theta0));
    let zero = vec![0.0; dimension];
    if dimension == 2 {
        gravity(&mut w, &[0.0, -G]);
    } else {
        gravity(&mut w, &[0.0, -G, 0.0]);
    }
    let pivot = body(&mut w, true, &zero, &zero, 0.0);
    let (position, rotation) = if dimension == 2 {
        (fs(&[l * s, -l * c]), fs(&[theta0]))
    } else {
        let half = theta0 / 2.0;
        (
            fs(&[l * s, -l * c, 0.0]),
            fs(&[0.0, 0.0, libm::sin(half), libm::cos(half)]),
        )
    };
    let bob = w
        .add_body(Body {
            fixed: false,
            position,
            rotation,
            velocity: vec![0.0; dimension],
            angular_velocity: vec![0.0; if dimension == 2 { 1 } else { 3 }],
            mass: 0.0,
            gravity_scale: 1.0,
        })
        .expect("the bob");
    shape(&mut w, bob, ball(r), 1.0, 0.0, 0.0);
    let mut anchor2 = vec![0.0; dimension];
    anchor2[1] = f(l);
    w.add_joint(Joint {
        kind: JointKind::Pin,
        body1: pivot,
        body2: bob,
        anchor1: vec![0.0; dimension],
        anchor2,
        axis: if dimension == 2 {
            vec![]
        } else {
            vec![0.0, 0.0, 1.0]
        },
        limits: None,
        contacts: false,
    })
    .expect("the rod");
    w.advance_to(ticks(10.0)).expect("10 s");
    (w, bob)
}

/// The swing's energy at every row, zero at the lowest point: translational, rotational about
/// the bob's centre (`inertia` times its mass), and potential.
fn swing_energy(w: &World, bob: u32, l: f64, inertia: f64) -> Vec<f64> {
    let m = f64::from(w.mass(bob).expect("a mass"));
    let spin = if w.dimension() == 2 { 2 } else { 5 };
    poses(w, bob)
        .iter()
        .zip(velocities(w, bob))
        .map(|(p, v)| {
            let speed2 = v[0] * v[0] + v[1] * v[1];
            let omega = v[spin];
            0.5 * m * speed2 + 0.5 * inertia * m * omega * omega + m * G * (p[1] + l)
        })
        .collect()
}

/// A physical pendulum's period at any amplitude: `4 sqrt(L/g) K(sin(theta0 / 2))`, `L` its
/// equivalent length `l + I_cm / (m l)`.
fn check_swing(case: &str, dimension: usize, theta0: f64, inertia: f64) {
    let (l, r) = (2.0, 0.1);
    let (w, bob) = swing(dimension, l, r, theta0);
    let p = poses(&w, bob);
    // The released angle as stored.
    let start = libm::atan2(p[0][0], -p[0][1]);
    let equivalent = l + inertia * r * r / l;
    let expected = 4.0 * (equivalent / G).sqrt() * elliptic_k(libm::sin(start / 2.0));
    let x: Vec<f64> = p.iter().map(|p| p[0]).collect();
    check(
        &format!("{case} period"),
        "T = 4 sqrt(L/g) K(sin(theta0/2)), L = l + I_cm/(m l)",
        expected,
        period(&x),
        0.005,
    );
    let energy = swing_energy(&w, bob, l, inertia * r * r);
    assert_energy(case, "zero at the lowest point", &energy, energy[0]);
}

#[test]
fn a_frictionless_swing_at_a_small_amplitude_keeps_its_period_and_energy() {
    check_swing("pendulum 2d, 0.1 rad", 2, 0.1, 0.5);
}

#[test]
fn a_frictionless_swing_at_a_large_amplitude_keeps_the_elliptic_period_and_energy() {
    check_swing("pendulum 2d, 2.0 rad", 2, 2.0, 0.5);
}

#[test]
fn a_frictionless_swing_in_3d_keeps_its_period_and_energy() {
    check_swing("pendulum 3d, 0.3 rad", 3, 0.3, 0.4);
}

/// Where a body launched from height 0 comes back to it, by interpolation between rows.
fn landing(p: &[Vec<f64>]) -> Vec<f64> {
    let i = (2..p.len())
        .find(|&i| p[i - 1][1] >= 0.0 && p[i][1] < 0.0)
        .expect("a landing");
    let s = p[i - 1][1] / (p[i - 1][1] - p[i][1]);
    p[i - 1]
        .iter()
        .zip(&p[i])
        .map(|(a, b)| a + s * (b - a))
        .collect()
}

#[test]
fn a_projectile_lands_at_its_range() {
    let mut w = world(2, Integrator::Solver);
    gravity(&mut w, &[0.0, -G]);
    let b = body(&mut w, false, &[0.0, 0.0], &[4.0, 4.0], 0.0);
    shape(&mut w, b, ball(0.1), 1.0, 0.0, 0.0);
    w.advance_to(ticks(2.0)).expect("2 s");
    let v0 = &velocities(&w, b)[0];
    let at = landing(&poses(&w, b));
    check(
        "projectile 2d range",
        "R = 2 vx vy / g",
        2.0 * v0[0] * v0[1] / G,
        at[0],
        0.01,
    );
}

#[test]
fn a_projectile_in_3d_lands_at_its_range() {
    let mut w = world(3, Integrator::Solver);
    gravity(&mut w, &[0.0, -G, 0.0]);
    let b = body(&mut w, false, &[0.0, 0.0, 0.0], &[3.0, 4.0, 2.0], 0.0);
    shape(&mut w, b, ball(0.1), 1.0, 0.0, 0.0);
    w.advance_to(ticks(2.0)).expect("2 s");
    let v0 = &velocities(&w, b)[0];
    let at = landing(&poses(&w, b));
    let horizontal = (v0[0] * v0[0] + v0[2] * v0[2]).sqrt();
    check(
        "projectile 3d range",
        "R = 2 vy |v_horizontal| / g",
        2.0 * v0[1] * horizontal / G,
        (at[0] * at[0] + at[2] * at[2]).sqrt(),
        0.01,
    );
}

#[test]
fn an_elastic_collision_keeps_momentum_and_energy() {
    let mut w = world(2, Integrator::Solver);
    let a = body(&mut w, false, &[-1.0, 0.0], &[2.0, 0.0], 0.0);
    shape(&mut w, a, ball(0.2), 1.0, 0.0, 1.0);
    let b = body(&mut w, false, &[0.5, 0.0], &[0.0, 0.0], 0.0);
    shape(&mut w, b, ball(0.3), 1.0, 0.0, 1.0);
    w.advance_to(ticks(1.5)).expect("1.5 s");
    let (ma, mb) = (
        f64::from(w.mass(a).expect("a")),
        f64::from(w.mass(b).expect("b")),
    );
    let u = velocities(&w, a)[0][0];
    let va = velocities(&w, a).last().expect("a row")[0];
    let vb = velocities(&w, b).last().expect("a row")[0];
    check(
        "elastic collision momentum",
        "p = ma u",
        ma * u,
        ma * va + mb * vb,
        0.001,
    );
    check(
        "elastic collision energy",
        "E = ma u^2 / 2",
        0.5 * ma * u * u,
        0.5 * ma * va * va + 0.5 * mb * vb * vb,
        0.01,
    );
    check(
        "elastic collision first velocity",
        "va = (ma - mb) u / (ma + mb)",
        (ma - mb) * u / (ma + mb),
        va,
        0.01,
    );
    check(
        "elastic collision second velocity",
        "vb = 2 ma u / (ma + mb)",
        2.0 * ma * u / (ma + mb),
        vb,
        0.01,
    );
}

/// A block resting on a fixed slope of angle `theta`, both of friction `mu`; returns its offset
/// along the slope and its speed along it at every row, over 2 s.
fn slope(theta: f64, mu: f64) -> (Vec<f64>, Vec<f64>) {
    let mut w = world(2, Integrator::Solver);
    gravity(&mut w, &[0.0, -G]);
    let (s, c) = (libm::sin(theta), libm::cos(theta));
    let place = |w: &mut World, fixed: bool, at: [f64; 2], half: [f64; 2]| {
        let b = w
            .add_body(Body {
                fixed,
                position: fs(&at),
                rotation: fs(&[theta]),
                velocity: vec![0.0; 2],
                angular_velocity: vec![0.0],
                mass: 0.0,
                gravity_scale: 1.0,
            })
            .expect("a body");
        shape(
            w,
            b,
            ShapeKind::Cuboid {
                half_extents: fs(&half),
            },
            1.0,
            mu,
            0.0,
        );
        b
    };
    place(&mut w, true, [0.0, 0.0], [4.0, 0.2]);
    // On the slope's surface: 0.2 + 0.1 along its normal.
    let block = place(&mut w, false, [-0.3 * s, 0.3 * c], [0.2, 0.1]);
    w.advance_to(ticks(2.0)).expect("2 s");
    let p = poses(&w, block);
    let along = p
        .iter()
        .map(|q| (q[0] - p[0][0]) * c + (q[1] - p[0][1]) * s)
        .collect();
    let speed = velocities(&w, block)
        .iter()
        .map(|v| v[0] * c + v[1] * s)
        .collect();
    (along, speed)
}

#[test]
fn a_block_slides_down_a_slope_at_the_closed_form_acceleration() {
    let (theta, mu) = (f64::from(f(std::f64::consts::PI / 6.0)), 0.2);
    let (_, speed) = slope(theta, mu);
    // Down the slope is the negative direction along it.
    let measured = -(speed[ticks(1.5)] - speed[ticks(0.5)]);
    check(
        "block sliding on a slope",
        "a = g (sin theta - mu cos theta)",
        G * (libm::sin(theta) - f64::from(f(mu)) * libm::cos(theta)),
        measured,
        0.01,
    );
}

#[test]
fn a_block_stays_on_a_slope_its_friction_holds() {
    let (theta, mu) = (f64::from(f(std::f64::consts::PI / 6.0)), 0.8);
    let (along, _) = slope(theta, mu);
    let moved = along.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let ok = moved < 0.005;
    report(
        "block resting on a slope",
        &format!(
            "no motion while mu >= tan theta ({mu} >= {:.6})",
            libm::tan(theta)
        ),
        "largest offset along the slope over 2 s, units",
        moved,
        "< 0.005",
        ok,
    );
    assert!(ok, "the block moved {moved} along the slope");
}

#[test]
fn a_bead_slides_down_a_straight_wire() {
    let theta = 0.5;
    let mut w = world(2, Integrator::Solver);
    gravity(&mut w, &[0.0, -G]);
    let anchor = body(&mut w, true, &[0.0, 0.0], &[0.0, 0.0], 0.0);
    let bead = body(&mut w, false, &[0.0, 0.0], &[0.0, 0.0], 1.0);
    let axis = fs(&[libm::cos(theta), -libm::sin(theta)]);
    w.add_joint(Joint {
        kind: JointKind::Slide,
        body1: anchor,
        body2: bead,
        anchor1: vec![0.0; 2],
        anchor2: vec![0.0; 2],
        axis: axis.clone(),
        limits: None,
        contacts: false,
    })
    .expect("the wire");
    w.advance_to(ticks(1.0)).expect("1 s");
    let (ax, ay) = (f64::from(axis[0]), f64::from(axis[1]));
    let norm = (ax * ax + ay * ay).sqrt();
    let p = &poses(&w, bead)[ticks(1.0)];
    let along = (p[0] * ax + p[1] * ay) / norm;
    let sin = -ay / norm;
    check(
        "bead on a straight wire",
        "s(1 s) = g sin(theta) t^2 / 2",
        0.5 * G * sin,
        along,
        0.01,
    );
}

#[test]
fn a_bead_on_a_circular_wire_reaches_the_bottom_at_the_closed_form_speed() {
    let (radius, r, theta0) = (1.5, 0.05, 1.2);
    let (w, bead) = swing(2, radius, r, theta0);
    let p = poses(&w, bead);
    let v = velocities(&w, bead);
    let start = libm::atan2(p[0][0], -p[0][1]);
    // The first time it passes under the centre.
    let i = (1..p.len())
        .find(|&i| p[i - 1][0] > 0.0 && p[i][0] <= 0.0)
        .expect("the bottom");
    let s = p[i - 1][0] / (p[i - 1][0] - p[i][0]);
    let speed = |v: &Vec<f64>| (v[0] * v[0] + v[1] * v[1]).sqrt();
    let measured = speed(&v[i - 1]) + s * (speed(&v[i]) - speed(&v[i - 1]));
    let disc = 0.5 * r * r / (radius * radius);
    check(
        "bead on a circular wire, speed at the bottom",
        "v = sqrt(2 g R (1 - cos theta0) / (1 + I_cm/(m R^2)))",
        (2.0 * G * radius * (1.0 - libm::cos(start)) / (1.0 + disc)).sqrt(),
        measured,
        0.01,
    );
    let energy = swing_energy(&w, bead, radius, 0.5 * r * r);
    assert_energy(
        "bead on a circular wire",
        "zero at the lowest point",
        &energy,
        energy[0],
    );
}

/// A point mass `m` on a spring of stiffness `k` and rest length 1 to a fixed anchor at the
/// origin, released at rest 0.5 beyond its rest length, with a linear drag `c`.
fn oscillator(m: f64, k: f64, c: f64) -> (World, u32) {
    let mut w = world(2, Integrator::PointMasses);
    let anchor = body(&mut w, true, &[0.0, 0.0], &[0.0, 0.0], 0.0);
    let mass = body(&mut w, false, &[1.5, 0.0], &[0.0, 0.0], m);
    w.add_force(Force::Spring {
        body_a: anchor,
        anchor_a: vec![0.0; 2],
        body_b: mass,
        anchor_b: vec![0.0; 2],
        stiffness: f(k),
        damping: 0.0,
        rest: 1.0,
    })
    .expect("the spring");
    if c > 0.0 {
        w.add_force(Force::LinearDrag {
            body: mass,
            coefficient: f(c),
        })
        .expect("the drag");
    }
    w.advance_to(ticks(10.0)).expect("10 s");
    (w, mass)
}

#[test]
fn a_spring_keeps_its_period_and_energy() {
    let (m, k) = (1.0, 10.0);
    let (w, mass) = oscillator(m, k, 0.0);
    let x: Vec<f64> = poses(&w, mass).iter().map(|p| p[0] - 1.0).collect();
    check(
        "spring period",
        "T = 2 pi sqrt(m/k)",
        2.0 * std::f64::consts::PI * (m / k).sqrt(),
        period(&x),
        0.005,
    );
    let energy: Vec<f64> = x
        .iter()
        .zip(velocities(&w, mass))
        .map(|(x, v)| 0.5 * m * (v[0] * v[0] + v[1] * v[1]) + 0.5 * k * x * x)
        .collect();
    assert_energy("spring", "zero at rest length", &energy, energy[0]);
}

#[test]
fn a_damped_oscillator_follows_its_decaying_envelope() {
    let (m, k, c) = (1.0, 10.0, 0.4);
    let (w, mass) = oscillator(m, k, c);
    let gamma = c / (2.0 * m);
    let wd = (k / m - gamma * gamma).sqrt();
    let amplitude = 0.5;
    let worst = poses(&w, mass)
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let t = i as f64 * TICK;
            let x = amplitude
                * libm::exp(-gamma * t)
                * (libm::cos(wd * t) + gamma / wd * libm::sin(wd * t));
            ((p[0] - 1.0) - x).abs() / amplitude
        })
        .fold(0.0, f64::max);
    let ok = worst < 0.01;
    report(
        "damped oscillator",
        "x(t) = A e^(-gamma t) (cos wd t + (gamma/wd) sin wd t), gamma = c/2m, wd = sqrt(k/m - gamma^2)",
        "max |x - x(t)| / A over 10 s",
        worst,
        "< 0.01",
        ok,
    );
    assert!(
        ok,
        "the oscillator left its closed form by {worst} of its amplitude"
    );
}

#[test]
fn a_circular_orbit_keeps_its_radius_period_and_energy() {
    let (big, r, strength): (f64, f64, f64) = (40.0, 2.0, 1.0);
    let speed = (strength * big / r).sqrt();
    let mut w = world(2, Integrator::PointMasses);
    let sun = body(&mut w, true, &[0.0, 0.0], &[0.0, 0.0], big);
    let planet = body(&mut w, false, &[r, 0.0], &[0.0, speed], 1.0);
    w.add_force(Force::Attraction {
        strength: f(strength),
        bodies: vec![sun, planet],
    })
    .expect("the attraction");
    w.advance_to(ticks(10.0)).expect("10 s");
    let p = poses(&w, planet);
    let radius = |p: &Vec<f64>| (p[0] * p[0] + p[1] * p[1]).sqrt();
    let worst = p
        .iter()
        .map(|p| (radius(p) - r).abs() / r)
        .fold(0.0, f64::max);
    let ok = worst < 0.005;
    report(
        "circular orbit radius",
        &format!("r(t) = {r}"),
        "max |r(t) - r| / r over 10 s",
        worst,
        "< 0.005",
        ok,
    );
    assert!(ok, "the orbit's radius moved by {worst}");
    let y: Vec<f64> = p.iter().map(|p| p[1]).collect();
    check(
        "circular orbit period",
        "T = 2 pi r^(3/2) / sqrt(G M)",
        2.0 * std::f64::consts::PI * r * r.sqrt() / (strength * big).sqrt(),
        period(&y),
        0.005,
    );
    let energy: Vec<f64> = p
        .iter()
        .zip(velocities(&w, planet))
        .map(|(p, v)| 0.5 * (v[0] * v[0] + v[1] * v[1]) - strength * big / radius(p))
        .collect();
    assert_energy(
        "circular orbit",
        "zero at infinite separation, negative when bound",
        &energy,
        energy[0].abs(),
    );
}

#[test]
fn n_bodies_attracting_each_other_keep_their_momentum() {
    let mut w = world(3, Integrator::PointMasses);
    let start = [
        ([3.0, 0.0, 0.0], [0.0, 0.8, 0.1]),
        ([0.9, 2.8, -0.3], [-0.7, 0.2, 0.0]),
        ([-2.4, 1.7, 0.2], [-0.3, -0.6, 0.2]),
        ([-2.4, -1.8, 0.0], [0.5, -0.4, -0.1]),
        ([0.9, -2.9, 0.4], [0.6, 0.3, 0.0]),
    ];
    let bodies: Vec<u32> = start
        .iter()
        .enumerate()
        .map(|(i, (x, v))| body(&mut w, false, x, v, (i + 1) as f64))
        .collect();
    w.add_force(Force::Attraction {
        strength: 0.5,
        bodies: bodies.clone(),
    })
    .expect("the attraction");
    w.advance_to(ticks(10.0)).expect("10 s");
    let masses: Vec<f64> = bodies
        .iter()
        .map(|&b| f64::from(w.mass(b).expect("a mass")))
        .collect();
    let velocity: Vec<Vec<Vec<f64>>> = bodies.iter().map(|&b| velocities(&w, b)).collect();
    let momentum = |row: usize| -> [f64; 3] {
        let mut p = [0.0; 3];
        for (m, v) in masses.iter().zip(&velocity) {
            for (k, p) in p.iter_mut().enumerate() {
                *p += m * v[row][k];
            }
        }
        p
    };
    let scale: f64 = masses
        .iter()
        .zip(&velocity)
        .map(|(m, v)| m * (v[0][0] * v[0][0] + v[0][1] * v[0][1] + v[0][2] * v[0][2]).sqrt())
        .sum();
    let p0 = momentum(0);
    let worst = (0..w.rows())
        .map(|row| {
            let p = momentum(row);
            let d: f64 = (0..3).map(|k| (p[k] - p0[k]) * (p[k] - p0[k])).sum();
            d.sqrt() / scale
        })
        .fold(0.0, f64::max);
    let ok = worst < 1e-4;
    report(
        "five bodies' momentum",
        "P(t) = P(0) = sum m v(0)",
        "max |P(t) - P(0)| / sum m |v(0)| over 10 s",
        worst,
        "< 0.0001",
        ok,
    );
    assert!(ok, "momentum moved by {worst} of its scale");
}

#[test]
fn the_energy_line_holds_a_trace_within_it() {
    let energy: Vec<f64> = (0..600).map(|i| 1.0 + 0.005 * (i % 2) as f64).collect();
    assert_energy("a trace 0.5% off", "test input", &energy, 1.0);
}

#[test]
#[should_panic(expected = "energy drift")]
fn the_energy_line_fails_a_trace_two_percent_off() {
    let energy: Vec<f64> = (0..600)
        .map(|i| if i == 300 { 1.02 } else { 1.0 })
        .collect();
    assert_energy("a trace 2% off", "test input", &energy, 1.0);
}
