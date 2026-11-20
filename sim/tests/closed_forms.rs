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
    assert_energy(
        case,
        "zero at the lowest point",
        &energy,
        energy[0],
        rows(10.0),
    );
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

/// Where a body launched from height 0 comes back to it, by interpolation between rows, and
/// the last row before it does.
fn landing(p: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let i = (2..p.len())
        .find(|&i| p[i - 1][1] >= 0.0 && p[i][1] < 0.0)
        .expect("a landing");
    let s = p[i - 1][1] / (p[i - 1][1] - p[i][1]);
    let at = p[i - 1]
        .iter()
        .zip(&p[i])
        .map(|(a, b)| a + s * (b - a))
        .collect();
    (at, i - 1)
}

/// A projectile of mass 1 launched from the origin at `v0` (as declared) under gravity, for
/// `seconds`: its world and body.
fn projectile(integrator: Integrator, v0: &[f64], seconds: f64) -> (World, u32) {
    let d = v0.len();
    let mut w = world(d, integrator);
    let mut g = vec![0.0; d];
    g[1] = -G;
    gravity(&mut w, &g);
    let zero = vec![0.0; d];
    let b = body(&mut w, false, &zero, v0, 1.0);
    if integrator == Integrator::Solver {
        shape(&mut w, b, ball(0.1), 0.0, 0.0, 0.0);
    }
    w.advance_to(ticks(seconds)).expect("the flight");
    (w, b)
}

/// The declared launch velocity as stored, read back as f64.
fn stored(v0: &[f64]) -> Vec<f64> {
    v0.iter().map(|&x| f64::from(f(x))).collect()
}

/// Its energy at every row: potential zero at the launch height, so `E(0)` is the launch's
/// kinetic energy.
fn projectile_energy(w: &World, b: u32) -> Vec<f64> {
    poses(w, b)
        .iter()
        .zip(velocities(w, b))
        .map(|(p, v)| {
            let speed2: f64 = v[..w.dimension()].iter().map(|x| x * x).sum();
            0.5 * speed2 + G * p[1]
        })
        .collect()
}

/// The projectile: a point mass, whose velocity Verlet is exact for uniform gravity but for
/// rounding, lands at the range its declared launch gives, and keeps its energy over 10 s of
/// flight (no floor: it falls about 440 units below its launch).
#[test]
fn a_projectile_lands_at_its_range_and_keeps_its_energy() {
    let v0 = [4.0, 4.0];
    let (w, b) = projectile(Integrator::PointMasses, &v0, 10.0);
    let v = stored(&v0);
    let (at, _) = landing(&poses(&w, b));
    check(
        "projectile 2d range",
        "R = 2 vx vy / g, v as declared",
        2.0 * v[0] * v[1] / G,
        at[0],
        0.01,
    );
    let energy = projectile_energy(&w, b);
    let scale = energy[0];
    assert_energy(
        "projectile 2d, over 10 s",
        "zero at the launch height",
        &energy,
        scale,
        rows(10.0),
    );
}

#[test]
fn a_projectile_in_3d_lands_at_its_range_and_keeps_its_energy() {
    let v0 = [3.0, 4.0, 2.0];
    let (w, b) = projectile(Integrator::PointMasses, &v0, 10.0);
    let v = stored(&v0);
    let (at, _) = landing(&poses(&w, b));
    let horizontal = (v[0] * v[0] + v[2] * v[2]).sqrt();
    check(
        "projectile 3d range",
        "R = 2 vy |v_horizontal| / g, v as declared",
        2.0 * v[1] * horizontal / G,
        (at[0] * at[0] + at[2] * at[2]).sqrt(),
        0.01,
    );
    let energy = projectile_energy(&w, b);
    let scale = energy[0];
    assert_energy(
        "projectile 3d, over 10 s",
        "zero at the launch height",
        &energy,
        scale,
        rows(10.0),
    );
}

/// The solver's route: a ball's range, from its declared launch. (Its semi-implicit Euler drifts
/// a projectile's energy at first order, about `g^2 h t / v0^2` of the launch's, `h` its
/// substep: the projectile whose energy is checked is the point mass's, above.)
#[test]
fn a_solver_projectile_lands_at_its_range() {
    let v0 = [3.0, 5.0];
    let (w, b) = projectile(Integrator::Solver, &v0, 2.0);
    let v = stored(&v0);
    let (at, _) = landing(&poses(&w, b));
    check(
        "projectile 2d range, solver",
        "R = 2 vx vy / g, v as declared",
        2.0 * v[0] * v[1] / G,
        at[0],
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
    w.advance_to(ticks(10.0)).expect("10 s");
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
    let energy: Vec<f64> = velocities(&w, a)
        .iter()
        .zip(velocities(&w, b))
        .map(|(va, vb)| {
            0.5 * ma * (va[0] * va[0] + va[1] * va[1]) + 0.5 * mb * (vb[0] * vb[0] + vb[1] * vb[1])
        })
        .collect();
    assert_energy(
        "elastic collision",
        "kinetic only, no potential",
        &energy,
        energy[0],
        rows(10.0),
    );
}

/// A block resting on a fixed slope of angle `theta`, both of friction `mu`; returns its offset
/// along the slope, its offset across it (along the slope's normal) and its speed along it at
/// every row, over 2 s.
fn slope(theta: f64, mu: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
    let across = p
        .iter()
        .map(|q| -(q[0] - p[0][0]) * s + (q[1] - p[0][1]) * c)
        .collect();
    let speed = velocities(&w, block)
        .iter()
        .map(|v| v[0] * c + v[1] * s)
        .collect();
    (along, across, speed)
}

#[test]
fn a_block_slides_down_a_slope_at_the_closed_form_acceleration() {
    let (theta, mu) = (f64::from(f(std::f64::consts::PI / 6.0)), 0.2);
    let (_, _, speed) = slope(theta, mu);
    complete("the block's speed", &speed, rows(2.0));
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
    let (along, across, _) = slope(theta, mu);
    for (direction, offsets) in [("along", along), ("across", across)] {
        let offsets: Vec<f64> = offsets.iter().map(|x| x.abs()).collect();
        let moved = largest("the block's offset", &offsets, rows(2.0));
        let ok = moved < 0.005;
        report(
            &format!("block resting on a slope, {direction} it"),
            &format!(
                "no motion while mu >= tan theta ({mu} >= {:.6})",
                libm::tan(theta)
            ),
            &format!("largest offset {direction} the slope over 2 s, units"),
            moved,
            "< 0.005",
            ok,
        );
        assert!(ok, "the block moved {moved} {direction} the slope");
    }
}

/// A bead of mass 1 on a straight wire: a slide joint from a fixed anchor turned by `turn`,
/// along `axis` in the anchor's frame (of any length), the bead itself turned by -0.4 rad.
/// Returns the angle of the wire below the horizontal, then at every row of 10 s the bead's
/// offset down the wire, its offset across the wire, its angle, and its energy, zero at the
/// lowest point the closed form reaches in those 10 s.
fn straight_wire(turn: f64, axis: [f64; 2]) -> (f64, [Vec<f64>; 4]) {
    let mut w = world(2, Integrator::Solver);
    gravity(&mut w, &[0.0, -G]);
    let place = |w: &mut World, fixed: bool, angle: f64, mass: f64| {
        w.add_body(Body {
            fixed,
            position: vec![0.0; 2],
            rotation: fs(&[angle]),
            velocity: vec![0.0; 2],
            angular_velocity: vec![0.0],
            mass: f(mass),
            gravity_scale: 1.0,
        })
        .expect("a body")
    };
    let anchor = place(&mut w, true, turn, 0.0);
    let bead = place(&mut w, false, -0.4, 1.0);
    w.add_joint(Joint {
        kind: JointKind::Slide,
        body1: anchor,
        body2: bead,
        anchor1: vec![0.0; 2],
        anchor2: vec![0.0; 2],
        axis: fs(&axis),
        limits: None,
        contacts: false,
    })
    .expect("the wire");
    w.advance_to(ticks(RUN)).expect("10 s");
    // The wire's direction in the world, as stored, pointed downhill.
    let (turn, ax, ay) = (
        f64::from(f(turn)),
        f64::from(f(axis[0])),
        f64::from(f(axis[1])),
    );
    let n = (ax * ax + ay * ay).sqrt();
    let (c, s) = (libm::cos(turn), libm::sin(turn));
    let (mut ux, mut uy) = ((c * ax - s * ay) / n, (s * ax + c * ay) / n);
    if uy > 0.0 {
        (ux, uy) = (-ux, -uy);
    }
    let theta = libm::asin(-uy);
    let p = poses(&w, bead);
    let down = p.iter().map(|q| q[0] * ux + q[1] * uy).collect();
    let across = p.iter().map(|q| -q[0] * uy + q[1] * ux).collect();
    let angle = p.iter().map(|q| q[2]).collect();
    let lowest = -0.5 * G * libm::sin(theta) * libm::sin(theta) * RUN * RUN;
    let energy = p
        .iter()
        .zip(velocities(&w, bead))
        .map(|(q, v)| 0.5 * (v[0] * v[0] + v[1] * v[1]) + G * (q[1] - lowest))
        .collect();
    (theta, [down, across, angle, energy])
}

/// The straight bead's run: 10 s on a shallow wire, about ten units long.
const RUN: f64 = 10.0;

/// The straight bead: its offset down the wire at 10 s; at every row, its offset across the
/// wire, its angle (a slide does not turn it) and its energy.
fn check_straight_wire(case: &str, turn: f64, axis: [f64; 2]) {
    let (theta, [down, across, angle, energy]) = straight_wire(turn, axis);
    complete("the bead's offset", &down, rows(RUN));
    check(
        &format!("{case}, down it"),
        &format!("s(10 s) = g sin(theta) t^2 / 2, theta = {theta:.6}"),
        0.5 * G * libm::sin(theta) * RUN * RUN,
        down[ticks(RUN)],
        0.01,
    );
    let off: Vec<f64> = across.iter().map(|x| x.abs()).collect();
    let worst = largest("the bead's offset across", &off, rows(RUN));
    let ok = worst < 0.001;
    report(
        &format!("{case}, across it"),
        "on the wire: no offset across it",
        "largest offset across the wire over 10 s, units",
        worst,
        "< 0.001",
        ok,
    );
    assert!(ok, "{case}: the bead left its wire by {worst}");
    let turned: Vec<f64> = angle.iter().map(|a| (a - angle[0]).abs()).collect();
    let worst = largest("the bead's angle", &turned, rows(RUN));
    let ok = worst < 0.001;
    report(
        &format!("{case}, its angle"),
        "a slide does not turn the bead: angle(t) = angle(0)",
        "largest |angle(t) - angle(0)| over 10 s, rad",
        worst,
        "< 0.001",
        ok,
    );
    assert!(ok, "{case}: the bead turned by {worst}");
    let scale = energy[0];
    assert_energy(
        case,
        "zero at the lowest point of its 10 s run",
        &energy,
        scale,
        rows(RUN),
    );
}

/// A wire whose slope has a sine of 0.02: the bead covers 9.8 units in 10 s.
#[test]
fn a_bead_slides_down_a_straight_wire() {
    let theta = libm::asin(0.02);
    check_straight_wire(
        "bead on a straight wire",
        0.0,
        [libm::cos(theta), -libm::sin(theta)],
    );
}

/// The wire's axis is declared in the anchor's frame, at any length: with the anchor turned by
/// 0.7 rad, the axis (2.26, -1.98) in its frame is a wire about 0.019 rad below the horizontal.
#[test]
fn a_bead_slides_down_a_wire_declared_in_a_turned_frame_at_any_length() {
    check_straight_wire(
        "bead on a turned wire, axis of length 3.0",
        0.7,
        [2.26, -1.98],
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
        rows(10.0),
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
    assert_energy(
        "spring",
        "zero at rest length",
        &energy,
        energy[0],
        rows(10.0),
    );
}

#[test]
fn a_damped_oscillator_follows_its_decaying_envelope() {
    let (m, k, c) = (1.0, 10.0, 0.4);
    let (w, mass) = oscillator(m, k, c);
    let gamma = c / (2.0 * m);
    let wd = (k / m - gamma * gamma).sqrt();
    let amplitude = 0.5;
    let off: Vec<f64> = poses(&w, mass)
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let t = i as f64 * TICK;
            let x = amplitude
                * libm::exp(-gamma * t)
                * (libm::cos(wd * t) + gamma / wd * libm::sin(wd * t));
            ((p[0] - 1.0) - x).abs() / amplitude
        })
        .collect();
    let worst = largest("the oscillator's offset", &off, rows(10.0));
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
    let off: Vec<f64> = p.iter().map(|p| (radius(p) - r).abs() / r).collect();
    let worst = largest("the orbit's radius", &off, rows(10.0));
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
        rows(10.0),
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
    let off: Vec<f64> = (0..w.rows())
        .map(|row| {
            let p = momentum(row);
            let d: f64 = (0..3).map(|k| (p[k] - p0[k]) * (p[k] - p0[k])).sum();
            d.sqrt() / scale
        })
        .collect();
    let worst = largest("the momentum's change", &off, rows(10.0));
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
    // The attraction acts: each body's velocity changes over the first tick by the closed-form
    // pull, `dt (a(x0) + a(x1)) / 2` with `a_i = strength sum_j m_j (x_j - x_i) / |x_j - x_i|^3`
    // at the first two rows' positions, in size and in direction.
    let position: Vec<Vec<Vec<f64>>> = bodies.iter().map(|&b| poses(&w, b)).collect();
    let pull = |row: usize, i: usize| -> [f64; 3] {
        let mut a = [0.0; 3];
        for (j, m) in masses.iter().enumerate() {
            if j == i {
                continue;
            }
            let r: Vec<f64> = (0..3)
                .map(|k| position[j][row][k] - position[i][row][k])
                .collect();
            let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            for (k, a) in a.iter_mut().enumerate() {
                *a += 0.5 * m * r[k] / (d * d * d);
            }
        }
        a
    };
    let mut worst: f64 = 0.0;
    for (i, v) in velocity.iter().enumerate() {
        let (a0, a1) = (pull(0, i), pull(1, i));
        let expected: Vec<f64> = (0..3).map(|k| TICK * (a0[k] + a1[k]) / 2.0).collect();
        let size = expected.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(size > 0.0, "body {i} feels no pull");
        let error = (0..3)
            .map(|k| {
                let dv = v[1][k] - v[0][k];
                (dv - expected[k]) * (dv - expected[k])
            })
            .sum::<f64>()
            .sqrt();
        worst = worst.max(error / size);
    }
    let ok = worst < 0.001;
    report(
        "five bodies' pull",
        "dv_i(first tick) = dt (a_i(x0) + a_i(x1)) / 2, a_i = G sum_j m_j r_ij / |r_ij|^3",
        "max |dv_i - closed form| / |closed form| over the five",
        worst,
        "< 0.001",
        ok,
    );
    assert!(ok, "the bodies' first change of velocity is off by {worst}");
}

/// `v` turned by the inverse of the unit quaternion `q = (x, y, z, w)`.
fn unturn(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let (u, w) = ([-q[0], -q[1], -q[2]], q[3]);
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let t = cross(u, v).map(|x| 2.0 * x);
    let c = cross(u, t);
    [0, 1, 2].map(|k| v[k] + w * t[k] + c[k])
}

/// A pin's axis is declared in the first body's frame, at any length, and the bodies may be
/// turned: a pivot turned a quarter turn about y, whose axis (4, 0, 0) is then the world's -z,
/// swings a sphere turned about x in the xy plane, at the physical pendulum's period.
#[test]
fn a_swing_on_a_turned_pivot_with_a_scaled_axis_keeps_its_plane_period_and_energy() {
    let (l, r, theta0) = (2.0, 0.1, 0.5);
    let mut w = world(3, Integrator::Solver);
    gravity(&mut w, &[0.0, -G, 0.0]);
    let quarter = std::f64::consts::FRAC_PI_4;
    let pivot = w
        .add_body(Body {
            fixed: true,
            position: vec![0.0; 3],
            rotation: fs(&[0.0, libm::sin(quarter), 0.0, libm::cos(quarter)]),
            velocity: vec![0.0; 3],
            angular_velocity: vec![0.0; 3],
            mass: 0.0,
            gravity_scale: 1.0,
        })
        .expect("the pivot");
    let at = [l * libm::sin(theta0), -l * libm::cos(theta0), 0.0];
    let q = [libm::sin(0.15), 0.0, 0.0, libm::cos(0.15)];
    let bob = w
        .add_body(Body {
            fixed: false,
            position: fs(&at),
            rotation: fs(&q),
            velocity: vec![0.0; 3],
            angular_velocity: vec![0.0; 3],
            mass: 0.0,
            gravity_scale: 1.0,
        })
        .expect("the bob");
    shape(&mut w, bob, ball(r), 1.0, 0.0, 0.0);
    // The pivot, seen from the bob.
    let anchor2 = unturn(q, at.map(|x| -x));
    w.add_joint(Joint {
        kind: JointKind::Pin,
        body1: pivot,
        body2: bob,
        anchor1: vec![0.0; 3],
        anchor2: fs(&anchor2),
        axis: fs(&[4.0, 0.0, 0.0]),
        limits: None,
        contacts: false,
    })
    .expect("the pin");
    w.advance_to(ticks(10.0)).expect("10 s");
    let p = poses(&w, bob);
    let out: Vec<f64> = p.iter().map(|p| p[2].abs()).collect();
    let worst = largest("the bob's offset out of its plane", &out, rows(10.0));
    let ok = worst < 0.001;
    report(
        "turned swing, its plane",
        "the swing stays in the plane normal to its axis: z(t) = 0",
        "largest |z| over 10 s, units",
        worst,
        "< 0.001",
        ok,
    );
    assert!(ok, "the swing left its plane by {worst}");
    let start = libm::atan2(p[0][0], -p[0][1]);
    let equivalent = l + 0.4 * r * r / l;
    let x: Vec<f64> = p.iter().map(|p| p[0]).collect();
    check(
        "turned swing period",
        "T = 4 sqrt(L/g) K(sin(theta0/2)), L = l + I_cm/(m l)",
        4.0 * (equivalent / G).sqrt() * elliptic_k(libm::sin(start / 2.0)),
        period(&x),
        0.005,
    );
    let energy = swing_energy(&w, bob, l, 0.4 * r * r);
    assert_energy(
        "turned swing",
        "zero at the lowest point",
        &energy,
        energy[0],
        rows(10.0),
    );
}

/// A motor's stiffness is a force per unit of offset, whatever the mass it drives: a slide's
/// motor toward an offset of 0.5 from rest moves a mass `m` as a spring of that stiffness does,
/// `x(t) = 0.5 (1 - cos(sqrt(k/m) t))`, at the period `2 pi sqrt(m/k)`.
#[test]
fn a_motor_drives_two_masses_at_the_periods_its_force_gives() {
    let k = 10.0;
    for m in [1.0, 4.0] {
        let mut w = world(2, Integrator::Solver);
        let anchor = body(&mut w, true, &[0.0, 0.0], &[0.0, 0.0], 0.0);
        let slider = body(&mut w, false, &[0.0, 0.0], &[0.0, 0.0], m);
        let slide = w
            .add_joint(Joint {
                kind: JointKind::Slide,
                body1: anchor,
                body2: slider,
                anchor1: vec![0.0; 2],
                anchor2: vec![0.0; 2],
                axis: vec![1.0, 0.0],
                limits: None,
                contacts: false,
            })
            .expect("the slide");
        w.add_force(Force::Motor {
            joint: slide,
            position: 0.5,
            velocity: 0.0,
            stiffness: f(k),
            damping: 0.0,
        })
        .expect("the motor");
        w.advance_to(ticks(10.0)).expect("10 s");
        let x: Vec<f64> = poses(&w, slider).iter().map(|p| p[0] - 0.5).collect();
        check(
            &format!("slide motor on a mass of {m}, period"),
            "T = 2 pi sqrt(m/k), k the motor's stiffness",
            2.0 * std::f64::consts::PI * (m / k).sqrt(),
            period(&x),
            0.005,
        );
    }
}

/// A slide's offset is zero as declared, wherever its anchors are: a body one unit along the axis
/// from its anchor, both anchors at the bodies' origins, driven by a motor toward offset 0,
/// stays one unit away.
#[test]
fn a_slide_holds_its_declared_separation_at_offset_zero() {
    let mut w = world(2, Integrator::Solver);
    let anchor = body(&mut w, true, &[0.0, 0.0], &[0.0, 0.0], 0.0);
    let slider = body(&mut w, false, &[1.0, 0.0], &[0.0, 0.0], 1.0);
    let slide = w
        .add_joint(Joint {
            kind: JointKind::Slide,
            body1: anchor,
            body2: slider,
            anchor1: vec![0.0; 2],
            anchor2: vec![0.0; 2],
            axis: vec![1.0, 0.0],
            limits: None,
            contacts: false,
        })
        .expect("the slide");
    w.add_force(Force::Motor {
        joint: slide,
        position: 0.0,
        velocity: 0.0,
        stiffness: 10.0,
        damping: 1.0,
    })
    .expect("the motor");
    w.advance_to(ticks(10.0)).expect("10 s");
    let off: Vec<f64> = poses(&w, slider)
        .iter()
        .map(|p| (p[0] - 1.0).abs())
        .collect();
    let worst = largest("the slider's offset", &off, rows(10.0));
    let ok = worst < 1e-4;
    report(
        "slide motor toward offset 0, anchors a unit apart",
        "x(t) = x(0) = 1: the offset is zero as declared",
        "largest |x(t) - 1| over 10 s, units",
        worst,
        "< 0.0001",
        ok,
    );
    assert!(ok, "the slider moved {worst} from where it was declared");
}

/// The stack of `boxes` boxes on a floor at x = 10, declared first in `w`.
fn stack(w: &mut World, boxes: usize) -> Vec<u32> {
    let floor = body(w, true, &[10.0, -0.5], &[0.0, 0.0], 0.0);
    shape(
        w,
        floor,
        ShapeKind::Cuboid {
            half_extents: fs(&[2.0, 0.5]),
        },
        1.0,
        0.5,
        0.0,
    );
    (0..boxes)
        .map(|i| {
            let b = body(w, false, &[10.0, 0.25 + 0.5 * i as f64], &[0.0, 0.0], 0.0);
            shape(
                w,
                b,
                ShapeKind::Cuboid {
                    half_extents: fs(&[0.25, 0.25]),
                },
                1.0,
                0.5,
                0.0,
            );
            b
        })
        .collect()
}

/// The jointed substeps are a component's, not the world's: beside a swing, a separate resting
/// stack is stepped exactly as it is alone (its rows the same bits), at the world's substeps,
/// and stays at rest, while the swing keeps its energy.
#[test]
fn a_swing_beside_a_resting_stack_takes_its_substeps_alone() {
    let mut alone = world(2, Integrator::Solver);
    gravity(&mut alone, &[0.0, -G]);
    let boxes = stack(&mut alone, 3);
    alone.advance_to(ticks(10.0)).expect("10 s");

    let (l, r, theta0) = (2.0, 0.1, 1.0);
    let mut mixed = world(2, Integrator::Solver);
    gravity(&mut mixed, &[0.0, -G]);
    stack(&mut mixed, 3);
    let pivot = body(&mut mixed, true, &[0.0, 0.0], &[0.0, 0.0], 0.0);
    let bob = mixed
        .add_body(Body {
            fixed: false,
            position: fs(&[l * libm::sin(theta0), -l * libm::cos(theta0)]),
            rotation: fs(&[theta0]),
            velocity: vec![0.0; 2],
            angular_velocity: vec![0.0],
            mass: 0.0,
            gravity_scale: 1.0,
        })
        .expect("the bob");
    shape(&mut mixed, bob, ball(r), 1.0, 0.0, 0.0);
    mixed
        .add_joint(Joint {
            kind: JointKind::Pin,
            body1: pivot,
            body2: bob,
            anchor1: vec![0.0; 2],
            anchor2: fs(&[0.0, l]),
            axis: vec![],
            limits: None,
            contacts: false,
        })
        .expect("the rod");
    mixed.advance_to(ticks(10.0)).expect("10 s");

    let mut worst: f64 = 0.0;
    for &b in &boxes {
        let (a, m) = (poses(&alone, b), poses(&mixed, b));
        assert_eq!(a, m, "box {b} moves otherwise beside the swing");
        let moved: Vec<f64> = m
            .iter()
            .map(|q| ((q[0] - m[0][0]).abs()).max((q[1] - m[0][1]).abs()))
            .collect();
        worst = worst.max(largest("a box's offset", &moved, rows(10.0)));
    }
    let ok = worst < 0.005;
    report(
        "resting stack beside a swing",
        "the stack's rows are the bits it has alone, at the world's 4 substeps, at rest",
        "largest offset of a box over 10 s, units (rows equal bit for bit)",
        worst,
        "< 0.005",
        ok,
    );
    assert!(ok, "the stack moved {worst}");
    let energy = swing_energy(&mixed, bob, l, 0.5 * r * r);
    assert_energy(
        "swing beside a stack",
        "zero at the lowest point",
        &energy,
        energy[0],
        rows(10.0),
    );
}

#[test]
fn the_energy_line_holds_a_trace_within_it() {
    let energy: Vec<f64> = (0..601).map(|i| 1.0 + 0.005 * (i % 2) as f64).collect();
    assert_energy("a trace 0.5% off", "test input", &energy, 1.0, rows(10.0));
}

#[test]
#[should_panic(expected = "energy drift")]
fn the_energy_line_fails_a_trace_two_percent_off() {
    let energy: Vec<f64> = (0..601)
        .map(|i| if i == 300 { 1.02 } else { 1.0 })
        .collect();
    assert_energy("a trace 2% off", "test input", &energy, 1.0, rows(10.0));
}

#[test]
#[should_panic(expected = "non-finite sample at row 300")]
fn the_energy_line_fails_a_trace_with_a_sample_not_a_number() {
    let energy: Vec<f64> = (0..601)
        .map(|i| if i == 300 { f64::NAN } else { 1.0 })
        .collect();
    assert_energy("a trace with a NaN", "test input", &energy, 1.0, rows(10.0));
}

#[test]
#[should_panic(expected = "not the 601 of its run")]
fn the_energy_line_fails_a_trace_short_of_its_run() {
    let energy = vec![1.0; 300];
    assert_energy("a trace cut short", "test input", &energy, 1.0, rows(10.0));
}

#[test]
#[should_panic(expected = "non-finite sample")]
fn a_measurement_fails_over_a_sample_not_a_number() {
    let off = [0.0, f64::NAN, 0.0];
    largest("a test input", &off, 3);
}
