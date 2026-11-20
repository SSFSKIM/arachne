//! What a world finds beside its rows: a body whose state went non-finite, a body above the
//! world's speed limits, and two bodies that passed through each other in a step; each with a
//! twin that finds nothing.

mod common;

use arachne_sim::{
    Body, Force, Integrator, Joint, JointKind, Options, Shape, ShapeKind, World, finding,
};
use common::{ball, body, f, shape};

const NONE: &[[u32; 4]] = &[];

fn options(continuous_collision: bool) -> Options {
    Options {
        continuous_collision,
        ..Options::default()
    }
}

/// A body at `position` moving at `velocity` and turning at `angular_velocity`, of point mass
/// `mass`.
fn moving(
    w: &mut World,
    position: &[f32],
    velocity: &[f32],
    angular_velocity: &[f32],
    mass: f32,
) -> u32 {
    let rotation = if w.dimension() == 2 {
        vec![0.0]
    } else {
        vec![0.0, 0.0, 0.0, 1.0]
    };
    w.add_body(Body {
        fixed: false,
        position: position.to_vec(),
        rotation,
        velocity: velocity.to_vec(),
        angular_velocity: angular_velocity.to_vec(),
        mass,
        gravity_scale: 1.0,
    })
    .expect("a body")
}

/// A shape on `body`, at its origin, of `density`.
fn shaped(w: &mut World, body: u32, kind: ShapeKind, density: f32) {
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
        density,
        friction: 0.0,
        restitution: 0.0,
    })
    .expect("a shape");
}

// A non-finite state

/// Two point masses that attract each other, `apart` units apart: at 0 their attraction is not a
/// number.
fn attracting_points(apart: f64) -> World {
    let mut w = World::new(2, Integrator::PointMasses, false).expect("a world");
    let a = body(&mut w, false, &[0.0, 0.0], &[0.0, 0.0], 1.0);
    let b = body(&mut w, false, &[apart, 0.0], &[0.0, 0.0], 1.0);
    w.add_force(Force::Attraction {
        strength: f(0.1),
        bodies: vec![a, b],
    })
    .expect("an attraction");
    w
}

#[test]
fn a_point_mass_whose_state_goes_non_finite_is_found_once() {
    let mut w = attracting_points(0.0);
    w.advance_to(10).expect("the rows");
    assert_eq!(
        w.findings(),
        &[
            [1, finding::NON_FINITE, 0, 0],
            [1, finding::NON_FINITE, 1, 1]
        ]
    );
    let mut twin = attracting_points(2.0);
    twin.advance_to(60).expect("the rows");
    assert_eq!(twin.findings(), NONE);
}

/// Two discs that attract each other, `apart` units apart, in a solver's world.
fn attracting_discs(apart: f32) -> World {
    let mut w = World::new(2, Integrator::Solver, false).expect("a world");
    let a = moving(&mut w, &[0.0, 0.0], &[0.0, 0.0], &[0.0], 0.0);
    shaped(&mut w, a, ball(0.2), 1.0);
    let b = moving(&mut w, &[apart, 0.0], &[0.0, 0.0], &[0.0], 0.0);
    shaped(&mut w, b, ball(0.2), 1.0);
    w.add_force(Force::Attraction {
        strength: f(0.01),
        bodies: vec![a, b],
    })
    .expect("an attraction");
    w
}

#[test]
fn a_body_the_solver_stops_for_a_non_finite_state_is_found_once() {
    let mut w = attracting_discs(0.0);
    w.advance_to(10).expect("the rows");
    assert_eq!(
        w.findings(),
        &[
            [1, finding::NON_FINITE, 0, 0],
            [1, finding::NON_FINITE, 1, 1]
        ]
    );
    let mut twin = attracting_discs(2.0);
    twin.advance_to(60).expect("the rows");
    assert_eq!(twin.findings(), NONE);
}

// A speed above the world's limits

fn limited(speed: f32, angular: f32, integrator: Integrator) -> World {
    World::with_options(
        2,
        integrator,
        false,
        Options {
            speed_limit: Some(speed),
            angular_speed_limit: Some(angular),
            continuous_collision: true,
        },
    )
    .expect("a world")
}

#[test]
fn a_body_pushed_past_the_speed_limit_is_found_where_it_rises_above() {
    // 10 u/s every second: 5.0 at row 30, 5.1667 at row 31.
    let mut w = limited(5.08, 20.0, Integrator::PointMasses);
    let b = body(&mut w, false, &[0.0, 0.0], &[0.0, 0.0], 1.0);
    w.add_force(Force::Constant {
        body: b,
        force: vec![10.0, 0.0],
        torque: vec![0.0],
    })
    .expect("a push");
    w.advance_to(90).expect("the rows");
    assert_eq!(w.findings(), &[[31, finding::SPEED, 0, 0]]);
}

#[test]
fn a_speed_at_or_under_the_limit_is_not_found() {
    // (6, 8) is exactly 10 u/s in single precision.
    for (velocity, found) in [
        ([6.0, 8.0], NONE),
        ([6.0, 7.99], NONE),
        ([6.0, 8.01], &[[0, finding::SPEED, 0, 0]][..]),
    ] {
        let mut w = limited(10.0, 20.0, Integrator::PointMasses);
        body(&mut w, false, &[0.0, 0.0], &velocity, 1.0);
        w.advance_to(60).expect("the rows");
        assert_eq!(w.findings(), found, "{velocity:?}");
    }
}

#[test]
fn an_angular_speed_above_the_limit_is_found() {
    for (spin, found) in [
        (20.0, NONE),
        (19.0, NONE),
        (21.0, &[[0, finding::ANGULAR_SPEED, 0, 0]][..]),
    ] {
        let mut w = limited(100.0, 20.0, Integrator::Solver);
        let b = moving(&mut w, &[0.0, 0.0], &[0.0, 0.0], &[spin], 0.0);
        shaped(&mut w, b, ball(0.5), 1.0);
        w.advance_to(60).expect("the rows");
        assert_eq!(w.findings(), found, "spin {spin}");
    }
}

#[test]
fn a_limit_is_positive_and_below_the_solver_s_cap() {
    let w = World::new(2, Integrator::Solver, false).expect("a world");
    assert_eq!(w.speed_limit(), arachne_sim::SPEED_LIMIT);
    assert_eq!(w.angular_speed_limit(), arachne_sim::ANGULAR_SPEED_LIMIT);
    assert!(w.speed_limit() < w.speed_cap() && w.speed_cap() == 400.0);
    assert!(w.angular_speed_limit() < w.angular_speed_cap());
    assert!((f64::from(w.angular_speed_cap()) - 15.0 * std::f64::consts::PI).abs() < 1e-4);
    let make = |speed: Option<f32>, angular: Option<f32>| {
        World::with_options(
            2,
            Integrator::Solver,
            false,
            Options {
                speed_limit: speed,
                angular_speed_limit: angular,
                continuous_collision: true,
            },
        )
    };
    let cap = w.angular_speed_cap();
    assert!(make(Some(399.0), Some(cap - 0.5)).is_ok());
    for (speed, angular) in [
        (Some(400.0), None),
        (Some(0.0), None),
        (Some(-1.0), None),
        (Some(f32::NAN), None),
        (None, Some(cap)),
        (None, Some(0.0)),
    ] {
        let err = make(speed, angular).expect_err("a refusal");
        assert!(err.0.contains("below the solver's own cap"), "{err}");
    }
}

#[test]
fn the_limits_are_declared_and_leave_the_trace_alone() {
    let make = |speed: f32| {
        let mut w = limited(speed, 20.0, Integrator::PointMasses);
        body(&mut w, false, &[0.0, 0.0], &[3.0, 0.0], 1.0);
        w.advance_to(30).expect("the rows");
        w
    };
    let (low, high) = (make(1.0), make(50.0));
    assert_eq!(low.findings(), &[[0, finding::SPEED, 0, 0]]);
    assert_eq!(high.findings(), NONE);
    assert!(low.trace_bytes() == high.trace_bytes());
    assert!(low.declaration_bytes() != high.declaration_bytes());
    assert_eq!(arachne_sim::DECLARATION_LAYOUT, 2);
}

// Two bodies that pass through each other

/// A 2D world with a fixed wall 0.04 thick whose middle is at x = 0, and a disc of radius 0.05
/// starting at x = -0.4 and moving toward it at 40 u/s (0.67 units a tick); returns the world and
/// the disc.
fn thin_wall(continuous: bool) -> (World, u32) {
    let mut w =
        World::with_options(2, Integrator::Solver, false, options(continuous)).expect("a world");
    let wall = body(&mut w, true, &[0.0, 0.0], &[0.0, 0.0], 0.0);
    shape(
        &mut w,
        wall,
        ShapeKind::Cuboid {
            half_extents: vec![0.02, 1.0],
        },
        1.0,
        0.0,
        0.0,
    );
    let disc = moving(&mut w, &[-0.4, 0.0], &[40.0, 0.0], &[0.0], 0.0);
    shaped(&mut w, disc, ball(0.05), 1.0);
    (w, disc)
}

#[test]
fn a_fast_body_through_a_thin_wall_is_found_without_continuous_detection() {
    let (mut w, disc) = thin_wall(false);
    w.advance_to(30).expect("the rows");
    assert_eq!(w.findings(), &[[1, finding::TUNNELLED, 0, disc]]);
    assert!(common::poses(&w, disc)[1][0] > 0.05, "it passed through");
    // The same findings however the rows are grouped into calls.
    let (mut pieces, _) = thin_wall(false);
    for row in [1, 2, 7, 30] {
        pieces.advance_to(row).expect("the rows");
    }
    assert_eq!(pieces.findings(), w.findings());
}

#[test]
fn continuous_detection_keeps_a_fast_body_on_its_side_and_finds_nothing() {
    let (mut w, disc) = thin_wall(true);
    w.advance_to(30).expect("the rows");
    assert_eq!(w.findings(), NONE);
    for (row, pose) in common::poses(&w, disc).iter().enumerate() {
        assert!(pose[0] < 0.0, "row {row}: x {}", pose[0]);
    }
}

/// A bar 2 long and 0.1 thick about its centre at the origin, turning at 30 rad/s (half a radian
/// a tick, under the angular speed limit), and a small fixed ball of radius 0.02 at `radius` from
/// the centre, a quarter radian ahead of the bar: the bar's end sweeps across it in the first
/// step unless it is beyond the bar's reach.
fn spinning_bar(dimension: usize, radius: f32) -> World {
    let mut w =
        World::with_options(dimension, Integrator::Solver, false, options(false)).expect("a world");
    let (zero, spin) = if dimension == 2 {
        (vec![0.0; 2], vec![30.0])
    } else {
        (vec![0.0; 3], vec![0.0, 0.0, 30.0])
    };
    let bar = moving(&mut w, &zero, &zero, &spin, 0.0);
    let half = if dimension == 2 {
        vec![1.0, 0.05]
    } else {
        vec![1.0, 0.05, 0.05]
    };
    shaped(&mut w, bar, ShapeKind::Cuboid { half_extents: half }, 1.0);
    let mut at = vec![radius * libm::cosf(0.25), radius * libm::sinf(0.25)];
    at.resize(dimension, 0.0);
    let pin = body(
        &mut w,
        true,
        &at.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        &zero.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        0.0,
    );
    shaped(&mut w, pin, ball(0.02), 1.0);
    w
}

#[test]
fn a_spinning_bar_whose_end_sweeps_through_a_small_shape_is_found() {
    for dimension in [2, 3] {
        let mut w = spinning_bar(dimension, 0.9);
        w.advance_to(1).expect("the row");
        assert_eq!(
            w.findings(),
            &[[1, finding::TUNNELLED, 0, 1]],
            "{dimension}D"
        );
        let mut twin = spinning_bar(dimension, 1.1);
        twin.advance_to(1).expect("the row");
        assert_eq!(twin.findings(), NONE, "{dimension}D, out of reach");
    }
}

/// Two discs of radius 0.05 at x = -0.4 and x = 0.4, the second `offset` above the first,
/// moving toward each other at 30 u/s each: in the first step their centres cross.
fn crossing(dimension: usize, offset: f32, joined: bool) -> World {
    let mut w = World::new(dimension, Integrator::Solver, false).expect("a world");
    let mut place = |x: f32, y: f32, vx: f32| {
        let (mut position, mut velocity) = (vec![x, y], vec![vx, 0.0]);
        position.resize(dimension, 0.0);
        velocity.resize(dimension, 0.0);
        let angular = vec![0.0; if dimension == 2 { 1 } else { 3 }];
        let b = moving(&mut w, &position, &velocity, &angular, 0.0);
        shaped(&mut w, b, ball(0.05), 1.0);
        b
    };
    let (a, b) = (place(-0.4, 0.0, 30.0), place(0.4, offset, -30.0));
    if joined {
        w.add_joint(Joint {
            kind: JointKind::Rope { length: 5.0 },
            body1: a,
            body2: b,
            anchor1: vec![0.0; dimension],
            anchor2: vec![0.0; dimension],
            axis: vec![],
            limits: None,
            contacts: false,
        })
        .expect("a rope");
    }
    w
}

#[test]
fn two_bodies_that_pass_through_each_other_in_a_step_are_found() {
    for dimension in [2, 3] {
        let mut w = crossing(dimension, 0.0, false);
        w.advance_to(10).expect("the rows");
        assert_eq!(
            w.findings(),
            &[[1, finding::TUNNELLED, 0, 1]],
            "{dimension}D"
        );
        for (offset, joined, why) in [(0.5, false, "a miss"), (0.0, true, "kept apart")] {
            let mut twin = crossing(dimension, offset, joined);
            twin.advance_to(10).expect("the rows");
            assert_eq!(twin.findings(), NONE, "{dimension}D, {why}");
        }
    }
}

#[test]
fn bodies_that_touch_and_rest_find_nothing() {
    // A disc dropped onto a floor bounces and comes to rest on it: touching is not passing.
    let mut w = World::new(2, Integrator::Solver, false).expect("a world");
    w.add_force(Force::Gravity {
        acceleration: vec![0.0, -9.8],
    })
    .expect("gravity");
    let floor = body(&mut w, true, &[0.0, -0.5], &[0.0, 0.0], 0.0);
    shape(
        &mut w,
        floor,
        ShapeKind::Cuboid {
            half_extents: vec![4.0, 0.5],
        },
        1.0,
        0.3,
        0.0,
    );
    let disc = moving(&mut w, &[0.0, 1.0], &[0.5, 0.0], &[0.0], 0.0);
    shape(&mut w, disc, ball(0.2), 1.0, 0.3, 0.5);
    w.advance_to(300).expect("5 s");
    assert!(!w.contacts().is_empty());
    assert_eq!(w.findings(), NONE);
}
