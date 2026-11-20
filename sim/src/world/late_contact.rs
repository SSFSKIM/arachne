//! The late-contact rule's fixtures (`World::findings`): a pair the sweep cannot show apart is not
//! found when it ends the step within the solver's prediction distance, on the side it began,
//! having penetrated no deeper than the allowed linear error. Each fixture is one step of a 2D
//! world with continuous collision off, and is checked with the whole rule and with each of its
//! two later conditions dropped, which shows what each condition keeps found.

use super::{Condition, Engine, Integrator, Options, World, finding};
use crate::encode::Settings;
use crate::spec::{Body, Shape, ShapeKind};

const FOUND: &[[u32; 4]] = &[[1, finding::TUNNELLED, 0, 1]];
const NONE: &[[u32; 4]] = &[];

/// A box body of half extents `half` at `position`, turned by `angle`, moving at `velocity` and
/// turning at `spin`.
fn plate(
    w: &mut World,
    fixed: bool,
    half: [f32; 2],
    position: [f32; 2],
    angle: f32,
    velocity: [f32; 2],
    spin: f32,
) {
    let body = w
        .add_body(Body {
            fixed,
            position: position.to_vec(),
            rotation: vec![angle],
            velocity: velocity.to_vec(),
            angular_velocity: vec![spin],
            mass: 0.0,
            gravity_scale: 1.0,
        })
        .expect("a body");
    w.add_shape(Shape {
        body,
        kind: ShapeKind::Cuboid {
            half_extents: half.to_vec(),
        },
        position: vec![0.0; 2],
        rotation: vec![0.0],
        density: 1.0,
        friction: 0.0,
        restitution: 0.0,
    })
    .expect("a shape");
}

fn world() -> World {
    let options = Options {
        continuous_collision: false,
        ..Options::default()
    };
    World::with_options(2, Integrator::Solver, false, options).expect("a world")
}

/// A fixed wall 0.002 thick about x = 0, and a plate 0.002 thick and 1 tall at x = -0.4 moving
/// toward it at 24.87 u/s, face on: in the first step the plate passes through the wall and ends
/// with its near face 0.0125 beyond the wall's far face, within the prediction distance (0.02).
/// Two shapes this thin overlap at most 0.002 deep, within the allowed linear error.
fn through_a_thin_wall() -> World {
    let mut w = world();
    plate(&mut w, true, [0.001, 1.0], [0.0, 0.0], 0.0, [0.0; 2], 0.0);
    plate(
        &mut w,
        false,
        [0.001, 0.5],
        [-0.4, 0.0],
        0.0,
        [24.87, 0.0],
        0.0,
    );
    w
}

/// A bar 2 long and 0.04 thick about its centre at the origin, at -0.3 rad and turning 0.52 rad a
/// tick (31.2 rad/s), beside a fixed wall 1 thick whose near face is at x = `face`. The bar's end
/// starts 0.029 short of 0.99 and 0.037 short of 0.998, reaches x = 1 at angle 0 and turns back
/// to 0.0097 short of 0.99 and 0.0177 short of 0.998 by the step's end: within the prediction
/// distance, on the side it began, having gone 0.0102 into the wall at 0.99 (deeper than the
/// allowed linear error) and 0.0022 into the wall at 0.998 (within it).
fn swinging(face: f32) -> World {
    let mut w = world();
    plate(&mut w, false, [1.0, 0.02], [0.0, 0.0], -0.3, [0.0; 2], 31.2);
    plate(
        &mut w,
        true,
        [0.5, 2.0],
        [face + 0.5, 0.0],
        0.0,
        [0.0; 2],
        0.0,
    );
    w
}

/// The first row's findings of `w`, with `dropped` dropped from the rule.
fn found(mut w: World, dropped: Option<Condition>) -> Vec<[u32; 4]> {
    match &mut w.engine {
        Engine::Two(s) => s.drop_condition(dropped),
        Engine::Three(s) => s.drop_condition(dropped),
        Engine::Points(_) => unreachable!("a solver's world"),
    }
    w.advance_to(1).expect("the row");
    eprintln!(
        "late contact: {dropped:?} dropped: findings {:?} values {:?}",
        w.findings(),
        w.finding_values()
    );
    w.findings().to_vec()
}

/// `w` with the solver's allowed linear error and prediction distance set.
fn tolerances(mut w: World, allowed_linear_error: f32, prediction_distance: f32) -> World {
    match &mut w.engine {
        Engine::Two(s) => s.set_tolerances(allowed_linear_error, prediction_distance),
        Engine::Three(s) => s.set_tolerances(allowed_linear_error, prediction_distance),
        Engine::Points(_) => unreachable!("a solver's world"),
    }
    w
}

#[test]
fn a_body_through_a_thin_shape_ending_close_beyond_it_is_found_for_its_side() {
    // Where it ended: on the far side, within the prediction distance, not touching.
    let mut w = through_a_thin_wall();
    w.advance_to(1).expect("the row");
    let gap = w.pose(1, 2)[3] - 0.002;
    assert!(gap > 0.005 && gap <= 0.02, "{gap}");
    assert_eq!(found(through_a_thin_wall(), None), FOUND);
    assert_eq!(
        found(through_a_thin_wall(), Some(Condition::Shallow)),
        FOUND
    );
    assert_eq!(
        found(through_a_thin_wall(), Some(Condition::SameSide)),
        NONE,
        "only the side keeps it found"
    );
}

#[test]
fn a_corner_that_swings_deep_and_back_out_on_its_side_is_found_for_its_depth() {
    assert_eq!(found(swinging(0.99), None), FOUND);
    assert_eq!(found(swinging(0.99), Some(Condition::SameSide)), FOUND);
    assert_eq!(
        found(swinging(0.99), Some(Condition::Shallow)),
        NONE,
        "only the depth keeps it found"
    );
}

#[test]
fn a_shallow_contact_taken_a_step_late_is_not_found() {
    assert_eq!(found(swinging(0.998), None), NONE);
}

#[test]
fn the_rule_reads_the_solver_s_prediction_distance_and_allowed_linear_error() {
    let defaults = Settings::of(2);
    let (allowed, prediction) = (defaults.allowed_linear_error, defaults.prediction_distance);
    assert_eq!((allowed, prediction), (0.005, 0.02));
    let late = |allowed, prediction| found(tolerances(swinging(0.998), allowed, prediction), None);
    assert_eq!(late(allowed, prediction), NONE);
    // The bar's end stops 0.0177 short of the wall: beyond a prediction distance of 0.015.
    assert_eq!(late(allowed, 0.015), FOUND, "the prediction distance");
    // It went 0.0022 into the wall: deeper than an allowed linear error of 0.002.
    assert_eq!(late(0.002, prediction), FOUND, "the allowed linear error");
}
