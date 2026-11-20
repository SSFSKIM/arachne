//! The 2D solver world: `solver.rs` compiled with 2D types, and what differs between dimensions.

use rapier2d::parry::bounding_volume::{Aabb, BoundingVolume};
use rapier2d::parry::query;
use rapier2d::parry::shape::Shape as Geometry;
use rapier2d::prelude::*;

use crate::encode::Settings;

// The same source as the other dimension's, compiled with this one's types (lib.rs).
#[path = "solver.rs"]
mod solver;
pub(crate) use solver::Solver;

/// Numbers in a vector and in an angular quantity.
const DIM: usize = 2;
const ANGULAR: usize = 1;

fn apply_settings(p: &mut IntegrationParameters, s: &Settings) {
    p.dt = s.dt;
    p.length_unit = s.length_unit;
    p.num_solver_iterations = s.solver_iterations as usize;
    p.num_internal_pgs_iterations = s.pgs_iterations as usize;
    p.num_internal_stabilization_iterations = s.stabilization_iterations as usize;
    p.max_ccd_substeps = s.ccd_substeps as usize;
    p.normalized_max_linear_velocity = s.max_linear_velocity;
    p.normalized_allowed_linear_error = s.allowed_linear_error;
    p.normalized_prediction_distance = s.prediction_distance;
    p.contact_recycling = s.contact_recycling;
    p.normalized_contact_recycle_distance = s.recycle_distance;
    p.contact_clustering = s.contact_clustering;
    p.warmstart_joints = s.warmstart_joints;
    p.friction_in_bias_pass = s.friction_in_bias_pass;
}

fn vector(v: &[f32]) -> Vector {
    Vector::new(v[0], v[1])
}

fn angular(w: &[f32]) -> AngVector {
    w[0]
}

fn no_inertia() -> AngVector {
    0.0
}

/// A pose from a position and an angle (its sine and cosine from the pure-Rust libm, through the
/// solver's own trait maths).
fn pose(position: &[f32], rotation: &[f32]) -> Pose {
    Pose::new(vector(position), rotation[0])
}

fn cuboid(half_extents: &[f32]) -> ColliderBuilder {
    ColliderBuilder::cuboid(half_extents[0], half_extents[1])
}

fn hull(points: &[f32]) -> Option<ColliderBuilder> {
    let points: Vec<Vector> = points
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| vector(p))
        .collect();
    ColliderBuilder::convex_hull(&points)
}

fn ball_joint() -> GenericJoint {
    unreachable!("a ball joint is refused in 2D before it reaches the solver")
}

/// `x, y, angle`: the angle in (-pi, pi], from the solver's unit complex number.
fn write_pose(body: &RigidBody, out: &mut Vec<f32>) {
    let (t, r) = (body.translation(), body.rotation());
    out.extend_from_slice(&[t.x, t.y, libm::atan2f(r.im, r.re)]);
}

fn write_velocity(body: &RigidBody, out: &mut Vec<f32>) {
    let v = body.linvel();
    out.extend_from_slice(&[v.x, v.y, body.angvel()]);
}

/// The turn from `from` to `to`: its angle, in (-pi, pi] (from the pure-Rust libm).
fn turn(from: Rotation, to: Rotation) -> AngVector {
    let r = to * from.inverse();
    libm::atan2f(r.im, r.re)
}

/// A turn's angle.
fn angle_of(turn: AngVector) -> f32 {
    turn.abs()
}
