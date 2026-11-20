//! The 3D solver world: `solver.rs` compiled with 3D types, and what differs between dimensions.

use rapier3d::parry::bounding_volume::{Aabb, BoundingVolume};
use rapier3d::parry::query;
use rapier3d::parry::shape::Shape as Geometry;
use rapier3d::prelude::*;

use crate::encode::Settings;

// The same source as the other dimension's, compiled with this one's types (lib.rs).
#[path = "solver.rs"]
mod solver;
pub(crate) use solver::Solver;

/// Numbers in a vector and in an angular quantity.
const DIM: usize = 3;
const ANGULAR: usize = 3;

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
    p.friction_model = if s.friction_model == 1 {
        FrictionModel::Coulomb
    } else {
        FrictionModel::Simplified
    };
    p.warmstart_joints = s.warmstart_joints;
    p.friction_in_bias_pass = s.friction_in_bias_pass;
}

fn vector(v: &[f32]) -> Vector {
    Vector::new(v[0], v[1], v[2])
}

fn angular(w: &[f32]) -> AngVector {
    Vector::new(w[0], w[1], w[2])
}

fn no_inertia() -> AngVector {
    Vector::ZERO
}

/// A pose from a position and a quaternion `(x, y, z, w)`, normalized (by a square root only).
fn pose(position: &[f32], rotation: &[f32]) -> Pose {
    let q = Rotation::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]).normalize();
    Pose::from_parts(vector(position), q)
}

fn cuboid(half_extents: &[f32]) -> ColliderBuilder {
    ColliderBuilder::cuboid(half_extents[0], half_extents[1], half_extents[2])
}

fn hull(points: &[f32]) -> Option<ColliderBuilder> {
    let points: Vec<Vector> = points
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| vector(p))
        .collect();
    ColliderBuilder::convex_hull(&points)
}

fn ball_joint() -> GenericJoint {
    GenericJointBuilder::new(JointAxesMask::LOCKED_SPHERICAL_AXES).build()
}

/// `x, y, z` and the unit quaternion `qx, qy, qz, qw`.
fn write_pose(body: &RigidBody, out: &mut Vec<f32>) {
    let (t, r) = (body.translation(), body.rotation());
    out.extend_from_slice(&[t.x, t.y, t.z, r.x, r.y, r.z, r.w]);
}

fn write_velocity(body: &RigidBody, out: &mut Vec<f32>) {
    let (v, w) = (body.linvel(), body.angvel());
    out.extend_from_slice(&[v.x, v.y, v.z, w.x, w.y, w.z]);
}

/// The turn from `from` to `to`, as a rotation vector: its axis times its angle, the angle in
/// [0, pi] (from the pure-Rust libm).
fn turn(from: Rotation, to: Rotation) -> AngVector {
    let q = to * from.inverse();
    let sign = if q.w < 0.0 { -1.0 } else { 1.0 };
    let axis = Vector::new(q.x, q.y, q.z) * sign;
    let half = axis.length();
    if half == 0.0 {
        return Vector::ZERO;
    }
    axis * (2.0 * libm::atan2f(half, q.w * sign) / half)
}

/// A turn's angle.
fn angle_of(turn: AngVector) -> f32 {
    turn.length()
}
