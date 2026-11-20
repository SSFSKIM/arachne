//! The 3D solver world: `solver.rs` compiled with 3D types, and what differs between dimensions.

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

fn pin(axis: &[f32]) -> RevoluteJointBuilder {
    RevoluteJointBuilder::new(vector(axis))
}

fn ball_joint(anchor1: Vector, anchor2: Vector) -> GenericJoint {
    SphericalJointBuilder::new()
        .local_anchor1(anchor1)
        .local_anchor2(anchor2)
        .into()
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
