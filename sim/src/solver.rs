//! A world stepped by the solver, in either dimension: this file is compiled twice, as
//! `two::solver` and `three::solver`, each with its dimension's types and helpers in scope
//! (`super`). The crate's own work per tick is the force laws, evaluated from the bodies' state
//! before the step and applied as forces held through it.

use super::*;
use crate::encode::Settings;
use crate::spec::{Body, Force, Joint, JointKind, Shape, ShapeKind};

pub(crate) struct Solver {
    world: PhysicsWorld,
    bodies: Vec<RigidBodyHandle>,
    /// Each body's mass: its point mass plus its shapes', fixed bodies included (for attraction).
    mass: Vec<f32>,
    joints: Vec<ImpulseJointHandle>,
    forces: Vec<Force>,
    joint_iterations: usize,
}

impl Solver {
    pub fn new(settings: &Settings) -> Self {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        apply_settings(&mut world.integration_parameters, settings);
        Solver {
            world,
            bodies: Vec::new(),
            mass: Vec::new(),
            joints: Vec::new(),
            forces: Vec::new(),
            joint_iterations: settings.joint_solver_iterations as usize,
        }
    }

    pub fn add_body(&mut self, b: &Body) {
        let builder = if b.fixed {
            RigidBodyBuilder::fixed()
        } else {
            RigidBodyBuilder::dynamic()
        };
        let mut builder = builder
            .pose(pose(&b.position, &b.rotation))
            .linvel(vector(&b.velocity))
            .angvel(angular(&b.angular_velocity))
            .gravity_scale(b.gravity_scale)
            .can_sleep(false)
            .allow_fast_rotation(false);
        if b.mass > 0.0 {
            builder = builder.additional_mass_properties(MassProperties::new(
                Vector::ZERO,
                b.mass,
                no_inertia(),
            ));
        }
        self.bodies.push(self.world.insert_body(builder));
        self.mass.push(b.mass);
    }

    /// Fails when a convex hull's points span no area (2D) or volume (3D).
    pub fn add_shape(&mut self, s: &Shape) -> Result<(), String> {
        let builder = match &s.kind {
            ShapeKind::Ball { radius } => ColliderBuilder::ball(*radius),
            ShapeKind::Cuboid { half_extents } => cuboid(half_extents),
            ShapeKind::Capsule {
                half_height,
                radius,
            } => ColliderBuilder::capsule_y(*half_height, *radius),
            ShapeKind::ConvexHull { points } => hull(points)
                .ok_or("its points span no area in 2D, or no volume in 3D".to_string())?,
        };
        let builder = builder
            .position(pose(&s.position, &s.rotation))
            .density(s.density)
            .friction(s.friction)
            .restitution(s.restitution)
            .user_data(u128::from(s.body));
        let body = s.body as usize;
        let handle = self.world.insert_collider(builder, Some(self.bodies[body]));
        self.mass[body] += self.world.colliders[handle].mass();
        Ok(())
    }

    pub fn add_joint(&mut self, j: &Joint) {
        let (h1, h2) = (self.bodies[j.body1 as usize], self.bodies[j.body2 as usize]);
        let (a1, a2) = (vector(&j.anchor1), vector(&j.anchor2));
        let mut joint: GenericJoint = match j.kind {
            JointKind::Pin => pin(&j.axis).local_anchor1(a1).local_anchor2(a2).into(),
            JointKind::Slide => PrismaticJointBuilder::new(vector(&j.axis))
                .local_anchor1(a1)
                .local_anchor2(a2)
                .into(),
            JointKind::Weld => {
                // The relative rotation the two bodies have now, kept.
                let r1 = *self.world.bodies[h1].rotation();
                let r2 = *self.world.bodies[h2].rotation();
                FixedJointBuilder::new()
                    .local_frame1(Pose::from_parts(a1, r1.inverse() * r2))
                    .local_frame2(Pose::from_parts(a2, Rotation::IDENTITY))
                    .into()
            }
            JointKind::Ball => ball_joint(a1, a2),
            JointKind::Rope { length } => RopeJointBuilder::new(length)
                .local_anchor1(a1)
                .local_anchor2(a2)
                .into(),
            JointKind::Spring {
                rest,
                stiffness,
                damping,
            } => SpringJointBuilder::new(rest, stiffness, damping)
                .local_anchor1(a1)
                .local_anchor2(a2)
                .into(),
        };
        joint.set_contacts_enabled(j.contacts);
        if let Some(limits) = j.limits {
            joint.set_limits(free_axis(&j.kind), limits);
        }
        self.joints
            .push(self.world.insert_impulse_joint(h1, h2, joint));
        // From its first joint on, the world steps in finer substeps (Settings).
        self.world.integration_parameters.num_solver_iterations = self.joint_iterations;
    }

    pub fn add_force(&mut self, f: &Force, joint_kinds: &[JointKind]) {
        match f {
            Force::Gravity { acceleration } => self.world.gravity += vector(acceleration),
            Force::Motor {
                joint,
                position,
                velocity,
                stiffness,
                damping,
            } => {
                let axis = free_axis(&joint_kinds[*joint as usize]);
                let handle = self.joints[*joint as usize];
                if let Some(j) = self.world.impulse_joints.get_mut(handle, true) {
                    j.data
                        .set_motor(axis, *position, *velocity, *stiffness, *damping);
                }
            }
            _ => self.forces.push(f.clone()),
        }
    }

    pub fn mass(&self, body: usize) -> f32 {
        self.mass[body]
    }

    /// One tick: the force laws from the state before it, `applied` (an open world's forces and
    /// torques, `dim` and `angular` numbers per body), then the solver's step.
    pub fn step(&mut self, applied: Option<(&[f32], &[f32])>) {
        let bodies = &mut self.world.bodies;
        for &h in &self.bodies {
            bodies[h].reset_forces(false);
            bodies[h].reset_torques(false);
        }
        for law in &self.forces {
            match law {
                Force::Attraction {
                    strength,
                    bodies: those,
                } => {
                    for (k, &a) in those.iter().enumerate() {
                        for &b in &those[k + 1..] {
                            let (ha, hb) = (self.bodies[a as usize], self.bodies[b as usize]);
                            let r = bodies[hb].translation() - bodies[ha].translation();
                            let d2 = r.dot(r);
                            let s = strength * self.mass[a as usize] * self.mass[b as usize]
                                / (d2 * d2.sqrt());
                            bodies[ha].add_force(r * s, false);
                            bodies[hb].add_force(r * -s, false);
                        }
                    }
                }
                Force::Spring {
                    body_a,
                    anchor_a,
                    body_b,
                    anchor_b,
                    stiffness,
                    damping,
                    rest,
                } => {
                    let (ha, hb) = (self.bodies[*body_a as usize], self.bodies[*body_b as usize]);
                    let pa = bodies[ha].position().transform_point(vector(anchor_a));
                    let pb = bodies[hb].position().transform_point(vector(anchor_b));
                    let d = pb - pa;
                    let length = d.dot(d).sqrt();
                    let n = d / length;
                    let separating = (bodies[hb].velocity_at_point(pb)
                        - bodies[ha].velocity_at_point(pa))
                    .dot(n);
                    let f = n * (stiffness * (length - rest) + damping * separating);
                    bodies[ha].add_force_at_point(f, pa, false);
                    bodies[hb].add_force_at_point(-f, pb, false);
                }
                Force::LinearDrag { body, coefficient } => {
                    let b = &mut bodies[self.bodies[*body as usize]];
                    let v = b.linvel();
                    b.add_force(v * -coefficient, false);
                }
                Force::QuadraticDrag { body, coefficient } => {
                    let b = &mut bodies[self.bodies[*body as usize]];
                    let v = b.linvel();
                    b.add_force(v * (-coefficient * v.dot(v).sqrt()), false);
                }
                Force::Constant {
                    body,
                    force,
                    torque,
                } => {
                    let b = &mut bodies[self.bodies[*body as usize]];
                    b.add_force(vector(force), false);
                    b.add_torque(angular(torque), false);
                }
                Force::Gravity { .. } | Force::Motor { .. } => {}
            }
        }
        if let Some((forces, torques)) = applied {
            let (fw, tw) = (DIM, ANGULAR);
            for (i, &h) in self.bodies.iter().enumerate() {
                bodies[h].add_force(vector(&forces[i * fw..(i + 1) * fw]), false);
                bodies[h].add_torque(angular(&torques[i * tw..(i + 1) * tw]), false);
            }
        }
        self.world.step();
    }

    /// Each body's pose and velocity now, and the pairs of bodies (by number, the smaller first,
    /// sorted) whose shapes are in contact.
    pub fn record(&self, pose: &mut Vec<f32>, velocity: &mut Vec<f32>) -> Vec<(u32, u32)> {
        for &h in &self.bodies {
            let body = &self.world.bodies[h];
            write_pose(body, pose);
            write_velocity(body, velocity);
        }
        let mut pairs: Vec<(u32, u32)> = self
            .world
            .contact_pairs()
            .filter(|p| p.has_any_active_contact())
            .map(|p| {
                let a = self.world.colliders[p.collider1].user_data as u32;
                let b = self.world.colliders[p.collider2].user_data as u32;
                (a.min(b), a.max(b))
            })
            .filter(|(a, b)| a != b)
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
}

/// The joint's one free axis, which a limit or a motor acts on.
fn free_axis(kind: &JointKind) -> JointAxis {
    match kind {
        JointKind::Slide => JointAxis::LinX,
        _ => JointAxis::AngX,
    }
}
