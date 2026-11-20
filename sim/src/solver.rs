//! A world stepped by the solver, in either dimension: this file is compiled twice, as
//! `two::solver` and `three::solver`, each with its dimension's types and helpers in scope
//! (`super`). The crate's own work per tick is the force laws, evaluated from the bodies' state
//! before the step and applied as forces held through it, and the step's checks: which bodies
//! the solver stopped for a non-finite state, and which passed through each other.

use super::*;
use crate::encode::Settings;
use crate::spec::{Body, Force, Joint, JointKind, Shape, ShapeKind};
use crate::world::Recorded;

pub(crate) struct Solver {
    world: PhysicsWorld,
    bodies: Vec<RigidBodyHandle>,
    /// Each body's mass: its point mass plus its shapes', fixed bodies included (for attraction).
    mass: Vec<f32>,
    fixed: Vec<bool>,
    /// Each shape and the body it is on, in declaration order.
    shapes: Vec<(ColliderHandle, u32)>,
    joints: Vec<ImpulseJointHandle>,
    /// The pairs of bodies (the smaller first) a joint keeps from colliding.
    apart: Vec<(u32, u32)>,
    forces: Vec<Force>,
    /// The substeps a jointed component takes beyond the world's (Settings::jointed_substeps).
    jointed_extra: usize,
    /// The pairs of shapes (by number, the smaller first, sorted) touching at the last row.
    touching: Vec<(u32, u32)>,
    /// Each body's pose as the step being taken began, until its row is recorded.
    before: Option<Vec<Pose>>,
    /// Each shape's reach: the radius of a sphere about its body's centre of mass that holds
    /// the shape however the body turns. Computed at the first step, once every shape is on.
    reach: Vec<f32>,
}

/// A body's motion over a step, as the step's checks sweep it: its centre of mass from `from`
/// to `to` along a line, and its turn `turn` about it (an angle in 2D, a rotation vector in 3D)
/// at a constant rate.
struct Motion {
    local: Vector,
    from: Vector,
    to: Vector,
    turn: AngVector,
    /// The turn's angle.
    angle: f32,
    /// Whether the solver still moves the body (not stopped for a non-finite state).
    enabled: bool,
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
            fixed: Vec::new(),
            shapes: Vec::new(),
            joints: Vec::new(),
            apart: Vec::new(),
            forces: Vec::new(),
            jointed_extra: settings
                .jointed_substeps
                .saturating_sub(settings.solver_iterations) as usize,
            touching: Vec::new(),
            before: None,
            reach: Vec::new(),
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
        self.fixed.push(b.fixed);
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
        self.shapes.push((handle, s.body));
        Ok(())
    }

    pub fn add_joint(&mut self, j: &Joint) {
        let (h1, h2) = (self.bodies[j.body1 as usize], self.bodies[j.body2 as usize]);
        let mut joint: GenericJoint = match j.kind {
            JointKind::Pin => GenericJointBuilder::new(JointAxesMask::LOCKED_REVOLUTE_AXES).build(),
            JointKind::Slide => {
                GenericJointBuilder::new(JointAxesMask::LOCKED_PRISMATIC_AXES).build()
            }
            JointKind::Weld => GenericJointBuilder::new(JointAxesMask::LOCKED_FIXED_AXES).build(),
            JointKind::Ball => ball_joint(),
            JointKind::Rope { length } => RopeJointBuilder::new(length).into(),
            JointKind::Spring {
                rest,
                stiffness,
                damping,
            } => SpringJointBuilder::new(rest, stiffness, damping).into(),
        };
        // One frame in the world, at the first anchor with its x axis along the joint's axis (in
        // the first body's frame, as declared), seen from each body as the bodies are now: the
        // frames' axes match, so a pin's angle is zero as declared.
        let axis = if j.axis.is_empty() {
            Rotation::IDENTITY
        } else {
            let a = vector(&j.axis);
            GenericJoint::complete_ang_frame(a / a.dot(a).sqrt())
        };
        let (p1, p2) = (
            *self.world.bodies[h1].position(),
            *self.world.bodies[h2].position(),
        );
        let (r1, r2) = (p1.rotation, p2.rotation);
        let mut anchor2 = vector(&j.anchor2);
        if j.kind == JointKind::Slide {
            // A slide's offset is the anchors' separation along the axis, zero as declared: the
            // second anchor moves along the axis (its free direction, which no lock reads) by
            // the separation the declared anchors have now.
            let along = r1 * (axis * Vector::X);
            let apart =
                (p2.transform_point(anchor2) - p1.transform_point(vector(&j.anchor1))).dot(along);
            anchor2 -= r2.inverse() * (along * apart);
        }
        joint.set_local_frame1(Pose::from_parts(vector(&j.anchor1), axis));
        joint.set_local_frame2(Pose::from_parts(anchor2, r2.inverse() * r1 * axis));
        joint.set_contacts_enabled(j.contacts);
        if let Some(limits) = j.limits {
            joint.set_limits(free_axis(&j.kind), limits);
        }
        self.joints
            .push(self.world.insert_impulse_joint(h1, h2, joint));
        if !j.contacts {
            let pair = (j.body1.min(j.body2), j.body1.max(j.body2));
            self.apart.push(pair);
        }
        // Each dynamic body it joins asks for the jointed substeps, which its whole connected
        // component then takes (Settings::jointed_substeps).
        for h in [h1, h2] {
            let body = &mut self.world.bodies[h];
            if body.is_dynamic() {
                body.set_additional_solver_iterations(self.jointed_extra);
            }
        }
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
                    // Force coefficients (Settings::motor_model), not the solver's default,
                    // which scales them by mass.
                    j.data
                        .set_motor_model(axis, MotorModel::ForceBased)
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
        self.before = Some(
            self.bodies
                .iter()
                .map(|&h| *self.world.bodies[h].position())
                .collect(),
        );
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
                    let dv = bodies[hb].velocity_at_point(pb) - bodies[ha].velocity_at_point(pa);
                    let Some(f) = spring_force(d, dv, *stiffness, *damping, *rest) else {
                        continue;
                    };
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

    /// Each body's pose and velocity now; the pairs of bodies (by number, the smaller first,
    /// sorted) that touch now (`touching`); and, after a step, the bodies the solver stopped in
    /// it for a non-finite state and the pairs that passed through each other in it
    /// (`tunnelled`).
    pub fn record(&mut self, pose: &mut Vec<f32>, velocity: &mut Vec<f32>) -> Recorded {
        for &h in &self.bodies {
            let body = &self.world.bodies[h];
            write_pose(body, pose);
            write_velocity(body, velocity);
        }
        let shapes = self.touching();
        let mut recorded = Recorded::default();
        if let Some(before) = self.before.take() {
            recorded.quarantined = self.quarantined();
            recorded.tunnelled = self.tunnelled(&before, &shapes);
        }
        recorded.touching = shapes
            .iter()
            .map(|&(i, k)| {
                let (a, b) = (self.shapes[i as usize].1, self.shapes[k as usize].1);
                (a.min(b), a.max(b))
            })
            .collect();
        recorded.touching.sort_unstable();
        recorded.touching.dedup();
        self.touching = shapes;
        recorded
    }

    /// The bodies (by number, sorted) the solver stopped in the last step for a non-finite
    /// state: its own, or one of its shapes'.
    fn quarantined(&self) -> Vec<u32> {
        let quarantine = self.world.quarantine();
        let mut found: Vec<u32> = quarantine
            .bodies()
            .iter()
            .filter_map(|h| self.bodies.iter().position(|b| b == h))
            .map(|i| i as u32)
            .chain(quarantine.colliders().iter().filter_map(|h| {
                self.shapes
                    .iter()
                    .find(|(shape, _)| shape == h)
                    .map(|&(_, body)| body)
            }))
            .collect();
        found.sort_unstable();
        found.dedup();
        found
    }

    /// The pairs of bodies (by number, the smaller first, sorted) two of whose shapes passed
    /// through each other in the last step: shapes the solver lets collide, that touched at
    /// neither end of the step (`now`, and the last row's), with no contact between them in the
    /// solver's step, and that met when swept through the step's motion (`Motion`).
    ///
    /// Only candidates are swept, and each test that rules a pair out is one no pair that meets
    /// can fail: a point of a shape is never farther from where it began than its body's
    /// centre moved plus its turn's angle times the shape's reach. So two shapes meet only if
    /// their boxes at the start, each grown by that much, overlap; if the gap between their
    /// spheres at the start is under the relative travel of the centres plus each turn's angle
    /// times the reach; and if the distance between the shapes at the start is under the same.
    fn tunnelled(&mut self, before: &[Pose], now: &[(u32, u32)]) -> Vec<(u32, u32)> {
        self.measure_reach();
        let motions: Vec<Motion> = self
            .bodies
            .iter()
            .zip(before)
            .map(|(&h, from)| {
                let body = &self.world.bodies[h];
                let local = body.local_center_of_mass();
                let to = *body.position();
                let turn = turn(from.rotation, to.rotation);
                Motion {
                    local,
                    from: from.transform_point(local),
                    to: to.transform_point(local),
                    turn,
                    angle: angle_of(turn),
                    enabled: body.is_enabled(),
                }
            })
            .collect();
        // Each shape: its pose at the start, its box there grown by how far any of its points
        // may move in the step, and how far that is.
        let swept: Vec<(Pose, Aabb, f32)> = self
            .shapes
            .iter()
            .enumerate()
            .map(|(s, &(handle, body))| {
                let m = &motions[body as usize];
                let pose = before[body as usize] * self.local_pose(handle);
                let moves = (m.to - m.from).length() + m.angle * self.reach[s];
                let aabb = self.world.colliders[handle]
                    .shape()
                    .compute_aabb(&pose)
                    .loosened(moves);
                (pose, aabb, moves)
            })
            .collect();
        let mut found = Vec::new();
        for (i, &(handle1, a)) in self.shapes.iter().enumerate() {
            for (k, &(handle2, b)) in self.shapes.iter().enumerate().skip(i + 1) {
                let (ma, mb) = (&motions[a as usize], &motions[b as usize]);
                let pair = (a.min(b), a.max(b));
                if a == b
                    || (self.fixed[a as usize] && self.fixed[b as usize])
                    || !ma.enabled
                    || !mb.enabled
                    || !swept[i].1.intersects(&swept[k].1)
                    || self.apart.contains(&pair)
                {
                    continue;
                }
                let travel = ((ma.to - ma.from) - (mb.to - mb.from)).length();
                let turning = ma.angle * self.reach[i] + mb.angle * self.reach[k];
                if travel == 0.0 && turning == 0.0 {
                    continue;
                }
                let gap = (mb.from - ma.from).length() - self.reach[i] - self.reach[k];
                let shapes = (i as u32, k as u32);
                if gap >= travel + turning
                    || self.touching.binary_search(&shapes).is_ok()
                    || now.binary_search(&shapes).is_ok()
                {
                    continue;
                }
                let (shape1, shape2) = (
                    self.world.colliders[handle1].shape(),
                    self.world.colliders[handle2].shape(),
                );
                let apart = query::distance(&swept[i].0, shape1, &swept[k].0, shape2);
                if matches!(apart, Ok(d) if d.distance >= travel + turning) {
                    continue;
                }
                let contact = self
                    .world
                    .narrow_phase
                    .contact_pair(handle1, handle2)
                    .is_some_and(|p| p.has_any_active_contact());
                if contact {
                    continue;
                }
                let motion = |s: usize, m: &Motion| {
                    let start = swept[s].0;
                    let local = self.local_pose(self.shapes[s].0);
                    query::NonlinearRigidMotion::new(
                        start,
                        local.inverse_transform_point(m.local),
                        m.to - m.from,
                        m.turn,
                    )
                };
                let met = query::cast_shapes_nonlinear(
                    &motion(i, ma),
                    shape1,
                    &motion(k, mb),
                    shape2,
                    0.0,
                    1.0,
                    true,
                );
                if matches!(met, Ok(Some(_))) {
                    found.push(pair);
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }

    /// A shape's pose in its body's frame.
    fn local_pose(&self, handle: ColliderHandle) -> Pose {
        *self.world.colliders[handle]
            .position_wrt_parent()
            .expect("a shape is on a body")
    }

    /// Each shape's reach (`reach`), once every shape is on its body.
    fn measure_reach(&mut self) {
        if self.reach.len() == self.shapes.len() {
            return;
        }
        self.reach = self
            .shapes
            .iter()
            .map(|&(handle, body)| {
                let sphere = self.world.colliders[handle]
                    .shape()
                    .compute_local_bounding_sphere();
                let centre = self.local_pose(handle).transform_point(sphere.center);
                let com = self.world.bodies[self.bodies[body as usize]].local_center_of_mass();
                (centre - com).length() + sphere.radius
            })
            .collect();
    }

    /// The pairs of shapes (by number, the smaller first, sorted) that are, at their poses now,
    /// at a distance at most the allowed linear error (touching or overlapping;
    /// Settings::allowed_linear_error), found by the solver's geometric query rather than read
    /// from its contact pairs: those include pairs within the prediction distance, and pairs
    /// whose points were computed before the step's motion. Pairs the solver never lets collide
    /// are left out: two shapes of one body, two of fixed bodies, and two of bodies a joint keeps
    /// from colliding. Two bodies touch when a pair of their shapes does.
    fn touching(&self) -> Vec<(u32, u32)> {
        let reach = self.world.integration_parameters.allowed_linear_error();
        let placed: Vec<(Pose, Aabb, u32)> = self
            .shapes
            .iter()
            .map(|&(handle, body)| {
                let shape = &self.world.colliders[handle];
                let on = self.world.bodies[self.bodies[body as usize]].position();
                let pose = *on * *shape.position_wrt_parent().expect("a shape is on a body");
                let aabb = shape.shape().compute_aabb(&pose).loosened(reach);
                (pose, aabb, body)
            })
            .collect();
        let mut pairs = Vec::new();
        for (i, (pose1, aabb1, a)) in placed.iter().enumerate() {
            for (k, (pose2, aabb2, b)) in placed.iter().enumerate().skip(i + 1) {
                if a == b
                    || (self.fixed[*a as usize] && self.fixed[*b as usize])
                    || !aabb1.intersects(aabb2)
                {
                    continue;
                }
                let pair = ((*a).min(*b), (*a).max(*b));
                if self.apart.contains(&pair) {
                    continue;
                }
                let shape1 = self.world.colliders[self.shapes[i].0].shape();
                let shape2 = self.world.colliders[self.shapes[k].0].shape();
                // Zero when they overlap.
                let distance = query::distance(pose1, shape1, pose2, shape2);
                if matches!(distance, Ok(d) if d.distance <= reach) {
                    pairs.push((i as u32, k as u32));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
}

/// A spring's force on its first anchor, from the second anchor's offset `d` and relative
/// velocity `dv`, along the line between the anchors: an elastic term `k d` at rest length 0
/// (no direction needed) and `k (|d| - rest) n` otherwise, `n` the line's direction, and a
/// damping term `c (dv . n) n`. None (`None`) for a tick at which the anchors coincide and the
/// line is undefined; at rest length 0 the elastic term is zero there too.
fn spring_force(d: Vector, dv: Vector, k: f32, c: f32, rest: f32) -> Option<Vector> {
    let length = d.dot(d).sqrt();
    if length == 0.0 {
        return None;
    }
    let n = d / length;
    let elastic = if rest == 0.0 {
        d * k
    } else {
        n * (k * (length - rest))
    };
    Some(elastic + n * (c * dv.dot(n)))
}

/// The joint's one free axis, which a limit or a motor acts on.
fn free_axis(kind: &JointKind) -> JointAxis {
    match kind {
        JointKind::Slide => JointAxis::LinX,
        _ => JointAxis::AngX,
    }
}
