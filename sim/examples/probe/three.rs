//! The probe's 3D world, from literal numbers only: bodies of every shape kind in contact, with
//! friction and restitution (among them a convex hull and a compound whose inertia is not
//! diagonal); a chain of pin and ball joints with and without limits and motors; slide, weld,
//! rope and spring joints; and every force law.

use rapier3d::prelude::*;

use crate::trace::{Trace, contact_rows};

const ATTRACTION: f32 = 2.0;
const QUADRATIC_DRAG: f32 = 0.05;

pub struct World {
    world: PhysicsWorld,
    bodies: Vec<RigidBodyHandle>,
    attracting: (RigidBodyHandle, RigidBodyHandle),
    dragged: RigidBodyHandle,
    pushed: RigidBodyHandle,
    pub trace: Trace,
}

/// The solver's settings, each set here rather than left to its default.
pub fn settings(p: &mut IntegrationParameters) {
    p.dt = 1.0 / 60.0;
    p.length_unit = 1.0;
    p.num_solver_iterations = 4;
    p.num_internal_pgs_iterations = 1;
    p.num_internal_stabilization_iterations = 1;
    p.max_ccd_substeps = 1;
    p.normalized_max_linear_velocity = 400.0;
    p.contact_recycling = true;
    p.contact_clustering = true;
    p.warmstart_joints = false;
    p.friction_in_bias_pass = false;
}

/// A unit quaternion from literal components (normalized here, by a square root only).
pub fn rotation(x: f32, y: f32, z: f32, w: f32) -> Rotation {
    Rotation::from_xyzw(x, y, z, w).normalize()
}

pub fn dynamic(x: f32, y: f32, z: f32) -> RigidBodyBuilder {
    RigidBodyBuilder::dynamic()
        .translation(Vector::new(x, y, z))
        .can_sleep(false)
        .allow_fast_rotation(false)
}

/// An asymmetric convex hull: its inertia tensor has off-diagonal terms.
pub const HULL: [[f32; 3]; 9] = [
    [-0.4, -0.3, -0.2],
    [0.5, -0.2, -0.3],
    [0.3, 0.4, -0.1],
    [-0.2, 0.5, 0.2],
    [0.1, -0.35, 0.4],
    [0.45, 0.25, 0.35],
    [-0.35, 0.1, 0.45],
    [0.0, 0.6, -0.05],
    [-0.1, -0.4, -0.45],
];

pub fn hull() -> ColliderBuilder {
    let points: Vec<Vector> = HULL.iter().map(|&p| Vector::from_array(p)).collect();
    ColliderBuilder::convex_hull(&points).expect("a convex hull")
}

/// A ball and a cuboid turned about an oblique axis: its inertia tensor is not diagonal.
pub fn compound() -> ColliderBuilder {
    ColliderBuilder::compound(vec![
        (Pose::translation(-0.3, 0.0, 0.0), SharedShape::ball(0.2)),
        (
            Pose::from_parts(Vector::new(0.3, 0.1, 0.0), rotation(0.2, 0.3, 0.1, 0.92)),
            SharedShape::cuboid(0.2, 0.1, 0.15),
        ),
    ])
}

fn add(
    world: &mut PhysicsWorld,
    bodies: &mut Vec<RigidBodyHandle>,
    body: RigidBodyBuilder,
    colliders: Vec<ColliderBuilder>,
) -> RigidBodyHandle {
    let handle = world.insert_body(body);
    for collider in colliders {
        world.insert_collider(collider, Some(handle));
    }
    bodies.push(handle);
    handle
}

impl World {
    pub fn new() -> Self {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::new(0.0, -9.8, 0.0);
        settings(&mut world.integration_parameters);
        let w = &mut world;
        let b = &mut Vec::new();

        let fixed =
            |x: f32, y: f32, z: f32| RigidBodyBuilder::fixed().translation(Vector::new(x, y, z));
        add(
            w,
            b,
            fixed(0.0, -0.5, 0.0),
            vec![ColliderBuilder::cuboid(8.0, 0.5, 8.0).friction(0.6)],
        );
        add(
            w,
            b,
            dynamic(-2.0, 3.0, 0.3).linvel(Vector::new(1.0, 0.0, -0.5)),
            vec![ColliderBuilder::ball(0.3).restitution(0.6).friction(0.4)],
        );
        let boxed = add(
            w,
            b,
            RigidBodyBuilder::dynamic()
                .pose(Pose::from_parts(
                    Vector::new(-1.0, 1.0, 0.0),
                    rotation(0.0, 0.0, 0.15, 0.99),
                ))
                .can_sleep(false)
                .allow_fast_rotation(false),
            vec![ColliderBuilder::cuboid(0.4, 0.25, 0.3).friction(0.7)],
        );
        add(
            w,
            b,
            dynamic(0.0, 2.0, -0.4)
                .rotation(Vector::new(0.2, 0.5, -0.1))
                .angvel(Vector::new(1.0, 2.0, 0.5)),
            vec![ColliderBuilder::capsule_y(0.3, 0.15)],
        );
        add(w, b, dynamic(1.0, 2.5, 0.2), vec![hull().friction(0.5)]);
        add(w, b, dynamic(2.0, 3.5, -0.2), vec![compound()]);

        // A chain hanging from a fixed anchor: pin joints free, limited, driven at a speed and
        // driven to an angle; then ball joints, limited and driven, and free.
        let anchor = add(w, b, fixed(-4.0, 5.0, 0.0), vec![]);
        let mut links = Vec::new();
        for x in [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5] {
            links.push(add(
                w,
                b,
                dynamic(x, 5.0, 0.0),
                vec![ColliderBuilder::capsule_x(0.4, 0.08)],
            ));
        }
        let ends = |first: bool| {
            let anchor1 = if first {
                Vector::ZERO
            } else {
                Vector::new(0.5, 0.0, 0.0)
            };
            (anchor1, Vector::new(-0.5, 0.0, 0.0))
        };
        let pin = |first: bool| {
            let (a1, a2) = ends(first);
            RevoluteJointBuilder::new(Vector::Z)
                .local_anchor1(a1)
                .local_anchor2(a2)
                .contacts_enabled(false)
        };
        w.insert_impulse_joint(anchor, links[0], pin(true));
        w.insert_impulse_joint(links[0], links[1], pin(false).limits([-0.8, 0.6]));
        w.insert_impulse_joint(
            links[1],
            links[2],
            pin(false).motor_velocity(1.5, 0.5).limits([-1.0, 1.0]),
        );
        w.insert_impulse_joint(
            links[2],
            links[3],
            pin(false).motor_position(0.4, 15.0, 1.5),
        );
        let (a1, a2) = ends(false);
        w.insert_impulse_joint(
            links[3],
            links[4],
            SphericalJointBuilder::new()
                .local_anchor1(a1)
                .local_anchor2(a2)
                .limits(JointAxis::AngX, [-0.5, 0.5])
                .motor_velocity(JointAxis::AngY, 1.0, 0.5)
                .contacts_enabled(false),
        );
        w.insert_impulse_joint(
            links[4],
            links[5],
            SphericalJointBuilder::new()
                .local_anchor1(a1)
                .local_anchor2(a2)
                .contacts_enabled(false),
        );

        // Slide joints, with a limit and a motor and without; a weld; a rope; a spring.
        let slider = add(
            w,
            b,
            dynamic(-4.0, 3.0, 0.0),
            vec![ColliderBuilder::cuboid(0.2, 0.2, 0.2)],
        );
        w.insert_impulse_joint(
            anchor,
            slider,
            PrismaticJointBuilder::new(Vector::X)
                .local_anchor1(Vector::new(0.0, -2.0, 0.0))
                .limits([-1.0, 1.0])
                .motor_position(0.5, 20.0, 2.0),
        );
        let glider = add(
            w,
            b,
            dynamic(-4.0, 2.0, 0.0).linvel(Vector::new(0.5, 0.0, 0.0)),
            vec![ColliderBuilder::cuboid(0.15, 0.1, 0.15)],
        );
        w.insert_impulse_joint(
            anchor,
            glider,
            PrismaticJointBuilder::new(Vector::X).local_anchor1(Vector::new(0.0, -3.0, 0.0)),
        );
        let welded = add(
            w,
            b,
            dynamic(-1.0, 1.4, 0.0),
            vec![ColliderBuilder::ball(0.1)],
        );
        w.insert_impulse_joint(
            boxed,
            welded,
            FixedJointBuilder::new().local_anchor1(Vector::new(0.0, 0.4, 0.0)),
        );
        let roped = add(
            w,
            b,
            dynamic(-4.5, 4.0, 0.3),
            vec![ColliderBuilder::ball(0.1)],
        );
        w.insert_impulse_joint(anchor, roped, RopeJointBuilder::new(1.2));
        let sprung = add(
            w,
            b,
            dynamic(-4.0, 6.5, 0.2),
            vec![ColliderBuilder::ball(0.1)],
        );
        w.insert_impulse_joint(anchor, sprung, SpringJointBuilder::new(1.0, 30.0, 0.8));

        // Two bodies attracting each other by the inverse square, out of gravity.
        let ball = || vec![ColliderBuilder::ball(0.15).density(2.0)];
        let one = add(
            w,
            b,
            dynamic(4.0, 6.0, 0.0)
                .gravity_scale(0.0)
                .linvel(Vector::new(0.0, 1.0, 0.2)),
            ball(),
        );
        let two = add(
            w,
            b,
            dynamic(5.0, 6.0, 0.0)
                .gravity_scale(0.0)
                .linvel(Vector::new(0.0, -1.0, -0.2)),
            ball(),
        );
        let dragged = add(
            w,
            b,
            dynamic(5.0, 1.0, 0.0)
                .linvel(Vector::new(-3.0, 4.0, 1.0))
                .linear_damping(0.2)
                .angular_damping(0.1),
            vec![ColliderBuilder::ball(0.2)],
        );
        let pushed = add(
            w,
            b,
            dynamic(6.0, 0.3, 0.0),
            vec![ColliderBuilder::cuboid(0.25, 0.25, 0.25)],
        );

        let bodies = std::mem::take(b);
        let mut this = Self {
            world,
            bodies,
            attracting: (one, two),
            dragged,
            pushed,
            trace: Trace::default(),
        };
        this.record();
        this
    }

    /// Steps `ticks` times, recording each new state.
    pub fn advance(&mut self, ticks: usize) {
        for _ in 0..ticks {
            self.apply_forces();
            self.world.step();
            self.record();
        }
    }

    /// The number of colliding pairs, read between steps.
    pub fn contacts(&self) -> usize {
        self.world
            .contact_pairs()
            .filter(|p| p.has_any_active_contact())
            .count()
    }

    fn apply_forces(&mut self) {
        let bodies = &mut self.world.bodies;
        let (a, b) = self.attracting;
        let r = bodies[b].translation() - bodies[a].translation();
        let d2 = r.dot(r);
        let f = r * (ATTRACTION * bodies[a].mass() * bodies[b].mass() / (d2 * d2.sqrt()));
        for (h, force) in [(a, f), (b, -f)] {
            bodies[h].reset_forces(true);
            bodies[h].add_force(force, true);
        }
        let dragged = &mut bodies[self.dragged];
        let v = dragged.linvel();
        dragged.reset_forces(true);
        dragged.add_force(v * (-QUADRATIC_DRAG * v.dot(v).sqrt()), true);
        let pushed = &mut bodies[self.pushed];
        pushed.reset_forces(true);
        pushed.reset_torques(true);
        pushed.add_force(Vector::new(-1.5, 0.0, 0.5), true);
        pushed.add_torque(Vector::new(0.0, 0.3, 0.1), true);
    }

    fn position(&self, collider: ColliderHandle) -> u32 {
        let body = self.world.colliders[collider]
            .parent()
            .expect("a collider on a body");
        self.bodies
            .iter()
            .position(|&h| h == body)
            .expect("a body of this world") as u32
    }

    fn record(&mut self) {
        for &h in &self.bodies {
            let body = &self.world.bodies[h];
            let (t, r, v, w) = (
                body.translation(),
                body.rotation(),
                body.linvel(),
                body.angvel(),
            );
            self.trace.f32s(&t.to_array());
            self.trace.f32s(&r.to_array());
            self.trace.f32s(&v.to_array());
            self.trace.f32s(&w.to_array());
        }
        let pairs = self
            .world
            .contact_pairs()
            .filter(|p| p.has_any_active_contact())
            .map(|p| (self.position(p.collider1), self.position(p.collider2)))
            .collect();
        let rows = contact_rows(pairs);
        self.trace.u32(rows.len() as u32);
        for (a, b) in rows {
            self.trace.u32(a);
            self.trace.u32(b);
        }
    }
}
