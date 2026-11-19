//! One case per audited path, each printing its result as the bits of its numbers. The cases
//! named `published` rebuild, with the same literal inputs, the cases of a published set of
//! determinism reproducers for rapier 0.36.1 (README.md, "Determinism").

pub fn all() -> Vec<(&'static str, String)> {
    vec![
        (
            "angular-motor-2d-published",
            hex(&two::angular_motor_published()),
        ),
        (
            "angular-limits-2d-published",
            hex(&two::angular_limits_published()),
        ),
        (
            "contact-coefficients-2d-published",
            hex(&two::contact_coefficients_published()),
        ),
        (
            "revolute-angle-3d-published",
            hex(&three::revolute_angle_published()),
        ),
        ("angular-limits-3d", hex(&three::angular_limits())),
        (
            "inertia-compound-3d",
            hex(&three::inertia(crate::three::compound())),
        ),
        (
            "inertia-hull-3d",
            hex(&three::inertia(crate::three::hull())),
        ),
        ("hull-at-rest-3d", hex(&three::hull_at_rest())),
    ]
}

fn hex(bits: &[u32]) -> String {
    bits.iter()
        .map(|b| format!("{b:08x}"))
        .collect::<Vec<_>>()
        .join(".")
}

mod two {
    use rapier2d::prelude::*;

    fn still_world() -> (PhysicsWorld, RigidBodyHandle, RigidBodyHandle) {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        let fixed = world.insert_body(RigidBodyBuilder::fixed());
        let body = world.insert_body(RigidBodyBuilder::dynamic().can_sleep(false));
        world.insert_collider(ColliderBuilder::ball(1.0), Some(body));
        (world, fixed, body)
    }

    /// A pin joint's motor driven to an angle, on a body jointed to a multibody link: the
    /// solver's generic joint builder takes the sine of the target angle.
    pub fn angular_motor_published() -> [u32; 1] {
        let (mut world, fixed, body) = still_world();
        world
            .multibody_joints
            .insert(fixed, body, RevoluteJointBuilder::new(), true)
            .expect("a multibody joint");
        let actuator = GenericJointBuilder::new(JointAxesMask::empty()).motor_position(
            JointAxis::AngX,
            0.364_942_55,
            10.0,
            1.0,
        );
        world.insert_impulse_joint(fixed, body, actuator);
        world.step();
        [world.bodies[body].rotation().im.to_bits()]
    }

    /// A pin joint's angular limit: the solver takes the sine and cosine of the limits' centre.
    pub fn angular_limits_published() -> [u32; 1] {
        let (mut world, fixed, body) = still_world();
        let center = -0.047_693_253;
        let half_width = 0.000_976_562_5;
        world.insert_impulse_joint(
            fixed,
            body,
            RevoluteJointBuilder::new().limits([center - half_width, center + half_width]),
        );
        world.step();
        [world.bodies[body].rotation().im.to_bits()]
    }

    /// Two friction coefficients of opposite zero signs combined by their maximum: which zero
    /// comes out is the processor's choice, not the maths library's.
    pub fn contact_coefficients_published() -> [u32; 1] {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        let fixed = world.insert_body(RigidBodyBuilder::fixed());
        world.insert_collider(
            ColliderBuilder::ball(1.0)
                .friction(-0.0)
                .friction_combine_rule(CoefficientCombineRule::Max),
            Some(fixed),
        );
        let moving = world.insert_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::Y * 1.9)
                .can_sleep(false),
        );
        world.insert_collider(
            ColliderBuilder::ball(1.0)
                .friction(0.0)
                .friction_combine_rule(CoefficientCombineRule::Max),
            Some(moving),
        );
        world.step();
        let pair = world.contact_pairs().next().expect("a contact pair");
        [pair
            .manifolds()
            .first()
            .expect("a manifold")
            .data
            .friction
            .to_bits()]
    }
}

mod three {
    use rapier3d::prelude::*;

    /// A pin joint's angle between two orientations: the joint takes an arcsine.
    pub fn revolute_angle_published() -> [u32; 1] {
        let x = std::hint::black_box(0.549_525_86_f32);
        let q = Rotation::from_xyzw(x, 0.0, 0.0, (1.0 - x * x).sqrt());
        [RevoluteJointBuilder::new(Vector::X)
            .build()
            .angle(&Rotation::IDENTITY, &q)
            .to_bits()]
    }

    /// The 3D form of the angular limit: the sine and cosine of half the limits' centre.
    pub fn angular_limits() -> [u32; 4] {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        let fixed = world.insert_body(RigidBodyBuilder::fixed());
        let body = world.insert_body(RigidBodyBuilder::dynamic().can_sleep(false));
        world.insert_collider(ColliderBuilder::ball(1.0), Some(body));
        let center = -0.047_693_253;
        let half_width = 0.000_976_562_5;
        world.insert_impulse_joint(
            fixed,
            body,
            RevoluteJointBuilder::new(Vector::Z).limits([center - half_width, center + half_width]),
        );
        world.step();
        world.bodies[body].rotation().to_array().map(f32::to_bits)
    }

    /// The mass properties of a shape whose inertia tensor is not diagonal: its principal
    /// inertia and their frame come from diagonalising it.
    pub fn inertia(shape: ColliderBuilder) -> [u32; 7] {
        let mut world = PhysicsWorld::new();
        let body = world.insert_body(RigidBodyBuilder::dynamic());
        world.insert_collider(shape, Some(body));
        let props = &world.bodies[body].mass_properties().local_mprops;
        let i = props.inv_principal_inertia.to_array();
        let q = props.principal_inertia_local_frame.to_array();
        [i[0], i[1], i[2], q[0], q[1], q[2], q[3]].map(f32::to_bits)
    }

    /// The convex hull dropped on a floor and left for two seconds: its contacts go through the
    /// polyhedron's support features.
    pub fn hull_at_rest() -> [u32; 7] {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::new(0.0, -9.8, 0.0);
        crate::three::settings(&mut world.integration_parameters);
        let floor = world.insert_body(RigidBodyBuilder::fixed());
        world.insert_collider(ColliderBuilder::cuboid(4.0, 0.5, 4.0), Some(floor));
        let body = world
            .insert_body(crate::three::dynamic(0.0, 1.5, 0.0).rotation(Vector::new(0.3, 0.1, 0.2)));
        world.insert_collider(crate::three::hull(), Some(body));
        for _ in 0..120 {
            world.step();
        }
        let body = &world.bodies[body];
        let (t, r) = (body.translation().to_array(), body.rotation().to_array());
        [t[0], t[1], t[2], r[0], r[1], r[2], r[3]].map(f32::to_bits)
    }
}
