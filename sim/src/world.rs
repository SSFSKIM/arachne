//! A world: what it was given, in order, and the rows computed from it.
//!
//! Row `i` is the world's state `i` steps of exactly 1/60 s after its first row. `advance_to(i)`
//! computes the rows up to `i`, in order, the first time any of them is asked for, and is a
//! lookup afterwards: the rows do not depend on how the steps were grouped into calls. Nothing
//! here takes a time step.
//!
//! A **closed** world advances by itself. An **open** world's bodies also feel forces its caller
//! computes at each tick (a callback of the scene): it advances one step at a time, given those
//! forces, which enter its declaration.
//!
//! Beside its rows a world records what it found at each one ([`World::findings`]): a body whose
//! state went non-finite, a body above the world's speed limits, and two bodies that passed
//! through each other. They are not part of the trace bytes.

use crate::encode::{self, Identity, Settings, Writer};
use crate::points::Points;
use crate::spec::{Body, Error, Force, Joint, JointKind, Shape, ShapeKind, error};
use crate::{three, two};

/// How a world is advanced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    /// The rigid-body solver: contacts, joints, rotation.
    Solver,
    /// Velocity Verlet on point masses, for worlds with no shape and no joint.
    PointMasses,
}

impl Integrator {
    fn code(self) -> u8 {
        match self {
            Integrator::Solver => 1,
            Integrator::PointMasses => 2,
        }
    }
}

/// What a world is given beside its bodies: the limits above which a body is unstable, and
/// whether its solver sweeps fast bodies against fixed ones. All enter the declaration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// The speed, units per second, above which a body is unstable; `None` for
    /// [`crate::SPEED_LIMIT`]. Positive and strictly below the solver's cap
    /// ([`World::speed_cap`]).
    pub speed_limit: Option<f32>,
    /// The angular speed, radians per second, above which a body is unstable; `None` for
    /// [`crate::ANGULAR_SPEED_LIMIT`]. Positive and strictly below the solver's cap
    /// ([`World::angular_speed_cap`]).
    pub angular_speed_limit: Option<f32>,
    /// Whether the solver's continuous collision detection sweeps a fast body against the fixed
    /// ones, so that it cannot pass through a thin one (on by default).
    pub continuous_collision: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            speed_limit: None,
            angular_speed_limit: None,
            continuous_collision: true,
        }
    }
}

/// The kinds of what a world finds at a row ([`World::findings`]).
pub mod finding {
    /// A body's state went non-finite: the solver set it back to its last finite pose and
    /// stopped it (its quarantine), or a point mass's numbers are not finite.
    pub const NON_FINITE: u32 = 1;
    /// A body's speed went above the world's speed limit.
    pub const SPEED: u32 = 2;
    /// A body's angular speed went above the world's angular speed limit.
    pub const ANGULAR_SPEED: u32 = 3;
    /// Two bodies passed through each other in the step to the row.
    pub const TUNNELLED: u32 = 4;
}

/// What a step's engine reports with a row: the pairs of bodies touching, the bodies its solver
/// stopped for a non-finite state, the pairs that passed through each other in the step, and
/// whether the solver scaled each body's speed, and its angular speed, down to its caps during
/// the step (empty where the engine has no caps).
#[derive(Default)]
pub(crate) struct Recorded {
    pub touching: Vec<(u32, u32)>,
    pub quarantined: Vec<u32>,
    pub tunnelled: Vec<(u32, u32)>,
    pub capped: Vec<(bool, bool)>,
}

/// A body's flags as of the last row: which of the unstable conditions hold for it.
const ABOVE_SPEED: u8 = 1;
const ABOVE_ANGULAR_SPEED: u8 = 2;
/// Once set, kept: a body whose state went non-finite is found once.
const NON_FINITE: u8 = 4;

enum Engine {
    Two(Box<two::Solver>),
    Three(Box<three::Solver>),
    Points(Points),
}

/// The crate's version, the solver's, and the vendored commits it is built from (vendor/VENDORED.md).
const SOLVER_VERSION: &str = "0.36.1 vendored: rapier b303bc13, parry dbcce409, glamx 78c97db7";
/// The toolchain, target and target features this crate was compiled with (build.rs).
const BUILD: &str = env!("ARACHNE_SIM_BUILD");

pub struct World {
    dim: usize,
    integrator: Integrator,
    open: bool,
    settings: Settings,
    engine: Engine,
    declaration: Writer,
    bodies: Vec<bool>,
    joints: Vec<JointKind>,
    rows: usize,
    pose: Vec<f32>,
    velocity: Vec<f32>,
    in_contact: Vec<(u32, u32)>,
    events: Vec<[u32; 4]>,
    /// Each body's flags as of the last row (`ABOVE_SPEED`, ...).
    flags: Vec<u8>,
    findings: Vec<[u32; 4]>,
    /// The value of each finding, in order: the speed or angular speed found above its limit;
    /// NaN for the others.
    finding_values: Vec<f32>,
}

impl World {
    /// A world in `dimension` (2 or 3) dimensions, with the default [`Options`].
    pub fn new(dimension: usize, integrator: Integrator, open: bool) -> Result<Self, Error> {
        World::with_options(dimension, integrator, open, Options::default())
    }

    /// A world in `dimension` (2 or 3) dimensions, given `options`.
    pub fn with_options(
        dimension: usize,
        integrator: Integrator,
        open: bool,
        options: Options,
    ) -> Result<Self, Error> {
        if dimension != 2 && dimension != 3 {
            return error(format!("A world has 2 or 3 dimensions, not {dimension}."));
        }
        let mut settings = Settings::of(dimension);
        if let Some(limit) = options.speed_limit {
            below(limit, settings.speed_cap(), "A world's speed limit", "u/s")?;
            settings.speed_limit = limit;
        }
        if let Some(limit) = options.angular_speed_limit {
            below(
                limit,
                settings.angular_speed_cap(),
                "A world's angular speed limit",
                "rad/s",
            )?;
            settings.angular_speed_limit = limit;
        }
        if !options.continuous_collision {
            settings.ccd_substeps = 0;
        }
        let engine = match (integrator, dimension) {
            (Integrator::PointMasses, _) => {
                Engine::Points(Points::new(dimension, settings.dt, settings.point_substeps))
            }
            (Integrator::Solver, 2) => Engine::Two(Box::new(two::Solver::new(&settings))),
            (Integrator::Solver, _) => Engine::Three(Box::new(three::Solver::new(&settings))),
        };
        Ok(World {
            dim: dimension,
            integrator,
            open,
            settings,
            engine,
            declaration: encode::head(dimension, open, integrator.code(), &settings),
            bodies: Vec::new(),
            joints: Vec::new(),
            rows: 0,
            pose: Vec::new(),
            velocity: Vec::new(),
            in_contact: Vec::new(),
            events: Vec::new(),
            flags: Vec::new(),
            findings: Vec::new(),
            finding_values: Vec::new(),
        })
    }

    /// The speed, units per second, above which a body is unstable.
    pub fn speed_limit(&self) -> f32 {
        self.settings.speed_limit
    }

    /// The angular speed, radians per second, above which a body is unstable.
    pub fn angular_speed_limit(&self) -> f32 {
        self.settings.angular_speed_limit
    }

    /// The solver's own cap on a body's speed, units per second: a speed limit is below it.
    pub fn speed_cap(&self) -> f32 {
        self.settings.speed_cap()
    }

    /// The solver's own cap on a body's angular speed, radians per second.
    pub fn angular_speed_cap(&self) -> f32 {
        self.settings.angular_speed_cap()
    }

    /// Whether the solver sweeps fast bodies against fixed ones ([`Options`]).
    pub fn continuous_collision(&self) -> bool {
        self.settings.ccd_substeps > 0
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn integrator(&self) -> Integrator {
        self.integrator
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// The number of bodies.
    pub fn bodies(&self) -> usize {
        self.bodies.len()
    }

    /// The number of rows computed.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Numbers per body in a row of `pose`: `x, y, angle` in 2D, `x, y, z, qx, qy, qz, qw` in 3D.
    pub fn pose_width(&self) -> usize {
        if self.dim == 2 { 3 } else { 7 }
    }

    /// Numbers per body in a row of `velocity`: `vx, vy, w` in 2D, `vx, vy, vz, wx, wy, wz` in 3D.
    pub fn velocity_width(&self) -> usize {
        if self.dim == 2 { 3 } else { 6 }
    }

    fn angular(&self) -> usize {
        if self.dim == 2 { 1 } else { 3 }
    }

    fn rotation(&self) -> usize {
        if self.dim == 2 { 1 } else { 4 }
    }

    /// The row an item declared now acts from: the next step's starting row.
    fn row(&self) -> u32 {
        self.rows.saturating_sub(1) as u32
    }

    fn vector(&self, v: &[f32], what: &str) -> Result<(), Error> {
        numbers(v, self.dim, what)
    }

    fn body(&self, body: u32, what: &str) -> Result<(), Error> {
        if (body as usize) < self.bodies.len() {
            return Ok(());
        }
        error(format!(
            "{what} names body {body}, but the world has {} bodies.",
            self.bodies.len()
        ))
    }

    fn before_first_row(&self, what: &str) -> Result<(), Error> {
        if self.rows == 0 {
            return Ok(());
        }
        error(format!(
            "{what} joins a world before its first row: every row has the same bodies. Declare it \
             in a new world."
        ))
    }

    fn solver_only(&self, what: &str) -> Result<(), Error> {
        if self.integrator == Integrator::Solver {
            return Ok(());
        }
        error(format!(
            "{what} needs the solver: a world of point masses has no shapes and no joints."
        ))
    }

    /// Declares a body; returns its number.
    pub fn add_body(&mut self, b: Body) -> Result<u32, Error> {
        self.before_first_row("A body")?;
        self.vector(&b.position, "A body's position")?;
        numbers(&b.rotation, self.rotation(), "A body's rotation")?;
        self.vector(&b.velocity, "A body's velocity")?;
        numbers(
            &b.angular_velocity,
            self.angular(),
            "A body's angular velocity",
        )?;
        numbers(
            &[b.mass, b.gravity_scale],
            2,
            "A body's mass and gravity scale",
        )?;
        if b.mass < 0.0 {
            return error("A body's mass is not negative.");
        }
        if self.dim == 3 && b.rotation.iter().all(|&x| x == 0.0) {
            return error("A body's rotation is a quaternion (x, y, z, w) that is not zero.");
        }
        let moving = b
            .velocity
            .iter()
            .chain(&b.angular_velocity)
            .any(|&x| x != 0.0);
        if b.fixed && moving {
            return error("A fixed body does not move: its velocities are zero.");
        }
        if self.integrator == Integrator::PointMasses
            && b.angular_velocity.iter().any(|&x| x != 0.0)
        {
            return error(
                "A point mass does not turn: its angular velocity is zero. Use the solver for a \
                 body that spins.",
            );
        }
        let number = self.bodies.len() as u32;
        let row = self.row();
        encode::body(&mut self.declaration, row, &b);
        match &mut self.engine {
            Engine::Two(s) => s.add_body(&b),
            Engine::Three(s) => s.add_body(&b),
            Engine::Points(p) => p.add_body(&b),
        }
        self.bodies.push(b.fixed);
        Ok(number)
    }

    /// Declares a shape on a body.
    pub fn add_shape(&mut self, s: Shape) -> Result<(), Error> {
        self.before_first_row("A shape")?;
        self.solver_only("A shape")?;
        self.body(s.body, "A shape")?;
        self.vector(&s.position, "A shape's position")?;
        numbers(&s.rotation, self.rotation(), "A shape's rotation")?;
        numbers(
            &[s.density, s.friction, s.restitution],
            3,
            "A shape's material",
        )?;
        if s.density < 0.0 || s.friction < 0.0 || s.restitution < 0.0 {
            return error("A shape's density, friction and restitution are not negative.");
        }
        if self.dim == 3 && s.rotation.iter().all(|&x| x == 0.0) {
            return error("A shape's rotation is a quaternion (x, y, z, w) that is not zero.");
        }
        match &s.kind {
            ShapeKind::Ball { radius } => positive(&[*radius], "A ball's radius")?,
            ShapeKind::Cuboid { half_extents } => {
                self.vector(half_extents, "A cuboid's half extents")?;
                positive(half_extents, "A cuboid's half extents")?;
            }
            ShapeKind::Capsule {
                half_height,
                radius,
            } => {
                positive(&[*radius], "A capsule's radius")?;
                numbers(&[*half_height], 1, "A capsule's half height")?;
                if *half_height < 0.0 {
                    return error("A capsule's half height is not negative.");
                }
            }
            ShapeKind::ConvexHull { points } => {
                let n = points.len() / self.dim;
                if points.len() % self.dim != 0 || n < self.dim + 1 {
                    return error(format!(
                        "A convex hull takes at least {} points of {} numbers each.",
                        self.dim + 1,
                        self.dim
                    ));
                }
                numbers(points, points.len(), "A convex hull's points")?;
            }
        }
        let added = match &mut self.engine {
            Engine::Two(e) => e.add_shape(&s),
            Engine::Three(e) => e.add_shape(&s),
            Engine::Points(_) => unreachable!("refused above"),
        };
        added.map_err(|why| Error(format!("A convex hull failed: {why}.")))?;
        let row = self.row();
        encode::shape(&mut self.declaration, row, &s);
        Ok(())
    }

    /// Declares a joint; returns its number.
    pub fn add_joint(&mut self, j: Joint) -> Result<u32, Error> {
        self.solver_only("A joint")?;
        self.body(j.body1, "A joint")?;
        self.body(j.body2, "A joint")?;
        if j.body1 == j.body2 {
            return error("A joint joins two different bodies.");
        }
        self.vector(&j.anchor1, "A joint's first anchor")?;
        self.vector(&j.anchor2, "A joint's second anchor")?;
        let axis =
            matches!(j.kind, JointKind::Slide) || (self.dim == 3 && j.kind == JointKind::Pin);
        if axis {
            self.vector(&j.axis, "A joint's axis")?;
            // Normalized by the solver: its squared length is a positive finite number.
            let squared: f32 = j.axis.iter().map(|x| x * x).sum();
            if !(squared > 0.0 && squared.is_finite()) {
                return error(
                    "A joint's axis is a direction: not zero, and of a length whose square the \
                     stored type holds.",
                );
            }
        } else if !j.axis.is_empty() {
            return error("Only a slide, and a pin in 3D, take an axis.");
        }
        match j.kind {
            JointKind::Ball if self.dim == 2 => {
                return error("A ball joint is 3D only; in 2D, a pin turns about its anchors.");
            }
            JointKind::Rope { length } => positive(&[length], "A rope's length")?,
            JointKind::Spring {
                rest,
                stiffness,
                damping,
            } => {
                numbers(&[rest, stiffness, damping], 3, "A spring joint's numbers")?;
                if rest < 0.0 || stiffness < 0.0 || damping < 0.0 {
                    return error(
                        "A spring's rest length, stiffness and damping are not negative.",
                    );
                }
            }
            _ => {}
        }
        if let Some([low, high]) = j.limits {
            numbers(&[low, high], 2, "A joint's limits")?;
            if !matches!(j.kind, JointKind::Pin | JointKind::Slide) {
                return error("Only a pin and a slide take limits.");
            }
            if low > high {
                return error("A joint's limits are [min, max], min at most max.");
            }
        }
        let number = self.joints.len() as u32;
        let row = self.row();
        encode::joint(&mut self.declaration, row, &j);
        match &mut self.engine {
            Engine::Two(e) => e.add_joint(&j),
            Engine::Three(e) => e.add_joint(&j),
            Engine::Points(_) => unreachable!("refused above"),
        }
        self.joints.push(j.kind);
        Ok(number)
    }

    /// Declares a force law.
    pub fn add_force(&mut self, f: Force) -> Result<(), Error> {
        match &f {
            Force::Gravity { acceleration } => self.vector(acceleration, "Gravity")?,
            Force::Attraction { strength, bodies } => {
                numbers(&[*strength], 1, "An attraction's strength")?;
                if bodies.len() < 2 {
                    return error("An attraction acts among two bodies or more.");
                }
                for (k, &b) in bodies.iter().enumerate() {
                    self.body(b, "An attraction")?;
                    if bodies[..k].contains(&b) {
                        return error("An attraction names each body once.");
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
                self.body(*body_a, "A spring")?;
                self.body(*body_b, "A spring")?;
                if body_a == body_b {
                    return error("A spring joins two different bodies.");
                }
                self.vector(anchor_a, "A spring's first anchor")?;
                self.vector(anchor_b, "A spring's second anchor")?;
                numbers(&[*stiffness, *damping, *rest], 3, "A spring's numbers")?;
                if *rest < 0.0 || *stiffness < 0.0 || *damping < 0.0 {
                    return error(
                        "A spring's rest length, stiffness and damping are not negative.",
                    );
                }
                let anchored = anchor_a.iter().chain(anchor_b).any(|&x| x != 0.0);
                if self.integrator == Integrator::PointMasses && anchored {
                    return error(
                        "A point mass has no extent: a spring between point masses is anchored \
                         at their origins.",
                    );
                }
            }
            Force::LinearDrag { body, coefficient }
            | Force::QuadraticDrag { body, coefficient } => {
                self.body(*body, "A drag")?;
                numbers(&[*coefficient], 1, "A drag's coefficient")?;
                if *coefficient < 0.0 {
                    return error("A drag's coefficient is not negative.");
                }
            }
            Force::Constant {
                body,
                force,
                torque,
            } => {
                self.body(*body, "A constant force")?;
                self.vector(force, "A constant force")?;
                numbers(torque, self.angular(), "A constant torque")?;
                if self.integrator == Integrator::PointMasses && torque.iter().any(|&x| x != 0.0) {
                    return error("A point mass does not turn: a torque on it is zero.");
                }
            }
            Force::Motor {
                joint,
                position,
                velocity,
                stiffness,
                damping,
            } => {
                let Some(kind) = self.joints.get(*joint as usize) else {
                    return error(format!(
                        "A motor names joint {joint}, but the world has {} joints.",
                        self.joints.len()
                    ));
                };
                if !matches!(kind, JointKind::Pin | JointKind::Slide) {
                    return error("A motor drives a pin or a slide.");
                }
                numbers(
                    &[*position, *velocity, *stiffness, *damping],
                    4,
                    "A motor's numbers",
                )?;
                if *stiffness < 0.0 || *damping < 0.0 {
                    return error("A motor's stiffness and damping are not negative.");
                }
            }
        }
        let row = self.row();
        encode::force(&mut self.declaration, row, &f);
        match &mut self.engine {
            Engine::Two(e) => e.add_force(&f, &self.joints),
            Engine::Three(e) => e.add_force(&f, &self.joints),
            Engine::Points(p) => p.add_force(&f),
        }
        Ok(())
    }

    /// A body's mass: its point mass and its shapes'.
    pub fn mass(&self, body: u32) -> Result<f32, Error> {
        self.body(body, "A mass")?;
        Ok(match &self.engine {
            Engine::Two(e) => e.mass(body as usize),
            Engine::Three(e) => e.mass(body as usize),
            Engine::Points(p) => p.mass(body as usize),
        })
    }

    /// Computes the rows up to `row`, if they are not computed yet. An open world computes its
    /// first row here and every later one through `step`.
    pub fn advance_to(&mut self, row: usize) -> Result<(), Error> {
        if row < self.rows {
            return Ok(());
        }
        if row > u32::MAX as usize - 1 {
            return error("A world holds fewer than 2^32 rows.");
        }
        if self.rows == 0 {
            self.first_row()?;
        }
        if row >= self.rows && self.open {
            return error(format!(
                "An open world computes row {row} by step(), given the forces of its callbacks \
                 for each tick; it has {} rows.",
                self.rows
            ));
        }
        while self.rows <= row {
            self.next_row(None);
        }
        Ok(())
    }

    /// An open world's next row, after a step under `forces` and `torques` (`dimension` and
    /// one or three numbers per body, in body order) besides its own force laws. Returns its
    /// number.
    pub fn step(&mut self, forces: &[f32], torques: &[f32]) -> Result<usize, Error> {
        if !self.open {
            return error("A closed world is advanced by advance_to(); only an open world steps.");
        }
        let n = self.bodies.len();
        numbers(forces, n * self.dim, "An open world's forces")?;
        numbers(torques, n * self.angular(), "An open world's torques")?;
        if self.integrator == Integrator::PointMasses && torques.iter().any(|&x| x != 0.0) {
            return error("A point mass does not turn: a torque on it is zero.");
        }
        if self.rows == 0 {
            self.first_row()?;
        }
        encode::applied(&mut self.declaration, self.rows as u32, forces, torques, n);
        self.next_row(Some((forces, torques)));
        Ok(self.rows - 1)
    }

    fn first_row(&mut self) -> Result<(), Error> {
        for (i, &fixed) in self.bodies.iter().enumerate() {
            let mass = self.mass(i as u32)?;
            if !fixed && mass <= 0.0 {
                return error(format!(
                    "Body {i} moves but has no mass: give it a point mass, or a shape with density."
                ));
            }
        }
        self.flags = vec![0; self.bodies.len()];
        self.record();
        Ok(())
    }

    fn next_row(&mut self, applied: Option<(&[f32], &[f32])>) {
        match &mut self.engine {
            Engine::Two(e) => e.step(applied),
            Engine::Three(e) => e.step(applied),
            Engine::Points(p) => p.step(applied.map(|(forces, _)| forces)),
        }
        self.record();
    }

    fn record(&mut self) {
        let recorded = match &mut self.engine {
            Engine::Two(e) => e.record(&mut self.pose, &mut self.velocity),
            Engine::Three(e) => e.record(&mut self.pose, &mut self.velocity),
            Engine::Points(p) => {
                p.record(&mut self.pose, &mut self.velocity);
                Recorded::default()
            }
        };
        let row = self.rows as u32;
        self.find(row, &recorded);
        let pairs = recorded.touching;
        // Merge the sorted pair lists: a pair in one only began or ended at this row.
        let (old, new) = (&self.in_contact, &pairs);
        let (mut i, mut j) = (0, 0);
        while i < old.len() || j < new.len() {
            if j == new.len() || (i < old.len() && old[i] < new[j]) {
                self.events.push([row, old[i].0, old[i].1, 0]);
                i += 1;
            } else if i == old.len() || new[j] < old[i] {
                self.events.push([row, new[j].0, new[j].1, 1]);
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }
        self.in_contact = pairs;
        self.rows += 1;
    }

    /// The rows `start..stop` of every body's pose, flat: `[rows][bodies][pose_width]`.
    pub fn pose(&self, start: usize, stop: usize) -> &[f32] {
        let w = self.bodies.len() * self.pose_width();
        &self.pose[start * w..stop * w]
    }

    /// The rows `start..stop` of every body's velocity, flat: `[rows][bodies][velocity_width]`.
    pub fn velocity(&self, start: usize, stop: usize) -> &[f32] {
        let w = self.bodies.len() * self.velocity_width();
        &self.velocity[start * w..stop * w]
    }

    /// What was found at the last row: for each moving body, a non-finite state (once, at the
    /// row it began), and a speed or an angular speed above the world's limit (at each row at
    /// which it rises above, after a row at which it did not); and each pair that passed through
    /// each other in the step to the row. A speed is above the limit when the row's is, or when
    /// the solver scaled it down to its cap during the step (the cap is above the limit); its
    /// value is then at least the cap, and the cap is what is known of it.
    fn find(&mut self, row: u32, recorded: &Recorded) {
        let n = self.bodies.len();
        let (pw, vw) = (self.pose_width(), self.velocity_width());
        let pose = &self.pose[self.pose.len() - n * pw..];
        let velocity = &self.velocity[self.velocity.len() - n * vw..];
        let (limit, angular_limit) = (self.settings.speed_limit, self.settings.angular_speed_limit);
        let mut found: Vec<([u32; 4], f32)> = Vec::new();
        for (b, &fixed) in self.bodies.iter().enumerate() {
            if fixed {
                continue;
            }
            let v = &velocity[b * vw..(b + 1) * vw];
            let finite = !recorded.quarantined.contains(&(b as u32))
                && pose[b * pw..(b + 1) * pw]
                    .iter()
                    .chain(v)
                    .all(|x| x.is_finite());
            let mut now = self.flags[b] & NON_FINITE;
            let (mut speed, mut angular) = (f32::NAN, f32::NAN);
            if !finite {
                now |= NON_FINITE;
            } else if now == 0 {
                (speed, angular) = speeds(v, self.dim);
                if let Some(&(linear, turning)) = recorded.capped.get(b) {
                    // Scaled down to the cap: it was above the cap, and so above the limit.
                    if linear {
                        speed = speed.max(self.settings.speed_cap());
                    }
                    if turning {
                        angular = angular.max(self.settings.angular_speed_cap());
                    }
                }
                if speed > limit {
                    now |= ABOVE_SPEED;
                }
                if angular > angular_limit {
                    now |= ABOVE_ANGULAR_SPEED;
                }
            }
            for (flag, kind, value) in [
                (NON_FINITE, finding::NON_FINITE, f32::NAN),
                (ABOVE_SPEED, finding::SPEED, speed),
                (ABOVE_ANGULAR_SPEED, finding::ANGULAR_SPEED, angular),
            ] {
                if now & flag != 0 && self.flags[b] & flag == 0 {
                    found.push(([row, kind, b as u32, b as u32], value));
                }
            }
            self.flags[b] = now;
        }
        found.extend(
            recorded
                .tunnelled
                .iter()
                .map(|&(a, b)| ([row, finding::TUNNELLED, a, b], f32::NAN)),
        );
        found.sort_unstable_by_key(|&(key, _)| key);
        for (key, value) in found {
            self.findings.push(key);
            self.finding_values.push(value);
        }
    }

    /// What was found, by row: `[row, kind, a, b]` (kind one of [`finding`]), sorted. A body's
    /// finding names it as both `a` and `b`; a pair's names its bodies, `a < b`. A non-finite
    /// state is found at the row it began, once for the body; a speed above a limit at each row
    /// it rises above it. Two bodies passed through each other in a step when neither touched
    /// the other at the step's start or end, the solver held no contact between them in the
    /// step, and the two, swept through the step's motion (each body's centre of mass along a
    /// line and its turn about it at a constant rate), met; it is found at the step's last row.
    /// Fixed bodies, two bodies a joint keeps from colliding, and a body found non-finite are
    /// not checked. None of it is in the trace bytes.
    pub fn findings(&self) -> &[[u32; 4]] {
        &self.findings
    }

    /// The value of each of [`World::findings`], in order: the speed (units per second) or
    /// angular speed (radians per second) found above its limit: the row's, or, when the solver
    /// scaled it down to its cap in the step, the larger of the row's and the cap (it was at
    /// least that); NaN for a non-finite state and for two bodies that passed through each
    /// other.
    pub fn finding_values(&self) -> &[f32] {
        &self.finding_values
    }

    /// The contacts that began and ended: `[row, a, b, kind]`, `a < b` bodies by number, kind 1
    /// for began and 0 for ended, sorted. A contact between two bodies begins at the first row
    /// (row 0 included) at which a shape of each is at most the allowed linear error apart
    /// (touching or overlapping; 0.005 units, in the declaration's settings), and ends at the
    /// first later row at which no two are; two fixed bodies, and two bodies a joint keeps from
    /// colliding, have none. A contact is between two bodies, whatever shapes of
    /// theirs touch.
    pub fn contacts(&self) -> &[[u32; 4]] {
        &self.events
    }

    /// The declaration bytes: the head and settings, then every item and every open step in
    /// the order they were given.
    pub fn declaration_bytes(&self) -> &[u8] {
        &self.declaration.0
    }

    /// What computed the rows.
    pub fn identity(&self) -> Identity {
        let (solver, solver_version, integrator, features): (&str, &str, String, &[&str]) =
            match self.integrator {
                Integrator::Solver => (
                    if self.dim == 2 {
                        "rapier2d"
                    } else {
                        "rapier3d"
                    },
                    SOLVER_VERSION,
                    format!(
                        "rapier/{} substeps, {} in a component with a joint",
                        self.settings.solver_iterations, self.settings.jointed_substeps
                    ),
                    if self.settings.block_solver {
                        &["block-solver", "enhanced-determinism", "f32"]
                    } else {
                        &["enhanced-determinism", "f32"]
                    },
                ),
                Integrator::PointMasses => (
                    "arachne-sim",
                    env!("CARGO_PKG_VERSION"),
                    format!("velocity-verlet/{} substeps", self.settings.point_substeps),
                    &["f32"],
                ),
            };
        Identity {
            producer: "arachne-sim".into(),
            producer_version: env!("CARGO_PKG_VERSION").into(),
            solver: solver.into(),
            solver_version: solver_version.into(),
            integrator,
            build: BUILD.into(),
            features: features.iter().map(|f| f.to_string()).collect(),
        }
    }

    /// The trace bytes: the canonical bytes of the bake content these rows make (channels
    /// `pose` and `velocity`, the event table `contact`).
    pub fn trace_bytes(&self) -> Vec<u8> {
        let units = self.units();
        let n = self.bodies.len() as u32;
        encode::content(
            &self.identity(),
            self.rows as u32,
            &[
                encode::Channel {
                    name: "pose",
                    count: n,
                    width: self.pose_width() as u32,
                    unit: units.0,
                    values: &self.pose,
                },
                encode::Channel {
                    name: "velocity",
                    count: n,
                    width: self.velocity_width() as u32,
                    unit: units.1,
                    values: &self.velocity,
                },
            ],
            &[("contact", &self.events)],
        )
    }

    /// The units of the `pose` and `velocity` channels.
    pub fn units(&self) -> (&'static str, &'static str) {
        if self.dim == 2 {
            ("u u rad", "u/s u/s rad/s")
        } else {
            ("u u u 1 1 1 1", "u/s u/s u/s rad/s rad/s rad/s")
        }
    }
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("dimension", &self.dim)
            .field("integrator", &self.integrator)
            .field("open", &self.open)
            .field("bodies", &self.bodies.len())
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

/// A body's speed and angular speed from its velocity row (`vx, vy, w` or
/// `vx, vy, vz, wx, wy, wz`).
fn speeds(v: &[f32], dimension: usize) -> (f32, f32) {
    if dimension == 2 {
        ((v[0] * v[0] + v[1] * v[1]).sqrt(), v[2].abs())
    } else {
        (
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
            (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]).sqrt(),
        )
    }
}

/// A limit: a positive number strictly below `cap`.
fn below(limit: f32, cap: f32, what: &str, unit: &str) -> Result<(), Error> {
    if limit.is_finite() && limit > 0.0 && limit < cap {
        return Ok(());
    }
    error(format!(
        "{what} is a positive number below the solver's own cap of {cap} {unit}, above which \
         the solver scales a body's motion down; not {limit}."
    ))
}

/// `count` finite numbers.
fn numbers(xs: &[f32], count: usize, what: &str) -> Result<(), Error> {
    if xs.len() != count {
        return error(format!("{what} takes {count} numbers, not {}.", xs.len()));
    }
    if xs.iter().any(|x| !x.is_finite()) {
        return error(format!("{what} takes finite numbers."));
    }
    Ok(())
}

fn positive(xs: &[f32], what: &str) -> Result<(), Error> {
    numbers(xs, xs.len(), what)?;
    if xs.iter().any(|&x| x <= 0.0) {
        return error(format!("{what} is greater than zero."));
    }
    Ok(())
}
