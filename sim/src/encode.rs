//! The two canonical encodings: a world's **declaration bytes** (everything it was given, in the
//! order it was given) and its **trace bytes** (what was computed: the bake content's canonical
//! bytes). Integers and floats are little-endian; a float is written as its bits.

use crate::spec::{Body, Force, Joint, JointKind, Shape, ShapeKind};

/// The version of the declaration bytes' layout.
pub const DECLARATION_LAYOUT: u32 = 1;
/// The version of the content's canonical layout (the record's `LAYOUT_VERSION`).
pub const CONTENT_LAYOUT: u32 = 1;
/// The one quiet NaN a non-finite value is stored as in the trace.
const NAN_BITS: u32 = 0x7fc0_0000;

/// The code of each declared item, its first byte.
pub(crate) mod kind {
    pub const BODY: u8 = 1;
    pub const SHAPE: u8 = 2;
    pub const JOINT: u8 = 3;
    pub const GRAVITY: u8 = 4;
    pub const ATTRACTION: u8 = 5;
    pub const SPRING: u8 = 6;
    pub const LINEAR_DRAG: u8 = 7;
    pub const QUADRATIC_DRAG: u8 = 8;
    pub const CONSTANT: u8 = 9;
    pub const MOTOR: u8 = 10;
    /// An open world's forces for one step.
    pub const APPLIED: u8 = 11;
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Writer(pub Vec<u8>);

impl Writer {
    pub fn u8(&mut self, x: u8) {
        self.0.push(x);
    }

    pub fn bool(&mut self, x: bool) {
        self.u8(u8::from(x));
    }

    pub fn u32(&mut self, x: u32) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }

    pub fn f32(&mut self, x: f32) {
        self.u32(x.to_bits());
    }

    pub fn f32s(&mut self, xs: &[f32]) {
        for &x in xs {
            self.f32(x);
        }
    }

    /// A list: its length, then its values.
    pub fn list(&mut self, xs: &[f32]) {
        self.u32(xs.len() as u32);
        self.f32s(xs);
    }

    pub fn text(&mut self, text: &str) {
        self.u32(text.len() as u32);
        self.0.extend_from_slice(text.as_bytes());
    }

    /// A trace value: a non-finite one is the one quiet NaN.
    pub fn stored(&mut self, x: f32) {
        self.u32(if x.is_finite() { x.to_bits() } else { NAN_BITS });
    }
}

/// The settings every world is stepped with: every solver default that moves numbers or a
/// teaching claim, set here rather than left to the solver (README.md, "Determinism", P9), and
/// written at the head of every declaration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Settings {
    /// The step, exactly one tick.
    pub dt: f32,
    pub length_unit: f32,
    pub solver_iterations: u32,
    /// The solver's substeps in a world with a joint: the solver's pass that removes its
    /// position corrections' velocity takes energy from a swinging joint in proportion to the
    /// substep, so a jointed world takes more of them (the closed-form swings' energy line).
    pub joint_solver_iterations: u32,
    pub pgs_iterations: u32,
    pub stabilization_iterations: u32,
    pub ccd_substeps: u32,
    pub max_linear_velocity: f32,
    pub contact_recycling: bool,
    pub recycle_distance: f32,
    pub contact_clustering: bool,
    /// The 3D solver's friction model: 0 for its simplified one (one friction constraint per
    /// manifold, its default), 1 for Coulomb's per contact point (2D's only model).
    pub friction_model: u8,
    pub warmstart_joints: bool,
    pub friction_in_bias_pass: bool,
    pub block_solver: bool,
    pub sleeping: bool,
    pub fast_rotation: bool,
    /// The point-mass integrator's substeps per tick.
    pub point_substeps: u32,
}

impl Settings {
    pub fn of(dimension: usize) -> Self {
        Settings {
            dt: 1.0 / 60.0,
            length_unit: 1.0,
            solver_iterations: 4,
            joint_solver_iterations: 128,
            pgs_iterations: 1,
            stabilization_iterations: 1,
            ccd_substeps: 1,
            max_linear_velocity: 400.0,
            contact_recycling: true,
            recycle_distance: 0.05,
            contact_clustering: true,
            friction_model: if dimension == 3 { 0 } else { 1 },
            warmstart_joints: false,
            friction_in_bias_pass: false,
            // A compile-time feature of the solver: on in 2D only (Cargo.toml).
            block_solver: dimension == 2,
            sleeping: false,
            fast_rotation: false,
            point_substeps: 8,
        }
    }

    pub fn write(&self, w: &mut Writer) {
        w.f32(self.dt);
        w.f32(self.length_unit);
        w.u32(self.solver_iterations);
        w.u32(self.joint_solver_iterations);
        w.u32(self.pgs_iterations);
        w.u32(self.stabilization_iterations);
        w.u32(self.ccd_substeps);
        w.f32(self.max_linear_velocity);
        w.bool(self.contact_recycling);
        w.f32(self.recycle_distance);
        w.bool(self.contact_clustering);
        w.u8(self.friction_model);
        w.bool(self.warmstart_joints);
        w.bool(self.friction_in_bias_pass);
        w.bool(self.block_solver);
        w.bool(self.sleeping);
        w.bool(self.fast_rotation);
        w.u32(self.point_substeps);
    }
}

/// The head of a declaration: the layout, the dimension and stored type, whether the world is
/// open, its integrator, and the settings.
pub(crate) fn head(dimension: usize, open: bool, integrator: u8, settings: &Settings) -> Writer {
    let mut w = Writer::default();
    w.u32(DECLARATION_LAYOUT);
    w.u8(dimension as u8);
    // The stored type: 1 for f32, the bake's own dtype code.
    w.u8(1);
    w.bool(open);
    w.u8(integrator);
    settings.write(&mut w);
    w
}

/// Each item is its kind, the row from which it acts, then its fields in a fixed order; a body
/// or joint it names is its number among the bodies or joints.
pub(crate) fn body(w: &mut Writer, row: u32, b: &Body) {
    w.u8(kind::BODY);
    w.u32(row);
    w.bool(b.fixed);
    w.f32s(&b.position);
    w.f32s(&b.rotation);
    w.f32s(&b.velocity);
    w.f32s(&b.angular_velocity);
    w.f32(b.mass);
    w.f32(b.gravity_scale);
}

pub(crate) fn shape(w: &mut Writer, row: u32, s: &Shape) {
    w.u8(kind::SHAPE);
    w.u32(row);
    w.u32(s.body);
    match &s.kind {
        ShapeKind::Ball { radius } => {
            w.u8(1);
            w.f32(*radius);
        }
        ShapeKind::Cuboid { half_extents } => {
            w.u8(2);
            w.f32s(half_extents);
        }
        ShapeKind::Capsule {
            half_height,
            radius,
        } => {
            w.u8(3);
            w.f32(*half_height);
            w.f32(*radius);
        }
        ShapeKind::ConvexHull { points } => {
            w.u8(4);
            w.list(points);
        }
    }
    w.f32s(&s.position);
    w.f32s(&s.rotation);
    w.f32(s.density);
    w.f32(s.friction);
    w.f32(s.restitution);
}

pub(crate) fn joint(w: &mut Writer, row: u32, j: &Joint) {
    w.u8(kind::JOINT);
    w.u32(row);
    let code = match j.kind {
        JointKind::Pin => 1,
        JointKind::Slide => 2,
        JointKind::Weld => 3,
        JointKind::Ball => 4,
        JointKind::Rope { .. } => 5,
        JointKind::Spring { .. } => 6,
    };
    w.u8(code);
    w.u32(j.body1);
    w.u32(j.body2);
    w.f32s(&j.anchor1);
    w.f32s(&j.anchor2);
    w.list(&j.axis);
    match j.kind {
        JointKind::Rope { length } => w.f32(length),
        JointKind::Spring {
            rest,
            stiffness,
            damping,
        } => w.f32s(&[rest, stiffness, damping]),
        _ => {}
    }
    match j.limits {
        Some(limits) => {
            w.bool(true);
            w.f32s(&limits);
        }
        None => w.bool(false),
    }
    w.bool(j.contacts);
}

pub(crate) fn force(w: &mut Writer, row: u32, f: &Force) {
    match f {
        Force::Gravity { acceleration } => {
            w.u8(kind::GRAVITY);
            w.u32(row);
            w.f32s(acceleration);
        }
        Force::Attraction { strength, bodies } => {
            w.u8(kind::ATTRACTION);
            w.u32(row);
            w.f32(*strength);
            w.u32(bodies.len() as u32);
            for &b in bodies {
                w.u32(b);
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
            w.u8(kind::SPRING);
            w.u32(row);
            w.u32(*body_a);
            w.f32s(anchor_a);
            w.u32(*body_b);
            w.f32s(anchor_b);
            w.f32s(&[*stiffness, *damping, *rest]);
        }
        Force::LinearDrag { body, coefficient } => {
            w.u8(kind::LINEAR_DRAG);
            w.u32(row);
            w.u32(*body);
            w.f32(*coefficient);
        }
        Force::QuadraticDrag { body, coefficient } => {
            w.u8(kind::QUADRATIC_DRAG);
            w.u32(row);
            w.u32(*body);
            w.f32(*coefficient);
        }
        Force::Constant {
            body,
            force,
            torque,
        } => {
            w.u8(kind::CONSTANT);
            w.u32(row);
            w.u32(*body);
            w.f32s(force);
            w.f32s(torque);
        }
        Force::Motor {
            joint,
            position,
            velocity,
            stiffness,
            damping,
        } => {
            w.u8(kind::MOTOR);
            w.u32(row);
            w.u32(*joint);
            w.f32s(&[*position, *velocity, *stiffness, *damping]);
        }
    }
}

/// An open world's step to `row`: the force and the torque on each body, in body order.
pub(crate) fn applied(w: &mut Writer, row: u32, forces: &[f32], torques: &[f32], bodies: usize) {
    w.u8(kind::APPLIED);
    w.u32(row);
    let (fw, tw) = (forces.len() / bodies.max(1), torques.len() / bodies.max(1));
    for i in 0..bodies {
        w.f32s(&forces[i * fw..(i + 1) * fw]);
        w.f32s(&torques[i * tw..(i + 1) * tw]);
    }
}

/// What computed a content: the bake identity's seven strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub producer: String,
    pub producer_version: String,
    pub solver: String,
    pub solver_version: String,
    pub integrator: String,
    pub build: String,
    /// The features that change numbers, sorted.
    pub features: Vec<String>,
}

/// One channel of a content: per row, `width` values about each of `count` subjects.
pub(crate) struct Channel<'a> {
    pub name: &'a str,
    pub count: u32,
    pub width: u32,
    pub unit: &'a str,
    pub values: &'a [f32],
}

/// The content's canonical bytes, as the record computes them: `ARBK`; the layout; the identity's
/// seven strings (the features joined by commas); the row count; the channels in bytewise order
/// of name, each with its type code (1, f32), its rank and dimensions of one row, its unit and its
/// values; the event tables in bytewise order of name, each with its column count, its row count
/// and its rows. `channels` and `events` come in bytewise order of name.
pub(crate) fn content(
    identity: &Identity,
    rows: u32,
    channels: &[Channel<'_>],
    events: &[(&str, &[[u32; 4]])],
) -> Vec<u8> {
    let mut w = Writer::default();
    w.0.extend_from_slice(b"ARBK");
    w.u32(CONTENT_LAYOUT);
    for text in [
        &identity.producer,
        &identity.producer_version,
        &identity.solver,
        &identity.solver_version,
        &identity.integrator,
        &identity.build,
    ] {
        w.text(text);
    }
    w.text(&identity.features.join(","));
    w.u32(rows);
    w.u32(channels.len() as u32);
    for c in channels {
        w.text(c.name);
        w.u8(1);
        w.u8(2);
        w.u32(c.count);
        w.u32(c.width);
        w.text(c.unit);
        for &x in c.values {
            w.stored(x);
        }
    }
    w.u32(events.len() as u32);
    for (name, rows) in events {
        w.text(name);
        w.u8(4);
        w.u32(rows.len() as u32);
        for row in rows.iter() {
            for &x in row {
                w.u32(x);
            }
        }
    }
    w.0
}
