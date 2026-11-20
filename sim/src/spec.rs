//! What a world is given: its bodies, their shapes, the joints between them and the force laws
//! that act on them, each as plain numbers in the stored type (`f32`). A vector has as many
//! components as the world has dimensions; a rotation is an angle in 2D and a quaternion
//! `(x, y, z, w)` in 3D; an angular velocity is one number in 2D and three in 3D.

/// A rigid body. Bodies are numbered in the order they are declared, from 0: that number is its
/// position on the bodies axis of every row.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    /// A fixed body never moves; its velocities are zero.
    pub fixed: bool,
    pub position: Vec<f32>,
    pub rotation: Vec<f32>,
    pub velocity: Vec<f32>,
    pub angular_velocity: Vec<f32>,
    /// A mass at the body's origin, beside what its shapes weigh; 0 for none. A point mass has no
    /// rotational inertia.
    pub mass: f32,
    /// How much of the world's gravity acts on the body (1: all of it).
    pub gravity_scale: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ShapeKind {
    /// A circle in 2D, a sphere in 3D.
    Ball { radius: f32 },
    /// A box, by its half extents.
    Cuboid { half_extents: Vec<f32> },
    /// A capsule along its local y axis: a segment of half length `half_height`, swept by
    /// `radius`.
    Capsule { half_height: f32, radius: f32 },
    /// The convex hull of points, given flat (`x0, y0, x1, y1, ...` in 2D).
    ConvexHull { points: Vec<f32> },
}

/// A shape fixed to a body, at a pose in the body's frame. Several shapes on one body make a
/// compound. A shape takes part in contacts and gives its body mass (`density` times its area in
/// 2D, its volume in 3D).
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub body: u32,
    pub kind: ShapeKind,
    pub position: Vec<f32>,
    pub rotation: Vec<f32>,
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum JointKind {
    /// Rotation about the anchors (2D) or about `axis` (3D); no translation between them.
    Pin,
    /// Translation along `axis`; no rotation.
    Slide,
    /// No relative motion: the two bodies keep the relative pose they had when it was declared.
    Weld,
    /// Rotation in every direction about the anchors (3D only).
    Ball,
    /// The anchors at most `length` apart.
    Rope { length: f32 },
    /// A spring between the anchors, solved by the solver as a joint.
    Spring {
        rest: f32,
        stiffness: f32,
        damping: f32,
    },
}

/// A joint between two bodies, at an anchor in each body's frame. Its frame is one frame in the
/// world as the bodies are posed when it is declared: at the first anchor, with its x axis along
/// `axis`. A pin's angle and a slide's offset (what limits and a motor act on) are measured from
/// that pose, zero as declared.
#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    pub kind: JointKind,
    pub body1: u32,
    pub body2: u32,
    pub anchor1: Vec<f32>,
    pub anchor2: Vec<f32>,
    /// A pin's axis in 3D and a slide's axis, in the first body's frame, of any length but zero
    /// (only its direction counts); empty otherwise.
    pub axis: Vec<f32>,
    /// A pin's angle or a slide's offset kept within `[min, max]`.
    pub limits: Option<[f32; 2]>,
    /// Whether the two bodies' shapes still collide with each other.
    pub contacts: bool,
}

/// A force law: the engine's own, evaluated in the crate.
#[derive(Clone, Debug, PartialEq)]
pub enum Force {
    /// A uniform acceleration on every body, scaled by each body's `gravity_scale`.
    Gravity { acceleration: Vec<f32> },
    /// Every pair of the bodies attracts by the inverse square of its distance:
    /// `strength * m_a * m_b / r^2`, along the line between their origins.
    Attraction { strength: f32, bodies: Vec<u32> },
    /// A damped spring between an anchor on each body, along the line between the anchors:
    /// `stiffness * (length - rest) + damping * (rate at which the anchors separate)`. At a rest
    /// length of 0 its elastic part is `stiffness * d`, `d` the second anchor's offset from the
    /// first, which needs no direction. For a tick at which the anchors coincide, where the line
    /// is undefined, it is zero (at rest length 0 its elastic part is zero there anyway, and its
    /// damping has no line to act along).
    Spring {
        body_a: u32,
        anchor_a: Vec<f32>,
        body_b: u32,
        anchor_b: Vec<f32>,
        stiffness: f32,
        damping: f32,
        rest: f32,
    },
    /// `-coefficient * v` on the body.
    LinearDrag { body: u32, coefficient: f32 },
    /// `-coefficient * |v| * v` on the body.
    QuadraticDrag { body: u32, coefficient: f32 },
    /// A constant force at the body's centre of mass, and a constant torque.
    Constant {
        body: u32,
        force: Vec<f32>,
        torque: Vec<f32>,
    },
    /// Drives a pin's angle or a slide's offset (zero as the joint was declared) toward
    /// `position` and its rate toward `velocity`: a force (a torque for a pin)
    /// `stiffness * (position - x) + damping * (velocity - dx/dt)`, whatever the bodies' masses.
    Motor {
        joint: u32,
        position: f32,
        velocity: f32,
        stiffness: f32,
        damping: f32,
    },
}

/// What the crate refuses, with what to do instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub(crate) fn error<T>(message: impl Into<String>) -> Result<T, Error> {
    Err(Error(message.into()))
}
