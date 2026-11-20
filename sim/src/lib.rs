//! Arachne's simulation crate: rigid-body worlds in 2D and 3D, stepped on a fixed clock of 60
//! ticks per second, whose results are the same bytes on every machine of one architecture.
//!
//! A [`World`] is given bodies, their shapes, joints and force laws ([`spec`]), each number
//! already in the stored type (`f32`), and computes its rows in order: row `i` is its state `i`
//! steps of exactly 1/60 s after its first. Everything it was given is kept as its declaration
//! bytes, and what it computed as its trace bytes (the bake content's canonical bytes), so two
//! runs are compared byte for byte.
//!
//! The solver is pinned with its determinism feature, and the arithmetic it can reach is
//! audited (README.md, "Determinism"); `examples/probe` is the audit's program. Worlds of point
//! masses (no shape, no joint) are integrated by the crate itself ([`Integrator`]).

// solver.rs is compiled twice on purpose, as two::solver and three::solver.
#![allow(clippy::duplicate_mod)]

mod encode;
mod points;
pub mod spec;
mod three;
mod two;
mod world;

#[cfg(feature = "python")]
mod python;

pub use encode::{CONTENT_LAYOUT, DECLARATION_LAYOUT, Identity};
pub use spec::{Body, Error, Force, Joint, JointKind, Shape, ShapeKind};
pub use world::{Integrator, World};
