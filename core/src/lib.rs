//! Arachne's CPU core: draws frames described in XF, the Frame IR.
//!
//! - [`xf`]: the IR's types, content keys and binary form.
//! - [`stream`]: a [`stream::Stream`] holds what a drawer keeps across frames; deltas change it and
//!   `draw` turns it into pixels.
//! - [`project`]: CE's camera maths (the 3D projection and the frame-to-pixel mappings).
//! - [`shade`]: the one place lighting lives (CE's per-face model, soft diffuse light).
//! - [`composite`]: the one place blending lives: fragments, the opaque and transparent passes,
//!   the two resolves, emission, and the frame buffer.
//! - [`mesh`]: the triangle rasterizer, a producer of mask fragments with depth.
//! - [`tiles`]: a frame cut into tiles, drawn on threads scoped to the draw call.
//!
//! The core's output may not depend on the machine: no runtime CPU-feature dispatch, no fused
//! multiply-add, transcendental functions from the pure-Rust `libm` crate (`clippy.toml` enforces
//! the last two), and every pixel's result a function of its own fragments only.

pub mod composite;
pub mod mesh;
pub mod project;
pub mod shade;
pub mod stream;
pub mod tiles;
pub mod xf;

#[cfg(feature = "python")]
mod python;
