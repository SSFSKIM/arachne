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
//! - [`path`]: the path rasterizer (fills, strokes, dashes, glyphs), a producer of
//!   exact-area fragments with depth, on [`scanline`]'s exact nonzero coverage.
//! - [`geometry`]: a path record's reveal, blend and paint blend made into what is drawn.
//! - [`depth`]: canonical planes, so coplanar things tie exactly, and depth along curves.
//! - [`instance`]: `instances` and `points` records drawn as their members; [`points`], point
//!   clouds as CE's square blocks.
//! - [`boxes`]: where each record lands on the frame, found without drawing (`screen_boxes`).
//! - [`tiles`]: a frame cut into tiles, drawn on threads scoped to the draw call.
//! - [`probes`]: the class probes, the parity kit's fixtures (`parity/probes.rs`).
//!
//! The core's output may not depend on the machine: no runtime CPU-feature dispatch, no fused
//! multiply-add, transcendental functions from the pure-Rust `libm` crate (`clippy.toml` enforces
//! the last two), and every pixel's result a function of its own fragments only.

pub mod boxes;
pub mod composite;
pub mod depth;
pub mod geometry;
pub mod instance;
pub mod mesh;
pub mod path;
pub mod points;
#[path = "../parity/probes.rs"]
pub mod probes;
pub mod project;
pub mod scanline;
pub mod shade;
pub mod stream;
pub mod tiles;
pub mod xf;

#[cfg(feature = "python")]
mod python;
