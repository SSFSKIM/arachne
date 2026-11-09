//! Arachne's CPU core: draws frames described in XF, the Frame IR.
//!
//! - [`xf`]: the IR's types, content keys and binary form.
//! - [`stream`]: a [`stream::Stream`] holds what a drawer keeps across frames; deltas change it and
//!   `draw` turns it into pixels.
//! - [`project`]: CE's camera maths (the 3D projection and the frame-to-pixel mappings).
//! - [`shade`]: the one place lighting lives (CE's per-face model, soft diffuse light).
//! - [`composite`]: the one place blending lives: fragments, the opaque and transparent passes,
//!   the two resolves, emission, and the frame buffer.
//! - [`order`]: the order fragments composite in at a pixel: depth where two cover one
//!   sub-sample, paint order elsewhere.
//! - [`mesh`]: the triangle rasterizer, a producer of mask fragments with depth.
//! - [`path`]: the path rasterizer (fills, strokes, dashes, glyphs), a producer of
//!   exact-area fragments with depth, on [`scanline`]'s exact nonzero coverage.
//! - [`geometry`]: a path record's reveal, blend and paint blend made into what is drawn.
//! - [`gradient`]: linear gradients, coloured as Cairo's pattern and pixman's fetch colour them.
//! - [`depth`]: canonical planes, so coplanar things tie exactly, and depth along curves.
//! - [`instance`]: `instances` and `points` records drawn as their members; [`points`], point
//!   clouds as CE's square blocks.
//! - [`boxes`]: where each record lands on the frame, found without drawing (`screen_boxes`).
//! - [`tiles`]: a frame cut into tiles, drawn on threads scoped to the draw call.
//! - [`contributors`]: per pixel, the drawn leaves its colour is made of, read from the passes the
//!   resolves read (`Stream::contributors`).
//! - [`diagnostics`]: internal, not a supported interface: per pixel, what its resolve composited,
//!   enough to composite it again (`Stream::diagnostic_resolves`).
//! - `counters`: call counts of the depth read along curves, in a build with the `counters`
//!   feature only.
//! - [`probes`]: the class probes, the parity kit's fixtures (`parity/probes.rs`).
//!
//! The core's output may not depend on the machine: no runtime CPU-feature dispatch, no fused
//! multiply-add, transcendental functions from the pure-Rust `libm` crate (`clippy.toml` enforces
//! the last two), and every pixel's result a function of its own fragments only.

/// Adds `n` (one if not given) to the count `counters::$name`, or with `most` raises it to `n`, in
/// a build with the `counters` feature; nothing otherwise.
macro_rules! count {
    ($name:ident) => {
        count!($name, 1)
    };
    ($name:ident, $n:expr) => {
        #[cfg(feature = "counters")]
        crate::counters::add(&crate::counters::$name, $n);
    };
    (most $name:ident, $n:expr) => {
        #[cfg(feature = "counters")]
        crate::counters::most(&crate::counters::$name, $n);
    };
}

pub mod boxes;
pub mod composite;
pub mod contributors;
#[cfg(feature = "counters")]
pub mod counters;
pub mod depth;
pub mod diagnostics;
pub mod geometry;
pub mod gradient;
pub mod instance;
pub mod mesh;
pub mod order;
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
