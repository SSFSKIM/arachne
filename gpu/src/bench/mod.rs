//! Tools for measuring this build on a GPU host, never part of its drawing API: the frame set
//! ([`frameset`]), a fixed list of frames drawn by code and archived by their digests for each
//! device a scene may be drawn on, and the read-back benchmark ([`timing`]), which times the
//! reference stacks with the one-frame copy `draw` makes and with a pipelined copy that takes one
//! frame out while the next draws.
//!
//! Compiled with the `bench` feature (`examples/frameset.rs`, `examples/readback_bench.rs`) and
//! in the crate's own tests. The caller opens every compositor, so these tools never choose a
//! device: the examples take the ordinary door, and the tests the testing door.

pub mod frameset;
mod json;
mod sha256;
pub mod stacks;
pub mod timing;

pub use sha256::hex_digest;

/// What a tool's run can fail with.
pub type Error = Box<dyn std::error::Error>;

/// Opens a compositor of a given frame size; the caller's choice of door.
pub type Open<'a> = dyn FnMut(u32, u32) -> Result<crate::Compositor, crate::OpenError> + 'a;
