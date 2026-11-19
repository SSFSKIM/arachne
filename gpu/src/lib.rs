//! Arachne's GPU build: pictures stacked as layers on a GPU, and the frame read back.
//!
//! A [`Compositor`] opens a Vulkan device with no window and holds textures of two kinds, each
//! under a content key of its own domain:
//! - an **image**, straight RGBA as an image object holds it, keyed by the Frame IR's image-shape
//!   key ([`image_key`]) and converted to painted values once, on upload ([`painted_from_straight`]);
//! - a **painted picture**, colour already multiplied by alpha as Cairo leaves it, keyed by
//!   [`painted_key`].
//!
//! A [`Frame`] stacks layers in drawing order, bottom first, with the painted "over" rule on the
//! frame's 8-bit values, in Cairo's integer arithmetic, the same on every adapter; the frame comes
//! back painted, in the layout of CE's pixel array. A quad's placement is computed in floating
//! point, so at a texel's or the quad's edge two adapters may pick different texels.
//!
//! The ordinary constructor draws only on a GPU (a discrete, integrated or virtual adapter) and
//! fails with [`OpenError::NoGpu`] otherwise; the one door to a software adapter is
//! [`testing::software_compositor`], for tests.

mod compositor;
mod device;
mod frame;
mod keys;
mod placement;
pub mod testing;

#[cfg(feature = "python")]
mod python;

pub use compositor::{Compositor, DrawError, OpenError};
pub use device::{AdapterSummary, NoGpu, Pin, adapters};
pub use frame::{Frame, Layer, NEAREST, Source};
pub use keys::{Key, PAINTED_TAG, image_key, painted_from_straight, painted_key};

/// The version of wgpu this build is locked to: the pin's `wgpu` field.
pub const WGPU_VERSION: &str = env!("ARACHNE_GPU_WGPU_VERSION");
