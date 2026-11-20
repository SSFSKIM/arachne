//! The one door to a software adapter, for tests only: no draft and no final is ever drawn on one;
//! and a count of a compositor's working textures. Nothing outside test code calls it.

use crate::compositor::{Compositor, OpenError};
use crate::device::Door;

/// A compositor on a software (CPU-type) adapter, such as Mesa's lavapipe. Its `software()` is
/// true. Fails with [`OpenError::NoGpu`] when the machine offers no software adapter.
pub fn software_compositor(width: u32, height: u32) -> Result<Compositor, OpenError> {
    Compositor::open(width, height, Door::Software)
}

/// The working textures a compositor holds: accumulators and effect scratch, of every level and
/// size of window it keeps them for (an uploaded picture is not one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkingTextures {
    /// Held now.
    pub live: usize,
    /// The most held at once since the compositor opened.
    pub peak: usize,
}

/// How many working textures `compositor` holds now, and has held at most.
pub fn working_textures(compositor: &Compositor) -> WorkingTextures {
    let (live, peak) = compositor.working_textures();
    WorkingTextures { live, peak }
}
