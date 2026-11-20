//! The one door to a software adapter, for tests only: no draft and no final is ever drawn on one.
//! Nothing outside test code calls it.

use crate::compositor::{Compositor, OpenError};
use crate::device::Door;

/// A compositor on a software (CPU-type) adapter, such as Mesa's lavapipe. Its `software()` is
/// true. Fails with [`OpenError::NoGpu`] when the machine offers no software adapter.
pub fn software_compositor(width: u32, height: u32) -> Result<Compositor, OpenError> {
    Compositor::open(width, height, Door::Software)
}
