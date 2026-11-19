//! Content keys of the two kinds of texture, and the one conversion between them.

/// A content key: BLAKE2b with a 16-byte digest.
pub type Key = [u8; 16];

/// The Frame IR's kind code of an image shape (`core/XF.md`, "Content keys").
const IMAGE_SHAPE: u8 = 3;

/// The leading tag of a painted picture's key. The Frame IR's kind codes count up from 1 and never
/// reach it, so no painted picture's key is ever an IR key.
pub const PAINTED_TAG: u8 = 255;

/// The canonical bytes of the Frame IR's content keys, for one u8 array of shape
/// (height, width, 4): the tag, the number of arrays, the array's dtype code (3, u8), its number of
/// dimensions, each dimension as u64, then the data, all little-endian.
fn rgba_key(tag: u8, width: u32, height: u32, rgba: &[u8]) -> Key {
    let mut state = blake2b_simd::Params::new().hash_length(16).to_state();
    state.update(&[tag, 1, 3, 3]);
    for dimension in [u64::from(height), u64::from(width), 4] {
        state.update(&dimension.to_le_bytes());
    }
    state.update(rgba);
    state
        .finalize()
        .as_bytes()
        .try_into()
        .expect("a 16-byte digest")
}

/// An image's key: the Frame IR's image-shape key over its straight RGBA bytes.
pub fn image_key(width: u32, height: u32, rgba: &[u8]) -> Key {
    rgba_key(IMAGE_SHAPE, width, height, rgba)
}

/// A painted picture's key, over its painted RGBA bytes.
pub fn painted_key(width: u32, height: u32, rgba: &[u8]) -> Key {
    rgba_key(PAINTED_TAG, width, height, rgba)
}

/// Straight RGBA to painted values: `c' = (c * a + 127) / 255` per colour channel, alpha kept. The
/// only place an image's values change, in integer arithmetic, so the same on every adapter.
pub fn painted_from_straight(rgba: &[u8]) -> Vec<u8> {
    let mut painted = rgba.to_vec();
    for pixel in painted.as_chunks_mut::<4>().0 {
        let alpha = u32::from(pixel[3]);
        for channel in &mut pixel[..3] {
            *channel = ((u32::from(*channel) * alpha + 127) / 255) as u8;
        }
    }
    painted
}
