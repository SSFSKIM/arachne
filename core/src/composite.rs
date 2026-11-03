//! The frame buffer, and the one place colour becomes frame bytes.
//!
//! Frames have the layout of CE's `camera.pixel_array`: `height * width * 4` bytes, row-major, in
//! R, G, B, A order (CE hands Cairo its colours with red and blue swapped, so Cairo's native ARGB32
//! words land in memory as R, G, B, A), with paint premultiplied as Cairo stores it and the
//! background as CE writes it.
//!
//! This module is where blending will live: 2D paint blended in sRGB-encoded space as Cairo's
//! ARGB32 blends it, lit colour computed in linear light and then encoded, emitted light kept in its
//! own linear buffer and tone-mapped once, the opaque depth buffer readable before the transparent
//! pass resolves, and the transparent pass taking fragments from any producer through one call.
//! Today a frame is its background only.

/// A frame being drawn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl Frame {
    /// A frame of one colour.
    pub fn filled(width: u32, height: u32, pixel: [u8; 4]) -> Frame {
        let n = width as usize * height as usize;
        let mut rgba = Vec::with_capacity(n * 4);
        for _ in 0..n {
            rgba.extend_from_slice(&pixel);
        }
        Frame {
            width,
            height,
            rgba,
        }
    }

    /// The top-left `width x height` of an image (straight RGBA, row-major), as CE crops a
    /// background image (`np.array(image)[:height, :width]`); `None` when the image is smaller.
    pub fn cropped(
        width: u32,
        height: u32,
        image_width: u32,
        image_height: u32,
        image: &[u8],
    ) -> Option<Frame> {
        if image_width < width || image_height < height {
            return None;
        }
        let row = width as usize * 4;
        let stride = image_width as usize * 4;
        let mut rgba = Vec::with_capacity(row * height as usize);
        for y in 0..height as usize {
            rgba.extend_from_slice(&image[y * stride..y * stride + row]);
        }
        Some(Frame {
            width,
            height,
            rgba,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The frame's bytes.
    pub fn into_rgba(self) -> Vec<u8> {
        self.rgba
    }
}

/// A background colour's bytes as CE writes them: `(rgba * 255).astype(int)` stored into uint8,
/// truncated toward zero and wrapped as NumPy's cast wraps, not premultiplied.
pub fn background_pixel(rgba: [f64; 4]) -> [u8; 4] {
    rgba.map(|c| (c * 255.0) as i64 as u8)
}
