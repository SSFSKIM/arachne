//! What one frame stacks: layers in drawing order, bottom first, each with its effects.

use crate::keys::Key;

/// PIL's value for nearest resampling: a quad's one resampling, and a window's by default.
pub const NEAREST: u8 = 0;
/// PIL's value for bilinear resampling, which a window may be placed with (a quad may not): each
/// pixel takes the four texels round its centre, weighted in fixed point, with transparent texels
/// past the window's edge, so the edge is antialiased and a window moved by a fraction of a pixel
/// moves by that fraction.
pub const BILINEAR: u8 = 2;

/// One frame: a background colour under every layer, then the layers, bottom first. `focus`, when
/// set, blurs every layer with a declared depth by its distance from the focus.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// The colour under every layer, painted RGBA.
    pub background: [u8; 4],
    pub layers: Vec<Layer>,
    pub focus: Option<Focus>,
}

impl Frame {
    /// A frame with no focus.
    pub fn new(background: [u8; 4], layers: Vec<Layer>) -> Self {
        Frame {
            background,
            layers,
            focus: None,
        }
    }

    pub fn with_focus(mut self, depth: f64, aperture: f64) -> Self {
        self.focus = Some(Focus { depth, aperture });
        self
    }
}

/// One layer: its texture, placed by its source, changed by its effects, scaled by its opacity,
/// stacked over what is below.
///
/// The effects apply in this order: the placed layer is blurred (by its own blur and by its depth
/// under the frame's focus, as one Gaussian), then multiplied by its soft mask, then scaled by its
/// opacity and stacked.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub source: Source,
    /// In 0..=1; the layer's values are scaled by it (as 8-bit, rounded) before stacking.
    pub opacity: f64,
    /// The layer's declared depth; blurred under the frame's focus, and nothing without one.
    pub depth: Option<f64>,
    pub blur: Option<Blur>,
    pub mask: Option<SoftMask>,
    /// The caller's stable id of the object this layer draws; the object map reports a footprint
    /// for each layer that has one.
    pub object_id: Option<String>,
}

/// Where a layer's texture comes from and where it goes.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// A texture of the frame's size, of either kind, covering the frame.
    Picture { key: Key },
    /// A texture placed at four corners in pixels (x right, y down), in CE's image order: upper
    /// left, upper right, lower left, lower right; resampled by PIL's `resampling` value.
    Quad {
        key: Key,
        corners: [[f64; 2]; 4],
        resampling: u8,
    },
    /// Another stack, composed as a frame of its own of `size` (width, height) pixels, or of the
    /// size of the frame that holds it when `size` is `None`, then placed at four corners as a
    /// quad is. Its pictures and masks are of its size, and its effects run on textures of its
    /// size, so they stay inside it. Its stack is drawn as any frame is, on 8-bit integers. Placed
    /// by [`NEAREST`] its texels are copied; by [`BILINEAR`] four are weighted in fixed point per
    /// pixel. At whole-pixel corners spanning exactly its size, either is a texel-for-texel copy.
    Window {
        frame: Box<Frame>,
        corners: [[f64; 2]; 4],
        resampling: u8,
        size: Option<[u32; 2]>,
    },
}

/// A Gaussian blur of standard deviation `sigma_px`, in pixels.
///
/// It runs on integers: fixed-point weights summing to exactly one, a pass along rows then one
/// along columns, with 8 fractional bits kept in 32-bit values between them. The layer is rounded
/// back to 8 bits once, half up, after its mask has multiplied it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blur {
    pub sigma_px: f64,
}

/// A picture (a texture of the frame's size, of either kind) whose alpha, feathered by
/// `feather_px`, multiplies the layer; `invert` keeps what it would remove.
///
/// Only the mask's alpha masks. The feather is a Gaussian of sigma `feather_px / 3`, run along
/// rows then columns as a [`Blur`] is, on integers, into 32-bit values with 8 fractional bits; its
/// taps reach the whole pixels within `feather_px` along each axis. So its cutoff is per axis: a
/// pixel whose centre is farther than `feather_px` from every pixel across the mask's edge, along
/// both axes, is kept or removed whole. The product of the blurred layer and the feathered alpha
/// is the one rounding back to 8 bits.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftMask {
    pub key: Key,
    pub feather_px: f64,
    pub invert: bool,
}

/// The depth in focus and how fast a layer blurs away from it: a layer at depth `d` is blurred
/// with `sigma_px = aperture * |d - depth|`, combined with its own [`Blur`] as the root of the sum
/// of their squares into one Gaussian, which runs and rounds as a blur does. A layer at the
/// focus depth is not defocused; its own blur still applies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Focus {
    pub depth: f64,
    pub aperture: f64,
}

impl Layer {
    fn of(source: Source) -> Self {
        Layer {
            source,
            opacity: 1.0,
            depth: None,
            blur: None,
            mask: None,
            object_id: None,
        }
    }

    /// A picture layer at full opacity, with no effect.
    pub fn picture(key: Key) -> Self {
        Layer::of(Source::Picture { key })
    }

    /// A quad layer at full opacity, with no effect.
    pub fn quad(key: Key, corners: [[f64; 2]; 4], resampling: u8) -> Self {
        Layer::of(Source::Quad {
            key,
            corners,
            resampling,
        })
    }

    /// A window onto `frame`, composed at the size of the frame that holds it, at full opacity,
    /// with no effect.
    pub fn window(frame: Frame, corners: [[f64; 2]; 4], resampling: u8) -> Self {
        Layer::of(Source::Window {
            frame: Box::new(frame),
            corners,
            resampling,
            size: None,
        })
    }

    /// A window onto `frame`, composed as a frame of `size` (width, height) pixels, at full
    /// opacity, with no effect.
    pub fn sized_window(
        frame: Frame,
        corners: [[f64; 2]; 4],
        resampling: u8,
        size: [u32; 2],
    ) -> Self {
        Layer::of(Source::Window {
            frame: Box::new(frame),
            corners,
            resampling,
            size: Some(size),
        })
    }

    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity;
        self
    }

    pub fn with_blur(mut self, sigma_px: f64) -> Self {
        self.blur = Some(Blur { sigma_px });
        self
    }

    pub fn with_depth(mut self, depth: f64) -> Self {
        self.depth = Some(depth);
        self
    }

    pub fn with_mask(mut self, key: Key, feather_px: f64, invert: bool) -> Self {
        self.mask = Some(SoftMask {
            key,
            feather_px,
            invert,
        });
        self
    }

    /// The layer, naming the object it draws.
    pub fn named(mut self, object_id: impl Into<String>) -> Self {
        self.object_id = Some(object_id.into());
        self
    }
}
