//! What one frame stacks: layers in drawing order, bottom first.

use crate::keys::Key;

/// PIL's value for nearest resampling, the one resampling this build draws.
pub const NEAREST: u8 = 0;

/// One frame: a background colour under every layer, then the layers, bottom first.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// The colour under every layer, painted RGBA.
    pub background: [u8; 4],
    pub layers: Vec<Layer>,
}

/// One layer: its texture, placed by its source, scaled by its opacity, stacked over what is below.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub source: Source,
    /// In 0..=1; the layer's values are scaled by it (as 8-bit, rounded) before stacking.
    pub opacity: f64,
}

/// Where a layer's texture goes.
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
}

impl Layer {
    /// A picture layer at full opacity.
    pub fn picture(key: Key) -> Self {
        Layer {
            source: Source::Picture { key },
            opacity: 1.0,
        }
    }

    /// A quad layer at full opacity.
    pub fn quad(key: Key, corners: [[f64; 2]; 4], resampling: u8) -> Self {
        Layer {
            source: Source::Quad {
                key,
                corners,
                resampling,
            },
            opacity: 1.0,
        }
    }

    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity;
        self
    }
}

impl Source {
    pub fn key(&self) -> &Key {
        match self {
            Source::Picture { key } | Source::Quad { key, .. } => key,
        }
    }
}
