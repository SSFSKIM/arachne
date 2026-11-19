//! The compositor's object map: which pixels each named layer's object covers.

/// Where one object covers the frame, as the compositor draws it.
///
/// `mask` covers the pixels where the layer, drawn with no effect, puts any alpha: its placed
/// texels scaled by its opacity, as it would be stacked. A layer's effects spread it by at most
/// `grow_px` pixels in each direction, so drawing the frame without the layer changes pixels only
/// inside the footprint grown by `grow_px` (taken to the next whole pixel, as a square). Objects
/// overlap: each keeps its whole footprint, whatever covers it.
#[derive(Clone, Debug, PartialEq)]
pub struct Footprint {
    /// The caller's stable id for the object.
    pub object_id: String,
    /// The index of the layer that draws it, in the frame's layers.
    pub layer: usize,
    /// `[x0, y0, x1, y1]` in pixels, half-open, y down: the smallest box holding every covered
    /// pixel; all zero when none is covered.
    pub r#box: [u32; 4],
    /// True where covered, row by row over the box: `(y1 - y0) * (x1 - x0)` values.
    pub mask: Vec<bool>,
    /// How far the layer's effects spread it, in pixels: three sigma of its blur, or its mask's
    /// feather, whichever is larger.
    pub grow_px: f64,
}

impl Footprint {
    /// Whether the object covers the pixel at (`x`, `y`).
    pub fn covers(&self, x: u32, y: u32) -> bool {
        let [x0, y0, x1, y1] = self.r#box;
        (x0..x1).contains(&x)
            && (y0..y1).contains(&y)
            && self.mask[((y - y0) * (x1 - x0) + (x - x0)) as usize]
    }
}

/// The footprint of the object whose coverage is bit `bit` of a coverage frame: 32 bits a pixel,
/// in its four 8-bit channels, bit `b` in channel `b / 8` as `1 << (b % 8)`.
pub(crate) fn from_coverage(
    coverage: &[u8],
    width: u32,
    bit: usize,
    object_id: String,
    layer: usize,
    grow_px: f64,
) -> Footprint {
    let (channel, flag) = (bit / 8, 1u8 << (bit % 8));
    let covered = |x: u32, y: u32| coverage[((y * width + x) * 4) as usize + channel] & flag != 0;
    let height = (coverage.len() / 4) as u32 / width;
    let mut bounds: Option<[u32; 4]> = None;
    for y in 0..height {
        for x in 0..width {
            if covered(x, y) {
                let b = bounds.get_or_insert([x, y, x + 1, y + 1]);
                *b = [b[0].min(x), b[1].min(y), b[2].max(x + 1), b[3].max(y + 1)];
            }
        }
    }
    let r#box = bounds.unwrap_or([0; 4]);
    let [x0, y0, x1, y1] = r#box;
    let mask = (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .map(|(x, y)| covered(x, y))
        .collect();
    Footprint {
        object_id,
        layer,
        r#box,
        mask,
        grow_px,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_footprint_is_the_box_and_mask_of_its_own_bit() {
        let (width, height) = (6u32, 4u32);
        let mut coverage = vec![0u8; (width * height * 4) as usize];
        let mut set = |x: u32, y: u32, bit: usize| {
            coverage[((y * width + x) * 4) as usize + bit / 8] |= 1 << (bit % 8);
        };
        // Bit 9 (channel 1): an L of three pixels; bit 31 (channel 3): one pixel; bit 0: none.
        set(1, 1, 9);
        set(1, 2, 9);
        set(2, 2, 9);
        set(5, 3, 31);
        let l = from_coverage(&coverage, width, 9, "l".into(), 4, 2.5);
        assert_eq!(l.r#box, [1, 1, 3, 3]);
        assert_eq!(l.mask, vec![true, false, true, true]);
        assert_eq!((l.layer, l.grow_px), (4, 2.5));
        assert!(l.covers(2, 2) && !l.covers(2, 1) && !l.covers(5, 3));
        let dot = from_coverage(&coverage, width, 31, "dot".into(), 0, 0.0);
        assert_eq!((dot.r#box, dot.mask), ([5, 3, 6, 4], vec![true]));
        let none = from_coverage(&coverage, width, 0, "none".into(), 1, 0.0);
        assert_eq!((none.r#box, none.mask.len()), ([0; 4], 0));
        assert!(!none.covers(0, 0));
    }
}
