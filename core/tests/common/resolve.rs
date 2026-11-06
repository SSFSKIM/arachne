//! An instrument for the resolves: a producer that sends a pixel's fragments to the sample resolve
//! without changing a byte of it, so a frame drawn with it beside the same frame drawn without it
//! shows how far each coverage-resolve pixel lands from the sample resolve of the same geometry.

use arachne_core::composite::{
    Coverage, FULL, Fragment, Plane, Producer, SAMPLE_OFFSETS, SAMPLES, TransparentPass,
};

/// Two fully transparent fragments over every pixel, one covering the sub-samples left of the
/// pixel's centre and one those right of it (none sits on the centre's column), both in front of
/// everything: a pixel holding two partial masks takes the sample resolve, where blending a
/// transparent fragment changes nothing. Not every drawn pixel: content ordered apart from depth
/// (fixed in frame, overlay) is not reached, and its pixels can stay on the fast or coverage
/// resolve. A measurement checks the instrument's frame took the sample resolve where it reads it.
pub struct EverySample;

/// The sub-samples left of the pixel's centre.
fn left() -> u16 {
    (0..SAMPLES)
        .filter(|s| SAMPLE_OFFSETS[*s][0] < 0)
        .fold(0, |m, s| m | 1 << s)
}

impl Producer for EverySample {
    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        let r = pass.rect();
        let front = Plane::constant(f64::NEG_INFINITY);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                for (sub, mask) in [left(), FULL & !left()].into_iter().enumerate() {
                    let f =
                        Fragment::new(front, [0; 4], Coverage::Mask(mask), u32::MAX, sub as u32);
                    pass.add(x, y, f);
                }
            }
        }
    }
}
