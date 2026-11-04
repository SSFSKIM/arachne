//! An instrument for the resolves: a producer that sends every pixel to the sample resolve without
//! changing a byte of it, so a frame drawn with it beside the same frame drawn without it shows
//! how far each coverage-resolve pixel lands from the sample resolve of the same geometry.

use arachne_core::composite::{Coverage, FULL, Fragment, Plane, Producer, TransparentPass};

/// A fully transparent fragment over every pixel, its depth so steep that it is in front of every
/// other fragment at the sub-samples left of the pixel's centre and behind it right of the centre
/// (none sits on the centre's column): every pixel with anything drawn takes the sample resolve,
/// where blending a transparent fragment changes nothing.
pub struct EverySample;

const STEEP: f64 = 1e12;

impl Producer for EverySample {
    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        let r = pass.rect();
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let depth = Plane {
                    a: -STEEP * (x as f64 + 0.5),
                    b: STEEP,
                    c: 0.0,
                };
                let f = Fragment::new(depth, [0; 4], Coverage::Mask(FULL), u32::MAX, 0);
                pass.add(x, y, f);
            }
        }
    }
}
