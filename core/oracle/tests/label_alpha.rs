//! A gradient fill's alpha at a pixel, as the label rule reads it (contract §10): bounded by its
//! stops' alpha bytes, one level wider each way for the walker's single-precision rounding, the
//! record's group opacity applied as to a solid. Where the bound holds 0 and a positive byte the
//! pixel is `label_alpha_straddle`; elsewhere the rule decides.

#[path = "common/scenes.rs"]
mod scenes;

use arachne_core::xf::Paint;
use arachne_oracle::decide::decide_at;
use arachne_oracle::out::{ROLE_ALPHA_STRADDLE, ROLE_ANCHOR};
use scenes::*;

/// W5 with `G`'s fill the two-stop gradient `stops` across it, at group opacity `opacity`: the
/// roles `LABL` gives `G` at pixel (32, 32).
fn roles(stops: [[f64; 4]; 2], opacity: f64) -> Vec<u8> {
    let mut b = Build::new(true);
    let g = b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), OPAQUE_BLUE);
    g.paint.fill = Paint::Two(stops);
    g.paint.gradient = Some([[-3.0, 0.0, 2.0], [3.0, 0.0, 2.0]]);
    g.opacity = opacity;
    b.label([0.0; 3]);
    let d = decide_at(&[b.delta(1, view_c())], 1, None).unwrap();
    let g = d.fragments.iter().find(|f| f.slot == 0).unwrap().id;
    let r: Vec<u8> = d
        .labels
        .iter()
        .filter(|r| (r.x, r.y, r.fragment) == (32, 32, g))
        .map(|r| r.role)
        .collect();
    assert_eq!(r.len(), 16);
    r
}

const GREEN: [f64; 4] = [0.0, 1.0, 0.0, 1.0];

#[test]
fn an_opaque_gradient_decides_as_a_solid_does() {
    assert_eq!(roles([OPAQUE_BLUE, GREEN], 1.0), [ROLE_ANCHOR; 16]);
    // Stops at 1 and 1/2: bytes 255 and 127, never 0.
    assert_eq!(
        roles([OPAQUE_BLUE, [0.0, 1.0, 0.0, 0.5]], 1.0),
        [ROLE_ANCHOR; 16]
    );
}

#[test]
fn a_gradient_whose_alpha_bound_holds_zero_and_a_positive_byte_straddles() {
    // A clear stop: the bound runs from 0 to 255.
    assert_eq!(
        roles([OPAQUE_BLUE, [0.0, 1.0, 0.0, 0.0]], 1.0),
        [ROLE_ALPHA_STRADDLE; 16]
    );
    // A stop of byte 1 (short 262 > 0xff, so the layer paints): widened, the bound holds 0.
    let a = 262.0 / 65535.0;
    assert_eq!(
        roles([[0.0, 0.0, 1.0, a], [0.0, 1.0, 0.0, a]], 1.0),
        [ROLE_ALPHA_STRADDLE; 16]
    );
    // The group opacity applies as to a solid's: opaque stops at that opacity are the same.
    assert_eq!(roles([OPAQUE_BLUE, GREEN], a), [ROLE_ALPHA_STRADDLE; 16]);
}
