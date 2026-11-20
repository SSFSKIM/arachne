//! The four effects: a blur, a depth under a focus, a soft mask and a window each do what their
//! declared numbers say, shown by closed-form probes.

mod common;

use arachne_gpu::{DrawError, Frame, Layer, NEAREST};
use common::{compositor, flat, max_difference, opaque, painted_picture, pixel, rectangle, upload};

const W: u32 = 480;
const H: u32 = 270;
const CLEAR: [u8; 4] = [0; 4];

/// The distance between the 10% and 90% levels of a Gaussian-blurred step: 2 x 1.2816 sigma.
const TEN_TO_NINETY: f64 = 2.563;

/// An opaque picture, white left of column `edge` and black from it on.
fn step(edge: u32) -> Vec<u8> {
    (0..W * H)
        .flat_map(|i| {
            if i % W < edge {
                [255, 255, 255, 255]
            } else {
                [0, 0, 0, 255]
            }
        })
        .collect()
}

/// The red values of row `y`.
fn row(rgba: &[u8], y: u32) -> Vec<f64> {
    (0..W).map(|x| f64::from(pixel(rgba, W, x, y)[0])).collect()
}

/// Where a falling row first drops below `level`, in pixels (pixel x's centre is x + 0.5),
/// interpolated linearly between pixel centres.
fn crossing(values: &[f64], level: f64) -> f64 {
    values
        .windows(2)
        .position(|pair| pair[0] >= level && pair[1] < level)
        .map(|x| {
            let (a, b) = (values[x], values[x + 1]);
            x as f64 + 0.5 + (a - level) / (a - b)
        })
        .unwrap_or_else(|| panic!("the row never falls below {level}"))
}

/// The 10-to-90 width of a falling edge.
fn edge_width(values: &[f64]) -> f64 {
    crossing(values, 0.1 * 255.0) - crossing(values, 0.9 * 255.0)
}

/// Asserts that every row of `drawn` has the 10-to-90 width a blur of `sigma` declares, within
/// 1 px, and that it stays an opaque grey.
fn assert_step_width(drawn: &[u8], sigma: f64) {
    for y in [0, H / 2, H - 1] {
        let width = edge_width(&row(drawn, y));
        let declared = TEN_TO_NINETY * sigma;
        assert!(
            (width - declared).abs() <= 1.0,
            "sigma {sigma}, row {y}: width {width}, declared {declared}"
        );
    }
    assert!(
        drawn
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[3] == 255 && p[0] == p[1] && p[1] == p[2]),
        "sigma {sigma}: a blurred opaque grey is no longer one"
    );
}

#[test]
fn a_blurred_step_edge_has_the_declared_width() {
    let mut c = compositor(W, H);
    let key = upload(&mut c, W, H, &step(240));
    for sigma in [2.0, 6.0, 12.0] {
        let drawn = c
            .draw(&Frame::new(
                CLEAR,
                vec![Layer::picture(key).with_blur(sigma)],
            ))
            .unwrap();
        assert_step_width(&drawn, sigma);
    }
}

#[test]
fn a_blur_leaves_a_constant_picture_and_this_pictures_mean_as_they_were() {
    // The mean holds for this fixture, whose border is like its interior. It is not kept for
    // every picture: the edge value continued past the border counts more than once.
    let mut c = compositor(W, H);
    let constant = flat(W, H, [90, 120, 30, 200]);
    let picture = painted_picture(W, H, 5);
    for (rgba, sigma) in [(&constant, 6.0), (&picture, 6.0), (&picture, 12.0)] {
        let key = upload(&mut c, W, H, rgba);
        let drawn = c
            .draw(&Frame::new(
                CLEAR,
                vec![Layer::picture(key).with_blur(sigma)],
            ))
            .unwrap();
        if rgba == &constant {
            assert!(max_difference(&drawn, rgba) <= 1);
        }
        for channel in 0..4 {
            let mean = |pixels: &[u8]| {
                let values = pixels.iter().skip(channel).step_by(4);
                values.map(|&v| f64::from(v)).sum::<f64>() / f64::from(W * H)
            };
            let (before, after) = (mean(rgba.as_slice()), mean(drawn.as_slice()));
            assert!(
                (before - after).abs() <= 1.0,
                "sigma {sigma}, channel {channel}: mean {before} became {after}"
            );
        }
    }
}

#[test]
fn a_layer_at_the_focus_returns_its_bytes_and_one_off_it_has_its_resolved_width() {
    let mut c = compositor(W, H);
    let picture = painted_picture(W, H, 9);
    let key = upload(&mut c, W, H, &picture);
    let in_focus =
        Frame::new(CLEAR, vec![Layer::picture(key).with_depth(3.0)]).with_focus(3.0, 1.5);
    assert!(c.draw(&in_focus).unwrap() == picture);

    let edge = upload(&mut c, W, H, &step(240));
    // 1.5 * |7 - 3| = 6.
    let off_focus =
        Frame::new(CLEAR, vec![Layer::picture(edge).with_depth(7.0)]).with_focus(3.0, 1.5);
    assert_step_width(&c.draw(&off_focus).unwrap(), 6.0);
    // Its own blur of 3 and a defocus of 2 * |5 - 3| = 4 make one of 5.
    let both = Frame::new(
        CLEAR,
        vec![Layer::picture(edge).with_depth(5.0).with_blur(3.0)],
    )
    .with_focus(3.0, 2.0);
    assert_step_width(&c.draw(&both).unwrap(), 5.0);
}

/// The standard normal distribution: Abramowitz and Stegun's 7.1.26 for erf, within 1.5e-7.
fn phi(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-x * x).exp();
    0.5 * (1.0 + erf.copysign(z))
}

/// A mask picture, painted: kept where `keep(x, y)`, with colour values unlike its alpha, since
/// only the alpha masks.
fn mask_picture(keep: impl Fn(u32, u32) -> bool) -> Vec<u8> {
    (0..W * H)
        .flat_map(|i| {
            if keep(i % W, i / W) {
                [40, 0, 90, 255]
            } else {
                [0; 4]
            }
        })
        .collect()
}

#[test]
fn a_masks_feather_is_the_gaussian_it_declares_along_each_axis() {
    // White over black, so the red channel reads the feathered alpha: a Gaussian of sigma
    // feather / 3 across the mask's edge, the closed form within 2 of 255. A mask with no
    // feather would read 255 and 0 here.
    let mut c = compositor(W, H);
    let feather = 10.0;
    let sigma = feather / 3.0;
    let white = upload(&mut c, W, H, &flat(W, H, [255; 4]));
    let black = upload(&mut c, W, H, &flat(W, H, [0, 0, 0, 255]));
    let (column, line) = (200u32, 120u32);
    let across_columns = upload(&mut c, W, H, &mask_picture(|x, _| x < column));
    let across_rows = upload(&mut c, W, H, &mask_picture(|_, y| y < line));
    for invert in [false, true] {
        for (mask, edge, along_x) in [(across_columns, column, true), (across_rows, line, false)] {
            let drawn = c
                .draw(&Frame::new(
                    CLEAR,
                    vec![
                        Layer::picture(black),
                        Layer::picture(white).with_mask(mask, feather, invert),
                    ],
                ))
                .unwrap();
            // Pixel centres at 0.5 to 11.5 px either side of the edge.
            for step in 0..12u32 {
                for (at, kept_side) in [(edge - 1 - step, true), (edge + step, false)] {
                    let distance = f64::from(step) + 0.5;
                    let kept = phi(if kept_side { distance } else { -distance } / sigma);
                    let expected = 255.0 * if invert { 1.0 - kept } else { kept };
                    let (x, y) = if along_x { (at, H / 2) } else { (W / 2, at) };
                    let found = f64::from(pixel(&drawn, W, x, y)[0]);
                    assert!(
                        (found - expected).abs() <= 2.0,
                        "invert {invert}, along x {along_x}, ({x}, {y}): {found}, closed form \
                         {expected}"
                    );
                }
            }
        }
    }
}

/// The distance from pixel (`x`, `y`)'s centre to the nearest pixel on the other side of a mask,
/// per axis (the larger of the two axis distances), searched up to `reach` pixels: k - 0.5 for
/// the nearest square ring k holding one; infinite when none is that near.
fn per_axis_distance(inside: &impl Fn(u32, u32) -> bool, x: u32, y: u32, reach: i64) -> f64 {
    let side = inside(x, y);
    let other = |i: i64, j: i64| {
        let (px, py) = (i64::from(x) + i, i64::from(y) + j);
        (0..i64::from(W)).contains(&px)
            && (0..i64::from(H)).contains(&py)
            && inside(px as u32, py as u32) != side
    };
    (1..=reach)
        .find(|&k| (-k..=k).any(|t| other(t, -k) || other(t, k) || other(-k, t) || other(k, t)))
        .map_or(f64::INFINITY, |k| k as f64 - 0.5)
}

#[test]
fn a_feather_reaches_its_width_per_axis_on_an_edge_of_any_direction() {
    // A disc: its edge runs in every direction. A pixel whose centre is farther than the feather
    // from every pixel across the edge, along both axes, is kept or removed whole; nearer, it may
    // be partly kept, even where it is farther than the feather in a straight line.
    let mut c = compositor(W, H);
    let feather = 8.0;
    let inside = |x: u32, y: u32| {
        let (dx, dy) = (f64::from(x) + 0.5 - 240.0, f64::from(y) + 0.5 - 135.0);
        dx * dx + dy * dy < 70.0 * 70.0
    };
    let mask = upload(&mut c, W, H, &mask_picture(inside));
    let below = opaque(painted_picture(W, H, 21));
    let layer = opaque(painted_picture(W, H, 22));
    let (kb, kl) = (upload(&mut c, W, H, &below), upload(&mut c, W, H, &layer));
    for invert in [false, true] {
        let drawn = c
            .draw(&Frame::new(
                CLEAR,
                vec![
                    Layer::picture(kb),
                    Layer::picture(kl).with_mask(mask, feather, invert),
                ],
            ))
            .unwrap();
        let mut partial = 0;
        for y in 0..H {
            for x in 0..W {
                let found = pixel(&drawn, W, x, y);
                let (kept, removed) = (pixel(&layer, W, x, y), pixel(&below, W, x, y));
                if per_axis_distance(&inside, x, y, 10) > feather {
                    let expected = if inside(x, y) != invert {
                        kept
                    } else {
                        removed
                    };
                    assert_eq!(found, expected, "invert {invert}, ({x}, {y})");
                } else if found != kept && found != removed {
                    partial += 1;
                }
            }
        }
        assert!(partial > 0, "invert {invert}: no pixel is partly kept");
    }
}

/// A stack of its own: an opaque background under a half-transparent picture and a quad.
fn inner_stack(c: &mut arachne_gpu::Compositor) -> Frame {
    let picture = painted_picture(W, H, 31);
    let texture = opaque(painted_picture(24, 16, 32));
    let kp = upload(c, W, H, &picture);
    let kt = upload(c, 24, 16, &texture);
    Frame::new(
        [20, 40, 60, 255],
        vec![
            Layer::picture(kp),
            Layer::quad(kt, rectangle(50.0, 30.0, 122.0, 78.0), NEAREST),
        ],
    )
}

#[test]
fn a_window_at_the_frames_corners_is_its_stack_composed_alone() {
    let mut c = compositor(W, H);
    let inner = inner_stack(&mut c);
    let alone = c.draw(&inner).unwrap();
    let full = rectangle(0.0, 0.0, f64::from(W), f64::from(H));
    let once = Frame::new(CLEAR, vec![Layer::window(inner.clone(), full, NEAREST)]);
    assert!(max_difference(&c.draw(&once).unwrap(), &alone) <= 1);
    // A window inside a window, too.
    let twice = Frame::new(CLEAR, vec![Layer::window(once, full, NEAREST)]);
    assert!(max_difference(&c.draw(&twice).unwrap(), &alone) <= 1);
}

#[test]
fn a_window_at_a_smaller_rectangle_leaves_the_parent_outside_it() {
    let mut c = compositor(W, H);
    let inner = inner_stack(&mut c);
    let alone = c.draw(&inner).unwrap();
    let parent = opaque(painted_picture(W, H, 41));
    let kp = upload(&mut c, W, H, &parent);
    let parent_alone = c
        .draw(&Frame::new([5, 6, 7, 255], vec![Layer::picture(kp)]))
        .unwrap();
    // A third of the frame's size, so every pixel centre falls in a texel's middle third.
    let (x0, y0) = (100u32, 60u32);
    let (x1, y1) = (x0 + W / 3, y0 + H / 3);
    let corners = rectangle(f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1));
    let drawn = c
        .draw(&Frame::new(
            [5, 6, 7, 255],
            vec![Layer::picture(kp), Layer::window(inner, corners, NEAREST)],
        ))
        .unwrap();
    for y in 0..H {
        for x in 0..W {
            let expected = if (x0..x1).contains(&x) && (y0..y1).contains(&y) {
                pixel(&alone, W, 3 * (x - x0) + 1, 3 * (y - y0) + 1)
            } else {
                pixel(&parent_alone, W, x, y)
            };
            assert_eq!(pixel(&drawn, W, x, y), expected, "({x}, {y})");
        }
    }
}

#[test]
fn an_effect_out_of_its_range_refuses_the_frame_whole() {
    let mut c = compositor(W, H);
    let key = upload(&mut c, W, H, &painted_picture(W, H, 51));
    let small = upload(&mut c, 8, 8, &flat(8, 8, [255; 4]));
    let refused = [
        Layer::picture(key).with_blur(-1.0),
        Layer::picture(key).with_blur(f64::NAN),
        Layer::picture(key).with_mask(small, 2.0, false),
        Layer::picture(key).with_mask(key, f64::INFINITY, false),
    ];
    for layer in refused {
        assert!(
            matches!(
                c.draw(&Frame::new(CLEAR, vec![Layer::picture(key), layer.clone()])),
                Err(DrawError::Invalid(_))
            ),
            "{layer:?}"
        );
    }
    // Sigma 161 reaches 483 px, past the frame's 480.
    assert!(matches!(
        c.draw(&Frame::new(
            CLEAR,
            vec![Layer::picture(key).with_blur(161.0)]
        )),
        Err(DrawError::Unsupported(_))
    ));
}
