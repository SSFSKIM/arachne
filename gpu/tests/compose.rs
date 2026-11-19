//! Stacking and read-back: what a frame of layers with no effect draws.

mod common;

use arachne_gpu::{
    DrawError, Frame, Layer, NEAREST, image_key, painted_from_straight, painted_key,
};
use common::{compositor, flat, mul_un8, painted_picture};

const W: u32 = 480;
const H: u32 = 270;
const CLEAR: [u8; 4] = [0, 0, 0, 0];

fn frame(background: [u8; 4], layers: Vec<Layer>) -> Frame {
    Frame::new(background, layers)
}

/// Painted "over" in real arithmetic, rounded: the closed form a stack is held to.
fn over(top: &[u8], below: &[u8]) -> Vec<u8> {
    top.as_chunks::<4>()
        .0
        .iter()
        .zip(below.as_chunks::<4>().0)
        .flat_map(|(s, d)| {
            let keep = 1.0 - f64::from(s[3]) / 255.0;
            (0..4).map(move |i| (f64::from(s[i]) + f64::from(d[i]) * keep).round() as u8)
        })
        .collect()
}

fn max_difference(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

#[test]
fn a_painted_picture_alone_comes_back_with_its_bytes() {
    let mut c = compositor(W, H);
    let picture = painted_picture(W, H, 7);
    let key = painted_key(W, H, &picture);
    c.upload_painted(key, W, H, &picture).unwrap();
    let f = frame(CLEAR, vec![Layer::picture(key)]);
    let first = c.draw(&f).unwrap();
    assert!(first == picture, "a picture passed through changed");
    assert!(
        c.draw(&f).unwrap() == first,
        "a second draw differs from the first"
    );
}

#[test]
fn an_opaque_image_comes_back_with_its_bytes() {
    let mut c = compositor(W, H);
    let mut image = painted_picture(W, H, 11);
    for pixel in image.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    let key = image_key(W, H, &image);
    c.upload_image(key, W, H, &image).unwrap();
    assert!(
        c.draw(&frame([9, 8, 7, 255], vec![Layer::picture(key)]))
            .unwrap()
            == image
    );
}

#[test]
fn a_straight_image_is_painted_exactly_for_every_colour_and_alpha() {
    // Every pair of a colour value (x) and an alpha (y), saturated or not.
    let (w, h) = (256, 256);
    let mut c = compositor(w, h);
    let mut image = Vec::with_capacity(256 * 256 * 4);
    for alpha in 0..=255u8 {
        for colour in 0..=255u8 {
            image.extend_from_slice(&[colour, 255 - colour, 255, alpha]);
        }
    }
    let key = image_key(w, h, &image);
    c.upload_image(key, w, h, &image).unwrap();
    let drawn = c.draw(&frame(CLEAR, vec![Layer::picture(key)])).unwrap();
    for (pixel, straight) in drawn
        .as_chunks::<4>()
        .0
        .iter()
        .zip(image.as_chunks::<4>().0)
    {
        let a = u32::from(straight[3]);
        for i in 0..3 {
            let expected = ((u32::from(straight[i]) * a + 127) / 255) as u8;
            assert_eq!(pixel[i], expected, "colour {} at alpha {a}", straight[i]);
        }
        assert_eq!(pixel[3], straight[3]);
    }
    assert_eq!(painted_from_straight(&image), drawn);
}

#[test]
fn a_key_must_be_its_bytes_key_of_its_own_kind() {
    let mut c = compositor(W, H);
    let image = painted_picture(W, H, 3);
    let wrong = [
        painted_key(W, H, &image),
        image_key(W, H, &painted_picture(W, H, 4)),
    ];
    for key in wrong {
        assert!(matches!(
            c.upload_image(key, W, H, &image),
            Err(DrawError::Invalid(_))
        ));
    }
    assert!(matches!(
        c.upload_painted(image_key(W, H, &image), W, H, &image),
        Err(DrawError::Invalid(_))
    ));
}

#[test]
fn an_image_and_a_painted_picture_of_equal_bytes_are_two_textures() {
    let mut c = compositor(W, H);
    // Half-transparent, so painting changes the values.
    let bytes = flat(W, H, [200, 100, 50, 128]);
    let image = image_key(W, H, &bytes);
    let painted = painted_key(W, H, &bytes);
    assert_ne!(image, painted);
    c.upload_image(image, W, H, &bytes).unwrap();
    c.upload_painted(painted, W, H, &bytes).unwrap();
    let from_image = c.draw(&frame(CLEAR, vec![Layer::picture(image)])).unwrap();
    let from_painted = c
        .draw(&frame(CLEAR, vec![Layer::picture(painted)]))
        .unwrap();
    assert_eq!(&from_image[..4], &[100, 50, 25, 128]);
    assert_eq!(&from_painted[..4], &[200, 100, 50, 128]);
    // Each is released on its own.
    assert!(c.release(&image));
    assert!(c.draw(&frame(CLEAR, vec![Layer::picture(painted)])).is_ok());
    assert_eq!(
        c.draw(&frame(CLEAR, vec![Layer::picture(image)])),
        Err(DrawError::UnknownKey(image))
    );
}

#[test]
fn two_half_transparent_pictures_stack_over_in_their_order() {
    let mut c = compositor(W, H);
    let a = painted_picture(W, H, 21);
    let b = painted_picture(W, H, 22);
    let (ka, kb) = (painted_key(W, H, &a), painted_key(W, H, &b));
    c.upload_painted(ka, W, H, &a).unwrap();
    c.upload_painted(kb, W, H, &b).unwrap();
    let background = [30, 60, 90, 255];
    let base = flat(W, H, background);
    let ab = c
        .draw(&frame(
            background,
            vec![Layer::picture(ka), Layer::picture(kb)],
        ))
        .unwrap();
    let ba = c
        .draw(&frame(
            background,
            vec![Layer::picture(kb), Layer::picture(ka)],
        ))
        .unwrap();
    assert!(max_difference(&ab, &over(&b, &over(&a, &base))) <= 1);
    assert!(max_difference(&ba, &over(&a, &over(&b, &base))) <= 1);
    assert!(ab != ba, "the order of the layers changed nothing");
}

#[test]
fn a_quad_at_integer_corners_shows_its_texture_there_and_the_layer_below_elsewhere() {
    let mut c = compositor(W, H);
    let below = painted_picture(W, H, 31);
    let kb = painted_key(W, H, &below);
    c.upload_painted(kb, W, H, &below).unwrap();
    let (tw, th) = (40u32, 30u32);
    let mut texture = painted_picture(tw, th, 32);
    for pixel in texture.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    let kt = painted_key(tw, th, &texture);
    c.upload_painted(kt, tw, th, &texture).unwrap();
    // At its own size, and twice its size.
    for (x0, y0, scale) in [(100u32, 50u32, 1u32), (300, 120, 2)] {
        let (x1, y1) = (x0 + tw * scale, y0 + th * scale);
        let corners = [
            [f64::from(x0), f64::from(y0)],
            [f64::from(x1), f64::from(y0)],
            [f64::from(x0), f64::from(y1)],
            [f64::from(x1), f64::from(y1)],
        ];
        let drawn = c
            .draw(&frame(
                CLEAR,
                vec![Layer::picture(kb), Layer::quad(kt, corners, NEAREST)],
            ))
            .unwrap();
        for y in 0..H {
            for x in 0..W {
                let at = ((y * W + x) * 4) as usize;
                let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
                let expected = if inside {
                    let t = (((y - y0) / scale * tw + (x - x0) / scale) * 4) as usize;
                    &texture[t..t + 4]
                } else {
                    &below[at..at + 4]
                };
                assert_eq!(
                    &drawn[at..at + 4],
                    expected,
                    "pixel ({x}, {y}), scale {scale}"
                );
            }
        }
    }
}

#[test]
fn a_layers_opacity_scales_it_and_zero_leaves_the_frame_as_without_it() {
    let mut c = compositor(W, H);
    let below = painted_picture(W, H, 41);
    let top = painted_picture(W, H, 42);
    let (kb, kt) = (painted_key(W, H, &below), painted_key(W, H, &top));
    c.upload_painted(kb, W, H, &below).unwrap();
    c.upload_painted(kt, W, H, &top).unwrap();
    let background = [12, 34, 56, 255];
    let without = c
        .draw(&frame(background, vec![Layer::picture(kb)]))
        .unwrap();
    let zero = c
        .draw(&frame(
            background,
            vec![Layer::picture(kb), Layer::picture(kt).with_opacity(0.0)],
        ))
        .unwrap();
    assert!(zero == without, "opacity 0 changed the frame");
    let half = c
        .draw(&frame(
            background,
            vec![Layer::picture(kb), Layer::picture(kt).with_opacity(0.5)],
        ))
        .unwrap();
    let scaled: Vec<u8> = top.iter().map(|&v| mul_un8(v, 128)).collect();
    assert!(max_difference(&half, &over(&scaled, &without)) <= 1);
}

#[test]
fn a_resampling_this_build_does_not_draw_refuses_the_frame_whole() {
    let mut c = compositor(W, H);
    let texture = flat(8, 8, [255, 0, 0, 255]);
    let key = painted_key(8, 8, &texture);
    c.upload_painted(key, 8, 8, &texture).unwrap();
    let corners = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0]];
    for resampling in [1u8, 2, 3, 4, 5] {
        let refused = c.draw(&frame(
            CLEAR,
            vec![
                Layer::quad(key, corners, NEAREST),
                Layer::quad(key, corners, resampling),
            ],
        ));
        assert!(
            matches!(refused, Err(DrawError::Unsupported(_))),
            "resampling {resampling}: {refused:?}"
        );
    }
}

/// A frame of one picture under one quad of `texture` at `corners`.
fn picture_and_quad(
    c: &mut arachne_gpu::Compositor,
    below: &[u8],
    texture: (&[u8], u32, u32),
    corners: [[f64; 2]; 4],
) -> Result<Vec<u8>, DrawError> {
    let kb = painted_key(W, H, below);
    c.upload_painted(kb, W, H, below).unwrap();
    let (pixels, tw, th) = texture;
    let kt = painted_key(tw, th, pixels);
    c.upload_painted(kt, tw, th, pixels).unwrap();
    c.draw(&frame(
        CLEAR,
        vec![Layer::picture(kb), Layer::quad(kt, corners, NEAREST)],
    ))
}

#[test]
fn a_sheared_quad_shows_the_texel_under_each_pixel_centre() {
    // A parallelogram: its texel coordinates at pixel centres stay at least 0.05 from a texel
    // boundary, so the nearest texel is the same whatever the float rounding.
    let mut c = compositor(W, H);
    let below = painted_picture(W, H, 51);
    let (tw, th) = (40u32, 30u32);
    let mut texture = painted_picture(tw, th, 52);
    for pixel in texture.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    let (ul, ur, ll) = ([100.0, 50.0], [140.0, 50.0], [109.0, 80.0]);
    let lr = [ur[0] + ll[0] - ul[0], ll[1]];
    let drawn = picture_and_quad(&mut c, &below, (&texture, tw, th), [ul, ur, ll, lr]).unwrap();
    let (e1, e2) = (
        [ur[0] - ul[0], ur[1] - ul[1]],
        [ll[0] - ul[0], ll[1] - ul[1]],
    );
    let det = e1[0] * e2[1] - e2[0] * e1[1];
    let mut inside = 0;
    for y in 0..H {
        for x in 0..W {
            let (px, py) = (f64::from(x) + 0.5 - ul[0], f64::from(y) + 0.5 - ul[1]);
            let u = (px * e2[1] - e2[0] * py) / det * f64::from(tw);
            let v = (e1[0] * py - px * e1[1]) / det * f64::from(th);
            let at = ((y * W + x) * 4) as usize;
            let expected = if (0.0..f64::from(tw)).contains(&u) && (0.0..f64::from(th)).contains(&v)
            {
                inside += 1;
                let t = ((v as u32 * tw + u as u32) * 4) as usize;
                &texture[t..t + 4]
            } else {
                &below[at..at + 4]
            };
            assert_eq!(&drawn[at..at + 4], expected, "pixel ({x}, {y})");
        }
    }
    assert_eq!(inside, tw * th);
}

#[test]
fn a_perspective_quad_covers_its_outline() {
    // A trapezoid, placed by a projective map: every pixel centre clearly inside it shows the
    // texture, every one clearly outside the picture below.
    let mut c = compositor(W, H);
    let below = painted_picture(W, H, 61);
    let texture = flat(16, 16, [10, 200, 30, 255]);
    let (ul, ur, ll, lr) = ([300.0, 40.0], [380.0, 40.0], [270.0, 140.0], [410.0, 140.0]);
    let drawn = picture_and_quad(&mut c, &below, (&texture, 16, 16), [ul, ur, ll, lr]).unwrap();
    let outline = [ul, ur, lr, ll];
    let mut counted = [0, 0];
    for y in 0..H {
        for x in 0..W {
            let p = [f64::from(x) + 0.5, f64::from(y) + 0.5];
            // The signed distance to each edge, positive inside (the outline runs clockwise on
            // screen, y down).
            let nearest = (0..4)
                .map(|i| {
                    let (a, b) = (outline[i], outline[(i + 1) % 4]);
                    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                    ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / -(dx * dx + dy * dy).sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            let at = ((y * W + x) * 4) as usize;
            if nearest > 0.01 {
                counted[0] += 1;
                assert_eq!(
                    &drawn[at..at + 4],
                    &[10, 200, 30, 255],
                    "inside at ({x}, {y})"
                );
            } else if nearest < -0.01 {
                counted[1] += 1;
                assert_eq!(
                    &drawn[at..at + 4],
                    &below[at..at + 4],
                    "outside at ({x}, {y})"
                );
            }
        }
    }
    // The trapezoid's area, 100 * (80 + 140) / 2 pixels, holds 11,000 pixel centres, none of
    // them within 0.01 of its outline.
    assert_eq!(counted[0], 11_000);
}

#[test]
fn a_quad_of_area_with_no_placement_refuses_the_frame_and_one_of_no_area_draws_nothing() {
    let mut c = compositor(W, H);
    let below = painted_picture(W, H, 71);
    let texture = flat(8, 8, [255, 0, 0, 255]);
    // Upper left, upper right and lower right on one line, the lower left off it: an outline
    // with area, onto which no projective map takes a square.
    let collinear = [[10.0, 10.0], [50.0, 10.0], [10.0, 60.0], [90.0, 10.0]];
    assert!(matches!(
        picture_and_quad(&mut c, &below, (&texture, 8, 8), collinear),
        Err(DrawError::Unsupported(_))
    ));
    // All four on one line: no area, no pixel.
    let flat_line = [[10.0, 10.0], [50.0, 10.0], [20.0, 10.0], [60.0, 10.0]];
    let drawn = picture_and_quad(&mut c, &below, (&texture, 8, 8), flat_line).unwrap();
    assert!(drawn == below, "a quad of no area changed the frame");
}

#[test]
fn a_crossed_or_concave_quad_refuses_the_frame() {
    let mut c = compositor(W, H);
    let below = painted_picture(W, H, 72);
    let texture = flat(8, 8, [255, 0, 0, 255]);
    let bow_tie = [
        [100.0, 100.0],
        [200.0, 200.0],
        [100.0, 200.0],
        [200.0, 100.0],
    ];
    let dart = [
        [100.0, 100.0],
        [200.0, 100.0],
        [100.0, 200.0],
        [125.0, 125.0],
    ];
    for corners in [bow_tie, dart] {
        assert!(
            matches!(
                picture_and_quad(&mut c, &below, (&texture, 8, 8), corners),
                Err(DrawError::Unsupported(_))
            ),
            "{corners:?}"
        );
    }
}
