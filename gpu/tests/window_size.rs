//! A window composed as a frame of its own size: at whole-pixel corners it is that frame drawn
//! alone, byte for byte; its effects stay inside it; windows nest, each at its own size; a picture
//! of another size is refused; the working textures its sizes need stay bounded; and placed off
//! the pixel grid, bilinear, it moves by the fraction its corners move.

mod common;

use arachne_gpu::testing::{WorkingTextures, working_textures};
use arachne_gpu::{BILINEAR, Compositor, DrawError, Frame, Key, Layer, NEAREST};
use common::{compositor, mul_un8, opaque, painted_picture, pixel, rectangle, size, upload};

const CLEAR: [u8; 4] = [0; 4];

/// A painted picture of `width` x `height`, opaque inside `keep` and transparent elsewhere.
fn region(width: u32, height: u32, seed: u64, keep: impl Fn(u32, u32) -> bool) -> Vec<u8> {
    let picture = opaque(painted_picture(width, height, seed));
    picture
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            let (x, y) = (i as u32 % width, i as u32 / width);
            if keep(x, y) { *p } else { [0; 4] }
        })
        .collect()
}

/// An opaque frame of `width` x `height` with every effect, its pictures uploaded to `c`: a
/// half-transparent picture, a quad, a blurred picture and a feathered mask whose kept side
/// reaches the frame's edge.
fn opaque_content(c: &mut Compositor, width: u32, height: u32, seed: u64) -> Frame {
    let picture = upload(c, width, height, &painted_picture(width, height, seed));
    let texture = opaque(painted_picture(12, 8, seed + 1));
    let quad = upload(c, 12, 8, &texture);
    let blurred = upload(c, width, height, &painted_picture(width, height, seed + 2));
    let masked = upload(
        c,
        width,
        height,
        &opaque(painted_picture(width, height, seed + 3)),
    );
    let mask = upload(
        c,
        width,
        height,
        &region(width, height, seed + 4, |x, _| x < width / 2),
    );
    let (qx, qy) = (f64::from(width / 5), f64::from(height / 5));
    Frame::new(
        [20, 40, 60, 255],
        vec![
            Layer::picture(picture),
            Layer::quad(quad, rectangle(qx, qy, qx + 24.0, qy + 16.0), NEAREST),
            Layer::picture(blurred).with_blur(3.0),
            Layer::picture(masked).with_mask(mask, 6.0, true),
        ],
    )
}

/// `frame` drawn alone at its own size, on a compositor of that size.
fn alone(width: u32, height: u32, content: impl Fn(&mut Compositor) -> Frame) -> Vec<u8> {
    let mut c = compositor(width, height);
    let frame = content(&mut c);
    c.draw(&frame).unwrap()
}

/// Painted "over", in the compositor's integer arithmetic.
fn over(top: [u8; 4], below: [u8; 4]) -> [u8; 4] {
    std::array::from_fn(|i| {
        (u16::from(top[i]) + u16::from(mul_un8(below[i], 255 - top[3]))).min(255) as u8
    })
}

/// Asserts that every pixel of `drawn` (the frame's size) is `inside(x - x0, y - y0)` within the
/// window at (`x0`, `y0`) of `window` and `outside(x, y)` elsewhere.
fn assert_placed(
    drawn: &[u8],
    (x0, y0): (u32, u32),
    window: (u32, u32),
    inside: impl Fn(u32, u32) -> [u8; 4],
    outside: impl Fn(u32, u32) -> [u8; 4],
    what: &str,
) {
    let (w, h) = size();
    for y in 0..h {
        for x in 0..w {
            let expected = if (x0..x0 + window.0).contains(&x) && (y0..y0 + window.1).contains(&y) {
                inside(x - x0, y - y0)
            } else {
                outside(x, y)
            };
            assert_eq!(pixel(drawn, w, x, y), expected, "{what}: ({x}, {y})");
        }
    }
}

/// The parent every window here is placed over: an opaque picture, and the frame it makes alone.
fn parent(c: &mut Compositor) -> (Layer, Vec<u8>) {
    let (w, h) = size();
    let key = upload(c, w, h, &opaque(painted_picture(w, h, 41)));
    let drawn = c
        .draw(&Frame::new(CLEAR, vec![Layer::picture(key)]))
        .unwrap();
    (Layer::picture(key), drawn)
}

#[test]
fn a_sized_window_at_whole_pixel_corners_is_its_frame_drawn_alone_byte_for_byte() {
    let (w, h) = size();
    let (iw, ih) = (160u32, 90u32);
    let inner = alone(iw, ih, |c| opaque_content(c, iw, ih, 30));
    let mut c = compositor(w, h);
    let (below, parent_alone) = parent(&mut c);
    let content = opaque_content(&mut c, iw, ih, 30);
    let (x0, y0) = (100u32, 60u32);
    let corners = rectangle(
        f64::from(x0),
        f64::from(y0),
        f64::from(x0 + iw),
        f64::from(y0 + ih),
    );
    for resampling in [NEAREST, BILINEAR] {
        let window = Layer::sized_window(content.clone(), corners, resampling, [iw, ih]);
        let drawn = c
            .draw(&Frame::new(CLEAR, vec![below.clone(), window]))
            .unwrap();
        assert_placed(
            &drawn,
            (x0, y0),
            (iw, ih),
            |x, y| pixel(&inner, iw, x, y),
            |x, y| pixel(&parent_alone, w, x, y),
            &format!("resampling {resampling}"),
        );
    }
}

/// A transparent frame of `width` x `height` whose effects reach its edges: a blurred bar along
/// its right edge and a feathered mask keeping a band along its left edge.
fn edge_effects(c: &mut Compositor, width: u32, height: u32) -> Frame {
    let bar = upload(
        c,
        width,
        height,
        &region(width, height, 81, |x, _| x >= width - 12),
    );
    let band = upload(
        c,
        width,
        height,
        &opaque(painted_picture(width, height, 82)),
    );
    let mask = upload(c, width, height, &region(width, height, 83, |x, _| x < 10));
    Frame::new(
        CLEAR,
        vec![
            Layer::picture(bar).with_blur(6.0),
            Layer::picture(band).with_mask(mask, 10.0, false),
        ],
    )
}

#[test]
fn a_sized_windows_inner_blur_and_feather_leave_nothing_outside_its_rectangle() {
    let (w, h) = size();
    let (iw, ih) = (120u32, 72u32);
    let inner = alone(iw, ih, |c| edge_effects(c, iw, ih));
    // The effects do reach the window's edges: the bar's blur and the band's feather leave
    // something in its first and last columns.
    for x in [0, iw - 1] {
        assert!(pixel(&inner, iw, x, ih / 2)[3] > 0, "column {x} is empty");
    }
    let mut c = compositor(w, h);
    let (below, parent_alone) = parent(&mut c);
    let content = edge_effects(&mut c, iw, ih);
    let (x0, y0) = (150u32, 90u32);
    let (fx0, fy0) = (f64::from(x0), f64::from(y0));
    for resampling in [NEAREST, BILINEAR] {
        let corners = rectangle(fx0, fy0, fx0 + f64::from(iw), fy0 + f64::from(ih));
        let window = Layer::sized_window(content.clone(), corners, resampling, [iw, ih]);
        let drawn = c
            .draw(&Frame::new(CLEAR, vec![below.clone(), window]))
            .unwrap();
        assert_placed(
            &drawn,
            (x0, y0),
            (iw, ih),
            |x, y| {
                over(
                    pixel(&inner, iw, x, y),
                    pixel(&parent_alone, w, x0 + x, y0 + y),
                )
            },
            |x, y| pixel(&parent_alone, w, x, y),
            &format!("resampling {resampling}"),
        );
    }
    // Off the pixel grid, bilinear: every pixel the true rectangle does not reach is the parent's.
    let (rx0, ry0) = (fx0 + 0.5, fy0 + 0.25);
    let (rx1, ry1) = (rx0 + f64::from(iw), ry0 + f64::from(ih));
    let window = Layer::sized_window(content, rectangle(rx0, ry0, rx1, ry1), BILINEAR, [iw, ih]);
    let drawn = c.draw(&Frame::new(CLEAR, vec![below, window])).unwrap();
    let mut changed = 0;
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (f64::from(x), f64::from(y));
            let reached = px + 1.0 > rx0 && px < rx1 && py + 1.0 > ry0 && py < ry1;
            if pixel(&drawn, w, x, y) != pixel(&parent_alone, w, x, y) {
                assert!(reached, "({x}, {y}) is changed outside the window");
                changed += 1;
            }
        }
    }
    assert!(changed > 0);
}

/// The innermost frame of the nested case, 80 x 48, opaque, with a blur and a mask.
const INNERMOST: (u32, u32) = (80, 48);
/// The middle frame, 200 x 120, holding the innermost at (30, 20).
const MIDDLE: (u32, u32) = (200, 120);
const INNERMOST_AT: (u32, u32) = (30, 20);

fn middle_content(c: &mut Compositor, resampling: u8, wrong: Option<Key>) -> Frame {
    let (iw, ih) = INNERMOST;
    let (mw, mh) = MIDDLE;
    let mut innermost = opaque_content(c, iw, ih, 60);
    if let Some(key) = wrong {
        innermost.layers.push(Layer::picture(key));
    }
    let (x0, y0) = (f64::from(INNERMOST_AT.0), f64::from(INNERMOST_AT.1));
    let corners = rectangle(x0, y0, x0 + f64::from(iw), y0 + f64::from(ih));
    let picture = upload(c, mw, mh, &opaque(painted_picture(mw, mh, 65)));
    Frame::new(
        CLEAR,
        vec![
            Layer::picture(picture),
            Layer::sized_window(innermost, corners, resampling, [iw, ih]),
        ],
    )
}

#[test]
fn a_sized_window_inside_a_sized_window_is_each_frame_drawn_alone() {
    let (w, h) = size();
    let (iw, ih) = INNERMOST;
    let (mw, mh) = MIDDLE;
    let innermost = alone(iw, ih, |c| opaque_content(c, iw, ih, 60));
    for (outer_resampling, inner_resampling) in [(BILINEAR, NEAREST), (NEAREST, BILINEAR)] {
        let middle = alone(mw, mh, |c| middle_content(c, inner_resampling, None));
        let (ax, ay) = INNERMOST_AT;
        assert_placed_in(&middle, mw, (ax, ay), INNERMOST, &innermost);

        let mut c = compositor(w, h);
        let (below, parent_alone) = parent(&mut c);
        let content = middle_content(&mut c, inner_resampling, None);
        let (x0, y0) = (140u32, 80u32);
        let corners = rectangle(
            f64::from(x0),
            f64::from(y0),
            f64::from(x0 + mw),
            f64::from(y0 + mh),
        );
        let window = Layer::sized_window(content, corners, outer_resampling, [mw, mh]);
        let drawn = c.draw(&Frame::new(CLEAR, vec![below, window])).unwrap();
        assert_placed(
            &drawn,
            (x0, y0),
            MIDDLE,
            |x, y| pixel(&middle, mw, x, y),
            |x, y| pixel(&parent_alone, w, x, y),
            &format!("outer {outer_resampling}, inner {inner_resampling}"),
        );
    }
}

/// Asserts that the window at (`x0`, `y0`) of `drawn`, a frame `width` wide, is `inner`.
fn assert_placed_in(
    drawn: &[u8],
    width: u32,
    (x0, y0): (u32, u32),
    window: (u32, u32),
    inner: &[u8],
) {
    for y in 0..window.1 {
        for x in 0..window.0 {
            assert_eq!(
                pixel(drawn, width, x0 + x, y0 + y),
                pixel(inner, window.0, x, y),
                "({x}, {y}) of the inner window"
            );
        }
    }
}

#[test]
fn a_picture_or_a_mask_not_of_its_windows_size_refuses_the_frame_whole() {
    let (w, h) = size();
    let mut c = compositor(w, h);
    let (mw, mh) = MIDDLE;
    let frame_sized = upload(&mut c, w, h, &opaque(painted_picture(w, h, 91)));
    let middle_sized = upload(&mut c, mw, mh, &opaque(painted_picture(mw, mh, 92)));
    let corners = rectangle(140.0, 80.0, 340.0, 200.0);
    let before = c
        .draw(&Frame::new(CLEAR, vec![Layer::picture(frame_sized)]))
        .unwrap();
    let counted = working_textures(&c);

    // Two windows deep: the innermost (80 x 48) holds a picture of the middle's size.
    let nested = middle_content(&mut c, NEAREST, Some(middle_sized));
    let outer = Frame::new(
        CLEAR,
        vec![
            Layer::picture(frame_sized),
            Layer::sized_window(nested, corners, NEAREST, [mw, mh]),
        ],
    );
    match c.draw(&outer) {
        Err(DrawError::SizeMismatch(what)) => assert_eq!(
            what,
            format!(
                "layer 1: in its window, layer 1: in its window, layer 4: a picture of {mw} x \
                 {mh} in a frame of 80 x 48: a frame's pictures and masks are of its size, and a \
                 window's frame is of the window's size"
            )
        ),
        other => panic!("{other:?}"),
    }
    // A mask of the outer frame's size in a window of its own size.
    let masked = Frame::new(
        CLEAR,
        vec![Layer::picture(middle_sized).with_mask(frame_sized, 2.0, false)],
    );
    let window = Layer::sized_window(masked, corners, NEAREST, [mw, mh]);
    assert!(matches!(
        c.draw(&Frame::new(CLEAR, vec![window])),
        Err(DrawError::SizeMismatch(_))
    ));
    // With no size given, the window is the frame's size, which the middle's picture is not, and
    // the refusal is the one it was before windows had sizes.
    let unsized_window = Frame::new(CLEAR, vec![Layer::picture(middle_sized)]);
    let window = Layer::window(unsized_window, corners, NEAREST);
    assert_eq!(
        c.draw(&Frame::new(CLEAR, vec![window])),
        Err(DrawError::Invalid(format!(
            "layer 0: in its window, layer 0: a picture of {mw} x {mh} in a frame of {w} x {h}"
        )))
    );
    // A refused frame draws nothing and keeps the working textures it found.
    assert_eq!(working_textures(&c), counted);
    let again = c
        .draw(&Frame::new(CLEAR, vec![Layer::picture(frame_sized)]))
        .unwrap();
    assert!(again == before);
}

#[test]
fn a_windows_working_textures_are_freed_once_a_frame_no_longer_uses_its_size() {
    let (w, h) = size();
    let mut c = compositor(w, h);
    // At rest: the frame's two accumulators.
    let baseline = WorkingTextures { live: 2, peak: 2 };
    assert_eq!(working_textures(&c), baseline);
    // One frame of this run needs the frame's two accumulators and, at the window's size, two
    // more and the blur's four.
    let one_frame = 8;
    let mut sizes = std::collections::HashSet::new();
    for i in 0..60u32 {
        // An inset growing a pixel a side every frame: 60 sizes.
        let (iw, ih) = (60 + 2 * i, 30 + 2 * i);
        sizes.insert((iw, ih));
        let picture = upload(&mut c, iw, ih, &painted_picture(iw, ih, u64::from(i)));
        let inner = Frame::new(CLEAR, vec![Layer::picture(picture).with_blur(1.5)]);
        let (x0, y0) = (f64::from(100 - i), f64::from(60 - i));
        let corners = rectangle(x0, y0, x0 + f64::from(iw), y0 + f64::from(ih));
        c.draw(&Frame::new(
            CLEAR,
            vec![Layer::sized_window(inner, corners, BILINEAR, [iw, ih])],
        ))
        .unwrap();
        let counted = working_textures(&c);
        assert_eq!(counted.live, one_frame, "frame {i}: {counted:?}");
        assert!(counted.peak <= one_frame, "frame {i}: {counted:?}");
        c.release(&picture);
    }
    assert_eq!(sizes.len(), 60);
    // A frame with no window frees the last size's.
    parent(&mut c);
    assert_eq!(
        working_textures(&c),
        WorkingTextures {
            live: baseline.live,
            peak: one_frame
        }
    );
}

/// The centroid of `drawn`'s alpha, pixel centres weighted by their alpha.
fn centroid(drawn: &[u8]) -> (f64, f64) {
    let (w, _) = size();
    let (mut sx, mut sy, mut total) = (0.0, 0.0, 0.0);
    for (i, p) in drawn.as_chunks::<4>().0.iter().enumerate() {
        let a = f64::from(p[3]);
        let (x, y) = ((i as u32 % w) as f64 + 0.5, (i as u32 / w) as f64 + 0.5);
        sx += a * x;
        sy += a * y;
        total += a;
    }
    (sx / total, sy / total)
}

/// The centroid of an opaque 64 x 48 window placed over nothing with `resampling` at its upper
/// left corner moved by a quarter of a pixel a frame, right and down, for 9 frames.
fn quarter_pixel_centroids(resampling: u8) -> Vec<(f64, f64)> {
    let (w, h) = size();
    let (iw, ih) = (64u32, 48u32);
    let mut c = compositor(w, h);
    let picture = upload(&mut c, iw, ih, &opaque(painted_picture(iw, ih, 71)));
    let inner = Frame::new(CLEAR, vec![Layer::picture(picture)]);
    (0..9)
        .map(|k| {
            let at = 0.25 * f64::from(k);
            let (x0, y0) = (100.0 + at, 60.0 + at);
            let corners = rectangle(x0, y0, x0 + f64::from(iw), y0 + f64::from(ih));
            let window = Layer::sized_window(inner.clone(), corners, resampling, [iw, ih]);
            centroid(&c.draw(&Frame::new(CLEAR, vec![window])).unwrap())
        })
        .collect()
}

/// How far a quarter-pixel move may stray from 0.25 px: the rounding of the antialiased edge's
/// alpha to 8 bits moves the centroid by at most about 0.005 px.
const QUARTER_TOLERANCE: f64 = 0.01;

#[test]
fn a_window_moved_by_a_quarter_pixel_a_frame_moves_its_content_a_quarter_pixel_a_frame() {
    let bilinear = quarter_pixel_centroids(BILINEAR);
    // At whole-pixel corners its content is where the window's is.
    let (cx, cy) = bilinear[0];
    assert!(
        (cx - 132.0).abs() < 1e-9 && (cy - 84.0).abs() < 1e-9,
        "{:?}",
        bilinear[0]
    );
    for (k, pair) in bilinear.windows(2).enumerate() {
        let (dx, dy) = (pair[1].0 - pair[0].0, pair[1].1 - pair[0].1);
        eprintln!(
            "bilinear frame {k} to {}: moved ({dx:.5}, {dy:.5}) px",
            k + 1
        );
        assert!(
            (dx - 0.25).abs() <= QUARTER_TOLERANCE && (dy - 0.25).abs() <= QUARTER_TOLERANCE,
            "frame {k} to {}: moved ({dx}, {dy}) px",
            k + 1
        );
    }
    // Nearest, the same move steps a whole pixel or not at all: what bilinear is for.
    let nearest = quarter_pixel_centroids(NEAREST);
    let steps: Vec<f64> = nearest.windows(2).map(|p| p[1].0 - p[0].0).collect();
    assert!(
        steps
            .iter()
            .all(|d| d.abs() <= QUARTER_TOLERANCE || (d - 1.0).abs() <= QUARTER_TOLERANCE),
        "{steps:?}"
    );
    assert!(
        steps.iter().any(|d| (d - 1.0).abs() <= QUARTER_TOLERANCE),
        "{steps:?}"
    );
}

/// The size a caller gives a window shown at an axis-aligned rectangle: per axis, its true size
/// rounded to the nearest whole pixel, ties half up, at least 1.
fn shown_size(x0: f64, y0: f64, x1: f64, y1: f64) -> [u32; 2] {
    let side = |length: f64| ((length + 0.5).floor() as u32).max(1);
    [side(x1 - x0), side(y1 - y0)]
}

/// How much of pixel `x`'s cell `[x, x + 1)` lies in `[a, b)`.
fn overlap(x: u32, a: f64, b: f64) -> f64 {
    let x = f64::from(x);
    ((x + 1.0).min(b) - x.max(a)).max(0.0)
}

#[test]
fn a_bilinear_window_paints_only_the_cells_its_true_rectangle_touches_and_each_by_its_coverage() {
    let (w, h) = size();
    let mut c = compositor(w, h);
    // An opaque window's alpha is the coverage it paints. Each rectangle's width rounds down
    // (1.49 to 1; 63.4 to 63), so its texels are wider than its pixels.
    for (x0, y0, x1, y1) in [(10.05, 20.0, 11.54, 21.0), (100.3, 60.6, 163.7, 108.2)] {
        let [iw, ih] = shown_size(x0, y0, x1, y1);
        let picture = upload(&mut c, iw, ih, &opaque(painted_picture(iw, ih, 101)));
        let inner = Frame::new(CLEAR, vec![Layer::picture(picture)]);
        let window = Layer::sized_window(inner, rectangle(x0, y0, x1, y1), BILINEAR, [iw, ih]);
        let drawn = c.draw(&Frame::new(CLEAR, vec![window])).unwrap();
        let mut worst: f64 = 0.0;
        for y in 0..h {
            for x in 0..w {
                let covered = overlap(x, x0, x1) * overlap(y, y0, y1);
                let found = pixel(&drawn, w, x, y);
                if covered == 0.0 {
                    assert_eq!(found, [0; 4], "({x}, {y}) is outside the cells it touches");
                } else {
                    let off = (f64::from(found[3]) - 255.0 * covered).abs();
                    worst = worst.max(off);
                    assert!(
                        off <= 1.0,
                        "({x}, {y}): alpha {} for cover {covered}",
                        found[3]
                    );
                }
            }
        }
        eprintln!(
            "edge [{x0}, {x1}) x [{y0}, {y1}) at {iw} x {ih}: alpha within {worst:.3} of 255 x \
             coverage; nothing outside"
        );
        let edge = |x: u32, y: u32| pixel(&drawn, w, x, y)[3];
        if iw == 1 {
            eprintln!(
                "  pixel (9, 20) {}, (10, 20) {}, (11, 20) {}, (12, 20) {}",
                edge(9, 20),
                edge(10, 20),
                edge(11, 20),
                edge(12, 20)
            );
        }
        c.release(&picture);
    }
}

/// The colour-weighted centre of an interior marker: the red of `drawn`, which only the marker
/// carries, weighting each pixel's centre.
fn marker_centre(drawn: &[u8]) -> (f64, f64) {
    let (w, _) = size();
    let (mut sx, mut sy, mut total) = (0.0, 0.0, 0.0);
    for (i, p) in drawn.as_chunks::<4>().0.iter().enumerate() {
        let red = f64::from(p[0]);
        let (x, y) = ((i as u32 % w) as f64 + 0.5, (i as u32 / w) as f64 + 0.5);
        sx += red * x;
        sy += red * y;
        total += red;
    }
    (sx / total, sy / total)
}

#[test]
fn a_windows_interior_marker_moves_a_quarter_pixel_a_frame_with_its_window() {
    let (w, _) = size();
    let (iw, ih) = (64u32, 48u32);
    let mut c = compositor(w, h);
    // An opaque blue field with a 3 x 3 red marker inside it: only the marker has red, so its
    // centre is read apart from the window's edge, whose alpha it does not weigh.
    let field: Vec<u8> = (0..iw * ih)
        .flat_map(|i| {
            let (x, y) = (i % iw, i / iw);
            if (30..33).contains(&x) && (20..23).contains(&y) {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        })
        .collect();
    let picture = upload(&mut c, iw, ih, &field);
    let inner = Frame::new(CLEAR, vec![Layer::picture(picture)]);
    let centres: Vec<(f64, f64)> = (0..9)
        .map(|k| {
            let at = 0.25 * f64::from(k);
            let (x0, y0) = (100.0 + at, 60.0 + at);
            let corners = rectangle(x0, y0, x0 + f64::from(iw), y0 + f64::from(ih));
            let window = Layer::sized_window(inner.clone(), corners, BILINEAR, [iw, ih]);
            marker_centre(&c.draw(&Frame::new(CLEAR, vec![window])).unwrap())
        })
        .collect();
    // At whole-pixel corners the marker's centre is its own: (100 + 31.5, 60 + 21.5).
    let (mx, my) = centres[0];
    assert!(
        (mx - 131.5).abs() < 1e-9 && (my - 81.5).abs() < 1e-9,
        "{:?}",
        centres[0]
    );
    for (k, pair) in centres.windows(2).enumerate() {
        let (dx, dy) = (pair[1].0 - pair[0].0, pair[1].1 - pair[0].1);
        eprintln!("marker frame {k} to {}: moved ({dx:.5}, {dy:.5}) px", k + 1);
        assert!(
            (dx - 0.25).abs() <= QUARTER_TOLERANCE && (dy - 0.25).abs() <= QUARTER_TOLERANCE,
            "frame {k} to {}: the marker moved ({dx}, {dy}) px",
            k + 1
        );
    }
}
