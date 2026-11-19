//! The compositor's object map: which pixels each named layer's object covers, and how far its
//! effects spread it.

mod common;

use arachne_gpu::{Compositor, Footprint, Frame, Layer, NEAREST};
use common::{compositor, opaque, painted_picture, rectangle, upload};

const W: u32 = 480;
const H: u32 = 270;
const CLEAR: [u8; 4] = [0; 4];

/// Asserts that `footprint` is the rectangle from (`x0`, `y0`) to (`x1`, `y1`) within 1 px on each
/// side, every pixel of its box covered.
fn assert_rectangle(footprint: &Footprint, [x0, y0, x1, y1]: [f64; 4]) {
    let found = footprint.r#box.map(f64::from);
    for (side, (found, analytic)) in found.iter().zip([x0, y0, x1, y1]).enumerate() {
        assert!(
            (found - analytic).abs() <= 1.0,
            "{}: side {side} at {found}, the rectangle's at {analytic}",
            footprint.object_id
        );
    }
    assert!(!footprint.mask.is_empty() && footprint.mask.iter().all(|&m| m));
}

/// An opaque texture of `width` x `height`.
fn texture(c: &mut Compositor, width: u32, height: u32, seed: u64) -> arachne_gpu::Key {
    upload(
        c,
        width,
        height,
        &opaque(painted_picture(width, height, seed)),
    )
}

#[test]
fn a_placed_quads_footprint_is_its_rectangle() {
    let mut c = compositor(W, H);
    let kt = texture(&mut c, 16, 16, 1);
    let analytic = [100.3, 50.7, 180.6, 120.2];
    let [x0, y0, x1, y1] = analytic;
    let frame = Frame::new(
        CLEAR,
        vec![Layer::quad(kt, rectangle(x0, y0, x1, y1), NEAREST).named("photo")],
    );
    let map = c.object_map(&frame).unwrap();
    assert_eq!(map.len(), 1);
    assert_eq!((map[0].object_id.as_str(), map[0].layer), ("photo", 0));
    assert_eq!(map[0].grow_px, 0.0);
    assert_rectangle(&map[0], analytic);
}

#[test]
fn overlapping_objects_each_keep_their_whole_footprint() {
    let mut c = compositor(W, H);
    let below = texture(&mut c, W, H, 2);
    let (ka, kb) = (texture(&mut c, 10, 10, 3), texture(&mut c, 10, 10, 4));
    let a = [100.0, 100.0, 200.0, 180.0];
    let b = [150.0, 140.0, 260.0, 230.0];
    let quad =
        |key, [x0, y0, x1, y1]: [f64; 4]| Layer::quad(key, rectangle(x0, y0, x1, y1), NEAREST);
    let frame = Frame::new(
        CLEAR,
        vec![
            Layer::picture(below),
            quad(ka, a).named("a"),
            quad(kb, b).named("b"),
        ],
    );
    let map = c.object_map(&frame).unwrap();
    let found: Vec<(&str, usize)> = map
        .iter()
        .map(|f| (f.object_id.as_str(), f.layer))
        .collect();
    assert_eq!(found, [("a", 1), ("b", 2)]);
    assert_rectangle(&map[0], a);
    assert_rectangle(&map[1], b);
    // The overlap is in both.
    assert!(map[0].covers(170, 160) && map[1].covers(170, 160));
}

#[test]
fn a_pictures_footprint_is_where_it_puts_alpha_and_none_at_opacity_zero() {
    let mut c = compositor(W, H);
    let picture = painted_picture(W, H, 5);
    let kp = upload(&mut c, W, H, &picture);
    let frame = Frame::new(
        CLEAR,
        vec![
            Layer::picture(kp).named("drawn"),
            Layer::picture(kp).with_opacity(0.0).named("hidden"),
        ],
    );
    let map = c.object_map(&frame).unwrap();
    for y in 0..H {
        for x in 0..W {
            let alpha = picture[((y * W + x) * 4 + 3) as usize];
            assert_eq!(map[0].covers(x, y), alpha > 0, "({x}, {y})");
        }
    }
    assert_eq!((map[1].r#box, map[1].mask.len()), ([0; 4], 0));
}

#[test]
fn a_window_covers_its_rectangle_and_more_than_32_objects_are_mapped() {
    let mut c = compositor(W, H);
    let kt = texture(&mut c, 4, 4, 6);
    let inner = Frame::new(
        [1, 2, 3, 255],
        vec![Layer::quad(kt, rectangle(10.0, 10.0, 50.0, 50.0), NEAREST)],
    );
    let window = [300.0, 20.0, 460.0, 110.0];
    let mut layers =
        vec![Layer::window(inner, rectangle(300.0, 20.0, 460.0, 110.0), NEAREST).named("window")];
    // 40 quads in a grid of 10 x 4 cells of 20 px, each 12 px square.
    let cells: Vec<[f64; 4]> = (0..40)
        .map(|i| {
            let (x, y) = (
                f64::from(i % 10) * 20.0 + 10.0,
                f64::from(i / 10) * 20.0 + 150.0,
            );
            [x, y, x + 12.0, y + 12.0]
        })
        .collect();
    for (i, [x0, y0, x1, y1]) in cells.iter().copied().enumerate() {
        layers.push(Layer::quad(kt, rectangle(x0, y0, x1, y1), NEAREST).named(format!("cell {i}")));
    }
    let map = c.object_map(&Frame::new(CLEAR, layers)).unwrap();
    assert_eq!(map.len(), 41);
    assert_rectangle(&map[0], window);
    for (i, cell) in cells.iter().enumerate() {
        assert_eq!(
            (map[i + 1].object_id.clone(), map[i + 1].layer),
            (format!("cell {i}"), i + 1)
        );
        assert_rectangle(&map[i + 1], *cell);
    }
}

/// Every pixel within `by` pixels of the footprint, as a square: how the comparator grows it.
fn grown(footprint: &Footprint, by: u32) -> Vec<bool> {
    let mut grown = vec![false; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            if footprint.covers(x, y) {
                for gy in y.saturating_sub(by)..(y + by + 1).min(H) {
                    for gx in x.saturating_sub(by)..(x + by + 1).min(W) {
                        grown[(gy * W + gx) as usize] = true;
                    }
                }
            }
        }
    }
    grown
}

#[test]
fn removing_an_effect_carrying_layer_changes_pixels_only_inside_its_grown_footprint() {
    let mut c = compositor(W, H);
    let below = texture(&mut c, W, H, 7);
    let kt = texture(&mut c, 32, 32, 8);
    let corners = rectangle(150.0, 80.0, 310.0, 190.0);
    let mask_picture: Vec<u8> = (0..W * H)
        .flat_map(|i| if i % W < 230 { [255; 4] } else { [0; 4] })
        .collect();
    let mask = upload(&mut c, W, H, &mask_picture);
    let inner = Frame::new(
        [200, 100, 50, 255],
        vec![Layer::quad(
            kt,
            rectangle(40.0, 40.0, 200.0, 120.0),
            NEAREST,
        )],
    );
    let effects = [
        (Layer::quad(kt, corners, NEAREST).with_blur(4.0), None, 12.0),
        (
            Layer::quad(kt, corners, NEAREST).with_depth(5.0),
            Some((1.0, 1.5)),
            18.0,
        ),
        (
            Layer::quad(kt, corners, NEAREST)
                .with_blur(2.0)
                .with_mask(mask, 8.0, false),
            None,
            8.0,
        ),
        (
            Layer::window(inner, corners, NEAREST)
                .with_blur(3.0)
                .with_opacity(0.6),
            None,
            9.0,
        ),
    ];
    let base = c
        .draw(&Frame::new(CLEAR, vec![Layer::picture(below)]))
        .unwrap();
    for (layer, focus, grow_px) in effects {
        let mut frame = Frame::new(CLEAR, vec![Layer::picture(below), layer.named("effect")]);
        if let Some((depth, aperture)) = focus {
            frame = frame.with_focus(depth, aperture);
        }
        let drawn = c.draw(&frame).unwrap();
        let map = c.object_map(&frame).unwrap();
        let footprint = &map[0];
        assert_eq!(footprint.grow_px, grow_px);
        let grown = grown(footprint, grow_px.ceil() as u32);
        let mut changed = 0;
        let mut beyond = 0;
        for y in 0..H {
            for x in 0..W {
                let at = ((y * W + x) * 4) as usize;
                if drawn[at..at + 4] != base[at..at + 4] {
                    changed += 1;
                    assert!(
                        grown[(y * W + x) as usize],
                        "({x}, {y}) changed outside, grow {grow_px}"
                    );
                    beyond += usize::from(!footprint.covers(x, y));
                }
            }
        }
        assert!(changed > 0);
        // A blur spreads the layer past its footprint.
        assert!(
            beyond > 0,
            "grow {grow_px}: nothing changed beyond the footprint"
        );
    }
}
