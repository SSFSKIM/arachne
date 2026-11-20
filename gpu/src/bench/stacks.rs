//! The stacks the frame set draws and the read-back benchmark times. Every picture is drawn here
//! by code, from a seed, with no asset; a picture's shapes sit at fractions of the frame, so a
//! stack drawn at another size shows the same scene at that size.
//!
//! - The **reference stack**: six layers, each a picture of the frame's size: an opaque backdrop,
//!   two pictures blurred at sigma 8 px, one under a soft mask (feather 16 px), one window onto a
//!   stack of three pictures placed inside the frame, and one plain picture.
//! - The **heavy stack**: an opaque backdrop and 23 pictures, four of them blurred at sigma
//!   24 px.
//! - One **probe frame** per effect probe of the crate's tests, with the probe's own numbers.
//!
//! The reference and heavy stacks' sigmas and feather are stated at 1920 x 1080 and scale with
//! the frame's height, as a scene's blur declared in scene units does: twice as many pixels at
//! 3840 x 2160, a quarter at 480 x 270. The probes' numbers are pixels at any size.

use crate::{Compositor, DrawError, Frame, Key, Layer, NEAREST, painted_key};

/// The version of the frame set: anything that changes what one of its frames draws changes it,
/// and frames are compared only within one version.
pub const VERSION: u32 = 1;

/// The sizes the reference and heavy stacks are drawn and timed at.
pub const REFERENCE_SIZES: [(u32, u32); 2] = [(1920, 1080), (3840, 2160)];

/// The size the probe frames are drawn at.
pub const PROBE_SIZE: (u32, u32) = (1920, 1080);

/// The probe frames, by name, in the set's order.
pub const PROBES: [&str; 12] = [
    "blur-step-sigma-2",
    "blur-step-sigma-6",
    "blur-step-sigma-12",
    "blur-shapes-sigma-6",
    "focus-off",
    "focus-and-blur",
    "mask-columns",
    "mask-rows",
    "mask-disc",
    "window-full",
    "window-third",
    "window-nested",
];

/// A painted texture drawn by code, under its painted key.
pub struct Texture {
    pub key: Key,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Texture {
    fn painted(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        Texture {
            key: painted_key(width, height, &rgba),
            width,
            height,
            rgba,
        }
    }
}

/// One frame of the set: its name, its size, its layers and the textures they name.
pub struct Stack {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub frame: Frame,
    pub textures: Vec<Texture>,
}

impl Stack {
    /// Uploads every texture the frame names.
    pub fn upload(&self, compositor: &mut Compositor) -> Result<(), DrawError> {
        for t in &self.textures {
            compositor.upload_painted(t.key, t.width, t.height, &t.rgba)?;
        }
        Ok(())
    }

    /// Lets go of every texture the frame names.
    pub fn release(&self, compositor: &mut Compositor) {
        for t in &self.textures {
            compositor.release(&t.key);
        }
    }
}

/// The reference stack at `width` x `height`. `variant` 0 is the frame set's; another variant
/// draws new pictures into the same layers (the benchmark's new uploads).
pub fn reference(width: u32, height: u32, variant: u64) -> Stack {
    let (w, h) = (f64::from(width), f64::from(height));
    let scale = h / 1080.0;
    let seed = |i: u64| variant * 1000 + i;
    let backdrop = Texture::painted(width, height, backdrop(width, height, seed(1)));
    let pictures: Vec<Texture> = (2..6)
        .map(|i| Texture::painted(width, height, shapes(width, height, seed(i))))
        .collect();
    let inner: Vec<Texture> = (6..9)
        .map(|i| Texture::painted(width, height, shapes(width, height, seed(i))))
        .collect();
    let mask = Texture::painted(
        width,
        height,
        ellipse(width, height, [0.5 * w, 0.5 * h], [0.36 * w, 0.36 * h]),
    );
    let inner_frame = Frame::new(
        [0; 4],
        inner.iter().map(|t| Layer::picture(t.key)).collect(),
    );
    let window = rectangle(
        (0.32 * w).round(),
        (0.10 * h).round(),
        (0.90 * w).round(),
        (0.68 * h).round(),
    );
    let layers = vec![
        Layer::picture(backdrop.key),
        Layer::picture(pictures[0].key).with_blur(8.0 * scale),
        Layer::picture(pictures[1].key),
        Layer::window(inner_frame, window, NEAREST),
        Layer::picture(pictures[2].key).with_mask(mask.key, 16.0 * scale, false),
        Layer::picture(pictures[3].key).with_blur(8.0 * scale),
    ];
    let mut textures = vec![backdrop];
    textures.extend(pictures);
    textures.extend(inner);
    textures.push(mask);
    Stack {
        name: format!("reference-{width}x{height}"),
        width,
        height,
        frame: Frame::new([0; 4], layers),
        textures,
    }
}

/// The heavy stack at `width` x `height`; `variant` as for [`reference`].
pub fn heavy(width: u32, height: u32, variant: u64) -> Stack {
    let scale = f64::from(height) / 1080.0;
    let seed = |i: u64| variant * 1000 + 100 + i;
    let mut textures = vec![Texture::painted(
        width,
        height,
        backdrop(width, height, seed(0)),
    )];
    textures
        .extend((1..24).map(|i| Texture::painted(width, height, shapes(width, height, seed(i)))));
    let layers = textures
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let layer = Layer::picture(t.key);
            if i % 6 == 5 {
                layer.with_blur(24.0 * scale)
            } else {
                layer
            }
        })
        .collect();
    Stack {
        name: format!("heavy-{width}x{height}"),
        width,
        height,
        frame: Frame::new([0; 4], layers),
        textures,
    }
}

/// The probe frame `name` (one of [`PROBES`]) at `width` x `height`.
///
/// # Panics
///
/// On a name that is not one of [`PROBES`].
pub fn probe(name: &str, width: u32, height: u32) -> Stack {
    let (w, h) = (f64::from(width), f64::from(height));
    let clear = [0; 4];
    let mut textures = Vec::new();
    let mut add = |rgba: Vec<u8>, tw: u32, th: u32| {
        let texture = Texture::painted(tw, th, rgba);
        let key = texture.key;
        textures.push(texture);
        key
    };
    let frame = match name {
        "blur-step-sigma-2" | "blur-step-sigma-6" | "blur-step-sigma-12" => {
            let sigma: f64 = name.rsplit('-').next().unwrap().parse().unwrap();
            let step = add(step(width, height, width / 2), width, height);
            Frame::new(clear, vec![Layer::picture(step).with_blur(sigma)])
        }
        "blur-shapes-sigma-6" => {
            let below = add(backdrop(width, height, 1), width, height);
            let shapes = add(shapes(width, height, 2), width, height);
            Frame::new(
                clear,
                vec![Layer::picture(below), Layer::picture(shapes).with_blur(6.0)],
            )
        }
        "focus-off" => {
            // 1.5 * |7 - 3| = 6.
            let step = add(step(width, height, width / 2), width, height);
            Frame::new(clear, vec![Layer::picture(step).with_depth(7.0)]).with_focus(3.0, 1.5)
        }
        "focus-and-blur" => {
            // Its own blur of 3 and a defocus of 2 * |5 - 3| = 4 make one of 5.
            let step = add(step(width, height, width / 2), width, height);
            Frame::new(
                clear,
                vec![Layer::picture(step).with_depth(5.0).with_blur(3.0)],
            )
            .with_focus(3.0, 2.0)
        }
        "mask-columns" | "mask-rows" => {
            let black = add(flat(width, height, [0, 0, 0, 255]), width, height);
            let white = add(flat(width, height, [255; 4]), width, height);
            let mask = if name == "mask-columns" {
                let column = width * 5 / 12;
                mask_picture(width, height, |x, _| x < column)
            } else {
                let line = height * 4 / 9;
                mask_picture(width, height, |_, y| y < line)
            };
            let mask = add(mask, width, height);
            Frame::new(
                clear,
                vec![
                    Layer::picture(black),
                    Layer::picture(white).with_mask(mask, 10.0, false),
                ],
            )
        }
        "mask-disc" => {
            let below = add(backdrop(width, height, 3), width, height);
            let shapes = add(shapes(width, height, 4), width, height);
            let radius = 0.26 * h;
            let disc = add(
                ellipse(width, height, [0.5 * w, 0.5 * h], [radius, radius]),
                width,
                height,
            );
            Frame::new(
                clear,
                vec![
                    Layer::picture(below),
                    Layer::picture(shapes).with_mask(disc, 8.0, false),
                ],
            )
        }
        "window-full" | "window-third" | "window-nested" => {
            let picture = add(shapes(width, height, 5), width, height);
            let tile = add(backdrop(24, 16, 6), 24, 16);
            let inner = Frame::new(
                [20, 40, 60, 255],
                vec![
                    Layer::picture(picture),
                    Layer::quad(
                        tile,
                        rectangle(
                            (w * 50.0 / 480.0).round(),
                            (h * 30.0 / 270.0).round(),
                            (w * 122.0 / 480.0).round(),
                            (h * 78.0 / 270.0).round(),
                        ),
                        NEAREST,
                    ),
                ],
            );
            let full = rectangle(0.0, 0.0, w, h);
            match name {
                "window-full" => Frame::new(clear, vec![Layer::window(inner, full, NEAREST)]),
                "window-nested" => {
                    let once = Frame::new(clear, vec![Layer::window(inner, full, NEAREST)]);
                    Frame::new(clear, vec![Layer::window(once, full, NEAREST)])
                }
                _ => {
                    // A third of the frame's size, as the window probe places it.
                    let parent = add(backdrop(width, height, 7), width, height);
                    let (x0, y0) = (width * 5 / 24, height * 2 / 9);
                    let corners = rectangle(
                        f64::from(x0),
                        f64::from(y0),
                        f64::from(x0 + width / 3),
                        f64::from(y0 + height / 3),
                    );
                    Frame::new(
                        [5, 6, 7, 255],
                        vec![
                            Layer::picture(parent),
                            Layer::window(inner, corners, NEAREST),
                        ],
                    )
                }
            }
        }
        other => panic!("no probe frame is named {other}"),
    };
    Stack {
        name: format!("probe-{name}-{width}x{height}"),
        width,
        height,
        frame,
        textures,
    }
}

/// Corners in CE's order for the rectangle from (`x0`, `y0`) to (`x1`, `y1`).
fn rectangle(x0: f64, y0: f64, x1: f64, y1: f64) -> [[f64; 2]; 4] {
    [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
}

/// A deterministic stream (a 64-bit LCG's high bits).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        let mut lcg = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
        lcg.advance();
        lcg
    }

    fn advance(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// A number in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.advance() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn byte(&mut self) -> u8 {
        (self.advance() >> 56) as u8
    }
}

/// One colour everywhere.
fn flat(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    rgba.repeat((width * height) as usize)
}

/// Opaque white left of column `edge`, opaque black from it on.
fn step(width: u32, height: u32, edge: u32) -> Vec<u8> {
    (0..width * height)
        .flat_map(|i| {
            if i % width < edge {
                [255; 4]
            } else {
                [0, 0, 0, 255]
            }
        })
        .collect()
}

/// A mask picture, painted: kept where `keep(x, y)`, its colour unlike its alpha, since only the
/// alpha masks.
fn mask_picture(width: u32, height: u32, keep: impl Fn(u32, u32) -> bool) -> Vec<u8> {
    (0..width * height)
        .flat_map(|i| {
            if keep(i % width, i / width) {
                [40, 0, 90, 255]
            } else {
                [0; 4]
            }
        })
        .collect()
}

/// An opaque backdrop: a smooth gradient whose colours the seed sets.
fn backdrop(width: u32, height: u32, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg::new(seed);
    let tint = [lcg.unit(), lcg.unit(), lcg.unit()];
    let (w, h) = (f64::from(width), f64::from(height));
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let v = (f64::from(y) + 0.5) / h;
        for x in 0..width {
            let u = (f64::from(x) + 0.5) / w;
            rgba.extend_from_slice(&[
                (30.0 + 150.0 * u + 60.0 * tint[0]).round() as u8,
                (20.0 + 140.0 * v + 60.0 * tint[1]).round() as u8,
                (40.0 + 100.0 * (1.0 - u) * v + 80.0 * tint[2]).round() as u8,
                255,
            ]);
        }
    }
    rgba
}

/// A transparent picture of five half-transparent discs with antialiased edges, stacked with the
/// painted "over" rule in integers. The seed sets where they are, how large and of what colour.
fn shapes(width: u32, height: u32, seed: u64) -> Vec<u8> {
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    let mut lcg = Lcg::new(seed);
    let (w, h) = (f64::from(width), f64::from(height));
    let short = w.min(h);
    for _ in 0..5 {
        let centre = [lcg.unit() * w, lcg.unit() * h];
        let radius = (0.06 + 0.2 * lcg.unit()) * short;
        let colour = [lcg.byte(), lcg.byte(), lcg.byte()];
        let alpha = 96 + (lcg.unit() * 128.0) as u32;
        let (x0, x1) = span(centre[0], radius, width);
        let (y0, y1) = span(centre[1], radius, height);
        for y in y0..y1 {
            for x in x0..x1 {
                let d = (f64::from(x) + 0.5 - centre[0]).hypot(f64::from(y) + 0.5 - centre[1]);
                let coverage = (radius - d + 0.5).clamp(0.0, 1.0);
                if coverage > 0.0 {
                    let at = ((y * width + x) * 4) as usize;
                    let a = (f64::from(alpha) * coverage).round() as u32;
                    over(&mut rgba[at..at + 4], colour, a);
                }
            }
        }
    }
    rgba
}

/// The whole pixels a disc of `radius` about `centre` can touch, within `0..size`.
fn span(centre: f64, radius: f64, size: u32) -> (u32, u32) {
    let low = (centre - radius - 1.0).floor().max(0.0) as u32;
    let high = ((centre + radius + 1.0).ceil().max(0.0) as u32).min(size);
    (low.min(size), high)
}

/// A straight colour at alpha `a`, painted and stacked over the painted pixel `below`.
fn over(below: &mut [u8], colour: [u8; 3], a: u32) {
    let keep = 255 - a;
    for (value, c) in below[..3].iter_mut().zip(colour) {
        let top = (u32::from(c) * a + 127) / 255;
        *value = (top + (u32::from(*value) * keep + 127) / 255) as u8;
    }
    below[3] = (a + (u32::from(below[3]) * keep + 127) / 255) as u8;
}

/// A mask picture: opaque white inside the ellipse about `centre` with radii `radii`, an
/// antialiased edge, and transparent outside.
fn ellipse(width: u32, height: u32, centre: [f64; 2], radii: [f64; 2]) -> Vec<u8> {
    let mean_radius = (radii[0] * radii[1]).sqrt();
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let dx = (f64::from(x) + 0.5 - centre[0]) / radii[0];
            let dy = (f64::from(y) + 0.5 - centre[1]) / radii[1];
            let inside = (1.0 - dx.hypot(dy)) * mean_radius;
            let a = ((inside + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba.extend_from_slice(&[a; 4]);
        }
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    fn painted(rgba: &[u8]) -> bool {
        rgba.as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[..3].iter().all(|&c| c <= p[3]))
    }

    #[test]
    fn every_texture_of_every_stack_is_valid_painted_rgba_of_its_size() {
        let (w, h) = (96, 54);
        let mut stacks = vec![reference(w, h, 0), heavy(w, h, 0), reference(w, h, 3)];
        stacks.extend(PROBES.iter().map(|name| probe(name, w, h)));
        for stack in &stacks {
            assert!(!stack.textures.is_empty(), "{}", stack.name);
            for t in &stack.textures {
                assert_eq!(t.rgba.len(), (t.width * t.height * 4) as usize);
                assert!(painted(&t.rgba), "{}", stack.name);
                assert_eq!(t.key, painted_key(t.width, t.height, &t.rgba));
            }
        }
    }

    #[test]
    fn the_stacks_have_the_layers_they_are_named_for() {
        let reference = reference(192, 108, 0);
        let layers = &reference.frame.layers;
        assert_eq!(layers.len(), 6);
        let blurs: Vec<f64> = layers
            .iter()
            .filter_map(|l| l.blur.map(|b| b.sigma_px))
            .collect();
        assert_eq!(blurs, [0.8, 0.8]);
        assert_eq!(layers.iter().filter(|l| l.mask.is_some()).count(), 1);
        let windows: Vec<&Frame> = layers
            .iter()
            .filter_map(|l| match &l.source {
                crate::Source::Window { frame, .. } => Some(frame.as_ref()),
                _ => None,
            })
            .collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].layers.len(), 3);

        // At 2160 px high, twice the pixels of 1080: sigma 48.
        let heavy = heavy(64, 2160, 0);
        assert_eq!(heavy.frame.layers.len(), 24);
        let blurs: Vec<f64> = heavy
            .frame
            .layers
            .iter()
            .filter_map(|l| l.blur.map(|b| b.sigma_px))
            .collect();
        assert_eq!(blurs, [48.0; 4]);
    }

    #[test]
    fn a_variant_draws_new_pictures_into_the_same_layers() {
        let (first, second) = (reference(96, 54, 0), reference(96, 54, 1));
        assert_eq!(first.textures.len(), second.textures.len());
        // Every picture is new; the mask is the stack's shape and stays.
        assert!(
            first
                .textures
                .iter()
                .zip(&second.textures)
                .filter(|(a, b)| a.key != b.key)
                .count()
                == 8
        );
        // The same variant draws the same bytes.
        let again = reference(96, 54, 0);
        assert!(
            first
                .textures
                .iter()
                .zip(&again.textures)
                .all(|(a, b)| a.key == b.key)
        );
    }
}
