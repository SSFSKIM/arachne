//! A frame checked whole and resolved before anything is drawn: what each layer places and where,
//! the effects it carries as integer kernels, and how far they spread it.

use crate::compositor::DrawError;
use crate::effects::{blur_reach, feather_reach, kernel, resolved_sigma};
use crate::frame::{BILINEAR, Focus, Frame, Layer, NEAREST, Source};
use crate::keys::Key;
use crate::placement::{Placement, place, rectangle};

/// How deep windows may nest: a frame holds windows, whose frames hold windows, to this depth.
/// Each level holds working textures of its own on the device.
pub const MAX_NESTING: usize = 4;

/// One stack, resolved.
pub(crate) struct Stack {
    pub background: [u8; 4],
    pub layers: Vec<Planned>,
}

/// One layer, resolved.
pub(crate) struct Planned {
    pub placed: Placed,
    /// 0..=255.
    pub opacity: u32,
    pub effect: Option<Effect>,
    pub object_id: Option<String>,
    /// How far the layer's effects spread it, in pixels: three sigma of its blur, or its mask's
    /// feather, whichever is larger; 0 with no effect.
    pub grow_px: f64,
}

/// What a layer places, and where.
pub(crate) enum Placed {
    /// Nothing: a quad or a window whose corners have no area.
    Nothing,
    Picture(Key),
    /// A texture at four corners: the rows of the map from a pixel to its texel.
    Quad {
        key: Key,
        rows: [[f32; 4]; 3],
    },
    /// A stack composed as a frame of `size`, placed as a quad, resampled by PIL's `resampling`
    /// value; `rect` is its outline, `[x0, y0, x1, y1]`, for a bilinear window's coverage (only an
    /// axis-aligned rectangle is placed bilinear).
    Window {
        stack: Box<Stack>,
        rows: [[f32; 4]; 3],
        size: [u32; 2],
        resampling: u8,
        rect: [f32; 4],
    },
}

/// A layer's effects: a blur (the kernel that changes nothing when it has none) and a soft mask.
pub(crate) struct Effect {
    pub blur: Vec<u32>,
    pub mask: Option<Mask>,
}

pub(crate) struct Mask {
    pub key: Key,
    pub feather: Vec<u32>,
    pub invert: bool,
}

/// What planning needs of the compositor: the size of the frame being planned, the device's
/// largest texture side, whether that frame is a sized window's or inside one, and the size of the
/// texture held under a key, if one is.
pub(crate) struct Canvas<'a> {
    pub width: u32,
    pub height: u32,
    pub max_dimension: u32,
    /// True inside a window given a size, at any depth: its refusals are worded for the window,
    /// and a size mismatch is [`DrawError::SizeMismatch`]. Outside one, every refusal is as it was
    /// before windows had sizes.
    pub sized: bool,
    pub size_of: &'a dyn Fn(&Key) -> Option<(u32, u32)>,
}

impl<'a> Canvas<'a> {
    /// The canvas of a window's frame: this one at `size`, sized if the window was given one.
    fn of_window(&self, size: [u32; 2], given: bool) -> Canvas<'a> {
        Canvas {
            width: size[0],
            height: size[1],
            max_dimension: self.max_dimension,
            sized: self.sized || given,
            size_of: self.size_of,
        }
    }

    /// Refuses a picture or a mask that is not of this frame's size: inside a sized window with
    /// [`DrawError::SizeMismatch`], elsewhere as before, [`DrawError::Invalid`].
    fn check_size(&self, index: usize, what: &str, key: &Key) -> Result<(), DrawError> {
        let (width, height) = self.held(key)?;
        if (width, height) == (self.width, self.height) {
            return Ok(());
        }
        let (fw, fh) = (self.width, self.height);
        Err(if self.sized {
            DrawError::SizeMismatch(format!(
                "layer {index}: a {what} of {width} x {height} in a frame of {fw} x {fh}: a \
                 frame's pictures and masks are of its size, and a window's frame is of the \
                 window's size"
            ))
        } else {
            DrawError::Invalid(format!(
                "layer {index}: a {what} of {width} x {height} in a frame of {fw} x {fh}"
            ))
        })
    }

    /// Refuses a kernel of `what` (a blur's or a mask's feather) reaching `reach` px, past the
    /// frame's longer side. Inside a sized window the refusal names the largest `what` this
    /// size allows, from `largest(side)`, and the fix. The bound is on the work a frame of this
    /// size does, not on the kernel: the kernel is neither clamped nor given a padded texture.
    fn check_reach(
        &self,
        index: usize,
        reach: f64,
        legacy: impl FnOnce() -> String,
        sized: impl FnOnce(f64) -> String,
    ) -> Result<(), DrawError> {
        let side = self.longest_reach();
        if reach > side {
            let what = if self.sized { sized(side) } else { legacy() };
            return Err(DrawError::Unsupported(format!("layer {index}: {what}")));
        }
        Ok(())
    }

    fn held(&self, key: &Key) -> Result<(u32, u32), DrawError> {
        (self.size_of)(key).ok_or(DrawError::UnknownKey(*key))
    }

    /// The longest reach a kernel may have: the frame's longer side.
    fn longest_reach(&self) -> f64 {
        f64::from(self.width.max(self.height))
    }
}

/// Plans `frame`, `nesting` windows deep. Refuses the whole frame on the first fault.
pub(crate) fn plan(frame: &Frame, canvas: &Canvas, nesting: usize) -> Result<Stack, DrawError> {
    if let Some(focus) = frame.focus
        && !(focus.depth.is_finite() && focus.aperture.is_finite() && focus.aperture >= 0.0)
    {
        return Err(DrawError::Invalid(format!(
            "a focus at depth {} with aperture {}: the depth is finite and the aperture finite \
             and not negative",
            focus.depth, focus.aperture
        )));
    }
    let layers = frame
        .layers
        .iter()
        .enumerate()
        .map(|(index, layer)| plan_layer(index, layer, frame.focus, canvas, nesting))
        .collect::<Result<_, _>>()?;
    Ok(Stack {
        background: frame.background,
        layers,
    })
}

fn plan_layer(
    index: usize,
    layer: &Layer,
    focus: Option<Focus>,
    canvas: &Canvas,
    nesting: usize,
) -> Result<Planned, DrawError> {
    let invalid = |what: String| DrawError::Invalid(format!("layer {index}: {what}"));
    let unsupported = |what: String| DrawError::Unsupported(format!("layer {index}: {what}"));

    let placed = match &layer.source {
        Source::Picture { key } => {
            canvas.check_size(index, "picture", key)?;
            Placed::Picture(*key)
        }
        Source::Quad {
            key,
            corners,
            resampling,
        } => {
            let (width, height) = canvas.held(key)?;
            match placement(corners, *resampling, false, width, height)
                .map_err(|e| e.labelled(index))?
            {
                Some(rows) => Placed::Quad { key: *key, rows },
                None => Placed::Nothing,
            }
        }
        Source::Window {
            frame,
            corners,
            resampling,
            size,
        } => {
            if nesting >= MAX_NESTING {
                return Err(unsupported(format!(
                    "windows nested deeper than {MAX_NESTING}"
                )));
            }
            let given = size.is_some();
            let size = size.unwrap_or([canvas.width, canvas.height]);
            let max = canvas.max_dimension;
            if !size.iter().all(|side| (1..=max).contains(side)) {
                return Err(invalid(format!(
                    "a window of {} x {} does not fit this device (1 to {max} a side)",
                    size[0], size[1]
                )));
            }
            let stack = plan(frame, &canvas.of_window(size, given), nesting + 1).map_err(
                |err| match err {
                    DrawError::Invalid(what) => invalid(format!("in its window, {what}")),
                    DrawError::Unsupported(what) => unsupported(format!("in its window, {what}")),
                    DrawError::SizeMismatch(what) => {
                        DrawError::SizeMismatch(format!("layer {index}: in its window, {what}"))
                    }
                    other => other,
                },
            )?;
            match placement(corners, *resampling, given, size[0], size[1])
                .map_err(|e| e.labelled(index))?
            {
                Some(rows) => Placed::Window {
                    stack: Box::new(stack),
                    rows,
                    size,
                    resampling: *resampling,
                    rect: rectangle(corners).unwrap_or_default().map(|v| v as f32),
                },
                None => Placed::Nothing,
            }
        }
    };

    if !(0.0..=1.0).contains(&layer.opacity) {
        return Err(invalid(format!(
            "opacity {} is not in 0..=1",
            layer.opacity
        )));
    }
    if let Some(depth) = layer.depth
        && !depth.is_finite()
    {
        return Err(invalid(format!("a depth of {depth} is not finite")));
    }
    let blur = layer.blur.map(|blur| blur.sigma_px);
    if let Some(sigma) = blur
        && !(sigma.is_finite() && sigma >= 0.0)
    {
        return Err(invalid(format!(
            "a blur's sigma of {sigma} px is not finite and not negative"
        )));
    }
    let sigma = resolved_sigma(blur, layer.depth, focus);
    let reach = blur_reach(sigma);
    canvas.check_reach(
        index,
        reach,
        || format!("a blur of sigma {sigma} px reaches {reach} px, past the frame's longer side"),
        |side| {
            format!(
                "a blur of sigma {sigma} px reaches {reach} px, past the window's longer side of \
                 {side} px; at this size the largest sigma is {} px (reduce the blur or show the \
                 window larger)",
                hundredths_down(side / 3.0)
            )
        },
    )?;

    let mask = match &layer.mask {
        None => None,
        Some(mask) => {
            canvas.check_size(index, "mask", &mask.key)?;
            let feather = mask.feather_px;
            if !(feather.is_finite() && feather >= 0.0) {
                return Err(invalid(format!(
                    "a mask's feather of {feather} px is not finite and not negative"
                )));
            }
            let reach = feather_reach(feather);
            canvas.check_reach(
                index,
                reach,
                || format!("a mask's feather of {feather} px is past the frame's longer side"),
                |side| {
                    format!(
                        "a mask's feather of {feather} px reaches {reach} px, past the window's \
                         longer side of {side} px; at this size the largest feather is {} px \
                         (reduce the feather or show the window larger)",
                        side + 0.99
                    )
                },
            )?;
            Some(Mask {
                key: mask.key,
                feather: kernel(feather / 3.0, reach as u32),
                invert: mask.invert,
            })
        }
    };

    let (effect, grow_px) = if sigma > 0.0 || mask.is_some() {
        let feather = layer.mask.as_ref().map_or(0.0, |m| m.feather_px);
        let grow_px = (3.0 * sigma).max(feather);
        let blur = kernel(sigma, reach as u32);
        (Some(Effect { blur, mask }), grow_px)
    } else {
        (None, 0.0)
    };

    Ok(Planned {
        placed,
        opacity: (layer.opacity * 255.0).round() as u32,
        effect,
        object_id: layer.object_id.clone(),
        grow_px,
    })
}

/// `value` rounded down to hundredths, so a stated largest value is one that is accepted.
fn hundredths_down(value: f64) -> f64 {
    (value * 100.0).floor() / 100.0
}

/// Why a quad's or a window's corners place nothing drawable.
enum Unplaceable {
    Unsupported(String),
    Invalid(String),
}

impl Unplaceable {
    fn labelled(self, index: usize) -> DrawError {
        match self {
            Unplaceable::Unsupported(what) => {
                DrawError::Unsupported(format!("layer {index}: {what}"))
            }
            Unplaceable::Invalid(what) => DrawError::Invalid(format!("layer {index}: {what}")),
        }
    }
}

/// The rows placing a `width` x `height` texture at `corners`, or `None` where the corners have no
/// area. A quad and a window of no stated size are resampled by nearest only, as before windows
/// had sizes; a sized window by nearest or bilinear.
fn placement(
    corners: &[[f64; 2]; 4],
    resampling: u8,
    sized_window: bool,
    width: u32,
    height: u32,
) -> Result<Option<[[f32; 4]; 3]>, Unplaceable> {
    if sized_window && ![NEAREST, BILINEAR].contains(&resampling) {
        return Err(Unplaceable::Unsupported(format!(
            "resampling {resampling} (PIL's value); this build resamples a sized window by \
             nearest ({NEAREST}) or bilinear ({BILINEAR})"
        )));
    }
    if !sized_window && resampling != NEAREST {
        return Err(Unplaceable::Unsupported(format!(
            "resampling {resampling} (PIL's value); this build resamples by nearest ({NEAREST}) \
             only"
        )));
    }
    if corners.iter().flatten().any(|c| !c.is_finite()) {
        return Err(Unplaceable::Invalid("the corners are not finite".into()));
    }
    if resampling == BILINEAR && rectangle(corners).is_none() {
        return Err(Unplaceable::Unsupported(format!(
            "bilinear resampling ({BILINEAR}) places a sized window at an axis-aligned rectangle \
             only; these corners are not one (place it by nearest ({NEAREST}))"
        )));
    }
    match place(corners, width, height) {
        Placement::Mapped(rows) => Ok(Some(rows)),
        Placement::Empty => Ok(None),
        Placement::Refused(why) => Err(Unplaceable::Unsupported(why.into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PICTURE: Key = [1; 16];
    const SMALL: Key = [2; 16];

    fn sizes(key: &Key) -> Option<(u32, u32)> {
        match *key {
            PICTURE => Some((100, 50)),
            SMALL => Some((8, 8)),
            _ => None,
        }
    }

    fn planned(frame: &Frame) -> Result<Stack, DrawError> {
        let canvas = Canvas {
            width: 100,
            height: 50,
            max_dimension: 1000,
            sized: false,
            size_of: &sizes,
        };
        plan(frame, &canvas, 0)
    }

    fn one(layer: Layer) -> Result<Planned, DrawError> {
        planned(&Frame::new([0; 4], vec![layer])).map(|mut s| s.layers.remove(0))
    }

    fn assert_invalid(result: Result<Planned, DrawError>) {
        assert!(
            matches!(result, Err(DrawError::Invalid(_))),
            "{:?}",
            result.err()
        );
    }

    fn assert_unsupported(result: Result<Planned, DrawError>) {
        assert!(
            matches!(result, Err(DrawError::Unsupported(_))),
            "{:?}",
            result.err()
        );
    }

    #[test]
    fn a_layer_with_no_effect_or_none_that_acts_is_planned_plain() {
        for layer in [
            Layer::picture(PICTURE),
            Layer::picture(PICTURE).with_blur(0.0),
            // A depth with no focus.
            Layer::picture(PICTURE).with_depth(3.0),
        ] {
            let planned = one(layer).unwrap();
            assert!(planned.effect.is_none());
            assert_eq!(planned.grow_px, 0.0);
        }
        // At the focus depth.
        let at_focus = planned(
            &Frame::new([0; 4], vec![Layer::picture(PICTURE).with_depth(2.0)]).with_focus(2.0, 5.0),
        )
        .unwrap();
        assert!(at_focus.layers[0].effect.is_none());
    }

    #[test]
    fn a_blurred_or_masked_layer_grows_by_three_sigma_or_its_feather() {
        let blurred = one(Layer::picture(PICTURE).with_blur(2.5)).unwrap();
        assert_eq!(blurred.grow_px, 7.5);
        assert_eq!(blurred.effect.unwrap().blur.len(), 9);
        let masked = one(Layer::picture(PICTURE).with_mask(PICTURE, 4.0, false)).unwrap();
        assert_eq!(masked.grow_px, 4.0);
        let effect = masked.effect.unwrap();
        assert_eq!(effect.blur, vec![crate::effects::ONE]);
        assert_eq!(effect.mask.unwrap().feather.len(), 5);
        let both = one(Layer::picture(PICTURE)
            .with_blur(1.0)
            .with_mask(PICTURE, 10.0, true))
        .unwrap();
        assert_eq!(both.grow_px, 10.0);
    }

    #[test]
    fn effects_out_of_their_range_refuse_the_frame() {
        let nan = f64::NAN;
        assert_invalid(one(Layer::picture(PICTURE).with_blur(-1.0)));
        assert_invalid(one(Layer::picture(PICTURE).with_blur(nan)));
        assert_invalid(one(Layer::picture(PICTURE).with_depth(f64::INFINITY)));
        assert_invalid(one(Layer::picture(PICTURE).with_mask(PICTURE, -2.0, false)));
        // A mask must be a picture of the frame's size, and held.
        assert_invalid(one(Layer::picture(PICTURE).with_mask(SMALL, 2.0, false)));
        assert!(matches!(
            one(Layer::picture(PICTURE).with_mask([9; 16], 2.0, false)),
            Err(DrawError::UnknownKey(_))
        ));
        // Past the frame's longer side (100 px): sigma 34 reaches 102.
        assert_unsupported(one(Layer::picture(PICTURE).with_blur(34.0)));
        assert!(one(Layer::picture(PICTURE).with_blur(33.0)).is_ok());
        assert_unsupported(one(Layer::picture(PICTURE).with_mask(PICTURE, 101.0, false)));
        for focus in [(nan, 1.0), (1.0, -1.0), (1.0, f64::INFINITY)] {
            assert!(matches!(
                planned(
                    &Frame::new([0; 4], vec![Layer::picture(PICTURE)]).with_focus(focus.0, focus.1)
                ),
                Err(DrawError::Invalid(_))
            ));
        }
    }

    #[test]
    fn windows_nest_to_their_limit_and_carry_their_faults_out() {
        let full = [[0.0, 0.0], [100.0, 0.0], [0.0, 50.0], [100.0, 50.0]];
        let mut frame = Frame::new([0; 4], vec![Layer::picture(PICTURE)]);
        for _ in 0..MAX_NESTING {
            frame = Frame::new([0; 4], vec![Layer::window(frame, full, NEAREST)]);
        }
        assert!(planned(&frame).is_ok());
        let deeper = Frame::new([0; 4], vec![Layer::window(frame, full, NEAREST)]);
        assert!(matches!(planned(&deeper), Err(DrawError::Unsupported(_))));
        let faulty = Frame::new([0; 4], vec![Layer::picture(PICTURE).with_blur(-1.0)]);
        let outer = Frame::new([0; 4], vec![Layer::window(faulty, full, NEAREST)]);
        match planned(&outer) {
            Err(DrawError::Invalid(what)) => assert!(what.contains("in its window"), "{what}"),
            other => panic!("{:?}", other.err()),
        }
        // A window whose corners have no area places nothing.
        let line = [[0.0, 0.0], [10.0, 0.0], [5.0, 0.0], [20.0, 0.0]];
        let inner = Frame::new([0; 4], vec![Layer::picture(PICTURE)]);
        assert!(matches!(
            one(Layer::window(inner, line, NEAREST)).unwrap().placed,
            Placed::Nothing
        ));
    }

    #[test]
    fn a_window_is_planned_at_its_own_size_and_its_pictures_and_masks_are_of_it() {
        let corners = [[10.0, 5.0], [18.0, 5.0], [10.0, 13.0], [18.0, 13.0]];
        // SMALL is 8 x 8: a window of that size holds it, and so does a window of no stated size
        // inside it, which takes the size of the frame holding it.
        let small = Frame::new([0; 4], vec![Layer::picture(SMALL)]);
        let nested = Frame::new([0; 4], vec![Layer::window(small.clone(), corners, NEAREST)]);
        for (frame, resampling) in [(small.clone(), NEAREST), (nested, BILINEAR)] {
            match one(Layer::sized_window(frame, corners, resampling, [8, 8]))
                .unwrap()
                .placed
            {
                Placed::Window {
                    size,
                    resampling: placed_by,
                    ..
                } => assert_eq!((size, placed_by), ([8, 8], resampling)),
                _ => panic!("a window placed nothing"),
            }
        }
        let frame_sized = Frame::new([0; 4], vec![Layer::picture(PICTURE)]);
        match one(Layer::sized_window(frame_sized, corners, NEAREST, [8, 8])) {
            Err(DrawError::SizeMismatch(what)) => assert_eq!(
                what,
                "layer 0: in its window, layer 0: a picture of 100 x 50 in a frame of 8 x 8: a \
                 frame's pictures and masks are of its size, and a window's frame is of the \
                 window's size"
            ),
            other => panic!("{:?}", other.err()),
        }
        let masked = Frame::new(
            [0; 4],
            vec![Layer::picture(SMALL).with_mask(PICTURE, 1.0, false)],
        );
        assert!(matches!(
            one(Layer::sized_window(masked, corners, NEAREST, [8, 8])),
            Err(DrawError::SizeMismatch(_))
        ));
        // With no size it is the frame's, 100 x 50, and the refusal is as before windows had
        // sizes.
        match one(Layer::window(small, corners, NEAREST)) {
            Err(DrawError::Invalid(what)) => assert_eq!(
                what,
                "layer 0: in its window, layer 0: a picture of 8 x 8 in a frame of 100 x 50"
            ),
            other => panic!("{:?}", other.err()),
        }
        for size in [[0, 8], [8, 0], [1001, 8]] {
            let empty = Frame::new([0; 4], vec![]);
            assert_invalid(one(Layer::sized_window(empty, corners, NEAREST, size)));
        }
        // A window's blur and feather reach no farther than the window's longer side, and the
        // refusal names the largest that fits: sigma 3 reaches 9 px; a feather of 9.5, 9 px.
        let blurred = Frame::new([0; 4], vec![Layer::picture(SMALL).with_blur(3.0)]);
        let feathered = Frame::new(
            [0; 4],
            vec![Layer::picture(SMALL).with_mask(SMALL, 9.5, false)],
        );
        for (frame, expected) in [
            (
                blurred,
                "layer 0: in its window, layer 0: a blur of sigma 3 px reaches 9 px, past the \
                 window's longer side of 8 px; at this size the largest sigma is 2.66 px (reduce \
                 the blur or show the window larger)",
            ),
            (
                feathered,
                "layer 0: in its window, layer 0: a mask's feather of 9.5 px reaches 9 px, past \
                 the window's longer side of 8 px; at this size the largest feather is 8.99 px \
                 (reduce the feather or show the window larger)",
            ),
        ] {
            match one(Layer::sized_window(frame, corners, NEAREST, [8, 8])) {
                Err(DrawError::Unsupported(what)) => assert_eq!(what, expected),
                other => panic!("{:?}", other.err()),
            }
        }
        // The largest stated values fit.
        let fits = [
            Layer::picture(SMALL).with_blur(2.66),
            Layer::picture(SMALL).with_mask(SMALL, 8.99, false),
        ];
        for layer in fits {
            let frame = Frame::new([0; 4], vec![layer]);
            assert!(one(Layer::sized_window(frame, corners, NEAREST, [8, 8])).is_ok());
        }
    }

    #[test]
    fn outside_a_sized_window_every_refusal_is_as_before_windows_had_sizes() {
        let corners = [[10.0, 5.0], [18.0, 5.0], [10.0, 13.0], [18.0, 13.0]];
        let full = [[0.0, 0.0], [100.0, 0.0], [0.0, 50.0], [100.0, 50.0]];
        let expect = |result: Result<Planned, DrawError>, expected: DrawError| match result {
            Err(found) => assert_eq!(found, expected),
            Ok(_) => panic!("planned, where {expected:?} was expected"),
        };
        expect(
            one(Layer::picture(PICTURE).with_mask(SMALL, 2.0, false)),
            DrawError::Invalid("layer 0: a mask of 8 x 8 in a frame of 100 x 50".into()),
        );
        expect(
            one(Layer::picture(SMALL)),
            DrawError::Invalid("layer 0: a picture of 8 x 8 in a frame of 100 x 50".into()),
        );
        expect(
            one(Layer::picture(PICTURE).with_blur(34.0)),
            DrawError::Unsupported(
                "layer 0: a blur of sigma 34 px reaches 102 px, past the frame's longer side"
                    .into(),
            ),
        );
        expect(
            one(Layer::picture(PICTURE).with_mask(PICTURE, 101.0, false)),
            DrawError::Unsupported(
                "layer 0: a mask's feather of 101 px is past the frame's longer side".into(),
            ),
        );
        let legacy_resampling = DrawError::Unsupported(
            "layer 0: resampling 2 (PIL's value); this build resamples by nearest (0) only".into(),
        );
        expect(
            one(Layer::quad(SMALL, corners, BILINEAR)),
            legacy_resampling.clone(),
        );
        let inner = Frame::new([0; 4], vec![Layer::picture(PICTURE)]);
        expect(one(Layer::window(inner, full, BILINEAR)), legacy_resampling);
    }

    #[test]
    fn a_window_is_placed_by_nearest_or_bilinear_and_a_quad_by_nearest() {
        let corners = [[10.0, 5.0], [18.0, 5.0], [10.0, 13.0], [18.0, 13.0]];
        let small = || Frame::new([0; 4], vec![Layer::picture(SMALL)]);
        for resampling in [NEAREST, BILINEAR] {
            assert!(one(Layer::sized_window(small(), corners, resampling, [8, 8])).is_ok());
        }
        for resampling in [1, 3, 4, 5] {
            assert_unsupported(one(Layer::sized_window(
                small(),
                corners,
                resampling,
                [8, 8],
            )));
        }
        assert_unsupported(one(Layer::quad(SMALL, corners, BILINEAR)));
    }

    #[test]
    fn a_sized_window_is_placed_bilinear_only_at_an_axis_aligned_rectangle() {
        let small = || Frame::new([0; 4], vec![Layer::picture(SMALL)]);
        // Upper left, upper right, lower left, lower right: a diamond, whose left vertex is
        // within a pixel of a cell it never touches.
        let diamond = [[15.01, 15.5], [20.01, 20.5], [10.01, 20.5], [15.01, 25.5]];
        match one(Layer::sized_window(small(), diamond, BILINEAR, [8, 8])) {
            Err(DrawError::Unsupported(what)) => assert_eq!(
                what,
                "layer 0: bilinear resampling (2) places a sized window at an axis-aligned \
                 rectangle only; these corners are not one (place it by nearest (0))"
            ),
            other => panic!("{:?}", other.err()),
        }
        // By nearest it is placed as any quad is.
        assert!(one(Layer::sized_window(small(), diamond, NEAREST, [8, 8])).is_ok());
        // A rectangle off the pixel grid, or mirrored, is axis-aligned.
        let off_grid = [[10.25, 5.5], [18.75, 5.5], [10.25, 13.0], [18.75, 13.0]];
        let mirrored = [[18.0, 5.0], [10.0, 5.0], [18.0, 13.0], [10.0, 13.0]];
        for corners in [off_grid, mirrored] {
            assert!(one(Layer::sized_window(small(), corners, BILINEAR, [8, 8])).is_ok());
        }
    }

    #[test]
    fn a_defocus_whose_sigma_is_not_a_number_plans_as_before_windows_had_sizes() {
        // No aperture times an infinite distance: the resolved sigma, and so its reach, are not a
        // number, which no reach bound refuses; the layer plans with no effect, as it always has.
        let far = Frame::new([0; 4], vec![Layer::picture(PICTURE).with_depth(f64::MAX)])
            .with_focus(-f64::MAX, 0.0);
        let planned = planned(&far).unwrap();
        assert!(planned.layers[0].effect.is_none());
    }
}
