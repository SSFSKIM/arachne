//! A diagnostic report of what the resolve composited at each pixel, for tools that read the
//! core's depth order against what it drew (`Stream::diagnostic_resolves`). Internal: it is not
//! a supported interface, drawing never calls it, and it changes nothing drawn.
//!
//! A pixel's report composites again to the pixel drawn (the frame without its overlay,
//! `Stream::render_under_overlay`) with the resolves' own arithmetic: for the fast and coverage
//! resolves, [`Pixel::units`] blended back to front over [`Pixel::background`] with
//! [`crate::composite::blend`]; for the sample resolve, at each sub-sample `k` the entries of
//! [`Pixel::samples`]`[k]` blended back to front over the background, each at its coverage there
//! ([`Entry::coverage`]`[k]`), then the sub-samples' mean, each channel `(sum + 8) / 16`; then,
//! when [`Pixel::emission`] is given, [`crate::composite::add_emission`].
//!
//! Asked for, a report also holds the pairwise orders the ranks read ([`Pixel::pairs`],
//! [`Pixel::sample_pairs`]): as the ranking decided them, not evaluated again.

pub use crate::composite::PairOrder;
use crate::composite::{Fragment, Resolve, ResolveTrace, SAMPLES};

/// Which part of its leaf a fragment paints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// A path's background stroke.
    BackgroundStroke,
    Fill,
    Stroke,
    /// A triangle of a surface (a mesh, or an instance of one).
    Surface,
    /// Any other mask fragment (a point cloud's block).
    Mask,
}

/// One fragment of a pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Its leaf, `(object, part)`, as the contributor report names it.
    pub leaf: (u32, u32),
    pub layer: Layer,
    /// Its place in the draw order (the order list's index) and its part there, by which the
    /// order breaks ties (`Fragment::order`, `Fragment::sub`).
    pub order: u32,
    pub sub: u32,
    /// Premultiplied, in the frame's byte order: what the resolve composites.
    pub colour: [u8; 4],
    /// Its 8-bit coverage at each sub-sample, as the sample resolve composites it.
    pub coverage: [u8; SAMPLES],
}

/// One unit the fast or coverage resolve blended: a fragment, or a run of one surface's
/// fragments composited as one, with the colour and 8-bit coverage it was blended with.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    /// Its fragments, indices into [`Pixel::entries`], front first.
    pub members: Vec<usize>,
    pub colour: [u8; 4],
    pub coverage: u8,
}

/// What one pixel's resolve composited.
#[derive(Clone, Debug, PartialEq)]
pub struct Pixel {
    pub resolve: Resolve,
    /// The pixel beneath every fragment: the frame's background there.
    pub background: [u8; 4],
    /// Every fragment the pixel holds, culled or not.
    pub entries: Vec<Entry>,
    /// Each sub-sample's entries covering it, front first, indices into [`Pixel::entries`]: the
    /// order the sample resolve composites them in, and for the fast and coverage resolves the
    /// units' order. Culled fragments are in none.
    pub samples: Vec<Vec<usize>>,
    /// The fast and coverage resolves: the units blended, back to front. Empty otherwise.
    pub units: Vec<Unit>,
    /// Emitted light added to the resolved pixel, linear RGB, when any.
    pub emission: Option<[f64; 3]>,
    /// Asked for: every pair of entries as the pixel's rank read it, `a < b` indices into
    /// [`Pixel::entries`]. Empty otherwise, and for a pixel of one entry.
    pub pairs: Vec<PairOrder>,
    /// Whether the sub-samples were ranked (`order::rank_sample`): then [`Pixel::sample_pairs`]
    /// holds, at each sub-sample, every pair of the entries it ranked; otherwise every pair of
    /// those it composited ([`Pixel::samples`]).
    pub sample_ranked: bool,
    /// Asked for, at a pixel of more than one entry: each sub-sample's pairwise orders as its rank
    /// reads them (depth where depth decides, paint between a path's fill and its own stroke, a
    /// tie otherwise). Empty otherwise.
    pub sample_pairs: Vec<Vec<PairOrder>>,
}

impl Pixel {
    pub(crate) fn of(
        frags: &[Fragment],
        trace: ResolveTrace,
        background: [u8; 4],
        emission: Option<[f64; 3]>,
        leaf: impl Fn(&Fragment) -> (u32, u32),
    ) -> Pixel {
        let entries = frags
            .iter()
            .enumerate()
            .map(|(i, f)| Entry {
                leaf: leaf(f),
                layer: if f.path_layer {
                    match f.sub % 3 {
                        0 => Layer::BackgroundStroke,
                        1 => Layer::Fill,
                        _ => Layer::Stroke,
                    }
                } else if f.surface != 0 {
                    Layer::Surface
                } else {
                    Layer::Mask
                },
                order: f.order,
                sub: f.sub,
                colour: f.colour,
                coverage: trace.coverage[i],
            })
            .collect();
        Pixel {
            resolve: trace.resolve,
            background,
            entries,
            samples: trace.samples,
            units: trace
                .units
                .into_iter()
                .map(|(members, colour, coverage)| Unit {
                    members,
                    colour,
                    coverage,
                })
                .collect(),
            emission,
            pairs: trace.pairs,
            sample_ranked: trace.sample_ranked,
            sample_pairs: trace.sample_pairs,
        }
    }
}
