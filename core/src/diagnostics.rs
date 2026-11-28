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
//!
//! Asked for apart ([`Pixel::reads`]), it holds what the order and the resolves read of each
//! fragment, which the entries do not carry: its 8-bit coverage of the whole pixel, its depth at
//! each sub-sample and that depth's error bound as the order reads them, the depths it is read at
//! against another fragment where they are not its own, and each sub-sample's rank of every
//! fragment covering it before any cull. A report not asked for them reads nothing more than
//! before.

pub use crate::composite::PairOrder;
use crate::composite::{
    Depths, Fragment, Relation, Resolve, ResolveTrace, SAMPLES, strokes_on_own_fills,
};
use crate::order;

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
    /// Each sub-sample's entries covering it, front first, indices into [`Pixel::entries`]. For
    /// the fast and coverage resolves, the units' entries in the units' order: what they
    /// composited, culled fragments in none. For the sample resolve, its sub-sample ranks as it
    /// composites them, which hold what the opaque drop removed where the cull ranked the
    /// sub-samples before dropping (those lie behind a layer opaque and covering the pixel whole,
    /// which blends over them exactly), and not where it dropped by paint order alone (a flat
    /// pixel). Every entry covering a sub-sample, culled or not: [`Reads::samples_all`].
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
    /// Asked for apart: what the order and the resolves read of each entry. `None` otherwise.
    pub reads: Option<Reads>,
}

/// What the order and the resolves read of a pixel's entries ([`Pixel::reads`]), each list
/// indexed as [`Pixel::entries`] is.
#[derive(Clone, Debug, PartialEq)]
pub struct Reads {
    /// Each entry's 8-bit coverage of the whole pixel: a mask's count, a path's exact area
    /// rounded once (never its sub-samples' bytes summed). What the fast and coverage resolves
    /// blend a unit of the entry alone with.
    pub coverage8: Vec<u8>,
    /// Each entry's depth at each sub-sample, as the order reads it against a fragment it shares
    /// no plane with: its producer's, a stroke's taking its own fill's where both cover and the
    /// fill is nearer (`composite::strokes_on_own_fills`).
    pub depths: Vec<[f64; SAMPLES]>,
    /// The error bound of each entry's [`Reads::depths`], as the order reads them.
    pub err: Vec<f64>,
    /// Asked for with the pairwise orders: every read of an entry's depths against another entry
    /// that is not its own [`Reads::depths`] (on a registered plane both lie on,
    /// `composite::Depths::against`). Empty otherwise.
    pub against: Vec<Against>,
    /// Each sub-sample's entries covering it (coverage above zero there), every one of them,
    /// front first as its rank ([`crate::order::rank_sample`]) orders them before any cull: a
    /// superset of [`Pixel::samples`], whose missing entries the opaque drop removed.
    pub samples_all: Vec<Vec<usize>>,
    /// Asked for with the pairwise orders: whether the resolve took the flat path (one depth row
    /// for every entry: paint order decides every pair, and no pair is read at its sub-samples;
    /// [`Reads::relations`] is then empty). False otherwise.
    pub flat: bool,
    /// Asked for with the pairwise orders: every pair of entries `a < b` sharing a sub-sample,
    /// with its orders at each sub-sample as the resolve read them before any rank or cull
    /// (`order::shared`: where both cover, where each is nearer, and each against the other in
    /// front, behind and strongly so). What [`Pixel::pairs`] is decided from
    /// (`order::Shared::decided`) and the sub-sample ranks and the cull read. Empty otherwise.
    pub relations: Vec<Relation>,
}

/// One entry's depths as read against another ([`Reads::against`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Against {
    /// The entry read, and the one it is read against: indices into [`Pixel::entries`].
    pub entry: usize,
    pub other: usize,
    /// Its depth at each sub-sample against `other`, and that depth's error bound.
    pub depths: [f64; SAMPLES],
    pub err: f64,
}

impl Reads {
    /// The reads of `frags`, given each one's coverage at every sub-sample (`coverage`) and its
    /// depths as the passes give them (`depths`), as the resolve reads them
    /// (`composite::resolve_pixel_traced`): a stroke takes its own fill's depth first, then each
    /// pair is ordered at each sub-sample both cover from the depths each is read at against the
    /// other.
    fn of(frags: &[Fragment], trace: &ResolveTrace, depths: &[Depths]) -> Reads {
        let (coverage, decisions) = (&trace.coverage, trace.decisions);
        let mut rows = depths.to_vec();
        strokes_on_own_fills(frags, &mut rows, |i| coverage[i]);
        Reads {
            coverage8: frags.iter().map(|f| f.coverage8()).collect(),
            depths: rows.iter().map(|d| d.samples).collect(),
            err: rows.iter().map(|d| d.err).collect(),
            against: if decisions {
                read_against(frags, &rows)
            } else {
                Vec::new()
            },
            samples_all: ranked_samples(frags, coverage, &rows),
            flat: trace.flat,
            relations: trace.relations.clone(),
        }
    }
}

/// Every read of a fragment's depths against another that is not its own row: where
/// [`Depths::against`] reads it on a plane both lie on.
fn read_against(frags: &[Fragment], rows: &[Depths]) -> Vec<Against> {
    let mut out = Vec::new();
    for (a, fa) in frags.iter().enumerate() {
        for (b, fb) in frags.iter().enumerate() {
            if a == b {
                continue;
            }
            let (depths, err) = rows[a].against(fa, fb);
            if !std::ptr::eq(depths, &rows[a].samples) {
                out.push(Against {
                    entry: a,
                    other: b,
                    depths: *depths,
                    err,
                });
            }
        }
    }
    out
}

/// Each sub-sample's fragments covering it, front first, ranked as the resolve ranks a
/// sub-sample ([`order::rank_sample`]) from the orders [`order::shared`] reads at it, each
/// fragment at its depths against the other ([`Depths::against`]): every fragment present there,
/// whatever a cull then drops.
// The sub-sample `k` indexes each fragment's coverage row.
#[allow(clippy::needless_range_loop)]
fn ranked_samples(
    frags: &[Fragment],
    coverage: &[[u8; SAMPLES]],
    rows: &[Depths],
) -> Vec<Vec<usize>> {
    let n = frags.len();
    let seen = |i: usize, other: usize| {
        let (depth, err) = rows[i].against(&frags[i], &frags[other]);
        order::Seen {
            fragment: &frags[i],
            coverage: &coverage[i],
            depth,
            err,
        }
    };
    let mut at = vec![order::AtSamples::default(); n * n];
    for a in 0..n {
        for b in a + 1..n {
            let shared = order::shared(seen(a, b), seen(b, a));
            (at[a * n + b], at[b * n + a]) = (shared.ab, shared.ba);
        }
    }
    let bit = |mask: u16, k: usize| mask >> k & 1 != 0;
    let mut out = Vec::with_capacity(SAMPLES);
    let mut covering = Vec::new();
    for k in 0..SAMPLES {
        covering.clear();
        covering.extend((0..n).filter(|&i| coverage[i][k] != 0));
        let mut ranked = Vec::new();
        order::rank_sample(
            &covering,
            |a, b| a != b && bit(at[a * n + b].front, k),
            |a, b| bit(at[a * n + b].strong, k),
            |a, b| order::painted(&frags[a], &frags[b]),
            &mut ranked,
        );
        out.push(ranked);
    }
    out
}

impl Pixel {
    /// The report of a pixel holding `frags`, whose resolve composited `trace`, over
    /// `background`, with `emission` added after; with `depths` (each fragment's, as the passes
    /// give them), its [`Pixel::reads`] too.
    pub(crate) fn of(
        frags: &[Fragment],
        trace: ResolveTrace,
        depths: Option<&[Depths]>,
        background: [u8; 4],
        emission: Option<[f64; 3]>,
        leaf: impl Fn(&Fragment) -> (u32, u32),
    ) -> Pixel {
        let reads = depths.map(|d| Reads::of(frags, &trace, d));
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
            reads,
        }
    }
}
