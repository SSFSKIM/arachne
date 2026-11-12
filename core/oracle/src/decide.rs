//! One frame's decisions: every fragment the order draws and its verdict at every sample of the
//! window, or the frame's refusal.

use arachne_core::xf::{Camera, Delta, View};

use crate::camera::FrameMap;
use crate::colour::{self, Light};
use crate::cover::{self, PixelCover, SAMPLE_OFFSETS, Window};
use crate::draw::{self, Item, Layer};
use crate::flatten;
use crate::keys::{self, Model, Offered};
use crate::label;
use crate::order::{Frag, Scene};
use crate::out::{
    COLOUR_GRADIENT, COLOUR_PER_PIXEL, COLOUR_SOLID, ColourRow, CoverRow, Decisions,
    FLAG_UNBOUNDED, FLAG_ZERO_ALPHA, Fragment, Header, LabelRow, NONE_U32, PixelColour, Refused,
    STATUS_DECIDED, STATUS_UNABLE,
};
use crate::pairs;
use crate::shift::Shift;
use crate::source::WireDelta;
use crate::stream::{Reason, Refusal, State};
use std::collections::HashMap;

/// A drawn path's projected centre where it is kept facing the camera, and whether it is a face
/// of a face group: what the label rule reads of it (contract §10).
type PathInfo = (Option<[f64; 2]>, bool);

/// What a fragment's colour is computed from (contract §9).
enum Spec {
    Solid([u8; 4]),
    /// A lit triangle: its corners' colours, its corners on the pixel plane, its alpha, and
    /// whether the corners are linear light.
    Lit {
        corners: [[f64; 3]; 3],
        at: [[f64; 2]; 3],
        alpha: f64,
        linear: bool,
    },
    /// A colour this oracle does not compute (a gradient), with the bounds of its alpha byte at
    /// any pixel: what the label rule reads of it.
    Gradient {
        alpha: (u8, u8),
    },
}

/// A gradient's alpha byte at a pixel lies between its stops' bytes (each `cairo_premultiply@1`'s
/// alpha at the record's group opacity, as a solid's is), one level wider each way for the
/// walker's single-precision interpolation; pad extend keeps it within them (contract §9). A
/// shaded layer's stops are its first two rows.
fn gradient_alpha(rows: &[[f64; 4]], opacity: f64, shaded: bool) -> (u8, u8) {
    let stops = if shaded {
        &rows[..rows.len().min(2)]
    } else {
        rows
    };
    let bytes = stops.iter().map(|r| colour::premultiply(*r, opacity)[3]);
    let lo = bytes.clone().min().unwrap_or(0);
    let hi = bytes.max().unwrap_or(255);
    (lo.saturating_sub(1), hi.saturating_add(1))
}

/// Why the oracle could not decide a frame at all (not a refusal: a usage error).
#[derive(Debug)]
pub enum DecideError {
    Xf(arachne_core::xf::wire::XfError),
    NoSuchSeq(u64),
}

impl std::fmt::Display for DecideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecideError::Xf(e) => write!(f, "the XF file does not read: {e:?}"),
            DecideError::NoSuchSeq(seq) => write!(f, "no delta has seq {seq}"),
        }
    }
}

fn refused(r: &Refusal) -> Refused {
    Refused {
        reason: r.reason.code(),
        kind: r.kind.unwrap_or(0xff),
        cause: r.stream.map_or(0, |(_, cause)| cause as u8),
        entry: r.entry.unwrap_or(NONE_U32),
        slot: r.slot.unwrap_or(NONE_U32),
        object: r.object.unwrap_or(NONE_U32),
        bits: r.bits,
        at_seq: r.stream.map_or(u64::MAX, |(seq, _)| seq),
        detail: r.detail.clone(),
    }
}

/// The decisions at the delta `seq` of `deltas` (built in memory, so carrying no source section),
/// applied in order from an empty stream. A delta that is not valid XF changes nothing; the frame
/// drawn at it is refused.
pub fn decide_at(
    deltas: &[Delta],
    seq: u64,
    window: Option<Window>,
) -> Result<Decisions, DecideError> {
    let wire: Vec<WireDelta> = deltas.iter().cloned().map(WireDelta::of).collect();
    decide_at_wire(&wire, seq, window)
}

/// The decisions at the delta `seq` of a file's deltas as read (`source::read_file`), applied in
/// order from an empty stream. A delta that does not read, or is not valid XF, changes nothing and
/// its frame is refused; one refused by an XF 1.5 source code refuses its frame and every later one.
pub fn decide_at_wire(
    deltas: &[WireDelta],
    seq: u64,
    window: Option<Window>,
) -> Result<Decisions, DecideError> {
    let mut state = State::new();
    for d in deltas {
        let applied = state.apply_wire(d);
        if d.seq == seq {
            return Ok(match applied {
                Ok(()) => decide(&state, d.version, seq, window),
                Err(r) => unable(&state, d.version, seq, window, &[r]),
            });
        }
    }
    Err(DecideError::NoSuchSeq(seq))
}

fn header(view: Option<&View>, xf: (u16, u16), seq: u64, window: Option<Window>) -> Header {
    let (w, h) = view.map_or((0, 0), |v| (v.pixel_width, v.pixel_height));
    let window = match window {
        Some([x0, y0, x1, y1]) => [
            x0.min(w),
            y0.min(h),
            x1.min(w).max(x0.min(w)),
            y1.min(h).max(y0.min(h)),
        ],
        None => [0, 0, w, h],
    };
    Header {
        xf,
        width: w,
        height: h,
        window,
        status: STATUS_DECIDED,
        seq,
        offsets: SAMPLE_OFFSETS,
    }
}

fn unable(
    state: &State,
    xf: (u16, u16),
    seq: u64,
    window: Option<Window>,
    refusals: &[Refusal],
) -> Decisions {
    Decisions {
        header: Header {
            status: STATUS_UNABLE,
            ..header(state.view.as_ref(), xf, seq, window)
        },
        fragments: Vec::new(),
        covers: Vec::new(),
        bands: Vec::new(),
        pairs: Vec::new(),
        colours: Vec::new(),
        pixel_colours: Vec::new(),
        labels: Vec::new(),
        refused: refusals.iter().map(refused).collect(),
    }
}

/// What the frame the state draws now should hold, over `window` (the whole frame by default).
pub fn decide(state: &State, xf: (u16, u16), seq: u64, window: Option<Window>) -> Decisions {
    decide_with_labels(state, xf, seq, window).0
}

/// [`decide`], and each camera-facing label pair's decision (contract §10), which counts compare
/// and the file does not hold.
pub fn decide_with_labels(
    state: &State,
    xf: (u16, u16),
    seq: u64,
    window: Option<Window>,
) -> (Decisions, Vec<label::PairDecision>) {
    let mut refusals = state.refusals();
    if xf.0 != crate::stream::KNOWN_XF.0 || xf.1 > crate::stream::KNOWN_XF.1 {
        refusals.push(Refusal {
            reason: Reason::XfNewer,
            entry: None,
            slot: None,
            object: None,
            kind: None,
            bits: 0,
            stream: None,
            detail: format!("the frame's delta is XF {}.{}", xf.0, xf.1),
        });
    }
    if !refusals.is_empty() {
        return (unable(state, xf, seq, window, &refusals), Vec::new());
    }
    let view = state.view.as_ref().expect("a frame not refused has a view");
    let header = header(Some(view), xf, seq, window);
    let window = header.window;
    let built = build(view, state, window);
    // Qualification is one decision per pair, made over the whole frame before any tile (contract
    // §10): a window's label facts are the whole frame's at its pixels.
    let whole = [0, 0, view.pixel_width, view.pixel_height];
    let (label_of, labels, label_pairs) =
        if window == whole || !matches!(view.camera, Camera::ThreeD(_)) {
            labels_of(view, state, &built)
        } else {
            let full = build(view, state, whole);
            crop(&full, &built, window, labels_of(view, state, &full))
        };
    let Built {
        mut fragments,
        mut covers,
        frags,
        offered,
        specs,
        extra,
        ..
    } = built;
    // The registry with the records that paint nothing offered too, where there are any.
    let with_invisible = (!extra.is_empty()).then(|| {
        let all = offered.iter().cloned().chain(extra).collect();
        Scene::new(frags.clone(), all)
    });
    let scene = Scene::new(frags, offered);
    let labelled: Vec<bool> = label_of.iter().map(|l| *l != NONE_U32).collect();
    let (bands, pairs) = pairs::frame(
        &scene,
        with_invisible.as_ref().unwrap_or(&scene),
        &labelled,
        &mut covers,
    );
    let (colours, pixel_colours) = colours(&specs, &covers);
    for (f, l) in fragments.iter_mut().zip(label_of) {
        f.label = l;
    }
    let decisions = Decisions {
        header,
        fragments,
        covers,
        bands,
        pairs,
        colours,
        pixel_colours,
        labels,
        refused: Vec::new(),
    };
    (decisions, label_pairs)
}

/// A frame's fragments over a window, and what the label rule reads of their paths.
struct Built {
    fragments: Vec<Fragment>,
    covers: Vec<CoverRow>,
    frags: Vec<Frag>,
    offered: Vec<Offered>,
    specs: Vec<Spec>,
    outlines: HashMap<(u32, u32), Vec<flatten::Ring>>,
    path_info: HashMap<(u32, u32), PathInfo>,
    /// The registry facts of the records that paint nothing.
    extra: Vec<Offered>,
    /// The bounded fragments that cover no pixel of the window, which `fragments` leaves out.
    uncovered: Vec<Fragment>,
}

/// Every fragment the frame draws over `window`, with its rows, depth and colour's spec.
fn build(view: &View, state: &State, window: Window) -> Built {
    let size = [view.pixel_width, view.pixel_height];
    let items = draw::items(
        view,
        &state.records,
        &state.order,
        &state.shapes,
        &state.paints,
    );
    let mut fragments = Vec::new();
    let mut covers = Vec::new();
    let mut frags: Vec<Frag> = Vec::new();
    let mut offered: Vec<Offered> = Vec::new();
    let mut specs: Vec<Spec> = Vec::new();
    let mut uncovered: Vec<Fragment> = Vec::new();
    // A fragment, its depth and its rows; `None` for an unbounded one, which is `truth_unable` at
    // every sample of the window and carries no rows (contract §6: never absent, never bounded).
    let mut add = |fragment: Fragment, depth: Frag, spec: Spec, rows: Option<Vec<PixelCover>>| {
        let id = fragments.len() as u32;
        let Some(rows) = rows else {
            specs.push(spec);
            frags.push(Frag {
                model: Model::Unbounded,
                row: None,
                lines: None,
                ..depth
            });
            fragments.push(Fragment {
                id,
                flags: fragment.flags | FLAG_UNBOUNDED,
                epsilon: f64::INFINITY,
                bbox: window.map(|v| v as i32),
                ..fragment
            });
            return;
        };
        if rows.is_empty() {
            uncovered.push(fragment);
            return;
        }
        frags.push(depth);
        specs.push(spec);
        let mut bbox = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
        for r in &rows {
            bbox = [
                bbox[0].min(r.x as i32),
                bbox[1].min(r.y as i32),
                bbox[2].max(r.x as i32 + 1),
                bbox[3].max(r.y as i32 + 1),
            ];
        }
        fragments.push(Fragment {
            id,
            bbox,
            ..fragment
        });
        covers.extend(rows.into_iter().map(|r| CoverRow {
            x: r.x,
            y: r.y,
            fragment: id,
            cover: r.cover,
            cell_lo: r.cell_lo,
            cell_hi: r.cell_hi,
            pixel_lo: r.pixel_lo,
            pixel_hi: r.pixel_hi,
            pixel_area: r.pixel_area,
            cell_area: r.cell_area,
            depth: [f64::NAN; 16],
            band: [f64::NAN; 16],
        }));
    };
    let mut group = 0u32;
    let mut outlines: HashMap<(u32, u32), Vec<flatten::Ring>> = HashMap::new();
    let mut path_info: HashMap<(u32, u32), PathInfo> = HashMap::new();
    for item in &items {
        match item {
            Item::Path(p) => {
                let c = &p.common;
                let eps = Shift::of(c, view, &p.map, &p.shown).of_points(&p.shown);
                let rings = flatten::layer_rings(p, size);
                // What the label rule reads of the path (contract §10): its layers' outlines, its
                // projected centre where it is kept facing the camera, whether it is a face.
                let centre = match &c.placement {
                    crate::camera::Placement::FixedOrientation { centre, .. } => {
                        let s = c.placement.show(c.projection.as_ref(), *centre);
                        Some(p.map.map(s[0], s[1]))
                    }
                    _ => None,
                };
                for (k, r) in rings.iter().enumerate() {
                    if let Some(r) = r {
                        outlines.insert((c.entry, 3 * p.index + k as u32), r.clone());
                    }
                }
                path_info.insert((c.entry, p.index), (centre, p.face));
                let pd = keys::path_depth(view, p, &rings);
                let reg = pd.offered.clone().map(|o| {
                    offered.push(o);
                    offered.len() - 1
                });
                let planeless = pd.offered.as_ref().is_some_and(|o| o.flat.is_none());
                for (k, layer_rings) in rings.iter().enumerate() {
                    let Some(layer_rings) = layer_rings else {
                        continue;
                    };
                    if layer_rings.iter().all(|r| r.is_empty()) {
                        continue;
                    }
                    let layer = [Layer::BackgroundStroke, Layer::Fill, Layer::Stroke][k];
                    let sub = 3 * p.index + k as u32;
                    let model = pd.layers[k].clone();
                    let depth = Frag {
                        entry: c.entry,
                        sub,
                        layer,
                        group,
                        lines: (planeless && k != 1)
                            .then(|| pairs::lines_of(&model))
                            .flatten(),
                        model,
                        row: pd.row.clone(),
                        offered: reg,
                    };
                    let bounded = eps.is_finite()
                        && layer_rings
                            .iter()
                            .flatten()
                            .flatten()
                            .all(|v| v.is_finite());
                    let spec = match &p.rows[k] {
                        // One colour, unshaded: one row, or rows that are all that colour.
                        Some(r) if one_colour(r) && !p.shaded => {
                            Spec::Solid(colour::premultiply(r[0], c.record.opacity))
                        }
                        Some(r) => Spec::Gradient {
                            alpha: gradient_alpha(r, c.record.opacity, p.shaded),
                        },
                        None => Spec::Gradient { alpha: (0, 255) },
                    };
                    add(
                        fragment(c, layer, sub, group, eps),
                        depth,
                        spec,
                        bounded.then(|| cover::path_layer(layer_rings, eps, window)),
                    );
                }
                group += 1;
            }
            Item::Mesh(m) => {
                let c = &m.common;
                let shown: Vec<[f64; 3]> = m.triangles.iter().flatten().copied().collect();
                let shift = Shift::of(c, view, &m.map, &shown);
                for (i, t) in m.triangles.iter().enumerate() {
                    let corners = t.map(|q| map(&m.map, q));
                    let eps = t.map(|q| shift.at(q));
                    let worst = eps.iter().copied().fold(0.0, crate::bound::max);
                    let bounded =
                        worst.is_finite() && corners.iter().flatten().all(|v| v.is_finite());
                    let (model, o) = keys::triangle_depth(view, m, i);
                    let reg = o.map(|o| {
                        offered.push(o);
                        offered.len() - 1
                    });
                    let depth = Frag {
                        entry: c.entry,
                        sub: m.sub + i as u32,
                        layer: Layer::Mesh,
                        group: NONE_U32,
                        model,
                        row: None,
                        offered: reg,
                        lines: None,
                    };
                    let spec = mesh_spec(view, m, i, corners);
                    add(
                        fragment(c, Layer::Mesh, m.sub + i as u32, NONE_U32, worst),
                        depth,
                        spec,
                        bounded.then(|| cover::triangle(corners, eps, window)),
                    );
                }
            }
            Item::Points(pts) => {
                let c = &pts.common;
                let map = FrameMap::of_paths(view);
                let shown: Vec<[f64; 3]> = pts.points.iter().map(|q| q.shown).collect();
                let shift = Shift::of(c, view, &map, &shown);
                let (t, n) = (pts.side, pts.count as i64);
                let lo = (-t as f64 / 2.0).floor() as i64 + 1;
                // A side below 1 draws no block (contract §3).
                let points = if t < 1 { &[][..] } else { &pts.points[..] };
                for q in points {
                    let bytes = pts
                        .colours
                        .get(q.index as usize)
                        .and_then(|rgba| colour::point_bytes(*rgba, c.record.opacity))
                        .unwrap_or([0; 4]);
                    let model = keys::point_depth(view, pts, q.placed);
                    let depth = |sub: u32| Frag {
                        entry: c.entry,
                        sub,
                        layer: Layer::Points,
                        group: NONE_U32,
                        model: model.clone(),
                        row: None,
                        offered: None,
                        lines: None,
                    };
                    let eps = shift.at(q.shown);
                    if !(eps.is_finite() && q.at.iter().all(|v| v.is_finite())) {
                        // Its truncation is not bounded: one fragment, its first offset's `sub`.
                        add(
                            fragment(c, Layer::Points, q.index, NONE_U32, eps),
                            depth(q.index),
                            Spec::Solid(bytes),
                            None,
                        );
                        continue;
                    }
                    for b in cover::block(q.at, eps, t, size, window) {
                        let (o, o2) = b.offset;
                        let index = (o - lo) * t + (o2 - lo);
                        let sub = (index * n + q.index as i64) as u32;
                        add(
                            fragment(c, Layer::Points, sub, NONE_U32, eps),
                            depth(sub),
                            Spec::Solid(bytes),
                            Some(vec![b.cover]),
                        );
                    }
                }
            }
        }
    }
    let extra = invisible(view, state, window, &mut fragments, &mut specs);
    covers.sort_by_key(|c| (c.y, c.x, c.fragment));
    for (f, d) in fragments.iter_mut().zip(&frags) {
        f.plane = d.offered.map_or(NONE_U32, |i| i as u32);
    }
    Built {
        fragments,
        covers,
        frags,
        offered,
        specs,
        outlines,
        path_info,
        extra,
        uncovered,
    }
}

type LabelFacts = (Vec<u32>, Vec<LabelRow>, Vec<label::PairDecision>);

/// The label rule over the fragments `b` holds (contract §10).
fn labels_of(view: &View, state: &State, b: &Built) -> LabelFacts {
    let is_path = |f: &Fragment| f.layer <= Layer::Stroke as u8;
    let outline: Vec<Option<Vec<flatten::Ring>>> = b
        .fragments
        .iter()
        .map(|f| {
            is_path(f)
                .then(|| b.outlines.get(&(f.entry, f.sub)).cloned())
                .flatten()
        })
        .collect();
    let info = |f: &Fragment| {
        is_path(f)
            .then(|| b.path_info.get(&(f.entry, f.sub / 3)).copied())
            .flatten()
    };
    let centre: Vec<Option<[f64; 2]>> = b
        .fragments
        .iter()
        .map(|f| info(f).and_then(|i| i.0))
        .collect();
    let face: Vec<bool> = b
        .fragments
        .iter()
        .map(|f| info(f).is_some_and(|i| i.1))
        .collect();
    let alpha: Vec<(u8, u8)> = b
        .specs
        .iter()
        .map(|s| match s {
            Spec::Solid(b) => (b[3], b[3]),
            Spec::Lit { alpha, .. } => {
                let a = draw::alpha_byte(*alpha) as u8;
                (a, a)
            }
            Spec::Gradient { alpha } => *alpha,
        })
        .collect();
    let uncovered: Vec<label::Uncovered> = b
        .uncovered
        .iter()
        .filter(|f| f.layer == Layer::Fill as u8)
        .filter_map(|f| {
            Some(label::Uncovered {
                slot: f.slot,
                outline: b.outlines.get(&(f.entry, f.sub))?.clone(),
                face: info(f).is_some_and(|i| i.1),
            })
        })
        .collect();
    label::place_with_pairs(&label::Input {
        view,
        state,
        fragments: &b.fragments,
        frags: &b.frags,
        covers: &b.covers,
        outlines: &outline,
        alpha: &alpha,
        centre: &centre,
        face: &face,
        uncovered: &uncovered,
    })
}

/// The whole frame's label facts (`full`'s) at the window's pixels, its fragments renumbered as
/// the window's (`tile`'s) are.
fn crop(full: &Built, tile: &Built, window: Window, facts: LabelFacts) -> LabelFacts {
    let (label_of, rows, pairs) = facts;
    let key = |f: &Fragment| (f.entry, f.sub, f.layer, f.flags & FLAG_ZERO_ALPHA);
    // One key may name several fragments, each at its own pixels: a point block's pixels under its
    // truncation candidates share a `sub` across pixels. Each is found by a pixel it stands at.
    let by_key = |b: &Built| {
        let mut m: HashMap<_, Vec<(u32, [i32; 4])>> = HashMap::new();
        for f in &b.fragments {
            m.entry(key(f)).or_default().push((f.id, f.bbox));
        }
        m
    };
    let (in_tile, in_full) = (by_key(tile), by_key(full));
    let at = |m: &HashMap<_, Vec<(u32, [i32; 4])>>, f: &Fragment, x: i32, y: i32| {
        let named = &m[&key(f)];
        named
            .iter()
            .find(|(_, b)| (b[0]..b[2]).contains(&x) && (b[1]..b[3]).contains(&y))
            .unwrap_or(&named[0])
            .0
    };
    let tile_label = tile
        .fragments
        .iter()
        .map(|f| label_of[at(&in_full, f, f.bbox[0], f.bbox[1]) as usize])
        .collect();
    let inside =
        |x: u32, y: u32| x >= window[0] && x < window[2] && y >= window[1] && y < window[3];
    let tile_rows = rows
        .into_iter()
        .filter(|r| inside(r.x, r.y))
        .map(|r| LabelRow {
            fragment: at(
                &in_tile,
                &full.fragments[r.fragment as usize],
                r.x as i32,
                r.y as i32,
            ),
            ..r
        })
        .collect();
    (tile_label, tile_rows, pairs)
}

/// What the frame drops for painting nothing (`alpha_round@1`): the registry facts of each such
/// record, which the counterfactual sets offer (contract §5), and a fragment flagged
/// [`FLAG_ZERO_ALPHA`] for each mesh triangle whose alpha byte is 0 and that would cover a pixel
/// of the window (§1, `zero_alpha_fragment`). Those fragments come after every other, carry no
/// rows and no depth, and their colour is `(0, 0, 0, 0)`.
fn invisible(
    view: &View,
    state: &State,
    window: Window,
    fragments: &mut Vec<Fragment>,
    specs: &mut Vec<Spec>,
) -> Vec<Offered> {
    let items = draw::items_of(
        view,
        &state.records,
        &state.order,
        &state.shapes,
        &state.paints,
        draw::Which::Invisible,
    );
    let mut extra = Vec::new();
    for item in &items {
        match item {
            Item::Path(p) => {
                if let Some(o) = keys::path_depth(view, p, &[None, None, None]).offered {
                    extra.push(o);
                }
            }
            Item::Mesh(m) => {
                let c = &m.common;
                let shown: Vec<[f64; 3]> = m.triangles.iter().flatten().copied().collect();
                let shift = Shift::of(c, view, &m.map, &shown);
                for (i, t) in m.triangles.iter().enumerate() {
                    if let (_, Some(o)) = keys::triangle_depth(view, m, i) {
                        extra.push(o);
                    }
                    let corners = t.map(|q| map(&m.map, q));
                    let eps = t.map(|q| shift.at(q));
                    let worst = eps.iter().copied().fold(0.0, crate::bound::max);
                    if !(worst.is_finite() && corners.iter().flatten().all(|v| v.is_finite())) {
                        continue;
                    }
                    let rows = cover::triangle(corners, eps, window);
                    if rows.is_empty() {
                        continue;
                    }
                    let mut bbox = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
                    for r in &rows {
                        bbox = [
                            bbox[0].min(r.x as i32),
                            bbox[1].min(r.y as i32),
                            bbox[2].max(r.x as i32 + 1),
                            bbox[3].max(r.y as i32 + 1),
                        ];
                    }
                    let sub = m.sub + i as u32;
                    fragments.push(Fragment {
                        id: fragments.len() as u32,
                        flags: FLAG_ZERO_ALPHA,
                        bbox,
                        ..fragment(c, Layer::Mesh, sub, NONE_U32, worst)
                    });
                    specs.push(Spec::Solid([0; 4]));
                }
            }
            Item::Points(_) => {}
        }
    }
    extra
}

/// Whether a layer's rows are one colour: at least one row, every row bit for bit the first.
fn one_colour(rows: &[[f64; 4]]) -> bool {
    rows.first().is_some_and(|f| {
        rows.iter()
            .all(|r| r.iter().zip(f).all(|(a, b)| a.to_bits() == b.to_bits()))
    })
}

/// A mesh triangle's colour source: a solid, or lit where the camera shades it.
fn mesh_spec(view: &View, m: &draw::MeshItem, i: usize, at: [[f64; 2]; 3]) -> Spec {
    let opacity = m.common.record.opacity;
    let (true, Some(pr), arachne_core::xf::Camera::ThreeD(cam)) =
        (m.shaded, m.common.projection.as_ref(), &view.camera)
    else {
        return Spec::Solid(colour::premultiply(m.fill, opacity));
    };
    let light = Light {
        model: &view.light,
        source: cam.light_source,
        eye: colour::eye(pr),
    };
    let normals = m.normals.as_ref().map(|n| &n[i]);
    match colour::lit_corners(&light, m.fill, &m.placed[i], normals) {
        Some(corners) => Spec::Lit {
            corners,
            at,
            alpha: m.fill[3] * opacity,
            linear: light.linear(),
        },
        None => {
            let a = draw::alpha_byte(m.fill[3] * opacity) as u8;
            Spec::Gradient { alpha: (a, a) }
        }
    }
}

/// Each fragment's colour row, and each lit fragment's bytes at its cover rows: at the centroid of
/// the samples it covers, or over every sample it may cover where one is `Either`.
fn colours(specs: &[Spec], covers: &[CoverRow]) -> (Vec<ColourRow>, Vec<PixelColour>) {
    let rows = specs
        .iter()
        .enumerate()
        .map(|(id, s)| {
            let (kind, rgba) = match s {
                Spec::Solid(b) => (COLOUR_SOLID, *b),
                Spec::Lit { .. } => (COLOUR_PER_PIXEL, [0; 4]),
                Spec::Gradient { .. } => (COLOUR_GRADIENT, [0; 4]),
            };
            ColourRow {
                fragment: id as u32,
                kind,
                rgba,
            }
        })
        .collect();
    let mut pixel = Vec::new();
    for c in covers {
        let Spec::Lit {
            corners,
            at,
            alpha,
            linear,
        } = &specs[c.fragment as usize]
        else {
            continue;
        };
        let (x, y) = (c.x as i64, c.y as i64);
        let point = |s: usize| cover::sample_point(x, y, s);
        let yes: Vec<usize> = (0..16).filter(|s| c.cover[*s] == 1).collect();
        let points: Vec<[f64; 2]> = if c.cover.contains(&2) || yes.is_empty() {
            (0..16).filter(|s| c.cover[*s] != 0).map(point).collect()
        } else {
            let n = yes.len() as f64;
            let sum = yes.iter().fold([0.0, 0.0], |a, s| {
                let p = point(*s);
                [a[0] + p[0], a[1] + p[1]]
            });
            vec![[sum[0] / n, sum[1] / n]]
        };
        if points.is_empty() {
            continue;
        }
        let (lo, hi) = colour::lit_bytes(*linear, corners, *at, *alpha, &points);
        pixel.push(PixelColour {
            x: c.x,
            y: c.y,
            fragment: c.fragment,
            lo,
            hi,
        });
    }
    (rows, pixel)
}

fn map(m: &FrameMap, q: [f64; 3]) -> [f64; 2] {
    m.map(q[0], q[1])
}

fn fragment(c: &draw::Common, layer: Layer, sub: u32, group: u32, eps: f64) -> Fragment {
    Fragment {
        id: 0,
        entry: c.entry,
        slot: c.record.slot,
        object: c.object,
        part: c.part,
        sub,
        layer: layer as u8,
        kind: c.record.kind.code(),
        flags: if eps.is_finite() { 0 } else { FLAG_UNBOUNDED },
        group,
        plane: NONE_U32,
        epsilon: eps,
        bbox: [0; 4],
        label: NONE_U32,
    }
}
