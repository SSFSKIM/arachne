//! One frame's decisions: every fragment the order draws and its verdict at every sample of the
//! window, or the frame's refusal.

use arachne_core::xf::{Delta, View};

use crate::camera::FrameMap;
use crate::colour::{self, Light};
use crate::cover::{self, PixelCover, SAMPLE_OFFSETS, Window};
use crate::draw::{self, Item, Layer};
use crate::flatten;
use crate::keys::{self, Model, Offered};
use crate::order::{Frag, Scene};
use crate::out::{
    COLOUR_GRADIENT, COLOUR_PER_PIXEL, COLOUR_SOLID, ColourRow, CoverRow, Decisions,
    FLAG_UNBOUNDED, FLAG_ZERO_ALPHA, Fragment, Header, NONE_U32, PixelColour, Refused,
    STATUS_DECIDED, STATUS_UNABLE,
};
use crate::pairs;
use crate::shift::Shift;
use crate::source::WireDelta;
use crate::stream::{Reason, Refusal, State};

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
    Gradient,
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
        return unable(state, xf, seq, window, &refusals);
    }
    let view = state.view.as_ref().expect("a frame not refused has a view");
    let header = header(Some(view), xf, seq, window);
    let window = header.window;
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
    for item in &items {
        match item {
            Item::Path(p) => {
                let c = &p.common;
                let eps = Shift::of(c, view, &p.map, &p.shown).of_points(&p.shown);
                let rings = flatten::layer_rings(p, size);
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
                        Some(r) if r.len() == 1 && !p.shaded => {
                            Spec::Solid(colour::premultiply(r[0], c.record.opacity))
                        }
                        _ => Spec::Gradient,
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
                    let worst = eps.iter().copied().fold(0.0, f64::max);
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
    // The registry with the records that paint nothing offered too, where there are any.
    let with_invisible = (!extra.is_empty()).then(|| {
        let all = offered.iter().cloned().chain(extra).collect();
        Scene::new(frags.clone(), all)
    });
    let scene = Scene::new(frags, offered);
    let (bands, pairs) = pairs::frame(
        &scene,
        with_invisible.as_ref().unwrap_or(&scene),
        &mut covers,
    );
    let (colours, pixel_colours) = colours(&specs, &covers);
    Decisions {
        header,
        fragments,
        covers,
        bands,
        pairs,
        colours,
        pixel_colours,
        labels: Vec::new(),
        refused: Vec::new(),
    }
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
                    let worst = eps.iter().copied().fold(0.0, f64::max);
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
        None => Spec::Gradient,
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
                Spec::Gradient => (COLOUR_GRADIENT, [0; 4]),
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
