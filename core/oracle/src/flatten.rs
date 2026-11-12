//! A path layer's outline, stroke and rings (contract §4 and §8), through the shared kurbo.
//!
//! The outline is made in frame units from the path's shown points; the fill is the outline
//! mapped to pixels and flattened at 0.1 px; a stroke is kurbo's `stroke` of the outline in frame
//! units at `0.1 / s`, edited at the joins the record carries, then mapped and flattened as the
//! fill is. kurbo is a contract term: what this module adds around it is written from the text.
//! The guard (`guard_chords@1`) and the dashes laid under it are [`crate::guard`]'s; the join
//! edits (`join_mask@1`) are [`crate::joins`]'s.

use arachne_core::xf::{Cap, Dash, Joint};
use kurbo::{BezPath, PathEl, Point, Shape};

use crate::camera::FrameMap;
use crate::draw::PathItem;
use crate::{guard, joins};

/// Cairo's default tolerance, in pixels.
pub const TOLERANCE: f64 = 0.1;
/// Cairo's miter limit.
pub const MITER_LIMIT: f64 = 10.0;
/// Cairo's least length: a dash whose gap is shorter strokes solid.
pub const SOLID_DASH_GAP: f64 = 1.0 / 512.0;

/// CE's tolerance for two coordinates being one: `|a − b| <= 1e-6 + 1e-5 |b|`.
pub fn ce_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 + 1e-5 * b.abs()
}

/// CE's subpaths of `shown` (cubic layout), as `[start, end)` point ranges of at least four
/// points: split where a cubic's end and the next cubic's start differ in `x` or `y`.
pub fn subpaths(shown: &[[f64; 3]]) -> Vec<(usize, usize)> {
    let n = shown.len() - shown.len() % 4;
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 4;
    while i < n {
        let (a, b) = (shown[i - 1], shown[i]);
        if !(ce_close(a[0], b[0]) && ce_close(a[1], b[1])) {
            out.push((start, i));
            start = i;
        }
        i += 4;
    }
    out.push((start, n));
    out.retain(|(s, e)| e - s >= 4);
    out
}

/// Whether the subpath `[s, e)` of `shown` closes: its first and last points are one by
/// [`ce_close`] in `x` and `y`.
pub fn closes(shown: &[[f64; 3]], s: usize, e: usize) -> bool {
    let (a, b) = (shown[s], shown[e - 1]);
    ce_close(a[0], b[0]) && ce_close(a[1], b[1])
}

fn pt(p: [f64; 3]) -> Point {
    Point::new(p[0], p[1])
}

/// The outline in frame units: per subpath a move to its first point and a curve per cubic; a
/// closed subpath ends exactly at its first point and closes.
pub fn outline(shown: &[[f64; 3]]) -> BezPath {
    let mut out = BezPath::new();
    for (s, e) in subpaths(shown) {
        let closed = closes(shown, s, e);
        out.move_to(pt(shown[s]));
        let mut i = s;
        while i + 4 <= e {
            let end = if closed && i + 4 == e {
                shown[s]
            } else {
                shown[i + 3]
            };
            out.curve_to(pt(shown[i + 1]), pt(shown[i + 2]), pt(end));
            i += 4;
        }
        if closed {
            out.close_path();
        }
    }
    out
}

/// The dash a record strokes with (`dash_start_least_length@1`): `None` where it strokes solid
/// (no dash, a duty of 1, or a gap under Cairo's least length), else its two entries.
pub fn dash_entries(dash: Option<Dash>) -> Option<[f64; 2]> {
    let d = dash?;
    let a = d.period * d.duty;
    let b = d.period * (1.0 - d.duty);
    // The gap is the total less the on length, never `b` itself; strictly below 1/512 is solid,
    // exactly 1/512 dashed. A duty of 1 makes `b` 0 and so the gap 0 (the period is finite, §11).
    if (a + b) - a < SOLID_DASH_GAP {
        return None;
    }
    Some([a, b])
}

/// Whether Cairo's dashed stroker starts a subpath in an `on` dash: the offset `phase · period`
/// taken modulo the entries' total, then stepped past every entry it reaches or passes.
pub fn dash_starts_on(d: &Dash, entries: [f64; 2]) -> bool {
    let total = entries[0] + entries[1];
    let mut o = libm::fmod(d.phase * d.period, total);
    if o < 0.0 {
        o += total;
    }
    // Cairo then sets an offset at most 0 to 0. Here that is only ±0 (`fmod` leaves `|o|` under
    // the total, so a raised offset is above 0), and a zero offset, −0 included, steps past no
    // entry: the walk below starts on either way.
    let mut on = true;
    let mut i = 0;
    while o > 0.0 && o >= entries[i] {
        o -= entries[i];
        on = !on;
        i = 1 - i;
    }
    on
}

fn kurbo_join(joint: Joint) -> kurbo::Join {
    match joint {
        Joint::Auto | Joint::Miter => kurbo::Join::Miter,
        Joint::Round => kurbo::Join::Round,
        Joint::Bevel => kurbo::Join::Bevel,
    }
}

fn kurbo_cap(cap: Cap) -> kurbo::Cap {
    match cap {
        Cap::Auto | Cap::Butt => kurbo::Cap::Butt,
        Cap::Round => kurbo::Cap::Round,
        Cap::Square => kurbo::Cap::Square,
    }
}

/// A path through the frame mapping, point by point.
pub fn mapped(path: &BezPath, map: &FrameMap) -> BezPath {
    let m = |p: Point| {
        let [x, y] = map.map(p.x, p.y);
        Point::new(x, y)
    };
    let els = path.elements().iter().map(|el| match *el {
        PathEl::MoveTo(p) => PathEl::MoveTo(m(p)),
        PathEl::LineTo(p) => PathEl::LineTo(m(p)),
        PathEl::QuadTo(a, b) => PathEl::QuadTo(m(a), m(b)),
        PathEl::CurveTo(a, b, c) => PathEl::CurveTo(m(a), m(b), m(c)),
        PathEl::ClosePath => PathEl::ClosePath,
    });
    BezPath::from_vec(els.collect())
}

/// A ring: a closed polygon of pixel points.
pub type Ring = Vec<[f64; 2]>;

/// `path` (in pixels) flattened by kurbo at [`TOLERANCE`] into rings: each run from a move to a
/// close, or to the next move; a ring of fewer than three points is dropped.
pub fn rings(path: &BezPath) -> Vec<Ring> {
    let mut rings = Vec::new();
    let mut ring: Ring = Vec::new();
    kurbo::flatten(path.elements().iter().copied(), TOLERANCE, |el| match el {
        PathEl::MoveTo(p) => {
            if ring.len() > 2 {
                rings.push(std::mem::take(&mut ring));
            }
            ring.clear();
            ring.push([p.x, p.y]);
        }
        PathEl::LineTo(p) => ring.push([p.x, p.y]),
        PathEl::ClosePath => {
            if ring.len() > 2 {
                rings.push(std::mem::take(&mut ring));
            }
            ring.clear();
        }
        PathEl::QuadTo(..) | PathEl::CurveTo(..) => {}
    });
    if ring.len() > 2 {
        rings.push(ring);
    }
    rings
}

/// Whether a face strokes as a dot or not at all (`round_cap_dot@1`): `Some(true)` for a dot,
/// `Some(false)` for nothing, `None` for kurbo's stroke.
fn dot_rule(item: &PathItem, entries: Option<[f64; 2]>) -> Option<bool> {
    let s = &item.shown;
    if !item.face || s.iter().any(|p| p[0] != s[0][0] || p[1] != s[0][1]) {
        return None;
    }
    let starts_on = match (item.dash, entries) {
        (Some(d), Some(e)) => dash_starts_on(&d, e),
        _ => true,
    };
    Some(item.cap == Cap::Round && starts_on)
}

/// Each layer's rings in pixels (background stroke, fill, stroke), `None` for a layer not drawn.
pub fn layer_rings(item: &PathItem, size: [u32; 2]) -> [Option<Vec<Ring>>; 3] {
    let map = &item.map;
    let user = outline(&item.shown);
    let widest = crate::bound::max(item.layers[0].unwrap_or(0.0), item.layers[2].unwrap_or(0.0));
    let area = guard::area(&user, map, size, widest);
    let guarded_outline = match area {
        Some(area) => guard::guarded(&user, area),
        None => user.clone(),
    };
    let scale = map.scale();
    let tolerance = TOLERANCE / scale;
    let entries = dash_entries(item.dash);
    // `join_mask@1` edits only a stroke under kurbo's miter, and only where the record carries
    // the field.
    let carried = match item.joint {
        Joint::Auto | Joint::Miter => item.joins.as_ref().map(|j| joins::carried(&item.shown, j)),
        Joint::Round | Joint::Bevel => None,
    };
    let dashes = match (item.dash, entries) {
        (Some(d), Some(pattern)) => Some(guard::dashed(&user, d.phase * d.period, &pattern, area)),
        _ => None,
    };
    let dot = dot_rule(item, entries);
    std::array::from_fn(|k| {
        let w = item.layers[k]?;
        if k == 1 {
            return Some(rings(&mapped(&guarded_outline, map)));
        }
        let stroked = match dot {
            Some(false) => return Some(Vec::new()),
            Some(true) => {
                let c = item.shown[0];
                kurbo::Circle::new(Point::new(c[0], c[1]), w / 2.0).to_path(tolerance)
            }
            None => {
                let style = kurbo::Stroke::new(w)
                    .with_join(kurbo_join(item.joint))
                    .with_caps(kurbo_cap(item.cap))
                    .with_miter_limit(MITER_LIMIT);
                let elements = dashes.as_ref().unwrap_or(&guarded_outline).elements();
                let stroked = kurbo::stroke(
                    elements.iter().copied(),
                    &style,
                    &kurbo::StrokeOpts::default(),
                    tolerance,
                );
                match &carried {
                    Some(carried) => joins::edit(stroked, elements, tolerance, w, carried),
                    None => stroked,
                }
            }
        };
        Some(rings(&mapped(&stroked, map)))
    })
}
