//! Which surfaces a camera-facing label goes behind: one decision per label and surface sheet,
//! made across the whole frame before any tile is resolved.
//!
//! A label is the records sharing one facing group (XF 1.5), read whole: a nearer surface covers
//! it only where the surface overlaps it, and never splits it where the surface crosses the
//! label's depth. Its depth is its carried centre's: the camera-facing plane its records are
//! keyed on, read at the centre's projection.
//!
//! A sheet is a connected part of one source surface (the outermost surface-like object its faces
//! are drawn from, [`label_class::surface_owner`]) whose faces all face one way on screen: faces
//! of that object sharing two corners on screen and wound the same way are one sheet. A front and
//! a back of one solid are two sheets, each decided on its own.
//!
//! The overlap is the order points both hold: every sub-sample point of the frame
//! ([`crate::composite::sample_point`], where the resolve orders fragments) inside a record of
//! the label (a fill or a stroke's outline) and inside a fill of the sheet painted in a colour
//! not wholly transparent, by the nonzero rule. A sliver of either between points is no overlap.
//! A sheet decides against the label ([`Outcome`]):
//!
//! - where a fill of the sheet covers the label's projected centre, by its depth on that ray
//!   against the label's, both read with their error bounds: shown nearer, it qualifies; shown
//!   farther or equal, it does not; else it is open ([`Open::Interval`]);
//! - elsewhere, by its depth at every point of the overlap, its fill's plane read there: shown
//!   nearer at every point, it qualifies; shown farther or equal at some point, it does not (a
//!   genuine straddle the points see, decided: the label stays in front); else it is open
//!   ([`Open::Interval`]). No point counted is no overlap.
//!
//! A sheet two of whose fills hold one point of the overlap (folded over itself on screen) is open
//! ([`Open::Folded`]), and so is each of two sheets of one object facing one way that both meet
//! the label ([`Open::Disconnected`]). A label whose records are not all paths placed by the
//! engine is open against everything ([`Open::Label`]), and so is a sheet drawn as a mesh
//! ([`Open::Mesh`]). A record kept facing the camera outside every label (its label not known) is
//! never a sheet: a label meeting it is open ([`Open::Label`]). An open pair never moves the
//! label: only sheets that qualify do.

use crate::composite::{Coverage, Fragment, Plane, Producer, SAMPLES};
use crate::label_class::{self, LabelClass};
use crate::order::Decided;
use crate::path::{LayerView, Paths};
use crate::project::{FrameMap, Projection};
use crate::scanline::Edge;
use crate::stream::is_overlay;
use crate::xf::{Camera, Record, RecordKind, View};
use std::collections::{BTreeMap, HashMap};

/// How a sheet stands against a label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Shown nearer: the label goes behind it.
    Qualifies,
    /// Shown farther or at the label's depth where it covers the centre, or somewhere in the
    /// overlap: the label stays in front of it.
    NotNearer,
    /// Undecided: the label stays in front of it, and the case is counted by its cause.
    Open(Open),
}

/// Why a pair is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Open {
    /// The error bounds cannot settle whether the sheet is nearer.
    Interval,
    /// Two fills of the sheet hold one point of the overlap.
    Folded,
    /// Another sheet of the same object, facing the same way, meets the label too.
    Disconnected,
    /// The sheet is drawn as a mesh, whose faces this planner does not read.
    Mesh,
    /// The label has a record that is not a path placed by the engine at a fixed orientation, or
    /// the sheet is a record kept facing the camera whose label is not known.
    Label,
}

impl Open {
    /// Its name, as the guard counts it.
    pub fn name(self) -> &'static str {
        match self {
            Open::Interval => "interval",
            Open::Folded => "folded",
            Open::Disconnected => "disconnected",
            Open::Mesh => "mesh",
            Open::Label => "label",
        }
    }
}

/// One label against one sheet it overlaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    /// The label's facing group.
    pub group: u32,
    /// The sheet's object, by its token.
    pub surface: u32,
    /// The sheet, by its index in the frame's sheets.
    pub sheet: u32,
    /// Whether a fill of the sheet covers the label's projected centre.
    pub centre: bool,
    pub outcome: Outcome,
}

/// What a fragment is to the labels' placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Ordered as it always is.
    Ordinary,
    /// Fixed in frame: in front of every label.
    Overlay,
    /// A record of the label at this index of [`LabelPlan::groups`].
    Label(u32),
}

/// The frame's labels and the sheets each goes behind.
#[derive(Clone, Debug, Default)]
pub struct LabelPlan {
    /// The labels, by facing group, in the order their first records are drawn.
    pub groups: Vec<u32>,
    /// Each order entry's label, as its index in `groups` plus one; `0` for none.
    label_of: Vec<u32>,
    overlay: Vec<bool>,
    /// The sheet each fill (order entry, sub-order) of a qualifying sheet belongs to.
    sheet_of: HashMap<(u32, u32), u32>,
    /// Per label, the sheets it goes behind, ascending.
    behind: Vec<Vec<u32>>,
    /// Every label against every sheet it overlaps, by label then sheet.
    pub pairs: Vec<Pair>,
}

impl LabelPlan {
    /// What fragment `f` is to the placement.
    pub fn role(&self, f: &Fragment) -> Role {
        let at = f.order as usize;
        match self.label_of.get(at) {
            Some(&g) if g != 0 => Role::Label(g - 1),
            _ if self.overlay.get(at).copied().unwrap_or(false) => Role::Overlay,
            _ => Role::Ordinary,
        }
    }

    /// Whether `f` is a fill of a sheet label `label` goes behind.
    pub fn backs(&self, label: u32, f: &Fragment) -> bool {
        is_fill(f)
            && self
                .sheet_of
                .get(&(f.order, f.sub))
                .is_some_and(|s| self.behind[label as usize].binary_search(s).is_ok())
    }

    /// The sheets label `label` (an index into `groups`) goes behind.
    pub fn behind(&self, label: u32) -> &[u32] {
        &self.behind[label as usize]
    }
}

/// Whether `f` is a path's fill: the layer a sheet qualifies by.
fn is_fill(f: &Fragment) -> bool {
    f.path_layer && f.sub % 3 == 1
}

/// One fill of a surface's face, as the planner reads it.
struct Face<'a> {
    owner: u32,
    /// The sign of its area on screen: which way it is wound, so which way it faces.
    facing: i8,
    corners: Vec<(i64, i64)>,
    layer: LayerView<'a>,
}

/// A corner on screen, on a grid fine enough that a corner two faces share falls in one place.
fn corner(x: f64, y: f64) -> (i64, i64) {
    const GRID: f64 = 4096.0;
    ((x * GRID).round() as i64, (y * GRID).round() as i64)
}

fn face_of(owner: u32, layer: LayerView<'_>) -> Face<'_> {
    let mut area = 0.0;
    let mut corners = Vec::with_capacity(layer.edges.len() * 2);
    for e in layer.edges {
        area += e.dir as f64 * (e.x0 + e.x1) * (e.y1 - e.y0) / 2.0;
        corners.push(corner(e.x0, e.y0));
        corners.push(corner(e.x1, e.y1));
    }
    corners.sort_unstable();
    corners.dedup();
    let facing = if area > 1e-9 {
        1
    } else if area < -1e-9 {
        -1
    } else {
        0
    };
    Face {
        owner,
        facing,
        corners,
        layer,
    }
}

/// Faces partitioned into sheets: faces of one owner wound the same way and sharing two corners
/// are one sheet; a face wound neither way is a sheet alone. The sheets are numbered in the order
/// of their first faces.
fn sheets(faces: &[Face<'_>]) -> Vec<u32> {
    let mut parent: Vec<usize> = (0..faces.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut at: HashMap<(u32, i8, (i64, i64)), Vec<usize>> = HashMap::new();
    for (i, f) in faces.iter().enumerate() {
        if f.facing == 0 {
            continue;
        }
        for c in &f.corners {
            at.entry((f.owner, f.facing, *c)).or_default().push(i);
        }
    }
    let mut shared: HashMap<(usize, usize), u32> = HashMap::new();
    for list in at.values() {
        for (p, &a) in list.iter().enumerate() {
            for &b in &list[p + 1..] {
                *shared.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
    }
    let mut pairs: Vec<(usize, usize)> = shared
        .into_iter()
        .filter(|(_, n)| *n >= 2)
        .map(|(p, _)| p)
        .collect();
    pairs.sort_unstable();
    for (a, b) in pairs {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    let mut number: HashMap<usize, u32> = HashMap::new();
    (0..faces.len())
        .map(|i| {
            let root = find(&mut parent, i);
            let next = number.len() as u32;
            *number.entry(root).or_insert(next)
        })
        .collect()
}

/// A label as the planner reads it.
struct Label {
    /// Its records' pixels, `[x0, y0, x1, y1]`, inside the frame; empty for none.
    bbox: [u32; 4],
    /// Its projected centre, in pixels, and its depth there with that depth's bound.
    centre: [f64; 2],
    depth: f64,
    err: f64,
    /// Every record is a path placed by the engine at its fixed orientation.
    readable: bool,
}

/// What one label has seen of one sheet.
#[derive(Default)]
struct Seen {
    overlap: bool,
    /// Over the overlap: some point shown not nearer, some point undecided.
    not_nearer: bool,
    undecided: bool,
    folded: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Nearer,
    NotNearer,
    Undecided,
}

/// A sheet's depth `d` (bound `e`) against the label's.
fn compare(d: f64, e: f64, label: &Label) -> Cmp {
    if d + e < label.depth - label.err {
        Cmp::Nearer
    } else if d - e >= label.depth + label.err {
        Cmp::NotNearer
    } else {
        Cmp::Undecided
    }
}

/// Whether `r` is kept facing the camera outside every label. In a frame drawing a facing group
/// (a stream read under XF 1.5's rules), such a record's label is not known (XF.md's "Facing
/// groups"): it is never a sheet, and a label meeting it is open.
fn unknown_label(r: &Record) -> bool {
    r.facing_group == 0 && r.depth.fixed_orientation_centre.is_some() && !is_overlay(r)
}

/// Whether point `p` lies on the segment from `a` to `b`.
fn on_segment(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> bool {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let length = dx * dx + dy * dy;
    let t = if length > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (qx, qy) = (a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
    qx * qx + qy * qy <= 1e-18
}

/// Whether the polygons whose outline is `edges` and `flats` (its horizontal segments) hold point
/// `p` (nonzero), a point on the outline held: a centre on the edge two faces share, or on a
/// sheet's border, lies on the sheet.
fn holds(edges: &[Edge], flats: &[[f64; 3]], p: [f64; 2]) -> bool {
    if flats
        .iter()
        .any(|&[y, x0, x1]| on_segment([x0, y], [x1, y], p))
    {
        return true;
    }
    let mut winding = 0;
    for e in edges {
        if on_segment([e.x0, e.y0], [e.x1, e.y1], p) {
            return true;
        }
        if e.y0 <= p[1] && p[1] < e.y1 && e.x_at(p[1]) < p[0] {
            winding += e.dir;
        }
    }
    winding != 0
}

/// The plan for the frame `view` draws at `width x height` from the drawn order entries
/// (`entries`, by order index, each with its record), whose producers are `producers` (`paths`
/// among them): `None` where no label is drawn, and the frame takes the order it always did.
// The sub-sample `k` indexes each fragment's coverage row and the sums alike.
#[allow(clippy::needless_range_loop)]
pub fn plan(
    view: &View,
    width: u32,
    height: u32,
    entries: &[Option<&Record>],
    paths: &Paths,
    producers: &[&dyn Producer],
) -> Option<LabelPlan> {
    let mut groups: Vec<u32> = Vec::new();
    let mut index: BTreeMap<u32, u32> = BTreeMap::new();
    let mut label_of = vec![0u32; entries.len()];
    let mut overlay = vec![false; entries.len()];
    for (at, record) in entries.iter().enumerate() {
        let Some(record) = record else { continue };
        if is_overlay(record) {
            overlay[at] = true;
        } else if record.facing_group != 0 {
            let next = groups.len() as u32;
            let g = *index.entry(record.facing_group).or_insert(next);
            if g == next {
                groups.push(record.facing_group);
            }
            label_of[at] = g + 1;
        }
    }
    if groups.is_empty() {
        return None;
    }
    let Camera::ThreeD(camera) = &view.camera else {
        return None;
    };
    let projection = Projection::new(camera, view.frame_center);
    let map = FrameMap::of_kind(view, RecordKind::Path, width, height);
    let empty = [u32::MAX, u32::MAX, 0, 0];
    let mut labels: Vec<Label> = Vec::new();
    for g in 0..groups.len() as u32 {
        let first = entries
            .iter()
            .zip(&label_of)
            .find(|(_, l)| **l == g + 1)
            .and_then(|(r, _)| *r)
            .expect("a label has a record");
        let centre = first.depth.fixed_orientation_centre;
        let readable = entries.iter().zip(&label_of).all(|(r, l)| {
            *l != g + 1
                || r.is_some_and(|r| {
                    r.kind == RecordKind::Path
                        && r.pre_projected.is_none()
                        && r.depth.fixed_orientation_centre == centre
                })
        }) && centre.is_some();
        let (shown, plane) = match centre {
            Some(c) => {
                let p = projection.project(c);
                (map.apply(p), projection.facing_plane(c, &map))
            }
            None => ([0.0; 2], Plane::constant(0.0)),
        };
        labels.push(Label {
            bbox: empty,
            centre: shown,
            depth: plane.at(shown[0], shown[1]),
            err: plane.err + plane.read_error(shown[0], shown[1]),
            readable,
        });
    }
    // The faces: every fill of a record read as a surface, by its order entry and sub-order.
    let mut faces: Vec<Face<'_>> = Vec::new();
    let mut face_at: HashMap<(u32, u32), u32> = HashMap::new();
    let mut surface: Vec<Option<u32>> = vec![None; entries.len()];
    for (at, record) in entries.iter().enumerate() {
        let Some(record) = record else { continue };
        if label_class::label_class(&record.source, record.facing_group) == LabelClass::Surface
            && !unknown_label(record)
        {
            surface[at] = label_class::surface_owner(&record.source);
        }
    }
    for layer in paths.layers() {
        let (order, sub, bbox) = (layer.order, layer.sub, layer.bbox);
        let at = order as usize;
        if let Some(&g) = label_of.get(at)
            && g != 0
        {
            let b = &mut labels[g as usize - 1].bbox;
            *b = [
                b[0].min(bbox[0]),
                b[1].min(bbox[1]),
                b[2].max(bbox[2]),
                b[3].max(bbox[3]),
            ];
        }
        if sub % 3 == 1
            && let Some(Some(owner)) = surface.get(at)
            && entries[at].is_some_and(|r| r.pre_projected.is_none())
        {
            face_at.insert((order, sub), faces.len() as u32);
            faces.push(face_of(*owner, layer));
        }
    }
    let sheet = sheets(&faces);
    let n_sheets = sheet.iter().map(|s| s + 1).max().unwrap_or(0) as usize;
    // Meshes read as surfaces, and records kept facing the camera whose label is not known, are
    // open sheets of their own, numbered after the paths' sheets: by order entry, each with its
    // number, its cause and its object.
    let mut open_sheet: HashMap<u32, (u32, Open, u32)> = HashMap::new();
    for (at, record) in entries.iter().enumerate() {
        let Some(r) = record else { continue };
        let open = if unknown_label(r) {
            let owner = label_class::surface_owner(&r.source)
                .or_else(|| r.source.first().and_then(|p| p.first()).map(|n| n.token));
            Some((Open::Label, owner.unwrap_or(0)))
        } else if r.kind == RecordKind::Mesh {
            surface[at].map(|owner| (Open::Mesh, owner))
        } else {
            None
        };
        if let Some((cause, owner)) = open {
            let next = (n_sheets + open_sheet.len()) as u32;
            open_sheet.insert(at as u32, (next, cause, owner));
        }
    }
    let open_of = |s: u32| open_sheet.values().find(|(x, _, _)| *x == s);
    let owner_of = |s: u32| -> u32 {
        if (s as usize) < n_sheets {
            let f = sheet
                .iter()
                .position(|x| *x == s)
                .expect("a sheet has a face");
            faces[f].owner
        } else {
            open_of(s).expect("an open sheet").2
        }
    };
    // What each label sees of each sheet, pixel by pixel of its box.
    let mut seen: BTreeMap<(u32, u32), Seen> = BTreeMap::new();
    let boxes: Vec<(u32, [u32; 4])> = labels
        .iter()
        .enumerate()
        .filter(|(_, l)| l.bbox[0] < l.bbox[2] && l.bbox[1] < l.bbox[3])
        .map(|(g, l)| (g as u32, l.bbox))
        .collect();
    for rect in crate::tiles::tile_rects(width, height) {
        let meets =
            |b: &[u32; 4]| b[0] < rect.x1 && rect.x0 < b[2] && b[1] < rect.y1 && rect.y0 < b[3];
        if !boxes.iter().any(|(_, b)| meets(b)) {
            continue;
        }
        // The order points a fragment holds: a mask's, or a path's point-in-outline test.
        let points = |f: &Fragment, x: u32, y: u32| match f.coverage {
            Coverage::Mask(m) => m,
            Coverage::Area { area, .. } if area >= 1.0 => u16::MAX,
            Coverage::Area { id, .. } => producers[f.source() as usize].mask(id, x, y),
        };
        crate::tiles::passes(rect, producers).visit_pixels(rect, |x, y, frags, _, depths| {
            for (g, b) in &boxes {
                if x < b[0] || x >= b[2] || y < b[1] || y >= b[3] {
                    continue;
                }
                let label = &labels[*g as usize];
                // The order points the label's records hold.
                let mut mask = 0u16;
                for f in frags {
                    if label_of.get(f.order as usize) == Some(&(g + 1)) {
                        mask |= points(f, x, y);
                    }
                }
                if mask == 0 {
                    continue;
                }
                // Per sheet met here, how many of its fills hold each order point.
                let mut held: BTreeMap<u32, [u32; SAMPLES]> = BTreeMap::new();
                for (i, f) in frags.iter().enumerate() {
                    let s = if let Some(&(s, _, _)) = open_sheet.get(&f.order) {
                        s
                    } else if is_fill(f)
                        && let Some(&face) = face_at.get(&(f.order, f.sub))
                    {
                        sheet[face as usize]
                    } else {
                        continue;
                    };
                    if f.colour[3] == 0 {
                        continue;
                    }
                    let entry = seen.entry((*g, s)).or_default();
                    let count = held.entry(s).or_insert([0; SAMPLES]);
                    let both = points(f, x, y) & mask;
                    for k in 0..SAMPLES {
                        if both & (1 << k) == 0 {
                            continue;
                        }
                        entry.overlap = true;
                        count[k] += 1;
                        match compare(depths[i].samples[k], depths[i].err, label) {
                            Cmp::Nearer => {}
                            Cmp::NotNearer => entry.not_nearer = true,
                            Cmp::Undecided => entry.undecided = true,
                        }
                    }
                }
                for (s, count) in held {
                    if count.iter().any(|c| *c > 1) {
                        seen.get_mut(&(*g, s)).expect("seen above").folded = true;
                    }
                }
            }
        });
    }
    // Where a sheet's fill holds the label's projected centre, its depth on that ray: the plane
    // the fill is keyed on, read there. Fills holding it on two planes that place it apart
    // leave it undecided.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_sheets];
    for (f, s) in sheet.iter().enumerate() {
        members[*s as usize].push(f);
    }
    let centre_of = |g: u32, s: u32| -> Option<Cmp> {
        let label = &labels[g as usize];
        let [cx, cy] = label.centre;
        let mut found = None;
        for &f in members.get(s as usize)? {
            let layer = &faces[f].layer;
            let b = layer.bbox;
            let near = |v: f64, lo: u32, hi: u32| v >= lo as f64 - 1.0 && v <= hi as f64 + 1.0;
            if layer.clear || !near(cx, b[0], b[2]) || !near(cy, b[1], b[3]) {
                continue;
            }
            if !holds(layer.edges, layer.flats, label.centre) {
                continue;
            }
            let c = match layer.plane {
                Some(p) => compare(p.at(cx, cy), p.err + p.read_error(cx, cy), label),
                None => Cmp::Undecided,
            };
            found = Some(match found {
                None => c,
                Some(old) if old == c => c,
                Some(_) => Cmp::Undecided,
            });
        }
        found
    };
    // Two sheets of one object facing one way, both meeting a label: both open.
    let facing_of = |s: u32| -> i8 {
        if (s as usize) < n_sheets {
            let f = sheet
                .iter()
                .position(|x| *x == s)
                .expect("a sheet has a face");
            faces[f].facing
        } else {
            0
        }
    };
    let mut met: HashMap<(u32, u32, i8), u32> = HashMap::new();
    for ((g, s), v) in &seen {
        if v.overlap {
            *met.entry((*g, owner_of(*s), facing_of(*s))).or_default() += 1;
        }
    }
    let mut pairs = Vec::new();
    let mut behind = vec![Vec::new(); groups.len()];
    for ((g, s), v) in &seen {
        if !v.overlap {
            continue;
        }
        let owner = owner_of(*s);
        let facing = facing_of(*s);
        let centre = centre_of(*g, *s);
        let outcome = if !labels[*g as usize].readable {
            Outcome::Open(Open::Label)
        } else if let Some(&(_, cause, _)) = open_of(*s) {
            Outcome::Open(cause)
        } else if v.folded {
            Outcome::Open(Open::Folded)
        } else if facing != 0 && met[&(*g, owner, facing)] > 1 {
            Outcome::Open(Open::Disconnected)
        } else {
            match centre {
                Some(Cmp::Nearer) => Outcome::Qualifies,
                Some(Cmp::NotNearer) => Outcome::NotNearer,
                Some(Cmp::Undecided) => Outcome::Open(Open::Interval),
                None if v.not_nearer => Outcome::NotNearer,
                None if v.undecided => Outcome::Open(Open::Interval),
                None => Outcome::Qualifies,
            }
        };
        if outcome == Outcome::Qualifies {
            behind[*g as usize].push(*s);
        }
        pairs.push(Pair {
            group: groups[*g as usize],
            surface: owner,
            sheet: *s,
            centre: centre.is_some(),
            outcome,
        });
    }
    let mut sheet_of = HashMap::new();
    let backing: std::collections::BTreeSet<u32> = behind.iter().flatten().copied().collect();
    for (key, face) in &face_at {
        let s = sheet[*face as usize];
        if backing.contains(&s) {
            sheet_of.insert(*key, s);
        }
    }
    Some(LabelPlan {
        groups,
        label_of,
        overlay,
        sheet_of,
        behind,
        pairs,
    })
}

/// `items`, the fragments covering one sub-sample, front first into `out`, with the labels
/// placed: the others ranked by `rank` (as the sample resolve ranks them, into its second
/// argument), then each label inserted immediately behind the rearmost of them that is a fill of a
/// sheet it goes behind, or, with none, behind the overlay's and in front of everything else.
/// Labels at one place keep paint order (`painted`, `Less` for the first in front).
pub(crate) fn rank_placed(
    plan: &LabelPlan,
    frags: &[Fragment],
    items: &[usize],
    rank: impl FnOnce(&[usize], &mut Vec<usize>),
    painted: impl Fn(usize, usize) -> std::cmp::Ordering,
    out: &mut Vec<usize>,
) {
    let (labels, others): (Vec<usize>, Vec<usize>) = items
        .iter()
        .partition(|i| matches!(plan.role(&frags[**i]), Role::Label(_)));
    let mut ranked = Vec::with_capacity(others.len());
    rank(&others, &mut ranked);
    let behind = |found: Option<usize>| found.map_or(0, |p| p + 1);
    let overlay = behind(
        ranked
            .iter()
            .rposition(|i| plan.role(&frags[*i]) == Role::Overlay),
    );
    // Each label with the place in `ranked` it goes before.
    let mut placed: Vec<(usize, usize)> = labels
        .iter()
        .map(|&l| {
            let Role::Label(g) = plan.role(&frags[l]) else {
                unreachable!("partitioned as a label")
            };
            let place = match ranked.iter().rposition(|i| plan.backs(g, &frags[*i])) {
                Some(p) => p + 1,
                None => overlay,
            };
            (place, l)
        })
        .collect();
    placed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| painted(a.1, b.1)));
    out.clear();
    let mut next = placed.iter().peekable();
    for (p, &i) in ranked.iter().enumerate() {
        while let Some(&&(place, l)) = next.peek()
            && place == p
        {
            out.push(l);
            next.next();
        }
        out.push(i);
    }
    out.extend(next.map(|(_, l)| *l));
}

/// The pixel's order of `a` against `b` where either is a label: two labels in paint order
/// (`painted`), the overlay in front of a label, and a label behind a fill of a sheet it goes
/// behind and in front of everything else, the placement deciding ([`Decided::Placed`]).
/// `None` where neither is a label. A pixel where a sheet a label goes behind covers a
/// sub-sample composites sample by sample ([`rank_placed`]); there this order breaks ties alone.
pub(crate) fn placed_pair(
    plan: &LabelPlan,
    a: &Fragment,
    b: &Fragment,
    painted: std::cmp::Ordering,
) -> Option<(std::cmp::Ordering, Decided)> {
    use std::cmp::Ordering::{Greater, Less};
    let placed = |front: bool| Some((if front { Less } else { Greater }, Decided::Placed));
    match (plan.role(a), plan.role(b)) {
        (Role::Label(_), Role::Label(_)) => Some((painted, Decided::Paint)),
        (Role::Label(_), Role::Overlay) => placed(false),
        (Role::Overlay, Role::Label(_)) => placed(true),
        (Role::Label(g), Role::Ordinary) => placed(!plan.backs(g, b)),
        (Role::Ordinary, Role::Label(g)) => placed(plan.backs(g, a)),
        _ => None,
    }
}

/// Whether a label among `frags` meets a fill of a sheet it goes behind: the pixel composites
/// sample by sample ([`rank_placed`]).
pub(crate) fn backed(plan: &LabelPlan, frags: &[Fragment]) -> bool {
    frags.iter().any(|l| match plan.role(l) {
        Role::Label(g) => frags.iter().any(|f| plan.backs(g, f)),
        _ => false,
    })
}

/// Whether `frags` hold a label.
pub(crate) fn holds_label(plan: &LabelPlan, frags: &[Fragment]) -> bool {
    frags.iter().any(|f| matches!(plan.role(f), Role::Label(_)))
}
