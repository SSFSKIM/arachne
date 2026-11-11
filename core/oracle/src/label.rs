//! `label_placement@2` (contract §10): which fragments are camera-facing labels, the sheets a
//! label may go behind, each pair's qualification decided once over the frame, and at every
//! sample a label shares with another fragment, that fragment's role in the label's placement
//! (`LABL`, `FORMAT.md`). The order itself is read from those roles and `PAIR` by whoever judges a
//! report; the oracle states the facts.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use arachne_core::xf::{Camera, Record, RecordKind, View};

use crate::flatten::Ring;
use crate::keys::{Band, Model, inside};
use crate::label_classes::{self, Cause, Outcome};
use crate::order::Frag;
use crate::out::{
    CoverRow, Fragment, LabelRow, NONE_U32, ROLE_ALPHA_STRADDLE, ROLE_ANCHOR, ROLE_FRONT,
    ROLE_KIND_UNKNOWN, ROLE_OPEN,
};
use crate::stream::State;

const FILL: u8 = 1;
const MESH: u8 = 3;

/// A pair's open cause (`LABL`'s `pair_cause`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairCause {
    Interval = 1,
    Folded = 2,
    Disconnected = 3,
    Mesh = 4,
    Label = 5,
}

fn class_code(c: Cause) -> u8 {
    match c {
        Cause::Unmapped => 1,
        Cause::GenericOnly => 2,
        Cause::PathsDisagree => 3,
        Cause::NoSource => 4,
        Cause::CylinderGrouped => 5,
        Cause::PointCloud => 6,
    }
}

/// What the label rule needs of each fragment of the frame, by fragment id.
pub struct Input<'a> {
    pub view: &'a View,
    pub state: &'a State,
    pub fragments: &'a [Fragment],
    /// The depth of each fragment that has rows (a prefix of `fragments`).
    pub frags: &'a [Frag],
    /// Sorted by `(y, x, fragment)`.
    pub covers: &'a [CoverRow],
    /// A path layer's flattened outline on screen, in frame pixels.
    pub outlines: &'a [Option<Vec<Ring>>],
    /// A fill's premultiplied alpha byte where one colour paints it everywhere; `None` where it
    /// varies with the pixel and the oracle does not compute it (a gradient).
    pub alpha: &'a [Option<u8>],
    /// A fragment's record's projected centre, in frame pixels, for a path kept facing the
    /// camera.
    pub centre: &'a [Option<[f64; 2]>],
    /// Whether a path fragment is a face of a face group.
    pub face: &'a [bool],
}

/// What a record is to the label rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// Never qualifies: a line, a stroke, a point block.
    Front,
    /// A surface: its fills form sheets, its mesh triangles a mesh sheet.
    Surface,
    /// A stream read as XF 1.4: a face or a mesh triangle, read both ways.
    KindUnknown,
    /// `label_classes@2` leaves it open.
    Open(Cause),
    /// A record kept facing the camera with no facing group, in a frame that draws one.
    Ungrouped,
}

/// A sheet: fills (or a mesh's triangles) of one surface object.
struct Sheet {
    object: (u8, u64),
    winding: i8,
    mesh: bool,
    kind_unknown: bool,
    members: Vec<u32>,
}

/// A counted point: its pixel, its sample, and the fill of the sheet that holds it.
type Point = (u32, u32, usize, u32);

/// A pair's decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    Qualifies,
    Not,
    Open(PairCause),
}

fn camera_facing(view: &View, rec: &Record) -> bool {
    matches!(view.camera, Camera::ThreeD(_)) && rec.depth.fixed_orientation_centre.is_some()
}

fn signed_area(rings: &[Ring]) -> f64 {
    let mut a = 0.0;
    for r in rings {
        for i in 0..r.len() {
            let (p, q) = (r[i], r[(i + 1) % r.len()]);
            a += p[0] * q[1] - q[0] * p[1];
        }
    }
    a / 2.0
}

fn winding(rings: &[Ring]) -> i8 {
    let a = signed_area(rings);
    if a.abs() <= 1e-9 {
        0
    } else if a > 0.0 {
        1
    } else {
        -1
    }
}

fn corners(rings: &[Ring]) -> BTreeSet<(i64, i64)> {
    rings
        .iter()
        .flatten()
        .map(|v| {
            (
                libm::round(v[0] * 4096.0) as i64,
                libm::round(v[1] * 4096.0) as i64,
            )
        })
        .collect()
}

/// Whether `p` lies on a segment of the rings, horizontal ones included.
fn on_outline(rings: &[Ring], p: [f64; 2]) -> bool {
    rings.iter().any(|r| {
        (0..r.len()).any(|i| {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
            cross == 0.0
                && p[0] >= a[0].min(b[0])
                && p[0] <= a[0].max(b[0])
                && p[1] >= a[1].min(b[1])
                && p[1] <= a[1].max(b[1])
        })
    })
}

fn find(parent: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while parent[r] != r {
        r = parent[r];
    }
    let mut j = i;
    while parent[j] != r {
        let n = parent[j];
        parent[j] = r;
        j = n;
    }
    r
}

/// How a band stands against the label's: wholly nearer, wholly farther or at it, or meeting it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stand {
    Nearer,
    NotNearer,
    Meets,
}

fn stand(s: &Band, label: &Band) -> Stand {
    if s.hi() < label.lo() {
        Stand::Nearer
    } else if s.lo() > label.hi()
        || (s.lo() == s.hi() && label.lo() == label.hi() && s.o == label.o)
    {
        Stand::NotNearer
    } else {
        Stand::Meets
    }
}

/// One plane's read at a point: `None` where the fill has no one plane (its key is a curve's) or
/// its read is not bounded.
fn plane_read(model: &Model, p: [f64; 2]) -> Option<Band> {
    match model {
        Model::Curve { .. } | Model::Unbounded => None,
        m => m.at(p).ok(),
    }
}

/// The frame's labels: each fragment's label index (`NONE_U32` for none) and the `LABL` rows.
pub fn place(input: &Input) -> (Vec<u32>, Vec<LabelRow>) {
    let (label_of, rows, _) = place_with_pairs(input);
    (label_of, rows)
}

/// A pair's decision, as counts compare it: `qualifies`, `not_nearer`, or an open cause's name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairDecision {
    pub label: u32,
    /// The sheet's index in the frame's sheets.
    pub sheet: u32,
    pub outcome: &'static str,
}

/// [`place`], and every pair's decision.
pub fn place_with_pairs(input: &Input) -> (Vec<u32>, Vec<LabelRow>, Vec<PairDecision>) {
    let st = input.state;
    let view = input.view;
    let n = input.fragments.len();
    let mut label_of = vec![NONE_U32; n];
    if !matches!(view.camera, Camera::ThreeD(_)) {
        return (label_of, Vec::new(), Vec::new());
    }
    let group_of = |slot: u32| st.meta.get(&slot).map_or(0, |m| m.facing_group);
    // A frame read under XF 1.5's rules that draws a facing group; otherwise read as XF 1.4.
    let rules15 = st.rules15 && st.order.iter().any(|e| group_of(e.slot) != 0);
    // The labels, by slot.
    let mut label_slots: BTreeMap<u32, u32> = BTreeMap::new();
    let mut members: Vec<Vec<u32>> = Vec::new();
    if rules15 {
        let mut by_group: BTreeMap<u32, u32> = BTreeMap::new();
        for e in &st.order {
            let g = group_of(e.slot);
            if g == 0 || label_slots.contains_key(&e.slot) {
                continue;
            }
            let next = by_group.len() as u32;
            let id = *by_group.entry(g).or_insert(next);
            if id as usize == members.len() {
                members.push(Vec::new());
            }
            members[id as usize].push(e.slot);
            label_slots.insert(e.slot, id);
        }
    } else {
        let mut last: Option<[u64; 3]> = None;
        for e in &st.order {
            let rec = &st.records[&e.slot];
            let centre = camera_facing(view, rec).then(|| {
                rec.depth
                    .fixed_orientation_centre
                    .unwrap()
                    .map(f64::to_bits)
            });
            match centre {
                Some(c) if last == Some(c) => {
                    let id = members.len() as u32 - 1;
                    if let std::collections::btree_map::Entry::Vacant(v) = label_slots.entry(e.slot)
                    {
                        v.insert(id);
                        members[id as usize].push(e.slot);
                    }
                }
                Some(c) => {
                    if let std::collections::btree_map::Entry::Vacant(v) = label_slots.entry(e.slot)
                    {
                        v.insert(members.len() as u32);
                        members.push(vec![e.slot]);
                    }
                    last = Some(c);
                }
                None => last = None,
            }
        }
    }
    if members.is_empty() {
        return (label_of, Vec::new(), Vec::new());
    }
    for (i, f) in input.fragments.iter().enumerate() {
        if let Some(l) = label_slots.get(&f.slot) {
            label_of[i] = *l;
        }
    }
    // A label drawn otherwise has a record that is not a path placed at a fixed orientation.
    let otherwise: Vec<bool> = members
        .iter()
        .map(|slots| {
            slots.iter().any(|s| {
                let r = &st.records[s];
                !(r.kind == RecordKind::Path && r.pre_projected.is_none() && camera_facing(view, r))
            })
        })
        .collect();
    // Each label's depth: its centre's key on its facing plane, read at the centre.
    let mut depth: Vec<Option<([f64; 2], Band)>> = vec![None; members.len()];
    for (i, f) in input.fragments.iter().enumerate() {
        let l = label_of[i];
        if l == NONE_U32 || depth[l as usize].is_some() || i >= input.frags.len() {
            continue;
        }
        if let Some(c) = input.centre[i]
            && let Ok(b) = input.frags[i].model.at(c)
        {
            let _ = f;
            depth[l as usize] = Some((c, b));
        }
    }
    // Each record's class.
    let class_of = |slot: u32, f: &Fragment, face: bool| -> Class {
        let rec = &st.records[&slot];
        if rules15 {
            if camera_facing(view, rec) && group_of(slot) == 0 {
                return Class::Ungrouped;
            }
            let meta = st.meta.get(&slot);
            let paths: Vec<Vec<&str>> = meta.map_or_else(Vec::new, |m| {
                m.source
                    .iter()
                    .map(|p| p.iter().map(|(_, c)| c.as_str()).collect())
                    .collect()
            });
            match label_classes::record_outcome(paths) {
                Outcome::Open(c) => Class::Open(c),
                Outcome::Line => Class::Front,
                Outcome::Surface if f.layer == FILL || f.layer == MESH => Class::Surface,
                Outcome::Surface => Class::Front,
            }
        } else if f.layer == MESH || (f.layer == FILL && face) {
            Class::KindUnknown
        } else if f.layer == FILL {
            Class::Surface
        } else {
            Class::Front
        }
    };
    // The surface object of a record: the outermost node on its first path whose outcome there is
    // a surface, else the path's first node; in a stream read as XF 1.4, the record.
    let object_of = |slot: u32| -> (u8, u64) {
        if !rules15 {
            return (0, slot as u64);
        }
        let Some(path) = st.meta.get(&slot).and_then(|m| m.source.first()) else {
            return (0, slot as u64);
        };
        let names: Vec<&str> = path.iter().map(|(_, c)| c.as_str()).collect();
        let at = (0..names.len())
            .rev()
            .find(|&i| label_classes::path_outcome(&names[i..]) == Outcome::Surface)
            .unwrap_or(0);
        (1, path[at].0 as u64)
    };
    let mut class = vec![Class::Front; n];
    for (i, f) in input.fragments.iter().enumerate() {
        if label_of[i] == NONE_U32 {
            class[i] = class_of(f.slot, f, input.face[i]);
        }
    }
    // Sheets: fills joined by winding and two shared corners; a mesh object's triangles as one.
    let mut sheets: Vec<Sheet> = Vec::new();
    let mut sheet_of = vec![usize::MAX; n];
    let mut by_object: BTreeMap<((u8, u64), bool), Vec<u32>> = BTreeMap::new();
    for (i, f) in input.fragments.iter().enumerate() {
        if matches!(class[i], Class::Surface | Class::KindUnknown) && i < input.frags.len() {
            by_object
                .entry((object_of(f.slot), f.layer == MESH))
                .or_default()
                .push(i as u32);
        }
    }
    for ((object, mesh), ids) in by_object {
        let kind_unknown = class[ids[0] as usize] == Class::KindUnknown;
        if mesh {
            for &i in &ids {
                sheet_of[i as usize] = sheets.len();
            }
            sheets.push(Sheet {
                object,
                winding: 0,
                mesh,
                kind_unknown,
                members: ids,
            });
            continue;
        }
        let empty = Vec::new();
        let rings = |i: u32| input.outlines[i as usize].as_ref().unwrap_or(&empty);
        let wind: Vec<i8> = ids.iter().map(|&i| winding(rings(i))).collect();
        let cs: Vec<BTreeSet<(i64, i64)>> = ids.iter().map(|&i| corners(rings(i))).collect();
        let mut parent: Vec<usize> = (0..ids.len()).collect();
        for a in 0..ids.len() {
            for b in a + 1..ids.len() {
                if wind[a] != 0 && wind[a] == wind[b] && cs[a].intersection(&cs[b]).count() >= 2 {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    parent[ra] = rb;
                }
            }
        }
        let mut class_sheet: HashMap<usize, usize> = HashMap::new();
        for k in 0..ids.len() {
            let root = find(&mut parent, k);
            let s = *class_sheet.entry(root).or_insert_with(|| {
                sheets.push(Sheet {
                    object,
                    winding: wind[k],
                    mesh: false,
                    kind_unknown,
                    members: Vec::new(),
                });
                sheets.len() - 1
            });
            sheets[s].members.push(ids[k]);
            sheet_of[ids[k] as usize] = s;
        }
    }
    let holds = |i: usize, p: [f64; 2], cover: u8| -> bool {
        if input.fragments[i].layer == MESH {
            return cover != 0;
        }
        input.alpha[i] != Some(0) && input.outlines[i].as_ref().is_some_and(|r| inside(r, p))
    };
    // Pixel by pixel: each label's counted points with each sheet, folded points, straddles.
    let mut counted: BTreeMap<(u32, usize), Vec<Point>> = BTreeMap::new();
    let mut folded: BTreeSet<(u32, usize)> = BTreeSet::new();
    let mut straddle: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
    let mut pixels: Vec<(usize, usize)> = Vec::new();
    let mut i0 = 0;
    while i0 < input.covers.len() {
        let mut i1 = i0;
        let (x, y) = (input.covers[i0].x, input.covers[i0].y);
        while i1 < input.covers.len() && (input.covers[i1].x, input.covers[i1].y) == (x, y) {
            i1 += 1;
        }
        pixels.push((i0, i1));
        let rows = &input.covers[i0..i1];
        let labels: BTreeSet<u32> = rows
            .iter()
            .map(|r| label_of[r.fragment as usize])
            .filter(|l| *l != NONE_U32)
            .collect();
        if !labels.is_empty() {
            for r in rows {
                let f = r.fragment as usize;
                if sheet_of[f] != usize::MAX
                    && input.fragments[f].layer == FILL
                    && input.alpha[f].is_none()
                {
                    straddle.insert((x, y, r.fragment));
                }
            }
            for s in 0..16 {
                let p = crate::cover::sample_point(x as i64, y as i64, s);
                for &l in &labels {
                    let in_label = rows.iter().any(|r| {
                        let f = r.fragment as usize;
                        label_of[f] == l
                            && r.cover[s] != 0
                            && (otherwise[l as usize]
                                || input.outlines[f].as_ref().is_some_and(|o| inside(o, p)))
                    });
                    if !in_label {
                        continue;
                    }
                    let mut by_sheet: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
                    for r in rows {
                        let f = r.fragment as usize;
                        if sheet_of[f] != usize::MAX && holds(f, p, r.cover[s]) {
                            by_sheet.entry(sheet_of[f]).or_default().push(r.fragment);
                        }
                    }
                    for (sh, fs) in by_sheet {
                        if fs.len() >= 2 && !sheets[sh].mesh {
                            folded.insert((l, sh));
                        }
                        counted.entry((l, sh)).or_default().push((x, y, s, fs[0]));
                    }
                }
            }
        }
        i0 = i1;
    }
    // Each pair's decision, in the contract's order.
    let mut decision: BTreeMap<(u32, usize), Decision> = BTreeMap::new();
    for (&(l, sh), points) in &counted {
        let sheet = &sheets[sh];
        let d = if otherwise[l as usize] {
            Decision::Open(PairCause::Label)
        } else if sheet.mesh {
            // A sheet drawn as a mesh, whose faces are not read (a kind-unknown mesh read as a
            // surface too).
            Decision::Open(PairCause::Mesh)
        } else if folded.contains(&(l, sh)) {
            Decision::Open(PairCause::Folded)
        } else if rules15
            && sheet.winding != 0
            && counted.keys().any(|&(l2, sh2)| {
                l2 == l
                    && sh2 != sh
                    && sheets[sh2].object == sheet.object
                    && sheets[sh2].winding == sheet.winding
                    && !sheets[sh2].mesh
            })
        {
            Decision::Open(PairCause::Disconnected)
        } else {
            match depth[l as usize] {
                None => Decision::Open(PairCause::Interval),
                Some((c, lb)) => {
                    by_centre(input, sheet, c, &lb).unwrap_or_else(|| by_points(input, points, &lb))
                }
            }
        };
        decision.insert((l, sh), d);
    }
    // The rows: each label and each other fragment present at a sample.
    let mut out = Vec::new();
    for (i0, i1) in pixels {
        let rows = &input.covers[i0..i1];
        let (x, y) = (rows[0].x, rows[0].y);
        for s in 0..16 {
            let p = crate::cover::sample_point(x as i64, y as i64, s);
            let present: BTreeSet<u32> = rows
                .iter()
                .filter(|r| r.cover[s] != 0)
                .map(|r| label_of[r.fragment as usize])
                .filter(|l| *l != NONE_U32)
                .collect();
            for &l in &present {
                for r in rows {
                    let f = r.fragment as usize;
                    if r.cover[s] == 0 || label_of[f] != NONE_U32 {
                        continue;
                    }
                    let sh = sheet_of[f];
                    let (role, pc, cc) = match class[f] {
                        Class::Open(c) => (ROLE_OPEN, 0, class_code(c)),
                        Class::Ungrouped => (ROLE_OPEN, PairCause::Label as u8, 0),
                        _ if sh == usize::MAX => (ROLE_FRONT, 0, 0),
                        _ => match decision.get(&(l, sh)) {
                            None | Some(Decision::Not) => (ROLE_FRONT, 0, 0),
                            Some(Decision::Open(c)) if !sheets[sh].kind_unknown => {
                                (ROLE_OPEN, *c as u8, 0)
                            }
                            Some(Decision::Open(_)) => (ROLE_KIND_UNKNOWN, 0, 0),
                            Some(Decision::Qualifies) => {
                                if straddle.contains(&(x, y, r.fragment)) {
                                    (ROLE_ALPHA_STRADDLE, 0, 0)
                                } else if holds(f, p, r.cover[s]) {
                                    if sheets[sh].kind_unknown {
                                        (ROLE_KIND_UNKNOWN, 0, 0)
                                    } else {
                                        (ROLE_ANCHOR, 0, 0)
                                    }
                                } else {
                                    (ROLE_FRONT, 0, 0)
                                }
                            }
                        },
                    };
                    let role = if role != ROLE_OPEN && straddle.contains(&(x, y, r.fragment)) {
                        ROLE_ALPHA_STRADDLE
                    } else {
                        role
                    };
                    out.push(LabelRow {
                        x,
                        y,
                        label: l,
                        fragment: r.fragment,
                        sample: s as u8,
                        role,
                        pair_cause: pc,
                        class_cause: cc,
                    });
                }
            }
        }
    }
    out.sort_by_key(|r| (r.y, r.x, r.sample, r.label, r.fragment));
    let pairs = decision
        .iter()
        .map(|(&(label, sheet), d)| PairDecision {
            label,
            sheet: sheet as u32,
            outcome: match d {
                Decision::Qualifies => "qualifies",
                Decision::Not => "not_nearer",
                Decision::Open(PairCause::Interval) => "interval",
                Decision::Open(PairCause::Folded) => "folded",
                Decision::Open(PairCause::Disconnected) => "disconnected",
                Decision::Open(PairCause::Mesh) => "mesh",
                Decision::Open(PairCause::Label) => "label",
            },
        })
        .collect();
    (label_of, out, pairs)
}

/// The centre test (rule 5): `None` where no fill of the sheet holds the centre.
fn by_centre(input: &Input, sheet: &Sheet, c: [f64; 2], label: &Band) -> Option<Decision> {
    let mut stands = Vec::new();
    for &i in &sheet.members {
        let i = i as usize;
        let Some(rings) = &input.outlines[i] else {
            continue;
        };
        if input.alpha[i] == Some(0) || !(inside(rings, c) || on_outline(rings, c)) {
            continue;
        }
        stands.push(plane_read(&input.frags[i].model, c).map(|b| stand(&b, label)));
    }
    if stands.is_empty() {
        return None;
    }
    let first = stands[0];
    Some(match first {
        _ if stands.iter().any(|s| *s != first) => Decision::Open(PairCause::Interval),
        None | Some(Stand::Meets) => Decision::Open(PairCause::Interval),
        Some(Stand::Nearer) => Decision::Qualifies,
        Some(Stand::NotNearer) => Decision::Not,
    })
}

/// The points test (rule 6): the sheet's depth at each counted point, by the fill holding it.
fn by_points(input: &Input, points: &[Point], label: &Band) -> Decision {
    let mut open = false;
    for &(x, y, s, f) in points {
        let b = input.frags[f as usize]
            .model
            .pixel(x as i64, y as i64, false)[s];
        match b.ok().map(|b| stand(&b, label)) {
            Some(Stand::NotNearer) => return Decision::Not,
            Some(Stand::Nearer) => {}
            _ => open = true,
        }
    }
    if open {
        Decision::Open(PairCause::Interval)
    } else {
        Decision::Qualifies
    }
}
