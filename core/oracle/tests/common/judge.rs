//! A report judged against the oracle's decisions at one sample, by the rules `FORMAT.md` states
//! for a reader: presence and cell bytes from `COVR`, the order from `PAIR`'s edges by
//! `cycle_rank@2`, a fragment's `truth_unable` from its own `BAND` status, a label's place from
//! `LABL`, and the named findings (`zero_alpha_fragment`, `invisible_registry_offer`,
//! `label_plate_model`). It recomputes nothing: depth, ties and
//! qualification are the oracle's facts, read here.
#![allow(dead_code)]

use std::collections::HashMap;

use arachne_oracle::order::{Edges, admissible};
use arachne_oracle::out::{
    Decisions, FLAG_UNBOUNDED, FLAG_ZERO_ALPHA, LABEL_UNABLE, NONE_U32, ROLE_ALPHA_STRADDLE,
    ROLE_ANCHOR, ROLE_KIND_UNKNOWN, ROLE_OPEN,
};

/// A fragment's identity as a report names it: `(object, part, layer, sub)`.
pub type Identity = (u32, u32, u8, u32);

/// One reported fragment at a pixel: its identity and its cell bytes at each sample.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub identity: Identity,
    pub cells: [u8; 16],
}

/// What a report says of one pixel: its fragments, and at each sample those covering it, front
/// first (indices into `entries`).
#[derive(Clone, Debug, PartialEq)]
pub struct Reported {
    pub x: u32,
    pub y: u32,
    pub entries: Vec<Entry>,
    pub order: Vec<Vec<usize>>,
}

/// A sample's verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Agree,
    /// `mismatch`, with the name it is counted under (`mismatch` where none).
    Mismatch(&'static str),
    /// `label_rule_open`: its pair cause and class cause codes (`LABL`).
    Open(u8, u8),
    /// `truth_unable`, by name.
    Unable(&'static str),
}

impl Verdict {
    pub fn word(&self) -> String {
        match self {
            Verdict::Agree => "agreement".into(),
            Verdict::Mismatch(n) => format!("mismatch:{n}"),
            Verdict::Open(p, c) => format!("label_rule_open:{p}:{c}"),
            Verdict::Unable(n) => format!("truth_unable:{n}"),
        }
    }
}

/// The oracle's fragment for each identity, and the zero-alpha ones apart.
pub fn ids(d: &Decisions) -> (HashMap<Identity, u32>, HashMap<Identity, u32>) {
    let (mut live, mut zero) = (HashMap::new(), HashMap::new());
    for f in &d.fragments {
        let key = (f.object, f.part, f.layer, f.sub);
        if f.flags & FLAG_ZERO_ALPHA != 0 {
            zero.insert(key, f.id);
        } else {
            live.insert(key, f.id);
        }
    }
    (live, zero)
}

/// Which of `PAIR`'s edge sets an order is read against.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Set {
    /// `edges`: the admissible orders.
    Edges,
    /// `offered`: with the records that paint nothing offered too.
    Offered,
    /// `plate`: the plate model's orders.
    Plate,
}

/// `PAIR`'s rows at a sample in `set`: each pair's bits, label and the provenance of its strong
/// edges (the plate model's read in the scenarios `edges`' are).
fn edges_of(
    d: &Decisions,
    x: u32,
    y: u32,
    s: usize,
    set: Set,
) -> HashMap<(u32, u32), (u8, u8, u8)> {
    d.pairs
        .iter()
        .filter(|p| (p.x, p.y, p.sample as usize) == (x, y, s))
        .map(|p| {
            let bits = match set {
                Set::Edges => p.edges,
                Set::Offered => p.offered,
                Set::Plate => p.plate,
            };
            let provenance = match set {
                Set::Offered => p.offered_provenance,
                Set::Edges | Set::Plate => p.provenance,
            };
            ((p.a, p.b), (bits, p.label, provenance))
        })
        .collect()
}

fn edges(bits: u8) -> Edges {
    Edges {
        weak: bits & 1 != 0,
        a_front: bits & 2 != 0,
        b_front: bits & 4 != 0,
    }
}

/// Whether `order` (fragment ids, front first) is admissible over its own fragments by `PAIR`'s
/// `set`.
fn admits(d: &Decisions, x: u32, y: u32, s: usize, order: &[u32], set: Set) -> Option<bool> {
    let table = edges_of(d, x, y, s, set);
    let mut present: Vec<usize> = order.iter().map(|f| *f as usize).collect();
    present.sort();
    let frag = |i: usize| &d.fragments[i];
    let paint = |i: usize| (frag(i).entry, frag(i).sub);
    let missing = std::cell::Cell::new(false);
    let edge = |i: usize, j: usize| {
        let (a, b) = (present[i] as u32, present[j] as u32);
        match table.get(&(a, b)) {
            Some((bits, _, provenance)) => (edges(*bits), *provenance),
            None => {
                missing.set(true);
                (Edges::ALL, 0)
            }
        }
    };
    let order: Vec<usize> = order.iter().map(|f| *f as usize).collect();
    let r = admissible(&present, &paint, &edge, &order);
    assert!(
        !missing.get(),
        "a pair of present fragments has no PAIR row"
    );
    r
}

/// Whether `order` is one of the label rule's orders at the sample, from the reported order of
/// the others: each label immediately behind the rearmost of its anchors there, else in front;
/// labels at one place, and a label's own fragments, in paint order. A kind-unknown fragment is
/// read as an anchor and not, for each label apart, so a label may stand behind any fragment it
/// may read as an anchor with no anchor read behind it: each label's place is decided on its own,
/// with no enumeration of the readings, however many they are.
fn label_order_admitted(d: &Decisions, x: u32, y: u32, s: usize, order: &[u32]) -> bool {
    let frag = |i: u32| &d.fragments[i as usize];
    let others: Vec<u32> = order
        .iter()
        .copied()
        .filter(|f| frag(*f).label == NONE_U32)
        .collect();
    let mut labels: Vec<u32> = order
        .iter()
        .map(|f| frag(*f).label)
        .filter(|l| *l != NONE_U32)
        .collect();
    labels.sort();
    labels.dedup();
    // The rows of the labels and fragments the order holds: no other is present (§7).
    let rows: Vec<_> = d
        .labels
        .iter()
        .filter(|r| {
            (r.x, r.y, r.sample as usize) == (x, y, s)
                && labels.contains(&r.label)
                && others.contains(&r.fragment)
        })
        .collect();
    let role = |l: u32, f: u32| {
        rows.iter()
            .find(|r| r.label == l && r.fragment == f)
            .map(|r| r.role)
    };
    let mut placed: Vec<(usize, (u32, u32), u32)> = Vec::new();
    for &l in &labels {
        // Its place in the report: the others in front of its first fragment.
        let first = order.iter().position(|f| frag(*f).label == l).unwrap();
        let place = order[..first]
            .iter()
            .filter(|f| frag(**f).label == NONE_U32)
            .count();
        // Behind its rearmost anchor, or a kind-unknown fragment behind that read as one.
        let anchored = others
            .iter()
            .rposition(|f| role(l, *f) == Some(ROLE_ANCHOR))
            .map_or(0, |i| i + 1);
        let may = place == anchored
            || (place > anchored && role(l, others[place - 1]) == Some(ROLE_KIND_UNKNOWN));
        if !may {
            return false;
        }
        for &f in order.iter().filter(|f| frag(**f).label == l) {
            placed.push((place, (frag(f).entry, frag(f).sub), f));
        }
    }
    // Front first: by place, then the later painted first.
    placed.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    let mut full = Vec::new();
    let mut k = 0;
    for (i, f) in others.iter().enumerate() {
        while k < placed.len() && placed[k].0 == i {
            full.push(placed[k].2);
            k += 1;
        }
        full.push(*f);
    }
    full.extend(placed[k..].iter().map(|p| p.2));
    full == order
}

/// `BAND`'s name for a status that leaves its fragment `truth_unable` whatever is read with it:
/// 2 `band_unbounded` and 4 `pivot_unknown`, each by fragment (§6, §2). A `curve_recut` read (3)
/// is unable only where a pair reads it, which its `PAIR` label says.
fn unable_status(status: u8) -> Option<&'static str> {
    match status {
        2 => Some("band_unbounded"),
        4 => Some("pivot_unknown"),
        _ => None,
    }
}

/// The verdict at sample `s` of the reported pixel.
pub fn judge(d: &Decisions, rep: &Reported, s: usize) -> Verdict {
    let (x, y) = (rep.x, rep.y);
    // An unbounded fragment may stand anywhere: it is `truth_unable` at every sample of its box,
    // the window, whatever the report holds there (FORMAT `flags` bit 0, §6). It has no rows, so
    // no report can show it absent.
    let inside =
        |b: [i32; 4]| (b[0]..b[2]).contains(&(x as i32)) && (b[1]..b[3]).contains(&(y as i32));
    if d.fragments
        .iter()
        .any(|f| f.flags & FLAG_UNBOUNDED != 0 && inside(f.bbox))
    {
        return Verdict::Unable("band_unbounded");
    }
    let (live, zero) = ids(d);
    let rows: Vec<_> = d.covers.iter().filter(|c| (c.x, c.y) == (x, y)).collect();
    let cover_of = |f: u32| rows.iter().find(|c| c.fragment == f);
    // Presence and cell bytes.
    let mut reported: Vec<Option<u32>> = Vec::new();
    for e in &rep.entries {
        let f = live.get(&e.identity).copied();
        if e.cells[s] > 0 {
            let Some(f) = f else {
                return Verdict::Mismatch(if zero.contains_key(&e.identity) {
                    "zero_alpha_fragment"
                } else {
                    "mismatch"
                });
            };
            // Its own key's status at the sample: a fragment with no key bounded is
            // `truth_unable` there, whether or not it makes a pair (§6).
            let own = d
                .bands
                .iter()
                .find(|b| (b.x, b.y, b.fragment, b.which) == (x, y, f, 0));
            if let Some(name) = own.and_then(|b| unable_status(b.status[s])) {
                return Verdict::Unable(name);
            }
            let Some(c) = cover_of(f) else {
                return Verdict::Mismatch("mismatch");
            };
            if c.cover[s] == 0 || e.cells[s] < c.cell_lo[s] || e.cells[s] > c.cell_hi[s] {
                return Verdict::Mismatch("mismatch");
            }
        }
        reported.push(f);
    }
    for c in &rows {
        let shown = rep
            .entries
            .iter()
            .zip(&reported)
            .any(|(e, f)| *f == Some(c.fragment) && e.cells[s] > 0);
        if c.cover[s] == 1 && !shown {
            return Verdict::Mismatch("mismatch");
        }
    }
    // The order: of the fragments reported present at the sample, each once (§7).
    let mut listed: Vec<usize> = rep.order[s]
        .iter()
        .copied()
        .filter(|&e| rep.entries[e].cells[s] > 0)
        .collect();
    listed.sort();
    let present: Vec<usize> = (0..rep.entries.len())
        .filter(|&e| rep.entries[e].cells[s] > 0)
        .collect();
    if listed != present {
        return Verdict::Mismatch("mismatch");
    }
    let order: Vec<u32> = rep.order[s]
        .iter()
        .filter(|&&e| rep.entries[e].cells[s] > 0)
        .map(|&e| reported[e].unwrap())
        .collect();
    let table = edges_of(d, x, y, s, Set::Edges);
    // A pair `truth_unable` at the sample is named by its fragments' reads there: the class of
    // the first status among them that states no bound (§6). Only a pair of fragments the report
    // holds present is present (§7): an `Either` one it leaves out is in no pair.
    let unable: Vec<u32> = table
        .iter()
        .filter(|&(&(a, b), &(_, label, _))| {
            label == LABEL_UNABLE && order.contains(&a) && order.contains(&b)
        })
        .flat_map(|(&(a, b), _)| [a, b])
        .collect();
    if !unable.is_empty() {
        let status_of = |st: u8| {
            d.bands
                .iter()
                .any(|r| (r.x, r.y) == (x, y) && unable.contains(&r.fragment) && r.status[s] == st)
        };
        let name = [
            (2, "band_unbounded"),
            (3, "curve_recut"),
            (4, "pivot_unknown"),
        ]
        .into_iter()
        .find(|&(st, _)| status_of(st))
        .map_or("truth_unable", |(_, n)| n);
        return Verdict::Unable(name);
    }
    let has_label = order
        .iter()
        .any(|f| d.fragments[*f as usize].label != NONE_U32);
    if !has_label {
        return match admits(d, x, y, s, &order, Set::Edges) {
            Some(true) => Verdict::Agree,
            None => Verdict::Unable("completion_bound"),
            Some(false) if admits(d, x, y, s, &order, Set::Offered) == Some(true) => {
                Verdict::Mismatch("invisible_registry_offer")
            }
            Some(false) => Verdict::Mismatch("mismatch"),
        };
    }
    // The label rule's rows of the fragments the report holds present, likewise.
    let lrows: Vec<_> = d
        .labels
        .iter()
        .filter(|r| {
            (r.x, r.y, r.sample as usize) == (x, y, s)
                && order.contains(&r.fragment)
                && order
                    .iter()
                    .any(|f| d.fragments[*f as usize].label == r.label)
        })
        .collect();
    if let Some(r) = lrows.iter().find(|r| r.role == ROLE_OPEN) {
        return Verdict::Open(r.pair_cause, r.class_cause);
    }
    if lrows.iter().any(|r| r.role == ROLE_ALPHA_STRADDLE) {
        return Verdict::Unable("label_alpha_straddle");
    }
    let others: Vec<u32> = order
        .iter()
        .copied()
        .filter(|f| d.fragments[*f as usize].label == NONE_U32)
        .collect();
    let Some(ordinary) = admits(d, x, y, s, &others, Set::Edges) else {
        return Verdict::Unable("completion_bound");
    };
    if ordinary && label_order_admitted(d, x, y, s, &order) {
        return Verdict::Agree;
    }
    // Outside the label rule's orders. The plate model's orders are those `PAIR`'s `plate`
    // relations admit, the label's pairs included; an undetermined one (`0`) admits none, so no
    // order is named there. Past the completion bound the mismatch is certain and its
    // name is not.
    match admits(d, x, y, s, &order, Set::Plate) {
        Some(true) => Verdict::Mismatch("label_plate_model"),
        _ => Verdict::Mismatch("mismatch"),
    }
}
