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

/// The label rule's orders at the sample, from the reported order of the others: each label
/// immediately behind the rearmost of its anchors there, else in front; labels at one place, and a
/// label's own fragments, in paint order. A kind-unknown fragment is read as an anchor and not.
fn label_orders(d: &Decisions, x: u32, y: u32, s: usize, order: &[u32]) -> Vec<Vec<u32>> {
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
    let rows: Vec<_> = d
        .labels
        .iter()
        .filter(|r| (r.x, r.y, r.sample as usize) == (x, y, s))
        .collect();
    let unknown: Vec<(u32, u32)> = rows
        .iter()
        .filter(|r| r.role == ROLE_KIND_UNKNOWN)
        .map(|r| (r.label, r.fragment))
        .collect();
    assert!(unknown.len() <= 12, "too many kind-unknown readings");
    let mut out = Vec::new();
    for mask in 0..(1u32 << unknown.len()) {
        let anchor = |l: u32, f: u32| {
            rows.iter()
                .any(|r| r.label == l && r.fragment == f && r.role == ROLE_ANCHOR)
                || unknown
                    .iter()
                    .enumerate()
                    .any(|(k, u)| *u == (l, f) && mask & (1 << k) != 0)
        };
        // Each label's place: the number of others in front of it.
        let mut placed: Vec<(usize, (u32, u32), u32)> = Vec::new();
        for &l in &labels {
            let place = others
                .iter()
                .rposition(|f| anchor(l, *f))
                .map_or(0, |i| i + 1);
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
        out.push(full);
    }
    out
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
            if d.fragments[f as usize].flags & FLAG_UNBOUNDED != 0 {
                return Verdict::Unable("band_unbounded");
            }
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
    if table.values().any(|(_, label, _)| *label == LABEL_UNABLE) {
        return Verdict::Unable("truth_unable");
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
    let lrows: Vec<_> = d
        .labels
        .iter()
        .filter(|r| (r.x, r.y, r.sample as usize) == (x, y, s))
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
    if ordinary && label_orders(d, x, y, s, &order).contains(&order) {
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
