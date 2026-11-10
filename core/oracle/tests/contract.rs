//! The contract tests that read the core's report (contract §6, "Contract test (soundness of
//! `R`)"): on every probe fixture, at every pixel and sample where the core reports a fragment with
//! a finite key, its error bound is at most the oracle's `R` and its key lies within `2R` of the
//! oracle's, for its own row and for every row it reads on its record's plane; a key of `−∞` is
//! equal. The core is read through its diagnostics alone, traced in tiles of at most 64 x 64.

#[path = "common/fixtures.rs"]
mod fixtures;

use std::collections::{BTreeMap, HashMap};

use arachne_core::diagnostics::{Layer as CoreLayer, Pixel};
use arachne_core::stream::Stream;
use arachne_oracle::decide::decide_at_wire;
use arachne_oracle::out::{BandRow, Decisions};
use arachne_oracle::source;

const TILE: u32 = 64;

/// A per-pixel colour's least and most bytes.
type Admitted = ([u8; 4], [u8; 4]);

fn layer_code(l: &CoreLayer) -> u8 {
    match l {
        CoreLayer::BackgroundStroke => 0,
        CoreLayer::Fill => 1,
        CoreLayer::Stroke => 2,
        CoreLayer::Surface => 3,
        CoreLayer::Mask => 4,
    }
}

/// The core's trace of the whole frame, in tiles.
fn trace(core: &Stream, w: u32, h: u32) -> BTreeMap<(u32, u32), Pixel> {
    let mut out = BTreeMap::new();
    for y0 in (0..h).step_by(TILE as usize) {
        for x0 in (0..w).step_by(TILE as usize) {
            let tile = core
                .resolve_trace(x0, y0, (x0 + TILE).min(w), (y0 + TILE).min(h), true)
                .unwrap();
            out.extend(tile);
        }
    }
    out
}

/// One excess: where, which fragment, which row, and by how much (read through `Debug` in the
/// failure message).
#[derive(Debug)]
#[allow(dead_code)]
struct Excess {
    probe: &'static str,
    seq: u64,
    at: (u32, u32, usize),
    layer: u8,
    kind: u8,
    row: &'static str,
    what: &'static str,
    core: (f64, f64),
    oracle: (f64, f64),
}

/// The checks of one read against one oracle band row at sample `s`.
fn check_read(depth: f64, err: f64, band: &BandRow, s: usize) -> Option<&'static str> {
    if band.status[s] != 1 {
        return None;
    }
    let (o, r) = (band.o[s], band.r[s]);
    if o == f64::NEG_INFINITY || depth == f64::NEG_INFINITY {
        return (depth != o).then_some("a key of -inf not equal");
    }
    // A NaN on either side is an excess too.
    let within = |v: f64, bound: f64| v.partial_cmp(&bound).is_some_and(|c| c.is_le());
    if !within(err, r) {
        return Some("err above R");
    }
    if !within((depth - o).abs(), 2.0 * r) {
        return Some("key beyond 2R");
    }
    None
}

/// What a frame's comparison read: own rows, plane rows, and core entries the oracle has no
/// fragment for (a coverage question, not this test's).
#[derive(Debug, Default)]
struct Tally {
    own: usize,
    plane: usize,
    unmatched: usize,
}

fn excesses_of(
    probe: &'static str,
    seq: u64,
    d: &Decisions,
    core: &Stream,
) -> (Vec<Excess>, Tally) {
    let (w, h) = (d.header.width, d.header.height);
    let pixels = trace(core, w, h);
    // The oracle's fragments by identity, and its band rows by pixel and fragment.
    let ids: HashMap<(u32, u32, u8, u32), u32> = d
        .fragments
        .iter()
        .map(|f| ((f.object, f.part, f.layer, f.sub), f.id))
        .collect();
    let mut bands: HashMap<(u32, u32, u32, u8), &BandRow> = HashMap::new();
    for b in &d.bands {
        bands.insert((b.x, b.y, b.fragment, b.which), b);
    }
    let mut out = Vec::new();
    let mut tally = Tally::default();
    for ((x, y), px) in &pixels {
        let Some(r) = &px.reads else { continue };
        let frag_of = |e: usize| {
            let en = &px.entries[e];
            ids.get(&(en.leaf.0, en.leaf.1, layer_code(&en.layer), en.sub))
                .copied()
        };
        let mut push = |s: usize, row: &'static str, what, c: (f64, f64), b: &BandRow| {
            let f = &d.fragments[b.fragment as usize];
            out.push(Excess {
                probe,
                seq,
                at: (*x, *y, s),
                layer: f.layer,
                kind: f.kind,
                row,
                what,
                core: c,
                oracle: (b.o[s], b.r[s]),
            });
        };
        for (e, en) in px.entries.iter().enumerate() {
            let Some(f) = frag_of(e) else {
                tally.unmatched += en.coverage.iter().any(|c| *c > 0) as usize;
                continue;
            };
            let Some(own) = bands.get(&(*x, *y, f, 0)) else {
                continue;
            };
            for s in 0..16 {
                if en.coverage[s] == 0 {
                    continue;
                }
                tally.own += 1;
                if let Some(what) = check_read(r.depths[e][s], r.err[e], own, s) {
                    push(s, "own", what, (r.depths[e][s], r.err[e]), own);
                }
            }
        }
        for a in &r.against {
            let Some(f) = frag_of(a.entry) else { continue };
            let Some(row) = bands.get(&(*x, *y, f, 1)) else {
                continue;
            };
            let (ca, cb) = (&px.entries[a.entry], &px.entries[a.other]);
            for s in 0..16 {
                if ca.coverage[s] == 0 || cb.coverage[s] == 0 {
                    continue;
                }
                tally.plane += 1;
                if let Some(what) = check_read(a.depths[s], a.err, row, s) {
                    push(s, "plane", what, (a.depths[s], a.err), row);
                }
            }
        }
    }
    (out, tally)
}

#[test]
fn the_core_samples_where_the_contract_places_its_samples() {
    assert_eq!(
        arachne_core::composite::SAMPLE_OFFSETS,
        arachne_oracle::cover::SAMPLE_OFFSETS
    );
}

#[test]
fn core_err_within_band() {
    let dir = fixtures::scratch("contract");
    let mut excess = Vec::new();
    let mut tally = Tally::default();
    for p in fixtures::probe_files(&dir) {
        let (_, wire) = source::read_file(&p.path).unwrap();
        let deltas = arachne_core::xf::wire::decode_file(&std::fs::read(&p.path).unwrap()).unwrap();
        let mut core = Stream::new();
        for (wd, delta) in wire.iter().zip(&deltas) {
            core.apply(delta).unwrap();
            let d = decide_at_wire(&wire, wd.seq, None).unwrap();
            assert_eq!(d.header.status, 0, "{}: {:?}", p.name, d.refused);
            let (e, n) = excesses_of(p.name, wd.seq, &d, &core);
            excess.extend(e);
            tally.own += n.own;
            tally.plane += n.plane;
            tally.unmatched += n.unmatched;
        }
    }
    let mut classes: BTreeMap<(u8, u8, &str, &str), usize> = BTreeMap::new();
    let mut by_probe: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &excess {
        *classes.entry((e.layer, e.kind, e.row, e.what)).or_default() += 1;
        *by_probe.entry(e.probe).or_default() += 1;
    }
    let reads = tally.own + tally.plane;
    assert!(tally.own > 100_000, "{tally:?}");
    assert!(
        excess.is_empty(),
        "{} of {reads} reads exceed the band; by (layer, kind, row, what): {classes:?}; by probe: \
         {by_probe:?}; first: {:#?}",
        excess.len(),
        &excess[..excess.len().min(8)]
    );
}

#[test]
fn core_colours_are_the_oracle_s() {
    // Contract §9: every fragment the core reports carries the colour the oracle computes from
    // XF: a solid's bytes exactly, a lit triangle's within the bytes the oracle admits there. The
    // oracle does not compute gradients; their fragments are counted, not compared.
    let dir = fixtures::scratch("contract-colour");
    let (mut solid, mut lit, mut gradient) = (0, 0, 0);
    let mut wrong = Vec::new();
    for p in fixtures::probe_files(&dir) {
        let (_, wire) = source::read_file(&p.path).unwrap();
        let deltas = arachne_core::xf::wire::decode_file(&std::fs::read(&p.path).unwrap()).unwrap();
        let mut core = Stream::new();
        for (wd, delta) in wire.iter().zip(&deltas) {
            core.apply(delta).unwrap();
            let d = decide_at_wire(&wire, wd.seq, None).unwrap();
            let ids: HashMap<(u32, u32, u8, u32), u32> = d
                .fragments
                .iter()
                .map(|f| ((f.object, f.part, f.layer, f.sub), f.id))
                .collect();
            let per_pixel: HashMap<(u32, u32, u32), Admitted> = d
                .pixel_colours
                .iter()
                .map(|c| ((c.x, c.y, c.fragment), (c.lo, c.hi)))
                .collect();
            let (w, h) = (d.header.width, d.header.height);
            for ((x, y), px) in trace(&core, w, h) {
                for en in &px.entries {
                    let key = (en.leaf.0, en.leaf.1, layer_code(&en.layer), en.sub);
                    let Some(f) = ids.get(&key) else { continue };
                    let c = &d.colours[*f as usize];
                    match c.kind {
                        arachne_oracle::out::COLOUR_SOLID => {
                            solid += 1;
                            if en.colour != c.rgba {
                                wrong.push((p.name, x, y, key, en.colour, c.rgba, c.rgba));
                            }
                        }
                        arachne_oracle::out::COLOUR_PER_PIXEL => {
                            lit += 1;
                            let Some((lo, hi)) = per_pixel.get(&(x, y, *f)) else {
                                continue;
                            };
                            if (0..4).any(|k| en.colour[k] < lo[k] || en.colour[k] > hi[k]) {
                                wrong.push((p.name, x, y, key, en.colour, *lo, *hi));
                            }
                        }
                        _ => gradient += 1,
                    }
                }
            }
        }
    }
    assert!(solid > 10_000 && lit > 1_000, "{solid} {lit} {gradient}");
    assert!(
        wrong.is_empty(),
        "{} of {} fragment colours differ (solid {solid}, lit {lit}); first: {:?}",
        wrong.len(),
        solid + lit,
        &wrong[..wrong.len().min(8)]
    );
}
