//! The contract tests that read the core's report (contract §6, "Contract test (soundness of
//! `R`)"): on every probe fixture, at every pixel and sample where the core reports a fragment with
//! a finite key, its error bound is at most the oracle's `R` and its key lies within `2R` of the
//! oracle's, for its own row and for every row it reads on its record's plane; a key of `−∞` is
//! equal. The core is read through its diagnostics alone, traced in tiles of at most 64 x 64.

#[path = "common/fixtures.rs"]
mod fixtures;
#[path = "common/judge.rs"]
mod judge;
#[path = "common/scenes.rs"]
mod scenes;

use std::collections::{BTreeMap, HashMap};

use arachne_core::diagnostics::{Layer as CoreLayer, Pixel};
use arachne_core::stream::Stream;
use arachne_oracle::decide::decide_at_wire;
use arachne_oracle::out::{BandRow, Decisions};
use arachne_oracle::source;
use judge::{Entry, Reported};

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

/// The core's trace of the frame `d` decides.
fn frame_trace(core: &Stream, d: &Decisions) -> BTreeMap<(u32, u32), Pixel> {
    trace(core, d.header.width, d.header.height)
}

/// The core's report of each traced pixel: its fragments and, at each sample, every one covering
/// it, front first, before any cull (`samples_all`).
fn core_reports(pixels: &BTreeMap<(u32, u32), Pixel>) -> HashMap<(u32, u32), Reported> {
    pixels
        .iter()
        .filter_map(|(&(x, y), p)| {
            let reads = p.reads.as_ref()?;
            Some((
                (x, y),
                Reported {
                    x,
                    y,
                    entries: p
                        .entries
                        .iter()
                        .map(|e| Entry {
                            identity: (e.leaf.0, e.leaf.1, layer_code(&e.layer), e.sub),
                            cells: e.coverage,
                        })
                        .collect(),
                    order: reads.samples_all.clone(),
                },
            ))
        })
        .collect()
}

/// One excess: where, which fragment, which row, and by how much (read through `Debug` in the
/// failure message).
#[derive(Debug)]
#[allow(dead_code)]
struct Excess {
    probe: &'static str,
    seq: u64,
    at: (u32, u32, usize),
    fragment: u32,
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
    /// Own reads of a stroke adjusted on its own fill where the fill's cover is `Either`: those the
    /// stroke's own unadjusted key explains, and those the key it takes does.
    either_own: usize,
    either_taken: usize,
}

/// Which of an adjusted stroke's two keys over an `Either` fill a core's key matched.
#[derive(Debug, PartialEq)]
enum Candidate {
    Taken,
    Own,
}

/// The check of an own read against its row `own` at sample `s` (§6's soundness test of `R`),
/// with `unadjusted` the stroke's own unadjusted key's row where it is adjusted on its own fill
/// and the fill's cover is `Either`: there the key passes within `2R` of either key, `R` the
/// row's (the Either bound). `Ok` names the key matched where there are two.
fn check_own(
    depth: f64,
    err: f64,
    own: &BandRow,
    unadjusted: Option<&BandRow>,
    s: usize,
) -> Result<Option<Candidate>, &'static str> {
    let two = unadjusted.filter(|u| own.status[s] == 1 && u.status[s] == 1);
    match (check_read(depth, err, own, s), two) {
        (None, None) => Ok(None),
        (None, Some(_)) => Ok(Some(Candidate::Taken)),
        (Some("key beyond 2R"), Some(u)) => {
            let near = (depth - u.o[s]).abs().partial_cmp(&(2.0 * own.r[s]));
            if near.is_some_and(|c| c.is_le()) {
                Ok(Some(Candidate::Own))
            } else {
                Err("key beyond 2R")
            }
        }
        (Some(what), _) => Err(what),
    }
}

#[test]
fn over_an_either_fill_a_stroke_s_key_passes_within_2r_of_either_key() {
    // §6, the soundness test of `R`, a stroke adjusted on its own fill: where the fill's cover is
    // `Either`, the core's key passes within `2R` of the stroke's own unadjusted key or of the key
    // it takes. The taken key −1.0592 (the fill's), R 5e-8; the stroke's own −1.05916.
    let row = |which: u8, o: f64| BandRow {
        x: 0,
        y: 0,
        fragment: 0,
        which,
        o: [o; 16],
        r: [5e-8; 16],
        s_minus: [0.0; 16],
        s_plus: [1e-5; 16],
        status: [1; 16],
    };
    let (own, unadjusted) = (row(0, -1.0592), row(2, -1.05916));
    assert_eq!(
        check_own(-1.05916, 5e-8, &own, Some(&unadjusted), 0),
        Ok(Some(Candidate::Own))
    );
    assert_eq!(
        check_own(-1.0592, 5e-8, &own, Some(&unadjusted), 0),
        Ok(Some(Candidate::Taken))
    );
    // Neither key: an excess; and where the fill covers certainly there is one key.
    assert_eq!(
        check_own(-1.0594, 5e-8, &own, Some(&unadjusted), 0),
        Err("key beyond 2R")
    );
    assert_eq!(
        check_own(-1.05916, 5e-8, &own, None, 0),
        Err("key beyond 2R")
    );
    // The bound is checked as before.
    assert_eq!(
        check_own(-1.05916, 1e-7, &own, Some(&unadjusted), 0),
        Err("err above R")
    );
}

fn excesses_of(
    probe: &'static str,
    seq: u64,
    d: &Decisions,
    pixels: &BTreeMap<(u32, u32), Pixel>,
) -> (Vec<Excess>, Tally) {
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
    for ((x, y), px) in pixels {
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
                fragment: b.fragment,
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
            let unadjusted = bands.get(&(*x, *y, f, 2)).copied();
            for s in 0..16 {
                if en.coverage[s] == 0 {
                    continue;
                }
                tally.own += 1;
                match check_own(r.depths[e][s], r.err[e], own, unadjusted, s) {
                    Ok(Some(Candidate::Own)) => tally.either_own += 1,
                    Ok(Some(Candidate::Taken)) => tally.either_taken += 1,
                    Ok(None) => {}
                    Err(what) => push(s, "own", what, (r.depths[e][s], r.err[e]), own),
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
            let (e, n) = excesses_of(p.name, wd.seq, &d, &frame_trace(&core, &d));
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

/// The named reads `core_err_within_band` must hold on, each in a scene of its own (View C): a
/// stroke whose own key is nearer than its fill's (`stroke_on_own_fill@2` keeps it), a stroke read
/// on a plane its path shares, outside its own fill (`shared_plane_read@1`), and a line read on the
/// second of the two flat identities it attaches to. Each of the last two is shown read by the
/// plane reads it adds to the same scene without the fragment it is read against.
#[test]
fn core_err_within_band_on_the_named_reads() {
    use scenes::*;
    let tilted = |z0: f64, slope: f64| {
        let z = |x: f64| z0 + slope * x;
        polygon(&[
            [-3.0, -3.0, z(-3.0)],
            [3.0, -3.0, z(3.0)],
            [3.0, 3.0, z(3.0)],
            [-3.0, 3.0, z(-3.0)],
        ])
    };
    // A square on z = x / 2, filled and stroked 1 unit wide: at the samples near its edges both
    // cover, the stroke's key, its curve's, is nearer than the plane on one side.
    let mut stroke_nearer = Build::new(true);
    let r = stroke_nearer.fill(tilted(0.0, 0.5), OPAQUE_BLUE);
    r.paint.stroke = arachne_core::xf::Paint::One(WHITE);
    r.paint.stroke_width = 1.0;
    // Two squares on z = 0 a quarter apart, the first stroked: where its stroke reaches past its own
    // fill onto the second, it is read on their shared plane. The fills meet at no sample.
    let stroked = |b: &mut Build| {
        let r = b.fill(rect([-4.0, 0.0], [-2.0, 2.0], 0.0), OPAQUE_BLUE);
        r.paint.stroke = arachne_core::xf::Paint::One(WHITE);
        r.paint.stroke_width = 1.0;
    };
    let mut alone = Build::new(true);
    stroked(&mut alone);
    let mut shared = Build::new(true);
    stroked(&mut shared);
    shared.fill(rect([0.25, 4.0], [-2.0, 2.0], 0.0), OPAQUE_RED);
    // A line along the y axis on z = 0, lying on a square on z = 0 and on a square on z = x / 2:
    // it attaches to both identities, and against the second square reads the second.
    let first = |b: &mut Build| {
        b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_BLUE);
    };
    let line = |b: &mut Build| {
        b.stroke(
            open(&[[0.0, -3.1875, 0.0], [0.0, 3.1875, 0.0]]),
            OPAQUE_GREEN,
            0.5,
        );
    };
    let mut one = Build::new(true);
    first(&mut one);
    line(&mut one);
    let mut secondary = Build::new(true);
    first(&mut secondary);
    secondary.fill(tilted(0.0, 0.5), OPAQUE_RED);
    line(&mut secondary);

    let run = |name: &str, b: Build| {
        let delta = b.delta(1, view_c());
        let wire = [source::WireDelta::of(delta.clone())];
        let d = decide_at_wire(&wire, 1, None).unwrap();
        assert_eq!(d.header.status, 0, "{name}: {:?}", d.refused);
        let mut core = Stream::new();
        core.apply(&delta).unwrap();
        let (excess, tally) = excesses_of("named", 1, &d, &frame_trace(&core, &d));
        assert!(excess.is_empty(), "{name}: {excess:#?}");
        assert!(
            tally.own > 1000 && tally.unmatched == 0,
            "{name}: {tally:?}"
        );
        (d, tally)
    };

    let (d, _) = run("stroke_nearer_than_fill", stroke_nearer);
    let own: Vec<_> = d.bands.iter().filter(|b| b.which == 0).collect();
    let nearer = own.iter().any(|s| {
        d.fragments[s.fragment as usize].layer == 2
            && own.iter().any(|f| {
                (f.x, f.y) == (s.x, s.y)
                    && d.fragments[f.fragment as usize].layer == 1
                    && (0..16).any(|k| s.status[k] == 1 && f.status[k] == 1 && s.o[k] < f.o[k])
            })
    });
    assert!(
        nearer,
        "no sample where the stroke's own key is nearer than its fill's"
    );

    let (_, without) = run("stroke_alone", alone);
    let (_, with) = run("stroke_on_shared_plane_outside_fill", shared);
    assert!(with.plane > without.plane, "{without:?} {with:?}");

    let (_, without) = run("line_on_one_identity", one);
    let (_, with) = run("secondary_identity", secondary);
    assert!(with.plane > without.plane, "{without:?} {with:?}");
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

#[test]
fn the_core_paints_at_the_alpha_band_s_edges_as_alpha_round_says() {
    // `alpha_round@1` (contract §1): an op whose alpha byte, trunc(d · 65535 + 0.5) >> 8, is 0
    // paints nothing and is no contributor; one whose byte is 1 paints. Just below, at and just
    // above 255.5 / 65535, a white square over black: the core reports a fragment exactly where the
    // oracle makes one.
    use arachne_core::xf::{
        Background, Camera, Delta, Layer, LightModel, OrderEntry, Paint, Record, RecordKind, Shape,
        ShapeData, Space, Time, View,
    };
    let view = View {
        pixel_width: 16,
        pixel_height: 16,
        frame_width: 16.0,
        frame_height: 16.0,
        frame_center: [0.0; 3],
        mapping_centre: [0.0; 3],
        background: Background::Color([0.0, 0.0, 0.0, 1.0]),
        camera: Camera::Flat,
        cairo_line_width_multiple: 1.0,
        use_z_index: false,
        light: LightModel::Ce,
        input_grid: 0.0,
    };
    let corners = [[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]];
    let mut points = Vec::new();
    for i in 0..4 {
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        let at = |t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.0];
        points.extend([at(0.0), at(1.0 / 3.0), at(2.0 / 3.0), at(1.0)]);
    }
    let edge = 255.5 / 65535.0;
    for d in [edge * (1.0 - 1e-9), edge, edge * (1.0 + 1e-9)] {
        let byte = ((d * 65535.0 + 0.5) as u32) >> 8;
        let shape = Shape::new(ShapeData::Path {
            points: points.clone(),
        });
        let mut r = Record::new(RecordKind::Path, 0, 100);
        r.geometry.shape_a = Some(shape.key());
        r.paint.fill = Paint::One([1.0, 1.0, 1.0, d]);
        r.depth.space = Space::Screen;
        let mut delta = Delta::new(Time { num: 1, den: 1 }, 1);
        delta.version = (1, 4);
        delta.view = Some(view.clone());
        delta.order = Some(vec![OrderEntry {
            layer: Layer::Moving,
            slot: 0,
            part: 0,
        }]);
        delta.upload = vec![shape];
        delta.set = vec![r];
        let oracle =
            arachne_oracle::decide::decide_at(std::slice::from_ref(&delta), 1, None).unwrap();
        let mut core = Stream::new();
        core.apply(&delta).unwrap();
        let px = core.resolve_trace(8, 8, 9, 9, false).unwrap();
        let entries = px.get(&(8, 8)).map_or(0, |p| p.entries.len());
        assert_eq!(oracle.fragments.is_empty(), byte == 0, "oracle at {d:e}");
        assert_eq!(entries > 0, byte > 0, "core at {d:e}, byte {byte}");
    }
}

/// One case class's measures over its frames (`soundness_measures`).
#[derive(Debug, Default)]
struct Measures {
    frames: usize,
    excess: usize,
    own_reads: usize,
    plane_reads: usize,
    /// Samples with a fragment present, and of them: a pair `TieBand`, a fragment `Either`, a
    /// pair under a tie policy, any of the three (`tie_share`), and a pair `ExactTie`.
    samples: usize,
    tie_band: usize,
    either: usize,
    policy_tie: usize,
    tied: usize,
    exact_tie: usize,
    /// Samples the oracle leaves unjudged, by reason (`open_share`).
    open: BTreeMap<String, usize>,
    /// Pair-samples by label, and stroke samples adjusted on their own fill (both `Yes`, or the
    /// fill `Either`).
    labels: BTreeMap<u8, usize>,
    on_fill: (usize, usize),
    /// Own reads over an `Either` fill that the stroke's own key explains, and that the taken key
    /// does.
    either_kept: (usize, usize),
    /// Samples (and orders) whose admitted orders hold one that only a cycle through a row read
    /// produces, one that only a cycle through a flat identity does, and one that only an own-key
    /// cycle would (the orders `cycle_rank@2` removes); samples past the completion bound.
    cycle_row: (usize, usize),
    cycle_flat: (usize, usize),
    cycle_own: (usize, usize),
    cycle_unbounded: usize,
    /// A label's pairs at a sample: `plate` determined (1, 2 or 4) and undetermined (0).
    plate: (usize, usize),
}

/// Whether some choice of strong edges at a sample can cycle: a directed cycle of three or more
/// fragments in the graph of every strong edge the pairs admit.
fn strong_cycle_possible(n: usize, pairs: &[(usize, usize, u8, u8)]) -> bool {
    let mut g = vec![vec![false; n]; n];
    for (i, j, e, _) in pairs {
        g[*i][*j] |= e & 2 != 0;
        g[*j][*i] |= e & 4 != 0;
    }
    (0..n).any(|i| {
        (0..n).filter(|&j| g[i][j]).any(|j| {
            // From j back to i, never through the pair {i, j}.
            let mut seen = vec![false; n];
            let mut stack = vec![j];
            seen[j] = true;
            while let Some(v) = stack.pop() {
                for w in 0..n {
                    if !g[v][w] || seen[w] || (v == j && w == i) {
                        continue;
                    }
                    if w == i {
                        return true;
                    }
                    seen[w] = true;
                    stack.push(w);
                }
            }
            false
        })
    })
}

/// The completion bound of the cycle lines: samples with more completions are counted apart.
const CYCLE_BOUND: f64 = 1024.0;

/// The ranks of every completion at a sample, as `admissible` forms them (§7), under four
/// readings: `cycle_rank@2` (a choice whose own-key strong edges cycle discarded), the same
/// discarding cycles through row reads too, the same discarding cycles through flat identities
/// too, and `cycle_rank@1` (none discarded). `None` past [`CYCLE_BOUND`].
#[allow(clippy::type_complexity)]
fn completion_ranks(
    n: usize,
    paint: &dyn Fn(usize) -> (u32, u32),
    pairs: &[(usize, usize, u8, u8)],
) -> Option<[std::collections::BTreeSet<Vec<usize>>; 4]> {
    let opts: Vec<Vec<u8>> = pairs
        .iter()
        .map(|(_, _, e, _)| [1u8, 2, 4].into_iter().filter(|b| e & b != 0).collect())
        .collect();
    let mut out: [std::collections::BTreeSet<Vec<usize>>; 4] = Default::default();
    if opts.iter().any(|o| o.is_empty()) {
        return Some(out);
    }
    if opts.iter().map(|o| o.len() as f64).product::<f64>() > CYCLE_BOUND {
        return None;
    }
    let cycles = |g: &[Vec<u8>], codes: &[u8]| {
        let mut left = vec![true; n];
        loop {
            let free = (0..n).find(|&j| {
                left[j]
                    && !(0..n).any(|i| left[i] && g[i][j] != 0 && codes.contains(&(g[i][j] - 1)))
            });
            match free {
                Some(j) => left[j] = false,
                None => return left.iter().any(|l| *l),
            }
        }
    };
    let mut choice = vec![0usize; pairs.len()];
    loop {
        let mut front = vec![vec![false; n]; n];
        let mut strong = vec![vec![false; n]; n];
        // A chosen strong edge, its provenance code plus 1; 0 for none.
        let mut code = vec![vec![0u8; n]; n];
        for (k, (i, j, _, prov)) in pairs.iter().enumerate() {
            let (i, j) = (*i, *j);
            match opts[k][choice[k]] {
                1 => {
                    let i_front = paint(i) > paint(j);
                    front[i][j] = i_front;
                    front[j][i] = !i_front;
                }
                2 => {
                    front[i][j] = true;
                    strong[i][j] = true;
                    code[i][j] = (prov & 3) + 1;
                }
                _ => {
                    front[j][i] = true;
                    strong[j][i] = true;
                    code[j][i] = ((prov >> 2) & 3) + 1;
                }
            }
        }
        let r = arachne_oracle::order::rank(n, &front, &strong, &paint);
        for (set, codes) in out.iter_mut().zip([&[0u8][..], &[0, 1], &[0, 2], &[]]) {
            if codes.is_empty() || !cycles(&code, codes) {
                set.insert(r.clone());
            }
        }
        let mut k = 0;
        loop {
            if k == pairs.len() {
                return Some(out);
            }
            choice[k] += 1;
            if choice[k] < opts[k].len() {
                break;
            }
            choice[k] = 0;
            k += 1;
        }
    }
}

/// The measures of frame `d`, the core's reports of its pixels `reports` read where the order
/// a sample admits needs the completion search.
fn measure(
    m: &mut Measures,
    d: &Decisions,
    excess: usize,
    tally: &Tally,
    reports: &HashMap<(u32, u32), Reported>,
) {
    use std::collections::BTreeSet;
    m.frames += 1;
    m.excess += excess;
    m.own_reads += tally.own;
    m.plane_reads += tally.plane;
    m.either_kept.0 += tally.either_own;
    m.either_kept.1 += tally.either_taken;
    let frag = |i: u32| &d.fragments[i as usize];
    let mut pairs: HashMap<(u32, u32, u8), Vec<&arachne_oracle::out::PairRow>> = HashMap::new();
    for p in &d.pairs {
        pairs.entry((p.x, p.y, p.sample)).or_default().push(p);
        *m.labels.entry(p.label).or_default() += 1;
    }
    let mut labl: HashMap<(u32, u32, u8), Vec<&arachne_oracle::out::LabelRow>> = HashMap::new();
    for r in &d.labels {
        labl.entry((r.x, r.y, r.sample)).or_default().push(r);
    }
    let mut status: HashMap<(u32, u32, u32), Vec<[u8; 16]>> = HashMap::new();
    for b in d.bands.iter().filter(|b| b.which != 2) {
        status
            .entry((b.x, b.y, b.fragment))
            .or_default()
            .push(b.status);
    }
    // The fragments `truth_unable` whatever is read with them (§6): by their own rows' statuses,
    // and an unbounded one at every sample of its box, the window, rows or none.
    let own_status: HashMap<(u32, u32, u32), [u8; 16]> = d
        .bands
        .iter()
        .filter(|b| b.which == 0)
        .map(|b| ((b.x, b.y, b.fragment), b.status))
        .collect();
    let unbounded: Vec<[i32; 4]> = d
        .fragments
        .iter()
        .filter(|f| f.flags & arachne_oracle::out::FLAG_UNBOUNDED != 0)
        .map(|f| f.bbox)
        .collect();
    let in_unbounded = |x: u32, y: u32| {
        let (x, y) = (x as i32, y as i32);
        unbounded
            .iter()
            .any(|b| (b[0]..b[2]).contains(&x) && (b[1]..b[3]).contains(&y))
    };
    let mut counted: std::collections::HashSet<(u32, u32, u8)> = Default::default();
    let mut i0 = 0;
    while i0 < d.covers.len() {
        let (x, y) = (d.covers[i0].x, d.covers[i0].y);
        let mut i1 = i0;
        while i1 < d.covers.len() && (d.covers[i1].x, d.covers[i1].y) == (x, y) {
            i1 += 1;
        }
        let rows = &d.covers[i0..i1];
        for s in 0..16u8 {
            let present: Vec<_> = rows.iter().filter(|r| r.cover[s as usize] != 0).collect();
            if present.is_empty() {
                continue;
            }
            m.samples += 1;
            counted.insert((x, y, s));
            let ps = pairs.get(&(x, y, s)).map_or(&[][..], |v| &v[..]);
            let tb = ps.iter().any(|p| p.label == 3);
            let ei = present.iter().any(|r| r.cover[s as usize] == 2);
            let po = ps.iter().any(|p| matches!(p.label, 6..=8));
            m.tie_band += tb as usize;
            m.either += ei as usize;
            m.policy_tie += po as usize;
            m.tied += (tb || ei || po) as usize;
            m.exact_tie += ps.iter().any(|p| p.label == 4) as usize;
            for p in ps {
                let labelled =
                    |f: u32| d.fragments[f as usize].label != arachne_oracle::out::NONE_U32;
                if labelled(p.a) || labelled(p.b) {
                    if p.plate == 0 {
                        m.plate.1 += 1;
                    } else {
                        m.plate.0 += 1;
                    }
                }
            }
            // The cycle lines: three or more fragments, two pairs or more with a choice.
            let ids: Vec<u32> = present.iter().map(|r| r.fragment).collect();
            if ids.len() >= 3 && ps.iter().filter(|p| p.edges.count_ones() > 1).count() >= 2 && {
                let pos = |f: u32| ids.iter().position(|&g| g == f).unwrap();
                let cp: Vec<(usize, usize, u8, u8)> = ps
                    .iter()
                    .map(|p| (pos(p.a), pos(p.b), p.edges, p.provenance))
                    .collect();
                strong_cycle_possible(ids.len(), &cp)
            } {
                let pos = |f: u32| ids.iter().position(|&g| g == f).unwrap();
                let cp: Vec<(usize, usize, u8, u8)> = ps
                    .iter()
                    .map(|p| (pos(p.a), pos(p.b), p.edges, p.provenance))
                    .collect();
                let paint = |i: usize| {
                    let f = frag(ids[i]);
                    (f.entry, f.sub)
                };
                match completion_ranks(ids.len(), &paint, &cp) {
                    Some([two, no_row, no_flat, one]) => {
                        let row = two.difference(&no_row).count();
                        let flat = two.difference(&no_flat).count();
                        let own = one.difference(&two).count();
                        for (line, k) in [
                            (&mut m.cycle_row, row),
                            (&mut m.cycle_flat, flat),
                            (&mut m.cycle_own, own),
                        ] {
                            line.0 += (k > 0) as usize;
                            line.1 += k;
                        }
                    }
                    None => m.cycle_unbounded += 1,
                }
            }
            let mut why: BTreeSet<String> = BTreeSet::new();
            // A pair `truth_unable` at the sample, by the statuses of its two fragments' reads.
            for p in ps.iter().filter(|p| p.label == 0) {
                let mut named = false;
                for f in [p.a, p.b] {
                    for st in status.get(&(x, y, f)).into_iter().flatten() {
                        if let Some(n) = ["band_unbounded", "curve_recut", "pivot_unknown"]
                            .get((st[s as usize] as usize).wrapping_sub(2))
                        {
                            why.insert(format!("truth_unable:{n}"));
                            named = true;
                        }
                    }
                }
                if !named {
                    why.insert("truth_unable:unnamed".into());
                }
            }
            // A present fragment `truth_unable` by its own key, a pair or none.
            for r in &present {
                let own = own_status.get(&(x, y, r.fragment)).map(|st| st[s as usize]);
                if let Some(n) = own.and_then(|st| match st {
                    2 => Some("band_unbounded"),
                    4 => Some("pivot_unknown"),
                    _ => None,
                }) {
                    why.insert(format!("truth_unable:{n}"));
                }
            }
            if in_unbounded(x, y) {
                why.insert("truth_unable:band_unbounded".into());
            }
            for r in &present {
                if d.colours[r.fragment as usize].kind == 0 {
                    why.insert("colour_unknown".into());
                }
            }
            for r in labl.get(&(x, y, s)).map_or(&[][..], |v| &v[..]) {
                match r.role {
                    2 => {
                        why.insert(format!(
                            "label_rule_open:pair{}:class{}",
                            r.pair_cause, r.class_cause
                        ));
                    }
                    4 => {
                        why.insert("label_alpha_straddle".into());
                    }
                    _ => {}
                }
            }
            // Past the completion bound, the judge searches only for a reported order that some
            // pair's edges do not admit (`admissible`'s sufficient test): the core's, read there.
            let completions: f64 = ps.iter().map(|p| p.edges.count_ones() as f64).product();
            if completions > 262_144.0
                && reports.get(&(x, y)).is_some_and(|r| {
                    judge::judge(d, r, s as usize) == judge::Verdict::Unable("completion_bound")
                })
            {
                why.insert("completion_bound".into());
            }
            for w in why {
                *m.open.entry(w).or_default() += 1;
            }
            // A stroke adjusted on its own fill: a fill of its group present at the sample.
            for r in &present {
                let f = frag(r.fragment);
                if f.layer != 2 || f.group == u32::MAX {
                    continue;
                }
                if let Some(fill) = present.iter().find(|q| {
                    let g = frag(q.fragment);
                    g.layer == 1
                        && g.group == f.group
                        && g.entry == f.entry
                        && g.sub / 3 == f.sub / 3
                }) {
                    if fill.cover[s as usize] == 1 && r.cover[s as usize] == 1 {
                        m.on_fill.0 += 1;
                    } else {
                        m.on_fill.1 += 1;
                    }
                }
            }
        }
        i0 = i1;
    }
    // An unbounded fragment's samples that no row reaches: present there, and unjudged.
    let frame = [0, 0, d.header.width as i32, d.header.height as i32];
    let mut beyond: std::collections::HashSet<(u32, u32, u8)> = Default::default();
    for b in &unbounded {
        for y in b[1].max(frame[1])..b[3].min(frame[3]) {
            for x in b[0].max(frame[0])..b[2].min(frame[2]) {
                for s in 0..16u8 {
                    let at = (x as u32, y as u32, s);
                    if !counted.contains(&at) {
                        beyond.insert(at);
                    }
                }
            }
        }
    }
    m.samples += beyond.len();
    if !beyond.is_empty() {
        *m.open
            .entry("truth_unable:band_unbounded".into())
            .or_default() += beyond.len();
    }
}

/// One frame's measures, decided from `b` under `v`.
fn measured(b: &scenes::Build, v: arachne_core::xf::View) -> Measures {
    let d = arachne_oracle::decide::decide_at(&[b.delta(1, v)], 1, None).unwrap();
    assert_eq!(d.header.status, 0, "{:?}", d.refused);
    let mut m = Measures::default();
    measure(&mut m, &d, 0, &Tally::default(), &HashMap::new());
    m
}

#[test]
fn a_fragment_whose_key_states_no_bound_is_counted_unjudged_with_a_pair_or_none() {
    // §6: a fragment with no key bounded at a sample is `truth_unable` there, whether or not it
    // makes a pair, and an unbounded fragment at every sample of the window, rows or none (FORMAT
    // `flags` bit 0). The open share counts each, as the judge leaves each unjudged.
    use scenes::*;
    let unbounded = |m: &Measures| m.open.get("truth_unable:band_unbounded").copied();
    // A lone glyph kept facing the camera about (0, 0, 400), under the exponential option at
    // focal distance 1: its key is finite, its error overflows.
    let mut b = Build::new(true);
    b.glyph(
        rect([-1.0, 1.0], [-1.0, 1.0], 400.0),
        OPAQUE_BLUE,
        [0.0, 0.0, 400.0],
    );
    let mut v = arachne_core::xf::View {
        camera: camera(1.0),
        input_grid: 0.0,
        ..view_c()
    };
    if let arachne_core::xf::Camera::ThreeD(c) = &mut v.camera {
        c.exponential_projection = true;
    }
    let m = measured(&b, v);
    assert!(m.samples > 0);
    assert_eq!(unbounded(&m), Some(m.samples), "{:?}", m.open);
    // A lone point placed where its key states no bound: no rows, unable over the whole frame.
    let mut b = Build::new(true);
    b.points(vec![[0.0, 0.0, 20.0 - 1e-7]], WHITE, 1.0);
    let m = measured(&b, view_c());
    assert_eq!(m.samples, 64 * 64 * 16);
    assert_eq!(unbounded(&m), Some(m.samples), "{:?}", m.open);
}

#[test]
fn the_completion_bound_is_counted_only_where_the_reported_order_needs_the_search() {
    // The judge accepts an order that every pair's edges admit before any search, and leaves a
    // sample unjudged at the completion bound only for an order that needs one (`admissible`).
    // Six squares across pixel (32, 32) on near planes at grid 0.05: two on z = 0, one
    // computation, weak alone; the rest a part in a thousand apart, every pair of them admitting
    // every edge. Fourteen free pairs exceed the bound. Reported in paint order, the sample is
    // judged; with the two on z = 0 reversed, it needs the search and is not.
    use scenes::*;
    let mut b = Build::new(true);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_RED);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_BLUE);
    for k in 1..=4 {
        b.fill(
            rect([-6.0, 6.0], [-6.0, 6.0], 0.001 * k as f64),
            OPAQUE_GREEN,
        );
    }
    let v = arachne_core::xf::View {
        input_grid: 0.05,
        ..view_c()
    };
    let mut delta = b.delta(1, v);
    let mut seen = std::collections::HashSet::new();
    delta.upload.retain(|s| seen.insert(s.key()));
    let d = arachne_oracle::decide::decide_at(&[delta], 1, None).unwrap();
    assert_eq!(d.header.status, 0, "{:?}", d.refused);
    let at: Vec<u8> = d
        .pairs
        .iter()
        .filter(|p| (p.x, p.y, p.sample) == (32, 32, 0))
        .map(|p| p.edges)
        .collect();
    assert_eq!(at.len(), 15);
    assert_eq!(at.iter().filter(|e| **e == 1).count(), 1, "{at:?}");
    assert_eq!(at.iter().filter(|e| **e == 7).count(), 14, "{at:?}");
    // Slot `k`'s fragment and its cells at (32, 32); the report front first, later painted first.
    let entries: Vec<Entry> = (0..6)
        .map(|slot| {
            let f = d.fragments.iter().find(|f| f.slot == slot).unwrap();
            let c = d
                .covers
                .iter()
                .find(|c| (c.x, c.y, c.fragment) == (32, 32, f.id))
                .unwrap();
            Entry {
                identity: (f.object, f.part, f.layer, f.sub),
                cells: c.cell_hi,
            }
        })
        .collect();
    let bound = |order: Vec<usize>| {
        let rep = Reported {
            x: 32,
            y: 32,
            entries: entries.clone(),
            order: vec![order; 16],
        };
        let mut m = Measures::default();
        measure(
            &mut m,
            &d,
            0,
            &Tally::default(),
            &HashMap::from([((32, 32), rep)]),
        );
        m.open.get("completion_bound").copied().unwrap_or(0)
    };
    assert_eq!(bound(vec![5, 4, 3, 2, 1, 0]), 0);
    assert_eq!(bound(vec![5, 4, 3, 2, 0, 1]), 16);
}

/// The class a stream at `rel` under `root` is measured in. A smoke case's directory
/// (`fuzz__<family>__<seed>`) holds `stream.xf`, the canonical stream the core is handed, class
/// `smoke:<family>`, and `made.xf`, the painter's unsnapped original, a real input but not a
/// production one: its own family, `smoke-made:<family>`, never summed with the other. Any other
/// stream is classed by its root's directory.
fn stream_class(root: &std::path::Path, rel: &str) -> String {
    match rel.split("__").nth(1) {
        Some(family) if rel.ends_with("/made.xf") => format!("smoke-made:{family}"),
        Some(family) => format!("smoke:{family}"),
        None => root
            .file_name()
            .map_or("streams".into(), |n| n.to_string_lossy().to_string()),
    }
}

/// The measures `soundness.md` records, by case class (`stream_class`): `core_err_within_band`'s
/// excess and read counts, `tie_share`, the open share by reason, and the pair labels, over the
/// probes and every `.xf` file under the directories `SOUNDNESS_STREAMS` names (colon-separated;
/// with `SOUNDNESS_NO_PROBES` set, those alone), a smoke case's `stream.xf` and `made.xf` each in
/// its own family. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn soundness_measures() {
    fn streams(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().collect();
        entries.sort_by_key(|e| e.path());
        for e in entries {
            let p = e.path();
            if p.is_dir() {
                streams(&p, out);
            } else if p.extension().is_some_and(|x| x == "xf") {
                out.push(p);
            }
        }
    }
    let dir = fixtures::scratch("soundness");
    // `SOUNDNESS_NO_PROBES` set: the streams alone (for a run split over processes).
    let probes = if std::env::var_os("SOUNDNESS_NO_PROBES").is_some() {
        Vec::new()
    } else {
        fixtures::probe_files(&dir)
    };
    let mut inputs: Vec<(String, String, std::path::PathBuf)> = probes
        .into_iter()
        .map(|p| (format!("probe:{}", p.name), p.name.to_string(), p.path))
        .collect();
    for root in std::env::var("SOUNDNESS_STREAMS")
        .unwrap_or_default()
        .split(':')
    {
        if root.is_empty() {
            continue;
        }
        let mut found = Vec::new();
        streams(std::path::Path::new(root), &mut found);
        for p in found {
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().to_string();
            let class = stream_class(std::path::Path::new(root), &rel);
            inputs.push((class, rel, p));
        }
    }
    let mut by_class: BTreeMap<String, Measures> = BTreeMap::new();
    for (class, name, path) in &inputs {
        let (_, wire) = source::read_file(path).unwrap();
        let deltas = arachne_core::xf::wire::decode_file(&std::fs::read(path).unwrap()).unwrap();
        let mut core = Stream::new();
        for (wd, delta) in wire.iter().zip(&deltas) {
            core.apply(delta).unwrap();
            let d = decide_at_wire(&wire, wd.seq, None).unwrap();
            if d.header.status != 0 {
                println!("refused {class} {name} seq {}: {:?}", wd.seq, d.refused);
                continue;
            }
            let pixels = frame_trace(&core, &d);
            let (e, tally) = excesses_of("measure", wd.seq, &d, &pixels);
            let reports = core_reports(&pixels);
            let m = by_class.entry(class.clone()).or_default();
            measure(m, &d, e.len(), &tally, &reports);
            let mut frame = Measures::default();
            measure(&mut frame, &d, e.len(), &tally, &reports);
            let mut kinds: BTreeMap<(u8, u8, &str, &str), usize> = BTreeMap::new();
            for x in &e {
                *kinds.entry((x.layer, x.kind, x.row, x.what)).or_default() += 1;
            }
            println!(
                "frame {class} {name} seq {} samples {} open {:?} excess {kinds:?}",
                wd.seq, frame.samples, frame.open
            );
            for x in e.iter().take(3) {
                println!("  excess {x:?}");
            }
            if let Ok(out) = std::env::var("SOUNDNESS_EXCESS") {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(out)
                    .unwrap();
                for x in &e {
                    let fr = &d.fragments[x.fragment as usize];
                    writeln!(
                        f,
                        "{}	{}	{}	{}	{}	{}	{}	{}	{}	{}	{:e}	{:e}	{:e}	{:e}",
                        path.display(),
                        wd.seq,
                        x.at.0,
                        x.at.1,
                        x.at.2,
                        x.fragment,
                        fr.entry,
                        fr.sub,
                        x.layer,
                        x.what,
                        x.core.0,
                        x.core.1,
                        x.oracle.0,
                        x.oracle.1
                    )
                    .unwrap();
                }
            }
        }
    }
    for (class, m) in &by_class {
        println!("class {class} {m:?}");
    }
}

#[test]
fn a_smoke_case_s_made_stream_is_its_own_family_beside_the_stream_the_core_is_handed() {
    // A smoke case holds `stream.xf`, the canonical stream the core is handed, and `made.xf`, the
    // painter's unsnapped original: each is measured under its own class, never summed.
    let root = std::path::Path::new("/runs/smoke");
    assert_eq!(
        stream_class(root, "fuzz__text__2924964735/stream.xf"),
        "smoke:text"
    );
    assert_eq!(
        stream_class(root, "fuzz__text__2924964735/made.xf"),
        "smoke-made:text"
    );
    // Any other stream set is classed by its directory, whatever its files' names.
    let root = std::path::Path::new("/runs/streams-551");
    assert_eq!(stream_class(root, "0a1b-3.xf"), "streams-551");
}
