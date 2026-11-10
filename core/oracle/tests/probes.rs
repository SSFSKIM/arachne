//! The class probes (the parity kit's), judged by the oracle from their XF bytes alone (contract
//! §7's admissible orders): at each check, the fragment the probe's bytes name is in front in an
//! admissible order at every sample where it is present, and in every admissible order outside the
//! checks the contract's bands leave tied; a hidden slot is in front in no admissible order.
//!
//! A check's bytes name a fragment by its record's solid colour, composited over what lies under
//! it. A check whose bytes are an overlay record's is outside the oracle's decisions: Cairo draws
//! the overlay over the frame (contract §1).

#[path = "common/fixtures.rs"]
mod fixtures;

use std::collections::HashMap;

use arachne_core::xf::{Background, Paint, Record};
use arachne_oracle::decide::decide_at_wire;
use arachne_oracle::order::{Edges, admissible};
use arachne_oracle::source::{self, WireDelta};
use arachne_oracle::stream::{State, is_overlay};
use fixtures::Expect;

/// The checks where the contract's bands tie the fragment the bytes name with another, so the
/// oracle admits both orders there: (probe, check). Each is listed so that a change shows.
const TIED: &[(&str, &str)] = &[
    (
        "edge_on_surface",
        "the ring where its near arc is behind the wall",
    ),
    (
        "edge_on_surface",
        "the ring's near arc in front of the wall, on the far arc's row (CE right)",
    ),
];

fn rgba_of(r: &Record, layer: u8) -> Option<[f64; 4]> {
    match if matches!(layer, 0 | 2) {
        &r.paint.stroke
    } else {
        &r.paint.fill
    } {
        Paint::One(c) => Some(*c),
        _ => None,
    }
}

fn over(top: [f64; 4], under: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| top[3] * top[k] + (1.0 - top[3]) * under[k])
}

fn close(c: [f64; 3], b: [u8; 4]) -> bool {
    (0..3).all(|k| (c[k] * 255.0 - b[k] as f64).abs() <= 2.0)
}

fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for (i, first) in items.iter().enumerate() {
        let rest: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| *v)
            .collect();
        for mut p in permutations(&rest) {
            p.insert(0, *first);
            out.push(p);
        }
    }
    out
}

/// The state after the delta `seq`.
fn state_at(wire: &[WireDelta], seq: u64) -> State {
    let mut state = State::new();
    for d in wire {
        let _ = state.apply_wire(d);
        if d.seq == seq {
            break;
        }
    }
    state
}

/// Per sample of the check's pixel: whether `front` is first in some admissible order, and in
/// every one.
struct Front {
    some: bool,
    every: bool,
}

#[test]
fn every_class_probe_check_holds_in_the_oracle_s_admissible_orders() {
    let dir = fixtures::scratch("probes");
    let mut judged = 0;
    let mut tied_seen = Vec::new();
    let mut overlays = 0;
    for p in fixtures::probe_files(&dir) {
        let (_, wire) = source::read_file(&p.path).unwrap();
        for c in &p.checks {
            let state = state_at(&wire, c.seq);
            let d = decide_at_wire(&wire, c.seq, Some([c.x, c.y, c.x + 1, c.y + 1])).unwrap();
            assert_eq!(d.header.status, 0, "{} {}: {:?}", p.name, c.what, d.refused);
            let frag = |id: u32| &d.fragments[id as usize];
            let paint = |id: usize| (frag(id as u32).entry, frag(id as u32).sub);
            let cover = |id: u32, s: usize| {
                d.covers
                    .iter()
                    .find(|r| r.fragment == id)
                    .map_or(0, |r| r.cover[s])
            };
            let edges: HashMap<(u32, u32, u8), u8> = d
                .pairs
                .iter()
                .map(|r| ((r.a, r.b, r.sample), r.edges))
                .collect();
            // Whether `who` is first in some and in every admissible order at sample `s`.
            let judge = |who: u32, s: usize| -> Front {
                let present: Vec<usize> = d
                    .fragments
                    .iter()
                    .filter(|f| cover(f.id, s) != 0)
                    .map(|f| f.id as usize)
                    .collect();
                let e = |i: usize, j: usize| {
                    let bits = edges[&(present[i] as u32, present[j] as u32, s as u8)];
                    Edges {
                        weak: bits & 1 != 0,
                        a_front: bits & 2 != 0,
                        b_front: bits & 4 != 0,
                    }
                };
                let mut f = Front {
                    some: false,
                    every: true,
                };
                for order in permutations(&present) {
                    if admissible(&present, &paint, &e, &order) != Some(true) {
                        continue;
                    }
                    if order[0] == who as usize {
                        f.some = true;
                    } else {
                        f.every = false;
                    }
                }
                f
            };
            match &c.expect {
                Expect::Hidden(slot) => {
                    for f in d.fragments.iter().filter(|f| f.slot == *slot) {
                        for s in (0..16).filter(|s| cover(f.id, *s) != 0) {
                            assert!(
                                !judge(f.id, s).some,
                                "{} {}: slot {slot} admissibly in front at sample {s}",
                                p.name,
                                c.what
                            );
                        }
                    }
                    judged += 1;
                }
                Expect::Bytes(b) => {
                    let bg = match &state.view.as_ref().unwrap().background {
                        Background::Color(c) => [c[0], c[1], c[2]],
                        other => panic!("{other:?}"),
                    };
                    // An overlay record whose colour the bytes are: outside the oracle's frame.
                    let overlay = state.records.values().any(|r| {
                        is_overlay(r) && rgba_of(r, 1).is_some_and(|c| close(over(c, bg), *b))
                    });
                    let colour = |id: u32| {
                        state
                            .records
                            .get(&frag(id).slot)
                            .and_then(|r| rgba_of(r, frag(id).layer))
                    };
                    let named: Vec<u32> = d
                        .fragments
                        .iter()
                        .filter(|f| {
                            let Some(top) = colour(f.id) else {
                                return false;
                            };
                            close(over(top, bg), *b)
                                || d.fragments.iter().any(|g| {
                                    g.id != f.id
                                        && colour(g.id)
                                            .is_some_and(|u| close(over(top, over(u, bg)), *b))
                                })
                        })
                        .map(|f| f.id)
                        .collect();
                    if named.is_empty() && overlay {
                        overlays += 1;
                        continue;
                    }
                    assert!(
                        !named.is_empty(),
                        "{} {}: no fragment's colour gives {b:?}",
                        p.name,
                        c.what
                    );
                    let tied = TIED.contains(&(p.name, c.what));
                    let mut every = true;
                    for s in 0..16 {
                        let present: Vec<u32> = named
                            .iter()
                            .copied()
                            .filter(|f| cover(*f, s) != 0)
                            .collect();
                        if present.is_empty() {
                            continue;
                        }
                        let fronts: Vec<Front> = present.iter().map(|f| judge(*f, s)).collect();
                        assert!(
                            fronts.iter().any(|f| f.some),
                            "{} {}: the named fragment is in front in no admissible order at \
                             sample {s}",
                            p.name,
                            c.what
                        );
                        every &= fronts.iter().any(|f| f.every);
                    }
                    if tied {
                        assert!(!every, "{} {}: listed as tied, now decided", p.name, c.what);
                        tied_seen.push((p.name, c.what));
                    } else {
                        assert!(every, "{} {}: tied", p.name, c.what);
                    }
                    judged += 1;
                }
            }
        }
    }
    assert_eq!(tied_seen.len(), TIED.len());
    assert_eq!(overlays, 1, "the fixed overlay label");
    assert!(judged >= 20, "{judged}");
}

#[test]
fn the_seam_probe_s_interior_has_no_gap_at_its_internal_edges() {
    // Wherever a pixel and its eight neighbours hold some fragment of the sheet `Yes`, every
    // sample of the pixel holds the sheet: `Yes`, or `Either` on at least two triangles, the two
    // sides of an internal edge within the mesh margin. The background never shows through.
    let dir = fixtures::scratch("seam");
    let p = fixtures::probe_files(&dir)
        .into_iter()
        .find(|p| p.name == "seam")
        .unwrap();
    let (_, wire) = source::read_file(&p.path).unwrap();
    for d in &wire {
        let dec = decide_at_wire(&wire, d.seq, None).unwrap();
        assert_eq!(dec.header.status, 0);
        let mut at: HashMap<(u32, u32), Vec<[u8; 16]>> = HashMap::new();
        for r in &dec.covers {
            at.entry((r.x, r.y)).or_default().push(r.cover);
        }
        let solid = |x: u32, y: u32| {
            at.get(&(x, y))
                .is_some_and(|rows| rows.iter().any(|c| c.contains(&1)))
        };
        let mut interior = 0;
        let mut edge_samples = 0;
        for (&(x, y), rows) in &at {
            let inside =
                x > 0 && y > 0 && (0..3).all(|i| (0..3).all(|j| solid(x + i - 1, y + j - 1)));
            if !inside {
                continue;
            }
            interior += 1;
            for s in 0..16 {
                let yes = rows.iter().any(|c| c[s] == 1);
                let either = rows.iter().filter(|c| c[s] == 2).count();
                assert!(
                    yes || either >= 2,
                    "seq {}: ({x}, {y}) sample {s} may show the background",
                    d.seq
                );
                edge_samples += (!yes) as usize;
            }
        }
        assert!(interior > 10_000, "{interior}");
        assert!(edge_samples > 0);
    }
}
