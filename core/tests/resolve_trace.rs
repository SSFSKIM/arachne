//! The diagnostic resolve trace (`Stream::diagnostic_resolves`): re-composited with the resolves'
//! own arithmetic, each pixel's trace gives back the pixel the core drew, over every pixel of
//! every frame of the parity kit's fixtures; asked for, its pairwise orders are those its ranks
//! ranked by.

use std::cmp::Ordering;

use arachne_core::composite::{Resolve, SAMPLES, add_emission, blend};
use arachne_core::diagnostics::{PairOrder, Pixel};
use arachne_core::order::{self, Decided, Ranked};
use arachne_core::probes;
use arachne_core::stream::Stream;

/// The pixel a trace composites to: the units back to front for the fast and coverage resolves;
/// each sub-sample's entries back to front at their coverage there, then the sub-samples' mean
/// for the sample resolve; the emitted light last.
fn recomposite(p: &Pixel) -> [u8; 4] {
    let mut px = match p.resolve {
        Resolve::Background => p.background,
        Resolve::Fast | Resolve::Coverage => p
            .units
            .iter()
            .fold(p.background, |px, u| blend(u.colour, u.coverage, px)),
        Resolve::Sample => {
            let mut sum = [0u32; 4];
            for k in 0..SAMPLES {
                let px = p.samples[k].iter().rev().fold(p.background, |px, &i| {
                    let e = &p.entries[i];
                    blend(e.colour, e.coverage[k], px)
                });
                for (s, c) in sum.iter_mut().zip(px) {
                    *s += c as u32;
                }
            }
            let n = SAMPLES as u32;
            sum.map(|v| ((v + n / 2) / n) as u8)
        }
    };
    if let Some(e) = p.emission {
        px = add_emission(px, e);
    }
    px
}

#[test]
fn every_pixel_s_resolve_trace_recomposites_to_the_pixel_drawn() {
    let mut counts = [0usize; 4];
    let mut wrong = Vec::new();
    for probe in probes::all() {
        let (w, h) = (probe.width, probe.height);
        let mut stream = Stream::new();
        for (frame, delta) in probe.deltas.iter().enumerate() {
            stream.apply(delta).unwrap();
            let drawn = stream.render_under_overlay(w, h, 2).unwrap();
            let traces = stream.diagnostic_resolves(0, 0, w, h, false).unwrap();
            assert_eq!(
                traces.len(),
                (w * h) as usize,
                "{} frame {frame}",
                probe.name
            );
            for ((x, y), p) in &traces {
                let i = (y * w + x) as usize;
                assert_eq!(
                    p.resolve as u8, drawn.resolves[i],
                    "{} ({x}, {y})",
                    probe.name
                );
                counts[p.resolve as usize] += 1;
                let got = recomposite(p);
                if got[..] != drawn.rgba[i * 4..i * 4 + 4] {
                    wrong.push((probe.name, frame, *x, *y, p.resolve, got));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} pixels re-composite to other bytes, first {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(5)]
    );
    eprintln!("pixels by resolve (background, fast, coverage, sample): {counts:?}");
    // Both resolves that composite several layers are exercised, and the fast path.
    assert!(
        counts[Resolve::Coverage as usize] > 1000 && counts[Resolve::Sample as usize] > 100,
        "{counts:?}"
    );
    assert!(counts[Resolve::Fast as usize] > 0, "{counts:?}");
}

#[test]
fn a_trace_names_each_entry_s_leaf_and_layer() {
    // The paint kinds fixture draws strokes and fills: its traces name both layers.
    let probe = probes::paint_kinds();
    let mut stream = Stream::new();
    for delta in &probe.deltas {
        stream.apply(delta).unwrap();
    }
    let traces = stream
        .diagnostic_resolves(0, 0, probe.width, probe.height, false)
        .unwrap();
    let layers: std::collections::BTreeSet<_> = traces
        .values()
        .flat_map(|p| p.entries.iter().map(|e| format!("{:?}", e.layer)))
        .collect();
    assert!(layers.contains("Stroke"), "{layers:?}");
    for p in traces.values() {
        for e in &p.entries {
            assert!(e.leaf.0 != 0, "an entry without its object");
        }
        for k in 0..SAMPLES {
            for &i in &p.samples[k] {
                assert!(
                    p.entries[i].coverage[k] != 0,
                    "a listed entry covers its sub-sample"
                );
            }
        }
    }
}

/// `items` ranked from `pairs` alone, as the pixel's rank or a sub-sample's ranks them: the
/// reported front, depth (or a majority of sub-samples) strong, paint order the later drawn
/// (`order`, then `sub`) in front.
fn rank_from(
    p: &Pixel,
    items: &[usize],
    pairs: &[PairOrder],
    sub_sample: bool,
) -> (Ranked, Vec<usize>) {
    let find = |a: usize, b: usize| {
        pairs
            .iter()
            .find(|q| (q.a, q.b) == (a.min(b), a.max(b)))
            .unwrap_or_else(|| panic!("no pair ({a}, {b}) reported"))
    };
    let front = |a: usize, b: usize| a != b && find(a, b).front == Some(a);
    let strong = |a: usize, b: usize| a != b && find(a, b).how.is_strong();
    let painted = |a: usize, b: usize| -> Ordering {
        let (ea, eb) = (&p.entries[a], &p.entries[b]);
        (eb.order, eb.sub).cmp(&(ea.order, ea.sub))
    };
    let mut out = Vec::new();
    let how = if sub_sample {
        order::rank_sample(items, front, strong, painted, &mut out)
    } else {
        order::rank(items, front, strong, painted, &mut out)
    };
    (how, out)
}

/// Every pair of `items`, `a < b`, sorted.
fn all_pairs(items: &[usize]) -> Vec<(usize, usize)> {
    let mut v: Vec<_> = items
        .iter()
        .enumerate()
        .flat_map(|(p, &i)| items[p + 1..].iter().map(move |&j| (i.min(j), i.max(j))))
        .collect();
    v.sort();
    v
}

#[test]
fn a_trace_s_pairwise_orders_are_those_its_ranks_ranked_by() {
    let mut kinds = std::collections::BTreeMap::new();
    let (mut reranked, mut pixels) = (0usize, 0usize);
    for probe in probes::all() {
        let (w, h) = (probe.width, probe.height);
        let mut stream = Stream::new();
        for (frame, delta) in probe.deltas.iter().enumerate() {
            stream.apply(delta).unwrap();
            let plain = stream.diagnostic_resolves(0, 0, w, h, false).unwrap();
            let traces = stream.diagnostic_resolves(0, 0, w, h, true).unwrap();
            assert_eq!(plain.len(), traces.len());
            for (((x, y), p), q) in traces.iter().zip(plain.values()) {
                let at = format!("{} frame {frame} ({x}, {y})", probe.name);
                // Not asked for, nothing is recorded; asked for, nothing else moves.
                assert!(q.pairs.is_empty() && q.sample_pairs.is_empty() && !q.sample_ranked);
                let mut bare = p.clone();
                (bare.pairs, bare.sample_pairs, bare.sample_ranked) = (vec![], vec![], false);
                assert_eq!(&bare, q, "{at}");
                let n = p.entries.len();
                if n < 2 {
                    assert!(p.pairs.is_empty() && p.sample_pairs.is_empty(), "{at}");
                    continue;
                }
                pixels += 1;
                // The pixel's rank: every pair; ranked from them alone, an order that keeps the
                // units it composited.
                let got: Vec<_> = p.pairs.iter().map(|q| (q.a, q.b)).collect();
                let entries: Vec<usize> = (0..n).collect();
                assert_eq!(got, all_pairs(&entries), "{at}");
                for q in &p.pairs {
                    *kinds.entry(("pixel", format!("{:?}", q.how))).or_insert(0) += 1;
                }
                let (how, out) = rank_from(p, &entries, &p.pairs, false);
                if how == Ranked::Strong {
                    assert_eq!(p.resolve, Resolve::Sample, "{at}");
                } else if p.resolve == Resolve::Coverage {
                    let shown: Vec<usize> = p
                        .units
                        .iter()
                        .rev()
                        .flat_map(|u| u.members.clone())
                        .collect();
                    let mut rest = out.iter();
                    assert!(
                        shown.iter().all(|i| rest.any(|j| j == i)),
                        "{at}: units {shown:?} out of the rank {out:?}"
                    );
                }
                // Each sub-sample: every pair of what it ranked or composited; where it ranked
                // what it composited, ranked from them alone, its order.
                assert_eq!(p.sample_pairs.len(), SAMPLES, "{at}");
                for k in 0..SAMPLES {
                    let pairs = &p.sample_pairs[k];
                    for q in pairs {
                        assert!(
                            matches!(q.how, Decided::Depth | Decided::Tie | Decided::Paint),
                            "{at}"
                        );
                        assert!(
                            q.front.is_some(),
                            "{at}: a sub-sample order without a front"
                        );
                        let kind = format!("{:?}", q.how);
                        *kinds.entry(("sub-sample", kind)).or_insert(0) += 1;
                    }
                    if p.sample_ranked && p.resolve != Resolve::Sample {
                        continue;
                    }
                    let items = &p.samples[k];
                    let got: Vec<_> = pairs.iter().map(|q| (q.a, q.b)).collect();
                    assert_eq!(got, all_pairs(items), "{at} sub-sample {k}");
                    if p.sample_ranked {
                        reranked += 1;
                        let (_, out) = rank_from(p, items, pairs, true);
                        assert_eq!(&out, items, "{at} sub-sample {k}");
                    }
                }
            }
        }
    }
    eprintln!("pixels of several entries {pixels}, sub-samples re-ranked {reranked}");
    eprintln!("pairs by kind {kinds:?}");
    assert!(reranked > 100, "{reranked}");
    for kind in ["Depth", "Tie", "Paint"] {
        assert!(
            kinds.contains_key(&("sub-sample", kind.to_string())),
            "{kinds:?}"
        );
    }
}
