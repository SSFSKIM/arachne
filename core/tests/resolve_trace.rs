//! The diagnostic resolve trace (`Stream::diagnostic_resolves`): re-composited with the resolves'
//! own arithmetic, each pixel's trace gives back the pixel the core drew, over every pixel of
//! every frame of the parity kit's fixtures; asked for, its pairwise orders are those its ranks
//! ranked by. With its reads (`Stream::resolve_trace`), nothing drawn and nothing else the trace
//! holds changes, and what the reads say is what the resolves read.

use std::cmp::Ordering;

use arachne_core::composite::{Resolve, SAMPLES, add_emission, blend};
use arachne_core::diagnostics::{Layer, PairOrder, Pixel, Reads};
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
                assert!(p.reads.is_none() && q.reads.is_none(), "{at}");
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

/// Each frame of every fixture, the stream holding it: `visit(name, frame, stream, width,
/// height)`.
fn each_frame(mut visit: impl FnMut(&str, usize, &Stream, u32, u32)) {
    for probe in probes::all() {
        let mut stream = Stream::new();
        for (frame, delta) in probe.deltas.iter().enumerate() {
            stream.apply(delta).unwrap();
            visit(probe.name, frame, &stream, probe.width, probe.height);
        }
    }
}

/// A pixel's reads, which `Stream::resolve_trace` always gives.
fn reads<'p>(p: &'p Pixel, at: &str) -> &'p Reads {
    p.reads
        .as_ref()
        .unwrap_or_else(|| panic!("{at}: a trace with its reads holds none"))
}

#[test]
fn the_reads_change_no_byte_drawn_and_nothing_else_the_trace_holds() {
    each_frame(|name, frame, stream, w, h| {
        let at = format!("{name} frame {frame}");
        let whole = stream.render(w, h, 2).unwrap();
        let under = stream.render_under_overlay(w, h, 2).unwrap();
        for decisions in [false, true] {
            let plain = stream.diagnostic_resolves(0, 0, w, h, decisions).unwrap();
            let read = stream.resolve_trace(0, 0, w, h, decisions).unwrap();
            assert_eq!(
                plain.keys().collect::<Vec<_>>(),
                read.keys().collect::<Vec<_>>(),
                "{at}"
            );
            for ((xy, p), q) in plain.iter().zip(read.values()) {
                let at = format!("{at} {xy:?} decisions {decisions}");
                assert!(p.reads.is_none(), "{at}: reads not asked for");
                let r = reads(q, &at);
                let n = q.entries.len();
                assert_eq!(
                    (r.coverage8.len(), r.depths.len(), r.err.len()),
                    (n, n, n),
                    "{at}"
                );
                assert_eq!(r.samples_all.len(), SAMPLES, "{at}");
                if !decisions {
                    assert!(r.against.is_empty(), "{at}: against without the decisions");
                }
                let mut bare = q.clone();
                bare.reads = None;
                assert!(bare == *p, "{at}: the reads moved what the trace holds");
            }
        }
        // Drawn again after both traces, every byte and resolve class is the same.
        assert!(stream.render(w, h, 2).unwrap() == whole, "{at}: render");
        assert!(
            stream.render_under_overlay(w, h, 2).unwrap() == under,
            "{at}: render_under_overlay"
        );
    });
}

#[test]
fn an_entry_s_coverage8_is_the_byte_the_fast_and_coverage_resolves_blend_it_alone_with() {
    let mut units = [0usize; 4];
    each_frame(|name, frame, stream, w, h| {
        for ((x, y), p) in stream.resolve_trace(0, 0, w, h, false).unwrap() {
            let at = format!("{name} frame {frame} ({x}, {y})");
            let r = reads(&p, &at);
            if !matches!(p.resolve, Resolve::Fast | Resolve::Coverage) {
                continue;
            }
            for u in &p.units {
                if let &[m] = u.members.as_slice() {
                    assert_eq!(u.coverage, r.coverage8[m], "{at} entry {m}");
                    units[p.resolve as usize] += 1;
                }
            }
        }
    });
    eprintln!("single-entry units checked (background, fast, coverage, sample): {units:?}");
    assert!(
        units[Resolve::Fast as usize] > 0 && units[Resolve::Coverage as usize] > 1000,
        "{units:?}"
    );
}

#[test]
fn samples_all_holds_every_entry_present_and_keeps_every_order_depth_decided() {
    let (mut hidden, mut depth_orders, mut against) = (0usize, 0usize, 0usize);
    let mut composited_under_cover = 0usize;
    each_frame(|name, frame, stream, w, h| {
        for ((x, y), p) in stream.resolve_trace(0, 0, w, h, true).unwrap() {
            let at = format!("{name} frame {frame} ({x}, {y})");
            let r = reads(&p, &at);
            against += r.against.len();
            for a in &r.against {
                assert!(a.entry != a.other, "{at}: an entry read against itself");
            }
            for k in 0..SAMPLES {
                let all = &r.samples_all[k];
                let present: Vec<usize> = (0..p.entries.len())
                    .filter(|&i| p.entries[i].coverage[k] != 0)
                    .collect();
                let mut listed = all.clone();
                listed.sort_unstable();
                assert_eq!(listed, present, "{at} sub-sample {k}: every entry present");
                assert!(
                    p.samples[k].iter().all(|i| all.contains(i)),
                    "{at} sub-sample {k}: composited {:?}, present {all:?}",
                    p.samples[k]
                );
                // What `samples` lacks the opaque drop removed: each lies behind an entry opaque and
                // covering the sub-sample whole. The sample resolve's own ranks may still hold such
                // an entry (they composite it under its cover, exactly).
                let covered = |i: usize| {
                    let e = &p.entries[i];
                    e.colour[3] == 255 && e.coverage[k] == 255
                };
                for (place, &i) in all.iter().enumerate() {
                    let behind_cover = all[..place].iter().any(|&j| covered(j));
                    if !p.samples[k].contains(&i) {
                        assert!(
                            behind_cover,
                            "{at} sub-sample {k}: entry {i} dropped with no opaque cover in front, \
                             ranked {all:?}, composited {:?}",
                            p.samples[k]
                        );
                        hidden += 1;
                    } else if behind_cover && p.resolve == Resolve::Sample {
                        composited_under_cover += 1;
                    }
                }
            }
            // Each sub-sample order depth decided (the resolve's own, `sample_pairs`) holds in
            // `samples_all`, and between the depths each entry is read at against the other: the
            // nearer, beyond both bounds, is in front.
            for (k, pairs) in p.sample_pairs.iter().enumerate() {
                for q in pairs.iter().filter(|q| q.how == Decided::Depth) {
                    let front = q.front.expect("a depth order has a front");
                    let back = if front == q.a { q.b } else { q.a };
                    let all = &r.samples_all[k];
                    let place = |i: usize| all.iter().position(|j| *j == i);
                    assert!(
                        matches!((place(front), place(back)), (Some(f), Some(b)) if f < b),
                        "{at} sub-sample {k}: {front} in front of {back} by depth, ranked {all:?}"
                    );
                    let read = |i: usize, other: usize| {
                        r.against
                            .iter()
                            .find(|a| (a.entry, a.other) == (i, other))
                            .map_or((r.depths[i][k], r.err[i]), |a| (a.depths[k], a.err))
                    };
                    let ((df, ef), (db, eb)) = (read(front, back), read(back, front));
                    assert!(
                        df.total_cmp(&db) == Ordering::Less && !order::tied(df, ef, db, eb),
                        "{at} sub-sample {k}: {front} at {df} (within {ef}) in front of {back} \
                         at {db} (within {eb})"
                    );
                    depth_orders += 1;
                }
            }
        }
    });
    eprintln!(
        "entries dropped from what was composited {hidden}, composited by the sample resolve \
         under an opaque cover {composited_under_cover}, depth orders checked {depth_orders}, \
         reads against another entry {against}"
    );
    assert!(hidden > 0 && depth_orders > 100, "{hidden} {depth_orders}");
}

#[test]
fn a_stream_traced_with_its_reads_and_decisions_draws_every_frame_as_one_never_traced() {
    // Two streams handed the same deltas, one traced over the whole frame with its reads and
    // decisions before each draw: every frame's bytes and resolve classes the same, with the
    // overlay and without.
    let mut frames = 0usize;
    for probe in probes::all() {
        let (w, h) = (probe.width, probe.height);
        let (mut untraced, mut traced) = (Stream::new(), Stream::new());
        for (frame, delta) in probe.deltas.iter().enumerate() {
            untraced.apply(delta).unwrap();
            traced.apply(delta).unwrap();
            let read = traced.resolve_trace(0, 0, w, h, true).unwrap();
            assert_eq!(read.len(), (w * h) as usize);
            let at = format!("{} frame {frame}", probe.name);
            assert!(
                traced.render_under_overlay(w, h, 2).unwrap()
                    == untraced.render_under_overlay(w, h, 2).unwrap(),
                "{at}: without the overlay"
            );
            assert!(
                traced.render(w, h, 2).unwrap() == untraced.render(w, h, 2).unwrap(),
                "{at}: with the overlay"
            );
            frames += 1;
        }
    }
    assert!(frames >= 18, "{frames}");
}

/// `order::painted` of entries `a` and `b`, as a pair's front: the later drawn, `None` for one
/// paint position.
fn painted(p: &Pixel, a: usize, b: usize) -> Option<usize> {
    let (ea, eb) = (&p.entries[a], &p.entries[b]);
    match eb.order.cmp(&ea.order).then(eb.sub.cmp(&ea.sub)) {
        Ordering::Less => Some(a),
        Ordering::Greater => Some(b),
        Ordering::Equal => None,
    }
}

/// `order::own_layers` of entries `a` and `b`, from what the trace says of them: one path's fill
/// and one of its own strokes (both path layers of one record and one path, a fill among them).
fn own_layers(p: &Pixel, a: usize, b: usize) -> bool {
    let path = |l: Layer| matches!(l, Layer::BackgroundStroke | Layer::Fill | Layer::Stroke);
    let (ea, eb) = (&p.entries[a], &p.entries[b]);
    path(ea.layer)
        && path(eb.layer)
        && ea.leaf == eb.leaf
        && ea.order == eb.order
        && ea.sub / 3 == eb.sub / 3
        && ea.sub != eb.sub
        && (ea.layer == Layer::Fill || eb.layer == Layer::Fill)
}

/// The pixel's order of entries `a < b` and how it was decided, from their relation alone (none
/// where they share no sub-sample), as `order::Shared::decided` decides it.
fn decided(p: &Pixel, a: usize, b: usize, s: Option<order::Shared>) -> (Option<usize>, Decided) {
    let s = s.unwrap_or_default();
    let (shared, an, bn) = (
        s.covered.count_ones(),
        s.a_nearer.count_ones(),
        s.b_nearer.count_ones(),
    );
    let paint = painted(p, a, b);
    if shared > 0 && own_layers(p, a, b) {
        return (paint, Decided::Paint);
    }
    match (shared, an, bn) {
        (0, _, _) => (paint, Decided::Paint),
        (_, 0, 0) => (paint, Decided::Tie),
        (_, _, 0) => (Some(a), Decided::Depth),
        (_, 0, _) => (Some(b), Decided::Depth),
        _ => (
            match an.cmp(&bn) {
                Ordering::Greater => Some(a),
                Ordering::Less => Some(b),
                Ordering::Equal => paint,
            },
            Decided::Majority,
        ),
    }
}

#[test]
fn the_relations_are_the_orders_the_pairs_are_decided_from_and_flat_is_the_flat_path() {
    let (mut alone, mut even, mut flat, mut ranked) = (0usize, 0usize, 0usize, 0usize);
    let mut by_how = [0usize; 4];
    each_frame(|name, frame, stream, w, h| {
        let read = stream.resolve_trace(0, 0, w, h, true).unwrap();
        let apart = stream.resolve_trace(0, 0, w, h, false).unwrap();
        for ((xy, p), q) in read.iter().zip(apart.values()) {
            let at = format!("{name} frame {frame} {xy:?}");
            let r = reads(p, &at);
            // Asked for without the decisions: none of it.
            let r0 = reads(q, &at);
            assert!(
                !r0.flat && r0.relations.is_empty(),
                "{at}: relations without the decisions"
            );
            let n = p.entries.len();
            let both = |a: usize, b: usize| {
                (0..SAMPLES).fold(0u16, |m, k| {
                    let e = &p.entries;
                    m | (((e[a].coverage[k] != 0 && e[b].coverage[k] != 0) as u16) << k)
                })
            };
            let pairs: Vec<(usize, usize)> = r.relations.iter().map(|q| (q.a, q.b)).collect();
            let mut sorted = pairs.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(pairs, sorted, "{at}: relations a < b, in order, once each");
            for q in &r.relations {
                assert!(q.a < q.b && q.b < n, "{at}");
                assert_eq!(
                    q.shared.covered,
                    both(q.a, q.b),
                    "{at}: covered ({}, {})",
                    q.a,
                    q.b
                );
            }
            if r.flat {
                flat += 1;
                assert!(r.relations.is_empty(), "{at}: a flat pixel's relations");
                assert!(
                    r.depths.windows(2).all(|d| d[0] == d[1]),
                    "{at}: flat, its entries at more than one depth"
                );
                for o in &p.pairs {
                    assert_eq!(
                        (o.front, o.how),
                        (painted(p, o.a, o.b), Decided::Paint),
                        "{at}: flat ({}, {})",
                        o.a,
                        o.b
                    );
                }
                continue;
            }
            // Not flat: a relation for exactly the pairs sharing a sub-sample, and each pair's
            // order decided from its relation as `order::Shared::decided` decides it.
            let wanted: Vec<(usize, usize)> = (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
                .filter(|&(a, b)| both(a, b) != 0)
                .collect();
            assert_eq!(pairs, wanted, "{at}: the pairs sharing a sub-sample");
            for o in &p.pairs {
                let q = r.relations.iter().find(|q| (q.a, q.b) == (o.a, o.b));
                let want = decided(p, o.a, o.b, q.map(|q| q.shared));
                assert_eq!(
                    (o.front, o.how),
                    want,
                    "{at}: ({}, {}) from {:?}",
                    o.a,
                    o.b,
                    q.map(|q| q.shared)
                );
                match q.map(|q| q.shared) {
                    None => alone += 1,
                    Some(s) => {
                        by_how[o.how as usize] += 1;
                        let (an, bn) = (s.a_nearer.count_ones(), s.b_nearer.count_ones());
                        even += (o.how == Decided::Majority && an == bn) as usize;
                    }
                }
            }
            // The sub-sample ranks read the same orders.
            if p.sample_ranked {
                for (k, orders) in p.sample_pairs.iter().enumerate() {
                    for o in orders {
                        let q = r
                            .relations
                            .iter()
                            .find(|q| (q.a, q.b) == (o.a, o.b))
                            .unwrap_or_else(|| panic!("{at} sub-sample {k}: ({}, {})", o.a, o.b));
                        let bit = |m: u16| m >> k & 1 != 0;
                        let s = q.shared;
                        assert!(bit(s.covered), "{at} sub-sample {k}");
                        let front = if bit(s.ab.front) {
                            Some(o.a)
                        } else if bit(s.ba.front) {
                            Some(o.b)
                        } else {
                            None
                        };
                        assert_eq!(o.front, front, "{at} sub-sample {k}: ({}, {})", o.a, o.b);
                        assert_eq!(
                            o.how == Decided::Depth,
                            bit(s.ab.strong),
                            "{at} sub-sample {k}"
                        );
                        ranked += 1;
                    }
                }
            }
        }
    });
    let rows: usize = by_how.iter().sum();
    eprintln!(
        "pair orders read from their relations {rows} (paint, tie, depth, majority {by_how:?}; \
         {even} majorities on equal votes), pairs sharing no sub-sample {alone}, flat pixels \
         {flat}, ranked sub-sample orders {ranked}"
    );
    assert!(
        rows > 1000 && alone > 0 && ranked > 100,
        "{rows} {alone} {ranked}"
    );
    assert!(by_how.iter().all(|&n| n > 0), "{by_how:?}");
}
