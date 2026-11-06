//! The contributor report: per pixel, the drawn leaves its colour is made of, in depth order.
//!
//! A leaf is one drawn thing, `(object, part)`: a record's object and its order entry's part (a
//! face of a face group), an `instances` record's member reporting its own object
//! ([`crate::stream::Stream::contributors`]). A leaf's fragments in a pixel (a mesh's triangles,
//! a path's background stroke, fill and stroke, a cloud's blocks) belong to it.
//!
//! The report reads the passes the resolves read, at the same sub-samples, with the same
//! coverage: at sub-sample `k` a fragment covers when its 8-bit coverage there is not zero (a
//! mask's bit, or its exact area inside the sub-sample's cell), and the fragments covering `k`
//! are in the depth order [`crate::composite`] composites them in (nearer first, then later in
//! the draw order, then the later part). From these, per pixel:
//!
//! - **`orders`**: each distinct sequence the sub-samples see, bottom first (the order
//!   composited: furthest first). A sequence is the sub-sample's fragments with each run of one
//!   leaf's fragments made one entry, with the run's greatest coverage there. A leaf whose
//!   fragments another leaf lies between comes in the sequence once per run (a fill behind a
//!   surface and its stroke in front of it). Every leaf covering a sub-sample is in it, hidden or
//!   not.
//! - **`crossing`**: the sequences order two leaves both ways (two surfaces crossing inside the
//!   pixel, or one leaf on both sides of another), or no single order of the leaves agrees with
//!   every sequence.
//! - **`leaves`**: every leaf covering a sub-sample, once each, bottom first: when the pixel is
//!   not a crossing, in the order every sequence agrees with (leaves no sub-sample orders
//!   against each other placed by their depth at the pixel's centre); at a crossing, by the
//!   depth of their nearest fragment at the pixel's centre. Each with:
//!   - its **coverage**: its sub-sample coverages summed over the 16 and scaled to 0..=255,
//!     `(sum + 8) / 16`;
//!   - its **colour**: premultiplied, its nearest fragment's at the first sub-sample it covers:
//!     the paint of that one fragment (a path's stroke, where it lies over the path's own fill),
//!     not what the leaf draws and not what the pixel shows. Neither this nor `drawn` is the
//!     leaf's part of the frame: a leaf behind an opaque front, or seen through a sliver of the
//!     pixel, keeps its whole colour in both while the pixel barely moves;
//!   - what it **draws** alone: premultiplied, the pixel the resolve itself makes of its
//!     fragments alone over a transparent pixel (`composite::resolve_pixel`: the same selection,
//!     the coverage or sample resolve as it would choose, the same integer blending);
//!   - whether it is **shown**: at one sub-sample at least it covers, no fragment of another
//!     leaf in front of it there is opaque and covers that sub-sample whole. A leaf that is not
//!     shown is hidden behind an opaque front, and is still reported: it is the evidence that
//!     something lies behind, where a leaf the core did not draw would be absent;
//!   - whether it **covers** the pixel: at every sub-sample one of its fragments is opaque and
//!     covers it whole, so nothing beneath it reaches the pixel.

use crate::composite::{Fragment, SAMPLES, Scratch, depth_error, resolve_pixel, sample_depths};
use crate::order::{self, Decided};
use std::cmp::Ordering;

/// One leaf of a pixel ([`PixelContributors::leaves`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leaf {
    pub object: u32,
    pub part: u32,
    /// Its share of the pixel's sub-samples, `0..=255`.
    pub coverage: u8,
    /// Its nearest fragment's paint, premultiplied RGBA: one fragment's colour, not the leaf's
    /// drawn colour ([`Leaf::drawn`]) nor the pixel's (the module's **colour**).
    pub colour: [u8; 4],
    /// What it draws alone over a transparent pixel, premultiplied RGBA.
    pub drawn: [u8; 4],
    pub shown: bool,
    pub covers: bool,
}

/// What the core composites at one pixel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PixelContributors {
    /// Bottom first, each leaf once.
    pub leaves: Vec<Leaf>,
    /// Each distinct sequence the sub-samples see, bottom first: `(index into leaves, coverage
    /// at the sub-sample)`, a leaf once per run of its fragments.
    pub orders: Vec<Vec<(usize, u8)>>,
    /// Each sub-sample's own sequence, in sample order (`SAMPLES` of them, a sub-sample no
    /// fragment covers empty), as `orders` gives them: `orders` is their distinct ones.
    pub samples: Vec<Vec<(usize, u8)>>,
    pub crossing: bool,
    /// How each pair of leaves was ordered, `(a, b, rule)` for indices `a < b` into `leaves`:
    /// the strongest rule (`Majority`, then `Depth`, then `Tie`, then `Paint`) that ordered a
    /// fragment of one against a fragment of the other ([`order::pair`]).
    pub decided: Vec<(usize, usize, Decided)>,
}

/// The report of pixel `(x, y)` from its fragments, their coverage at each sub-sample, and the
/// leaf each belongs to.
// The sub-sample `k` indexes each fragment's coverage row and places the sub-sample's point.
#[allow(clippy::needless_range_loop)]
pub(crate) fn of_pixel(
    x: u32,
    y: u32,
    frags: &[Fragment],
    coverage: &[[u8; SAMPLES]],
    leaf_of: &dyn Fn(&Fragment) -> (u32, u32),
) -> PixelContributors {
    // Leaves by first appearance among the fragments, and each fragment's leaf.
    let mut ids: Vec<(u32, u32)> = Vec::new();
    let of: Vec<usize> = frags
        .iter()
        .map(|f| {
            let id = leaf_of(f);
            match ids.iter().position(|i| *i == id) {
                Some(i) => i,
                None => {
                    ids.push(id);
                    ids.len() - 1
                }
            }
        })
        .collect();
    let n = ids.len();
    let depths: Vec<[f64; SAMPLES]> = frags.iter().map(|f| sample_depths(f, x, y)).collect();
    let errs: Vec<f64> = frags.iter().map(|f| depth_error(f, x, y)).collect();
    let seen = |i: usize| order::Seen {
        fragment: &frags[i],
        coverage: &coverage[i],
        depth: &depths[i],
        err: errs[i],
    };
    let pair = |a: usize, b: usize| order::pair(seen(a), seen(b));
    let painted = |a: usize, b: usize| order::painted(&frags[a], &frags[b]);
    let mut covered = vec![false; n];
    let mut sums = vec![0u32; n];
    let mut shown = vec![false; n];
    let mut covers = vec![true; n];
    let mut colour: Vec<Option<[u8; 4]>> = vec![None; n];
    // before[a][b]: some sub-sample composites a run of a in front of a run of b.
    let mut before = vec![vec![false; n]; n];
    let mut sequences: Vec<Vec<(usize, u8)>> = Vec::new();
    let mut per_sample: Vec<Vec<(usize, u8)>> = vec![Vec::new(); SAMPLES];
    let mut at_k: Vec<usize> = Vec::new();
    let mut ranked: Vec<usize> = Vec::new();
    for k in 0..SAMPLES {
        // The sub-sample's fragments as the sample resolve ranks them.
        ranked.clear();
        ranked.extend((0..frags.len()).filter(|i| coverage[*i][k] != 0));
        let how = order::rank(
            &ranked,
            |a, b| {
                a != b
                    && order::at_sample(
                        &frags[a],
                        depths[a][k],
                        errs[a],
                        &frags[b],
                        depths[b][k],
                        errs[b],
                    ) == Ordering::Less
            },
            |a, b| {
                order::strong_at(
                    &frags[a],
                    depths[a][k],
                    errs[a],
                    &frags[b],
                    depths[b][k],
                    errs[b],
                )
            },
            painted,
            &mut at_k,
        );
        assert_ne!(how, order::Ranked::Strong, "{}", order::ONE_SAMPLE);
        // Front to back: runs of one leaf's fragments, each with its greatest coverage here.
        let mut seq: Vec<(usize, u8)> = Vec::new();
        let mut seen = vec![false; n];
        let mut whole = vec![false; n];
        let mut covered_whole = false; // an opaque fragment in front covers k whole
        for &i in &at_k {
            let l = of[i];
            let c = coverage[i][k];
            match seq.last_mut() {
                Some(run) if run.0 == l => run.1 = run.1.max(c),
                _ => seq.push((l, c)),
            }
            if !seen[l] {
                seen[l] = true;
                // A leaf's own fragments never hide it: it is judged at its nearest one.
                if !covered_whole {
                    shown[l] = true;
                }
                if colour[l].is_none() {
                    colour[l] = Some(frags[i].colour);
                }
            }
            if frags[i].colour[3] == 255 && c == 255 {
                covered_whole = true;
                whole[l] = true;
            }
        }
        for l in 0..n {
            if !whole[l] {
                covers[l] = false;
            }
        }
        let mut counted = vec![false; n];
        for (pos, &(a, c)) in seq.iter().enumerate() {
            covered[a] = true;
            if !counted[a] {
                counted[a] = true;
                let most = seq.iter().filter(|(s, _)| *s == a).map(|(_, c)| *c).max();
                sums[a] += most.unwrap_or(c) as u32;
            }
            for &(b, _) in &seq[pos + 1..] {
                if b != a {
                    before[a][b] = true;
                }
            }
        }
        if !seq.is_empty() {
            seq.reverse();
            per_sample[k] = seq.clone();
            if !sequences.contains(&seq) {
                sequences.push(seq);
            }
        }
    }
    let mut crossing = (0..n).any(|a| (a + 1..n).any(|b| before[a][b] && before[b][a]));
    // What each leaf draws alone: the resolve's own pixel from its fragments alone over a
    // transparent pixel, with the coverage the resolve reads.
    let mut scratch = Scratch::default();
    let mut rows: Vec<[u8; SAMPLES]> = Vec::new();
    let drawn: Vec<[u8; 4]> = (0..n)
        .map(|l| {
            scratch.fragments.clear();
            rows.clear();
            for (i, f) in frags.iter().enumerate() {
                if of[i] == l {
                    scratch.fragments.push(*f);
                    rows.push(coverage[i]);
                }
            }
            resolve_pixel(&mut scratch, [0; 4], x, y, |i, _| rows[i]).1
        })
        .collect();
    // The pixel's one order, as the resolve ranks it: each leaf at its front-most fragment in it
    // is the tie-break, and a crossing's order. Where the strong orders cycle the resolve takes no
    // order of the pixel, and this one, broken by wins, orders the listing alone.
    let all: Vec<usize> = (0..frags.len()).collect();
    let mut one: Vec<usize> = Vec::new();
    order::rank_through_cycles(
        &all,
        |a, b| a != b && pair(a, b).0 == Ordering::Less,
        |a, b| pair(a, b).1.is_strong(),
        painted,
        &mut one,
    );
    let mut at = vec![usize::MAX; frags.len()];
    for (pos, &i) in one.iter().enumerate() {
        at[i] = pos;
    }
    let nearest: Vec<usize> = (0..n)
        .map(|l| {
            (0..frags.len())
                .filter(|i| of[*i] == l)
                .map(|i| at[i])
                .min()
                .unwrap_or(usize::MAX)
        })
        .collect();
    let centre = |a: usize, b: usize| -> Ordering { nearest[a].cmp(&nearest[b]).then(a.cmp(&b)) };
    let live: Vec<usize> = (0..n).filter(|l| covered[*l]).collect();
    // Front to back: every sub-sample's order kept, the rest by depth at the centre; where no
    // order keeps them all (a cycle across sub-samples), a crossing.
    let mut order: Vec<usize> = Vec::with_capacity(live.len());
    if !crossing {
        let mut left = live.clone();
        while !left.is_empty() {
            let ready = left
                .iter()
                .copied()
                .filter(|&b| !left.iter().any(|&a| a != b && before[a][b]))
                .min_by(|&a, &b| centre(a, b));
            match ready {
                Some(l) => {
                    order.push(l);
                    left.retain(|&m| m != l);
                }
                None => {
                    crossing = true;
                    break;
                }
            }
        }
    }
    if crossing {
        order = live.clone();
        order.sort_by(|&a, &b| centre(a, b));
    }
    let mut index = vec![None; n];
    for (pos, &l) in order.iter().rev().enumerate() {
        index[l] = Some(pos);
    }
    let leaves = order
        .iter()
        .rev()
        .map(|&l| Leaf {
            object: ids[l].0,
            part: ids[l].1,
            coverage: ((sums[l] + SAMPLES as u32 / 2) / SAMPLES as u32) as u8,
            colour: colour[l].unwrap_or([0; 4]),
            drawn: drawn[l],
            shown: shown[l],
            covers: covers[l],
        })
        .collect();
    let reindexed = |s: Vec<(usize, u8)>| -> Vec<(usize, u8)> {
        s.into_iter()
            .filter_map(|(l, c)| index[l].map(|i| (i, c)))
            .collect()
    };
    let orders = sequences.into_iter().map(reindexed).collect();
    let samples = per_sample.into_iter().map(reindexed).collect();
    // How each pair of leaves was ordered: from their fragments' pairs, the strongest rule.
    let mut ruled = vec![vec![Decided::Paint; n]; n];
    for a in 0..frags.len() {
        for b in a + 1..frags.len() {
            let (la, lb) = (of[a], of[b]);
            if la != lb {
                let d = pair(a, b).1;
                ruled[la][lb] = ruled[la][lb].max(d);
                ruled[lb][la] = ruled[la][lb];
            }
        }
    }
    let mut decided = Vec::new();
    for (p, &la) in order.iter().rev().enumerate() {
        for (q, &lb) in order.iter().rev().enumerate().skip(p + 1) {
            decided.push((p, q, ruled[la][lb]));
        }
    }
    PixelContributors {
        leaves,
        orders,
        samples,
        crossing,
        decided,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::{Coverage, Plane};

    /// A fragment of leaf `object` at depth `a + b x` (nearer is smaller), drawn `order`th.
    fn fragment(a: f64, b: f64, colour: [u8; 4], order: u32, object: u32) -> Fragment {
        Fragment::new(
            Plane::new(a, b, 0.0),
            colour,
            Coverage::Mask(0),
            order,
            object,
        )
    }

    /// The leaf a test fragment belongs to: its `sub` is its object.
    fn by_sub(f: &Fragment) -> (u32, u32) {
        (f.sub, 0)
    }

    /// `f` with the mask its full-coverage sub-samples in `row` make, as a producer gives it.
    fn masked(mut f: Fragment, row: &[u8; SAMPLES]) -> Fragment {
        f.coverage = Coverage::Mask(
            (0..SAMPLES)
                .filter(|k| row[*k] == 255)
                .fold(0, |m, k| m | 1 << k),
        );
        f
    }

    fn only(samples: &[usize]) -> [u8; SAMPLES] {
        let mut row = [0; SAMPLES];
        for &s in samples {
            row[s] = 255;
        }
        row
    }

    fn objects(p: &PixelContributors, order: &[(usize, u8)]) -> Vec<u32> {
        order.iter().map(|(i, _)| p.leaves[*i].object).collect()
    }

    #[test]
    fn a_leaf_on_both_sides_of_another_comes_once_per_run_and_is_a_crossing() {
        // Leaf 1 far and near, leaf 2 between them, all over the whole pixel.
        let frags = [
            fragment(3.0, 0.0, [255, 0, 0, 255], 0, 1),
            fragment(2.0, 0.0, [0, 255, 0, 128], 1, 2),
            fragment(1.0, 0.0, [0, 0, 128, 128], 2, 1),
        ];
        let rows = [[255; SAMPLES]; 3];
        let p = of_pixel(0, 0, &frags, &rows, &by_sub);
        assert_eq!(p.orders.len(), 1);
        assert_eq!(objects(&p, &p.orders[0]), vec![1, 2, 1]);
        assert!(p.crossing);
        let mut ids: Vec<u32> = p.leaves.iter().map(|l| l.object).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn a_leaf_draws_all_its_fragments_composited() {
        // An opaque red fill under its half-transparent blue stroke: one leaf, two layers.
        let full = [255; SAMPLES];
        let fill = masked(fragment(1.0, 0.0, [255, 0, 0, 255], 0, 0), &full);
        let stroke = masked(fragment(1.0, 0.0, [0, 0, 128, 128], 0, 1), &full);
        let one_leaf = |_: &Fragment| (1, 0);
        let p = of_pixel(0, 0, &[fill, stroke], &[full; 2], &one_leaf);
        assert_eq!(p.leaves.len(), 1);
        assert_eq!(p.leaves[0].drawn, [127, 0, 128, 255]);
        assert!(p.leaves[0].covers);
        // Two triangles of one mesh tiling the pixel, coloured apart: their average.
        let halves = [
            only(&(0..8).collect::<Vec<_>>()),
            only(&(8..16).collect::<Vec<_>>()),
        ];
        let tiles = [
            masked(fragment(1.0, 0.0, [200, 0, 0, 255], 0, 0), &halves[0]),
            masked(fragment(1.0, 0.0, [0, 100, 0, 255], 0, 0), &halves[1]),
        ];
        let p = of_pixel(0, 0, &tiles, &halves, &one_leaf);
        assert_eq!(p.leaves[0].drawn, [100, 50, 0, 255]);
        assert_eq!(p.leaves[0].coverage, 255);
        // Opaque at every sub-sample between them: nothing beneath reaches the pixel.
        assert!(p.leaves[0].covers);
    }

    #[test]
    fn a_leaf_draws_what_the_resolve_makes_of_its_coincident_partial_layers() {
        // One path's opaque red background stroke under its opaque blue stroke of the same width,
        // their shared outer edge halfway through the pixel: two area fragments over the same
        // sub-samples. The resolve takes its coverage resolve (red at 128, then blue at 128):
        // (64, 0, 128, 192), where compositing per sub-sample would give (0, 0, 128, 128).
        let half = only(&[0, 2, 4, 6, 8, 10, 12, 14]);
        let area = |colour, sub| {
            let mut f = fragment(1.0, 0.0, colour, 0, sub);
            f.coverage = Coverage::Area { area: 0.5, id: 0 };
            f
        };
        let frags = [area([255, 0, 0, 255], 0), area([0, 0, 255, 255], 1)];
        let one_leaf = |_: &Fragment| (1, 0);
        let p = of_pixel(0, 0, &frags, &[half, half], &one_leaf);
        assert_eq!(p.leaves[0].drawn, [64, 0, 128, 192]);
        let mut scratch = Scratch::default();
        scratch.fragments.extend_from_slice(&frags);
        let resolved = resolve_pixel(&mut scratch, [0; 4], 0, 0, |_, _| half).1;
        assert_eq!(p.leaves[0].drawn, resolved);
    }

    #[test]
    fn sub_samples_ordering_three_leaves_in_a_cycle_make_a_crossing() {
        // Sub-samples 0, 1, 2 sit at x 0.03, 0.47, 0.72 of the pixel. Each sees two leaves:
        // 0 sees leaf 1 over 2, 1 sees 2 over 3, 2 sees 3 over 1. No pair comes both ways.
        let frags = [
            fragment(0.0, 1.0, [255, 0, 0, 255], 0, 1),
            fragment(0.5, 0.0, [0, 255, 0, 255], 1, 2),
            fragment(1.0, -1.0, [0, 0, 255, 255], 2, 3),
        ];
        let rows = [only(&[0, 2]), only(&[0, 1]), only(&[1, 2])];
        let p = of_pixel(0, 0, &frags, &rows, &by_sub);
        assert!(p.crossing);
        assert_eq!(p.leaves.len(), 3);
        let mut seen: Vec<Vec<u32>> = p.orders.iter().map(|o| objects(&p, o)).collect();
        seen.sort();
        // Bottom first: the far leaf of each pair first.
        assert_eq!(seen, vec![vec![1, 3], vec![2, 1], vec![3, 2]]);
    }

    #[test]
    fn each_sub_sample_keeps_its_own_sequence_and_the_orders_are_their_distinct_ones() {
        // The cycle above, with leaf 1's coverage partial at sub-sample 2: each sub-sample's
        // sequence in sample order, each leaf's coverage there, and nothing at the rest.
        let frags = [
            fragment(0.0, 1.0, [255, 0, 0, 255], 0, 1),
            fragment(0.5, 0.0, [0, 255, 0, 255], 1, 2),
            fragment(1.0, -1.0, [0, 0, 255, 255], 2, 3),
        ];
        let mut one = only(&[0]);
        one[2] = 40;
        let rows = [one, only(&[0, 1]), only(&[1, 2])];
        let p = of_pixel(0, 0, &frags, &rows, &by_sub);
        assert_eq!(p.samples.len(), SAMPLES);
        let at = |k: usize| -> Vec<(u32, u8)> {
            p.samples[k]
                .iter()
                .map(|(i, c)| (p.leaves[*i].object, *c))
                .collect()
        };
        assert_eq!(at(0), vec![(2, 255), (1, 255)]);
        assert_eq!(at(1), vec![(3, 255), (2, 255)]);
        assert_eq!(at(2), vec![(1, 40), (3, 255)]);
        assert!((3..SAMPLES).all(|k| p.samples[k].is_empty()));
        let mut distinct: Vec<Vec<(usize, u8)>> = Vec::new();
        for s in p.samples.iter().filter(|s| !s.is_empty()) {
            if !distinct.contains(s) {
                distinct.push(s.clone());
            }
        }
        assert_eq!(distinct, p.orders);
    }

    #[test]
    fn two_leaves_crossing_inside_the_pixel_are_ordered_as_most_shared_sub_samples_order_them() {
        // Leaf 1 is in front left of x = 0.6 (and at the centre), leaf 2 right of it; leaf 2
        // covers one sub-sample left of the crossing and the rest of what it covers right of it.
        let x = |k: usize| crate::composite::sample_point(0, 0, k)[0];
        let mut covered: Vec<usize> = (0..SAMPLES).filter(|k| x(*k) > 0.6).collect();
        covered.push((0..SAMPLES).find(|k| x(*k) < 0.6).unwrap());
        let frags = [
            fragment(-0.6, 1.0, [255, 0, 0, 255], 0, 1),
            fragment(0.0, 0.0, [0, 0, 255, 255], 1, 2),
        ];
        let rows = [[255; SAMPLES], only(&covered)];
        let p = of_pixel(0, 0, &frags, &rows, &by_sub);
        assert!(p.crossing);
        // Bottom first: leaf 2 is in front at more of the sub-samples both cover.
        let objects: Vec<u32> = p.leaves.iter().map(|l| l.object).collect();
        assert_eq!(objects, vec![1, 2]);
        assert_eq!(p.decided, vec![(0, 1, Decided::Majority)]);
    }

    #[test]
    fn the_report_says_how_each_pair_of_leaves_was_ordered() {
        // Leaf 1 behind leaf 2 by depth where both cover; leaf 3 coincident with leaf 2 and drawn
        // after it; leaf 4 in front, sharing no sub-sample with leaf 1.
        let frags = [
            fragment(2.0, 0.0, [255, 0, 0, 128], 0, 1),
            fragment(1.0, 0.0, [0, 255, 0, 128], 1, 2),
            fragment(1.0, 0.0, [0, 0, 255, 128], 2, 3),
            fragment(0.5, 0.0, [9, 9, 9, 128], 3, 4),
        ];
        let rows = [
            only(&[0, 1]),
            only(&[0, 1, 2]),
            only(&[1, 2]),
            only(&[2, 3]),
        ];
        let p = of_pixel(0, 0, &frags, &rows, &by_sub);
        let objects: Vec<u32> = p.leaves.iter().map(|l| l.object).collect();
        assert_eq!(objects, vec![1, 2, 3, 4]);
        assert_eq!(
            p.decided,
            vec![
                (0, 1, Decided::Depth),
                (0, 2, Decided::Depth),
                (0, 3, Decided::Paint),
                (1, 2, Decided::Tie),
                (1, 3, Decided::Depth),
                (2, 3, Decided::Depth),
            ]
        );
    }
}
