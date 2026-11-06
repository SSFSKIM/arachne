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
//!   - its **colour**: premultiplied, its nearest fragment's at the first sub-sample it covers;
//!   - what it **draws** alone: premultiplied, each sub-sample's composite of its own fragments
//!     there (back to front, each at its coverage), averaged over the 16, as the sample resolve
//!     composites it over a transparent pixel;
//!   - whether it is **shown**: at one sub-sample at least it covers, no fragment of another
//!     leaf in front of it there is opaque and covers that sub-sample whole. A leaf that is not
//!     shown is hidden behind an opaque front, and is still reported: it is the evidence that
//!     something lies behind, where a leaf the core did not draw would be absent;
//!   - whether it **covers** the pixel: at every sub-sample one of its fragments is opaque and
//!     covers it whole, so nothing beneath it reaches the pixel.

use crate::composite::{Fragment, SAMPLES, front_to_back, sample_point};
use std::cmp::Ordering;

/// One leaf of a pixel ([`PixelContributors::leaves`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leaf {
    pub object: u32,
    pub part: u32,
    /// Its share of the pixel's sub-samples, `0..=255`.
    pub coverage: u8,
    /// Its nearest fragment's colour, premultiplied RGBA.
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
    pub crossing: bool,
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
    let mut covered = vec![false; n];
    let mut sums = vec![0u32; n];
    let mut shown = vec![false; n];
    let mut covers = vec![true; n];
    let mut colour: Vec<Option<[u8; 4]>> = vec![None; n];
    let mut drawn = vec![[0.0f64; 4]; n];
    // before[a][b]: some sub-sample composites a run of a in front of a run of b.
    let mut before = vec![vec![false; n]; n];
    let mut sequences: Vec<Vec<(usize, u8)>> = Vec::new();
    let mut at_k: Vec<usize> = Vec::new();
    for k in 0..SAMPLES {
        let [px, py] = sample_point(x, y, k);
        at_k.clear();
        at_k.extend((0..frags.len()).filter(|i| coverage[*i][k] != 0));
        at_k.sort_by(|&i, &j| {
            front_to_back(
                &frags[i],
                frags[i].depth.at(px, py),
                &frags[j],
                frags[j].depth.at(px, py),
            )
        });
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
        // What each leaf draws alone here: its own fragments, back to front.
        let mut alone = vec![[0.0f64; 4]; n];
        for &i in at_k.iter().rev() {
            let a = coverage[i][k] as f64 / 255.0;
            let src = frags[i].colour.map(|v| v as f64 / 255.0 * a);
            let acc = &mut alone[of[i]];
            for ch in 0..4 {
                acc[ch] = src[ch] + acc[ch] * (1.0 - src[3]);
            }
        }
        for l in 0..n {
            for ch in 0..4 {
                drawn[l][ch] += alone[l][ch];
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
            if !sequences.contains(&seq) {
                sequences.push(seq);
            }
        }
    }
    let mut crossing = (0..n).any(|a| (a + 1..n).any(|b| before[a][b] && before[b][a]));
    // Each leaf's nearest fragment at the pixel's centre: the tie-break, and a crossing's order.
    let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
    let nearest: Vec<Option<usize>> = (0..n)
        .map(|l| {
            (0..frags.len()).filter(|i| of[*i] == l).min_by(|&i, &j| {
                front_to_back(
                    &frags[i],
                    frags[i].depth.at(cx, cy),
                    &frags[j],
                    frags[j].depth.at(cx, cy),
                )
            })
        })
        .collect();
    let centre = |a: usize, b: usize| -> Ordering {
        match (nearest[a], nearest[b]) {
            (Some(i), Some(j)) => front_to_back(
                &frags[i],
                frags[i].depth.at(cx, cy),
                &frags[j],
                frags[j].depth.at(cx, cy),
            ),
            _ => a.cmp(&b),
        }
    };
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
            drawn: drawn[l].map(|v| (v / SAMPLES as f64 * 255.0).round().clamp(0.0, 255.0) as u8),
            shown: shown[l],
            covers: covers[l],
        })
        .collect();
    let orders = sequences
        .into_iter()
        .map(|s| {
            s.into_iter()
                .filter_map(|(l, c)| index[l].map(|i| (i, c)))
                .collect()
        })
        .collect();
    PixelContributors {
        leaves,
        orders,
        crossing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::{Coverage, Plane};

    /// A fragment of leaf `object` at depth `a + b x` (nearer is smaller), drawn `order`th.
    fn fragment(a: f64, b: f64, colour: [u8; 4], order: u32, object: u32) -> Fragment {
        Fragment::new(
            Plane { a, b, c: 0.0 },
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
        let fill = fragment(1.0, 0.0, [255, 0, 0, 255], 0, 0);
        let stroke = fragment(1.0, 0.0, [0, 0, 128, 128], 0, 1);
        let one_leaf = |_: &Fragment| (1, 0);
        let p = of_pixel(0, 0, &[fill, stroke], &[[255; SAMPLES]; 2], &one_leaf);
        assert_eq!(p.leaves.len(), 1);
        assert_eq!(p.leaves[0].drawn, [127, 0, 128, 255]);
        assert!(p.leaves[0].covers);
        // Two triangles of one mesh tiling the pixel, coloured apart: their average.
        let tiles = [
            fragment(1.0, 0.0, [200, 0, 0, 255], 0, 0),
            fragment(1.0, 0.0, [0, 100, 0, 255], 0, 0),
        ];
        let halves = [
            only(&(0..8).collect::<Vec<_>>()),
            only(&(8..16).collect::<Vec<_>>()),
        ];
        let p = of_pixel(0, 0, &tiles, &halves, &one_leaf);
        assert_eq!(p.leaves[0].drawn, [100, 50, 0, 255]);
        assert_eq!(p.leaves[0].coverage, 255);
        // Opaque at every sub-sample between them: nothing beneath reaches the pixel.
        assert!(p.leaves[0].covers);
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
}
