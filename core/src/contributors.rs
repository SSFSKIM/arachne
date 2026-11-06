//! The contributor report: per pixel, the drawn leaves its colour is made of, in depth order.
//!
//! A leaf is one drawn thing, `(object, part)`: a record's object and its order entry's part (a
//! face of a face group), an `instances` record's member reporting its own object
//! ([`crate::stream::Stream::contributors`]). A leaf's fragments in a pixel (a mesh's triangles,
//! a path's background stroke, fill and stroke) are one leaf.
//!
//! The report reads the passes the resolves read, at the same sub-samples, with the same
//! coverage: at sub-sample `k` a fragment covers when its 8-bit coverage there is not zero (a
//! mask's bit, or its exact area inside the sub-sample's cell), and the fragments covering `k`
//! are in the depth order [`crate::composite`] composites them in (nearer first, then later in
//! the draw order, then the later part). From these, per pixel:
//!
//! - **`orders`**: each distinct sequence of leaves the sub-samples see, bottom first (the order
//!   composited: furthest first), each leaf with its coverage at that sub-sample. Every
//!   leaf covering a sub-sample is in it, hidden or not.
//! - **`crossing`**: two leaves come in one order at one sub-sample and in the other order at
//!   another (two surfaces crossing inside the pixel).
//! - **`leaves`**: every leaf covering a sub-sample, bottom first, in an order every sequence of
//!   `orders` agrees with when the pixel is not a crossing (the leaves no sub-sample orders
//!   against each other placed by their depth at the pixel's centre); at a crossing, in their
//!   depth order at the pixel's centre. Each with its **coverage** (its sub-sample coverages
//!   summed over the 16 and scaled to 0..=255, `(sum + 8) / 16`), its **colour** (premultiplied,
//!   the colour of its nearest fragment at the first sub-sample it covers), and whether it is
//!   **shown**: at one sub-sample at least it covers, no fragment of another leaf in front of
//!   it there is opaque and covers that sub-sample whole. A leaf that is not shown is hidden
//!   behind an opaque front, and is still reported: it is the evidence that something lies
//!   behind, where a leaf the core did not draw would be absent.

use crate::composite::{Fragment, SAMPLES, front_to_back, sample_point};
use std::cmp::Ordering;

/// One leaf of a pixel ([`PixelContributors::leaves`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leaf {
    pub object: u32,
    pub part: u32,
    /// Its share of the pixel's sub-samples, `0..=255`.
    pub coverage: u8,
    /// Premultiplied RGBA.
    pub colour: [u8; 4],
    pub shown: bool,
}

/// What the core composites at one pixel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PixelContributors {
    /// Bottom first.
    pub leaves: Vec<Leaf>,
    /// Each distinct sequence the sub-samples see, bottom first: `(index into leaves, coverage
    /// at the sub-sample)`.
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
    let mut colour: Vec<Option<[u8; 4]>> = vec![None; n];
    // before[a][b]: some sub-sample sees a in front of b.
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
        // Front to back: each leaf at its nearest fragment, with its greatest coverage here.
        let mut seq: Vec<(usize, u8)> = Vec::new();
        let mut covered_whole = false; // an opaque fragment of an earlier leaf covers k whole
        for &i in &at_k {
            let l = of[i];
            let c = coverage[i][k];
            match seq.iter_mut().find(|(s, _)| *s == l) {
                Some(entry) => entry.1 = entry.1.max(c),
                None => {
                    seq.push((l, c));
                    if !covered_whole {
                        shown[l] = true;
                    }
                    if colour[l].is_none() {
                        colour[l] = Some(frags[i].colour);
                    }
                }
            }
            if frags[i].colour[3] == 255 && c == 255 {
                // Hides what comes after it, except more of its own leaf.
                covered_whole = true;
            }
        }
        // A leaf's own later fragments never hide it: `shown` was set at its first fragment.
        for (pos, &(a, c)) in seq.iter().enumerate() {
            covered[a] = true;
            sums[a] += c as u32;
            for &(b, _) in &seq[pos + 1..] {
                before[a][b] = true;
            }
        }
        if !seq.is_empty() {
            seq.reverse();
            if !sequences.contains(&seq) {
                sequences.push(seq);
            }
        }
    }
    let crossing = (0..n).any(|a| (a + 1..n).any(|b| before[a][b] && before[b][a]));
    // Each leaf's nearest fragment at the pixel's centre: the tie-break, and a crossing's order.
    let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
    let nearest: Vec<usize> = (0..n)
        .map(|l| {
            (0..frags.len())
                .filter(|i| of[*i] == l)
                .min_by(|&i, &j| {
                    front_to_back(
                        &frags[i],
                        frags[i].depth.at(cx, cy),
                        &frags[j],
                        frags[j].depth.at(cx, cy),
                    )
                })
                .expect("a leaf has a fragment")
        })
        .collect();
    let centre = |a: usize, b: usize| -> Ordering {
        let (i, j) = (nearest[a], nearest[b]);
        front_to_back(
            &frags[i],
            frags[i].depth.at(cx, cy),
            &frags[j],
            frags[j].depth.at(cx, cy),
        )
    };
    let live: Vec<usize> = (0..n).filter(|l| covered[*l]).collect();
    // Front to back.
    let order: Vec<usize> = if crossing {
        let mut o = live.clone();
        o.sort_by(|&a, &b| centre(a, b));
        o
    } else {
        // Every sub-sample's order kept; the rest by depth at the centre.
        let mut left = live.clone();
        let mut o = Vec::with_capacity(left.len());
        while !left.is_empty() {
            let ready = left
                .iter()
                .copied()
                .filter(|&b| !left.iter().any(|&a| a != b && before[a][b]))
                .min_by(|&a, &b| centre(a, b))
                .expect("an order without a crossing has no cycle");
            o.push(ready);
            left.retain(|&l| l != ready);
        }
        o
    };
    let index: Vec<Option<usize>> = {
        let mut index = vec![None; n];
        for (pos, &l) in order.iter().rev().enumerate() {
            index[l] = Some(pos);
        }
        index
    };
    let leaves = order
        .iter()
        .rev()
        .map(|&l| Leaf {
            object: ids[l].0,
            part: ids[l].1,
            coverage: ((sums[l] + SAMPLES as u32 / 2) / SAMPLES as u32) as u8,
            colour: colour[l].unwrap_or([0; 4]),
            shown: shown[l],
        })
        .collect();
    let orders = sequences
        .into_iter()
        .map(|s| {
            s.into_iter()
                .map(|(l, c)| (index[l].expect("a leaf in a sequence covers"), c))
                .collect()
        })
        .collect();
    PixelContributors {
        leaves,
        orders,
        crossing,
    }
}
