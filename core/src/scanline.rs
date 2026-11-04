//! Exact coverage of polygons filled by the nonzero rule, on the pixel grid, and point-sampled
//! masks of the same polygons at the compositor's sub-samples.
//!
//! Coverage is the exact area of each pixel inside the filled region. Each pixel row is cut into
//! bands at every vertex inside it and every crossing of two edges inside it; within a band no two
//! edges cross, so the filled region is a union of trapezoids between the edges where the winding
//! number becomes or stops being nonzero, and only those transition edges are accumulated (each
//! adds the signed area to its right, prefix-summed along the row). This is exact for any winding,
//! self-overlaps included, where plain signed-area accumulation (`min(|sum|, 1)`) over-covers the
//! pixels in which an outline overlaps itself (a stroke's caps, dash ends and inner joins).
//! Crossings are found as order changes: within a band free of vertices, the edges sorted by their
//! x at its top are insertion-sorted by their x at its bottom, and each swap is a crossing.
//!
//! Rows are independent, and a rectangle of the frame (a tile) is covered on its own: edges left of
//! it count only for their winding, edges right of it not at all. Accumulation is sparse (an entry
//! per cell an edge crosses, sorted along the row), so a pixel's cost is the edges that cross it,
//! not the path's bounding box. Arithmetic is plain f64 in a fixed order, so the bytes are the same
//! on every machine.

use crate::composite::{Rect, SAMPLES, sample_point};

/// One edge of a polygon on the pixel plane, oriented downward (`y0 < y1`), with the winding it
/// adds where it crosses a ray: `dir` is +1 for an edge the polygon walks downward, -1 upward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
    pub dir: i32,
}

impl Edge {
    /// The edge's x at height `y` (exact at its two ends).
    pub fn x_at(&self, y: f64) -> f64 {
        if y <= self.y0 {
            self.x0
        } else if y >= self.y1 {
            self.x1
        } else {
            self.x0 + (y - self.y0) * (self.x1 - self.x0) / (self.y1 - self.y0)
        }
    }

    pub fn x_min(&self) -> f64 {
        self.x0.min(self.x1)
    }

    pub fn x_max(&self) -> f64 {
        self.x0.max(self.x1)
    }
}

/// The edges of closed polygons (each ring closes back to its first point); horizontal edges add
/// no winding and are left out. Rings with a non-finite point are left out whole.
pub fn edges_of(rings: &[Vec<[f64; 2]>]) -> Vec<Edge> {
    let mut edges = Vec::new();
    for ring in rings {
        if ring.len() < 2 || ring.iter().flatten().any(|v| !v.is_finite()) {
            continue;
        }
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            if a[1] < b[1] {
                edges.push(Edge {
                    x0: a[0],
                    y0: a[1],
                    x1: b[0],
                    y1: b[1],
                    dir: 1,
                });
            } else if a[1] > b[1] {
                edges.push(Edge {
                    x0: b[0],
                    y0: b[1],
                    x1: a[0],
                    y1: a[1],
                    dir: -1,
                });
            }
        }
    }
    edges
}

/// Buffers reused from row to row and path to path.
#[derive(Default)]
pub struct Scratch {
    active: Vec<u32>,
    cuts: Vec<f64>,
    spanning: Vec<(f64, f64, u32)>,
    by_bottom: Vec<(f64, f64, u32)>,
    crossings: Vec<f64>,
    band: Vec<(f64, u32)>,
    entries: Vec<(i64, f64)>,
}

/// Areas within this of 0 or 1 are taken as 0 or 1: the sums that make a pixel full or empty
/// round in their last bits.
const SNAP: f64 = 1e-9;

/// The exact nonzero coverage of the polygons whose `edges` are listed (by index) in `candidates`,
/// over the pixels of `rect`: calls `emit(y, x0, x1, area)` for each run of pixels `x0..x1` of row
/// `y` that share one area in `(0, 1]`, left to right, row by row. `candidates` must hold every
/// edge that overlaps the rectangle's rows and starts left of its right side; others may be there.
pub fn cover(
    edges: &[Edge],
    candidates: &[u32],
    rect: Rect,
    scratch: &mut Scratch,
    mut emit: impl FnMut(u32, u32, u32, f64),
) {
    let lo = rect.x0 as f64 - 1.0;
    let hi = rect.x1 as f64 + 1.0;
    for y in rect.y0..rect.y1 {
        let (top, bottom) = (y as f64, y as f64 + 1.0);
        let s = &mut *scratch;
        s.active.clear();
        s.active.extend(candidates.iter().copied().filter(|&i| {
            let e = &edges[i as usize];
            e.y0 < bottom && e.y1 > top && e.x_min() < hi
        }));
        if s.active.is_empty() {
            continue;
        }
        s.cuts.clear();
        s.cuts.extend([top, bottom]);
        for &i in &s.active {
            let e = &edges[i as usize];
            for v in [e.y0, e.y1] {
                if v > top && v < bottom {
                    s.cuts.push(v);
                }
            }
        }
        s.cuts.sort_by(f64::total_cmp);
        s.cuts.dedup();
        s.entries.clear();
        for k in 0..s.cuts.len() - 1 {
            let (ya, yb) = (s.cuts[k], s.cuts[k + 1]);
            interval(edges, ya, yb, lo, hi, s);
        }
        emit_row(y, rect, s, &mut emit);
    }
}

/// Accumulates the interval `ya..yb` of a row, inside which no edge starts or ends.
fn interval(edges: &[Edge], ya: f64, yb: f64, lo: f64, hi: f64, s: &mut Scratch) {
    s.spanning.clear();
    for &i in &s.active {
        let e = &edges[i as usize];
        if e.y0 <= ya && e.y1 >= yb {
            s.spanning.push((e.x_at(ya), e.x_at(yb), i));
        }
    }
    if s.spanning.is_empty() {
        return;
    }
    s.spanning
        .sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    // Insertion-sort by x at the bottom: every swap is a pair that crosses inside the interval.
    s.by_bottom.clear();
    s.by_bottom.extend_from_slice(&s.spanning);
    s.crossings.clear();
    for i in 1..s.by_bottom.len() {
        let mut j = i;
        while j > 0 && s.by_bottom[j - 1].1 > s.by_bottom[j].1 {
            let (a, b) = (s.by_bottom[j - 1], s.by_bottom[j]);
            let d0 = a.0 - b.0;
            let d1 = a.1 - b.1;
            let yc = ya + d0 / (d0 - d1) * (yb - ya);
            if yc > ya && yc < yb {
                s.crossings.push(yc);
            }
            s.by_bottom.swap(j - 1, j);
            j -= 1;
        }
    }
    if s.crossings.is_empty() {
        // One band: the order at the top holds throughout.
        let mut winding = 0;
        for &(xa, xb, i) in &s.spanning {
            let before = winding;
            winding += edges[i as usize].dir;
            if let Some(sign) = transition(before, winding) {
                accumulate(&mut s.entries, xa, xb, sign * (yb - ya), lo, hi);
            }
        }
        return;
    }
    s.crossings.sort_by(f64::total_cmp);
    s.crossings.dedup();
    let mut ba = ya;
    for k in 0..=s.crossings.len() {
        let bb = if k < s.crossings.len() {
            s.crossings[k]
        } else {
            yb
        };
        let mid = 0.5 * (ba + bb);
        s.band.clear();
        s.band.extend(
            s.spanning
                .iter()
                .map(|&(_, _, i)| (edges[i as usize].x_at(mid), i)),
        );
        s.band.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut winding = 0;
        for &(_, i) in &s.band {
            let e = &edges[i as usize];
            let before = winding;
            winding += e.dir;
            if let Some(sign) = transition(before, winding) {
                accumulate(
                    &mut s.entries,
                    e.x_at(ba),
                    e.x_at(bb),
                    sign * (bb - ba),
                    lo,
                    hi,
                );
            }
        }
        ba = bb;
    }
}

/// +1 where the winding becomes nonzero, -1 where it returns to zero.
fn transition(before: i32, after: i32) -> Option<f64> {
    match (before == 0, after == 0) {
        (true, false) => Some(1.0),
        (false, true) => Some(-1.0),
        _ => None,
    }
}

/// Adds a boundary segment of one band, from `xa` at its top to `xb` at its bottom with signed
/// height `d`, to the row's entries. The part left of `lo` counts as a vertical boundary at `lo`
/// (it changes the winding of everything to its right, inside the rectangle); the part right of
/// `hi` touches no pixel of the rectangle and is dropped.
fn accumulate(entries: &mut Vec<(i64, f64)>, xa: f64, xb: f64, d: f64, lo: f64, hi: f64) {
    if xa <= lo && xb <= lo {
        entries.push((lo as i64, d));
        return;
    }
    if xa >= hi && xb >= hi {
        return;
    }
    if xa >= lo && xb >= lo && xa <= hi && xb <= hi {
        cell_row(entries, xa, xb, d);
        return;
    }
    // Clip by the fraction of the band each part spans (x is linear in y along the segment).
    let at = |x: f64| (x - xa) / (xb - xa);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    if xa < lo {
        t0 = at(lo);
    } else if xa > hi {
        t0 = at(hi);
    }
    if xb < lo {
        t1 = at(lo);
    } else if xb > hi {
        t1 = at(hi);
    }
    // The part left of `lo`.
    let left = if xa < lo {
        t0
    } else if xb < lo {
        1.0 - t1
    } else {
        0.0
    };
    if left > 0.0 {
        entries.push((lo as i64, d * left));
    }
    if t1 > t0 {
        let x = |t: f64| (xa + (xb - xa) * t).clamp(lo, hi);
        cell_row(entries, x(t0), x(t1), d * (t1 - t0));
    }
}

/// One boundary piece inside one band, from `x` to `xnext` with signed height `d`: the signed area
/// to its right in each cell it crosses, as prefix-sum entries (a cell's coverage is the sum of the
/// entries at it and left of it).
fn cell_row(entries: &mut Vec<(i64, f64)>, x: f64, xnext: f64, d: f64) {
    let (x0, x1) = if x < xnext { (x, xnext) } else { (xnext, x) };
    let x0floor = x0.floor();
    let x0i = x0floor as i64;
    let x1ceil = x1.ceil();
    let x1i = x1ceil as i64;
    if x1i <= x0i + 1 {
        let xmf = 0.5 * (x + xnext) - x0floor;
        entries.push((x0i, d - d * xmf));
        entries.push((x0i + 1, d * xmf));
        return;
    }
    let s = 1.0 / (x1 - x0);
    let x0f = x0 - x0floor;
    let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
    let x1f = x1 - x1ceil + 1.0;
    let am = 0.5 * s * x1f * x1f;
    entries.push((x0i, d * a0));
    if x1i == x0i + 2 {
        entries.push((x0i + 1, d * (1.0 - a0 - am)));
    } else {
        let a1 = s * (1.5 - x0f);
        entries.push((x0i + 1, d * (a1 - a0)));
        for xi in x0i + 2..x1i - 1 {
            entries.push((xi, d * s));
        }
        let a2 = a1 + (x1i - x0i - 3) as f64 * s;
        entries.push((x1i - 1, d * (1.0 - a2 - am)));
    }
    entries.push((x1i, d * am));
}

/// Sums a row's entries left to right and emits its runs inside the rectangle.
fn emit_row(y: u32, rect: Rect, s: &mut Scratch, emit: &mut impl FnMut(u32, u32, u32, f64)) {
    // A stable sort: entries at one cell are summed in the order they were made.
    s.entries.sort_by_key(|e| e.0);
    let (rx0, rx1) = (rect.x0 as i64, rect.x1 as i64);
    let mut sum = 0.0;
    let mut k = 0;
    while k < s.entries.len() {
        let cell = s.entries[k].0;
        while k < s.entries.len() && s.entries[k].0 == cell {
            sum += s.entries[k].1;
            k += 1;
        }
        let next = s.entries.get(k).map_or(rx1, |e| e.0);
        let (x0, x1) = (cell.max(rx0), next.min(rx1));
        if x0 >= x1 {
            continue;
        }
        let area = if sum > 1.0 - SNAP {
            1.0
        } else if sum < SNAP {
            continue;
        } else {
            sum
        };
        emit(y, x0 as u32, x1 as u32, area);
    }
}

/// The nonzero winding of the polygons at each sub-sample of pixel `(x, y)`, as a mask (bit `s` is
/// sub-sample `s`): the point-sampled twin of [`cover`], for pixels the compositor resolves sample
/// by sample. `candidates` must hold every edge that overlaps the pixel's row and starts left of
/// its right side.
pub fn mask(edges: &[Edge], candidates: &[u32], x: u32, y: u32) -> u16 {
    let points: [[f64; 2]; SAMPLES] = std::array::from_fn(|s| sample_point(x, y, s));
    let mut winding = [0i32; SAMPLES];
    let (top, bottom, right) = (y as f64, y as f64 + 1.0, x as f64 + 1.0);
    for &i in candidates {
        let e = &edges[i as usize];
        if e.y1 <= top || e.y0 >= bottom || e.x_min() >= right {
            continue;
        }
        for (s, [px, py]) in points.iter().enumerate() {
            // A ray to the left: edges crossing the sample's row (half-open in y) left of it.
            if e.y0 <= *py && *py < e.y1 && e.x_at(*py) < *px {
                winding[s] += e.dir;
            }
        }
    }
    (0..SAMPLES).fold(0u16, |m, s| if winding[s] != 0 { m | 1 << s } else { m })
}
