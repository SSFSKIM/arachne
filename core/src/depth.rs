//! Where drawn things sit in depth: the planes of flat things, made canonical so that things on one
//! plane get bit-identical depth, and the depth along a curve for things that are not flat.
//!
//! - A **world plane** is found from a record's placed points (a path's points, a mesh triangle's
//!   corners) and made canonical: its unit normal turned so its largest component is positive, its
//!   offset taken from that normal, then both rounded to multiples of [`PLANE_GRID`]. Two paths or
//!   meshes on one plane find normals and offsets that differ in their last bits (from other points,
//!   another triangulation, another summation); rounded, they are the same numbers, so their depth
//!   planes on screen (`Projection::key_plane`) are bit-identical and they tie by draw order
//!   instead of fighting. (Two values straddling a rounding boundary still differ; with the grid at
//!   2^-20 that needs the plane to sit within a few ulps of a half-grid point.) A path whose points
//!   lie on a line has no plane of its own: it takes the axis plane of a coordinate all its points
//!   share exactly (z first, then y, then x), so an axis on the floor ties with the floor.
//! - A **centreline** carries the depth of something that is not on one plane (a curve in space,
//!   or a plane seen edge-on): its screen polyline with a depth key at each vertex. A pixel takes
//!   the depth of a segment, as the plane along that segment, so where it crosses another object
//!   the crossing still falls between sub-samples: the front-most segment that reaches the pixel
//!   (a circle seen edge-on puts its near and far arcs on one line, and the pixel shows the near
//!   one), else the nearest.

use crate::composite::Plane;

/// The grid world planes are rounded to: 2^-20, in scene units for offsets and in unit-normal
/// components.
pub const PLANE_GRID: f64 = 1.0 / 1_048_576.0;

/// How far, relative to the points' extent, points may lie off the plane they define, for
/// points not snapped to a grid (and the floor under the grid's bounds: rounding in the fit).
const PLANAR_TOLERANCE: f64 = 1e-9;

/// How far past the first-order bound of a snapped fit its tolerance sits: room for the bound's
/// second-order terms (a grid offset times a grid offset) and the fit's own rounding.
const FIT_MARGIN: f64 = 1.5;

/// The grid the view's geometry was snapped to, in scene units (`0.0`: not snapped). Every
/// planarity bound reads it from here.
pub(crate) fn input_grid(_view: &crate::xf::View) -> f64 {
    0.0
}

/// How far a point snapped to `grid` may lie from where it was: half a step in each coordinate.
fn snap_reach(grid: f64) -> f64 {
    grid * 0.75f64.sqrt()
}

/// The outline's perimeter, closed: Newell's normal sums over its closing edge too.
fn perimeter(points: &[[f64; 3]]) -> f64 {
    (0..points.len())
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            dot(d, d).sqrt()
        })
        .sum()
}

/// The points' farthest distance from `first`.
fn reach(points: &[[f64; 3]], first: [f64; 3]) -> f64 {
    points
        .iter()
        .map(|p| {
            let d = [p[0] - first[0], p[1] - first[1], p[2] - first[2]];
            dot(d, d).sqrt()
        })
        .fold(0.0, f64::max)
}

/// A plane in world space, canonical: `normal . p = offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPlane {
    pub normal: [f64; 3],
    pub offset: f64,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn round_to_grid(v: f64) -> f64 {
    (v / PLANE_GRID).round() * PLANE_GRID + 0.0
}

/// The direction of canonical planes: a unit normal turned so its largest component is positive,
/// kept for finding many planes with it through different points.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Direction {
    unit: [f64; 3],
    normal: [f64; 3],
}

impl Direction {
    /// The direction of `n` (any length but zero), or `None` for a zero or non-finite `n`.
    pub(crate) fn of(n: [f64; 3]) -> Option<Direction> {
        let norm = dot(n, n).sqrt();
        if !(norm > 0.0 && norm.is_finite()) {
            return None;
        }
        let mut unit = n.map(|v| v / norm);
        let largest = (0..3)
            .max_by(|a, b| unit[*a].abs().total_cmp(&unit[*b].abs()).then(b.cmp(a)))
            .unwrap();
        if unit[largest] < 0.0 {
            unit = unit.map(|v| -v);
        }
        Some(Direction {
            unit,
            normal: unit.map(round_to_grid),
        })
    }

    /// The rounded normal every plane in this direction has.
    pub(crate) fn normal(&self) -> [f64; 3] {
        self.normal
    }

    /// The canonical plane in this direction through `p`.
    pub(crate) fn through(&self, p: [f64; 3]) -> Option<WorldPlane> {
        let offset = dot(self.unit, p);
        if !offset.is_finite() {
            return None;
        }
        Some(WorldPlane {
            normal: self.normal,
            offset: round_to_grid(offset),
        })
    }
}

impl WorldPlane {
    /// The canonical plane with normal `n` (any length but zero) through `p`.
    pub(crate) fn canonical(n: [f64; 3], p: [f64; 3]) -> Option<WorldPlane> {
        Direction::of(n)?.through(p)
    }

    /// The plane through a triangle's corners, snapped to `grid`, or `None` for a degenerate
    /// triangle (its corners on a line, to within rounding and the grid: [`points_somewhere`]).
    pub fn of_triangle(p: [[f64; 3]; 3], grid: f64) -> Option<WorldPlane> {
        if let Some(axis) = shared_axis(&p, grid) {
            return Some(axis);
        }
        let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let v = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        if !points_somewhere(n, extent(&p), perimeter(&p), grid) {
            return None;
        }
        WorldPlane::canonical(n, p[0])
    }

    /// The plane the points lie on (Newell's normal, through their first point), the axis plane
    /// of a coordinate they all share when they lie on a line, or `None` when they are not on one
    /// plane. Newell's normal sums the outline's signed areas, so an outline crossing itself on a
    /// plane (a bowtie) can cancel it; the points then take a second estimate, the normal of the
    /// widest triangle on their first point, before they are found to lie on no plane.
    ///
    /// Points snapped to `grid` (each up to [`snap_reach`] `δ` from where content put it) lie off
    /// their plane by up to `2δ` against the first point, and their normal is off by the
    /// snapping's change to it, which Newell's sum bounds by `2δP` for the outline's perimeter
    /// `P`: through the first point, a point at distance `R` from it lies off the fitted plane by
    /// at most `2δ(1 + 2RP/|n|)`. Points within that bound (with [`FIT_MARGIN`]) are on the
    /// plane; content bent by more than it is not. Unsnapped points (`grid` zero) keep the bound
    /// relative to their extent, and snapped ones keep it where it is the wider.
    pub fn of_points(points: &[[f64; 3]], grid: f64) -> Option<WorldPlane> {
        let first = *points.first()?;
        if let Some(axis) = shared_axis(points, grid) {
            return Some(axis);
        }
        let extent = extent(points);
        if !(extent > 0.0 && extent.is_finite()) {
            return None;
        }
        let n = newell(points, first);
        let (perimeter, reach) = (perimeter(points), reach(points, first));
        // Whether every point lies on the plane with normal `n` through the first point.
        let holds = |n: [f64; 3]| {
            if !points_somewhere(n, extent, perimeter, grid) {
                return false;
            }
            let length = dot(n, n).sqrt();
            let fit = 2.0 * snap_reach(grid) * (1.0 + 2.0 * reach * perimeter / length);
            let tolerance = (PLANAR_TOLERANCE * extent.max(1.0)).max(FIT_MARGIN * fit);
            let unit = n.map(|v| v / length);
            let d0 = dot(unit, first);
            points
                .iter()
                .all(|p| (dot(unit, *p) - d0).abs() <= tolerance)
        };
        if holds(n) {
            return WorldPlane::canonical(n, first);
        }
        let widest = widest_normal(points, first);
        if holds(widest) {
            return WorldPlane::canonical(widest, first);
        }
        None // Not on one plane, or on a line with no coordinate shared: no plane of its own.
    }

    /// The plane the points lie on ([`WorldPlane::of_points`]), or, when they lie on none (a
    /// surface's face, curved a little), the plane nearest them: Newell's normal through their
    /// centroid. `None` for points on a line (to within rounding and the grid:
    /// [`points_somewhere`]) or a point.
    pub fn nearest(points: &[[f64; 3]], grid: f64) -> Option<WorldPlane> {
        if let Some(plane) = WorldPlane::of_points(points, grid) {
            return Some(plane);
        }
        let first = *points.first()?;
        let n = newell(points, first);
        if !points_somewhere(n, extent(points), perimeter(points), grid) {
            return None;
        }
        let mut sum = [0.0; 3];
        for p in points {
            for k in 0..3 {
                sum[k] += p[k] - first[k];
            }
        }
        let count = points.len() as f64;
        let centroid = std::array::from_fn(|k| first[k] + sum[k] / count);
        WorldPlane::canonical(n, centroid)
    }
}

/// The widest of the points' extents along the three axes.
fn extent(points: &[[f64; 3]]) -> f64 {
    (0..3)
        .map(|k| {
            let (lo, hi) = points
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                    (lo.min(p[k]), hi.max(p[k]))
                });
            hi - lo
        })
        .fold(0.0, f64::max)
}

/// Newell's normal of the outline through `points`, about `first` so it does not depend on where
/// the points are.
fn newell(points: &[[f64; 3]], first: [f64; 3]) -> [f64; 3] {
    let mut n = [0.0; 3];
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        let (a, b) = (
            [a[0] - first[0], a[1] - first[1], a[2] - first[2]],
            [b[0] - first[0], b[1] - first[1], b[2] - first[2]],
        );
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    n
}

/// Whether `n`, a normal found from points spanning `extent` around an outline of `perimeter`,
/// snapped to `grid`, is long enough to point anywhere. Points on a line (or a sum that
/// cancelled) give a normal of noise, whose direction an ulp in their input, or their snapping,
/// decides: none of their planes is theirs. Rounding's noise scales as the normal does, with the
/// square of the points' size; snapping's is at most `2δP` (Newell's sum moved by each point's
/// snap, [`WorldPlane::of_points`]), and the bound is twice that.
fn points_somewhere(n: [f64; 3], extent: f64, perimeter: f64, grid: f64) -> bool {
    let rounding = PLANAR_TOLERANCE * extent * extent;
    let snapping = 4.0 * snap_reach(grid) * perimeter;
    dot(n, n).sqrt() > rounding.max(snapping)
}

/// The normal of the widest triangle with a corner at `first`: its second corner the point
/// farthest from `first`, its third the point farthest from the line through those two.
fn widest_normal(points: &[[f64; 3]], first: [f64; 3]) -> [f64; 3] {
    let from_first = |p: &[f64; 3]| [p[0] - first[0], p[1] - first[1], p[2] - first[2]];
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let longest = |a: &[f64; 3], b: &[f64; 3]| dot(*a, *a).total_cmp(&dot(*b, *b));
    let far = points
        .iter()
        .map(from_first)
        .max_by(longest)
        .unwrap_or([0.0; 3]);
    points
        .iter()
        .map(|p| cross(far, from_first(p)))
        .max_by(longest)
        .unwrap_or([0.0; 3])
}

/// The axis plane of a coordinate every point shares, preferring z, then y, then x, at the first
/// point's value rounded to the grid. A coordinate is shared when every point lies within the
/// planarity tolerance of the first point's, or within a step of `grid` for snapped points (one
/// value snaps to one value, and values an ulp apart that straddle a grid boundary to two a step
/// apart): content places an edge between two vertices at one value, and arithmetic that differs
/// between machines in the last bit must not turn its plane into a centreline on one of them. The
/// boundary sits at that tolerance, which content reaches only by coincidence, against a fixed
/// reference (the first point), so the test is the same whatever the other points' order.
fn shared_axis(points: &[[f64; 3]], grid: f64) -> Option<WorldPlane> {
    let first = points.first()?;
    let tolerance = (PLANAR_TOLERANCE * extent(points).max(1.0)).max(grid);
    for k in [2, 1, 0] {
        if points.iter().all(|p| (p[k] - first[k]).abs() <= tolerance) {
            let mut normal = [0.0; 3];
            normal[k] = 1.0;
            return Some(WorldPlane {
                normal,
                offset: round_to_grid(first[k]),
            });
        }
    }
    None
}

/// A curve's depth on screen: segments of its screen polyline with a depth key at each end, binned
/// by the tiles they come near.
#[derive(Clone, Debug, Default)]
pub struct Centreline {
    segments: Vec<([f64; 2], f64, [f64; 2], f64)>,
}

impl Centreline {
    /// A centreline through screen points `(point, key)`, in order. Runs of points are joined;
    /// `breaks` lists the indices where a new run starts.
    pub fn new(points: &[([f64; 2], f64)], breaks: &[usize]) -> Centreline {
        let mut segments = Vec::new();
        for i in 1..points.len() {
            if breaks.contains(&i) {
                continue;
            }
            let (a, ka) = points[i - 1];
            let (b, kb) = points[i];
            if [a[0], a[1], b[0], b[1], ka, kb]
                .iter()
                .all(|v| v.is_finite())
            {
                segments.push((a, ka, b, kb));
            }
        }
        if segments.is_empty()
            && let Some(&(p, k)) = points
                .iter()
                .find(|(p, k)| p[0].is_finite() && p[1].is_finite() && k.is_finite())
        {
            segments.push((p, k, p, k));
        }
        Centreline { segments }
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The segments' screen bounds, each widened by `margin`, for binning.
    pub fn segment_bounds(&self, margin: f64) -> impl Iterator<Item = [f64; 4]> + '_ {
        self.segments.iter().map(move |(a, _, b, _)| {
            [
                a[0].min(b[0]) - margin,
                a[1].min(b[1]) - margin,
                a[0].max(b[0]) + margin,
                a[1].max(b[1]) + margin,
            ]
        })
    }

    /// The depth plane at pixel `(x, y)`, among the segments `candidates` (indices into the
    /// segments; all segments when empty): along the front-most of the segments within `reach` of
    /// the pixel's centre (the part of the curve the pixel shows, where parts of it overlap on
    /// screen), else along the nearest; the key interpolated along the segment.
    pub fn plane_at(&self, x: u32, y: u32, candidates: &[u32], reach: f64) -> Plane {
        let c = [x as f64 + 0.5, y as f64 + 0.5];
        let reach2 = reach * reach;
        // (outside reach, then the key there inside it or the distance outside it): least wins.
        let mut best: Option<(bool, f64, usize)> = None;
        let mut consider = |i: usize| {
            let (a, ka, b, kb) = self.segments[i];
            let (t, _) = project(a, b, c);
            let p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            let d = (p[0] - c[0]) * (p[0] - c[0]) + (p[1] - c[1]) * (p[1] - c[1]);
            let this = if d <= reach2 {
                (false, ka + (kb - ka) * t, i)
            } else {
                (true, d, i)
            };
            if best.is_none_or(|(out, v, _)| (this.0, this.1) < (out, v)) {
                best = Some(this);
            }
        };
        if candidates.is_empty() {
            (0..self.segments.len()).for_each(&mut consider);
        } else {
            candidates.iter().for_each(|i| consider(*i as usize));
        }
        let (_, _, i) = best.expect("a centreline has a segment");
        let (a, ka, b, kb) = self.segments[i];
        let (_, len2) = project(a, b, c);
        if len2 == 0.0 || ka == kb {
            return Plane::constant(ka.min(kb));
        }
        // key(p) = ka + (kb - ka) * ((p - a) . (b - a)) / |b - a|^2, an affine function of p.
        let g = (kb - ka) / len2;
        let (bx, by) = (g * (b[0] - a[0]), g * (b[1] - a[1]));
        Plane {
            a: ka - bx * a[0] - by * a[1],
            b: bx,
            c: by,
        }
    }
}

/// The parameter in `0..=1` of the point of segment `a`-`b` nearest `p`, and the segment's squared
/// length.
fn project(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> (f64, f64) {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return (0.0, 0.0);
    }
    (
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0),
        len2,
    )
}
