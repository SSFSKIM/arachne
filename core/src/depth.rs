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
//! - **Whether** a path is flat ([`Planarity`]) is decided apart from **fitting** its plane: a
//!   writer that snaps geometry to a grid decides on the values as made and carries the decision
//!   (XF's `planes`), since snapped points of a slightly curved face and of a flat one can be the
//!   same points; the drawer fits the decided plane to the snapped points.
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

/// How far, relative to the points' extent, points may lie off the plane they define.
const PLANAR_TOLERANCE: f64 = 1e-9;

/// The grid the view's geometry was snapped to, in scene units (`0.0`: not snapped).
pub(crate) fn input_grid(view: &crate::xf::View) -> f64 {
    view.input_grid
}

/// How far a point snapped to `grid` may lie from where it was: half a step in each coordinate.
fn snap_reach(grid: f64) -> f64 {
    grid * 0.75f64.sqrt()
}

/// The outline's perimeter, closed.
fn perimeter(points: &[[f64; 3]]) -> f64 {
    (0..points.len())
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            dot(d, d).sqrt()
        })
        .sum()
}

/// How a path's points lie, as [`WorldPlane::decide`] finds it from the values content gave them;
/// [`WorldPlane::fit`] then fits that plane to the points as drawn. Snapped to a grid, points of a
/// curve and points of a plane can be the same points, so the decision is made before the snap
/// and carried with the record (XF's `planes`), and the grid never decides whether something is
/// flat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Planarity {
    /// On no plane of its own: keyed by its curve.
    Curve,
    /// On a plane: Newell's normal, through the first point.
    Newell,
    /// On a plane whose outline crosses itself, cancelling Newell's normal: the widest triangle's
    /// normal on the first point, through it.
    Widest,
    /// A surface's face on no one plane: the plane nearest it, Newell's normal through the
    /// points' centroid.
    Nearest,
    /// On a line sharing coordinate `k` (0 x, 1 y, 2 z): that axis plane, at the first point's
    /// value.
    Axis(u8),
}

impl Planarity {
    /// The decision's code in XF's `planes`.
    pub fn code(self) -> u8 {
        match self {
            Planarity::Curve => 0,
            Planarity::Newell => 1,
            Planarity::Widest => 2,
            Planarity::Nearest => 3,
            Planarity::Axis(k) => 4 + k,
        }
    }

    /// The decision a code in XF's `planes` names, or `None` for a code it does not.
    pub fn of_code(code: u8) -> Option<Planarity> {
        Some(match code {
            0 => Planarity::Curve,
            1 => Planarity::Newell,
            2 => Planarity::Widest,
            3 => Planarity::Nearest,
            4..=6 => Planarity::Axis(code - 4),
            _ => return None,
        })
    }
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

    /// The plane through a triangle's corners, snapped to `grid` (`0.0`: not snapped), or `None`
    /// for a degenerate triangle (its corners on a line, to within rounding and the grid:
    /// [`points_somewhere`]). A triangle the snap leaves under about a grid step wide is found
    /// degenerate and takes its depth from its own corners on screen instead: exact for a
    /// triangle, and only coplanar ties hang on the world plane.
    pub fn of_triangle(p: [[f64; 3]; 3], grid: f64) -> Option<WorldPlane> {
        if let Some(k) = shared_axis(&p, grid) {
            return Some(axis_plane(k, p[0]));
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

    /// How the points lie ([`Planarity`]): on the axis plane of a coordinate they all share when
    /// they lie on a line; on the plane of Newell's normal through their first point; else, as
    /// Newell's sum of the outline's signed areas cancels for an outline crossing itself on a
    /// plane (a bowtie), on the plane of the widest triangle on their first point; else, for a
    /// surface's face, on the plane nearest them; else on none. A point lies on a plane within
    /// [`PLANAR_TOLERANCE`] of the points' extent.
    pub fn decide(points: &[[f64; 3]], surface: bool) -> Planarity {
        let Some(&first) = points.first() else {
            return Planarity::Curve;
        };
        if let Some(k) = shared_axis(points, 0.0) {
            return Planarity::Axis(k);
        }
        let extent = extent(points);
        if !(extent > 0.0 && extent.is_finite()) {
            return Planarity::Curve;
        }
        let tolerance = PLANAR_TOLERANCE * extent.max(1.0);
        // Whether every point lies on the plane with normal `n` through the first point.
        let holds = |n: [f64; 3]| {
            if !points_somewhere(n, extent, 0.0, 0.0) {
                return false;
            }
            let unit = n.map(|v| v / dot(n, n).sqrt());
            let d0 = dot(unit, first);
            points
                .iter()
                .all(|p| (dot(unit, *p) - d0).abs() <= tolerance)
        };
        let n = newell(points, first);
        if holds(n) {
            Planarity::Newell
        } else if holds(widest_normal(points, first)) {
            Planarity::Widest
        } else if surface && points_somewhere(n, extent, 0.0, 0.0) {
            Planarity::Nearest
        } else {
            Planarity::Curve // Not on one plane, or on a line with no coordinate shared.
        }
    }

    /// The canonical plane `planarity` puts the points on, fitted to them as they are (snapped,
    /// for the core's input): `None` for a curve, or for a fit whose normal is zero.
    pub fn fit(points: &[[f64; 3]], planarity: Planarity) -> Option<WorldPlane> {
        let first = *points.first()?;
        match planarity {
            Planarity::Curve => None,
            Planarity::Newell => WorldPlane::canonical(newell(points, first), first),
            Planarity::Widest => WorldPlane::canonical(widest_normal(points, first), first),
            Planarity::Nearest => {
                let mut sum = [0.0; 3];
                for p in points {
                    for k in 0..3 {
                        sum[k] += p[k] - first[k];
                    }
                }
                let count = points.len() as f64;
                let centroid = std::array::from_fn(|k| first[k] + sum[k] / count);
                WorldPlane::canonical(newell(points, first), centroid)
            }
            Planarity::Axis(k) => Some(axis_plane(k, first)),
        }
    }

    /// The plane the points lie on, decided and fitted from these points
    /// ([`WorldPlane::decide`], [`WorldPlane::fit`]), or `None` when they are not on one plane.
    pub fn of_points(points: &[[f64; 3]]) -> Option<WorldPlane> {
        WorldPlane::fit(points, WorldPlane::decide(points, false))
    }

    /// The plane the points lie on ([`WorldPlane::of_points`]), or, when they lie on none (a
    /// surface's face, curved a little), the plane nearest them: Newell's normal through their
    /// centroid. `None` for points on a line (to within rounding: [`points_somewhere`]) or a
    /// point.
    pub fn nearest(points: &[[f64; 3]]) -> Option<WorldPlane> {
        WorldPlane::fit(points, WorldPlane::decide(points, true))
    }
}

/// The bucket sides of the registry's lookup: in unit-normal components, and in scene units of
/// offset. A lookup takes the bucket and its neighbours, so a plane is found from a fit off by up
/// to a side; a candidate is taken only when the points lie on it ([`Registry::lies_on`]).
const NORMAL_BUCKET: f64 = 1.0 / 64.0;
const OFFSET_BUCKET: f64 = 0.5;

/// How many snap reaches a point may lie off a registered plane and still be on it: its own
/// snap's, the snap of the points the plane was fitted to, and their fit's tilt across the record.
const ON_PLANE_REACHES: f64 = 6.0;

/// The planes of a frame's flat records, in the order they were registered, each with the points
/// of the record that registered it. A flat record on a plane registered before it takes that
/// plane, bit for bit, so records on one plane tie by draw order: the planes fitted to their
/// snapped points differ by the snap's noise, about the grid over their size, and would cross.
///
/// Two records are on one plane when either's points lie on the other's plane: each point within
/// [`ON_PLANE_REACHES`] snap reaches of it (and the planes' rounding to [`PLANE_GRID`]). Either
/// way, since a small record's plane tilts by the snap's noise over its own size, and a larger
/// record's far points leave it by that tilt over their distance; the larger record's plane holds
/// the small one's points. The first plane registered that a record is on wins.
#[derive(Debug, Default)]
pub(crate) struct Registry {
    grid: f64,
    planes: Vec<(WorldPlane, Vec<[f64; 3]>)>,
    buckets: std::collections::HashMap<[i64; 4], Vec<usize>>,
}

impl Registry {
    /// An empty registry for points snapped to `grid` (`0.0`: not snapped).
    pub(crate) fn new(grid: f64) -> Registry {
        Registry {
            grid,
            ..Registry::default()
        }
    }

    fn bucket(plane: &WorldPlane) -> [i64; 4] {
        let n = plane.normal;
        [
            (n[0] / NORMAL_BUCKET).floor() as i64,
            (n[1] / NORMAL_BUCKET).floor() as i64,
            (n[2] / NORMAL_BUCKET).floor() as i64,
            (plane.offset / OFFSET_BUCKET).floor() as i64,
        ]
    }

    /// Whether every point lies on `plane`.
    fn lies_on(&self, plane: &WorldPlane, points: &[[f64; 3]]) -> bool {
        let largest = points.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        let tolerance = ON_PLANE_REACHES * snap_reach(self.grid)
            + PLANE_GRID * (1.0 + 3.0 * largest)
            + PLANAR_TOLERANCE * extent(points).max(1.0);
        points
            .iter()
            .all(|p| (dot(plane.normal, *p) - plane.offset).abs() <= tolerance)
    }

    /// The plane a flat record takes, `own` the plane fitted to its points: the first plane
    /// registered near `own` (the same, or turned the other way: the canonical normal's largest
    /// component can flip sign between two fits of one plane) that it is on, else `own`,
    /// registered.
    pub(crate) fn merge(&mut self, own: WorldPlane, points: &[[f64; 3]]) -> WorldPlane {
        let mut first: Option<usize> = None;
        let turned = WorldPlane {
            normal: own.normal.map(|v| -v),
            offset: -own.offset,
        };
        for near in [own, turned] {
            let key = Registry::bucket(&near);
            for d in 0..81i64 {
                let step = [d % 3 - 1, d / 3 % 3 - 1, d / 9 % 3 - 1, d / 27 - 1];
                let at = std::array::from_fn(|k| key[k] + step[k]);
                for &i in self.buckets.get(&at).into_iter().flatten() {
                    let (plane, theirs) = &self.planes[i];
                    if first.is_none_or(|f| i < f)
                        && (self.lies_on(plane, points) || self.lies_on(&own, theirs))
                    {
                        first = Some(i);
                    }
                }
            }
        }
        match first {
            Some(i) => self.planes[i].0,
            None => {
                self.buckets
                    .entry(Registry::bucket(&own))
                    .or_default()
                    .push(self.planes.len());
                self.planes.push((own, points.to_vec()));
                own
            }
        }
    }

    /// The first plane registered that every point lies on: the plane a record with no plane of
    /// its own (a line, a curve) lying on a flat record's plane is keyed on.
    pub(crate) fn containing(&self, points: &[[f64; 3]]) -> Option<WorldPlane> {
        self.planes
            .iter()
            .map(|(plane, _)| *plane)
            .find(|plane| self.lies_on(plane, points))
    }
}

/// The axis plane of coordinate `k` through `p`, its offset rounded to the grid.
fn axis_plane(k: u8, p: [f64; 3]) -> WorldPlane {
    let mut normal = [0.0; 3];
    normal[k as usize] = 1.0;
    WorldPlane {
        normal,
        offset: round_to_grid(p[k as usize]),
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
pub(crate) fn newell(points: &[[f64; 3]], first: [f64; 3]) -> [f64; 3] {
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
/// snapped to `grid` (`0.0`: not snapped), is long enough to point anywhere. Points on a line (or a
/// sum that cancelled) give a normal of noise, whose direction an ulp in their input, or their
/// snapping, decides: none of their planes is theirs. Rounding's noise scales as the normal does,
/// with the square of the points' size; snapping's is at most `2δP` for points each moved up to
/// `δ` ([`snap_reach`]: Newell's sum, or a cross product, moved by each point's snap), and the
/// bound is twice that.
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

/// The coordinate every point shares, preferring z, then y, then x. A coordinate is shared when
/// every point lies within the planarity tolerance of the first point's, or within a step of
/// `grid` for snapped points (one value snaps to one value, and values an ulp apart that straddle
/// a grid boundary to two a step apart): content places an edge between two vertices at one value,
/// and arithmetic that differs between machines in the last bit must not turn its plane into a
/// centreline on one of them. The boundary sits at that tolerance, which content reaches only by
/// coincidence, against a fixed reference (the first point), so the test is the same whatever the
/// other points' order.
fn shared_axis(points: &[[f64; 3]], grid: f64) -> Option<u8> {
    let first = points.first()?;
    let tolerance = (PLANAR_TOLERANCE * extent(points).max(1.0)).max(grid);
    [2u8, 1, 0].into_iter().find(|&k| {
        points
            .iter()
            .all(|p| (p[k as usize] - first[k as usize]).abs() <= tolerance)
    })
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
