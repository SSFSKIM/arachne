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

use crate::composite::{Plane, U};

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

/// A plane in world space, canonical: `normal . p = offset`, and the plane it was rounded from.
/// Two planes are equal when their rounded numbers are, however each was found.
#[derive(Clone, Copy, Debug)]
pub struct WorldPlane {
    pub normal: [f64; 3],
    pub offset: f64,
    /// The plane before its rounding to the grid: a depth key across this plane is held to the
    /// key across that one (`Projection::key_plane`).
    pub unrounded: Unrounded,
}

impl PartialEq for WorldPlane {
    fn eq(&self, other: &WorldPlane) -> bool {
        self.normal == other.normal && self.offset == other.offset
    }
}

/// A world plane before its rounding to the grid: the normal it was found with, the offset found
/// from it, and how far that offset may lie from the exact offset, along that normal, of the point
/// it was found through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unrounded {
    pub normal: [f64; 3],
    pub offset: f64,
    pub offset_err: f64,
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

    /// The unit normal before its rounding.
    pub(crate) fn unit(&self) -> [f64; 3] {
        self.unit
    }

    /// The canonical plane in this direction through `p`: its offset is `unit . p`, which three
    /// products and two sums round by at most `3u` of `|unit| . |p|` (taken as `4u`).
    pub(crate) fn through(&self, p: [f64; 3]) -> Option<WorldPlane> {
        let offset = dot(self.unit, p);
        if !offset.is_finite() {
            return None;
        }
        let size = dot(self.unit.map(f64::abs), p.map(f64::abs));
        Some(WorldPlane::rounded(self.unit, offset, 4.0 * U * size))
    }
}

impl WorldPlane {
    /// The plane `normal . p = offset` with its normal and offset rounded to [`PLANE_GRID`];
    /// `offset_err` bounds how far `offset` lies from the exact offset.
    pub fn rounded(normal: [f64; 3], offset: f64, offset_err: f64) -> WorldPlane {
        WorldPlane {
            normal: normal.map(round_to_grid),
            offset: round_to_grid(offset),
            unrounded: Unrounded {
                normal,
                offset,
                offset_err,
            },
        }
    }

    /// The same plane facing the other way.
    fn turned(&self) -> WorldPlane {
        let u = self.unrounded;
        WorldPlane {
            normal: self.normal.map(|v| -v),
            offset: -self.offset,
            unrounded: Unrounded {
                normal: u.normal.map(|v| -v),
                offset: -u.offset,
                ..u
            },
        }
    }

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

/// The planes of a frame's mesh triangles and flat paths, in the order they were registered, each
/// with the points that registered it and identified from 1 by that order ([`Fragment::plane`]):
/// the meshes' triangles first (`mesh::lower`), each its own facet's plane, then the paths
/// (`path::Lowering::coplanar`), in draw order. One plane has one identity whichever producer drew
/// what lies on it, and two fragments of one identity keep paint order (`order::one_plane`), so
/// things on one plane tie by draw order: the planes fitted to their snapped points differ by the
/// snap's noise, about the grid over their size, and would cross. A path on a plane registered
/// before it also takes that plane, bit for bit; a triangle keeps its own.
///
/// Two records are on one plane when either's points lie on the other's plane: each point within
/// [`ON_PLANE_REACHES`] snap reaches of it (and the planes' rounding to [`PLANE_GRID`]). Either
/// way, since a small record's plane tilts by the snap's noise over its own size, and a larger
/// record's far points leave it by that tilt over their distance; the larger record's plane holds
/// the small one's points. The first plane registered that a record is on wins.
///
/// [`Fragment::plane`]: crate::composite::Fragment::plane
#[derive(Debug, Default)]
pub struct Registry {
    grid: f64,
    planes: Vec<(WorldPlane, Vec<[f64; 3]>)>,
    buckets: std::collections::HashMap<[i64; 4], Vec<usize>>,
}

impl Registry {
    /// An empty registry for a frame of `view`, whose points are snapped to its grid.
    pub fn new(view: &crate::xf::View) -> Registry {
        Registry {
            grid: input_grid(view),
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

    /// How far the points may lie off a plane and still be on it.
    fn tolerance(&self, points: &[[f64; 3]]) -> f64 {
        let largest = points.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        ON_PLANE_REACHES * snap_reach(self.grid)
            + PLANE_GRID * (1.0 + 3.0 * largest)
            + PLANAR_TOLERANCE * extent(points).max(1.0)
    }

    /// Whether every point lies within `tolerance` of `plane`.
    fn within(plane: &WorldPlane, points: &[[f64; 3]], tolerance: f64) -> bool {
        points
            .iter()
            .all(|p| (dot(plane.normal, *p) - plane.offset).abs() <= tolerance)
    }

    /// Whether every point lies on `plane`.
    fn lies_on(&self, plane: &WorldPlane, points: &[[f64; 3]]) -> bool {
        Registry::within(plane, points, self.tolerance(points))
    }

    /// The plane a flat record takes and its identity, `own` the plane fitted to its points: the
    /// first plane registered near `own` (the same, or turned the other way: the canonical
    /// normal's largest component can flip sign between two fits of one plane) that it is on,
    /// else `own`, registered.
    pub(crate) fn merge(&mut self, own: WorldPlane, points: &[[f64; 3]]) -> (WorldPlane, u32) {
        let tolerance = self.tolerance(points);
        let mut first: Option<usize> = None;
        for near in [own, own.turned()] {
            let key = Registry::bucket(&near);
            for d in 0..81i64 {
                let step = [d % 3 - 1, d / 3 % 3 - 1, d / 9 % 3 - 1, d / 27 - 1];
                let at = std::array::from_fn(|k| key[k] + step[k]);
                for &i in self.buckets.get(&at).into_iter().flatten() {
                    let (plane, theirs) = &self.planes[i];
                    if first.is_none_or(|f| i < f)
                        && (Registry::within(plane, points, tolerance)
                            || self.lies_on(&own, theirs))
                    {
                        first = Some(i);
                    }
                }
            }
        }
        let i = first.unwrap_or_else(|| {
            self.buckets
                .entry(Registry::bucket(&own))
                .or_default()
                .push(self.planes.len());
            self.planes.push((own, points.to_vec()));
            self.planes.len() - 1
        });
        (self.planes[i].0, i as u32 + 1)
    }

    /// The first plane registered that every point lies on, and its identity: the plane a record
    /// with no plane of its own (a line, a curve) lying on a flat record's or a triangle's plane
    /// is keyed on.
    pub(crate) fn containing(&self, points: &[[f64; 3]]) -> Option<(WorldPlane, u32)> {
        let tolerance = self.tolerance(points);
        let i = self
            .planes
            .iter()
            .position(|(plane, _)| Registry::within(plane, points, tolerance))?;
        Some((self.planes[i].0, i as u32 + 1))
    }
}

/// The axis plane of coordinate `k` through `p`, its offset rounded to the grid (the axis and the
/// coordinate exact before it).
fn axis_plane(k: u8, p: [f64; 3]) -> WorldPlane {
    let mut normal = [0.0; 3];
    normal[k as usize] = 1.0;
    WorldPlane::rounded(normal, p[k as usize], 0.0)
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
/// bound is twice that. Public for the binding's `_points_somewhere` and its tests.
pub fn points_somewhere(n: [f64; 3], extent: f64, perimeter: f64, grid: f64) -> bool {
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
    /// Each segment's ends' keys' greater error.
    errors: Vec<f64>,
    /// The largest magnitude of a coordinate of a segment's end.
    scale: f64,
}

impl Centreline {
    /// A centreline through screen points `(point, key)`, in order. Runs of points are joined;
    /// `breaks` lists the indices where a new run starts.
    pub fn new(points: &[([f64; 2], f64)], breaks: &[usize]) -> Centreline {
        Centreline::with_errors(points, &[], breaks)
    }

    /// [`Centreline::new`] with each point's key known within `errors[i]` (none given: exact).
    pub fn with_errors(points: &[([f64; 2], f64)], errors: &[f64], breaks: &[usize]) -> Centreline {
        let err = |i: usize| errors.get(i).copied().unwrap_or(0.0);
        let mut segments = Vec::new();
        let mut errs = Vec::new();
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
                errs.push(err(i - 1).max(err(i)));
            }
        }
        if segments.is_empty()
            && let Some(i) = points
                .iter()
                .position(|(p, k)| p[0].is_finite() && p[1].is_finite() && k.is_finite())
        {
            let (p, k) = points[i];
            segments.push((p, k, p, k));
            errs.push(err(i));
        }
        let scale = segments
            .iter()
            .flat_map(|(a, _, b, _)| [a[0], a[1], b[0], b[1]])
            .fold(0.0, |m: f64, v| m.max(v.abs()));
        Centreline {
            segments,
            errors: errs,
            scale,
        }
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

    /// The squared distance from `p` to segment `i`, as [`Centreline::key_under`] measures it.
    pub fn distance2(&self, i: usize, p: [f64; 2]) -> f64 {
        count!(FEET);
        let (a, _, b, _) = self.segments[i];
        foot(a, b, p).1
    }

    /// The squared distance from segment `i`'s screen bounds to the box `area` (`[x0, y0, x1,
    /// y1]`): nothing of the segment lies nearer any point of the box than its root.
    pub fn gap2(&self, i: usize, area: [f64; 4]) -> f64 {
        count!(GAPS);
        let (a, _, b, _) = self.segments[i];
        let gx = (a[0].min(b[0]) - area[2])
            .max(area[0] - a[0].max(b[0]))
            .max(0.0);
        let gy = (a[1].min(b[1]) - area[3])
            .max(area[1] - a[1].max(b[1]))
            .max(0.0);
        gx * gx + gy * gy
    }

    /// Segment `i`'s screen bounds, `[x0, y0, x1, y1]`.
    pub fn segment_box(&self, i: usize) -> [f64; 4] {
        let (a, _, b, _) = self.segments[i];
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[0].max(b[0]),
            a[1].max(b[1]),
        ]
    }

    /// The number of segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// How far the distances [`Centreline::key_under`] compares may stand from the exact
    /// distances they round, where every coordinate, distance and body compared is at most
    /// `extent` (the segments' own coordinates are taken in): `2^-46` of the largest magnitude.
    /// A projection rounds within `17u` of the largest magnitude `S` of its coordinates (its
    /// direction, step and offset, each a rounding of a sum within `3S`), its squared length and
    /// root within `1.5u` of their size, a parameter moved by rounding within `12uS` of the
    /// nearest point: a read's distance lies within `34uS` of the exact one, a box's gap (the
    /// root of its square) within `9uS`, a root and the sums compared within `6uS` more; a decision that holds by `128uS` holds
    /// for the rounded distances as for the exact ones.
    pub fn rounding(&self, extent: f64) -> f64 {
        128.0 * U * self.scale.max(extent).max(1.0)
    }

    /// The key at point `p` of the part of the curve that covers it, among the segments
    /// `candidates` (indices into the segments; all segments when empty), with how far it may
    /// stand from the exact key of the curve's inputs (its points' keys, within their errors): the
    /// front-most of the segments whose footprint holds `p` (its projection falls between the
    /// segment's ends) and whose stroke body holds it (`p` within `body` of the projection), keyed
    /// at the projection; where no segment's body holds it (a cap, a join, the antialiasing fringe
    /// beyond the body, a fill), the key of the curve's nearest point. Either way a key between the
    /// two ends' keys of the segment it is read on, so no point takes a key the curve's part there
    /// does not. The front-most of several is their least key, which moves by no more than the
    /// largest of their errors.
    pub fn key_under(&self, p: [f64; 2], candidates: &[u32], body: f64) -> (f64, f64) {
        let found = self.find(p, candidates, body);
        found.front.unwrap_or_else(|| {
            let (_, i, raw) = found.nearest.expect("a centreline has a segment");
            let (a, ka, b, kb) = self.segments[i];
            key_on((a, ka), (b, kb), self.errors[i], p, raw)
        })
    }

    /// What [`Centreline::key_under`] finds at `p` among `candidates`, before it keys the
    /// nearest segment.
    fn find(&self, p: [f64; 2], candidates: &[u32], body: f64) -> Found {
        count!(KEY_UNDER);
        let body2 = body * body;
        // The front-most key among the segments holding p with the largest error among them; the
        // nearest segment's (distance, index, projection), keyed only if no segment holds p.
        let mut front: Option<(f64, f64)> = None;
        let mut nearest: Option<(f64, usize, f64)> = None;
        let mut consider = |i: usize| {
            let (a, ka, b, kb) = self.segments[i];
            count!(FEET);
            let (raw, d2) = foot(a, b, p);
            if (0.0..=1.0).contains(&raw) && d2 <= body2 {
                let (key, err) = key_on((a, ka), (b, kb), self.errors[i], p, raw);
                front = Some(match front {
                    Some((f, e)) => (f.min(key), e.max(err)),
                    None => (key, err),
                });
            }
            if nearest.is_none_or(|(n, _, _)| d2 < n) {
                nearest = Some((d2, i, raw));
            }
        };
        if candidates.is_empty() {
            count!(most MOST_FEET, self.segments.len());
            (0..self.segments.len()).for_each(&mut consider);
        } else {
            count!(most MOST_FEET, candidates.len());
            candidates.iter().for_each(|i| consider(*i as usize));
        }
        Found { front, nearest }
    }

    /// [`Centreline::key_under`] at each of `points` into `out`, among `candidates` (all
    /// segments when empty), and the largest of their bounds; `centre` is one of the points. The
    /// reads are made among the candidates `candidates` is cut down to: those whose bounds lie
    /// within `reach` of the points' box, the further of `body` and the distance of the centre's
    /// nearest candidate plus how far the furthest point lies from the centre. A point's nearest
    /// segment lies no further from it than the centre's nearest does, so within `reach`; a
    /// segment holding it in its body lies within `body`; a segment further than `reach` from
    /// every point is neither for any point, and dropping it changes no read: each point's front
    /// key and bound fold over the same segments in the same order, and its nearest is the same
    /// (a dropped segment lies strictly further than it, so it would not have displaced it,
    /// whatever its place in the order). Each comparison holds with room for its rounding
    /// ([`Centreline::rounding`]).
    pub fn keys_under(
        &self,
        centre: [f64; 2],
        points: &[[f64; 2]],
        candidates: &mut Vec<u32>,
        body: f64,
        out: &mut [f64],
    ) -> f64 {
        self.cut(centre, points, candidates, body);
        self.keys_among(points, candidates, body, out)
    }

    /// [`Centreline::key_under`] at each of `points` into `out`, among `kept`, and the largest of
    /// their bounds: [`Centreline::keys_under`]'s reads, its candidates already cut.
    pub fn keys_among(&self, points: &[[f64; 2]], kept: &[u32], body: f64, out: &mut [f64]) -> f64 {
        // One bound for every point, as a plane's is: the largest of theirs.
        let mut err = 0.0f64;
        for (p, d) in points.iter().zip(out.iter_mut()) {
            let (key, e) = self.key_under(*p, kept, body);
            *d = key;
            err = err.max(e);
        }
        err
    }

    /// Cuts `candidates` (all segments when empty) down to the ones [`Centreline::keys_under`]
    /// reads `points` among, `centre` among them. The centre's nearest candidate is found among
    /// those whose bounds lie within the nearest distance found so far, the candidate whose bounds
    /// lie nearest it found first: one whose bounds lie further cannot be nearer, so the least
    /// distance is the least of all the candidates'.
    pub fn cut(&self, centre: [f64; 2], points: &[[f64; 2]], candidates: &mut Vec<u32>, body: f64) {
        if candidates.is_empty() {
            candidates.extend(0..self.segments.len() as u32);
        }
        let at = [centre[0], centre[1], centre[0], centre[1]];
        let room = self.rounding(centre[0].abs().max(centre[1].abs()));
        let first = candidates
            .iter()
            .map(|&i| (self.gap2(i as usize, at), i))
            .fold((f64::INFINITY, candidates[0]), |m, g| {
                if g.0 < m.0 { g } else { m }
            })
            .1;
        let mut nearest = self.distance2(first as usize, centre);
        for &i in candidates.iter() {
            let within = nearest.sqrt() + room;
            if self.gap2(i as usize, at) <= within * within {
                nearest = nearest.min(self.distance2(i as usize, centre));
            }
        }
        let (area, limit) = self.reach(centre, points, nearest, body);
        candidates.retain(|&i| self.gap2(i as usize, area) <= limit * limit);
    }

    /// The box of `points` and `centre`, and how far from it [`Centreline::cut`] keeps a candidate's
    /// bounds, given `nearest2`, the least squared distance from `centre` to a candidate: the
    /// further of `body` and the centre's nearest distance plus how far the furthest point lies
    /// from the centre, with room for rounding.
    pub fn reach(
        &self,
        centre: [f64; 2],
        points: &[[f64; 2]],
        nearest2: f64,
        body: f64,
    ) -> ([f64; 4], f64) {
        let nearest = nearest2.sqrt();
        let mut area = [centre[0], centre[1], centre[0], centre[1]];
        let mut spread = 0.0f64;
        for p in points {
            area = [
                area[0].min(p[0]),
                area[1].min(p[1]),
                area[2].max(p[0]),
                area[3].max(p[1]),
            ];
            spread = spread.max(libm::hypot(p[0] - centre[0], p[1] - centre[1]));
        }
        let reach = (nearest + spread).max(body);
        let extent = area.iter().fold(reach, |m, v| m.max(v.abs()));
        (area, reach + self.rounding(extent))
    }
}

/// What a read along a curve finds at a point: the front-most key among the segments holding the
/// point in their bodies with the largest error among them, and the nearest segment's squared
/// distance, index and projection.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Found {
    front: Option<(f64, f64)>,
    nearest: Option<(f64, usize, f64)>,
}

/// The key at `p` of the segment from `a` to `b` (each a point and its key, the keys within `ends`
/// of their exact values), `raw` the parameter of `p`'s projection onto it, and how far that key
/// may stand from the exact key there: where `p` projects inside the segment, as the plane the key
/// runs along through the segment reads it at `p` (`ka + g (p - a) . (b - a)`, summed as
/// [`Plane::at`] sums it); past an end, that end's key; held within the two keys either way.
/// Inside, the key is a convex combination of the ends', so their errors carry over at most once;
/// the gradient rounds by about `4u` of the change along it, times how far along the segment `p`
/// lies, the plane's `a` by `3u` of its terms, and reading it at `p` by `4u` of its terms. A
/// segment of no length, or of one key, is its lesser key, which may stand from the exact key
/// anywhere along it by the keys' spread.
fn key_on(
    (a, ka): ([f64; 2], f64),
    (b, kb): ([f64; 2], f64),
    ends: f64,
    p: [f64; 2],
    raw: f64,
) -> (f64, f64) {
    count!(KEY_ON);
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 || ka == kb {
        return (ka.min(kb), ends + (ka - kb).abs());
    }
    if raw < 0.0 {
        return (ka, ends);
    }
    if raw > 1.0 {
        return (kb, ends);
    }
    let g = (kb - ka) / len2;
    let (bx, by) = (g * dx, g * dy);
    let plane = Plane::new(ka - bx * a[0] - by * a[1], bx, by);
    let along = libm::hypot(p[0] - a[0], p[1] - a[1]) / len2.sqrt();
    let err = ends
        + 4.0 * U * (kb - ka).abs() * (1.0 + along)
        + 3.0 * U * (ka.abs() + (bx * a[0]).abs() + (by * a[1]).abs())
        + plane.read_error(p[0], p[1]);
    (plane.at(p[0], p[1]).clamp(ka.min(kb), ka.max(kb)), err)
}

/// Point `p` against segment `a`-`b`: the raw parameter of its projection onto the segment's line
/// (`0` for a segment of no length), and the squared distance from `p` to the segment's nearest
/// point.
fn foot(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> (f64, f64) {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let raw = if len2 == 0.0 {
        0.0
    } else {
        ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2
    };
    let t = raw.clamp(0.0, 1.0);
    let (ex, ey) = (a[0] + dx * t - p[0], a[1] + dy * t - p[1]);
    (raw, ex * ex + ey * ey)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::{SAMPLES, sample_point};

    /// A small seeded generator (xorshift64*), so the cases are the same on every run.
    struct Seeded(u64);

    impl Seeded {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn pick<T: Copy>(&mut self, from: &[T]) -> T {
            from[self.below(from.len() as u64) as usize]
        }
    }

    type Bits = (Option<(u64, u64)>, Option<(u64, usize, u64)>);

    fn bits(f: &Found) -> Bits {
        (
            f.front.map(|(k, e)| (k.to_bits(), e.to_bits())),
            f.nearest.map(|(d, i, r)| (d.to_bits(), i, r.to_bits())),
        )
    }

    /// A curve of a few runs of points on a coarse grid (so distances tie exactly and segments
    /// repeat, cross and shrink to points), some far from the origin, some keys equal.
    fn curve(s: &mut Seeded) -> Centreline {
        let offset = s.pick(&[0.0, 0.0, 1000.0, 4096.5, 65536.0]);
        let step = s.pick(&[1.0, 0.5, 0.25, 0.1, 3.0]);
        let n = 1 + s.below(14) as usize;
        let mut points: Vec<([f64; 2], f64)> = Vec::new();
        for i in 0..n {
            let at = if i > 0 && s.below(5) == 0 {
                points[i - 1]
            } else {
                let x = offset + step * (s.below(17) as f64 - 4.0);
                let y = offset + step * (s.below(17) as f64 - 4.0);
                let key = s.pick(&[-0.5, -0.5, 0.25, 0.0, 1.0]) + s.below(3) as f64 * 0.125;
                ([x, y], key)
            };
            points.push(at);
        }
        let breaks: Vec<usize> = (1..n).filter(|_| s.below(6) == 0).collect();
        let errors: Vec<f64> = (0..n).map(|_| s.pick(&[0.0, 1e-9, 0.5])).collect();
        Centreline::with_errors(&points, &errors, &breaks)
    }

    #[test]
    fn cutting_the_candidates_down_changes_no_read() {
        let mut s = Seeded(0x5eed_2026_1107);
        let (mut cases, mut cut) = (0u64, 0u64);
        while cases < 20_000 {
            let line = curve(&mut s);
            if line.is_empty() {
                continue;
            }
            cases += 1;
            let segments = line.segments.len() as u32;
            let all: Vec<u32> = (0..segments).collect();
            let given: Vec<u32> = if s.below(3) == 0 {
                Vec::new()
            } else {
                let some: Vec<u32> = all.iter().copied().filter(|_| s.below(3) != 0).collect();
                if some.is_empty() { all.clone() } else { some }
            };
            let among = if given.is_empty() { &all } else { &given };
            let base = line.segments[s.below(segments as u64) as usize].0;
            let x = (base[0] + s.below(9) as f64 - 4.0).floor().max(0.0) as u32;
            let y = (base[1] + s.below(9) as f64 - 4.0).floor().max(0.0) as u32;
            let body = s.pick(&[0.0, 0.1, 0.5, 1.0, 2.5]);
            let mut points = [[x as f64 + 0.5, y as f64 + 0.5]; SAMPLES + 1];
            for (k, p) in points.iter_mut().take(SAMPLES).enumerate() {
                *p = sample_point(x, y, k);
            }
            let mut kept = given.clone();
            let mut out = [0.0; SAMPLES + 1];
            let err = line.keys_under(points[SAMPLES], &points, &mut kept, body, &mut out);
            assert!(!kept.is_empty() && kept.windows(2).all(|w| w[0] < w[1]));
            cut += (among.len() - kept.len()) as u64;
            let mut whole = 0.0f64;
            for (k, p) in points.iter().enumerate() {
                let (a, b) = (line.find(*p, among, body), line.find(*p, &kept, body));
                assert_eq!(bits(&a), bits(&b), "case {cases}, point {k}");
                let (key, e) = line.key_under(*p, among, body);
                assert_eq!(key.to_bits(), out[k].to_bits(), "case {cases}, point {k}");
                whole = whole.max(e);
            }
            assert_eq!(whole.to_bits(), err.to_bits(), "case {cases}");
            // The centre alone, as a pixel's raster reads it.
            let centre = points[SAMPLES];
            let mut alone = given.clone();
            line.cut(centre, &[centre], &mut alone, body);
            let (a, b) = (
                line.find(centre, among, body),
                line.find(centre, &alone, body),
            );
            assert_eq!(bits(&a), bits(&b), "case {cases}, the centre alone");
            cut += (among.len() - alone.len()) as u64;
        }
        // The cases exercise the cut: it drops candidates in them.
        assert!(cut > cases, "{cut} candidates cut over {cases} cases");
    }
}
