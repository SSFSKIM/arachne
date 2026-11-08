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

/// The grid a writer that snaps positions to the input grid snaps a placement's linear part to,
/// as it snaps every ratio: 2^-24. The placement's translation is a position.
const RATIO_GRID: f64 = 1.0 / 16_777_216.0;

/// How far, along the unit normal `n`, the snap of a record's inputs to `grid` may have moved its
/// placed points from where content put them: `place(outer, place(inner, p))` for each `p` of
/// `points` (its shape's), `places` its placements outer first (the identity where there is none;
/// one placement exactly the identity is taken as none, as the snap makes the identity of nothing
/// else that matters). In a frame whose points were not snapped (`grid` `0`) no snap moved them,
/// and only the core's own placing rounds them.
///
/// The snap rounds each coordinate of a point and of a placement's translation to a multiple of
/// `grid`, so within half a step, and each entry of a placement's linear part to a multiple of
/// [`RATIO_GRID`], within half of that. A placement `x -> L x + t`, seen through the linear parts
/// of those outside it (their product `C`: the identity for the outer one), so moves a point along
/// `n` by at most `(grid / 2) |C^T n|_1` for its translation and `2^-25 |C^T n|_1 |x|_1` for its
/// linear part, `x` the point it places; the shape's own points move by at most
/// `(grid / 2) |(L_outer L_inner)^T n|_1`. For an unplaced record the snap's part is
/// `(grid / 2) |n|_1`: half a step for a plane facing an axis, `sqrt(3) / 2` steps for one facing
/// a diagonal.
///
/// The core's own placing rounds the points besides, snapped or not: it composes the two
/// placements (`xf::compose`, each entry a sum of at most four products and terms, so within
/// `gamma_4` of their magnitudes, `gamma_k = k u / (1 - k u)`) and places each point by the
/// product (`xf::place`, four terms again). Each placed coordinate so lies within
/// `2 gamma_4 (1 + gamma_4)` of the magnitudes `A = |outer| (|inner| |p| + |t_inner|) +
/// |t_outer|` of the exact placing, and the point within that times `|n| . A` along `n`: `8u` of
/// it to the first order, the rest (`64u^2` of it) far below any key's own rounding.
pub(crate) fn snap_departure(
    grid: f64,
    n: [f64; 3],
    places: &[crate::xf::Affine; 2],
    points: &[[f64; 3]],
) -> f64 {
    use crate::xf::{IDENTITY, place};
    let snapped = grid > 0.0;
    let l1 = |v: [f64; 3]| v[0].abs() + v[1].abs() + v[2].abs();
    let [outer, inner] = places;
    // The largest `|x|_1` of the points a placement places: the inner one's are the shape's, the
    // outer one's those the inner placed.
    let largest = |by: Option<&crate::xf::Affine>| {
        points
            .iter()
            .map(|p| l1(by.map_or(*p, |m| place(m, *p))))
            .fold(0.0, f64::max)
    };
    let mut seen = n; // C^T n
    let mut moved = 0.0;
    for (m, by) in [(outer, Some(inner)), (inner, None)] {
        if !snapped || *m == IDENTITY {
            continue;
        }
        moved += l1(seen) * (grid / 2.0 + RATIO_GRID / 2.0 * largest(by));
        seen = std::array::from_fn(|j| m[0][j] * seen[0] + m[1][j] * seen[1] + m[2][j] * seen[2]);
    }
    if snapped {
        moved += grid / 2.0 * l1(seen);
    }
    let abs = |m: &crate::xf::Affine| m.map(|row| row.map(f64::abs));
    let (outer_abs, inner_abs) = (abs(outer), abs(inner));
    let size = points
        .iter()
        .map(|p| {
            let placed = place(&outer_abs, place(&inner_abs, p.map(f64::abs)));
            dot(n.map(f64::abs), placed)
        })
        .fold(0.0, f64::max);
    moved + 8.0 * U * size
}

/// How far the points lie off `plane` along its normal, from the plane its key is held to: its
/// unrounded normal through the exact offset (`Projection::key_plane`), which lies within its
/// `offset_err` of the offset found; with the rounding of reading each point, `4u` of its terms.
/// Zero, but for rounding, for the corners of the triangle it was found through, or a path on the
/// axis plane all its points share; for a face of more points, the snap's noise tilts the plane
/// fitted through its first point (Newell's normal), and its other points stand off it.
pub(crate) fn departure(plane: &WorldPlane, points: &[[f64; 3]]) -> f64 {
    let u = plane.unrounded;
    points
        .iter()
        .map(|p| {
            let terms = dot(u.normal.map(f64::abs), p.map(f64::abs)) + u.offset.abs();
            (dot(u.normal, *p) - u.offset).abs() + 4.0 * U * terms
        })
        .fold(0.0, f64::max)
        + u.offset_err
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
/// offset. A flat record looks up the identities whose planes lie in the buckets around its own
/// plane's, as far as its own plane may stand from a plane its points lie on ([`Registry::near`]).
const NORMAL_BUCKET: f64 = 1.0 / 64.0;
const OFFSET_BUCKET: f64 = 0.5;

/// The most buckets a lookup reaches past a record's own on each side. A record whose own plane
/// may stand farther than that from a plane its points lie on (a face under about 0.02 across at
/// the writer's grid of 2^-14) can miss an identity it lies on.
const MOST_REACH: i64 = 2;

/// The most identities a record carries ([`Fragment::plane`]): a record with no plane of its own
/// lying on more keeps the first ones, in the order of the identities, and is counted
/// ([`Registry::capped`]).
///
/// [`Fragment::plane`]: crate::composite::Fragment::plane
pub const MOST_IDENTITIES: usize = 4;

/// A record offered to the registry. A fragment of a flat record carries it
/// ([`Fragment::record`]), which the near-parallel test reads ([`near_parallel`]).
///
/// [`Fragment::record`]: crate::composite::Fragment::record
#[derive(Clone, Debug, PartialEq)]
pub struct Offer {
    /// Its own plane: a flat record's, fitted to its points; for a record with no plane of its own,
    /// the axis plane its points share, if they do.
    own: Option<WorldPlane>,
    /// Whether it takes part in the partition ([`Registry::offer_flat`]).
    flat: bool,
    /// For a record with no plane of its own on an axis plane: the coordinate its points share.
    axis: Option<usize>,
    /// Its points placed in the world, as drawn.
    points: Vec<[f64; 3]>,
    /// Its placements, outer first, and its shape's own points: what the snap's departure of its
    /// points is found from ([`snap_departure`]).
    places: [crate::xf::Affine; 2],
    shape: Vec<[f64; 3]>,
    /// How far its points lie off its own plane ([`departure`]); zero for a record with no plane of
    /// its own.
    dep: f64,
    /// Its tolerance's bound along every normal ([`snap_departure_bound`] and `dep`).
    widest: f64,
    /// How far, as an angle, its own plane may stand from a plane holding its points within its
    /// tolerance: `widest` times the perimeter over the area of the widest triangle on its points
    /// (the plane through those three corners within `widest` of each tilts by at most `widest`
    /// over each height, summed). Infinite for a record with no plane of its own.
    spread: f64,
    /// For a flat record, its tolerance along its own plane's unit normal
    /// ([`Registry::tolerance`]): how far, at each of its points, its own plane may stand from the
    /// plane content put them on ([`near_parallel`]). Zero for a record with no plane of its own.
    own_tolerance: f64,
    /// Its handle in the registry that holds it.
    handle: u32,
    /// The near-parallel verdicts found for it against records of higher handles ([`tested`]).
    tested: Tested,
}

#[cfg(feature = "counters")]
impl Offer {
    /// How many points it has.
    pub(crate) fn points(&self) -> usize {
        self.points.len()
    }
}

/// How many partners' near-parallel verdicts a record keeps ([`tested`]).
const TESTED: usize = 8;

/// A record's last near-parallel verdicts against partners of higher handles, each one word:
/// the partner's handle in the low 32 bits, then a bit marking the word filled, the verdict, and
/// whether the dihedral passed [`WIDE`]. A verdict is a function of the two records alone, so
/// every thread that fills a word fills it with the same bits, and a thread that reads another
/// partner's word, or none, finds the verdict again. A clone keeps none.
#[derive(Default)]
struct Tested([std::sync::atomic::AtomicU64; TESTED]);

impl Clone for Tested {
    fn clone(&self) -> Tested {
        Tested::default()
    }
}

impl PartialEq for Tested {
    fn eq(&self, _: &Tested) -> bool {
        true
    }
}

impl std::fmt::Debug for Tested {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Tested")
    }
}

/// A dihedral past which a near-parallel pair is counted apart (`counters` builds): a record
/// small across the axis has a large `θ_R`, as it should, but a crossing of clearly distinct
/// surfaces turned to paint order over a visible band is one to see.
pub(crate) const WIDE: f64 = 1e-2;

/// [`near_parallel`]'s verdict on two records of one registry, as [`tested`] keeps it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Verdict {
    /// Whether the two are near-parallel.
    pub(crate) near: bool,
    /// Whether their dihedral passes [`WIDE`].
    pub(crate) wide: bool,
    /// Whether this call ran the test, rather than reading a verdict kept.
    pub(crate) fresh: bool,
}

/// [`near_parallel`]'s verdict on flat records `a` and `b` of one registry, run once per pair for
/// as long as their registry lives: the record of the lower handle keeps it, among its last
/// [`TESTED`] partners' ([`Tested`]). The same bits whichever thread asks, and in whichever order
/// the two come, as [`near_parallel`] gives them.
pub(crate) fn tested(a: &Offer, b: &Offer) -> Verdict {
    use std::sync::atomic::Ordering::Relaxed;
    const FILLED: u64 = 1 << 32;
    const NEAR: u64 = 1 << 33;
    const PAST: u64 = 1 << 34;
    let (low, high) = if a.handle <= b.handle { (a, b) } else { (b, a) };
    let slot = &low.tested.0[high.handle as usize % TESTED];
    let word = slot.load(Relaxed);
    if word & FILLED != 0 && word as u32 == high.handle {
        return Verdict {
            near: word & NEAR != 0,
            wide: word & PAST != 0,
            fresh: false,
        };
    }
    let tilt = near_parallel(low, high);
    let wide = tilt.dihedral > WIDE;
    let bits = |on: bool, bit: u64| if on { bit } else { 0 };
    slot.store(
        u64::from(high.handle) | FILLED | bits(tilt.near, NEAR) | bits(wide, PAST),
        Relaxed,
    );
    Verdict {
        near: tilt.near,
        wide,
        fresh: true,
    }
}

/// An identity while the registry settles.
#[derive(Clone, Debug)]
struct Identity {
    /// Its members, by offer, in the order they joined.
    members: Vec<usize>,
    /// Its plane, which holds every member's points within the member's tolerance.
    plane: WorldPlane,
    /// Three of its members' points, `(offer, point)`, spanning the widest triangle found among
    /// them: what bounds how far another plane holding the members may stand from its plane
    /// elsewhere ([`Registry::surely_apart`]).
    corners: [(usize, usize); 3],
}

/// How far past 1 a fit's largest share ([`Registry::minimax`]), each point's tolerance taken along
/// one normal, may fall back within 1 with the tolerances taken along the normal of the plane
/// fitted, a turn of at most the corners' tilt away: a fit over the corners, a necessary test, gives
/// up only past it, and a fit of every member past 1 but within it is tried again along the fitted
/// normal ([`Registry::fit`]).
const REFIT_ROOM: f64 = 1.0 / 1024.0;

/// Which pass of [`Registry::settle`] a fit runs in, for its counts (`counters` builds).
#[derive(Clone, Copy)]
enum Pass {
    /// The partition of the flat records.
    Partition,
    /// The records with no plane of their own attaching to the partition's identities.
    Attach,
}

impl Pass {
    /// Counts one linear program run ([`chebyshev`]).
    fn program(self) {
        match self {
            Pass::Partition => {
                count!(PARTITION_PROGRAMS);
            }
            Pass::Attach => {
                count!(ATTACH_PROGRAMS);
            }
        }
    }

    /// Counts one fit that ran over every member and found no plane.
    fn failed(self) {
        match self {
            Pass::Partition => {
                count!(PARTITION_FAILED_FITS);
            }
            Pass::Attach => {
                count!(ATTACH_FAILED_FITS);
            }
        }
    }
}

/// What an identity's corners bound, found from them alone ([`Registry::corners`]): its plane, the
/// corners on the plane of the two coordinates other than its normal's largest, each corner's
/// tolerance along its normal and `gap` (that and the corner's bound along every normal: how far
/// another plane holding the members may stand from its plane there), the corners' doubled signed
/// area, and `tilt`, how far the normals of two such planes may part ([`Registry::surely_apart`]).
#[derive(Clone, Copy, Debug)]
struct Corners {
    plane: WorldPlane,
    axes: [usize; 2],
    at: [[f64; 2]; 3],
    tolerance: [f64; 3],
    gap: [f64; 3],
    area: f64,
    tilt: f64,
}

/// What the attach pass of [`Registry::settle`] reads of a settled flat identity, found once from
/// its flat members alone before any record attaches (an attached record changes none of it): its
/// corners' bounds, its members' points' hull, and the parts of the bound within which a record
/// with no plane of its own that lies on it stands off its plane ([`Reach::bound`]).
#[derive(Debug)]
struct Reach {
    corners: Corners,
    /// The largest over its members of their tolerance and their snap's departure
    /// ([`snap_departure`]), both along its plane's normal.
    members: f64,
    /// The largest over its members of their tolerance's bound along every normal.
    widest: f64,
    /// Twice how far its plane's unit normal may stand from the normal of a plane holding its
    /// members within their tolerances' bounds along every normal: twice its corners' tilt, and
    /// at most 4 (two unit normals stand at most 2 apart).
    turn: f64,
    /// Its members' points on the plane of its corners' two coordinates (`Corners::axes`): their
    /// convex hull, counterclockwise ([`convex_hull`]).
    hull: Vec<[f64; 2]>,
    /// How fast, per unit of those coordinates away from the hull, the difference of its plane and
    /// content's through its members may grow ([`Reach::bound`]).
    slope: f64,
}

impl Reach {
    /// `p` on the corners' two coordinates.
    fn flat(&self, p: [f64; 3]) -> [f64; 2] {
        let [i, j] = self.corners.axes;
        [p[i], p[j]]
    }

    /// How far `p` lies from the hull on those coordinates, 0 inside it.
    fn distance(&self, p: [f64; 3]) -> f64 {
        hull_distance(&self.hull, self.flat(p))
    }

    /// Whether a point `distance` from the hull, of a record whose bound along every normal is
    /// `widest`, lies within the snap's reach of it, the record's bound and the members' along
    /// every normal: as near as the snap can tell, in the hull.
    fn inside(&self, widest: f64, distance: f64) -> bool {
        distance <= widest + self.widest
    }

    /// How far off the identity's plane a record with no plane of its own may stand where content
    /// put it on one plane with the identity's members, at a point `distance` from their points'
    /// hull: `tolerance + members + (turn + slope) (widest + Reach::widest) + slope distance`,
    /// `tolerance` its tolerance along the plane's normal and `widest` its bound along every
    /// normal.
    ///
    /// The identity's plane `n . x = d` holds each member's points within its tolerance `t_m` along
    /// `n`; the content's plane `v . x = e` holds them within their snap's departure along `v`,
    /// `s_m(v)`, each moved from where content put it by at most its bound `w_m` along every
    /// normal. The two planes' difference is affine, its gradient `n - v`, of length `a`: at a
    /// member's point as content put it, at most `t_m(n) + s_m(v) + a w_m`. The snap's departure
    /// is a seminorm of the normal, at most `w` on unit normals, so `s(v)` is at most
    /// `s(n) + a w`: at those points the difference is at most
    /// `g = max_m (t_m(n) + s_m(n)) + 2 a max_m w_m`, and so within their hull. A point of the
    /// record, moved from content's point on `v` by at most `w`, stands off `n . x = d` by at most
    /// the difference there, `a w`, and `s(v)`: within the hull,
    /// `t(n) + max_m (t_m(n) + s_m(n)) + 2 a (w + max_m w_m)`. The corners bound `a` by their tilt
    /// ([`Registry::surely_apart`]), and twice it is `turn`.
    ///
    /// Outside the hull: on content's plane, read on the corners' two coordinates, the
    /// difference is affine, at most `g` on the hull of the members' points as content put them,
    /// so across that hull it changes by its gradient's length times the hull's width along the
    /// gradient: the gradient is at most `2 g` over the hull's least width. Those points stand
    /// within `max_m w_m` of the drawn ones, so that hull's least width is at least this hull's
    /// less twice that: the correction only narrows the hull, and only steepens the slope. On
    /// content's plane the coordinate left out, `k` (`n`'s largest component, at least
    /// `1 / sqrt(3)`), is `x_k = (e - v_i x_i - v_j x_j) / v_k`, so there the difference's gradient
    /// on the two coordinates is `(n_i v_k - n_k v_i, n_j v_k - n_k v_j) / v_k`: two components of
    /// `n x v`, of length at most `a`, over `v_k`, at least `1 / sqrt(3) - a`. So the gradient is
    /// also at most `sqrt(3) a / (1 - sqrt(3) a)`, unbounded where `sqrt(3) a` reaches 1
    /// ([`turned_slope`]); `slope` is the lesser. A point of the record `distance` from this hull
    /// lies, as content put it, within `distance + w + max_m w_m` of that hull, where the
    /// difference is at most `g` and `slope` times that more.
    fn bound(&self, tolerance: f64, widest: f64, distance: f64) -> f64 {
        tolerance
            + self.members
            + (self.turn + self.slope) * (widest + self.widest)
            + self.slope * distance
    }
}

/// The planes of a frame's mesh triangles and paths, and their identities ([`Fragment::plane`],
/// from 1), which what lies on one plane shares whichever producer drew it, so it ties by draw
/// order there (`order::one_plane`): the planes fitted to snapped points of things on one plane
/// differ by the snap's noise, about the grid over their size, and would cross.
///
/// Records are offered in any order (the meshes' triangles by `mesh::lower`, the paths by
/// `path::Lowering::coplanar`), then the registry settles once ([`Registry::settle`]). The
/// identities are a function of the set of records offered, not of their order:
///
/// - A **flat** record ([`Registry::offer_flat`]: a mesh triangle, or a path content made flat
///   whose points do not lie on a line) takes part in the partition. In a canonical order of the
///   records ([`canonical`]), each joins the first identity (in the order they were founded) on
///   which one plane holds every member's points and its own, each within its own tolerance
///   ([`Registry::tolerance`]: the snap's departure of its points along that plane's normal and
///   their departure from its own plane); else it founds one. So every member lies on its
///   identity's final plane within its own tolerance: a slightly curved surface, each face of it
///   on one plane with its neighbours, does not chain into one identity, and a flat one is one
///   identity because one plane holds all its faces, whatever plane each face's snapped points
///   fit. Records content put on one plane always may share one: that plane holds each within the
///   snap's departure of its points.
/// - A record with **no plane of its own** ([`Registry::offer_planeless`]: points on a line,
///   however the writer coded them, or a curve) neither founds an identity nor joins one: once the
///   partition settles, it attaches to every identity on which one plane holds the members' points
///   and its own within their tolerances (a line along a cube's edge, to both faces'), whose
///   plane its points stand off by no more than content's plane through the members would
///   leave them ([`Reach::bound`]: within the members' hull, and growing with the distance from it
///   as fast as that plane may part from the identity's), and whose members' hull it enters
///   ([`Registry::enters`]), the first [`MOST_IDENTITIES`] of them: a record that passes beside an
///   identity's members, or far off its plane past them, does not. One
///   that attaches to none and lies on an axis plane shares an identity with the others on exactly
///   that axis plane, its axis and offset.
///
/// An identity's plane is its founder's own until a record joins that it does not hold; then the
/// plane holding every member with the most room, as a share of each one's tolerance ([`minimax`]).
/// A flat record stays keyed on its own plane; an identity's plane keys only what attaches to it.
///
/// [`Fragment::plane`]: crate::composite::Fragment::plane
#[derive(Debug, Default)]
pub struct Registry {
    grid: f64,
    offers: Vec<Offer>,
    /// Once settled: each offer's identities, ascending, zero-padded.
    sets: Vec<[u32; 4]>,
    /// Once settled: whether each offer with no plane of its own attached to flat records'
    /// identities.
    attached: Vec<bool>,
    /// Once settled: each identity's plane, identity `i` at `i - 1`.
    planes: Vec<WorldPlane>,
    /// Once settled: what each identity the partition made gives the attach pass, identity `i` at
    /// `i - 1` ([`Reach`]); none where no record with no plane of its own was offered.
    reaches: Vec<Reach>,
    /// Records that lay on more than [`MOST_IDENTITIES`] identities.
    capped: usize,
    settled: bool,
}

impl Registry {
    /// An empty registry for a frame of `view`, whose points are snapped to its grid.
    pub fn new(view: &crate::xf::View) -> Registry {
        Registry::of_grid(input_grid(view))
    }

    /// An empty registry for points snapped to `grid` (`0.0`: not snapped).
    pub fn of_grid(grid: f64) -> Registry {
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

    /// Offers a flat record, `own` its plane fitted to `points` (its points placed in the world),
    /// placed by `places` (outer first) from its shape's `shape`; its handle.
    pub(crate) fn offer_flat(
        &mut self,
        own: WorldPlane,
        points: &[[f64; 3]],
        places: &[crate::xf::Affine; 2],
        shape: &[[f64; 3]],
    ) -> u32 {
        let dep = departure(&own, points);
        let widest = snap_departure_bound(self.grid, places, shape) + dep;
        let [a, b, c] = widest_triangle(points).map(|k| points[k]);
        let spread = widest * (length(sub(b, a)) + length(sub(c, b)) + length(sub(a, c)))
            / (length(cross(sub(b, a), sub(c, a))) / 2.0);
        let own_tolerance = snap_departure(self.grid, own.unrounded.normal, places, shape) + dep;
        self.push(Offer {
            own: Some(own),
            flat: true,
            axis: None,
            points: points.to_vec(),
            places: *places,
            shape: shape.to_vec(),
            dep,
            widest,
            spread: if spread.is_nan() {
                f64::INFINITY
            } else {
                spread
            },
            own_tolerance,
            handle: 0,
            tested: Tested::default(),
        })
    }

    /// Offers a record with no plane of its own ([`Registry`]): `axis` the coordinate its points
    /// share when they lie on that axis plane; its handle.
    pub(crate) fn offer_planeless(
        &mut self,
        axis: Option<u8>,
        points: &[[f64; 3]],
        places: &[crate::xf::Affine; 2],
        shape: &[[f64; 3]],
    ) -> u32 {
        self.push(Offer {
            own: axis.map(|k| axis_plane(k, points[0])),
            flat: false,
            axis: axis.map(usize::from),
            points: points.to_vec(),
            places: *places,
            shape: shape.to_vec(),
            dep: 0.0,
            widest: snap_departure_bound(self.grid, places, shape),
            spread: f64::INFINITY,
            own_tolerance: 0.0,
            handle: 0,
            tested: Tested::default(),
        })
    }

    /// Offers a path, `points` its points placed in the world, `planarity` as its writer decided
    /// it (or as decided from its points), placed by `places` (outer first) from its shape's
    /// `shape`: flat ([`Registry::offer_flat`]) on its plane fitted ([`WorldPlane::fit`]) where
    /// content made it flat (Newell's, the widest triangle's or an axis plane), its points do not
    /// lie on a line ([`on_a_line`]) and the fit finds a plane; else with no plane of its own
    /// ([`Registry::offer_planeless`]). Its handle, whether it is flat, and its plane fitted.
    pub(crate) fn offer_path(
        &mut self,
        points: &[[f64; 3]],
        planarity: Planarity,
        places: &[crate::xf::Affine; 2],
        shape: &[[f64; 3]],
    ) -> (u32, bool, Option<WorldPlane>) {
        let own = WorldPlane::fit(points, planarity);
        let flat = matches!(
            planarity,
            Planarity::Newell | Planarity::Widest | Planarity::Axis(_)
        ) && !on_a_line(points, self.grid);
        match (flat, own) {
            (true, Some(plane)) => (self.offer_flat(plane, points, places, shape), true, own),
            _ => {
                let axis = match planarity {
                    Planarity::Axis(k) => Some(k),
                    _ => None,
                };
                let handle = self.offer_planeless(axis, points, places, shape);
                (handle, false, own)
            }
        }
    }

    /// Offers a mesh triangle, `corners` its corners placed in the world, placed by `places` (outer
    /// first) from its shape's `shape`: flat ([`Registry::offer_flat`]) on `plane`, the plane
    /// through its corners ([`WorldPlane::of_triangle`]), where they do not lie on a line
    /// ([`on_a_line`]); else, as any record whose points lie on a line, with no plane of its own
    /// ([`Registry::offer_planeless`]), on the axis plane its corners share if they do. Its handle
    /// and whether it is flat.
    pub(crate) fn offer_triangle(
        &mut self,
        plane: Option<WorldPlane>,
        corners: &[[f64; 3]; 3],
        places: &[crate::xf::Affine; 2],
        shape: &[[f64; 3]],
    ) -> (u32, bool) {
        match plane.filter(|_| !on_a_line(corners, self.grid)) {
            Some(plane) => (self.offer_flat(plane, corners, places, shape), true),
            None => {
                let axis = shared_axis(corners, self.grid);
                (self.offer_planeless(axis, corners, places, shape), false)
            }
        }
    }

    fn push(&mut self, mut offer: Offer) -> u32 {
        assert!(!self.settled, "a record offered after the registry settled");
        let handle = self.offers.len() as u32;
        offer.handle = handle;
        self.offers.push(offer);
        handle
    }

    /// How far `offer`'s points may lie off a plane of unit normal `n` and still be on it: the
    /// snap's departure of its points along `n` ([`snap_departure`]) and their departure from its
    /// own plane.
    fn tolerance(&self, offer: &Offer, n: [f64; 3]) -> f64 {
        snap_departure(self.grid, n, &offer.places, &offer.shape) + offer.dep
    }

    /// The tolerance of the record offered as `handle` along unit normal `n`: how far its points
    /// may lie off a plane of that normal and still be on it ([`Registry`]), the snap's departure
    /// of its points along `n` and, for a flat record, their departure from its own plane.
    pub fn tolerance_along(&self, handle: u32, n: [f64; 3]) -> f64 {
        self.tolerance(&self.offers[handle as usize], n)
    }

    /// Whether the record offered as `handle` is flat: it takes part in the partition.
    pub(crate) fn is_flat(&self, handle: u32) -> bool {
        self.offers[handle as usize].flat
    }

    /// How far the points of the record offered as `handle` may stand off settled identity
    /// `identity`'s plane from where content put them: its tolerance along that plane's normal and
    /// their departure from it. What bounds how far a key read on those points stands from that
    /// plane's, within their hull.
    pub(crate) fn off_identity(&self, handle: u32, identity: u32) -> f64 {
        let (offer, plane) = (&self.offers[handle as usize], self.plane(identity));
        self.tolerance(offer, plane.unrounded.normal) + departure(&plane, &offer.points)
    }

    /// [`Registry::off_identity`] where the record offered as `handle` meets `identity`'s flat
    /// members: for a record with no plane of its own on an identity the partition made, at most
    /// the bound within the members' hull ([`Reach::bound`] there), within which content's plane
    /// through it and the members holds it there, where they lie.
    pub(crate) fn off_members(&self, handle: u32, identity: u32) -> f64 {
        let off = self.off_identity(handle, identity);
        let offer = &self.offers[handle as usize];
        match self.reaches.get(identity as usize - 1) {
            Some(reach) if !offer.flat => {
                let tolerance = self.tolerance(offer, self.plane(identity).unrounded.normal);
                off.min(reach.bound(tolerance, offer.widest, 0.0))
            }
            _ => off,
        }
    }

    /// Whether `plane` holds every point of offer `o` within its tolerance along its normal.
    fn holds(&self, plane: &WorldPlane, o: usize) -> bool {
        let offer = &self.offers[o];
        let tolerance = self.tolerance(offer, plane.unrounded.normal);
        offer.points.iter().all(|p| off(plane, *p) <= tolerance)
    }

    /// Settles the identities of every record offered ([`Registry`]). Settling again does nothing.
    pub fn settle(&mut self) {
        if self.settled {
            return;
        }
        self.settled = true;
        #[cfg(feature = "counters")]
        let settling = std::time::Instant::now();
        let offers = self.offers.len();
        let mut flat: Vec<usize> = (0..offers).filter(|&o| self.offers[o].flat).collect();
        flat.sort_by(|&a, &b| canonical(&self.offers[a], &self.offers[b]));
        let mut ids: Vec<Identity> = Vec::new();
        let mut index: std::collections::HashMap<[i64; 4], Vec<usize>> = Default::default();
        self.sets = vec![[0; 4]; offers];
        self.attached = vec![false; offers];
        for o in flat {
            let joined = self
                .near(&index, o)
                .into_iter()
                .find_map(|i| self.joins(&ids[i], o).map(|plane| (i, plane)));
            let i = match joined {
                Some((i, plane)) => {
                    if let Some(plane) = plane {
                        let (was, now) =
                            (Registry::bucket(&ids[i].plane), Registry::bucket(&plane));
                        if was != now {
                            index.entry(was).or_default().retain(|&k| k != i);
                            index.entry(now).or_default().push(i);
                        }
                        ids[i].plane = plane;
                    }
                    ids[i].members.push(o);
                    ids[i].corners = self.widened(ids[i].corners, o);
                    i
                }
                None => {
                    let plane = self.offers[o].own.expect("a flat record has its own plane");
                    index
                        .entry(Registry::bucket(&plane))
                        .or_default()
                        .push(ids.len());
                    ids.push(Identity {
                        members: vec![o],
                        plane,
                        corners: widest_triangle(&self.offers[o].points).map(|k| (o, k)),
                    });
                    ids.len() - 1
                }
            };
            self.sets[o][0] = i as u32 + 1;
        }
        let flats = ids.len();
        #[cfg(feature = "counters")]
        crate::counters::since(&crate::counters::PARTITION_NS, settling);
        #[cfg(feature = "counters")]
        let attaching = std::time::Instant::now();
        let mut alone = Vec::new();
        let planeless: Vec<usize> = (0..offers).filter(|&o| !self.offers[o].flat).collect();
        let reach: Vec<Reach> = if planeless.is_empty() {
            Vec::new()
        } else {
            ids.iter().map(|id| self.reach(id)).collect()
        };
        count!(FLAT_IDENTITIES, flats);
        count!(ATTACH_RECORDS, planeless.len());
        count!(ATTACH_PAIRS, planeless.len() * flats);
        for o in planeless {
            // Its first point first, against its bound along every normal at no less than its
            // distance from the hull (its distance from the hull's first corner): one product and
            // one root a pair.
            let (first, widest) = (
                self.offers[o].points.first().copied(),
                self.offers[o].widest,
            );
            let mut through = false;
            let on: Vec<usize> = (0..flats)
                .filter(|&i| {
                    let r = &reach[i];
                    first.is_none_or(|p| {
                        let [a, b] = r.flat(p);
                        let far = r
                            .hull
                            .first()
                            .map_or(f64::INFINITY, |c| libm::hypot(a - c[0], b - c[1]));
                        within(&ids[i].plane, p) <= r.bound(widest, widest, far)
                    }) && self.contains(&ids[i], r, o).is_some_and(|outside| {
                        through |= outside;
                        true
                    })
                })
                .collect();
            if through {
                count!(ATTACH_THROUGH_ALLOWANCE);
            }
            if on.len() > MOST_IDENTITIES {
                self.capped += 1;
                count!(CAPPED_SETS);
            }
            for (k, i) in on.iter().take(MOST_IDENTITIES).enumerate() {
                self.sets[o][k] = *i as u32 + 1;
            }
            self.attached[o] = !on.is_empty();
            if on.is_empty() && self.offers[o].axis.is_some() {
                alone.push(o);
            }
        }
        self.reaches = reach;
        #[cfg(feature = "counters")]
        crate::counters::since(&crate::counters::ATTACH_NS, attaching);
        // What attaches to no flat identity and lies on an axis plane shares an identity only with
        // the others on exactly that plane: the same axis and the same offset, the coordinate of
        // its first point, which is its plane. No tolerance and no fit: an axis plane is authored,
        // and a plane its records only lie near, or that a fit finds through them, does not decide
        // an identity. Its plane is that axis plane, which no flat record's plane replaces (the
        // partition is made); a flat identity holding such a record took it above. Numbered in
        // the canonical order of each plane's first record, as the partition is.
        alone.sort_by(|&a, &b| canonical(&self.offers[a], &self.offers[b]));
        let mut runs: std::collections::HashMap<(usize, u64), usize> = Default::default();
        for o in alone {
            let offer = &self.offers[o];
            let plane = offer.own.expect("an axis record has its axis plane");
            let at = (
                offer.axis.expect("an axis record"),
                (plane.unrounded.offset + 0.0).to_bits(),
            );
            let i = *runs.entry(at).or_insert_with(|| {
                ids.push(Identity {
                    members: Vec::new(),
                    plane,
                    corners: [(o, 0); 3],
                });
                ids.len() - 1
            });
            ids[i].members.push(o);
            self.sets[o][0] = i as u32 + 1;
        }
        count!(AXIS_RUN_IDENTITIES, runs.len());
        self.planes = ids.iter().map(|id| id.plane).collect();
        #[cfg(feature = "counters")]
        crate::counters::since(&crate::counters::SETTLE_NS, settling);
    }

    /// The identities whose planes lie in the buckets within reach of flat offer `o`'s own plane
    /// (either way round: the canonical normal's largest component can flip sign between two fits
    /// of one plane), ascending. Its own plane stands within its `spread` of a plane its points
    /// lie on, and an identity's plane within its founder's of that one; the canonical order
    /// founds identities from records with the least spread first, so twice its spread reaches
    /// across both, and the offset as far as that tilts the plane at its farthest point, with its
    /// tolerance, both ways.
    fn near(
        &self,
        index: &std::collections::HashMap<[i64; 4], Vec<usize>>,
        o: usize,
    ) -> Vec<usize> {
        let offer = &self.offers[o];
        let own = offer.own.expect("a flat record has its own plane");
        let reach = |span: f64, side: f64| {
            let steps = (span / side).ceil();
            if steps < MOST_REACH as f64 {
                (steps as i64).max(1)
            } else {
                MOST_REACH
            }
        };
        let far = offer.points.iter().fold(0.0f64, |m, p| m.max(length(*p)));
        let normal = reach(2.0 * offer.spread, NORMAL_BUCKET);
        let offset = reach(2.0 * offer.spread * far + 2.0 * offer.widest, OFFSET_BUCKET);
        let mut found = Vec::new();
        for plane in [own, own.turned()] {
            let key = Registry::bucket(&plane);
            for a in -normal..=normal {
                for b in -normal..=normal {
                    for c in -normal..=normal {
                        for d in -offset..=offset {
                            let at = [key[0] + a, key[1] + b, key[2] + c, key[3] + d];
                            found.extend(index.get(&at).into_iter().flatten());
                        }
                    }
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }

    /// Whether flat offer `o` joins identity `id`, and the identity's plane then: its own
    /// (`Some(None)`) where that holds `o`'s points within its tolerance, else the plane holding
    /// every member's with the most room ([`Registry::fit`]), if one holds them all.
    fn joins(&self, id: &Identity, o: usize) -> Option<Option<WorldPlane>> {
        if self.holds(&id.plane, o) {
            return Some(None);
        }
        if self.surely_apart(&self.corners(id), o) {
            return None;
        }
        self.fit(id, o, Pass::Partition).map(Some)
    }

    /// What settled flat identity `id` gives the attach pass ([`Reach`]), from its members alone:
    /// its corners' bounds; its members' tolerances and snap's departures along its plane's
    /// normal, and their bounds along every normal, at their largest; their points' hull; and the
    /// slope the bound grows by outside it ([`Reach::bound`]), widened by `1e-12` for the hull's
    /// own arithmetic.
    fn reach(&self, id: &Identity) -> Reach {
        let n = id.plane.unrounded.normal;
        let (mut members, mut widest) = (0.0f64, 0.0f64);
        for &m in &id.members {
            let offer = &self.offers[m];
            let snap = snap_departure(self.grid, n, &offer.places, &offer.shape);
            members = members.max(2.0 * snap + offer.dep);
            widest = widest.max(offer.widest);
        }
        let corners = self.corners(id);
        let turn = 2.0 * corners.tilt.min(2.0);
        let [i, j] = corners.axes;
        let hull = convex_hull(
            id.members
                .iter()
                .flat_map(|&m| self.offers[m].points.iter().map(|p| [p[i], p[j]]))
                .collect(),
        );
        let narrowed = least_width(&hull) - 2.0 * widest;
        let across = if narrowed > 0.0 {
            2.0 * (members + turn * widest) / narrowed
        } else {
            f64::INFINITY
        };
        let slope = across.min(turned_slope(turn)) * (1.0 + 1e-12);
        Reach {
            corners,
            members,
            widest,
            turn,
            hull,
            slope,
        }
    }

    /// Whether offer `o`, with no plane of its own, lies on settled identity `id`, `reach` what
    /// `id` gives the attach pass, and if so whether a point of it lies outside the members' hull
    /// (beyond the snap's reach of it, [`Reach::inside`]). It lies on `id` where:
    ///
    /// - every point stands off `id`'s plane within the bound content's plane through the members
    ///   leaves it there ([`Reach::bound`]);
    /// - it enters the members' hull ([`Registry::enters`]): a record that never does is read on
    ///   no part of the members' plane they bound, and stands off it, along a sight line, by at
    ///   least its distance from them, which depth tells apart (and an identity whose plane its
    ///   members leave unbounded, their hull thinner than the snap, would hold any record);
    /// - one plane holds its points and every member's, each within its own tolerance: `id`'s
    ///   plane, or the plane fitted to them all ([`Registry::fit`]).
    fn contains(&self, id: &Identity, reach: &Reach, o: usize) -> Option<bool> {
        count!(ATTACH_NEAR);
        let offer = &self.offers[o];
        let tolerance = self.tolerance(offer, id.plane.unrounded.normal);
        let mut outside = false;
        for p in &offer.points {
            let distance = reach.distance(*p);
            if within(&id.plane, *p) > reach.bound(tolerance, offer.widest, distance) {
                return None;
            }
            outside |= !reach.inside(offer.widest, distance);
        }
        let held = || {
            offer.points.iter().all(|p| off(&id.plane, *p) <= tolerance)
                || !self.surely_apart(&reach.corners, o) && self.fit(id, o, Pass::Attach).is_some()
        };
        if !self.enters(id, reach, o, tolerance) {
            #[cfg(feature = "counters")]
            {
                let flat = tolerance + reach.members + reach.turn * (offer.widest + reach.widest);
                if offer.points.iter().all(|p| within(&id.plane, *p) <= flat) && held() {
                    crate::counters::add(&crate::counters::ATTACH_UNENTERED, 1);
                }
            }
            return None;
        }
        held().then_some(outside)
    }

    /// Whether offer `o` enters `id`'s members' hull (`reach` what `id` gives the attach pass,
    /// `tolerance` its tolerance along `id`'s normal): a point of it lies within the snap's reach
    /// of the hull ([`Reach::inside`]), or a segment between two of its points, taken in turn
    /// and closed, does at a point standing off `id`'s plane within the bound there
    /// ([`Reach::bound`]). A segment crossing the hull is tested where it is in the hull: its
    /// distance from the plane is affine along it, so its least over that part is at an end of
    /// the part, or nought where it crosses the plane. A segment missing the hull is tested at its
    /// point nearest the hull, which lies over one of the hull's corners (its ends lie beyond the
    /// snap's reach, as every point does here), at its distance from the hull.
    fn enters(&self, id: &Identity, reach: &Reach, o: usize, tolerance: f64) -> bool {
        let offer = &self.offers[o];
        let points = &offer.points;
        if points
            .iter()
            .any(|p| reach.inside(offer.widest, reach.distance(*p)))
        {
            return true;
        }
        let (n, d) = (id.plane.unrounded.normal, id.plane.unrounded.offset);
        let bound = reach.bound(tolerance, offer.widest, 0.0);
        (0..points.len()).any(|k| {
            let (a, b) = (points[k], points[(k + 1) % points.len()]);
            let at = |t: f64| -> [f64; 3] { std::array::from_fn(|c| a[c] + (b[c] - a[c]) * t) };
            let (fa, fb) = (reach.flat(a), reach.flat(b));
            let Some((t0, t1)) = clip(&reach.hull, fa, fb) else {
                let u = [fb[0] - fa[0], fb[1] - fa[1]];
                let along = u[0] * u[0] + u[1] * u[1];
                let nearest = reach
                    .hull
                    .iter()
                    .map(|c| {
                        let w = [c[0] - fa[0], c[1] - fa[1]];
                        let t = if along > 0.0 {
                            ((u[0] * w[0] + u[1] * w[1]) / along).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        (reach.distance(at(t)), at(t))
                    })
                    .min_by(|x, y| x.0.total_cmp(&y.0));
                return nearest.is_some_and(|(distance, q)| {
                    reach.inside(offer.widest, distance)
                        && within(&id.plane, q) <= reach.bound(tolerance, offer.widest, distance)
                });
            };
            let (f0, f1) = (dot(n, at(t0)) - d, dot(n, at(t1)) - d);
            let t = if f0.signum() != f1.signum() && f0 != f1 {
                t0 + (t1 - t0) * f0 / (f0 - f1)
            } else if f0.abs() <= f1.abs() {
                t0
            } else {
                t1
            };
            within(&id.plane, at(t)) <= bound
        })
    }

    /// The plane holding every member of `id` and offer `o` with the most room ([`minimax`]),
    /// where it holds each within its own tolerance along its own normal; `None` where none does.
    /// The fit weighs each record by its tolerance along `id`'s plane's normal: it is tried and,
    /// where its tolerances taken again along its own normal do not hold or its largest share
    /// passes 1 by no more than [`REFIT_ROOM`], fitted once more from there. A fit over `id`'s
    /// corners and `o` alone, a necessary test, is tried first.
    fn fit(&self, id: &Identity, o: usize, pass: Pass) -> Option<WorldPlane> {
        let n = id.plane.unrounded.normal;
        let point = |(m, k): (usize, usize)| (m, self.offers[m].points[k]);
        let ours = self.offers[o].points.iter().map(|p| (o, *p));
        pass.program();
        if let Some((_, t)) = self.minimax(id.corners.map(point).into_iter().chain(ours), n)
            && t > 1.0 + REFIT_ROOM
        {
            return None;
        }
        let records = || id.members.iter().copied().chain([o]);
        let mut along = n;
        for again in [false, true] {
            let points = records().flat_map(|m| self.offers[m].points.iter().map(move |p| (m, *p)));
            pass.program();
            let Some((plane, t)) = self.minimax(points, along) else {
                break;
            };
            if t <= 1.0 && records().all(|m| self.holds(&plane, m)) {
                return Some(plane);
            }
            if again || t > 1.0 + REFIT_ROOM {
                break;
            }
            along = plane.unrounded.normal;
        }
        pass.failed();
        None
    }

    /// The plane minimising the largest of `points`' distances (each `(offer, point)`, an offer's
    /// points together) as shares of their offers' tolerances along `n`, and that largest share
    /// (`1` or less: every point within its tolerance). Fitted as `z = a u + b v + c` about the
    /// first point, `z` along `n`'s largest component and the distance along the plane's normal
    /// taken as `z`'s residual times that component of `n` ([`chebyshev`]); `None` where the fit
    /// fails (points on a line, a singular step).
    fn minimax(
        &self,
        points: impl Iterator<Item = (usize, [f64; 3])>,
        n: [f64; 3],
    ) -> Option<(WorldPlane, f64)> {
        let axis = largest_component(n);
        let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
        let along = 1.0 / n[axis].abs();
        let (mut rows, mut weight, mut origin) = (Vec::new(), None, None);
        for (o, p) in points {
            let w = match weight {
                Some((m, w)) if m == o => w,
                _ => {
                    let w = along * self.tolerance(&self.offers[o], n);
                    weight = Some((o, w));
                    w
                }
            };
            let at: [f64; 3] = *origin.get_or_insert(p);
            rows.push([p[i] - at[i], p[j] - at[j], p[axis] - at[axis], w]);
        }
        let at = origin?;
        let most = rows.iter().fold(0.0f64, |m, r| m.max(r[3]));
        if !(most > 0.0 && most.is_finite() && rows.iter().all(|r| r[3] > 0.0)) {
            return None;
        }
        for r in &mut rows {
            r[3] /= most;
        }
        let [a, b, c, t] = chebyshev(&rows)?;
        let mut m = [0.0; 3];
        (m[i], m[j], m[axis]) = (-a, -b, 1.0);
        let offset = c + at[axis] - a * at[i] - b * at[j];
        let unit = Direction::of(m)?.unit();
        let sign = if dot(unit, m) > 0.0 { 1.0 } else { -1.0 };
        let plane = WorldPlane::rounded(unit, sign * offset / dot(m, m).sqrt(), 0.0);
        (plane.unrounded.offset.is_finite() && t.is_finite()).then_some((plane, t / most))
    }

    /// Whether no plane can hold every member of `id` and offer `o` within their tolerances, found
    /// from `id`'s corners alone, without fitting one. Another plane holding the members stands
    /// within `gap` (a corner's tolerance along `id`'s plane's normal and its bound along any
    /// other) of `id`'s plane at each corner, so, the gap between two planes being affine, within
    /// the corners' gaps by the absolute barycentric weights of a point elsewhere; and the two
    /// planes' normals part by at most `tilt` (the gaps over the corners' heights, as the planes'
    /// heights along the axis of `id`'s normal's largest component, which takes a distance up to
    /// `sqrt(3)` times), which moves a point's distance by that share of its height off the plane.
    /// A point of `o` farther from `id`'s plane than its own bound along any normal, that reach,
    /// and the tilt's room, lies off every such plane. Not decided (`false`) where the corners'
    /// tilt passes a sixteenth. `corners` are `id`'s ([`Registry::corners`]).
    fn surely_apart(&self, corners: &Corners, o: usize) -> bool {
        let Corners {
            plane,
            axes: [i, j],
            at: [a, b, c],
            tolerance,
            gap,
            area,
            tilt,
        } = *corners;
        if tilt.is_nan() || tilt > 1.0 / 16.0 {
            return false;
        }
        let offer = &self.offers[o];
        offer.points.iter().any(|x| {
            let p = [x[i], x[j]];
            let weights = [twice(p, b, c), twice(a, p, c), twice(a, b, p)].map(|w| w / area);
            let reach: f64 = (0..3)
                .map(|v| weights[v].abs() * (gap[v] + 2.0 * tilt * tolerance[v]))
                .sum();
            within(&plane, *x) * (1.0 - 2.0 * tilt) > offer.widest + reach
        })
    }

    /// What identity `id`'s corners bound ([`Corners`]), as [`Registry::surely_apart`] reads it.
    fn corners(&self, id: &Identity) -> Corners {
        let plane = id.plane;
        let n = plane.unrounded.normal;
        let axis = largest_component(n);
        let axes = [(axis + 1) % 3, (axis + 2) % 3];
        let at = id.corners.map(|(m, k)| {
            let p = self.offers[m].points[k];
            [p[axes[0]], p[axes[1]]]
        });
        let tolerance = id.corners.map(|(m, _)| self.tolerance(&self.offers[m], n));
        let gap: [f64; 3] =
            std::array::from_fn(|v| tolerance[v] + self.offers[id.corners[v].0].widest);
        let [a, b, c] = at;
        let area = twice(a, b, c);
        let side = |p: [f64; 2], q: [f64; 2]| libm::hypot(p[0] - q[0], p[1] - q[1]);
        let sides = [side(b, c), side(c, a), side(a, b)];
        let tilt = 3f64.sqrt() * (0..3).map(|v| gap[v] * sides[v] / area.abs()).sum::<f64>();
        Corners {
            plane,
            axes,
            at,
            tolerance,
            gap,
            area,
            tilt,
        }
    }

    /// `corners` widened by offer `o`'s points: each replaces a corner where the triangle grows.
    fn widened(&self, mut corners: [(usize, usize); 3], o: usize) -> [(usize, usize); 3] {
        let area = |c: &[(usize, usize); 3]| {
            let [a, b, c] = c.map(|(m, k)| self.offers[m].points[k]);
            length(cross(sub(b, a), sub(c, a)))
        };
        for k in 0..self.offers[o].points.len() {
            for v in 0..3 {
                let mut tried = corners;
                tried[v] = (o, k);
                if area(&tried) > area(&corners) {
                    corners = tried;
                }
            }
        }
        corners
    }

    /// The identities of the record offered as `handle`, ascending, zero-padded (once settled).
    pub(crate) fn identities(&self, handle: u32) -> [u32; 4] {
        self.sets[handle as usize]
    }

    /// The record offered as `handle` where it is flat, as its fragments carry it
    /// ([`Fragment::record`]); `None` for one with no plane of its own.
    ///
    /// [`Fragment::record`]: crate::composite::Fragment::record
    pub(crate) fn flat_record(&self, handle: u32) -> Option<&Offer> {
        let offer = &self.offers[handle as usize];
        offer.flat.then_some(offer)
    }

    /// Whether the record with no plane of its own offered as `handle` attached to flat records'
    /// identities (once settled), and is read on their planes.
    pub(crate) fn attached(&self, handle: u32) -> bool {
        self.attached[handle as usize]
    }

    /// Identity `identity`'s plane (once settled).
    pub(crate) fn plane(&self, identity: u32) -> WorldPlane {
        self.planes[identity as usize - 1]
    }

    /// How many records lay on more identities than they carry ([`MOST_IDENTITIES`]).
    pub fn capped(&self) -> usize {
        self.capped
    }
}

/// How far `p` may lie off `plane` along its normal, from the plane its keys are held to (its
/// unrounded normal through its exact offset, [`Projection::key_plane`]): as computed, with the
/// rounding of computing it (`4u` of its terms) and the offset's bound, as [`departure`] takes it.
///
/// [`Projection::key_plane`]: crate::project::Projection::key_plane
fn off(plane: &WorldPlane, p: [f64; 3]) -> f64 {
    let u = plane.unrounded;
    let terms = dot(u.normal.map(f64::abs), p.map(f64::abs)) + u.offset.abs();
    (dot(u.normal, p) - u.offset).abs() + 4.0 * U * terms + u.offset_err
}

/// How far `p` lies off `plane` at least: [`off`] less its rounding and the offset's bound.
fn within(plane: &WorldPlane, p: [f64; 3]) -> f64 {
    let u = plane.unrounded;
    let terms = dot(u.normal.map(f64::abs), p.map(f64::abs)) + u.offset.abs();
    ((dot(u.normal, p) - u.offset).abs() - 4.0 * U * terms - u.offset_err).max(0.0)
}

/// How two flat records' own planes stand to each other, as [`near_parallel`] finds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tilt {
    /// Whether the two are near-parallel.
    pub(crate) near: bool,
    /// Their acute dihedral `θ`, as computed.
    pub(crate) dihedral: f64,
    /// `θ_A + θ_B`, as computed, before the rounding the test takes against itself.
    pub(crate) room: f64,
}

/// Whether flat records `a` and `b` (each offered with its own plane, [`Registry::offer_flat`]) are
/// near-parallel: whether the snapped input leaves their own planes' relative orientation
/// undecided. Two such records on no one identity tie where their keys lie within the snap's term
/// of each other (`order::coplanar_tied`), as two on one identity do. Their identities answer
/// whether one plane holds both within their tolerances over their whole extents; at a pixel the
/// question is whether the input determines how the two are turned to each other. Where it does
/// not, their order there is the snap's noise, which changes from frame to frame, and two such
/// sheets ordered by it shimmer.
///
/// Each record `R` is keyed on its own plane `F_R` (its unrounded unit normal `n_R`), fitted to
/// its snapped points; content put those points on a plane `P_R`. `P_R` holds the snapped points
/// within the snap's departure `δ_R` along `n_R` ([`snap_departure`]), and `F_R` holds them within
/// their departure `dep_R` from it ([`departure`]), so at each point of `R` the two planes part by
/// at most `τ_R = δ_R + dep_R`, its tolerance along its own normal (the registry's).
///
/// Let `a = n_A × n_B / |n_A × n_B|`, the axis of the two planes' dihedral, and `θ` that dihedral,
/// acute. A turn of a plane by `φ` about a line along `a` moves it, at a point `s` across the line
/// (`s = p · (n_R × a)` less the line's), by `φ s`. `R`'s points span
/// `L_R = max - min` of `p · (n_R × a)` over all of them: its whole extent across `a`, which its
/// own plane was fitted over. Turned about the middle of that span by `φ`, `F_R` stays within
/// `τ_R` of where it was at every point of `R` while `φ L_R / 2 <= τ_R`; turned about any line along
/// `a`, it moves by `φ L_R` more at one end of the span than at the other, so by more than `τ_R` at
/// one of them once `φ > 2 τ_R / L_R`. So `θ_R = 2 τ_R / L_R` is the most a plane turned from
/// `F_R` about `a` may turn and still hold `R`'s points within `τ_R` of `F_R`.
///
/// The two are near-parallel when `θ <= θ_A + θ_B`: then `P_A` and `P_B` may be parallel, each
/// turned about `a` from its own plane by at most its `θ_R`, and only where the two lie (their keys
/// at the pixel) can decide them. Beyond it, no turn about `a` that each one's snap allows makes
/// them parallel: they cross at an angle the input determines, and depth orders them beyond their
/// error bounds. Only turns about `a` count. A record whose points allow a larger turn about `a`
/// together with one across it (a slanted sliver) is held to the smaller. So the test can keep
/// depth order where the input does not decide it, but never ties a pair the input decides.
///
/// It is a test of two records, symmetric in them (swapping them negates `a`, which leaves each
/// `L_R`, `θ_R` and their sum the same bits), and made of their own inputs alone: it makes no
/// identity and does not chain (`A` near-parallel to `B` and `B` to `C` says nothing of `A` and
/// `C`). A small record has a large `θ_R`, as it should. A record with no plane of its own (a line,
/// a curve) has no `n_R` and is never tested.
///
/// Rounding is taken against the test: the dihedral rounded up, and each `θ_R` and their sum down,
/// by bounds on each step's rounding (`u = 2^-53`; each normal within `4u` of unit length):
/// - `n_A × n_B` lies within `8u` of the exact product and `|n_A · n_B|` within `4u`, so `atan2`
///   of their lengths, its own error under `4u` of itself, lies within `16u` and `16u` of itself of
///   the exact dihedral;
/// - the axis, from a cross product of length `s` at least `32u`, lies within `32u / s + 4u` of the
///   exact one, and so `n_R × a` within `e = 40u / s + 16u`, rounding included; a coordinate
///   across `a` of a point `d` from `R`'s first point, taken from it, lies within `d (e + 16u)` of
///   the exact one (for `e` at most 1); so `L_R` is taken as its computed extent and twice that at
///   the farthest point from the first, `far`;
/// - no extent of `R` exceeds `2 far`, which holds `L_R` where the axis is less well known (`s`
///   under `32u`, or `e` past 1); the extent is then rounded up by `8u` of itself;
/// - `2 τ_R / L_R` and the sum are each rounded down by `2u` of themselves.
///
/// That moves the test by about `1e-14` rad where `θ_R` is about `1e-5` (the writer's grid,
/// `2^-14`, over a few units).
pub(crate) fn near_parallel(a: &Offer, b: &Offer) -> Tilt {
    let normal = |r: &Offer| {
        r.own
            .expect("a flat record has its own plane")
            .unrounded
            .normal
    };
    let (na, nb) = (normal(a), normal(b));
    let c = cross(na, nb);
    let s = length(c);
    let dihedral = libm::atan2(s, dot(na, nb).abs());
    let high = dihedral * (1.0 + 16.0 * U) + 16.0 * U;
    let off_axis = if s >= 32.0 * U {
        40.0 * U / s + 16.0 * U
    } else {
        f64::INFINITY
    };
    let axis = c.map(|v| v / s);
    // `θ_R` as computed, and rounded down.
    let turn = |r: &Offer| -> (f64, f64) {
        let across = cross(normal(r), axis);
        let first = r.points[0];
        let (mut lo, mut hi, mut far) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
        for p in &r.points {
            let q = sub(*p, first);
            let t = dot(q, across);
            (lo, hi, far) = (lo.min(t), hi.max(t), far.max(length(q)));
        }
        let extent = if s > 0.0 { hi - lo } else { 2.0 * far };
        let read = if off_axis <= 1.0 {
            extent + 2.0 * far * (off_axis + 16.0 * U)
        } else {
            f64::INFINITY
        };
        let span = read.min(2.0 * far) * (1.0 + 8.0 * U);
        let twice = 2.0 * r.own_tolerance;
        (twice / extent, twice / span * (1.0 - 2.0 * U))
    };
    let ((ta, ta_low), (tb, tb_low)) = (turn(a), turn(b));
    Tilt {
        near: high <= (ta_low + tb_low) * (1.0 - 2.0 * U),
        dihedral,
        room: ta + tb,
    }
}

/// The order the registry partitions flat records in, and groups the rest: the least spread first
/// (the records whose own planes stand surest), then by the bucket of their own plane, its
/// coefficients rounded and unrounded, their points, their departure from it, their placements
/// and their shapes' points. Total: two records it does not tell apart are the same record in
/// every number the registry reads, and either order of the two gives one partition.
fn canonical(a: &Offer, b: &Offer) -> std::cmp::Ordering {
    let header = |o: &Offer| -> [f64; 18] {
        let mut h = [0.0; 18];
        h[0] = o.spread;
        if let Some(p) = o.own {
            h[1] = 1.0;
            for (k, v) in Registry::bucket(&p).into_iter().enumerate() {
                h[2 + k] = v as f64;
            }
            h[6..9].copy_from_slice(&p.normal);
            h[9] = p.offset;
            h[10..13].copy_from_slice(&p.unrounded.normal);
            h[13] = p.unrounded.offset;
            h[14] = p.unrounded.offset_err;
        }
        h[15] = o.dep;
        h[16] = f64::from(u8::from(o.flat));
        h[17] = o.axis.map_or(-1.0, |k| k as f64);
        h
    };
    let numbers = |a: &[f64], b: &[f64]| {
        a.len().cmp(&b.len()).then_with(|| {
            a.iter()
                .zip(b)
                .map(|(x, y)| x.total_cmp(y))
                .find(|o| o.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    };
    numbers(&header(a), &header(b))
        .then_with(|| numbers(a.points.as_flattened(), b.points.as_flattened()))
        .then_with(|| {
            let places =
                |o: &Offer| -> Vec<f64> { o.places.iter().flatten().flatten().copied().collect() };
            numbers(&places(a), &places(b))
        })
        .then_with(|| numbers(a.shape.as_flattened(), b.shape.as_flattened()))
}

/// The plane `z = a u + b v + c` nearest the rows `(u, v, z, w)` in the minimax sense, each `w`
/// positive: the least `t` with every `|z - a u - b v - c| <= t w`, and `[a, b, c, t]`; `None`
/// where the rows' `(u, v)` lie on a line or a step's arithmetic fails.
///
/// The linear program's dual, solved by the simplex method: weights `y` on the rows' two signs
/// (column `(s u, s v, s, w)`, cost `s z`, for `s` = 1 and -1) with the weighted columns summing to
/// `(0, 0, 0, 1)`, maximising the weighted costs, whose optimum is the least `t`, and whose basis's
/// multipliers are `[a, b, c, t]` (weak duality: any weights' cost is a weighted sum of residuals,
/// at most `t`). Its four rows start from the rows spanning the widest triangle, the first one's
/// two signs weighted alike (cost 0: the plane through the three); the row whose residual most
/// passes `t w` enters (the first such once four steps in a row stall: Bland's rule, which does
/// not cycle), until none does, within a billionth of `t w` and the rounding of its residual.
fn chebyshev(rows: &[[f64; 4]]) -> Option<[f64; 4]> {
    let column = |j: usize| {
        let r = rows[j / 2];
        let s = if j.is_multiple_of(2) { 1.0 } else { -1.0 };
        [s * r[0], s * r[1], s, r[3]]
    };
    let cost = |j: usize| {
        if j.is_multiple_of(2) {
            rows[j / 2][2]
        } else {
            -rows[j / 2][2]
        }
    };
    let flat: Vec<[f64; 3]> = rows.iter().map(|r| [r[0], r[1], 0.0]).collect();
    let [p, q, r] = widest_triangle(&flat);
    let spans = length(cross(sub(flat[q], flat[p]), sub(flat[r], flat[p])));
    if spans.is_nan() || spans <= 0.0 {
        return None;
    }
    let mut basis = [2 * p, 2 * p + 1, 2 * q, 2 * r];
    let mut stalled = 0;
    for _ in 0..64 + 4 * rows.len() {
        let m: [[f64; 4]; 4] =
            std::array::from_fn(|i| std::array::from_fn(|k| column(basis[k])[i]));
        let mt: [[f64; 4]; 4] = std::array::from_fn(|i| std::array::from_fn(|k| m[k][i]));
        let level = solve4(m, [0.0, 0.0, 0.0, 1.0])?;
        let fit = solve4(mt, basis.map(cost))?;
        let [a, b, c, t] = fit;
        let bland = stalled >= 4;
        let mut enter: Option<(usize, f64)> = None;
        for (i, row) in rows.iter().enumerate() {
            let residual = row[2] - a * row[0] - b * row[1] - c;
            let rounding =
                8.0 * U * (row[2].abs() + (a * row[0]).abs() + (b * row[1]).abs() + c.abs());
            let excess = residual.abs() - row[3] * t;
            let room = 1e-9 * row[3] * t.abs() + rounding;
            if excess.partial_cmp(&room) != Some(std::cmp::Ordering::Greater) {
                continue;
            }
            let j = 2 * i + usize::from(residual < 0.0);
            if basis.contains(&j) {
                continue;
            }
            let score = excess / row[3];
            if enter.is_none_or(|(_, best)| score > best) {
                enter = Some((j, score));
            }
            if bland {
                break;
            }
        }
        let Some((j, _)) = enter else {
            return Some(fit);
        };
        let step = solve4(m, column(j))?;
        let big = step.iter().fold(0.0f64, |s, v| s.max(v.abs()));
        let mut leave: Option<(usize, f64)> = None;
        for (k, d) in step.iter().enumerate() {
            if *d > 1e-12 * big {
                let ratio = level[k].max(0.0) / d;
                if leave
                    .is_none_or(|(l, best)| ratio < best || ratio == best && basis[k] < basis[l])
                {
                    leave = Some((k, ratio));
                }
            }
        }
        let (k, ratio) = leave?;
        stalled = if ratio > 0.0 { 0 } else { stalled + 1 };
        basis[k] = j;
    }
    None
}

/// `m x = rhs` for a 4 by 4 `m` (`m[row][column]`), by Gaussian elimination with partial pivoting;
/// `None` for a pivot within `1e-13` of `m`'s largest entry of zero.
fn solve4(m: [[f64; 4]; 4], rhs: [f64; 4]) -> Option<[f64; 4]> {
    let scale = m.iter().flatten().fold(0.0f64, |s, v| s.max(v.abs()));
    let mut a: [[f64; 5]; 4] = std::array::from_fn(|r| {
        let mut row = [0.0; 5];
        row[..4].copy_from_slice(&m[r]);
        row[4] = rhs[r];
        row
    });
    for col in 0..4 {
        let pivot = (col..4).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col].abs().partial_cmp(&(1e-13 * scale)) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
        a.swap(col, pivot);
        let row = a[col];
        for below in a.iter_mut().skip(col + 1) {
            let f = below[col] / row[col];
            for (v, p) in below[col..].iter_mut().zip(&row[col..]) {
                *v -= f * p;
            }
        }
    }
    let mut x = [0.0; 4];
    for r in (0..4).rev() {
        let s = (r + 1..4).fold(a[r][4], |s, k| s - a[r][k] * x[k]);
        x[r] = s / a[r][r];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Twice the signed area of the triangle `p, q, r` on a plane.
fn twice(p: [f64; 2], q: [f64; 2], r: [f64; 2]) -> f64 {
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
}

/// How fast the difference of a plane of unit normal `n` and one of unit normal `v`, `turn / 2`
/// apart at most, may grow on `v`'s plane per unit of the two coordinates other than `n`'s
/// largest component ([`Reach::bound`]): `sqrt(3) a / (1 - sqrt(3) a)` for `a = turn / 2`, its
/// denominator taken less its rounding, and infinite where that is not positive.
fn turned_slope(turn: f64) -> f64 {
    let steep = 3f64.sqrt() * turn / 2.0;
    let room = 1.0 - steep - 4.0 * f64::EPSILON;
    if room > 0.0 {
        steep / room
    } else {
        f64::INFINITY
    }
}

/// The convex hull of `points` on a plane, counterclockwise from the least by x then y, without
/// points on its edges (Andrew's monotone chain); its one or two points where they lie on a point
/// or a line, none for none.
fn convex_hull(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    points.retain(|p| p[0].is_finite() && p[1].is_finite());
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    points.dedup();
    if points.len() < 3 {
        return points;
    }
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(2 * points.len());
    let turns = |hull: &[[f64; 2]], p: [f64; 2]| {
        twice(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
    };
    for p in &points {
        while hull.len() >= 2 && turns(&hull, *p) {
            hull.pop();
        }
        hull.push(*p);
    }
    let lower = hull.len() + 1;
    for p in points.iter().rev().skip(1) {
        while hull.len() >= lower && turns(&hull, *p) {
            hull.pop();
        }
        hull.push(*p);
    }
    hull.pop();
    hull
}

/// The least width of the convex polygon `hull` (counterclockwise): the least, over its edges, of
/// the farthest of its corners from the edge's line (rotating calipers); nought for fewer than three
/// corners.
fn least_width(hull: &[[f64; 2]]) -> f64 {
    let n = hull.len();
    if n < 3 {
        return 0.0;
    }
    let (mut far, mut least) = (1, f64::INFINITY);
    for i in 0..n {
        let (a, b) = (hull[i], hull[(i + 1) % n]);
        for _ in 0..n {
            if twice(a, b, hull[(far + 1) % n]) > twice(a, b, hull[far]) {
                far = (far + 1) % n;
            } else {
                break;
            }
        }
        let length = libm::hypot(b[0] - a[0], b[1] - a[1]);
        least = least.min(twice(a, b, hull[far]) / length);
    }
    least
}

/// How far `p` lies from the convex polygon `hull` (counterclockwise; one or two corners for a
/// point or a segment), nought in it; infinite for no corners.
fn hull_distance(hull: &[[f64; 2]], p: [f64; 2]) -> f64 {
    let n = hull.len();
    let to_segment = |a: [f64; 2], b: [f64; 2]| {
        let (u, w) = ([b[0] - a[0], b[1] - a[1]], [p[0] - a[0], p[1] - a[1]]);
        let along = u[0] * u[0] + u[1] * u[1];
        let t = if along > 0.0 {
            ((u[0] * w[0] + u[1] * w[1]) / along).clamp(0.0, 1.0)
        } else {
            0.0
        };
        libm::hypot(w[0] - t * u[0], w[1] - t * u[1])
    };
    match n {
        0 => f64::INFINITY,
        1 => to_segment(hull[0], hull[0]),
        _ if n >= 3 && (0..n).all(|i| twice(hull[i], hull[(i + 1) % n], p) >= 0.0) => 0.0,
        _ => (0..n)
            .map(|i| to_segment(hull[i], hull[(i + 1) % n]))
            .fold(f64::INFINITY, f64::min),
    }
}

/// The part of the segment from `a` to `b` in the convex polygon `hull` (counterclockwise, three
/// corners or more), as its least and greatest parameters along it (Cyrus and Beck); `None` where
/// it misses the polygon.
fn clip(hull: &[[f64; 2]], a: [f64; 2], b: [f64; 2]) -> Option<(f64, f64)> {
    let n = hull.len();
    if n < 3 {
        return None;
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for i in 0..n {
        let (p, q) = (hull[i], hull[(i + 1) % n]);
        let (g0, g1) = (twice(p, q, a), twice(p, q, b));
        if g0 < 0.0 && g1 < 0.0 {
            return None;
        }
        if g0 < 0.0 {
            lo = lo.max(g0 / (g0 - g1));
        } else if g1 < 0.0 {
            hi = hi.min(g0 / (g0 - g1));
        }
        if lo > hi {
            return None;
        }
    }
    Some((lo, hi))
}

/// The index of `n`'s largest component in magnitude, the first of equals.
fn largest_component(n: [f64; 3]) -> usize {
    (0..3)
        .max_by(|a, b| n[*a].abs().total_cmp(&n[*b].abs()).then(b.cmp(a)))
        .expect("three components")
}

/// The widest triangle on `points` as a greedy pick finds it, by index: the first point, the one
/// farthest from it, and the one farthest from the line through those two (the last of equals).
fn widest_triangle(points: &[[f64; 3]]) -> [usize; 3] {
    let far = |by: &dyn Fn(usize) -> f64| {
        (0..points.len())
            .max_by(|a, b| by(*a).total_cmp(&by(*b)))
            .unwrap_or(0)
    };
    let q = far(&|k| length(sub(points[k], points[0])));
    let r = far(&|k| length(cross(sub(points[q], points[0]), sub(points[k], points[0]))));
    [0, q, r]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn length(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

/// A bound on [`snap_departure`] along every unit normal: each `|C^T n|_1` there at most `sqrt(3)`
/// times the Frobenius norm of `C`, and the placing's `|n| . A` at most the length of `A`.
pub(crate) fn snap_departure_bound(
    grid: f64,
    places: &[crate::xf::Affine; 2],
    points: &[[f64; 3]],
) -> f64 {
    use crate::xf::{IDENTITY, place};
    let snapped = grid > 0.0;
    let l1 = |v: [f64; 3]| v[0].abs() + v[1].abs() + v[2].abs();
    let [outer, inner] = places;
    let largest = |by: Option<&crate::xf::Affine>| {
        points
            .iter()
            .map(|p| l1(by.map_or(*p, |m| place(m, *p))))
            .fold(0.0, f64::max)
    };
    let frobenius = |c: &[[f64; 3]; 3]| c.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    let mut c = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut moved = 0.0;
    for (m, by) in [(outer, Some(inner)), (inner, None)] {
        if !snapped || *m == IDENTITY {
            continue;
        }
        moved += 3f64.sqrt() * frobenius(&c) * (grid / 2.0 + RATIO_GRID / 2.0 * largest(by));
        c = std::array::from_fn(|r| {
            std::array::from_fn(|k| (0..3).map(|l| c[r][l] * m[l][k]).sum())
        });
    }
    if snapped {
        moved += grid / 2.0 * 3f64.sqrt() * frobenius(&c);
    }
    let abs = |m: &crate::xf::Affine| m.map(|row| row.map(f64::abs));
    let (outer_abs, inner_abs) = (abs(outer), abs(inner));
    let size = points
        .iter()
        .map(|p| length(place(&outer_abs, place(&inner_abs, p.map(f64::abs)))))
        .fold(0.0, f64::max);
    (moved + 8.0 * U * size) * (1.0 + 1e-12)
}

/// Whether `points` lie on a line, to within rounding and their snap to `grid` (`0.0`: not
/// snapped): the normal of their widest triangle on the first point points nowhere
/// ([`points_somewhere`]). Such a record has no plane of its own, whatever its writer coded it.
pub fn on_a_line(points: &[[f64; 3]], grid: f64) -> bool {
    let Some(&first) = points.first() else {
        return true;
    };
    !points_somewhere(
        widest_normal(points, first),
        extent(points),
        perimeter(points),
        grid,
    )
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

    /// The writer's grid for a 14.2-unit frame 854 pixels wide: 2^-14.
    const GRID: f64 = 1.0 / 16384.0;

    const NONE: [crate::xf::Affine; 2] = [crate::xf::IDENTITY, crate::xf::IDENTITY];

    fn registry() -> Registry {
        Registry {
            grid: GRID,
            ..Registry::default()
        }
    }

    /// The square on corners `c`, in CE's cubic layout.
    fn ring(c: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for (i, a) in c.iter().enumerate() {
            let b = c[(i + 1) % c.len()];
            let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
            out.extend([*a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
        }
        out
    }

    /// Offers the flat record on `points` (unplaced, its shape its points), fitted as Newell's.
    fn flat(r: &mut Registry, points: &[[f64; 3]]) -> u32 {
        let own = WorldPlane::fit(points, Planarity::Newell).expect("a plane");
        r.offer_flat(own, points, &NONE, points)
    }

    /// The records offered to `r`, settled, grouped by identity: what the partition is, whatever
    /// numbers the identities carry and in whatever order the records came.
    fn groups(r: &Registry, of: impl Fn(u32) -> usize) -> std::collections::BTreeSet<Vec<usize>> {
        let mut by: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
        for h in 0..r.offers.len() as u32 {
            for id in r.identities(h).into_iter().filter(|id| *id != 0) {
                by.entry(id).or_default().push(of(h));
            }
        }
        by.into_values()
            .map(|mut g| {
                g.sort_unstable();
                g
            })
            .collect()
    }

    /// Whether every member of every identity of settled `r` lies on its identity's plane within
    /// its own tolerance along that plane's normal.
    fn linked(r: &Registry) -> bool {
        (0..r.offers.len() as u32).all(|h| {
            let id = r.identities(h)[0];
            !r.offers[h as usize].flat || id != 0 && r.holds(&r.plane(id), h as usize)
        })
    }

    /// The minimax fit of a square's corners with one raised by `h`, all weighed alike: the plane
    /// halfway between the two diagonals' planes leaves each corner `h / 4` off, which no plane
    /// betters; and at random rows every residual lies within the fit's share of its weight, which
    /// no small move of the plane lowers.
    #[test]
    fn the_minimax_fit_is_the_chebyshev_plane() {
        let rows = [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.5, 1.0],
        ];
        let [_, _, _, t] = chebyshev(&rows).expect("a fit");
        assert!((t - 0.125).abs() < 1e-15, "{t}");
        let mut s = Seeded(0x5eed_2026_1107);
        let unit = |s: &mut Seeded| s.below(1 << 20) as f64 / (1 << 20) as f64;
        for case in 0..2000 {
            let n = 3 + s.below(40) as usize;
            let rows: Vec<[f64; 4]> = (0..n)
                .map(|_| {
                    let (u, v) = (unit(&mut s) * 4.0 - 2.0, unit(&mut s) * 4.0 - 2.0);
                    let z = 0.3 * u - 0.2 * v + 1e-4 * (unit(&mut s) - 0.5);
                    [u, v, z, 0.5 + unit(&mut s)]
                })
                .collect();
            let Some([a, b, c, t]) = chebyshev(&rows) else {
                continue;
            };
            let worst = |a: f64, b: f64, c: f64| {
                rows.iter()
                    .map(|r| (r[2] - a * r[0] - b * r[1] - c).abs() / r[3])
                    .fold(0.0, f64::max)
            };
            assert!(worst(a, b, c) <= t * (1.0 + 1e-8) + 1e-15, "case {case}");
            for _ in 0..20 {
                let d = |s: &mut Seeded| (unit(s) - 0.5) * 1e-6;
                let moved = worst(a + d(&mut s), b + d(&mut s), c + d(&mut s));
                assert!(moved >= t * (1.0 - 1e-8), "case {case}: {moved} below {t}");
            }
        }
    }

    /// Two parallel squares facing z, one unplaced (its snap's departure half a grid step) and one
    /// placed from coordinates in the thousands (its placement's snapped linear part moving its
    /// points by up to about another step), apart by just less and just more than the two own
    /// tolerances together: one identity, then two. A plane between them holds each within its
    /// own tolerance exactly as far as their sum; the smaller tolerance for both would part them
    /// at the nearer gap, the larger hold them at the farther.
    #[test]
    fn each_member_is_held_within_its_own_tolerance() {
        let square = |z: f64| ring(&[[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]]);
        let local = ring(&[
            [-1000.0, -1000.0, 0.0],
            [1000.0, -1000.0, 0.0],
            [1000.0, 1000.0, 0.0],
            [-1000.0, 1000.0, 0.0],
        ]);
        let placed = |z: f64| -> crate::xf::Affine {
            let s = 1.0 / 2000.0;
            [[s, 0.0, 0.0, 0.5], [0.0, s, 0.0, 0.5], [0.0, 0.0, s, z]]
        };
        let n = [0.0, 0.0, 1.0];
        let tol_a = {
            let p = square(0.0);
            let own = WorldPlane::fit(&p, Planarity::Newell).unwrap();
            snap_departure(GRID, n, &NONE, &p) + departure(&own, &p)
        };
        let tol_b = |z: f64| {
            let places = [crate::xf::IDENTITY, placed(z)];
            let p: Vec<[f64; 3]> = local
                .iter()
                .map(|q| crate::xf::place(&places[1], *q))
                .collect();
            let own = WorldPlane::fit(&p, Planarity::Newell).unwrap();
            snap_departure(GRID, n, &places, &local) + departure(&own, &p)
        };
        assert!(tol_b(0.0) > 2.5 * tol_a, "{} against {tol_a}", tol_b(0.0));
        for (share, one) in [(0.999, true), (1.001, false)] {
            let gap = share * (tol_a + tol_b(0.0));
            let mut r = registry();
            let a = flat(&mut r, &square(0.0));
            let places = [crate::xf::IDENTITY, placed(gap)];
            let p: Vec<[f64; 3]> = local
                .iter()
                .map(|q| crate::xf::place(&places[1], *q))
                .collect();
            let own = WorldPlane::fit(&p, Planarity::Newell).unwrap();
            let b = r.offer_flat(own, &p, &places, &local);
            r.settle();
            assert_eq!(
                r.identities(a) == r.identities(b),
                one,
                "{share} of the two tolerances apart: {:?} and {:?}",
                r.identities(a),
                r.identities(b)
            );
            assert!(linked(&r));
        }
    }

    /// A square 2 across facing z, and inside it a small quad whose corners stand `b` above and
    /// below alternately (a writer's flat face its snap bowed), `b` past both snaps' departures
    /// together but within the quad's own points' departure from its own plane: the quad's
    /// tolerance holds that departure, so they share an identity; its snap's departure alone would
    /// part them.
    #[test]
    fn a_member_s_departure_from_its_own_plane_is_in_its_tolerance() {
        let mut r = registry();
        let big = ring(&[
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]);
        let snaps = 2.0 * snap_departure(GRID, [0.0, 0.0, 1.0], &NONE, &big);
        let b = 1.5 * snaps;
        let small = ring(&[
            [-0.1, -0.1, b],
            [0.1, -0.1, -b],
            [0.1, 0.1, b],
            [-0.1, 0.1, -b],
        ]);
        let own = WorldPlane::fit(&small, Planarity::Newell).unwrap();
        assert!(
            departure(&own, &small) > b,
            "the quad departs from its own plane"
        );
        let (one, two) = (flat(&mut r, &big), flat(&mut r, &small));
        r.settle();
        assert_eq!(r.identities(one), r.identities(two));
        assert!(linked(&r));
    }

    /// Three unit squares facing z in a row, the middle one raised by 1.98 times its tolerance:
    /// the plane halfway up holds each within 0.99 of its tolerance, while the least-squares plane
    /// of their points, a third of the way up, leaves the middle one 1.32 of its tolerance off. In
    /// every order offered the three are one identity, on a plane holding each within its own.
    #[test]
    fn a_group_held_only_by_its_minimax_plane_is_one_identity() {
        let up = [0.0, 0.0, 1.0];
        let tolerance = |p: &[[f64; 3]]| {
            let own = WorldPlane::fit(p, Planarity::Newell).unwrap();
            snap_departure(GRID, up, &NONE, p) + departure(&own, p)
        };
        let square = |x: f64, z: f64| {
            ring(&[
                [x, 0.0, z],
                [x + 1.0, 0.0, z],
                [x + 1.0, 1.0, z],
                [x, 1.0, z],
            ])
        };
        let rise = 1.98 * tolerance(&square(2.0, 0.0));
        let squares = [square(0.0, 0.0), square(2.0, rise), square(4.0, 0.0)];
        let worst = squares
            .iter()
            .map(|s| tolerance(&s[..]))
            .fold(f64::INFINITY, f64::min);
        // The least-squares plane z = a x + b y + c of all their points (the normal equations,
        // by Cramer's rule), and how far it leaves the middle square, as a share of its tolerance.
        let points: Vec<[f64; 3]> = squares.iter().flatten().copied().collect();
        let mut m = [[0.0; 3]; 3];
        let mut v = [0.0; 3];
        for p in &points {
            let row = [p[0], p[1], 1.0];
            for ((line, value), r) in m.iter_mut().zip(v.iter_mut()).zip(row) {
                for (entry, s) in line.iter_mut().zip(row) {
                    *entry += r * s;
                }
                *value += r * p[2];
            }
        }
        let det = |m: [[f64; 3]; 3]| dot(m[0], cross(m[1], m[2]));
        let solved: [f64; 3] = std::array::from_fn(|k| {
            let mut with = m;
            for (row, value) in with.iter_mut().zip(v) {
                row[k] = value;
            }
            det(with) / det(m)
        });
        let off_least_squares = squares[1]
            .iter()
            .map(|p| (p[2] - solved[0] * p[0] - solved[1] * p[1] - solved[2]).abs())
            .fold(0.0, f64::max);
        assert!(
            off_least_squares > 1.3 * worst,
            "the least-squares plane leaves the middle square {off_least_squares:e} off, against \
             {worst:e}"
        );
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut r = registry();
            let handles = order.map(|k| flat(&mut r, &squares[k]));
            r.settle();
            let ids = handles.map(|h| r.identities(h));
            assert!(
                ids[0][0] != 0 && ids.iter().all(|i| *i == ids[0]),
                "{order:?}: {ids:?}"
            );
            assert!(linked(&r), "{order:?}");
        }
    }

    /// A line along the edge where two squares meet, its points sharing x and y (its writer codes
    /// it on the axis plane of x): points on a line have no plane of their own, whatever their
    /// code, so it lies on both squares' identities.
    #[test]
    fn a_line_coded_on_an_axis_plane_lies_on_every_identity_that_holds_it() {
        let mut r = registry();
        let x_face = flat(
            &mut r,
            &ring(&[
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
            ]),
        );
        let y_face = flat(
            &mut r,
            &ring(&[
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
            ]),
        );
        let edge = [[0.5, 0.5, -0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]];
        let (line, is_flat, _) = r.offer_path(&edge, Planarity::Axis(0), &NONE, &edge);
        assert!(!is_flat);
        r.settle();
        let ids = r.identities(line);
        for face in [x_face, y_face] {
            let id = r.identities(face)[0];
            assert!(
                id != 0 && ids.contains(&id),
                "the line on {ids:?}, a face on {id}"
            );
        }
    }

    /// A mesh triangle whose corners lie on a line sharing z (the plane through them the axis
    /// plane of z) has no plane of its own: on a square on that plane it attaches to the square's
    /// identity; with nothing to hold it, it shares an identity with another such triangle on that
    /// axis plane, as lines coded on an axis plane do.
    #[test]
    fn a_triangle_on_a_line_attaches_to_what_holds_it_and_else_keeps_its_axis_plane() {
        let z = 0.25;
        let sliver = |y: f64| [[-0.5, y, z], [0.5, y, z], [0.0, y + GRID, z]];
        let offer = |r: &mut Registry, at: [[f64; 3]; 3]| {
            let plane = WorldPlane::of_triangle(at, GRID);
            assert!(plane.is_some(), "the axis plane through {at:?}");
            r.offer_triangle(plane, &at, &NONE, &at)
        };
        let mut r = registry();
        let square = ring(&[
            [-1.0, -1.0, z],
            [1.0, -1.0, z],
            [1.0, 1.0, z],
            [-1.0, 1.0, z],
        ]);
        let face = flat(&mut r, &square);
        let (on, is_flat) = offer(&mut r, sliver(0.0));
        assert!(!is_flat);
        r.settle();
        assert!(r.attached(on));
        assert_eq!(r.identities(on), r.identities(face));
        let mut r = registry();
        let (one, _) = offer(&mut r, sliver(0.0));
        let (two, _) = offer(&mut r, sliver(0.5));
        r.settle();
        assert!(!r.attached(one) && !r.attached(two));
        assert!(r.identities(one)[0] != 0 && r.identities(one) == r.identities(two));
    }

    /// Records that differ in any one number the registry reads (a point's last bit, their
    /// placements, their shapes' points, their own plane's offset bound, flat or not) never take
    /// the same place in the canonical order, and the order is antisymmetric; a record is equal
    /// only to itself.
    #[test]
    fn the_canonical_order_is_total() {
        let square = ring(&[
            [0.1, 0.2, 0.3],
            [1.1, 0.2, 0.35],
            [1.1, 1.2, 0.4],
            [0.1, 1.2, 0.35],
        ]);
        let mut nudged = square.clone();
        nudged[5][1] = f64::from_bits(nudged[5][1].to_bits() + 1);
        let own = WorldPlane::fit(&square, Planarity::Newell).unwrap();
        let mut shifted = crate::xf::IDENTITY;
        shifted[2][3] = 0.0;
        shifted[0][0] = 1.0 + f64::EPSILON;
        let mut shape = square.clone();
        shape[0][0] += 1e-9;
        let mut wider = own;
        wider.unrounded.offset_err *= 2.0;
        let mut r = registry();
        r.offer_flat(own, &square, &NONE, &square);
        r.offer_flat(
            WorldPlane::fit(&nudged, Planarity::Newell).unwrap(),
            &nudged,
            &NONE,
            &nudged,
        );
        r.offer_flat(own, &square, &[crate::xf::IDENTITY, shifted], &square);
        r.offer_flat(own, &square, &NONE, &shape);
        r.offer_flat(wider, &square, &NONE, &square);
        r.offer_planeless(None, &square, &NONE, &square);
        let offers = &r.offers;
        for (i, a) in offers.iter().enumerate() {
            assert_eq!(canonical(a, &a.clone()), std::cmp::Ordering::Equal);
            for (j, b) in offers.iter().enumerate().filter(|(j, _)| *j != i) {
                let (ab, ba) = (canonical(a, b), canonical(b, a));
                assert_ne!(ab, std::cmp::Ordering::Equal, "records {i} and {j}");
                assert_eq!(ab, ba.reverse(), "records {i} and {j}");
            }
        }
    }

    /// The faces of a slightly curved strip (`z = kappa x^2`: neighbours within a fraction of their
    /// tolerance of one plane, the ends far off any plane through the middle), each also offered
    /// as its two triangles, as a mesh of it would be, offered in eight orders, the triangles before
    /// the faces, after them, and shuffled among them: one partition every time, and in it every
    /// member lies on its identity's plane within its own tolerance.
    #[test]
    fn the_partition_does_not_depend_on_the_order_offered() {
        let side = 0.15;
        let tolerance = snap_departure(GRID, [0.0, 0.0, 1.0], &NONE, &[[1.0, 1.0, 0.0]]);
        let kappa = 0.45 * tolerance / (side * side);
        let at = |x: f64, y: f64| [x, y, kappa * x * x];
        let mut records: Vec<Vec<[f64; 3]>> = Vec::new();
        for i in 0..24 {
            let (a, b) = (-1.8 + i as f64 * side, -1.8 + (i + 1) as f64 * side);
            let c = [at(a, -side), at(b, -side), at(b, side), at(a, side)];
            records.push(ring(&c));
            records.push(vec![c[0], c[1], c[2]]);
            records.push(vec![c[0], c[2], c[3]]);
        }
        let n = records.len();
        let partition = |order: &[usize]| {
            let mut r = registry();
            for &k in order {
                flat(&mut r, &records[k]);
            }
            r.settle();
            assert!(linked(&r), "order {order:?}");
            groups(&r, |h| order[h as usize])
        };
        let faces_first: Vec<usize> = (0..n)
            .filter(|k| k % 3 == 0)
            .chain((0..n).filter(|k| k % 3 != 0))
            .collect();
        let faces_last: Vec<usize> = faces_first.iter().rev().copied().collect();
        let base = partition(&(0..n).collect::<Vec<_>>());
        assert!(base.len() > 1, "a curved strip on one identity");
        let mut s = Seeded(0x0bad_5eed);
        let mut orders = vec![faces_first, faces_last];
        for _ in 0..6 {
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, s.below(i as u64 + 1) as usize);
            }
            orders.push(order);
        }
        for order in &orders {
            assert_eq!(partition(order), base, "order {order:?}");
        }
    }

    /// Five triangles turning about one edge, 0.3 radians apart, and a line along the edge: the
    /// line lies on all five planes, keeps the first four identities, and is counted.
    #[test]
    fn a_line_on_more_planes_than_a_fragment_carries_keeps_four_and_is_counted() {
        let mut r = registry();
        let faces: Vec<u32> = (0..5)
            .map(|k| {
                let a = 0.3 * k as f64;
                let tip = [0.5, libm::cos(a), libm::sin(a)];
                flat(&mut r, &ring(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], tip]))
            })
            .collect();
        let line = [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [2.0 / 3.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let on = r.offer_planeless(None, &line, &NONE, &line);
        r.settle();
        let mut ids: Vec<u32> = faces.iter().map(|f| r.identities(*f)[0]).collect();
        ids.sort_unstable();
        assert_eq!(r.identities(on), [ids[0], ids[1], ids[2], ids[3]]);
        assert!(r.attached(on));
        assert_eq!(r.capped(), 1);
    }

    /// A square 0.1 across facing z, and a tick 0.3 long along y five units off along x, raised 20
    /// grid steps: the plane rising from the square's centre to the tick holds the square within
    /// half its tolerance and the tick on it, so one plane holds both, and five units out the bound
    /// content's plane through the square leaves a record ([`Reach::bound`]) has grown past 20
    /// steps; but the tick never enters the square's hull ([`Registry::enters`]), so it does not
    /// lie on the square's identity, whose plane would key it.
    #[test]
    fn a_record_far_off_a_small_identity_s_plane_does_not_lie_on_it() {
        let s = 0.05;
        let square = ring(&[[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]]);
        let h = 20.0 * GRID;
        let tick = [[5.0, -0.15, h], [5.0, 0.0, h], [5.0, 0.15, h]];
        let rise = h / 5.0;
        let n = [-rise, 0.0, 1.0].map(|v| v / (1.0 + rise * rise).sqrt());
        let held = |p: &[[f64; 3]]| {
            let tolerance = snap_departure(GRID, n, &NONE, p);
            p.iter().all(|q| dot(n, *q).abs() < tolerance / 2.0)
        };
        assert!(held(&square) && held(&tick));
        let mut r = registry();
        let face = flat(&mut r, &square);
        let line = r.offer_planeless(None, &tick, &NONE, &tick);
        r.settle();
        let id = r.identities(face)[0];
        assert!(id != 0);
        let (reach, offer) = (&r.reaches[id as usize - 1], &r.offers[line as usize]);
        let tolerance = r.tolerance(offer, r.plane(id).unrounded.normal);
        for p in &tick {
            let bound = reach.bound(tolerance, offer.widest, reach.distance(*p));
            assert!(bound > h, "the bound at the tick is {} steps", bound / GRID);
        }
        assert!(
            !r.attached(line) && r.identities(line) == [0; 4],
            "the tick lies on {:?}",
            r.identities(line)
        );
    }

    /// A face 2 across whose corners at `x = 1` stand 0.99 of their snap's departure below
    /// `z = 0` (its own plane tilted by the snap), and a line along that edge content put on
    /// `z = 0` too, 0.99 of its own above: the line stands off the face's plane by 0.99 of the two
    /// tolerances together, so that plane does not hold it, but `z = 0` holds both, and it stands
    /// within the bound content's plane through the face leaves it ([`Reach::bound`], 1.5 of the
    /// face's tolerance): it lies on the face's identity.
    #[test]
    fn a_line_on_a_face_s_edge_lies_on_it_off_the_face_s_plane() {
        let snap = snap_departure(GRID, [0.0, 0.0, 1.0], &NONE, &[[1.0, 1.0, 0.0]]);
        let a = 0.99 * snap;
        let face = ring(&[
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, -a],
            [1.0, 1.0, -a],
            [-1.0, 1.0, 0.0],
        ]);
        let edge = [[1.0, -1.0, a], [1.0, 0.0, a], [1.0, 1.0, a]];
        let mut r = registry();
        let id = flat(&mut r, &face);
        let line = r.offer_planeless(None, &edge, &NONE, &edge);
        r.settle();
        let plane = r.plane(r.identities(id)[0]);
        let n = plane.unrounded.normal;
        let both = r.tolerance_along(id, n) + r.tolerance_along(line, n);
        let apart = edge.iter().map(|p| within(&plane, *p)).fold(0.0, f64::max);
        assert!(
            apart > 0.98 * both && apart < both,
            "the line stands {apart:e} off the face's plane, against {both:e}"
        );
        assert!(r.attached(line));
        assert_eq!(r.identities(line), r.identities(id));
    }

    /// Two lines crossing at the origin, one coded on the axis plane of x (its points share x
    /// alone) and one on that of y, with nothing to attach to: two axis planes, two identities.
    #[test]
    fn lines_on_two_axis_planes_that_cross_share_no_identity() {
        let on_x = [[0.0, -1.0, -1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
        let on_y = [[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]];
        let mut r = registry();
        let (a, ..) = r.offer_path(&on_x, Planarity::Axis(0), &NONE, &on_x);
        let (b, ..) = r.offer_path(&on_y, Planarity::Axis(1), &NONE, &on_y);
        r.settle();
        let (on_a, on_b) = (r.identities(a), r.identities(b));
        assert!(
            on_a[0] != 0 && on_b[0] != 0 && on_a != on_b,
            "{on_a:?} and {on_b:?}"
        );
    }

    /// Two lines coded on the axis plane of z with nothing to attach to: at offsets one tolerance
    /// apart, two identities, though one value of z holds both within their tolerances; at one
    /// offset, one.
    #[test]
    fn lines_on_one_axis_at_offsets_a_tolerance_apart_share_no_identity() {
        let line = |y: f64, z: f64| [[-1.0, y, z], [0.0, y, z], [1.0, y, z]];
        let tolerance = snap_departure(GRID, [0.0, 0.0, 1.0], &NONE, &line(0.0, 0.0));
        for (z, one) in [(tolerance, false), (0.0, true)] {
            let (low, high) = (line(0.0, 0.0), line(0.5, z));
            let mut r = registry();
            let (a, ..) = r.offer_path(&low, Planarity::Axis(2), &NONE, &low);
            let (b, ..) = r.offer_path(&high, Planarity::Axis(2), &NONE, &high);
            r.settle();
            let (on_a, on_b) = (r.identities(a), r.identities(b));
            assert!(
                on_a[0] != 0 && on_b[0] != 0 && (on_a == on_b) == one,
                "{z:e} apart: {on_a:?} and {on_b:?}"
            );
        }
    }

    /// Lines coded on the axis plane of z across a square on that plane, past its edges: the
    /// square's plane holds them, so they attach to its identity and form none of their own. A
    /// third beside it, 0.2 past its edge, which that plane holds as well, never enters the
    /// square's hull: it does not attach, and forms its axis plane's identity.
    #[test]
    fn lines_across_a_face_on_its_axis_plane_join_its_identity_and_one_beside_it_does_not() {
        let z = 0.25;
        let mut r = registry();
        let square = ring(&[
            [-1.0, -1.0, z],
            [1.0, -1.0, z],
            [1.0, 1.0, z],
            [-1.0, 1.0, z],
        ]);
        let face = flat(&mut r, &square);
        let [across, near, beside] = [0.0, 0.9, 1.2].map(|y| {
            let points = [[-1.5, y, z], [0.0, y, z], [1.5, y, z]];
            r.offer_path(&points, Planarity::Axis(2), &NONE, &points).0
        });
        r.settle();
        for line in [across, near] {
            assert!(r.attached(line));
            assert_eq!(r.identities(line), r.identities(face));
        }
        assert!(!r.attached(beside));
        let own = r.identities(beside);
        assert!(own[0] != 0 && own != r.identities(face), "{own:?}");
        assert_eq!(r.planes.len(), 2);
    }

    /// Two squares 0.01 across facing z, 0.01 apart along x, the second raised by 4.002 times
    /// their tolerance along z: along z no plane holds both (the best leaves the four columns of
    /// corners a quarter of the rise off alternately, 1.0005 of that tolerance), but the plane
    /// rising through them, turned about 6e-3, holds each within its tolerance along its own
    /// normal, which that turn widens by 0.6%. Along z the fit's largest share passes 1 by less
    /// than [`REFIT_ROOM`], so it is fitted again along its normal: one identity.
    #[test]
    fn squares_held_only_along_a_turned_normal_are_one_identity() {
        let s = 0.01;
        let square =
            |x: f64, z: f64| ring(&[[x, 0.0, z], [x + s, 0.0, z], [x + s, s, z], [x, s, z]]);
        let tolerance = |p: &[[f64; 3]], n: [f64; 3]| {
            let own = WorldPlane::fit(p, Planarity::Newell).unwrap();
            snap_departure(GRID, n, &NONE, p) + departure(&own, p)
        };
        let up = [0.0, 0.0, 1.0];
        let worst = tolerance(&square(0.0, 0.0), up).max(tolerance(&square(2.0 * s, 0.0), up));
        let rise = 4.002 * worst;
        let squares = [square(0.0, 0.0), square(2.0 * s, rise)];
        let share = rise / 4.0 / worst;
        assert!(share > 1.0 && share < 1.0 + REFIT_ROOM, "{share}");
        // The plane z = rise (x / (2 s) - 1 / 4), leaving each column of corners a quarter of the
        // rise off along z, holds each square within its tolerance along its own normal.
        let slope = rise / (2.0 * s);
        let n = [-slope, 0.0, 1.0].map(|v| v / (1.0 + slope * slope).sqrt());
        let d = -rise / 4.0 * n[2];
        for p in &squares {
            let off = p.iter().map(|q| (dot(n, *q) - d).abs()).fold(0.0, f64::max);
            assert!(off < 0.996 * tolerance(p, n), "{off:e}");
        }
        let mut r = registry();
        let [a, b] = [0, 1].map(|k| flat(&mut r, &squares[k]));
        r.settle();
        assert!(
            r.identities(a)[0] != 0 && r.identities(a) == r.identities(b),
            "{:?} and {:?}",
            r.identities(a),
            r.identities(b)
        );
        assert!(linked(&r));
    }

    /// The square `side` across centred on the origin of the plane through it turned from z = 0
    /// by `theta` about the x axis, the plane lifted by `lift` along its normal off the points: its
    /// corners, and that plane.
    fn turned_square(side: f64, theta: f64, lift: f64) -> (Vec<[f64; 3]>, WorldPlane) {
        let (s, c) = (libm::sin(theta), libm::cos(theta));
        let h = side / 2.0;
        let points = [[-h, -h], [h, -h], [h, h], [-h, h]]
            .map(|[x, y]| [x, y * c, y * s])
            .to_vec();
        (points, WorldPlane::rounded([0.0, -s, c], lift, 0.0))
    }

    /// Two squares (see [`turned_square`]) offered alone to a registry for points snapped to
    /// `grid`, each on its own plane: `side_a` across on z = 0 and `side_b` across turned from it
    /// by `theta`, both planes lifted by `lift`. Their offers, each one's own tolerance checked
    /// against the registry's along its own normal.
    fn squares(grid: f64, theta: f64, side_a: f64, side_b: f64, lift: f64) -> (Offer, Offer) {
        let mut r = Registry {
            grid,
            ..Registry::default()
        };
        for (side, turn) in [(side_a, 0.0), (side_b, theta)] {
            let (points, own) = turned_square(side, turn, lift);
            let h = r.offer_flat(own, &points, &NONE, &points);
            let offer = &r.offers[h as usize];
            assert_eq!(
                offer.own_tolerance,
                r.tolerance_along(h, own.unrounded.normal)
            );
        }
        (r.offers[0].clone(), r.offers[1].clone())
    }

    /// The dihedral `ratio` times the turn the squares' snaps allow about the x axis, `θ_A + θ_B`
    /// (each `2 τ_R / L_R`, `L_R` its side, as the axis runs along an edge), with the turned
    /// square's tolerance read at that dihedral; and the squares at it.
    fn squares_at(
        ratio: f64,
        grid: f64,
        side_a: f64,
        side_b: f64,
        lift: f64,
    ) -> (f64, Offer, Offer) {
        let room =
            |a: &Offer, b: &Offer| 2.0 * a.own_tolerance / side_a + 2.0 * b.own_tolerance / side_b;
        let mut theta = 0.0;
        for _ in 0..8 {
            let (a, b) = squares(grid, theta, side_a, side_b, lift);
            theta = ratio * room(&a, &b);
        }
        let (a, b) = squares(grid, theta, side_a, side_b, lift);
        let exact = theta / room(&a, &b);
        assert!((exact - ratio).abs() < 1e-14 * ratio, "{exact} for {ratio}");
        (theta, a, b)
    }

    /// Two faces turned apart by a dihedral within the turn their snaps allow are near-parallel,
    /// and past it are not, at the writer's grid (2^-14) over a few units (two squares 3 across:
    /// `θ` about 4.07e-5), and with one square ten times the other across: its turn is read over
    /// its whole extent, so the larger allows a tenth of the smaller's. Just within is within
    /// the rounding the test takes against itself.
    #[test]
    fn faces_are_near_parallel_up_to_the_turn_their_snaps_allow_and_not_past_it() {
        for (side_a, side_b) in [(3.0, 3.0), (30.0, 3.0), (3.0, 30.0)] {
            for (ratio, near) in [(1.0 - 1e-8, true), (1.0 + 1e-12, false), (0.5, true)] {
                let (theta, a, b) = squares_at(ratio, GRID, side_a, side_b, 0.0);
                if side_a == side_b && ratio > 0.9 {
                    assert!(theta > 4.0e-5 && theta < 4.2e-5, "{theta:e}");
                }
                for (x, y) in [(&a, &b), (&b, &a)] {
                    let tilt = near_parallel(x, y);
                    assert_eq!(
                        tilt.near, near,
                        "{side_a} and {side_b} across, ratio {ratio}: {tilt:?} at {theta:e}"
                    );
                    assert!((tilt.dihedral / theta - 1.0).abs() < 1e-12, "{tilt:?}");
                    assert!(
                        (tilt.dihedral / tilt.room - ratio).abs() < 1e-12,
                        "{tilt:?}"
                    );
                }
            }
        }
    }

    /// Where the dihedral is so small that its axis is known only roughly (here to about 0.3%,
    /// two squares 3 across lifted 1e-12 off their own planes, on no grid, so each allows a turn of
    /// about 6.7e-13), the test reads each record's extent across it as wide as that error allows:
    /// a pair at 0.995 of the turn by the computed axis is not near-parallel, and one at 0.98 is.
    #[test]
    fn where_the_dihedral_s_axis_is_known_roughly_the_test_holds_to_the_widest_extent_it_allows() {
        for (ratio, near) in [(0.98, true), (0.995, false), (1.0 + 1e-12, false)] {
            let (theta, a, b) = squares_at(ratio, 0.0, 3.0, 3.0, 1e-12);
            assert!(theta > 1e-12 && theta < 2e-12, "{theta:e}");
            for (x, y) in [(&a, &b), (&b, &a)] {
                let tilt = near_parallel(x, y);
                assert_eq!(tilt.near, near, "ratio {ratio}: {tilt:?}");
            }
        }
    }

    /// Offers to `r` a flat record of a random outline of 3 to 6 corners 0.5 to 4.5 from the
    /// origin, on a plane through it turned up to 1e-4 from z = 0.
    fn offer_turned(r: &mut Registry, s: &mut Seeded) {
        let unit = |s: &mut Seeded| s.below(1 << 30) as f64 / (1u64 << 30) as f64;
        let theta = 1e-4 * unit(s);
        let phi = 6.0 * unit(s);
        let n = [
            libm::sin(theta) * libm::cos(phi),
            libm::sin(theta) * libm::sin(phi),
            libm::cos(theta),
        ];
        let corners = 3 + s.below(4) as usize;
        let size = 0.5 + 4.0 * unit(s);
        let points: Vec<[f64; 3]> = (0..corners)
            .map(|k| {
                let t = (k as f64 + 0.5 * unit(s)) / corners as f64 * std::f64::consts::TAU;
                let (x, y) = (size * libm::cos(t), size * libm::sin(t));
                [x, y, -(n[0] * x + n[1] * y) / n[2]]
            })
            .collect();
        let own = WorldPlane::rounded(n, 0.0, 0.0);
        r.offer_flat(own, &points, &NONE, &points);
    }

    /// The verdict a record keeps for a pair is the test's own, whichever of the two asks and
    /// however often, with more partners than a record keeps verdicts for (so kept verdicts are
    /// replaced); and a pair asked again reads its verdict rather than running the test.
    #[test]
    fn a_verdict_kept_for_a_pair_is_the_test_s_own() {
        let mut s = Seeded(0x7465_7374_6564);
        let mut r = registry();
        for _ in 0..24 {
            offer_turned(&mut r, &mut s);
        }
        let n = r.offers.len();
        let (mut near, mut kept) = (0, 0);
        for _ in 0..3 {
            for i in 0..n {
                for j in (0..n).filter(|j| *j != i) {
                    let (a, b) = (&r.offers[i], &r.offers[j]);
                    let (verdict, tilt) = (tested(a, b), near_parallel(a, b));
                    assert_eq!(
                        (verdict.near, verdict.wide),
                        (tilt.near, tilt.dihedral > WIDE),
                        "{i} and {j}: {verdict:?}, {tilt:?}"
                    );
                    near += usize::from(verdict.near);
                    kept += usize::from(!verdict.fresh);
                }
            }
        }
        assert!(near > 0 && near < 3 * n * (n - 1), "{near}");
        assert!(kept > 0, "no verdict read back");
        let (a, b) = (&r.offers[3], &r.offers[17]);
        tested(a, b);
        assert!(!tested(b, a).fresh, "a pair asked again runs the test");
    }

    /// The test is one of the pair, whichever comes first: over flat records of random outlines
    /// on planes turned a little from each other, every number it gives is the same bits both
    /// ways, and it holds both ways at once or neither.
    #[test]
    fn the_near_parallel_test_is_the_same_whichever_record_comes_first() {
        let mut s = Seeded(0x6e70_7061_6972);
        let (mut near, mut cases) = (0, 0);
        for _ in 0..400 {
            let mut r = registry();
            for _ in 0..2 {
                offer_turned(&mut r, &mut s);
            }
            let (a, b) = (&r.offers[0], &r.offers[1]);
            let (x, y) = (near_parallel(a, b), near_parallel(b, a));
            assert_eq!(x.near, y.near, "{x:?} and {y:?}");
            assert_eq!(
                x.dihedral.to_bits(),
                y.dihedral.to_bits(),
                "{x:?} and {y:?}"
            );
            assert_eq!(x.room.to_bits(), y.room.to_bits(), "{x:?} and {y:?}");
            near += usize::from(x.near);
            cases += 1;
        }
        assert!(near > 0 && near < cases, "{near} of {cases}");
    }

    /// A record with no plane of its own (points on a line, a triangle whose corners lie on one,
    /// a line on an axis plane) is no flat record, and its fragments carry none: it is never
    /// near-parallel to anything.
    #[test]
    fn a_record_with_no_plane_of_its_own_is_never_tested_for_near_parallel() {
        let mut r = registry();
        let line = [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.5, 0.0],
        ];
        let (on_line, is_flat, _) = r.offer_path(&line, Planarity::Newell, &NONE, &line);
        assert!(!is_flat);
        let (on_axis, is_flat, _) = r.offer_path(&line, Planarity::Axis(2), &NONE, &line);
        assert!(!is_flat);
        let corners = [line[0], line[1], line[3]];
        let plane = WorldPlane::of_triangle(corners, GRID);
        let (degenerate, is_flat) = r.offer_triangle(plane, &corners, &NONE, &corners);
        assert!(!is_flat);
        let square = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let face = flat(&mut r, &square);
        for h in [on_line, on_axis, degenerate] {
            assert!(r.flat_record(h).is_none(), "record {h}");
        }
        assert!(r.flat_record(face).is_some());
    }

    /// A square 0.2 across facing z and a line through its centre along x, rising at three
    /// quarters of the slope the bound grows by outside the square's hull: every point of the line
    /// stands within the bound ([`Reach::bound`]), past what it is within the hull, and the line
    /// crosses the square's plane within the hull, so it enters it; but no plane holds the
    /// square's corners within their tolerances and the line within its own, rising that steeply
    /// across the square, so it does not attach.
    #[test]
    fn a_line_piercing_a_face_s_plane_within_the_bound_s_growth_does_not_lie_on_it() {
        let s = 0.1;
        let square = ring(&[[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]]);
        let slope = {
            let mut r = registry();
            flat(&mut r, &square);
            let far = [[9.0, 9.0, 9.0], [9.5, 9.0, 9.0]];
            r.offer_planeless(None, &far, &NONE, &far);
            r.settle();
            r.reaches[0].slope
        };
        let rise = 0.75 * slope;
        let line = [-3.0, -1.0, 1.0, 3.0].map(|x| [x, 0.0, rise * x]);
        let mut r = registry();
        let face = flat(&mut r, &square);
        let on = r.offer_planeless(None, &line, &NONE, &line);
        r.settle();
        let id = r.identities(face)[0];
        let (plane, reach, offer) = (
            r.plane(id),
            &r.reaches[id as usize - 1],
            &r.offers[on as usize],
        );
        let tolerance = r.tolerance(offer, plane.unrounded.normal);
        let in_hull = reach.bound(tolerance, offer.widest, 0.0);
        for p in &line {
            let off = within(&plane, *p);
            assert!(off <= reach.bound(tolerance, offer.widest, reach.distance(*p)));
            assert!(
                off > in_hull,
                "{} steps off, {} in the hull",
                off / GRID,
                in_hull / GRID
            );
        }
        assert!(!r.attached(on), "the line lies on {:?}", r.identities(on));
    }

    /// A face 0.08 across on the plane `z = 0.3 x` and a line 5 long along x through it, content's
    /// both, snapped, at the position of a search where the line's points stand farthest off the
    /// face's own plane: the snap tilts that plane, the identity's, so the line's ends stand off it
    /// by more than twice the bound within the face's hull, and the line attaches through the
    /// bound's growth outside it. Where it meets the face ([`Registry::off_members`]) its points
    /// stand off the plane by at most the bound within the hull, not their whole departure from
    /// it ([`Registry::off_identity`]).
    #[test]
    fn a_line_attached_past_a_small_face_s_hull_carries_the_bound_within_it_as_its_share() {
        let at = |x: f64, y: f64| [x, y, 0.3 * x];
        let snapped = |p: &[[f64; 3]]| -> Vec<[f64; 3]> {
            p.iter()
                .map(|q| q.map(|v| (v / GRID).round() * GRID + 0.0))
                .collect()
        };
        let h = 0.04;
        let placed = |k: usize| {
            let (cx, cy) = (-0.6 + 0.000713 * k as f64, -0.3 + 0.000419 * k as f64);
            let face = snapped(&ring(&[
                at(cx - h, cy - h),
                at(cx + h, cy - h),
                at(cx + h, cy + h),
                at(cx - h, cy + h),
            ]));
            let line = snapped(&[0.0, 1.0, 2.0, 3.0].map(|t| at(cx - 2.5 + 5.0 * t / 3.0, cy)));
            let own = WorldPlane::fit(&face, Planarity::Newell).expect("a plane");
            (departure(&own, &line), face, line)
        };
        let (_, face, line) = (0..4000)
            .map(placed)
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .expect("positions");
        let mut r = registry();
        let id = flat(&mut r, &face);
        let on = r.offer_path(&line, Planarity::Axis(1), &NONE, &line).0;
        r.settle();
        assert!(r.attached(on) && r.identities(on) == r.identities(id));
        let id = r.identities(id)[0];
        let (plane, reach, offer) = (
            r.plane(id),
            &r.reaches[id as usize - 1],
            &r.offers[on as usize],
        );
        let tolerance = r.tolerance(offer, plane.unrounded.normal);
        let whole = tolerance + departure(&plane, &line);
        let in_hull = reach.bound(tolerance, offer.widest, 0.0);
        assert!(
            whole > 2.0 * in_hull,
            "the line stands {} steps off, {} in the hull",
            whole / GRID,
            in_hull / GRID
        );
        assert_eq!(r.off_members(on, id), in_hull);
        assert_eq!(r.off_identity(on, id), whole);
    }

    /// The five triangles turning about one edge and the line along it, offered in shuffled
    /// orders: the line keeps the same four of the five faces' identities in every order.
    #[test]
    fn a_line_on_more_planes_than_a_fragment_carries_keeps_the_same_four_in_any_order() {
        let faces: Vec<Vec<[f64; 3]>> = (0..5)
            .map(|k| {
                let a = 0.3 * k as f64;
                ring(&[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, libm::cos(a), libm::sin(a)],
                ])
            })
            .collect();
        let line = [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [2.0 / 3.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        // The faces (by index) whose identities the line keeps, its record offered at `5`.
        let kept = |order: &[usize]| -> Vec<usize> {
            let mut r = registry();
            let mut handles = [0u32; 6];
            for &k in order {
                handles[k] = match faces.get(k) {
                    Some(face) => flat(&mut r, face),
                    None => r.offer_planeless(None, &line, &NONE, &line),
                };
            }
            r.settle();
            assert!(r.attached(handles[5]) && r.capped() == 1, "order {order:?}");
            let on = r.identities(handles[5]);
            (0..5)
                .filter(|&f| on.contains(&r.identities(handles[f])[0]))
                .collect()
        };
        let base = kept(&[0, 1, 2, 3, 4, 5]);
        assert_eq!(base.len(), 4);
        let mut s = Seeded(0x5eed_f00d);
        for _ in 0..8 {
            let mut order: Vec<usize> = (0..6).collect();
            for i in (1..6).rev() {
                order.swap(i, s.below(i as u64 + 1) as usize);
            }
            assert_eq!(kept(&order), base, "order {order:?}");
        }
    }

    /// Unit normals `n` and `v` at most `a` apart: on `v`'s plane, read on the two coordinates
    /// other than `n`'s largest component, `n . x` grows by at most `sqrt(3) a / (1 - sqrt(3) a)`
    /// per unit, [`turned_slope`] of the turn `2 a`; by more than `sqrt(3) a` where `n` faces a
    /// diagonal and `v` turns from it so its component there falls fastest. Where `sqrt(3) a`
    /// reaches 1, to within rounding, the slope has no bound.
    #[test]
    fn the_slope_of_a_turn_holds_every_plane_turned_within_it() {
        let gradient = |n: [f64; 3], v: [f64; 3]| {
            let k = largest_component(n);
            let (i, j) = ((k + 1) % 3, (k + 2) % 3);
            libm::hypot(n[i] - n[k] * v[i] / v[k], n[j] - n[k] * v[j] / v[k])
        };
        let unit = |p: [f64; 3]| p.map(|c| c / length(p));
        // `n` turned toward `d`, a unit normal to it, until the two stand `a` apart.
        let turned = |n: [f64; 3], d: [f64; 3], a: f64| -> [f64; 3] {
            let phi = 2.0 * libm::asin(a / 2.0);
            std::array::from_fn(|c| libm::cos(phi) * n[c] + libm::sin(phi) * d[c])
        };
        let diagonal = unit([1.0, 1.0, 1.0]);
        let mut s = Seeded(0x0007_5106_e5ab);
        let mut uniform = move || s.below(1 << 40) as f64 / (1u64 << 39) as f64 - 1.0;
        for a in [1e-4, 1e-3, 1e-2, 1.0 / 16.0, 0.2, 0.5] {
            let bound = turned_slope(2.0 * a);
            let k = largest_component(diagonal);
            let mut fall = [1.0; 3];
            fall[k] = -2.0;
            let steepest = gradient(diagonal, turned(diagonal, unit(fall), a));
            assert!(
                steepest > 3f64.sqrt() * a * (1.0 + a / 2.0) && steepest <= bound,
                "a {a}: {steepest} against {bound}"
            );
            for case in 0..20_000 {
                let r = [uniform(), uniform(), uniform()];
                let n = if case % 2 == 0 {
                    unit(r.map(|c| c.signum() + 0.02 * c))
                } else {
                    unit(r)
                };
                let q = [uniform(), uniform(), uniform()];
                let d = unit(sub(q, n.map(|c| c * dot(q, n))));
                let g = gradient(n, turned(n, d, a));
                assert!(
                    g <= bound,
                    "a {a}: {g} against {bound} for {n:?} turned to {d:?}"
                );
            }
        }
        let edge = 2.0 / 3f64.sqrt();
        assert_eq!(turned_slope(edge), f64::INFINITY);
        assert_eq!(
            turned_slope(edge * (1.0 - 2.0 * f64::EPSILON)),
            f64::INFINITY
        );
        assert!(turned_slope(edge * (1.0 - 1e-9)) > 1e8);
        assert!(turned_slope(edge * (1.0 - 1e-9)).is_finite());
    }

    /// A square 0.2 across facing z, its points moved toward its centre line along x by the
    /// members' bound along every normal, as content may have put them (the snap moved each by no
    /// more): their hull is narrower than the drawn one by twice that bound. A difference of two
    /// planes within `g` of each of those points ([`Reach::bound`]: the members' part, and the
    /// turn's of their bound), `g` along one side and `-g` along the other, grows by `2 g` across
    /// that narrower width, where the turn admits so steep a difference ([`turned_slope`]); the
    /// slope the bound grows by outside the hull holds it. Read on the bound's own terms: whether
    /// a record attaches cannot show the narrowing's direction, as the fit (one plane holding the
    /// members and the record within their tolerances) refuses a record before the bound does
    /// wherever the bound's slope is at least the fit's, which the members' part and the turn's
    /// make it, about twice over.
    #[test]
    fn the_bound_s_slope_holds_a_difference_across_the_narrowest_hull_content_may_have_made() {
        let s = 0.1;
        let square = ring(&[[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]]);
        let mut r = registry();
        flat(&mut r, &square);
        let far = [[9.0, 9.0, 9.0], [9.5, 9.0, 9.0]];
        r.offer_planeless(None, &far, &NONE, &far);
        r.settle();
        let reach = &r.reaches[0];
        let w = reach.widest;
        let content: Vec<[f64; 2]> = square
            .iter()
            .map(|p| {
                let [x, y] = reach.flat(*p);
                [x, y - w * y.signum()]
            })
            .collect();
        let narrowest = least_width(&convex_hull(content));
        assert!(narrowest < least_width(&reach.hull) - 1.99 * w);
        let steepest = 2.0 * (reach.members + reach.turn * w) / narrowest;
        assert!(steepest <= turned_slope(reach.turn));
        assert!(
            reach.slope >= steepest,
            "the slope {:e} against {steepest:e}",
            reach.slope
        );
    }

    /// A square 0.2 across facing z, a line along x through it to 3 either way, and a short line
    /// across the first at `x = 2`, all on `z = 0`: the first line enters the square's hull and
    /// lies on its identity; the second lies only where the first does, outside the members' hull
    /// and beyond the snap's reach of it, and lies on nothing, whichever is offered first. The hull
    /// is the flat members' alone: what attaches does not widen it.
    #[test]
    fn a_line_crossing_only_an_attached_line_outside_the_members_hull_does_not_lie_on_it() {
        let s = 0.1;
        let square = ring(&[[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]]);
        let long = [-3.0, -1.0, 1.0, 3.0].map(|x| [x, 0.0, 0.0]);
        let short = [-0.02, -0.02 / 3.0, 0.02 / 3.0, 0.02].map(|y| [2.0, y, 0.0]);
        for long_first in [true, false] {
            let mut r = registry();
            let face = flat(&mut r, &square);
            let (a, b) = if long_first {
                let a = r.offer_planeless(None, &long, &NONE, &long);
                (a, r.offer_planeless(None, &short, &NONE, &short))
            } else {
                let b = r.offer_planeless(None, &short, &NONE, &short);
                (r.offer_planeless(None, &long, &NONE, &long), b)
            };
            r.settle();
            assert!(r.attached(a) && r.identities(a) == r.identities(face));
            assert!(
                !r.attached(b),
                "long first {long_first}: the short line lies on {:?}",
                r.identities(b)
            );
        }
    }

    /// A square 0.2 across facing z and a line along its edge `y = -0.1`, placed by a translation
    /// (so its tolerance is a step, twice the square's), raised 1.25 steps: its first point, at
    /// the hull's first corner, stands off the square's plane by more than the bound less the
    /// line's tolerance, and within the bound; `z = 0.4` steps holds the square within its
    /// tolerance and the line within its own, so the line lies on the square's identity. The
    /// first-point test that passes every pair on to the attach test ([`Registry::settle`]) takes
    /// the record's bound along every normal for its tolerance there, and lets it through.
    #[test]
    fn a_placed_line_within_its_own_tolerance_of_a_plane_through_a_face_lies_on_it() {
        let s = 0.1;
        let square = ring(&[[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]]);
        let h = 1.25 * GRID;
        let line = [0.0, 1.0, 2.0, 3.0].map(|t| [-s + 2.0 * s * t / 3.0, -s, h]);
        let shift = [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let shape = line.map(|p| [p[0] - 0.5, p[1], p[2]]);
        let mut r = registry();
        let face = flat(&mut r, &square);
        let on = r.offer_planeless(None, &line, &[crate::xf::IDENTITY, shift], &shape);
        r.settle();
        let id = r.identities(face)[0];
        let (plane, reach, offer) = (
            r.plane(id),
            &r.reaches[id as usize - 1],
            &r.offers[on as usize],
        );
        let tolerance = r.tolerance(offer, plane.unrounded.normal);
        assert_eq!(reach.flat(line[0]), reach.hull[0]);
        assert!(
            tolerance > 1.9 * r.tolerance_along(face, plane.unrounded.normal)
                && within(&plane, line[0]) > reach.bound(0.0, offer.widest, 0.0)
                && within(&plane, line[0]) <= reach.bound(tolerance, offer.widest, 0.0),
            "{} steps off, its tolerance {} steps",
            within(&plane, line[0]) / GRID,
            tolerance / GRID
        );
        assert!(r.attached(on) && r.identities(on) == r.identities(face));
    }
}
