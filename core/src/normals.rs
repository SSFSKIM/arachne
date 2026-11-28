//! Smooth normals across a surface's uv grid: the soft light's normals at the corners of a face
//! group's faces ([`corner_normals`]), decided from the face group's own points.
//!
//! A face of CE's surface is four straight cubics through its corners (`Surface`'s
//! `set_points_as_corners`), corner `k` at point `4k`, its (u, v) indices `(i, j)` naming grid
//! vertices `(i, j)`, `(i + 1, j)`, `(i + 1, j + 1)` and `(i, j + 1)` in corner order. The indices
//! say which corners could be one vertex; whether the grid's edges meet (a closed seam, a pole)
//! they do not say, since an open sheet and a sphere share one index grid. So the points decide:
//!
//! - the grid's first and last columns are one where every pair of their corners lies within
//!   [`WELD`] of the face group's extent (a closed seam, its corners differing in their last bits),
//!   and likewise its first and last rows; an open sheet's edges, far apart, stay apart;
//! - a grid line (a row or a column) whose corners all lie within that of one another is a pole
//!   (a sphere's, a cone's apex): one vertex;
//! - a corner joins its grid vertex only where it lies within that of the vertex's first corner,
//!   so a face moved apart from its neighbours keeps its own normal.
//!
//! A vertex's normal is the sum of the area vectors of the faces meeting there, each once,
//! normalized: smooth across the grid, and at a pole the direction the faces around it face
//! together. A face outside the grid's layout (another number of points, indices outside the grid)
//! takes its own normal at all four corners: flat.

use crate::xf::FaceGroup;

type V3 = [f64; 3];

/// How close, as a share of the face group's extent, corners lie to be one vertex where the
/// grid's edges may meet.
pub const WELD: f64 = 1e-9;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `v` made unit, or the zero vector where it has no direction.
fn unit(v: V3) -> V3 {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm > 0.0 && norm.is_finite() {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    } else {
        [0.0; 3]
    }
}

/// Whether `a` and `b` lie within `tol` of each other in every coordinate.
fn near(a: V3, b: V3, tol: f64) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
}

/// The corners of a face in CE's layout for a uv face: four cubics, corner `k` at point `4k`.
pub fn face_corners(points: &[V3]) -> Option<[V3; 4]> {
    (points.len() == 16).then(|| [points[0], points[4], points[8], points[12]])
}

/// A face's area vector: for four corners, half the cross product of its diagonals; for any other
/// outline, Newell's normal over its points.
fn area_vector(points: &[V3]) -> V3 {
    match face_corners(points) {
        Some(c) => cross(sub(c[2], c[0]), sub(c[3], c[1])).map(|v| v / 2.0),
        None => match points.first() {
            Some(first) => crate::depth::newell(points, *first),
            None => [0.0; 3],
        },
    }
}

/// The widest of the points' extents along the three axes.
fn extent(points: &[V3]) -> f64 {
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

/// Disjoint sets of grid vertices.
struct Sets(Vec<usize>);

impl Sets {
    fn root(&mut self, mut v: usize) -> usize {
        while self.0[v] != v {
            self.0[v] = self.0[self.0[v]];
            v = self.0[v];
        }
        v
    }

    /// Joins the sets of `a` and `b`, the lower root ruling, so the result does not depend on the
    /// order of joins.
    fn join(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.root(a), self.root(b));
        if ra != rb {
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.0[hi] = lo;
        }
    }
}

/// The normals the soft light lights each face of a face group by at its four corners, in face
/// order, four per face (corner `k` at the face's point `4k`), from `world`, the face group's
/// points placed in the world: smooth across the grid, welded at a closed seam and joined at a
/// pole (the module's rules). The zero vector where the faces meeting at a corner cancel and the
/// face has no direction of its own.
pub fn corner_normals(world: &[V3], faces: &FaceGroup) -> Vec<V3> {
    let n = faces.uv.len();
    if n == 0 {
        return Vec::new();
    }
    let runs: Vec<&[V3]> = (0..n).map(|i| &world[faces.run(world.len(), i)]).collect();
    let areas: Vec<V3> = runs.iter().map(|p| area_vector(p)).collect();
    let [nu, nv] = faces.grid.map(|g| g as usize);
    let columns = nu + 1;
    let vertex = |i: usize, j: usize| j * columns + i;
    // Each face in the grid's layout: its corners and the grid vertices they name.
    let gridded: Vec<Option<([V3; 4], [usize; 4])>> = (0..n)
        .map(|f| {
            let [i, j] = faces.uv[f].map(|v| v as usize);
            let corners = face_corners(runs[f])?;
            (i < nu && j < nv).then(|| {
                (
                    corners,
                    [
                        vertex(i, j),
                        vertex(i + 1, j),
                        vertex(i + 1, j + 1),
                        vertex(i, j + 1),
                    ],
                )
            })
        })
        .collect();
    let tol = WELD * extent(world);
    // Each grid vertex's position: its first face's corner there.
    let mut at: Vec<Option<V3>> = vec![None; columns * (nv + 1)];
    for (corners, ids) in gridded.iter().flatten() {
        for k in 0..4 {
            at[ids[k]].get_or_insert(corners[k]);
        }
    }
    let mut sets = Sets((0..at.len()).collect());
    // The first and last columns (rows) weld where every pair of their corners present lies within
    // the tolerance, and at least one pair is present.
    let weld = |sets: &mut Sets, pairs: Vec<(usize, usize)>| {
        let present: Vec<(usize, usize, V3, V3)> = pairs
            .into_iter()
            .filter_map(|(a, b)| Some((a, b, at[a]?, at[b]?)))
            .collect();
        if !present.is_empty() && present.iter().all(|(_, _, p, q)| near(*p, *q, tol)) {
            for (a, b, _, _) in present {
                sets.join(a, b);
            }
        }
    };
    if nu > 0 {
        weld(
            &mut sets,
            (0..=nv).map(|j| (vertex(0, j), vertex(nu, j))).collect(),
        );
    }
    if nv > 0 {
        weld(
            &mut sets,
            (0..=nu).map(|i| (vertex(i, 0), vertex(i, nv))).collect(),
        );
    }
    // A grid line whose corners present all lie within the tolerance of one another is a pole.
    let pole = |sets: &mut Sets, line: Vec<usize>| {
        let present: Vec<(usize, V3)> =
            line.into_iter().filter_map(|v| Some((v, at[v]?))).collect();
        if present.len() < 2 {
            return;
        }
        let spread_ok = (0..3).all(|k| {
            let (lo, hi) = present
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (_, p)| {
                    (lo.min(p[k]), hi.max(p[k]))
                });
            hi - lo <= tol
        });
        if spread_ok {
            for w in present.windows(2) {
                sets.join(w[0].0, w[1].0);
            }
        }
    };
    for j in 0..=nv {
        pole(&mut sets, (0..=nu).map(|i| vertex(i, j)).collect());
    }
    for i in 0..=nu {
        pole(&mut sets, (0..=nv).map(|j| vertex(i, j)).collect());
    }
    // Which set each face corner joins, where it lies on its vertex.
    let joined: Vec<[Option<usize>; 4]> = gridded
        .iter()
        .map(|g| match g {
            Some((corners, ids)) => std::array::from_fn(|k| {
                let first = at[ids[k]].expect("a corner placed its vertex");
                near(corners[k], first, tol).then(|| sets.root(ids[k]))
            }),
            None => [None; 4],
        })
        .collect();
    // Each set's sum of the area vectors of the faces meeting there, each face once.
    let mut sums: Vec<V3> = vec![[0.0; 3]; at.len()];
    for (f, roots) in joined.iter().enumerate() {
        for k in 0..4 {
            if let Some(r) = roots[k]
                && !roots[..k].contains(&Some(r))
            {
                sums[r] = add(sums[r], areas[f]);
            }
        }
    }
    let mut out = Vec::with_capacity(4 * n);
    for f in 0..n {
        let own = unit(areas[f]);
        for root in joined[f] {
            let smooth = root.map(|r| unit(sums[r])).unwrap_or([0.0; 3]);
            out.push(if smooth == [0.0; 3] { own } else { smooth });
        }
    }
    out
}
