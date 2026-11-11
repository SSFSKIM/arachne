//! Each drawn thing's key model (contract §5) and its band (§6): what a fragment's key is at a
//! sample, its arithmetic bound `R` and its input bound `S`; and each registry-offered record's
//! facts the tie policies read (§7).

use std::sync::Arc;

use arachne_core::xf::{Camera, View};

use crate::band::{PlaneHull, departure_plus, kr};
use crate::camera::{FrameMap, Placement, Projection};
use crate::centreline::{Centreline, Joint, joints};
use crate::depth::{
    Planarity, ScreenKey, WorldPlane, departure_from, extent, facing_plane, key_error, key_plane,
    nu, on_a_line, planarity, plane_r_region, points_somewhere, three_corner_plane,
};
use crate::draw::{MeshItem, PathItem, Pivot, PointsItem};
use crate::flatten::Ring;
use crate::shift::{Shift, U, departure_of, depth_share};

/// Why a sample's key is not bounded: `band_unbounded` or `curve_recut` (contract §6), or
/// `pivot_unknown`: a pre-projected record's pivot with no first point to move it by (§2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unable {
    BandUnbounded,
    CurveRecut,
    PivotUnknown,
}

impl Unable {
    pub fn name(self) -> &'static str {
        match self {
            Unable::BandUnbounded => "band_unbounded",
            Unable::CurveRecut => "curve_recut",
            Unable::PivotUnknown => "pivot_unknown",
        }
    }
}

/// A key and its band at one read: `I = [o − S⁻ − 3R, o + S⁺ + 3R]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Band {
    pub o: f64,
    pub r: f64,
    pub s_minus: f64,
    pub s_plus: f64,
}

impl Band {
    pub fn exact(o: f64) -> Band {
        Band {
            o,
            r: 0.0,
            s_minus: 0.0,
            s_plus: 0.0,
        }
    }
    pub fn lo(&self) -> f64 {
        if self.o == f64::NEG_INFINITY {
            return self.o;
        }
        self.o - self.s_minus - 3.0 * self.r
    }
    pub fn hi(&self) -> f64 {
        if self.o == f64::NEG_INFINITY {
            return self.o;
        }
        self.o + self.s_plus + 3.0 * self.r
    }
    /// `β = max(S⁻, S⁺) + 3R`.
    pub fn beta(&self) -> f64 {
        self.s_minus.max(self.s_plus) + 3.0 * self.r
    }
}

/// A world plane's input-bound terms: `(|o| + R) kr(P, δ)`, `δ = τ_P(n_P) + θ_P dist(x, H)`.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldS {
    pub big_d: f64,
    pub delta0: f64,
    pub theta: f64,
    pub hull: PlaneHull,
    pub projection: Projection,
    pub map: FrameMap,
}

/// How a world plane's input bound is formed.
#[derive(Clone, Debug, PartialEq)]
pub enum PlaneS {
    World(Box<WorldS>),
    /// `(|o| + R) ds(Δ_c)`: a facing plane.
    Facing {
        ds: f64,
    },
}

/// A key read as one computation (contract §7, `ExactTie`): two reads of one computation are
/// one value bit for bit in every conforming evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum Computation {
    /// A world plane's key across the pixel plane, read at a point.
    World(ScreenKey, [f64; 2]),
    /// The plane on screen through a triangle's corners, read at a point.
    Screen(ScreenKey, [f64; 2]),
    /// The key `−∞`.
    NegInf,
    /// A facing centre's own key.
    Centre(CentreKey),
    /// A curve's key along its centreline, read at a point.
    Curve(Arc<CurveKey>, [f64; 2]),
}

/// What a curve's key along its centreline is computed from (contract §5, `curve_key@1`): its
/// drawn points and its shown points, the projection and frame map that key and place them, and
/// the body `h` its read holds segments by. Two reads of one such input at one point are one
/// computation; equal keys of other inputs are not.
#[derive(Clone, Debug, PartialEq)]
pub struct CurveKey {
    pub world: Vec<[f64; 3]>,
    pub shown: Vec<[f64; 3]>,
    pub projection: Projection,
    pub map: FrameMap,
    pub h: f64,
}

/// What a facing centre's own key `−factor((centre − c) · R₂)` is computed from (contract §5,
/// `facing_plane@1`): the rotation's depth row, the frame centre, the focal distance, the
/// projection's kind, and the centre. Two keys of one such input are one computation; equal keys
/// of other inputs are not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CentreKey {
    pub depth_row: [f64; 3],
    pub frame_centre: [f64; 3],
    pub d: f64,
    pub exponential: bool,
    pub centre: [f64; 3],
}

impl CentreKey {
    fn of(pr: &Projection, centre: [f64; 3]) -> CentreKey {
        CentreKey {
            depth_row: pr.r[2],
            frame_centre: pr.c,
            d: pr.d,
            exponential: pr.exponential,
            centre,
        }
    }
}

/// A key model.
#[derive(Clone, Debug, PartialEq)]
pub enum Model {
    NegInf,
    /// A world plane keyed on screen: its key, its region's `R` before `ρ` and `ν`, any extra
    /// bound (a `Nearest` face's outline departure), its `S`, and for a sloped layer the rings
    /// its fringe reads beyond (`fringe_key@1`).
    Plane {
        key: ScreenKey,
        r0: f64,
        extra: f64,
        s: PlaneS,
        fringe: Option<Vec<Ring>>,
    },
    /// The plane on screen through a triangle's corners.
    Tri {
        key: ScreenKey,
        bound: f64,
        sigma: f64,
        kmax: f64,
        eps: f64,
    },
    /// Along a curve's centreline, at body `h`, computed from `of`.
    Curve {
        line: Centreline,
        h: f64,
        eps: f64,
        sigma: f64,
        joints: Vec<Joint>,
        map: FrameMap,
        of: Arc<CurveKey>,
    },
    /// One key everywhere (a facing plane that does not key: the centre's own key, computed from
    /// `at`).
    Const {
        k: f64,
        r: f64,
        s: f64,
        at: CentreKey,
    },
    /// No bound can be stated (an infinite screen shift or share).
    Unbounded,
    /// No key can be stated: a pre-projected pivot with no first point to move it by (§2).
    PivotUnknown,
}

/// The pixel `(px, py)`'s 16 sample points.
pub fn samples(px: i64, py: i64) -> [[f64; 2]; 16] {
    std::array::from_fn(|s| crate::cover::sample_point(px, py, s))
}

/// Whether a point lies inside rings by the nonzero rule, half-open in `y`.
pub fn inside(rings: &[Ring], p: [f64; 2]) -> bool {
    let mut w = 0i32;
    for ring in rings {
        for i in 0..ring.len() {
            let a = ring[i];
            let b = ring[(i + 1) % ring.len()];
            if a[1] == b[1] {
                continue;
            }
            let (lo, hi, dir) = if a[1] < b[1] { (a, b, 1) } else { (b, a, -1) };
            if p[1] >= lo[1] && p[1] < hi[1] {
                let x = lo[0] + (p[1] - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1]);
                if x > p[0] {
                    w += dir;
                }
            }
        }
    }
    w != 0
}

/// The point of the rings nearest `p`.
fn nearest_on(rings: &[Ring], p: [f64; 2]) -> [f64; 2] {
    let mut best = (f64::INFINITY, p);
    for ring in rings {
        for i in 0..ring.len() {
            let a = ring[i];
            let b = ring[(i + 1) % ring.len()];
            let ab = [b[0] - a[0], b[1] - a[1]];
            let l2 = ab[0] * ab[0] + ab[1] * ab[1];
            let t = if l2 == 0.0 {
                0.0
            } else {
                (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0)
            };
            let q = [a[0] + ab[0] * t, a[1] + ab[1] * t];
            let d = (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1]);
            if d < best.0 {
                best = (d, q);
            }
        }
    }
    best.1
}

impl Model {
    /// The band at each of the pixel's 16 samples, with one `R` for the whole pixel; `Err` where
    /// a sample's key is not bounded. `dist_hull` reads a row's distance from its record's hull
    /// (a stroke's row on its own plane); own reads are inside it.
    pub fn pixel(&self, px: i64, py: i64, dist_hull: bool) -> [Result<Band, Unable>; 16] {
        self.at_points(samples(px, py), px, py, dist_hull)
    }

    /// What its read at the point `p` computes (contract §7, `ExactTie`), where that is one
    /// computation named by its inputs: a world plane's key at the point it reads (its fringe's
    /// outline point outside the outline), a triangle's plane on screen at `p`, a facing centre's
    /// own key, a curve's centreline read at `p`, or `−∞`.
    pub fn computation_at(&self, p: [f64; 2]) -> Option<Computation> {
        match self {
            Model::NegInf => Some(Computation::NegInf),
            Model::Plane { key, fringe, .. } => Some(Computation::World(
                *key,
                match fringe {
                    Some(rings) if !inside(rings, p) => nearest_on(rings, p),
                    _ => p,
                },
            )),
            Model::Tri { key, .. } => Some(Computation::Screen(*key, p)),
            Model::Const { at, .. } => Some(Computation::Centre(*at)),
            Model::Curve { of, .. } => Some(Computation::Curve(of.clone(), p)),
            _ => None,
        }
    }

    /// The band at one screen point `p` of pixel `(px, py)`, read as at a sample there (the
    /// label rule's reads at a centre or an order point, contract §10).
    pub fn at(&self, p: [f64; 2]) -> Result<Band, Unable> {
        let (px, py) = (p[0].floor() as i64, p[1].floor() as i64);
        self.at_points([p; 16], px, py, false)[0]
    }

    /// The band at 16 points of pixel `(px, py)`, with one `R` over them.
    fn at_points(
        &self,
        pts: [[f64; 2]; 16],
        px: i64,
        py: i64,
        dist_hull: bool,
    ) -> [Result<Band, Unable>; 16] {
        match self {
            Model::NegInf => std::array::from_fn(|_| Ok(Band::exact(f64::NEG_INFINITY))),
            Model::Unbounded => std::array::from_fn(|_| Err(Unable::BandUnbounded)),
            Model::PivotUnknown => std::array::from_fn(|_| Err(Unable::PivotUnknown)),
            Model::Const { k, r, s, .. } => std::array::from_fn(|_| {
                if !(s.is_finite() && r.is_finite()) {
                    return Err(Unable::BandUnbounded);
                }
                Ok(Band {
                    o: *k,
                    r: *r,
                    s_minus: *s,
                    s_plus: *s,
                })
            }),
            Model::Plane {
                key,
                r0,
                extra,
                s,
                fringe,
            } => {
                let at: [[f64; 2]; 16] = std::array::from_fn(|k| match fringe {
                    Some(rings) if !inside(rings, pts[k]) => nearest_on(rings, pts[k]),
                    _ => pts[k],
                });
                let os: [f64; 16] = std::array::from_fn(|k| key.at(at[k][0], at[k][1]));
                let omax = os.iter().map(|o| o.abs()).fold(0.0, f64::max);
                let r = r0 + extra + key.rho(px, py) + nu(omax);
                std::array::from_fn(|k| {
                    let o = os[k];
                    let s = match s {
                        PlaneS::Facing { ds } => (o.abs() + r) * ds,
                        PlaneS::World(w) => {
                            let dist = if dist_hull {
                                w.hull
                                    .distance(unproject_on(&w.projection, &w.map, at[k], o))
                            } else {
                                0.0
                            };
                            let delta = if dist == 0.0 {
                                w.delta0
                            } else {
                                w.delta0 + w.theta * dist
                            };
                            (o.abs() + r) * kr(w.big_d, delta)
                        }
                    };
                    if !(s.is_finite() && r.is_finite()) {
                        return Err(Unable::BandUnbounded);
                    }
                    Ok(Band {
                        o,
                        r,
                        s_minus: s,
                        s_plus: s,
                    })
                })
            }
            Model::Tri {
                key,
                bound,
                sigma,
                kmax,
                eps,
            } => {
                let os: [f64; 16] = std::array::from_fn(|k| key.at(pts[k][0], pts[k][1]));
                let omax = os.iter().map(|o| o.abs()).fold(0.0, f64::max);
                let r = bound + key.rho(px, py) + nu(omax);
                let s = sigma * kmax + libm::hypot(key.b, key.c) * eps;
                std::array::from_fn(|k| {
                    if !(s.is_finite() && r.is_finite()) {
                        return Err(Unable::BandUnbounded);
                    }
                    Ok(Band {
                        o: os[k],
                        r,
                        s_minus: s,
                        s_plus: s,
                    })
                })
            }
            Model::Curve {
                line,
                h,
                eps,
                sigma,
                joints,
                map,
                ..
            } => {
                if !(eps.is_finite() && sigma.is_finite()) {
                    return std::array::from_fn(|_| Err(Unable::BandUnbounded));
                }
                let centre = [px as f64 + 0.5, py as f64 + 0.5];
                let reads: Vec<_> = pts
                    .iter()
                    .chain(std::iter::once(&centre))
                    .map(|p| line.read(*p, *h, [px, py]))
                    .collect();
                let r = reads.iter().map(|rd| rd.r).fold(0.0, f64::max);
                std::array::from_fn(|k| {
                    let o = reads[k].key;
                    if o == f64::NEG_INFINITY {
                        return Ok(Band::exact(o));
                    }
                    let p = pts[k];
                    let d = line.distance(p);
                    let rr = h.max(d + eps) + eps;
                    if line.recut(p, rr, *eps, map, joints) {
                        return Err(Unable::CurveRecut);
                    }
                    let (lo, hi, m) = line.keys_within(p, rr).unwrap_or((o, o, o.abs()));
                    let sm = (o - lo).max(0.0) + sigma * m;
                    let sp = (hi - o).max(0.0) + sigma * m;
                    if !(sm.is_finite() && sp.is_finite() && r.is_finite()) {
                        return Err(Unable::BandUnbounded);
                    }
                    Ok(Band {
                        o,
                        r,
                        s_minus: sm,
                        s_plus: sp,
                    })
                })
            }
        }
    }
}

/// The world point on a plane under a screen read at `p` keyed `k`.
fn unproject_on(pr: &Projection, map: &FrameMap, p: [f64; 2], k: f64) -> [f64; 3] {
    let f = -k;
    let q = [
        (p[0] - map.x0) / (map.xx * f * pr.zoom),
        (p[1] - map.y0) / (map.yy * f * pr.zoom),
        pr.d * (1.0 - 1.0 / f),
    ];
    std::array::from_fn(|i| pr.c[i] + (q[0] * pr.r[0][i] + q[1] * pr.r[1][i]) + q[2] * pr.r[2][i])
}

/// A registry-offered record's facts (contract §5, §7).
#[derive(Clone, Debug, PartialEq)]
pub struct Offered {
    /// Its own plane where it is flat.
    pub flat: Option<WorldPlane>,
    /// Whether a flat record retains its identity (its own plane keys on screen).
    pub retains: bool,
    /// Its world points.
    pub points: Vec<[f64; 3]>,
    /// `τ⁺`, its tolerance along any normal.
    pub tau_plus: f64,
    /// `τ_P(n_P)` along its own plane's normal (flat), for `near_parallel@1`.
    pub tau_own: f64,
    /// The axis planes a record with no plane of its own holds, `(axis, its first point's
    /// coordinate)`.
    pub axes: Vec<(usize, f64)>,
    /// Its snap share: `kr(P, τ_P(n_P))` for a flat record; the largest over its keyed axis planes
    /// of `kr(A, Δ(n_A) + dep_A)` for one on axis planes.
    pub snap: f64,
    /// A path record (else a triangle).
    pub path: bool,
}

/// A path item's per-layer models, its own-plane row and its registry facts.
#[derive(Clone, Debug)]
pub struct PathDepth {
    pub layers: [Model; 3],
    /// The row on its own plane, for a flat path whose plane keys (no fringe).
    pub row: Option<Model>,
    pub offered: Option<Offered>,
    /// `σ` and `ε` of the record (for `lines@1`).
    pub sigma: f64,
    pub eps: f64,
}

fn camera_3d(view: &View) -> bool {
    matches!(view.camera, Camera::ThreeD(_))
}

/// The region a plane's `R` is evaluated over for a path: its shown points' screen box widened
/// by `h + 2` px.
fn path_region(item: &PathItem, h: f64) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in &item.shown {
        let q = item.map.map(p[0], p[1]);
        b = [
            b[0].min(q[0]),
            b[1].min(q[1]),
            b[2].max(q[0]),
            b[3].max(q[1]),
        ];
    }
    [
        b[0] - h - 2.0,
        b[1] - h - 2.0,
        b[2] + h + 2.0,
        b[3] + h + 2.0,
    ]
}

fn facing_region(map: &FrameMap) -> [f64; 4] {
    [
        -map.x0.abs(),
        -map.y0.abs(),
        3.0 * map.x0.abs(),
        3.0 * map.y0.abs(),
    ]
}

/// The plane model of `plane` keyed on screen through `pr`, or `None` where it does not key.
#[allow(clippy::too_many_arguments)]
fn plane_model(
    pr: &Projection,
    map: &FrameMap,
    plane: &WorldPlane,
    q_first: [f64; 3],
    min_cos: f64,
    region: [f64; 4],
    s: PlaneS,
    fringe: Option<Vec<Ring>>,
) -> Option<Model> {
    let key = key_plane(pr, map, plane, q_first, min_cos)?;
    let r0 = plane_r_region(pr, map, plane, region);
    Some(Model::Plane {
        key,
        r0,
        extra: 0.0,
        s,
        fringe: fringe.filter(|_| key.sloped()),
    })
}

fn world_s(
    pr: &Projection,
    map: &FrameMap,
    plane: &WorldPlane,
    points: &[[f64; 3]],
    tau: f64,
) -> PlaneS {
    let r = &pr.r;
    let n = plane.unrounded_n;
    let m2 = (r[2][0] * n[0] + r[2][1] * n[1]) + r[2][2] * n[2];
    let big_d = ((plane.unrounded_e - ((n[0] * pr.c[0] + n[1] * pr.c[1]) + n[2] * pr.c[2]))
        - m2 * pr.d)
        .abs();
    let hull = PlaneHull::of(plane, points);
    // A read outside a hull no wider than `2 τ_P` is unbounded (contract §6, `band_input@2`).
    let theta = if hull.width > 2.0 * tau {
        2.0 * tau / hull.width
    } else {
        f64::INFINITY
    };
    PlaneS::World(Box::new(WorldS {
        big_d,
        delta0: tau,
        theta,
        hull,
        projection: pr.clone(),
        map: *map,
    }))
}

/// `Δ_c = (g/2) |R_2|_1`: how far a centre as carried, on the input grid `g`, may move along the
/// depth row of `pr` (contract §6, on a facing plane, for a label).
fn carried_centre_delta(pr: &Projection, g: f64) -> f64 {
    (g / 2.0) * (pr.r[2][0].abs() + pr.r[2][1].abs() + pr.r[2][2].abs())
}

/// The facing model through a world `centre` (contract §5, `facing_plane@1`): its plane keyed
/// with min_cos 0 where it keys, else the centre's own key everywhere; `S` from `ds(Δ_c)`.
fn facing_model(pr: &Projection, map: &FrameMap, centre: [f64; 3], delta_c: f64) -> Model {
    let q = pr.rotated(centre);
    let ds = depth_share(pr, delta_c, std::iter::once(q[2]));
    if !ds.is_finite() {
        return Model::Unbounded;
    }
    if pr.keyed_planar(q[2])
        && let Some(plane) = facing_plane(pr, centre)
        && let Some(m) = plane_model(
            pr,
            map,
            &plane,
            q,
            0.0,
            facing_region(map),
            PlaneS::Facing { ds },
            None,
        )
    {
        return m;
    }
    let k = pr.key(q[2]);
    let shown = pr.project(centre);
    let r = key_error(pr, shown) + nu(k);
    Model::Const {
        k,
        r,
        s: (k.abs() + r) * ds,
        at: CentreKey::of(pr, centre),
    }
}

/// A path item's depth (contract §5), its layers' rings given for the fringe.
pub fn path_depth(view: &View, item: &PathItem, rings: &[Option<Vec<Ring>>; 3]) -> PathDepth {
    let c = &item.common;
    let g = view.input_grid;
    let map = item.map;
    let shift = Shift::of(c, view, &map, &item.shown);
    let eps = shift.of_points(&item.shown);
    let sigma = shift.sigma;
    let none = |m: Model| PathDepth {
        layers: [m.clone(), m.clone(), m],
        row: None,
        offered: None,
        sigma,
        eps,
    };
    let (true, Some(pr)) = (camera_3d(view), c.projection.as_ref()) else {
        return none(Model::NegInf);
    };
    let rec = c.record;
    let half = |k: usize| item.layers[k].unwrap_or(0.0) * map.xx.abs() / 2.0;
    let widest = half(0).max(half(2));
    let pre = rec.pre_projected.is_some();
    let curve_with = |pr: &Projection, k: usize| Model::Curve {
        line: Centreline::of(&item.shown, pr, &map),
        h: half(k),
        eps,
        sigma,
        joints: joints(
            &item.shown,
            pr,
            &map,
            &Centreline::of(&item.shown, pr, &map),
        ),
        map,
        of: Arc::new(CurveKey {
            world: item.world.clone(),
            shown: item.shown.clone(),
            projection: pr.clone(),
            map,
            h: half(k),
        }),
    };
    match (&c.placement, pre) {
        (Placement::FixedOrientation { centre, .. }, _) => {
            let delta_c = carried_centre_delta(pr, g);
            return none(facing_model(pr, &map, *centre, delta_c));
        }
        (_, true) => {
            let cam = pr.in_camera_axes();
            match c.pivot {
                Some(Pivot::At(pivot)) => {
                    let delta_c = carried_centre_delta(&cam, g);
                    return none(facing_model(&cam, &map, pivot, delta_c));
                }
                Some(Pivot::Unknown) => return none(Model::PivotUnknown),
                None => {}
            }
            let code = rec
                .planes
                .as_ref()
                .and_then(|p| p.get(item.index as usize))
                .copied();
            // Each shown point put back by `unproject`, in the camera's axes (contract §2).
            let put_back: Vec<[f64; 3]> = item.world.iter().map(|p| cam.unproject(*p)).collect();
            let (how, plane) = planarity(&put_back, code, item.face);
            let strokes = [curve_with(&cam, 0), curve_with(&cam, 2)];
            let fill = plane
                .filter(|_| how != Planarity::Curve)
                .and_then(|p| {
                    let tau =
                        departure_of(g, p.unrounded_n, &c.placings) + departure_from(&p, &put_back);
                    plane_model(
                        &cam,
                        &map,
                        &p,
                        cam.rotated(put_back[0]),
                        0.0,
                        path_region(item, widest),
                        world_s(&cam, &map, &p, &put_back, tau),
                        rings[1].clone(),
                    )
                })
                .unwrap_or_else(|| curve_with(&cam, 1));
            let [b, s] = strokes;
            return PathDepth {
                layers: [b, fill, s],
                row: None,
                offered: None,
                sigma,
                eps,
            };
        }
        (Placement::Projected, false) => {}
        _ => return none(Model::NegInf),
    }
    if !eps.is_finite() || !sigma.is_finite() {
        return none(Model::Unbounded);
    }
    let curve = |k: usize| curve_with(pr, k);
    let planar = item
        .world
        .iter()
        .all(|p| pr.keyed_planar(pr.rotated(*p)[2]));
    let code = rec
        .planes
        .as_ref()
        .and_then(|p| p.get(item.index as usize))
        .copied();
    let (how, plane) = planarity(&item.world, code, item.face);
    if !planar {
        return PathDepth {
            layers: [curve(0), curve(1), curve(2)],
            row: None,
            offered: None,
            sigma,
            eps,
        };
    }
    let q0 = pr.rotated(item.world[0]);
    if how == Planarity::Nearest {
        let fill = plane
            .and_then(|p| {
                let tau =
                    departure_of(g, p.unrounded_n, &c.placings) + departure_from(&p, &item.world);
                let mut m = plane_model(
                    pr,
                    &map,
                    &p,
                    q0,
                    0.0,
                    path_region(item, widest),
                    world_s(pr, &map, &p, &item.world, tau),
                    rings[1].clone(),
                )?;
                if let Model::Plane { key, extra, .. } = &mut m {
                    *extra = nearest_departure(pr, &map, item, key);
                }
                Some(m)
            })
            .unwrap_or_else(|| curve(1));
        return PathDepth {
            layers: [curve(0), fill, curve(2)],
            row: None,
            offered: None,
            sigma,
            eps,
        };
    }
    let flat = plane.filter(|_| {
        matches!(
            how,
            Planarity::Newell | Planarity::Widest | Planarity::Axis(_)
        ) && !on_a_line(&item.world, g)
    });
    let tau_plus = departure_plus(g, &c.placings);
    match flat {
        Some(p) => {
            let dep = departure_from(&p, &item.world);
            let tau = departure_of(g, p.unrounded_n, &c.placings) + dep;
            let s = world_s(pr, &map, &p, &item.world, tau);
            let region = path_region(item, widest);
            let fill_model =
                plane_model(pr, &map, &p, q0, 0.0, region, s.clone(), rings[1].clone());
            let row = plane_model(pr, &map, &p, q0, 0.0, region, s, None);
            let big_d = match &row {
                Some(Model::Plane {
                    s: PlaneS::World(w),
                    ..
                }) => w.big_d,
                _ => 0.0,
            };
            PathDepth {
                layers: [curve(0), fill_model.unwrap_or_else(|| curve(1)), curve(2)],
                offered: Some(Offered {
                    flat: Some(p),
                    retains: row.is_some(),
                    points: item.world.clone(),
                    tau_plus: tau_plus + dep,
                    tau_own: tau,
                    axes: Vec::new(),
                    snap: kr(big_d, tau),
                    path: true,
                }),
                row,
                sigma,
                eps,
            }
        }
        None => {
            // No plane of its own: the axis planes it holds (coded, or every point's coordinate
            // exact), keyed where their rows form.
            let mut axes = Vec::new();
            if let Planarity::Axis(k) = how {
                axes.push((k, item.world[0][k]));
            }
            for k in 0..3 {
                if !axes.iter().any(|(a, _)| *a == k)
                    && item.world.iter().all(|p| p[k] == item.world[0][k])
                {
                    axes.push((k, item.world[0][k]));
                }
            }
            // It keeps only the axis identities whose rows form (contract §5).
            axes.retain(|(k, _)| {
                key_plane(pr, &map, &WorldPlane::axis(*k, item.world[0]), q0, 0.0).is_some()
            });
            let snap = axes
                .iter()
                .map(|(k, _)| {
                    let a = WorldPlane::axis(*k, item.world[0]);
                    let dep = departure_from(&a, &item.world);
                    let n = a.unrounded_n;
                    let m2 = (pr.r[2][0] * n[0] + pr.r[2][1] * n[1]) + pr.r[2][2] * n[2];
                    let big_d = ((a.unrounded_e
                        - ((n[0] * pr.c[0] + n[1] * pr.c[1]) + n[2] * pr.c[2]))
                        - m2 * pr.d)
                        .abs();
                    kr(big_d, departure_of(g, n, &c.placings) + dep)
                })
                .fold(0.0, f64::max);
            PathDepth {
                layers: [curve(0), curve(1), curve(2)],
                row: None,
                offered: Some(Offered {
                    flat: None,
                    retains: true,
                    points: item.world.clone(),
                    tau_plus,
                    tau_own: 0.0,
                    axes,
                    snap,
                    path: true,
                }),
                sigma,
                eps,
            }
        }
    }
}

/// A `Nearest` face's outline departure: the largest over its subpaths' points `q` (and each
/// closed subpath's closing point read at its first point's position) of
/// `gap_q + ρ(q) + key_error(q) + 4u |k_q| + u gap_q`, plus `hypot(b, c) · 0.1`.
fn nearest_departure(pr: &Projection, map: &FrameMap, item: &PathItem, key: &ScreenKey) -> f64 {
    let mut worst: f64 = 0.0;
    for (s, e) in crate::flatten::subpaths(&item.shown) {
        let mut pts: Vec<([f64; 2], [f64; 3])> = item.shown[s..e]
            .iter()
            .map(|q| (map.map(q[0], q[1]), *q))
            .collect();
        if crate::flatten::closes(&item.shown, s, e) {
            let first = map.map(item.shown[s][0], item.shown[s][1]);
            pts.push((first, item.shown[e - 1]));
        }
        for (at, q) in pts {
            let k = pr.key(q[2]);
            let gap = (key.at(at[0], at[1]) - k).abs();
            let rho = 4.0 * U * (key.a.abs() + (key.b * at[0]).abs() + (key.c * at[1]).abs());
            worst = worst.max(gap + rho + key_error(pr, q) + 4.0 * U * k.abs() + U * gap);
        }
    }
    worst + libm::hypot(key.b, key.c) * 0.1
}

/// A triangle's depth: its model and, where offered, its registry facts.
pub fn triangle_depth(view: &View, item: &MeshItem, i: usize) -> (Model, Option<Offered>) {
    let c = &item.common;
    let map = item.map;
    let g = view.input_grid;
    let (true, Some(pr)) = (camera_3d(view), c.projection.as_ref()) else {
        return (Model::NegInf, None);
    };
    let shift = Shift::of(
        c,
        view,
        &map,
        &item.triangles.iter().flatten().copied().collect::<Vec<_>>(),
    );
    let shown = item.triangles[i];
    let eps = shown.iter().map(|q| shift.at(*q)).fold(0.0, f64::max);
    match &c.placement {
        Placement::FixedOrientation { centre, .. } => {
            let delta_c = carried_centre_delta(pr, g);
            return (facing_model(pr, &map, *centre, delta_c), None);
        }
        Placement::Projected => {}
        _ => return (Model::NegInf, None),
    }
    if !(eps.is_finite() && shift.sigma.is_finite()) {
        return (Model::Unbounded, None);
    }
    let corners = item.placed[i];
    let xy = shown.map(|q| map.map(q[0], q[1]));
    let keys = shown.map(|q| pr.key(q[2]));
    let errs = shown.map(|q| key_error(pr, q));
    let screen = || {
        let (key, bound) = three_corner_plane(xy, keys, errs);
        Model::Tri {
            key,
            bound,
            sigma: shift.sigma,
            kmax: keys.iter().map(|k| k.abs()).fold(0.0, f64::max),
            eps,
        }
    };
    let planar = corners.iter().all(|p| pr.keyed_planar(pr.rotated(*p)[2]));
    if !planar {
        return (screen(), None);
    }
    let pts = corners.to_vec();
    let tol = 1e-9 * extent(&pts).max(1.0);
    let share = |k: usize| {
        corners
            .iter()
            .all(|p| (p[k] - corners[0][k]).abs() <= tol.max(g))
    };
    let axis = [2, 1, 0].into_iter().find(|k| share(*k));
    let normal = crate::depth::cross(
        [
            corners[1][0] - corners[0][0],
            corners[1][1] - corners[0][1],
            corners[1][2] - corners[0][2],
        ],
        [
            corners[2][0] - corners[0][0],
            corners[2][1] - corners[0][1],
            corners[2][2] - corners[0][2],
        ],
    );
    let perim = crate::depth::perimeter(&pts);
    let plane = match axis {
        Some(k) => Some(WorldPlane::axis(k, corners[0])),
        None if points_somewhere(normal, extent(&pts), g, perim) => {
            WorldPlane::canonical(normal, corners[0])
        }
        None => None,
    };
    let flat = plane.filter(|_| !on_a_line(&pts, g));
    let q0 = pr.rotated(corners[0]);
    let tau_plus = departure_plus(g, &c.placings);
    let model = plane
        .and_then(|p| {
            let dep = departure_from(&p, &pts);
            let tau = departure_of(g, p.unrounded_n, &c.placings) + dep;
            let region = {
                let xs = xy.map(|q| q[0]);
                let ys = xy.map(|q| q[1]);
                [
                    xs.iter().copied().fold(f64::INFINITY, f64::min),
                    ys.iter().copied().fold(f64::INFINITY, f64::min),
                    xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ]
            };
            plane_model(
                pr,
                &map,
                &p,
                q0,
                1e-9,
                region,
                world_s(pr, &map, &p, &pts, tau),
                None,
            )
        })
        .unwrap_or_else(screen);
    let offered = match flat {
        Some(p) => {
            let dep = departure_from(&p, &pts);
            let tau = departure_of(g, p.unrounded_n, &c.placings) + dep;
            let big_d = match &model {
                Model::Plane {
                    s: PlaneS::World(w),
                    ..
                } => w.big_d,
                _ => 0.0,
            };
            Offered {
                flat: Some(p),
                retains: matches!(model, Model::Plane { .. }),
                points: pts,
                tau_plus: tau_plus + dep,
                tau_own: tau,
                axes: Vec::new(),
                snap: kr(big_d, tau),
                path: false,
            }
        }
        None => Offered {
            flat: None,
            retains: true,
            axes: axis.map(|k| vec![(k, corners[0][k])]).unwrap_or_default(),
            points: pts,
            tau_plus,
            tau_own: 0.0,
            snap: 0.0,
            path: false,
        },
    };
    (model, Some(offered))
}

/// A points record's block model (contract §5, `facing_plane@1`): the facing plane through the
/// point's placed position, its `Δ_c = Δ(R_2)` through the placements; a fixed-orientation
/// record's through its centre; a pre-projected record's in the camera's axes, through its pivot
/// or else the point put back by `unproject` (§2).
pub fn point_depth(view: &View, item: &PointsItem, placed: [f64; 3]) -> Model {
    let c = &item.common;
    let g = view.input_grid;
    let (true, Some(pr)) = (camera_3d(view), c.projection.as_ref()) else {
        return Model::NegInf;
    };
    let map = FrameMap::of_paths(view);
    if c.record.pre_projected.is_some() {
        let cam = pr.in_camera_axes();
        return match c.pivot {
            Some(Pivot::At(pivot)) => {
                facing_model(&cam, &map, pivot, carried_centre_delta(&cam, g))
            }
            Some(Pivot::Unknown) => Model::PivotUnknown,
            None => facing_model(
                &cam,
                &map,
                cam.unproject(placed),
                departure_of(g, cam.r[2], &c.placings),
            ),
        };
    }
    match &c.placement {
        Placement::FixedOrientation { centre, .. } => {
            let delta_c = carried_centre_delta(pr, g);
            facing_model(pr, &map, *centre, delta_c)
        }
        Placement::Projected => {
            facing_model(pr, &map, placed, departure_of(g, pr.r[2], &c.placings))
        }
        _ => Model::NegInf,
    }
}
