//! A path's centreline and the key along it (contract §5, `curve_key@1`), with its arithmetic bound
//! `R`, its input bound `S` and the reads `curve_recut` makes `truth_unable` (§6).

use crate::camera::{FrameMap, Projection};
use crate::depth::{key_error, nu};
use crate::flatten::subpaths;
use crate::shift::U;

/// One vertex: its screen position, key and error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub at: [f64; 2],
    pub key: f64,
    pub err: f64,
}

/// One segment between consecutive vertices of a subpath.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub a: Vertex,
    pub b: Vertex,
    pub err: f64,
    /// The cubic it is a piece of (index into [`Centreline::cubics`]), and whether it is the
    /// cubic's first piece after a joint (it then starts from the previous cubic's last vertex).
    pub cubic: usize,
    pub joint: bool,
}

/// A cubic of the centreline on screen: its control points and keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cubic {
    pub x: [[f64; 2]; 4],
    pub k: [f64; 4],
    /// `L`, its control polygon's length on screen.
    pub l: f64,
    /// Its count `n`.
    pub n: usize,
    /// For a cubic after the first of its subpath: the previous cubic's last vertex (the joint).
    pub prev: Option<Vertex>,
    /// Its `t = 1/n` vertex.
    pub first_vertex: Vertex,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Centreline {
    pub segments: Vec<Segment>,
    pub cubics: Vec<Cubic>,
    /// Every key is `−∞`: no finite vertex.
    pub infinite: bool,
    /// The largest screen coordinate magnitude of its vertices.
    pub scale: f64,
}

/// A read: its key, its arithmetic bound before `ν`, and the parameter facts `S` needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Read {
    pub key: f64,
    pub r: f64,
}

fn dist2(p: [f64; 2], q: [f64; 2]) -> f64 {
    let (dx, dy) = (p[0] - q[0], p[1] - q[1]);
    dx * dx + dy * dy
}

impl Centreline {
    /// The centreline of a path's shown points (`z` the rotated depth, or a pre-projected
    /// point's own), keyed `−factor(z)` through `pr`, mapped by `map`.
    pub fn of(shown: &[[f64; 3]], pr: &Projection, map: &FrameMap) -> Centreline {
        let mut segments = Vec::new();
        let mut cubics = Vec::new();
        let mut finite_first: Option<Vertex> = None;
        let mut scale: f64 = 0.0;
        for (s, e) in subpaths(shown) {
            if e - s < 4 {
                continue;
            }
            let p0 = shown[s];
            let mut last = Vertex {
                at: map.map(p0[0], p0[1]),
                key: pr.key(p0[2]),
                err: key_error(pr, p0),
            };
            let mut first_cubic = true;
            let mut i = s;
            while i + 3 < e {
                let c = [shown[i], shown[i + 1], shown[i + 2], shown[i + 3]];
                let x: [[f64; 2]; 4] = c.map(|q| map.map(q[0], q[1]));
                let k: [f64; 4] = c.map(|q| pr.key(q[2]));
                let l = (libm::hypot(x[1][0] - x[0][0], x[1][1] - x[0][1])
                    + libm::hypot(x[2][0] - x[1][0], x[2][1] - x[1][1]))
                    + libm::hypot(x[3][0] - x[2][0], x[3][1] - x[2][1]);
                let n = if l.is_finite() {
                    (libm::ceil(l / 2.0)).clamp(1.0, 64.0) as usize
                } else {
                    1
                };
                let err = c
                    .iter()
                    .map(|q| key_error(pr, *q))
                    .fold(0.0, crate::bound::max)
                    + 4.0 * U * k.iter().map(|v| v.abs()).fold(0.0, crate::bound::max);
                let prev = (!first_cubic).then_some(last);
                let idx = cubics.len();
                let mut fv = last;
                for step in 1..=n {
                    let t = step as f64 / n as f64;
                    let v = 1.0 - t;
                    let w = [v * v * v, 3.0 * v * v * t, 3.0 * v * t * t, t * t * t];
                    let mix =
                        |f: [f64; 4]| ((w[0] * f[0] + w[1] * f[1]) + w[2] * f[2]) + w[3] * f[3];
                    let vert = Vertex {
                        at: [mix(x.map(|p| p[0])), mix(x.map(|p| p[1]))],
                        key: mix(k),
                        err,
                    };
                    if step == 1 {
                        fv = vert;
                    }
                    let seg = Segment {
                        a: last,
                        b: vert,
                        err: crate::bound::max(last.err, vert.err),
                        cubic: idx,
                        joint: step == 1 && !first_cubic,
                    };
                    segments.push(seg);
                    last = vert;
                }
                cubics.push(Cubic {
                    x,
                    k,
                    l,
                    n,
                    prev,
                    first_vertex: fv,
                });
                first_cubic = false;
                i += 4;
            }
        }
        for sg in &segments {
            for v in [sg.a, sg.b] {
                if v.at.iter().all(|c| c.is_finite()) {
                    scale =
                        crate::bound::max(crate::bound::max(scale, v.at[0].abs()), v.at[1].abs());
                    if finite_first.is_none() && v.key.is_finite() {
                        finite_first = Some(v);
                    }
                }
            }
        }
        let fine = |v: &Vertex| v.at.iter().all(|c| c.is_finite()) && v.key.is_finite();
        let kept: Vec<Segment> = segments
            .into_iter()
            .filter(|s| fine(&s.a) && fine(&s.b))
            .collect();
        let infinite = kept.is_empty() && finite_first.is_none();
        let segments = if kept.is_empty() {
            finite_first
                .map(|v| {
                    vec![Segment {
                        a: v,
                        b: v,
                        err: v.err,
                        cubic: usize::MAX,
                        joint: false,
                    }]
                })
                .unwrap_or_default()
        } else {
            kept
        };
        Centreline {
            segments,
            cubics,
            infinite,
            scale,
        }
    }

    /// Each segment's foot parameter `t` and squared distance at `p`.
    fn foot(s: &Segment, p: [f64; 2]) -> (f64, f64) {
        let (a, b) = (s.a.at, s.b.at);
        let ab = [b[0] - a[0], b[1] - a[1]];
        let l2 = ab[0] * ab[0] + ab[1] * ab[1];
        let t = if l2 == 0.0 {
            0.0
        } else {
            ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2
        };
        let tc = t.clamp(0.0, 1.0);
        let q = [a[0] + ab[0] * tc, a[1] + ab[1] * tc];
        (t, dist2(p, q))
    }

    /// A segment's key at `p` and its bound (before `ν`).
    fn key_on(s: &Segment, p: [f64; 2], t: f64, px: [i64; 2]) -> Read {
        let (ka, kb, e) = (s.a.key, s.b.key, s.err);
        let (a, b) = (s.a.at, s.b.at);
        let ab = [b[0] - a[0], b[1] - a[1]];
        let l2 = ab[0] * ab[0] + ab[1] * ab[1];
        if l2 == 0.0 || ka == kb {
            return Read {
                key: ka.min(kb),
                r: e + (ka - kb).abs(),
            };
        }
        if t < 0.0 {
            return Read { key: ka, r: e };
        }
        if t > 1.0 {
            return Read { key: kb, r: e };
        }
        let g = (kb - ka) / l2;
        let (bx, by) = (g * ab[0], g * ab[1]);
        let a0 = (ka - bx * a[0]) - by * a[1];
        let key = (((a0 + bx * p[0]) + by * p[1]) + 0.0).clamp(ka.min(kb), ka.max(kb));
        let tp = libm::sqrt(dist2(p, a) / l2);
        let (fx, fy) = ((px[0] + 1) as f64, (px[1] + 1) as f64);
        let rho = 4.0 * U * (a0.abs() + (bx * fx).abs() + (by * fy).abs());
        let r = e
            + 4.0 * U * (kb - ka).abs() * (1.0 + tp)
            + 3.0 * U * (ka.abs() + (bx * a[0]).abs() + (by * a[1]).abs())
            + rho;
        Read { key, r }
    }

    /// The read of a layer of body `h` (half its stroke width in pixels, 0 for a fill) at `p`,
    /// in pixel `px`: the least key among the holding segments, else the nearest's.
    pub fn read(&self, p: [f64; 2], h: f64, px: [i64; 2]) -> Read {
        if self.infinite || self.segments.is_empty() {
            return Read {
                key: f64::NEG_INFINITY,
                r: 0.0,
            };
        }
        let mut held: Option<Read> = None;
        let mut nearest: Option<(f64, Read)> = None;
        for s in &self.segments {
            let (t, d2) = Self::foot(s, p);
            let read = Self::key_on(s, p, t, px);
            if (0.0..=1.0).contains(&t) && d2 <= h * h {
                held = Some(match held {
                    None => read,
                    Some(h0) => Read {
                        key: h0.key.min(read.key),
                        r: crate::bound::max(h0.r, read.r),
                    },
                });
            }
            if nearest.is_none_or(|(best, _)| d2 < best) {
                nearest = Some((d2, read));
            }
        }
        let read = held.unwrap_or_else(|| nearest.expect("a segment").1);
        Read {
            key: read.key,
            r: read.r + nu(read.key),
        }
    }

    /// `P`'s distance from the centreline.
    pub fn distance(&self, p: [f64; 2]) -> f64 {
        crate::bound::least(
            self.segments.iter().map(|s| Self::foot(s, p).1),
            f64::INFINITY,
        )
        .sqrt()
    }

    /// The keys of every centreline point within `r` of `p`: `(min, max, max |k|)`, or `None`
    /// where none lies within it.
    pub fn keys_within(&self, p: [f64; 2], r: f64) -> Option<(f64, f64, f64)> {
        let mut out: Option<(f64, f64, f64)> = None;
        for s in &self.segments {
            let (a, b) = (s.a.at, s.b.at);
            let ab = [b[0] - a[0], b[1] - a[1]];
            let l2 = ab[0] * ab[0] + ab[1] * ab[1];
            let ap = [p[0] - a[0], p[1] - a[1]];
            // |a + t ab − p|² <= r²: l2 t² − 2 (ap·ab) t + |ap|² − r² <= 0. A zero-length
            // segment within reach is so at every parameter: its keys run end to end.
            let (t0, t1) = if l2 == 0.0 {
                if ap[0] * ap[0] + ap[1] * ap[1] <= r * r {
                    (0.0, 1.0)
                } else {
                    continue;
                }
            } else {
                let bq = ap[0] * ab[0] + ap[1] * ab[1];
                let cq = ap[0] * ap[0] + ap[1] * ap[1] - r * r;
                let disc = bq * bq - l2 * cq;
                if disc < 0.0 {
                    continue;
                }
                let sq = libm::sqrt(disc);
                let (lo, hi) = ((bq - sq) / l2, (bq + sq) / l2);
                let (lo, hi) = (lo.max(0.0), hi.min(1.0));
                if lo > hi {
                    continue;
                }
                (lo, hi)
            };
            for t in [t0, t1] {
                let k = s.a.key + (s.b.key - s.a.key) * t;
                // A key that is not a number holds every key and bounds nothing.
                let (klo, khi) = if k.is_nan() {
                    (f64::NEG_INFINITY, f64::INFINITY)
                } else {
                    (k, k)
                };
                out = Some(match out {
                    None => (klo, khi, crate::bound::max(0.0, k.abs())),
                    Some((lo, hi, m)) => (lo.min(klo), hi.max(khi), crate::bound::max(m, k.abs())),
                });
            }
        }
        out
    }

    /// `room(x) = 128u max(scale, x, 1)`.
    pub fn room(&self, x: f64) -> f64 {
        128.0 * U * crate::bound::max(crate::bound::max(self.scale, x), 1.0)
    }

    /// Whether a read at `p` within `r` is exposed to a rebuilt centreline (`curve_recut`, §6):
    /// a count change of a cubic whose control hull lies within `r`, or a join flip whose
    /// segment, in either version, lies within `r + ε̂ + room(0)`. `xx`, `yy` the map's scales and
    /// `shown_first` each cubic's own first point as shown, in frame units.
    pub fn recut(&self, p: [f64; 2], r: f64, eps: f64, map: &FrameMap, joints: &[Joint]) -> bool {
        let eh = eps * (1.0 + 1.0 / 1_099_511_627_776.0);
        for c in &self.cubics {
            let step = 2.0 * (libm::round(c.l / 2.0)).clamp(1.0, 63.0);
            let exposed = (c.l - step).abs() <= 6.0 * eh * (1.0 + 1e-9) + 2.0 * self.room(c.l);
            if exposed && hull_distance(&c.x, p) <= r {
                return true;
            }
        }
        let _ = map;
        for j in joints {
            if !j.flip_exposed(eh, self.room(0.0)) {
                continue;
            }
            let reach = r + eh + self.room(0.0);
            for (a, b) in [(j.prev_last, j.to), (j.own_first, j.to)] {
                let s = Segment {
                    a: Vertex {
                        at: a,
                        key: 0.0,
                        err: 0.0,
                    },
                    b: Vertex {
                        at: b,
                        key: 0.0,
                        err: 0.0,
                    },
                    err: 0.0,
                    cubic: 0,
                    joint: false,
                };
                if Self::foot(&s, p).1.sqrt() <= reach {
                    return true;
                }
            }
        }
        false
    }
}

/// A joint between a cubic and the one before it in the point list, as `lines@1` and
/// `curve_recut` read it (contract §6, §7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Joint {
    /// The previous cubic's last vertex, and the cubic's own first point, on screen.
    pub prev_last: [f64; 2],
    pub own_first: [f64; 2],
    /// The cubic's `t = 1/n` vertex.
    pub to: [f64; 2],
    /// Per screen axis, the two points' separation and CE's tolerance there, in pixels.
    pub sep: [f64; 2],
    pub tol: [f64; 2],
    /// Whether CE's test joins them.
    pub joined: bool,
    /// The two points' key difference.
    pub keygap: f64,
}

impl Joint {
    /// A join flip is possible: on either axis the separation lies within
    /// `(2 + 10^-5) ε̂ + room(0)` of the axis's tolerance.
    pub fn flip_exposed(&self, eh: f64, room0: f64) -> bool {
        (0..2).any(|k| (self.sep[k] - self.tol[k]).abs() <= (2.0 + 1e-5) * eh + room0)
    }
}

/// Every joint of a path's shown points (frame units, `z` keyed through `pr`): each cubic but the
/// path's first, against the cubic before it in the point list.
pub fn joints(
    shown: &[[f64; 3]],
    pr: &Projection,
    map: &FrameMap,
    line: &Centreline,
) -> Vec<Joint> {
    let mut out = Vec::new();
    let mut ci = 0usize;
    let mut prev_last: Option<([f64; 2], f64, [f64; 3])> = None;
    let mut i = 0;
    while i + 3 < shown.len() {
        let c0 = shown[i];
        let c3 = shown[i + 3];
        if let Some((pl, pk, pshown)) = prev_last {
            let own = map.map(c0[0], c0[1]);
            let tol_x = (1e-6 + 1e-5 * c0[0].abs()) * map.xx.abs();
            let tol_y = (1e-6 + 1e-5 * c0[1].abs()) * map.yy.abs();
            let joined = crate::flatten::ce_close(pshown[0], c0[0])
                && crate::flatten::ce_close(pshown[1], c0[1]);
            let to = line.cubics.get(ci).map_or(own, |c| c.first_vertex.at);
            out.push(Joint {
                prev_last: pl,
                own_first: own,
                to,
                sep: [(pl[0] - own[0]).abs(), (pl[1] - own[1]).abs()],
                tol: [tol_x, tol_y],
                joined,
                keygap: (pk - pr.key(c0[2])).abs(),
            });
        }
        prev_last = Some((map.map(c3[0], c3[1]), pr.key(c3[2]), c3));
        ci += 1;
        i += 4;
    }
    out
}

/// The distance from `p` to the convex hull of four points.
pub fn hull_distance(x: &[[f64; 2]; 4], p: [f64; 2]) -> f64 {
    let mut pts: Vec<[f64; 2]> = x.to_vec();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    let crossz = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f64; 2]> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let iter: Vec<[f64; 2]> = if pass == 0 {
            pts.clone()
        } else {
            pts.iter().rev().copied().collect()
        };
        for q in iter {
            while hull.len() >= start + 2
                && crossz(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0
            {
                hull.pop();
            }
            hull.push(q);
        }
        hull.pop();
    }
    if hull.len() >= 3
        && (0..hull.len()).all(|i| crossz(hull[i], hull[(i + 1) % hull.len()], p) >= 0.0)
    {
        return 0.0;
    }
    let seg = |a: [f64; 2], b: [f64; 2]| {
        let ab = [b[0] - a[0], b[1] - a[1]];
        let l2 = ab[0] * ab[0] + ab[1] * ab[1];
        let t = if l2 == 0.0 {
            0.0
        } else {
            (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0)
        };
        dist2(p, [a[0] + ab[0] * t, a[1] + ab[1] * t]).sqrt()
    };
    if hull.is_empty() {
        return dist2(p, pts[0]).sqrt();
    }
    (0..hull.len())
        .map(|i| seg(hull[i], hull[(i + 1) % hull.len()]))
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_segment(a: [f64; 2], ka: f64, b: [f64; 2], kb: f64) -> Centreline {
        let v = |at, key| Vertex { at, key, err: 0.0 };
        Centreline {
            segments: vec![Segment {
                a: v(a, ka),
                b: v(b, kb),
                err: 0.0,
                cubic: usize::MAX,
                joint: false,
            }],
            cubics: Vec::new(),
            infinite: false,
            scale: 0.0,
        }
    }

    #[test]
    fn a_zero_length_segment_within_reach_gives_both_its_ends_keys() {
        // §6, along a curve: `K` is the keys of every centreline point within `r`, each a moved
        // point's unmoved twin, same segment and same parameter. A zero-length segment's points
        // run from one end's key to the other's: input motion may part its ends on screen.
        let line = one_segment([0.0, 0.0], -2.0, [0.0, 0.0], -1.0);
        assert_eq!(line.keys_within([0.0, 0.0], 1.0), Some((-2.0, -1.0, 2.0)));
        assert_eq!(line.keys_within([2.0, 0.0], 1.0), None);
        // A segment of length keeps the keys of its parameters within reach alone.
        let line = one_segment([0.0, 0.0], -2.0, [4.0, 0.0], -1.0);
        assert_eq!(line.keys_within([0.0, 0.0], 1.0), Some((-2.0, -1.75, 2.0)));
    }

    #[test]
    fn a_distance_with_a_nan_segment_is_unbounded() {
        // The P-NaN invariant: a segment whose foot reads NaN (here `∞ / ∞`, its length squared
        // and its projection both overflowing) states no distance (`∞`), not the least of the
        // segments that stayed finite.
        let mut line = one_segment([0.0, 0.0], -2.0, [1.0, 0.0], -1.0);
        let far = one_segment([1e200, 1e200], -2.0, [-1e200, -1e200], -1.0);
        line.segments.extend(far.segments);
        assert_eq!(line.distance([0.0, 0.5]), f64::INFINITY);
    }
}
