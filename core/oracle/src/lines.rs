//! `lines@1`'s share bound `O⁺` (contract §7): an upper bound, from XF, of the share the core
//! reports for a stroke of a record with no plane of its own, term by term along the core's bound
//! on a read's move.

use crate::centreline::{Centreline, Joint, Segment};
use crate::shift::U;

const WIDEN: f64 = 1.0 / 1_099_511_627_776.0;

/// The pixel-independent terms of a centreline's `O⁺`: each cubic's `step⁺`, each segment's
/// `slack`, and `slack⁺`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinesBound {
    eps_hat: f64,
    sigma_hat: f64,
    h: f64,
    slack_plus: f64,
    /// Per segment, `moves⁺_s`.
    moves: Vec<f64>,
}

/// A segment's key at `p`'s foot, clamped to its ends.
fn kappa(s: &Segment, p: [f64; 2]) -> (f64, f64) {
    let (a, b) = (s.a.at, s.b.at);
    let ab = [b[0] - a[0], b[1] - a[1]];
    let l2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if l2 == 0.0 {
        0.0
    } else {
        (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0)
    };
    let q = [a[0] + ab[0] * t, a[1] + ab[1] * t];
    let d = libm::hypot(p[0] - q[0], p[1] - q[1]);
    (d, s.a.key + (s.b.key - s.a.key) * t)
}

/// `step⁺` of a cubic (`lines@1`): control points on screen `x`, keys `k`, its `L`, with `ε̂`, `σ̂`
/// and the centreline's `scale`.
pub fn step_plus(x: &[[f64; 2]; 4], k: &[f64; 4], l: f64, eh: f64, sh: f64, scale: f64) -> f64 {
    let room = |v: f64| 128.0 * U * scale.max(v).max(1.0);
    let step = 2.0 * libm::round(l / 2.0).clamp(1.0, 63.0);
    let exposed = (l - step).abs() <= 6.0 * eh * (1.0 + 1e-9) + 2.0 * room(l);
    if !exposed {
        return 0.0;
    }
    let d3 = [x[3][0] - x[0][0], x[3][1] - x[0][1]];
    let v = libm::hypot(d3[0], d3[1]);
    let dj = |j: usize| [x[j][0] - x[0][0], x[j][1] - x[0][1]];
    let far = libm::hypot(dj(1)[0], dj(1)[1]).max(libm::hypot(dj(2)[0], dj(2)[1]));
    let along: [f64; 2] = std::array::from_fn(|i| {
        if v == 0.0 {
            0.0
        } else {
            let d = dj(i + 1);
            (d[0] * d3[0] + d[1] * d3[1]) / (v * v)
        }
    });
    let r: [f64; 2] = std::array::from_fn(|i| {
        if v == 0.0 {
            0.0
        } else {
            k[i + 1] - k[0] - (k[3] - k[0]) * along[i]
        }
    });
    let low = 0.0f64.min(r[0]).min(r[1]);
    let high = 0.0f64.max(r[0]).max(r[1]);
    let kmax = k.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let kends = (k[3] - k[0]).abs();
    let span = k.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - k.iter().copied().fold(f64::INFINITY, f64::min);
    let e = sh * (1.0 + sh) * kmax;
    let lean = 2.0 * eh;
    let a = span + 2.0 * e;
    let amax = along[0].abs().max(along[1].abs());
    let (b, c) = if v > 2.0 * lean {
        let eps = 128.0 * U * (kmax + kends * (1.0 + amax) * (1.0 + scale / v));
        let b = (1.0 + sh).max(2.0) * (high - low + eps);
        let short = v - lean;
        let du = (lean * (v + far) + lean * lean) / (short * short)
            + far * v * (2.0 * v + lean) * lean / (v * v * (short * short));
        let wide = amax + du;
        let c = high - low + 2.0 * (2.0 * e * (1.0 + wide) + kends * du);
        (b, c)
    } else {
        (0.0, 0.0)
    };
    (1.0 + 16.0 * U) * (a.max(b).max(c) + 72.0 * U * (kmax + span)) * (1.0 + WIDEN)
}

impl LinesBound {
    /// The bound's terms for a stroke of body `h` on `line`, with its record's screen shift `eps`
    /// and depth share `sigma`.
    pub fn of(line: &Centreline, joints: &[Joint], h: f64, eps: f64, sigma: f64) -> LinesBound {
        let eh = eps * (1.0 + WIDEN);
        let sh = sigma * (1.0 + WIDEN);
        let mut slack_plus: f64 = 0.0;
        for j in joints {
            let gap = libm::hypot(
                j.prev_last[0] - j.own_first[0],
                j.prev_last[1] - j.own_first[1],
            );
            let tol = libm::hypot(j.tol[0], j.tol[1]) * (1.0 + 1e-9);
            if j.joined || gap - tol <= (2.0 + 2e-5) * eh + (gap + tol) / 1_073_741_824.0 {
                slack_plus = slack_plus.max(gap + 2e-5 * eh);
            }
        }
        let steps: Vec<f64> = line
            .cubics
            .iter()
            .map(|c| step_plus(&c.x, &c.k, c.l, eh, sh, line.scale))
            .collect();
        // A cubic that starts a joint: its keygap charges its first segment.
        let keygap = |cubic: usize| -> f64 {
            if cubic == 0 || cubic == usize::MAX {
                return 0.0;
            }
            joints.get(cubic - 1).map_or(0.0, |j| j.keygap)
        };
        let mut seen = std::collections::HashSet::new();
        let moves = line
            .segments
            .iter()
            .map(|s| {
                let spread = (s.a.key - s.b.key).abs();
                let kmax = s.a.key.abs().max(s.b.key.abs());
                let first = s.cubic != usize::MAX && seen.insert(s.cubic);
                let slack = if first {
                    (1.0 + sh) * keygap(s.cubic)
                } else {
                    0.0
                };
                let step = steps.get(s.cubic).copied().unwrap_or(0.0);
                (1.0 + sh) * spread + sh * kmax + slack + step
            })
            .collect();
        LinesBound {
            eps_hat: eh,
            sigma_hat: sh,
            h,
            slack_plus,
            moves,
        }
    }

    /// `O⁺` at pixel `(x, y)`, reading `line` at the pixel's 16 samples and centre.
    pub fn share(&self, line: &Centreline, x: i64, y: i64, samples: &[[f64; 2]; 16]) -> f64 {
        if line.segments.is_empty() {
            return f64::INFINITY;
        }
        let room = line.room((x.max(y) + 1) as f64);
        let reach = self.eps_hat + self.slack_plus + room;
        let centre = [x as f64 + 0.5, y as f64 + 0.5];
        let mut l_minus = f64::INFINITY;
        let mut slide: f64 = 0.0;
        for p in samples.iter().chain(std::iter::once(&centre)) {
            let reads: Vec<(f64, f64)> = line.segments.iter().map(|s| kappa(s, *p)).collect();
            let d_min = reads.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
            let limit = (self.h + reach).max(d_min + 2.0 * reach) + room;
            let (mut lo, mut hi, mut kmax, mut m) =
                (f64::INFINITY, f64::NEG_INFINITY, 0.0f64, 0.0f64);
            for (i, (d, k)) in reads.iter().enumerate() {
                if *d > limit {
                    continue;
                }
                lo = lo.min(*k);
                hi = hi.max(*k);
                kmax = kmax.max(k.abs());
                m = m.max(self.moves[i]);
                l_minus = l_minus.min(k.abs());
            }
            let j = hi - lo;
            let s = ((1.0 + self.sigma_hat) * (j + 8.0 * U * kmax) + m) * (1.0 + 64.0 * U);
            slide = slide.max(s);
        }
        if l_minus == 0.0 || !l_minus.is_finite() {
            return f64::INFINITY;
        }
        (self.sigma_hat + (1.0 + 4.0 * U) * slide / l_minus) * (1.0 + WIDEN)
    }
}
