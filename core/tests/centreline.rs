//! A curve's depth on screen (`depth::Centreline`): a stroke's depth at a point is the depth of the
//! part of its curve that covers the point (the front-most segment whose footprint and body hold
//! it), else of the curve's nearest point; never a key the curve's part there does not take.

use arachne_core::composite::{SAMPLES, sample_point};
use arachne_core::depth::Centreline;

/// A face's outline as its curve keys it, from a frame where its front-most segment within reach of
/// pixel (273, 194) is a short one (0.56 px) whose near end lies about 1 px before the pixel centre.
fn face() -> Centreline {
    let points = [
        ([274.391034, 193.074739], -0.8805768606),
        ([273.497671, 194.518750], -0.8798725826),
        ([272.603679, 195.961377], -0.8791694030),
        ([271.710517, 197.402780], -0.8784676691),
        ([270.901576, 198.182970], -0.8802049508),
        ([271.929616, 196.610001], -0.8806193128),
        ([272.959315, 195.036044], -0.8810339350),
        ([273.988646, 193.461099], -0.8814488175),
        ([274.391034, 193.074739], -0.8805768606),
    ];
    Centreline::new(&points, &[0])
}

/// Half the stroke, in pixels, as the frame drew the face.
const HALF: f64 = 0.1501171863815398;

#[test]
fn a_face_never_takes_a_key_nearer_than_every_point_of_it() {
    // Every key of the face's points lies in -0.88145..=-0.87847 (larger is further). Another leaf
    // over the whole pixel at -0.8820 is nearer than every one of them, so wherever both cover a
    // sub-sample it is in front.
    let nearest_of_face = -0.8814488175;
    let other = -0.8820;
    assert!(other < nearest_of_face);
    let face = face();
    for k in 0..SAMPLES {
        let p = sample_point(273, 194, k);
        let (key, _) = face.key_under(p, &[], HALF);
        assert!(
            key >= nearest_of_face,
            "sub-sample {k}: the face keyed {key}, nearer than every point of it ({nearest_of_face})"
        );
        assert!(
            other < key,
            "sub-sample {k}: the face in front of a leaf nearer than all of it"
        );
    }
}

#[test]
fn a_far_edge_s_body_covering_a_sub_sample_keys_it_over_a_near_edge_that_only_reaches_it() {
    // The face's outline runs out along one edge (points 0 to 3, keys -0.8806 to -0.8785) and back
    // along the other, nearer one (points 4 to 8, keys -0.8814 to -0.8802), the two under a pixel
    // apart on screen. Sub-sample 3 of pixel (273, 194) lies 0.14 px from the far edge's first
    // segment and 0.29 px from the near edge's segment 6; sub-sample 13 lies 0.01 px from the far
    // edge's second segment and 0.18 px from segment 6. Half the stroke is 0.15 px: only the far
    // edge's body covers either, and the near edge only reaches them (within half the stroke and a
    // pixel of antialiasing), so each takes the far edge's key there.
    let face = face();
    let near_edge = -0.8814488175..=-0.8810339350;
    for (k, far_edge) in [
        (3, -0.8805768606..=-0.8798725826),
        (13, -0.8798725826..=-0.8791694030),
    ] {
        let (key, err) = face.key_under(sample_point(273, 194, k), &[], HALF);
        assert!(
            far_edge.contains(&key),
            "sub-sample {k}: keyed {key}, not on the far edge covering it ({far_edge:?})"
        );
        assert!(
            key - err > *near_edge.end(),
            "sub-sample {k}: keyed {key} within {err}, not behind the near edge ({near_edge:?})"
        );
    }
}

#[test]
fn a_near_arc_whose_fringe_grazes_a_point_leaves_it_to_the_far_arc_whose_body_covers_it() {
    // Two arcs of one stroke, 1.4 px apart on screen: the near one (key -1.2) along y = 10, the far
    // one (key -0.9) along y = 11.4. The point (10, 11.2) is 1.2 px from the near arc, inside its
    // reach (half the stroke, 0.5, and a pixel of antialiasing) but outside its body, and 0.2 px
    // from the far arc, inside its body: the far arc is what the stroke paints there.
    let line = Centreline::new(
        &[
            ([0.0, 10.0], -1.2),
            ([20.0, 10.0], -1.2),
            ([20.0, 11.4], -0.9),
            ([0.0, 11.4], -0.9),
        ],
        &[2],
    );
    let (half, reach) = (0.5, 1.5);
    let p = [10.0, 11.2];
    assert_eq!(line.key_under(p, &[], half).0, -0.9);
    // Selected within the reach instead, the near arc's fringe would win it.
    assert_eq!(line.key_under(p, &[], reach).0, -1.2);
}

#[test]
fn where_both_arcs_bodies_cover_a_point_the_near_one_wins_it() {
    // A stroke folded back over itself: both arcs' bodies hold the point, and the near arc is
    // what shows.
    let line = Centreline::new(
        &[
            ([0.0, 10.0], -1.2),
            ([20.0, 10.0], -1.2),
            ([20.0, 10.6], -0.9),
            ([0.0, 10.6], -0.9),
        ],
        &[2],
    );
    assert_eq!(line.key_under([10.0, 10.3], &[], 0.5).0, -1.2);
}

/// A small deterministic generator (no dependency): xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

/// Point `p` against segment `a`-`b`: its projection's raw parameter, that parameter held to the
/// segment, and the squared distance to the segment's nearest point.
fn against(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> (f64, f64, f64) {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let raw = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2;
    let t = raw.clamp(0.0, 1.0);
    let q = [a[0] + dx * t, a[1] + dy * t];
    let (ex, ey) = (q[0] - p[0], q[1] - p[1]);
    (raw, t, ex * ex + ey * ey)
}

#[test]
fn every_sub_sample_s_curve_key_lies_within_the_keys_of_the_part_of_the_curve_under_it() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut covered, mut uncovered) = (0, 0);
    for _ in 0..4000 {
        // A polyline of two to five segments, short or long, folded or not, each point with its
        // own key; a stroke body up to 3 px; a pixel near it.
        let n = 3 + (rng.next() * 4.0) as usize;
        let mut points = vec![(
            [rng.range(10.0, 20.0), rng.range(10.0, 20.0)],
            -rng.range(0.8, 1.2),
        )];
        for _ in 1..n {
            let [x, y] = points.last().unwrap().0;
            let length = if rng.next() < 0.5 {
                rng.range(0.05, 1.0)
            } else {
                rng.range(1.0, 6.0)
            };
            let angle = rng.range(0.0, std::f64::consts::TAU);
            points.push((
                [x + length * libm::cos(angle), y + length * libm::sin(angle)],
                -rng.range(0.8, 1.2),
            ));
        }
        let line = Centreline::new(&points, &[0]);
        let body = rng.range(0.0, 3.0);
        let [x0, y0] = points[0].0;
        let x = (x0 + rng.range(-4.0, 4.0)).floor() as u32;
        let y = (y0 + rng.range(-4.0, 4.0)).floor() as u32;
        for k in 0..SAMPLES {
            let p = sample_point(x, y, k);
            let (key, err) = line.key_under(p, &[], body);
            assert!(
                err.is_finite() && err >= 0.0,
                "{points:?} body {body}, pixel ({x}, {y}) sub-sample {k}: key {key} with no error bound ({err})"
            );
            let segments: Vec<_> = points
                .windows(2)
                .map(|w| (w[0].1, w[1].1, against(w[0].0, w[1].0, p)))
                .collect();
            // The part under p: the segments whose footprint and body hold it, else the nearest.
            let holding: Vec<_> = segments
                .iter()
                .filter(|(_, _, (raw, _, d2))| (0.0..=1.0).contains(raw) && *d2 <= body * body)
                .collect();
            let under: Vec<_> = if holding.is_empty() {
                uncovered += 1;
                let least = segments.iter().map(|s| s.2.2).fold(f64::INFINITY, f64::min);
                segments.iter().filter(|s| s.2.2 == least).collect()
            } else {
                covered += 1;
                holding.clone()
            };
            assert!(
                under
                    .iter()
                    .any(|(ka, kb, _)| (ka.min(*kb)..=ka.max(*kb)).contains(&key)),
                "{points:?} body {body}, pixel ({x}, {y}) sub-sample {k}: key {key} outside the keys of the part under it"
            );
            // Where parts overlap, the front-most of them (to rounding: a key is read as its
            // segment's plane reads it, here as a lerp).
            for (ka, kb, (_, t, _)) in &holding {
                let there = ka + (kb - ka) * t;
                assert!(
                    key <= there + 1e-12,
                    "{points:?} body {body}, pixel ({x}, {y}) sub-sample {k}: key {key} behind a part holding it ({there})"
                );
            }
        }
    }
    assert!(
        covered > 5000 && uncovered > 5000,
        "too few sub-samples on each side of a body: {covered} covered, {uncovered} not"
    );
}

/// A double-double: `hi + lo`, `|lo|` within half an ulp of `hi`, about 106 bits.
#[derive(Clone, Copy)]
struct Dd(f64, f64);

impl Dd {
    fn of(v: f64) -> Dd {
        Dd(v, 0.0)
    }

    fn norm(s: f64, e: f64) -> Dd {
        let hi = s + e;
        Dd(hi, e - (hi - s))
    }

    fn add(self, o: Dd) -> Dd {
        let s = self.0 + o.0;
        let v = s - self.0;
        let e = (self.0 - (s - v)) + (o.0 - v);
        Dd::norm(s, e + self.1 + o.1)
    }

    fn neg(self) -> Dd {
        Dd(-self.0, -self.1)
    }

    /// Halves of `v`'s 53 bits, each of 26 or fewer (Veltkamp), so their products are exact.
    fn split(v: f64) -> (f64, f64) {
        let c = 134_217_729.0 * v;
        let hi = c - (c - v);
        (hi, v - hi)
    }

    fn mul(self, o: Dd) -> Dd {
        // The exact product of the two highs (Dekker), without a fused multiply-add.
        let p = self.0 * o.0;
        let ((ah, al), (bh, bl)) = (Dd::split(self.0), Dd::split(o.0));
        let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
        Dd::norm(p, e + self.0 * o.1 + self.1 * o.0)
    }

    fn div(self, o: Dd) -> Dd {
        let q1 = self.0 / o.0;
        let r = self.add(o.mul(Dd::of(q1)).neg());
        let q2 = r.0 / o.0;
        let r = r.add(o.mul(Dd::of(q2)).neg());
        let q3 = r.0 / o.0;
        Dd::norm(q1, q2).add(Dd::of(q3))
    }
}

#[test]
fn a_straight_wire_s_key_stands_within_its_declared_error_of_the_exact_key_of_its_inputs() {
    // Straight wires on screen, their ends on a 2^-20 grid and their keys on a 2^-40 grid, so the
    // exact key at a sub-sample (its end's key plus the key's change times the sub-sample's
    // projection along the wire, held to the wire) is computed here in double-double from exact
    // products: wherever the wire keys a sub-sample, it is within the error bound it declares.
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let on = |v: f64, bits: u32| {
        let scale = (1u64 << bits) as f64;
        (v * scale).round() / scale
    };
    let mut inside = 0;
    for _ in 0..4000 {
        let a = [
            on(rng.range(10.0, 900.0), 20),
            on(rng.range(10.0, 500.0), 20),
        ];
        let length = if rng.next() < 0.5 {
            rng.range(0.05, 2.0)
        } else {
            rng.range(2.0, 300.0)
        };
        let angle = rng.range(0.0, std::f64::consts::TAU);
        let b = [
            on(a[0] + length * libm::cos(angle), 20),
            on(a[1] + length * libm::sin(angle), 20),
        ];
        let (ka, kb) = (on(-rng.range(0.8, 1.2), 40), on(-rng.range(0.8, 1.2), 40));
        if a == b || ka == kb {
            continue;
        }
        let line = Centreline::new(&[(a, ka), (b, kb)], &[0]);
        let t = rng.next();
        let at = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let (x, y) = (
            (at[0] + rng.range(-1.5, 1.5)).floor() as u32,
            (at[1] + rng.range(-1.5, 1.5)).floor() as u32,
        );
        for k in 0..SAMPLES {
            let p = sample_point(x, y, k);
            let (key, err) = line.key_under(p, &[], 2.0);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let along = Dd::of(p[0] - a[0])
                .mul(Dd::of(dx))
                .add(Dd::of(p[1] - a[1]).mul(Dd::of(dy)));
            let length2 = Dd::of(dx).mul(Dd::of(dx)).add(Dd::of(dy).mul(Dd::of(dy)));
            let exact = if along.0 <= 0.0 {
                Dd::of(ka)
            } else if along.add(length2.neg()).0 >= 0.0 {
                Dd::of(kb)
            } else {
                inside += 1;
                Dd::of(ka).add(Dd::of(kb - ka).mul(along.div(length2)))
            };
            let off = Dd::of(key).add(exact.neg()).0.abs();
            assert!(
                off <= err,
                "{a:?} {ka} to {b:?} {kb}, pixel ({x}, {y}) sub-sample {k}: key {key} stands {off:e} from the exact key, its bound {err:e}"
            );
        }
    }
    assert!(
        inside > 30_000,
        "{inside} sub-samples projecting inside a wire"
    );
}
