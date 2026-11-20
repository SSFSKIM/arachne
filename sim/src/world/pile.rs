//! Two contact-rich scenes, each of which finds nothing: the 200-box pile (a walled tray, and 8
//! layers of 5 x 5 spinning cubes dropped into it, stepped 3,600 ticks: W200's literals), and a
//! smaller pour of balls and boxes. For any pair the step's checks find tunnelled, the evidence
//! of what happened is printed for CI's log: the sweep's class and value, the pair's distance
//! through the step, and what the solver's contact graph held between the two after the step
//! before it, after the step itself (the contacts its every substep solved: one narrow-phase pass
//! per step, since the world takes one continuous-collision substep) and after the next (computed
//! from the step's end).

use std::collections::HashMap;
use std::fmt;

use rapier3d::parry::query::{self, NonlinearRigidMotion};
use rapier3d::prelude::{ColliderHandle, PhysicsWorld, Pose, Rotation, Vector};
use sha2::{Digest, Sha256};

use super::{Engine, Integrator, Options, World, finding};
use crate::spec::{Body, Force, Shape, ShapeKind};

const LAST_TICK: usize = 3600;
const GRAVITY: [f64; 3] = [0.0, 0.0, -9.8];
const BOX_SIDE: f64 = 0.4;
/// Boxes along x, along y, and layers.
const GRID: [usize; 3] = [5, 5, 8];
const SPACING: f64 = 0.6;
const FIRST_LAYER_Z: f64 = 1.0;
const ODD_LAYER_SHIFT: f64 = 0.3;
const SPINS: [[f64; 3]; 5] = [
    [1.5, 0.0, 0.0],
    [0.0, 1.5, 0.0],
    [0.0, 0.0, 1.5],
    [-1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0],
];
/// Every shape's material: the boxes' and the tray's (a fixed body's density is the default, 1).
const DENSITY: f64 = 1.0;
const FRICTION: f64 = 0.5;
const RESTITUTION: f64 = 0.1;
/// The tray's fixed parts, each its size and centre: the floor, then the four walls.
const TRAY: [([f64; 3], [f64; 3]); 5] = [
    ([8.0, 8.0, 0.2], [0.0, 0.0, -0.1]),
    ([0.2, 4.4, 2.5], [-2.1, 0.0, 1.25]),
    ([0.2, 4.4, 2.5], [2.1, 0.0, 1.25]),
    ([4.0, 0.2, 2.5], [0.0, -2.1, 1.25]),
    ([4.0, 0.2, 2.5], [0.0, 2.1, 1.25]),
];
const POSE: usize = 7;
const VELOCITY: usize = 6;
/// Instants at which a finding's pair is measured through its step.
const SAMPLES: usize = 1024;

fn stored(x: f64) -> f32 {
    x as f32
}

/// A body of one shape at its centre, of density 1 and friction 0.5.
fn solid(
    w: &mut World,
    fixed: bool,
    kind: ShapeKind,
    centre: [f64; 3],
    spin: [f64; 3],
    restitution: f64,
) {
    let body = w
        .add_body(Body {
            fixed,
            position: centre.map(stored).to_vec(),
            rotation: vec![0.0, 0.0, 0.0, 1.0],
            velocity: vec![0.0; 3],
            angular_velocity: spin.map(stored).to_vec(),
            mass: 0.0,
            gravity_scale: 1.0,
        })
        .expect("a body");
    w.add_shape(Shape {
        body,
        kind,
        position: vec![0.0; 3],
        rotation: vec![0.0, 0.0, 0.0, 1.0],
        density: stored(DENSITY),
        friction: stored(FRICTION),
        restitution: stored(restitution),
    })
    .expect("a shape");
}

fn cuboid(size: [f64; 3]) -> ShapeKind {
    ShapeKind::Cuboid {
        half_extents: size.map(|x| stored(x / 2.0)).to_vec(),
    }
}

/// A 3D world under the pile's gravity.
fn falling() -> World {
    let mut w =
        World::with_options(3, Integrator::Solver, false, Options::default()).expect("a world");
    w.add_force(Force::Gravity {
        acceleration: GRAVITY.map(stored).to_vec(),
    })
    .expect("gravity");
    w
}

/// The pile as the authoring API declares it: gravity, the tray's parts, then the boxes layer by
/// layer, y then x within a layer, each odd layer shifted by half a spacing.
fn pile() -> World {
    let mut w = falling();
    for (size, centre) in TRAY {
        solid(&mut w, true, cuboid(size), centre, [0.0; 3], RESTITUTION);
    }
    let mut boxes = 0;
    for layer in 0..GRID[2] {
        let shift = if layer % 2 == 1 { ODD_LAYER_SHIFT } else { 0.0 };
        let z = FIRST_LAYER_Z + layer as f64 * SPACING;
        for j in 0..GRID[1] {
            for i in 0..GRID[0] {
                let x = (i as f64 - (GRID[0] - 1) as f64 / 2.0) * SPACING + shift;
                let y = (j as f64 - (GRID[1] - 1) as f64 / 2.0) * SPACING + shift;
                let spin = SPINS[boxes % SPINS.len()];
                solid(
                    &mut w,
                    false,
                    cuboid([BOX_SIDE; 3]),
                    [x, y, z],
                    spin,
                    RESTITUTION,
                );
                boxes += 1;
            }
        }
    }
    w
}

/// Another contact-rich scene, not tuned to the rule: 36 balls and boxes of four sizes, in 4
/// layers of 3 x 3 (0.6 apart, each odd layer shifted 0.2), spinning as the pile's boxes do,
/// poured into a narrower tray (inner faces at x, y = +-1.1, walls 1.5 high), for 600 ticks.
fn pour() -> World {
    let mut w = falling();
    let tray: [([f64; 3], [f64; 3]); 5] = [
        ([4.0, 4.0, 0.2], [0.0, 0.0, -0.1]),
        ([0.2, 2.4, 1.5], [-1.2, 0.0, 0.75]),
        ([0.2, 2.4, 1.5], [1.2, 0.0, 0.75]),
        ([2.2, 0.2, 1.5], [0.0, -1.2, 0.75]),
        ([2.2, 0.2, 1.5], [0.0, 1.2, 0.75]),
    ];
    for (size, centre) in tray {
        solid(&mut w, true, cuboid(size), centre, [0.0; 3], RESTITUTION);
    }
    let kinds = [
        (ShapeKind::Ball { radius: 0.12 }, 0.3),
        (cuboid([0.3, 0.2, 0.24]), RESTITUTION),
        (ShapeKind::Ball { radius: 0.08 }, 0.3),
        (cuboid([0.2, 0.2, 0.12]), RESTITUTION),
    ];
    let mut n = 0;
    for layer in 0..4 {
        let shift = if layer % 2 == 1 { 0.2 } else { 0.0 };
        let z = 0.5 + f64::from(layer) * 0.5;
        for j in 0..3 {
            for i in 0..3 {
                let (x, y) = (
                    f64::from(i - 1) * 0.6 + shift,
                    f64::from(j - 1) * 0.6 + shift,
                );
                let (kind, restitution) = kinds[n % kinds.len()].clone();
                solid(
                    &mut w,
                    false,
                    kind,
                    [x, y, z],
                    SPINS[n % SPINS.len()],
                    restitution,
                );
                n += 1;
            }
        }
    }
    w
}

fn physics(w: &World) -> (&PhysicsWorld, &[(ColliderHandle, u32)]) {
    match &w.engine {
        Engine::Three(s) => s.physics(),
        _ => unreachable!("the piles are 3D solver's worlds"),
    }
}

/// A body's name: its number, and whether it is fixed.
fn name(w: &World, body: u32) -> String {
    let kind = if w.bodies[body as usize] {
        "fixed"
    } else {
        "moving"
    };
    format!("{kind} {body}")
}

/// What the solver's contact graph held between two shapes after a step: the pair's manifolds,
/// their points, the contacts handed to the solver (an active contact), and the closest of each.
#[derive(Clone, Copy)]
struct Held {
    manifolds: usize,
    points: usize,
    solver: usize,
    closest_point: f32,
    closest_solver: f32,
}

impl Held {
    fn active(held: Option<Held>) -> bool {
        held.is_some_and(|h| h.solver > 0)
    }
}

/// A pair's entry, or `none` when the graph holds no pair of the two (their boxes apart).
struct Entry(Option<Held>);

impl fmt::Display for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => f.write_str("none"),
            Some(h) => write!(
                f,
                "{} manifolds {} points (closest {:.6}) {} solver contacts (closest {:.6})",
                h.manifolds, h.points, h.closest_point, h.solver, h.closest_solver
            ),
        }
    }
}

/// Every pair of bodies the solver's contact graph holds now.
fn held(w: &World) -> HashMap<(u32, u32), Held> {
    let (physics, shapes) = physics(w);
    let body: HashMap<ColliderHandle, u32> = shapes.iter().copied().collect();
    let mut found = HashMap::new();
    for pair in physics.narrow_phase.contact_pairs() {
        let (a, b) = (body[&pair.collider1], body[&pair.collider2]);
        let mut h = Held {
            manifolds: 0,
            points: 0,
            solver: 0,
            closest_point: f32::INFINITY,
            closest_solver: f32::INFINITY,
        };
        for m in pair.manifolds() {
            h.manifolds += 1;
            h.points += m.points.len();
            for p in &m.points {
                h.closest_point = h.closest_point.min(p.dist);
            }
        }
        for m in pair.solver_manifolds() {
            h.solver += m.data.solver_contacts.len();
            for c in &m.data.solver_contacts {
                h.closest_solver = h.closest_solver.min(c.dist);
            }
        }
        assert_eq!(h.solver > 0, pair.has_any_active_contact());
        found.insert((a.min(b), a.max(b)), h);
    }
    found
}

fn body_pose(row: &[f32], b: u32) -> Pose {
    let p = &row[b as usize * POSE..(b as usize + 1) * POSE];
    Pose::from_parts(
        Vector::new(p[0], p[1], p[2]),
        Rotation::from_xyzw(p[3], p[4], p[5], p[6]),
    )
}

/// The turn from `from` to `to` as a rotation vector, as the step's checks take it.
fn turn(from: Rotation, to: Rotation) -> Vector {
    let q = to * from.inverse();
    let sign = if q.w < 0.0 { -1.0 } else { 1.0 };
    let axis = Vector::new(q.x, q.y, q.z) * sign;
    let half = axis.length();
    if half == 0.0 {
        return Vector::ZERO;
    }
    axis * (2.0 * libm::atan2f(half, q.w * sign) / half)
}

/// The pair's distance through the step, its two shapes swept as the step's checks sweep them
/// (each body's centre of mass along a line, its turn at a constant rate): at its start and end,
/// the closest (zero when they intersect) and the deepest penetration (a contact's negative
/// distance), each with its instant; and the rate the checks bound the closing by.
struct Profile {
    start: f32,
    end: f32,
    closest: (f32, f32),
    deepest: (f32, f32),
    rate: f32,
}

fn profile(w: &World, pair: (u32, u32), row: usize) -> Profile {
    let (physics, shapes) = physics(w);
    let n = w.bodies();
    let (before, after) = (w.pose(row - 1, row), w.pose(row, row + 1));
    let mut moving = Vec::new();
    for b in [pair.0, pair.1] {
        let (handle, _) = shapes.iter().find(|&&(_, on)| on == b).expect("a shape");
        let collider = &physics.colliders[*handle];
        let local = *collider
            .position_wrt_parent()
            .expect("a shape is on a body");
        let com = physics.bodies[collider.parent().expect("a body")].local_center_of_mass();
        let (from, to) = (body_pose(before, b), body_pose(after, b));
        let start = from * local;
        let spin = turn(from.rotation, to.rotation);
        let linvel = to.transform_point(com) - from.transform_point(com);
        let sphere = collider.shape().compute_local_bounding_sphere();
        let reach = (local.transform_point(sphere.center) - com).length() + sphere.radius;
        moving.push((
            NonlinearRigidMotion::new(start, local.inverse_transform_point(com), linvel, spin),
            collider.shape(),
            linvel,
            spin.length() * reach,
        ));
    }
    assert_eq!(before.len(), n * POSE);
    let ((m1, s1, v1, r1), (m2, s2, v2, r2)) = (&moving[0], &moving[1]);
    let at = |t: f32| -> (f32, f32) {
        let (p1, p2) = (m1.position_at_time(t), m2.position_at_time(t));
        let distance = query::distance(&p1, *s1, &p2, *s2).map_or(f32::NAN, |d| d.distance);
        let depth = match query::contact(&p1, *s1, &p2, *s2, 0.0) {
            Ok(Some(c)) => c.dist,
            _ => f32::INFINITY,
        };
        (distance, depth)
    };
    let mut p = Profile {
        start: at(0.0).0,
        end: at(1.0).0,
        closest: (f32::INFINITY, 0.0),
        deepest: (f32::INFINITY, 0.0),
        rate: (*v1 - *v2).length() + r1 + r2,
    };
    for i in 0..=SAMPLES {
        let t = i as f32 / SAMPLES as f32;
        let (distance, depth) = at(t);
        if distance < p.closest.0 {
            p.closest = (distance, t);
        }
        if depth < p.deepest.0 {
            p.deepest = (depth, t);
        }
    }
    p
}

fn class(value: f32) -> &'static str {
    if value.is_nan() {
        "indeterminate"
    } else if value < 0.0 {
        "overlap"
    } else {
        "near"
    }
}

fn speed(row: &[f32], b: u32) -> (f32, f32) {
    let v = &row[b as usize * VELOCITY..(b as usize + 1) * VELOCITY];
    let length = |x: &[f32]| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    (length(&v[..3]), length(&v[3..]))
}

/// One tunnelling finding and its evidence.
struct Evidence {
    row: usize,
    pair: (u32, u32),
    value: f32,
    profile: Profile,
    /// After the step before, the step itself, and the next step.
    held: [Option<Held>; 3],
    speeds: [(f32, f32); 2],
    heights: [f32; 2],
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Advances `w` row by row through `last`, and gathers each tunnelling finding's evidence;
/// prints any other finding.
fn observe(w: &mut World, last: usize) -> Vec<Evidence> {
    w.advance_to(0).expect("the first row");
    let mut before = held(w);
    let mut seen = w.findings().len();
    let (mut pending, mut found): (Vec<Evidence>, Vec<Evidence>) = (Vec::new(), Vec::new());
    for row in 1..=last {
        w.advance_to(row).expect("the rows");
        let now = held(w);
        for mut e in pending.drain(..) {
            e.held[2] = now.get(&e.pair).copied();
            found.push(e);
        }
        let (findings, values) = (w.findings(), w.finding_values());
        for (&[at, kind, a, b], &value) in findings.iter().zip(values).skip(seen) {
            if kind != finding::TUNNELLED {
                eprintln!("pile: row {at}: finding kind {kind} of {a}, {b}, value {value:e}");
                continue;
            }
            assert_eq!(at as usize, row);
            let pair = (a, b);
            let (last, this) = (w.pose(row, row + 1), w.velocity(row, row + 1));
            pending.push(Evidence {
                row,
                pair,
                value,
                profile: profile(w, pair, row),
                held: [before.get(&pair).copied(), now.get(&pair).copied(), None],
                speeds: [speed(this, a), speed(this, b)],
                heights: [last[a as usize * POSE + 2], last[b as usize * POSE + 2]],
            });
        }
        seen = findings.len();
        before = now;
    }
    found.append(&mut pending);
    found
}

/// Prints each tunnelling finding's evidence and a line of counts, prefixed with `scene`.
fn report(scene: &str, w: &World, found: &[Evidence]) {
    let contacts = w.contacts();
    for e in found {
        let (a, b) = e.pair;
        let p = &e.profile;
        let near: Vec<String> = contacts
            .iter()
            .filter(|c| (c[1], c[2]) == e.pair && c[0].abs_diff(e.row as u32) <= 10)
            .map(|c| format!("{}@{}", if c[3] == 1 { "began" } else { "ended" }, c[0]))
            .collect();
        eprintln!(
            "{scene}-finding: row {} pair ({a}, {b}) [{} | {}] class {} value {:e}\n  \
             distance: start {:.6} end {:.6} closest {:.6} at t={:.4} deepest contact {:.6} at \
             t={:.4} rate {:.6}\n  \
             solver held: before {} | during {} | after (from the end) {}\n  \
             active: before {} during {} after {}; speeds (u/s, rad/s) {:?} {:?}; heights {:.4} \
             {:.4}; contact events within 10 rows {:?}",
            e.row,
            name(w, a),
            name(w, b),
            class(e.value),
            e.value,
            p.start,
            p.end,
            p.closest.0,
            p.closest.1,
            p.deepest.0,
            p.deepest.1,
            p.rate,
            Entry(e.held[0]),
            Entry(e.held[1]),
            Entry(e.held[2]),
            Held::active(e.held[0]),
            Held::active(e.held[1]),
            Held::active(e.held[2]),
            e.speeds[0],
            e.speeds[1],
            e.heights[0],
            e.heights[1],
            near,
        );
    }
    let count = |c: &str| found.iter().filter(|e| class(e.value) == c).count();
    eprintln!(
        "{scene}: {} tunnelling findings: {} overlap, {} near, {} indeterminate; with an active \
         contact before {}, during {}, after {}; contact events {}",
        found.len(),
        count("overlap"),
        count("near"),
        count("indeterminate"),
        found.iter().filter(|e| Held::active(e.held[0])).count(),
        found.iter().filter(|e| Held::active(e.held[1])).count(),
        found.iter().filter(|e| Held::active(e.held[2])).count(),
        contacts.len(),
    );
}

/// The build the pile's trace is pinned on: the shipped wheel's (x86-64 Linux, the pinned
/// toolchain, baseline target features). Elsewhere the bytes differ by design (README.md,
/// "Determinism"), and the pin does not apply.
const PINNED_BUILD: &str =
    "rustc 1.99.1 (58cf8f9c3 2026-10-13); x86_64-unknown-linux-gnu; fxsr,sse,sse2";
/// The pile's trace bytes on that build, as they were before the late-contact rule: the rule
/// reads the motion and must not move it.
const PINNED_TRACE: &str = "e958ce6207ee2a3b94fa5d511ae9b0cf49359124d68b8e3830a6f5d9ed13f6d5";

#[test]
fn the_pile_comes_to_rest_with_nothing_found_and_its_trace_unchanged() {
    let started = std::time::Instant::now();
    let mut w = pile();
    let found = observe(&mut w, LAST_TICK);
    let seconds = started.elapsed().as_secs_f64();
    report("pile", &w, &found);

    // What the last row shows (W200's rest facts), and the run's bytes.
    let n = w.bodies();
    let (last, velocity) = (
        w.pose(LAST_TICK, LAST_TICK + 1),
        w.velocity(LAST_TICK, LAST_TICK + 1),
    );
    let boxes = TRAY.len() as u32..n as u32;
    let max_speed = boxes
        .clone()
        .map(|b| speed(velocity, b).0)
        .fold(0.0, f32::max);
    let top = boxes
        .clone()
        .map(|b| last[b as usize * POSE + 2])
        .fold(f32::MIN, f32::max);
    let in_tray = boxes
        .filter(|&b| {
            let p = &last[b as usize * POSE..];
            p[0].abs() < 2.0 && p[1].abs() < 2.0
        })
        .count();
    let (max_speed, top) = (f64::from(max_speed), f64::from(top));
    let (build, trace) = (
        w.identity().build,
        hex(Sha256::digest(w.trace_bytes()).as_slice()),
    );
    eprintln!(
        "pile: rows {} bodies {n}; at the last row max speed {max_speed:?} top centre z {top:?} \
         boxes in the tray {in_tray}; advanced and read in {seconds:.3} s; build {build:?}; \
         trace sha256 {trace}",
        w.rows(),
    );
    assert_eq!(in_tray, n - TRAY.len());
    assert_eq!(w.findings(), &[] as &[[u32; 4]], "the pile finds nothing");
    if build == PINNED_BUILD {
        assert_eq!(trace, PINNED_TRACE, "the pile's motion moved");
    } else {
        eprintln!("pile: not the pinned build; its trace is not compared");
    }
}

#[test]
fn a_pour_of_balls_and_boxes_finds_nothing() {
    let mut w = pour();
    let found = observe(&mut w, 600);
    report("pour", &w, &found);
    assert_eq!(w.findings(), &[] as &[[u32; 4]], "the pour finds nothing");
}
