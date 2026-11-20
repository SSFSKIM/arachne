//! A world's rows and bytes: the same however its steps are grouped into calls, read or not
//! between them, and after another world in the same process; its declaration holds everything
//! it was given, in order; its trace is the bake content's canonical bytes; and what it cannot
//! compute it refuses by name.

mod common;

use arachne_sim::{Body, Force, Integrator, Joint, JointKind, Shape, ShapeKind, World};
use common::{body, f};

fn rotation(dimension: usize, angle: f32) -> Vec<f32> {
    if dimension == 2 {
        vec![angle]
    } else {
        // A turn about an oblique axis, from literal components.
        vec![0.1, angle, 0.05, 0.98]
    }
}

fn v(dimension: usize, xs: [f32; 3]) -> Vec<f32> {
    xs[..dimension].to_vec()
}

fn shape(w: &mut World, body: u32, kind: ShapeKind, at: [f32; 3], friction: f32, restitution: f32) {
    let d = w.dimension();
    w.add_shape(Shape {
        body,
        kind,
        position: v(d, at),
        rotation: rotation(d, 0.2),
        density: 1.5,
        friction,
        restitution,
    })
    .expect("a shape");
}

fn joint(
    w: &mut World,
    kind: JointKind,
    a: u32,
    b: u32,
    axis: Vec<f32>,
    limits: Option<[f32; 2]>,
) -> u32 {
    let d = w.dimension();
    w.add_joint(Joint {
        kind,
        body1: a,
        body2: b,
        anchor1: v(d, [0.3, -0.2, 0.0]),
        anchor2: v(d, [-0.3, 0.1, 0.0]),
        axis,
        limits,
        contacts: false,
    })
    .expect("a joint")
}

/// A world of every shape kind in contact on a floor, a chain of every joint kind (limits and
/// motors among them), and every force law.
fn everything(dimension: usize) -> World {
    let d = dimension;
    let mut w = World::new(d, Integrator::Solver, false).expect("a world");
    w.add_force(Force::Gravity {
        acceleration: v(d, [0.0, -9.8, 0.0]),
    })
    .expect("gravity");
    let floor = body(&mut w, true, &[0.0, -0.5, 0.0][..d], &[0.0; 3][..d], 0.0);
    shape(
        &mut w,
        floor,
        ShapeKind::Cuboid {
            half_extents: v(d, [8.0, 0.5, 8.0]),
        },
        [0.0; 3],
        0.6,
        0.0,
    );
    let mut bodies = Vec::new();
    let hull: Vec<f32> = if d == 2 {
        vec![-0.4, -0.3, 0.5, -0.2, 0.3, 0.4, -0.2, 0.5]
    } else {
        vec![
            -0.4, -0.3, -0.2, 0.5, -0.2, -0.3, 0.3, 0.4, -0.1, -0.2, 0.5, 0.2, 0.1, -0.35, 0.4,
        ]
    };
    let kinds = [
        ShapeKind::Ball { radius: 0.3 },
        ShapeKind::Cuboid {
            half_extents: v(d, [0.4, 0.25, 0.3]),
        },
        ShapeKind::Capsule {
            half_height: 0.3,
            radius: 0.15,
        },
        ShapeKind::ConvexHull { points: hull },
    ];
    for (i, kind) in kinds.into_iter().enumerate() {
        let x = -3.0 + 1.5 * i as f32;
        let b = w
            .add_body(Body {
                fixed: false,
                position: v(d, [x, 1.0 + 0.4 * i as f32, 0.1]),
                rotation: rotation(d, 0.1 * i as f32),
                velocity: v(d, [0.5, 0.0, -0.2]),
                angular_velocity: if d == 2 {
                    vec![1.0]
                } else {
                    vec![0.2, 1.0, 0.0]
                },
                mass: 0.0,
                gravity_scale: 1.0,
            })
            .expect("a body");
        shape(&mut w, b, kind, [0.0; 3], 0.5, 0.4);
        bodies.push(b);
    }
    // A compound: two shapes on one body.
    let compound = body(&mut w, false, &[3.5, 2.0, 0.0][..d], &[0.0; 3][..d], 0.0);
    shape(
        &mut w,
        compound,
        ShapeKind::Ball { radius: 0.2 },
        [-0.3, 0.0, 0.0],
        0.4,
        0.2,
    );
    shape(
        &mut w,
        compound,
        ShapeKind::Cuboid {
            half_extents: v(d, [0.2, 0.1, 0.15]),
        },
        [0.3, 0.1, 0.0],
        0.4,
        0.2,
    );

    // A chain from a fixed anchor: pin (limited, driven), slide (limited, driven), weld, ball,
    // rope, spring.
    let anchor = body(&mut w, true, &[-4.0, 5.0, 0.0][..d], &[0.0; 3][..d], 0.0);
    let link = |w: &mut World, x: f32| {
        let b = body(w, false, &[x as f64, 5.0, 0.0][..d], &[0.0; 3][..d], 0.5);
        shape(
            w,
            b,
            ShapeKind::Capsule {
                half_height: 0.3,
                radius: 0.08,
            },
            [0.0; 3],
            0.3,
            0.0,
        );
        b
    };
    let z = if d == 2 { vec![] } else { vec![0.0, 0.0, 1.0] };
    let a = link(&mut w, -3.5);
    let pin = joint(
        &mut w,
        JointKind::Pin,
        anchor,
        a,
        z.clone(),
        Some([-0.8, 0.6]),
    );
    let b = link(&mut w, -2.9);
    let slide = joint(
        &mut w,
        JointKind::Slide,
        a,
        b,
        v(d, [1.0, 0.0, 0.0]),
        Some([-0.2, 0.3]),
    );
    let c = link(&mut w, -2.3);
    joint(&mut w, JointKind::Weld, b, c, vec![], None);
    let e = link(&mut w, -1.7);
    if d == 3 {
        joint(&mut w, JointKind::Ball, c, e, vec![], None);
    } else {
        joint(&mut w, JointKind::Pin, c, e, vec![], None);
    }
    let g = link(&mut w, -1.1);
    joint(&mut w, JointKind::Rope { length: 0.9 }, e, g, vec![], None);
    let h = link(&mut w, -0.5);
    joint(
        &mut w,
        JointKind::Spring {
            rest: 0.6,
            stiffness: 30.0,
            damping: 0.8,
        },
        g,
        h,
        vec![],
        None,
    );
    for (joint, position) in [(pin, 0.4), (slide, 0.1)] {
        w.add_force(Force::Motor {
            joint,
            position,
            velocity: 0.5,
            stiffness: 15.0,
            damping: 1.5,
        })
        .expect("a motor");
    }

    // Every force law.
    let one = body(
        &mut w,
        false,
        &[4.0, 6.0, 0.0][..d],
        &[0.0, 1.0, 0.0][..d],
        1.0,
    );
    let two = body(
        &mut w,
        false,
        &[5.0, 6.0, 0.0][..d],
        &[0.0, -1.0, 0.0][..d],
        1.0,
    );
    let three = body(
        &mut w,
        false,
        &[4.5, 7.0, 0.0][..d],
        &[0.5, 0.0, 0.0][..d],
        2.0,
    );
    w.add_force(Force::Attraction {
        strength: 2.0,
        bodies: vec![one, two, three],
    })
    .expect("an attraction");
    w.add_force(Force::Spring {
        body_a: bodies[0],
        anchor_a: v(d, [0.1, 0.05, 0.0]),
        body_b: bodies[1],
        anchor_b: v(d, [-0.1, 0.0, 0.0]),
        stiffness: 20.0,
        damping: 0.6,
        rest: 1.0,
    })
    .expect("a spring");
    w.add_force(Force::LinearDrag {
        body: bodies[2],
        coefficient: 0.3,
    })
    .expect("a drag");
    w.add_force(Force::QuadraticDrag {
        body: bodies[3],
        coefficient: 0.05,
    })
    .expect("a drag");
    w.add_force(Force::Constant {
        body: compound,
        force: v(d, [-1.5, 0.0, 0.2]),
        torque: if d == 2 {
            vec![0.3]
        } else {
            vec![0.0, 0.3, 0.1]
        },
    })
    .expect("a constant force");
    w
}

/// A world of point masses: an orbit, a spring and drag, and a constant force.
fn points(dimension: usize) -> World {
    let d = dimension;
    let mut w = World::new(d, Integrator::PointMasses, false).expect("a world");
    w.add_force(Force::Gravity {
        acceleration: v(d, [0.0, -1.0, 0.0]),
    })
    .expect("gravity");
    let sun = body(&mut w, true, &[0.0; 3][..d], &[0.0; 3][..d], 40.0);
    let planet = body(
        &mut w,
        false,
        &[2.0, 0.0, 0.0][..d],
        &[0.0, 4.4, 0.3][..d],
        1.0,
    );
    let bob = body(&mut w, false, &[-1.5, 0.0, 0.0][..d], &[0.0; 3][..d], 0.5);
    w.add_force(Force::Attraction {
        strength: 1.0,
        bodies: vec![sun, planet],
    })
    .expect("an attraction");
    w.add_force(Force::Spring {
        body_a: sun,
        anchor_a: vec![0.0; d],
        body_b: bob,
        anchor_b: vec![0.0; d],
        stiffness: 10.0,
        damping: 0.1,
        rest: 1.0,
    })
    .expect("a spring");
    w.add_force(Force::QuadraticDrag {
        body: bob,
        coefficient: 0.2,
    })
    .expect("a drag");
    w.add_force(Force::Constant {
        body: planet,
        force: v(d, [0.1, 0.0, 0.0]),
        torque: vec![0.0; if d == 2 { 1 } else { 3 }],
    })
    .expect("a force");
    w
}

const TICKS: usize = 600;
const PIECES: [usize; 6] = [1, 7, 60, 2, 230, 300];

fn same_in_pieces(make: fn(usize) -> World, dimension: usize) {
    let mut whole = make(dimension);
    whole.advance_to(TICKS).expect("the rows");
    let mut pieces = make(dimension);
    let mut row = 0;
    for ticks in PIECES {
        row += ticks;
        pieces.advance_to(row).expect("the rows");
        assert!(pieces.contacts().len() < 100_000);
        // A row already computed is looked up, never computed again.
        pieces.advance_to(row / 2).expect("a lookup");
    }
    let mut second = make(dimension);
    second.advance_to(TICKS).expect("the rows");
    assert_eq!(whole.rows(), TICKS + 1);
    assert_eq!(pieces.rows(), TICKS + 1);
    for other in [&pieces, &second] {
        assert!(whole.declaration_bytes() == other.declaration_bytes());
        assert!(whole.trace_bytes() == other.trace_bytes());
    }
}

#[test]
fn a_2d_world_in_pieces_or_after_another_gives_the_same_bytes() {
    same_in_pieces(everything, 2);
}

#[test]
fn a_3d_world_in_pieces_or_after_another_gives_the_same_bytes() {
    same_in_pieces(everything, 3);
}

#[test]
fn a_world_of_point_masses_in_pieces_gives_the_same_bytes() {
    same_in_pieces(points, 2);
    same_in_pieces(points, 3);
}

#[test]
fn every_body_falls_and_touches_the_floor() {
    for d in [2, 3] {
        let mut w = everything(d);
        w.advance_to(TICKS).expect("the rows");
        let began: Vec<u32> = w
            .contacts()
            .iter()
            .filter(|e| e[3] == 1 && e[1] == 0)
            .map(|e| e[2])
            .collect();
        for body in 1..=4 {
            assert!(
                began.contains(&body),
                "{d}d: body {body} never touched the floor"
            );
        }
        let rows = w.pose(0, w.rows());
        assert!(
            rows.iter().all(|x| x.is_finite()),
            "{d}d: a non-finite pose"
        );
    }
}

#[test]
fn contact_events_are_sorted_and_alternate_per_pair() {
    for d in [2, 3] {
        let mut w = everything(d);
        w.advance_to(TICKS).expect("the rows");
        let events = w.contacts();
        assert!(!events.is_empty());
        assert!(events.windows(2).all(|p| p[0] < p[1]), "{d}d: not sorted");
        let mut pairs: Vec<(u32, u32)> = events.iter().map(|e| (e[1], e[2])).collect();
        pairs.sort_unstable();
        pairs.dedup();
        for (a, b) in pairs {
            assert!(a < b);
            let kinds: Vec<u32> = events
                .iter()
                .filter(|e| (e[1], e[2]) == (a, b))
                .map(|e| e[3])
                .collect();
            for (i, kind) in kinds.iter().enumerate() {
                assert_eq!(
                    *kind,
                    u32::from(i % 2 == 0),
                    "{d}d: pair {a}, {b}: {kinds:?}"
                );
            }
        }
    }
}

#[test]
fn an_open_world_steps_under_the_forces_it_is_given() {
    let make = || {
        let mut w = World::new(2, Integrator::Solver, true).expect("an open world");
        let b = body(&mut w, false, &[0.0, 0.0], &[0.0, 0.0], 1.0);
        let _ = b;
        w
    };
    let push = |w: &mut World, x: f32| {
        for k in 0..30 {
            let row = w.step(&[x, 0.1 * k as f32], &[0.0]).expect("a step");
            assert_eq!(row, k + 1);
        }
    };
    let (mut a, mut b, mut c) = (make(), make(), make());
    push(&mut a, 1.0);
    push(&mut b, 1.0);
    push(&mut c, -1.0);
    assert_eq!(a.rows(), 31);
    assert!(a.declaration_bytes() == b.declaration_bytes());
    assert!(a.trace_bytes() == b.trace_bytes());
    assert!(a.declaration_bytes() != c.declaration_bytes());
    let moved = a.pose(30, 31)[0];
    assert!(moved > 0.0, "the force did not move the body: {moved}");
    // Each step's forces enter the declaration: 1 kind byte, 1 row, 2 + 1 numbers.
    let mut d = make();
    let before = d.declaration_bytes().len();
    d.step(&[1.0, 0.0], &[0.0]).expect("a step");
    // The first step also records the first row, and adds nothing to the declaration for it.
    assert_eq!(d.declaration_bytes().len(), before + 1 + 4 + 3 * 4);

    let err = a
        .advance_to(40)
        .expect_err("an open world does not advance alone");
    assert!(err.0.contains("step()"), "{err}");
    a.advance_to(20).expect("a row it has");
    let mut closed = World::new(2, Integrator::Solver, false).expect("a world");
    body(&mut closed, false, &[0.0, 0.0], &[0.0, 0.0], 1.0);
    let err = closed
        .step(&[0.0, 0.0], &[0.0])
        .expect_err("a closed world");
    assert!(err.0.contains("advance_to"), "{err}");
}

#[test]
fn a_change_between_rows_acts_from_the_next_and_is_declared_with_its_row() {
    let make = || {
        let mut w = World::new(2, Integrator::Solver, false).expect("a world");
        body(&mut w, false, &[0.0, 0.0], &[1.0, 0.0], 1.0);
        w
    };
    let drag = Force::LinearDrag {
        body: 0,
        coefficient: 0.5,
    };
    let mut early = make();
    early.add_force(drag.clone()).expect("a drag");
    early.advance_to(20).expect("the rows");
    let mut late = make();
    late.advance_to(10).expect("the rows");
    late.add_force(drag).expect("a drag");
    late.advance_to(20).expect("the rows");
    // The same rows up to 10, then the late world slows from row 10 on.
    assert!(early.pose(0, 1) == late.pose(0, 1));
    assert!(early.pose(10, 11) != late.pose(10, 11));
    assert!(late.pose(10, 11) == make_advanced(10).pose(10, 11));
    assert!(late.velocity(11, 12)[0] < 1.0);
    // The declarations differ by the drag's row only (10 against 0).
    let (a, b) = (early.declaration_bytes(), late.declaration_bytes());
    assert_eq!(a.len(), b.len());
    let differ: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
    assert_eq!(differ.len(), 1, "{differ:?}");
    assert_eq!((a[differ[0]], b[differ[0]]), (0, 10));

    fn make_advanced(rows: usize) -> World {
        let mut w = World::new(2, Integrator::Solver, false).expect("a world");
        body(&mut w, false, &[0.0, 0.0], &[1.0, 0.0], 1.0);
        w.advance_to(rows).expect("the rows");
        w
    }
}

#[test]
fn a_number_is_declared_as_given() {
    let make = |x: f32| {
        let mut w = World::new(2, Integrator::Solver, false).expect("a world");
        body(&mut w, false, &[0.0, 0.0], &[0.0, 0.0], 1.0);
        w.add_force(Force::Constant {
            body: 0,
            force: vec![x, 0.0],
            torque: vec![0.0],
        })
        .expect("a force");
        w
    };
    // Negative zero is a value of its own, kept.
    assert!(make(0.0).declaration_bytes() != make(-0.0).declaration_bytes());
    let bits = f(0.1).to_bits().to_le_bytes();
    let declared = make(f(0.1));
    let bytes = declared.declaration_bytes();
    assert!(bytes.windows(4).any(|w| w == bits));
}

#[test]
fn the_trace_is_the_bake_content_canonical_bytes() {
    let mut w = everything(2);
    w.advance_to(5).expect("the rows");
    let trace = w.trace_bytes();
    let identity = w.identity();
    let mut expected = b"ARBK".to_vec();
    expected.extend(1u32.to_le_bytes());
    let text = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u32).to_le_bytes());
        out.extend(s.as_bytes());
    };
    for s in [
        &identity.producer,
        &identity.producer_version,
        &identity.solver,
        &identity.solver_version,
        &identity.integrator,
        &identity.build,
        &identity.features.join(","),
    ] {
        text(&mut expected, s);
    }
    expected.extend(6u32.to_le_bytes());
    expected.extend(2u32.to_le_bytes());
    let n = w.bodies() as u32;
    let (pose_unit, velocity_unit) = w.units();
    for (name, width, unit, values) in [
        ("pose", 3u32, pose_unit, w.pose(0, 6)),
        ("velocity", 3u32, velocity_unit, w.velocity(0, 6)),
    ] {
        text(&mut expected, name);
        expected.extend([1u8, 2u8]);
        expected.extend(n.to_le_bytes());
        expected.extend(width.to_le_bytes());
        text(&mut expected, unit);
        for x in values {
            expected.extend(x.to_bits().to_le_bytes());
        }
    }
    expected.extend(1u32.to_le_bytes());
    text(&mut expected, "contact");
    expected.push(4);
    expected.extend((w.contacts().len() as u32).to_le_bytes());
    for e in w.contacts() {
        for x in e {
            expected.extend(x.to_le_bytes());
        }
    }
    assert!(trace == expected);
}

#[test]
fn the_identity_names_the_vendored_sources_and_the_build() {
    let vendored = include_str!("../vendor/VENDORED.md");
    let w = everything(3);
    let identity = w.identity();
    for commit in ["b303bc13", "dbcce409", "78c97db7"] {
        assert!(identity.solver_version.contains(commit));
        assert!(vendored.contains(commit), "VENDORED.md lacks {commit}");
    }
    assert_eq!(identity.solver, "rapier3d");
    assert!(identity.build.starts_with("rustc "), "{}", identity.build);
    assert!(
        identity.build.contains(std::env::consts::ARCH),
        "{}",
        identity.build
    );
    let points = World::new(2, Integrator::PointMasses, false).expect("a world");
    assert!(points.identity().integrator.starts_with("velocity-verlet"));
}

fn refused(result: Result<impl std::fmt::Debug, arachne_sim::Error>, says: &str) {
    let err = result.expect_err("a refusal");
    assert!(err.0.contains(says), "{err} (expected: {says})");
}

#[test]
fn what_a_world_cannot_compute_it_refuses_by_name() {
    let mut w = World::new(2, Integrator::Solver, false).expect("a world");
    let b = body(&mut w, false, &[0.0, 0.0], &[0.0, 0.0], 1.0);
    refused(
        w.add_force(Force::Gravity {
            acceleration: vec![f32::NAN, 0.0],
        }),
        "finite",
    );
    refused(
        w.add_force(Force::Gravity {
            acceleration: vec![0.0, -9.8, 0.0],
        }),
        "takes 2 numbers",
    );
    refused(
        w.add_joint(Joint {
            kind: JointKind::Ball,
            body1: b,
            body2: b,
            anchor1: vec![0.0; 2],
            anchor2: vec![0.0; 2],
            axis: vec![],
            limits: None,
            contacts: false,
        }),
        "two different bodies",
    );
    let other = body(&mut w, false, &[1.0, 0.0], &[0.0, 0.0], 1.0);
    refused(
        w.add_joint(Joint {
            kind: JointKind::Ball,
            body1: b,
            body2: other,
            anchor1: vec![0.0; 2],
            anchor2: vec![0.0; 2],
            axis: vec![],
            limits: None,
            contacts: false,
        }),
        "3D only",
    );
    refused(
        w.add_force(Force::LinearDrag {
            body: 7,
            coefficient: 1.0,
        }),
        "names body 7",
    );
    refused(
        w.add_body(Body {
            fixed: true,
            position: vec![0.0; 2],
            rotation: vec![0.0],
            velocity: vec![1.0, 0.0],
            angular_velocity: vec![0.0],
            mass: 0.0,
            gravity_scale: 1.0,
        }),
        "does not move",
    );
    w.advance_to(1).expect("the rows");
    refused(
        w.add_body(Body {
            fixed: false,
            position: vec![0.0; 2],
            rotation: vec![0.0],
            velocity: vec![0.0; 2],
            angular_velocity: vec![0.0],
            mass: 1.0,
            gravity_scale: 1.0,
        }),
        "before its first row",
    );

    let mut massless = World::new(2, Integrator::Solver, false).expect("a world");
    body(&mut massless, false, &[0.0, 0.0], &[0.0, 0.0], 0.0);
    refused(massless.advance_to(1), "no mass");

    let mut p = World::new(2, Integrator::PointMasses, false).expect("a world");
    let a = body(&mut p, false, &[0.0, 0.0], &[0.0, 0.0], 1.0);
    let c = body(&mut p, false, &[1.0, 0.0], &[0.0, 0.0], 1.0);
    refused(
        p.add_shape(Shape {
            body: a,
            kind: ShapeKind::Ball { radius: 0.1 },
            position: vec![0.0; 2],
            rotation: vec![0.0],
            density: 1.0,
            friction: 0.0,
            restitution: 0.0,
        }),
        "needs the solver",
    );
    refused(
        p.add_force(Force::Spring {
            body_a: a,
            anchor_a: vec![0.1, 0.0],
            body_b: c,
            anchor_b: vec![0.0; 2],
            stiffness: 1.0,
            damping: 0.0,
            rest: 1.0,
        }),
        "anchored at their origins",
    );
    refused(
        p.add_force(Force::Constant {
            body: a,
            force: vec![0.0; 2],
            torque: vec![1.0],
        }),
        "does not turn",
    );
    refused(
        p.add_body(Body {
            fixed: false,
            position: vec![0.0; 2],
            rotation: vec![0.0],
            velocity: vec![0.0; 2],
            angular_velocity: vec![2.0],
            mass: 1.0,
            gravity_scale: 1.0,
        }),
        "does not turn",
    );
    refused(
        World::new(4, Integrator::Solver, false),
        "2 or 3 dimensions",
    );
}
