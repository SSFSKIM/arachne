# arachne-sim

Arachne's simulation crate: rigid-body worlds in 2D and 3D, stepped on a fixed clock of 60 ticks
per second, whose results are the same bytes on every machine of one architecture. It is built on
[rapier](https://rapier.rs) with its `enhanced-determinism` feature, stands apart from the core
(`core/`): its own manifest, lock file, toolchain and clippy rules, and no dependency on
`arachne-core` either way.

A world is given bodies (with shapes: balls, boxes, capsules, convex hulls, and compounds of
these), joints (pin, slide, weld, ball in 3D, rope, spring) and the engine's own force laws
(uniform gravity, attraction by the inverse square, a damped spring between anchors, linear and
quadratic drag, a constant force or torque, a motor on a joint), every number already in the stored
type (`f32`). `advance_to(i)` computes its rows up to `i`, each one step of exactly 1/60 s, in order,
the first time any is asked for; nothing takes a time step. Everything a world was given is kept
as its **declaration bytes**, and what it computed as its **trace bytes**, the canonical bytes of the
bake content its rows make (channels `pose` and `velocity`, event table `contact`), so two runs are
compared byte for byte. An **open** world also feels forces its caller gives at each tick (`step`),
which enter its declaration.

Two integrators, named in a bake's identity: the solver, for contacts, joints and rotation; and
velocity Verlet in 8 substeps per tick, for worlds of point masses (no shape, no joint: springs,
orbits, N bodies, damped oscillators, projectiles), whose energy stays within 1% at the periods a
lesson uses where the solver's, which holds an applied force through the tick, does not. The
solver's semi-implicit Euler also drifts a projectile's energy at first order: under uniform
gravity a body falls `g h t / 2` too far by time `t`, `h` its substep, about `g^2 h t / v0^2` of a
launch's energy (1.2% over a one-second flight at 4 substeps, 12% over 10 s), and velocity
Verlet is exact there but for rounding.

What the numbers mean where the solver leaves a choice:

- **Substeps.** The solver steps a world in 4 substeps per tick, and the bodies connected to a
  joint (through joints and contacts, fixed bodies not connecting) in 128: each dynamic body a
  joint joins asks for them, and the solver takes, per connected component, the most its bodies
  ask. Its pass that removes its position corrections' velocity takes energy from a swinging
  joint in proportion to the substep; at 4 a swing of 2 rad loses 15% of its energy in 10 s, at
  128 under 1%, and a component with no joint keeps the cost of 4.
- **A joint's frame** is one frame in the world as its bodies are posed when it is declared, at
  the first anchor, its x axis along the joint's axis (given in the first body's frame, at any
  length but zero), seen from each body. A pin's angle and a slide's offset (what limits and a
  motor act on) are zero as declared: a slide's offset is its anchors' separation along the axis
  less the separation they have when it is declared.
- **A motor** takes force coefficients: its stiffness is a force (a torque, for a pin) per unit
  of offset and its damping one per unit of rate, whatever the masses it drives (the solver's
  force-based model, in the declaration's settings; its default scales them by mass).
- **A spring force** acts along the line between its anchors, `k (|d| - rest) + c (dv . n)`, `d`
  the second anchor's offset from the first, `n` its direction and `dv` their relative velocity;
  at a rest length of 0 its elastic part is `k d`, which needs no direction. For a tick at which
  the anchors coincide, where the line is undefined, it is zero.
- **A contact** between two bodies begins at the first row, row 0 included, at which a shape of
  each is at most the allowed linear error apart (touching or overlapping: 0.005 units, in the
  declaration's settings), and ends at the first later row at which none is. It is found at each
  row by the solver's geometric query, not read from its contact pairs, which hold pairs within
  the prediction distance (0.02) and points computed before the step's motion. Two fixed bodies,
  and two bodies a joint keeps from colliding, have none. The query looks at every pair of shapes
  whose boxes overlap, so its cost grows with the square of the shapes.

| Path | What it is |
| --- | --- |
| `Cargo.toml`, `Cargo.lock` | the solver pinned by exact version, its features listed one by one, patched to the vendored sources |
| `vendor/` | the solver's three corrected crates, as upstream wrote them (`vendor/VENDORED.md`) |
| `rust-toolchain.toml` | the core's toolchain, 1.99.1 |
| `clippy.toml` | the core's rules: no fused multiply-add, no platform transcendental function, in this crate's code |
| `src/` | the world (`world.rs`), what it is given (`spec.rs`), its two encodings (`encode.rs`), the solver in each dimension (`solver.rs`, compiled as `two` and `three`), the point-mass integrator (`points.rs`) and the Python binding (`python.rs`) |
| `python/arachne_sim/`, `pyproject.toml` | the Python package, the engine's `sim` extra; its `World` hands rows out as read-only arrays |
| `examples/probe/` | the determinism probe (below) |
| `tests/` | the closed-form cases (`closed_forms.rs`), a world's bytes and refusals (`world.rs`), the probe's worlds in pieces (`stepping.rs`), the binding's tests (`python/`) |
| `ci/` | the gates' scripts and the checks they run |

The gates (`.github/ci-kit.yml`) run when `sim/` changes: `sim-lint` (format, clippy, the checks'
own tests, the dependency tree and the vendored sources' origin, the census of run-time CPU
detection, the probe's import list), `sim-test` (`cargo test --release --locked`; each closed-form
case prints its closed form, tolerance and measured value), `sim-identity` (the probe built for
`x86-64` and for `x86-64-v3`, run twice, against the output committed in `ci/probe-x86_64.txt`),
`sim-binding` (the wheel; its extension's import list and links; the binding's tests under Python
3.12 and 3.14) and `sim-macos`. The workflow `sim-probe.yml` runs the probe on Linux x86-64, Windows
x86-64 and macOS arm64 and compares the outputs; `sim-wheels.yml` builds the wheels.

## What a world finds

Beside its rows, never in the trace, a world records what it found (`World::findings`, each
with a value, `World::finding_values`): a body whose state went non-finite; a body whose speed or
angular speed rose above the world's limit, as its row shows it or as the solver's cap scaled it
down in the step (the vendored solver flags that, `vendor/VENDORED.md`; the value is then a lower
bound, the cap); and two bodies that may have passed through each other in a step. The last is
decided by a bound that does not miss a crossing and by conservative advancement through the
step, which reports two shapes seen penetrating each other, two that came within the allowed
linear error (0.005 units) of each other, and two it could not show apart (a number that is not
finite, a query that cannot answer, a separation it cannot certify). It advances only by a
certified separation: along the measured closest direction, from the shapes' support maps, less
a rounding of 5e-4 units, which covers single-precision poses and dot products while the shapes
and their motion stay within about 1000 units of the origin, where a number is exact to about
6e-5 units; beyond that the sweep claims nothing.

The solver prepares a contact only within its prediction distance (0.02 units) of the poses a step
starts from, so two bodies that close faster than that in a tick meet a tick before it holds them
in contact. Such a pair is not found when it ends the step within the prediction distance, the
direction between the two at the end agrees with the direction at the start, and the deepest the
step's motion takes them into each other, measured at instants over which they close by at most a
quarter of the allowed linear error and bounded between them, is at most that error. Both numbers
are read from the solver's settings. In short: two bodies that pass through each other and end on
swapped sides, that sink into each other deeper than the solver's allowed linear error, or that
come within it and part again without the solver registering a contact stay found; two that end
the step within the solver's prediction distance, on the side they began, never deeper into each
other than the allowed linear error, are a contact the solver registered a tick late, and are not
found.

## Determinism

"The same bytes" rests on three things: the same declaration (the same numbers, inserted in the
same order), the same solver at the same pinned version and commits, and the same compiler. Given
those, the solver promises the same results on every machine. That promise was checked against the
solver's source, not taken from its documentation, and the arithmetic the shipped build can reach
was audited. This section is the record; the checks in `ci/` keep it true.

### The pin

Read on 2026-11-19, when rapier 0.36.1 (2026-10-10) was the newest release.

| Crate | Version | Source | Features |
| --- | --- | --- | --- |
| `rapier2d` | 0.36.1 | `vendor/rapier`, upstream's [dimforge/rapier@b303bc13](https://github.com/dimforge/rapier/commit/b303bc13e7e69abcec31c595f095de71edf2d3d3): the v0.36.1 tag, two commits to `bevy_rapier` only, then the fix | `dim2`, `f32`, `alloc`, `block-solver`, `enhanced-determinism` |
| `rapier3d` | 0.36.1 | the same commit | `dim3`, `f32`, `alloc`, `enhanced-determinism` |
| `parry2d`, `parry3d` | 0.31.1 | `vendor/parry`, upstream's [dimforge/parry@dbcce409](https://github.com/dimforge/parry/commit/dbcce4095c5c65277de1d3363d59c9a75cc02a87): the v0.31.1 tag, one commit to restoring deserialized contact workspaces (built only with `serde-serialize`, which is off), then the fix | as rapier asks: `required-features`, `alloc`, `enhanced-determinism` |
| `glamx` | 0.3.1 | `vendor/glamx`, upstream's [dimforge/glamx@78c97db7](https://github.com/dimforge/glamx/commit/78c97db71c2724b096cf751d2024c405c65d8612): the v0.3.1 tag, then the fix | as parry asks |
| `simba` | 0.10.2 | crates.io | `libm_force` (from `enhanced-determinism`) |
| `nalgebra` | 0.35.0 | crates.io | `macros`, no `std` |
| `libm` | 0.2.16 | crates.io | the pure-Rust maths every transcendental function goes to |

Single precision, no `std`, no `parallel`, no `simd8`. `block-solver` is each dimension's upstream
default (on in 2D, off in 3D), named rather than left to the default.

The three commits are upstream's own fix for the calls described below, open as pull requests on
the day of the pin: rapier pull request 1074, parry pull request 468 and glamx pull request 10
(rapier issue 1054). Their sources are vendored under `vendor/`, byte for byte, so the pin does not
depend on a branch staying reachable (`vendor/VENDORED.md`). The fix commits' source changes
are the call sites in the table of paths, each rewritten from a method call (`x.sin()`) to the
trait's function (`<Real as ComplexField>::sin(x)`), plus tests and changelog lines; the other
commits between each release tag and its pin are named in the table above. The pin moves to the
first release that carries the fixes, and the vendored copy is deleted.

### Why the calls needed a correction

`enhanced-determinism` turns on simba's `libm_force`, which routes the transcendental functions of
simba's `ComplexField` and `RealField` traits to the pure-Rust `libm`. A method call on a concrete
float, such as `center.sin_cos()` on an `f32`, does not go through the trait: it resolves to the
standard library's own method, which calls the platform's maths library, whenever the standard
library is in the crate graph. Rapier and parry import the trait for these calls only under
`#[cfg(not(feature = "std"))]`, and an inherent method wins over a trait method in any case.

Building the solver without `std` does not help: `enhanced-determinism` enables `indexmap` with its
default features, so the standard library is in the graph anyway (`ci/tree_check.py deps` prints
`built with std: indexmap`). Measured: the build without `std` imported `acosf asinf atan2f expf
log2f powf sincosf sinf` on Linux, as the build with it did, and gave the same bits.

### Properties

| Id | Property | What was read at the pin | Verdict |
| --- | --- | --- | --- |
| P1 | The feature switches the solver's trait maths to machine-independent implementations | `rapier{2,3}d`: `enhanced-determinism = ["simba/libm_force", "parry{2,3}d/enhanced-determinism"]`; `parry{2,3}d`: `enhanced-determinism = ["simba/libm_force", "indexmap", "glamx/libm", "glamx/scalar-math"]`; `glamx`: `libm = ["glam/libm", "simba/libm_force", "num-traits/libm"]` | Holds for calls made through the traits; method calls on concrete floats were not covered (P8) |
| P2 | Same declaration, solver and compiler: the same trace on two machines | The determinism page promises it, under "the same initial conditions" and IEEE 754 platforms | Holds once P8 is corrected; the probe's worlds and cases agree on three systems |
| P3 | The feature excludes only the 8-lane vector path | One `compile_error!` each in rapier's and parry's `src/lib.rs`, under `all(feature = "simd8", feature = "enhanced-determinism")` | Holds |
| P4 | Results do not depend on the number of solver threads | The changelog: "bitwise identical for any thread-pool size" with `parallel` | Not relied on: `parallel` is off and `rayon` is not in the tree |
| P5 | Nothing in the built tree picks its arithmetic from the CPU it runs on | `ci/census.py` over what Cargo compiles of the shipped tree (each library target's source directory, wherever it is, and every file a target's root file reaches by `mod` declarations, a build script's modules among them; this crate's own included): one file, `libm`'s detection, used only to choose the processor's fused multiply-add or libm's software one, which IEEE 754 defines to give the same bits. Without `std`, `nalgebra` drops `matrixmultiply`, whose kernels are chosen at run time | Holds (`ci/census.txt`) |
| P6 | The result does not depend on the vector instructions the compiler was allowed | The probe built for `x86-64` and for `x86-64-v3` | Holds: byte-identical (`sim-identity`) |
| P7 | Stepping in one call or in pieces, in a fresh process or a second world of one process, gives the same bytes | The probe run twice; `tests/stepping.rs` | Holds. Not covered and not done: restoring a cloned joint set into a live pipeline ([rapier#1053](https://github.com/dimforge/rapier/issues/1053)) |
| P8 | The shipped artifact calls no transcendental function of the platform's maths library and no fused multiply-add | `ci/imports.py` on the probe | False at the release; holds at the pinned commits (`imports: no platform maths` on Linux, macOS and Windows) |
| P9 | Solver defaults that move numbers or a teaching claim are known and set | See below | A reading |

The solver defaults read at the pin, each of which the crate sets explicitly:

- **Sleeping**: an island sleeps after 0.5 s below 0.05 units/s and 0.5 rad/s
  (`RigidBodyActivation`). Every body is built with `can_sleep(false)`.
- **Speed caps**: after each substep a body's linear speed is clamped to
  `normalized_max_linear_velocity` (400) times `length_unit` (1), in units per second, and its
  angular speed to 45 degrees per step (`MAX_ROTATION` times the step rate, about 47 rad/s at
  60 Hz) unless the body allows fast rotation.
- **Continuous collision detection** runs for fast bodies against fixed colliders whenever
  `max_ccd_substeps` is above 0 (default 1); `ccd_enabled` extends it to moving bodies.
- **Contact recycling** (default on): a pair that moved less than
  `normalized_contact_recycle_distance` (0.05, set explicitly) keeps its previous contact points,
  so contact data can be stale by up to that distance.
- **Contact clustering** (3D, default on), **joint warm-starting** (off), **friction in the bias
  pass** (off), 4 solver iterations (128 for a component with a joint, above), 1 internal and 1
  stabilization iteration, a step of 1/60 s, an allowed linear error of 0.005 and a prediction
  distance of 0.02, and the motors' force-based model.
- **`block-solver`** is a compile-time feature, on by default in 2D only; **precision** is the
  crate's choice of `rapier2d`/`rapier3d` (`f32`) over the `-f64` crates.

### The arithmetic audit

The probe is built with the features the crate ships; `ci/imports.py` lists its undefined symbols
and fails on any of `sin cos sincos tan asin acos atan atan2 exp exp2 exp10 expm1 log log2 log10
log1p pow cbrt hypot sinh cosh tanh asinh acosh atanh fma`, each with its `f` form. When it fails,
`ci/imports.py --where` prints, for each call, the source lines it comes from (a build with line
tables). The linker drops what nothing reaches, so the probe uses every feature the crate will
expose.

At the release, the probe imported `acosf asinf atan2f cosf expf log2f powf sincosf sinf` on Linux.
At the pinned commits it imports none, on Linux, macOS and Windows.

The one local change to the solver's sources (`vendor/VENDORED.md`, `vendor/rapier-speed-caps.patch`:
`rapier/src/dynamics/solver/solver_body.rs`, `solver/staged_island_solver/worker.rs` and
`rigid_body.rs`) sets and copies integer flags where the speed caps scale a velocity down, so the
crate can find a speed above its limit that the step's last velocity no longer shows. It adds and
changes no floating-point operation and no call, so this audit's conclusions hold for it; the
probe's digests do not move.

In the table, † marks a call site `ci/imports.py --where` resolved in the build at the release;
‡ one read in the source only (that run resolved no site for `cosf` and `log2f`, and the constant
`sin_cos` may have been folded at compile time, which no import list can see).

A **path** is one solver feature a scene can ask for. Each is in one of three states:
**corrected** (a pinned change after which its calls are gone from the list and its probe case
gives the same bits on Linux x86-64, Windows x86-64 and macOS arm64), **excluded** (refused by the
crate when a scene declares it) or **not linked** (not in the built artifact). None is excluded.

| Path | Call at the release | State |
| --- | --- | --- |
| Angular limit of a pin joint, 2D and 3D (and of a ball joint's axis) | `sincosf`, rapier `joint_constraint_helper.rs:62` (2D), `:64` (3D), from `joint_constraint_builder.rs:53,302` and `generic_joint_constraint_builder.rs:690` † | Corrected: `angular-limits-2d-published`, `angular-limits-3d` |
| 2D motor driving a joint to an angle, on a body jointed to a multibody link | `sinf` ×2, rapier `generic_joint_constraint_builder.rs:766` † | Corrected: `angular-motor-2d-published` |
| 3D motor driving a joint to an angle, on a body jointed to a multibody link | `sinf`, rapier `generic_joint_constraint_builder.rs:769` † | Corrected: `angular-motor-3d` |
| Angle of a 3D pin joint (`RevoluteJoint::angle`) | `asinf`, rapier `revolute_joint.rs` † | Corrected: `revolute-angle-3d-published` |
| Inertia of a 3D shape whose tensor is not diagonal (convex hulls, compounds, rotated parts) | `acosf`, glamx `eigen3.rs:101` †; `cosf`, `eigen3.rs:105,106` ‡ | Corrected: `inertia-hull-3d`, `inertia-compound-3d` |
| Broad phase (its tree's optimisation budget) | `log2f`, parry `bvh_optimize.rs` ‡ | Corrected: every world |
| A ball against a 3D convex polyhedron (the contact asks the polyhedron for the feature nearest a point, `support_feature_id_toward`, whose tolerance is an angle) | `sin_cos` of a constant, parry `convex_polyhedron.rs` ‡ | Corrected: `ball-on-hull-3d`. Contacts between two polyhedra choose features by a dot-product scan (`local_support_feature_toward`) and take no transcendental function |
| Triangle-mesh pseudo-normals | `acosf`, parry `vector_ext.rs:60`, from `trimesh.rs:1451-1453` † | Corrected (calls). No declared input reaches it: there is no triangle mesh in the authoring API; its calls are gone from the import list |
| Soft bodies (elasticity, plasticity, damping, tearing, shape matching, dihedral angles, builders) | `expf`, `powf`, `atan2f`, `acosf`, rapier `soft_*`, reached from the step with no soft body declared † | Corrected (calls). No declared input reaches it: there is no soft body in the authoring API; its calls are gone from the import list |
| Soft-body Delaunay refinement | spade's `sin`, `log2`, `exp2` | Not linked: `spade` is not in the tree without `std` |
| 2D singular value decomposition | `atan2`, `sin`, `cos`, glamx `svd2.rs` | Not linked: no call in the artifact |
| Multibody inverse kinematics | none (a matrix kernel that differs between native code and WebAssembly) | Not linked: the crate exposes no inverse kinematics |
| Missed tear under a forced thread schedule | none (a scheduling race) | Not linked: `parallel` is off |
| Combining friction coefficients of opposite zero signs by their maximum | none (`f32::max` of `-0.0` and `0.0`, which may return either) | Not a maths-library call, and it moves no body: all three systems give `00000000` at the pin (`contact-coefficients-2d-published`); the reproducers saw `80000000` on x86-64 |

### The probe

`examples/probe/` builds a 2D and a 3D world from literal numbers only (bodies of every shape kind
in contact, with friction and restitution; chains of pin and ball joints with and without limits
and motors; slide, weld, rope and spring joints; uniform gravity; and, applied as forces computed
at each tick (`add_force`, `add_force_at_point`), attraction by the inverse square, linear drag
`-k v`, quadratic drag `-c |v| v`, a damped spring between anchors on two bodies, and a constant
force and torque; rapier's own linear and angular damping too), steps each for 600 ticks, and
prints the SHA-256 of its trace: every body's position, rotation and velocities, and the pairs in
contact, at every tick. Then each case, as the bits of its result. The cases named `published`
rebuild, with the same literal inputs, cases from a
[published set of reproducers](https://github.com/caspark/2026-10-27-rapier-determinism-breakage-repros)
for rapier 0.36.1.

At the release, built as the reproducers were (with `std`):

| Case | Linux x86-64 | Windows x86-64 | macOS arm64 |
| --- | --- | --- | --- |
| `angular-motor-2d-published` | `39a3ae90` | `39a3ae8e` | `39a3ae90` |
| `angular-limits-2d-published` | `bd3f472d` | `bd3f472d` | `bd3f472e` |
| `revolute-angle-3d-published` | `3f94f0a0` | `3f94f0a0` | `3f94f0a0` |

The published differences reproduce. At the pinned commits every case and both worlds agree on the
three systems, and the outputs of the two x86-64 systems are byte-identical: `39a3ae8e`,
`bd3f472d`, `3f94f09f`, the values `libm` gives. The worlds as first written (before the applied
drag and spring were added) gave the same digests at the release and at the pin, on every system:
their inputs reach none of the calls with a value at which the platform's functions round
differently. That is why two agreeing runs prove
nothing, and why the audit reads the import list.

Run it with `cargo run --release --locked --example probe` from `sim/`.
