# arachne-sim

Arachne's simulation crate: rigid-body worlds in 2D and 3D, stepped on a fixed clock of 60 ticks
per second, whose results are the same bytes on every machine of one architecture. It is built on
[rapier](https://rapier.rs) with its `enhanced-determinism` feature, stands apart from the core
(`core/`): its own manifest, lock file, toolchain and clippy rules, and no dependency on
`arachne-core` either way.

At this stage the crate holds the pinned solver, the determinism probe and the checks; the Python
binding comes next.

| Path | What it is |
| --- | --- |
| `Cargo.toml`, `Cargo.lock` | the solver pinned by exact version and commit, its features listed one by one |
| `rust-toolchain.toml` | the core's toolchain, 1.99.1 |
| `clippy.toml` | the core's rules: no fused multiply-add, no platform transcendental function, in this crate's code |
| `examples/probe/` | the determinism probe (below) |
| `tests/stepping.rs` | a world's bytes do not depend on how its steps are grouped, nor on a world before it |
| `ci/` | the gates' scripts and the checks they run |

The gates (`.github/ci-kit.yml`) run when `sim/` changes: `sim-lint` (format, clippy, the checks'
own tests, the dependency tree, the census of run-time CPU detection, the probe's import list),
`sim-test` (`cargo test --release --locked`), `sim-identity` (the probe built for `x86-64` and
for `x86-64-v3`, run twice, against the output committed in `ci/probe-x86_64.txt`) and
`sim-macos`. The workflow `sim-probe.yml` runs the probe on Linux x86-64, Windows x86-64 and macOS
arm64 and compares the outputs.

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
| `rapier2d` | 0.36.1 | [dimforge/rapier@b303bc13](https://github.com/dimforge/rapier/commit/b303bc13e7e69abcec31c595f095de71edf2d3d3) | `dim2`, `f32`, `alloc`, `block-solver`, `enhanced-determinism` |
| `rapier3d` | 0.36.1 | the same commit | `dim3`, `f32`, `alloc`, `enhanced-determinism` |
| `parry2d`, `parry3d` | 0.31.1 | [dimforge/parry@dbcce409](https://github.com/dimforge/parry/commit/dbcce4095c5c65277de1d3363d59c9a75cc02a87) | as rapier asks: `required-features`, `alloc`, `enhanced-determinism` |
| `glamx` | 0.3.1 | [dimforge/glamx@78c97db7](https://github.com/dimforge/glamx/commit/78c97db71c2724b096cf751d2024c405c65d8612) | as parry asks |
| `simba` | 0.10.2 | crates.io | `libm_force` (from `enhanced-determinism`) |
| `nalgebra` | 0.35.0 | crates.io | `macros`, no `std` |
| `libm` | 0.2.16 | crates.io | the pure-Rust maths every transcendental function goes to |

Single precision, no `std`, no `parallel`, no `simd8`. `block-solver` is each dimension's upstream
default (on in 2D, off in 3D), named rather than left to the default.

The three commits are upstream's own fix for the calls described below, open as pull requests on
the day of the pin: [rapier#1074](https://github.com/dimforge/rapier/pull/1074),
[parry#468](https://github.com/dimforge/parry/pull/468) and
[glamx#10](https://github.com/dimforge/glamx/pull/10) (issue
[rapier#1054](https://github.com/dimforge/rapier/issues/1054)). Each is the release plus one
commit; the source changes are the call sites in the table of paths, each rewritten from a method
call (`x.sin()`) to the trait's function (`<Real as ComplexField>::sin(x)`), plus tests and
changelog lines. The pin moves to the first release that carries them.

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
| P5 | Nothing in the built tree picks its arithmetic from the CPU it runs on | `ci/census.py` over the shipped tree: one file, `libm`'s detection, used only to choose the processor's fused multiply-add or libm's software one, which IEEE 754 defines to give the same bits. Without `std`, `nalgebra` drops `matrixmultiply`, whose kernels are chosen at run time | Holds (`ci/census.txt`) |
| P6 | The result does not depend on the vector instructions the compiler was allowed | The probe built for `x86-64` and for `x86-64-v3` | Holds: byte-identical (`sim-identity`) |
| P7 | Stepping in one call or in pieces, in a fresh process or a second world of one process, gives the same bytes | The probe run twice; `tests/stepping.rs` | Holds. Not covered and not done: restoring a cloned joint set into a live pipeline ([rapier#1053](https://github.com/dimforge/rapier/issues/1053)) |
| P8 | The shipped artifact calls no transcendental function of the platform's maths library and no fused multiply-add | `ci/imports.py` on the probe | False at the release; holds at the pinned commits (`imports: no platform maths` on Linux, macOS and Windows) |
| P9 | Solver defaults that move numbers or a teaching claim are known and set | See below | A reading |

The solver defaults read at the pin, each of which the crate sets explicitly:

- **Sleeping**: an island sleeps after 0.5 s below 0.05 units/s and 0.5 rad/s
  (`RigidBodyActivation`). Every body is built with `can_sleep(false)`.
- **Speed caps**: linear speed is capped at `normalized_max_linear_velocity` (400) times
  `length_unit` (1) per substep, and rotation at 45 degrees per step (`MAX_ROTATION`, about 47
  rad/s at 60 Hz) unless a body allows fast rotation.
- **Continuous collision detection** runs for fast bodies against fixed colliders whenever
  `max_ccd_substeps` is above 0 (default 1); `ccd_enabled` extends it to moving bodies.
- **Contact recycling** (default on): a pair that moved less than 0.05 keeps its previous contact
  points, so contact data can be stale by up to that distance.
- **Contact clustering** (3D, default on), **joint warm-starting** (off), **friction in the bias
  pass** (off), 4 solver iterations, 1 internal and 1 stabilization iteration, a step of 1/60 s.
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

A **path** is one solver feature a scene can ask for. Each is in one of three states:
**corrected** (a pinned change after which its calls are gone from the list and its probe case
gives the same bits on Linux x86-64, Windows x86-64 and macOS arm64), **excluded** (refused by the
crate when a scene declares it) or **not linked** (not in the built artifact). None is excluded.

| Path | Call at the release | State |
| --- | --- | --- |
| Angular limit of a pin joint, 2D and 3D (and of a ball joint's axis) | `sin_cos`, rapier `joint_constraint_helper.rs:62,64` | Corrected: `angular-limits-2d-published`, `angular-limits-3d` |
| Motor driving a joint to an angle, on a body jointed to a multibody link | `sin` ×2, rapier `generic_joint_constraint_builder.rs:766,769` | Corrected: `angular-motor-2d-published` |
| Angle of a 3D pin joint (`RevoluteJoint::angle`) | `asin`, rapier `revolute_joint.rs` | Corrected: `revolute-angle-3d-published` |
| Inertia of a 3D shape whose tensor is not diagonal (convex hulls, compounds, rotated parts) | `acos`, `cos`, glamx `eigen3.rs:101,105,106` | Corrected: `inertia-hull-3d`, `inertia-compound-3d` |
| Broad phase (its tree's optimisation budget) | `log2`, parry `bvh_optimize.rs` | Corrected: every world |
| Contacts of a 3D convex polyhedron | `sin_cos`, parry `convex_polyhedron.rs` | Corrected: `hull-at-rest-3d` |
| Triangle-mesh pseudo-normals | `acos`, parry `vector_ext.rs:60` | Corrected (calls); no case: the crate exposes no triangle mesh |
| Soft bodies (elasticity, plasticity, damping, tearing, shape matching, dihedral angles, builders) | `exp`, `powf`, `atan2`, `acos`, `sin`, `cos`, rapier `soft_*` | Corrected (calls); no case: the step reaches this code with no soft body declared, and the crate exposes none |
| Soft-body Delaunay refinement | spade's `sin`, `log2`, `exp2` | Not linked: `spade` is not in the tree without `std` |
| 2D singular value decomposition | `atan2`, `sin`, `cos`, glamx `svd2.rs` | Not linked: no call in the artifact |
| Multibody inverse kinematics | none (a matrix kernel that differs between native code and WebAssembly) | Not linked: the crate exposes no inverse kinematics |
| Missed tear under a forced thread schedule | none (a scheduling race) | Not linked: `parallel` is off |
| Combining friction coefficients of opposite zero signs by their maximum | none (`f32::max` of `-0.0` and `0.0`, which may return either) | Not a maths-library call, and it moves no body: all three systems give `00000000` at the pin (`contact-coefficients-2d-published`); the reproducers saw `80000000` on x86-64 |

### The probe

`examples/probe/` builds a 2D and a 3D world from literal numbers only (bodies of every shape kind
in contact, with friction and restitution; chains of pin and ball joints with and without limits
and motors; slide, weld, rope and spring joints; uniform gravity, attraction by the inverse
square, linear and quadratic drag, a constant force and torque), steps each for 600 ticks, and
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
three systems: `39a3ae8e`, `bd3f472d`, `3f94f09f`, the values `libm` gives. The worlds' digests did
not move between the release and the pin, on any system: their inputs reach none of the calls with
a value at which the platform's functions round differently. That is why two agreeing runs prove
nothing, and why the audit reads the import list.

Run it with `cargo run --release --locked --example probe` from `sim/`.
