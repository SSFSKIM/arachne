# Vendored sources

The solver's determinism rests on three upstream changes that no release carries yet: each
rewrites calls to the platform's maths library (`x.sin()` on a concrete float) as calls through
simba's traits, which the solver's `enhanced-determinism` feature routes to the pure-Rust `libm`
(`../README.md`, "Determinism"). Their exact source is part of the pin, so it is kept here rather
than fetched: `../Cargo.toml` patches each crate to its directory below, and `ci/tree_check.py
deps` fails if one resolves from anywhere else or the lock file names a git source.

Only what the build compiles is kept: each crate's sources and manifest, and each upstream
license. Examples, benches, tests, testbeds, assets, documentation and CI files are left out, and
so are the source files that only a `#[cfg(test)]` module declaration names (upstream's unit
tests, which a dependency's build never compiles).

| Directory | Upstream | Base release | Commit | Carries |
| --- | --- | --- | --- | --- |
| `rapier/` | github.com/dimforge/rapier | v0.36.1 | `b303bc13e7e69abcec31c595f095de71edf2d3d3` | dimforge/rapier pull request 1074: the `sin_cos`, `sin` and `asin` calls of joint limits, angular motors and a revolute joint's angle go through the scalar traits (and two commits after the release that touch `bevy_rapier` only, nothing compiled here) |
| `parry/` | github.com/dimforge/parry | v0.31.1 | `dbcce4095c5c65277de1d3363d59c9a75cc02a87` | dimforge/parry pull request 468: the broad phase's `log2`, a convex polyhedron's feature tolerance (`sin_cos`) and a triangle mesh's pseudo-normals (`acos`) go through the scalar traits (and one commit after the release, built only with `serde-serialize`, which is off) |
| `glamx/` | github.com/dimforge/glamx | v0.3.1 | `78c97db71c2724b096cf751d2024c405c65d8612` | dimforge/glamx pull request 10: a symmetric 3x3 eigen-decomposition's `acos` and `cos` (the inertia of a convex hull or a compound) go through the scalar traits |

What is changed from upstream: the manifests, and one change to rapier's sources (below).
In the manifests:

- `rapier/Cargo.toml` and `parry/Cargo.toml`, the workspace roots the crates inherit from: the
  members are the two crates kept; `default-members`, `exclude`, the `[patch]` and `[profile]`
  sections, comments, and the workspace dependencies no kept crate inherits are dropped.
- Every crate's manifest: `[dev-dependencies]`, `[[example]]` and `readme` are dropped (none of
  their files is kept), and a two-line notice at its head says it was modified.

The upstream licenses (`rapier/LICENSE`, `parry/LICENSE`, `glamx/LICENSE-APACHE`,
`glamx/LICENSE-MIT`) ship in the simulation wheel too, under its `.dist-info/licenses/vendor/`
(`../pyproject.toml`, `license-files`; `ci/binding.sh` checks they are there).

**The one change to a source: the speed caps flag the bodies they scale down.** The solver
scales a body's velocity down to its caps inside each substep (linear: `max_linear_velocity`;
angular: 45 degrees a step, unless the body allows fast rotation), and a later constraint, or a
length that overflowed to infinity (which scales the velocity to zero), can leave the step's
last velocity at or under the crate's speed limit, so the crate could not tell from its rows
that a speed went above the limit (`../src/world.rs`, `World::findings`). The change sets a flag
where each cap acts and copies it to the body as the step writes the body back; the crate reads
it after the step. It is integer flags only: no floating-point operation is added or changed,
every number the solver computes is computed as before, and the trace's bytes do not change (the
probe's digests are the same on the three systems). In `rapier/src/dynamics/`:

- `solver/solver_body.rs`, after `SOLVER_BODY_ALLOW_FAST_ROTATION`: the flags
  `SOLVER_BODY_LINEAR_CAPPED` and `SOLVER_BODY_ANGULAR_CAPPED`, kept in the per-body flag byte
  that `SolverBodies::copy_from` already rewrites for each body every step (which clears them);
- `solver/staged_island_solver/worker.rs`: their import; in the substep's speed cap (the
  integrate-positions stage), `flags[i] |= ...` inside each of the three branches that scale a
  velocity down (linear, 2D angular, 3D angular); and in the velocity writeback, after
  `rb.vels = new_vels`, the two flags copied to the body as booleans;
- `rigid_body.rs`: the field `speeds_capped: (bool, bool)` (false when made, skipped by
  `RigidBody::copy_from` as the engine's own) and its reader `RigidBody::speeds_capped()`.

Each of the three files carries a two-line notice at its head that it was modified. The change
is recorded as a patch beside the sources, `rapier-speed-caps.patch` (the lines it adds, with
their context); `ci/tree_check.py patches` (in `ci/lint.sh`) fails unless the sources hold it.

`git diff --no-index` of `rapier/src/` against a checkout of the commit shows that change and
nothing else besides the removals below. The rest of the sources kept are byte for byte
upstream's at the commit: a diff shows nothing but the removal of these unit-test files:

- `parry/src/partitioning/bvh/bvh_tests.rs`
- `rapier/src/pipeline/physics_pipeline/test.rs` and `test_staged.rs`
- `rapier/src/dynamics/solver/soft_fem/system/tests.rs`
- `rapier/src/dynamics/solver/soft_constraint/soft_element_constraint/test.rs`
- `rapier/src/dynamics/joint/multibody_joint/multibody_regression_tests.rs`
- `rapier/src/dynamics/soft_body/tearing/tests.rs`
- `rapier/src/geometry/narrow_phase/test.rs`

**Exit condition.** When a release of each crate carries its fix, the pin moves to that release
(crates.io, by exact version, as `Cargo.toml` pins the rest) and the copy here is deleted, after
the probe (`../examples/probe`) gives the same bits on the three systems as it does on these
sources. Rapier's speed-cap flags are not upstream: when its pin moves, the new sources are
copied here and `rapier-speed-caps.patch` is applied to them (`git apply
--directory=sim/vendor/rapier sim/vendor/rapier-speed-caps.patch`, from the repository's root;
`tree_check.py patches` fails until it is), unless a release carries an equivalent report the
crate reads instead, in which case the patch is deleted with the copy.
