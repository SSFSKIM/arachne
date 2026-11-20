# Vendored sources

The solver's determinism rests on three upstream changes that no release carries yet: each
rewrites calls to the platform's maths library (`x.sin()` on a concrete float) as calls through
simba's traits, which the solver's `enhanced-determinism` feature routes to the pure-Rust `libm`
(`../README.md`, "Determinism"). Their exact source is part of the pin, so it is kept here rather
than fetched: `../Cargo.toml` patches each crate to its directory below, and `ci/tree_check.py
deps` fails if one resolves from anywhere else or the lock file names a git source.

Only what the build compiles is kept: each crate's sources and manifest, and each upstream
license. Examples, benches, tests, testbeds, assets, documentation and CI files are left out.

| Directory | Upstream | Base release | Commit | Carries |
| --- | --- | --- | --- | --- |
| `rapier/` | github.com/dimforge/rapier | v0.36.1 | `b303bc13e7e69abcec31c595f095de71edf2d3d3` | dimforge/rapier pull request 1074: the `sin_cos`, `sin` and `asin` calls of joint limits, angular motors and a revolute joint's angle go through the scalar traits (and two commits after the release that touch `bevy_rapier` only, nothing compiled here) |
| `parry/` | github.com/dimforge/parry | v0.31.1 | `dbcce4095c5c65277de1d3363d59c9a75cc02a87` | dimforge/parry pull request 468: the broad phase's `log2`, a convex polyhedron's feature tolerance (`sin_cos`) and a triangle mesh's pseudo-normals (`acos`) go through the scalar traits (and one commit after the release, built only with `serde-serialize`, which is off) |
| `glamx/` | github.com/dimforge/glamx | v0.3.1 | `78c97db71c2724b096cf751d2024c405c65d8612` | dimforge/glamx pull request 10: a symmetric 3x3 eigen-decomposition's `acos` and `cos` (the inertia of a convex hull or a compound) go through the scalar traits |

What is changed from upstream, all in manifests and none in a source file:

- `rapier/Cargo.toml` and `parry/Cargo.toml`, the workspace roots the crates inherit from: the
  members are the two crates kept; `default-members`, `exclude`, the `[patch]` and `[profile]`
  sections, comments, and the workspace dependencies no kept crate inherits are dropped.
- Every crate's manifest: `[dev-dependencies]`, `[[example]]` and `readme` are dropped (none of
  their files is kept).

The sources are byte for byte upstream's at the commit: `git diff --no-index` of `src/` against a
checkout of the commit shows nothing.

**Exit condition.** When a release of each crate carries its fix, the pin moves to that release
(crates.io, by exact version, as `Cargo.toml` pins the rest) and the copy here is deleted, after
the probe (`../examples/probe`) gives the same bits on the three systems as it does on these
sources.
