# Notice

Arachne is a visual engine for teaching, developed as a fork of
[Manim Community Edition](https://github.com/ManimCommunity/manim) (forked at v0.22.1).

## Third-party code

Portions of this repository are derived from Manim Community Edition and, through it, from
[3Blue1Brown's Manim](https://github.com/3b1b/manim). That code remains under its original MIT licenses,
whose copyright and permission notices are kept in full:

- `LICENSES/manim-3b1b-MIT.txt`: Copyright (c) 2018 3Blue1Brown LLC
- `LICENSES/manim-community-MIT.txt`: Copyright (c) 2024, the Manim Community Developers

Those notices apply to the code that came from those projects, including the history of this repository up to and
including the Manim Community Edition v0.22.1 release commit.

## Ideas credited

These ideas came from other projects; no code was taken from them.

- Immutable geometry and paint values behind a mutable-looking mobject API (read-only points, style as one immutable
  `Paint` value): [ManimGX](https://github.com/academa-labs/manimgx) by Academa, Inc. (MIT).
- Holding cost laws with tests that count work (values built, hashed and copied) instead of timing it: ManimGX, as
  above.
- A rational scene clock, on which durations become simple fractions and spans line up exactly
  (`arachne/timeline/clock.py`): ManimGX's `clock.py`, as above.
- Naming the mobjects a finding is about from the local variables of the scene's frames
  (`arachne/timeline/source.py`): ManimGX's `cli/diagnostics.py`, as above.
- Finding the scene's own line by walking the stack out of the library, and refusing `play()` while another play
  runs, as from an updater (`arachne/timeline/source.py`, `arachne/manager.py`): Manim Community Edition
  [pull request #5006](https://github.com/ManimCommunity/manim/pull/5006).
- A frame description made of records that reference content-keyed shapes uploaded once, each placed by an affine
  map, blended between two placed shapes, with a reveal window and a paint mix, so that a morph frame or a camera
  move ships no geometry (the core's XF): ManimGX, as above.

The core's content keys use BLAKE2b (RFC 7693, by Jean-Philippe Aumasson, Samuel Neves, Zooko Wilcox-O'Hearn and
Christian Winnerlein), implemented in `core/src/xf/key.rs` from the RFC.

## Dependencies of the core

The core (`core/`, the crate `arachne-core` and its Python binding) is built with these Rust crates, under their own
licences, none of them GPL:

| Crate | Licence | Used for |
|---|---|---|
| `libm` | MIT | transcendental functions in pure Rust |
| `pyo3`, `pyo3-ffi`, `pyo3-macros`, `pyo3-macros-backend`, `pyo3-build-config` | MIT OR Apache-2.0 | the Python binding |
| `numpy` (rust-numpy) | BSD-2-Clause | NumPy arrays in the binding |
| `ndarray` | MIT OR Apache-2.0 | through `numpy` |
| `matrixmultiply`, `rawpointer` | MIT OR Apache-2.0 | through `ndarray` (never on the drawing path) |
| `num-complex`, `num-integer`, `num-traits` | MIT OR Apache-2.0 | through `numpy` and `ndarray` |
| `libc`, `once_cell`, `rustc-hash`, `portable-atomic`, `portable-atomic-util`, `heck`, `autocfg` | MIT OR Apache-2.0 | through `pyo3`, `numpy` and `ndarray` |
| `proc-macro2`, `quote`, `syn` | MIT OR Apache-2.0 | build time, through `pyo3`'s macros |
| `unicode-ident` | (MIT OR Apache-2.0) AND Unicode-3.0 | build time, through `syn` |
| `target-lexicon` | Apache-2.0 WITH LLVM-exception | build time, through `pyo3-build-config` |

Its wheel is built by `maturin` (MIT OR Apache-2.0) and depends on NumPy (BSD-3-Clause) at run time.

## Arachne's own work

No license is granted for the changes and additions made in this repository after the fork. All rights to them are
reserved by their authors. The repository is public so that its development and CI are visible, not as an offer of a
license.
