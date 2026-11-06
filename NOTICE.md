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
including the Manim Community Edition v0.22.1 release commit, and the core's Rust ports of CE's camera, frame and
shading arithmetic (`core/src/project.rs`: projection and the frame-to-pixel mappings; `core/src/composite.rs`: the
background's bytes and the background image's crop; `core/src/shade.rs`: `get_shaded_rgb`, `get_unit_normal` and
the per-face stops of `ThreeDCamera.modified_rgbas`).

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
- In the lint (`arachne/lint/`): reading a text whole, as the ink of its glyphs, and two copies of one text with the
  same ink in one place as one text; marking the parts of grids (number planes, vector fields) apart; a `note`
  severity for findings that are likely meant; folding more than three look-alike findings into the first,
  which counts the rest; and 20 px at 1080 px tall as the bound for small text: ManimGX's `cli/diagnostics.py`, as
  above.
- Finding the scene's own line by walking the stack out of the library, and refusing `play()` while another play
  runs, as from an updater (`arachne/timeline/source.py`, `arachne/manager.py`): Manim Community Edition
  [pull request #5006](https://github.com/ManimCommunity/manim/pull/5006).
- A frame description made of records that reference content-keyed shapes uploaded once, each placed by an affine
  map, blended between two placed shapes, with a reveal window and a paint mix, so that a morph frame or a camera
  move ships no geometry (the core's XF): ManimGX, as above.
- Resolving each pixel's fragments, each with a coverage mask over sub-samples, by depth, so antialiasing, hidden
  surfaces and transparency resolve together (the core's compositor): the A-buffer, Loren Carpenter, "The A-buffer,
  an antialiased hidden surface method", SIGGRAPH 1984.

## Algorithms the core reimplements

These are written in Rust in the core, from a specification or from another project's implementation.

The core's content keys use BLAKE2b (RFC 7693, by Jean-Philippe Aumasson, Samuel Neves, Zooko Wilcox-O'Hearn and
Christian Winnerlein), implemented in `core/src/xf/key.rs` from the RFC. The core's tests pin bytes with SHA-256
(FIPS 180-4, a NIST standard), implemented in `core/tests/wire.rs` from the standard. Neither specification needs a
licence file.

The core's rotation matrices (`core/src/project.rs`, `rotation_matrix`) are a Rust port of the arithmetic of
[SciPy](https://scipy.org)'s `Rotation.from_rotvec` and `Rotation.as_matrix`, including the Taylor-series constants
it uses for small angles, so that they match CE's `space_ops.rotation_matrix`, which calls them. That arithmetic
remains under SciPy's BSD-3-Clause license, whose copyright notice, conditions and disclaimer are kept in full:

- `LICENSES/scipy-BSD-3-Clause.txt`: Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers

The core blends 2D paint (`core/src/composite.rs`, `premultiply` and `blend`) with the 8-bit arithmetic of
[Cairo](https://cairographics.org)'s ARGB32 image surface: a colour premultiplied to 16 bits and truncated to 8, as
Cairo stores a solid source; a translucent source multiplied by the coverage and laid over the pixel with
[pixman](https://pixman.org)'s rounded `x * y / 255`; an opaque source interpolated toward the pixel with the rounding
of Cairo's span compositor. The formulas were written from that arithmetic, not copied, and are checked against bytes
Cairo 1.18 wrote (`core/tests/data/gen_blend.py`). Also from specifications: the sRGB transfer functions (IEC
61966-2-1), and Direct3D's standard 16-sample pattern, moved half a cell so its centroid is the pixel's centre
(`core/src/composite.rs`).

The path rasterizer's coverage (`core/src/scanline.rs`) accumulates each edge's signed area into the cells it crosses
and prefix-sums them along the row, the analytic method font rasterizers use (among them
[font-rs](https://github.com/raphlinus/font-rs), by Raph Levien), written from its published description, not from
code; it cuts rows at crossings and accumulates only the edges where the nonzero
winding changes, so self-overlapping outlines stay exact. The planes of flat paths (`core/src/depth.rs`) use Newell's
method for a polygon's normal (M. E. Newell, as given in Sutherland, Sproull and Schumacker, "A Characterization of
Ten Hidden-Surface Algorithms", 1974). CE's subpath split and close rules are reimplemented from CE 0.20.1
(`VMobject.get_subpath_split_indices_from_points`, `consider_points_equals_2d`), under CE's MIT licence like the rest of
this fork.

## Dependencies of the core

The core (`core/`, the crate `arachne-core` and its Python binding) is built with these Rust crates, under their own
licences, none of them GPL:

| Crate | Licence | Used for |
|---|---|---|
| `libm` | MIT | transcendental functions in pure Rust |
| `kurbo` | Apache-2.0 OR MIT | stroking, dashing and flattening paths (built without `std`, on `libm`) |
| `polycool`, `arrayvec`, `smallvec` | MIT OR Apache-2.0 | through `kurbo` |
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
