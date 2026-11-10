# The truth contract

Version 1, written against the core at commit `3ce63425` (XF 1.4).

This text states what it means for Arachne's CPU core to have **decided right** when it draws a
frame from an XF stream: at each of the 16 sub-samples of every pixel, which fragments cover the
sub-sample and in what order they composite, and how those decisions become the pixel's four bytes.
Three things are written from this text alone and never from the core's source: the **oracle**
(which recomputes the decisions from the XF stream by brute force), the **recompute** (a second
implementation of the compositing arithmetic, which turns decisions and colours into bytes), and the
**contract tests** (which hold the core to the text).

## How to read it

- **Core rule** paragraphs state what the core does. Each is true of the core at `3ce63425` and
  names the module, function and tests it rests on (paths are under `core/`). A change to what the
  core does is a new version of the rule it changes, never an edit in place.
- **Oracle reading** paragraphs state how a decision is judged: what the oracle computes, and which
  of the core's decisions it admits. A reading errs toward admitting. A reading that admits a
  decision the core could not have made costs sensitivity, never a false verdict; a reading that
  refuses a decision the core makes under a stated rule is a defect of the reading, and the contract
  tests exist to find one.
- Every rule beyond plain depth carries a name and a version, `name@1`. The rules counted per
  sub-sample (the `per_policy` counts) are listed with ✓ in the table below. A change to a rule's
  text bumps its version, and every verdict it moves is replayed and reported.
- The oracle's outputs: per fragment per sub-sample, `Yes`, `Either` or absent (with the fragment's
  cell area for a path); per pair of present fragments per sub-sample, one relation, `Depth(front)`,
  `TieBand` or `Policy(name@1)`; per frame, the records it refuses (`truth_unable`).
- Numbers: `u = 2^-53`, the unit roundoff of IEEE binary64. Arithmetic is binary64 in the order the
  formulas give, with no fused multiply-add and transcendental functions from a correctly specified
  pure library (the core uses the `libm` crate); `round(v)` is `f64::round` (half away from zero).
  A depth **key** is the projection's factor negated, so a smaller key is nearer.

### The named rules

| Rule | § | What it decides | Counted |
|---|---|---|---|
| `mesh_fixed_point@1` | 3 | triangle corners rounded to 1/256 px before the sample test | |
| `mesh_edge_ownership@1` | 3 | a sample on a shared edge belongs to one triangle | |
| `point_blocks@1` | 3 | a point is CE's square block of whole pixels | |
| `cell_presence@1` | 4 | a path fragment covers a sample where its cell byte is not 0 | |
| `guard_chords@1` | 4 | curves far beyond the frame drawn as chords | |
| `curve_key@1` | 5 | the key along a curve's centreline | |
| `fringe_key@1` | 5 | a plane-keyed layer read at its outline beyond the outline | |
| `facing_plane@1` | 5 | camera-facing labels and points on the plane facing the camera | |
| `stroke_on_own_fill@1` | 5 | a stroke at least as near as its own fill where both cover | |
| `shared_plane_read@1` | 5 | a layer read on a shared identity's plane against what shares it | |
| `band_input@1` | 6 | how far the input grid's snap may move a key (`S`) | |
| `band_arith@1` | 6 | how far a conforming computation may stand from the exact key (`R`) | |
| `own_layers@1` | 7 | a fill against its own strokes keeps paint order | ✓ |
| `one_plane@1` | 7 | two fragments on one registered plane tie within the snap's term | ✓ |
| `near_parallel@1` | 7 | two flat records the input cannot tell apart in orientation tie | ✓ |
| `lines@1` | 7 | two strokes of paths with no plane of their own tie within their shares | ✓ |
| `nearest_plane_departure@1` | 7 | a face on no one plane carries its outline's departure | ✓ |
| `paint_order@1` | 7 | the order where no depth decides: later drawn in front | ✓ |
| `majority@1` | 7 | one order for the pixel where its sub-samples disagree | |
| `cycle_rank@1` | 7 | the order where pairwise orders cycle | ✓ |
| `join_mask@1` | 8 | Cairo's join decisions carried on the wire (XF 1.4) | ✓ |
| `miter_limit_10@1` | 8 | Cairo's miter limit, joints and caps | |
| `dash_start_least_length@1` | 8 | dashes laid on the whole outline; gaps under 1/512 stroke solid | ✓ |
| `round_cap_dot@1` | 8 | a face at one spot under a round cap strokes a dot | ✓ |
| `width_in_scene_units@1` | 8 | a stroke's width never shrinks with depth | |
| `opaque_drop@1` | 9 | what every sub-sample hides behind an opaque whole cover is dropped | |
| `unit_merge@1` | 9 | one surface's non-overlapping masks composite as one unit | |
| `resolve_select@1` | 9 | which resolve a pixel takes | |
| `cairo_premultiply@1` | 9 | a straight colour to premultiplied bytes | |
| `cairo_linear_gradient@1` | 9 | a gradient's bytes at a pixel, by its composite run | |
| `ce_shading@1` | 9 | CE's two lit stops of a shaded path; lit mesh corners | |
| `label_interim@1` | 10 | a contested sample of a camera-facing label is `Either` | `label_rule_open` |

The stroke policies are counted as the sub-samples covered by a layer whose outline the policy
changed (a join edited, a dash laid, a dot drawn). `paint_order@1` counts the sub-samples at which a
pair's relation is `TieBand`; `nearest_plane_departure@1` counts those at which it is `TieBand` only
by that term; `cycle_rank@1` counts the sub-samples whose admissible orders come from a cycle.

## 1. The frame and its pixels

**Core rule.** A frame of `W x H` pixels is `H * W * 4` bytes, row by row, each pixel R, G, B, A,
paint premultiplied as Cairo stores it (`composite.rs` module doc). Pixel `(x, y)` spans
`[x, x+1) x [y, y+1)` in pixel coordinates, `y` down. The frame starts as the background: a colour
written as CE writes it, each channel `(c * 255)` truncated toward zero and wrapped to a byte, not
premultiplied (`composite::background_pixel`); or an image's top-left `W x H` corner, refused when the
image is smaller (`Frame::cropped`). Each pixel holding a fragment is then resolved over it (§9).

The frame judged is the one the core draws without its **overlay** (`Stream::render_under_overlay`,
Python `draw(overlay=False)`): records in overlay space, `image` records and records with feature bit 0
are Cairo's, drawn over the core's frame in CE's order (`stream::is_overlay`). Their order entries
are neither drawn nor refused; every other entry keeps its index in the order, which is its place in
paint order (§7).

**Sample positions.** Sub-sample `s` (0..16) of pixel `(x, y)` lies at

    (x + 1/2 + (dx_s + 1/2) / 16,  y + 1/2 + (dy_s + 1/2) / 16)

for `[dx_s, dy_s] = SAMPLE_OFFSETS[s]`, in sixteenths of a pixel from its centre
(`composite::SAMPLE_OFFSETS`, `sample_point`):

    s:   0      1      2      3      4      5      6      7
        (-8,-7) (-1,-8) (3,-6)  (5,-5)  (-6,-1) (-4,-4) (0,-3)  (6,-2)
    s:   8      9      10     11     12     13     14     15
        (-7,2)  (-2,0)  (2,3)   (4,1)   (-5,5)  (-3,4)  (1,7)   (7,6)

Each row and each column of the 16 x 16 grid holds one sample (n-rooks), and each **cell** of the
pixel's 4 x 4 grid of 1/4 px cells holds one: sample `s` lies in cell column `(dx_s + 8) div 4`,
row `(dy_s + 8) div 4` (`composite::sample_cell`). The samples' centroid is the pixel's centre. In the
1/256 px fixed point the mesh rasterizer uses, sample `s` sits at `128 + 16 dx_s + 8`,
`128 + 16 dy_s + 8` from the pixel's corner, exactly the same point (`composite::sample_fixed`).
Tests: `tests/composite.rs::the_sample_pattern_is_stratified_and_centred_on_the_pixel`;
`tests/python/test_contributor_samples.py` restates the table.

**Oracle reading.** The oracle decides at exactly these 16 points and cells, and nowhere else.

## 2. The camera, placement and clipping

All of this is CE's camera arithmetic as `project.rs` restates it, in CE's order of operations; a
conforming implementation evaluates it in the order given, so its projected points are the core's to
the last bit.

**Placement.** A record's point `p` is placed by its 3 x 4 placement `M`:
`p'_i = M[i][0] p_0 + M[i][1] p_1 + M[i][2] p_2 + M[i][3]` (`xf::place`). Two placements compose
as `xf::compose(outer, inner)` (each entry a sum of the products, then the outer translation added).
A path record's drawn points are its shape placed, then cut to its reveal window (CE's
`pointwise_become_partial`), then blended with its second shape (CE's `interpolate`,
`(1 - t) p + t q`) (`geometry::path_points`, `geometry::revealed`); an instance's placement composes
after its record's (`instance.rs`); a face group's face `i` is its run of the record's points
(`FaceGroup::run`: at the face ends, or equal runs).

**The 3D projection** (`project::Projection`), under a view whose camera is 3D (phi `φ`, theta `θ`,
gamma `γ`, focal distance `d`, zoom `z0`, the exponential option), about the view's centre `c`
(`frame_center`):

- `R = Rz(γ) · Rx(-φ) · Rz(-θ - 90°)`, built by multiplying each factor onto the identity from the
  left in the order `Rz(-θ - 90°)`, `Rx(-φ)`, `Rz(γ)` (`rotation_matrix`); `Rz(a) = [[cos a, -sin a,
  0], [sin a, cos a, 0], [0, 0, 1]]`; `Rx(a)` is SciPy's rotation vector `a (1, 0, 0)` turned to a
  matrix through its quaternion, with `sin(|v|/2) / |v|` replaced by `1/2 - |v|²/48 + |v|⁴/3840`
  where `|v| <= 1e-3` (`rotation_about_axis`); a degree is `τ / 360`.
- `q = R (p - c)`, each row `q_k = (p - c)_0 R[k][0] + (p - c)_1 R[k][1] + (p - c)_2 R[k][2]`.
- The factor at depth `z = q_2`: under the exponential option `d / (d - z)` for `z < 0`, else
  `exp(z / d)`; otherwise CE's **behind-the-eye factor** `10^6` where `d - z < 0`, else `d / (d - z)`
  (so a point on the eye's plane takes `d / 0`) (`Projection::factor`).
- The projected point is `(q_0 f z0, q_1 f z0, q_2)`: the depth kept rotated, not scaled.
- The key of a projected point is `-f`. A point is **keyed planar** when the option is off and
  `d - q_2 > 0` (`keys_planar_at`); only there is a key affine across a plane on screen.

**Before display** (`project::pre_display`, CE's `transform_points_pre_display`): a mobject with any
non-finite point is drawn as one point at the origin (for a path, nothing: a path needs four points,
`path::drawn_world`); points fixed in frame and every point under a flat or moving camera are kept;
points of a record with a fixed-orientation centre `C` are moved by `project(C) - C`; the rest are
projected. Which placement a record takes (`project::placement`): a `world` record under a 3D
camera is projected (fixed-orientation when it has a centre); an `overlay` record is fixed in frame
(and is the overlay's, §1); a `screen` record, and every record under a flat or moving camera, is
kept. Records kept or fixed in frame are in front of all depth (key `-inf`, §5).

**The frame mapping** (`project::FrameMap`). From frame units to pixels, CE's Cairo context matrix:
`xx = W / fw`, `yy = -(H / fh)`, `x0 = W/2 - m_0 (W / fw)`, `y0 = H/2 + m_1 (H / fh)`, applied as
`x' = (xx x + 0 y) + x0`, `y' = (0 x + yy y) + y0` (`FrameMap::cairo`, `apply`). The centre `m` is
the view's `mapping_centre` for what CE draws through its Cairo context (paths, face groups' faces,
meshes, instances) and `frame_center` for point clouds and images (`mapping_centre_of`); point
clouds map by `subpixel_coords`, `(p_0 - c_0)(W / fw) + W/2`, `(p_1 - c_1)(-(H / fh)) + H/2`. The flat
and the moving camera map the same way; only the view's numbers differ.

**Pre-projected records** (feature bit 1, XF 1.1). The tail's points are taken as placed (identity
placement), its paints and gradient ends as produced, never projected or lit again
(`instance::produced_path`). Their depth is keyed in the camera's own axes: each shown point is put
back by `unproject`, `(x / (f(z) z0), y / (f(z) z0), z)`, and keyed through the same projection
without rotation or centre (`Projection::in_camera_axes`); a fixed-orientation centre is moved as
the record's first point moved (`project::produced_pivot`). Tests:
`tests/pre_projected.rs::a_pre_projected_record_never_moves_with_the_view_s_rotation`.

**Clipping.** Nothing is clipped at the eye: a point behind it takes the factor `10^6` and is drawn
where that puts it, and a non-finite position drops what it belongs to (a triangle with a non-finite
corner, a ring with a non-finite point: `mesh::Meshes::push`, `scanline::edges_of`)
(`tests/projection.rs::a_point_behind_the_eye_takes_ces_large_factor`). Fragments exist only on the
frame's pixels. Triangles reaching past `±65536` px are clipped there and fanned from the clipped
polygon's first corner (`mesh::clip_to_guard`), pieces of one triangle. Path curves beyond the guard
area are drawn as chords (`guard_chords@1`, §4).

**Oracle reading.** The oracle projects every record by this text. Its camera is checked against
closed forms and against the engine's own `arachne.camera.projection` (screen boxes within 1 px).

## 3. Mesh coverage

**Core rule** (`mesh.rs` module doc, `Meshes::push`, `Meshes::raster`). A mesh triangle's corners,
placed, shown (§2) and mapped (`FrameMap`), are rounded to the 1/256 px grid
(`mesh_fixed_point@1`: `round(v * 256)` per coordinate); a triangle whose rounded corners are
collinear covers nothing, and one wound clockwise is turned. Each edge `a → b` gives the integer edge
function `e(p) = dx (p_y - a_y) - dy (p_x - a_x)` (`dx = b_x - a_x`, `dy = b_y - a_y`); a sample is
inside where every edge function is positive, and on an edge where its function is zero and the
edge runs up (`dy < 0`) or right along a row (`dy = 0`, `dx > 0`) (`mesh_edge_ownership@1`): a sample
on an edge two triangles share belongs to exactly one of them. The fragment's coverage is the mask of
its inside samples (bit `s` for sample `s`); a pixel with an empty mask holds no fragment.
Tests: `tests/mesh.rs::a_mesh_far_larger_than_the_frame_covers_it_without_cracks`,
`::a_meshs_coverage_is_its_exact_area_within_the_sample_patterns_bound`,
`tests/probes.rs::no_background_shows_through_a_surfaces_internal_edges_opaque_or_transparent`.

A mesh fragment's identity is `(object, part, mesh, triangle)`: `sub` is the triangle's index in its
shape, plus `t i` for an instances record's member `i` of a `t`-triangle prototype
(`instance::mesh_members`). One mesh drawn is one **surface** (`Fragment::surface`): its triangles
may merge into one unit (§9).

**Point blocks** (`point_blocks@1`, `points.rs` module doc). A `points` record's point `i`, placed
and shown, maps by `subpixel_coords` and truncates toward zero to the pixel `(X, Y)`. Its block has
side `t = trunc(1 + (radius - 1))` (none for `t < 1`); it covers whole pixels `X + o`, `Y + o'` for
`o, o'` in `floor(-t / 2) + 1 .. floor(-t / 2) + 1 + t`, clipped to the frame pixel by pixel, each
with the full mask. A block pixel's `sub` is `(offset index) * n + i` for `n` points, the offset index
being `(X + o - X - lo) * t + (Y + o' - Y - lo)` with `lo = floor(-t / 2) + 1`, so where blocks
overlap at one depth the later write in CE's offset-major order is on top
(`tests/path.rs::overlapping_blocks_keep_ces_offset_major_write_order`).

**Oracle reading.** A mesh fragment is `Yes` at a sample inside the triangle of its projected,
mapped corners, absent outside, and `Either` within `m` of an edge, where
`m = ε_in + 1/256 px`: `ε_in` is how far the input grid lets the triangle's corners move on screen
(§6, the screen shift `ε`, the larger at the edge's two corners) and `1/256` px covers the core's
corner rounding (at most `√2 / 512` px) and its edge ownership, whichever way the oracle's own
rounding goes. A triangle thinner than `2m` is `Either` wherever it covers. A point block's pixels
are `Yes` at every sample, exactly (truncation of an exact coordinate; a coordinate within `ε_in` of
an integer makes the block's edge pixels `Either`).

## 4. Path coverage

**The outline** (`path::subpaths`, `path::outline`). A path's drawn points, shown and kept in frame
units, split into CE's subpaths where a cubic's end and the next one's start differ in x or y by more
than CE's tolerance, `|a - b| <= 1e-6 + 1e-5 |b|` per coordinate meaning equal (`path::close`); a
subpath of fewer than four points is dropped. Each subpath is a move to its first point and its
cubics (anchor, handle, handle, anchor); it is closed when its first and last points are equal by
the same test, its last point then taken as its first, and a close added. The outline is mapped to
pixels by the frame mapping (`path::mapped`). A path of fewer than four points, or with a non-finite
point, draws nothing (`path::drawn_world`).

**Layers.** A path paints up to three layers, in this order, as three fragments at a pixel:
background stroke, fill, stroke (`sub = 3 i + k`, `k` 0, 1, 2, for its record's `i`-th path: an
instances record's member `i`; a face group's face is its own order entry with `i = 0`). A layer with
no paint, a solid paint of zero alpha after premultiplying, a clear gradient (every stop's alpha
short at most `0xff`), or a stroke whose width in frame units is not above 0, draws nothing
(`Lowering::layers`, `Lowering::push`, `gradient::source`).

- **Fill:** the mapped outline flattened by kurbo 0.13.1's `flatten` at tolerance `0.1` px (Cairo's
  default, `path::TOLERANCE`), into closed rings (`path::rings`: a ring of fewer than three points
  is dropped), filled by the **nonzero** rule.
- **Strokes:** the outline in frame units stroked by kurbo 0.13.1's `stroke` (§8: width, joins,
  caps, miter limit, dashes, join decisions, dots) at tolerance `0.1 / s` with
  `s = max(|xx|, |yy|)`, then mapped and flattened as the fill is, and filled by the nonzero rule
  (the stroker's outline overlaps itself at caps and inner joins; nonzero covers each point once).
- **The shared component.** The flattening and the stroker are kurbo's, at the version and
  tolerances above, for the core and the oracle alike: a second flattening differs from the core's
  outline by up to the tolerance at every curved edge, which is tenths of a cell. Cairo on 2D
  witnesses the component.
- `guard_chords@1`: in frame units, with `reach = w_max * 10` (the widest stroke's width times the
  miter limit), where some point of the outline lies within `reach` of the edge of the guard band (the
  frame widened by `65536` px on every side) or beyond it, the area is the frame's box widened by
  `reach + 1`, and each curve piece whose control hull misses that area is drawn as its chord (pieces
  split as `path::cull_pieces` splits them), before stroking and filling (`Lowering::guard`,
  `path::culled`); dashes are laid on the whole outline first (`path::dashed`). A curve and its chord
  both lie in the hull, so it changes nothing inside the area, and no pixel of the frame.

**Coverage** (`scanline.rs`). From the flattened rings, horizontal edges are dropped and every other
edge kept with its winding direction (`scanline::edges_of`). A pixel's area is the exact area of the
pixel inside the region where the winding number is nonzero, computed in binary64 as trapezoids
between transition edges within bands cut at every vertex and crossing; a sum within `1e-9` of 0 is 0
and within `1e-9` of 1 is 1 (`scanline::SNAP`); a pixel whose area is 0 holds no fragment of the layer
(`scanline::cover`). A cell's area is the same computation over the pixel magnified 4 times about its
corner, each cell a unit (`scanline::cell_areas`). Areas are stored as binary32: the pixel's
(`Coverage::Area`) and each cell's (`Producer::cell_areas`). Tests: `tests/scanline.rs` (all four),
`tests/path.rs::a_fill_covers_each_pixel_by_its_exact_area`,
`::a_self_crossing_outline_fills_by_the_nonzero_rule`,
`tests/coverage_bound.rs::a_rasterized_paths_area_lies_within_the_drift_of_its_cells_mean`
(`|area - mean of cells| <= 1e-6`, `composite::AREA_DRIFT`); the cell area's reference shape is
`cell_area` in `tests/python/test_contributor_samples.py`.

**Presence at a sample** (`cell_presence@1`). A path layer's 8-bit coverage at sample `s` is
`floor(clamp(A_s, 0, 1) * 255 + 1/2)` of its cell's binary32 area `A_s` (taken back to binary64),
or 255 at every sample when its pixel area is 1 (`TransparentPass::sample_coverage`,
`composite::area_coverage`). The fragment **covers** the sample where that byte is not 0, that is
where `A_s >= 1/510`: every order (§7), resolve (§9) and report reads that byte, never the area
(`order::shared`, `SampleRanks::rank`, `contributors.rs` module doc). A layer whose pixel area is
positive may cover no sample.

**Oracle reading.** The oracle computes each cell's exact nonzero area of the same flattened,
stroked rings by its own trapezoid-span area (written from this text, not from `scanline.rs`). A
path fragment is `Yes` at a sample where the area is at least `1/510 + μ`, absent where at most
`1/510 - μ`, `Either` between; `μ` is how far the cell's area can move when the outline moves by the
input grid's screen shift `ε` (§6): `ε` times the length of the outline inside the cell, over the
cell's area `1/16`, plus binary32's rounding (`2^-24` of the area) and the oracle's own arithmetic
bound. A fragment's cell byte is the core's formula above applied to the oracle's area, `Either`
admitting both bytes it straddles.

## 5. Depth

Each fragment carries, at each sample, a key (smaller is nearer) and an error bound (`Depths::err`,
`order.rs` module doc). What the key is depends on the record (`path.rs`, `mesh.rs`, `points.rs`
module docs).

**Keys in front of all depth.** Every fragment of a record kept or fixed in frame (§2), and every
fragment under a flat or moving camera, has key `-inf` at every sample: such fragments tie and paint
in order (`project::placement`; `tests/mesh.rs::under_a_flat_camera_meshes_paint_in_draw_order_whatever_their_z`).

**World planes** (`depth.rs` module doc, `WorldPlane`). A plane is made canonical: its normal
divided by its length, turned so its largest component (by magnitude, the lower axis on equal
magnitudes) is positive, its offset `unit · p` through its point `p` (bound `4u (|unit| · |p|)`),
then the normal's components and the offset each rounded to a multiple of `PLANE_GRID = 2^-20`
(`round(v / 2^-20) * 2^-20`); the unrounded plane is kept beside it (`Direction`, `WorldPlane::rounded`).
How a path lies is its record's XF 1.2 `planes` code for that path when carried, else decided from
its placed points (`WorldPlane::decide`): an axis plane where every point shares a coordinate within
`1e-9 * max(extent, 1)` (z, then y, then x); else Newell's normal through the first point where every
point lies on it within that tolerance; else the widest triangle's normal on the first point (an
outline crossing itself); else, for a face of a face group, the plane nearest it (Newell's normal
through the centroid, `Nearest`); else none (a curve). The decided plane is fitted to the points as
drawn (`WorldPlane::fit`). Points on a line (to within rounding and twice the snap's reach,
`depth::on_a_line`, `points_somewhere`) have no plane of their own whatever their code.

**A plane's key on screen** (`Projection::key_plane`). For a world plane `n · p = e`, with
`m = R n`, `D = e - n · c - m_2 d` (the plane's distance from the eye) and the frame map `(xx, yy,
x0, y0)`, the key across the pixel plane is `a + b x + c y` with `g_x = m_0 / (z0 xx)`,
`g_y = m_1 / (z0 yy)`, `a = -(-g_x x0 - g_y y0 - m_2 d) / D`, `b = -g_x / D`, `c = -g_y / D`, from the
rounded plane; its error bound is §6's `R`. It is not formed under the exponential option, at zoom
0, where `D` is 0 or its unrounded twin lies within its offset's bound of 0, or where the plane is
seen within `min_cos` of edge-on (`min_cos` 0 for paths, `10^-9` for mesh triangles,
`mesh::MIN_PLANE_COS`). A plane read at a point is `a + b x + c y + 0.0` (`Plane::at`).

**Mesh triangles** (`mesh::lower`). Under a 3D camera in world space, where every corner is keyed
planar: the canonical plane through the placed corners (`WorldPlane::of_triangle`; the axis plane
where the corners share a coordinate within the planarity tolerance or a grid step; none where its
normal points nowhere), keyed by `key_plane`; where that gives none, the plane on screen through the
corners' `(x, y, key)` (`Plane::through_with_errors`), as also for a triangle with a corner not keyed
planar. A fixed-orientation mesh takes its facing plane (below).

**Paths in world space** (`Lowering::coplanar`, `Lowering::world_depth`). A path with every point
keyed planar, not pre-projected and not a `Nearest` face, is offered to the frame's registry (§7):
**flat** where its planarity is Newell's, the widest triangle's or an axis plane, its points do not
lie on a line and the fit finds a plane; otherwise with **no plane of its own** (a curve, or points on
a line). Its layers are keyed:

- a flat path's **fill** on its own plane's key, however nearly edge-on (min_cos 0), where the camera
  keys that plane; else along its curve;
- a `Nearest` face's fill on its fitted plane, its bound carrying the outline's departure
  (`nearest_plane_departure@1`, §7);
- every **stroke** and background stroke, and the fill of a path with no plane of its own, along its
  curve (`curve_key@1`, below);
- a path with a point not keyed planar: every layer along its curve.

`fringe_key@1` (`path::fringe_depths`): a layer keyed on a sloped plane (`b` or `c` not 0), at a
sample outside its outline by the point-in-outline test (nonzero winding at the sample point,
half-open in y, `scanline::mask`), reads the plane at the outline's point nearest the sample
(`Layer::nearest_on_outline`), not at the sample; inside, at the sample. Tests:
`tests/path.rs::beside_a_grazing_sliver_its_fill_beyond_its_outline_is_behind_the_nearer_face`.

`curve_key@1` (`path::curve_depth`, `depth::Centreline::key_under`). A path's centreline is, per
subpath and cubic, the polyline through `n = clamp(ceil(L / 2), 1, 64)` equal steps of the cubic's
parameter (`L` the control polygon's length on screen), each vertex's position and key the
cubic's Bernstein weights `(1-t)³, 3(1-t)²t, 3(1-t)t², t³` of the control points' mapped positions and
keys (`-f` of each projected control point). At a point `P`, a layer of body `h` (half the stroke's
width in pixels, `w |xx| / 2`; 0 for a fill) takes the front-most (least) key among the segments whose
foot parameter at `P` lies in `[0, 1]` and whose distance from `P` is at most `h`; where none does,
the key of the nearest segment, clamped to its end where the foot lies past one. A segment's key at
`P` is its two ends' keys interpolated along it as a plane through the segment, held between the two
(`depth::key_on`); a segment of no length or of one key is its lesser key. Tests: `tests/centreline.rs`
(all six), `tests/probes.rs::a_curve_passing_behind_a_surface_is_hidden_there`.

`facing_plane@1` (`Projection::facing_plane`). A fixed-orientation record's fragments (paths, meshes,
point blocks: every camera-facing label) are keyed on the camera-facing plane through its centre: the
world plane of normal `R`'s third row made canonical through the centre, keyed by `key_plane` read
over the box `[-|x0|, -|y0|, 3|x0|, 3|y0|]`; where no plane keys (the exponential option, zoom 0, a
centre not keyed planar), the centre's own key everywhere, with `key_error` as its bound. A
pre-projected record with a fixed-orientation centre is keyed so in its camera's axes at its pivot.
Every projected point of a `points` record is keyed on the facing plane through it, so points and
camera-facing surfaces at one depth tie. Tests:
`tests/mesh.rs::a_fixed_orientation_mesh_is_a_camera_facing_plane_at_its_centres_depth`,
`::a_fixed_orientation_label_on_a_surface_facing_the_camera_ties_by_draw_order`,
`::points_at_one_depth_tie_by_draw_order_whether_or_not_fixed_in_orientation`.

**Glyphs.** A glyph is a path (`XF.md`): a glyph of text placed in the world is keyed as any flat path
on its own plane, and one of a camera-facing label on the label's facing plane. No rule is a glyph's
alone.

**The two stroke-depth adjustments.** They are different rules and both apply.

1. `stroke_on_own_fill@1` (`composite::strokes_on_own_fills`). For a path's stroke or background
   stroke drawn at or after its own fill (same producer, same order entry, same path: `own_layers`
   eligibility, §7, with the stroke's `sub` not below the fill's), at every sample where both cover,
   the stroke's key becomes the nearer of its own and the fill's (whatever the fill is keyed on);
   elsewhere it keeps its own. Its error bound becomes the largest, over the samples, of the bound of
   the key it took there (both bounds where the two keys tie within them, §7's `tied`), and its own
   where they do not both cover. The adjusted row is the stroke's own row (`Depths::samples`, `err`);
   rows on shared planes (below) are not adjusted. Tests: `order.rs`
   `inside_its_fill_a_stroke_is_at_least_as_near_as_its_fill`,
   `outside_its_fill_a_stroke_keeps_the_wire_s_depth`,
   `a_stroke_nearer_than_its_fill_within_their_bounds_is_not_behind_what_the_fill_is_not`.
   (A background stroke is drawn before its fill: its `sub` is lower, so it is never adjusted.)
2. `shared_plane_read@1` (`composite::Depths::against`, `Producer::plane_depths`). Against a
   fragment `b` with which it shares a registered identity (`one_plane`, §7), a path layer `a` is
   read at each sample on the plane of the first identity in `a`'s own list that `b` holds too, with
   that row's error bound, instead of its own row: a flat path's row is its own plane's key read at
   the sample wherever the sample lies (no fringe); a path with no plane of its own has a row per
   identity it lies on, that identity's plane keyed by `key_plane` (an identity whose plane the
   camera keys nowhere is dropped from its list). Against every other fragment it keeps its own row.
   Mesh triangles and point blocks have no rows: they are always read on their own. So a fragment's
   key and bound at a sample are a function of the pair. Tests:
   `composite.rs::a_stroke_is_read_on_its_path_s_plane_only_against_what_shares_that_plane`,
   `tests/path.rs::a_stroke_on_a_face_s_registered_plane_ties_it_exactly_and_meets_another_plane_by_depth`.

**A flat pixel.** Where every fragment's row (own and shared) holds one key for all at every sample
(a flat scene: every key `-inf`, or every fragment on one plane), paint order decides every pair at the
pixel whatever they cover (`resolve_pixel_traced`, `flat`).

**Oracle reading.** The oracle computes every key and row by this section's model from XF, with the
band of §6, and applies both adjustments and the pair-dependent read itself.

## 6. The tie band

The band of a fragment `f` at sample `k` is a formula of the inputs and the position:

    β_f(k) = S_f(k) + 3 R_f(k)

`R_f(k)` bounds how far any conforming computation of `f`'s key at `k` (the core's and the
oracle's alike) may stand from the exact key of §5's model of `f`'s XF inputs. `S_f(k)` bounds how far
the snap of those inputs to the view's `input_grid` may have moved that exact key from where the
scene put it. Two fragments are one depth at `k` (`TieBand`) where their oracle keys `o_a`, `o_b`
satisfy `|o_a - o_b| <= β_a(k) + β_b(k)`.

*Why `3 R`:* the core ties two keys `c_a`, `c_b` where `|c_a - c_b| <= e_a + e_b`, its own bounds,
each at most `R`; each `c` lies within `R` of the exact key and each `o` within `R` of it, so a core tie
lies within `3 (R_a + R_b)` of the oracle's keys; and beyond that the exact keys part by more than
`2 (R_a + R_b)`, so the core cannot tie them and its depth order is the exact one. `S` widens the band
by the inputs' own precision, which the core does not count (`order.rs` module doc: the snap is
common input).

Keys of `-inf` have `β = 0` and are one depth only with each other. Writing `|v|_1` for the sum of
magnitudes, `g` for `input_grid` (0 when the view states none), and the record's placements as
`[outer, inner]` (the identity for none):

**The snap's departure along a unit normal `n`**, `Δ(n)` (`depth::snap_departure`). With `C^T n`
starting at `n`, for each placement `M` that is not the identity, outer first: add
`|C^T n|_1 (g/2 + 2^-25 X_M)` where `X_M` is the largest `|x|_1` over the points `M` places (the
inner-placed points for the outer placement, the shape's points for the inner), then replace `C^T n`
by `L_M^T C^T n` (`L_M` its linear part); finally add `(g/2) |C^T n|_1` for the shape's points (when
`g > 0`; with `g = 0` no placement term is added either) and `8u max_p (|n| · (|outer| (|inner| |p|)))`
for the core's own placing (absolute values entrywise). It derives from the writer's rules: positions
and translations snap to multiples of `g` (within `g/2`), a placement's linear entries to multiples of
`2^-24` (within `2^-25`).

**What a departure moves a key by.** On a world plane `P` (unrounded normal `n_P` and offset `e_P`,
`D_P` its distance from the eye as in §5): `kr(P, δ) = δ / (|D_P| - δ)`, infinite where `|D_P| <= δ`
(`Projection::key_reach`). Along a sight axis for points at rotated depths `z_i`:
`ds(δ) = max_i δ / (d - z_i - δ)`, infinite where a denominator is not positive; under the exponential
option `δ / (d - δ)` (`project::depth_share`). On screen, for a projected point `q` whose inputs move
by `a_0`, `a_1` along the camera's first two axes and whose factor moves by share `σ`:
`ε(q) = hypot(|xx| x_0, |yy| y_0)` with `x_k = |f z0| (1 + σ) a_k + |q_k| σ`; for a point not projected
`x_k = a_k`; times `(1 + 64u)`; infinite for an infinite `σ` (`path::screen_shift`). For a record,
`ε` is the largest over its points; `a_k = Δ(R_k)` and `σ = ds(Δ(R_2))` with `R_k` the rotation's
rows (the points' own axes for a record not projected).

**`S_f(k)`, by key** (`band_input@1`):

- on a world plane `P` (a flat path's fill and its shared rows, a flat triangle, an identity row):
  `(|o| + R) kr(P, Δ(n_P) + dep_P)`, `dep_P` the record's points' largest departure from `P`
  (`depth::departure`: `|n_P · p - e_P|` plus `4u` of its terms, plus the offset's bound);
- along a curve, or on the plane through a triangle's corners: `(|o| + R) ds(Δ(R_2)) + G ε`, `G` the
  key's gradient on screen at `k` (the segment's or the plane's) and `ε` the record's screen shift;
- on a facing plane (labels, points): `(|o| + R) ds(Δ_c)` with `Δ_c = (g/2) |R_2|_1` for the centre or
  point;
- for a `Nearest` face: as on a world plane, its own fitted plane.

**`R_f(k)`, by key** (`band_arith@1`). Each term is the bound the cited constructor derives; read at
a pixel `(x, y)`, the reading's rounding `ρ(P, x, y) = 4u (|a| + |b x| + |c y|)` is taken at the
pixel's far corner `(x + 1, y + 1)` and at every point actually read (`Plane::read_error`).

- *A world plane keyed on screen* (`Projection::key_plane_error`): over the region the record is read
  in (its screen box widened by the stroke's reach and a pixel, `Lowering::region`; the triangle's
  box; the frame box above for a facing plane), the largest at its four corners `(X, Y)` of
  `gap + A + shift`, where `gap = |K_r(X, Y) - K_u(X, Y)| + 4u (terms of both reads)` is the
  difference between the keys of the rounded and the unrounded plane, read there (`PLANE_GRID`'s
  rounding, as it happened); `A = (dN + f dD) / |D| + 4u f` with
  `s = ((X - x0) / (z0 xx), (Y - y0) / (z0 yy), d)`, `t_x = g_x (X - x0)`, `t_y = g_y (Y - y0)`,
  `dn = 8u |m|`, `de = 4u (|e| + |m| |c|)`, `dD = de + dn |eye| + 4u |m_2 d|` (`eye = c + d R_2`),
  `f = |m_2 d - t_x - t_y| / |D|`, `dN = dn |s| + 4u (|t_x| + |t_y| + |m_2 d|)`, all of the unrounded
  plane; and `shift = f ε_e / (|D| - ε_e)` for the offset's bound `ε_e`. Plus `ρ`.
- *A plane through three keyed corners* (`Plane::through_with_errors`): the corners' largest bound;
  plus, with `e1`, `e2` the edges from the first corner, `det = e1x e2y - e2x e1y`,
  `g1 = d1 - d0`, `g2 = d2 - d0`: `d_det = 4u (|e1x e2y| + |e2x e1y|)`,
  `d_b = (4u (|g1 e2y| + |g2 e1y|) + |b| d_det) / |det| + u |b|`, `d_c` likewise, the spread
  `max(d_b |e1x| + d_c |e1y|, d_b |e2x| + d_c |e2y|)` and `3u (|d0| + |b x0| + |c y0|)`; a constant plane
  where the corners share one key, their mean with its spread where `det` is 0. Plus `ρ`.
- *A projected point's key* (`Projection::key_error`): `u (2 |f| + 4 f² / |d| r)` with
  `r = |z| + hypot(x, y) / |f z0| + |c|`.
- *A curve's key* (`depth::key_on`, `path::curve_depth`): its segment's ends' bound (the largest of
  its cubic's control points' `key_error`, plus `4u` of their largest key), plus
  `4u |k_b - k_a| (1 + t) + 3u (|k_a| + |b_x a_0| + |b_y a_1|) + ρ` at the point (`t` its distance
  along the segment over the segment's length); for a segment of no length or one key, `|k_a - k_b|`
  more; the front-most of several, the largest of their bounds; a pixel's samples, the largest over
  its samples (`Centreline::keys_among`).
- *A `Nearest` face*: its plane's bound plus its outline's departure (`nearest_plane_departure@1`,
  §7).
- *A stroke adjusted on its own fill* (§5): as `stroke_on_own_fill@1` states.

**Contract test.** On every probe fixture, at every pixel and sample where the core reports a
fragment, its `Depths::err` for its own row and for every pair-dependent row is at most `R_f(k)`
(so at most `β_f(k)`), and its key lies within `2 R_f(k)` of the oracle's. An excess is a finding
about the core, listed by class; the band is never widened to fit it.

## 7. The order at a sample: tie policies

**The relation of a pair at a sample.** For two fragments `a`, `b` both covering sample `k`, read
at keys `d_a`, `d_b` with bounds `e_a`, `e_b` (each read against the other, §5), the core's order is
**weak** (paint order decides) where the pair is `own_layers`, where `tied(d_a, e_a, d_b, e_b)`, or
where `coplanar_tied` holds; otherwise **strong**, the nearer in front (`order::at_sample`,
`strong_at`). `tied(a, ea, b, eb)` is `a == b`, or both finite and `|a - b| <= ea + eb`; keys that are
not finite tie only with themselves, tested before any subtraction (`order::tied`;
`order.rs::depths_that_are_not_finite_tie_only_with_themselves_before_any_subtraction`). At one
sample the strong orders are strict orders of finite keys and do not cycle on one read, but a
fragment read two ways (`shared_plane_read@1`) can close a cycle
(`order.rs::at_one_sub_sample_the_strong_orders_never_cycle`,
`a_stroke_read_two_ways_settles_a_cycle_at_one_sub_sample_by_wins_then_paint_order`).

**`paint_order@1`** (`order::painted`). Of two fragments, the one later in the order is in front;
on one order entry, the higher `sub`. Every weak order is paint order.

**`own_layers@1`** (`order::own_layers`). A path's fill and one of its own strokes or background
strokes (same producer, same order entry, same path `sub div 3`, different layers, one of them the
fill) keep paint order whatever their keys: the stroke over its fill, the background stroke under
it. Two strokes of one path keep depth. Tests:
`order.rs::a_path_s_fill_and_its_own_strokes_keep_paint_order_and_two_strokes_keep_depth`,
`tests/path.rs::a_grazing_face_s_stroke_is_over_its_own_fill_wherever_the_stroke_covers_a_pixel`.

**`one_plane@1`** (`order::one_plane`, `order::coplanar_tied`, `order::within_term`). The frame's
**registry** (`depth::Registry`, its doc and `settle`) gives every registry-offered record (§5:
paths in world space keyed planar, and mesh triangles keyed on a world plane) up to four identities
(`depth::MOST_IDENTITIES`):

- a **flat** record's tolerance along a unit normal `n` is `τ(n) = Δ(n) + dep` (`dep` its points'
  departure from its own plane); in a canonical order of the flat records (least `spread` first,
  `depth::canonical`), each joins the first identity, among those whose planes lie in the lookup
  buckets around its own (`NORMAL_BUCKET = 1/64` of normal component, `OFFSET_BUCKET = 0.5` of
  offset, at most `MOST_REACH = 2` buckets past its own on each side), for which one plane holds every
  member's points and its own each within its own tolerance (the identity's plane, or the minimax
  plane of them all, `Registry::fit`), else founds one; an identity's plane is its founder's until a
  record joins that it does not hold, then that minimax plane;
- a record with **no plane of its own** attaches, once the partition is made, to every identity
  whose plane holds the members' points and its own within their tolerances, which its points stand
  off by no more than `Reach::bound` (within the members' hull `t + members + (turn + slope)(w +
  widest)`, growing by `slope` per unit of distance from it), and whose members' hull it enters
  (`Registry::contains`, `Registry::enters`), the first four of them; one that attaches to none
  shares an identity with the others on exactly each axis plane it lies on (the one its writer coded,
  and for a path every coordinate all its points hold exactly), with no tolerance.

Two fragments are on **one plane** where their identity lists meet. There they tie (weak) where
`within_term`: with each one's share `s` (its `wide` where both carry one, else its `snap`),
`tied(d_a, e_a + (|d_a| + e_a) s_a, d_b, e_b + (|d_b| + e_b) s_b)`, or where either share is infinite.
The shares (`Fragment::snap`, `Fragment::wide`): a flat path's layers and a flat triangle,
`kr(P, Δ(n_P) + dep_P)` on its own plane (0 for a flat path content did not make flat); a record with
no plane of its own attached to flat identities, `snap` the largest over its identities of
`kr(I, off_members)` and `wide` the largest of `kr(I, off_identity)` (`Registry::off_members`,
`off_identity`: its tolerance along the identity's normal plus its points' departure from its plane,
and at most the bound within the members' hull); one on axis planes alone, the largest over the
axis planes keyed of `kr(A, Δ(n_A) + dep_A)`. Tests: `order.rs::on_one_plane_a_zero_share_parts_beyond_the_errors_and_an_unbounded_one_ties`,
`a_record_with_no_plane_of_its_own_takes_its_whole_share_only_against_another`,
`tests/path.rs::a_path_on_a_meshs_plane_ties_with_it_by_draw_order`,
`::a_line_on_a_tilted_plane_ties_with_it_by_draw_order`,
`tests/mesh.rs::coplanar_meshes_tie_by_draw_order_the_later_on_top`, `depth.rs`
`each_member_is_held_within_its_own_tolerance`, `the_partition_does_not_depend_on_the_order_offered`.

**`near_parallel@1`** (`order::near_parallel`, `depth::near_parallel`, `depth::tested`). Two
fragments of flat records (each carrying its registry record, which a flat layer keyed with a finite
share and a flat triangle do) on no one identity tie within `within_term` where their own planes'
dihedral is within what each record's snap lets its plane turn: with `a` the unit axis
`n_A × n_B / |n_A × n_B|`, `θ = atan2(|n_A × n_B|, |n_A · n_B|)`, and for each record
`θ_R = 2 τ_R / L_R`, `τ_R` its tolerance along its own normal and `L_R` its points' extent along
`n_R × a`, they are near-parallel where `θ <= θ_A + θ_B`, every rounding taken against the test (the
dihedral rounded up by `16u` of itself and `16u`, each `θ_R` and the sum down by `2u` of themselves,
the extent widened by the axis's uncertainty `40u / s + 16u` at the farthest point, or twice the
farthest distance where `s < 32u`). Pairwise and symmetric; it gives no identity and does not chain.
Tests: `depth.rs::faces_are_near_parallel_up_to_the_turn_their_snaps_allow_and_not_past_it`,
`the_near_parallel_test_is_the_same_whichever_record_comes_first`,
`a_record_with_no_plane_of_its_own_is_never_tested_for_near_parallel`.

**`lines@1`** (`order::lines`, `order::lines_term`). Two strokes (or background strokes) of paths
with no plane of their own, under a 3D camera, on no one identity, tie where
`tied(d_a, e_a + (|d_a| + e_a) o_a, d_b, e_b + (|d_b| + e_b) o_b)` with each one's own share `o`
(`Fragment::own`): its path's `ds` of the snap's departure along the axis its depth is read along (the
camera's sight axis for a projected path, the points' own z otherwise), plus, at the pixel, how far a
read may slide along the curve and between its pieces where the snap moves the curve on screen by its
shift, as a share of the least read (`Paths::own_at`, `Centreline::slides_among`); infinite where the
least read is not positive. `Stream.own_shares` reports each such fragment's share. Fills never carry
one.

`coplanar_tied` is `one_plane && within_term`, else `lines && lines_term`, else (both of flat
records) `within_term && near_parallel`: the three are disjoint.

**`nearest_plane_departure@1`** (`path::nearest_plane_departure`). A face of a face group on no one
plane (`Nearest`) is keyed on the plane nearest it and its bound carries how far that plane's key
stands from its drawn outline's keys wherever its fill is read: the largest over its points (and a
closed subpath's closing point) of `|key at the point - its own key|` plus each read's rounding, plus
`hypot(b, c) * 0.1` for the flattening. So it ties with what that plane cannot tell it from. It is
not offered to the registry.

**One order for the pixel** (`majority@1`, `order::pair`, `Shared::decided`). Each pair's order at
the pixel comes from the samples both cover: none shared, or a fill against its own stroke: paint
order (`Paint`); every shared sample tied (`tied` or `coplanar_tied`): paint order (`Tie`); every
untied shared sample with the same nearer: that one (`Depth`); the untied samples disagreeing: the one
nearer at more of them, paint order on an even split (`Majority`). `Depth` and `Majority` are strong,
`Paint` and `Tie` weak. This order is read by the opaque drop, the units and the coverage resolve
(§9), never by a sample's own order. Tests:
`order.rs::a_pair_crossing_inside_the_pixel_takes_the_order_most_shared_sub_samples_give`,
`a_stroke_one_ulp_nearer_where_it_does_not_cover_keeps_paint_order`.

**`cycle_rank@1`, the rank** (`order::rank`, `rank_sample`, `rank_through_cycles`). Pairwise orders
are edges, strong or weak. Where they form one order (inserting each fragment before the first it
is in front of yields an order every pair agrees with), that order. Otherwise, front first: among the
fragments no remaining fragment is strongly in front of, the later painted; and where every remaining
fragment has a strong edge into it (the strong edges alone cycle), at a sample the one in front of the
most remaining fragments, the later painted on ties. So weak edges yield before any strong edge is
broken: a paint-order or `own_layers` edge gives way first. At the pixel a strong cycle is not broken:
the pixel takes the sample resolve (§9). Tests: `order.rs::a_cycle_through_a_weak_order_keeps_every_strong_order`,
`the_report_breaks_a_cycle_of_strong_orders_by_wins_then_paint_order`,
`a_pixel_whose_strong_orders_cycle_composites_each_sub_sample_in_its_own_order`.

**Oracle reading: the relation.** For each pair of fragments present at `k` (each `Yes`, or `Either`
and present in the core's report), the oracle gives the first of these that holds:

1. `Policy(own_layers@1)`, exactly by the eligibility above: paint order only (a weak edge).
2. `Policy(one_plane@1)`, `Policy(lines@1)` or `Policy(near_parallel@1)`, in the core's order of the
   three, where the pair is eligible and its term holds with margin:
   `|o_a - o_b| <= T_ab + 3 (R_a + R_b)`, `T_ab` the term's right side at the oracle's keys and
   shares. Eligibility for `one_plane@1` is pairwise: both records registry-offered and one plane holds
   every point of both within each one's tolerance (for two records with no plane of their own, also
   where both lie exactly on one axis plane). Every pair the core's partition puts on one identity
   passes it (each member lies within its tolerance on its identity's final plane; an attached record
   is held with the members by one plane), so it admits a superset; the shares of a record with no
   plane of its own are taken on that plane with both records' tolerances. `near_parallel@1`'s test is
   evaluated as stated, its roundings taken in its favour; `lines@1` is eligible exactly as stated.
   Both orders are admissible.
3. `TieBand` where `|o_a - o_b| <= β_a + β_b`: both orders admissible.
4. Otherwise `Depth(front)`, the nearer: a strong edge.

**Oracle reading: the admissible orders at a sample.** A **completion** of the pairs present at `k`
gives each `Depth(front)` pair a strong edge to `front`, each `own_layers@1` pair a weak paint-order
edge, and each other pair one of: a strong edge either way, or a weak paint-order edge. An order of
the present fragments is **admissible** where the rank (`cycle_rank@1`, through cycles) of some
completion produces it. Where the `Depth` and `own_layers` edges alone are acyclic, every order that
keeps all of them is admissible (complete each other pair strongly in that order's direction), which
a comparator may test first; other orders may be admissible too, through a completion that makes a
weak edge yield. The check is over every pair, never adjacent pairs alone: bands are not transitive
(keys 0, 1, 2 with bands 0.6 tie the adjacent pairs while the ends are `Depth`). A sample's order that
no completion produces is a `mismatch`.

## 8. Stroke policies

A path's strokes are kurbo 0.13.1's `stroke` of its outline in frame units at tolerance `0.1 / s`,
then mapped (`Lowering::path`).

- `width_in_scene_units@1`: the width is CE's stroke width (the record's, or a face's own, XF 1.3;
  a paint blend's mix, `geometry::drawn_paint`) times the view's `cairo_line_width_multiple`, in frame
  units, as Cairo strokes in user space: the frame mapping scales it, the perspective never does. A
  width not above 0 (or NaN) draws nothing (`tests/path.rs::a_stroke_covers_its_outline_at_ces_width`).
- `miter_limit_10@1`: joints auto and miter are kurbo's miter, round and bevel its own; caps auto and
  butt are butt, round and square their own; the miter limit is Cairo's 10 (`path::MITER_LIMIT`,
  `kurbo_join`, `kurbo_cap`; `tests/path.rs::a_sharp_joint_is_mitred_up_to_cairos_limit`).
- `join_mask@1` (`join_mask.rs` module doc, `path::decided_joins`, `join_mask::apply`): a path record
  carrying XF 1.4 `joins` (one byte per cubic of its points: bit 0 Cairo joins into the cubic, bit 1
  it mitres there; a closed subpath's first cubic's byte its closing join) has kurbo's stroke edited at
  each join whose carried decision differs from kurbo's own: a carried bevel where kurbo mitred drops
  the miter's two vertices (leaving kurbo's bevel); a carried miter where kurbo bevelled inserts them,
  the tip on the bisector at the miter limit's length; where kurbo's tangents are parallel the bevel
  stays; a join whose vertices are not found in kurbo's outline, a join the record does not decide
  (bit 0 clear, no byte), and a cubic whose points are all one (skipped: the join is from the cubic
  before) are left as kurbo drew them. Joins are matched by their two cubics' points' bits, one
  decision per occurrence. Tests: `tests/join_mask.rs` (all eight).
- `dash_start_least_length@1` (`path::drawn_dash`, `path::dashed`): a dash of duty 1, or whose gap
  (the entries' total less the on length) is under Cairo's least length `1/512`, strokes solid; any
  other is the pattern `[period * duty, period * (1 - duty)]` laid by kurbo's `dash` on the whole
  outline (each subpath starting `phase * period` into it; a closed subpath's last dash joined to its
  first where both are on; pieces beyond the guard area measured, not dashed), then stroked. Tests:
  `tests/path.rs::a_dashed_stroke_follows_its_period_duty_and_phase`,
  `::a_dashed_stroke_returning_from_far_beyond_the_frame_keeps_its_dash_phase`.
- `round_cap_dot@1` (`path::strokes_as_dot`): a face of a face group whose shown points are all at one
  spot in x and y, under a round cap, undashed or dashed on at its subpath's start (Cairo's dash start:
  the offset `phase * period` taken modulo the total and stepped past every entry it reaches or
  passes), strokes as a circle of radius `w / 2` about its first point, flattened at `0.1 / s`.
  Under a butt or square cap it strokes nothing.

## 9. The resolve text and colour

At a pixel the fragments are the opaque pass's (fragments whose colour is opaque wherever they
paint) then the others, each pass in producer order (meshes, paths, point blocks) and within a
producer in the order added. `opaque` means alpha 255; **full** means a mask of all 16 samples or an
area of 1; a fragment's 8-bit **coverage** is `(bits * 255 + 8) div 16` for a mask
(`mask_coverage`) and `floor(clamp(area, 0, 1) * 255 + 1/2)` of its binary32 area for a path
(`area_coverage`), never the sum of its cell bytes; its coverage at sample `k` is 255 or 0 for a
mask, and §4's cell byte for a path (`sample_coverage`).

**`resolve_select@1`** (`composite::resolve_pixel_traced`):

1. No fragment: the background (`Background`).
2. One fragment: opaque and full, its colour (`Fast`); otherwise `blend(colour, coverage, bg)`
   (`Coverage`).
3. Several: every fragment's keys (§5, both adjustments) and every pair's pixel order
   (`majority@1`, or paint order on a flat pixel); the pixel's rank (`cycle_rank@1`). Where the strong
   orders alone cycle: the **sample resolve** over every fragment.
4. **`opaque_drop@1`**: where the pixel's order holds a fragment opaque and full, at place `p`: on a
   flat pixel, everything after the first such is dropped; otherwise each sample ranks the fragments
   covering it (`rank_sample`) and a fragment is kept where some sample's rank reaches it before or
   at the first opaque, full fragment there, or where it covers no sample and stands before place `p`
   in the pixel's order; the rest are dropped. A single kept fragment that is opaque and full: `Fast`.
   Tests: `order.rs::the_cull_keeps_what_the_rank_shows_and_drops_what_it_hides`,
   `only_a_layer_covering_the_pixel_whole_hides_what_is_behind_it`.
5. **`unit_merge@1`**: runs, in the pixel's order, of one surface's mask fragments (same producer and
   surface) whose masks do not overlap form one unit; every other fragment is a unit alone. Where a
   unit of two or more fragments is opaque and its masks' union is full, every later unit is dropped
   where no sample's rank shows any fragment of it before an opaque, full fragment or a member of that
   unit (flat: always).
6. Count the units whose mask union is partial; if any is, or any path unit's area is below 1, add the
   full mask units whose members differ in colour. The **coverage resolve** applies where that count
   is 0, or 1 with no partial path area, and at every sample no fragment of a unit is behind (at that
   sample's own relation, §7) a fragment of a later unit where both cover it. Otherwise the **sample
   resolve**. A 2D scene never takes it (`tests/resolve_switch.rs::a_2d_scene_never_takes_the_sample_resolve`).

**The coverage resolve.** Back to front over the units in the pixel's order, `px = blend(colour,
coverage, px)` from the background. A unit of one fragment blends with that fragment's colour and
coverage (a path's whole-pixel byte). A merged unit's colour is, per channel,
`(Σ c_i n_i + N div 2) div N` over its members, `n_i` each one's mask's bit count and `N` their sum, and
its coverage `mask_coverage` of their union.

**The sample resolve.** Each sample `k` composites, back to front from the background, the fragments
its rank orders at `k` (the ranks the drops above read where they ranked, else ranked now over the
kept fragments), each at its coverage there; the pixel is each channel's `(Σ_k + 8) div 16`. A
fragment the drop removed but a rank kept lies behind an opaque, full cover at every sample, which
blends over it exactly.

**Blend arithmetic** (`composite::blend`, Cairo's ARGB32 image surface): with
`mul(a, b) = ((t >> 8) + t) >> 8`, `t = a b + 0x80`, and `mul_7f(a, b) = (t + (t >> 8)) >> 8`,
`t = a b + 0x7f`: an opaque source at coverage 255 replaces the pixel; an opaque source at
coverage `c` gives `min(mul_7f(s, c) + mul_7f(d, 255 - c), 255)` per channel; any other source is
first scaled, `s' = mul(s, c)` per channel when `c < 255`, then laid over,
`min(s' + mul(d, 255 - s'_alpha), 255)`. Tests:
`tests/composite.rs::two_d_paint_blends_with_cairos_argb32_arithmetic`,
`composite.rs::the_byte_products_round_within_their_stated_errors`. The coverage resolve lands within
`composite::coverage_to_sample_bound` (`COVERAGE_TO_SAMPLE_LSB = 3` for one partial mask) of the
sample resolve, a property, not a rule the recompute uses
(`tests/coverage_bound.rs`).

**Colour, from XF** (each fragment's premultiplied bytes):

- `cairo_premultiply@1` (`composite::premultiply`): a straight colour, alpha times the record's
  group opacity, each channel clamped to `[0, 1]`; channel `k` is `((c_k a) * 65535 + 1/2)`
  truncated, shifted right 8, and alpha likewise.
- A path's solid paint: its one row, through `cairo_premultiply@1`. A paint of two or more rows is a
  gradient between the record's ends shown for display (§2).
- `cairo_linear_gradient@1` (`gradient.rs` module doc and `source`, `Linear::at`): Cairo's pattern
  (stops at `np.linspace(0, 1, n)`, colours clamped and kept as 16-bit shorts `(v * 65535 + 1/2)`
  truncated; all stops one colour, or ends within `f64::EPSILON` in both coordinates, is a solid:
  the stop's, or the first and last stops' average) under the inverse of the context matrix (Cairo's
  `cairo_matrix_invert`), handed to pixman as Cairo does (16.16 fixed point by Cairo's magic-number
  rounding; the matrix's translation split per `_cairo_matrix_to_pixman_matrix_offset` about the
  middle of the composite's extents, the outline's box in 24.8 fixed point rounded out and cut to the
  frame); each pixel's parameter taken at the start of its **composite run** (Cairo's in-place span
  renderer, `RUN_LENGTH = 256`: a full span of at least 256 pixels on its own, the rest gathered from
  the first pixel and cut at a gap once longer than 256, `path::composite_runs`, from the layer's own
  coverage in its box row) and stepped `t + trunc(inc * i)` to the pixel; colour by pixman's pad
  walker in binary32, premultiplied, rounded to bytes. Tests: `tests/gradient.rs` (all five).
- `ce_shading@1` (`shade.rs`): a `shade_in_3d` path under a 3D camera that shades is lit, per layer,
  into two stops at its corners `0` and `((n - 1) div 6) 3` by CE's `get_shaded_rgb` (`to_sun` the unit
  vector to the light, `δ = 0.5 (n · to_sun)³`, halved where negative, added to the encoded colour,
  unclamped) with the normals its record carries (XF 1.2; a zero normal lit by the face's Newell
  normal, or CE's where that cancels) or CE's own (`get_3d_vmob_unit_normal`), drawn as a gradient
  between its recorded ends, else its corners. A lit mesh triangle (`Light::triangle`): each corner lit
  (CE's model: clamped encoded colour; soft diffuse, light model 1: `ambient + (1 - ambient) *
  max((n · l + wrap) / (1 + wrap), 0)` on the linear colour, the normal turned toward the eye),
  interpolated as planes through the three corners on the pixel plane, read at the centroid of the
  samples the fragment covers, clamped, encoded where linear, then `cairo_premultiply@1`.
- A point block (`points.rs`): bytes `trunc(255 c)` per channel and alpha (times the group
  opacity); none where the alpha byte is 0; where it is below 255, each channel `(c a + 127) div 255`.

Nothing a frame of this engine draws emits light (emission records are refused, §11), so the
emission step (`composite::add_emission`) never applies.

**Oracle reading.** The recompute reads, per pixel, the core's per-sample sequences after its drops,
each fragment's colour, whole-pixel coverage byte and cell bytes, selects the resolve by this
section, and composites by this arithmetic, never through the core's resolve and never from its
resolved colours. With the oracle's colours it computes them by the rules above from XF.

## 10. The interim label rule

**Core rule** (`facing_plane@1`, §5). A camera-facing label (a record with a fixed-orientation
centre under a 3D camera, or a pre-projected record keyed at its pivot) is the **plate model**: every
fragment of it lies on the camera-facing plane through its centre, and depth, ties and policies order
it like any other fragment, so something nearer can cut it.

**The whole-label rule** (to be drawn by a later version of the label rule): a camera-facing label is
drawn whole. It goes under a filled surface that is nearer than the label's centre, and otherwise
stays on top of everything, lines and arrows included; labels among themselves keep paint order.
Line3D, Arrow3D, Vector and cylinder shafts count as lines and arrows; surfaces, solids and filled
planes or polygons count as surfaces.

**The open label cases** (the rule above does not yet settle them):

1. **Partial overlap away from the centre:** a filled surface nearer than the label at some of the
   label's samples but not at its centre, or nearer at its centre but not at a sample: whether the
   surface's depth is compared with the label's centre at the sample or at the centre's point.
2. **Translucent surfaces:** whether a nearer surface that is not opaque counts as a surface the
   label goes under.
3. **What counts as a surface:** lines, arrows and cylinder shafts against surfaces, solids and
   filled planes. XF does not say which a record is: a mesh triangle, a face group's face or a path's
   fill may be a shaft or an arrow's tip as well as a surface.

**`label_interim@1`.** At a sample covered by a fragment `L` of a camera-facing label and a fragment
`G` of another leaf, the oracle forms `P`, the orders admissible under the plate model (§7), and `W`,
the orders the whole-label rule gives under every reading of the open cases:

- `G` of another camera-facing label: paint order;
- `G` a stroke or background stroke, or a point block: `L` in front;
- `G` a fill (a path's fill, a face, a mesh triangle): `L` in front where `G` is read as no surface
  (case 3); read as a surface, `G` in front where it is nearer than `L`'s centre key, compared at the
  sample and at the centre's point (case 1, where `G`'s key there is defined), and `L` in front where
  it is not; read as translucent (`G`'s alpha below 255) both counting and not (case 2).

Where `P` and `W` are each one order and the same order, the pair's relation stands as §7 gives it.
Otherwise both orders are admissible at that sample and the sample counts `label_rule_open`. The
rule ends when the open cases are settled and the whole-label rule is drawn; its replacement is a new
version.

## 11. What the oracle refuses

**Core rule** (`stream::refusal`, `stream::field_refusal`; Python `render.unsupported`). The core
refuses a frame whole, drawing nothing, when an entry of the order it draws (overlay entries excluded,
§1) holds: a feature bit outside `KNOWN_FEATURES` (bits 0 and 1); an unknown record kind; a reserved
kind (volume 64, field 65, emission 66); an `image` record; a clip; a blend, reveal or paint blend on
any kind but a path, or on a pre-projected path; a paint blend of row counts numpy does not
broadcast (`geometry::drawn_paint`); a mesh (or mesh instance) with a stroke, a gradient
fill or a dash; a shaded path under a light model other than CE's; a gradient paint without ends on an
unshaded path, or on an instances record; a dash whose period is not finite and above 0, whose duty
is not in `(0, 1]` or whose phase is not finite; a path blend of shapes of different lengths; a face
group whose points do not divide into its faces; an instances record of more than `2^32` parts; a
points record of more than `2^32` block writes, or a pre-projected one placing other than one point
per colour. It also refuses a view whose light model is unknown. A record outside the order is not
drawn and not refused. Tests: `tests/mesh.rs::a_mesh_carrying_what_this_build_does_not_draw_is_refused_whole`,
`tests/path.rs::what_paths_and_points_do_not_draw_yet_is_refused_whole`.

**Oracle reading.** The oracle answers `truth_unable` for the whole frame, never drawing it partly,
when the frame's order (overlay entries excluded) holds a reserved kind (volume, field, emission), a
kind or a feature bit it does not know, or the view names a light model it does not know (it knows
CE's, code 0, and soft diffuse, code 1); and wherever the core refuses the frame (there is nothing to
judge). It reads deltas of XF major 1 up to minor 4 and refuses a delta of a newer minor, whose
skipped fields it cannot know the meaning of. Each refusal names the slot, the object and the kind or
bit.
