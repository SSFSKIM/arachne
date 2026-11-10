# The truth contract

Version 1, written against the core at commit `3ce63425` (XF 1.4). The oracle's reading of the
wire is stated through XF 1.5.

This contract defines what it means for Arachne's CPU core to have **decided right** when it draws
a frame from an XF stream. At each of the 16 sub-samples of every pixel, a decision is which
fragments cover the sub-sample and in what order they composite. The contract also defines how
those decisions become the pixel's four bytes.

Three things are written from this text alone, never from the core's source:

- **The oracle** computes, from the XF stream, the set of decisions it admits.
- **The recompute** is a second implementation of the compositing arithmetic. It turns decisions
  and colours into bytes.
- **The contract tests** hold the core to the text.

## How to read it

### Paragraph kinds and citations

- **Core rule** paragraphs state what the core does. Each is true of the core at `3ce63425`. The
  bracketed **Rests on** lists after them cite the module, function and tests that show it (paths
  under `core/`).
- **Oracle reading** paragraphs state how a decision is judged: what the oracle computes from XF,
  and which of the core's decisions it admits.
- A definition never points at the core's code. Every rule the oracle needs is stated here in words
  and formulas, or pinned to one of the external definitions listed below, with each of the core's
  deviations from it stated. A citation is evidence that the core keeps a rule. It is never where
  the rule lives.

### What the oracle may admit

**Judged from the inputs.** The oracle judges each of the core's decisions by what the inputs
admit, never by a class some other comparison assigns to a difference from a reference render. A
difference such a comparison would call immaterial is still a `mismatch` where the core's decision
is not admissible, and a difference is no finding where the decision is.

Where the inputs or the arithmetic leave a decision open, the oracle admits a **conservative set**
of outcomes rather than transcribing the core. Every such set carries two proofs, and both are
tested:

- **Soundness.** The core's own decisions fall inside the set. The contract tests check this.
- **Discrimination.** A named injected defect falls outside the set and gets its named verdict,
  `mismatch` unless §12 names another with its reason; never a tie and never `Either`. Each set
  names its defect beside it (**Rejects:**), and §12 collects them with their witness fixtures.

A class of decisions that no set narrow enough to discriminate can contain is named in §12 as
outside what the oracle promises. Its samples are reported by class for a human to view. They are
never counted as ties.

**When the contract is proven.** A bound in this text is a claim until the executable proof
passes, and the proof has two parts:

1. every witness fixture of §12 yields its named verdict against the oracle (each mutation's
   verdict, and agreement on each soundness fixture);
2. on the real corpus, the core's reported error, shares and decisions never leave the band and
   sets this text gives, with each violation triaged by name as a contract defect (the text is
   wrong, and is corrected by a new version) or a core defect (the core is wrong).

Until both pass, the rule that a mismatch of any size fails does not bind: an excess or a mismatch
is evidence to triage, not yet a verdict on the core. Once both pass, the sets are frozen at their
versions, an excess is a finding about the core listed by class, and no set is widened to fit
it.

### Names and versions

- Every rule beyond plain depth carries a name and a version, `name@1`.
- The rules counted per sub-sample are marked ✓ in the table below. Their counts are the
  `per_policy` counts.
- A change to what the core does is a new version of the rule it moves, never an edit in place.
- A change to a rule's text bumps its version, and every verdict it moves is replayed and
  reported.

### What the oracle writes

- **Per fragment, per sub-sample:** `Yes`, `Either`, absent, or `truth_unable` (§6, an unbounded
  key). For a path, it also writes the interval of cell bytes it admits.
- **Per pair of present fragments, per sub-sample:** a relation label (`Depth(front)`, `TieBand`,
  `ExactTie`, or `Policy(name@1)`) and the set of edges it admits (§7).
- **Per frame:** `truth_unable` when it refuses the whole frame (§11).
- **Shares:** the share of samples left in a tie, per label. This measures what conservatism costs.

### Numbers

- `u = 2^-53` is the unit roundoff of IEEE binary64.
- Arithmetic is binary64, evaluated left to right in the order the formulas give, with no fused
  multiply-add.
- `round(v)` rounds half away from zero.
- `trunc` truncates toward zero.
- `|v|_1` is a vector's sum of magnitudes.
- A depth **key** is the projection's perspective factor negated, so a smaller key is nearer.
- `−∞` is a key, the key of what lies in front of all depth.

### Shared components and pinned external definitions

The oracle shares exactly these with the core. Each is named with its version:

- **The XF reader**, `arachne_core::xf`. It decodes the stream, places points (`place`) and
  composes placements (`compose`). It is the one core path the oracle uses.
- **kurbo 0.13.1.** It supplies `flatten`, `stroke` with `StrokeOpts::default()`, `dash`,
  `CubicBez::subdivide`, `CubicBez::bounding_box`, `ParamCurveArclen::arclen` and
  `Circle::to_path`. A path's outline is flattened and stroked by kurbo for both the core and the
  oracle. A second flattening would differ from the core's by up to the tolerance at every curved
  edge, so the flattening is part of the definition.
- **libm 0.2.16.** It supplies `sin`, `cos`, `exp`, `pow`, `atan2` and `hypot`. Square roots are
  IEEE `sqrt`. It supplies `hypot` for §8's join test too: kurbo 0.13.1 is built without `std`, so
  its float functions are libm's.

The pinned external definition is **Cairo 1.18's ARGB32 image surface, with pixman 0.42 computing
its colours**. It defines a linear gradient's bytes (§9), with the core's deviations stated there.
Elsewhere, Cairo's rules that this text relies on are restated in full.

### The named rules

| Rule | § | What it decides | Counted |
|---|---|---|---|
| `alpha_round@1` | 1 | an op whose alpha rounds to zero paints nothing | |
| `mesh_fixed_point@1` | 3 | triangle corners rounded to 1/256 px before the sample test | |
| `mesh_edge_ownership@1` | 3 | a sample on an edge belongs to one triangle | |
| `point_blocks@1` | 3 | a point is CE's square block of whole pixels | |
| `cell_presence@1` | 4 | a path fragment covers a sample where its cell byte is not 0 | |
| `guard_chords@1` | 4 | curves far beyond the frame drawn as chords | |
| `curve_key@1` | 5 | the key along a curve's centreline | |
| `fringe_key@1` | 5 | a sloped plane-keyed layer read at its outline beyond the outline | |
| `facing_plane@1` | 5 | camera-facing records and points keyed on the plane facing the camera | |
| `stroke_on_own_fill@1` | 5 | a stroke at least as near as its own fill where both cover | |
| `shared_plane_read@1` | 5 | a layer read on a shared identity's plane against what shares it | |
| `band_input@1` | 6 | how far the input grid's snap may move a key (`S`) | |
| `band_arith@1` | 6 | how far a conforming computation may stand from the exact key (`R`) | |
| `own_layers@1` | 7 | a fill against its own strokes keeps paint order | ✓ |
| `one_plane@1` | 7 | two fragments on one registered plane tie within the snap's term | ✓ |
| `near_parallel@1` | 7 | two flat records whose input cannot fix their relative tilt tie | ✓ |
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
| `paint_rows@1` | 9 | which colour rows a layer paints with, and how a paint blend mixes them | |
| `cairo_linear_gradient@1` | 9 | a gradient's bytes at a pixel | |
| `ce_shading@1` | 9 | CE's two lit stops of a shaded path; lit mesh corners | |
| `label_placement@2` | 10 | where a camera-facing label sits among what covers it (replaces @1; the change is stated in §10) | `label_rule_open`, by cause (§10) |
| `label_classes@2` | 10 | which sources are lines and which are surfaces (replaces @1; the change is stated in §10) | `label_rule_open`, by cause (§10) |

How the counted rules are counted:

- **Stroke policies:** the sub-samples covered by a layer whose outline the policy changed (a join
  edited, a dash laid, a dot drawn).
- **`paint_order@1`:** the sub-samples at which a pair's label is `TieBand` or `ExactTie`.
- **`nearest_plane_departure@1`:** those at which the label is `TieBand` only by that term.
- **`cycle_rank@1`:** the sub-samples whose admissible orders include one that only a cycle
  produces.

## 1. The frame and its pixels

**Core rule.**

- **Layout.** A frame of `W x H` pixels is `H * W * 4` bytes, row by row. Each pixel is R, G, B, A,
  premultiplied as Cairo stores it. Pixel `(x, y)` spans `[x, x+1) x [y, y+1)` in pixel
  coordinates, with `y` pointing down.
- **Background.** The frame starts as the background, in one of two forms:
  - a colour written as CE writes it, each channel `trunc(c * 255)` wrapped to a byte and not
    premultiplied;
  - an image's top-left `W x H` corner. An image smaller than the frame is refused.

  Each pixel that holds a fragment is then resolved over the background (§9).
- **The overlay.** The frame judged is the one the core draws without its overlay, which Python
  calls with `draw(overlay=False)`. Three kinds of record are Cairo's, drawn over the core's frame
  in CE's order:
  - records in overlay space;
  - `image` records;
  - records with feature bit 0.

  Their order entries are neither drawn nor refused. Every other entry keeps its index in the
  order, which is its place in paint order (§7).

[Rests on: `composite.rs` module doc and `background_pixel`; `Frame::cropped`;
`Stream::render_under_overlay`; `stream::is_overlay`.]

**Sample positions.** Sub-sample `s` (0..16) of pixel `(x, y)` lies at

    (x + 1/2 + (dx_s + 1/2) / 16,  y + 1/2 + (dy_s + 1/2) / 16)

where `[dx_s, dy_s]` is the offset below, in sixteenths of a pixel from the pixel's centre:

    s:   0      1      2      3      4      5      6      7
        (-8,-7) (-1,-8) (3,-6)  (5,-5)  (-6,-1) (-4,-4) (0,-3)  (6,-2)
    s:   8      9      10     11     12     13     14     15
        (-7,2)  (-2,0)  (2,3)   (4,1)   (-5,5)  (-3,4)  (1,7)   (7,6)

Properties of the pattern:

- Each row and each column of the 16 x 16 grid holds one sample (n-rooks).
- Each **cell** of the pixel's 4 x 4 grid of 1/4 px cells holds one sample: sample `s` lies in cell
  column `(dx_s + 8) div 4`, row `(dy_s + 8) div 4`.
- The samples' centroid is the pixel's centre.
- In 1/256 px fixed point, sample `s` sits at `128 + 16 dx_s + 8`, `128 + 16 dy_s + 8` from the
  pixel's corner. This is exactly the same point.

[Rests on: `composite::SAMPLE_OFFSETS`, `sample_point`, `sample_cell`, `sample_fixed`;
`tests/composite.rs::the_sample_pattern_is_stratified_and_centred_on_the_pixel`;
`tests/python/test_contributor_samples.py`.]

**`alpha_round@1`: what paints.** Presence and drawn-ness are defined from the quantity the
renderer writes.

- **The rule.** An op that Cairo paints (a path's fill, stroke or background stroke; a mesh
  triangle) has its alpha written as Cairo writes it:

      alpha_byte = trunc(d * 65535 + 0.5) >> 8

  where `d` is the alpha clamped to `[0, 1]` and already multiplied by the record's group opacity.
- **Consequence.** An op whose `alpha_byte` is 0, which is every `d < 255.5 / 65535`, paints
  nothing. It covers no sample and is never a contributor to any pixel. For a gradient, the rule
  applies to every stop's 16-bit alpha short: a gradient whose every stop's short is at most
  `0xff` paints nothing (§9).
- **Point blocks.** They write CE's own truncated bytes, `trunc(255 * a)` (§9, point bytes), and a
  block whose alpha byte is 0 paints nothing.
- **Known error.** Taking a Cairo-painted op's alpha by truncation, `trunc(255 * d)`, or counting
  any `d > 0` as drawn, is a known error. Neither the oracle nor the recompute may use it.

**Core rule.**

- Paths and point blocks keep to `alpha_round@1`: a layer whose solid paint's alpha byte is 0, or
  whose gradient is clear, makes no fragment.
- **Deviation.** A mesh triangle whose alpha byte is 0 still makes fragments. They carry the colour
  `(0, 0, 0, 0)`, take part in the pixel's order and resolve selection, and composite to nothing.
  Under the contract they are not present. The contract tests list each one the core reports as a
  finding (`zero_alpha_fragment`).

[Rests on: `Lowering::layers`, `gradient::source`, `points.rs`, and `mesh::lower` (no drop).]

**Oracle reading.** The oracle decides at exactly the 16 points and cells above, and nowhere else.
It never makes a fragment of an op that paints nothing.

**Rejects:** a fill whose alpha rounds to zero, reported as present.

## 2. The camera, placement and clipping

Every formula in this section is written in the core's order of operations. A conforming
implementation evaluates them as written, with the shared libm, so its projected and mapped points
equal the core's to the last bit. That is what lets the shared kurbo flatten identical input.

**Why the last bit matters.** kurbo's subdivision count is a step function of its input, so a
difference in the last bit can, rarely, move every flattened edge of a curve by up to the
tolerance. No presence band narrow enough to see a 0.3 px stroke could absorb that.

### Placement and drawn points

- **Placement.** A point is placed by its record's 3 x 4 placement with the XF reader's `place`.
  An instance's placement composes after its record's by the reader's `compose(member, record)`.
- **A path record's drawn points** are made in three steps:
  1. Its shape is placed.
  2. If it has a reveal window `(lo, hi)`, the window is cut by CE's
     `pointwise_become_partial`, unless `lo <= 0` and `hi >= 1`:
     - With `C = len / 4` cubics, the window's ends map to `(i, r)`. Here `i` is `C - 1` and `r` is
       1 when the value is at least 1, `i = 0` and `r = 0` when it is at most 0, and otherwise
       `i = trunc(C * v)` and `r = (C * v) mod 1` (the Euclidean remainder).
     - Where `i_lo = i_hi`, the cut is `partial(c_{i_lo}, r_lo, r_hi)`.
     - Otherwise it is `partial(c_{i_lo}, r_lo, 1)`, then the whole cubics `i_lo + 1` up to, but not
       including, `i_hi`, then `partial(c_{i_hi}, 0, r_hi)`.
     - `partial(c, a, b)` is all four points `c_3` when `a >= 1`. Otherwise it is
       `head(tail(c, a), (b - a) / (1 - a))`, using `lerp(p, q, t) = p + (q - p) * t` per
       coordinate:
       - `tail(c, t) = [lerp(abc, bcd, t), bcd, cd, c_3]`
       - `head(c, t) = [c_0, ab, abc, lerp(abc, bcd, t)]`
       - `ab = lerp(c_0, c_1, t)`, `bc = lerp(c_1, c_2, t)`, `cd = lerp(c_2, c_3, t)`,
         `abc = lerp(ab, bc, t)`, `bcd = lerp(bc, cd, t)`.
  3. If it has a shape blend, each point `p` is blended with its second shape's point `q`, placed
     by the blend's own placement, as `(1 - t) * p + t * q` per coordinate.
- **A face group.** Face `i` is a run of the record's points: from the previous face end (0 for the
  first) to `face_end_i`. Where the record states no face ends, the runs are equal, of
  `len / faces` points each.

### The 3D projection

The projection applies under a view whose camera is 3D. Its inputs are `φ`, `θ`, `γ`, the focal
distance `d`, zoom `z0` and the exponential option, about the view's `frame_center` `c`. A degree
is `DEG = τ / 360`.

    Rz(a) = [[cos a, −sin a, 0], [sin a, cos a, 0], [0, 0, 1]]

    Rx(a): v = (a·1, a·0, a·0);  N = sqrt((v0·v0 + v1·v1) + v2·v2)
           κ = N <= 1e-3 ? ((0.5 − n2/48) + (n2·n2)/3840) with n2 = N·N : sin(N/2) / N
           (x, y, z, w) = (κ·v0, κ·v1, κ·v2, cos(N/2))
           x2 = x·x, y2 = y·y, z2 = z·z, w2 = w·w
           xy = x·y, zw = z·w, xz = x·z, yw = y·w, yz = y·z, xw = x·w
           [[((x2 − y2) − z2) + w2,   2·(xy − zw),             2·(xz + yw)],
            [2·(xy + zw),             ((−x2 + y2) − z2) + w2,  2·(yz − xw)],
            [2·(xz − yw),             2·(yz + xw),             ((−x2 − y2) + z2) + w2]]

    (M·N)[i][j] = (M[i][0]·N[0][j] + M[i][1]·N[1][j]) + M[i][2]·N[2][j]
    R = I; then R = F·R for F in [Rz(−θ − 90·DEG), Rx(−φ), Rz(γ)], in that order

    rotated(p): q = (p0 − c0, p1 − c1, p2 − c2);  r_k = (q0·R[k][0] + q1·R[k][1]) + q2·R[k][2]

    factor(z): exponential:  z < 0 ? d / (d − z) : exp(z / d)
               otherwise:    d − z < 0 ? 1e6 : d / (d − z)

    project(p):   (x, y, z) = rotated(p);  s = factor(z)·z0;  (x·s, y·s, z)
    unproject(p): s = factor(p2)·z0;  (p0 / s, p1 / s, p2)

Notes on these formulas:

- `1e6` is CE's **behind-the-eye factor**. A point on the eye's plane takes `d / 0`.
- The **key** of a projected point is `−factor(z)`.
- A point is **keyed planar** when the option is off and `d − z > 0`. Only there is a key affine
  across a plane on screen.
- **In the camera's own axes,** the same formulas apply with `R = I` and `c = 0`.

[Rests on: `project.rs` (`rotation_about_z`, `rotation_about_axis`, `rotation_matrix`, `matmul`,
`Projection::rotated`, `project`, `unproject`, `factor`, `keys_planar_at`, `in_camera_axes`).]

### Before display and placement kinds

- **Before display.** A record's placed points become shown points as follows:
  - Points of a record **kept** or **fixed in frame** are unchanged.
  - **Projected** points go through `project`.
  - Points of a **fixed-orientation** record with centre `C` are each moved by `o = project(C) − C`,
    taken per coordinate, as `p + o`.

  A mobject with any non-finite point would be shown as one point at the origin, but each producer
  meets non-finite input first (below).
- **Placement kinds.** Under a 3D camera:
  - A `world` record is projected, or fixed-orientation where it carries a centre.
  - An `overlay` record is fixed in frame. It is the overlay's (§1).
  - A `screen` record is kept.

  Under a flat or a moving camera, every record is kept. Kept and fixed-in-frame records are in
  front of all depth: key `−∞` (§5).

### Non-finite input, per producer

- **A mesh** with any non-finite placed vertex draws nothing.
- **A path** with fewer than 4 drawn points, or with any non-finite placed point, draws nothing.
- **A points record:**
  - A point whose placed position is not finite is skipped, and the others are drawn.
  - A point whose screen position is not finite, or whose key is NaN, is skipped.
  - A point whose screen position exceeds `4e18` in magnitude is skipped.

[Rests on: `mesh::lower`, `path::drawn_world`, `Lowering::depth_points`, `points.rs`.]

### The frame mapping

The mapping from frame units to pixels is CE's Cairo context matrix:

    xx = W / fw;  yy = −(H / fh);  x0 = (W / 2) − m0·(W / fw);  y0 = (H / 2) + m1·(H / fh)
    map(p) = ((xx·p0 + 0·p1) + x0,  (0·p0 + yy·p1) + y0)

The centre `m` depends on what is drawn:

- **Paths, face groups' faces, meshes and instances** use the view's `mapping_centre`. CE draws
  these through its Cairo context.
- **Point clouds and images** use `frame_center`. A point maps as

      ((p0 − c0)·(W / fw) + W / 2,  (p1 − c1)·(−(H / fh)) + H / 2)

The flat camera and the moving camera map the same way. Only the view's numbers differ.

### Pre-projected records and clipping

**Pre-projected records** (feature bit 1, XF 1.1):

- The tail's points are taken as placed (identity placement), and its paints and gradient ends as
  produced. Nothing is projected or lit again.
- Their depth is keyed in the camera's own axes. Each shown point is put back by `unproject` and
  keyed there.
- A fixed-orientation pivot is the pivot as produced, moved as the record's first point moved.

[Rests on: `instance::produced_path`, `project::produced_key`, `produced_pivot`;
`tests/pre_projected.rs::a_pre_projected_record_never_moves_with_the_view_s_rotation`.]

**Clipping.**

- Nothing is clipped at the eye. A point behind the eye takes the factor `1e6` and is drawn where
  that puts it.
- Fragments exist only on the frame's pixels.
- A triangle is clipped at `±65536` px (§3).
- A path's curves beyond the guard band are drawn as chords (`guard_chords@1`, §4).

[Rests on: `tests/projection.rs::a_point_behind_the_eye_takes_ces_large_factor`.]

**Oracle reading.** The oracle projects every record by this text and checks its camera against
closed forms, and against the engine's `arachne.camera.projection` (screen boxes within 1 px).

## 3. Mesh coverage

**Core rule.**

- **Corner rounding (`mesh_fixed_point@1`).** A triangle's corners are placed, shown (§2) and
  mapped. Each coordinate is then rounded to the 1/256 px grid as `round(v * 256)`.
- **Degenerate and clockwise triangles.** A triangle whose rounded corners are collinear covers
  nothing. One whose rounded corners wind clockwise is turned.
- **The edge test.** Each edge `a → b` gives the integer function
  `e(p) = dx (p_y − a_y) − dy (p_x − a_x)`, where `dx = b_x − a_x` and `dy = b_y − a_y`, evaluated
  at the sample's fixed-point position (§1). A sample is inside where every edge's function is
  positive.
- **Edge ownership (`mesh_edge_ownership@1`).** A sample on an edge (its function is zero) is inside
  where the edge runs up (`dy < 0`), or runs right along a row (`dy = 0`, `dx > 0`). So a sample on
  an edge two triangles share belongs to exactly one of them.
- **Guard clipping.** A triangle reaching past `±65536` px is clipped there. The clipped polygon is
  fanned from its first corner, and the pieces belong to one triangle.
- **The fragment.** Its coverage is the mask of its inside samples (bit `s` for sample `s`). A
  pixel with an empty mask holds no fragment.

[Rests on: `mesh.rs` module doc, `Meshes::push`, `Meshes::raster`, `clip_to_guard`;
`tests/mesh.rs::a_mesh_far_larger_than_the_frame_covers_it_without_cracks`,
`::a_meshs_coverage_is_its_exact_area_within_the_sample_patterns_bound`,
`tests/probes.rs::no_background_shows_through_a_surfaces_internal_edges_opaque_or_transparent`.]

**Identity of a mesh fragment.** Its `sub` is the triangle's index in its shape. For an instances
record's member `i` of a `t`-triangle prototype, it is `t i` plus that index. All the triangles of
one mesh drawn form one **surface**, and they may merge into one unit (§9).

**Point blocks (`point_blocks@1`).** For a `points` record's point `i`:

- **Position.** It is placed and shown, mapped as a point cloud (§2), and truncated toward zero to
  the pixel `(X, Y)`.
- **The block.** Its side is `t = trunc(1 + (radius − 1))`, with no block for `t < 1`. With
  `lo = floor(−t / 2) + 1`, it covers the whole pixels `(X + o, Y + o')` for `o, o'` in
  `lo .. lo + t`, clipped to the frame. Every pixel it covers gets the full mask.
- **Its `sub`.** It is `(offset index) * n + i` for `n` points. The offset index is
  `(o − lo) * t + (o' − lo)`. So where blocks overlap at one depth, the later write in CE's
  offset-major order is on top.

[Rests on: `points.rs` module doc;
`tests/path.rs::overlapping_blocks_keep_ces_offset_major_write_order`.]

**Oracle reading.**

- **The margin.** Let `m = ε + 1/256` px, where `ε` is the triangle's screen shift (§6): how far the
  input grid lets the corners move, the larger at an edge's two corners.
  - Any input within the snap moves each unrounded corner by at most `ε`.
  - The rounding to 1/256 px then moves it by at most `√2/512 < 1/256` more. So every rounded edge
    a conforming input can produce lies, over the triangle, within `m` of the nominal unrounded
    edge, whichever way a coordinate near a half-step rounds.
  - Edge ownership decides only samples on a rounded edge, which lie within `m` of the unrounded
    one.
- **The presence verdicts.** Taking distances from the sample to the nominal triangle's unrounded
  edges:
  - **`Yes`** where the sample lies inside the unrounded triangle at distance greater than `m` from
    each of its edges' lines.
  - **Absent** where it lies outside the triangle by more than `m`. That is, it lies outside some
    edge's line by more than `m`, or farther than `m` from the triangle.
  - **`Either`** otherwise.
  - A triangle with no point farther than `m` from all its edges is `Either` wherever it is not
    absent.
  - The nominal fixed-point mask lies inside this verdict. The oracle may also compute it exactly by
    the rules above, but a verdict never depends on which way a rounding near a half-step went.
- **Point blocks.**
  - A block pixel is `Yes` at every sample.
  - Where a point's screen coordinate lies within `ε` of an integer, the block's pixels along that
    side are `Either`. Its truncation can then move by a pixel.

**Rejects:** a mesh triangle missing from the core's fragments at a sample farther than `m` inside
it (witness W7, §12).

## 4. Path coverage

**The outline.**

- **Subpaths.** A path's drawn points, shown and kept in frame units, split into CE's subpaths. A
  split falls where a cubic's end and the next cubic's start differ, in `x` or `y`, by more than
  CE's tolerance: `|a − b| <= 1e-6 + 1e-5 |b|`, per coordinate, means equal. A subpath of fewer
  than four points is dropped.
- **Path elements.** Each subpath becomes a kurbo path: a move to its first point, then a curve per
  cubic (anchor, handle, handle, anchor).
- **Closing.** A subpath is closed when its first and last points are equal by the same test. Its
  last point is then taken as its first, and a close is added.

**Layers.** A path paints up to three layers, in this order, as three fragments at a pixel:
background stroke, fill, stroke. Their `sub` is `3 i + k` (`k` = 0, 1, 2) for the record's
`i`-th path:

- an instances record's member `i`;
- `i = 0` for a face of a face group, since each face is its own order entry.

A layer draws nothing in any of these cases:

- it has no colour rows (`paint_rows@1`, §9);
- its solid colour paints nothing (`alpha_round@1`);
- its gradient is clear;
- it is a stroke whose width in frame units, `w`, is not above 0, or is NaN.

### The guard (`guard_chords@1`)

The guard works in frame units, with `reach = w_max * 10`, the widest stroke's width times the
miter limit:

- **The band.** The guard band is the frame's pixel box `[0, W] × [0, H]` widened by `65536` px on
  every side, mapped back to frame units through the frame mapping (§2) axis by axis: `x` from
  `(−65536 − x0) / xx` to `(W + 65536 − x0) / xx`, `y` from `(−65536 − y0) / yy` to
  `(H + 65536 − y0) / yy`, each pair taken low to high.
- **When it applies.** The guard applies when some point of the outline lies within `reach` of the
  band's edge, or beyond it: on some axis, its coordinate is at most the band's low end plus
  `reach`, or at least its high end less `reach`. Every point counts: each move, and every point of
  every curve element, its control points included.
- **The area.** It is then the frame's box in frame units (the pixel box `[0, W] × [0, H]` mapped
  back as above) widened by `reach + 1` frame units on every side.
- **Splitting.** Each cubic is cut into pieces by a recursive split with depth limit 40:
  - a piece whose four control points all lie strictly beyond one side of the area (every `x`
    below the area's low `x`, or every `x` above its high `x`, or likewise in `y`) misses it, and
    is kept as is;
  - a piece whose `CubicBez::bounding_box` has its larger side at most `64 * size` (`size` the
    area's larger side) is kept as is;
  - a piece at depth 0 is kept as is;
  - otherwise the piece is halved by `CubicBez::subdivide`, and each half is split again with the
    limit one lower.
- **Chords.** A piece that misses the area is drawn as a line to its end. Others are drawn as
  curves. A quadratic is raised to a cubic first. The guarded outline is the outline with each
  piece in place: every move, line and `ClosePath` is kept, so a closed subpath stays closed and
  its closing join is stroked as before.
- **Which layers.** The fill and the undashed strokes use this outline. A dashed stroke is laid by
  §8's dash rule on the unguarded outline.

A piece and its chord both lie inside its control hull, which misses the area, so the guard changes
nothing inside the area.

[Rests on: `Lowering::guard`, `path::misses`, `cull_pieces`, `culled`.]

### Fill, strokes and coverage

- **Fill.** The outline is mapped to pixels and flattened by kurbo's `flatten` at tolerance `0.1`
  px (Cairo's default). Each run from a move to a close, or to the next move, is a ring. A ring of
  fewer than three points is dropped. Rings are filled by the **nonzero** rule.
- **Strokes.** The outline is stroked in frame units by kurbo's `stroke` at tolerance `0.1 / s`,
  where `s = max(|xx|, |yy|)` and the style is given by §8. The join edits of §8 are applied, and
  the result is mapped and flattened as the fill is, then filled by the nonzero rule.
- **Edges.** In the rings, horizontal edges are dropped, and a ring holding a non-finite point is
  dropped. Every other edge is kept with its winding direction.
- **Pixel area.** A pixel's area is the exact area of the pixel inside the region where the winding
  number is nonzero. A sum within `1e-9` of 0 is 0, and one within `1e-9` of 1 is 1. A pixel whose
  area is 0 holds no fragment of the layer.
- **Cell areas.** A cell's area is the same computation over the pixel magnified 4 times about its
  corner, each cell then a unit square.
- **Storage.** The core stores each pixel's area and each cell's area as binary32.

[Rests on: `path::subpaths`, `close`, `outline`, `rings`, `TOLERANCE`; `scanline::edges_of`, `SNAP`,
`cover`, `cell_areas`; `tests/scanline.rs`; `tests/path.rs::a_fill_covers_each_pixel_by_its_exact_area`,
`::a_self_crossing_outline_fills_by_the_nonzero_rule`; `tests/coverage_bound.rs::a_rasterized_paths_area_lies_within_the_drift_of_its_cells_mean`;
the reference shape `cell_area` in `tests/python/test_contributor_samples.py`.]

**Presence at a sample (`cell_presence@1`).**

- **The cell byte.** A path layer's 8-bit coverage at sample `s` is
  `floor(clamp(A_s, 0, 1) * 255 + 1/2)`, where `A_s` is the binary32 area of the sample's cell. It
  is 255 at every sample when the pixel's area is 1.
- **Presence.** The layer **covers** the sample where that byte is not 0, that is, where
  `A_s >= 1/510`. Every order (§7), resolve (§9) and report reads that byte, never the area.
- A layer whose pixel area is positive may cover no sample.

[Rests on: `TransparentPass::sample_coverage`, `composite::area_coverage`, `order::shared`.]

**Oracle reading.** The oracle computes each cell's exact nonzero area `A` of the same kurbo rings
(identical input, §2), by its own trapezoid integration written from this text. Its admitted
interval is `[A − μ, A + μ]`, where

    μ = α + 2^-24 + μ_ε
    μ_ε = 16 (2 ε L + π ε² m)

- `α = 2^-30 (1 + n_e)` bounds the difference between two binary64 integrations. `n_e` is the
  number of ring edges meeting the cell's band of rows.
- `2^-24` covers binary32 storage. The step from the stored area to the byte needs no term of its
  own: it widens the binary32 area to binary64 exactly, then clamps, multiplies by 255, adds `1/2`
  and truncates in binary64, whose rounding is under `2^-44` of a byte, far inside the `2^-24`
  term. A sample whose area lies within `μ` of a half-byte boundary is admitted at both bytes by
  the interval rule below.
- `μ_ε` bounds the area that moving the flattened outline by up to `ε` (its screen shift, §6) can
  bring into or take out of the cell, as a fraction of the cell. That area lies within the outline's
  `ε`-neighbourhood:
  - `L` is the length of the rings inside the cell widened by `ε` on every side;
  - `m` is the number of connected pieces of the rings there;
  - a piece of length `l` has an `ε`-neighbourhood of area at most `2 ε l + π ε²`;
  - the factor 16 is the inverse of a cell's area.

  The band models moving the flattened outline. It does not model re-flattening a moved curve:
  the flattening is part of the definition.

From the interval:

- The layer is `Yes` at the sample where `A − μ >= 1/510`.
- It is absent where `A + μ < 1/510`.
- It is `Either` otherwise.
- The admitted cell bytes are every value of `floor(clamp(v, 0, 1) * 255 + 1/2)` for `v` in the
  interval.
- The admitted whole-pixel bytes are found the same way, from the pixel's area and its own `μ`.

**Rejects:** a stroke 0.3 px wide or less, missing from the core's fragments at a sample whose cell
it covers by at least `1/510 + μ`; and a cell byte outside its admitted interval.

## 5. Depth

Each fragment carries, at each sample, a key (smaller is nearer) and an error bound. What the key
is depends on the record.

**In front of all depth.** Every fragment of a kept or fixed-in-frame record (§2), and every
fragment under a flat or moving camera, has key `−∞` at every sample. Two such fragments tie and
paint in order. Against a finite key, `−∞` is in front.

[Rests on: `project::placement`;
`tests/mesh.rs::under_a_flat_camera_meshes_paint_in_draw_order_whatever_their_z`.]

### World planes

A plane is written `n · p = e`. Its **canonical** form is built in three steps:

1. Its normal is divided by its length and turned so that its largest component (by magnitude; on
   equal magnitudes, the lower axis) is positive.
2. Its offset is `unit · p` through its point `p`, with error bound `4u (|unit| · |p|)`.
3. The normal's components and the offset are each rounded to a multiple of `PLANE_GRID = 2^-20`
   as `round(v / 2^-20) * 2^-20`. The unrounded plane is kept beside the rounded one.

The **axis plane** of coordinate `k` through `p` has normal `e_k` and offset `p_k`, rounded to the
grid.

**How a path lies (its planarity).** Where its record carries an XF 1.2 `planes` code for the path,
the code decides. Otherwise it is decided from its points `P`, with `tol = 1e-9 * max(extent, 1)`
where `extent` is the widest of the points' extents along the axes:

- **Axis.** It is the axis plane of the first coordinate, in the order z, y, x, that every point
  shares with the first point within `tol`.
- **Newell.** Otherwise, Newell's normal through the first point, where every point lies on that
  plane within `tol`.
- **Widest.** Otherwise, the normal of the widest triangle on the first point, again where every
  point lies on it within `tol`. This is the case of an outline crossing itself. The widest triangle
  is the first point, the point farthest from it, and the point farthest from the line through
  those two (the last of equals).
- **Nearest.** Otherwise, for a face of a face group, the plane nearest it: Newell's normal through
  the centroid.
- **Curve.** Otherwise none.

The plane is fitted to the points as drawn: the axis plane through the first point, or the canonical
plane of the chosen normal through the first point (or through the centroid, for `Nearest`).

[Rests on: `WorldPlane::decide`, `fit`, `rounded`, `canonical`, `axis_plane`.]

**The normals.** Take the points relative to the first point `f`.

- **Newell's normal** sums, over each point `a` and the next `b` (the last point's next being the
  first), the components `(a_y − b_y)(a_z + b_z)`, `(a_z − b_z)(a_x + b_x)` and
  `(a_x − b_x)(a_y + b_y)`.
- **The widest normal** is `far × (p − f)` with the largest length over the points, where `far` is
  the point (relative to `f`) farthest from `f`. On equal lengths, both picks take the last.
- **Points somewhere.** A normal `n` **points somewhere** when
  `|n| > max(1e-9 · extent², 4 · g · sqrt(0.75) · perimeter)`. Here `extent` is the points' widest
  extent along an axis, `perimeter` is their closed polygon's length, and `g` is the snap grid (0
  where not snapped).
- **When a plane holds.** Planarity's tests take `g = 0` and a perimeter of 0. A normal's plane
  holds the points when the normal points somewhere and, for every point `p`, `|û · p − û · f|` is
  within `tol`, where `û` is the normal divided by its length.

**On a line.** A record's points lie **on a line** when their widest normal does not point
somewhere, with the record's own grid and perimeter. Points on a line have no plane of their own,
whatever their code.

[Rests on: `depth::on_a_line`, `points_somewhere`.]

### A plane's key on screen

For a world plane `n · p = e`, its key across the pixel plane, written `key_plane` in what follows,
is `a + b x + c y`, with:

- `m = R n`;
- `D = e − n · c − m_2 d`, the plane's distance from the eye;
- `g_x = m_0 / (z0 xx)` and `g_y = m_1 / (z0 yy)`;
- `a = −(−g_x x0 − g_y y0 − m_2 d) / D`, `b = −g_x / D` and `c = −g_y / D`, computed from the
  rounded plane.

The key is not formed in any of these cases:

- under the exponential option;
- at zoom 0;
- where `D` is 0;
- where the unrounded plane's `|D|` is within its offset's bound of 0;
- where the plane is seen too near edge-on: `|D| / (|m| · |q − (0, 0, d)|) < min_cos`, or NaN,
  for `q` the rotated first point of the record. `min_cos` is 0 for a path (however nearly edge-on,
  it is keyed) and `10^-9` for a mesh triangle;
- where a coefficient is not finite.

A plane read at a point is `a + b x + c y + 0.0`. The `+ 0.0` turns `−0` into `+0`, so equal keys
compare equal.

[Rests on: `Projection::key_plane`, `Plane::at`, `mesh::MIN_PLANE_COS`.]

### Mesh triangles

Mesh triangles are keyed as follows, under a 3D camera:

- **Fixed-orientation meshes** take their facing plane (below), with no registry handle.
- **Offered triangles.** A projected triangle in world space whose three placed corners are all
  keyed planar, and whose alpha byte is not 0 (§1), is **offered** to the frame's registry (§7),
  whether or not it has a plane of its own. (Below, under Paths: the core offers it whatever its
  alpha, the deviation `invisible_registry_offer`.)
  - **Its plane.** It is the axis plane through its first corner where the corners share a
    coordinate within `max(tol, g)`, with `g` the view's `input_grid`. Otherwise it is the
    canonical plane of `(c1 − c0) × (c2 − c0)` through the first corner. It has no plane where that
    normal points nowhere.
  - **Flat or not.** It is offered **flat** when it has a plane and its corners are not on a line.
    Otherwise it is offered with no plane of its own, on the axis plane its corners share, if any.
  - **Its key.** It is keyed on its plane by `key_plane` where that forms. Otherwise it is keyed on
    the plane on screen through its corners' `(x, y, key)` (§6). A flat triangle keyed this way
    then carries no identity, though it still took part in the partition and shaped the identities
    of others. A triangle with no plane of its own keeps its handle, and takes the
    identities it lies on once the registry settles (§7).
- **Every other triangle** is keyed on the plane on screen through its corners. This includes a
  triangle with a corner not keyed planar.

[Rests on: `mesh::lower`, `WorldPlane::of_triangle`, `Registry::offer_triangle`, `Meshes::identify`.]

### Paths

**Paths in world space** (registered). A path is offered to the registry when all of these hold:

- it is under a 3D camera, projected (not fixed-orientation, not pre-projected);
- every placed point is keyed planar;
- its planarity is not `Nearest`;
- **it paints something:** some layer of it has a fragment under `alpha_round@1` (§1).

It is offered **flat** when its planarity is Newell, Widest or Axis, its points are not on a line,
and the fit finds a plane. Otherwise it is offered with no plane of its own.

**Only painting records are offered.** The same holds for a triangle: one whose alpha byte is 0
(§1) is never offered. A record that paints nothing has no depth anyone sees, so it may neither
join nor split the identities of records that do. An invisible record on the plane of two visible
ones changes neither their order nor their ties.

**Deviation (`invisible_registry_offer`).** The core this contract describes offers a path to the
registry before it drops its layers that paint nothing, and offers every mesh triangle whatever its
alpha (`Lowering::coplanar`, `mesh::lower`). An invisible record offered there takes part in the
partition: its flat identity can attach a visible record and so take it out of an identity it
shared with another, or bring two visible records into one, and either can turn a tie into a
depth decision or the reverse. Under a 3D camera, two overlapping lines on `z = 0` are an
`ExactTie` decided by paint order; an invisible polygon on their plane attaches one of them, the
pair is then decided by depth, and the lines' visible pixels change (witness W17, §12). The oracle
builds its sets (§7) from painting records alone, so such a relation is a `mismatch`. Where the
reported relation is inside the sets built with the invisible records offered too, the mismatch is
counted under the name `invisible_registry_offer`, beside `zero_alpha_fragment`.

**What it retains:**

- **A flat path** retains its one identity only where its own plane keys on screen (min_cos 0, over
  its region). Its fill is keyed on that plane, and every layer's **row** for that identity is that
  same plane read at the sample (no fringe). Where its plane does not key, it has no identity, and
  every layer is keyed along its curve.
- **A path with no plane of its own attached to flat identities** has rows on each attached
  identity's plane, keyed by `key_plane`:
  - **The first identity.** Where its row does not form (the eye in that plane), the path has no
    identities at all.
  - **A later identity.** Where its row does not form, that identity stays in the list with no
    row. Against a fragment it shares only on that identity, the path reads its own key.
- **A path with no plane of its own on axis planes alone** keeps only the axis identities whose
  rows form. Where none forms, it has no identities.

**A path with no plane of its own** always keys its own fill and strokes along its curve.

**Unregistered planes.** Two kinds of path are never offered. Each takes the plane fitted to its
own points by its planarity, with no `on a line` test:

- a pre-projected path, fitted in the camera's own axes;
- a `Nearest` face.

Its fill is keyed on that plane where it keys, and along its curve otherwise. Its strokes are keyed
along its curve. It has no identity, and takes part in no tie policy that needs one.

**Not keyed planar.** A world path with a point that is not keyed planar keys every layer along its
curve, and is not offered.

[Rests on: `Lowering::coplanar`, `Lowering::depth_points`, `Lowering::world_depth`,
`Registry::offer_path`, `composite::Depths::against`.]

### Key rules

**`fringe_key@1`.** This applies to a layer keyed on a sloped plane (`b` or `c` not 0). At a sample
outside the layer's outline, it reads the plane at the outline's point nearest the sample, not at
the sample. Outside is decided by the point-in-outline test on its flattened rings: nonzero winding
at the sample, half-open in `y`. Inside, it reads at the sample.

[Rests on: `path::fringe_depths`, `Layer::nearest_on_outline`, `scanline::mask`;
`tests/path.rs::beside_a_grazing_sliver_its_fill_beyond_its_outline_is_behind_the_nearer_face`.]

**`curve_key@1`: the centreline.** A path's centreline is built from its shown points. Each point's
screen position is `map(p)`, and its key is `−factor(p_2)`.

- **Vertices.** For each subpath of at least four points, in order:
  - The first vertex is the subpath's first point. Its error is `key_error` of that point (§6).
  - For each cubic with control points `c_0..c_3`: let `L = |X_1 − X_0| + |X_2 − X_1| + |X_3 − X_2|`
    be the length of its control polygon on screen. Then `n = clamp(ceil(L / 2), 1, 64)`, or
    `n = 1` where `L` is not finite. Its vertices are `s = 1 .. n`:
    - `t = s / n`, `v = 1 − t`;
    - the weights are `(v·v·v, 3·v·v·t, 3·v·t·t, t·t·t)`;
    - the position and the key are each `((w0·f0 + w1·f1) + w2·f2) + w3·f3` of the control points'
      values;
    - every vertex of the cubic carries the error
      `max_j key_error(c_j) + 4u · max_j |k_j|`.
- **The joint substitution.** A cubic's own first control point is never a vertex after the
  subpath's first. So a joined cubic's first piece runs from the previous cubic's last vertex, with
  that vertex's position, key and error, to its own `t = 1/n` vertex. The CE tolerance that joins
  two cubics compares `x` and `y` only, so the two points may differ in key by any amount.
- **Segments.** A segment joins consecutive vertices within a subpath. Its error is the larger of
  its two vertices' errors. A segment with a non-finite coordinate or key is dropped. If none
  remains, the centreline is the zero-length segment at its first finite vertex. If there is no
  finite vertex, every layer read on it has key `−∞`.
- **The read at a point `P`.** A layer of **body** `h` reads as follows. `h` is half the stroke's
  width in pixels, `w |xx| / 2`, and 0 for a fill.
  - For each segment `A → B`, the foot parameter is
    `t = ((P − A) · (B − A)) / |B − A|²` (0 for a zero-length segment), and the distance is from `P`
    to the segment's point at `clamp(t, 0, 1)`.
  - A segment **holds** `P` where `t ∈ [0, 1]` and the squared distance is at most `h²`.
  - The read is the least key among the holding segments, with error the largest of theirs.
  - Where none holds, the read is the key of the **nearest** segment (least squared distance; the
    first on ties).
- **A segment's key at `P`.** Write `k_A`, `k_B` for its ends' keys and `e` for its error.
  - Where its length is 0 or `k_A = k_B`, the key is `min(k_A, k_B)`, with error `e + |k_A − k_B|`.
  - Where `t < 0`, it is `k_A`, with error `e`. Where `t > 1`, it is `k_B`, with error `e`.
  - Otherwise, with `G = (k_B − k_A) / |B − A|²`, `b_x = G (B − A)_x` and `b_y = G (B − A)_y`, it is
    the plane `(k_A − b_x A_x − b_y A_y) + b_x x + b_y y` read at `P`, clamped to
    `[min(k_A, k_B), max(k_A, k_B)]`. Its error is in §6.

[Rests on: `path::curve_depth`, `depth::Centreline` (segments, `key_under`, `find`), `key_on`,
`foot`; `tests/centreline.rs`; `tests/probes.rs::a_curve_passing_behind_a_surface_is_hidden_there`.]

**`facing_plane@1`.**

- **Fixed-orientation records.** A fixed-orientation record's fragments are keyed on the
  camera-facing plane through its centre. This covers paths, meshes and point blocks: every
  camera-facing label. The plane is the canonical plane of normal `R`'s third row through the
  centre, keyed by `key_plane` with min_cos 0. Where no plane keys (the exponential option, zoom 0,
  or a centre that is not keyed planar), the centre's own key is used everywhere.
- **Pre-projected records.** A pre-projected record with a pivot is keyed the same way, in its
  camera's axes, at its pivot.
- **Points.** Every projected point of a `points` record is keyed on the facing plane through its
  own placed position. So points and camera-facing surfaces at one depth tie.

[Rests on: `Projection::facing_plane`, `facing_planes`;
`tests/mesh.rs::a_fixed_orientation_mesh_is_a_camera_facing_plane_at_its_centres_depth`,
`::a_fixed_orientation_label_on_a_surface_facing_the_camera_ties_by_draw_order`,
`::points_at_one_depth_tie_by_draw_order_whether_or_not_fixed_in_orientation`.]

**Glyphs.** A glyph is a path. A glyph placed in the world is keyed as any path. A glyph of a
camera-facing label is keyed on the label's facing plane. No rule belongs to glyphs alone.

### The two stroke-depth adjustments

They are different rules, and both apply.

1. **`stroke_on_own_fill@1`.** This applies to a path's stroke or background stroke drawn at or
   after its own fill: same producer, same order entry, same path, with the stroke's `sub` not
   below the fill's. A background stroke is drawn before its fill, so it is never adjusted.
   - At every sample where both cover, the stroke's key becomes the nearer of its own and the
     fill's.
   - Elsewhere it keeps its own.
   - Its error bound becomes the largest, over the samples, of the bound of the key it took there.
     Where the two keys tie within their bounds (§7's `tied`), both bounds count. Where they do not
     both cover, its own bound counts.
   - Only the stroke's own row is adjusted. Rows on shared planes are not.

   [Rests on: `composite::strokes_on_own_fills`; `order.rs`
   `inside_its_fill_a_stroke_is_at_least_as_near_as_its_fill`,
   `outside_its_fill_a_stroke_keeps_the_wire_s_depth`,
   `a_stroke_nearer_than_its_fill_within_their_bounds_is_not_behind_what_the_fill_is_not`.]
2. **`shared_plane_read@1`.** Against a fragment `b` with which it shares a registered identity
   (§7), a path layer `a` is read at each sample on its **row** for the first identity in its own
   list that `b` holds too, with that row's error bound. Where that identity has no row (above), it
   reads its own key. Against every other fragment it reads its own key. Mesh triangles and point
   blocks have no rows; they are always read on their own. So a fragment's key and bound at a
   sample are a function of the pair.

   [Rests on: `composite::Depths::against`, `Producer::plane_depths`;
   `composite.rs::a_stroke_is_read_on_its_path_s_plane_only_against_what_shares_that_plane`,
   `tests/path.rs::a_stroke_on_a_face_s_registered_plane_ties_it_exactly_and_meets_another_plane_by_depth`.]

**A flat pixel.** Where every fragment's rows, own and shared, hold one key for all fragments at
every sample, paint order decides every pair at the pixel. This happens when every key is `−∞`, or
every fragment is read on one plane.

**Oracle reading.** The oracle computes every key and row by this section from XF. It applies both
adjustments and the pair-dependent read. Where the registry decides which identities a pair shares
or which row it reads, the oracle uses §7's conservative sets.

## 6. The tie band

For each fragment `f` at each read point (a sample `k`, or the pixel's centre for curve reads), the
oracle computes three things from XF:

- `o_f(k)`, the key of §5's model;
- `R_f(k)`, its arithmetic bound (`band_arith@1`);
- `S_f(k) = (S⁻, S⁺)`, its input bound (`band_input@1`).

These give the fragment's **interval**:

    I_f(k) = [o − S⁻ − 3R,  o + S⁺ + 3R]

The oracle never reads the core's reported error bounds to form `R` or `S`.

### Why the band has this shape

- **`R`** bounds how far any conforming computation of the model's key stands from its exact value
  on the XF input. This holds for the core's computation and the oracle's alike.
- **Ties.** Where the core ties two keys `c_a`, `c_b`, it has `|c_a − c_b| <= e_a + e_b` with its
  own bounds `e <= R` (the contract test below). Each `c` and each `o` lies within `R` of the exact
  key. So a core tie lies within `3 (R_a + R_b)`: the intervals meet.
- **Strong orders.** Where the core orders `a` strongly in front, `c_a < c_b` gives
  `o_a − 2R_a < o_b + 2R_b`. So `a` can be in front only where `I_a` starts below where `I_b` ends.
- **`S`** widens the interval by the input's own precision. It is what lets CE's order, drawn before
  the snap, be judged on the same sets. It is not needed to contain the core's decisions: the core
  decides on the snapped input.

### Non-finite keys

- A key of `−∞` has `R = S = 0` and the interval `{−∞}`.
- Two `−∞` keys are equal, and the label is `ExactTie` (§7).
- Against a finite key, `−∞` is strictly in front.

The contract test asserts that every such key equals the core's exactly, and no bound is asserted
for it. The core's own bound for a constant `−∞` plane is infinite, and is not compared.

### Unbounded keys (`truth_unable`)

Where `S` or `R` is infinite, the oracle states no bound. This happens where a share `σ`, a screen
shift `ε`, or a key reach `kr` is infinite, for example when a plane passes within its departure of
the eye. The fragment's samples are then `truth_unable` (class `band_unbounded`). They are never
counted as ties.

An infinite `ε` also bounds no presence. The mesh margin (§3) and the presence band (§4) never read
it as absence, and never as `Either`: every sample of the record within the window the oracle is
asked to judge is `truth_unable`, class `band_unbounded`. A move of unbounded size can bring the
record anywhere in the window, so no sample there is known to be clear of it.

### The snap's departure

Write `g` for `input_grid` (0 when the view states none), and write the record's placements as
`[outer, inner]` (the identity for a missing one).

**`Δ(n)`, the departure along a unit normal `n`.**

- Start with `C^T n = n`, and run through each placement `M` that is not the identity, outer first:
  - add `|C^T n|_1 (g/2 + 2^-25 X_M)`, where `X_M` is the largest `|x|_1` over the points `M`
    places (the inner-placed points for the outer placement, the shape's points for the inner);
  - then replace `C^T n` by `L_M^T C^T n`, where `L_M` is `M`'s linear part.
- Finally add `(g/2) |C^T n|_1` for the shape's points.
- When `g = 0`, none of these terms is added.
- In every case, add `8u max_p |n| · (|outer| (|inner| |p|))` (entrywise absolute values) for the
  placing's own rounding.

This follows from the writer's rules: positions and translations snap to multiples of `g` (within
`g/2`), and a placement's linear entries snap to multiples of `2^-24` (within `2^-25`).

**A shape blend.** XF does not constrain a blend's `t`, and the oracle does not refuse one outside
`[0, 1]`. A blended point `(1 − t) p + t q` (§2) departs by at most `|1 − t| Δ_p + |t| Δ_q`, where
`Δ_p` is the departure of `p` through the record's placements and `Δ_q` that of `q` through the
blend's own, both along the same normal; the same coefficients scale `Δ⁺`. The blend's own rounding
adds `4u (|1 − t| |p| + |t| |q|)`. Extrapolation therefore widens the bound by the coefficients'
size and never leaves it unstated.

**`Δ⁺`, the departure along any normal.** It is `Δ` with each `|C^T n|_1` replaced by
`√3 ‖C‖_F`, and the rounding term by the largest length of `|outer| (|inner| |p|)`, all times
`(1 + 10^-12)`.

**Tolerance and departure from a plane.**

- `dep_P` is the largest `|n_P · p − e_P| + 4u (|n_P| · |p| + |e_P|)` over the record's points, plus
  the plane offset's bound. It is computed on the unrounded plane `P`.
- A record's **tolerance** along `n` is `τ(n) = Δ(n) + dep`, and along any normal
  `τ⁺ = Δ⁺ + dep`. `dep` is the departure from its own plane for a flat record, and 0 otherwise.

### What a move does to a key

- **Key reach.** On a world plane `P`, `kr(P, δ) = δ / (|D_P| − δ)`, infinite where `|D_P| <= δ`.
- **Depth share.** Along the sight axis, for points at rotated depths `z_i`,
  `ds(δ) = max_i δ / (d − z_i − δ)`, infinite where a denominator is not positive. Under the
  exponential option it is `δ / (d − δ)`.
- **Screen shift.** Take a point whose inputs move by `a_0`, `a_1` along the camera's first two
  axes, with factor `f` and factor share `σ`. For a projected point `q`,
  `x_k = |f z0| (1 + σ) a_k + |q_k| σ`; for a point not projected, `x_k = a_k`. Then:
  - `ε(q) = hypot(|xx| x_0, |yy| y_0) (1 + 64u)`, infinite for an infinite `σ`;
  - for a record, `ε` is the largest over its points, with `a_k = Δ(R_k)` and
    `σ = ds(Δ(R_2))` (`R_k` the rotation's rows, or the points' own axes for a record not
    projected).

### `S`, the input bound, by key (`band_input@1`)

- **On a world plane `P`.** This covers a flat path's fill, any layer's row on its own plane, a flat
  triangle, and an unregistered plane. Read at the world point `x` on `P` under the read, the
  bound is
  `S⁻ = S⁺ = (|o| + R) kr(P, δ)` with `δ = τ_P(n_P) + θ_P · dist(x, H)`.
  - `H` is the hull of the record's points on `P`.
  - `θ_P = 2 τ_P / w_P`, where `w_P` is `H`'s least width.
  - Reads inside `H` take `dist = 0`. Every fill read is inside `H`, including fringe reads, which
    land on the outline.
  - Where a read lies outside `H` and `w_P <= 2 τ_P`, the bound is infinite (above).
- **On the screen plane through a triangle's corners.**
  `S⁻ = S⁺ = σ max_i |k_i| + G ε`, where `G` is the plane's gradient on screen. A moved triangle
  that still covers the sample maps it, by the affine map between the two triangles, to a point
  within `ε` of it. Where the moved triangle may not cover it, the sample is `Either` (§3).
- **Along a curve.** Take a read at `P` whose model read is `o`. Let `d` be `P`'s distance from the
  centreline and `r = max(h, d + ε) + ε`. Let `K` be the keys of every centreline point within `r`
  of `P`.
  - Then `S⁻ = o − min K + σ max |K|` and `S⁺ = max K − o + σ max |K|`.
  - Why this holds: a centreline whose vertices move by at most `ε` on screen, with keys moving by
    at most `σ` of themselves, reads at `P` the key of a moved point whose unmoved twin (same
    segment, same parameter) lies within `r` of `P`. A moved segment holding `P` has its twin within
    `h + ε` of `P`. The moved nearest segment lies within `d + ε` of `P`, so its twin lies within
    `d + 2ε`.
  - This bounds the reads across nearest-segment and body switches, the discontinuities a local
    gradient misses.
  - **Where the centreline itself may be rebuilt (`curve_recut`).** The twin argument holds only
    while input motion keeps every cubic's count `n` and every joint's join. Two changes break it,
    and a read exposed to either is `truth_unable`, never bounded (`ε̂` and `room` as `lines@1`
    defines them, §7):
    - **A count change.** A cubic's `L` moves by at most `6ε` under input motion, so its `n` may
      change where the cubic is exposed as `lines@1` defines it (§7):
      `|L − 2 clamp(round(L / 2), 1, 63)| <= 6ε̂ (1 + 1e-9) + 2 room(L)`. A read is exposed where
      such a cubic's control hull lies within `r` of `P`.
    - **A join flip.** CE's equality test that joins a cubic to the one before it, or splits them,
      may flip where, on either screen axis, the two points' separation is within
      `(2 + 10^-5) ε̂ + room(0)` of that axis's tolerance in pixels. A flip swaps the joint's
      vertex between the previous cubic's last vertex and the cubic's own first point, whose keys
      may differ by any amount (§5), and so changes the key along the cubic's whole first segment.
      A read is exposed where either version of that segment lies within `r + ε̂ + room(0)` of
      `P`: the segment from the previous cubic's last vertex to the cubic's `t = 1/n` vertex, and
      the segment from the cubic's own first point to that vertex.

    Both need the scene precision to bring a boundary within a few `ε`, so the class is rare; it is
    counted under its own name.
- **On a facing plane.**
  - For a label: `S⁻ = S⁺ = (|o| + R) ds(Δ_c)`, where `Δ_c = (g/2) |R_2|_1` for a label's centre
    as carried.
  - For a point of a `points` record: `Δ_c = Δ(R_2)` through the record's placements. The point is
    placed before it is keyed.
- **A `Nearest` face.** As on a world plane, its own fitted plane.
- **A stroke adjusted on its own fill.** The hull of the two keys' intervals.

### `R`, the arithmetic bound, by key (`band_arith@1`)

**General terms.**

- Every `R` includes `ν = 2^-44 (|o| + 1)`, which covers last-bit differences between conforming
  evaluations of the formulas below.
- A plane read at a pixel `(x, y)` adds its reading's rounding,
  `ρ(x, y) = 4u (|a| + |b x| + |c y|)`, taken at the pixel's far corner `(x + 1, y + 1)`.
- One `R` serves the whole pixel per fragment. For a curve, it is the largest over the 16 samples
  and the pixel's centre.

**A world plane keyed on screen.** Take the region the record is read in:

- for a path, the screen box of its shown points widened by `h + 2` px, where `h` is half its
  widest stroke in pixels;
- for a triangle, its screen box;
- for a facing plane, `[−|x0|, −|y0|, 3|x0|, 3|y0|]`.

`R` is the largest, at the region's four corners `(X, Y)`, of `gap + A + shift`:

- **`gap`** is the largest `|K_r(X, Y) − K_u(X, Y)| + 4u (terms of both reads)`, where `K_r` is the
  key of a rounded plane a conforming computation can produce, and `K_u` is the key of the unrounded
  plane:
  - The oracle's own rounding gives one rounded plane.
  - For any coefficient `v` with `v / 2^-20` within `2^-30` of a half-integer, the other
    neighbouring grid value is also possible, so a conforming computation may round the other way.
- **`A`** is `(dN + f dD) / |D| + 4u f`, with:
  - `s = ((X − x0) / (z0 xx), (Y − y0) / (z0 yy), d)`, `t_x = g_x (X − x0)`, `t_y = g_y (Y − y0)`;
  - `dn = 8u |m|`, `de = 4u (|e| + |m| |c|)`;
  - `dD = de + dn |eye| + 4u |m_2 d|`, with `eye = c + d R_2`;
  - `f = |m_2 d − t_x − t_y| / |D|`;
  - `dN = dn |s| + 4u (|t_x| + |t_y| + |m_2 d|)`;
  - all of the unrounded plane.
- **`shift`** is `f ε_e / (|D| − ε_e)` for the offset's bound `ε_e`.

Then add `ρ`.

**A plane through three keyed corners.** With corner keys `d0`, `d1`, `d2` and corner bounds
`err_i`, let `in = max err_i`:

- **All three keys equal:** the plane is the constant `d0`, with bound `in`.
- **A key not finite:** the plane is the constant `min(d0, d1, d2)`, with bound 0. A `−∞` key is
  then the non-finite case above.
- **Otherwise,** write `e1 = c1 − c0` and `e2 = c2 − c0` on screen, and
  `det = e1x e2y − e2x e1y`.
  - **`det` zero or not finite:** the plane is the constant `mean = ((d0 + d1) + d2) / 3`, with bound
    `spread + in + 4u |mean|`, where `spread` is the largest `|d_i − mean|`.
  - **Otherwise:** with `g1 = d1 − d0` and `g2 = d2 − d0`, the bound is
    `in + max(d_b |e1x| + d_c |e1y|, d_b |e2x| + d_c |e2y|) + 3u (|d0| + |b x0| + |c y0|)`, where:
    - `d_det = 4u (|e1x e2y| + |e2x e1y|)`;
    - `d_b = (4u (|g1 e2y| + |g2 e1y|) + |b| d_det) / |det| + u |b|`;
    - `d_c = (4u (|e1x g2| + |e2x g1|) + |c| d_det) / |det| + u |c|`.

Then add `ρ`.

**A projected point's key**, written `key_error`. It is `u (2 |f| + 4 f² / |d| r)`, where `f` is
the point's factor, `(x, y, z)` its shown position, and `r = |z| + hypot(x, y) / |f z0| + |c|`.

**A curve's key at a read.** It is the segment's error `e` (its vertices' larger, §5), plus:

- for an interior foot, `4u |k_B − k_A| (1 + t') + 3u (|k_A| + |b_x A_x| + |b_y A_y|)` plus that
  plane's reading rounding at `P`, where `t'` is `P`'s distance from `A` over the segment's length;
- for a segment of no length or one key, `|k_A − k_B|`;
- for the front-most of several holding segments, the largest of their errors.

The pixel's bound is the largest over the 16 samples and the centre. The own-fill adjustment is
applied after it.

**A `Nearest` face.** It is its plane's bound plus its outline's departure: the largest over its
subpaths' points `q`, and each closed subpath's closing point read at its first point's position, of

    gap_q + ρ(q) + key_error(q) + 4u |k_q| + u gap_q,    gap_q = |plane(map(q)) − k_q|

plus `hypot(b, c) · 0.1` for the flattening.

**A stroke adjusted on its own fill.** As `stroke_on_own_fill@1` states.

### Contract test (soundness of `R`)

On every probe fixture, at every pixel and sample where the core reports a fragment with a finite
key:

- the core's own error bound, for its own row and for every pair-dependent row the oracle derives
  from XF, is at most `R_f(k)`;
- its key lies within `2 R_f(k)` of `o_f(k)`.

**Scope.** The rows tested are the fragment's own row and each row read on a plane the oracle
settles from XF alone: an own plane, a flat path's fill plane, and a registered plane both records
share in every scenario (§7). A **planeless record that may share** a registered plane (§7, the
may-share set) has attached rows whose plane depends on which identity the registry settles; the
oracle does not settle it and the test never reads the core's choice. Those rows are left out of
the numeric test, and their pairs are covered by the class that admits every edge (§7). Their own
rows stay in.

For keys of `−∞`, the test asserts equality. Once the band is proven (How to read it), an excess is
a finding about the core, listed by class, and the band is never widened to fit it. Before then, an
excess is triaged by name as a contract defect or a core defect.

**Rejects:** a mesh face whose plane is shifted, against a surface it shares no registered plane
with, by more than its interval (snap precision plus arithmetic), so that the order at a crossing
sample reverses. Also a line moved behind or in front of a surface by more than its curve
interval.

## 7. The order at a sample: tie policies

**Core rule: the relation of a pair at a sample.** Take two fragments `a` and `b` both covering
sample `k`, read at keys `d_a` and `d_b` with bounds `e_a` and `e_b` (each read against the other,
§5). The core's order between them is **weak** (paint order decides) where:

- the pair is `own_layers`;
- or `tied(d_a, e_a, d_b, e_b)`;
- or `coplanar_tied` holds.

Otherwise it is **strong**, with the nearer in front.

`tied(a, ea, b, eb)` holds where `a == b`, or where both are finite and `|a − b| <= ea + eb`. Keys
that are not finite tie only with themselves, and this is tested before any subtraction.

[Rests on: `order::at_sample`, `strong_at`, `tied`;
`order.rs::depths_that_are_not_finite_tie_only_with_themselves_before_any_subtraction`,
`at_one_sub_sample_the_strong_orders_never_cycle`,
`a_stroke_read_two_ways_settles_a_cycle_at_one_sub_sample_by_wins_then_paint_order`.]

**`paint_order@1`.** Of two fragments, the one later in the order is in front. On one order entry,
the one with the higher `sub` is in front. Every weak order is paint order. [Rests on:
`order::painted`.]

**`own_layers@1`.**

- **The rule.** A path's fill and one of its own strokes or background strokes keep paint order
  whatever their keys: the stroke over its fill, the background stroke under it.
- **Eligibility.** Same producer, same order entry, same path `sub div 3`, different layers, and one
  of them is the fill.
- **Two strokes** of one path keep depth.

[Rests on: `order::own_layers`;
`order.rs::a_path_s_fill_and_its_own_strokes_keep_paint_order_and_two_strokes_keep_depth`,
`tests/path.rs::a_grazing_face_s_stroke_is_over_its_own_fill_wherever_the_stroke_covers_a_pixel`.]

**Rejects:** a stroke drawn under its own fill at a sample both cover.

### `one_plane@1`

**Core rule.** The frame's **registry** partitions the registry-offered records (§5) into
identities.

- **Flat records.**
  - Each flat record has a tolerance `τ(n)` along every unit normal (§6).
  - The flat records are taken in a canonical order, least `spread` first.
  - Each joins the first identity, among those whose planes lie near its own, for which one plane
    holds every member's points and its own, each within its own tolerance along that plane's
    normal. That plane is the identity's own, or a minimax plane fitted to them all.
  - A record that joins no identity founds one.
- **Records with no plane of its own.** Once the partition is made, each attaches to every flat
  identity that meets three conditions:
  - one plane holds its points and every member's, each within its own tolerance;
  - its points stand off the identity's plane by no more than a bound that grows with distance
    from the members' hull;
  - it enters the members' hull.

  Of these identities, it keeps the first four.
- **The axis rule.** A record with no plane of its own that attaches to none shares an identity
  with the others on each axis plane it holds, with no tolerance. The plane is:
  - the one its writer coded, or, for a triangle, the one its corners share within the grid, taken
    through its first point;
  - and, for a path, each axis whose coordinate all its points hold exactly.

  Two such records share where the axis is the same and their first points' coordinates on it are
  equal.
- **On one plane.** Two fragments are on **one plane** where their identity lists meet.
- **`within_term`.** There they tie where `within_term` holds. Each one's share `s` is its `wide`
  where both carry one, and otherwise its `snap`. The term is
  `tied(d_a, e_a + (|d_a| + e_a) s_a, d_b, e_b + (|d_b| + e_b) s_b)`, or either share is infinite.
- **The shares.**
  - A flat record's layers, and a flat triangle: `snap = kr(P, τ_P(n_P))` on its own plane, and no
    `wide`.
  - A record with no plane of its own attached to flat identities: `snap` and `wide` come from how
    far its points stand off each identity's plane, taken in the members' hull for `snap`. Either
    can be infinite.
  - One on axis planes alone: the largest over its keyed axis planes of `kr(A, Δ(n_A) + dep_A)`.

The partition's ordering details are the core's means of choosing one partition among those its
tolerances allow. The contract does not use them, because the oracle's set below contains every
outcome. Those details are the canonical order's tie-breaks, the lookup buckets, the refit rule and
the attach bound.

[Rests on: `depth::Registry` (`settle`, `near`, `joins`, `fit`, `contains`, `enters`,
`off_members`, `off_identity`), `order::one_plane`, `coplanar_tied`, `within_term`;
`order.rs::on_one_plane_a_zero_share_parts_beyond_the_errors_and_an_unbounded_one_ties`,
`a_record_with_no_plane_of_its_own_takes_its_whole_share_only_against_another`,
`tests/path.rs::a_path_on_a_meshs_plane_ties_with_it_by_draw_order`,
`::a_line_on_a_tilted_plane_ties_with_it_by_draw_order`,
`tests/mesh.rs::coplanar_meshes_tie_by_draw_order_the_later_on_top`; `depth.rs`
`each_member_is_held_within_its_own_tolerance`, `the_partition_does_not_depend_on_the_order_offered`.]

**Oracle reading: the set of pairs that may share an identity.** Two fragments **may share** an
identity where both records are registry-offered, both may retain an identity (§5), and one of these
holds:

1. **Width.** For a pair at least one of whose records is flat:
   `w(A ∪ B) <= 2 max(τ⁺_A, τ⁺_B) (1 + 2^-40)`, where `w` is the minimum width of the two records'
   points together. An implementation may use any lower bound of `w`; that only grows the set. It
   must still fail the test on every §12 witness that relies on failing it (W4, W9, W10, W11, W12
   and W16), whose true widths are at least 1 frame unit against tolerances under `10^-4`; any
   lower bound within a factor 2 of `w` does. (W17 relies on the opposite: its records all lie on
   `z = 0`, so the width test passes wherever it is evaluated, and its discrimination rests on
   which records are offered.)
2. **A third record.** For a pair with a record of no plane of its own: some flat registry-offered
   record `M` passes (1) with each of them. `M` may be either of them.
3. **The axis rule.** Both lie on one axis plane by the axis rule above: the same axis, and first
   points equal on it.

Only flat records found identities, so two records with no plane of their own share one only
through a flat record, (2), or by the axis rule, (3); the width test alone never joins them.

Why this contains every outcome:

- Every pair the core puts on one identity passes. Members of one identity are each held by its
  final plane within their own tolerances, which bounds the width of their union by twice the larger
  tolerance.
- An attached record is held together with every member by one plane. That gives (1) with each
  member, and gives (2) for two records attached to one identity, which need not fit one plane
  together.
- Axis-plane sharing is exactly (3).

**Oracle reading: what may share admits.**

- **Two flat records** (paths and triangles). Their rows on a shared identity are their own planes
  read at the sample, which the oracle computes exactly. Two scenarios are possible:
  - **Shared:** each reads its row, and the term is `within_term` with the snap shares.
  - **Not shared:** each reads its own key, with fringe, and `near_parallel@1` or plain depth
    applies.

  The admitted edges are the union over both scenarios.
- **A record with no plane of its own, through a flat identity** ((1) or (2)). Its rows lie on
  identity planes that the partition fixes. So the oracle admits every edge (weak, and strong either
  way) for such a pair, and labels it `Policy(one_plane@1)`. This class is outside what the oracle
  promises (§12).
- **Records with no plane of their own, by the axis rule alone** ((3), and no flat record passes
  (1) with either). The axis rule then holds in every scenario, and its plane is the axis plane
  itself, which the oracle computes exactly: each reads its row on that plane at the sample, and
  the term is `within_term` with their axis shares. Two such rows read at one sample are one
  computation, so the pair is an `ExactTie` (§7) wherever both rows are read there.

**Rejects:** a line crossing a surface beyond the width test (its points standing off the surface's
plane by more than twice the larger tolerance), drawn over the surface where it is behind it.

### `near_parallel@1`

**Core rule.** This applies to two fragments of flat registry records (a flat path's layers, a flat
triangle) that are not on one plane. They tie under `within_term`, with their own keys and snap
shares, where their own planes are near-parallel.

Take the unrounded normals `n_A` and `n_B`, with `c = n_A × n_B` and `s = |c|`. Then:

- `θ = atan2(s, |n_A · n_B|)`;
- `high = θ (1 + 16u) + 16u`;
- `off = 40u / s + 16u` where `s >= 32u`, otherwise infinite;
- the axis is `a = c / s`.

For each record `R`, with `τ_R` its tolerance along its own normal and `first` its first point:

- `t_p = (p − first) · (n_R × a)` over its points, with `lo` and `hi` their least and greatest, and
  `far` the largest `|p − first|`;
- `extent = hi − lo` where `s > 0`, otherwise `2 far`;
- `read = extent + 2 far (off + 16u)` where `off <= 1`, otherwise infinite;
- `span = min(read, 2 far) (1 + 8u)`;
- `θ_R = 2 τ_R / extent` and `θ_R⁻ = (2 τ_R / span)(1 − 2u)`.

They are near-parallel where `high <= (θ_A⁻ + θ_B⁻)(1 − 2u)`. The test is pairwise and symmetric.
It gives no identity, and it does not chain.

[Rests on: `depth::near_parallel`, `tested`, `order::near_parallel`;
`depth.rs::faces_are_near_parallel_up_to_the_turn_their_snaps_allow_and_not_past_it`,
`the_near_parallel_test_is_the_same_whichever_record_comes_first`,
`a_record_with_no_plane_of_its_own_is_never_tested_for_near_parallel`.]

**Oracle reading.** The oracle evaluates the test as stated. Where the two sides lie within
`2^-40 (high + 1)` of each other, it admits both outcomes.

**Rejects:** two faces meeting at a dihedral well past `θ_A + θ_B`, their order reversed at a sample
beyond their intervals.

### `lines@1`

**Core rule.** Two strokes (or background strokes) of registry-offered paths with no plane of their
own, not on one plane, tie where

    tied(d_a, e_a + (|d_a| + e_a) o_a, d_b, e_b + (|d_b| + e_b) o_b)

Each fragment's own share `o` at the pixel is

    o = σ + most / least · (1 + 4u)

infinite where `least` is 0. In this formula:

- `σ` is the path's depth share of the snap's departure along its sight axis (`ds(Δ(R_2))`).
- `most` is the largest, over the pixel's 16 samples and its centre, of how far the core bounds a
  read's move when the snap moves the curve on screen.
- `least` is the least **magnitude** among those reads.

Keys are negative, so the magnitude is what divides. Fills never carry a share.
`Stream.own_shares` reports each fragment's share.

The core's bound on a read's move covers four kinds of move:

- nearest and body switches;
- the own-fill read;
- joint slack;
- a step of a cubic's piece count where the snap may cross one.

It states two limits:

- a cubic curved on screen moves its pieces at a step;
- under the exponential option the pieces' keys depart from the curve's.

[Rests on: `order::lines`, `lines_term`, `Paths::own_at`, `Centreline::slides_among`,
`slide_under`, `step`, `slack`.]

**Oracle reading.** The oracle bounds the core's reported share from above, from XF, term by term
along the core's bound on a read's move. Write `ε̂ = ε (1 + 2^-40)` and `σ̂ = σ (1 + 2^-40)`, with
`ε` the record's screen shift and `σ` its depth share (§6), and `room(x) = 128u max(scale, x, 1)`,
where `scale` is the largest screen coordinate magnitude of the centreline's vertices (§5).

- **Joints.** Every cubic but the path's first starts a joint with the cubic before it in the point
  list. Its `gap` is the screen distance from the previous cubic's last vertex to the cubic's own
  first point, its `tol` is `hypot((1e-6 + 1e-5 |S_0|) |xx|, (1e-6 + 1e-5 |S_1|) |yy|) (1 + 1e-9)`
  at that first point `S`, and its `keygap` is the two points' key difference. CE's test (§4) on
  the shown points says whether the drawn frame joins or splits them; the oracle evaluates it as
  the core does, on the same shown points.
- **Reach.** `slack⁺` is the largest `gap + 2·10^-5 ε̂` over the joints that are joined, and the
  split ones with `gap − tol <= (2 + 2·10^-5) ε̂ + 2^-30 (gap + tol)`; 0 where there are none.
  For pixel `(x, y)`, `reach⁺ = ε̂ + slack⁺ + room(max(x, y) + 1)`.
- **The segments a read may take.** For each of the 17 reads `P` and each segment `s`: `d_s` is
  `P`'s distance from it, `κ_s` its key at `P`'s foot clamped to its ends, `spread_s` its two keys'
  difference and `kmax_s` their larger magnitude. With `d_min` the least `d_s`,
  `Σ(P) = { s : d_s <= max(h + reach⁺, d_min + 2 reach⁺) + room(max(x, y) + 1) }`.
  This holds every segment the core's bound reads at `P`: its holders, the ones that may hold `P`
  after the move, and the near-nearest, for the stroke's read and the own fill's alike.
- **A segment's move.**
  `moves⁺_s = (1 + σ̂) spread_s + σ̂ kmax_s + slack_s + step⁺_s`, where `slack_s` is
  `(1 + σ̂) keygap` for a segment that starts a joint and 0 otherwise.
- **A step of a cubic's count (`step⁺_s`).** For a segment that is a piece of a cubic, take the
  cubic's control points on screen `X_0..X_3` and keys `k_0..k_3`; its `L` (§5); its chord
  `v = |X_3 − X_0|`; `far`, the larger of `|X_1 − X_0|` and `|X_2 − X_0|`; for `j = 1, 2`,
  `along_j = ((X_j − X_0) · (X_3 − X_0)) / v²` and `r_j = k_j − k_0 − (k_3 − k_0) along_j`
  (both 0 where `v = 0`); `low` and `high`, the least and most of 0, `r_1` and `r_2`;
  `kmax = max |k_j|`, `kends = |k_3 − k_0|`, `span = max k_j − min k_j`; `E = σ̂ (1 + σ̂) kmax`
  and `lean = 2ε̂`.
  - The cubic is **exposed** where `|L − 2 clamp(round(L / 2), 1, 63)| <= 6ε̂ (1 + 1e-9) +
    2 room(L)`. Otherwise `step⁺_s = 0`, and for a segment of no cubic it is 0.
  - For an exposed cubic, `step⁺_s = (1 + 16u)(max(A, B, C) + 72u (kmax + span))(1 + 2^-40)`:
    - `A = span + 2E`;
    - where `v > 2 lean`, `B = max(1 + σ̂, 2)(high − low + eps)` with
      `eps = 128u (kmax + kends (1 + max_j |along_j|)(1 + scale / v))`, and
      `C = high − low + 2 (2E (1 + wide) + kends du)` with `short = v − lean`,
      `du = (lean (v + far) + lean²) / short² + far v (2v + lean) lean / (v² short²)` and
      `wide = max_j |along_j| + du`;
    - where `v <= 2 lean`, `B = C = 0`.

    These are the core's three charges (a chord short enough to turn freely, a cubic collinear in
    place and key, and the band), each taken whichever branch the core takes. The core's choice of
    the lesser of the band and the adjacent-count departure never exceeds the band. The extra `8u`
    in the rounding term covers last-bit differences between conforming evaluations of `r_j`.
- **A read's bound.** `slide⁺(P) = ((1 + σ̂)(J(P) + 8u max_Σ |κ|) + M(P))(1 + 64u)`, where `J(P)`
  is the most less the least `κ_s` over `Σ(P)` and `M(P)` the largest `moves⁺_s` there.
- **The share.** With `L⁻` the least `|κ_s|` over every read's `Σ(P)`,

      O⁺ = (σ̂ + (1 + 4u) max_P slide⁺(P) / L⁻)(1 + 2^-40)

  infinite where `L⁻ = 0`.

Why it bounds the core's share: each term is the core's own charge with every condition that
lowers it dropped, at the oracle's values of the same quantities widened by their last bits; the
core's read and its low and high keys are keys of segments in `Σ(P)`, so its jump is at most `J`;
and its magnitude is a key in `Σ(P)`, at least `L⁻`. On a straight cubic of four keys `−1`,
controls 6 px apart, unplaced, the core charges `320u (1 + 16u)` at the step, and `B` gives
`step⁺ >= 328u` (witness `lines_straight_cubic_step`, §12).

The pair admits the weak edge where it is eligible and

    |o_a − o_b| <= β_a + β_b + (|o_a| + 2R_a) O⁺_a + (|o_b| + 2R_b) O⁺_b

where `β_f = max(S⁻, S⁺) + 3R`. The contract test checks the core's reported share against `O⁺`.
Once the band is proven (How to read it), an excess is the core finding `share_beyond_bound`.
Before then, an excess is first a question about this bound: it is triaged by name as a contract
defect (the bound misses a charge) or a core defect (the core charges what it does not need).

**Rejects:** two lines crossing at depths apart by more than their bounds, their order reversed.

`coplanar_tied` is `one_plane && within_term`, else `lines && lines_term`, else (both flat records)
`within_term && near_parallel`. The three cases are disjoint.

### `nearest_plane_departure@1`

A face of a face group on no one plane (`Nearest`) is keyed on the plane nearest it. Its bound
carries its outline's departure from that plane (§6), so it ties with what that plane cannot tell it
from. It is never offered to the registry.

[Rests on: `path::nearest_plane_departure`.]

### One order for the pixel (`majority@1`)

**Core rule.** Each pair's order at the pixel comes from the samples both cover:

- **`Paint`** (paint order): none shared, or a fill against its own stroke.
- **`Tie`** (paint order): every shared sample tied (`tied` or `coplanar_tied`).
- **`Depth`:** every untied shared sample has the same nearer, and that one is in front.
- **`Majority`:** the untied samples disagree. The one nearer at more of them is in front, with
  paint order on an even split.

`Depth` and `Majority` are strong; `Paint` and `Tie` are weak. This order drives the opaque drop,
the units and the coverage resolve (§9). It never decides a sample's own order.

[Rests on: `order::pair`, `Shared::decided`;
`order.rs::a_pair_crossing_inside_the_pixel_takes_the_order_most_shared_sub_samples_give`,
`a_stroke_one_ulp_nearer_where_it_does_not_cover_keeps_paint_order`.]

### `cycle_rank@1`: the rank

**Core rule.** Pairwise orders are edges, strong or weak.

1. **One order.** Insert each fragment before the first one it is in front of. Where every pair
   agrees with the resulting order, that is the order.
2. **Otherwise,** build it front first:
   - among the fragments that no remaining fragment is strongly in front of, take the later
     painted;
   - where every remaining fragment has a strong edge into it (the strong edges alone cycle), take,
     at a sample, the one in front of the most remaining fragments, the later painted on ties.

So weak edges yield before any strong edge is broken. A paint-order or `own_layers` edge gives way
first. At the pixel, a strong cycle is not broken: the pixel takes the sample resolve (§9).

[Rests on: `order::rank`, `rank_sample`, `rank_through_cycles`;
`order.rs::a_cycle_through_a_weak_order_keeps_every_strong_order`,
`the_report_breaks_a_cycle_of_strong_orders_by_wins_then_paint_order`,
`a_pixel_whose_strong_orders_cycle_composites_each_sub_sample_in_its_own_order`.]

### Oracle reading: the edges a pair admits

For each pair present at `k` (each `Yes`, or `Either` and present in the core's report), the oracle
writes a set `E` of admitted edges, drawn from: weak (paint order), strong `a`-front, and strong
`b`-front. It also writes a label.

1. **`own_layers@1`**, exactly by its eligibility: `E = {weak}`.
2. **`ExactTie`**: `E = {weak}`. This applies where every scenario the sets allow gives the two keys
   bit for bit:
   - both keys are `−∞`;
   - or both fragments are fills of flat paths, or flat triangles, whose planarity is the axis plane
     of one coordinate, with their records' first points equal on it, and each is read at the sample
     itself in every scenario. A fragment is read at the sample itself when it is a triangle, or the
     sample lies inside its outline by the point-in-outline test, or its plane's key is constant on
     screen;
   - or both are records with no plane of their own sharing an axis plane by the axis rule alone,
     each read on its row on that plane at the sample (above). Two overlapping lines on `z = 0`
     are this case.

   These bullets are examples. The sentence above them is the rule, and any other pair whose keys
   are one computation in every scenario is an `ExactTie` too.
3. **A pair that may share an identity with a record of no plane of its own, through a flat
   identity:** every edge (`one_plane@1`, above).
4. **Otherwise**, `E` is the union, over every scenario of §5's pair-dependent reads that §7's sets
   allow (rows or own keys), of three parts, each computed with that scenario's keys and intervals:
   - **strong `a`-front** where `min I_a < max I_b`, and likewise for `b`;
   - **weak** where `I_a` and `I_b` meet;
   - **weak** where a policy may hold. With `β_f = max(S⁻, S⁺) + 3R` and the policy's shares `s`,
     the policy's term is taken as
     `|o_a − o_b| <= β_a + β_b + (|o_a| + 2R_a) s_a + (|o_b| + 2R_b) s_b`. This is
     `one_plane@1` with the snap shares, in the shared scenario of a pair that may share; or
     `near_parallel@1` with the snap shares, where its test may hold; or `lines@1` with `O⁺`.

The label is decided in this order:

- `Depth(front)` where `E` is the single strong edge;
- `TieBand` where the intervals meet;
- otherwise `Policy(name@1)`, naming the first policy that admits the weak edge in the core's order
  (`one_plane`, `lines`, `near_parallel`).

**Rejects:** two overlapping layers whose keys are both `−∞` (a 2D scene), and two glyph fills on one
axis plane inside both outlines, each pair with its paint order reversed.

### Oracle reading: the admissible orders at a sample

- **Completions.** A **completion** picks one edge from each present pair's `E`.
- **Admissible orders.** An order of the present fragments is **admissible** where the rank
  (`cycle_rank@1`, through cycles) of some completion produces it.
- **A sufficient test.** An order is admissible if every pair agrees with some edge in its `E`. The
  agreeing edge is a strong edge in that order's direction, or a weak edge whose paint order matches.
  A comparator may test this first.
- **Cycles.** Other orders may be admissible through a completion whose weak edges yield in a cycle.
- **Every pair.** The check covers every pair, not adjacent pairs alone. Intervals are not
  transitive: keys 0, 1 and 2 with half-widths 0.6 let the adjacent pairs meet while the ends are
  strong.
- **Mismatch.** A sample's order that no completion produces is a `mismatch`.

**Rejects:** an order that breaks a strong edge where a weak edge could have yielded.

## 8. Stroke policies

A path's strokes are kurbo's `stroke` of its outline in frame units, at tolerance `0.1 / s`, then
mapped (§4).

**`width_in_scene_units@1`.**

- The width is CE's stroke width times the view's `cairo_line_width_multiple`, in frame units, as
  Cairo strokes in user space. CE's stroke width is the record's, a face's own (XF 1.3), or a paint
  blend's mix (`paint_rows@1`).
- The frame mapping scales it. The perspective never does.
- A width not above 0, or NaN, draws nothing.

[Rests on: `tests/path.rs::a_stroke_covers_its_outline_at_ces_width`.]

**`miter_limit_10@1`.**

- Joints `auto` and `miter` are kurbo's miter; `round` and `bevel` are kurbo's own.
- Caps `auto` and `butt` are butt; `round` and `square` are kurbo's own.
- The miter limit is Cairo's, 10.

[Rests on: `path::MITER_LIMIT`, `kurbo_join`, `kurbo_cap`;
`tests/path.rs::a_sharp_joint_is_mitred_up_to_cairos_limit`.]

**`join_mask@1`.**

- **The joins field.** A path record may carry the XF 1.4 `joins` field: one byte per cubic of its
  points, cubic `c` being points `4c .. 4c + 3` of the whole record. Bit 0 means Cairo joins into
  the cubic; bit 1 means it mitres there. A byte with bit 0 clear, or a cubic with no byte, decides
  nothing.
- **When edits apply.** Only to a stroke whose style is kurbo's miter, that is, joints `auto` and
  `miter`, and only where the record carries the field. A round or bevel stroke is never edited.

**The carried decisions.** They are read from the outline as §4 builds it (each subpath of at
least four points, a closed one ending exactly at its first point). For each subpath, walk its
cubics in order, keeping `p0`, the end of the last cubic kept (at first the subpath's first point):

1. A cubic is `[p0, c1, c2, c3]`, with `c3` the subpath's first point for a closed subpath's last
   cubic. A cubic whose `c1`, `c2` and `c3` all equal `p0` is skipped: it makes no join, and the
   join after it is from the cubic before it.
2. The first cubic kept is the subpath's **first**. Each later one makes the join **from** the
   last cubic kept **into** it, decided by its own byte.
3. A closed subpath also makes its **closing join**, from its last cubic kept into its first,
   decided by the byte of the subpath's first cubic (its first in the points, kept or not).

A join's **key** is the bit patterns of its two cubics' eight points (`x` and `y`, the four points
of the one it leaves, then the four of the one it enters) and whether it is a closing join. The
decisions of one key form a list, one per occurrence, in the order the walk meets them; an
undecided occurrence keeps its place in the list.

**The joins kurbo makes.** The stroke is kurbo's `stroke` of the path handed to it: the guarded
outline, or the dashes (§4, and the dash rule below), in frame units at tolerance `0.1 / s`. The
oracle walks that path's elements as kurbo's stroker does, with `join_thresh = 2 · tolerance /
width` and `limit² = pow(10, 2)`:

- **Elements and their tangents.**
  - A line to its own start, and a cubic all of whose points equal its start, are skipped.
  - A line's tangent at both ends is its end less its start; its points as a key are `[p0, p0, p0,
    p1]`.
  - A cubic's start tangent is `c1 − p0`, or where that has squared length at most `10^-12`,
    `c2 − p0`, or where that does too, `c3 − p0`. Its end tangent is `c3 − c2`, or `c3 − c1`, or
    `c3 − p0`, by the same threshold.
  - A quadratic's start tangent is `c1 − p0`, else `c2 − p0`; its end tangent `c2 − c1`, else
    `c2 − p0`, by the same threshold. A quadratic carries no key.
- **A join** is made between each element and the one before it in its subpath, and, at a
  `ClosePath`, first at the closing line where the subpath does not end at its start (a line that
  carries no key), then at the subpath's start, from its last element into its first. With `ab`
  the tangent arriving and `cd` the one leaving, and `h = hypot(ab × cd, ab · cd)`, kurbo **joins**
  only where `ab · cd <= 0` or `|ab × cd| >= h · join_thresh`; a slighter turn makes no join and is
  suppressed. A join kurbo makes **mitres** where `2 h < (h + ab · cd) limit²`. Here `hypot` is
  libm 0.2.16's `hypot`, which kurbo's `do_join` reaches through its float functions, kurbo 0.13.1
  being built without `std`. A join or miter decision is therefore the same on every platform, at
  its boundary too, and the oracle computes it with the same function. A contract test against the
  core's output on Linux and macOS holds the core to libm's decision at corners where libm's
  `hypot` and the platform's disagree.
- **Its key** is its two elements' points, as above, and whether it is the closing join. A join
  with an element of no key (a quadratic, a closing line) has none.
- **Joins inside a near-straight cubic.** A cubic kurbo finds nearly collinear and possibly cusped
  is stroked as lines with round joins of its own (kurbo 0.13.1, `stroke.rs`: `do_cubic`'s linear
  test, then `do_linear`): from its start tangent into its first line, at each cusp inside it, and
  from its last line into its end tangent. These joins are round whatever the style, carry no key
  and are not occurrences, and they are drawn as kurbo draws them; their lines and arcs belong to
  the contours an edit searches. The joins into and out of such a cubic are made with its start
  and end tangents, as for any cubic.
- **Contours.** kurbo's outline holds each stroked subpath's contours in turn, each begun by a move:
  one for an open subpath, two for a closed one.

**Matching.** The oracle takes the joins kurbo makes in order. For each with a key, the next unused
decision of its key's list applies, and is then used; occurrences are counted only over the joins
kurbo makes. A join whose key has no list, or whose list is used up, is left as kurbo drew it, as is
an occurrence that decides nothing.

**The edits**, where a decision differs from kurbo's own, within the join's subpath's contours (the
elements from its first contour's move to the next subpath's first move). With `p0` the join's
point, and these quantities computed as kurbo's `do_join` computes them (`stroke.rs`):

- `|v| = sqrt(v_x · v_x + v_y · v_y)`, kurbo's `Vec2::hypot`, not libm's `hypot` (the edit's
  `|m − p0|` too);
- `norm(t) = (0.5 width / |t|) (−t_y, t_x)`: the scalar first, then each coordinate times it;
  `n_a = norm(ab)`, `n = norm(cd)`;
- `c = ab × cd`, and `s = −1` where `c > 0`, else `1`;
- `f_a = p0 + s n_a` and `f = p0 + s n` (that is, `p0 − n_a` and `p0 − n` where `s = −1`);
- the miter point `m = f − cd · k`, with `k = (ab × (f − f_a)) / c` computed first and `cd` then
  multiplied by it coordinate by coordinate.

The edits:

- **Parallel tangents** (`c = 0`): no edit, mitring or not; a carried miter there stays bevelled.
- **A carried bevel where kurbo mitred.** Find the one line to `m` and the one line to `p0` among
  those elements; drop both. kurbo's bevel remains.
- **A carried miter where kurbo bevelled.** The tip is `p0 + (m − p0) · (10 · 0.5 width / |m − p0|)`,
  on the bisector at the limit's length: the scalar left to right, `((10 · 0.5) · width) / |m − p0|`,
  then each coordinate of `m − p0` times it, then added to `p0`. The tip is a vertex the edit
  inserts, never one it looks for. The forward place is the one line to `p0 − n`; the
  backward place is the one line whose element before it ends at `p0 + n`. The outer place is the
  forward one where `c > 0`, else the backward one; the inner place is the other. Insert a line to
  the tip before the outer place and a line to `p0` before the inner place, the later place first.
- **Not found.** Where a line the edit needs is absent, or not the only one, the join is left as
  kurbo drew it.
- **Everything else is kurbo's.** An edit changes only the lines it drops or inserts in its join's
  subpath's contours. Every other element, and every contour the walk accounts for no edited join
  in (a cap, a dot, the round joins inside a near-straight cubic, the contours of a subpath with no
  edit), is left as kurbo drew it.

All of this arithmetic is binary64 in the order written, so the vertices an edit looks for (`m`,
`p0`, `p0 − n`, `p0 + n`) are found by value.

[Rests on: `join_mask.rs` module doc and `apply`, `path::decided_joins`; `tests/join_mask.rs`.]

**`dash_start_least_length@1`.**

- **The entries.** A dash's entries are `a = period · duty` and `b = period · (1 − duty)`.
- **Solid.** A dash of duty 1 strokes solid. So does a dash whose gap is under Cairo's least length,
  `1/512` (half of one unit of Cairo's 24.8 fixed point). The gap is `(a + b) − a`: the total is
  summed first and the on length taken from it, never `b` itself. A gap strictly below `1/512` is
  solid; a gap of exactly `1/512` is dashed. (Cairo 1.18, `cairo-gstate.c`,
  `_cairo_gstate_set_dash`: `dash_total - on_total < CAIRO_FIXED_ERROR_DOUBLE`, the total summing
  the entries that are not 0.)
- **The pattern.** Any other dash is the pattern `[a, b]`. Each subpath starts `phase · period`
  into it.
- **Without the guard,** kurbo's `dash` lays the pattern on the whole unguarded outline, and the
  stroke follows.
- **Where the guard applies (§4),** each subpath is cut into §4's pieces, each marked as missing the
  area or not:
  - **Runs.** A **run** is a maximal sequence of consecutive pieces that do not miss the area. A
    closed subpath has no closing-line piece: §4 ends it exactly at its first point, so its
    `ClosePath` adds no line.
  - **No piece misses.** A subpath none of whose pieces misses the area is dashed whole by kurbo's
    `dash` at offset `phase · period`, as one path of all its pieces that keeps its `ClosePath`
    where the subpath is closed, so the dash runs on across the seam as it does without the guard.
    This holds even where another subpath of the record brought the guard into force.
  - **Dashing a run.** Each run is dashed on its own by kurbo's `dash`. Its offset is
    `phase · period + at`, where `at` is the summed `arclen(1e-6)` of every piece before it in its
    subpath, missing or not. Pieces that miss are measured and never dashed.
  - **The closing seam.** Take a closed subpath with more than one run whose first and last pieces
    both do not miss, and `(x, y)` its first point. Where the first run's first dash starts exactly
    at `(x, y)` (its move equal in both coordinates), it is joined onto the last dash of the last
    run that ends near `(x, y)`: within `tol = 1e-9 (1 + max(|x|, |y|))` of it in each coordinate,
    `|e_x − x| <= tol` and `|e_y − y| <= tol`. Joining appends the opening dash's elements, without
    its move, after that dash's last element. The joined dash is emitted where that dash is, and
    the opening dash is not emitted on its own. Where no dash of the last run ends near `(x, y)`,
    nothing is joined.
  - **Order.** The dashes are emitted in run order, the opening dash of a joined seam inside the
    last run's dash.

[Rests on: `path::drawn_dash`, `dashed`, `dash_subpath`, `DASH_ACCURACY`;
`tests/path.rs::a_dashed_stroke_follows_its_period_duty_and_phase`,
`::a_dashed_stroke_returning_from_far_beyond_the_frame_keeps_its_dash_phase`.]

**`round_cap_dot@1`.**

- **When it applies.** To a face of a face group whose shown points are all at one spot in `x` and
  `y`, under a round cap, and either undashed or dashed on at its subpath's start.
- **Cairo's dash start** (Cairo 1.18, `cairo-gstate.c`, `_cairo_gstate_set_dash`, then
  `cairo-stroke-dash.c`, `_cairo_stroker_dash_start`), with `a` and `b` the entries above and
  `total = a + b`:
  - the offset `o = phase · period` becomes `fmod(o, total)` (C's `fmod`, exact); where that is
    below 0, `total` is added; where it is then at most 0, `o` is 0, so `−0` is 0;
  - the start is on, at entry `a`. While `o > 0` and `o` is at least the current entry, `o` loses
    the entry, on and off swap, and the next entry is taken (after `b`, `a` again);
  - the dash is on at the subpath's start where this ends on. An offset exactly equal to an entry
    steps past it: at `o = a` the start is off, at `o = 0` it is on.
- **What it strokes.** A circle of radius `w / 2` about its first shown point, made by kurbo's
  `Circle::to_path` at `0.1 / s`.
- Under a butt or square cap, the face strokes nothing.

[Rests on: `path::strokes_as_dot`.]

## 9. The resolve text and colour

**Fragments at a pixel.**

- **Order.** The opaque pass's fragments (those whose colour is opaque wherever they paint) come
  first, then the others. Each pass is in producer order (meshes, paths, point blocks), and within a
  producer in the order added.
- **Opaque and full.** `opaque` means alpha 255. **full** means a mask of all 16 samples, or an
  area of 1.
- **Coverage.** A fragment's 8-bit **coverage** is `(bits · 255 + 8) div 16` for a mask, and
  `floor(clamp(area, 0, 1) · 255 + 1/2)` of its binary32 pixel area for a path. It is never the sum
  of its cell bytes.
- **Coverage at a sample.** At sample `k` it is 255 or 0 for a mask, and §4's cell byte for a path.

### `resolve_select@1`

**Core rule.**

1. **No fragment:** the background (`Background`).
2. **One fragment:** if opaque and full, its colour (`Fast`); otherwise
   `blend(colour, coverage, bg)` (`Coverage`).
3. **Several fragments:** compute every fragment's keys (§5, both adjustments) and every pair's
   pixel order (`majority@1`, or paint order on a flat pixel). Rank the pixel (`cycle_rank@1`).
   Where the strong orders alone cycle, take the **sample resolve** over every fragment.
4. **`opaque_drop@1`.** Where the pixel's order holds a fragment that is opaque and full, the first
   such at place `p`:
   - **On a flat pixel,** everything after `p` is dropped.
   - **Otherwise,** each sample ranks the fragments covering it. A fragment is kept in either case:
     - some sample's rank reaches it before, or at, the first opaque, full fragment there;
     - or it covers no sample and stands before `p`.

     The rest are dropped.
   - If a single fragment is kept and it is opaque and full: `Fast`.
5. **`unit_merge@1`.**
   - **Units.** Runs, in the pixel's order, of one surface's mask fragments (same producer and
     surface) whose masks do not overlap form one unit. Every other fragment is a unit on its own.
   - **The tile drop.** Take the **first** unit of two or more fragments that is opaque with a full
     mask union. If, at every sample, the rank shows no fragment of any later unit before the first
     fragment that is opaque and full or belongs to that unit, then **all** later units are dropped.
     Otherwise none is. On a flat pixel they are always dropped.
6. **Counting partials.**
   - Count the units whose mask union is partial.
   - If any is, or any path unit's pixel area is below 1, also count the full mask units whose
     members differ in colour.
   - The **coverage resolve** applies where the count is 0, or 1 with no partial path area, and at
     every sample no fragment of a unit is behind a fragment of a later unit where both cover it.
     "Behind" is that sample's own relation (§7).
   - Otherwise the **sample resolve** applies.
   - A frame of paths alone under a flat camera never takes the sample resolve, because it has no
     partial mask unit. Two partial mask units can take it under any camera.

[Rests on: `composite::resolve_pixel_traced`;
`order.rs::the_cull_keeps_what_the_rank_shows_and_drops_what_it_hides`,
`only_a_layer_covering_the_pixel_whole_hides_what_is_behind_it`;
`tests/resolve_switch.rs::a_2d_scene_never_takes_the_sample_resolve` (no mesh).]

**Oracle reading: inputs to the recompute.**

- The recompute takes the **resolve class and unit membership the core recorded** as explicit
  inputs. Per-sample sequences do not determine them: the selection also reads the pixel's pair
  votes and strengths, and fragments absent from every sequence.
- **The selection is tested apart.** The selection check reads exactly these recorded quantities,
  per pixel:
  - the fragments: identity, producer and surface, mask or cell bytes, coverage byte, colour, and
    whether each is opaque and full;
  - the trace's pre-drop per-sample pair relations and strengths: for every ordered pair of
    fragments covering a sample, whether the first is in front of the second there, behind it, or
    neither, and whether depth decides there (a strong relation), recorded before any drop;
  - the pixel's pair votes and strengths (`majority@1`'s result);
  - the recorded class and units.

  It then checks, in order:
  1. every per-sample relation is one §7 admits for that pair at that sample: a strong relation
     only where the strong edge is admitted, a weak one only where the weak edge is. A relation
     outside the admitted set is `mismatch`;
  2. `majority@1` over the per-sample relations gives the recorded votes and strengths;
  3. the rule above, from those relations and votes, gives the sample ranks, the drops of
     `opaque_drop@1` and `unit_merge@1`, the units and the class, its "behind" read from the
     per-sample relations.

  A disagreement in step 2 or 3 is `report_inconsistent`, never a tie. The relations are recorded
  only under the trace's opt-in flag, as one additive key beside the pixel votes.

**Rejects:** a recorded class that the rule, applied to the recorded votes, does not give.

### The two resolves

**The coverage resolve.**

- Go back to front over the units in the pixel's order, starting from the background:
  `px = blend(colour, coverage, px)`.
- A unit of one fragment blends with that fragment's colour and coverage. For a path, that is the
  whole-pixel byte.
- A merged unit's colour is, per channel, `(Σ c_i n_i + N div 2) div N` over its members, where
  `n_i` is each member's mask bit count and `N` their sum. Its coverage is the mask coverage of
  their union.

**The sample resolve.**

- Each sample `k` composites, back to front from the background, the fragments its rank orders at
  `k`, each at its coverage there. Those ranks are the ones the drops above read where they ranked,
  and otherwise are ranked now over the kept fragments.
- The pixel is `(Σ_k v_k + 8) div 16` per channel.
- A fragment that the drop removed may still appear in a sample's sequence, because the rank was
  read before the drop. It always lies behind an opaque, full cover there, which blends over it
  exactly.

### Blend arithmetic

This is Cairo's ARGB32 image surface. Define

    mul(a, b)    = ((t >> 8) + t) >> 8,   t = a b + 0x80
    mul_7f(a, b) = (t + (t >> 8)) >> 8,   t = a b + 0x7f

- **An opaque source at coverage 255** replaces the pixel.
- **An opaque source at coverage `c`** gives `min(mul_7f(s, c) + mul_7f(d, 255 − c), 255)` per
  channel.
- **Any other source** is first scaled, `s' = mul(s, c)` per channel when `c < 255`. It is then
  laid over the pixel as `min(s' + mul(d, 255 − s'_alpha), 255)`.

The coverage resolve lands within 3 levels of the sample resolve for one partial mask. That is a
property, not a rule the recompute uses.

[Rests on: `composite::blend`, `mul`, `mul_7f`;
`tests/composite.rs::two_d_paint_blends_with_cairos_argb32_arithmetic`,
`composite.rs::the_byte_products_round_within_their_stated_errors`; `tests/coverage_bound.rs`.]

### Colour, from XF

These rules give each fragment's premultiplied bytes.

**`paint_rows@1`.** A record's paints are background stroke, fill and stroke. Each is no paint, one
straight RGBA colour, two colours, or a paint table's rows. Its widths are the stroke and
background-stroke widths.

- **Paint blends.** Under a paint blend toward `paint_b` at `mix`:
  - each width is `paint_b`'s where `mix = 1`, and otherwise `(1 − mix) a + mix z`;
  - each paint's rows are `paint_b`'s own where `mix = 1`, even when they are empty;
  - otherwise rows broadcast as numpy does: an empty side gives no rows, one row stretches to the
    other's count, and equal counts pair. Each row is `(1 − mix) p_k + mix q_k` per channel. Other
    counts are refused (§11).
- **A face** of a face group fills with row `i` of the face fill table, strokes with row `i` of the
  face stroke table (if it has one) at its own width (or the record's), and background-strokes with
  the record's rows.
- **An instances member `i`** fills with row `i` of the member fill table where the record's fill is
  one colour, and strokes likewise with the member stroke table. Its background stroke is the
  record's one colour. Otherwise it has no paint.
- **One row versus several.** A layer with one row is a solid. A layer with two or more rows is a
  gradient between the record's ends shown for display (§2).

[Rests on: `geometry::drawn_paint`, `mixed`, `rows`; `instance.rs` (members, faces).]

**`cairo_premultiply@1`.**

- The record's group opacity multiplies the alpha.
- Each channel is clamped to `[0, 1]`.
- Channel `k` is `trunc((c_k a) · 65535 + 0.5) >> 8`, and alpha is `trunc(a · 65535 + 0.5) >> 8`
  (`alpha_round@1`).

[Rests on: `composite::premultiply`.]

**`cairo_linear_gradient@1`.** The normative definition is Cairo 1.18's `cairo.LinearGradient`
filling or stroking the layer's outline on an ARGB32 image surface whose colours pixman 0.42
computes. The context matrix is the frame mapping (§2), which is diagonal. The chain, in Cairo's
and pixman's own steps:

1. **Stops.** Offsets are `i / (n − 1)`, the last exactly 1, as `np.linspace(0, 1, n)` gives them.
   Each is converted to 16.16 by `_cairo_fixed_16_16_from_double`: add `1.5 · 2^36` in binary64
   and keep the low 32 bits of the result's bit pattern as a signed integer, which rounds to the
   nearest 16.16 value with ties to even. Colours are clamped to `[0, 1]` and kept straight as the
   16-bit shorts `trunc(c · 65535 + 0.5)`.
2. **Solids.** A gradient every one of whose alpha shorts is at most `0xff` is clear and paints
   nothing. Ends closer than binary64's epsilon in both coordinates paint the average of the first
   and last stops. Stops whose shorts are all equal paint the first stop. Each solid is
   premultiplied by `cairo_premultiply@1`.
3. **The pattern matrix** is `cairo_matrix_invert` of the context matrix: for a diagonal matrix,
   negate the translation, then for each axis whose scale is not 1 take its reciprocal and multiply
   that axis's translation by it.
4. **To pixman.** `_cairo_matrix_to_pixman_matrix_offset` about the middle `(xc, yc)` of the
   composite extents, with no offset given: the integer-translation shortcut for a unit linear
   part; otherwise the translation split between the matrix and an integer pixel offset
   `(ix, iy)` (the candidate of least magnitude over `i, j ∈ {−1, 1}`, floored); the 16.16 entries
   by the step-1 conversion; then, unless the matrix has unity scale (Cairo's test at `1/256`), up
   to five fix iterations that move the translation until the fixed matrix agrees with the
   floating one at `(xc, yc)`.
5. **The ends** are the record's ends shown for display (§2), converted to 16.16 by step 1:
   `p1`, `p2`, with `d = p2 − p1` and `l = d_x² + d_y²`.
6. **The parameter.** Pixel `(x, y)` is the 16.16 point `v = pixman_transform_point_31_16_3d` of
   `((x + ix) · 2^16 + 2^15, (y + iy) · 2^16 + 2^15, 2^16)`. pixman's linear fetch evaluates
   `t = (d · v − d · p1) · 2^32 / (l · v_2)` in binary64 at a composite run's first pixel, its
   step `inc` likewise, and gives pixel `i` of the run the position `trunc(t) + trunc(inc · i)`.
7. **The colour** at a position is `pixman_gradient_walker_pixel_32` under pad extend, in single
   precision, as pixman 0.42 writes it.

The exact real parameter `τ(x, y) = 2^16 (d_x (v_0 − p1_x) + d_y (v_1 − p1_y)) / l` follows from
the integers of steps 4–6 alone; `v_2 = 2^16` for these matrices.

The core deviates from this chain in three ways:

1. **The ends are not fitted.** Cairo first applies `_cairo_gradient_pattern_fit_to_range` with a
   limit of 16383: where every end coordinate is at most 16383 in magnitude the fit is the
   identity, and beyond it Cairo rescales the ends and the matrix. The core never fits, and takes
   pixman's 64-bit products of the unfitted ends in 128 bits. Within the limit the two chains are
   one.
2. **Composite runs.** The core takes them from its own coverage spans in the row, not from Cairo's
   rasterizer spans. A run starts at the first covered pixel. A fully covered span at least 256
   pixels long is a run of its own. Otherwise spans are gathered, and a gap longer than one pixel
   cuts the gathered run once it is longer than 256 pixels. A one-pixel gap never cuts.
3. **The middle of the extents** comes from the outline's box in 24.8 fixed point, rounded out and
   cut to the frame, where Cairo uses its approximate path extents.

[Rests on: `gradient.rs` module doc, `source`, `to_pixman`, `Linear::parameter`, `Linear::at`;
`path::composite_runs`, `RUN_LENGTH`; `tests/gradient.rs`.]

**Oracle reading for gradients.** The oracle reproduces steps 1–5 and 7 exactly, at the middle of
its own outline's box (deviation 3's definition). It does not reproduce the runs. It computes
`τ` exactly from the integers, in rational arithmetic, and admits every integer position `p` with

    |p − τ| <= 2.5 + Δ_box,   Δ_box = 2^16 (|d_x| (|e_x| + 1) + |d_y| (|e_y| + 1)) / l

where `e_x` and `e_y` are the fixed matrix's linear entries minus `2^16` times the floating ones.
The admitted colours are the walker's at each admitted position.

- **Why the run need not be known.** For a diagonal matrix the rounding term of
  `pixman_transform_point_31_16_3d` does not depend on `x`, so `v` is exactly linear along a row
  and a run's start enters only through the two truncations, which move the position by less than
  2. Binary64 evaluation adds less than 1/2 while every end and every pixel's point is within the
  fit limit.
- **Why `Δ_box`.** The oracle's box and the core's agree except where an edge of the outline lies
  within reach of a pixel boundary, and a one-pixel move of the box moves the fixed translation by
  at most `|e| + 1` units per axis.
- **Why this holds for any stops.** The walker is evaluated at the positions themselves, so stop
  count and colour slope do not enter the bound. With 1,025 alternating black and white stops,
  run start and pixel centre give positions 16383 and 16384, colours about 4 and 0, and both are
  admitted (fixture `gradient_dense_stops`, §12).
- **Outside the promise.** An end, or a gradient pixel's point in pattern space, beyond 16383 in
  magnitude is `gradient_far_ends`: the oracle reports `truth_unable` for that layer's pixels.

The recompute carries each gradient fragment as the per-channel range of its admitted colours, by
interval arithmetic: every blend step is monotone in each operand.

**Rejects:** a gradient fragment's colour outside the walker's colours at every admitted position.
A shift of two levels in a two-stop gradient whose window spans less than one level is rejected
(witness W14, §12).

**`ce_shading@1`.**

- **Shaded paths.** A `shade_in_3d` path under a 3D camera that shades is lit per layer into two
  stops:
  - **Source colours.** A layer of one row lights that row twice. Otherwise it lights its **first
    two** rows, and the rest are dropped.
  - **Where.** At its world points `0` and `((n − 1) div 6) · 3`, as CE's `get_shaded_rgb` lights
    them. With `to_sun` the unit vector from the point toward the camera's light source,
    `δ = 0.5 (n · to_sun)³`, halved where negative, is added to each encoded channel, unclamped.
  - **Normals.** Each lit corner takes the normal its record carries (XF 1.2). A zero carried normal
    is lit by the face's Newell normal over its points, normalized, or by CE's corner normal where
    that is zero. A record with no carried normals uses CE's corner normal.
  - **CE's corner normal** at point `i` of a face of `n` points:
    - it is `(0, 1, 0)` where the face has at most two anchors, the anchor count being
      `2 · min(ceil(n / 4), (n − 4) div 4 + 1)` for `n >= 4`;
    - otherwise take `v1 = p[j] − p[i]` and `v2 = p[k] − p[i]`, where the conditions read the
      original index `i`: `j = i + 3` if `i < n − 3`, else `j = 3`; `k = i − 3` if `i > 2`, else
      `k = n − 4`. (For `n = 8`, the corner `i = 3` reads points 6 and 0.) Each is divided by its largest
      magnitude; where both are zero the normal is `(0, −1, 0)`, and where one is zero the other is
      used alone as `u`;
    - their cross product, where its length exceeds `1e-6`, divided by its length, is the normal;
    - otherwise `u` is the first of them, and the normal is `(0, −1, 0)` where `|u_x|` and `|u_y|`
      are both under `1e-6`, else `(−u_x u_z, −u_y u_z, u_x² + u_y²)` divided by its length;
    - a zero result is replaced by `(0, 1, 0)`.
  - **Drawing.** The two stops are drawn as a gradient between the recorded ends, or else between
    the corners.
- **Lit mesh triangles.**
  - **Normals.** A carried vertex normal `n` is transformed by the placement's linear part `L`
    through its cofactors, `n'_i = Σ_j cof(L)_{ij} n_j` with `cof(L)_{ij} = L_{i+1, j+1} L_{i+2, j+2}
    − L_{i+1, j+2} L_{i+2, j+1}` (indices mod 3), then divided by its length (left as is where the
    length is 0). A mesh with no normals uses CE's unit normal of `v1 = c1 − c0` and `v2 = c2 − c0`
    of the placed triangle (the cross-product rule above, from its second step), at all three
    corners.
  - **Lighting.** Each corner is lit at its placed position.
    - Under CE's model, the corner takes the shading above of the encoded colour, then is clamped to
      `[0, 1]`.
    - Under soft diffuse (light model 1), each encoded channel is first clamped to `[0, 1]`, then
      decoded by the sRGB transfer function: `c / 12.92` for `c <= 0.04045`, else
      `pow((c + 0.055) / 1.055, 2.4)`. The normal is turned to
      face the eye, `frame_center + d R_2`, where its dot product with the direction to the eye is
      negative. The linear colour is then scaled by
      `ambient + (1 − ambient) max((n · l + wrap) / (1 + wrap), 0)`, where `l` is the unit vector
      toward the light source.
  - **Interpolation and bytes.** The corner colours are interpolated as planes through the three
    corners on the pixel plane (the three-corner plane of §6), and read at the centroid of the
    samples the fragment covers, which is the pixel's centre for a full mask. The result is clamped
    to `[0, 1]`, encoded where linear (`12.92 v` for `v <= 0.0031308`, else
    `1.055 pow(v, 1 / 2.4) − 0.055`), then passed through `cairo_premultiply@1`. Corners of one
    colour give a solid.

[Rests on: `shade.rs` (`face_stops`, `Light::triangle`), `mesh::transform_normal`.]

**Point bytes.**

- Each channel and the alpha (times the group opacity) is the byte `trunc(255 c)`, as CE writes
  it.
- There is no block where the alpha byte is 0.
- Where the alpha byte is below 255, each channel becomes `(c · a + 127) div 255`.

[Rests on: `points.rs`.]

**Emission.** Nothing a frame of this engine draws emits light: emission records are refused
(§11).

**Oracle reading for colour.**

- **Inputs.** The recompute reads, per pixel:
  - the core's recorded class and units;
  - its per-sample sequences, after the drops;
  - each fragment's colour, whole-pixel coverage byte and cell bytes.

  It composites by this section's arithmetic, never through the core's resolve, and never from its
  resolved colours.
- **Colours from XF.** With the oracle's colours, it computes them by the rules above from XF.
- **Alpha.** Every Cairo-painted op's alpha goes through `alpha_round@1`.

## 10. Camera-facing labels

**Core rule (the plate model, `facing_plane@1`, §5).** A camera-facing label is a record with a
fixed-orientation centre in world space under a 3D camera, or a pre-projected record keyed at its
pivot. Every fragment of it lies on the camera-facing plane through its centre. Depth, ties and
policies order it like any other fragment, so something nearer can cut it, and a surface crossing
the label's depth splits it there.

### `label_placement@2`: the rule

*A camera-facing label goes under a filled surface nearer than its centre and stays over everything
else, lines and arrows included.* Lines and arrows never push a label back on their own, but a line
in front of a surface the label sits behind is in front of the label too.

**Terms.**

- **A label** is the family of camera-facing records drawn as one: the records sharing one
  `facing_group` (§10, the source section), in a stream read under XF 1.5's rules at the frame. A
  stream is read under them at a frame once a delta at or before that frame has carried the
  section; the oracle decides this in stream order, never from a later delta. Until then the
  stream is read as XF 1.4 reads it and its records carry no facing group: a label is a run of
  consecutive camera-facing records in the order whose carried centres are bit-identical, and two
  labels drawn next to each other with one centre are judged as one, a stated limit. Scene lowering
  writes the section in the first delta that sets a record, so a correct XF 1.5 stream is never
  read as XF 1.4 after its first record. In a stream read under XF 1.5's rules, a camera-facing
  record with no facing group belongs to no label the oracle can name: it is `label_ungrouped`
  (below).
- **Its outline** is its records' outlines on screen: each fill's outline and each stroke's outline
  (the stroke as §8 draws it).
- **Its depth** is its centre's key on its facing plane (§5), with that key's interval (§6). The
  label has one depth: no glyph is tested on its own.
- **The order points** are the 16 sub-sample points of each pixel (§1, *Sample positions*), the
  points at which the resolve orders fragments. In any one pixel column `x` they sit at
  `x + (2k + 1) / 32`, `k = 0 .. 15`, and likewise in each row.
- **A fill's alpha at a point** (an order point, or the label's projected centre) is that of the
  fill's colour at the pixel holding the point, as the frame paints it (§9): one colour per pixel,
  never the paint evaluated at the point itself. It is read as the frame stores it: premultiplied,
  8 bits, after pixman's rounding, with the record's group opacity applied, and it is **positive**
  at 1/255 or more. For a linear gradient the colour depends on the composite run the pixel is in
  as well as on the pixel (§9), and past a gradient's clear end pad extend gives exactly 0 whatever
  the run.
  - **The pixel box.** A fill's pixel box is, of its outline's extent in frame pixels, the columns
    `floor(x_min)` to `ceil(x_max)` and the rows `floor(y_min)` to `ceil(y_max) − 1`, within the
    frame. Where the point's pixel lies outside the box (a centre exactly on the fill's bottom edge,
    its largest `y`, at a whole pixel, or beyond the frame), the alpha is read at the box's nearest
    pixel.
  - **The oracle's reading.** For a gradient, the oracle reads the alpha at the pixel as the range
    of the alphas of the colours it admits there (§9, *Oracle reading for gradients*), which spans
    every run start the row allows. Where that range holds both 0 and a value of 1/255 or more, the
    pixel is oracle-unable: it is reported as `label_alpha_straddle` (§12), and never agrees. Past
    a clear end, only the last visible pixel before it can be such a pixel.
- **A sheet** is a connected part of one **surface object**, read from its fills on screen:
  - The surface object of a record is the outermost node of its first source path whose outcome
    under `label_classes@2`, read at that node, is a surface: a cube's faces belong to the cube and
    a sphere's to the sphere. A cylinder counts only where the cylinder rule makes it a surface;
    one left open (inside a group, say) does not, and its faces fall to the next node out with a
    surface outcome. Where the path has none, the object is the path's first node, the record's
    maker.
  - In a stream read as XF 1.4, where no source names an object, each record is its own object:
    its fills join only one another, and no sheet of it is ever disconnected.
  - Only the object's fills are read. Strokes never form or widen a sheet, and a record whose
    class (`label_classes@2`, below) is not a surface never forms one.
  - A fill's **winding** is the sign of the signed area of its flattened outline on screen (§4's
    flattening, in frame pixels): positive or negative, or neither where the area's magnitude is at
    most `1e-9` px².
  - Two fills of one object **join** where they have the same winding, and it is not neither, and
    share at least two distinct corners of their flattened outlines on screen, each coordinate `v`
    compared as `round(v · 4096)`. Joining is transitive: a sheet is a class of the union. A fill
    wound neither way is a sheet on its own.
  - A front-facing and a back-facing part of one solid are wound opposite ways, so they are two
    sheets.
- **An open sheet** is one no label goes behind, whatever its source:
  - a sheet drawn as a mesh, whose faces are not read;
  - in a frame that draws a facing group, a record kept facing the camera (it carries a
    fixed-orientation centre) whose facing group is none (`label_ungrouped`, below). Its label is
    unknown, never another kind, so it is an open sheet of its own. Its object is read as any
    record's is, above.
- **A sheet holds a point** where one of its fills whose alpha at the point is positive holds it by
  the nonzero rule. Qualification counts with this predicate, the centre test reads the same alpha,
  and insertion anchors with the same predicate.
- **The overlap** of a label and a sheet `S` is its set of **counted** order points. An order point
  is counted where it lies inside an outline of one of the label's records by the nonzero rule and
  `S` holds it. A sliver of either between order points holds none. A pair of the label and `S` is
  made only where at least one point is counted: a sheet that counts none does not overlap the
  label, even where it holds the label's centre.
- **A label drawn otherwise.** A label that has a record that is not a path placed at a fixed
  orientation (a pre-projected record, a mesh, points, or a record with no carried centre) is read
  over the whole frame: it meets `S` wherever a fragment of any of its records and `S` hold one
  order point.
- **Classes** come from `label_classes@2` below: a sheet qualifies only if its records' outcome is
  a surface. A line's or arrow's parts never qualify.

**Qualification: one decision per pair.** It is made once over the whole frame, before any tile,
and never retested at a pixel or a glyph. Each pair (a label and a sheet `S` with at least one
counted point) is decided by the first of these that applies:

1. **`label`, of the label.** The label is drawn otherwise (Terms, above). Every pair of that label
   is open.
2. **An open sheet.** `S` is a record kept facing the camera whose label is unknown: the pair is
   open with cause `label`, even where `S` is also drawn as a mesh, since `label` comes before
   `mesh`. Otherwise, where `S` is drawn as a mesh, the pair is open with cause `mesh`.
3. **`folded`.** Two fills of `S` hold one counted point. The pair is open.
4. **`disconnected`.** `S` shares its surface object and its winding with another sheet that also
   counts a point with the label. Every such sheet's pair is open, not only the second one's.
5. **By the centre.** Where `S` covers the label's centre: one of its fills whose alpha at the
   centre is positive holds the centre's screen point by the nonzero rule, a point on the fill's
   outline counting as held, on any of its segments, its horizontal ones included, whatever the
   fill's depth is keyed on. Compare that fill's own plane, read on the ray through the centre (its
   key there, §5, with its interval, §6), with the label's depth. Wholly nearer qualifies the whole
   pair, even where `S` crosses behind the label's depth elsewhere in the overlap. Wholly farther,
   or at the label's depth, does not qualify, and the label stays in front. The pair is open
   (`interval`) where the intervals meet, where the fill is on no one plane (its depth keyed on its
   curve, so it has no plane to read at the centre), or where two fills of `S` hold the centre and
   their comparisons disagree.
6. **By the points.** Otherwise `S` qualifies only where it is wholly nearer than the label's depth
   at every counted point, its depth there the plane of the fill holding the point, read at the
   point, with its interval (never an extended plane). Where it is wholly farther or at the label's
   depth at some counted point, it does not qualify: a **genuine straddle** the points can see, a
   decided outcome, and the label stays in front. Where no point is decided farther or equal but
   some point's interval meets the label's, the pair is open (`interval`).

The cost of reading at the points: a farther sliver of `S` that holds no order point drops out, so
`S` can qualify and hide the label at the points where it is nearer. The depth split that implies
lies below the sampling and cannot show. A straddle the points can see still lets the label show
through.

**Placement at a sample.** At each sample `k` the label covers:

- take the ordinary order (§7) of the fragments at `k` other than labels;
- among the qualifying sheets of the label that **cover** `k`, take the rearmost in that order, and
  insert the label immediately behind it. A sheet covers `k` exactly where it holds `k`'s order
  point (Terms, above), the predicate qualification counts with: a qualifying sheet that covers
  some of `k`'s cell but not its order point, or whose fill's alpha there is 0, does not anchor the
  label there;
- where no qualifying sheet covers `k`, the label is in front of every fragment there but the
  overlay.

So a qualifying sheet hides or dims the label only where it holds the order points (**partial
cover**), and anything in front of that sheet is in front of the label. A pixel where a label meets
a sheet it goes behind is resolved sample by sample, each fragment compositing its own coverage of
the sample's cell. Labels whose insertion is the same place keep CE's order, which is paint order.
Otherwise placement decides between labels, whatever their paint order. Every fragment composites
at its own alpha, once: a label behind a translucent sheet is seen through it, and a label between
two translucent sheets of one source (a translucent box's front and back) sits between them.

**The admissible orders** at `k` are the orders this insertion gives over every admissible ordinary
order of the other fragments (§7): an ordinary tie may change which qualifying sheet is rearmost.

**Open cases.** `label_rule_open` is counted by name, on two separate axes:

- **A pair left open,** by its pair cause from the qualification above: `label`, `mesh`,
  `folded`, `disconnected` or `interval`. These are the rule's open cases; `label_ungrouped`
  (below) is a reason under `label`, never a cause of its own.
- **A class left open** (`label_classes@2`, below), by its class cause: a record whose outcome is
  `label_rule_open` is never a surface, never forms a sheet and never qualifies, and the order at
  the samples it shares with a label is counted under its class cause: `unmapped` (a class the
  table does not place), `generic_only` (a source generic all the way up), `paths_disagree`,
  `no_source`, `cylinder_grouped` or `point_cloud`.

**`label_ungrouped`.** In a stream read under XF 1.5's rules at the frame (Terms, above), a
camera-facing record (one with a fixed-orientation centre) that has no facing group: set with no
row, set by a delta of an older minor, or given a facing group of 0 by a row. It is not thereby
outside every label: its label is unknown, never another kind. Its drawing is unaffected, and no
reader refuses it.

- In a frame that draws a facing group, it is an open sheet of its own (Terms, above): every pair a
  label makes with it is open with cause `label`, reported under `label`, `label_ungrouped` being
  only the reason.
- In a frame that draws no facing group, the frame is drawn as it would be without its source
  (*What XF states*, below), and the oracle judges it as it judges a frame read as XF 1.4 (Terms,
  above).

An open pair does not push the label back; the rendering it expects is the label on top of that
sheet, and other sheets still qualify and place it. The oracle blesses neither order for the
samples the open pair decides: they are reported under its cause, outside the promise, never
`mismatch` and never agreement.

**`label_kind_unknown`.** In a stream read as XF 1.4 at the frame (Terms, above), a face of a face
group and a mesh triangle may be a surface, a solid, or a line's or arrow's shaft or tip. The oracle reads each
both ways, as a qualifying surface sheet and as a line, and admits the orders of both readings
where they part. It is a class outside the promise (§12), not `label_rule_open`. It does not
arise in a stream read under XF 1.5's rules, where the table decides from each record's source.

**The interim trade.** In such a stream the oracle reads a path by its layers: its strokes and
background strokes are lines, and its fill (not a face of a face group) is a surface. A 2D arrow's
filled tip is such a fill, so it is read as a surface, a sheet that can put a label behind it,
there. A point block is no surface and never qualifies.

**Deviation (`label_plate_model`).** The core this contract describes orders a label by the plate
model: it draws a nearer line or arrow over the label, and lets a surface crossing the label's
depth split it. At a label sample where the reported order is outside the admissible orders, and is
the plate model's order, the result is a `mismatch` counted under the name `label_plate_model`,
beside `zero_alpha_fragment`, and never routed to `label_rule_open` (witnesses W18, W22, §12).

**Rejects:** a label drawn over an opaque filled polygon that qualifies by covering its centre
wholly nearer (W5); a label hidden by a nearer line it is not behind a surface for (W18); and each
planted failure of W19 to W26 and of the witnesses of the point reading (§12).

**Where the rule comes from.** Restated from the label design's addendum, SHA-256
`9529f44c870153f8c0902ea2418e7db907099dcdad5b5525e24fdbd6e245e3d6`, read with its correction 1,
SHA-256 `361d0280ca2a86ffc360ce5ff1e6f4138f7a56b137830801d76f7b7b16efc7ee`, which withdraws a
count of open cases: the rule states none.

**Change from `label_placement@1`.**

- The overlap is read at the order points: a point counts inside both a record of the label and a
  fill of the sheet whose alpha at the point is positive, by the nonzero rule, where @1 read every
  sample both footprints covered, antialias included. A sheet counting no point makes no pair, even
  where it holds the centre, and the centre is consulted only for a pair.
- A fill's alpha at a point is the painted pixel's, 8 bits and premultiplied; the centre test,
  qualification and insertion all read it, and a gradient pixel whose admitted alpha straddles 0
  and 1/255 is `label_alpha_straddle`.
- Insertion anchors a label only at the samples whose order point a qualifying sheet holds, where
  @1 anchored it wherever the sheet covered the sample.
- A sheet is joined from fills of one surface object by winding and two shared corners at a
  quantum of 1/4096 px, where @1 named a sheet by the way its faces face.
- The centre is held on any segment of a fill's outline, horizontal ones included, and a fill on
  no one plane leaves the pair open (`interval`).
- The open cases `label_unsettled` and `label_sheet_irregular` are replaced by the causes
  `interval`, `folded` and `disconnected`, and the causes `label` and `mesh` are new. `label`
  opens every pair of a label drawn otherwise, which is read over the whole frame, and every pair
  with an ungrouped record kept facing the camera, an open sheet. A pair is decided by `label`,
  then an open sheet (`label`, then `mesh`), `folded`, `disconnected`, then the centre or else the
  points.
- Every sheet sharing its object and winding with another sheet that meets the label is open, not
  only the second.
- `label_ungrouped` is no longer counted on its own: in a frame that draws a facing group its pairs
  are open with cause `label`, and a frame that draws none is judged as one read as XF 1.4.

**What XF states.** Quoted from `core/XF.md` at `30538b2e9c977092a0deb8480d4d791932a46e66`. The
source paragraph and the drawing bullet match that text exactly, and the fixture sentences match it
once whitespace is normalised. The rule above restates these for the oracle.

The source paragraph:

> XF 1.5 says which objects a record was drawn from, for readers that classify what a record is (a
> label, a line, a surface) by its objects rather than its geometry. A record's `source` is one or
> more paths of containment, each the record's owner first (the leaf, or the surface a face group
> draws), then the objects that contain it, outermost last; a child shared by two parents has a path
> through each. Each node is a token the writer keeps for that object across the stream, and the
> object's nearest library class, fully qualified (a scene's own subclass is named by the library
> class it derives from; no address and no name of the scene's own is written). A record's
> `facing_group` is a token its writer gives a camera-facing label: the records of one label share
> it, and two labels never do (0: none). The scene lowering writes both. A drawer may read them to
> place camera-facing labels (this core does: `label_plan.rs`); a frame with no facing group is
> the same bytes with them or without. A record set again replaces both, and a record
> set without them has neither: they never outlive their record. When a record's objects move (it is
> reparented) and its drawing does not, its writer does not set it again: a row for a slot the delta
> neither sets nor removes gives the held record its new source and facing group, so the bytes an
> older minor carries do not move. They are carried in a section after the delta's order, keyed by
> slot, its class names stated once per delta, so an XF 1.4 reader, which stops after the order,
> reads the delta as it was. A reader of XF 1.4 that writes the delta again writes it without them.

The drawing bullet:

> - **Camera-facing labels (XF 1.5).** The records sharing a `facing_group` are one label, placed
>   whole and never split by depth; every other record is read from its `source` as a line, a
>   surface or `label_rule_open` (`label_class.rs`), and only a surface can go in front of a label. A
>   label's depth is its centre's: the camera-facing plane through it, read at the centre's
>   projection, with its error bound. A sheet is a connected part of one surface object (the
>   outermost surface or cylinder on a face's first path) whose fills are wound one way on screen
>   (by the sign of their flattened outline's area), two fills sharing two corners of their flattened
>   outlines on screen, each rounded to 1/4096 pixel, joined; a fill wound neither way is a sheet
>   alone, and strokes are never part of a sheet. A sheet goes
>   in front of a label when it is shown nearer: where one of its fills holds the label's projected
>   centre (a point on an edge held), on the ray through it, by that fill's plane read there; else at
>   every order point of their overlap, by the plane of the fill holding the point read there; each
>   depth with its error bound. The order points are the 16 sub-sample points of each pixel, where
>   the resolve orders fragments: pixel `(x, y)`'s point `s` is
>   `(x + 1/2 + (dx + 1/2)/16, y + 1/2 + (dy + 1/2)/16)` for `[dx, dy]` the `s`-th of
>   `SAMPLE_OFFSETS`: `[-8, -7]`, `[-1, -8]`, `[3, -6]`, `[5, -5]`, `[-6, -1]`, `[-4, -4]`, `[0, -3]`,
>   `[6, -2]`, `[-7, 2]`, `[-2, 0]`, `[2, 3]`, `[4, 1]`, `[-5, 5]`, `[-3, 4]`, `[1, 7]`, `[7, 6]`. The
>   overlap is the order points inside both a record of the label (a fill or a stroke's outline) and
>   a fill of the sheet in a colour not wholly transparent, each by the nonzero rule: a sliver of
>   either between points holds none, and a sheet holding no point of the label's does not overlap
>   it. A sheet shown farther or at the label's depth there leaves the label in front; one the bounds
>   cannot place, one two of whose fills hold one point of the overlap (folded), one sharing its
>   object and its winding with another sheet that meets the label, one drawn as a mesh, and every
>   sheet against a label not drawn wholly as engine-placed paths is open, and leaves the label in
>   front too. At each sub-sample the label is inserted immediately behind the rearmost sheet it goes
>   behind that covers the sub-sample, in the order everything else takes there, or, with none, in
>   front of everything but the overlay; labels at one place keep draw order. A pixel where a label
>   meets a sheet it goes behind takes the sample resolve. A frame with no facing group draws as it
>   would without its `source`.

The fixture sentences of the binary form:

> XF streams are generated by code, never committed, except the SOURCE fixtures under
> `tests/fixtures/xf15/` that readers outside the core read (their STREAMS.md); the core's test
> regenerates them and checks them byte for byte. A reader may refuse the whole file where this
> text refuses a delta; it must never apply a delta this text refuses.

### Source classes (`label_classes@2`)

**Status.** The table below is an inventory written independently of the core's own mapping, from
the library's source at commit `0f8def6e`: every class there that derives from
`arachne.mobject.mobject.Mobject`, each named by its defining module and qualified name. It has not
yet been checked against the classes real scenes use. Every open outcome is `label_rule_open` with
a cause, and fails closed: the oracle blesses no order for the pairs it decides.

**Label first.** A record with a facing group (not 0) is a label, and is read by the label rule
above before any class is read. Everything below classifies the other records, the sources a sheet
may come from.

Read from the record's source, which the wire carries from XF 1.5 (below). A record's source lists,
for each of its paths, the containment chain from the object that made the record (its **maker**)
outward through its **containers**, each node's nearest library class (a user's subclass maps to
the library class it derives from). A class is named in the source, and in the table below, by the
string `<defining module>.<qualified name>`: the module that defines the class, then its
`__qualname__`, for example `arachne.mobject.three_d.three_dimensions.Surface`. A node is **generic** when its class is one of these six:
`arachne.mobject.mobject.Mobject`, `arachne.mobject.mobject.Group`,
`arachne.mobject.types.vectorized_mobject.VMobject`,
`arachne.mobject.types.vectorized_mobject.VGroup`,
`arachne.mobject.types.vectorized_mobject.VDict` or
`arachne.mobject.three_d.three_dimensions.ThreeDVMobject`, a user's subclass of one of them
included. The list is closed: no other class is generic, whatever it contains. A node is
**surface-like** when the
table makes its class a surface or gives it the cylinder rule. Each path gives an outcome, by the
first of these that applies:

1. **Line ancestry.** Where any node of the path, the maker or any container, is a line class, the
   outcome is line. Nodes of other classes between them do not stop it: a cone, a cylinder or a
   filled tip inside an arrow is a line, and so is a rectangle behind a label inside a labelled
   line.
2. **Generic nodes read through.** Where the maker is generic, the path is read from its nearest
   container that is not generic, as if that container had made the record. Where every node is
   generic, the outcome is `label_rule_open` with cause `generic_only`.
3. **Parts follow a surface-like container.** Where the node read is not a cylinder (rule 4
   decides a cylinder itself), walk up from it through its containers to the nearest surface-like
   one. The walk passes over generic nodes and over nodes whose class is open (`unmapped` or
   `point_cloud`), and stops at the first cylinder it meets, never passing over one. Where it finds
   one, the outcome is that container's own: a surface for a surface class, and for a cylinder what
   rule 4 gives it where it stands. So a cap of a cylinder is a surface or open with its cylinder,
   even where a surface holds the cylinder, and a part under an open node inside a surface is a
   surface.
4. **The cylinder rule.** A node of class `arachne.mobject.three_d.three_dimensions.Cylinder` with
   no container is a surface: a solid. Inside any container (one that is not a line, which rule 1
   has taken), a surface-like one included, it is `label_rule_open` with cause `cylinder_grouped`:
   a cylinder is never read through its container by rule 3. Nothing reads its shape: a shaft
   added to the scene ungrouped is read as a solid and can hide a label.
5. **The node's own class,** from the table: a line, a surface, or open with the cause the table
   gives. A class the table does not list is not generic by default, a container class included:
   its outcome is decided when the class is added to the table, and until then it falls closed to
   `label_rule_open` with cause `unmapped`.

The record's outcome is its paths' outcome where they all agree. Where they differ, it is
`label_rule_open` with cause `paths_disagree`; a record with no source is `label_rule_open` with
cause `no_source`.

**What a class decides.** Only a surface changes where a label goes: a sheet qualifies only if its
record's outcome is a surface, and a sheet is read from the record's fills of alpha above 0 (§10,
Terms). Its strokes never form or widen a sheet, so a surface record drawn by strokes alone never
qualifies, a decided outcome that changes no class. A line never qualifies. An open record never
forms a sheet; the label is expected on top of it, and the order at the samples it shares with a
label is counted `label_rule_open` with its cause.

| Class | Outcome | Decided by |
|---|---|---|
| `arachne.mobject.three_d.three_dimensions.Arrow3D`, `arachne.mobject.three_d.three_dimensions.Line3D` | line | rule 5: 3D lines, shafts, caps and tips included |
| `arachne.mobject.geometry.line.Angle`, `arachne.mobject.geometry.line.Elbow`, `arachne.mobject.geometry.line.RightAngle` | line | rule 5: angle marks |
| `arachne.mobject.geometry.tips.ArrowCircleFilledTip`, `arachne.mobject.geometry.tips.ArrowCircleTip`, `arachne.mobject.geometry.tips.ArrowSquareFilledTip`, `arachne.mobject.geometry.tips.ArrowSquareTip`, `arachne.mobject.geometry.tips.ArrowTip`, `arachne.mobject.geometry.tips.ArrowTriangleFilledTip`, `arachne.mobject.geometry.tips.ArrowTriangleTip`, `arachne.mobject.geometry.tips.StealthTip` | line | rule 5: arrow tips, filled or not |
| `arachne.animation.changing.AnimatedBoundary`, `arachne.animation.changing.TracedPath`, `arachne.mobject.geometry.arc.CubicBezier`, `arachne.mobject.graphing.functions.FunctionGraph`, `arachne.mobject.graphing.functions.ImplicitFunction`, `arachne.mobject.graphing.functions.ParametricFunction`, `arachne.mobject.types.vectorized_mobject.CurvesAsSubmobjects`, `arachne.mobject.types.vectorized_mobject.DashedVMobject`, `arachne.mobject.vector_field.StreamLine` | line | rule 5: curves |
| `arachne.mobject.vector_field.ArrowVectorField`, `arachne.mobject.vector_field.StreamLines`, `arachne.mobject.vector_field.VectorField` | line | rule 5: fields, drawn as curves and arrows |
| `arachne.mobject.geometry.labeled.LabeledArrow`, `arachne.mobject.geometry.labeled.LabeledLine`, `arachne.mobject.geometry.line.Arrow`, `arachne.mobject.geometry.line.DashedLine`, `arachne.mobject.geometry.line.DoubleArrow`, `arachne.mobject.geometry.line.Line`, `arachne.mobject.geometry.line.TangentLine`, `arachne.mobject.geometry.line.Vector`, `arachne.mobject.geometry.shape_matchers.Cross`, `arachne.mobject.geometry.shape_matchers.Underline` | line | rule 5: lines and arrows |
| `arachne.mobject.geometry.arc.Arc`, `arachne.mobject.geometry.arc.ArcBetweenPoints`, `arachne.mobject.geometry.arc.CurvedArrow`, `arachne.mobject.geometry.arc.CurvedDoubleArrow`, `arachne.mobject.geometry.arc.TangentialArc` | line | rule 5: the arc family |
| `arachne.mobject.geometry.arc.TipableVMobject` | line | rule 5: the base of curves that take tips |
| `arachne.mobject.graphing.coordinate_systems.Axes`, `arachne.mobject.graphing.coordinate_systems.ComplexPlane`, `arachne.mobject.graphing.coordinate_systems.NumberPlane`, `arachne.mobject.graphing.coordinate_systems.PolarPlane`, `arachne.mobject.graphing.coordinate_systems.ThreeDAxes`, `arachne.mobject.graphing.number_line.NumberLine`, `arachne.mobject.graphing.number_line.UnitInterval` | line | rule 5: the plane and axes family |
| `arachne.mobject.graphing.probability.BarChart` | line | rule 5: the plane and axes family; its bars are lines by rule 1 |
| `arachne.mobject.geometry.arc.ArcPolygon`, `arachne.mobject.geometry.arc.ArcPolygonFromArcs` | surface | rule 5: arc polygons |
| `arachne.mobject.geometry.shape_matchers.BackgroundRectangle`, `arachne.mobject.geometry.shape_matchers.SurroundingRectangle` | surface | rule 5: background and surrounding rectangles |
| `arachne.mobject.graphing.probability.SampleSpace` | surface | rule 5: filled planar shapes |
| `arachne.mobject.frame.FullScreenRectangle`, `arachne.mobject.frame.ScreenRectangle` | surface | rule 5: screen rectangles |
| `arachne.mobject.three_d.polyhedra.ConvexHull3D`, `arachne.mobject.three_d.polyhedra.Dodecahedron`, `arachne.mobject.three_d.polyhedra.Icosahedron`, `arachne.mobject.three_d.polyhedra.Octahedron`, `arachne.mobject.three_d.polyhedra.Polyhedron`, `arachne.mobject.three_d.polyhedra.Tetrahedron`, `arachne.mobject.three_d.three_dimensions.Cone`, `arachne.mobject.three_d.three_dimensions.Cube`, `arachne.mobject.three_d.three_dimensions.Dot3D`, `arachne.mobject.three_d.three_dimensions.Prism`, `arachne.mobject.three_d.three_dimensions.Sphere`, `arachne.mobject.three_d.three_dimensions.Surface`, `arachne.mobject.three_d.three_dimensions.Torus` | surface | rule 5: surfaces and solids |
| `arachne.mobject.geometry.boolean_ops.Difference`, `arachne.mobject.geometry.boolean_ops.Exclusion`, `arachne.mobject.geometry.boolean_ops.Intersection`, `arachne.mobject.geometry.boolean_ops.Union`, `arachne.mobject.geometry.boolean_ops._BooleanOps` | surface | rule 5: the boolean operations |
| `arachne.mobject.geometry.arc.AnnotationDot`, `arachne.mobject.geometry.arc.Annulus`, `arachne.mobject.geometry.arc.Circle`, `arachne.mobject.geometry.arc.Dot`, `arachne.mobject.geometry.arc.Ellipse`, `arachne.mobject.geometry.arc.LabeledDot` | surface | rule 5: the circle and dot family |
| `arachne.mobject.geometry.labeled.LabeledPolygram`, `arachne.mobject.geometry.polygram.ConvexHull`, `arachne.mobject.geometry.polygram.Cutout`, `arachne.mobject.geometry.polygram.Polygon`, `arachne.mobject.geometry.polygram.Polygram`, `arachne.mobject.geometry.polygram.Rectangle`, `arachne.mobject.geometry.polygram.RegularPolygon`, `arachne.mobject.geometry.polygram.RegularPolygram`, `arachne.mobject.geometry.polygram.RoundedRectangle`, `arachne.mobject.geometry.polygram.Square`, `arachne.mobject.geometry.polygram.Star`, `arachne.mobject.geometry.polygram.Triangle` | surface | rule 5: the polygram family |
| `arachne.mobject.geometry.arc.AnnularSector`, `arachne.mobject.geometry.arc.Sector` | surface | rule 5: the sectors |
| `arachne.mobject.three_d.three_dimensions.Cylinder` | the cylinder rule | rule 4 |
| `arachne.mobject.mobject.Group`, `arachne.mobject.mobject.Mobject`, `arachne.mobject.three_d.three_dimensions.ThreeDVMobject`, `arachne.mobject.types.vectorized_mobject.VDict`, `arachne.mobject.types.vectorized_mobject.VGroup`, `arachne.mobject.types.vectorized_mobject.VMobject` | generic, read through | rule 2 |
| `arachne.frame_ir.cairo_painter._PointsProxy`, `arachne.mobject.types.point_cloud_mobject.Mobject1D`, `arachne.mobject.types.point_cloud_mobject.Mobject2D`, `arachne.mobject.types.point_cloud_mobject.PGroup`, `arachne.mobject.types.point_cloud_mobject.PMobject`, `arachne.mobject.types.point_cloud_mobject.Point`, `arachne.mobject.types.point_cloud_mobject.PointCloudDot` | open, `point_cloud` | rule 5: a cloud of points is not a sheet |
| `arachne.frame_ir.cairo_painter._ImageProxy`, `arachne.frame_ir.cairo_painter._PathProxy`, `arachne.frame_ir.geometry._Path`, `arachne.mobject.geometry.labeled.Label`, `arachne.mobject.graph.DiGraph`, `arachne.mobject.graph.GenericGraph`, `arachne.mobject.graph.Graph`, `arachne.mobject.logo.ManimBanner`, `arachne.mobject.matrix.DecimalMatrix`, `arachne.mobject.matrix.IntegerMatrix`, `arachne.mobject.matrix.Matrix`, `arachne.mobject.matrix.MobjectMatrix`, `arachne.mobject.svg.brace.ArcBrace`, `arachne.mobject.svg.brace.Brace`, `arachne.mobject.svg.brace.BraceBetweenPoints`, `arachne.mobject.svg.brace.BraceLabel`, `arachne.mobject.svg.brace.BraceText`, `arachne.mobject.svg.svg_mobject.SVGMobject`, `arachne.mobject.svg.svg_mobject.VMobjectFromSVGPath`, `arachne.mobject.table.DecimalTable`, `arachne.mobject.table.IntegerTable`, `arachne.mobject.table.MathTable`, `arachne.mobject.table.MobjectTable`, `arachne.mobject.table.Table`, `arachne.mobject.text.code_mobject.Code`, `arachne.mobject.text.numbers.DecimalNumber`, `arachne.mobject.text.numbers.Integer`, `arachne.mobject.text.numbers.Variable`, `arachne.mobject.text.tex_mobject.BulletedList`, `arachne.mobject.text.tex_mobject.MathTex`, `arachne.mobject.text.tex_mobject.MathTexPart`, `arachne.mobject.text.tex_mobject.SingleStringMathTex`, `arachne.mobject.text.tex_mobject.Tex`, `arachne.mobject.text.tex_mobject.Title`, `arachne.mobject.text.text_mobject.MarkupText`, `arachne.mobject.text.text_mobject.Paragraph`, `arachne.mobject.text.text_mobject.Text`, `arachne.mobject.text.typst_mobject.MathTypst`, `arachne.mobject.text.typst_mobject.Typst`, `arachne.mobject.types.image_mobject.AbstractImageMobject`, `arachne.mobject.types.image_mobject.ImageMobject`, `arachne.mobject.types.image_mobject.ImageMobjectFromCamera`, `arachne.mobject.types.vectorized_mobject.VectorizedPoint`, `arachne.mobject.value_tracker.ComplexValueTracker`, `arachne.mobject.value_tracker.ValueTracker` | open, `unmapped` | rule 5: neither a line nor a surface |

- The key is the nearest library class itself, never membership of a base class:
  `arachne.mobject.geometry.arc.Circle` derives from `arachne.mobject.geometry.arc.Arc` and is a
  surface; `arachne.mobject.geometry.arc.Dot` is read as itself.
- The table is the oracle's own and is written from this text alone.

**Change from `label_classes@1`.**

- A facing group makes a record a label before any class is read.
- Line ancestry is read per path, and paths that disagree are open (`paths_disagree`), where @1
  made a record a line when any node of any path was one.
- A generic maker reads through to its nearest non-generic container, where @1 found no class for
  it.
- A record under a surface-like container takes that container's outcome.
- A cylinder with no container is a surface, and one in a container that is not a line is open
  (`cylinder_grouped`), where @1 left every bare cylinder unmapped.
- A surface's strokes no longer make it a line: its class comes from its source alone, and its
  sheet is its filled footprint.
- Point clouds are open (`point_cloud`), where @1 said they never qualify.
- `label_source_unmapped` is gone: every open outcome is `label_rule_open` with a cause.
- Classes are named in full, and the table lists every mobject class of the library.

**How the wire carries a source (XF 1.5).** A delta may end, after its order, with a source
section: its class names, then its rows. Each row is `u32 slot`, `u32 facing_group`, a path count,
then the paths, each a list of nodes `(object token, class index)` into the section's class names.
A row carries both fields, the record's source and its facing group.

- **Keyed by slot.** A row is keyed by its slot alone. The order of the rows decides only which
  code a refused delta reports (§11); the state a valid delta leaves does not depend on it.
- **A set's row.** A row whose slot the delta sets is that record's source and facing group; of a
  slot set more than once in the delta, its last record's. A record the delta sets with no row has
  neither source nor facing group.
- **A held row.** Any other row is a held row. It replaces both fields of the record the stream
  holds in that slot, and a field the row does not carry becomes none: a facing group of 0 makes
  the held record's facing group none, and no paths make its source none. It leaves the record's
  drawing as it was: its geometry, paint, placement and place in the order.
- **Removed.** Removing a slot removes the record and both its fields.
- **Version.** The section is present, and the delta is XF 1.5, when it holds at least one row of
  either kind. A delta with none keeps its XF 1.4 bytes, and a 1.4 reader stops before the
  section.
- **What a writer sends.** A held slot whose source or facing group changes while its drawing does
  not travels as a held row: a reparent, a change of facing group alone, or both. A changed drawing
  travels as a set, carrying both fields in its row. A writer with nothing to carry for a slot
  writes no row for it.
- **Facing groups.** A writer of XF 1.5 gives every record it sets with a fixed-orientation centre
  a facing group, the records of one camera-facing label sharing it, and gives a held one a row
  whenever its group changes; scene lowering sets no fixed-orientation record outside such a label,
  and writes the section in the first delta that sets a record. These rules hold at a frame once a
  delta at or before it has carried the section (the label term, above). In a stream read under
  them, a fixed-orientation record that has no facing group is not thereby outside every label: it
  is `label_ungrouped` (above), its drawing unaffected, and no reader refuses it.

**The source a decision reads.** A label decision for a frame reads each record's source and facing
group as the stream holds them at that frame: the slot's latest set or held row, in delta order. A
row carries the same source content a set does, and the class rule above reads that content, never
how it arrived, so `label_classes@2` stands as it is: after a held row, the source read is the
row's (witness W27, §12).

## 11. What the oracle refuses

**Core rule.** The core refuses a frame whole, drawing nothing, when an entry of the order it draws
(overlay entries excluded, §1) holds any of these:

- **Unknown or reserved input:**
  - a feature bit outside bits 0 and 1;
  - an unknown record kind;
  - a reserved kind (volume 64, field 65, emission 66);
  - an `image` record.
- **Unsupported geometry:**
  - a clip;
  - a blend, reveal or paint blend on any kind but a path, or on a pre-projected path;
  - a paint blend whose row counts numpy does not broadcast;
  - a mesh (or mesh instance) with a stroke, a gradient fill or a dash;
  - a path blend of shapes of different lengths;
  - a face group whose points do not divide into its faces.
- **Unsupported paint:**
  - a shaded path under a light model other than CE's;
  - a gradient paint without ends on an unshaded path, or on an instances record;
  - a dash whose period is not finite and above 0, whose duty is not in `(0, 1]`, or whose phase is
    not finite.
- **Too large:**
  - an instances record of more than `2^32` parts;
  - a points record of more than `2^32` block writes;
  - a pre-projected points record that places other than one point per colour.

It also refuses a view whose light model is unknown. A record outside the order is neither drawn nor
refused.

[Rests on: `stream::refusal`, `stream::field_refusal`, Python `render.unsupported`;
`tests/mesh.rs::a_mesh_carrying_what_this_build_does_not_draw_is_refused_whole`,
`tests/path.rs::what_paths_and_points_do_not_draw_yet_is_refused_whole`.]

**Oracle reading.** The oracle answers `truth_unable` for the whole frame, never drawing it partly,
in these cases (overlay entries excluded):

- the frame's order holds a reserved kind (volume, field, emission), or a kind or feature bit it
  does not know;
- the view names a light model it does not know (it knows CE's, code 0, and soft diffuse, code 1);
- the core refuses the frame (there is nothing to judge);
- the stream is a delta of a newer XF minor version than 5. It reads deltas of XF major 1 up to
  minor 5 (minor 5's source section, §10), and cannot know the meaning of a newer minor's skipped
  fields;
- the view's background is an image smaller than the frame in either dimension (§1).

Each refusal names the slot, the object, and the kind or bit.

**Invalid XF is not a frame.** A delta is validated whole before any of it applies, in the two
places the XF specification's *Where each check is made* lists, in its order:

- **Decode**, by the XF reader both sides share, in the order of the delta's bytes: among its
  checks, bytes that end inside a field, a code the build does not know in the view or a record,
  a shape uploaded under a key other than its content's, an order entry's layer code the build
  does not know, and the section's codes. The reader checks nothing against what a stream holds,
  and, but for the section's codes, no part of a delta against another.
- **Apply**, against the stream and the delta as a whole: among its checks, a record or the view
  (a background image, say) naming a shape or paint table the stream would not hold after the
  delta, a removal or release of what is not there, an order drawing a slot not set, two display
  records drawn in the same layer of one view's order with the same capture rank, and last
  `slot_not_held`.

A delta that fails is refused atomically: nothing of it applies, and the stream stands as
before it. The oracle never judges a frame built from part of one, and reports the delta's refusal
by its reason, `invalid_delta`. The source section's seven checks (below) are XF's too, but a delta
whose one refusal is one of them is the stream's refusal, not `invalid_delta`.

**A source section refused (XF 1.5).** The source section has seven refusal codes:

1. `class_named_twice`: the section's class names hold one name twice.
2. Then, reading the rows in the order written, at the first row that offends, the first of these
   that applies to it:
   - `slot_named_twice`: a slot an earlier row named;
   - `slot_removed`: a slot the delta removes, whether or not it also sets it;
   - `row_empty`: a row of either kind with a facing group of 0 and no paths;
   - `path_empty`: a path of no objects;
   - `class_out_of_range`: a class index at or past the section's names.
3. `slot_not_held`, at apply: a held row whose slot the stream does not hold before the delta.

**One refusal per delta, the first of all XF's checks.** A delta has exactly one refusal, the first
of all XF's checks in this order, each check made where *Where each check is made* places it:

1. decode's checks, in the order of the delta's bytes: a refusal of anything written before the
   section is the delta's ordinary refusal, not one of the seven codes, and the section's own codes
   come in the order above (1, then 2);
2. then apply's other checks;
3. then `slot_not_held`, which is the delta's refusal only when nothing else in it is refused.

So three precedences follow. A decode refusal of bytes before the section, such as an order
entry's unknown layer code, is the delta's ordinary refusal even where the section would also be
refused. A section code wins over every ordinary refusal apply makes: a record naming a shape the
delta neither uploads nor finds held is refused by apply, never by the reader, so a delta that
carries both is refused by the section's code. An ordinary refusal apply makes, such as an order
drawing a slot not set, wins over `slot_not_held`. The order of the rows decides only
which code is reported. A refused delta is not applied. This order is the wire's rule, and the
oracle states it whatever the core does.

The oracle maps every ordinary refusal to `invalid_delta`, and each of the seven codes to
`StreamRefused { at_seq, cause }`, where `at_seq` is the refused delta's sequence number and
`cause` is the code. The code is diagnostic, since the stream is `truth_unable` whichever code
applies, but the oracle and the core must report the same one.

The two refusals are exclusive: a frame is under one of them at most.

- **`invalid_delta`:** a delta whose first refusal, in this order, is an ordinary one, in a stream
  not yet refused. Its own frame is `truth_unable` under `invalid_delta`, and the frames after it
  are judged on the stream as it stood before it.
- **`StreamRefused`:** a delta whose first refusal, in this order, is one of the seven. The refused
  delta's frame and every later frame of the stream are `truth_unable` under this one reason,
  whichever code applies, and none carries a verdict. A later delta never clears it, and a refused
  stream does not resume. A later delta that would be an invalid delta adds nothing and changes
  nothing: its frame stays under `StreamRefused`, with the refused delta's `at_seq` and `cause`. A
  writer that sends any of the seven has lost track of which slots it holds and what each carries,
  so no state after it is one the oracle can trust. It is never a verdict on the core.

## 12. Cases the contract must reject

### Fixtures

Each case below is a **witness**: a concrete scene, a fixture under
`core/oracle/tests/fixtures/<name>/`, and the verdict the oracle must give. A fixture directory
holds:

- `scene.sha256`, the SHA-256 of the XF stream bytes the fixture's test code builds for the scene
  the witness gives. The scene is built in code and never committed as a binary file; the hash
  pins the bytes, so a change to the builder or to the wire shows as a changed hash;
- for a mutation, `defect.json`: the one recorded quantity the defect changes (which quantity, at
  which pixel and samples, and its injected value), applied to the conforming core's report of
  that scene. Every other recorded quantity is that report's own, so the fixture differs from the
  conforming case by the defect alone;
- `expected.json`: the verdict, the pixels and samples it falls on, and the label or count name
  it carries.

A mutation's verdict is `mismatch` unless its row names another, with its reason. A soundness
fixture's verdict is agreement: the conforming report passes. Each witness's trace below gives
every pair's `E` at the samples it names, so that no admitted route (a policy, a may-share pair, a
label case, a cycle) can absorb the defect.

### The views

Every scene uses one of two views, an input grid of `2^-20`, no placement, a black background, and
opaque colours unless a witness says otherwise. A line from `p` to `q` is one cubic with controls
`p`, `(2p + q) / 3`, `(p + 2q) / 3`, `q`; every coordinate below is a
value given exactly, and control points are written in binary64 as the nearest values to
the thirds. A polygon's edges are such lines, its subpath closed.

- **View F** (flat camera). 16 × 16 px, frame 16 × 16 units, mapping centre `(0, 0)`: `xx = 1`,
  `yy = −1`, `x0 = y0 = 8`. Pixel `(i, j)` holds the points `x ∈ [i − 8, i − 7]`,
  `y ∈ [7 − j, 8 − j]`. Every key is `−∞`.
- **View C** (3D camera). 64 × 64 px, frame 16 × 16 units centred at the origin, `phi = 0`,
  `theta = −90°`, `gamma = 0` (the rotation is the identity), focal distance 20, zoom 1: `xx = 4`,
  `yy = −4`, `x0 = y0 = 32`. A point `(x, y, z)` has factor `f = 20 / (20 − z)` and key `−f`, and
  lands at `(32 + 4 f x, 32 − 4 f y)`. At `z = 0`, pixel `(i, j)` holds
  `x ∈ [(i − 32) / 4, (i − 31) / 4]`, `y ∈ [(31 − j) / 4, (32 − j) / 4]`.

Every curve below is kept clear of `curve_recut`: each cubic's `L` lies at least 0.06 px from a step
of its count, and no joint is near a flip, against `6ε̂` under `10^-5` px.

### Mutations

**W1, `exact_tie_2d`.** Set: `ExactTie` (§7). View F. Square `A` = `[−4, 2]²`, fill red, then
square `B` = `[−2, 4]²`, fill blue. Pixel `(8, 8)` (points `[0, 1] × [−1, 0]`) is inside both.
Trace: both keys `−∞`, `E = {weak}`, `B` in front. Defect: at its 16 samples, `A` in front.
Escapes: two fragments, so no cycle; no label; `ExactTie` admits no strong edge.

**W2, `exact_tie_axis_fills`.** Set: `ExactTie` (§7). View C. Squares `A` = `[−4, 2]²` and
`B` = `[−2, 4]²` at `z = 0`, each coded on the axis plane `z`, fills red then blue. Pixel
`(32, 32)` (points `[0, 0.25] × [−0.25, 0]`) is inside both outlines. Trace: both read the axis
plane at the sample in every scenario, so `E = {weak}`, `B` in front. Defect: at its 16 samples,
`A` in front. Escapes: as W1.

**W3, `thin_stroke_present`.** Set: presence band `μ` (§4). View F. An open line from
`(−6, −0.5)` to `(6, −0.5)`, stroke black, width `0.25`, no fill. In pixel `(8, 8)` it covers the
cell rows `y ∈ [−0.75, −0.5]` and `[−0.5, −0.25]` by `0.5` each, far above `1/510 + μ`. Trace:
`Yes` at the 8 samples of those rows. Defect: the fragment absent there. Escapes: `Either` needs a
cell within `μ` of `1/510`.

**W4, `mesh_face_reversed`.** Set: intervals `S` and `R` (§6). View C. A mesh triangle `T` with
corners `(−6, −6, 0)`, `(6, −6, 0)`, `(0, 6, 0)`, red, then a polygon `Q` on the plane `z = x / 2`
with corners `(−6, −6, −3)`, `(6, −6, 3)`, `(6, 6, 3)`, `(−6, 6, −3)`, fill blue. Pixel
`(36, 32)`: `T` keys `−1`; `Q` lies at `x ≈ 1.094`, `z ≈ 0.547`, keys about `−1.028`. Trace: no
shared plane (the union is at least 1 unit wide; no third record; `T` is flat, so no axis rule);
no `near_parallel@1` (dihedral `atan(1/2) ≈ 26.6°` against `θ_A + θ_B` under `10^-5`); no
`lines@1` (no strokes); the keys part by `0.028` against bands under `10^-5`. `E = {strong Q}`.
Defect: at its 16 samples, `T` in front. Escapes: two fragments; no label.

**W5, `label_under_surface`.** Set: `label_placement@2`, a sheet covering the centre (§10). View C.
A square polygon `G` = `[−3, 3]²` at `z = 2`, fill blue, alpha 1, not a face of a face group, then a
camera-facing label `L`: a square glyph `[−1.5, 1.5]²` with a fixed-orientation centre at the
origin, fill white. `G` is a path's fill, a surface (the interim trade), one sheet; it covers `L`'s
centre, where it keys `−20/18 ≈ −1.111` against `L`'s depth `−1`, intervals apart: it qualifies.
Pixel `(32, 32)`: the ordinary order is `G` alone, `L` goes immediately behind it, so `G` in front
is the only admissible order. Defect: at its 16 samples, `L` in front. Escapes: no open case (the
intervals are apart, `G` is one flat sheet, its class is given), `G` is no face or mesh triangle,
so no `label_kind_unknown`; two fragments.

**W6, `alpha_rounds_to_zero`.** Set: `alpha_round@1` (§1). View F. Square `[−4, 4]²`, fill
`(1, 0, 0, 0.003)`: `trunc(0.003 · 65535 + 0.5) >> 8 = 0`. Trace: no fragment anywhere. Defect: a
fragment of it reported at pixel `(8, 8)`, full mask.

**W7, `mesh_inside_margin`.** Set: mesh margin `m = ε + 1/256` (§3). View F. One mesh triangle
with corners `(−6, −6, 0)`, `(6, −6, 0)`, `(0, 6, 0)`, red. Every sample of pixel `(8, 8)` lies at
least 1.9 px inside the unrounded triangle, so `Yes`. Defect: the fragment absent at its 16
samples.

**W8, `cell_byte_outside`.** Set: presence band `μ` (§4). View F. Polygon `[21/32, 4] × [−4, 4]`,
fill white. In pixel `(8, 8)` the cell column `x ∈ [0.5, 0.75]` is covered `0.375`, cell byte
`floor(0.375 · 255 + 0.5) = 96`, and `μ` admits only 96. Defect: those four cell bytes reported as
104.

**W9, `line_before_surface`.** Set: curve `S` (§6). View C. A polygon `[−6, 6]²` at `z = 0`, fill
blue, then an open line from `(−4.5, 0, 1)` to `(4.5, 0, 1)`, stroke white, width `0.5` (2 px).
Pixel `(32, 31)`: the line keys `−20/19 ≈ −1.053`, the surface `−1`. Trace: no shared plane (the
union is 1 unit wide, and the only flat record fails the width test with the line, so no third
record; the line holds `y = 0` and `z = 1` exactly, the surface neither); not two strokes, so no
`lines@1`; `E = {strong line}`. Defect: at the samples both cover, the surface in front. Escapes:
two fragments; the line's `L ≈ 37.9` px is 0.1 px from a step.

**W10, `line_through_surface`.** Set: the may-share set (§7). View C. A polygon `[−6, 6]²` at
`z = 0`, fill blue, then an open line from `(−4.5, 0, −1.5)` to `(4.5, 0, 1.5)`, stroke white,
width `0.5`. At world `x = −3` the line lies at `z = −1`, keys `−20/21 ≈ −0.952`, on screen at
`x ≈ 20.57`: pixel `(20, 31)`, where the surface keys `−1`. Trace: the union is at least 1.2 units
wide, the surface is the only flat record and fails with the line, and the line holds `y = 0`
alone, so no route shares; `E = {strong surface}`. Defect: at the samples both cover, the line in
front. Escapes: no third flat record in the frame; two fragments.

**W11, `faces_past_parallel`.** Set: `near_parallel@1` (§7). View C. A polygon `[−6, 6]²` at
`z = 0`, fill red, then a polygon on `z = x / 4` with corners `(−6, −6, −1.5)`, `(6, −6, 1.5)`,
`(6, 6, 1.5)`, `(−6, 6, −1.5)`, fill blue. Pixel `(40, 32)`: red keys `−1`; blue lies at
`x ≈ 2.07`, `z ≈ 0.518`, keys about `−1.027`. Trace: no shared plane (union at least 1 unit wide,
no third record); dihedral `atan(1/4) ≈ 14°` far past `θ_A + θ_B`; `E = {strong blue}`. Defect: at
its 16 samples, red in front.

**W12, `lines_crossing_apart`.** Set: `lines@1` bound `O⁺` (§7). View C. Line `A` from
`(−3.1875, 0, 0)` to `(3.1875, 0, 0)`, then line `B` from `(0, −3.1875, 1)` to `(0, 3.1875, 1)`,
strokes white then green, width `0.5`. Pixel `(32, 32)` is covered by both: `A` keys `−1`, `B`
`−20/19`. Trace: two strokes of paths with no plane of their own; neither cubic is near a step
(`L = 25.5` and `≈ 26.84` px), there are no joints, every key along `A` is `−1` and along `B`
`−20/19`, so `J = 0` and each `O⁺` is under `10^-6`; the policy's term fails by `0.05`; no shared
plane (two such records share only through a flat record, of which there is none, or by the axis
rule, and `A`'s `z = 0` and `B`'s `z = 1` differ); `E = {strong B}`. Defect: at the samples both
cover, `A` in front.

**W13, `resolve_class_inconsistent`.** Verdict `report_inconsistent`: the recorded class does not
follow from the recorded relations, which says the report contradicts itself, not that the scene
admits no rendering. Set: the selection check (§9). Scene as W1. Pixel `(8, 8)`: both fragments
opaque and full, `B` in front, so `opaque_drop@1` keeps `B` alone and the class is `Fast`. Defect:
the recorded class `Coverage`.

**W14, `gradient_shifted`.** Set: `cairo_linear_gradient@1` (§9). View F. Polygon
`[−8, 8] × [−2, 2]`, fill a gradient of two stops, black `(0, 0, 0, 1)` to white `(1, 1, 1, 1)`,
ends `(−8, 0)` and `(8, 0)`. At pixel `(8, 8)` the admitted positions span under one level, so at
most two adjacent bytes per channel are admitted. Defect: the fragment's colour, and the pixel's,
raised by 2 levels in each colour channel.

**W15, `stroke_under_own_fill`.** Set: `own_layers@1` (§7). View C. Polygon `[−4, 4]²` at `z = 0`,
fill red, stroke blue, width `0.5`. Pixel `(47, 32)` (points `[3.75, 4] × [−0.25, 0]`) is covered
whole by the fill and by the stroke along `x = 4`, and by nothing else. Trace: `E = {weak}`, the
stroke in front. Defect: at its 16 samples, the fill in front. Escapes: two fragments, so no cycle
in which the weak edge could yield.

**W16, `cycle_breaks_strong`.** Set: completions and `cycle_rank@1` (§7). View C. A closed polygon
`P` through `(−3.1875, 0, 1.5)`, `(3.1875, 0, 1.5)`, `(3.1875, 3/64, −1.5)`,
`(−3.1875, 3/64, −1.5)`, coded `planes` Curve (no plane of its own), fill green, background
stroke red of width `0.25` (1 px), no stroke; then a polygon `B` = `[−6, 6]²` at `z = 0`, coded on
the axis plane `z`, fill `(0, 0, 1, 0.5)`. On screen `P`'s near edge lies at `y = 32` and its far
edge at `y ≈ 31.8256`. In pixel `(32, 31)` the fill covers the bottom cell row (`y ∈ [31.75, 32]`,
about 0.70 of each cell), so it is present at samples 12, 13, 14 and 15, whose `y` (§1) are
31.84375, 31.78125, 31.96875 and 31.90625. The background stroke `A` (both edges' bodies, `y` from
about 31.33 to 32.5) covers them too, and `B` covers every sample.

- **Samples 12, 13 and 15** lie nearer the far edge than the near one. `A` is held by both edges
  and reads the near one, key `−20/18.5 ≈ −1.081`; the fill `C` reads its nearest edge, the far
  one, key `−20/21.5 ≈ −0.930`; `B` keys `−1`.
- **Sample 14** (`y = 31.96875`) lies nearer the near edge, so `C` reads it too, key `−1.081`.

Trace at samples 12, 13 and 15: `E(A, C) = {weak}` with `C` in front (`own_layers@1`: the
background stroke under its fill); `E(A, B) = {strong A}` and `E(B, C) = {strong B}` (no shared
plane: the union with `B` is over 2 units wide, `B` is the only flat record, and `P` holds no axis
coordinate). The one completion cycles, and `cycle_rank@1` lets the weak edge yield: the order is
`A, B, C`, and the conforming report passes by that route. At sample 14, `E(C, B) = {strong C}`,
the completion does not cycle, and the order is `C, A, B`. Nothing is opaque and full there, so
nothing is dropped. Defect: at samples 12, 13 and 15 the order `C, A, B`, which keeps the weak edge
and breaks `B` over `C`; no completion gives it there. Sample 14 already has that order and is not
part of the defect.

**W17, `invisible_registry_offer`.** Set: registry eligibility (§5). Two forms. Neither relies on
the width test failing: every record lies on `z = 0`, so the width test passes for every pair, and
the discrimination rests on which records are offered.

- **Primary, the reference scene.** The view: 160 × 90 px, frame `128/9 × 8` units centred at the
  origin (mapping centre the origin), a 3D camera at `phi = 0.8`, `theta = −π/2`, `gamma = 0`,
  focal distance 20, zoom 1, no shading, `cairo_line_width_multiple = 0.01`, input grid `2^-12`, the
  angles and frame sizes as XF carries them (binary32-snapped: `phi = 0.800000011920929`,
  `theta = −1.5707963109016418`, frame width `14.222222328186035`). The records, in draw order:
  - a line from `(−3, −0.1, 0)` to `(3, −0.1, 0)`, stroke `(1, 0, 0, 0.5)`, width 60 (0.6 frame
    units);
  - a line from `(−3, 0.1, 0)` to `(3, 0.1, 0)`, stroke `(0, 0, 1, 0.5)`, width 60;
  - a polygon through `(−4, −1, 0)`, `(4, −1, 0)`, `(4, 0, 0)`, `(−4, 0, 0)`, fill opacity 0, stroke
    width 0 and stroke opacity 0.

  Where both strokes cover a pixel the two lines are an `ExactTie` (both on the axis plane
  `z = 0` by the axis rule, every key one computation), decided by paint order. The core this
  contract describes offers the invisible polygon; its identity attaches one line, and the pair is
  decided by depth: 408 pixels change, among them `(80, 45)`, from `[64, 0, 128]` to
  `[128, 0, 64]`, though Cairo paints the frame byte for byte the same with and without the
  polygon. Verdict on that core: `mismatch` at those pixels' samples, counted
  `invisible_registry_offer`. The fixture holds the scene and that core's report; it needs no
  `defect.json`.
- **Second, synthetic.** View C. Line `R` from `(−4.25, 0, 0)` to `(2.125, 0, 0)`, stroke
  `(1, 0, 0, 0.5)`, then line `U` from `(−2.125, 0, 0)` to `(4.25, 0, 0)`, stroke `(0, 0, 1, 0.5)`,
  both width `0.5`, then a polygon `[−6, 6]²` at `z = 0` with fill and stroke `(0, 1, 0, 0)`.
  Pixel `(32, 32)`: both lines cover it whole. Trace: the polygon paints nothing and is not
  offered, so the lines share by the axis rule alone, `E = {weak}`, `U` in front. Defect: at its
  16 samples, `R` in front by a strong relation. Verdict `mismatch`, counted
  `invisible_registry_offer`: in the counterfactual sets built with the polygon offered, the
  polygon is a flat record passing the width test with each line, so the pair may share through a
  flat identity and every edge is admitted, the injected one included.

**W18, `label_over_line`.** Set: `label_placement@2`, a line never pushes a label back (§10). View
C. The label `L` of W5 (a square glyph `[−1.5, 1.5]²`, fixed-orientation centre at the origin,
fill white), then an open line from `(−3.1875, 0, 2)` to `(3.1875, 0, 2)`, stroke green, width
`0.5` (2 px). Pixel `(32, 32)` is covered whole by both: `L`'s depth is `−1`, the line keys
`−20/18 ≈ −1.111`, nearer. Trace: the line is a stroke, a line, so it never qualifies; no sheet
covers `L`, which is in front of everything: `L` in front is the only admissible order. The
line's `L ≈ 28.3` px lies 0.33 px from a step. Verdict on the core this contract describes, which
draws the line over the label: `mismatch` at the 16 samples, counted `label_plate_model`. The
fixture holds the scene and that core's report; it needs no `defect.json`.

**Label witnesses W19 to W26.** Each plants one of the failures the rule must catch. Their
conforming report is the one a core keeping §10 gives; the core this contract describes keeps the
plate model, so each fixture's `defect.json` applies to a report built by the rule. All use View C
and the label `L` of W5 unless a witness says otherwise, and every polygon is a path's fill (a
surface by the interim trade) except in W20.

**W19, `label_backed_under_line`.** Planted: a surface-backed label drawn over a line above that
surface. A polygon `S` = `[−3, 3]²` at `z = 2`, fill `(0, 0, 1, 0.5)`; then a line from
`(−3.1875, 0, 2.5)` to `(3.1875, 0, 2.5)`, stroke green, width `0.5`; then `L`. `S` covers `L`'s
centre, keys `−1.111` against `−1`: it qualifies. Pixel `(32, 32)`: the ordinary order is the line
(`−20/17.5 ≈ −1.143`), then `S`; `L` goes behind `S`: line, `S`, `L`. Defect: at the samples the
line covers, `L` in front of the line. The line's `L ≈ 29.1` px lies 0.86 px from a step.

**W20, `label_inside_box`.** Planted: a wrong sheet partition. It needs XF 1.5's source
section (without it its faces are `label_kind_unknown`). A cube of
side 4 centred at the origin, a face group whose source path is
`[(c, arachne.mobject.three_d.three_dimensions.Cube)]`, fill `(0, 0, 1, 0.5)`; `L` at its
centre. Its front face (`z = 2`) and back face (`z = −2`) are two sheets; the side faces are edge-on.
The front sheet covers `L`'s centre and is nearer: it qualifies; the back sheet is farther: it does
not. Pixel `(32, 32)`: front, `L`, back. Defect: front, back, `L`, as if the box were one occluder.
Its soundness twin `label_behind_box` puts `L`'s centre at `(0, 0, −4)`: both sheets qualify, and
front, back, `L` is the expected order.

**W21, `label_off_centre_farther`.** Planted: a wrong qualification, by an extended plane. A polygon
`S` through `(0.5, −1.5, −0.25)`, `(1.5, −1.5, −2.25)`, `(1.5, 1.5, −2.25)`, `(0.5, 1.5, −0.25)` (on
the plane `z = 0.75 − 2x`), fill blue. `S` misses `L`'s centre; over the overlap it lies at
`z <= −0.25`, farther than `L`: it does not qualify, though its plane extended to the centre would be
nearer. Pixel `(36, 32)` (`S` at `x ≈ 1`, `z ≈ −1.25`): `L` in front. Defect: there, `S` in front.

**W22, `label_genuine_straddle`.** Planted: a straddle treated as cover. A polygon `S` through
`(0.5, −1.5, 1.2)`, `(2.5, −1.5, −2)`, `(2.5, 1.5, −2)`, `(0.5, 1.5, 1.2)` (on `z = 2 − 1.6x`), fill
blue. `S` misses `L`'s centre; over the overlap it is nearer where `x < 1.25` and settled farther
where `x > 1.25` (the samples of pixel `(37, 32)`, `z < 0` by far more than either interval): a
genuine straddle, which does not qualify. `L`
is in front everywhere. Defect: at pixel `(35, 32)` (`S` at `z ≈ 0.43` to `0.85`, nearer), `S` in
front, the plate model's order: on the core this contract describes, that pixel is `mismatch`
counted `label_plate_model`.

**W23, `label_interval_straddle`.** Verdict `label_rule_open` with cause `interval`, not `mismatch`: the intervals cannot
settle the pair, so the oracle blesses neither order. A polygon `S` = `[0.5, 1.5] × [−1.5, 1.5]` at
`z = 2^-30`, fill blue: nearer than `L` by far less than either interval at every counted point of
the overlap, and missing its centre. Expected: the overlap's samples counted `label_rule_open`
with cause `interval`, whatever order is reported. Beside W22 it separates a decided straddle from an
open one.

**W24, `label_placement_over_order`.** Planted: CE's order applied across placements. A polygon `S` =
`[−3, 3]²` at `z = 2`, fill `(0, 0, 1, 0.5)`; then a label `L2`, the glyph of `L` with its centre at
`(0, 0, 3)`; then `L`. `S` is farther than `L2`'s depth (`−20/17 ≈ −1.176`) and does not qualify
for it; it qualifies for `L`. Pixel `(32, 32)`: `L2`, `S`, `L`, although `L` is painted after `L2`.
Defect: `L`, `L2`, `S`.

**W25, `label_through_translucent`.** Planted: a translucent sheet composited twice. Scene as W5
with `G`'s fill `(0, 0, 1, 0.5)`. Pixel `(32, 32)`: `G`, then `L`; the colour is `G` blended once
over `L` over the background. Defect: the pixel's recorded colour as `G` blended twice over `L`,
every recorded fragment unchanged.

**W26, `label_cross_tile`.** Planted: a pair decided per tile. A wider View C, 128 × 64 px and
frame 32 × 16 units (`x0 = 64`), so a tile edge falls between pixel columns 63 and 64. `L` with its
centre at `(−1, 0, 0)`; a polygon `S` = `[−0.75, 2] × [−3, 3]` at `z = 1`, fill blue, which misses
`L`'s centre and is nearer over the whole overlap (screen `x` from about 60.8 to 66): it qualifies,
and `L` is behind `S` at both pixels `(63, 32)` and `(64, 32)`. Defect: at pixel `(64, 32)`, `L` in
front of `S`.

**W27, `source_row_reparent`.** Set: the source a decision reads (§10). It needs XF 1.5's source
section. View C, a stream of two deltas over the scene of W5 (`G` = `[−3, 3]²` at `z = 2`, opaque,
then `L`):

- **Delta 1** sets `G` with a row: facing group 0, the source path
  `[(g, arachne.mobject.geometry.polygram.Polygon), (v, arachne.mobject.geometry.line.Vector)]`; and `L` with a row of
  facing group 1 and its source. `G` is made by a polygon inside a vector, so it is a line by
  line ancestry (rule 1 of `label_classes@2`) and never qualifies. Frame 1,
  pixel `(32, 32)`: `L` in front of `G`.
- **Delta 2** sets nothing and removes nothing, and carries one held row for `G`'s slot: facing
  group 0, the source path `[(g, arachne.mobject.geometry.polygram.Polygon)]`, the polygon reparented out of the vector. Its drawing
  is untouched. `G` is now a surface; it covers `L`'s centre, nearer: it qualifies. Frame 2, pixel
  `(32, 32)`: `G` in front of `L`.

Defect: frame 2's order with `L` in front, the decision read from delta 1's source. Verdict
`mismatch` at frame 2's 16 samples; frame 1 agrees. A change of facing group alone travels the same
way, as a held row with the drawing untouched (`held_row_clears_group`, below).

**W28, `stream_refused`.** Refusal reason `StreamRefused`; no verdict. A stream of four deltas over
W27's scene, seq 1 to 4: delta 1 sets slot `s` (`G`) and slot `t` (`L`), each with a row; delta 2
carries a held row for a slot `u` that no delta has set; deltas 3 and 4 are valid (each moves `L`).
Expected: frame 1 judged as usual; frames 2, 3 and 4 all `truth_unable` under
`StreamRefused { at_seq: 2, cause: slot_not_held }`, none with a verdict, the valid deltas after it
clearing nothing.

**Refusals, by code.** Each has W28's shape, delta 2 being the refused one, and every row it
carries is well formed unless stated; expected: frames 2 to 4
`StreamRefused { at_seq: 2, cause }` with the code named.

- **`refused_class_named_twice`** (`class_named_twice`): delta 2's class names list
  `arachne.mobject.geometry.polygram.Polygon` twice.
- **`refused_slot_named_twice`** (`slot_named_twice`): delta 2 carries two rows for `s`.
- **`refused_slot_removed`** (`slot_removed`): delta 2 removes `s` and carries a row for `s`.
- **`refused_set_and_removed`** (`slot_removed`): delta 2 sets a new slot `w`, removes `w`, and
  carries a row for `w`: a slot the delta removes, though it also sets it.
- **`refused_row_empty`** (`row_empty`): delta 2's row for `s` has facing group 0 and no paths.
- **`refused_set_row_empty`** (`row_empty`): delta 2 sets a new slot `w` with a row of facing
  group 0 and no paths: the code holds for a set's row as for a held one.
- **`refused_path_empty`** (`path_empty`): delta 2's row for `s` has a path of no objects.
- **`refused_class_out_of_range`** (`class_out_of_range`): delta 2's row for `s` names class index
  `k`, where the section holds `k` names.
- **`refused_slot_not_held`** (`slot_not_held`): W28 itself.

**Precedence.** Each has W28's shape:

- **`refused_class_before_rows`:** delta 2's class names list one name twice, and its first row
  names a slot it removes. Expected `class_named_twice`: the class names are read first.
- **`refused_first_offending_row`:** delta 2's first row is for `s`, which it removes, and has
  facing group 0 and no paths; its second row has a path of no objects. Expected `slot_removed`:
  the first row offends first, and of its two faults `slot_removed` comes before `row_empty`.
- **`refused_twice_before_removed`:** delta 2 removes `t`; its first row is a held row for `s`,
  well formed; its second row is for `s` again; its third is for `t`. Expected `slot_named_twice`:
  the second row is the first that offends, so the removed slot's row after it is never reached.
- **`refused_reading_before_apply`:** delta 2's first row is a held row for `u`, never set, and its
  second row has a path of no objects. Expected `path_empty`: the reader's codes come before
  apply's, and `slot_not_held` comes last.
- **`refused_apply_before_slot_not_held`:** delta 2's order draws a slot `x` no delta has set,
  which apply refuses, and its one row is a held row for `u`, never set. Expected the ordinary
  refusal, not `slot_not_held`: apply's other checks come before it. Frame 2 is `truth_unable`
  under `invalid_delta`, and frames 3 and 4 are judged on the stream as delta 1 left it.
- **`refused_section_before_apply`:** delta 2 removes `s`, sets `t` again naming a shape the
  delta neither uploads nor finds held, which apply refuses, and carries a row for `s`. Expected
  `slot_removed`: a section code comes before apply's ordinary checks.
- **`refused_decode_before_section`:** delta 2's order holds an entry whose layer code the build
  does not know, which the reader refuses as it decodes, and its section lists one class name
  twice. Expected the ordinary refusal: decode runs in the order of the delta's bytes, and the
  order comes before the section. Frame 2 is `truth_unable` under `invalid_delta`; the stream is
  not refused, and frames 3 and 4 are judged on the stream as delta 1 left it.

**`held_row_clears_group`.** W27's scene, `L` set in delta 1 with a row of facing group 7 and its
source. Delta 2 carries a held row for `L`'s slot with facing group 0 and the same source path.
Expected: frame 1 judged as usual. From frame 2, `L`'s facing group is none and its source is the
row's, and its drawing is as delta 1 set it. Frame 2 draws no facing group, so it is judged as one
read as XF 1.4 (§10, `label_ungrouped`): `L`, a camera-facing record, is a label by its centre,
`G` a surface by the interim trade that covers its centre nearer, and at pixel `(32, 32)` `G` in
front of `L` is the only admissible order, which the core's plate model also draws: agreement.

**`label_ungrouped_no_row`.** The scene of W5 in one delta of XF 1.5: `G` set with a row (facing
group 0, the source path `[(g, arachne.mobject.geometry.polygram.Polygon)]`), `L` set with no
row. Expected: no refusal. The frame draws no facing group, so it is judged as one read as XF 1.4:
at pixel `(32, 32)`, `G` in front of `L`, agreement. With a facing group drawn, the same record is
`ungrouped_sheet_open`'s.

**`class_cylinder_grouped`.** The scene of W5 in one delta of XF 1.5: `L` set with a row of facing
group 1 and its source, and `G` set with a row of facing group 0 and the source path
`[(y, arachne.mobject.three_d.three_dimensions.Cylinder), (b, arachne.mobject.three_d.three_dimensions.Cube)]`, a cylinder's own
record inside a cube. Expected: `G` is `label_rule_open` with cause `cylinder_grouped` (rule 4;
rule 3 never reads a cylinder through its container), so `L` is expected on top and the samples of
pixel `(32, 32)`, which the pair decides, are counted `label_rule_open` with cause
`cylinder_grouped`, neither `mismatch` nor agreement. Reading the cube as the cylinder's container would
make `G` a surface that qualifies, with `L` behind it.

**`class_part_follows_cylinder`.** As `class_cylinder_grouped`, with `G`'s source path
`[(c, arachne.mobject.geometry.arc.Circle), (y, arachne.mobject.three_d.three_dimensions.Cylinder),
(f, arachne.mobject.three_d.three_dimensions.Surface)]`: a cap under a cylinder under a surface.
Expected: the walk of rule 3 stops at the cylinder, whose outcome where it stands, inside the
surface, is open: `G` is `label_rule_open` with cause `cylinder_grouped`, counted as in
`class_cylinder_grouped`. Following the surface past the cylinder would make `G` a surface that
qualifies, with `L` behind it.

**Witnesses of the point reading (`label_placement@2`, §10).** Each is one delta of XF 1.5. A
label `L` is set with a row of facing group 1 and its source, and every other record with a row of
facing group 0 and the source path given. `Polygon` below is
`arachne.mobject.geometry.polygram.Polygon` and `Surface` is
`arachne.mobject.three_d.three_dimensions.Surface`. All use View C unless a witness says otherwise.

**`label_points_sliver`.** Set: the overlap is read at the order points. View O: 128 × 64 px, the
3D camera of View C with focal distance `10^6`, in effect orthographic: `xx = 4`, `yy = −4`,
`x0 = 64`, `y0 = 32`, so a point near `z = 0` lands within `10^-4` px of `(64 + 4x, 32 − 4y)`.
`L` is a rectangle glyph over screen `x` 70 to 110 and `y` 28 to 36, at `z = 0`, fill white, its
fixed-orientation centre at `(4, 0, 0)` (screen `x` 80). `S`, source `[(s, Polygon)]`, fill blue,
alpha 1, spans screen `x` 90 to `e` and `y` 30 to 34, on the plane whose depth at screen `x = p` is
`z = (c − p) / 40`: nearer than `L` left of `c`, farther right of it. `S` misses `L`'s centre, so
it is decided by the points. The order points of pixel column 100 sit at `100 + (2k + 1) / 32`.
Three variants, `(c, e)`:

| Farther sliver, screen `x` | Order points it holds | Expected |
|---|---|---|
| `c = 100.10` to `e = 100.15` | none (the points are at 100.09375 and 100.15625) | `S` qualifies |
| `c = 100.08` to `e = 100.12` | 100.09375 | `S` does not qualify |
| `c = 100.225` to `e = 100.275`, across the cell boundary at 100.25 | none (the points are at 100.21875 and 100.28125) | `S` qualifies |

Expected: where `S` qualifies, `L` is behind `S` at every sample both cover (at pixel `(95, 32)`,
`S` in front); where it does not, `L` is in front everywhere. Rejects: reading the overlap by area,
so that a sliver between points still counts, which leaves `L` in front of `S` in the first and
third variants.

**`sheet_join_quantum`.** Set: fills join at two shared corners, rounded to 1/4096 px. `L` is the
glyph of W5 with its centre at `(0, 0, −4)` (depth `−20/24 ≈ −0.833`). Two fills of one surface
object, at `z = 0` (key `−1`, nearer), fill blue, alpha 1: `F1` through `(−2, −1, 0)`, `(0, −1, 0)`,
`(0, 1, 0)`, source `[(f1, Polygon), (o, Surface)]`, and `F2` through `(0, −1, 0)`, `(2, 0, 0)`,
`(0, 1 + δ, 0)`, source `[(f2, Polygon), (o, Surface)]`, wound the same way. On screen they share
the corner `(32, 36)` exactly, and `F2`'s third corner `(32, 28 − 4δ)` meets `F1`'s `(32, 28)`.
With `δ = 2^-16` (`4δ = 2^-14` px, a quarter of the quantum) both round to the same corner, so
`F1` and `F2` join into one sheet, which holds `L`'s centre on their shared edge and is nearer:
it qualifies. Expected: `L` behind the sheet at every sample both cover (at pixel `(30, 32)`, the
sheet in front).

**`sheet_join_near_miss`.** As `sheet_join_quantum` with `δ = 3 · 2^-16` (`4δ` is three quarters
of the quantum): the corners round to neighbouring values, so `F1` and `F2` are two sheets of one
object, wound the same way, each counting points with `L`. Expected: both pairs open with cause
`disconnected`, `L` expected in front, and the samples each pair decides counted
`label_rule_open` with cause `disconnected`, at pixel `(30, 32)` (in `F1`) as at pixel `(34, 32)`
(in `F2`). Rejects: leaving the first sheet decided and only the second open, which puts `L`
behind `F1` at pixel `(30, 32)`.

**`sheets_opposite_winding`.** As `sheet_join_quantum` with `δ = 0` and `F2`'s corners listed the
other way round, so it is wound opposite to `F1`: two sheets of one object that never join and are
not disconnected, each decided on its own. Both hold `L`'s centre on their shared edge and are
nearer: both qualify. Expected: `L` behind each sheet where it covers. Rejects: reading the two as
disconnected, which leaves `L` in front, counted open.

**`no_pair_centre_without_points`.** Set: no counted point, no pair, even over the centre. `L` is a
square ring glyph at `z = −4`, its centre at `(0, 0, −4)`: outer square `[−1.5, 1.5]²`, inner hole
`[−0.5, 0.5]²` wound the other way, so on screen it covers `x` and `y` from 27 to 37 outside the
hole from about 30.33 to 33.67. Two fills of one surface object at `z = 0`, wound the same way and
sharing no corner, fill blue, alpha 1: `S1` = `[−0.25, 0.25]²` (screen 31 to 33 in both axes),
source `[(s1, Polygon), (o, Surface)]`, which holds `L`'s centre but whose order points all lie in
the hole; and `S2` = `[1, 1.25] × [−0.25, 0.25]` (screen `x` 36 to 37, `y` 31 to 33), source
`[(s2, Polygon), (o, Surface)]`, inside the ring, nearer, missing the centre. Expected: `S1` counts
no point and makes no pair; `S2` alone meets `L`, is decided by its points and qualifies: at pixel
`(36, 32)`, `S2` in front of `L`. Rejects: making a pair of `S1` because it holds the centre, which
makes `S1` and `S2` disconnected and leaves pixel `(36, 32)` counted open.

**`pair_cause_mesh`.** The scene of W5 with `G` drawn as a mesh of two triangles over
`[−3, 3]²` at `z = 2`, fill blue, alpha 1, source `[(g, Surface)]`: a surface that covers `L`'s
centre nearer and would qualify. Expected: the pair is open with cause `mesh`: `L` expected in
front, and the samples of pixel `(32, 32)` counted `label_rule_open` with cause `mesh`.

**`pair_cause_label`.** The scene of W5, `G` a polygon with source `[(g, Polygon)]`, and `L`
carrying, in its facing group, one more record: a pre-projected path. Expected: every pair of `L`
is open with cause `label`; `G`, which would qualify, leaves `L` expected in front, and the samples
of pixel `(32, 32)` are counted `label_rule_open` with cause `label`.

**`pair_cause_precedence`.** `pair_cause_mesh`'s `G` with `pair_cause_label`'s `L`: both causes
apply, and `label` comes first. Expected: the pair open with cause `label`, counted under `label`
and never under `mesh`.

**`insertion_at_points`.** Set: insertion anchors only where the sheet holds the order point.
`label_points_sliver`'s third variant (`c = 100.225`, `e = 100.275`), where `S` qualifies. At pixel
`(100, 32)`, `S` covers the first cell column whole and a tenth of each cell of the second column,
whose order points (at `x = 100.28125`) it does not hold. Expected: at the four samples of the
first column, `S` in front of `L`; at the four of the second, `L` in front of `S`. Rejects:
anchoring the label wherever the sheet covers part of a sample's cell, which puts `L` behind `S`
at the second column's samples too.

**`gradient_clear_point`.** Set: a fill's alpha at a point is the painted pixel's. The scene of
W5, `G` with source `[(g, Polygon)]` and its fill a linear gradient from opaque blue at `x = −3` to
clear at `x = −0.5`, pad extend, so its alpha is exactly 0 for `x` past `−0.5`. `G` does not cover
`L`'s centre, whose pixel's alpha is 0, so it is decided by the points: it counts only the points
where its alpha is positive (`x` below about `−0.5`), is nearer at all of them, and qualifies.
Expected: at pixel `(28, 32)` (`x ≈ −0.9`), `G` in front of `L`; at pixel `(36, 32)` (`x ≈ 1`,
alpha 0), `L` in front: `G` neither counts nor anchors there. Rejects: reading the outline without
the alpha, which holds the centre and anchors `L` behind `G`'s clear part.

**`gradient_alpha_straddle`.** Set: the oracle's reading of a gradient's alpha. As
`gradient_clear_point`, with the clear end placed so that the last visible pixel `q` before it, in
`L`'s overlap, has an admitted alpha range (§9, *Oracle reading for gradients*) that holds both 0
and a value of 1/255 or more: some run start the row allows paints it clear, another does not.
Expected: `q` is oracle-unable, its samples reported as `label_alpha_straddle`, never agreement,
whatever the core reports there; the other pixels are judged as in `gradient_clear_point`.

**`centre_on_horizontal_edge`.** Set: the centre test holds a point on a horizontal edge. `S`
through `(−3, 0, 2)`, `(3, 0, 2)`, `(3, 3, −7)`, `(−3, 3, −7)` (on the plane `z = 2 − 3y`), source
`[(s, Polygon)]`, fill blue, alpha 1; then `L` of W5, its centre at the origin. On screen `S`'s
bottom edge, at `y = 0`, is horizontal and passes through `L`'s centre `(32, 32)`, a whole pixel,
so its alpha is read at the pixel box's nearest pixel `(32, 31)`, where it is positive. `S` holds
the centre, nearer there (`z = 2`): it qualifies the whole pair, though it lies farther than `L`
above `y ≈ 0.67`. Expected: `L` behind `S` at every sample `S` holds, at pixel `(32, 31)` as at
pixel `(32, 28)` (`S` farther there). Rejects: leaving the horizontal edge unheld, which decides
`S` by the points, finds the straddle and leaves `L` in front.

**`ungrouped_sheet_open`.** Set: an ungrouped record kept facing the camera is an open sheet. View
C. `U`, a square glyph `[−3, 3]²` with a fixed-orientation centre at `(0, 0, 2)`, fill blue, alpha
1, set with a row of facing group 0 and the source path `[(u, Polygon)]`; then `L` of W5, without
W5's `G`, so `U` is the only record that could hide `L`. The frame draws a facing group (`L`'s).
`U` covers `L`'s centre, nearer, and would qualify as a polygon. Expected: the pair of `L` and `U`
is open with cause `label`, and no sheet qualifies, so `L` is drawn in front of `U` at every sample
of pixel `(32, 32)`, those samples counted `label_rule_open` with cause `label`. Rejects: reading
`U` by its source as a surface, which qualifies it by the centre and puts `L` behind it.

**`label_drawn_otherwise`.** Set: a label not drawn wholly as placed paths is read over the whole
frame. `L` of W5 with one more record in its facing group: a points record at `(2.5, 0, 0)`.
`S` = `[2, 3] × [−0.5, 0.5]` at `z = 2`, source `[(s, Polygon)]`, fill blue, alpha 1, which meets
`L`'s point block and not its glyph. Expected: `L` meets `S` where the point block's fragments and
`S` hold one order point, and the pair is open with cause `label`: the label expected in front,
those samples counted `label_rule_open` with cause `label`. Rejects: reading the label by its
path outlines alone, which makes no pair and counts nothing.

### Soundness fixtures

**`gradient_dense_stops`.** View F. Polygon `[−8, 8] × [−2, 2]`, fill a gradient of 1,025 opaque
stops alternating black and white (stop 0 black), ends `(−8, 0)` and `(2, 0)`: device `x` 0 to 10.
A run starting at pixel 0 gives pixel 2 the position 16383 (colour about 4); its centre gives 16384
(colour 0). Both are admitted, so the core's colour passes, whatever its run. A tolerance of one
level would reject it.

**`lines_straight_cubic_step`.** View C, input grid 0. An open line from `(−0.75, 0, 0)` to
`(0.75, 0, 0)` (controls 6 px apart end to end, every key `−1`), stroke white, width `0.5`. Its
cubic sits on a step (`L = 6`), so the core charges `320u (1 + 16u)` there. Expected: the core's
reported share is at most `O⁺` (§7: `B` gives at least `328u`), no `share_beyond_bound`.

**`selection_relations`.** Scene as W16, pixel `(32, 31)`, the conforming report. The selection
check (§9) reads the trace's pre-drop per-sample pair relations and strengths there:

- `A` against `B`: strong, `A` in front, at every sample both cover (the cells `A` covers, rows 1
  to 3);
- `B` against `C`: strong, `B` in front, at samples 12, 13 and 15; strong, `C` in front, at sample
  14;
- `A` against `C`: weak (`own_layers@1`), `C` in front, at samples 12 to 15.

It checks each against its `E` (W16's trace), then derives the pixel's votes (`A` over `B` by
`Depth`; `B` over `C` by `Majority`, three samples to one; `A` and `C` by `Paint`), the rank
`A, B, C` (the weak edge yields in the cycle), and from them the drops, units and class by §9's
rule. Expected: agreement with the recorded votes, ranks, units and class.

**Stroke behaviour fixtures (§8).** Each holds a scene whose expected coverage follows from §8
alone; the conforming report agrees, and the named wrong reading would not.

- **`join_occurrences`.** View F. One path record, stroke white, width 1, joint miter, cap butt,
  no fill, of four open subpaths of two straight cubics each. Subpaths 1 and 2 are the same spike,
  `(−7.1, 4)` to `(−1, 4)` and back to `(−7, 5)`; subpaths 3 and 4 the same spike 8 units lower,
  `(−7.1, −4)` to `(−1, −4)` and back to `(−7, −3)`. `joins` bytes per cubic
  `0, 0b11, 0, 0b01, 0, 0b01, 0, 0b11`.
  - **Clear of `curve_recut`.** Each spike starts at `x = −7.1`, not `−7`. From `x = −7` its first
    cubic had `L = 6`, on a step, 0 px from it, and so exposed; from `−7.1` it has `L = 6.1`, 0.1 px
    clear of the step 6. Its second cubic keeps `L = √37 ≈ 6.083`, 0.083 px clear of 6. Both clear
    the 0.06 px every curve here keeps. The legs keep their directions, so the corner, its miter
    point and the tips below are unchanged.
  - **The keys.** Each spike's join at its corner is one key, met twice, once in each of its
    subpaths. The upper key's decisions are miter, then bevel; the lower key's are bevel, then
    miter. The corner is sharp (an angle of about 9.5° between the legs), so kurbo bevels it: its
    miter point `m`, about `(5.04138, 3.5)` for the upper spike, lies about 6.06 units from the
    corner, beyond the limit's 5.
  - **Each edit alone in its subpath.** A carried miter inserts the tip in its own subpath's
    contour, where its lines are the only ones, so no edit is voided. The two copies of a spike
    are drawn over each other, so a spike shows the tip when either of its occurrences mitres.
  - **The tips.** The upper tip is about `(3.98296, 3.58760)`, its triangle closed by the bevel's
    ends `(−1, 3.5)` and `(−0.91780, 4.49320)`; the lower is the same 8 units lower.
  - **Expected** (both tips drawn): pixel `(8, 4)` (points `[0, 1] × [3, 4]`) covered over about
    0.474 of its area, its upper half above frame `y` ≈ 3.52, by the upper tip; pixel `(8, 12)`
    (points `[0, 1] × [−5, −4]`) the same by the lower tip. No other part of the record reaches
    either pixel.
  - **Rejects.** Applying each key's first decision to both occurrences bevels the lower spike:
    pixel `(8, 12)` uncovered, every cell byte 0. Applying each key's last decision to both bevels
    the upper spike: pixel `(8, 4)` uncovered.
- **`join_closing`.** View F. A closed square through `(−4, −4)`, `(4, −4)`, `(4, 4)`, `(−4, 4)`,
  stroke width 1, joint miter, bytes `0b01, 0b11, 0b11, 0b11`. The closing join at `(−4, −4)` takes
  the first cubic's byte, a bevel; the other three corners mitre. Expected: the outer corner square
  of `(−4, −4)` (pixel `(3, 12)`, its cells beyond the bevel) uncovered, the other three outer
  corners covered. Reading the closing join from the last cubic's byte would mitre it.
- **`guarded_seam`.** View F. A closed subpath through `(−4, −4)`, `(4, −4)`, `(100000, −4)`,
  `(100000, 4)`, `(−4, 4)` and back, as straight cubics, stroke width 1, joint miter, dashed with
  period 2, duty 0.75 and phase 0.25. Its far points bring the guard into force; the pieces beyond
  the area miss and are measured, so the subpath has two runs, and its first and last pieces do
  not miss. Its length, 200024, puts the pattern on at both ends of the subpath, so the last run's
  final dash is joined onto the first run's opening dash at `(−4, −4)`. Expected: that corner
  stroked as a mitred join (pixel `(3, 12)`'s outer cells covered), as without the guard. Dashing
  the runs apart would leave two butt caps there.
### Classes outside what the oracle promises

These are reported by class for a human to view, never counted as ties:

- **One plane, through a flat identity.** Orders between a record with no plane of its own and a
  record it may share a flat identity with. The oracle admits every order (`Policy(one_plane@1)`),
  because the partition fixes those rows.
- **Unbounded bands.** Samples whose band is unbounded (`band_unbounded`, §6). They are
  `truth_unable` per sample.
- **Rebuilt centrelines.** Reads exposed to a change of a cubic's count or of a joint's join
  (`curve_recut`, §6). They are `truth_unable` per sample.
- **Far gradient ends.** Gradient layers with an end or a pixel's point beyond the fit limit
  (`gradient_far_ends`, §9). They are `truth_unable` for that layer's pixels.
- **Open label cases.** Samples counted `label_rule_open` (§10), only by its named cases: the
  pair causes `label`, `mesh`, `folded`, `disconnected` and `interval`, and, on the other axis, a
  class left open by `label_classes@2` under its class cause (`unmapped`, `generic_only`, `paths_disagree`,
  `no_source`, `cylinder_grouped`, `point_cloud`).
- **A sheet's alpha the oracle cannot place.** Pixels where the label rule reads a gradient
  fill's alpha and the range the oracle admits holds both 0 and a value of 1/255 or more
  (`label_alpha_straddle`, §10). They are named and never agree.
- **Label against an object of unknown kind.** Samples where a face of a face group or a mesh
  triangle, which a stream read as XF 1.4 cannot place on the line or the surface
  side, decides a label's placement (`label_kind_unknown`, §10). The orders of both readings are
  admitted. The class does not arise in a stream read under XF 1.5's rules.

### Findings counted under their own names

These are `mismatch` or failing results that name a known way the core departs from this text:

- `zero_alpha_fragment` (§1): a mesh fragment whose alpha byte is 0.
- `invisible_registry_offer` (§5): a relation admitted only with records that paint nothing offered.
- `label_plate_model` (§10): a label sample ordered as the plate model orders it and not as the
  rule does, such as a nearer line or arrow drawn over a label, or a label split by a surface
  crossing its depth.
- `share_beyond_bound` (§7): a reported `lines@1` share above `O⁺`, a core finding once the
  contract is proven, and until then triaged by name.

The share of samples in each tie label and each class above is reported per case, so the cost of
conservatism is measured on real scenes.
