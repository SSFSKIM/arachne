# XF, the Frame IR

XF describes frames for a drawer. A **stream** holds what a drawer keeps across frames: shapes,
paint tables, objects, records, the view and the draw order. Each drawn frame is a **delta** against
the stream, empty for a held frame. A **record** describes one drawn object; a **shape** is geometry
uploaded once under its content key. The Rust types are in `src/xf/mod.rs`; this file describes their
meaning and the binary form (`src/xf/wire.rs`), which fixtures and dumped streams use.

## Versions and refusal

The version is (major, minor). This build is **XF 1.5**. From here the layout is append-only:
a new field or kind bumps the minor, and nothing removes or retypes a field. A writer emits the
lowest supported version that carries a delta's content: **1.0 without a pre-projected tail, 1.1
with one, 1.2 with an input grid (on the view or a sub-frame's view) or a record's planes or normals,
1.3 with a face group's face ends or face widths, 1.4 with a record's joins, 1.5 with a record's
source or facing group, or a held record's new one**. A file's version is the highest version its deltas need (1.0 for an empty file).
Explicit foreign or newer version declarations are retained, since this writer cannot determine
what their content requires. `XF_VERSION` names the build's supported version, not every delta's.

- A reader refuses a major it does not know, and reads a newer minor of its own major, skipping what
  that minor appended (every record, shape, paint table, object and view is length-prefixed, and
  so is each delta of a file: what follows a delta's last known section is skipped).
- A record kind the reader does not know is kept, with its slot and object, and its body unread.
- A drawer refuses a whole frame (`render.unsupported`) when a record in the draw order has a kind or
  a feature bit it does not draw, or the view names a light model it does not know. It never draws a
  frame partly. A record outside the order is not drawn, so it is not refused.

XF 1 defines feature bit 0 (value 1), `background_image`: the record's fill and strokes are painted
through CE's background-image fill. CE draws the path's paint into a separate array and multiplies
it by an image at the frame's pixel size. The fixed `fill_image` field holds that image's key and is
present exactly when bit 0 is set; for a known record kind a decoder refuses a disagreement as
invalid XF (`XfError`). The image must be an image shape held by the stream or uploaded in the delta,
and a key still used by `fill_image` cannot be released. This build refuses drawn records with bit 0
(`render.unsupported`): **drawn by Cairo only**.

XF 1.1 defines feature bit 1 (value 2), `pre_projected`: a scene's own camera class, whose
projection or shading differs from the engine's, carries what it produced. A tail after the kind
body is present exactly when the bit is set. It is valid only on `path`, `points` and `image`
records; every other known kind, including `face_group` and `mesh`, is invalid XF (`XfError`).
The decoder reads the tail if and only if the bit is set; the encoder asserts agreement between
the bit and its optional field. The points key must name a path shape held or uploaded, the
three paints' table keys must exist, and no key still used by the tail can be released.
This build draws bit-1 `path` and `points` records from the tail alone: the tail's points as
placed, its paints and gradient ends as produced, never projected or lit again, keyed in depth in
the camera's own axes (the view's angles are never used for them); a bit-1 `image` is Cairo's
overlay's, as every image is. An XF 1.0 reader skips the tail under its newer-minor rule, but
still reads the feature bit in the fixed header; its drawer refuses the unknown bit rather than
projecting or shading those points again. Bytes after the tail remain newer-minor fields and
are skipped. Geometry flags are unchanged: bits outside `0b111` remain invalid.

XF 1.2 carries snapped geometry. A writer may snap a delta's geometry to a grid (so that a scene's
arithmetic, which differs between machines in its last bits, reaches the drawer as the same bytes
everywhere). It then states the grid as the view's `input_grid` (scene units; absent when the
geometry is not snapped), and it decides how each path a record draws lies before it snaps, on the
values as made, carrying the decision as the record's `planes`: one planarity code per path, in
the order the record draws them (a face group's faces, an instances record's members, else one).
Snapped, the points of a curve and of a plane can be the same points; carried, the decision never
depends on the grid. The codes: 0 a curve (keyed by its curve's depth), 1 a plane of Newell's
normal through the first point, 2 a plane of the widest triangle's normal on the first point
(an outline crossing itself), 3 a surface's face on no one plane (Newell's normal through the
centroid), 4, 5, 6 the axis plane x, y or z through the first point (points on a line sharing
that coordinate). The drawer fits the decided plane to the points it is handed. A record without
`planes` is decided by the drawer from its points, as before XF 1.2: right for geometry that was
not snapped. The binding's `Stream.planes(slots)` gives the codes for records a stream holds,
decided as the drawer decides them, for a writer to carry.

The same holds for what CE's lighting derives from nearby points. A record the camera shades
(`shade_in_3d` under a camera that shades) is lit at two corners of each path it draws, by CE's
normal there: a cross product of the chords to the points three on and three back, which on a
finely sampled curve are nearly opposite, so the snap's step turns it by tens of degrees. The
writer decides those normals on the values as made and carries them as the record's `normals`:
two per path, in the order of `planes`. A corner is degenerate when a point its normal is found
from lies off it by less than one grid step in every coordinate, coincident points included (a
sphere's pole or a cone's apex, whose points coincide or lie the float noise of `cos(pi / 2)`
apart): CE's normal there is its fixed fallback or the noise's direction, whichever the rounding
of the machine that made the points gives, so the writer carries the zero vector and the drawer
lights that corner by the face's Newell normal over the points it is handed (where that cancels
too, CE's normal from them). Without `normals` the drawer
finds them from its points, as before XF 1.2. `Stream.normals(slots, grid)` gives them for records
a stream holds. An XF 1.1 reader skips these fields under its newer-minor rule.

So does a stroke's miter join (XF 1.4). Cairo decides each join from slopes between CE's points
rounded to its 24.8 fixed point (`_cairo_spline_init`'s slopes, `outer_join`'s
`2 <= ml² (1 + in · out)`); where a control arm is a few 1/256 px long, or the turn is near the
miter limit, the snap's step moves that decision. The writer decides each join of a path that has
such a join on the values as made and carries the decisions as the record's `joins`: one byte per
cubic of its points, bit 0 set where Cairo joins into the cubic, bit 1 where it mitres there; for
the first cubic of a closed subpath, the closing join (none where Cairo strokes a closing line
first). The drawer strokes as before and, at a join whose carried decision differs from its own,
drops its miter or puts one in, its tip at the miter limit's length (on parallel tangents, where no
tip can be placed, the bevel stays). Without `joins`, or for a join the record does not decide, the
drawer decides from the points it is handed. Only a path record carries `joins`, one byte per cubic
of the points it draws (its shape's cut to its reveal window, or its pre-projected tail's); `apply`
refuses them on any other kind or at any other count. An XF 1.3 reader skips the field.

XF 1.5 says which objects a record was drawn from, for readers that classify what a record is (a
label, a line, a surface) by its objects rather than its geometry. A record's `source` is one or
more paths of containment, each the record's owner first (the leaf, or the surface a face group
draws), then the objects that contain it, outermost last; a child shared by two parents has a path
through each. Each node is a token the writer keeps for that object across the stream, and the
object's nearest library class, fully qualified (a scene's own subclass is named by the library
class it derives from; no address and no name of the scene's own is written). A record's
`facing_group` is a token its writer gives a camera-facing label: the records of one label share
it, and two labels never do (0: none). The scene lowering writes both. A drawer may read them to
place camera-facing labels (this core does: `label_plan.rs`); a frame with no facing group is
the same bytes with them or without. A record set again replaces both, and a record
set without them has neither: they never outlive their record. When a record's objects move (it is
reparented) and its drawing does not, its writer does not set it again: a row for a slot the delta
neither sets nor removes gives the held record its new source and facing group, so the bytes an
older minor carries do not move. They are carried in a section after the delta's order, keyed by
slot, its class names stated once per delta, so an XF 1.4 reader, which stops after the order,
reads the delta as it was. A reader of XF 1.4 that writes the delta again writes it without them.

What a row is. A row is keyed by its slot alone. A row whose slot the delta sets is that record's
source and facing group (of a slot set more than once, its last record's); a record the delta sets
with no row has neither. Any other row is a held row: it replaces both fields of the record the
stream holds in that slot (a field the row does not carry, a facing group of 0 or no paths, becomes
none), and leaves its drawing as it was. The section is present, and the delta is XF 1.5, when it
holds at least one row of either kind. The order of the rows decides only which code a refused delta
reports; the state a valid delta leaves does not depend on it.

Facing groups. A writer of XF 1.5 gives every record it sets with a fixed-orientation centre a
facing group, the records of one camera-facing label sharing it, and gives a held one a row whenever
its group changes; scene lowering sets no fixed-orientation record outside such a label, and writes
the section in the first delta that sets a record. A stream is read under these rules at a frame
once a delta at or before that frame has carried the section; a reader decides this in stream order
and never from a later delta. Until then the stream is read as XF 1.4 reads it, and its records
carry no facing group. In a stream read under these rules, a fixed-orientation record that has no
facing group (set with no row, by a delta of an older minor, or given 0 by a row) is not thereby
outside every label: whether it belongs to one is unknown, and a reader that classifies labels
counts it as unknown, never as a record of another kind. Its drawing is unaffected, and no reader
refuses it.

Refusals, each by its code. A reader refuses the delta (invalid XF) with `class_named_twice` when
the section's class names hold one name twice; then, reading the rows in the order written, at the
first row that offends, with the first of these that applies to it: `slot_named_twice` (a slot an
earlier row named), `slot_removed` (a slot the delta removes, whether or not it also sets it),
`row_empty` (a row of either kind with a facing group of 0 and no paths: a writer with nothing to
carry for a slot writes no row), `path_empty` (a path of no objects), `class_out_of_range` (a class
index at or past the section's names). `apply` refuses the delta with `slot_not_held` for a held row
whose slot the stream does not hold before the delta. A delta has exactly one refusal, the first of
all XF's checks in this order: the reader's, in the order of the delta's bytes (so a refusal of
anything written before the section is the delta's refusal, not one of these codes, and the
section's codes come in the order above); then `apply`'s other checks; then `slot_not_held`, which
is the delta's refusal only when nothing else in it is refused. A refused delta is not applied.

A newer minor that appends after the section writes it, empty, when no record carries either.

This build draws the background (a colour, or an image cropped to the frame as CE crops it); `mesh`
records filled with one colour; `path` records, their paints one colour or a linear gradient between
the record's ends (coloured as Cairo and pixman colour it), lit as CE lights a `shade_in_3d` path
under a camera that shades, and drawn through a reveal, a blend and a paint blend as CE's
interpolation makes them; `face_group` records, each face as the path of that face; `points`
records; and `instances` records of a path or a mesh. It refuses a clip on any of them; a blend,
reveal or paint blend on any kind but a path, a blend of shapes of different lengths and a paint
blend of row counts that do not broadcast; a path with a gradient paint and no ends, or an instance's gradient; a dash it
cannot read; CE's lighting under another light model; a mesh with a stroke, a gradient fill or a
dash; and the reserved kinds. `image` records, records fixed in frame and background-image fills
are drawn by Cairo over the core's frame, in CE's order (the overlay). How records are drawn is
described under "Drawing" below.

### Where each check is made

Besides the section's codes, the core refuses a delta for each of these, whole. In Python every
refusal of either kind is `arachne_core.XfError`, carrying the Rust error's text.

**Decode** (`wire::decode_delta`), in the order of the delta's bytes, each an `XfError`:

- anywhere: bytes that end inside a field or an item, or a count of items longer than the bytes left
  (`Truncated`); a bool byte other than 0 or 1, a string that is not UTF-8 (`Invalid`);
- the version: a major other than the build's (`Version`);
- the time: a denominator of 0 or less (`Invalid`);
- the view: a background or camera code the build does not know (`Invalid`); a light model code it
  does not know is kept, and drawing refuses it;
- each uploaded shape: an image whose width × height × 4 overflows (`Invalid`); for a path, mesh,
  image, volume or field shape, a key other than its content's (`KeyMismatch`); a shape code the
  build does not know is kept, its key unchecked;
- each paint table: a key other than its rows' (`KeyMismatch`);
- each record: geometry flags outside `0b111`; a paint, joint, cap, space or image-source code the
  build does not know; the `background_image` bit and the `fill_image` field disagreeing; the
  `pre_projected` bit on a kind other than `path`, `points` or `image` (each `Invalid`). A record
  kind the build does not know is kept, its body unread and unchecked;
- the order: a layer code the build does not know (`Invalid`);
- the section: its codes, above.

A file (`wire::decode_file`) is refused for its magic (`BadMagic`) or its major (`Version`) before
its deltas are read. The reader checks nothing against what a stream holds, and, but for the
section's codes, no part of a delta against another: a record or a view naming a shape or a paint
table the delta does not upload, an order drawing a slot the delta does not set, a removal or a
release of what is not there, are all read; whether they are there is known only against the stream,
by `apply`.

**Apply** (`Stream::apply`), against the stream's state and the delta as a whole, in this order,
each a `StreamError`:

1. a major other than the build's (`Version`), as the reader's first check;
2. the section's codes, for a delta built in memory (`Invalid`, by its code; a delta the reader read
   has passed them);
3. a key uploaded twice in the delta, as two shapes, two paint tables or one of each (`Invalid`);
4. each uploaded shape: a mesh triangle naming a vertex past its vertices, mesh normals other than
   one per vertex, an image whose bytes are not width × height × 4; a key the stream holds as a
   paint table, or as a shape of other content (each `Invalid`);
5. each uploaded paint table: a key the stream holds as a shape (`Invalid`);
6. a release of a key the stream does not hold and the delta does not upload (`Missing`);
7. a removal of a slot the stream does not hold and the delta does not set (`Missing`);
8. a release of a key that a record the delta leaves, or the view, still uses (`InUse`);
9. when the delta removes or sets a record, the order of every sub-frame display it leaves, checked
   as in 12;
10. each record the delta sets and does not remove, but one of a kind the build does not know: the
    `background_image` bit and `fill_image` disagreeing, the `pre_projected` bit and its tail
    disagreeing, the tail on a kind other than `path`, `points` or `image` (the reader's checks
    again, which only a delta built in memory reaches), a kind that needs a shape given none, a
    shape of a kind the record cannot take, a sub-frame display given a shape, a body of another
    kind's (each `Invalid`); a shape (its own, its blend's, its clip's, its fill image, its tail's
    points) or a paint table neither held nor uploaded (`Missing`; one the delta releases is
    refused by 8 first); a sub-frame's view and order, as in 11 and 12; a face group's,
    points record's or instances record's table of other than one row per face, point or placement,
    an instances record whose placements and objects differ in number, a face group's face ends or
    widths (XF 1.3) and joins (XF 1.4), as described above (each `Invalid`);
11. the delta's view: soft diffuse light outside its ranges, a background image that is not an image
    shape (`Invalid`) or neither held nor uploaded (`Missing`);
12. the order, the delta's or, when it has none, the stream's: a slot not set (`Missing`); two
    displays sharing a capture rank in one layer, a part past the record's parts (each `Invalid`);
13. `slot_not_held` (`Missing`).

So a shape key that a record names and that the delta neither uploads nor finds held is refused by
`apply` (10), never by the reader. An order entry names a slot, not a key: a slot not set is refused
by `apply` (12), and a layer code the build does not know by the reader. A body of another kind's,
and an instances record whose placements and objects differ in number, are reached only by a delta
built in memory: on the wire a record's kind decides its body, and its placements and objects share
one count.

## Content keys

A key is BLAKE2b with a 16-byte digest (Python: `hashlib.blake2b(canonical, digest_size=16)`) over
canonical bytes: the kind code (u8), the number of arrays (u8), then per array its dtype code (u8:
1 = f64, 2 = u32, 3 = u8), its number of dimensions (u8), each dimension (u64), and its data, all
little-endian.

| What | Kind code | Arrays |
|---|---|---|
| path shape | 1 | points f64 (n, 3) |
| mesh shape | 2 | positions f64 (v, 3), triangles u32 (t, 3), normals f64 (v, 3) if present |
| image shape | 3 | rgba u8 (height, width, 4) |
| volume, field shape (reserved) | 4, 5 | payload u8 (len,) |
| paint table | 16 | rows f64 (n, 4) |

Keys find duplicates only: no drawer derives geometry from a key. A reader checks every uploaded
shape's and table's key against its content, except for shape kinds it does not know.

## Meaning

- **Shapes.** A path holds points in CE's cubic layout (anchor, handle, handle, anchor per curve);
  subpaths are split by CE's rule when drawn, never stored. A `points` record's positions are a path
  shape too. A mesh holds triangles with optional per-vertex normals. An image is straight RGBA.
- **Paint tables** hold RGBA rows (CE's floats: straight alpha, sRGB-encoded channels in 0..1): the
  stops of a paint of 3 or more stops, or one colour per face, point or instance.
- **Records** carry fixed fields, then a body by kind:
  - *geometry:* `shape_a` placed by `place_a` (3x4, row-major: `p' = M[:, :3] p + M[:, 3]`);
    optionally `shape_b` placed by `place_b` and a blend `t` (a straight-path blend by CE's
    `interpolate`); optionally a reveal window (lo, hi) with `pointwise_become_partial`'s meaning;
  - *paint:* fill, stroke and background stroke, each nothing, one or two inline RGBA stops, or a
    table key; stroke and background widths in CE's units; gradient endpoints; joint; cap; and a paint
    blend (`paint_b`, mixed in by `mix`);
  - *dash:* period, duty, phase (never set from CE, whose dashes are submobjects);
  - *depth:* space (screen, world, overlay), z-index, sort point, `shade_in_3d`, and a
    fixed-orientation centre;
  - *clip* (reserved, never set from CE), group *opacity* (1 from CE), a *layer*, and optional
    *fill_image* (the image key for feature bit 0);
  - the body: nothing for `path` and `mesh`; for `face_group`, per-face (u, v), the grid's size and
    per-face fill (and stroke) tables, the faces' stacked points being the record's path shape, and
    (XF 1.3) per face the end of its run of those points and its stroke width (without them: one
    equal run per face, each at the record's stroke width; a surface partly drawn holds faces of
    different lengths, and one being written strokes each face at its own width); for
    `points`, a per-point colour table and the radius (CE's stroke width); for `image`, either the
    picture placed at four corners with its resampling (PIL's value) and opacity, or a sub-frame
    (a capture rank, the same corners, resampling and opacity, then a view and an order over the same
    slots, for a MultiCamera display). Corners are CE's image points in CE's order: upper left, upper
    right, lower left, lower right. CE projects all four independently; after transforms the fourth
    is not exactly `c1 + c2 - c0`. A picture uses `shape_a`; a sub-frame has no `shape_a` and is refused
    if one is set. Neither image form is drawn by this build yet. For `instances`, per-instance
    placements, fill and stroke tables and object refs, the record's shape and paint block being the
    prototype every instance shares; for reserved kinds, opaque bytes.
- **Pre-projected output.** The tail's path shape is the camera's
  `transform_points_pre_display` of the record's points (an image's four corners); its point count
  need not equal the original shape's. Its optional gradient holds the camera's pre-display of
  the gradient ends, and its fill, stroke and background stroke are the colours the camera drew
  with. A drawer places this output without projecting or shading it again. The third column of
  `points` is the camera's own z: under a 3D camera's projection, the rotated depth **before the
  focal factor**. Drawers with depth key from it as they key a projected world point.
- **The view** holds the pixel size, the frame's width, height and centre (the camera's own values:
  CE fits the frame's height to the pixels' aspect ratio), the background (RGBA with the background
  opacity as alpha, or an image key), the camera (flat, moving, or 3D with phi, theta, gamma, focal
  distance, zoom, the exponential option, the light source, shading factor and whether shading
  applies), `mapping_centre`, CE's `cairo_line_width_multiple`, `use_z_index`, and the light model:
  CE's (code 0, no parameters) or soft diffuse (code 1, parameters `[ambient, wrap]`: the share of light a surface
  facing away from the light keeps, in 0..=1, and how far the light wraps past the terminator, at
  least 0; a placeholder whose parameters are set with the 3D look). A delta whose view gives soft
  diffuse light parameters outside those ranges is refused. `mapping_centre` is the centre CE's
  cached Cairo context froze into its frame-to-pixel matrix for the pixel array this view draws
  into. `Camera.get_cairo_context` bakes `frame_center` into that matrix the first time it sees a
  pixel array and caches the context, while a 3D camera's projection subtracts the live
  `frame_center`: paths are mapped to pixels around `mapping_centre`, projected around
  `frame_center`. This applies to sub-frame views too. A drawer maps as CE does: a
  record drawn as a path (`path`, a face group's faces, `mesh`, `instances`) around
  `mapping_centre`, a `points` or `image` record around `frame_center`. `Stream.screen_boxes`
  reports, per slot in the order, the pixel box of a record's points placed and mapped this way,
  without drawing.
- **Sub-frame capture.** A display's `rank` is its place in its camera's capture sequence, CE's
  `image_mobjects_from_cameras` order, starting at 0. CE captures displays one after another at the
  start of each capture call. The sub-frame order lists the slots that display's camera draws in
  that capture, its own slot included when CE lets a camera capture its own display. A display
  drawn inside a sub-frame shows that display's picture as it stands at that point, not a recursively
  redrawn view. The sub-frames of the displays in one layer of the order are drawn, in rank order,
  before that layer's entries. Two display records drawn in the same layer of one view's order may
  not share a rank; apply refuses that as invalid XF (`XfError`). Ranks need not follow draw order;
  the same rank in different layers or views is allowed.
- **The order** lists (layer, slot, part) in CE's draw order. The layer is static, moving or overlay;
  the part addresses a face of a face group, else 0.
- **A delta** applies, in this order: the view, objects, shape and table uploads, records set,
  records removed, keys released, the order. A slot set more than once takes the last record set
  in it, and a slot set and removed in one delta is removed. A key holds one content: a key the
  stream holds keeps it, and uploading it again is no change. A delta is checked whole before
  anything changes, against what it would leave: a reference to a missing shape, table or slot, a
  release of a key something still uses, an order entry naming a part that does not exist, a key
  uploaded twice in the delta, an upload under a key the stream holds with other content (a shape
  kind this build does not know keeps the key it was read with, unchecked), an image whose bytes are
  not its width times its height times 4, or a mesh whose triangles name a vertex it lacks (or whose
  normals are not one per vertex) refuses it.
- **Frames** (`Stream.draw`) have the layout of CE's `camera.pixel_array`: `height * width * 4` bytes
  in R, G, B, A order, paint premultiplied as Cairo stores it, the background as CE writes it
  (`(rgba * 255).astype(int)`).

## Drawing

- **Placement.** A record's points go through its placement, then as CE places a mobject's points for
  display: under a 3D camera, a `world` record is projected (moved by its centre's projection when it
  has a fixed orientation) and an `overlay` record is kept in frame; under any other camera, and for
  `screen` records, the points are kept. Then CE's frame-to-pixel mapping, around the view's
  `mapping_centre` for what CE draws through its Cairo context and its `frame_center` for point
  clouds and images.
- **Depth.** A `world` record under a 3D camera has depth: the projection's factor, negated (smaller is
  nearer), which is affine across a projected triangle; a fixed-orientation record lies on the
  camera-facing plane through its centre. Every other record is in front of all depth, and records
  tie by draw order, the later on top. Under a flat or moving camera everything ties, so records
  paint in order. Flat things share planes exactly: a path's plane (Newell's normal over its placed
  points, or where that cancels, as on an outline crossing itself, the normal of the widest
  triangle on its first point), a mesh triangle's and a fixed orientation's are made canonical (unit
  normal, largest component positive, offset from it, both rounded to multiples of 2^-20; points on
  a line take the axis plane of a coordinate they share), so text, axes, labels and surfaces on one
  plane tie by draw order instead of fighting. Whether a path is flat is its record's `planes` when
  the record carries them (XF 1.2: decided before its points were snapped), else found from its
  points; a mesh triangle is found from its corners, within the grid's noise when the view states
  an `input_grid` (a triangle the snap leaves under about a grid step wide takes its depth from its
  own corners). A path that is not
  flat, or whose plane is within about 3 degrees of edge-on, takes the depth of its own curve: each
  pixel the front-most part of the curve that reaches it (half the widest stroke and a pixel), else
  the nearest.
- **Camera-facing labels (XF 1.5).** The records sharing a `facing_group` are one label, placed
  whole and never split by depth; every other record is read from its `source` as a line, a
  surface or `label_rule_open` (`label_class.rs`), and only a surface can go in front of a label. A
  label's depth is its centre's: the camera-facing plane through it, read at the centre's
  projection, with its error bound. A sheet is a connected part of one surface object (the
  outermost surface or cylinder on a face's first path) whose fills are wound one way on screen,
  two fills sharing two corners on screen joined; strokes are never part of a sheet. A sheet goes
  in front of a label when it is shown nearer: where one of its fills holds the label's projected
  centre (a point on an edge held), on the ray through it, by that fill's plane read there; else at
  every order point of their overlap, by the plane of the fill holding the point read there; each
  depth with its error bound. The order points are the 16 sub-sample points of each pixel, where
  the resolve orders fragments: pixel `(x, y)`'s point `s` is
  `(x + 1/2 + (dx + 1/2)/16, y + 1/2 + (dy + 1/2)/16)` for `[dx, dy]` the `s`-th of
  `SAMPLE_OFFSETS`: `[-8, -7]`, `[-1, -8]`, `[3, -6]`, `[5, -5]`, `[-6, -1]`, `[-4, -4]`, `[0, -3]`,
  `[6, -2]`, `[-7, 2]`, `[-2, 0]`, `[2, 3]`, `[4, 1]`, `[-5, 5]`, `[-3, 4]`, `[1, 7]`, `[7, 6]`. The
  overlap is the order points inside both a record of the label (a fill or a stroke's outline) and
  a fill of the sheet in a colour not wholly transparent, each by the nonzero rule: a sliver of
  either between points holds none, and a sheet holding no point of the label's does not overlap
  it. A sheet shown farther or at the label's depth there leaves the label in front; one the bounds
  cannot place, one two of whose fills hold one point of the overlap (folded), one sharing its
  object and its winding with another sheet that meets the label, one drawn as a mesh, and every
  sheet against a label not drawn wholly as engine-placed paths is open, and leaves the label in
  front too. At each sub-sample the label is inserted immediately behind the rearmost sheet it goes
  behind that covers the sub-sample, in the order everything else takes there, or, with none, in
  front of everything but the overlay; labels at one place keep draw order. A pixel where a label
  meets a sheet it goes behind takes the sample resolve. A frame with no facing group draws as it
  would without its `source`.
- **Paths.** CE's subpaths (split where a cubic's end and the next one's start differ in x or y by
  more than CE's tolerance; closed when the first and last points are equal within it, the last then
  taken as the first), painted in CE's order as three layers tying in that order: the background
  stroke, the fill (nonzero), the stroke. Strokes are expanded in the frame's units, as Cairo strokes
  in user space: CE's width times the view's `cairo_line_width_multiple`, the joint (auto is a miter)
  and cap (auto is butt), Cairo's miter limit of 10, and the dash (a period in the frame's units,
  `duty` of it drawn, the pattern started `phase` periods in on each subpath, laid along the whole
  outline even where parts far outside the frame are simplified; a duty of 1 is solid). Each layer is
  flattened on the pixel plane to Cairo's tolerance of 0.1 pixel and covers each pixel by its exact
  area. Glyphs are paths. A pixel-aligned rectangle covers whole pixels exactly; Cairo's own box
  path differs from it by at most one step per byte where an edge falls inside a pixel.
- **Points and instances.** A `points` record draws as CE's point cloud: point `i` is a square block
  of `t = int(1 + (radius − 1))` pixels a side (none when t < 1) about the pixel its subpixel place
  truncates to (toward zero), offsets `range(−t//2 + 1, t//2 + 1)` on each axis, clipped to the
  frame pixel by pixel; its colour is colour row `i`, alpha times the group opacity, each channel
  `trunc(255·c)`; each block pixel covers its pixel exactly, at the depth of the plane facing the
  camera through the point (so a point ties with a camera-facing surface or a fixed-orientation
  record through it), and where blocks overlap at one depth the later write in CE's order wins
  (offset-major: every point at the first offset, then every point at the next). A translucent
  point composites where CE overwrites. An
  `instances` record's member `i` is its prototype placed by
  `placements[i]` after the record's own placement, its fill and stroke row `i` of the record's
  tables where the prototype fills or strokes; members tie in order, each above the one before, and
  draw the bytes the same members drawn as consecutive records draw.
- **Light.** A `shade_in_3d` record under a 3D camera that applies shading is lit at each triangle
  corner by the view's light model, at the corner's placed position and normal (the mesh's normals
  through the placement, else the triangle's own): CE's model adds CE's light to the encoded colour
  (each corner clamped as Cairo clamps a gradient stop); soft diffuse lights the linear colour, the
  normal turned toward the eye. Corner colours interpolate across the triangle on screen and are
  evaluated at the centroid of the sub-samples each pixel's fragment covers; lit linear colour is
  encoded there. Alpha is the fill's times the group opacity.
- **Compositing.** Each pixel holds fragments: a depth plane, a premultiplied colour, a coverage (a
  mask over 16 sub-samples for meshes; an exact area for paths) and the draw order. The 16
  sub-samples are a fixed stratified pattern (`SAMPLE_OFFSETS`): one in each cell of the pixel's
  4 x 4 grid and one in each of its 16 rows and columns. Fragments resolve by depth at the
  sub-samples, ties by draw order: a pixel with one depth order at every sub-sample and at most one
  partially covering mask, and no partial mask beside a path's partial area, composites back to
  front with its coverage as Cairo's ARGB32 surface composites paint (the coverage resolve); a
  crossing inside the pixel, two or more partial masks, or a partial mask (a surface's silhouette)
  with a path's partial area composites each sub-sample and averages them (the sample resolve), a
  mask covering a sub-sample or not and a path compositing with its exact area inside the
  sub-sample's cell, so a path's edge lands where the coverage resolve puts it, up to rounding.
  The silhouette rule keeps a pixel's resolve from flipping as crossings come and go, and needs a
  mesh in the pixel, so 2D drawing never takes the sample resolve. An opaque fragment covering
  the pixel in front of everything takes a fast path. One mesh's triangles meeting in a pixel
  without overlapping (a surface's internal edges) count as one fragment, its colour their average
  over the sub-samples each covers, so internal edges take the coverage resolve and never show what
  lies behind; for meshes alone, and a path over a surface tiling the pixel in one colour, that is
  the sample resolve's result up to rounding. Triangles of several colours (a lit surface) count as
  a partial mask where a partial mask or a path's partial area shares the pixel, since the average
  would lose which colour that layer covers. The coverage resolve lands within 2 of the sample
  resolve per byte where at most one partial path layer is in the pixel; two or more blend as
  independent coverages there (as Cairo blends them) and as their overlap in the sample resolve.
  `Stream.draw_resolves` reports each pixel's resolve. A GPU
  backend computes masks in a compute stage at this pattern's positions; hardware multisampling's
  positions are out, since WebGPU guarantees only 1 and 4 samples per pixel.

## Binary form

Little-endian throughout. `f64` is IEEE 754 binary64; `bool` is a u8, 0 or 1; `key` is 16 bytes;
`str` is a u32 byte count and UTF-8; `affine` is 12 f64, row by row; `count` is a u32; `item` is a
u32 byte length followed by that many bytes; `[x]?` is a bool, then x if it is 1.

```
file      "ARXF"  u16 major  u16 minor  count  item(delta)*
delta     u16 major  u16 minor  i64 t_num  i64 t_den (> 0)  u64 seq
          [item(view)]?
          count item(object)*          object: u32 ref  str stable_id  str source ("file:line")
          count item(shape)*
          count item(paint table)*     paint table: key  count  (f64 x4)*
          count key*                   (released)
          count item(record)*
          count u32*                   (removed slots)
          [count (u8 layer  u32 slot  u32 part)*]?     (the order; layer 0 static, 1 moving, 2 overlay)
          source                       (XF 1.5, when a record the delta sets carries a source or a
                                        facing group, or a held record is given a new one; anything
                                        after it belongs to a newer minor)
source    count str*                   (class names: each once, in the order the rows first use them)
          count row*                   (one per record carrying either, in the order set; of a slot
                                        set more than once, its last record's alone; then one per
                                        held record given a new one, for a slot the delta neither
                                        sets nor removes)
row       u32 slot  u32 facing_group (0: none)  count path*
path      count (u32 token  u32 class)*   (the record's owner first, then the objects containing it;
                                        class indexes the section's class names)
shape     u8 kind  key  body
          path: count (f64 x3)*
          mesh: count (f64 x3)*  count (u32 x3)*  bool has_normals  [(f64 x3) per vertex]
          image: u32 width  u32 height  u8 * (width * height * 4)
          other kinds: the rest of the item
record    u8 kind  u64 features  u32 slot  u32 object
          (a kind the reader does not know: the rest of the item is kept unread)
          u8 geometry flags (1 shape_a, 2 blend, 4 reveal)  [key shape_a]  affine place_a
            [key shape_b  affine place_b  f64 t]  [f64 lo  f64 hi]
          paint fill  paint stroke  paint background  f64 stroke_width  f64 background_width
            [f64 x6 gradient]?  u8 joint  u8 cap
            [paint fill  paint stroke  paint background  f64 stroke_width  f64 background_width]?
            f64 mix
          [f64 period  f64 duty  f64 phase]?           (dash)
          u8 space  f64 z_index  f64 x3 sort_point  bool shade_in_3d  [f64 x3]?   (fixed-orientation centre)
          [key  affine]?                                (clip)
          f64 opacity  u16 layer  [key fill_image]?
          body by kind:
            path 1, mesh 6: nothing
            face_group 2: count (u32 u  u32 v)*  u32 grid_u  u32 grid_v  key fill  [key stroke]?
            points 3: key colors  f64 radius
            image 4: u8 0  f64 x12 corners  u8 resampling  f64 opacity
                   | u8 1  u32 rank  f64 x12 corners  u8 resampling  f64 opacity
                     item(view)  count (u8 layer  u32 slot  u32 part)*
            instances 5: count affine*  key fill  key stroke  u32 object per instance
            volume 64, field 65, emission 66 (reserved): item(payload)
          tail (bit 1 set; XF 1.1), after the body:
            key points
            [f64 x6 gradient]?
            paint fill  paint stroke  paint background
          planes and normals (XF 1.2), after the body and tail, both when the record carries
          either (an empty list: not carried):
            count u8*   (one planarity code per path the record draws)
            count (f64 x3)*   (two normals per path the record draws)
          face ends and face widths (XF 1.3, face_group only), after the planes and normals
          (written, as empty lists, when the record carries neither), both when the record
          carries either or joins follow (an empty list: not carried):
            count u32*   (per face, one past its last point: rising, the last the points' count)
            count f64*   (per face, its stroke width: finite, at least 0)
          joins (XF 1.4), after the planes and normals and a face group's face lists (written,
          empty, when the record carries none of them), when the record carries them:
            count u8*   (one per cubic of the record's points: bit 0 joined, bit 1 mitred)
          (anything after the known fields belongs to a newer minor and is skipped)
paint     u8 0 (none) | u8 1  f64 x4 | u8 2  f64 x8 | u8 3  key
view      u32 pixel_width  u32 pixel_height  f64 frame_width  f64 frame_height
          f64 x3 centre  f64 x3 mapping_centre
          u8 0  f64 x4 rgba | u8 1  key image          (background)
          u8 0 flat | u8 1 moving | u8 2  f64 phi theta gamma focal_distance zoom
            bool exponential  f64 x3 light_source  f64 shading_factor  bool should_apply_shading
          f64 cairo_line_width_multiple  bool use_z_index
          u8 light model (0: CE's, 1: soft diffuse)  count f64*   (its parameters)
          [f64 input_grid]   (XF 1.2, when the geometry was snapped: nonzero)
```

Joint codes: 0 auto, 1 round, 2 bevel, 3 miter. Cap codes: 0 auto, 1 round, 2 butt, 3 square. Space
codes: 0 screen, 1 world, 2 overlay. XF streams are generated by code, never committed. Two kinds of
test hold the layout: `tests/layout.rs` states every field's byte offset and every enum code above as
literals and checks the writer and the reader against them, so a drift made the same way in both fails
it; and the canonical samples (`tests/common/mod.rs` in Rust, `canonical_sample()` in the binding's
tests), which use every enum code and give swappable fields values of their own, pin the sha256 of their
bytes, so any change to what the samples write fails the pin.

## Python construction

`FrameDelta.set_view(..., mapping_centre=...)` defaults `mapping_centre` to the given
`frame_center` (itself `[0, 0, 0]` by default), also for a sub-frame's view dictionary.
`set_record("image", ..., corners=...)` takes a `(4, 3)` array-like in CE's corner order.
For a display, use `set_record("image", ..., corners=..., resampling=..., image_opacity=...,
sub_frame=(rank, view_dict, order))`, with no `shape`; resampling defaults to 3 and image opacity
to 1 for both forms.

`set_record(..., fill_image=key)` sets the optional field and bit 0 together, preserving other
`features` bits. Passing bit 0 in `features` without `fill_image` raises `XfError`; Python cannot
construct a disagreement. Likewise, `set_record(..., pre_projected={"points": key_bytes,
"gradient": None, "fill": paint, "stroke": paint, "background_stroke": paint})` sets bit 1,
preserving the other feature bits. `gradient` accepts `None`, six floats or two 3-tuples; omitted
paint names mean none, and each paint takes the fixed paints' forms (RGBA, one or two inline
stops, or a table key). `points` is required. Passing bit 1 without `pre_projected` raises
`XfError`, as does using the field on a disallowed kind. The package exports
`FEATURES = {"background_image": 1, "pre_projected": 2}`.

XF 1.2: `set_view(..., input_grid=g)` states the grid the geometry was snapped to (omitted or 0:
not snapped, and the field is not written); `set_record(..., planes=[codes])` carries the record's
planarity codes, and `normals=[(x, y, z), ...]` its lit corners' normals. `Stream.planes(slots)`
returns, for each slot holding a record that draws paths, the list of codes the drawer decides from
the points the stream holds; `Stream.normals(slots, grid)`, for each slot holding a record the
camera shades, its normals decided for points snapped to `grid`.

XF 1.3: `set_record("face_group", ..., face_ends=[...], face_widths=[...])` carries each face's end
in the points and its stroke width (omitted: not written, the faces equal runs at the record's
width). `apply` refuses face ends not one per face, not rising, or not ending at the points' count,
and face widths not one per face or not finite and at least 0.

XF 1.5: `set_record(..., source=[[(token, class), ...], ...], facing_group=g)` carries the
record's source paths, each `(token, class)` pairs owner first, and its facing group (omitted,
empty or 0: not carried). A path of no objects raises `XfError`; tokens and the group are u32.
`set_source(slot, source=..., facing_group=g)` gives the record held in `slot` a new source and
facing group, its drawing unchanged; a row carrying neither raises `XfError`.

## Python read-back

`FrameDelta.from_bytes(raw).contents()` returns the decoded delta held by the core, not a
reconstruction from drawing state. Its keys are `version`, `t` (both pairs), `seq`, `view` (a dict
or `None`), `objects` (a list of `(ref, stable_id, source)` tuples), `upload`, `upload_paints`,
`release` (key bytes), `set`, `remove` (slots), `order` (a list or `None`), and `source` (a list
of `(slot, source, facing_group)` rows: the held records given a new source, `source` in a record's
form).

- A view dictionary uses `set_view`'s keyword names. `camera` is `flat`, `moving` or `three_d`, with
  the 3D parameters present only for `three_d`; `background` is an RGBA 4-tuple or image key bytes;
  `light_model` is always `(code, [parameters])`. Both centres are 3-tuples. `input_grid` is present
  only when the view carries one.
- Each upload is `(kind_name, key_bytes, data)`: a path's data is float64 `(n, 3)`; a mesh's is a
  dictionary with `positions` (float64 `(v, 3)`), `triangles` (uint32 `(t, 3)`) and `normals`
  (float64 `(v, 3)` or `None`); an image's is uint8 `(height, width, 4)`; a reserved or unknown
  shape's is opaque bytes (`volume`, `field`, or `kind<code>`). Each paint upload is
  `(key_bytes, float64 (n, 4) array)`. Keys are always 16-byte `bytes` values.
- Each record is `(kind_name, slot, object, fields)`. For a known kind, `fields` uses
  `set_record`'s keyword names, including the body's arguments. `features` always contains the
  full bitset; when `fill_image` is present its bit is already set, and `set_record` accepts it
  unchanged. Absent optional fields are omitted (`planes`, when present, is a tuple of codes, `normals` a
  tuple of `(x, y, z)` tuples, `source` a tuple of paths, each a tuple of `(token, class)` tuples,
  and `facing_group` an int, on every kind); paints are `None`, float64 `(1, 4)` or `(2, 4)`
  arrays, or table key bytes. `paint_b`, when present, is a dictionary of its paint and width
  arguments. `joint`, `cap` and `space` are names. `place` and clip/blend placements are float64
  `(3, 4)` arrays, `placements` is float64 `(n, 3, 4)`, `corners` is float64 `(4, 3)`, `uv` is
  uint32 `(n, 2)`, and `payload` is bytes. `gradient` is a flat 6-tuple; `set_record` accepts it or
  the two 3-tuples accepted by construction. A sub-frame is `(rank, view_dict, order)`, with its
  corners, resampling and image opacity alongside it in `fields`. An unknown record kind exposes
  only `features` and the unread `payload` after its slot/object header, not invented fixed fields.
- A record with `pre_projected` reads it back as a dictionary with `points` (16-byte bytes),
  `gradient` (a flat 6-tuple or `None`), and `fill`, `stroke`, `background_stroke` (each `None`,
  a float64 `(1, 4)` or `(2, 4)` array, or table key bytes). All five names are present in the
  dictionary, including absent gradient and paints as `None`. Rebuilding this dictionary
  preserves every f64 bit, just as the fixed fields do.
- An order entry is `(layer_name, slot, part)`, with `static`, `moving` or `overlay` as the name.

All arrays are independent NumPy copies, including empty arrays with their full dimensions.
Read-back performs no floating-point arithmetic: every f64 bit, including signed zero, subnormals
and NaN payloads, survives rebuilding a current-version delta made through Python's construction
methods byte for byte.
Opaque uploads and unknown record kinds are readable but have no Python construction method.
