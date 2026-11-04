# XF, the Frame IR

XF describes frames for a drawer. A **stream** holds what a drawer keeps across frames: shapes,
paint tables, objects, records, the view and the draw order. Each drawn frame is a **delta** against
the stream, empty for a held frame. A **record** describes one drawn object; a **shape** is geometry
uploaded once under its content key. The Rust types are in `src/xf/mod.rs`; this file describes their
meaning and the binary form (`src/xf/wire.rs`), which fixtures and dumped streams use.

## Versions and refusal

The version is (major, minor). This build is **XF 0.0**: before XF 1 the layout may still change.
From XF 1 on it is append-only: a new field or kind bumps the minor, and nothing removes or retypes a
field.

- A reader refuses a major it does not know, and reads a newer minor of its own major, skipping what
  that minor appended (every record, shape, paint table, object and view is length-prefixed).
- A record kind the reader does not know is kept, with its slot and object, and its body unread.
- A drawer refuses a whole frame (`render.unsupported`) when a record in the draw order has a kind or
  a feature bit it does not draw, or the view names a light model it does not know. It never draws a
  frame partly. A record outside the order is not drawn, so it is not refused.

XF 0 defines no feature bits. This build draws the background (a colour, or an image cropped to the
frame as CE crops it), `mesh` records filled with one colour, `path` records whose fill and strokes
are each one colour, `points` records, and `instances` records of a path or a mesh. It refuses any of
those carrying a blend, a reveal, a paint blend or a clip; a path with a gradient paint (two stops or a
table) or a dash it cannot read; a mesh with a stroke, a gradient fill or a dash; and the
`face_group` and `image` kinds and the reserved ones, until their drawers exist. How records are drawn is described under
"Drawing" below.

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
  - *clip* (reserved, never set from CE), group *opacity* (1 from CE), and a *layer*;
  - the body: nothing for `path` and `mesh`; for `face_group`, per-face (u, v), the grid's size and
    per-face fill (and stroke) tables, the faces' stacked points being the record's path shape; for
    `points`, a per-point colour table and the radius (CE's stroke width); for `image`, either the
    picture placed at three corners with its resampling (PIL's value) and opacity, or a sub-frame (a
    view and an order over the same slots, for a MultiCamera display); for `instances`, per-instance
    placements, fill and stroke tables and object refs, the record's shape and paint block being the
    prototype every instance shares; for reserved kinds, opaque bytes.
- **The view** holds the pixel size, the frame's width, height and centre (the camera's own values:
  CE fits the frame's height to the pixels' aspect ratio), the background (RGBA with the background
  opacity as alpha, or an image key), the camera (flat, moving, or 3D with phi, theta, gamma, focal
  distance, zoom, the exponential option, the light source, shading factor and whether shading
  applies), CE's `cairo_line_width_multiple`, `use_z_index`, and the light model: CE's (code 0, no
  parameters) or soft diffuse (code 1, parameters `[ambient, wrap]`: the share of light a surface
  facing away from the light keeps, in 0..=1, and how far the light wraps past the terminator, at
  least 0; a placeholder whose parameters are set with the 3D look). A delta whose view gives soft
  diffuse light parameters outside those ranges is refused.
- **The order** lists (layer, slot, part) in CE's draw order. The layer is static, moving or overlay;
  the part addresses a face of a face group, else 0.
- **A delta** applies, in this order: the view, objects, shape and table uploads, records set,
  records removed, keys released, the order. A delta is checked whole before anything changes: a
  reference to a missing shape, table or slot, a release of a key something still uses, an order
  entry naming a part that does not exist, or a mesh whose triangles name a vertex it lacks (or
  whose normals are not one per vertex) refuses it.
- **Frames** (`Stream.draw`) have the layout of CE's `camera.pixel_array`: `height * width * 4` bytes
  in R, G, B, A order, paint premultiplied as Cairo stores it, the background as CE writes it
  (`(rgba * 255).astype(int)`).

## Drawing

- **Placement.** A record's points go through its placement, then as CE places a mobject's points for
  display: under a 3D camera, a `world` record is projected (moved by its centre's projection when it
  has a fixed orientation) and an `overlay` record is kept in frame; under any other camera, and for
  `screen` records, the points are kept. Then CE's frame-to-pixel mapping.
- **Depth.** A `world` record under a 3D camera has depth: the projection's factor, negated (smaller is
  nearer), which is affine across a projected triangle; a fixed-orientation record is a camera-facing
  plane at its centre's depth. Every other record is in front of all depth, and records tie by draw
  order, the later on top. Under a flat or moving camera everything ties, so records paint in order.
  Flat things share planes exactly: a path's plane (Newell's normal over its placed points) and a
  mesh triangle's are made canonical (unit normal, largest component positive, offset from it, both
  rounded to multiples of 2^-20; points on a line take the axis plane of a coordinate they share), so
  text, axes and surfaces on one plane tie by draw order instead of fighting. A path that is not
  flat, or whose plane is within about 3 degrees of edge-on, takes the depth of its own curve: each
  pixel the front-most part of the curve that reaches it (half the widest stroke and a pixel), else
  the nearest.
- **Paths.** CE's subpaths (split where a cubic's end and the next one's start differ in x or y by
  more than CE's tolerance; closed when the first and last points are equal within it, the last then
  taken as the first), painted in CE's order as three layers tying in that order: the background
  stroke, the fill (nonzero), the stroke. Strokes are expanded in the frame's units, as Cairo strokes
  in user space: CE's width times the view's `cairo_line_width_multiple`, the joint (auto is a miter)
  and cap (auto is butt), Cairo's miter limit of 10, and the dash (a period in the frame's units,
  `duty` of it drawn, the pattern started `phase` periods in; a duty of 1 is solid). Each layer is
  flattened on the pixel plane to Cairo's tolerance of 0.1 pixel and covers each pixel by its exact
  area. Glyphs are paths. A pixel-aligned rectangle covers whole pixels exactly; Cairo's own box
  path differs from it by at most one step per byte where an edge falls inside a pixel.
- **Points and instances.** A `points` record's point `i` is a disc `int(radius)` pixels across (CE
  draws a point as a block that wide; under 1 nothing), centred where CE places it, in colour row
  `i`, at the point's depth. An `instances` record's member `i` is its prototype placed by
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
  mask over 16 sub-samples for meshes; an exact area for paths, with the mask on demand) and the
  draw order. Fragments resolve by depth at the sub-samples, ties by draw order: a pixel with one
  depth order at every sub-sample and at most one partially covering mask composites back to front
  with its coverage as Cairo's ARGB32 surface composites paint (the coverage resolve); a crossing
  inside the pixel, or two or more partial masks, composites each sub-sample and averages them (the
  sample resolve); an opaque fragment covering the pixel in front of everything takes a fast path.
  One mesh's triangles meeting in a pixel without overlapping (a surface's internal edges) count as
  one fragment, its colour their average over the sub-samples each covers, so internal edges take
  the coverage resolve and never show what lies behind. `Stream.draw_resolves` reports each pixel's
  resolve.

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
          f64 opacity  u16 layer
          body by kind:
            path 1, mesh 6: nothing
            face_group 2: count (u32 u  u32 v)*  u32 grid_u  u32 grid_v  key fill  [key stroke]?
            points 3: key colors  f64 radius
            image 4: u8 0  f64 x9 corners  u8 resampling  f64 opacity
                   | u8 1  item(view)  count (u8 layer  u32 slot  u32 part)*
            instances 5: count affine*  key fill  key stroke  u32 object per instance
            volume 64, field 65, emission 66 (reserved): item(payload)
          (anything after the body belongs to a newer minor and is skipped)
paint     u8 0 (none) | u8 1  f64 x4 | u8 2  f64 x8 | u8 3  key
view      u32 pixel_width  u32 pixel_height  f64 frame_width  f64 frame_height  f64 x3 centre
          u8 0  f64 x4 rgba | u8 1  key image          (background)
          u8 0 flat | u8 1 moving | u8 2  f64 phi theta gamma focal_distance zoom
            bool exponential  f64 x3 light_source  f64 shading_factor  bool should_apply_shading
          f64 cairo_line_width_multiple  bool use_z_index
          u8 light model (0: CE's, 1: soft diffuse)  count f64*   (its parameters)
```

Joint codes: 0 auto, 1 round, 2 bevel, 3 miter. Cap codes: 0 auto, 1 round, 2 butt, 3 square. Space
codes: 0 screen, 1 world, 2 overlay. XF streams are generated by code, never committed. Two kinds of
test hold the layout: `tests/layout.rs` states every field's byte offset and every enum code above as
literals and checks the writer and the reader against them, so a drift made the same way in both fails
it; and the canonical samples (`tests/common/mod.rs` in Rust, `canonical_sample()` in the binding's
tests), which use every enum code and give swappable fields values of their own, pin the sha256 of their
bytes, so any change to what the samples write fails the pin.
