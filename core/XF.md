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
frame as CE crops it) and refuses every record kind until its drawer exists.

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
  applies), CE's `cairo_line_width_multiple`, `use_z_index`, and the light model (CE's, code 0).
- **The order** lists (layer, slot, part) in CE's draw order. The layer is static, moving or overlay;
  the part addresses a face of a face group, else 0.
- **A delta** applies, in this order: the view, objects, shape and table uploads, records set,
  records removed, keys released, the order. A delta is checked whole before anything changes: a
  reference to a missing shape, table or slot, a release of a key something still uses, or an order
  entry naming a part that does not exist refuses it.
- **Frames** (`Stream.draw`) have the layout of CE's `camera.pixel_array`: `height * width * 4` bytes
  in R, G, B, A order, paint premultiplied as Cairo stores it, the background as CE writes it
  (`(rgba * 255).astype(int)`).

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
          u8 light model (0: CE's)  count f64*          (its parameters)
```

Joint codes: 0 auto, 1 round, 2 bevel, 3 miter. Cap codes: 0 auto, 1 round, 2 butt, 3 square. Space
codes: 0 screen, 1 world, 2 overlay. XF streams are generated by code, never committed. Two kinds of
test hold the layout: `tests/layout.rs` states every field's byte offset and every enum code above as
literals and checks the writer and the reader against them, so a drift made the same way in both fails
it; and the canonical samples (`tests/common/mod.rs` in Rust, `canonical_sample()` in the binding's
tests), which use every enum code and give swappable fields values of their own, pin the sha256 of their
bytes, so any change to what the samples write fails the pin.
