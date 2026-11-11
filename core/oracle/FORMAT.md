# The oracle's decisions file: `ORACLE_FORMAT` 1

`arachne-oracle decide --xf <file> --seq <delta seq> --out <file> [--pixels x0,y0,x1,y1]` applies the
XF file's deltas in order, from an empty stream, up to and including the delta whose `seq` is given,
and writes what the frame drawn there should hold: every fragment the draw order draws and its
coverage verdict at each of the 16 samples of every pixel of the window (the whole frame by
default), with each present fragment's depth band and each present pair's admitted edges and
label at every sample, or why the frame is `truth_unable`. It carries decisions, never pixel bytes.

`read.py` beside this file is the one reader for other programs (standard library only); its
`DTYPES` give each section's layout as a numpy dtype, so a section also reads whole with
`numpy.fromfile(path, numpy.dtype(DTYPES[tag]), count, offset=offset)`.

The exit status is 0 when the file is written, a refused frame included, and 2 for a usage or
input error (an XF file that does not read, a `seq` it does not hold).

## Layout

Little-endian throughout. A fixed header of 88 bytes, a directory of sections, then each section
on an 8-byte boundary. Sections are arrays of fixed-size records; a reader skips a tag it does not
know, and treats a section that is absent as empty. Later sections arrive as new tags.

```
header     4s  magic "AROR"
           u16 format (1)          u16 header size (88)
           u16 XF major            u16 XF minor       (the decided delta's version)
           u32 width               u32 height         (the view's pixel size)
           u32 x0, y0, x1, y1                         (the window decided, [x0, x1) x [y0, y1))
           u8  status (0 decided, 1 truth_unable)     3 bytes reserved
           u64 seq
           i8  x 32                                   (the sample pattern: 16 offsets (dx, dy))
           u32 section count                          u32 reserved
directory  per section: 4s tag  u32 record size  u64 offset  u64 count
```

Sample `s` of pixel `(x, y)` lies at `(x + 1/2 + (dx + 1/2) / 16, y + 1/2 + (dy + 1/2) / 16)`, in
cell `((dx + 8) div 4, (dy + 8) div 4)` of the pixel's 4 x 4 grid.

| Tag | Record | What |
|---|---|---|
| `FRAG` | 64 bytes | one per fragment that has a row, or is unbounded, then one per zero-alpha mesh triangle |
| `COVR` | 456 bytes | one per fragment at a pixel, sorted by `(y, x, fragment)` |
| `PIXL` | 12 bytes | one per pixel of the window, row by row: its first `COVR` row and their count |
| `BAND` | 544 bytes | one per `COVR` row, in its order (its own row), then one more where the fragment has a row on its record's plane |
| `PAIR` | 24 bytes | one per pair of fragments present at a sample, sorted by `(y, x, sample, a, b)` |
| `COLR` | 12 bytes | one per fragment, by id: its colour's kind and a solid's bytes |
| `COLP` | 24 bytes | one per `COVR` row of a per-pixel fragment, in `COVR`'s order |
| `LABL` | 20 bytes | one per camera-facing label and other fragment present at one sample, sorted by `(y, x, sample, label, fragment)` |
| `REFU` | 40 bytes | one per refusal; empty when the frame is decided |
| `TEXT` | 1 byte | the refusals' words, UTF-8 |

### `FRAG`

```
u32 id  u32 entry (order index)  u32 slot  u32 object  u32 part  u32 sub
u8 layer  u8 kind (record kind code)  u16 flags  u32 group
f64 epsilon (screen shift, px)  i32 x0, y0, x1, y1 (its rows' pixel box)  u32 plane  u32 label
```

- **Identity.** `(object, part, layer, sub)` names a fragment as the core's report does. `layer`:
  0 background stroke, 1 fill, 2 stroke, 3 mesh triangle, 4 point block. `sub` is a path layer's
  `3 i + k` (`i` the instance member, else 0; `k` the layer), a mesh triangle's index in its
  shape (`t i` plus it for member `i` of a `t`-triangle prototype), a point block pixel's
  `(offset index) · n + i`. An instance's fragments carry the instance's object.
- **`group`**: the drawn path a layer belongs to (its three layers share it); `0xffffffff` for a
  mesh triangle or a point block. **`plane`**: the index, in the frame, of the registry-offered
  record the fragment belongs to (a path's layers share it), else `0xffffffff`. The oracle does
  not settle the registry's identities; which records may share one is in `PAIR`'s labels
  (contract §7).
- **`flags`**: bit 0, *unbounded*: the fragment's screen shift is infinite (or its nominal
  geometry is not finite), so it may stand anywhere: it is `truth_unable` at every sample of the
  window, never absent. Its `epsilon` is infinite, its box the window, and it has no `COVR` rows; a
  reader treats it as present-unknown at every pixel. An unbounded point has one fragment, the
  `sub` of its first block offset. Bit 1, *zero alpha*: a mesh triangle whose alpha byte is 0
  (contract §1, `alpha_round@1`). The contract makes it not present, so it has no `COVR` rows, no
  `BAND` rows and no pairs, and its colour is `(0, 0, 0, 0)`; its box is the pixels it would
  cover. It is written so that a report carrying that fragment is named: a fragment the oracle
  holds no `COVR` row for is a `mismatch`, and where its identity is a zero-alpha fragment's it is
  counted `zero_alpha_fragment`. These rows follow every other fragment's.
- **`label`**: the camera-facing label the fragment belongs to (contract §10,
  `label_placement@2`: a facing group read under XF 1.5's rules, else a run of camera-facing
  records with one carried centre), numbered in the frame from 0; `0xffffffff` for a fragment of
  no label. The order at a sample holding a label is read through `LABL`.

### `COVR`

```
u32 x  u32 y  u32 fragment
u8 cover[16]  u8 cell_lo[16]  u8 cell_hi[16]  u8 pixel_lo  u8 pixel_hi  2 bytes pad
f64 pixel_area  f64 cell_area[16]  f64 depth[16]  f64 band[16]
```

- **`cover[s]`** at sample `s`: 0 absent, 1 `Yes`, 2 `Either`.
- **`cell_lo[s]`, `cell_hi[s]`**: the cell bytes admitted at `s` (a mask: `255, 255` for `Yes`,
  `0, 255` for `Either`).
- **`pixel_lo`, `pixel_hi`**: the whole-pixel coverage bytes admitted.
- **`pixel_area`, `cell_area[s]`**: a path layer's exact nonzero-rule area of the pixel and of
  sample `s`'s cell, each as a fraction of it (1 covered whole); NaN for a mesh triangle or a point
  block, whose verdicts are masks.
- **`depth[s]`, `band[s]`**: the fragment's own key `o` and arithmetic bound `R` at sample `s`
  (contract §5, §6), as its own `BAND` row holds them; NaN where it is absent or its key is not
  bounded.
- A row is written where some sample is not absent or the admitted pixel bytes reach above 0.

### `BAND`

```
u32 x  u32 y  u32 fragment  u8 which  3 bytes pad
f64 o[16]  f64 r[16]  f64 s_minus[16]  f64 s_plus[16]  u8 status[16]
```

A fragment's band at each sample: `I = [o − S⁻ − 3R, o + S⁺ + 3R]` (contract §6), `I = {−∞}` for a
key of `−∞`.

- **`which`**: 0, its own row: its own key, a stroke's adjusted on its own fill where both cover
  (`stroke_on_own_fill@1`: keyed at the nearer key, its interval the hull of both keys'
  intervals, met or not, §6);
  1, its row on its record's plane (`shared_plane_read@1`), read against a fragment that may share
  it. A fragment has at most one of each at a pixel, its own first.
- **`status[s]`**: 0 absent, 1 bounded, 2 `band_unbounded`, 3 `curve_recut` (the sample is
  `truth_unable`, contract §6). The floats are NaN unless the status is 1.

### `PAIR`

```
u32 x  u32 y  u32 a  u32 b  u8 sample  u8 edges  u8 label  u8 provenance  u8 offered  u8 plate  2 bytes pad
```

Every pair of fragments present (`Yes` or `Either`) at a sample, `a < b` by fragment id (contract
§7, "the edges a pair admits").

- **`edges`**: bit 0 weak (paint order: the later in the order, then the higher `sub`, in front),
  bit 1 strong `a` in front, bit 2 strong `b` in front.
- **`label`**: 1 `Depth(a)`, 2 `Depth(b)`, 3 `TieBand`, 4 `ExactTie`, 5 `own_layers@1`,
  6 `one_plane@1`, 7 `lines@1`, 8 `near_parallel@1`; 0 where a read of either is not bounded
  (`truth_unable`, every edge).
- **`provenance`**: how each strong edge is read, a two-bit code for each: bits 0–1 the edge of
  `a` in front (`edges` bit 1), bits 2–3 that of `b` in front (`edges` bit 2). 0 *own keys*: every
  scenario that admits the edge reads both fragments at their own keys (each one number at the
  sample, `stroke_on_own_fill@1` included), or `edges` holds no such edge; 1 *row read*: some
  scenario admits it through the pair's rows on a plane they may share (`shared_plane_read@1`);
  2 *flat identity*: the pair may share a flat identity with a record of no plane of its own
  (`one_plane@1`, every edge). 3 is not written. The completions rule that discards a strong cycle
  of own-key edges reads it.
- **`offered`**: the pair's edges, in `edges`' bits, in the sets built with the records that paint
  nothing offered to the registry too (contract §5). Equal to `edges` where the frame has no such
  record. A reported order outside `edges`' admissible orders and inside `offered`'s is counted
  `invisible_registry_offer`.
- **`plate`**: where `a` or `b` belongs to a label (`FRAG` `label`), the pair's relation as the
  plate model orders it (§10, `facing_plane@1`), read in the scenarios `edges` reads from the two
  keys without their input bound (`S⁻ = S⁺ = 0`, §6) and with no snap share in a policy's term,
  since the core reads its keys from the inputs it holds. It has three states, and is one only
  where every scenario gives it:
  - **tie**, `1` (`edges`' weak bit): the keys are equal (`tied`, §7), a weak order, which is
    paint order;
  - **strong**, `2` (`a` in front) or `4` (`b` in front): the keys are separated beyond their
    bounds (`±3R`), and no tie policy's term holds;
  - **undetermined**, `0`: neither. A pair whose rows hang on what the registry settles (a record
    with no plane of its own that may share a flat identity) is undetermined.

  For every other pair `plate` equals `edges`. The plate model's orders at a sample are those the
  `plate` relations admit; where a label's pair there is undetermined, no order is the plate
  model's. A label 2^-30 behind a line at its depth is inside the snap, so `edges` admits either
  order, and `plate` is `2`, the label's in front. A line on the label's plane ties it (`plate`
  `1`), and the plate model draws the later painted in front; a line 2^-45 nearer is inside the
  keys' bounds, and `plate` is `0`.
- **Admissible orders** of the fragments present at a sample are the ranks (`cycle_rank@1`) of the
  completions that pick one edge per pair; an order every pair agrees with is one. At a sample
  that holds a label fragment, `PAIR` still holds every pair, the label's included, as §7 reads
  them; the label rule's order is read through `LABL`.
- An unbounded fragment (`FRAG` flag bit 0) has no rows and no pairs: it is `truth_unable`
  everywhere.

### `LABL`

```
u32 x  u32 y  u32 label  u32 fragment  u8 sample  u8 role  u8 pair_cause  u8 class_cause
```

A camera-facing label's placement facts at a sample (contract §10, `label_placement@2`): one row
for each label with a fragment present at the sample and each other fragment present there that
belongs to no label.

- **`role`**: 0 *front*: the label goes in front of the fragment unless the fragment is in front of
  an anchor; 1 *anchor*: the fragment is a fill of a sheet that qualifies for the label and holds
  the sample's order point; 2 *open*: the pair, or the fragment's class, is open
  (`label_rule_open`), the causes below; 3 *kind unknown* (`label_kind_unknown`, a stream read as
  XF 1.4): an anchor under one reading and not under the other; 4 *alpha straddle*
  (`label_alpha_straddle`): the fill's admitted alpha at the pixel holds 0 and a positive byte, and
  the oracle cannot place it. A gradient fill's admitted alpha (its colour is not computed, `COLR`
  kind 0) is bounded by its stops: their premultiplied alpha bytes at the record's group opacity,
  as a solid's (`cairo_premultiply@1`), from the least to the greatest, one level wider each way
  for the walker's single-precision interpolation; pad extend keeps every pixel within them.
- **`pair_cause`** (role 2): 1 `interval`, 2 `folded`, 3 `disconnected`, 4 `mesh`, 5 `label`; 0
  where the class is open instead. **`class_cause`** (role 2): 1 `unmapped`, 2 `generic_only`,
  3 `paths_disagree`, 4 `no_source`, 5 `cylinder_grouped`, 6 `point_cloud`; 0 where the pair is
  open instead.
- **The label rule's order** at the sample. Take the reported order of the fragments of no label;
  it must be admissible by `PAIR` alone. Each label is inserted immediately behind the rearmost, in
  that order, of its anchors (role 1), or in front of every fragment where it has none; labels at
  one place keep paint order, and so do a label's own fragments. A role-3 fragment is read both
  as an anchor and not, and the orders of both readings are admitted. A sample where some row of a
  label is role 2 is counted `label_rule_open` under its causes, and one with a role-4 row is
  `label_alpha_straddle`: neither is a mismatch nor an agreement.
- A reported order outside the label rule's orders that is one of the plate model's orders
  (admitted by `PAIR`'s `plate` relations, as `edges` admit the ordinary ones) is a `mismatch`
  counted `label_plate_model`; any other order outside them is a plain `mismatch`, every order
  at a sample where a label's `plate` is undetermined included. So the count of
  `label_plate_model` is exact where `plate` is determined. Past the completion bound the
  mismatch is certain and its name is not.

### `COLR`

```
u32 fragment  u8 kind  u8 rgba[4]  3 bytes pad
```

Each fragment's premultiplied colour from XF (contract §9). `kind`: 1 solid (`rgba` at every
pixel: `cairo_premultiply@1`, or a point block's bytes); 2 per pixel (a lit mesh triangle,
`ce_shading@1`; its bytes in `COLP`); 0 a gradient (`cairo_linear_gradient@1`, a layer of two or
more rows or a shaded path), which this oracle does not compute: `rgba` is 0 and a reader takes
the colour as unknown.

### `COLP`

```
u32 x  u32 y  u32 fragment  u8 lo[4]  u8 hi[4]  u32 reserved
```

A per-pixel fragment's bytes at a pixel: the corners' colours interpolated as planes on the pixel
plane, read at the centroid of the samples it covers (`lo = hi`), or, where one of its samples is
`Either`, the least and most bytes over every sample it may cover.

### `REFU`

```
u8 reason  u8 kind (0xff: none)  u8 cause (0: none)  1 byte pad
u32 entry  u32 slot  u32 object (0xffffffff: none)
u64 bits (unknown feature bits)  u64 at_seq (0xffffffffffffffff: none)
u32 text start  u32 text length (into TEXT)
```

Reasons: 1 `unknown_kind`, 2 `reserved_kind`, 3 `unknown_feature`, 4 `unknown_light_model`,
5 `xf_newer` (a delta of an XF minor above 5 at or before the frame), 6 `invalid_delta`, 7 `no_view`,
8 `core_refuses` (a refusal of the core's own list), 9 `stream_refused`.

- **`invalid_delta`**: the frame's delta is not valid XF and is not applied; the stream keeps what it
  held, and the next frame is judged as usual. It is per frame, on a stream not yet refused.
- **`stream_refused`** is `StreamRefused { at_seq, cause }`: the delta at `at_seq` was refused by one
  of XF 1.5's source-row codes, its `cause`: 1 `class_named_twice`, 2 `slot_named_twice`,
  3 `slot_removed`, 4 `row_empty`, 5 `path_empty`, 6 `class_out_of_range`, 7 `slot_not_held`.
  That delta's frame and every later frame of the stream are `truth_unable` under it, with no
  verdict. No later delta clears it: a refused stream does not resume. A later delta that would
  itself be refused adds nothing; every frame keeps the first refused delta's `at_seq` and `cause`.
- **One refusal per delta.** A delta's refusal is the first of all XF's checks in the order `XF.md`
  ("Where each check is made") gives them. One of the seven codes above makes it
  `stream_refused`; any other check makes it `invalid_delta`.

## A worked example

A 16 x 16 flat view whose frame units are pixels, and one path record filling the rectangle from
`(2, 2)` to `(6.5, 6)` (pixels), decided at `seq` 4:

```
$ arachne-oracle decide --xf stream.xf --seq 4 --out d.oracle
decide: seq 4 16x16 window [0, 0, 16, 16]: 1 fragments, 20 rows
$ python3 read.py --canonical d.oracle
header format=1 xf=1.0 size=16x16 window=0,0,16,16 status=decided seq=4
offsets -8,-7 -1,-8 3,-6 5,-5 -6,-1 -4,-4 0,-3 6,-2 -7,2 -2,0 2,3 4,1 -5,5 -3,4 1,7 7,6
fragment id=0 entry=0 slot=0 object=100 part=0 sub=1 layer=1 kind=1 flags=0 group=0 plane=4294967295 epsilon=3d00f876ccdf6cfb bbox=2,2,7,6
cover x=2 y=2 fragment=0 cover=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 cell=255-255,... pixel=255-255 area=3ff0000000000000 ...
...
cover x=6 y=3 fragment=0 cover=1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 cell=255-255,255-255,0-0,0-0,... pixel=127-128 area=3fe0000000000000 ...
...
```

`--canonical` prints one line per item, floats as the hex of their IEEE bits (`3ff0000000000000` is
1, `3fe0000000000000` is 0.5, `7ff8000000000000` NaN), so two decisions files compare as text.

- The fill is `sub` 1 (layer 1 of path 0); its screen shift is about `7.5e-15` px, the rounding of
  its placing alone.
- 20 rows: pixels 2 to 6 across, 2 to 5 down. Pixel `(2, 2)` is covered whole: every sample `Yes`,
  every cell `255-255`, area 1.
- Pixel `(6, 3)` is covered on its left half. The samples whose `dx` is negative (0, 1, 4, 5, 8, 9,
  12, 13) lie in the two left cell columns and are `Yes` with cell area 1; the rest are absent with
  area 0. Its area is exactly 0.5, whose byte `127.5` sits on a rounding boundary, so the pixel
  admits both `127` and `128`.
