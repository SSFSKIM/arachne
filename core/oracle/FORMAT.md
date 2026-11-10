# The oracle's decisions file: `ORACLE_FORMAT` 1

`arachne-oracle decide --xf <file> --seq <delta seq> --out <file> [--pixels x0,y0,x1,y1]` applies the
XF file's deltas in order, from an empty stream, up to and including the delta whose `seq` is given,
and writes what the frame drawn there should hold: every fragment the draw order draws and its
coverage verdict at each of the 16 samples of every pixel of the window (the whole frame by
default), or why the frame is `truth_unable`. It carries decisions, never pixel bytes.

`read.py` beside this file is the one reader for other programs (standard library only); its
`DTYPES` give each section's layout as a numpy dtype, so a section also reads whole with
`numpy.fromfile(path, numpy.dtype(DTYPES[tag]), count, offset=offset)`.

The exit status is 0 when the file is written, a refused frame included, and 2 for a usage or
input error (an XF file that does not read, a `seq` it does not hold).

## Layout

Little-endian throughout. A fixed header of 88 bytes, a directory of sections, then each section
on an 8-byte boundary. Sections are arrays of fixed-size records; a reader skips a tag it does not
know, and treats a section that is absent as empty. Later sections (pairwise relations, colours)
arrive as new tags.

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
| `FRAG` | 64 bytes | one per fragment that has a row |
| `COVR` | 456 bytes | one per fragment at a pixel, sorted by `(y, x, fragment)` |
| `PIXL` | 12 bytes | one per pixel of the window, row by row: its first `COVR` row and their count |
| `REFU` | 32 bytes | one per refusal; empty when the frame is decided |
| `TEXT` | 1 byte | the refusals' words, UTF-8 |

### `FRAG`

```
u32 id  u32 entry (order index)  u32 slot  u32 object  u32 part  u32 sub
u8 layer  u8 kind (record kind code)  u16 flags  u32 group
f64 epsilon (screen shift, px)  i32 x0, y0, x1, y1 (its rows' pixel box)  u32 plane  u32 reserved
```

- **Identity.** `(object, part, layer, sub)` names a fragment as the core's report does. `layer`:
  0 background stroke, 1 fill, 2 stroke, 3 mesh triangle, 4 point block. `sub` is a path layer's
  `3 i + k` (`i` the instance member, else 0; `k` the layer), a mesh triangle's index in its
  shape (`t i` plus it for member `i` of a `t`-triangle prototype), a point block pixel's
  `(offset index) · n + i`. An instance's fragments carry the instance's object.
- **`group`**: the drawn path a layer belongs to (its three layers share it); `0xffffffff` for a
  mesh triangle or a point block. **`plane`**: the registered plane identity, `0xffffffff` until
  depth is written.
- **`flags`**: bit 0, *unbounded*: the record's screen shift is infinite, so its samples are
  `truth_unable` (its rows mark every sample `3`).

### `COVR`

```
u32 x  u32 y  u32 fragment
u8 cover[16]  u8 cell_lo[16]  u8 cell_hi[16]  u8 pixel_lo  u8 pixel_hi  2 bytes pad
f64 pixel_area  f64 cell_area[16]  f64 depth[16]  f64 band[16]
```

- **`cover[s]`** at sample `s`: 0 absent, 1 `Yes`, 2 `Either`, 3 unbounded (`truth_unable`).
- **`cell_lo[s]`, `cell_hi[s]`**: the cell bytes admitted at `s` (a mask: `255, 255` for `Yes`,
  `0, 255` for `Either`).
- **`pixel_lo`, `pixel_hi`**: the whole-pixel coverage bytes admitted.
- **`pixel_area`, `cell_area[s]`**: a path layer's exact nonzero-rule area of the pixel and of
  sample `s`'s cell, each as a fraction of it (1 covered whole); NaN for a mesh triangle or a point
  block, whose verdicts are masks.
- **`depth`, `band`**: NaN in this version.
- A row is written where some sample is not absent or the admitted pixel bytes reach above 0.

### `REFU`

```
u8 reason  u8 kind (0xff: none)  2 bytes pad  u32 entry  u32 slot  u32 object (0xffffffff: none)
u64 bits (unknown feature bits)  u32 text start  u32 text length (into TEXT)
```

Reasons: 1 `unknown_kind`, 2 `reserved_kind`, 3 `unknown_feature`, 4 `unknown_light_model`,
5 `xf_newer` (a delta of an XF minor above 4 at or before the frame), 6 `invalid_delta` (the frame's
delta is not valid XF; the stream keeps what it held), 7 `no_view`, 8 `core_refuses` (a refusal of
the core's own list).

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
