# XF 1.5 SOURCE fixtures: the stream contract

These are XF files (`ARXF`, XF.md's Binary form) for readers of XF 1.5's section, the SOURCE
section, outside this core. Every file is committed, together with `expected.json`, which states
what XF.md's text makes of each delta. `core/tests/xf15_fixtures.rs` writes the files, checks that
the committed bytes are the writer's, and reads and applies each file against `expected.json`.

Each file `<name>` is committed as hex text, `<name>.hex`, in exactly one form: its bytes, in
order, as two lowercase hex digits each (`0-9`, `a-f`), written 64 digits to a line; the last line
holds the digits left over (64 or fewer, an even number) with no padding; every line, the last
included, ends with a single LF (0x0a), so the file ends with a newline; nothing else is in the
file. A reader decodes it by dropping the LFs and reading each pair of digits as one byte. Every
`name` and `sha256` in `expected.json` is the decoded file's.

## How the expected values were derived

The files' bytes are inputs; the writer made them, and some were written byte by byte to hold
sections no writer would write. Every value in `expected.json` was derived by hand from XF.md's text
(the "XF 1.5 says which objects ..." paragraph and the ones after it, "Refusals, each by its code",
"Where each check is made", and the Binary form), from the deltas described below, and never from
this core's reading of the files. Where the core reads a file otherwise, that is a finding against
the core or the text, and `expected.json` is not edited to match.

## expected.json, schema `xf15-fixtures/1`

```
{ "schema": "xf15-fixtures/1",
  "files": [ { "name", "sha256", "file_minor", "deltas": [ entry, ... ] }, ... ] }
```

- `name`: the file, committed here as `<name>.hex`. `sha256`: of its decoded bytes, as hex.
  `file_minor`: the minor its header states.
- `deltas`: one entry per delta of the file, in stream order. A reader reads each delta as its own
  item: XF.md refuses a file only for its magic or its major, and a refused delta is not applied. (A
  reader may refuse the whole file where a delta is refused; it never applies a refused delta.)
- An applied delta: `{"seq", "minor", "read_under", "records"}`.
  - `seq` is the delta's sequence number, and `minor` the minor its bytes state.
  - `read_under` is `"1.5"` once this delta, or an earlier one, has carried the section (a delta
    stating minor 5 or later), and `"1.4"` until then. Under `"1.4"` no record carries a facing
    group. Under `"1.5"`, a record kept facing the camera with no facing group is of unknown label
    membership (XF.md's "Facing groups" paragraph).
  - `records` lists every record the stream holds after the delta, by slot ascending, as
    `{"slot", "source", "facing_group"}`. `source` lists the record's paths of containment, each as
    `[token, class]` nodes, the record's owner first and its outermost container last; `[]` is no
    source. `facing_group` 0 is none.
- A delta refused by one of the section's seven codes: `{"refused": "<code>", "at_seq": n}`, `n`
  being the sequence number its bytes state. The codes are `class_named_twice`, `slot_named_twice`,
  `slot_removed`, `row_empty`, `path_empty` and `class_out_of_range` (the reader's), and
  `slot_not_held` (apply's).
- A delta refused otherwise: `{"refused": {"stage": "decode" | "apply", "error": "<name>"},
  "at_seq": n}`, with XF.md's names from "Where each check is made": at decode `Truncated`,
  `Invalid`, `Version` or `KeyMismatch` (and `BadMagic` for a file); at apply the `StreamError`
  names. The stream goes on: later deltas apply to the state before the refused one.
- After a delta refused by a section code, every later delta of the file is
  `{"after_refusal": true}`, with no state. From that delta on, the stream's SOURCE cannot be
  judged (the truth check counts it unable), and a reader that refuses the whole file refuses this
  one; with no state stated, both kinds of reader agree with the file's entry.

## The deltas

Every file's first delta sets a 16 x 16 flat view and uploads one square path. Every record is a
`path` record drawing that square. Records "facing the camera" carry a fixed-orientation centre at
the origin; the others carry none. Delta `n` has sequence number `n` and time `n`/30. Class names
are written in full in `expected.json`; below they are short:

| Short | Class |
|---|---|
| P | `arachne.mobject.svg.svg_mobject.VMobjectFromSVGPath` |
| M | `arachne.mobject.text.tex_mobject.MathTex` |
| G | `arachne.mobject.types.vectorized_mobject.VGroup` |
| U | `arachne.Ünïcødé€` (a name that is not ASCII) |
| S | `arachne.mobject.three_d.three_dimensions.Surface` |
| SQ | `arachne.mobject.geometry.polygram.Square` |
| C | `arachne.mobject.geometry.arc.Circle` |

A path is written owner first, e.g. `[11 P, 10 M, 3 G]`: object 11, a P, inside 10, inside 3.

1. **`01-older-minor.arxf`**: no delta carries the section. Delta 0 sets slot 1 (facing) and slot
   2, and orders both; delta 1 removes slot 2; delta 2 is empty.
2. **`02-set-records.arxf`**: delta 0 sets
   - slot 1, facing, group 3, two paths `[11 P, 10 M, 3 G]` and `[11 P, 10 M, 5 U]` (object 10 is
     in two parents);
   - slot 2, facing, group 3, `[12 P, 10 M, 3 G]`;
   - slot 4, group 0, `[20 S]`;
   - slot 5, with no row.

   Its section's class names are `P, M, G, U, S`, in the order its rows first use them. Delta 1 is
   empty.
3. **`03-slot-set-twice.arxf`**: delta 0 sets slot 1 twice, first with `[7 SQ]` and group 0, then
   with `[8 C]` and group 2; its section names `C` alone. Delta 1 sets slot 1 twice, first with
   `[9 SQ]` and group 4, then with group 6 and no path; its section has no class names and one row.
4. **`04-held-rows.arxf`**:
   - delta 0 sets slots 1 and 2, both facing and group 3, with `[11 P, 10 M]` and `[12 P, 10 M]`;
   - delta 1 holds rows `(1, group 3, [11 P, 13 M])`, the glyph reparented, and
     `(2, group 5, [12 P, 10 M])`, the group changed;
   - delta 2 holds rows `(1, group 0, [11 P, 14 G])` and `(2, group 6, no path)`;
   - delta 3 is empty;
   - delta 4 sets slot 1 again (facing) with no row;
   - delta 5 sets slot 3 (not facing) and holds rows `(3, group 7, [16 S])`, a row for the slot
     the delta sets, and `(2, group 0, [12 P])`; its class names are `S, P`.
5. **`05-section-mid-stream.arxf`**: deltas 0 and 1 carry no section; delta 0 sets slot 1
   (facing) and delta 1 sets it again. Delta 2 sets slot 2, facing, group 3, `[12 P, 10 M]`: the
   first section. Delta 3 holds a row `(1, group 4, [11 P, 10 M])`.
6. **`06-newer-minor.arxf`**: delta 0 states minor 6. It sets slot 1, facing, group 3, `[11 P]`,
   and eight bytes follow its section (`a5 5a 00 ff 01 02 03 04`), which belong to that newer
   minor. Delta 1 is empty. The header states minor 6.
7. **`07-refused-before-the-section.arxf`**: delta 0 sets slot 1, facing, group 3, `[11 P]`. Delta
   1 sets slot 2, facing, group 3, `[12 P]`, and orders slots 1 and 2, but the layer code of its
   order's second entry is 9, which no build knows. Delta 2 holds a row `(1, group 0,
   [11 P, 10 M])`.
8. **`08-` to `18-`**: refused by a section code.
   - Delta 0 sets slots 1 and 2, both facing and group 3, with `[11 P, 10 M]` and `[12 P, 10 M]`.
   - Delta 1 sets slot 3 (no row of its own) and states minor 5; its section, written byte by
     byte, is below. Rows are `(slot, group, paths)`, and a node is `token:class index`. Where
     stated, delta 1 also removes slot 2.
   - Delta 2 holds a row `(1, group 0, [11 P])`.

   | File | Class names | Rows | Removes |
   |---|---|---|---|
   | `08-class-named-twice.arxf` | P, P | `(1, 0, [11:0])` | |
   | `09-slot-named-twice.arxf` | P | `(1, 0, [11:0])`, `(1, 3, [11:0])` | |
   | `10-slot-removed.arxf` | P | `(2, 0, [12:0])` | 2 |
   | `11-row-empty.arxf` | P | `(1, 0, no path)` | |
   | `12-path-empty.arxf` | P | `(1, 0, one path of no node)` | |
   | `13-class-out-of-range.arxf` | P | `(1, 0, [11:1])` | |
   | `14-names-before-rows.arxf` | P, P | `(1, 0, no path)` | |
   | `15-first-row-that-offends.arxf` | P | `(1, 0, no path)`, `(2, 0, [12:0])` | 2 |
   | `16-first-code-of-a-row.arxf` | P | `(2, 0, no path)` | 2 |
   | `18-reader-before-apply.arxf` | P | `(5, 0, [13:0])`, `(1, 0, no path)` | |

   `17-slot-not-held.arxf`'s delta 1 is as the writer writes it: one held row `(5, group 0,
   [13 P])`, for a slot the stream does not hold.
9. **`19-last-record-carries-nothing.arxf`**: delta 0 sets slot 1 twice, first with `[7 SQ]` and
   group 2, then with nothing, and orders it. Only a slot's last record has a row, so the delta
   holds none: it has no section and is not XF 1.5. Delta 1 is empty.

## The files' sha256

Per file: the sha256 of its decoded bytes (the ARXF file, as `expected.json` pins it), and the
sha256 of the committed `<name>.hex` text.

| File | sha256 of the decoded bytes | sha256 of `<name>.hex` |
|---|---|---|
| `01-older-minor.arxf` | `8e968ef5098a9b0ddd0ac586d14dba332d93b4ac700136dc55e4fe0476ef22f4` | `e967bd94f2f040670e59c0ed1ce043136c6ac848b50a88a1ea044b6b495983a7` |
| `02-set-records.arxf` | `719ad384c540932af25a1576ddd996b58f386714de7a391aa16a09e733896533` | `898462ccc850db0e09b26516ca63964b29a313d1912afe4dbe4b7cd991c19424` |
| `03-slot-set-twice.arxf` | `3c390ec4db0d32ffae212fc5c9f8e6c32874f3ff5304e20b266f0c600cb97b72` | `f36ae5d5cd8cc978c69a77abdbd37901ff46c780e26584cb68c273385957f4e1` |
| `04-held-rows.arxf` | `e152278bfa4d8409f6847544878501267c2c0f381fab07d5bd5f0f6a9fe395b5` | `a0f9fc96d368b8bea70f47e5d0370d69bde067878abf3d19ab1c56b2fcfb2ca8` |
| `05-section-mid-stream.arxf` | `a0787e04d966d9d0f605f4c75c3c6ec438ab31806ea08c8b3133873c597c0806` | `34fba5e5a8e925cc740abd769878818832d75401b434c3454ec179c7f33c3f66` |
| `06-newer-minor.arxf` | `6b9e71aa1d1f29259990e87ed58271a4b13cc2172a01695efc073151a795274f` | `da3b4215ca2cf29472d1481f97b9004abe841994930e98b0e9cda438da0c864e` |
| `07-refused-before-the-section.arxf` | `0fbfae68d4d5f09cdae8ebf9d5c99c08d01b5f5940331617a9bfb5c12deebc7b` | `cbb85c1e7207f13de609db2480caad3ec30c70437d773af3f242ed707b1e3c16` |
| `08-class-named-twice.arxf` | `8b72af1551013a6c44f7750ec8a398ebe7333184a268ae62bd5f695a99490b60` | `04d5c95f87b5b7e9a97e8f5a66c7afec543dc23d31c900c550ce722f88ed9cf3` |
| `09-slot-named-twice.arxf` | `a1fc5f52bf3d333a5bf24c929ed4893197a1854c005cbc7661355ac96bcac17a` | `23c91d0610be8f91bdf0a5c39bb7a1c2e6589bc229e7127641342b092b9db9c6` |
| `10-slot-removed.arxf` | `40a3fa636b804fc97388b08ee83bda75b55027d51e83e80adf3077998c607595` | `935b561f92f5554e5b804e59a3b7bbec938fb6962b6fda46cee1864f6a3d7cf7` |
| `11-row-empty.arxf` | `a20986c0a4c0de0a28e905cbd93acbdd078da62638816d04a307c9431424ea7e` | `8c188c57875de55859936813be24c8fd19a7a6ea0ce3783c17277e31867aec38` |
| `12-path-empty.arxf` | `09cef264750377bf332c7633651993359ac1ba8b10c1641a3b8c2ca4d1f0d6df` | `3e4a99050f3f0bfb8662ebe38e4c72674c7571634d402782c864e5226eb7415c` |
| `13-class-out-of-range.arxf` | `3345bc07a99b6fe8b299016af9b2412792efef35ad504cf1a6dcd0d829f9a17e` | `be8e64760c78ea1fab5890af0c1b8f9da28ba6025f363aff307c57d832b1e9c1` |
| `14-names-before-rows.arxf` | `eb42bd0d882c28b6c0a47c1648d7b8060421f929ef54ca2d9cf2da8a1fd50b67` | `5980670ca9e0062b4c9e910f60ead7aac856cb1c5f72a2c32c6b6fa5c6fe332d` |
| `15-first-row-that-offends.arxf` | `5fbe85ea9e2e28c3b9af5373284c707056e05fccaca679ab4508288352540b63` | `6f902538f1d08d45bfe03c360a44a06797a0e1018d91c06b3d26f899ef8b6a8c` |
| `16-first-code-of-a-row.arxf` | `25363e5c29387033f7da7aeea608a4445c471cdd9c65026523dbf7efe79dd249` | `ab5aaf99841780f3565a728ecc0a1ef744e6bd64d0c8cece3d11cf54abda80e3` |
| `17-slot-not-held.arxf` | `0a33d76e3d28a7024aced12dea035b35336f2259ec452f3f402d586878654768` | `93e9d668203615479ab6448b0872bc64956d6c3edc1b8ec99f1c0e3581d5e7f5` |
| `18-reader-before-apply.arxf` | `c8af207edc1d30df39f1e53f47aae1bfbc4c60a58bc43d9ee1f6ca19e74f6090` | `11a05f83ce4755c5947ece51ca76153dfc120172b630b1a961344d9af09463ae` |
| `19-last-record-carries-nothing.arxf` | `f4ae2e3e435acb0e48fbc8741ccaf76fad1f6fed90c0f9aa3baa578a656ccac0` | `16eab0e825c4e550ffaf0318b2b4cd278e7c993388b8a6eaad220af300f7820f` |
