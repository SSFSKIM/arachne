//! The SOURCE fixtures (`tests/fixtures/xf15/`): XF files for readers of XF 1.5's section outside
//! the core, each committed as hex text (`<name>.hex`), and `expected.json`, the outcomes XF.md's
//! text gives them (derived by hand, as `STREAMS.md` states; never from this core's reading). This
//! test writes each file and checks that the committed text is those bytes' one hex form
//! ([`hex_text`]) and that `expected.json` pins their sha256; then it
//! reads each file delta by delta (a refused delta refuses that delta alone) and applies it,
//! checking every outcome against `expected.json`. A disagreement is a finding against the core
//! or the text, never a reason to edit `expected.json` to match.

mod common;

use arachne_core::stream::{Stream, StreamError};
use arachne_core::xf::wire::{self, XfError};
use arachne_core::xf::*;
use common::sha256_hex;
use std::collections::BTreeMap;
use std::path::PathBuf;

const P: &str = "arachne.mobject.svg.svg_mobject.VMobjectFromSVGPath";
const M: &str = "arachne.mobject.text.tex_mobject.MathTex";
const G: &str = "arachne.mobject.types.vectorized_mobject.VGroup";
const U: &str = "arachne.Ünïcødé€";
const S: &str = "arachne.mobject.three_d.three_dimensions.Surface";
const SQ: &str = "arachne.mobject.geometry.polygram.Square";
const C: &str = "arachne.mobject.geometry.arc.Circle";

/// The section's codes, as XF.md names them.
const READER_CODES: [&str; 6] = [
    "class_named_twice",
    "slot_named_twice",
    "slot_removed",
    "row_empty",
    "path_empty",
    "class_out_of_range",
];
const APPLY_CODE: &str = "slot_not_held";

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/xf15")
}

// ---------------------------------------------------------------------------------------------
// The writer

fn square() -> Shape {
    Shape::new(ShapeData::Path {
        points: common::square_points(1.0),
    })
}

/// An empty delta, sequence number `seq` at `seq` / 30 s.
fn at(seq: u64) -> Delta {
    Delta::new(
        Time {
            num: seq as i64,
            den: 30,
        },
        seq,
    )
}

/// A stream's first delta: its view and the square every record draws.
fn first() -> Delta {
    let mut d = at(0);
    d.view = Some(common::flat_view(16, 16, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![square()];
    d
}

/// A square in `slot`, carrying nothing.
fn sq(slot: u32) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, slot);
    r.geometry.shape_a = Some(square().key());
    r.paint.fill = Paint::One([0.2, 0.6, 1.0, 0.75]);
    r.depth.space = Space::World;
    r
}

/// A square kept facing the camera (a fixed-orientation centre), carrying nothing.
fn facing(slot: u32) -> Record {
    let mut r = sq(slot);
    r.depth.fixed_orientation_centre = Some([0.0, 0.0, 0.0]);
    r
}

fn paths(paths: &[&[(u32, &str)]]) -> Vec<Vec<SourceNode>> {
    paths
        .iter()
        .map(|p| {
            p.iter()
                .map(|(token, class)| SourceNode {
                    token: *token,
                    class: (*class).into(),
                })
                .collect()
        })
        .collect()
}

fn sourced(mut r: Record, facing_group: u32, source: &[&[(u32, &str)]]) -> Record {
    r.facing_group = facing_group;
    r.source = paths(source);
    r
}

fn row(slot: u32, facing_group: u32, source: &[&[(u32, &str)]]) -> SourceRow {
    SourceRow {
        slot,
        facing_group,
        source: paths(source),
    }
}

fn entries(slots: &[u32]) -> Vec<OrderEntry> {
    slots
        .iter()
        .map(|slot| OrderEntry {
            layer: Layer::Moving,
            slot: *slot,
            part: 0,
        })
        .collect()
}

fn u32b(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// A row's bytes: its slot, its facing group, then each path's (token, class index) nodes.
fn row_bytes(slot: u32, facing_group: u32, paths: &[&[(u32, u32)]]) -> Vec<u8> {
    let mut b = [u32b(slot), u32b(facing_group), u32b(paths.len() as u32)].concat();
    for path in paths {
        b.extend(u32b(path.len() as u32));
        for (token, class) in path.iter() {
            b.extend(u32b(*token));
            b.extend(u32b(*class));
        }
    }
    b
}

/// A section's bytes: its class names, then its rows.
fn section_bytes(classes: &[&str], rows: &[Vec<u8>]) -> Vec<u8> {
    let mut b = u32b(classes.len() as u32);
    for c in classes {
        b.extend(u32b(c.len() as u32));
        b.extend(c.as_bytes());
    }
    b.extend(u32b(rows.len() as u32));
    rows.iter().for_each(|r| b.extend(r));
    b
}

/// `d`'s bytes (it carries no source) as XF 1.5, followed by `section`: a section the writer
/// would never write.
fn with_section(d: &Delta, section: &[u8]) -> Vec<u8> {
    let mut b = wire::encode_delta(d);
    b[2..4].copy_from_slice(&5u16.to_le_bytes());
    b.extend_from_slice(section);
    b
}

/// A file of these deltas' bytes, at the highest minor they state.
fn file(deltas: &[Vec<u8>]) -> Vec<u8> {
    let minor = deltas
        .iter()
        .map(|d| u16::from_le_bytes([d[2], d[3]]))
        .max()
        .unwrap_or(0);
    let mut b = b"ARXF".to_vec();
    b.extend(1u16.to_le_bytes());
    b.extend(minor.to_le_bytes());
    b.extend(u32b(deltas.len() as u32));
    for d in deltas {
        b.extend(u32b(d.len() as u32));
        b.extend(d);
    }
    b
}

/// Every fixture, by its file name, in the order `expected.json` lists them.
fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
    let e = wire::encode_delta;
    let mut out = Vec::new();

    // An older minor's stream: no delta carries the section.
    let mut d0 = first();
    d0.set = vec![facing(1), sq(2)];
    d0.order = Some(entries(&[1, 2]));
    let mut d1 = at(1);
    d1.remove = vec![2];
    d1.order = Some(entries(&[1]));
    out.push(("01-older-minor.arxf", file(&[e(&d0), e(&d1), e(&at(2))])));

    // Sources and facing groups on set records: a glyph with two paths (its parent shared by two
    // groups), a second glyph of its label, a surface's record, a record with no row.
    let mut d0 = first();
    d0.set = vec![
        sourced(
            facing(1),
            3,
            &[&[(11, P), (10, M), (3, G)], &[(11, P), (10, M), (5, U)]],
        ),
        sourced(facing(2), 3, &[&[(12, P), (10, M), (3, G)]]),
        sourced(sq(4), 0, &[&[(20, S)]]),
        sq(5),
    ];
    d0.order = Some(entries(&[1, 2, 4, 5]));
    out.push(("02-set-records.arxf", file(&[e(&d0), e(&at(1))])));

    // A slot set twice in a delta: its last record's row alone.
    let mut d0 = first();
    d0.set = vec![
        sourced(sq(1), 0, &[&[(7, SQ)]]),
        sourced(sq(1), 2, &[&[(8, C)]]),
    ];
    d0.order = Some(entries(&[1]));
    let mut d1 = at(1);
    d1.set = vec![sourced(sq(1), 4, &[&[(9, SQ)]]), sourced(sq(1), 6, &[])];
    out.push(("03-slot-set-twice.arxf", file(&[e(&d0), e(&d1)])));

    // Held rows: a reparented glyph, a changed group, each field cleared, a held frame, a record
    // set again with no row, and a row for a slot the delta sets.
    let mut d0 = first();
    d0.set = vec![
        sourced(facing(1), 3, &[&[(11, P), (10, M)]]),
        sourced(facing(2), 3, &[&[(12, P), (10, M)]]),
    ];
    d0.order = Some(entries(&[1, 2]));
    let mut d1 = at(1);
    d1.source = vec![
        row(1, 3, &[&[(11, P), (13, M)]]),
        row(2, 5, &[&[(12, P), (10, M)]]),
    ];
    let mut d2 = at(2);
    d2.source = vec![row(1, 0, &[&[(11, P), (14, G)]]), row(2, 6, &[])];
    let mut d4 = at(4);
    d4.set = vec![facing(1)];
    let mut d5 = at(5);
    d5.set = vec![sourced(sq(3), 0, &[&[(15, S)]])];
    d5.source = vec![row(3, 7, &[&[(16, S)]]), row(2, 0, &[&[(12, P)]])];
    out.push((
        "04-held-rows.arxf",
        file(&[e(&d0), e(&d1), e(&d2), e(&at(3)), e(&d4), e(&d5)]),
    ));

    // The section first carried mid-stream, after deltas of an older minor.
    let mut d0 = first();
    d0.set = vec![facing(1)];
    d0.order = Some(entries(&[1]));
    let mut d1 = at(1);
    d1.set = vec![facing(1)];
    let mut d2 = at(2);
    d2.set = vec![sourced(facing(2), 3, &[&[(12, P), (10, M)]])];
    d2.order = Some(entries(&[1, 2]));
    let mut d3 = at(3);
    d3.source = vec![row(1, 4, &[&[(11, P), (10, M)]])];
    out.push((
        "05-section-mid-stream.arxf",
        file(&[e(&d0), e(&d1), e(&d2), e(&d3)]),
    ));

    // A newer minor's delta: what follows its section is that minor's, and skipped.
    let mut d0 = first();
    d0.set = vec![sourced(facing(1), 3, &[&[(11, P)]])];
    d0.order = Some(entries(&[1]));
    let mut newer = e(&d0);
    newer[2..4].copy_from_slice(&6u16.to_le_bytes());
    newer.extend([0xa5, 0x5a, 0x00, 0xff, 0x01, 0x02, 0x03, 0x04]);
    out.push(("06-newer-minor.arxf", file(&[newer, e(&at(1))])));

    // A delta refused for something written before its section (an order layer code no build
    // knows), between two that apply.
    let mut d0 = first();
    d0.set = vec![sourced(facing(1), 3, &[&[(11, P)]])];
    d0.order = Some(entries(&[1]));
    let mut d1 = at(1);
    d1.set = vec![sourced(facing(2), 3, &[&[(12, P)]])];
    d1.order = Some(entries(&[1, 2]));
    let mut bad = e(&d1);
    let mut bare = d1.clone();
    bare.set[0].source.clear();
    bare.set[0].facing_group = 0;
    // The order is the last field before the section: its last entry's layer byte.
    let layer = e(&bare).len() - 9;
    assert_eq!(bad[layer], Layer::Moving as u8);
    bad[layer] = 9;
    let mut d2 = at(2);
    d2.source = vec![row(1, 0, &[&[(11, P), (10, M)]])];
    out.push((
        "07-refused-before-the-section.arxf",
        file(&[e(&d0), bad, e(&d2)]),
    ));

    // Refused by a section code: a stream holding two glyphs, a refused delta, and a delta after
    // it. Delta 1 sets a square in slot 3 and, where stated, removes slot 2.
    let held = || {
        let mut d0 = first();
        d0.set = vec![
            sourced(facing(1), 3, &[&[(11, P), (10, M)]]),
            sourced(facing(2), 3, &[&[(12, P), (10, M)]]),
        ];
        d0.order = Some(entries(&[1, 2]));
        e(&d0)
    };
    let later = || {
        let mut d2 = at(2);
        d2.source = vec![row(1, 0, &[&[(11, P)]])];
        e(&d2)
    };
    let refused = |remove: &[u32], classes: &[&str], rows: &[Vec<u8>]| {
        let mut d1 = at(1);
        d1.set = vec![sq(3)];
        d1.remove = remove.to_vec();
        file(&[
            held(),
            with_section(&d1, &section_bytes(classes, rows)),
            later(),
        ])
    };
    out.push((
        "08-class-named-twice.arxf",
        refused(&[], &[P, P], &[row_bytes(1, 0, &[&[(11, 0)]])]),
    ));
    out.push((
        "09-slot-named-twice.arxf",
        refused(
            &[],
            &[P],
            &[
                row_bytes(1, 0, &[&[(11, 0)]]),
                row_bytes(1, 3, &[&[(11, 0)]]),
            ],
        ),
    ));
    out.push((
        "10-slot-removed.arxf",
        refused(&[2], &[P], &[row_bytes(2, 0, &[&[(12, 0)]])]),
    ));
    out.push((
        "11-row-empty.arxf",
        refused(&[], &[P], &[row_bytes(1, 0, &[])]),
    ));
    out.push((
        "12-path-empty.arxf",
        refused(&[], &[P], &[row_bytes(1, 0, &[&[]])]),
    ));
    out.push((
        "13-class-out-of-range.arxf",
        refused(&[], &[P], &[row_bytes(1, 0, &[&[(11, 1)]])]),
    ));
    // Two causes in one delta: the class names come before the rows.
    out.push((
        "14-names-before-rows.arxf",
        refused(&[], &[P, P], &[row_bytes(1, 0, &[])]),
    ));
    // Two causes in two rows: the first row that offends.
    out.push((
        "15-first-row-that-offends.arxf",
        refused(
            &[2],
            &[P],
            &[row_bytes(1, 0, &[]), row_bytes(2, 0, &[&[(12, 0)]])],
        ),
    ));
    // Two causes in one row: the first in XF.md's list.
    out.push((
        "16-first-code-of-a-row.arxf",
        refused(&[2], &[P], &[row_bytes(2, 0, &[])]),
    ));
    // A held row for a slot the stream does not hold: refused by apply.
    let mut d1 = at(1);
    d1.source = vec![row(5, 0, &[&[(13, P)]])];
    out.push(("17-slot-not-held.arxf", file(&[held(), e(&d1), later()])));
    // A row for a slot not held and a row carrying nothing: the reader's code comes first.
    out.push((
        "18-reader-before-apply.arxf",
        refused(
            &[],
            &[P],
            &[row_bytes(5, 0, &[&[(13, 0)]]), row_bytes(1, 0, &[])],
        ),
    ));

    // A slot set twice whose last record carries nothing: no row, so no section.
    let mut d0 = first();
    d0.set = vec![sourced(sq(1), 2, &[&[(7, SQ)]]), sq(1)];
    d0.order = Some(entries(&[1]));
    out.push((
        "19-last-record-carries-nothing.arxf",
        file(&[e(&d0), e(&at(1))]),
    ));
    out
}

// ---------------------------------------------------------------------------------------------
// expected.json, read with a small JSON reader (the crate has no dependency for it)

#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(i64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(m) => m.get(key).unwrap_or(&Json::Null),
            _ => &Json::Null,
        }
    }
    fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            other => panic!("not a string: {other:?}"),
        }
    }
    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a,
            other => panic!("not an array: {other:?}"),
        }
    }
    fn obj(pairs: Vec<(&str, Json)>) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn ws(&mut self) {
        while self.at < self.b.len() && self.b[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }
    fn eat(&mut self, c: u8) {
        self.ws();
        assert_eq!(
            self.b[self.at], c,
            "expected.json: {} at {}",
            c as char, self.at
        );
        self.at += 1;
    }
    fn value(&mut self) -> Json {
        self.ws();
        match self.b[self.at] {
            b'{' => {
                self.at += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.b[self.at] == b'}' {
                    self.at += 1;
                    return Json::Obj(m);
                }
                loop {
                    self.ws();
                    let Json::Str(k) = self.value() else {
                        panic!("expected.json: a key at {}", self.at)
                    };
                    self.eat(b':');
                    let v = self.value();
                    assert!(m.insert(k, v).is_none(), "expected.json: a key twice");
                    self.ws();
                    self.at += 1;
                    match self.b[self.at - 1] {
                        b',' => continue,
                        b'}' => return Json::Obj(m),
                        c => panic!("expected.json: {} at {}", c as char, self.at),
                    }
                }
            }
            b'[' => {
                self.at += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b[self.at] == b']' {
                    self.at += 1;
                    return Json::Arr(a);
                }
                loop {
                    a.push(self.value());
                    self.ws();
                    self.at += 1;
                    match self.b[self.at - 1] {
                        b',' => continue,
                        b']' => return Json::Arr(a),
                        c => panic!("expected.json: {} at {}", c as char, self.at),
                    }
                }
            }
            b'"' => {
                self.at += 1;
                let mut s = Vec::new();
                loop {
                    let c = self.b[self.at];
                    self.at += 1;
                    match c {
                        b'"' => return Json::Str(String::from_utf8(s).expect("UTF-8")),
                        b'\\' => {
                            let e = self.b[self.at];
                            self.at += 1;
                            s.push(match e {
                                b'"' | b'\\' | b'/' => e,
                                b'n' => b'\n',
                                _ => panic!("expected.json: an escape it does not use"),
                            });
                        }
                        _ => s.push(c),
                    }
                }
            }
            b't' | b'f' | b'n' => {
                for (word, v) in [
                    ("true", Json::Bool(true)),
                    ("false", Json::Bool(false)),
                    ("null", Json::Null),
                ] {
                    if self.b[self.at..].starts_with(word.as_bytes()) {
                        self.at += word.len();
                        return v;
                    }
                }
                panic!("expected.json: a word at {}", self.at)
            }
            _ => {
                let start = self.at;
                while self.at < self.b.len()
                    && (self.b[self.at] == b'-' || self.b[self.at].is_ascii_digit())
                {
                    self.at += 1;
                }
                let text = std::str::from_utf8(&self.b[start..self.at]).unwrap();
                Json::Num(text.parse().expect("expected.json: an integer"))
            }
        }
    }
}

fn parse(text: &str) -> Json {
    let mut r = Reader {
        b: text.as_bytes(),
        at: 0,
    };
    let v = r.value();
    r.ws();
    assert_eq!(r.at, r.b.len(), "expected.json: trailing bytes");
    v
}

// ---------------------------------------------------------------------------------------------
// Reading each file as XF.md reads it

/// A file's deltas, each its own item: XF.md refuses a file only for its magic or its major, and
/// a delta refused refuses that delta alone.
fn items(bytes: &[u8]) -> (u16, Vec<&[u8]>) {
    assert_eq!(&bytes[..4], b"ARXF");
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 1);
    let minor = u16::from_le_bytes([bytes[6], bytes[7]]);
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut at = 12;
    let mut out = Vec::new();
    for _ in 0..n {
        let len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        out.push(&bytes[at + 4..at + 4 + len]);
        at += 4 + len;
    }
    assert_eq!(at, bytes.len());
    (minor, out)
}

fn decode_refusal(e: &XfError) -> Json {
    if let XfError::Invalid(why) = e
        && let Some(code) = READER_CODES.iter().find(|c| why.contains(**c))
    {
        return Json::Str((*code).into());
    }
    let error = match e {
        XfError::BadMagic => "BadMagic",
        XfError::Version { .. } => "Version",
        XfError::Truncated { .. } => "Truncated",
        XfError::Invalid(_) => "Invalid",
        XfError::KeyMismatch { .. } => "KeyMismatch",
        XfError::Io(_) => "Io",
    };
    Json::obj(vec![
        ("stage", Json::Str("decode".into())),
        ("error", Json::Str(error.into())),
    ])
}

fn apply_refusal(e: &StreamError) -> Json {
    if e.to_string().contains(APPLY_CODE) {
        return Json::Str(APPLY_CODE.into());
    }
    let error = match e {
        StreamError::Version(_) => "Version",
        StreamError::Missing(_) => "Missing",
        StreamError::InUse(_) => "InUse",
        StreamError::Invalid(_) => "Invalid",
    };
    Json::obj(vec![
        ("stage", Json::Str("apply".into())),
        ("error", Json::Str(error.into())),
    ])
}

/// Every record the stream holds: its slot, source paths (owner first, outermost last) and facing
/// group, by slot.
fn state(s: &Stream) -> Json {
    let records = (0..64)
        .filter_map(|slot| s.record(slot))
        .map(|r| {
            let source = r
                .source
                .iter()
                .map(|path| {
                    Json::Arr(
                        path.iter()
                            .map(|n| {
                                Json::Arr(vec![
                                    Json::Num(n.token.into()),
                                    Json::Str(n.class.clone()),
                                ])
                            })
                            .collect(),
                    )
                })
                .collect();
            Json::obj(vec![
                ("slot", Json::Num(r.slot.into())),
                ("source", Json::Arr(source)),
                ("facing_group", Json::Num(r.facing_group.into())),
            ])
        })
        .collect();
    Json::Arr(records)
}

/// What reading and applying `bytes` delta by delta gives, in `expected.json`'s form.
fn read(bytes: &[u8]) -> (u16, Vec<Json>) {
    let (minor, items) = items(bytes);
    let mut s = Stream::new();
    let mut sectioned = false;
    let mut refused = false;
    let mut out = Vec::new();
    for item in items {
        if refused {
            out.push(Json::obj(vec![("after_refusal", Json::Bool(true))]));
            continue;
        }
        let seq = u64::from_le_bytes(item[20..28].try_into().unwrap()) as i64;
        let refusal = |why: Json| Json::obj(vec![("refused", why), ("at_seq", Json::Num(seq))]);
        let outcome = match wire::decode_delta(item) {
            Err(e) => Err(decode_refusal(&e)),
            Ok(d) => match s.apply(&d) {
                Err(e) => Err(apply_refusal(&e)),
                Ok(_) => Ok(d.version.1),
            },
        };
        out.push(match outcome {
            Err(why) => {
                refused = matches!(why, Json::Str(_));
                refusal(why)
            }
            Ok(delta_minor) => {
                sectioned |= delta_minor >= 5;
                Json::obj(vec![
                    ("seq", Json::Num(seq)),
                    ("minor", Json::Num(delta_minor.into())),
                    (
                        "read_under",
                        Json::Str(if sectioned { "1.5" } else { "1.4" }.into()),
                    ),
                    ("records", state(&s)),
                ])
            }
        });
    }
    (minor, out)
}

// ---------------------------------------------------------------------------------------------
// The tests

#[test]
fn the_committed_fixtures_are_the_writer_s_bytes() {
    let written = fixtures();
    let mut wrong = Vec::new();
    for (name, bytes) in &written {
        let committed = std::fs::read(dir().join(format!("{name}.hex"))).ok();
        if committed.as_deref() != Some(hex_text(bytes).as_bytes()) {
            wrong.push(format!("XF15 {name} {}", hex(bytes)));
        }
    }
    let mut present: Vec<String> = std::fs::read_dir(dir())
        .map(|d| {
            d.map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect()
        })
        .unwrap_or_default();
    present.sort();
    let mut listed: Vec<String> = written.iter().map(|(n, _)| format!("{n}.hex")).collect();
    listed.extend(["STREAMS.md".into(), "expected.json".into()]);
    listed.sort();
    assert!(
        wrong.is_empty() && present == listed,
        "the committed fixtures are not the writer's (present {present:?}):\n{}",
        wrong.join("\n")
    );
}

#[test]
fn each_fixture_reads_as_expected_json_states() {
    let expected = parse(&std::fs::read_to_string(dir().join("expected.json")).unwrap());
    assert_eq!(expected.get("schema").str(), "xf15-fixtures/1");
    let files = expected.get("files").arr();
    let written = fixtures();
    let names: Vec<&str> = files.iter().map(|f| f.get("name").str()).collect();
    let writer: Vec<&str> = written.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, writer, "expected.json lists every fixture, in order");
    let mut wrong = Vec::new();
    for (f, (name, bytes)) in files.iter().zip(&written) {
        let sha = sha256_hex(bytes);
        if f.get("sha256").str() != sha {
            wrong.push(format!("{name}: sha256 is {sha}"));
        }
        let (minor, deltas) = read(bytes);
        if f.get("file_minor") != &Json::Num(minor.into()) {
            wrong.push(format!("{name}: file_minor is {minor}"));
        }
        let want = f.get("deltas").arr();
        if want.len() != deltas.len() {
            wrong.push(format!("{name}: {} deltas", deltas.len()));
            continue;
        }
        for (i, (w, got)) in want.iter().zip(&deltas).enumerate() {
            if w != got {
                wrong.push(format!(
                    "{name}: delta {i}\n  expected {w:?}\n  read     {got:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A fixture's one committed form (`STREAMS.md`): its bytes as lowercase hex digits, 64 to a
/// line, the last line as short as the digits left, every line ending in LF.
fn hex_text(bytes: &[u8]) -> String {
    let digits = hex(bytes);
    let mut text = String::new();
    for line in digits.as_bytes().chunks(64) {
        text.push_str(std::str::from_utf8(line).unwrap());
        text.push('\n');
    }
    text
}
