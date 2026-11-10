//! XF 1.5's SOURCE: which objects each record was drawn from, carried in a section after a delta's
//! order. Its bytes, its refusals, what a stream keeps of it, and that nothing drawn reads it.

mod common;

use arachne_core::stream::{Stream, StreamError};
use arachne_core::xf::wire::{self, XfError};
use arachne_core::xf::*;

fn u32(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn string(s: &str) -> Vec<u8> {
    [u32(s.len() as u32), s.as_bytes().to_vec()].concat()
}

fn node(token: u32, class: &str) -> SourceNode {
    SourceNode {
        token,
        class: class.into(),
    }
}

fn empty() -> Delta {
    Delta::new(Time { num: 0, den: 1 }, 0)
}

/// A glyph of a label, under the label's group, and a second parent path through a shared one.
fn glyph(slot: u32) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, slot);
    r.source = vec![
        vec![node(7, "a.Glyph"), node(3, "a.Group")],
        vec![node(7, "a.Glyph"), node(5, "a.Ünïcødé€")],
    ];
    r.facing_group = 3;
    r
}

/// A held record's new source, its drawing unchanged: a row of its own.
fn held_row(slot: u32, facing_group: u32, source: Vec<Vec<SourceNode>>) -> SourceRow {
    SourceRow {
        slot,
        facing_group,
        source,
    }
}

/// The delta's bytes up to and including its order: what an XF 1.4 reader reads of it.
fn without_section(d: &Delta) -> Delta {
    let mut bare = d.clone();
    bare.source = Vec::new();
    for r in &mut bare.set {
        r.source = Vec::new();
        r.facing_group = 0;
    }
    bare
}

#[test]
fn the_section_follows_the_order_field_by_field() {
    let mut d = empty();
    let mut line = Record::new(RecordKind::Path, 4, 4);
    line.source = vec![vec![node(2, "a.Group")]];
    d.set = vec![glyph(9), line, Record::new(RecordKind::Mesh, 6, 6)];
    d.order = Some(Vec::new());
    d.version = (1, 5);
    let bytes = wire::encode_delta(&d);
    assert_eq!(&bytes[..4], &[1, 0, 5, 0], "XF 1.5");
    let bare = wire::encode_delta(&without_section(&d));
    assert_eq!(
        &bare[..4],
        &[1, 0, 0, 0],
        "the same delta without it is XF 1.0"
    );
    // Everything up to the order is the bare delta's, byte for byte, but the minor.
    assert_eq!(&bytes[4..bare.len()], &bare[4..]);
    let section = [
        u32(3), // class names, interned in the order rows first use them
        string("a.Glyph"),
        string("a.Group"),
        string("a.Ünïcødé€"),
        u32(2), // rows: the records carrying a source or a facing group, in set order
        u32(9), // slot
        u32(3), // facing group
        u32(2), // paths
        u32(2), // nodes of path 0, owner first
        u32(7),
        u32(0),
        u32(3),
        u32(1),
        u32(2), // nodes of path 1
        u32(7),
        u32(0),
        u32(5),
        u32(2),
        u32(4), // slot
        u32(0), // no facing group
        u32(1),
        u32(1),
        u32(2),
        u32(1),
    ]
    .concat();
    assert_eq!(&bytes[bare.len()..], section.as_slice());
    assert_eq!(wire::decode_delta(&bytes).unwrap(), d);
}

#[test]
fn a_held_records_new_source_is_a_row_for_a_slot_the_delta_does_not_set() {
    // A record whose drawing is unchanged is not set again when only its source moves: its new
    // source is a row of its own, after the rows of the records the delta sets, and the bytes
    // before the section are the delta's without it.
    let mut d = empty();
    d.set = vec![glyph(9)];
    d.source = vec![held_row(4, 0, vec![vec![node(2, "a.Group")]])];
    d.version = (1, 5);
    let bytes = wire::encode_delta(&d);
    assert_eq!(&bytes[..4], &[1, 0, 5, 0], "XF 1.5");
    let bare = wire::encode_delta(&without_section(&d));
    assert_eq!(&bytes[4..bare.len()], &bare[4..]);
    let section = [
        u32(3),
        string("a.Glyph"),
        string("a.Group"),
        string("a.Ünïcødé€"),
        u32(2), // rows: the records the delta sets, then the held ones
        u32(9),
        u32(3),
        u32(2),
        u32(2),
        u32(7),
        u32(0),
        u32(3),
        u32(1),
        u32(2),
        u32(7),
        u32(0),
        u32(5),
        u32(2),
        u32(4), // a held slot
        u32(0),
        u32(1),
        u32(1),
        u32(2),
        u32(1),
    ]
    .concat();
    assert_eq!(&bytes[bare.len()..], section.as_slice());
    assert_eq!(wire::decode_delta(&bytes).unwrap(), d);
    // Alone, a row makes the delta XF 1.5 though it sets nothing, and not empty.
    let mut alone = empty();
    alone.source = vec![held_row(4, 6, Vec::new())];
    assert_eq!(wire::encoded_version(&alone), (1, 5));
    assert!(!alone.is_empty());
    alone.version = (1, 5);
    assert_eq!(
        wire::decode_delta(&wire::encode_delta(&alone)).unwrap(),
        alone
    );
}

#[test]
fn a_facing_group_alone_is_a_row_without_paths() {
    let mut d = empty();
    let mut r = Record::new(RecordKind::Path, 2, 2);
    r.facing_group = 11;
    d.set = vec![r];
    d.version = (1, 5);
    let bytes = wire::encode_delta(&d);
    let bare = wire::encode_delta(&without_section(&d));
    let section = [u32(0), u32(1), u32(2), u32(11), u32(0)].concat();
    assert_eq!(&bytes[bare.len()..], section.as_slice());
    assert_eq!(wire::decode_delta(&bytes).unwrap(), d);
}

#[test]
fn a_delta_without_a_source_keeps_its_minor_and_its_bytes() {
    // The canonical samples' pins hold their bytes; a record given a source and then cleared
    // writes what it would never having had one.
    for sample in common::sample_deltas() {
        assert!(wire::encoded_version(&sample).1 < 5);
        let mut given = sample.clone();
        given
            .set
            .iter_mut()
            .for_each(|r| r.source = glyph(r.slot).source);
        if !given.set.is_empty() {
            assert_eq!(wire::encoded_version(&given).1, 5);
        }
        given.set.iter_mut().for_each(|r| r.source = Vec::new());
        assert_eq!(wire::encode_delta(&given), wire::encode_delta(&sample));
    }
    let mut joined = empty();
    let mut r = Record::new(RecordKind::Path, 1, 1);
    r.joins = Some(vec![1]);
    joined.set = vec![r];
    assert_eq!(&wire::encode_delta(&joined)[..4], &[1, 0, 4, 0]);
}

#[test]
fn an_xf_1_4_reader_reads_the_delta_without_its_section() {
    // A reader of the previous minor stops after the order: what it reads is the delta without
    // SOURCE, and each delta of a file is length-prefixed, so the next one is found. Re-encoded,
    // it carries no source: an older reader keeps nothing it skipped.
    let mut d = empty();
    d.set = vec![glyph(9)];
    d.order = Some(vec![OrderEntry {
        layer: Layer::Moving,
        slot: 9,
        part: 0,
    }]);
    d.version = (1, 5);
    let bytes = wire::encode_delta(&d);
    let bare = wire::encode_delta(&without_section(&d));
    let mut as_1_4 = bytes[..bare.len()].to_vec();
    as_1_4[2..4].copy_from_slice(&4u16.to_le_bytes());
    let read = wire::decode_delta(&as_1_4).unwrap();
    assert_eq!(read.set, without_section(&d).set);
    let file = wire::encode_file(&[d.clone(), d.clone()]);
    assert_eq!(&file[4..8], &[1, 0, 5, 0]);
    assert_eq!(wire::decode_file(&file).unwrap(), vec![d.clone(), d]);
}

#[test]
fn bytes_after_the_section_belong_to_a_newer_minor() {
    let mut d = empty();
    d.set = vec![glyph(9)];
    let mut bytes = wire::encode_delta(&d);
    bytes.extend_from_slice(&[0xff, 0xfe, 0xfd]);
    bytes[2..4].copy_from_slice(&6u16.to_le_bytes());
    let read = wire::decode_delta(&bytes).unwrap();
    assert_eq!(read.set, d.set);
    // A newer minor writes the section empty when no record carries one.
    let mut bytes = wire::encode_delta(&empty());
    bytes.extend_from_slice(&[u32(0), u32(0), vec![0xff]].concat());
    assert_eq!(wire::decode_delta(&bytes).unwrap().set, Vec::new());
}

#[test]
fn a_slot_set_twice_carries_the_last_records_source() {
    let mut d = empty();
    let mut first = glyph(9);
    first.facing_group = 1;
    let second = glyph(9);
    d.set = vec![first, second.clone()];
    let read = wire::decode_delta(&wire::encode_delta(&d)).unwrap();
    assert_eq!(read.set[1], second);
    assert!(read.set[0].source.is_empty() && read.set[0].facing_group == 0);
    // Only the last record of a slot is written: one that carries nothing writes no row, and a
    // delta holding no row has no section and is not XF 1.5, whatever the slot's earlier record
    // carried.
    d.set[1] = Record::new(RecordKind::Path, 9, 9);
    let bytes = wire::encode_delta(&d);
    assert_eq!(wire::encoded_version(&d), (1, 0));
    assert_eq!(bytes, wire::encode_delta(&without_section(&d)));
    let read = wire::decode_delta(&bytes).unwrap();
    assert!(read.set.iter().all(|r| !r.carries_source()));
}

/// The bytes of a delta setting `slots` (paths, carrying nothing) and removing `removed`,
/// followed by `section`.
fn with_section(slots: &[u32], removed: &[u32], section: &[u8]) -> Vec<u8> {
    let mut d = empty();
    d.set = slots
        .iter()
        .map(|s| Record::new(RecordKind::Path, *s, *s))
        .collect();
    d.remove = removed.to_vec();
    let mut bytes = wire::encode_delta(&d);
    bytes[2..4].copy_from_slice(&5u16.to_le_bytes());
    bytes.extend_from_slice(section);
    bytes
}

#[test]
fn a_section_that_does_not_name_what_the_delta_sets_is_invalid() {
    let one_path =
        |slot: u32, class: u32| [u32(slot), u32(0), u32(1), u32(1), u32(7), u32(class)].concat();
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "a class index past the table",
            [u32(1), string("a.B"), u32(1), one_path(1, 1)].concat(),
        ),
        (
            "a slot named twice",
            [
                u32(1),
                string("a.B"),
                u32(2),
                one_path(1, 0),
                one_path(1, 0),
            ]
            .concat(),
        ),
        (
            "a held slot named twice",
            [
                u32(1),
                string("a.B"),
                u32(2),
                one_path(3, 0),
                one_path(3, 0),
            ]
            .concat(),
        ),
        (
            "a slot the delta removes",
            [u32(1), string("a.B"), u32(1), one_path(2, 0)].concat(),
        ),
        (
            "a path of no objects",
            [u32(0), u32(1), u32(1), u32(0), u32(1), u32(0)].concat(),
        ),
        (
            "a row carrying nothing",
            [u32(0), u32(1), u32(1), u32(0), u32(0)].concat(),
        ),
        (
            "a class named twice",
            [u32(2), string("a.B"), string("a.B"), u32(1), one_path(1, 0)].concat(),
        ),
    ];
    for (why, section) in cases {
        let bytes = with_section(&[1], &[2], &section);
        assert!(
            matches!(wire::decode_delta(&bytes), Err(XfError::Invalid(_))),
            "{why}: {:?}",
            wire::decode_delta(&bytes)
        );
    }
    // A slot the delta neither sets nor removes is a held record's new source.
    let held = [u32(1), string("a.B"), u32(1), one_path(3, 0)].concat();
    let read = wire::decode_delta(&with_section(&[1], &[2], &held)).unwrap();
    assert_eq!(
        read.source,
        vec![held_row(3, 0, vec![vec![node(7, "a.B")]])]
    );
    assert!(!read.set[0].carries_source());
    let not_utf8 = [u32(1), u32(2), vec![0xc3, 0x28], u32(0)].concat();
    assert!(matches!(
        wire::decode_delta(&with_section(&[1], &[], &not_utf8)),
        Err(XfError::Invalid(_))
    ));
    // Every cut inside the section is truncation, never a section read short.
    let whole = [u32(1), string("a.B"), u32(1), one_path(1, 0)].concat();
    assert!(wire::decode_delta(&with_section(&[1], &[], &whole)).is_ok());
    for cut in 1..whole.len() {
        assert!(
            matches!(
                wire::decode_delta(&with_section(&[1], &[], &whole[..cut])),
                Err(XfError::Truncated { .. })
            ),
            "cut at {cut}"
        );
    }
}

fn square_at(slot: u32, shape: &Shape) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, slot);
    r.geometry.shape_a = Some(shape.key());
    r.paint.fill = Paint::One([0.2, 0.6, 1.0, 0.75]);
    r.paint.stroke = Paint::One([1.0, 1.0, 1.0, 1.0]);
    r.paint.stroke_width = 4.0;
    r.depth.space = Space::World;
    r
}

fn entry(slot: u32) -> OrderEntry {
    OrderEntry {
        layer: Layer::Moving,
        slot,
        part: 0,
    }
}

#[test]
fn a_stream_keeps_a_records_source_until_it_is_set_again() {
    let shape = Shape::new(ShapeData::Path {
        points: common::square_points(1.0),
    });
    let mut s = Stream::new();
    let mut d = empty();
    d.view = Some(common::flat_view(8, 8, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![shape.clone()];
    let mut sourced = square_at(1, &shape);
    sourced.source = glyph(1).source;
    sourced.facing_group = 3;
    d.set = vec![sourced.clone()];
    d.order = Some(vec![entry(1)]);
    s.apply(&d).unwrap();
    assert_eq!(s.record(1), Some(&sourced));
    // A held frame carries nothing and changes nothing.
    s.apply(&empty()).unwrap();
    assert_eq!(s.record(1), Some(&sourced));
    // A record set again without one has none: no stale source survives in its slot.
    let mut d = empty();
    d.set = vec![square_at(1, &shape)];
    s.apply(&d).unwrap();
    let held = s.record(1).unwrap();
    assert!(held.source.is_empty() && held.facing_group == 0);
    // A source path naming no object is refused, as its bytes are.
    let mut d = empty();
    let mut bad = square_at(1, &shape);
    bad.source = vec![Vec::new()];
    d.set = vec![bad];
    assert!(matches!(s.apply(&d), Err(StreamError::Invalid(_))));
}

/// A stream holding a square in slot 1, carrying a source and a facing group; and its shape.
fn holding_a_sourced_square() -> (Stream, Shape, Record) {
    let shape = Shape::new(ShapeData::Path {
        points: common::square_points(1.0),
    });
    let mut s = Stream::new();
    let mut d = empty();
    d.view = Some(common::flat_view(8, 8, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![shape.clone()];
    let mut sourced = square_at(1, &shape);
    sourced.source = glyph(1).source;
    sourced.facing_group = 3;
    d.set = vec![sourced.clone()];
    d.order = Some(vec![entry(1)]);
    s.apply(&d).unwrap();
    (s, shape, sourced)
}

#[test]
fn a_row_gives_a_held_record_its_new_source_and_nothing_else() {
    let (mut s, _, sourced) = holding_a_sourced_square();
    let drawn = s.draw(8, 8, 1).unwrap();
    let mut moved = Delta::new(Time { num: 1, den: 15 }, 1);
    let other = vec![vec![node(8, "a.Other")]];
    moved.source = vec![held_row(1, 0, other.clone())];
    s.apply(&moved).unwrap();
    let mut expected = sourced;
    expected.source = other;
    expected.facing_group = 0;
    assert_eq!(s.record(1), Some(&expected));
    assert_eq!(s.draw(8, 8, 1).unwrap(), drawn);
}

#[test]
fn apply_refuses_a_held_row_by_its_code() {
    // Refused, changing nothing: a held row for a slot the stream does not hold, a second row for a
    // slot, a row for a slot the delta removes, a row carrying nothing, a path of no objects.
    let (mut s, _, sourced) = holding_a_sourced_square();
    let refused = [
        (
            "slot_not_held",
            vec![held_row(5, 0, glyph(5).source)],
            vec![],
        ),
        (
            "slot_named_twice",
            vec![held_row(1, 0, glyph(1).source), held_row(1, 2, Vec::new())],
            vec![],
        ),
        (
            "slot_removed",
            vec![held_row(1, 0, glyph(1).source)],
            vec![1],
        ),
        ("row_empty", vec![held_row(1, 0, Vec::new())], vec![]),
        ("path_empty", vec![held_row(1, 0, vec![Vec::new()])], vec![]),
        // The first in the stated order: a row carrying nothing, for a slot not held.
        ("row_empty", vec![held_row(5, 0, Vec::new())], vec![]),
    ];
    for (code, rows, remove) in refused {
        let mut d = Delta::new(Time { num: 2, den: 15 }, 2);
        d.source = rows;
        d.remove = remove;
        let why = s.apply(&d).unwrap_err().to_string();
        assert!(why.contains(code), "{code}: {why}");
        assert_eq!(s.record(1), Some(&sourced), "{code}");
    }
}

#[test]
fn a_row_for_a_slot_the_delta_sets_is_that_records_metadata() {
    // Not a refusal: the row is the set record's own source and facing group, for a slot held
    // before the delta or not, and it is written as that record's row.
    let (mut s, shape, _) = holding_a_sourced_square();
    let mut d = Delta::new(Time { num: 1, den: 15 }, 1);
    d.set = vec![square_at(1, &shape), square_at(6, &shape)];
    d.source = vec![
        held_row(1, 4, vec![vec![node(8, "a.Other")]]),
        held_row(6, 2, Vec::new()),
    ];
    s.apply(&d).unwrap();
    let one = s.record(1).unwrap();
    assert_eq!(one.source, vec![vec![node(8, "a.Other")]]);
    assert_eq!(one.facing_group, 4);
    assert_eq!(s.record(6).unwrap().facing_group, 2);
    let read = wire::decode_delta(&wire::encode_delta(&d)).unwrap();
    assert!(read.source.is_empty());
    assert_eq!(read.set[0].source, vec![vec![node(8, "a.Other")]]);
    assert_eq!(read.set[0].facing_group, 4);
    assert_eq!(read.set[1].facing_group, 2);
}

#[test]
fn a_held_row_replaces_both_fields_and_one_it_does_not_carry_becomes_none() {
    let (mut s, _, sourced) = holding_a_sourced_square();
    let through_the_wire = |d: &Delta| wire::decode_delta(&wire::encode_delta(d)).unwrap();
    // Paths and a facing group of 0: the record's facing group is cleared.
    let mut d = Delta::new(Time { num: 1, den: 15 }, 1);
    d.source = vec![held_row(1, 0, vec![vec![node(8, "a.Other")]])];
    s.apply(&through_the_wire(&d)).unwrap();
    let held = s.record(1).unwrap();
    assert_eq!(held.source, vec![vec![node(8, "a.Other")]]);
    assert_eq!(held.facing_group, 0);
    // A facing group and no paths: its source is cleared.
    let mut d = Delta::new(Time { num: 2, den: 15 }, 2);
    d.source = vec![held_row(1, 6, Vec::new())];
    s.apply(&through_the_wire(&d)).unwrap();
    let held = s.record(1).unwrap();
    assert!(held.source.is_empty());
    assert_eq!(held.facing_group, 6);
    // Its drawing is as it was.
    let mut drawn = sourced;
    drawn.source = Vec::new();
    drawn.facing_group = 6;
    assert_eq!(s.record(1), Some(&drawn));
}

/// A row's bytes: its slot, its facing group, then each path's (token, class index) nodes.
fn row_bytes(slot: u32, facing_group: u32, paths: &[&[(u32, u32)]]) -> Vec<u8> {
    let mut b = [u32(slot), u32(facing_group), u32(paths.len() as u32)].concat();
    for path in paths {
        b.extend(u32(path.len() as u32));
        for (token, class) in path.iter() {
            b.extend(u32(*token));
            b.extend(u32(*class));
        }
    }
    b
}

/// A section's bytes: its class names, then its rows.
fn section_bytes(classes: &[&str], rows: &[Vec<u8>]) -> Vec<u8> {
    let mut b = u32(classes.len() as u32);
    classes.iter().for_each(|c| b.extend(string(c)));
    b.extend(u32(rows.len() as u32));
    rows.iter().for_each(|r| b.extend(r));
    b
}

/// The reader's refusal of a delta setting slot 1 and removing slot 2, with this section: it
/// reports `code`, not `later`.
fn refused_with(code: &str, later: &str, classes: &[&str], rows: &[Vec<u8>]) {
    let bytes = with_section(&[1], &[2], &section_bytes(classes, rows));
    match wire::decode_delta(&bytes) {
        Err(XfError::Invalid(why)) => {
            assert!(why.contains(code) && !why.contains(later), "{code}: {why}")
        }
        other => panic!("{code}: not refused as invalid: {other:?}"),
    }
}

#[test]
fn the_reader_reports_the_first_refusal_in_order() {
    let one = &["a.B"];
    let held = || row_bytes(3, 0, &[&[(7, 0)]]);
    let removed = || row_bytes(2, 0, &[&[(7, 0)]]);
    refused_with(
        "class_named_twice",
        "slot_named_twice",
        &["a.B", "a.B"],
        &[held(), held()],
    );
    // A slot named twice and a removed slot are never one row's: the first row that offends.
    refused_with(
        "slot_named_twice",
        "slot_removed",
        one,
        &[held(), held(), removed()],
    );
    refused_with(
        "slot_removed",
        "slot_named_twice",
        one,
        &[removed(), held(), held()],
    );
    refused_with("slot_removed", "row_empty", one, &[row_bytes(2, 0, &[])]);
    // A set slot's row carrying nothing is refused as a held one's is.
    refused_with("row_empty", "slot_removed", one, &[row_bytes(1, 0, &[])]);
    refused_with(
        "slot_removed",
        "class_out_of_range",
        one,
        &[row_bytes(2, 0, &[&[(7, 9)]])],
    );
    // A row carrying nothing has no path to be empty: the first row that offends.
    refused_with(
        "row_empty",
        "path_empty",
        one,
        &[row_bytes(4, 0, &[]), row_bytes(5, 0, &[&[]])],
    );
    // An empty path before a class past the names, wherever each is in the row.
    refused_with(
        "path_empty",
        "class_out_of_range",
        one,
        &[row_bytes(3, 0, &[&[(7, 5)], &[]])],
    );
    refused_with(
        "path_empty",
        "class_out_of_range",
        one,
        &[row_bytes(3, 0, &[&[], &[(7, 5)]])],
    );
    refused_with(
        "class_out_of_range",
        "path_empty",
        one,
        &[row_bytes(3, 0, &[&[(7, 1)]])],
    );
    // The rows' order does not change what a row is: a set slot's row after a held one's.
    let read = wire::decode_delta(&with_section(
        &[1],
        &[2],
        &section_bytes(one, &[held(), row_bytes(1, 4, &[])]),
    ))
    .unwrap();
    assert_eq!(read.set[0].facing_group, 4);
    assert_eq!(
        read.source,
        vec![held_row(3, 0, vec![vec![node(7, "a.B")]])]
    );
}

#[test]
fn a_row_for_a_slot_the_delta_sets_and_removes_is_refused_as_removed() {
    // The delta sets slot 1 and removes it, as XF's apply removes a slot set and removed.
    let row = row_bytes(1, 0, &[&[(7, 0)]]);
    let bytes = with_section(&[1], &[1], &section_bytes(&["a.B"], &[row]));
    match wire::decode_delta(&bytes) {
        Err(XfError::Invalid(why)) => assert!(why.contains("slot_removed"), "{why}"),
        other => panic!("not refused as invalid: {other:?}"),
    }
    let (mut s, shape, sourced) = holding_a_sourced_square();
    let mut d = Delta::new(Time { num: 1, den: 15 }, 1);
    d.set = vec![square_at(1, &shape)];
    d.remove = vec![1];
    d.source = vec![held_row(1, 4, Vec::new())];
    let why = s.apply(&d).unwrap_err().to_string();
    assert!(why.contains("slot_removed"), "{why}");
    assert_eq!(s.record(1), Some(&sourced));
}

#[test]
fn a_readers_refusal_comes_before_apply_s_slot_not_held() {
    let (mut s, _, sourced) = holding_a_sourced_square();
    // A held row for slot 9, which the stream does not hold, and a row carrying nothing.
    let not_held = row_bytes(9, 0, &[&[(7, 0)]]);
    let both = section_bytes(&["a.B"], &[not_held.clone(), row_bytes(4, 0, &[])]);
    match wire::decode_delta(&with_section(&[], &[], &both)) {
        Err(XfError::Invalid(why)) => {
            assert!(
                why.contains("row_empty") && !why.contains("slot_not_held"),
                "{why}"
            )
        }
        other => panic!("not refused as invalid: {other:?}"),
    }
    // Without the empty row it is read, and apply refuses it.
    let alone = section_bytes(&["a.B"], &[not_held]);
    let read = wire::decode_delta(&with_section(&[], &[], &alone)).unwrap();
    let why = s.apply(&read).unwrap_err().to_string();
    assert!(why.contains("slot_not_held"), "{why}");
    assert_eq!(s.record(1), Some(&sourced));
}

/// Whether `bytes` are refused by the reader as XF's ordinary invalid XF (`XfError::Invalid`)
/// naming `what`, and with none of the section's codes.
fn refused_before_the_section(bytes: &[u8], what: &str) {
    match wire::decode_delta(bytes) {
        Err(XfError::Invalid(why)) => {
            assert!(why.contains(what) && !why.contains("row_empty"), "{why}")
        }
        other => panic!("not refused as invalid: {other:?}"),
    }
}

#[test]
fn a_refusal_of_anything_written_before_the_section_is_the_deltas() {
    // Each delta's section holds a row carrying nothing; something written before it is refused
    // first: the time, its first field after the version, and the order, its last before the
    // section.
    let section = section_bytes(&["a.B"], &[row_bytes(4, 0, &[])]);
    let mut timed = with_section(&[1], &[2], &section);
    timed[12..20].copy_from_slice(&0i64.to_le_bytes());
    refused_before_the_section(&timed, "denominator 0");
    let mut d = empty();
    d.order = Some(vec![entry(1)]);
    let mut ordered = wire::encode_delta(&d);
    let layer = ordered.len() - 9;
    ordered[layer] = 7;
    ordered[2..4].copy_from_slice(&5u16.to_le_bytes());
    ordered.extend_from_slice(&section);
    refused_before_the_section(&ordered, "7 is not a valid layer");
}

#[test]
fn apply_s_other_checks_come_before_slot_not_held() {
    // Each delta also gives slot 9, which the stream does not hold, a held row; apply's other
    // refusal is the delta's: a record naming a shape the stream would not hold, and an order
    // drawing a slot not set, each `StreamError::Missing`.
    let (mut s, _, sourced) = holding_a_sourced_square();
    let unheld = Shape::new(ShapeData::Path {
        points: common::square_points(2.0),
    });
    let not_held = || vec![held_row(9, 0, glyph(9).source)];
    let mut shaped = Delta::new(Time { num: 1, den: 15 }, 1);
    shaped.set = vec![square_at(2, &unheld)];
    shaped.source = not_held();
    let mut ordered = Delta::new(Time { num: 1, den: 15 }, 1);
    ordered.order = Some(vec![entry(1), entry(7)]);
    ordered.source = not_held();
    for (d, what) in [
        (shaped, "the stream would not hold"),
        (ordered, "the order draws slot 7"),
    ] {
        match s.apply(&d).unwrap_err() {
            StreamError::Missing(why) => {
                assert!(
                    why.contains(what) && !why.contains("slot_not_held"),
                    "{why}"
                )
            }
            other => panic!("{what}: not refused as missing: {other}"),
        }
        assert_eq!(s.record(1), Some(&sourced), "{what}");
    }
}

#[test]
fn a_section_code_comes_before_apply_s_other_checks() {
    // A delta built in memory, not read: its section's codes are still the reader's, so they come
    // before apply's other refusals, here a shape uploaded twice (`StreamError::Invalid`) and a
    // record naming a shape the stream would not hold (`StreamError::Missing`).
    let (mut s, shape, sourced) = holding_a_sourced_square();
    let unheld = Shape::new(ShapeData::Path {
        points: common::square_points(2.0),
    });
    let at = || Delta::new(Time { num: 1, den: 15 }, 1);
    let mut empty_row = at();
    empty_row.upload = vec![shape.clone(), shape.clone()];
    empty_row.source = vec![held_row(1, 0, Vec::new())];
    let mut removed = at();
    removed.upload = vec![shape.clone(), shape.clone()];
    removed.remove = vec![1];
    removed.source = vec![held_row(1, 0, glyph(1).source)];
    let mut empty_path = at();
    let mut pathless = square_at(2, &unheld);
    pathless.source = vec![Vec::new()];
    empty_path.upload = vec![shape.clone(), shape.clone()];
    empty_path.set = vec![pathless];
    // A record the delta sets and removes, carrying a source: its row is refused as removed.
    let mut set_and_removed = at();
    set_and_removed.upload = vec![shape.clone(), shape.clone()];
    set_and_removed.set = vec![sourced.clone()];
    set_and_removed.remove = vec![1];
    for (d, code) in [
        (empty_row, "row_empty"),
        (removed, "slot_removed"),
        (empty_path, "path_empty"),
        (set_and_removed, "slot_removed"),
    ] {
        match s.apply(&d).unwrap_err() {
            StreamError::Invalid(why) => assert!(why.contains(code), "{code}: {why}"),
            other => panic!("{code}: not refused as invalid: {other}"),
        }
        assert_eq!(s.record(1), Some(&sourced), "{code}");
    }
    // Read from bytes, the reader refuses it, and apply never sees the delta: a record with no
    // shape, which apply refuses (`StreamError::Invalid`), and a row carrying nothing.
    let bare = wire::decode_delta(&with_section(&[1], &[], &[])).unwrap();
    let why = s.apply(&bare).unwrap_err().to_string();
    assert!(why.contains("needs a shape"), "{why}");
    let rowed = with_section(&[1], &[], &section_bytes(&[], &[row_bytes(1, 0, &[])]));
    match wire::decode_delta(&rowed) {
        Err(XfError::Invalid(why)) => assert!(why.contains("row_empty"), "{why}"),
        other => panic!("not refused as invalid: {other:?}"),
    }
}

#[test]
fn the_reader_refuses_a_row_for_a_removed_slot_or_a_slot_named_twice() {
    // The delta sets slot 1 and removes slot 2; slot 3 is held.
    let row = |slot: u32| [u32(slot), u32(0), u32(1), u32(1), u32(7), u32(0)].concat();
    let section =
        |rows: &[Vec<u8>]| [u32(1), string("a.B"), u32(rows.len() as u32), rows.concat()].concat();
    for (code, rows) in [
        ("slot_removed", vec![row(2)]),
        ("slot_named_twice", vec![row(3), row(3)]),
        ("slot_named_twice", vec![row(1), row(1)]),
        ("slot_named_twice", vec![row(1), row(3), row(1)]),
    ] {
        let read = wire::decode_delta(&with_section(&[1], &[2], &section(&rows)));
        let Err(XfError::Invalid(why)) = read else {
            panic!("{code}: {read:?}");
        };
        assert!(why.contains(code), "{code}: {why}");
    }
    let read = wire::decode_delta(&with_section(&[1], &[2], &section(&[row(1), row(3)])));
    assert_eq!(read.unwrap().source.len(), 1);
}

/// With the labels' placement off, nothing drawn reads a record's source or facing group; with it
/// on (a new stream's), the label they make is placed.
#[test]
fn nothing_drawn_reads_a_records_source_but_the_labels_placement() {
    // A label kept facing the camera, half behind a slanted sheet, and the sheet: drawn from deltas
    // carrying a source and a facing group, and from the same deltas without, through the wire.
    let flat = Shape::new(ShapeData::Path {
        points: common::square_points(1.0),
    });
    let slanted = Shape::new(ShapeData::Path {
        points: common::square_points(1.6)
            .into_iter()
            .map(|[x, y, _]| [x, y, 0.5 * x])
            .collect(),
    });
    let mut label = square_at(1, &flat);
    label.depth.fixed_orientation_centre = Some([0.0, 0.0, 0.0]);
    label.source = glyph(1).source;
    label.facing_group = 3;
    let mut sheet = square_at(2, &slanted);
    sheet.paint.fill = Paint::One([0.9, 0.3, 0.1, 0.6]);
    sheet.source = vec![vec![node(8, "a.Surface")]];
    let mut first = empty();
    first.view = Some(common::three_d_view(48, 32));
    first.upload = vec![flat, slanted];
    first.set = vec![label.clone(), sheet];
    first.order = Some(vec![entry(1), entry(2)]);
    // The second frame moves the label's source alone.
    let mut second = Delta::new(Time { num: 1, den: 15 }, 1);
    label.source = vec![vec![node(7, "a.Glyph")]];
    second.set = vec![label];
    // The third moves it again, through a row: the record is not set again.
    let mut third = Delta::new(Time { num: 2, den: 15 }, 2);
    third.source = vec![held_row(1, 3, vec![vec![node(9, "a.Other")]])];
    let frames = |deltas: &[Delta], placed: bool| {
        let mut s = Stream::new();
        s.set_label_placement(placed);
        deltas
            .iter()
            .map(|d| {
                let read = wire::decode_delta(&wire::encode_delta(d)).unwrap();
                assert_eq!(read.set, d.set);
                assert_eq!(read.source, d.source);
                s.apply(&read).unwrap();
                s.draw(48, 32, 2).unwrap()
            })
            .collect::<Vec<_>>()
    };
    let deltas = [first, second, third];
    let bare: Vec<Delta> = deltas.iter().map(without_section).collect();
    let drawn = frames(&deltas, false);
    assert!(drawn[0].chunks(4).any(|px| px != [0, 0, 0, 255]));
    assert_eq!(drawn, frames(&bare, false));
    assert_eq!(frames(&bare, true), frames(&bare, false));
    assert_ne!(frames(&deltas, true), drawn);
}
