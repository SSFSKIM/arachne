//! XF's binary form: round trips, byte stability, and refusing what a reader does not know without
//! failing to parse.

mod common;

use arachne_core::xf::wire::{self, XfError};
use arachne_core::xf::*;
use std::path::PathBuf;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

#[test]
fn every_field_round_trips() {
    for delta in common::sample_deltas() {
        let bytes = wire::encode_delta(&delta);
        assert_eq!(wire::decode_delta(&bytes).unwrap(), delta);
    }
    let deltas = common::sample_deltas();
    assert_eq!(
        wire::decode_file(&wire::encode_file(&deltas)).unwrap(),
        deltas
    );
}

/// Writes bytes as lowercase hex, 64 bytes to a line.
fn to_hex(bytes: &[u8]) -> String {
    bytes
        .chunks(64)
        .map(|line| line.iter().map(|b| format!("{b:02x}")).collect::<String>() + "\n")
        .collect()
}

fn from_hex(text: &str) -> Vec<u8> {
    let digits: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    digits
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

/// The sample's bytes are pinned (as hex text, since the repository holds no binary fixtures): a
/// change to the layout shows here. Regenerate the file with `XF_BLESS=1 cargo test --test wire`
/// only when the layout is meant to change.
#[test]
fn the_layout_is_byte_stable() {
    let path = data("sample-v0.xf.hex");
    let bytes = wire::encode_file(&common::sample_deltas());
    if std::env::var_os("XF_BLESS").is_some() {
        std::fs::write(&path, to_hex(&bytes)).unwrap();
    }
    let pinned = from_hex(&std::fs::read_to_string(&path).unwrap());
    assert!(bytes == pinned, "the encoded sample differs from {path:?}");
    assert_eq!(
        wire::decode_file(&pinned).unwrap(),
        common::sample_deltas(),
        "reading the pinned bytes"
    );
}

#[test]
fn an_empty_delta_is_small_and_empty() {
    let delta = Delta::new(Time { num: 3, den: 2 }, 9);
    assert!(delta.is_empty());
    let back = wire::decode_delta(&wire::encode_delta(&delta)).unwrap();
    assert!(back.is_empty());
    assert_eq!(back.t, Time { num: 3, den: 2 });
    assert_eq!(back.seq, 9);
}

/// Splices raw record bytes into a delta whose `set` section holds exactly one record.
fn delta_with_raw_record(record: &[u8]) -> Vec<u8> {
    let mut delta = Delta::new(Time { num: 0, den: 1 }, 0);
    delta.set = vec![Record::new(RecordKind::Path, 0, 0)];
    let template = wire::encode_delta(&delta);
    let own = wire::encode_record(&delta.set[0]);
    let at = template
        .windows(own.len())
        .position(|w| w == own.as_slice())
        .expect("the record's bytes are in the delta");
    // The record is preceded by its u32 length.
    let mut out = template[..at - 4].to_vec();
    out.extend_from_slice(&(record.len() as u32).to_le_bytes());
    out.extend_from_slice(record);
    out.extend_from_slice(&template[at + own.len()..]);
    out
}

#[test]
fn a_kind_this_reader_does_not_know_is_kept_not_parsed() {
    // Header: kind 200, features, slot 42, object 7; then bytes no reader of XF 0 can parse.
    let mut raw = vec![200u8];
    raw.extend_from_slice(&5u64.to_le_bytes());
    raw.extend_from_slice(&42u32.to_le_bytes());
    raw.extend_from_slice(&7u32.to_le_bytes());
    raw.extend_from_slice(&[0xff; 11]);
    let delta = wire::decode_delta(&delta_with_raw_record(&raw)).unwrap();
    let record = &delta.set[0];
    assert_eq!(record.kind, RecordKind::Unknown(200));
    assert_eq!((record.features, record.slot, record.object), (5, 42, 7));
    assert_eq!(record.body, Body::Opaque(vec![0xff; 11]));
    // And it is written back as it came.
    assert_eq!(wire::encode_record(record), raw);
}

#[test]
fn trailing_fields_of_a_newer_minor_are_skipped() {
    let mut record = Record::new(RecordKind::Path, 3, 4);
    record.opacity = 0.25;
    let mut raw = wire::encode_record(&record);
    raw.extend_from_slice(&[1, 2, 3, 4, 5]);
    let delta = wire::decode_delta(&delta_with_raw_record(&raw)).unwrap();
    assert_eq!(delta.set[0], record);
}

#[test]
fn another_major_is_refused() {
    let mut delta = Delta::new(Time { num: 0, den: 1 }, 0);
    delta.version = (XF_VERSION.0 + 1, 0);
    let err = wire::decode_delta(&wire::encode_delta(&delta)).unwrap_err();
    assert!(matches!(err, XfError::Version { .. }), "{err:?}");
}

#[test]
fn a_newer_minor_is_read() {
    let mut delta = Delta::new(Time { num: 0, den: 1 }, 0);
    delta.version = (XF_VERSION.0, XF_VERSION.1 + 1);
    let back = wire::decode_delta(&wire::encode_delta(&delta)).unwrap();
    assert_eq!(back.version, delta.version);
}

#[test]
fn a_key_that_does_not_match_its_shape_is_refused() {
    let mut delta = Delta::new(Time { num: 0, den: 1 }, 0);
    let mut shape = Shape::new(ShapeData::Path {
        points: common::square_points(1.0),
    });
    shape.key.0[0] ^= 1;
    delta.upload = vec![shape];
    let err = wire::decode_delta(&wire::encode_delta(&delta)).unwrap_err();
    assert!(matches!(err, XfError::KeyMismatch { .. }), "{err:?}");
}

#[test]
fn truncated_and_foreign_bytes_are_errors_not_panics() {
    let bytes = wire::encode_file(&common::sample_deltas());
    for cut in [0, 3, 7, 12, bytes.len() / 2, bytes.len() - 1] {
        assert!(wire::decode_file(&bytes[..cut]).is_err(), "cut at {cut}");
    }
    let mut foreign = bytes.clone();
    foreign[0] = b'Z';
    assert!(matches!(
        wire::decode_file(&foreign),
        Err(XfError::BadMagic)
    ));
    let one = wire::encode_delta(&common::sample_deltas()[0]);
    for cut in (0..one.len()).step_by(97) {
        assert!(
            wire::decode_delta(&one[..cut]).is_err(),
            "delta cut at {cut}"
        );
    }
}

#[test]
fn a_zero_denominator_is_refused() {
    let delta = Delta::new(Time { num: 1, den: 0 }, 0);
    assert!(wire::decode_delta(&wire::encode_delta(&delta)).is_err());
}
