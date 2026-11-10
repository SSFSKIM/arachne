//! XF's binary form: round trips, byte stability, and refusing what a reader does not know without
//! failing to parse.

mod common;

use arachne_core::xf::wire::{self, XfError};
use arachne_core::xf::*;
use common::sha256_hex;

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

#[test]
fn sha256_matches_its_standard_vectors() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

/// The sha256 of the canonical sample's bytes (`common::sample_deltas`: every enum code, both
/// branches of every optional field, swappable fields given values of their own). It fails on any
/// change to what the sample writes; `tests/layout.rs` pins the layout field by field. XF streams
/// are generated, never committed. Change the pin only when such a change is meant.
const SAMPLE_SHA256: &str = "87821af1401a64a279e12046d9e31c05c8f7232d1fb2a4a29ae4254ba4c27dc0";

#[test]
fn the_canonical_samples_bytes_are_pinned() {
    let bytes = wire::encode_file(&common::sample_deltas());
    assert_eq!(
        sha256_hex(&bytes),
        SAMPLE_SHA256,
        "the canonical sample's bytes changed"
    );
    assert_eq!(wire::decode_file(&bytes).unwrap(), common::sample_deltas());
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
    // Header: kind 200, features, slot 42, object 7; then bytes no reader of XF 1 can parse.
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
    record.planes = Some(vec![1]);
    // The fields of XF 1.4, the last this build knows: a newer minor's bytes follow them.
    record.joins = Some(vec![1, 3]);
    let mut raw = wire::encode_record(&record);
    raw.extend_from_slice(&[1, 2, 3, 4, 5]);
    let delta = wire::decode_delta(&delta_with_raw_record(&raw)).unwrap();
    assert_eq!(delta.set[0], record);
}

/// XF 1.2's normals follow the planes, both written when either is carried; an empty list reads
/// as one not carried. A newer minor's bytes after them (and after XF 1.4's joins) are skipped.
#[test]
fn a_records_normals_follow_its_planes() {
    let mut record = Record::new(RecordKind::Path, 3, 4);
    record.normals = Some(vec![[0.0, 0.6, 0.8], [0.0; 3]]);
    record.joins = Some(vec![]);
    let mut raw = wire::encode_record(&record);
    raw.extend_from_slice(&[1, 2, 3]);
    let delta = wire::decode_delta(&delta_with_raw_record(&raw)).unwrap();
    assert_eq!(delta.set[0], record);
    record.joins = None;
    record.planes = Some(vec![1]);
    let delta = wire::decode_delta(&delta_with_raw_record(&wire::encode_record(&record))).unwrap();
    assert_eq!(delta.set[0], record);
    let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
    record.planes = None;
    d.set = vec![record];
    assert_eq!(&wire::encode_delta(&d)[..4], &[1, 0, 2, 0]);
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
    let shape = Shape::new(ShapeData::Path {
        points: common::square_points(1.0),
    });
    let mut other = shape.key();
    other.0[0] ^= 1;
    delta.upload = vec![shape.clone()];
    let mut bytes = wire::encode_delta(&delta);
    common::restate_key(&mut bytes, shape.key(), other);
    let err = wire::decode_delta(&bytes).unwrap_err();
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

#[test]
fn a_background_image_bit_without_its_field_is_invalid() {
    let mut record = Record::new(RecordKind::Path, 3, 4);
    record.features = 1;
    let bytes = delta_with_raw_record(&wire::encode_record(&record));
    assert!(matches!(
        wire::decode_delta(&bytes),
        Err(XfError::Invalid(_))
    ));
}

#[test]
fn a_fill_image_field_without_its_bit_is_invalid() {
    let mut record = Record::new(RecordKind::Path, 3, 4);
    record.fill_image = Some(Key([0xa1; 16]));
    let bytes = delta_with_raw_record(&wire::encode_record(&record));
    assert!(matches!(
        wire::decode_delta(&bytes),
        Err(XfError::Invalid(_))
    ));
}

#[test]
fn the_previous_major_is_refused() {
    let mut delta = Delta::new(Time { num: 0, den: 1 }, 0);
    delta.version = (0, 0);
    assert!(matches!(
        wire::decode_delta(&wire::encode_delta(&delta)),
        Err(XfError::Version { found: (0, 0) })
    ));
    let mut file = wire::encode_file(&[]);
    file[4..6].copy_from_slice(&0u16.to_le_bytes());
    assert!(matches!(
        wire::decode_file(&file),
        Err(XfError::Version { found: (0, 0) })
    ));
}

/// A minimal bit-1 tail, built independently of the encoder: key, no gradient, three no paints.
fn raw_pre_projected(mut record: Vec<u8>) -> Vec<u8> {
    record[1..9].copy_from_slice(&2u64.to_le_bytes());
    record.extend_from_slice(&[0xa1; 16]);
    record.extend_from_slice(&[0, 0, 0, 0]);
    record
}

#[test]
fn a_pre_projected_tail_round_trips_instead_of_being_skipped() {
    let raw = raw_pre_projected(wire::encode_record(&Record::new(RecordKind::Path, 3, 4)));
    let mut bytes = delta_with_raw_record(&raw);
    bytes[2..4].copy_from_slice(&1u16.to_le_bytes());
    let read = wire::decode_delta(&bytes).unwrap();
    assert_eq!(wire::encode_record(&read.set[0]), raw);
}

#[test]
fn pre_projected_is_invalid_on_face_groups() {
    let mut record = Record::new(RecordKind::FaceGroup, 3, 4);
    record.body = Body::FaceGroup(FaceGroup {
        uv: vec![],
        grid: [0, 0],
        fill: Key([0xb2; 16]),
        stroke: None,
        face_ends: Vec::new(),
        face_widths: Vec::new(),
    });
    let raw = raw_pre_projected(wire::encode_record(&record));
    assert!(matches!(
        wire::decode_delta(&delta_with_raw_record(&raw)),
        Err(XfError::Invalid(_))
    ));
}

#[test]
fn pre_projected_is_invalid_on_meshes() {
    let raw = raw_pre_projected(wire::encode_record(&Record::new(RecordKind::Mesh, 3, 4)));
    assert!(matches!(
        wire::decode_delta(&delta_with_raw_record(&raw)),
        Err(XfError::Invalid(_))
    ));
}

#[test]
fn a_pre_projected_bit_requires_a_complete_tail() {
    let raw = raw_pre_projected(wire::encode_record(&Record::new(RecordKind::Path, 3, 4)));
    for cut in raw.len() - 20..raw.len() {
        assert!(
            matches!(
                wire::decode_delta(&delta_with_raw_record(&raw[..cut])),
                Err(XfError::Truncated { .. })
            ),
            "cut {cut}"
        );
    }
}

#[test]
fn tails_round_trip_on_each_allowed_kind() {
    let bodies = [
        (RecordKind::Path, Body::None),
        (
            RecordKind::Points,
            Body::Points(Points {
                colors: Key([0xb2; 16]),
                radius: 2.5,
            }),
        ),
        (
            RecordKind::Image,
            Body::Image(ImageSource::Picture {
                corners: [[0.0; 3]; 4],
                resampling: 3,
                opacity: 0.75,
            }),
        ),
        (
            RecordKind::Image,
            Body::Image(ImageSource::SubFrame {
                rank: 0,
                corners: [[0.0; 3]; 4],
                resampling: 2,
                opacity: 0.5,
                view: Box::new(common::flat_view(4, 4, [0.0; 4])),
                order: vec![],
            }),
        ),
    ];
    for (kind, body) in bodies {
        let mut record = Record::new(kind, 3, 4);
        record.body = body;
        let raw = raw_pre_projected(wire::encode_record(&record));
        let read = wire::decode_delta(&delta_with_raw_record(&raw)).unwrap();
        assert_eq!(wire::encode_record(&read.set[0]), raw);
    }
}

#[test]
fn bytes_after_a_tail_are_still_skipped_and_do_not_shift_the_next_record() {
    let raw = raw_pre_projected(wire::encode_record(&Record::new(RecordKind::Path, 3, 4)));
    let expected = wire::decode_delta(&delta_with_raw_record(&raw))
        .unwrap()
        .set
        .remove(0);
    // XF 1.2's planes and normals (none) follow the tail, then XF 1.4's joins; a newer minor's
    // bytes follow them.
    let mut planned = raw.clone();
    planned.extend_from_slice(&[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 3]);
    let mut expected_planned = expected.clone();
    expected_planned.planes = Some(vec![1]);
    expected_planned.joins = Some(vec![3]);
    let mut extended = planned.clone();
    extended.extend_from_slice(&[0xff, 0xfe, 0xfd]);
    let mut bytes = delta_with_raw_record(&extended);
    bytes[2..4].copy_from_slice(&5u16.to_le_bytes());
    assert_eq!(wire::decode_delta(&bytes).unwrap().set[0], expected_planned);

    let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
    d.set = vec![expected_planned, Record::new(RecordKind::Path, 9, 10)];
    let encoded = wire::encode_delta(&d);
    // First item at 53; extend it without touching the second record or the delta's suffix.
    let mut appended = encoded[..49].to_vec();
    appended.extend_from_slice(&((planned.len() + 3) as u32).to_le_bytes());
    appended.extend_from_slice(&planned);
    appended.extend_from_slice(&[0xff, 0xfe, 0xfd]);
    appended.extend_from_slice(&encoded[53 + planned.len()..]);
    assert_eq!(wire::decode_delta(&appended).unwrap().set, d.set);
}

#[test]
fn the_writer_uses_the_lowest_minor_that_carries_the_content() {
    let mut d = Delta::new(Time { num: 0, den: 1 }, 0);
    d.version = (1, 1);
    assert_eq!(&wire::encode_delta(&d)[..4], &[1, 0, 0, 0]);
    assert_eq!(&wire::encode_file(&[d.clone()])[4..8], &[1, 0, 0, 0]);
    let raw = raw_pre_projected(wire::encode_record(&Record::new(RecordKind::Path, 3, 4)));
    d.set = wire::decode_delta(&delta_with_raw_record(&raw))
        .unwrap()
        .set;
    d.version = (1, 0);
    assert_eq!(&wire::encode_delta(&d)[..4], &[1, 0, 1, 0]);
    assert_eq!(&wire::encode_file(&[d.clone()])[4..8], &[1, 0, 1, 0]);
    // XF 1.2: a record's planes, or a view's input grid (the frame's or a sub-frame's).
    let mut planned = d.clone();
    planned.set[0].planes = Some(vec![1]);
    assert_eq!(&wire::encode_delta(&planned)[..4], &[1, 0, 2, 0]);
    let mut gridded = Delta::new(Time { num: 0, den: 1 }, 0);
    let mut view = common::flat_view(16, 9, [0.0, 0.0, 0.0, 1.0]);
    view.input_grid = 1.0 / 32768.0;
    gridded.view = Some(view);
    assert_eq!(&wire::encode_delta(&gridded)[..4], &[1, 0, 2, 0]);
    assert_eq!(
        &wire::encode_file(&[d.clone(), gridded])[4..8],
        &[1, 0, 2, 0]
    );
    // XF 1.4: a record's joins, alone or after planes; read back as written.
    for planes in [None, Some(vec![1])] {
        let mut joined = d.clone();
        joined.set[0].planes = planes;
        joined.set[0].joins = Some(vec![0, 1, 3]);
        let bytes = wire::encode_delta(&joined);
        assert_eq!(&bytes[..4], &[1, 0, 4, 0]);
        assert_eq!(wire::decode_delta(&bytes).unwrap().set, joined.set);
    }
}

#[test]
fn geometry_flag_bit_three_is_still_invalid() {
    let mut raw = wire::encode_record(&Record::new(RecordKind::Path, 3, 4));
    raw[17] |= 8;
    assert!(matches!(
        wire::decode_delta(&delta_with_raw_record(&raw)),
        Err(XfError::Invalid(_))
    ));
}

#[test]
fn the_pre_projected_canonical_samples_bytes_are_pinned() {
    let delta = common::pre_projected_sample();
    let bytes = wire::encode_file(std::slice::from_ref(&delta));
    assert_eq!(
        sha256_hex(&bytes),
        "df6b52077f443b5db275ed07837ef7235a089298af0c410c770150ddf796983c"
    );
    assert_eq!(wire::decode_file(&bytes).unwrap(), vec![delta]);
}

#[test]
fn the_snapped_canonical_samples_bytes_are_pinned() {
    let delta = common::snapped_sample();
    let bytes = wire::encode_file(std::slice::from_ref(&delta));
    assert_eq!(&bytes[4..8], &[1, 0, 2, 0]);
    assert_eq!(
        sha256_hex(&bytes),
        "217cef58ae3b74302edc0d74b1740fd16264efb2780b8694f56b24c79585a866"
    );
    assert_eq!(wire::decode_file(&bytes).unwrap(), vec![delta]);
}

#[test]
fn the_face_ends_canonical_samples_bytes_are_pinned() {
    let delta = common::face_ends_sample();
    let bytes = wire::encode_file(std::slice::from_ref(&delta));
    assert_eq!(&bytes[4..8], &[1, 0, 3, 0]);
    assert_eq!(
        sha256_hex(&bytes),
        "7588b900c5535131e9ae39e07f0b5bcdd45769e0714c1e04d6c7a094c67d00f1"
    );
    assert_eq!(wire::decode_file(&bytes).unwrap(), vec![delta]);
}

#[test]
fn a_face_group_without_face_ends_is_written_and_read_at_its_earlier_minor() {
    // The canonical sample's face groups carry neither list: their deltas are written at the
    // minor they were (their bytes pinned, `the_canonical_samples_bytes_are_pinned`), and read
    // back with neither.
    let deltas = common::sample_deltas();
    let mut groups = 0;
    for d in &deltas {
        assert!(wire::encoded_version(d).1 < 3);
        for r in &wire::decode_delta(&wire::encode_delta(d)).unwrap().set {
            if let Body::FaceGroup(f) = &r.body {
                assert!(f.face_ends.is_empty() && f.face_widths.is_empty());
                groups += 1;
            }
        }
    }
    assert_eq!(groups, 2);
}
