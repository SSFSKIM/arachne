//! XF's binary form: round trips, byte stability, and refusing what a reader does not know without
//! failing to parse.

mod common;

use arachne_core::xf::wire::{self, XfError};
use arachne_core::xf::*;
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

/// SHA-256 (FIPS 180-4), for pinning the sample's bytes without a dependency.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in message.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            (hh, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
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
    let mut raw = wire::encode_record(&record);
    raw.extend_from_slice(&[1, 2, 3, 4, 5]);
    let delta = wire::decode_delta(&delta_with_raw_record(&raw)).unwrap();
    assert_eq!(delta.set[0], record);
}

/// XF 1.2's normals follow the planes, both written when either is carried; an empty list reads
/// as one not carried. A newer minor's bytes after them are skipped.
#[test]
fn a_records_normals_follow_its_planes() {
    let mut record = Record::new(RecordKind::Path, 3, 4);
    record.normals = Some(vec![[0.0, 0.6, 0.8], [0.0; 3]]);
    let mut raw = wire::encode_record(&record);
    raw.extend_from_slice(&[1, 2, 3]);
    let delta = wire::decode_delta(&delta_with_raw_record(&raw)).unwrap();
    assert_eq!(delta.set[0], record);
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
    // XF 1.2's planes and normals (none) follow the tail; a newer minor's bytes follow them.
    let mut planned = raw.clone();
    planned.extend_from_slice(&[1, 0, 0, 0, 1, 0, 0, 0, 0]);
    let mut expected_planned = expected.clone();
    expected_planned.planes = Some(vec![1]);
    let mut extended = planned.clone();
    extended.extend_from_slice(&[0xff, 0xfe, 0xfd]);
    let mut bytes = delta_with_raw_record(&extended);
    bytes[2..4].copy_from_slice(&3u16.to_le_bytes());
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
    assert_eq!(&wire::encode_file(&[d, gridded])[4..8], &[1, 0, 2, 0]);
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
