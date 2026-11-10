//! Where each of XF's checks is made (XF.md, "Where each check is made"): each check no other test
//! pins, refused by the reader (`wire::decode_delta`) or by `apply`, where that list places it.

mod common;

use arachne_core::stream::{Stream, StreamError};
use arachne_core::xf::wire::{self, XfError};
use arachne_core::xf::*;

fn delta(seq: u64) -> Delta {
    Delta::new(
        Time {
            num: seq as i64,
            den: 15,
        },
        seq,
    )
}

fn square() -> Shape {
    Shape::new(ShapeData::Path {
        points: common::square_points(1.0),
    })
}

/// A key nothing holds.
const NOWHERE: Key = Key([7; 16]);

fn drawn(slot: u32, shape: &Shape) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, slot);
    r.geometry.shape_a = Some(shape.key());
    r.paint.fill = Paint::One([1.0; 4]);
    r
}

/// A stream holding a square, drawn in slot 1; and the square.
fn holding() -> (Stream, Shape) {
    let shape = square();
    let mut d = delta(0);
    d.view = Some(common::flat_view(8, 8, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![shape.clone()];
    d.set = vec![drawn(1, &shape)];
    d.order = Some(vec![OrderEntry {
        layer: Layer::Moving,
        slot: 1,
        part: 0,
    }]);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    (s, shape)
}

/// `a`'s bytes with the byte `back` before the first where they differ from `b`'s set to `to`.
fn patched(a: &Delta, b: &Delta, back: usize, to: u8) -> Vec<u8> {
    let (mut x, y) = (wire::encode_delta(a), wire::encode_delta(b));
    let at = x
        .iter()
        .zip(&y)
        .position(|(p, q)| p != q)
        .expect("the deltas differ");
    x[at - back] = to;
    x
}

fn refused_at_decode(bytes: &[u8], what: &str) -> XfError {
    match wire::decode_delta(bytes) {
        Err(err) => {
            assert!(err.to_string().contains(what), "{what}: {err}");
            err
        }
        Ok(_) => panic!("{what}: read"),
    }
}

fn invalid_at_decode(bytes: &[u8], what: &str) {
    let err = refused_at_decode(bytes, what);
    assert!(matches!(err, XfError::Invalid(_)), "{what}: {err}");
}

/// A delta setting `rec`.
fn setting(rec: Record) -> Delta {
    let mut d = delta(0);
    d.set = vec![rec];
    d
}

#[test]
fn the_readers_checks_are_made_at_decode() {
    // A bool byte other than 0 or 1: the delta's view-present flag.
    let mut viewed = delta(0);
    viewed.view = Some(common::flat_view(8, 8, [0.0; 4]));
    invalid_at_decode(&patched(&delta(0), &viewed, 0, 2), "2 is not a bool");
    // A negative time denominator.
    let mut bytes = wire::encode_delta(&delta(0));
    bytes[12..20].copy_from_slice(&(-1i64).to_le_bytes());
    invalid_at_decode(&bytes, "denominator -1");

    // The view's background and camera codes.
    let at = |background: [f64; 4], camera: Camera| {
        let mut d = delta(0);
        let mut view = common::flat_view(8, 8, background);
        view.camera = camera;
        d.view = Some(view);
        d
    };
    let tinted = [f64::from_bits(1), 0.0, 0.0, 1.0];
    let flat = at([0.0, 0.0, 0.0, 1.0], Camera::Flat);
    let background = patched(&flat, &at(tinted, Camera::Flat), 1, 9);
    invalid_at_decode(&background, "9 is not a valid background");
    let camera = patched(&flat, &at([0.0, 0.0, 0.0, 1.0], Camera::Moving), 0, 9);
    invalid_at_decode(&camera, "9 is not a valid camera");

    // An uploaded image whose size overflows.
    let picture = Shape::new(ShapeData::Image {
        width: 1,
        height: 1,
        rgba: vec![0; 4],
    });
    let mut d = delta(0);
    d.upload = vec![picture.clone()];
    let mut bytes = wire::encode_delta(&d);
    let key = bytes
        .windows(16)
        .position(|w| w == picture.key().0)
        .expect("the key is in the bytes");
    bytes[key + 16..key + 24].copy_from_slice(&[0xff; 8]);
    invalid_at_decode(&bytes, "an image's size overflows");

    // A paint table under a key other than its rows'.
    let stops = Stops::new(vec![[1.0; 4]]);
    let mut d = delta(0);
    d.upload_paints = vec![stops.clone()];
    let mut bytes = wire::encode_delta(&d);
    common::restate_key(&mut bytes, stops.key(), NOWHERE);
    let err = refused_at_decode(&bytes, "");
    assert!(matches!(err, XfError::KeyMismatch { .. }), "{err}");

    // A record's paint, joint, cap, space and image-source codes.
    let shape = square();
    let record = |edit: &dyn Fn(&mut Record)| {
        let mut r = drawn(2, &shape);
        edit(&mut r);
        setting(r)
    };
    let painted = record(&|r| {
        r.paint.fill = Paint::None;
        r.paint.stroke = Paint::One([1.0; 4]);
    });
    let other = record(&|r| {
        r.paint.fill = Paint::One([1.0; 4]);
        r.paint.stroke = Paint::None;
    });
    invalid_at_decode(&patched(&painted, &other, 0, 9), "9 is not a valid paint");
    let plain = record(&|_| {});
    let joint = record(&|r| r.paint.joint = Joint::Round);
    invalid_at_decode(&patched(&plain, &joint, 0, 9), "9 is not a valid joint");
    let cap = record(&|r| r.paint.cap = Cap::Round);
    invalid_at_decode(&patched(&plain, &cap, 0, 9), "9 is not a valid cap");
    let screen = record(&|r| r.depth.space = Space::Screen);
    let world = record(&|r| r.depth.space = Space::World);
    invalid_at_decode(&patched(&screen, &world, 0, 9), "9 is not a valid space");
    let image = |resampling: u8| {
        let mut r = Record::new(RecordKind::Image, 2, 2);
        r.geometry.shape_a = Some(picture.key());
        r.body = Body::Image(ImageSource::Picture {
            corners: [[0.0; 3]; 4],
            resampling,
            opacity: 1.0,
        });
        setting(r)
    };
    // The source code is 97 bytes before the resampling byte: the code, then four corners.
    let source = patched(&image(0), &image(1), 97, 9);
    invalid_at_decode(&source, "9 is not a valid image source");
}

/// `apply`'s refusal of `d`, which leaves the stream as it was.
fn refused(s: &mut Stream, d: &Delta) -> StreamError {
    let before = s.clone();
    let err = s.apply(d).unwrap_err();
    assert_eq!(*s, before, "a refused delta changed the stream ({err})");
    err
}

/// That `d` is read from its bytes, and refused by `apply` as `what`.
fn refused_by_apply(s: &mut Stream, d: &Delta, what: &str) -> StreamError {
    let read = wire::decode_delta(&wire::encode_delta(d)).unwrap();
    let err = refused(s, &read);
    assert!(err.to_string().contains(what), "{what}: {err}");
    err
}

/// A delta uploading an unknown shape under `key`, read from its bytes.
fn unknown_under(key: Key, mut d: Delta) -> Delta {
    let unknown = Shape::new(ShapeData::Reserved {
        code: 99,
        payload: vec![1, 2, 3],
    });
    d.upload.insert(0, unknown.clone());
    let mut bytes = wire::encode_delta(&d);
    common::restate_key(&mut bytes, unknown.key(), key);
    wire::decode_delta(&bytes).unwrap()
}

#[test]
fn apply_s_checks_are_made_by_apply() {
    let (mut s, shape) = holding();
    let missing = |err: StreamError| matches!(err, StreamError::Missing(_));
    let invalid = |err: StreamError| matches!(err, StreamError::Invalid(_));

    // A shape uploaded under a key the stream holds as a paint table, and the other way round.
    let stops = Stops::new(vec![[0.5; 4]]);
    let mut tabled = delta(1);
    tabled.upload_paints = vec![stops.clone()];
    s.apply(&tabled).unwrap();
    let err = refused(&mut s, &unknown_under(stops.key(), delta(2)));
    assert!(invalid(err), "a shape under a table's key");
    let other = Stops::new(vec![[0.25; 4]]);
    s.apply(&unknown_under(other.key(), delta(2))).unwrap();
    let mut d = delta(3);
    d.upload_paints = vec![other];
    let err = refused_by_apply(&mut s, &d, "holds with other content");
    assert!(invalid(err), "a table under a shape's key");

    // A release of a key, and a removal of a slot, that are not there.
    let mut d = delta(3);
    d.release = vec![NOWHERE];
    assert!(missing(refused_by_apply(&mut s, &d, "a release of")));
    let mut d = delta(3);
    d.remove = vec![42];
    assert!(missing(refused_by_apply(&mut s, &d, "a removal of slot 42")));

    // A record's blend and clip naming a shape the stream would not hold.
    let mut blended = drawn(2, &shape);
    blended.geometry.blend = Some(Blend {
        shape_b: NOWHERE,
        place_b: IDENTITY,
        t: 0.5,
    });
    let mut clipped = drawn(2, &shape);
    clipped.clip = Some((NOWHERE, IDENTITY));
    for rec in [blended, clipped] {
        let err = refused_by_apply(&mut s, &setting(rec), "the stream would not hold");
        assert!(missing(err));
    }

    // A table of other than one row per point.
    let one_row = Stops::new(vec![[1.0; 4]]);
    let mut points = Record::new(RecordKind::Points, 2, 2);
    points.geometry.shape_a = Some(shape.key());
    points.body = Body::Points(Points {
        colors: one_row.key(),
        radius: 1.0,
    });
    let mut d = setting(points);
    d.upload_paints = vec![one_row.clone()];
    assert!(invalid(refused_by_apply(&mut s, &d, "a table of 1 rows")));

    // The view's background image: a shape that is not an image, and one not there.
    let mut d = delta(3);
    d.view = Some(View {
        background: Background::Image(shape.key()),
        ..common::flat_view(8, 8, [0.0; 4])
    });
    assert!(invalid(refused_by_apply(&mut s, &d, "must be an image shape")));
    let mut d = delta(3);
    d.view = Some(View {
        background: Background::Image(NOWHERE),
        ..common::flat_view(8, 8, [0.0; 4])
    });
    assert!(missing(refused_by_apply(&mut s, &d, "background image")));
}

#[test]
fn apply_s_checks_of_what_the_reader_never_hands_it_are_made_by_apply() {
    // Deltas built in memory, as the reader never reads them: a feature bit and its field
    // disagreeing (the reader's own check, again), a tail on a mesh, a body of another kind's,
    // and an instances record whose placements and objects differ in number (on the wire they
    // share one count).
    let (mut s, shape) = holding();
    let mut unfilled = drawn(2, &shape);
    unfilled.features |= FEATURE_BACKGROUND_IMAGE;
    let mut tailless = drawn(2, &shape);
    tailless.features |= FEATURE_PRE_PROJECTED;
    let mut mesh = Record::new(RecordKind::Mesh, 2, 2);
    mesh.features = FEATURE_PRE_PROJECTED;
    mesh.pre_projected = Some(PreProjected {
        points: shape.key(),
        gradient: None,
        fill: Paint::None,
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let mut bodied = drawn(2, &shape);
    bodied.body = Body::Points(Points {
        colors: NOWHERE,
        radius: 1.0,
    });
    let one_row = Stops::new(vec![[1.0; 4]]);
    let mut instances = Record::new(RecordKind::Instances, 2, 2);
    instances.geometry.shape_a = Some(shape.key());
    instances.body = Body::Instances(Instances {
        placements: vec![IDENTITY],
        fill: one_row.key(),
        stroke: one_row.key(),
        objects: Vec::new(),
    });
    for (rec, what) in [
        (unfilled, "background_image feature bit and fill_image field disagree"),
        (tailless, "pre_projected feature bit and tail field disagree"),
        (mesh, "pre_projected is not valid on mesh records"),
        (bodied, "with another kind's body"),
        (instances, "1 placements and 0 objects"),
    ] {
        let mut d = setting(rec);
        d.upload_paints = vec![one_row.clone()];
        let err = refused(&mut s, &d);
        assert!(
            matches!(&err, StreamError::Invalid(why) if why.contains(what)),
            "{what}: {err}"
        );
    }
}
