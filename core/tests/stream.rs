//! The Stream: applying deltas atomically, and drawing what this build draws (the background here;
//! meshes, `tests/mesh.rs`; paths, points and instances, `tests/path.rs`) while refusing every
//! record it cannot draw yet, whole.

mod common;

use arachne_core::stream::{DrawError, Stream, StreamError};
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

fn path_record(slot: u32, shape: &Shape) -> Record {
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(shape.key());
    r.paint.fill = Paint::One([1.0, 0.0, 0.0, 1.0]);
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
fn the_background_is_ces_bytes() {
    let mut s = Stream::new();
    let mut d = delta(0);
    d.view = Some(common::flat_view(8, 4, [0.5, 0.25, 1.0, 1.0]));
    s.apply(&d).unwrap();
    let frame = s.draw(8, 4, 1).unwrap();
    assert_eq!(frame.len(), 8 * 4 * 4);
    // CE: (rgba * 255).astype(int), stored as uint8 RGBA, not premultiplied.
    for px in frame.chunks(4) {
        assert_eq!(px, [127, 63, 255, 255]);
    }
    let mut d = delta(1);
    d.view = Some(common::flat_view(2, 2, [0.2, 0.4, 0.6, 0.3]));
    s.apply(&d).unwrap();
    assert_eq!(s.draw(2, 2, 1).unwrap()[..4], [51, 102, 153, 76]);
}

#[test]
fn a_background_image_is_cropped_as_ce_crops_it() {
    let rgba: Vec<u8> = (0..3 * 2 * 4).map(|i| i as u8).collect();
    let picture = Shape::new(ShapeData::Image {
        width: 3,
        height: 2,
        rgba: rgba.clone(),
    });
    let mut d = delta(0);
    d.view = Some(View {
        background: Background::Image(picture.key()),
        ..common::flat_view(2, 1, [0.0; 4])
    });
    d.upload = vec![picture];
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    assert_eq!(s.draw(2, 1, 1).unwrap(), rgba[..8].to_vec());
    assert!(matches!(s.draw(4, 1, 1), Err(DrawError::Invalid(_))));
}

#[test]
fn drawing_needs_a_view_and_a_size() {
    let s = Stream::new();
    assert!(matches!(s.draw(4, 4, 1), Err(DrawError::NoView)));
    let mut s = Stream::new();
    let mut d = delta(0);
    d.view = Some(common::flat_view(4, 4, [0.0, 0.0, 0.0, 1.0]));
    s.apply(&d).unwrap();
    assert!(matches!(s.draw(0, 4, 1), Err(DrawError::Invalid(_))));
    assert!(matches!(s.draw(4, 4, 0), Err(DrawError::Invalid(_))));
}

#[test]
fn every_record_this_build_cannot_draw_is_refused_whole() {
    // Kinds with no drawer yet are refused, and the drawn kinds are refused for what they carry
    // that this build does not draw: the sample's path a blend, its points a reveal, its mesh a
    // blend. Its instances record draws.
    let mut s = Stream::new();
    let deltas = common::sample_deltas();
    s.apply(&deltas[0]).unwrap();
    // The sample's view carries a light model this build does not know.
    let err = s.draw(64, 36, 1).unwrap_err();
    let DrawError::Unsupported(u) = err else {
        panic!("{err:?}")
    };
    assert_eq!((u.kind.as_str(), u.slot, u.object), ("view", None, None));

    let mut d = delta(1);
    d.view = Some(common::three_d_view(64, 36));
    s.apply(&d).unwrap();
    for (slot, kind) in [
        (1, "path"),
        (2, "face_group"),
        (3, "points"),
        (4, "image"),
        (5, "image"),
        (6, "instances"),
        (7, "mesh"),
        (8, "volume"),
        (9, "field"),
        (10, "emission"),
    ] {
        let mut d = delta(2);
        d.order = Some(vec![OrderEntry {
            layer: Layer::Static,
            slot,
            part: 0,
        }]);
        s.apply(&d).unwrap();
        match s.draw(64, 36, 1) {
            Ok(_) if kind == "instances" => {}
            Err(DrawError::Unsupported(u)) if kind != "instances" => {
                assert_eq!(u.kind, kind);
                assert_eq!(u.slot, Some(slot));
                assert_eq!(u.object, Some(100 + slot));
                let carried = match kind {
                    "path" | "mesh" => "blend",
                    "points" => "reveal",
                    _ => "",
                };
                assert!(u.reason.contains(carried), "slot {slot}: {}", u.reason);
            }
            other => panic!("slot {slot}: {:?}", other.map(|f| f.len())),
        }
    }
}

#[test]
fn an_unknown_feature_bit_is_refused() {
    let shape = square();
    let mut record = path_record(1, &shape);
    record.features = 1 << 40;
    let mut d = delta(0);
    d.view = Some(common::flat_view(4, 4, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![shape];
    d.set = vec![record];
    d.order = Some(vec![entry(1)]);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    let Err(DrawError::Unsupported(u)) = s.draw(4, 4, 1) else {
        panic!("not refused")
    };
    assert!(u.reason.contains("feature"), "{}", u.reason);
}

#[test]
fn a_record_outside_the_order_is_not_drawn_so_not_refused() {
    let mut s = Stream::new();
    s.apply(&common::sample_deltas()[0]).unwrap();
    let mut d = delta(1);
    d.view = Some(common::flat_view(4, 4, [0.0, 0.0, 0.0, 1.0]));
    d.order = Some(Vec::new());
    s.apply(&d).unwrap();
    assert_eq!(s.draw(4, 4, 1).unwrap()[..4], [0, 0, 0, 255]);
}

#[test]
fn apply_reports_what_it_changed_and_a_held_delta_changes_nothing() {
    let mut s = Stream::new();
    let stats = s.apply(&common::sample_deltas()[0]).unwrap();
    assert_eq!((stats.uploaded_shapes, stats.changed_records), (8, 10));
    let before = s.clone();
    let held = delta(1);
    assert!(held.is_empty());
    let stats = s.apply(&held).unwrap();
    assert_eq!((stats.uploaded_shapes, stats.changed_records), (0, 0));
    assert_eq!(s, before);
    let stats = s.apply(&common::sample_deltas()[1]).unwrap();
    assert_eq!((stats.uploaded_shapes, stats.changed_records), (0, 2));
    assert!(s.record(7).is_none() && s.record(8).is_none());
    let stats = s.apply(&common::sample_deltas()[2]).unwrap();
    assert_eq!((stats.uploaded_shapes, stats.changed_records), (0, 1));
    assert!(matches!(&s.record(2).unwrap().body, Body::FaceGroup(f) if f.stroke.is_none()));
    assert!(s.order().is_empty());
}

/// A delta that fails validation leaves the stream as it was.
fn refused(s: &mut Stream, d: &Delta) -> StreamError {
    let before = s.clone();
    let err = s.apply(d).unwrap_err();
    assert_eq!(*s, before, "a refused delta changed the stream ({err})");
    err
}

#[test]
fn deltas_are_validated_before_anything_changes() {
    let shape = square();
    let mut s = Stream::new();
    let mut d = delta(0);
    d.view = Some(common::flat_view(4, 4, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![shape.clone()];
    d.set = vec![path_record(1, &shape)];
    d.order = Some(vec![entry(1)]);
    s.apply(&d).unwrap();

    // A record whose shape was never uploaded.
    let other = Shape::new(ShapeData::Path {
        points: common::square_points(3.0),
    });
    let mut d = delta(1);
    d.set = vec![path_record(2, &other)];
    assert!(matches!(refused(&mut s, &d), StreamError::Missing(_)));

    // A paint table that was never uploaded.
    let mut d = delta(1);
    let mut r = path_record(2, &shape);
    r.paint.stroke = Paint::Table(Stops::new(vec![[0.0; 4]; 3]).key());
    d.set = vec![r];
    assert!(matches!(refused(&mut s, &d), StreamError::Missing(_)));

    // Releasing a shape a record still uses, even with an unrelated upload beside it.
    let mut d = delta(1);
    d.upload = vec![other.clone()];
    d.release = vec![shape.key()];
    assert!(matches!(refused(&mut s, &d), StreamError::InUse(_)));

    // An order entry naming a slot nobody set, or a part a path does not have.
    let mut d = delta(1);
    d.order = Some(vec![entry(1), entry(9)]);
    assert!(matches!(refused(&mut s, &d), StreamError::Missing(_)));
    let mut d = delta(1);
    d.order = Some(vec![OrderEntry {
        layer: Layer::Static,
        slot: 1,
        part: 3,
    }]);
    assert!(matches!(refused(&mut s, &d), StreamError::Invalid(_)));

    // Removing the record the order still draws.
    let mut d = delta(1);
    d.remove = vec![1];
    assert!(matches!(refused(&mut s, &d), StreamError::Missing(_)));

    // A shape of the wrong kind for its record.
    let mut d = delta(1);
    let mut r = Record::new(RecordKind::Mesh, 3, 103);
    r.geometry.shape_a = Some(shape.key());
    d.set = vec![r];
    assert!(matches!(refused(&mut s, &d), StreamError::Invalid(_)));

    // Another major version.
    let mut d = delta(1);
    d.version = (XF_VERSION.0 + 1, 0);
    assert!(matches!(refused(&mut s, &d), StreamError::Version(_)));

    // Releasing what is unused, removing and reordering in one delta is fine.
    let mut d = delta(2);
    d.remove = vec![1];
    d.order = Some(Vec::new());
    d.release = vec![shape.key()];
    let stats = s.apply(&d).unwrap();
    assert_eq!(stats.changed_records, 1);
    assert!(s.record(1).is_none());
}

#[test]
fn a_face_group_is_addressed_by_face() {
    let mut s = Stream::new();
    s.apply(&common::sample_deltas()[0]).unwrap();
    // The sample's face group has two faces: part 1 is valid, part 2 is not.
    let mut d = delta(1);
    d.order = Some(vec![OrderEntry {
        layer: Layer::Static,
        slot: 2,
        part: 2,
    }]);
    assert!(matches!(refused(&mut s, &d), StreamError::Invalid(_)));
}

#[test]
fn a_slot_a_display_still_draws_cannot_be_removed() {
    let mut s = Stream::new();
    s.apply(&common::sample_deltas()[0]).unwrap();
    // Slot 5 is a display whose sub-frame draws slot 1.
    let mut d = delta(1);
    d.remove = vec![1];
    d.order = Some(Vec::new());
    assert!(matches!(refused(&mut s, &d), StreamError::Missing(_)));
    d.remove = vec![1, 5];
    s.apply(&d).unwrap();
}

/// A display at `slot` whose sub-frame draws `order`.
fn display(slot: u32, order: Vec<OrderEntry>) -> Record {
    let mut r = Record::new(RecordKind::Image, slot, 100 + slot);
    r.body = Body::Image(ImageSource::SubFrame {
        view: Box::new(common::flat_view(8, 4, [0.0, 0.0, 0.0, 1.0])),
        order,
    });
    r
}

#[test]
fn re_setting_a_slot_cannot_strand_a_sub_frame() {
    let mut s = Stream::new();
    s.apply(&common::sample_deltas()[0]).unwrap();
    // A display drawing the face group's second face (slot 2, part 1), which only it draws.
    let mut d = delta(1);
    d.order = Some(Vec::new());
    d.set = vec![display(
        12,
        vec![OrderEntry {
            layer: Layer::Static,
            slot: 2,
            part: 1,
        }],
    )];
    s.apply(&d).unwrap();
    // Slot 2 re-set as a path, which has one part: the display would draw a part that is gone.
    let shape = square();
    let mut d = delta(2);
    d.upload = vec![shape.clone()];
    d.set = vec![path_record(2, &shape)];
    assert!(matches!(refused(&mut s, &d), StreamError::Invalid(_)));
}

#[test]
fn a_display_cannot_draw_itself() {
    let mut s = Stream::new();
    s.apply(&common::sample_deltas()[0]).unwrap();
    let mut d = delta(1);
    d.set = vec![display(
        12,
        vec![OrderEntry {
            layer: Layer::Static,
            slot: 12,
            part: 0,
        }],
    )];
    assert!(matches!(refused(&mut s, &d), StreamError::Invalid(_)));
}
