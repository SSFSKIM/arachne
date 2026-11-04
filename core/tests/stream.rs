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
                    "path" => "Cairo",
                    "mesh" => "blend",
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
        rank: 0,
        corners: [
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        resampling: 3,
        opacity: 1.0,
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
fn a_display_can_capture_itself() {
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
    s.apply(&d).unwrap();
}

#[test]
fn a_shaded_path_under_a_shading_camera_is_refused_not_drawn_unlit() {
    // CE lights a `shade_in_3d` path under a 3D camera that shades: each paint becomes two stops,
    // lit at the path's start and end corners, drawn as a linear gradient. Paths draw no gradient
    // yet, so such a path, or an instances record of one, refuses the frame instead of drawing
    // unlit. Without shading, or without `shade_in_3d`, it draws.
    let shape = square();
    let colours = Stops::new(vec![[1.0, 0.0, 0.0, 1.0]]);
    for instanced in [false, true] {
        for (shade_in_3d, shading) in [(true, true), (false, true), (true, false)] {
            let mut r = path_record(1, &shape);
            if instanced {
                r.kind = RecordKind::Instances;
                r.body = Body::Instances(Instances {
                    placements: vec![IDENTITY],
                    fill: colours.key(),
                    stroke: colours.key(),
                    objects: vec![7],
                });
            }
            r.depth.space = Space::World;
            r.depth.shade_in_3d = shade_in_3d;
            let mut view = common::three_d_view(16, 9);
            let Camera::ThreeD(c) = &mut view.camera else {
                unreachable!()
            };
            c.should_apply_shading = shading;
            let mut d = delta(0);
            d.view = Some(view);
            d.upload = vec![shape.clone()];
            d.upload_paints = vec![colours.clone()];
            d.set = vec![r];
            d.order = Some(vec![entry(1)]);
            let mut s = Stream::new();
            s.apply(&d).unwrap();
            let drawn = s.draw(16, 9, 1);
            let what =
                format!("instanced {instanced}, shade_in_3d {shade_in_3d}, shading {shading}");
            if shade_in_3d && shading {
                let Err(DrawError::Unsupported(u)) = drawn else {
                    panic!("{what}: not refused")
                };
                let kind = if instanced { "instances" } else { "path" };
                assert_eq!((u.kind.as_str(), u.slot), (kind, Some(1)), "{what}");
                assert!(u.reason.contains("shad"), "{what}: {}", u.reason);
            } else {
                assert!(drawn.is_ok(), "{what}: {drawn:?}");
            }
        }
    }
}

#[test]
fn an_image_whose_pixels_are_not_its_size_is_refused_by_apply() {
    for (width, height, bytes) in [(1, 1, 0), (2, 2, 15), (2, 2, 17), (3, 1, 4)] {
        let picture = Shape::new(ShapeData::Image {
            width,
            height,
            rgba: vec![0; bytes],
        });
        let mut d = delta(0);
        d.view = Some(View {
            background: Background::Image(picture.key()),
            ..common::flat_view(1, 1, [0.0; 4])
        });
        d.upload = vec![picture];
        assert!(
            matches!(Stream::new().apply(&d), Err(StreamError::Invalid(_))),
            "{width} x {height} in {bytes} bytes"
        );
    }
}

#[test]
fn only_the_assignments_a_delta_leaves_are_validated() {
    // As commit applies them: the last record set in a slot wins, then removals. A record set and
    // then replaced, or set and then removed, in the same delta references nothing afterwards.
    let a = square();
    let b = Shape::new(ShapeData::Path {
        points: common::square_points(2.0),
    });
    let start = || {
        let mut s = Stream::new();
        let mut d = delta(0);
        d.view = Some(common::flat_view(4, 4, [0.0, 0.0, 0.0, 1.0]));
        d.upload = vec![a.clone()];
        d.set = vec![path_record(1, &a)];
        d.order = Some(vec![entry(1)]);
        s.apply(&d).unwrap();
        s
    };

    let mut s = start();
    let mut d = delta(1);
    d.upload = vec![b.clone()];
    d.set = vec![path_record(1, &a), path_record(1, &b)];
    d.release = vec![a.key()];
    s.apply(&d).unwrap();
    assert_eq!(s.record(1).unwrap().geometry.shape_a, Some(b.key()));
    assert!(!s.has_shape(&a.key()));

    let mut s = start();
    let mut d = delta(1);
    d.set = vec![path_record(1, &a)];
    d.remove = vec![1];
    d.order = Some(Vec::new());
    d.release = vec![a.key()];
    s.apply(&d).unwrap();
    assert!(s.record(1).is_none() && !s.has_shape(&a.key()));

    // What survives is still checked: a last assignment referencing a released key is refused.
    let mut s = start();
    let mut d = delta(1);
    d.upload = vec![b.clone()];
    d.set = vec![path_record(1, &b), path_record(1, &a)];
    d.release = vec![a.key()];
    assert!(matches!(refused(&mut s, &d), StreamError::InUse(_)));
}

/// `d` with a shape of a kind this build does not know uploaded first, under `key`: another
/// content's key, which only XF's binary form can state.
fn unknown_under(key: Key, mut d: Delta) -> Delta {
    let unknown = Shape::new(ShapeData::Reserved {
        code: 99,
        payload: vec![1, 2, 3],
    });
    d.upload.insert(0, unknown.clone());
    let mut bytes = wire::encode_delta(&d);
    common::restate_key(&mut bytes, unknown.key(), key);
    let read = wire::decode_delta(&bytes).unwrap();
    assert!(matches!(
        read.upload[0].data(),
        ShapeData::Reserved { code: 99, .. }
    ));
    assert_eq!(read.upload[0].key(), key);
    read
}

#[test]
fn one_key_holds_one_content_in_validation_and_in_commit() {
    let picture = Shape::new(ShapeData::Image {
        width: 2,
        height: 2,
        rgba: vec![255; 16],
    });
    let pictured = |seq: u64| {
        let mut d = delta(seq);
        d.view = Some(View {
            background: Background::Image(picture.key()),
            ..common::flat_view(2, 2, [0.0; 4])
        });
        d
    };

    // A delta uploading one key twice is refused, whether the contents agree or not.
    let mut d = pictured(0);
    d.upload = vec![picture.clone(), picture.clone()];
    assert!(matches!(
        Stream::new().apply(&d),
        Err(StreamError::Invalid(_))
    ));
    let mut d = delta(0);
    d.upload_paints = vec![Stops::new(vec![[1.0; 4]]), Stops::new(vec![[1.0; 4]])];
    assert!(matches!(
        Stream::new().apply(&d),
        Err(StreamError::Invalid(_))
    ));
    let mut d = pictured(0);
    d.upload = vec![picture.clone()];
    let d = unknown_under(picture.key(), d);
    assert!(matches!(
        Stream::new().apply(&d),
        Err(StreamError::Invalid(_))
    ));

    // A key the stream holds is never uploaded again with other content: with an unknown kind held
    // under the picture's key, the view cannot be checked against the picture and drawn from the
    // unknown shape.
    let mut s = Stream::new();
    s.apply(&unknown_under(picture.key(), delta(0))).unwrap();
    let mut d = pictured(1);
    d.upload = vec![picture.clone()];
    assert!(matches!(refused(&mut s, &d), StreamError::Invalid(_)));

    // Nor the other way round: the picture held, an unknown kind sent under its key.
    let mut s = Stream::new();
    let mut d = pictured(0);
    d.upload = vec![picture.clone()];
    s.apply(&d).unwrap();
    assert!(matches!(
        refused(&mut s, &unknown_under(picture.key(), delta(1))),
        StreamError::Invalid(_)
    ));

    // The same content again changes nothing.
    let mut d = delta(1);
    d.upload = vec![picture.clone()];
    s.apply(&d).unwrap();
    assert_eq!(s.draw(2, 2, 1).unwrap(), vec![255; 16]);
}

fn fill_image() -> Shape {
    Shape::new(ShapeData::Image {
        width: 4,
        height: 4,
        rgba: vec![255; 64],
    })
}

#[test]
fn a_fill_image_must_name_an_image_held_or_uploaded() {
    let shape = square();
    let picture = fill_image();
    let table = Stops::new(vec![[1.0; 4]]);
    let mut s = Stream::new();
    let mut d = delta(0);
    d.upload = vec![shape.clone()];
    d.upload_paints = vec![table.clone()];
    s.apply(&d).unwrap();
    let mut r = path_record(1, &shape);
    r.features = FEATURE_BACKGROUND_IMAGE;
    r.fill_image = Some(picture.key());
    let mut d = delta(1);
    d.set = vec![r.clone()];
    assert!(matches!(refused(&mut s, &d), StreamError::Missing(_)));
    for key in [shape.key(), table.key()] {
        d.set[0].fill_image = Some(key);
        assert!(s.apply(&d).is_err());
    }
    d.set[0] = r.clone();
    d.upload = vec![picture.clone()];
    s.apply(&d).unwrap(); // Uploaded in the same delta.
    let mut d = delta(2);
    r.slot = 2;
    d.set = vec![r];
    s.apply(&d).unwrap(); // Already held.
    let mut d = delta(3);
    d.release = vec![picture.key()];
    assert!(matches!(refused(&mut s, &d), StreamError::InUse(_)));
    d.remove = vec![1];
    assert!(matches!(refused(&mut s, &d), StreamError::InUse(_)));
    d.remove = vec![1, 2];
    s.apply(&d).unwrap();
    assert!(!s.has_shape(&picture.key()));
}

#[test]
fn background_image_fill_is_refused_only_when_drawn() {
    let shape = square();
    let picture = fill_image();
    let mut r = path_record(1, &shape);
    r.features = FEATURE_BACKGROUND_IMAGE;
    r.fill_image = Some(picture.key());
    let mut d = delta(0);
    d.view = Some(common::flat_view(4, 4, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![shape, picture];
    d.set = vec![r];
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    assert!(s.draw(4, 4, 1).is_ok());
    let mut d = delta(1);
    d.order = Some(vec![entry(1)]);
    s.apply(&d).unwrap();
    let Err(DrawError::Unsupported(u)) = s.draw(4, 4, 1) else {
        panic!("not refused")
    };
    assert_eq!(
        (u.kind.as_str(), u.slot, u.object),
        ("path", Some(1), Some(101))
    );
    assert!(u.reason.contains("Cairo"), "{}", u.reason);
}

#[test]
fn display_ranks_are_unique_per_layer_in_each_views_order() {
    let mut a = display(1, vec![entry(1)]); // Self capture is allowed.
    let mut b = display(2, vec![]);
    let mut d = delta(0);
    d.set = vec![a.clone(), b.clone()];
    let mut s = Stream::new();
    s.apply(&d).unwrap(); // Outside the order, ranks may coincide.
    let mut d = delta(1);
    d.order = Some(vec![entry(1), entry(2)]);
    assert!(matches!(refused(&mut s, &d), StreamError::Invalid(_)));
    d.order.as_mut().unwrap()[1].layer = Layer::Overlay;
    s.apply(&d).unwrap(); // Same rank, different layer.
    let Body::Image(ImageSource::SubFrame { rank, .. }) = &mut b.body else {
        unreachable!()
    };
    *rank = 1;
    d.set = vec![b.clone()];
    d.order = Some(vec![entry(2), entry(1)]); // Rank order need not be draw order.
    s.apply(&d).unwrap();
    // A re-set can strand an unchanged sub-frame's order, even off the main order.
    let Body::Image(ImageSource::SubFrame { order, .. }) = &mut a.body else {
        unreachable!()
    };
    *order = vec![entry(1), entry(2)];
    d.set = vec![a];
    d.order = Some(vec![]);
    s.apply(&d).unwrap();
    let Body::Image(ImageSource::SubFrame { rank, .. }) = &mut b.body else {
        unreachable!()
    };
    *rank = 0;
    d.set = vec![b];
    assert!(matches!(refused(&mut s, &d), StreamError::Invalid(_)));
}

#[test]
fn a_path_maps_around_the_mapping_centre_and_a_point_cloud_around_the_frame_s() {
    // 16 x 9 pixels over a 14.2 x 8 frame: a unit is 1.125 px.
    let shape = square();
    let cloud = Shape::new(ShapeData::Path {
        points: vec![[0.0, 0.0, 0.0]],
    });
    let colours = Stops::new(vec![[0.0, 1.0, 0.0, 1.0]]);
    let mut points = Record::new(RecordKind::Points, 2, 102);
    points.geometry.shape_a = Some(cloud.key());
    points.body = Body::Points(Points {
        colors: colours.key(),
        radius: 1.0,
    });
    let mut d = delta(0);
    d.view = Some(common::flat_view(16, 9, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![shape.clone(), cloud.clone()];
    d.upload_paints = vec![colours];
    d.set = vec![path_record(1, &shape), points];
    d.order = Some(vec![entry(1), entry(2)]);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    let before = s.draw(16, 9, 1).unwrap();
    let boxes = s.screen_boxes().unwrap();
    let mut d = delta(1);
    let mut view = s.view().unwrap().clone();
    // Four frame units right: the path moves 4.5 px left, the cloud stays.
    view.mapping_centre = [4.0, 0.0, 0.0];
    d.view = Some(view);
    s.apply(&d).unwrap();
    let after = s.draw(16, 9, 1).unwrap();
    assert_ne!(after, before);
    let moved = s.screen_boxes().unwrap();
    assert!(
        (boxes[&1][0] - moved[&1][0] - 4.5).abs() < 1e-9,
        "{boxes:?} {moved:?}"
    );
    assert_eq!(boxes[&2], moved[&2]);
    let green = |frame: &[u8]| -> Vec<usize> {
        (0..16 * 9)
            .filter(|i| frame[4 * i + 1] == 255 && frame[4 * i] == 0)
            .collect()
    };
    assert_eq!(green(&after), green(&before));
}
