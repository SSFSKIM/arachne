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
    // blend. Its instances record and its face group draw.
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
            Ok(_) if matches!(kind, "instances" | "face_group") => {}
            Err(DrawError::Unsupported(u)) if !matches!(kind, "instances" | "face_group") => {
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
fn a_shaded_path_is_lit_as_ce_lights_it_and_refused_under_another_light() {
    // CE lights a `shade_in_3d` path under a 3D camera that shades: each paint becomes two stops,
    // lit at the path's start and end corners, drawn as a linear gradient between them. Under CE's
    // light model a path, or an instances record of one, is drawn so; without shading, or without
    // `shade_in_3d`, it is drawn unlit. Under another light model a path has no look yet: refused.
    let shape = square();
    let colours = Stops::new(vec![[1.0, 0.0, 0.0, 1.0]]);
    for instanced in [false, true] {
        let draw = |shade_in_3d: bool, shading: bool, light: LightModel| {
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
            let mut view = common::three_d_view(48, 27);
            view.light = light;
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
            s.draw(48, 27, 1)
        };
        let what = format!("instanced {instanced}");
        let lit = draw(true, true, LightModel::Ce).unwrap();
        let unlit = draw(false, true, LightModel::Ce).unwrap();
        assert_eq!(unlit, draw(true, false, LightModel::Ce).unwrap(), "{what}");
        assert_ne!(lit, unlit, "{what}: drawn unlit");
        let soft = LightModel::SoftDiffuse {
            ambient: 0.2,
            wrap: 0.5,
        };
        let Err(DrawError::Unsupported(u)) = draw(true, true, soft) else {
            panic!("{what}: drawn under soft diffuse light")
        };
        let kind = if instanced { "instances" } else { "path" };
        assert_eq!((u.kind.as_str(), u.slot), (kind, Some(1)), "{what}");
        assert!(u.reason.contains("light model"), "{what}: {}", u.reason);
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
fn the_frame_under_the_overlay_leaves_its_entries_to_cairo_in_order() {
    // Slot 1 is drawn by the core; 2 is fixed in frame, 3 a display, 4 filled through a
    // background image: the overlay, in that order, neither drawn nor refused under it.
    let shape = square();
    let picture = fill_image();
    let drawn = path_record(1, &shape);
    let mut fixed = path_record(2, &shape);
    fixed.paint.fill = Paint::One([0.0, 1.0, 0.0, 1.0]);
    fixed.depth.space = Space::Overlay;
    let mut filled = path_record(4, &shape);
    filled.features = FEATURE_BACKGROUND_IMAGE;
    filled.fill_image = Some(picture.key());
    let mut d = delta(0);
    d.view = Some(common::flat_view(4, 4, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![shape, picture];
    d.set = vec![drawn, fixed, display(3, vec![]), filled];
    d.order = Some(vec![entry(2), entry(1), entry(3), entry(4)]);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    let slots: Vec<u32> = s.overlay_entries().iter().map(|e| e.slot).collect();
    assert_eq!(slots, [2, 3, 4]);
    assert!(matches!(s.draw(4, 4, 1), Err(DrawError::Unsupported(_))));
    let under = s.render_under_overlay(4, 4, 1).unwrap().rgba;
    let mut alone = delta(1);
    alone.order = Some(vec![entry(1)]);
    s.apply(&alone).unwrap();
    assert_eq!(under, s.draw(4, 4, 1).unwrap());
    assert!(s.overlay_entries().is_empty());
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

fn pre_projected_path(shape: &Shape, points: Key) -> Record {
    let mut r = path_record(1, shape);
    r.features = FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points,
        gradient: None,
        fill: Paint::None,
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    r
}

#[test]
fn pre_projected_points_must_be_a_path_held_or_uploaded() {
    let shape = square();
    let points = Shape::new(ShapeData::Path {
        points: vec![[1.0, 2.0, 3.0]; 2],
    });
    let mut s = Stream::new();
    let mut d = delta(0);
    d.upload = vec![shape.clone(), fill_image()];
    s.apply(&d).unwrap();
    let mut d = delta(1);
    d.set = vec![pre_projected_path(&shape, points.key())];
    let err = refused(&mut s, &d);
    assert!(matches!(err, StreamError::Missing(_)));
    assert!(err.to_string().contains("slot 1 references shape"));
    d.set[0].pre_projected.as_mut().unwrap().points = fill_image().key();
    assert!(matches!(refused(&mut s, &d), StreamError::Invalid(_)));
    d.set[0].pre_projected.as_mut().unwrap().points = points.key();
    d.upload = vec![points.clone()];
    s.apply(&d).unwrap(); // Different counts are allowed; upload in this delta.
    d.upload.clear();
    s.apply(&d).unwrap(); // Already held.
    let mut release = delta(2);
    release.release = vec![points.key()];
    assert!(matches!(refused(&mut s, &release), StreamError::InUse(_)));
    release.remove = vec![1];
    s.apply(&release).unwrap();
}

#[test]
fn every_pre_projected_paint_table_must_exist_and_cannot_be_released_in_use() {
    let shape = square();
    let table = Stops::new(vec![[0.2, 0.4, 0.6, 0.8]; 3]);
    for which in 0..3 {
        let mut r = pre_projected_path(&shape, shape.key());
        let p = r.pre_projected.as_mut().unwrap();
        match which {
            0 => p.fill = Paint::Table(table.key()),
            1 => p.stroke = Paint::Table(table.key()),
            _ => p.background_stroke = Paint::Table(table.key()),
        }
        let mut d = delta(0);
        d.upload = vec![shape.clone()];
        d.set = vec![r];
        let mut s = Stream::new();
        let err = refused(&mut s, &d);
        assert!(matches!(err, StreamError::Missing(_)));
        assert!(err.to_string().contains("paint table"));
        d.upload_paints = vec![table.clone()];
        s.apply(&d).unwrap();
        let mut release = delta(1);
        release.release = vec![table.key()];
        assert!(matches!(refused(&mut s, &release), StreamError::InUse(_)));
        release.remove = vec![1];
        s.apply(&release).unwrap();
    }
}

#[test]
fn a_pre_projected_record_is_drawn_from_its_tail_alone() {
    // The record's own paint fills the square; its camera painted nothing, and nothing is drawn.
    let shape = square();
    let mut d = delta(0);
    d.view = Some(common::flat_view(4, 4, [0.0, 0.0, 0.0, 1.0]));
    d.upload = vec![shape.clone()];
    d.set = vec![pre_projected_path(&shape, shape.key())];
    d.order = Some(vec![entry(1)]);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    assert_eq!(s.draw(4, 4, 1).unwrap(), [0, 0, 0, 255].repeat(16));
    let mut painted = pre_projected_path(&shape, shape.key());
    painted.pre_projected.as_mut().unwrap().fill = Paint::One([0.0, 1.0, 0.0, 1.0]);
    let mut d = delta(1);
    d.set = vec![painted];
    s.apply(&d).unwrap();
    // A square a fraction of a pixel across: some green is drawn.
    assert!(
        s.draw(4, 4, 1)
            .unwrap()
            .chunks(4)
            .any(|p| p[1] > 0 && p[0] == 0)
    );
}

/// The frame one path record draws alone, over black, at 48 x 27 pixels (a flat camera, so no
/// depth can differ), with `uploads` beside its shape.
fn drawn_alone(record: Record, uploads: Vec<Shape>) -> Result<Vec<u8>, DrawError> {
    let mut d = delta(0);
    d.view = Some(common::flat_view(48, 27, [0.0, 0.0, 0.0, 1.0]));
    d.upload = uploads;
    d.order = Some(vec![entry(record.slot)]);
    d.set = vec![record];
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    s.draw(48, 27, 1)
}

fn outlined(slot: u32, shape: &Shape) -> Record {
    let mut r = path_record(slot, shape);
    r.paint.stroke = Paint::One([1.0, 1.0, 0.0, 1.0]);
    r.paint.stroke_width = 8.0;
    r
}

#[test]
fn a_reveal_draws_the_window_it_cuts() {
    use arachne_core::geometry::revealed;
    let shape = Shape::new(ShapeData::Path {
        points: common::square_points(2.0),
    });
    let mut windowed = outlined(1, &shape);
    windowed.geometry.reveal = Some((0.2, 0.65));
    let cut = Shape::new(ShapeData::Path {
        points: revealed(&common::square_points(2.0), 0.2, 0.65),
    });
    let got = drawn_alone(windowed, vec![shape.clone()]).unwrap();
    assert_eq!(got, drawn_alone(outlined(1, &cut), vec![cut]).unwrap());
    assert_ne!(got, drawn_alone(outlined(1, &shape), vec![shape]).unwrap());
}

#[test]
fn a_blend_and_its_paint_blend_draw_what_they_mix() {
    use arachne_core::geometry::{drawn_paint, path_points};
    let (a, b) = (common::square_points(1.0), common::square_points(3.0));
    let (shape_a, shape_b) = (
        Shape::new(ShapeData::Path { points: a.clone() }),
        Shape::new(ShapeData::Path { points: b.clone() }),
    );
    let mut blending = outlined(1, &shape_a);
    blending.geometry.blend = Some(Blend {
        shape_b: shape_b.key(),
        place_b: IDENTITY,
        t: 0.3,
    });
    blending.paint.paint_b = Some(PaintB {
        fill: Paint::One([0.0, 0.0, 1.0, 0.5]),
        stroke: Paint::One([0.0, 1.0, 1.0, 1.0]),
        background_stroke: Paint::None,
        stroke_width: 2.0,
        background_stroke_width: 0.0,
    });
    blending.paint.mix = 0.3;
    let mixed = Shape::new(ShapeData::Path {
        points: path_points(&blending.geometry, &a, Some(&b)).unwrap(),
    });
    let mut plain = path_record(1, &mixed);
    let drawn = drawn_paint(&blending.paint, |_: &Key| unreachable!("no tables")).unwrap();
    let paint = |rows: Option<Vec<[f64; 4]>>| match rows.as_deref() {
        None => Paint::None,
        Some([c]) => Paint::One(*c),
        Some([c, d]) => Paint::Two([*c, *d]),
        Some(_) => unreachable!("one or two rows"),
    };
    plain.paint.fill = paint(drawn.fill);
    plain.paint.stroke = paint(drawn.stroke);
    plain.paint.background_stroke = paint(drawn.background_stroke);
    plain.paint.stroke_width = drawn.stroke_width;
    plain.paint.background_stroke_width = drawn.background_stroke_width;
    let got = drawn_alone(blending, vec![shape_a.clone(), shape_b]).unwrap();
    assert_eq!(got, drawn_alone(plain, vec![mixed]).unwrap());
    assert_ne!(
        got,
        drawn_alone(outlined(1, &shape_a), vec![shape_a]).unwrap()
    );
}

#[test]
fn a_blend_of_shapes_of_different_lengths_is_refused_whole() {
    // CE aligns a morph's points before it blends them: never made from CE.
    let shape = square();
    let short = Shape::new(ShapeData::Path {
        points: common::square_points(1.0)[..4].to_vec(),
    });
    let mut uneven = path_record(1, &shape);
    uneven.geometry.blend = Some(Blend {
        shape_b: short.key(),
        place_b: IDENTITY,
        t: 0.5,
    });
    let Err(DrawError::Unsupported(u)) = drawn_alone(uneven, vec![shape.clone(), short]) else {
        panic!("drawn")
    };
    assert!(u.reason.contains("different lengths"), "{}", u.reason);
}

#[test]
fn a_paint_blend_of_tables_draws_what_its_mix_draws() {
    // A morph between gradients of three colours or more: CE mixes the aligned rows one by one.
    let shape = Shape::new(ShapeData::Path {
        points: common::square_points(6.0),
    });
    let start = Stops::new(vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]);
    let end = Stops::new(vec![
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.5],
    ]);
    let mix = Stops::new(vec![
        [1.0, 0.25, 0.25, 1.0],
        [0.0, 0.75, 0.0, 1.0],
        [0.25, 0.25, 0.75, 0.875],
    ]);
    let ends = Some([[-6.0, -6.0, 0.0], [6.0, 6.0, 0.0]]);
    let mut tabled = path_record(1, &shape);
    tabled.paint.fill = Paint::Table(start.key());
    tabled.paint.gradient = ends;
    tabled.paint.paint_b = Some(PaintB {
        fill: Paint::Table(end.key()),
        stroke: Paint::None,
        background_stroke: Paint::None,
        stroke_width: 0.0,
        background_stroke_width: 0.0,
    });
    tabled.paint.mix = 0.25;
    let mut plain = path_record(1, &shape);
    plain.paint.fill = Paint::Table(mix.key());
    plain.paint.gradient = ends;
    let frame = |record: Record| {
        let mut d = delta(0);
        d.view = Some(common::flat_view(48, 27, [0.0, 0.0, 0.0, 1.0]));
        d.upload = vec![shape.clone()];
        d.upload_paints = vec![start.clone(), end.clone(), mix.clone()];
        d.order = Some(vec![entry(1)]);
        d.set = vec![record];
        let mut s = Stream::new();
        s.apply(&d).unwrap();
        s.draw(48, 27, 1).unwrap()
    };
    let got = frame(tabled);
    assert_eq!(got, frame(plain));
    assert!(
        got.chunks(4)
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 20
    );
}

/// Two unit squares side by side, as one shape of two faces: the left face, then the right.
fn two_faces() -> Vec<[f64; 3]> {
    let square = |x0: f64| {
        let shifted: Vec<[f64; 3]> = common::square_points(0.5)
            .iter()
            .map(|p| [p[0] + x0, p[1], p[2]])
            .collect();
        shifted
    };
    [square(-0.5), square(0.5)].concat()
}

#[test]
fn a_face_group_draws_each_face_as_the_path_of_that_face() {
    // CE draws each face of a surface as a path of its own: its points, its row of the face
    // tables, the record's background stroke and widths. Under a 3D camera that shades, each face
    // is lit as a shaded path is.
    let points = two_faces();
    let shape = Shape::new(ShapeData::Path {
        points: points.clone(),
    });
    let fills = Stops::new(vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]);
    let strokes = Stops::new(vec![[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]);
    for shading in [false, true] {
        let mut view = common::three_d_view(64, 36);
        let Camera::ThreeD(c) = &mut view.camera else {
            unreachable!()
        };
        c.should_apply_shading = shading;
        let style = |r: &mut Record| {
            r.depth.space = Space::World;
            r.depth.shade_in_3d = true;
            r.paint.stroke_width = 2.0;
        };
        let mut faces = Record::new(RecordKind::FaceGroup, 1, 101);
        faces.geometry.shape_a = Some(shape.key());
        faces.body = Body::FaceGroup(FaceGroup {
            uv: vec![[0, 0], [1, 0]],
            grid: [2, 1],
            fill: fills.key(),
            stroke: Some(strokes.key()),
            face_ends: Vec::new(),
            face_widths: Vec::new(),
        });
        style(&mut faces);
        let mut d = delta(0);
        d.view = Some(view.clone());
        d.upload = vec![shape.clone()];
        d.upload_paints = vec![fills.clone(), strokes.clone()];
        d.set = vec![faces];
        d.order = Some(
            [1, 0]
                .map(|part| OrderEntry {
                    layer: Layer::Moving,
                    slot: 1,
                    part,
                })
                .to_vec(),
        );
        let mut s = Stream::new();
        s.apply(&d).unwrap();
        let grouped = s.draw(64, 36, 1).unwrap();

        // The same faces as two path records, in the same order.
        let (left, right) = points.split_at(16);
        let mut d = delta(0);
        d.view = Some(view);
        let mut set = Vec::new();
        for (slot, half, row) in [(2, right, 1), (3, left, 0)] {
            let face = Shape::new(ShapeData::Path {
                points: half.to_vec(),
            });
            let mut r = path_record(slot, &face);
            r.paint.fill = Paint::One(fills.rows()[row]);
            r.paint.stroke = Paint::One(strokes.rows()[row]);
            style(&mut r);
            d.upload.push(face);
            set.push(r);
        }
        d.set = set;
        d.order = Some(vec![entry(2), entry(3)]);
        let mut s = Stream::new();
        s.apply(&d).unwrap();
        assert_eq!(grouped, s.draw(64, 36, 1).unwrap(), "shading {shading}");
    }
}

#[test]
fn a_face_group_whose_points_do_not_divide_into_its_faces_is_refused() {
    let shape = Shape::new(ShapeData::Path {
        points: two_faces()[..13].to_vec(),
    });
    let fills = Stops::new(vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]);
    let mut faces = Record::new(RecordKind::FaceGroup, 1, 101);
    faces.geometry.shape_a = Some(shape.key());
    faces.body = Body::FaceGroup(FaceGroup {
        uv: vec![[0, 0], [1, 0]],
        grid: [2, 1],
        fill: fills.key(),
        stroke: None,
        face_ends: Vec::new(),
        face_widths: Vec::new(),
    });
    let mut d = delta(0);
    d.view = Some(common::three_d_view(16, 9));
    d.upload = vec![shape];
    d.upload_paints = vec![fills];
    d.set = vec![faces];
    d.order = Some(vec![entry(1)]);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    let Err(DrawError::Unsupported(u)) = s.draw(16, 9, 1) else {
        panic!("drawn")
    };
    assert!(u.reason.contains("divide"), "{}", u.reason);
}

#[test]
fn a_face_group_draws_faces_of_their_own_lengths_and_widths_as_their_paths() {
    // A surface partly drawn (`Create`) holds faces of different lengths, and one mid-`Write`
    // strokes each face at a width of its own: each face is the path of its own run of the points,
    // ending at its face end, stroked at its own width.
    let points = two_faces();
    let (left, right) = points.split_at(16);
    let right = &right[..8];
    let shape = Shape::new(ShapeData::Path {
        points: [left, right].concat(),
    });
    let fills = Stops::new(vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]);
    let strokes = Stops::new(vec![[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]);
    let widths = [2.0, 0.75];
    for shading in [false, true] {
        let mut view = common::three_d_view(64, 36);
        let Camera::ThreeD(c) = &mut view.camera else {
            unreachable!()
        };
        c.should_apply_shading = shading;
        let style = |r: &mut Record, width: f64| {
            r.depth.space = Space::World;
            r.depth.shade_in_3d = true;
            r.paint.stroke_width = width;
        };
        let mut faces = Record::new(RecordKind::FaceGroup, 1, 101);
        faces.geometry.shape_a = Some(shape.key());
        faces.body = Body::FaceGroup(FaceGroup {
            uv: vec![[0, 0], [1, 0]],
            grid: [2, 1],
            fill: fills.key(),
            stroke: Some(strokes.key()),
            face_ends: vec![16, 24],
            face_widths: widths.to_vec(),
        });
        style(&mut faces, widths[0]);
        let mut d = delta(0);
        d.view = Some(view.clone());
        d.upload = vec![shape.clone()];
        d.upload_paints = vec![fills.clone(), strokes.clone()];
        d.set = vec![faces];
        d.order = Some(
            [1, 0]
                .map(|part| OrderEntry {
                    layer: Layer::Moving,
                    slot: 1,
                    part,
                })
                .to_vec(),
        );
        let mut s = Stream::new();
        s.apply(&d).unwrap();
        let grouped = s.draw(64, 36, 1).unwrap();

        let mut d = delta(0);
        d.view = Some(view);
        let mut set = Vec::new();
        for (slot, face, row) in [(2, right, 1), (3, left, 0)] {
            let face = Shape::new(ShapeData::Path {
                points: face.to_vec(),
            });
            let mut r = path_record(slot, &face);
            r.paint.fill = Paint::One(fills.rows()[row]);
            r.paint.stroke = Paint::One(strokes.rows()[row]);
            style(&mut r, widths[row]);
            d.upload.push(face);
            set.push(r);
        }
        d.set = set;
        d.order = Some(vec![entry(2), entry(3)]);
        let mut s = Stream::new();
        s.apply(&d).unwrap();
        assert_eq!(grouped, s.draw(64, 36, 1).unwrap(), "shading {shading}");
    }
}

/// A delta setting a face group of `faces` faces over `points` of the two faces' points, with
/// `face_ends` and `face_widths`.
fn faces_with_ends(
    faces: usize,
    points: usize,
    face_ends: Vec<u32>,
    face_widths: Vec<f64>,
) -> Delta {
    let shape = Shape::new(ShapeData::Path {
        points: two_faces()[..points].to_vec(),
    });
    let fills = Stops::new(vec![[1.0, 0.0, 0.0, 1.0]; faces]);
    let mut record = Record::new(RecordKind::FaceGroup, 1, 101);
    record.geometry.shape_a = Some(shape.key());
    record.body = Body::FaceGroup(FaceGroup {
        uv: (0..faces as u32).map(|u| [u, 0]).collect(),
        grid: [faces as u32, 1],
        fill: fills.key(),
        stroke: None,
        face_ends,
        face_widths,
    });
    let mut d = delta(0);
    d.view = Some(common::three_d_view(16, 9));
    d.upload = vec![shape];
    d.upload_paints = vec![fills];
    d.set = vec![record];
    d.order = Some(vec![entry(1)]);
    d
}

/// The reason `apply` refuses `d` with, which must be the record's own.
fn invalid(d: &Delta) -> String {
    let mut s = Stream::new();
    match refused(&mut s, d) {
        StreamError::Invalid(reason) => reason,
        other => panic!("refused otherwise: {other}"),
    }
}

#[test]
fn a_face_group_s_face_ends_are_accepted_as_one_run_per_face_ending_at_its_points() {
    let mut s = Stream::new();
    s.apply(&faces_with_ends(
        3,
        24,
        vec![4, 16, 24],
        vec![1.0, 2.0, 0.0],
    ))
    .unwrap();
    s.apply(&faces_with_ends(2, 24, Vec::new(), Vec::new()))
        .unwrap();
}

#[test]
fn a_face_group_whose_face_ends_fall_back_is_refused() {
    let reason = invalid(&faces_with_ends(3, 24, vec![16, 8, 24], Vec::new()));
    assert!(reason.contains("face ends"), "{reason}");
    // A face end equal to the one before: a face of no points.
    let reason = invalid(&faces_with_ends(3, 24, vec![8, 8, 24], Vec::new()));
    assert!(reason.contains("face ends"), "{reason}");
}

#[test]
fn a_face_group_whose_last_face_end_is_not_its_points_count_is_refused() {
    let reason = invalid(&faces_with_ends(2, 24, vec![16, 20], Vec::new()));
    assert!(reason.contains("face ends"), "{reason}");
    let reason = invalid(&faces_with_ends(2, 24, vec![16, 28], Vec::new()));
    assert!(reason.contains("face ends"), "{reason}");
}

#[test]
fn a_face_group_with_face_ends_other_than_one_per_face_is_refused() {
    let reason = invalid(&faces_with_ends(2, 24, vec![24], Vec::new()));
    assert!(reason.contains("face ends"), "{reason}");
    let reason = invalid(&faces_with_ends(2, 24, vec![8, 16, 24], Vec::new()));
    assert!(reason.contains("face ends"), "{reason}");
}

#[test]
fn a_face_group_with_face_widths_other_than_one_per_face_is_refused() {
    let reason = invalid(&faces_with_ends(2, 24, Vec::new(), vec![1.0]));
    assert!(reason.contains("face widths"), "{reason}");
    for width in [-1.0, f64::NAN, f64::INFINITY] {
        let reason = invalid(&faces_with_ends(2, 24, Vec::new(), vec![1.0, width]));
        assert!(reason.contains("face widths"), "{reason}");
    }
}

#[test]
fn a_face_group_read_without_face_ends_draws_equal_runs_at_the_record_s_width() {
    // A face group of an earlier minor carries neither list: read back from its bytes, it draws
    // as the same faces stating equal runs and the record's width for each.
    use arachne_core::xf::wire;
    let shape = Shape::new(ShapeData::Path {
        points: two_faces(),
    });
    let fills = Stops::new(vec![[1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]]);
    let strokes = Stops::new(vec![[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]);
    let drawn = |face_ends: Vec<u32>, face_widths: Vec<f64>| {
        let mut faces = Record::new(RecordKind::FaceGroup, 1, 101);
        faces.geometry.shape_a = Some(shape.key());
        faces.depth.space = Space::World;
        faces.paint.stroke_width = 1.5;
        faces.body = Body::FaceGroup(FaceGroup {
            uv: vec![[0, 0], [1, 0]],
            grid: [2, 1],
            fill: fills.key(),
            stroke: Some(strokes.key()),
            face_ends,
            face_widths,
        });
        let mut d = delta(0);
        d.view = Some(common::three_d_view(64, 36));
        d.upload = vec![shape.clone()];
        d.upload_paints = vec![fills.clone(), strokes.clone()];
        d.set = vec![faces];
        d.order = Some(
            [0, 1]
                .map(|part| OrderEntry {
                    layer: Layer::Moving,
                    slot: 1,
                    part,
                })
                .to_vec(),
        );
        let read = wire::decode_delta(&wire::encode_delta(&d)).unwrap();
        let mut s = Stream::new();
        s.apply(&read).unwrap();
        (read.version, s.draw(64, 36, 1).unwrap())
    };
    let (old, legacy) = drawn(Vec::new(), Vec::new());
    let (new, stated) = drawn(vec![16, 32], vec![1.5, 1.5]);
    assert!(old.1 < 3 && new == (1, 3), "{old:?} {new:?}");
    assert_eq!(legacy, stated);
}

#[test]
fn each_absent_face_list_reads_as_uniform_whatever_the_other() {
    // Neither list, face ends alone, face widths alone, both: each absent list is read as its
    // uniform form (equal runs; the record's width) on its own, so each group, read back from its
    // bytes, carries what it was sent and draws as the same faces stating both lists.
    use arachne_core::xf::wire;
    let points = two_faces();
    let (left, right) = points.split_at(16);
    let fills = Stops::new(vec![[1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]]);
    let strokes = Stops::new(vec![[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]);
    let record_width = 1.5;
    let drawn = |shape: &Shape, face_ends: Vec<u32>, face_widths: Vec<f64>| {
        let mut faces = Record::new(RecordKind::FaceGroup, 1, 101);
        faces.geometry.shape_a = Some(shape.key());
        faces.depth.space = Space::World;
        faces.paint.stroke_width = record_width;
        faces.body = Body::FaceGroup(FaceGroup {
            uv: vec![[0, 0], [1, 0]],
            grid: [2, 1],
            fill: fills.key(),
            stroke: Some(strokes.key()),
            face_ends,
            face_widths,
        });
        let mut d = delta(0);
        d.view = Some(common::three_d_view(64, 36));
        d.upload = vec![shape.clone()];
        d.upload_paints = vec![fills.clone(), strokes.clone()];
        d.set = vec![faces];
        d.order = Some(
            [0, 1]
                .map(|part| OrderEntry {
                    layer: Layer::Moving,
                    slot: 1,
                    part,
                })
                .to_vec(),
        );
        let read = wire::decode_delta(&wire::encode_delta(&d)).unwrap();
        let Body::FaceGroup(f) = &read.set[0].body else {
            unreachable!()
        };
        let lists = (f.face_ends.clone(), f.face_widths.clone());
        let mut s = Stream::new();
        s.apply(&read).unwrap();
        (read.version, lists, s.draw(64, 36, 1).unwrap())
    };
    let whole = Shape::new(ShapeData::Path {
        points: points.clone(),
    });
    let partial = Shape::new(ShapeData::Path {
        points: [left, &right[..8]].concat(),
    });
    let widths = vec![2.0, 0.75];
    for (name, shape, ends, sent_widths, uniform_ends) in [
        ("neither", &whole, vec![], vec![], vec![16, 32]),
        ("ends alone", &partial, vec![16, 24], vec![], vec![16, 24]),
        ("widths alone", &whole, vec![], widths.clone(), vec![16, 32]),
        ("both", &partial, vec![16, 24], widths.clone(), vec![16, 24]),
    ] {
        let carried = !ends.is_empty() || !sent_widths.is_empty();
        let (version, lists, frame) = drawn(shape, ends.clone(), sent_widths.clone());
        assert_eq!(lists, (ends, sent_widths.clone()), "{name}: read back as sent");
        assert_eq!(version.1 == 3, carried, "{name}: {version:?}");
        let stated_widths = if sent_widths.is_empty() {
            vec![record_width; 2]
        } else {
            sent_widths
        };
        let (_, _, stated) = drawn(shape, uniform_ends, stated_widths);
        assert_eq!(frame, stated, "{name}");
    }
}

/// A face group of one face whose points are at one spot (a face not yet drawn), stroked under
/// `cap` and `dash`: its frame.
fn one_spot_face(cap: Cap, dash: Option<Dash>) -> Vec<u8> {
    let shape = Shape::new(ShapeData::Path {
        points: vec![[0.25, 0.0, 0.0]; 4],
    });
    let fills = Stops::new(vec![[1.0, 0.0, 0.0, 1.0]]);
    let strokes = Stops::new(vec![[1.0, 1.0, 1.0, 1.0]]);
    let mut face = Record::new(RecordKind::FaceGroup, 1, 101);
    face.geometry.shape_a = Some(shape.key());
    face.depth.space = Space::World;
    face.paint.stroke_width = 8.0;
    face.paint.cap = cap;
    face.dash = dash;
    face.body = Body::FaceGroup(FaceGroup {
        uv: vec![[0, 0]],
        grid: [1, 1],
        fill: fills.key(),
        stroke: Some(strokes.key()),
        face_ends: Vec::new(),
        face_widths: Vec::new(),
    });
    let mut d = delta(0);
    d.view = Some(common::three_d_view(64, 36));
    d.upload = vec![shape];
    d.upload_paints = vec![fills, strokes];
    d.set = vec![face];
    d.order = Some(vec![entry(1)]);
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    s.draw(64, 36, 1).unwrap()
}

#[test]
fn a_face_at_one_spot_under_a_round_cap_is_a_dot_only_where_its_dash_is_on() {
    // Cairo's dashed stroker starts each subpath at the dash's offset, and strokes a subpath of
    // no length as a round cap's dot only when the dash is on there (`dash_starts_on`): a face at
    // one spot is the dot undashed or dashed on at its start, and nothing dashed in a gap there.
    let nothing = one_spot_face(Cap::Butt, None);
    let undashed = one_spot_face(Cap::Round, None);
    assert_ne!(undashed, nothing, "a round cap's dot");
    let dashed = |phase: f64| {
        one_spot_face(
            Cap::Round,
            Some(Dash {
                period: 1.0,
                duty: 0.5,
                phase,
            }),
        )
    };
    assert_eq!(dashed(0.0), undashed, "dashed on at its start");
    assert_eq!(dashed(0.25), undashed, "dashed on at its start");
    assert_eq!(dashed(0.75), nothing, "dashed in a gap at its start");
    // Where the offset meets the dash's end, Cairo has turned the dash off.
    assert_eq!(dashed(0.5), nothing, "dashed in a gap at its start");
}
