//! The stream state and what the oracle refuses (contract §11): a frame it cannot judge is
//! `truth_unable` whole, never drawn partly.

mod common;

use arachne_core::xf::{
    Background, Body, Delta, FEATURE_BACKGROUND_IMAGE, FEATURE_PRE_PROJECTED, FaceGroup,
    ImageSource, Instances, Key, Layer as XfLayer, LightModel, OrderEntry, Paint, Points,
    PreProjected, Record, RecordKind, Shape, ShapeData, Space, Stops, XF_VERSION, wire,
};
use arachne_oracle::out::{STATUS_DECIDED, STATUS_UNABLE};
use arachne_oracle::stream::{KNOWN_XF, Reason, State};
use common::*;

/// A filled square path in slot `slot` and its shape.
fn square(slot: u32) -> (arachne_core::xf::Shape, Record) {
    let v = view(16, 16);
    let s = path_shape(polygon(&[
        frame(&v, [2.0, 2.0]),
        frame(&v, [10.0, 2.0]),
        frame(&v, [10.0, 10.0]),
        frame(&v, [2.0, 10.0]),
    ]));
    let r = path(slot, s.key(), true, None);
    (s, r)
}

fn opaque(kind: RecordKind, slot: u32) -> Record {
    let mut r = Record::new(kind, slot, 100 + slot);
    r.body = Body::Opaque(vec![1, 2, 3]);
    r
}

#[test]
fn the_oracle_reads_the_xf_version_the_core_writes() {
    // An XF change updates the oracle in the same change: this fails until it does.
    // The oracle reads XF 1.5's source section itself, whatever minor the shared reader writes.
    assert_eq!(KNOWN_XF, (1, 5));
    assert!(KNOWN_XF.0 == XF_VERSION.0 && KNOWN_XF.1 >= XF_VERSION.1);
}

#[test]
fn a_judged_frame_is_decided() {
    let (s, r) = square(0);
    let d = decide(&[delta(1, Some(view(16, 16)), vec![s], vec![r])]);
    assert_eq!(d.header.status, STATUS_DECIDED);
    assert!(d.refused.is_empty());
    assert!(!d.fragments.is_empty());
}

#[test]
fn an_unknown_kind_in_the_order_is_truth_unable_for_the_whole_frame() {
    let (s, r) = square(0);
    let d = decide(&[delta(
        1,
        Some(view(16, 16)),
        vec![s],
        vec![r, opaque(RecordKind::Unknown(99), 1)],
    )]);
    assert_eq!(d.header.status, STATUS_UNABLE);
    // Nothing is drawn partly: the square, which the oracle could judge, has no fragment.
    assert!(d.fragments.is_empty() && d.covers.is_empty());
    assert_eq!(d.refused.len(), 1);
    let r = &d.refused[0];
    assert_eq!(
        (r.reason, r.kind, r.entry, r.slot, r.object),
        (Reason::UnknownKind.code(), 99, 1, 1, 101)
    );
}

#[test]
fn an_unknown_kind_outside_the_order_is_not_refused() {
    let (s, r) = square(0);
    let mut d = delta(
        1,
        Some(view(16, 16)),
        vec![s],
        vec![r, opaque(RecordKind::Unknown(99), 1)],
    );
    d.order = Some(vec![entry(0)]);
    let out = decide(&[d]);
    assert_eq!(out.header.status, STATUS_DECIDED);
}

#[test]
fn reserved_kinds_unknown_bits_and_light_models_are_refused_by_name() {
    let (s, r) = square(0);
    let reserved = decide(&[delta(
        1,
        Some(view(16, 16)),
        vec![s.clone()],
        vec![r.clone(), opaque(RecordKind::Volume, 1)],
    )]);
    assert_eq!(reserved.refused[0].reason, Reason::ReservedKind.code());

    let mut bits = r.clone();
    bits.features = 4;
    let unknown = decide(&[delta(1, Some(view(16, 16)), vec![s.clone()], vec![bits])]);
    assert_eq!(unknown.header.status, STATUS_UNABLE);
    assert_eq!(
        (unknown.refused[0].reason, unknown.refused[0].bits),
        (Reason::UnknownFeature.code(), 4)
    );

    let mut v = view(16, 16);
    v.light = LightModel::Other {
        code: 7,
        params: vec![],
    };
    let light = decide(&[delta(1, Some(v), vec![s], vec![r])]);
    assert_eq!(light.refused[0].reason, Reason::UnknownLightModel.code());
}

#[test]
fn what_the_core_refuses_is_refused_and_the_overlay_is_not() {
    let (s, mut r) = square(0);
    r.clip = Some((s.key(), arachne_core::xf::IDENTITY));
    let clipped = decide(&[delta(
        1,
        Some(view(16, 16)),
        vec![s.clone()],
        vec![r.clone()],
    )]);
    assert_eq!(clipped.refused[0].reason, Reason::CoreRefuses.code());
    // The same record in overlay space is Cairo's: neither drawn nor refused.
    r.depth.space = Space::Overlay;
    let overlay = decide(&[delta(1, Some(view(16, 16)), vec![s], vec![r])]);
    assert_eq!(overlay.header.status, STATUS_DECIDED);
    assert!(overlay.fragments.is_empty());
}

#[test]
fn a_newer_xf_minor_refuses_its_frame_and_every_frame_after() {
    let (s, r) = square(0);
    let mut first = delta(1, Some(view(16, 16)), vec![s], vec![r]);
    first.version = (KNOWN_XF.0, KNOWN_XF.1 + 1);
    let mut held = delta(2, None, vec![], vec![]);
    held.order = None;
    let at_first = decide(std::slice::from_ref(&first));
    assert_eq!(at_first.refused[0].reason, Reason::XfNewer.code());
    let after = decide(&[first, held]);
    assert_eq!(after.header.status, STATUS_UNABLE);
    assert_eq!(after.refused[0].reason, Reason::XfNewer.code());
}

#[test]
fn a_delta_that_is_not_valid_xf_changes_nothing_and_its_frame_is_refused() {
    let (s, r) = square(0);
    let good = delta(1, Some(view(16, 16)), vec![s], vec![r]);
    // A record naming a shape the stream lacks.
    let mut bad = delta(2, None, vec![], vec![path(1, Key([7; 16]), true, None)]);
    bad.order = Some(vec![entry(0), entry(1)]);
    let mut held = delta(3, None, vec![], vec![]);
    held.order = None;
    let deltas = [good.clone(), bad, held];
    let at_bad = arachne_oracle::decide::decide_at(&deltas, 2, None).unwrap();
    assert_eq!(at_bad.header.status, STATUS_UNABLE);
    assert_eq!(at_bad.refused[0].reason, Reason::InvalidDelta.code());
    // The next frame draws the stream as the good delta left it.
    let after = decide(&deltas);
    let before = decide(&[good]);
    assert_eq!(cover_lines(&after), cover_lines(&before));
    let mut state = State::new();
    assert!(state.apply(&deltas[0]).is_ok());
    assert!(state.apply(&deltas[1]).is_err());
    assert_eq!(state.records.len(), 1);
}

#[test]
fn records_set_removed_and_reordered_apply_in_xfs_order() {
    let (s, r) = square(0);
    let mut state = State::new();
    state
        .apply(&delta(
            1,
            Some(view(16, 16)),
            vec![s.clone()],
            vec![r.clone()],
        ))
        .unwrap();
    // A slot set and removed in one delta is removed.
    let mut d = delta(2, None, vec![], vec![r.clone()]);
    d.remove = vec![0];
    d.order = Some(vec![]);
    state.apply(&d).unwrap();
    assert!(state.records.is_empty());
    // A key still used cannot be released.
    let mut d = delta(3, None, vec![], vec![r]);
    d.release = vec![s.key()];
    assert!(state.apply(&d).is_err());
}

/// A display in slot `slot`: capture rank `rank`, its own 8 x 8 view drawing `order`.
fn display(slot: u32, rank: u32, order: Vec<OrderEntry>) -> Record {
    let mut r = Record::new(RecordKind::Image, slot, 100 + slot);
    r.body = Body::Image(ImageSource::SubFrame {
        rank,
        corners: [[0.0; 3]; 4],
        resampling: 0,
        opacity: 1.0,
        view: Box::new(view(8, 8)),
        order,
    });
    r
}

fn at(layer: XfLayer, slot: u32) -> OrderEntry {
    OrderEntry {
        layer,
        slot,
        part: 0,
    }
}

/// An image shape of `w` x `h` pixels, or with `extra` bytes more than that holds.
fn image(w: u32, h: u32, extra: usize) -> Shape {
    Shape::new(ShapeData::Image {
        width: w,
        height: h,
        rgba: vec![0; (w * h * 4) as usize + extra],
    })
}

/// A delta that is refused whole leaves the stream as it was: its frame is `invalid_delta`, and
/// the next frame draws what the delta before it left (here the square, never the square `bad`
/// moves along with its fault). Gives the refusal's detail.
fn assert_refused_whole(good: Delta, mut bad: Delta) -> String {
    let (moved, r) = {
        let v = view(16, 16);
        let s = path_shape(polygon(&[
            frame(&v, [4.0, 4.0]),
            frame(&v, [12.0, 4.0]),
            frame(&v, [12.0, 12.0]),
            frame(&v, [4.0, 12.0]),
        ]));
        let r = path(0, s.key(), true, None);
        (s, r)
    };
    bad.upload.push(moved);
    bad.set.push(r);
    let mut held = delta(bad.seq + 1, None, vec![], vec![]);
    held.order = None;
    let deltas = [good.clone(), bad.clone(), held];
    let at_bad = arachne_oracle::decide::decide_at(&deltas, bad.seq, None).unwrap();
    assert_eq!(at_bad.header.status, STATUS_UNABLE);
    assert_eq!(at_bad.refused[0].reason, Reason::InvalidDelta.code());
    assert!(at_bad.fragments.is_empty());
    let after = decide(&deltas);
    assert_eq!(after.header.status, STATUS_DECIDED);
    assert_eq!(cover_lines(&after), cover_lines(&decide(&[good])));
    at_bad.refused[0].detail.clone()
}

#[test]
fn two_displays_sharing_a_rank_in_one_layer_refuse_the_delta_whole() {
    let (s, r) = square(0);
    let good = delta(1, Some(view(16, 16)), vec![s], vec![r]);
    let mut bad = delta(
        2,
        None,
        vec![],
        vec![display(1, 0, vec![]), display(2, 0, vec![])],
    );
    bad.order = Some(vec![entry(0), entry(1), entry(2)]);
    assert_refused_whole(good, bad);
}

#[test]
fn a_rank_shared_across_layers_or_views_or_by_one_display_drawn_twice_is_allowed() {
    let (s, r) = square(0);
    let mut d = delta(
        1,
        Some(view(16, 16)),
        vec![s],
        vec![
            r,
            display(1, 0, vec![at(XfLayer::Moving, 0), at(XfLayer::Moving, 3)]),
            display(2, 0, vec![]),
            display(3, 0, vec![]),
        ],
    );
    // Slots 1 and 2 share rank 0 in different layers; slot 3 shares it with slot 1 inside slot 1's
    // own view; slot 1 is drawn twice in one layer.
    d.order = Some(vec![
        at(XfLayer::Moving, 0),
        at(XfLayer::Moving, 1),
        at(XfLayer::Moving, 1),
        at(XfLayer::Static, 2),
    ]);
    let mut state = State::new();
    assert!(state.apply(&d).is_ok());
    assert_eq!(decide(&[d]).header.status, STATUS_DECIDED);
}

#[test]
fn two_displays_sharing_a_rank_in_a_displays_own_order_refuse_the_delta_whole() {
    let (s, r) = square(0);
    let good = delta(1, Some(view(16, 16)), vec![s], vec![r]);
    let inner = vec![at(XfLayer::Moving, 2), at(XfLayer::Moving, 3)];
    let mut bad = delta(
        2,
        None,
        vec![],
        vec![
            display(1, 0, inner),
            display(2, 5, vec![]),
            display(3, 5, vec![]),
        ],
    );
    bad.order = Some(vec![entry(0), entry(1)]);
    assert_refused_whole(good, bad);
}

#[test]
fn a_view_naming_a_background_it_lacks_refuses_the_delta_whole() {
    let (s, r) = square(0);
    let good = delta(1, Some(view(16, 16)), vec![s], vec![r]);
    let mut v = view(16, 16);
    v.background = Background::Image(Key([9; 16]));
    let mut bad = delta(2, Some(v), vec![], vec![]);
    bad.order = Some(vec![entry(0)]);
    assert_refused_whole(good, bad);
}

#[test]
fn a_display_whose_view_or_order_names_what_the_stream_lacks_refuses_the_delta_whole() {
    let (s, r) = square(0);
    let good = delta(1, Some(view(16, 16)), vec![s.clone()], vec![r.clone()]);
    let mut bad = delta(
        2,
        None,
        vec![],
        vec![display(1, 0, vec![at(XfLayer::Moving, 7)])],
    );
    bad.order = Some(vec![entry(0), entry(1)]);
    assert_refused_whole(good.clone(), bad);
    let mut lost = display(1, 0, vec![]);
    if let Body::Image(ImageSource::SubFrame { view, .. }) = &mut lost.body {
        view.background = Background::Image(Key([9; 16]));
    }
    let mut bad = delta(2, None, vec![], vec![lost]);
    bad.order = Some(vec![entry(0), entry(1)]);
    assert_refused_whole(good, bad);
}

#[test]
fn a_background_image_still_held_cannot_be_released_and_a_short_image_is_not_valid_xf() {
    let (s, r) = square(0);
    let bg = image(16, 16, 0);
    let mut v = view(16, 16);
    v.background = Background::Image(bg.key());
    let good = delta(1, Some(v), vec![s, bg.clone()], vec![r]);
    let mut bad = delta(2, None, vec![], vec![]);
    bad.release = vec![bg.key()];
    bad.order = Some(vec![entry(0)]);
    assert_refused_whole(good.clone(), bad);
    let mut bad = delta(2, None, vec![image(4, 4, 3)], vec![]);
    bad.order = Some(vec![entry(0)]);
    assert_refused_whole(good, bad);
}

#[test]
fn a_background_image_smaller_than_the_frame_refuses_the_frame_and_draws_nothing() {
    for (w, h, refused) in [
        (16, 16, false),
        (32, 20, false),
        (15, 16, true),
        (16, 8, true),
    ] {
        let (s, r) = square(0);
        let bg = image(w, h, 0);
        let mut v = view(16, 16);
        v.background = Background::Image(bg.key());
        let d = decide(&[delta(1, Some(v), vec![s, bg], vec![r])]);
        if refused {
            assert_eq!(d.header.status, STATUS_UNABLE, "{w} x {h}");
            assert_eq!(d.refused.len(), 1);
            assert_eq!(d.refused[0].reason, Reason::CoreRefuses.code());
            assert!(d.fragments.is_empty() && d.covers.is_empty());
        } else {
            assert_eq!(d.header.status, STATUS_DECIDED, "{w} x {h}");
            assert!(!d.fragments.is_empty());
        }
    }
}

#[test]
fn a_delta_releasing_a_key_the_stream_does_not_hold_refuses_the_delta_whole() {
    let (s, r) = square(0);
    let good = delta(1, Some(view(16, 16)), vec![s], vec![r]);
    let mut bad = delta(2, None, vec![], vec![]);
    bad.release = vec![Key([5; 16])];
    bad.order = Some(vec![entry(0)]);
    assert_refused_whole(good, bad);
}

#[test]
fn a_delta_removing_a_slot_the_stream_does_not_hold_refuses_the_delta_whole() {
    let (s, r) = square(0);
    let good = delta(1, Some(view(16, 16)), vec![s], vec![r]);
    let mut bad = delta(2, None, vec![], vec![]);
    bad.remove = vec![9];
    bad.order = Some(vec![entry(0)]);
    assert_refused_whole(good, bad);
}

#[test]
fn a_key_the_delta_uploads_may_be_released_and_a_slot_it_sets_removed() {
    let (s, r) = square(0);
    let mut state = State::new();
    state
        .apply(&delta(1, Some(view(16, 16)), vec![s.clone()], vec![r]))
        .unwrap();
    let extra = image(2, 2, 0);
    let mut d = delta(
        2,
        None,
        vec![extra.clone()],
        vec![path(5, s.key(), true, None)],
    );
    d.remove = vec![5];
    d.release = vec![extra.key()];
    d.order = Some(vec![entry(0)]);
    assert!(state.apply(&d).is_ok());
    assert!(!state.records.contains_key(&5) && !state.shapes.contains_key(&extra.key()));
}

#[test]
fn a_delta_faulty_twice_reports_the_fault_xf_checks_first() {
    // A release of a key neither held nor uploaded (apply's check 6) and an order naming a slot
    // the stream lacks (check 12): the refusal is the release.
    let (s, r) = square(0);
    let good = delta(1, Some(view(16, 16)), vec![s], vec![r]);
    let mut bad = delta(2, None, vec![], vec![]);
    bad.release = vec![Key([5; 16])];
    bad.order = Some(vec![entry(0), entry(7)]);
    let d = arachne_oracle::decide::decide_at(&[good, bad], 2, None).unwrap();
    assert_eq!(d.refused[0].reason, Reason::InvalidDelta.code());
    assert!(
        d.refused[0].detail.contains("releases"),
        "{}",
        d.refused[0].detail
    );
}

/// A shape under `key` though its content is other: one of a code the reader does not know, whose
/// key it keeps unchecked (XF.md, Decode), read off the wire with `key` written in for its own.
fn forged(key: Key) -> Shape {
    let own = Shape::new(ShapeData::Reserved {
        code: 200,
        payload: vec![1, 2, 3],
    });
    let mut bytes = wire::encode_delta(&delta(1, None, vec![own.clone()], vec![]));
    let at = bytes
        .windows(17)
        .position(|w| w[0] == 200 && w[1..] == own.key().0)
        .unwrap();
    bytes[at + 1..at + 17].copy_from_slice(&key.0);
    let shape = wire::decode_delta(&bytes).unwrap().upload.remove(0);
    assert_eq!(shape.key(), key);
    shape
}

#[test]
fn a_shape_uploaded_under_a_key_the_stream_holds_as_a_paint_table_refuses_the_delta_whole() {
    let (s, r) = square(0);
    let table = Stops::new(vec![[1.0, 0.0, 0.0, 1.0]]);
    let mut good = delta(1, Some(view(16, 16)), vec![s], vec![r]);
    good.upload_paints = vec![table.clone()];
    let mut bad = delta(2, None, vec![forged(table.key())], vec![]);
    bad.order = Some(vec![entry(0)]);
    assert_refused_whole(good, bad);
}

#[test]
fn a_shape_uploaded_under_a_key_the_stream_holds_with_other_content_refuses_the_delta_whole() {
    let (s, r) = square(0);
    let mut bad = delta(2, None, vec![forged(s.key())], vec![]);
    bad.order = Some(vec![entry(0)]);
    assert_refused_whole(delta(1, Some(view(16, 16)), vec![s], vec![r]), bad);
}

#[test]
fn a_paint_table_uploaded_under_a_key_the_stream_holds_as_a_shape_refuses_the_delta_whole() {
    let (s, r) = square(0);
    let table = Stops::new(vec![[1.0, 0.0, 0.0, 1.0]]);
    let good = delta(1, Some(view(16, 16)), vec![s, forged(table.key())], vec![r]);
    let mut bad = delta(2, None, vec![], vec![]);
    bad.upload_paints = vec![table];
    bad.order = Some(vec![entry(0)]);
    assert_refused_whole(good, bad);
}

// One test per check `apply` makes (XF.md, "Where each check is made", Apply), each naming the
// refusal it expects by its detail, so that the check's absence fails it even where a later check
// would still refuse the delta. Each delta a reader can carry goes through the wire first.

/// `d` encoded and read back: a delta the wire carries, past the reader's own checks.
fn wired(d: Delta) -> Delta {
    wire::decode_delta(&wire::encode_delta(&d)).unwrap()
}

/// `bad` refused whole after `good` (see `assert_refused_whole`), for a fault its detail names.
fn assert_refused_for(good: Delta, bad: Delta, detail: &str) {
    let got = assert_refused_whole(good, bad);
    assert!(got.contains(detail), "refused for {got:?}, not {detail:?}");
}

/// The square in slot 0, drawn.
fn held() -> Delta {
    let (s, r) = square(0);
    delta(1, Some(view(16, 16)), vec![s], vec![r])
}

/// A delta at 2 uploading `shapes` and setting `records`, its order the square's alone.
fn next(shapes: Vec<Shape>, records: Vec<Record>) -> Delta {
    let mut d = delta(2, None, shapes, records);
    d.order = Some(vec![entry(0)]);
    d
}

/// A triangle path shape, unlike the square's.
fn triangle() -> Shape {
    let v = view(16, 16);
    path_shape(polygon(&[
        frame(&v, [1.0, 1.0]),
        frame(&v, [5.0, 1.0]),
        frame(&v, [1.0, 5.0]),
    ]))
}

/// A two-face face group in `slot` over two squares' stacked points, its fill table `fill`.
fn faces(slot: u32, fill: Key) -> (Shape, Record) {
    let v = view(16, 16);
    let mut points = polygon(&[
        frame(&v, [1.0, 1.0]),
        frame(&v, [3.0, 1.0]),
        frame(&v, [3.0, 3.0]),
        frame(&v, [1.0, 3.0]),
    ]);
    points.extend(polygon(&[
        frame(&v, [5.0, 5.0]),
        frame(&v, [7.0, 5.0]),
        frame(&v, [7.0, 7.0]),
        frame(&v, [5.0, 7.0]),
    ]));
    let s = path_shape(points);
    let mut r = Record::new(RecordKind::FaceGroup, slot, 100 + slot);
    r.geometry.shape_a = Some(s.key());
    r.body = Body::FaceGroup(FaceGroup {
        uv: vec![[0, 0], [1, 0]],
        grid: [2, 1],
        fill,
        stroke: None,
        face_ends: Vec::new(),
        face_widths: Vec::new(),
    });
    (s, r)
}

fn rows(n: usize) -> Stops {
    Stops::new(vec![[1.0, 1.0, 1.0, 1.0]; n])
}

#[test]
fn apply_1_a_major_other_than_the_oracles() {
    let mut bad = next(vec![], vec![]);
    bad.version = (KNOWN_XF.0 + 1, 0);
    assert_refused_for(held(), bad, "XF major 2");
}

#[test]
fn apply_3_a_key_uploaded_twice() {
    let bad = next(vec![image(2, 2, 0), image(2, 2, 0)], vec![]);
    assert_refused_for(held(), wired(bad), "uploaded twice");
}

#[test]
fn apply_4_a_mesh_naming_a_vertex_it_lacks() {
    let mesh = Shape::new(ShapeData::Mesh {
        positions: vec![[0.0; 3]; 3],
        triangles: vec![[0, 1, 3]],
        normals: None,
    });
    assert_refused_for(
        held(),
        wired(next(vec![mesh], vec![])),
        "names a vertex it lacks",
    );
}

#[test]
fn apply_4_a_mesh_whose_normals_are_not_one_per_vertex() {
    // The wire reads one normal per vertex: only a delta built in memory carries this.
    let mesh = Shape::new(ShapeData::Mesh {
        positions: vec![[0.0; 3]; 3],
        triangles: vec![[0, 1, 2]],
        normals: Some(vec![[0.0, 0.0, 1.0]; 2]),
    });
    assert_refused_for(
        held(),
        next(vec![mesh], vec![]),
        "normals not one per vertex",
    );
}

/// The square drawn, and in slot 1, outside the order, `extra` drawing `shape`.
fn held_with(shape: Shape, extra: Record, paints: Vec<Stops>) -> Delta {
    let mut good = held();
    good.upload.push(shape);
    good.set.push(extra);
    good.upload_paints = paints;
    good
}

#[test]
fn apply_8_a_release_of_a_shape_a_held_record_uses() {
    let t = triangle();
    let mut good = held_with(t.clone(), path(1, t.key(), true, None), vec![]);
    good.order = Some(vec![entry(0)]);
    let mut bad = next(vec![], vec![]);
    bad.release = vec![t.key()];
    assert_refused_for(good, wired(bad), "which slot 1 uses");
}

#[test]
fn apply_8_a_release_of_a_paint_table_a_held_record_uses() {
    let t = triangle();
    let table = rows(3);
    let mut r = path(1, t.key(), false, None);
    r.paint.fill = Paint::Table(table.key());
    let mut good = held_with(t, r, vec![table.clone()]);
    good.order = Some(vec![entry(0)]);
    let mut bad = next(vec![], vec![]);
    bad.release = vec![table.key()];
    assert_refused_for(good, wired(bad), "which slot 1 uses");
}

#[test]
fn apply_8_a_release_of_a_held_displays_background() {
    let bg = image(8, 8, 0);
    let mut shown = display(1, 0, vec![]);
    if let Body::Image(ImageSource::SubFrame { view, .. }) = &mut shown.body {
        view.background = Background::Image(bg.key());
    }
    let mut good = held_with(bg.clone(), shown, vec![]);
    good.order = Some(vec![entry(0)]);
    let mut bad = next(vec![], vec![]);
    bad.release = vec![bg.key()];
    assert_refused_for(good, wired(bad), "which slot 1 uses");
}

#[test]
fn apply_9_a_held_displays_order_left_naming_a_removed_slot() {
    let t = triangle();
    let mut good = held_with(t.clone(), path(2, t.key(), true, None), vec![]);
    good.set.push(display(1, 0, vec![at(XfLayer::Moving, 2)]));
    good.order = Some(vec![entry(0)]);
    // The delta removes slot 2, which the held display's own order draws.
    let mut bad = next(vec![], vec![]);
    bad.remove = vec![2];
    assert_refused_for(good, wired(bad), "slot 1's display's order");
}

#[test]
fn apply_10_the_background_image_bit_and_fill_image_disagreeing() {
    // The reader refuses the disagreement: only a delta built in memory reaches apply with it.
    let (s, mut r) = square(1);
    r.features |= FEATURE_BACKGROUND_IMAGE;
    assert_refused_for(held(), next(vec![s], vec![r]), "background_image bit");
}

#[test]
fn apply_10_the_pre_projected_bit_and_its_tail_disagreeing() {
    let (s, mut r) = square(1);
    r.features |= FEATURE_PRE_PROJECTED;
    assert_refused_for(held(), next(vec![s], vec![r]), "pre_projected bit");
}

#[test]
fn apply_10_a_pre_projected_tail_on_a_kind_that_takes_none() {
    let (s, r) = faces(1, rows(2).key());
    let mut r = r;
    r.features |= FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: s.key(),
        gradient: None,
        fill: Paint::None,
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    let mut bad = next(vec![s], vec![r]);
    bad.upload_paints = vec![rows(2)];
    assert_refused_for(held(), bad, "tail on a face_group record");
}

#[test]
fn apply_10_a_kind_that_needs_a_shape_given_none() {
    let mut r = path(1, Key([0; 16]), true, None);
    r.geometry.shape_a = None;
    assert_refused_for(held(), wired(next(vec![], vec![r])), "given no shape");
}

#[test]
fn apply_10_a_shape_of_a_kind_the_record_cannot_take() {
    let picture = image(2, 2, 0);
    let r = path(1, picture.key(), true, None);
    assert_refused_for(held(), wired(next(vec![picture], vec![r])), "cannot take");
}

#[test]
fn apply_10_a_sub_frame_display_given_a_shape() {
    let picture = image(2, 2, 0);
    let mut r = display(1, 0, vec![]);
    r.geometry.shape_a = Some(picture.key());
    assert_refused_for(
        held(),
        wired(next(vec![picture], vec![r])),
        "sub-frame display given a shape",
    );
}

#[test]
fn apply_10_a_body_of_another_kinds() {
    // The reader reads the body a kind has: only a delta built in memory carries another's.
    let (s, mut r) = square(1);
    r.body = Body::Points(Points {
        colors: rows(1).key(),
        radius: 1.0,
    });
    let mut bad = next(vec![s], vec![r]);
    bad.upload_paints = vec![rows(1)];
    assert_refused_for(held(), bad, "another kind's body");
}

#[test]
fn apply_10_a_paint_table_the_stream_would_not_hold() {
    let t = triangle();
    let mut r = path(1, t.key(), false, None);
    r.paint.fill = Paint::Table(Key([3; 16]));
    assert_refused_for(held(), wired(next(vec![t], vec![r])), "names paint table");
}

#[test]
fn apply_10_a_sub_frame_view_with_soft_diffuse_light_out_of_range() {
    let mut r = display(1, 0, vec![]);
    if let Body::Image(ImageSource::SubFrame { view, .. }) = &mut r.body {
        view.light = LightModel::SoftDiffuse {
            ambient: 2.0,
            wrap: 0.0,
        };
    }
    assert_refused_for(
        held(),
        wired(next(vec![], vec![r])),
        "display's soft diffuse",
    );
}

#[test]
fn apply_10_a_sub_frame_view_naming_a_background_it_lacks() {
    let mut r = display(1, 0, vec![]);
    if let Body::Image(ImageSource::SubFrame { view, .. }) = &mut r.body {
        view.background = Background::Image(Key([9; 16]));
    }
    assert_refused_for(
        held(),
        wired(next(vec![], vec![r])),
        "display names background image",
    );
}

#[test]
fn apply_10_a_face_groups_fill_table_not_one_row_per_face() {
    // Two faces, one row: drawn, the second face would read a row the table lacks.
    let one = rows(1);
    let (s, r) = faces(1, one.key());
    let mut bad = next(vec![s], vec![r]);
    bad.upload_paints = vec![one];
    assert_refused_for(held(), wired(bad), "fill table holds 1 rows");
}

#[test]
fn apply_10_a_points_records_colour_table_not_one_row_per_point() {
    let v = view(16, 16);
    let s = path_shape(vec![frame(&v, [2.0, 2.0]), frame(&v, [4.0, 4.0])]);
    let one = rows(1);
    let mut r = Record::new(RecordKind::Points, 1, 101);
    r.geometry.shape_a = Some(s.key());
    r.body = Body::Points(Points {
        colors: one.key(),
        radius: 1.0,
    });
    let mut bad = next(vec![s], vec![r]);
    bad.upload_paints = vec![one];
    assert_refused_for(held(), wired(bad), "colour table holds 1 rows");
}

/// An instances record in slot 1 of the triangle, placed `n` times, its stroke table `stroke`
/// rows, its objects `objects`.
fn instances(n: usize, stroke: usize, objects: Vec<u32>) -> Delta {
    let t = triangle();
    let (fill, strokes) = (rows(n), rows(stroke));
    let mut r = Record::new(RecordKind::Instances, 1, 101);
    r.geometry.shape_a = Some(t.key());
    r.paint.fill = Paint::One([1.0; 4]);
    r.body = Body::Instances(Instances {
        placements: vec![arachne_core::xf::IDENTITY; n],
        fill: fill.key(),
        stroke: strokes.key(),
        objects,
    });
    let mut bad = next(vec![t], vec![r]);
    bad.upload_paints = if n == stroke {
        vec![fill]
    } else {
        vec![fill, strokes]
    };
    bad
}

#[test]
fn apply_10_an_instances_table_not_one_row_per_placement() {
    assert_refused_for(
        held(),
        wired(instances(2, 1, vec![1, 2])),
        "stroke table holds 1 rows",
    );
}

#[test]
fn apply_10_an_instances_record_whose_placements_and_objects_differ_in_number() {
    // The writer refuses to encode it: only a delta built in memory carries it.
    assert_refused_for(held(), instances(2, 2, vec![1]), "differ in number");
}

#[test]
fn apply_10_face_ends_not_rising() {
    let two = rows(2);
    let (s, mut r) = faces(1, two.key());
    if let Body::FaceGroup(f) = &mut r.body {
        f.face_ends = vec![32, 16];
    }
    let mut bad = next(vec![s], vec![r]);
    bad.upload_paints = vec![two];
    assert_refused_for(held(), wired(bad), "face ends");
}

#[test]
fn apply_10_face_widths_below_0() {
    let two = rows(2);
    let (s, mut r) = faces(1, two.key());
    if let Body::FaceGroup(f) = &mut r.body {
        f.face_widths = vec![1.0, -1.0];
    }
    let mut bad = next(vec![s], vec![r]);
    bad.upload_paints = vec![two];
    assert_refused_for(held(), wired(bad), "face widths");
}

#[test]
fn apply_10_joins_on_a_record_that_is_not_a_path() {
    let two = rows(2);
    let (s, mut r) = faces(1, two.key());
    r.joins = Some(vec![0; 8]);
    let mut bad = next(vec![s], vec![r]);
    bad.upload_paints = vec![two];
    assert_refused_for(held(), bad, "joins on a record that is not a path");
}

#[test]
fn apply_10_joins_not_one_per_cubic() {
    let t = triangle();
    let mut r = path(1, t.key(), true, None);
    r.joins = Some(vec![0]);
    assert_refused_for(held(), wired(next(vec![t], vec![r])), "one per cubic");
}

#[test]
fn apply_10_checks_each_record_set_not_only_the_last_in_its_slot() {
    // Slot 1 set twice: the first names a shape the stream lacks, the last is whole.
    let t = triangle();
    let lost = path(1, Key([7; 16]), true, None);
    let whole = path(1, t.key(), true, None);
    assert_refused_for(
        held(),
        wired(next(vec![t], vec![lost, whole])),
        "names shape",
    );
}

#[test]
fn apply_10_checks_the_order_of_a_display_set_and_set_again() {
    // Slot 1 set twice: first a display whose own order draws slot 7, which no record holds, then
    // one drawing nothing. Check 9 reads only the last; check 10 reads each.
    let lost = display(1, 0, vec![at(XfLayer::Moving, 7)]);
    let whole = display(1, 0, vec![]);
    assert_refused_for(
        held(),
        wired(next(vec![], vec![lost, whole])),
        "its display's order: it names slot 7 it lacks",
    );
}

#[test]
fn apply_10_comes_before_the_view_and_the_order() {
    // A record's fault (item 10) beside the delta's view's (11) and the order's (12): the
    // record's is the refusal.
    let one = rows(1);
    let (s, r) = faces(1, one.key());
    let mut bad = next(vec![s], vec![r]);
    bad.upload_paints = vec![one];
    let mut v = view(16, 16);
    v.light = LightModel::SoftDiffuse {
        ambient: 2.0,
        wrap: 0.0,
    };
    bad.view = Some(v);
    bad.order = Some(vec![entry(0), entry(7)]);
    assert_refused_for(held(), wired(bad), "fill table holds 1 rows");
}

#[test]
fn apply_11_the_views_soft_diffuse_light_out_of_range() {
    let mut v = view(16, 16);
    v.light = LightModel::SoftDiffuse {
        ambient: 0.5,
        wrap: -1.0,
    };
    let mut bad = next(vec![], vec![]);
    bad.view = Some(v);
    assert_refused_for(held(), wired(bad), "the view's soft diffuse");
}

#[test]
fn apply_11_the_views_background_not_an_image_shape() {
    let (s, _) = square(0);
    let mut v = view(16, 16);
    v.background = Background::Image(s.key());
    let mut bad = next(vec![], vec![]);
    bad.view = Some(v);
    assert_refused_for(held(), wired(bad), "not an image shape");
}

#[test]
fn apply_12_an_order_entry_naming_a_part_the_record_lacks() {
    let mut bad = next(vec![], vec![]);
    bad.order = Some(vec![OrderEntry {
        layer: XfLayer::Moving,
        slot: 0,
        part: 1,
    }]);
    assert_refused_for(held(), wired(bad), "names part 1 of slot 0");
}

#[test]
fn apply_10_a_fill_image_that_is_not_an_image_shape() {
    let (s, mut r) = square(1);
    r.features |= FEATURE_BACKGROUND_IMAGE;
    r.fill_image = Some(s.key());
    assert_refused_for(held(), wired(next(vec![s], vec![r])), "cannot take there");
}

#[test]
fn apply_10_a_tail_whose_points_are_not_a_path_shape() {
    let (s, mut r) = square(1);
    let picture = image(2, 2, 0);
    r.features |= FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: picture.key(),
        gradient: None,
        fill: Paint::None,
        stroke: Paint::None,
        background_stroke: Paint::None,
    });
    assert_refused_for(
        held(),
        wired(next(vec![s, picture], vec![r])),
        "cannot take there",
    );
}
