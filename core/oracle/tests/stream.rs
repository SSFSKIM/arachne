//! The stream state and what the oracle refuses (contract §11): a frame it cannot judge is
//! `truth_unable` whole, never drawn partly.

mod common;

use arachne_core::xf::{
    Background, Body, Delta, ImageSource, Key, Layer as XfLayer, LightModel, OrderEntry, Record,
    RecordKind, Shape, ShapeData, Space, Stops, XF_VERSION, wire,
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
    assert_eq!(KNOWN_XF, XF_VERSION);
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
    first.version = (1, 5);
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
/// moves along with its fault).
fn assert_refused_whole(good: Delta, mut bad: Delta) {
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
