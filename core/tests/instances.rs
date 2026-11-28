//! Instances draw what their members draw as records, one after another: a run of one prototype,
//! and a run repeating a group of leaves through parts (XF 1.6: an arrow's shaft and tip, each
//! shaft at a width of its own), byte for byte, beside ordinary records coplanar with them and translucent, split by a foreign record
//! into two runs, with strokes of other widths, joints and caps, under every light model; a mesh
//! prototype lit; and a cloud of coincident points drawn as its last point alone.

mod common;

use arachne_core::stream::{DrawError, Stream, StreamError};
use arachne_core::xf::wire::{decode_delta, encode_delta, encoded_version};
use arachne_core::xf::*;
use common::flat_view;

const W: u32 = 160;
const H: u32 = 90;

/// The light models a view may name, every one drawn the same way here.
fn lights() -> [LightModel; 2] {
    [
        LightModel::Ce,
        LightModel::SoftDiffuse {
            ambient: 0.35,
            wrap: 0.4,
        },
    ]
}

fn lit(mut view: View, light: &LightModel) -> View {
    view.light = light.clone();
    view
}

/// A view down onto the plane z = 0 at an angle, every world record on it coplanar.
fn looking_down() -> View {
    arachne_core::probes::view(W, H, 40f64.to_radians(), -60f64.to_radians())
}

/// Straight segments through `points` in CE's cubic layout.
fn lines(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        out.extend([a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
    }
    out
}

fn ring(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut closed = points.to_vec();
    closed.push(points[0]);
    lines(&closed)
}

fn path(points: Vec<[f64; 3]>) -> Shape {
    Shape::new(ShapeData::Path { points })
}

/// A turn by `angle` in the plane, a scale by `s`, then a shift.
fn placed(angle: f64, s: f64, dx: f64, dy: f64) -> Affine {
    let (c, n) = (libm::cos(angle) * s, libm::sin(angle) * s);
    [[c, -n, 0.0, dx], [n, c, 0.0, dy], [0.0, 0.0, 1.0, 0.0]]
}

/// An arrow's shaft, a unit line, and its tip, a triangle beyond it: two prototypes in unit space.
fn arrow() -> (Shape, Shape) {
    (
        path(lines(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
        path(ring(&[
            [1.0, -0.25, 0.0],
            [1.45, 0.0, 0.0],
            [1.0, 0.25, 0.0],
        ])),
    )
}

/// The shaft's paint block: a wide stroke, mitred and square-capped; the tip's: filled, with a
/// thin bevelled stroke of its own and a background stroke.
fn arrow_paints() -> (PaintBlock, PaintBlock) {
    let shaft = PaintBlock {
        stroke: Paint::One([1.0, 1.0, 1.0, 1.0]),
        stroke_width: 9.0,
        joint: Joint::Miter,
        cap: Cap::Square,
        ..PaintBlock::default()
    };
    let tip = PaintBlock {
        fill: Paint::One([1.0, 1.0, 1.0, 1.0]),
        stroke: Paint::One([1.0, 1.0, 1.0, 1.0]),
        stroke_width: 3.0,
        background_stroke: Paint::One([0.1, 0.1, 0.4, 0.8]),
        background_stroke_width: 7.0,
        joint: Joint::Bevel,
        cap: Cap::Round,
        ..PaintBlock::default()
    };
    (shaft, tip)
}

/// Row `i` of a table of `n` colours, some translucent.
fn colour(i: usize, salt: f64) -> [f64; 4] {
    let t = i as f64 * 0.37 + salt;
    [
        0.5 + 0.5 * libm::sin(t),
        0.5 + 0.5 * libm::sin(t + 2.0),
        0.5 + 0.5 * libm::sin(t + 4.0),
        if i % 3 == 1 { 0.55 } else { 1.0 },
    ]
}

/// Groups of an arrow placed to overlap each other and the records about them.
fn arrow_placements(groups: usize) -> Vec<Affine> {
    let mut found = Vec::new();
    for g in 0..groups {
        let t = g as f64;
        let place = placed(0.9 * t, 1.3 + 0.35 * t, -3.5 + 1.1 * t, -1.0 + 0.45 * t);
        // The shaft and the tip of one arrow share its placement.
        found.extend([place, place]);
    }
    found
}

/// What one test draws: a stream's records in order, its paints, the view, and the shapes its
/// instances records' parts name beside the records' own.
struct Scene {
    view: View,
    items: Vec<(Shape, Record)>,
    paints: Vec<Stops>,
    parts: Vec<Shape>,
}

impl Scene {
    fn draw(&self) -> Vec<u8> {
        let order: Vec<u32> = self.items.iter().map(|(_, r)| r.slot).collect();
        let mut d = arachne_core::probes::first_delta(self.view.clone(), &self.items, &order);
        for shape in &self.parts {
            if !d.upload.iter().any(|s| s.key() == shape.key()) {
                d.upload.push(shape.clone());
            }
        }
        d.upload_paints = self.paints.clone();
        let mut s = Stream::new();
        s.apply(&d).expect("the delta applies");
        s.draw(W, H, 2).unwrap()
    }
}

/// An ordinary record about the batch: a translucent filled square at z = 0, coplanar with the
/// arrows, at `slot`, or a stroked one.
fn ordinary(slot: u32, stroked: bool) -> (Shape, Record) {
    let square = path(ring(&[
        [-1.6, -1.4, 0.0],
        [1.9, -1.2, 0.0],
        [1.7, 1.5, 0.0],
        [-1.4, 1.3, 0.0],
    ]));
    let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
    r.geometry.shape_a = Some(square.key());
    if stroked {
        r.paint.stroke = Paint::One([0.9, 0.8, 0.1, 0.6]);
        r.paint.stroke_width = 12.0;
    } else {
        r.paint.fill = Paint::One([0.2, 0.6, 0.9, 0.5]);
    }
    r.depth.space = Space::World;
    (square, r)
}

/// An instances record of `placements`, member `i` drawing prototype `i % prototypes.len()`
/// with its paint block, its fill and stroke rows from `rows`, its stroke at `widths[i]` where
/// `widths` is not empty, at `slot`; its tables.
fn batch(
    slot: u32,
    prototypes: &[(Shape, PaintBlock)],
    placements: &[Affine],
    rows: &[[f64; 4]],
    widths: &[f64],
    objects: u32,
) -> (Shape, Record, Vec<Stops>) {
    let fills = Stops::new(rows.to_vec());
    let strokes = Stops::new(rows.iter().map(|c| [c[2], c[0], c[1], c[3]]).collect());
    let mut r = Record::new(RecordKind::Instances, slot, slot);
    r.geometry.shape_a = Some(prototypes[0].0.key());
    r.paint = prototypes[0].1.clone();
    r.depth.space = Space::World;
    let parts: Vec<Part> = prototypes[1..]
        .iter()
        .map(|(shape, paint)| Part {
            shape: shape.key(),
            paint: paint.clone(),
        })
        .collect();
    if !parts.is_empty() || !widths.is_empty() {
        r.features |= FEATURE_MEMBERS;
    }
    r.body = Body::Instances(Instances {
        placements: placements.to_vec(),
        fill: fills.key(),
        stroke: strokes.key(),
        objects: (0..placements.len() as u32).map(|i| objects + i).collect(),
        parts,
        widths: widths.to_vec(),
    });
    let tables = if strokes.key() == fills.key() {
        vec![fills]
    } else {
        vec![fills, strokes]
    };
    (prototypes[0].0.clone(), r, tables)
}

/// The members of a `batch` as path records, one after another from `slot`, each its prototype
/// at its placement, painted by its prototype's block with its own rows where that block fills or
/// strokes and its own width where `widths` gives one.
fn members(
    slot: u32,
    prototypes: &[(Shape, PaintBlock)],
    placements: &[Affine],
    rows: &[[f64; 4]],
    widths: &[f64],
) -> Vec<(Shape, Record)> {
    placements
        .iter()
        .enumerate()
        .map(|(i, place)| {
            let (shape, paint) = &prototypes[i % prototypes.len()];
            let mut r = Record::new(RecordKind::Path, slot + i as u32, 1000 + slot + i as u32);
            r.geometry.shape_a = Some(shape.key());
            r.geometry.place_a = *place;
            r.paint = paint.clone();
            let c = rows[i];
            if !matches!(paint.fill, Paint::None) {
                r.paint.fill = Paint::One(c);
            }
            if !matches!(paint.stroke, Paint::None) {
                r.paint.stroke = Paint::One([c[2], c[0], c[1], c[3]]);
            }
            if let Some(w) = widths.get(i) {
                r.paint.stroke_width = *w;
            }
            r.depth.space = Space::World;
            (shape.clone(), r)
        })
        .collect()
}

fn drawn_pixels(frame: &[u8]) -> usize {
    let background = &frame[..4];
    frame.chunks(4).filter(|p| p != &background).count()
}

/// Arrows as a run of groups of a shaft and a tip, between two ordinary records coplanar with
/// them (a fill under, a stroke over), translucent where they cross: the bytes their members draw
/// as records, on a flat camera and looking down on their plane, under each light model.
#[test]
fn a_run_of_arrows_draws_what_its_shafts_and_tips_draw_as_records() {
    let (shaft, tip) = arrow();
    let (shaft_paint, tip_paint) = arrow_paints();
    let prototypes = [(shaft, shaft_paint), (tip, tip_paint)];
    let placements = arrow_placements(6);
    let rows: Vec<[f64; 4]> = (0..placements.len()).map(|i| colour(i, 0.0)).collect();
    // CE thins each short arrow's shaft by its length; the tips keep their own width.
    let widths: Vec<f64> = (0..placements.len())
        .map(|i| {
            if i % 2 == 0 {
                4.0 + 1.5 * i as f64
            } else {
                3.0
            }
        })
        .collect();
    for light in lights() {
        for view in [flat_view(W, H, [0.0, 0.0, 0.0, 1.0]), looking_down()] {
            let view = lit(view, &light);
            let (shape, record, paints) = batch(2, &prototypes, &placements, &rows, &widths, 300);
            let batched = Scene {
                view: view.clone(),
                items: vec![ordinary(1, false), (shape, record), ordinary(3, true)],
                paints,
                parts: vec![prototypes[1].0.clone()],
            }
            .draw();
            let mut items = vec![ordinary(1, false)];
            items.extend(members(10, &prototypes, &placements, &rows, &widths));
            items.push(ordinary(3, true));
            let one_by_one = Scene {
                view: view.clone(),
                items,
                paints: Vec::new(),
                parts: Vec::new(),
            }
            .draw();
            assert!(
                batched == one_by_one,
                "{light:?}, {:?}: the run draws otherwise",
                view.camera
            );
            let n = drawn_pixels(&batched);
            assert!(n > 1000, "{n} pixels drawn");
        }
    }
}

/// A foreign record between two runs of one prototype (exact repeats: each member the shape at its
/// own placement): the two batches and the record draw what every member and the record draw one
/// after another, the record over the first run and under the second.
#[test]
fn a_foreign_record_splits_a_run_into_two_batches_drawing_as_its_members() {
    let (shaft, _) = arrow();
    let paint = PaintBlock {
        stroke: Paint::One([1.0; 4]),
        stroke_width: 14.0,
        joint: Joint::Round,
        cap: Cap::Butt,
        ..PaintBlock::default()
    };
    let prototypes = [(shaft, paint)];
    let placements: Vec<Affine> = (0..8)
        .map(|g| placed(0.4 * g as f64, 2.5, -3.0 + 0.8 * g as f64, -1.5))
        .collect();
    let rows: Vec<[f64; 4]> = (0..8).map(|i| colour(i, 1.0)).collect();
    for light in lights() {
        let view = lit(looking_down(), &light);
        let (s1, r1, mut paints) = batch(2, &prototypes, &placements[..4], &rows[..4], &[], 300);
        let (s2, r2, more) = batch(4, &prototypes, &placements[4..], &rows[4..], &[], 304);
        paints.extend(more);
        let batched = Scene {
            view: view.clone(),
            items: vec![(s1, r1), ordinary(3, false), (s2, r2)],
            paints,
            parts: Vec::new(),
        }
        .draw();
        let mut items = members(10, &prototypes, &placements[..4], &rows[..4], &[]);
        items.push(ordinary(3, false));
        items.extend(members(20, &prototypes, &placements[4..], &rows[4..], &[]));
        let one_by_one = Scene {
            view,
            items,
            paints: Vec::new(),
            parts: Vec::new(),
        }
        .draw();
        assert!(batched == one_by_one, "{light:?}");
        let n = drawn_pixels(&batched);
        assert!(n > 600, "{n} pixels drawn");
    }
}

/// Shaded spheres as instances of one mesh, translucent ones among them, around an ordinary
/// sphere: the bytes the members draw as mesh records, under each light model.
#[test]
fn shaded_mesh_instances_draw_what_their_members_draw_under_each_light() {
    let sphere = common::meshes::sphere([0.0; 3], 0.6, 6, 10);
    let placements: Vec<Affine> = (0..9)
        .map(|i| {
            let t = i as f64;
            let mut m = placed(0.3 * t, 0.7 + 0.1 * t, -3.0 + 0.75 * t, -1.2 + 0.3 * t);
            m[2][3] = 0.4 * libm::sin(t);
            m
        })
        .collect();
    let rows: Vec<[f64; 4]> = (0..9).map(|i| colour(i, 2.0)).collect();
    let fill = PaintBlock {
        fill: Paint::One([1.0; 4]),
        ..PaintBlock::default()
    };
    let centre = {
        let mut r = common::meshes::mesh_record(1, &sphere, [0.8, 0.8, 0.8, 1.0], Space::World);
        r.depth.shade_in_3d = true;
        r.geometry.place_a = placed(0.0, 1.6, 0.0, 0.0);
        (sphere.clone(), r)
    };
    for light in lights() {
        let view = lit(looking_down(), &light);
        let (s, mut r, paints) = batch(
            2,
            &[(sphere.clone(), fill.clone())],
            &placements,
            &rows,
            &[],
            300,
        );
        r.depth.shade_in_3d = true;
        let batched = Scene {
            view: view.clone(),
            items: vec![centre.clone(), (s, r)],
            paints,
            parts: Vec::new(),
        }
        .draw();
        let mut items = vec![centre.clone()];
        for (i, place) in placements.iter().enumerate() {
            let mut m = common::meshes::mesh_record(10 + i as u32, &sphere, rows[i], Space::World);
            m.geometry.place_a = *place;
            m.depth.shade_in_3d = true;
            items.push((sphere.clone(), m));
        }
        let one_by_one = Scene {
            view,
            items,
            paints: Vec::new(),
            parts: Vec::new(),
        }
        .draw();
        assert!(batched == one_by_one, "{light:?}");
        let n = drawn_pixels(&batched);
        assert!(n > 600, "{n} pixels drawn");
    }
}

/// Parts and widths on the wire (XF 1.6): read back as written, the delta at minor 6 and the
/// record's members bit set; a record whose bit and member styles disagree, widths not one per
/// placement or negative, or placements that are not whole groups, refused when applied.
#[test]
fn parts_travel_on_the_wire_and_are_checked_when_applied() {
    let (shaft, tip) = arrow();
    let (shaft_paint, tip_paint) = arrow_paints();
    let prototypes = [(shaft.clone(), shaft_paint), (tip.clone(), tip_paint)];
    let placements = arrow_placements(3);
    let rows: Vec<[f64; 4]> = (0..6).map(|i| colour(i, 0.0)).collect();
    let widths = vec![2.0; 6];
    let (_, record, paints) = batch(2, &prototypes, &placements, &rows, &widths, 300);
    let delta = |record: Record| {
        let mut d = arachne_core::probes::first_delta(
            flat_view(W, H, [0.0, 0.0, 0.0, 1.0]),
            &[(shaft.clone(), record)],
            &[2],
        );
        d.upload.push(tip.clone());
        d.upload_paints = paints.clone();
        d
    };
    let d = delta(record.clone());
    assert_eq!(encoded_version(&d), (1, 6));
    let read = decode_delta(&encode_delta(&d)).unwrap();
    let mut expected = d.clone();
    expected.version = (1, 6);
    assert_eq!(read, expected);
    assert!(Stream::new().apply(&read).is_ok());

    let invalid = |r: Record| match Stream::new().apply(&delta(r)) {
        Err(StreamError::Invalid(why)) => why,
        other => panic!("applied: {other:?}"),
    };
    let mut unmarked = record.clone();
    unmarked.features = 0;
    assert!(invalid(unmarked).contains("members feature bit"));
    let mut short = record.clone();
    if let Body::Instances(i) = &mut short.body {
        i.widths.pop();
    }
    assert!(invalid(short).contains("6 placements and 5 widths"));
    let mut negative = record.clone();
    if let Body::Instances(i) = &mut negative.body {
        i.widths[3] = -1.0;
    }
    assert!(invalid(negative).contains("stroke width"));
    let mut odd = record.clone();
    if let Body::Instances(i) = &mut odd.body {
        i.placements.pop();
        i.objects.pop();
        i.widths.pop();
    }
    // Five placements: the tables' six rows are refused first, so give them five.
    let five = Stops::new(rows[..5].to_vec());
    if let Body::Instances(i) = &mut odd.body {
        i.fill = five.key();
        i.stroke = five.key();
    }
    let mut d = delta(odd);
    d.upload_paints.push(five);
    match Stream::new().apply(&d) {
        Err(StreamError::Invalid(why)) => assert!(why.contains("whole groups"), "{why}"),
        other => panic!("applied: {other:?}"),
    }
}

/// An XF 1.5 reader (this reader at minor 5: what XF 1.6 appended skipped) reads a delta with
/// parts and widths as one plain instances record carrying a feature bit it does not know, which
/// its drawer refuses, and reads every other record and section as written: the record's own
/// bytes are XF 1.5's, its member styles in a section of the delta after XF 1.5's last.
#[test]
fn an_xf_1_5_reader_skips_parts_and_widths_and_its_drawer_refuses_the_record() {
    let (shaft, tip) = arrow();
    let (shaft_paint, tip_paint) = arrow_paints();
    let prototypes = [(shaft.clone(), shaft_paint), (tip.clone(), tip_paint)];
    let placements = arrow_placements(3);
    let rows: Vec<[f64; 4]> = (0..6).map(|i| colour(i, 0.0)).collect();
    let (_, record, paints) = batch(2, &prototypes, &placements, &rows, &[2.0; 6], 300);
    let mut d = arachne_core::probes::first_delta(
        flat_view(W, H, [0.0, 0.0, 0.0, 1.0]),
        &[(shaft.clone(), record.clone()), ordinary(3, false)],
        &[2, 3],
    );
    d.upload.push(tip.clone());
    d.upload_paints = paints;
    let bytes = encode_delta(&d);
    let old = arachne_core::xf::wire::decode_delta_as(&bytes, 5).expect("a 1.5 reader reads it");
    let mut plain = record.clone();
    if let Body::Instances(i) = &mut plain.body {
        i.parts.clear();
        i.widths.clear();
    }
    assert_eq!(old.set[0], plain);
    assert_eq!(old.set[1], d.set[1]);
    assert_eq!((old.upload.len(), old.order), (d.upload.len(), d.order.clone()));
    // The bit XF 1.5 does not know: its drawer refuses the frame rather than drawing every member
    // as the first prototype.
    const XF_1_5_FEATURES: u64 = FEATURE_BACKGROUND_IMAGE | FEATURE_PRE_PROJECTED;
    assert_ne!(old.set[0].features & !XF_1_5_FEATURES, 0);
    // This build reads the section back onto the record.
    assert_eq!(decode_delta(&bytes).unwrap().set[0], record);
}

/// A drawer that does not know a feature bit refuses the record whole rather than drawing it
/// partly: so an instances record's parts are never drawn as copies of its first prototype by a
/// drawer of XF 1.5, which does not know the members bit. This build knows it, and refuses the bit
/// above it.
#[test]
fn an_unknown_feature_bit_on_an_instances_record_is_refused_when_drawn() {
    let (shaft, _) = arrow();
    let (paint, _) = arrow_paints();
    let (shape, mut record, paints) =
        batch(2, &[(shaft, paint)], &[IDENTITY], &[[1.0; 4]], &[], 300);
    record.features |= FEATURE_MEMBERS << 1;
    let mut d = arachne_core::probes::first_delta(
        flat_view(W, H, [0.0, 0.0, 0.0, 1.0]),
        &[(shape, record)],
        &[2],
    );
    d.upload_paints = paints;
    let mut s = Stream::new();
    s.apply(&d).unwrap();
    assert!(matches!(s.draw(W, H, 1), Err(DrawError::Unsupported(_))));
}

/// A prototype's paint given as a table of one stop draws as that stop given inline, for the
/// record's prototype and a part's alike, path and mesh; a part's gradient (a table of two stops)
/// is refused, never drawn without its paint.
#[test]
fn a_prototype_s_paint_in_a_table_of_one_stop_draws_as_the_stop() {
    let (shaft, tip) = arrow();
    let (shaft_paint, tip_paint) = arrow_paints();
    let placements = arrow_placements(4);
    let rows: Vec<[f64; 4]> = (0..placements.len()).map(|i| colour(i, 0.3)).collect();
    let tabled = |paint: &Paint, tables: &mut Vec<Stops>| match paint {
        Paint::One(c) => {
            let t = Stops::new(vec![*c]);
            let key = t.key();
            tables.push(t);
            Paint::Table(key)
        }
        other => other.clone(),
    };
    let draw = |shaft_paint: PaintBlock, tip_paint: PaintBlock, more: Vec<Stops>| {
        let prototypes = [(shaft.clone(), shaft_paint), (tip.clone(), tip_paint)];
        let (shape, record, mut paints) = batch(2, &prototypes, &placements, &rows, &[], 300);
        paints.extend(more);
        let mut d = arachne_core::probes::first_delta(
            looking_down(),
            &[(shape, record)],
            &[2],
        );
        d.upload.push(tip.clone());
        for t in paints {
            if !d.upload_paints.iter().any(|u| u.key() == t.key()) {
                d.upload_paints.push(t);
            }
        }
        let mut s = Stream::new();
        s.apply(&d).expect("the delta applies");
        s.draw(W, H, 2)
    };
    let inline = draw(shaft_paint.clone(), tip_paint.clone(), Vec::new()).unwrap();
    let mut tables = Vec::new();
    let mut shaft_tabled = shaft_paint.clone();
    shaft_tabled.stroke = tabled(&shaft_paint.stroke, &mut tables);
    let mut tip_tabled = tip_paint.clone();
    tip_tabled.fill = tabled(&tip_paint.fill, &mut tables);
    tip_tabled.stroke = tabled(&tip_paint.stroke, &mut tables);
    tip_tabled.background_stroke = tabled(&tip_paint.background_stroke, &mut tables);
    let from_tables = draw(shaft_tabled.clone(), tip_tabled.clone(), tables.clone()).unwrap();
    assert!(inline == from_tables, "a one-stop table draws otherwise");
    assert!(drawn_pixels(&inline) > 300, "{} pixels", drawn_pixels(&inline));
    let two = Stops::new(vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]);
    let mut graded = tip_tabled;
    graded.fill = Paint::Table(two.key());
    tables.push(two);
    assert!(matches!(
        draw(shaft_tabled, graded, tables),
        Err(DrawError::Unsupported(_))
    ));

    // A mesh prototype filled from a table of one stop.
    let ball = common::meshes::sphere([0.0, 0.0, 0.0], 1.0, 6, 10);
    let mesh = |fill: Paint, more: Vec<Stops>| {
        let (shape, mut record, mut paints) = batch(
            2,
            &[(
                ball.clone(),
                PaintBlock {
                    fill,
                    ..PaintBlock::default()
                },
            )],
            &[placed(0.0, 0.8, -1.0, 0.0), placed(0.0, 0.8, 1.2, 0.3)],
            &rows[..2],
            &[],
            300,
        );
        record.depth.shade_in_3d = true;
        paints.extend(more);
        let mut d = arachne_core::probes::first_delta(looking_down(), &[(shape, record)], &[2]);
        d.upload_paints = paints;
        let mut s = Stream::new();
        s.apply(&d).expect("the delta applies");
        s.draw(W, H, 2).unwrap()
    };
    let white = Stops::new(vec![[1.0; 4]]);
    let key = white.key();
    let inline = mesh(Paint::One([1.0; 4]), Vec::new());
    assert!(inline == mesh(Paint::Table(key), vec![white]));
    assert!(drawn_pixels(&inline) > 300);
}

/// A thousand coincident points of one cloud, each a 64-pixel block: the frame its last point
/// draws alone. (The opaque pass keeps one block's fragment per pixel: `composite.rs`.)
#[test]
fn a_thousand_coincident_points_draw_as_the_last_alone() {
    let cloud = |n: usize| {
        let shape = path(vec![[0.3, -0.2, 0.0]; n]);
        let colours = Stops::new(
            (0..n)
                .map(|i| {
                    let c = colour(i + 1000 - n, 0.5);
                    [c[0], c[1], c[2], 1.0]
                })
                .collect(),
        );
        let mut r = Record::new(RecordKind::Points, 1, 101);
        r.geometry.shape_a = Some(shape.key());
        r.depth.space = Space::World;
        r.body = Body::Points(Points {
            colors: colours.key(),
            radius: 64.0,
        });
        (shape, r, colours)
    };
    for view in [flat_view(W, H, [0.0, 0.0, 0.0, 1.0]), looking_down()] {
        let (shape, record, colours) = cloud(1000);
        let all = Scene {
            view: view.clone(),
            items: vec![(shape, record)],
            paints: vec![colours],
            parts: Vec::new(),
        }
        .draw();
        let (shape, record, colours) = cloud(1);
        let last = Scene {
            view,
            items: vec![(shape, record)],
            paints: vec![colours],
            parts: Vec::new(),
        }
        .draw();
        assert!(all == last);
        assert!(drawn_pixels(&all) >= 64 * 64 - 64 * 2);
    }
}
