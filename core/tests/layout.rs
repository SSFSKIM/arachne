//! XF's binary layout, field by field, as `XF.md` states it. Every field's byte offset and every
//! enum code is a literal here, never taken from the writer, and each layout is checked both ways:
//! the writer puts every field at its offset, and the reader gives the value back from bytes built
//! by hand. A drift made the same way in the writer and the reader (a renumbered code, two fields
//! swapped) fails these tests even though a round trip still holds.

use arachne_core::xf::wire;
use arachne_core::xf::*;

/// One field of a layout: its byte offset in the item, its name, and its bytes.
type Field = (usize, &'static str, Vec<u8>);

fn u8(v: u8) -> Vec<u8> {
    vec![v]
}
fn u16(v: u16) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn u32(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn u64(v: u64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn i64(v: i64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn f64(v: f64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn f64s(vs: &[f64]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn u32s(vs: &[u32]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn key(k: &Key) -> Vec<u8> {
    k.0.to_vec()
}
/// An affine, row by row.
fn affine(a: &Affine) -> Vec<u8> {
    f64s(&a.concat())
}

/// The item a layout describes, after checking each literal offset against the sizes before it.
fn laid_out(fields: &[Field]) -> Vec<u8> {
    let mut out = Vec::new();
    for (offset, name, bytes) in fields {
        assert_eq!(*offset, out.len(), "the table's offset for {name}");
        out.extend_from_slice(bytes);
    }
    out
}

/// Asserts `actual` holds every field at its offset and nothing after the last, naming the first
/// field that differs.
fn assert_layout(actual: &[u8], fields: &[Field]) {
    for (offset, name, bytes) in fields {
        assert_eq!(
            actual.get(*offset..offset + bytes.len()),
            Some(bytes.as_slice()),
            "{name} at byte {offset}"
        );
    }
    let end = fields.last().map_or(0, |(o, _, b)| o + b.len());
    assert_eq!(actual.len(), end, "the item's length");
}

/// A delta's bytes around the given items, every other section empty: the frame
/// `a_delta_lays_out_its_sections_in_order` pins. Without a view, a shape item starts at byte 41
/// and a record item at byte 53; a view item starts at byte 33.
fn hand_delta(view: Option<&[u8]>, shapes: &[&[u8]], records: &[&[u8]]) -> Vec<u8> {
    let items = |out: &mut Vec<u8>, items: &[&[u8]]| {
        out.extend(u32(items.len() as u32));
        for item in items {
            out.extend(u32(item.len() as u32));
            out.extend_from_slice(item);
        }
    };
    let mut out = [u16(1), u16(0), i64(0), i64(1), u64(0)].concat();
    match view {
        None => out.extend(u8(0)),
        Some(v) => {
            out.extend(u8(1));
            out.extend(u32(v.len() as u32));
            out.extend_from_slice(v);
        }
    }
    out.extend(u32(0)); // objects
    items(&mut out, shapes);
    out.extend(u32(0)); // paint tables
    out.extend(u32(0)); // released
    items(&mut out, records);
    out.extend(u32(0)); // removed
    out.extend(u8(0)); // no order
    out
}

fn empty_delta() -> Delta {
    Delta::new(Time { num: 0, den: 1 }, 0)
}

const KA: Key = Key([0xa1; 16]);
const KB: Key = Key([0xb2; 16]);
const KC: Key = Key([0xc3; 16]);
const KD: Key = Key([0xd4; 16]);
const KE: Key = Key([0xe5; 16]);
const KF: Key = Key([0xf6; 16]);

const A1: Affine = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
];
const A2: Affine = [
    [101.0, 102.0, 103.0, 104.0],
    [105.0, 106.0, 107.0, 108.0],
    [109.0, 110.0, 111.0, 112.0],
];
const A3: Affine = [
    [201.0, 202.0, 203.0, 204.0],
    [205.0, 206.0, 207.0, 208.0],
    [209.0, 210.0, 211.0, 212.0],
];
const IDENTITY_BYTES: [f64; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

// ---------------------------------------------------------------------------------------------
// The file and the delta

/// An empty delta at `t_num / t_den`, sequence `seq`: every section absent or empty.
fn empty_delta_fields(t_num: i64, t_den: i64, seq: u64) -> Vec<Field> {
    vec![
        (0, "major", u16(1)),
        (2, "minor", u16(0)),
        (4, "t_num", i64(t_num)),
        (12, "t_den", i64(t_den)),
        (20, "seq", u64(seq)),
        (28, "has view", u8(0)),
        (29, "object count", u32(0)),
        (33, "shape count", u32(0)),
        (37, "paint table count", u32(0)),
        (41, "released count", u32(0)),
        (45, "record count", u32(0)),
        (49, "removed count", u32(0)),
        (53, "has order", u8(0)),
    ]
}

#[test]
fn a_file_is_its_magic_version_and_length_prefixed_deltas() {
    let deltas = [
        Delta::new(Time { num: 1, den: 15 }, 4),
        Delta::new(Time { num: -2, den: 3 }, 5),
    ];
    let fields = vec![
        (0, "magic", b"ARXF".to_vec()),
        (4, "major", u16(1)),
        (6, "minor", u16(0)),
        (8, "delta count", u32(2)),
        (12, "delta[0] length", u32(54)),
        (16, "delta[0]", laid_out(&empty_delta_fields(1, 15, 4))),
        (70, "delta[1] length", u32(54)),
        (74, "delta[1]", laid_out(&empty_delta_fields(-2, 3, 5))),
    ];
    assert_layout(&wire::encode_file(&deltas), &fields);
    assert_eq!(wire::decode_file(&laid_out(&fields)).unwrap(), deltas);
}

#[test]
fn a_delta_lays_out_its_sections_in_order() {
    let view = flat_view();
    let point = Shape::new(ShapeData::Path {
        points: vec![[1.0, 2.0, 3.0]],
    });
    let table = Stops::new(vec![[0.1, 0.2, 0.3, 0.4]]);
    let mut d = Delta::new(
        Time {
            num: -7,
            den: 9_000_000_000,
        },
        0x0102_0304_0506_0708,
    );
    d.version = (1, 3);
    d.view = Some(view);
    d.objects = vec![ObjectInfo {
        object: 0x0a0b_0c0d,
        stable_id: "Dot#2".into(),
        source: "s.py:9".into(),
    }];
    d.upload = vec![point.clone()];
    d.upload_paints = vec![table.clone()];
    d.release = vec![KB];
    d.set = vec![plain(RecordKind::Mesh)];
    d.remove = vec![5, 8];
    d.order = Some(vec![
        OrderEntry {
            layer: Layer::Static,
            slot: 1,
            part: 0,
        },
        OrderEntry {
            layer: Layer::Moving,
            slot: 2,
            part: 3,
        },
        OrderEntry {
            layer: Layer::Overlay,
            slot: 4,
            part: 5,
        },
    ]);
    let object = vec![
        (0, "object ref", u32(0x0a0b_0c0d)),
        (4, "stable_id length", u32(5)),
        (8, "stable_id", b"Dot#2".to_vec()),
        (13, "source length", u32(6)),
        (17, "source", b"s.py:9".to_vec()),
    ];
    let shape = vec![
        (0, "shape kind", u8(1)),
        (1, "shape key", key(&point.key())),
        (17, "point count", u32(1)),
        (21, "points", f64s(&[1.0, 2.0, 3.0])),
    ];
    let paint_table = vec![
        (0, "paint table key", key(&table.key())),
        (16, "row count", u32(1)),
        (20, "rows", f64s(&[0.1, 0.2, 0.3, 0.4])),
    ];
    let fields = vec![
        (0, "major", u16(1)),
        (2, "minor", u16(3)),
        (4, "t_num", i64(-7)),
        (12, "t_den", i64(9_000_000_000)),
        (20, "seq", u64(0x0102_0304_0506_0708)),
        (28, "has view", u8(1)),
        (29, "view length", u32(120)),
        (33, "view", laid_out(&flat_view_fields())),
        (153, "object count", u32(1)),
        (157, "object length", u32(23)),
        (161, "object", laid_out(&object)),
        (184, "shape count", u32(1)),
        (188, "shape length", u32(45)),
        (192, "shape", laid_out(&shape)),
        (237, "paint table count", u32(1)),
        (241, "paint table length", u32(52)),
        (245, "paint table", laid_out(&paint_table)),
        (297, "released count", u32(1)),
        (301, "released key", key(&KB)),
        (317, "record count", u32(1)),
        (321, "record length", u32(209)),
        (325, "record", laid_out(&plain_fields(6, 0, 0, 0))),
        (534, "removed count", u32(2)),
        (538, "removed[0]", u32(5)),
        (542, "removed[1]", u32(8)),
        (546, "has order", u8(1)),
        (547, "order count", u32(3)),
        (551, "order[0] layer (static)", u8(0)),
        (552, "order[0] slot", u32(1)),
        (556, "order[0] part", u32(0)),
        (560, "order[1] layer (moving)", u8(1)),
        (561, "order[1] slot", u32(2)),
        (565, "order[1] part", u32(3)),
        (569, "order[2] layer (overlay)", u8(2)),
        (570, "order[2] slot", u32(4)),
        (574, "order[2] part", u32(5)),
    ];
    assert_layout(&wire::encode_delta(&d), &fields);
    assert_eq!(wire::decode_delta(&laid_out(&fields)).unwrap(), d);
}

#[test]
fn the_hand_built_delta_frame_is_the_writers() {
    // `hand_delta` frames the items of the tests below; it must be the writer's frame.
    let mut d = empty_delta();
    d.set = vec![plain(RecordKind::Mesh)];
    let record = laid_out(&plain_fields(6, 0, 0, 0));
    assert_eq!(wire::encode_delta(&d), hand_delta(None, &[], &[&record]));
    assert_eq!(
        wire::encode_delta(&empty_delta()),
        laid_out(&empty_delta_fields(0, 1, 0))
    );
}

// ---------------------------------------------------------------------------------------------
// Views: every camera, both backgrounds, both light models

fn flat_view() -> View {
    View {
        pixel_width: 854,
        pixel_height: 480,
        frame_width: 14.0,
        frame_height: 7.875,
        frame_center: [0.5, -0.25, 0.125],
        mapping_centre: [-0.75, 0.625, -0.125],
        background: Background::Color([0.1, 0.2, 0.3, 0.4]),
        camera: Camera::Flat,
        cairo_line_width_multiple: 0.01,
        use_z_index: true,
        light: LightModel::Ce,
    }
}

fn flat_view_fields() -> Vec<Field> {
    vec![
        (0, "pixel_width", u32(854)),
        (4, "pixel_height", u32(480)),
        (8, "frame_width", f64(14.0)),
        (16, "frame_height", f64(7.875)),
        (24, "frame centre", f64s(&[0.5, -0.25, 0.125])),
        (48, "mapping_centre", f64s(&[-0.75, 0.625, -0.125])),
        (72, "background (colour)", u8(0)),
        (73, "background rgba", f64s(&[0.1, 0.2, 0.3, 0.4])),
        (105, "camera (flat)", u8(0)),
        (106, "cairo_line_width_multiple", f64(0.01)),
        (114, "use_z_index", u8(1)),
        (115, "light model (CE's)", u8(0)),
        (116, "light parameter count", u32(0)),
    ]
}

/// A view both ways, carried by a delta: its item starts at byte 33, and 25 bytes of empty
/// sections follow it.
fn view_both_ways(view: &View, fields: &[Field]) {
    let mut d = empty_delta();
    d.view = Some(view.clone());
    let written = wire::encode_delta(&d);
    assert_layout(&written[33..written.len() - 25], fields);
    let hand = hand_delta(Some(&laid_out(fields)), &[], &[]);
    assert_eq!(written, hand);
    assert_eq!(wire::decode_delta(&hand).unwrap().view.as_ref(), Some(view));
}

#[test]
fn a_flat_view_with_a_colour_and_ces_light() {
    view_both_ways(&flat_view(), &flat_view_fields());
}

#[test]
fn a_moving_view_with_an_image_and_another_light_model() {
    let view = View {
        background: Background::Image(KA),
        camera: Camera::Moving,
        use_z_index: false,
        light: LightModel::Other {
            code: 7,
            params: vec![0.25, 0.5],
        },
        ..flat_view()
    };
    let mut fields = flat_view_fields()[..6].to_vec();
    fields.extend([
        (72, "background (image)", u8(1)),
        (73, "background image key", key(&KA)),
        (89, "camera (moving)", u8(1)),
        (90, "cairo_line_width_multiple", f64(0.01)),
        (98, "use_z_index", u8(0)),
        (99, "light model code", u8(7)),
        (100, "light parameter count", u32(2)),
        (104, "light parameters", f64s(&[0.25, 0.5])),
    ]);
    view_both_ways(&view, &fields);
}

#[test]
fn a_view_in_soft_diffuse_light_is_light_model_1_with_its_two_parameters() {
    let view = View {
        light: LightModel::SoftDiffuse {
            ambient: 0.25,
            wrap: 0.5,
        },
        ..flat_view()
    };
    let mut fields = flat_view_fields()[..11].to_vec();
    fields.extend([
        (115, "light model (soft diffuse)", u8(1)),
        (116, "light parameter count", u32(2)),
        (120, "light parameters: ambient, wrap", f64s(&[0.25, 0.5])),
    ]);
    view_both_ways(&view, &fields);
}

#[test]
fn a_three_d_view_lays_out_every_camera_value() {
    let view = View {
        camera: Camera::ThreeD(ThreeDCamera {
            phi: 1.25,
            theta: -0.75,
            gamma: 0.125,
            focal_distance: 20.0,
            zoom: 1.5,
            exponential_projection: true,
            light_source: [-7.0, -9.0, 10.0],
            shading_factor: 0.2,
            should_apply_shading: false,
        }),
        ..flat_view()
    };
    let mut fields = flat_view_fields()[..8].to_vec();
    fields.extend([
        (105, "camera (3D)", u8(2)),
        (106, "phi", f64(1.25)),
        (114, "theta", f64(-0.75)),
        (122, "gamma", f64(0.125)),
        (130, "focal_distance", f64(20.0)),
        (138, "zoom", f64(1.5)),
        (146, "exponential_projection", u8(1)),
        (147, "light_source", f64s(&[-7.0, -9.0, 10.0])),
        (171, "shading_factor", f64(0.2)),
        (179, "should_apply_shading", u8(0)),
        (180, "cairo_line_width_multiple", f64(0.01)),
        (188, "use_z_index", u8(1)),
        (189, "light model (CE's)", u8(0)),
        (190, "light parameter count", u32(0)),
    ]);
    view_both_ways(&view, &fields);
}

// ---------------------------------------------------------------------------------------------
// Shapes: every kind

/// A shape both ways, carried by a delta: its item starts at byte 41, and 17 bytes of empty
/// sections follow it.
fn shape_both_ways(shape: &Shape, fields: &[Field]) {
    let mut d = empty_delta();
    d.upload = vec![shape.clone()];
    let written = wire::encode_delta(&d);
    assert_layout(&written[41..written.len() - 17], fields);
    let hand = hand_delta(None, &[&laid_out(fields)], &[]);
    assert_eq!(written, hand);
    assert_eq!(
        wire::decode_delta(&hand).unwrap().upload,
        vec![shape.clone()]
    );
}

#[test]
fn a_path_shape() {
    let shape = Shape::new(ShapeData::Path {
        points: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    });
    shape_both_ways(
        &shape,
        &[
            (0, "shape kind (path)", u8(1)),
            (1, "key", key(&shape.key())),
            (17, "point count", u32(2)),
            (21, "points", f64s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ],
    );
}

#[test]
fn a_mesh_shape_with_and_without_normals() {
    let positions = vec![[0.0, 0.5, 1.0], [1.5, 2.0, 2.5], [3.0, 3.5, 4.0]];
    let position_bytes = f64s(&[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    let lit = Shape::new(ShapeData::Mesh {
        positions: positions.clone(),
        triangles: vec![[0, 1, 2]],
        normals: Some(vec![[0.0, 0.0, 1.0], [0.0, 0.6, 0.8], [0.6, 0.0, 0.8]]),
    });
    shape_both_ways(
        &lit,
        &[
            (0, "shape kind (mesh)", u8(2)),
            (1, "key", key(&lit.key())),
            (17, "vertex count", u32(3)),
            (21, "positions", position_bytes.clone()),
            (93, "triangle count", u32(1)),
            (97, "triangles", u32s(&[0, 1, 2])),
            (109, "has normals", u8(1)),
            (
                110,
                "normals",
                f64s(&[0.0, 0.0, 1.0, 0.0, 0.6, 0.8, 0.6, 0.0, 0.8]),
            ),
        ],
    );
    let bare = Shape::new(ShapeData::Mesh {
        positions,
        triangles: vec![[2, 0, 1]],
        normals: None,
    });
    shape_both_ways(
        &bare,
        &[
            (0, "shape kind (mesh)", u8(2)),
            (1, "key", key(&bare.key())),
            (17, "vertex count", u32(3)),
            (21, "positions", position_bytes),
            (93, "triangle count", u32(1)),
            (97, "triangles", u32s(&[2, 0, 1])),
            (109, "has normals", u8(0)),
        ],
    );
}

#[test]
fn an_image_shape() {
    let shape = Shape::new(ShapeData::Image {
        width: 2,
        height: 1,
        rgba: vec![255, 0, 0, 255, 0, 0, 255, 128],
    });
    shape_both_ways(
        &shape,
        &[
            (0, "shape kind (image)", u8(3)),
            (1, "key", key(&shape.key())),
            (17, "width", u32(2)),
            (21, "height", u32(1)),
            (25, "rgba", vec![255, 0, 0, 255, 0, 0, 255, 128]),
        ],
    );
}

#[test]
fn the_reserved_shape_kinds() {
    let volume = Shape::new(ShapeData::Reserved {
        code: ShapeKind::Volume as u8,
        payload: vec![1, 2, 3],
    });
    shape_both_ways(
        &volume,
        &[
            (0, "shape kind (volume)", u8(4)),
            (1, "key", key(&volume.key())),
            (17, "payload", vec![1, 2, 3]),
        ],
    );
    let field = Shape::new(ShapeData::Reserved {
        code: ShapeKind::Field as u8,
        payload: vec![4, 5, 6, 7],
    });
    shape_both_ways(
        &field,
        &[
            (0, "shape kind (field)", u8(5)),
            (1, "key", key(&field.key())),
            (17, "payload", vec![4, 5, 6, 7]),
        ],
    );
}

// ---------------------------------------------------------------------------------------------
// Records: every fixed field, every enum code, every kind's body

/// A record both ways: the writer puts every field at its offset, and the reader, given the
/// laid-out bytes in a delta, gives the record back.
fn record_both_ways(record: &Record, fields: &[Field]) {
    assert_layout(&wire::encode_record(record), fields);
    let hand = hand_delta(None, &[], &[&laid_out(fields)]);
    assert_eq!(&wire::decode_delta(&hand).unwrap().set[0], record);
}

#[test]
fn a_path_record_lays_out_every_fixed_field() {
    let mut r = Record::new(RecordKind::Path, 0x0a0b_0c0d, 0x0102_0304);
    r.features = 0x1122_3344_5566_7789;
    r.geometry = Geometry {
        shape_a: Some(KA),
        place_a: A1,
        blend: Some(Blend {
            shape_b: KB,
            place_b: A2,
            t: 0.375,
        }),
        reveal: Some((0.125, 0.875)),
    };
    r.paint = PaintBlock {
        fill: Paint::One([0.1, 0.2, 0.3, 0.4]),
        stroke: Paint::Two([[0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 0.05, 0.15]]),
        background_stroke: Paint::Table(KC),
        stroke_width: 4.0,
        background_stroke_width: 1.5,
        gradient: Some([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]),
        joint: Joint::Round,
        cap: Cap::Square,
        paint_b: Some(PaintB {
            fill: Paint::One([0.25, 0.35, 0.45, 0.55]),
            stroke: Paint::None,
            background_stroke: Paint::Table(KD),
            stroke_width: 2.0,
            background_stroke_width: 2.5,
        }),
        mix: 0.625,
    };
    r.dash = Some(Dash {
        period: 0.5,
        duty: 0.25,
        phase: 0.0625,
    });
    r.depth = Depth {
        space: Space::World,
        z_index: -1.5,
        sort_point: [7.0, 8.0, 9.0],
        shade_in_3d: true,
        fixed_orientation_centre: Some([10.0, 11.0, 12.0]),
    };
    r.clip = Some((KE, A3));
    r.opacity = 0.75;
    r.layer = 0x0102;
    r.fill_image = Some(KF);
    record_both_ways(
        &r,
        &[
            (0, "record kind (path)", u8(1)),
            (1, "features", u64(0x1122_3344_5566_7789)),
            (9, "slot", u32(0x0a0b_0c0d)),
            (13, "object", u32(0x0102_0304)),
            (17, "geometry flags (shape_a, blend, reveal)", u8(7)),
            (18, "shape_a", key(&KA)),
            (34, "place_a", affine(&A1)),
            (130, "shape_b", key(&KB)),
            (146, "place_b", affine(&A2)),
            (242, "blend t", f64(0.375)),
            (250, "reveal lo", f64(0.125)),
            (258, "reveal hi", f64(0.875)),
            (266, "fill (one stop)", u8(1)),
            (267, "fill rgba", f64s(&[0.1, 0.2, 0.3, 0.4])),
            (299, "stroke (two stops)", u8(2)),
            (
                300,
                "stroke stops",
                f64s(&[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.05, 0.15]),
            ),
            (364, "background stroke (table)", u8(3)),
            (365, "background stroke key", key(&KC)),
            (381, "stroke_width", f64(4.0)),
            (389, "background_stroke_width", f64(1.5)),
            (397, "has gradient", u8(1)),
            (398, "gradient", f64s(&[-1.0, -2.0, -3.0, 4.0, 5.0, 6.0])),
            (446, "joint (round)", u8(1)),
            (447, "cap (square)", u8(3)),
            (448, "has paint_b", u8(1)),
            (449, "paint_b fill (one stop)", u8(1)),
            (450, "paint_b fill rgba", f64s(&[0.25, 0.35, 0.45, 0.55])),
            (482, "paint_b stroke (none)", u8(0)),
            (483, "paint_b background stroke (table)", u8(3)),
            (484, "paint_b background stroke key", key(&KD)),
            (500, "paint_b stroke_width", f64(2.0)),
            (508, "paint_b background_stroke_width", f64(2.5)),
            (516, "mix", f64(0.625)),
            (524, "has dash", u8(1)),
            (525, "dash period", f64(0.5)),
            (533, "dash duty", f64(0.25)),
            (541, "dash phase", f64(0.0625)),
            (549, "space (world)", u8(1)),
            (550, "z_index", f64(-1.5)),
            (558, "sort_point", f64s(&[7.0, 8.0, 9.0])),
            (582, "shade_in_3d", u8(1)),
            (583, "has fixed-orientation centre", u8(1)),
            (584, "fixed-orientation centre", f64s(&[10.0, 11.0, 12.0])),
            (608, "has clip", u8(1)),
            (609, "clip key", key(&KE)),
            (625, "clip placement", affine(&A3)),
            (721, "opacity", f64(0.75)),
            (729, "layer", u16(0x0102)),
            (731, "has fill_image", u8(1)),
            (732, "fill_image", key(&KF)),
        ],
    );
}

/// A record of `kind` in slot 7, object 70, with shape KA and the fixed fields set plainly but
/// each to a value of its own.
fn plain(kind: RecordKind) -> Record {
    let mut r = Record::new(kind, 7, 70);
    r.geometry.shape_a = Some(KA);
    r.paint.stroke_width = 0.5;
    r.paint.background_stroke_width = 0.25;
    r.paint.mix = 0.125;
    r.depth.z_index = 2.5;
    r.depth.sort_point = [0.5, 1.5, 2.5];
    r.opacity = 0.875;
    r.layer = 9;
    r
}

/// `plain`'s fixed fields, with the kind, joint, cap and space codes given; the body starts at
/// byte 209.
fn plain_fields(kind: u8, joint: u8, cap: u8, space: u8) -> Vec<Field> {
    vec![
        (0, "record kind", u8(kind)),
        (1, "features", u64(0)),
        (9, "slot", u32(7)),
        (13, "object", u32(70)),
        (17, "geometry flags (shape_a)", u8(1)),
        (18, "shape_a", key(&KA)),
        (34, "place_a", f64s(&IDENTITY_BYTES)),
        (130, "fill (none)", u8(0)),
        (131, "stroke (none)", u8(0)),
        (132, "background stroke (none)", u8(0)),
        (133, "stroke_width", f64(0.5)),
        (141, "background_stroke_width", f64(0.25)),
        (149, "has gradient", u8(0)),
        (150, "joint", u8(joint)),
        (151, "cap", u8(cap)),
        (152, "has paint_b", u8(0)),
        (153, "mix", f64(0.125)),
        (161, "has dash", u8(0)),
        (162, "space", u8(space)),
        (163, "z_index", f64(2.5)),
        (171, "sort_point", f64s(&[0.5, 1.5, 2.5])),
        (195, "shade_in_3d", u8(0)),
        (196, "has fixed-orientation centre", u8(0)),
        (197, "has clip", u8(0)),
        (198, "opacity", f64(0.875)),
        (206, "layer", u16(9)),
        (208, "has fill_image", u8(0)),
    ]
}

/// `plain` without a shape (a sub-frame display, an emission).
fn shapeless(kind: RecordKind) -> Record {
    let mut r = plain(kind);
    r.geometry.shape_a = None;
    r
}

/// `shapeless`'s fixed fields; the body starts at byte 193.
fn shapeless_fields(kind: u8) -> Vec<Field> {
    vec![
        (0, "record kind", u8(kind)),
        (1, "features", u64(0)),
        (9, "slot", u32(7)),
        (13, "object", u32(70)),
        (17, "geometry flags (none)", u8(0)),
        (18, "place_a", f64s(&IDENTITY_BYTES)),
        (114, "fill (none)", u8(0)),
        (115, "stroke (none)", u8(0)),
        (116, "background stroke (none)", u8(0)),
        (117, "stroke_width", f64(0.5)),
        (125, "background_stroke_width", f64(0.25)),
        (133, "has gradient", u8(0)),
        (134, "joint (auto)", u8(0)),
        (135, "cap (auto)", u8(0)),
        (136, "has paint_b", u8(0)),
        (137, "mix", f64(0.125)),
        (145, "has dash", u8(0)),
        (146, "space (screen)", u8(0)),
        (147, "z_index", f64(2.5)),
        (155, "sort_point", f64s(&[0.5, 1.5, 2.5])),
        (179, "shade_in_3d", u8(0)),
        (180, "has fixed-orientation centre", u8(0)),
        (181, "has clip", u8(0)),
        (182, "opacity", f64(0.875)),
        (190, "layer", u16(9)),
        (192, "has fill_image", u8(0)),
    ]
}

#[test]
fn every_joint_cap_and_space_code() {
    let joints = [
        (Joint::Auto, 0),
        (Joint::Round, 1),
        (Joint::Bevel, 2),
        (Joint::Miter, 3),
    ];
    for (joint, code) in joints {
        let mut r = plain(RecordKind::Mesh);
        r.paint.joint = joint;
        record_both_ways(&r, &plain_fields(6, code, 0, 0));
    }
    let caps = [
        (Cap::Auto, 0),
        (Cap::Round, 1),
        (Cap::Butt, 2),
        (Cap::Square, 3),
    ];
    for (cap, code) in caps {
        let mut r = plain(RecordKind::Mesh);
        r.paint.cap = cap;
        record_both_ways(&r, &plain_fields(6, 0, code, 0));
    }
    let spaces = [(Space::Screen, 0), (Space::World, 1), (Space::Overlay, 2)];
    for (space, code) in spaces {
        let mut r = plain(RecordKind::Mesh);
        r.depth.space = space;
        record_both_ways(&r, &plain_fields(6, 0, 0, code));
    }
}

#[test]
fn a_mesh_record_has_no_body() {
    record_both_ways(&plain(RecordKind::Mesh), &plain_fields(6, 0, 0, 0));
}

#[test]
fn background_image_is_feature_bit_zero_with_an_optional_key() {
    let mut r = plain(RecordKind::Path);
    r.features = FEATURE_BACKGROUND_IMAGE;
    r.fill_image = Some(KF);
    let mut fields = plain_fields(1, 0, 0, 0);
    fields[1] = (1, "features (background_image)", u64(1));
    fields.pop();
    fields.extend([
        (208, "has fill_image", u8(1)),
        (209, "fill_image", key(&KF)),
    ]);
    record_both_ways(&r, &fields);
}

#[test]
fn a_face_group_with_and_without_strokes() {
    let mut r = plain(RecordKind::FaceGroup);
    r.body = Body::FaceGroup(FaceGroup {
        uv: vec![[0, 1], [2, 3]],
        grid: [4, 5],
        fill: KB,
        stroke: Some(KC),
    });
    let mut fields = plain_fields(2, 0, 0, 0);
    fields.extend([
        (209, "face count", u32(2)),
        (213, "uv", u32s(&[0, 1, 2, 3])),
        (229, "grid_u", u32(4)),
        (233, "grid_v", u32(5)),
        (237, "fill key", key(&KB)),
        (253, "has stroke", u8(1)),
        (254, "stroke key", key(&KC)),
    ]);
    record_both_ways(&r, &fields);

    let Body::FaceGroup(f) = &mut r.body else {
        unreachable!()
    };
    f.stroke = None;
    fields.truncate(fields.len() - 2);
    fields.push((253, "has stroke", u8(0)));
    record_both_ways(&r, &fields);
}

#[test]
fn a_points_record() {
    let mut r = plain(RecordKind::Points);
    r.body = Body::Points(Points {
        colors: KB,
        radius: 2.25,
    });
    let mut fields = plain_fields(3, 0, 0, 0);
    fields.extend([(209, "colors key", key(&KB)), (225, "radius", f64(2.25))]);
    record_both_ways(&r, &fields);
}

#[test]
fn an_image_record_showing_a_picture() {
    let mut r = plain(RecordKind::Image);
    r.body = Body::Image(ImageSource::Picture {
        corners: [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 13.0],
        ],
        resampling: 3,
        opacity: 0.75,
    });
    let mut fields = plain_fields(4, 0, 0, 0);
    fields.extend([
        (209, "image source (picture)", u8(0)),
        (
            210,
            "corners",
            f64s(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0,
            ]),
        ),
        (306, "resampling", u8(3)),
        (307, "image opacity", f64(0.75)),
    ]);
    record_both_ways(&r, &fields);
}

#[test]
fn an_image_record_showing_a_sub_frame() {
    let mut r = shapeless(RecordKind::Image);
    r.body = Body::Image(ImageSource::SubFrame {
        rank: 0x0102_0304,
        corners: [
            [-1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 13.0],
        ],
        resampling: 2,
        opacity: 0.625,
        view: Box::new(flat_view()),
        order: vec![
            OrderEntry {
                layer: Layer::Overlay,
                slot: 3,
                part: 1,
            },
            OrderEntry {
                layer: Layer::Static,
                slot: 4,
                part: 0,
            },
        ],
    });
    let mut fields = shapeless_fields(4);
    fields.extend([
        (193, "image source (sub-frame)", u8(1)),
        (194, "capture rank", u32(0x0102_0304)),
        (
            198,
            "corners",
            f64s(&[
                -1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0,
            ]),
        ),
        (294, "resampling", u8(2)),
        (295, "image opacity", f64(0.625)),
        (303, "sub-frame view length", u32(120)),
        (307, "sub-frame view", laid_out(&flat_view_fields())),
        (427, "sub-frame order count", u32(2)),
        (431, "order[0] layer (overlay)", u8(2)),
        (432, "order[0] slot", u32(3)),
        (436, "order[0] part", u32(1)),
        (440, "order[1] layer (static)", u8(0)),
        (441, "order[1] slot", u32(4)),
        (445, "order[1] part", u32(0)),
    ]);
    record_both_ways(&r, &fields);
}

#[test]
fn an_instances_record() {
    let mut r = plain(RecordKind::Instances);
    r.body = Body::Instances(Instances {
        placements: vec![A1, A2],
        fill: KB,
        stroke: KC,
        objects: vec![301, 302],
    });
    let mut fields = plain_fields(5, 0, 0, 0);
    fields.extend([
        (209, "instance count", u32(2)),
        (213, "placements", [affine(&A1), affine(&A2)].concat()),
        (405, "fill key", key(&KB)),
        (421, "stroke key", key(&KC)),
        (437, "objects", u32s(&[301, 302])),
    ]);
    record_both_ways(&r, &fields);
}

#[test]
fn the_reserved_record_kinds_carry_their_payload_as_an_item() {
    for (kind, code) in [(RecordKind::Volume, 64), (RecordKind::Field, 65)] {
        let mut r = plain(kind);
        r.body = Body::Opaque(vec![9, 8, 7]);
        let mut fields = plain_fields(code, 0, 0, 0);
        fields.extend([
            (209, "payload length", u32(3)),
            (213, "payload", vec![9, 8, 7]),
        ]);
        record_both_ways(&r, &fields);
    }
    let mut r = shapeless(RecordKind::Emission);
    r.body = Body::Opaque(vec![6, 5]);
    let mut fields = shapeless_fields(66);
    fields.extend([
        (193, "payload length", u32(2)),
        (197, "payload", vec![6, 5]),
    ]);
    record_both_ways(&r, &fields);
}

#[test]
fn a_blend_without_a_reveal_sets_flag_bit_2_only() {
    let mut r = plain(RecordKind::Path);
    r.geometry.blend = Some(Blend {
        shape_b: KB,
        place_b: A2,
        t: 0.375,
    });
    record_both_ways(
        &r,
        &[
            (0, "record kind (path)", u8(1)),
            (1, "features", u64(0)),
            (9, "slot", u32(7)),
            (13, "object", u32(70)),
            (17, "geometry flags (shape_a, blend)", u8(3)),
            (18, "shape_a", key(&KA)),
            (34, "place_a", f64s(&IDENTITY_BYTES)),
            (130, "shape_b", key(&KB)),
            (146, "place_b", affine(&A2)),
            (242, "blend t", f64(0.375)),
            (250, "fill (none)", u8(0)),
            (251, "stroke (none)", u8(0)),
            (252, "background stroke (none)", u8(0)),
            (253, "stroke_width", f64(0.5)),
            (261, "background_stroke_width", f64(0.25)),
            (269, "has gradient", u8(0)),
            (270, "joint (auto)", u8(0)),
            (271, "cap (auto)", u8(0)),
            (272, "has paint_b", u8(0)),
            (273, "mix", f64(0.125)),
            (281, "has dash", u8(0)),
            (282, "space (screen)", u8(0)),
            (283, "z_index", f64(2.5)),
            (291, "sort_point", f64s(&[0.5, 1.5, 2.5])),
            (315, "shade_in_3d", u8(0)),
            (316, "has fixed-orientation centre", u8(0)),
            (317, "has clip", u8(0)),
            (318, "opacity", f64(0.875)),
            (326, "layer", u16(9)),
            (328, "has fill_image", u8(0)),
        ],
    );
}

#[test]
fn a_reveal_without_a_blend_sets_flag_bit_4_only() {
    let mut r = plain(RecordKind::Path);
    r.geometry.reveal = Some((0.125, 0.875));
    record_both_ways(
        &r,
        &[
            (0, "record kind (path)", u8(1)),
            (1, "features", u64(0)),
            (9, "slot", u32(7)),
            (13, "object", u32(70)),
            (17, "geometry flags (shape_a, reveal)", u8(5)),
            (18, "shape_a", key(&KA)),
            (34, "place_a", f64s(&IDENTITY_BYTES)),
            (130, "reveal lo", f64(0.125)),
            (138, "reveal hi", f64(0.875)),
            (146, "fill (none)", u8(0)),
            (147, "stroke (none)", u8(0)),
            (148, "background stroke (none)", u8(0)),
            (149, "stroke_width", f64(0.5)),
            (157, "background_stroke_width", f64(0.25)),
            (165, "has gradient", u8(0)),
            (166, "joint (auto)", u8(0)),
            (167, "cap (auto)", u8(0)),
            (168, "has paint_b", u8(0)),
            (169, "mix", f64(0.125)),
            (177, "has dash", u8(0)),
            (178, "space (screen)", u8(0)),
            (179, "z_index", f64(2.5)),
            (187, "sort_point", f64s(&[0.5, 1.5, 2.5])),
            (211, "shade_in_3d", u8(0)),
            (212, "has fixed-orientation centre", u8(0)),
            (213, "has clip", u8(0)),
            (214, "opacity", f64(0.875)),
            (222, "layer", u16(9)),
            (224, "has fill_image", u8(0)),
        ],
    );
}


#[test]
fn pre_projected_is_a_tail_after_the_body_not_a_geometry_flag() {
    let mut r = plain(RecordKind::Points);
    r.body = Body::Points(Points { colors: KB, radius: 2.25 });
    r.features = FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: KC,
        gradient: Some([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]),
        fill: Paint::One([0.1, 0.2, 0.3, 0.4]),
        stroke: Paint::Two([[0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 0.05, 0.15]]),
        background_stroke: Paint::Table(KD),
    });
    let mut fields = plain_fields(3, 0, 0, 0);
    fields[1] = (1, "features (pre_projected)", u64(2));
    fields.extend([
        (209, "colors key", key(&KB)),
        (225, "radius", f64(2.25)),
        (233, "pre_projected points", key(&KC)),
        (249, "pre_projected has gradient", u8(1)),
        (250, "pre_projected gradient", f64s(&[-1.0, -2.0, -3.0, 4.0, 5.0, 6.0])),
        (298, "pre_projected fill (one)", u8(1)),
        (299, "pre_projected fill rgba", f64s(&[0.1, 0.2, 0.3, 0.4])),
        (331, "pre_projected stroke (two)", u8(2)),
        (332, "pre_projected stroke rgba", f64s(&[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.05, 0.15])),
        (396, "pre_projected background (table)", u8(3)),
        (397, "pre_projected background key", key(&KD)),
    ]);
    record_both_ways(&r, &fields);
    r.pre_projected = Some(PreProjected {
        points: KC, gradient: None, fill: Paint::None,
        stroke: Paint::None, background_stroke: Paint::None,
    });
    fields.retain(|(offset, _, _)| *offset <= 225);
    fields.extend([
        (233, "pre_projected points", key(&KC)),
        (249, "pre_projected has gradient", u8(0)),
        (250, "pre_projected fill (none)", u8(0)),
        (251, "pre_projected stroke (none)", u8(0)),
        (252, "pre_projected background (none)", u8(0)),
    ]);
    record_both_ways(&r, &fields);
    r.features = 0;
    r.pre_projected = None;
    fields.retain(|(offset, _, _)| *offset < 233);
    fields[1] = (1, "features", u64(0));
    record_both_ways(&r, &fields);
}

#[test]
fn the_writer_asserts_both_pre_projected_presence_disagreements() {
    let mut r = plain(RecordKind::Path);
    r.features = FEATURE_PRE_PROJECTED;
    assert!(std::panic::catch_unwind(|| wire::encode_record(&r)).is_err());
    r.features = 0;
    r.pre_projected = Some(PreProjected {
        points: KC, gradient: None, fill: Paint::None,
        stroke: Paint::None, background_stroke: Paint::None,
    });
    assert!(std::panic::catch_unwind(|| wire::encode_record(&r)).is_err());
}
