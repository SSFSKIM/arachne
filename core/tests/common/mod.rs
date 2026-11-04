//! Engine-authored XF content shared by the tests and the cross-build identity example.
#![allow(dead_code)]

pub mod meshes;
pub mod resolve;

use arachne_core::xf::*;

pub fn flat_view(width: u32, height: u32, background: [f64; 4]) -> View {
    View {
        pixel_width: width,
        pixel_height: height,
        frame_width: 14.222222222222221,
        frame_height: 8.0,
        frame_center: [0.0; 3],
        mapping_centre: [0.0; 3],
        background: Background::Color(background),
        camera: Camera::Flat,
        cairo_line_width_multiple: 0.01,
        use_z_index: true,
        light: LightModel::Ce,
    }
}

pub fn three_d_view(width: u32, height: u32) -> View {
    View {
        camera: Camera::ThreeD(ThreeDCamera {
            phi: 1.2217304763960306,
            theta: -std::f64::consts::FRAC_PI_4,
            gamma: 0.1,
            focal_distance: 20.0,
            zoom: 1.0,
            exponential_projection: false,
            light_source: [-7.0, -9.0, 10.0],
            shading_factor: 0.2,
            should_apply_shading: true,
        }),
        frame_center: [0.5, -0.25, 0.0],
        ..flat_view(width, height, [0.0, 0.0, 0.0, 1.0])
    }
}

fn place(dx: f64, dy: f64, dz: f64) -> Affine {
    [
        [1.0, 0.0, 0.0, dx],
        [0.0, 1.0, 0.0, dy],
        [0.0, 0.0, 1.0, dz],
    ]
}

/// Encoded XF with the first occurrence of key `from` replaced by `to`: a stated key that is not
/// its content's, which only the binary form can carry.
pub fn restate_key(bytes: &mut [u8], from: Key, to: Key) {
    let at = bytes
        .windows(16)
        .position(|w| w == from.0)
        .expect("the key is in the bytes");
    bytes[at..at + 16].copy_from_slice(&to.0);
}

/// A square in CE's cubic layout.
pub fn square_points(half: f64) -> Vec<[f64; 3]> {
    let c = [
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ];
    let mut points = Vec::new();
    for i in 0..4 {
        let a = c[i];
        let b = c[(i + 1) % 4];
        let lerp = |t: f64| {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };
        points.extend([a, lerp(1.0 / 3.0), lerp(2.0 / 3.0), b]);
    }
    points
}

/// Three deltas that between them give every field XF 1 has a value distinct from its neighbours',
/// use every enum code, and take both branches of every optional field and flag: the first sets
/// everything, the second changes the view and removes and releases, the third re-sets a face group
/// without strokes and leaves the view and the order as they are.
pub fn sample_deltas() -> Vec<Delta> {
    let square = Shape::new(ShapeData::Path {
        points: square_points(1.0),
    });
    let wide = Shape::new(ShapeData::Path {
        points: square_points(2.0),
    });
    let tri = Shape::new(ShapeData::Mesh {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 1.0, -0.5]],
        triangles: vec![[0, 1, 2]],
        normals: Some(vec![[0.0, 0.0, 1.0], [0.0, 0.6, 0.8], [0.6, 0.0, 0.8]]),
    });
    let bare = Shape::new(ShapeData::Mesh {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 2, 1]],
        normals: None,
    });
    let picture = Shape::new(ShapeData::Image {
        width: 2,
        height: 1,
        rgba: vec![255, 0, 0, 255, 0, 0, 255, 128],
    });
    let volume = Shape::new(ShapeData::Reserved {
        code: ShapeKind::Volume as u8,
        payload: vec![1, 2, 3],
    });
    let field = Shape::new(ShapeData::Reserved {
        code: ShapeKind::Field as u8,
        payload: vec![4, 5, 6, 7],
    });
    let gradient = Stops::new(vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.5],
    ]);
    let per_face = Stops::new(vec![[0.2, 0.4, 0.6, 1.0], [0.6, 0.4, 0.2, 0.75]]);
    let face_strokes = Stops::new(vec![[0.1, 0.0, 0.3, 1.0], [0.0, 0.9, 0.7, 0.5]]);
    let per_point = Stops::new(vec![[1.0, 1.0, 1.0, 1.0]; 3]);
    let instance_strokes = Stops::new(vec![
        [0.9, 0.1, 0.1, 1.0],
        [0.1, 0.9, 0.1, 1.0],
        [0.1, 0.1, 0.9, 1.0],
    ]);

    let mut path = Record::new(RecordKind::Path, 1, 101);
    path.features = FEATURE_BACKGROUND_IMAGE;
    path.fill_image = Some(picture.key());
    path.geometry = Geometry {
        shape_a: Some(square.key()),
        place_a: place(0.5, 0.0, 0.0),
        blend: Some(Blend {
            shape_b: wide.key(),
            place_b: place(-0.5, 0.25, 0.0),
            t: 0.375,
        }),
        reveal: Some((0.125, 0.875)),
    };
    path.paint = PaintBlock {
        fill: Paint::One([0.98, 0.5, 0.25, 0.5]),
        stroke: Paint::Two([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        background_stroke: Paint::Table(gradient.key()),
        stroke_width: 4.0,
        background_stroke_width: 1.5,
        gradient: Some([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]),
        joint: Joint::Round,
        cap: Cap::Square,
        paint_b: Some(PaintB {
            fill: Paint::One([0.0, 0.0, 1.0, 1.0]),
            stroke: Paint::None,
            background_stroke: Paint::Table(gradient.key()),
            stroke_width: 2.0,
            background_stroke_width: 1.0,
        }),
        mix: 0.25,
    };
    path.dash = Some(Dash {
        period: 0.5,
        duty: 0.25,
        phase: 0.1,
    });
    path.depth = Depth {
        space: Space::World,
        z_index: -1.5,
        sort_point: [0.25, -0.5, 1.0],
        shade_in_3d: true,
        fixed_orientation_centre: Some([1.0, 2.0, 3.0]),
    };
    path.clip = Some((wide.key(), place(0.0, 0.0, 0.0)));
    path.opacity = 0.5;
    path.layer = 3;

    let mut faces = Record::new(RecordKind::FaceGroup, 2, 102);
    faces.geometry.shape_a = Some(wide.key());
    faces.paint.stroke_width = 0.5;
    faces.paint.joint = Joint::Bevel;
    faces.paint.cap = Cap::Round;
    faces.body = Body::FaceGroup(FaceGroup {
        uv: vec![[0, 0], [1, 0]],
        grid: [2, 1],
        fill: per_face.key(),
        stroke: Some(face_strokes.key()),
    });

    let cloud_shape = Shape::new(ShapeData::Path {
        points: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    });
    let mut cloud = Record::new(RecordKind::Points, 3, 103);
    cloud.geometry.shape_a = Some(cloud_shape.key());
    cloud.geometry.reveal = Some((0.25, 0.75));
    cloud.body = Body::Points(Points {
        colors: per_point.key(),
        radius: 2.0,
    });

    let mut image = Record::new(RecordKind::Image, 4, 104);
    image.geometry.shape_a = Some(picture.key());
    image.depth.space = Space::Overlay;
    image.body = Body::Image(ImageSource::Picture {
        corners: [
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.125, -1.25, 0.125],
        ],
        resampling: 3,
        opacity: 0.75,
    });

    let mut display = Record::new(RecordKind::Image, 5, 105);
    display.body = Body::Image(ImageSource::SubFrame {
        rank: 2,
        corners: [
            [-2.0, 1.0, 0.25],
            [2.0, 1.0, 0.5],
            [-2.0, -1.0, 0.75],
            [2.25, -1.125, 1.0],
        ],
        resampling: 2,
        opacity: 0.625,
        view: Box::new(View {
            frame_center: [0.25, -0.5, 0.75],
            mapping_centre: [-0.125, 0.375, 0.875],
            ..flat_view(32, 18, [0.4, 0.3, 0.2, 1.0])
        }),
        order: vec![OrderEntry {
            layer: Layer::Static,
            slot: 1,
            part: 0,
        }],
    });

    let mut batch = Record::new(RecordKind::Instances, 6, 106);
    batch.geometry.shape_a = Some(square.key());
    batch.paint.stroke_width = 1.0;
    batch.paint.joint = Joint::Miter;
    batch.paint.cap = Cap::Butt;
    batch.body = Body::Instances(Instances {
        placements: vec![
            place(0.0, 0.0, 0.0),
            place(1.0, 0.0, 0.0),
            place(2.0, 0.0, 0.0),
        ],
        fill: per_point.key(),
        stroke: instance_strokes.key(),
        objects: vec![201, 202, 203],
    });

    let mut mesh = Record::new(RecordKind::Mesh, 7, 107);
    mesh.geometry.shape_a = Some(tri.key());
    mesh.geometry.blend = Some(Blend {
        shape_b: bare.key(),
        place_b: place(0.0, 0.0, 1.0),
        t: 0.5,
    });
    mesh.paint.fill = Paint::One([0.5, 0.5, 0.5, 1.0]);
    mesh.depth.space = Space::World;

    let mut vol = Record::new(RecordKind::Volume, 8, 108);
    vol.features = 0x0102_0304_0506_0708;
    vol.geometry.shape_a = Some(volume.key());
    vol.body = Body::Opaque(vec![9, 8, 7]);

    // The other two reserved kinds, outside the order.
    let mut fld = Record::new(RecordKind::Field, 9, 109);
    fld.geometry.shape_a = Some(field.key());
    fld.body = Body::Opaque(vec![6, 5]);
    let mut emission = Record::new(RecordKind::Emission, 10, 110);
    emission.body = Body::Opaque(vec![4]);

    let mut first = Delta::new(Time { num: 1, den: 15 }, 0);
    let mut view = three_d_view(64, 36);
    view.mapping_centre = [-0.75, 0.625, -0.125];
    view.background = Background::Color([0.1, 0.2, 0.3, 0.9]);
    view.light = LightModel::Other {
        code: 7,
        params: vec![0.25, 0.5],
    };
    first.view = Some(view);
    first.objects = vec![
        ObjectInfo {
            object: 101,
            stable_id: "Square#1".into(),
            source: "scene.py:12".into(),
        },
        ObjectInfo {
            object: 102,
            stable_id: "Surface#1".into(),
            source: "scene.py:14".into(),
        },
    ];
    first.upload = vec![square, wide, tri, bare, picture, volume, cloud_shape, field];
    first.upload_paints = vec![
        gradient,
        per_face.clone(),
        face_strokes,
        per_point,
        instance_strokes,
    ];
    first.set = vec![
        path, faces, cloud, image, display, batch, mesh, vol, fld, emission,
    ];
    first.order = Some(vec![
        OrderEntry {
            layer: Layer::Static,
            slot: 2,
            part: 1,
        },
        OrderEntry {
            layer: Layer::Moving,
            slot: 1,
            part: 0,
        },
        OrderEntry {
            layer: Layer::Overlay,
            slot: 4,
            part: 0,
        },
    ]);

    let mut second = Delta::new(
        Time {
            num: -7,
            den: 9_000_000_000,
        },
        1,
    );
    second.view = Some(View {
        background: Background::Image(first.upload[4].key()),
        camera: Camera::Moving,
        ..flat_view(64, 36, [0.0; 4])
    });
    second.remove = vec![8, 7];
    second.release = vec![first.upload[5].key(), first.upload[2].key()];
    second.order = Some(Vec::new());

    let mut third = Delta::new(Time { num: 1, den: 3 }, 2);
    let mut bare_faces = Record::new(RecordKind::FaceGroup, 2, 112);
    bare_faces.geometry.shape_a = first.set[1].geometry.shape_a;
    bare_faces.body = Body::FaceGroup(FaceGroup {
        uv: vec![[1, 0], [0, 0]],
        grid: [2, 1],
        fill: per_face.key(),
        stroke: None,
    });
    third.set = vec![bare_faces];

    vec![first, second, third]
}

/// XF 1.1's canonical sample, separate so XF 1.0's samples retain their exact bytes.
pub fn pre_projected_sample() -> Delta {
    let mut d = Delta::new(Time { num: 2, den: 15 }, 3);
    d.version = (1, 1);
    let original = Shape::new(ShapeData::Path {
        points: square_points(1.0),
    });
    let projected = Shape::new(ShapeData::Path {
        points: vec![[1.25, -2.5, 0.125], [3.5, 4.75, -0.625]],
    });
    let stops = Stops::new(vec![
        [0.125, 0.25, 0.375, 0.5],
        [0.625, 0.75, 0.875, 1.0],
        [0.9, 0.8, 0.7, 0.6],
    ]);
    let mut r = Record::new(RecordKind::Path, 11, 111);
    r.geometry.shape_a = Some(original.key());
    r.features = FEATURE_PRE_PROJECTED;
    r.pre_projected = Some(PreProjected {
        points: projected.key(),
        gradient: Some([[-1.25, -2.5, -3.75], [4.125, 5.25, 6.5]]),
        fill: Paint::One([0.15, 0.25, 0.35, 0.45]),
        stroke: Paint::Two([[0.55, 0.65, 0.75, 0.85], [0.95, 0.05, 0.125, 0.375]]),
        background_stroke: Paint::Table(stops.key()),
    });
    d.upload = vec![original, projected];
    d.upload_paints = vec![stops];
    d.set = vec![r];
    d
}
