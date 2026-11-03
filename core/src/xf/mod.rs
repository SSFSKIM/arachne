//! XF, the Frame IR: a versioned, typed description of frames.
//!
//! A [`crate::stream::Stream`] holds what a drawer keeps across frames (shapes, paints, objects,
//! records, the view, the draw order); each drawn frame is a [`Delta`] against it. A record
//! describes one drawn object; a shape is geometry uploaded once under its content key. The binary
//! form, used for fixtures, is in [`wire`] and described in `core/XF.md`.

pub mod key;
pub mod wire;

pub use key::Key;

/// The XF version this build reads, writes and draws: (major, minor).
///
/// Before the freeze (XF 1) the major is 0 and the layout may change; after it, the layout is
/// append-only and a new field or kind bumps the minor.
pub const XF_VERSION: (u16, u16) = (0, 0);

/// The feature bits this build's drawer knows. A record carrying any other bit is refused
/// (`render.unsupported`), never drawn partly. XF 0 defines none.
pub const KNOWN_FEATURES: u64 = 0;

/// An exact time: numerator over a positive denominator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Time {
    pub num: i64,
    pub den: i64,
}

/// A 3x4 affine placement, row-major: `p' = M[:, :3] p + M[:, 3]`.
pub type Affine = [[f64; 4]; 3];

/// The identity placement.
pub const IDENTITY: Affine = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

/// The kind codes of shapes and paint tables. They lead the canonical bytes a key hashes, so a
/// shape and a paint table never share a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ShapeKind {
    Path = 1,
    Mesh = 2,
    Image = 3,
    /// Reserved: typed and versioned, refused by this drawer.
    Volume = 4,
    /// Reserved: typed and versioned, refused by this drawer.
    Field = 5,
}

/// The kind code of a paint table in canonical bytes.
pub const PAINT_TABLE_KIND: u8 = 0x10;

/// Geometry uploaded once and found by its content key.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeData {
    /// Points in CE's cubic layout (anchor, handle, handle, anchor per curve); subpaths are split
    /// by CE's rule when drawn, never stored. A `points` record's positions use this kind too.
    Path { points: Vec<[f64; 3]> },
    /// A triangle mesh, with optional per-vertex normals.
    Mesh {
        positions: Vec<[f64; 3]>,
        triangles: Vec<[u32; 3]>,
        normals: Option<Vec<[f64; 3]>>,
    },
    /// Straight (not premultiplied) RGBA pixels, row-major, `height * width * 4` bytes.
    Image {
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    },
    /// A reserved kind (volume, field) or a kind code this build does not know, carried as opaque
    /// bytes and never drawn.
    Reserved { code: u8, payload: Vec<u8> },
}

/// A shape and its key.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub key: Key,
    pub data: ShapeData,
}

impl Shape {
    /// A shape keyed by its canonical bytes.
    pub fn new(data: ShapeData) -> Shape {
        Shape {
            key: key::shape_key(&data),
            data,
        }
    }
}

/// A table of RGBA rows: the stops of a paint of 3+ stops, or a per-face or per-point colour table.
/// Colours are CE's floats (straight alpha, sRGB-encoded channels in 0..1).
#[derive(Clone, Debug, PartialEq)]
pub struct Stops {
    pub key: Key,
    pub rows: Vec<[f64; 4]>,
}

impl Stops {
    /// A table keyed by its canonical bytes.
    pub fn new(rows: Vec<[f64; 4]>) -> Stops {
        Stops {
            key: key::paint_key(&rows),
            rows,
        }
    }
}

/// What a findings report says about a drawn object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectInfo {
    pub object: u32,
    pub stable_id: String,
    /// Where the object was made, as `file:line`.
    pub source: String,
}

/// The record kinds. The reserved ones are typed and versioned, and refused by an XF 0 or XF 1
/// drawer; a code this build does not know is kept as `Unknown` and refused the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordKind {
    Path,
    FaceGroup,
    Points,
    Image,
    Instances,
    Mesh,
    Volume,
    Field,
    Emission,
    Unknown(u8),
}

impl RecordKind {
    pub fn code(self) -> u8 {
        match self {
            RecordKind::Path => 1,
            RecordKind::FaceGroup => 2,
            RecordKind::Points => 3,
            RecordKind::Image => 4,
            RecordKind::Instances => 5,
            RecordKind::Mesh => 6,
            RecordKind::Volume => 64,
            RecordKind::Field => 65,
            RecordKind::Emission => 66,
            RecordKind::Unknown(code) => code,
        }
    }

    pub fn from_code(code: u8) -> RecordKind {
        match code {
            1 => RecordKind::Path,
            2 => RecordKind::FaceGroup,
            3 => RecordKind::Points,
            4 => RecordKind::Image,
            5 => RecordKind::Instances,
            6 => RecordKind::Mesh,
            64 => RecordKind::Volume,
            65 => RecordKind::Field,
            66 => RecordKind::Emission,
            other => RecordKind::Unknown(other),
        }
    }

    /// The kind's name as XF writes it (`kind<code>` for a code this build does not know).
    pub fn name(self) -> String {
        match self {
            RecordKind::Path => "path".into(),
            RecordKind::FaceGroup => "face_group".into(),
            RecordKind::Points => "points".into(),
            RecordKind::Image => "image".into(),
            RecordKind::Instances => "instances".into(),
            RecordKind::Mesh => "mesh".into(),
            RecordKind::Volume => "volume".into(),
            RecordKind::Field => "field".into(),
            RecordKind::Emission => "emission".into(),
            RecordKind::Unknown(code) => format!("kind{code}"),
        }
    }

    pub fn from_name(name: &str) -> Option<RecordKind> {
        Some(match name {
            "path" => RecordKind::Path,
            "face_group" => RecordKind::FaceGroup,
            "points" => RecordKind::Points,
            "image" => RecordKind::Image,
            "instances" => RecordKind::Instances,
            "mesh" => RecordKind::Mesh,
            "volume" => RecordKind::Volume,
            "field" => RecordKind::Field,
            "emission" => RecordKind::Emission,
            _ => return None,
        })
    }

    /// Whether XF reserves this kind for later drawers.
    pub fn is_reserved(self) -> bool {
        matches!(
            self,
            RecordKind::Volume | RecordKind::Field | RecordKind::Emission
        )
    }
}

/// A straight-path blend of two placed shapes, by CE's `interpolate`.
#[derive(Clone, Debug, PartialEq)]
pub struct Blend {
    pub shape_b: Key,
    pub place_b: Affine,
    pub t: f64,
}

/// Where a record's geometry comes from.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    /// Absent only for an image record whose picture is a sub-frame, and for reserved kinds.
    pub shape_a: Option<Key>,
    pub place_a: Affine,
    pub blend: Option<Blend>,
    /// A window with `pointwise_become_partial`'s meaning.
    pub reveal: Option<(f64, f64)>,
}

impl Default for Geometry {
    fn default() -> Geometry {
        Geometry {
            shape_a: None,
            place_a: IDENTITY,
            blend: None,
            reveal: None,
        }
    }
}

/// One paint: nothing, one or two inline stops, or a table of 3+ stops.
#[derive(Clone, Debug, PartialEq)]
pub enum Paint {
    None,
    One([f64; 4]),
    Two([[f64; 4]; 2]),
    Table(Key),
}

/// CE's joint types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Joint {
    Auto = 0,
    Round = 1,
    Bevel = 2,
    Miter = 3,
}

/// CE's cap styles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cap {
    Auto = 0,
    Round = 1,
    Butt = 2,
    Square = 3,
}

/// The second paint of a paint blend: what CE's style interpolation moves.
#[derive(Clone, Debug, PartialEq)]
pub struct PaintB {
    pub fill: Paint,
    pub stroke: Paint,
    pub background_stroke: Paint,
    pub stroke_width: f64,
    pub background_stroke_width: f64,
}

/// A record's paint block.
#[derive(Clone, Debug, PartialEq)]
pub struct PaintBlock {
    pub fill: Paint,
    pub stroke: Paint,
    pub background_stroke: Paint,
    /// Widths in CE's units.
    pub stroke_width: f64,
    pub background_stroke_width: f64,
    /// Gradient endpoints, as `get_gradient_start_and_end_points` gives them.
    pub gradient: Option<[[f64; 3]; 2]>,
    pub joint: Joint,
    pub cap: Cap,
    /// A paint blend: this block mixed toward `paint_b` by `mix`.
    pub paint_b: Option<PaintB>,
    pub mix: f64,
}

impl Default for PaintBlock {
    fn default() -> PaintBlock {
        PaintBlock {
            fill: Paint::None,
            stroke: Paint::None,
            background_stroke: Paint::None,
            stroke_width: 0.0,
            background_stroke_width: 0.0,
            gradient: None,
            joint: Joint::Auto,
            cap: Cap::Auto,
            paint_b: None,
            mix: 0.0,
        }
    }
}

/// A dash pattern (never set from CE, whose dashes are submobjects).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dash {
    pub period: f64,
    pub duty: f64,
    pub phase: f64,
}

/// The space a record is drawn in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Space {
    Screen = 0,
    World = 1,
    Overlay = 2,
}

/// How a record takes part in depth.
#[derive(Clone, Debug, PartialEq)]
pub struct Depth {
    pub space: Space,
    pub z_index: f64,
    pub sort_point: [f64; 3],
    pub shade_in_3d: bool,
    pub fixed_orientation_centre: Option<[f64; 3]>,
}

impl Default for Depth {
    fn default() -> Depth {
        Depth {
            space: Space::Screen,
            z_index: 0.0,
            sort_point: [0.0; 3],
            shade_in_3d: false,
            fixed_orientation_centre: None,
        }
    }
}

/// A Surface's uv faces as one record: the faces' stacked points are the record's path shape.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceGroup {
    /// Per face, its (u, v) indices.
    pub uv: Vec<[u32; 2]>,
    /// The grid's size (u count, v count).
    pub grid: [u32; 2],
    /// Per-face fill colours (one row per face).
    pub fill: Key,
    /// Per-face stroke colours (one row per face), when the faces are stroked.
    pub stroke: Option<Key>,
}

/// A point cloud: positions are the record's path-kind shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Points {
    /// Per-point colours (one row per point).
    pub colors: Key,
    /// CE's stroke width.
    pub radius: f64,
}

/// What an image record shows.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageSource {
    /// The record's image shape, placed at three corners.
    Picture {
        corners: [[f64; 3]; 3],
        /// PIL's `Image.Resampling` value.
        resampling: u8,
        opacity: f64,
    },
    /// A MultiCamera display: another view over the same slots.
    SubFrame {
        view: Box<View>,
        order: Vec<OrderEntry>,
    },
}

/// One run of batched leaves sharing a prototype shape and paint block.
#[derive(Clone, Debug, PartialEq)]
pub struct Instances {
    pub placements: Vec<Affine>,
    /// Per-instance fill and stroke colours (one row per instance).
    pub fill: Key,
    pub stroke: Key,
    /// Per-instance object refs.
    pub objects: Vec<u32>,
}

/// The kind-specific part of a record.
#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    /// `path` and `mesh` records carry nothing beyond the fixed fields.
    None,
    FaceGroup(FaceGroup),
    Points(Points),
    Image(ImageSource),
    Instances(Instances),
    /// A reserved kind's body, or the whole remainder of a kind this build does not know.
    Opaque(Vec<u8>),
}

/// One drawn object.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub kind: RecordKind,
    pub features: u64,
    pub slot: u32,
    pub object: u32,
    pub geometry: Geometry,
    pub paint: PaintBlock,
    pub dash: Option<Dash>,
    pub depth: Depth,
    /// Reserved: a clip shape and its placement (never set from CE).
    pub clip: Option<(Key, Affine)>,
    /// Group opacity (1 from CE).
    pub opacity: f64,
    pub layer: u16,
    pub body: Body,
}

impl Record {
    /// A record of `kind` with every fixed field at its default.
    pub fn new(kind: RecordKind, slot: u32, object: u32) -> Record {
        Record {
            kind,
            features: 0,
            slot,
            object,
            geometry: Geometry::default(),
            paint: PaintBlock::default(),
            dash: None,
            depth: Depth::default(),
            clip: None,
            opacity: 1.0,
            layer: 0,
            body: Body::None,
        }
    }

    /// Every shape key the record references.
    pub fn shape_keys(&self) -> Vec<Key> {
        let mut keys = Vec::new();
        keys.extend(self.geometry.shape_a);
        if let Some(blend) = &self.geometry.blend {
            keys.push(blend.shape_b);
        }
        if let Some((clip, _)) = &self.clip {
            keys.push(*clip);
        }
        if let Body::Image(ImageSource::SubFrame { view, .. }) = &self.body
            && let Background::Image(key) = &view.background
        {
            keys.push(*key);
        }
        keys
    }

    /// Every paint table key the record references.
    pub fn paint_keys(&self) -> Vec<Key> {
        let mut keys = Vec::new();
        let mut paint = |p: &Paint| {
            if let Paint::Table(k) = p {
                keys.push(*k);
            }
        };
        paint(&self.paint.fill);
        paint(&self.paint.stroke);
        paint(&self.paint.background_stroke);
        if let Some(b) = &self.paint.paint_b {
            paint(&b.fill);
            paint(&b.stroke);
            paint(&b.background_stroke);
        }
        match &self.body {
            Body::FaceGroup(f) => {
                keys.push(f.fill);
                keys.extend(f.stroke);
            }
            Body::Points(p) => keys.push(p.colors),
            Body::Instances(i) => {
                keys.push(i.fill);
                keys.push(i.stroke);
            }
            Body::None | Body::Image(_) | Body::Opaque(_) => {}
        }
        keys
    }
}

/// A view's background.
#[derive(Clone, Debug, PartialEq)]
pub enum Background {
    /// CE's background colour with its opacity as alpha.
    Color([f64; 4]),
    /// An image shape, cropped to the frame as CE crops it.
    Image(Key),
}

/// The 3D camera's values.
#[derive(Clone, Debug, PartialEq)]
pub struct ThreeDCamera {
    pub phi: f64,
    pub theta: f64,
    pub gamma: f64,
    pub focal_distance: f64,
    pub zoom: f64,
    pub exponential_projection: bool,
    pub light_source: [f64; 3],
    pub shading_factor: f64,
    pub should_apply_shading: bool,
}

/// Which camera drew the view.
#[derive(Clone, Debug, PartialEq)]
pub enum Camera {
    Flat,
    Moving,
    ThreeD(ThreeDCamera),
}

/// How 3D records are lit.
#[derive(Clone, Debug, PartialEq)]
pub enum LightModel {
    /// CE's per-face shading.
    Ce,
    /// A model this build does not know: its code and parameters, refused when drawn.
    Other { code: u8, params: Vec<f64> },
}

/// The view a frame is drawn from.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub pixel_width: u32,
    pub pixel_height: u32,
    pub frame_width: f64,
    pub frame_height: f64,
    pub frame_center: [f64; 3],
    pub background: Background,
    pub camera: Camera,
    pub cairo_line_width_multiple: f64,
    pub use_z_index: bool,
    pub light: LightModel,
}

/// The layer an order entry draws in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Layer {
    Static = 0,
    Moving = 1,
    Overlay = 2,
}

/// One place in CE's draw order: a slot and the part of it drawn there (a face of a face group,
/// else 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderEntry {
    pub layer: Layer,
    pub slot: u32,
    pub part: u32,
}

/// One drawn frame, against the stream. Empty for a held frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Delta {
    pub version: (u16, u16),
    pub t: Time,
    pub seq: u64,
    pub view: Option<View>,
    pub objects: Vec<ObjectInfo>,
    pub upload: Vec<Shape>,
    pub upload_paints: Vec<Stops>,
    pub release: Vec<Key>,
    pub set: Vec<Record>,
    pub remove: Vec<u32>,
    pub order: Option<Vec<OrderEntry>>,
}

impl Delta {
    /// An empty delta at time `t`, sequence number `seq`.
    pub fn new(t: Time, seq: u64) -> Delta {
        Delta {
            version: XF_VERSION,
            t,
            seq,
            view: None,
            objects: Vec::new(),
            upload: Vec::new(),
            upload_paints: Vec::new(),
            release: Vec::new(),
            set: Vec::new(),
            remove: Vec::new(),
            order: None,
        }
    }

    /// Whether the delta changes nothing (a held frame).
    pub fn is_empty(&self) -> bool {
        self.view.is_none()
            && self.objects.is_empty()
            && self.upload.is_empty()
            && self.upload_paints.is_empty()
            && self.release.is_empty()
            && self.set.is_empty()
            && self.remove.is_empty()
            && self.order.is_none()
    }
}
