//! XF's binary form, for fixtures and dumped streams: versioned, little-endian, byte-stable, and
//! length-prefixed per item, so a reader keeps what it does not know instead of failing to parse.
//! `core/XF.md` describes the layout field by field.

use super::*;
use std::fmt;
use std::path::Path;

/// The first four bytes of an XF file.
pub const MAGIC: [u8; 4] = *b"ARXF";

/// Why bytes could not be read as XF.
#[derive(Debug)]
pub enum XfError {
    BadMagic,
    /// A major version this build does not read.
    Version {
        found: (u16, u16),
    },
    Truncated {
        what: &'static str,
    },
    Invalid(String),
    /// An uploaded shape or paint whose stated key is not the key of its bytes.
    KeyMismatch {
        stated: Key,
        computed: Key,
    },
    Io(std::io::Error),
}

impl fmt::Display for XfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XfError::BadMagic => write!(f, "not an XF file"),
            XfError::Version { found } => write!(
                f,
                "XF {}.{} cannot be read by this build (XF {}.{})",
                found.0, found.1, XF_VERSION.0, XF_VERSION.1
            ),
            XfError::Truncated { what } => write!(f, "truncated XF: {what}"),
            XfError::Invalid(why) => write!(f, "invalid XF: {why}"),
            XfError::KeyMismatch { stated, computed } => write!(
                f,
                "a key does not match its content: stated {}, computed {}",
                stated.hex(),
                computed.hex()
            ),
            XfError::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for XfError {}

type Result<T> = std::result::Result<T, XfError>;

// ---------------------------------------------------------------------------------------------
// Writing

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64s(&mut self, vs: &[f64]) {
        for v in vs {
            self.f64(*v);
        }
    }
    fn count(&mut self, n: usize) {
        self.u32(u32::try_from(n).expect("an XF count fits in u32"));
    }
    fn key(&mut self, k: &Key) {
        self.0.extend_from_slice(&k.0);
    }
    fn str(&mut self, s: &str) {
        self.count(s.len());
        self.0.extend_from_slice(s.as_bytes());
    }
    fn affine(&mut self, a: &Affine) {
        for row in a {
            self.f64s(row);
        }
    }
    /// An item preceded by its byte length.
    fn item(&mut self, bytes: &[u8]) {
        self.count(bytes.len());
        self.0.extend_from_slice(bytes);
    }
}

fn encoded(f: impl FnOnce(&mut W)) -> Vec<u8> {
    let mut w = W(Vec::new());
    f(&mut w);
    w.0
}

fn write_paint(w: &mut W, p: &Paint) {
    match p {
        Paint::None => w.u8(0),
        Paint::One(c) => {
            w.u8(1);
            w.f64s(c);
        }
        Paint::Two([a, b]) => {
            w.u8(2);
            w.f64s(a);
            w.f64s(b);
        }
        Paint::Table(k) => {
            w.u8(3);
            w.key(k);
        }
    }
}

fn write_order(w: &mut W, order: &[OrderEntry]) {
    w.count(order.len());
    for e in order {
        w.u8(e.layer as u8);
        w.u32(e.slot);
        w.u32(e.part);
    }
}

fn encode_view(v: &View) -> Vec<u8> {
    encoded(|w| {
        w.u32(v.pixel_width);
        w.u32(v.pixel_height);
        w.f64(v.frame_width);
        w.f64(v.frame_height);
        w.f64s(&v.frame_center);
        w.f64s(&v.mapping_centre);
        match &v.background {
            Background::Color(c) => {
                w.u8(0);
                w.f64s(c);
            }
            Background::Image(k) => {
                w.u8(1);
                w.key(k);
            }
        }
        match &v.camera {
            Camera::Flat => w.u8(0),
            Camera::Moving => w.u8(1),
            Camera::ThreeD(c) => {
                w.u8(2);
                w.f64s(&[c.phi, c.theta, c.gamma, c.focal_distance, c.zoom]);
                w.bool(c.exponential_projection);
                w.f64s(&c.light_source);
                w.f64(c.shading_factor);
                w.bool(c.should_apply_shading);
            }
        }
        w.f64(v.cairo_line_width_multiple);
        w.bool(v.use_z_index);
        let (code, params) = v.light.code_and_params();
        w.u8(code);
        w.count(params.len());
        w.f64s(&params);
        if v.input_grid != 0.0 {
            w.f64(v.input_grid);
        }
    })
}

fn encode_shape(s: &Shape) -> Vec<u8> {
    encoded(|w| {
        match s.data() {
            ShapeData::Path { .. } => w.u8(ShapeKind::Path as u8),
            ShapeData::Mesh { .. } => w.u8(ShapeKind::Mesh as u8),
            ShapeData::Image { .. } => w.u8(ShapeKind::Image as u8),
            ShapeData::Reserved { code, .. } => w.u8(*code),
        }
        w.key(&s.key());
        match s.data() {
            ShapeData::Path { points } => {
                w.count(points.len());
                points.iter().for_each(|p| w.f64s(p));
            }
            ShapeData::Mesh {
                positions,
                triangles,
                normals,
            } => {
                w.count(positions.len());
                positions.iter().for_each(|p| w.f64s(p));
                w.count(triangles.len());
                triangles.iter().flatten().for_each(|i| w.u32(*i));
                w.bool(normals.is_some());
                if let Some(n) = normals {
                    n.iter().for_each(|p| w.f64s(p));
                }
            }
            ShapeData::Image {
                width,
                height,
                rgba,
            } => {
                w.u32(*width);
                w.u32(*height);
                w.0.extend_from_slice(rgba);
            }
            ShapeData::Reserved { payload, .. } => w.0.extend_from_slice(payload),
        }
    })
}

fn encode_stops(s: &Stops) -> Vec<u8> {
    encoded(|w| {
        w.key(&s.key());
        w.count(s.rows().len());
        s.rows().iter().for_each(|r| w.f64s(r));
    })
}

/// A record's bytes, without the length prefix a delta puts before them.
pub fn encode_record(r: &Record) -> Vec<u8> {
    encoded(|w| {
        w.u8(r.kind.code());
        w.u64(r.features);
        w.u32(r.slot);
        w.u32(r.object);
        if let RecordKind::Unknown(_) = r.kind {
            let Body::Opaque(rest) = &r.body else {
                panic!("a record of an unknown kind carries its bytes opaque");
            };
            w.0.extend_from_slice(rest);
            return;
        }

        assert_eq!(
            r.features & FEATURE_PRE_PROJECTED != 0,
            r.pre_projected.is_some(),
            "pre_projected feature bit and tail field disagree"
        );

        let g = &r.geometry;
        w.u8(g.shape_a.is_some() as u8
            | (g.blend.is_some() as u8) << 1
            | (g.reveal.is_some() as u8) << 2);
        if let Some(k) = &g.shape_a {
            w.key(k);
        }
        w.affine(&g.place_a);
        if let Some(b) = &g.blend {
            w.key(&b.shape_b);
            w.affine(&b.place_b);
            w.f64(b.t);
        }
        if let Some((lo, hi)) = g.reveal {
            w.f64(lo);
            w.f64(hi);
        }

        let p = &r.paint;
        write_paint(w, &p.fill);
        write_paint(w, &p.stroke);
        write_paint(w, &p.background_stroke);
        w.f64(p.stroke_width);
        w.f64(p.background_stroke_width);
        w.bool(p.gradient.is_some());
        if let Some([a, b]) = &p.gradient {
            w.f64s(a);
            w.f64s(b);
        }
        w.u8(p.joint as u8);
        w.u8(p.cap as u8);
        w.bool(p.paint_b.is_some());
        if let Some(b) = &p.paint_b {
            write_paint(w, &b.fill);
            write_paint(w, &b.stroke);
            write_paint(w, &b.background_stroke);
            w.f64(b.stroke_width);
            w.f64(b.background_stroke_width);
        }
        w.f64(p.mix);

        w.bool(r.dash.is_some());
        if let Some(d) = &r.dash {
            w.f64s(&[d.period, d.duty, d.phase]);
        }

        let d = &r.depth;
        w.u8(d.space as u8);
        w.f64(d.z_index);
        w.f64s(&d.sort_point);
        w.bool(d.shade_in_3d);
        w.bool(d.fixed_orientation_centre.is_some());
        if let Some(c) = &d.fixed_orientation_centre {
            w.f64s(c);
        }

        w.bool(r.clip.is_some());
        if let Some((k, a)) = &r.clip {
            w.key(k);
            w.affine(a);
        }
        w.f64(r.opacity);
        w.u16(r.layer);
        w.bool(r.fill_image.is_some());
        if let Some(key) = &r.fill_image {
            w.key(key);
        }

        match (r.kind, &r.body) {
            (RecordKind::Path | RecordKind::Mesh, Body::None) => {}
            (RecordKind::FaceGroup, Body::FaceGroup(f)) => {
                w.count(f.uv.len());
                f.uv.iter().flatten().for_each(|i| w.u32(*i));
                w.u32(f.grid[0]);
                w.u32(f.grid[1]);
                w.key(&f.fill);
                w.bool(f.stroke.is_some());
                if let Some(k) = &f.stroke {
                    w.key(k);
                }
            }
            (RecordKind::Points, Body::Points(p)) => {
                w.key(&p.colors);
                w.f64(p.radius);
            }
            (
                RecordKind::Image,
                Body::Image(ImageSource::Picture {
                    corners,
                    resampling,
                    opacity,
                }),
            ) => {
                w.u8(0);
                corners.iter().for_each(|c| w.f64s(c));
                w.u8(*resampling);
                w.f64(*opacity);
            }
            (
                RecordKind::Image,
                Body::Image(ImageSource::SubFrame {
                    rank,
                    corners,
                    resampling,
                    opacity,
                    view,
                    order,
                }),
            ) => {
                w.u8(1);
                w.u32(*rank);
                corners.iter().for_each(|c| w.f64s(c));
                w.u8(*resampling);
                w.f64(*opacity);
                w.item(&encode_view(view));
                write_order(w, order);
            }
            (RecordKind::Instances, Body::Instances(i)) => {
                w.count(i.placements.len());
                i.placements.iter().for_each(|a| w.affine(a));
                w.key(&i.fill);
                w.key(&i.stroke);
                assert_eq!(
                    i.objects.len(),
                    i.placements.len(),
                    "one object per instance"
                );
                i.objects.iter().for_each(|o| w.u32(*o));
            }
            (RecordKind::Volume | RecordKind::Field | RecordKind::Emission, Body::Opaque(b)) => {
                w.item(b);
            }
            (kind, body) => panic!("a {} record cannot carry {body:?}", kind.name()),
        }
        if let Some(p) = &r.pre_projected {
            w.key(&p.points);
            w.bool(p.gradient.is_some());
            if let Some([a, b]) = &p.gradient {
                w.f64s(a);
                w.f64s(b);
            }
            write_paint(w, &p.fill);
            write_paint(w, &p.stroke);
            write_paint(w, &p.background_stroke);
        }
        // XF 1.2: planes, then normals, both written when either is carried (an empty list
        // stands for one not carried), so a newer minor's fields follow at a known place.
        let faces = match &r.body {
            Body::FaceGroup(f) if f.carries_face_lists() => Some(f),
            _ => None,
        };
        if r.planes.is_some() || r.normals.is_some() || faces.is_some() {
            let planes = r.planes.as_deref().unwrap_or_default();
            w.count(planes.len());
            planes.iter().for_each(|code| w.u8(*code));
            let normals = r.normals.as_deref().unwrap_or_default();
            w.count(normals.len());
            normals.iter().for_each(|n| w.f64s(n));
        }
        // XF 1.3: a face group's face ends, then its face widths, both written when either is
        // carried (an empty list stands for one not carried).
        if let Some(f) = faces {
            w.count(f.face_ends.len());
            f.face_ends.iter().for_each(|end| w.u32(*end));
            w.count(f.face_widths.len());
            w.f64s(&f.face_widths);
        }
    })
}

/// The lowest supported version that carries this delta's content. Foreign and future version
/// declarations are retained: this writer cannot decide what their content requires. Re-encoding
/// a delta decoded from a newer minor keeps its version but not what this reader skipped of it
/// (a record's tail past what it knows): those bytes were dropped on read.
pub fn encoded_version(d: &Delta) -> (u16, u16) {
    if d.version.0 != XF_VERSION.0 || d.version.1 > XF_VERSION.1 {
        return d.version;
    }
    let sub_views = d.set.iter().filter_map(|r| match &r.body {
        Body::Image(ImageSource::SubFrame { view, .. }) => Some(&**view),
        _ => None,
    });
    let snapped = d.view.iter().chain(sub_views).any(|v| v.input_grid != 0.0)
        || d.set
            .iter()
            .any(|r| r.planes.is_some() || r.normals.is_some());
    let face_lists = d
        .set
        .iter()
        .any(|r| matches!(&r.body, Body::FaceGroup(f) if f.carries_face_lists()));
    let minor = if face_lists {
        3
    } else if snapped {
        2
    } else {
        u16::from(
            d.set
                .iter()
                .any(|r| r.features & FEATURE_PRE_PROJECTED != 0),
        )
    };
    (1, minor)
}

/// A delta's bytes, at the lowest version that carries its content.
pub fn encode_delta(d: &Delta) -> Vec<u8> {
    encoded(|w| {
        let version = encoded_version(d);
        w.u16(version.0);
        w.u16(version.1);
        w.i64(d.t.num);
        w.i64(d.t.den);
        w.u64(d.seq);
        w.bool(d.view.is_some());
        if let Some(v) = &d.view {
            w.item(&encode_view(v));
        }
        w.count(d.objects.len());
        for o in &d.objects {
            w.item(&encoded(|w| {
                w.u32(o.object);
                w.str(&o.stable_id);
                w.str(&o.source);
            }));
        }
        w.count(d.upload.len());
        d.upload.iter().for_each(|s| w.item(&encode_shape(s)));
        w.count(d.upload_paints.len());
        d.upload_paints
            .iter()
            .for_each(|s| w.item(&encode_stops(s)));
        w.count(d.release.len());
        d.release.iter().for_each(|k| w.key(k));
        w.count(d.set.len());
        d.set.iter().for_each(|r| w.item(&encode_record(r)));
        w.count(d.remove.len());
        d.remove.iter().for_each(|s| w.u32(*s));
        w.bool(d.order.is_some());
        if let Some(order) = &d.order {
            write_order(w, order);
        }
    })
}

/// A file of deltas: the magic, the version its deltas require, then each length-prefixed delta.
pub fn encode_file(deltas: &[Delta]) -> Vec<u8> {
    encoded(|w| {
        w.0.extend_from_slice(&MAGIC);
        let version = deltas.iter().map(encoded_version).max().unwrap_or((1, 0));
        w.u16(version.0);
        w.u16(version.1);
        w.count(deltas.len());
        deltas.iter().for_each(|d| w.item(&encode_delta(d)));
    })
}

// ---------------------------------------------------------------------------------------------
// Reading

struct R<'a> {
    b: &'a [u8],
    at: usize,
    what: &'static str,
}

impl<'a> R<'a> {
    fn new(b: &'a [u8], what: &'static str) -> R<'a> {
        R { b, at: 0, what }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.b.len() - self.at < n {
            return Err(XfError::Truncated { what: self.what });
        }
        let out = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().unwrap())
    }
    fn rest(&mut self) -> &'a [u8] {
        let out = &self.b[self.at..];
        self.at = self.b.len();
        out
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(XfError::Invalid(format!(
                "{}: {v} is not a bool",
                self.what
            ))),
        }
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.array()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.array()?))
    }
    fn f64s<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut out = [0.0; N];
        for v in &mut out {
            *v = self.f64()?;
        }
        Ok(out)
    }
    /// A count of items of at least `item_bytes` each, checked against what is left, so a corrupt
    /// count fails as truncation instead of a huge allocation.
    fn count(&mut self, item_bytes: usize) -> Result<usize> {
        let n = self.u32()? as usize;
        if n.saturating_mul(item_bytes) > self.b.len() - self.at {
            return Err(XfError::Truncated { what: self.what });
        }
        Ok(n)
    }
    fn key(&mut self) -> Result<Key> {
        Ok(Key(self.array()?))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.count(1)?;
        String::from_utf8(self.take(n)?.to_vec())
            .map_err(|_| XfError::Invalid(format!("{}: a string is not UTF-8", self.what)))
    }
    fn affine(&mut self) -> Result<Affine> {
        Ok([self.f64s()?, self.f64s()?, self.f64s()?])
    }
    /// The next length-prefixed item, as its own reader.
    fn item(&mut self, what: &'static str) -> Result<R<'a>> {
        let n = self.count(1)?;
        Ok(R::new(self.take(n)?, what))
    }
    fn invalid(&self, field: &str, v: u8) -> XfError {
        XfError::Invalid(format!("{}: {v} is not a valid {field}", self.what))
    }
}

fn read_paint(r: &mut R) -> Result<Paint> {
    Ok(match r.u8()? {
        0 => Paint::None,
        1 => Paint::One(r.f64s()?),
        2 => Paint::Two([r.f64s()?, r.f64s()?]),
        3 => Paint::Table(r.key()?),
        v => return Err(r.invalid("paint", v)),
    })
}

fn read_order(r: &mut R) -> Result<Vec<OrderEntry>> {
    let n = r.count(9)?;
    (0..n)
        .map(|_| {
            let layer = match r.u8()? {
                0 => Layer::Static,
                1 => Layer::Moving,
                2 => Layer::Overlay,
                v => return Err(r.invalid("layer", v)),
            };
            Ok(OrderEntry {
                layer,
                slot: r.u32()?,
                part: r.u32()?,
            })
        })
        .collect()
}

fn decode_view(r: &mut R) -> Result<View> {
    let pixel_width = r.u32()?;
    let pixel_height = r.u32()?;
    let frame_width = r.f64()?;
    let frame_height = r.f64()?;
    let frame_center = r.f64s()?;
    let mapping_centre = r.f64s()?;
    let background = match r.u8()? {
        0 => Background::Color(r.f64s()?),
        1 => Background::Image(r.key()?),
        v => return Err(r.invalid("background", v)),
    };
    let camera = match r.u8()? {
        0 => Camera::Flat,
        1 => Camera::Moving,
        2 => {
            let [phi, theta, gamma, focal_distance, zoom] = r.f64s()?;
            Camera::ThreeD(ThreeDCamera {
                phi,
                theta,
                gamma,
                focal_distance,
                zoom,
                exponential_projection: r.bool()?,
                light_source: r.f64s()?,
                shading_factor: r.f64()?,
                should_apply_shading: r.bool()?,
            })
        }
        v => return Err(r.invalid("camera", v)),
    };
    let cairo_line_width_multiple = r.f64()?;
    let use_z_index = r.bool()?;
    let code = r.u8()?;
    let n = r.count(8)?;
    let params = (0..n).map(|_| r.f64()).collect::<Result<Vec<_>>>()?;
    let light = LightModel::from_code(code, params);
    // XF 1.2: the input grid, when the view carries one.
    let input_grid = if r.at < r.b.len() { r.f64()? } else { 0.0 };
    Ok(View {
        pixel_width,
        pixel_height,
        frame_width,
        frame_height,
        frame_center,
        mapping_centre,
        background,
        camera,
        cairo_line_width_multiple,
        use_z_index,
        light,
        input_grid,
    })
}

fn rows3(r: &mut R, n: usize) -> Result<Vec<[f64; 3]>> {
    (0..n).map(|_| r.f64s()).collect()
}

fn decode_shape(r: &mut R) -> Result<Shape> {
    let code = r.u8()?;
    let stated = r.key()?;
    let data = match code {
        c if c == ShapeKind::Path as u8 => {
            let n = r.count(24)?;
            ShapeData::Path {
                points: rows3(r, n)?,
            }
        }
        c if c == ShapeKind::Mesh as u8 => {
            let v = r.count(24)?;
            let positions = rows3(r, v)?;
            let t = r.count(12)?;
            let triangles = (0..t)
                .map(|_| Ok([r.u32()?, r.u32()?, r.u32()?]))
                .collect::<Result<Vec<_>>>()?;
            let normals = if r.bool()? { Some(rows3(r, v)?) } else { None };
            ShapeData::Mesh {
                positions,
                triangles,
                normals,
            }
        }
        c if c == ShapeKind::Image as u8 => {
            let width = r.u32()?;
            let height = r.u32()?;
            let n = (width as usize)
                .checked_mul(height as usize)
                .and_then(|p| p.checked_mul(4))
                .ok_or_else(|| XfError::Invalid("an image's size overflows".into()))?;
            ShapeData::Image {
                width,
                height,
                rgba: r.take(n)?.to_vec(),
            }
        }
        code => ShapeData::Reserved {
            code,
            payload: r.rest().to_vec(),
        },
    };
    let known = matches!(
        data,
        ShapeData::Path { .. } | ShapeData::Mesh { .. } | ShapeData::Image { .. }
    ) || code == ShapeKind::Volume as u8
        || code == ShapeKind::Field as u8;
    if known {
        let computed = key::shape_key(&data);
        if computed != stated {
            return Err(XfError::KeyMismatch { stated, computed });
        }
    }
    Ok(Shape::read(stated, data))
}

fn decode_stops(r: &mut R) -> Result<Stops> {
    let stated = r.key()?;
    let n = r.count(32)?;
    let rows = (0..n).map(|_| r.f64s()).collect::<Result<Vec<_>>>()?;
    let computed = key::paint_key(&rows);
    if computed != stated {
        return Err(XfError::KeyMismatch { stated, computed });
    }
    Ok(Stops::read(stated, rows))
}

fn decode_record(r: &mut R) -> Result<Record> {
    let kind = RecordKind::from_code(r.u8()?);
    let mut rec = Record::new(kind, 0, 0);
    rec.features = r.u64()?;
    rec.slot = r.u32()?;
    rec.object = r.u32()?;
    if let RecordKind::Unknown(_) = kind {
        rec.body = Body::Opaque(r.rest().to_vec());
        return Ok(rec);
    }

    let flags = r.u8()?;
    if flags & !0b111 != 0 {
        return Err(r.invalid("geometry flag set", flags));
    }
    let g = &mut rec.geometry;
    g.shape_a = if flags & 1 != 0 { Some(r.key()?) } else { None };
    g.place_a = r.affine()?;
    if flags & 2 != 0 {
        g.blend = Some(Blend {
            shape_b: r.key()?,
            place_b: r.affine()?,
            t: r.f64()?,
        });
    }
    if flags & 4 != 0 {
        g.reveal = Some((r.f64()?, r.f64()?));
    }

    let p = &mut rec.paint;
    p.fill = read_paint(r)?;
    p.stroke = read_paint(r)?;
    p.background_stroke = read_paint(r)?;
    p.stroke_width = r.f64()?;
    p.background_stroke_width = r.f64()?;
    if r.bool()? {
        p.gradient = Some([r.f64s()?, r.f64s()?]);
    }
    p.joint = match r.u8()? {
        0 => Joint::Auto,
        1 => Joint::Round,
        2 => Joint::Bevel,
        3 => Joint::Miter,
        v => return Err(r.invalid("joint", v)),
    };
    p.cap = match r.u8()? {
        0 => Cap::Auto,
        1 => Cap::Round,
        2 => Cap::Butt,
        3 => Cap::Square,
        v => return Err(r.invalid("cap", v)),
    };
    if r.bool()? {
        p.paint_b = Some(PaintB {
            fill: read_paint(r)?,
            stroke: read_paint(r)?,
            background_stroke: read_paint(r)?,
            stroke_width: r.f64()?,
            background_stroke_width: r.f64()?,
        });
    }
    p.mix = r.f64()?;

    if r.bool()? {
        let [period, duty, phase] = r.f64s()?;
        rec.dash = Some(Dash {
            period,
            duty,
            phase,
        });
    }

    let d = &mut rec.depth;
    d.space = match r.u8()? {
        0 => Space::Screen,
        1 => Space::World,
        2 => Space::Overlay,
        v => return Err(r.invalid("space", v)),
    };
    d.z_index = r.f64()?;
    d.sort_point = r.f64s()?;
    d.shade_in_3d = r.bool()?;
    if r.bool()? {
        d.fixed_orientation_centre = Some(r.f64s()?);
    }

    if r.bool()? {
        rec.clip = Some((r.key()?, r.affine()?));
    }
    rec.opacity = r.f64()?;
    rec.layer = r.u16()?;
    rec.fill_image = if r.bool()? { Some(r.key()?) } else { None };
    if (rec.features & FEATURE_BACKGROUND_IMAGE != 0) != rec.fill_image.is_some() {
        return Err(XfError::Invalid(
            "background_image feature bit and fill_image field disagree".into(),
        ));
    }

    rec.body = match kind {
        RecordKind::Path | RecordKind::Mesh => Body::None,
        RecordKind::FaceGroup => {
            let n = r.count(8)?;
            let uv = (0..n)
                .map(|_| Ok([r.u32()?, r.u32()?]))
                .collect::<Result<Vec<_>>>()?;
            let grid = [r.u32()?, r.u32()?];
            let fill = r.key()?;
            let stroke = if r.bool()? { Some(r.key()?) } else { None };
            Body::FaceGroup(FaceGroup {
                uv,
                grid,
                fill,
                stroke,
                face_ends: Vec::new(),
                face_widths: Vec::new(),
            })
        }
        RecordKind::Points => Body::Points(Points {
            colors: r.key()?,
            radius: r.f64()?,
        }),
        RecordKind::Image => Body::Image(match r.u8()? {
            0 => ImageSource::Picture {
                corners: [r.f64s()?, r.f64s()?, r.f64s()?, r.f64s()?],
                resampling: r.u8()?,
                opacity: r.f64()?,
            },
            1 => ImageSource::SubFrame {
                rank: r.u32()?,
                corners: [r.f64s()?, r.f64s()?, r.f64s()?, r.f64s()?],
                resampling: r.u8()?,
                opacity: r.f64()?,
                view: Box::new(decode_view(&mut r.item("a sub-frame's view")?)?),
                order: read_order(r)?,
            },
            v => return Err(r.invalid("image source", v)),
        }),
        RecordKind::Instances => {
            let n = r.count(96)?;
            let placements = (0..n).map(|_| r.affine()).collect::<Result<Vec<_>>>()?;
            let fill = r.key()?;
            let stroke = r.key()?;
            let objects = (0..n).map(|_| r.u32()).collect::<Result<Vec<_>>>()?;
            Body::Instances(Instances {
                placements,
                fill,
                stroke,
                objects,
            })
        }
        RecordKind::Volume | RecordKind::Field | RecordKind::Emission => {
            Body::Opaque(r.item("a reserved record's body")?.rest().to_vec())
        }
        RecordKind::Unknown(_) => unreachable!("returned above"),
    };
    if rec.features & FEATURE_PRE_PROJECTED != 0 {
        if !matches!(
            kind,
            RecordKind::Path | RecordKind::Points | RecordKind::Image
        ) {
            return Err(XfError::Invalid(format!(
                "pre_projected is not valid on {} records",
                kind.name()
            )));
        }
        rec.pre_projected = Some(PreProjected {
            points: r.key()?,
            gradient: if r.bool()? {
                Some([r.f64s()?, r.f64s()?])
            } else {
                None
            },
            fill: read_paint(r)?,
            stroke: read_paint(r)?,
            background_stroke: read_paint(r)?,
        });
    }
    // XF 1.2: the record's planes, then its normals, when it carries them (an empty list: not
    // carried).
    if r.at < r.b.len() {
        let n = r.count(1)?;
        let planes = (0..n).map(|_| r.u8()).collect::<Result<Vec<_>>>()?;
        rec.planes = (!planes.is_empty()).then_some(planes);
    }
    if r.at < r.b.len() {
        let n = r.count(24)?;
        let normals = (0..n).map(|_| r.f64s()).collect::<Result<Vec<_>>>()?;
        rec.normals = (!normals.is_empty()).then_some(normals);
    }
    // XF 1.3: a face group's face ends, then its face widths (an empty list: not carried).
    if r.at < r.b.len()
        && let Body::FaceGroup(f) = &mut rec.body
    {
        let n = r.count(4)?;
        f.face_ends = (0..n).map(|_| r.u32()).collect::<Result<Vec<_>>>()?;
        let n = r.count(8)?;
        f.face_widths = (0..n).map(|_| r.f64()).collect::<Result<Vec<_>>>()?;
    }
    // Whatever follows belongs to a newer minor version: skipped.
    Ok(rec)
}

/// Reads one delta. A major version other than this build's is refused; a newer minor is read,
/// with what it appended skipped.
pub fn decode_delta(bytes: &[u8]) -> Result<Delta> {
    let mut r = R::new(bytes, "a delta");
    let version = (r.u16()?, r.u16()?);
    if version.0 != XF_VERSION.0 {
        return Err(XfError::Version { found: version });
    }
    let t = Time {
        num: r.i64()?,
        den: r.i64()?,
    };
    if t.den <= 0 {
        return Err(XfError::Invalid(format!(
            "a delta's time has denominator {}",
            t.den
        )));
    }
    let mut d = Delta::new(t, r.u64()?);
    d.version = version;
    if r.bool()? {
        d.view = Some(decode_view(&mut r.item("a view")?)?);
    }
    let n = r.count(4)?;
    for _ in 0..n {
        let mut o = r.item("an object")?;
        d.objects.push(ObjectInfo {
            object: o.u32()?,
            stable_id: o.str()?,
            source: o.str()?,
        });
    }
    let n = r.count(4)?;
    for _ in 0..n {
        d.upload.push(decode_shape(&mut r.item("a shape")?)?);
    }
    let n = r.count(4)?;
    for _ in 0..n {
        d.upload_paints
            .push(decode_stops(&mut r.item("a paint table")?)?);
    }
    let n = r.count(16)?;
    for _ in 0..n {
        d.release.push(r.key()?);
    }
    let n = r.count(4)?;
    for _ in 0..n {
        d.set.push(decode_record(&mut r.item("a record")?)?);
    }
    let n = r.count(4)?;
    for _ in 0..n {
        d.remove.push(r.u32()?);
    }
    if r.bool()? {
        d.order = Some(read_order(&mut r)?);
    }
    Ok(d)
}

/// Reads a file of deltas.
pub fn decode_file(bytes: &[u8]) -> Result<Vec<Delta>> {
    let mut r = R::new(bytes, "an XF file");
    if r.array::<4>()? != MAGIC {
        return Err(XfError::BadMagic);
    }
    let version = (r.u16()?, r.u16()?);
    if version.0 != XF_VERSION.0 {
        return Err(XfError::Version { found: version });
    }
    let n = r.count(4)?;
    (0..n)
        .map(|_| decode_delta(r.item("a delta")?.rest()))
        .collect()
}

pub fn read_file(path: &Path) -> Result<Vec<Delta>> {
    decode_file(&std::fs::read(path).map_err(XfError::Io)?)
}

pub fn write_file(path: &Path, deltas: &[Delta]) -> Result<()> {
    std::fs::write(path, encode_file(deltas)).map_err(XfError::Io)
}
