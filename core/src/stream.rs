//! A Stream: what a drawer holds across frames. Deltas change it, atomically; `draw` turns it into
//! one frame's pixels, or refuses the whole frame when any drawn record carries what this build
//! cannot draw (`render.unsupported`).

use crate::composite::{self, Frame, Producer, Rendered};
use crate::mesh::{self, MeshDraw};
use crate::xf::*;
use crate::{instance, path, points, tiles};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

/// Shapes, paint tables, objects, records, the view and the draw order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stream {
    shapes: HashMap<Key, Shape>,
    paints: HashMap<Key, Stops>,
    objects: BTreeMap<u32, ObjectInfo>,
    records: BTreeMap<u32, Record>,
    view: Option<View>,
    order: Vec<OrderEntry>,
}

/// What one delta changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplyStats {
    /// Shapes the delta uploaded.
    pub uploaded_shapes: usize,
    /// Records the delta set or removed.
    pub changed_records: usize,
}

/// Why a delta was refused. A refused delta changes nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamError {
    /// A major version this build does not draw.
    Version(String),
    /// A reference to a shape, paint table or slot the stream would not hold.
    Missing(String),
    /// A release of a key something still uses.
    InUse(String),
    /// A record or order entry that contradicts itself or what it references.
    Invalid(String),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::Version(m)
            | StreamError::Missing(m)
            | StreamError::InUse(m)
            | StreamError::Invalid(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for StreamError {}

/// A frame refused because a drawn record (or the view) carries a kind or feature this build does
/// not draw: `render.unsupported`. Nothing is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Unsupported {
    /// The record kind's name, or `view`.
    pub kind: String,
    pub slot: Option<u32>,
    pub object: Option<u32>,
    pub reason: String,
}

/// Why `draw` gave no frame.
#[derive(Clone, Debug, PartialEq)]
pub enum DrawError {
    NoView,
    Invalid(String),
    Unsupported(Unsupported),
    /// A drawing thread the system would not start.
    Threads(String),
}

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrawError::NoView => f.write_str("the stream has no view yet"),
            DrawError::Invalid(m) => f.write_str(m),
            DrawError::Threads(m) => write!(f, "a drawing thread could not start: {m}"),
            DrawError::Unsupported(u) => write!(
                f,
                "render.unsupported: {} (slot {:?}, object {:?}): {}",
                u.kind, u.slot, u.object, u.reason
            ),
        }
    }
}

impl std::error::Error for DrawError {}

/// The stream as a delta would leave it, read without changing anything, by the rules commit
/// applies: a key the stream holds keeps what it holds (a validated delta re-uploads only the same
/// content under it, and uploads each key once); the last record set in a slot wins, then removals.
struct After<'a> {
    stream: &'a Stream,
    uploads: HashMap<Key, &'a Shape>,
    upload_paints: HashMap<Key, &'a Stops>,
    /// The records the delta sets, by slot (the last one set wins).
    set: BTreeMap<u32, &'a Record>,
    released: HashSet<Key>,
    removed: HashSet<u32>,
}

impl<'a> After<'a> {
    fn new(stream: &'a Stream, delta: &'a Delta) -> After<'a> {
        After {
            stream,
            uploads: delta.upload.iter().map(|s| (s.key(), s)).collect(),
            upload_paints: delta.upload_paints.iter().map(|s| (s.key(), s)).collect(),
            set: delta.set.iter().map(|r| (r.slot, r)).collect(),
            released: delta.release.iter().copied().collect(),
            removed: delta.remove.iter().copied().collect(),
        }
    }

    fn shape(&self, key: &Key) -> Option<&'a ShapeData> {
        if self.released.contains(key) {
            return None;
        }
        self.stream
            .shapes
            .get(key)
            .or_else(|| self.uploads.get(key).copied())
            .map(|s| s.data())
    }

    fn paint(&self, key: &Key) -> Option<&'a Stops> {
        if self.released.contains(key) {
            return None;
        }
        self.stream
            .paints
            .get(key)
            .or_else(|| self.upload_paints.get(key).copied())
    }

    fn record(&self, slot: u32) -> Option<&'a Record> {
        if self.removed.contains(&slot) {
            return None;
        }
        self.set
            .get(&slot)
            .copied()
            .or_else(|| self.stream.records.get(&slot))
    }

    /// The records the delta sets that it does not also remove: the ones commit keeps.
    fn kept_sets(&self) -> impl Iterator<Item = &'a Record> + '_ {
        self.set
            .values()
            .copied()
            .filter(|r| !self.removed.contains(&r.slot))
    }

    fn records(&self) -> impl Iterator<Item = &'a Record> + '_ {
        self.stream
            .records
            .values()
            .filter(|r| !self.set.contains_key(&r.slot))
            .chain(self.set.values().copied())
            .filter(|r| !self.removed.contains(&r.slot))
    }
}

fn missing_key(what: &str, key: &Key, slot: Option<u32>) -> StreamError {
    match slot {
        Some(slot) => StreamError::Missing(format!(
            "slot {slot} references {what} {} the stream would not hold",
            key.hex()
        )),
        None => StreamError::Missing(format!(
            "the view references {what} {} the stream would not hold",
            key.hex()
        )),
    }
}

fn shape_kind_name(data: &ShapeData) -> &'static str {
    match data {
        ShapeData::Path { .. } => "path",
        ShapeData::Mesh { .. } => "mesh",
        ShapeData::Image { .. } => "image",
        ShapeData::Reserved { .. } => "reserved",
    }
}

impl After<'_> {
    /// A shape the record needs, of one of the kinds it accepts.
    fn needs_shape(
        &self,
        rec: &Record,
        key: Option<&Key>,
        accepts: &[&str],
    ) -> Result<Option<&ShapeData>, StreamError> {
        let Some(key) = key else {
            return Err(StreamError::Invalid(format!(
                "a {} record (slot {}) needs a shape",
                rec.kind.name(),
                rec.slot
            )));
        };
        let data = self
            .shape(key)
            .ok_or_else(|| missing_key("shape", key, Some(rec.slot)))?;
        if !accepts.is_empty() && !accepts.contains(&shape_kind_name(data)) {
            return Err(StreamError::Invalid(format!(
                "a {} record (slot {}) cannot take a {} shape",
                rec.kind.name(),
                rec.slot,
                shape_kind_name(data)
            )));
        }
        Ok(Some(data))
    }

    fn needs_table(&self, rec: &Record, key: &Key, rows: usize) -> Result<(), StreamError> {
        let stops = self
            .paint(key)
            .ok_or_else(|| missing_key("paint table", key, Some(rec.slot)))?;
        if stops.rows().len() != rows {
            return Err(StreamError::Invalid(format!(
                "slot {}: a table of {} rows where {rows} are needed",
                rec.slot,
                stops.rows().len()
            )));
        }
        Ok(())
    }

    fn check_view(&self, view: &View, slot: Option<u32>) -> Result<(), StreamError> {
        if let LightModel::SoftDiffuse { ambient, wrap } = view.light
            && !((0.0..=1.0).contains(&ambient) && wrap.is_finite() && wrap >= 0.0)
        {
            return Err(StreamError::Invalid(format!(
                "soft diffuse light needs an ambient share in 0..=1 and a wrap of at least 0, \
                 not {ambient} and {wrap}"
            )));
        }
        if let Background::Image(key) = &view.background {
            match self.shape(key) {
                Some(ShapeData::Image { .. }) => {}
                Some(_) => {
                    return Err(StreamError::Invalid(
                        "a background image must be an image shape".into(),
                    ));
                }
                None => return Err(missing_key("background image", key, slot)),
            }
        }
        Ok(())
    }

    /// Checks an order against the records the delta leaves. `display` is the slot of the display
    /// whose sub-frame draws the order, which may not draw itself.
    fn check_order(&self, order: &[OrderEntry], display: Option<u32>) -> Result<(), StreamError> {
        for e in order {
            if Some(e.slot) == display {
                return Err(StreamError::Invalid(format!(
                    "the sub-frame of slot {} draws its own slot",
                    e.slot
                )));
            }
            let rec = self.record(e.slot).ok_or_else(|| {
                StreamError::Missing(format!("the order draws slot {}, which is not set", e.slot))
            })?;
            let parts = match (&rec.kind, &rec.body) {
                (RecordKind::Unknown(_), _) => continue,
                (_, Body::FaceGroup(f)) => f.uv.len(),
                _ => 1,
            };
            if e.part as usize >= parts {
                return Err(StreamError::Invalid(format!(
                    "the order draws part {} of slot {}, which has {parts}",
                    e.part, e.slot
                )));
            }
        }
        Ok(())
    }

    fn check_record(&self, rec: &Record) -> Result<(), StreamError> {
        if let RecordKind::Unknown(_) = rec.kind {
            return Ok(()); // Nothing in it can be read; drawing refuses it.
        }
        let g = &rec.geometry;
        let shape_a = g.shape_a.as_ref();
        let a = match (rec.kind, &rec.body) {
            (RecordKind::Path, Body::None) => self.needs_shape(rec, shape_a, &["path"])?,
            (RecordKind::Mesh, Body::None) => self.needs_shape(rec, shape_a, &["mesh"])?,
            (RecordKind::FaceGroup, Body::FaceGroup(_)) => {
                self.needs_shape(rec, shape_a, &["path"])?
            }
            (RecordKind::Points, Body::Points(_)) => self.needs_shape(rec, shape_a, &["path"])?,
            (RecordKind::Instances, Body::Instances(_)) => {
                self.needs_shape(rec, shape_a, &["path", "mesh"])?
            }
            (RecordKind::Image, Body::Image(ImageSource::Picture { .. })) => {
                self.needs_shape(rec, shape_a, &["image"])?
            }
            (RecordKind::Image, Body::Image(ImageSource::SubFrame { view, order })) => {
                if shape_a.is_some() {
                    return Err(StreamError::Invalid(format!(
                        "slot {}: a sub-frame display has no shape",
                        rec.slot
                    )));
                }
                self.check_view(view, Some(rec.slot))?;
                self.check_order(order, Some(rec.slot))?;
                None
            }
            (RecordKind::Volume | RecordKind::Field | RecordKind::Emission, Body::Opaque(_)) => {
                match shape_a {
                    Some(_) => self.needs_shape(rec, shape_a, &[])?,
                    None => None,
                }
            }
            (kind, _) => {
                return Err(StreamError::Invalid(format!(
                    "slot {}: a {} record with another kind's body",
                    rec.slot,
                    kind.name()
                )));
            }
        };
        if let Some(b) = &g.blend {
            let accepts = match a {
                Some(data) => vec![shape_kind_name(data)],
                None => Vec::new(),
            };
            self.needs_shape(rec, Some(&b.shape_b), &accepts)?;
        }
        if let Some((clip, _)) = &rec.clip {
            self.needs_shape(rec, Some(clip), &["path"])?;
        }
        for key in rec.paint_keys() {
            if self.paint(&key).is_none() {
                return Err(missing_key("paint table", &key, Some(rec.slot)));
            }
        }
        let points = match a {
            Some(ShapeData::Path { points }) => points.len(),
            _ => 0,
        };
        match &rec.body {
            Body::FaceGroup(f) => {
                self.needs_table(rec, &f.fill, f.uv.len())?;
                if let Some(stroke) = &f.stroke {
                    self.needs_table(rec, stroke, f.uv.len())?;
                }
            }
            Body::Points(p) => self.needs_table(rec, &p.colors, points)?,
            Body::Instances(i) => {
                let n = i.placements.len();
                if i.objects.len() != n {
                    return Err(StreamError::Invalid(format!(
                        "slot {}: {n} placements and {} objects",
                        rec.slot,
                        i.objects.len()
                    )));
                }
                self.needs_table(rec, &i.fill, n)?;
                self.needs_table(rec, &i.stroke, n)?;
            }
            Body::None | Body::Image(_) | Body::Opaque(_) => {}
        }
        Ok(())
    }
}

impl Stream {
    pub fn new() -> Stream {
        Stream::default()
    }

    /// Applies a delta: the view, objects, uploads, records set, records removed, keys released,
    /// then the order. Everything is checked first; a refused delta changes nothing.
    pub fn apply(&mut self, delta: &Delta) -> Result<ApplyStats, StreamError> {
        if delta.version.0 != XF_VERSION.0 {
            return Err(StreamError::Version(format!(
                "XF {}.{} cannot be drawn by this build (XF {}.{})",
                delta.version.0, delta.version.1, XF_VERSION.0, XF_VERSION.1
            )));
        }
        self.check(delta)?;

        if let Some(view) = &delta.view {
            self.view = Some(view.clone());
        }
        for o in &delta.objects {
            self.objects.insert(o.object, o.clone());
        }
        for s in &delta.upload {
            self.shapes.entry(s.key()).or_insert_with(|| s.clone());
        }
        for s in &delta.upload_paints {
            self.paints.entry(s.key()).or_insert_with(|| s.clone());
        }
        for r in &delta.set {
            self.records.insert(r.slot, r.clone());
        }
        for slot in &delta.remove {
            self.records.remove(slot);
        }
        for key in &delta.release {
            self.shapes.remove(key);
            self.paints.remove(key);
        }
        if let Some(order) = &delta.order {
            self.order = order.clone();
        }
        Ok(ApplyStats {
            uploaded_shapes: delta.upload.len(),
            changed_records: delta.set.len() + delta.remove.len(),
        })
    }

    fn check(&self, delta: &Delta) -> Result<(), StreamError> {
        let mut uploaded = HashSet::new();
        for key in delta
            .upload
            .iter()
            .map(|s| s.key())
            .chain(delta.upload_paints.iter().map(|s| s.key()))
        {
            if !uploaded.insert(key) {
                return Err(StreamError::Invalid(format!(
                    "a delta uploads {} twice",
                    key.hex()
                )));
            }
        }
        let held_with_other = |key: Key| {
            Err(StreamError::Invalid(format!(
                "an upload of {}, which the stream holds with other content",
                key.hex()
            )))
        };
        for shape in &delta.upload {
            check_shape(shape)?;
            let held = self.shapes.get(&shape.key());
            if self.paints.contains_key(&shape.key())
                || held.is_some_and(|h| !same_content(h.data(), shape.data()))
            {
                return held_with_other(shape.key());
            }
        }
        for stops in &delta.upload_paints {
            if self.shapes.contains_key(&stops.key()) {
                return held_with_other(stops.key());
            }
        }
        let after = After::new(self, delta);
        for key in &delta.release {
            let held = self.shapes.contains_key(key)
                || self.paints.contains_key(key)
                || after.uploads.contains_key(key)
                || after.upload_paints.contains_key(key);
            if !held {
                return Err(StreamError::Missing(format!(
                    "a release of {}, which the stream does not hold",
                    key.hex()
                )));
            }
        }
        for slot in &delta.remove {
            if !self.records.contains_key(slot) && !after.set.contains_key(slot) {
                return Err(StreamError::Missing(format!(
                    "a removal of slot {slot}, which is not set"
                )));
            }
        }
        let view = delta.view.as_ref().or(self.view.as_ref());
        if !after.released.is_empty() {
            let in_use = |key: &Key, by: String| {
                if after.released.contains(key) {
                    Err(StreamError::InUse(format!(
                        "a release of {}, which {by} still uses",
                        key.hex()
                    )))
                } else {
                    Ok(())
                }
            };
            for rec in after.records() {
                for key in rec.shape_keys().iter().chain(&rec.paint_keys()) {
                    in_use(key, format!("slot {}", rec.slot))?;
                }
            }
            if let Some(View {
                background: Background::Image(key),
                ..
            }) = view
            {
                in_use(key, "the view".into())?;
            }
        }
        // Untouched records were checked when set; only a release (above), or a removal or a re-set
        // of a slot a display's sub-frame draws, can strand them.
        if !after.removed.is_empty() || !after.set.is_empty() {
            for rec in after.records() {
                if let Body::Image(ImageSource::SubFrame { order, .. }) = &rec.body {
                    after.check_order(order, Some(rec.slot))?;
                }
            }
        }
        for rec in after.kept_sets() {
            after.check_record(rec)?;
        }
        if let Some(view) = &delta.view {
            after.check_view(view, None)?;
        }
        after.check_order(delta.order.as_deref().unwrap_or(&self.order), None)
    }

    /// Draws the frame at `width x height` pixels, its tiles spread over `threads` threads.
    pub fn draw(&self, width: u32, height: u32, threads: usize) -> Result<Vec<u8>, DrawError> {
        Ok(self.render(width, height, threads)?.rgba)
    }

    /// Draws the frame, and reports how each pixel was resolved.
    ///
    /// This build draws the background (the view's colour, or its image cropped as CE crops it)
    /// and `mesh`, `path`, `points` and `instances` records. It refuses, whole, any frame whose
    /// order draws what it cannot: `face_group` and `image` records until their drawers land, a
    /// record carrying a field it does not draw yet, reserved or unknown kinds, unknown feature
    /// bits, and unknown light models.
    pub fn render(&self, width: u32, height: u32, threads: usize) -> Result<Rendered, DrawError> {
        self.render_with(width, height, threads, &[])
    }

    /// Draws the frame as [`Stream::render`] does, with `extra` producers drawing into the same
    /// passes after the stream's own: a probe's instrument (a producer that sends every pixel to
    /// the sample resolve, say), never part of what a stream draws.
    pub fn render_with(
        &self,
        width: u32,
        height: u32,
        threads: usize,
        extra: &[&dyn Producer],
    ) -> Result<Rendered, DrawError> {
        if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
            return Err(DrawError::Invalid(format!(
                "a frame of {width} x {height} pixels (each side 1 to {MAX_SIDE})"
            )));
        }
        if threads == 0 {
            return Err(DrawError::Invalid(
                "drawing needs at least one thread".into(),
            ));
        }
        let view = self.view.as_ref().ok_or(DrawError::NoView)?;
        if let LightModel::Other { code, .. } = &view.light {
            return Err(DrawError::Unsupported(Unsupported {
                kind: "view".into(),
                slot: None,
                object: None,
                reason: format!("light model {code} is not one this build knows"),
            }));
        }
        let shading = matches!(&view.camera, Camera::ThreeD(c) if c.should_apply_shading);
        if let Some(refused) = self
            .order
            .iter()
            .find_map(|e| refusal(&self.records[&e.slot], &self.shapes, shading))
        {
            return Err(DrawError::Unsupported(refused));
        }
        let frame = match &view.background {
            Background::Color(c) => Frame::filled(width, height, composite::background_pixel(*c)),
            Background::Image(key) => {
                let Some(ShapeData::Image {
                    width: iw,
                    height: ih,
                    rgba,
                }) = self.shapes.get(key).map(|s| s.data())
                else {
                    unreachable!("apply checked the background image");
                };
                Frame::cropped(width, height, *iw, *ih, rgba).ok_or_else(|| {
                    DrawError::Invalid(format!(
                        "the background image ({iw} x {ih}) is smaller than the frame \
                         ({width} x {height})"
                    ))
                })?
            }
        };
        let (mut meshes, mut paths, mut clouds) = (Vec::new(), Vec::new(), Vec::new());
        for (order, e) in self.order.iter().enumerate() {
            let record = &self.records[&e.slot];
            let order = order as u32;
            let shape = record
                .geometry
                .shape_a
                .and_then(|k| self.shapes.get(&k))
                .map(|s| s.data());
            let table = |key: &Key| &self.paints[key];
            match (record.kind, shape, &record.body) {
                (
                    RecordKind::Mesh,
                    Some(ShapeData::Mesh {
                        positions,
                        triangles,
                        normals,
                    }),
                    _,
                ) => {
                    if let Some(fill) = instance::one(&record.paint.fill) {
                        meshes.push(MeshDraw {
                            record,
                            positions,
                            triangles,
                            normals: normals.as_deref(),
                            place: record.geometry.place_a,
                            fill,
                            order,
                            sub: 0,
                        });
                    }
                }
                (RecordKind::Path, Some(ShapeData::Path { points }), _) => {
                    paths.push(instance::path_record(record, points, order));
                }
                (RecordKind::Points, Some(ShapeData::Path { points }), Body::Points(p)) => {
                    clouds.push(instance::points(record, points, table(&p.colors), order));
                }
                (RecordKind::Instances, Some(shape), Body::Instances(body)) => match shape {
                    ShapeData::Path { points } => instance::path_members(
                        record,
                        points,
                        body,
                        table(&body.fill),
                        table(&body.stroke),
                        order,
                        &mut paths,
                    ),
                    ShapeData::Mesh {
                        positions,
                        triangles,
                        normals,
                    } => instance::mesh_members(
                        record,
                        positions,
                        triangles,
                        normals.as_deref(),
                        body,
                        table(&body.fill),
                        order,
                        &mut meshes,
                    ),
                    _ => unreachable!("apply checked the instances record's shape"),
                },
                _ => unreachable!("drawing refused every other record"),
            }
        }
        let meshes = mesh::lower(view, width, height, &meshes);
        let mut lowering = path::Lowering::new(view, width, height);
        for draw in &paths {
            lowering.path(draw);
        }
        let paths = lowering.finish();
        let blocks = points::lower(view, width, height, &clouds);
        let mut producers: Vec<&dyn Producer> = vec![&meshes, &paths, &blocks];
        producers.extend_from_slice(extra);
        tiles::render(frame, &producers, threads).map_err(|e| DrawError::Threads(e.to_string()))
    }

    pub fn view(&self) -> Option<&View> {
        self.view.as_ref()
    }

    pub fn record(&self, slot: u32) -> Option<&Record> {
        self.records.get(&slot)
    }

    pub fn order(&self) -> &[OrderEntry] {
        &self.order
    }

    pub fn object(&self, object: u32) -> Option<&ObjectInfo> {
        self.objects.get(&object)
    }

    pub fn has_shape(&self, key: &Key) -> bool {
        self.shapes.contains_key(key)
    }
}

/// Whether a shape held under a key and one uploaded under it have one content. A key is computed
/// from, or checked against, the contents of every kind this build knows, so two such shapes of one
/// kind are the same; a kind it does not know carries the key it was stated with, so its bytes are
/// compared.
fn same_content(held: &ShapeData, uploaded: &ShapeData) -> bool {
    match (held, uploaded) {
        (
            ShapeData::Reserved { code, payload },
            ShapeData::Reserved {
                code: other,
                payload: other_payload,
            },
        ) => code == other && payload == other_payload,
        (ShapeData::Reserved { .. }, _) | (_, ShapeData::Reserved { .. }) => false,
        (a, b) => std::mem::discriminant(a) == std::mem::discriminant(b),
    }
}

/// The longest side of a frame this build draws, in pixels.
pub const MAX_SIDE: u32 = 16384;

/// A shape that contradicts itself: a triangle naming a vertex the mesh lacks, normals that are not
/// one per vertex, or an image whose pixels are not its width times its height.
fn check_shape(shape: &Shape) -> Result<(), StreamError> {
    let invalid = |why: String| {
        Err(StreamError::Invalid(format!(
            "{}: {why}",
            shape.key().hex()
        )))
    };
    match shape.data() {
        ShapeData::Mesh {
            positions,
            triangles,
            normals,
        } => {
            if triangles
                .iter()
                .flatten()
                .any(|i| *i as usize >= positions.len())
            {
                return invalid(format!(
                    "a mesh's triangle names a vertex past its {}",
                    positions.len()
                ));
            }
            if normals.as_ref().is_some_and(|n| n.len() != positions.len()) {
                return invalid("a mesh needs one normal per vertex".into());
            }
        }
        ShapeData::Image {
            width,
            height,
            rgba,
        } => {
            let size = (*width as usize)
                .checked_mul(*height as usize)
                .and_then(|n| n.checked_mul(4));
            if size != Some(rgba.len()) {
                return invalid(format!(
                    "an image of {width} x {height} pixels in {} bytes",
                    rgba.len()
                ));
            }
        }
        ShapeData::Path { .. } | ShapeData::Reserved { .. } => {}
    }
    Ok(())
}

/// What of a drawn record's fixed fields this build does not draw yet, if anything: blends,
/// reveals, paint blends and clips for every kind; for meshes (and instances of a mesh) strokes,
/// gradient fills and dashes; for paths (and instances of a path) CE's lighting under a camera
/// that shades (`shading`), gradient paints and dashes it cannot read.
fn field_refusal(rec: &Record, mesh: bool, shading: bool) -> Option<String> {
    let kind = rec.kind.name();
    let p = &rec.paint;
    let drawn = |paint: &Paint, width: Option<f64>| {
        !matches!(paint, Paint::None) && width.is_none_or(|w| w > 0.0)
    };
    let gradient = |paint: &Paint| matches!(paint, Paint::Two(_) | Paint::Table(_));
    let strokes = [
        (&p.stroke, p.stroke_width),
        (&p.background_stroke, p.background_stroke_width),
    ];
    let not_yet = |what: &str| Some(format!("a {kind}'s {what} is not drawn yet"));
    if rec.geometry.blend.is_some() {
        not_yet("blend")
    } else if rec.geometry.reveal.is_some() {
        not_yet("reveal")
    } else if p.paint_b.is_some() {
        not_yet("paint blend")
    } else if rec.clip.is_some() {
        not_yet("clip")
    } else if rec.kind == RecordKind::Points {
        None
    } else if mesh {
        if strokes.iter().any(|(paint, w)| drawn(paint, Some(*w))) {
            not_yet("stroke")
        } else if gradient(&p.fill) {
            not_yet("gradient fill")
        } else if rec.dash.is_some() {
            not_yet("dash")
        } else {
            None
        }
    } else if shading
        && rec.depth.shade_in_3d
        && (drawn(&p.fill, None) || strokes.iter().any(|(paint, w)| drawn(paint, Some(*w))))
    {
        // CE lights each paint into two stops, at the path's start and end corners
        // (`ThreeDCamera.modified_rgbas`), drawn as a linear gradient: drawn unlit it would be a
        // wrong frame.
        not_yet("shading")
    } else if gradient(&p.fill)
        || strokes
            .iter()
            .any(|(paint, w)| drawn(paint, Some(*w)) && gradient(paint))
    {
        not_yet("gradient paint")
    } else if let Some(d) = rec.dash
        && !(d.period > 0.0
            && d.period.is_finite()
            && d.duty > 0.0
            && d.duty <= 1.0
            && d.phase.is_finite())
    {
        Some(format!(
            "a {kind}'s dash needs a finite period above 0, a duty in (0, 1] and a finite phase"
        ))
    } else {
        None
    }
}

/// Why this build refuses to draw a record, if it does, under a camera that shades or not
/// (`shading`). It draws `mesh`, `path`, `points` and `instances` records; `face_group` and
/// `image` records wait for their drawers.
fn refusal(rec: &Record, shapes: &HashMap<Key, Shape>, shading: bool) -> Option<Unsupported> {
    let prototype = rec
        .geometry
        .shape_a
        .and_then(|k| shapes.get(&k))
        .map(|s| s.data());
    let reason = if rec.features & !KNOWN_FEATURES != 0 {
        format!(
            "feature bits {:#x} are not ones this build knows",
            rec.features & !KNOWN_FEATURES
        )
    } else if let RecordKind::Unknown(code) = rec.kind {
        format!("record kind {code} is not one this build knows")
    } else if rec.kind.is_reserved() {
        format!("{} is a reserved kind", rec.kind.name())
    } else if matches!(
        rec.kind,
        RecordKind::Mesh | RecordKind::Path | RecordKind::Points | RecordKind::Instances
    ) {
        let mesh = matches!(prototype, Some(ShapeData::Mesh { .. }));
        if let Some(reason) = field_refusal(rec, mesh, shading) {
            reason
        } else if let (Body::Instances(body), Some(data)) = (&rec.body, prototype) {
            // Each instance's parts take sub-orders of their own, in 32 bits.
            let per = match data {
                ShapeData::Mesh { triangles, .. } => triangles.len(),
                _ => 3,
            };
            if (body.placements.len() as u64) * (per as u64) <= u32::MAX as u64 {
                return None;
            }
            "an instances record of more parts than 2^32".into()
        } else if let (Body::Points(body), Some(ShapeData::Path { points: positions })) =
            (&rec.body, prototype)
        {
            // Each block pixel's write takes a sub-order of its own (its offset, then its point),
            // in 32 bits.
            let side = points::side(body.radius) as u128;
            let writes = side
                .checked_mul(side)
                .and_then(|s| s.checked_mul(positions.len() as u128));
            if writes.is_some_and(|w| w <= u32::MAX as u128 + 1) {
                return None;
            }
            "a points record of more block pixels than 2^32".into()
        } else {
            return None;
        }
    } else {
        format!("this build does not draw {} records yet", rec.kind.name())
    };
    Some(Unsupported {
        kind: rec.kind.name(),
        slot: Some(rec.slot),
        object: Some(rec.object),
        reason,
    })
}
