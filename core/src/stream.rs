//! A Stream: what a drawer holds across frames. Deltas change it, atomically; `draw` turns it into
//! one frame's pixels, or refuses the whole frame when any drawn record carries what this build
//! cannot draw (`render.unsupported`).

use crate::composite::{self, Frame, Producer, Rendered};
use crate::mesh::{self, MeshDraw};
use crate::xf::*;
use crate::{geometry, instance, path, points, tiles};
use std::borrow::Cow;
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

    /// Checks an order against the records the delta leaves. Displays in each layer capture in
    /// rank order before its entries draw; self capture reads the picture as it then stands.
    fn check_order(&self, order: &[OrderEntry]) -> Result<(), StreamError> {
        let mut ranks = HashMap::new();
        for e in order {
            let rec = self.record(e.slot).ok_or_else(|| {
                StreamError::Missing(format!("the order draws slot {}, which is not set", e.slot))
            })?;
            if let Body::Image(ImageSource::SubFrame { rank, .. }) = &rec.body
                && let Some(other) = ranks.insert((e.layer as u8, *rank), e.slot)
                && other != e.slot
            {
                return Err(StreamError::Invalid(format!(
                    "display slots {other} and {} share capture rank {rank} in layer {}",
                    e.slot, e.layer as u8
                )));
            }
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
        if (rec.features & FEATURE_BACKGROUND_IMAGE != 0) != rec.fill_image.is_some() {
            return Err(StreamError::Invalid(
                "background_image feature bit and fill_image field disagree".into(),
            ));
        }
        if let Some(key) = &rec.fill_image {
            self.needs_shape(rec, Some(key), &["image"])?;
        }
        if (rec.features & FEATURE_PRE_PROJECTED != 0) != rec.pre_projected.is_some() {
            return Err(StreamError::Invalid(
                "pre_projected feature bit and tail field disagree".into(),
            ));
        }
        if let Some(p) = &rec.pre_projected {
            if !matches!(
                rec.kind,
                RecordKind::Path | RecordKind::Points | RecordKind::Image
            ) {
                return Err(StreamError::Invalid(format!(
                    "pre_projected is not valid on {} records",
                    rec.kind.name()
                )));
            }
            self.needs_shape(rec, Some(&p.points), &["path"])?;
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
            (RecordKind::Image, Body::Image(ImageSource::SubFrame { view, order, .. })) => {
                if shape_a.is_some() {
                    return Err(StreamError::Invalid(format!(
                        "slot {}: a sub-frame display has no shape",
                        rec.slot
                    )));
                }
                self.check_view(view, Some(rec.slot))?;
                self.check_order(order)?;
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
                    after.check_order(order)?;
                }
            }
        }
        for rec in after.kept_sets() {
            after.check_record(rec)?;
        }
        if let Some(view) = &delta.view {
            after.check_view(view, None)?;
        }
        after.check_order(delta.order.as_deref().unwrap_or(&self.order))
    }

    /// Draws the frame at `width x height` pixels, its tiles spread over `threads` threads.
    pub fn draw(&self, width: u32, height: u32, threads: usize) -> Result<Vec<u8>, DrawError> {
        Ok(self.render(width, height, threads)?.rgba)
    }

    /// Draws the frame, and reports how each pixel was resolved.
    ///
    /// This build draws the background (the view's colour, or its image cropped as CE crops it)
    /// and `mesh`, `path`, `face_group`, `points` and `instances` records. It refuses, whole, any
    /// frame whose order draws what it cannot: `image` records (Cairo's overlay draws them), a
    /// record carrying a field it does not draw yet, reserved or unknown kinds, unknown feature
    /// bits, and unknown light models.
    pub fn render(&self, width: u32, height: u32, threads: usize) -> Result<Rendered, DrawError> {
        self.render_with(width, height, threads, &[])
    }

    /// Draws the frame as [`Stream::render`] does, without the overlay's entries
    /// ([`is_overlay`]): a 3D scene's frame before Cairo draws its overlay over it, in CE's order.
    /// Those entries are neither drawn nor refused here; the others keep their places in the
    /// order, so ties break as in the whole order.
    pub fn render_under_overlay(
        &self,
        width: u32,
        height: u32,
        threads: usize,
    ) -> Result<Rendered, DrawError> {
        self.render_entries(width, height, threads, &[], true)
    }

    /// The order's entries in the overlay ([`is_overlay`]), in order.
    pub fn overlay_entries(&self) -> Vec<OrderEntry> {
        self.order
            .iter()
            .filter(|e| is_overlay(&self.records[&e.slot]))
            .copied()
            .collect()
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
        self.render_entries(width, height, threads, extra, false)
    }

    fn render_entries(
        &self,
        width: u32,
        height: u32,
        threads: usize,
        extra: &[&dyn Producer],
        without_overlay: bool,
    ) -> Result<Rendered, DrawError> {
        check_size(width, height)?;
        if threads == 0 {
            return Err(DrawError::Invalid(
                "drawing needs at least one thread".into(),
            ));
        }
        self.with_producers(width, height, extra, without_overlay, |frame, producers| {
            tiles::render(frame, producers, threads).map_err(|e| DrawError::Threads(e.to_string()))
        })?
    }

    /// The leaves the core composites at each pixel of `x0..x1` by `y0..y1` (clipped to the
    /// view's frame) that holds a fragment, from the frame [`Stream::render_under_overlay`] draws:
    /// what a pixel's colour is made of, read from the same passes the resolves read
    /// (`contributors.rs`). The overlay's entries are Cairo's and are not reported. Refused as
    /// drawing refuses.
    pub fn contributors(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> Result<BTreeMap<(u32, u32), crate::contributors::PixelContributors>, DrawError> {
        let mut out = BTreeMap::new();
        let leaf = |f: &composite::Fragment| self.leaf_of(f);
        self.visit_contributors(x0, y0, x1, y1, |x, y, frags, coverage, depths, _| {
            out.insert(
                (x, y),
                crate::contributors::of_pixel(frags, coverage, depths, &leaf),
            );
        })?;
        Ok(out)
    }

    /// The edges the rasterizer consumes for every leaf [`Stream::contributors`] reports with
    /// an exact area in the same box, from the same pass: per leaf `(object, part)`, each of its
    /// layers once (a fill, a stroke's outline after the stroker), as `[x0, y0, x1, y1, dir]` in
    /// device pixels (`scanline::Edge`: oriented downward, `dir` the winding it adds). A leaf
    /// drawn by masks (a mesh's triangles, a cloud's blocks) has none.
    #[allow(clippy::type_complexity)]
    pub fn contributor_edges(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> Result<BTreeMap<(u32, u32), Vec<Vec<[f64; 5]>>>, DrawError> {
        let mut out: BTreeMap<(u32, u32), Vec<Vec<[f64; 5]>>> = BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        self.visit_contributors(x0, y0, x1, y1, |_, _, frags, _, _, producers| {
            for f in frags {
                if let composite::Coverage::Area { id, .. } = f.coverage
                    && seen.insert((f.source(), id))
                    && let Some(edges) = producers[f.source() as usize].edges(id)
                {
                    out.entry(self.leaf_of(f)).or_default().push(
                        edges
                            .iter()
                            .map(|e| [e.x0, e.y0, e.x1, e.y1, e.dir as f64])
                            .collect(),
                    );
                }
            }
        })?;
        Ok(out)
    }

    /// Visits each pixel of `x0..x1` by `y0..y1` (clipped to the view's frame) that holds a
    /// fragment, with its fragments, their coverage at each sub-sample, their depths and the
    /// producers that added them, from the frame [`Stream::render_under_overlay`] draws: the one pass the
    /// contributor reports read.
    #[allow(clippy::type_complexity)]
    fn visit_contributors(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        mut visit: impl FnMut(
            u32,
            u32,
            &[composite::Fragment],
            &[[u8; composite::SAMPLES]],
            &[composite::Depths],
            &[&dyn Producer],
        ),
    ) -> Result<(), DrawError> {
        let view = self.view.as_ref().ok_or(DrawError::NoView)?;
        let (width, height) = (view.pixel_width, view.pixel_height);
        let keep = composite::Rect {
            x0: x0.min(width),
            y0: y0.min(height),
            x1: x1.min(width),
            y1: y1.min(height),
        };
        self.with_producers(width, height, &[], true, |_, producers| {
            if keep.x0 >= keep.x1 || keep.y0 >= keep.y1 {
                return;
            }
            for rect in tiles::tile_rects(width, height) {
                if rect.x1 <= keep.x0
                    || rect.x0 >= keep.x1
                    || rect.y1 <= keep.y0
                    || rect.y0 >= keep.y1
                {
                    continue;
                }
                tiles::passes(rect, producers)
                    .visit_pixels(keep, |x, y, frags, coverage, depths| {
                        visit(x, y, frags, coverage, depths, producers)
                    });
            }
        })
    }

    /// The leaf a fragment drawn from this stream belongs to, as `(object, part)`: the object of
    /// the record its order entry draws and the entry's part (a face of a face group), except
    /// that an `instances` record's member reports its own object (instance `i` of a path
    /// prototype draws sub-orders `3i..3i + 3`, of a mesh prototype `t * i..t * (i + 1)` for its
    /// `t` triangles: `instance.rs`).
    fn leaf_of(&self, f: &composite::Fragment) -> (u32, u32) {
        let entry = &self.order[f.order as usize];
        let record = &self.records[&entry.slot];
        if let Body::Instances(body) = &record.body {
            let per = match record
                .geometry
                .shape_a
                .and_then(|k| self.shapes.get(&k))
                .map(|s| s.data())
            {
                Some(ShapeData::Mesh { triangles, .. }) => triangles.len().max(1) as u32,
                _ => 3,
            };
            let i = (f.sub / per) as usize;
            return (body.objects[i], 0);
        }
        (record.object, entry.part)
    }

    /// The frame's background and its producers, lowered from the records the order draws (all,
    /// or all but the overlay's), handed to `f`. Refused as [`Stream::render`] refuses.
    fn with_producers<T>(
        &self,
        width: u32,
        height: u32,
        extra: &[&dyn Producer],
        without_overlay: bool,
        f: impl FnOnce(Frame, &[&dyn Producer]) -> T,
    ) -> Result<T, DrawError> {
        let drawn = |e: &OrderEntry| !(without_overlay && is_overlay(&self.records[&e.slot]));
        check_size(width, height)?;
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
        let ce_light = view.light == LightModel::Ce;
        if let Some(refused) = self.order.iter().filter(|e| drawn(e)).find_map(|e| {
            refusal(
                &self.records[&e.slot],
                &self.shapes,
                &self.paints,
                shading,
                ce_light,
            )
        }) {
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
            if !drawn(e) {
                continue;
            }
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
                (RecordKind::FaceGroup, Some(ShapeData::Path { points }), Body::FaceGroup(f)) => {
                    paths.push(instance::face(record, points, f, e.part, table, order));
                }
                (RecordKind::Path, Some(ShapeData::Path { points: own }), _)
                    if let Some(tail) = &record.pre_projected =>
                {
                    let placed = path_shape(&self.shapes, &tail.points)
                        .expect("apply checked the tail's points");
                    paths.push(instance::produced_path(
                        record, own, placed, tail, table, order,
                    ));
                }
                (RecordKind::Path, Some(ShapeData::Path { .. }), _) => {
                    let (points, place) =
                        drawn_points(record, &self.shapes).expect("drawing refused the others");
                    let paint = geometry::drawn_paint(&record.paint, table)
                        .expect("drawing refused a paint it cannot mix");
                    paths.push(instance::path_record(record, points, place, paint, order));
                }
                (RecordKind::Points, Some(ShapeData::Path { points }), Body::Points(p)) => {
                    let positions = match &record.pre_projected {
                        Some(tail) => path_shape(&self.shapes, &tail.points)
                            .expect("apply checked the tail's points"),
                        None => points,
                    };
                    clouds.push(instance::points(
                        record,
                        points,
                        positions,
                        table(&p.colors),
                        order,
                    ));
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
        let mut registry = crate::depth::Registry::new(view);
        let meshes = mesh::lower(view, width, height, &meshes, &mut registry);
        let mut lowering = path::Lowering::new(view, width, height);
        let planes = lowering.coplanar(&paths, &mut registry);
        for (draw, plane) in paths.iter().zip(planes) {
            lowering.path(draw, plane);
        }
        let paths = lowering.finish();
        let blocks = points::lower(view, width, height, &clouds);
        let mut producers: Vec<&dyn Producer> = vec![&meshes, &paths, &blocks];
        producers.extend_from_slice(extra);
        Ok(f(frame, &producers))
    }

    /// Per slot of `slots`, how each path its record draws lies, decided from the points the
    /// stream holds ([`path::Lowering::planarity`], codes of [`crate::depth::Planarity`]), in the
    /// order of the record's `planes`: what a writer that snaps the record's geometry afterwards
    /// carries with it. A slot holding no record that draws paths (or one the drawer would refuse)
    /// has no entry. Nothing is drawn.
    pub fn planes(&self, slots: &[u32]) -> Result<BTreeMap<u32, Vec<u8>>, DrawError> {
        self.each_paths(slots, |lowering, paths| {
            Some(paths.iter().map(|d| lowering.planarity(d).code()).collect())
        })
    }

    /// Per slot of `slots` holding a record the camera shades, the normals each path it draws is
    /// lit by at its two lit corners ([`path::Lowering::normals`]), decided from the points the
    /// stream holds as a writer that snaps them to `grid` carries them: two per path, in the order
    /// of the record's `planes`. Other slots have no entry. Nothing is drawn.
    pub fn normals(
        &self,
        slots: &[u32],
        grid: f64,
    ) -> Result<BTreeMap<u32, Vec<[f64; 3]>>, DrawError> {
        self.each_paths(slots, |lowering, paths| {
            let mut found = Vec::new();
            for draw in paths {
                found.extend(lowering.normals(draw, grid)?);
            }
            Some(found)
        })
    }

    /// Per slot of `slots` holding a record the camera shades, which of the two lit corners of each
    /// path it draws the drawer lights by the face's Newell normal ([`path::Lowering::newell_corners`]):
    /// two per path, in the order of the record's `planes`. Other slots have no entry. Nothing is
    /// drawn.
    pub fn newell_corners(
        &self,
        slots: &[u32],
    ) -> Result<BTreeMap<u32, Vec<[bool; 2]>>, DrawError> {
        self.each_paths(slots, |lowering, paths| {
            let mut found = Vec::new();
            for draw in paths {
                found.push(lowering.newell_corners(draw)?);
            }
            Some(found)
        })
    }

    /// Per slot of `slots` holding a record the camera shades, the two stops each layer of each
    /// path it draws is painted with ([`path::Lowering::face_stops`]: background stroke, fill and
    /// stroke, lit at the path's lit corners, with the record's group opacity), one entry per path
    /// in the order of the record's `planes`, `None` for a path the drawer draws nothing of. Other
    /// slots (unshaded, pre-projected, refused) have no entry. Nothing is drawn.
    pub fn face_stops(
        &self,
        slots: &[u32],
    ) -> Result<BTreeMap<u32, Vec<Option<path::FaceStops>>>, DrawError> {
        self.each_paths(slots, |lowering, paths| {
            let found: Vec<_> = paths.iter().map(|d| lowering.face_stops(d)).collect();
            found.iter().any(Option::is_some).then_some(found)
        })
    }

    /// `answer(lowering, paths)` per slot of `slots` holding a record that draws paths the drawer
    /// would not refuse, `paths` those it draws, in order; a `None` answer has no entry.
    fn each_paths<T>(
        &self,
        slots: &[u32],
        answer: impl Fn(&path::Lowering<'_>, &[path::PathDraw<'_>]) -> Option<T>,
    ) -> Result<BTreeMap<u32, T>, DrawError> {
        let view = self.view.as_ref().ok_or(DrawError::NoView)?;
        let lowering = path::Lowering::new(view, view.pixel_width, view.pixel_height);
        let shading = matches!(&view.camera, Camera::ThreeD(c) if c.should_apply_shading);
        let ce_light = view.light == LightModel::Ce;
        let mut found = BTreeMap::new();
        for slot in slots {
            let Some(record) = self.records.get(slot) else {
                continue;
            };
            if refusal(record, &self.shapes, &self.paints, shading, ce_light).is_some() {
                continue;
            }
            let shape = record
                .geometry
                .shape_a
                .and_then(|k| self.shapes.get(&k))
                .map(|s| s.data());
            let table = |key: &Key| &self.paints[key];
            let mut paths = Vec::new();
            match (record.kind, shape, &record.body) {
                (RecordKind::FaceGroup, Some(ShapeData::Path { points }), Body::FaceGroup(f)) => {
                    for part in 0..f.uv.len() as u32 {
                        paths.push(instance::face(record, points, f, part, table, 0));
                    }
                }
                (RecordKind::Path, Some(ShapeData::Path { points: own }), _)
                    if let Some(tail) = &record.pre_projected =>
                {
                    if let Some(placed) = path_shape(&self.shapes, &tail.points) {
                        paths.push(instance::produced_path(record, own, placed, tail, table, 0));
                    }
                }
                (RecordKind::Path, Some(ShapeData::Path { .. }), _) => {
                    if let (Some((points, place)), Ok(paint)) = (
                        drawn_points(record, &self.shapes),
                        geometry::drawn_paint(&record.paint, table),
                    ) {
                        paths.push(instance::path_record(record, points, place, paint, 0));
                    }
                }
                (
                    RecordKind::Instances,
                    Some(ShapeData::Path { points }),
                    Body::Instances(body),
                ) => instance::path_members(
                    record,
                    points,
                    body,
                    table(&body.fill),
                    table(&body.stroke),
                    0,
                    &mut paths,
                ),
                _ => continue,
            }
            if let Some(answer) = answer(&lowering, &paths) {
                found.insert(*slot, answer);
            }
        }
        Ok(found)
    }

    /// Per slot the order draws, the pixel box of its record's points as the camera places them
    /// for display and maps them to pixels by the mapping its kind is drawn through
    /// ([`crate::boxes::record_box`]); a record placing no point has no entry. Nothing is drawn,
    /// so this answers for records the drawer would refuse too.
    pub fn screen_boxes(&self) -> Result<BTreeMap<u32, crate::boxes::PixelBox>, DrawError> {
        let view = self.view.as_ref().ok_or(DrawError::NoView)?;
        let mut boxes = BTreeMap::new();
        for entry in &self.order {
            if boxes.contains_key(&entry.slot) {
                continue;
            }
            let record = &self.records[&entry.slot];
            let found = crate::boxes::record_box(view, record, |key| {
                self.shapes.get(key).map(|s| s.data())
            });
            if let Some(b) = found {
                boxes.insert(entry.slot, b);
            }
        }
        Ok(boxes)
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

/// Refuses a frame size the drawer does not draw.
fn check_size(width: u32, height: u32) -> Result<(), DrawError> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(DrawError::Invalid(format!(
            "a frame of {width} x {height} pixels (each side 1 to {MAX_SIDE})"
        )));
    }
    Ok(())
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

/// What of a drawn record's fixed fields this build does not draw yet, if anything, its paint
/// tables in `paints`: blends, reveals and paint blends of any kind but a path, a paint blend of
/// row counts numpy does not broadcast (`geometry::drawn_paint`), clips for every kind; for meshes (and instances of a mesh) strokes, gradient fills and dashes; for paths (and
/// instances of a path) CE's lighting under a camera that shades (`shading`) with a light model
/// other than CE's (`ce_light`), a gradient paint without its ends or on an instance, and dashes
/// it cannot read. A pre-projected record is judged by its tail's paints and ends (lit already, so
/// never refused for shading), and refused with a blend, reveal or paint blend.
fn field_refusal(
    rec: &Record,
    paints: &HashMap<Key, Stops>,
    mesh: bool,
    shading: bool,
    ce_light: bool,
) -> Option<String> {
    let kind = rec.kind.name();
    // A pre-projected record's paints are the colours its camera painted with, lit already, and
    // its gradient runs between the ends that camera placed.
    let tail = rec.pre_projected.as_ref();
    let shaded = tail.is_none() && shading && rec.depth.shade_in_3d;
    // A path's blend, reveal and paint blend are drawn as what they make of it (`geometry`); the
    // paint judged below is the one drawn.
    let shaped = rec.geometry.blend.is_some()
        || rec.geometry.reveal.is_some()
        || rec.paint.paint_b.is_some();
    // Tables are checked to exist before drawing.
    let table = |key: &Key| &paints[key];
    let mut p = match geometry::drawn_paint(&rec.paint, table) {
        Ok(p) => p,
        Err(what) => return Some(format!("a {kind}'s {what} is not drawn")),
    };
    let mut ends = rec.paint.gradient;
    if let Some(t) = tail {
        p.fill = geometry::rows(&t.fill, table);
        p.stroke = geometry::rows(&t.stroke, table);
        p.background_stroke = geometry::rows(&t.background_stroke, table);
        ends = t.gradient;
    }
    let p = &p;
    type Rows = Option<Vec<[f64; 4]>>;
    let drawn = |paint: &Rows, width: Option<f64>| paint.is_some() && width.is_none_or(|w| w > 0.0);
    let gradient = |paint: &Rows| paint.as_ref().is_some_and(|r| r.len() > 1);
    let strokes = [
        (&p.stroke, p.stroke_width),
        (&p.background_stroke, p.background_stroke_width),
    ];
    let not_yet = |what: &str| Some(format!("a {kind}'s {what} is not drawn yet"));
    if shaped && rec.kind != RecordKind::Path {
        // CE's blends and reveals are of paths alone; extraction states no other.
        not_yet(if rec.geometry.blend.is_some() {
            "blend"
        } else if rec.geometry.reveal.is_some() {
            "reveal"
        } else {
            "paint blend"
        })
    } else if shaped && tail.is_some() {
        // The tail is what its camera drew of the mobject as it stood; extraction never states
        // both.
        Some(format!(
            "a pre-projected {kind} with a blend, reveal or paint blend is not drawn"
        ))
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
    } else if shaded
        && !ce_light
        && (rec.kind == RecordKind::FaceGroup
            || drawn(&p.fill, None)
            || strokes.iter().any(|(paint, w)| drawn(paint, Some(*w))))
    {
        // CE lights each paint into two stops, at the path's start and end corners
        // (`ThreeDCamera.modified_rgbas`), drawn as a linear gradient; another light model has no
        // look for a path yet.
        not_yet("shading under a light model other than CE's")
    } else if (gradient(&p.fill)
        || strokes
            .iter()
            .any(|(paint, w)| drawn(paint, Some(*w)) && gradient(paint)))
        && (rec.kind == RecordKind::Instances || (!shaded && ends.is_none()))
    {
        // A path's gradient runs between its ends (or a shaded path's corners); an instance's
        // colours are rows of a table, one each.
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

/// Whether a 3D scene's frame leaves `rec` to Cairo's overlay, drawn over the core's frame in CE's
/// order: a record fixed in frame, an image (a display's included), or a path filled through a
/// background image.
pub fn is_overlay(rec: &Record) -> bool {
    rec.depth.space == Space::Overlay
        || rec.kind == RecordKind::Image
        || rec.features & FEATURE_BACKGROUND_IMAGE != 0
}

/// The points of the path shape `key` names, if it names one.
fn path_shape<'a>(shapes: &'a HashMap<Key, Shape>, key: &Key) -> Option<&'a [[f64; 3]]> {
    match shapes.get(key).map(|s| s.data()) {
        Some(ShapeData::Path { points }) => Some(points.as_slice()),
        _ => None,
    }
}

/// The points a `path` record draws and the placement they still take: its shape and its own
/// placement, or, under a reveal or a blend, the points those make of it
/// ([`geometry::path_points`]), placed already. `None` when it has no path shape, or blends two
/// shapes of different lengths.
fn drawn_points<'a>(
    rec: &Record,
    shapes: &'a HashMap<Key, Shape>,
) -> Option<(Cow<'a, [[f64; 3]]>, Affine)> {
    let path = |key: &Key| path_shape(shapes, key);
    let a = path(rec.geometry.shape_a.as_ref()?)?;
    let g = &rec.geometry;
    if g.blend.is_none() && g.reveal.is_none() {
        return Some((Cow::Borrowed(a), g.place_a));
    }
    let b = match &g.blend {
        Some(blend) => Some(path(&blend.shape_b)?),
        None => None,
    };
    Some((Cow::Owned(geometry::path_points(g, a, b)?), IDENTITY))
}

/// Why this build refuses to draw a record, if it does, under a camera that shades or not
/// (`shading`) and CE's light model or another (`ce_light`), its paint tables in `paints`. It draws
/// `mesh`, `path`, `face_group`, `points` and `instances` records, pre-projected ones included;
/// `image` records are Cairo's overlay's.
fn refusal(
    rec: &Record,
    shapes: &HashMap<Key, Shape>,
    paints: &HashMap<Key, Stops>,
    shading: bool,
    ce_light: bool,
) -> Option<Unsupported> {
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
    } else if rec.features & FEATURE_BACKGROUND_IMAGE != 0 {
        "background_image is drawn by Cairo only".into()
    } else if let RecordKind::Unknown(code) = rec.kind {
        format!("record kind {code} is not one this build knows")
    } else if rec.kind.is_reserved() {
        format!("{} is a reserved kind", rec.kind.name())
    } else if matches!(
        rec.kind,
        RecordKind::Mesh
            | RecordKind::Path
            | RecordKind::FaceGroup
            | RecordKind::Points
            | RecordKind::Instances
    ) {
        let mesh = matches!(prototype, Some(ShapeData::Mesh { .. }));
        if let Some(reason) = field_refusal(rec, paints, mesh, shading, ce_light) {
            reason
        } else if rec.kind == RecordKind::Path && drawn_points(rec, shapes).is_none() {
            "a path's blend of two shapes of different lengths".into()
        } else if let (Body::FaceGroup(faces), Some(ShapeData::Path { points })) =
            (&rec.body, prototype)
        {
            if !faces.uv.is_empty() && points.len() % faces.uv.len() == 0 {
                return None;
            }
            "a face group whose points do not divide into its faces".into()
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
        } else if let (Body::Points(body), Some(ShapeData::Path { points: own })) =
            (&rec.body, prototype)
        {
            let positions = match &rec.pre_projected {
                Some(tail) => path_shape(shapes, &tail.points).unwrap_or(&[]),
                None => own,
            };
            if rec.pre_projected.is_some()
                && paints.get(&body.colors).map(|t| t.rows().len()) != Some(positions.len())
            {
                return Some(Unsupported {
                    kind: rec.kind.name(),
                    slot: Some(rec.slot),
                    object: Some(rec.object),
                    reason: "a pre-projected points record placing other than one point per \
                             colour"
                        .into(),
                });
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order;

    /// A closed polygon's outline through `corners` in CE's cubic layout.
    fn ring(corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for (i, a) in corners.iter().enumerate() {
            let b = corners[(i + 1) % corners.len()];
            let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
            out.extend([*a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
        }
        out
    }

    /// How many times, over every sub-sample of the frame three or more fragments cover, a path's
    /// fill ranks in front of its own stroke drawn after it, in the rank the sample resolve builds
    /// there.
    // The sub-sample `k` indexes each fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn own_cover(s: &Stream, width: u32, height: u32) -> usize {
        let mut hits = 0;
        let mut out = Vec::new();
        s.visit_contributors(0, 0, width, height, |_, _, frags, cov, d, _| {
            let mut d = d.to_vec();
            composite::strokes_on_own_fills(frags, &mut d, |i| cov[i]);
            for k in 0..composite::SAMPLES {
                let items: Vec<usize> = (0..frags.len()).filter(|i| cov[*i][k] != 0).collect();
                if items.len() < 3 {
                    continue;
                }
                let at = |a: usize, b: usize| {
                    (
                        &frags[a],
                        d[a].samples[k],
                        d[a].err,
                        &frags[b],
                        d[b].samples[k],
                        d[b].err,
                    )
                };
                order::rank(
                    &items,
                    |a, b| {
                        let (fa, da, ea, fb, db, eb) = at(a, b);
                        a != b && order::at_sample(fa, da, ea, fb, db, eb).is_lt()
                    },
                    |a, b| {
                        let (fa, da, ea, fb, db, eb) = at(a, b);
                        order::strong_at(fa, da, ea, fb, db, eb)
                    },
                    |a, b| order::painted(&frags[a], &frags[b]),
                    &mut out,
                );
                for (p, &f) in out.iter().enumerate() {
                    let (fill, behind) = (&frags[f], &out[p + 1..]);
                    hits += behind
                        .iter()
                        .filter(|&&s| {
                            fill.sub % 3 == 1
                                && frags[s].sub > fill.sub
                                && order::own_layers(fill, &frags[s])
                        })
                        .count();
                }
            }
        })
        .expect("the stream draws");
        hits
    }

    /// A small saddle surface of 16 x 16 faces, each a path with its fill and its mesh lines (a
    /// stroke of CE's Surface width), seen from above at a slant: no mesh line ranks behind its
    /// own face's fill at any sub-sample, as CE draws each face's stroke over its fill.
    #[test]
    fn a_dense_surface_s_mesh_lines_never_rank_behind_their_own_face() {
        let (w, h) = (320, 180);
        let view = crate::probes::view(w, h, 65f64.to_radians(), -40f64.to_radians());
        let n = 16;
        let at = |i: usize, j: usize| {
            let (u, v) = (
                2.0 * i as f64 / n as f64 - 1.0,
                2.0 * j as f64 / n as f64 - 1.0,
            );
            [1.2 * u, 1.2 * v, 0.6 * (u * u - v * v)]
        };
        let mut items = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let slot = (j * n + i) as u32 + 1;
                let corners = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
                let shape = Shape::new(ShapeData::Path {
                    points: ring(&corners),
                });
                let mut r = Record::new(RecordKind::Path, slot, 1000 + slot);
                r.geometry.shape_a = Some(shape.key());
                r.paint.fill = Paint::One(if (i + j) % 2 == 0 {
                    [0.16, 0.38, 0.75, 1.0]
                } else {
                    [0.36, 0.62, 0.95, 1.0]
                });
                r.paint.stroke = Paint::One([0.85, 0.85, 0.85, 1.0]);
                r.paint.stroke_width = 0.5;
                r.depth.space = Space::World;
                items.push((shape, r));
            }
        }
        let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
        let mut s = Stream::new();
        s.apply(&crate::probes::first_delta(view, &items, &order))
            .expect("the delta applies");
        assert_eq!(own_cover(&s, w, h), 0);
    }

    /// A sub-sample of a fill's fringe: its cell reaches into the fill (coverage above 0) while
    /// its point lies outside the fill's outline (the exact point-in-outline mask).
    struct FringeSample {
        point: [f64; 2],
        /// The fill's key there, as the resolve reads it, and its error bound.
        key: f64,
        err: f64,
        plane: composite::Plane,
        /// The outline's point nearest `point`, and how far it is.
        nearest: [f64; 2],
        distance: f64,
    }

    /// The point of `edges` nearest `p`, and its distance.
    fn nearest_on(edges: &[crate::scanline::Edge], p: [f64; 2]) -> ([f64; 2], f64) {
        let mut best = ([f64::NAN; 2], f64::INFINITY);
        for e in edges {
            let (dx, dy) = (e.x1 - e.x0, e.y1 - e.y0);
            let t =
                (((p[0] - e.x0) * dx + (p[1] - e.y0) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
            let q = [e.x0 + t * dx, e.y0 + t * dy];
            let d = libm::hypot(q[0] - p[0], q[1] - p[1]);
            if d < best.1 {
                best = (q, d);
            }
        }
        best
    }

    /// A sub-sample inside a fill's outline: its point, the fill's key there and its plane.
    type InsideSample = ([f64; 2], f64, composite::Plane);

    /// A lone face through `corners`, filled and not stroked, in `view`: its fringe sub-samples
    /// and its inside ones.
    // The sub-sample `k` indexes each fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn fill_samples(view: View, corners: &[[f64; 3]]) -> (Vec<FringeSample>, Vec<InsideSample>) {
        let (w, h) = (view.pixel_width, view.pixel_height);
        let shape = Shape::new(ShapeData::Path {
            points: ring(corners),
        });
        let r = crate::probes::fill_record(1, &shape, [0.5, 0.5, 0.5, 1.0], Space::World);
        let mut s = Stream::new();
        s.apply(&crate::probes::first_delta(view, &[(shape, r)], &[1]))
            .expect("the delta applies");
        let (mut fringe, mut inside) = (Vec::new(), Vec::new());
        s.visit_contributors(0, 0, w, h, |x, y, frags, cov, d, producers| {
            for (i, f) in frags.iter().enumerate() {
                let composite::Coverage::Area { id, .. } = f.coverage else {
                    continue;
                };
                let producer = producers[f.source() as usize];
                let edges = producer.edges(id).expect("a path gives its edges");
                // Each end of an edge is an end of another: no horizontal edge, which the
                // rasterizer's edges leave out, is missing from the outline they give.
                for e in edges {
                    for end in [[e.x0, e.y0], [e.x1, e.y1]] {
                        let ends = edges
                            .iter()
                            .filter(|o| [[o.x0, o.y0], [o.x1, o.y1]].contains(&end))
                            .count();
                        assert!(ends >= 2, "a horizontal edge at {end:?}");
                    }
                }
                let mask = producer.mask(id, x, y);
                for k in 0..composite::SAMPLES {
                    let point = composite::sample_point(x, y, k);
                    if mask & (1 << k) != 0 {
                        inside.push((point, d[i].samples[k], f.depth));
                    } else if cov[i][k] != 0 {
                        let (nearest, distance) = nearest_on(edges, point);
                        fringe.push(FringeSample {
                            point,
                            key: d[i].samples[k],
                            err: d[i].err,
                            plane: f.depth,
                            nearest,
                            distance,
                        });
                    }
                }
            }
        })
        .expect("the stream draws");
        (fringe, inside)
    }

    /// A face of a regular octahedron seen two degrees from edge-on, a sliver 1.6 to 3 pixels
    /// across: its depth runs steeply across the screen, so its plane read beyond its outline is
    /// nearer or farther than any point of the face. Wherever a sub-sample's cell reaches into
    /// its fill and its point lies outside the outline, the fill's key is its plane read at the
    /// outline's point nearest it: the depth of the face's edge there. Inside, its plane at the
    /// point.
    #[test]
    fn a_steep_fill_s_fringe_sub_samples_read_the_depth_of_its_edge() {
        let view = crate::probes::view(854, 480, 1.3089969158172607, 0.5235987901687622);
        let o = 0.70709228515625;
        let (fringe, inside) = fill_samples(view, &[[0.0, -o, 0.0], [0.0, 0.0, -o], [o, 0.0, 0.0]]);
        assert!(fringe.len() > 100, "{} fringe sub-samples", fringe.len());
        let steep = fringe
            .iter()
            .filter(|f| {
                (f.plane.at(f.point[0], f.point[1]) - f.plane.at(f.nearest[0], f.nearest[1])).abs()
                    > 1e-6
            })
            .count();
        assert!(
            steep > fringe.len() / 2,
            "{steep} sub-samples beyond the outline read the plane off its edge by 1e-6"
        );
        for f in &fringe {
            let edge = f.plane.at(f.nearest[0], f.nearest[1]);
            assert!(
                (f.key - edge).abs() <= f.err,
                "at {:?}, {:.3} px outside the outline: key {} against the edge's {} (err {:e})",
                f.point,
                f.distance,
                f.key,
                edge,
                f.err
            );
        }
        for (p, key, plane) in &inside {
            assert_eq!(*key, plane.at(p[0], p[1]), "inside, at {p:?}");
        }
    }

    /// A face seen at a slant: beyond its outline its fill's key moves from its plane at the point
    /// by at most the plane's slope times the point's distance from the outline, and inside it
    /// stays the plane.
    #[test]
    fn an_ordinary_fill_s_fringe_keys_move_by_at_most_slope_times_distance() {
        let view = crate::probes::view(320, 180, 65f64.to_radians(), -40f64.to_radians());
        let (fringe, inside) = fill_samples(
            view,
            &[[-1.0, -0.5, 0.2], [1.0, -0.8, -0.1], [0.2, 1.0, 0.3]],
        );
        assert!(fringe.len() > 100, "{} fringe sub-samples", fringe.len());
        assert!(inside.len() > 1000, "{} inside sub-samples", inside.len());
        for f in &fringe {
            let slope = libm::hypot(f.plane.b, f.plane.c);
            let moved = (f.key - f.plane.at(f.point[0], f.point[1])).abs();
            assert!(
                moved <= slope * f.distance + f.err,
                "at {:?}, {:.3} px outside: moved {moved:e}, slope {slope:e}",
                f.point,
                f.distance
            );
        }
        for (p, key, plane) in &inside {
            assert_eq!(*key, plane.at(p[0], p[1]), "inside, at {p:?}");
        }
    }

    fn dot3(u: [f64; 3], v: [f64; 3]) -> f64 {
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    }

    fn cross3(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    }

    /// The projection of a probe view's 3D camera.
    fn projection_of(view: &View) -> crate::project::Projection {
        let Camera::ThreeD(c) = &view.camera else {
            unreachable!("a probe view is 3D")
        };
        crate::project::Projection::new(c, view.frame_center)
    }

    /// The cubic arc of the circle about `centre` from `a` to `b` (both on it, less than half a
    /// turn apart), its handles along the tangents, `4/3 tan(angle / 4)` of the radius long.
    fn arc(centre: [f64; 3], a: [f64; 3], b: [f64; 3]) -> [[f64; 3]; 4] {
        let (ra, rb): ([f64; 3], [f64; 3]) = (
            std::array::from_fn(|k| a[k] - centre[k]),
            std::array::from_fn(|k| b[k] - centre[k]),
        );
        let r2 = dot3(ra, ra);
        let m = cross3(ra, rb);
        let m = m.map(|v| v / dot3(m, m).sqrt());
        let angle = libm::acos((dot3(ra, rb) / r2).clamp(-1.0, 1.0));
        let k = 4.0 / 3.0 * libm::tan(angle / 4.0);
        // The tangents, as long as the radius, in the direction of travel.
        let (ta, tb) = (cross3(m, ra), cross3(m, rb));
        [
            a,
            std::array::from_fn(|i| a[i] + k * ta[i]),
            std::array::from_fn(|i| b[i] - k * tb[i]),
            b,
        ]
    }

    /// The point of the unit sphere at longitude `lon` and latitude `lat` (radians).
    fn on_sphere(lon: f64, lat: f64) -> [f64; 3] {
        [
            libm::cos(lat) * libm::cos(lon),
            libm::cos(lat) * libm::sin(lon),
            libm::sin(lat),
        ]
    }

    /// The patch of the unit sphere `half_lon` either side of longitude `lon` and `half_lat`
    /// either side of latitude `lat` (radians), as a surface's face is drawn: its outline as four
    /// cubic arcs, along a parallel, a meridian, a parallel and a meridian.
    fn sphere_patch(lon: f64, lat: f64, half_lon: f64, half_lat: f64) -> Vec<[f64; 3]> {
        let (w, e, s, n) = (
            lon - half_lon,
            lon + half_lon,
            lat - half_lat,
            lat + half_lat,
        );
        let parallel = |la: f64| [0.0, 0.0, libm::sin(la)];
        [
            arc(parallel(s), on_sphere(w, s), on_sphere(e, s)),
            arc([0.0; 3], on_sphere(e, s), on_sphere(e, n)),
            arc(parallel(n), on_sphere(e, n), on_sphere(w, n)),
            arc([0.0; 3], on_sphere(w, n), on_sphere(w, s)),
        ]
        .concat()
    }

    /// The exact range over `0..=1` of the cubic Bernstein polynomial with coefficients `c`.
    fn cubic_range(c: [f64; 4]) -> (f64, f64) {
        let at = |t: f64| {
            let u = 1.0 - t;
            u * u * u * c[0] + 3.0 * u * u * t * c[1] + 3.0 * u * t * t * c[2] + t * t * t * c[3]
        };
        // Its derivative over 3: (d0 - 2 d1 + d2) t^2 + 2 (d1 - d0) t + d0.
        let (d0, d1, d2) = (c[1] - c[0], c[2] - c[1], c[3] - c[2]);
        let (qa, qb, qc) = (d0 - 2.0 * d1 + d2, 2.0 * (d1 - d0), d0);
        let mut ts = vec![0.0, 1.0];
        if qa != 0.0 {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc >= 0.0 {
                ts.extend([
                    (-qb + disc.sqrt()) / (2.0 * qa),
                    (-qb - disc.sqrt()) / (2.0 * qa),
                ]);
            }
        } else if qb != 0.0 {
            ts.push(-qc / qb);
        }
        ts.into_iter()
            .filter(|t| (0.0..=1.0).contains(t))
            .map(at)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            })
    }

    /// The range of `value` along an outline of cubics, each taken as the cubic through its
    /// control points' values: exact for a value affine in the point (a depth in the camera's
    /// axes), and the depth key along the outline as the core keys a curve (`path::curve_depth`).
    fn outline_range(outline: &[[f64; 3]], value: impl Fn([f64; 3]) -> f64) -> (f64, f64) {
        outline
            .chunks(4)
            .map(|c| cubic_range(std::array::from_fn(|k| value(c[k]))))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (a, b)| {
                (lo.min(a), hi.max(b))
            })
    }

    /// The range of depth keys along `outline` in `view`.
    fn outline_keys(view: &View, outline: &[[f64; 3]]) -> (f64, f64) {
        let p = projection_of(view);
        outline_range(outline, |q| -p.factor(p.project(q)[2]))
    }

    /// The range of depths in the camera's axes (larger is nearer) along `outline` in `view`.
    fn outline_depths(view: &View, outline: &[[f64; 3]]) -> (f64, f64) {
        let p = projection_of(view);
        outline_range(outline, |q| p.project(q)[2])
    }

    /// A square facing the camera at depth `z` in its axes, `half` across each way of its centre
    /// on the sight line through world point `through`.
    fn facing_square(view: &View, through: [f64; 3], z: f64, half: f64) -> Vec<[f64; 3]> {
        let p = projection_of(view);
        let Camera::ThreeD(c) = &view.camera else {
            unreachable!("a probe view is 3D")
        };
        let d = c.focal_distance;
        let q = p.unproject(p.project(through));
        let scale = (d - z) / (d - q[2]);
        let r = p.rotation();
        let world = |a: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|j| {
                view.frame_center[j] + r[0][j] * a[0] + r[1][j] * a[1] + r[2][j] * a[2]
            })
        };
        let (cx, cy) = (q[0] * scale, q[1] * scale);
        ring(&[
            world([cx - half, cy - half, z]),
            world([cx + half, cy - half, z]),
            world([cx + half, cy + half, z]),
            world([cx - half, cy + half, z]),
        ])
    }

    /// The angle between the unit sphere's normal at `n` and the sight line from `n` to the eye.
    fn sight_angle(view: &View, n: [f64; 3]) -> f64 {
        let p = projection_of(view);
        let Camera::ThreeD(c) = &view.camera else {
            unreachable!("a probe view is 3D")
        };
        let r = p.rotation();
        let eye: [f64; 3] =
            std::array::from_fn(|j| view.frame_center[j] + c.focal_distance * r[2][j]);
        let s: [f64; 3] = std::array::from_fn(|j| eye[j] - n[j]);
        libm::acos(dot3(n, s) / (dot3(n, n) * dot3(s, s)).sqrt())
    }

    /// The camera's direction (from the frame's centre toward the eye) as a longitude and latitude.
    fn toward_camera(view: &View) -> (f64, f64) {
        let v = projection_of(view).rotation()[2];
        (libm::atan2(v[1], v[0]), libm::asin(v[2]))
    }

    /// The longitude at latitude `lat` where the unit sphere's normal stands `angle` from the
    /// sight line, on the far side of the meridian facing the camera.
    fn longitude_at_sight_angle(view: &View, lat: f64, angle: f64) -> f64 {
        let (facing, _) = toward_camera(view);
        let (mut lo, mut hi) = (facing, facing + std::f64::consts::PI);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if sight_angle(view, on_sphere(mid, lat)) < angle {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// A path record in world space of object `100 + slot`, filled with `rgba`, stroked `stroke`
    /// wide when given, with the planarity `planes` decided for it when given.
    fn face_record(
        slot: u32,
        rgba: [f64; 4],
        stroke: Option<f64>,
        planes: Option<crate::depth::Planarity>,
    ) -> Record {
        let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
        r.paint.fill = Paint::One(rgba);
        r.depth.space = Space::World;
        if let Some(width) = stroke {
            r.paint.stroke = Paint::One([0.9, 0.9, 0.9, 1.0]);
            r.paint.stroke_width = width;
        }
        r.planes = planes.map(|p| vec![p.code()]);
        r
    }

    /// A stream of paths, each an outline and its record, drawn in their order.
    fn paths_stream(view: View, items: Vec<(Vec<[f64; 3]>, Record)>) -> Stream {
        let items: Vec<(Shape, Record)> = items
            .into_iter()
            .map(|(points, r)| {
                let shape = Shape::new(ShapeData::Path { points });
                let mut r = r;
                r.geometry.shape_a = Some(shape.key());
                (shape, r)
            })
            .collect();
        let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
        let mut s = Stream::new();
        s.apply(&crate::probes::first_delta(view, &items, &order))
            .expect("the delta applies");
        s
    }

    /// A key read beyond a range by more than its error bound: `(x, y, sub-sample, key, err)`.
    type Beyond = (u32, u32, usize, f64, f64);

    /// The keys of `object`'s fragments at the sub-samples they cover, its strokes on its own fill
    /// as the resolve orders them: those beyond `(lo, hi)` by more than their error bound, and how
    /// many keys were read.
    // The sub-sample `k` indexes each fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn keys_beyond(s: &Stream, object: u32, (lo, hi): (f64, f64)) -> (Vec<Beyond>, usize) {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let (mut beyond, mut read) = (Vec::new(), 0);
        s.visit_contributors(0, 0, w, h, |x, y, frags, cov, d, _| {
            let mut d = d.to_vec();
            composite::strokes_on_own_fills(frags, &mut d, |i| cov[i]);
            for (i, f) in frags.iter().enumerate() {
                if s.leaf_of(f).0 != object {
                    continue;
                }
                for k in 0..composite::SAMPLES {
                    if cov[i][k] == 0 {
                        continue;
                    }
                    read += 1;
                    let (key, err) = (d[i].samples[k], d[i].err);
                    if key < lo - err || key > hi + err {
                        beyond.push((x, y, k, key, err));
                    }
                }
            }
        })
        .expect("the stream draws");
        (beyond, read)
    }

    /// The worst of `beyond` (from [`keys_beyond`]) against `(lo, hi)`: how far past the range,
    /// and its error bound.
    fn worst_beyond(beyond: &[Beyond], (lo, hi): (f64, f64)) -> String {
        beyond
            .iter()
            .max_by(|a, b| {
                let past = |e: &Beyond| (lo - e.3).max(e.3 - hi);
                past(a).total_cmp(&past(b))
            })
            .map(|(x, y, k, key, err)| {
                format!(
                    "{} keys beyond [{lo}, {hi}], the worst at ({x}, {y}) sub-sample {k}: key \
                     {key}, {:e} past, err {err:e}",
                    beyond.len(),
                    (lo - key).max(key - hi)
                )
            })
            .unwrap_or_default()
    }

    /// Of the sub-samples where leaves of objects `front` and `back` both lie, in the order the
    /// resolve composites them: how many put a run of `back` above a run of `front`, and how many
    /// there are.
    fn above(s: &Stream, back: u32, front: u32) -> (usize, usize) {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let (mut wrong, mut both) = (0, 0);
        for c in s
            .contributors(0, 0, w, h)
            .expect("the stream draws")
            .values()
        {
            let leaf = |object: u32| c.leaves.iter().position(|l| l.object == object);
            let (Some(f), Some(b)) = (leaf(front), leaf(back)) else {
                continue;
            };
            for seq in &c.samples {
                let runs = |leaf: usize| {
                    seq.iter()
                        .enumerate()
                        .filter(move |(_, e)| e.0 == leaf)
                        .map(|(at, _)| at)
                };
                let (Some(lowest_front), Some(highest_back)) = (runs(f).min(), runs(b).max())
                else {
                    continue;
                };
                both += 1;
                if highest_back > lowest_front {
                    wrong += 1;
                }
            }
        }
        (wrong, both)
    }

    /// A patch of the unit sphere 18 by 9 degrees, filled and stroked, whose normal stands 88.5
    /// degrees from the sight line: a sliver whose plane nearest it (it lies on none) runs steeply
    /// across the screen, so read off its middle the plane leaves the depths its outline spans.
    /// Its key's bound holds that departure: every key of its fill and of its stroke on that fill
    /// lies within its bound of its outline's keys, and an opaque square drawn after it, nearer
    /// than every point of its outline by 0.1, is in front of it at every sub-sample (read as its
    /// plane alone, its fill came 0.18 nearer than its outline).
    #[test]
    fn a_grazing_curved_face_keys_within_its_outline_s_depths() {
        let view = crate::probes::view(854, 480, 80f64.to_radians(), 30f64.to_radians());
        let lat = 0.0;
        let lon = longitude_at_sight_angle(&view, lat, 88.5f64.to_radians());
        let patch = sphere_patch(lon, lat, 9f64.to_radians(), 4.5f64.to_radians());
        let (_, nearest) = outline_depths(&view, &patch);
        let square = facing_square(&view, on_sphere(lon, lat), nearest + 0.1, 0.5);
        let keys = outline_keys(&view, &patch);
        let s = paths_stream(
            view,
            vec![
                (
                    patch,
                    face_record(
                        1,
                        [0.2, 0.4, 0.8, 1.0],
                        Some(0.5),
                        Some(crate::depth::Planarity::Nearest),
                    ),
                ),
                (square, face_record(2, [0.8, 0.2, 0.2, 1.0], None, None)),
            ],
        );
        let (beyond, read) = keys_beyond(&s, 101, keys);
        let (wrong, both) = above(&s, 101, 102);
        assert!(read > 20, "{read} keys read");
        assert!(
            both > 20,
            "{both} sub-samples hold the patch and the square"
        );
        assert!(
            beyond.is_empty() && wrong == 0,
            "{}; of {both} sub-samples, the patch above the square at {wrong}",
            worst_beyond(&beyond, keys)
        );
    }

    /// A face whose corners lie on the plane nearest it and whose sides bow out of that plane
    /// between them, up and down in turn, seen at a slant: the far side bows toward the eye, so the
    /// plane read under it lies beyond every depth of the outline. A bound found at the corners
    /// alone is the plane's rounding; the key's bound holds the sides' departure between them, and
    /// every key of the fill lies within it of the outline's keys.
    #[test]
    fn a_face_bowed_between_its_corners_keys_within_its_outline_s_depths() {
        let view = crate::probes::view(320, 180, 60f64.to_radians(), -80f64.to_radians());
        let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let mut outline = Vec::new();
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            let z = if i % 2 == 0 { 0.5 } else { -0.5 };
            let at = |t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            let (h1, h2) = (at(0.25), at(0.75));
            outline.extend([
                [a[0], a[1], 0.0],
                [h1[0], h1[1], z],
                [h2[0], h2[1], z],
                [b[0], b[1], 0.0],
            ]);
        }
        let plane = crate::depth::WorldPlane::fit(&outline, crate::depth::Planarity::Nearest)
            .expect("the nearest plane");
        let off = |p: &[f64; 3]| (dot3(plane.normal, *p) - plane.offset).abs();
        for (i, p) in outline.iter().enumerate() {
            if i % 4 == 0 || i % 4 == 3 {
                assert!(
                    off(p) <= 1e-12,
                    "corner {p:?} lies {:e} off the plane",
                    off(p)
                );
            } else {
                assert!(
                    off(p) >= 0.4,
                    "handle {p:?} lies {:e} off the plane",
                    off(p)
                );
            }
        }
        let keys = outline_keys(&view, &outline);
        let s = paths_stream(
            view,
            vec![(
                outline,
                face_record(
                    1,
                    [0.2, 0.4, 0.8, 1.0],
                    None,
                    Some(crate::depth::Planarity::Nearest),
                ),
            )],
        );
        let (beyond, read) = keys_beyond(&s, 101, keys);
        assert!(read > 1000, "{read} keys read");
        assert!(beyond.is_empty(), "{}", worst_beyond(&beyond, keys));
    }

    /// A face on one plane exactly, at a slant: its fill is keyed on its plane with the bound the
    /// camera's key across that plane carries, bit for bit, wherever it is read.
    #[test]
    fn an_exactly_planar_face_keeps_its_plane_s_key_and_bound() {
        let view = crate::probes::view(320, 180, 65f64.to_radians(), -40f64.to_radians());
        let (w, h) = (view.pixel_width, view.pixel_height);
        let outline = ring(&[[-1.0, -0.5, 0.2], [1.0, -0.8, -0.1], [0.2, 1.0, 0.3]]);
        let p = projection_of(&view);
        let map = crate::project::FrameMap::of_kind(&view, RecordKind::Path, w, h);
        let plane = crate::depth::WorldPlane::fit(&outline, crate::depth::Planarity::Newell)
            .expect("a plane");
        // Where the lowering reads it: the outline's box on screen, a pixel's reach and a pixel
        // beyond (a fill reaches no further than its antialiasing).
        let shown =
            crate::project::pre_display(&outline, Some(&p), crate::project::Placement::Projected);
        let mut region = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for q in &shown {
            let [x, y] = map.apply(*q);
            region = [
                region[0].min(x - 2.0),
                region[1].min(y - 2.0),
                region[2].max(x + 2.0),
                region[3].max(y + 2.0),
            ];
        }
        let expected = p
            .key_plane(&plane, &map, outline[0], 0.0, region)
            .expect("the plane keys");
        let s = paths_stream(
            view,
            vec![(
                outline,
                face_record(
                    1,
                    [0.2, 0.4, 0.8, 1.0],
                    None,
                    Some(crate::depth::Planarity::Newell),
                ),
            )],
        );
        let mut read = 0;
        s.visit_contributors(0, 0, w, h, |_, _, frags, _, _, _| {
            for f in frags {
                assert_eq!(f.depth, expected);
                read += 1;
            }
        })
        .expect("the stream draws");
        assert!(read > 100, "{read} fragments");
    }

    /// A patch of the unit sphere 18 by 9 degrees, filled and stroked, seen face on, between an
    /// opaque square drawn before it and 0.3 nearer than every point of its outline and one drawn
    /// after it and 0.3 farther than every point: its departure from the plane nearest it is a
    /// fraction of that, so depth orders the three at every sub-sample, against their draw order.
    #[test]
    fn a_curved_face_seen_face_on_keeps_a_strict_order_against_faces_clearly_before_and_behind() {
        let view = crate::probes::view(854, 480, 70f64.to_radians(), -110f64.to_radians());
        let (lon, lat) = toward_camera(&view);
        let patch = sphere_patch(lon, lat, 9f64.to_radians(), 4.5f64.to_radians());
        let (farthest, nearest) = outline_depths(&view, &patch);
        let middle = on_sphere(lon, lat);
        let front = facing_square(&view, middle, nearest + 0.3, 0.5);
        let back = facing_square(&view, middle, farthest - 0.3, 0.5);
        let s = paths_stream(
            view,
            vec![
                (front, face_record(1, [0.8, 0.2, 0.2, 1.0], None, None)),
                (
                    patch,
                    face_record(
                        2,
                        [0.2, 0.4, 0.8, 1.0],
                        Some(0.5),
                        Some(crate::depth::Planarity::Nearest),
                    ),
                ),
                (back, face_record(3, [0.2, 0.8, 0.2, 1.0], None, None)),
            ],
        );
        let (wrong, both) = above(&s, 102, 101);
        assert!(
            both > 1000,
            "{both} sub-samples hold the patch and the square before it"
        );
        assert_eq!(
            wrong, 0,
            "of {both} sub-samples, the patch above the square before it at {wrong}"
        );
        let (wrong, both) = above(&s, 103, 102);
        assert!(
            both > 1000,
            "{both} sub-samples hold the patch and the square behind it"
        );
        assert_eq!(
            wrong, 0,
            "of {both} sub-samples, the square behind the patch above it at {wrong}"
        );
    }

    /// Three wires along one line, drawn in turn: the first two the same, the third moved by
    /// 2^-51 in z, which moves its key by less than the keys' error bound. Over five frames whose
    /// camera angles step on the grid the engine keeps them to (2^-24), paint order decides them
    /// at every sub-sample they cover: depth tells them apart nowhere.
    #[test]
    // The sub-sample `k` indexes each fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn coincident_wires_keep_paint_order_at_every_sub_sample_of_every_frame() {
        let (w, h) = (320, 180);
        let grid = (1u64 << 24) as f64;
        let wire = |slot: u32, dz: f64, rgba: [f64; 4]| {
            let shape = crate::probes::polyline(&[[-2.0, -1.0, 0.3 + dz], [2.0, 1.0, -0.2 + dz]]);
            let r = crate::probes::stroke_record(slot, &shape, rgba, 10.0);
            (shape, r)
        };
        let items = [
            wire(1, 0.0, [1.0, 0.0, 0.0, 1.0]),
            wire(2, 0.0, [0.0, 1.0, 0.0, 1.0]),
            wire(3, 2.0 * f64::EPSILON, [0.0, 0.0, 1.0, 1.0]),
        ];
        for frame in 0..5 {
            let phi = (1.1 * grid).round() / grid + f64::from(frame) * 37.0 / grid;
            let theta = (-0.7 * grid).round() / grid + f64::from(frame) * 53.0 / grid;
            let view = crate::probes::view(w, h, phi, theta);
            let mut s = Stream::new();
            s.apply(&crate::probes::first_delta(view, &items, &[1, 2, 3]))
                .expect("the delta applies");
            // The third wire's key stands from the first's by less than either's error bound.
            let (mut apart, mut least_err, mut shared) = (0.0f64, f64::INFINITY, 0);
            s.visit_contributors(0, 0, w, h, |_, _, frags, cov, d, _| {
                if frags.len() != 3 {
                    return;
                }
                for k in 0..composite::SAMPLES {
                    if cov.iter().all(|c| c[k] != 0) {
                        shared += 1;
                        apart = apart.max((d[2].samples[k] - d[0].samples[k]).abs());
                        least_err = least_err.min(d[0].err.min(d[2].err));
                    }
                }
            })
            .expect("the stream draws");
            assert!(
                shared > 1000,
                "frame {frame}: {shared} sub-samples all three cover"
            );
            assert!(
                apart < least_err,
                "frame {frame}: keys apart by {apart:e}, error bound {least_err:e}"
            );
            let found = s.contributors(0, 0, w, h).expect("the stream draws");
            let mut seen = 0;
            for ((x, y), pixel) in &found {
                for (k, sample) in pixel.samples.iter().enumerate() {
                    let order: Vec<u32> = sample
                        .iter()
                        .map(|(l, _)| pixel.leaves[*l].object)
                        .collect();
                    if order.len() < 2 {
                        continue;
                    }
                    seen += 1;
                    assert!(
                        order.windows(2).all(|p| p[0] < p[1]),
                        "frame {frame}, pixel ({x}, {y}) sub-sample {k}: {order:?}, bottom first"
                    );
                }
            }
            assert!(
                seen > 1000,
                "frame {frame}: {seen} sub-samples two wires cover"
            );
        }
    }
}
